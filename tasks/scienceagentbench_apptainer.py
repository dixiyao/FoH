"""Apptainer backend for the pinned ScienceAgentBench verifier.

Slurm hosts such as the UChicago DS cluster give jobs neither a Docker socket
nor rootless Podman, but they do allow ``apptainer build --fakeroot``. This
backend runs the same verifier without Docker and without editing upstream:

* the base image is built from upstream's own ``_DOCKERFILE_BASE`` text,
  translated instruction by instruction into an Apptainer definition, and
  cached as one SIF keyed by the hash of that text;
* every evaluation gets a fresh writable overlay on top of the base SIF, in
  which upstream's per-instance Dockerfile steps run (``pipreqs`` plus the
  program's own dependencies), so no two evaluations share installed packages
  and nothing survives into a later round;
* the container then runs the pinned ``compute_scores.py`` from ``/testbed``
  with the same five mounts the Docker harness creates, and the result is read
  from the same ``logs/run_evaluation/<run_id>/<instance_id>/output/result.json``.

The Gemini visual judge is mounted exactly as the Docker driver mounts it.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

OPENAI_ENV_NAMES = (
    "OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME",
)
BUILD_TIMEOUT_SECONDS = 5400
MOUNT_POINTS = (
    "/testbed/benchmark", "/testbed/pred_programs", "/testbed/instance_path",
)
MOUNT_FILES = (
    "/testbed/compute_scores.py", "/testbed/gpt4_visual_judge.py",
    "/testbed/foh_original_compute_scores.py", "/testbed/foh_original_visual_judge.py",
)


# ---------------------------------------------------------------------------
# Dockerfile translation
# ---------------------------------------------------------------------------

def parse_dockerfile(text: str) -> List[Tuple[str, str]]:
    """Split a Dockerfile into ``(INSTRUCTION, arguments)`` pairs.

    Follows Docker's rules for the subset upstream uses: comment lines are
    dropped (also inside a continued instruction), blank lines inside a
    continuation are skipped, and a trailing backslash joins the next line
    without inserting a newline.
    """
    steps: List[Tuple[str, str]] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if not stripped:
            continue
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1]
            continue
        pending += line
        logical = pending.strip()
        pending = ""
        instruction, _, arguments = logical.partition(" ")
        steps.append((instruction.upper(), arguments.strip()))
    if pending.strip():
        instruction, _, arguments = pending.strip().partition(" ")
        steps.append((instruction.upper(), arguments.strip()))
    return steps


def _from_image(arguments: str) -> str:
    tokens = [token for token in arguments.split() if not token.startswith("--")]
    if not tokens:
        raise ValueError(f"Cannot parse FROM {arguments!r}")
    return tokens[0]


def _assignment(arguments: str) -> Tuple[str, str]:
    if "=" in arguments.split()[0]:
        name, _, value = arguments.partition("=")
    else:
        name, _, value = arguments.partition(" ")
    return name.strip(), value.strip()


def base_definition(dockerfile: str) -> str:
    """Translate upstream's base Dockerfile into an Apptainer definition."""
    steps = parse_dockerfile(dockerfile)
    image = None
    environment: List[str] = []
    post: List[str] = []
    for instruction, arguments in steps:
        if instruction == "FROM":
            if image is not None:
                raise ValueError("Multi-stage base Dockerfiles are not supported")
            image = _from_image(arguments)
        elif instruction == "ARG":
            name, value = _assignment(arguments)
            post.append(f"export {name}={value}")
        elif instruction == "ENV":
            name, value = _assignment(arguments)
            line = f"export {name}={value}"
            environment.append(line)
            post.append(line)
        elif instruction == "RUN":
            post.append(f"/bin/sh -c {shlex.quote(arguments)} || exit 1")
        elif instruction == "WORKDIR":
            post.append(f"mkdir -p {shlex.quote(arguments)} && cd {shlex.quote(arguments)}")
        else:
            raise ValueError(f"Unsupported base Dockerfile instruction {instruction}")
    if image is None:
        raise ValueError("Base Dockerfile has no FROM")
    # FoH-owned: create the bind targets the harness mounts into /testbed.
    post.append("mkdir -p " + " ".join(MOUNT_POINTS))
    post.append("touch " + " ".join(MOUNT_FILES))
    return "\n".join(
        [
            "Bootstrap: docker",
            f"From: {image}",
            "",
            "%environment",
            *(f"    {line}" for line in environment),
            "",
            "%post",
            *(f"    {line}" for line in post),
            "",
        ]
    )


def instance_script(dockerfile: str) -> str:
    """Translate upstream's instance Dockerfile into a shell script.

    ``COPY`` sources are read from ``/foh_build`` (the upstream build context).
    Every ``RUN`` keeps Docker's ``/bin/sh -c`` semantics and aborts the build
    on failure, as ``docker build`` does.
    """
    lines = ["set -e"]
    saw_from = False
    for instruction, arguments in parse_dockerfile(dockerfile):
        if instruction == "FROM":
            saw_from = True
        elif instruction == "COPY":
            source, destination = shlex.split(arguments)
            source = "/foh_build/" + source.lstrip("./")
            lines.append(f"mkdir -p {shlex.quote(os.path.dirname(destination) or '/')}")
            lines.append(f"cp {shlex.quote(source)} {shlex.quote(destination)}")
        elif instruction == "WORKDIR":
            lines.append(f"mkdir -p {shlex.quote(arguments)} && cd {shlex.quote(arguments)}")
        elif instruction == "RUN":
            lines.append(f"/bin/sh -c {shlex.quote(arguments)}")
        elif instruction in {"ENV", "ARG"}:
            # Runtime environment is passed explicitly at exec time; the only
            # ENV lines upstream uses here carry the (blanked) OpenAI settings.
            continue
        else:
            raise ValueError(f"Unsupported instance Dockerfile instruction {instruction}")
    if not saw_from:
        raise ValueError("Instance Dockerfile has no FROM")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------

def _arch() -> str:
    return "arm64" if platform.machine() in {"aarch64", "arm64"} else "x86_64"


def _upstream(bench_dir: Path):
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    from evaluation.harness import dockerfiles, test_spec  # type: ignore

    return dockerfiles, test_spec


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        # POSIX record locks also work across nodes on the NFS home/project mounts.
        fcntl.lockf(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(handle, fcntl.LOCK_UN)


def scratch_root(explicit: Optional[str] = None) -> Path:
    for candidate in (explicit, os.environ.get("FOH_APPTAINER_SCRATCH"),
                      os.environ.get("TMPDIR"), "/tmp"):
        if candidate:
            path = Path(candidate).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return path.resolve()
    raise RuntimeError("No scratch directory available")


def _apptainer_environment(scratch: Path) -> Dict[str, str]:
    environment = dict(os.environ)
    tmp = scratch / "apptainer-tmp"
    cache = Path(os.environ.get("APPTAINER_CACHEDIR") or scratch / "apptainer-cache")
    tmp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("APPTAINER_TMPDIR", str(tmp))
    environment.setdefault("APPTAINER_CACHEDIR", str(cache))
    return environment


def ensure_base_image(bench_dir: Path, cache_dir: Path, *, scratch: Path,
                      log: Optional[Path] = None) -> Path:
    """Return the base SIF, building it once (under a cross-node lock)."""
    dockerfiles, _ = _upstream(bench_dir)
    arch = _arch()
    platform_name = "linux/x86_64" if arch == "x86_64" else "linux/arm64/v8"
    definition = base_definition(dockerfiles.get_dockerfile_base(platform_name, arch))
    digest = hashlib.sha256(definition.encode("utf-8")).hexdigest()[:16]
    target = cache_dir / f"sab.base.{arch}.{digest}.sif"
    if target.is_file():
        return target
    with _file_lock(cache_dir / f"sab.base.{arch}.lock"):
        if target.is_file():
            return target
        cache_dir.mkdir(parents=True, exist_ok=True)
        definition_path = cache_dir / f"sab.base.{arch}.{digest}.def"
        definition_path.write_text(definition, encoding="utf-8")
        partial = scratch / f"sab.base.{arch}.{digest}.{os.getpid()}.sif"
        command = ["apptainer", "build", "--fakeroot", "--force", str(partial), str(definition_path)]
        log_path = log or cache_dir / f"sab.base.{arch}.{digest}.build.log"
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, text=True,
                env=_apptainer_environment(scratch), timeout=4 * 3600,
            )
        if completed.returncode != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Apptainer base image build failed (exit {completed.returncode}); see {log_path}"
            )
        shutil.move(str(partial), str(target))
        print(f"[sab-apptainer] built {target.name} in {time.time() - started:.0f}s", flush=True)
    return target


# ---------------------------------------------------------------------------
# One evaluation
# ---------------------------------------------------------------------------

def _run(command: List[str], *, env: Dict[str, str], timeout: float,
         cwd: Optional[Path] = None) -> Tuple[int, str, bool]:
    """Run in a new session so a timeout kills every descendant."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env, start_new_session=True, cwd=str(cwd) if cwd else None,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output or "", False
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return -9, output or "", True


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    # Fakeroot-created trees can contain read-only directories.
    subprocess.run(["chmod", "-R", "u+rwX", str(path)], capture_output=True, check=False)
    shutil.rmtree(path, ignore_errors=True)


def run_eval(
    instance: dict,
    *,
    bench_dir: Path,
    benchmark_root: Path,
    pred_dir: Path,
    log_dir: Path,
    run_id: str,
    timeout: int,
    base_image: Path,
    scratch: Path,
    judge_model: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    hf_cache: Optional[Path] = None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Grade one instance. Returns (result.json payload or None, diagnostics)."""
    _, test_spec_module = _upstream(bench_dir)
    instance_id = str(instance["instance_id"])
    spec = test_spec_module.make_test_spec(
        {**instance, "instance_id": instance_id}, str(benchmark_root), str(pred_dir)
    )
    instance_path = bench_dir / "logs" / "run_evaluation" / run_id / instance_id
    _remove_tree(instance_path)
    for name in ("input", "output", "pred_results"):
        (instance_path / name).mkdir(parents=True, exist_ok=True)
    (instance_path / "input" / "input.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    program_name = "pred_" + spec.gold_program_name
    build_dir = (
        bench_dir / "logs" / "build_images" / "instances"
        / f"{spec.instance_image_key.replace(':', '__')}.{run_id}"
    )
    _remove_tree(build_dir)
    build_dir.mkdir(parents=True)
    shutil.copyfile(pred_dir / program_name, build_dir / program_name)
    shutil.copyfile(bench_dir / "config_conda_env.py", build_dir / "config_conda_env.py")
    script = instance_script(spec.instance_dockerfile)
    (build_dir / "foh_instance_build.sh").write_text(script, encoding="utf-8")

    work = Path(tempfile.mkdtemp(prefix=f"foh-sab-{instance_id}-", dir=str(scratch)))
    overlay = work / "overlay"
    overlay.mkdir()
    environment = _apptainer_environment(scratch)
    # Nothing from the host environment enters the container except these.
    for name in list(environment):
        if name.startswith(("APPTAINERENV_", "SINGULARITYENV_")):
            environment.pop(name)
    environment["APPTAINERENV_TZ"] = "Etc/UTC"
    common = ["apptainer", "exec", "--fakeroot", "--containall", "--cleanenv",
              "--overlay", str(overlay)]
    diagnostics: Dict[str, Any] = {"backend": "apptainer", "base_image": base_image.name}
    transcript: List[str] = []
    try:
        build_command = [*common, "--bind", f"{build_dir}:/foh_build:ro", "--pwd", "/",
                         str(base_image), "/bin/sh", "/foh_build/foh_instance_build.sh"]
        started = time.time()
        code, output, timed_out = _run(build_command, env=environment, timeout=BUILD_TIMEOUT_SECONDS)
        diagnostics["build_seconds"] = round(time.time() - started, 1)
        transcript += ["$ " + " ".join(build_command), output[-20000:]]
        if code != 0:
            diagnostics["error"] = (
                "Instance build timed out" if timed_out else f"Instance build failed (exit {code})"
            )
            return None, diagnostics

        binds = [
            (benchmark_root, "/testbed/benchmark", "rw"),
            (pred_dir, "/testbed/pred_programs", "rw"),
            (instance_path, "/testbed/instance_path", "rw"),
        ]
        tasks_dir = Path(__file__).resolve().parent
        if judge_model:
            launcher = work / "compute_scores.py"
            from tasks.scienceagentbench_eval_driver import COMPUTE_LAUNCHER  # noqa: WPS433

            launcher.write_text(COMPUTE_LAUNCHER, encoding="utf-8")
            binds += [
                (tasks_dir / "scienceagentbench_visual_judge.py", "/testbed/gpt4_visual_judge.py", "ro"),
                (launcher, "/testbed/compute_scores.py", "ro"),
                (bench_dir / "gpt4_visual_judge.py", "/testbed/foh_original_visual_judge.py", "ro"),
                (bench_dir / "compute_scores.py", "/testbed/foh_original_compute_scores.py", "ro"),
            ]
            for name in OPENAI_ENV_NAMES:
                environment[f"APPTAINERENV_{name}"] = ""
            environment["APPTAINERENV_FOH_JUDGE_MODEL"] = judge_model
            environment["APPTAINERENV_FOH_JUDGE_API_KEY"] = judge_api_key or ""
        else:
            binds += [
                (bench_dir / "compute_scores.py", "/testbed/compute_scores.py", "rw"),
                (bench_dir / "gpt4_visual_judge.py", "/testbed/gpt4_visual_judge.py", "rw"),
            ]
            for name in OPENAI_ENV_NAMES:
                environment[f"APPTAINERENV_{name}"] = os.environ.get(name, "")
        if hf_cache is not None:
            hf_cache.mkdir(parents=True, exist_ok=True)
            binds.append((hf_cache, "/foh_hf_cache", "rw"))
            environment["APPTAINERENV_HF_HOME"] = "/foh_hf_cache"
        bind_args: List[str] = []
        for source, target, mode in binds:
            bind_args += ["--bind", f"{Path(source).resolve()}:{target}:{mode}"]
        run_command = [*common, *bind_args, "--pwd", "/testbed", str(base_image),
                       "python", "compute_scores.py"]
        started = time.time()
        code, output, timed_out = _run(run_command, env=environment, timeout=timeout)
        diagnostics["run_seconds"] = round(time.time() - started, 1)
        transcript += ["$ " + " ".join(run_command), output[-20000:]]
        if timed_out:
            diagnostics["error"] = f"compute_scores.py exceeded {timeout}s"
        result_path = instance_path / "output" / "result.json"
        if not result_path.is_file():
            diagnostics.setdefault(
                "error", f"compute_scores.py produced no result.json (exit {code})"
            )
            diagnostics["harness_output"] = output.strip()[-2000:]
            return None, diagnostics
        return json.loads(result_path.read_text(encoding="utf-8")), diagnostics
    finally:
        log_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(transcript)
        if judge_api_key:
            text = text.replace(judge_api_key, "[REDACTED]")
        (log_dir / f"{run_id}.log").write_text(text, encoding="utf-8")
        diagnostics["harness_log"] = str(log_dir / f"{run_id}.log")
        _remove_tree(work)
        _remove_tree(build_dir)
