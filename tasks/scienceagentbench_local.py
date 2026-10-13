"""Container-free backend for the pinned ScienceAgentBench verifier.

Runs the same verifier steps as upstream's Docker harness directly on the
host, for Slurm nodes without a Docker daemon:

* the base environment is a Miniconda install reproducing upstream's
  ``_DOCKERFILE_BASE`` (same Miniconda release, ``python=3.10`` and the exact
  ``pip install`` line, both read from the pinned Dockerfile text), built once
  under a cross-node lock and keyed by the hash of that text;
* every evaluation gets a fresh ``venv --system-site-packages`` on top of the
  base, and upstream's per-instance Dockerfile steps (``pipreqs`` and the
  program's own requirements, including the per-package special cases) run
  into that venv, so evaluations never share installed packages;
* a per-evaluation ``testbed`` directory stands in for ``/testbed``: it holds
  the submitted program and symlinks to the benchmark, predictions and
  instance directories, and the pinned ``compute_scores.py`` (plus the Gemini
  judge adapter when used) run from it with ``/testbed`` rewritten to that
  directory. The pinned files themselves are never edited.

System libraries the Dockerfile installs with apt (compilers, libffi, libtiff,
libxrender) must already exist on the host, as they do on the DS cluster.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tasks import scienceagentbench_apptainer as common

UPSTREAM_ROOT = "/opt/miniconda3"
UPSTREAM_TESTBED = "/testbed"
OPENAI_ENV_NAMES = common.OPENAI_ENV_NAMES
BUILD_TIMEOUT_SECONDS = common.BUILD_TIMEOUT_SECONDS
_SAFE_HOST_ENV = ("LANG", "LC_ALL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
                  "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def base_recipe(dockerfile: str) -> Dict[str, Any]:
    """Pull the Miniconda installer and the conda/pip steps out of upstream's base."""
    steps = [arguments for instruction, arguments in common.parse_dockerfile(dockerfile)
             if instruction == "RUN"]
    installer = next((match.group(0) for step in steps
                      for match in [re.search(r"https://repo\.anaconda\.com/miniconda/\S+?\.sh", step)]
                      if match), None)
    conda_install = next((step for step in steps if "conda install" in step), None)
    pip_install = next((step for step in steps if f"{UPSTREAM_ROOT}/bin/pip install" in step), None)
    channels = next((step for step in steps if "conda config" in step), "")
    if not installer or not conda_install or not pip_install:
        raise ValueError("Pinned base Dockerfile no longer matches the expected Miniconda recipe")
    conda_args = conda_install.split("conda install", 1)[1].strip()
    return {
        "installer": installer,
        "conda_install_args": conda_args,
        "append_conda_forge": "--append channels conda-forge" in channels,
        "pip_install": pip_install,
    }


def _run_logged(command: List[str] | str, *, log, env: Dict[str, str], cwd: Optional[Path] = None,
                timeout: float = BUILD_TIMEOUT_SECONDS, shell: bool = False) -> None:
    log.write(f"$ {command}\n")
    log.flush()
    code, output, timed_out = common._run(
        ["/bin/sh", "-c", command] if shell else list(command), env=env, timeout=timeout, cwd=cwd,
    )
    log.write(output + "\n")
    log.flush()
    if code != 0:
        raise RuntimeError(f"{'Timed out' if timed_out else f'Exit {code}'}: {str(command)[:200]}")


def _host_env(extra: Dict[str, str]) -> Dict[str, str]:
    environment = {name: os.environ[name] for name in _SAFE_HOST_ENV if name in os.environ}
    environment.setdefault("LANG", "C.UTF-8")
    environment["TZ"] = "Etc/UTC"
    environment.update(extra)
    return environment


def ensure_base_env(bench_dir: Path, cache_dir: Path, *, scratch: Path) -> Path:
    """Return the base Miniconda prefix, installing it once under a lock."""
    dockerfiles, _ = common._upstream(bench_dir)
    arch = common._arch()
    platform_name = "linux/x86_64" if arch == "x86_64" else "linux/arm64/v8"
    dockerfile = dockerfiles.get_dockerfile_base(platform_name, arch)
    recipe = base_recipe(dockerfile)
    digest = hashlib.sha256(dockerfile.encode("utf-8")).hexdigest()[:16]
    prefix = cache_dir / f"sab.base.{arch}.{digest}"
    marker = prefix / ".foh_base_ready"
    if marker.is_file():
        return prefix
    with common._file_lock(cache_dir / f"sab.base.{arch}.lock"):
        if marker.is_file():
            return prefix
        cache_dir.mkdir(parents=True, exist_ok=True)
        common._remove_tree(prefix)
        log_path = cache_dir / f"sab.base.{arch}.{digest}.build.log"
        work = Path(tempfile.mkdtemp(prefix="foh-sab-base-", dir=str(scratch)))
        started = time.time()
        try:
            home = work / "home"
            home.mkdir()
            env = _host_env({
                "HOME": str(home), "TMPDIR": str(work),
                "PATH": f"{prefix}/bin:/usr/local/bin:/usr/bin:/bin",
                "CONDARC": str(prefix / ".condarc"), "PIP_NO_INPUT": "1",
                "PYTHONNOUSERSITE": "1",
            })
            with open(log_path, "w", encoding="utf-8") as log:
                installer = work / "miniconda.sh"
                _run_logged(["curl", "-fsSL", "-o", str(installer), recipe["installer"]], log=log, env=env)
                _run_logged(["bash", str(installer), "-b", "-p", str(prefix)], log=log, env=env)
                condarc = "channels:\n  - defaults\n"
                if recipe["append_conda_forge"]:
                    condarc += "  - conda-forge\n"
                (prefix / ".condarc").write_text(condarc, encoding="utf-8")
                _run_logged(f"{prefix}/bin/conda install {recipe['conda_install_args']}",
                            log=log, env=env, shell=True)
                _run_logged(recipe["pip_install"].replace(UPSTREAM_ROOT, str(prefix)),
                            log=log, env=env, shell=True)
            marker.write_text(json.dumps({"dockerfile_sha256": digest, "built_seconds":
                                          round(time.time() - started)}) + "\n", encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"ScienceAgentBench base environment build failed: {exc}; "
                               f"see {log_path}") from exc
        finally:
            common._remove_tree(work)
        print(f"[sab-local] built base env {prefix.name} in {time.time() - started:.0f}s", flush=True)
    return prefix


def _place(source: Path, target: Path, testbed: Path) -> None:
    """Copy a harness file into the testbed with ``/testbed`` rewritten."""
    text = source.read_text(encoding="utf-8").replace(UPSTREAM_TESTBED, str(testbed))
    target.write_text(text, encoding="utf-8")


def run_eval(
    instance: dict,
    *,
    bench_dir: Path,
    benchmark_root: Path,
    pred_dir: Path,
    log_dir: Path,
    run_id: str,
    timeout: int,
    base_prefix: Path,
    scratch: Path,
    judge_model: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    hf_cache: Optional[Path] = None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Grade one instance. Returns (result.json payload or None, diagnostics)."""
    _, test_spec_module = common._upstream(bench_dir)
    instance_id = str(instance["instance_id"])
    spec = test_spec_module.make_test_spec(
        {**instance, "instance_id": instance_id}, str(benchmark_root), str(pred_dir)
    )
    instance_path = bench_dir / "logs" / "run_evaluation" / run_id / instance_id
    common._remove_tree(instance_path)
    for name in ("input", "output", "pred_results"):
        (instance_path / name).mkdir(parents=True, exist_ok=True)
    (instance_path / "input" / "input.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    program_name = "pred_" + spec.gold_program_name
    work = Path(tempfile.mkdtemp(prefix=f"foh-sab-{instance_id}-", dir=str(scratch)))
    build_dir = work / "build"
    testbed = work / "testbed"
    venv = work / "venv"
    home = work / "home"
    for path in (build_dir, testbed, home, work / "tmp"):
        path.mkdir(parents=True)
    shutil.copyfile(pred_dir / program_name, build_dir / program_name)
    shutil.copyfile(bench_dir / "config_conda_env.py", build_dir / "config_conda_env.py")

    diagnostics: Dict[str, Any] = {"backend": "local", "base_env": base_prefix.name}
    transcript: List[str] = []
    environment = _host_env({
        "HOME": str(home), "TMPDIR": str(work / "tmp"), "PYTHONNOUSERSITE": "1",
        "PIP_NO_INPUT": "1", "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv}/bin:{base_prefix}/bin:/usr/local/bin:/usr/bin:/bin",
    })
    try:
        started = time.time()
        # Upstream installs straight into the image's base env; a venv layered
        # on it gives each evaluation the same view without sharing installs.
        script = common.instance_script(spec.instance_dockerfile)
        script = (
            script.replace(f"{UPSTREAM_ROOT}/bin/pipreqs", f"{base_prefix}/bin/pipreqs")
            .replace(f"{UPSTREAM_ROOT}/bin/", f"{venv}/bin/")
            .replace("/foh_build/", f"{build_dir}/")
            .replace(UPSTREAM_TESTBED, str(testbed))
        )
        script = (f"{base_prefix}/bin/python -m venv --system-site-packages {venv}\n") + script
        # instance_script starts with `set -e`; keep it first.
        script = "set -e\n" + script.replace("set -e\n", "", 1)
        (build_dir / "foh_instance_build.sh").write_text(script, encoding="utf-8")
        code, output, timed_out = common._run(
            ["/bin/sh", str(build_dir / "foh_instance_build.sh")], env=environment,
            timeout=BUILD_TIMEOUT_SECONDS, cwd=work,
        )
        diagnostics["build_seconds"] = round(time.time() - started, 1)
        transcript += ["$ sh foh_instance_build.sh", script, output[-20000:]]
        if code != 0:
            diagnostics["error"] = (
                "Instance setup timed out" if timed_out else f"Instance setup failed (exit {code})"
            )
            return None, diagnostics

        for source, name in ((benchmark_root, "benchmark"), (pred_dir, "pred_programs"),
                             (instance_path, "instance_path")):
            (testbed / name).symlink_to(Path(source).resolve(), target_is_directory=True)
        tasks_dir = Path(__file__).resolve().parent
        if judge_model:
            from tasks.scienceagentbench_eval_driver import COMPUTE_LAUNCHER  # noqa: WPS433

            (testbed / "compute_scores.py").write_text(
                COMPUTE_LAUNCHER.replace(UPSTREAM_TESTBED, str(testbed)), encoding="utf-8")
            _place(tasks_dir / "scienceagentbench_visual_judge.py", testbed / "gpt4_visual_judge.py", testbed)
            _place(bench_dir / "gpt4_visual_judge.py", testbed / "foh_original_visual_judge.py", testbed)
            _place(bench_dir / "compute_scores.py", testbed / "foh_original_compute_scores.py", testbed)
            environment.update({name: "" for name in OPENAI_ENV_NAMES})
            environment.update(FOH_JUDGE_MODEL=judge_model, FOH_JUDGE_API_KEY=judge_api_key or "")
        else:
            _place(bench_dir / "compute_scores.py", testbed / "compute_scores.py", testbed)
            _place(bench_dir / "gpt4_visual_judge.py", testbed / "gpt4_visual_judge.py", testbed)
            environment.update({name: os.environ.get(name, "") for name in OPENAI_ENV_NAMES})
        if hf_cache is not None:
            hf_cache.mkdir(parents=True, exist_ok=True)
            environment["HF_HOME"] = str(hf_cache)
        started = time.time()
        command = [str(venv / "bin" / "python"), "compute_scores.py"]
        code, output, timed_out = common._run(command, env=environment, timeout=timeout, cwd=testbed)
        diagnostics["run_seconds"] = round(time.time() - started, 1)
        transcript += ["$ " + " ".join(command), output[-20000:]]
        if timed_out:
            diagnostics["error"] = f"compute_scores.py exceeded {timeout}s"
        result_path = instance_path / "output" / "result.json"
        if not result_path.is_file():
            diagnostics.setdefault("error", f"compute_scores.py produced no result.json (exit {code})")
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
        with contextlib.suppress(Exception):
            common._remove_tree(work)
