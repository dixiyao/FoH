"""ScienceAgentBench runner for Federation over Harness.

The 102-task verified release is pinned locally. The upstream checkout owns the
official containerized verifier, the Hugging Face annotation sheet owns the task
inputs, and the password-protected artifact archive owns datasets, gold programs
and evaluation scripts. The existing FoH lifecycle owns solving, reflection,
aggregation, and evolution.

Every task asks for one self-contained Python program. FoH hands the agent a
workspace holding that task's dataset folder, and grading copies the program the
agent saved under ``pred_programs/`` into the upstream Docker harness, which
installs the program's own dependencies, executes it, and runs the official
per-task evaluation script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from ._bootstrap import BENCHMARKS_DIR, REPO_ROOT
except ImportError:  # pragma: no cover
    from _bootstrap import BENCHMARKS_DIR, REPO_ROOT

from agent_runtime.config import ModelSettings, load_config
from agent_runtime.contracts import AgentOutcome
from agent_runtime.federation import BenchmarkCase, run_eval_only, run_federated_experiment
from agent_runtime.hermes import preflight_external_hermes
from agent_runtime.provider_errors import FatalProviderError, raise_for_gemini_error_text
from prompt_hub import add_prompt_file_argument
from tasks import scienceagentbench_apptainer as sab_apptainer
from tasks import scienceagentbench_local as sab_local

def _split_fraction(value: str) -> float:
    """Argparse validator for a nonempty train/held-out partition."""
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split must be a number in (0, 1)") from exc
    if not 0.0 < fraction < 1.0:
        raise argparse.ArgumentTypeError("split must be strictly between 0 and 1")
    return fraction


def _resolve_eval_agent_version(output_dir: Path, explicit: Optional[str]) -> Path:
    """The client_agent directory to evaluate: explicit path, else the active
    version recorded in the experiment manifest, else the checked-in seed."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    manifest_path = output_dir / "agent_versions" / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = Path(str(manifest.get("client_agent") or "")).expanduser()
            if candidate.is_dir():
                return candidate.resolve()
        except Exception:
            pass
    return (REPO_ROOT / "client_agent").resolve()



LOCK_PATH = REPO_ROOT / "third_party" / "scienceagentbench.lock.json"
DEFAULT_BENCH_DIR = BENCHMARKS_DIR / "scienceagentbench"
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# Upstream keeps generated state inside the checkout (its .gitignore already
# covers benchmark/, pred_programs/, logs/ and *.jsonl), so only the pinned
# evaluation code has to stay untouched for a run to be official.
PINNED_PATHS = (
    "evaluation",
    "compute_scores.py",
    "config_conda_env.py",
    "gpt4_visual_judge.py",
    "run_eval.py",
)
REQUIRED_FIELDS = (
    "instance_id",
    "domain",
    "subtask_categories",
    "github_name",
    "task_inst",
    "domain_knowledge",
    "dataset_folder_tree",
    "dataset_preview",
    "src_file_or_path",
    "gold_program_name",
    "output_fname",
    "eval_script_name",
)
ARTIFACT_DIRS = ("datasets", "eval_programs", "gold_programs")
HARNESS_MODULES = ("docker", "datasets", "dotenv", "requests", "tqdm")


def judge_metadata(model: str) -> dict:
    """Persist judge identity so different visual judges cannot share checkpoints."""
    adapter = Path(__file__).with_name("scienceagentbench_visual_judge.py")
    driver = Path(__file__).with_name("scienceagentbench_eval_driver.py")
    return {
        "provider": "gemini", "model": model, "samples": 3,
        "transport": "google_native_generateContent",
        "prompt": "pinned_upstream_PROMPT_ORIGIN",
        "adapter_sha256": hashlib.sha256(adapter.read_bytes() + driver.read_bytes()).hexdigest(),
    }


def judge_key_environment(explicit: Optional[str], environment: Dict[str, str]) -> str:
    if explicit:
        return explicit
    return next((name for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY")
                 if environment.get(name)), "GEMINI_API_KEY")


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


# ---------------------------------------------------------------------------
# Pinned checkout, task inputs, and benchmark artifacts
# ---------------------------------------------------------------------------

def resolve_checkout(raw: Optional[str], *, clone: bool = True) -> Path:
    lock = _lock()
    path = Path(raw).expanduser().resolve() if raw else DEFAULT_BENCH_DIR
    if not path.exists():
        if not clone:
            raise FileNotFoundError(
                f"Missing ScienceAgentBench checkout: {path}; allow cloning first"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", lock["repository"], str(path)], check=True)
        # The upstream release is a branch head rather than a tag, so pin the
        # commit explicitly instead of trusting whatever main points at today.
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", lock["commit"]], check=True
        )
    actual = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if actual != lock["commit"]:
        raise ValueError(
            f"Expected ScienceAgentBench {lock['commit']}, found {actual}. "
            "Use a separate --bench-dir."
        )
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--", *PINNED_PATHS],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(
            f"Pinned ScienceAgentBench evaluation code has local changes in {path}:\n{dirty}"
        )
    return path


def _download_annotations(lock: dict, split: str) -> List[dict]:
    """Fetch the annotation sheet, preferring the pinned dataset revision."""
    try:
        from datasets import load_dataset  # type: ignore

        config = "validation" if split == "validation" else lock["dataset_config"]
        dataset = load_dataset(
            lock["dataset"], config, split=split, revision=lock["dataset_revision"]
        )
        return [dict(row) for row in dataset]
    except ImportError:
        pass
    # Without `datasets` installed the public rows API serves the same parquet
    # content; --validate-only and prompt building then work on a bare env.
    rows: List[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": lock["dataset"],
                "config": "validation" if split == "validation" else lock["dataset_config"],
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        with urllib.request.urlopen(
            "https://datasets-server.huggingface.co/rows?" + query, timeout=120
        ) as response:
            payload = json.load(response)
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(payload.get("num_rows_total", len(rows))):
            break
    return rows


def load_instances(bench_dir: Path, split: str, *, refresh: bool = False) -> List[dict]:
    """Return the pinned annotation sheet, caching it inside the checkout."""
    lock = _lock()
    cache = bench_dir / "foh_annotations" / f"{split}.jsonl"
    if cache.is_file() and not refresh:
        instances = [
            json.loads(line)
            for line in cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        instances = _download_annotations(lock, split)
        text = (
            "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) for row in instances
            )
            + "\n"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
        cache.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "dataset": lock["dataset"],
                    "config": "validation" if split == "validation" else lock["dataset_config"],
                    "split": split,
                    "revision": lock["dataset_revision"],
                    "count": len(instances),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return instances


def validate_instances(instances: Sequence[dict], split: str) -> List[dict]:
    lock = _lock()
    if split == lock["dataset_split"] and len(instances) != lock["task_count"]:
        raise ValueError(
            f"Expected {lock['task_count']} ScienceAgentBench tasks, found {len(instances)}"
        )
    identifiers = [str(instance.get("instance_id")) for instance in instances]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ScienceAgentBench instance IDs must be unique")
    for instance in instances:
        # domain_knowledge is optional per task; everything else is structural.
        missing = [
            field
            for field in REQUIRED_FIELDS
            if field != "domain_knowledge" and not str(instance.get(field, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"Instance {instance.get('instance_id')!r} is missing fields: {missing}"
            )
        dataset_folder(instance)
    return list(instances)


def dataset_folder(instance: dict) -> str:
    """The task's folder under ``benchmark/datasets``, as upstream derives it."""
    first_line = str(instance["dataset_folder_tree"]).split("\n")[0]
    if not first_line.startswith("|-- "):
        raise ValueError(
            f"Instance {instance.get('instance_id')!r} has an unexpected dataset tree: {first_line!r}"
        )
    return first_line[4:].strip().strip("/")


def resolve_artifacts(bench_dir: Path, archive: Optional[str] = None) -> Path:
    """Return ``<checkout>/benchmark``, unzipping the release archive if given."""
    root = bench_dir / "benchmark"
    if archive:
        unpack_artifacts(Path(archive).expanduser().resolve(), root)
    missing = [name for name in ARTIFACT_DIRS if not (root / name).is_dir()]
    if missing:
        lock = _lock()
        artifacts = lock["artifacts"]
        raise FileNotFoundError(
            f"Missing benchmark artifacts {missing} under {root}. Download "
            f"{artifacts['archive']} from {artifacts['url']}, then pass "
            f"--benchmark-archive <path> once (password: {artifacts['password']}). "
            "Upstream asks that the unzipped data files are not redistributed."
        )
    return root


def unpack_artifacts(archive: Path, root: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"No such benchmark archive: {archive}")
    password = _lock()["artifacts"]["password"]
    staging = root.parent / ".foh_artifact_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    unzip = shutil.which("unzip")
    if unzip is None:
        raise RuntimeError(
            "`unzip` is required to open the password-protected benchmark archive"
        )
    result = subprocess.run(
        [unzip, "-q", "-o", "-P", password, str(archive), "-d", str(staging)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"Could not unzip {archive.name} (exit {result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[-800:]}"
        )
    # The published archive nests the four artifact directories one or two
    # levels down depending on the release, so locate them rather than guess.
    candidates = [staging, *(path for path in staging.rglob("*") if path.is_dir())]
    source = next(
        (
            path
            for path in candidates
            if all((path / name).is_dir() for name in ARTIFACT_DIRS)
        ),
        None,
    )
    if source is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"{archive.name} does not contain {list(ARTIFACT_DIRS)}; check the download"
        )
    root.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = root / child.name
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        shutil.move(str(child), str(destination))
    shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Selection and prompts
# ---------------------------------------------------------------------------

def select_instances(instances: Sequence[dict], suite: str) -> List[dict]:
    if suite == "all":
        return list(instances)
    normalized = suite.strip().lower()
    if normalized.startswith("first_") and normalized[6:].isdigit():
        return list(instances)[: int(normalized[6:])]
    for prefix, field in (
        ("domain:", "domain"),
        ("subtask:", "subtask_categories"),
        ("repo:", "github_name"),
    ):
        if normalized.startswith(prefix):
            wanted = _slug(normalized.split(":", 1)[1])
            return [
                instance
                for instance in instances
                if wanted and wanted in _slug(instance[field])
            ]
    wanted = {value.strip() for value in suite.split(",") if value.strip()}
    return [
        instance
        for instance in instances
        if str(instance["instance_id"]) in wanted
        or instance["gold_program_name"] in wanted
    ]


def split_instances(instances: Sequence[dict], fraction: float, seed: int):
    indices = list(range(len(instances)))
    random.Random(seed).shuffle(indices)
    count = max(1, min(int(len(instances) * fraction), len(instances) - 1))
    train = [instances[index] for index in indices[:count]]
    held_out = [instances[index] for index in indices[count:]]
    return train, held_out


def task_prompt(instance: dict, *, use_knowledge: bool) -> str:
    folder = dataset_folder(instance)
    program = "pred_" + instance["gold_program_name"]
    sections = [str(instance["task_inst"]).strip()]
    knowledge = str(instance.get("domain_knowledge") or "").strip()
    if use_knowledge and knowledge:
        sections.append("Expert-provided domain knowledge:\n" + knowledge)
    sections.append(
        f"You can access the dataset at `benchmark/datasets/{folder}/`, relative to your "
        "workspace root. Here is the directory structure of the dataset:\n"
        f"```\n{instance['dataset_folder_tree']}\n```\n"
        f"Here are some helpful previews for the dataset file(s):\n{instance['dataset_preview']}"
    )
    sections.append(
        "Deliverable:\n"
        f"- Save one complete, self-contained Python program to `pred_programs/{program}` in "
        "your workspace. That single file is the only artifact the official evaluator reads.\n"
        f"- The evaluator copies it to `program_to_eval/{program}` inside a fresh container "
        f"and runs `python -m program_to_eval.{program[:-3]}` from the directory that holds "
        "`benchmark/`, so use relative paths such as "
        f"`benchmark/datasets/{folder}/...` and never absolute paths from this workspace.\n"
        f"- The program must save its output to `{instance['output_fname']}` exactly, "
        "creating the directory if needed.\n"
        "- Third-party packages are installed automatically from the imports the program "
        "declares, so import every package you use at module top level and never use "
        "interactive commands such as `!pip install numpy`.\n"
        "- Compute the result; do not hard-code expected outputs, and do not read the "
        "evaluation scripts.\n"
        "You may install packages, run, and debug the program inside your workspace before "
        "saving the final version."
    )
    return "\n\n".join(sections)


def prepare_workspace(
    workspace: Path, instance: dict, benchmark_root: Path, *, copy_datasets: bool
) -> None:
    """Reset the workspace to this task's declared inputs, as upstream does."""
    instance_id = str(instance["instance_id"])
    if workspace.name != instance_id:
        raise ValueError(
            f"Refusing to reset non-task workspace {workspace}; expected a path ending in {instance_id!r}"
        )
    if workspace.is_symlink() or workspace.is_file():
        workspace.unlink()
    elif workspace.exists():
        shutil.rmtree(workspace)
    folder = dataset_folder(instance)
    source = benchmark_root / "datasets" / folder
    if not source.is_dir():
        raise FileNotFoundError(
            f"Instance {instance_id} needs benchmark/datasets/{folder}; "
            "unpack the benchmark archive first"
        )
    (workspace / "pred_programs").mkdir(parents=True, exist_ok=True)
    (workspace / "pred_results").mkdir(parents=True, exist_ok=True)
    datasets_dir = workspace / "benchmark" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    target = datasets_dir / folder
    if copy_datasets:
        shutil.copytree(source, target)
    else:
        target.symlink_to(source, target_is_directory=True)


def make_case(
    instance: dict,
    *,
    workspace_root: Path,
    benchmark_root: Path,
    use_knowledge: bool,
    copy_datasets: bool,
) -> BenchmarkCase:
    instance_id = str(instance["instance_id"])
    workspace = workspace_root / instance_id
    program = "pred_" + instance["gold_program_name"]
    return BenchmarkCase(
        case_id=instance_id,
        prompt=task_prompt(instance, use_knowledge=use_knowledge),
        workspace=workspace,
        attachments=[str(workspace / "benchmark" / "datasets" / dataset_folder(instance))],
        metadata={
            "benchmark": "scienceagentbench",
            "instance_id": instance_id,
            "domain": instance["domain"],
            "subtask_categories": instance["subtask_categories"],
            "github_name": instance["github_name"],
            "output_fname": instance["output_fname"],
            "program_path": f"pred_programs/{program}",
        },
        prepare=lambda target, selected=instance: prepare_workspace(
            target, selected, benchmark_root, copy_datasets=copy_datasets
        ),
    )


# ---------------------------------------------------------------------------
# Official containerized evaluation
# ---------------------------------------------------------------------------

def locate_program(workspace: Path, instance: dict) -> Optional[Path]:
    """Find the submitted program, tolerating the two obvious near-misses."""
    gold = instance["gold_program_name"]
    candidates = (
        workspace / "pred_programs" / f"pred_{gold}",
        workspace / "pred_programs" / gold,
        workspace / f"pred_{gold}",
    )
    return next((path for path in candidates if path.is_file()), None)


def purge_instance_images(instance_id: str) -> None:
    """Drop any cached evaluation image for this instance before grading.

    The upstream instance image bakes the predicted program in at build time
    and is reused whenever the tag already exists, so a leftover image from an
    interrupted round would otherwise re-grade that round's program.
    """
    docker = shutil.which("docker")
    if docker is None:
        return
    tags = [f"sab.eval.{arch}.{instance_id}:latest" for arch in ("arm64", "x86_64")]
    subprocess.run(
        [docker, "image", "rm", "-f", *tags],
        capture_output=True,
        text=True,
        check=False,
    )


def run_official_eval(
    instance: dict,
    *,
    bench_dir: Path,
    benchmark_root: Path,
    pred_dir: Path,
    log_dir: Path,
    split: str,
    run_id: str,
    cache_level: str,
    timeout: int,
    judge_model: Optional[str] = None,
    judge_api_key_env: Optional[str] = None,
    judge_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the pinned Docker harness, optionally replacing its visual judge."""
    instance_id = str(instance["instance_id"])
    purge_instance_images(instance_id)
    command = [
        sys.executable,
        "-u",
        "-m",
        "evaluation.harness.run_evaluation",
        "--benchmark_path", str(benchmark_root),
        "--pred_program_path", str(pred_dir),
        "--log_fname", str(log_dir / f"{run_id}.jsonl"),
        "--dataset_name", _lock()["dataset"],
        "--split", split,
        "--instance_ids", instance_id,
        "--max_workers", "1",
        "--cache_level", cache_level,
        "--clean", "True",
        "--run_id", run_id,
        "--timeout", str(timeout),
    ]
    environment = dict(os.environ)
    if judge_api_key is not None:
        judge_api_key_env = "FOH_JUDGE_PROCESS_KEY"
        environment[judge_api_key_env] = judge_api_key
    if judge_model:
        command = [
            sys.executable, "-u", str(Path(__file__).with_name("scienceagentbench_eval_driver.py")),
            "--judge-model", judge_model,
            "--judge-api-key-env", judge_key_environment(judge_api_key_env, dict(os.environ)),
            "--", *command[4:],
        ]
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript = log_dir / f"{run_id}.log"
    try:
        completed = subprocess.run(
            command,
            cwd=str(bench_dir),
            capture_output=True,
            text=True,
            env=environment,
            # The harness caps the instance itself; this only bounds image
            # builds and cleanup around it.
            timeout=timeout + 5400,
        )
    except subprocess.TimeoutExpired as expired:
        # Partial output is bytes on some Python versions even in text mode.
        partial = expired.stderr or expired.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        transcript.write_text(
            f"$ {' '.join(command)}\n\n[timeout after {expired.timeout}s]\n{partial[-4000:]}",
            encoding="utf-8",
        )
        purge_instance_images(instance_id)
        return {
            "verifier_status": "error",
            "score": 0.0,
            "error": f"Official harness did not finish within {expired.timeout}s",
            "harness_log": str(transcript),
        }
    transcript.write_text(
        f"$ {' '.join(command)}\n\n[stdout]\n{completed.stdout}\n[stderr]\n{completed.stderr}",
        encoding="utf-8",
    )
    if judge_model:
        raise_for_gemini_error_text(
            completed.stdout + "\n" + completed.stderr, model=judge_model,
        )
    result_path = (
        bench_dir / "logs" / "run_evaluation" / run_id / instance_id / "output" / "result.json"
    )
    if not result_path.is_file():
        tail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        return {
            "verifier_status": "error",
            "score": 0.0,
            "error": f"Official harness produced no result.json (exit {completed.returncode})",
            "harness_output": tail,
            "harness_log": str(transcript),
        }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if judge_model:
        raise_for_gemini_error_text(json.dumps(payload), model=judge_model)
    return result_from_payload(payload, harness_log=str(transcript))


def result_from_payload(payload: Any, *, harness_log: str) -> Dict[str, Any]:
    """Map the pinned compute_scores.py output tuple onto an FoH grade."""
    try:
        valid_program, codebert_score, success_rate, log_info = payload
    except (TypeError, ValueError):
        return {
            "verifier_status": "error",
            "score": 0.0,
            "error": f"Unexpected official result payload: {str(payload)[:500]}",
            "harness_log": harness_log,
        }
    return {
        "verifier_status": "completed",
        "score": 1.0 if success_rate == 1 else 0.0,
        "valid_program": int(valid_program),
        "codebert_score": (
            float(codebert_score) if isinstance(codebert_score, (int, float)) else None
        ),
        "success_rate": int(success_rate),
        "log_info": str(log_info)[:8000],
    }


def _backend_eval(
    backend: Any,
    runtime: str,
    instance: dict,
    *,
    judge_model: Optional[str],
    judge_api_key_env: Optional[str],
    judge_api_key: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a Docker-free verifier backend and map its output onto an FoH grade."""
    judge_key = judge_api_key
    if judge_model and judge_key is None:
        judge_key = os.environ.get(judge_key_environment(judge_api_key_env, dict(os.environ)), "")
    try:
        payload, diagnostics = backend.run_eval(
            instance, judge_model=judge_model, judge_api_key=judge_key, **kwargs,
        )
    except FatalProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - one grade must not kill the batch
        if judge_model:
            raise_for_gemini_error_text(str(exc), model=judge_model, secret=judge_key or "")
        return {
            "verifier_status": "error",
            "score": 0.0,
            "error": f"{runtime} harness crashed: {type(exc).__name__}: {str(exc)[:500]}",
            "container_runtime": runtime,
        }
    if judge_model:
        raise_for_gemini_error_text(
            str(diagnostics.get("harness_output", "")) + "\n" + json.dumps(payload),
            model=judge_model, secret=judge_key or "",
        )
    extra = {
        key: diagnostics[key]
        for key in ("build_seconds", "run_seconds", "base_image", "base_env")
        if key in diagnostics
    }
    extra["container_runtime"] = runtime
    if payload is None:
        return {
            "verifier_status": "error",
            "score": 0.0,
            "error": diagnostics.get("error", f"{runtime} harness produced no result"),
            "harness_output": diagnostics.get("harness_output", ""),
            "harness_log": diagnostics.get("harness_log", ""),
            **extra,
        }
    return {
        **result_from_payload(payload, harness_log=diagnostics.get("harness_log", "")),
        **extra,
    }


def _backend_kwargs(kwargs: Dict[str, Any], *names: str) -> Dict[str, Any]:
    return {name: kwargs[name] for name in names if name in kwargs}


_COMMON_BACKEND_ARGS = ("bench_dir", "benchmark_root", "pred_dir", "log_dir", "run_id",
                        "timeout", "scratch", "hf_cache")


def run_apptainer_eval(instance: dict, *, judge_model: Optional[str] = None,
                       judge_api_key_env: Optional[str] = None,
                       judge_api_key: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Run the pinned verifier steps in a fakeroot Apptainer overlay."""
    return _backend_eval(
        sab_apptainer, "apptainer", instance, judge_model=judge_model,
        judge_api_key_env=judge_api_key_env,
        judge_api_key=judge_api_key,
        **_backend_kwargs(kwargs, *_COMMON_BACKEND_ARGS, "base_image"),
    )


def run_local_eval(instance: dict, *, judge_model: Optional[str] = None,
                   judge_api_key_env: Optional[str] = None,
                   judge_api_key: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Run the pinned verifier steps directly on the host (no container)."""
    return _backend_eval(
        sab_local, "local", instance, judge_model=judge_model,
        judge_api_key_env=judge_api_key_env,
        judge_api_key=judge_api_key,
        **_backend_kwargs(kwargs, *_COMMON_BACKEND_ARGS, "base_prefix"),
    )


def preflight(*, grading: bool, environment: Dict[str, str],
              judge_model: Optional[str] = None, judge_api_key_env: Optional[str] = None,
              judge_api_key: Optional[str] = None,
              container_runtime: str = "docker") -> None:
    if not grading:
        return
    if container_runtime in {"apptainer", "local"}:
        if container_runtime == "apptainer" and shutil.which("apptainer") is None:
            raise RuntimeError("--container-runtime apptainer requires `apptainer` on PATH")
        if not judge_model:
            _require_openai_judge(environment)
        else:
            key_env = judge_key_environment(judge_api_key_env, environment)
            if not (judge_api_key if judge_api_key is not None else environment.get(key_env)):
                raise RuntimeError(f"Gemini visual grading requires a Google API key: set {key_env}")
        return
    missing = []
    for module in HARNESS_MODULES:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"The official ScienceAgentBench harness needs {missing}; install "
            "requirements-scienceagentbench.txt"
        )
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Official ScienceAgentBench grading requires a local Docker daemon")
    info = subprocess.run([docker, "info"], capture_output=True, text=True)
    if info.returncode != 0:
        raise RuntimeError(
            "Docker is installed but not usable: "
            + (info.stderr or info.stdout).strip()[-500:]
        )
    if judge_model:
        key_env = judge_key_environment(judge_api_key_env, environment)
        if not (judge_api_key if judge_api_key is not None else environment.get(key_env)):
            raise RuntimeError(f"Gemini visual grading requires a Google API key: set {key_env}")
        return
    _require_openai_judge(environment)


def _require_openai_judge(environment: Dict[str, str]) -> None:
    azure = (
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    )
    if not environment.get("OPENAI_API_KEY") and not all(environment.get(name) for name in azure):
        raise RuntimeError(
            "The official evaluation judges output figures with GPT-4o: export "
            "OPENAI_API_KEY or the four AZURE_OPENAI_* variables"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bench-dir", default=None)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument(
        "--benchmark-archive",
        default=None,
        help=(
            "Password-protected release archive (benchmark_verified.zip). Unpacked "
            "into <bench-dir>/benchmark once; omit it afterwards."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Annotation sheet split (default: the pinned 'verified' split)",
    )
    parser.add_argument("--refresh-annotations", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Grade one selected task's own gold program through the official "
            "harness; no model calls. It must score 1.0."
        ),
    )
    parser.add_argument(
        "--suite",
        default="all",
        help=(
            "all, first_N, domain:<slug>, subtask:<slug>, repo:<slug>, or a "
            "comma-separated list of instance IDs / gold program names"
        ),
    )
    parser.add_argument(
        "--use-knowledge",
        action="store_true",
        help="Include each task's expert-provided domain knowledge in the prompt",
    )
    parser.add_argument(
        "--copy-datasets",
        action="store_true",
        help="Copy each task's dataset folder into the workspace instead of symlinking it",
    )
    parser.add_argument("--iterations", "--rounds", dest="iterations", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--agent-version")
    parser.add_argument("--provider", choices=("openrouter", "gemini"),
                        help="Override model provider; changing providers also requires --agent-model")
    parser.add_argument("--agent-model", help="Model ID for the selected provider (e.g. gemini-2.5-flash)")
    parser.add_argument(
        "--api_key", "--api-key", dest="api_key", metavar="KEY",
        help=("Explicit API key, required for model runs; environment credentials are not "
              "used as a fallback. Also used by --judge-model. The value is not saved in configs."),
    )
    parser.add_argument("--split", type=_split_fraction, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument("--judge-model", help="Use this Gemini model as the visual judge; default is upstream GPT-4o")
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help=(
            "Concurrent official evaluations. Each one holds a container plus its "
            "instance image, so budget roughly 25 GB of disk per worker."
        ),
    )
    parser.add_argument(
        "--cache-level",
        choices=("none", "base"),
        default="base",
        help=(
            "Docker image cache kept between evaluations. Upstream's 'instance' "
            "level is rejected here: an instance image bakes in the program it "
            "was built from, so reusing it across rounds would grade a stale "
            "program. 'base' keeps the shared base image, 'none' keeps nothing."
        ),
    )
    parser.add_argument(
        "--container-runtime",
        choices=("docker", "apptainer", "local"),
        default="docker",
        help=(
            "Where the official verifier runs. 'local' replays upstream's "
            "Dockerfile steps directly on the host (a shared Miniconda base plus "
            "a fresh venv per evaluation); 'apptainer' replays them in a "
            "fakeroot Apptainer overlay. Both need no Docker daemon."
        ),
    )
    parser.add_argument(
        "--apptainer-cache",
        default=None,
        help=(
            "Shared directory for the base environment/SIF and model caches "
            "(default: <bench-dir>/.foh_local or <bench-dir>/.foh_apptainer)"
        ),
    )
    parser.add_argument(
        "--apptainer-scratch",
        default=None,
        help="Node-local scratch for per-evaluation overlays (default: $FOH_APPTAINER_SCRATCH, $TMPDIR or /tmp)",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=1800,
        help="Per-instance execution timeout inside the evaluation container",
    )
    parser.add_argument("--output-dir", default="runs/scienceagentbench")
    parser.add_argument("--workspace-root")
    parser.add_argument("--run-iteration", type=int)
    parser.add_argument("--force-iteration", action="store_true")
    add_prompt_file_argument(parser)
    args = parser.parse_args()
    if args.api_key is not None and not args.api_key.strip():
        parser.error("--api_key must not be empty")
    if args.iterations < 1 or args.num_workers < 1 or args.eval_workers < 1:
        parser.error("iterations, num-workers and eval-workers must be positive")
    if args.eval_only and (args.run_iteration is not None or args.force_iteration):
        parser.error("--run-iteration and --force-iteration apply only to optimization")
    if args.smoke and args.no_grade:
        parser.error("--smoke exists to exercise grading; drop --no-grade")
    if args.judge_model and args.no_grade:
        parser.error("--judge-model enables Gemini grading; drop --no-grade")
    if args.api_key is None and not args.validate_only and (not args.smoke or args.judge_model):
        parser.error("--api_key is required for model runs; environment credentials are not used as a fallback")
    if args.judge_model:
        try:
            args.judge_model = ModelSettings(provider="gemini", name=args.judge_model).name
        except ValueError as exc:
            parser.error(str(exc))

    lock = _lock()
    split = args.dataset_split or lock["dataset_split"]
    bench_dir = resolve_checkout(args.bench_dir, clone=not args.no_clone)
    all_instances = validate_instances(
        load_instances(bench_dir, split, refresh=args.refresh_annotations), split
    )
    selected = select_instances(all_instances, args.suite)
    if not selected:
        parser.error(f"Suite {args.suite!r} selected no ScienceAgentBench tasks")

    if args.validate_only:
        # Unpacking is still allowed here so the artifact archive can be
        # installed and checked without containers or model calls.
        if args.benchmark_archive:
            resolve_artifacts(bench_dir, args.benchmark_archive)
        print(
            json.dumps(
                {
                    "benchmark": lock["name"],
                    "checkout_commit": lock["commit"],
                    "dataset": f"{lock['dataset']}@{lock['dataset_revision']}",
                    "dataset_split": split,
                    "total_tasks": len(all_instances),
                    "selected_tasks": len(selected),
                    "domains": sorted({instance["domain"] for instance in all_instances}),
                    "artifacts_present": all(
                        (bench_dir / "benchmark" / name).is_dir() for name in ARTIFACT_DIRS
                    ),
                    "selected_instance_ids": [
                        str(instance["instance_id"]) for instance in selected
                    ],
                },
                indent=2,
            )
        )
        return

    benchmark_root = resolve_artifacts(bench_dir, args.benchmark_archive)
    grading = not args.no_grade
    preflight(grading=grading, environment=dict(os.environ),
              judge_model=args.judge_model, judge_api_key=args.api_key,
              container_runtime=args.container_runtime)
    judge_options = ({"judge_model": args.judge_model,
                      "judge_api_key": args.api_key} if args.judge_model else {})
    evaluate = run_official_eval
    if grading and args.container_runtime in {"apptainer", "local"}:
        backend_cache = (
            Path(args.apptainer_cache).expanduser().resolve() if args.apptainer_cache
            else bench_dir / (".foh_apptainer" if args.container_runtime == "apptainer" else ".foh_local")
        )
        scratch = sab_apptainer.scratch_root(args.apptainer_scratch)
        judge_options.update(scratch=scratch, hf_cache=backend_cache / "hf_cache")
        if args.container_runtime == "apptainer":
            judge_options["base_image"] = sab_apptainer.ensure_base_image(
                bench_dir, backend_cache, scratch=scratch)
            evaluate = run_apptainer_eval
        else:
            judge_options["base_prefix"] = sab_local.ensure_base_env(
                bench_dir, backend_cache, scratch=scratch)
            evaluate = run_local_eval
    judge_identity = judge_metadata(args.judge_model) if args.judge_model else None

    output = Path(args.output_dir).expanduser().resolve()
    eval_logs = output / "eval_logs"
    staging_root = output / "pred_programs"
    instance_by_id = {str(instance["instance_id"]): instance for instance in selected}
    eval_slots = threading.Semaphore(args.eval_workers)
    run_counter = {"value": 0}
    counter_lock = threading.Lock()

    def next_run_id(instance_id: str) -> str:
        with counter_lock:
            run_counter["value"] += 1
            index = run_counter["value"]
        return f"foh-{os.getpid()}-{index:04d}-{instance_id}"

    if args.smoke:
        if len(selected) != 1:
            parser.error("--smoke requires exactly one task")
        instance = selected[0]
        instance_id = str(instance["instance_id"])
        pred_dir = staging_root / f"gold-{instance_id}"
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        pred_dir.mkdir(parents=True)
        shutil.copyfile(
            benchmark_root / "gold_programs" / instance["gold_program_name"],
            pred_dir / ("pred_" + instance["gold_program_name"]),
        )
        grade_result = evaluate(
            instance,
            bench_dir=bench_dir,
            benchmark_root=benchmark_root,
            pred_dir=pred_dir,
            log_dir=eval_logs,
            split=split,
            run_id=next_run_id(instance_id),
            cache_level=args.cache_level,
            timeout=args.eval_timeout,
            **judge_options,
        )
        if judge_identity:
            grade_result.update(judge=judge_identity, grading_type="scienceagentbench_gemini_visual_judge")
        print(json.dumps(grade_result, indent=2))
        if grade_result.get("score") != 1.0:
            raise RuntimeError("The gold program did not pass its own official evaluation")
        return

    config = load_config(args.config)
    if args.provider and args.provider != config.model.provider:
        if not args.agent_model:
            parser.error("Changing --provider requires --agent-model")
        config = replace(config, model=replace(
            config.model,
            provider=args.provider,
            name=args.agent_model,
            api_key_env="GEMINI_API_KEY" if args.provider == "gemini" else "OPENROUTER_API_KEY",
            base_url=None,
        ))
    if args.agent_model:
        config = replace(
            config,
            model=replace(
                config.model,
                name=args.agent_model,
            ),
        )
    config = replace(config, model=replace(config.model, api_key_value=args.api_key),
                     apply_all_revisions=True)
    preflight_external_hermes(config)

    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else output / "workspaces"
    )
    cases = [
        make_case(
            instance,
            workspace_root=workspace_root,
            benchmark_root=benchmark_root,
            use_knowledge=args.use_knowledge,
            copy_datasets=args.copy_datasets,
        )
        for instance in selected
    ]

    def grade(case: BenchmarkCase, outcome: AgentOutcome) -> Dict[str, Any]:
        instance = instance_by_id[case.case_id]
        program_name = "pred_" + instance["gold_program_name"]
        base = {
            "task_id": case.case_id,
            "instance_id": case.case_id,
            "max_score": 1.0,
            "grading_type": "official_scienceagentbench_docker_harness",
            "agent_status": outcome.status,
        }
        if judge_identity:
            base.update(judge=judge_identity, grading_type="scienceagentbench_gemini_visual_judge")
        if args.no_grade:
            return {**base, "score": None, "grading_type": "disabled"}
        program = locate_program(case.workspace, instance)
        if program is None:
            return {
                **base,
                "score": 0.0,
                "verifier_status": "error",
                "error": f"The agent saved no pred_programs/{program_name}",
            }
        run_id = next_run_id(case.case_id)
        # The program file is the whole submission, so an agent that ran out of
        # budget after writing one is still graded on what it produced.
        pred_dir = staging_root / run_id
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        pred_dir.mkdir(parents=True)
        shutil.copyfile(program, pred_dir / program_name)
        with eval_slots:
            result = evaluate(
                instance,
                bench_dir=bench_dir,
                benchmark_root=benchmark_root,
                pred_dir=pred_dir,
                log_dir=eval_logs,
                split=split,
                run_id=run_id,
                cache_level=args.cache_level,
                timeout=args.eval_timeout,
                **judge_options,
            )
        return {
            **base,
            **result,
            "run_id": run_id,
            "submitted_program": str(program.relative_to(case.workspace)),
            "program": str(pred_dir / program_name),
        }

    identity = {
        "benchmark": lock["name"],
        "checkout_commit": lock["commit"],
        "dataset": lock["dataset"],
        "dataset_revision": lock["dataset_revision"],
        "dataset_split": split,
        "grader": lock["grader"],
        "grading_disabled": bool(args.no_grade),
        "use_knowledge": bool(args.use_knowledge),
        "cache_level": args.cache_level,
        "eval_timeout": args.eval_timeout,
    }
    if judge_identity:
        identity.update(judge=judge_identity, grader="scienceagentbench_gemini_visual_judge")
    if grading and args.container_runtime in {"apptainer", "local"}:
        adapter = Path((sab_apptainer if args.container_runtime == "apptainer" else sab_local).__file__)
        identity.update(
            container_runtime=args.container_runtime,
            verifier_adapter_sha256=hashlib.sha256(
                adapter.read_bytes() + Path(sab_apptainer.__file__).read_bytes()).hexdigest(),
        )
    common = dict(
        grade=grade,
        output_dir=output,
        rounds=args.iterations,
        num_workers=args.num_workers,
        local_epochs=args.local_epochs,
        config=config,
        config_path=args.config,
        prompt_file=args.prompt_file,
        grade_identity=identity,
    )
    if args.eval_only:
        result = run_eval_only(
            cases=cases,
            agent_version=_resolve_eval_agent_version(output, args.agent_version),
            **common,
        )
    else:
        train, held_out = split_instances(selected, args.split, args.seed)
        case_by_id = {case.case_id: case for case in cases}
        manifest = {
            "suite": args.suite,
            "split": args.split,
            "seed": args.seed,
            "train": [str(instance["instance_id"]) for instance in train],
            "held_out": [str(instance["instance_id"]) for instance in held_out],
        }
        result = run_federated_experiment(
            train_cases=[case_by_id[str(instance["instance_id"])] for instance in train],
            held_out_cases=[case_by_id[str(instance["instance_id"])] for instance in held_out],
            run_round=args.run_iteration,
            force_round=args.force_iteration,
            client_seed=REPO_ROOT / "client_agent",
            server_seed=REPO_ROOT / "server_agent",
            split_manifest=manifest,
            record_iteration_zero=True,
            **common,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
