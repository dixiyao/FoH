"""Shared benchmark lifecycle: solve, grade, aggregate, evolve, and evaluate."""

from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import json
import os
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterable, Iterator, List, Optional, Sequence

import parallel_utils
from client import solve_task_sync, solve_task_with_local_epochs_sync
from server import bootstrap_agents, evolve_harness
from prompt_hub import resolve_prompt_file

from .config import AppConfig, load_config
from .contracts import (
    AgentOutcome,
    ClientReflection,
    RuntimeContext,
    SessionTurn,
    normalize_session_turns,
)
from .evolution import VersionStore
from .harness_git import HarnessGitSettings, publish_iteration
from .hermes import preflight_external_hermes
from .loader import directory_digest, read_embedded_library, write_embedded_library
from .model import ModelClient
from .privacy import sanitize_library, sanitize_text
from .provider_errors import FatalProviderError, ProviderBatchAbort, provider_batch_abort

@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    workspace: Path
    attachments: Sequence[str] = ()
    skills: Sequence[str] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_turns: Sequence[SessionTurn] = ()
    prepare: Optional[Callable[[Path], None]] = field(default=None, compare=False, repr=False)
    preflight: Optional[Callable[[], None]] = field(default=None, compare=False, repr=False)
    runtime_scope: Optional[
        Callable[[RuntimeContext], ContextManager[RuntimeContext]]
    ] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_turns", tuple(normalize_session_turns(self.session_turns)))

    def context(self) -> RuntimeContext:
        return RuntimeContext(
            workspace=self.workspace,
            attachments=[str(value) for value in self.attachments],
            skills=[str(value) for value in self.skills],
            metadata=dict(self.metadata),
            session_turns=list(self.session_turns),
        )


GradeCallback = Callable[[BenchmarkCase, AgentOutcome], Dict[str, Any]]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _output_directory_lock(root: Path) -> Iterator[None]:
    """Hold one fail-fast advisory lock for the full output mutation.

    Thread-only locks do not protect an experiment when two Slurm jobs or a
    resumed shell invocation point at the same output directory.  ``flock``
    is released by the kernel if the process dies, which makes it suitable
    for crash recovery without stale-lock cleanup.  The small JSON payload is
    diagnostic only; ownership is the kernel lock, not the file contents.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".foh-run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Output directory is already in use by another FoH process: {root} ({owner})"
            ) from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(root)}))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_experiment_output(function: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Decorate a keyword-only runner with its output-directory lock."""
    @functools.wraps(function)
    def locked(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        if "output_dir" not in kwargs:
            raise TypeError("output_dir must be supplied as a keyword argument")
        # Fail before even creating/locking an experiment directory.  A mock,
        # disabled, or missing external backend is a deployment error, not an
        # experiment whose zero scores/checkpoints should be persisted.
        config = kwargs.get("config") or load_config(kwargs.get("config_path", "config.yaml"))
        preflight_external_hermes(config, test_model=kwargs.get("test_model"))
        root = Path(kwargs["output_dir"]).expanduser().resolve()
        with _output_directory_lock(root):
            return function(*args, **kwargs)

    return locked


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: str(item),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _effective_grade_identity(
    grade: GradeCallback,
    explicit: Optional[Any],
) -> Dict[str, Any]:
    """Return a non-answer-bearing identity for cached grading semantics.

    Judge model/backend and disabled-grading flags live outside AppConfig in
    benchmark adapters, so callers should pass those as ``grade_identity``.
    The callable fingerprint is a safe fallback for library users; only its
    digest is persisted, never closure values or grader data.
    """

    if explicit is not None:
        return {"source": "explicit", "digest": _stable_digest(explicit)}
    code = getattr(grade, "__code__", None)
    descriptor = {
        "module": str(getattr(grade, "__module__", "")),
        "qualname": str(getattr(grade, "__qualname__", type(grade).__qualname__)),
        "bytecode": getattr(code, "co_code", b"").hex() if code is not None else "",
        "constants": repr(getattr(code, "co_consts", ())),
        "names": list(getattr(code, "co_names", ())),
    }
    return {"source": "callable_fallback", "digest": _stable_digest(descriptor)}


def _case_spec(case: BenchmarkCase) -> Dict[str, Any]:
    """Return a portable, non-answer-bearing identity for one benchmark case."""
    workspace = case.workspace.expanduser().resolve()
    attachments: List[str] = []
    for raw in case.attachments:
        path = Path(raw).expanduser()
        try:
            attachments.append(str(path.resolve().relative_to(workspace)))
        except (OSError, ValueError):
            attachments.append(str(raw))
    content = {
        "prompt": case.prompt,
        "attachments": attachments,
        "skills": [str(value) for value in case.skills],
        "metadata": case.metadata,
        "session_turns": [turn.to_dict() for turn in case.session_turns],
    }
    return {"case_id": case.case_id, "content_digest": _stable_digest(content)}


def _validated_case_ids(
    cases: Sequence[BenchmarkCase],
    *,
    label: str,
) -> List[str]:
    """Reject ambiguous case rosters before bootstrap, solving, or grading."""

    identifiers = [str(case.case_id).strip() for case in cases]
    if any(not value for value in identifiers):
        raise ValueError(f"{label} case IDs must be nonempty")
    counts: Dict[str, int] = {}
    for value in identifiers:
        counts[value] = counts.get(value, 0) + 1
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} case IDs must be unique; duplicates={duplicates}")
    return identifiers


def _build_experiment_spec(
    *,
    train: Sequence[BenchmarkCase],
    held_out: Sequence[BenchmarkCase],
    config: AppConfig,
    prompt_file: Optional[str | Path],
    worker_count: int,
    local_epochs: int,
    grade_identity: Optional[Dict[str, Any]] = None,
    execution_backend: str = "external_hermes",
) -> Dict[str, Any]:
    """Fingerprint immutable inputs while deliberately allowing more rounds.

    ``rounds`` and ``run_round`` are absent so an experiment may be extended
    in the same directory.  Task membership/content, the effective runtime
    configuration, prompt bundle, and local-epoch semantics may not drift.
    """
    train_spec = [_case_spec(case) for case in train]
    held_out_spec = [_case_spec(case) for case in held_out]
    split_payload = {"train": train_spec, "held_out": held_out_spec}
    config_payload = {
        "app_config": config.public_dict(),
        "effective_num_workers": int(worker_count),
        "local_epochs": int(local_epochs),
        "execution_backend": str(execution_backend),
    }
    prompt_path = resolve_prompt_file(prompt_file)
    immutable = {
        "schema_version": 1,
        "mode": "federated",
        "split_digest": _stable_digest(split_payload),
        "config_digest": _stable_digest(config_payload),
        "prompt_digest": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "grade_identity": dict(grade_identity or {"source": "unspecified", "digest": _stable_digest(None)}),
        "execution_backend": str(execution_backend),
    }
    return {
        **immutable,
        "fingerprint": _stable_digest(immutable),
        "train_case_ids": [item["case_id"] for item in train_spec],
        "held_out_case_ids": [item["case_id"] for item in held_out_spec],
        "note": "round count may increase; split, prompt, config, workers, local_epochs, and grading semantics are immutable",
    }


def _ensure_experiment_spec(root: Path, expected: Dict[str, Any]) -> None:
    path = root / "experiment_spec.json"
    if path.is_file():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Experiment spec is unreadable: {path}") from exc
        if not isinstance(observed, dict) or observed.get("fingerprint") != expected["fingerprint"]:
            raise RuntimeError(
                "Output directory belongs to a different experiment specification; "
                "use a fresh output directory instead of mixing task splits or configuration"
            )
        return
    _write_json(path, expected)


def _ensure_split_manifest(
    root: Path,
    *,
    train_case_ids: Sequence[str],
    held_out_case_ids: Sequence[str],
    supplied: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist the authoritative split only while the output lock is held."""

    value = dict(supplied or {})
    expected_train = [str(item) for item in train_case_ids]
    expected_held_out = [str(item) for item in held_out_case_ids]
    if "train" in value and [str(item) for item in value["train"]] != expected_train:
        raise ValueError("split_manifest train roster does not match train_cases")
    if "held_out" in value and [str(item) for item in value["held_out"]] != expected_held_out:
        raise ValueError("split_manifest held_out roster does not match held_out_cases")
    value["train"] = expected_train
    value["held_out"] = expected_held_out
    path = root / "split_manifest.json"
    if path.is_file():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Split manifest is unreadable: {path}") from exc
        if observed != value:
            raise RuntimeError(
                "Output directory has a different committed split manifest; "
                "use a fresh output directory"
            )
        return value
    _write_json(path, value)
    return value


def _public_result(
    case: BenchmarkCase,
    outcome: AgentOutcome,
    grade: Dict[str, Any],
    *,
    include_traces: bool,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "case_id": case.case_id,
        "status": outcome.status,
        "grade": grade,
        "usage": outcome.usage.to_dict(),
        "workspace": outcome.workspace,
        "artifacts": list(outcome.artifacts),
        "telemetry": dict(outcome.telemetry),
    }
    if include_traces:
        value["reasoning_traces"] = [trace.to_dict() for trace in outcome.reasoning_traces]
        value["reflection"] = outcome.reflection.to_dict() if outcome.reflection else None
    return value


def _grade_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    graded = [row["grade"] for row in rows if isinstance(row.get("grade"), dict)]
    score = sum(float(item.get("score") or 0.0) for item in graded if item.get("score") is not None)
    maximum = sum(float(item.get("max_score") or 0.0) for item in graded if item.get("score") is not None)
    return {
        "graded": sum(1 for item in graded if item.get("score") is not None),
        "score": score,
        "max_score": maximum,
        "accuracy": score / maximum if maximum else None,
    }


def _publish_harness_branch(
    config: AppConfig,
    *,
    root: Path,
    manifest: Dict[str, Any],
    iteration: int,
    note: str,
) -> Optional[Dict[str, Any]]:
    """Publish the active overlays as git branch ``<prefix>NNN`` when
    ``harness_git.enabled`` is set; a no-op otherwise.  Publishing failures
    are reported, not swallowed: a client waiting on the branch must not be
    told silently that it exists."""
    git = config.harness_git
    if not git.enabled:
        return None
    settings = HarnessGitSettings(
        enabled=True,
        repo=git.repo,
        branch_prefix=git.branch_prefix,
        push=git.push,
        author_name=git.author_name,
        author_email=git.author_email,
    )
    record = publish_iteration(
        settings,
        output_dir=root,
        iteration=iteration,
        client_agent=manifest["client_agent"],
        server_agent=manifest["server_agent"],
        insight_library=read_embedded_library(manifest["client_agent"]),
        note=note,
    )
    print(
        f"[harness_git] iteration {iteration:03d} -> branch {record['branch']} "
        f"@ {record['commit'][:12]} ({record['repo']})",
        flush=True,
    )
    return record


def _record_iteration_zero(
    *,
    root: Path,
    train_rows: Sequence[Dict[str, Any]],
    held_out_rows: Sequence[Dict[str, Any]],
    frozen_version: int,
    client_agent_digest: str,
    execution_backend: str,
    train_results: Path,
    held_out_results: Path,
) -> Dict[str, Any]:
    """Persist the post-bootstrap, pre-feedback baseline without rerunning tasks.

    Round 1 already evaluates both rosters with the same frozen v000 client
    before the server sees any reflection.  Reusing those rows makes the
    iteration-0 metric exact while avoiding a second paid pass over the full
    benchmark.  Training reflections remain private to round 1 and are not
    copied into this baseline checkpoint.
    """

    summary = {
        "iteration": 0,
        "stage": "post_bootstrap_pre_feedback",
        "frozen_version": int(frozen_version),
        "client_agent_digest": str(client_agent_digest),
        "execution_backend": str(execution_backend),
        "train_tasks": len(train_rows),
        "held_out_tasks": len(held_out_rows),
        "train_grade": _grade_summary(train_rows),
        "held_out_grade": _grade_summary(held_out_rows),
        "train_results": str(train_results),
        "held_out_results": str(held_out_results),
        "learning_source": None,
        "labels_used_for_learning": False,
        "note": (
            "Both scores use the bootstrapped v000 client before round-1 "
            "feedback or evolution; the same executions become round-1 inputs."
        ),
    }
    checkpoint = root / "iteration_000" / "checkpoint.json"
    _write_json(checkpoint, {"status": "completed", "summary": summary})
    return summary


def _read_committed_rows(path: Path, *, round_number: int, label: str) -> List[Dict[str, Any]]:
    """Read an artifact that must predate a committed activation."""
    if not path.is_file():
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: missing {label} artifact {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: unreadable {label} artifact {path}"
        ) from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: {label} artifact is not a result list"
        )
    return value


def _committed_history_entry(manifest: Dict[str, Any], round_number: int) -> Dict[str, Any]:
    matches = [
        item
        for item in manifest.get("history", [])
        if isinstance(item, dict) and int(item.get("version", -1)) == round_number
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: expected exactly one manifest history entry, "
            f"found {len(matches)}"
        )
    return matches[0]


def _library_from_committed_history(
    history: Dict[str, Any],
    *,
    manifest: Dict[str, Any],
    round_dir: Path,
) -> Dict[str, str]:
    """Recover the library committed with activation, including old manifests."""
    candidates: List[Any] = [history.get("insight_library")]
    server_result = history.get("server")
    if isinstance(server_result, dict):
        candidates.append(server_result.get("proposed_library"))
    for path in (round_dir / "insight_library.json", round_dir / "evolution.json"):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.append(value.get("library") if path.name == "evolution.json" and isinstance(value, dict) else value)
    # Last-resort migration path for manifests written before the atomic
    # history field existed. If the server produced no new library, the
    # intended value is the already-active client's previous embedded copy.
    candidates.append(read_embedded_library(manifest["client_agent"]))
    for value in candidates:
        if isinstance(value, dict):
            return sanitize_library(value)
    raise RuntimeError("Committed activation has no recoverable insight library")


def _revision_logs_from_history(
    version_store: VersionStore,
    round_number: int,
    history: Dict[str, Any],
) -> Dict[str, str]:
    stored = history.get("revision_logs")
    if isinstance(stored, dict):
        logs = {str(key): str(value) for key, value in stored.items()}
    else:
        base = version_store.root / "logs"
        prefix = f"v{round_number:03d}"
        logs = {
            "client_agent": str(base / f"{prefix}_client_agent.diff"),
            "client_agent_trace": str(base / f"{prefix}_client_agent_trace.json"),
            "server_agent": str(base / f"{prefix}_server_agent.diff"),
            "server_agent_trace": str(base / f"{prefix}_server_agent_trace.json"),
            "summary": str(base / f"{prefix}_summary.md"),
        }
    missing = [path for path in logs.values() if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: revision logs are missing: {missing}"
        )
    return logs


def _recover_committed_round(
    *,
    version_store: VersionStore,
    manifest: Dict[str, Any],
    round_number: int,
    round_dir: Path,
    train_case_ids: Sequence[str],
    held_out_case_ids: Sequence[str],
) -> Dict[str, Any]:
    """Finish files for an activation whose atomic manifest commit survived.

    No task, grader, or evolution code is invoked here.  Result rows and logs
    were durably written before ``VersionStore.activate``; the history entry
    written by activate contains the remaining activation/library state.
    The checkpoint is written last, so this reconstruction is itself safely
    repeatable after another interruption.
    """
    history = _committed_history_entry(manifest, round_number)
    train_path = round_dir / "train_results.json"
    held_out_path = round_dir / "held_out_results.json"
    train_rows = _read_committed_rows(
        train_path, round_number=round_number, label="training results"
    )
    held_out_rows = _read_committed_rows(
        held_out_path, round_number=round_number, label="held-out results"
    )
    if [str(row.get("case_id")) for row in train_rows] != [str(value) for value in train_case_ids]:
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: training result roster does not match experiment spec"
        )
    if [str(row.get("case_id")) for row in held_out_rows] != [str(value) for value in held_out_case_ids]:
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: held-out result roster does not match experiment spec"
        )

    learning_path = round_dir / "learning_inputs.json"
    if not learning_path.is_file():
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: missing learning input manifest"
        )
    try:
        learning = json.loads(learning_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: unreadable learning input manifest"
        ) from exc
    client_result = history.get("client") if isinstance(history.get("client"), dict) else {}
    server_result = history.get("server") if isinstance(history.get("server"), dict) else {}
    expected_digests = {
        "client_agent_digest": client_result.get("previous_digest"),
        "server_agent_digest": server_result.get("previous_digest"),
    }
    if not isinstance(learning, dict) or any(
        expected is not None and learning.get(key) != expected
        for key, expected in expected_digests.items()
    ):
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: learning inputs do not match activation lineage"
        )

    library = _library_from_committed_history(
        history, manifest=manifest, round_dir=round_dir
    )
    # Repair both mirrors before exposing a completed checkpoint. This is the
    # exact crash window that previously lost a newly proposed library.
    write_embedded_library(manifest["server_agent"], library)
    write_embedded_library(manifest["client_agent"], library)
    logs = _revision_logs_from_history(version_store, round_number, history)
    rejected_roles = [str(value) for value in history.get("rejected_roles", [])]
    rejection_streaks = {
        str(key): int(value)
        for key, value in (history.get("rejection_streaks") or {}).items()
    }
    stagnant_roles = [
        role
        for role, result in (("client_agent", client_result), ("server_agent", server_result))
        if isinstance(result.get("report"), dict) and result["report"].get("still_stagnant")
    ]
    alerts = [
        {
            "severity": "error",
            "role": role,
            "message": "Candidate revision was rejected; the previous validated implementation remains active.",
            "consecutive_rejections": int(rejection_streaks.get(role, 0)),
            "repeated_rejection": int(rejection_streaks.get(role, 0)) > 1,
        }
        for role in rejected_roles
    ]
    activation_status = str(history.get("activation_status") or "accepted")
    evolution = {
        "version": round_number,
        "revision_counts": {"client_agent": 1, "server_agent": 1},
        "activation_status": activation_status,
        "rejected_roles": rejected_roles,
        "rejection_streaks": rejection_streaks,
        "stagnant_roles": stagnant_roles,
        "alerts": alerts,
        "client": client_result,
        "server": server_result,
        "library": library,
        "manifest": manifest,
        "logs": logs,
        "recovered_from": "committed_manifest_history",
    }
    version_dir = version_store.root / f"v{round_number:03d}"
    if not version_dir.is_dir():
        raise RuntimeError(
            f"Cannot recover committed round {round_number}: version directory is missing: {version_dir}"
        )
    _write_json(version_dir / "evolution_report.json", evolution)
    _write_json(
        version_dir / "REVISION_STATUS.json",
        {
            "status": activation_status,
            "alerts": alerts,
            "stagnant_roles": stagnant_roles,
            "logs": logs,
            "recovered_from": "committed_manifest_history",
            "note": "Invalid candidates remain archived; the previous validated implementation stays active.",
        },
    )
    _write_json(round_dir / "insight_library.json", library)
    _write_json(round_dir / "evolution.json", evolution)
    summary = {
        "round": round_number,
        "frozen_version": round_number - 1,
        "activated_version": round_number,
        "train_tasks": len(train_rows),
        "held_out_tasks": len(held_out_rows),
        "train_grade": _grade_summary(train_rows),
        "held_out_grade": _grade_summary(held_out_rows),
        "held_out_evaluated_with": "frozen_pre_round_agent",
        "learning_source": "privacy_safe_typed_client_reflections_only",
        "labels_used_for_learning": False,
        "revision_status": activation_status,
        "rejected_roles": rejected_roles,
        "stagnant_roles": stagnant_roles,
        "revision_logs": logs,
        "train_results": str(train_path),
        "held_out_results": str(held_out_path),
        "resume_recovery": "committed_activation",
    }
    _write_json(
        round_dir / "checkpoint.json",
        {
            "status": "completed",
            "recovered_from": "committed_manifest_history",
            "summary": summary,
        },
    )
    return summary


def _library_digest(library: Dict[str, str]) -> str:
    """Stable identity for evaluation state excluded from the code digest."""
    payload = json.dumps(library, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _isolated_case_failure(
    case: BenchmarkCase,
    *,
    expected_digest: str,
    stage: str,
    error: Exception,
) -> AgentOutcome:
    """Return a non-sensitive failed outcome for one case execution error.

    A runner exception may contain the task prompt, a provider response, or a
    credential.  None of that belongs in persistent benchmark artifacts or
    federated learning inputs, so only the sanitized exception *type* and the
    coarse execution stage are retained.  The frozen digest is still attached
    so the batch's version invariant remains auditable for this failed case.
    """
    return AgentOutcome(
        solution="",
        status="failed",
        workspace=str(case.workspace.resolve()),
        raw_execution={
            "status": "failed",
            "transcript": [],
            "empty_assistant_response": True,
        },
        telemetry={
            "agent_version_digest": expected_digest,
            "case_failure_isolated": True,
            "case_failure_stage": sanitize_text(stage, max_chars=80),
            "case_failure_type": sanitize_text(type(error).__name__, max_chars=120),
        },
        # A failed execution has no trustworthy client-authored reflection.
        # Keeping this empty also prevents exception/grader text from entering
        # the server's learning batch.
        reflection=None,
    )


def _isolated_grade_failure(*, stage: str, error: Exception) -> Dict[str, Any]:
    """Return a deterministic zero without persisting grader error text."""
    return {
        "score": 0.0,
        "max_score": 1.0,
        "grading_type": "error",
        "grade_status": "failed",
        "error_stage": sanitize_text(stage, max_chars=80),
        "error_type": sanitize_text(type(error).__name__, max_chars=120),
    }


def _run_batch(
    cases: Sequence[BenchmarkCase],
    *,
    insight_library: Dict[str, str],
    agent_version: str | Path,
    config: AppConfig,
    config_path: str | Path,
    prompt_file: Optional[str | Path],
    num_workers: int,
    grade: GradeCallback,
    include_traces: bool,
    local_epochs: int = 1,
    test_model: Optional[ModelClient] = None,
) -> tuple[List[Dict[str, Any]], List[AgentOutcome], List[List[ClientReflection]]]:
    # Shared deployment/configuration defects must fail before workers make
    # paid calls.  Ordinary exceptions raised after a case starts are handled
    # independently below and become scoreable failed outcomes.
    for case in cases:
        if case.preflight is not None:
            case.preflight()
    if cases:
        preflight_external_hermes(config, test_model=test_model)
    expected_digest = directory_digest(agent_version)
    abort = ProviderBatchAbort()

    def finish_one(
        case: BenchmarkCase,
        outcome: AgentOutcome,
    ) -> tuple[AgentOutcome, List[ClientReflection], Dict[str, Any]]:
        abort.check()
        observed_digest = outcome.telemetry.get("agent_version_digest")
        if observed_digest != expected_digest:
            raise RuntimeError(
                "Client outcome did not use its frozen agent version: "
                f"expected {expected_digest}, observed {observed_digest}"
            )
        # Freeze this case's learning evidence before invoking any official
        # grader for it. Grades, labels, reference answers, and grader side
        # effects can therefore never enter server aggregation or code
        # optimization, even though grading now runs immediately after this
        # same case's solve instead of after the whole batch finishes.
        reflection_snapshot = (
            [ClientReflection.from_value(outcome.reflection.to_dict())]
            if outcome.reflection is not None
            else []
        )
        try:
            local_grade = grade(case, outcome)
        except FatalProviderError:
            raise
        except Exception as exc:
            # An official grader can fail independently of the paid solve.
            # Preserve the pre-grader reflection snapshot, persist a zero, and
            # keep siblings. Never retain the exception message: judges often
            # echo answers, prompts, provider payloads, or credentials there.
            local_grade = _isolated_grade_failure(stage="grader", error=exc)
        return outcome, reflection_snapshot, local_grade

    def run_one(case: BenchmarkCase) -> tuple[AgentOutcome, List[ClientReflection], Dict[str, Any]]:
        token = provider_batch_abort.set(abort)
        try:
            abort.check()
            return run_case(case)
        except FatalProviderError as exc:
            abort.abort(exc)
            raise
        finally:
            provider_batch_abort.reset(token)

    def run_case(case: BenchmarkCase) -> tuple[AgentOutcome, List[ClientReflection], Dict[str, Any]]:
        """Solve and grade one case as a single independent unit of work.

        Every case is only an API-bound client call plus a grader call, so it
        runs fully isolated from every other case in the batch: nothing here
        waits on another case's solve, another case's grade, or on the batch
        as a whole. The only cross-case barrier in the round lifecycle is the
        later server aggregation step, which genuinely needs every case's
        frozen trace snapshot at once.

        local_epochs > 1 permits several *local* retries of a non-completed
        attempt with the same prompt and workspace before grading — see
        client.solve_task_with_local_epochs. Only the final local round's
        outcome ever reaches grading or the trace snapshot below, so nothing
        from an intermediate local round is ever graded or uploaded.
        """
        # Workspace ownership is established before a benchmark runtime scope
        # starts.  If that local setup fails, the grader still receives a
        # typed failed outcome and can record a deterministic zero.
        try:
            marker = case.workspace.parent / f".{case.workspace.name}.foh-managed"
            managed_path = str(case.workspace.resolve())
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == managed_path:
                shutil.rmtree(case.workspace)
            case.workspace.mkdir(parents=True, exist_ok=True)
            marker.write_text(managed_path + "\n", encoding="utf-8")
        except Exception as exc:
            outcome = _isolated_case_failure(
                case,
                expected_digest=expected_digest,
                stage="workspace_setup",
                error=exc,
            )
            return finish_one(case, outcome)

        base_context = case.context()
        try:
            scope = (
                case.runtime_scope(base_context)
                if case.runtime_scope is not None
                else nullcontext(base_context)
            )
        except Exception as exc:
            return finish_one(
                case,
                _isolated_case_failure(
                    case,
                    expected_digest=expected_digest,
                    stage="runtime_scope_factory",
                    error=exc,
                ),
            )
        # Benchmark-owned services/environment must stay alive through
        # preparation, execution, and grading, and must also clean up when a
        # runner or BaseException exits early.
        try:
            runtime_context = scope.__enter__()
        except Exception as exc:
            return finish_one(
                case,
                _isolated_case_failure(
                    case,
                    expected_digest=expected_digest,
                    stage="runtime_scope_enter",
                    error=exc,
                ),
            )
        try:
            stage = "case_prepare"
            try:
                if case.prepare is not None:
                    case.prepare(case.workspace)
                abort.check()
                stage = "client_solve"
                solve = solve_task_with_local_epochs_sync if local_epochs > 1 else solve_task_sync
                extra_kwargs = {"local_epochs": local_epochs} if local_epochs > 1 else {}
                outcome = solve(
                    case.prompt,
                    insight_library,
                    runtime_context,
                    task_id=case.case_id,
                    agent_version=agent_version,
                    config=config,
                    config_path=config_path,
                    prompt_file=prompt_file,
                    model=test_model,
                    **extra_kwargs,
                )
            except FatalProviderError:
                raise
            except Exception as exc:
                # Isolate ordinary per-case adapter/runner failures so sibling
                # paid calls and their results are not discarded. BaseException
                # subclasses such as KeyboardInterrupt and SystemExit remain
                # fail-loud while this scope's __exit__ still performs cleanup.
                outcome = _isolated_case_failure(
                    case,
                    expected_digest=expected_digest,
                    stage=stage,
                    error=exc,
                )
            result = finish_one(case, outcome)
        except BaseException as body_error:
            # Always give the scope its cleanup callback, but never let it
            # suppress or replace KeyboardInterrupt/SystemExit/BaseException
            # (nor a fail-loud invariant raised by finish_one).
            try:
                scope.__exit__(
                    type(body_error), body_error, body_error.__traceback__
                )
            except BaseException as cleanup_error:
                # A cleanup KeyboardInterrupt/SystemExit must itself remain
                # visible when the body error was only an ordinary Exception.
                # Otherwise preserve the original, more relevant body failure.
                if isinstance(body_error, Exception) and not isinstance(
                    cleanup_error, Exception
                ):
                    raise
            raise
        try:
            scope.__exit__(None, None, None)
        except Exception as exc:
            # The case result was already graded inside the live scope. Do not
            # invoke a potentially paid/non-idempotent grader twice; replace it
            # with a deterministic zero and discard its learning reflection.
            outcome = _isolated_case_failure(
                case,
                expected_digest=expected_digest,
                stage="runtime_scope_exit",
                error=exc,
            )
            return outcome, [], _isolated_grade_failure(
                stage="runtime_scope_exit", error=exc
            )
        return result

    per_case = parallel_utils.parallel_map_ordered(run_one, cases, num_workers=max(1, int(num_workers)))
    outcomes = [item[0] for item in per_case]
    trace_snapshots = [item[1] for item in per_case]
    grades = [item[2] for item in per_case]
    rows = [
        _public_result(case, outcome, local_grade, include_traces=include_traces)
        for case, outcome, local_grade in zip(cases, outcomes, grades)
    ]
    return rows, outcomes, trace_snapshots


@_lock_experiment_output
def run_federated_experiment(
    *,
    train_cases: Iterable[BenchmarkCase],
    held_out_cases: Iterable[BenchmarkCase],
    grade: GradeCallback,
    output_dir: str | Path,
    rounds: int = 1,
    run_round: Optional[int] = None,
    force_round: bool = False,
    num_workers: Optional[int] = None,
    local_epochs: int = 1,
    config: Optional[AppConfig] = None,
    config_path: str | Path = "config.yaml",
    prompt_file: Optional[str | Path] = None,
    client_seed: str | Path = "client_agent",
    server_seed: str | Path = "server_agent",
    grade_identity: Optional[Any] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    test_model: Optional[ModelClient] = None,
    record_iteration_zero: bool = False,
) -> Dict[str, Any]:
    """Run canonical rounds while keeping held-out traces out of learning.

    local_epochs is the inner, per-task loop (same-prompt retries for a
    non-completed attempt, see client.solve_task_with_local_epochs) that
    sits underneath this function's own per-round ("global iteration") loop;
    the two are independent knobs. It applies uniformly to both held-out and
    train cases, since it's about how client_agent solves one task, not
    about the training/upload pipeline specifically — only the final local
    round's outcome is ever visible to grading or the server either way.
    """
    # ``test_model`` is deliberately API-only: task CLIs and YAML do not
    # expose it. Passing one marks persisted identities/results so a
    # deterministic unit test cannot be mistaken for an operational run.
    train = list(train_cases)
    held_out = list(held_out_cases)
    if not train:
        raise ValueError("Federated evolution requires training cases; use run_eval_only for evaluation")
    train_case_ids = _validated_case_ids(train, label="training")
    held_out_case_ids = _validated_case_ids(held_out, label="held-out")
    overlap = sorted(set(train_case_ids) & set(held_out_case_ids))
    if overlap:
        raise ValueError(
            "Training and held-out case IDs must be disjoint; "
            f"overlap={overlap}"
        )
    config = config or load_config(config_path)
    # Bootstrap itself is a paid server-model operation. Validate the shared
    # external runtime before creating a VersionStore or invoking it; the
    # defensive _run_batch preflight is necessarily too late for round zero.
    preflight_external_hermes(config, test_model=test_model)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker_count = num_workers if num_workers is not None else config.num_workers
    experiment_spec = _build_experiment_spec(
        train=train,
        held_out=held_out,
        config=config,
        prompt_file=prompt_file,
        worker_count=worker_count,
        local_epochs=local_epochs,
        grade_identity=_effective_grade_identity(grade, grade_identity),
        execution_backend=(
            "injected_test_responder" if test_model is not None else "external_hermes"
        ),
    )
    _ensure_experiment_spec(root, experiment_spec)
    _ensure_split_manifest(
        root,
        train_case_ids=train_case_ids,
        held_out_case_ids=held_out_case_ids,
        supplied=split_manifest,
    )
    version_store = VersionStore(root)
    manifest = version_store.initialize(client_seed, server_seed)
    # One-time, before round 1: give v000 a chance to be a real designed
    # implementation instead of the checked-in seed (see
    # server.bootstrap_agents). Idempotent via manifest["bootstrapped"], so
    # this is a cheap no-op on every later resume of an already-bootstrapped
    # run. Never advances active_version — round 1 below still finds
    # exactly the same active_version==0 precondition it always did.
    if not manifest.get("bootstrapped"):
        manifest = asyncio.run(
            bootstrap_agents(
                version_store=version_store,
                config=config,
                config_path=config_path,
                prompt_file=prompt_file,
                model=test_model,
            )
        )
        _publish_harness_branch(
            config,
            root=root,
            manifest=manifest,
            iteration=0,
            note="bootstrapped overlays designed from the guides before round 1",
        )
    summaries: List[Dict[str, Any]] = []
    iteration_zero: Optional[Dict[str, Any]] = None

    # Backfill the explicit baseline when resuming an older completed round-1
    # directory that already contains the exact pre-evolution result rows.
    iteration_zero_checkpoint = root / "iteration_000" / "checkpoint.json"
    if record_iteration_zero and iteration_zero_checkpoint.is_file():
        value = json.loads(iteration_zero_checkpoint.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("status") == "completed":
            candidate = value.get("summary")
            if isinstance(candidate, dict):
                iteration_zero = candidate
    if record_iteration_zero and iteration_zero is None:
        round_one_checkpoint = root / "round_001" / "checkpoint.json"
        if round_one_checkpoint.is_file():
            value = json.loads(round_one_checkpoint.read_text(encoding="utf-8"))
            round_one_summary = value.get("summary") if isinstance(value, dict) else None
            train_path = root / "round_001" / "train_results.json"
            held_out_path = root / "round_001" / "held_out_results.json"
            if (
                isinstance(value, dict)
                and value.get("status") == "completed"
                and isinstance(round_one_summary, dict)
                and train_path.is_file()
                and held_out_path.is_file()
            ):
                train_rows = _read_committed_rows(
                    train_path, round_number=1, label="training results"
                )
                held_out_rows = _read_committed_rows(
                    held_out_path, round_number=1, label="held-out results"
                )
                digests = {
                    str(row.get("telemetry", {}).get("agent_version_digest"))
                    for row in [*train_rows, *held_out_rows]
                    if isinstance(row.get("telemetry"), dict)
                    and row.get("telemetry", {}).get("agent_version_digest")
                }
                if len(digests) != 1:
                    raise RuntimeError(
                        "Cannot reconstruct iteration 0: round-1 rows do not "
                        "identify one frozen client digest"
                    )
                iteration_zero = _record_iteration_zero(
                    root=root,
                    train_rows=train_rows,
                    held_out_rows=held_out_rows,
                    frozen_version=int(round_one_summary.get("frozen_version", 0)),
                    client_agent_digest=next(iter(digests)),
                    execution_backend=(
                        "injected_test_responder"
                        if test_model is not None
                        else "external_hermes"
                    ),
                    train_results=train_path,
                    held_out_results=held_out_path,
                )

    total_rounds = max(1, int(rounds))
    if run_round is not None and not 1 <= int(run_round) <= total_rounds:
        raise ValueError("run_round must be between 1 and rounds")
    selected_rounds = [int(run_round)] if run_round is not None else list(range(1, total_rounds + 1))
    for round_number in selected_rounds:
        round_dir = root / f"round_{round_number:03d}"
        checkpoint = round_dir / "checkpoint.json"
        if checkpoint.is_file():
            existing = json.loads(checkpoint.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and not force_round:
                summaries.append(existing["summary"])
                manifest = version_store.load()
                continue
            if existing.get("status") == "completed" and force_round:
                raise RuntimeError("A completed evolution round cannot be safely forced in place; use a new output directory")

        active_version = int(manifest["active_version"])
        if active_version == round_number:
            # activate() is the round's atomic commit.  A missing checkpoint
            # after that point means bookkeeping was interrupted, not that
            # learning should run again.  Reconstruct it solely from durable
            # pre-commit artifacts plus the committed manifest history.
            recovered = _recover_committed_round(
                version_store=version_store,
                manifest=manifest,
                round_number=round_number,
                round_dir=round_dir,
                train_case_ids=[case.case_id for case in train],
                held_out_case_ids=[case.case_id for case in held_out],
            )
            summaries.append(recovered)
            manifest = version_store.load()
            continue
        if active_version != round_number - 1:
            raise RuntimeError(
                f"Round {round_number} requires active version {round_number - 1}, "
                f"found {manifest['active_version']}"
            )

        frozen = dict(manifest)
        # The library shipped inside frozen["client_agent"] — not a separate
        # tracked variable — is this round's input: it is whatever the
        # currently-active client_agent directory carries, so a version
        # directory stays one self-contained (code, library) unit.
        library = read_embedded_library(frozen["client_agent"])

        # Held-out runs first, against the same frozen (pre-round) client_agent
        # and library as training below — never against whatever this same
        # round is about to produce. That keeps "this round's held-out grade"
        # a clean measurement of the agent as it stood coming into the round,
        # not one already informed by this round's own training feedback. Its
        # raw execution is available only to the local grader; its traces are
        # neither persisted nor supplied to either learning phase.
        eval_rows, held_out_outcomes, _ = _run_batch(
            held_out,
            insight_library=library,
            agent_version=frozen["client_agent"],
            config=config,
            config_path=config_path,
            prompt_file=prompt_file,
            num_workers=worker_count,
            grade=grade,
            include_traces=False,
            local_epochs=local_epochs,
            test_model=test_model,
        )
        _write_json(round_dir / "held_out_results.json", eval_rows)
        if held_out_outcomes:
            held_out_digests = {
                str(outcome.telemetry.get("agent_version_digest")) for outcome in held_out_outcomes
            }
            if held_out_digests != {str(frozen["client_digest"])}:
                raise RuntimeError(
                    "Held-out clients did not all use the one frozen client_agent implementation: "
                    f"expected {frozen['client_digest']}, observed {sorted(held_out_digests)}"
                )

        train_rows, train_outcomes, training_reflection_groups = _run_batch(
            train,
            insight_library=library,
            agent_version=frozen["client_agent"],
            config=config,
            config_path=config_path,
            prompt_file=prompt_file,
            num_workers=worker_count,
            grade=grade,
            include_traces=True,
            local_epochs=local_epochs,
            test_model=test_model,
        )
        _write_json(round_dir / "train_results.json", train_rows)
        training_reflections = [reflection for group in training_reflection_groups for reflection in group]
        if not training_reflections:
            raise RuntimeError("No training reflections were collected; cannot evolve the harnesses")
        train_digests = {
            str(outcome.telemetry.get("agent_version_digest")) for outcome in train_outcomes
        }
        if train_digests != {str(frozen["client_digest"])}:
            raise RuntimeError(
                "Training clients did not all use the one frozen client_agent implementation: "
                f"expected {frozen['client_digest']}, observed {sorted(train_digests)}"
            )
        if record_iteration_zero and round_number == 1:
            iteration_zero = _record_iteration_zero(
                root=root,
                train_rows=train_rows,
                held_out_rows=eval_rows,
                frozen_version=int(frozen["active_version"]),
                client_agent_digest=str(frozen["client_digest"]),
                execution_backend=(
                    "injected_test_responder"
                    if test_model is not None
                    else "external_hermes"
                ),
                train_results=round_dir / "train_results.json",
                held_out_results=round_dir / "held_out_results.json",
            )
        _write_json(
            round_dir / "learning_inputs.json",
            {
                "source": "privacy_safe_typed_client_reflections",
                "reflection_count": len(training_reflections),
                "labels_used": False,
                "accuracy_used": False,
                "client_agent_digest": frozen["client_digest"],
                "server_agent_digest": frozen["server_digest"],
                "note": "This manifest is written from the pre-grading trace snapshot.",
            },
        )

        # Sequential, not parallel: Algorithm 1 has the server agent revise
        # its own harness first (producing the shared insight library as
        # part of that revision), then revise the client harness using its
        # own just-updated harness as input -- see server.evolve_harness for
        # the two-step mechanism this single call runs.
        evolution = asyncio.run(
            evolve_harness(
                version_store=version_store,
                reasoning_traces=training_reflections,
                config=config,
                config_path=config_path,
                prompt_file=prompt_file,
                model=test_model,
            )
        )
        next_library = evolution["library"]
        # Keep a per-round snapshot for audit/history — the embedded copy
        # evolve_harness writes into manifest["client_agent"] is overwritten
        # in place on every round that keeps that same directory active (e.g.
        # after a rejected code revision rolls back to an earlier directory),
        # so this round-scoped file is the only place round N's library is
        # guaranteed to still be readable once round N+1 has run.
        _write_json(round_dir / "insight_library.json", next_library)
        _write_json(round_dir / "evolution.json", evolution)
        manifest = version_store.load()
        branch_record = _publish_harness_branch(
            config,
            root=root,
            manifest=manifest,
            iteration=round_number,
            note=f"activation status: {evolution['activation_status']}",
        )
        if branch_record is not None:
            _write_json(round_dir / "harness_branch.json", branch_record)

        summary = {
            "round": round_number,
            # held-out and training both ran against this same frozen version.
            "frozen_version": frozen["active_version"],
            # what's active now, going into the next round.
            "activated_version": manifest["active_version"],
            "execution_backend": (
                "injected_test_responder" if test_model is not None else "external_hermes"
            ),
            "train_tasks": len(train_rows),
            "held_out_tasks": len(eval_rows),
            "train_grade": _grade_summary(train_rows),
            "held_out_grade": _grade_summary(eval_rows),
            "held_out_evaluated_with": "frozen_pre_round_agent",
            "learning_source": "privacy_safe_typed_client_reflections_only",
            "labels_used_for_learning": False,
            "revision_status": evolution["activation_status"],
            "rejected_roles": evolution["rejected_roles"],
            "stagnant_roles": evolution.get("stagnant_roles", []),
            "revision_logs": evolution["logs"],
            "train_results": str(round_dir / "train_results.json"),
            "held_out_results": str(round_dir / "held_out_results.json"),
        }
        _write_json(checkpoint, {"status": "completed", "summary": summary})
        summaries.append(summary)

    # Every in-round held-out metric above intentionally measures the frozen
    # client that *entered* that round.  Once the requested final version is
    # active, run one separately-labelled held-out pass so the experiment's
    # final accuracy actually measures the code/library it returns.  These
    # outcomes are evaluation-only: no reflections are persisted or supplied
    # to evolution.
    final_held_out: Optional[Dict[str, Any]] = None
    if held_out and int(manifest["active_version"]) == total_rounds:
        final_checkpoint = root / "final_held_out_checkpoint.json"
        final_library = read_embedded_library(manifest["client_agent"])
        final_identity = {
            "experiment_fingerprint": str(experiment_spec["fingerprint"]),
            "grade_identity": dict(experiment_spec["grade_identity"]),
            "active_version": int(manifest["active_version"]),
            "client_version": int(manifest.get("client_version", 0)),
            "client_agent": str(manifest["client_agent"]),
            "client_agent_digest": str(manifest["client_digest"]),
            "insight_library_digest": _library_digest(final_library),
        }
        cached: Optional[Dict[str, Any]] = None
        if final_checkpoint.is_file():
            value = json.loads(final_checkpoint.read_text(encoding="utf-8"))
            if isinstance(value, dict) and all(value.get(key) == expected for key, expected in final_identity.items()):
                cached = value
        if cached is not None:
            final_held_out = cached
        else:
            final_rows, final_outcomes, _ = _run_batch(
                held_out,
                insight_library=final_library,
                agent_version=manifest["client_agent"],
                config=config,
                config_path=config_path,
                prompt_file=prompt_file,
                num_workers=worker_count,
                grade=grade,
                include_traces=False,
                local_epochs=local_epochs,
                test_model=test_model,
            )
            observed = {
                str(outcome.telemetry.get("agent_version_digest"))
                for outcome in final_outcomes
            }
            if observed != {str(manifest["client_digest"])}:
                raise RuntimeError(
                    "Final held-out clients did not use the final active client_agent: "
                    f"expected {manifest['client_digest']}, observed {sorted(observed)}"
                )
            final_results_path = root / "final_held_out_results.json"
            _write_json(final_results_path, final_rows)
            final_held_out = {
                **final_identity,
                "held_out_tasks": len(final_rows),
                "held_out_grade": _grade_summary(final_rows),
                "evaluated_with": "final_active_client",
                "held_out_results": str(final_results_path),
                "learning_source": None,
                "reflections_used_for_learning": False,
            }
            _write_json(final_checkpoint, final_held_out)

    # Read fresh from whatever is now embedded in the final active
    # client_agent rather than tracking a variable across the loop: a run
    # that resumes entirely from completed-round checkpoints never executes
    # the body above, so a loop-local variable would not reflect reality.
    result = {
        "execution_backend": (
            "injected_test_responder" if test_model is not None else "external_hermes"
        ),
        "rounds": summaries,
        "manifest": manifest,
        "insight_library": read_embedded_library(manifest["client_agent"]),
        "final_held_out": final_held_out,
    }
    if record_iteration_zero:
        result["iteration_zero"] = iteration_zero
    _write_json(root / "experiment_summary.json", result)
    return result


@_lock_experiment_output
def run_eval_only(
    *,
    cases: Iterable[BenchmarkCase],
    grade: GradeCallback,
    output_dir: str | Path,
    agent_version: str | Path,
    insight_library: Optional[Dict[str, str]] = None,
    rounds: int = 1,
    num_workers: Optional[int] = None,
    local_epochs: int = 1,
    config: Optional[AppConfig] = None,
    config_path: str | Path = "config.yaml",
    prompt_file: Optional[str | Path] = None,
    grade_identity: Optional[Any] = None,
    test_model: Optional[ModelClient] = None,
) -> Dict[str, Any]:
    """Run and grade cases against one fixed agent version and library.

    This is the eval-only counterpart to ``run_federated_experiment``: it
    measures an already-trained ``agent_version`` without any train/held-out
    split, server aggregation, or code evolution. Nothing here writes to a
    VersionStore or changes which client/server implementation is active.

    ``insight_library`` defaults to whatever is embedded inside
    ``agent_version`` itself — pass an explicit dict (even ``{}``) only to
    override it for this call.
    """
    # This responder seam is likewise keyword-only and absent from task
    # CLIs/YAML. Make its use visible in every persisted result.
    cases = list(cases)
    _validated_case_ids(cases, label="evaluation")
    config = config or load_config(config_path)
    preflight_external_hermes(config, test_model=test_model)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker_count = num_workers if num_workers is not None else config.num_workers
    library = sanitize_library(
        read_embedded_library(agent_version) if insight_library is None else insight_library
    )
    agent_path = Path(agent_version).expanduser().resolve()
    agent_digest = directory_digest(agent_path)
    effective_grade_identity = _effective_grade_identity(grade, grade_identity)
    prompt_path = resolve_prompt_file(prompt_file)
    evaluation_identity = {
        "schema_version": 1,
        "mode": "eval_only",
        "case_digest": _stable_digest([_case_spec(case) for case in cases]),
        "config_digest": _stable_digest(
            {
                "app_config": config.public_dict(),
                "effective_num_workers": int(worker_count),
                "local_epochs": int(local_epochs),
                "execution_backend": (
                    "injected_test_responder" if test_model is not None else "external_hermes"
                ),
            }
        ),
        "prompt_digest": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "agent_version": str(agent_path),
        "agent_version_digest": agent_digest,
        "insight_library_digest": _library_digest(library),
        "grade_identity": effective_grade_identity,
    }
    evaluation_identity_digest = _stable_digest(evaluation_identity)
    summaries: List[Dict[str, Any]] = []
    for round_number in range(1, max(1, int(rounds)) + 1):
        round_dir = root / f"eval_round_{round_number:03d}"
        checkpoint = round_dir / "checkpoint.json"
        if checkpoint.is_file():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("status") == "completed"
                and cached.get("evaluation_identity_digest")
                == evaluation_identity_digest
                and isinstance(cached.get("summary"), dict)
            ):
                summaries.append(cached["summary"])
                continue
        rows, _, _ = _run_batch(
            cases,
            insight_library=library,
            agent_version=agent_path,
            config=config,
            config_path=config_path,
            prompt_file=prompt_file,
            num_workers=worker_count,
            grade=grade,
            include_traces=False,
            local_epochs=local_epochs,
            test_model=test_model,
        )
        _write_json(round_dir / "eval_results.json", rows)
        summary = {
            "round": round_number,
            "execution_backend": (
                "injected_test_responder" if test_model is not None else "external_hermes"
            ),
            "eval_tasks": len(rows),
            "eval_grade": _grade_summary(rows),
            "agent_version": str(agent_path),
            "agent_version_digest": agent_digest,
            "insight_library_digest": evaluation_identity["insight_library_digest"],
            "evaluation_identity_digest": evaluation_identity_digest,
            "grade_identity": effective_grade_identity,
            "eval_results": str(round_dir / "eval_results.json"),
        }
        _write_json(
            checkpoint,
            {
                "status": "completed",
                "evaluation_identity_digest": evaluation_identity_digest,
                "summary": summary,
            },
        )
        summaries.append(summary)

    result = {
        "mode": "eval_only",
        "execution_backend": (
            "injected_test_responder" if test_model is not None else "external_hermes"
        ),
        "rounds": summaries,
        "insight_library": library,
    }
    _write_json(root / "experiment_summary.json", result)
    return result
