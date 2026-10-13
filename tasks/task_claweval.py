# Task entry point; implementation lives in the shared repository package.
"""
Claw-Eval (general split) runner for Federation over Harness.

Claw-Eval (https://claw-eval.github.io/, https://github.com/claw-eval/claw-eval)
is a 300-task agent benchmark split general(161)/multimodal(101)/multi_turn(38)
(task-id prefixes T/M/C respectively). This runner covers the general split
only: each task launches one or more mock HTTP services with fixture data
(e.g. a fake Gmail), exposes a handful of task-specific tools bound to those
services (e.g. gmail_list_messages), and grades via a *per-task* grader.py
that itself imports from the claw_eval package (scoring_components like
tool_called/categories_present, plus an LLM-judge rubric, plus safety_checks
like "must not call gmail_send_message").

Architecture note (why this file is shaped the way it is): claw_eval's own
reference agent loop (claw_eval.runner.loop.run_task) drives a single LLM
directly via an OpenAI-compatible provider — plugging FoH's own evolvable
client_agent in as a "provider" would only let it answer individual turns,
not actually run its own decision loop (spawn_agent, retries, its own finish
protocol, ...), which defeats the point of benchmarking the evolved agent at
all. So this runner bypasses claw_eval.runner.loop/providers entirely and
instead reuses, directly, the pieces that matter for fidelity:
  - claw_eval.models.task.TaskDefinition.from_yaml  -- task loading
  - claw_eval.runner.services.ServiceManager         -- mock-service lifecycle
  - claw_eval.runner.dispatcher.ToolDispatcher        -- HTTP tool dispatch
  - claw_eval.graders.registry.get_grader             -- official per-task grading
  - claw_eval.graders.llm_judge.LLMJudge              -- the judge graders call
  - claw_eval.models.scoring.compute_task_score/is_pass
while FoH's own client_agent (via client.solve_task -> load_agent -> the
active, possibly-evolved agent code) drives every decision.

The remaining wrinkle: agent_runtime.federation._run_batch (the shared
solve/grade pipeline every task_*.py plugs into) has no per-case hook to hand
solve_task() a custom ToolRegistry. The one existing per-task tool-extension
point it does have is RuntimeContext.metadata["mcp_servers"], already wired
through agent_runtime.mcp.register_mcp_servers. So each task's own tools
(gmail_list_messages, etc.) are exposed through a tiny per-task MCP server
(tasks/_claweval_mcp_bridge.py) rather than a bespoke ToolRegistry — see that
file's docstring for why it's stateless per call.

Known fidelity gap (disclosed, not silent): BenchmarkCase.context() never
sets RuntimeContext.allowed_tools, so the agent also keeps FoH's generic
list_files/read_file/write_file/run_command/fetch_url tools alongside the
task's own MCP-bridged ones, instead of being restricted to only the latter
the way claw_eval's own reference agent is. Scoring is still accurate for
what it checks (tool_called/categories_present/safety_checks key off of
which tools were actually dispatched, tracked faithfully regardless), but an
agent could in principle solve a task via generic tools without ever
exercising the intended mock-service tools.

Train phase:
  - load Claw-Eval general-split tasks (tasks/T*/task.yaml under the bench repo)
  - run train tasks through client.solve_task(...)
  - aggregate train residuals with HarnessServer to produce guide/library

Eval phase:
  - run held-out tasks through client.solve_task(...) with the learned guide/library
  - grade with each task's own official grader.py (Pass^1 here; claw_eval's own
    Pass^3 multi-trial metric is not replicated by this single-trial runner)

Example:
    python task_claweval.py --config config.yaml --suite first_3 --rounds 2 --split 0.5

If benchmarks/claweval is missing, the runner clones
https://github.com/claw-eval/claw-eval there. That clone brings its own
src/claw_eval package (added to sys.path at runtime, not installed) and
mock_services/ tree; its own Python dependencies (pydantic, httpx, openai)
and the optional "mcp" package (see requirements.txt) must already be
installed in this environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from ._bootstrap import BENCHMARKS_DIR, REPO_ROOT
except ImportError:  # pragma: no cover - direct script execution
    from _bootstrap import BENCHMARKS_DIR, REPO_ROOT

from agent_runtime.config import load_config
from agent_runtime.contracts import AgentOutcome, RuntimeContext
from agent_runtime.federation import BenchmarkCase, run_eval_only, run_federated_experiment
from agent_runtime.hermes import preflight_external_hermes
from prompt_hub import add_prompt_file_argument

DEFAULT_BENCH_DIR = BENCHMARKS_DIR / "claweval"
DEFAULT_BENCH_REPO = "https://github.com/claw-eval/claw-eval.git"
MCP_BRIDGE_SCRIPT = Path(__file__).resolve().parent / "_claweval_mcp_bridge.py"

# ---------------------------------------------------------------------------
# Repo management (same shape as every other task_*.py's bench-dir handling)
# ---------------------------------------------------------------------------


def _is_bench_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    tasks_dir = path / "tasks"
    src_dir = path / "src" / "claw_eval"
    return tasks_dir.is_dir() and src_dir.is_dir() and any(tasks_dir.glob("*/task.yaml"))


def _clone_bench(target: Path, repo_url: str) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileNotFoundError(
            f"{target} exists but is not a valid Claw-Eval checkout. Move it aside or pass --bench-dir."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Claw-Eval not found; cloning {repo_url} into {target}")
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to clone Claw-Eval.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def _resolve_bench_dir(raw: Optional[str], *, clone_if_missing: bool = True, repo_url: str = DEFAULT_BENCH_REPO) -> Path:
    path = Path(raw).expanduser() if raw else DEFAULT_BENCH_DIR
    if _is_bench_dir(path):
        return path.resolve()
    if clone_if_missing:
        _clone_bench(path, repo_url)
        if _is_bench_dir(path):
            return path.resolve()
    raise FileNotFoundError(f"Claw-Eval directory not found at {path}. Pass --bench-dir or allow cloning.")


def _add_claw_eval_to_path(bench_dir: Path) -> None:
    src = str(bench_dir / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# ---------------------------------------------------------------------------
# Task discovery (general split == "T"-prefixed task IDs)
# ---------------------------------------------------------------------------


def _find_general_task_dirs(bench_dir: Path) -> List[Path]:
    tasks_root = bench_dir / "tasks"
    if not tasks_root.is_dir():
        return []
    return sorted(
        child for child in tasks_root.iterdir()
        if child.is_dir() and child.name.startswith("T") and (child / "task.yaml").is_file()
    )


def _load_tasks(bench_dir: Path, suite: str) -> List["Any"]:
    """Load general-split TaskDefinition objects, then apply --suite filtering."""
    _add_claw_eval_to_path(bench_dir)
    from claw_eval.models.task import TaskDefinition

    all_tasks = []
    for task_dir in _find_general_task_dirs(bench_dir):
        try:
            all_tasks.append(TaskDefinition.from_yaml(task_dir / "task.yaml"))
        except Exception as exc:
            print(f"Warning: failed to load {task_dir}: {exc}", file=sys.stderr)
    if not all_tasks:
        raise RuntimeError(f"No Claw-Eval general-split tasks found in {bench_dir}.")

    suite = str(suite).strip()
    if not suite:
        raise ValueError("Claw-Eval --suite must not be empty")
    if suite == "all":
        return all_tasks
    match = re.match(r"^first_?(\d+)$", suite)
    if match:
        count = int(match.group(1))
        if count < 1:
            raise ValueError("Claw-Eval first_N suite requires N >= 1")
        return all_tasks[:count]
    if suite.startswith("difficulty:"):
        value = suite[len("difficulty:"):].strip().lower()
        selected = [t for t in all_tasks if t.difficulty.lower() == value]
        if not value or not selected:
            raise ValueError(f"Unknown or empty Claw-Eval difficulty suite: {suite!r}")
        return selected
    if suite.startswith("category:"):
        value = suite[len("category:"):].strip().lower()
        selected = [t for t in all_tasks if t.category.lower() == value]
        if not value or not selected:
            raise ValueError(f"Unknown or empty Claw-Eval category suite: {suite!r}")
        return selected
    ids = {item.strip() for item in suite.split(",") if item.strip()}
    selected = [t for t in all_tasks if t.task_id in ids]
    missing = sorted(ids - {t.task_id for t in selected})
    if not ids or missing:
        raise ValueError(
            "Unknown or empty Claw-Eval task suite"
            + (f": missing task IDs {missing}" if missing else "")
        )
    return selected


# ---------------------------------------------------------------------------
# Per-worker port-offset slots, so concurrent tasks' mock services don't
# collide on the ports task.yaml hard-codes (mirrors claw-eval's own `batch`
# CLI, which assigns port_offset = slot * 50 per parallel worker).
# ---------------------------------------------------------------------------


class _PortOffsetPool:
    def __init__(self, size: int) -> None:
        self._lock = threading.Lock()
        self._free = list(range(max(1, size)))

    def acquire(self) -> int:
        with self._lock:
            return self._free.pop()

    def release(self, slot: int) -> None:
        with self._lock:
            self._free.append(slot)


# ---------------------------------------------------------------------------
# Case construction: services lifecycle + MCP bridge wiring
# ---------------------------------------------------------------------------


def _make_case(task: Any, *, bench_dir: Path, workspace_root: Path, slots: _PortOffsetPool) -> BenchmarkCase:
    """One Claw-Eval task -> one BenchmarkCase.

    case.runtime_scope starts this task's mock services (with a claimed port-offset
    slot applied via task.apply_port_offset so concurrent cases don't
    collide) and stashes everything _grade needs afterward -- the running
    ServiceManager, the acquired slot, and a fresh dispatch-log path -- into
    a plain dict placed in case.metadata (see module docstring: _run_batch
    calls prepare() then solve() then grade(case, outcome) for the same case
    within the same call, so a dict built here and referenced from both
    runtime-scope closure and case.metadata is visible to _grade once it runs).

    The mcp_servers entry is built in the runtime context after a slot is
    acquired. The shared controller keeps that scope alive through official
    grading, so audit endpoints remain reachable and cleanup is guaranteed on
    prepare, solve, and grader failures. The bridge script gets the same offset
    via --port-offset since it reloads task.yaml fresh per call.
    """
    _add_claw_eval_to_path(bench_dir)
    task_dir = bench_dir / "tasks" / task.task_id
    workspace = workspace_root / task.task_id
    # Deterministic and known before prepare() runs -- case.workspace is a
    # pure function of task_id, so the dispatch-log path can be computed
    # once here and handed to both the MCP bridge's argv and prepare/_grade.
    dispatch_log = workspace / "claweval_dispatches.jsonl"
    state: Dict[str, Any] = {"task": task, "task_dir": task_dir, "bench_dir": bench_dir, "dispatch_log": dispatch_log}
    metadata_dict: Dict[str, Any] = {
        "benchmark": "claweval",
        "category": task.category,
        "grading_type": "automated",
        "timeout_seconds": task.environment.timeout_seconds,
        "_claweval_state": state,
    }

    @contextmanager
    def runtime_scope(base: RuntimeContext) -> Iterator[RuntimeContext]:
        from claw_eval.runner.services import ServiceManager

        slot = slots.acquire()
        manager = None
        try:
            runtime_task = copy.deepcopy(task)
            state["slot"] = slot
            port_offset = slot * 50
            runtime_task.apply_port_offset(port_offset)
            state["task"] = runtime_task
            manager = ServiceManager(list(runtime_task.services), cwd=bench_dir)
            manager.__enter__()
            state["service_manager"] = manager
            dispatch_log.write_text("", encoding="utf-8")
            # Do not expose the private grader state to the agent request.
            runtime_metadata = {
                key: value
                for key, value in base.metadata.items()
                if key != "_claweval_state"
            }
            runtime_metadata["mcp_servers"] = [
                {
                    "name": "claweval",
                    # The external Hermes virtualenv is first on PATH and is
                    # bootstrapped with its pinned MCP extra. Using `python`
                    # keeps this declaration portable across installs.
                    "command": "python",
                    "args": [
                        str(MCP_BRIDGE_SCRIPT),
                        "--task-dir", str(task_dir),
                        "--bench-dir", str(bench_dir),
                        "--dispatch-log", str(dispatch_log),
                        "--port-offset", str(port_offset),
                    ],
                }
            ]
            yield replace(
                base,
                metadata=runtime_metadata,
                read_only_paths=list(
                    dict.fromkeys(
                        [
                            *base.read_only_paths,
                            str(MCP_BRIDGE_SCRIPT),
                            str(bench_dir),
                        ]
                    )
                ),
            )
        finally:
            try:
                if manager is not None:
                    manager.__exit__(*sys.exc_info())
            finally:
                state.pop("service_manager", None)
                state.pop("slot", None)
                state["task"] = task
                slots.release(slot)

    return BenchmarkCase(
        case_id=task.task_id,
        prompt=task.prompt.text,
        workspace=workspace,
        metadata=metadata_dict,
        runtime_scope=runtime_scope,
    )


# ---------------------------------------------------------------------------
# Grading: reconstruct the official trace shape and invoke the task's own
# grader.py while runtime_scope still owns the live services.
# ---------------------------------------------------------------------------


def _read_dispatches(dispatch_log: Path, ToolDispatch: Any) -> List[Any]:
    if not dispatch_log.is_file():
        return []
    records = []
    for line in dispatch_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(ToolDispatch.model_validate(json.loads(line)))
        except Exception as exc:
            print(f"Warning: dropping unparseable dispatch record: {exc}", file=sys.stderr)
    return records


def _reconstruct_messages(task: Any, dispatches: List[Any], final_text: str, TraceMessage: Any, Message: Any, TextBlock: Any, ToolUseBlock: Any, ToolResultBlock: Any) -> List[Any]:
    """Rebuild a claw_eval-shaped conversation from what actually happened.

    The privacy boundary intentionally does not send raw Hermes reasoning to
    the server, so the grader-side task-tool transcript is reconstructed from
    the complete dispatch log every MCP-bridged call wrote. Every task tool
    call, its exact arguments, its exact response, and the final answer are
    real and in order -- which
    is what the graders' helpers (_get_final_assistant_text,
    format_conversation, get_service_actions, tool_called/categories_present
    scoring components) actually key off of.
    """
    messages: List[Any] = [
        TraceMessage(trace_id=task.task_id, message=Message(role="user", content=[TextBlock(text=task.prompt.text)]))
    ]
    for dispatch in dispatches:
        messages.append(
            TraceMessage(
                trace_id=task.task_id,
                message=Message(
                    role="assistant",
                    content=[ToolUseBlock(id=dispatch.tool_use_id, name=dispatch.tool_name, input=dispatch.request_body)],
                ),
            )
        )
        response_text = (
            dispatch.response_body if isinstance(dispatch.response_body, str)
            else json.dumps(dispatch.response_body, ensure_ascii=False, default=str)
        )
        messages.append(
            TraceMessage(
                trace_id=task.task_id,
                message=Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id=dispatch.tool_use_id,
                            content=[TextBlock(text=response_text)],
                            is_error=dispatch.response_status >= 400,
                        )
                    ],
                ),
            )
        )
    if final_text.strip():
        messages.append(
            TraceMessage(trace_id=task.task_id, message=Message(role="assistant", content=[TextBlock(text=final_text)]))
        )
    return messages


def _fetch_audit_data(task: Any) -> Dict[str, Any]:
    import httpx

    audit: Dict[str, Any] = {}
    for service in task.services:
        if not service.reset_endpoint:
            continue
        audit_url = service.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
        try:
            response = httpx.get(audit_url, timeout=5.0)
            audit[service.name] = response.json()
        except Exception:
            pass  # audit fetch is best-effort, same as the official CLI
    return audit


def _resolve_judge(judge_model: Optional[str], default_model: str, LLMJudge: Any) -> Any:
    import os

    return LLMJudge(
        model_id=judge_model or default_model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _make_grade(*, bench_dir: Path, judge_model: Optional[str], default_model: str):
    """Return a GradeCallback closing over run-level config (bench_dir, judge model)."""

    def grade(case: BenchmarkCase, outcome: AgentOutcome) -> Dict[str, Any]:
        _add_claw_eval_to_path(bench_dir)
        from claw_eval.graders.llm_judge import LLMJudge
        from claw_eval.graders.registry import get_grader
        from claw_eval.models.content import TextBlock, ToolResultBlock, ToolUseBlock
        from claw_eval.models.message import Message
        from claw_eval.models.scoring import compute_task_score, is_pass
        from claw_eval.models.trace import ToolDispatch, TraceMessage

        state = case.metadata.get("_claweval_state", {})
        task = state.get("task")
        if task is None:
            return {"score": 0.0, "max_score": 1.0, "grading_type": "automated", "notes": "Internal error: task definition missing from case metadata."}
        if outcome.status != "completed":
            return {
                "score": 0.0, "max_score": 1.0, "grading_type": "automated",
                "notes": f"Not graded: agent status was {outcome.status!r}, not 'completed'",
            }

        dispatches = _read_dispatches(state["dispatch_log"], ToolDispatch)
        messages = _reconstruct_messages(task, dispatches, outcome.solution, TraceMessage, Message, TextBlock, ToolUseBlock, ToolResultBlock)
        audit_data = _fetch_audit_data(task)
        judge = _resolve_judge(judge_model, default_model, LLMJudge)

        try:
            grader = get_grader(task.task_id, tasks_dir=bench_dir / "tasks", task_dir=state["task_dir"])
            scores = grader.grade(messages, dispatches, task, audit_data=audit_data, judge=judge, media_events=None, env_snapshot=None)
        except Exception as exc:
            return {"score": 0.0, "max_score": 1.0, "grading_type": "automated", "notes": f"Grading failed: {type(exc).__name__}: {exc}"}

        task_score = compute_task_score(scores)
        return {
            "score": task_score,
            "max_score": 1.0,
            "grading_type": "automated",
            "passed": is_pass(task_score),
            "breakdown": {
                "completion": scores.completion,
                "robustness": scores.robustness,
                "communication": scores.communication,
                "safety": scores.safety,
            },
            "tool_calls": len(dispatches),
        }

    return grade


def _preflight_claweval_runtime(config: Any) -> None:
    """Validate the external runtime before benchmark state is created."""
    preflight_external_hermes(config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR), help="Path to a Claw-Eval checkout (default: clone if missing).")
    parser.add_argument("--bench-repo", default=DEFAULT_BENCH_REPO)
    parser.add_argument("--no-clone", action="store_true", help="Fail instead of cloning Claw-Eval when missing.")
    parser.add_argument("--suite", default="all", help='"all", "first_N", "difficulty:<level>", "category:<name>", or comma-separated task IDs')
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "claweval"))
    parser.add_argument("--workspace-root", default=str(REPO_ROOT / "runs" / "claweval_workspaces"))
    parser.add_argument("--rounds", "--iterations", "--global-iterations", dest="rounds", type=int, default=1)
    parser.add_argument("--run-round", type=int, default=None)
    parser.add_argument("--force-round", action="store_true")
    parser.add_argument("--split", type=float, default=None, help="Train/held-out split ratio in (0, 1); omit to train on everything selected.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--agent-model", default=None, help="Overrides model.name for client_agent/server_agent, forcing provider=openrouter.")
    parser.add_argument("--judge-model", default=None, help="Judge model for grading (default: the agent's own model).")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--agent-version", default=None, help="client_agent directory to evaluate in --eval-only mode.")
    add_prompt_file_argument(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.agent_model:
        config = replace(config, model=replace(config.model, provider="openrouter", name=args.agent_model, api_key_env="OPENROUTER_API_KEY"))
    _preflight_claweval_runtime(config)

    bench_dir = _resolve_bench_dir(args.bench_dir, clone_if_missing=not args.no_clone, repo_url=args.bench_repo)
    tasks = _load_tasks(bench_dir, args.suite)
    print(f"Loaded {len(tasks)} Claw-Eval general-split task(s) for suite={args.suite!r}")

    worker_count = args.num_workers or config.num_workers
    slots = _PortOffsetPool(worker_count)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    cases = [_make_case(task, bench_dir=bench_dir, workspace_root=workspace_root, slots=slots) for task in tasks]
    grade = _make_grade(
        bench_dir=bench_dir,
        judge_model=args.judge_model,
        default_model=config.model.name,
    )
    grade_identity = {
        "benchmark": "claw-eval",
        "judge_backend": "configured_claweval_judge",
        "judge_model": args.judge_model or config.model.name,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.eval_only:
        agent_version = (
            Path(args.agent_version).expanduser().resolve()
            if args.agent_version
            else (REPO_ROOT / "client_agent").resolve()
        )
        result = run_eval_only(
            cases=cases, grade=grade, output_dir=output_dir, agent_version=agent_version,
            rounds=args.rounds, num_workers=worker_count, config=config, config_path=args.config,
            prompt_file=args.prompt_file, grade_identity=grade_identity,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "insight_library"}, indent=2, ensure_ascii=False))
        return

    if args.split is not None:
        if not 0.0 < args.split < 1.0:
            parser.error("--split must be a float strictly between 0 and 1")
        if len(cases) < 2:
            parser.error("--split requires at least two selected Claw-Eval tasks")
        import random as _random
        indices = list(range(len(cases)))
        _random.Random(args.seed).shuffle(indices)
        cut = max(1, min(int(len(indices) * args.split), len(indices) - 1))
        train_cases = [cases[i] for i in indices[:cut]]
        held_out_cases = [cases[i] for i in indices[cut:]]
    else:
        train_cases, held_out_cases = cases, []

    result = run_federated_experiment(
        train_cases=train_cases, held_out_cases=held_out_cases, grade=grade,
        output_dir=output_dir, rounds=args.rounds, run_round=args.run_round, force_round=args.force_round,
        num_workers=worker_count, config=config, config_path=args.config, prompt_file=args.prompt_file,
        client_seed=REPO_ROOT / "client_agent", server_seed=REPO_ROOT / "server_agent",
        grade_identity=grade_identity,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "insight_library"}, indent=2, ensure_ascii=False))
    print(f"Per-round summaries: {output_dir}/round_XXX/checkpoint.json")


if __name__ == "__main__":
    main()
