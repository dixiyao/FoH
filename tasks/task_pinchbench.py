"""Canonical PinchBench runner for the autonomous agent lifecycle."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ._bootstrap import BENCHMARKS_DIR, REPO_ROOT
except ImportError:  # pragma: no cover
    from _bootstrap import BENCHMARKS_DIR, REPO_ROOT

from agent_runtime.contracts import AgentOutcome, SessionTurn, normalize_session_turns
from agent_runtime.config import load_config
from agent_runtime.federation import BenchmarkCase, run_eval_only, run_federated_experiment
from agent_runtime.hermes import preflight_external_hermes
from agent_runtime.transcripts import execution_for_benchmark
from prompt_hub import add_prompt_file_argument
from tasks._pinchbench_fws import (
    DEFAULT_TOOLS_ROOT,
    preflight_fws_tasks,
    runtime_scope_for_task,
)


DEFAULT_BENCH_DIR = BENCHMARKS_DIR / "pinchbench"
DEFAULT_CLIENT_AGENT = REPO_ROOT / "client_agent"

# The 23-task V1 roster from the PinchBench paper, matched against this
# checkout's current task IDs. --V1-only/--exclude-V1 select this fixed
# boundary rather than any property the pinchbench package itself tracks.
PINCHBENCH_V1_TASK_IDS = {
    "task_sanity", "task_calendar", "task_stock", "task_blog", "task_weather",
    "task_summary", "task_events", "task_email", "task_memory", "task_files",
    "task_workflow", "task_clawdhub", "task_skill_search", "task_image_gen",
    "task_humanizer", "task_daily_summary", "task_email_triage", "task_email_search",
    "task_market_research", "task_spreadsheet_summary", "task_eli5_pdf_summary",
    "task_openclaw_comprehension", "task_second_brain",
}


def _apply_v1_filter(tasks: List[Any], *, v1_only: bool, exclude_v1: bool) -> List[Any]:
    if v1_only:
        return [task for task in tasks if task.task_id in PINCHBENCH_V1_TASK_IDS]
    if exclude_v1:
        return [task for task in tasks if task.task_id not in PINCHBENCH_V1_TASK_IDS]
    return tasks


def _select_exact_v1_roster(tasks: List[Any]) -> List[Any]:
    """Return the fixed paper roster, rejecting incomplete/duplicate checkouts."""
    selected = [task for task in tasks if task.task_id in PINCHBENCH_V1_TASK_IDS]
    selected_ids = [str(task.task_id) for task in selected]
    missing = sorted(PINCHBENCH_V1_TASK_IDS - set(selected_ids))
    duplicates = sorted(
        task_id for task_id in set(selected_ids) if selected_ids.count(task_id) > 1
    )
    if missing or duplicates or len(selected) != len(PINCHBENCH_V1_TASK_IDS):
        details = []
        if missing:
            details.append(f"missing={missing}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        raise ValueError(
            "PinchBench V1 requires the exact 23-task paper roster; "
            + (", ".join(details) or f"found={len(selected)}")
        )
    return selected


def _load_api(bench_dir: Path):
    scripts = bench_dir / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from lib_grading import grade_task  # type: ignore
    from lib_tasks import TaskLoader  # type: ignore

    return TaskLoader, grade_task


def _select_tasks(tasks: List[Any], loader: Any, suite: str) -> List[Any]:
    if suite == "all":
        return tasks
    if suite == "core":
        wanted = set(loader.core_tasks)
        return [task for task in tasks if task.task_id in wanted]
    if suite == "automated":
        return [task for task in tasks if task.grading_type == "automated"]
    wanted = {item.strip() for item in suite.split(",") if item.strip()}
    selected = [task for task in tasks if task.task_id in wanted]
    missing = wanted - {task.task_id for task in selected}
    if missing:
        raise ValueError(f"Unknown PinchBench task IDs: {sorted(missing)}")
    return selected


def _prepare_workspace(bench_dir: Path, workspace: Path, task: Any) -> None:
    # Match PinchBench's native prepare_task_workspace contract: every task
    # attempt starts from its declared fixtures, never files left by an older
    # round or evaluation.  This happens once before local epochs/session
    # turns, so retries and one task's conversation still share their state.
    task_id = str(task.task_id)
    if not task_id or workspace.name != task_id:
        raise ValueError(
            f"Refusing to reset non-task workspace {workspace}; expected a path ending in {task_id!r}"
        )
    if workspace.is_symlink() or workspace.is_file():
        workspace.unlink()
    elif workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for spec in task.workspace_files:
        if "content" in spec:
            destination = workspace / str(spec.get("path", spec.get("dest", "input.txt")))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(spec["content"]), encoding="utf-8")
            continue
        source = (bench_dir / "assets" / str(spec["source"])).resolve()
        assets_root = (bench_dir / "assets").resolve()
        if assets_root not in source.parents:
            raise PermissionError(f"Fixture escapes PinchBench assets: {spec['source']}")
        destination = workspace / str(spec["dest"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _task_session_turns(task: Any) -> List[SessionTurn]:
    raw = (task.frontmatter or {}).get("sessions") or []
    if not raw:
        if bool((task.frontmatter or {}).get("multi_session")):
            raise ValueError(
                f"PinchBench task {task.task_id!r} declares multi_session but has no sessions"
            )
        return []
    if not isinstance(raw, list):
        raise ValueError(f"PinchBench task {task.task_id!r} sessions must be a list")
    return normalize_session_turns(raw)


def _task_prompt(task: Any, session_turns: Optional[List[SessionTurn]] = None) -> str:
    turns = _task_session_turns(task) if session_turns is None else session_turns
    if turns:
        # The first real turn is the compatibility prompt used by generic FoH
        # interfaces. The full exact sequence travels separately below; never
        # send PinchBench's documentation-only "see frontmatter" placeholder.
        return turns[0].prompt
    return (
        f"{task.prompt}\n\n"
        "Work only in the supplied workspace. Verify all requested files and return a concise "
        "completion response; the official benchmark grader will inspect both the response and workspace."
    )


def _make_case(
    task: Any,
    *,
    workspace_root: Path,
    bench_dir: Path,
    tools_root: Path = DEFAULT_TOOLS_ROOT,
) -> BenchmarkCase:
    workspace = workspace_root / task.task_id
    session_turns = _task_session_turns(task)
    fws_scope = runtime_scope_for_task(task, tools_root=tools_root)
    fws_preflight = None
    if fws_scope is not None:
        def fws_preflight():
            preflight_fws_tasks([task], tools_root=tools_root, verify_versions=False)
    return BenchmarkCase(
        case_id=task.task_id,
        prompt=_task_prompt(task, session_turns),
        workspace=workspace,
        attachments=[
            str(workspace / str(spec.get("dest", spec.get("path", ""))))
            for spec in task.workspace_files
        ],
        metadata={
            "benchmark": "pinchbench",
            "category": task.category,
            "grading_type": task.grading_type,
            "timeout_seconds": task.timeout_seconds,
        },
        session_turns=session_turns,
        preflight=fws_preflight,
        runtime_scope=fws_scope,
        prepare=lambda target, selected=task: _prepare_workspace(
            bench_dir, target, selected
        ),
    )


def _resolve_workspace_root(output_dir: Path, explicit: Optional[str]) -> Path:
    """Keep mutable fixtures inside their experiment unless explicitly shared."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return output_dir.expanduser().resolve() / "workspaces"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--suite", default="core", help="all, core, automated, or comma-separated task IDs")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "pinchbench"))
    parser.add_argument(
        "--workspace-root",
        default=None,
        help=(
            "Mutable per-task workspaces (default: <resolved --output-dir>/workspaces). "
            "Use an explicit path only when workspace sharing is intentional."
        ),
    )
    parser.add_argument(
        "--pinchbench-tools-root",
        default=str(DEFAULT_TOOLS_ROOT),
        help=(
            "Pinned Node/gws/gh/fws installation created by "
            "scripts/bootstrap_pinchbench_tools.py. Selected integration tasks "
            "are preflighted before any agent/model call."
        ),
    )
    parser.add_argument(
        "--rounds", "--iterations", "--global-iterations", dest="rounds", type=int, default=1,
        help="Global iterations: outer rounds of train/held-out + server aggregation and evolution.",
    )
    parser.add_argument("--run-iteration", type=int, default=None)
    parser.add_argument("--force-iteration", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--local-epochs", type=int, default=1,
        help=(
            "Local epochs: local retries of *one* task before it's graded or (for train cases) "
            "reflected to the server. A failed/non-completed attempt may retry in the same "
            "workspace, while a completed attempt stops immediately. The client overlay owns "
            "any adaptive recovery policy; this wrapper does not make a separate prompt-rewrite "
            "model call. Only the final local outcome is graded or uploaded. Default 1 disables "
            "local retries."
        ),
    )
    parser.add_argument(
        "--split", default=None,
        help=(
            "Train/held-out split. A float between 0 and 1 (e.g. 0.7) splits the selected "
            "tasks proportionally, train first. The literal value 'v1v2' instead ignores "
            "--suite/--V1-only/--exclude-V1 and uses the full 23-task V1 roster as train and "
            "every non-V1 task as held-out, in one combined round loop: each round runs "
            "held-out (V2) first against that round's starting client_agent, then train (V1), "
            "and sends feedback from the train/V1 results only — held-out never enters "
            "learning either way. If omitted, all selected tasks are used for training and no "
            "held-out check runs this round."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Run and grade the selected tasks against --agent-version only (using whatever "
            "insight_library.json is embedded inside it); skip the train/held-out split, "
            "server aggregation, and code evolution."
        ),
    )
    parser.add_argument(
        "--V1-only", "--v1-only", dest="v1_only", action="store_true",
        help="Run only the 23-task PinchBench V1 roster (see PINCHBENCH_V1_TASK_IDS).",
    )
    parser.add_argument(
        "--exclude-V1", "--exclude-v1", dest="exclude_v1", action="store_true",
        help="Run every selected task except the 23-task PinchBench V1 roster.",
    )
    parser.add_argument(
        "--agent-version", default=str(DEFAULT_CLIENT_AGENT),
        help=(
            "client_agent directory to evaluate in --eval-only mode (default: the seed "
            "client_agent). Its insight_library.json travels with it; there is no separate "
            "--library flag."
        ),
    )
    parser.add_argument(
        "--agent-model", default=None,
        help=(
            "Overrides model.name for client_agent + server_agent from --config, and forces "
            "model.provider=openrouter / model.api_key_env=OPENROUTER_API_KEY (export that "
            "env var yourself; never pass a key on the command line or in a config file). "
            "Omit to use the OpenRouter model in --config; operational FoH rejects "
            "mock and non-OpenRouter providers."
        ),
    )
    parser.add_argument(
        "--agent-max-tokens", type=int, default=None,
        help=(
            "Overrides model.max_tokens (task-solving completions) for --agent-model. Only "
            "meaningful together with --agent-model; measure before raising -- a value your "
            "model doesn't support fails the API call outright rather than clamping."
        ),
    )
    parser.add_argument(
        "--agent-reasoning-effort", default=None, choices=["minimal", "low", "medium", "high", "xhigh", "none"],
        help=(
            "Overrides model.reasoning_effort for --agent-model, sent as OpenRouter's "
            "{\"reasoning\": {\"effort\": ...}}. For models with mandatory reasoning at a high "
            "default effort (e.g. stealth/ox-alpha), measured live: leaving this unset can burn "
            "an entire max_tokens budget on hidden reasoning with zero visible output on a real "
            "share of calls, and exhaust the total-token budget on input alone before finishing. "
            "'low' measured 4x less wall-clock and 60%% fewer output tokens than leaving this "
            "unset on an identical task, and let it actually reach completion. Not every model "
            "supports every value here -- check the model's own listing first."
        ),
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Judge model for grading (default: the same model configured for the agent).",
    )
    add_prompt_file_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.agent_model:
        config = replace(
            config,
            model=replace(
                config.model, provider="openrouter", name=args.agent_model, api_key_env="OPENROUTER_API_KEY"
            ),
        )
    if args.agent_max_tokens:
        config = replace(config, model=replace(config.model, max_tokens=args.agent_max_tokens))
    if args.agent_reasoning_effort:
        config = replace(config, model=replace(config.model, reasoning_effort=args.agent_reasoning_effort))
    model_settings = config.model

    split_mode: Optional[str] = None
    split_value: Optional[float] = None
    if args.split is not None:
        if str(args.split).strip().lower() == "v1v2":
            split_mode = "v1v2"
        else:
            try:
                split_value = float(args.split)
            except ValueError:
                parser.error("--split must be a float between 0 and 1, or the literal value 'v1v2'")
            if not 0.0 < split_value < 1.0:
                parser.error("--split must be between 0 and 1, or the literal value 'v1v2'")
            split_mode = "ratio"
    if args.v1_only and args.exclude_v1:
        parser.error("--V1-only and --exclude-V1 are mutually exclusive")
    if split_mode == "v1v2" and (args.v1_only or args.exclude_v1):
        parser.error("--split v1v2 already splits V1 (train) from non-V1 (held-out); drop --V1-only/--exclude-V1")

    # Validate the one supported operational backend before loading the
    # benchmark or creating any workspace/output state. Task CLIs have no
    # responder injection seam, so mock/non-OpenRouter configurations fail.
    preflight_external_hermes(config)

    bench_dir = Path(args.bench_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    workspace_root = _resolve_workspace_root(output_dir, args.workspace_root)
    tools_root = Path(args.pinchbench_tools_root).expanduser().resolve()
    TaskLoader, grade_task = _load_api(bench_dir)
    loader = TaskLoader(bench_dir / "tasks")
    all_tasks = loader.load_all_tasks()

    if split_mode == "v1v2":
        if args.suite != "all":
            print(
                f"Warning: --split v1v2 uses the full task roster (train=V1, held-out=non-V1) "
                f"and ignores --suite {args.suite!r}.",
                file=sys.stderr,
            )
        try:
            _select_exact_v1_roster(all_tasks)
        except ValueError as exc:
            parser.error(str(exc))
        tasks = all_tasks
    elif args.v1_only:
        if args.suite != "all":
            print(
                f"Warning: --V1-only uses the full fixed 23-task roster and ignores "
                f"--suite {args.suite!r}.",
                file=sys.stderr,
            )
        try:
            tasks = _select_exact_v1_roster(all_tasks)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        tasks = _select_tasks(all_tasks, loader, args.suite)
        tasks = _apply_v1_filter(tasks, v1_only=args.v1_only, exclude_v1=args.exclude_v1)
        if not tasks:
            parser.error("No tasks remain after applying --suite/--V1-only/--exclude-V1")
    preflight_fws_tasks(tasks, tools_root=tools_root)
    task_by_id = {task.task_id: task for task in tasks}
    grade_identity = {
        "benchmark": "pinchbench",
        "grading_disabled": bool(args.no_grade),
        "judge_backend": "api",
        "judge_model": args.judge_model or model_settings.name,
    }

    def make_case(task: Any) -> BenchmarkCase:
        return _make_case(
            task,
            workspace_root=workspace_root,
            bench_dir=bench_dir,
            tools_root=tools_root,
        )

    def grade(case: BenchmarkCase, outcome: AgentOutcome) -> Dict[str, Any]:
        if args.no_grade:
            return {"task_id": case.case_id, "score": None, "grading_type": "disabled"}
        if outcome.status != "completed":
            # The agent didn't finish normally (budget exhausted, timeout,
            # cancelled, ...) — score 0 directly rather than handing an
            # incomplete/empty response to the judge or an automated grader
            # that scores whatever partial state happens to be on disk.
            return {
                "task_id": case.case_id,
                "score": 0.0,
                "max_score": 1.0,
                "grading_type": task_by_id[case.case_id].grading_type,
                "breakdown": {},
                "notes": f"Not graded: agent status was {outcome.status!r}, not 'completed'",
            }
        execution = execution_for_benchmark(
            outcome.raw_execution,
            solution=outcome.solution,
            status=outcome.status,
            workspace=str(case.workspace),
        )
        result = grade_task(
            task=task_by_id[case.case_id],
            execution_result=execution,
            skill_dir=bench_dir,
            judge_model=args.judge_model or model_settings.name,
            judge_backend="api",
        )
        return result.to_dict()

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        eval_cases = [make_case(task) for task in tasks]
        result = run_eval_only(
            cases=eval_cases,
            grade=grade,
            output_dir=output_dir,
            agent_version=args.agent_version,
            rounds=args.rounds,
            num_workers=args.num_workers,
            local_epochs=args.local_epochs,
            config=config,
            config_path=args.config,
            prompt_file=args.prompt_file,
            grade_identity=grade_identity,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if split_mode == "v1v2":
        train_tasks = [task for task in tasks if task.task_id in PINCHBENCH_V1_TASK_IDS]
        held_out_tasks = [task for task in tasks if task.task_id not in PINCHBENCH_V1_TASK_IDS]
        if not train_tasks or not held_out_tasks:
            parser.error(
                "--split v1v2 needs both V1 and non-V1 tasks in this checkout; "
                f"found {len(train_tasks)} V1 and {len(held_out_tasks)} non-V1"
            )
    elif split_mode is None:
        # No split requested: train on every selected task and skip the
        # in-round held-out check. This is the right choice when a separate
        # --eval-only pass over a disjoint roster is the real held-out test.
        train_tasks, held_out_tasks = tasks, []
    else:
        if len(tasks) < 2:
            parser.error("At least two tasks are required for train/held-out isolation")
        rng = random.Random(args.seed)
        rng.shuffle(tasks)
        cut = max(1, min(len(tasks) - 1, round(len(tasks) * split_value)))
        train_tasks, held_out_tasks = tasks[:cut], tasks[cut:]
    train_cases = [make_case(task) for task in train_tasks]
    held_out_cases = [make_case(task) for task in held_out_tasks]

    split_manifest = {
        "seed": args.seed,
        "train": [task.task_id for task in train_tasks],
        "held_out": [task.task_id for task in held_out_tasks],
    }
    result = run_federated_experiment(
        train_cases=train_cases,
        held_out_cases=held_out_cases,
        grade=grade,
        output_dir=output_dir,
        rounds=args.rounds,
        run_round=args.run_iteration,
        force_round=args.force_iteration,
        num_workers=args.num_workers,
        local_epochs=args.local_epochs,
        config=config,
        config_path=args.config,
        prompt_file=args.prompt_file,
        client_seed=REPO_ROOT / "client_agent",
        server_seed=REPO_ROOT / "server_agent",
        grade_identity=grade_identity,
        split_manifest=split_manifest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
