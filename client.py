"""Stable interface between benchmark tasks and a versioned client agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from agent_runtime.config import AppConfig, load_config
from agent_runtime.contracts import AgentContext, AgentOutcome, ClientReflection, RuntimeContext, Usage
from agent_runtime.evolution import VersionStore
from agent_runtime.hermes import preflight_external_hermes
from agent_runtime.loader import load_agent, read_embedded_library, read_system_prompt
from agent_runtime.mcp import register_mcp_servers
from agent_runtime.model import ModelClient
from agent_runtime.privacy import sanitize_reflection, sanitize_traces
from agent_runtime.runtime import AgentBudgetExhausted, AgentRuntime
from agent_runtime.tools import ToolRegistry, prepare_tool_registry
from agent_runtime.transcripts import execution_for_benchmark
from prompt_hub import add_prompt_file_argument


# Benchmark-supplied paths an agent may additionally read from (its own repo
# checkout, a kubeconfig, ...), beyond plain task attachments. server.py's
# aggregation role has no equivalent — it isn't solving a benchmark task.
_CLIENT_EXTRA_SCOPED_READ_METADATA_KEYS = ("kubeconfig", "scenario_dir", "bench_dir", "repository_path")


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CLIENT_AGENT = REPO_ROOT / "client_agent"


def _effective_task_timeout(
    runtime_context: RuntimeContext,
    configured_seconds: float,
) -> float:
    """Cap one solve by a benchmark's declared whole-task timeout.

    Benchmark metadata is untrusted adapter input.  Reject malformed limits
    instead of silently running for the much larger global default; every
    session turn shares the returned deadline.
    """
    configured = float(configured_seconds)
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("limits.max_elapsed_seconds must be a positive finite number")
    raw = runtime_context.metadata.get("timeout_seconds")
    if raw is None:
        return configured
    if isinstance(raw, bool):
        raise ValueError("runtime timeout_seconds must be a positive finite number")
    try:
        task_seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime timeout_seconds must be a positive finite number"
        ) from exc
    if not math.isfinite(task_seconds) or task_seconds <= 0:
        raise ValueError("runtime timeout_seconds must be a positive finite number")
    return min(configured, task_seconds)


async def solve_task(
    task_prompt: str,
    insight_library: Optional[Dict[str, str]] = None,
    runtime_context: Optional[RuntimeContext] = None,
    *,
    task_id: str = "task",
    agent_version: str | Path = DEFAULT_CLIENT_AGENT,
    config: Optional[AppConfig] = None,
    config_path: str | Path = REPO_ROOT / "config.yaml",
    prompt_file: Optional[str | Path] = None,
    model: Optional[ModelClient] = None,
    tools: Optional[ToolRegistry] = None,
) -> AgentOutcome:
    """Submit one task; all workflow decisions belong to client_agent."""
    # A library shipped inside agent_version is the default: the version
    # directory is meant to be one self-contained, directly-runnable unit of
    # (code, current library). Pass insight_library explicitly (even {}) to
    # override what's embedded there.
    resolved_library = read_embedded_library(agent_version) if insight_library is None else dict(insight_library)
    app_config = config or load_config(config_path)
    context_value = runtime_context or RuntimeContext(
        workspace=Path("foh_runs") / re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    )
    backend_config = (
        replace(app_config, model=model.settings)
        if model is not None and not model.has_responder
        else app_config
    )
    preflight_external_hermes(
        backend_config,
        test_model=model if model is not None and model.has_responder else None,
    )
    effective_timeout = _effective_task_timeout(
        context_value, app_config.limits.max_elapsed_seconds
    )
    effective_limits = replace(
        app_config.limits, max_elapsed_seconds=effective_timeout
    )
    registry = prepare_tool_registry(
        context_value,
        app_config,
        tools=tools,
        extra_scoped_read_metadata_keys=_CLIENT_EXTRA_SCOPED_READ_METADATA_KEYS,
        mcp_registrar=register_mcp_servers,
    )
    loaded_version, runner = load_agent(agent_version)
    runtime = AgentRuntime(
        model=model or ModelClient(app_config.model),
        limits=effective_limits,
        tools=registry,
        base_system_prompt=read_system_prompt(
            agent_version, default="You are an autonomous problem-solving agent."
        ),
        runner=runner,
        hermes_settings=app_config.hermes,
    )
    context = AgentContext(
        objective=str(task_prompt),
        role="task-solving orchestrator",
        insight_library=resolved_library,
        runtime_context=context_value,
        task_id=str(task_id),
    )
    try:
        outcome = AgentOutcome.from_value(
            await asyncio.wait_for(runner(context, runtime), timeout=effective_timeout)
        )
    except (asyncio.TimeoutError, TimeoutError):
        outcome = AgentOutcome(solution="", status="timeout")
    except AgentBudgetExhausted as exc:
        # One task running long on tokens/model-calls/tool-calls shouldn't
        # crash the whole batch (see AgentBudgetExhausted's docstring) — mark
        # just this task as failed with an empty transcript, which grading
        # (task_pinchbench.grade) treats as an automatic score of 0 rather
        # than handing an empty response to the judge/automated grader.
        outcome = AgentOutcome(
            solution="",
            status="failed",
            raw_execution={"status": "failed", "transcript": [], "empty_assistant_response": True},
            telemetry={"error": str(exc)},
        )
    finally:
        await runtime.shutdown()
    outcome.usage = runtime.usage()
    outcome.workspace = str(context_value.workspace)
    outcome.telemetry = {
        **outcome.telemetry,
        **runtime.telemetry(),
        "agent_version_digest": loaded_version.digest,
    }
    forbidden_values = [
        str(task_id), str(task_prompt), outcome.solution,
        str(context_value.workspace), *context_value.attachments,
        *(turn.prompt for turn in context_value.session_turns),
    ]
    outcome.reasoning_traces = sanitize_traces(
        outcome.reasoning_traces,
        forbidden_values=forbidden_values,
        max_chars=app_config.limits.max_trace_chars,
    )
    fallback_reflection = ClientReflection(
        summary="The client completed a task execution and emitted reusable feedback.",
        actions=[trace.action for trace in outcome.reasoning_traces],
        observations=[trace.observation for trace in outcome.reasoning_traces],
        reusable_insights=[trace.reusable_feedback for trace in outcome.reasoning_traces],
        outcome=outcome.status,
    )
    reflection_value = (
        outcome.reflection
        if outcome.reflection is not None and outcome.reflection.has_content()
        else fallback_reflection
    )
    clean_reflection = sanitize_reflection(
        reflection_value,
        forbidden_values=forbidden_values,
        max_chars=app_config.limits.max_trace_chars,
    )
    # A model-supplied object can be structurally valid yet empty (or become
    # empty after privacy filtering).  Never upload that as if a meaningful
    # client reflection existed.
    outcome.reflection = (
        clean_reflection
        if clean_reflection.has_content()
        else sanitize_reflection(
            fallback_reflection,
            max_chars=app_config.limits.max_trace_chars,
        )
    )
    return outcome


def solve_task_sync(*args: Any, **kwargs: Any) -> AgentOutcome:
    return asyncio.run(solve_task(*args, **kwargs))


async def solve_task_with_local_epochs(
    task_prompt: str,
    insight_library: Optional[Dict[str, str]] = None,
    runtime_context: Optional[RuntimeContext] = None,
    *,
    task_id: str = "task",
    agent_version: str | Path = DEFAULT_CLIENT_AGENT,
    config: Optional[AppConfig] = None,
    config_path: str | Path = REPO_ROOT / "config.yaml",
    prompt_file: Optional[str | Path] = None,
    model: Optional[ModelClient] = None,
    tools: Optional[ToolRegistry] = None,
    local_epochs: int = 1,
) -> AgentOutcome:
    """Run client_agent for up to local_epochs local rounds on one task.

    This is the inner loop underneath the outer, per-round "global iteration"
    loop (agent_runtime.federation.run_federated_experiment): local epochs
    are local retries of one task, not more tasks or another server round.

    client.py stays a plain interface here too: a non-completed round is
    just retried with the same prompt, in the same workspace (so a
    file-producing task can build on or overwrite what the previous local
    round left, matching "retry", not "start over") — this function does not
    itself decide how to adapt between attempts. If a harness wants richer
    retry behavior (revising its own approach using what it left in the
    workspace, or what its own reasoning traces said), that belongs inside
    client_agent's own evolvable code, which already runs across multiple
    rounds within a single solve_task call and can read back its own prior
    work from the shared workspace.

    Only the final local round's AgentOutcome is returned. Earlier rounds'
    outputs and reasoning traces are otherwise discarded here — the caller
    (and everything downstream, including whatever eventually gets uploaded
    to the server) only ever sees the one outcome this function returns.
    """
    epochs = max(1, int(local_epochs))
    app_config = config or load_config(config_path)
    shared_model = model or ModelClient(app_config.model)
    outcome: Optional[AgentOutcome] = None
    attempts = 0
    consumed: list[Usage] = []
    for local_round in range(1, epochs + 1):
        attempts = local_round
        outcome = await solve_task(
            task_prompt,
            insight_library,
            runtime_context,
            task_id=task_id,
            agent_version=agent_version,
            config=app_config,
            config_path=config_path,
            prompt_file=prompt_file,
            model=shared_model,
            tools=tools,
        )
        consumed.append(outcome.usage)
        # Local epochs are retries, not a requirement to overwrite a task that
        # already completed. The old fixed-count loop multiplied cost and often
        # replaced good artifacts with a worse later attempt.
        if outcome.status == "completed":
            break
    assert outcome is not None  # epochs >= 1 guarantees at least one solve_task call above
    outcome.usage = Usage(
        model_calls=sum(item.model_calls for item in consumed),
        input_tokens=sum(item.input_tokens for item in consumed),
        output_tokens=sum(item.output_tokens for item in consumed),
        elapsed_seconds=sum(item.elapsed_seconds for item in consumed),
        peak_memory_bytes=max((item.peak_memory_bytes for item in consumed), default=0),
        memory_delta_bytes=max((item.memory_delta_bytes for item in consumed), default=0),
    )
    outcome.telemetry = {
        **outcome.telemetry,
        "local_epochs_requested": epochs,
        "local_epochs_attempted": attempts,
        "local_retry_stopped_on_completion": outcome.status == "completed" and attempts < epochs,
        "local_total_model_calls": outcome.usage.model_calls,
    }
    return outcome


def solve_task_with_local_epochs_sync(*args: Any, **kwargs: Any) -> AgentOutcome:
    return asyncio.run(solve_task_with_local_epochs(*args, **kwargs))


class HarnessClient:
    """Temporary adapter for benchmark loaders while they migrate to solve_task."""

    def __init__(
        self,
        client_id: str,
        base_harness: Any = None,
        *,
        workspace_root: str = "foh_runs",
        residual_dir: str = "residuals",
        agent_cfg: Optional[Dict[str, Any]] = None,
        agent_version: str | Path = DEFAULT_CLIENT_AGENT,
        config_path: str | Path = REPO_ROOT / "config.yaml",
        prompt_file: Optional[str] = None,
    ) -> None:
        self.client_id = client_id
        self.base_harness = dict(base_harness) if isinstance(base_harness, dict) else {}
        self.workspace_root = Path(workspace_root)
        self.residual_dir = Path(residual_dir)
        self.agent_cfg = dict(agent_cfg or {})
        self.version_store_dir = Path(
            self.agent_cfg.get("version_store_dir", self.workspace_root)
        )
        self.agent_version = Path(agent_version)
        self.config_path = Path(self.agent_cfg.get("config_path", config_path))
        self.prompt_file = self.agent_cfg.get("prompt_file", prompt_file)
        # None (not {}): defer to solve_task's own default of whatever
        # library is embedded in the resolved agent_version, unless receive()
        # is called with an explicit override.
        self.library: Optional[Dict[str, str]] = None
        self.residual_dir.mkdir(parents=True, exist_ok=True)
        VersionStore(self.version_store_dir).initialize(DEFAULT_CLIENT_AGENT, REPO_ROOT / "server_agent")

    def receive(self, guide: Any = None, library: Any = None) -> None:
        if isinstance(library, dict):
            self.library = {str(k): str(v) for k, v in library.items()}

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(problem.get("id", "task"))
        workspace_hint = problem.get("local_workspace") or problem.get("local_tmp_workspace")
        workspace = (
            Path(str(workspace_hint)).expanduser().resolve()
            if workspace_hint
            else self.workspace_root / self.client_id / re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        )
        private_metadata_keys = {
            "prompt", "answer", "solution", "expected_behavior", "grading_criteria",
            "automated_checks", "llm_judge_rubric", "reference_answer",
        }
        public_metadata = {
            k: v
            for k, v in problem.items()
            if k not in private_metadata_keys and not k.endswith("_task")
        }
        context = RuntimeContext(
            workspace=workspace,
            attachments=[str(v) for v in problem.get("attachments", [])],
            skills=[
                str(v)
                for v in list(self.base_harness.get("skills") or []) + list(problem.get("skills") or [])
            ],
            metadata={
                **public_metadata,
                "benchmark_instructions": str(self.base_harness.get("system_prompt", "")),
            },
        )
        app_config = load_config(self.config_path)
        if self.agent_cfg.get("run_timeout"):
            app_config = replace(
                app_config,
                limits=replace(
                    app_config.limits,
                    max_elapsed_seconds=max(1, int(self.agent_cfg["run_timeout"])),
                ),
            )
        selected_version = self.agent_version
        store = VersionStore(self.version_store_dir)
        if store.manifest_path.is_file():
            selected_version = Path(store.load()["client_agent"])
        outcome = solve_task_sync(
            str(problem.get("prompt", "")),
            self.library,
            context,
            task_id=task_id,
            agent_version=selected_version,
            config=app_config,
            config_path=self.config_path,
            prompt_file=self.prompt_file,
        )
        residual = {
            "client_id": self.client_id,
            "reflection": outcome.reflection.to_dict() if outcome.reflection else None,
            "reasoning_traces": [trace.to_dict() for trace in outcome.reasoning_traces],
            "agent_version": load_agent(selected_version)[0].digest,
        }
        residual_path = self.residual_dir / f"problem_{re.sub(r'[^A-Za-z0-9_.-]+', '_', self.client_id)}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', task_id)}.json"
        residual_path.write_text(json.dumps(residual, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_execution = execution_for_benchmark(
            outcome.raw_execution,
            solution=outcome.solution,
            status=outcome.status,
            workspace=outcome.workspace,
        )
        return {
            "problem_id": task_id,
            "solution": outcome.solution,
            "reasoning_traces": residual["reasoning_traces"],
            "actual": {"agent_result": raw_execution},
            "compare": {"reasoning_traces": residual["reasoning_traces"]},
            "residual_path": str(residual_path),
            "workspace": outcome.workspace,
            "outcome": outcome.to_dict(),
        }


def main() -> None:
    """The command-line hook onto client_agent: run one real task through
    whichever version of the harness you point it at and get its answer
    back, without going through the benchmark batch machinery in
    agent_runtime.federation. See also ask_client_agent.sh, a thin bash
    wrapper around this for one-line use.
    """
    parser = argparse.ArgumentParser(description="Run one autonomous client agent task")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-id", default="task")
    parser.add_argument(
        "--library", default=None,
        help="Optional insight_library.json to override what's embedded in --agent-version.",
    )
    parser.add_argument("--workspace", default="foh_runs/client")
    parser.add_argument(
        "--agent-version", default=None,
        help=(
            "client_agent directory to run (its insight_library.json travels with it "
            f"unless --library overrides it). Defaults to the checked-in seed ({DEFAULT_CLIENT_AGENT}), "
            "or to --run-dir's currently active version if that is given instead."
        ),
    )
    parser.add_argument(
        "--run-dir", default=None,
        help=(
            "An experiment output directory (agent_versions/manifest.json underneath it) "
            "to resolve the CURRENTLY ACTIVE client_agent version from, instead of the "
            "checked-in seed -- e.g. the same --out-dir/--output-dir a real run used. "
            "Ignored if --agent-version is also given."
        ),
    )
    parser.add_argument(
        "--harness-repo", default=None,
        help=(
            "Git repository the server publishes evolved harnesses to (config.yaml "
            "harness_git.repo). Pulls --harness-branch into --harness-dest and runs its "
            "client_agent. Overrides --agent-version and --run-dir."
        ),
    )
    parser.add_argument(
        "--harness-branch", default=None,
        help="Branch to pull from --harness-repo (default: the latest iteration-NNN branch).",
    )
    parser.add_argument(
        "--harness-dest", default="foh_runs/harness",
        help="Local clone directory for --harness-repo (default: foh_runs/harness).",
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="Print only the final answer text, not the full JSON outcome (usage, traces, telemetry).",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    add_prompt_file_argument(parser)
    args = parser.parse_args()
    library: Optional[Dict[str, str]] = None
    if args.library:
        value = json.loads(Path(args.library).read_text(encoding="utf-8"))
        if isinstance(value, dict):
            library = {str(k): str(v) for k, v in value.items()}
    if args.harness_repo:
        from agent_runtime.harness_git import latest_branch, pull_iteration

        branch = args.harness_branch or latest_branch(args.harness_repo)
        if branch is None:
            raise SystemExit(f"no iteration branch found on {args.harness_repo}")
        pulled = pull_iteration(args.harness_repo, branch, args.harness_dest)
        print(f"[harness_git] using {branch} from {args.harness_repo} -> {pulled}", file=sys.stderr)
        agent_version = pulled / "client_agent"
    elif args.agent_version:
        agent_version = args.agent_version
    elif args.run_dir:
        agent_version = VersionStore(args.run_dir).load()["client_agent"]
    else:
        agent_version = DEFAULT_CLIENT_AGENT
    result = solve_task_sync(
        args.task,
        library,
        RuntimeContext(workspace=Path(args.workspace)),
        task_id=args.task_id,
        agent_version=agent_version,
        config_path=args.config,
        prompt_file=args.prompt_file,
    )
    if args.plain:
        print(result.solution)
    else:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
