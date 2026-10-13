"""Minimal autonomous action loop used by the checked-in seed agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from .contracts import AgentContext, AgentOutcome, ReasoningTrace
from .parsing import first_json_object, render_json
from .privacy import sanitize_text
from .runtime import AgentRuntime


_MAX_CONSECUTIVE_FORMAT_FAILURES = 2


def _skill_summaries(paths: List[str]) -> List[Dict[str, str]]:
    summaries: List[Dict[str, str]] = []
    for raw in paths[:30]:
        path = Path(raw).expanduser()
        if path.is_dir():
            path = path / "SKILL.md"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:12_000]
        except Exception:
            text = ""
        summaries.append({"path": str(raw), "content": text})
    return summaries


def _feedback(value: Any) -> List[ReasoningTrace]:
    if not isinstance(value, list):
        value = [value] if value else []
    return [ReasoningTrace.from_value(item) for item in value]


def _safe_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(key): _safe_metadata(item, depth + 1)
            for key, item in value.items()
            if not any(label in str(key).lower() for label in ("api_key", "token", "password", "secret", "credential"))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item, depth + 1) for item in list(value)[:100]]
    if isinstance(value, str):
        return sanitize_text(value, max_chars=20_000)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return sanitize_text(str(value), max_chars=2000)


def _format_failure(
    response_text: str,
    runtime: AgentRuntime,
    context: AgentContext,
    finish_kind: Literal["solution", "insight_library"],
    reason: str,
) -> AgentOutcome:
    """Stop immediately on malformed model output and make the fallback auditable."""
    return AgentOutcome(
        solution=response_text if finish_kind == "solution" else "{}",
        reasoning_traces=[
            ReasoningTrace(
                stage="output_format",
                action="parsed the agent response against the required action schema",
                observation=reason,
                outcome="format_failure",
                reusable_feedback="Return exactly one schema-valid JSON action and include the required finish payload.",
            )
        ],
        status="failed",
        usage=runtime.usage(),
        raw_execution=context.runtime_context.raw_execution,
        telemetry={**runtime.telemetry(), "format_failure": True, "format_failure_reason": reason},
    )


def _format_fallback(
    response_text: str,
    runtime: AgentRuntime,
    context: AgentContext,
    reason: str,
) -> AgentOutcome:
    """Preserve a final plain-text client answer after one schema repair failed.

    Official graders can still inspect the answer and workspace. Treating it
    as an automatic zero discarded otherwise usable single-agent work in the
    cluster runs.
    """
    return AgentOutcome(
        solution=response_text,
        reasoning_traces=[
            ReasoningTrace(
                stage="output_format",
                action="retried the required action schema and preserved the final client response",
                observation=reason,
                outcome="format_fallback",
                reusable_feedback="Use a schema-valid finish action, while preserving completed work if formatting recovery is needed.",
            )
        ],
        status="completed",
        usage=runtime.usage(),
        raw_execution=context.runtime_context.raw_execution,
        telemetry={
            **runtime.telemetry(),
            "format_fallback": True,
            "format_failure_reason": reason,
        },
    )


def _round_limit_fallback(
    response_text: str,
    runtime: AgentRuntime,
    context: AgentContext,
) -> AgentOutcome:
    """Keep completed workspace work gradeable when the controller omits finish."""
    return AgentOutcome(
        solution=response_text,
        reasoning_traces=[
            ReasoningTrace(
                stage="termination",
                action="preserved completed work after the controller action limit",
                observation="the agent did not emit an explicit finish action",
                outcome="round_limit_fallback",
                reusable_feedback="Emit the finish action as soon as requested work is complete.",
            )
        ],
        status="completed",
        usage=runtime.usage(),
        raw_execution=context.runtime_context.raw_execution,
        telemetry={**runtime.telemetry(), "round_limit_fallback": True},
    )


def _normalize_action(
    response_text: str,
    available_names: set[str],
    finish_kind: Literal["solution", "insight_library"],
) -> Dict[str, Any] | None:
    """Accept the common equivalent action shapes emitted by hosted models."""
    action = first_json_object(response_text)
    if not isinstance(action, dict):
        return None

    # Some models wrap the requested object one level too deeply.
    nested = action.get("action")
    if isinstance(nested, dict):
        action = dict(nested)

    name = str(action.get("action", "")).strip()
    if not name and isinstance(action.get("tool"), str):
        name = "tool"
        action = {**action, "action": "tool"}
    if finish_kind == "solution":
        solution_keys = ("solution", "final_answer", "answer", "response", "output")
        selected_key = next((key for key in solution_keys if key in action), None)
        if selected_key is not None and (not name or name in {"finish", "final", "complete", "done"}):
            return {
                **action,
                "action": "finish",
                "solution": action.get(selected_key, ""),
            }
    if not name and finish_kind == "insight_library" and action:
        return {"action": "finish", "insight_library": action, "feedback": []}

    direct_tool = ""
    if name in available_names:
        direct_tool = name
    elif name.startswith("tool:") and name[5:] in available_names:
        direct_tool = name[5:]
    if direct_tool:
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {
                key: value
                for key, value in action.items()
                if key not in {"action", "tool", "arguments"}
            }
        return {"action": "tool", "tool": direct_tool, "arguments": arguments}
    return action


async def run_autonomous_loop(
    context: AgentContext,
    runtime: AgentRuntime,
    *,
    finish_kind: Literal["solution", "insight_library"],
) -> AgentOutcome:
    """Let the model select tools and collaboration structure at runtime."""
    history: List[Dict[str, Any]] = []
    last_text = ""
    insight_file_hint = (
        "\nFor insight_library specifically: you may write the library directly "
        "to insight_library.json in your workspace with the write_file/edit_file "
        "tools (a normal artifact you build up over the run, just like any other "
        "file) instead of assembling the whole dict inline in the finish action -- "
        "if that file exists when you finish, its content is used as the library "
        "and the inline field becomes optional."
        if finish_kind == "insight_library" else ""
    )
    internal_instruction = f"""You control an autonomous agent through JSON actions.
Choose the workflow from the objective; do not spawn agents unless delegation is useful.
You may use tools, start background agents, communicate, wait for results, verify work,
and adapt after failures. You may think out loud and plan in plain text before the
JSON action if that helps on a long or multi-file task -- write it as prose ahead of
the action, and use write_file for your own scratch notes, a design doc, or a running
plan in the workspace when a task spans many rounds; that working memory is yours to
create and use. One JSON action per response is still required.{insight_file_hint}

Actions:
{{"action":"tool","tool":"name","arguments":{{}}}}
{{"action":"spawn_agent","role":"specialized role","objective":"bounded delegated objective","workspace_mode":"shared or isolated"}}
{{"action":"list_agents"}}
{{"action":"wait_agent","agent_id":"..."}}
{{"action":"send_message","agent_id":"...","message":"..."}}
{{"action":"receive_messages"}}
{{"action":"finish","{finish_kind}":{('{"insight_name":"description"}' if finish_kind == 'insight_library' else '"completed answer"')},"feedback":[{{"stage":"...","action":"summary","observation":"summary","outcome":"...","reusable_feedback":"..."}}],"artifacts":["relative/path"]}}

Feedback must be a concise reusable execution summary. Never put the task answer,
secrets, personal data, task identifiers, or copied task text in feedback -- summarize
your own reasoning rather than pasting it verbatim, since feedback is what crosses into
shared, aggregated evidence other rounds will read.
The action field is the action TYPE, not a tool name. To call read_file, for example,
use {{"action":"tool","tool":"read_file","arguments":{{"path":"..."}}}}.
For long deliverables, write a manageable first chunk and use append_file for later
chunks so a single JSON response is never truncated. After verification, reserve a
turn for the finish action.
The state includes rounds_remaining. When it is 1, do not call another tool:
emit the finish action immediately so completed work is not discarded.
When rounds_remaining is 4 or less, stop exploratory work, create the required
deliverable immediately, perform at most one essential verification, and finish.
"""

    consecutive_format_failures = 0
    for round_index in range(1, runtime.limits.max_rounds + 1):
        inbox = await runtime.receive_messages(context)
        rounds_remaining = runtime.limits.max_rounds - round_index + 1
        prompt_state = {
            "role": context.role,
            "objective": context.objective,
            "is_subagent": context.is_subagent,
            "depth": context.depth,
            "insight_library": context.insight_library,
            "workspace": str(context.runtime_context.workspace),
            "attachments": context.runtime_context.attachments,
            "skills": _skill_summaries(context.runtime_context.skills),
            "runtime_metadata": _safe_metadata(context.runtime_context.metadata),
            "available_actions": runtime.action_catalog(context),
            "new_messages": inbox,
            # Full history, not a fixed-size tail -- see the matching
            # comment in agent_runtime.evolution's optimizer loop for why: a
            # hardcoded slice discards real working memory regardless of
            # whether there was room for it. render_json's own size-based
            # bounding (the max_chars override below) still keeps this from
            # growing without limit.
            "recent_history": history,
            "round": round_index,
            "max_rounds": runtime.limits.max_rounds,
            "rounds_remaining": rounds_remaining,
            "must_finish_now": rounds_remaining == 1,
        }
        try:
            response = await runtime.generate(
                "Current agent state:\n" + render_json(prompt_state, max_chars=200_000),
                system=internal_instruction,
            )
        except TimeoutError as exc:
            reason = str(exc) or "Model call timed out"
            history.append(
                {
                    "action": "model_timeout",
                    "observation": {
                        "error": reason,
                        "required": "Retry concisely and continue from completed workspace work.",
                    },
                }
            )
            if "elapsed-time budget" not in reason and rounds_remaining > 1:
                continue
            if finish_kind == "solution" and (
                last_text.strip() or runtime.telemetry().get("tool_calls", 0)
            ):
                return _round_limit_fallback(
                    last_text or "Task work was completed in the workspace.", runtime, context
                )
            raise
        last_text = response.text
        action_names = {
            str(item.get("name", ""))
            for item in runtime.action_catalog(context)
            if item.get("name")
        }
        controller_actions = {
            "spawn_agent", "list_agents", "wait_agent", "send_message", "receive_messages", "finish"
        }
        available_names = action_names - controller_actions
        action = _normalize_action(response.text, available_names, finish_kind)
        if action is None:
            consecutive_format_failures += 1
            reason = "The model response did not contain a complete JSON action."
            history.append(
                {
                    "action": "format_repair",
                    "observation": {
                        "error": reason,
                        "finish_reason": response.finish_reason,
                        "response_excerpt": response.text[-2000:],
                        "required": "Return exactly one complete JSON action using an available action name.",
                    },
                }
            )
            if finish_kind == "solution" and rounds_remaining > 1:
                continue
            if consecutive_format_failures < _MAX_CONSECUTIVE_FORMAT_FAILURES:
                continue
            if finish_kind == "solution" and response.text.strip():
                return _format_fallback(response.text, runtime, context, reason)
            return _format_failure(response.text, runtime, context, finish_kind, reason)

        action_name = str(action.get("action", "")).strip()
        try:
            if action_name == "tool":
                tool_name = str(action.get("tool", ""))
                observation = await runtime.call_tool(
                    context, tool_name, action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
                )
                history.append({"action": f"tool:{tool_name}", "observation": observation})
            elif action_name == "spawn_agent":
                agent_id = await runtime.spawn_agent(
                    context,
                    role=str(action.get("role", "background specialist")),
                    objective=str(action.get("objective", "Assist with the assigned objective.")),
                    workspace_mode=str(action.get("workspace_mode", "isolated")),
                )
                history.append({"action": "spawn_agent", "observation": {"agent_id": agent_id}})
            elif action_name == "list_agents":
                history.append({"action": "list_agents", "observation": runtime.list_agents()})
            elif action_name == "wait_agent":
                agent_id = str(action.get("agent_id", ""))
                child = await runtime.wait_agent(context, agent_id)
                history.append(
                    {
                        "action": "wait_agent",
                        "observation": {
                            "agent_id": agent_id,
                            "status": child.status,
                            "solution": child.solution,
                            "artifacts": child.artifacts,
                        },
                    }
                )
            elif action_name == "send_message":
                await runtime.send_message(
                    context,
                    str(action.get("agent_id", "")),
                    str(action.get("message", "")),
                )
                history.append({"action": "send_message", "observation": "message delivered"})
            elif action_name == "receive_messages":
                history.append(
                    {"action": "receive_messages", "observation": await runtime.receive_messages(context)}
                )
            elif action_name == "finish":
                value = action.get(finish_kind)
                if finish_kind == "insight_library":
                    # An explicit finish payload is authoritative.  The
                    # candidate workspace is normally pre-seeded with the
                    # previous round's insight_library.json, so merely finding
                    # that file does *not* mean the agent wrote a new library.
                    # Preferring it here used to silently replace a fresh
                    # inline proposal with stale seed data on every ordinary
                    # server completion.  The file is only a fallback when the
                    # finish action omitted a usable inline dictionary.
                    if not isinstance(value, dict):
                        library_path = context.runtime_context.workspace / "insight_library.json"
                        if library_path.is_file():
                            try:
                                value = json.loads(library_path.read_text(encoding="utf-8"))
                            except Exception as exc:
                                history.append(
                                    {
                                        "action": "format_repair",
                                        "observation": {"error": f"insight_library.json was not valid JSON: {exc}"},
                                    }
                                )
                    if not isinstance(value, dict):
                        consecutive_format_failures += 1
                        reason = (
                            "The server finish action omitted a dictionary insight_library, "
                            "inline or via a written insight_library.json."
                        )
                        history.append({"action": "format_repair", "observation": {"error": reason}})
                        if consecutive_format_failures < _MAX_CONSECUTIVE_FORMAT_FAILURES:
                            continue
                        return _format_failure(response.text, runtime, context, finish_kind, reason)
                    solution = json.dumps(value, ensure_ascii=False)
                else:
                    if value is None and action.get("solution") is None:
                        consecutive_format_failures += 1
                        reason = "The client finish action omitted its solution."
                        history.append({"action": "format_repair", "observation": {"error": reason}})
                        if consecutive_format_failures < _MAX_CONSECUTIVE_FORMAT_FAILURES:
                            continue
                        return _format_failure(response.text, runtime, context, finish_kind, reason)
                    solution = str(value if value is not None else action.get("solution", ""))
                traces = _feedback(action.get("feedback"))
                if not traces:
                    traces = [
                        ReasoningTrace(
                            stage="completion",
                            action="completed the selected workflow",
                            observation="runtime checks reached a finish decision",
                            outcome="completed",
                            reusable_feedback="Choose delegation and tools only when their expected value exceeds their coordination cost.",
                        )
                    ]
                return AgentOutcome(
                    solution=solution,
                    reasoning_traces=traces,
                    status="completed",
                    usage=runtime.usage(),
                    artifacts=[str(item) for item in action.get("artifacts", []) if isinstance(item, str)],
                    raw_execution=context.runtime_context.raw_execution,
                    telemetry=runtime.telemetry(),
                )
            else:
                consecutive_format_failures += 1
                reason = f"The model returned an unsupported action: {action_name or '[missing]'}."
                history.append(
                    {
                        "action": "format_repair",
                        "observation": {
                            "error": reason,
                            "available_actions": sorted(action_names | {"finish"}),
                        },
                    }
                )
                if finish_kind == "solution" and rounds_remaining > 1:
                    continue
                if consecutive_format_failures >= _MAX_CONSECUTIVE_FORMAT_FAILURES:
                    if finish_kind == "solution" and response.text.strip():
                        return _format_fallback(response.text, runtime, context, reason)
                    return _format_failure(response.text, runtime, context, finish_kind, reason)
        except Exception as exc:
            history.append(
                {
                    "action": action_name or "unknown",
                    "observation": {"error_type": type(exc).__name__, "error": str(exc)[:2000]},
                }
            )
        else:
            if action_name != "finish" and action_name in {
                "tool", "spawn_agent", "list_agents", "wait_agent", "send_message", "receive_messages"
            }:
                consecutive_format_failures = 0

    if finish_kind == "solution" and (
        last_text.strip() or runtime.telemetry().get("tool_calls", 0)
    ):
        return _round_limit_fallback(
            last_text or "Task work was completed in the workspace.", runtime, context
        )
    return AgentOutcome(
        solution=last_text if finish_kind == "solution" else "{}",
        reasoning_traces=[
            ReasoningTrace(
                stage="termination",
                action="used the autonomous workflow budget",
                observation="the agent did not produce a valid finish action",
                outcome="failed",
                reusable_feedback="Reserve enough budget for explicit verification and a schema-valid finish action.",
            )
        ],
        status="failed",
        usage=runtime.usage(),
        raw_execution=context.runtime_context.raw_execution,
        telemetry=runtime.telemetry(),
    )
