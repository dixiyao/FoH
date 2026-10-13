"""Bounded autonomous-agent scheduler and collaboration runtime."""

from __future__ import annotations

import asyncio
import resource
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .config import HermesSettings, RuntimeLimits
from .contracts import AgentContext, AgentOutcome, ModelResponse, Usage
from .model import ModelClient
from .privacy import sanitize_text
from .tools import ToolRegistry


AgentRunner = Callable[[AgentContext, "AgentRuntime"], Awaitable[AgentOutcome]]


class AgentBudgetExhausted(RuntimeError):
    """A configured runtime budget (tokens, model calls, tool calls,
    recursion depth, total sub-agents) ran out mid-task.

    Deliberately a distinct type from plain RuntimeError: callers (see
    client.solve_task) catch this specifically and turn it into a graceful
    per-task failure instead of letting it crash the whole batch — one task
    running long shouldn't take down every other task's results with it.
    Errors indicating an actual bug (e.g. "no agent implementation bound")
    stay as plain RuntimeError so they keep propagating loudly.
    """


def _peak_process_memory_bytes() -> int:
    """Return the process high-water RSS in bytes using only the standard library."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and other common Unix platforms report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _generic_role(value: str) -> str:
    text = str(value).lower()
    for label in ("verifier", "reviewer", "researcher", "analyst", "planner", "coder", "writer", "specialist"):
        if label in text:
            return label
    return "background_agent"


@dataclass
class _AgentRecord:
    agent_id: str
    parent_id: Optional[str]
    role: str
    depth: int
    status: str = "running"
    outcome: Optional[AgentOutcome] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task[AgentOutcome]] = None
    inbox: asyncio.Queue[Dict[str, str]] = field(default_factory=asyncio.Queue)


@dataclass
class _SharedState:
    started_at: float
    started_peak_memory_bytes: int
    model_semaphore: asyncio.Semaphore
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    records: Dict[str, _AgentRecord] = field(default_factory=dict)
    inboxes: Dict[str, asyncio.Queue[Dict[str, str]]] = field(default_factory=dict)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)


class AgentRuntime:
    """Capabilities passed to an evolvable agent implementation.

    Agent code can choose its workflow, but cannot replace the model client,
    budget accounting, collaboration scheduler, or workspace tool boundary.
    """

    _global_gate_lock = threading.Lock()
    _global_gates: Dict[int, threading.BoundedSemaphore] = {}

    def __init__(
        self,
        *,
        model: ModelClient,
        limits: RuntimeLimits,
        tools: ToolRegistry,
        base_system_prompt: str,
        runner: Optional[AgentRunner] = None,
        shared_state: Optional[_SharedState] = None,
        hermes_settings: Optional[HermesSettings] = None,
    ) -> None:
        self.model = model
        self.limits = limits
        self.tools = tools
        self.base_system_prompt = base_system_prompt.strip()
        self.hermes_settings = hermes_settings or HermesSettings(enabled=False)
        self._runner = runner
        self._state = shared_state or _SharedState(
            started_at=time.monotonic(),
            started_peak_memory_bytes=_peak_process_memory_bytes(),
            model_semaphore=asyncio.Semaphore(limits.max_parallel_agents),
        )

    def _global_gate(self) -> threading.BoundedSemaphore:
        capacity = self.limits.max_global_model_concurrency
        with self._global_gate_lock:
            return self._global_gates.setdefault(capacity, threading.BoundedSemaphore(capacity))

    def bind_runner(self, runner: AgentRunner) -> None:
        self._runner = runner

    def _check_time(self) -> None:
        if time.monotonic() - self._state.started_at > self.limits.max_elapsed_seconds:
            raise TimeoutError("Agent runtime elapsed-time budget exhausted")

    async def generate(self, prompt: str, *, system: str = "") -> ModelResponse:
        self._check_time()
        estimated_input = max(1, (len(prompt) + len(system) + len(self.base_system_prompt)) // 4)
        async with self._state.lock:
            if self._state.model_calls >= self.limits.max_model_calls:
                raise AgentBudgetExhausted("Agent runtime model-call budget exhausted")
            if self._state.input_tokens + self._state.output_tokens + estimated_input > self.limits.max_total_tokens:
                raise AgentBudgetExhausted("Agent runtime token budget exhausted")
            self._state.model_calls += 1
            self._state.input_tokens += estimated_input
        combined_system = self.base_system_prompt
        if system.strip():
            combined_system = f"{combined_system}\n\n{system.strip()}".strip()
        gate = self._global_gate()
        while not gate.acquire(blocking=False):
            self._check_time()
            await asyncio.sleep(0.01)
        try:
            async with self._state.model_semaphore:
                remaining = self.limits.max_elapsed_seconds - (
                    time.monotonic() - self._state.started_at
                )
                timeout = min(float(self.model.settings.timeout_seconds), remaining)
                if timeout <= 0:
                    raise TimeoutError("Agent runtime elapsed-time budget exhausted")
                remaining_tokens = max(
                    1,
                    self.limits.max_total_tokens
                    - self._state.input_tokens
                    - self._state.output_tokens,
                )
                try:
                    response = await asyncio.wait_for(
                        self.model.generate(
                            prompt,
                            system=combined_system,
                            max_tokens=min(self.model.settings.max_tokens, remaining_tokens),
                            json_mode=True,
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError as exc:
                    # asyncio.TimeoutError and built-in TimeoutError are not
                    # aliases on every supported Python version.
                    raise TimeoutError("Model call exceeded its timeout") from exc
        finally:
            gate.release()
        async with self._state.lock:
            actual_input = response.input_tokens or estimated_input
            actual_output = response.output_tokens or max(1, len(response.text) // 4)
            self._state.input_tokens += actual_input - estimated_input
            self._state.output_tokens += actual_output
        return response

    def action_catalog(self, context: AgentContext) -> List[Dict[str, str]]:
        actions = self.tools.definitions(context.runtime_context.allowed_tools)
        actions.extend(
            [
                {
                    "name": "spawn_agent",
                    "description": "Start a background agent using this same version.",
                    "arguments": '{"role":"...","objective":"...","workspace_mode":"shared|isolated"}',
                },
                {
                    "name": "list_agents",
                    "description": "List this task's agents and statuses.",
                    "arguments": "{}",
                },
                {
                    "name": "wait_agent",
                    "description": "Wait for one background agent and obtain its result.",
                    "arguments": '{"agent_id":"..."}',
                },
                {
                    "name": "send_message",
                    "description": "Send a message to another running agent.",
                    "arguments": '{"agent_id":"...","message":"..."}',
                },
                {
                    "name": "receive_messages",
                    "description": "Read queued messages sent to the current agent.",
                    "arguments": "{}",
                },
            ]
        )
        return actions

    async def call_tool(
        self, context: AgentContext, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        self._check_time()
        async with self._state.lock:
            if self._state.tool_calls >= self.limits.max_tool_calls:
                raise AgentBudgetExhausted("Agent runtime tool-call budget exhausted")
            self._state.tool_calls += 1
        started = time.monotonic()
        try:
            value = await self.tools.call(
                name,
                arguments,
                allowed=context.runtime_context.allowed_tools,
            )
            self._event(
                context,
                "tool",
                {"tool_category": name, "outcome": "completed", "elapsed": time.monotonic() - started},
            )
            return value
        except Exception as exc:
            self._event(
                context,
                "tool",
                {"tool_category": name, "outcome": "failed", "error_type": type(exc).__name__},
            )
            raise

    async def spawn_agent(
        self,
        context: AgentContext,
        *,
        role: str,
        objective: str,
        workspace_mode: str = "isolated",
    ) -> str:
        if self._runner is None:
            raise RuntimeError("No current agent implementation is bound")
        self._check_time()
        if context.depth >= self.limits.max_agent_depth:
            raise AgentBudgetExhausted("Agent recursion-depth budget exhausted")
        async with self._state.lock:
            if len(self._state.records) >= self.limits.max_total_agents:
                raise AgentBudgetExhausted("Total-agent budget exhausted")
            agent_id = f"agent_{uuid.uuid4().hex[:10]}"
            if workspace_mode == "shared":
                workspace = context.runtime_context.workspace
            elif workspace_mode == "isolated":
                workspace = context.runtime_context.workspace / ".agents" / agent_id
                workspace.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError("workspace_mode must be shared or isolated")
            child_context = context.delegated(
                objective=objective,
                role=role,
                agent_id=agent_id,
                workspace=workspace,
            )
            record = _AgentRecord(
                agent_id=agent_id,
                parent_id=context.agent_id,
                role=_generic_role(role),
                depth=child_context.depth,
            )
            self._state.records[agent_id] = record
            self._state.inboxes.setdefault(context.agent_id, asyncio.Queue())
            self._state.inboxes[agent_id] = record.inbox

        child_runtime = AgentRuntime(
            model=self.model,
            limits=self.limits,
            tools=self.tools.clone_for(workspace),
            base_system_prompt=self.base_system_prompt,
            runner=self._runner,
            shared_state=self._state,
            hermes_settings=self.hermes_settings,
        )

        async def _execute() -> AgentOutcome:
            try:
                outcome = AgentOutcome.from_value(
                    await self._runner(child_context, child_runtime)  # type: ignore[misc]
                )
                record.status = outcome.status
                record.outcome = outcome
                return outcome
            except asyncio.CancelledError:
                record.status = "cancelled"
                raise
            except Exception as exc:
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                outcome = AgentOutcome(solution="", status="failed")
                record.outcome = outcome
                return outcome

        record.task = asyncio.create_task(_execute(), name=agent_id)
        self._event(context, "spawn", {"child": agent_id, "role": record.role, "depth": record.depth})
        return agent_id

    async def wait_agent(self, context: AgentContext, agent_id: str) -> AgentOutcome:
        record = self._state.records.get(agent_id)
        if record is None or record.task is None:
            raise KeyError(f"Unknown background agent: {agent_id}")
        remaining = self.limits.max_elapsed_seconds - (
            time.monotonic() - self._state.started_at
        )
        if remaining <= 0:
            await self.cancel_agent(agent_id)
            raise TimeoutError("Agent runtime elapsed-time budget exhausted")
        try:
            outcome = await asyncio.wait_for(asyncio.shield(record.task), timeout=remaining)
        except asyncio.TimeoutError:
            await self.cancel_agent(agent_id)
            raise TimeoutError("Background agent exceeded the task elapsed-time budget")
        self._event(context, "join", {"child": agent_id, "outcome": outcome.status})
        return outcome

    def list_agents(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent_id": record.agent_id,
                "parent_id": record.parent_id,
                "role": record.role,
                "depth": record.depth,
                "status": record.status,
                "error": record.error,
            }
            for record in self._state.records.values()
        ]

    async def send_message(self, context: AgentContext, agent_id: str, message: str) -> None:
        inbox = self._state.inboxes.get(agent_id)
        if inbox is None:
            raise KeyError(f"Unknown agent: {agent_id}")
        await inbox.put(
            {
                "from": context.agent_id,
                "message": sanitize_text(str(message), max_chars=4000),
            }
        )
        self._event(context, "message", {"recipient": agent_id})

    async def receive_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        inbox = self._state.inboxes.setdefault(context.agent_id, asyncio.Queue())
        messages: List[Dict[str, str]] = []
        while True:
            try:
                messages.append(inbox.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    async def cancel_agent(self, agent_id: str) -> None:
        record = self._state.records.get(agent_id)
        if record and record.task and not record.task.done():
            record.task.cancel()
            try:
                await record.task
            except asyncio.CancelledError:
                record.status = "cancelled"

    async def shutdown(self) -> None:
        """Cancel unfinished children when their top-level task has finished."""
        pending = [
            record.agent_id
            for record in self._state.records.values()
            if record.task is not None and not record.task.done()
        ]
        await asyncio.gather(*(self.cancel_agent(agent_id) for agent_id in pending))

    def usage(self) -> Usage:
        peak_memory = _peak_process_memory_bytes()
        return Usage(
            model_calls=self._state.model_calls,
            input_tokens=self._state.input_tokens,
            output_tokens=self._state.output_tokens,
            elapsed_seconds=time.monotonic() - self._state.started_at,
            peak_memory_bytes=peak_memory,
            memory_delta_bytes=max(0, peak_memory - self._state.started_peak_memory_bytes),
        )

    async def record_external_usage(
        self,
        *,
        model_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Account for a subprocess-backed agent at the protected boundary."""
        async with self._state.lock:
            self._state.model_calls += max(0, int(model_calls or 0))
            self._state.input_tokens += max(0, int(input_tokens or 0))
            self._state.output_tokens += max(0, int(output_tokens or 0))

    def telemetry(self) -> Dict[str, Any]:
        return {
            "agent_count": len(self._state.records) + 1,
            "events": list(self._state.events[-500:]),
            "tool_calls": self._state.tool_calls,
            "model_calls": self._state.model_calls,
        }

    def _event(self, context: AgentContext, event: str, detail: Dict[str, Any]) -> None:
        self._state.events.append(
            {
                "event": event,
                "agent_depth": context.depth,
                **detail,
            }
        )
