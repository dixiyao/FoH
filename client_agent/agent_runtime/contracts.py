"""Stable contracts shared by interfaces and evolvable agent code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional


AgentStatus = Literal["completed", "failed", "cancelled", "timeout"]


@dataclass
class Usage:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int = 0
    memory_delta_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResponse:
    text: str
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"


@dataclass
class ReasoningTrace:
    """A privacy-safe execution summary, never raw chain-of-thought."""

    stage: str
    action: str
    observation: str
    outcome: str
    reusable_feedback: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "ReasoningTrace":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(
                stage="execution",
                action="worked on the assigned objective",
                observation="execution summary was supplied",
                outcome="unknown",
                reusable_feedback=value,
            )
        if isinstance(value, dict):
            return cls(
                stage=str(value.get("stage", "execution")),
                action=str(value.get("action", value.get("action_summary", "worked on the objective"))),
                observation=str(value.get("observation", value.get("observation_summary", ""))),
                outcome=str(value.get("outcome", "unknown")),
                reusable_feedback=str(value.get("reusable_feedback", value.get("feedback", ""))),
            )
        return cls("execution", "worked on the objective", "", "unknown", str(value))


@dataclass
class ClientReflection:
    """Typed, privacy-safe learning output kept separate from the answer."""

    summary: str
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    reusable_insights: List[str] = field(default_factory=list)
    outcome: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def has_content(self) -> bool:
        """Whether this reflection contains reusable evidence, not just a type."""
        return any(
            str(item).strip()
            for item in (
                self.summary,
                *self.actions,
                *self.observations,
                *self.reusable_insights,
            )
        )

    @classmethod
    def from_value(cls, value: Any) -> "ClientReflection":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            def strings(name: str) -> List[str]:
                raw = value.get(name) or []
                if isinstance(raw, str):
                    raw = [raw]
                return [str(item) for item in raw] if isinstance(raw, list) else []

            return cls(
                summary=str(value.get("summary", "")),
                actions=strings("actions"),
                observations=strings("observations"),
                reusable_insights=strings("reusable_insights"),
                outcome=str(value.get("outcome", "unknown")),
            )
        return cls(summary=str(value or ""))

    def to_reasoning_trace(self) -> ReasoningTrace:
        return ReasoningTrace(
            stage="client_reflection",
            action="; ".join(self.actions),
            observation="; ".join(self.observations),
            outcome=self.outcome,
            reusable_feedback="; ".join(self.reusable_insights) or self.summary,
        )


@dataclass(frozen=True)
class SessionTurn:
    """One normalized user turn in a benchmark-defined conversation chain."""

    turn_id: str
    prompt: str
    new_session: bool = False

    @classmethod
    def from_value(cls, value: Any, *, index: int = 0) -> "SessionTurn":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            prompt = value
            turn_id = f"turn_{index + 1}"
            new_session = False
        elif isinstance(value, dict):
            raw_prompt = value.get("prompt")
            if not raw_prompt:
                raw_prompt = value.get("message")
            if not isinstance(raw_prompt, str):
                raise ValueError(f"session turn {index + 1} must contain a string prompt")
            prompt = raw_prompt
            turn_id = str(value.get("id") or value.get("turn_id") or f"turn_{index + 1}")
            new_session = bool(value.get("new_session", False))
        else:
            raise ValueError(f"session turn {index + 1} must be a string or mapping")
        if not prompt.strip():
            raise ValueError(f"session turn {index + 1} prompt must not be empty")
        return cls(turn_id=turn_id, prompt=prompt, new_session=new_session)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.turn_id,
            "prompt": self.prompt,
            "new_session": self.new_session,
        }


def normalize_session_turns(values: Iterable[Any]) -> List[SessionTurn]:
    return [SessionTurn.from_value(value, index=index) for index, value in enumerate(values)]


@dataclass
class RuntimeContext:
    """Benchmark-neutral capabilities and metadata available to one agent."""

    workspace: Path
    attachments: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    allowed_tools: Optional[List[str]] = None
    raw_execution: Optional[Dict[str, Any]] = None
    session_turns: List[SessionTurn] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    read_only_paths: List[str] = field(default_factory=list)
    writable_paths: List[str] = field(default_factory=list)
    executable_paths: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.session_turns = normalize_session_turns(self.session_turns)
        self.environment = {
            str(name): str(value) for name, value in self.environment.items()
        }
        self.read_only_paths = [str(Path(value).expanduser().resolve()) for value in self.read_only_paths]
        self.writable_paths = [str(Path(value).expanduser().resolve()) for value in self.writable_paths]
        self.executable_paths = [str(Path(value).expanduser().resolve()) for value in self.executable_paths]


@dataclass
class AgentContext:
    objective: str
    role: str
    insight_library: Dict[str, str]
    runtime_context: RuntimeContext
    task_id: str = "task"
    agent_id: str = "root"
    parent_id: Optional[str] = None
    depth: int = 0
    is_subagent: bool = False
    messages: List[Dict[str, str]] = field(default_factory=list)

    def delegated(
        self,
        *,
        objective: str,
        role: str,
        agent_id: str,
        workspace: Path,
    ) -> "AgentContext":
        runtime_context = RuntimeContext(
            workspace=workspace,
            attachments=list(self.runtime_context.attachments),
            skills=list(self.runtime_context.skills),
            metadata=dict(self.runtime_context.metadata),
            allowed_tools=(
                list(self.runtime_context.allowed_tools)
                if self.runtime_context.allowed_tools is not None
                else None
            ),
            session_turns=list(self.runtime_context.session_turns),
            environment=dict(self.runtime_context.environment),
            read_only_paths=list(self.runtime_context.read_only_paths),
            writable_paths=list(self.runtime_context.writable_paths),
            executable_paths=list(self.runtime_context.executable_paths),
        )
        return AgentContext(
            objective=objective,
            role=role,
            insight_library=dict(self.insight_library),
            runtime_context=runtime_context,
            task_id=self.task_id,
            agent_id=agent_id,
            parent_id=self.agent_id,
            depth=self.depth + 1,
            is_subagent=True,
        )


@dataclass
class AgentOutcome:
    solution: str
    reasoning_traces: List[ReasoningTrace] = field(default_factory=list)
    status: AgentStatus = "completed"
    usage: Usage = field(default_factory=Usage)
    workspace: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    raw_execution: Optional[Dict[str, Any]] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    reflection: Optional[ClientReflection] = None

    def to_dict(self, *, include_raw_execution: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "solution": self.solution,
            "reasoning_traces": [trace.to_dict() for trace in self.reasoning_traces],
            "status": self.status,
            "usage": self.usage.to_dict(),
            "workspace": self.workspace,
            "artifacts": list(self.artifacts),
            "telemetry": dict(self.telemetry),
            "reflection": self.reflection.to_dict() if self.reflection is not None else None,
        }
        if include_raw_execution:
            payload["raw_execution"] = self.raw_execution
        return payload

    @classmethod
    def from_value(cls, value: Any) -> "AgentOutcome":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls(solution=str(value))
        usage_value = value.get("usage")
        usage = Usage(**{k: usage_value.get(k, 0) for k in Usage.__dataclass_fields__}) if isinstance(usage_value, dict) else Usage()
        return cls(
            solution=str(value.get("solution", value.get("final_answer", ""))),
            reasoning_traces=[ReasoningTrace.from_value(v) for v in value.get("reasoning_traces", [])],
            status=str(value.get("status", "completed")),  # type: ignore[arg-type]
            usage=usage,
            workspace=str(value["workspace"]) if value.get("workspace") is not None else None,
            artifacts=[str(v) for v in value.get("artifacts", [])],
            raw_execution=value.get("raw_execution"),
            telemetry=dict(value.get("telemetry") or {}),
            reflection=(
                ClientReflection.from_value(value.get("reflection"))
                if value.get("reflection") is not None
                else None
            ),
        )
