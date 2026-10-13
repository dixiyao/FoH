"""Compatibility helpers for benchmark transcript consumers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _content_blocks(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        blocks: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and item.get("type"):
                blocks.append(dict(item))
            elif item is not None:
                blocks.append({"type": "text", "text": str(item)})
        return blocks
    if value is None:
        return []
    return [{"type": "text", "text": str(value)}]


def canonical_transcript(
    transcript: Iterable[Dict[str, Any]] | None,
    *,
    fallback_assistant_text: str = "",
) -> List[Dict[str, Any]]:
    """Return the message-event shape expected by PinchBench/OpenClaw graders.

    FoH's internal outcome is intentionally benchmark-neutral, while the
    checked-in PinchBench graders consume transcript events shaped as
    ``{"type":"message", "message":{"role":..., "content":[...]}}``.
    Preserve already-canonical events and upgrade the legacy flat
    ``{"role":..., "content":...}`` adapter representation.
    """
    events: List[Dict[str, Any]] = []
    for raw in transcript or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "message" and isinstance(raw.get("message"), dict):
            event = dict(raw)
            message = dict(raw["message"])
            message["content"] = _content_blocks(message.get("content"))
            event["message"] = message
            events.append(event)
            continue
        role = raw.get("role")
        if role:
            events.append(
                {
                    "type": "message",
                    "message": {
                        "role": str(role),
                        "content": _content_blocks(raw.get("content")),
                    },
                }
            )
            continue
        events.append(dict(raw))
    if not events and fallback_assistant_text:
        events.append(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(fallback_assistant_text)}],
                },
            }
        )
    return events


def execution_for_benchmark(
    raw_execution: Dict[str, Any] | None,
    *,
    solution: str,
    status: str,
    workspace: str | None = None,
) -> Dict[str, Any]:
    """Build a grading payload without discarding a supplied raw execution."""
    execution = dict(raw_execution or {})
    execution["status"] = status
    execution["transcript"] = canonical_transcript(
        execution.get("transcript"), fallback_assistant_text=solution
    )
    execution.setdefault("final_answer", solution)
    if workspace is not None:
        execution["workspace"] = workspace
    return execution
