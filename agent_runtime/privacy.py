"""Deterministic privacy controls for federated traces and libraries."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from .contracts import ClientReflection, ReasoningTrace


_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.I),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
]


def sanitize_text(
    value: Any,
    *,
    forbidden_values: Sequence[str] = (),
    max_chars: int = 1200,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    for forbidden in forbidden_values:
        forbidden = re.sub(r"\s+", " ", str(forbidden or "")).strip()
        if len(forbidden) >= 4:
            text = re.sub(re.escape(forbidden), "[REDACTED]", text, flags=re.I)
    # Strip long copied spans from a task/solution while retaining short,
    # reusable procedural language.
    for forbidden in forbidden_values:
        words = re.findall(r"\S+", str(forbidden or ""))
        for start in range(0, max(0, len(words) - 11), 8):
            phrase = " ".join(words[start : start + 12])
            if len(phrase) >= 48:
                text = re.sub(re.escape(phrase), "[REDACTED]", text, flags=re.I)
    return text[:max_chars]


def sanitize_traces(
    traces: Iterable[Any],
    *,
    forbidden_values: Sequence[str] = (),
    max_chars: int = 1200,
    max_items: int = 2000,
) -> List[ReasoningTrace]:
    clean: List[ReasoningTrace] = []
    for index, raw in enumerate(traces):
        if index >= max_items:
            break
        trace = ReasoningTrace.from_value(raw)
        item = ReasoningTrace(
            stage=sanitize_text(trace.stage, forbidden_values=forbidden_values, max_chars=80),
            action=sanitize_text(trace.action, forbidden_values=forbidden_values, max_chars=240),
            observation=sanitize_text(trace.observation, forbidden_values=forbidden_values, max_chars=240),
            outcome=sanitize_text(trace.outcome, forbidden_values=forbidden_values, max_chars=120),
            reusable_feedback=sanitize_text(
                trace.reusable_feedback,
                forbidden_values=forbidden_values,
                max_chars=max_chars,
            ),
        )
        if any((item.action, item.observation, item.reusable_feedback)):
            clean.append(item)
    return clean


def sanitize_library(
    library: Any,
    *,
    forbidden_values: Sequence[str] = (),
    max_entries: int = 500,
    max_chars: int = 1600,
) -> Dict[str, str]:
    if not isinstance(library, dict):
        return {}
    clean: Dict[str, str] = {}
    for index, (key, value) in enumerate(library.items()):
        if index >= max_entries:
            break
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))[:100].strip("_")
        safe_value = sanitize_text(
            value, forbidden_values=forbidden_values, max_chars=max_chars
        )
        if safe_key and safe_value:
            clean[safe_key] = safe_value
    return clean


def sanitize_reflection(
    reflection: Any,
    *,
    forbidden_values: Sequence[str] = (),
    max_chars: int = 1200,
) -> ClientReflection:
    """Apply the fixed privacy boundary to the dedicated client reflection."""
    value = ClientReflection.from_value(reflection)

    def clean_many(items: Iterable[Any], *, item_chars: int = 300) -> List[str]:
        clean: List[str] = []
        for item in list(items)[:30]:
            text = sanitize_text(
                item,
                forbidden_values=forbidden_values,
                max_chars=item_chars,
            )
            if text:
                clean.append(text)
        return clean

    return ClientReflection(
        summary=sanitize_text(
            value.summary,
            forbidden_values=forbidden_values,
            max_chars=max_chars,
        ),
        actions=clean_many(value.actions),
        observations=clean_many(value.observations),
        reusable_insights=clean_many(value.reusable_insights, item_chars=max_chars),
        outcome=sanitize_text(value.outcome, max_chars=120),
    )


def contains_sensitive_text(value: str) -> bool:
    return any(pattern.search(value or "") for pattern in _SECRET_PATTERNS)
