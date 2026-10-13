"""Small robust parsers for model-produced action objects."""

from __future__ import annotations

import json
from typing import Any, Collection, Dict, Optional


def first_json_object(
    text: str,
    *,
    required_keys: Collection[str] = (),
) -> Optional[Dict[str, Any]]:
    """Decode one balanced object without greedy first/last-brace slicing.

    ``required_keys`` lets schema-aware callers skip unrelated objects such as
    an empty ``{}`` printed before the actual model response. Doubled template
    braces are deliberately not normalized here: they must be rendered to
    literal braces before model invocation, rather than silently rewriting
    arbitrary response text.
    """
    decoder = json.JSONDecoder()
    expected = {str(key) for key in required_keys}
    source = str(text or "").strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines).strip()
    for index, char in enumerate(source):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and expected.issubset(value):
            return value
    return None


def _bounded_value(value: Any, *, string_limit: int, item_limit: int, depth: int = 0) -> Any:
    """Return a JSON-safe, structurally intact bounded copy of ``value``."""
    if depth >= 8:
        return "[TRUNCATED: maximum nesting depth]"
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            str(key): _bounded_value(
                item,
                string_limit=string_limit,
                item_limit=item_limit,
                depth=depth + 1,
            )
            for key, item in items[:item_limit]
        }
        if len(items) > item_limit:
            bounded["__truncated_items__"] = len(items) - item_limit
        return bounded
    if isinstance(value, (list, tuple)):
        items = list(value)
        bounded = [
            _bounded_value(
                item,
                string_limit=string_limit,
                item_limit=item_limit,
                depth=depth + 1,
            )
            for item in items[:item_limit]
        ]
        if len(items) > item_limit:
            bounded.append(f"[TRUNCATED: {len(items) - item_limit} more items]")
        return bounded
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        omitted = len(value) - string_limit
        head = max(1, string_limit * 3 // 4)
        tail = max(0, string_limit - head)
        suffix = value[-tail:] if tail else ""
        return f"{value[:head]}\n[TRUNCATED: {omitted} characters omitted]\n{suffix}"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _bounded_value(
        str(value),
        string_limit=string_limit,
        item_limit=item_limit,
        depth=depth,
    )


def render_json(value: Any, *, max_chars: int = 40_000) -> str:
    """Render bounded JSON without ever slicing it into invalid syntax.

    Agent state frequently contains long tool output. The old implementation
    cut the serialized string at ``max_chars``, leaving the model with a
    half-written JSON object and often removing the newest history and round
    fields entirely. Progressively compact values instead, preserving a valid
    document at every size.
    """
    limit = max(64, int(max_chars))
    attempts = (
        (20_000, 100),
        (8_000, 60),
        (3_000, 40),
        (1_000, 25),
        (300, 15),
        (80, 8),
    )
    for string_limit, item_limit in attempts:
        bounded = _bounded_value(
            value,
            string_limit=string_limit,
            item_limit=item_limit,
        )
        text = json.dumps(bounded, ensure_ascii=False, indent=2, default=str)
        if len(text) <= limit:
            return text
    fallback = json.dumps(
        {"truncated": True, "summary": _bounded_value(str(value), string_limit=32, item_limit=1)},
        ensure_ascii=False,
    )
    if len(fallback) <= limit:
        return fallback
    return "{}"
