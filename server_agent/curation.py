"""Pure, mechanical curation of a proposed insight library.

No model call and no judgment about which entries are more general than
others — that call belongs to the aggregation objective itself. This only
enforces a hard ceiling on entry count and removes near-duplicate entries by
crude word overlap, so a library that would otherwise only ever grow keeps a
bound regardless of how closely the model follows the aggregation
instructions on any given round.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import List, Tuple

from agent_runtime.contracts import AgentOutcome
from agent_runtime.parsing import first_json_object

_MAX_ENTRIES = 20
_DUPLICATE_OVERLAP_THRESHOLD = 0.6


def _tokens(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = sum((a & b).values())
    total = sum((a | b).values())
    return shared / total if total else 0.0


def curate(outcome: AgentOutcome) -> AgentOutcome:
    parsed = first_json_object(outcome.solution)
    if not isinstance(parsed, dict) or not parsed:
        return outcome
    kept: List[Tuple[str, str]] = []
    kept_tokens: List[Counter] = []
    dropped_duplicates = 0
    for key, value in parsed.items():
        tokens = _tokens(f"{key} {value}")
        if any(_overlap(tokens, existing) > _DUPLICATE_OVERLAP_THRESHOLD for existing in kept_tokens):
            dropped_duplicates += 1
            continue
        kept.append((str(key), str(value)))
        kept_tokens.append(tokens)
    dropped_overflow = max(0, len(kept) - _MAX_ENTRIES)
    kept = kept[:_MAX_ENTRIES]
    if dropped_duplicates or dropped_overflow:
        outcome.solution = json.dumps(dict(kept), ensure_ascii=False)
        outcome.telemetry = {
            **outcome.telemetry,
            "curation_dropped_duplicates": dropped_duplicates,
            "curation_dropped_overflow": dropped_overflow,
        }
    return outcome
