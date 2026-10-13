# Context and memory patterns

## The two problems this file is about

Long or many-round work runs into two distinct problems that get conflated if you are not careful: (1) the raw conversation/action history grows past what is useful or affordable to keep verbatim, and (2) a shared body of accumulated knowledge (an insight library, memory file, or set of skills) grows past what is useful to apply uniformly -- not everything ever learned is relevant to the task in front of you right now. Compaction addresses the first; relevance filtering addresses the second. Both matter more the longer a harness runs or the more rounds of self-evolution accumulate.

## Compact stale history into a durable note, don't rely on truncation alone

`run_autonomous_loop`'s own `recent_history` is the full action history, size-bounded only by `render_json`'s character cap -- a reasonable default, but a harness that wants more deliberate context management on long tasks can do better than "keep everything until it gets cut off." First-hand knowledge of Claude-Code/Codex-style harnesses: when a task runs long, the harness periodically asks itself to summarize the stale portion of history into a compact, durable note (what was tried, what worked, what is still open, what NOT to retry), then continues working from that note plus only the recent tail of raw history -- rather than either (a) keeping everything verbatim forever, which is expensive and eventually gets silently truncated by whatever bound exists, or (b) truncating blindly, which can throw away exactly the one fact that mattered.

omp.sh's real, source-verified `compress` module does this concretely: it compresses long sessions into reusable summaries it calls "mental models," loaded into later parts of the same session rather than the raw transcript. hermes-agent's `context_compressor.py` does a closely related thing via an auxiliary (often cheaper) model when context passes a threshold, and specifically tracks session lineage across compressions (a summary of a summary knows what it was derived from) -- that lineage-tracking detail is documented for hermes-agent's compressor, not confirmed for omp's `compress` module, so it is attributed only to hermes-agent here rather than to both. The adaptable shape, independent of either implementation: compaction should be an explicit, occasional, written act (produces a real artifact you can inspect) triggered by a size/round threshold, not an implicit property of whatever happens to survive a context window.

## Frozen-snapshot memory loading (hermes-agent, verified) -- a cache-freshness tradeoff worth naming explicitly

hermes-agent loads its long-term memory files (`MEMORY.md`, `USER.md`) as a frozen snapshot at session start: edits made during a session hit disk immediately (so a tool that reads the file later in the same session sees the update) but do NOT re-enter the model's own prompt context until the next session, specifically to protect a cached prompt prefix from being invalidated by a mid-session edit. This is a real, deliberate tradeoff, not an oversight: freshness within a round is sacrificed for cache stability and behavioral predictability within that round. If a harness loads an insight library or memory note into its prompt state once per round (as this project's `run_autonomous_loop` does via `context.insight_library`), the same tradeoff applies implicitly -- worth being deliberate about rather than accidentally inconsistent (e.g. silently re-reading a file mid-round and getting different answers depending on when in the round you happen to ask).

## Read the shared insight library selectively, by relevance, not uniformly

The insight library (`insight_library.json`, seeded into every harness's workspace by `evolve_agent`'s `seed_library`) is explicitly meant to be "a small, curated, general set of reusable strategies," per this project's own guide -- not a per-task cookbook and not something every entry of which applies to every task. A harness that dumps the entire library into every prompt verbatim is doing the analogue of "keep everything until truncation" for memory instead of history: cheap to implement, but it either drowns the model in irrelevant entries or gets silently cut off in the same uninformative way raw history does. The better pattern -- consistent with how hermes-agent's own `session_search` is deliberately raw retrieval rather than blanket inclusion, and with real coding-agent harnesses that decide what context is relevant to the current file/task before including it -- is to filter or rank library entries by relevance to the CURRENT objective before deciding what to surface, and to actually do that filtering yourself (a cheap model call or a keyword/category match against the objective) rather than assuming the consuming model will do the filtering silently and reliably every time given the whole list.

## Adapted pseudocode skeleton

```python
# Illustrative -- against agent_runtime.runtime.AgentRuntime and the built-in
# tools, showing both compaction and relevance-filtered library use as
# explicit steps a harness's own control flow can take.

async def compact_if_stale(context, runtime, history: list, *, threshold_rounds: int = 8):
    """Fold everything but a recent tail into a durable note when history
    grows past a threshold, instead of relying only on render_json's
    character-bound truncation to decide what survives.
    """
    if len(history) <= threshold_rounds:
        return history  # nothing stale yet
    stale, recent = history[:-threshold_rounds], history[-threshold_rounds:]
    try:
        existing_note = await runtime.call_tool(context, "read_file", {"path": "context_notes.md"})
    except Exception:
        existing_note = ""
    summary = await runtime.generate(
        "Existing durable note (may be empty):\n" + existing_note +
        "\n\nStale history to fold in:\n" + str(stale)[:20_000] +
        "\n\nProduce an updated, compact note: what was tried, what worked, what is "
        "still open, and anything that should explicitly NOT be retried. Keep it "
        "short enough to re-read cheaply every round.",
        system=context.objective,
    )
    await runtime.call_tool(context, "write_file", {"path": "context_notes.md", "content": summary.text})
    # Continue working from the compact note plus only the recent tail --
    # not from the full stale history any more.
    return recent


async def relevant_library_entries(context, runtime, library: dict, *, max_entries: int = 5) -> dict:
    """Filter a (potentially large) shared insight library down to what is
    actually relevant to the CURRENT objective, instead of surfacing every
    entry uniformly regardless of task.
    """
    if len(library) <= max_entries:
        return library
    ranked = await runtime.generate(
        "Objective: " + context.objective + "\n\nCandidate library entries "
        "(name: description):\n" + "\n".join(f"- {k}: {v}" for k, v in library.items()) +
        f"\n\nReturn the JSON list of at most {max_entries} entry names that are "
        "actually relevant to THIS objective -- omit entries that are generically "
        "true but not specifically useful here.",
        system=context.objective,
    )
    import json
    try:
        names = json.loads(ranked.text)
        return {k: v for k, v in library.items() if k in names}
    except Exception:
        return dict(list(library.items())[:max_entries])  # fall back rather than crash
```

The properties this preserves: compaction produces a real, inspectable artifact (`context_notes.md`) rather than an implicit side effect of a size cap, and is triggered by a stated threshold rather than happening invisibly; library use is filtered by an explicit relevance step before being surfaced, with a safe fallback if that step itself fails, rather than either dumping the whole library or silently trusting the consuming model to ignore what does not apply.
