# Self-evolution patterns

## Where this fits in this project already

Unlike the other files in this directory, this one is about a mechanism this project already implements for you at the outer level: `agent_runtime.evolution.evolve_agent` copies your current harness directory to a candidate, loads and runs your OWN current `agent.py` against that candidate (this IS the paper's Algorithm 1 point -- there is no separate, permanently-fixed optimizer prompt standing between a harness and its own evolution; whatever your `run(context, runtime)` currently does is what performs the revision), diffs the result, runs `run_validation`-equivalent checks, and activates the candidate only if it changed and validated. `VersionStore` keeps every version archived with diffs and traces, so rollback and audit already exist. **None of that outer machinery is yours to rebuild.** What IS yours to design is what happens *inside* your own `run(context, runtime)` during a self-revision round: what you read about your own past, how you decide what actually needs to change, and how you verify before you call it done. That is what this file is about.

## What hermes-agent's self-evolution actually looks like (verified, and narrower than the name suggests)

hermes-agent (NousResearch/hermes-agent) markets itself as "self-evolving," and the mechanism is real but text-scoped, not code-scoped, in its always-on form:

- Two markdown artifact types change automatically: factual memory (`MEMORY.md`/`USER.md`, via a `memory` tool with add/replace/remove) and procedural skills (`SKILL.md`, via a `skill_manage` tool with create/patch/edit/delete). Triggers are (a) the agent's own in-context judgment ("this workflow is worth repeating," "the user corrected my approach") and (b) an automatic post-turn "background review" pass, often on a cheaper auxiliary model, that re-reads the transcript and proposes edits.
- Every such write can be forced into a pending queue requiring explicit human accept/reject before it takes effect, and hub-sourced skills pass a security scanner before being trusted.
- Memory files are loaded as a **frozen snapshot at session start** -- edits during a session hit disk immediately but do not re-enter the prompt until the next session, protecting the cached prefix from being invalidated mid-session.
- The base system prompt, the core agent code, and tool policy are explicitly NOT touched by this layer -- the project's own docs state this plainly.
- A separate, actually-built-but-narrow layer exists in a sister repo (`hermes-agent-self-evolution`): an offline, explicitly-invoked (not scheduled) DSPy+GEPA pipeline that mutates skill text against a held-out eval set using a composite fitness function (LLM-judge quality score, minus a length penalty near size caps, plus a semantic-similarity-to-original constraint so the mutation cannot drift arbitrarily far from the original intent), gated by a full test-pass requirement, and always landing as a human-reviewed pull request -- never an automatic deployment. Only this "skill text" phase is implemented; phases that would touch tool descriptions, fixed system-prompt sections, or actual tool-implementation code are roadmap only.

The honest takeaway: call this "bounded, gated text accretion plus an occasional offline search over that same text," not "an agent that rewrites itself." The adaptable ideas are the *shape* of the gating, not any specific line of code.

## Adaptable ideas, concretely

1. **Split hot-path accretion from cold-path revision.** Cheap, immediate edits (this round's insight-library entry, this round's note-to-self) are a different kind of operation from a deliberate, evidence-gated rewrite of your own control flow or prompt. Both should exist, but they should not be the same code path with the same scrutiny -- a library entry earns its place from one well-stated generalization; a control-flow rewrite earns its place from a specific, named failure pattern across the reflections you actually read this round.
2. **Frozen-snapshot loading, live mutation.** If you keep a durable note about your own past performance, load it once per round rather than re-reading it after every edit you make to it within the same round -- the point is that what you reasoned from is a stable, inspectable snapshot, not a moving target you partially mutated mid-thought.
3. **Fitness/acceptance criteria that resist self-flattery.** When you judge whether a candidate change to your own harness is actually better, do not use a single "quality" self-score. Check it against something concrete: does `python -m agent_runtime.validate_candidate --role <target_role> .` still pass, did the change address a *specific* reflection or gap you can name, did it avoid ballooning a file/prompt without bound. A self-revision that cannot point to a specific reason is unaccountable churn that a concrete acceptance check like this should catch on its own terms; a self-revision that always says "no change needed" is the opposite failure mode, and that is the one this project's guide explicitly warns against ("reserve leaving something unchanged for when you can point to a specific reason a further change would make it worse... not for when a change would simply take more effort, and not by default").
4. **Sanitize anything self-authored before it becomes trusted context.** Your `insight_library.json` (or any skill/notes file you maintain) is content you write now and read again as if-trusted context in a later round, or that a client harness reads as if-trusted context. Treat it the way omp.sh's sandboxed `managed-skills` mechanism treats agent-authored skill files (an experimental feature in that project, explicitly flagged as such in its own source, but the discipline is sound regardless): keep it a plain, bounded, inspectable artifact -- no embedded directives phrased as instructions to a future reader, no unbounded growth, entries earn their place individually rather than being appended wholesale. You are the one at risk of prompt-injecting your own future self through your own artifact.
5. **Make the revision itself reversible in spirit, even though the reversibility mechanism already exists for you.** `evolve_agent`/`VersionStore` will reject a candidate that fails validation or never changed anything measurable, and archive it either way. Do not treat "I will just try something, the outer validator will catch it" as license to skip verifying yourself first -- catching a broken candidate at the outer level means the round produced nothing, which is a worse outcome than not attempting a risky change without first checking it.

## Adapted pseudocode skeleton (against this repo's own primitives)

This is illustrative of what your own `run(context, runtime)` can do internally during a revision round -- not a loop you are required to implement verbatim, and not something that replaces `evolve_agent`'s own outer bookkeeping (candidate copy, digest diff, stagnation retries, activation) which already happens around whatever this does.

```python
# Sketch only -- expressed against agent_runtime.runtime.AgentRuntime and the
# built-in tools (list_files, read_file, write_file, edit_file, run_command),
# not against hermes-agent's or any other external system's own API.

async def self_revision_round(context, runtime):
    # 1. Read your own past outcomes. evolve_agent already folds this
    #    round's collected reflections into context.objective -- but you can
    #    ALSO keep your own durable cross-round note as an ordinary workspace
    #    file, the way hermes-agent freezes a memory snapshot at the start of
    #    a unit of work rather than re-deriving it from scratch every time.
    try:
        past_notes = await runtime.call_tool(context, "read_file", {"path": "self_notes/last_round.md"})
    except Exception:
        past_notes = "(no prior self-notes -- first round or first time keeping them)"
    library_raw = await runtime.call_tool(context, "read_file", {"path": "insight_library.json"})

    # 2. Decide what changed -- ask for NAMED gaps, not a general "improve
    #    it". This is the step that keeps a revision honest: a gap you can
    #    point to in the actual reflection text is a real reason to change
    #    something; "it could probably be better" is not.
    analysis = await runtime.generate(
        "Past self-notes:\n" + past_notes + "\n\nCurrent insight library:\n" + library_raw +
        "\n\nFrom the reflections and context already in your objective, name the "
        "SPECIFIC gaps they point to -- not a general 'make it better'. For each: "
        "is it a prompt issue, a control-flow issue, a missing verification step, "
        "or a missing insight-library entry? If you genuinely find none, say so "
        "and state the specific reason a change would make things worse, not "
        "merely that a change would take effort.",
        system=context.objective,
    )
    await runtime.call_tool(context, "write_file", {"path": "self_notes/analysis.md", "content": analysis.text})

    # 3. Write the new version -- ordinary file edits in the workspace you
    #    were handed (already a candidate copy of your own prior harness).
    #    A real pass touches several files; this shows one representative
    #    edit, not the whole rewrite.
    current_agent_py = await runtime.call_tool(context, "read_file", {"path": "agent.py"})
    revision = await runtime.generate(
        "Current agent.py:\n" + current_agent_py + "\n\nNamed gaps:\n" + analysis.text +
        "\n\nProduce the complete next agent.py addressing them, preserving what "
        "still works.",
        system=context.objective,
    )
    await runtime.call_tool(context, "write_file", {"path": "agent.py", "content": revision.text})

    # 4. Verify before finishing. External Hermes does not receive FoH's
    #    in-process run_validation registry entry, so invoke the protected CLI
    #    from this candidate workspace (replace client_agent when revising the
    #    server role).
    validation = await runtime.call_tool(context, "run_command", {
        "command": "python -m agent_runtime.validate_candidate --role client_agent ."
    })
    if not validation.get("passed"):
        failed = [c for c in validation.get("checks", []) if c.get("result") == "failed"]
        # fix the specific failure and re-check, or, if truly stuck, leave a
        # clear account in your final trace of what failed and why -- don't
        # finish silently on top of a broken candidate.
        ...

    # 5. Leave a durable note for step 1 of the NEXT round, and return a
    #    real prose account -- not a status flag -- of what changed and why.
    await runtime.call_tool(context, "write_file", {
        "path": "self_notes/last_round.md",
        "content": "This round's analysis:\n" + analysis.text[:2000] + "\n\nChanged: agent.py (see version diff).",
    })
    return {
        "solution": revision.text,          # or the insight_library payload, depending on finish_kind
        "feedback": [
            {"stage": "self_revision", "action": "analyzed reflections and rewrote agent.py",
             "observation": analysis.text[:500], "outcome": "completed",
             "reusable_feedback": "Name specific gaps before rewriting; run the protected validation CLI before finishing."},
        ],
    }
```

The important properties this preserves: step 2 forces a *specific*, falsifiable account of what needs to change (so "nothing changed this round" has to be a stated judgment, not silence); step 4 checks your own work with the same tool a downstream check would use, instead of discovering failure only after the round is over; step 5 gives the *next* round's step 1 something real to read, so self-revision accumulates understanding across rounds rather than re-deriving it from zero every time.
