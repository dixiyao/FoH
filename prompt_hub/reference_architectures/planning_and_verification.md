# Planning and verification patterns

## Attribution up front

Most of this file draws on first-hand, general knowledge of how real coding-agent harnesses like Claude Code and OpenAI Codex are actually built -- externalized plan/todo state, permission-gated tool use, verify-by-actually-running-things -- not on the four researched external systems, which is called out explicitly rather than misattributed. Two pieces ARE drawn from this round's research and are marked as such below, at two different confidence levels: omp.sh's plan-mode/model-transition gate is a source-verified feature (confirmed by reading its actual repository -- omp.sh's own marketing site returned 403 and was never read); zcode's Goal-Mode act/verify split is vendor/maintainer-documented only, since zcode is closed-source and no independent source verification was possible -- see `external_references_notes.md`.

## Externalize the plan instead of holding it only in one prompt turn

The single most load-bearing habit in Claude-Code-and-Codex-style harnesses is that the plan is a real, persistent artifact the agent reads and updates across rounds, not something reconstructed from scratch in one model call's reasoning each time:

- A running todo/plan list with explicit status per item (not-started / in-progress / done), so "what have I already decided to do and how far did I get" survives a context reset, a sub-agent handoff, or a long multi-round task, instead of depending on the model re-inferring it from raw history every round.
- The plan is created once real understanding of the task exists (after an initial exploration/read pass, not before), and is revised when reality contradicts it -- a discovered blocker becomes a new plan item, not a silent detour that never gets written down.
- Marking a step done happens only after the step's actual result is known, not preemptively -- a plan is a record of what happened, not a wishlist that only ever gets checked off optimistically.

This project's harnesses have no bespoke "todo" API -- the workspace is where persisted state like this lives, via ordinary `write_file`/`read_file`/`edit_file` calls, exactly as the top-level guide already frames it ("a persisted plan or todo state in the workspace that is updated as work proceeds rather than reconstructed from scratch each round").

## Gate execution behind a plan step, and route planning separately when it helps (omp.sh, verified)

omp.sh's (oh-my-pi's) plan-mode subsystem is a real, source-verified feature: a dedicated module gates tool execution behind an explicit *approved* plan object -- the agent cannot act until a plan has been proposed and accepted -- and a separate `model-transition` component can route the planning phase to a different model or role than the one that executes the plan. The adaptable idea, independent of that specific implementation: separating "decide what to do" from "actually do it" as two distinguishable phases (even within a single model/session) makes it possible to check a plan's soundness *before* any side effect happens, rather than discovering a bad plan only after several tool calls already ran against it. It does not require an actual second model call -- even a single harness can require itself to write the plan to a file and treat the next round as "execute against what is already written," rather than re-planning implicitly inside every round's action choice.

## Verify with actual evidence, not a self-report (zcode Goal Mode, vendor-documented)

zcode's documented "Goal Mode" separates each round into an act phase and an independent verify phase: verification is required to produce concrete evidence -- changed files, command output, test results -- rather than accepting the model's own claim that the goal was met, and that verification's findings (not a fresh guess) become the next round's task. This is a genuinely good discipline regardless of whether zcode's own implementation is inspectable (it is not -- treat the specific mechanism as vendor-documented, not source-verified). The adoptable rule: **never let "I believe this is done" be the terminating condition by itself.** If the deliverable is code, run it. If it is a claim about a file's contents, read the file back. If it is a multi-step task, check that every plan item that was marked done actually has evidence behind that mark, not just a status flag flipped by the same reasoning pass that also wrote the deliverable.

## Verify before finishing by actually running what was produced

This is the concrete, repo-native version of the pattern above: before returning a `finish` action, call `run_command` (or the equivalent check for the deliverable's domain) on what was actually produced, and read the real output rather than asserting success. A test suite that "should pass," a script that "should run cleanly," or a file that "should be valid JSON" are all claims that cost one tool call to actually check. If verification fails, the correct response is to iterate on the *specific* failure (fix and re-check) or to note it plainly in the reasoning trace, not to finish on top of an unverified or known-failing result.

## Adapted pseudocode skeleton

```python
# Illustrative -- against agent_runtime.runtime.AgentRuntime and the built-in
# tools (read_file/write_file/edit_file/run_command), inside a harness's own
# run(context, runtime) or a step it takes within run_autonomous_loop-style
# control flow.

PLAN_PATH = "PLAN.md"

async def ensure_plan(context, runtime):
    try:
        plan_text = await runtime.call_tool(context, "read_file", {"path": PLAN_PATH})
    except Exception:
        plan_text = ""
    if plan_text.strip():
        return plan_text
    # Gate: don't act before a plan exists and is written down.
    drafted = await runtime.generate(
        "Objective: " + context.objective + "\n\nDraft a short, concrete PLAN.md as a "
        "markdown checklist of specific steps (- [ ] step). Keep it to what is actually "
        "known now; you will revise it as you learn more.",
        system=context.objective,
    )
    await runtime.call_tool(context, "write_file", {"path": PLAN_PATH, "content": drafted.text})
    return drafted.text

async def mark_step_done(context, runtime, step_text: str, evidence: str):
    plan_text = await runtime.call_tool(context, "read_file", {"path": PLAN_PATH})
    # Only mark done once real evidence exists -- see the zcode-style rule above.
    updated_line_old = "- [ ] " + step_text
    updated_line_new = "- [x] " + step_text + f"  (evidence: {evidence[:120]})"
    if updated_line_old in plan_text:
        await runtime.call_tool(context, "edit_file", {
            "path": PLAN_PATH, "old": updated_line_old, "new": updated_line_new,
        })

async def verify_before_finish(context, runtime, check_command: str):
    result = await runtime.call_tool(context, "run_command", {"command": check_command})
    passed = result.get("exit_code") == 0
    if not passed:
        # Iterate on the SPECIFIC failure -- feed the real output back in,
        # don't re-ask the model to just try again with no new information.
        diagnosis = await runtime.generate(
            "The verification command failed:\n" + result.get("output", "")[-4000:] +
            "\n\nDiagnose the specific cause and describe the fix.",
            system=context.objective,
        )
        await runtime.call_tool(context, "edit_file", {
            "path": PLAN_PATH,
            "old": "- [ ] verify",
            "new": "- [ ] fix: " + diagnosis.text[:200],
        })
    return passed, result

# Usage shape within a round:
#   plan = await ensure_plan(context, runtime)
#   ... take the next unchecked plan step, do the work, then:
#   await mark_step_done(context, runtime, step_text, evidence="ran X, output Y")
#   ... once every step is checked off:
#   ok, result = await verify_before_finish(context, runtime, "python -m pytest -q")
#   if ok: return {"solution": ..., "feedback": [...]}
#   # else: loop again next round rather than finishing on top of a known failure.
```

The properties this preserves: the plan is a real file that persists and is inspectable independent of any one model call's reasoning; a step is only marked done alongside stated evidence, not by assertion; and the terminating condition for the whole task is a real command's real exit code, not the model's own claim that things are fine.
