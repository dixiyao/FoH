# Delegation and composability patterns

## Two separate concerns

"Delegation" here means spawning background agents for real division of labor and synthesizing their results -- a runtime behavior. "Composability" means keeping a harness's OWN code organized as clean, separable units -- a code-organization property. The four researched systems mostly demonstrate the first at the product level and the second at the source-tree level; this file adapts both to what this repo's primitives actually allow, including being explicit about one thing they do NOT allow (see the composability section's last paragraph).

## Delegation: real division of labor, not spawn-and-forget

The runtime primitives are `runtime.spawn_agent(context, role=, objective=, workspace_mode="shared"|"isolated")` (returns an `agent_id`), `runtime.wait_agent(context, agent_id)` (returns an `AgentOutcome`), `runtime.list_agents()`, `runtime.send_message`/`receive_messages`. Budget caps (`max_total_agents`, `max_parallel_agents`, `max_agent_depth`) are already enforced by the runtime -- a harness does not need to reimplement quota tracking, only decide the policy of when delegation is actually worth its coordination cost.

zcode's documented subagent design (vendor-documented, not source-verified -- see `external_references_notes.md`) offers a few concrete, worth-copying policy ideas even though this repo's primitives do not expose all of them as separate knobs:

- A distinct read-only "Explore" role for safe parallel research work, distinct from a role that can write/execute -- this repo does not let a parent restrict a specific child's `allowed_tools` through `spawn_agent` itself (a spawned child inherits the parent's `runtime_context.allowed_tools` unchanged, per `AgentContext.delegated`), so the practical equivalent is a *behavioral* contract stated in the child's `objective` string ("you are in a read-only research role; report findings, do not modify files") rather than a hard permission boundary -- weaker than zcode's own claimed mechanism, and worth being honest about rather than assuming the primitive enforces something it does not.
- Freezing what a child can see/do at spawn time rather than letting it renegotiate scope mid-run -- this repo already does this structurally: `workspace_mode="isolated"` gives a child its own subdirectory (`.agents/<agent_id>/`) so parallel children cannot clobber each other's or the parent's files, which is the same worktree-isolation idea omp.sh implements with real git worktrees (source-verified) for its own subagent fan-out.

Claude Code's own real delegation pattern (first-hand knowledge, not attributed to any of the four researched systems): a sub-agent is given a narrow, bounded objective and works in isolation, but the *parent* is responsible for an explicit synthesis step after collecting results -- reading each child's output, reconciling disagreements, and producing one coherent answer -- rather than concatenating sub-agent outputs and calling that the answer. The synthesis step is real work, not a formality: it is where conflicting findings get resolved and where the parent decides what actually mattered.

### Adapted pseudocode skeleton

```python
# Illustrative -- real division of labor with an explicit synthesis step,
# against runtime.spawn_agent / wait_agent / list_agents.

async def delegate_with_synthesis(context, runtime, subtasks: list[dict]):
    """subtasks: [{"role": "...", "objective": "..."}] -- a real partition of
    the work, not an arbitrary fan-out. Each objective should be independently
    checkable, not merely 'help with the task.'
    """
    agent_ids = []
    for subtask in subtasks:
        agent_id = await runtime.spawn_agent(
            context,
            role=subtask["role"],
            objective=subtask["objective"],  # state the read-only/write contract explicitly here
            workspace_mode="isolated",       # each child gets its own subdirectory
        )
        agent_ids.append(agent_id)

    results = []
    for agent_id in agent_ids:
        outcome = await runtime.wait_agent(context, agent_id)
        results.append({"agent_id": agent_id, "status": outcome.status, "solution": outcome.solution})

    # Explicit synthesis -- real reconciliation, not concatenation.
    synthesis = await runtime.generate(
        "Objective: " + context.objective + "\n\nSub-agent results:\n" +
        "\n\n".join(f"[{r['agent_id']}] ({r['status']}): {r['solution']}" for r in results) +
        "\n\nReconcile these into ONE coherent answer. If results disagree, state which "
        "you trust and why. Note anything a sub-agent reported that turned out to be "
        "irrelevant or wrong, rather than including it uncritically.",
        system=context.objective,
    )
    return synthesis.text
```

Delegate only when the objectives are genuinely separable and each is independently checkable -- coordination (spawning, waiting, synthesizing) has a real cost, and a task that is actually sequential or that needs shared running context does not benefit from being split just because the primitive exists.

## Composability: organize a harness's own code as clean, separable units

deepseek-harness's real architecture (source-verified for the framework it is built on, Cordis) treats every capability as a plugin: a module exporting a name and an `apply(ctx)`, registering tools/services/prompt sections through reversible effects so unloading is automatic and total, with a "seam" abstraction (service/provider/consumer) so swapping one provider cascades correctly to everything that depends on it. zcode's documented Skills/Plugins system and omp.sh's source-verified `capability/` module (`tool.ts`, `skill.ts`, `mcp.ts`, `hook.ts`, `rule.ts`, `extension.ts`) both express a similar idea: a capability is a self-contained, named unit with its own description and implementation, composed into the running agent rather than inlined into one monolithic control loop.

The adaptable idea, translated to plain Python in this repo, does not require (and should not attempt) live plugin mounting:

- Give each real concern its own module -- planning, verification, insight-library curation, delegation policy -- each with a small, clear function surface, the way this project's own top-level guide already asks for ("separate modules for planning, verification, and insight curation read like a real project's own separation of concerns"). This is composability as code organization, and it is fully available to you.
- **This repo's own `skills` mechanism is the closest built-in analogue to a "plugin."** `RuntimeContext.skills` is a list of paths; `run_autonomous_loop`'s own `_skill_summaries` helper already reads each path (a file, or a directory containing `SKILL.md`) and folds its content into the prompt state every round. A self-contained "skill" -- a markdown file describing a specific reusable procedure, with its own scope and usage note -- is a real, working composability unit already wired into the default loop: write one per distinct capability you want to make legible and reusable, rather than folding every procedure's instructions into one system prompt.

**What is NOT available here, and should not be attempted:** deepseek-harness's `cordis_define`/`cordis_run` -- an agent writing JavaScript and mounting it as a genuinely new, model-callable tool/plugin into its own live runtime -- is a kind of self-extension this repo does not support. omp.sh's experimental `autolearn` is a much narrower, different case that should not be conflated with that: per the raw research it only writes bounded, sanitized `SKILL.md` markdown text into an isolated directory, and is explicitly *not* described as creating new tool implementations -- the raw finding is emphatic that this is "bounded memory/skill accretion, not modification of the agent's own source code, weights, or system prompt template." `agent_runtime.evolution`'s own source-policy validation explicitly forbids dynamic execution (`eval`/`exec`/`compile`/`__import__`/`getattr`/`setattr`) and direct references to `runtime.tools`/`ToolRegistry` from ordinary harness code (a "protected-runtime bypass" violation) precisely because that is the mechanism that would let harness-authored code become something other than what static validation read -- this rules out deepseek-harness's live-tool-mounting pattern specifically, not omp-style skill-writing (which this repo's own skills mechanism already supports, see above). A harness in this project composes by organizing its OWN code cleanly and by writing skill markdown files that the existing loop already surfaces -- not by having the running agent register or mount new tool implementations into the live runtime.
