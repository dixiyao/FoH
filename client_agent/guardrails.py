"""Mechanical preprocessing around the mandatory Hermes task loop.

The task-solving loop itself lives behind
``agent_runtime.hermes.run_hermes_agent`` and cannot be replaced by an
overlay. This module only appends a static objective reminder. In particular
it makes no direct model call: every model/tool turn stays inside the Hermes
hook, so an evolved overlay cannot silently become a second, less-capable
provider loop.
"""

from __future__ import annotations

from dataclasses import replace

from agent_runtime.contracts import AgentContext

_OBJECTIVE_HINT = (
    "Execution reminder: work from the assigned workspace, treat paths not "
    "explicitly supplied with the task as out of scope, use only tools that "
    "are actually available in this run, and verify observable results before "
    "finishing."
)


def annotate_objective(context: AgentContext) -> AgentContext:
    """Prepend a short, static reminder the model will actually read.

    This reminder is transport-neutral: upstream Hermes and the embedded
    test adapter expose different concrete tool names, so claiming that a
    particular tool exists here would make one of those paths systematically
    choose an unavailable action.
    """
    if _OBJECTIVE_HINT in context.objective:
        return context
    return replace(context, objective=f"{context.objective}\n\n{_OBJECTIVE_HINT}")
