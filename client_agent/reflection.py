"""Dedicated client reflection module.

The task answer remains in ``AgentOutcome.solution``.  This module converts
the agent's privacy-safe execution summaries into a typed learning artifact
for the server, guaranteeing that every completed client run has an explicit
reflection even when the model omitted optional feedback.
"""

from __future__ import annotations

from agent_runtime.contracts import AgentOutcome, ClientReflection


def attach(outcome: AgentOutcome) -> AgentOutcome:
    if outcome.reflection is not None and outcome.reflection.has_content():
        return outcome
    traces = list(outcome.reasoning_traces)
    actions = [trace.action for trace in traces if trace.action]
    observations = [trace.observation for trace in traces if trace.observation]
    insights = [trace.reusable_feedback for trace in traces if trace.reusable_feedback]
    if not insights:
        insights = [
            "Inspect available state before acting, recover from tool failures, and verify the requested result before finishing."
        ]
    outcome.reflection = ClientReflection(
        summary=(
            "The task run completed with a reusable execution summary."
            if outcome.status == "completed"
            else "The task run did not complete; preserve the failure mode for the next harness revision."
        ),
        actions=actions[:20],
        observations=observations[:20],
        reusable_insights=insights[:20],
        outcome=outcome.status,
    )
    return outcome
