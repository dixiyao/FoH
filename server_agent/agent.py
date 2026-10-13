"""Server-agent seed: mechanical curation around the shared Hermes loop.

The aggregation loop itself (run_hermes_agent) is a protected module and
cannot be edited here. What this package adds is a small, evidence-free
curation pass in curation.py: a hard size cap and near-duplicate removal on
whatever library the loop proposes, giving the aggregation objective's own
generality instructions a mechanical backstop.
"""

from __future__ import annotations

from agent_runtime.contracts import AgentContext, AgentOutcome
from agent_runtime.hermes import run_hermes_agent
from agent_runtime.runtime import AgentRuntime

from . import curation


async def run(context: AgentContext, runtime: AgentRuntime) -> AgentOutcome:
    # The server authors both harnesses.  Its own revision/aggregation step
    # returns the shared library, while a client-harness revision must finish
    # with an ordinary solution report.  The target lives in fixed runtime
    # metadata because ``context.role`` is descriptive prose and may evolve.
    target_role = context.runtime_context.metadata.get("target_role")
    finish_kind = "solution" if target_role == "client_agent" else "insight_library"
    outcome = AgentOutcome.from_value(
        await run_hermes_agent(context, runtime, finish_kind=finish_kind)
    )
    return outcome if finish_kind == "solution" else curation.curate(outcome)
