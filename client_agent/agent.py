"""Client-agent seed: safe pre/post-processing around the shared task loop.

The server may evolve this implementation in run-local version directories.
The protected runtime, contracts, and budgets remain outside this package.
The task loop itself (run_hermes_agent) is one such protected piece and
cannot be edited from here; what this package can and does add is a thin
wrapper around it, split out into guardrails.py so this entrypoint stays a
short, readable summary of the seed's own structure.
"""

from __future__ import annotations

from agent_runtime.contracts import AgentContext, AgentOutcome
from agent_runtime.hermes import run_hermes_agent
from agent_runtime.runtime import AgentRuntime

from . import guardrails, reflection


async def run(context: AgentContext, runtime: AgentRuntime) -> AgentOutcome:
    context = guardrails.annotate_objective(context)
    outcome = AgentOutcome.from_value(
        await run_hermes_agent(context, runtime, finish_kind="solution")
    )
    return reflection.attach(outcome)
