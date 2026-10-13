"""Per-task MCP bridge: exposes one Claw-Eval task's own tools over MCP.

task_claweval.py cannot inject a custom ToolRegistry into FoH's shared
run_federated_experiment/_run_batch pipeline (there is no per-case hook for
that) — but agent_runtime.mcp.register_mcp_servers already reads
RuntimeContext.metadata["mcp_servers"] and spawns a stdio MCP server per
configured entry. This script IS that server. External Hermes keeps one stdio
process alive across the task's tool calls; the compatibility FoH MCP client
may reconnect. Either way every call appends its corresponding
claw_eval.models.trace.ToolDispatch record to --dispatch-log as one JSON line.
task_claweval.py reads that file after the client finishes to reconstruct the
trace Claw-Eval's graders expect.

This script is Claw-Eval-specific infrastructure, not agent-evolvable code —
it lives in tasks/ (this repo's harness layer), not client_agent/server_agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any


def _add_claw_eval_to_path(bench_dir: Path) -> None:
    src = str(bench_dir / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


async def _run(task_dir: Path, bench_dir: Path, dispatch_log: Path, port_offset: int) -> None:
    _add_claw_eval_to_path(bench_dir)
    from claw_eval.models.content import ToolUseBlock
    from claw_eval.models.task import TaskDefinition
    from claw_eval.runner.dispatcher import ToolDispatcher
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    task = TaskDefinition.from_yaml(task_dir / "task.yaml")
    # This subprocess is spawned fresh per tool call (see module docstring)
    # and re-loads task.yaml from scratch each time, so it has no way to
    # know what port offset task_claweval.py's ServiceManager applied to ITS
    # own in-memory TaskDefinition unless told explicitly -- apply the same
    # offset here so this dispatcher's endpoint URLs actually match the
    # ports the real services are listening on.
    task.apply_port_offset(port_offset)
    dispatcher = ToolDispatcher(task.get_endpoint_map())

    async def on_list_tools(ctx: Any, params: Any) -> "types.ListToolsResult":
        return types.ListToolsResult(
            tools=[
                types.Tool(name=spec.name, description=spec.description, inputSchema=spec.input_schema)
                for spec in task.tools
            ]
        )

    async def on_call_tool(ctx: Any, params: "types.CallToolRequestParams") -> "types.CallToolResult":
        tool_use = ToolUseBlock(
            id=f"mcpcall_{uuid.uuid4().hex[:12]}",
            name=str(params.name),
            input=dict(params.arguments or {}),
        )
        result_block, dispatch_event = dispatcher.dispatch(tool_use, trace_id=task.task_id)
        with dispatch_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dispatch_event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        text = "\n".join(block.text for block in result_block.content)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            isError=result_block.is_error,
        )

    server = Server("claweval", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        dispatcher.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--dispatch-log", required=True)
    parser.add_argument("--port-offset", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(_run(Path(args.task_dir), Path(args.bench_dir), Path(args.dispatch_log), args.port_offset))


if __name__ == "__main__":
    main()
