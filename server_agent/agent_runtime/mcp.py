"""Optional protected bridge to explicitly configured MCP stdio servers."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable

from .tools import ToolRegistry


def register_mcp_servers(registry: ToolRegistry, configurations: Iterable[Dict[str, Any]]) -> None:
    """Register generic MCP tools without exposing server credentials to agents.

    Each configuration accepts ``name``, ``command``, ``args``, and an optional
    ``env`` mapping from child variable name to host environment-variable name.
    """
    for raw in configurations:
        name = str(raw.get("name", ""))
        command = str(raw.get("command", ""))
        arguments = [str(value) for value in raw.get("args", [])]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not command:
            raise ValueError("MCP servers require a safe name and command")
        env_names = raw.get("env") if isinstance(raw.get("env"), dict) else {}
        child_env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"}
        }
        child_env.update({
            str(child_name): os.environ[str(host_name)]
            for child_name, host_name in env_names.items()
            if str(host_name) in os.environ
        })

        async def call_mcp(
            payload: Dict[str, Any],
            *,
            server_command: str = command,
            server_arguments: list[str] = arguments,
            server_env: Dict[str, str] = child_env,
        ) -> Any:
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("The optional 'mcp' package is required for MCP tools") from exc
            parameters = StdioServerParameters(
                command=server_command,
                args=server_arguments,
                env=server_env or None,
            )
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    if str(payload.get("operation", "call")) == "list":
                        result = await session.list_tools()
                    else:
                        result = await session.call_tool(
                            str(payload.get("tool", "")),
                            payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
                        )
            if hasattr(result, "model_dump"):
                return result.model_dump(mode="json")
            return str(result)

        registry.register(
            f"mcp_{name}",
            f"List or call tools on the protected {name} MCP server.",
            '{"operation":"list|call","tool":"tool_name","arguments":{}}',
            call_mcp,
        )
