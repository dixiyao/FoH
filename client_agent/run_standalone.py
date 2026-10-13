#!/usr/bin/env python3
"""Run this client role overlay as a standalone program.

The overlay directory carries its protected FoH runtime copy, system prompt,
helpers, and insight library.  Upstream Hermes Agent is intentionally a
separate, shared, immutable kernel rather than another copy inside every
overlay.  Bootstrap the pinned kernel with FoH's scripts/bootstrap_hermes.py,
then run, for example:

    python run_standalone.py "your task or question" \
      --hermes-install-root /opt/foh/.hermes

The install root must contain hermes-agent/, venv/, and install.json.  The
protected bridge verifies install.json against FoH's pinned commit and fails
closed; a real OpenRouter run never silently falls back to the embedded test
loop.  Use --hermes-source and --hermes-python together when the checkout and
virtualenv are not under one install root.

This entrypoint puts its own directory on sys.path before importing
agent_runtime, so the rest of the FoH repository is not required at runtime.
OPENROUTER_API_KEY is read only from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Tuple


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent_runtime.config import (  # noqa: E402
    HERMES_AGENT_RELEASE_COMMIT,
    HermesSettings,
    ModelSettings,
    RuntimeLimits,
)
from agent_runtime.contracts import AgentContext, RuntimeContext  # noqa: E402
from agent_runtime.loader import (  # noqa: E402
    load_agent,
    read_embedded_library,
    read_system_prompt,
)
from agent_runtime.model import ModelClient  # noqa: E402
from agent_runtime.runtime import AgentRuntime  # noqa: E402
from agent_runtime.tools import ToolRegistry  # noqa: E402


def _add_hermes_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hermes-install-root",
        default=os.getenv("FOH_HERMES_INSTALL_ROOT"),
        help=(
            "Directory created by scripts/bootstrap_hermes.py; may also be set "
            "with FOH_HERMES_INSTALL_ROOT."
        ),
    )
    parser.add_argument(
        "--hermes-source",
        default=os.getenv("FOH_HERMES_SOURCE"),
        help="Explicit pinned hermes-agent checkout; overrides the install-root default.",
    )
    parser.add_argument(
        "--hermes-python",
        default=os.getenv("FOH_HERMES_PYTHON"),
        help="Explicit Hermes virtualenv Python; overrides the install-root default.",
    )


def _absolute_without_resolving_symlinks(value: str | Path) -> Path:
    """Keep a virtualenv launcher path instead of resolving it to base Python."""
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def _resolve_hermes(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Tuple[Path, Path]:
    root = (
        Path(args.hermes_install_root).expanduser().resolve()
        if args.hermes_install_root
        else None
    )
    source = (
        Path(args.hermes_source).expanduser().resolve()
        if args.hermes_source
        else (root / "hermes-agent" if root is not None else None)
    )
    python = (
        _absolute_without_resolving_symlinks(args.hermes_python)
        if args.hermes_python
        else (
            root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if root is not None
            else None
        )
    )
    if source is None or python is None:
        parser.error(
            "real OpenRouter standalone runs require --hermes-install-root "
            "or both --hermes-source and --hermes-python"
        )
    return source, python


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="The task or question to solve.")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    parser.add_argument("--max-rounds", type=int, default=16)
    _add_hermes_arguments(parser)
    args = parser.parse_args()

    api_key_env = "OPENROUTER_API_KEY"
    if not os.getenv(api_key_env):
        raise SystemExit(f"Set {api_key_env} in the environment before running standalone.")
    hermes_source, hermes_python = _resolve_hermes(parser, args)

    _, run = load_agent(HERE)
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = AgentRuntime(
        model=ModelClient(
            ModelSettings(provider="openrouter", name=args.model, api_key_env=api_key_env)
        ),
        limits=RuntimeLimits(max_rounds=max(1, args.max_rounds)),
        tools=ToolRegistry(workspace),
        base_system_prompt=read_system_prompt(
            HERE, default="You are an autonomous problem-solving agent."
        ),
        runner=run,
        hermes_settings=HermesSettings(
            enabled=True,
            source_dir=str(hermes_source),
            python_executable=str(hermes_python),
            revision=HERMES_AGENT_RELEASE_COMMIT,
            max_iterations=max(1, args.max_rounds),
            require_external_for_remote=True,
        ),
    )
    context = AgentContext(
        objective=args.task,
        role="task-solving orchestrator",
        insight_library=read_embedded_library(HERE),
        runtime_context=RuntimeContext(workspace=workspace),
    )
    try:
        outcome = await run(context, runtime)
    finally:
        await runtime.shutdown()
    print(
        json.dumps(
            {"solution": outcome.solution, "status": outcome.status},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
