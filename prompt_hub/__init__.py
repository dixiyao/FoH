"""Prompt-bundle loading for autonomous agent evolution."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


PROMPT_HUB_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = PROMPT_HUB_DIR / "general.py"


def resolve_prompt_file(path: Optional[Union[str, Path]] = None) -> Path:
    if path is None:
        return DEFAULT_PROMPT_FILE.resolve()
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    hub_candidate = PROMPT_HUB_DIR / candidate
    if hub_candidate.is_file():
        return hub_candidate.resolve()
    raise FileNotFoundError(
        f"Prompt file not found: {path}. Pass a path or a filename in {PROMPT_HUB_DIR}."
    )


def _load_python_bundle(prompt_path: Path) -> Any:
    """Execute a prompt-bundle module (see general.py's own docstring for
    why this is Python, not JSON) and return its GUIDES dict.

    Safe to exec: this is repo-controlled prompt content, not agent- or
    benchmark-supplied input -- the same trust level as config.yaml or any
    other file a human operator points --prompt-file at, not a new category
    of risk. A unique module name per call avoids colliding with a real
    "general" import elsewhere and avoids reusing a stale sys.modules entry
    if this is ever called twice with edited content in the same process.
    """
    spec = importlib.util.spec_from_file_location(f"prompt_hub._bundle_{prompt_path.stem}", prompt_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{prompt_path}: could not load as a Python prompt-bundle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    guides = getattr(module, "GUIDES", None)
    if guides is None:
        raise ValueError(f"{prompt_path}: Python prompt-bundle modules must define a GUIDES dict")
    return guides


def load_prompt_bundle(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    prompt_path = resolve_prompt_file(path)
    if prompt_path.suffix == ".py":
        data = _load_python_bundle(prompt_path)
    else:
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{prompt_path}: top-level value must be an object")
    # Gs and Gc in the paper's Algorithm 1: the only two human-authored
    # artifacts. The server agent (A_s) is the one entity that reads them —
    # once to revise its own harness (server guide), once immediately after
    # to revise the client harness (client guide) using its now-updated self.
    # The base system prompt used to live here too ("client prompt"/"server
    # prompt"), but that is genuine harness content per the paper's own
    # framing (system prompt is part of H) -- it now ships inside
    # client_agent/server_agent's own SYSTEM_PROMPT.md (see
    # agent_runtime.loader.read_system_prompt) instead of this
    # off-limits-to-evolution bundle.
    expected = {"server guide", "client guide"}
    missing = expected - set(data)
    if missing:
        raise ValueError(f"{prompt_path}: missing prompt fields {sorted(missing)}")
    server_guide = data.get("server guide")
    if not isinstance(server_guide, str) or not server_guide.strip():
        raise ValueError(f"{prompt_path}: 'server guide' must be a non-empty string")
    client_guide = data.get("client guide")
    if not isinstance(client_guide, str) or not client_guide.strip():
        raise ValueError(f"{prompt_path}: 'client guide' must be a non-empty string")
    return data


def load_prompt_state(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return named prompts; compatibility aliases are intentionally minimal."""
    bundle = load_prompt_bundle(path)
    return {
        "server_guide": bundle["server guide"],
        "client_guide": bundle["client guide"],
    }


def add_prompt_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help=(
            "Prompt bundle path or filename in prompt_hub (default: general.py, a Python "
            "module defining a GUIDES dict; a .json file with the same two keys also works)."
        ),
    )


__all__ = [
    "DEFAULT_PROMPT_FILE",
    "PROMPT_HUB_DIR",
    "add_prompt_file_argument",
    "load_prompt_bundle",
    "load_prompt_state",
    "resolve_prompt_file",
]
