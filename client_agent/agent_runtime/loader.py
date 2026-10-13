"""Load an exact versioned agent implementation without import-name collisions."""

from __future__ import annotations

import difflib
import hashlib
import importlib
import importlib.util
import inspect
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


INSIGHT_LIBRARY_FILENAME = "insight_library.json"
SYSTEM_PROMPT_FILENAME = "SYSTEM_PROMPT.md"
# Directory name a vendored, refreshed copy of this very agent_runtime
# package lives under inside a harness version directory -- see
# refresh_vendored_runtime. Not agent-authored content: excluded from
# digests/diffs the same way insight_library.json is (a byte-for-byte,
# mechanically-refreshed copy showing up as "changed" every round would be
# pure noise, since it always matches the current canonical package).
VENDORED_RUNTIME_DIRNAME = "agent_runtime"
# agent_runtime.model's real (non-mock) provider path does `import utils` --
# a second, separate top-level module at the repo root, outside the
# agent_runtime package itself, that a vendored copy needs too (found live:
# a genuinely standalone run of client_agent otherwise fails with
# ModuleNotFoundError the moment it tries to reach a real model). It has no
# repo-internal dependencies of its own beyond the stdlib and (for the
# openrouter/openai-compatible path) the third-party `requests` package.
VENDORED_UTILS_FILENAME = "utils.py"
_CANONICAL_AGENT_RUNTIME_DIR = Path(__file__).resolve().parent
_CANONICAL_UTILS_FILE = _CANONICAL_AGENT_RUNTIME_DIR.parent / VENDORED_UTILS_FILENAME


@dataclass(frozen=True)
class AgentVersion:
    directory: Path
    digest: str


def _tracked_files(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix not in {".pyc", ".pyo"}
        and "__pycache__" not in path.parts
        and path.name != INSIGHT_LIBRARY_FILENAME
        and VENDORED_RUNTIME_DIRNAME not in path.relative_to(root).parts[:1]
        and path.relative_to(root) != Path(VENDORED_UTILS_FILENAME)
    }


def refresh_vendored_runtime(directory: str | Path) -> None:
    """Copy the canonical, protected agent_runtime package (and utils.py --
    see VENDORED_UTILS_FILENAME) alongside a harness.

    Makes the FoH role overlay portable without the rest of this repository:
    copy the version directory and use its bootstrap entrypoint, which puts
    the directory on sys.path first (see client_agent/run_standalone.py), and
    `import agent_runtime` resolves to this vendored copy. Real remote-model
    execution still requires access to the separately bootstrapped, pinned
    upstream Hermes installation; the immutable kernel is intentionally not
    duplicated inside every overlay directory.

    This has NO effect on how the harness actually runs from within this
    repo/session: agent.py's own `from agent_runtime...` imports keep
    resolving to the ONE real agent_runtime package already on sys.path
    from wherever this process itself was launched (Python caches imports
    by dotted name in sys.modules, and this vendored copy is never itself
    added to sys.path here) -- so there is no class-identity split between
    "the real running AgentContext/AgentOutcome" and a nested copy. This is
    a courtesy copy for a different deployment scenario, not a second copy
    this process ever imports from.

    Always overwrites whatever was already there, so it is safe (and
    intended) to call this both before a revision run -- so the harness
    sees the canonical version if it inspects it -- and again right before
    validation/promotion, so nothing written there during the run, even by
    mistake, survives into what gets activated. See
    agent_runtime.evolution.validate_agent_directory's own exclusion of
    this subtree from every content check: it is refreshed infrastructure,
    not agent-authored content, the same role insight_library.json's own
    pre-seeding plays for data instead of code.
    """
    root = Path(directory).expanduser().resolve()
    target = root / VENDORED_RUNTIME_DIRNAME
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        _CANONICAL_AGENT_RUNTIME_DIR,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # See VENDORED_UTILS_FILENAME's own comment: agent_runtime.model's real
    # provider path reaches for this second, separate top-level module.
    if _CANONICAL_UTILS_FILE.is_file():
        (root / VENDORED_UTILS_FILENAME).write_bytes(_CANONICAL_UTILS_FILE.read_bytes())


def read_embedded_library(agent_version: str | Path) -> Dict[str, str]:
    """Read the insight library shipped inside an agent version directory.

    The library ships as plain data alongside the agent's own code, not as a
    side-channel argument the caller must separately track — a version
    directory is meant to be one self-contained, directly-runnable unit of
    (code, current library). It is intentionally excluded from
    directory_digest/directory_unified_diff: the library legitimately
    advances every round regardless of whether that round's code revision is
    accepted, and must not appear to change a version's code identity or
    show up as code churn in a revision diff.
    """
    path = Path(agent_version).expanduser().resolve() / INSIGHT_LIBRARY_FILENAME
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(key): str(item) for key, item in value.items()} if isinstance(value, dict) else {}


def write_embedded_library(agent_version: str | Path, library: Dict[str, str]) -> None:
    root = Path(agent_version).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / INSIGHT_LIBRARY_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_system_prompt(agent_version: str | Path, *, default: str) -> str:
    """Read the base system prompt shipped inside an agent version directory.

    Unlike the insight library, this is genuine harness content (see the
    paper's framing of the system prompt as part of H) -- it is NOT excluded
    from directory_digest/directory_unified_diff, so editing it counts as a
    real code-identity change like any other file. `default` only matters
    for a directory that predates this file (there should be none once the
    checked-in seeds carry SYSTEM_PROMPT.md, but a missing file must never
    hard-crash a real run over something this recoverable).
    """
    path = Path(agent_version).expanduser().resolve() / SYSTEM_PROMPT_FILENAME
    if not path.is_file():
        return default
    text = path.read_text(encoding="utf-8").strip()
    return text or default


def directory_digest(directory: str | Path) -> str:
    root = Path(directory).expanduser().resolve()
    digest = hashlib.sha256()
    for relative, path in _tracked_files(root).items():
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def directory_unified_diff(
    previous_directory: str | Path,
    candidate_directory: str | Path,
    *,
    previous_label: str = "previous",
    candidate_label: str = "candidate",
) -> str:
    """Render a human-readable unified diff between two agent directory trees.

    Covers added, removed, and modified files so a reviewer can see exactly
    what a code-evolution round changed, independent of the digest used for
    activation and rollback decisions.
    """
    previous_files = _tracked_files(Path(previous_directory).expanduser().resolve())
    candidate_files = _tracked_files(Path(candidate_directory).expanduser().resolve())
    chunks: list[str] = []
    for relative in sorted(set(previous_files) | set(candidate_files)):
        before_path = previous_files.get(relative)
        after_path = candidate_files.get(relative)
        before_text = before_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if before_path else []
        after_text = after_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if after_path else []
        if before_text == after_text:
            continue
        chunks.append(
            "".join(
                difflib.unified_diff(
                    before_text,
                    after_text,
                    fromfile=f"{previous_label}/{relative}" if before_path else "/dev/null",
                    tofile=f"{candidate_label}/{relative}" if after_path else "/dev/null",
                )
            )
        )
    if not chunks:
        return f"# No file differences between {previous_label} and {candidate_label}\n"
    return "".join(chunks)


def load_agent(directory: str | Path) -> tuple[AgentVersion, Any]:
    root = Path(directory).expanduser().resolve()
    init_path = root / "__init__.py"
    agent_path = root / "agent.py"
    if not init_path.is_file() or not agent_path.is_file():
        raise FileNotFoundError(f"Agent directory must contain __init__.py and agent.py: {root}")
    digest = directory_digest(root)
    # Code identity alone is not a safe import-cache key.  Two version
    # directories can intentionally have identical code digests while
    # carrying different role-local resources or living in different task
    # workspaces.  Reusing the first package in that case makes relative
    # imports and ``__file__`` resolve against the wrong version directory.
    # Include the canonical path while retaining the digest for auditability.
    path_digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    package_name = f"_foh_agent_{digest[:16]}_{path_digest}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    agent_module = importlib.import_module(f"{package_name}.agent")
    runner = getattr(agent_module, "run", None)
    if runner is None or not inspect.iscoroutinefunction(runner):
        raise TypeError(f"{agent_path} must define async def run(context, runtime)")
    signature = inspect.signature(runner)
    if len(signature.parameters) != 2:
        raise TypeError(f"Agent run entrypoint must accept exactly (context, runtime): {signature}")
    return AgentVersion(root, digest), runner
