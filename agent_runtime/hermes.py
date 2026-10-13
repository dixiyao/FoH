"""FoH boundary around the real, pinned upstream Hermes Agent runtime.

Hermes is process-global by design (provider client, registry, sessions,
memory and tool caches).  Running one isolated subprocess per task/role keeps
parallel benchmark workers and evolving role versions from sharing that
mutable state while preserving upstream's actual ReAct/tool loop.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional
from urllib.parse import urlparse

from .autonomous import run_autonomous_loop
from .config import AppConfig, HERMES_AGENT_RELEASE_COMMIT, HermesSettings, OPERATIONAL_PROVIDERS
from .contracts import AgentContext, AgentOutcome, ClientReflection, ReasoningTrace
from .parsing import first_json_object, render_json
from .privacy import sanitize_library, sanitize_text
from .provider_errors import (
    FatalProviderError, provider_batch_abort, raise_for_gemini_error_text,
    raise_for_provider_error,
)
from .model import ModelClient
from .runtime import AgentRuntime
from .transcripts import canonical_transcript


class HermesBridgeError(RuntimeError):
    """The pinned external Hermes kernel is unavailable or failed."""


_DRIVER = Path(__file__).resolve().with_name("hermes_driver.py")
_SAFE_ENV_NAMES = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
}
_SCOPED_RUNTIME_ENV_NAMES = {
    "GOOGLE_WORKSPACE_CLI_CONFIG_DIR", "GOOGLE_WORKSPACE_CLI_TOKEN",
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "SSL_CERT_FILE", "GH_TOKEN", "GH_REPO", "NO_PROXY", "no_proxy",
    "FOH_CONTAINER_PROXY_URL", "FOH_CONTAINER_PROXY_TOKEN",
}
_SYSTEM_READ_ONLY_PATHS = (
    Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"),
    Path("/etc"),
)
_SCOPED_READ_METADATA_KEYS = (
    "kubeconfig", "scenario_dir", "bench_dir", "repository_path"
)
_MAX_TRANSCRIPT_EVENTS = 200
_PROCESS_TERMINATE_GRACE_SECONDS = 1.0
_GEMINI_RATE_LOCK = threading.Lock()
_GEMINI_RATE_DIRECTORIES: dict[str, Any] = {}


def _gemini_rate_limit_request(temporary: Path) -> dict[str, Any]:
    """Share quota state across this process's isolated Hermes workers.

    Hard links expose only quota counters in each existing scratch mount,
    without granting workers access to another task's workspace.
    """
    budget = int(os.environ.get("FOH_GEMINI_TPM", "0"))
    if budget < 0:
        raise ValueError("FOH_GEMINI_TPM must be nonnegative")
    # Keep shared overload cooldowns active even when proactive TPM pacing
    # is disabled (including direct Python launches).
    parent = str(temporary.parent)
    with _GEMINI_RATE_LOCK:
        directory = _GEMINI_RATE_DIRECTORIES.get(parent)
        if directory is None:
            directory = tempfile.TemporaryDirectory(prefix="foh-gemini-quota-", dir=parent)
            _GEMINI_RATE_DIRECTORIES[parent] = directory
            state = Path(directory.name) / "state.json"
            state.write_text("{}", encoding="utf-8")
            state.chmod(0o600)
        linked_state = temporary / "gemini-quota.json"
        os.link(Path(directory.name) / "state.json", linked_state)
    return {"state_path": str(linked_state), "tokens_per_minute": budget}


async def _communicate_with_provider_abort(process):
    abort = provider_batch_abort.get()
    if abort is None:
        return await process.communicate()
    communication = asyncio.create_task(process.communicate())
    try:
        while True:
            abort.check()
            done, _ = await asyncio.wait({communication}, timeout=0.05)
            if done:
                abort.check()
                return communication.result()
    finally:
        if not communication.done():
            communication.cancel()
        with suppress(asyncio.CancelledError):
            await communication


def _safe_role(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80] or "agent"


def _external_requested(runtime: AgentRuntime) -> bool:
    # An injected responder is a test/control transport and must remain fully
    # in-process.  A real remote provider uses upstream Hermes unless the
    # operator explicitly disables it.
    return bool(
        runtime.hermes_settings.enabled
        and runtime.model.settings.provider != "mock"
        and not runtime.model.has_responder
    )


def _absolute_without_resolving_symlinks(value: Any) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(value or ""))))


def _resolve_external(settings: HermesSettings) -> tuple[Path, Path, Path]:
    source = Path(str(settings.source_dir or "")).expanduser().resolve()
    # Keep the virtualenv executable path itself. Resolving its symlink to a
    # base interpreter bypasses pyvenv.cfg/site-packages and made the freshly
    # bootstrapped Hermes install unusable on uv-managed Python installs.
    python = _absolute_without_resolving_symlinks(settings.python_executable)
    sandbox = _absolute_without_resolving_symlinks(settings.sandbox_executable)
    problems = []
    if not source.is_dir():
        problems.append(f"source checkout not found at {source}")
    if not python.is_file():
        problems.append(f"Hermes Python not found at {python}")
    if not sandbox.is_file() or not os.access(sandbox, os.X_OK):
        problems.append(f"required bubblewrap sandbox not executable at {sandbox}")
    if problems:
        raise HermesBridgeError(
            "; ".join(problems)
            + ". Run `python scripts/bootstrap_hermes.py` before a real OpenRouter run."
        )
    lock_path = source.parent / "install.json"
    if not lock_path.is_file():
        raise HermesBridgeError(
            f"Hermes install lock missing at {lock_path}; rerun scripts/bootstrap_hermes.py"
        )
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HermesBridgeError(f"Hermes install lock is invalid: {exc}") from exc
    installed = str(lock.get("revision") or "")
    if installed != settings.revision or installed != HERMES_AGENT_RELEASE_COMMIT:
        raise HermesBridgeError(
            "Hermes revision mismatch: "
            f"configured={settings.revision}, installed={installed}, required={HERMES_AGENT_RELEASE_COMMIT}"
        )
    return source, python, sandbox


def _validated_credential_environment_name(value: Any) -> str:
    key_env = str(value or "OPENROUTER_API_KEY")
    if (
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env) is None
        or key_env in _SAFE_ENV_NAMES
        or key_env in _SCOPED_RUNTIME_ENV_NAMES
        or key_env.startswith(("HERMES_", "PYTHON"))
    ):
        raise HermesBridgeError(
            f"Unsafe credential environment-variable name: {key_env!r}"
        )
    return key_env


def _validate_injected_test_model(model: Any) -> bool:
    """Recognize only an explicit in-memory responder as a test backend."""
    if model is None:
        return False
    if not isinstance(model, ModelClient) or model.has_responder is not True:
        raise HermesBridgeError(
            "test_model must be a ModelClient with an explicitly injected responder"
        )
    return True


def preflight_external_hermes(
    config: AppConfig,
    *,
    test_model: Any = None,
) -> None:
    """Validate mandatory process-wide Hermes inputs before a paid batch.

    Per-case runtime failures are recoverable batch results, but a missing
    pinned checkout, interpreter, sandbox, install lock, or model credential
    affects every case.  Checking those shared prerequisites once, before any
    worker starts, prevents a broken deployment from being misreported as a
    batch of ordinary task failures.
    """
    if _validate_injected_test_model(test_model):
        return
    if config.model.provider not in OPERATIONAL_PROVIDERS:
        raise HermesBridgeError(
            "Operational FoH runs require model.provider='openrouter' or 'gemini'; no mock "
            "or alternate-provider backend may masquerade as a Hermes experiment. "
            "Tests must pass an in-memory responder through the keyword-only "
            "test_model seam."
        )
    if not config.hermes.enabled:
        raise HermesBridgeError(
            "Operational FoH runs require hermes.enabled=true; the embedded loop "
            "is available only with an explicitly injected test responder"
        )
    if not config.hermes.require_external_for_remote:
        raise HermesBridgeError(
            "Operational FoH runs require hermes.require_external_for_remote=true; "
            "remote-to-embedded fallback is forbidden"
        )
    _resolve_external(config.hermes)
    key_env = _validated_credential_environment_name(config.model.api_key_env)
    if not config.model.api_key():
        raise HermesBridgeError(
            f"Missing remote-model credential: set {key_env}"
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _context_read_only_paths(context: AgentContext) -> list[Path]:
    declared_values = [
        *context.runtime_context.read_only_paths,
        *context.runtime_context.executable_paths,
    ]
    executable_values = {
        str(Path(value).expanduser().resolve())
        for value in context.runtime_context.executable_paths
    }
    raw_values: list[Any] = [
        *context.runtime_context.attachments,
        *declared_values,
    ]
    raw_values.extend(
        context.runtime_context.metadata.get(key)
        for key in _SCOPED_READ_METADATA_KEYS
        if context.runtime_context.metadata.get(key)
    )
    paths: list[Path] = []
    for raw in raw_values:
        candidate = Path(str(raw)).expanduser().resolve()
        if not candidate.exists():
            if raw in declared_values:
                raise HermesBridgeError(f"Declared read-only task path does not exist: {candidate}")
            continue
        if candidate == Path("/"):
            raise HermesBridgeError("Refusing to expose the host filesystem root as task context")
        if str(candidate) in executable_values and (
            not candidate.is_file() or not os.access(candidate, os.X_OK)
        ):
            raise HermesBridgeError(f"Declared task executable is not executable: {candidate}")
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _context_writable_paths(context: AgentContext) -> list[Path]:
    paths: list[Path] = []
    for raw in context.runtime_context.writable_paths:
        candidate = Path(str(raw)).expanduser().resolve()
        if candidate == Path("/"):
            raise HermesBridgeError("Refusing to expose the host filesystem root as writable task context")
        if not candidate.exists():
            raise HermesBridgeError(f"Declared writable task path does not exist: {candidate}")
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _context_environment(context: AgentContext) -> Dict[str, str]:
    environment = dict(context.runtime_context.environment)
    unexpected = set(environment) - _SCOPED_RUNTIME_ENV_NAMES
    if unexpected:
        raise HermesBridgeError(
            f"Runtime context requested unsupported environment variables: {sorted(unexpected)}"
        )
    for name, value in environment.items():
        if "\x00" in value or "\n" in value or "\r" in value:
            raise HermesBridgeError(f"Runtime context environment variable {name!r} is malformed")
    return environment


def _safe_profile_name(value: Any, *, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-.")
    return name[:80] or fallback


def _normalized_external_mcp_servers(context: AgentContext) -> Dict[str, Dict[str, Any]]:
    """Translate protected FoH MCP declarations to upstream Hermes config."""
    raw = context.runtime_context.metadata.get("mcp_servers")
    if raw is None:
        raw = context.runtime_context.metadata.get("_external_mcp_servers")
    if raw in (None, []):
        return {}
    if not isinstance(raw, list):
        raise HermesBridgeError("Runtime MCP configuration must be a list")
    servers: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise HermesBridgeError(f"MCP server {index + 1} must be a mapping")
        name = _safe_profile_name(entry.get("name"), fallback=f"server-{index + 1}")
        if name in servers:
            raise HermesBridgeError(f"Duplicate MCP server name: {name}")
        command = entry.get("command")
        args = entry.get("args", [])
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise HermesBridgeError(f"MCP server {name!r} requires a safe command")
        if not isinstance(args, list) or not all(
            isinstance(value, str) and "\x00" not in value for value in args
        ):
            raise HermesBridgeError(f"MCP server {name!r} args must be strings")
        config: Dict[str, Any] = {
            "command": command,
            "args": list(args),
            "enabled": True,
            # These declarations are controller-owned and run inside the
            # already isolated benchmark sandbox. They never come from model
            # output, so mark them trusted for upstream's write-tool gate.
            "trust": "full",
        }
        for key in ("timeout", "connect_timeout"):
            if key in entry:
                try:
                    config[key] = max(1.0, float(entry[key]))
                except (TypeError, ValueError) as exc:
                    raise HermesBridgeError(
                        f"MCP server {name!r} has invalid {key}"
                    ) from exc
        servers[name] = config
    return servers


def _normalized_external_search(context: AgentContext) -> Optional[Dict[str, Any]]:
    """Validate the protected FoH search declaration for the Hermes bridge."""
    raw = context.runtime_context.metadata.get("_external_search")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise HermesBridgeError("External search configuration must be a mapping")
    provider = str(raw.get("provider") or "").strip().lower()
    if provider not in {"tavily", "brave", "openrouter"}:
        raise HermesBridgeError(f"Unsupported external search provider: {provider!r}")
    key_env = _validated_credential_environment_name(raw.get("api_key_env"))
    try:
        max_results = max(1, min(10, int(raw.get("max_results") or 5)))
    except (TypeError, ValueError) as exc:
        raise HermesBridgeError("External search max_results must be an integer") from exc
    base_url = str(raw.get("base_url") or "").strip() or None
    if base_url and not base_url.startswith("https://"):
        raise HermesBridgeError("External search base_url must use HTTPS")
    return {
        "provider": provider,
        "api_key_env": key_env,
        "base_url": base_url,
        "max_results": max_results,
    }


def _stage_external_skills(home: Path, context: AgentContext) -> None:
    """Copy configured task skills into this turn's isolated Hermes profile."""
    skills_root = home / "skills"
    used: set[str] = set()
    for index, raw in enumerate(context.runtime_context.skills):
        source = Path(str(raw)).expanduser().resolve()
        if not source.exists() or source == Path("/"):
            raise HermesBridgeError(f"Configured skill path does not exist: {source}")
        base = _safe_profile_name(
            source.name if source.is_dir() else source.stem,
            fallback=f"skill-{index + 1}",
        )
        name = base
        suffix = 2
        while name in used:
            name = f"{base}-{suffix}"
            suffix += 1
        used.add(name)
        destination = skills_root / name
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=False)
        elif source.is_file():
            destination.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source, destination / "SKILL.md")
        else:
            raise HermesBridgeError(f"Unsupported skill path: {source}")


def _stage_hermes_profile(
    home: Path,
    context: AgentContext,
    *,
    toolsets: Iterable[str],
) -> list[str]:
    """Materialize per-task tool configuration without persisting secrets."""
    selected = {str(value) for value in toolsets}
    config: Dict[str, Any] = {}
    configured_search = _normalized_external_search(context)
    if configured_search and "web" not in selected:
        raise HermesBridgeError(
            "search.provider is configured but the Hermes 'web' toolset is disabled"
        )
    if "web" in selected:
        # The pinned kernel defaults this on, but persist the choice in every
        # isolated profile so native search remains available without a
        # separate search credential. Credentialed providers still take
        # precedence when an operator configures one upstream.
        config["web"] = {"keyless_fallback": True}
        if configured_search:
            config["web"]["search_backend"] = "foh-configured"
    if "image_gen" in selected:
        # The provider resolves the already-supplied main OpenRouter runtime
        # in memory; no credential is written to this profile.
        config["image_gen"] = {"provider": "openrouter"}
    servers = _normalized_external_mcp_servers(context)
    if servers:
        config["mcp_servers"] = servers
    _stage_external_skills(home, context)
    (home / "config.yaml").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return list(servers)


def _no_proxy_for_model(existing: str, base_url: str) -> str:
    values = [value.strip() for value in str(existing or "").split(",") if value.strip()]
    hostname = urlparse(base_url).hostname
    for value in ("127.0.0.1", "localhost", hostname or ""):
        if value and value not in values:
            values.append(value)
    return ",".join(values)


def _resolver_read_only_paths() -> list[Path]:
    """Make an /etc/resolv.conf symlink target reachable inside the sandbox."""
    resolver = Path("/etc/resolv.conf")
    try:
        target = resolver.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return []
    return [target] if not _is_within(target, Path("/etc")) else []


def _mount_parent_directories(destinations: Iterable[Path]) -> list[Path]:
    existing_roots = {Path("/tmp"), Path("/proc"), Path("/dev")}
    existing_roots.update(path for path in _SYSTEM_READ_ONLY_PATHS if path.exists())
    parents: set[Path] = set()
    for destination in destinations:
        parent = destination.parent
        while parent != Path("/"):
            if any(parent == root or _is_within(parent, root) for root in existing_roots):
                break
            parents.add(parent)
            parent = parent.parent
    return sorted(parents, key=lambda path: (len(path.parts), str(path)))


def _sandbox_command(
    context: AgentContext,
    *,
    source: Path,
    python: Path,
    sandbox: Path,
    temporary: Path,
    request_path: Path,
    result_path: Path,
) -> list[str]:
    """Build the fail-closed bubblewrap boundary for one Hermes turn."""
    workspace = context.runtime_context.workspace.resolve()
    venv = python.parent.parent
    python_prefix = python.resolve().parent.parent
    active_prefix = _absolute_without_resolving_symlinks(sys.prefix)
    base_prefix = _absolute_without_resolving_symlinks(
        getattr(sys, "base_prefix", "")
        or Path(getattr(sys, "_base_executable", sys.executable)).parent.parent
    )
    protected: list[Path] = []
    for path in [
        source, venv, python_prefix, active_prefix, base_prefix, _DRIVER,
        *_resolver_read_only_paths(),
    ]:
        absolute = _absolute_without_resolving_symlinks(path)
        for candidate in (absolute, absolute.resolve()):
            if candidate == Path("/"):
                raise HermesBridgeError("Refusing to expose the host filesystem root")
            if candidate not in protected:
                protected.append(candidate)
    writable = [workspace, temporary.resolve()]
    for path in _context_writable_paths(context):
        if path not in writable:
            writable.append(path)
    # Benchmark workspace_files are also listed as attachments, but they are
    # mutable task inputs. Only context genuinely outside the task workspace
    # receives a read-only mount; in-workspace attachments inherit the
    # workspace's writable bind.
    context_read_only = [
        path
        for path in _context_read_only_paths(context)
        if not any(_is_within(path, root) for root in writable)
    ]
    read_only = [*protected]
    for path in context_read_only:
        if path not in read_only:
            read_only.append(path)
    for path in writable:
        if path in (workspace, temporary.resolve()):
            continue
        if any(
            _is_within(path, protected_path) or _is_within(protected_path, path)
            for protected_path in protected
        ):
            raise HermesBridgeError(
                f"Refusing writable task mount that overlaps protected runtime code: {path}"
            )
    outer_read_only = [
        path
        for path in read_only
        if not any(_is_within(path, root) for root in writable)
    ]
    protected_inner = [
        path
        for path in protected
        if any(_is_within(path, root) for root in writable)
    ]
    destinations = [*writable, *read_only]
    command = [
        str(sandbox),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--share-net",
        "--cap-drop", "ALL",
    ]
    for path in _SYSTEM_READ_ONLY_PATHS:
        if path.exists():
            command.extend(("--ro-bind", str(path), str(path)))
    command.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
    for parent in _mount_parent_directories(destinations):
        command.extend(("--dir", str(parent)))
    for path in outer_read_only:
        command.extend(("--ro-bind", str(path), str(path)))
    for path in writable:
        command.extend(("--bind", str(path), str(path)))
    # If a task workspace happens to contain protected runtime/source files,
    # overlay them read-only after the writable workspace mount.
    for path in protected_inner:
        command.extend(("--ro-bind", str(path), str(path)))
    command.extend(
        (
            "--chdir", str(workspace),
            "--",
            str(python),
            str(_DRIVER),
            str(request_path),
            str(result_path),
        )
    )
    return command


async def _terminate_and_drain_process(process: Any) -> None:
    """Stop Bubblewrap and reap its child after timeout or cancellation.

    ``--die-with-parent`` makes the namespace child die with Bubblewrap, but
    the parent still has to signal and reap Bubblewrap itself.  A short TERM
    grace lets Hermes close files; KILL is the fail-safe for a stuck process.
    Re-running ``communicate`` after the original await was cancelled also
    drains the pipe transports so no subprocess/zombie survives the task.
    """
    if getattr(process, "returncode", None) is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(
            process.communicate(), timeout=_PROCESS_TERMINATE_GRACE_SECONDS
        )
        return
    except (asyncio.TimeoutError, ProcessLookupError, RuntimeError):
        pass
    if getattr(process, "returncode", None) is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(
            process.communicate(), timeout=_PROCESS_TERMINATE_GRACE_SECONDS
        )
    except (asyncio.TimeoutError, ProcessLookupError, RuntimeError):
        # The OS will reap the child when the event-loop transport closes;
        # cleanup must never mask the original timeout/cancellation.
        pass


async def _cleanup_without_cancellation(process: Any) -> None:
    """Complete subprocess cleanup even if the caller is cancelled again."""
    cleanup = asyncio.create_task(_terminate_and_drain_process(process))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    # Retrieve a possible exception so asyncio does not report an abandoned
    # task.  Cleanup is best-effort and the original cancellation wins.
    try:
        cleanup.result()
    except Exception:
        pass


def _finish_instruction(finish_kind: Literal["solution", "insight_library"]) -> str:
    return (
        (
            "Finish by returning only one JSON object with this exact envelope: "
            '{"foh_envelope_version":1,"solution":"the complete benchmark-facing '
            'answer","reflection":{"summary":"privacy-safe execution summary",'
            '"actions":["generic action"],"observations":["generic outcome"],'
            '"reusable_insights":["general lesson"],"outcome":"completed"}}. '
            "Keep solution exactly what the task should receive. The reflection is "
            "separate learning for the server: do not copy task wording, answer "
            "content, credentials, personal data, or workspace paths into it."
        )
        if finish_kind == "solution"
        else (
            "Produce the shared insight library. Write the final dictionary to "
            "insight_library.json in the current workspace and/or return that JSON "
            "object as the final response; do not return an unrelated prose-only summary."
        )
    )


def _validation_instruction(context: AgentContext) -> str:
    target_role = str(context.runtime_context.metadata.get("target_role") or "")
    return (
        "\n\nBefore finishing this harness revision, run exactly: "
        f"python -m agent_runtime.validate_candidate --role {target_role} .\n"
        "Repair every failed check and rerun it. There is no run_validation "
        "tool in the external Hermes registry; use this protected CLI."
        if target_role in {"client_agent", "server_agent"}
        else ""
    )


def _execution_state(context: AgentContext, *, include_objective: bool) -> Dict[str, Any]:
    workspace = context.runtime_context.workspace.resolve()
    attachments = []
    for raw in context.runtime_context.attachments:
        candidate = Path(str(raw)).expanduser()
        attachments.append(str(candidate.resolve()) if candidate.exists() else str(raw))
    runtime_metadata = {
        str(key): value
        for key, value in context.runtime_context.metadata.items()
        if key not in {"mcp_servers", "_external_mcp_servers", "_external_search"}
        and not any(token in str(key).lower() for token in ("key", "token", "secret", "password"))
    }
    for key in _SCOPED_READ_METADATA_KEYS:
        raw = runtime_metadata.get(key)
        if raw:
            candidate = Path(str(raw)).expanduser()
            if candidate.exists():
                runtime_metadata[key] = str(candidate.resolve())
    state: Dict[str, Any] = {
        "role": context.role,
        "shared_insight_library": context.insight_library,
        "workspace": str(workspace),
        "attachments": attachments,
        "skills": context.runtime_context.skills,
        "runtime_metadata": runtime_metadata,
    }
    if include_objective:
        state["objective"] = context.objective
    return state


def _turn_system_context(context: AgentContext) -> str:
    return (
        "Execute the supplied benchmark turn in the persistent task workspace. "
        "The user message is authoritative for this turn. Same-session turns receive "
        "their prior conversation explicitly; a new-session turn receives none, but "
        "files in the workspace persist.\n\nFoH execution context:\n"
        f"{render_json(_execution_state(context, include_objective=False), max_chars=180_000)}"
    )


def _prompt(context: AgentContext, finish_kind: Literal["solution", "insight_library"]) -> str:
    return (
        f"{_finish_instruction(finish_kind)}{_validation_instruction(context)}"
        "\n\nFoH execution context:\n"
        f"{render_json(_execution_state(context, include_objective=True), max_chars=180_000)}"
    )


def _overall_timeout_seconds(context: AgentContext, runtime: AgentRuntime) -> float:
    timeout = float(runtime.limits.max_elapsed_seconds)
    raw = context.runtime_context.metadata.get("timeout_seconds")
    try:
        task_timeout = float(raw) if raw is not None else timeout
    except (TypeError, ValueError):
        task_timeout = timeout
    if task_timeout > 0:
        timeout = min(timeout, task_timeout)
    return max(1.0, timeout)


def _solution_envelope(final: str) -> tuple[str, Optional[ClientReflection], bool]:
    """Unwrap FoH's client result without mistaking task-authored JSON for it."""
    value = first_json_object(
        final, required_keys=("foh_envelope_version", "solution")
    )
    if not isinstance(value, dict) or value.get("foh_envelope_version") != 1:
        return final, None, False
    solution = value.get("solution", "")
    if not isinstance(solution, str):
        solution = json.dumps(solution, ensure_ascii=False)
    raw_reflection = value.get("reflection")
    reflection = (
        ClientReflection.from_value(raw_reflection)
        if isinstance(raw_reflection, dict)
        else None
    )
    return solution, reflection, True


def _external_execution(
    context: AgentContext,
    result: Dict[str, Any],
    *,
    final: str,
    status: str,
    secret: str,
) -> Dict[str, Any]:
    """Preserve bounded tool evidence for graders, never for federation."""
    execution = dict(context.runtime_context.raw_execution or {})
    existing = canonical_transcript(execution.get("transcript"))
    external = canonical_transcript(
        result.get("transcript") if isinstance(result.get("transcript"), list) else []
    )
    events = (existing + external)[: _MAX_TRANSCRIPT_EVENTS - 1]
    safe_final = str(final or "").replace(secret, "[REDACTED]")[:50_000]
    if safe_final:
        events.append(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": safe_final}],
                },
            }
        )
    execution.update(
        {
            "status": status,
            "transcript": events,
            "final_answer": safe_final,
        }
    )
    return execution


def _external_reflection(result: Dict[str, Any], status: str) -> ReasoningTrace:
    names = [sanitize_text(value, max_chars=80) for value in result.get("tool_names") or []]
    tool_note = ", ".join(value for value in names if value) or "no external tools"
    failures = int(result.get("tool_failures") or 0)
    return ReasoningTrace(
        stage="hermes_execution",
        action=f"used Hermes Agent with {tool_note}",
        observation=(
            f"completed {int(result.get('api_calls') or 0)} model iteration(s) and "
            f"{int(result.get('tool_calls') or 0)} tool call(s); tool failures={failures}"
        ),
        outcome=status,
        reusable_feedback=(
            "Keep the successful tool sequence, recover explicitly from failed calls, "
            "and verify externally visible artifacts before finishing."
        ),
    )


async def _run_external(
    context: AgentContext,
    runtime: AgentRuntime,
    *,
    finish_kind: Literal["solution", "insight_library"],
) -> AgentOutcome:
    settings = runtime.hermes_settings
    source, python, sandbox = _resolve_external(settings)
    key_env = _validated_credential_environment_name(
        runtime.model.settings.api_key_env
    )
    key = runtime.model.settings.api_key()
    if not key:
        raise HermesBridgeError(f"Missing remote-model credential: set {key_env}")
    workspace = context.runtime_context.workspace.resolve()
    overall_timeout = _overall_timeout_seconds(context, runtime)
    library_path = workspace / "insight_library.json"
    library_before = library_path.read_bytes() if library_path.is_file() else None
    # This pinned image-generation integration is OpenRouter-only. Do not
    # advertise it or let a Google credential reach an OpenRouter tool.
    toolsets = tuple(
        name for name in settings.toolsets
        if not (runtime.model.settings.provider == "gemini" and name == "image_gen")
    )

    with tempfile.TemporaryDirectory(prefix="foh-hermes-") as temporary_value:
        temporary = Path(temporary_value).resolve()
        request_path = temporary / "request.json"
        result_path = temporary / "result.json"
        home = temporary / f"home-{_safe_role(context.role)}-{uuid.uuid4().hex[:8]}"
        home.mkdir(parents=True, exist_ok=True)
        mcp_server_names = _stage_hermes_profile(
            home,
            context,
            toolsets=toolsets,
        )
        configured_search = _normalized_external_search(context)
        base_url = runtime.model.settings.base_url or (
            "https://openrouter.ai/api/v1"
            if runtime.model.settings.provider == "openrouter"
            else ""
        )
        request = {
            "prompt": _prompt(context, finish_kind),
            "session_turns": [
                turn.to_dict() for turn in context.runtime_context.session_turns
            ],
            "turn_system_context": _turn_system_context(context),
            "finish_instruction": (
                _finish_instruction(finish_kind) + _validation_instruction(context)
            ),
            "system_prompt": runtime.base_system_prompt,
            "provider": runtime.model.settings.provider,
            "base_url": base_url,
            "model": runtime.model.settings.name,
            "api_key_env": key_env,
            "max_tokens": runtime.model.settings.max_tokens,
            "temperature": runtime.model.settings.temperature,
            "top_p": runtime.model.settings.top_p,
            "top_k": runtime.model.settings.top_k,
            "reasoning_effort": runtime.model.settings.reasoning_effort,
            "max_iterations": min(
                settings.max_iterations,
                runtime.limits.max_rounds,
                runtime.limits.max_model_calls,
            ),
            "run_budget_seconds": overall_timeout,
            "toolsets": list(toolsets),
            "mcp_server_names": mcp_server_names,
            "configured_search": (
                {
                    key: value
                    for key, value in configured_search.items()
                    if key != "api_key_env"
                }
                if configured_search
                else None
            ),
            "session_id": f"foh-{_safe_role(context.role)}-{uuid.uuid4().hex}",
        }
        if runtime.model.settings.provider == "gemini":
            request["gemini_rate_limit"] = _gemini_rate_limit_request(temporary)
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        environment = {
            name: value for name, value in os.environ.items() if name in _SAFE_ENV_NAMES
        }
        environment.update(
            {
                key_env: key,
                "HOME": str(home),
                "HERMES_HOME": str(home),
                "HERMES_YOLO_MODE": "1",
                "HERMES_ACCEPT_HOOKS": "1",
                "PATH": os.pathsep.join(
                    (str(python.parent), environment.get("PATH", "/usr/bin:/bin"))
                ),
                "PYTHONPATH": str(source),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(temporary),
                # Verified against pinned Hermes' request timeout resolver:
                # HERMES_API_TIMEOUT is its supported per-request fallback.
                "HERMES_API_TIMEOUT": str(runtime.model.settings.timeout_seconds),
                "TERMINAL_ENV": "local",
                "TERMINAL_CWD": str(workspace),
                "TERMINAL_TIMEOUT": str(runtime.limits.max_command_seconds),
            }
        )
        if configured_search:
            search_key = os.getenv(str(configured_search["api_key_env"]))
            if not search_key:
                raise HermesBridgeError(
                    "Missing configured search credential: set "
                    + str(configured_search["api_key_env"])
                )
            # The driver consumes and removes this before the first prompt or
            # terminal command. The request file contains no credential.
            environment["FOH_CONFIGURED_SEARCH_KEY"] = search_key
        scoped_environment = _context_environment(context)
        environment.update(scoped_environment)
        executable_directories: list[str] = []
        for raw in context.runtime_context.executable_paths:
            directory = str(Path(raw).expanduser().resolve().parent)
            if directory not in executable_directories:
                executable_directories.append(directory)
        environment["PATH"] = os.pathsep.join(
            (
                str(python.parent),
                *executable_directories,
                environment.get("PATH", "/usr/bin:/bin"),
            )
        )
        no_proxy = _no_proxy_for_model(
            environment.get("NO_PROXY") or environment.get("no_proxy", ""),
            base_url,
        )
        environment["NO_PROXY"] = no_proxy
        environment["no_proxy"] = no_proxy
        sandbox_command = _sandbox_command(
            context,
            source=source,
            python=python,
            sandbox=sandbox,
            temporary=temporary,
            request_path=request_path,
            result_path=result_path,
        )
        process = await asyncio.create_subprocess_exec(
            *sandbox_command,
            cwd=str(workspace),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                _communicate_with_provider_abort(process), timeout=overall_timeout
            )
        except asyncio.TimeoutError as exc:
            await _cleanup_without_cancellation(process)
            raise TimeoutError("Hermes subprocess exceeded the FoH task timeout") from exc
        except asyncio.CancelledError:
            # client.py applies an outer wait_for around the entire agent run.
            # Its cancellation must not orphan Bubblewrap, Hermes, or an
            # in-flight OpenRouter request.
            await _cleanup_without_cancellation(process)
            raise
        except FatalProviderError:
            await _cleanup_without_cancellation(process)
            raise
        if not result_path.is_file():
            evidence = sanitize_text(
                (stderr or stdout).decode(errors="replace")[-3000:],
                forbidden_values=(key,),
                max_chars=3000,
            )
            raise HermesBridgeError(
                f"Hermes process exited {process.returncode} without a result: {evidence}"
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HermesBridgeError(f"Hermes result was not valid JSON: {exc}") from exc

        if not isinstance(result, dict):
            raise HermesBridgeError("Hermes result must be a JSON object")
        process_returncode = process.returncode

    await runtime.record_external_usage(
        model_calls=int(result.get("api_calls") or 0),
        input_tokens=int(result.get("input_tokens") or 0),
        output_tokens=int(result.get("output_tokens") or 0),
    )
    provider_error = result.get("provider_error")
    if isinstance(provider_error, dict):
        raise_for_provider_error(
            provider=runtime.model.settings.provider, model=runtime.model.settings.name,
            status_code=provider_error.get("status_code"),
            message=str(provider_error.get("message") or "Provider request failed"),
            reason=str(provider_error.get("reason") or ""), secret=key,
        )
    # Also recognize failed results from older drivers before an insight-library
    # finish can replace the actual provider error with an empty dictionary.
    if runtime.model.settings.provider == "gemini" and (
        result.get("failed") or result.get("bridge_ok") is not True
    ):
        raise_for_gemini_error_text(
            str(result.get("error") or result.get("final_response") or ""),
            model=runtime.model.settings.name, secret=key,
        )
    if (result.get("failed") and int(result.get("api_calls") or 0) > 0
            and not int(result.get("output_tokens") or 0)
            and not int(result.get("tool_calls") or 0)):
        evidence = sanitize_text(
            str(result.get("error") or result.get("final_response") or "No model output"),
            forbidden_values=(key,), max_chars=2000,
        )
        raise FatalProviderError(
            f"{runtime.model.settings.provider} model {runtime.model.settings.name} failed "
            f"before producing any output or tool calls: {evidence}"
        )
    if result.get("bridge_ok") is not True or process_returncode != 0:
        evidence = sanitize_text(
            result.get("error")
            or (stderr or stdout).decode(errors="replace")[-3000:]
            or "external driver did not report a successful runtime boundary",
            forbidden_values=(key,),
            max_chars=3000,
        )
        raise HermesBridgeError(
            f"Hermes process failed at the runtime boundary "
            f"(exit {process_returncode}): {evidence}"
        )
    final = str(result.get("final_response") or "")
    status = "completed" if result.get("completed") and not result.get("failed") and final.strip() else "failed"
    reflection: Optional[ClientReflection] = None
    structured_envelope = False
    if finish_kind == "insight_library":
        inline = first_json_object(final)
        library: Any = inline if isinstance(inline, dict) else None
        library_after = library_path.read_bytes() if library_path.is_file() else None
        if library is None and library_after is not None and library_after != library_before:
            try:
                library = json.loads(library_after.decode("utf-8"))
            except Exception:
                library = None
        if not isinstance(library, dict):
            status = "failed"
            library = {}
        final = json.dumps(sanitize_library(library), ensure_ascii=False)
    else:
        final, reflection, structured_envelope = _solution_envelope(final)
        if not final.strip():
            status = "failed"
    trace = _external_reflection(result, status)
    raw_execution = _external_execution(
        context, result, final=final, status=status, secret=key
    )
    return AgentOutcome(
        solution=final,
        reasoning_traces=[trace],
        status=status,  # type: ignore[arg-type]
        raw_execution=raw_execution,
        artifacts=[],
        telemetry={
            "agent_kernel": "hermes-agent",
            "hermes_backend": "external_process",
            "hermes_process_sandbox": "bubblewrap",
            "hermes_revision": settings.revision,
            "hermes_tool_names": result.get("tool_names") or [],
            "hermes_tool_calls": int(result.get("tool_calls") or 0),
            "hermes_tool_failures": int(result.get("tool_failures") or 0),
            "hermes_exit_code": process.returncode,
            "hermes_turn_exit_reason": result.get("turn_exit_reason"),
            "hermes_session_turns": int(result.get("session_turn_count") or 1),
            "hermes_session_resets": int(result.get("session_resets") or 0),
            "hermes_structured_envelope": structured_envelope,
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "bridge_error": sanitize_text(
                result.get("error", ""), forbidden_values=(key,), max_chars=1000
            ),
        },
        reflection=reflection,
    )


async def run_hermes_agent(
    context: AgentContext,
    runtime: AgentRuntime,
    *,
    finish_kind: Literal["solution", "insight_library"],
) -> AgentOutcome:
    """Run external Hermes, except for an explicit injected-responder test."""
    if runtime.model.has_responder:
        outcome = AgentOutcome.from_value(
            await run_autonomous_loop(context, runtime, finish_kind=finish_kind)
        )
        outcome.telemetry = {
            **outcome.telemetry,
            "agent_kernel": "hermes-agent",
            "hermes_backend": "embedded_injected_test_adapter",
            "hermes_revision": runtime.hermes_settings.revision,
            "test_only_backend": True,
        }
        return outcome
    if runtime.model.settings.provider not in OPERATIONAL_PROVIDERS:
        raise HermesBridgeError(
            "Operational Hermes execution requires model.provider='openrouter' or 'gemini'; "
            "inject an in-memory responder explicitly for tests"
        )
    if not runtime.hermes_settings.enabled:
        raise HermesBridgeError(
            "Hermes is disabled, but operational client/server execution requires "
            "the pinned external kernel"
        )
    if not runtime.hermes_settings.require_external_for_remote:
        raise HermesBridgeError(
            "Remote-to-embedded Hermes fallback is forbidden; set "
            "require_external_for_remote=true"
        )
    if not _external_requested(runtime):  # defensive invariant, not a fallback
        raise HermesBridgeError("External Hermes backend selection failed closed")
    return await _run_external(context, runtime, finish_kind=finish_kind)


__all__ = ["HermesBridgeError", "preflight_external_hermes", "run_hermes_agent"]
