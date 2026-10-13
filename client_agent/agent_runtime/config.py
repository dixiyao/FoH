"""YAML configuration and process-only explicit credentials."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPERATIONAL_PROVIDERS = {"openrouter", "gemini"}


@dataclass(frozen=True)
class ModelSettings:
    # The dataclass default is deliberately non-operational so constructing
    # AppConfig() in a unit test never selects a paid backend accidentally.
    # Checked-in config.yaml supplies OpenRouter; operational preflight rejects
    # this sentinel unless a ModelClient responder was explicitly injected.
    provider: str = "mock"
    name: str = "mock"
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.4
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 8192
    # Revision completion ceiling; None inherits max_tokens. Tune per model.
    revision_max_tokens: Optional[int] = None
    # None preserves the provider default.
    reasoning_effort: Optional[str] = None
    timeout_seconds: int = 180
    retries: int = 2
    # Runtime-only credential supplied by an entrypoint, never loaded from YAML.
    api_key_value: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.provider != "gemini":
            return
        # The pinned Hermes adapter speaks Google's native generateContent API.
        endpoint = (self.base_url or GEMINI_BASE_URL).rstrip("/")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "generativelanguage.googleapis.com"
            or parsed.path not in {"/v1", "/v1beta"}
            or parsed.query or parsed.fragment
        ):
            raise ValueError(
                "provider=gemini requires Google's native API endpoint; set "
                f"model.base_url to null or {GEMINI_BASE_URL}"
            )
        name = self.name.strip()
        for prefix in ("google/", "gemini/", "models/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if not name or "/" in name or name == "mock":
            raise ValueError("provider=gemini requires a Google model ID, e.g. gemini-2.5-flash")
        if self.top_k:
            raise ValueError("The pinned Hermes Gemini adapter does not support top_k; set model.top_k=0")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base_url", endpoint)
        object.__setattr__(self, "api_key_env", self.api_key_env or "GEMINI_API_KEY")

    def api_key(self) -> Optional[str]:
        if self.api_key_value is not None:
            return self.api_key_value
        return os.getenv(self.api_key_env) if self.api_key_env else None

    def public_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "name": self.name,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "revision_max_tokens": self.revision_max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
        }


@dataclass(frozen=True)
class SearchSettings:
    """Optional keyed search override; None uses native Hermes keyless search."""

    provider: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    max_results: int = 5

    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env) if self.api_key_env else None

    def public_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "max_results": self.max_results,
        }


_SEARCH_PROVIDERS = {"tavily", "brave", "openrouter"}


# Exact upstream release used by the process bridge.  Pinning a release
# commit (rather than moving ``main``) keeps cluster runs reproducible.
# ``v2026.8.19`` is an annotated tag.  Pin the peeled commit, not the tag
# object's SHA: ``git checkout`` resolves the tag to this commit and
# ``git rev-parse HEAD`` can then verify the installation exactly.
HERMES_AGENT_RELEASE_COMMIT = "fcbd1076a93841fa88855acce810e342a5b78101"


@dataclass(frozen=True)
class HermesSettings:
    """Pinned external Hermes Agent kernel used by real model runs.

    Upstream Hermes owns a global tool registry, provider client, session
    store, and synchronous conversation loop.  FoH therefore runs it behind
    a subprocess/file boundary instead of importing it into the concurrent
    benchmark process. Only an explicitly responder-injected Python test call
    may use the embedded loop; configuration files cannot select it.
    """

    enabled: bool = True
    source_dir: Optional[str] = ".hermes/hermes-agent"
    python_executable: Optional[str] = ".hermes/venv/bin/python"
    # Mandatory outer process sandbox for real Hermes runs.  Bubblewrap is
    # available on the DS cluster and preserves the host network namespace
    # while giving the agent a deliberately sparse mount namespace.
    sandbox_executable: Optional[str] = "/usr/bin/bwrap"
    upstream_url: str = "https://github.com/NousResearch/hermes-agent.git"
    revision: str = HERMES_AGENT_RELEASE_COMMIT
    toolsets: tuple[str, ...] = (
        "terminal", "file", "todo", "web", "skills", "image_gen"
    )
    max_iterations: int = 15
    require_external_for_remote: bool = True

    def public_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_dir": self.source_dir,
            "python_executable": self.python_executable,
            "sandbox_executable": self.sandbox_executable,
            "upstream_url": self.upstream_url,
            "revision": self.revision,
            "toolsets": list(self.toolsets),
            "max_iterations": self.max_iterations,
            "require_external_for_remote": self.require_external_for_remote,
        }


@dataclass(frozen=True)
class RuntimeLimits:
    max_rounds: int = 16
    max_model_calls: int = 64
    max_total_tokens: int = 250_000
    max_global_model_concurrency: int = 32
    max_total_agents: int = 8
    max_parallel_agents: int = 4
    max_agent_depth: int = 2
    max_tool_calls: int = 64
    max_elapsed_seconds: int = 1800
    max_trace_chars: int = 1200
    max_command_seconds: int = 300


@dataclass(frozen=True)
class HarnessGitSettings:
    """Optional git distribution of evolved harnesses (see agent_runtime.harness_git)."""

    enabled: bool = False
    repo: str = ""
    branch_prefix: str = "iteration-"
    push: bool = True
    author_name: str = "FoH server"
    author_email: str = "foh-server@localhost"


@dataclass(frozen=True)
class AppConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    num_workers: int = 8
    skill_paths: tuple[str, ...] = ()
    mcp_servers: tuple[Dict[str, Any], ...] = ()
    # Preserve positional compatibility with the original configuration fields.
    search: SearchSettings = field(default_factory=SearchSettings)
    hermes: HermesSettings = field(default_factory=HermesSettings)
    # Entrypoint-selected policy: record validation but always apply authored revisions.
    apply_all_revisions: bool = False
    # Optional: publish every evolved harness as a git branch (iteration-NNN).
    harness_git: HarnessGitSettings = field(default_factory=HarnessGitSettings)

    def public_dict(self) -> Dict[str, Any]:
        value = asdict(replace(self, model=replace(self.model, api_key_value=None)))
        value["model"].pop("api_key_value", None)
        if not self.apply_all_revisions:
            value.pop("apply_all_revisions", None)
        # Harness distribution is transport, not an experiment parameter: it
        # must not change the experiment fingerprint or block a resume.
        value.pop("harness_git", None)
        return value


def _is_local_git_path(repo: str) -> bool:
    """True for filesystem repositories (bare dirs on a shared mount), false for URLs/SSH."""
    return "://" not in repo and not repo.startswith("git@") and ":" not in repo.split("/")[0]


def _mapping(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to load config.yaml") from exc
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "config")
    unknown_root = set(root) - {"model", "limits", "search", "hermes", "num_workers", "skill_paths", "mcp_servers", "harness_git"}
    if unknown_root:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown_root)}")
    model_raw = _mapping(root.get("model"), "model")
    limits_raw = _mapping(root.get("limits"), "limits")
    search_raw = _mapping(root.get("search"), "search")
    hermes_raw = _mapping(root.get("hermes"), "hermes")
    harness_git_raw = _mapping(root.get("harness_git"), "harness_git")
    unknown_harness_git = set(harness_git_raw) - set(HarnessGitSettings.__dataclass_fields__)
    if unknown_harness_git:
        raise ValueError(f"Unknown harness_git settings: {sorted(unknown_harness_git)}")
    harness_git = HarnessGitSettings(**harness_git_raw)
    if harness_git.enabled and not str(harness_git.repo).strip():
        raise ValueError("harness_git.repo is required when harness_git.enabled is true")
    if harness_git.enabled and _is_local_git_path(str(harness_git.repo)):
        candidate = Path(str(harness_git.repo)).expanduser()
        if not candidate.is_absolute():
            harness_git = replace(harness_git, repo=str(config_path.parent / candidate))

    model_fields = set(ModelSettings.__dataclass_fields__) - {"api_key_value"}
    limit_fields = set(RuntimeLimits.__dataclass_fields__)
    search_fields = set(SearchSettings.__dataclass_fields__)
    hermes_fields = set(HermesSettings.__dataclass_fields__)
    unknown_model = set(model_raw) - model_fields
    unknown_limits = set(limits_raw) - limit_fields
    unknown_search = set(search_raw) - search_fields
    unknown_hermes = set(hermes_raw) - hermes_fields
    if unknown_model:
        raise ValueError(f"Unknown model settings: {sorted(unknown_model)}")
    if unknown_limits:
        raise ValueError(f"Unknown runtime limits: {sorted(unknown_limits)}")
    if unknown_search:
        raise ValueError(f"Unknown search settings: {sorted(unknown_search)}")
    if unknown_hermes:
        raise ValueError(f"Unknown Hermes settings: {sorted(unknown_hermes)}")

    model = ModelSettings(**model_raw)
    limits = RuntimeLimits(**limits_raw)
    search = SearchSettings(**search_raw)
    hermes_values = dict(hermes_raw)
    if "toolsets" in hermes_values:
        raw_toolsets = hermes_values["toolsets"]
        if not isinstance(raw_toolsets, list) or not all(isinstance(value, str) for value in raw_toolsets):
            raise ValueError("hermes.toolsets must be a list of names")
        hermes_values["toolsets"] = tuple(raw_toolsets)
    for path_field in ("source_dir", "python_executable", "sandbox_executable"):
        raw_path = hermes_values.get(path_field)
        if raw_path:
            candidate = Path(str(raw_path)).expanduser()
            absolute = candidate if candidate.is_absolute() else config_path.parent / candidate
            # Keep the virtualenv interpreter path itself: resolving its
            # final symlink selects the base interpreter and loses the
            # venv's pyvenv.cfg/site-packages at execution time.
            hermes_values[path_field] = str(
                Path(os.path.abspath(absolute))
                if path_field == "python_executable"
                else absolute.resolve()
            )
    hermes = HermesSettings(**hermes_values)
    if hermes.max_iterations <= 0:
        raise ValueError("hermes.max_iterations must be positive")
    if hermes.enabled and not hermes.revision:
        raise ValueError("hermes.revision must pin an exact upstream commit")
    local_providers = {"mock", "huggingface", "hf", "local", "local-hf", "local_hf"}
    if model.provider not in local_providers:
        if not model.api_key_env:
            raise ValueError("Remote model providers require model.api_key_env")
        # Presence is an execution prerequisite, not a parsing prerequisite.
        # The canonical operational boundary validates it in
        # preflight_external_hermes; deferring the lookup is what lets unit
        # tests load the real checked-in config and explicitly inject their
        # in-memory responder without adding fake credentials to YAML/env.
    if search.provider is not None:
        if search.provider not in _SEARCH_PROVIDERS:
            raise ValueError(f"Unknown search.provider: {search.provider!r} (choose one of {sorted(_SEARCH_PROVIDERS)})")
        if search.provider == "openrouter" and model.provider != "openrouter":
            raise ValueError(
                "search.provider=openrouter piggybacks a real chat completion on the "
                "configured model client (see agent_runtime.tools.register_search_tool) "
                "and only makes sense when model.provider is also openrouter"
            )
        if not search.api_key_env:
            raise ValueError("A configured search.provider requires search.api_key_env")
        if not search.api_key():
            raise RuntimeError(f"Missing search API credential: set environment variable {search.api_key_env}")
    if (
        limits.max_total_agents < 1
        or limits.max_parallel_agents < 1
        or limits.max_global_model_concurrency < 1
    ):
        raise ValueError("Agent limits must be positive")
    for field_name in RuntimeLimits.__dataclass_fields__:
        value = getattr(limits, field_name)
        if field_name == "max_agent_depth":
            if value < 0:
                raise ValueError("max_agent_depth cannot be negative")
        elif value <= 0:
            raise ValueError(f"{field_name} must be positive")
    skill_paths_raw = root.get("skill_paths") or []
    mcp_servers_raw = root.get("mcp_servers") or []
    if not isinstance(skill_paths_raw, list) or not all(isinstance(value, str) for value in skill_paths_raw):
        raise ValueError("skill_paths must be a list of paths")
    if not isinstance(mcp_servers_raw, list) or not all(isinstance(value, dict) for value in mcp_servers_raw):
        raise ValueError("mcp_servers must be a list of mappings")
    return AppConfig(
        model=model,
        limits=limits,
        search=search,
        hermes=hermes,
        num_workers=max(1, int(root.get("num_workers", 8))),
        skill_paths=tuple(
            str((config_path.parent / value).resolve()) if not Path(value).expanduser().is_absolute()
            else str(Path(value).expanduser().resolve())
            for value in skill_paths_raw
        ),
        mcp_servers=tuple(dict(value) for value in mcp_servers_raw),
        harness_git=harness_git,
    )
