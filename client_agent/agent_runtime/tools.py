"""Workspace-scoped tools exposed to autonomous agents."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import socket
import shutil
import sys
import urllib.request
from ipaddress import ip_address
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlencode, urlparse

from .config import AppConfig, SearchSettings
from .contracts import RuntimeContext


ToolHandler = Callable[[Dict[str, Any]], Any | Awaitable[Any]]


async def _kill_and_drain(process: asyncio.subprocess.Process) -> None:
    """Terminate a subprocess and close its pipes before loop shutdown."""
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    # asyncio documents communicate() after kill for timeout cleanup. wait()
    # alone reaps the child but can leave pipe transports alive until after
    # asyncio.run() has already closed their event loop.
    try:
        await asyncio.shield(process.communicate())
    except (BrokenPipeError, ConnectionResetError, RuntimeError):
        await asyncio.shield(process.wait())


def _validate_public_http_endpoint(url: str) -> None:
    """Reject anything but a public HTTP(S) host -- the same SSRF guard
    fetch_url applies to agent-supplied URLs, reused here for a search
    provider endpoint. base_url is operator-configured, not agent-supplied,
    so the risk is lower, but a misconfigured or malicious override
    shouldn't get a free pass into the local network either.
    """
    if not url.startswith(("https://", "http://")):
        raise ValueError("Search endpoint requires an HTTP(S) URL")
    host = urlparse(url).hostname
    if not host or host.lower() == "localhost":
        raise PermissionError("Local network search endpoints are not allowed")
    for address in socket.getaddrinfo(host, None):
        if not ip_address(address[4][0]).is_global:
            raise PermissionError("Private and local network search endpoints are not allowed")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments: str
    handler: ToolHandler

    def public_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


class ToolRegistry:
    _BUILTIN_NAMES = {
        "list_files",
        "read_file",
        "write_file",
        "append_file",
        "edit_file",
        "run_command",
        "fetch_url",
    }

    def __init__(
        self,
        workspace: Path,
        *,
        command_timeout: int = 300,
        allowed_read_paths: Optional[Iterable[Path]] = None,
        allow_shell_network: bool = False,
    ):
        self.workspace = workspace.expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.command_timeout = max(1, int(command_timeout))
        self.allowed_read_paths = [
            Path(value).expanduser().resolve() for value in (allowed_read_paths or [])
        ]
        self.allow_shell_network = bool(allow_shell_network)
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_builtins()

    def register(
        self,
        name: str,
        description: str,
        arguments: str,
        handler: ToolHandler,
    ) -> None:
        self._tools[name] = ToolDefinition(name, description, arguments, handler)

    def definitions(self, allowed: Optional[Iterable[str]] = None) -> list[Dict[str, str]]:
        allowed_set = set(allowed) if allowed is not None else None
        return [
            tool.public_dict()
            for name, tool in sorted(self._tools.items())
            if allowed_set is None or name in allowed_set
        ]

    def clone_for(self, workspace: Path) -> "ToolRegistry":
        clone = ToolRegistry(
            workspace,
            command_timeout=self.command_timeout,
            allowed_read_paths=self.allowed_read_paths,
            allow_shell_network=self.allow_shell_network,
        )
        for name, tool in self._tools.items():
            if name not in self._BUILTIN_NAMES:
                clone._tools[name] = tool
        return clone

    def register_container_tools(self, container: str, docker_host: Optional[str] = None) -> None:
        """Expose explicit shell/file operations in an official benchmark container."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", container):
            raise ValueError("Unsafe container identifier")
        environment = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        if docker_host:
            environment["DOCKER_HOST"] = str(docker_host)

        async def _docker(arguments: list[str], *, stdin: Optional[bytes] = None) -> Dict[str, Any]:
            process = await asyncio.create_subprocess_exec(
                "docker", *arguments,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(stdin), timeout=self.command_timeout
                )
            except asyncio.TimeoutError:
                await _kill_and_drain(process)
                raise TimeoutError(f"Container command exceeded {self.command_timeout}s")
            except asyncio.CancelledError:
                await _kill_and_drain(process)
                raise
            return {"exit_code": process.returncode, "output": stdout.decode(errors="replace")[-100_000:]}

        async def run_container_command(args: Dict[str, Any]) -> Dict[str, Any]:
            command = str(args.get("command", ""))
            if not command:
                raise ValueError("run_container_command requires command")
            return await _docker(["exec", container, "sh", "-lc", command])

        async def read_container_file(args: Dict[str, Any]) -> str:
            path = str(args.get("path", ""))
            if not path.startswith("/"):
                raise ValueError("Container file path must be absolute")
            result = await _docker(["exec", container, "sh", "-lc", "cat -- \"$1\"", "sh", path])
            if result["exit_code"] != 0:
                raise FileNotFoundError(result["output"])
            return str(result["output"])

        async def write_container_file(args: Dict[str, Any]) -> str:
            path = str(args.get("path", ""))
            if not path.startswith("/"):
                raise ValueError("Container file path must be absolute")
            content = str(args.get("content", "")).encode()
            result = await _docker(
                ["exec", "-i", container, "sh", "-lc", "mkdir -p -- \"$(dirname \"$1\")\" && cat > \"$1\"", "sh", path],
                stdin=content,
            )
            if result["exit_code"] != 0:
                raise OSError(result["output"])
            return path

        self.register(
            "run_container_command", "Run a shell command inside the official benchmark container.",
            '{"command":"..."}', run_container_command,
        )
        self.register(
            "read_container_file", "Read an absolute file path inside the official benchmark container.",
            '{"path":"/workspace/file"}', read_container_file,
        )
        self.register(
            "write_container_file", "Write an absolute file path inside the official benchmark container.",
            '{"path":"/workspace/file","content":"..."}', write_container_file,
        )

    def register_search_tool(self, settings: SearchSettings, *, model_name: str = "") -> None:
        """Expose a bounded search_web tool backed by one configured provider.

        Deliberately provider-agnostic (see SearchSettings's own docstring
        for why): "tavily" is a standalone search-for-agents HTTP API, no
        model call involved. "openrouter" is different in kind, not just in
        endpoint -- OpenRouter has no standalone search endpoint; its web
        search only exists as a "plugins" addition to a real chat
        completion request (confirmed against OpenRouter's own web-search
        docs), so this branch makes one real completion call per search,
        using model_name (this deployment's already-configured, already
        measured model) rather than inventing an unvetted second model --
        and it costs real money with no free quota either way. All three
        branches return the same normalized shape ({title, url, snippet})
        so agent code -- and any harness that evolves this call -- has one
        stable contract regardless of which provider a deployment's key is
        for.
        """
        provider = str(settings.provider or "").strip().lower()
        api_key = settings.api_key()
        if not api_key:
            raise RuntimeError(
                f"search.provider={provider!r} is configured without a resolvable API key "
                f"(set the {settings.api_key_env} environment variable)"
            )
        configured_max = max(1, min(10, int(settings.max_results)))

        def _bounded_count(args: Dict[str, Any]) -> int:
            requested = args.get("max_results")
            try:
                requested_int = int(requested) if requested is not None else configured_max
            except (TypeError, ValueError):
                requested_int = configured_max
            return max(1, min(configured_max, requested_int))

        def _normalize(items: Iterable[Any], *, title_key: str, url_key: str, snippet_key: str) -> List[Dict[str, str]]:
            normalized: List[Dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized.append(
                    {
                        "title": str(item.get(title_key, ""))[:300],
                        "url": str(item.get(url_key, ""))[:2000],
                        "snippet": str(item.get(snippet_key, ""))[:1000],
                    }
                )
            return normalized

        def _request(
            url: str, *, method: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None,
            json_body: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _validate_public_http_endpoint(url)
            full_url = f"{url}?{urlencode(params)}" if params else url
            data = None
            if json_body is not None:
                data = json.dumps(json_body).encode("utf-8")
                headers = {**headers, "Content-Type": "application/json"}
            request = urllib.request.Request(full_url, data=data, headers=headers, method=method)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=20) as response:
                return json.loads(response.read(500_000).decode(errors="replace"))

        def _run(query: str, count: int) -> List[Dict[str, str]]:
            if provider == "tavily":
                body = {"api_key": api_key, "query": query, "max_results": count, "search_depth": "basic"}
                result = _request(settings.base_url or "https://api.tavily.com/search", method="POST", headers={}, json_body=body)
                items = result.get("results", []) if isinstance(result, dict) else []
                return _normalize(items, title_key="title", url_key="url", snippet_key="content")
            if provider == "brave":
                result = _request(
                    settings.base_url or "https://api.search.brave.com/res/v1/web/search",
                    method="GET",
                    headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                    params={"q": query, "count": count},
                )
                web = result.get("web") if isinstance(result, dict) else None
                items = web.get("results", []) if isinstance(web, dict) else []
                return _normalize(items, title_key="title", url_key="url", snippet_key="description")
            if provider == "openrouter":
                body = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": query}],
                    "plugins": [{"id": "web", "max_results": count}],
                    "max_tokens": 1,
                }
                result = _request(
                    settings.base_url or "https://openrouter.ai/api/v1/chat/completions",
                    method="POST",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json_body=body,
                )
                choices = result.get("choices", []) if isinstance(result, dict) else []
                message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
                annotations = message.get("annotations") or []
                citations = [
                    annotation.get("url_citation", {})
                    for annotation in annotations
                    if isinstance(annotation, dict)
                ]
                return _normalize(citations[:count], title_key="title", url_key="url", snippet_key="content")
            raise ValueError(f"Unsupported search provider: {provider!r}")

        async def search_web(args: Dict[str, Any]) -> List[Dict[str, str]]:
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("search_web requires a non-empty query")
            count = _bounded_count(args)
            return await asyncio.to_thread(_run, query, count)

        self.register(
            "search_web",
            f"Search the web via the configured {provider} provider; returns up to max_results "
            "{title,url,snippet} items ranked by relevance.",
            '{"query":"...","max_results":5}',
            search_web,
        )

    async def call(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        allowed: Optional[Iterable[str]] = None,
    ) -> Any:
        if allowed is not None and name not in set(allowed):
            raise PermissionError(f"Tool is not allowed for this task: {name}")
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        value = tool.handler(dict(arguments or {}))
        return await value if inspect.isawaitable(value) else value

    def _path(self, raw: Any) -> Path:
        candidate = (self.workspace / str(raw or ".")).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise PermissionError(f"Path escapes the workspace: {raw}")
        return candidate

    def _read_path(self, raw: Any) -> Path:
        candidate = (self.workspace / str(raw or ".")).resolve()
        roots = [self.workspace, *self.allowed_read_paths]
        if not any(candidate == root or root in candidate.parents for root in roots):
            raise PermissionError(f"Read path escapes the benchmark scope: {raw}")
        return candidate

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace))
        except ValueError:
            return str(path)

    def _register_builtins(self) -> None:
        def list_files(args: Dict[str, Any]) -> Any:
            root = self._read_path(args.get("path", "."))
            if not root.exists():
                return []
            if root.is_file():
                return [self._display_path(root)]
            return [
                self._display_path(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ][:2000]

        def read_file(args: Dict[str, Any]) -> str:
            path = self._read_path(args.get("path"))
            text = path.read_text(encoding="utf-8", errors="replace")
            offset = max(0, int(args.get("offset", 0) or 0))
            max_chars = max(1, min(100_000, int(args.get("max_chars", 30_000) or 30_000)))
            chunk = text[offset : offset + max_chars]
            end = offset + len(chunk)
            if end < len(text):
                chunk += (
                    f"\n[TRUNCATED: showing characters {offset}-{end} of {len(text)}; "
                    f"call read_file again with offset={end}]"
                )
            return chunk

        def write_file(args: Dict[str, Any]) -> str:
            path = self._path(args.get("path"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("content", "")), encoding="utf-8")
            return str(path.relative_to(self.workspace))

        def append_file(args: Dict[str, Any]) -> str:
            path = self._path(args.get("path"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(str(args.get("content", "")))
            return str(path.relative_to(self.workspace))

        def edit_file(args: Dict[str, Any]) -> str:
            path = self._path(args.get("path"))
            old = str(args.get("old", ""))
            new = str(args.get("new", ""))
            text = path.read_text(encoding="utf-8")
            count = text.count(old)
            if not old or count != 1:
                raise ValueError(f"edit_file requires old text to occur exactly once; found {count}")
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return str(path.relative_to(self.workspace))

        async def run_command(args: Dict[str, Any]) -> Dict[str, Any]:
            command = str(args.get("command", ""))
            if not command:
                raise ValueError("run_command requires command")
            if "\x00" in command or re.search(
                r"(?:^|[/\s])\.\.(?:/|$)|~[/]|\$HOME|\$\{HOME\}|\$\(|`",
                command,
            ):
                raise PermissionError("Command attempts to escape the scoped workspace")
            if not self.allow_shell_network and re.search(
                r"(?:^|\s)(?:curl|wget|nc|ncat|ssh|scp|sftp|ftp|telnet)\b", command
            ):
                raise PermissionError("Shell network access is disabled; use the protected web tool")
            if re.search(
                r"(?:^|\s)(?:pip\d*\s+install|python\d*\s+-m\s+pip\s+install|uv\s+(?:add|pip)|conda\s+install|npm\s+install|apt(?:-get)?\s+install|brew\s+install)\b",
                command,
            ):
                raise PermissionError("Dependency installation is not allowed inside an agent run")
            permitted_roots = [self.workspace, *self.allowed_read_paths]
            for raw_path in re.findall(r"(?<![A-Za-z0-9_])(/[A-Za-z0-9_.\-/]+)", command):
                path = Path(raw_path).resolve()
                if str(path) == "/dev/null":
                    continue
                if not any(path == root or root in path.parents for root in permitted_roots):
                    raise PermissionError(f"Absolute path is outside the benchmark scope: {raw_path}")
            environment = {
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "KUBECONFIG"}
            }
            runtime_home = self.workspace / ".runtime_home"
            runtime_tmp = self.workspace / ".runtime_tmp"
            runtime_home.mkdir(exist_ok=True)
            runtime_tmp.mkdir(exist_ok=True)
            environment.update({"HOME": str(runtime_home), "TMPDIR": str(runtime_tmp)})
            argv = ["sh", "-c", command]
            sandbox_executable = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
            if sandbox_executable:
                def literal(path: Path | str) -> str:
                    return str(path).replace("\\", "\\\\").replace('"', '\\"')

                readable = [self.workspace, *self.allowed_read_paths]
                read_rules = " ".join(f'(subpath "{literal(path)}")' for path in readable)
                network_rule = "(allow network*)" if self.allow_shell_network else ""
                profile = (
                    '(version 1) (deny default) (allow process*) '
                    '(allow sysctl-read) (allow mach-lookup) '
                    '(allow file-read*) '
                    '(deny file-read* (subpath "/Users") (subpath "/home") '
                    '(subpath "/root") (subpath "/tmp") (subpath "/private/tmp") '
                    '(subpath "/private/var/folders")) '
                    f'(allow file-read* {read_rules}) '
                    f'(allow file-write* (subpath "{literal(self.workspace)}") (literal "/dev/null")) '
                    f'{network_rule}'
                )
                argv = [sandbox_executable, "-p", profile, "sh", "-c", command]
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self.command_timeout
                )
            except asyncio.TimeoutError:
                await _kill_and_drain(proc)
                raise TimeoutError(f"Command exceeded {self.command_timeout}s")
            except asyncio.CancelledError:
                await _kill_and_drain(proc)
                raise
            return {
                "exit_code": proc.returncode,
                "output": stdout.decode(errors="replace")[-100_000:],
            }

        async def fetch_url(args: Dict[str, Any]) -> str:
            url = str(args.get("url", ""))
            max_chars = max(1, min(100_000, int(args.get("max_chars", 30_000) or 30_000)))
            def validate_public_url(candidate: str) -> None:
                if not candidate.startswith(("https://", "http://")):
                    raise ValueError("fetch_url requires an HTTP(S) URL")
                host = urlparse(candidate).hostname
                if not host or host.lower() == "localhost":
                    raise PermissionError("Local network URLs are not allowed")
                for address in socket.getaddrinfo(host, None):
                    if not ip_address(address[4][0]).is_global:
                        raise PermissionError("Private and local network URLs are not allowed")

            validate_public_url(url)

            def _fetch() -> str:
                class SafeRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, headers, newurl):
                        validate_public_url(newurl)
                        return super().redirect_request(req, fp, code, msg, headers, newurl)

                request = urllib.request.Request(url, headers={"User-Agent": "foh-agent/1.0"})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), SafeRedirect())
                with opener.open(request, timeout=30) as response:
                    return response.read(max_chars).decode(errors="replace")

            return await asyncio.to_thread(_fetch)

        self.register("list_files", "List files under the scoped workspace.", '{"path":"relative/path"}', list_files)
        self.register(
            "read_file",
            "Read a bounded UTF-8 chunk from a scoped file; continue with offset when marked truncated.",
            '{"path":"relative/path","offset":0,"max_chars":30000}',
            read_file,
        )
        self.register("write_file", "Write a file in the scoped workspace.", '{"path":"relative/path","content":"..."}', write_file)
        self.register("append_file", "Append one chunk to a file in the scoped workspace.", '{"path":"relative/path","content":"..."}', append_file)
        self.register("edit_file", "Replace one exact text occurrence in a workspace file.", '{"path":"relative/path","old":"...","new":"..."}', edit_file)
        self.register("run_command", "Run a shell command in the scoped workspace.", '{"command":"..."}', run_command)
        self.register(
            "fetch_url",
            "Fetch a bounded HTTP(S) response when internet use is allowed.",
            '{"url":"https://...","max_chars":30000}',
            fetch_url,
        )


def prepare_tool_registry(
    context_value: RuntimeContext,
    app_config: AppConfig,
    *,
    tools: Optional[ToolRegistry] = None,
    extra_scoped_read_metadata_keys: Iterable[str] = (),
    mcp_registrar: Optional[Callable[[ToolRegistry, List[Dict[str, Any]]], None]] = None,
) -> ToolRegistry:
    """Merge skills, create the workspace, and build (or reuse) a scoped registry.

    Shared setup that client.solve_task and agent_runtime.evolution.evolve_agent
    both need: merge context_value.skills with app_config.skill_paths, create the
    workspace, and — unless the caller already supplied ``tools`` — build a
    ToolRegistry scoped to read-only attachment paths (plus any
    ``extra_scoped_read_metadata_keys``, e.g. a benchmark's own repository or
    kubeconfig path) and register container/MCP tools when configured.

    ``mcp_registrar`` is injected (normally ``agent_runtime.mcp.register_mcp_servers``)
    rather than imported here: mcp.py already imports ToolRegistry from this
    module, so importing it back would be circular.
    """
    context_value.skills = list(dict.fromkeys([*context_value.skills, *app_config.skill_paths]))
    context_value.workspace.mkdir(parents=True, exist_ok=True)
    if app_config.search.provider:
        # The external Hermes process cannot see this in-process registry.
        # Preserve only the public configuration; hermes.py transfers the
        # credential separately through a one-use environment variable.
        context_value.metadata["_external_search"] = app_config.search.public_dict()
    if tools is not None:
        return tools
    scoped_read_paths = [Path(value) for value in context_value.attachments if Path(value).exists()]
    for key in extra_scoped_read_metadata_keys:
        value = context_value.metadata.get(key)
        if value and Path(str(value)).exists():
            scoped_read_paths.append(Path(str(value)))
    registry = ToolRegistry(
        context_value.workspace,
        command_timeout=app_config.limits.max_command_seconds,
        allowed_read_paths=scoped_read_paths,
        allow_shell_network=bool(context_value.metadata.get("kubeconfig")),
    )
    container = context_value.metadata.get("docker_container")
    if isinstance(container, str) and container:
        registry.register_container_tools(container, str(context_value.metadata.get("docker_host") or "") or None)
    if app_config.search.provider:
        registry.register_search_tool(app_config.search, model_name=app_config.model.name)
    if mcp_registrar is not None:
        mcp_servers = context_value.metadata.get("mcp_servers", list(app_config.mcp_servers))
        if isinstance(mcp_servers, list):
            # Preserve the protected declaration for the external Hermes
            # subprocess. It is deliberately kept under a private metadata
            # key so hermes.py can omit command lines from the model prompt.
            if "mcp_servers" not in context_value.metadata and mcp_servers:
                context_value.metadata["_external_mcp_servers"] = [
                    dict(value) for value in mcp_servers if isinstance(value, dict)
                ]
            mcp_registrar(registry, [value for value in mcp_servers if isinstance(value, dict)])
    return registry
