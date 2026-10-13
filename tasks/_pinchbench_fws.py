"""Isolated fake-web-service runtime for the PinchBench integration tasks.

The upstream PinchBench runner mutates ``os.environ`` and controls one global
``fws`` daemon.  That is unsafe once FoH evaluates cases concurrently: one
case can stop another case's server or inherit its mutable fake account.  This
module instead owns one foreground daemon, data directory, and pair of ports
per case and returns explicit runtime bindings for the Hermes subprocess.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from agent_runtime.contracts import RuntimeContext


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS_ROOT = REPO_ROOT / ".pinchbench-tools"
TOOLS_LOCK_PATH = REPO_ROOT / "third_party" / "pinchbench-tools.lock.json"
BOOTSTRAP_COMMAND = "python scripts/bootstrap_pinchbench_tools.py"

# These are the four tasks in the checked-in PinchBench release whose
# frontmatter declares npm:@juppytt/fws.  Detection below remains metadata
# driven so a future task with the same prerequisite gets the safe runtime.
PINCHBENCH_FWS_TASK_IDS = frozenset(
    {
        "task_gh_issue_triage",
        "task_gws_email_triage",
        "task_gws_cross_service",
        "task_gws_task_management",
    }
)

_DISCOVERY_FILES = (
    "gmail_v1.json",
    "calendar_v3.json",
    "drive_v3.json",
    "tasks_v1.json",
    "sheets_v4.json",
    "people_v1.json",
)
_FWS_ENVIRONMENT_NAMES = frozenset(
    {
        "GOOGLE_WORKSPACE_CLI_CONFIG_DIR",
        "GOOGLE_WORKSPACE_CLI_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "https_proxy",
        "http_proxy",
        "SSL_CERT_FILE",
        "GH_TOKEN",
        "GH_REPO",
        "NO_PROXY",
        "no_proxy",
    }
)


class FwsPrerequisiteError(RuntimeError):
    """The pinned PinchBench integration toolchain is absent or invalid."""


@dataclass(frozen=True)
class FwsRequirements:
    cli_names: tuple[str, ...]


@dataclass(frozen=True)
class FwsToolchain:
    root: Path
    node: Path
    fws: Path
    fws_typescript: Path
    tsx: Path
    gws: Path
    gh: Path
    source_config: Path
    platform_key: str

    @classmethod
    def load(
        cls,
        root: str | Path = DEFAULT_TOOLS_ROOT,
        *,
        required_clis: Iterable[str] = ("gws", "gh"),
        verify_versions: bool = True,
    ) -> "FwsToolchain":
        install_root = Path(root).expanduser().resolve()
        manifest_path = install_root / "install.json"
        problems: list[str] = []
        try:
            lock_bytes = TOOLS_LOCK_PATH.read_bytes()
            lock = json.loads(lock_bytes.decode("utf-8"))
        except Exception as exc:
            raise FwsPrerequisiteError(
                f"PinchBench tool lock is missing or invalid at {TOOLS_LOCK_PATH}: {exc}"
            ) from exc
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            manifest = {}
            problems.append(f"install manifest missing at {manifest_path}")
        except Exception as exc:
            manifest = {}
            problems.append(f"install manifest is invalid: {exc}")

        expected_lock_digest = hashlib.sha256(lock_bytes).hexdigest()
        platform_key = current_platform_key()
        if manifest.get("lock_sha256") != expected_lock_digest:
            problems.append("install manifest does not match the checked-in tool lock")
        if manifest.get("platform") != platform_key:
            problems.append(
                f"installed platform is {manifest.get('platform')!r}, current platform is {platform_key!r}"
            )

        toolchain = cls(
            root=install_root,
            node=install_root / "node" / "bin" / "node",
            fws=install_root / "bin" / "fws",
            fws_typescript=(
                install_root
                / "fws"
                / "node_modules"
                / "@juppytt"
                / "fws"
                / "bin"
                / "fws.ts"
            ),
            tsx=(
                install_root
                / "fws"
                / "node_modules"
                / "tsx"
                / "dist"
                / "cli.mjs"
            ),
            gws=install_root / "bin" / ("gws.exe" if os.name == "nt" else "gws"),
            gh=install_root / "bin" / ("gh.exe" if os.name == "nt" else "gh"),
            source_config=install_root / "fws-source-config",
            platform_key=platform_key,
        )
        required_paths: Dict[str, Path] = {
            "node": toolchain.node,
            "fws": toolchain.fws,
            "fws entrypoint": toolchain.fws_typescript,
            "tsx entrypoint": toolchain.tsx,
            "fws discovery config": toolchain.source_config,
        }
        requested = tuple(sorted(set(str(value) for value in required_clis)))
        unknown = set(requested) - {"gws", "gh"}
        if unknown:
            problems.append(f"unsupported fws task CLI prerequisites: {sorted(unknown)}")
        if "gws" in requested:
            required_paths["gws"] = toolchain.gws
        if "gh" in requested:
            required_paths["gh"] = toolchain.gh
        for label, path in required_paths.items():
            if not path.exists():
                problems.append(f"{label} is missing at {path}")
            elif label in {"node", "fws", "gws", "gh"} and not os.access(path, os.X_OK):
                problems.append(f"{label} is not executable at {path}")
        missing_discovery = [
            name
            for name in _DISCOVERY_FILES
            if not (toolchain.source_config / "cache" / name).is_file()
        ]
        if missing_discovery:
            problems.append(
                "fws discovery cache is incomplete: " + ", ".join(missing_discovery)
            )
        if problems:
            raise FwsPrerequisiteError(
                "; ".join(problems)
                + f". Run `{BOOTSTRAP_COMMAND}` before selecting an fws PinchBench task."
            )
        if verify_versions:
            toolchain.verify_versions(requested, lock)
        return toolchain

    def _host_environment(self, *, home: Path, data_dir: Path) -> Dict[str, str]:
        environment = {
            name: os.environ[name]
            for name in ("LANG", "LC_ALL", "LC_CTYPE", "TERM")
            if os.environ.get(name)
        }
        environment.update(
            {
                "HOME": str(home),
                "FWS_DATA_DIR": str(data_dir),
                "GWS_SOURCE_CONFIG_DIR": str(self.source_config),
                "PATH": os.pathsep.join(
                    (str(self.node.parent), "/usr/local/bin", "/usr/bin", "/bin")
                ),
            }
        )
        return environment

    def verify_versions(self, required_clis: Sequence[str], lock: Mapping[str, Any]) -> None:
        expected = lock.get("versions") if isinstance(lock.get("versions"), dict) else {}
        commands: list[tuple[str, Sequence[str], str]] = [
            ("node", (str(self.node), "--version"), str(expected.get("node", ""))),
            ("fws", (str(self.fws), "--version"), str(expected.get("fws", ""))),
        ]
        if "gws" in required_clis:
            commands.append(("gws", (str(self.gws), "--version"), str(expected.get("gws", ""))))
        if "gh" in required_clis:
            commands.append(("gh", (str(self.gh), "--version"), str(expected.get("gh", ""))))
        env = self._host_environment(home=self.root, data_dir=self.root / "version-check")
        for label, command, expected_version in commands:
            try:
                completed = subprocess.run(
                    command,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
            except Exception as exc:
                raise FwsPrerequisiteError(
                    f"Could not execute pinned {label}: {exc}. Run `{BOOTSTRAP_COMMAND}`."
                ) from exc
            output = completed.stdout.strip()
            if completed.returncode != 0 or expected_version not in output:
                raise FwsPrerequisiteError(
                    f"Pinned {label} version check failed (expected {expected_version!r}, "
                    f"exit={completed.returncode}, output={output[:300]!r}). "
                    f"Run `{BOOTSTRAP_COMMAND}`."
                )


def current_platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    normalized_machine = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }.get(machine, machine)
    key = f"{system}-{normalized_machine}"
    try:
        lock = json.loads(TOOLS_LOCK_PATH.read_text(encoding="utf-8"))
        supported = set((lock.get("platforms") or {}).keys())
    except Exception:
        supported = set()
    if supported and key not in supported:
        raise FwsPrerequisiteError(
            f"Unsupported PinchBench tool platform {key!r}; supported: {sorted(supported)}"
        )
    return key


def fws_requirements(task: Any) -> Optional[FwsRequirements]:
    frontmatter = getattr(task, "frontmatter", None) or {}
    raw_prerequisites = frontmatter.get("prerequisites") or []
    prerequisites = [str(value).strip() for value in raw_prerequisites]
    if "npm:@juppytt/fws" not in prerequisites:
        return None
    cli_names = tuple(
        name
        for name in ("gws", "gh")
        if f"cli:{name}" in prerequisites
    )
    if not cli_names:
        raise FwsPrerequisiteError(
            f"Task {getattr(task, 'task_id', '<unknown>')!r} requires fws but declares no supported CLI"
        )
    return FwsRequirements(cli_names=cli_names)


def preflight_fws_tasks(
    tasks: Iterable[Any],
    *,
    tools_root: str | Path = DEFAULT_TOOLS_ROOT,
    verify_versions: bool = True,
) -> Optional[FwsToolchain]:
    """Fail before bootstrap/model calls when selected fws tasks cannot run."""
    required: set[str] = set()
    for task in tasks:
        requirements = fws_requirements(task)
        if requirements is not None:
            required.update(requirements.cli_names)
    if not required:
        return None
    return FwsToolchain.load(
        tools_root, required_clis=required, verify_versions=verify_versions
    )


def _allocate_ports() -> tuple[int, int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(2):
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            handle.bind(("127.0.0.1", 0))
            sockets.append(handle)
            ports.append(int(handle.getsockname()[1]))
    finally:
        for handle in sockets:
            handle.close()
    return ports[0], ports[1]


def _ready(server_port: int, proxy_port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server_port}/__fws/status", timeout=0.5
        ) as response:
            if int(response.status) != 200:
                return False
        with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.5):
            return True
    except OSError:
        return False


def _merge_no_proxy(existing: str = "") -> str:
    values = [value.strip() for value in existing.split(",") if value.strip()]
    for value in ("127.0.0.1", "localhost", "openrouter.ai", ".openrouter.ai"):
        if value not in values:
            values.append(value)
    return ",".join(values)


@dataclass(frozen=True)
class FwsRuntimeBindings:
    environment: Dict[str, str]
    read_only_paths: tuple[str, ...]
    writable_paths: tuple[str, ...]
    executable_paths: tuple[str, ...]


class FwsCaseSession:
    """Own one foreground fws daemon and its ephemeral mutable state."""

    def __init__(
        self,
        toolchain: FwsToolchain,
        *,
        task_id: str,
        cli_names: Sequence[str],
        process_factory: Callable[..., Any] = subprocess.Popen,
        ready_probe: Callable[[int, int], bool] = _ready,
        port_allocator: Callable[[], tuple[int, int]] = _allocate_ports,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        self.toolchain = toolchain
        self.task_id = str(task_id)
        self.cli_names = tuple(cli_names)
        self.process_factory = process_factory
        self.ready_probe = ready_probe
        self.port_allocator = port_allocator
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self._temporary: Optional[tempfile.TemporaryDirectory[str]] = None
        self._process: Any = None
        self._log_handle: Any = None

    def __enter__(self) -> FwsRuntimeBindings:
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_id)[:60] or "task"
        self._temporary = tempfile.TemporaryDirectory(prefix=f"foh-fws-{safe_task}-")
        root = Path(self._temporary.name).resolve()
        home = root / "home"
        data_dir = root / "data"
        home.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        host_environment = self.toolchain._host_environment(home=home, data_dir=data_dir)
        log_path = root / "server.log"
        self._log_handle = log_path.open("wb")
        try:
            info_path = data_dir / "server.json"
            overall_deadline = time.monotonic() + self.startup_timeout_seconds
            last_ports = (0, 0)
            # Closing the reservation sockets before fws binds leaves a tiny
            # unavoidable race. Retry with fresh OS-assigned ports when a
            # competing worker wins either port.
            for attempt in range(3):
                info_path.unlink(missing_ok=True)
                server_port, proxy_port = self.port_allocator()
                last_ports = (server_port, proxy_port)
                command = [
                    str(self.toolchain.node),
                    str(self.toolchain.tsx),
                    str(self.toolchain.fws_typescript),
                    "server",
                    "start",
                    "--foreground",
                    "--port",
                    str(server_port),
                    "--proxy-port",
                    str(proxy_port),
                ]
                self._process = self.process_factory(
                    command,
                    cwd=str(data_dir),
                    env=host_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                remaining_attempts = 3 - attempt
                attempt_budget = max(
                    1.0,
                    (overall_deadline - time.monotonic()) / remaining_attempts,
                )
                attempt_deadline = min(
                    overall_deadline, time.monotonic() + attempt_budget
                )
                while time.monotonic() < attempt_deadline:
                    if self._process.poll() is not None:
                        break
                    if info_path.is_file() and self.ready_probe(server_port, proxy_port):
                        return self._bindings(data_dir, proxy_port, info_path)
                    time.sleep(0.05)
                self._stop_process()
            evidence = ""
            try:
                self._log_handle.flush()
                evidence = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            except Exception:
                pass
            raise RuntimeError(
                f"fws failed to become ready for {self.task_id!r} on ports "
                f"{last_ports[0]}/{last_ports[1]} after 3 attempts: "
                f"{evidence or 'no server log'}"
            )
        except BaseException:
            self._cleanup()
            raise

    def _bindings(
        self, data_dir: Path, proxy_port: int, info_path: Path
    ) -> FwsRuntimeBindings:
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"fws wrote an invalid server manifest: {exc}") from exc
        config_dir = (data_dir / "config").resolve()
        bundle = Path(
            str(info.get("bundlePath") or data_dir / "certs" / "ca-bundle.crt")
        ).expanduser().resolve()
        for label, path in (("config", config_dir), ("CA bundle", bundle)):
            try:
                path.relative_to(data_dir)
            except ValueError as exc:
                raise RuntimeError(f"fws {label} escaped its per-case data directory: {path}") from exc
            if not path.exists():
                raise RuntimeError(f"fws {label} was not created: {path}")
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        no_proxy = _merge_no_proxy(os.environ.get("NO_PROXY", ""))
        environment = {
            "HTTPS_PROXY": proxy_url,
            "HTTP_PROXY": proxy_url,
            "https_proxy": proxy_url,
            "http_proxy": proxy_url,
            "SSL_CERT_FILE": str(bundle),
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        read_only_paths = [str(bundle)]
        writable_paths: list[str] = []
        if "gws" in self.cli_names:
            environment.update(
                {
                    "GOOGLE_WORKSPACE_CLI_CONFIG_DIR": str(config_dir),
                    "GOOGLE_WORKSPACE_CLI_TOKEN": "fake",
                }
            )
            # gws refreshes discovery/timezone caches under its config root.
            # This directory is already unique to the case, so expose it RW
            # without making the daemon's store or certificate key writable.
            writable_paths.append(str(config_dir))
        if "gh" in self.cli_names:
            environment.update(
                {"GH_TOKEN": "fake", "GH_REPO": "testuser/my-project"}
            )
        if not set(environment).issubset(_FWS_ENVIRONMENT_NAMES):
            raise AssertionError("FWS environment allowlist drifted")
        executables = tuple(
            str(getattr(self.toolchain, name).resolve()) for name in self.cli_names
        )
        return FwsRuntimeBindings(
            environment=environment,
            read_only_paths=tuple(read_only_paths),
            writable_paths=tuple(writable_paths),
            executable_paths=executables,
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._cleanup()
        return False

    def _cleanup(self) -> None:
        self._stop_process()
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        process_group = (
            int(process.pid)
            if os.name == "posix" and isinstance(getattr(process, "pid", None), int)
            else None
        )
        if process.poll() is None:
            try:
                if process_group is not None:
                    os.killpg(process_group, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    if process_group is not None:
                        os.killpg(process_group, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=3)
                except Exception:
                    pass
        # start_new_session makes the foreground Node process its group
        # leader. If it spawned a helper that survived the leader, reap that
        # process group too before deleting its working directory.
        if process_group is not None:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                pass
            else:
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._process = None


def runtime_scope_for_task(
    task: Any,
    *,
    tools_root: str | Path = DEFAULT_TOOLS_ROOT,
) -> Optional[Callable[[RuntimeContext], Any]]:
    requirements = fws_requirements(task)
    if requirements is None:
        return None
    task_id = str(getattr(task, "task_id", "task"))

    @contextmanager
    def scope(base: RuntimeContext) -> Iterator[RuntimeContext]:
        # Load inside the scope as a second line of defense for programmatic
        # users that bypass task_pinchbench.main's command-level preflight.
        toolchain = FwsToolchain.load(
            tools_root,
            required_clis=requirements.cli_names,
            verify_versions=False,
        )
        with FwsCaseSession(
            toolchain, task_id=task_id, cli_names=requirements.cli_names
        ) as bindings:
            yield replace(
                base,
                environment={**base.environment, **bindings.environment},
                read_only_paths=[*base.read_only_paths, *bindings.read_only_paths],
                writable_paths=[*base.writable_paths, *bindings.writable_paths],
                executable_paths=[*base.executable_paths, *bindings.executable_paths],
            )

    return scope


__all__ = [
    "BOOTSTRAP_COMMAND",
    "DEFAULT_TOOLS_ROOT",
    "FwsCaseSession",
    "FwsPrerequisiteError",
    "FwsRequirements",
    "FwsRuntimeBindings",
    "FwsToolchain",
    "PINCHBENCH_FWS_TASK_IDS",
    "fws_requirements",
    "preflight_fws_tasks",
    "runtime_scope_for_task",
]
