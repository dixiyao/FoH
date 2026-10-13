"""Optional git distribution of evolved harnesses (one branch per iteration).

When ``harness_git.enabled`` is true in ``config.yaml``, the federation
controller publishes the active ``server_agent/`` and ``client_agent/``
overlays (plus the shared ``insight_library.json``) to a git repository after
every evolution step:

* bootstrap (before round 1)  -> branch ``<prefix>000``
* round N activation          -> branch ``<prefix>NNN``

Each branch is created on top of the previous iteration's branch, so
``git log`` of the repository is the evolution history and ``git diff
iteration-004 iteration-005`` is the round-5 harness change.  The server
records what it pushed in ``harness_branches.json`` under the experiment
output directory and in ``round_NNN/harness_branch.json``; clients pull the
branch named there instead of reading the server's filesystem:

    python -m agent_runtime.harness_git pull --repo <url-or-path> \
        --branch iteration-005 --dest runs/pulled_harness
    python client.py --task "..." --agent-version runs/pulled_harness/client_agent

``client.py --harness-repo <url> --harness-branch <name>`` does the same pull
inline.  The repository may be any URL git can push to (a bare path on a
shared filesystem, SSH or HTTPS); it is created as a bare repository when
``repo`` is a local path that does not exist yet.  Only overlay code and the
library are committed: never credentials, task data, traces, or results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROLE_DIRS = ("client_agent", "server_agent")
LIBRARY_FILE = "insight_library.json"
ITERATION_FILE = "ITERATION.json"
_SKIP_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}


@dataclass(frozen=True)
class HarnessGitSettings:
    """``harness_git`` section of config.yaml."""

    enabled: bool = False
    repo: str = ""
    branch_prefix: str = "iteration-"
    push: bool = True
    author_name: str = "FoH server"
    author_email: str = "foh-server@localhost"

    def validate(self) -> None:
        if self.enabled and not str(self.repo).strip():
            raise ValueError("harness_git.repo is required when harness_git.enabled is true")
        if not self.branch_prefix or any(ch.isspace() for ch in self.branch_prefix):
            raise ValueError("harness_git.branch_prefix must be a non-empty name without spaces")

    def branch_name(self, iteration: int) -> str:
        return f"{self.branch_prefix}{int(iteration):03d}"


def _git(args: list[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> str:
    full_env = dict(os.environ)
    full_env.update(env or {})
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, env=full_env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd} (exit {proc.returncode}):\n{proc.stdout}{proc.stderr}"
        )
    return proc.stdout.strip()


def _copy_tree(source: Path, target: Path) -> None:
    """Replace ``target`` with a clean copy of ``source`` (no caches)."""
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(*_SKIP_NAMES, "*.pyc"),
        symlinks=False,
    )


def _is_local_path(repo: str) -> bool:
    return "://" not in repo and not repo.startswith("git@") and ":" not in repo.split("/")[0]


def ensure_remote(settings: HarnessGitSettings) -> str:
    """Create a bare repository when ``repo`` is a local path that does not
    exist yet, so a shared filesystem works without any manual setup."""
    repo = str(settings.repo)
    if _is_local_path(repo):
        path = Path(repo).expanduser()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            _git(["init", "--bare", "--quiet", str(path)], cwd=path.parent)
        return str(path.resolve())
    return repo


def _working_clone(settings: HarnessGitSettings, output_dir: Path) -> Path:
    """The server's working clone under the experiment directory."""
    remote = ensure_remote(settings)
    clone = output_dir / "harness_repo"
    if not (clone / ".git").is_dir():
        clone.mkdir(parents=True, exist_ok=True)
        _git(["init", "--quiet"], cwd=clone)
        _git(["remote", "add", "origin", remote], cwd=clone)
    else:
        _git(["remote", "set-url", "origin", remote], cwd=clone)
    _git(["config", "user.name", settings.author_name], cwd=clone)
    _git(["config", "user.email", settings.author_email], cwd=clone)
    return clone


def _remote_has_branch(clone: Path, branch: str) -> bool:
    try:
        out = _git(["ls-remote", "--heads", "origin", branch], cwd=clone)
    except RuntimeError:
        return False
    return bool(out.strip())


def publish_iteration(
    settings: HarnessGitSettings,
    *,
    output_dir: str | Path,
    iteration: int,
    client_agent: str | Path,
    server_agent: str | Path,
    insight_library: Dict[str, str],
    note: str = "",
) -> Dict[str, Any]:
    """Commit the active overlays as branch ``<prefix>NNN`` and push it.

    Idempotent per iteration: re-running for an iteration whose branch
    already exists on the remote re-checks it out and amends nothing unless
    the content changed.
    """
    settings.validate()
    root = Path(output_dir).expanduser().resolve()
    clone = _working_clone(settings, root)
    branch = settings.branch_name(iteration)
    parent = settings.branch_name(iteration - 1) if iteration > 0 else None

    _git(["fetch", "--quiet", "origin"], cwd=clone)
    if _remote_has_branch(clone, branch):
        _git(["checkout", "--quiet", "-B", branch, f"origin/{branch}"], cwd=clone)
    elif parent and _remote_has_branch(clone, parent):
        _git(["checkout", "--quiet", "-B", branch, f"origin/{parent}"], cwd=clone)
    else:
        # First branch of this repository (or a parent that was never
        # published): start an orphan history for this iteration.
        _git(["checkout", "--quiet", "--orphan", branch], cwd=clone)
        _git(["rm", "-rfq", "--ignore-unmatch", "."], cwd=clone)

    for role, source in (("client_agent", client_agent), ("server_agent", server_agent)):
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"{role} directory not found: {source_path}")
        _copy_tree(source_path, clone / role)
    (clone / LIBRARY_FILE).write_text(
        json.dumps(insight_library, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record = {
        "iteration": int(iteration),
        "branch": branch,
        "parent_branch": parent,
        "client_agent_source": str(Path(client_agent).expanduser().resolve()),
        "server_agent_source": str(Path(server_agent).expanduser().resolve()),
        "library_entries": len(insight_library),
        "note": note,
    }
    # The committed record carries no timestamp so that re-publishing an
    # unchanged iteration produces no new commit (idempotent re-runs/resumes).
    (clone / ITERATION_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["published_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (clone / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")

    _git(["add", "-A"], cwd=clone)
    status = _git(["status", "--porcelain"], cwd=clone)
    if status:
        message = f"FoH iteration {iteration:03d}: evolved client_agent + server_agent"
        if note:
            message += f"\n\n{note}"
        _git(["commit", "--quiet", "-m", message], cwd=clone)
    commit = _git(["rev-parse", "HEAD"], cwd=clone)
    if settings.push:
        _git(["push", "--quiet", "-u", "origin", branch], cwd=clone)
    record.update({"commit": commit, "repo": ensure_remote(settings), "pushed": bool(settings.push)})

    # Server-side bookkeeping clients can read: one file per round plus a
    # running index of every published branch.
    index_path = root / "harness_branches.json"
    index: Dict[str, Any] = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}
    index[str(int(iteration))] = record
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return record


def pull_iteration(
    repo: str,
    branch: str,
    dest: str | Path,
) -> Path:
    """Client side: check out ``branch`` of ``repo`` into ``dest`` and return
    ``dest``.  ``dest/client_agent`` is then a runnable client overlay and
    ``dest/server_agent`` the server overlay that produced it."""
    target = Path(dest).expanduser().resolve()
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"{target} exists and is not a git clone; choose an empty --dest")
        _git(["clone", "--quiet", "--branch", branch, "--single-branch", repo, str(target)],
             cwd=target.parent)
    else:
        _git(["remote", "set-url", "origin", repo], cwd=target)
        # Explicit refspec: a --single-branch clone only tracks the branch it
        # was created with, so a plain fetch would not create origin/<branch>.
        _git(["fetch", "--quiet", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], cwd=target)
        _git(["checkout", "--quiet", "-B", branch, f"origin/{branch}"], cwd=target)
    for role in ROLE_DIRS:
        if not (target / role).is_dir():
            raise RuntimeError(f"branch {branch} of {repo} has no {role}/ directory")
    return target


def latest_branch(repo: str, prefix: str = "iteration-") -> Optional[str]:
    """Highest-numbered ``<prefix>NNN`` branch on ``repo`` (or None)."""
    proc = subprocess.run(
        ["git", "ls-remote", "--heads", repo], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-remote {repo} failed:\n{proc.stderr}")
    best: Optional[tuple[int, str]] = None
    for line in proc.stdout.splitlines():
        name = line.split("refs/heads/", 1)[-1].strip() if "refs/heads/" in line else ""
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            number = int(name[len(prefix):])
            if best is None or number > best[0]:
                best = (number, name)
    return best[1] if best else None


def settings_from_mapping(raw: Dict[str, Any]) -> HarnessGitSettings:
    unknown = set(raw) - set(HarnessGitSettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown harness_git settings: {sorted(unknown)}")
    settings = HarnessGitSettings(**raw)
    settings.validate()
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull or list FoH harness iteration branches.")
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull", help="check out one iteration branch")
    pull.add_argument("--repo", required=True, help="git URL or path the server publishes to")
    pull.add_argument("--branch", default=None, help="branch name (default: latest iteration-NNN)")
    pull.add_argument("--dest", required=True, help="local clone directory")
    pull.add_argument("--branch-prefix", default="iteration-")
    latest = sub.add_parser("latest", help="print the latest iteration branch name")
    latest.add_argument("--repo", required=True)
    latest.add_argument("--branch-prefix", default="iteration-")
    args = parser.parse_args()
    if args.command == "latest":
        name = latest_branch(args.repo, args.branch_prefix)
        if name is None:
            raise SystemExit(f"no {args.branch_prefix}NNN branch on {args.repo}")
        print(name)
        return
    branch = args.branch or latest_branch(args.repo, args.branch_prefix)
    if branch is None:
        raise SystemExit(f"no {args.branch_prefix}NNN branch on {args.repo}")
    dest = pull_iteration(args.repo, branch, args.dest)
    print(json.dumps({"branch": branch, "dest": str(dest), "client_agent": str(dest / "client_agent")}, indent=2))


if __name__ == "__main__":
    main()
