#!/usr/bin/env python3
"""Install the exact Hermes Agent release used by FoH into .hermes/.

The operation is intentionally explicit and idempotent.  It never replaces a
dirty checkout and writes install.json only after the pinned revision imports
successfully in its Python 3.11-3.13 virtual environment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "third_party" / "hermes-agent.lock.json"
FOH_REQUIREMENTS = ROOT / "requirements.txt"


def run(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            + completed.stdout[-5000:]
        )
    return completed.stdout.strip()


def checkout_release(source: Path, lock: dict[str, object]) -> str:
    """Materialize and verify the exact, fully-hydrated release checkout."""
    upstream = str(lock["upstream"])
    release = str(lock["release"])
    revision = str(lock["revision"])
    incomplete_empty_checkout = False
    if source.exists():
        if not (source / ".git").exists():
            raise RuntimeError(f"Refusing to replace non-git directory: {source}")
        # A failed legacy ``git clone --no-checkout`` can leave exactly one
        # entry: .git/.  Git reports every indexed path as deleted even
        # though no user working-tree content ever existed.  This is the one
        # provably safe dirty state we can repair automatically; any other
        # entry could be user data and keeps the refusal below.
        incomplete_empty_checkout = not any(
            entry.name != ".git" for entry in source.iterdir()
        )
        dirty = run("git", "status", "--porcelain", cwd=source)
        if dirty and not incomplete_empty_checkout:
            raise RuntimeError(f"Hermes checkout has local changes; refusing to overwrite them:\n{dirty}")
        origin = run("git", "remote", "get-url", "origin", cwd=source)
        if origin.rstrip("/") != upstream.rstrip("/"):
            raise RuntimeError(
                f"Hermes checkout origin mismatch: expected {upstream}, found {origin}"
            )
        # ``--refetch --no-filter`` is intentional.  Earlier bootstrap
        # versions made a blob-filtered partial clone, which could die during
        # checkout when Git attempted hundreds of lazy network fetches.  A
        # refetch hydrates the one pinned release as a normal shallow clone
        # and also repairs an otherwise-clean partial checkout left by that
        # implementation.
        run(
            "git", "fetch", "--force", "--no-tags", "--no-filter",
            "--refetch", "--depth", "1", "origin",
            f"+refs/tags/{release}:refs/tags/{release}",
            cwd=source,
        )
    else:
        run(
            "git", "clone", "--no-checkout", "--no-tags", "--depth", "1",
            "--branch", release, upstream, str(source),
        )
    tag_revision = run(
        "git", "rev-parse", f"refs/tags/{release}^{{commit}}", cwd=source
    )
    if tag_revision != revision:
        raise RuntimeError(
            f"release tag verification failed: {release} resolves to {tag_revision}, "
            f"expected {revision}"
        )
    checkout_arguments = ["git", "checkout"]
    if incomplete_empty_checkout:
        checkout_arguments.append("--force")
    checkout_arguments.extend(("--detach", revision))
    run(*checkout_arguments, cwd=source)
    installed_revision = run("git", "rev-parse", "HEAD", cwd=source)
    if installed_revision != revision:
        raise RuntimeError(
            f"revision verification failed: expected {revision}, found {installed_revision}"
        )
    return installed_revision


def _python_for_venv(requested: str) -> str:
    """Resolve a Python selector when uv is unavailable (common on HPC)."""
    candidates = [requested]
    if requested and requested[0].isdigit():
        candidates.extend((f"python{requested}", "python3"))
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        f"Could not resolve Python {requested!r}; install Python 3.11-3.13 or uv"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-root", default=str(ROOT / ".hermes"))
    parser.add_argument("--python", default="3.12", help="Python version/path for uv venv (must be >=3.11,<3.14).")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    install_root = Path(args.install_root).expanduser().resolve()
    source = install_root / "hermes-agent"
    venv = install_root / "venv"
    venv_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    install_root.mkdir(parents=True, exist_ok=True)

    installed_revision = checkout_release(source, lock)

    uv = shutil.which("uv")
    if not venv_python.is_file():
        if uv:
            run(uv, "venv", str(venv), "--python", str(args.python))
        else:
            # Editable installs are explicitly supported upstream.  A plain
            # stdlib venv + pip keeps shared clusters usable when uv is not
            # installed on login/compute nodes.
            run(_python_for_venv(str(args.python)), "-m", "venv", str(venv))
    version_text = run(str(venv_python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    major, minor = (int(value) for value in version_text.split(".")[:2])
    if (major, minor) < (3, 11) or (major, minor) >= (3, 14):
        raise RuntimeError(f"Hermes requires Python >=3.11,<3.14; selected {version_text}")

    # Hermes' terminal tool deliberately uses this virtualenv's ``python``.
    # Install FoH's document/data runtime dependencies into the same isolated
    # environment; otherwise spreadsheet/PDF benchmark tasks fail imports even
    # when the outer FoH environment has those packages installed.
    if uv:
        run(
            uv,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "-r",
            str(FOH_REQUIREMENTS),
        )
        run(
            uv,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "-e",
            f"{source}[mcp]",
        )
    else:
        run(str(venv_python), "-m", "pip", "install", "-r", str(FOH_REQUIREMENTS))
        run(str(venv_python), "-m", "pip", "install", "-e", f"{source}[mcp]")
    run(
        str(venv_python),
        "-c",
        f"import sys; sys.path.insert(0, {str(source)!r}); import mcp; from run_agent import AIAgent; print(AIAgent.__name__)",
    )
    manifest = {
        **lock,
        "revision": installed_revision,
        "source_dir": str(source),
        "python_executable": str(venv_python),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = install_root / "install.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(install_root / "install.json")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
