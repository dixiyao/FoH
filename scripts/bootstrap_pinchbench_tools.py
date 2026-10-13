#!/usr/bin/env python3
"""Install the exact local CLI stack required by PinchBench fws tasks.

The install is checksum-verified, platform-specific, and staged atomically in
``.pinchbench-tools``.  Node, gws, gh, fws, and fws's complete npm dependency
graph are pinned by files under ``third_party/``.  No global package manager or
administrator access is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "third_party" / "pinchbench-tools.lock.json"
FWS_PACKAGE_PATH = ROOT / "third_party" / "pinchbench-fws-package.json"
FWS_PACKAGE_LOCK_PATH = ROOT / "third_party" / "pinchbench-fws-package-lock.json"
DEFAULT_INSTALL_ROOT = ROOT / ".pinchbench-tools"


def _fws_compatibility_launcher(install_root: Path) -> Path:
    return (
        install_root
        / "fws"
        / "node_modules"
        / "@juppytt"
        / "fws"
        / "node_modules"
        / ".bin"
        / "tsx"
    )


def _write_fws_compatibility_launcher(install_root: Path) -> None:
    """Supply the nested tsx path assumed by fws 0.4.4's server command."""

    launcher = _fws_compatibility_launcher(install_root)
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        "#!/bin/sh\n"
        "tool_root=$(CDPATH= cd -- \"$(dirname -- \"$0\")/../../../../../..\" && pwd)\n"
        "exec \"$tool_root/node/bin/node\" "
        "\"$tool_root/fws/node_modules/tsx/dist/cli.mjs\" \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
DISCOVERY_DOCUMENTS = {
    "gmail_v1.json": "https://www.googleapis.com/discovery/v1/apis/gmail/v1/rest",
    "calendar_v3.json": "https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest",
    "drive_v3.json": "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest",
    "tasks_v1.json": "https://www.googleapis.com/discovery/v1/apis/tasks/v1/rest",
    "sheets_v4.json": "https://www.googleapis.com/discovery/v1/apis/sheets/v4/rest",
    "people_v1.json": "https://www.googleapis.com/discovery/v1/apis/people/v1/rest",
}


def platform_key() -> str:
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return f"{platform.system().lower()}-{machine}"


def run(*arguments: str, cwd: Path, environment: Mapping[str, str]) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=str(cwd),
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            + completed.stdout[-5000:]
        )
    return completed.stdout.strip()


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "FoH-PinchBench-bootstrap/1"})


def download_checked(url: str, destination: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(_request(url), timeout=60) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {url}: expected {expected_sha256}, observed {observed}"
        )


def _safe_member_name(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError(f"archive member escapes extraction root: {name!r}")
    return value


def _relative_path_stays_inside(value: PurePosixPath) -> bool:
    depth = 0
    for part in value.parts:
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                return False
        else:
            depth += 1
    return True


def extract_archive(archive: Path, destination: Path, archive_format: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_format == "tar":
        with tarfile.open(archive, "r:*") as handle:
            for member in handle.getmembers():
                member_name = _safe_member_name(member.name)
                if member.issym() or member.islnk():
                    link = PurePosixPath(member.linkname)
                    combined = member_name.parent / link if member.issym() else link
                    if link.is_absolute() or not _relative_path_stays_inside(combined):
                        raise RuntimeError(
                            f"archive link escapes extraction root: {member.name!r} -> {member.linkname!r}"
                        )
            try:
                handle.extractall(destination, filter="data")
            except TypeError:  # Python 3.11 compatibility after validation above.
                handle.extractall(destination)
        return
    if archive_format == "zip":
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                _safe_member_name(name)
            handle.extractall(destination)
        return
    raise RuntimeError(f"unsupported archive format: {archive_format!r}")


def _single_match(root: Path, pattern: str) -> Path:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {root}, found {matches}")
    return matches[0]


def _copy_executable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_archive_tool(
    name: str,
    spec: Mapping[str, Any],
    *,
    staging: Path,
    downloads: Path,
) -> None:
    archive = downloads / f"{name}.archive"
    download_checked(str(spec["url"]), archive, str(spec["sha256"]))
    extracted = downloads / f"{name}-extracted"
    extract_archive(archive, extracted, str(spec["format"]))
    if name == "node":
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        source = roots[0] if int(spec.get("strip_components", 0)) == 1 and len(roots) == 1 else extracted
        shutil.copytree(source, staging / "node", symlinks=True)
        return
    if name == "gws":
        source = _single_match(extracted, "**/gws")
        _copy_executable(source, staging / "bin" / "gws")
        return
    if name == "gh":
        source = _single_match(extracted, "**/bin/gh")
        _copy_executable(source, staging / "bin" / "gh")
        return
    raise RuntimeError(f"unknown archive tool {name!r}")


def install_fws(staging: Path) -> None:
    fws_root = staging / "fws"
    fws_root.mkdir(parents=True)
    shutil.copy2(FWS_PACKAGE_PATH, fws_root / "package.json")
    shutil.copy2(FWS_PACKAGE_LOCK_PATH, fws_root / "package-lock.json")
    node_bin = staging / "node" / "bin"
    npm = node_bin / ("npm.cmd" if os.name == "nt" else "npm")
    environment = {
        name: os.environ[name]
        for name in ("LANG", "LC_ALL", "LC_CTYPE", "TERM", "SSL_CERT_FILE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")
        if os.environ.get(name)
    }
    environment["PATH"] = os.pathsep.join((str(node_bin), "/usr/local/bin", "/usr/bin", "/bin"))
    run(
        str(npm),
        "ci",
        "--omit=dev",
        "--no-audit",
        "--no-fund",
        cwd=fws_root,
        environment=environment,
    )
    # fws 0.4.4's published wrapper assumes npm's global nested layout.
    # ``npm ci`` deliberately uses a local, fully locked graph, so provide a
    # relocation-safe launcher for that same pinned entrypoint/dependency.
    launcher = staging / "bin" / "fws"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        "#!/bin/sh\n"
        "tool_root=$(CDPATH= cd -- \"$(dirname -- \"$0\")/..\" && pwd)\n"
        "exec \"$tool_root/node/bin/node\" "
        "\"$tool_root/fws/node_modules/tsx/dist/cli.mjs\" "
        "\"$tool_root/fws/node_modules/@juppytt/fws/bin/fws.ts\" \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    _write_fws_compatibility_launcher(staging)


def install_discovery_cache(staging: Path) -> Dict[str, str]:
    cache = staging / "fws-source-config" / "cache"
    cache.mkdir(parents=True)
    digests: Dict[str, str] = {}
    for filename, url in DISCOVERY_DOCUMENTS.items():
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            payload = response.read()
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict) or not value.get("name"):
            raise RuntimeError(f"Google discovery response was invalid for {url}")
        (cache / filename).write_bytes(payload)
        digests[filename] = hashlib.sha256(payload).hexdigest()
    return digests


def verify_install(staging: Path, lock: Mapping[str, Any]) -> Dict[str, str]:
    node = staging / "node" / "bin" / "node"
    fws = staging / "bin" / "fws"
    gws = staging / "bin" / "gws"
    gh = staging / "bin" / "gh"
    environment = {
        name: os.environ[name]
        for name in ("LANG", "LC_ALL", "LC_CTYPE", "TERM")
        if os.environ.get(name)
    }
    environment.update(
        {
            "HOME": str(staging / "verify-home"),
            "PATH": os.pathsep.join((str(node.parent), "/usr/local/bin", "/usr/bin", "/bin")),
        }
    )
    expected = lock["versions"]
    outputs = {
        "node": run(str(node), "--version", cwd=staging, environment=environment),
        "fws": run(str(fws), "--version", cwd=staging, environment=environment),
        "gws": run(str(gws), "--version", cwd=staging, environment=environment),
        "gh": run(str(gh), "--version", cwd=staging, environment=environment),
    }
    for name, output in outputs.items():
        if str(expected[name]) not in output:
            raise RuntimeError(
                f"{name} version mismatch: expected {expected[name]!r}, output={output!r}"
            )
    return outputs


def _existing_install_is_current(
    install_root: Path, *, lock_digest: str, selected_platform: str
) -> bool:
    try:
        manifest = json.loads((install_root / "install.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    required = (
        install_root / "node" / "bin" / "node",
        install_root / "bin" / "fws",
        install_root / "bin" / "gws",
        install_root / "bin" / "gh",
        _fws_compatibility_launcher(install_root),
    )
    return (
        manifest.get("lock_sha256") == lock_digest
        and manifest.get("platform") == selected_platform
        and all(path.is_file() for path in required)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-root", default=str(DEFAULT_INSTALL_ROOT))
    args = parser.parse_args()

    lock_bytes = LOCK_PATH.read_bytes()
    lock = json.loads(lock_bytes.decode("utf-8"))
    lock_digest = hashlib.sha256(lock_bytes).hexdigest()
    npm_lock = json.loads(FWS_PACKAGE_LOCK_PATH.read_text(encoding="utf-8"))
    npm_fws = (npm_lock.get("packages") or {}).get("node_modules/@juppytt/fws") or {}
    if (
        npm_fws.get("version") != lock["versions"]["fws"]
        or npm_fws.get("integrity") != lock["fws"]["npm_integrity"]
    ):
        raise RuntimeError("fws npm package lock does not match pinchbench-tools.lock.json")
    selected_platform = platform_key()
    platform_spec = (lock.get("platforms") or {}).get(selected_platform)
    if not isinstance(platform_spec, dict):
        raise RuntimeError(
            f"Unsupported platform {selected_platform!r}; supported: {sorted((lock.get('platforms') or {}).keys())}"
        )

    install_root = Path(args.install_root).expanduser().resolve()
    install_root.parent.mkdir(parents=True, exist_ok=True)
    if install_root.exists():
        # Schema-v1 installations created before the compatibility launcher
        # are otherwise byte-for-byte current. Repair only this generated
        # wrapper in place; never replace downloaded or user-authored files.
        compatibility_launcher = _fws_compatibility_launcher(install_root)
        actual_tsx = install_root / "fws" / "node_modules" / "tsx" / "dist" / "cli.mjs"
        try:
            existing_manifest = json.loads(
                (install_root / "install.json").read_text(encoding="utf-8")
            )
        except Exception:
            existing_manifest = {}
        if (
            not compatibility_launcher.exists()
            and actual_tsx.is_file()
            and existing_manifest.get("lock_sha256") == lock_digest
            and existing_manifest.get("platform") == selected_platform
        ):
            _write_fws_compatibility_launcher(install_root)
        if _existing_install_is_current(
            install_root,
            lock_digest=lock_digest,
            selected_platform=selected_platform,
        ):
            print((install_root / "install.json").read_text(encoding="utf-8"))
            return 0
        raise RuntimeError(
            f"Refusing to overwrite incomplete or mismatched install at {install_root}; "
            "move it aside and rerun this command"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=".pinchbench-tools-staging-", dir=str(install_root.parent))
    ).resolve()
    try:
        downloads = staging / "downloads"
        downloads.mkdir()
        for name in ("node", "gws", "gh"):
            install_archive_tool(
                name,
                platform_spec[name],
                staging=staging,
                downloads=downloads,
            )
        install_fws(staging)
        discovery_digests = install_discovery_cache(staging)
        version_outputs = verify_install(staging, lock)
        shutil.rmtree(downloads)
        manifest = {
            "schema_version": 1,
            "lock_sha256": lock_digest,
            "platform": selected_platform,
            "versions": lock["versions"],
            "fws_revision": lock["fws"]["revision"],
            "fws_npm_integrity": lock["fws"]["npm_integrity"],
            "archive_sha256": {
                name: platform_spec[name]["sha256"] for name in ("node", "gws", "gh")
            },
            "discovery_sha256": discovery_digests,
            "version_outputs": version_outputs,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "install.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(install_root)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except BaseException:
        if staging.exists() and staging.parent == install_root.parent:
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
