"""Initialize and evolve server then client harnesses in Algorithm 1 order."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_runtime.config import AppConfig, load_config
from agent_runtime.contracts import ClientReflection, ReasoningTrace
from agent_runtime.evolution import (
    EvolutionResult,
    ValidationResult,
    VersionStore,
    evolve_agent,
    write_revision_logs,
)
from agent_runtime.loader import directory_digest, read_embedded_library, write_embedded_library
from agent_runtime.hermes import preflight_external_hermes
from agent_runtime.model import ModelClient
from agent_runtime.provider_errors import FatalProviderError
from agent_runtime.privacy import (
    sanitize_library,
    sanitize_reflection,
    sanitize_text,
    sanitize_traces,
)
from prompt_hub import add_prompt_file_argument, load_prompt_state


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER_AGENT = REPO_ROOT / "server_agent"
DEFAULT_CLIENT_AGENT = REPO_ROOT / "client_agent"


def _revision_library(result: EvolutionResult) -> Dict[str, str]:
    """Keep completed library updates even when the code candidate is rejected."""
    library = result.proposed_library
    if library is None and result.report.get("outcome_status") == "completed":
        library = read_embedded_library(result.candidate_directory)
    if library is None:
        library = read_embedded_library(result.previous_directory)
    return sanitize_library(library)


def _rescue(
    target_role: str, previous: str | Path, candidate: Path, exc: Exception
) -> EvolutionResult:
    """Guarantee activate() always has a real (rejected) candidate to inspect.

    Physical necessity, not a policy choice: if a revision run crashed
    before evolve_agent's own copytree ever happened (or, more likely,
    partway through it, leaving a half-edited but real directory on disk),
    there is nothing to validate without this. Either way the crash is
    recorded plainly in the report/history rather than silently retried or
    turned into a synthetic "unchanged".
    """
    if not candidate.exists():
        shutil.copytree(previous, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return EvolutionResult(
        target_role=target_role,
        previous_directory=str(Path(previous).resolve()),
        candidate_directory=str(candidate.resolve()),
        previous_digest=directory_digest(previous),
        candidate_digest=directory_digest(candidate),
        activated=False,
        report={
            "status": "failed",
            "files_changed": [],
            "outcome_status": "crashed",
            "summary": sanitize_text(f"{type(exc).__name__}: {exc}"),
        },
        validation=ValidationResult(
            False, [{"check": "controller", "result": "failed", "evidence": type(exc).__name__}]
        ),
    )


async def evolve_harness(
    *,
    version_store: VersionStore,
    reasoning_traces: Iterable[ClientReflection | ReasoningTrace | Dict[str, Any]],
    next_version: Optional[int] = None,
    config: Optional[AppConfig] = None,
    config_path: str | Path = REPO_ROOT / "config.yaml",
    prompt_file: Optional[str | Path] = None,
    model: Optional[ModelClient] = None,
) -> Dict[str, Any]:
    app_config = config or load_config(config_path)
    backend_config = (
        replace(app_config, model=model.settings)
        if model is not None and not model.has_responder
        else app_config
    )
    preflight_external_hermes(
        backend_config,
        test_model=model if model is not None and model.has_responder else None,
    )
    prompt_state = load_prompt_state(prompt_file)
    manifest = version_store.load()
    version = int(next_version if next_version is not None else manifest["active_version"] + 1)
    version_dir = version_store.root / f"v{version:03d}"
    if version_dir.exists():
        archived = version_store.root / f"v{version:03d}.interrupted-{int(time.time())}"
        version_dir.replace(archived)
    version_dir.mkdir(parents=True, exist_ok=False)
    clean_traces: List[ClientReflection | ReasoningTrace] = []
    for raw in reasoning_traces:
        if isinstance(raw, ClientReflection) or (
            isinstance(raw, dict) and ("reusable_insights" in raw or "summary" in raw)
        ):
            clean_traces.append(
                sanitize_reflection(raw, max_chars=app_config.limits.max_trace_chars)
            )
        else:
            clean_traces.extend(
                sanitize_traces([raw], max_chars=app_config.limits.max_trace_chars)
            )

    # Step 1: the server agent, running its own current code, revises itself
    # -- H_s^(t+1) <- A_s^(t)(R_1:K, Gs, H_s^(t), H_c^(t)) -- and produces the
    # shared insight library as part of that revision.
    server_previous = manifest["server_agent"]
    server_candidate = version_dir / "server_agent"
    try:
        server_result = await evolve_agent(
            target_role="server_agent",
            previous_directory=server_previous,
            candidate_directory=server_candidate,
            guide=str(prompt_state["server_guide"]),
            traces=clean_traces,
            context_role="client_agent",
            context_directory=manifest["client_agent"],
            seed_library=read_embedded_library(server_previous),
            config=app_config,
            model=model,
        )
    except FatalProviderError:
        raise
    except Exception as exc:
        if app_config.apply_all_revisions:
            raise
        server_result = _rescue("server_agent", server_previous, server_candidate, exc)

    # The next client is authored by the newly validated server (Algorithm 1, 10).
    server_active_dir = server_result.candidate_directory if server_result.activated else server_result.previous_directory
    next_library = _revision_library(server_result)

    # Step 2: the server agent, now running the harness step 1 produced (or
    # its previous one, if that candidate was rejected), revises the client
    # harness -- H_c^(t+1) <- A_s^(t+1)(R_1:K, Gc, H_c^(t), H_s^(t+1)).
    client_previous = manifest["client_agent"]
    client_candidate = version_dir / "client_agent"
    try:
        client_result = await evolve_agent(
            target_role="client_agent",
            previous_directory=client_previous,
            candidate_directory=client_candidate,
            agent_source_directory=server_active_dir,
            guide=str(prompt_state["client_guide"]),
            traces=clean_traces,
            context_role="server_agent",
            context_directory=server_active_dir,
            seed_library=next_library,
            config=app_config,
            model=model,
        )
    except FatalProviderError:
        raise
    except Exception as exc:
        if app_config.apply_all_revisions:
            raise
        client_result = _rescue("client_agent", client_previous, client_candidate, exc)

    log_paths = write_revision_logs(version_store, version, client_result, server_result)
    updated = version_store.activate(
        version,
        client_result,
        server_result,
        insight_library=next_library,
        revision_logs=log_paths,
    )
    # The insight library legitimately advances every round regardless of
    # whether either candidate was accepted (see loader.read_embedded_library's
    # own docstring) -- write it into whichever client_agent directory is now
    # active so a rejected client candidate still carries this round's
    # library forward instead of silently reverting to an older one.
    write_embedded_library(updated["server_agent"], next_library)
    write_embedded_library(updated["client_agent"], next_library)
    rejected_roles = [
        role
        for role, result in (("client_agent", client_result), ("server_agent", server_result))
        if not result.activated
    ]
    rejection_streaks = {
        "client_agent": int(updated.get("client_rejection_streak", 0)),
        "server_agent": int(updated.get("server_rejection_streak", 0)),
    }
    activation_status = (
        "accepted"
        if not rejected_roles
        else "full_fallback" if len(rejected_roles) == 2 else "partial_fallback"
    )
    alerts = [
        {
            "severity": "error",
            "role": role,
            "message": "Candidate revision was rejected; the previous validated implementation remains active.",
            "consecutive_rejections": rejection_streaks[role],
            "repeated_rejection": rejection_streaks[role] > 1,
        }
        for role in rejected_roles
    ]
    # Retained for reports from older or custom evolution implementations.
    stagnant_roles = [
        role
        for role, result in (("client_agent", client_result), ("server_agent", server_result))
        if isinstance(result.report, dict) and result.report.get("still_stagnant")
    ]
    report = {
        "version": version,
        "revision_counts": {"client_agent": 1, "server_agent": 1},
        "activation_status": activation_status,
        "rejected_roles": rejected_roles,
        "rejection_streaks": rejection_streaks,
        "stagnant_roles": stagnant_roles,
        "alerts": alerts,
        "client": client_result.to_dict(),
        "server": server_result.to_dict(),
        "library": next_library,
        "manifest": updated,
        "logs": log_paths,
    }
    (version_dir / "evolution_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (version_dir / "REVISION_STATUS.json").write_text(
        json.dumps(
            {
                "status": activation_status,
                "alerts": alerts,
                "stagnant_roles": stagnant_roles,
                "logs": log_paths,
                "note": "Invalid candidates remain archived; the previous validated implementation stays active.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for alert in alerts:
        repeated = " REPEATED FAILURE." if alert["repeated_rejection"] else ""
        print(
            f"REVISION REJECTED [{alert['role']}]: previous implementation retained; "
            f"consecutive rejections={alert['consecutive_rejections']}.{repeated}",
            file=sys.stderr,
        )
    for role in stagnant_roles:
        print(
            f"STILL STAGNANT [{role}]: activated, but made no observable change even after "
            "reconsideration -- see its evolve_agent report's `summary` for its own stated reason.",
            file=sys.stderr,
        )
    print(f"v{version:03d} revision logs: {log_paths['summary']}", file=sys.stderr)
    return report


async def bootstrap_agents(
    *,
    version_store: VersionStore,
    config: Optional[AppConfig] = None,
    config_path: str | Path = REPO_ROOT / "config.yaml",
    prompt_file: Optional[str | Path] = None,
    model: Optional[ModelClient] = None,
) -> Dict[str, Any]:
    """Initialize the server, then let it author the client (Algorithm 1, 1-2).

    Both steps use guides without task reflections. The client-design step
    runs the newly validated server, or the seed if that revision failed.
    Commit both roles atomically without advancing the round counter.
    """
    app_config = config or load_config(config_path)
    backend_config = (
        replace(app_config, model=model.settings)
        if model is not None and not model.has_responder
        else app_config
    )
    preflight_external_hermes(
        backend_config,
        test_model=model if model is not None and model.has_responder else None,
    )
    prompt_state = load_prompt_state(prompt_file)
    manifest = version_store.load()
    if manifest.get("bootstrapped"):
        committed_library = manifest.get("bootstrap_insight_library")
        if int(manifest["active_version"]) == 0 and isinstance(committed_library, dict):
            committed_library = sanitize_library(committed_library)
            # Repair bootstrap mirrors only before later rounds replace the library.
            write_embedded_library(manifest["server_agent"], committed_library)
            write_embedded_library(manifest["client_agent"], committed_library)
        return manifest
    prepared_root = version_store.root / "v000_bootstrapped"
    if prepared_root.exists():
        # record_bootstrap commits the complete prepared tree before its
        # atomic manifest replace.  If the process dies in that narrow gap,
        # the old manifest still points at intact v000; preserve the orphaned
        # evidence and rebuild rather than mistaking it for a committed init.
        archived = version_store.root / (
            f"v000_bootstrapped.interrupted-{time.time_ns()}"
        )
        prepared_root.replace(archived)
    bootstrap_root = version_store.root / "v000_bootstrap"
    # A process may die after creating one bootstrap candidate but before the
    # atomic manifest records completion. Reusing that half-written directory
    # makes evolve_agent raise FileExistsError for both roles and used to mark
    # bootstrap complete with two rejected leftovers. Preserve the interrupted
    # evidence, then start the idempotent initialization attempt from clean
    # candidate paths.
    if bootstrap_root.exists():
        archived = version_store.root / f"v000_bootstrap.interrupted-{int(time.time())}"
        suffix = 1
        while archived.exists():
            archived = version_store.root / (
                f"v000_bootstrap.interrupted-{int(time.time())}-{suffix}"
            )
            suffix += 1
        bootstrap_root.replace(archived)
    seed_server = manifest["server_agent"]

    async def attempt(
        target_role: str,
        guide_key: str,
        *,
        source: str,
        context_directory: str,
        library: Dict[str, str],
    ) -> EvolutionResult:
        previous = manifest[target_role]
        other_role = "server_agent" if target_role == "client_agent" else "client_agent"
        candidate = bootstrap_root / target_role
        try:
            return await evolve_agent(
                target_role=target_role,
                previous_directory=previous,
                candidate_directory=candidate,
                agent_source_directory=source,
                guide=str(prompt_state[guide_key]),
                traces=[],
                context_role=other_role,
                context_directory=context_directory,
                seed_library=library,
                config=app_config,
                model=model,
            )
        except FatalProviderError:
            raise
        except Exception as exc:
            if app_config.apply_all_revisions:
                raise
            return _rescue(target_role, previous, candidate, exc)

    server_result = await attempt(
        "server_agent", "server_guide",
        source=seed_server,
        context_directory=manifest["client_agent"],
        library=read_embedded_library(seed_server),
    )
    bootstrap_library = _revision_library(server_result)
    server_active = (
        server_result.candidate_directory
        if server_result.activated else server_result.previous_directory
    )
    client_result = await attempt(
        "client_agent", "client_guide",
        source=server_active,
        context_directory=server_active,
        library=bootstrap_library,
    )
    log_paths = write_revision_logs(version_store, 0, client_result, server_result)
    updated = version_store.record_bootstrap(
        client_result,
        server_result,
        insight_library=bootstrap_library,
    )
    committed_library = sanitize_library(
        updated.get("bootstrap_insight_library", bootstrap_library)
    )
    write_embedded_library(updated["server_agent"], committed_library)
    write_embedded_library(updated["client_agent"], committed_library)
    for role, result in (("client_agent", client_result), ("server_agent", server_result)):
        if not result.activated:
            verdict = "kept the checked-in seed after rejecting the candidate"
        elif result.candidate_digest != result.previous_digest:
            verdict = "adopted a newly designed implementation"
        else:
            verdict = "retained the validated seed because no code change was proposed"
        print(f"BOOTSTRAP [{role}]: {verdict}.", file=sys.stderr)
    print(f"v000 bootstrap logs: {log_paths['summary']}", file=sys.stderr)
    try:
        bootstrap_root.rmdir()  # only succeeds if every candidate was moved or archived out of it
    except OSError:
        pass
    return updated


class HarnessServer:
    """Compatibility wrapper around the new evolution controller."""

    def __init__(
        self,
        *,
        server_id: str = "server",
        residual_dir: str = "residuals",
        out_dir: str = "federation_state",
        workspace_root: str = "foh_runs",
        agent_cfg: Optional[Dict[str, Any]] = None,
        config_path: str | Path = REPO_ROOT / "config.yaml",
        prompt_file: Optional[str] = None,
    ) -> None:
        self.server_id = server_id
        self.residual_dir = Path(residual_dir)
        self.out_dir = Path(out_dir)
        self.workspace_root = Path(workspace_root)
        self.agent_cfg = dict(agent_cfg or {})
        self.config_path = Path(self.agent_cfg.get("config_path", config_path))
        self.prompt_file = self.agent_cfg.get("prompt_file", prompt_file)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        version_store_dir = self.agent_cfg.get("version_store_dir", self.out_dir)
        if {"individual_states", "residual_states"}.intersection(self.out_dir.parts):
            version_store_dir = self.out_dir
        self.version_store = VersionStore(version_store_dir)
        self.version_store.initialize(DEFAULT_CLIENT_AGENT, DEFAULT_SERVER_AGENT)

    def aggregate_residual_updates(self) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        if not self.residual_dir.exists():
            return values
        for path in sorted(self.residual_dir.glob("problem_*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(value, dict):
                values.append(value)
        return values

    def run_round(self, prev_guide: Any = None, prev_library: Any = None) -> Dict[str, Any]:
        """prev_guide is accepted for call-site compatibility only -- the
        server guide now lives in prompt_hub, not as a per-round value a
        caller threads through. prev_library, when explicitly given, seeds
        the server harness's own embedded library before this round's
        revision step reads it (agent_versions/*/server_agent's copy is
        otherwise the durable, persisted source of truth across repeated
        HarnessServer(...) construction on the same out_dir, so most callers
        never need to pass this at all).
        """
        if isinstance(prev_library, dict):
            manifest = self.version_store.load()
            write_embedded_library(manifest["server_agent"], {str(k): str(v) for k, v in prev_library.items()})
        residuals = self.aggregate_residual_updates()
        raw_traces: List[Any] = []
        for residual in residuals:
            if residual.get("reflection") is not None:
                raw_traces.append(residual["reflection"])
            else:
                raw_traces.extend(residual.get("reasoning_traces") or [])

        evolution = asyncio.run(
            evolve_harness(
                version_store=self.version_store,
                reasoning_traces=raw_traces,
                config_path=self.config_path,
                prompt_file=self.prompt_file,
            )
        )
        library = evolution["library"]
        library_path = self.out_dir / "insight_library.json"
        guide_path = self.out_dir / "guide.json"
        library_path.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")
        guide_path.write_text("{}\n", encoding="utf-8")
        return {
            "round": evolution["version"],
            "num_residuals": len(residuals),
            "guide": {},
            "library": library,
            "paths": {"guide": str(guide_path), "library": str(library_path)},
            "evolution": evolution,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate traces and evolve versioned agents")
    parser.add_argument("--residual-dir", default="residuals")
    parser.add_argument("--out-dir", default="federation_state")
    parser.add_argument("--workspace-root", default="foh_runs")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    add_prompt_file_argument(parser)
    args = parser.parse_args()
    # The compatibility wrapper creates a VersionStore in its constructor.
    # Reject a non-operational backend before that filesystem mutation so a
    # failed mock/disabled run cannot leave experiment-looking state behind.
    preflight_external_hermes(load_config(args.config))
    controller = HarnessServer(
        residual_dir=args.residual_dir,
        out_dir=args.out_dir,
        workspace_root=args.workspace_root,
        config_path=args.config,
        prompt_file=args.prompt_file,
    )
    print(json.dumps(controller.run_round(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
