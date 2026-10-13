"""Git distribution of evolved harnesses: one branch per iteration."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime import federation
from agent_runtime.config import AppConfig, HarnessGitSettings as ConfigGitSettings, ModelSettings, load_config
from agent_runtime.harness_git import (
    HarnessGitSettings,
    latest_branch,
    publish_iteration,
    pull_iteration,
)
from agent_runtime.loader import directory_digest, read_embedded_library
from agent_runtime.model import ModelClient

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = {"verification": "Inspect outputs before declaring completion."}


def _branches(repo: Path) -> list[str]:
    out = subprocess.run(["git", "branch", "--list"], cwd=repo, capture_output=True, text=True, check=True).stdout
    return sorted(line.strip("* ").strip() for line in out.splitlines() if line.strip())


def test_publish_creates_chained_branches_and_pull_checks_them_out(tmp_path):
    remote = tmp_path / "shared" / "harness.git"   # created as a bare repo on first publish
    settings = HarnessGitSettings(enabled=True, repo=str(remote))
    out = tmp_path / "experiment"

    first = publish_iteration(
        settings, output_dir=out, iteration=0,
        client_agent=ROOT / "client_agent", server_agent=ROOT / "server_agent",
        insight_library=LIBRARY, note="bootstrap",
    )
    assert first["branch"] == "iteration-000" and first["pushed"]
    assert _branches(remote) == ["iteration-000"]

    # Round 1 evolves the client: a modified copy is published on top of iteration-000.
    evolved = tmp_path / "v001" / "client_agent"
    import shutil
    shutil.copytree(ROOT / "client_agent", evolved, ignore=shutil.ignore_patterns("__pycache__"))
    (evolved / "SYSTEM_PROMPT.md").write_text("evolved prompt\n", encoding="utf-8")
    second = publish_iteration(
        settings, output_dir=out, iteration=1,
        client_agent=evolved, server_agent=ROOT / "server_agent",
        insight_library={**LIBRARY, "new": "A lesson learned in round 1."},
    )
    assert second["parent_branch"] == "iteration-000"
    assert _branches(remote) == ["iteration-000", "iteration-001"]
    assert latest_branch(str(remote)) == "iteration-001"
    # iteration-001 descends from iteration-000, so the diff between them is the round's change.
    log = subprocess.run(
        ["git", "log", "--oneline", "iteration-001"], cwd=remote, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert len(log) == 2

    index = json.loads((out / "harness_branches.json").read_text())
    assert set(index) == {"0", "1"} and index["1"]["commit"] == second["commit"]

    # Client side: pull the branch and get a runnable overlay with the library.
    pulled = pull_iteration(str(remote), "iteration-001", tmp_path / "client_clone")
    assert (pulled / "client_agent" / "SYSTEM_PROMPT.md").read_text() == "evolved prompt\n"
    assert json.loads((pulled / "insight_library.json").read_text())["new"].startswith("A lesson")
    assert directory_digest(pulled / "client_agent") == directory_digest(evolved)
    record = json.loads((pulled / "ITERATION.json").read_text())
    assert record["iteration"] == 1 and record["branch"] == "iteration-001"
    # Pulling again into the same clone updates in place (fetch + checkout).
    pulled_again = pull_iteration(str(remote), "iteration-000", tmp_path / "client_clone")
    assert (pulled_again / "client_agent" / "SYSTEM_PROMPT.md").read_text() != "evolved prompt\n"

    # Re-publishing an iteration whose content is unchanged is idempotent.
    third = publish_iteration(
        settings, output_dir=out, iteration=1,
        client_agent=evolved, server_agent=ROOT / "server_agent",
        insight_library={**LIBRARY, "new": "A lesson learned in round 1."},
    )
    assert third["commit"] == second["commit"]


def test_settings_validation():
    with pytest.raises(ValueError):
        HarnessGitSettings(enabled=True, repo="").validate()
    with pytest.raises(ValueError):
        HarnessGitSettings(enabled=True, repo="x", branch_prefix="bad prefix").validate()
    HarnessGitSettings(enabled=False, repo="").validate()


def test_config_loads_harness_git_and_keeps_fingerprint_stable(tmp_path):
    base = load_config(ROOT / "config.yaml")
    assert base.harness_git.enabled is False
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        (ROOT / "config.yaml").read_text().replace("  enabled: false\n  repo: \"\"", "  enabled: true\n  repo: shared/harness.git"),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.harness_git.enabled is True
    assert Path(loaded.harness_git.repo).is_absolute()  # relative paths resolve against the config file
    # Distribution never changes the experiment identity (hermes paths differ only
    # because they resolve relative to the config file's directory).
    seen, expected = loaded.public_dict(), base.public_dict()
    seen.pop("hermes"); expected.pop("hermes")
    assert seen == expected and "harness_git" not in seen
    bad = tmp_path / "bad.yaml"
    bad.write_text("harness_git:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_federated_experiment_publishes_a_branch_per_iteration(tmp_path):
    model = ModelClient(ModelSettings(), responder=lambda prompt, system: json.dumps({
        "action": "finish", "solution": "Finished", "insight_library": LIBRARY,
        "feedback": [{"reusable_feedback": "Check the result before finishing."}],
    }))
    remote = tmp_path / "harness.git"
    config = AppConfig(harness_git=ConfigGitSettings(enabled=True, repo=str(remote)))
    out = tmp_path / "run"
    federation.run_federated_experiment(
        train_cases=[federation.BenchmarkCase("case-001", "TRAIN_INPUT", tmp_path / "train")],
        held_out_cases=[],
        grade=lambda case, outcome: {"score": 1.0},
        output_dir=out, rounds=2, num_workers=1,
        config=config, test_model=model,
        client_seed=ROOT / "client_agent", server_seed=ROOT / "server_agent",
    )
    # bootstrap -> iteration-000, round 1 -> iteration-001, round 2 -> iteration-002
    assert _branches(remote) == ["iteration-000", "iteration-001", "iteration-002"]
    index = json.loads((out / "harness_branches.json").read_text())
    assert [index[k]["branch"] for k in sorted(index)] == ["iteration-000", "iteration-001", "iteration-002"]
    assert (out / "round_002" / "harness_branch.json").is_file()
    manifest = json.loads((out / "agent_versions" / "manifest.json").read_text())
    pulled = pull_iteration(str(remote), "iteration-002", tmp_path / "pulled")
    assert directory_digest(pulled / "client_agent") == manifest["client_digest"]
    assert read_embedded_library(pulled / "client_agent") == read_embedded_library(manifest["client_agent"])
