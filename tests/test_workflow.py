import asyncio
import json
import shutil
from pathlib import Path

import pytest

import server
from agent_runtime import evolution, federation
from agent_runtime.config import AppConfig, ModelSettings, load_config
from agent_runtime.contracts import AgentContext, ClientReflection, RuntimeContext
from agent_runtime.evolution import EvolutionResult, ValidationResult, VersionStore
from agent_runtime.hermes import _stage_hermes_profile
from agent_runtime.loader import directory_digest, read_embedded_library, write_embedded_library
from agent_runtime.model import ModelClient


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = {"verification": "Inspect outputs before declaring completion."}


@pytest.fixture
def test_model():
    return ModelClient(ModelSettings(), responder=lambda prompt, system: json.dumps({
        "action": "finish", "solution": "Finished", "insight_library": LIBRARY,
        "feedback": [{"reusable_feedback": "Check the result before finishing."}],
    }))


@pytest.fixture
def store(tmp_path):
    value = VersionStore(tmp_path / "experiment")
    value.initialize(ROOT / "client_agent", ROOT / "server_agent")
    return value


@pytest.mark.parametrize("bootstrap", [True, False])
@pytest.mark.parametrize("server_state", ["accepted", "rejected", "crashed"])
def test_server_precedes_client_and_preserves_library(
    store, test_model, monkeypatch, bootstrap, server_state,
):
    original = store.load()
    calls = []

    async def revise(**kwargs):
        role = kwargs["target_role"]
        calls.append(kwargs)
        previous = Path(kwargs["previous_directory"])
        candidate = Path(kwargs["candidate_directory"])
        if role == "server_agent" and server_state == "crashed":
            raise RuntimeError("revision failed before creating its candidate")
        shutil.copytree(previous, candidate)
        if role == "server_agent":
            (candidate / "SYSTEM_PROMPT.md").write_text("Updated server design")
            write_embedded_library(candidate, LIBRARY)
        else:
            assert [call["target_role"] for call in calls] == ["server_agent", "client_agent"]
            expected_server = (
                str(calls[0]["candidate_directory"])
                if server_state == "accepted" else original["server_agent"]
            )
            assert str(kwargs["agent_source_directory"]) == expected_server
            assert str(kwargs["context_directory"]) == expected_server
            assert kwargs["seed_library"] == (LIBRARY if server_state != "crashed" else {})
        accepted = role == "client_agent" or server_state == "accepted"
        return EvolutionResult(
            target_role=role,
            previous_directory=str(previous), candidate_directory=str(candidate),
            previous_digest=directory_digest(previous), candidate_digest=directory_digest(candidate),
            activated=accepted, report={"outcome_status": "completed"},
            validation=ValidationResult(accepted, []),
            proposed_library=LIBRARY if role == "server_agent" else None,
        )

    monkeypatch.setattr(server, "evolve_agent", revise)
    if bootstrap:
        manifest = asyncio.run(server.bootstrap_agents(
            version_store=store, config=AppConfig(), model=test_model,
        ))
        assert all(call["traces"] == [] for call in calls)
        assert manifest["active_version"] == 0
        assert manifest["history"] == []
        assert asyncio.run(server.bootstrap_agents(
            version_store=store, config=AppConfig(), model=test_model,
        )) == manifest
        assert len(calls) == 2
    else:
        reflection = ClientReflection(summary="Inspect intermediate outputs.")
        report = asyncio.run(server.evolve_harness(
            version_store=store, config=AppConfig(), model=test_model,
            reasoning_traces=[reflection],
        ))
        manifest = report["manifest"]
        assert manifest["active_version"] == 1
        assert calls[0]["traces"] == calls[1]["traces"]
        assert calls[0]["traces"][0].summary == reflection.summary
    expected_library = LIBRARY if server_state != "crashed" else {}
    for role in ("server_agent", "client_agent"):
        assert read_embedded_library(manifest[role]) == expected_library


@pytest.mark.parametrize("role", ["server_agent", "client_agent"])
def test_completed_revision_does_not_force_code_changes(tmp_path, test_model, role):
    result = asyncio.run(evolution.evolve_agent(
        target_role=role, previous_directory=ROOT / role,
        candidate_directory=tmp_path / role,
        agent_source_directory=ROOT / "server_agent",
        guide="Keep sound code and collect reusable insights.",
        config=AppConfig(), model=test_model,
    ))
    assert result.activated, result.validation.checks
    assert result.previous_digest == result.candidate_digest
    assert result.report["attempts_used"] == 1
    assert result.report["model_calls"] == 1
    assert result.report["input_tokens"] > 0
    assert result.report["output_tokens"] > 0
    assert result.report["total_tokens"] == (
        result.report["input_tokens"] + result.report["output_tokens"]
    )
    assert result.report["reconsidered"] is False
    if role == "server_agent":
        assert result.proposed_library == LIBRARY
        assert result.report["library_changed"] is True


def test_failed_protocol_still_retries(tmp_path):
    replies = iter(["{}", "{}", json.dumps({
        "action": "finish", "insight_library": LIBRARY,
    })])
    model = ModelClient(ModelSettings(), responder=lambda prompt, system: next(replies))
    result = asyncio.run(evolution.evolve_agent(
        target_role="server_agent", previous_directory=ROOT / "server_agent",
        candidate_directory=tmp_path / "candidate", guide="Maintain the library.",
        config=AppConfig(), model=model,
    ))
    assert result.activated, result.validation.checks
    assert result.report["attempts_used"] == 2
    assert result.report["model_calls"] == 3


def test_repeated_bootstrap_does_not_revert_an_evolved_library(store, test_model):
    asyncio.run(server.bootstrap_agents(
        version_store=store, config=AppConfig(), model=test_model,
    ))
    updated_library = {"recovery": "Check failed tool calls before continuing."}
    model = ModelClient(ModelSettings(), responder=lambda prompt, system: json.dumps({
        "action": "finish", "solution": "Finished", "insight_library": updated_library,
    }))
    report = asyncio.run(server.evolve_harness(
        version_store=store, config=AppConfig(), model=model,
        reasoning_traces=[ClientReflection(summary="Handle tool failures.")],
    ))
    assert report["library"] == updated_library
    manifest = asyncio.run(server.bootstrap_agents(
        version_store=store, config=AppConfig(), model=test_model,
    ))
    for role in ("server_agent", "client_agent"):
        assert read_embedded_library(manifest[role]) == updated_library


def test_round_keeps_grader_and_held_out_feedback_out_of_evolution(tmp_path):
    seen_revisions = []
    held_out_calls = []

    def respond(prompt, system):
        if 'harness revision' in json.loads(prompt.split('Current agent state:\n', 1)[1])['role']:
            seen_revisions.append(prompt)
            feedback = "Revision completed."
        elif "HELD_OUT_INPUT" in prompt:
            held_out_calls.append(prompt)
            feedback = "HELD_OUT_REFLECTION"
        else:
            feedback = "TRAIN_REFLECTION"
        return json.dumps({
            "action": "finish", "solution": "PRIVATE_ANSWER", "insight_library": LIBRARY,
            "feedback": [{"reusable_feedback": feedback}],
        })

    def grade(case, outcome):
        outcome.reflection.summary = "GRADER_MUTATION"
        outcome.reflection.reusable_insights.append("GRADER_LABEL")
        return {"score": 1.0, "answer": "GRADER_REFERENCE_ANSWER"}

    model = ModelClient(ModelSettings(), responder=respond)
    kwargs = dict(
        train_cases=[federation.BenchmarkCase("case-001", "TRAIN_INPUT", tmp_path / "train")],
        held_out_cases=[federation.BenchmarkCase("case-002", "HELD_OUT_INPUT", tmp_path / "held")],
        grade=grade, output_dir=tmp_path / "run", rounds=1, num_workers=1,
        config=AppConfig(), test_model=model,
        client_seed=ROOT / "client_agent", server_seed=ROOT / "server_agent",
    )
    result = federation.run_federated_experiment(**kwargs)
    assert len(seen_revisions) == 4  # two bootstrap steps, then server and client evolution
    for prompt in seen_revisions:
        for forbidden in ("HELD_OUT_REFLECTION", "HELD_OUT_INPUT", "GRADER_MUTATION", "GRADER_LABEL", "GRADER_REFERENCE_ANSWER", "PRIVATE_ANSWER"):
            assert forbidden not in prompt
    assert all("TRAIN_REFLECTION" in prompt for prompt in seen_revisions[2:])
    assert all("TRAIN_REFLECTION" not in prompt for prompt in seen_revisions[:2])
    assert len(held_out_calls) == 2  # frozen pre-round and final returned-agent evaluation
    assert result["final_held_out"] is not None
    federation.run_federated_experiment(**kwargs)
    assert len(seen_revisions) == 4
    assert len(held_out_calls) == 2


def test_default_hermes_profile_uses_keyless_search(tmp_path):
    config = load_config(ROOT / "config.yaml")
    assert config.search.provider is None
    assert config.hermes.enabled
    context = AgentContext(
        objective="Research a public question", role="client", insight_library={},
        runtime_context=RuntimeContext(workspace=tmp_path),
    )
    _stage_hermes_profile(tmp_path, context, toolsets=config.hermes.toolsets)
    profile = json.loads((tmp_path / "config.yaml").read_text())
    assert profile["web"] == {"keyless_fallback": True}
    assert "api_key" not in json.dumps(profile)


def test_empty_training_roster_fails_before_bootstrap(tmp_path, test_model):
    with pytest.raises(ValueError, match="requires training cases"):
        federation.run_federated_experiment(
            train_cases=[], held_out_cases=[], grade=lambda case, outcome: {},
            output_dir=tmp_path, config=AppConfig(), test_model=test_model,
        )
    assert not (tmp_path / "agent_versions").exists()


def test_missing_reflections_do_not_start_an_evolution_round(tmp_path, test_model, monkeypatch):
    def batch(cases, **kwargs):
        return [], [], [[] for case in cases]

    async def unexpected_evolution(**kwargs):
        pytest.fail("Evolution must wait for training reflections")

    monkeypatch.setattr(federation, "_run_batch", batch)
    monkeypatch.setattr(federation, "evolve_harness", unexpected_evolution)
    with pytest.raises(RuntimeError, match="No training reflections"):
        federation.run_federated_experiment(
            train_cases=[federation.BenchmarkCase("case-001", "Task", tmp_path / "task")],
            held_out_cases=[], grade=lambda case, outcome: {}, output_dir=tmp_path,
            config=AppConfig(), test_model=test_model,
            client_seed=ROOT / "client_agent", server_seed=ROOT / "server_agent",
        )
    assert VersionStore(tmp_path).load()["active_version"] == 0


@pytest.mark.parametrize("role", ["client_agent", "server_agent"])
def test_standalone_runtime_matches_canonical(role):
    for canonical in (ROOT / "agent_runtime").glob("*.py"):
        assert (ROOT / role / "agent_runtime" / canonical.name).read_bytes() == canonical.read_bytes()
    assert (ROOT / role / "utils.py").read_bytes() == (ROOT / "utils.py").read_bytes()
