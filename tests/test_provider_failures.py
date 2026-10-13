"""Fatal API failures must never become task zeros or committed fallback rounds."""
import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import parallel_utils
import server
from agent_runtime import evolution, federation, hermes, hermes_driver
from agent_runtime.config import AppConfig, ModelSettings
from agent_runtime.contracts import AgentOutcome
from agent_runtime.evolution import VersionStore
from agent_runtime.loader import directory_digest
from agent_runtime.model import ModelClient
from agent_runtime.provider_errors import (
    FatalProviderError, ProviderBatchAbort, provider_batch_abort,
    raise_for_gemini_error_text,
)
from tasks import task_scienceagentbench as sab

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("status", [401, 402, 403, 404, 429, 503])
def test_native_guard_exits_before_hermes_can_retry(monkeypatch, status):
    error = RuntimeError(f"Gemini HTTP {status}: blocked")
    error.status_code = status
    error.details = {"reason": "API_KEY_SERVICE_BLOCKED"}
    original = lambda response, **kwargs: error
    adapter = SimpleNamespace(gemini_http_error=original)
    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(gemini_native_adapter=adapter))
    restore = hermes_driver._install_gemini_error_guard()
    try:
        if status in {401, 402, 403, 404}:
            calls = 0
            with pytest.raises(hermes_driver._FatalProviderResponse) as caught:
                for _ in range(30):
                    calls += 1
                    try:
                        raise adapter.gemini_http_error(None)
                    except Exception:
                        pass
            assert calls == 1
            assert caught.value.details["reason"] == "API_KEY_SERVICE_BLOCKED"
        else:
            assert adapter.gemini_http_error(None) is error
    finally:
        restore()
    assert adapter.gemini_http_error is original


@pytest.mark.parametrize("status", [402, 403])
@pytest.mark.parametrize("stream", [False, True])
def test_pinned_gemini_transport_reports_fatal_http_error(status, stream):
    source = ROOT / ".hermes/hermes-agent"
    python = ROOT / ".hermes/venv/bin/python"
    if not python.exists() or not (source / "agent/gemini_native_adapter.py").exists():
        pytest.skip("Pinned Hermes is not installed")
    script = r'''
import importlib.util, sys
import httpx
from agent.gemini_native_adapter import GeminiNativeClient
spec = importlib.util.spec_from_file_location("foh_driver", sys.argv[1])
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
status, stream = int(sys.argv[2]), sys.argv[3] == "True"
requests = []
def respond(request):
    requests.append(request)
    return httpx.Response(status, json={"error": {
        "message": "Blocked or unpaid", "status": "PERMISSION_DENIED",
        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                     "reason": "API_KEY_SERVICE_BLOCKED"}],
    }})
client = GeminiNativeClient(api_key="fake-key", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
restore = driver._install_gemini_error_guard()
try:
    try:
        result = client.chat.completions.create(
            model="gemini-test", messages=[{"role":"user", "content":"Hello"}], stream=stream,
        )
        if stream:
            list(result)
        raise AssertionError("Provider failure was swallowed")
    except driver._FatalProviderResponse as error:
        assert error.details["status_code"] == status
        assert error.details["reason"] == "API_KEY_SERVICE_BLOCKED"
    assert len(requests) == 1
finally:
    restore()
'''
    result = subprocess.run(
        [str(python), "-c", script, str(ROOT / "agent_runtime/hermes_driver.py"), str(status), str(stream)],
        env={**os.environ, "PYTHONPATH": str(source)}, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("phase", ["bootstrap", "revision"])
def test_server_does_not_rescue_provider_failure_as_fallback(tmp_path, monkeypatch, phase):
    store = VersionStore(tmp_path / "experiment")
    store.initialize(ROOT / "client_agent", ROOT / "server_agent")
    before = store.load()
    calls = []

    async def fail(**kwargs):
        calls.append(kwargs["target_role"])
        raise FatalProviderError("Gemini HTTP 402: credits depleted")

    monkeypatch.setattr(server, "evolve_agent", fail)
    model = ModelClient(ModelSettings(), responder=lambda *a: "unused")
    with pytest.raises(FatalProviderError, match="402"):
        if phase == "bootstrap":
            asyncio.run(server.bootstrap_agents(version_store=store, config=AppConfig(), model=model))
        else:
            asyncio.run(server.evolve_harness(
                version_store=store, config=AppConfig(), model=model, reasoning_traces=[],
            ))
    assert calls == ["server_agent"]
    assert store.load() == before
    assert not list(store.root.glob("logs/v*_summary.md"))


def test_revision_does_not_retry_provider_failure(tmp_path, monkeypatch):
    calls = []

    async def fail(context, runtime):
        calls.append(context)
        raise FatalProviderError("Gemini HTTP 403: blocked")

    monkeypatch.setattr(evolution, "load_agent", lambda source: (None, fail))
    with pytest.raises(FatalProviderError, match="403"):
        asyncio.run(evolution.evolve_agent(
            target_role="server_agent", previous_directory=ROOT / "server_agent",
            candidate_directory=tmp_path / "candidate", guide="Maintain the library.",
            config=AppConfig(), model=ModelClient(ModelSettings(), responder=lambda *a: "unused"),
        ))
    assert len(calls) == 1


@pytest.mark.parametrize("outcome_status", ["completed", "failed"])
def test_always_apply_promotes_authored_revision_despite_validation(tmp_path, monkeypatch, outcome_status):
    calls = []

    async def revise(context, runtime):
        calls.append(context)
        (context.runtime_context.workspace / "SYSTEM_PROMPT.md").write_text("Revised strategy")
        return AgentOutcome(solution="Revised strategy", status=outcome_status)

    monkeypatch.setattr(evolution, "load_agent", lambda source: (None, revise))
    monkeypatch.setattr(evolution, "validate_agent_directory", lambda *a, **kw:
                        evolution.ValidationResult(False, [{"check": "advisory", "result": "failed"}]))
    result = asyncio.run(evolution.evolve_agent(
        target_role="client_agent", previous_directory=ROOT / "client_agent",
        candidate_directory=tmp_path / "candidate", guide="Revise the strategy.",
        config=AppConfig(apply_all_revisions=True),
        model=ModelClient(ModelSettings(), responder=lambda *a: "unused"),
    ))
    assert result.activated
    assert not result.validation.passed
    assert result.report["revision_policy"] == "always_apply"
    assert result.report["files_changed"]
    assert len(calls) == 1


def test_always_apply_aborts_if_failed_revision_authored_nothing(tmp_path, monkeypatch):
    async def fail(context, runtime):
        return AgentOutcome(solution="", status="failed")

    monkeypatch.setattr(evolution, "load_agent", lambda source: (None, fail))
    monkeypatch.setattr(evolution, "_MAX_REVISION_ATTEMPTS", 1)
    with pytest.raises(RuntimeError, match="failed without producing any changes"):
        asyncio.run(evolution.evolve_agent(
            target_role="client_agent", previous_directory=ROOT / "client_agent",
            candidate_directory=tmp_path / "candidate", guide="Revise the strategy.",
            # Copying a library into the workspace is not an authored revision.
            seed_library={"previous": "An existing insight"},
            config=AppConfig(apply_all_revisions=True),
            model=ModelClient(ModelSettings(), responder=lambda *a: "unused"),
        ))


@pytest.mark.parametrize("phase", ["bootstrap", "revision"])
def test_always_apply_does_not_rescue_crashes(tmp_path, monkeypatch, phase):
    store = VersionStore(tmp_path / "experiment")
    store.initialize(ROOT / "client_agent", ROOT / "server_agent")
    before = store.load()

    async def fail(**kwargs):
        raise RuntimeError("No revision was produced")

    monkeypatch.setattr(server, "evolve_agent", fail)
    kwargs = dict(version_store=store, config=AppConfig(apply_all_revisions=True),
                  model=ModelClient(ModelSettings(), responder=lambda *a: "unused"))
    with pytest.raises(RuntimeError, match="No revision was produced"):
        if phase == "bootstrap":
            asyncio.run(server.bootstrap_agents(**kwargs))
        else:
            asyncio.run(server.evolve_harness(**kwargs, reasoning_traces=[]))
    assert store.load() == before


@pytest.mark.parametrize("failure_stage", ["solve", "grade"])
def test_batch_stops_and_cleans_scope_on_provider_failure(tmp_path, monkeypatch, failure_stage):
    calls, grades, closed = [], [], []

    @contextmanager
    def scope(context):
        try:
            yield context
        finally:
            closed.append(1)

    cases = [federation.BenchmarkCase(str(i), "task", tmp_path / str(i), runtime_scope=scope)
             for i in range(5)]

    def solve(*args, **kwargs):
        calls.append(kwargs["task_id"])
        if failure_stage == "solve":
            raise FatalProviderError("Gemini HTTP 402: credits depleted")
        return AgentOutcome(solution="done", telemetry={
            "agent_version_digest": directory_digest(ROOT / "client_agent"),
        })

    def grade(*args):
        grades.append(1)
        raise FatalProviderError("Gemini HTTP 403: judge blocked")

    monkeypatch.setattr(federation, "solve_task_sync", solve)
    with pytest.raises(FatalProviderError):
        federation._run_batch(
            cases, insight_library={}, agent_version=ROOT / "client_agent",
            config=AppConfig(), config_path=ROOT / "config.yaml", prompt_file=None,
            num_workers=1, grade=grade, include_traces=True,
            test_model=ModelClient(ModelSettings(), responder=lambda *a: "unused"),
        )
    assert calls == ["0"]
    assert closed == [1]
    assert len(grades) == (1 if failure_stage == "grade" else 0)


def test_parallel_failure_is_observed_without_submitting_remaining_cases():
    started = []
    slow_started, failed = threading.Event(), threading.Event()

    def work(i):
        started.append(i)
        if i == 0:
            slow_started.set()
            assert failed.wait(2)
            time.sleep(0.05)
            return i
        assert slow_started.wait(2)
        failed.set()
        raise FatalProviderError("Gemini HTTP 403: blocked")

    with pytest.raises(FatalProviderError):
        parallel_utils.parallel_map_ordered(work, range(100), num_workers=2)
    assert sorted(started) == [0, 1]
    assert parallel_utils.parallel_map_ordered(lambda n: n * 2, range(6), num_workers=3) == list(range(0, 12, 2))


def test_batch_abort_cancels_active_communication():
    async def check():
        abort = ProviderBatchAbort()
        token = provider_batch_abort.set(abort)
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def communicate():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        try:
            task = asyncio.create_task(hermes._communicate_with_provider_abort(
                SimpleNamespace(communicate=communicate),
            ))
            await started.wait()
            abort.abort(FatalProviderError("Gemini HTTP 402: credits depleted"))
            with pytest.raises(FatalProviderError, match="402"):
                await asyncio.wait_for(task, timeout=1)
            assert cancelled.is_set()
        finally:
            provider_batch_abort.reset(token)

    asyncio.run(check())


@pytest.mark.parametrize("status", [401, 402, 403, 404, 429, 503])
@pytest.mark.parametrize("in_payload", [True, False])
def test_local_grader_api_failure_is_fatal(monkeypatch, status, in_payload):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    error = f"RuntimeError: Gemini visual judge HTTP error {status}"
    backend = SimpleNamespace(run_eval=lambda *a, **kw: (
        ([1, 0.0, 0, error] if in_payload else None),
        ({} if in_payload else {"harness_output": error}),
    ))
    with pytest.raises(FatalProviderError, match=f"HTTP {status}"):
        sab._backend_eval(backend, "local", {}, judge_model="gemini-test", judge_api_key_env=None)


def test_ordinary_grading_failure_is_still_a_task_result():
    backend = SimpleNamespace(run_eval=lambda *a, **kw: (None, {"error": "program crashed"}))
    grade = sab._backend_eval(backend, "local", {}, judge_model=None, judge_api_key_env=None)
    assert grade["score"] == 0 and grade["verifier_status"] == "error"
    raise_for_gemini_error_text("Generated program failed its assertion", model="gemini-test")


def test_retry_exhausted_agent_http_error_is_fatal():
    with pytest.raises(FatalProviderError, match="503"):
        raise_for_gemini_error_text("Gemini HTTP 503 (UNAVAILABLE): retry exhausted", model="gemini-test")


def test_direct_provider_does_not_retry_billing_error(monkeypatch):
    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "fake-provider-key")

    def chat(*args, **kwargs):
        calls.append(1)
        error = RuntimeError("credits depleted: fake-provider-key")
        error.code = 402
        raise error

    monkeypatch.setitem(sys.modules, "utils", SimpleNamespace(chat=chat))
    model = ModelClient(ModelSettings(provider="gemini", name="gemini-test", retries=5))
    with pytest.raises(FatalProviderError, match="402") as error:
        asyncio.run(model.generate("Hello"))
    assert len(calls) == 1
    assert "fake-provider-key" not in str(error.value)


def test_failed_batch_never_creates_completed_round_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / "experiment"
    calls = []

    async def bootstrap(**kwargs):
        return kwargs["version_store"].load()

    def solve(*args, **kwargs):
        calls.append(kwargs["task_id"])
        raise FatalProviderError("Gemini HTTP 403: blocked")

    monkeypatch.setattr(federation, "bootstrap_agents", bootstrap)
    monkeypatch.setattr(federation, "solve_task_sync", solve)
    with pytest.raises(FatalProviderError, match="403"):
        federation.run_federated_experiment(
            train_cases=[federation.BenchmarkCase("train", "task", tmp_path / "train")],
            held_out_cases=[federation.BenchmarkCase("held", "task", tmp_path / "held")],
            grade=lambda *a: pytest.fail("Provider errors must not reach grading"),
            output_dir=output, rounds=5, num_workers=1, config=AppConfig(),
            client_seed=ROOT / "client_agent", server_seed=ROOT / "server_agent",
            test_model=ModelClient(ModelSettings(), responder=lambda *a: "unused"),
        )
    assert calls == ["held"]
    assert not list(output.glob("round_*/checkpoint.json"))
    assert not (output / "final_held_out_checkpoint.json").exists()
    assert VersionStore(output).load()["active_version"] == 0
