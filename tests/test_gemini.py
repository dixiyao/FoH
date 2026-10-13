"""Direct Gemini routing, CLI, and credential-boundary regressions (no paid calls)."""
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime import hermes, hermes_driver
from agent_runtime.config import AppConfig, GEMINI_BASE_URL, ModelSettings, load_config
from agent_runtime.contracts import AgentContext, RuntimeContext
from agent_runtime.model import ModelClient
from agent_runtime.runtime import AgentRuntime
from agent_runtime.tools import ToolRegistry
from agent_runtime.provider_errors import FatalProviderError
from tasks import task_scienceagentbench as sab

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("prefix", ["", "google/", "gemini/", "models/"])
def test_gemini_config_defaults_and_model_names(prefix):
    settings = ModelSettings(provider="gemini", name=prefix + "gemini-2.5-flash")
    assert settings.name == "gemini-2.5-flash"
    assert settings.base_url == GEMINI_BASE_URL
    assert settings.api_key_env == "GEMINI_API_KEY"
    configured = load_config(ROOT / "config_gemini.yaml").model
    assert configured.provider == "gemini"
    assert configured.base_url == GEMINI_BASE_URL
    assert configured.api_key_env == "GEMINI_API_KEY"


@pytest.mark.parametrize("endpoint", [
    "https://openrouter.ai/api/v1", "http://generativelanguage.googleapis.com/v1beta",
    GEMINI_BASE_URL + "/openai", GEMINI_BASE_URL + "?key=secret",
])
def test_gemini_rejects_incompatible_endpoints(endpoint):
    with pytest.raises(ValueError, match="native API endpoint"):
        ModelSettings(provider="gemini", name="gemini-2.5-flash", base_url=endpoint)


def test_gemini_rejects_unsupported_sampling():
    with pytest.raises(ValueError, match="top_k"):
        ModelSettings(provider="gemini", name="gemini-2.5-flash", top_k=10)


@pytest.mark.parametrize("key_env", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_gemini_preflight_requires_selected_credential(monkeypatch, tmp_path, key_env):
    monkeypatch.setattr(hermes, "_resolve_external", lambda settings: (tmp_path,) * 3)
    monkeypatch.delenv(key_env, raising=False)
    config = AppConfig(model=ModelSettings(
        provider="gemini", name="gemini-2.5-flash", api_key_env=key_env,
    ))
    with pytest.raises(hermes.HermesBridgeError, match=key_env):
        hermes.preflight_external_hermes(config)
    monkeypatch.setenv(key_env, "test-google-key")
    hermes.preflight_external_hermes(config)
    with pytest.raises(hermes.HermesBridgeError, match="provider"):
        hermes.preflight_external_hermes(AppConfig())


@pytest.mark.parametrize("switch_provider", [False, True])
@pytest.mark.parametrize("key_flag", ["--api_key", "--api-key"])
def test_science_cli_preserves_or_switches_provider(monkeypatch, tmp_path, switch_provider, key_flag):
    monkeypatch.setenv("GEMINI_API_KEY", "old-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "old-google-key")
    config = load_config(ROOT / ("config.yaml" if switch_provider else "config_gemini.yaml"))
    if not switch_provider:
        config = replace(config, model=replace(config.model, api_key_env="GOOGLE_API_KEY"))
    monkeypatch.setattr(sab, "load_config", lambda path: config)
    monkeypatch.setattr(sab, "resolve_checkout", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(sab, "load_instances", lambda *a, **kw: [])
    monkeypatch.setattr(sab, "validate_instances", lambda *a: [])
    monkeypatch.setattr(sab, "select_instances", lambda *a: [{"instance_id": 1}])
    monkeypatch.setattr(sab, "resolve_artifacts", lambda *a: tmp_path)
    monkeypatch.setattr(sab, "preflight", lambda **kw: None)
    captured = []

    class CapturedConfig(Exception):
        pass

    def capture(actual):
        captured.append(actual)
        raise CapturedConfig

    monkeypatch.setattr(sab, "preflight_external_hermes", capture)
    argv = ["science", "--no-grade", "--agent-model", "google/gemini-2.5-pro"]
    if switch_provider:
        argv += ["--provider", "gemini"]
    argv += [key_flag, "test-cli-secret"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(CapturedConfig):
        sab.main()
    assert captured[0].model.provider == "gemini"
    assert captured[0].model.name == "gemini-2.5-pro"
    assert captured[0].model.base_url == GEMINI_BASE_URL
    assert captured[0].apply_all_revisions
    assert captured[0].model.api_key_value == "test-cli-secret"
    assert captured[0].model.api_key() == "test-cli-secret"
    assert "test-cli-secret" not in json.dumps(captured[0].model.public_dict())
    assert "test-cli-secret" not in json.dumps(captured[0].public_dict())
    assert "test-cli-secret" not in repr(captured[0])
    assert os.environ["GEMINI_API_KEY"] == "old-gemini-key"
    assert os.environ["GOOGLE_API_KEY"] == "old-google-key"


@pytest.mark.parametrize("flags", [[], ["--no-grade"], ["--eval-only"],
                                   ["--smoke", "--judge-model", "gemini-3.8-flash"]])
def test_science_cli_never_falls_back_to_inherited_key(monkeypatch, capsys, flags):
    for key_env in ("FOH_SCIENCEAGENTBENCH_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                    "GOOGLE_AI_STUDIO_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key_env, "inherited-secret")
    monkeypatch.setattr(sys, "argv", ["science", *flags])

    def unexpected_setup(*args, **kwargs):
        pytest.fail("Missing explicit key must fail before benchmark setup")

    monkeypatch.setattr(sab, "resolve_checkout", unexpected_setup)
    with pytest.raises(SystemExit) as error:
        sab.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--api_key is required" in stderr
    assert "inherited-secret" not in stderr


def test_science_cli_key_reaches_judge_preflight(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "old-gemini-key")
    monkeypatch.setattr(sab, "resolve_checkout", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(sab, "load_instances", lambda *a, **kw: [])
    monkeypatch.setattr(sab, "validate_instances", lambda *a: [])
    monkeypatch.setattr(sab, "select_instances", lambda *a: [{"instance_id": 1}])
    monkeypatch.setattr(sab, "resolve_artifacts", lambda *a: tmp_path)

    class ReachedPreflight(Exception):
        pass

    def preflight(**kwargs):
        assert kwargs["judge_api_key"] == "test-cli-secret"
        assert kwargs["environment"]["GEMINI_API_KEY"] == "old-gemini-key"
        raise ReachedPreflight

    monkeypatch.setattr(sab, "preflight", preflight)
    argv = ["science", "--api_key", "test-cli-secret", "--judge-model", "gemini-3.8-flash"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedPreflight):
        sab.main()


@pytest.mark.parametrize("key", ["", "   "])
def test_science_cli_rejects_empty_key(monkeypatch, capsys, key):
    monkeypatch.setattr(sys, "argv", ["science", "--api_key", key])
    with pytest.raises(SystemExit) as error:
        sab.main()
    assert error.value.code == 2
    assert "--api_key must not be empty" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
@pytest.mark.parametrize("finish_kind", ["solution", "insight_library"])
@pytest.mark.parametrize("failure_status", [None, 401, 402, 403, 404, "empty"])
def test_external_bridge_and_driver_routing(monkeypatch, tmp_path, provider, finish_kind, failure_status):
    """Exercise the real parent request and driver, replacing only OS/kernel execution."""
    key_env = "GOOGLE_API_KEY" if provider == "gemini" else "OPENROUTER_API_KEY"
    key = "test-provider-secret"
    monkeypatch.setenv(key_env, "wrong-environment-key")
    monkeypatch.setattr(hermes_driver, "_install_gemini_error_guard", lambda *args: lambda: None)
    settings = ModelSettings(provider=provider, name="gemini-2.5-flash", api_key_env=key_env,
                             api_key_value=key)
    expected_url = GEMINI_BASE_URL if provider == "gemini" else "https://openrouter.ai/api/v1"
    config = AppConfig(model=settings)
    context = AgentContext(
        objective="Write a Python program", role="client" if finish_kind == "solution" else "server",
        insight_library={}, runtime_context=RuntimeContext(workspace=tmp_path),
    )
    runtime = AgentRuntime(model=ModelClient(settings), limits=config.limits,
                           tools=ToolRegistry(tmp_path), base_system_prompt="Use tools.",
                           hermes_settings=config.hermes)
    monkeypatch.setattr(hermes, "_resolve_external", lambda settings: (tmp_path, Path(sys.executable), tmp_path))
    monkeypatch.setattr(hermes, "_sandbox_command", lambda context, **kw: [str(kw["request_path"]), str(kw["result_path"])])
    constructed = []

    class FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            assert kwargs["provider"] == kwargs["requested_provider"] == provider
            assert kwargs["base_url"] == expected_url
            assert kwargs["api_key"] == key
            assert kwargs["api_mode"] == "chat_completions"
            assert ("image_gen" in kwargs["enabled_toolsets"]) == (provider == "openrouter")

        def run_conversation(self, *args, **kwargs):
            assert key_env not in os.environ
            if failure_status == "empty":
                return {"failed": True, "completed": False, "api_calls": 1,
                        "output_tokens": 0, "tool_calls": 0,
                        "final_response": f"Invalid API response after 3 retries: {key}"}
            if failure_status:
                raise hermes_driver._FatalProviderResponse({
                    "status_code": failure_status, "reason": "API_KEY_SERVICE_BLOCKED",
                    "message": f"HTTP {failure_status}: rejected {key}",
                })
            return {"completed": True, "api_calls": 1, "input_tokens": 10,
                    "output_tokens": 5, "final_response": (
                        "Done" if finish_kind == "solution" else '{"testing": "Verify outputs."}'
                    )}

        def close(self):
            pass

    async def launch(request_path, result_path, **kwargs):
        request = json.loads(Path(request_path).read_text())
        assert key not in json.dumps(request)
        assert request["provider"] == provider
        assert kwargs["env"][key_env] == key
        if provider == "gemini":
            assert "generativelanguage.googleapis.com" in kwargs["env"]["NO_PROXY"]
        profile = json.loads((Path(kwargs["env"]["HERMES_HOME"]) / "config.yaml").read_text())
        assert key not in json.dumps(profile)
        assert ("image_gen" in profile) == (provider == "openrouter")
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=FakeAgent))
            patch.setattr(sys, "argv", ["driver", request_path, result_path])
            patch.setenv(key_env, key)
            # Restore driver's process-only environment changes after this test.
            patch.setenv("HERMES_YOLO_MODE", "1")
            patch.setenv("HERMES_ACCEPT_HOOKS", "1")
            returncode = hermes_driver.main()
        assert key not in Path(result_path).read_text()

        async def communicate():
            return b"", b""

        return SimpleNamespace(returncode=returncode, communicate=communicate)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    if failure_status:
        message = "failed before producing any output" if failure_status == "empty" else f"HTTP {failure_status}"
        with pytest.raises(FatalProviderError, match=message) as error:
            asyncio.run(hermes.run_hermes_agent(context, runtime, finish_kind=finish_kind))
        assert key not in str(error.value)
        if failure_status != "empty":
            assert "API_KEY_SERVICE_BLOCKED" in str(error.value)
        assert len(constructed) == 1
        return
    outcome = asyncio.run(hermes.run_hermes_agent(context, runtime, finish_kind=finish_kind))
    assert outcome.status == "completed"
    assert len(constructed) == 1
    assert outcome.telemetry["hermes_backend"] == "external_process"


def test_pinned_native_adapter_tool_roundtrip(tmp_path):
    """Optional installed-kernel integration: real native wire conversion, fake HTTP."""
    source = ROOT / ".hermes/hermes-agent"
    if not (source / "agent/gemini_native_adapter.py").is_file():
        pytest.skip("Pinned Hermes checkout is not installed")
    script = r'''
import json
import httpx
from agent.gemini_native_adapter import GeminiNativeClient

requests = []
def respond(request):
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert request.headers["x-goog-api-key"] == "fake-google-key"
    body = json.loads(request.content)
    requests.append(body)
    parts = ([{"functionCall": {"name": "inspect_file", "args": {"path": "data.csv"}}}]
             if len(requests) == 1 else [{"text": "Done"}])
    return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": parts},
                                                  "finishReason": "STOP"}],
                                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}})

client = GeminiNativeClient(api_key="fake-google-key", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
messages = [{"role": "user", "content": "Inspect data.csv"}]
tools = [{"type": "function", "function": {"name": "inspect_file", "description": "Inspect a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
response = client.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools,
                                          max_tokens=100, temperature=0.4, top_p=1.0)
message = response.choices[0].message
call = message.tool_calls[0]
assert call.function.name == "inspect_file"
messages += [{"role": "assistant", "content": None, "tool_calls": [{"id": call.id, "type": "function",
              "function": {"name": call.function.name, "arguments": call.function.arguments}}]},
             {"role": "tool", "tool_call_id": call.id, "content": "3 rows"}]
response = client.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)
assert response.choices[0].message.content == "Done"
assert requests[0]["generationConfig"]["maxOutputTokens"] == 100
assert any("functionResponse" in p for c in requests[1]["contents"] for p in c["parts"])
client.close()
'''
    environment = dict(os.environ, PYTHONPATH=str(source), HERMES_HOME=str(tmp_path))
    completed = subprocess.run([sys.executable, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
