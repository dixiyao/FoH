"""Shared Gemini pacing and recovery, without network calls or real waits."""
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from agent_runtime import hermes, hermes_driver as driver


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 1000.0
        sleeps = []

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    value = Clock()
    monkeypatch.setattr(driver.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(driver.time, "sleep", value.sleep)
    monkeypatch.setattr(driver.random, "uniform", lambda *args: 0)
    return value


def quota(tmp_path, budget=100):
    path = tmp_path / "quota.json"
    path.write_text("{}")
    return driver._GeminiQuota(str(path), budget)


def test_workers_share_budget_and_cooldown_but_models_are_independent(tmp_path, clock):
    first = quota(tmp_path, 10)
    alias = tmp_path / "alias.json"
    os.link(first.path, alias)
    second = driver._GeminiQuota(str(alias), 10)
    first.acquire("flash", {"text": "a" * 15})
    second.acquire("flash", {"text": "b" * 15})
    assert clock.now == 1060
    first.cooldown("flash", 90)
    second.acquire("judge", {"text": "c"})
    assert clock.now == 1060
    second.acquire("flash", {"text": "d"})
    assert clock.now == 1150
    assert "text" not in Path(first.path).read_text()


def test_single_request_larger_than_budget_does_not_deadlock(tmp_path, clock):
    limiter = quota(tmp_path, 1)
    limiter.acquire("flash", {"text": "large"})
    limiter.acquire("flash", {"text": "large"})
    assert clock.sleeps == [60]


def test_parent_links_one_state_into_each_sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("FOH_GEMINI_TPM", "1500000")
    paths = [tmp_path / "one", tmp_path / "two"]
    for path in paths:
        path.mkdir()
    configs = [hermes._gemini_rate_limit_request(path) for path in paths]
    assert os.path.samefile(configs[0]["state_path"], configs[1]["state_path"])
    assert configs[0]["tokens_per_minute"] == 1500000
    monkeypatch.setenv("FOH_GEMINI_TPM", "0")
    third = tmp_path / "three"
    third.mkdir()
    disabled = hermes._gemini_rate_limit_request(third)
    assert disabled["tokens_per_minute"] == 0
    assert os.path.samefile(disabled["state_path"], configs[0]["state_path"])
    monkeypatch.setenv("FOH_GEMINI_TPM", "-1")
    with pytest.raises(ValueError):
        hermes._gemini_rate_limit_request(tmp_path)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status, expected_delay", [(429, 90), (503, 600)])
@pytest.mark.parametrize("budget", [0, 100])
def test_transient_retries_preserve_request_and_honor_shared_delay(tmp_path, clock, stream, status, expected_delay, budget):
    requests = []

    def respond(request):
        requests.append(request.content)
        if len(requests) == 1:
            return httpx.Response(status, headers={"Retry-After": "75"}, json={"error": {
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "90s"}]
            }})
        return httpx.Response(200, json={"ok": True})

    client = driver._GeminiRateLimitedHTTP(httpx.Client(transport=httpx.MockTransport(respond)), quota(tmp_path, budget))
    url = "https://example.test/models/flash:generateContent"
    if stream:
        with client.stream("POST", url, json={"contents": "same"}) as response:
            response.read()
            assert response.json() == {"ok": True}
    else:
        assert client.post(url, json={"contents": "same"}).json() == {"ok": True}
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert clock.now == 1000 + expected_delay
    client.close()


def test_overload_honors_longer_provider_delay(clock):
    response = httpx.Response(503, headers={"Retry-After": "900"})
    assert driver._gemini_retry_delay(response, 0) == 900


def test_cooldown_remains_shared_when_token_pacing_is_disabled(tmp_path, clock):
    first = quota(tmp_path, 0)
    second = driver._GeminiQuota(first.path, 0)
    for _ in range(3):
        first.acquire("flash", {"text": "large"})
    assert clock.sleeps == []
    first.cooldown("flash", 600)
    second.acquire("judge", {})
    assert clock.now == 1000
    second.acquire("flash", {})
    assert clock.now == 1600


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 500])
def test_transport_leaves_other_errors_to_existing_policy(tmp_path, clock, status):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status)

    client = driver._GeminiRateLimitedHTTP(httpx.Client(transport=httpx.MockTransport(respond)), quota(tmp_path))
    assert client.post("https://example.test/models/flash:generateContent", json={}).status_code == status
    assert len(calls) == 1
    assert clock.sleeps == []
    client.close()


def test_quota_updates_are_atomic_across_processes(tmp_path):
    limiter = quota(tmp_path, 1000000)
    script = """
import sys
from agent_runtime.hermes_driver import _GeminiQuota
q = _GeminiQuota(sys.argv[1], 1000000)
for _ in range(20):
    q.acquire('flash', {})
"""
    processes = [subprocess.Popen([sys.executable, "-c", script, limiter.path]) for _ in range(4)]
    for process in processes:
        assert process.wait(timeout=15) == 0
    assert len(json.loads(Path(limiter.path).read_text())["flash"]["events"]) == 80


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [429, 503])
def test_pinned_adapter_recovers_and_bounds_retries(tmp_path, stream, status):
    root = Path(__file__).resolve().parents[1]
    source = root / ".hermes/hermes-agent"
    if not (source / "agent/gemini_native_adapter.py").is_file():
        pytest.skip("Pinned Hermes checkout is not installed")
    script = r'''
import json
import sys
from pathlib import Path
import httpx
from agent_runtime import hermes_driver as d
from agent.gemini_native_adapter import GeminiNativeClient

now = [1000.0]
d.time.monotonic = lambda: now[0]
d.time.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
d.random.uniform = lambda *args: 0
path = Path(sys.argv[1]) / 'quota.json'
path.write_text('{}')
original_init = GeminiNativeClient.__init__
restore = d._install_gemini_error_guard({'state_path': str(path), 'tokens_per_minute': 1500000})
stream = sys.argv[2] == 'True'
status = int(sys.argv[3])
calls = []
always_fail = False

def respond(request):
    calls.append(request)
    if len(calls) == 1 or always_fail:
        return httpx.Response(status, json={'error': {'status': 'UNAVAILABLE' if status == 503 else 'RESOURCE_EXHAUSTED', 'message': 'busy'}})
    payload = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Done'}]},
                               'finishReason': 'STOP'}]}
    if stream:
        return httpx.Response(200, headers={'Content-Type': 'text/event-stream'},
                              content=('data: ' + json.dumps(payload) + '\n\n').encode())
    return httpx.Response(200, json=payload)

client = GeminiNativeClient(api_key='fake-key', http_client=httpx.Client(transport=httpx.MockTransport(respond)))
def generate():
    response = client.chat.completions.create(model='gemini-3.7-flash',
                                             messages=[{'role': 'user', 'content': 'hello'}], stream=stream)
    return list(response) if stream else response

response = generate()
if stream:
    assert any(chunk.choices[0].delta.content == 'Done' for chunk in response)
else:
    assert response.choices[0].message.content == 'Done'
assert len(calls) == 2
assert now[0] == (1600 if status == 503 else 1060)
always_fail = True
try:
    generate()
except d._FatalProviderResponse as exc:
    assert exc.details['status_code'] == status
    assert exc.details['reason'] == ('overload_retries_exhausted' if status == 503 else 'rate_limit_retries_exhausted')
else:
    raise AssertionError('Persistent provider errors must remain failures')
assert len(calls) == 8
if status == 503:
    assert now[0] == 4600  # One recovered retry, then exactly five failed retries.
client.close()
restore()
assert GeminiNativeClient.__init__ is original_init
'''
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(source)]), HERMES_HOME=str(tmp_path))
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(stream), str(status)],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
