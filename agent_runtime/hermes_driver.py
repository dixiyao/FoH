#!/usr/bin/env python3
"""Small process-side adapter for one pinned upstream Hermes Agent turn.

This file intentionally contains no FoH imports.  It is launched with the
Hermes virtualenv Python and imports ``run_agent.AIAgent`` from the pinned
source checkout placed first on ``PYTHONPATH`` by the parent bridge.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import ipaddress
import json
import math
import os
import random
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_MAX_TRANSCRIPT_EVENTS = 199  # parent reserves one final-answer event
_MAX_TRANSCRIPT_TEXT = 8_000


class _FatalProviderResponse(BaseException):
    """Escape Hermes' task-level Exception handlers on a permanent API failure."""

    def __init__(self, details: Dict[str, Any]):
        self.details = details
        super().__init__(details["message"])


class _GeminiQuota:
    """Rolling estimated input-token budget shared by worker processes.

    Only counters/timestamps are persisted, never request text or credentials.
    Estimates use native request UTF-8 bytes / 3 with headroom configured by
    the operator; this is not an exact tokenizer or a project-wide quota meter.
    """

    def __init__(self, path: str, budget: int):
        self.path = path
        self.budget = budget

    @contextmanager
    def _locked(self):
        with open(self.path, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = json.load(handle)
            yield state
            handle.seek(0)
            json.dump(state, handle)
            handle.truncate()
            handle.flush()

    def acquire(self, model: str, payload: Dict[str, Any]) -> None:
        cost = max(1, math.ceil(len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) / 3))
        while True:
            with self._locked() as state:
                now = time.monotonic()
                quota = state.setdefault(model, {})
                events = [event for event in quota.get("events", []) if event[0] > now - 60]
                quota["events"] = events
                delay = max(0.0, quota.get("cooldown", 0) - now)
                if self.budget > 0 and events and sum(event[1] for event in events) + cost > self.budget:
                    delay = max(delay, events[0][0] + 60 - now)
                if delay <= 0:
                    # An oversized request gets an otherwise empty window;
                    # never deadlock waiting for it to fit a smaller budget.
                    events.append([now, cost])
                    return
            time.sleep(min(delay, 60))

    def cooldown(self, model: str, seconds: float) -> None:
        with self._locked() as state:
            quota = state.setdefault(model, {})
            quota["cooldown"] = max(quota.get("cooldown", 0), time.monotonic() + seconds)


def _gemini_retry_delay(response: Any, attempt: int) -> float:
    overloaded = response.status_code == 503
    delay = 600.0 if overloaded else min(240.0, 60.0 * 2 ** attempt)
    hints = [response.headers.get("Retry-After", "")]
    try:
        details = response.json().get("error", {}).get("details", [])
        hints.extend(
            str(detail.get("retryDelay", "")).removesuffix("s")
            for detail in details if isinstance(detail, dict)
            and str(detail.get("@type", "")).endswith("/google.rpc.RetryInfo")
        )
    except (ValueError, AttributeError, TypeError):
        pass
    for hint in hints:
        try:
            seconds = float(hint)
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        except (ValueError, TypeError):
            pass
    return delay if overloaded else delay + random.uniform(0, 5)


class _GeminiRateLimitedHTTP:
    """Throttle each native request, including retries and streaming opens."""

    def __init__(self, transport: Any, quota: _GeminiQuota):
        self.transport = transport
        self.quota = quota

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def _model(self, url: str) -> str:
        return str(url).split("/models/", 1)[-1].split(":", 1)[0]

    def _retry(self, response: Any, model: str, attempt: int) -> None:
        if attempt == 5:
            from agent.gemini_native_adapter import gemini_http_error
            error = gemini_http_error(response)
            raise _FatalProviderResponse({
                "status_code": response.status_code,
                "reason": ("overload_retries_exhausted" if response.status_code == 503
                           else "rate_limit_retries_exhausted"),
                "message": str(error),
            })
        self.quota.cooldown(model, _gemini_retry_delay(response, attempt))

    def post(self, url, **kwargs):
        model = self._model(url)
        for attempt in range(6):
            self.quota.acquire(model, kwargs.get("json") or {})
            response = self.transport.post(url, **kwargs)
            if response.status_code not in {429, 503}:
                return response
            try:
                self._retry(response, model, attempt)
            finally:
                response.close()

    @contextmanager
    def stream(self, method, url, **kwargs):
        model = self._model(url)
        for attempt in range(6):
            self.quota.acquire(model, kwargs.get("json") or {})
            with self.transport.stream(method, url, **kwargs) as response:
                if response.status_code not in {429, 503}:
                    # Never retry after any stream output has been delivered.
                    yield response
                    return
                response.read()
                self._retry(response, model, attempt)


def _install_gemini_error_guard(rate_limit=None):
    """Observe native HTTP failures before Hermes turns them into failed turns."""
    from agent import gemini_native_adapter

    original = gemini_native_adapter.gemini_http_error
    original_init = None
    if rate_limit:
        original_init = gemini_native_adapter.GeminiNativeClient.__init__
        quota = _GeminiQuota(rate_limit["state_path"], rate_limit["tokens_per_minute"])

        def limited_init(client, *args, **kwargs):
            original_init(client, *args, **kwargs)
            client._http = _GeminiRateLimitedHTTP(client._http, quota)

        gemini_native_adapter.GeminiNativeClient.__init__ = limited_init

    def guarded(response, **kwargs):
        error = original(response, **kwargs)
        if error.status_code in {401, 402, 403, 404}:
            raise _FatalProviderResponse({
                "status_code": error.status_code,
                "reason": error.details.get("reason", ""),
                "message": str(error),
            })
        return error

    gemini_native_adapter.gemini_http_error = guarded

    def restore():
        gemini_native_adapter.gemini_http_error = original
        if original_init is not None:
            gemini_native_adapter.GeminiNativeClient.__init__ = original_init

    return restore


_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.I),
)


def _public_https_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("configured search endpoint must be a public HTTPS URL")
    for address in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ValueError("configured search endpoint resolves to a private address")
    return value


def _install_configured_search(config: Dict[str, Any], api_key: str, model: str) -> None:
    """Register FoH's selected search provider without exposing its key."""
    from agent.web_search_provider import WebSearchProvider
    from agent.web_search_registry import register_provider

    provider_name = str(config.get("provider") or "").lower()
    configured_max = max(1, min(10, int(config.get("max_results") or 5)))
    base_url = str(config.get("base_url") or "").strip()

    class FoHConfiguredSearch(WebSearchProvider):
        @property
        def name(self) -> str:
            return "foh-configured"

        @property
        def display_name(self) -> str:
            return f"FoH configured {provider_name}"

        def is_available(self) -> bool:
            return bool(api_key)

        def supports_extract(self) -> bool:
            return False

        def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
            count = max(1, min(configured_max, int(limit or configured_max)))
            if provider_name == "tavily":
                url = base_url or "https://api.tavily.com/search"
                payload = {
                    "api_key": api_key,
                    "query": query,
                    "max_results": count,
                    "search_depth": "basic",
                }
                headers: Dict[str, str] = {}
            elif provider_name == "brave":
                root = base_url or "https://api.search.brave.com/res/v1/web/search"
                url = root + "?" + urllib.parse.urlencode({"q": query, "count": count})
                payload = None
                headers = {
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                }
            elif provider_name == "openrouter":
                url = base_url or "https://openrouter.ai/api/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "plugins": [{"id": "web", "max_results": count}],
                    "max_tokens": 1,
                }
                headers = {"Authorization": f"Bearer {api_key}"}
            else:
                return {"success": False, "error": "unsupported FoH search provider"}
            try:
                data = None
                if payload is not None:
                    data = json.dumps(payload).encode("utf-8")
                    headers = {**headers, "Content-Type": "application/json"}
                request = urllib.request.Request(
                    _public_https_url(url), data=data, headers=headers,
                    method="POST" if payload is not None else "GET",
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=20) as response:
                    result = json.loads(response.read(500_000).decode("utf-8"))
                if provider_name == "tavily":
                    raw_items = result.get("results") or []
                    items = [
                        {
                            "title": str(item.get("title") or "")[:300],
                            "url": str(item.get("url") or "")[:2000],
                            "description": str(item.get("content") or "")[:1000],
                            "position": index,
                        }
                        for index, item in enumerate(raw_items[:count], start=1)
                        if isinstance(item, dict)
                    ]
                elif provider_name == "brave":
                    raw_items = (result.get("web") or {}).get("results") or []
                    items = [
                        {
                            "title": str(item.get("title") or "")[:300],
                            "url": str(item.get("url") or "")[:2000],
                            "description": str(item.get("description") or "")[:1000],
                            "position": index,
                        }
                        for index, item in enumerate(raw_items[:count], start=1)
                        if isinstance(item, dict)
                    ]
                else:
                    choices = result.get("choices") or []
                    message = choices[0].get("message") if choices else {}
                    citations = (message or {}).get("annotations") or []
                    items = []
                    for annotation in citations:
                        citation = annotation.get("url_citation") if isinstance(annotation, dict) else None
                        if isinstance(citation, dict):
                            items.append({
                                "title": str(citation.get("title") or "")[:300],
                                "url": str(citation.get("url") or "")[:2000],
                                "description": str(citation.get("content") or "")[:1000],
                                "position": len(items) + 1,
                            })
                        if len(items) >= count:
                            break
                return {"success": True, "data": {"web": items}}
            except Exception as exc:
                error = f"configured search failed: {type(exc).__name__}: {exc}"
                if api_key:
                    error = error.replace(api_key, "[REDACTED]")
                return {"success": False, "error": error}

    register_provider(FoHConfiguredSearch())


def _safe_text(value: Any, secret: str, *, limit: int = _MAX_TRANSCRIPT_TEXT) -> str:
    text = str(value or "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:limit]


def _bounded_value(value: Any, secret: str, *, depth: int = 0) -> Any:
    if depth >= 6:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            _safe_text(key, secret, limit=100): _bounded_value(item, secret, depth=depth + 1)
            for key, item in list(value.items())[:80]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, secret, depth=depth + 1) for item in list(value)[:80]]
    if isinstance(value, str):
        return _safe_text(value, secret)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_text(value, secret)


def _tool_arguments(function: Dict[str, Any], secret: str) -> Dict[str, Any]:
    raw = function.get("arguments", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {"raw": raw}
    if not isinstance(raw, dict):
        raw = {"value": raw}
    return _bounded_value(raw, secret)


def _text_blocks(content: Any, secret: str) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        text = _safe_text(content, secret)
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []
    blocks: List[Dict[str, Any]] = []
    for raw in content[:80]:
        if isinstance(raw, str):
            text = _safe_text(raw, secret)
        elif isinstance(raw, dict) and raw.get("type") in {"text", "output_text"}:
            text = _safe_text(raw.get("text", ""), secret)
        else:
            continue
        if text:
            blocks.append({"type": "text", "text": text})
    return blocks


def _canonical_transcript(
    messages: Iterable[Any], secret: str, final_response: str
) -> List[Dict[str, Any]]:
    """Convert upstream OpenAI-style messages to PinchBench message events."""
    events: List[Dict[str, Any]] = []
    for raw in messages or []:
        if len(events) >= _MAX_TRANSCRIPT_EVENTS or not isinstance(raw, dict):
            break
        role = str(raw.get("role") or "")
        if role == "assistant":
            blocks = _text_blocks(raw.get("content"), secret)
            calls = raw.get("tool_calls") or []
            for call in calls[:80] if isinstance(calls, list) else []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    continue
                original_name = _safe_text(function.get("name", ""), secret, limit=100)
                if not original_name:
                    continue
                mapped_name = {
                    "terminal": "execute_command",
                    # PinchBench's grader names the same capability this way.
                    "image_generate": "generate_image",
                }.get(original_name, original_name)
                block: Dict[str, Any] = {
                    "type": "toolCall",
                    "name": mapped_name,
                    "arguments": _tool_arguments(function, secret),
                }
                if call.get("id"):
                    block["id"] = _safe_text(call["id"], secret, limit=200)
                if mapped_name != original_name:
                    block["hermes_name"] = original_name
                blocks.append(block)
            # The parent appends the unwrapped benchmark-facing solution.
            # Drop Hermes' raw final envelope when it is already the last
            # assistant content so graders do not see wrapper JSON.
            text_only = "".join(
                block.get("text", "") for block in blocks if block.get("type") == "text"
            )
            if not calls and text_only.strip() == str(final_response or "").strip():
                blocks = []
            if blocks:
                events.append(
                    {"type": "message", "message": {"role": "assistant", "content": blocks}}
                )
        elif role == "tool":
            content = _safe_text(raw.get("content", ""), secret)
            event = {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": content}],
                },
            }
            if raw.get("tool_call_id"):
                event["message"]["toolCallId"] = _safe_text(
                    raw["tool_call_id"], secret, limit=200
                )
            events.append(event)
        elif role == "user":
            blocks = _text_blocks(raw.get("content"), secret)
            if blocks:
                events.append(
                    {"type": "message", "message": {"role": "user", "content": blocks}}
                )
    return events


def _tool_summary(messages: Iterable[Any]) -> Dict[str, Any]:
    names: List[str] = []
    failures = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name") if isinstance(function, dict) else None
            if name:
                names.append(str(name)[:100])
        if message.get("role") == "tool":
            content = str(message.get("content", "")).lower()
            if any(marker in content for marker in ("error", "failed", "exception", "traceback")):
                failures += 1
    return {
        "tool_names": sorted(set(names))[:100],
        "tool_calls": len(names),
        "tool_failures": failures,
    }


def _write_result(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _session_turns(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a validated turn sequence, or one legacy FoH prompt.

    The parent serializes typed ``SessionTurn`` objects, but this process is a
    trust boundary and must not let a malformed request silently skip a
    benchmark turn.  A missing/empty list retains the original one-shot
    behavior used by server evolution and non-session benchmarks.
    """
    raw = request.get("session_turns") or []
    if not raw:
        prompt = str(request.get("prompt") or "")
        if not prompt.strip():
            raise ValueError("Hermes request must contain a non-empty prompt")
        return [{"id": "turn_1", "prompt": prompt, "new_session": False}]
    if not isinstance(raw, list):
        raise ValueError("session_turns must be a list")
    turns: List[Dict[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ValueError(f"session turn {index + 1} must be an object")
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"session turn {index + 1} prompt must not be empty")
        turns.append(
            {
                "id": str(value.get("id") or f"turn_{index + 1}"),
                "prompt": prompt,
                "new_session": bool(value.get("new_session", False)),
            }
        )
    return turns


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    return ""


def _turn_messages(
    raw_messages: Any,
    prior_history: List[Dict[str, Any]],
    *,
    prompt: str,
    final_response: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split Hermes' full-history result into one non-duplicated turn delta.

    Pinned Hermes returns the supplied conversation history again.  Some test
    adapters and compatible future releases may instead return only new
    messages, so prefix detection supports both without duplicating earlier
    turns.  Missing user/final messages are synthesized for a complete,
    benchmark-compatible history and transcript.
    """
    returned = [
        dict(message)
        for message in (raw_messages if isinstance(raw_messages, list) else [])
        if isinstance(message, dict)
    ]
    if (
        len(returned) >= len(prior_history)
        and returned[: len(prior_history)] == prior_history
    ):
        delta = returned[len(prior_history) :]
    else:
        delta = returned

    has_user_prompt = any(
        message.get("role") == "user" and _message_text(message).strip() == prompt.strip()
        for message in delta
    )
    if not has_user_prompt:
        delta.insert(0, {"role": "user", "content": prompt})

    has_final = any(
        message.get("role") == "assistant"
        and _message_text(message).strip() == final_response.strip()
        for message in delta
    )
    if final_response and not has_final:
        delta.append({"role": "assistant", "content": final_response})

    history = [*prior_history, *copy.deepcopy(delta)]
    return history, delta


def _usage_delta(current: Any, previous: float) -> Tuple[float, float]:
    """Convert a cumulative per-agent counter into a per-turn increment."""
    if current is None:
        return 0.0, previous
    try:
        value = max(0.0, float(current))
    except (TypeError, ValueError):
        return 0.0, previous
    increment = value - previous if value >= previous else value
    return increment, value


def _close_agent(agent: Any) -> None:
    if agent is None:
        return
    try:
        agent.close()
    except Exception:
        pass


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: hermes_driver.py REQUEST.json RESULT.json", file=sys.stderr)
        return 2
    request_path = Path(sys.argv[1]).resolve()
    result_path = Path(sys.argv[2]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # Session tasks place every future user turn in this request.  Remove the
    # file before registering terminal tools so turn 1 cannot inspect future
    # instructions from the shared scratch directory.
    try:
        request_path.unlink()
    except FileNotFoundError:
        pass
    key_env = str(request.get("api_key_env") or "OPENROUTER_API_KEY")
    # The parent transmits the credential only through this process's
    # environment.  Consume it before Hermes registers terminal tools so a
    # model-issued ``env`` command cannot recover the API key; AIAgent gets
    # the value directly below.
    api_key = os.environ.pop(key_env, None)
    search_key = os.environ.pop("FOH_CONFIGURED_SEARCH_KEY", None)
    if not api_key:
        _write_result(
            result_path,
            {
                "bridge_ok": False,
                "failed": True,
                "error": f"missing credential environment variable {key_env}",
            },
        )
        return 2

    # One-shot benchmark workers must never wait for an approval UI or launch
    # Hermes' independent background memory/skill reviewer.
    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["HERMES_ACCEPT_HOOKS"] = "1"

    agent = None
    mcp_started = False
    restore_error_guard = None
    try:
        from run_agent import AIAgent

        if request.get("provider") == "gemini":
            restore_error_guard = _install_gemini_error_guard(request.get("gemini_rate_limit"))

        configured_search = request.get("configured_search")
        if configured_search:
            if not isinstance(configured_search, dict) or not search_key:
                raise RuntimeError("configured search credential was not transferred")
            _install_configured_search(
                configured_search,
                search_key,
                str(request.get("model") or ""),
            )

        enabled_toolsets = [str(value) for value in request.get("toolsets") or []]
        mcp_server_names = [
            str(value) for value in request.get("mcp_server_names") or []
        ]
        if mcp_server_names:
            from tools.mcp_tool import (
                discover_mcp_tools,
                get_registered_mcp_server_names,
            )

            discovered = discover_mcp_tools()
            mcp_started = True
            connected = get_registered_mcp_server_names()
            missing = sorted(set(mcp_server_names) - set(connected))
            if missing:
                raise RuntimeError(
                    "Configured MCP servers did not expose tools: " + ", ".join(missing)
                )
            if not discovered:
                raise RuntimeError("Configured MCP servers exposed no tools")
            for name in mcp_server_names:
                alias = f"mcp-{name}"
                if alias not in enabled_toolsets:
                    enabled_toolsets.append(alias)

        turns = _session_turns(request)
        reasoning = request.get("reasoning_effort")
        request_overrides = {
            "temperature": float(request.get("temperature", 0.4)),
            "top_p": float(request.get("top_p", 1.0)),
        }
        top_k = int(request.get("top_k") or 0)
        if top_k > 0:
            # OpenAI's SDK has no first-class top_k argument; OpenRouter
            # accepts it as a provider-specific request body field.
            request_overrides["extra_body"] = {"top_k": top_k}
        overall_budget = max(1.0, float(request.get("run_budget_seconds") or 1800))
        deadline = time.monotonic() + overall_budget
        max_iterations = max(1, int(request.get("max_iterations") or 15))
        base_session_id = str(request.get("session_id") or "foh-hermes-turn")
        history: List[Dict[str, Any]] = []
        transcript: List[Dict[str, Any]] = []
        all_delta_messages: List[Dict[str, Any]] = []
        total_api_calls = 0
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        total_reasoning_tokens = 0.0
        total_cost = 0.0
        prior_input_tokens = 0.0
        prior_output_tokens = 0.0
        prior_reasoning_tokens = 0.0
        prior_cost = 0.0
        session_number = 0
        session_resets = 0
        attempted_turns = 0
        final_response = ""
        completed = False
        failed = False
        partial = False
        turn_exit_reason = ""
        result_model = str(request.get("model") or "")
        result_provider = str(request.get("provider") or "")

        for index, turn in enumerate(turns):
            if index > 0 and turn["new_session"]:
                _close_agent(agent)
                agent = None
                history = []
                prior_input_tokens = 0.0
                prior_output_tokens = 0.0
                prior_reasoning_tokens = 0.0
                prior_cost = 0.0
                session_resets += 1

            remaining_seconds = deadline - time.monotonic()
            remaining_iterations = max_iterations - total_api_calls
            if remaining_seconds <= 0:
                failed = True
                partial = attempted_turns > 0
                turn_exit_reason = "task_deadline_exceeded"
                break
            if remaining_iterations <= 0:
                failed = True
                partial = attempted_turns > 0
                turn_exit_reason = "task_iteration_budget_exhausted"
                break

            if agent is None:
                session_number += 1
                # Provider plugins perform their one-time availability check
                # while AIAgent builds the tool snapshot. Expose the key only
                # for that trusted constructor, then remove it before any user
                # prompt or terminal tool can run. The image provider receives
                # the same key later from Hermes' in-memory runtime context.
                os.environ[key_env] = api_key
                try:
                    agent = AIAgent(
                        api_key=api_key,
                        base_url=str(request.get("base_url") or ""),
                        provider=str(request.get("provider") or "openrouter"),
                        requested_provider=str(request.get("provider") or "openrouter"),
                        api_mode="chat_completions",
                        model=str(request["model"]),
                        enabled_toolsets=enabled_toolsets,
                        max_iterations=max(1, remaining_iterations),
                        max_tokens=max(1, int(request.get("max_tokens") or 8192)),
                        run_budget_seconds=max(1.0, remaining_seconds),
                        reasoning_config={"effort": str(reasoning)} if reasoning else None,
                        request_overrides=request_overrides,
                        ephemeral_system_prompt=str(request.get("system_prompt") or ""),
                        quiet_mode=True,
                        save_trajectories=False,
                        skip_context_files=True,
                        skip_memory=True,
                        skip_background_review=True,
                        platform="cli",
                        session_id=(
                            base_session_id
                            if session_number == 1
                            else f"{base_session_id}-session-{session_number}"
                        ),
                    )
                finally:
                    os.environ.pop(key_env, None)
            else:
                # Pinned Hermes reads these values on each conversation run.
                # Updating them prevents a same-session turn from receiving a
                # fresh full-task time/model-call allowance.
                agent.run_budget_seconds = max(1.0, remaining_seconds)
                agent.max_iterations = max(1, remaining_iterations)

            final_turn = index == len(turns) - 1
            turn_system_context = str(request.get("turn_system_context") or "").strip()
            if final_turn:
                turn_instruction = str(request.get("finish_instruction") or "").strip()
            else:
                turn_instruction = (
                    "Respond directly to this benchmark turn in its requested format. "
                    "Do not emit the FoH final envelope or a task-level reflection yet."
                )
            system_message = "\n\n".join(
                value
                for value in (
                    str(request.get("system_prompt") or "").strip(),
                    turn_system_context,
                    turn_instruction,
                )
                if value
            )
            prior_history = copy.deepcopy(history)
            raw = agent.run_conversation(
                turn["prompt"],
                system_message=system_message or None,
                conversation_history=history or None,
                task_id=f"{base_session_id}:{turn['id']}",
            )
            if not isinstance(raw, dict):
                raise TypeError("Hermes run_conversation result must be a mapping")
            attempted_turns += 1
            final_response = str(raw.get("final_response") or "")
            history, delta = _turn_messages(
                raw.get("messages"),
                prior_history,
                prompt=turn["prompt"],
                final_response=final_response,
            )
            all_delta_messages.extend(delta)
            transcript.extend(
                _canonical_transcript(
                    delta,
                    api_key,
                    final_response if final_turn else "",
                )
            )
            transcript = transcript[:_MAX_TRANSCRIPT_EVENTS]

            total_api_calls += max(0, int(raw.get("api_calls") or 0))
            increment, prior_input_tokens = _usage_delta(
                raw.get("input_tokens"), prior_input_tokens
            )
            total_input_tokens += increment
            increment, prior_output_tokens = _usage_delta(
                raw.get("output_tokens"), prior_output_tokens
            )
            total_output_tokens += increment
            increment, prior_reasoning_tokens = _usage_delta(
                raw.get("reasoning_tokens"), prior_reasoning_tokens
            )
            total_reasoning_tokens += increment
            increment, prior_cost = _usage_delta(
                raw.get("estimated_cost_usd"), prior_cost
            )
            total_cost += increment

            completed = bool(raw.get("completed"))
            failed = bool(raw.get("failed")) or not completed
            partial = bool(raw.get("partial"))
            turn_exit_reason = str(raw.get("turn_exit_reason") or "")
            result_model = str(raw.get("model") or result_model)
            result_provider = str(raw.get("provider") or result_provider)
            if failed:
                break

        completed = (
            attempted_turns == len(turns)
            and completed
            and not failed
            and bool(final_response.strip())
        )
        failed = failed or not completed
        summary = _tool_summary(all_delta_messages)
        result = {
            "bridge_ok": True,
            "final_response": final_response.replace(api_key, "[REDACTED]") if api_key else final_response,
            "transcript": transcript,
            "completed": completed,
            "failed": failed,
            "partial": partial or (0 < attempted_turns < len(turns)),
            "turn_exit_reason": turn_exit_reason,
            "api_calls": total_api_calls,
            "input_tokens": int(total_input_tokens),
            "output_tokens": int(total_output_tokens),
            "reasoning_tokens": int(total_reasoning_tokens),
            "estimated_cost_usd": total_cost,
            "model": result_model,
            "provider": result_provider,
            "session_turn_count": attempted_turns,
            "session_resets": session_resets,
            **summary,
        }
        _write_result(result_path, result)
        # Process success means the Hermes boundary itself worked.  Agent
        # non-completion is represented by the typed result fields and must
        # not be confused with an import/configuration/runtime crash.
        return 0
    except BaseException as exc:  # noqa: BLE001 - process boundary must report every failure
        error = f"{type(exc).__name__}: {exc}"
        if api_key:
            error = error.replace(api_key, "[REDACTED]")
        if search_key:
            error = error.replace(search_key, "[REDACTED]")
        failure = {"bridge_ok": False, "failed": True, "error": error[:4000]}
        if isinstance(exc, _FatalProviderResponse):
            failure["provider_error"] = {
                "status_code": exc.details["status_code"],
                "reason": _safe_text(exc.details["reason"], api_key, limit=120),
                "message": _safe_text(exc.details["message"], api_key, limit=2000),
            }
        _write_result(result_path, failure)
        return 1
    finally:
        if restore_error_guard is not None:
            restore_error_guard()
        os.environ.pop(key_env, None)
        os.environ.pop("FOH_CONFIGURED_SEARCH_KEY", None)
        _close_agent(agent)
        if mcp_started:
            try:
                from tools.mcp_tool import shutdown_mcp_servers

                shutdown_mcp_servers()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
