"""
utils.py — Federation over Harness (FoH)

A small, self-contained helper layer shared by agent.py / client.py /
the protected model boundary and benchmark graders.

The one thing everything depends on is a *unified chat API* so that the
multi-round agent (agent.py) can talk to any provider with the same call:

    text, info = chat(messages, provider="gemini", model=None, ...)

The agent (agent.py) can be powered by EITHER a local open-source model loaded
from HuggingFace OR a hosted API — both go through the same chat() call, you
just switch `provider`.

Supported providers (all configurable, default = gemini):
    - "huggingface"        local open-source model via transformers (no API key)
    - "gemini"             google-genai
    - "openai"             OpenAI chat-completions
    - "anthropic"          Anthropic messages API
    - "openrouter"         OpenAI-compatible (openrouter.ai)
    - "grok"               OpenAI-compatible (api.x.ai)
    - "openai-compatible"  any OpenAI-compatible endpoint (base_url + key)
    - "mock"               deterministic offline backend for smoke tests

Design note (Occam's razor): OpenAI / OpenRouter / Grok / openai-compatible all
share one code path (OpenAI chat-completions schema). Only huggingface, gemini
and anthropic need their own path. `requests` is used for HTTP so there is no
hard SDK dependency except google-genai (provider="gemini") and
transformers/torch (provider="huggingface"), each imported lazily on use.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

# Default model per provider. Override with model=... or --model on the CLI.
DEFAULT_MODELS: Dict[str, str] = {
    "huggingface": "Qwen/Qwen2.5-7B-Instruct",
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "openrouter": "anthropic/claude-sonnet-4.5",
    "grok": "grok-4",
    "openai-compatible": "default",
    "mock": "mock-1",
}

# Default base URLs for the OpenAI-compatible family.
_OPENAI_COMPATIBLE_BASE: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "grok": "https://api.x.ai/v1",
}

# Environment variables checked (in order) for each provider's API key.
_ENV_KEYS: Dict[str, List[str]] = {
    "huggingface": [],  # local model — no API key
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
    "openai-compatible": ["OPENAI_API_KEY", "LLM_API_KEY"],
    "mock": [],
}

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openrouter", "grok", "openai-compatible"}


def resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Return an explicit api_key, else the first matching env var."""
    if api_key:
        return api_key
    for env_name in _ENV_KEYS.get(provider, []):
        val = os.getenv(env_name)
        if val:
            return val
    return None


def default_model(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, "default")


# ---------------------------------------------------------------------------
# Unified chat() entry point
# ---------------------------------------------------------------------------

def chat(
    messages: List[Dict[str, str]],
    *,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    timeout: float = 120.0,
    retries: int = 2,
    json_mode: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Unified multi-turn chat call.

    Args:
        messages: list of {"role": "user"|"assistant"|"system", "content": str}.
                  A leading system message is honoured; `system` (below) wins
                  if both are given.
        provider: one of the keys in DEFAULT_MODELS.
        model:    model id (defaults to DEFAULT_MODELS[provider]).
        api_key:  explicit key (else resolved from env).
        base_url: override endpoint (OpenAI-compatible / openai-compatible).
        system:   system prompt (separated out for providers that want it
                  distinct, e.g. anthropic / gemini system_instruction).
        max_tokens, temperature, timeout, retries: generation controls.
        json_mode: request a JSON object from compatible providers.
        reasoning_effort: forwarded to OpenAI-compatible providers only (see
                  _chat_openai_compatible) as {"reasoning": {"effort": ...}} --
                  a model-specific knob for reasoning-capable models; None
                  leaves the provider's own default alone.

    Returns:
        (text, info) where info has at least {"provider", "model",
        "output_tokens", "finish_reason"}.
    """
    provider = (provider or "gemini").lower()
    model = model or default_model(provider)
    key = resolve_api_key(provider, api_key)

    # Pull a leading system message out of `messages` if `system` not supplied.
    sys_text = system
    convo: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system" and sys_text is None:
            sys_text = content
            continue
        convo.append({"role": role, "content": content})

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if provider == "mock":
                return _chat_mock(convo, sys_text, model)
            if provider in ("huggingface", "hf"):
                return _chat_hf(convo, sys_text, model, max_tokens, temperature, top_p, top_k, base_url)
            if provider == "gemini":
                return _chat_gemini(convo, sys_text, model, key, max_tokens, temperature, top_p, top_k)
            if provider == "anthropic":
                return _chat_anthropic(
                    convo, sys_text, model, key, base_url, max_tokens, temperature, timeout, top_p, top_k
                )
            if provider in OPENAI_COMPATIBLE_PROVIDERS:
                return _chat_openai_compatible(
                    provider, convo, sys_text, model, key, base_url,
                    max_tokens, temperature, timeout, top_p, top_k, json_mode,
                    reasoning_effort=reasoning_effort,
                )
            raise ValueError(f"Unknown provider: {provider}")
        except Exception as exc:  # noqa: BLE001 - surface after retries
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise
    raise last_err  # pragma: no cover


def single_call(
    prompt: str,
    *,
    system: Optional[str] = None,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Convenience wrapper for a one-shot prompt (no conversation history)."""
    return chat(
        [{"role": "user", "content": prompt}],
        provider=provider, model=model, api_key=api_key, base_url=base_url,
        system=system, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

# Local HuggingFace models are cached so the multi-round agent loop reuses one
# loaded model instead of reloading per call. Keyed by (model, device).
_HF_CACHE: Dict[str, Any] = {}


def _chat_hf(convo, sys_text, model, max_tokens, temperature, top_p=1.0, top_k=0, device=None):
    """Run a local open-source model from HuggingFace via transformers.

    Multi-turn: messages (with an optional system turn) are rendered with the
    tokenizer's chat template, so the same agent loop works for API and local
    models. `device` may be passed via base_url ('cuda'/'cpu') or auto-detected.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers + torch are required for provider='huggingface' "
            "(pip install torch transformers)"
        ) from exc

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model}@{dev}"
    print(
        "[hf] request "
        f"pid={os.getpid()} model={model} device={dev} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    if torch.cuda.is_available():
        try:
            gpu_names = []
            for idx in range(torch.cuda.device_count()):
                gpu_names.append(f"{idx}:{torch.cuda.get_device_name(idx)}")
            print(f"[hf] cuda_devices={'; '.join(gpu_names)}", flush=True)
        except Exception as exc:
            print(f"[hf] cuda_device_query_error={exc}", flush=True)
    if cache_key not in _HF_CACHE:
        print(f"[hf] loading tokenizer/model for {cache_key}", flush=True)
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if dev == "cuda":
            kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
        else:
            kwargs["torch_dtype"] = torch.float32
        mdl = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _HF_CACHE[cache_key] = (mdl, tok)
        try:
            first_device = next(mdl.parameters()).device
        except Exception:
            first_device = "unknown"
        print(f"[hf] model_loaded model={model} first_parameter_device={first_device}", flush=True)
    mdl, tok = _HF_CACHE[cache_key]

    def _model_input_device():
        dev_attr = getattr(mdl, "device", None)
        if dev_attr is not None and str(dev_attr) != "meta":
            return dev_attr
        try:
            return next(mdl.parameters()).device
        except Exception:
            return dev

    def _encoded_to_inputs(encoded):
        target = _model_input_device()
        if isinstance(encoded, _torch.Tensor):
            return {"input_ids": encoded.to(target)}
        if hasattr(encoded, "to"):
            encoded = encoded.to(target)
        if isinstance(encoded, Mapping):
            return {
                key: value.to(target) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        input_ids_value = getattr(encoded, "input_ids", None)
        if input_ids_value is not None:
            return {"input_ids": input_ids_value.to(target)}
        raise TypeError(f"Unsupported HuggingFace tokenizer output: {type(encoded)!r}")

    msgs: List[Dict[str, str]] = []
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    msgs.extend(convo)
    import torch as _torch
    try:
        encoded = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
        model_inputs = _encoded_to_inputs(encoded)
    except Exception:
        # Fallback for tokenizers without a chat template.
        flat = (sys_text + "\n\n" if sys_text else "") + "\n\n".join(
            f"{m['role']}: {m['content']}" for m in convo
        ) + "\n\nassistant:"
        model_inputs = _encoded_to_inputs(tok(flat, return_tensors="pt"))

    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        raise TypeError("HuggingFace tokenizer output did not include input_ids")
    print(
        f"[hf] generate_start model={model} input_tokens={int(input_ids.shape[1])} "
        f"max_new_tokens={int(max_tokens)}",
        flush=True,
    )
    with _torch.no_grad():
        gen_kwargs = dict(model_inputs)
        gen_kwargs.update({
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-4),
            "pad_token_id": tok.eos_token_id,
        })
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k and int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)
        out = mdl.generate(**gen_kwargs)
    generated_len = int(out.shape[1] - input_ids.shape[1])
    gen = out[0].narrow(0, int(input_ids.shape[1]), generated_len)
    text = tok.decode(gen, skip_special_tokens=True).strip()
    info = {
        "provider": "huggingface",
        "model": model,
        "output_tokens": int(gen.shape[0]),
        "finish_reason": "stop",
    }
    print(f"[hf] generate_done model={model} output_tokens={int(gen.shape[0])}", flush=True)
    return text, info


def _chat_gemini(convo, sys_text, model, key, max_tokens, temperature, top_p: Optional[float] = None, top_k: Optional[int] = None):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-genai is required for provider='gemini' "
            "(pip install google-genai)"
        ) from exc
    if not key:
        raise RuntimeError("No Gemini API key (set GEMINI_API_KEY).")
    model = _normalize_gemini_model(model)
    client = genai.Client(api_key=key)
    # Gemini roles: user / model.
    contents = []
    for m in convo:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    gen_cfg = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    if sys_text:
        gen_cfg["systemInstruction"] = sys_text
    if top_p is not None:
        gen_cfg["topP"] = top_p
    if top_k is not None:
        gen_cfg["topK"] = top_k
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**gen_cfg),
    )
    text = _gemini_text(resp)
    usage = getattr(resp, "usage_metadata", None)
    info = {
        "provider": "gemini",
        "model": model,
        "output_tokens": _safe_int(
            getattr(usage, "candidates_token_count", 0)
            or getattr(usage, "output_token_count", 0)
        ),
        "finish_reason": _gemini_finish_reason(resp) or "stop",
    }
    return text, info


def _normalize_gemini_model(model: str) -> str:
    """Accept LiteLLM/OpenClaw-style Gemini IDs on the native Gemini backend."""
    model = str(model or default_model("gemini"))
    if model.startswith("google/") or model.startswith("gemini/"):
        return model.split("/", 1)[1]
    return model


def _gemini_text(resp) -> str:
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    parts_text = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if t:
                parts_text.append(t)
    return "\n".join(parts_text)


def _gemini_finish_reason(resp) -> str:
    for cand in getattr(resp, "candidates", []) or []:
        reason = getattr(cand, "finish_reason", None)
        if reason:
            return str(reason)
    return ""


def _chat_openai_compatible(provider, convo, sys_text, model, key, base_url,
                            max_tokens, temperature, timeout, top_p: Optional[float] = None,
                            top_k: Optional[int] = None, json_mode: bool = False,
                            reasoning_effort: Optional[str] = None):
    import requests
    base = base_url or _OPENAI_COMPATIBLE_BASE.get(provider) or _OPENAI_COMPATIBLE_BASE["openai"]
    url = base.rstrip("/") + "/chat/completions"
    msgs: List[Dict[str, str]] = []
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    msgs.extend(convo)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/federation-over-harness"
        headers["X-Title"] = "Federation over Harness"
    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        # OpenRouter's own reasoning-control shape (confirmed against their
        # API parameter docs): a nested object, not a flat field. Only sent
        # when explicitly configured -- omitting it leaves whatever default
        # the model/provider already uses untouched, since not every
        # OpenAI-compatible provider or model supports this at all, and a
        # value the target model doesn't recognize should fail loudly at the
        # API rather than being silently sent everywhere.
        payload["reasoning"] = {"effort": reasoning_effort}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if json_mode and r.status_code == 400:
        # Some OpenAI-compatible routes do not implement response_format.
        # Retry once without that optional field instead of turning every
        # controller call into a hard task failure.
        payload.pop("response_format", None)
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "") or ""
    usage = data.get("usage") or {}
    info = {
        "provider": provider,
        "model": model,
        "output_tokens": _safe_int(usage.get("completion_tokens", 0)),
        "finish_reason": choice.get("finish_reason", "stop"),
    }
    return text, info


def _chat_anthropic(convo, sys_text, model, key, base_url, max_tokens, temperature, timeout, top_p: Optional[float] = None, top_k: Optional[int] = None):
    import requests
    base = base_url or "https://api.anthropic.com/v1"
    url = base.rstrip("/") + "/messages"
    if not key:
        raise RuntimeError("No Anthropic API key (set ANTHROPIC_API_KEY).")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    # Anthropic wants alternating user/assistant; system is a top-level field.
    msgs = [{"role": ("assistant" if m["role"] == "assistant" else "user"),
             "content": m["content"]} for m in convo]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if sys_text:
        payload["system"] = sys_text
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = ""
    for block in data.get("content", []) or []:
        if block.get("type") == "text":
            text += block.get("text", "")
    usage = data.get("usage") or {}
    info = {
        "provider": "anthropic",
        "model": model,
        "output_tokens": _safe_int(usage.get("output_tokens", 0)),
        "finish_reason": data.get("stop_reason", "stop"),
    }
    return text, info


def _chat_mock(convo, sys_text, model):
    """Deterministic offline backend.

    It inspects the last user message for lightweight cues so the agent loop
    and the algorithm stubs can be exercised end-to-end without any network:
      - If asked for JSON, returns a minimal valid JSON object.
      - If it looks like an agent action turn, emits a `finish` action.
      - Otherwise echoes a short acknowledgement.
    """
    last_user = ""
    for m in reversed(convo):
        if m["role"] == "user":
            last_user = m["content"]
            break
    lower = last_user.lower()
    # Retain deterministic responses for older benchmark-grader prompt markers.

    if "strict task grader" in (sys_text or "").lower() or "score 0.0 to 1.0" in lower:
        text = json.dumps({
            "score": 0.0,
            "max_score": 1.0,
            "reasoning": "Mock grader cannot evaluate real task completion.",
        })
    elif "json action" in lower or '"tool"' in lower:
        # Agent protocol turn -> finish immediately with a stub answer.
        text = json.dumps({
            "thought": "Mock agent: completing the task in one step.",
            "tool": "finish",
            "answer": "[MOCK] Task completed by the mock agent backend.",
        })
    elif "mock_task=residual" in lower:
        text = json.dumps({
            "residual_summary": "[MOCK] Actual closely matched the expectation.",
            "score_gap": 0.0,
            "discrepancies": [],
            "reflection": "Mock reflection: harness behaved roughly as predicted.",
            "harness_revision_hints": ["no change needed (mock)"],
        })
    elif "mock_task=expectation" in lower:
        text = json.dumps({
            "hypothesis": "[MOCK] The harness should solve the task in a few rounds.",
            "expected_results": "A correct artifact written to the workspace.",
            "expected_steps": ["read problem", "produce artifact", "finish"],
            "predicted_score": 0.7,
            "risks": ["mock backend has no real reasoning"],
        })
    elif "mock_task=harness_design" in lower:
        text = json.dumps({
            "system_prompt": (
                "[MOCK] You are a task-adapted FoH agent. Use the guide and "
                "insight library to solve the workspace task, create requested "
                "artifacts, and finish concisely."
            ),
            "task_prompt": (
                "[MOCK] Solve the original task completely. Use the workspace "
                "files and produce any requested artifacts."
            ),
            "tool_policy": (
                "[MOCK] Use filesystem and bash tools when helpful; do not use "
                "web/search unless the task explicitly requires current external information."
            ),
            "provider": "mock",
            "model": None,
            "backend": "fallback",
            "max_rounds": 3,
            "temperature": 0.2,
            "max_tokens": 2048,
            "allow_bash": True,
            "allow_web": False,
            "allow_search": False,
            "mcp_servers": [],
            "api_access": ["mock"],
            "memory_file": None,
            "agent_id": "foh-mock-designed",
            "selected_skills": [],
            "selected_insights": [],
            "orchestration": {"mode": "single", "use_search": False},
            "rationale": "[MOCK] Minimal single-agent harness for smoke tests.",
        })
    elif "mock_task=guide_library" in lower:
        text = json.dumps({
            "guide": {
                "mock_generated_comprehensive_guidance": (
                    "[MOCK] This placeholder represents a comprehensive guide generated "
                    "from residual updates. In a real run, the server agent should decide "
                    "its own organization and content from the residual digest and prior "
                    "state. The guide should be instructive enough for later harness "
                    "creation, explaining how to interpret recurring prediction gaps, how "
                    "to turn them into future harness changes, how to calibrate expected "
                    "outcomes, and how to avoid overfitting to any one client or task. "
                    "This mock text intentionally avoids prescribing concrete section "
                    "names, workflows, tools, artifacts, or domain-specific rules."
                ),
                "mock_generated_operational_detail": (
                    "[MOCK] A real guide should contain actionable decision criteria "
                    "derived from the observed residuals, preserve useful prior guidance, "
                    "revise weak prior guidance, and state applicability boundaries. It "
                    "should be long enough to be useful. The server agent is responsible "
                    "for inventing the structure that best represents what it learned."
                ),
            },
            "insight_library": {
                "mock_generated_insight_1": (
                    "[MOCK] Placeholder for an insight generated from residuals. A real "
                    "entry should describe the learned pattern, why it matters for harness "
                    "evolution, what kind of future adjustment it suggests, and when the "
                    "lesson should not be applied."
                ),
                "mock_generated_insight_2": (
                    "[MOCK] Placeholder for a second residual-derived insight. The server "
                    "agent should create as many entries as the evidence supports, using "
                    "its own names and organization."
                ),
                "mock_generated_insight_3": (
                    "[MOCK] Placeholder for accumulated learning across rounds. A real "
                    "entry should merge useful prior knowledge with new residual evidence "
                    "and should remain reusable for unseen tasks."
                ),
            },
        })
    else:
        text = f"[MOCK:{model}] ack: {last_user}"
    return text, {"provider": "mock", "model": model, "output_tokens": len(text.split()), "finish_reason": "stop"}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def slugify(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^\w\s-]", "", str(text))
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s or "item"


def extract_first_json(text: str) -> Optional[Any]:
    """Extract the first balanced JSON object/array from `text`.

    Handles ```json fenced blocks and trailing commas. Returns the parsed
    object, or None if nothing parseable is found. (Shared by the agent's
    action parser and by the algorithm stubs that ask the LLM for JSON.)
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = None
        for i, ch in enumerate(text):
            if ch in "{[":
                start = i
                break
        if start is None:
            return None
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        end = None
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        stop = end if end else len(text)
        candidate = "".join(text[i] for i in range(start, stop))
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def read_text(path: str, default: str = "") -> str:
    """Read a text file, tolerant of real-world non-UTF-8 content.

    Many benchmark fixtures (e.g. syslogs, scraped data) contain stray
    non-UTF-8 bytes. A strict utf-8 read raises UnicodeDecodeError (a
    ValueError, NOT caught by the old (FileNotFoundError, OSError) handler),
    which previously crashed the `read_file` tool and blocked the agent on
    files it had every right to read. Reading the bytes is never the ability
    under test, so we degrade gracefully: try utf-8, then utf-8 with
    replacement, so a valid file is ALWAYS readable.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except (FileNotFoundError, OSError):
        return default
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Salvage: replace undecodable bytes rather than failing the read.
        return raw.decode("utf-8", errors="replace")


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# FoT prompt-optimization compatibility API


def _module_available(name: str) -> bool:
    """Return false when an optional package or one of its parents is absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError):
        return False


HAS_GEMINI = _module_available("google.genai")

SUPPORTED_API_PROVIDERS = ("gemini", "openrouter")
DEFAULT_API_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "openrouter": "google/gemini-2.5-flash-lite",
}
API_KEY_ENV_VARS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def check_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def setup_gemini(api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite") -> Dict[str, Optional[str]]:
    return setup_api("gemini", api_key=api_key, model_name=model_name)


def normalize_api_provider(provider: Optional[str]) -> str:
    normalized = str(provider or "gemini").strip().lower()
    if normalized not in SUPPORTED_API_PROVIDERS:
        choices = ", ".join(SUPPORTED_API_PROVIDERS)
        raise ValueError(f"Unsupported API provider {provider!r}; choose one of: {choices}")
    return normalized


def normalize_api_model(provider: Optional[str], model_name: Optional[str]) -> str:
    normalized_provider = normalize_api_provider(provider)
    model = str(model_name or DEFAULT_API_MODELS[normalized_provider]).strip()
    if normalized_provider == "openrouter":
        model = model.removeprefix("openrouter/")
        # Accept the same bare Gemini model IDs used by the native backend.
        if model.startswith("gemini-"):
            model = f"google/{model}"
    return model



def setup_api(
    provider: Optional[str] = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    normalized_provider = normalize_api_provider(provider)
    return {
        "provider": normalized_provider,
        "api_key": resolve_api_key(normalized_provider, api_key),
        "model_name": normalize_api_model(normalized_provider, model_name),
    }


def _info_with_output_tokens(text: str, info: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(info or {})
    merged.setdefault("output_tokens", len((text or "").split()))
    return merged


def call_gemini(
    gemini_model: Any,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    model_name = "gemini-2.5-flash-lite"
    api_key = None
    if isinstance(gemini_model, dict):
        model_name = gemini_model.get("model_name") or model_name
        api_key = gemini_model.get("api_key")
    text, info = single_call(
        prompt,
        system=system_prompt,
        provider="gemini",
        model=model_name,
        api_key=api_key,
        max_tokens=max_new_tokens or 32768,
    )
    return text, _info_with_output_tokens(text, info)

def call_api(
    api_config: Any,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Call either native Gemini or OpenRouter using one FoT configuration."""
    config = api_config if isinstance(api_config, dict) else {}
    provider = normalize_api_provider(config.get("provider"))
    model_name = normalize_api_model(provider, config.get("model_name"))
    api_key = resolve_api_key(provider, config.get("api_key"))
    text, info = single_call(
        prompt,
        system=system_prompt,
        provider=provider,
        model=model_name,
        api_key=api_key,
        max_tokens=max_new_tokens or 32768,
    )
    return text, _info_with_output_tokens(text, info)


def call_gemini_thinking(
    api_key: Optional[str],
    model_name: str,
    prompt: str,
    thinking_level: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    text, info = single_call(
        prompt,
        provider="gemini",
        model=model_name or "gemini-2.5-flash-lite",
        api_key=api_key,
        max_tokens=max_new_tokens or 32768,
    )
    info = _info_with_output_tokens(text, info)
    if thinking_level:
        info["thinking_level"] = thinking_level
    return text, info


def call_openrouter(
    api_key: Optional[str],
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    return call_api(
        setup_api("openrouter", api_key=api_key, model_name=model),
        prompt,
        system_prompt,
        max_new_tokens,
    )


def load_hf_model(
    model_name: str,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
) -> Tuple[None, None]:
    # The shared root implementation caches HuggingFace models inside chat().
    return None, None


def call_hf_model(
    model: Any,
    tokenizer: Any,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    text, info = single_call(
        prompt,
        system=system_prompt,
        provider="huggingface",
        model=model_name,
        base_url=device,
        max_tokens=max_new_tokens or 32768,
    )
    return text, _info_with_output_tokens(text, info)
