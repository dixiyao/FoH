"""Gemini replacement for the pinned benchmark's gpt4_visual_judge module.

Mounted only for --judge-model runs. Uses stdlib HTTP so benchmark images do
not need an extra SDK. The upstream prompt and score_figure interface are kept.
"""
from __future__ import annotations

import ast
import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# The scoring launcher imports this module before executing the submitted
# program. Consume the credential so that program's subprocess cannot inherit it.
_API_KEY = os.environ.pop("FOH_JUDGE_API_KEY", "")
_MODEL = os.environ.pop("FOH_JUDGE_MODEL", "")
ORIGINAL_JUDGE = Path("/testbed/foh_original_visual_judge.py")


def _prompt() -> str:
    # Reading the constant avoids importing the upstream OpenAI client.
    tree = ast.parse(ORIGINAL_JUDGE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROMPT_ORIGIN"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError("Pinned visual judge has no PROMPT_ORIGIN string")


def encode_image(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _generate(parts: list) -> str:
    if not _API_KEY or not _MODEL:
        raise RuntimeError("Gemini visual judge requires a model and Google API credential")
    model = urllib.parse.quote(_MODEL, safe="")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": _API_KEY},
        data=json.dumps({
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.2, "topP": 0.95,
                # Includes thinking tokens on reasoning-capable Gemini models.
                "maxOutputTokens": 8192,
            },
        }).encode("utf-8"),
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in {429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(f"Gemini visual judge HTTP error {status}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError("Gemini visual judge request failed after 3 attempts") from None
        time.sleep(2 ** attempt)
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini visual judge returned no candidate")
    text = "".join(
        str(part.get("text", ""))
        for part in candidates[0].get("content", {}).get("parts", [])
        if not part.get("thought")
    )
    if not text.strip():
        raise RuntimeError("Gemini visual judge returned no visible text")
    return text


def score_figure(pred_fig, gold_fig):
    parts = [
        {"text": _prompt()},
        {"inlineData": {"mimeType": "image/png", "data": pred_fig}},
        {"inlineData": {"mimeType": "image/png", "data": gold_fig}},
    ]
    # Three independent requests work with models that allow only one candidate.
    full_responses = [_generate(parts) for _ in range(3)]
    # Match the pinned parser, including its zero for an unparseable rating.
    matches = [re.search(r"\[FINAL SCORE\]: (\d{1,3})", value, re.DOTALL)
               for value in full_responses]
    scores = [int(match.group(1)) if match else 0 for match in matches]
    return full_responses, sum(scores) / len(scores)
