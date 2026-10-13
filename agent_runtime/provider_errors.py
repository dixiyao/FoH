"""Provider failures that must abort an experiment instead of becoming scores."""

from __future__ import annotations

import re
import threading
from contextvars import ContextVar
from typing import Any

from .privacy import sanitize_text


FATAL_HTTP_STATUSES = frozenset({401, 402, 403, 404})


class FatalProviderError(RuntimeError):
    """A permanent or retry-exhausted provider error prevents valid task results."""


class ProviderBatchAbort:
    """Share the first fatal error with concurrent task subprocesses."""

    def __init__(self):
        self._lock = threading.Lock()
        self._message = ""

    def abort(self, error: FatalProviderError) -> None:
        with self._lock:
            if not self._message:
                self._message = str(error)

    def check(self) -> None:
        with self._lock:
            message = self._message
        if message:
            raise FatalProviderError(message)


provider_batch_abort: ContextVar[ProviderBatchAbort | None] = ContextVar(
    "provider_batch_abort", default=None,
)


def raise_for_provider_error(
    *, provider: str, model: str, status_code: Any, message: str,
    secret: str = "", reason: str = "", terminal: bool = False,
) -> None:
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        return
    if status not in FATAL_HTTP_STATUSES and not (terminal and 400 <= status < 600):
        return
    detail = sanitize_text(message, forbidden_values=(secret,) if secret else (), max_chars=2000)
    reason = sanitize_text(reason, max_chars=120)
    raise FatalProviderError(
        f"{provider} model {model}: HTTP {status}"
        + (f" ({reason})" if reason else "")
        + f": {detail}. Experiment aborted; fix provider credentials, permissions, "
        "billing, model access, or service availability before resuming. This is not a task score."
    )


def raise_for_gemini_error_text(text: str, *, model: str, secret: str = "") -> None:
    """Recognize terminal adapter errors after its transport retry policy ends."""
    match = re.search(r"Gemini (?:HTTP|visual judge HTTP error) ([45]\d{2})\b[^\n]*", text)
    if match:
        raise_for_provider_error(
            provider="gemini", model=model, status_code=match[1],
            message=match[0], secret=secret, terminal=True,
        )
