"""One protected model interface used by all agent roles."""

from __future__ import annotations

import asyncio
import inspect
from typing import Awaitable, Callable, Optional

from .config import ModelSettings
from .provider_errors import raise_for_provider_error
from .contracts import ModelResponse


Responder = Callable[[str, str], str | Awaitable[str]]


class ModelClient:
    def __init__(self, settings: ModelSettings, responder: Optional[Responder] = None):
        self.settings = settings
        self._responder = responder

    @property
    def has_responder(self) -> bool:
        """Whether this client is an injected deterministic/test transport."""
        return self._responder is not None

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        if self._responder is not None:
            value = self._responder(prompt, system)
            text = await value if inspect.isawaitable(value) else value
            return ModelResponse(
                text=str(text), provider=self.settings.provider, model=self.settings.name
            )
        if self.settings.provider == "mock":
            raise RuntimeError(
                "model.provider='mock' has no implicit completion backend. "
                "Tests must construct ModelClient(settings, responder=...) explicitly; "
                "operational runs must configure a remote provider."
            )

        # Keep provider details behind this boundary. Existing multi-provider
        # transport remains reusable while the old client/server APIs are removed.
        import utils

        # Retry in the cancellable coroutine, not inside the worker thread.
        # Cancelling asyncio.to_thread cannot stop its current HTTP request,
        # but it must prevent that abandoned thread from launching two more
        # full-timeout requests after the task has already ended.
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.retries + 1):
            try:
                text, info = await asyncio.to_thread(
                    utils.chat,
                    [{"role": "user", "content": prompt}],
                    system=system,
                    provider=self.settings.provider,
                    model=self.settings.name,
                    api_key=self.settings.api_key(),
                    base_url=self.settings.base_url,
                    max_tokens=max(1, int(max_tokens or self.settings.max_tokens)),
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    top_k=self.settings.top_k,
                    timeout=self.settings.timeout_seconds,
                    retries=0,
                    json_mode=json_mode,
                    reasoning_effort=self.settings.reasoning_effort,
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve provider errors after retries
                raise_for_provider_error(
                    provider=self.settings.provider, model=self.settings.name,
                    status_code=getattr(exc, "status_code", None)
                    or getattr(getattr(exc, "response", None), "status_code", None)
                    or getattr(exc, "code", None),
                    message=str(exc), secret=self.settings.api_key() or "",
                    terminal=attempt >= self.settings.retries,
                )
                last_error = exc
                if attempt >= self.settings.retries:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
        else:  # pragma: no cover - the loop either succeeds or raises
            assert last_error is not None
            raise last_error
        return ModelResponse(
            text=text,
            provider=str(info.get("provider", self.settings.provider)),
            model=str(info.get("model", self.settings.name)),
            input_tokens=int(info.get("input_tokens", 0) or 0),
            output_tokens=int(info.get("output_tokens", 0) or 0),
            finish_reason=str(info.get("finish_reason", "stop")),
        )
