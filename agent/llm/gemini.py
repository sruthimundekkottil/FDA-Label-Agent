"""Gemini adapter: google-genai SDK, ``client.models.generate_content``.

Written against the Gemini API docs as read on 2026-10-08
(ai.google.dev/gemini-api/docs/generate-content/function-calling):

* tools are ``types.Tool(function_declarations=[...])``; automatic function
  calling is disabled because the agent loop runs the tools itself;
* tool results go back as ``function_response`` parts in a ``user`` turn, with
  the exact ``id`` of the ``function_call`` they answer;
* model turns are replayed verbatim (``provider_content``) so Gemini 3 thought
  signatures survive.

The API key comes from ``GEMINI_API_KEY`` (or ``LLM_API_KEY``) and is passed to
the client explicitly. It is never logged and is scrubbed from error messages.
"""

from __future__ import annotations

import json
import random
import re
import time

from agent.llm.base import (
    GenerationSettings,
    LLMError,
    LLMResult,
    QuotaExhaustedError,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from src.utils import get_logger

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_QUOTA_ID_RE = re.compile(r'"quotaId"\s*:\s*"([^"]+)"')
# A 429 asking us to wait longer than this is not a short rate limit.
LONG_QUOTA_WAIT_S = 300.0

logger = get_logger("agent.llm.gemini")


class GeminiAdapter:
    provider = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str = "",
        client=None,
        min_interval_s: float = 1.0,
        max_retries: int = 5,
        max_backoff_s: float = 60.0,
    ):
        if not model:
            raise LLMError("no Gemini model configured (set LLM_MODEL)")
        self.model = model
        self._api_key = api_key
        self._client = client
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.max_backoff_s = max_backoff_s
        self._last_request = 0.0

    def __repr__(self) -> str:  # never show the key
        return f"GeminiAdapter(model={self.model!r})"

    # -- client --------------------------------------------------------------
    @property
    def client(self):
        """Created lazily, so replay-only runs need no key at all."""

        if self._client is None:
            if not self._api_key:
                raise LLMError("no Gemini API key (set GEMINI_API_KEY or LLM_API_KEY in .env)")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # -- public ----------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        settings: GenerationSettings | None = None,
    ) -> LLMResult:
        settings = settings or GenerationSettings()
        contents = to_contents(messages)
        config = build_config(system, tools, settings)

        started = time.perf_counter()
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            self._respect_min_interval()
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as exc:  # SDK errors + transport errors
                code = getattr(exc, "code", None)
                if code == 429:
                    exhausted = long_quota_exhausted(exc)
                    if exhausted:
                        raise QuotaExhaustedError(
                            self._scrub(
                                f"Gemini quota used up for model {self.model} ({exhausted}). "
                                "Retrying will not help: wait for the quota to reset, use another "
                                "key or model, or replay recorded runs with LLM_CACHE_MODE=replay."
                            )
                        ) from None
                last_error = self._scrub(f"{type(exc).__name__}: {exc}")
                retryable = code in RETRYABLE_CODES or code is None and _is_transport_error(exc)
                if not retryable or attempt == self.max_retries:
                    raise LLMError(last_error) from None
                delay = self._backoff_delay(attempt, exc)
                logger.warning(
                    "gemini %s, attempt %d/%d, sleeping %.1fs",
                    code or type(exc).__name__,
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            return parse_response(response, self.model, (time.perf_counter() - started) * 1000)
        raise LLMError(f"giving up after {self.max_retries} attempts: {last_error}")

    # -- internals ------------------------------------------------------------
    def _respect_min_interval(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _backoff_delay(self, attempt: int, exc: Exception) -> float:
        delay = min(self.max_backoff_s, 2.0**attempt) + random.uniform(0, 1)
        details = _error_details(exc)
        if details:
            match = _RETRY_DELAY_RE.search(details)
            if match:
                delay = max(delay, min(120.0, float(match.group(1))))
        return delay

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text


def _error_details(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    return json.dumps(details, default=str) if details else ""


def long_quota_exhausted(exc: Exception) -> str | None:
    """Describe a 429 that is a long-term quota (daily), or ``None`` for a short rate limit.

    Gemini uses the same "You exceeded your current quota" message for per-minute
    and per-day limits, so the message cannot tell them apart. The error details
    can: the violated ``quotaId`` (e.g. ``GenerateRequestsPerDayPerProjectPerModel-FreeTier``)
    and the server's ``retryDelay`` (hours for a daily quota, seconds per minute).
    """

    details = _error_details(exc)
    daily = [quota for quota in _QUOTA_ID_RE.findall(details) if "PerDay" in quota]
    if daily:
        return f"daily quota {daily[0]}"
    match = _RETRY_DELAY_RE.search(details)
    if match and float(match.group(1)) > LONG_QUOTA_WAIT_S:
        hours = float(match.group(1)) / 3600
        return f"server asks to wait {hours:.1f} h"
    return None


def _is_transport_error(exc: Exception) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


# --------------------------------------------------------------------------- #
# Translation: neutral types <-> google.genai.types
# --------------------------------------------------------------------------- #
def to_contents(messages: list[Message]) -> list:
    from google.genai import types

    contents: list = []
    for message in messages:
        if message.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message.text)]))
        elif message.role == "assistant":
            if message.provider == GeminiAdapter.provider and message.provider_content:
                contents.append(types.Content.model_validate(message.provider_content))
                continue
            parts = [types.Part(text=message.text)] if message.text else []
            parts += [
                types.Part(
                    function_call=types.FunctionCall(id=call.id or None, name=call.name, args=call.args)
                )
                for call in message.tool_calls
            ]
            contents.append(types.Content(role="model", parts=parts))
        else:  # tool results
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=result.call_id or None,
                                name=result.name,
                                response=result.content,
                            )
                        )
                        for result in message.tool_results
                    ],
                )
            )
    return contents


def build_config(system: str, tools: list[ToolSpec] | None, settings: GenerationSettings):
    from google.genai import types

    kwargs: dict = {
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "thinking_config": types.ThinkingConfig(thinking_level=settings.thinking_level.upper()),
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if system:
        kwargs["system_instruction"] = system
    if settings.seed is not None:
        kwargs["seed"] = settings.seed
    if tools:
        kwargs["tools"] = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.parameters,
                    )
                    for tool in tools
                ]
            )
        ]
        kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=settings.tool_mode)
        )
    return types.GenerateContentConfig(**kwargs)


def parse_response(response, model: str, latency_ms: float) -> LLMResult:
    candidates = getattr(response, "candidates", None) or []
    if not candidates or candidates[0].content is None:
        feedback = getattr(response, "prompt_feedback", None)
        raise LLMError(f"Gemini returned no candidate (prompt_feedback={feedback})")

    candidate = candidates[0]
    content = candidate.content
    texts: list[str] = []
    calls: list[ToolCall] = []
    for index, part in enumerate(content.parts or []):
        if part.function_call is not None:
            calls.append(
                ToolCall(
                    id=part.function_call.id or f"call_{index}",
                    name=part.function_call.name or "",
                    args=dict(part.function_call.args or {}),
                )
            )
        elif part.text and not part.thought:
            texts.append(part.text)

    meta = getattr(response, "usage_metadata", None)
    usage = Usage(
        prompt_tokens=getattr(meta, "prompt_token_count", None),
        completion_tokens=getattr(meta, "candidates_token_count", None),
        thoughts_tokens=getattr(meta, "thoughts_token_count", None),
        total_tokens=getattr(meta, "total_token_count", None),
    )
    finish = candidate.finish_reason
    return LLMResult(
        text="".join(texts).strip(),
        tool_calls=calls,
        provider=GeminiAdapter.provider,
        provider_content=content.model_dump(mode="json", exclude_none=True),
        model=model,
        model_version=getattr(response, "model_version", None) or "",
        finish_reason=getattr(finish, "value", str(finish or "")),
        usage=usage,
        latency_ms=latency_ms,
    )
