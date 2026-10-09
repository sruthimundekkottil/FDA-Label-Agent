"""LLM adapter layer: pick a provider from ``.env`` and wrap it in record/replay.

    from agent.llm import get_adapter, settings_from_env
    llm = get_adapter(cassette=Path("results/cassettes/run1.jsonl"))
    result = llm.generate([Message.user("hi")], settings=settings_from_env())

Environment (all optional except the key/model for real calls):

    LLM_PROVIDER         gemini | mock      (default: gemini if a key is set, else mock)
    LLM_MODEL            e.g. gemini-3.8-flash
    GEMINI_API_KEY       falls back to LLM_API_KEY
    LLM_CACHE_MODE       auto | record | replay | off   (default auto)
    LLM_TEMPERATURE      default 0
    LLM_THINKING_LEVEL   minimal | low | medium | high  (default low)
    LLM_MAX_OUTPUT_TOKENS, LLM_SEED
    LLM_MOCK=1           force the mock provider
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.llm.base import (
    CacheMissError,
    GenerationSettings,
    LLMAdapter,
    LLMError,
    LLMResult,
    Message,
    QuotaExhaustedError,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from agent.llm.cache import CachedAdapter
from agent.llm.mock import MockAdapter
from src.utils import env_flag, load_env

__all__ = [
    "CacheMissError",
    "CachedAdapter",
    "GenerationSettings",
    "LLMAdapter",
    "LLMError",
    "LLMResult",
    "Message",
    "QuotaExhaustedError",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "get_adapter",
    "settings_from_env",
]


def _api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()


def settings_from_env(**overrides) -> GenerationSettings:
    load_env()
    values: dict = {}
    if os.environ.get("LLM_TEMPERATURE"):
        values["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    if os.environ.get("LLM_MAX_OUTPUT_TOKENS"):
        values["max_output_tokens"] = int(os.environ["LLM_MAX_OUTPUT_TOKENS"])
    if os.environ.get("LLM_THINKING_LEVEL"):
        values["thinking_level"] = os.environ["LLM_THINKING_LEVEL"].strip()
    if os.environ.get("LLM_SEED"):
        values["seed"] = int(os.environ["LLM_SEED"])
    values.update({key: value for key, value in overrides.items() if value is not None})
    return GenerationSettings(**values)


def get_adapter(
    *,
    provider: str | None = None,
    cache_mode: str | None = None,
    cassette: Path | None = None,
    cache_dir: Path | None = None,
) -> CachedAdapter:
    load_env()
    if env_flag("LLM_MOCK", False):
        provider = "mock"
    mode = (cache_mode or os.environ.get("LLM_CACHE_MODE") or "auto").strip().lower()
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        # Replay needs no key, so a configured model is enough to pick Gemini.
        provider = "gemini" if (_api_key() or mode == "replay") and os.environ.get("LLM_MODEL") else "mock"

    if provider == "gemini":
        from agent.llm.gemini import GeminiAdapter

        inner = GeminiAdapter(os.environ.get("LLM_MODEL", "").strip(), api_key=_api_key())
    elif provider == "mock":
        inner = MockAdapter()
        mode = "off"  # nothing worth recording
    else:
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}; expected gemini or mock")
    return CachedAdapter(inner, mode=mode, cassette=cassette, cache_dir=cache_dir)
