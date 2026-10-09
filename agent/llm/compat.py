"""Lets the unchanged single-shot pipeline (``src.rag.answer_question``) run on the new adapter.

The baseline and the agent then share one model, one set of generation settings
and one record/replay cache, so the comparison is apples to apples.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.llm.base import GenerationSettings, LLMAdapter, LLMError, Message
from agent.llm.cache import CachedAdapter
from src import llm as legacy


@dataclass
class _Config:
    model: str


class CompatClient:
    """Duck-types ``src.llm.LLMClient``: ``.mock``, ``.config.model``, ``.complete()``."""

    def __init__(self, adapter: LLMAdapter, settings: GenerationSettings | None = None):
        self.adapter = adapter
        self.settings = settings or GenerationSettings()
        self.config = _Config(model=adapter.model)

    @property
    def mock(self) -> bool:
        return self.adapter.provider == "mock"

    def complete(self, prompt: str, *, system: str = "", use_cache: bool = True, **_ignored) -> legacy.LLMResponse:
        adapter = self.adapter
        if not use_cache and isinstance(adapter, CachedAdapter):
            adapter = adapter.inner
        try:
            result = adapter.generate([Message.user(prompt)], system=system, settings=self.settings)
        except LLMError as exc:
            raise legacy.LLMError(str(exc)) from None
        return legacy.LLMResponse(
            text=result.text,
            model=result.model_version or result.model,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=result.latency_ms,
            cached=result.cached,
        )
