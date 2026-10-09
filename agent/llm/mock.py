"""Offline adapters: a scripted one for tests and a heuristic one for no-key runs."""

from __future__ import annotations

import re
from typing import Callable

from agent.llm.base import GenerationSettings, LLMResult, Message, ToolCall, ToolSpec

_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)")
_CHUNK_RE = re.compile(r"\[\d+\] drug=(.+?) \| section=(.+?) \| chunk_id=(\S+)")


class ScriptedAdapter:
    """Returns pre-written results in order and records every request it saw."""

    provider = "scripted"

    def __init__(self, script: list[LLMResult | Callable[[list[Message]], LLMResult]], model: str = "scripted"):
        self.script = list(script)
        self.model = model
        self.requests: list[dict] = []

    def generate(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        settings: GenerationSettings | None = None,
    ) -> LLMResult:
        self.requests.append(
            {"messages": list(messages), "system": system, "tools": tools, "settings": settings}
        )
        if not self.script:
            raise AssertionError("ScriptedAdapter ran out of scripted responses")
        step = self.script.pop(0)
        result = step(messages) if callable(step) else step
        result.provider = result.provider or self.provider
        result.model = result.model or self.model
        return result


class MockAdapter:
    """Keeps the agent runnable with no API key.

    Whenever tools are offered it calls the first one, filling the required
    arguments: string arguments get the question, except ``answer``, which quotes
    the top evidence chunk with its citation; ``*chunk*`` arrays get that chunk's id.
    """

    provider = "mock"
    model = "(mock)"

    def generate(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        settings: GenerationSettings | None = None,
    ) -> LLMResult:
        prompt = next((m.text for m in reversed(messages) if m.role == "user"), "")
        match = _QUESTION_RE.search(prompt)
        question = match.group(1).strip() if match else prompt[:200]
        if not tools:
            return LLMResult(text=f"[mock mode] {question}", provider=self.provider, model=self.model)

        top = _CHUNK_RE.search(prompt)
        spec = tools[0]
        properties = spec.parameters.get("properties", {})
        args: dict = {}
        for name in spec.parameters.get("required", []):
            kind = properties.get(name, {}).get("type")
            if kind == "array":
                args[name] = [top.group(3)] if top and "chunk" in name else []
            elif name == "answer" and top:
                args[name] = (
                    "[mock mode: no LLM configured, citing the top retrieved passage] "
                    f"[{top.group(1)} | {top.group(2)} | {top.group(3)}]"
                )
            else:
                args[name] = question
        call = ToolCall(id="mock_0", name=spec.name, args=args)
        return LLMResult(text="", tool_calls=[call], provider=self.provider, model=self.model)
