"""Provider-neutral message, tool and result types shared by every adapter.

The agent loop only ever sees these dataclasses. Each adapter (``gemini.py``,
``mock.py``) translates them to and from its provider's wire format, so swapping
providers never touches the loop.

``provider_content`` carries the provider's own serialisation of an assistant
turn. Gemini 3 attaches thought signatures to its parts and requires them to be
sent back unchanged, so the adapter replays that blob verbatim instead of
rebuilding the turn from ``text``/``tool_calls``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

ROLES = ("user", "assistant", "tool")
TOOL_MODES = ("AUTO", "ANY", "NONE", "VALIDATED")
THINKING_LEVELS = ("minimal", "low", "medium", "high")


class LLMError(RuntimeError):
    """The provider could not produce a response (after retries)."""


class QuotaExhaustedError(LLMError):
    """A long-term quota (e.g. requests per day) is used up; retrying cannot help."""


class CacheMissError(LLMError):
    """Replay mode found no recorded response for a request."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments object


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: dict


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider: str = ""
    provider_content: dict | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {ROLES}")

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", text=text)

    @classmethod
    def tool(cls, results: list[ToolResult]) -> "Message":
        return cls(role="tool", tool_results=list(results))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationSettings:
    temperature: float = 0.0
    max_output_tokens: int = 4096  # Gemini counts thinking tokens against this
    thinking_level: str = "low"
    tool_mode: str = "AUTO"
    seed: int | None = None

    def __post_init__(self) -> None:
        self.tool_mode = self.tool_mode.upper()
        self.thinking_level = self.thinking_level.lower()
        if self.tool_mode not in TOOL_MODES:
            raise ValueError(f"unknown tool_mode {self.tool_mode!r}; expected one of {TOOL_MODES}")
        if self.thinking_level not in THINKING_LEVELS:
            raise ValueError(
                f"unknown thinking_level {self.thinking_level!r}; expected one of {THINKING_LEVELS}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thoughts_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    provider_content: dict | None = None
    model: str = ""
    model_version: str = ""
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cached: bool = False

    def to_message(self) -> Message:
        """The assistant turn to append to the history."""

        return Message(
            role="assistant",
            text=self.text,
            tool_calls=list(self.tool_calls),
            provider=self.provider,
            provider_content=self.provider_content,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "LLMResult":
        data = dict(payload)
        data["tool_calls"] = [ToolCall(**call) for call in data.get("tool_calls") or []]
        data["usage"] = Usage(**(data.get("usage") or {}))
        return cls(**data)


class LLMAdapter(Protocol):
    provider: str
    model: str

    def generate(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        settings: GenerationSettings | None = None,
    ) -> LLMResult: ...


def request_fingerprint(
    *,
    provider: str,
    model: str,
    messages: list[Message],
    system: str,
    tools: list[ToolSpec] | None,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Everything that can change a response - the input to the cache key."""

    return {
        "provider": provider,
        "model": model,
        "system": system,
        "messages": [message.to_dict() for message in messages],
        "tools": [asdict(tool) for tool in tools or []],
        "settings": settings.to_dict(),
    }
