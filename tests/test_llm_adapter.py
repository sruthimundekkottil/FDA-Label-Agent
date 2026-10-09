"""LLM adapter tests: Gemini translation, retries, record/replay, compat shim.

No network: the Gemini client is replaced by a fake that returns real
``google.genai.types`` objects, so the translation code is exercised for real.
"""

from __future__ import annotations

import json

import pytest
from google.genai import errors, types

from agent.llm import CachedAdapter, GenerationSettings, Message, ToolResult, ToolSpec
from agent.llm.base import CacheMissError, LLMError, LLMResult, QuotaExhaustedError
from agent.llm.compat import CompatClient
from agent.llm.gemini import GeminiAdapter, to_contents
from agent.llm.mock import ScriptedAdapter
from src.rag import answer_question

SECRET = "AIza-test-secret-key-123"
SIGNATURE = b"\x00\x01opaque-signature"

SEARCH = ToolSpec(
    name="search_labels",
    description="Search drug labels.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


def _response(parts: list, *, model_version: str = "gemini-3.8-flash-001") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=5, thoughts_token_count=3, total_token_count=20
        ),
        model_version=model_version,
    )


def _call_part(call_id: str = "call-1", query: str = "warfarin pregnancy") -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(id=call_id, name="search_labels", args={"query": query}),
        thought_signature=SIGNATURE,
    )


class _FakeModels:
    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list):
        self.models = _FakeModels(outcomes)


def _gemini(outcomes: list) -> GeminiAdapter:
    adapter = GeminiAdapter("gemini-3.8-flash", api_key=SECRET, client=_FakeClient(outcomes), min_interval_s=0)
    return adapter


# --------------------------------------------------------------------------- #
# Gemini translation
# --------------------------------------------------------------------------- #
def test_function_call_is_parsed_with_usage_and_version() -> None:
    adapter = _gemini([_response([_call_part()])])
    result = adapter.generate([Message.user("q")], tools=[SEARCH])
    assert [(c.id, c.name, c.args) for c in result.tool_calls] == [
        ("call-1", "search_labels", {"query": "warfarin pregnancy"})
    ]
    assert result.text == ""
    assert result.model_version == "gemini-3.8-flash-001"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.thoughts_tokens) == (12, 5, 3)


def test_config_disables_afc_and_declares_tools() -> None:
    adapter = _gemini([_response([types.Part(text="ok")])])
    settings = GenerationSettings(temperature=0.0, thinking_level="low", tool_mode="AUTO", seed=7)
    adapter.generate([Message.user("q")], system="sys", tools=[SEARCH], settings=settings)
    config = adapter.client.models.calls[0]["config"]
    assert config.automatic_function_calling.disable is True
    assert config.temperature == 0.0 and config.seed == 7
    assert config.system_instruction == "sys"
    assert config.tools[0].function_declarations[0].name == "search_labels"
    assert config.tool_config.function_calling_config.mode.value == "AUTO"


def test_thought_signature_and_call_id_round_trip() -> None:
    adapter = _gemini([_response([_call_part()]), _response([types.Part(text="final")])])
    history = [Message.user("q")]
    first = adapter.generate(history, tools=[SEARCH])
    history += [
        first.to_message(),
        Message.tool([ToolResult(call_id="call-1", name="search_labels", content={"chunks": []})]),
    ]
    adapter.generate(history, tools=[SEARCH])

    sent = adapter.client.models.calls[1]["contents"]
    assert sent[1].role == "model"
    assert sent[1].parts[0].thought_signature == SIGNATURE
    response_part = sent[2].parts[0].function_response
    assert (sent[2].role, response_part.id, response_part.name) == ("user", "call-1", "search_labels")


def test_thought_text_is_not_returned_as_answer() -> None:
    adapter = _gemini([_response([types.Part(text="secret reasoning", thought=True), types.Part(text="answer")])])
    assert adapter.generate([Message.user("q")]).text == "answer"


def test_retries_on_429_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("agent.llm.gemini.time.sleep", lambda _s: None)
    throttled = errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    adapter = _gemini([throttled, _response([types.Part(text="ok")])])
    assert adapter.generate([Message.user("q")]).text == "ok"
    assert len(adapter.client.models.calls) == 2


def _quota_429(quota_id: str, retry_delay: str) -> errors.APIError:
    """Shaped like the real 429 seen on 2026-10-09 (same message for minute and day limits)."""

    return errors.APIError(429, {"error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                             "quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    }})


def test_daily_quota_429_fails_immediately_without_retrying(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("agent.llm.gemini.time.sleep", sleeps.append)
    adapter = _gemini([_quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "50157s")])
    with pytest.raises(QuotaExhaustedError, match="daily quota") as info:
        adapter.generate([Message.user("q")])
    assert len(adapter.client.models.calls) == 1 and sleeps == []
    assert "Retrying will not help" in str(info.value) and SECRET not in str(info.value)


def test_long_retry_delay_alone_counts_as_exhausted_quota(monkeypatch) -> None:
    monkeypatch.setattr("agent.llm.gemini.time.sleep", lambda _s: None)
    adapter = _gemini([_quota_429("SomeUnknownQuota", "7200s")])
    with pytest.raises(QuotaExhaustedError, match="2.0 h"):
        adapter.generate([Message.user("q")])


def test_per_minute_quota_429_with_same_message_is_retried(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("agent.llm.gemini.time.sleep", sleeps.append)
    adapter = _gemini([
        _quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "17s"),
        _response([types.Part(text="ok")]),
    ])
    assert adapter.generate([Message.user("q")]).text == "ok"
    assert len(adapter.client.models.calls) == 2
    assert any(delay >= 17 for delay in sleeps)  # honours the server's short retry hint


def test_503_is_still_retried(monkeypatch) -> None:
    monkeypatch.setattr("agent.llm.gemini.time.sleep", lambda _s: None)
    busy = errors.APIError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
    adapter = _gemini([busy, busy, _response([types.Part(text="ok")])])
    assert adapter.generate([Message.user("q")]).text == "ok"


def test_non_retryable_error_raises_without_leaking_key(monkeypatch) -> None:
    monkeypatch.setattr("agent.llm.gemini.time.sleep", lambda _s: None)
    bad = errors.APIError(400, {"error": {"code": 400, "message": f"bad key {SECRET}", "status": "INVALID_ARGUMENT"}})
    adapter = _gemini([bad])
    with pytest.raises(LLMError) as info:
        adapter.generate([Message.user("q")])
    assert SECRET not in str(info.value)
    assert SECRET not in repr(adapter)


def test_missing_key_is_reported_only_when_a_call_is_made() -> None:
    adapter = GeminiAdapter("gemini-3.8-flash", api_key="")
    with pytest.raises(LLMError, match="API key"):
        adapter.generate([Message.user("q")])


def test_assistant_turn_without_provider_content_is_rebuilt() -> None:
    message = LLMResult(text="", tool_calls=[], provider="other").to_message()
    message.text = "hello"
    contents = to_contents([Message.user("q"), message])
    assert contents[1].role == "model" and contents[1].parts[0].text == "hello"


# --------------------------------------------------------------------------- #
# Record / replay
# --------------------------------------------------------------------------- #
def _counting_adapter(texts: list[str]) -> ScriptedAdapter:
    return ScriptedAdapter([LLMResult(text=text) for text in texts])


def test_auto_mode_records_then_serves_from_cache(tmp_path) -> None:
    inner = _counting_adapter(["first"])
    llm = CachedAdapter(inner, mode="auto", cache_dir=tmp_path)
    one = llm.generate([Message.user("q")])
    two = llm.generate([Message.user("q")])
    assert (one.text, two.text) == ("first", "first")
    assert (one.cached, two.cached) == (False, True)
    assert len(inner.requests) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"system": "other"},
        {"tools": [SEARCH]},
        {"settings": GenerationSettings(temperature=1.0)},
        {"settings": GenerationSettings(thinking_level="high")},
        {"messages": [Message.user("different")]},
    ],
)
def test_any_request_change_is_a_cache_miss(tmp_path, change) -> None:
    inner = _counting_adapter(["a", "b"])
    llm = CachedAdapter(inner, mode="auto", cache_dir=tmp_path)
    base = {"messages": [Message.user("q")], "system": "", "tools": None, "settings": GenerationSettings()}
    llm.generate(**base)
    llm.generate(**{**base, **change})
    assert len(inner.requests) == 2


def test_replay_miss_raises_and_never_calls_provider(tmp_path) -> None:
    inner = _counting_adapter([])
    llm = CachedAdapter(inner, mode="replay", cache_dir=tmp_path)
    with pytest.raises(CacheMissError):
        llm.generate([Message.user("q")])
    assert inner.requests == []


def test_cassette_replays_a_run_without_the_working_cache(tmp_path) -> None:
    cassette = tmp_path / "cassettes" / "run.jsonl"
    recorder = CachedAdapter(_counting_adapter(["recorded"]), mode="auto", cache_dir=tmp_path / "c1", cassette=cassette)
    recorder.generate([Message.user("q")])

    # Fresh working cache (as on a fresh clone): only the cassette is available.
    replayer = CachedAdapter(_counting_adapter([]), mode="replay", cache_dir=tmp_path / "c2", cassette=cassette)
    assert replayer.generate([Message.user("q")]).text == "recorded"


def test_cache_hits_are_also_written_to_the_run_cassette(tmp_path) -> None:
    CachedAdapter(_counting_adapter(["x"]), mode="auto", cache_dir=tmp_path).generate([Message.user("q")])
    cassette = tmp_path / "run.jsonl"
    CachedAdapter(_counting_adapter([]), mode="auto", cache_dir=tmp_path, cassette=cassette).generate(
        [Message.user("q")]
    )
    assert len(cassette.read_text(encoding="utf-8").splitlines()) == 1


def test_gemini_tool_turn_survives_cache_round_trip(tmp_path) -> None:
    """A cached model turn must still carry its thought signature when replayed."""

    llm = CachedAdapter(_gemini([_response([_call_part()])]), mode="auto", cache_dir=tmp_path)
    llm.generate([Message.user("q")], tools=[SEARCH])
    replayed = CachedAdapter(_gemini([]), mode="replay", cache_dir=tmp_path).generate(
        [Message.user("q")], tools=[SEARCH]
    )
    contents = to_contents([Message.user("q"), replayed.to_message()])
    assert contents[1].parts[0].thought_signature == SIGNATURE
    for path in tmp_path.glob("*.json"):
        assert SECRET not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Compat shim: the unchanged single-shot pipeline on the new adapter
# --------------------------------------------------------------------------- #
class _StubRetriever:
    strategy = "section"
    name = "stub"

    def retrieve(self, query, k=5, drug_filter=None):
        return [
            {
                "chunk_id": "c1",
                "drug": "warfarin",
                "section": "contraindications",
                "text": "Warfarin is contraindicated in pregnancy.",
                "score": 1.0,
            }
        ]


def test_compat_client_runs_answer_question(tmp_path) -> None:
    scripted = ScriptedAdapter([LLMResult(text="Contraindicated [warfarin | contraindications | c1]")])
    client = CompatClient(CachedAdapter(scripted, mode="auto", cache_dir=tmp_path))
    result = answer_question("q", _StubRetriever(), client=client, k=1)
    assert result.citations == [{"drug": "warfarin", "section": "contraindications", "chunk_id": "c1"}]
    assert result.mock is False
    assert json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))["result"]["text"]
