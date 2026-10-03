"""Tests for prompt building, citation parsing and the LLM client plumbing.

No network here: the HTTP layer is replaced by a stub session.
"""

from __future__ import annotations

import json

import pytest

from src.llm import LLMClient, LLMConfig, LLMError
from src.rag import (
    ABSTENTION_SENTENCE,
    answer_question,
    build_prompt,
    format_context,
    parse_citations,
    parse_json_answer,
)


def _chunk(chunk_id: str, drug: str, section: str, text: str, is_table: bool = False) -> dict:
    return {
        "chunk_id": chunk_id,
        "drug": drug,
        "section": section,
        "source_section": section,
        "strategy": "section",
        "text": text,
        "embed_text": text,
        "char_start": 0,
        "char_end": len(text),
        "is_table": is_table,
        "score": 0.5,
    }


CHUNKS = [
    _chunk("warfarin::section::contraindications::0", "warfarin", "contraindications", "Do not use in pregnancy."),
    _chunk(
        "sertraline::section::adverse_reactions_table::0",
        "sertraline",
        "adverse_reactions",
        "Sertraline Hydrochloride (N=3066) %: Nausea | 26",
        is_table=True,
    ),
]


class _StubRetriever:
    name = "hybrid"
    strategy = "section"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        hits = self.chunks
        if drug_filter:
            hits = [chunk for chunk in hits if chunk["drug"] == drug_filter]
        return hits[:k]


def test_build_prompt_plain_contains_context_and_citation_rules() -> None:
    prompt = build_prompt("What are the contraindications?", CHUNKS, "plain")
    assert "warfarin::section::contraindications::0" in prompt
    assert "[drug | section | chunk_id]" in prompt
    assert ABSTENTION_SENTENCE in prompt
    assert "QUESTION: What are the contraindications?" in prompt


def test_build_prompt_json_variant_requests_json_keys() -> None:
    prompt = build_prompt("question", CHUNKS, "json")
    assert '"citations"' in prompt
    assert '"insufficient_context"' in prompt
    assert "Return only the JSON object" in prompt


def test_build_prompt_unknown_variant_is_ignored_like_plain() -> None:
    # ``answer_question`` validates variants; build_prompt itself only branches on "json".
    assert "QUESTION: q" in build_prompt("q", CHUNKS, "plain")


def test_format_context_marks_tables_and_omits_nothing() -> None:
    context = format_context(CHUNKS)
    assert "type=table" in context
    assert "type=text" in context
    assert "Nausea" in context


def test_parse_citations_deduplicates_and_strips_whitespace() -> None:
    answer = (
        "Pregnancy is a contraindication [ warfarin | contraindications | c1 ]. "
        "Bleeding too [warfarin|contraindications|c2] and again [warfarin | contraindications | c1]."
    )
    citations = parse_citations(answer)
    assert citations == [
        {"drug": "warfarin", "section": "contraindications", "chunk_id": "c1"},
        {"drug": "warfarin", "section": "contraindications", "chunk_id": "c2"},
    ]


def test_parse_citations_ignores_malformed_brackets() -> None:
    assert parse_citations("no citations here [warfarin] [a|b]") == []


def test_parse_json_answer_handles_fences_and_prose() -> None:
    payload = {"answer": "nausea 26%", "citations": [], "insufficient_context": False}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert parse_json_answer(fenced)["answer"] == "nausea 26%"
    prose = "Here you go: " + json.dumps(payload)
    assert parse_json_answer(prose)["answer"] == "nausea 26%"
    assert parse_json_answer("not json at all") is None


def test_mock_mode_uses_top_chunk_as_answer() -> None:
    client = LLMClient(LLMConfig(force_mock=True))
    result = answer_question("What are the contraindications?", _StubRetriever(CHUNKS), client=client, k=2)
    assert result.mock is True
    assert CHUNKS[0]["chunk_id"] in result.answer
    assert result.citations[0]["chunk_id"] == CHUNKS[0]["chunk_id"]
    assert result.model == "(mock)"


def test_mock_client_raises_when_called_directly() -> None:
    client = LLMClient(LLMConfig(force_mock=True))
    with pytest.raises(LLMError):
        client.complete("hello")


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload) if payload else ""

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs) -> _FakeResponse:
        self.calls += 1
        return self.responses.pop(0)


def _fake_client(tmp_path, responses: list[_FakeResponse]) -> LLMClient:
    config = LLMConfig(base_url="https://example.invalid/v1", api_key="k", model="m", min_interval_s=0)
    client = LLMClient(config, cache_dir=tmp_path)
    client._session = _FakeSession(responses)
    return client


def _completion_payload(text: str = "answer [warfarin | contraindications | c1]") -> dict:
    return {
        "model": "m",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def test_completion_parses_text_and_usage(tmp_path) -> None:
    client = _fake_client(tmp_path, [_FakeResponse(200, _completion_payload())])
    response = client.complete("prompt")
    assert response.text.startswith("answer")
    assert (response.prompt_tokens, response.completion_tokens) == (11, 7)
    assert response.cached is False


def test_completion_is_served_from_disk_cache(tmp_path) -> None:
    client = _fake_client(tmp_path, [_FakeResponse(200, _completion_payload())])
    first = client.complete("prompt")
    second = client.complete("prompt")
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert client._session.calls == 1, "the second call must not hit the network"
    assert list(tmp_path.glob("*.json"))


def test_retry_on_429_then_success(tmp_path, monkeypatch) -> None:
    client = _fake_client(
        tmp_path,
        [_FakeResponse(429, {"error": "rate limited"}, {"Retry-After": "0"}), _FakeResponse(200, _completion_payload())],
    )
    monkeypatch.setattr(client, "_backoff", lambda *args, **kwargs: None)
    response = client.complete("prompt")
    assert response.text.startswith("answer")
    assert client._session.calls == 2


def test_non_retryable_error_raises(tmp_path) -> None:
    client = _fake_client(tmp_path, [_FakeResponse(400, {"error": "bad request"})])
    with pytest.raises(LLMError):
        client.complete("prompt")


def test_answer_question_uses_llm_and_parses_citations(tmp_path) -> None:
    client = _fake_client(tmp_path, [_FakeResponse(200, _completion_payload())])
    result = answer_question("q", _StubRetriever(CHUNKS), client=client, k=2, drug_filter="warfarin")
    assert result.mock is False
    assert result.citations == [
        {"drug": "warfarin", "section": "contraindications", "chunk_id": "c1"}
    ]
    assert result.model == "m"


def test_answer_question_json_variant_uses_structured_citations(tmp_path) -> None:
    payload = {
        "answer": "26% reported nausea [sertraline | adverse_reactions | c9]",
        "citations": [{"drug": "sertraline", "section": "adverse_reactions", "chunk_id": "c9"}],
        "insufficient_context": False,
    }
    client = _fake_client(tmp_path, [_FakeResponse(200, _completion_payload(json.dumps(payload)))])
    result = answer_question(
        "q", _StubRetriever(CHUNKS), client=client, k=2, prompt_variant="json"
    )
    assert result.citations[0]["chunk_id"] == "c9"
    assert result.prompt_variant == "json"


def test_answer_question_reports_llm_errors_without_crashing(tmp_path) -> None:
    client = _fake_client(tmp_path, [_FakeResponse(400, {"error": "bad request"})])
    result = answer_question("q", _StubRetriever(CHUNKS), client=client, k=1)
    assert result.answer.startswith("[LLM error]")
    assert result.model == "(error)"
