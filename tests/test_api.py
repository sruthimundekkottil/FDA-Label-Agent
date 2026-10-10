"""API tests: mock model + stub retrievers, so no key, no network and no data."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.llm import LLMError
from agent.llm.mock import MockAdapter
from agent.tools import ToolBox
from api.limits import DailyCap, SlidingWindowLimiter
from api.main import DISCLAIMER, ApiSettings, create_app

C1 = {"chunk_id": "warfarin::section::contraindications::0", "drug": "warfarin",
      "section": "contraindications", "text": "Contraindicated in pregnancy.", "is_table": False}


class _Stub:
    def retrieve(self, query, k=5, drug_filter=None):
        return [C1] if drug_filter in (None, "warfarin") else []


class _CountingToolBox(ToolBox):
    loads = 0

    def load(self):
        type(self).loads += 1
        return super().load()


def _toolbox() -> ToolBox:
    return _CountingToolBox(drugs=["warfarin", "sertraline"], retrievers={"labels": _Stub(), "tables": _Stub()})


def _client(llm=None, **settings) -> TestClient:
    config = ApiSettings(**{"rate_limit_per_minute": 100, "daily_call_cap": 100, "log_path": "", **settings})
    return TestClient(create_app(llm=llm or MockAdapter(), toolbox=_toolbox(), settings=config))


ASK = {"question": "What are the contraindications of warfarin?"}


def test_health_reports_model_and_usage() -> None:
    with _client() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["provider"] == "mock"
    assert body["auth_required"] is False and body["calls_today"] == 0


def test_ask_returns_answer_citations_and_stop_reason() -> None:
    with _client() as client:
        response = client.post("/ask", json=ASK)
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "answered"
    assert body["citations"] == [{"drug": "warfarin", "section": "contraindications", "chunk_id": C1["chunk_id"]}]
    assert body["answer"] and body["n_steps"] == 3 and body["disclaimer"] == DISCLAIMER


def test_retrievers_are_loaded_once_at_startup_not_per_request() -> None:
    _CountingToolBox.loads = 0
    with _client() as client:
        for _ in range(3):
            client.post("/ask", json=ASK)
    assert _CountingToolBox.loads == 1


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}])
def test_invalid_questions_are_rejected(payload) -> None:
    with _client() as client:
        assert client.post("/ask", json=payload).status_code == 422


# --------------------------------------------------------------------------- #
# API key
# --------------------------------------------------------------------------- #
def test_api_key_is_required_only_when_configured() -> None:
    with _client(api_key="s3cret") as client:
        assert client.post("/ask", json=ASK).status_code == 401
        assert client.post("/ask", json=ASK, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/ask", json=ASK, headers={"X-API-Key": "s3cret"}).status_code == 200
        health = client.get("/health")
    assert health.status_code == 200 and health.json()["auth_required"] is True
    assert "s3cret" not in health.text


def test_rejected_key_does_not_use_the_daily_cap() -> None:
    with _client(api_key="s3cret", daily_call_cap=1) as client:
        client.post("/ask", json=ASK)
        assert client.post("/ask", json=ASK, headers={"X-API-Key": "s3cret"}).status_code == 200


# --------------------------------------------------------------------------- #
# Rate limit and daily cap
# --------------------------------------------------------------------------- #
def test_per_ip_rate_limit_returns_429_with_retry_after() -> None:
    with _client(rate_limit_per_minute=2) as client:
        assert [client.post("/ask", json=ASK).status_code for _ in range(2)] == [200, 200]
        blocked = client.post("/ask", json=ASK)
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) >= 1


def test_rate_limit_is_per_ip_when_proxy_headers_are_trusted() -> None:
    with _client(rate_limit_per_minute=1, trust_proxy_headers=True) as client:
        assert client.post("/ask", json=ASK, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert client.post("/ask", json=ASK, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
        assert client.post("/ask", json=ASK, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429


def test_forwarded_header_is_ignored_unless_trusted() -> None:
    with _client(rate_limit_per_minute=1) as client:  # spoofing X-Forwarded-For must not bypass the limit
        assert client.post("/ask", json=ASK, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert client.post("/ask", json=ASK, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_daily_cap_blocks_after_n_calls() -> None:
    with _client(daily_call_cap=2) as client:
        assert [client.post("/ask", json=ASK).status_code for _ in range(2)] == [200, 200]
        blocked = client.post("/ask", json=ASK)
        assert client.get("/health").json()["calls_today"] == 2
    assert blocked.status_code == 429 and "daily limit" in blocked.json()["detail"]


def test_sliding_window_frees_slots_as_time_passes() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(2, 60.0, clock=lambda: now[0])
    assert limiter.check("ip") is None and limiter.check("ip") is None
    assert limiter.check("ip") == pytest.approx(60.0)
    now[0] = 61.0
    assert limiter.check("ip") is None


def test_daily_cap_resets_on_a_new_utc_day() -> None:
    day = ["2026-10-10"]
    cap = DailyCap(1, today=lambda: day[0])
    assert cap.try_take() and not cap.try_take()
    day[0] = "2026-10-11"
    assert cap.try_take()


# --------------------------------------------------------------------------- #
# Model failures
# --------------------------------------------------------------------------- #
class _FailingAdapter(MockAdapter):
    def __init__(self, message: str):
        self.message = message

    def generate(self, *args, **kwargs):
        raise LLMError(self.message)


@pytest.mark.parametrize(
    ("message", "expected"),
    [("503 UNAVAILABLE high demand", "unavailable"), ("Gemini quota used up for model x", "quota")],
)
def test_model_failure_returns_503_without_internal_details(message, expected) -> None:
    with _client(llm=_FailingAdapter(message)) as client:
        response = client.post("/ask", json=ASK)
    assert response.status_code == 503
    assert expected in response.json()["detail"] and "model x" not in response.json()["detail"]
