"""Offline tests for the answer-quality evaluator (no API calls)."""

from __future__ import annotations

import csv

import pytest

from src.evaluate_answers import (
    AnswerMetrics,
    ConfigSpec,
    best_config_from_csv,
    content_terms,
    is_abstention,
    keypoint_coverage,
    parse_config,
    score_citations,
    summarize_config,
)


class _FakeResult:
    def __init__(self, answer: str, citations: list[dict], chunk_ids: list[str]):
        self.answer = answer
        self.citations = citations
        self.retrieved_chunks = [{"chunk_id": chunk_id} for chunk_id in chunk_ids]


def _question(section: str = "contraindications", drug: str = "warfarin", keypoints=None) -> dict:
    return {
        "id": "q001",
        "question": "question?",
        "drug": drug,
        "gold_section": section,
        "gold_answer_keypoints": keypoints or [],
        "question_type": "factual",
    }


def test_parse_config_validates_fields() -> None:
    assert parse_config("section:hybrid:auto").label == "section/hybrid/auto"
    with pytest.raises(SystemExit):
        parse_config("section:hybrid")
    with pytest.raises(SystemExit):
        parse_config("section:magic:auto")


def test_best_config_from_csv_uses_recall_then_mrr(tmp_path) -> None:
    path = tmp_path / "retrieval_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chunking", "retrieval", "k", "drug_filter", "question_type", "recall", "mrr"],
        )
        writer.writeheader()
        writer.writerow({"chunking": "fixed", "retrieval": "vector", "k": 5, "drug_filter": "none", "question_type": "overall", "recall": 0.9, "mrr": 0.9})
        writer.writerow({"chunking": "section", "retrieval": "hybrid_rerank", "k": 5, "drug_filter": "auto", "question_type": "overall", "recall": 1.0, "mrr": 0.88})
        writer.writerow({"chunking": "table_aware", "retrieval": "hybrid_rerank", "k": 5, "drug_filter": "auto", "question_type": "overall", "recall": 1.0, "mrr": 0.89})
        writer.writerow({"chunking": "section", "retrieval": "bm25", "k": 1, "drug_filter": "none", "question_type": "overall", "recall": 0.5, "mrr": 0.5})

    best = best_config_from_csv(path)
    assert best.label == "table_aware/hybrid_rerank/auto"


def test_keypoint_coverage_is_lexical() -> None:
    keypoints = ["Warfarin is contraindicated in pregnancy", "Hemorrhagic tendencies"]
    assert keypoint_coverage("Pregnancy is a contraindication.", keypoints) == 0.5
    assert keypoint_coverage("nothing relevant here", keypoints) == 0.0
    assert keypoint_coverage("anything", []) is None


def test_content_terms_drop_question_words() -> None:
    terms = content_terms("What are the contraindications of warfarin according to the label?")
    assert "contraindications" in terms
    assert "label" not in terms
    assert "according" not in terms


def test_is_abstention_detects_declining_answers() -> None:
    assert is_abstention("The provided labels do not contain this information")
    assert is_abstention("The label does not provide the price.")
    assert not is_abstention("Warfarin is contraindicated in pregnancy.")


def test_score_citations_flags_wrong_section_and_fabricated_ids() -> None:
    question = _question()
    good = _FakeResult("a", [{"drug": "warfarin", "section": "contraindications", "chunk_id": "c1"}], ["c1"])
    assert score_citations(question, good) == (1, 1, 0, True)

    wrong_section = _FakeResult("a", [{"drug": "warfarin", "section": "warnings_and_cautions", "chunk_id": "c1"}], ["c1"])
    assert score_citations(question, wrong_section) == (1, 0, 0, False)

    fabricated = _FakeResult("a", [{"drug": "warfarin", "section": "contraindications", "chunk_id": "ghost"}], ["c1"])
    assert score_citations(question, fabricated) == (1, 0, 1, False)

    uncited = _FakeResult("a", [], ["c1"])
    assert score_citations(question, uncited) == (0, 0, 0, False)


def test_summarize_config_aggregates_metrics() -> None:
    spec = ConfigSpec("section", "hybrid_rerank", "auto")
    details = [
        AnswerMetrics(
            question_id="q1", question="?", question_type="factual", drug="warfarin",
            gold_section="contraindications", answer="a", n_citations=1, citations_correct=1,
            citation_ok=True, keypoint_coverage=1.0, latency_ms=100, prompt_tokens=10, completion_tokens=5,
        ),
        AnswerMetrics(
            question_id="q2", question="?", question_type="factual", drug="warfarin",
            gold_section="contraindications", answer="b", n_citations=1, citations_correct=0,
            citation_ok=False, keypoint_coverage=0.0, latency_ms=300, prompt_tokens=20, completion_tokens=5,
        ),
        AnswerMetrics(
            question_id="q3", question="?", question_type="unanswerable", drug="warfarin",
            gold_section="", answer="not in the label", abstained=True, latency_ms=200,
        ),
    ]
    summary = summarize_config(spec, details, faithfulness=False)
    assert summary["n_questions"] == 3
    assert summary["n_answerable"] == 2
    assert summary["n_unanswerable"] == 1
    assert summary["citation_accuracy"] == 0.5
    assert summary["citation_precision"] == 0.5
    assert summary["abstention_accuracy"] == 1.0
    assert summary["keypoint_coverage"] == 0.5
    assert summary["avg_latency_ms"] == 200.0
    assert summary["total_prompt_tokens"] == 30
