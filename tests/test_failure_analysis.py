"""Offline tests for the failure dump and the drafted markdown."""

from __future__ import annotations

from src.failure_analysis import (
    category_counts,
    collect_failures,
    draft_hypothesis,
    draft_markdown,
    is_retrieval_hit,
    select_cases,
)
from src.utils import write_jsonl


def _chunk(chunk_id: str, drug: str, section: str, text: str = "text", overlapping=None) -> dict:
    return {
        "chunk_id": chunk_id,
        "drug": drug,
        "section": section,
        "source_section": section,
        "strategy": "section",
        "text": text,
        "is_table": False,
        "sections_overlapping": overlapping or [section],
        "n_chars": len(text),
    }


def _question(qid: str, qtype: str = "factual", section: str = "contraindications") -> dict:
    return {
        "id": qid,
        "question": f"question {qid}?",
        "drug": "warfarin",
        "gold_section": section,
        "gold_answer_keypoints": ["kp"],
        "question_type": qtype,
        "verified": True,
    }


def _detail(qid: str, **overrides) -> dict:
    base = {
        "question_id": qid,
        "question": f"question {qid}?",
        "question_type": "factual",
        "drug": "warfarin",
        "gold_section": "contraindications",
        "answer": "answer text",
        "citations": [],
        "retrieved_chunk_ids": ["c1"],
        "citation_ok": True,
        "abstained": None,
        "keypoint_coverage": 1.0,
        "faithfulness_ratio": 1.0,
    }
    base.update(overrides)
    return base


def test_is_retrieval_hit_matches_drug_section_and_overlap() -> None:
    chunk_index = {
        "c1": _chunk("c1", "warfarin", "pregnancy"),
        "c2": _chunk("c2", "warfarin", "mixed", overlapping=["warnings_and_cautions", "contraindications"]),
    }
    question = _question("q1")
    assert not is_retrieval_hit(question, ["c1"], chunk_index)
    assert is_retrieval_hit(question, ["c2"], chunk_index)


def test_collect_failures_classifies_categories(tmp_path) -> None:
    sections_path = tmp_path / "sections.jsonl"
    write_jsonl(
        sections_path,
        [
            {
                "drug": "warfarin",
                "set_id": "s",
                "section": "contraindications",
                "text": "Warfarin is contraindicated in pregnancy.",
                "source_field": "contraindications",
                "effective_time": "20250101",
            }
        ],
    )
    chunk_index = {"c1": _chunk("c1", "warfarin", "pregnancy")}
    questions = [_question("q1"), _question("q2", qtype="unanswerable", section="")]
    details = [
        _detail("q1", citation_ok=False, faithfulness_ratio=0.5),
        _detail("q2", question_type="unanswerable", gold_section="", abstained=False, citation_ok=False),
    ]
    failures = collect_failures(details, questions, chunk_index, sections_path=sections_path)
    counts = category_counts(failures)
    assert counts["retrieval_miss"] == 1
    assert counts["citation_error"] == 1
    assert counts["unsupported_claim"] == 1
    assert counts["no_abstention"] == 1
    assert "gold_text" in failures[0]


def test_draft_hypothesis_branches() -> None:
    table_question = _question("q1", qtype="table", section="adverse_reactions")
    assert "parsed table" in draft_hypothesis(table_question, "retrieval_miss", ["adverse_reactions"])
    abbreviation = {"question": "What does the label say about CrCl?", "question_type": "factual", "gold_section": "dosage_and_administration"}
    assert "Abbreviation" in draft_hypothesis(abbreviation, "retrieval_miss", [])
    negation = _question("q3", qtype="negation")
    assert "Negation" in draft_hypothesis(negation, "retrieval_miss", [])
    assert "unanswerable" in draft_hypothesis(_question("q4"), "no_abstention", [])


def test_select_cases_prefers_one_per_category_and_respects_limit() -> None:
    failures = [
        {"question_id": "q1", "category": "retrieval_miss"},
        {"question_id": "q2", "category": "citation_error"},
        {"question_id": "q3", "category": "citation_error"},
        {"question_id": "q4", "category": "unsupported_claim"},
        {"question_id": "q5", "category": "no_abstention"},
    ]
    cases = select_cases(failures, max_cases=4)
    assert [case["category"] for case in cases] == [
        "retrieval_miss",
        "citation_error",
        "unsupported_claim",
        "no_abstention",
    ]
    assert len(select_cases(failures, max_cases=2)) == 2


def test_draft_markdown_contains_cases_and_confirmation_note() -> None:
    failures = [
        {
            "category": "retrieval_miss",
            "question_id": "q1",
            "question": "What are the contraindications of warfarin?",
            "question_type": "factual",
            "drug": "warfarin",
            "gold_section": "contraindications",
            "answer": "an answer",
            "citations": [],
            "citation_ok": False,
            "keypoint_coverage": 0.0,
            "faithfulness_ratio": None,
            "retrieved": [{"chunk_id": "c1", "drug": "warfarin", "section": "pregnancy", "is_table": False, "text": "..."}],
            "retrieved_sections": ["pregnancy"],
            "gold_text": "Warfarin is contraindicated in pregnancy.",
            "hypothesis": "wording mismatch",
        }
    ]
    markdown = draft_markdown(
        failures,
        spec_label="section / hybrid_rerank / auto",
        model="gemini-test",
        n_questions=21,
        n_answerable=18,
        n_unanswerable=3,
    )
    assert "# Failure analysis" in markdown
    assert "must read and confirm" in markdown
    assert "## Case 1" in markdown
    assert "What should have been retrieved" in markdown


def test_draft_markdown_handles_no_failures() -> None:
    markdown = draft_markdown([], spec_label="section / hybrid / none", model="m", n_questions=1, n_answerable=1, n_unanswerable=0)
    assert "No failures were detected" in markdown
