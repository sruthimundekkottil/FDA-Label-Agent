"""Tests for keypoint selection and the in-place keypoint refresh."""

from __future__ import annotations

import json

from src.draft_eval_questions import (
    backup_question_file,
    refresh_keypoints,
    select_keypoints,
)
from src.utils import write_jsonl


def _sections(tmp_path, records: list[dict]):
    path = tmp_path / "sections.jsonl"
    write_jsonl(path, records)
    return path


def _section(drug: str, section: str, text: str, source_field: str | None = None) -> dict:
    return {
        "drug": drug,
        "set_id": "set-1",
        "section": section,
        "text": text,
        "source_field": source_field or section,
        "effective_time": "20250101",
    }


def _question(qid: str, drug: str, section: str, question: str, keypoints: list[str], *, verified=False, qtype="factual"):
    return {
        "id": qid,
        "question": question,
        "drug": drug,
        "gold_section": section,
        "gold_answer_keypoints": keypoints,
        "question_type": qtype,
        "verified": verified,
    }


# -- keypoint selection -------------------------------------------------------
def test_reporting_lines_are_not_keypoints() -> None:
    text = (
        "Most common adverse reactions (incidence ≥8%) were: dizziness, somnolence, "
        "and peripheral edema. To report SUSPECTED ADVERSE REACTIONS, contact ACI "
        "HealthcareUSA, Inc. at 1-888-802-1213 or www.acihealthcareusa.com or FDA at "
        "1-800-FDA-1088 or www.fda.gov/medwatch."
    )
    keypoints = select_keypoints(
        text,
        "What adverse reactions are reported in the gabapentin label?",
        section="adverse_reactions",
        drug="gabapentin",
    )
    joined = " ".join(keypoints).lower()
    assert "aci healthcareusa" not in joined
    assert "1-888" not in joined
    assert "www." not in joined
    assert "most common adverse reactions" in joined


def test_stewardship_and_section_number_headers_are_dropped() -> None:
    text = (
        "1 INDICATIONS AND USAGE To reduce the development of drug-resistant bacteria "
        "and maintain the effectiveness of amoxicillin, amoxicillin should be used only "
        "to treat infections that are proven or strongly suspected to be caused by "
        "susceptible bacteria. Amoxicillin is indicated for treatment of the following: "
        "• Lower respiratory tract infections • Acute bacterial otitis media • Sinusitis"
    )
    keypoints = select_keypoints(
        text,
        "What is amoxicillin indicated for?",
        section="indications_and_usage",
        drug="amoxicillin",
    )
    joined = " ".join(keypoints).lower()
    assert "drug-resistant bacteria" not in joined
    assert "indications and usage" not in joined
    assert "lower respiratory tract infections" in joined
    assert "acute bacterial otitis media" in joined


def test_list_intro_items_are_captured() -> None:
    text = (
        "Drug X is contraindicated in: • Pregnancy • Severe renal impairment "
        "• History of angioedema"
    )
    keypoints = select_keypoints(
        text, "What are the contraindications of drug x?", section="contraindications", drug="drug x"
    )
    joined = " ".join(keypoints)
    assert "Pregnancy" in joined
    assert "Severe renal impairment" in joined
    assert "History of angioedema" in joined


def test_question_terms_choose_the_answer_sentence() -> None:
    text = (
        "Nausea was reported by 26% of patients. Dizziness was reported by 4% of "
        "patients. Headache was reported by 3% of patients."
    )
    keypoints = select_keypoints(
        text,
        "What percentage of patients reported nausea?",
        section="adverse_reactions",
        drug="drug x",
        question_type="factual",
    )
    assert "Nausea" in keypoints[0]
    assert "26%" in keypoints[0]


def test_table_question_returns_the_matching_row_whole() -> None:
    text = (
        "Table 1:\n"
        "Hyperuricemia | Olmesartan/HCTZ (N=247) (%): 4 | Olmesartan (N=125) (%): 0\n"
        "Dizziness | Olmesartan/HCTZ (N=247) (%): 9 | Olmesartan (N=125) (%): 2"
    )
    keypoints = select_keypoints(
        text,
        "In the drug table, what does the label report for Hyperuricemia?",
        section="adverse_reactions",
        question_type="table",
        drug="drug x",
    )
    assert len(keypoints) == 1
    assert keypoints[0].startswith("Hyperuricemia")
    assert "Olmesartan/HCTZ (N=247) (%): 4" in keypoints[0]


def test_cross_reference_only_fragments_are_dropped() -> None:
    text = (
        "Severe renal impairment (eGFR below 30 mL/min) is a contraindication. "
        "[see Warnings and Precautions (5.1)] ( 4 )"
    )
    keypoints = select_keypoints(
        text, "What are the contraindications?", section="contraindications", drug="drug x"
    )
    assert all("see Warnings and Precautions" not in keypoint for keypoint in keypoints)


# -- refresh -----------------------------------------------------------------
def test_refresh_only_updates_unverified_keypoints(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    records = [
        _question(
            "q001",
            "gabapentin",
            "adverse_reactions",
            "What adverse reactions are reported in the gabapentin label?",
            ["hand-written keypoint that must survive"],
            verified=True,
        ),
        _question(
            "q002",
            "gabapentin",
            "adverse_reactions",
            "What adverse reactions are reported in the gabapentin label?",
            ["old boilerplate keypoint at 1-888-802-1213"],
        ),
    ]
    write_jsonl(questions, records)
    sections = _sections(
        tmp_path,
        [
            _section(
                "gabapentin",
                "adverse_reactions",
                "Most common adverse reactions were: dizziness, somnolence, and "
                "peripheral edema. To report SUSPECTED ADVERSE REACTIONS, contact ACI "
                "HealthcareUSA at 1-888-802-1213.",
            )
        ],
    )

    result = refresh_keypoints(questions, sections_path=sections)
    assert result["changed"] == 1
    assert result["verified_skipped"] == 1

    updated = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines()]
    assert updated[0]["gold_answer_keypoints"] == ["hand-written keypoint that must survive"]
    assert updated[0]["verified"] is True
    assert "1-888" not in " ".join(updated[1]["gold_answer_keypoints"])
    assert updated[1]["gold_answer_keypoints"]
    # structure is untouched
    assert (updated[1]["id"], updated[1]["question"], updated[1]["drug"], updated[1]["gold_section"]) == (
        "q002",
        "What adverse reactions are reported in the gabapentin label?",
        "gabapentin",
        "adverse_reactions",
    )
    assert updated[1]["verified"] is False


def test_refresh_dry_run_leaves_the_file_alone(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    write_jsonl(
        questions,
        [
            _question(
                "q001",
                "gabapentin",
                "adverse_reactions",
                "What adverse reactions are reported?",
                ["stale keypoint"],
            )
        ],
    )
    before = questions.read_bytes()
    sections = _sections(
        tmp_path,
        [_section("gabapentin", "adverse_reactions", "Most common adverse reactions were: dizziness.")],
    )
    result = refresh_keypoints(questions, sections_path=sections, dry_run=True)
    assert result["changed"] == 1
    assert questions.read_bytes() == before


def test_refresh_skips_unanswerable_questions(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    write_jsonl(
        questions,
        [
            _question(
                "q001",
                "gabapentin",
                "",
                "What is the price of gabapentin?",
                [],
                qtype="unanswerable",
            )
        ],
    )
    sections = _sections(
        tmp_path, [_section("gabapentin", "adverse_reactions", "Most common reactions were: dizziness.")]
    )
    result = refresh_keypoints(questions, sections_path=sections)
    assert result["changed"] == 0


def test_backup_question_file_writes_a_copy(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [_question("q001", "d", "s", "q?", ["kp"])])
    backup = backup_question_file(questions)
    assert backup.exists()
    assert "backup-" in backup.name
    assert backup.read_bytes() == questions.read_bytes()
