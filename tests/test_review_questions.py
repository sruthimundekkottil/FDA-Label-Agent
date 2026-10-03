"""Tests for the interactive question reviewer (no real terminal needed)."""

from __future__ import annotations

import json

from src.review_questions import (
    clip_text,
    counts,
    load_section_index,
    make_backup,
    run_review,
    save_records,
    section_preview,
)
from src.utils import write_jsonl


def _question(qid: str, drug: str, section: str, question_type: str = "factual") -> dict:
    return {
        "id": qid,
        "question": f"question {qid}",
        "drug": drug,
        "gold_section": section,
        "gold_answer_keypoints": [f"keypoint {qid}"],
        "question_type": question_type,
        "verified": False,
    }


def _sections_file(tmp_path):
    path = tmp_path / "sections.jsonl"
    write_jsonl(
        path,
        [
            {
                "drug": "warfarin",
                "set_id": "set-1",
                "section": "contraindications",
                "text": "Warfarin is contraindicated in pregnancy.",
                "source_field": "contraindications",
                "effective_time": "20250101",
            },
            {
                "drug": "warfarin",
                "set_id": "set-1",
                "section": "adverse_reactions_table",
                "text": "<table><tr><td>Nausea</td><td>26</td></tr></table>",
                "source_field": "adverse_reactions_table",
                "effective_time": "20250101",
            },
        ],
    )
    return path


def test_clip_text_keeps_whole_rows() -> None:
    text = "\n".join(f"row {index} | value {index}" for index in range(50))
    clipped = clip_text(text, 60)
    assert clipped.endswith("… (truncated)")
    for line in clipped.split("\n")[:-1]:
        assert line in text.split("\n"), line


def test_section_preview_uses_plain_text_for_factual_questions(tmp_path) -> None:
    plain, tables = load_section_index(_sections_file(tmp_path))
    label, preview, length = section_preview(
        _question("q001", "warfarin", "contraindications"), plain, tables
    )
    assert "contraindications" in label
    assert "pregnancy" in preview
    assert length == len("Warfarin is contraindicated in pregnancy.")


def test_section_preview_uses_parsed_table_for_table_questions(tmp_path) -> None:
    plain, tables = load_section_index(_sections_file(tmp_path))
    label, preview, _ = section_preview(
        _question("q002", "warfarin", "adverse_reactions", question_type="table"),
        plain,
        tables,
    )
    assert "parsed table" in label
    assert "<table>" not in preview
    assert "Nausea" in preview and "26" in preview


def test_section_preview_handles_missing_gold_section(tmp_path) -> None:
    plain, tables = load_section_index(_sections_file(tmp_path))
    label, preview, length = section_preview(
        _question("q003", "aspirin", ""), plain, tables
    )
    assert "not found" in label
    assert preview == "" and length == 0


def test_make_backup_copies_the_file(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [_question("q001", "warfarin", "contraindications")])
    backup = make_backup(questions)
    assert backup.name == "questions.backup.jsonl"
    assert backup.read_bytes() == questions.read_bytes()


def test_save_records_is_atomic_and_leaves_no_temp_file(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    records = [_question("q001", "warfarin", "contraindications")]
    save_records(questions, records)
    assert [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines()] == records
    assert not (tmp_path / "questions.jsonl.tmp").exists()


def test_run_review_verify_delete_skip_quit(tmp_path) -> None:
    records = [
        _question("q001", "warfarin", "contraindications"),
        _question("q002", "warfarin", "contraindications"),
        _question("q003", "warfarin", "contraindications"),
        _question("q004", "warfarin", "contraindications"),
    ]
    plain, tables = load_section_index(_sections_file(tmp_path))
    answers = iter(["s", "v", "d", "q"])
    printed: list[str] = []
    saved: list[list[dict]] = []

    summary = run_review(
        records,
        plain,
        tables,
        save_fn=lambda items: saved.append([dict(item) for item in items]),
        input_fn=lambda prompt="": next(answers),
        print_fn=printed.append,
    )

    assert summary["verified"] == 1
    assert summary["deleted"] == 1
    assert summary["skipped"] == 1
    assert [record["id"] for record in records] == ["q001", "q002", "q004"]
    assert records[1]["verified"] is True
    assert len(saved) == 2, "verify and delete must both save immediately"
    assert any("verified so far" in line for line in printed)


def test_run_review_stops_on_eof(tmp_path) -> None:
    records = [_question("q001", "warfarin", "contraindications")]
    plain, tables = load_section_index(_sections_file(tmp_path))

    def _raise(prompt=""):
        raise EOFError

    summary = run_review(
        records,
        plain,
        tables,
        save_fn=lambda items: None,
        input_fn=_raise,
        print_fn=lambda *args: None,
    )
    assert summary["verified"] == 0
    assert counts(records) == (0, 1, 1)
