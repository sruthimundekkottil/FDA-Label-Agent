"""Tests for the retrieval evaluation (offline: retrievers are stubbed)."""

from __future__ import annotations

from src.evaluate_retrieval import (
    default_output_paths,
    evaluate_configuration,
    first_hit_rank,
    is_answerable,
    is_hit,
    run_kind_for,
)
from src.utils import load_eval_questions, write_jsonl


def _question(qid: str, drug: str, section: str, question_type: str = "factual") -> dict:
    return {
        "id": qid,
        "question": f"question {qid}",
        "drug": drug,
        "gold_section": section,
        "gold_answer_keypoints": ["kp"],
        "question_type": question_type,
        "verified": False,
    }


def _chunk(chunk_id: str, drug: str, section: str, overlapping: list[str] | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "drug": drug,
        "section": section,
        "sections_overlapping": overlapping if overlapping is not None else [section],
        "text": f"text for {chunk_id}",
        "is_table": False,
    }


class _StubRetriever:
    name = "stub"
    strategy = "section"

    def __init__(self, hits_by_question: dict[str, list[dict]]):
        self.hits_by_question = hits_by_question
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        self.calls.append((query, drug_filter))
        return self.hits_by_question.get(query, [])[:k]


# -- ground truth ------------------------------------------------------------
def test_is_hit_requires_matching_drug_and_section() -> None:
    question = _question("q1", "warfarin", "contraindications")
    assert is_hit(_chunk("c1", "warfarin", "contraindications"), question)
    assert not is_hit(_chunk("c2", "warfarin", "pregnancy"), question)
    assert not is_hit(_chunk("c3", "apixaban", "contraindications"), question)


def test_is_hit_counts_fixed_chunk_overlap() -> None:
    question = _question("q1", "warfarin", "contraindications")
    overlapping = _chunk("c1", "warfarin", "mixed", overlapping=["warnings_and_cautions", "contraindications"])
    assert is_hit(overlapping, question)


def test_first_hit_rank_returns_none_when_nothing_matches() -> None:
    question = _question("q1", "warfarin", "contraindications")
    hits = [_chunk("c1", "warfarin", "pregnancy"), _chunk("c2", "warfarin", "description")]
    assert first_hit_rank(hits, question) is None
    assert first_hit_rank([_chunk("c3", "warfarin", "contraindications")], question) == 1


def test_unanswerable_and_empty_gold_section_are_excluded() -> None:
    assert is_answerable(_question("q1", "warfarin", "contraindications"))
    assert not is_answerable(_question("q2", "warfarin", "", "unanswerable"))
    assert not is_answerable(_question("q3", "warfarin", "", "factual"))


# -- metrics -----------------------------------------------------------------
def test_evaluate_configuration_computes_recall_and_mrr() -> None:
    questions = [
        _question("q1", "warfarin", "contraindications"),
        _question("q2", "warfarin", "contraindications"),
        _question("q3", "warfarin", "contraindications"),
    ]
    other = _chunk("c-other", "warfarin", "pregnancy")
    hit = _chunk("c-hit", "warfarin", "contraindications")
    retriever = _StubRetriever(
        {
            "question q1": [hit],                      # rank 1
            "question q2": [other, other, hit],        # rank 3
            "question q3": [other, other, other, other],  # miss
        }
    )
    rows = {
        (row["question_type"], row["k"]): row
        for row in evaluate_configuration(
            retriever, questions, k_values=(1, 3), drug_filter_mode="none"
        )
    }

    assert rows[("overall", 1)]["recall"] == round(1 / 3, 4)
    assert rows[("overall", 1)]["mrr"] == round(1 / 3, 4)
    assert rows[("overall", 3)]["recall"] == round(2 / 3, 4)
    assert rows[("overall", 3)]["mrr"] == round((1 + 1 / 3) / 3, 4)
    assert rows[("overall", 3)]["n_questions"] == 3
    assert rows[("factual", 3)]["recall"] == rows[("overall", 3)]["recall"]


def test_evaluate_configuration_breaks_down_by_question_type() -> None:
    questions = [
        _question("q1", "warfarin", "contraindications", "factual"),
        _question("q2", "warfarin", "contraindications", "negation"),
    ]
    hit = _chunk("c-hit", "warfarin", "contraindications")
    miss = _chunk("c-miss", "warfarin", "pregnancy")
    retriever = _StubRetriever({"question q1": [hit], "question q2": [miss]})
    rows = {
        (row["question_type"], row["k"]): row
        for row in evaluate_configuration(retriever, questions, k_values=(1,), drug_filter_mode="none")
    }
    assert rows[("factual", 1)]["recall"] == 1.0
    assert rows[("negation", 1)]["recall"] == 0.0
    assert rows[("overall", 1)]["recall"] == 0.5


def test_evaluate_configuration_skips_unanswerable_questions() -> None:
    questions = [
        _question("q1", "warfarin", "contraindications"),
        _question("q2", "warfarin", "", "unanswerable"),
    ]
    retriever = _StubRetriever({"question q1": [_chunk("c1", "warfarin", "contraindications")]})
    rows = evaluate_configuration(retriever, questions, k_values=(1,), drug_filter_mode="none")
    assert rows[0]["n_questions"] == 1
    assert len(retriever.calls) == 1, "unanswerable questions must not be retrieved for"


# -- run labelling -----------------------------------------------------------
def test_run_kind_labels_preliminary_runs() -> None:
    assert run_kind_for({"verified": 0, "unverified": 40}, False) == "preliminary"
    assert run_kind_for({"verified": 40, "unverified": 0}, False) == "verified_only"
    assert run_kind_for({"verified": 12, "unverified": 28}, True) == "verified_only"


def test_default_output_paths_are_distinct() -> None:
    preliminary_csv, preliminary_figure = default_output_paths("preliminary")
    final_csv, final_figure = default_output_paths("verified_only")
    assert preliminary_csv.name == "retrieval_results_preliminary.csv"
    assert final_csv.name == "retrieval_results.csv"
    assert preliminary_figure.name != final_figure.name


def test_load_eval_questions_verified_only_filter(tmp_path) -> None:
    path = tmp_path / "questions.jsonl"
    verified = _question("q1", "warfarin", "contraindications")
    verified["verified"] = True
    write_jsonl(path, [verified, _question("q2", "warfarin", "pregnancy")])

    everything, stats_all = load_eval_questions(path)
    assert len(everything) == 2
    assert (stats_all["verified"], stats_all["unverified"], stats_all["selected"]) == (1, 1, 2)

    only_verified, stats_verified = load_eval_questions(path, verified_only=True)
    assert [record["id"] for record in only_verified] == ["q1"]
    assert stats_verified["selected"] == 1


def test_portable_path_is_project_relative_with_forward_slashes() -> None:
    from src.evaluate_retrieval import portable_path
    from src.utils import PROJECT_ROOT

    assert portable_path(PROJECT_ROOT / "results" / "figures" / "x.png") == "results/figures/x.png"
    assert portable_path("data/eval/questions.jsonl") == "data/eval/questions.jsonl"
