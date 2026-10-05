"""Dump every failed case and draft ``results/failures.md``.

    python -m src.failure_analysis --verified-only
    python -m src.failure_analysis --verified-only --strategy table_aware --retriever hybrid --drug-filter auto

The script reads the per-question details written by ``evaluate_answers`` and the
chunk index, classifies each failure, and writes:

* ``results/failure_dump.jsonl`` - the raw dump (every failure with the question,
  the answer, the retrieved chunks, what should have been retrieved, and an
  automatically drafted hypothesis);
* ``results/failures.md`` - 3-5 curated cases in the spec's format.

The hypotheses are mechanical (rule-based) drafts: the spec requires a human to
read and confirm them, which the generated file states explicitly.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils import (
    QUESTIONS_JSONL,
    RESULTS_DIR,
    SECTIONS_JSONL,
    ensure_dirs,
    get_logger,
    is_table_field,
    load_env,
    load_eval_questions,
    parse_table,
    read_jsonl,
    truncate,
    write_jsonl,
)

logger = get_logger("failure_analysis")

FAILURE_DUMP = RESULTS_DIR / "failure_dump.jsonl"
FAILURES_MD = RESULTS_DIR / "failures.md"

ABBREVIATION_HINTS = ("inr", "crcl", "egfr", "maoi", "nsaid", "qt ", "cyp", "hba1c")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_questions(path: Path | str, *, verified_only: bool) -> tuple[list[dict], dict]:
    return load_eval_questions(path, verified_only=verified_only)


def load_chunk_index(strategy: str, index_dir: Path) -> dict[str, dict]:
    path = index_dir / strategy / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found - build the index with `python -m src.indexing`")
    return {chunk["chunk_id"]: chunk for chunk in read_jsonl(path)}


def load_gold_text(question: dict, sections_path: Path | str = SECTIONS_JSONL) -> str:
    """Text of the gold section (parsed table for table questions)."""

    records = read_jsonl(sections_path)
    drug = question.get("drug")
    gold = question.get("gold_section") or ""
    if question.get("question_type") == "table":
        for record in records:
            if record["drug"] == drug and record["section"] == f"{gold}_table":
                return parse_table(record["text"]).text
    for record in records:
        if record["drug"] == drug and record["section"] == gold and not is_table_field(record["source_field"]):
            return record["text"]
    return ""


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def hit_sections(chunk_ids: list[str], chunk_index: dict[str, dict]) -> list[str]:
    return [chunk_index[chunk_id]["section"] for chunk_id in chunk_ids if chunk_id in chunk_index]


def is_retrieval_hit(question: dict, chunk_ids: list[str], chunk_index: dict[str, dict]) -> bool:
    gold = question.get("gold_section") or ""
    drug = question.get("drug", "").lower()
    for chunk_id in chunk_ids:
        chunk = chunk_index.get(chunk_id)
        if not chunk:
            continue
        if str(chunk.get("drug", "")).lower() != drug:
            continue
        if chunk.get("section") == gold or gold in (chunk.get("sections_overlapping") or []):
            return True
    return False


def draft_hypothesis(question: dict, category: str, retrieved_sections: list[str]) -> str:
    question_type = question.get("question_type", "")
    gold = question.get("gold_section") or ""
    lowered = question.get("question", "").lower()
    if category == "no_abstention":
        return (
            "The model answered an unanswerable question instead of declining. The label "
            "contains no such information, so any answer is by definition unsupported."
        )
    if category == "citation_error":
        return (
            "At least one citation did not match the gold drug/section, or pointed at a chunk "
            "id that was never retrieved (fabricated citation). The answer text may still be "
            "useful, but it is not traceable to the gold section."
        )
    if category == "unsupported_claim":
        return (
            "The LLM-as-judge marked at least one claim as unsupported by the retrieved "
            "context; the model likely generalised beyond the label text it was given."
        )
    # retrieval_miss
    if question_type == "table" and any(section == gold for section in retrieved_sections):
        return (
            "Narrative chunks from the same section outranked the parsed table, so the row "
            "that answers the question never reached the top-k."
        )
    if any(hint in lowered for hint in ABBREVIATION_HINTS):
        return (
            "Abbreviation mismatch: the question uses an acronym or short form that the "
            "label spells out (or vice versa), so lexical and embedding matching both miss."
        )
    if question_type == "negation":
        return (
            "Negation: the label states the information affirmatively somewhere else, so the "
            "retriever surfaced affirmative mentions instead of the exclusion statement."
        )
    if question_type in {"multi-section", "cross-drug"}:
        return (
            "The answer spans more than one section, but only a single gold section is scored, "
            "so the chunk that was retrieved answers a different part of the question."
        )
    return (
        "Wording mismatch between the question and the label text - the answering chunk uses "
        "different vocabulary, so it ranked below the top-k."
    )


def collect_failures(
    details: list[dict],
    questions: list[dict],
    chunk_index: dict[str, dict],
    *,
    sections_path: Path | str = SECTIONS_JSONL,
) -> list[dict]:
    questions_by_id = {question.get("id"): question for question in questions}
    failures: list[dict] = []

    for entry in details:
        question = questions_by_id.get(entry.get("question_id"))
        if question is None:
            continue
        chunk_ids = entry.get("retrieved_chunk_ids") or []
        retrieved = [
            {
                "chunk_id": chunk["chunk_id"],
                "drug": chunk["drug"],
                "section": chunk["section"],
                "is_table": chunk.get("is_table", False),
                "text": truncate(chunk["text"], 500),
            }
            for chunk_id in chunk_ids
            if (chunk := chunk_index.get(chunk_id))
        ]

        categories: list[str] = []
        if question.get("question_type") == "unanswerable":
            if entry.get("abstained") is False:
                categories.append("no_abstention")
        else:
            if not is_retrieval_hit(question, chunk_ids, chunk_index):
                categories.append("retrieval_miss")
            if not entry.get("citation_ok"):
                categories.append("citation_error")
            ratio = entry.get("faithfulness_ratio")
            if ratio is not None and ratio < 1.0:
                categories.append("unsupported_claim")

        for category in categories:
            failures.append(
                {
                    "category": category,
                    "question_id": entry.get("question_id"),
                    "question": entry.get("question"),
                    "question_type": entry.get("question_type"),
                    "drug": entry.get("drug"),
                    "gold_section": entry.get("gold_section"),
                    "answer": entry.get("answer"),
                    "citations": entry.get("citations") or [],
                    "citation_ok": entry.get("citation_ok"),
                    "keypoint_coverage": entry.get("keypoint_coverage"),
                    "faithfulness_ratio": entry.get("faithfulness_ratio"),
                    "retrieved": retrieved,
                    "retrieved_sections": hit_sections(chunk_ids, chunk_index),
                    "gold_text": truncate(load_gold_text(question, sections_path), 900),
                    "hypothesis": draft_hypothesis(
                        question, category, hit_sections(chunk_ids, chunk_index)
                    ),
                }
            )
    return failures


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def category_counts(failures: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure["category"]] = counts.get(failure["category"], 0) + 1
    return counts


def select_cases(failures: list[dict], *, max_cases: int = 5) -> list[dict]:
    """One case per category first (rarest/most informative first), then extras."""

    preferred_order = ["retrieval_miss", "citation_error", "unsupported_claim", "no_abstention"]
    chosen: list[dict] = []
    seen: set[str] = set()
    for category in preferred_order:
        for failure in failures:
            if failure["category"] != category:
                continue
            if failure["question_id"] in seen:
                continue
            chosen.append(failure)
            seen.add(failure["question_id"])
            break
    for failure in failures:
        if len(chosen) >= max_cases:
            break
        if failure["question_id"] in seen:
            continue
        chosen.append(failure)
        seen.add(failure["question_id"])
    return chosen[:max_cases]


def draft_markdown(
    failures: list[dict],
    *,
    spec_label: str,
    model: str,
    n_questions: int,
    n_answerable: int,
    n_unanswerable: int,
    max_cases: int = 5,
) -> str:
    counts = category_counts(failures)
    lines = [
        "# Failure analysis",
        "",
        f"Drafted by Codex from `results/failure_dump.jsonl` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}. "
        f"Configuration: **{spec_label}**, k=5, model `{model}`, "
        f"{n_questions} verified questions ({n_answerable} answerable, {n_unanswerable} unanswerable).",
        "",
        "> These are **hypotheses drafted automatically** from the retrieved chunks and the "
        "gold sections. A human must read and confirm them before they are treated as findings.",
        "",
        "Failure counts (a question can fail in more than one way): "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())),
        "",
    ]
    cases = select_cases(failures, max_cases=max_cases)
    if not cases:
        lines += ["No failures were detected for this configuration.", ""]
        return "\n".join(lines)

    for index, case in enumerate(cases, start=1):
        lines += [
            f"## Case {index} - {case['category'].replace('_', ' ')} ({case['question_id']}, "
            f"{case['question_type']})",
            "",
            f"**Question:** {case['question']}",
            "",
            f"**Gold section:** `{case['gold_section']}` (drug: {case['drug']})",
            "",
            "**What was retrieved:**",
            "",
        ]
        if case["retrieved"]:
            for rank, chunk in enumerate(case["retrieved"][:3], start=1):
                marker = "table" if chunk["is_table"] else "text"
                lines.append(
                    f"{rank}. `{chunk['chunk_id']}` ({chunk['drug']} | {chunk['section']} | "
                    f"{marker}) - {truncate(chunk['text'], 260)}"
                )
        else:
            lines.append("_nothing was retrieved_")
        lines += [
            "",
            "**What should have been retrieved:** "
            f"`{case['gold_section']}` - {truncate(case['gold_text'], 320) or '_gold section not found_'}",
            "",
            f"**Answer given:** {truncate(case['answer'], 400)}",
            "",
            f"**Hypothesis:** {case['hypothesis']}",
            "",
            f"_(keypoint coverage {case['keypoint_coverage']}, faithfulness {case['faithfulness_ratio']})_",
            "",
        ]
    lines += [
        "## Raw dump",
        "",
        "`results/failure_dump.jsonl` contains every failure (not just these cases) with the full "
        "retrieved chunk list, so the cases above can be re-checked or extended.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTIONS_JSONL))
    parser.add_argument("--sections", default=str(SECTIONS_JSONL))
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--details", default=None, help="per-question JSONL from evaluate_answers")
    parser.add_argument("--strategy", default="section")
    parser.add_argument("--retriever", default="hybrid_rerank")
    parser.add_argument("--drug-filter", default="auto")
    parser.add_argument("--index-dir", default=str(Path("data/index")))
    parser.add_argument("--out", default=str(FAILURE_DUMP))
    parser.add_argument("--markdown", default=str(FAILURES_MD))
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--model", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    ensure_dirs()

    details_path = (
        Path(args.details)
        if args.details
        else RESULTS_DIR
        / f"answer_details_{args.strategy}_{args.retriever}_{args.drug_filter}.jsonl"
    )
    if not details_path.exists():
        logger.error(
            "%s not found - run `python -m src.evaluate_answers --verified-only` first",
            details_path,
        )
        return 1
    details = read_jsonl(details_path)
    questions, stats = load_questions(args.questions, verified_only=args.verified_only)
    questions_by_id = {question.get("id"): question for question in questions}
    details = [entry for entry in details if entry.get("question_id") in questions_by_id]

    chunk_index = load_chunk_index(args.strategy, Path(args.index_dir))
    failures = collect_failures(details, questions, chunk_index, sections_path=args.sections)
    write_jsonl(args.out, failures)

    answerable = sum(1 for entry in details if entry.get("question_type") != "unanswerable")
    unanswerable = len(details) - answerable
    model = args.model or next((entry.get("model") for entry in details if entry.get("model")), "unknown")
    spec_label = f"{args.strategy} / {args.retriever} / {args.drug_filter}"
    markdown = draft_markdown(
        failures,
        spec_label=spec_label,
        model=str(model),
        n_questions=len(details),
        n_answerable=answerable,
        n_unanswerable=unanswerable,
        max_cases=args.max_cases,
    )
    Path(args.markdown).write_text(markdown, encoding="utf-8")

    print(f"questions evaluated : {len(details)} ({answerable} answerable, {unanswerable} unanswerable)")
    print(f"failures found      : {len(failures)}")
    for name, count in sorted(category_counts(failures).items()):
        print(f"  {name:<18} {count}")
    print(f"raw dump            : {args.out}")
    print(f"draft analysis      : {args.markdown}")
    counts = category_counts(failures)
    print()
    print("failures per question:")
    for entry in details:
        per_question = [f["category"] for f in failures if f["question_id"] == entry.get("question_id")]
        if per_question:
            print(f"  {entry.get('question_id')} ({entry.get('question_type')}): {', '.join(per_question)}")
    if not counts:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
