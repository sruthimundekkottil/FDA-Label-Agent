"""Interactive review of the drafted evaluation questions.

    python -m src.review_questions                 # review data/eval/questions.jsonl
    python -m src.review_questions --limit 5       # stop after 5 decisions
    python -m src.review_questions --questions tmp/questions.jsonl

For every question that still has ``"verified": false`` the tool prints the
question, the gold drug/section, the question type, the gold keypoints and a
preview of the gold label text (for table questions: the parsed table rows).
It then asks for one of:

    [v]erify   set "verified": true
    [d]elete   remove the record from the file
    [s]kip     leave it as it is and move on
    [q]uit     save and stop

The file is saved after **every** action, so a crash or Ctrl-C never loses a
decision, and a backup copy of the question file is written before the first
edit (``data/eval/questions.backup.jsonl`` by default, gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.utils import (
    QUESTIONS_JSONL,
    SECTIONS_JSONL,
    get_logger,
    is_table_field,
    normalize_text,
    parse_table,
    read_jsonl,
    write_jsonl,
)

logger = get_logger("review_questions")

DEFAULT_BACKUP = QUESTIONS_JSONL.with_name("questions.backup.jsonl")
DEFAULT_PREVIEW_CHARS = 1500

VERIFY_ANSWERS = {"v", "verify", "y", "yes"}
DELETE_ANSWERS = {"d", "delete", "x"}
SKIP_ANSWERS = {"s", "skip", ""}
QUIT_ANSWERS = {"q", "quit", "exit"}


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def save_records(path: Path | str, records: list[dict]) -> None:
    """Write the question file atomically (temp file + replace)."""

    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    write_jsonl(temp, records)
    os.replace(temp, path)


def make_backup(path: Path | str, backup_path: Path | str | None = None) -> Path:
    """Copy the question file before it is edited. Returns the backup path."""

    path = Path(path)
    backup = (
        Path(backup_path)
        if backup_path
        else path.with_name(f"{path.stem}.backup{path.suffix}")
    )
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(path.read_bytes())
    return backup


def load_section_index(
    sections_path: Path | str = SECTIONS_JSONL,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """Return ``(plain_sections, tables)`` keyed by ``(drug, section)``."""

    plain: dict[tuple[str, str], dict] = {}
    tables: dict[tuple[str, str], dict] = {}
    for record in read_jsonl(sections_path):
        key = (record["drug"], record["section"])
        if is_table_field(record["source_field"]):
            tables[key] = record
        else:
            plain[key] = record
    return plain, tables


def clip_text(text: str, max_chars: int) -> str:
    """Clip to ``max_chars`` without cutting a table row in half."""

    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    newline = clipped.rfind("\n")
    if newline > max_chars * 0.6:
        clipped = clipped[:newline]
    return clipped.rstrip() + "\n… (truncated)"


def section_preview(
    question: dict,
    plain: dict[tuple[str, str], dict],
    tables: dict[tuple[str, str], dict],
    *,
    max_chars: int = DEFAULT_PREVIEW_CHARS,
) -> tuple[str, str, int]:
    """``(source_label, preview_text, full_length)`` for one question.

    Table questions show the parsed table (rows with column labels), everything
    else shows the plain section text.
    """

    drug = question.get("drug", "")
    gold = question.get("gold_section") or ""
    table_key = (drug, f"{gold}_table")

    is_table = question.get("question_type") == "table"
    if is_table and table_key in tables:
        record = tables[table_key]
        text = parse_table(record["text"]).text
        label = (
            f"{drug} | {record['section']} (parsed table, parent section: {gold or '-'}) "
            f"| set_id={record.get('set_id', '?')} | effective_time={record.get('effective_time', '?')}"
        )
        return label, clip_text(text, max_chars), len(text)

    record = plain.get((drug, gold))
    if record is None:
        # fall back to any table of this drug when the gold section has no match
        candidates = [key for key in tables if key[0] == drug]
        if candidates and is_table:
            record = tables[candidates[0]]
            text = parse_table(record["text"]).text
            label = f"{drug} | {record['section']} (parsed table) | set_id={record.get('set_id', '?')}"
            return label, clip_text(text, max_chars), len(text)
        return f"{drug} | {gold or '(no gold section)'} (not found in sections.jsonl)", "", 0

    label = (
        f"{drug} | {gold} | set_id={record.get('set_id', '?')} "
        f"| effective_time={record.get('effective_time', '?')}"
    )
    text = normalize_text(record["text"])
    return label, clip_text(text, max_chars), len(text)


def counts(records: list[dict]) -> tuple[int, int, int]:
    verified = sum(1 for record in records if record.get("verified"))
    return verified, len(records) - verified, len(records)


# --------------------------------------------------------------------------- #
# Review loop
# --------------------------------------------------------------------------- #
def format_header(index: int, total: int, question: dict) -> str:
    return (
        f"[{index + 1}/{total}] {question.get('question_type', '?')} | "
        f"{question.get('drug', '?')} | {question.get('gold_section') or '(no gold section)'}"
    )


def format_block(
    index: int,
    records: list[dict],
    question: dict,
    plain: dict,
    tables: dict,
    max_chars: int,
    verified_so_far: int,
) -> str:
    label, preview, full_length = section_preview(question, plain, tables, max_chars=max_chars)
    lines = [
        "",
        "=" * 78,
        format_header(index, len(records), question),
        f"id: {question.get('id', '?')} | verified: {bool(question.get('verified'))}",
        "-" * 78,
        f"Q: {question.get('question', '')}",
        "",
        "Gold keypoints:",
    ]
    keypoints = question.get("gold_answer_keypoints") or []
    if keypoints:
        for number, point in enumerate(keypoints, start=1):
            lines.append(f"  {number}. {point}")
    else:
        lines.append("  (none - expected answer is abstention)")

    shown = min(len(preview), max_chars)
    lines += [
        "",
        f"Gold label text: {label}",
        f"  (showing {shown} of {full_length} characters)",
    ]
    if preview:
        lines += ["  " + line for line in preview.split("\n")]
    else:
        lines.append("  (no matching text found)")
    lines += [
        "-" * 78,
        f"verified so far: {verified_so_far}/{len(records)}",
    ]
    return "\n".join(lines)


def run_review(
    records: list[dict],
    plain: dict,
    tables: dict,
    *,
    save_fn,
    input_fn=input,
    print_fn=print,
    limit: int | None = None,
    max_chars: int = DEFAULT_PREVIEW_CHARS,
) -> dict:
    """Review unverified questions one by one; returns a small summary dict."""

    decisions = {"verified": 0, "deleted": 0, "skipped": 0}
    index = 0
    while index < len(records):
        question = records[index]
        if question.get("verified"):
            index += 1
            continue
        if limit is not None and sum(decisions.values()) >= limit:
            print_fn(f"\nreached --limit {limit}; stopping")
            break

        verified_so_far, _, _ = counts(records)
        print_fn(format_block(index, records, question, plain, tables, max_chars, verified_so_far))

        try:
            answer = input_fn("[v]erify  [d]elete  [s]kip  [q]uit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print_fn("\ninput closed - saving and stopping")
            break

        if answer in VERIFY_ANSWERS:
            records[index]["verified"] = True
            save_fn(records)
            decisions["verified"] += 1
            verified_now, _, total = counts(records)
            print_fn(f"  -> verified. verified so far: {verified_now}/{total}")
            index += 1
        elif answer in DELETE_ANSWERS:
            removed = records.pop(index)
            save_fn(records)
            decisions["deleted"] += 1
            verified_now, _, total = counts(records)
            print_fn(
                f"  -> deleted {removed.get('id')}. verified so far: {verified_now}/{total} "
                f"(restore from the backup if this was a mistake)"
            )
        elif answer in SKIP_ANSWERS:
            decisions["skipped"] += 1
            print_fn("  -> skipped (unchanged)")
            index += 1
        elif answer in QUIT_ANSWERS:
            print_fn("  -> quitting")
            break
        else:
            print_fn("  -> please answer v, d, s or q")

    decisions["verified_total"], decisions["unverified_total"], decisions["total"] = counts(records)
    return decisions


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTIONS_JSONL))
    parser.add_argument("--sections", default=str(SECTIONS_JSONL))
    parser.add_argument(
        "--backup",
        default=None,
        help=f"backup file (default: {DEFAULT_BACKUP.name} next to the question file)",
    )
    parser.add_argument("--no-backup", action="store_true", help="skip the backup copy")
    parser.add_argument("--limit", type=int, default=None, help="stop after N decisions")
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:  # keep table glyphs (≥, †, …) readable in Windows terminals
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - older/odd stdout objects
        pass
    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("no question file at %s", questions_path)
        return 1

    records = read_jsonl(questions_path)
    if not records:
        logger.error("no questions in %s", questions_path)
        return 1

    backup_path: Path | None = None
    if not args.no_backup:
        backup_path = make_backup(questions_path, args.backup)
        print(f"backup written to {backup_path}")

    plain, tables = load_section_index(args.sections)
    verified, unverified, total = counts(records)
    print(
        f"reviewing {questions_path}\n"
        f"{total} questions | verified: {verified} | unverified: {unverified}\n"
        f"decisions are saved after every action"
    )

    decisions = run_review(
        records,
        plain,
        tables,
        save_fn=lambda items: save_records(questions_path, items),
        limit=args.limit,
        max_chars=args.preview_chars,
    )

    print()
    print("-" * 78)
    print(
        f"this session: verified {decisions['verified']}, deleted {decisions['deleted']}, "
        f"skipped {decisions['skipped']}"
    )
    print(
        f"file now: {decisions['verified_total']}/{decisions['total']} verified, "
        f"{decisions['unverified_total']} still unverified"
    )
    if decisions["deleted"] and backup_path is not None:
        print(f"deleted records can be restored from {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
