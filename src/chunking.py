"""Chunking strategies: ``fixed``, ``section`` and ``table_aware``.

Every chunk carries the metadata the spec requires:

    chunk_id, drug, set_id, section, strategy, text, char_start, char_end

plus a few extras used by the experiment:

    embed_text           text with the optional "Drug: x | Section: y" header
    source_section       the original section record name (may be a *_table field)
    is_table             True only for chunks built from a parsed table
    offset_scope         "drug" (fixed) or "section" (section / table_aware)
    sections_overlapping section names the chunk overlaps - used for hit-checking

Design choice (see README, Method): the ``fixed`` and ``section`` baselines use
**only** plain-text section records. ``table_aware`` adds parsed tables, each kept
as its own chunk, never split inside a row, labelled with the *parent* section so
that retrieval hit-checking works.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from src.utils import (
    PROCESSED_DIR,
    SECTIONS_JSONL,
    get_logger,
    is_table_field,
    load_env,
    normalize_text,
    parent_section_name,
    parse_table,
    read_jsonl,
    write_jsonl,
)

CHUNK_DIR = PROCESSED_DIR / "chunks"

STRATEGIES = ("fixed", "section", "table_aware")

DEFAULT_FIXED_SIZE = 800
DEFAULT_FIXED_OVERLAP = 150
DEFAULT_SECTION_MAX = 1500
DEFAULT_SECTION_OVERLAP = 200
DEFAULT_TABLE_MAX = 1500

logger = get_logger("chunking")

_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?\s")


# --------------------------------------------------------------------------- #
# Splitting helpers
# --------------------------------------------------------------------------- #
def _last_boundary(window: str) -> int | None:
    """Index just after the best break point near the end of ``window``."""

    for separator in ("\n\n", "\n"):
        index = window.rfind(separator)
        if index > 0:
            return index + len(separator)
    matches = list(_SENTENCE_END_RE.finditer(window))
    if matches:
        return matches[-1].end()
    index = window.rfind(" ")
    if index > 0:
        return index + 1
    return None


def split_text(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Sliding window over ``text`` with paragraph/sentence-aware cut points.

    Returns ``(start, end)`` character spans relative to ``text``. Cuts prefer a
    paragraph break, then a newline, then a sentence end, then a space, so chunks
    rarely start or end mid-word.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))
    total = len(text)
    if total <= size:
        return [(0, total)] if total else []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + size, total)
        if end < total:
            cut = _last_boundary(text[start:end])
            if cut is not None and cut > size // 2:
                end = start + cut
        if end <= start:
            end = min(start + size, total)
        spans.append((start, end))
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return spans


def _header(drug: str, section: str, is_table: bool = False) -> str:
    label = f"Drug: {drug} | Section: {section}"
    return f"{label} | Table content" if is_table else label


def _make_chunk(
    *,
    drug: str,
    set_id: str,
    section: str,
    source_section: str,
    strategy: str,
    text: str,
    char_start: int,
    char_end: int,
    offset_scope: str,
    is_table: bool,
    sections_overlapping: list[str],
    index: int,
    add_header: bool,
) -> dict:
    text = text.strip()
    embed_text = f"{_header(drug, section, is_table)}\n{text}" if add_header else text
    return {
        "chunk_id": f"{drug}::{strategy}::{source_section}::{index}",
        "drug": drug,
        "set_id": set_id,
        "section": section,
        "source_section": source_section,
        "strategy": strategy,
        "text": text,
        "embed_text": embed_text,
        "char_start": char_start,
        "char_end": char_end,
        "offset_scope": offset_scope,
        "is_table": is_table,
        "sections_overlapping": sections_overlapping,
        "n_chars": len(text),
    }


def _group_by_drug(sections: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in sections:
        grouped[record["drug"]].append(record)
    return grouped


def plain_sections(sections: list[dict]) -> list[dict]:
    """Plain-text section records only (no ``*_table`` records)."""

    return [record for record in sections if not is_table_field(record["source_field"])]


def table_sections(sections: list[dict]) -> list[dict]:
    return [record for record in sections if is_table_field(record["source_field"])]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def chunk_fixed(
    sections: list[dict],
    *,
    size: int = DEFAULT_FIXED_SIZE,
    overlap: int = DEFAULT_FIXED_OVERLAP,
    add_header: bool = True,
) -> list[dict]:
    """Sliding window over each drug's concatenated plain text.

    Section boundaries are deliberately ignored; each chunk's ``section`` is the
    section its first character falls in, or ``mixed`` when it straddles several.
    """

    chunks: list[dict] = []
    for drug, records in _group_by_drug(plain_sections(sections)).items():
        pieces: list[str] = []
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for record in records:
            text = record["text"]
            spans.append((record["section"], cursor, cursor + len(text)))
            pieces.append(text)
            cursor += len(text) + 2  # the "\n\n" joiner
        document = "\n\n".join(pieces)
        if not document.strip():
            continue

        for index, (start, end) in enumerate(split_text(document, size, overlap)):
            overlapping = [
                name for name, sec_start, sec_end in spans if sec_start < end and sec_end > start
            ]
            section = "mixed" if len(overlapping) != 1 else overlapping[0]
            chunks.append(
                _make_chunk(
                    drug=drug,
                    set_id=records[0]["set_id"],
                    section=section,
                    source_section=section,
                    strategy="fixed",
                    text=document[start:end],
                    char_start=start,
                    char_end=end,
                    offset_scope="drug",
                    is_table=False,
                    sections_overlapping=overlapping,
                    index=index,
                    add_header=add_header,
                )
            )
    return chunks


def chunk_section(
    sections: list[dict],
    *,
    max_chars: int = DEFAULT_SECTION_MAX,
    overlap: int = DEFAULT_SECTION_OVERLAP,
    add_header: bool = True,
    strategy: str = "section",
) -> list[dict]:
    """One chunk per plain-text section, long sections split with overlap."""

    chunks: list[dict] = []
    for record in plain_sections(sections):
        text = record["text"]
        spans = split_text(text, max_chars, overlap) or [(0, len(text))]
        for index, (start, end) in enumerate(spans):
            chunks.append(
                _make_chunk(
                    drug=record["drug"],
                    set_id=record["set_id"],
                    section=record["section"],
                    source_section=record["section"],
                    strategy=strategy,
                    text=text[start:end],
                    char_start=start,
                    char_end=end,
                    offset_scope="section",
                    is_table=False,
                    sections_overlapping=[record["section"]],
                    index=index,
                    add_header=add_header,
                )
            )
    return chunks


def _table_lines(record: dict) -> tuple[list[str], list[str], list[str]]:
    """Return ``(all_lines, prefix_lines, row_lines)`` for one table record.

    ``prefix_lines`` (caption + context such as a grouped header) are repeated at
    the top of every chunk built from this table so each piece is self-contained.
    """

    parsed = parse_table(record["text"])
    rows = [line for line in parsed.rows if line.strip()]
    if not rows:  # pragma: no cover - parse_table already falls back to plain text
        rows = [line for line in normalize_text(record["text"]).split("\n") if line.strip()]
    prefix = [line for line in [parsed.caption, *parsed.context_lines] if line.strip()]
    return [*prefix, *rows], prefix, rows


def chunk_table_aware(
    sections: list[dict],
    *,
    max_chars: int = DEFAULT_SECTION_MAX,
    overlap: int = DEFAULT_SECTION_OVERLAP,
    table_max_chars: int = DEFAULT_TABLE_MAX,
    add_header: bool = True,
) -> list[dict]:
    """Section chunks for plain text + one chunk per parsed table.

    Table rows are never split: chunks are filled line by line and a single row
    that exceeds ``table_max_chars`` still stays whole in its own chunk.
    """

    chunks = chunk_section(
        sections,
        max_chars=max_chars,
        overlap=overlap,
        add_header=add_header,
        strategy="table_aware",
    )

    for record in table_sections(sections):
        parent = parent_section_name(record["section"])
        all_lines, prefix, rows = _table_lines(record)
        table_text = "\n".join(all_lines)
        # Character offsets of each row inside the reconstructed table text.
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for line in all_lines:
            offsets.append((cursor, cursor + len(line)))
            cursor += len(line) + 1

        groups: list[list[int]] = []  # row indices per chunk
        current: list[int] = []
        current_chars = sum(len(line) + 1 for line in prefix)
        for row_index in range(len(prefix), len(all_lines)):
            line_len = len(all_lines[row_index]) + 1
            if current and current_chars + line_len > table_max_chars:
                groups.append(current)
                current = []
                current_chars = sum(len(line) + 1 for line in prefix)
            current.append(row_index)
            current_chars += line_len
        if current:
            groups.append(current)

        for index, group in enumerate(groups):
            body = [all_lines[row_index] for row_index in group]
            text = "\n".join([*prefix, *body])
            start = offsets[group[0]][0] if group else 0
            end = offsets[group[-1]][1] if group else len(table_text)
            chunks.append(
                _make_chunk(
                    drug=record["drug"],
                    set_id=record["set_id"],
                    section=parent,
                    source_section=record["section"],
                    strategy="table_aware",
                    text=text,
                    char_start=start,
                    char_end=end,
                    offset_scope="section",
                    is_table=True,
                    sections_overlapping=[parent],
                    index=index,
                    add_header=add_header,
                )
            )
    return chunks


BUILDERS = {
    "fixed": chunk_fixed,
    "section": chunk_section,
    "table_aware": chunk_table_aware,
}


def build_chunks(sections: list[dict], strategy: str, **kwargs) -> list[dict]:
    if strategy not in BUILDERS:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[strategy](sections, **kwargs)


def build_all(sections: list[dict], **kwargs) -> dict[str, list[dict]]:
    return {strategy: build_chunks(sections, strategy, **kwargs) for strategy in STRATEGIES}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    parser.add_argument("--sections", default=str(SECTIONS_JSONL))
    parser.add_argument("--out-dir", default=str(CHUNK_DIR))
    parser.add_argument("--fixed-size", type=int, default=DEFAULT_FIXED_SIZE)
    parser.add_argument("--fixed-overlap", type=int, default=DEFAULT_FIXED_OVERLAP)
    parser.add_argument("--section-max", type=int, default=DEFAULT_SECTION_MAX)
    parser.add_argument("--section-overlap", type=int, default=DEFAULT_SECTION_OVERLAP)
    parser.add_argument("--table-max", type=int, default=DEFAULT_TABLE_MAX)
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="do not prepend 'Drug: x | Section: y' to embed_text (ablation)",
    )
    parser.add_argument("--show", type=int, default=2, help="sample chunks to print")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    sections = read_jsonl(args.sections)
    if not sections:
        logger.error("no sections at %s - run `python -m src.parse_labels` first", args.sections)
        return 1

    from pathlib import Path

    out_dir = Path(args.out_dir)
    add_header = not args.no_header
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    print(f"{'strategy':<13} {'chunks':>7} {'table':>7} {'mean_chars':>11} {'max_chars':>10}")
    print("-" * 54)
    for strategy in strategies:
        chunks = build_chunks(
            sections,
            strategy,
            add_header=add_header,
            **(
                {
                    "fixed": {"size": args.fixed_size, "overlap": args.fixed_overlap},
                    "section": {"max_chars": args.section_max, "overlap": args.section_overlap},
                    "table_aware": {
                        "max_chars": args.section_max,
                        "overlap": args.section_overlap,
                        "table_max_chars": args.table_max,
                    },
                }[strategy]
            ),
        )
        path = out_dir / f"{strategy}.jsonl"
        write_jsonl(path, chunks)
        n_table = sum(1 for chunk in chunks if chunk["is_table"])
        mean_chars = sum(chunk["n_chars"] for chunk in chunks) / max(1, len(chunks))
        max_chars = max((chunk["n_chars"] for chunk in chunks), default=0)
        print(f"{strategy:<13} {len(chunks):>7} {n_table:>7} {mean_chars:>11.0f} {max_chars:>10}")
        logger.info("wrote %d chunks to %s", len(chunks), path)
        if args.show:
            for chunk in chunks[: args.show]:
                print("-" * 54)
                print(
                    f"  {chunk['chunk_id']} | section={chunk['section']} "
                    f"| table={chunk['is_table']} | chars={chunk['n_chars']}"
                )
                print("  " + chunk["text"][:300].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
