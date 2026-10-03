"""Tests for the three chunking strategies.

Everything here runs on a small synthetic label so the tests stay fast and
deterministic; the invariants are exactly the ones the experiment relies on.
"""

from __future__ import annotations

import pytest

from src.chunking import (
    STRATEGIES,
    build_chunks,
    chunk_fixed,
    chunk_section,
    chunk_table_aware,
    split_text,
)
from src.utils import parent_section_name, parse_table


def _section(drug: str, section: str, text: str, source_field: str | None = None) -> dict:
    return {
        "drug": drug,
        "set_id": "set-1",
        "section": section,
        "text": text,
        "source_field": source_field or section,
        "effective_time": "20250101",
    }


ADVERSE_TABLE_HTML = (
    "<table><caption>Table 1: Adverse reactions (sertraline)</caption>"
    "<tr><td></td><td>Sertraline (N=3066) %</td><td>Placebo (N=2293) %</td></tr>"
    "<tr><td>Cardiac disorders</td><td></td><td></td></tr>"
    "<tr><td>Palpitations</td><td>4</td><td>2</td></tr>"
    "<tr><td>Nausea</td><td>26</td><td>12</td></tr>"
    "</table>"
)


@pytest.fixture()
def sections() -> list[dict]:
    long_text = "\n\n".join(
        f"Paragraph {index}: " + ("warning text about pregnancy and bleeding. " * 12)
        for index in range(12)
    )
    return [
        _section(
            "metformin",
            "indications_and_usage",
            "Metformin is indicated for type 2 diabetes. " * 12,
        ),
        _section("metformin", "warnings_and_cautions", long_text),
        _section(
            "sertraline",
            "adverse_reactions_table",
            ADVERSE_TABLE_HTML,
            source_field="adverse_reactions_table",
        ),
        _section(
            "sertraline",
            "adverse_reactions",
            "The following adverse reactions were reported. " * 30,
        ),
    ]


def _all_chunks(sections: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for strategy in STRATEGIES:
        chunks.extend(build_chunks(sections, strategy))
    return chunks


def test_no_empty_chunks(sections: list[dict]) -> None:
    for chunk in _all_chunks(sections):
        assert chunk["text"].strip(), chunk["chunk_id"]
        assert chunk["n_chars"] == len(chunk["text"])
        assert chunk["embed_text"].strip()


def test_required_metadata_present(sections: list[dict]) -> None:
    required = {
        "chunk_id",
        "drug",
        "set_id",
        "section",
        "strategy",
        "text",
        "char_start",
        "char_end",
    }
    for chunk in _all_chunks(sections):
        assert required.issubset(chunk)
        assert chunk["char_end"] >= chunk["char_start"]


def test_no_strategy_contains_markup(sections: list[dict]) -> None:
    """No chunk may contain raw table markup, in any strategy."""

    for chunk in _all_chunks(sections):
        lowered = chunk["text"].lower()
        for marker in ("<table", "<td", "<tr", "<item", "</"):
            assert marker not in lowered, (chunk["chunk_id"], marker)


def test_fixed_overlaps_and_stays_within_size(sections: list[dict]) -> None:
    chunks = [
        chunk
        for chunk in chunk_fixed(sections, size=300, overlap=60)
        if chunk["drug"] == "metformin"
    ]
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk["n_chars"] <= 300
    for previous, current in zip(chunks, chunks[1:]):
        assert current["char_start"] < previous["char_end"], "fixed chunks must overlap"


def test_fixed_keeps_section_metadata(sections: list[dict]) -> None:
    known = {
        "indications_and_usage",
        "warnings_and_cautions",
        "adverse_reactions",
        "mixed",
    }
    for chunk in chunk_fixed(sections, size=300, overlap=60):
        assert chunk["section"] in known
        assert chunk["sections_overlapping"], chunk["chunk_id"]
        if chunk["section"] != "mixed":
            assert chunk["section"] in chunk["sections_overlapping"]


def test_section_strategy_is_one_section_per_chunk(sections: list[dict]) -> None:
    for chunk in chunk_section(sections, max_chars=400, overlap=80):
        record = next(
            record
            for record in sections
            if record["drug"] == chunk["drug"] and record["section"] == chunk["source_section"]
        )
        assert chunk["text"] in record["text"]
        assert chunk["sections_overlapping"] == [chunk["section"]]
        assert chunk["strategy"] == "section"


def test_section_splitting_overlaps(sections: list[dict]) -> None:
    chunks = [
        chunk
        for chunk in chunk_section(sections, max_chars=400, overlap=80)
        if chunk["source_section"] == "warnings_and_cautions"
    ]
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current["char_start"] < previous["char_end"]


def test_split_text_snaps_to_boundaries() -> None:
    text = "First sentence. Second sentence.\n\nThird paragraph starts here and continues."
    spans = split_text(text, size=30, overlap=5)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for start, end in spans:
        assert 0 <= start < end <= len(text)


def test_baselines_exclude_tables(sections: list[dict]) -> None:
    """fixed/section use plain text only; table_aware adds the parsed tables."""

    table_marker = "Table 1: Adverse reactions"
    for strategy in ("fixed", "section"):
        chunks = build_chunks(sections, strategy)
        assert chunks, strategy
        assert not any(chunk["is_table"] for chunk in chunks)
        assert not any(table_marker in chunk["text"] for chunk in chunks)
        assert not any(chunk["source_section"].endswith("_table") for chunk in chunks)

    aware = build_chunks(sections, "table_aware")
    table_chunks = [chunk for chunk in aware if chunk["is_table"]]
    assert table_chunks
    assert any(table_marker in chunk["text"] for chunk in table_chunks)


def test_table_chunks_use_parent_section(sections: list[dict]) -> None:
    for chunk in build_chunks(sections, "table_aware"):
        if not chunk["is_table"]:
            continue
        assert chunk["section"] == "adverse_reactions"
        assert chunk["source_section"] == "adverse_reactions_table"
        assert chunk["sections_overlapping"] == ["adverse_reactions"]
        assert parent_section_name(chunk["source_section"]) == chunk["section"]


def test_table_rows_are_never_split(sections: list[dict]) -> None:
    """Every source row must appear whole, in order, in exactly one chunk."""

    record = next(r for r in sections if r["source_field"] == "adverse_reactions_table")
    parsed = parse_table(record["text"])
    expected_rows = parsed.rows
    assert expected_rows

    chunks = [c for c in chunk_table_aware(sections, table_max_chars=120) if c["is_table"]]
    assert len(chunks) >= 2, "the fixture table should split into several chunks"

    seen: list[str] = []
    for chunk in chunks:
        for line in chunk["text"].split("\n"):
            if line in expected_rows:
                seen.append(line)
            else:
                # only the repeated caption/context prefix may appear as a non-row
                assert "Table 1" in line or line in parsed.context_lines
    assert seen == expected_rows, "rows must appear complete and in order"


def test_table_chunk_keeps_prefix_for_context(sections: list[dict]) -> None:
    chunks = [c for c in chunk_table_aware(sections, table_max_chars=120) if c["is_table"]]
    assert all(c["text"].startswith("Table 1: Adverse reactions") for c in chunks)


def test_header_flag_is_toggleable(sections: list[dict]) -> None:
    with_header = chunk_section(sections, max_chars=400, add_header=True)[0]
    without_header = chunk_section(sections, max_chars=400, add_header=False)[0]
    assert with_header["embed_text"].startswith("Drug: ")
    assert without_header["embed_text"] == without_header["text"]


def test_single_giant_row_stays_whole() -> None:
    giant = "x" * 4000
    record = _section(
        "drugx",
        "adverse_reactions_table",
        f"<table><tr><td>{giant}</td><td>2</td></tr>"
        "<tr><td>Nausea</td><td>3</td></tr><tr><td>Headache</td><td>4</td></tr></table>",
        source_field="adverse_reactions_table",
    )
    chunks = [c for c in chunk_table_aware([record], table_max_chars=500) if c["is_table"]]
    big = [c for c in chunks if giant[:50] in c["text"]]
    assert len(big) == 1
    assert big[0]["n_chars"] > 500
