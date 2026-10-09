"""Shared citation parser: grouped ids, and the same rule for baseline and agent."""

from __future__ import annotations

from agent.llm import LLMResult
from agent.llm.compat import CompatClient
from agent.llm.mock import ScriptedAdapter
from eval.baseline import answer_baseline
from eval.citations import cited_chunk_ids, normalize_citations, parse_citations, split_ids

A, B = "warfarin::section::contraindications::0", "warfarin::section::contraindications::1"


def test_single_citation_matches_src_format() -> None:
    assert parse_citations(f"x [warfarin | contraindications | {A}]") == [
        {"drug": "warfarin", "section": "contraindications", "chunk_id": A}
    ]


def test_grouped_ids_are_split_into_one_citation_each() -> None:
    text = f"x [warfarin | contraindications | {A}, {B}] y [warfarin | contraindications | {B}; {A}]"
    assert cited_chunk_ids(text) == [A, B]
    assert len(parse_citations(text)) == 2  # duplicates across brackets collapse


def test_split_and_normalize_handle_structured_citations() -> None:
    assert split_ids(f" {A} ,, {B} ") == [A, B]
    assert normalize_citations([{"drug": "warfarin", "section": "contraindications", "chunk_id": f"{A}, {B}"}]) == [
        {"drug": "warfarin", "section": "contraindications", "chunk_id": A},
        {"drug": "warfarin", "section": "contraindications", "chunk_id": B},
    ]


def test_copied_context_labels_are_stripped() -> None:
    # Seen live from gemini-3.8-flash on 2026-10-09.
    text = f"x [warfarin | section=contraindications | chunk_id={A}] y [drug=warfarin | contraindications | {B}]"
    assert parse_citations(text) == [
        {"drug": "warfarin", "section": "contraindications", "chunk_id": A},
        {"drug": "warfarin", "section": "contraindications", "chunk_id": B},
    ]


def test_no_citations_gives_empty_list() -> None:
    assert parse_citations("") == [] and parse_citations("no brackets here") == []


class _Retriever:
    strategy, name = "section", "stub"

    def retrieve(self, query, k=5, drug_filter=None):
        return [
            {"chunk_id": cid, "drug": "warfarin", "section": "contraindications", "text": "t", "score": 1.0}
            for cid in (A, B)
        ]


def test_baseline_uses_the_shared_parser_for_grouped_ids() -> None:
    llm = ScriptedAdapter([LLMResult(text=f"Pregnancy and bleeding [warfarin | contraindications | {A}, {B}].")])
    result = answer_baseline("q", _Retriever(), client=CompatClient(llm), k=2)
    assert [c["chunk_id"] for c in result.citations] == [A, B]
