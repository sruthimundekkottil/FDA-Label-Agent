"""Retrieval tests: fusion, BM25, drug filtering and the shared interface.

These tests never download a model: BM25 is built in memory and the vector side
is replaced by a deterministic stub, so ``pytest`` stays fast and offline.
"""

from __future__ import annotations

from src.retrieval import (
    BM25Retriever,
    HybridRetriever,
    _apply_drug_filter,
    reciprocal_rank_fusion,
)
from src.utils import detect_drug_in_query


def _chunk(chunk_id: str, drug: str, section: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "drug": drug,
        "section": section,
        "source_section": section,
        "strategy": "section",
        "text": text,
        "embed_text": f"Drug: {drug} | Section: {section}\n{text}",
        "char_start": 0,
        "char_end": len(text),
        "is_table": False,
    }


CHUNKS = [
    _chunk("c1", "warfarin", "contraindications", "Warfarin is contraindicated in pregnancy."),
    _chunk("c2", "warfarin", "dosage_and_administration", "The usual dose of warfarin is 2 to 5 mg daily."),
    _chunk("c3", "apixaban", "contraindications", "Apixaban is contraindicated in active bleeding."),
    _chunk("c4", "sertraline", "adverse_reactions", "Nausea was reported by 26% of patients."),
]


class _StubVector:
    """Returns a fixed ranking, ignoring the query."""

    name = "vector"

    def __init__(self, order: list[str]):
        self.order = order

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        by_id = {chunk["chunk_id"]: chunk for chunk in CHUNKS}
        hits = [dict(by_id[chunk_id], score=1.0 / (index + 1), retriever="vector")
                for index, chunk_id in enumerate(self.order)]
        return _apply_drug_filter(hits, drug_filter, k)


def test_reciprocal_rank_fusion_prefers_documents_in_both_rankings() -> None:
    first = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    second = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    fused = reciprocal_rank_fusion([first, second], rrf_k=60, top_k=3)
    assert [chunk["chunk_id"] for chunk in fused] == ["b", "a", "c"]
    # 1/(60+2) + 1/(60+1) for b
    assert fused[0]["score"] == round(1 / 62 + 1 / 61, 6)


def test_reciprocal_rank_fusion_respects_top_k() -> None:
    fused = reciprocal_rank_fusion([[{"chunk_id": str(i)} for i in range(10)]], top_k=4)
    assert len(fused) == 4


def test_bm25_ranks_lexical_match_first() -> None:
    retriever = BM25Retriever.from_chunks(CHUNKS)
    hits = retriever.retrieve("warfarin contraindicated pregnancy", k=2)
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["retriever"] == "bm25"


def test_drug_filter_restricts_results() -> None:
    retriever = BM25Retriever.from_chunks(CHUNKS)
    hits = retriever.retrieve("contraindicated", k=5, drug_filter="apixaban")
    assert [chunk["chunk_id"] for chunk in hits] == ["c3"]


def test_apply_drug_filter_is_case_insensitive() -> None:
    hits = _apply_drug_filter([dict(chunk) for chunk in CHUNKS], "WARFARIN", 5)
    assert {chunk["drug"] for chunk in hits} == {"warfarin"}


def test_hybrid_fuses_vector_and_bm25() -> None:
    vector = _StubVector(["c2", "c1", "c3", "c4"])
    hybrid = HybridRetriever(
        "memory", vector=vector, bm25=BM25Retriever.from_chunks(CHUNKS), candidate_k=4
    )
    hits = hybrid.retrieve("warfarin contraindicated pregnancy", k=2)
    ids = [chunk["chunk_id"] for chunk in hits]
    assert ids[0] == "c1"  # BM25 first, vector second -> wins the fusion
    assert hits[0]["retriever"] == "hybrid"


def test_empty_query_returns_nothing() -> None:
    retriever = BM25Retriever.from_chunks(CHUNKS)
    assert retriever.retrieve("   ", k=3) == []


def test_auto_detect_drug_picks_longest_match() -> None:
    drugs = ["insulin", "insulin glargine", "metformin"]
    assert detect_drug_in_query("What is the dose of insulin glargine?", drugs) == "insulin glargine"
    assert detect_drug_in_query("What is the dose of metformin?", drugs) == "metformin"
    assert detect_drug_in_query("What is the dose?", drugs) is None
