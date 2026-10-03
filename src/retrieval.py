"""Retrieval strategies behind one interface: vector, bm25, hybrid, hybrid_rerank.

    retriever.retrieve(query, k=5, drug_filter="warfarin")

* ``vector``       FAISS cosine similarity over sentence-transformers embeddings
* ``bm25``         rank_bm25 over lowercased alphanumeric tokens
* ``hybrid``       Reciprocal Rank Fusion of the two rankings (k = 60)
* ``hybrid_rerank``  optional cross-encoder reranking of the hybrid top 20

Query instruction: bge models expect a short instruction in front of the query
(``QUERY_INSTRUCTION`` below); it is applied to queries only, never to documents.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.indexing import DEFAULT_EMBED_MODEL, index_dir, load_sentence_transformer
from src.chunking import STRATEGIES
from src.utils import (
    DRUGS_FILE,
    INDEX_DIR,
    detect_drug_in_query,
    get_logger,
    load_drug_list,
    load_env,
    read_json,
    read_jsonl,
    tokenize,
    truncate,
)

RETRIEVERS = ("vector", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60
DEFAULT_CANDIDATES = 20

# bge-small/en-v1.5 is trained with this instruction on the query side.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

logger = get_logger("retrieval")


class IndexMissingError(RuntimeError):
    """Raised when an index has not been built yet."""


def _require(path: Path, strategy: str) -> Path:
    if not path.exists():
        raise IndexMissingError(
            f"missing {path} - build the index first: python -m src.indexing --strategy {strategy}"
        )
    return path


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    *,
    rrf_k: int = RRF_K,
    top_k: int = 5,
) -> list[dict]:
    """Fuse ranked lists with ``score = sum(1 / (rrf_k + rank))`` (rank starts at 1)."""

    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = chunk["chunk_id"]
            chunks.setdefault(key, chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    fused: list[dict] = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        merged = dict(chunks[key])
        merged["score"] = round(score, 6)
        merged["retriever_score"] = score
        fused.append(merged)
    return fused


def _apply_drug_filter(chunks: list[dict], drug_filter: str | None, k: int) -> list[dict]:
    if drug_filter:
        wanted = drug_filter.strip().lower()
        chunks = [chunk for chunk in chunks if str(chunk.get("drug", "")).lower() == wanted]
    return chunks[:k]


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #
class VectorRetriever:
    """FAISS inner-product search over normalised embeddings."""

    name = "vector"

    def __init__(self, strategy: str, *, base_dir: Path | None = None, model_name: str | None = None):
        import faiss  # noqa: F401  (import error should surface here, not later)

        self.strategy = strategy
        self.dir = index_dir(strategy, base_dir)
        self.meta = read_json(_require(self.dir / "meta.json", strategy))
        self.chunks = read_jsonl(_require(self.dir / "chunks.jsonl", strategy))
        self._faiss = faiss
        self.index = faiss.read_index(str(_require(self.dir / "index.faiss", strategy)))
        self.model_name = model_name or self.meta.get("embed_model", DEFAULT_EMBED_MODEL)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_sentence_transformer(self.model_name)
        return self._model

    def _encode_query(self, query: str):
        text = QUERY_INSTRUCTION + query if "bge" in self.model_name.lower() else query
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vector.astype("float32")

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        import numpy as np

        if not query.strip():
            return []
        n_probe = self.index.ntotal if drug_filter else min(self.index.ntotal, max(k * 20, 100))
        scores, indices = self.index.search(self._encode_query(query), n_probe)
        results: list[dict] = []
        for score, index in zip(np.ravel(scores), np.ravel(indices)):
            if index < 0:
                continue
            chunk = dict(self.chunks[int(index)])
            chunk["score"] = float(score)
            chunk["retriever"] = self.name
            results.append(chunk)
        return _apply_drug_filter(results, drug_filter, k)


class BM25Retriever:
    """Okapi BM25 over lowercased alphanumeric tokens."""

    name = "bm25"

    def __init__(self, strategy: str, *, base_dir: Path | None = None):
        from rank_bm25 import BM25Okapi

        self.strategy = strategy
        self.dir = index_dir(strategy, base_dir)
        self.meta = read_json(_require(self.dir / "meta.json", strategy))
        self.chunks = read_jsonl(_require(self.dir / "chunks.jsonl", strategy))
        self.tokens = read_jsonl(_require(self.dir / "tokens.jsonl", strategy))
        self.bm25 = BM25Okapi(self.tokens)
        self._BM25Okapi = BM25Okapi

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "BM25Retriever":
        """In-memory index - handy for tests and for the Streamlit fallback."""

        instance = cls.__new__(cls)
        from rank_bm25 import BM25Okapi

        instance.strategy = chunks[0].get("strategy", "memory") if chunks else "memory"
        instance.dir = None
        instance.meta = {"n_chunks": len(chunks)}
        instance.chunks = chunks
        instance.tokens = [
            tokenize(chunk.get("embed_text") or chunk["text"]) for chunk in chunks
        ]
        instance.bm25 = BM25Okapi(instance.tokens)
        instance._BM25Okapi = BM25Okapi
        return instance

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        if not query.strip():
            return []
        scores = list(self.bm25.get_scores(tokenize(query)))
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: list[dict] = []
        for index in order:
            if scores[index] <= 0 and not drug_filter:
                break
            chunk = dict(self.chunks[index])
            chunk["score"] = float(scores[index])
            chunk["retriever"] = self.name
            results.append(chunk)
            if len(results) >= max(k * 20, 100):
                break
        return _apply_drug_filter(results, drug_filter, k)


class HybridRetriever:
    """RRF of vector and BM25 rankings."""

    name = "hybrid"

    def __init__(
        self,
        strategy: str,
        *,
        base_dir: Path | None = None,
        vector: VectorRetriever | None = None,
        bm25: BM25Retriever | None = None,
        candidate_k: int = DEFAULT_CANDIDATES,
        rrf_k: int = RRF_K,
    ):
        self.strategy = strategy
        self.vector = vector or VectorRetriever(strategy, base_dir=base_dir)
        self.bm25 = bm25 or BM25Retriever(strategy, base_dir=base_dir)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        candidates = max(self.candidate_k, k)
        vector_hits = self.vector.retrieve(query, candidates, drug_filter)
        bm25_hits = self.bm25.retrieve(query, candidates, drug_filter)
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], rrf_k=self.rrf_k, top_k=k)
        for chunk in fused:
            chunk["retriever"] = self.name
        return fused


class RerankRetriever:
    """Cross-encoder reranking of the hybrid candidate list (optional)."""

    name = "hybrid_rerank"

    def __init__(
        self,
        strategy: str,
        *,
        base_dir: Path | None = None,
        hybrid: HybridRetriever | None = None,
        model_name: str | None = None,
        candidate_k: int = DEFAULT_CANDIDATES,
    ):
        import os

        self.strategy = strategy
        self.hybrid = hybrid or HybridRetriever(strategy, base_dir=base_dir)
        self.model_name = (
            model_name
            or os.environ.get("RERANK_MODEL")
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.candidate_k = candidate_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from src.indexing import model_cache_folder, FALLBACK_EMBED_MODEL  # noqa: F401

            folder = model_cache_folder()
            try:
                self._model = CrossEncoder(
                    self.model_name, cache_folder=str(folder), local_files_only=True
                )
            except Exception:
                self._model = CrossEncoder(self.model_name, cache_folder=str(folder))
        return self._model

    def retrieve(self, query: str, k: int = 5, drug_filter: str | None = None) -> list[dict]:
        candidates = self.hybrid.retrieve(query, max(self.candidate_k, k), drug_filter)
        if not candidates:
            return []
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        results: list[dict] = []
        for chunk, score in ranked[:k]:
            item = dict(chunk)
            item["score"] = float(score)
            item["retriever"] = self.name
            item["rrf_score"] = chunk.get("score")
            results.append(item)
        return results


def load_retriever(
    strategy: str,
    kind: str = "hybrid",
    *,
    base_dir: Path | None = None,
    **kwargs,
):
    """Factory used by the evaluation scripts and the Streamlit app."""

    if kind == "vector":
        return VectorRetriever(strategy, base_dir=base_dir, **kwargs)
    if kind == "bm25":
        return BM25Retriever(strategy, base_dir=base_dir, **kwargs)
    if kind == "hybrid":
        return HybridRetriever(strategy, base_dir=base_dir, **kwargs)
    if kind == "hybrid_rerank":
        return RerankRetriever(strategy, base_dir=base_dir, **kwargs)
    raise ValueError(f"unknown retriever {kind!r}; expected one of {RETRIEVERS}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="section", choices=list(STRATEGIES))
    parser.add_argument("--retriever", default="hybrid", choices=list(RETRIEVERS))
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drug", default=None, help="restrict to one drug")
    parser.add_argument(
        "--auto-drug",
        action="store_true",
        help="detect the drug name in the query and use it as a filter",
    )
    parser.add_argument("--base-dir", default=str(INDEX_DIR))
    parser.add_argument("--show-text", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()

    drug_filter = args.drug
    if args.auto_drug and not drug_filter:
        drug_filter = detect_drug_in_query(args.query, load_drug_list())
        if drug_filter:
            print(f"[auto] detected drug filter: {drug_filter}")
        else:
            print("[auto] no drug name detected in the query")

    started = time.perf_counter()
    retriever = load_retriever(args.strategy, args.retriever, base_dir=Path(args.base_dir))
    hits = retriever.retrieve(args.query, k=args.k, drug_filter=drug_filter)
    elapsed_ms = (time.perf_counter() - started) * 1000

    print(
        f"strategy={args.strategy} retriever={args.retriever} k={args.k} "
        f"drug_filter={drug_filter or '-'} hits={len(hits)} in {elapsed_ms:.0f} ms"
    )
    print("-" * 78)
    for rank, chunk in enumerate(hits, start=1):
        print(
            f"{rank:>2}. score={chunk['score']:.4f} | {chunk['drug']} | {chunk['section']} "
            f"| table={chunk.get('is_table', False)} | {chunk['chunk_id']}"
        )
        print("    " + truncate(chunk["text"], args.show_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
