"""Build the retrieval indexes: FAISS (vectors) + BM25 (lexical) per strategy.

Usage
-----
    python -m src.indexing --strategy all
    python -m src.indexing --strategy section --limit 200     # quick smoke test

Artefacts written to ``data/index/<strategy>/``:

    index.faiss   normalised embeddings, inner-product = cosine similarity
    chunks.jsonl  the chunks, in index order (row i of the index is chunk i)
    tokens.jsonl  BM25 tokens per chunk, one JSON list per line
    meta.json     embedding model, dimension, counts, library versions, timestamp

The BM25 model itself is rebuilt from ``tokens.jsonl`` at load time: tokens are
the expensive part to recompute, and rebuilding the statistics is fast and avoids
pickle-compatibility problems.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.chunking import STRATEGIES
from src.utils import (
    CACHE_DIR,
    INDEX_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    get_logger,
    load_env,
    read_jsonl,
    tokenize,
    write_json,
    write_jsonl,
)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logger = get_logger("indexing")


def chunk_file(strategy: str, chunk_dir: Path | None = None) -> Path:
    return (chunk_dir or (PROCESSED_DIR / "chunks")) / f"{strategy}.jsonl"


def index_dir(strategy: str, base: Path | None = None) -> Path:
    return (base or INDEX_DIR) / strategy


def embed_model_name(explicit: str | None = None) -> str:
    import os

    return explicit or os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL


def model_cache_folder(cache_folder: Path | None = None) -> Path:
    folder = Path(cache_folder) if cache_folder else (CACHE_DIR / "hf")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_sentence_transformer(
    model_name: str,
    *,
    cache_folder: Path | None = None,
    allow_download: bool = True,
):
    """Load a SentenceTransformer, preferring the local cache.

    Loading with ``local_files_only=True`` first keeps re-runs fast and works
    offline; if the model is not cached yet we fall back to downloading it.
    Results are memoised so a process never loads the same model twice.
    """

    return _load_sentence_transformer_cached(
        model_name,
        str(model_cache_folder(cache_folder)),
        bool(allow_download),
    )


@lru_cache(maxsize=4)
def _load_sentence_transformer_cached(model_name: str, cache_folder: str, allow_download: bool):
    import os

    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        return SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=True)
    except Exception:
        if not allow_download:
            raise
        return SentenceTransformer(model_name, cache_folder=cache_folder)


def load_embedder(model_name: str, cache_folder: Path | None = None):
    """Load a sentence-transformers model, falling back to MiniLM if needed."""

    try:
        return load_sentence_transformer(model_name, cache_folder=cache_folder), model_name
    except Exception as exc:  # pragma: no cover - network / model availability
        logger.warning(
            "could not load %s (%s); falling back to %s",
            model_name,
            exc,
            FALLBACK_EMBED_MODEL,
        )
        return (
            load_sentence_transformer(FALLBACK_EMBED_MODEL, cache_folder=cache_folder),
            FALLBACK_EMBED_MODEL,
        )


def build_index(
    strategy: str,
    *,
    chunks: list[dict] | None = None,
    model_name: str | None = None,
    limit: int | None = None,
    chunk_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Embed the chunks of one strategy and persist the FAISS + BM25 index."""

    import faiss
    import numpy as np

    chunks = chunks if chunks is not None else read_jsonl(chunk_file(strategy, chunk_dir))
    if not chunks:
        raise SystemExit(f"no chunks for strategy {strategy!r}; run `python -m src.chunking` first")
    if limit:
        chunks = chunks[:limit]

    target = index_dir(strategy, out_dir)
    target.mkdir(parents=True, exist_ok=True)

    requested_model = embed_model_name(model_name)
    started = time.time()
    embedder, used_model = load_embedder(requested_model)
    texts = [chunk.get("embed_text") or chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(target / "index.faiss"))

    write_jsonl(target / "chunks.jsonl", chunks)
    write_jsonl(target / "tokens.jsonl", [tokenize(text) for text in texts])

    meta = {
        "strategy": strategy,
        "n_chunks": len(chunks),
        "n_table_chunks": sum(1 for chunk in chunks if chunk.get("is_table")),
        "embed_model": used_model,
        "embed_model_requested": requested_model,
        "embed_dim": int(embeddings.shape[1]),
        "similarity": "cosine (normalised inner product)",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - started, 1),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
    }
    try:
        import sentence_transformers

        meta["versions"]["sentence_transformers"] = sentence_transformers.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import rank_bm25

        meta["versions"]["rank_bm25"] = getattr(rank_bm25, "__version__", "0.2.2")
    except Exception:  # pragma: no cover
        pass

    write_json(target / "meta.json", meta)
    logger.info(
        "built %s index: %d chunks, dim %d, model %s, %.1fs -> %s",
        strategy,
        meta["n_chunks"],
        meta["embed_dim"],
        used_model,
        meta["build_seconds"],
        target,
    )
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    parser.add_argument("--model", default=None, help="override EMBED_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="only index the first N chunks")
    parser.add_argument("--out-dir", default=str(INDEX_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    ensure_dirs()
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    out_dir = Path(args.out_dir)

    print(f"{'strategy':<13} {'chunks':>7} {'dim':>5} {'seconds':>9}  model")
    print("-" * 66)
    for strategy in strategies:
        meta = build_index(
            strategy,
            model_name=args.model,
            limit=args.limit,
            out_dir=out_dir,
        )
        print(
            f"{strategy:<13} {meta['n_chunks']:>7} {meta['embed_dim']:>5} "
            f"{meta['build_seconds']:>9.1f}  {meta['embed_model']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
