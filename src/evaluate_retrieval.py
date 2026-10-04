"""Retrieval evaluation: Recall@k and MRR over the chunking x retrieval grid.

    python -m src.evaluate_retrieval                     # all questions
    python -m src.evaluate_retrieval --verified-only     # final, verified questions only
    python -m src.evaluate_retrieval --limit-questions 3 --strategies section --retrievers bm25

Grid: 3 chunking strategies x 4 retrievers x k in {1, 3, 5, 10} x drug filter
{none, auto-detected}. Metrics are reported overall and broken down by
``question_type``.

Ground truth: a retrieved chunk is a hit when its ``drug`` matches the question
and its ``section`` matches ``gold_section``; for ``fixed`` chunking the section
metadata of the chunk start is used *and* any chunk overlapping the gold
section's character range also counts (``sections_overlapping``). Unanswerable
questions are excluded from retrieval metrics - they only test abstention in
``evaluate_answers``.

Honesty guard: while any question is unverified the run is labelled *preliminary*
in the output filename, in a ``run_kind`` column and on the console.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import STRATEGIES
from src.retrieval import RETRIEVERS
from src.utils import (
    FIGURES_DIR,
    INDEX_DIR,
    QUESTIONS_JSONL,
    RESULTS_DIR,
    detect_drug_in_query,
    ensure_dirs,
    get_logger,
    load_drug_list,
    load_env,
    load_eval_questions,
    read_json,
)

DEFAULT_K = (1, 3, 5, 10)

logger = get_logger("evaluate_retrieval")


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def is_answerable(question: dict) -> bool:
    return question.get("question_type") != "unanswerable" and bool(question.get("gold_section"))


def is_hit(chunk: dict, question: dict) -> bool:
    """Hit = same drug and the gold section (or an overlapping section range)."""

    if str(chunk.get("drug", "")).lower() != str(question.get("drug", "")).lower():
        return False
    gold = question.get("gold_section")
    if not gold:
        return False
    if chunk.get("section") == gold:
        return True
    # fixed chunking: a chunk may start in a neighbouring section but overlap it
    return gold in (chunk.get("sections_overlapping") or [])


def first_hit_rank(hits: list[dict], question: dict) -> int | None:
    for rank, chunk in enumerate(hits, start=1):
        if is_hit(chunk, question):
            return rank
    return None


def run_kind_for(stats: dict, verified_only: bool) -> str:
    """``preliminary`` while unverified questions are included, else ``verified_only``."""

    if verified_only:
        return "verified_only"
    return "preliminary" if stats.get("unverified") else "verified_only"


def default_output_paths(run_kind: str) -> tuple[Path, Path]:
    suffix = "_preliminary" if run_kind == "preliminary" else ""
    return (
        RESULTS_DIR / f"retrieval_results{suffix}.csv",
        FIGURES_DIR / f"recall_at_5{suffix}.png",
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_configuration(
    retriever,
    questions: list[dict],
    *,
    k_values: tuple[int, ...],
    drug_filter_mode: str,
) -> list[dict]:
    """Rows ``(question_type, k, recall, mrr, n_questions)`` for one configuration."""

    max_k = max(k_values)
    per_type: dict[str, list[int | None]] = {"overall": []}
    drug_list = load_drug_list()
    for question in questions:
        if not is_answerable(question):
            continue  # unanswerable questions only test abstention, not retrieval
        drug_filter = None
        if drug_filter_mode == "auto":
            drug_filter = detect_drug_in_query(question["question"], drug_list)
        hits = retriever.retrieve(question["question"], k=max_k, drug_filter=drug_filter)
        rank = first_hit_rank(hits, question)
        per_type["overall"].append(rank)
        per_type.setdefault(question["question_type"], []).append(rank)

    rows: list[dict] = []
    for question_type, ranks in per_type.items():
        if not ranks:
            continue
        for k in k_values:
            recall = sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)
            mrr = (
                sum((1.0 / rank) for rank in ranks if rank is not None and rank <= k)
                / len(ranks)
            )
            rows.append(
                {
                    "question_type": question_type,
                    "k": k,
                    "recall": round(recall, 4),
                    "mrr": round(mrr, 4),
                    "n_questions": len(ranks),
                }
            )
    return rows


def build_retrievers(strategies: list[str], retrievers: list[str], index_dir: Path) -> dict:
    """Build one retriever per (strategy, retriever), sharing vector/BM25 models."""

    from src.retrieval import BM25Retriever, HybridRetriever, RerankRetriever, VectorRetriever

    built: dict[tuple[str, str], object] = {}
    for strategy in strategies:
        cache: dict[str, object] = {}

        def vector_for():
            if "vector" not in cache:
                cache["vector"] = VectorRetriever(strategy, base_dir=index_dir)
            return cache["vector"]

        def bm25_for():
            if "bm25" not in cache:
                cache["bm25"] = BM25Retriever(strategy, base_dir=index_dir)
            return cache["bm25"]

        def hybrid_for():
            if "hybrid" not in cache:
                cache["hybrid"] = HybridRetriever(
                    strategy, vector=vector_for(), bm25=bm25_for()
                )
            return cache["hybrid"]

        for retriever_name in retrievers:
            if retriever_name == "vector":
                built[(strategy, "vector")] = vector_for()
            elif retriever_name == "bm25":
                built[(strategy, "bm25")] = bm25_for()
            elif retriever_name == "hybrid":
                built[(strategy, "hybrid")] = hybrid_for()
            elif retriever_name == "hybrid_rerank":
                if "rerank" not in cache:
                    cache["rerank"] = RerankRetriever(strategy, hybrid=hybrid_for())
                built[(strategy, "hybrid_rerank")] = cache["rerank"]
            else:
                raise ValueError(f"unknown retriever {retriever_name!r}")
    return built


def evaluate(
    questions: list[dict],
    *,
    strategies: list[str],
    retrievers: list[str],
    k_values: tuple[int, ...],
    drug_filter_modes: list[str],
    index_dir: Path,
    limit_questions: int | None = None,
) -> tuple[list[dict], dict]:
    answerable = [question for question in questions if is_answerable(question)]
    if limit_questions:
        answerable = answerable[:limit_questions]
    if not answerable:
        raise SystemExit("no answerable questions to evaluate")

    built = build_retrievers(strategies, retrievers, index_dir)
    rows: list[dict] = []
    for strategy in strategies:
        for retriever_name in retrievers:
            retriever = built[(strategy, retriever_name)]
            for drug_filter_mode in drug_filter_modes:
                logger.info(
                    "chunking=%s retriever=%s drug_filter=%s questions=%d",
                    strategy,
                    retriever_name,
                    drug_filter_mode,
                    len(answerable),
                )
                for row in evaluate_configuration(
                    retriever,
                    answerable,
                    k_values=k_values,
                    drug_filter_mode=drug_filter_mode,
                ):
                    rows.append(
                        {
                            "chunking": strategy,
                            "retrieval": retriever_name,
                            "k": row["k"],
                            "drug_filter": drug_filter_mode,
                            "recall": row["recall"],
                            "mrr": row["mrr"],
                            "n_questions": row["n_questions"],
                            "question_type": row["question_type"],
                        }
                    )

    metadata = {
        "strategies": strategies,
        "retrievers": retrievers,
        "k_values": list(k_values),
        "drug_filter": drug_filter_modes,
        "n_answerable_evaluated": len(answerable),
        "index_models": {
            strategy: read_json(index_dir / strategy / "meta.json").get("embed_model", "unknown")
            for strategy in strategies
            if (index_dir / strategy / "meta.json").exists()
        },
    }
    return rows, metadata


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "chunking",
    "retrieval",
    "k",
    "drug_filter",
    "question_type",
    "recall",
    "mrr",
    "n_questions",
    "n_verified",
    "n_unverified",
    "run_kind",
]


def write_csv(rows: list[dict], path: Path, *, run_kind: str, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "n_verified": stats["verified"],
                    "n_unverified": stats["unverified"],
                    "run_kind": run_kind,
                }
            )
    logger.info("wrote %d result rows to %s", len(rows), path)


def write_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_recall(rows: list[dict], path: Path, *, k: int = 5, title_suffix: str = "") -> Path | None:
    """Bar chart of Recall@k per (chunking, retrieval) configuration."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is in requirements
        logger.warning("matplotlib unavailable; skipping the figure")
        return None

    overall = [row for row in rows if row["question_type"] == "overall" and row["k"] == k]
    if not overall:
        return None

    filters = sorted({row["drug_filter"] for row in overall})
    configs = sorted({(row["chunking"], row["retrieval"]) for row in overall})
    labels = [f"{chunking}\n{retriever}" for chunking, retriever in configs]
    width = 0.8 / max(1, len(filters))

    figure, axis = plt.subplots(figsize=(max(8.0, 1.35 * len(configs)), 4.5))
    for index, drug_filter in enumerate(filters):
        values = [
            next(
                (
                    row["recall"]
                    for row in overall
                    if (row["chunking"], row["retrieval"]) == config
                    and row["drug_filter"] == drug_filter
                ),
                0.0,
            )
            for config in configs
        ]
        positions = [position + index * width for position in range(len(configs))]
        bars = axis.bar(positions, values, width=width, label=f"drug filter: {drug_filter}")
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)

    axis.set_xticks(
        [position + width * (len(filters) - 1) / 2 for position in range(len(configs))]
    )
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel(f"Recall@{k}")
    axis.set_ylim(0, 1.05)
    axis.set_title(f"Recall@{k} by configuration{title_suffix}")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("wrote figure to %s", path)
    return path


def print_table(rows: list[dict], k: int = 5, *, question_type: str = "overall") -> None:
    selected = [
        row
        for row in rows
        if row["question_type"] == question_type and row["k"] == k
    ]
    if not selected:
        return
    header = (
        f"{'chunking':<13} {'retrieval':<14} {'drug_filter':<12} "
        f"{'recall':>7} {'mrr':>7} {'n':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(
        selected, key=lambda item: (-item["recall"], -item["mrr"], item["chunking"], item["retrieval"])
    ):
        print(
            f"{row['chunking']:<13} {row['retrieval']:<14} {row['drug_filter']:<12} "
            f"{row['recall']:>7.3f} {row['mrr']:>7.3f} {row['n_questions']:>4}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTIONS_JSONL))
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="evaluate only questions with verified=true (the reported run)",
    )
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS), choices=list(RETRIEVERS))
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument(
        "--drug-filter",
        nargs="+",
        default=["none", "auto"],
        choices=["none", "auto"],
        help="none: no filter; auto: detect the drug name in the question",
    )
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--out", default=None, help="output CSV (default depends on run kind)")
    parser.add_argument("--figure", default=None)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--show-types", action="store_true", help="also print per-question-type rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    ensure_dirs()

    questions, stats = load_eval_questions(args.questions, verified_only=args.verified_only)
    if not questions:
        logger.error(
            "nothing to evaluate: verified_only=%s, verified=%d/%d "
            "(review questions with `python -m src.review_questions`)",
            args.verified_only,
            stats["verified"],
            stats["total"],
        )
        return 1

    run_kind = run_kind_for(stats, args.verified_only)
    if run_kind == "preliminary":
        print(
            f"WARNING: PRELIMINARY RUN - {stats['unverified']} of {stats['total']} questions are "
            "still unverified; these numbers must not go into the README."
        )
    print(
        f"questions used: {stats['selected']} of {stats['total']} "
        f"(verified={stats['verified']}, unverified={stats['unverified']})"
    )

    default_csv, default_figure = default_output_paths(run_kind)
    out_path = Path(args.out) if args.out else default_csv
    figure_path = Path(args.figure) if args.figure else default_figure
    metadata_path = out_path.with_name(out_path.stem + "_meta.json")

    k_values = tuple(sorted(set(args.k)))
    rows, metadata = evaluate(
        questions,
        strategies=args.strategies,
        retrievers=args.retrievers,
        k_values=k_values,
        drug_filter_modes=args.drug_filter,
        index_dir=Path(args.index_dir),
        limit_questions=args.limit_questions,
    )

    metadata.update(
        {
            "run_kind": run_kind,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question_file": str(args.questions),
            "questions_total": stats["total"],
            "questions_verified": stats["verified"],
            "questions_unverified": stats["unverified"],
            "questions_selected": stats["selected"],
            "csv": str(out_path),
            "figure": None if args.no_figure else str(figure_path),
        }
    )

    write_csv(rows, out_path, run_kind=run_kind, stats=stats)
    write_metadata(metadata, metadata_path)

    print()
    print(f"overall metrics at k=5 ({run_kind})")
    print_table(rows, k=5)
    print()
    print(f"overall metrics at k=1,3,5,10 per configuration ({run_kind})")
    for k in k_values:
        print(f"\n  k={k}")
        print_table(rows, k=k)
    if args.show_types:
        print()
        print(f"per question_type at k=5 ({run_kind})")
        header = (
            f"{'chunking':<13} {'retrieval':<14} {'drug_filter':<12} {'question_type':<18} "
            f"{'recall':>7} {'mrr':>7} {'n':>4}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            if row["k"] != 5 or row["question_type"] == "overall":
                continue
            print(
                f"{row['chunking']:<13} {row['retrieval']:<14} {row['drug_filter']:<12} "
                f"{row['question_type']:<18} {row['recall']:>7.3f} {row['mrr']:>7.3f} "
                f"{row['n_questions']:>4}"
            )

    if not args.no_figure:
        plot_recall(
            rows,
            figure_path,
            k=5,
            title_suffix=" (preliminary, unverified questions)" if run_kind == "preliminary" else "",
        )

    print()
    print(f"run kind       : {run_kind}")
    print(f"questions used : {metadata['n_answerable_evaluated']} answerable "
          f"({stats['selected']} selected, {stats['total']} in file)")
    print(f"csv            : {out_path}")
    print(f"metadata       : {metadata_path}")
    print(f"figure         : {figure_path if not args.no_figure else 'not run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
