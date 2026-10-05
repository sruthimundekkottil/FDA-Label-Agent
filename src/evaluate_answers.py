"""Answer-quality evaluation: citations, abstention, keypoint coverage, faithfulness.

    python -m src.evaluate_answers --verified-only --configs best fixed:vector:none

For each configuration the script asks the RAG pipeline every verified question
and scores:

* **citation accuracy** - does every ``[drug | section | chunk_id]`` the answer
  cites refer to the gold section *and* to a chunk that was actually retrieved
  (fabricated citations are counted separately);
* **abstention accuracy** - on unanswerable questions, does the answer decline?
* **keypoint coverage** - lexical coverage of ``gold_answer_keypoints`` (a
  keypoint counts as covered when >=60% of its content words appear in the
  answer); this is a cheap string-matching proxy, not a judge;
* **faithfulness** (optional, LLM-as-judge) - every claim in the answer is
  checked against the retrieved context and marked supported/unsupported.

Configuration ``best`` is resolved from ``results/retrieval_results.csv``
(highest Recall@5, then MRR). Per-question details are written next to the CSV so
``failure_analysis`` can reuse them.

As with the retrieval evaluation, an unverified run is labelled *preliminary* and
writes ``answer_results_preliminary.csv`` so it can never be mistaken for the
reported run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import STRATEGIES
from src.indexing import index_dir
from src.llm import LLMClient, LLMConfig
from src.rag import ABSTENTION_SENTENCE, answer_question, parse_json_answer
from src.retrieval import RETRIEVERS, load_retriever
from src.utils import (
    INDEX_DIR,
    QUESTIONS_JSONL,
    RESULTS_DIR,
    detect_drug_in_query,
    ensure_dirs,
    get_logger,
    load_drug_list,
    load_env,
    load_eval_questions,
    read_jsonl,
    tokenize,
    truncate,
    write_jsonl,
)

RETRIEVAL_RESULTS = RESULTS_DIR / "retrieval_results.csv"
PRELIMINARY_RETRIEVAL_RESULTS = RESULTS_DIR / "retrieval_results_preliminary.csv"

logger = get_logger("evaluate_answers")

# Same list as the question drafter: question words rather than answer words.
STOPWORDS = {
    "a", "about", "according", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "both", "by", "can", "describe", "did", "do", "does", "during", "each", "for", "from",
    "has", "have", "how", "if", "in", "include", "includes", "including", "into", "is", "it",
    "its", "label", "labels", "list", "listed", "may", "must", "not", "of", "on", "or", "per",
    "please", "report", "reported", "reporting", "says", "see", "should", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "use", "used", "using", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "within", "would", "patient", "patients", "and/or", "your", "you",
}

ABSTENTION_MARKERS = (
    "do not contain",
    "does not contain",
    "not contain this information",
    "do not include this information",
    "insufficient",
    "cannot answer",
    "can not answer",
    "no information",
    "not provided in",
    "does not provide",
    "do not provide",
    "does not include information",
    "not mentioned in",
    "not specified in",
    "not available in the",
    "no information about",
    "the provided labels do not contain this information",
)

FAITHFULNESS_PROMPT = """You are auditing one answer against the context it was given.

CONTEXT
{context}

ANSWER
{answer}

Split the answer into atomic claims. For each claim decide whether it is
supported by the context above. Reply with a single JSON object:
{{"claims": [{{"claim": "text", "supported": true}}], "supported": 2, "unsupported": 1}}
Do not use markdown fences, and do not add commentary."""


@dataclass
class ConfigSpec:
    chunking: str
    retrieval: str
    drug_filter: str

    @property
    def label(self) -> str:
        return f"{self.chunking}/{self.retrieval}/{self.drug_filter}"

    def to_dict(self) -> dict:
        return {"chunking": self.chunking, "retrieval": self.retrieval, "drug_filter": self.drug_filter}


@dataclass
class AnswerMetrics:
    question_id: str
    question: str
    question_type: str
    drug: str
    gold_section: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    n_citations: int = 0
    citations_correct: int = 0
    citations_fabricated: int = 0
    citation_ok: bool = False
    abstained: bool | None = None
    keypoint_coverage: float | None = None
    faithfulness_ratio: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    mock: bool = False
    error: str | None = None


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
def parse_config(text: str) -> ConfigSpec:
    parts = text.split(":")
    if len(parts) != 3:
        raise SystemExit(f"configuration must be chunking:retriever:drug_filter, got {text!r}")
    chunking, retrieval, drug_filter = parts
    if chunking not in STRATEGIES:
        raise SystemExit(f"unknown chunking {chunking!r}")
    if retrieval not in RETRIEVERS:
        raise SystemExit(f"unknown retriever {retrieval!r}")
    if drug_filter not in {"none", "auto"}:
        raise SystemExit(f"unknown drug filter {drug_filter!r}")
    return ConfigSpec(chunking, retrieval, drug_filter)


def best_config_from_csv(results_path: Path, *, k: int = 5) -> ConfigSpec:
    """Best configuration in results/retrieval_results.csv (Recall@k, then MRR)."""

    if not results_path.exists():
        raise SystemExit(
            f"{results_path} not found - run `python -m src.evaluate_retrieval --verified-only` first"
        )
    rows = [
        row
        for row in csv.DictReader(results_path.open(encoding="utf-8"))
        if row["question_type"] == "overall" and int(row["k"]) == k
    ]
    if not rows:
        raise SystemExit(f"no overall rows at k={k} in {results_path}")
    best = max(rows, key=lambda row: (float(row["recall"]), float(row["mrr"])))
    return ConfigSpec(best["chunking"], best["retrieval"], best["drug_filter"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def content_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) >= 2}


def _terms_match(left: str, right: str) -> bool:
    """Terms match exactly, or when they share a six-character stem.

    Labels and answers disagree morphologically all the time
    ("contraindicated" vs "contraindication"), which exact matching would punish.
    """

    if left == right:
        return True
    return len(left) >= 6 and len(right) >= 6 and left[:6] == right[:6]


def term_overlap(left: set[str], right: set[str]) -> int:
    return sum(1 for term in left if any(_terms_match(term, other) for other in right))


def keypoint_coverage(answer: str, keypoints: list[str], *, threshold: float = 0.6) -> float | None:
    """Fraction of gold keypoints whose content words appear in the answer."""

    if not keypoints:
        return None
    answer_terms = content_terms(answer)
    covered = 0
    for keypoint in keypoints:
        terms = content_terms(keypoint)
        if not terms:
            continue
        if term_overlap(terms, answer_terms) / len(terms) >= threshold:
            covered += 1
    return covered / len(keypoints)


def is_abstention(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


def score_citations(question: dict, result) -> tuple[int, int, int, bool]:
    """``(n_citations, correct, fabricated, citation_ok)`` for one answer."""

    gold_drug = str(question.get("drug", "")).lower()
    gold_section = question.get("gold_section") or ""
    retrieved = {chunk["chunk_id"] for chunk in result.retrieved_chunks}
    correct = 0
    fabricated = 0
    for citation in result.citations:
        chunk_id = citation.get("chunk_id", "")
        same_drug = str(citation.get("drug", "")).lower() == gold_drug
        same_section = citation.get("section", "") == gold_section
        exists = chunk_id in retrieved
        if not exists:
            fabricated += 1
        elif same_drug and same_section:
            correct += 1
    n_citations = len(result.citations)
    citation_ok = n_citations > 0 and correct == n_citations and fabricated == 0
    return n_citations, correct, fabricated, citation_ok


def judge_faithfulness(client: LLMClient, answer: str, chunks: list[dict]) -> tuple[float | None, int, int]:
    """LLM-as-judge: returns ``(supported_ratio, supported, unsupported)``."""

    if client.mock or not chunks or not answer.strip():
        return None, 0, 0
    context = "\n\n".join(
        f"[{index}] {chunk['drug']} | {chunk['section']} | {chunk['chunk_id']}\n"
        f"{truncate(chunk['text'], 1200)}"
        for index, chunk in enumerate(chunks, start=1)
    )
    try:
        response = client.complete(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer),
            system="You are a strict but fair fact-checker. Reply with JSON only.",
        )
    except Exception as exc:  # pragma: no cover - network/provider issues
        logger.warning("faithfulness judge failed: %s", exc)
        return None, 0, 0
    payload = parse_json_answer(response.text)
    if not payload:
        return None, 0, 0
    claims = payload.get("claims") or []
    supported = sum(1 for claim in claims if isinstance(claim, dict) and claim.get("supported"))
    unsupported = sum(1 for claim in claims if isinstance(claim, dict) and not claim.get("supported"))
    if payload.get("supported") is not None:
        supported = max(supported, int(payload.get("supported") or 0))
    if payload.get("unsupported") is not None:
        unsupported = max(unsupported, int(payload.get("unsupported") or 0))
    total = supported + unsupported
    if total == 0:
        return None, 0, 0
    return supported / total, supported, unsupported


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_config(
    spec: ConfigSpec,
    questions: list[dict],
    *,
    client: LLMClient,
    k: int,
    prompt_variant: str,
    index_dir_path: Path,
    faithfulness: bool,
    limit_questions: int | None = None,
) -> tuple[dict, list[dict]]:
    retriever = load_retriever(spec.chunking, spec.retrieval, base_dir=index_dir_path)
    selected = questions[:limit_questions] if limit_questions else questions
    drug_list = load_drug_list()
    details: list[AnswerMetrics] = []

    for question in selected:
        drug_filter = None
        if spec.drug_filter == "auto":
            drug_filter = detect_drug_in_query(question["question"], drug_list)
        result = answer_question(
            question["question"],
            retriever,
            client=client,
            k=k,
            drug_filter=drug_filter,
            prompt_variant=prompt_variant,
        )
        n_citations, correct, fabricated, citation_ok = score_citations(question, result)
        entry = AnswerMetrics(
            question_id=question.get("id", ""),
            question=question["question"],
            question_type=question.get("question_type", ""),
            drug=question.get("drug", ""),
            gold_section=question.get("gold_section") or "",
            answer=result.answer,
            citations=result.citations,
            retrieved_chunk_ids=[chunk["chunk_id"] for chunk in result.retrieved_chunks],
            n_citations=n_citations,
            citations_correct=correct,
            citations_fabricated=fabricated,
            citation_ok=citation_ok,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            mock=result.mock,
        )
        if question.get("question_type") == "unanswerable":
            entry.abstained = is_abstention(result.answer)
            entry.keypoint_coverage = None
        else:
            entry.keypoint_coverage = keypoint_coverage(
                result.answer, question.get("gold_answer_keypoints") or []
            )
        if entry.answer.startswith("[LLM error]"):
            entry.error = entry.answer
        if faithfulness and question.get("question_type") != "unanswerable":
            ratio, supported, unsupported = judge_faithfulness(client, result.answer, result.retrieved_chunks)
            entry.faithfulness_ratio = ratio
            entry.__dict__["faithfulness_supported"] = supported
            entry.__dict__["faithfulness_unsupported"] = unsupported
        details.append(entry)
        logger.info(
            "%s | %s | citations=%s/%s fabricated=%s abstained=%s coverage=%s",
            spec.label,
            entry.question_id,
            correct,
            n_citations,
            fabricated,
            entry.abstained,
            None if entry.keypoint_coverage is None else round(entry.keypoint_coverage, 2),
        )

    summary = summarize_config(spec, details, faithfulness=faithfulness)
    return summary, [entry.__dict__ for entry in details]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_config(spec: ConfigSpec, details: list[AnswerMetrics], *, faithfulness: bool) -> dict:
    answerable = [entry for entry in details if entry.question_type != "unanswerable"]
    unanswerable = [entry for entry in details if entry.question_type == "unanswerable"]
    cited = [entry for entry in answerable if entry.n_citations > 0]

    citation_accuracy = (
        sum(1 for entry in answerable if entry.citation_ok) / len(answerable) if answerable else None
    )
    citation_precision = (
        sum(entry.citations_correct for entry in cited)
        / max(1, sum(entry.n_citations for entry in cited))
        if cited
        else None
    )
    abstention_accuracy = (
        sum(1 for entry in unanswerable if entry.abstained) / len(unanswerable) if unanswerable else None
    )
    coverage = [entry.keypoint_coverage for entry in answerable if entry.keypoint_coverage is not None]
    faithfulness_values = [
        entry.faithfulness_ratio for entry in answerable if entry.faithfulness_ratio is not None
    ]

    return {
        **spec.to_dict(),
        "config": spec.label,
        "n_questions": len(details),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "citation_accuracy": None if citation_accuracy is None else round(citation_accuracy, 4),
        "citation_precision": None if citation_precision is None else round(citation_precision, 4),
        "fabricated_citations": sum(entry.citations_fabricated for entry in details),
        "abstention_accuracy": None if abstention_accuracy is None else round(abstention_accuracy, 4),
        "keypoint_coverage": None if not coverage else round(_mean(coverage), 4),
        "faithfulness_supported_ratio": (
            None if not faithfulness_values else round(_mean(faithfulness_values), 4)
        ),
        "faithfulness_judged": len(faithfulness_values),
        "avg_latency_ms": round(_mean([entry.latency_ms for entry in details]) or 0.0, 1),
        "total_prompt_tokens": sum(entry.prompt_tokens or 0 for entry in details),
        "total_completion_tokens": sum(entry.completion_tokens or 0 for entry in details),
        "errors": sum(1 for entry in details if entry.error),
        "mock": any(entry.mock for entry in details),
        "faithfulness_enabled": faithfulness,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "run_kind",
    "n_verified",
    "n_unverified",
    "prompt_variant",
    "model",
    "config",
    "chunking",
    "retrieval",
    "drug_filter",
    "n_questions",
    "n_answerable",
    "n_unanswerable",
    "citation_accuracy",
    "citation_precision",
    "fabricated_citations",
    "abstention_accuracy",
    "keypoint_coverage",
    "faithfulness_supported_ratio",
    "faithfulness_judged",
    "avg_latency_ms",
    "total_prompt_tokens",
    "total_completion_tokens",
    "errors",
    "mock",
]


def write_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("wrote %d configuration rows to %s", len(rows), path)


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'config':<30} {'cited':>6} {'cite_prec':>9} {'fab':>4} {'abstain':>8} "
        f"{'keypts':>7} {'faith':>6} {'lat_ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(value, digits=3):
            return "-" if value is None else f"{value:.{digits}f}"

        print(
            f"{row['config']:<30} {fmt(row['citation_accuracy']):>6} "
            f"{fmt(row['citation_precision']):>9} {row['fabricated_citations']:>4} "
            f"{fmt(row['abstention_accuracy']):>8} {fmt(row['keypoint_coverage']):>7} "
            f"{fmt(row['faithfulness_supported_ratio']):>6} {row['avg_latency_ms']:>9.0f}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTIONS_JSONL))
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["best", "fixed:vector:none"],
        help="'best' resolves the top configuration from results/retrieval_results.csv",
    )
    parser.add_argument("--retrieval-results", default=str(RETRIEVAL_RESULTS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--prompt-variant", default="plain", choices=["plain", "json"])
    parser.add_argument("--faithfulness", default="best", choices=["none", "best", "all"])
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--out", default=None)
    parser.add_argument("--details-dir", default=str(RESULTS_DIR))
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="force mock mode (no API calls)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    ensure_dirs()

    questions, stats = load_eval_questions(args.questions, verified_only=args.verified_only)
    if not questions:
        logger.error("no questions to evaluate (verified_only=%s)", args.verified_only)
        return 1

    # Resolve configurations ("best" -> top row of the retrieval results CSV).
    retrieval_path = Path(args.retrieval_results)
    if not retrieval_path.exists() and retrieval_path == RETRIEVAL_RESULTS and PRELIMINARY_RETRIEVAL_RESULTS.exists():
        logger.warning("%s not found; falling back to the preliminary retrieval results", retrieval_path)
        retrieval_path = PRELIMINARY_RETRIEVAL_RESULTS
    specs: list[ConfigSpec] = []
    for text in args.configs:
        specs.append(best_config_from_csv(retrieval_path, k=args.k) if text == "best" else parse_config(text))

    client = LLMClient(LLMConfig.from_env(force_mock=True if args.mock else None))
    run_kind = "verified_only" if args.verified_only else ("preliminary" if stats["unverified"] else "verified_only")
    if run_kind == "preliminary":
        print(
            f"WARNING: PRELIMINARY RUN - {stats['unverified']} of {stats['total']} questions are "
            "still unverified; these numbers must not go into the README."
        )
    print(
        f"questions used: {stats['selected']} of {stats['total']} "
        f"(verified={stats['verified']}, unverified={stats['unverified']})"
    )
    print(
        f"configs: {', '.join(spec.label for spec in specs)}\n"
        f"model: {client.config.model if not client.mock else '(mock)'} | "
        f"prompt variant: {args.prompt_variant} | faithfulness: {args.faithfulness}"
    )

    suffix = "_preliminary" if run_kind == "preliminary" else ""
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"answer_results{suffix}.csv"
    details_dir = Path(args.details_dir)

    rows: list[dict] = []
    for index, spec in enumerate(specs):
        use_faithfulness = args.faithfulness == "all" or (args.faithfulness == "best" and index == 0)
        print(f"\n--- {spec.label} (faithfulness={'on' if use_faithfulness else 'off'}) ---")
        summary, details = evaluate_config(
            spec,
            questions,
            client=client,
            k=args.k,
            prompt_variant=args.prompt_variant,
            index_dir_path=Path(args.index_dir),
            faithfulness=use_faithfulness,
            limit_questions=args.limit_questions,
        )
        summary.update(
            {
                "run_kind": run_kind,
                "n_verified": stats["verified"],
                "n_unverified": stats["unverified"],
                "prompt_variant": args.prompt_variant,
                "model": client.config.model if not client.mock else "(mock)",
            }
        )
        rows.append(summary)
        details_path = details_dir / f"answer_details_{spec.chunking}_{spec.retrieval}_{spec.drug_filter}{suffix}.jsonl"
        write_jsonl(details_path, details)
        print(f"per-question details: {details_path}")

    write_results(rows, out_path)
    print()
    print_table(rows)
    print()
    print(f"run kind       : {run_kind}")
    print(f"questions used : {stats['selected']} of {stats['total']}")
    print(f"csv            : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
