"""Retrieval-augmented answering with citations.

    python -m src.rag --query "What are the contraindications of warfarin?" \
        --strategy table_aware --retriever hybrid --k 5 --auto-drug

The prompt forces the model to answer only from the retrieved label sections, to
cite as ``[drug | section | chunk_id]`` after each claim, and to abstain with
"The provided labels do not contain this information" when the context is
insufficient. Two prompt variants are implemented so they can be compared:
``plain`` (free-text answer) and ``json`` (structured answer + citation fields).

In mock mode (no API key) the answer is the top retrieved chunk, so the pipeline,
the evaluation harness and the Streamlit app all still run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.chunking import STRATEGIES
from src.llm import LLMClient, LLMConfig, LLMError
from src.retrieval import RETRIEVERS, load_retriever
from src.utils import (
    INDEX_DIR,
    detect_drug_in_query,
    get_logger,
    load_drug_list,
    load_env,
    truncate,
)

PROMPT_VARIANTS = ("plain", "json")

ABSTENTION_SENTENCE = "The provided labels do not contain this information"

SYSTEM_PROMPT = (
    "You are a careful biomedical information assistant answering questions about "
    "FDA drug labels. You answer ONLY from the provided context. Never use outside "
    "knowledge and never guess. If the context does not contain the answer, reply "
    f"exactly: \"{ABSTENTION_SENTENCE}\". You are not a doctor: do not give medical "
    "advice beyond what the label states, and never recommend a treatment change."
)

CITATION_INSTRUCTIONS = (
    "Cite the source of every claim using the format [drug | section | chunk_id] "
    "immediately after the claim. Only cite chunk ids that appear in the context."
)

_CITATION_RE = re.compile(r"\[\s*([^\[\]|]+?)\s*\|\s*([^\[\]|]+?)\s*\|\s*([^\[\]]+?)\s*\]")


@dataclass
class RagResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str = ""
    mock: bool = False
    prompt_variant: str = "plain"
    strategy: str = ""
    retriever: str = ""
    drug_filter: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["retrieved_chunks"] = [chunk["chunk_id"] for chunk in self.retrieved_chunks]
        return payload


def format_context(chunks: list[dict], max_chars_per_chunk: int = 1500) -> str:
    """Numbered context blocks the model can cite."""

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        kind = "table" if chunk.get("is_table") else "text"
        blocks.append(
            f"[{index}] drug={chunk['drug']} | section={chunk['section']} | "
            f"chunk_id={chunk['chunk_id']} | type={kind}\n"
            f"{truncate(chunk['text'], max_chars_per_chunk)}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict], variant: str = "plain") -> str:
    context = format_context(chunks)
    if variant == "json":
        return (
            "Answer the question using only the context below.\n\n"
            f"CONTEXT\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"{CITATION_INSTRUCTIONS}\n"
            "Return a single JSON object with exactly these keys:\n"
            '  "answer": string (the answer, each claim followed by its citation),\n'
            '  "citations": list of objects with "drug", "section", "chunk_id",\n'
            f'  "insufficient_context": boolean (true if the answer is "{ABSTENTION_SENTENCE}").\n'
            "Return only the JSON object, with no markdown fences."
        )
    return (
        "Answer the question using only the context below. "
        f"{CITATION_INSTRUCTIONS}\n"
        f"If the context is insufficient, answer exactly: \"{ABSTENTION_SENTENCE}\"\n\n"
        f"CONTEXT\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def parse_citations(text: str) -> list[dict]:
    """Extract ``[drug | section | chunk_id]`` citations from an answer."""

    citations: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _CITATION_RE.finditer(text or ""):
        drug, section, chunk_id = (part.strip() for part in match.groups())
        key = (drug.lower(), section.lower(), chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"drug": drug, "section": section, "chunk_id": chunk_id})
    return citations


def parse_json_answer(text: str) -> dict | None:
    """Best-effort JSON parsing of a ``json`` variant answer."""

    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _mock_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return f"[mock mode] {ABSTENTION_SENTENCE}"
    top = chunks[0]
    return (
        f"[mock mode: no LLM configured, showing the top retrieved passage] "
        f"[{top['drug']} | {top['section']} | {top['chunk_id']}] {truncate(top['text'], 600)}"
    )


def answer_question(
    question: str,
    retriever,
    *,
    client: LLMClient | None = None,
    k: int = 5,
    drug_filter: str | None = None,
    prompt_variant: str = "plain",
    use_cache: bool = True,
) -> RagResult:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {prompt_variant!r}")

    started = time.perf_counter()
    chunks = retriever.retrieve(question, k=k, drug_filter=drug_filter)
    client = client or LLMClient(LLMConfig.from_env())
    result = RagResult(
        question=question,
        answer="",
        retrieved_chunks=chunks,
        mock=client.mock,
        prompt_variant=prompt_variant,
        strategy=getattr(retriever, "strategy", ""),
        retriever=getattr(retriever, "name", ""),
        drug_filter=drug_filter,
    )

    if client.mock:
        result.answer = _mock_answer(question, chunks)
        result.citations = [
            {"drug": chunk["drug"], "section": chunk["section"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunks[:1]
        ]
        result.model = "(mock)"
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    try:
        response = client.complete(
            build_prompt(question, chunks, prompt_variant),
            system=SYSTEM_PROMPT,
            use_cache=use_cache,
        )
    except LLMError as exc:
        result.answer = f"[LLM error] {exc}"
        result.model = "(error)"
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    answer_text = response.text
    citations: list[dict] = []
    if prompt_variant == "json":
        payload = parse_json_answer(answer_text)
        if payload is not None:
            answer_text = str(payload.get("answer", "")).strip() or answer_text
            raw_citations = payload.get("citations") or []
            for item in raw_citations:
                if isinstance(item, dict) and {"drug", "section", "chunk_id"} <= set(item):
                    citations.append(
                        {
                            "drug": str(item["drug"]).strip(),
                            "section": str(item["section"]).strip(),
                            "chunk_id": str(item["chunk_id"]).strip(),
                        }
                    )
    if not citations:
        citations = parse_citations(answer_text)

    result.answer = answer_text
    result.citations = citations
    result.model = response.model
    result.prompt_tokens = response.prompt_tokens
    result.completion_tokens = response.completion_tokens
    result.latency_ms = (time.perf_counter() - started) * 1000
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--strategy", default="table_aware", choices=list(STRATEGIES))
    parser.add_argument("--retriever", default="hybrid", choices=list(RETRIEVERS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drug", default=None)
    parser.add_argument("--auto-drug", action="store_true")
    parser.add_argument("--prompt-variant", default="plain", choices=list(PROMPT_VARIANTS))
    parser.add_argument("--no-llm", action="store_true", help="force mock mode")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--base-dir", default=str(INDEX_DIR))
    parser.add_argument("--show-context", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()

    drug_filter = args.drug
    if args.auto_drug and not drug_filter:
        drug_filter = detect_drug_in_query(args.query, load_drug_list())
        print(f"[auto] drug filter: {drug_filter or '-'}")

    retriever = load_retriever(args.strategy, args.retriever, base_dir=Path(args.base_dir))
    client = LLMClient(LLMConfig.from_env(force_mock=True if args.no_llm else None))
    result = answer_question(
        args.query,
        retriever,
        client=client,
        k=args.k,
        drug_filter=drug_filter,
        prompt_variant=args.prompt_variant,
        use_cache=not args.no_cache,
    )

    print("=" * 78)
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print("-" * 78)
    print(f"citations: {json.dumps(result.citations, ensure_ascii=False)}")
    print(
        f"strategy={result.strategy} retriever={result.retriever} k={args.k} "
        f"drug_filter={result.drug_filter or '-'} variant={result.prompt_variant} "
        f"mock={result.mock} model={result.model}"
    )
    print(
        f"latency_ms={result.latency_ms:.0f} prompt_tokens={result.prompt_tokens} "
        f"completion_tokens={result.completion_tokens}"
    )
    for index, chunk in enumerate(result.retrieved_chunks[: args.show_context], start=1):
        print(
            f"  [{index}] {chunk['drug']} | {chunk['section']} | table={chunk.get('is_table', False)} "
            f"| {chunk['chunk_id']} | score={chunk['score']:.4f}"
        )
    print("NOTE: for information only - not medical advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
