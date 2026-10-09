"""Single-shot baseline: the unchanged ``src.rag.answer_question``, re-scored citations.

``answer_question`` parses citations with ``src``'s own parser. Here they are
replaced by the shared parser in ``eval.citations``, the same one the agent
uses, so grouped ids count identically in both arms.
"""

from __future__ import annotations

from eval.citations import normalize_citations, parse_citations
from src.rag import RagResult, answer_question


def answer_baseline(question: str, retriever, *, client, **kwargs) -> RagResult:
    result = answer_question(question, retriever, client=client, **kwargs)
    if result.prompt_variant == "json" and result.citations:
        # Structured citations from the JSON variant: same splitting rule.
        result.citations = normalize_citations(result.citations)
    else:
        result.citations = parse_citations(result.answer)
    return result
