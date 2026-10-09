"""The one citation parser used for both arms of the evaluation.

Answers cite as ``[drug | section | chunk_id]``. Models sometimes group several
ids in one bracket, ``[drug | section | id1, id2]``, or copy the context labels,
``[drug | section=s | chunk_id=id]``; ``src.rag.parse_citations`` reads either as
a non-existent id. This module builds on that parser (``src/`` stays unchanged),
splits grouped ids and strips the labels, so the baseline and the agent are
scored by exactly the same rule. Whether an id is real is still checked against
the retrieved chunks by the caller.
"""

from __future__ import annotations

import re

from src.rag import parse_citations as _parse_bracketed

_ID_SEPARATORS = re.compile(r"[,;]")
# The context blocks read "drug=x | section=y | chunk_id=z" and models sometimes
# copy those labels into the citation: [x | section=y | chunk_id=z].
_FIELD_LABEL = re.compile(r"^\s*(?:drug|section|chunk_id)\s*=\s*", re.IGNORECASE)


def _strip_label(value: str) -> str:
    return _FIELD_LABEL.sub("", value or "").strip()


def split_ids(raw: str) -> list[str]:
    """``"a, chunk_id=b; c"`` -> ``["a", "b", "c"]``. Chunk ids never contain , ; or =."""

    return [_strip_label(part) for part in _ID_SEPARATORS.split(raw or "") if _strip_label(part)]


def normalize_citations(citations: list[dict]) -> list[dict]:
    """Split grouped ids in already-parsed citations and drop duplicates (order kept)."""

    normalised: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        drug = _strip_label(str(citation.get("drug", "")))
        section = _strip_label(str(citation.get("section", "")))
        for chunk_id in split_ids(str(citation.get("chunk_id", ""))):
            key = (drug.lower(), section.lower(), chunk_id)
            if key in seen:
                continue
            seen.add(key)
            normalised.append({"drug": drug, "section": section, "chunk_id": chunk_id})
    return normalised


def parse_citations(text: str) -> list[dict]:
    """All ``[drug | section | id(, id)*]`` citations in ``text``, one dict per id."""

    return normalize_citations(_parse_bracketed(text or ""))


def cited_chunk_ids(text: str) -> list[str]:
    return list(dict.fromkeys(citation["chunk_id"] for citation in parse_citations(text)))
