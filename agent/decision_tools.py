"""Decision tools: how the model reports a verdict instead of free text.

The search tools live in ``agent/tools.py``. These four are not executed against
anything; calling one *is* the decision, and its arguments carry the details.
Every node calls the model in tool mode ``ANY``, so it must pick one of them.

    check_evidence     -> ready_to_answer(reason) | search_again(missing)
    answer_or_abstain  -> finish(answer, chunk_ids) | abstain(reason)
"""

from __future__ import annotations

from agent.llm.base import ToolSpec

READY = "ready_to_answer"
SEARCH_AGAIN = "search_again"
FINISH = "finish"
ABSTAIN = "abstain"

CHECK_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name=READY,
        description=(
            "Stop searching. Use when the evidence answers the question, or when it clearly "
            "shows the answer is not available (for example the drug is not in the corpus)."
        ),
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "One sentence."}},
            "required": ["reason"],
        },
    ),
    ToolSpec(
        name=SEARCH_AGAIN,
        description="Search again: the evidence lacks something specific that another search could find.",
        parameters={
            "type": "object",
            "properties": {
                "missing": {"type": "string", "description": "What is missing, in a few words."}
            },
            "required": ["missing"],
        },
    ),
]

ANSWER_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name=FINISH,
        description=(
            "Give the final answer. Put [drug | section | chunk_id] after every claim and list "
            "every cited chunk id in chunk_ids. Only ids shown in the evidence are valid."
        ),
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "chunk_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "chunk_ids"],
        },
    ),
    ToolSpec(
        name=ABSTAIN,
        description="Decline: the evidence does not contain the answer.",
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "One sentence."}},
            "required": ["reason"],
        },
    ),
]
