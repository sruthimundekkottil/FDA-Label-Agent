"""The agent as a LangGraph state graph.

    python -m agent.loop --query "What are the contraindications of warfarin?"
    python -m agent.draw_graph            # write the diagram to docs/agent_graph.md

Edges (diagram: docs/agent_graph.md):

    START             -> rewrite_query
    rewrite_query     -> retrieve            (new_calls)
                      -> answer_or_abstain   (all_repeated: nothing new to search)
    retrieve          -> check_evidence      (always)
    check_evidence    -> rewrite_query       (search_again, if a full round still fits)
                      -> answer_or_abstain   (enough_or_out_of_steps)
    answer_or_abstain -> answer_or_abstain   (fix_citations: once, if a step is left)
                      -> END                 (done)
    any model node    -> END                 (error)

LangGraph only supplies the control flow (state, nodes, edges). Every model call
goes through our own adapter (``agent.llm``) with its record/replay cache; no
LangChain model wrappers are used.

Budget: one *step* is one model call, at most ``max_steps`` (5) per question.
rewrite_query, check_evidence and answer_or_abstain each cost one step; retrieve
is free. Routing never starts a search round it cannot finish, so the final
answer call always fits in the budget.

Every node writes a JSONL ``node`` record; model calls and tool calls get their
own ``llm_call`` / ``tool_call`` records.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.decision_tools import ABSTAIN, ANSWER_TOOLS, CHECK_TOOLS, FINISH, READY, SEARCH_AGAIN
from agent.llm import GenerationSettings, LLMError, Message, ToolCall, get_adapter, settings_from_env
from agent.prompts import answer_prompt, build_system_prompt, check_prompt, rewrite_prompt
from agent.tools import ToolBox
from eval.citations import cited_chunk_ids, split_ids
from src.rag import ABSTENTION_SENTENCE

MAX_STEPS = 5
MAX_CALLS_PER_ROUND = 3
MAX_ANSWER_ATTEMPTS = 2  # one retry when the citation check fails
STEP_COST_OF_SEARCH_ROUND = 3  # rewrite_query + check_evidence + the answer that must still fit


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    # input
    question: str
    run_id: str
    question_id: str
    # search
    rewritten_query: str
    queries_tried: list[str]
    pending_calls: list[dict]  # planned by rewrite_query, run by retrieve
    call_signatures: list[str]  # every call already executed (repeated-call detection)
    retrieved_chunks: list[dict]
    observations: list[str]  # one line per executed tool call
    tools_used: list[str]
    # judgement
    evidence_verdict: str  # ready | search_again
    missing: str
    # answer
    draft_answer: str
    citations: list[dict]
    unsupported_citations: list[dict]
    citation_feedback: str
    answer_attempts: int
    # bookkeeping
    step_count: int
    stop_reason: str  # answered | abstained | no_answer | llm_error
    events: list[str]  # step_cap, repeated_call:..., citation_check_failed, ...
    usage: dict
    model_version: str
    all_cached: bool


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    unsupported_citations: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    events: list[str] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    n_steps: int = 0
    n_tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thoughts_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    model_version: str = ""
    all_cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class JsonlLog:
    """Append-only JSONL log; a ``None`` path makes it a no-op."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if not self.path:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def call_signature(name: str, args: dict) -> str:
    """Normalised identity of a tool call: same tool + same arguments = repeat."""

    normalised = {
        key: (" ".join(str(value).lower().split()) if isinstance(value, str) else value)
        for key, value in (args or {}).items()
        if value not in (None, "")
    }
    return json.dumps({"tool": name, "args": normalised}, sort_keys=True)


def _observation(name: str, args: dict, output: dict, n_new: int) -> str:
    shown = json.dumps(args, ensure_ascii=False, sort_keys=True)
    if "error" in output:
        return f"{name}({shown}) -> error: {output['error']}"
    n = len(output.get("chunks", []))
    return f"{name}({shown}) -> {n} chunks ({n_new} new)"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class AgentGraph:
    """Builds and runs the compiled graph. Nodes are methods; routers are pure."""

    def __init__(
        self,
        *,
        llm,
        toolbox: ToolBox,
        settings: GenerationSettings | None = None,
        max_steps: int = MAX_STEPS,
        log: JsonlLog | None = None,
    ):
        if max_steps < STEP_COST_OF_SEARCH_ROUND:
            raise ValueError(f"max_steps must be at least {STEP_COST_OF_SEARCH_ROUND} (rewrite, check, answer)")
        self.llm = llm
        self.toolbox = toolbox
        self.settings = settings or GenerationSettings()
        self.max_steps = max_steps
        self.log = log or JsonlLog(None)
        self.system_prompt = build_system_prompt(toolbox.drugs)
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("rewrite_query", self._logged("rewrite_query", self.rewrite_query))
        builder.add_node("retrieve", self._logged("retrieve", self.retrieve))
        builder.add_node("check_evidence", self._logged("check_evidence", self.check_evidence))
        builder.add_node("answer_or_abstain", self._logged("answer_or_abstain", self.answer_or_abstain))

        builder.add_edge(START, "rewrite_query")
        builder.add_conditional_edges(
            "rewrite_query",
            self.route_after_rewrite,
            {"new_calls": "retrieve", "all_repeated": "answer_or_abstain", "error": END},
        )
        builder.add_edge("retrieve", "check_evidence")
        builder.add_conditional_edges(
            "check_evidence",
            self.route_after_check,
            {
                # One label per target: LangGraph draws a single edge per node pair.
                # Which case it was is in the events ("step_cap").
                "enough_or_out_of_steps": "answer_or_abstain",
                "search_again": "rewrite_query",
                "error": END,
            },
        )
        builder.add_conditional_edges(
            "answer_or_abstain",
            self.route_after_answer,
            {"fix_citations": "answer_or_abstain", "done": END},
        )
        return builder.compile()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    # -- shared helpers ----------------------------------------------------------
    def _logged(self, name: str, node):
        def wrapper(state: AgentState) -> dict:
            started = time.perf_counter()
            update = node(state)
            summary = {
                key: value
                for key, value in update.items()
                if key not in ("retrieved_chunks", "usage", "observations", "citations", "call_signatures")
            }
            if "draft_answer" in summary:
                summary["draft_answer"] = summary["draft_answer"][:200]
            if "retrieved_chunks" in update:
                summary["n_retrieved_chunks"] = len(update["retrieved_chunks"])
            if "citations" in update:
                summary["cited_chunk_ids"] = [c["chunk_id"] for c in update["citations"]]
            self.log.write(
                {
                    "run_id": state.get("run_id", ""),
                    "question_id": state.get("question_id", ""),
                    "type": "node",
                    "node": name,
                    "step_count": update.get("step_count", state.get("step_count", 0)),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "update": summary,
                }
            )
            return update

        return wrapper

    def _call_llm(self, state: AgentState, node: str, prompt: str, tools) -> tuple[object | None, dict]:
        """One model call in tool mode ANY. Returns ``(reply or None, state update)``."""

        step = state.get("step_count", 0) + 1
        update: dict = {"step_count": step}
        settings = replace(self.settings, tool_mode="ANY")
        try:
            reply = self.llm.generate([Message.user(prompt)], system=self.system_prompt, tools=tools, settings=settings)
        except LLMError as exc:
            self.log.write(
                {
                    "run_id": state.get("run_id", ""),
                    "question_id": state.get("question_id", ""),
                    "type": "llm_error",
                    "node": node,
                    "step": step,
                    "error": str(exc),
                }
            )
            update.update(stop_reason="llm_error", draft_answer=f"[LLM error] {exc}")
            return None, update

        usage = dict(state.get("usage") or {})
        for key, value in asdict(reply.usage).items():
            usage[key] = usage.get(key, 0) + (value or 0)
        update.update(
            usage=usage,
            model_version=reply.model_version or state.get("model_version", ""),
            all_cached=reply.cached if step == 1 else bool(state.get("all_cached")) and reply.cached,
        )
        self.log.write(
            {
                "run_id": state.get("run_id", ""),
                "question_id": state.get("question_id", ""),
                "type": "llm_call",
                "node": node,
                "step": step,
                "tool_calls": [{"name": c.name, "args": c.args} for c in reply.tool_calls],
                "finish_reason": reply.finish_reason,
                "usage": asdict(reply.usage),
                "cached": reply.cached,
                "latency_ms": round(reply.latency_ms, 1),
            }
        )
        return reply, update

    def _events(self, state: AgentState, *new: str) -> list[str]:
        return [*state.get("events", []), *new]

    def _can_search_again(self, step_count: int) -> bool:
        return step_count + STEP_COST_OF_SEARCH_ROUND <= self.max_steps

    # -- nodes ---------------------------------------------------------------------
    def rewrite_query(self, state: AgentState) -> dict:
        """Ask the model to turn the question (plus what is missing) into search calls."""

        prompt = rewrite_prompt(
            state["question"],
            state.get("queries_tried", []),
            state.get("observations", []),
            state.get("missing", ""),
        )
        reply, update = self._call_llm(state, "rewrite_query", prompt, self.toolbox.specs)
        if reply is None:
            return update

        done = set(state.get("call_signatures", []))
        pending: list[dict] = []
        repeated: list[str] = []
        for call in reply.tool_calls[:MAX_CALLS_PER_ROUND]:
            signature = call_signature(call.name, call.args)
            if signature in done or any(p["signature"] == signature for p in pending):
                repeated.append(call.name)
                continue
            pending.append({"id": call.id, "name": call.name, "args": call.args, "signature": signature})

        queries = [
            str(p["args"].get("query") or f"{p['name']}({json.dumps(p['args'], sort_keys=True)})")
            for p in pending
        ]
        events = []
        if repeated:
            events.append(f"repeated_call:{','.join(repeated)}")
        if not pending:
            events.append("no_new_calls")
        update.update(
            pending_calls=pending,
            rewritten_query=" ; ".join(queries) if queries else state.get("rewritten_query", ""),
            queries_tried=[*state.get("queries_tried", []), *queries],
            events=self._events(state, *events),
        )
        return update

    def retrieve(self, state: AgentState) -> dict:
        """Run the planned search calls with the existing retrievers. No model call."""

        chunks = {chunk["chunk_id"]: chunk for chunk in state.get("retrieved_chunks", [])}
        observations = list(state.get("observations", []))
        signatures = list(state.get("call_signatures", []))
        tools_used = list(state.get("tools_used", []))
        for planned in state.get("pending_calls", []):
            call = ToolCall(planned["id"], planned["name"], planned["args"])
            output, latency_ms = self.toolbox.execute(call)
            returned = output.get("chunks", [])
            new_ids = [chunk["chunk_id"] for chunk in returned if chunk["chunk_id"] not in chunks]
            for chunk in returned:
                chunks.setdefault(chunk["chunk_id"], chunk)
            signatures.append(planned["signature"])
            tools_used.append(call.name)
            observations.append(_observation(call.name, call.args, output, len(new_ids)))
            self.log.write(
                {
                    "run_id": state.get("run_id", ""),
                    "question_id": state.get("question_id", ""),
                    "type": "tool_call",
                    "node": "retrieve",
                    "call_id": call.id,
                    "tool": call.name,
                    "args": call.args,
                    "ok": "error" not in output,
                    "error": output.get("error"),
                    "chunk_ids": [chunk["chunk_id"] for chunk in returned],
                    "new_chunk_ids": new_ids,
                    "latency_ms": round(latency_ms, 1),
                }
            )
        return {
            "retrieved_chunks": list(chunks.values()),
            "observations": observations,
            "call_signatures": signatures,
            "tools_used": tools_used,
            "pending_calls": [],
        }

    def check_evidence(self, state: AgentState) -> dict:
        """Ask the model whether the evidence is enough, or what is still missing."""

        prompt = check_prompt(state["question"], state.get("retrieved_chunks", []), state.get("observations", []))
        reply, update = self._call_llm(state, "check_evidence", prompt, CHECK_TOOLS)
        if reply is None:
            return update

        decision = next((c for c in reply.tool_calls if c.name in (READY, SEARCH_AGAIN)), None)
        events = []
        if decision is None:  # no usable verdict: fail safe towards answering
            verdict, missing = "ready", ""
            events.append("check_without_decision")
        elif decision.name == SEARCH_AGAIN:
            verdict, missing = "search_again", str(decision.args.get("missing", ""))
            if not self._can_search_again(update["step_count"]):
                events.append("step_cap")
        else:
            verdict, missing = "ready", ""
        update.update(evidence_verdict=verdict, missing=missing, events=self._events(state, *events))
        return update

    def answer_or_abstain(self, state: AgentState) -> dict:
        """Ask for finish(answer, chunk_ids) or abstain(reason), then check the citations."""

        chunks = state.get("retrieved_chunks", [])
        prompt = answer_prompt(
            state["question"], chunks, state.get("observations", []), state.get("citation_feedback", "")
        )
        reply, update = self._call_llm(state, "answer_or_abstain", prompt, ANSWER_TOOLS)
        attempts = state.get("answer_attempts", 0) + 1
        update["answer_attempts"] = attempts
        if reply is None:
            return update

        decision = next((c for c in reply.tool_calls if c.name in (FINISH, ABSTAIN)), None)
        if decision is not None and decision.name == ABSTAIN:
            update.update(
                draft_answer=ABSTENTION_SENTENCE,
                citations=[],
                unsupported_citations=[],
                citation_feedback="",
                stop_reason="abstained",
            )
            return update

        if decision is None and not reply.text:
            update.update(draft_answer="", citations=[], unsupported_citations=[], stop_reason="no_answer")
            return update

        answer = str(decision.args.get("answer", "")) if decision else reply.text
        # Shared parser (eval/citations.py): the baseline is scored by the same rule,
        # including grouped ids such as [drug | section | id1, id2].
        listed = [i for raw in (decision.args.get("chunk_ids") or [] if decision else []) for i in split_ids(str(raw))]
        cited_ids = list(dict.fromkeys([*listed, *cited_chunk_ids(answer)]))

        # Citation check: every cited id must be a chunk a tool actually returned.
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        citations = [
            {"drug": by_id[i]["drug"], "section": by_id[i]["section"], "chunk_id": i}
            for i in cited_ids
            if i in by_id
        ]
        unsupported = [{"chunk_id": i} for i in cited_ids if i not in by_id]
        failed = bool(unsupported) or not citations
        feedback = ""
        events = []
        if failed:
            feedback = (
                f"cited ids not in the evidence: {[u['chunk_id'] for u in unsupported]}"
                if unsupported
                else "the answer cites no chunk id"
            )
            events.append("citation_check_failed")
        update.update(
            draft_answer=answer,
            citations=citations,
            unsupported_citations=unsupported,
            citation_feedback=feedback,
            stop_reason="answered",
            events=self._events(state, *events),
        )
        return update

    # -- routers (edges): pure functions of the state ------------------------------
    def route_after_rewrite(self, state: AgentState) -> str:
        if state.get("stop_reason") == "llm_error":
            return "error"
        return "new_calls" if state.get("pending_calls") else "all_repeated"

    def route_after_check(self, state: AgentState) -> str:
        if state.get("stop_reason") == "llm_error":
            return "error"
        if state.get("evidence_verdict") == "search_again" and self._can_search_again(state["step_count"]):
            return "search_again"
        return "enough_or_out_of_steps"

    def route_after_answer(self, state: AgentState) -> str:
        retry = (
            state.get("stop_reason") == "answered"
            and bool(state.get("citation_feedback"))
            and state.get("answer_attempts", 0) < MAX_ANSWER_ATTEMPTS
            and state.get("step_count", 0) < self.max_steps
        )
        return "fix_citations" if retry else "done"

    # -- run -------------------------------------------------------------------------
    def run(self, question: str, *, run_id: str = "", question_id: str = "") -> AgentResult:
        started = time.perf_counter()
        initial: AgentState = {
            "question": question,
            "run_id": run_id,
            "question_id": question_id,
            "step_count": 0,
            "events": [],
        }
        state = self.graph.invoke(initial, config={"recursion_limit": 25})
        usage = state.get("usage") or {}
        result = AgentResult(
            question=question,
            answer=state.get("draft_answer", ""),
            citations=state.get("citations", []),
            unsupported_citations=state.get("unsupported_citations", []),
            retrieved_chunks=state.get("retrieved_chunks", []),
            stop_reason=state.get("stop_reason", ""),
            events=state.get("events", []),
            rewritten_queries=state.get("queries_tried", []),
            n_steps=state.get("step_count", 0),
            n_tool_calls=len(state.get("tools_used", [])),
            tools_used=state.get("tools_used", []),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            thoughts_tokens=usage.get("thoughts_tokens", 0),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.llm.model,
            model_version=state.get("model_version", ""),
            all_cached=bool(state.get("all_cached")),
        )
        self.log.write(
            {
                "run_id": run_id,
                "question_id": question_id,
                "type": "final",
                "stop_reason": result.stop_reason,
                "events": result.events,
                "n_steps": result.n_steps,
                "n_tool_calls": result.n_tool_calls,
                "n_citations": len(result.citations),
                "n_unsupported_citations": len(result.unsupported_citations),
            }
        )
        return result


def run_agent(
    question: str,
    *,
    llm,
    toolbox: ToolBox,
    settings: GenerationSettings | None = None,
    max_steps: int = MAX_STEPS,
    log: JsonlLog | None = None,
    run_id: str = "",
    question_id: str = "",
) -> AgentResult:
    """Convenience wrapper: build the graph and answer one question."""

    graph = AgentGraph(llm=llm, toolbox=toolbox, settings=settings, max_steps=max_steps, log=log)
    return graph.run(question, run_id=run_id, question_id=question_id)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--log", default="results/runs/adhoc/agent_log.jsonl")
    parser.add_argument("--cache-mode", default=None, choices=["auto", "record", "replay", "off"])
    parser.add_argument("--mock", action="store_true", help="use the offline mock model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm = get_adapter(provider="mock" if args.mock else None, cache_mode=args.cache_mode)
    result = run_agent(
        args.query,
        llm=llm,
        toolbox=ToolBox(),
        settings=settings_from_env(),
        max_steps=args.max_steps,
        log=JsonlLog(Path(args.log)),
        run_id="adhoc",
    )
    print("=" * 78)
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print("-" * 78)
    print(f"queries: {' | '.join(result.rewritten_queries) or '(none)'}")
    print(f"tools: {' -> '.join(result.tools_used) or '(none)'}   events: {', '.join(result.events) or '-'}")
    print(
        f"steps={result.n_steps} stop={result.stop_reason} model={result.model_version or result.model} "
        f"cached={result.all_cached} latency_ms={result.latency_ms:.0f} "
        f"tokens(in/out/thought)={result.prompt_tokens}/{result.completion_tokens}/{result.thoughts_tokens}"
    )
    if result.unsupported_citations:
        print(f"WARNING: cites chunks no tool returned: {result.unsupported_citations}")
    print(f"log: {args.log}")
    print("NOTE: for information only - not medical advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
