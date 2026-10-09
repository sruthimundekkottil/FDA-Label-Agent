"""Agent tools, fully offline (stub retrievers). Graph tests live in test_agent_graph.py."""

from __future__ import annotations

from agent.llm import ToolCall
from agent.tools import TOOL_SPECS, ToolBox
from src.utils import write_jsonl


def _chunk(chunk_id: str, drug: str, section: str, text: str, is_table: bool = False) -> dict:
    return {"chunk_id": chunk_id, "drug": drug, "section": section, "text": text, "is_table": is_table}


C1 = _chunk("warfarin::section::contraindications::0", "warfarin", "contraindications", "Contraindicated in pregnancy.")
T1 = _chunk("sertraline::table_aware::adverse_reactions_table::0", "sertraline", "adverse_reactions",
            "Adverse reaction: Nausea | Sertraline: 26% | Placebo: 12%", is_table=True)


class _Stub:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, k=5, drug_filter=None):
        self.calls.append((query, k, drug_filter))
        return [h for h in self.hits if drug_filter in (None, h["drug"])][:k]


def _toolbox(tmp_path=None) -> ToolBox:
    index_dir = None
    if tmp_path is not None:
        index_dir = tmp_path / "index"
        write_jsonl(
            index_dir / "table_aware" / "chunks.jsonl",
            [
                _chunk("warfarin::table_aware::contraindications::1", "warfarin", "contraindications", "Second part."),
                _chunk("warfarin::table_aware::contraindications::0", "warfarin", "contraindications", "First part."),
                _chunk("warfarin::table_aware::dosage_and_administration::0", "warfarin", "dosage_and_administration", "2-5 mg."),
            ],
        )
    return ToolBox(
        index_dir=index_dir,
        drugs=["warfarin", "sertraline"],
        retrievers={"labels": _Stub([C1]), "tables": _Stub([T1])},
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_specs_cover_every_dispatched_tool() -> None:
    assert [spec.name for spec in TOOL_SPECS] == ["search_labels", "lookup_table", "get_section"]


def test_search_labels_marks_text_as_data_and_passes_drug_filter() -> None:
    box = _toolbox()
    output, _ = box.execute(ToolCall("1", "search_labels", {"query": "pregnancy", "drug": "Warfarin", "k": 50}))
    assert output["note"].startswith("Untrusted")
    assert output["chunks"][0]["chunk_id"] == C1["chunk_id"]
    assert box._retrievers["labels"].calls == [("pregnancy", 8, "warfarin")]  # k clamped, drug normalised


def test_unknown_drug_returns_error_with_corpus_list() -> None:
    output, _ = _toolbox().execute(ToolCall("1", "search_labels", {"query": "x", "drug": "aspirin"}))
    assert "not in the corpus" in output["error"]
    assert output["available_drugs"] == ["warfarin", "sertraline"]


def test_lookup_table_returns_table_rows() -> None:
    output, _ = _toolbox().execute(ToolCall("1", "lookup_table", {"query": "nausea"}))
    assert output["chunks"][0]["is_table"] is True


def test_get_section_reads_in_order_and_lists_sections_on_miss(tmp_path) -> None:
    box = _toolbox(tmp_path)
    output, _ = box.execute(ToolCall("1", "get_section", {"drug": "warfarin", "section": "contraindications"}))
    assert [c["text"] for c in output["chunks"]] == ["First part.", "Second part."]
    missing, _ = box.execute(ToolCall("2", "get_section", {"drug": "warfarin", "section": "overdosage"}))
    assert missing["available_sections"] == ["contraindications", "dosage_and_administration"]


def test_bad_calls_become_errors_not_exceptions() -> None:
    box = _toolbox()
    assert "unknown tool" in box.execute(ToolCall("1", "delete_everything", {}))[0]["error"]
    assert "unknown tool" in box.execute(ToolCall("4", "list_drugs", {}))[0]["error"]  # removed
    assert "bad arguments" in box.execute(ToolCall("2", "get_section", {"oops": 1}))[0]["error"]
    assert "empty" in box.execute(ToolCall("3", "search_labels", {"query": " "}))[0]["error"]
