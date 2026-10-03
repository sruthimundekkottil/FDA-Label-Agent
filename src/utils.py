"""Shared helpers: paths, config, text cleaning and openFDA section definitions.

This module deliberately avoids heavy third-party imports (no pandas, no torch)
so that every script can import it cheaply.
"""

from __future__ import annotations

import json
import logging
import os
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SECTIONS_JSONL = PROCESSED_DIR / "sections.jsonl"
FETCH_MANIFEST = RAW_DIR / "_fetch_manifest.json"
DRUGS_FILE = CONFIGS_DIR / "drugs.txt"
QUESTIONS_JSONL = EVAL_DIR / "questions.jsonl"

ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    EVAL_DIR,
    INDEX_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    CONFIGS_DIR,
]

# --------------------------------------------------------------------------- #
# openFDA label sections
# --------------------------------------------------------------------------- #
# Order matters: the first populated field wins. Fields that are merged into one
# canonical section are listed in preference order.
TARGET_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("boxed_warning", ("boxed_warning", "boxed_warnings")),
    ("indications_and_usage", ("indications_and_usage",)),
    ("dosage_and_administration", ("dosage_and_administration",)),
    ("dosage_forms_and_strengths", ("dosage_forms_and_strengths",)),
    ("contraindications", ("contraindications",)),
    ("warnings_and_cautions", ("warnings_and_cautions", "warnings")),
    ("adverse_reactions", ("adverse_reactions",)),
    ("drug_interactions", ("drug_interactions",)),
    ("use_in_specific_populations", ("use_in_specific_populations",)),
    ("pregnancy", ("pregnancy",)),
    ("pediatric_use", ("pediatric_use",)),
    ("geriatric_use", ("geriatric_use",)),
    ("overdosage", ("overdosage",)),
    ("description", ("description",)),
    ("clinical_pharmacology", ("clinical_pharmacology",)),
]

SECTION_NAMES: tuple[str, ...] = tuple(name for name, _ in TARGET_SECTIONS)


def is_table_field(field: str) -> bool:
    """openFDA stores some content as HTML-ish ``*_table`` fields."""

    return field.endswith("_table")


# --------------------------------------------------------------------------- #
# Config / env
# --------------------------------------------------------------------------- #
def load_env() -> None:
    """Load ``.env`` from the project root if python-dotenv is available."""

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in requirements.txt
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    """``"insulin glargine"`` -> ``"insulin_glargine"``."""

    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def load_drug_list(path: Path | None = None) -> list[str]:
    path = path or DRUGS_FILE
    drugs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drugs.append(line)
    return drugs


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def as_text(value: Any) -> str:
    """openFDA values are usually lists of strings; join them into one text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [as_text(item) for item in value]
        return "\n\n".join(part for part in parts if part.strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""

    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for source, replacement in _UNICODE_FIXES.items():
        text = text.replace(source, replacement)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Characters that survive HTML entities but hurt CSV output and tokenisation.
_UNICODE_FIXES = {
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero width space
    "\ufeff": "",    # BOM
    "\u2212": "-",   # minus sign
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": " - ",  # em dash
    "\u2015": " - ",  # horizontal bar
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
}


_SECTION_HEADING_RE = re.compile(r"^[\s\d\.\)\(]*$")


def strip_repeated_headings(text: str, section: str) -> str:
    """Drop lines that merely repeat the section heading.

    Drug labels often keep the section title (and its numbering) as the first
    line of the field value. It adds no information and hurts retrieval, so we
    remove lines that are only the heading or only numbering.
    """

    heading_key = re.sub(r"[^a-z]+", "", section.lower())
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _SECTION_HEADING_RE.match(stripped):
            continue
        key = re.sub(r"[^a-z]+", "", stripped.lower())
        if key and (key == heading_key or key == re.sub(r"[^a-z]+", "", heading_key + "s")):
            continue
        kept.append(stripped)
    return normalize_text("\n".join(kept))


def clean_section_text(raw: str, section: str) -> str:
    text = as_text(raw)
    # A few labels carry stray inline markup in normal sections; drop it.
    if _ANY_TAG_RE.search(text):
        text = strip_html_tags(text)
    return strip_repeated_headings(normalize_text(text), section)


# --------------------------------------------------------------------------- #
# HTML tables -> readable rows
# --------------------------------------------------------------------------- #
_ANY_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")


class _TableToRows(HTMLParser):
    """Collect caption, header row and data rows from (possibly nested) HTML tables.

    openFDA puts table content in HTML-ish ``*_table`` fields. Raw markup is
    useless for embeddings, so we recover the rows:

        <tr><th>Adverse reaction</th><th>Drug</th></tr>
        <tr><td>Nausea</td><td>25%</td></tr>

    becomes ``Adverse reaction: Nausea | Drug: 25%``.
    """

    LIST_BREAK_TAGS = {"li", "item", "listitem", "p", "div", "br"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.caption_parts: list[str] = []
        self.rows: list[list[str]] = []
        self.header_rows: list[bool] = []

        self._table_depth = 0
        self._in_caption = False
        self._row: list[str] | None = None
        self._row_is_header = False
        self._cell_parts: list[str] | None = None
        self._cell_is_header = False
        self._skip_text = False

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _clean(chunks: list[str]) -> str:
        text = " ".join(chunks)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"^(?:;\s*)+|(?:;\s*)+$", "", text).strip()

    # -- HTMLParser hooks -------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth == 0:
            return

        if tag == "caption":
            self._in_caption = True
            return

        if tag == "tr":
            if self._table_depth == 1:
                self._row = []
                self._row_is_header = True
            # Nested tables (depth > 1) keep flowing into the current cell.
            return

        if tag in {"td", "th"}:
            if self._table_depth == 1:
                self._cell_parts = []
                self._cell_is_header = tag == "th"
                if tag == "td":
                    self._row_is_header = False
            return

        if tag in self.LIST_BREAK_TAGS and self._cell_parts is not None:
            self._cell_parts.append("; ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth == 0:
            return

        if tag == "caption":
            self._in_caption = False
            return

        if tag in {"td", "th"}:
            if self._table_depth == 1 and self._cell_parts is not None and self._row is not None:
                self._row.append(self._clean(self._cell_parts))
            self._cell_parts = None
            return

        if tag == "tr":
            if self._table_depth == 1 and self._row is not None:
                if any(cell for cell in self._row):
                    self.rows.append(self._row)
                    self.header_rows.append(self._row_is_header and len(self._row) > 1)
                self._row = None
            return

    def handle_data(self, data: str) -> None:
        if self._table_depth == 0:
            return
        if self._in_caption:
            self.caption_parts.append(data)
        elif self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_entityref(self, name: str) -> None:  # pragma: no cover - convert_charrefs
        self.handle_data(unescape(f"&{name};"))


def strip_html_tags(text: str) -> str:
    """Remove inline markup, keeping the visible text."""

    if not text:
        return ""
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"(?i)</?(?:li|item|listitem|p|div|ul|ol|list)[^>]*>", "; ", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = unescape(text)
    return normalize_text(re.sub(r"\s*;\s*", "; ", text))


def html_table_to_text(html: str, *, max_header_chars: int = 60) -> str:
    """Convert an openFDA ``*_table`` field into readable row-wise text.

    Falls back to plain tag-stripped text when no rows can be recovered.
    """

    if not html or not html.strip():
        return ""

    parser = _TableToRows()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # pragma: no cover - malformed markup must never break parsing
        return strip_html_tags(html)

    caption = normalize_text(" ".join(parser.caption_parts).replace("\u00a0", " "))
    rows = parser.rows
    if not rows:
        return strip_html_tags(html)

    headers: list[str] | None = None
    data_rows = rows
    if parser.header_rows and parser.header_rows[0]:
        headers = rows[0]
        data_rows = rows[1:]
    elif len(rows) >= 3 and len(rows[0]) > 1:
        # Some tables use a plain first row as the header: short, label-like cells
        # that contain no digits or percentages.
        first = rows[0]
        if all(len(cell) <= max_header_chars and not re.search(r"\d", cell) for cell in first):
            headers = first
            data_rows = rows[1:]

    if headers is not None and any(len(cell) > max_header_chars for cell in headers):
        headers = None
        data_rows = rows

    lines: list[str] = []
    if caption:
        lines.append(caption)

    for row in data_rows:
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        if headers and 1 < len(cells) <= len(headers):
            pairs = [f"{headers[index]}: {cell}" for index, cell in enumerate(cells)]
            lines.append(" | ".join(pairs))
        elif headers and len(cells) == 1:
            # A single spanning cell (rowspan/colspan) is a sub-heading.
            lines.append(cells[0])
        else:
            lines.append(" | ".join(cells))

    text = "\n".join(lines)
    return normalize_text(text)


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation used by BM25 and overlap metrics."""

    return TOKEN_RE.findall(text.lower())


def truncate(text: str, limit: int = 240) -> str:
    text = normalize_text(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def detect_drug_in_query(query: str, drugs: Iterable[str]) -> str | None:
    """Very simple drug-name matcher: longest drug name appearing in the query."""

    query_lower = query.lower()
    best: str | None = None
    for drug in drugs:
        if drug.lower() in query_lower:
            if best is None or len(drug) > len(best):
                best = drug
    return best


# --------------------------------------------------------------------------- #
# openFDA label helpers
# --------------------------------------------------------------------------- #
def label_field_text(label: dict, field: str) -> str:
    """Return the cleaned text of one raw openFDA field (empty string if absent)."""

    if field not in label:
        return ""
    return normalize_text(as_text(label[field]))


def label_section_map(label: dict) -> dict[str, tuple[str, str]]:
    """Map canonical section name -> (source_field, text) for one label.

    * ``warnings_and_cautions`` falls back to the older ``warnings`` field.
    * Any additional ``*_table`` field is kept as its own section so the
      table-aware chunker can use it.
    """

    sections: dict[str, tuple[str, str]] = {}
    for section, candidates in TARGET_SECTIONS:
        for field in candidates:
            text = label_field_text(label, field)
            if text:
                sections[section] = (field, text)
                break
    for field, value in label.items():
        if is_table_field(field) and field not in {f for _, f in sections.values()}:
            text = label_field_text(label, field)
            if text:
                sections[field] = (field, text)
    return sections


def count_populated_sections(label: dict) -> int:
    """How many of the target sections this label actually populates."""

    return sum(
        1
        for section, candidates in TARGET_SECTIONS
        if any(label_field_text(label, field) for field in candidates)
    )


def label_total_chars(label: dict) -> int:
    return sum(len(text) for _, text in label_section_map(label).values())


def label_identity(label: dict) -> str:
    """A stable identifier for one label: ``set_id`` when present, else ``id``."""

    set_id = as_text(label.get("set_id")).strip()
    if set_id:
        return set_id
    return as_text(label.get("id")).strip()


def label_effective_time(label: dict) -> str:
    return as_text(label.get("effective_time")).strip()


def select_best_label(results: list[dict]) -> int:
    """Index of the label with the most populated sections (ties: longest text).

    openFDA returns up to ``limit`` labels for a generic name; older labels are
    often sparse. We keep the one that carries the most usable content.
    """

    if not results:
        raise ValueError("select_best_label() needs at least one label")

    best_index = 0
    best_key = (-1, -1)
    for index, label in enumerate(results):
        key = (count_populated_sections(label), label_total_chars(label))
        if key > best_key:
            best_key = key
            best_index = index
    return best_index
