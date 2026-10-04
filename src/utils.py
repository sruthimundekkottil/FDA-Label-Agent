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
from dataclasses import dataclass, field
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


def parent_section_name(field: str) -> str:
    """``adverse_reactions_table`` -> ``adverse_reactions``.

    Table records are stored under their openFDA field name (``..._table``).
    Retrieval hit-checking works on canonical section names, so table chunks are
    labelled with the section they belong to. Unknown ``*_table`` fields keep
    their stripped name (they simply never match a gold section).
    """

    if not is_table_field(field):
        return field
    return field[: -len("_table")]


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


def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_questions(
    path: Path | str | None = None,
    *,
    verified_only: bool = False,
) -> tuple[list[dict], dict]:
    """Load the evaluation questions plus a small stats dict.

    Shared by every evaluation script so that ``--verified-only`` means the same
    thing everywhere and each run can state exactly how many questions it used.
    """

    records = read_jsonl(path or QUESTIONS_JSONL)
    verified = [record for record in records if record.get("verified")]
    selected = verified if verified_only else records
    stats = {
        "total": len(records),
        "verified": len(verified),
        "unverified": len(records) - len(verified),
        "selected": len(selected),
        "verified_only": verified_only,
    }
    return selected, stats


def write_jsonl(path: Path | str, records: Iterable[dict]) -> int:
    path = Path(path)
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
# HTML tables -> readable rows (BeautifulSoup + lxml)
# --------------------------------------------------------------------------- #
_ANY_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_SEPARATOR_BLOCKS = {"p", "div", "li", "item", "listitem", "caption"}


def _soup(markup: str):
    """Parse HTML with lxml when available, falling back to the stdlib parser."""

    from bs4 import BeautifulSoup

    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(markup, parser)
        except Exception:  # pragma: no cover - only if lxml is missing/broken
            continue
    raise RuntimeError("no usable HTML parser")  # pragma: no cover


def _node_text(node) -> str:
    """Visible text of a node, flattening lists into ``; ``-separated text."""

    from bs4 import NavigableString

    if isinstance(node, NavigableString):
        return str(node)
    name = getattr(node, "name", "") or ""
    inner = "".join(_node_text(child) for child in node.children)
    if name == "br":
        return " "
    if name in _SEPARATOR_BLOCKS:
        return f"; {inner}"
    return inner


def _clean_cell(raw: str) -> str:
    text = unescape(raw).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"(?:;\s*){2,}", "; ", text)
    return re.sub(r"^(?:;\s*)+|(?:\s*;)+$", "", text).strip()


def strip_html_tags(text: str) -> str:
    """Remove markup from a field that is not a table, keeping the visible text."""

    if not text:
        return ""
    if not _ANY_TAG_RE.search(text):
        return normalize_text(text)
    try:
        soup = _soup(text)
        return normalize_text(soup.get_text(" "))
    except Exception:  # pragma: no cover - never let cleaning crash the pipeline
        return normalize_text(_ANY_TAG_RE.sub(" ", unescape(text)))


@dataclass
class ParsedTable:
    """Structured view of one openFDA ``*_table`` field.

    ``rows`` holds one formatted line per table row, which is what makes it safe
    for the table-aware chunker: a row is never split, because chunk boundaries
    only ever fall between lines.
    """

    caption: str = ""
    header: list[str] = field(default_factory=list)
    rows: list[str] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        lines = [line for line in [self.caption, *self.context_lines, *self.rows] if line]
        return normalize_text("\n".join(lines))


_MEASUREMENT_RE = re.compile(
    r"(?:"
    r"\d\s*%"                                   # 26%
    r"|\d\s*\(\d+(?:\.\d+)?\)"                  # 7 (4)
    r"|\d\s*(?:mg|ml|mcg|g|iu|units?|hours?|days?|weeks?|years?|tablets?|capsules?)\b"
    r"|^<?\d+(?:[.,]\d+)?$"                     # a bare number
    r"|\bn\s*=\s*\d+"                           # N = 176
    r")",
    re.IGNORECASE,
)


def _cell_is_measurement(value: str) -> bool:
    """True when a cell looks like a measured value rather than a column label."""

    return bool(_MEASUREMENT_RE.search(value.strip()))


def parse_table(markup: str) -> ParsedTable:
    """Parse an openFDA table field into caption + header + one line per row."""

    table = ParsedTable()
    if not markup or not markup.strip():
        return table

    try:
        soup = _soup(markup)
    except Exception:  # pragma: no cover
        table.rows = [line for line in strip_html_tags(markup).split("\n") if line]
        return table

    top_table = soup.find("table")
    if top_table is None:
        table.rows = [line for line in strip_html_tags(markup).split("\n") if line]
        return table

    caption_tag = top_table.find("caption")
    if caption_tag is not None:
        table.caption = _clean_cell(_node_text(caption_tag))

    # Only rows of the outermost table: nested tables inside a cell are flattened
    # into that cell's text instead of becoming rows of their own.
    trs = [tr for tr in top_table.find_all("tr") if tr.find_parent("table") is top_table]

    indexed_rows: list[list[tuple[int, str]]] = []
    header_rows: list[dict[int, str]] = []
    pending: dict[int, tuple[str, int]] = {}  # column -> (value, rows left)

    for tr in trs:
        cells = tr.find_all(["td", "th"], recursive=False)
        if not cells:
            continue

        entries: list[tuple[int, str]] = []
        column = 0
        is_header_row = True
        for cell in cells:
            while column in pending:
                value, remaining = pending[column]
                if remaining > 0:
                    entries.append((column, value))
                if remaining <= 1:
                    pending.pop(column, None)
                else:
                    pending[column] = (value, remaining - 1)
                column += 1

            value = _clean_cell(_node_text(cell))
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
            except (TypeError, ValueError):
                colspan = 1
            try:
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except (TypeError, ValueError):
                rowspan = 1

            if cell.name != "th":
                is_header_row = False
            if value:
                entries.append((column, value))
                if rowspan > 1:
                    pending[column] = (value, rowspan - 1)
            column += colspan

        # Flush any still-pending rowspan values that belong to trailing columns.
        for pending_column in sorted(pending):
            if pending_column >= column:
                value, remaining = pending[pending_column]
                if remaining > 0:
                    entries.append((pending_column, value))
                if remaining <= 1:
                    pending.pop(pending_column, None)
                else:
                    pending[pending_column] = (value, remaining - 1)

        if not entries:
            continue
        indexed_rows.append(sorted(entries))
        if is_header_row:
            header_rows.append({column: value for column, value in entries})

    if not indexed_rows:
        table.rows = [line for line in strip_html_tags(markup).split("\n") if line]
        return table

    # Leading single-cell rows are titles/sub-headings (spanning rows), not data.
    lead = 0
    while lead < len(indexed_rows) and len(indexed_rows[lead]) == 1:
        table.context_lines.append(indexed_rows[lead][0][1])
        lead += 1

    def _looks_like_values(entries: list[tuple[int, str]]) -> bool:
        values = [value for _, value in entries]
        if len(values) < 2:
            return False
        measured = sum(1 for value in values if _cell_is_measurement(value))
        return measured >= max(1, len(values) // 2)

    def _is_labelish(entries: list[tuple[int, str]]) -> bool:
        """Short, non-measurement cells - column labels such as genotype codes."""

        values = [value for _, value in entries]
        return bool(values) and all(
            len(value) <= 60 and not _cell_is_measurement(value) for value in values
        )

    # Column -> header label. Column indices (not positions) are used on purpose:
    # openFDA tables pad the header row with empty corner cells, so positional
    # matching would shift every label by one column.
    header: dict[int, str] = {}
    n_header_rows = 0
    if header_rows:
        header = dict(header_rows[-1])
        n_header_rows = len(header_rows)
        for coarse in header_rows[:-1]:
            if set(coarse) != set(header):
                table.context_lines.append(" / ".join(value for _, value in sorted(coarse.items())))
    else:
        # No <th> at all: treat the first multi-cell row as a header when a later
        # row looks like data. Header cells that carry numbers (dose levels such
        # as "Sitagliptin 100 mg once daily") are also kept as a data line, so a
        # wrong guess cannot silently delete content.
        candidate = next(
            (index for index in range(lead, len(indexed_rows)) if len(indexed_rows[index]) > 1),
            None,
        )
        if candidate is not None and len(indexed_rows) - candidate >= 3:
            next_data = next(
                (
                    index
                    for index in range(candidate + 1, len(indexed_rows))
                    if _looks_like_values(indexed_rows[index])
                ),
                None,
            )
            # A header is a label row; very long cells mean we grabbed a data or
            # title row instead, so leave the table unlabelled in that case.
            plausible = all(len(value) <= 120 for _, value in indexed_rows[candidate])
            if next_data is not None and plausible:
                header = dict(indexed_rows[candidate])
                n_header_rows = candidate + 1
                # Grouped header: a coarse row ("VKORC1 | CYP2C9") followed by a
                # finer row with more columns ("*1/*1 | *1/*2 | ..."). The finer
                # row is the real column header.
                follow = candidate + 1
                if (
                    follow < len(indexed_rows)
                    and len(indexed_rows[follow]) > len(indexed_rows[candidate])
                    and _is_labelish(indexed_rows[candidate])
                    and _is_labelish(indexed_rows[follow])
                ):
                    header = dict(indexed_rows[follow])
                    n_header_rows = follow + 1
                    table.context_lines.append(
                        " / ".join(value for _, value in indexed_rows[candidate])
                    )
                if any(re.search(r"\d", value) for value in header.values()):
                    table.context_lines.append(
                        " | ".join(value for _, value in indexed_rows[candidate])
                    )

    header = {
        column: (value if len(value) <= 80 else value[:79].rstrip() + "…")
        for column, value in header.items()
    }

    # Column alignment: several openFDA tables leave the row-label corner empty,
    # so the header can sit one (or more) columns left of the data it describes.
    data_columns = [
        max(column for column, _ in entries)
        for entries in indexed_rows[max(n_header_rows, lead):]
        if len(entries) > 1
    ]
    if header and data_columns:
        shift = max(data_columns) - max(header)
        if shift != 0:
            header = {column + shift: value for column, value in header.items()}

    table.header = [value for _, value in sorted(header.items())]

    rows: list[str] = []
    for entries in indexed_rows[max(n_header_rows, lead):]:
        if not entries:
            continue
        if len(entries) == 1:
            # A spanning cell: a sub-heading or a footnote, never a labelled row.
            rows.append(entries[0][1])
            continue
        parts: list[str] = []
        for column, value in entries:
            label = header.get(column)
            parts.append(f"{label}: {value}" if label else value)
        if parts:
            rows.append(" | ".join(parts))
    table.rows = rows
    return table


def html_table_to_text(markup: str, *, max_header_chars: int = 60) -> str:
    """Readable row-wise text for an openFDA ``*_table`` field."""

    return parse_table(markup).text


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
