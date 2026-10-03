"""Shared helpers: paths, config, text cleaning and openFDA section definitions.

This module deliberately avoids heavy third-party imports (no pandas, no torch)
so that every script can import it cheaply.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    return strip_repeated_headings(normalize_text(as_text(raw)), section)


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
