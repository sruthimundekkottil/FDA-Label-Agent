"""Draft candidate evaluation questions from the processed label sections.

    python -m src.draft_eval_questions

Every drafted question is written with ``"verified": false``. A human must review
the question, the ``gold_section`` and the ``gold_answer_keypoints`` and flip the
flag to ``true`` before the question is used for any reported metric.

Question mix (spec section 10): ~15 factual, ~6 negation, ~6 table, ~5
abbreviation, ~4 unanswerable and ~4 multi-section / cross-drug questions.
Keypoints are verbatim snippets from the label text - the draft never invents
facts, it only selects and truncates them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from src.utils import (
    QUESTIONS_JSONL,
    SECTIONS_JSONL,
    get_logger,
    is_table_field,
    normalize_text,
    parent_section_name,
    parse_table,
    read_jsonl,
    tokenize,
    truncate,
    write_jsonl,
)

logger = get_logger("draft_eval_questions")

TARGET_COUNTS = {
    "factual": 15,
    "negation": 6,
    "table": 6,
    "abbreviation": 5,
    "unanswerable": 4,
    "multi-section": 2,
    "cross-drug": 2,
}

FACTUAL_PLAN: list[tuple[str, str]] = [
    ("boxed_warning", "What does the boxed warning for {drug} say?"),
    ("contraindications", "What are the contraindications of {drug}?"),
    ("indications_and_usage", "What is {drug} indicated for?"),
    ("dosage_and_administration", "What is the recommended dosage of {drug}?"),
    ("adverse_reactions", "What adverse reactions are reported in the {drug} label?"),
    ("drug_interactions", "What drug interactions does the {drug} label list?"),
]

NEGATION_PLAN: list[tuple[str, str]] = [
    ("contraindications", "Which patients should NOT take {drug}?"),
    ("contraindications", "In which situations is {drug} contraindicated?"),
    ("warnings_and_cautions", "What does the label warn patients not to do while taking {drug}?"),
    ("warnings_and_cautions", "Which patients should avoid {drug} according to the label?"),
    ("contraindications", "What conditions rule out treatment with {drug}?"),
    ("warnings_and_cautions", "What does the label say about who should not use {drug}?"),
]

NEGATION_CUES = (
    "contraindicat",
    "should not",
    "not recommended",
    "must not",
    "do not",
    "avoid",
    "exclude",
)

ABBREVIATIONS: list[tuple[str, str]] = [
    ("INR", "international normalized ratio"),
    ("CrCl", "creatinine clearance"),
    ("MAOI", "monoamine oxidase inhibitor"),
    ("QT", "QT interval"),
    ("NSAID", "nonsteroidal anti-inflammatory drug"),
    ("CYP2C9", "cytochrome P450 2C9"),
    ("eGFR", "estimated glomerular filtration rate"),
    ("HbA1c", "glycated hemoglobin"),
]

# Where an abbreviation shows up first says a lot about whether the question is
# about that drug at all (INR belongs to warfarin, not to a statin).
ABBREVIATION_SECTION_PRIORITY = [
    "boxed_warning",
    "contraindications",
    "dosage_and_administration",
    "warnings_and_cautions",
    "use_in_specific_populations",
    "drug_interactions",
    "adverse_reactions",
    "overdosage",
    "clinical_pharmacology",
    "description",
]

FOOTNOTE_MARKERS = "†‡§¶*"

# Row labels that are really column/table furniture rather than a data point.
GENERIC_ROW_LABELS = {
    "placebo",
    "drug",
    "active control",
    "adverse reactions",
    "adversereactions",
    "adverse reaction",
    "meddra preferred term",
    "preferred term",
    "system organ class",
    "body system",
    "number of patients (%)",
    "patients",
    "total",
    "incidence",
    "table",
    "study",
    "parameter",
    "comparator",
}

UNANSWERABLE_QUESTIONS = [
    "What is the retail price of a 30-day supply of {drug}?",
    "Which insurance plans cover {drug}?",
    "What is the FDA approval date of {drug}?",
    "Does {drug} cure the condition it is used to treat?",
    "Which pharmacy chain has {drug} in stock today?",
    "How many people in the United States took {drug} last year?",
]


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text).replace("\n", " "))
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def has_abbreviation(text: str, abbreviation: str) -> bool:
    return re.search(rf"\b{re.escape(abbreviation)}\b", text) is not None


# --------------------------------------------------------------------------- #
# Keypoint selection
#
# Drafted keypoints must answer *the question*, not just be the first sentences
# of the section. Label text is full of boilerplate (reporting lines, stewardship
# statements, "discussed in greater detail elsewhere", trial caveats) and of
# section headings, so the selector below:
#
#   1. splits the section into candidate *units* (paragraphs, bullets, sentences),
#   2. drops boilerplate, headings and cross-reference-only fragments,
#   3. scores the rest against the content words of the question, and
#   4. when the best unit introduces a list ("indicated for the following:",
#      "... were:"), pulls in the actual items as their own keypoint.
# --------------------------------------------------------------------------- #
QUESTION_STOPWORDS = {
    "a", "about", "according", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "both", "by", "can", "describe", "did", "do", "does", "during", "each", "for", "from",
    "has", "have", "how", "if", "in", "include", "includes", "including", "into", "is", "it",
    "its", "label", "labels", "list", "listed", "may", "must", "not", "of", "on", "or", "per",
    "please", "report", "reported", "reporting", "says", "see", "should", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "use", "used", "using", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "within", "would", "patient", "patients",
}

# Sentences that carry no answer value for a label question.
BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"to report suspected adverse reactions",
        r"suspected adverse reactions",
        r"\bmedwatch\b",
        r"1-?800-?fda-?1088",
        r"\bfda at\s*1-?8",
        r"\bwww\.[a-z0-9.\-/]+",
        r"https?://",
        r"\bstewardship\b",
        r"drug-resistant bacteria",
        r"susceptibility patterns",
        r"culture and susceptibility",
        r"susceptibility information",
        r"modifying antibacterial therapy",
        r"empiric(?:al)? (?:selection|therapy)",
        r"proven or strongly suspected",
        r"clinical trials are conducted under widely varying conditions",
        r"directly compared to (?:the )?rates",
        r"may not reflect the rates observed in practice",
        r"discussed in greater detail in other sections",
        r"described in greater detail",
        r"in (?:greater|more) detail",
        r"in other sections",
        r"^for information about",
        r"\brefer to\b",
        r"see full prescribing information",
        r"please (?:see|refer)",
        r"^keep (?:this|out of the reach)",
        r"manufactured (?:by|for)",
        r"distributed by",
        r"^\s*revised\s*:",
        r"all rights reserved",
        r"^(?:table|figure)\s*\d*[.:]?$",
        r"you (?:are|may) (?:encouraged|report)",
        r"^\s*\(?\s*\d+(?:\.\d+)*\s*\)?\s*$",
    )
]

LIST_SECTIONS = {
    "indications_and_usage",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
    "boxed_warning",
    "warnings_and_cautions",
}

_BULLET_RE = re.compile(r"[•·●▪]")
NOISE_UNITS = {"none", "total", "n/a", "not applicable", "yes", "no", "and", "or", "the", "table"}
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9(\[])"
    r"|(?<=[\]\)])\s+(?=[A-Z][a-z])"   # after a citation, e.g. "... (5.1) ] Most common ..."
    r"|(?<=:)\s+(?=[A-Z])"             # list intro, e.g. "... were: Postherpetic neuralgia: ..."
)
_REF_ONLY_RE = re.compile(r"[\[(]\s*(?:see\b[^\]()]*|\d+(?:\.\d+)*\s*(?:,\s*\d+(?:\.\d+)*)*)\s*[\])]", re.I)
_TRAILING_REF_RE = re.compile(r"\s*[\[(]\s*\d+(?:\.\d+)*\s*[\])]\s*$")
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*")


def _strip_leading_allcaps(unit: str) -> str:
    """Drop a leading ALL-CAPS heading run: "INDICATIONS AND USAGE Gabapentin is ..."."""

    words = unit.split()
    index = 0
    while index < len(words) and words[index].isupper() and any(c.isalpha() for c in words[index]):
        index += 1
    if 0 < index <= 7 and index < len(words):
        return " ".join(words[index:])
    return unit


def content_terms(text: str, drug: str = "") -> set[str]:
    """Content words of a question, without stopwords or the drug name."""

    tokens = set(tokenize(text)) - QUESTION_STOPWORDS
    for part in re.split(r"[^a-z0-9]+", drug.lower()):
        tokens.discard(part)
    return {token for token in tokens if len(token) >= 2}


def section_terms(section: str) -> set[str]:
    """Words taken from the section name itself ("adverse_reactions", ...)."""

    terms: set[str] = set()
    for part in re.split(r"[^a-z0-9]+", (section or "").lower()):
        if not part:
            continue
        terms.add(part)
        if part.endswith("s"):
            terms.add(part[:-1])
    return terms


def specific_question_terms(question: str, drug: str, section: str) -> set[str]:
    """Question terms that say more than the section name already does.

    For "What are the contraindications of X?" every sentence in the
    contraindications section is a candidate, so the section words carry no
    information: the section text itself (in document order) is the answer.
    For "What percentage of patients reported nausea?" the specific words
    ("percentage", "nausea") are what actually select the answer sentence.
    """

    section_stems = set()
    for term in section_terms(section):
        section_stems.add(term[:6] if len(term) >= 6 else term)

    specific: set[str] = set()
    for term in content_terms(question, drug):
        stem = term[:6] if len(term) >= 6 else term
        if stem not in section_stems:
            specific.add(term)
    return specific


def clean_unit(unit: str) -> str:
    unit = _BULLET_RE.sub(" ", unit)
    unit = re.sub(r"^\s*[-*]\s+", "", unit)
    unit = _LEADING_NUMBER_RE.sub("", unit)
    unit = _strip_leading_allcaps(unit)
    unit = re.sub(r"\s+", " ", unit)
    return unit.strip(" ;,-")  # keep a trailing ':' so list intros stay detectable


def split_units(text: str) -> list[str]:
    """Paragraphs, bullets and sentences as separate candidate units."""

    text = normalize_text(text)
    text = text.replace(" .", ".").replace(" ,", ",").replace(" ;", ";")
    text = _BULLET_RE.sub("\n• ", text)
    text = re.sub(r"\s+(?=\d+\.\d+\s+[A-Z])", "\n", text)  # numbered subsections
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for piece in _SENTENCE_SPLIT_RE.split(line):
            cleaned = clean_unit(piece)
            # Short bullets are legitimate list items ("Pregnancy", "Sinusitis").
            if len(cleaned) >= 4 and cleaned.lower() not in NOISE_UNITS:
                units.append(cleaned)
    return units


def is_boilerplate(unit: str) -> bool:
    if any(pattern.search(unit) for pattern in BOILERPLATE_PATTERNS):
        return True
    # headings such as "6 ADVERSE REACTIONS" or "6.1 Clinical Trials Experience"
    if _LEADING_NUMBER_RE.match(unit) and len(unit.split()) <= 6 and not unit.endswith((".", "!", "?")):
        return True
    words = unit.split()
    if (
        len(words) <= 6
        and not unit.endswith((".", "!", "?"))
        and all(word.isupper() for word in words if any(c.isalpha() for c in word))
    ):
        return True
    # cross-reference-only fragments ("[see Warnings and Precautions (5.1)]",
    # "( 4 )", "see Section 5.1") - but a short bullet with a trailing citation
    # such as "Sinusitis ( 1.3 )" is real content and must survive.
    if _REF_ONLY_RE.search(unit):
        stripped = _REF_ONLY_RE.sub(" ", unit).strip(" .;,")
        if len(stripped) < 15:
            return True
        if re.match(r"^(?:\[?\s*see\b|\()", unit, re.IGNORECASE) and len(stripped) < 40:
            return True
    return False


def _has_list_intro(unit: str) -> bool:
    if unit.rstrip().endswith(":"):
        return True
    return bool(
        re.search(
            r"\b(?:the following|were|are|include[sd]?|indicated for|contraindicated in|"
            r"consists? of|such as|including)\b[^.]{0,60}:",
            unit,
            re.IGNORECASE,
        )
    )


def _score_unit(
    unit: str,
    question_terms: set[str],
    *,
    section: str,
    cues: tuple[str, ...],
    wants_numbers: bool,
) -> float:
    # Score on the unit's own words: citation text such as
    # "[see Warnings and Precautions ]" must not create keyword matches.
    terms = set(tokenize(_REF_ONLY_RE.sub(" ", unit))) - QUESTION_STOPWORDS
    score = 2.0 * _term_overlap(terms, question_terms)
    lower = unit.lower()
    if cues and any(cue in lower for cue in cues):
        score += 1.5
    if section in LIST_SECTIONS and _has_list_intro(unit):
        score += 1.0
    if wants_numbers and re.search(r"\d", unit):
        score += 0.5
    if len(unit) < 30:
        score -= 1.0
    return score


def _term_overlap(unit_terms: set[str], question_terms: set[str]) -> int:
    """Count question terms present in the unit, allowing shared word stems.

    Label wording rarely matches the question exactly ("pregnancy" in the
    question vs "pregnant" in the label), so terms count as a match when they
    share a six-character stem.
    """

    matches = 0
    for question_term in question_terms:
        for unit_term in unit_terms:
            if unit_term == question_term or (
                len(question_term) >= 6
                and len(unit_term) >= 6
                and unit_term[:6] == question_term[:6]
            ):
                matches += 1
                break
    return matches


def _expand_list(units: list[str], index: int, limit: int, max_groups: int = 2) -> tuple[list[str], set[int]]:
    """Collect the items that follow a list intro, packed into keypoint-sized groups.

    "Amoxicillin is indicated for the following:" + five bullet infections turns
    into one keypoint holding all five items; a longer list (gabapentin's adverse
    reactions per indication) spills into a second keypoint instead of being
    cut off.
    """

    groups: list[str] = []
    current: list[str] = []
    consumed: set[int] = set()
    pending_intro: str | None = None
    skipped_intros = 0
    for position in range(index + 1, len(units)):
        item = _TRAILING_REF_RE.sub("", units[position]).strip(" .;,")
        if not item:
            break
        if _has_list_intro(item):
            # A repeat of the intro, or a sub-heading such as "Postherpetic
            # neuralgia:" whose items follow - skip it and keep collecting.
            if skipped_intros < 2:
                # Short sub-headings prefix their items ("Postherpetic neuralgia: ...");
                # a repeated long intro is simply dropped.
                pending_intro = item.rstrip(":").strip() if len(item.split()) <= 8 else None
                skipped_intros += 1
                consumed.add(position)
                continue
            break
        if len(item) > 120:
            # A long item (a label that glues several list entries into one
            # sentence) still becomes its own keypoint rather than being lost.
            if not current and groups == [] and len(groups) < max_groups:
                groups.append(item[:limit])
                consumed.add(position)
            break
        if pending_intro:
            item = f"{pending_intro}: {item}"
            pending_intro = None
        candidate = "; ".join([*current, item])
        if current and len(candidate) > limit:
            groups.append("; ".join(current))
            current = []
            if len(groups) >= max_groups:
                break
            candidate = item
        if len(candidate) > limit:
            groups.append(candidate[:limit])
            consumed.add(position)
            break
        current.append(item)
        consumed.add(position)
    if current and len(groups) < max_groups:
        groups.append("; ".join(current))
    return groups, consumed


def select_keypoints(
    text: str,
    question: str,
    *,
    section: str = "",
    question_type: str = "factual",
    cues: tuple[str, ...] = (),
    drug: str = "",
    n: int = 3,
    limit: int = 170,
) -> list[str]:
    """Keypoints that answer ``question``, taken verbatim from the section text."""

    if question_type == "table":
        # A table row already is the answer; keep it whole and do not glue rows.
        n = 1
        limit = max(limit, 600)

    units: list[str] = []
    seen: set[str] = set()
    for unit in split_units(text):
        if is_boilerplate(unit):
            continue
        key = re.sub(r"\W+", " ", unit.lower()).strip()
        if key in seen:
            continue  # label tables often repeat a heading verbatim
        seen.add(key)
        units.append(unit)
    if not units:
        fallback = split_sentences(text)
        return [truncate(item, limit) for item in fallback[:n]] or [truncate(text, limit)]

    # Two-tier scoring: words that only repeat the section name ("contraindications"
    # in a contraindications question) do not discriminate between candidates.
    question_terms = specific_question_terms(question, drug, section)
    wants_numbers = bool(
        re.search(r"\b(?:percentage|percent|how many|how much|what dose|incidence|rate)\b", question, re.I)
    )
    if question_type == "table":
        # The question names the row it asks about, so the row label matters most.
        scores = []
        for unit in units:
            label = unit.split(" | ")[0]
            clean_unit_text = _REF_ONLY_RE.sub(" ", unit)
            label_terms = set(tokenize(_REF_ONLY_RE.sub(" ", label))) - QUESTION_STOPWORDS
            row_terms = set(tokenize(clean_unit_text)) - QUESTION_STOPWORDS
            scores.append(
                2.0 * _term_overlap(label_terms, question_terms)
                + 1.0 * _term_overlap(row_terms, question_terms)
            )
    else:
        scores = [
            _score_unit(
                unit,
                question_terms,
                section=section,
                cues=cues,
                wants_numbers=wants_numbers,
            )
            for unit in units
        ]
    ranked = sorted(range(len(units)), key=lambda position: (-scores[position], position))
    discriminative = scores[ranked[0]] > 0
    if question_type == "table" and not discriminative:
        # No row label matched the question: fall back to the first row that
        # actually carries values instead of a caption-like first row.
        for position in ranked:
            if re.search(r"\d", units[position]):
                ranked = [position, *[item for item in ranked if item != position]]
                break

    chosen: list[str] = []
    consumed: set[int] = set()
    best = ranked[0]
    expanded = False
    if question_type != "table" and section in LIST_SECTIONS and _has_list_intro(units[best]):
        chosen.append(units[best])
        consumed.add(best)
        groups, item_indices = _expand_list(units, best, limit, max_groups=max(1, n - 1))
        if groups:
            chosen.extend(groups)
            consumed |= item_indices
            expanded = True

    for position in ranked:
        if len(chosen) >= n:
            break
        if position in consumed:
            continue
        # Once a list has supplied the answer, only add another unit if it scores
        # *better* than the list intro - otherwise it is unrelated filler.
        if expanded and scores[position] <= scores[best]:
            continue
        if discriminative and chosen and scores[position] <= 0:
            break  # nothing left that actually relates to the question
        if _too_similar(units[position], chosen):
            continue  # near-duplicate intro sentence or repeated label sentence
        chosen.append(units[position])
        consumed.add(position)

    return [truncate(keypoint, limit) for keypoint in chosen[:n]]


def _too_similar(candidate: str, chosen: list[str], threshold: float = 0.8) -> bool:
    """True when ``candidate`` repeats most of a keypoint already chosen."""

    candidate_terms = set(tokenize(candidate))
    if not candidate_terms:
        return True
    for item in chosen:
        item_terms = set(tokenize(item))
        if not item_terms:
            continue
        overlap = len(candidate_terms & item_terms) / max(1, min(len(candidate_terms), len(item_terms)))
        if overlap >= threshold:
            return True
    return False


def keypoints_with_cue(
    text: str,
    cues: tuple[str, ...],
    n: int = 2,
    *,
    question: str = "",
    section: str = "",
    question_type: str = "factual",
    drug: str = "",
) -> list[str]:
    """Backwards-compatible wrapper: cue-bearing sentences are scored up."""

    return select_keypoints(
        text,
        question,
        section=section,
        question_type=question_type,
        cues=cues,
        drug=drug,
        n=n,
    )


# --------------------------------------------------------------------------- #
# Drafting
# --------------------------------------------------------------------------- #
def load_corpus(path: Path) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    records = read_jsonl(path)
    sections: dict[str, dict[str, dict]] = {}
    tables: list[dict] = []
    for record in records:
        if is_table_field(record["source_field"]):
            tables.append(record)
        else:
            sections.setdefault(record["drug"], {})[record["section"]] = record
    return sections, tables


def _draft_factual(sections, drugs, rng, used) -> list[dict]:
    pools = {
        section: [drug for drug in drugs if section in sections.get(drug, {})]
        for section, _ in FACTUAL_PLAN
    }
    for pool in pools.values():
        rng.shuffle(pool)

    out: list[dict] = []
    index = 0
    while len(out) < TARGET_COUNTS["factual"] and index < 200:
        section, template = FACTUAL_PLAN[index % len(FACTUAL_PLAN)]
        index += 1
        pool = pools[section]
        while pool:
            drug = pool.pop()
            if (drug, section) in used:
                continue
            used.add((drug, section))
            record = sections[drug][section]
            question_text = template.format(drug=drug)
            out.append(
                {
                    "question": question_text,
                    "drug": drug,
                    "gold_section": section,
                    "gold_answer_keypoints": select_keypoints(
                        record["text"],
                        question_text,
                        section=section,
                        question_type="factual",
                        drug=drug,
                        n=3,
                    ),
                    "question_type": "factual",
                }
            )
            break
    return out


def _draft_negation(sections, drugs, rng, used) -> list[dict]:
    out: list[dict] = []
    index = 0
    while len(out) < TARGET_COUNTS["negation"] and index < 200:
        section, template = NEGATION_PLAN[len(out) % len(NEGATION_PLAN)]
        index += 1
        candidates = [drug for drug in drugs if section in sections.get(drug, {})]
        rng.shuffle(candidates)
        for drug in candidates:
            if (drug, section) in used:
                continue
            used.add((drug, section))
            record = sections[drug][section]
            question_text = template.format(drug=drug)
            out.append(
                {
                    "question": question_text,
                    "drug": drug,
                    "gold_section": section,
                    "gold_answer_keypoints": select_keypoints(
                        record["text"],
                        question_text,
                        section=section,
                        question_type="negation",
                        cues=NEGATION_CUES,
                        drug=drug,
                        n=3,
                    ),
                    "question_type": "negation",
                }
            )
            break
    return out


def _draft_table(tables, rng) -> list[dict]:
    out: list[dict] = []
    adverse = [record for record in tables if record["section"] == "adverse_reactions_table"]
    rng.shuffle(adverse)

    for record in adverse:
        if len(out) >= TARGET_COUNTS["table"]:
            break
        parsed = parse_table(record["text"])
        candidates: list[tuple[str, str]] = []
        header_labels = {value.strip().lower().replace(" ", "") for value in parsed.header}
        for row in parsed.rows:
            if row in parsed.context_lines or " | " not in row:
                continue
            label = row.split(" | ")[0].split(":")[0].strip().strip(FOOTNOTE_MARKERS).strip()
            if not label or len(label) > 60 or len(label.split()) > 6:
                continue
            if any(character in label for character in "%=") or re.search(r"\d", label):
                continue
            if label.lower().replace(" ", "") in header_labels:
                continue  # a column header leaked into the rows
            if label.lower() == record["drug"].lower():
                continue  # the drug name as a row label is a header artefact
            if label.lower() in GENERIC_ROW_LABELS:
                continue
            if len(re.findall(r"\d", row)) < 2:
                continue
            candidates.append((label, row))
        if not candidates:
            continue
        label, row = rng.choice(candidates)
        out.append(
            {
                "question": (
                    f"In the {record['drug']} adverse reactions table, what does the label "
                    f"report for {label}?"
                ),
                "drug": record["drug"],
                "gold_section": parent_section_name(record["section"]),
                "gold_answer_keypoints": [row],
                "question_type": "table",
            }
        )
    return out


def _draft_abbreviation(sections, drugs, rng, used) -> list[dict]:
    out: list[dict] = []
    for abbreviation, expansion in ABBREVIATIONS:
        if len(out) >= TARGET_COUNTS["abbreviation"]:
            break
        # Score every (drug, section) by how often the abbreviation appears and
        # how central that section is, then take the best.
        scored: list[tuple[int, int, str, str, dict]] = []
        for drug in drugs:
            for section, record in sections.get(drug, {}).items():
                occurrences = len(
                    re.findall(rf"\b{re.escape(abbreviation)}\b", record["text"])
                )
                if not occurrences:
                    continue
                priority = (
                    ABBREVIATION_SECTION_PRIORITY.index(section)
                    if section in ABBREVIATION_SECTION_PRIORITY
                    else len(ABBREVIATION_SECTION_PRIORITY)
                )
                scored.append((occurrences, -priority, drug, section, record))
        if not scored:
            logger.warning("no drug mentions %s", abbreviation)
            continue
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        occurrences, _, drug, section, record = scored[0]
        question_text = f"What does the {drug} label say about {abbreviation} ({expansion})?"
        out.append(
            {
                "question": question_text,
                "drug": drug,
                "gold_section": section,
                "gold_answer_keypoints": select_keypoints(
                    record["text"],
                    question_text,
                    section=section,
                    question_type="abbreviation",
                    cues=(abbreviation.lower(), abbreviation),
                    drug=drug,
                    n=2,
                ),
                "question_type": "abbreviation",
            }
        )
        used.add((drug, section))
    return out


def _draft_unanswerable(drugs, rng) -> list[dict]:
    out: list[dict] = []
    picked = rng.sample(drugs, TARGET_COUNTS["unanswerable"])
    for drug, template in zip(picked, UNANSWERABLE_QUESTIONS):
        out.append(
            {
                "question": template.format(drug=drug),
                "drug": drug,
                "gold_section": "",
                "gold_answer_keypoints": [],
                "question_type": "unanswerable",
            }
        )
    return out


def _draft_multi_section(sections, drugs, rng, used) -> list[dict]:
    out: list[dict] = []
    pool = [
        drug
        for drug in drugs
        if "pregnan" in sections.get(drug, {}).get("contraindications", {}).get("text", "").lower()
    ]
    rng.shuffle(pool)
    if pool:
        drug = pool.pop()
        text = " ".join(
            [
                sections[drug]["contraindications"]["text"],
                sections[drug]["pregnancy"]["text"],
            ]
        )
        question_text = (
            f"What does the {drug} label say about pregnancy in the contraindications "
            "and use-in-specific-populations sections?"
        )
        out.append(
            {
                "question": question_text,
                "drug": drug,
                "gold_section": "contraindications",
                "gold_answer_keypoints": select_keypoints(
                    text,
                    question_text,
                    section="contraindications",
                    question_type="multi-section",
                    cues=("pregnan",),
                    drug=drug,
                    n=3,
                ),
                "question_type": "multi-section",
            }
        )
    else:
        pool = [
            drug
            for drug in drugs
            if "pregnan" in sections.get(drug, {}).get("pregnancy", {}).get("text", "").lower()
        ]
        rng.shuffle(pool)
        if pool:
            drug = pool.pop()
            record = sections[drug]["pregnancy"]
            question_text = f"What does the {drug} label say about use during pregnancy?"
            out.append(
                {
                    "question": question_text,
                    "drug": drug,
                    "gold_section": "pregnancy",
                    "gold_answer_keypoints": select_keypoints(
                        record["text"],
                        question_text,
                        section="pregnancy",
                        question_type="multi-section",
                        cues=("pregnan",),
                        drug=drug,
                        n=3,
                    ),
                    "question_type": "multi-section",
                }
            )
    pool = [
        drug
        for drug in drugs
        if {"dosage_and_administration", "dosage_forms_and_strengths"} <= set(sections.get(drug, {}))
    ]
    rng.shuffle(pool)
    if pool:
        drug = pool.pop()
        text = " ".join(
            [
                sections[drug]["dosage_and_administration"]["text"],
                sections[drug]["dosage_forms_and_strengths"]["text"],
            ]
        )
        question_text = (
            f"How is {drug} dosed, and what dosage forms and strengths are available "
            "according to the label?"
        )
        out.append(
            {
                "question": question_text,
                "drug": drug,
                "gold_section": "dosage_and_administration",
                "gold_answer_keypoints": select_keypoints(
                    text,
                    question_text,
                    section="dosage_and_administration",
                    question_type="multi-section",
                    drug=drug,
                    n=3,
                ),
                "question_type": "multi-section",
            }
        )
    return out


def _draft_cross_drug(sections, drugs, rng, used) -> list[dict]:
    out: list[dict] = []
    boxed = [drug for drug in drugs if "boxed_warning" in sections.get(drug, {})]
    rng.shuffle(boxed)
    if len(boxed) >= 2:
        first, second = boxed[0], boxed[1]
        question_text = (
            f"Both {first} and {second} carry boxed warnings. What does the {first} "
            "label's boxed warning say?"
        )
        out.append(
            {
                "question": question_text,
                "drug": first,
                "gold_section": "boxed_warning",
                "gold_answer_keypoints": select_keypoints(
                    sections[first]["boxed_warning"]["text"],
                    question_text,
                    section="boxed_warning",
                    question_type="cross-drug",
                    drug=first,
                    n=2,
                ),
                "question_type": "cross-drug",
            }
        )
    interactions = [drug for drug in drugs if "drug_interactions" in sections.get(drug, {})]
    rng.shuffle(interactions)
    if len(interactions) >= 2:
        first, second = interactions[0], interactions[1]
        question_text = (
            f"What drug interactions does the {first} label list, and how do they "
            f"compare with those listed for {second}?"
        )
        out.append(
            {
                "question": question_text,
                "drug": first,
                "gold_section": "drug_interactions",
                "gold_answer_keypoints": select_keypoints(
                    sections[first]["drug_interactions"]["text"],
                    question_text,
                    section="drug_interactions",
                    question_type="cross-drug",
                    drug=first,
                    n=2,
                ),
                "question_type": "cross-drug",
            }
        )
    return out


def draft_questions(
    sections_path: Path = SECTIONS_JSONL, *, seed: int = 42
) -> list[dict]:
    sections, tables = load_corpus(sections_path)
    drugs = sorted(sections)
    rng = random.Random(seed)
    used: set[tuple[str, str]] = set()

    drafted: list[dict] = []
    drafted += _draft_factual(sections, drugs, rng, used)
    drafted += _draft_negation(sections, drugs, rng, used)
    drafted += _draft_table(tables, rng)
    drafted += _draft_abbreviation(sections, drugs, rng, used)
    drafted += _draft_unanswerable(drugs, rng)
    drafted += _draft_multi_section(sections, drugs, rng, used)
    drafted += _draft_cross_drug(sections, drugs, rng, used)

    for index, question in enumerate(drafted, start=1):
        question["id"] = f"q{index:03d}"
        question["verified"] = False

    ordered = ["id", "question", "drug", "gold_section", "gold_answer_keypoints", "question_type", "verified"]
    return [{key: question[key] for key in ordered} for question in drafted]


# --------------------------------------------------------------------------- #
# Refresh keypoints of an existing (partially reviewed) question file
# --------------------------------------------------------------------------- #
# Which extra sections a multi-section question draws on, keyed by the words in
# the question itself.
SECTION_HINTS: dict[str, tuple[str, ...]] = {
    "pregnan": ("pregnancy", "use_in_specific_populations"),
    "dose": ("dosage_and_administration", "dosage_forms_and_strengths"),
    "dosage": ("dosage_and_administration", "dosage_forms_and_strengths"),
    "dosed": ("dosage_and_administration", "dosage_forms_and_strengths"),
    "forms": ("dosage_forms_and_strengths",),
    "strengths": ("dosage_forms_and_strengths",),
    "boxed": ("boxed_warning",),
    "interaction": ("drug_interactions",),
    "contraindicat": ("contraindications",),
    "adverse": ("adverse_reactions", "adverse_reactions_table"),
    "indication": ("indications_and_usage",),
}


def question_source_text(
    record: dict,
    sections: dict[str, dict[str, dict]],
    tables_index: dict[tuple[str, str], dict],
) -> tuple[str, tuple[str, ...]]:
    """Return ``(label_text, cues)`` that a question's keypoints should come from."""

    drug = record.get("drug", "")
    gold = record.get("gold_section") or ""
    question = record.get("question", "")
    question_type = record.get("question_type", "factual")
    if question_type == "unanswerable":
        return "", ()

    if question_type == "table":
        table = tables_index.get((drug, f"{gold}_table"))
        if table is not None:
            # Rows only: the caption/context lines are not answers.
            return "\n".join(parse_table(table["text"]).rows), ()

    drug_sections = sections.get(drug, {})
    texts: list[str] = []
    if gold and gold in drug_sections:
        texts.append(drug_sections[gold]["text"])

    lowered = question.lower()
    cues: list[str] = []
    remaining = lowered
    # Longest hints first, and consume what matched, so "contraindications" does
    # not also trigger the "indication" hint.
    for hint in sorted(SECTION_HINTS, key=len, reverse=True):
        if hint not in remaining:
            continue
        cues.append(hint)
        remaining = remaining.replace(hint, " ")
        section_names = SECTION_HINTS[hint]
        for name in section_names:
            if name == gold:
                continue
            if name in drug_sections:
                texts.append(drug_sections[name]["text"])
    return "\n\n".join(dict.fromkeys(texts)), tuple(dict.fromkeys(cues))


def regenerate_keypoints(
    record: dict,
    sections: dict[str, dict[str, dict]],
    tables_index: dict[tuple[str, str], dict],
    *,
    n: int = 3,
    limit: int = 170,
) -> list[str] | None:
    """New keypoints for one question, or ``None`` when there is no source text."""

    question_type = record.get("question_type", "factual")
    if question_type == "unanswerable":
        return []
    text, cues = question_source_text(record, sections, tables_index)
    if not text.strip():
        return None
    return select_keypoints(
        text,
        record.get("question", ""),
        section=record.get("gold_section") or "",
        question_type=question_type,
        cues=cues,
        drug=record.get("drug", ""),
        n=n,
        limit=limit,
    )


def refresh_keypoints(
    path: Path | str = QUESTIONS_JSONL,
    *,
    sections_path: Path | str = SECTIONS_JSONL,
    dry_run: bool = False,
    n: int = 3,
    limit: int = 170,
) -> dict:
    """Recompute ``gold_answer_keypoints`` for ``verified: false`` records only.

    ``id``, ``question``, ``drug``, ``gold_section``, ``question_type`` and
    ``verified`` are never modified; verified questions are skipped entirely.
    """

    path = Path(path)
    records = read_jsonl(path)
    sections, tables = load_corpus(Path(sections_path))
    tables_index = {(record["drug"], record["section"]): record for record in tables}

    changes: list[dict] = []
    skipped_verified = 0
    for record in records:
        if record.get("verified"):
            skipped_verified += 1
            continue
        new = regenerate_keypoints(record, sections, tables_index, n=n, limit=limit)
        if new is None:
            continue
        old = record.get("gold_answer_keypoints") or []
        if new != old:
            changes.append(
                {
                    "id": record.get("id"),
                    "question": record.get("question", ""),
                    "drug": record.get("drug", ""),
                    "gold_section": record.get("gold_section", ""),
                    "question_type": record.get("question_type", ""),
                    "old": old,
                    "new": new,
                }
            )
            if not dry_run:
                record["gold_answer_keypoints"] = new

    if not dry_run and changes:
        temp = path.with_name(path.name + ".tmp")
        write_jsonl(temp, records)
        os.replace(temp, path)

    return {
        "path": str(path),
        "dry_run": dry_run,
        "total": len(records),
        "verified_skipped": skipped_verified,
        "unverified_checked": len(records) - skipped_verified,
        "changed": len(changes),
        "changes": changes,
    }


def backup_question_file(path: Path | str, backup_dir: Path | None = None) -> Path:
    """Copy the question file to a timestamped backup and return its path."""

    path = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target_dir = Path(backup_dir) if backup_dir else path.parent
    backup = target_dir / f"{path.stem}.backup-{stamp}{path.suffix}"
    backup.write_bytes(path.read_bytes())
    return backup


def print_changes(result: dict, *, examples: int = 5) -> None:
    changes = result["changes"]
    print()
    print(f"questions in file     : {result['total']}")
    print(f"verified (untouched)  : {result['verified_skipped']}")
    print(f"unverified checked    : {result['unverified_checked']}")
    print(f"keypoint sets changed : {result['changed']}")
    if not changes:
        return
    print()
    print(f"before/after (first {min(examples, len(changes))} of {len(changes)})")
    for change in changes[:examples]:
        print("=" * 92)
        print(f"{change['id']} | {change['question_type']} | {change['drug']} | {change['gold_section']}")
        print(f"Q: {change['question']}")
        print("  BEFORE:")
        for item in change["old"] or ["(none)"]:
            print(f"    - {item[:180]}")
        print("  AFTER:")
        for item in change["new"]:
            print(f"    + {item[:180]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default=str(SECTIONS_JSONL))
    parser.add_argument("--out", default=str(QUESTIONS_JSONL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument(
        "--update-in-place",
        action="store_true",
        help=(
            "do not draft: refresh gold_answer_keypoints of the unverified records in "
            "--out (verified records are never touched)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="with --update-in-place: show changes only")
    parser.add_argument("--no-backup", action="store_true", help="with --update-in-place: skip the backup copy")
    parser.add_argument("--examples", type=int, default=5, help="before/after examples to print")
    parser.add_argument("--limit", type=int, default=170, help="character cap per keypoint")
    parser.add_argument("--n-keypoints", type=int, default=3, help="maximum keypoints per question")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.update_in_place:
        questions_path = Path(args.out)
        if not questions_path.exists():
            logger.error("no question file at %s", questions_path)
            return 1
        backup = None
        if not args.dry_run and not args.no_backup:
            backup = backup_question_file(questions_path)
            print(f"backup written to {backup}")
        result = refresh_keypoints(
            questions_path,
            sections_path=args.sections,
            dry_run=args.dry_run,
            n=args.n_keypoints,
            limit=args.limit,
        )
        print_changes(result, examples=args.examples)
        if args.dry_run:
            print()
            print("dry run: nothing was written")
        elif result["changed"] and backup is not None:
            print()
            print(f"wrote {result['changed']} updated keypoint sets to {questions_path}")
            print(f"original file kept at {backup}")
        return 0

    questions = draft_questions(Path(args.sections), seed=args.seed)
    write_jsonl(args.out, questions)

    by_type: dict[str, int] = {}
    for question in questions:
        by_type[question["question_type"]] = by_type.get(question["question_type"], 0) + 1
    verified = sum(1 for question in questions if question["verified"])

    print(f"wrote {len(questions)} questions to {args.out}")
    print(f"verified: {verified} | unverified: {len(questions) - verified}")
    print()
    print(f"{'question_type':<16} {'count':>5}   target")
    print("-" * 40)
    for question_type, target in TARGET_COUNTS.items():
        print(f"{question_type:<16} {by_type.get(question_type, 0):>5}   {target}")
    print("-" * 40)
    print(f"{'TOTAL':<16} {len(questions):>5}   {sum(TARGET_COUNTS.values())}")

    drugs_covered = len({question["drug"] for question in questions})
    sections_covered = len({question["gold_section"] for question in questions if question["gold_section"]})
    print(f"\ndrugs covered: {drugs_covered} | distinct gold sections: {sections_covered}")

    for question in questions[: args.show]:
        print("-" * 70)
        print(json.dumps(question, ensure_ascii=False, indent=2)[:900])

    logger.info("all questions are marked verified=false until a human reviews them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
