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
import random
import re
from pathlib import Path

from src.utils import (
    QUESTIONS_JSONL,
    SECTIONS_JSONL,
    get_logger,
    is_table_field,
    parent_section_name,
    parse_table,
    read_jsonl,
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
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def keypoints(text: str, n: int = 3, limit: int = 170) -> list[str]:
    """Verbatim snippets from the label, used as answer keypoints."""

    sentences = split_sentences(text)
    if not sentences:
        return [truncate(text, limit)]
    return [sentence if len(sentence) <= limit else truncate(sentence, limit) for sentence in sentences[:n]]


def keypoints_with_cue(text: str, cues: tuple[str, ...], n: int = 2) -> list[str]:
    sentences = split_sentences(text)
    matching = [
        sentence for sentence in sentences if any(cue in sentence.lower() for cue in cues)
    ]
    return keypoints(" ".join(matching), n=n) if matching else keypoints(text, n=n)


def has_abbreviation(text: str, abbreviation: str) -> bool:
    return re.search(rf"\b{re.escape(abbreviation)}\b", text) is not None


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
            out.append(
                {
                    "question": template.format(drug=drug),
                    "drug": drug,
                    "gold_section": section,
                    "gold_answer_keypoints": keypoints(record["text"], n=3),
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
            out.append(
                {
                    "question": template.format(drug=drug),
                    "drug": drug,
                    "gold_section": section,
                    "gold_answer_keypoints": keypoints_with_cue(
                        record["text"], NEGATION_CUES, n=2
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
        out.append(
            {
                "question": (
                    f"What does the {drug} label say about {abbreviation} ({expansion})?"
                ),
                "drug": drug,
                "gold_section": section,
                "gold_answer_keypoints": keypoints_with_cue(
                    record["text"], (abbreviation.lower(), abbreviation), n=2
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
        pregnancy_points = keypoints_with_cue(text, ("pregnan",), n=3)
        out.append(
            {
                "question": (
                    f"What does the {drug} label say about pregnancy in the contraindications "
                    "and use-in-specific-populations sections?"
                ),
                "drug": drug,
                "gold_section": "contraindications",
                "gold_answer_keypoints": pregnancy_points,
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
            out.append(
                {
                    "question": f"What does the {drug} label say about use during pregnancy?",
                    "drug": drug,
                    "gold_section": "pregnancy",
                    "gold_answer_keypoints": keypoints_with_cue(record["text"], ("pregnan",), n=3),
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
        out.append(
            {
                "question": (
                    f"How is {drug} dosed, and what dosage forms and strengths are available "
                    "according to the label?"
                ),
                "drug": drug,
                "gold_section": "dosage_and_administration",
                "gold_answer_keypoints": keypoints(text, n=3),
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
        out.append(
            {
                "question": (
                    f"Both {first} and {second} carry boxed warnings. What does the {first} "
                    "label's boxed warning say?"
                ),
                "drug": first,
                "gold_section": "boxed_warning",
                "gold_answer_keypoints": keypoints(sections[first]["boxed_warning"]["text"], n=2),
                "question_type": "cross-drug",
            }
        )
    interactions = [drug for drug in drugs if "drug_interactions" in sections.get(drug, {})]
    rng.shuffle(interactions)
    if len(interactions) >= 2:
        first, second = interactions[0], interactions[1]
        out.append(
            {
                "question": (
                    f"What drug interactions does the {first} label list, and how do they "
                    f"compare with those listed for {second}?"
                ),
                "drug": first,
                "gold_section": "drug_interactions",
                "gold_answer_keypoints": keypoints(
                    sections[first]["drug_interactions"]["text"], n=2
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default=str(SECTIONS_JSONL))
    parser.add_argument("--out", default=str(QUESTIONS_JSONL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
