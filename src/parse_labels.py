"""Extract and clean label sections from the cached openFDA responses.

Usage
-----
    python -m src.parse_labels
    python -m src.parse_labels --min-chars 40

Reads ``data/raw/_fetch_manifest.json`` (written by ``src.fetch_openfda``) and
writes one JSON object per section to ``data/processed/sections.jsonl``:

    {"drug": ..., "set_id": ..., "section": ..., "text": ...,
     "source_field": ..., "effective_time": ...}
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from src.utils import (
    FETCH_MANIFEST,
    SECTIONS_JSONL,
    TARGET_SECTIONS,
    clean_section_text,
    ensure_dirs,
    get_logger,
    is_table_field,
    label_effective_time,
    label_identity,
    label_section_map,
    read_json,
    select_best_label,
    truncate,
    write_jsonl,
)

logger = get_logger("parse_labels")

# Sections shorter than this are usually artefacts such as "None" or "-".
MIN_SECTION_CHARS = 20


def extract_sections(label: dict, drug: str, min_chars: int = MIN_SECTION_CHARS) -> list[dict]:
    """Turn one raw label into a list of section records."""

    set_id = label_identity(label)
    effective_time = label_effective_time(label)
    records: list[dict] = []

    for section, (source_field, raw_text) in label_section_map(label).items():
        text = clean_section_text(raw_text, section)
        if len(text) < min_chars:
            continue
        records.append(
            {
                "drug": drug,
                "set_id": set_id,
                "section": section,
                "text": text,
                "source_field": source_field,
                "effective_time": effective_time,
            }
        )
    return records


def load_selected_labels(manifest: dict) -> list[tuple[str, dict]]:
    """Yield ``(drug, best_label)`` for every successful fetch in the manifest."""

    selected: list[tuple[str, dict]] = []
    for slug, record in sorted(manifest.items(), key=lambda item: item[1].get("drug", "")):
        if record.get("status") != "ok":
            logger.warning("%s: skipped (%s)", record.get("drug", slug), record.get("status"))
            continue
        cache_path = record.get("cache_file")
        if not cache_path:
            logger.warning("%s: no cache_file in manifest, skipped", record.get("drug", slug))
            continue
        from src.utils import PROJECT_ROOT  # local import keeps the header light

        path = PROJECT_ROOT / cache_path
        if not path.exists():
            logger.warning("%s: cache file %s missing (re-run fetch), skipped", record.get("drug"), path)
            continue
        payload = read_json(path)
        results = payload.get("results", [])
        if not results:
            logger.warning("%s: cached payload has no results, skipped", record.get("drug"))
            continue
        selected.append((record["drug"], results[select_best_label(results)]))
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-chars", type=int, default=MIN_SECTION_CHARS)
    parser.add_argument("--show", type=int, default=3, help="how many sample records to print")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs()

    if not FETCH_MANIFEST.exists():
        logger.error("no manifest at %s - run `python -m src.fetch_openfda` first", FETCH_MANIFEST)
        return 1

    manifest = read_json(FETCH_MANIFEST)
    labels = load_selected_labels(manifest)
    if not labels:
        logger.error("no usable labels found in the manifest")
        return 1

    all_records: list[dict] = []
    per_drug_sections: Counter[str] = Counter()
    per_drug_tables: Counter[str] = Counter()
    per_drug_chars: Counter[str] = Counter()
    section_drug_coverage: defaultdict[str, int] = defaultdict(int)

    for drug, label in labels:
        records = extract_sections(label, drug, min_chars=args.min_chars)
        if not records:
            logger.warning("%s: no sections above %d chars, skipped", drug, args.min_chars)
            continue
        all_records.extend(records)
        per_drug_sections[drug] = len(records)
        per_drug_tables[drug] = sum(1 for record in records if is_table_field(record["source_field"]))
        per_drug_chars[drug] = sum(len(record["text"]) for record in records)
        for record in records:
            section_drug_coverage[record["section"]] += 1

    count = write_jsonl(SECTIONS_JSONL, all_records)
    logger.info("wrote %d section records to %s", count, SECTIONS_JSONL)

    print_summary(per_drug_sections, per_drug_tables, per_drug_chars, section_drug_coverage, all_records)
    print_samples(all_records, args.show)
    return 0


def print_summary(
    per_drug_sections: Counter[str],
    per_drug_tables: Counter[str],
    per_drug_chars: Counter[str],
    section_drug_coverage: defaultdict[str, int],
    all_records: list[dict],
) -> None:
    print()
    print(f"{'drug':<24} {'text_sections':>13} {'table_sections':>14} {'chars':>9}")
    print("-" * 64)
    for drug in sorted(per_drug_sections):
        text_sections = per_drug_sections[drug] - per_drug_tables[drug]
        print(
            f"{drug:<24} {text_sections:>13} {per_drug_tables[drug]:>14} "
            f"{per_drug_chars[drug]:>9}"
        )
    print("-" * 64)
    total_text = sum(per_drug_sections.values()) - sum(per_drug_tables.values())
    print(
        f"{'TOTAL':<24} {total_text:>13} {sum(per_drug_tables.values()):>14} "
        f"{sum(per_drug_chars.values()):>9}   ({len(per_drug_sections)} drugs)"
    )

    print()
    print("section coverage (drugs with this section):")
    canonical = [name for name, _ in TARGET_SECTIONS]
    extra = sorted(set(section_drug_coverage) - set(canonical))
    for section in canonical + extra:
        if section_drug_coverage.get(section):
            print(f"  {section:<32} {section_drug_coverage[section]:>3}")


def print_samples(records: list[dict], show: int) -> None:
    if show <= 0 or not records:
        return
    print()
    print(f"sample records (first {show} of {len(records)}):")
    for record in records[:show]:
        print("-" * 80)
        print(
            f"drug={record['drug']} | section={record['section']} | "
            f"source_field={record['source_field']} | set_id={record['set_id']} | "
            f"effective_time={record['effective_time']} | chars={len(record['text'])}"
        )
        print(truncate(record["text"], 400))


if __name__ == "__main__":
    raise SystemExit(main())
