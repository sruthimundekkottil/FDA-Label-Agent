"""Download drug labels from the openFDA Drug Label API and cache them.

Usage
-----
    python -m src.fetch_openfda                # fetch everything in configs/drugs.txt
    python -m src.fetch_openfda --drugs metformin warfarin
    python -m src.fetch_openfda --refresh      # ignore the local cache

Design notes
------------
* Query shape: ``search=openfda.generic_name:<name>`` with **no quotes**. For
  multi-word names each term becomes its own clause joined with ``+AND+``, e.g.
  ``openfda.generic_name:insulin+AND+openfda.generic_name:glargine``.
* ``limit=5``: we fetch up to five labels and keep the one with the most
  populated sections (see ``utils.select_best_label``).
* Every raw response is cached verbatim under ``data/raw/<slug>.json`` so re-runs
  do not hit the API again.
* A drug that returns nothing (HTTP 404 / ``NOT_FOUND``) or keeps failing is
  logged and skipped - one bad name never crashes the run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

from src.utils import (
    FETCH_MANIFEST,
    RAW_DIR,
    count_populated_sections,
    ensure_dirs,
    get_logger,
    label_effective_time,
    label_identity,
    label_total_chars,
    load_drug_list,
    load_env,
    read_json,
    select_best_label,
    slugify,
    write_json,
)

API_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_LIMIT = 5
DEFAULT_SLEEP = 0.6
MAX_RETRIES = 4
TIMEOUT = 40

logger = get_logger("fetch_openfda")


def build_search_query(drug: str) -> str:
    """``insulin glargine`` -> ``openfda.generic_name:insulin+AND+openfda.generic_name:glargine``."""

    terms = [term for term in drug.strip().split() if term]
    clauses = [f"openfda.generic_name:{quote(term, safe='')}" for term in terms]
    return "+AND+".join(clauses)


def build_url(search: str, limit: int, api_key: str | None = None) -> str:
    # The URL is assembled by hand on purpose: letting requests encode the params
    # would percent-encode the '+' separators that openFDA expects.
    url = f"{API_URL}?search={search}&limit={limit}"
    if api_key:
        url += f"&api_key={quote(api_key, safe='')}"
    return url


def fetch_label_payload(
    drug: str,
    *,
    limit: int,
    refresh: bool,
    session: requests.Session,
    api_key: str | None,
) -> tuple[Path, dict | None, str, str | None]:
    """Return ``(cache_path, payload, status, error)`` for one drug.

    ``payload`` is the full raw API response (or ``None`` when there is no
    match / the request failed). ``status`` is one of ``ok``, ``no_match``,
    ``error``.
    """

    cache_path = RAW_DIR / f"{slugify(drug)}.json"
    if cache_path.exists() and not refresh:
        payload = read_json(cache_path)
        logger.info("%s: using cached response (%s)", drug, cache_path.name)
        return cache_path, payload, "ok", None

    search = build_search_query(drug)
    url = build_url(search, limit, api_key)
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:  # network hiccup
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: request error (%s), retry %d/%d", drug, last_error, attempt, MAX_RETRIES)
            time.sleep(min(30, 2 ** attempt) + random.uniform(0, 1))
            continue

        if response.status_code == 200:
            payload = response.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            return cache_path, payload, "ok", None

        if response.status_code == 404:
            # openFDA answers 404 + NOT_FOUND when the query has no matches.
            logger.warning("%s: no openFDA match (404)", drug)
            return cache_path, None, "no_match", "no match for query"

        if response.status_code == 429 or response.status_code >= 500:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            last_error = f"HTTP {response.status_code}"
            logger.warning("%s: %s, backing off %.1fs", drug, last_error, wait)
            time.sleep(wait)
            continue

        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        logger.error("%s: unexpected response %s", drug, last_error)
        break

    logger.error("%s: giving up after %d attempts (%s)", drug, MAX_RETRIES, last_error)
    return cache_path, None, "error", last_error


def fetch_all(
    drugs: list[str],
    *,
    limit: int = DEFAULT_LIMIT,
    sleep_s: float = DEFAULT_SLEEP,
    refresh: bool = False,
    prune: bool = False,
) -> dict[str, dict]:
    """Fetch every drug and write ``data/raw/_fetch_manifest.json``.

    With ``prune=True`` (used when the list comes from ``configs/drugs.txt``)
    entries for drugs that are no longer configured are dropped, so removing a
    drug from the file really removes it from the corpus.
    """

    ensure_dirs()
    load_env()
    api_key = os.environ.get("OPENFDA_API_KEY") or None
    if not api_key:
        logger.info("OPENFDA_API_KEY not set - using the public rate-limited tier")

    manifest: dict[str, dict] = {}
    if FETCH_MANIFEST.exists() and not refresh:
        # Keep earlier results (e.g. drugs removed from the list) but refresh keys.
        try:
            manifest = read_json(FETCH_MANIFEST)
        except json.JSONDecodeError:
            logger.warning("manifest was not valid JSON; starting a fresh one")
            manifest = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "drug-label-rag/0.1 (educational project)"})

    for position, drug in enumerate(drugs, start=1):
        slug = slugify(drug)
        record: dict = manifest.get(slug, {})
        record.update({"drug": drug, "query": build_search_query(drug)})

        cache_path, payload, status, error = fetch_label_payload(
            drug, limit=limit, refresh=refresh, session=session, api_key=api_key
        )
        record["status"] = status
        record["error"] = error
        record["cache_file"] = str(cache_path.relative_to(cache_path.parents[2]))
        record["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if status == "ok" and payload is not None:
            results = payload.get("results", [])
            record["n_candidates"] = len(results)
            if results:
                best = results[select_best_label(results)]
                record["selected_set_id"] = label_identity(best)
                record["effective_time"] = label_effective_time(best)
                record["sections"] = count_populated_sections(best)
                record["total_chars"] = label_total_chars(best)
                logger.info(
                    "[%d/%d] %s: %d candidates, kept %s (%d sections, %d chars)",
                    position,
                    len(drugs),
                    drug,
                    len(results),
                    record["selected_set_id"] or "n/a",
                    record["sections"],
                    record["total_chars"],
                )
            else:
                record["status"] = "no_match"
                record["error"] = "empty results list"
                record["n_candidates"] = 0
                logger.warning("[%d/%d] %s: API returned an empty results list", position, len(drugs), drug)
        else:
            record.setdefault("n_candidates", 0)
            logger.warning("[%d/%d] %s: skipped (%s)", position, len(drugs), drug, status)

        manifest[slug] = record
        write_json(FETCH_MANIFEST, manifest)  # save after every drug: resumable

        if position < len(drugs):
            time.sleep(sleep_s)

    if prune:
        wanted = {slugify(drug) for drug in drugs}
        for slug in [key for key in manifest if key not in wanted]:
            logger.info("pruning %s from the manifest (no longer configured)", slug)
            manifest.pop(slug)
        write_json(FETCH_MANIFEST, manifest)

    return manifest


def summarize(manifest: dict[str, dict]) -> None:
    ok = [record for record in manifest.values() if record.get("status") == "ok" and record.get("selected_set_id")]
    skipped = [record for record in manifest.values() if record not in ok]
    print()
    print(f"drugs requested : {len(manifest)}")
    print(f"labels kept     : {len(ok)}")
    print(f"skipped         : {len(skipped)}")
    if skipped:
        print("skipped drugs   : " + ", ".join(f"{r['drug']} ({r.get('status')})" for r in skipped))
    print()
    header = f"{'drug':<24} {'status':<9} {'cand':>4} {'sect':>5} {'chars':>7}  set_id"
    print(header)
    print("-" * len(header))
    for record in sorted(manifest.values(), key=lambda r: r["drug"]):
        print(
            f"{record['drug']:<24} {record.get('status', '?'):<9} "
            f"{record.get('n_candidates', 0):>4} {record.get('sections', 0):>5} "
            f"{record.get('total_chars', 0):>7}  {record.get('selected_set_id', '')[:40]}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drugs", nargs="*", help="override the drug list")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="labels per drug (default 5)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="seconds between requests")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from_config = not args.drugs
    drugs = load_drug_list() if from_config else args.drugs
    if not drugs:
        logger.error("no drugs to fetch (configs/drugs.txt is empty)")
        return 1
    manifest = fetch_all(
        drugs,
        limit=args.limit,
        sleep_s=args.sleep,
        refresh=args.refresh,
        prune=from_config,
    )
    summarize(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
