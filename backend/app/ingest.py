"""Ingestion: fetch every company in companies.yaml and upsert into Postgres.

Run from backend/:  python -m app.ingest
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config import DEFAULT_COMPANIES_FILE
from app.db import get_engine
from app.fetchers.ashby import AshbyFetcher
from app.fetchers.base import Fetcher
from app.fetchers.greenhouse import GreenhouseFetcher
from app.fetchers.http import FetchError
from app.fetchers.lever import LeverFetcher
from app.normalize import content_hash
from app.skills import extract_skills_from_text

log = logging.getLogger("jobradar.ingest")

FETCHER_CLASSES: dict[str, type[Fetcher]] = {
    "greenhouse": GreenhouseFetcher,
    "lever": LeverFetcher,
    "ashby": AshbyFetcher,
}

UPSERT_COMPANY = text("""
    INSERT INTO companies (name, ats, slug) VALUES (:name, :ats, :slug)
    ON CONFLICT (ats, slug) DO UPDATE SET name = EXCLUDED.name
    RETURNING id
""")

# `xmax = 0` is true only for freshly inserted rows, which lets us count new vs updated.
# The embedding is cleared when title/description change so it can never go stale.
UPSERT_JOB = text("""
    INSERT INTO jobs (source, external_id, company_id, title, description, location, is_remote,
                      seniority, skills, apply_url, posted_at, last_seen_at, is_active, content_hash)
    VALUES (:source, :external_id, :company_id, :title, :description, :location, :is_remote,
            :seniority, CAST(:skills AS TEXT[]), :apply_url, :posted_at, :seen_at, TRUE, :content_hash)
    ON CONFLICT (source, external_id) DO UPDATE SET
        company_id = EXCLUDED.company_id,
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        location = EXCLUDED.location,
        is_remote = EXCLUDED.is_remote,
        seniority = EXCLUDED.seniority,
        skills = EXCLUDED.skills,
        apply_url = EXCLUDED.apply_url,
        posted_at = EXCLUDED.posted_at,
        last_seen_at = EXCLUDED.last_seen_at,
        is_active = TRUE,
        content_hash = EXCLUDED.content_hash,
        embedding = CASE
            WHEN jobs.title IS DISTINCT FROM EXCLUDED.title
              OR jobs.description IS DISTINCT FROM EXCLUDED.description
            THEN NULL ELSE jobs.embedding END,
        embedding_model = CASE
            WHEN jobs.title IS DISTINCT FROM EXCLUDED.title
              OR jobs.description IS DISTINCT FROM EXCLUDED.description
            THEN NULL ELSE jobs.embedding_model END
    RETURNING (xmax = 0) AS inserted
""")

# Anything for this company not touched in this run has disappeared from the feed -> closed.
DEACTIVATE_MISSING = text("""
    UPDATE jobs SET is_active = FALSE
    WHERE company_id = :company_id AND source = :source
      AND is_active AND last_seen_at < :seen_at
""")


@dataclass
class IngestStats:
    company: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    deactivated: int = 0


def ingest_company(engine: Engine, fetcher: Fetcher, *, name: str, slug: str) -> IngestStats:
    seen_at = datetime.now(timezone.utc)
    # Fetch BEFORE opening a transaction. If this raises, nothing is written, so an outage can
    # never mark a company's jobs as closed.
    jobs = fetcher.fetch(slug)
    stats = IngestStats(company=name, fetched=len(jobs))

    with engine.begin() as conn:
        company_id = conn.execute(
            UPSERT_COMPANY, {"name": name, "ats": fetcher.source, "slug": slug}
        ).scalar_one()

        for job in jobs:
            inserted = conn.execute(
                UPSERT_JOB,
                {
                    "source": job.source,
                    "external_id": job.external_id,
                    "company_id": company_id,
                    "title": job.title,
                    "description": job.description,
                    "location": job.location,
                    "is_remote": job.is_remote,
                    "seniority": job.seniority,
                    "skills": extract_skills_from_text(f"{job.title}\n{job.description}"),
                    "apply_url": job.apply_url,
                    "posted_at": job.posted_at,
                    "seen_at": seen_at,
                    "content_hash": content_hash(name, job.title, job.location),
                },
            ).scalar_one()
            if inserted:
                stats.inserted += 1
            else:
                stats.updated += 1

        if jobs:
            result = conn.execute(
                DEACTIVATE_MISSING,
                {"company_id": company_id, "source": fetcher.source, "seen_at": seen_at},
            )
            stats.deactivated = result.rowcount
        else:
            # An empty board is far more often a glitch than "every job closed at once".
            log.warning("%s returned 0 jobs; leaving existing jobs untouched", name)
    return stats


def load_companies(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("companies", [])


def run_ingest(
    engine: Engine,
    companies: list[dict],
    *,
    only: str | None = None,
    delay: float = 1.0,
) -> tuple[list[IngestStats], list[str]]:
    fetchers: dict[str, Fetcher] = {}
    results: list[IngestStats] = []
    failures: list[str] = []

    for entry in companies:
        ats, slug = entry["ats"], entry["slug"]
        name = entry.get("name", slug)
        if only and slug != only:
            continue
        fetcher_cls = FETCHER_CLASSES.get(ats)
        if fetcher_cls is None:
            log.warning("No fetcher for ats=%r (company %s); skipping", ats, name)
            continue
        fetcher = fetchers.setdefault(ats, fetcher_cls())
        try:
            stats = ingest_company(engine, fetcher, name=name, slug=slug)
        except FetchError as exc:
            log.error("Fetch failed for %s (%s/%s): %s", name, ats, slug, exc)
            failures.append(name)
        else:
            results.append(stats)
            log.info(
                "%s: fetched=%d new=%d updated=%d closed=%d",
                stats.company, stats.fetched, stats.inserted, stats.updated, stats.deactivated,
            )
        time.sleep(delay)  # be polite between boards
    return results, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest jobs from companies.yaml")
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES_FILE)
    parser.add_argument("--only", help="ingest just this slug (handy for debugging)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results, failures = run_ingest(get_engine(), load_companies(args.companies), only=args.only)

    print(
        f"\nDone: {len(results)} companies ok, {len(failures)} failed, "
        f"{sum(r.fetched for r in results)} jobs fetched, "
        f"{sum(r.inserted for r in results)} new, {sum(r.deactivated for r in results)} closed."
    )
    if failures:
        print("Failed:", ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
