from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from app.fetchers.base import Fetcher, NormalizedJob
from app.fetchers.http import FetchError, NotFound, get_json, make_client
from app.normalize import html_to_text, infer_seniority, looks_remote

# Public Postings API, no auth. Companies hosted in Lever's EU region answer on the EU host,
# and the US host returns an empty list for them, so we try both.
API_BASES = (
    "https://api.lever.co/v0/postings",
    "https://api.eu.lever.co/v0/postings",
)


def _created_at(value: Any) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)  # Lever uses epoch milliseconds
    except (OverflowError, OSError, ValueError):
        return None


def _description(posting: dict[str, Any]) -> str:
    """Lever splits a posting into an intro, headed lists (requirements etc.) and a closing
    blurb. Skills usually live in the lists, so all three are needed."""
    parts = [posting.get("descriptionPlain") or html_to_text(posting.get("description") or "")]
    for block in posting.get("lists") or []:
        heading = (block.get("text") or "").strip()
        body = html_to_text(block.get("content") or "")
        if heading or body:
            parts.append(f"{heading}\n{body}".strip())
    parts.append(posting.get("additionalPlain") or html_to_text(posting.get("additional") or ""))
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def parse_postings(payload: Any) -> list[NormalizedJob]:
    """Pure function (no network): Lever JSON -> normalized jobs."""
    if not isinstance(payload, list):
        raise FetchError("Unexpected Lever response (expected a JSON list)")
    jobs: list[NormalizedJob] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        job_id = raw.get("id")
        title = (raw.get("text") or "").strip()
        url = raw.get("hostedUrl") or raw.get("applyUrl")
        if not job_id or not title or not url:
            continue
        categories = raw.get("categories") or {}
        all_locations = categories.get("allLocations") or []
        location = (categories.get("location") or (all_locations[0] if all_locations else "") or "").strip() or None
        jobs.append(
            NormalizedJob(
                source="lever",
                external_id=str(job_id),
                title=title,
                description=_description(raw),
                location=location,
                is_remote=raw.get("workplaceType") == "remote" or looks_remote(location, title),
                apply_url=url,
                posted_at=_created_at(raw.get("createdAt")),
                seniority=infer_seniority(title),
            )
        )
    return jobs


class LeverFetcher(Fetcher):
    source = "lever"

    def __init__(self, client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client or make_client()
        self._sleep = sleep

    def fetch(self, slug: str) -> list[NormalizedJob]:
        not_found = 0
        for base in API_BASES:
            try:
                payload = get_json(self.client, f"{base}/{slug}", params={"mode": "json"}, sleep=self._sleep)
            except NotFound:
                not_found += 1
                continue
            jobs = parse_postings(payload)
            if jobs:
                return jobs
        if not_found == len(API_BASES):
            raise NotFound(f"Lever board {slug!r} not found (tried US and EU)")
        return []  # board exists but has no open postings
