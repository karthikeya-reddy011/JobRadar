from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

import httpx

from app.fetchers.base import Fetcher, NormalizedJob
from app.fetchers.http import FetchError, get_json, make_client
from app.normalize import html_to_text, infer_seniority, looks_remote

# Public Job Posting API, no auth. One response contains the whole board.
API_URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"


def _parse_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_jobs(payload: Any) -> list[NormalizedJob]:
    """Pure function (no network): Ashby JSON -> normalized jobs."""
    if not isinstance(payload, dict):
        raise FetchError("Unexpected Ashby response (expected a JSON object)")
    jobs: list[NormalizedJob] = []
    for raw in payload.get("jobs") or []:
        if not isinstance(raw, dict) or raw.get("isListed") is False:
            continue  # unlisted postings are not shown on the public board
        job_id = raw.get("id")
        title = (raw.get("title") or "").strip()
        url = raw.get("jobUrl") or raw.get("applyUrl")
        if not job_id or not title or not url:
            continue
        location = (raw.get("location") or "").strip() or None
        workplace = str(raw.get("workplaceType") or "").lower()
        jobs.append(
            NormalizedJob(
                source="ashby",
                external_id=str(job_id),
                title=title,
                description=html_to_text(raw.get("descriptionHtml") or "") or (raw.get("descriptionPlain") or "").strip(),
                location=location,
                is_remote=bool(raw.get("isRemote")) or workplace == "remote" or looks_remote(location, title),
                apply_url=url,
                posted_at=_parse_dt(raw.get("publishedAt")),
                seniority=infer_seniority(title),
            )
        )
    return jobs


class AshbyFetcher(Fetcher):
    source = "ashby"

    def __init__(self, client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client or make_client()
        self._sleep = sleep

    def fetch(self, slug: str) -> list[NormalizedJob]:
        payload = get_json(self.client, API_URL.format(slug=slug), sleep=self._sleep)
        return parse_jobs(payload)
