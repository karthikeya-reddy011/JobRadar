from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

import httpx

from app.fetchers.base import Fetcher, NormalizedJob
from app.fetchers.http import get_json, make_client
from app.normalize import html_to_text, infer_seniority, looks_remote

# Public Job Board API: no auth. content=true includes the full HTML description per job.
API_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


def _parse_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_jobs(payload: dict[str, Any]) -> list[NormalizedJob]:
    """Pure function (no network): Greenhouse JSON -> normalized jobs. Easy to unit test."""
    jobs: list[NormalizedJob] = []
    for raw in payload.get("jobs", []):
        job_id = raw.get("id")
        title = (raw.get("title") or "").strip()
        url = raw.get("absolute_url")
        if job_id is None or not title or not url:
            continue  # unusable posting: skip rather than store junk
        location = ((raw.get("location") or {}).get("name") or "").strip() or None
        jobs.append(
            NormalizedJob(
                source="greenhouse",
                external_id=str(job_id),
                title=title,
                description=html_to_text(raw.get("content") or ""),
                location=location,
                is_remote=looks_remote(location, title),
                apply_url=url,
                # Greenhouse's list endpoint gives updated_at, which moves when a posting is
                # edited. Treat it as "roughly when posted", and prefer first_published if present.
                posted_at=_parse_dt(raw.get("first_published")) or _parse_dt(raw.get("updated_at")),
                seniority=infer_seniority(title),
            )
        )
    return jobs


class GreenhouseFetcher(Fetcher):
    source = "greenhouse"

    def __init__(
        self,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client or make_client()
        self._sleep = sleep

    def fetch(self, slug: str) -> list[NormalizedJob]:
        payload = get_json(
            self.client,
            API_URL.format(slug=slug),
            params={"content": "true"},
            sleep=self._sleep,
        )
        return parse_jobs(payload)
