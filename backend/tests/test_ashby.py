import json
from pathlib import Path

import httpx

from app.fetchers.ashby import AshbyFetcher, parse_jobs

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ashby_sample.json").read_text())


def test_parse_jobs_skips_unlisted_and_normalizes():
    jobs = parse_jobs(FIXTURE)
    assert [j.title for j in jobs] == ["Senior Data Engineer", "Product Designer"]

    first = jobs[0]
    assert first.source == "ashby"
    assert first.seniority == "senior"
    assert first.is_remote is False
    assert "Own our data platform." in first.description
    assert "- Airflow" in first.description
    assert first.posted_at is not None and first.posted_at.year == 2026

    second = jobs[1]
    assert second.is_remote is True
    assert second.description == "Design delightful things."  # falls back to descriptionPlain


def test_fetch_hits_the_board_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=FIXTURE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    jobs = AshbyFetcher(client=client, sleep=lambda s: None).fetch("acme")
    assert len(jobs) == 2
    assert seen["url"].startswith("https://api.ashbyhq.com/posting-api/job-board/acme")
