import json
from pathlib import Path

import httpx

from app.fetchers.greenhouse import GreenhouseFetcher, parse_jobs

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "greenhouse_sample.json").read_text())


def test_parse_jobs_normalizes_fields_and_skips_broken_postings():
    jobs = parse_jobs(FIXTURE)
    assert [j.external_id for j in jobs] == ["1001", "1002"]  # 1003 has no URL -> skipped

    first = jobs[0]
    assert first.source == "greenhouse"
    assert first.title == "Senior Backend Engineer"
    assert first.location == "Remote - India"
    assert first.is_remote is True
    assert first.seniority == "senior"
    assert "Python & Postgres" in first.description
    assert "- Kubernetes" in first.description
    assert "<" not in first.description
    assert first.posted_at is not None and first.posted_at.year == 2026

    second = jobs[1]
    assert second.is_remote is False
    assert second.seniority == "intern"
    assert second.posted_at is not None  # "Z" suffix parses too


def test_fetch_requests_full_content_and_sends_user_agent():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json=FIXTURE)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), headers={"User-Agent": "TestAgent/1.0"}
    )
    jobs = GreenhouseFetcher(client=client, sleep=lambda s: None).fetch("acme")

    assert len(jobs) == 2
    assert "boards-api.greenhouse.io/v1/boards/acme/jobs" in seen["url"]
    assert "content=true" in seen["url"]
    assert seen["ua"] == "TestAgent/1.0"
