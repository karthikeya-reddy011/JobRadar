import json
from pathlib import Path

import httpx
import pytest

from app.fetchers.http import NotFound
from app.fetchers.lever import LeverFetcher, parse_postings

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "lever_sample.json").read_text())


def make_fetcher(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LeverFetcher(client=client, sleep=lambda s: None)


def test_parse_postings_normalizes_fields_and_skips_broken_ones():
    jobs = parse_postings(FIXTURE)
    assert [j.title for j in jobs] == ["Staff Software Engineer, Platform", "Data Analyst"]

    first = jobs[0]
    assert first.source == "lever"
    assert first.location == "Hyderabad, India"
    assert first.is_remote is False
    assert first.seniority == "staff"
    assert first.posted_at is not None and first.posted_at.year == 2026
    # skills live in the headed lists, so they must be part of the description
    assert "Design APIs in Go" in first.description
    assert "- Run Kubernetes clusters on AWS" in first.description
    assert "equal opportunity employer" in first.description

    assert jobs[1].is_remote is True  # workplaceType == "remote"
    assert jobs[1].apply_url.endswith("0002")  # falls back from hostedUrl only when missing


def test_fetch_uses_us_host_first_and_stops_when_it_has_jobs():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(200, json=FIXTURE)

    jobs = make_fetcher(handler).fetch("acme")
    assert len(jobs) == 2
    assert hosts == ["api.lever.co"]


def test_fetch_falls_back_to_eu_when_us_is_empty():
    def handler(request):
        if request.url.host == "api.lever.co":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=FIXTURE)

    assert len(make_fetcher(handler).fetch("acme-eu")) == 2


def test_unknown_board_raises_not_found():
    with pytest.raises(NotFound):
        make_fetcher(lambda request: httpx.Response(404)).fetch("nope")


def test_empty_board_returns_empty_list():
    def handler(request):
        if request.url.host == "api.lever.co":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    assert make_fetcher(handler).fetch("quiet") == []
