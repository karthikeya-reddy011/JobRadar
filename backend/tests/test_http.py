import httpx
import pytest

from app.fetchers.http import FetchError, NotFound, get_json


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_retries_on_429_then_succeeds_and_honors_retry_after():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": True})

    sleeps = []
    result = get_json(make_client(handler), "https://example.com/x", sleep=sleeps.append)
    assert result == {"ok": True}
    assert len(calls) == 3
    assert sleeps == [2.0, 2.0]


def test_404_raises_not_found_without_retrying():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(NotFound):
        get_json(make_client(handler), "https://example.com/x", sleep=lambda s: None)
    assert len(calls) == 1


def test_gives_up_after_max_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": "temporary outage"})

    with pytest.raises(FetchError, match="temporary outage"):
        get_json(make_client(handler), "https://example.com/x", max_attempts=3, sleep=lambda s: None)
    assert len(calls) == 3
