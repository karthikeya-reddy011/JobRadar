from __future__ import annotations

import random
import time
from typing import Any, Callable

import httpx

from app.config import USER_AGENT

# 529 is what the Claude API returns when it is temporarily overloaded.
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class FetchError(Exception):
    pass


class NotFound(FetchError):
    """HTTP 404: board slug (or model name) does not exist. Not worth retrying."""


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )


def _body_snippet(resp: Any) -> str:
    try:
        return (resp.text or "")[:300]
    except Exception:  # noqa: BLE001
        return ""


def _request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
    headers: dict[str, str] | None = None,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Request with retries and exponential backoff on 429/5xx/network errors. Honors Retry-After."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        delay = base_delay * (2**attempt) + random.uniform(0, 0.25)
        try:
            resp = client.request(method, url, params=params, json=json, headers=headers)
        except httpx.TransportError as exc:
            last_error = exc
        else:
            if resp.status_code == 404:
                raise NotFound(f"404 from {url}: {_body_snippet(resp)}")
            if resp.status_code in RETRY_STATUSES:
                last_error = FetchError(
                    f"HTTP {resp.status_code} from {url}: {_body_snippet(resp)}"
                )
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = min(float(retry_after), 60.0)
            elif resp.is_error:
                raise FetchError(f"HTTP {resp.status_code} from {url}: {_body_snippet(resp)}")
            else:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise FetchError(f"Invalid JSON from {url}") from exc
        if attempt < max_attempts - 1:
            sleep(delay)
    raise FetchError(
        f"Giving up on {url} after {max_attempts} attempts"
        + (f"; last response: {last_error}" if last_error else "")
    ) from last_error


def get_json(client: httpx.Client, url: str, *, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    return _request_json(client, "GET", url, params=params, **kwargs)


def post_json(
    client: httpx.Client,
    url: str,
    *,
    json: Any,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    return _request_json(client, "POST", url, json=json, headers=headers, **kwargs)
