"""Integration test for upsert + close-detection. Needs `docker compose up -d` and
`alembic upgrade head`; otherwise it skips itself."""
import pytest
from sqlalchemy import text

from app.db import get_engine
from app.fetchers.base import Fetcher, NormalizedJob
from app.ingest import ingest_company

SLUG = "__test_company__"


class FakeFetcher(Fetcher):
    source = "greenhouse"

    def __init__(self, jobs):
        self.jobs = jobs

    def fetch(self, slug):
        return self.jobs


def make_job(i, title=None):
    return NormalizedJob(
        source="greenhouse",
        external_id=f"test-{i}",
        title=title or f"Engineer {i}",
        description="desc",
        location="Remote",
        is_remote=True,
        apply_url=f"https://example.com/{i}",
    )


def _cleanup(conn):
    conn.execute(text("DELETE FROM jobs WHERE company_id IN (SELECT id FROM companies WHERE slug = :s)"), {"s": SLUG})
    conn.execute(text("DELETE FROM companies WHERE slug = :s"), {"s": SLUG})


@pytest.fixture
def engine():
    eng = get_engine()
    try:
        with eng.begin() as conn:
            conn.execute(text("SELECT 1 FROM jobs LIMIT 1"))
            _cleanup(conn)
    except Exception as exc:  # no DB, or migrations not applied
        pytest.skip(f"database not ready: {exc.__class__.__name__}")
    yield eng
    with eng.begin() as conn:
        _cleanup(conn)


def _active_ids(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT external_id FROM jobs j JOIN companies c ON c.id = j.company_id WHERE c.slug = :s AND j.is_active"),
            {"s": SLUG},
        )
        return {r[0] for r in rows}


def test_second_run_is_idempotent_and_closes_missing_jobs(engine):
    s1 = ingest_company(engine, FakeFetcher([make_job(1), make_job(2)]), name="Test Co", slug=SLUG)
    assert (s1.inserted, s1.updated, s1.deactivated) == (2, 0, 0)

    s2 = ingest_company(engine, FakeFetcher([make_job(1)]), name="Test Co", slug=SLUG)
    assert (s2.inserted, s2.updated, s2.deactivated) == (0, 1, 1)
    assert _active_ids(engine) == {"test-1"}


def test_job_reappearing_is_reactivated(engine):
    ingest_company(engine, FakeFetcher([make_job(1), make_job(2)]), name="Test Co", slug=SLUG)
    ingest_company(engine, FakeFetcher([make_job(1)]), name="Test Co", slug=SLUG)
    ingest_company(engine, FakeFetcher([make_job(1), make_job(2)]), name="Test Co", slug=SLUG)
    assert _active_ids(engine) == {"test-1", "test-2"}


def test_empty_fetch_does_not_close_everything(engine):
    ingest_company(engine, FakeFetcher([make_job(1)]), name="Test Co", slug=SLUG)
    ingest_company(engine, FakeFetcher([]), name="Test Co", slug=SLUG)
    assert _active_ids(engine) == {"test-1"}
