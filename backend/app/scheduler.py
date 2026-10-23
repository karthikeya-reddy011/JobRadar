"""Daily ingestion on a schedule. Run in its own terminal:

    python -m app.scheduler --run-now                       # ingest now, then daily at 06:00 UTC
    python -m app.scheduler --hour 7 --minute 30 --tz Asia/Kolkata

Alternatives: cron or Windows Task Scheduler calling `python -m app.ingest`.
"""
from __future__ import annotations

import argparse
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import DEFAULT_COMPANIES_FILE
from app.db import get_engine
from app.ingest import load_companies, run_ingest

log = logging.getLogger("jobradar.scheduler")


def run_once() -> None:
    try:
        results, failures = run_ingest(get_engine(), load_companies(DEFAULT_COMPANIES_FILE))
        log.info(
            "Ingest finished: %d companies ok, %d failed, %d jobs fetched",
            len(results), len(failures), sum(r.fetched for r in results),
        )
    except Exception:  # noqa: BLE001  one bad run must never kill the scheduler
        log.exception("Ingest run crashed; will retry at the next scheduled time")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ingestion daily")
    parser.add_argument("--hour", type=int, default=6)
    parser.add_argument("--minute", type=int, default=0)
    parser.add_argument("--tz", default="UTC", help="e.g. Asia/Kolkata")
    parser.add_argument("--run-now", action="store_true", help="also ingest immediately on startup")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scheduler = BlockingScheduler(timezone=args.tz)
    scheduler.add_job(
        run_once,
        CronTrigger(hour=args.hour, minute=args.minute, timezone=args.tz),
        id="daily-ingest",
        max_instances=1,      # never overlap two runs
        coalesce=True,        # if the machine slept through several runs, do just one
        misfire_grace_time=3600,
    )
    log.info("Scheduled daily ingest at %02d:%02d %s. Ctrl+C to stop.", args.hour, args.minute, args.tz)
    if args.run_now:
        run_once()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
