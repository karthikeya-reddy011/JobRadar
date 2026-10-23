"""Backfill or refresh 1536-dimensional embeddings for active jobs and profiles.

Run from backend/: python -m app.embeddings
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db import get_engine
from app.embedding_text import job_embedding_text, profile_embedding_text
from app.llm import (
    EMBEDDING_DIMENSIONS,
    Embedder,
    LLMError,
    LLMNotConfigured,
    get_embedder,
)

log = logging.getLogger("jobradar.embeddings")

SELECT_JOBS = text("""
    SELECT id, title, description
    FROM jobs
    WHERE is_active
      AND id > :after_id
      AND (:force OR embedding IS NULL OR embedding_model IS DISTINCT FROM :model)
    ORDER BY id
    LIMIT :batch_size
""")

SELECT_PROFILES = text("""
    SELECT id, skills, titles
    FROM profiles
    WHERE id > :after_id
      AND (
          COALESCE(cardinality(skills), 0) > 0
          OR COALESCE(cardinality(titles), 0) > 0
      )
      AND (:force OR embedding IS NULL OR embedding_model IS DISTINCT FROM :model)
    ORDER BY id
    LIMIT :batch_size
""")

UPDATE_JOB = text("""
    UPDATE jobs
    SET embedding = CAST(:embedding AS vector), embedding_model = :model
    WHERE id = :id AND title = :title AND description = :description
""")

UPDATE_PROFILE = text("""
    UPDATE profiles
    SET embedding = CAST(:embedding AS vector), embedding_model = :model
    WHERE id = :id
      AND skills IS NOT DISTINCT FROM CAST(:skills AS TEXT[])
      AND titles IS NOT DISTINCT FROM CAST(:titles AS TEXT[])
""")


@dataclass
class EmbeddingStats:
    jobs_embedded: int = 0
    profiles_embedded: int = 0
    skipped_changed: int = 0


def format_vector(vector: list[float], dimensions: int = EMBEDDING_DIMENSIONS) -> str:
    if len(vector) != dimensions:
        raise ValueError(f"Expected {dimensions}-dimensional vector, got {len(vector)}.")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding vector contains a non-finite value.")
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


def _embed_jobs(
    engine: Engine,
    embedder: Embedder,
    *,
    model: str,
    batch_size: int,
    limit: int | None,
    force: bool,
    stats: EmbeddingStats,
) -> None:
    after_id = 0
    processed = 0
    while limit is None or processed < limit:
        current_batch_size = min(batch_size, limit - processed) if limit is not None else batch_size
        with engine.connect() as conn:
            rows = conn.execute(
                SELECT_JOBS,
                {
                    "after_id": after_id,
                    "force": force,
                    "model": model,
                    "batch_size": current_batch_size,
                },
            ).mappings().all()
        if not rows:
            break

        vectors = embedder.embed(
            [job_embedding_text(row["title"], row["description"] or "") for row in rows],
            task_type="RETRIEVAL_DOCUMENT",
        )
        if len(vectors) != len(rows):
            raise ValueError("Embedder returned an unexpected number of job vectors.")
        with engine.begin() as conn:
            for row, vector in zip(rows, vectors):
                updated = conn.execute(
                    UPDATE_JOB,
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "description": row["description"],
                        "model": model,
                        "embedding": format_vector(vector, embedder.dimensions),
                    },
                ).rowcount
                if updated:
                    stats.jobs_embedded += 1
                else:
                    stats.skipped_changed += 1
                    log.warning("Job %s changed during embedding; leaving it for the next run", row["id"])
        after_id = rows[-1]["id"]
        processed += len(rows)


def _embed_profiles(
    engine: Engine,
    embedder: Embedder,
    *,
    model: str,
    batch_size: int,
    limit: int | None,
    force: bool,
    stats: EmbeddingStats,
) -> None:
    after_id = 0
    processed = 0
    while limit is None or processed < limit:
        current_batch_size = min(batch_size, limit - processed) if limit is not None else batch_size
        with engine.connect() as conn:
            rows = conn.execute(
                SELECT_PROFILES,
                {
                    "after_id": after_id,
                    "force": force,
                    "model": model,
                    "batch_size": current_batch_size,
                },
            ).mappings().all()
        if not rows:
            break

        vectors = embedder.embed(
            [profile_embedding_text(row["skills"] or [], row["titles"] or []) for row in rows],
            task_type="RETRIEVAL_QUERY",
        )
        if len(vectors) != len(rows):
            raise ValueError("Embedder returned an unexpected number of profile vectors.")
        with engine.begin() as conn:
            for row, vector in zip(rows, vectors):
                updated = conn.execute(
                    UPDATE_PROFILE,
                    {
                        "id": row["id"],
                        "skills": row["skills"],
                        "titles": row["titles"],
                        "model": model,
                        "embedding": format_vector(vector, embedder.dimensions),
                    },
                ).rowcount
                if updated:
                    stats.profiles_embedded += 1
                else:
                    stats.skipped_changed += 1
                    log.warning(
                        "Profile %s changed during embedding; leaving it for the next run",
                        row["id"],
                    )
        after_id = rows[-1]["id"]
        processed += len(rows)


def backfill_embeddings(
    engine: Engine,
    embedder: Embedder,
    *,
    model: str,
    entity: str = "all",
    batch_size: int = 32,
    limit: int | None = None,
    force: bool = False,
) -> EmbeddingStats:
    if embedder.dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedder has {embedder.dimensions} dimensions; schema requires {EMBEDDING_DIMENSIONS}."
        )
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive.")
    if entity not in {"all", "jobs", "profiles"}:
        raise ValueError("entity must be all, jobs, or profiles.")

    stats = EmbeddingStats()
    if entity in {"all", "jobs"}:
        _embed_jobs(
            engine, embedder, model=model, batch_size=batch_size, limit=limit,
            force=force, stats=stats,
        )
    if entity in {"all", "profiles"}:
        _embed_profiles(
            engine, embedder, model=model, batch_size=batch_size, limit=limit,
            force=force, stats=stats,
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", choices=("all", "jobs", "profiles"), default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="regenerate vectors even if current")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        embedder = get_embedder()
        stats = backfill_embeddings(
            get_engine(),
            embedder,
            model=embedder.model,
            entity=args.entity,
            batch_size=args.batch_size,
            limit=args.limit,
            force=args.force,
        )
    except (LLMError, LLMNotConfigured, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Embedded {stats.jobs_embedded} job(s) and {stats.profiles_embedded} profile(s); "
        f"skipped {stats.skipped_changed} changed record(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
