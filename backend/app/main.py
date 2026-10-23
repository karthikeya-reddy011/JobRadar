import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.api_profile import router as profile_router
from app.db import get_engine

app = FastAPI(title="JobRadar", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "PUT", "DELETE", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(profile_router)

JobStatus = Literal["saved", "applied", "dismissed"]


class JobStatusIn(BaseModel):
    profile: str = Field(default="me", min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$")
    status: JobStatus


JOB_SELECT = """
    SELECT j.id, c.name AS company, j.title, j.location, j.is_remote,
           j.seniority, j.apply_url, j.posted_at,
           left(j.description, 1800) AS description,
           js.status AS user_status,
           CASE WHEN p.embedding IS NOT NULL AND j.embedding IS NOT NULL
                THEN 1 - (j.embedding <=> p.embedding)
                ELSE NULL END AS match_score
    FROM jobs j
    JOIN companies c ON c.id = j.company_id
    LEFT JOIN profiles p ON p.name = :profile
    LEFT JOIN job_status js ON js.profile_id = p.id AND js.job_id = j.id
"""


def _job_filters(
    *, q: str | None, remote: bool | None, location: str | None,
    seniority: str | None, status: JobStatus | None,
) -> tuple[list[str], dict[str, object]]:
    clauses = ["j.is_active"]
    params: dict[str, object] = {"profile": "me"}
    if q and q.strip():
        clauses.append("(j.title ILIKE :q OR j.description ILIKE :q OR c.name ILIKE :q)")
        params["q"] = f"%{q.strip()}%"
    if remote is not None:
        clauses.append("j.is_remote = :remote")
        params["remote"] = remote
    if location and location.strip():
        clauses.append("(j.location ILIKE :location OR j.location IS NULL)")
        params["location"] = f"%{location.strip()}%"
    if seniority:
        clauses.append("j.seniority = :seniority")
        params["seniority"] = seniority
    if status:
        clauses.append("js.status = :status")
        params["status"] = status
    return clauses, params


@app.get("/health")
def health():
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    with get_engine().connect() as conn:
        row = conn.execute(text("""
            SELECT (SELECT count(*) FROM companies) AS companies,
                   count(*) FILTER (WHERE is_active) AS active_jobs,
                   count(*) AS total_jobs,
                   max(last_seen_at) AS last_fetch
            FROM jobs
        """)).mappings().one()
    return dict(row)


@app.get("/api/jobs")
def list_jobs(
    q: str | None = None,
    remote: bool | None = None,
    location: str | None = Query(None, min_length=1, max_length=100),
    seniority: str | None = None,
    status: JobStatus | None = None,
    sort: Literal["newest", "recommended"] = "newest",
    profile: str = Query("me", min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    clauses, params = _job_filters(
        q=q, remote=remote, location=location, seniority=seniority, status=status
    )
    params.update({"profile": profile, "limit": limit, "offset": offset})
    ordering = (
        "match_score DESC NULLS LAST, j.posted_at DESC NULLS LAST, j.id DESC"
        if sort == "recommended"
        else "j.posted_at DESC NULLS LAST, j.id DESC"
    )
    sql = f"{JOB_SELECT} WHERE {' AND '.join(clauses)} ORDER BY {ordering} LIMIT :limit OFFSET :offset"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(row) for row in rows]


@app.get("/api/recommendations")
def recommendations(
    profile: str = Query("me", min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$"),
    q: str | None = None,
    remote: bool | None = None,
    location: str | None = Query(None, min_length=1, max_length=100),
    limit: int = Query(30, ge=1, le=100),
):
    clauses, params = _job_filters(
        q=q, remote=remote, location=location, seniority=None, status=None
    )
    clauses.append("p.id IS NOT NULL")
    params.update({"profile": profile, "limit": limit})
    sql = (
        f"{JOB_SELECT} WHERE {' AND '.join(clauses)} "
        "ORDER BY match_score DESC NULLS LAST, j.posted_at DESC NULLS LAST, j.id DESC "
        "LIMIT :limit"
    )
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        profile_row = conn.execute(
            text("SELECT embedding IS NOT NULL AS has_embedding FROM profiles WHERE name = :profile"),
            {"profile": profile},
        ).mappings().first()
    if profile_row is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile!r} does not exist. Save a profile first.")
    return {
        "profile": profile,
        "ranking_mode": "embeddings" if profile_row["has_embedding"] else "newest_fallback",
        "results": [dict(row) for row in rows],
    }


@app.put("/api/jobs/{job_id}/status")
def set_job_status(job_id: int, body: JobStatusIn):
    with get_engine().begin() as conn:
        profile = conn.execute(
            text("SELECT id FROM profiles WHERE name = :name"), {"name": body.profile}
        ).mappings().first()
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Profile {body.profile!r} does not exist.")
        exists = conn.execute(
            text("SELECT id FROM jobs WHERE id = :id AND is_active"), {"id": job_id}
        ).first()
        if exists is None:
            raise HTTPException(status_code=404, detail="Active job not found.")
        conn.execute(text("""
            INSERT INTO job_status (profile_id, job_id, status)
            VALUES (:profile_id, :job_id, :status)
            ON CONFLICT (profile_id, job_id) DO UPDATE SET status = EXCLUDED.status
        """), {"profile_id": profile["id"], "job_id": job_id, "status": body.status})
    return {"job_id": job_id, "profile": body.profile, "status": body.status}


@app.delete("/api/jobs/{job_id}/status")
def clear_job_status(
    job_id: int,
    profile: str = Query("me", min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$"),
):
    with get_engine().begin() as conn:
        result = conn.execute(text("""
            DELETE FROM job_status js USING profiles p
            WHERE js.profile_id = p.id AND p.name = :profile AND js.job_id = :job_id
        """), {"profile": profile, "job_id": job_id})
    return {"deleted": bool(result.rowcount)}
