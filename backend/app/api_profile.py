import logging
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.db import get_engine
from app.embedding_text import profile_embedding_text
from app.embeddings import format_vector
from app.llm import LLMError, LLMNotConfigured, get_embedder, get_llm
from app.profile import (
    MAX_PDF_BYTES,
    ProfileError,
    clean_strings,
    extract_profile,
    extract_text_from_pdf,
    heuristic_extract,
)
from app.skills import normalize_skills

router = APIRouter(prefix="/api", tags=["profile"])
log = logging.getLogger("jobradar.profile")

Seniority = Literal["intern", "junior", "mid", "senior", "lead", "staff", "principal", "manager"]


class ProfileDraft(BaseModel):
    skills: list[str] = Field(default_factory=list, max_length=200)
    years_experience: float | None = Field(default=None, ge=0, le=60)
    titles: list[str] = Field(default_factory=list, max_length=30)
    locations: list[str] = Field(default_factory=list, max_length=20)
    remote_ok: bool | None = None
    seniority: Seniority | None = None


class ProfileIn(ProfileDraft):
    # Short handle. Phase 3's eval file refers to profiles by this name.
    name: str = Field(default="me", min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$")


class ProfileOut(ProfileIn):
    id: int
    embedding_status: Literal["ready", "pending", "not_applicable"] = "not_applicable"
    embedding_message: str | None = None


class ResumeResponse(BaseModel):
    profile: ProfileDraft
    warnings: list[str]
    used_llm: bool


UPSERT_PROFILE = text("""
    INSERT INTO profiles (name, skills, years_experience, titles, locations, remote_ok, seniority, updated_at)
    VALUES (:name, CAST(:skills AS TEXT[]), :years, CAST(:titles AS TEXT[]),
            CAST(:locations AS TEXT[]), :remote_ok, :seniority, now())
    ON CONFLICT (name) DO UPDATE SET
        skills = EXCLUDED.skills,
        years_experience = EXCLUDED.years_experience,
        titles = EXCLUDED.titles,
        locations = EXCLUDED.locations,
        remote_ok = EXCLUDED.remote_ok,
        seniority = EXCLUDED.seniority,
        embedding = CASE
            WHEN profiles.skills IS DISTINCT FROM EXCLUDED.skills
              OR profiles.titles IS DISTINCT FROM EXCLUDED.titles
            THEN NULL ELSE profiles.embedding END,
        embedding_model = CASE
            WHEN profiles.skills IS DISTINCT FROM EXCLUDED.skills
              OR profiles.titles IS DISTINCT FROM EXCLUDED.titles
            THEN NULL ELSE profiles.embedding_model END,
        updated_at = now()
    RETURNING id, name, skills, years_experience, titles, locations, remote_ok, seniority,
              embedding IS NOT NULL AS has_embedding
""")

SELECT_PROFILE = text("""
    SELECT id, name, skills, years_experience, titles, locations, remote_ok, seniority,
           embedding IS NOT NULL AS has_embedding
    FROM profiles WHERE name = :name
""")

UPDATE_PROFILE_EMBEDDING = text("""
    UPDATE profiles
    SET embedding = CAST(:embedding AS vector), embedding_model = :model
    WHERE id = :id
      AND skills IS NOT DISTINCT FROM CAST(:skills AS TEXT[])
      AND titles IS NOT DISTINCT FROM CAST(:titles AS TEXT[])
""")


def _to_out(row) -> ProfileOut:
    return ProfileOut(
        id=row["id"],
        name=row["name"],
        skills=row["skills"] or [],
        years_experience=float(row["years_experience"]) if row["years_experience"] is not None else None,
        titles=row["titles"] or [],
        locations=row["locations"] or [],
        remote_ok=row["remote_ok"],
        seniority=row["seniority"],
    )


@router.post("/resume", response_model=ResumeResponse)
def upload_resume(file: UploadFile = File(...)):
    """Upload a text PDF, get back an editable profile. Nothing is saved: the user reviews and
    edits it, then PUTs it to /api/profile. The resume text is discarded after this request."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")
    content = file.file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="That PDF is too large (limit 5 MB).")
    try:
        resume_text = extract_text_from_pdf(content)
        try:
            llm = get_llm()
        except LLMNotConfigured:
            result = heuristic_extract(resume_text)
        else:
            result = extract_profile(resume_text, llm)
    except ProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ResumeResponse(
        profile=ProfileDraft(**result.profile.to_dict()),
        warnings=result.warnings,
        used_llm=result.used_llm,
    )


@router.put("/profile", response_model=ProfileOut)
def save_profile(body: ProfileIn):
    """Create or update the profile called `name`. Skills are re-normalized server-side, so
    typing "reactjs" in the editor is stored as "React"."""
    params = {
        "name": body.name,
        "skills": normalize_skills(body.skills),
        "years": body.years_experience,
        "titles": clean_strings(body.titles, max_items=30),
        "locations": clean_strings(body.locations, max_items=20),
        "remote_ok": body.remote_ok,
        "seniority": body.seniority,
    }
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(UPSERT_PROFILE, params).mappings().one()

    result = _to_out(row)
    if not (result.skills or result.titles):
        return result.model_copy(update={"embedding_status": "not_applicable"})
    if row["has_embedding"]:
        return result.model_copy(update={"embedding_status": "ready"})

    try:
        embedder = get_embedder()
        vectors = embedder.embed(
            [profile_embedding_text(result.skills, result.titles)],
            task_type="RETRIEVAL_QUERY",
        )
        if len(vectors) != 1:
            raise LLMError("Embedding provider did not return exactly one profile vector.")
        vector = vectors[0]
    except (LLMError, LLMNotConfigured) as exc:
        log.warning("Profile %s saved but embedding refresh is pending: %s", result.name, exc)
        return result.model_copy(update={
            "embedding_status": "pending",
            "embedding_message": str(exc),
        })

    with engine.begin() as conn:
        updated = conn.execute(
            UPDATE_PROFILE_EMBEDDING,
            {
                "id": result.id,
                "skills": result.skills,
                "titles": result.titles,
                "embedding": format_vector(vector, embedder.dimensions),
                "model": embedder.model,
            },
        ).rowcount
    if not updated:
        message = "Profile changed while its embedding was being generated; save it again to retry."
        log.warning("Profile %s changed during embedding refresh", result.name)
        return result.model_copy(update={
            "embedding_status": "pending",
            "embedding_message": message,
        })
    return result.model_copy(update={"embedding_status": "ready"})


@router.get("/profile/{name}", response_model=ProfileOut)
def get_profile(name: str):
    with get_engine().connect() as conn:
        row = conn.execute(SELECT_PROFILE, {"name": name}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="No such profile.")
    result = _to_out(row)
    return result.model_copy(update={
        "embedding_status": "ready" if row["has_embedding"] else "pending"
    })


@router.delete("/profile/{name}")
def delete_profile(name: str):
    """Delete-my-data: removes the profile; job_status rows go with it (ON DELETE CASCADE)."""
    with get_engine().begin() as conn:
        deleted = conn.execute(text("DELETE FROM profiles WHERE name = :name"), {"name": name}).rowcount
    return {"deleted": bool(deleted)}
