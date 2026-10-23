"""Role-focused text serialization used consistently by stored and eval embeddings."""
from __future__ import annotations

MAX_EMBEDDING_TEXT_CHARS = 24_000


def job_embedding_text(title: str, description: str) -> str:
    title = title.strip()
    description = description.strip()
    remaining = max(0, MAX_EMBEDDING_TEXT_CHARS - len(title) - 1)
    return f"{title}\n{description[:remaining]}".strip()


def profile_embedding_text(skills: list[str] | None, titles: list[str] | None) -> str:
    parts = []
    if titles:
        parts.append("Target and past job titles: " + ", ".join(titles))
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    return "\n".join(parts)[:MAX_EMBEDDING_TEXT_CHARS]
