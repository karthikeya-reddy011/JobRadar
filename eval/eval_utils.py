"""Shared helpers for the frozen evaluation snapshot."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "snapshot.json"
DEFAULT_LABELS = Path(__file__).resolve().parent / "labels.json"
SNAPSHOT_VERSION = 3
SUPPORTED_SNAPSHOT_VERSIONS = {1, 2, SNAPSHOT_VERSION}


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    content = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_snapshot(path: Path = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read snapshot {path}: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise ValueError(f"Snapshot {path} must contain a JSON object.")
    if snapshot.get("version") not in SUPPORTED_SNAPSHOT_VERSIONS:
        raise ValueError(f"Unsupported snapshot version in {path}.")
    if not all(key in snapshot for key in ("snapshot_id", "profiles", "jobs", "candidate_pool")):
        raise ValueError(f"Snapshot {path} is missing required fields.")
    if not all(isinstance(snapshot[key], dict) for key in ("profiles", "jobs", "candidate_pool")):
        raise ValueError(f"Snapshot {path} has invalid profile, job, or candidate-pool data.")
    if not isinstance(snapshot["snapshot_id"], str):
        raise ValueError(f"Snapshot {path} has an invalid snapshot ID.")
    if any(not isinstance(value, dict) for value in snapshot["profiles"].values()):
        raise ValueError(f"Snapshot {path} contains an invalid profile.")
    if any(not isinstance(value, dict) for value in snapshot["jobs"].values()):
        raise ValueError(f"Snapshot {path} contains an invalid job.")
    for profile_name, job_ids in snapshot["candidate_pool"].items():
        if profile_name not in snapshot["profiles"] or not isinstance(job_ids, list):
            raise ValueError(f"Snapshot {path} contains an invalid candidate pool.")
        if any(str(job_id) not in snapshot["jobs"] for job_id in job_ids):
            raise ValueError(f"Snapshot {path} candidate pool references an unknown job.")
    if snapshot.get("version") >= 2:
        sources = snapshot.get("candidate_sources")
        if not isinstance(sources, dict):
            raise ValueError(f"Snapshot {path} is missing candidate source metadata.")
        for profile_name, job_ids in snapshot["candidate_pool"].items():
            profile_sources = sources.get(profile_name)
            if not isinstance(profile_sources, dict) or set(profile_sources) != {
                str(job_id) for job_id in job_ids
            }:
                raise ValueError(f"Snapshot {path} has invalid candidate source metadata.")
            if snapshot["version"] >= 3 and not set(profile_sources.values()).issubset(
                {"skill_keyword", "title_match", "random"}
            ):
                raise ValueError(f"Snapshot {path} contains a non-role candidate source.")
    if snapshot["snapshot_id"] != snapshot_digest(snapshot):
        raise ValueError(f"Snapshot {path} has changed since it was created.")
    return snapshot


@lru_cache(maxsize=1024)
def _term_pattern(term: str) -> re.Pattern[str] | None:
    normalized = re.escape(term.strip())
    if not normalized:
        return None
    normalized = normalized.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){normalized}(?!\w)", flags=re.IGNORECASE)


def term_matches(text: str, term: str) -> bool:
    pattern = _term_pattern(term)
    return pattern.search(text) is not None if pattern else False


def keyword_score(profile: dict[str, Any], job: dict[str, Any]) -> int:
    """Count distinct profile skills and titles present in the job's title or description."""
    job_title = str(job.get("title") or "")
    job_text = f"{job_title}\n{job.get('description') or ''}"
    score = sum(term_matches(job_text, str(skill)) for skill in profile.get("skills", []))
    score += 2 * sum(term_matches(job_title, str(title)) for title in profile.get("titles", []))
    return score


_SENIORITY_GROUP = {
    "intern": "early",
    "junior": "early",
    "mid": "experienced",
    "senior": "experienced",
    "lead": "leadership",
    "staff": "leadership",
    "principal": "leadership",
    "manager": "management",
}


def _location_status(profile: dict[str, Any], job: dict[str, Any]) -> str:
    is_remote = job.get("is_remote")
    remote_ok = profile.get("remote_ok")
    if is_remote is True:
        if remote_ok is True:
            return "compatible"
        return "incompatible" if remote_ok is False else "unknown"

    profile_locations = [
        str(location).strip().casefold()
        for location in profile.get("locations", [])
        if str(location).strip()
    ]
    job_location = str(job.get("location") or "").strip().casefold()
    if not profile_locations or not job_location:
        return "unknown"
    if any(location in job_location or job_location in location for location in profile_locations):
        return "compatible"
    return "incompatible"


def _experience_status(profile: dict[str, Any], job: dict[str, Any]) -> str:
    years = profile.get("years_experience")
    minimum = job.get("experience_min_years")
    if (
        not isinstance(years, (int, float))
        or isinstance(years, bool)
        or not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
    ):
        return "unknown"
    return "compatible" if years >= minimum else "incompatible"


def eligibility_summary(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Report filter coverage separately; filters never alter role-relevance ranking metrics."""
    result: dict[str, dict[str, Any]] = {}
    for profile_name, job_ids in snapshot["candidate_pool"].items():
        profile = snapshot["profiles"][profile_name]
        criteria = {
            "location_remote": {"compatible": 0, "incompatible": 0, "unknown": 0},
            "experience": {"compatible": 0, "incompatible": 0, "unknown": 0},
        }
        excluded = 0
        for job_id in job_ids:
            job = snapshot["jobs"][str(job_id)]
            location_status = _location_status(profile, job)
            experience_status = _experience_status(profile, job)
            criteria["location_remote"][location_status] += 1
            criteria["experience"][experience_status] += 1
            if "incompatible" in (location_status, experience_status):
                excluded += 1
        result[profile_name] = {
            "candidate_count": len(job_ids),
            "not_excluded_by_known_filters": len(job_ids) - excluded,
            "excluded_by_known_filter": excluded,
            "criteria": criteria,
        }
    return result


def uses_role_only_candidate_sources(snapshot: dict[str, Any]) -> bool:
    """Return whether source metadata proves that pool selection ignored logistics."""
    sources = snapshot.get("candidate_sources")
    if not isinstance(sources, dict) or not sources:
        return False
    allowed_sources = {"skill_keyword", "title_match", "random"}
    return all(
        isinstance(profile_sources, dict)
        and set(profile_sources.values()).issubset(allowed_sources)
        for profile_sources in sources.values()
    )
