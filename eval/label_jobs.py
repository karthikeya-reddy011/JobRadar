"""Export a frozen eval snapshot and interactively label its candidate jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sqlalchemy import text

from app.db import get_engine
from eval_utils import (
    DEFAULT_LABELS,
    DEFAULT_SNAPSHOT,
    SNAPSHOT_VERSION,
    load_snapshot,
    snapshot_digest,
    term_matches,
)

PROFILE_QUERY = text("""
    SELECT name, skills, years_experience, titles, locations, remote_ok, seniority
    FROM profiles
    WHERE (:all_profiles OR name = ANY(:profile_names))
    ORDER BY name
""")

JOB_QUERY = text("""
    SELECT j.id, c.name AS company, j.title, j.description, j.location,
           j.is_remote, j.seniority
    FROM jobs AS j
    JOIN companies AS c ON c.id = j.company_id
    WHERE j.is_active
    ORDER BY j.id
""")


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it.")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(value, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _candidate_groups(
    profile: dict[str, Any],
    jobs: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    keyword_ids = [
        job_id for job_id, job in jobs.items()
        if any(
            term_matches(f"{job['title']}\n{job['description']}", str(skill))
            for skill in profile.get("skills", [])
        )
    ]
    keyword_ids.sort(
        key=lambda job_id: (
            -sum(
                term_matches(
                    f"{jobs[job_id]['title']}\n{jobs[job_id]['description']}",
                    str(skill),
                )
                for skill in profile.get("skills", [])
            ),
            int(job_id),
        )
    )
    profile_titles = [str(title) for title in profile.get("titles", []) if str(title).strip()]
    title_ids = [
        job_id for job_id, job in jobs.items()
        if any(term_matches(job["title"], title) for title in profile_titles)
    ]
    title_ids.sort(
        key=lambda job_id: (
            -sum(term_matches(jobs[job_id]["title"], title) for title in profile_titles),
            int(job_id),
        )
    )
    return {
        "skill_keyword": keyword_ids,
        "title_match": title_ids,
    }


def _sample_candidate_pool(
    profile: dict[str, Any],
    jobs: dict[str, dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> tuple[list[str], dict[str, str]]:
    groups = _candidate_groups(profile, jobs)
    random_ids = list(jobs)
    random.Random(seed).shuffle(random_ids)
    groups["random"] = random_ids

    pool: list[str] = []
    sources: dict[str, str] = {}
    quota = (size + len(groups) - 1) // len(groups)
    target_size = min(size, len(jobs))

    for source, candidates in groups.items():
        added = 0
        for job_id in candidates:
            if job_id not in sources:
                pool.append(job_id)
                sources[job_id] = source
                added += 1
                if len(pool) >= target_size or added >= quota:
                    break

    while len(pool) < target_size:
        progressed = False
        for source, candidates in groups.items():
            for job_id in candidates:
                if job_id not in sources:
                    pool.append(job_id)
                    sources[job_id] = source
                    progressed = True
                    break
            if len(pool) >= target_size:
                break
        if not progressed:
            break
    return pool, sources


def _candidate_jobs(
    jobs: dict[str, dict[str, Any]],
    candidate_pool: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Keep only sampled candidates in the frozen artifact, not the full live job table."""
    candidate_ids = {
        str(job_id)
        for profile_job_ids in candidate_pool.values()
        for job_id in profile_job_ids
    }
    return {
        job_id: jobs[job_id]
        for job_id in jobs
        if job_id in candidate_ids
    }


def _candidate_pool(
    profile: dict[str, Any],
    jobs: dict[str, dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> list[str]:
    """Compatibility helper returning just job IDs; exports also preserve selection reasons."""
    pool, _ = _sample_candidate_pool(profile, jobs, size=size, seed=seed)
    return pool


def export_snapshot(
    snapshot_path: Path,
    *,
    profile_names: list[str] | None,
    pool_size: int,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    if pool_size < 1:
        raise ValueError("--pool-size must be positive.")

    engine = get_engine()
    with engine.connect() as conn:
        profile_rows = conn.execute(
            PROFILE_QUERY,
            {
                "all_profiles": profile_names is None,
                "profile_names": profile_names or [],
            },
        ).mappings().all()
        job_rows = conn.execute(JOB_QUERY).mappings().all()

    if not profile_rows:
        raise ValueError("No matching profiles found in the database.")
    if not job_rows:
        raise ValueError("No active jobs found in the database.")

    profiles = {
        row["name"]: {
            "skills": row["skills"] or [],
            "years_experience": (
                float(row["years_experience"])
                if row["years_experience"] is not None else None
            ),
            "titles": row["titles"] or [],
            "locations": row["locations"] or [],
            "remote_ok": row["remote_ok"],
            "seniority": row["seniority"],
        }
        for row in profile_rows
    }
    jobs = {
        str(row["id"]): {
            "company": row["company"],
            "title": row["title"],
            "description": row["description"],
            "location": row["location"],
            "is_remote": row["is_remote"],
            "seniority": row["seniority"],
        }
        for row in job_rows
    }
    sampled = {
        name: _sample_candidate_pool(
            profile,
            jobs,
            size=pool_size,
            seed=seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big"),
        )
        for name, profile in profiles.items()
    }
    candidate_pool = {name: result[0] for name, result in sampled.items()}
    candidate_sources = {name: result[1] for name, result in sampled.items()}
    snapshot: dict[str, Any] = {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "profiles": profiles,
        "jobs": _candidate_jobs(jobs, candidate_pool),
        "candidate_pool": candidate_pool,
        "candidate_sources": candidate_sources,
    }
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    _write_json(snapshot_path, snapshot, overwrite=force)
    return snapshot


def _labels_by_key(labels: dict[str, Any], snapshot_id: str) -> dict[tuple[str, str], int]:
    if not isinstance(labels, dict) or not isinstance(labels.get("labels"), list):
        raise ValueError("Labels file must contain a JSON object with a labels list.")
    if labels.get("snapshot_id") != snapshot_id:
        raise ValueError("Labels belong to a different snapshot; use a separate labels file.")
    result: dict[tuple[str, str], int] = {}
    for item in labels.get("labels", []):
        if not isinstance(item, dict) or not all(
            key in item for key in ("profile", "job_id", "label")
        ):
            raise ValueError("Every saved label must include profile, job_id, and label.")
        profile_name = item["profile"]
        job_id = item["job_id"]
        if (
            not isinstance(profile_name, str)
            or not isinstance(job_id, (int, str))
            or isinstance(job_id, bool)
        ):
            raise ValueError("Saved label profile names and job IDs must be strings or integers.")
        grade = item["label"]
        if (
            not isinstance(grade, int)
            or isinstance(grade, bool)
            or grade not in (0, 1, 2)
        ):
            raise ValueError(f"Invalid saved label for {(profile_name, job_id)}.")
        key = (profile_name, str(job_id))
        if key in result:
            raise ValueError(f"Duplicate saved label for {key}.")
        result[key] = grade
    return result


def _render_pair(profile_name: str, profile: dict[str, Any], job: dict[str, Any]) -> str:
    left = [
        f"Profile: {profile_name}",
        f"Skills: {', '.join(profile.get('skills') or []) or '—'}",
        f"Titles: {', '.join(profile.get('titles') or []) or '—'}",
    ]
    right = [
        f"Company: {job.get('company') or '—'}",
        f"Title: {job.get('title') or '—'}",
        "Description:",
        *(job.get("description") or "No description provided.").splitlines(),
    ]
    width = 46
    left_lines = [
        wrapped
        for line in left
        for wrapped in (textwrap.wrap(line, width=width) or [""])
    ]
    right_lines = [
        wrapped
        for line in right
        for wrapped in (textwrap.wrap(line, width=width) or [""])
    ]
    rows = [
        f"{'PROFILE':<{width}} | {'JOB':<{width}}",
        f"{'-' * width}-+-{'-' * width}",
    ]
    rows.extend(
        f"{left_lines[index] if index < len(left_lines) else '':<{width}} | "
        f"{right_lines[index] if index < len(right_lines) else ''}"
        for index in range(max(len(left_lines), len(right_lines)))
    )
    return "\n".join(rows)


def label_snapshot(snapshot_path: Path, labels_path: Path) -> int:
    snapshot = load_snapshot(snapshot_path)
    print(
        "Labeling guideline: Judge role fit using the job title, responsibilities, and skills "
        "against the profile's skills and titles. Use 2 for a strong fit, 1 for a plausible "
        "but imperfect fit, and 0 for a poor role/skills fit. Ignore location, remote "
        "arrangement, seniority, and years of experience; those are separate eligibility "
        "filters, not relevance labels. Do not infer missing details or use scores, sources, "
        "extracted skills, or candidate-selection reasons."
    )
    if labels_path.exists():
        try:
            labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read labels {labels_path}: {exc}") from exc
    else:
        labels_data = {"snapshot_id": snapshot["snapshot_id"], "labels": []}
    labels = _labels_by_key(labels_data, snapshot["snapshot_id"])
    for profile_name, job_id in labels:
        if (
            profile_name not in snapshot["candidate_pool"]
            or job_id not in snapshot["candidate_pool"][profile_name]
        ):
            raise ValueError(
                f"Saved label references a pair outside this snapshot: {profile_name}/{job_id}."
            )
    pending = [
        (profile_name, job_id)
        for profile_name, job_ids in snapshot["candidate_pool"].items()
        for job_id in job_ids
        if (profile_name, str(job_id)) not in labels
    ]
    for index, (profile_name, job_id) in enumerate(pending, start=1):
        print(f"\nPair {index}/{len(pending)}")
        print(_render_pair(
            profile_name,
            snapshot["profiles"][profile_name],
            snapshot["jobs"][str(job_id)],
        ))
        while True:
            response = input("Relevance [0=not relevant, 1=somewhat, 2=strong, q=quit]: ").strip().lower()
            if response == "q":
                return len(labels)
            if response in {"0", "1", "2"}:
                labels[(profile_name, str(job_id))] = int(response)
                _write_json(
                    labels_path,
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "labels": [
                            {"profile": name, "job_id": int(identifier), "label": grade}
                            for (name, identifier), grade in sorted(labels.items())
                        ],
                    },
                )
                break
            print("Enter 0, 1, 2, or q.")
    return len(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="freeze profiles, jobs, and label candidates")
    export_parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    export_parser.add_argument("--profile", action="append", dest="profiles")
    export_parser.add_argument("--pool-size", type=int, default=100)
    export_parser.add_argument("--seed", type=int, default=42)
    export_parser.add_argument("--force", action="store_true")
    label_parser = subparsers.add_parser("label", help="blind-label candidates from a frozen snapshot")
    label_parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    label_parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args(argv)

    try:
        if args.command == "export":
            snapshot = export_snapshot(
                args.snapshot,
                profile_names=args.profiles,
                pool_size=args.pool_size,
                seed=args.seed,
                force=args.force,
            )
            print(
                f"Snapshot {snapshot['snapshot_id'][:12]} exported: "
                f"{len(snapshot['profiles'])} profile(s), "
                f"{len(snapshot['jobs'])} active jobs, "
                f"{sum(map(len, snapshot['candidate_pool'].values()))} candidates."
            )
        else:
            total = label_snapshot(args.snapshot, args.labels)
            print(f"Saved {total} label(s) to {args.labels}.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
