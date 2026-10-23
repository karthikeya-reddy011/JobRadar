"""Evaluate the frozen keyword baseline against human-labeled snapshot pairs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from eval_utils import (
    DEFAULT_LABELS,
    DEFAULT_SNAPSHOT,
    eligibility_summary,
    keyword_score,
    load_snapshot,
    uses_role_only_candidate_sources,
)


def precision_at_k(ranked_grades: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    return sum(grade >= 1 for grade in ranked_grades[:k]) / k


def ndcg_at_k(ranked_grades: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")

    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades[:k]))

    actual = dcg(ranked_grades)
    ideal = dcg(sorted(ranked_grades, reverse=True))
    return actual / ideal if ideal else 0.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cosine similarity requires non-empty vectors of equal length.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return dot / (left_norm * right_norm)


def load_embedding_artifact(path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read embedding artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ValueError("Embedding artifact must contain a JSON object.")
    if artifact.get("version") != 1:
        raise ValueError("Unsupported embedding artifact version.")
    if artifact.get("snapshot_id") != snapshot["snapshot_id"]:
        raise ValueError("Embedding artifact belongs to a different snapshot.")
    if artifact.get("dimensions") != 1536:
        raise ValueError("Embedding artifact dimensions must match VECTOR(1536).")
    if not isinstance(artifact.get("model"), str) or not artifact["model"]:
        raise ValueError("Embedding artifact is missing its model name.")
    profiles = artifact.get("profiles")
    jobs = artifact.get("jobs")
    if not isinstance(profiles, dict) or not isinstance(jobs, dict):
        raise ValueError("Embedding artifact must contain profile and job vectors.")

    expected_profiles = set(snapshot["candidate_pool"])
    expected_jobs = {
        str(job_id)
        for job_ids in snapshot["candidate_pool"].values()
        for job_id in job_ids
    }
    if set(profiles) != expected_profiles or set(jobs) != expected_jobs:
        raise ValueError("Embedding artifact does not contain exactly the snapshot candidates.")
    for vector in [*profiles.values(), *jobs.values()]:
        if (
            not isinstance(vector, list)
            or len(vector) != artifact["dimensions"]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError("Embedding artifact contains an invalid vector.")
    return artifact


def load_gemini_rerank_artifact(path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Gemini reranking artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("version") != 1:
        raise ValueError("Unsupported or invalid Gemini reranking artifact.")
    if artifact.get("snapshot_id") != snapshot["snapshot_id"]:
        raise ValueError("Gemini reranking artifact belongs to a different snapshot.")
    if artifact.get("candidate_source") != "keyword_score":
        raise ValueError("Gemini reranking artifact must use the keyword candidate shortlist.")
    if not isinstance(artifact.get("model"), str) or not artifact["model"]:
        raise ValueError("Gemini reranking artifact is missing its model name.")
    rankings = artifact.get("rankings")
    shortlist_sizes = artifact.get("shortlist_sizes")
    if (
        not isinstance(rankings, dict)
        or not isinstance(shortlist_sizes, dict)
        or set(rankings) != set(snapshot["candidate_pool"])
        or set(shortlist_sizes) != set(snapshot["candidate_pool"])
    ):
        raise ValueError("Gemini reranking artifact does not match the snapshot profiles.")
    for profile_name, ranked_ids in rankings.items():
        candidate_ids = {str(job_id) for job_id in snapshot["candidate_pool"][profile_name]}
        shortlist_size = shortlist_sizes[profile_name]
        if (
            not isinstance(ranked_ids, list)
            or any(not isinstance(job_id, str) for job_id in ranked_ids)
            or len(ranked_ids) != len(set(ranked_ids))
            or not ranked_ids
            or not set(ranked_ids).issubset(candidate_ids)
            or not isinstance(shortlist_size, int)
            or isinstance(shortlist_size, bool)
            or shortlist_size != len(ranked_ids)
        ):
            raise ValueError(f"Gemini reranking artifact has an invalid ranking for {profile_name!r}.")
    return artifact


def load_labels(path: Path, snapshot: dict[str, Any]) -> dict[tuple[str, str], int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read labels {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("labels"), list):
        raise ValueError(f"Labels file {path} must contain a JSON object with a labels list.")
    if data.get("snapshot_id") != snapshot["snapshot_id"]:
        raise ValueError("Labels belong to a different snapshot.")
    labels: dict[tuple[str, str], int] = {}
    for item in data.get("labels", []):
        if not isinstance(item, dict) or not all(
            key in item for key in ("profile", "job_id", "label")
        ):
            raise ValueError("Every label must include profile, job_id, and label.")
        profile_name = item.get("profile")
        raw_job_id = item.get("job_id")
        grade = item.get("label")
        if (
            not isinstance(profile_name, str)
            or not isinstance(raw_job_id, (int, str))
            or isinstance(raw_job_id, bool)
        ):
            raise ValueError("Label profile names and job IDs must be strings or integers.")
        job_id = str(raw_job_id)
        if profile_name not in snapshot["candidate_pool"]:
            raise ValueError(f"Label references unknown profile {profile_name!r}.")
        if job_id not in snapshot["candidate_pool"][profile_name]:
            raise ValueError(f"Label references job {job_id!r} outside the candidate pool.")
        if not isinstance(grade, int) or isinstance(grade, bool) or grade not in (0, 1, 2):
            raise ValueError(f"Invalid relevance label {grade!r}; expected 0, 1, or 2.")
        key = (profile_name, job_id)
        if key in labels:
            raise ValueError(f"Duplicate label for profile {profile_name!r}, job {job_id!r}.")
        labels[key] = grade

    expected = {
        (profile_name, str(job_id))
        for profile_name, job_ids in snapshot["candidate_pool"].items()
        for job_id in job_ids
    }
    missing = expected - labels.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} candidate(s) are unlabeled; finish labeling before evaluation."
        )
    return labels


def _evaluate_ranker(
    snapshot: dict[str, Any],
    labels: dict[tuple[str, str], int],
) -> dict[str, Any]:
    profile_results: dict[str, dict[str, float]] = {}
    for profile_name, job_ids in snapshot["candidate_pool"].items():
        ranked_ids = _keyword_order(snapshot, profile_name)
        grades = [labels[(profile_name, job_id)] for job_id in ranked_ids]
        profile_results[profile_name] = {
            "precision@5": precision_at_k(grades, 5),
            "nDCG@10": ndcg_at_k(grades, 10),
        }
    count = len(profile_results)
    if not count:
        raise ValueError("Snapshot contains no profiles to evaluate.")
    return {
        "profiles": profile_results,
        "macro_average": {
            metric: sum(result[metric] for result in profile_results.values()) / count
            for metric in ("precision@5", "nDCG@10")
        },
    }


def _evaluate_embedding_ranker(
    snapshot: dict[str, Any],
    labels: dict[tuple[str, str], int],
    embeddings: dict[str, Any],
) -> dict[str, Any]:
    profile_results: dict[str, dict[str, float]] = {}
    for profile_name, job_ids in snapshot["candidate_pool"].items():
        profile_vector = embeddings["profiles"][profile_name]
        ranked_ids = sorted(
            (str(job_id) for job_id in job_ids),
            key=lambda job_id: (
                -cosine_similarity(profile_vector, embeddings["jobs"][job_id]),
                int(job_id),
            ),
        )
        grades = [labels[(profile_name, job_id)] for job_id in ranked_ids]
        profile_results[profile_name] = {
            "precision@5": precision_at_k(grades, 5),
            "nDCG@10": ndcg_at_k(grades, 10),
        }
    count = len(profile_results)
    if not count:
        raise ValueError("Snapshot contains no profiles to evaluate.")
    return {
        "profiles": profile_results,
        "macro_average": {
            metric: sum(result[metric] for result in profile_results.values()) / count
            for metric in ("precision@5", "nDCG@10")
        },
    }


def _evaluate_gemini_reranker(
    snapshot: dict[str, Any],
    labels: dict[tuple[str, str], int],
    reranking: dict[str, Any],
) -> dict[str, Any]:
    profile_results: dict[str, dict[str, float]] = {}
    for profile_name, job_ids in snapshot["candidate_pool"].items():
        reranked_ids = reranking["rankings"][profile_name]
        reranked_set = set(reranked_ids)
        remaining_ids = [
            job_id for job_id in _keyword_order(snapshot, profile_name)
            if job_id not in reranked_set
        ]
        ranked_ids = [*reranked_ids, *remaining_ids]
        grades = [labels[(profile_name, job_id)] for job_id in ranked_ids]
        profile_results[profile_name] = {
            "precision@5": precision_at_k(grades, 5),
            "nDCG@10": ndcg_at_k(grades, 10),
        }
    count = len(profile_results)
    if not count:
        raise ValueError("Snapshot contains no profiles to evaluate.")
    return {
        "profiles": profile_results,
        "macro_average": {
            metric: sum(result[metric] for result in profile_results.values()) / count
            for metric in ("precision@5", "nDCG@10")
        },
    }


def _keyword_order(snapshot: dict[str, Any], profile_name: str) -> list[str]:
    profile = snapshot["profiles"][profile_name]
    return sorted(
        (str(job_id) for job_id in snapshot["candidate_pool"][profile_name]),
        key=lambda job_id: (
            -keyword_score(profile, snapshot["jobs"][job_id]),
            int(job_id),
        ),
    )


def evaluate(
    snapshot: dict[str, Any],
    labels: dict[tuple[str, str], int],
    embeddings: dict[str, Any] | None = None,
    gemini_reranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure role relevance; summarize eligibility filters without affecting rankings."""
    label_counts = {str(grade): 0 for grade in (0, 1, 2)}
    for grade in labels.values():
        label_counts[str(grade)] += 1

    candidate_sources: dict[str, dict[str, int]] = {}
    for profile_name, job_ids in snapshot["candidate_pool"].items():
        assignments = snapshot.get("candidate_sources", {}).get(profile_name, {})
        counts: dict[str, int] = {}
        for job_id in job_ids:
            source = assignments.get(str(job_id), "unknown")
            counts[source] = counts.get(source, 0) + 1
        candidate_sources[profile_name] = counts

    result = {
        "role_relevance_keyword_baseline": _evaluate_ranker(snapshot, labels),
        "label_distribution": label_counts,
        "candidate_sources": candidate_sources,
        "eligibility_filter_summary": eligibility_summary(snapshot),
    }
    if embeddings is not None:
        result["embedding_ranker"] = _evaluate_embedding_ranker(
            snapshot, labels, embeddings
        )
        result["embedding_model"] = embeddings["model"]
    if gemini_reranking is not None:
        result["gemini_reranker"] = _evaluate_gemini_reranker(
            snapshot, labels, gemini_reranking
        )
        result["gemini_reranker_model"] = gemini_reranking["model"]
        result["gemini_reranker_shortlist_sizes"] = gemini_reranking["shortlist_sizes"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="snapshot-bound vectors generated by eval/embed_snapshot.py",
    )
    parser.add_argument(
        "--gemini-rerank",
        type=Path,
        help="snapshot-bound reranking artifact generated by eval/rerank_gemini.py",
    )
    args = parser.parse_args(argv)
    try:
        snapshot = load_snapshot(args.snapshot)
        labels = load_labels(args.labels, snapshot)
        embeddings = (
            load_embedding_artifact(args.embeddings, snapshot)
            if args.embeddings
            else None
        )
        gemini_reranking = (
            load_gemini_rerank_artifact(args.gemini_rerank, snapshot)
            if args.gemini_rerank
            else None
        )
        results = evaluate(snapshot, labels, embeddings, gemini_reranking)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(
        {
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_version": snapshot["version"],
            "profile_count": len(snapshot["candidate_pool"]),
            "candidate_count": sum(map(len, snapshot["candidate_pool"].values())),
            **({
                "warning": (
                    "This is a legacy snapshot without role-only candidate-source metadata. "
                    "Regenerate and relabel it using the current workflow before comparing "
                    "its relevance metrics."
                )
            } if snapshot["version"] < 3 and not uses_role_only_candidate_sources(snapshot) else {}),
            **results,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
