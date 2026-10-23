"""Generate a snapshot-bound embedding artifact for the offline ranking experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.embedding_text import job_embedding_text, profile_embedding_text
from app.llm import EMBEDDING_DIMENSIONS, LLMError, LLMNotConfigured, get_embedder
from eval_utils import DEFAULT_SNAPSHOT, load_snapshot

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "embeddings.json"


def _write_json(path: Path, value: dict[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it.")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temp_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def generate_snapshot_embeddings(
    snapshot: dict[str, Any],
    *,
    output_path: Path,
    batch_size: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    embedder = get_embedder()
    if embedder.dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedder has {embedder.dimensions} dimensions; schema/eval requires "
            f"{EMBEDDING_DIMENSIONS}."
        )
    profiles = snapshot["profiles"]
    candidate_job_ids = sorted(
        {str(job_id) for job_ids in snapshot["candidate_pool"].values() for job_id in job_ids},
        key=int,
    )
    profile_names = sorted(snapshot["candidate_pool"])
    profile_vectors: dict[str, list[float]] = {}
    job_vectors: dict[str, list[float]] = {}

    profile_inputs = [
        profile_embedding_text(
            profiles[name].get("skills") or [],
            profiles[name].get("titles") or [],
        )
        for name in profile_names
    ]
    if any(not value.strip() for value in profile_inputs):
        raise ValueError("Cannot embed a profile with no role skills or titles.")

    for start in range(0, len(profile_names), batch_size):
        vectors = embedder.embed(
            profile_inputs[start : start + batch_size],
            task_type="RETRIEVAL_QUERY",
        )
        if len(vectors) != len(profile_names[start : start + batch_size]):
            raise ValueError("Embedder returned an unexpected number of profile vectors.")
        profile_vectors.update(zip(profile_names[start : start + batch_size], vectors))

    for start in range(0, len(candidate_job_ids), batch_size):
        job_ids = candidate_job_ids[start : start + batch_size]
        texts = [
            job_embedding_text(
                snapshot["jobs"][job_id].get("title") or "",
                snapshot["jobs"][job_id].get("description") or "",
            )
            for job_id in job_ids
        ]
        vectors = embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        if len(vectors) != len(job_ids):
            raise ValueError("Embedder returned an unexpected number of job vectors.")
        job_vectors.update(zip(job_ids, vectors))

    artifact: dict[str, Any] = {
        "version": 1,
        "snapshot_id": snapshot["snapshot_id"],
        "model": embedder.model,
        "dimensions": embedder.dimensions,
        "profiles": profile_vectors,
        "jobs": job_vectors,
    }
    _write_json(output_path, artifact, force=force)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        snapshot = load_snapshot(args.snapshot)
        artifact = generate_snapshot_embeddings(
            snapshot,
            output_path=args.output,
            batch_size=args.batch_size,
            force=args.force,
        )
    except (OSError, ValueError, LLMError, LLMNotConfigured) as exc:
        parser.error(str(exc))
    print(
        f"Saved {len(artifact['profiles'])} profile and {len(artifact['jobs'])} job vectors "
        f"({artifact['dimensions']} dimensions, model {artifact['model']}) to {args.output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
