import json
import sys
from pathlib import Path

import httpx
import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from app.embedding_text import job_embedding_text, profile_embedding_text
from app.embeddings import format_vector
from app.llm import (
    EMBEDDING_DIMENSIONS,
    LLMError,
    OpenAIEmbedder,
)
from run_eval import _evaluate_embedding_ranker, cosine_similarity, load_embedding_artifact


def _vector(value):
    return [value] * EMBEDDING_DIMENSIONS


def test_openai_embedder_requests_schema_dimensions_and_restores_input_order():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        seen["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": _vector(0.2)},
                    {"index": 0, "embedding": _vector(0.1)},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = OpenAIEmbedder("test-key", client=client, sleep=lambda _: None)
        vectors = embedder.embed(["first", "second"])

    assert seen["payload"]["model"] == "text-embedding-3-small"
    assert seen["payload"]["dimensions"] == 1536
    assert seen["authorization"] == "Bearer test-key"
    assert len(vectors) == 2
    assert vectors[0][0] == pytest.approx(0.1)
    assert vectors[1][0] == pytest.approx(0.2)


def test_openai_embedder_rejects_wrong_dimension():
    def handler(request):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = OpenAIEmbedder("test-key", client=client, sleep=lambda _: None)
        with pytest.raises(LLMError, match="dimension mismatch"):
            embedder.embed(["text"])


def test_openai_embedder_rejects_blank_input_and_skips_empty_request():
    embedder = OpenAIEmbedder("test-key")

    assert embedder.embed([]) == []
    with pytest.raises(LLMError, match="non-empty"):
        embedder.embed(["  "])


def test_embedding_text_is_role_focused_and_bounded():
    profile_text = profile_embedding_text(["Python"], ["Backend Engineer"])
    job_text = job_embedding_text("Backend Engineer", "x" * 30_000)

    assert "Python" in profile_text
    assert "Backend Engineer" in profile_text
    assert "Seattle" not in profile_embedding_text(["Python"], ["Backend Engineer"])
    assert "Backend Engineer" in job_text
    assert len(job_text) <= 24_000


def test_format_vector_validates_dimension_and_finite_values():
    assert format_vector([0.5, -1.25], dimensions=2) == "[0.5,-1.25]"
    with pytest.raises(ValueError, match="Expected"):
        format_vector([1.0], dimensions=2)
    with pytest.raises(ValueError, match="non-finite"):
        format_vector([float("nan")], dimensions=1)


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [0.8, 0.6]) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0], [1.0])


def test_embedding_ranker_uses_cosine_similarity():
    snapshot = {
        "profiles": {"me": {}},
        "jobs": {"1": {}, "2": {}},
        "candidate_pool": {"me": ["1", "2"]},
    }
    labels = {("me", "1"): 2, ("me", "2"): 0}
    vectors = {
        "model": "test-model",
        "profiles": {"me": [1.0, 0.0]},
        "jobs": {"1": [0.9, 0.1], "2": [0.0, 1.0]},
    }

    metrics = _evaluate_embedding_ranker(snapshot, labels, vectors)

    assert metrics["profiles"]["me"]["nDCG@10"] == 1.0


def test_embedding_artifact_must_match_snapshot_and_vector_shape(tmp_path):
    snapshot = {
        "snapshot_id": "frozen",
        "candidate_pool": {"me": ["1"]},
    }
    artifact_path = tmp_path / "embeddings.json"
    artifact = {
        "version": 1,
        "snapshot_id": "frozen",
        "model": "test-model",
        "dimensions": EMBEDDING_DIMENSIONS,
        "profiles": {"me": _vector(0.1)},
        "jobs": {"1": _vector(0.2)},
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert load_embedding_artifact(artifact_path, snapshot)["model"] == "test-model"
    artifact["snapshot_id"] = "other"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="different snapshot"):
        load_embedding_artifact(artifact_path, snapshot)
