import json
import sys
from pathlib import Path

import httpx
import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from app.llm import GeminiReranker, LLMError
from run_eval import (
    _evaluate_gemini_reranker,
    load_gemini_rerank_artifact,
)
import rerank_gemini


def test_gemini_reranker_uses_structured_output_and_validates_ids():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["x-goog-api-key"]
        seen["accept"] = request.headers["Accept"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"ranked_job_ids":["2","1"]}'}]
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reranker = GeminiReranker(
            "test-key",
            client=client,
            sleep=lambda _: None,
        )
        ranked_ids = reranker.rerank(
            system="Rank by role relevance.",
            user="Profile and jobs.",
            job_ids=["1", "2"],
        )

    assert ranked_ids == ["2", "1"]
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert seen["authorization"] == "test-key"
    assert seen["accept"] == "application/json"
    assert seen["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["payload"]["generationConfig"]["responseSchema"]["required"] == [
        "ranked_job_ids"
    ]


@pytest.mark.parametrize(
    "response_text",
    [
        '{"ranked_job_ids":["1","1"]}',
        '{"ranked_job_ids":["1"]}',
        '{"ranked_job_ids":["1","3"]}',
    ],
)
def test_gemini_reranker_rejects_incomplete_or_invalid_job_ids(response_text):
    def handler(_request):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": response_text}]}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reranker = GeminiReranker("test-key", client=client, sleep=lambda _: None)
        with pytest.raises(LLMError, match="each supplied job ID exactly once"):
            reranker.rerank(system="system", user="prompt", job_ids=["1", "2"])


def test_gemini_reranker_rejects_invalid_json():
    def handler(_request):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reranker = GeminiReranker("test-key", client=client, sleep=lambda _: None)
        with pytest.raises(LLMError, match="invalid JSON"):
            reranker.rerank(system="system", user="prompt", job_ids=["1"])


def test_rerank_artifact_is_snapshot_bound_and_metrics_cover_full_pool(tmp_path):
    snapshot = {
        "snapshot_id": "frozen",
        "profiles": {"me": {"skills": [], "titles": []}},
        "jobs": {str(job_id): {"title": "", "description": ""} for job_id in range(1, 6)},
        "candidate_pool": {"me": ["1", "2", "3", "4", "5"]},
    }
    artifact = {
        "version": 1,
        "snapshot_id": "frozen",
        "model": "gemini-2.5-flash",
        "candidate_source": "keyword_score",
        "shortlist_sizes": {"me": 2},
        "rankings": {"me": ["5", "1"]},
    }
    artifact_path = tmp_path / "rerank.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_gemini_rerank_artifact(artifact_path, snapshot)
    labels = {
        ("me", "1"): 0,
        ("me", "2"): 0,
        ("me", "3"): 0,
        ("me", "4"): 0,
        ("me", "5"): 2,
    }
    metrics = _evaluate_gemini_reranker(snapshot, labels, loaded)

    assert metrics["profiles"]["me"]["nDCG@10"] == 1.0
    artifact["snapshot_id"] = "different"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="different snapshot"):
        load_gemini_rerank_artifact(artifact_path, snapshot)


def test_rerank_artifact_rejects_duplicate_or_non_candidate_ids(tmp_path):
    snapshot = {
        "snapshot_id": "frozen",
        "candidate_pool": {"me": ["1", "2"]},
    }
    artifact = {
        "version": 1,
        "snapshot_id": "frozen",
        "model": "gemini-2.5-flash",
        "candidate_source": "keyword_score",
        "shortlist_sizes": {"me": 2},
        "rankings": {"me": ["1", "1"]},
    }
    artifact_path = tmp_path / "rerank.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid ranking"):
        load_gemini_rerank_artifact(artifact_path, snapshot)


def test_generation_uses_role_only_shortlist_and_writes_snapshot_artifact(
    tmp_path, monkeypatch
):
    seen = {}

    class FakeReranker:
        model = "fake-gemini"

        def rerank(self, *, system, user, job_ids):
            seen["system"] = system
            seen["user"] = user
            seen["job_ids"] = job_ids
            return list(reversed(job_ids))

    monkeypatch.setattr(rerank_gemini, "get_gemini_reranker", lambda: FakeReranker())
    snapshot = {
        "snapshot_id": "frozen",
        "profiles": {
            "me": {
                "skills": ["Python"],
                "titles": ["Backend Engineer"],
                "locations": ["Seattle"],
                "remote_ok": True,
            }
        },
        "jobs": {
            "1": {"title": "Backend Engineer", "description": "Build APIs", "location": "NYC"},
            "2": {"title": "Developer", "description": "Python services", "is_remote": True},
            "3": {"title": "Analyst", "description": "Data analysis"},
        },
        "candidate_pool": {"me": ["1", "2", "3"]},
    }
    output_path = tmp_path / "gemini-rerank.json"

    artifact = rerank_gemini.generate_gemini_reranking(
        snapshot,
        output_path=output_path,
        top_k=2,
    )

    assert seen["job_ids"] == ["1", "2"]
    assert "Python" in seen["user"]
    assert "Seattle" not in seen["user"]
    assert "NYC" not in seen["user"]
    assert "remote" in seen["system"].lower()
    assert artifact["rankings"] == {"me": ["2", "1"]}
    assert artifact["snapshot_id"] == "frozen"
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
