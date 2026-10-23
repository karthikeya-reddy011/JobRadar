import json

import httpx
import pytest

from app.llm import EMBEDDING_DIMENSIONS, GeminiEmbedder, LLMError, get_embedder


def _vector(value):
    return [value] * EMBEDDING_DIMENSIONS


def test_gemini_embedder_batches_inputs_with_requested_dimension_and_task():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["accept"] = request.headers["Accept"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": _vector(0.1)},
                    {"values": _vector(0.2)},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = GeminiEmbedder("test-key", client=client, sleep=lambda _: None)
        vectors = embedder.embed(
            ["profile", "another profile"],
            task_type="RETRIEVAL_QUERY",
        )

    requests = seen["payload"]["requests"]
    assert seen["url"].endswith("/models/gemini-embedding-001:batchEmbedContents")
    assert seen["key"] == "test-key"
    assert seen["accept"] == "application/json"
    assert len(requests) == 2
    assert requests[0]["model"] == "models/gemini-embedding-001"
    assert requests[0]["content"]["parts"][0]["text"] == "profile"
    assert requests[0]["taskType"] == "RETRIEVAL_QUERY"
    assert requests[0]["outputDimensionality"] == EMBEDDING_DIMENSIONS
    assert requests[0]["embedContentConfig"]["taskType"] == "RETRIEVAL_QUERY"
    assert (
        requests[0]["embedContentConfig"]["outputDimensionality"]
        == EMBEDDING_DIMENSIONS
    )
    assert len(vectors) == 2
    assert vectors[0][0] == pytest.approx(1 / EMBEDDING_DIMENSIONS**0.5)
    assert vectors[1][0] == pytest.approx(1 / EMBEDDING_DIMENSIONS**0.5)


def test_gemini_embedder_validates_document_task_and_response_vectors():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["requests"][0]["embedContentConfig"]["taskType"] == "RETRIEVAL_DOCUMENT"
        return httpx.Response(200, json={"embeddings": [{"values": [0.1]}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = GeminiEmbedder("test-key", client=client, sleep=lambda _: None)
        with pytest.raises(LLMError, match="dimension mismatch"):
            embedder.embed(["job text"], task_type="RETRIEVAL_DOCUMENT")


def test_gemini_embedder_reports_actual_response_dimension():
    def handler(_request):
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = GeminiEmbedder("test-key", client=client, sleep=lambda _: None)
        with pytest.raises(
            LLMError,
            match=rf"expected {EMBEDDING_DIMENSIONS} values, received 3",
        ):
            embedder.embed(["job text"], task_type="RETRIEVAL_DOCUMENT")


def test_gemini_embedding_001_truncates_default_3072_vector_and_normalizes():
    def handler(_request):
        vector = [1.0] * 3072
        vector[-1] = 50.0
        return httpx.Response(200, json={"embeddings": [{"values": vector}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = GeminiEmbedder("test-key", client=client, sleep=lambda _: None)
        (vector,) = embedder.embed(
            ["text"],
            task_type="RETRIEVAL_DOCUMENT",
        )

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_gemini_embedder_rejects_invalid_inputs_and_task_type():
    embedder = GeminiEmbedder("test-key")
    assert embedder.embed([], task_type="RETRIEVAL_DOCUMENT") == []

    with pytest.raises(LLMError, match="non-empty"):
        embedder.embed([" "], task_type="RETRIEVAL_DOCUMENT")
    with pytest.raises(LLMError, match="task_type"):
        embedder.embed(["text"], task_type="SEMANTIC_SIMILARITY")


def test_gemini_embedder_rejects_non_finite_values():
    def handler(_request):
        vector = _vector(0.1)
        vector[0] = float("nan")
        return httpx.Response(
            200,
            content=json.dumps({"embeddings": [{"values": vector}]}).encode(),
            headers={"content-type": "application/json"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = GeminiEmbedder("test-key", client=client, sleep=lambda _: None)
        with pytest.raises(LLMError, match="non-finite"):
            embedder.embed(["text"], task_type="RETRIEVAL_QUERY")


def test_configured_embedder_uses_gemini_environment_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)

    embedder = get_embedder()

    assert isinstance(embedder, GeminiEmbedder)
    assert embedder.model == "gemini-embedding-001"
    assert embedder.dimensions == 1536
