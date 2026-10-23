"""The two seams to model providers. Everything else depends on these interfaces, so you can
swap providers and use a fake in tests."""
from __future__ import annotations

import json
import math
import os
import time
from abc import ABC, abstractmethod
from typing import Callable

import httpx

from app.config import USER_AGENT
from app.fetchers.http import FetchError, post_json

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_EMBEDDINGS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
# Override with LLM_MODEL if your account uses a different model id.
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 1536


class LLMNotConfigured(RuntimeError):
    """No API key available."""


class LLMError(RuntimeError):
    """The provider call failed."""


class LLMClient(ABC):
    @abstractmethod
    def complete(self, *, system: str, user: str, max_tokens: int = 1500) -> str:
        """Return the model's text reply."""


class Embedder(ABC):
    """Provider-neutral embedding interface; dimensions must match VECTOR(n) in migrations."""

    dimensions: int

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        ...


class AnthropicLLM(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT})
        self._sleep = sleep

    def complete(self, *, system: str, user: str, max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        try:
            data = post_json(
                self.client,
                ANTHROPIC_URL,
                json=payload,
                headers={"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION},
                sleep=self._sleep,
            )
        except FetchError as exc:
            raise LLMError(f"Claude API request failed: {exc}") from exc
        blocks = data.get("content") if isinstance(data, dict) else None
        text = "".join(
            b.get("text", "") for b in (blocks or []) if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip():
            raise LLMError("Claude API returned no text")
        return text


class GeminiReranker:
    """Use Gemini structured output to order a supplied set of candidate job IDs."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.Client(timeout=90.0, headers={"User-Agent": USER_AGENT})
        self._sleep = sleep

    def rerank(self, *, system: str, user: str, job_ids: list[str]) -> list[str]:
        if not job_ids or len(set(job_ids)) != len(job_ids):
            raise LLMError("Gemini reranking requires a non-empty list of unique job IDs.")
        schema = {
            "type": "OBJECT",
            "properties": {
                "ranked_job_ids": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                }
            },
            "required": ["ranked_job_ids"],
        }
        try:
            data = post_json(
                self.client,
                f"{GEMINI_GENERATE_URL}/{self.model}:generateContent",
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": schema,
                        "temperature": 0,
                        "maxOutputTokens": max(256, len(job_ids) * 16),
                    },
                },
                headers={
                    "x-goog-api-key": self.api_key,
                    "Accept": "application/json",
                },
                sleep=self._sleep,
            )
        except FetchError as exc:
            raise LLMError(f"Gemini reranking request failed: {exc}") from exc

        candidates = data.get("candidates") if isinstance(data, dict) else None
        first_candidate = candidates[0] if isinstance(candidates, list) and candidates else None
        content = first_candidate.get("content") if isinstance(first_candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        response_text = "".join(
            part.get("text", "") for part in (parts or []) if isinstance(part, dict)
        )
        if not response_text.strip():
            raise LLMError("Gemini returned no reranking output.")
        try:
            result = json.loads(response_text)
        except ValueError as exc:
            raise LLMError("Gemini returned invalid JSON for the reranking result.") from exc
        ranked_ids = result.get("ranked_job_ids") if isinstance(result, dict) else None
        if (
            not isinstance(ranked_ids, list)
            or any(not isinstance(job_id, str) for job_id in ranked_ids)
            or len(ranked_ids) != len(job_ids)
            or set(ranked_ids) != set(job_ids)
        ):
            raise LLMError("Gemini reranking output must contain each supplied job ID exactly once.")
        return ranked_ids


class GeminiEmbedder(Embedder):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = EMBEDDING_DIMENSIONS
        self.client = client or httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT})
        self._sleep = sleep

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(value, str) or not value.strip() for value in texts):
            raise LLMError("Embedding inputs must be non-empty strings.")
        if task_type not in {"RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"}:
            raise LLMError(
                "Gemini embedding task_type must be RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT."
            )

        try:
            data = post_json(
                self.client,
                f"{GEMINI_EMBEDDINGS_URL}/{self.model}:batchEmbedContents",
                json={
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": value}]},
                            "taskType": task_type,
                            "outputDimensionality": self.dimensions,
                            "embedContentConfig": {
                                "taskType": task_type,
                                "outputDimensionality": self.dimensions,
                            },
                        }
                        for value in texts
                    ]
                },
                headers={
                    "x-goog-api-key": self.api_key,
                    "Accept": "application/json",
                },
                sleep=self._sleep,
            )
        except FetchError as exc:
            raise LLMError(f"Gemini embeddings request failed: {exc}") from exc

        items = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise LLMError("Gemini embeddings response did not contain one vector per input.")
        result: list[list[float]] = []
        for item in items:
            vector = item.get("values") if isinstance(item, dict) else None
            if not isinstance(vector, list):
                actual_dimensions = "missing"
                raise LLMError(
                    f"Embedding dimension mismatch: expected {self.dimensions} values, "
                    f"received {actual_dimensions}."
                )
            if len(vector) == 3072 and self.model in {
                "gemini-embedding-001",
                "gemini-embedding-2",
            }:
                vector = vector[: self.dimensions]
            if len(vector) != self.dimensions:
                actual_dimensions = len(vector) if isinstance(vector, list) else "missing"
                raise LLMError(
                    f"Embedding dimension mismatch: expected {self.dimensions} values, "
                    f"received {actual_dimensions}."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise LLMError("Gemini embeddings response contained a non-finite value.")
            normalized = [float(value) for value in vector]
            if self.model == "gemini-embedding-001":
                magnitude = math.sqrt(sum(value * value for value in normalized))
                if magnitude == 0:
                    raise LLMError("Gemini embeddings response contained a zero vector.")
                normalized = [value / magnitude for value in normalized]
            result.append(normalized)
        return result


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = EMBEDDING_DIMENSIONS
        self.client = client or httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT})
        self._sleep = sleep

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(value, str) or not value.strip() for value in texts):
            raise LLMError("Embedding inputs must be non-empty strings.")

        try:
            data = post_json(
                self.client,
                OPENAI_EMBEDDINGS_URL,
                json={
                    "model": self.model,
                    "input": texts,
                    "dimensions": self.dimensions,
                    "encoding_format": "float",
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                sleep=self._sleep,
            )
        except FetchError as exc:
            raise LLMError(f"OpenAI embeddings request failed: {exc}") from exc

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise LLMError("OpenAI embeddings response did not contain one vector per input.")

        embeddings: list[list[float] | None] = [None] * len(texts)
        for item in items:
            if not isinstance(item, dict):
                raise LLMError("OpenAI embeddings response contained an invalid item.")
            index = item.get("index")
            vector = item.get("embedding")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(texts):
                raise LLMError("OpenAI embeddings response contained an invalid input index.")
            if embeddings[index] is not None:
                raise LLMError("OpenAI embeddings response contained a duplicate input index.")
            if not isinstance(vector, list) or len(vector) != self.dimensions:
                raise LLMError(
                    f"Embedding dimension mismatch: expected {self.dimensions} values."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise LLMError("OpenAI embeddings response contained a non-finite value.")
            embeddings[index] = [float(value) for value in vector]

        if any(vector is None for vector in embeddings):
            raise LLMError("OpenAI embeddings response omitted an input.")
        return [vector for vector in embeddings if vector is not None]


def get_llm() -> LLMClient:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise LLMNotConfigured("Set ANTHROPIC_API_KEY (a personal key, not an employer's) to enable LLM extraction.")
    return AnthropicLLM(api_key=key, model=os.getenv("LLM_MODEL", DEFAULT_MODEL))


def get_embedder() -> GeminiEmbedder:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMNotConfigured("Set GEMINI_API_KEY to enable Gemini embeddings.")
    return GeminiEmbedder(
        api_key=key,
        model=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )


def get_gemini_reranker() -> GeminiReranker:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMNotConfigured("Set GEMINI_API_KEY to enable Gemini reranking.")
    return GeminiReranker(
        api_key=key,
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )
