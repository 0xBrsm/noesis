"""
tests/unit/test_provider.py — Unit tests for memweave/embedding/provider.py

Tests use mocking to avoid real LiteLLM API calls. The tests verify:
- Protocol conformance (duck typing)
- Batch splitting logic
- Retry behavior on transient errors
- Fast failure on non-retryable errors
- L2 normalization of returned vectors
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memweave.config import EmbeddingConfig
from memweave.embedding.provider import (
    EmbeddingProvider,
    LiteLLMEmbeddingProvider,
    _get_status_code,
)
from memweave.exceptions import EmbeddingError


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def make_provider(
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> LiteLLMEmbeddingProvider:
    """Helper: create a provider with the given config."""
    config = EmbeddingConfig(
        model=model,
        batch_size=batch_size,
    )
    return LiteLLMEmbeddingProvider(config)


def make_litellm_response(vectors: list[list[float]]) -> MagicMock:
    """Helper: build a fake LiteLLM aembedding response."""
    response = MagicMock()
    response.data = [{"embedding": v} for v in vectors]
    return response


# ── Protocol conformance ──────────────────────────────────────────────────────

class TestEmbeddingProviderProtocol:
    def test_litellm_provider_conforms(self):
        """LiteLLMEmbeddingProvider should satisfy the EmbeddingProvider protocol."""
        provider = make_provider()
        assert isinstance(provider, EmbeddingProvider)

    def test_duck_type_conforms(self):
        """Any object with embed_query and embed_batch methods conforms."""
        class MyProvider:
            async def embed_query(self, text: str) -> list[float]:
                return [1.0]
            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[1.0]] * len(texts)

        assert isinstance(MyProvider(), EmbeddingProvider)

    def test_incomplete_duck_type_does_not_conform(self):
        """Object with only one method should NOT conform."""
        class Incomplete:
            async def embed_query(self, text: str) -> list[float]:
                return [1.0]
            # Missing embed_batch

        assert not isinstance(Incomplete(), EmbeddingProvider)


# ── embed_query ───────────────────────────────────────────────────────────────

class TestEmbedQuery:
    async def test_empty_text_raises(self):
        """Empty text should raise EmbeddingError before calling API."""
        provider = make_provider()
        with pytest.raises(EmbeddingError, match="empty text"):
            await provider.embed_query("")

    async def test_whitespace_only_raises(self):
        provider = make_provider()
        with pytest.raises(EmbeddingError, match="empty text"):
            await provider.embed_query("   ")

    async def test_returns_normalized_vector(self):
        """embed_query should return an L2-normalized vector."""
        raw_vec = [3.0, 4.0]  # norm = 5 → normalized = [0.6, 0.8]
        response = make_litellm_response([raw_vec])

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(return_value=response)):
            result = await provider.embed_query("test query")

        assert abs(_l2_norm(result) - 1.0) < 1e-6

    async def test_returns_list_of_floats(self):
        raw_vec = [0.1, 0.2, 0.3]
        response = make_litellm_response([raw_vec])

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(return_value=response)):
            result = await provider.embed_query("some text")

        assert isinstance(result, list)
        for v in result:
            assert isinstance(v, float)


# ── embed_batch ───────────────────────────────────────────────────────────────

class TestEmbedBatch:
    async def test_empty_list_returns_empty(self):
        provider = make_provider()
        result = await provider.embed_batch([])
        assert result == []

    async def test_single_text(self):
        raw_vec = [1.0, 0.0]
        response = make_litellm_response([raw_vec])

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(return_value=response)):
            result = await provider.embed_batch(["hello"])

        assert len(result) == 1
        assert abs(_l2_norm(result[0]) - 1.0) < 1e-6

    async def test_batch_splitting(self):
        """Texts exceeding batch_size should cause multiple API calls."""
        # batch_size=2, 5 texts → 3 API calls (2, 2, 1)
        provider = make_provider(batch_size=2)
        calls = []

        async def fake_embed(**kwargs):
            calls.append(kwargs["input"])
            n = len(kwargs["input"])
            return make_litellm_response([[float(i), 0.0] for i in range(n)])

        with patch("litellm.aembedding", new=AsyncMock(side_effect=fake_embed)):
            result = await provider.embed_batch(["t1", "t2", "t3", "t4", "t5"])

        assert len(calls) == 3        # 3 batches
        assert len(calls[0]) == 2     # first batch: 2 items
        assert len(calls[1]) == 2     # second batch: 2 items
        assert len(calls[2]) == 1     # third batch: 1 item
        assert len(result) == 5       # 5 total results

    async def test_result_order_preserved(self):
        """Results must be in the same order as the input texts."""
        # Use non-zero vectors so normalization doesn't collapse them
        vecs = [[float(i + 1), 0.0] for i in range(5)]  # [1,0], [2,0], ... all non-zero
        response = make_litellm_response(vecs)

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(return_value=response)):
            result = await provider.embed_batch([f"text{i}" for i in range(5)])

        assert len(result) == 5
        # After normalization of [x, 0.0], first element should be 1.0 for all
        for vec in result:
            assert abs(vec[0] - 1.0) < 1e-6
            assert abs(vec[1]) < 1e-6

    async def test_all_normalized(self):
        """All returned vectors should be unit-normalized."""
        import random
        rng = random.Random(42)
        raw_vecs = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(10)]
        response = make_litellm_response(raw_vecs)

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(return_value=response)):
            result = await provider.embed_batch([f"text{i}" for i in range(10)])

        for vec in result:
            assert abs(_l2_norm(vec) - 1.0) < 1e-6


# ── Retry behavior ────────────────────────────────────────────────────────────

class TestRetryBehavior:
    async def test_retries_on_rate_limit(self):
        """A 429 error should trigger retries."""
        rate_limit_exc = Exception("Rate limit exceeded")
        rate_limit_exc.status_code = 429

        success_response = make_litellm_response([[1.0, 0.0]])
        call_count = 0

        async def fake_embed(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise rate_limit_exc
            return success_response

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(side_effect=fake_embed)):
            with patch("asyncio.sleep", new=AsyncMock()):  # skip actual delays
                result = await provider.embed_batch(["text"])

        assert call_count == 3
        assert len(result) == 1

    async def test_fails_fast_on_auth_error(self):
        """A 401 error should raise immediately without retrying."""
        auth_exc = Exception("Unauthorized")
        auth_exc.status_code = 401

        call_count = 0

        async def fake_embed(**kwargs):
            nonlocal call_count
            call_count += 1
            raise auth_exc

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(side_effect=fake_embed)):
            with pytest.raises(EmbeddingError, match="non-retryable"):
                await provider.embed_batch(["text"])

        assert call_count == 1  # Only one attempt before failing fast

    async def test_raises_after_max_retries(self):
        """Should raise EmbeddingError after exhausting all retry attempts."""
        server_exc = Exception("Internal server error")
        server_exc.status_code = 500

        provider = make_provider()
        with patch("litellm.aembedding", new=AsyncMock(side_effect=server_exc)):
            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(EmbeddingError, match="3 attempts"):
                    await provider.embed_batch(["text"])


# ── Provider properties ───────────────────────────────────────────────────────

class TestProviderProperties:
    def test_model_property(self):
        provider = make_provider(model="text-embedding-3-large")
        assert provider.model == "text-embedding-3-large"

    def test_model_is_str(self):
        provider = make_provider()
        assert isinstance(provider.model, str)


# ── _get_status_code ──────────────────────────────────────────────────────────

class TestGetStatusCode:
    def test_exception_with_status_code(self):
        exc = Exception("test")
        exc.status_code = 429
        assert _get_status_code(exc) == 429

    def test_exception_without_status_code(self):
        assert _get_status_code(ValueError("no status")) is None

    def test_none_status_code(self):
        exc = Exception()
        exc.status_code = None
        assert _get_status_code(exc) is None
