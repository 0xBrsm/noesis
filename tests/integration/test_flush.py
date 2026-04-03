"""
tests/integration/test_flush.py — Happy-path: flush() LLM extraction, append, and searchability.

Covers:
- Substantive conversation: returns non-empty string and creates dated .md file.
- Second flush: appends — first content is still present in the file.
- Trivial conversation: returns None or str (no crash).
- Flushed content is searchable after flush() completes.

Requires: live embedding API (--embedding-model) + live LLM API (--llm-model).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, FlushConfig, MemWeave, MemoryConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_flush(workspace: Path, embedding_model: str, llm_model: str) -> None:
    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        flush=FlushConfig(enabled=True, model=llm_model),
    )

    today = date.today().isoformat()
    dated_file = workspace / "memory" / f"{today}.md"

    async with MemWeave(config) as mem:
        # First flush — substantive technical decision
        conv1 = [
            {"role": "user", "content": "We just decided to use Valkey instead of Redis for caching."},
            {"role": "assistant", "content": "Got it. I'll note that Valkey is the new caching layer."},
            {"role": "user", "content": "Also, we're targeting a 5ms p99 latency SLA for the cache."},
        ]
        result1 = await mem.flush(conv1)

        assert result1 is not None, "Expected facts extracted from substantive conversation"
        assert len(result1) > 0, "Expected non-empty extracted text"
        assert dated_file.exists(), f"Expected {dated_file} created after flush()"
        content_after_first = dated_file.read_text()

        # Second flush — different topic with enough substance for the LLM to extract facts;
        # must APPEND to the same dated file, not overwrite it.
        conv2 = [
            {"role": "user", "content": "We decided all backend services must emit OpenTelemetry spans. Jaeger was chosen as the tracing backend over Zipkin because of its better UI and sampling controls."},
            {"role": "assistant", "content": "Noted. OpenTelemetry with Jaeger is the mandatory distributed tracing stack."},
            {"role": "user", "content": "We also agreed on a 30-day retention policy for trace data to keep storage costs down."},
        ]
        result2 = await mem.flush(conv2)
        assert result2 is not None, (
            "Second flush with substantive content should extract facts"
        )

        content_after_second = dated_file.read_text()
        assert content_after_first.strip() in content_after_second, (
            "Second flush should append, not overwrite first flush content"
        )

        # Trivial conversation — should return None or str, no crash
        conv_trivial = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "ok"},
        ]
        result_trivial = await mem.flush(conv_trivial)
        assert result_trivial is None or isinstance(result_trivial, str), (
            "flush() must return None or str"
        )

        # Flushed content must be searchable
        results = await mem.search("Valkey caching latency", min_score=0.1)
        assert len(results) > 0, "Flushed content should be searchable after flush()"
