"""
tests/integration/test_snippet_truncation.py — Happy-path: snippet_max_chars is a hard cap.

Covers:
- snippet_max_chars=100: all returned snippets are ≤ 100 chars, none empty.
- snippet_max_chars=2000: snippets are longer than with the 100-char cap.
- Default snippet_max_chars=700: all snippets ≤ 700 chars.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig, QueryConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_snippet_truncation(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    long_text = "\n".join(
        f"Line {i:03d}: The system uses event-driven architecture with Kafka for async messaging "
        f"between microservices. Each service publishes domain events on state changes."
        for i in range(1, 30)
    )
    (mem_dir / "2026-04-01.md").write_text(long_text)

    query = "event-driven architecture Kafka microservices"

    def _config(snippet_max_chars: int) -> MemoryConfig:
        return MemoryConfig(
            workspace_dir=workspace,
            embedding=EmbeddingConfig(model=embedding_model),
            query=QueryConfig(min_score=0.0, max_results=5, snippet_max_chars=snippet_max_chars),
        )

    # Index once; reuse the same workspace for all three searches
    async with MemWeave(_config(100)) as mem:
        await mem.index()
        r100 = await mem.search(query, min_score=0.0)

    async with MemWeave(_config(2000)) as mem:
        r2000 = await mem.search(query, min_score=0.0)

    async with MemWeave(_config(700)) as mem:
        r_default = await mem.search(query, min_score=0.0)

    assert len(r100) > 0, "Expected results with snippet_max_chars=100"
    assert all(len(r.snippet) <= 100 for r in r100), (
        f"Snippet exceeds 100 chars: {[len(r.snippet) for r in r100 if len(r.snippet) > 100]}"
    )
    assert all(r.snippet for r in r100), "Empty snippet found"

    assert len(r2000) > 0, "Expected results with snippet_max_chars=2000"
    assert all(len(r.snippet) <= 2000 for r in r2000), "Snippet exceeds 2000 chars"
    assert max(len(r.snippet) for r in r2000) >= max(len(r.snippet) for r in r100), (
        "Larger snippet_max_chars should allow longer snippets"
    )

    assert all(len(r.snippet) <= 700 for r in r_default), (
        "Default 700-char cap violated"
    )
