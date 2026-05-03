"""
tests/integration/test_search_keyword.py — Happy-path: FTS5 keyword search strategy.

Covers:
- strategy="keyword" returns results; text_score is populated, vector_score is None.
- More query terms (AND semantics) returns fewer or equal results than fewer terms.
- A query consisting entirely of stop words returns 0 results gracefully (no crash).

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_search_keyword(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / "2026-04-01.md").write_text(
        "We chose PostgreSQL as the primary database for its JSONB capabilities.\n"
        "MongoDB was rejected because of the SSPL licensing concern raised by legal.\n"
    )
    (mem_dir / "frontend.md").write_text(
        "React 18 was selected as the frontend framework after evaluating Vue 3.\n"
        "The decision was driven by team expertise and Next.js App Router maturity.\n"
    )
    (mem_dir / "infra.md").write_text(
        "All services run on AWS ECS Fargate to eliminate node-level maintenance.\n"
        "Auto-scaling triggers at 70 percent CPU utilization.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        # Basic keyword match
        results = await mem.search("PostgreSQL JSONB", strategy="keyword", min_score=0.01)
        assert len(results) > 0, "Expected keyword match for 'PostgreSQL JSONB'"
        assert (
            results[0].text_score is not None and results[0].text_score > 0
        ), "text_score must be populated for keyword strategy"
        assert results[0].vector_score is None, "vector_score must be None for pure keyword search"

        # More required terms (AND semantics) → fewer or equal results
        r_fewer = await mem.search("database SSPL", strategy="keyword", min_score=0.01)
        r_more = await mem.search(
            "database SSPL licensing concern", strategy="keyword", min_score=0.01
        )
        assert len(r_fewer) >= len(
            r_more
        ), "More query terms should not produce more results than fewer terms (AND semantics)"

        # All stop words → graceful empty result, no crash
        r_stops = await mem.search("the a is are", strategy="keyword", min_score=0.0)
        assert (
            len(r_stops) == 0
        ), f"Stop-word-only query should return 0 results, got {len(r_stops)}"
