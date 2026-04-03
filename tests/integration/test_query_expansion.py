"""
tests/integration/test_query_expansion.py — Happy-path: stop word handling and unicode safety.

Covers (via public API only):
- A stop-word-only keyword query returns 0 results gracefully (no crash).
- Adding stop words to a working query returns fewer or equal results (AND semantics).
- A unicode query does not raise an exception.

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
async def test_query_expansion(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / "2026-04-01.md").write_text(
        "PostgreSQL was chosen as the primary database for ACID compliance.\n"
        "Redis handles session caching and ephemeral data storage.\n"
    )
    (mem_dir / "frontend.md").write_text(
        "React 18 with Next.js provides the frontend framework.\n"
        "TypeScript is used throughout for type safety.\n"
    )
    (mem_dir / "infra.md").write_text(
        "AWS ECS Fargate eliminates cluster node management.\n"
        "Auto-scaling is configured at 70 percent CPU threshold.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        # All stop words → 0 results, no crash
        r_stops = await mem.search("the a is are", strategy="keyword", min_score=0.0)
        assert len(r_stops) == 0, (
            f"Stop-word-only query should return 0 results, got {len(r_stops)}"
        )

        # More terms (AND semantics) → fewer or equal results
        r_fewer = await mem.search("database ACID", strategy="keyword", min_score=0.01)
        r_more  = await mem.search("database ACID compliance storage", strategy="keyword", min_score=0.01)
        assert len(r_fewer) >= len(r_more), (
            "More required terms should not produce more results (AND semantics)"
        )

        # Unicode query → no exception raised
        r_unicode = await mem.search("données système 데이터베이스", min_score=0.0)
        assert isinstance(r_unicode, list), "Unicode query should return a list without crashing"
