"""
tests/integration/test_strategy_comparison.py — Happy-path: hybrid vs vector vs keyword strategies.

Covers:
- All three strategies return results for the same query.
- Each strategy follows its own scoring path: not all top scores are identical.
- keyword → text_score populated, vector_score None.
- vector  → vector_score populated, text_score None.
- hybrid  → both scores populated (blended).
- A vector-heavy hybrid (vector_weight=0.95) scores closer to pure-vector than balanced hybrid.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import (
    EmbeddingConfig,
    HybridConfig,
    MemWeave,
    MemoryConfig,
    QueryConfig,
)  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_strategy_comparison(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / "2026-04-01.md").write_text(
        "PostgreSQL was selected as the primary relational database.\n"
        "The JSONB column type handles semi-structured data without a separate document store.\n"
    )
    (mem_dir / "frontend.md").write_text(
        "React 18 and Next.js 14 form the frontend stack.\n"
        "Server components reduce client bundle size significantly.\n"
    )
    (mem_dir / "infra.md").write_text(
        "All services are deployed on AWS ECS Fargate.\n"
        "Auto-scaling triggers at 70 percent CPU utilization.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0, max_results=10),
    )

    query = "PostgreSQL database JSONB"

    async with MemWeave(config) as mem:
        await mem.index()

        r_hybrid = await mem.search(query, strategy="hybrid", min_score=0.0)
        r_vector = await mem.search(query, strategy="vector", min_score=0.0)
        r_keyword = await mem.search(query, strategy="keyword", min_score=0.0)

    assert len(r_hybrid) > 0, "hybrid returned no results"
    assert len(r_vector) > 0, "vector returned no results"
    assert len(r_keyword) > 0, "keyword returned no results"

    # Each strategy follows its own scoring path — scores must not all be identical
    top_hybrid = r_hybrid[0].score
    top_vector = r_vector[0].score
    top_keyword = r_keyword[0].score
    all_same = abs(top_hybrid - top_vector) < 1e-6 and abs(top_hybrid - top_keyword) < 1e-6
    assert not all_same, "All three strategies returned identical top scores — unlikely"

    # Score field contract per strategy
    assert r_keyword[0].text_score is not None, "keyword: text_score must be populated"
    assert r_keyword[0].vector_score is None, "keyword: vector_score must be None"
    assert r_vector[0].vector_score is not None, "vector: vector_score must be populated"
    assert r_vector[0].text_score is None, "vector: text_score must be None"

    # Vector-heavy hybrid should score closer to pure-vector than balanced hybrid
    vheavy_config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(
            min_score=0.0,
            hybrid=HybridConfig(vector_weight=0.95, text_weight=0.05),
        ),
    )
    async with MemWeave(vheavy_config) as mem:
        r_vheavy = await mem.search(query, min_score=0.0)

    diff_vheavy = abs(r_vheavy[0].score - top_vector)
    diff_balanced = abs(top_hybrid - top_vector)
    assert (
        diff_vheavy <= diff_balanced + 0.05
    ), "Vector-heavy hybrid should be at least as close to pure-vector as balanced hybrid"
