"""
tests/integration/test_mmr_diversity.py — Happy-path: MMR lambda controls relevance vs. diversity.

Covers per-call mmr_lambda kwarg:
- mmr_lambda=1.0 (pure relevance): result set clusters around the query topic.
- mmr_lambda=0.0 (pure diversity): result set spreads across different topics.
- db_count_diversity <= db_count_relevance (diversity picks fewer near-duplicates).
- The two result sets differ.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig, MMRConfig, QueryConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_mmr_diversity(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"

    # 3 near-duplicate database chunks (same topic, slight variation)
    (mem_dir / "db1.md").write_text(
        "PostgreSQL is the primary relational database for transactional data.\n"
        "Connection pooling via PgBouncer reduces overhead on the database server.\n"
    )
    (mem_dir / "db2.md").write_text(
        "PostgreSQL handles all OLTP workloads in the production environment.\n"
        "Read replicas are used for analytics queries to offload the primary.\n"
    )
    (mem_dir / "db3.md").write_text(
        "PostgreSQL schema migrations are managed with Alembic and run on deploy.\n"
        "All migration scripts are reversible and tested in staging before production.\n"
    )
    # 2 chunks from completely different topics
    (mem_dir / "frontend.md").write_text(
        "React 18 with Next.js App Router is the frontend framework.\n"
        "Server components handle data fetching without client-side hydration.\n"
    )
    (mem_dir / "infra.md").write_text(
        "AWS ECS Fargate runs all backend services without node-level management.\n"
        "Auto-scaling is configured at 70 percent CPU utilization threshold.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(
            min_score=0.0,
            max_results=4,
            mmr=MMRConfig(enabled=True, lambda_param=0.7),
        ),
    )

    query = "PostgreSQL database production"

    async with MemWeave(config) as mem:
        await mem.index()

        r_relevance = await mem.search(query, min_score=0.0, max_results=4, mmr_lambda=1.0)
        r_diversity = await mem.search(query, min_score=0.0, max_results=4, mmr_lambda=0.0)

    assert len(r_relevance) > 0, "Expected results with mmr_lambda=1.0"
    assert len(r_diversity) > 0, "Expected results with mmr_lambda=0.0"

    db_count_relevance = sum(1 for r in r_relevance if "db" in r.path)
    db_count_diversity = sum(1 for r in r_diversity if "db" in r.path)

    assert db_count_diversity <= db_count_relevance, (
        f"Diversity mode should select fewer near-duplicate db results: "
        f"diversity={db_count_diversity} relevance={db_count_relevance}"
    )

    paths_relevance = {r.path for r in r_relevance}
    paths_diversity = {r.path for r in r_diversity}
    assert paths_relevance != paths_diversity, (
        "mmr_lambda=0.0 and mmr_lambda=1.0 should produce different result sets"
    )
