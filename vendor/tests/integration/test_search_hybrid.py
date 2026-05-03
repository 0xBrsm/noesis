"""
tests/integration/test_search_hybrid.py — Happy-path: hybrid search correctness.

Covers the most important user-facing API:
- Results are returned with score > 0.
- SearchResult fields are all populated.
- Results are sorted descending by score.
- max_results caps the result count.

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
async def test_search_hybrid(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / "2026-04-01.md").write_text(
        "We chose PostgreSQL 16 for JSONB and mature replication. MongoDB was rejected (SSPL license).\n"
        "Redis was chosen as the caching layer; pool size set to 20 after load testing.\n"
        "All services deployed to AWS ECS Fargate; Kubernetes was too operationally heavy.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        await mem.index()
        results = await mem.search(
            "which database did we choose and why?", max_results=3, min_score=0.1
        )

    assert len(results) > 0, "Expected at least one result"
    assert len(results) <= 3, f"max_results=3 violated: got {len(results)}"

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), f"Results not sorted descending: {scores}"

    for r in results:
        assert r.path, "path must be non-empty"
        assert r.snippet, "snippet must be non-empty"
        assert r.source, "source must be non-empty"
        assert r.start_line >= 1, f"start_line must be ≥ 1, got {r.start_line}"
        assert r.end_line >= r.start_line, "end_line must be ≥ start_line"
        assert 0.0 <= r.score <= 1.0, f"score out of range: {r.score}"

    assert results[0].score > 0.3, f"Top result score too low: {results[0].score}"
