"""
tests/integration/test_max_results.py — Happy-path: max_results hard cap.

Covers:
- max_results=1 returns exactly 1 result and it is the global top scorer.
- max_results=3 returns ≤ 3 results matching the true top-3 scores.
- max_results=50 returns ≤ 50 results.
- Results are always sorted descending by score.

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
async def test_max_results(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    for i in range(5):
        (mem_dir / f"note_{i:02d}.md").write_text(
            f"Architecture decision note {i}.\n"
            f"We chose approach {i} for the service mesh configuration.\n"
            f"This reduces latency by {(i + 1) * 5} percent in production.\n"
        )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0, max_results=50),
    )

    query = "architecture service mesh configuration latency"

    async with MemWeave(config) as mem:
        await mem.index()

        r_all   = await mem.search(query, min_score=0.0, max_results=50)
        r_one   = await mem.search(query, min_score=0.0, max_results=1)
        r_three = await mem.search(query, min_score=0.0, max_results=3)

    assert len(r_all) <= 50, f"Returned more than 50 results: {len(r_all)}"

    assert len(r_one) == 1, f"max_results=1 returned {len(r_one)} results"
    if r_all:
        assert abs(r_one[0].score - r_all[0].score) < 1e-9, (
            f"max_results=1 result score {r_one[0].score:.6f} "
            f"!= global top {r_all[0].score:.6f}"
        )

    assert len(r_three) <= 3, f"max_results=3 returned {len(r_three)} results"
    if len(r_all) >= 3:
        top3_expected = sorted([r.score for r in r_all], reverse=True)[:3]
        top3_actual   = sorted([r.score for r in r_three], reverse=True)
        assert all(
            abs(a - e) < 1e-9 for a, e in zip(top3_actual, top3_expected)
        ), f"max_results=3 not returning true top-3: {top3_actual} vs {top3_expected}"

    scores = [r.score for r in r_all]
    assert scores == sorted(scores, reverse=True), "Results not sorted descending by score"
