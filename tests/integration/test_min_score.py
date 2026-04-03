"""
tests/integration/test_min_score.py — Happy-path: min_score threshold filtering.

Covers:
- min_score=0.0 returns more results than min_score=0.5 (lower threshold = more results).
- min_score=1.0 returns 0 results (no chunk can score exactly 1.0).
- Every returned result has score >= the requested threshold.
- Per-call override is independent of config default.

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
async def test_min_score(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / "2026-04-01.md").write_text(
        "The deployment pipeline uses GitHub Actions for CI and ArgoCD for CD.\n"
        "Rollbacks are triggered automatically when error rate exceeds 5 percent.\n"
        "Canary deployments use a 10-percent traffic split for the first 15 minutes.\n"
    )
    (mem_dir / "infra.md").write_text(
        "Kubernetes cluster runs on EKS with node groups split by workload type.\n"
        "PodDisruptionBudgets ensure at least one replica during maintenance windows.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0),
    )

    query = "deployment pipeline rollback strategy"

    async with MemWeave(config) as mem:
        await mem.index()

        r_low       = await mem.search(query, min_score=0.0)
        r_high      = await mem.search(query, min_score=0.5)
        r_impossible = await mem.search(query, min_score=1.0)
        r_override  = await mem.search(query, min_score=0.7)

    assert len(r_low) >= len(r_high), (
        f"Higher min_score should return ≤ results: low={len(r_low)} high={len(r_high)}"
    )
    assert len(r_impossible) == 0, (
        f"min_score=1.0 should return 0 results, got {len(r_impossible)}"
    )
    assert all(r.score >= 0.5 for r in r_high), (
        f"Result below min_score=0.5 threshold: {[r.score for r in r_high if r.score < 0.5]}"
    )
    assert all(r.score >= 0.7 for r in r_override), (
        f"Per-call min_score=0.7 override not respected: {[r.score for r in r_override if r.score < 0.7]}"
    )
