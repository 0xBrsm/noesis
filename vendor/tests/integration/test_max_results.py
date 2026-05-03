"""
tests/integration/test_max_results.py — Happy-path: max_results hard cap.

Covers:
- max_results=1 returns exactly 1 result and it is the same document as the global top scorer.
- max_results=3 returns ≤ 3 results.
- max_results=50 returns ≤ 50 results.
- Results are always sorted descending by score.

Note on score comparisons across different max_results calls:
  The hybrid search fetches candidate_limit = max_results × candidate_multiplier
  candidates from each backend before merging. A chunk that appears in both the
  vector and keyword backends receives a full combined score (0.7 × vs + 0.3 × ts).
  A chunk that only makes it into one backend (because the other backend's pool was
  too small) gets 0.0 for the missing component — a slightly lower combined score.
  This means scores for the same document can legitimately differ across calls with
  different max_results values. Exact score equality across calls is therefore not
  a contract the library makes.

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

        r_all = await mem.search(query, min_score=0.0, max_results=50)
        r_one = await mem.search(query, min_score=0.0, max_results=1)
        r_three = await mem.search(query, min_score=0.0, max_results=3)

    # ── Length caps ───────────────────────────────────────────────────────────
    assert len(r_all) <= 50, f"Returned more than 50 results: {len(r_all)}"
    assert len(r_one) == 1, f"max_results=1 returned {len(r_one)} results"
    assert len(r_three) <= 3, f"max_results=3 returned {len(r_three)} results"

    # ── Sort order — all result sets must be sorted descending ────────────────
    for label, results in [("r_all", r_all), ("r_one", r_one), ("r_three", r_three)]:
        scores = [r.score for r in results]
        assert scores == sorted(
            scores, reverse=True
        ), f"{label} results not sorted descending by score: {scores}"

    # ── Top document identity ─────────────────────────────────────────────────
    # max_results=1 must return the same document as the global top scorer.
    # Scores may differ slightly (see module docstring) so we compare by path,
    # not by score value.
    if r_all and r_one:
        assert r_one[0].path == r_all[0].path, (
            f"max_results=1 top result ({r_one[0].path}) differs from "
            f"max_results=50 top result ({r_all[0].path})"
        )
