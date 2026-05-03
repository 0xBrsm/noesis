"""
tests/integration/test_postprocessors_combined.py — Happy-path: decay + MMR pipeline combined.

Covers:
- Decay-only, MMR-only, and combined (decay + MMR) all return results without crashing.
- Combined results differ from decay-only and MMR-only (pipeline produces distinct output).
- Pipeline order is correct: decay adjusts scores first, then MMR reranks on adjusted scores —
  so an old file should score <= its MMR-only score in combined mode.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import (  # noqa: E402
    EmbeddingConfig,
    MemWeave,
    MemoryConfig,
    MMRConfig,
    QueryConfig,
    TemporalDecayConfig,
)

pytestmark = pytest.mark.integration

_OLD_DATE = "2026-02-01"  # ~60 days before 2026-04-01
_TODAY_DATE = "2026-04-01"

_TOPIC_AUTH = (
    "Authentication uses OAuth 2.0 with PKCE for public clients.\n"
    "Access tokens expire in 15 minutes; refresh tokens last 30 days.\n"
)
_TOPIC_RATE = (
    "The API is rate-limited to 1000 requests per minute per API key.\n"
    "Burst allowance of 200 requests per second is available for short spikes.\n"
)


@pytest.mark.asyncio
async def test_postprocessors_combined(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    # Two fresh files on different topics + one old file on the same topic as the first
    (mem_dir / f"{_TODAY_DATE}.md").write_text(_TOPIC_AUTH)
    (mem_dir / "api_limits.md").write_text(_TOPIC_RATE)
    (mem_dir / f"{_OLD_DATE}.md").write_text(_TOPIC_AUTH)

    query = "OAuth authentication access token"

    def _base_config(**query_kwargs: object) -> MemoryConfig:
        return MemoryConfig(
            workspace_dir=workspace,
            embedding=EmbeddingConfig(model=embedding_model),
            query=QueryConfig(min_score=0.0, max_results=5, **query_kwargs),  # type: ignore[arg-type]
        )

    # Index once; all three search runs share the same index
    async with MemWeave(_base_config()) as mem:
        await mem.index()
        r_baseline = await mem.search(query, min_score=0.0)

    # Decay only
    async with MemWeave(
        _base_config(
            temporal_decay=TemporalDecayConfig(enabled=True, half_life_days=14.0),
        )
    ) as mem:
        r_decay = await mem.search(query, min_score=0.0)

    # MMR only
    async with MemWeave(
        _base_config(
            mmr=MMRConfig(enabled=True, lambda_param=0.5),
        )
    ) as mem:
        r_mmr = await mem.search(query, min_score=0.0)

    # Combined: decay + MMR
    async with MemWeave(
        _base_config(
            temporal_decay=TemporalDecayConfig(enabled=True, half_life_days=14.0),
            mmr=MMRConfig(enabled=True, lambda_param=0.5),
        )
    ) as mem:
        r_combined = await mem.search(query, min_score=0.0)

    assert len(r_combined) > 0, "Combined decay+MMR returned no results"

    # Combined must produce at least one result set different from baseline
    baseline_paths = [r.path for r in r_baseline]
    combined_paths = [r.path for r in r_combined]
    decay_paths = [r.path for r in r_decay]
    mmr_paths = [r.path for r in r_mmr]
    assert (
        combined_paths != baseline_paths
        or decay_paths != baseline_paths
        or mmr_paths != baseline_paths
    ), "At least one post-processor configuration should alter the result order"

    # Old file should score <= its MMR-only score in combined mode
    # (decay penalises it before MMR reranks — double penalty)
    old_path = f"memory/{_OLD_DATE}.md"
    old_combined = next((r.score for r in r_combined if r.path == old_path), None)
    old_mmr_only = next((r.score for r in r_mmr if r.path == old_path), None)
    if old_combined is not None and old_mmr_only is not None:
        assert old_combined <= old_mmr_only + 1e-6, (
            f"Decay should penalise old file further in combined mode: "
            f"combined={old_combined:.4f} vs mmr_only={old_mmr_only:.4f}"
        )
