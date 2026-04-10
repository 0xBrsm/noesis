"""
tests/integration/test_cache_lru.py — Happy-path: LRU cap prevents unbounded cache growth.

Covers:
- CacheConfig(max_entries=3): after indexing 6 files, cache_entries <= 3.
- status().cache_max_entries reflects the configured cap.
- force re-index after LRU eviction: at least the evicted chunks are recomputed
  (embeddings_computed >= evicted_count).
- CacheConfig(enabled=False): embeddings_cached == 0 on both first and force re-index.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import CacheConfig, EmbeddingConfig, MemWeave, MemoryConfig  # noqa: E402

pytestmark = pytest.mark.integration


def _write_distinct_files(mem_dir: Path, n: int) -> None:
    """Write n files with distinct content so each produces a distinct cache entry."""
    for i in range(n):
        (mem_dir / f"note_{i:02d}.md").write_text(
            f"Decision note {i}: We evaluated option {i} for the infrastructure choice.\n"
            f"Option {i} was preferred because it reduces operational complexity by {i * 10} percent.\n"
        )


@pytest.mark.asyncio
async def test_cache_lru_cap(workspace: Path, embedding_model: str) -> None:
    _write_distinct_files(workspace / "memory", n=6)

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        cache=CacheConfig(enabled=True, max_entries=3),
    )

    async with MemWeave(config) as mem:
        r1 = await mem.index()
        s1 = await mem.status()

        assert r1.files_indexed == 6, f"Expected 6 files indexed, got {r1.files_indexed}"
        assert s1.cache_entries <= 3, f"Cache should be capped at 3 entries, got {s1.cache_entries}"
        assert (
            s1.cache_max_entries == 3
        ), f"status().cache_max_entries should be 3, got {s1.cache_max_entries}"

        # Force re-index: 6 files, only 3 still in cache → at least 3 must be recomputed
        r2 = await mem.index(force=True)
        assert (
            r2.embeddings_computed >= 3
        ), f"Expected ≥3 recomputed after LRU eviction, got {r2.embeddings_computed}"


@pytest.mark.asyncio
async def test_cache_disabled(workspace: Path, embedding_model: str) -> None:
    _write_distinct_files(workspace / "memory", n=2)

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        cache=CacheConfig(enabled=False),
    )

    async with MemWeave(config) as mem:
        r1 = await mem.index()
        assert (
            r1.embeddings_cached == 0
        ), f"cache.enabled=False should produce 0 cached, got {r1.embeddings_cached}"

        r2 = await mem.index(force=True)
        assert (
            r2.embeddings_cached == 0
        ), f"cache.enabled=False force re-index should still produce 0 cached, got {r2.embeddings_cached}"
