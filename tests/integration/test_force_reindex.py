"""
tests/integration/test_force_reindex.py — Happy-path: force=True bypasses file hash, not embedding cache.

Covers:
- Normal re-index: unchanged files are skipped (files_skipped == files_scanned).
- force=True: files_skipped == 0, files_indexed > 0.
- force=True does NOT bypass the content-addressed embedding cache:
  embeddings_computed + embeddings_cached > 0 (embeddings were handled).
- Search works correctly after force reindex.

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
async def test_force_reindex(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / "2026-04-01.md").write_text(
        "Fargate was selected over ECS on EC2 to eliminate node-level maintenance overhead.\n"
        "Auto-scaling policy: scale up at 70% CPU, scale down at 30% CPU.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        r1 = await mem.index()
        assert r1.files_indexed > 0

        # Normal re-index — unchanged files skipped
        r2 = await mem.index()
        assert r2.files_skipped == r2.files_scanned, (
            f"All unchanged files should be skipped: scanned={r2.files_scanned} skipped={r2.files_skipped}"
        )

        # force=True — must re-process all files regardless of hash
        # but content-addressed embedding cache still applies (same text → cache hit)
        r3 = await mem.index(force=True)
        assert r3.files_skipped == 0, (
            f"force=True should skip nothing, got files_skipped={r3.files_skipped}"
        )
        assert r3.files_indexed > 0, "force=True should re-index all files"
        assert r3.embeddings_computed + r3.embeddings_cached > 0, (
            "Embeddings must be handled (computed or from cache) after force reindex"
        )

        # Search must still work after force reindex
        results = await mem.search("Fargate auto scaling CPU", min_score=0.1)
        assert len(results) > 0, "Search failed after force reindex"
