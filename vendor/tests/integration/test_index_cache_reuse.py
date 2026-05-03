"""
tests/integration/test_index_cache_reuse.py — Happy-path: hash-based dedup + embedding cache.

Covers second index() skipping unchanged files:
- First index(): embeddings_computed > 0, embeddings_cached == 0.
- Second index() (no changes): files_skipped == files_scanned, embeddings_computed == 0.

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
async def test_index_cache_reuse(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / "2026-04-01.md").write_text(
        "We chose PostgreSQL 16 for its JSONB support and mature replication story.\n"
        "MongoDB was rejected due to a licensing change (SSPL) flagged by legal.\n"
    )
    (workspace / "memory" / "team-conventions.md").write_text(
        "All PRs require two approvals. Hotfixes go through the fast-track review process.\n"
        "Branch naming: feature/<ticket-id>-short-description.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        r1 = await mem.index()

        assert r1.files_indexed > 0, "Expected files indexed on first run"
        assert r1.embeddings_computed > 0, "Expected embeddings computed on cold start"
        assert r1.embeddings_cached == 0, "Expected no cache hits on cold start"
        assert r1.duration_ms > 0

        r2 = await mem.index()

        assert (
            r2.files_skipped == r2.files_scanned
        ), f"All unchanged files should be skipped: scanned={r2.files_scanned} skipped={r2.files_skipped}"
        assert (
            r2.embeddings_computed == 0
        ), f"No new embeddings should be computed on unchanged re-index, got {r2.embeddings_computed}"
