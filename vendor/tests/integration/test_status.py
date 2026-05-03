"""
tests/integration/test_status.py — Happy-path: status() lifecycle correctness.

Covers diagnostic API at each lifecycle stage:
- Before index(): chunks == 0, dirty == True, watcher_active == False.
- After index(): chunks > 0, dirty == False, files > 0.
- search_mode is a known value; db_path exists on disk.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig  # noqa: E402

pytestmark = pytest.mark.integration

_KNOWN_SEARCH_MODES = {"hybrid", "fts-only", "vector-only", "unavailable"}


@pytest.mark.asyncio
async def test_status_lifecycle(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / "2026-04-01.md").write_text(
        "The CI pipeline runs unit tests, integration tests, and security scans.\n"
        "Deployment to staging is automatic; production requires manual approval.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        s0 = await mem.status()

        assert s0.chunks == 0, f"Expected 0 chunks before index, got {s0.chunks}"
        assert s0.dirty is True, "Expected dirty=True before index"
        assert s0.watcher_active is False, "Expected watcher_active=False before start_watching()"
        assert s0.search_mode in _KNOWN_SEARCH_MODES, f"Unknown search_mode: {s0.search_mode!r}"

        await mem.index()
        s1 = await mem.status()

        assert s1.chunks > 0, f"Expected chunks > 0 after index, got {s1.chunks}"
        assert s1.dirty is False, f"Expected dirty=False after index, got {s1.dirty}"
        assert s1.files > 0, f"Expected files > 0 after index, got {s1.files}"
        assert s1.search_mode in _KNOWN_SEARCH_MODES, f"Unknown search_mode: {s1.search_mode!r}"
        assert s1.db_path and Path(s1.db_path).exists(), "db_path should exist on disk"
        assert s1.workspace_dir, "workspace_dir should be non-empty"
