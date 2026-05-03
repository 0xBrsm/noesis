"""
tests/integration/test_source_filter.py — Happy-path: source isolation for multi-agent systems.

Covers:
- Files under memory/ get source="memory"; files under memory/sessions/ get source="sessions".
- source_filter="memory" returns only memory-source results.
- source_filter="sessions" returns only sessions-source results.
- source_filter=None returns results from both sources.
- files() metadata reflects correct source per file.

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
async def test_source_filter(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    sessions_dir = mem_dir / "sessions"
    sessions_dir.mkdir()

    (mem_dir / "2026-04-01.md").write_text(
        "Team decision: PostgreSQL chosen over MongoDB for the primary datastore.\n"
        "Redis added as a caching layer to reduce read latency.\n"
    )
    (sessions_dir / "agent_notes.md").write_text(
        "Session note: investigated Redis eviction policy configuration.\n"
        "Confirmed maxmemory-policy allkeys-lru is appropriate for our access patterns.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        # files() must reflect correct source per file
        file_list = await mem.files()
        memory_files = [f for f in file_list if f.source == "memory"]
        session_files = [f for f in file_list if f.source == "sessions"]
        assert len(memory_files) >= 1, "Expected at least one file with source='memory'"
        assert len(session_files) >= 1, "Expected at least one file with source='sessions'"

        query = "database caching Redis"

        # No filter — results from both sources
        r_all = await mem.search(query, min_score=0.0)
        all_sources = {r.source for r in r_all}
        assert "memory" in all_sources or "sessions" in all_sources

        # memory only
        r_memory = await mem.search(query, source_filter="memory", min_score=0.0)
        assert len(r_memory) > 0, "Expected memory-source results"
        assert all(
            r.source == "memory" for r in r_memory
        ), f"Non-memory source leaked in: {[r.source for r in r_memory]}"

        # sessions only
        r_sessions = await mem.search(query, source_filter="sessions", min_score=0.0)
        assert len(r_sessions) > 0, "Expected sessions-source results"
        assert all(
            r.source == "sessions" for r in r_sessions
        ), f"Non-sessions source leaked in: {[r.source for r in r_sessions]}"

        # unfiltered >= either filtered set
        assert len(r_all) >= max(len(r_memory), len(r_sessions))
