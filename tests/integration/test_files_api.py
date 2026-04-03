"""
tests/integration/test_files_api.py — Happy-path: files() metadata correctness.

Covers:
- files() returns one FileInfo per indexed file with all fields populated.
- MEMORY.md is is_evergreen=True; dated file is is_evergreen=False.
- sum(f.chunks for f in files()) == status().chunks.
- A file added via add() appears in files() immediately.

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
async def test_files_api(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / "MEMORY.md").write_text(
        "Core architecture: event-driven microservices on Kubernetes.\n"
        "Primary language: Python 3.12; API framework: FastAPI.\n"
    )
    (mem_dir / "2026-04-01.md").write_text(
        "Team decided to migrate from REST to GraphQL for the public API.\n"
        "Timeline: prototype by end of Q2, production rollout in Q3.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        file_list = await mem.files()
        assert len(file_list) >= 2, f"Expected ≥2 files, got {len(file_list)}"

        # All fields must be populated
        for f in file_list:
            assert f.path, f"path is empty for {f}"
            assert f.hash, f"hash is empty for {f.path}"
            assert f.size > 0, f"size is 0 for {f.path}"
            assert f.chunks >= 0, f"chunks is negative for {f.path}"
            assert f.source, f"source is empty for {f.path}"

        # Evergreen flags
        memory_md = next((f for f in file_list if f.path == "memory/MEMORY.md"), None)
        dated_file = next((f for f in file_list if "2026-04-01" in f.path), None)
        assert memory_md is not None, "MEMORY.md not found in files()"
        assert memory_md.is_evergreen, "MEMORY.md should be is_evergreen=True"
        assert dated_file is not None, "Dated file not found in files()"
        assert not dated_file.is_evergreen, "Dated file should be is_evergreen=False"

        # Chunk sum must match status().chunks
        status = await mem.status()
        total_chunks = sum(f.chunks for f in file_list)
        assert total_chunks == status.chunks, (
            f"Chunk count mismatch: files()={total_chunks} vs status()={status.chunks}"
        )

        # add() must be reflected in files() immediately
        new_file = mem_dir / "new_note.md"
        new_file.write_text("New decision: adopt OpenTelemetry for distributed tracing.\n")
        await mem.add(new_file)
        paths_after = [f.path for f in await mem.files()]
        assert any("new_note" in p for p in paths_after), (
            f"new_note.md not found in files() after add(): {paths_after}"
        )
