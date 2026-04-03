"""
tests/integration/test_add_and_index.py — Happy-path: write a file and index it.

Covers mem.add() + mem.index() end-to-end:
- At least one file is indexed with chunks and embeddings computed.
- IndexResult fields are all meaningful.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_add_and_index(workspace: Path, embedding_model: str) -> None:
    memory_dir = workspace / "memory"
    note = memory_dir / f"{date.today()}.md"
    note.write_text(
        "We chose React 18 with Next.js 14 as the frontend framework.\n"
        "Tailwind CSS was adopted as the sole styling solution.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        result = await mem.add(note)

    assert result.files_indexed >= 1, f"Expected ≥1 file indexed, got {result.files_indexed}"
    assert result.chunks_created >= 1, f"Expected ≥1 chunk, got {result.chunks_created}"
    assert result.embeddings_computed >= 1, f"Expected ≥1 embedding, got {result.embeddings_computed}"
    assert result.duration_ms > 0, "Expected positive duration"
