"""
tests/integration/test_cli_files.py — CLI integration: memweave files against a real index.

Covers: listing all files, source label filtering, --json output.
Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig  # noqa: E402
from memweave.cli import cli  # noqa: E402

pytestmark = pytest.mark.integration


def _invoke(*args: str) -> "CliRunner.Result":  # type: ignore[name-defined]
    return CliRunner().invoke(cli, list(args))


@pytest.mark.asyncio
async def test_files_lists_all_indexed_files(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "PostgreSQL 16 chosen for JSONB support.\n"
    )
    (workspace / "memory" / "sessions").mkdir()
    (workspace / "memory" / "sessions" / f"{date.today()}.md").write_text(
        "Standup: reviewed deployment pipeline.\n"
    )

    config = MemoryConfig(workspace_dir=workspace, embedding=EmbeddingConfig(model=embedding_model))
    async with MemWeave(config) as mem:
        await mem.index()

    result = await asyncio.to_thread(_invoke, "files", "--workspace", str(workspace))

    assert result.exit_code == 0, f"files exited {result.exit_code}: {result.output}"
    assert f"memory/{date.today()}.md" in result.output
    assert f"memory/sessions/{date.today()}.md" in result.output
    assert "memory" in result.output
    assert "sessions" in result.output


@pytest.mark.asyncio
async def test_files_source_filter(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text("Redis selected as caching layer.\n")
    (workspace / "memory" / "sessions").mkdir()
    (workspace / "memory" / "sessions" / f"{date.today()}.md").write_text(
        "Sprint planning: agreed on two-week cycles.\n"
    )

    config = MemoryConfig(workspace_dir=workspace, embedding=EmbeddingConfig(model=embedding_model))
    async with MemWeave(config) as mem:
        await mem.index()

    result = await asyncio.to_thread(
        _invoke, "files", "--workspace", str(workspace), "--source", "sessions"
    )

    assert result.exit_code == 0
    assert f"memory/sessions/{date.today()}.md" in result.output
    assert f"memory/{date.today()}.md" not in result.output


@pytest.mark.asyncio
async def test_files_json_output(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "Fargate chosen to reduce operational overhead.\n"
    )

    config = MemoryConfig(workspace_dir=workspace, embedding=EmbeddingConfig(model=embedding_model))
    async with MemWeave(config) as mem:
        await mem.index()

    result = await asyncio.to_thread(_invoke, "files", "--workspace", str(workspace), "--json")

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) >= 1

    entry = data[0]
    assert set(entry.keys()) == {
        "path",
        "size",
        "hash",
        "mtime",
        "chunks",
        "is_evergreen",
        "source",
    }
    assert entry["chunks"] >= 1
    assert entry["size"] > 0
    assert len(entry["hash"]) == 64  # SHA-256 hex
