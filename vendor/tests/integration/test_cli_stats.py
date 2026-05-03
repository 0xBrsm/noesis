"""
tests/integration/test_cli_stats.py — CLI integration: memweave stats against a real index.

Covers end-to-end path: Python API indexes real files → CLI reads and formats the
resulting StoreStatus. Verifies that the CLI output reflects actual index state.
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
    """Run a CLI command in a thread so asyncio.run() inside the CLI can create its own loop."""
    return CliRunner().invoke(cli, list(args))


@pytest.mark.asyncio
async def test_stats_reflects_real_index_state(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "We deployed to AWS ECS Fargate after evaluating Kubernetes.\n"
        "Fargate was chosen to reduce operational overhead for the small team.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )
    async with MemWeave(config) as mem:
        await mem.index()

    result = await asyncio.to_thread(_invoke, "stats", "--workspace", str(workspace))

    assert result.exit_code == 0, f"stats exited {result.exit_code}: {result.output}"
    assert "Files" in result.output
    assert "Chunks" in result.output
    assert "Dirty:            no" in result.output
    assert "Workspace" in result.output
    assert "Search mode" in result.output


@pytest.mark.asyncio
async def test_stats_shows_nonzero_counts_after_index(
    workspace: Path, embedding_model: str
) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "PostgreSQL 16 was chosen for JSONB and mature replication.\n"
        "MongoDB was rejected due to the SSPL license.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )
    async with MemWeave(config) as mem:
        await mem.index()

    result = await asyncio.to_thread(_invoke, "stats", "--workspace", str(workspace))

    assert result.exit_code == 0
    # File and chunk counts must be > 0 — extract from output
    lines = {
        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
        for line in result.output.splitlines()
        if ":" in line and not line.startswith("─")
    }
    assert int(lines["Files"]) >= 1, f"Expected Files >= 1, got: {lines.get('Files')}"
    assert int(lines["Chunks"]) >= 1, f"Expected Chunks >= 1, got: {lines.get('Chunks')}"


@pytest.mark.asyncio
async def test_stats_dirty_before_index(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "Redis was selected as the caching layer with a pool size of 20.\n"
    )

    # Do NOT index — stats should show dirty=True and the stale warning
    result = await asyncio.to_thread(_invoke, "stats", "--workspace", str(workspace))

    assert result.exit_code == 0
    assert "Dirty:            yes" in result.output
    assert "stale" in result.output


@pytest.mark.asyncio
async def test_stats_json_output_after_index(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "The CI pipeline runs unit tests, integration tests, and security scans.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )
    async with MemWeave(config) as mem:
        await mem.index()

    result = await asyncio.to_thread(_invoke, "stats", "--workspace", str(workspace), "--json")

    assert result.exit_code == 0
    data = json.loads(result.output)

    assert data["files"] >= 1, f"Expected files >= 1, got {data['files']}"
    assert data["chunks"] >= 1, f"Expected chunks >= 1, got {data['chunks']}"
    assert data["dirty"] is False
    assert data["search_mode"] in {"hybrid", "fts-only", "vector-only"}
    assert data["workspace_dir"] == str(workspace)
