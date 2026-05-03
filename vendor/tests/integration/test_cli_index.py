"""
tests/integration/test_cli_index.py — CLI integration: memweave index against real files.

Covers: full index run, cache hit on second run, --force triggers re-embed from cache.
Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave.cli import cli  # noqa: E402

pytestmark = pytest.mark.integration


def _invoke(*args: str) -> "CliRunner.Result":  # type: ignore[name-defined]
    return CliRunner().invoke(cli, list(args))


def _parse_lines(output: str) -> dict[str, str]:
    return {
        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
        for line in output.splitlines()
        if ":" in line
    }


@pytest.mark.asyncio
async def test_index_counts_after_first_run(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "We chose PostgreSQL 16 for its JSONB support and mature replication.\n"
        "MongoDB was rejected due to the SSPL license.\n"
    )
    (workspace / "memory" / "infrastructure.md").write_text(
        "Production runs on AWS ECS Fargate with autoscaling enabled.\n"
    )

    result = await asyncio.to_thread(
        _invoke, "index", "--workspace", str(workspace), "--embedding-model", embedding_model
    )

    assert result.exit_code == 0, f"index exited {result.exit_code}: {result.output}"
    lines = _parse_lines(result.output)
    assert int(lines["Files scanned"]) == 2
    assert int(lines["Files indexed"]) == 2
    assert int(lines["Files skipped"]) == 0
    assert int(lines["Chunks created"]) >= 1
    assert int(lines["Embeddings computed"]) >= 1
    assert "ms" in lines["Duration"]


@pytest.mark.asyncio
async def test_index_second_run_uses_hash_cache(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "Redis was selected as the caching layer with a pool size of 20.\n"
    )

    # First run — indexes everything
    await asyncio.to_thread(
        _invoke, "index", "--workspace", str(workspace), "--embedding-model", embedding_model
    )

    # Second run — hash unchanged, everything skipped
    result = await asyncio.to_thread(
        _invoke, "index", "--workspace", str(workspace), "--embedding-model", embedding_model
    )

    assert result.exit_code == 0
    lines = _parse_lines(result.output)
    assert int(lines["Files indexed"]) == 0
    assert int(lines["Files skipped"]) == 1


@pytest.mark.asyncio
async def test_index_force_reindexes_all_files(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "The CI pipeline runs unit tests, integration tests, and security scans.\n"
    )

    # First run — computes embeddings
    await asyncio.to_thread(
        _invoke, "index", "--workspace", str(workspace), "--embedding-model", embedding_model
    )

    # --force run — re-indexes; embeddings come from cache (no new API call)
    result = await asyncio.to_thread(
        _invoke,
        "index",
        "--force",
        "--workspace",
        str(workspace),
        "--embedding-model",
        embedding_model,
    )

    assert result.exit_code == 0
    lines = _parse_lines(result.output)
    assert int(lines["Files indexed"]) == 1
    assert int(lines["Files skipped"]) == 0
    assert int(lines["Embeddings cached"]) >= 1
