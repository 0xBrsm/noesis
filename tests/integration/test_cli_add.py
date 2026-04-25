"""
tests/integration/test_cli_add.py — CLI integration: memweave add against a single real file.

Covers: first-time add, hash-cache skip on re-add.
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
async def test_add_indexes_single_file(workspace: Path, embedding_model: str) -> None:
    file_path = workspace / "memory" / f"{date.today()}.md"
    file_path.write_text(
        "Chose Redis for caching with a pool size of 20 connections.\n"
        "Memcached was ruled out due to lack of native persistence.\n"
    )

    result = await asyncio.to_thread(
        _invoke,
        "add",
        str(file_path),
        "--workspace",
        str(workspace),
        "--embedding-model",
        embedding_model,
    )

    assert result.exit_code == 0, f"add exited {result.exit_code}: {result.output}"
    lines = _parse_lines(result.output)
    assert int(lines["Files scanned"]) == 1
    assert int(lines["Files indexed"]) == 1
    assert int(lines["Files skipped"]) == 0
    assert int(lines["Chunks created"]) >= 1
    assert int(lines["Embeddings computed"]) >= 1
    assert "ms" in lines["Duration"]


@pytest.mark.asyncio
async def test_add_second_call_skips_unchanged_file(workspace: Path, embedding_model: str) -> None:
    file_path = workspace / "memory" / f"{date.today()}.md"
    file_path.write_text("The SQS dead-letter queue retains failed messages for 14 days.\n")

    # First add
    await asyncio.to_thread(
        _invoke,
        "add",
        str(file_path),
        "--workspace",
        str(workspace),
        "--embedding-model",
        embedding_model,
    )

    # Second add — hash unchanged, should be skipped
    result = await asyncio.to_thread(
        _invoke,
        "add",
        str(file_path),
        "--workspace",
        str(workspace),
        "--embedding-model",
        embedding_model,
    )

    assert result.exit_code == 0
    lines = _parse_lines(result.output)
    assert int(lines["Files indexed"]) == 0
    assert int(lines["Files skipped"]) == 1


@pytest.mark.asyncio
async def test_add_missing_file_exits_with_error(workspace: Path, embedding_model: str) -> None:
    result = await asyncio.to_thread(
        _invoke, "add", "memory/does-not-exist.md", "--workspace", str(workspace)
    )

    assert result.exit_code == 1
    assert "file not found" in result.output.lower()
