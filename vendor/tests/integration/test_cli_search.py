"""
tests/integration/test_cli_search.py — CLI integration: memweave search against a real index.

Covers: basic query returns results, max-results cap, source-filter, JSON output,
        keyword strategy fallback, result quality (score > 0.3, sorted descending).
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


@pytest.fixture()
async def indexed_workspace(workspace: Path, embedding_model: str) -> Path:
    """Workspace with two indexed files covering database and caching topics."""
    (workspace / "memory" / f"{date.today()}.md").write_text(
        "PostgreSQL 16 was chosen for its JSONB support and full-text search.\n"
        "The decision was made after evaluating MySQL and SQLite.\n"
    )
    (workspace / "memory" / "sessions").mkdir()
    (workspace / "memory" / "sessions" / f"{date.today()}.md").write_text(
        "Redis was selected as the caching layer for session storage.\n"
        "ElastiCache r6g nodes provide low-latency access.\n"
    )

    config = MemoryConfig(workspace_dir=workspace, embedding=EmbeddingConfig(model=embedding_model))
    async with MemWeave(config) as mem:
        await mem.index()

    return workspace


@pytest.mark.asyncio
async def test_search_returns_results(indexed_workspace: Path, embedding_model: str) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "PostgreSQL",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
    )

    assert result.exit_code == 0, f"search exited {result.exit_code}: {result.output}"
    assert "Score" in result.output
    assert "Path" in result.output


@pytest.mark.asyncio
async def test_search_top_result_score_above_threshold(
    indexed_workspace: Path, embedding_model: str
) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "PostgreSQL",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
        "--json",
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) >= 1
    assert data[0]["score"] > 0.3
    scores = [r["score"] for r in data]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_search_max_results_caps_output(
    indexed_workspace: Path, embedding_model: str
) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "PostgreSQL",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
        "--max-results",
        "1",
        "--json",
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1


@pytest.mark.asyncio
async def test_search_source_filter(indexed_workspace: Path, embedding_model: str) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "caching",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
        "--source-filter",
        "sessions",
        "--json",
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert all(r["source"] == "sessions" for r in data)


@pytest.mark.asyncio
async def test_search_json_contains_all_fields(
    indexed_workspace: Path, embedding_model: str
) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "PostgreSQL",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
        "--json",
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) >= 1

    entry = data[0]
    assert set(entry.keys()) == {
        "path",
        "start_line",
        "end_line",
        "score",
        "snippet",
        "source",
        "vector_score",
        "text_score",
    }
    assert entry["vector_score"] is not None
    assert entry["score"] > 0


@pytest.mark.asyncio
async def test_search_keyword_strategy(indexed_workspace: Path, embedding_model: str) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "PostgreSQL",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
        "--strategy",
        "keyword",
        "--min-score",
        "0.0",
    )

    assert result.exit_code == 0
    assert "Score" in result.output


@pytest.mark.asyncio
async def test_search_empty_result_no_error(indexed_workspace: Path, embedding_model: str) -> None:
    result = await asyncio.to_thread(
        _invoke,
        "search",
        "xyzzy nonexistent gibberish query zzz",
        "--workspace",
        str(indexed_workspace),
        "--embedding-model",
        embedding_model,
        "--min-score",
        "0.99",
    )

    assert result.exit_code == 0
    assert "No results found" in result.output
