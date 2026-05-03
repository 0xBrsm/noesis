"""tests/unit/cli/test_stats.py — Milestone 2: memweave stats unit tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from memweave.cli import cli
from memweave.exceptions import StorageError
from memweave.types import StoreStatus


def _make_status(
    *,
    dirty: bool = False,
    cache_max_entries: int | None = None,
    model: str | None = "text-embedding-3-small",
) -> StoreStatus:
    return StoreStatus(
        files=14,
        chunks=87,
        dirty=dirty,
        workspace_dir="/project",
        db_path="/project/memory/.memweave/index.sqlite",
        search_mode="hybrid",
        provider="litellm",
        model=model,
        fts_available=True,
        vector_available=True,
        cache_entries=52,
        cache_max_entries=cache_max_entries,
        watcher_active=False,
    )


def _mock_mem(status: StoreStatus) -> AsyncMock:
    mem = AsyncMock()
    mem.__aenter__.return_value = mem
    mem.status.return_value = status
    return mem


class TestStatsDefaultOutput:
    def test_contains_all_storestatus_fields(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status())):
            result = CliRunner().invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "Workspace" in result.output
        assert "DB path" in result.output
        assert "Search mode" in result.output
        assert "Provider" in result.output
        assert "Model" in result.output
        assert "Files" in result.output
        assert "Chunks" in result.output
        assert "Cache entries" in result.output
        assert "Cache max" in result.output
        assert "Dirty" in result.output
        assert "Watcher active" in result.output
        assert "FTS available" in result.output
        assert "Vector available" in result.output

    def test_shows_correct_values(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status())):
            result = CliRunner().invoke(cli, ["stats"])

        assert "/project" in result.output
        assert "hybrid" in result.output
        assert "litellm" in result.output
        assert "text-embedding-3-small" in result.output
        assert "14" in result.output
        assert "87" in result.output

    def test_cache_max_none_shows_unlimited(self) -> None:
        with patch(
            "memweave.cli.MemWeave", return_value=_mock_mem(_make_status(cache_max_entries=None))
        ):
            result = CliRunner().invoke(cli, ["stats"])

        assert "unlimited" in result.output

    def test_cache_max_int_shows_number(self) -> None:
        with patch(
            "memweave.cli.MemWeave", return_value=_mock_mem(_make_status(cache_max_entries=500))
        ):
            result = CliRunner().invoke(cli, ["stats"])

        assert "500" in result.output

    def test_model_none_shows_placeholder(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status(model=None))):
            result = CliRunner().invoke(cli, ["stats"])

        assert "(none)" in result.output


class TestStatsDirtyWarning:
    def test_dirty_true_shows_stale_warning(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status(dirty=True))):
            result = CliRunner().invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "stale" in result.output
        assert "memweave index" in result.output

    def test_dirty_false_no_stale_warning(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status(dirty=False))):
            result = CliRunner().invoke(cli, ["stats"])

        assert "stale" not in result.output


class TestStatsJsonOutput:
    def test_json_flag_outputs_valid_json(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status())):
            result = CliRunner().invoke(cli, ["stats", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_contains_all_fields(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_status())):
            result = CliRunner().invoke(cli, ["stats", "--json"])

        data = json.loads(result.output)
        expected_fields = {
            "files",
            "chunks",
            "dirty",
            "workspace_dir",
            "db_path",
            "search_mode",
            "provider",
            "model",
            "fts_available",
            "vector_available",
            "cache_entries",
            "cache_max_entries",
            "watcher_active",
        }
        assert expected_fields == set(data.keys())

    def test_json_values_match_status(self) -> None:
        status = _make_status(dirty=True, cache_max_entries=200)
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(status)):
            result = CliRunner().invoke(cli, ["stats", "--json"])

        data = json.loads(result.output)
        assert data["files"] == 14
        assert data["chunks"] == 87
        assert data["dirty"] is True
        assert data["cache_max_entries"] == 200
        assert data["search_mode"] == "hybrid"

    def test_json_cache_max_none_serialises_as_null(self) -> None:
        with patch(
            "memweave.cli.MemWeave", return_value=_mock_mem(_make_status(cache_max_entries=None))
        ):
            result = CliRunner().invoke(cli, ["stats", "--json"])

        data = json.loads(result.output)
        assert data["cache_max_entries"] is None


class TestStatsErrorHandling:
    def test_storage_error_exits_nonzero_with_message(self) -> None:
        mem = AsyncMock()
        mem.__aenter__.return_value = mem
        mem.status.side_effect = StorageError("database locked")

        with patch("memweave.cli.MemWeave", return_value=mem):
            result = CliRunner().invoke(cli, ["stats"])

        assert result.exit_code == 1
        assert "database locked" in result.output
