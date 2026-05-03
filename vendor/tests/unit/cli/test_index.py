"""tests/unit/cli/test_index.py — Milestone 3: memweave index unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from memweave.cli import cli
from memweave.exceptions import StorageError
from memweave.types import IndexResult


def _make_result(
    *,
    files_scanned: int = 14,
    files_indexed: int = 2,
    files_skipped: int = 12,
    files_deleted: int = 0,
    chunks_created: int = 18,
    embeddings_cached: int = 8,
    embeddings_computed: int = 10,
    duration_ms: float = 1243.0,
) -> IndexResult:
    return IndexResult(
        files_scanned=files_scanned,
        files_indexed=files_indexed,
        files_skipped=files_skipped,
        files_deleted=files_deleted,
        chunks_created=chunks_created,
        embeddings_cached=embeddings_cached,
        embeddings_computed=embeddings_computed,
        duration_ms=duration_ms,
    )


def _mock_mem(result: IndexResult) -> AsyncMock:
    mem = AsyncMock()
    mem.__aenter__.return_value = mem
    mem.index.return_value = result
    return mem


class TestIndexDefaultOutput:
    def test_contains_all_fields(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_result())):
            result = CliRunner().invoke(cli, ["index"])

        assert result.exit_code == 0
        assert "Files scanned" in result.output
        assert "Files indexed" in result.output
        assert "Files skipped" in result.output
        assert "Files deleted" in result.output
        assert "Chunks created" in result.output
        assert "Embeddings cached" in result.output
        assert "Embeddings computed" in result.output
        assert "Duration" in result.output

    def test_shows_correct_values(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_result())):
            result = CliRunner().invoke(cli, ["index"])

        assert "14" in result.output
        assert "2" in result.output
        assert "12" in result.output
        assert "18" in result.output
        assert "1243ms" in result.output

    def test_duration_rounded_to_ms(self) -> None:
        with patch(
            "memweave.cli.MemWeave", return_value=_mock_mem(_make_result(duration_ms=312.7))
        ):
            result = CliRunner().invoke(cli, ["index"])

        assert "313ms" in result.output

    def test_force_flag_passed_through(self) -> None:
        mem = _mock_mem(_make_result())
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["index", "--force"])

        mem.index.assert_awaited_once_with(force=True)

    def test_no_force_flag_defaults_false(self) -> None:
        mem = _mock_mem(_make_result())
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["index"])

        mem.index.assert_awaited_once_with(force=False)

    def test_embedding_model_passed_to_config(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["index", "--embedding-model", "sap/text-embedding-3-small"])

        config = mock_cls.call_args[0][0]
        assert config.embedding.model == "sap/text-embedding-3-small"

    def test_no_embedding_model_uses_default(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["index"])

        config = mock_cls.call_args[0][0]
        assert config.embedding.model == "text-embedding-3-small"

    def test_progress_enabled_by_default(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["index"])

        config = mock_cls.call_args[0][0]
        assert config.progress is True

    def test_quiet_flag_disables_progress(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["index", "--quiet"])

        config = mock_cls.call_args[0][0]
        assert config.progress is False


class TestIndexErrorHandling:
    def test_storage_error_exits_nonzero_with_message(self) -> None:
        mem = AsyncMock()
        mem.__aenter__.return_value = mem
        mem.index.side_effect = StorageError("database locked")

        with patch("memweave.cli.MemWeave", return_value=mem):
            result = CliRunner().invoke(cli, ["index"])

        assert result.exit_code == 1
        assert "database locked" in result.output
