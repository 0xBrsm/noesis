"""tests/unit/cli/test_files.py — Milestone 4: memweave files unit tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from memweave.cli import cli
from memweave.exceptions import StorageError
from memweave.types import FileInfo


def _make_file(
    path: str = "memory/2026-04-18.md",
    source: str = "memory",
    chunks: int = 6,
    is_evergreen: bool = False,
) -> FileInfo:
    return FileInfo(
        path=path,
        size=512,
        hash="abc123",
        mtime=1714000000.0,
        chunks=chunks,
        is_evergreen=is_evergreen,
        source=source,
    )


def _mock_mem(files: list[FileInfo]) -> AsyncMock:
    mem = AsyncMock()
    mem.__aenter__.return_value = mem
    mem.files.return_value = files
    return mem


_SAMPLE_FILES = [
    _make_file("memory/2026-04-18.md", source="memory", chunks=6),
    _make_file("memory/2026-03-02.md", source="memory", chunks=4),
    _make_file("memory/infrastructure.md", source="memory", chunks=8, is_evergreen=True),
    _make_file("memory/researcher/findings.md", source="researcher", chunks=3),
]


class TestFilesDefaultOutput:
    def test_shows_header_columns(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files"])

        assert result.exit_code == 0
        assert "Path" in result.output
        assert "Source" in result.output
        assert "Chunks" in result.output
        assert "Evergreen" in result.output

    def test_shows_all_files(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files"])

        assert "memory/2026-04-18.md" in result.output
        assert "memory/2026-03-02.md" in result.output
        assert "memory/infrastructure.md" in result.output
        assert "memory/researcher/findings.md" in result.output

    def test_evergreen_shown_as_yes_no(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files"])

        assert "yes" in result.output  # infrastructure.md is evergreen
        assert "no" in result.output

    def test_empty_index_shows_message(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([])):
            result = CliRunner().invoke(cli, ["files"])

        assert result.exit_code == 0
        assert "No files indexed" in result.output


class TestFilesSourceFilter:
    def test_source_filter_keeps_matching_entries(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files", "--source", "memory"])

        assert result.exit_code == 0
        assert "memory/2026-04-18.md" in result.output
        assert "memory/researcher/findings.md" not in result.output

    def test_source_filter_researcher(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files", "--source", "researcher"])

        assert "memory/researcher/findings.md" in result.output
        assert "memory/2026-04-18.md" not in result.output

    def test_unknown_source_returns_empty(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files", "--source", "nonexistent"])

        assert result.exit_code == 0
        assert "No files indexed" in result.output

    def test_source_filter_is_case_sensitive(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files", "--source", "Memory"])

        assert "No files indexed" in result.output  # "Memory" ≠ "memory"


class TestFilesJsonOutput:
    def test_json_flag_outputs_valid_json(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 4

    def test_json_contains_all_fileinfo_fields(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([_make_file()])):
            result = CliRunner().invoke(cli, ["files", "--json"])

        data = json.loads(result.output)
        assert len(data) == 1
        assert set(data[0].keys()) == {
            "path",
            "size",
            "hash",
            "mtime",
            "chunks",
            "is_evergreen",
            "source",
        }

    def test_json_source_filter_applied(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_FILES)):
            result = CliRunner().invoke(cli, ["files", "--source", "researcher", "--json"])

        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["source"] == "researcher"

    def test_json_empty_returns_empty_list(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([])):
            result = CliRunner().invoke(cli, ["files", "--json"])

        assert json.loads(result.output) == []


class TestFilesErrorHandling:
    def test_storage_error_exits_nonzero_with_message(self) -> None:
        mem = AsyncMock()
        mem.__aenter__.return_value = mem
        mem.files.side_effect = StorageError("database locked")

        with patch("memweave.cli.MemWeave", return_value=mem):
            result = CliRunner().invoke(cli, ["files"])

        assert result.exit_code == 1
        assert "database locked" in result.output
