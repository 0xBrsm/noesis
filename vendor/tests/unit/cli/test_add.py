"""tests/unit/cli/test_add.py — Milestone 3: memweave add unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from memweave.cli import cli
from memweave.exceptions import StorageError
from memweave.types import IndexResult


def _make_result(
    *,
    files_scanned: int = 1,
    files_indexed: int = 1,
    files_skipped: int = 0,
    files_deleted: int = 0,
    chunks_created: int = 6,
    embeddings_cached: int = 4,
    embeddings_computed: int = 2,
    duration_ms: float = 312.0,
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
    mem.add.return_value = result
    return mem


class TestAddDefaultOutput:
    def test_contains_all_fields(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_make_result())):
            result = CliRunner().invoke(cli, ["add", "memory/foo.md"])

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
            result = CliRunner().invoke(cli, ["add", "memory/foo.md"])

        assert "1" in result.output
        assert "6" in result.output
        assert "312ms" in result.output

    def test_file_argument_resolved_to_absolute(self) -> None:
        mem = _mock_mem(_make_result())
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["add", "memory/foo.md"])

        actual_path = mem.add.call_args[0][0]
        assert actual_path.endswith("memory/foo.md")
        assert actual_path.startswith("/")  # resolved to absolute

    def test_force_flag_passed_through(self) -> None:
        mem = _mock_mem(_make_result())
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["add", "memory/foo.md", "--force"])

        mem.add.assert_awaited_once_with(mem.add.call_args[0][0], force=True)

    def test_embedding_model_passed_to_config(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(
                cli, ["add", "memory/foo.md", "--embedding-model", "sap/text-embedding-3-small"]
            )

        config = mock_cls.call_args[0][0]
        assert config.embedding.model == "sap/text-embedding-3-small"

    def test_no_embedding_model_uses_default(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["add", "memory/foo.md"])

        config = mock_cls.call_args[0][0]
        assert config.embedding.model == "text-embedding-3-small"

    def test_progress_enabled_by_default(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["add", "memory/foo.md"])

        config = mock_cls.call_args[0][0]
        assert config.progress is True

    def test_quiet_flag_disables_progress(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_make_result())
            CliRunner().invoke(cli, ["add", "memory/foo.md", "--quiet"])

        config = mock_cls.call_args[0][0]
        assert config.progress is False

    def test_skipped_when_hash_unchanged(self) -> None:
        with patch(
            "memweave.cli.MemWeave",
            return_value=_mock_mem(_make_result(files_indexed=0, files_skipped=1)),
        ):
            result = CliRunner().invoke(cli, ["add", "memory/foo.md"])

        assert result.exit_code == 0
        assert "0" in result.output  # files_indexed = 0


class TestAddErrorHandling:
    def test_missing_file_exits_nonzero_with_message(self) -> None:
        mem = AsyncMock()
        mem.__aenter__.return_value = mem
        mem.add.side_effect = FileNotFoundError("File not found: /workspace/memory/missing.md")

        with patch("memweave.cli.MemWeave", return_value=mem):
            result = CliRunner().invoke(cli, ["add", "memory/missing.md"])

        assert result.exit_code == 1
        assert "file not found" in result.output.lower()
        assert "memory/missing.md" in result.output

    def test_storage_error_exits_nonzero_with_message(self) -> None:
        mem = AsyncMock()
        mem.__aenter__.return_value = mem
        mem.add.side_effect = StorageError("write failed")

        with patch("memweave.cli.MemWeave", return_value=mem):
            result = CliRunner().invoke(cli, ["add", "memory/foo.md"])

        assert result.exit_code == 1
        assert "write failed" in result.output
