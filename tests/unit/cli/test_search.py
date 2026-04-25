"""tests/unit/cli/test_search.py — Milestone 5: memweave search unit tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from memweave.cli import cli
from memweave.exceptions import StorageError
from memweave.types import SearchResult


def _make_result(
    path: str = "memory/2026-04-25.md",
    start_line: int = 1,
    end_line: int = 8,
    score: float = 0.85,
    snippet: str = "PostgreSQL 16 was chosen for its JSONB support.",
    source: str = "memory",
    vector_score: float | None = 0.85,
    text_score: float | None = 0.60,
) -> SearchResult:
    return SearchResult(
        path=path,
        start_line=start_line,
        end_line=end_line,
        score=score,
        snippet=snippet,
        source=source,
        vector_score=vector_score,
        text_score=text_score,
    )


def _mock_mem(results: list[SearchResult]) -> AsyncMock:
    mem = AsyncMock()
    mem.__aenter__.return_value = mem
    mem.search.return_value = results
    return mem


_SAMPLE_RESULTS = [
    _make_result(
        "memory/2026-04-25.md",
        score=0.91,
        source="memory",
        snippet="PostgreSQL 16 chosen for JSONB support.",
        vector_score=0.91,
        text_score=0.70,
    ),
    _make_result(
        "memory/infrastructure.md",
        start_line=4,
        end_line=11,
        score=0.74,
        source="memory",
        snippet="Production Redis runs on ElastiCache r6g.",
        vector_score=0.74,
        text_score=0.55,
    ),
    _make_result(
        "memory/researcher/findings.md",
        start_line=23,
        end_line=30,
        score=0.61,
        source="researcher",
        snippet="Discussed moving from Memcached to Redis.",
        vector_score=0.61,
        text_score=None,
    ),
]


class TestSearchDefaultOutput:
    def test_shows_header_columns(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "database"])

        assert result.exit_code == 0
        assert "Score" in result.output
        assert "Path" in result.output
        assert "Lines" in result.output
        assert "Source" in result.output
        assert "Preview" in result.output

    def test_shows_all_results(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "database"])

        assert "memory/2026-04-25.md" in result.output
        assert "memory/infrastructure.md" in result.output
        assert "memory/researcher/findings.md" in result.output

    def test_shows_scores(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "database"])

        assert "0.91" in result.output
        assert "0.74" in result.output
        assert "0.61" in result.output

    def test_shows_line_ranges(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "database"])

        assert "1–8" in result.output
        assert "4–11" in result.output

    def test_shows_snippets(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "database"])

        assert "PostgreSQL" in result.output
        assert "ElastiCache" in result.output

    def test_empty_index_shows_message(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([])):
            result = CliRunner().invoke(cli, ["search", "anything"])

        assert result.exit_code == 0
        assert "No results found" in result.output


class TestSearchFlagWiring:
    def test_max_results_passed_through(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query", "--max-results", "2"])

        mem.search.assert_awaited_once()
        call_kwargs = mem.search.call_args
        assert call_kwargs.kwargs["max_results"] == 2

    def test_min_score_passed_through(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query", "--min-score", "0.7"])

        call_kwargs = mem.search.call_args
        assert call_kwargs.kwargs["min_score"] == pytest.approx(0.7)

    def test_strategy_passed_through(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query", "--strategy", "keyword"])

        call_kwargs = mem.search.call_args
        assert call_kwargs.kwargs["strategy"] == "keyword"

    def test_source_filter_passed_through(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query", "--source-filter", "memory"])

        call_kwargs = mem.search.call_args
        assert call_kwargs.kwargs["source_filter"] == "memory"

    def test_mmr_lambda_passed_as_kwarg(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query", "--mmr-lambda", "0.5"])

        call_kwargs = mem.search.call_args
        assert call_kwargs.kwargs["mmr_lambda"] == pytest.approx(0.5)

    def test_decay_half_life_passed_as_kwarg(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query", "--decay-half-life-days", "14.0"])

        call_kwargs = mem.search.call_args
        assert call_kwargs.kwargs["decay_half_life_days"] == pytest.approx(14.0)

    def test_mmr_lambda_absent_not_passed(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query"])

        call_kwargs = mem.search.call_args
        assert "mmr_lambda" not in call_kwargs.kwargs

    def test_decay_half_life_absent_not_passed(self) -> None:
        mem = _mock_mem(_SAMPLE_RESULTS)
        with patch("memweave.cli.MemWeave", return_value=mem):
            CliRunner().invoke(cli, ["search", "query"])

        call_kwargs = mem.search.call_args
        assert "decay_half_life_days" not in call_kwargs.kwargs

    def test_snippet_chars_injected_into_config(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_SAMPLE_RESULTS)
            CliRunner().invoke(cli, ["search", "query", "--snippet-chars", "200"])

        config = mock_cls.call_args[0][0]
        assert config.query.snippet_max_chars == 200

    def test_embedding_model_passed_to_config(self) -> None:
        with patch("memweave.cli.MemWeave") as mock_cls:
            mock_cls.return_value = _mock_mem(_SAMPLE_RESULTS)
            CliRunner().invoke(
                cli, ["search", "query", "--embedding-model", "sap/text-embedding-3-small"]
            )

        config = mock_cls.call_args[0][0]
        assert config.embedding.model == "sap/text-embedding-3-small"

    def test_invalid_strategy_exits_nonzero(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "query", "--strategy", "foobar"])

        assert result.exit_code != 0


class TestSearchJsonOutput:
    def test_json_flag_outputs_valid_json(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem(_SAMPLE_RESULTS)):
            result = CliRunner().invoke(cli, ["search", "database", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_json_contains_all_search_result_fields(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([_make_result()])):
            result = CliRunner().invoke(cli, ["search", "database", "--json"])

        data = json.loads(result.output)
        assert len(data) == 1
        assert set(data[0].keys()) == {
            "path",
            "start_line",
            "end_line",
            "score",
            "snippet",
            "source",
            "vector_score",
            "text_score",
        }

    def test_json_vector_and_text_scores_present(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([_make_result()])):
            result = CliRunner().invoke(cli, ["search", "database", "--json"])

        data = json.loads(result.output)
        assert data[0]["vector_score"] is not None
        assert data[0]["text_score"] is not None

    def test_json_null_scores_allowed(self) -> None:
        r = _make_result(vector_score=None, text_score=None)
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([r])):
            result = CliRunner().invoke(cli, ["search", "database", "--json"])

        data = json.loads(result.output)
        assert data[0]["vector_score"] is None
        assert data[0]["text_score"] is None

    def test_json_empty_returns_empty_list(self) -> None:
        with patch("memweave.cli.MemWeave", return_value=_mock_mem([])):
            result = CliRunner().invoke(cli, ["search", "anything", "--json"])

        assert json.loads(result.output) == []


class TestSearchErrorHandling:
    def test_storage_error_exits_nonzero_with_message(self) -> None:
        mem = AsyncMock()
        mem.__aenter__.return_value = mem
        mem.search.side_effect = StorageError("database locked")

        with patch("memweave.cli.MemWeave", return_value=mem):
            result = CliRunner().invoke(cli, ["search", "query"])

        assert result.exit_code == 1
        assert "database locked" in result.output
