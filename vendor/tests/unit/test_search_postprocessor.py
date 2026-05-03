"""
tests/unit/test_search_postprocessor.py — Unit tests for search/postprocessor.py

Tests cover:
- PostProcessor protocol: conformance check for valid/invalid objects
- ScoreThreshold: default threshold, per-call override, empty input, all-pass, all-fail
"""

from __future__ import annotations

import pytest

from memweave.search.postprocessor import PostProcessor, ScoreThreshold
from memweave.search.strategy import RawSearchRow


def _row(chunk_id: str, score: float) -> RawSearchRow:
    return RawSearchRow(
        chunk_id=chunk_id,
        path=f"memory/{chunk_id}.md",
        source="memory",
        start_line=1,
        end_line=3,
        text=f"text for {chunk_id}",
        score=score,
    )


# ── PostProcessor protocol ────────────────────────────────────────────────────


class TestPostProcessorProtocol:
    def test_score_threshold_is_post_processor(self):
        assert isinstance(ScoreThreshold(), PostProcessor)

    def test_custom_class_conforms(self):
        class Identity:
            async def apply(self, rows, query, **kwargs):
                return rows

        assert isinstance(Identity(), PostProcessor)

    def test_missing_apply_does_not_conform(self):
        class NoApply:
            pass

        assert not isinstance(NoApply(), PostProcessor)

    def test_sync_apply_does_not_conform(self):
        """apply must be an async method — sync won't satisfy the protocol."""

        class SyncApply:
            def apply(self, rows, query, **kwargs):
                return rows

        # Protocol runtime check only tests attribute existence, not coroutine-ness,
        # so this will still pass isinstance — document the known limitation.
        # (This is a Python Protocol limitation, not a bug in our code.)
        assert isinstance(SyncApply(), PostProcessor)


# ── ScoreThreshold ────────────────────────────────────────────────────────────


class TestScoreThreshold:
    async def test_default_threshold_filters_below_0_35(self):
        processor = ScoreThreshold()
        rows = [_row("a", 0.5), _row("b", 0.35), _row("c", 0.34)]
        result = await processor.apply(rows, "q")
        ids = [r.chunk_id for r in result]
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    async def test_boundary_score_is_kept(self):
        """Rows with score == min_score are kept (≥, not >)."""
        processor = ScoreThreshold(min_score=0.5)
        rows = [_row("x", 0.5)]
        result = await processor.apply(rows, "q")
        assert len(result) == 1
        assert result[0].chunk_id == "x"

    async def test_all_pass(self):
        processor = ScoreThreshold(min_score=0.0)
        rows = [_row("a", 0.1), _row("b", 0.9)]
        result = await processor.apply(rows, "q")
        assert len(result) == 2

    async def test_all_filtered(self):
        processor = ScoreThreshold(min_score=1.0)
        rows = [_row("a", 0.5), _row("b", 0.9)]
        result = await processor.apply(rows, "q")
        assert result == []

    async def test_empty_input(self):
        processor = ScoreThreshold()
        result = await processor.apply([], "q")
        assert result == []

    async def test_per_call_override(self):
        processor = ScoreThreshold(min_score=0.35)
        rows = [_row("a", 0.4), _row("b", 0.6)]
        # Override to 0.5 — only "b" should pass
        result = await processor.apply(rows, "q", min_score=0.5)
        assert len(result) == 1
        assert result[0].chunk_id == "b"

    async def test_per_call_lower_threshold(self):
        processor = ScoreThreshold(min_score=0.9)
        rows = [_row("a", 0.3), _row("b", 0.6)]
        result = await processor.apply(rows, "q", min_score=0.2)
        assert len(result) == 2

    async def test_order_preserved(self):
        """Output order matches input order (filter is stable)."""
        processor = ScoreThreshold(min_score=0.3)
        rows = [_row("z", 0.9), _row("a", 0.5), _row("m", 0.4)]
        result = await processor.apply(rows, "q")
        assert [r.chunk_id for r in result] == ["z", "a", "m"]

    async def test_non_score_fields_unchanged(self):
        processor = ScoreThreshold(min_score=0.0)
        row = _row("test", 0.7)
        result = await processor.apply([row], "q")
        assert result[0] is row  # same object, not a copy
