"""
tests/unit/test_search_hybrid.py — Unit tests for search/hybrid.py

Tests cover:
- merge_hybrid_results: score calculation, ordering, deduplication,
  snippet preference, limit, single-backend inputs
- HybridSearch: protocol conformance, candidate pool sizing
"""

from __future__ import annotations

import pytest

from memweave.search.hybrid import HybridSearch, merge_hybrid_results
from memweave.search.strategy import RawSearchRow, SearchStrategy


def row(
    chunk_id: str,
    score: float,
    *,
    vs: float | None = None,
    ts: float | None = None,
    text: str = "text",
) -> RawSearchRow:
    """Convenience constructor for test rows."""
    return RawSearchRow(
        chunk_id=chunk_id,
        path=f"memory/{chunk_id}.md",
        source="memory",
        start_line=1,
        end_line=3,
        text=text,
        score=score,
        vector_score=vs,
        text_score=ts,
    )


# ── merge_hybrid_results ──────────────────────────────────────────────────────


class TestMergeHybridResults:
    def test_vector_only_input(self):
        """When no keyword results, scores come entirely from vector weight."""
        vec_rows = [row("id1", 0.9, vs=0.9), row("id2", 0.5, vs=0.5)]
        result = merge_hybrid_results(vec_rows, [], vector_weight=0.7, text_weight=0.3)
        assert len(result) == 2
        # id1: 0.7 * 0.9 + 0.3 * 0.0 = 0.63
        assert abs(result[0].score - 0.63) < 1e-9
        # id2: 0.7 * 0.5 + 0.3 * 0.0 = 0.35
        assert abs(result[1].score - 0.35) < 1e-9

    def test_keyword_only_input(self):
        """When no vector results, scores come entirely from text weight."""
        kw_rows = [row("id1", 0.8, ts=0.8), row("id2", 0.4, ts=0.4)]
        result = merge_hybrid_results([], kw_rows, vector_weight=0.7, text_weight=0.3)
        assert len(result) == 2
        # id1: 0.7 * 0.0 + 0.3 * 0.8 = 0.24
        assert abs(result[0].score - 0.24) < 1e-9
        # id2: 0.7 * 0.0 + 0.3 * 0.4 = 0.12
        assert abs(result[1].score - 0.12) < 1e-9

    def test_both_backends_same_chunk(self):
        """Chunk in both backends: combined = vector_weight*vs + text_weight*ts."""
        vec_rows = [row("id1", 0.9, vs=0.9)]
        kw_rows = [row("id1", 0.6, ts=0.6)]
        result = merge_hybrid_results(vec_rows, kw_rows, vector_weight=0.7, text_weight=0.3)
        assert len(result) == 1
        # 0.7 * 0.9 + 0.3 * 0.6 = 0.63 + 0.18 = 0.81
        assert abs(result[0].score - 0.81) < 1e-9

    def test_both_backends_different_chunks(self):
        """Union: each backend contributes its unique chunks with 0 for missing."""
        vec_rows = [row("id1", 0.9, vs=0.9)]
        kw_rows = [row("id2", 0.8, ts=0.8)]
        result = merge_hybrid_results(vec_rows, kw_rows, vector_weight=0.7, text_weight=0.3)
        assert len(result) == 2
        scores = {r.chunk_id: r.score for r in result}
        assert abs(scores["id1"] - 0.63) < 1e-9  # 0.7*0.9 + 0.3*0
        assert abs(scores["id2"] - 0.24) < 1e-9  # 0.7*0 + 0.3*0.8

    def test_ordered_by_score_descending(self):
        vec_rows = [row("low", 0.2, vs=0.2), row("high", 0.9, vs=0.9)]
        result = merge_hybrid_results(vec_rows, [])
        assert result[0].chunk_id == "high"
        assert result[1].chunk_id == "low"

    def test_limit_truncates_result(self):
        vec_rows = [row(f"id{i}", float(i) / 10, vs=float(i) / 10) for i in range(1, 6)]
        result = merge_hybrid_results(vec_rows, [], limit=3)
        assert len(result) == 3

    def test_limit_keeps_top_scores(self):
        vec_rows = [row(f"id{i}", float(i) / 10, vs=float(i) / 10) for i in range(1, 6)]
        result = merge_hybrid_results(vec_rows, [], limit=2)
        assert {r.chunk_id for r in result} == {"id4", "id5"}

    def test_both_scores_populated(self):
        """Merged rows always have both vector_score and text_score set."""
        vec_rows = [row("id1", 0.9, vs=0.9)]
        kw_rows = [row("id2", 0.6, ts=0.6)]
        result = merge_hybrid_results(vec_rows, kw_rows)
        for r in result:
            assert r.vector_score is not None
            assert r.text_score is not None

    def test_missing_vector_score_is_zero(self):
        """Chunk found only in keyword results gets vector_score=0."""
        kw_rows = [row("id1", 0.7, ts=0.7)]
        result = merge_hybrid_results([], kw_rows)
        assert result[0].vector_score == 0.0

    def test_missing_text_score_is_zero(self):
        """Chunk found only in vector results gets text_score=0."""
        vec_rows = [row("id1", 0.8, vs=0.8)]
        result = merge_hybrid_results(vec_rows, [])
        assert result[0].text_score == 0.0

    def test_snippet_preference_keyword(self):
        """When chunk appears in both, keyword snippet is preferred."""
        vec_rows = [row("id1", 0.9, vs=0.9, text="vector text")]
        kw_rows = [row("id1", 0.6, ts=0.6, text="keyword text")]
        result = merge_hybrid_results(vec_rows, kw_rows)
        assert result[0].text == "keyword text"

    def test_empty_inputs_returns_empty(self):
        assert merge_hybrid_results([], []) == []

    def test_equal_weights_splits_evenly(self):
        """With 0.5/0.5 weights, chunks from both backends are weighted equally."""
        vec_rows = [row("id1", 0.8, vs=0.8)]
        kw_rows = [row("id1", 0.8, ts=0.8)]
        result = merge_hybrid_results(vec_rows, kw_rows, vector_weight=0.5, text_weight=0.5)
        # 0.5*0.8 + 0.5*0.8 = 0.8
        assert abs(result[0].score - 0.8) < 1e-9

    def test_weight_sum_not_required_to_be_one(self):
        """Weights don't need to sum to 1 — formula is a plain weighted sum."""
        vec_rows = [row("id1", 1.0, vs=1.0)]
        kw_rows = [row("id1", 1.0, ts=1.0)]
        result = merge_hybrid_results(vec_rows, kw_rows, vector_weight=0.6, text_weight=0.6)
        assert abs(result[0].score - 1.2) < 1e-9

    def test_large_candidate_pool_deduped_correctly(self):
        """100 unique chunks from both backends → 100 entries, no duplicates."""
        vec_rows = [row(f"v{i}", float(i) / 100, vs=float(i) / 100) for i in range(50)]
        kw_rows = [row(f"k{i}", float(i) / 100, ts=float(i) / 100) for i in range(50)]
        result = merge_hybrid_results(vec_rows, kw_rows)
        assert len(result) == 100
        ids = [r.chunk_id for r in result]
        assert len(set(ids)) == 100

    def test_overlapping_pool_deduped(self):
        """50 shared + 25 vector-only + 25 keyword-only = 100 total entries."""
        shared_vec = [row(f"s{i}", float(i) / 100, vs=float(i) / 100) for i in range(50)]
        shared_kw = [row(f"s{i}", float(i) / 100, ts=float(i) / 100) for i in range(50)]
        unique_vec = [row(f"v{i}", 0.1, vs=0.1) for i in range(25)]
        unique_kw = [row(f"k{i}", 0.1, ts=0.1) for i in range(25)]
        result = merge_hybrid_results(shared_vec + unique_vec, shared_kw + unique_kw)
        assert len(result) == 100


# ── HybridSearch protocol conformance ────────────────────────────────────────


class TestHybridSearchProtocol:
    def test_conforms_to_search_strategy(self):
        hs = HybridSearch()
        assert isinstance(hs, SearchStrategy)

    def test_default_weights(self):
        hs = HybridSearch()
        assert hs.vector_weight == 0.7
        assert hs.text_weight == 0.3

    def test_custom_weights(self):
        hs = HybridSearch(vector_weight=0.5, text_weight=0.5)
        assert hs.vector_weight == 0.5
        assert hs.text_weight == 0.5

    def test_default_candidate_multiplier(self):
        hs = HybridSearch()
        assert hs.candidate_multiplier == 4
