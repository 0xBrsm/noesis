"""
tests/unit/test_search_mmr.py — Unit tests for search/mmr.py

Tests cover:
- tokenize_for_mmr: token extraction, lowercase, dedup
- jaccard_similarity: identity, disjoint, partial, empty-set edge cases
- compute_mmr_score: formula verification, lambda extremes
- mmr_rerank: single/zero items, lambda=1 pure relevance, lambda=0 pure diversity,
  result is a permutation of input, tiebreaker on equal MMR
- MMRReranker wrapper: async apply, per-call lambda override
"""

from __future__ import annotations

import pytest

from memweave.search.mmr import (
    MMRReranker,
    compute_mmr_score,
    jaccard_similarity,
    mmr_rerank,
    tokenize_for_mmr,
)
from memweave.search.strategy import RawSearchRow


def _row(chunk_id: str, score: float, text: str = "") -> RawSearchRow:
    return RawSearchRow(
        chunk_id=chunk_id,
        path=f"memory/{chunk_id}.md",
        source="memory",
        start_line=1,
        end_line=3,
        text=text or f"text for {chunk_id}",
        score=score,
    )


# ── tokenize_for_mmr ──────────────────────────────────────────────────────────

class TestTokenizeForMmr:
    def test_basic_extraction(self):
        tokens = tokenize_for_mmr("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tokens = tokenize_for_mmr("PostgreSQL")
        assert "postgresql" in tokens

    def test_digits_included(self):
        tokens = tokenize_for_mmr("python3 version2")
        assert "python3" in tokens
        assert "version2" in tokens

    def test_underscore_included(self):
        tokens = tokenize_for_mmr("snake_case variable")
        assert "snake_case" in tokens

    def test_punctuation_stripped(self):
        tokens = tokenize_for_mmr("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        # commas and exclamation marks should not appear
        for t in tokens:
            assert "," not in t
            assert "!" not in t

    def test_returns_frozenset(self):
        assert isinstance(tokenize_for_mmr("hello"), frozenset)

    def test_deduplication(self):
        tokens = tokenize_for_mmr("the the the")
        assert len(tokens) == 1

    def test_empty_string(self):
        assert tokenize_for_mmr("") == frozenset()

    def test_only_punctuation(self):
        assert tokenize_for_mmr("!!! ???") == frozenset()


# ── jaccard_similarity ────────────────────────────────────────────────────────

class TestJaccardSimilarity:
    def test_identical_sets(self):
        a = frozenset({"a", "b", "c"})
        assert jaccard_similarity(a, a) == 1.0

    def test_disjoint_sets(self):
        a = frozenset({"a"})
        b = frozenset({"b"})
        assert jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset({"a", "b"})
        b = frozenset({"b", "c"})
        # |{b}| / |{a,b,c}| = 1/3
        assert abs(jaccard_similarity(a, b) - 1 / 3) < 1e-9

    def test_both_empty(self):
        assert jaccard_similarity(frozenset(), frozenset()) == 1.0

    def test_one_empty(self):
        a = frozenset({"a"})
        assert jaccard_similarity(a, frozenset()) == 0.0
        assert jaccard_similarity(frozenset(), a) == 0.0

    def test_subset(self):
        a = frozenset({"a", "b"})
        b = frozenset({"a", "b", "c"})
        # |{a,b}| / |{a,b,c}| = 2/3
        assert abs(jaccard_similarity(a, b) - 2 / 3) < 1e-9

    def test_symmetry(self):
        a = frozenset({"x", "y"})
        b = frozenset({"y", "z"})
        assert jaccard_similarity(a, b) == jaccard_similarity(b, a)


# ── compute_mmr_score ─────────────────────────────────────────────────────────

class TestComputeMmrScore:
    def test_pure_relevance(self):
        # lambda=1: MMR = 1*relevance - 0*similarity = relevance
        assert compute_mmr_score(0.8, 0.5, lam=1.0) == pytest.approx(0.8)

    def test_pure_diversity(self):
        # lambda=0: MMR = 0*relevance - 1*similarity = -similarity
        assert compute_mmr_score(0.8, 0.5, lam=0.0) == pytest.approx(-0.5)

    def test_balanced(self):
        # lambda=0.7: 0.7*0.8 - 0.3*0.5 = 0.56 - 0.15 = 0.41
        assert compute_mmr_score(0.8, 0.5, lam=0.7) == pytest.approx(0.41)

    def test_zero_similarity_no_penalty(self):
        assert compute_mmr_score(0.6, 0.0, lam=0.5) == pytest.approx(0.3)

    def test_negative_result_possible(self):
        # High similarity, low relevance, low lambda
        score = compute_mmr_score(0.1, 0.9, lam=0.1)
        assert score < 0


# ── mmr_rerank ────────────────────────────────────────────────────────────────

class TestMmrRerank:
    def test_empty_list(self):
        assert mmr_rerank([]) == []

    def test_single_item(self):
        row = _row("a", 0.9)
        assert mmr_rerank([row]) == [row]

    def test_returns_same_items(self):
        rows = [_row("a", 0.9), _row("b", 0.7), _row("c", 0.5)]
        result = mmr_rerank(rows)
        assert len(result) == len(rows)
        assert {r.chunk_id for r in result} == {"a", "b", "c"}

    def test_lambda_1_pure_relevance_order(self):
        """lambda=1 should return rows sorted by score descending."""
        rows = [_row("b", 0.6), _row("c", 0.3), _row("a", 0.9)]
        result = mmr_rerank(rows, lam=1.0)
        assert [r.chunk_id for r in result] == ["a", "b", "c"]

    def test_lambda_0_diversity_first(self):
        """lambda=0 penalises similarity heavily; first pick is highest score,
        subsequent picks should diverge in content."""
        # Two very similar rows + one very different
        rows = [
            _row("similar1", 0.9, "the quick brown fox jumps over"),
            _row("similar2", 0.85, "the quick brown fox leaps over"),  # nearly same
            _row("different", 0.8, "database connection pooling architecture"),
        ]
        result = mmr_rerank(rows, lam=0.0)
        # First pick: highest relevance (all scores are normalised to 1.0 at λ=0,
        # so it's still the top-score row due to tiebreaker)
        assert result[0].chunk_id == "similar1"
        # Second pick: the very different row should beat the near-duplicate
        assert result[1].chunk_id == "different"

    def test_lambda_clamped_below_0(self):
        rows = [_row("a", 0.9), _row("b", 0.5)]
        # Should not raise; clamped to 0.0
        result = mmr_rerank(rows, lam=-1.0)
        assert len(result) == 2

    def test_lambda_clamped_above_1(self):
        rows = [_row("a", 0.9), _row("b", 0.5)]
        result = mmr_rerank(rows, lam=2.0)
        # Clamped to 1.0 → pure relevance
        assert result[0].chunk_id == "a"

    def test_all_identical_scores(self):
        """Tiebreaker by raw score; all same score → arbitrary but stable."""
        rows = [_row("a", 0.5, "alpha beta"), _row("b", 0.5, "gamma delta")]
        result = mmr_rerank(rows, lam=0.7)
        assert len(result) == 2

    def test_row_objects_preserved(self):
        """Returned rows are the same objects (not copies)."""
        rows = [_row("a", 0.9), _row("b", 0.7)]
        result = mmr_rerank(rows, lam=0.7)
        assert result[0] in rows
        assert result[1] in rows

    def test_uniform_scores_dont_crash(self):
        """score_range == 0 → all normalised to 1.0; should not divide by zero."""
        rows = [_row("a", 0.5, "alpha beta gamma"), _row("b", 0.5, "delta epsilon")]
        result = mmr_rerank(rows, lam=0.7)
        assert len(result) == 2


# ── MMRReranker (PostProcessor wrapper) ──────────────────────────────────────

class TestMMRReranker:
    async def test_apply_returns_reranked(self):
        reranker = MMRReranker(lam=1.0)
        rows = [_row("b", 0.5), _row("a", 0.9)]
        result = await reranker.apply(rows, "query")
        assert result[0].chunk_id == "a"

    async def test_per_call_lambda_override(self):
        reranker = MMRReranker(lam=1.0)
        rows = [_row("a", 0.9, "the same same same"), _row("b", 0.5, "the same same same")]
        # With lam=1.0 default, order is by relevance. With lam=0 via kwarg,
        # the second call penalises duplicates — just check it doesn't crash.
        result = await reranker.apply(rows, "query", mmr_lambda=0.0)
        assert len(result) == 2

    async def test_empty_input(self):
        reranker = MMRReranker()
        result = await reranker.apply([], "query")
        assert result == []

    async def test_default_lambda_stored(self):
        reranker = MMRReranker(lam=0.3)
        assert reranker.lam == 0.3
