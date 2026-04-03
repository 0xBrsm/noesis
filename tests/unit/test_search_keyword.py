"""
tests/unit/test_search_keyword.py — Unit tests for search/keyword.py

Tests cover:
- build_fts_query: tokenization, quoting, sanitization, edge cases
- bm25_rank_to_score: formula correctness, monotonicity, edge cases
- is_stop_word: spot-checks across all seven language sets
- extract_keywords: stop word removal, deduplication, ordering
- KeywordSearch.search: SQL execution against a live in-memory FTS5 table
"""

from __future__ import annotations

import pytest
import aiosqlite

from memweave.search.keyword import (
    KeywordSearch,
    bm25_rank_to_score,
    build_fts_query,
    extract_keywords,
    is_stop_word,
)


# ── build_fts_query ───────────────────────────────────────────────────────────

class TestBuildFtsQuery:
    def test_simple_two_words(self):
        assert build_fts_query("hello world") == '"hello" AND "world"'

    def test_single_word(self):
        assert build_fts_query("database") == '"database"'

    def test_punctuation_stripped(self):
        # Hyphens act as token separators; they are not included in tokens
        result = build_fts_query("FOO_bar baz-1")
        assert result == '"FOO_bar" AND "baz" AND "1"'

    def test_trailing_punctuation(self):
        result = build_fts_query("which database did we pick?")
        assert result == '"which" AND "database" AND "did" AND "we" AND "pick"'

    def test_double_quotes_stripped_from_tokens(self):
        # Embedded " chars must be removed to avoid breaking FTS5 phrase syntax
        result = build_fts_query('say "hello" world')
        assert result == '"say" AND "hello" AND "world"'

    def test_pure_punctuation_returns_none(self):
        assert build_fts_query("???") is None
        assert build_fts_query("---") is None
        assert build_fts_query("   ") is None

    def test_empty_string_returns_none(self):
        assert build_fts_query("") is None

    def test_unicode_cjk_single_token(self):
        # CJK string treated as one token
        result = build_fts_query("金银价格")
        assert result == '"金银价格"'

    def test_mixed_ascii_cjk(self):
        result = build_fts_query("価格 2026年")
        assert result == '"価格" AND "2026年"'

    def test_preserves_case(self):
        # Token case is preserved for FTS5 (FTS5 is case-insensitive by default)
        result = build_fts_query("PostgreSQL Connection")
        assert result == '"PostgreSQL" AND "Connection"'


# ── bm25_rank_to_score ────────────────────────────────────────────────────────

class TestBm25RankToScore:
    def test_strong_negative_rank_gives_high_score(self):
        # -4.2 → 4.2 / (1 + 4.2) ≈ 0.808
        score = bm25_rank_to_score(-4.2)
        assert abs(score - (4.2 / 5.2)) < 1e-9

    def test_weak_negative_rank_gives_lower_score(self):
        score_strong = bm25_rank_to_score(-4.2)
        score_weak = bm25_rank_to_score(-0.5)
        assert score_strong > score_weak

    def test_rank_zero_gives_one(self):
        # 1 / (1 + 0) = 1.0
        assert bm25_rank_to_score(0.0) == 1.0

    def test_positive_rank_gives_low_score(self):
        # 1 / (1 + 10) ≈ 0.091
        score = bm25_rank_to_score(10.0)
        assert abs(score - (1.0 / 11.0)) < 1e-9

    def test_monotonic_negative(self):
        """More negative rank (better match) → higher score."""
        ranks = [-0.1, -0.5, -1.0, -2.0, -5.0, -10.0]
        scores = [bm25_rank_to_score(r) for r in ranks]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], f"scores[{i}] >= scores[{i+1}]"

    def test_monotonic_positive(self):
        """More positive rank (weaker) → lower score."""
        ranks = [0.0, 1.0, 5.0, 10.0, 50.0]
        scores = [bm25_rank_to_score(r) for r in ranks]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_nan_returns_near_zero(self):
        score = bm25_rank_to_score(float("nan"))
        assert abs(score - 1.0 / 1000.0) < 1e-9

    def test_pos_inf_returns_near_zero(self):
        score = bm25_rank_to_score(float("inf"))
        assert abs(score - 1.0 / 1000.0) < 1e-9

    def test_neg_inf_returns_near_zero(self):
        # -inf is non-finite → fallback, not the negative formula
        score = bm25_rank_to_score(float("-inf"))
        assert abs(score - 1.0 / 1000.0) < 1e-9

    def test_score_always_positive(self):
        for rank in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            assert bm25_rank_to_score(rank) > 0


# ── is_stop_word ──────────────────────────────────────────────────────────────

class TestIsStopWord:
    def test_english_stop_words(self):
        for word in ["the", "a", "is", "and", "what", "how", "please"]:
            assert is_stop_word(word), f"Expected '{word}' to be a stop word"

    def test_english_non_stop_words(self):
        for word in ["database", "postgresql", "memory", "vector"]:
            assert not is_stop_word(word), f"Expected '{word}' not to be a stop word"

    def test_spanish_stop_words(self):
        assert is_stop_word("el")
        assert is_stop_word("la")
        assert is_stop_word("que")

    def test_chinese_stop_words(self):
        assert is_stop_word("的")
        assert is_stop_word("我")
        assert is_stop_word("是")

    def test_japanese_stop_words(self):
        assert is_stop_word("これ")
        assert is_stop_word("です")

    def test_korean_stop_words(self):
        assert is_stop_word("나")
        assert is_stop_word("그")

    def test_case_sensitive_for_non_english(self):
        # Stop word sets store exact forms; CJK/other scripts are exact
        assert is_stop_word("的")
        assert not is_stop_word("DATABASE")  # uppercase English not in set


# ── extract_keywords ──────────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_filters_english_stop_words(self):
        keywords = extract_keywords("which database did we pick?")
        assert "which" not in keywords
        assert "did" not in keywords
        assert "we" not in keywords
        assert "database" in keywords
        assert "pick" in keywords

    def test_filters_short_ascii(self):
        # Single letters and 2-char ASCII words are filtered
        keywords = extract_keywords("a db configuration")
        assert "a" not in keywords
        assert "db" not in keywords
        assert "configuration" in keywords

    def test_deduplicates_case_insensitively(self):
        keywords = extract_keywords("PostgreSQL postgresql POSTGRESQL")
        assert len([k for k in keywords if k.lower() == "postgresql"]) == 1

    def test_preserves_first_occurrence_case(self):
        keywords = extract_keywords("PostgreSQL postgresql")
        assert keywords[0] == "PostgreSQL"

    def test_pure_numbers_filtered(self):
        keywords = extract_keywords("2026 version config")
        assert "2026" not in keywords
        assert "version" in keywords
        assert "config" in keywords

    def test_empty_query_returns_empty(self):
        assert extract_keywords("") == []

    def test_all_stop_words_returns_empty(self):
        assert extract_keywords("what is the thing") == []

    def test_mixed_language(self):
        keywords = extract_keywords("PostgreSQL 데이터베이스 connection")
        # ASCII stop words filtered; non-stop CJK kept
        assert "PostgreSQL" in keywords
        assert "connection" in keywords


# ── KeywordSearch integration ─────────────────────────────────────────────────

async def _make_fts_db() -> aiosqlite.Connection:
    """In-memory SQLite DB with chunks + chunks_fts tables for testing."""
    db = await aiosqlite.connect(":memory:")
    await db.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            path TEXT,
            source TEXT,
            model TEXT,
            start_line INTEGER,
            end_line INTEGER,
            text TEXT
        )
    """)
    await db.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            id UNINDEXED,
            path UNINDEXED,
            source UNINDEXED,
            model UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED
        )
    """)
    # Seed some rows
    rows = [
        ("id1", "memory/2026-01-01.md", "memory", "test-model", 1, 5,
         "We chose PostgreSQL for its JSONB support and reliability."),
        ("id2", "memory/2026-01-02.md", "memory", "test-model", 1, 4,
         "React was selected as the frontend framework for the dashboard."),
        ("id3", "memory/2026-01-03.md", "memory", "test-model", 1, 3,
         "Redis is used for session caching with a 24-hour TTL."),
        ("id4", "memory/2026-01-04.md", "sessions", "test-model", 1, 2,
         "PostgreSQL indexes were added to speed up the user query."),
    ]
    await db.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    await db.executemany(
        "INSERT INTO chunks_fts(id, path, source, model, start_line, end_line, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows],
    )
    await db.commit()
    return db


class TestKeywordSearch:
    async def test_basic_match(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL", None, "test-model", limit=10)
        assert len(rows) == 2
        ids = {r.chunk_id for r in rows}
        assert "id1" in ids
        assert "id4" in ids
        await db.close()

    async def test_results_ordered_by_score_descending(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL JSONB", None, "test-model", limit=10)
        # id1 mentions both "PostgreSQL" and "JSONB" → higher score than id4
        assert rows[0].chunk_id == "id1"
        await db.close()

    async def test_no_match_returns_empty(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "Kubernetes", None, "test-model", limit=10)
        assert rows == []
        await db.close()

    async def test_empty_query_returns_empty(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "???", None, "test-model", limit=10)
        assert rows == []
        await db.close()

    async def test_source_filter(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(
            db, "PostgreSQL", None, "test-model", limit=10,
            source_filter="sessions",
        )
        assert len(rows) == 1
        assert rows[0].chunk_id == "id4"
        await db.close()

    async def test_limit_respected(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL", None, "test-model", limit=1)
        assert len(rows) == 1
        await db.close()

    async def test_scores_in_range(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL connection caching", None, "test-model", limit=10)
        for row in rows:
            assert 0.0 < row.score <= 1.0, f"score {row.score} out of range"
        await db.close()

    async def test_text_score_populated(self):
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL", None, "test-model", limit=10)
        for row in rows:
            assert row.text_score is not None
            assert row.text_score == row.score
        await db.close()

    async def test_model_filter(self):
        """Chunks with a different model are not returned."""
        db = await _make_fts_db()
        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL", None, "other-model", limit=10)
        assert rows == []
        await db.close()
