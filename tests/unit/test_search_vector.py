"""
tests/unit/test_search_vector.py — Unit tests for search/vector.py

Tests cover:
- _vec_to_blob: correct float32 binary serialization
- VectorSearchUnavailableError: raised when sqlite-vec not loaded
- VectorSearch.search: SQL execution against a live sqlite-vec table
  (skipped when sqlite-vec is not installed in the test environment)
"""

from __future__ import annotations

import math
import struct

import pytest
import aiosqlite

from memweave.search.vector import (
    VectorSearch,
    VectorSearchUnavailableError,
    _vec_to_blob,
)
from memweave.embedding.vectors import normalize_embedding

# ── _vec_to_blob ──────────────────────────────────────────────────────────────


class TestVecToBlob:
    def test_two_element_vector(self):
        blob = _vec_to_blob([1.0, 2.0])
        assert len(blob) == 8  # 2 × 4 bytes
        unpacked = struct.unpack("2f", blob)
        assert abs(unpacked[0] - 1.0) < 1e-6
        assert abs(unpacked[1] - 2.0) < 1e-6

    def test_three_element_vector(self):
        blob = _vec_to_blob([0.5, -0.5, 0.0])
        assert len(blob) == 12
        unpacked = struct.unpack("3f", blob)
        assert abs(unpacked[0] - 0.5) < 1e-6
        assert abs(unpacked[1] - (-0.5)) < 1e-6
        assert abs(unpacked[2]) < 1e-6

    def test_single_element(self):
        blob = _vec_to_blob([3.14])
        assert len(blob) == 4
        (val,) = struct.unpack("1f", blob)
        assert abs(val - 3.14) < 1e-4

    def test_large_vector(self):
        vec = [float(i) / 1536 for i in range(1536)]
        blob = _vec_to_blob(vec)
        assert len(blob) == 1536 * 4


# ── VectorSearchUnavailableError ──────────────────────────────────────────────


class TestVectorSearchUnavailable:
    async def test_raises_when_sqlite_vec_not_loaded(self):
        """A plain DB connection without sqlite-vec should raise."""
        db = await aiosqlite.connect(":memory:")
        vs = VectorSearch()
        query_vec = normalize_embedding([1.0, 0.0, 0.0])
        with pytest.raises(VectorSearchUnavailableError):
            await vs.search(db, "", query_vec, "test-model", limit=5)
        await db.close()

    async def test_raises_with_helpful_message(self):
        db = await aiosqlite.connect(":memory:")
        vs = VectorSearch()
        query_vec = normalize_embedding([1.0, 0.0, 0.0])
        with pytest.raises(VectorSearchUnavailableError, match="sqlite-vec"):
            await vs.search(db, "", query_vec, "test-model", limit=5)
        await db.close()

    async def test_raises_on_none_query_vec(self):
        db = await aiosqlite.connect(":memory:")
        vs = VectorSearch()
        with pytest.raises(ValueError, match="query_vec"):
            await vs.search(db, "", None, "test-model", limit=5)
        await db.close()


# ── VectorSearch with sqlite-vec ──────────────────────────────────────────────


def _sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401

        return True
    except ImportError:
        return False


requires_sqlite_vec = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason="sqlite-vec not installed (pip install memweave[vector])",
)


async def _make_vec_db(dims: int = 4) -> aiosqlite.Connection:
    """In-memory DB with sqlite-vec loaded and test data seeded."""
    import sqlite_vec

    db = await aiosqlite.connect(":memory:")
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)

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
    await db.execute(f"""
        CREATE VIRTUAL TABLE chunks_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[{dims}]
        )
    """)

    # Seed three chunks with known normalized vectors
    # [1,0,0,0] = "identical to query"
    # [0,1,0,0] = "orthogonal"
    # [-1,0,0,0] = "opposite"
    vecs = {
        "id1": normalize_embedding([1.0, 0.0, 0.0, 0.0]),
        "id2": normalize_embedding([0.0, 1.0, 0.0, 0.0]),
        "id3": normalize_embedding([-1.0, 0.0, 0.0, 0.0]),
    }
    for chunk_id, vec in vecs.items():
        await db.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                chunk_id,
                f"memory/{chunk_id}.md",
                "memory",
                "test-model",
                1,
                3,
                f"text for {chunk_id}",
            ),
        )
        await db.execute(
            "INSERT INTO chunks_vec(id, embedding) VALUES (?, ?)",
            (chunk_id, _vec_to_blob(vec)),
        )
    await db.commit()
    return db


@requires_sqlite_vec
class TestVectorSearch:
    async def test_identical_vector_is_first(self):
        db = await _make_vec_db()
        vs = VectorSearch()
        query = normalize_embedding([1.0, 0.0, 0.0, 0.0])
        rows = await vs.search(db, "", query, "test-model", limit=3)
        assert rows[0].chunk_id == "id1"
        assert rows[0].score > 0.99
        await db.close()

    async def test_orthogonal_vector_has_mid_score(self):
        db = await _make_vec_db()
        vs = VectorSearch()
        query = normalize_embedding([1.0, 0.0, 0.0, 0.0])
        rows = await vs.search(db, "", query, "test-model", limit=3)
        id2_row = next(r for r in rows if r.chunk_id == "id2")
        # cos([1,0,0,0], [0,1,0,0]) = 0  → dist=1 → score=0
        assert abs(id2_row.score) < 0.05
        await db.close()

    async def test_results_ordered_by_score_descending(self):
        db = await _make_vec_db()
        vs = VectorSearch()
        query = normalize_embedding([1.0, 0.0, 0.0, 0.0])
        rows = await vs.search(db, "", query, "test-model", limit=3)
        scores = [r.score for r in rows]
        assert scores == sorted(scores, reverse=True)
        await db.close()

    async def test_limit_respected(self):
        db = await _make_vec_db()
        vs = VectorSearch()
        query = normalize_embedding([1.0, 0.0, 0.0, 0.0])
        rows = await vs.search(db, "", query, "test-model", limit=1)
        assert len(rows) == 1
        await db.close()

    async def test_vector_score_populated(self):
        db = await _make_vec_db()
        vs = VectorSearch()
        query = normalize_embedding([1.0, 0.0, 0.0, 0.0])
        rows = await vs.search(db, "", query, "test-model", limit=3)
        for row in rows:
            assert row.vector_score is not None
            assert row.vector_score == row.score
        await db.close()

    async def test_source_filter(self):
        """Source filter restricts results to the specified source."""
        import sqlite_vec

        db = await aiosqlite.connect(":memory:")
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)

        await db.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY, path TEXT, source TEXT,
                model TEXT, start_line INTEGER, end_line INTEGER, text TEXT
            )
        """)
        await db.execute(
            "CREATE VIRTUAL TABLE chunks_vec USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[2])"
        )
        await db.execute("INSERT INTO chunks VALUES ('a','p','memory','m',1,1,'t')")
        await db.execute("INSERT INTO chunks VALUES ('b','p','sessions','m',1,1,'t')")
        vec = normalize_embedding([1.0, 0.0])
        await db.execute("INSERT INTO chunks_vec VALUES ('a', ?)", (_vec_to_blob(vec),))
        await db.execute("INSERT INTO chunks_vec VALUES ('b', ?)", (_vec_to_blob(vec),))
        await db.commit()

        vs = VectorSearch()
        rows = await vs.search(db, "", vec, "m", limit=10, source_filter="memory")
        assert len(rows) == 1
        assert rows[0].chunk_id == "a"
        await db.close()
