"""
tests/unit/test_embedding_cache.py — Unit tests for memweave/embedding/cache.py
"""

from __future__ import annotations

import pytest
import aiosqlite

from memweave._internal.hashing import make_provider_key, sha256_text
from memweave.embedding.cache import (
    evict_cache_if_needed,
    get_cached_embedding,
    get_cached_embeddings,
    merge_embeddings,
    split_into_hits_and_misses,
    store_embedding,
    store_embeddings_bulk,
)
from memweave.storage.schema import ensure_schema
from memweave.storage.sqlite_store import SQLiteStore

MODEL = "text-embedding-3-small"
PROVIDER_KEY = make_provider_key("litellm", MODEL, None)


@pytest.fixture
async def store():
    """Provide an in-memory SQLiteStore with schema applied."""
    async with aiosqlite.connect(":memory:") as db:
        await ensure_schema(db)
        yield SQLiteStore(db)


# ── get_cached_embeddings (bulk) ──────────────────────────────────────────────

class TestGetCachedEmbeddings:
    async def test_empty_hashes_returns_empty(self, store):
        result = await get_cached_embeddings(store, [], MODEL, PROVIDER_KEY)
        assert result == {}

    async def test_cache_miss_returns_empty(self, store):
        result = await get_cached_embeddings(
            store, ["nonexistent_hash"], MODEL, PROVIDER_KEY
        )
        assert result == {}

    async def test_cache_hit_returns_vector(self, store):
        text = "PostgreSQL was chosen for JSONB support."
        text_hash = sha256_text(text)
        vec = [0.1, 0.2, 0.3]
        await store_embedding(store, text_hash, vec, MODEL, PROVIDER_KEY)
        await store.commit()

        result = await get_cached_embeddings(store, [text_hash], MODEL, PROVIDER_KEY)
        assert text_hash in result
        assert result[text_hash] == vec

    async def test_partial_hit(self, store):
        """Some hashes cached, some not — only hits returned."""
        text1 = "First text"
        text2 = "Second text"
        hash1 = sha256_text(text1)
        hash2 = sha256_text(text2)

        # Only store text1
        await store_embedding(store, hash1, [1.0, 0.0], MODEL, PROVIDER_KEY)
        await store.commit()

        result = await get_cached_embeddings(store, [hash1, hash2], MODEL, PROVIDER_KEY)
        assert hash1 in result
        assert hash2 not in result

    async def test_multiple_hits(self, store):
        """Multiple cached entries all returned."""
        entries = {sha256_text(f"text{i}"): [float(i)] * 4 for i in range(5)}
        for h, v in entries.items():
            await store_embedding(store, h, v, MODEL, PROVIDER_KEY)
        await store.commit()

        result = await get_cached_embeddings(store, list(entries.keys()), MODEL, PROVIDER_KEY)
        assert len(result) == 5
        for h, v in entries.items():
            assert result[h] == v


# ── get_cached_embedding (single) ────────────────────────────────────────────

class TestGetCachedEmbedding:
    async def test_miss_returns_none(self, store):
        result = await get_cached_embedding(store, "unknown text", MODEL, PROVIDER_KEY)
        assert result is None

    async def test_hit_returns_vector(self, store):
        text = "Redis used for caching."
        text_hash = sha256_text(text)
        vec = [0.5, 0.6, 0.7]
        await store_embedding(store, text_hash, vec, MODEL, PROVIDER_KEY)
        await store.commit()

        result = await get_cached_embedding(store, text, MODEL, PROVIDER_KEY)
        assert result == vec


# ── store_embedding ───────────────────────────────────────────────────────────

class TestStoreEmbedding:
    async def test_store_and_retrieve(self, store):
        text_hash = sha256_text("test content")
        vec = [1.0, 2.0, 3.0]
        await store_embedding(store, text_hash, vec, MODEL, PROVIDER_KEY)
        await store.commit()

        result = await store.get_embedding(
            provider="litellm",
            model=MODEL,
            provider_key=PROVIDER_KEY,
            hash_=text_hash,
        )
        assert result == vec

    async def test_overwrite_existing(self, store):
        """Storing a new vector for the same hash should replace the old one."""
        text_hash = sha256_text("same content")
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]

        await store_embedding(store, text_hash, vec1, MODEL, PROVIDER_KEY)
        await store.commit()
        await store_embedding(store, text_hash, vec2, MODEL, PROVIDER_KEY)
        await store.commit()

        result = await store.get_embedding(
            provider="litellm", model=MODEL, provider_key=PROVIDER_KEY, hash_=text_hash
        )
        assert result == vec2

    async def test_dims_stored(self, store):
        """Dimensionality should match the vector length."""
        text_hash = sha256_text("dims test")
        vec = [0.1] * 1536
        await store_embedding(store, text_hash, vec, MODEL, PROVIDER_KEY)
        await store.commit()
        count = await store.count_cache_entries()
        assert count == 1


# ── store_embeddings_bulk ─────────────────────────────────────────────────────

class TestStoreEmbeddingsBulk:
    async def test_stores_all(self, store):
        data = {sha256_text(f"text{i}"): [float(i)] * 4 for i in range(10)}
        count = await store_embeddings_bulk(store, data, MODEL, PROVIDER_KEY)
        await store.commit()

        assert count == 10
        assert await store.count_cache_entries() == 10

    async def test_empty_dict_stores_nothing(self, store):
        count = await store_embeddings_bulk(store, {}, MODEL, PROVIDER_KEY)
        assert count == 0


# ── evict_cache_if_needed ─────────────────────────────────────────────────────

class TestEvictCacheIfNeeded:
    async def test_none_max_entries_skips_eviction(self, store):
        # Store 5 entries
        for i in range(5):
            await store_embedding(store, sha256_text(f"t{i}"), [float(i)], MODEL, PROVIDER_KEY)
        await store.commit()

        deleted = await evict_cache_if_needed(store, MODEL, max_entries=None)
        assert deleted == 0
        assert await store.count_cache_entries() == 5

    async def test_within_cap_skips_eviction(self, store):
        for i in range(3):
            await store_embedding(store, sha256_text(f"t{i}"), [float(i)], MODEL, PROVIDER_KEY)
        await store.commit()

        deleted = await evict_cache_if_needed(store, MODEL, max_entries=10)
        assert deleted == 0

    async def test_exceeds_cap_evicts_oldest(self, store):
        import asyncio
        for i in range(5):
            await store_embedding(store, sha256_text(f"t{i}"), [float(i)], MODEL, PROVIDER_KEY)
            await store.commit()
            await asyncio.sleep(0.01)  # ensure different updated_at timestamps

        deleted = await evict_cache_if_needed(store, MODEL, max_entries=3)
        await store.commit()

        assert deleted == 2
        assert await store.count_cache_entries() == 3


# ── split_into_hits_and_misses ────────────────────────────────────────────────

class TestSplitIntoHitsAndMisses:
    def test_all_misses(self):
        texts = ["text1", "text2"]
        cached = {}
        hits, misses = split_into_hits_and_misses(texts, cached)
        assert hits == {}
        assert [idx for idx, _ in misses] == [0, 1]
        assert [t for _, t in misses] == texts

    def test_all_hits(self):
        texts = ["text1", "text2"]
        cached = {
            sha256_text("text1"): [1.0, 0.0],
            sha256_text("text2"): [0.0, 1.0],
        }
        hits, misses = split_into_hits_and_misses(texts, cached)
        assert len(hits) == 2
        assert misses == []

    def test_partial_hits(self):
        texts = ["a", "b", "c"]
        cached = {sha256_text("b"): [0.5, 0.5]}
        hits, misses = split_into_hits_and_misses(texts, cached)
        assert list(hits.keys()) == [1]
        assert [idx for idx, _ in misses] == [0, 2]

    def test_order_preserved(self):
        texts = ["z", "a", "m"]
        hits, misses = split_into_hits_and_misses(texts, {})
        assert [idx for idx, _ in misses] == [0, 1, 2]

    def test_empty_texts(self):
        hits, misses = split_into_hits_and_misses([], {})
        assert hits == {}
        assert misses == []


# ── merge_embeddings ──────────────────────────────────────────────────────────

class TestMergeEmbeddings:
    def test_all_misses_no_hits(self):
        hits = {}
        misses = [(0, "a"), (1, "b"), (2, "c")]
        new_vecs = [[1.0], [2.0], [3.0]]
        result = merge_embeddings(hits, misses, new_vecs)
        assert result == [[1.0], [2.0], [3.0]]

    def test_all_hits_no_misses(self):
        hits = {0: [1.0], 1: [2.0], 2: [3.0]}
        result = merge_embeddings(hits, [], [])
        assert result == [[1.0], [2.0], [3.0]]

    def test_mixed_hits_and_misses(self):
        # indices 0, 2 are misses; index 1 is a hit
        hits = {1: [0.5, 0.5]}
        misses = [(0, "text_a"), (2, "text_c")]
        new_vecs = [[1.0, 0.0], [0.0, 1.0]]
        result = merge_embeddings(hits, misses, new_vecs)
        assert result[0] == [1.0, 0.0]  # miss at index 0
        assert result[1] == [0.5, 0.5]  # hit at index 1
        assert result[2] == [0.0, 1.0]  # miss at index 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="new_embeddings length"):
            merge_embeddings({}, [(0, "text")], [[1.0], [2.0]])  # 2 vecs for 1 miss

    def test_single_item_miss(self):
        result = merge_embeddings({}, [(0, "only")], [[9.0]])
        assert result == [[9.0]]

    def test_roundtrip_with_split(self):
        """split_into_hits_and_misses + merge_embeddings should reconstruct original order."""
        texts = ["apple", "banana", "cherry"]
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

        # Simulate: "banana" is cached
        cached = {sha256_text("banana"): vecs[1]}
        hits, misses = split_into_hits_and_misses(texts, cached)

        # "apple" and "cherry" are misses; simulate embedding them
        new_vecs = [vecs[0], vecs[2]]  # apple, cherry
        result = merge_embeddings(hits, misses, new_vecs)

        assert result[0] == vecs[0]  # apple
        assert result[1] == vecs[1]  # banana (from cache)
        assert result[2] == vecs[2]  # cherry
