"""
tests/unit/test_hashing.py — Unit tests for memweave/_internal/hashing.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from memweave._internal.hashing import (
    batched,
    make_chunk_id,
    make_provider_key,
    normalize_path,
    run_with_concurrency,
    sha256_bytes,
    sha256_file,
    sha256_text,
    truncate_snippet,
)


class TestSha256Text:
    def test_deterministic(self):
        assert sha256_text("hello") == sha256_text("hello")

    def test_different_inputs(self):
        assert sha256_text("hello") != sha256_text("world")

    def test_returns_hex_string(self):
        result = sha256_text("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        result = sha256_text("")
        assert len(result) == 64


class TestSha256File:
    def test_consistent(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert sha256_file(f) == sha256_file(f)

    def test_changes_on_edit(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("original")
        hash1 = sha256_file(f)
        f.write_text("modified")
        hash2 = sha256_file(f)
        assert hash1 != hash2

    def test_matches_text_hash(self, tmp_path):
        content = "test content"
        f = tmp_path / "test.txt"
        f.write_bytes(content.encode("utf-8"))
        assert sha256_file(f) == sha256_bytes(content.encode("utf-8"))


class TestSha256Bytes:
    def test_deterministic(self):
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")

    def test_different(self):
        assert sha256_bytes(b"a") != sha256_bytes(b"b")


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = make_chunk_id("memory", "memory/2026-03-21.md", 1, 5, "abc", "model-1")
        id2 = make_chunk_id("memory", "memory/2026-03-21.md", 1, 5, "abc", "model-1")
        assert id1 == id2

    def test_different_paths(self):
        id1 = make_chunk_id("memory", "memory/a.md", 1, 5, "abc", "model")
        id2 = make_chunk_id("memory", "memory/b.md", 1, 5, "abc", "model")
        assert id1 != id2

    def test_different_lines(self):
        id1 = make_chunk_id("memory", "memory/a.md", 1, 5, "abc", "model")
        id2 = make_chunk_id("memory", "memory/a.md", 6, 10, "abc", "model")
        assert id1 != id2

    def test_different_model(self):
        id1 = make_chunk_id("memory", "memory/a.md", 1, 5, "abc", "model-a")
        id2 = make_chunk_id("memory", "memory/a.md", 1, 5, "abc", "model-b")
        assert id1 != id2

    def test_returns_hex(self):
        chunk_id = make_chunk_id("memory", "path.md", 1, 2, "hash", "model")
        assert len(chunk_id) == 64


class TestMakeProviderKey:
    def test_same_config(self):
        k1 = make_provider_key("openai", "text-embedding-3-small", None)
        k2 = make_provider_key("openai", "text-embedding-3-small", None)
        assert k1 == k2

    def test_different_model(self):
        k1 = make_provider_key("openai", "text-embedding-3-small", None)
        k2 = make_provider_key("openai", "text-embedding-3-large", None)
        assert k1 != k2

    def test_different_api_base(self):
        k1 = make_provider_key("openai", "text-embedding-3-small", None)
        k2 = make_provider_key("openai", "text-embedding-3-small", "http://localhost:11434")
        assert k1 != k2

    def test_none_api_base_produces_same_key(self):
        """Two calls with api_base=None must be stable."""
        k1 = make_provider_key("litellm", "text-embedding-3-small", None)
        k2 = make_provider_key("litellm", "text-embedding-3-small", None)
        assert k1 == k2


class TestRunWithConcurrency:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(self):
        results = []

        async def task(i: int) -> int:
            results.append(i)
            return i

        output = await run_with_concurrency([lambda i=i: task(i) for i in range(5)], max_concurrent=2)
        assert sorted(output) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        async def task(i: int) -> int:
            return i * 2

        output = await run_with_concurrency([lambda i=i: task(i) for i in range(5)], max_concurrent=3)
        assert output == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        output = await run_with_concurrency([], max_concurrent=4)
        assert output == []

    @pytest.mark.asyncio
    async def test_single_task(self):
        async def task() -> str:
            return "done"

        output = await run_with_concurrency([task], max_concurrent=1)
        assert output == ["done"]


class TestNormalizePath:
    def test_absolute_path_under_base(self):
        base = Path("/project")
        result = normalize_path(Path("/project/memory/2026-03-21.md"), base)
        assert result == "memory/2026-03-21.md"

    def test_relative_path_passthrough(self):
        base = Path("/project")
        result = normalize_path("memory/2026-03-21.md", base)
        assert result == "memory/2026-03-21.md"

    def test_path_outside_base(self):
        base = Path("/project")
        result = normalize_path(Path("/other/path.md"), base)
        assert result == "/other/path.md"


class TestTruncateSnippet:
    def test_no_truncation_needed(self):
        text = "Short text."
        assert truncate_snippet(text, 100) == text

    def test_truncates_at_word_boundary(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = truncate_snippet(text, 20)
        assert len(result) <= 21  # +1 for ellipsis char
        assert result.endswith("…")

    def test_truncates_hard_if_no_space(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        result = truncate_snippet(text, 10)
        assert result.endswith("…")

    def test_exact_length(self):
        text = "exact"
        assert truncate_snippet(text, 5) == text


class TestBatched:
    def test_even_split(self):
        result = batched([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = batched([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty(self):
        assert batched([], 3) == []

    def test_batch_larger_than_list(self):
        assert batched([1, 2], 10) == [[1, 2]]
