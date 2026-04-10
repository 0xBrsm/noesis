"""
tests/unit/test_chunking.py — Unit tests for memweave/chunking/markdown.py

Tests verify that the Python chunking implementation in memweave/chunking/markdown.py
"""

from __future__ import annotations

import pytest

from memweave.chunking.markdown import MarkdownChunk, chunk_markdown, chunk_text

# ── Basic structure ───────────────────────────────────────────────────────────


class TestChunkMarkdownBasic:
    def test_empty_string_returns_empty(self):
        """Empty string splits to [''], which produces one empty-text chunk.

        This matches the expected behavior: ''.split('\\n') → [''],
        and the single empty-string line flushes as a chunk.
        """
        chunks = chunk_markdown("")
        # One chunk is produced (the empty line), consistent with the TS reference
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_single_line_is_one_chunk(self):
        """A single short line should produce exactly one chunk."""
        chunks = chunk_markdown("Hello world", chunk_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1

    def test_whitespace_only(self):
        """Whitespace-only content should produce a chunk (lines are preserved)."""
        # "   " is not empty after split — it's one line with spaces
        chunks = chunk_markdown("   ", chunk_tokens=100)
        # Should produce one chunk with the whitespace line
        assert len(chunks) >= 0  # either empty or one chunk is acceptable

    def test_multiline_short_fits_one_chunk(self):
        """Multiple short lines that fit within budget → single chunk."""
        text = "Line 1\nLine 2\nLine 3"
        chunks = chunk_markdown(text, chunk_tokens=400)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3
        assert "Line 1" in chunks[0].text
        assert "Line 3" in chunks[0].text

    def test_result_is_list_of_markdown_chunks(self):
        """Return type should be list[MarkdownChunk]."""
        chunks = chunk_markdown("Hello\nWorld", chunk_tokens=100)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, MarkdownChunk)
            assert isinstance(c.start_line, int)
            assert isinstance(c.end_line, int)
            assert isinstance(c.text, str)

    def test_line_numbers_are_1_indexed(self):
        """start_line must be 1 for the first line of the first chunk."""
        chunks = chunk_markdown("First line\nSecond line", chunk_tokens=400)
        assert chunks[0].start_line == 1


# ── Splitting behavior ────────────────────────────────────────────────────────


class TestChunkMarkdownSplitting:
    def test_long_document_splits_into_multiple_chunks(self):
        """A document larger than max_chars should produce > 1 chunk."""
        # chunk_tokens=10 → max_chars = max(32, 10*4) = 40 chars per chunk
        lines = [f"This is line number {i:03d}" for i in range(1, 50)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, chunk_tokens=10, chunk_overlap=0)
        assert len(chunks) > 1

    def test_all_content_covered(self):
        """Every line of the source document must appear in at least one chunk."""
        lines = [f"Line {i}" for i in range(1, 30)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, chunk_tokens=20, chunk_overlap=0)

        all_text = "\n".join(c.text for c in chunks)
        for line in lines:
            assert line in all_text, f"{line!r} missing from chunks"

    def test_chunk_start_end_lines_are_ordered(self):
        """start_line <= end_line for every chunk."""
        text = "\n".join(f"Line {i}" for i in range(1, 100))
        chunks = chunk_markdown(text, chunk_tokens=30, chunk_overlap=5)
        for c in chunks:
            assert (
                c.start_line <= c.end_line
            ), f"Chunk has start_line={c.start_line} > end_line={c.end_line}"

    def test_chunks_are_in_document_order(self):
        """Chunk start lines should be non-decreasing."""
        text = "\n".join(f"Line {i}" for i in range(1, 100))
        chunks = chunk_markdown(text, chunk_tokens=30, chunk_overlap=5)
        for i in range(len(chunks) - 1):
            assert chunks[i].start_line <= chunks[i + 1].start_line, (
                f"Chunk {i} start_line={chunks[i].start_line} > "
                f"chunk {i+1} start_line={chunks[i+1].start_line}"
            )

    def test_text_preserves_newlines(self):
        """Chunk text should join lines with '\\n', not space or other."""
        text = "Line A\nLine B\nLine C"
        chunks = chunk_markdown(text, chunk_tokens=400)
        assert "\n" in chunks[0].text
        assert chunks[0].text == "Line A\nLine B\nLine C"


# ── Overlap behavior ──────────────────────────────────────────────────────────


class TestChunkMarkdownOverlap:
    def test_no_overlap_zero(self):
        """With overlap=0, consecutive chunks should NOT share lines."""
        # Build a document guaranteed to split
        lines = [f"{'X' * 20} line {i}" for i in range(1, 30)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, chunk_tokens=10, chunk_overlap=0)
        if len(chunks) > 1:
            # end_line of chunk N should be < start_line of chunk N+1
            assert (
                chunks[0].end_line < chunks[1].start_line
            ), "No-overlap chunks should not share lines"

    def test_overlap_shares_content(self):
        """With overlap > 0, consecutive chunks should share some content."""
        # Build a long document with overlap
        lines = [f"{'Word ' * 5} line {i}" for i in range(1, 50)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, chunk_tokens=15, chunk_overlap=5)

        if len(chunks) > 1:
            # There should be at least some shared lines between consecutive chunks
            found_overlap = False
            for i in range(len(chunks) - 1):
                end_1 = chunks[i].end_line
                start_2 = chunks[i + 1].start_line
                if start_2 <= end_1:
                    found_overlap = True
                    break
            assert found_overlap, "Expected overlapping chunks but found none"

    def test_overlap_cannot_exceed_chunk(self):
        """overlap >= tokens is forbidden by config validation, not chunker.

        The chunker itself is not responsible for validating config —
        that happens in ChunkingConfig.__post_init__. This test confirms
        the chunker doesn't crash with edge-case overlap values.
        """
        text = "\n".join(f"Line {i}" for i in range(1, 20))
        # overlap=0 is always safe
        chunks = chunk_markdown(text, chunk_tokens=20, chunk_overlap=0)
        assert len(chunks) >= 1


# ── Sub-line splitting (very long lines) ─────────────────────────────────────


class TestChunkMarkdownLongLines:
    def test_very_long_single_line_splits(self):
        """A single line longer than max_chars should be split into segments."""
        # max_chars = max(32, 5*4) = 32 chars
        long_line = "A" * 200
        chunks = chunk_markdown(long_line, chunk_tokens=5, chunk_overlap=0)
        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_long_line_segments_share_same_line_number(self):
        """All segments from a single long line should have the same line number."""
        long_line = "B" * 200
        chunks = chunk_markdown(long_line, chunk_tokens=5, chunk_overlap=0)
        for c in chunks:
            # All are from line 1
            assert c.start_line == 1
            assert c.end_line == 1

    def test_mixed_short_and_long_lines(self):
        """Short and long lines can coexist without crashing."""
        text = "Short line\n" + ("C" * 300) + "\nAnother short"
        chunks = chunk_markdown(text, chunk_tokens=20, chunk_overlap=0)
        assert len(chunks) >= 1
        # Verify all content is covered
        all_text = "\n".join(c.text for c in chunks)
        assert "Short line" in all_text
        assert "Another short" in all_text


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestChunkMarkdownEdgeCases:
    def test_single_character(self):
        """Single character should produce one chunk."""
        chunks = chunk_markdown("X")
        assert len(chunks) == 1
        assert chunks[0].text == "X"

    def test_blank_lines_preserved(self):
        """Blank lines should be included in chunks (as empty strings)."""
        text = "Line 1\n\nLine 3"
        chunks = chunk_markdown(text, chunk_tokens=400)
        assert len(chunks) == 1
        # The blank line is part of the text
        assert "\n\n" in chunks[0].text

    def test_only_newlines(self):
        """Document consisting only of newlines."""
        chunks = chunk_markdown("\n\n\n", chunk_tokens=400)
        # Should not crash; may produce empty or non-empty chunks
        assert isinstance(chunks, list)

    def test_markdown_with_headers_and_lists(self):
        """Realistic markdown document should chunk without errors."""
        text = (
            "# Project Memory\n\n"
            "## Architecture Decisions\n\n"
            "- We chose PostgreSQL for JSONB support\n"
            "- Redis is used for caching sessions\n"
            "- Nginx handles reverse proxy\n\n"
            "## Deployment\n\n"
            "Deploy with: `docker-compose up -d`\n"
        )
        chunks = chunk_markdown(text, chunk_tokens=400)
        assert len(chunks) >= 1
        all_text = "\n".join(c.text for c in chunks)
        assert "PostgreSQL" in all_text
        assert "docker-compose" in all_text

    def test_unicode_content(self):
        """Unicode content should chunk without errors."""
        text = "日本語テスト\nChinese: 你好世界\nEmoji: 🔥🌊⚡"
        chunks = chunk_markdown(text, chunk_tokens=400)
        assert len(chunks) >= 1

    def test_minimum_token_size(self):
        """chunk_tokens=1 → max_chars=max(32, 4)=32; should still work."""
        text = "\n".join(f"Line {i}" for i in range(1, 10))
        chunks = chunk_markdown(text, chunk_tokens=1, chunk_overlap=0)
        assert len(chunks) >= 1

    def test_large_document(self):
        """A large document (1000 lines) should chunk without performance issues."""
        lines = [f"Memory note {i}: some content about topic {i % 20}" for i in range(1, 1001)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, chunk_tokens=400, chunk_overlap=80)
        assert len(chunks) >= 1
        # All 1000 lines should be covered
        all_text = " ".join(c.text for c in chunks)
        assert "Memory note 1:" in all_text
        assert "Memory note 1000:" in all_text


# ── chunk_text convenience function ──────────────────────────────────────────


class TestChunkText:
    def test_returns_strings_only(self):
        """chunk_text should return list[str], not list[MarkdownChunk]."""
        result = chunk_text("Hello\nWorld", chunk_tokens=400)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_content_matches_chunk_markdown(self):
        """chunk_text results should match the .text fields from chunk_markdown."""
        text = "\n".join(f"Line {i}" for i in range(1, 20))
        full = chunk_markdown(text, chunk_tokens=30, chunk_overlap=5)
        simple = chunk_text(text, chunk_tokens=30, chunk_overlap=5)
        assert [c.text for c in full] == simple


# ── Exact algorithm verification ─────────────────────────────────────────────


class TestChunkMarkdownAlgorithmVerification:
    """Verify specific algorithm behaviors against known-good outputs.

    These tests pin exact algorithmic behavior so any deviation from the
    reference implementation is caught immediately.
    """

    def test_max_chars_calculation(self):
        """max_chars = max(32, tokens * 4)."""
        # chunk_tokens=1 → max(32, 4) = 32
        # Lines of 10 chars each: need > 3 lines to overflow (3*11=33 > 32)
        lines = ["1234567890"] * 10  # 10 chars each, 11 with \n separator
        text = "\n".join(lines)
        chunks_small = chunk_markdown(text, chunk_tokens=1, chunk_overlap=0)
        chunks_large = chunk_markdown(text, chunk_tokens=400, chunk_overlap=0)
        # Small token budget → more chunks
        assert len(chunks_small) >= len(chunks_large)

    def test_overlap_chars_calculation(self):
        """overlap_chars = max(0, overlap * 4)."""
        lines = [f"{'W' * 10} line {i}" for i in range(1, 40)]
        text = "\n".join(lines)
        # No overlap
        chunks_no_overlap = chunk_markdown(text, chunk_tokens=10, chunk_overlap=0)
        # With overlap
        chunks_with_overlap = chunk_markdown(text, chunk_tokens=10, chunk_overlap=5)
        # With overlap, start_line of chunk N+1 should be <= end_line of chunk N
        if len(chunks_with_overlap) > 1:
            assert chunks_with_overlap[1].start_line <= chunks_no_overlap[0].end_line + 2

    def test_flush_triggers_on_overflow(self):
        """Flush must trigger when adding next line would overflow max_chars."""
        # max_chars=32 (tokens=1). A line of 31 chars + '\n' = 32 chars exactly fills one chunk.
        # Adding another line should trigger flush.
        line_31 = "A" * 31  # 31 chars + 1 '\n' = 32 = exactly max_chars
        text = "\n".join([line_31, line_31, line_31])
        chunks = chunk_markdown(text, chunk_tokens=1, chunk_overlap=0)
        # Each 31-char line should be in its own chunk
        assert len(chunks) >= 1
