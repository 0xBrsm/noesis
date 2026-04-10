"""
tests/unit/test_vectors.py — Unit tests for memweave/embedding/vectors.py
"""

from __future__ import annotations

import math

from memweave.embedding.vectors import normalize_embedding

# ── normalize_embedding ───────────────────────────────────────────────────────


class TestNormalizeEmbedding:
    def test_unit_vector_unchanged(self):
        """A vector already on the unit sphere should be unchanged (within float error)."""
        v = [1.0, 0.0, 0.0]
        result = normalize_embedding(v)
        assert abs(math.sqrt(sum(x * x for x in result)) - 1.0) < 1e-9

    def test_3_4_5_triangle(self):
        """Classic 3-4-5 right triangle gives [0.6, 0.8]."""
        result = normalize_embedding([3.0, 4.0])
        assert abs(result[0] - 0.6) < 1e-9
        assert abs(result[1] - 0.8) < 1e-9

    def test_unit_length_after_normalization(self):
        """Any non-zero vector should have norm 1.0 after normalization."""
        v = [1.0, 2.0, 3.0, 4.0]
        result = normalize_embedding(v)
        assert abs(math.sqrt(sum(x * x for x in result)) - 1.0) < 1e-9

    def test_zero_vector_returned_unchanged(self):
        """Zero vector has no direction; returned as-is (no div-by-zero)."""
        v = [0.0, 0.0, 0.0]
        result = normalize_embedding(v)
        assert result == [0.0, 0.0, 0.0]

    def test_single_element(self):
        """Single-element vector normalizes to [1.0] or [-1.0]."""
        result = normalize_embedding([5.0])
        assert abs(result[0] - 1.0) < 1e-9

    def test_negative_values(self):
        """Negative values should normalize correctly."""
        v = [-3.0, -4.0]
        result = normalize_embedding(v)
        assert abs(result[0] - (-0.6)) < 1e-9
        assert abs(result[1] - (-0.8)) < 1e-9
        assert abs(math.sqrt(sum(x * x for x in result)) - 1.0) < 1e-9

    def test_large_vector(self):
        """1536-dimensional vector should normalize cleanly."""
        import random

        rng = random.Random(42)
        v = [rng.gauss(0, 1) for _ in range(1536)]
        result = normalize_embedding(v)
        assert abs(math.sqrt(sum(x * x for x in result)) - 1.0) < 1e-6

    def test_returns_new_list(self):
        """Should return a new list, not mutate the input."""
        v = [3.0, 4.0]
        original = v[:]
        normalize_embedding(v)
        assert v == original
