"""
tests/unit/test_config.py — Unit tests for memweave/config.py

Tests cover: defaults, validation constraints, path resolution, serialization.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memweave.config import (
    CacheConfig,
    ChunkingConfig,
    EmbeddingConfig,
    FlushConfig,
    HybridConfig,
    MemoryConfig,
    MMRConfig,
    QueryConfig,
    SyncConfig,
    TemporalDecayConfig,
    VectorConfig,
)


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == "text-embedding-3-small"
        assert cfg.timeout == 60.0
        assert cfg.batch_size == 64
        assert cfg.api_key is None
        assert cfg.api_base is None

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout"):
            EmbeddingConfig(timeout=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingConfig(batch_size=0)

    def test_valid_custom(self):
        cfg = EmbeddingConfig(
            model="text-embedding-3-large",
            timeout=30.0,
            batch_size=128,
        )
        assert cfg.model == "text-embedding-3-large"


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.tokens == 400
        assert cfg.overlap == 80

    def test_max_chars(self):
        cfg = ChunkingConfig(tokens=400)
        assert cfg.max_chars == 1600  # max(32, 400 * 4)

    def test_max_chars_minimum(self):
        cfg = ChunkingConfig(tokens=5, overlap=0)
        assert cfg.max_chars == 32  # max(32, 5*4=20) = 32

    def test_overlap_chars(self):
        cfg = ChunkingConfig(tokens=400, overlap=80)
        assert cfg.overlap_chars == 320  # 80 * 4

    def test_invalid_tokens(self):
        with pytest.raises(ValueError, match="tokens"):
            ChunkingConfig(tokens=0)

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="overlap"):
            ChunkingConfig(overlap=-1)

    def test_overlap_must_be_less_than_tokens(self):
        with pytest.raises(ValueError, match="overlap"):
            ChunkingConfig(tokens=100, overlap=100)


class TestHybridConfig:
    def test_defaults(self):
        cfg = HybridConfig()
        assert cfg.vector_weight == 0.7
        assert cfg.text_weight == 0.3
        assert cfg.candidate_multiplier == 4

    def test_weights_sum_to_one(self):
        # Valid
        cfg = HybridConfig(vector_weight=0.6, text_weight=0.4)
        assert cfg.vector_weight == 0.6

    def test_weights_dont_sum_to_one(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            HybridConfig(vector_weight=0.5, text_weight=0.4)

    def test_invalid_vector_weight(self):
        with pytest.raises(ValueError, match="vector_weight"):
            HybridConfig(vector_weight=1.5, text_weight=-0.5)

    def test_invalid_candidate_multiplier(self):
        with pytest.raises(ValueError, match="candidate_multiplier"):
            HybridConfig(candidate_multiplier=0)


class TestMMRConfig:
    def test_defaults(self):
        cfg = MMRConfig()
        assert cfg.enabled is False
        assert cfg.lambda_param == 0.7

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_param"):
            MMRConfig(lambda_param=1.5)


class TestTemporalDecayConfig:
    def test_defaults(self):
        cfg = TemporalDecayConfig()
        assert cfg.enabled is False
        assert cfg.half_life_days == 30.0

    def test_invalid_half_life(self):
        with pytest.raises(ValueError, match="half_life_days"):
            TemporalDecayConfig(half_life_days=0)


class TestQueryConfig:
    def test_defaults(self):
        cfg = QueryConfig()
        assert cfg.strategy == "hybrid"
        assert cfg.max_results == 6
        assert cfg.min_score == 0.35
        assert cfg.snippet_max_chars == 700

    def test_invalid_max_results(self):
        with pytest.raises(ValueError, match="max_results"):
            QueryConfig(max_results=0)

    def test_invalid_min_score(self):
        with pytest.raises(ValueError, match="min_score"):
            QueryConfig(min_score=1.5)

    def test_invalid_snippet_max_chars(self):
        with pytest.raises(ValueError, match="snippet_max_chars"):
            QueryConfig(snippet_max_chars=0)


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.enabled is True
        assert cfg.max_entries is None

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=0)

    def test_none_max_entries_ok(self):
        cfg = CacheConfig(max_entries=None)
        assert cfg.max_entries is None


class TestFlushConfig:
    def test_defaults(self):
        cfg = FlushConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            FlushConfig(max_tokens=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            FlushConfig(temperature=3.0)


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig(workspace_dir="/tmp/test_project")
        assert isinstance(cfg.workspace_dir, Path)
        assert cfg.workspace_dir == Path("/tmp/test_project")

    def test_resolved_db_path_default(self):
        cfg = MemoryConfig(workspace_dir="/tmp/project")
        assert cfg.resolved_db_path == Path("/tmp/project/.memweave/index.sqlite")

    def test_resolved_db_path_explicit(self):
        cfg = MemoryConfig(workspace_dir="/tmp/project", db_path="/tmp/custom.sqlite")
        assert cfg.resolved_db_path == Path("/tmp/custom.sqlite")

    def test_memory_dir(self):
        cfg = MemoryConfig(workspace_dir="/tmp/project")
        assert cfg.memory_dir == Path("/tmp/project/memory")

    def test_tilde_expansion(self):
        cfg = MemoryConfig(workspace_dir="~/my_project")
        # Should not contain literal ~
        assert "~" not in str(cfg.workspace_dir)

    def test_to_dict_roundtrip(self):
        cfg = MemoryConfig(workspace_dir="/tmp/project")
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "embedding" in d
        assert "chunking" in d
        assert "query" in d

    def test_evergreen_patterns_default(self):
        cfg = MemoryConfig(workspace_dir="/tmp/project")
        assert "MEMORY.md" in cfg.evergreen_patterns

    def test_bootstrap_files_default(self):
        cfg = MemoryConfig(workspace_dir="/tmp/project")
        assert "MEMORY.md" in cfg.bootstrap_files
