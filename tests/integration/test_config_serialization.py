"""
tests/integration/test_config_serialization.py — Config round-trip serialization.

Covers:
- config.to_dict() produces a JSON-serializable dict.
- MemoryConfig.from_dict(config.to_dict()) produces an equivalent config.
- Non-default values (MMR, decay, hybrid weights, cache, flush, etc.) survive round-trip.
- from_dict() with missing fields applies dataclass defaults.

No live API calls required — purely tests config serialization logic.
"""

from __future__ import annotations

import json

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import (  # noqa: E402
    CacheConfig,
    ChunkingConfig,
    EmbeddingConfig,
    FlushConfig,
    HybridConfig,
    MemoryConfig,
    MMRConfig,
    QueryConfig,
    TemporalDecayConfig,
    VectorConfig,
)
from memweave.config import SyncConfig  # noqa: E402

pytestmark = pytest.mark.integration


def test_default_config_round_trip(tmp_path: Path) -> None:
    cfg = MemoryConfig(workspace_dir=tmp_path)
    json_str = json.dumps(cfg.to_dict())
    assert json_str, "to_dict() must produce non-empty JSON-serializable output"

    restored = MemoryConfig.from_dict(json.loads(json_str))
    assert cfg.to_dict() == restored.to_dict(), "Default config round-trip failed"


def test_custom_config_round_trip(tmp_path: Path) -> None:
    cfg = MemoryConfig(
        workspace_dir=tmp_path,
        timezone="America/New_York",
        extra_paths=["docs", "notes"],
        evergreen_patterns=["MEMORY.md", "REFERENCE.md"],
        embedding=EmbeddingConfig(model="text-embedding-3-large", batch_size=32, timeout=30.0),
        chunking=ChunkingConfig(tokens=200, overlap=40),
        query=QueryConfig(
            strategy="keyword",
            max_results=12,
            min_score=0.25,
            snippet_max_chars=500,
            hybrid=HybridConfig(vector_weight=0.8, text_weight=0.2),
            mmr=MMRConfig(enabled=True, lambda_param=0.4),
            temporal_decay=TemporalDecayConfig(enabled=True, half_life_days=14.0),
        ),
        cache=CacheConfig(enabled=True, max_entries=500),
        sync=SyncConfig(on_search=False, watch_debounce_ms=2000),
        flush=FlushConfig(enabled=True, model="gpt-4o", max_tokens=2048, temperature=0.1),
        vector=VectorConfig(enabled=True),
    )

    restored = MemoryConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert cfg.to_dict() == restored.to_dict(), "Custom config round-trip lost values"

    # Spot-check non-default values
    assert restored.query.mmr.enabled is True
    assert restored.query.mmr.lambda_param == 0.4
    assert restored.query.temporal_decay.half_life_days == 14.0
    assert restored.query.hybrid.vector_weight == 0.8
    assert restored.cache.max_entries == 500
    assert restored.timezone == "America/New_York"
    assert restored.extra_paths == ["docs", "notes"]
    assert "REFERENCE.md" in restored.evergreen_patterns


def test_partial_dict_uses_defaults(tmp_path: Path) -> None:
    minimal = MemoryConfig.from_dict({"workspace_dir": str(tmp_path)})
    assert minimal.query.max_results == 6, "Missing field should use dataclass default"
    assert minimal.cache.enabled is True
