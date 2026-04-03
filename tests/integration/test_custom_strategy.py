"""
tests/integration/test_custom_strategy.py — Happy-path: register_strategy() extensibility.

Covers:
- A custom SearchStrategy registered with register_strategy() is invoked by search().
- The custom result's path and snippet are returned correctly.
- An unknown strategy name raises SearchError.
- Built-in strategies (hybrid) still work after a custom one is registered.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig, SearchError  # noqa: E402
from memweave.search.strategy import RawSearchRow  # noqa: E402

pytestmark = pytest.mark.integration

_SENTINEL = "CUSTOM_STRATEGY_RESULT"


class FixedResultStrategy:
    """Always returns a single hardcoded result, regardless of query."""

    async def search(
        self,
        db: aiosqlite.Connection,
        query: str,
        query_vec: list[float] | None,
        model: str,
        limit: int,
        *,
        source_filter: str | None = None,
    ) -> list[RawSearchRow]:
        return [
            RawSearchRow(
                chunk_id="custom-001",
                path="custom/injected.md",
                source="custom",
                start_line=1,
                end_line=1,
                text=_SENTINEL,
                score=0.99,
                vector_score=None,
                text_score=None,
            )
        ]


@pytest.mark.asyncio
async def test_custom_strategy(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / "2026-04-01.md").write_text(
        "The project uses a hexagonal architecture for the backend services.\n"
        "Domain logic is isolated from infrastructure concerns via ports and adapters.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        mem.register_strategy("my-custom", FixedResultStrategy())

        results = await mem.search("anything", strategy="my-custom", min_score=0.0)
        assert len(results) == 1, f"Expected 1 result from custom strategy, got {len(results)}"
        assert results[0].snippet == _SENTINEL, (
            f"Expected sentinel text, got: {results[0].snippet!r}"
        )
        assert results[0].path == "custom/injected.md"

        # Unknown strategy → SearchError
        with pytest.raises(SearchError):
            await mem.search("test", strategy="does-not-exist")

        # Built-in hybrid still works after custom registration
        r_hybrid = await mem.search("hexagonal architecture", strategy="hybrid", min_score=0.0)
        assert len(r_hybrid) > 0, "hybrid strategy broken after custom strategy registration"
