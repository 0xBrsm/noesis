"""
tests/integration/test_search_vector.py — Happy-path: vector (semantic) search strategy.

Covers:
- strategy="vector" finds semantically relevant results for a paraphrased query
  with no exact term overlap with the stored content.
- vector_score is populated, text_score is None.
- Top result snippet contains at least one expected semantic term.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_search_vector(workspace: Path, embedding_model: str) -> None:
    (workspace / "memory" / "2026-04-01.md").write_text(
        "We selected React 18 with Next.js 14 as the frontend framework after evaluation.\n"
        "Vue 3 was considered but rejected due to a thinner hiring market and less mature SSR story.\n"
        "The decision was driven by existing team expertise and Next.js App Router maturity.\n"
    )

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        # Paraphrased query — no exact term overlap with stored content
        results = await mem.search(
            "UI framework selection decision", strategy="vector", min_score=0.01
        )

    assert len(results) > 0, "Expected semantic match for paraphrased query, got none"
    assert (
        results[0].vector_score is not None and results[0].vector_score > 0
    ), "vector_score must be populated for vector strategy"
    assert results[0].text_score is None, "text_score must be None for pure vector search"

    top_snippet = results[0].snippet.lower()
    relevant_terms = ["react", "vue", "next", "frontend", "framework"]
    assert any(
        t in top_snippet for t in relevant_terms
    ), f"Top result doesn't seem semantically relevant. Snippet: {results[0].snippet[:200]}"
