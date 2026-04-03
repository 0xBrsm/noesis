"""
tests/integration/test_evergreen_files.py — Happy-path: MEMORY.md and non-dated files exempt from decay.

Covers:
- MEMORY.md inside memory/ has is_evergreen=True in FileInfo.
- Non-dated files under memory/ (e.g. architecture.md) also have is_evergreen=True.
- Dated files have is_evergreen=False.
- With aggressive decay, MEMORY.md retains its score while a 60-day-old dated
  file with identical content is heavily penalised.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig, QueryConfig  # noqa: E402

pytestmark = pytest.mark.integration

_OLD_DATE = "2026-02-01"  # ~60 days before 2026-04-01
_SHARED_CONTENT = (
    "The primary API gateway handles authentication and rate limiting.\n"
    "JWT tokens expire after 24 hours and are rotated on every refresh.\n"
)


@pytest.mark.asyncio
async def test_evergreen_files(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / "MEMORY.md").write_text(_SHARED_CONTENT)
    (mem_dir / "architecture.md").write_text(_SHARED_CONTENT)
    (mem_dir / f"{_OLD_DATE}.md").write_text(_SHARED_CONTENT)

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0, max_results=10),
    )

    async with MemWeave(config) as mem:
        await mem.index()

        file_list = await mem.files()

        memory_md = next((f for f in file_list if f.path == "memory/MEMORY.md"), None)
        arch_file = next((f for f in file_list if "architecture" in f.path), None)
        old_file = next((f for f in file_list if _OLD_DATE in f.path), None)

        assert memory_md is not None, "MEMORY.md not found"
        assert memory_md.is_evergreen, "MEMORY.md must be is_evergreen=True"

        assert arch_file is not None, "architecture.md not found"
        assert arch_file.is_evergreen, "architecture.md must be is_evergreen=True"

        assert old_file is not None, f"{_OLD_DATE}.md not found"
        assert not old_file.is_evergreen, f"{_OLD_DATE}.md must be is_evergreen=False"

        # With half_life=1d, a 60-day-old file scores near zero;
        # MEMORY.md (evergreen) must retain its score — score >> old file
        query = "API gateway authentication JWT tokens"
        r_decay = await mem.search(query, min_score=0.0, decay_half_life_days=1.0)
        scores = {r.path: r.score for r in r_decay}

        memory_score = scores.get("memory/MEMORY.md", 0.0)
        old_score = scores.get(f"memory/{_OLD_DATE}.md", 0.0)

        assert memory_score > 0, "MEMORY.md should appear in results"
        if old_score > 0:
            assert memory_score > old_score * 10, (
                f"MEMORY.md ({memory_score:.4f}) should score >>10× the old dated file "
                f"({old_score:.6f}) with half_life=1d"
            )
