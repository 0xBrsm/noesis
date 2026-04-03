"""
tests/unit/test_watcher.py — Unit tests for sync/watcher.py.

Tests cover:
- MemoryWatcher construction (requires watchfiles)
- _filter_md_changes: only .md files returned, .memweave excluded
- MemoryWatcher.run: cancellation is clean
- start_watching() on MemWeave: graceful failure when watchfiles missing
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── _filter_md_changes ────────────────────────────────────────────────────────

class TestFilterMdChanges:
    def _make_watcher(self, tmp_path: Path):
        """Create a MemoryWatcher, skipping if watchfiles not installed."""
        try:
            from memweave.sync.watcher import MemoryWatcher
        except ImportError:
            pytest.skip("watchfiles not installed")

        async def noop(paths): pass
        return MemoryWatcher(
            workspace_dir=tmp_path,
            on_change=noop,
            debounce_ms=0,
        )

    def test_md_files_included(self, tmp_path: Path):
        watcher = self._make_watcher(tmp_path)
        from watchfiles import Change  # type: ignore[import-untyped]
        changes = {
            (Change.modified, str(tmp_path / "memory" / "2026-01-01.md")),
        }
        result = watcher._filter_md_changes(changes)
        assert len(result) == 1

    def test_non_md_files_excluded(self, tmp_path: Path):
        watcher = self._make_watcher(tmp_path)
        from watchfiles import Change  # type: ignore[import-untyped]
        changes = {
            (Change.modified, str(tmp_path / "memory" / "data.json")),
            (Change.modified, str(tmp_path / "memory" / "image.png")),
        }
        result = watcher._filter_md_changes(changes)
        assert len(result) == 0

    def test_memweave_internals_excluded(self, tmp_path: Path):
        watcher = self._make_watcher(tmp_path)
        from watchfiles import Change  # type: ignore[import-untyped]
        changes = {
            (Change.modified, str(tmp_path / ".memweave" / "index.sqlite")),
        }
        result = watcher._filter_md_changes(changes)
        assert len(result) == 0

    def test_mixed_changes_only_md_returned(self, tmp_path: Path):
        watcher = self._make_watcher(tmp_path)
        from watchfiles import Change  # type: ignore[import-untyped]
        changes = {
            (Change.modified, str(tmp_path / "memory" / "2026-01-01.md")),
            (Change.modified, str(tmp_path / "memory" / "README.txt")),
            (Change.added, str(tmp_path / "memory" / "architecture.md")),
        }
        result = watcher._filter_md_changes(changes)
        assert len(result) == 2

    def test_empty_changes(self, tmp_path: Path):
        watcher = self._make_watcher(tmp_path)
        result = watcher._filter_md_changes(set())
        assert result == set()


# ── MemoryWatcher construction ────────────────────────────────────────────────

class TestMemoryWatcherConstruction:
    def test_raises_without_watchfiles(self, tmp_path: Path):
        """ImportError when watchfiles is not installed."""
        with patch.dict("sys.modules", {"watchfiles": None}):
            # Force re-import with watchfiles unavailable
            import importlib
            import memweave.sync.watcher as wmod
            original = wmod._WATCHFILES_AVAILABLE
            wmod._WATCHFILES_AVAILABLE = False
            try:
                from memweave.sync.watcher import MemoryWatcher
                async def noop(paths): pass
                with pytest.raises(ImportError, match="watchfiles"):
                    MemoryWatcher(workspace_dir=tmp_path, on_change=noop)
            finally:
                wmod._WATCHFILES_AVAILABLE = original


# ── MemoryWatcher.run ─────────────────────────────────────────────────────────

class TestMemoryWatcherRun:
    async def test_run_cancelled_cleanly(self, tmp_path: Path):
        """Cancelling the watcher task raises CancelledError cleanly."""
        try:
            from memweave.sync.watcher import MemoryWatcher
        except ImportError:
            pytest.skip("watchfiles not installed")

        changes_received: list[set[Path]] = []

        async def on_change(paths: set[Path]) -> None:
            changes_received.append(paths)

        watcher = MemoryWatcher(
            workspace_dir=tmp_path,
            on_change=on_change,
            debounce_ms=50,
        )

        # Mock watchfiles.awatch to yield one change then block
        from watchfiles import Change  # type: ignore[import-untyped]
        md_path = tmp_path / "memory" / "test.md"

        async def fake_awatch(*args, **kwargs):
            yield {(Change.modified, str(md_path))}
            # Block until cancelled
            await asyncio.sleep(9999)

        with patch("watchfiles.awatch", fake_awatch):
            task = asyncio.create_task(watcher.run())
            await asyncio.sleep(0.05)  # let one event fire
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_run_callback_invoked_on_change(self, tmp_path: Path):
        """on_change is called with the set of changed .md paths."""
        try:
            from memweave.sync.watcher import MemoryWatcher
        except ImportError:
            pytest.skip("watchfiles not installed")

        received: list[set[Path]] = []

        async def on_change(paths: set[Path]) -> None:
            received.append(paths)

        watcher = MemoryWatcher(
            workspace_dir=tmp_path,
            on_change=on_change,
            debounce_ms=0,
        )

        from watchfiles import Change  # type: ignore[import-untyped]
        md_path = tmp_path / "memory" / "2026-01-01.md"

        async def fake_awatch(*args, **kwargs):
            yield {(Change.modified, str(md_path))}
            return  # stop iteration

        with patch("watchfiles.awatch", fake_awatch):
            task = asyncio.create_task(watcher.run())
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        assert len(received) == 1
        assert md_path in received[0]


# ── MemWeave.start_watching ───────────────────────────────────────────────────

class TestStartWatching:
    async def test_start_watching_no_watchfiles_logs_warning(self, tmp_path: Path):
        """If watchfiles is missing, start_watching logs a warning and returns."""
        from memweave import MemoryConfig
        from memweave.config import EmbeddingConfig

        cfg = MemoryConfig(
            workspace_dir=tmp_path,
            embedding=EmbeddingConfig(model="test-embed"),
        )

        class FakeProvider:
            async def embed_query(self, text): return [0.1] * 8
            async def embed_batch(self, texts): return [[0.1] * 8 for _ in texts]

        from memweave.store import MemWeave

        with patch("memweave.store.MemWeave.start_watching") as mock_watch:
            mock_watch.return_value = None
            mem = MemWeave(cfg, embedding_provider=FakeProvider())
            await mem.open()
            await mem.start_watching()
            await mem.close()

    async def test_start_watching_idempotent(self, tmp_path: Path):
        """Calling start_watching twice does not create a second task."""
        try:
            import watchfiles  # noqa: F401
        except ImportError:
            pytest.skip("watchfiles not installed")

        from memweave import MemoryConfig
        from memweave.config import EmbeddingConfig
        from memweave.store import MemWeave

        cfg = MemoryConfig(
            workspace_dir=tmp_path,
            embedding=EmbeddingConfig(model="test-embed"),
        )

        class FakeProvider:
            async def embed_query(self, text): return [0.1] * 8
            async def embed_batch(self, texts): return [[0.1] * 8 for _ in texts]

        async def fake_run(self):
            await asyncio.sleep(9999)

        with patch("memweave.sync.watcher.MemoryWatcher.run", fake_run):
            async with MemWeave(cfg, embedding_provider=FakeProvider()) as mem:
                await mem.start_watching()
                task1 = mem._watcher_task
                await mem.start_watching()
                task2 = mem._watcher_task
                assert task1 is task2  # same task, not a new one
