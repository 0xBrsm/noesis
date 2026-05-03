"""
tests/integration/conftest.py — Shared fixtures and CLI options for integration tests.

Run integration tests with:
    pytest tests/integration/ \\
        --embedding-model text-embedding-3-small \\
        --llm-model gpt-4o-mini \\
        -v
"""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model identifier to use in integration tests",
    )
    parser.addoption(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model identifier to use in integration tests (required by flush)",
    )


@pytest.fixture(scope="session")
def embedding_model(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("--embedding-model")  # type: ignore[return-value]


@pytest.fixture(scope="session")
def llm_model(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("--llm-model")  # type: ignore[return-value]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Isolated workspace with an empty memory/ directory.

    Each test gets its own tmp_path; pytest cleans it up automatically after the test.
    Tests write their own fixture files into workspace / "memory".
    """
    (tmp_path / "memory").mkdir()
    return tmp_path
