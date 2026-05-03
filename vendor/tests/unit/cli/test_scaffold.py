"""tests/unit/cli/test_scaffold.py — Milestone 1: CLI scaffolding tests."""

from __future__ import annotations

from click.testing import CliRunner

from memweave import __version__
from memweave.cli import cli


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "memweave" in result.output


def test_version_exits_zero() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_unknown_command_exits_nonzero() -> None:
    result = CliRunner().invoke(cli, ["nonexistent-command"])
    assert result.exit_code != 0
