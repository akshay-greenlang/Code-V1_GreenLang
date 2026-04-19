# -*- coding: utf-8 -*-
"""
Basic CLI tests to catch command structure issues
"""

from typer.testing import CliRunner
from greenlang.cli.main import app

runner = CliRunner()


def test_help():
    """Test that help command works"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GreenLang" in result.stdout
    assert "Infrastructure for Climate Intelligence" in result.stdout


def test_version():
    """Test that version command works"""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "GreenLang" in result.stdout
    assert "0.2.0" in result.stdout


def test_pack_help():
    """Test that pack subcommand help works"""
    result = runner.invoke(app, ["pack", "--help"])
    assert result.exit_code == 0
    assert "Pack management commands" in result.stdout


def test_pack_list():
    """Test that pack list command exists"""
    result = runner.invoke(app, ["pack", "list", "--help"])
    assert result.exit_code == 0
    assert "List installed packs" in result.stdout


def test_doctor():
    """Test that doctor command works"""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "GreenLang Environment Check" in result.stdout


def test_run_help():
    """Test that run command help works"""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run a pipeline" in result.stdout


def test_no_args_shows_help():
    """Test that running without args shows help"""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "--help" in result.stdout