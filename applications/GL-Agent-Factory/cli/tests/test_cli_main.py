"""
Tests for main CLI functionality
"""

import pytest
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()


def test_version():
    """Test --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "GreenLang Agent Factory CLI" in result.stdout
    assert "version" in result.stdout


def test_help():
    """Test --help flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GreenLang Agent Factory CLI" in result.stdout
    assert "agent" in result.stdout
    assert "template" in result.stdout


def test_no_args():
    """Test CLI with no arguments shows help."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_agent_help():
    """Test agent command help."""
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "create" in result.stdout
    assert "validate" in result.stdout
    assert "test" in result.stdout
    assert "publish" in result.stdout


def test_template_help():
    """Test template command help."""
    result = runner.invoke(app, ["template", "--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "init" in result.stdout


def test_registry_help():
    """Test registry command help."""
    result = runner.invoke(app, ["registry", "--help"])
    assert result.exit_code == 0
    assert "search" in result.stdout
    assert "pull" in result.stdout
    assert "push" in result.stdout
