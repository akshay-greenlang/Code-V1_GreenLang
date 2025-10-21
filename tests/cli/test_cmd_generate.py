"""
Tests for gl generate command (Agent Factory CLI integration)

This test suite validates the CLI interface for LLM-powered agent generation,
including command structure, argument parsing, error handling, and integration
with the AgentFactory.

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-203 (Agent Factory CLI Integration)
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import yaml
import json

from greenlang.cli.main import app
from greenlang.factory.agent_factory import GenerationResult, AgentFactory
from greenlang.specs import AgentSpecV2

runner = CliRunner()


# Fixtures

@pytest.fixture
def sample_spec_yaml(tmp_path):
    """Create a sample AgentSpec YAML file for testing."""
    spec_content = {
        "schema_version": "2.0.0",
        "id": "test/sample-agent",
        "name": "Sample Test Agent",
        "version": "0.1.0",
        "summary": "Test agent for CLI testing",
        "tags": ["test", "compute"],
        "compute": {
            "entrypoint": "python://sample_agent.agent:compute",
            "deterministic": True,
            "timeout_s": 30,
            "memory_limit_mb": 512,
            "python_version": "3.11",
            "dependencies": ["pydantic>=2.7"],
            "inputs": {
                "value": {
                    "dtype": "float64",
                    "unit": "m^3",
                    "required": True,
                    "ge": 0.0,
                    "description": "Test input value"
                }
            },
            "outputs": {
                "result": {
                    "dtype": "float64",
                    "unit": "kgCO2e",
                    "description": "Test output result"
                }
            }
        },
        "provenance": {
            "pin_ef": True,
            "gwp_set": "AR6GWP100",
            "record": ["inputs", "outputs", "timestamp"]
        }
    }

    spec_file = tmp_path / "sample_spec.yaml"
    with open(spec_file, "w") as f:
        yaml.dump(spec_content, f)

    return spec_file


@pytest.fixture
def sample_spec_json(tmp_path):
    """Create a sample AgentSpec JSON file for testing."""
    spec_content = {
        "schema_version": "2.0.0",
        "id": "test/sample-agent",
        "name": "Sample Test Agent",
        "version": "0.1.0",
        "summary": "Test agent for CLI testing",
        "tags": ["test", "compute"],
        "compute": {
            "entrypoint": "python://sample_agent.agent:compute",
            "deterministic": True,
            "timeout_s": 30,
            "memory_limit_mb": 512,
            "python_version": "3.11",
            "dependencies": ["pydantic>=2.7"],
            "inputs": {
                "value": {
                    "dtype": "float64",
                    "unit": "m^3",
                    "required": True,
                    "ge": 0.0
                }
            },
            "outputs": {
                "result": {
                    "dtype": "float64",
                    "unit": "kgCO2e"
                }
            }
        },
        "provenance": {
            "pin_ef": True,
            "gwp_set": "AR6GWP100",
            "record": ["inputs", "outputs"]
        }
    }

    spec_file = tmp_path / "sample_spec.json"
    with open(spec_file, "w") as f:
        json.dump(spec_content, f)

    return spec_file


@pytest.fixture
def mock_successful_generation():
    """Mock successful agent generation result."""
    return GenerationResult(
        success=True,
        agent_code="# Generated agent code\nclass SampleAgent:\n    pass",
        test_code="# Generated test code\ndef test_agent():\n    pass",
        docs="# Sample Agent Documentation",
        demo_script="# Demo script\nprint('Demo')",
        validation_result=Mock(
            passed=True,
            syntax_errors=[],
            type_errors=[],
            lint_warnings=[],
            test_failures=[]
        ),
        provenance={
            "timestamp": "2025-10-21T12:00:00Z",
            "llm_model": "gpt-4",
            "factory_version": "0.1.0"
        },
        duration_seconds=45.3,
        total_cost_usd=1.25,
        attempts=1
    )


@pytest.fixture
def mock_failed_generation():
    """Mock failed agent generation result."""
    return GenerationResult(
        success=False,
        error="Validation failed: syntax errors in generated code",
        validation_result=Mock(
            passed=False,
            syntax_errors=["SyntaxError: invalid syntax at line 10"],
            type_errors=["Type error: incompatible types"],
            lint_warnings=["W: Unused import"],
            test_failures=["test_compute failed: AssertionError"]
        ),
        duration_seconds=120.0,
        total_cost_usd=3.5,
        attempts=3
    )


# Basic Command Structure Tests

def test_generate_help():
    """Test that generate command help works."""
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "Generate agents using LLM-powered code generation" in result.stdout


def test_generate_agent_help():
    """Test that generate agent subcommand help works."""
    result = runner.invoke(app, ["generate", "agent", "--help"])
    assert result.exit_code == 0
    assert "Generate a GreenLang agent from AgentSpec" in result.stdout
    assert "--budget" in result.stdout
    assert "--max-attempts" in result.stdout
    assert "--skip-tests" in result.stdout
    assert "--skip-docs" in result.stdout


def test_generate_agent_requires_spec_file():
    """Test that generate agent requires a spec file argument."""
    result = runner.invoke(app, ["generate", "agent"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stdout or "required" in result.stdout.lower()


def test_generate_agent_spec_file_must_exist():
    """Test that spec file must exist."""
    result = runner.invoke(app, ["generate", "agent", "nonexistent_spec.yaml"], input="n\n")
    assert result.exit_code != 0


def test_generate_agent_unsupported_file_format(tmp_path):
    """Test that unsupported file formats are rejected."""
    spec_file = tmp_path / "spec.txt"
    spec_file.write_text("invalid spec")

    result = runner.invoke(app, ["generate", "agent", str(spec_file)], input="n\n")
    assert result.exit_code != 0
    assert "Unsupported file format" in result.stdout


# YAML Spec Loading Tests

@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_loads_yaml_spec(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test that YAML spec files are loaded correctly."""
    # Mock factory instance
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml)],
        input="y\n"  # Confirm generation
    )

    # Should succeed (exit code 0)
    assert result.exit_code == 0
    assert "Sample Test Agent" in result.stdout
    assert "v0.1.0" in result.stdout


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_loads_json_spec(mock_factory_class, sample_spec_json, mock_successful_generation):
    """Test that JSON spec files are loaded correctly."""
    # Mock factory instance
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_json)],
        input="y\n"  # Confirm generation
    )

    # Should succeed
    assert result.exit_code == 0
    assert "Sample Test Agent" in result.stdout


def test_generate_agent_invalid_yaml(tmp_path):
    """Test handling of invalid YAML spec files."""
    invalid_spec = tmp_path / "invalid.yaml"
    invalid_spec.write_text("invalid: yaml: content: [")

    result = runner.invoke(app, ["generate", "agent", str(invalid_spec)], input="n\n")
    assert result.exit_code != 0
    assert "Failed to load spec" in result.stdout


# Option Parsing Tests

@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_with_custom_budget(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test --budget option."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--budget", "10.0"],
        input="y\n"
    )

    assert result.exit_code == 0
    assert "$10.00" in result.stdout

    # Verify factory was initialized with custom budget
    mock_factory_class.assert_called_once()
    call_kwargs = mock_factory_class.call_args.kwargs
    assert call_kwargs['budget_per_agent_usd'] == 10.0


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_with_custom_max_attempts(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test --max-attempts option."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--max-attempts", "5"],
        input="y\n"
    )

    assert result.exit_code == 0

    # Verify factory was initialized with custom max attempts
    call_kwargs = mock_factory_class.call_args.kwargs
    assert call_kwargs['max_refinement_attempts'] == 5


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_skip_tests(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test --skip-tests option."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--skip-tests"],
        input="y\n"
    )

    assert result.exit_code == 0
    assert "Skipped" in result.stdout

    # Verify generate_agent was called with skip_tests=True
    mock_factory.generate_agent.assert_called_once()
    call_kwargs = mock_factory.generate_agent.call_args.kwargs
    assert call_kwargs['skip_tests'] is True


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_skip_docs(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test --skip-docs option."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--skip-docs"],
        input="y\n"
    )

    assert result.exit_code == 0

    # Verify generate_agent was called with skip_docs=True
    call_kwargs = mock_factory.generate_agent.call_args.kwargs
    assert call_kwargs['skip_docs'] is True


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_skip_demo(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test --skip-demo option."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--skip-demo"],
        input="y\n"
    )

    assert result.exit_code == 0

    # Verify generate_agent was called with skip_demo=True
    call_kwargs = mock_factory.generate_agent.call_args.kwargs
    assert call_kwargs['skip_demo'] is True


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_generate_agent_custom_output_dir(mock_factory_class, sample_spec_yaml, mock_successful_generation, tmp_path):
    """Test --output option for custom output directory."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    custom_output = tmp_path / "custom_output"

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--output", str(custom_output)],
        input="y\n"
    )

    assert result.exit_code == 0
    assert str(custom_output) in result.stdout


# Successful Generation Tests

@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_successful_generation_display(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test that successful generation displays correct information."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml)],
        input="y\n"
    )

    assert result.exit_code == 0

    # Check for success indicators
    assert "Generated Successfully" in result.stdout or "✓" in result.stdout

    # Check for statistics
    assert "45.1s" in result.stdout or "45.3s" in result.stdout  # Duration
    assert "$1.25" in result.stdout  # Cost

    # Check for generated files mention
    assert "agent.py" in result.stdout
    assert "test_agent.py" in result.stdout or "Tests:" in result.stdout

    # Check for next steps
    assert "Next Steps" in result.stdout
    assert "pytest" in result.stdout


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_successful_generation_verbose(mock_factory_class, sample_spec_yaml, mock_successful_generation):
    """Test verbose output for successful generation."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_successful_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--verbose"],
        input="y\n"
    )

    assert result.exit_code == 0

    # Verbose should show provenance
    assert "Provenance" in result.stdout or "gpt-4" in result.stdout


# Failed Generation Tests

@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_failed_generation_display(mock_factory_class, sample_spec_yaml, mock_failed_generation):
    """Test that failed generation displays error information."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_failed_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml)],
        input="y\n"
    )

    assert result.exit_code != 0

    # Check for failure indicators
    assert "Failed" in result.stdout or "✗" in result.stdout

    # Check for error message
    assert "Validation failed" in result.stdout or "syntax errors" in result.stdout

    # Check for suggestions
    assert "Suggestions" in result.stdout or "increase" in result.stdout.lower()


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_failed_generation_shows_validation_errors(mock_factory_class, sample_spec_yaml, mock_failed_generation):
    """Test that validation errors are displayed on failure."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(return_value=mock_failed_generation)
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml)],
        input="y\n"
    )

    assert result.exit_code != 0

    # Should show validation errors
    assert "SyntaxError" in result.stdout or "Syntax Errors" in result.stdout
    assert "Type" in result.stdout or "type error" in result.stdout.lower()


# User Confirmation Tests

@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_user_can_cancel_generation(mock_factory_class, sample_spec_yaml):
    """Test that user can cancel generation at confirmation prompt."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock()
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml)],
        input="n\n"  # Decline generation
    )

    assert result.exit_code == 0
    assert "cancelled" in result.stdout.lower()

    # Factory should not have generated anything
    mock_factory.generate_agent.assert_not_called()


# Edge Cases

def test_budget_out_of_range_low(sample_spec_yaml):
    """Test that budget below minimum is rejected."""
    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--budget", "0.05"],
        input="n\n"
    )

    # Typer should reject this before reaching the command logic
    assert result.exit_code != 0


def test_budget_out_of_range_high(sample_spec_yaml):
    """Test that budget above maximum is rejected."""
    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml), "--budget", "100.0"],
        input="n\n"
    )

    # Typer should reject this
    assert result.exit_code != 0


@patch('greenlang.cli.cmd_generate.AgentFactory')
def test_exception_during_generation_is_handled(mock_factory_class, sample_spec_yaml):
    """Test that exceptions during generation are handled gracefully."""
    mock_factory = Mock()
    mock_factory.generate_agent = AsyncMock(side_effect=Exception("LLM API error"))
    mock_factory_class.return_value = mock_factory

    result = runner.invoke(
        app,
        ["generate", "agent", str(sample_spec_yaml)],
        input="y\n"
    )

    assert result.exit_code != 0
    assert "failed" in result.stdout.lower()
    assert "LLM API error" in result.stdout


# Integration-style Tests (minimal mocking)

def test_generate_command_in_main_app():
    """Test that generate command is registered in main app."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout.lower()


def test_generate_agent_command_available():
    """Test that generate agent command is available."""
    result = runner.invoke(app, ["generate", "agent", "--help"])
    assert result.exit_code == 0
    assert "AgentSpec" in result.stdout
