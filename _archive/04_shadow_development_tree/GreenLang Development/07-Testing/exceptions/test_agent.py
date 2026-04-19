"""
Tests for GreenLang Agent Exceptions

Tests agent-specific exception classes.
"""

import pytest

from greenlang.exceptions.agent import (
    AgentException,
    ValidationError,
    ExecutionError,
    TimeoutError,
    ConfigurationError,
)


class TestAgentException:
    """Test AgentException base class."""

    def test_agent_exception_prefix(self):
        """Test that AgentException uses GL_AGENT prefix."""
        exc = AgentException("Test error")
        assert exc.error_code.startswith("GL_AGENT")


class TestValidationError:
    """Test ValidationError functionality."""

    def test_basic_validation_error(self):
        """Test creating basic validation error."""
        exc = ValidationError("Invalid input", agent_name="TestAgent")
        assert exc.message == "Invalid input"
        assert exc.agent_name == "TestAgent"

    def test_validation_error_with_invalid_fields(self):
        """Test validation error with invalid_fields."""
        invalid_fields = {"fuel_type": "Required", "amount": "Must be positive"}
        exc = ValidationError(
            "Validation failed",
            agent_name="TestAgent",
            invalid_fields=invalid_fields
        )
        assert exc.context["invalid_fields"] == invalid_fields


class TestExecutionError:
    """Test ExecutionError functionality."""

    def test_basic_execution_error(self):
        """Test creating basic execution error."""
        exc = ExecutionError("Execution failed", agent_name="TestAgent")
        assert exc.message == "Execution failed"

    def test_execution_error_with_step(self):
        """Test execution error with step information."""
        exc = ExecutionError(
            "Execution failed",
            agent_name="TestAgent",
            step="calculate"
        )
        assert exc.context["step"] == "calculate"

    def test_execution_error_with_cause(self):
        """Test execution error with cause exception."""
        original_error = ValueError("Division by zero")
        exc = ExecutionError(
            "Execution failed",
            agent_name="TestAgent",
            cause=original_error
        )
        assert exc.context["cause"] == "Division by zero"
        assert exc.context["cause_type"] == "ValueError"


class TestTimeoutError:
    """Test TimeoutError functionality."""

    def test_basic_timeout_error(self):
        """Test creating basic timeout error."""
        exc = TimeoutError("Execution timed out", agent_name="TestAgent")
        assert exc.message == "Execution timed out"

    def test_timeout_error_with_times(self):
        """Test timeout error with timing information."""
        exc = TimeoutError(
            "Execution timed out",
            agent_name="TestAgent",
            timeout_seconds=30.0,
            elapsed_seconds=31.5
        )
        assert exc.context["timeout_seconds"] == 30.0
        assert exc.context["elapsed_seconds"] == 31.5


class TestConfigurationError:
    """Test ConfigurationError functionality."""

    def test_basic_configuration_error(self):
        """Test creating basic configuration error."""
        exc = ConfigurationError("Missing config", agent_name="TestAgent")
        assert exc.message == "Missing config"
        assert exc.agent_name == "TestAgent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
