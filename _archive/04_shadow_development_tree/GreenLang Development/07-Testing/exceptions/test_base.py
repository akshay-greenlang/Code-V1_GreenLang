"""
Tests for GreenLang Base Exception

Tests base exception functionality including error codes, context, and serialization.
"""

import pytest
from datetime import datetime

from greenlang.exceptions.base import GreenLangException


class TestGreenLangException:
    """Test GreenLangException functionality."""

    def test_basic_exception(self):
        """Test creating basic exception."""
        exc = GreenLangException("Test error")
        assert exc.message == "Test error"
        assert exc.error_code.startswith("GL_")
        assert exc.context == {}

    def test_exception_with_error_code(self):
        """Test exception with custom error code."""
        exc = GreenLangException("Test error", error_code="GL_CUSTOM_001")
        assert exc.error_code == "GL_CUSTOM_001"

    def test_exception_with_agent_name(self):
        """Test exception with agent name."""
        exc = GreenLangException("Test error", agent_name="TestAgent")
        assert exc.agent_name == "TestAgent"

    def test_exception_with_context(self):
        """Test exception with context dictionary."""
        context = {"key": "value", "count": 42}
        exc = GreenLangException("Test error", context=context)
        assert exc.context == context

    def test_exception_has_timestamp(self):
        """Test that exception captures timestamp."""
        exc = GreenLangException("Test error")
        assert isinstance(exc.timestamp, datetime)

    def test_exception_has_traceback(self):
        """Test that exception captures traceback."""
        exc = GreenLangException("Test error")
        assert isinstance(exc.traceback_str, str)
        assert len(exc.traceback_str) > 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        exc = GreenLangException(
            "Test error",
            error_code="GL_TEST_001",
            agent_name="TestAgent",
            context={"key": "value"}
        )
        result = exc.to_dict()

        assert result["error_type"] == "GreenLangException"
        assert result["error_code"] == "GL_TEST_001"
        assert result["message"] == "Test error"
        assert result["agent_name"] == "TestAgent"
        assert result["context"] == {"key": "value"}
        assert "timestamp" in result
        assert "traceback" in result

    def test_to_json(self):
        """Test conversion to JSON."""
        exc = GreenLangException("Test error")
        json_str = exc.to_json()
        assert isinstance(json_str, str)
        assert "Test error" in json_str
        assert "error_code" in json_str

    def test_str_representation(self):
        """Test string representation."""
        exc = GreenLangException("Test error", error_code="GL_TEST_001")
        str_repr = str(exc)
        assert "[GL_TEST_001]" in str_repr
        assert "Test error" in str_repr

    def test_str_with_agent_name(self):
        """Test string representation with agent name."""
        exc = GreenLangException(
            "Test error",
            error_code="GL_TEST_001",
            agent_name="TestAgent"
        )
        str_repr = str(exc)
        assert "[GL_TEST_001]" in str_repr
        assert "Agent: TestAgent" in str_repr
        assert "Test error" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        exc = GreenLangException(
            "Test error",
            error_code="GL_TEST_001",
            agent_name="TestAgent"
        )
        repr_str = repr(exc)
        assert "GreenLangException" in repr_str
        assert "Test error" in repr_str
        assert "GL_TEST_001" in repr_str
        assert "TestAgent" in repr_str

    def test_error_code_generation(self):
        """Test automatic error code generation."""
        exc = GreenLangException("Test error")
        # Should generate GL_GREEN_LANG_EXCEPTION
        assert exc.error_code == "GL_GREEN_LANG_EXCEPTION"


class TestErrorCodeGeneration:
    """Test error code generation from class names."""

    def test_camel_case_to_screaming_snake(self):
        """Test CamelCase to SCREAMING_SNAKE_CASE conversion."""
        class ValidationError(GreenLangException):
            ERROR_PREFIX = "GL_AGENT"

        exc = ValidationError("Test")
        # Should generate GL_AGENT_VALIDATION_ERROR
        assert "VALIDATION_ERROR" in exc.error_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
