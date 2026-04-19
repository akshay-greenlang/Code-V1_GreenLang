"""
GreenLang Observability - Logging Module Tests
===============================================

Comprehensive unit tests for structured logging functionality.
"""

import pytest
import json
import logging
import threading
from io import StringIO
from unittest.mock import MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from observability.logging import (
    StructuredLogger,
    LogConfig,
    LogLevel,
    CorrelationContext,
    BoundLogger,
    SensitiveDataRedactor,
    JSONFormatter,
    TextFormatter,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    add_log_context,
    clear_log_context,
    redact_sensitive_data,
    configure_root_logger,
)


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LogConfig()

        assert config.level == LogLevel.INFO
        assert config.format == "json"
        assert config.include_timestamp is True
        assert config.include_correlation_id is True
        assert config.output == "stdout"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = LogConfig(
            level=LogLevel.DEBUG,
            format="text",
            include_location=True,
            pretty_print=True,
        )

        assert config.level == LogLevel.DEBUG
        assert config.format == "text"
        assert config.include_location is True
        assert config.pretty_print is True

    def test_default_redact_patterns(self) -> None:
        """Test default redaction patterns are set."""
        config = LogConfig()

        assert len(config.redact_patterns) > 0
        assert "password" in config.redact_fields
        assert "secret" in config.redact_fields
        assert "api_key" in config.redact_fields

    def test_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "text",
            "LOG_PRETTY": "true",
        }):
            config = LogConfig.from_env()

            assert config.level == LogLevel.DEBUG
            assert config.format == "text"
            assert config.pretty_print is True


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test log level numeric values."""
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.INFO.value == 20
        assert LogLevel.WARNING.value == 30
        assert LogLevel.ERROR.value == 40
        assert LogLevel.CRITICAL.value == 50

    def test_from_string(self) -> None:
        """Test parsing log level from string."""
        assert LogLevel.from_string("debug") == LogLevel.DEBUG
        assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_string("info") == LogLevel.INFO
        assert LogLevel.from_string("warning") == LogLevel.WARNING
        assert LogLevel.from_string("warn") == LogLevel.WARNING
        assert LogLevel.from_string("error") == LogLevel.ERROR
        assert LogLevel.from_string("critical") == LogLevel.CRITICAL
        assert LogLevel.from_string("fatal") == LogLevel.CRITICAL
        assert LogLevel.from_string("unknown") == LogLevel.INFO  # Default


class TestSensitiveDataRedactor:
    """Tests for SensitiveDataRedactor."""

    def test_redact_email(self) -> None:
        """Test redacting email addresses."""
        redactor = SensitiveDataRedactor()
        text = "Contact us at test@example.com for more info"
        result = redactor.redact_string(text)

        assert "test@example.com" not in result
        assert "[REDACTED]" in result

    def test_redact_phone(self) -> None:
        """Test redacting phone numbers."""
        redactor = SensitiveDataRedactor()
        text = "Call 123-456-7890 or 1234567890"
        result = redactor.redact_string(text)

        assert "123-456-7890" not in result

    def test_redact_password_field(self) -> None:
        """Test redacting password fields."""
        redactor = SensitiveDataRedactor(fields={"password", "secret"})
        data = {
            "username": "testuser",
            "password": "supersecret123",
            "email": "test@test.com",
        }
        result = redactor.redact_dict(data)

        assert result["username"] == "testuser"
        assert result["password"] == "[REDACTED]"

    def test_redact_nested_dict(self) -> None:
        """Test redacting nested dictionaries."""
        redactor = SensitiveDataRedactor(fields={"api_key"})
        data = {
            "config": {
                "name": "test",
                "credentials": {
                    "api_key": "secret123",
                },
            },
        }
        result = redactor.redact_dict(data)

        assert result["config"]["credentials"]["api_key"] == "[REDACTED]"
        assert result["config"]["name"] == "test"

    def test_redact_list(self) -> None:
        """Test redacting data in lists."""
        redactor = SensitiveDataRedactor(fields={"token"})
        data = {
            "items": [
                {"name": "item1", "token": "abc123"},
                {"name": "item2", "token": "def456"},
            ],
        }
        result = redactor.redact_dict(data)

        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][1]["token"] == "[REDACTED]"
        assert result["items"][0]["name"] == "item1"

    def test_redact_password_in_string(self) -> None:
        """Test redacting password patterns in strings."""
        redactor = SensitiveDataRedactor()
        text = 'config: password="mysecret" and password: anothersecret'
        result = redactor.redact_string(text)

        assert "mysecret" not in result
        assert "anothersecret" not in result


class TestCorrelationContext:
    """Tests for CorrelationContext."""

    def test_correlation_context_auto_id(self) -> None:
        """Test auto-generating correlation ID."""
        with CorrelationContext() as ctx:
            assert ctx.correlation_id is not None
            assert len(ctx.correlation_id) > 0
            assert get_correlation_id() == ctx.correlation_id

        # After context, should be None
        assert get_correlation_id() is None

    def test_correlation_context_custom_id(self) -> None:
        """Test using custom correlation ID."""
        custom_id = "my-custom-id-123"
        with CorrelationContext(custom_id) as ctx:
            assert ctx.correlation_id == custom_id
            assert get_correlation_id() == custom_id

    def test_correlation_context_extra_context(self) -> None:
        """Test extra context in correlation context."""
        with CorrelationContext(
            correlation_id="test-id",
            extra_context={"request_id": "req-123", "user": "test"},
        ):
            # Context should be available
            pass

    def test_nested_correlation_context(self) -> None:
        """Test nested correlation contexts."""
        with CorrelationContext("outer") as outer:
            assert get_correlation_id() == "outer"

            with CorrelationContext("inner") as inner:
                assert get_correlation_id() == "inner"

            # After inner exits, should be back to outer
            assert get_correlation_id() == "outer"

        assert get_correlation_id() is None


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_logger_creation(self) -> None:
        """Test logger creation."""
        config = LogConfig(level=LogLevel.DEBUG)
        logger = StructuredLogger("test-logger", config)

        assert logger.name == "test-logger"
        assert logger.config == config

    def test_log_levels(self) -> None:
        """Test all log levels."""
        output = StringIO()
        config = LogConfig(
            level=LogLevel.DEBUG,
            format="json",
            output="stdout",
        )

        # Create logger with captured output
        logger = StructuredLogger("test", config)

        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_log_with_extra_fields(self) -> None:
        """Test logging with extra fields."""
        config = LogConfig(format="json")
        logger = StructuredLogger("test", config)

        # Should include extra fields
        logger.info(
            "Calculation completed",
            agent_id="GL-006",
            duration_ms=150.5,
            result_count=42,
        )

    def test_log_exception(self) -> None:
        """Test exception logging."""
        config = LogConfig(format="json")
        logger = StructuredLogger("test", config)

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("An error occurred", context="testing")

    def test_bound_logger(self) -> None:
        """Test creating bound logger."""
        logger = StructuredLogger("test")
        bound = logger.bind(agent_id="GL-006", version="1.0")

        assert isinstance(bound, BoundLogger)

        # Bound fields should be included
        bound.info("Test message", extra="value")

    def test_bound_logger_nested_bind(self) -> None:
        """Test nested bound loggers."""
        logger = StructuredLogger("test")
        bound1 = logger.bind(agent_id="GL-006")
        bound2 = bound1.bind(calculation="pinch")

        # Should have both bound fields
        bound2.info("Test")

    def test_with_correlation_id(self) -> None:
        """Test using correlation context with logger."""
        logger = StructuredLogger("test")

        with logger.with_correlation_id("test-correlation"):
            assert get_correlation_id() == "test-correlation"
            logger.info("Within correlation context")


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_json_output(self) -> None:
        """Test JSON formatted output."""
        config = LogConfig(format="json", include_timestamp=True)
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_json_with_extra_fields(self) -> None:
        """Test JSON with extra fields."""
        config = LogConfig(format="json")
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra = {"agent_id": "GL-006", "count": 42}

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["agent_id"] == "GL-006"
        assert parsed["count"] == 42

    def test_json_with_correlation_id(self) -> None:
        """Test JSON with correlation ID."""
        config = LogConfig(format="json", include_correlation_id=True)
        formatter = JSONFormatter(config)

        with CorrelationContext("test-id"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            parsed = json.loads(output)

            assert parsed.get("correlation_id") == "test-id"

    def test_json_redaction(self) -> None:
        """Test sensitive data redaction in JSON."""
        config = LogConfig(format="json")
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra = {"password": "secret123", "username": "testuser"}

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["password"] == "[REDACTED]"
        assert parsed["username"] == "testuser"


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_text_output(self) -> None:
        """Test text formatted output."""
        config = LogConfig(format="text", include_timestamp=True)
        formatter = TextFormatter(config, use_colors=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "Test message" in output
        assert "[INFO" in output or "INFO" in output


class TestLogContextFunctions:
    """Tests for log context functions."""

    def test_set_get_correlation_id(self) -> None:
        """Test setting and getting correlation ID."""
        token = set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

    def test_add_log_context(self) -> None:
        """Test adding to log context."""
        clear_log_context()
        add_log_context("key1", "value1")
        add_log_context("key2", 42)

        # Context should be available for logging

    def test_clear_log_context(self) -> None:
        """Test clearing log context."""
        add_log_context("test", "value")
        clear_log_context()
        # Should be cleared


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_creates_logger(self) -> None:
        """Test getting a new logger."""
        logger = get_logger("new-logger")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "new-logger"

    def test_get_logger_returns_same_instance(self) -> None:
        """Test getting same logger instance."""
        logger1 = get_logger("shared-logger")
        logger2 = get_logger("shared-logger")

        assert logger1 is logger2

    def test_get_logger_with_config(self) -> None:
        """Test getting logger with custom config."""
        config = LogConfig(level=LogLevel.DEBUG)
        logger = get_logger("configured-logger", config)

        assert logger.config.level == LogLevel.DEBUG


class TestRedactSensitiveData:
    """Tests for redact_sensitive_data function."""

    def test_redact_string(self) -> None:
        """Test redacting sensitive string."""
        text = "My email is test@example.com"
        result = redact_sensitive_data(text)

        assert "test@example.com" not in result

    def test_redact_dict(self) -> None:
        """Test redacting sensitive dictionary."""
        data = {"password": "secret", "name": "test"}
        result = redact_sensitive_data(
            data,
            fields={"password"},
        )

        assert result["password"] == "[REDACTED]"
        assert result["name"] == "test"


class TestThreadSafety:
    """Tests for thread safety of logging."""

    def test_correlation_id_thread_isolation(self) -> None:
        """Test correlation ID is thread-isolated."""
        results = {}

        def set_and_check(thread_id: int) -> None:
            with CorrelationContext(f"thread-{thread_id}"):
                # Small delay to allow interleaving
                import time
                time.sleep(0.01)
                results[thread_id] = get_correlation_id()

        threads = []
        for i in range(10):
            t = threading.Thread(target=set_and_check, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own correlation ID
        for i in range(10):
            assert results[i] == f"thread-{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
