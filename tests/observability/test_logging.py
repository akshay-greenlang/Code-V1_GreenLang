"""
Tests for structured logging
"""

import pytest
import json
from datetime import datetime
from greenlang.observability import (
    LogLevel,
    LogContext,
    LogEntry,
    StructuredLogger,
    LogFormatter,
    LogAggregator,
    get_logger,
    configure_logging,
)


class TestLogContext:
    """Test LogContext functionality"""

    def test_log_context_initialization(self):
        """Test log context can be initialized"""
        context = LogContext(tenant_id="test", user_id="user123")
        assert context.tenant_id == "test"
        assert context.user_id == "user123"

    def test_log_context_to_dict(self):
        """Test converting log context to dictionary"""
        context = LogContext(
            tenant_id="test", component="api", operation="process_request"
        )
        result = context.to_dict()
        assert result["tenant_id"] == "test"
        assert result["component"] == "api"
        assert result["operation"] == "process_request"

    def test_log_context_to_dict_excludes_none(self):
        """Test that None values are excluded from dict"""
        context = LogContext(tenant_id="test")
        result = context.to_dict()
        assert "tenant_id" in result
        assert "user_id" not in result


class TestLogEntry:
    """Test LogEntry functionality"""

    def test_log_entry_creation(self):
        """Test creating a log entry"""
        context = LogContext(tenant_id="test")
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            message="Test message",
            context=context,
        )
        assert entry.message == "Test message"
        assert entry.level == LogLevel.INFO

    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary"""
        context = LogContext(tenant_id="test")
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.ERROR,
            message="Error occurred",
            context=context,
            data={"error_code": "E001"},
        )
        result = entry.to_dict()
        assert result["level"] == "ERROR"
        assert result["message"] == "Error occurred"
        assert result["tenant_id"] == "test"
        assert result["data"]["error_code"] == "E001"

    def test_log_entry_to_json(self):
        """Test converting log entry to JSON"""
        context = LogContext(tenant_id="test")
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.WARNING,
            message="Warning message",
            context=context,
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert parsed["level"] == "WARNING"
        assert parsed["message"] == "Warning message"


class TestStructuredLogger:
    """Test StructuredLogger functionality"""

    def test_logger_initialization(self):
        """Test logger initialization"""
        context = LogContext(component="test")
        logger = StructuredLogger("test_logger", context)
        assert logger.name == "test_logger"
        assert logger.context.component == "test"

    def test_logger_debug(self):
        """Test debug logging"""
        logger = StructuredLogger("test")
        logger.debug("Debug message", key="value")
        # Should not raise exception

    def test_logger_info(self):
        """Test info logging"""
        logger = StructuredLogger("test")
        logger.info("Info message", count=42)
        # Should not raise exception

    def test_logger_warning(self):
        """Test warning logging"""
        logger = StructuredLogger("test")
        logger.warning("Warning message", status="degraded")
        # Should not raise exception

    def test_logger_error(self):
        """Test error logging"""
        logger = StructuredLogger("test")
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Error occurred", exception=e, operation="test")
        # Should not raise exception

    def test_logger_critical(self):
        """Test critical logging"""
        logger = StructuredLogger("test")
        logger.critical("Critical issue", severity="high")
        # Should not raise exception

    def test_logger_with_context(self):
        """Test temporary context"""
        logger = StructuredLogger("test", LogContext(tenant_id="default"))
        with logger.with_context(tenant_id="custom"):
            assert logger.context.tenant_id == "custom"
        assert logger.context.tenant_id == "default"


class TestLogAggregator:
    """Test LogAggregator functionality"""

    def test_aggregator_initialization(self):
        """Test aggregator initialization"""
        aggregator = LogAggregator(max_logs=100)
        assert aggregator.max_logs == 100
        assert len(aggregator.logs) == 0

    def test_add_log_entry(self):
        """Test adding log entry"""
        aggregator = LogAggregator()
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            message="Test",
            context=LogContext(),
        )
        aggregator.add_log(entry)
        assert len(aggregator.logs) == 1

    def test_get_logs_filtered_by_level(self):
        """Test filtering logs by level"""
        aggregator = LogAggregator()
        aggregator.add_log(
            LogEntry(
                datetime.utcnow(), LogLevel.INFO, "Info", LogContext(component="test")
            )
        )
        aggregator.add_log(
            LogEntry(
                datetime.utcnow(), LogLevel.ERROR, "Error", LogContext(component="test")
            )
        )

        errors = aggregator.get_logs(level=LogLevel.ERROR)
        assert len(errors) == 1
        assert errors[0].level == LogLevel.ERROR

    def test_get_logs_filtered_by_component(self):
        """Test filtering logs by component"""
        aggregator = LogAggregator()
        aggregator.add_log(
            LogEntry(datetime.utcnow(), LogLevel.INFO, "A", LogContext(component="api"))
        )
        aggregator.add_log(
            LogEntry(
                datetime.utcnow(), LogLevel.INFO, "B", LogContext(component="worker")
            )
        )

        api_logs = aggregator.get_logs(component="api")
        assert len(api_logs) == 1
        assert api_logs[0].context.component == "api"

    def test_get_statistics(self):
        """Test getting log statistics"""
        aggregator = LogAggregator()
        aggregator.add_log(
            LogEntry(datetime.utcnow(), LogLevel.INFO, "Test", LogContext())
        )
        aggregator.add_log(
            LogEntry(datetime.utcnow(), LogLevel.ERROR, "Error", LogContext())
        )

        stats = aggregator.get_statistics()
        assert stats["total_logs"] == 2
        assert "log_counts" in stats

    def test_get_error_summary(self):
        """Test getting error summary"""
        aggregator = LogAggregator()
        aggregator.add_log(
            LogEntry(
                datetime.utcnow(),
                LogLevel.ERROR,
                "Error 1",
                LogContext(),
                exception={"type": "ValueError", "message": "Test"},
            )
        )
        aggregator.add_log(
            LogEntry(
                datetime.utcnow(),
                LogLevel.ERROR,
                "Error 2",
                LogContext(),
                exception={"type": "TypeError", "message": "Test"},
            )
        )

        summary = aggregator.get_error_summary()
        assert summary["total_errors"] == 2
        assert "error_types" in summary


class TestLogFormatter:
    """Test LogFormatter functionality"""

    def test_formatter_json_output(self):
        """Test formatter produces JSON output"""
        import logging

        formatter = LogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"


class TestGlobalLoggerInstances:
    """Test global logger instances"""

    def test_get_logger(self):
        """Test getting logger"""
        logger = get_logger("test_module")
        assert logger.name == "test_module"

    def test_get_logger_with_context(self):
        """Test getting logger with context"""
        context = LogContext(tenant_id="test")
        logger = get_logger("test_module_with_context", context)
        assert logger.context.tenant_id == "test"

    def test_configure_logging(self):
        """Test configuring logging system"""
        configure_logging(level="DEBUG", format_json=True)
        # Should not raise exception
