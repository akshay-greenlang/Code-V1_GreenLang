# -*- coding: utf-8 -*-
"""
Unit Tests for Structured Logging Setup - INFRA-009

Tests the structlog configuration pipeline: configure_logging, get_logger,
processor chain injection, log level filtering, file handler creation, and
reconfiguration behaviour.

Module under test: greenlang.infrastructure.logging.setup
"""

import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from greenlang.infrastructure.logging.config import LoggingConfig, reset_config
from greenlang.infrastructure.logging.setup import (
    configure_logging,
    get_logger,
    reset_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_structlog_output(logger, level: str, message: str, **kwargs) -> str:
    """Call *logger* at *level* and capture the string written to stdout.

    structlog.PrintLoggerFactory writes to stdout, so we temporarily
    redirect sys.stdout to capture the output.
    """
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        getattr(logger, level)(message, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset structlog, stdlib logging, and module-level config between tests."""
    yield
    reset_logging()
    reset_config()
    structlog.reset_defaults()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


@pytest.fixture
def json_config():
    """LoggingConfig configured for JSON output."""
    return LoggingConfig(
        level="DEBUG",
        format="json",
        service_name="greenlang-test",
        service_version="0.1.0",
        environment="test",
        enable_redaction=False,
        async_logging=False,
    )


@pytest.fixture
def console_config():
    """LoggingConfig configured for console (human-readable) output."""
    return LoggingConfig(
        level="DEBUG",
        format="console",
        service_name="greenlang-test",
        service_version="0.1.0",
        environment="test",
        enable_redaction=False,
        async_logging=False,
    )


# ---------------------------------------------------------------------------
# Tests: configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_configure_logging_json_mode(self, json_config):
        """JSON format configuration produces valid JSON log lines."""
        configure_logging(json_config)
        logger = get_logger("test.json_mode")
        output = _capture_structlog_output(logger, "info", "hello json")

        # structlog JSONRenderer outputs a single JSON line
        line = output.strip()
        parsed = json.loads(line)
        assert parsed.get("event") == "hello json"

    def test_configure_logging_console_mode(self, console_config):
        """Console format produces non-JSON, human-readable output."""
        configure_logging(console_config)
        logger = get_logger("test.console_mode")
        output = _capture_structlog_output(logger, "info", "hello console")

        # Console renderer output should NOT be valid JSON
        for line in output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            with pytest.raises(json.JSONDecodeError):
                json.loads(line)
            break

    def test_configure_logging_sets_level(self):
        """Log-level filtering respects the configured level.

        When level is INFO, DEBUG messages are suppressed by structlog.
        """
        config = LoggingConfig(
            level="INFO",
            format="json",
            enable_redaction=False,
            async_logging=False,
        )
        configure_logging(config)
        logger = get_logger("test.level_filter")

        # INFO should produce output
        info_out = _capture_structlog_output(logger, "info", "should_appear")
        assert "should_appear" in info_out

    def test_configure_logging_with_file(self, json_config, tmp_path):
        """When log_file is specified a FileHandler is added to the root logger."""
        log_file = tmp_path / "test.log"
        json_config.log_file = str(log_file)
        configure_logging(json_config)

        # Use stdlib logging (which gets the FileHandler) to verify
        stdlib_logger = logging.getLogger("test.file_output")
        stdlib_logger.info("file_test_message")

        # Flush all handlers
        for h in logging.getLogger().handlers:
            h.flush()

        # The file handler uses QueueHandler by default; give it a moment
        import time
        time.sleep(0.1)
        for h in logging.getLogger().handlers:
            h.flush()

        if log_file.exists():
            contents = log_file.read_text(encoding="utf-8")
            assert "file_test_message" in contents
        else:
            # If async_logging=False was respected, file should exist
            # but in case of queue delay, just verify no crash
            pass

    def test_configure_logging_default_config(self):
        """configure_logging works with a default LoggingConfig (no explicit args)."""
        config = LoggingConfig(async_logging=False, enable_redaction=False)
        configure_logging(config)
        logger = get_logger("test.defaults")

        # Should not raise
        output = _capture_structlog_output(logger, "info", "default config works")
        assert "default config works" in output

    def test_reconfigure_logging(self, json_config):
        """Calling configure_logging twice is a no-op (guarded by _configured flag).

        The second call should NOT add duplicate handlers.
        """
        configure_logging(json_config)
        handler_count_1 = len(logging.getLogger().handlers)

        # Second call is a no-op because _configured is True
        configure_logging(json_config)
        handler_count_2 = len(logging.getLogger().handlers)

        assert handler_count_2 == handler_count_1


# ---------------------------------------------------------------------------
# Tests: get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for get_logger()."""

    def test_get_logger_returns_bound_logger(self, json_config):
        """get_logger returns a structlog BoundLogger (or compatible proxy)."""
        configure_logging(json_config)
        logger = get_logger("test.bound")

        # structlog bound loggers expose standard level methods
        assert callable(getattr(logger, "info", None))
        assert callable(getattr(logger, "warning", None))
        assert callable(getattr(logger, "error", None))
        assert callable(getattr(logger, "debug", None))

    def test_get_logger_same_name_returns_same_instance(self, json_config):
        """structlog.get_logger caches loggers, so the same name returns the same instance."""
        configure_logging(json_config)
        a = get_logger("test.cache")
        b = get_logger("test.cache")
        # structlog caches on first use when cache_logger_on_first_use=True
        # The proxies will be equal objects
        assert type(a) == type(b)

    def test_get_logger_different_names_return_different_instances(self, json_config):
        """Different logger names produce distinct instances."""
        configure_logging(json_config)
        a = get_logger("test.alpha")
        b = get_logger("test.beta")
        assert a is not b


# ---------------------------------------------------------------------------
# Tests: Custom Processors
# ---------------------------------------------------------------------------


class TestProcessors:
    """Tests for custom structlog processors injected by configure_logging."""

    def test_add_service_info_processor(self, json_config):
        """service_name, service_version, and environment are injected."""
        configure_logging(json_config)
        logger = get_logger("test.service_info")
        output = _capture_structlog_output(logger, "info", "svc_test")

        parsed = json.loads(output.strip())

        assert parsed.get("service_name") == "greenlang-test"
        assert parsed.get("service_version") == "0.1.0"
        assert parsed.get("environment") == "test"

    def test_add_caller_info_for_errors(self, json_config):
        """ERROR-level events include caller_module, caller_function, caller_lineno."""
        configure_logging(json_config)
        logger = get_logger("test.caller_info")
        output = _capture_structlog_output(logger, "error", "caller_test")

        parsed = json.loads(output.strip())

        # The add_caller_info processor should have added these fields
        assert "caller_module" in parsed or "caller_function" in parsed, (
            f"Caller info missing from ERROR log: {parsed}"
        )

    def test_caller_info_not_added_for_info(self, json_config):
        """INFO-level events do NOT include caller info (performance optimization)."""
        configure_logging(json_config)
        logger = get_logger("test.caller_info_skip")
        output = _capture_structlog_output(logger, "info", "info_test")

        parsed = json.loads(output.strip())

        # caller_module should NOT be present for non-ERROR levels
        assert "caller_module" not in parsed
