# -*- coding: utf-8 -*-
"""
Unit Tests for Log Formatters - INFRA-009

Tests JsonFormatter, ConsoleFormatter, and the get_formatter factory function.
Validates JSON output structure, timestamp format, exception serialization,
colour coding for console output, and rejection of unknown format names.

Module under test: greenlang.infrastructure.logging.formatters
"""

import json
import logging
import sys
from datetime import datetime, timezone

import pytest

from greenlang.infrastructure.logging.formatters import (
    ConsoleFormatter,
    JsonFormatter,
    get_formatter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log_record(
    msg: str = "test message",
    level: int = logging.INFO,
    name: str = "test.logger",
    exc_info=None,
    extra: dict = None,
) -> logging.LogRecord:
    """Build a stdlib LogRecord suitable for formatter testing."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test_formatters.py",
        lineno=42,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    # Inject extra fields the way logging.Logger.makeRecord does
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


def _make_exception_record(msg: str = "error occurred") -> logging.LogRecord:
    """Build a LogRecord that carries real exc_info from a caught exception."""
    try:
        raise ValueError("intentional test error")
    except ValueError:
        return _make_log_record(
            msg=msg,
            level=logging.ERROR,
            exc_info=sys.exc_info(),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def json_formatter() -> JsonFormatter:
    return JsonFormatter()


@pytest.fixture
def console_formatter() -> ConsoleFormatter:
    return ConsoleFormatter()


# ---------------------------------------------------------------------------
# JsonFormatter Tests
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_json_formatter_basic(self, json_formatter):
        """A basic INFO record is formatted as valid JSON."""
        record = _make_log_record("hello world")
        output = json_formatter.format(record)
        parsed = json.loads(output)
        assert parsed.get("message") == "hello world"

    def test_json_formatter_includes_timestamp(self, json_formatter):
        """The JSON output contains a timestamp in ISO-8601 format."""
        record = _make_log_record()
        output = json_formatter.format(record)
        parsed = json.loads(output)

        ts = parsed.get("timestamp")
        assert ts is not None, f"No timestamp field in output: {parsed}"
        # Should be parseable as ISO datetime (starts with a year)
        assert ts[:4].isdigit(), f"Timestamp does not start with year: {ts}"

    def test_json_formatter_includes_level(self, json_formatter):
        """The JSON output contains a 'level' field matching the record level."""
        for level_int, level_name in [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]:
            record = _make_log_record(level=level_int)
            output = json_formatter.format(record)
            parsed = json.loads(output)
            assert parsed.get("level") == level_name, (
                f"Expected level={level_name}, got {parsed.get('level')}"
            )

    def test_json_formatter_includes_logger_name(self, json_formatter):
        """The JSON output contains the logger name."""
        record = _make_log_record(name="myapp.module.sub")
        output = json_formatter.format(record)
        parsed = json.loads(output)
        assert parsed.get("logger") == "myapp.module.sub"

    def test_json_formatter_with_exception(self, json_formatter):
        """Exception info is serialized into a structured dict."""
        record = _make_exception_record("something failed")
        output = json_formatter.format(record)
        parsed = json.loads(output)

        exc_data = parsed.get("exception")
        assert exc_data is not None, f"No exception field in output: {parsed}"
        assert exc_data.get("type") == "ValueError"
        assert "intentional test error" in str(exc_data.get("message", ""))
        assert exc_data.get("traceback") is not None

    def test_json_formatter_extra_fields(self, json_formatter):
        """Extra fields on the LogRecord (e.g. request_id) are included in JSON."""
        record = _make_log_record(
            extra={"request_id": "req-42", "tenant_id": "acme"}
        )
        output = json_formatter.format(record)
        parsed = json.loads(output)
        assert parsed.get("request_id") == "req-42"
        assert parsed.get("tenant_id") == "acme"

    def test_json_formatter_unicode(self, json_formatter):
        """Non-ASCII characters are preserved (ensure_ascii=False)."""
        record = _make_log_record("Emissionen: 42 t CO\u2082e -- \u00e9mission")
        output = json_formatter.format(record)
        # ensure_ascii=False means unicode chars are NOT escaped
        assert "\u2082" in output
        assert "\u00e9" in output
        # Must still be valid JSON
        json.loads(output)

    def test_json_formatter_non_serializable(self, json_formatter):
        """Non-serializable objects are converted via default=str without crashing."""
        record = _make_log_record(
            extra={"custom_obj": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)}
        )
        output = json_formatter.format(record)
        parsed = json.loads(output)
        # The datetime should have been stringified somehow
        flat = json.dumps(parsed)
        assert "2025" in flat


# ---------------------------------------------------------------------------
# ConsoleFormatter Tests
# ---------------------------------------------------------------------------


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_console_formatter_basic(self, console_formatter):
        """A basic record is formatted to a non-empty string containing the message."""
        record = _make_log_record("console test")
        output = console_formatter.format(record)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "console test" in output

    def test_console_formatter_error_level_red(self, console_formatter):
        """ERROR-level output contains ANSI red escape code \\033[31m."""
        record = _make_log_record("red error", level=logging.ERROR)
        output = console_formatter.format(record)
        # _AnsiColors.RED = "\033[31m"
        assert "\033[31" in output, (
            f"Expected ANSI red escape code in ERROR output. Got: {output!r}"
        )

    def test_console_formatter_info_level_green(self, console_formatter):
        """INFO-level output contains ANSI green escape code \\033[32m."""
        record = _make_log_record("green info", level=logging.INFO)
        output = console_formatter.format(record)
        # _AnsiColors.GREEN = "\033[32m"
        assert "\033[32" in output, (
            f"Expected ANSI green escape code in INFO output. Got: {output!r}"
        )

    def test_console_formatter_includes_logger_name(self, console_formatter):
        """Logger name appears in brackets in the console output."""
        record = _make_log_record(name="myapp.worker")
        output = console_formatter.format(record)
        # ConsoleFormatter uses [record.name] format
        assert "[myapp.worker]" in output


# ---------------------------------------------------------------------------
# get_formatter Factory Tests
# ---------------------------------------------------------------------------


class TestGetFormatter:
    """Tests for the get_formatter() factory function."""

    def test_get_formatter_json(self):
        """get_formatter('json') returns a JsonFormatter instance."""
        fmt = get_formatter("json")
        assert isinstance(fmt, JsonFormatter)

    def test_get_formatter_console(self):
        """get_formatter('console') returns a ConsoleFormatter instance."""
        fmt = get_formatter("console")
        assert isinstance(fmt, ConsoleFormatter)

    def test_get_formatter_invalid(self):
        """get_formatter('xml') raises ValueError for unknown format names."""
        with pytest.raises(ValueError, match="xml"):
            get_formatter("xml")

    @pytest.mark.parametrize("bad_name", ["JSON", "Console", "TEXT", "yaml"])
    def test_get_formatter_rejects_wrong_case(self, bad_name):
        """get_formatter only accepts lowercase 'json' and 'console'."""
        with pytest.raises(ValueError):
            get_formatter(bad_name)
