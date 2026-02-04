"""
Log Formatters - INFRA-009

Provides stdlib :class:`logging.Formatter` implementations for structured
JSON output (production) and colorized console output (development). These
formatters are used by the stdlib logging integration layer so that log
records produced by both structlog and traditional ``logging.getLogger()``
callers share the same output format.

Classes:
    - JsonFormatter: Produces single-line JSON log records for machine parsing.
    - ConsoleFormatter: Produces colorized, human-readable console output.

Functions:
    - get_formatter: Factory that returns the appropriate formatter by name.

Example:
    >>> import logging
    >>> from greenlang.infrastructure.logging.formatters import get_formatter
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(get_formatter("json"))
    >>> logging.getLogger().addHandler(handler)
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# ANSI Color Codes
# ---------------------------------------------------------------------------

class _AnsiColors:
    """ANSI escape sequences for terminal colorization."""

    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"

    CYAN: str = "\033[36m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    RED: str = "\033[31m"
    RED_BOLD: str = "\033[1;31m"
    BLUE: str = "\033[34m"
    MAGENTA: str = "\033[35m"
    GRAY: str = "\033[90m"


# Level name -> color mapping
_LEVEL_COLORS: dict[str, str] = {
    "DEBUG": _AnsiColors.CYAN,
    "INFO": _AnsiColors.GREEN,
    "WARNING": _AnsiColors.YELLOW,
    "ERROR": _AnsiColors.RED,
    "CRITICAL": _AnsiColors.RED_BOLD,
}


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Produces a JSON object per log record containing: ``timestamp`` (ISO 8601
    UTC), ``level``, ``logger``, ``message``, ``module``, ``function``,
    ``lineno``, and any extra fields added by structlog processors or
    stdlib ``extra`` kwargs.

    If the record contains exception info, it is serialized into an
    ``exception`` sub-object with ``type``, ``message``, and ``traceback``
    fields.

    Example output::

        {"timestamp":"2026-01-15T08:30:00.123456Z","level":"INFO",
         "logger":"greenlang.agents.emission","message":"Calculation complete",
         "request_id":"req-abc","duration_ms":42.5}
    """

    # Keys that belong to the stdlib LogRecord and should not be copied
    # into the structured output as "extra" fields.
    _STDLIB_KEYS: frozenset[str] = frozenset({
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "pathname", "thread", "threadName",
        "process", "processName", "levelname", "levelno", "message",
        "msecs", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The stdlib log record.

        Returns:
            A single-line JSON string.
        """
        # Ensure record.message is populated
        record.message = record.getMessage()

        # Build the base log dict
        log_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "function": record.funcName,
            "lineno": record.lineno,
        }

        # Merge extra fields (added by structlog or logging extra={})
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._STDLIB_KEYS:
                continue
            log_dict[key] = value

        # Serialize exception info
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            log_dict["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": (
                    traceback.format_exception(exc_type, exc_value, exc_tb)
                    if exc_tb
                    else None
                ),
            }

        # Serialize stack info
        if record.stack_info:
            log_dict["stack_info"] = record.stack_info

        return json.dumps(log_dict, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Console Formatter
# ---------------------------------------------------------------------------


class ConsoleFormatter(logging.Formatter):
    """Formats log records with ANSI colors for terminal readability.

    Output format::

        2026-01-15T08:30:00.123Z INFO  [greenlang.agents.emission] Calculation complete  request_id=req-abc duration_ms=42.5

    Level names are colorized:
        - DEBUG: cyan
        - INFO: green
        - WARNING: yellow
        - ERROR: red
        - CRITICAL: red + bold

    Timestamps are dimmed, logger names are shown in brackets, and extra
    key-value pairs are appended in ``key=value`` format.
    """

    _STDLIB_KEYS: frozenset[str] = frozenset({
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "pathname", "thread", "threadName",
        "process", "processName", "levelname", "levelno", "message",
        "msecs", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with ANSI colors.

        Args:
            record: The stdlib log record.

        Returns:
            A colorized string suitable for terminal output.
        """
        record.message = record.getMessage()

        # Timestamp (dimmed)
        ts = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        ts_str = f"{_AnsiColors.DIM}{ts}{_AnsiColors.RESET}"

        # Level (colorized, padded to 8 chars)
        level_color = _LEVEL_COLORS.get(record.levelname, _AnsiColors.RESET)
        level_str = f"{level_color}{record.levelname:<8}{_AnsiColors.RESET}"

        # Logger name (blue, in brackets)
        logger_str = f"{_AnsiColors.BLUE}[{record.name}]{_AnsiColors.RESET}"

        # Message
        message_str = record.message

        # Extra fields (gray key=value pairs)
        extras: list[str] = []
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._STDLIB_KEYS:
                continue
            extras.append(
                f"{_AnsiColors.GRAY}{key}{_AnsiColors.RESET}"
                f"={_AnsiColors.MAGENTA}{value}{_AnsiColors.RESET}"
            )
        extras_str = "  " + " ".join(extras) if extras else ""

        # Assemble the line
        line = f"{ts_str} {level_str} {logger_str} {message_str}{extras_str}"

        # Append exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            exc_text = self.formatException(record.exc_info)
            line += f"\n{_AnsiColors.RED}{exc_text}{_AnsiColors.RESET}"

        # Append stack info if present
        if record.stack_info:
            line += f"\n{_AnsiColors.DIM}{record.stack_info}{_AnsiColors.RESET}"

        return line


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_formatter(format_type: str) -> logging.Formatter:
    """Return the appropriate formatter for the given format type.

    Args:
        format_type: Either ``"json"`` or ``"console"``.

    Returns:
        A :class:`JsonFormatter` for ``"json"`` or a :class:`ConsoleFormatter`
        for ``"console"``.

    Raises:
        ValueError: If ``format_type`` is not recognized.

    Example:
        >>> formatter = get_formatter("json")
        >>> isinstance(formatter, JsonFormatter)
        True
    """
    if format_type == "json":
        return JsonFormatter()
    if format_type == "console":
        return ConsoleFormatter()
    raise ValueError(
        f"Unknown format type '{format_type}'. Expected 'json' or 'console'."
    )
