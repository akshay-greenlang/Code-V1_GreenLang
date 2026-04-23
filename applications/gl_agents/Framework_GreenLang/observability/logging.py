"""
GreenLang Framework - Structured Logging Module
================================================

Provides JSON-structured logging with correlation ID injection, log level
configuration, log aggregation support, and sensitive data redaction.

Features:
- JSON-formatted log output for ELK/Loki ingestion
- Correlation ID injection for request tracing
- Configurable log levels per module
- Sensitive data redaction (PII, secrets, API keys)
- Context propagation across async boundaries
- OpenTelemetry trace context integration
- Performance-optimized with lazy formatting

Example:
    >>> from greenlang_observability.logging import StructuredLogger, get_logger
    >>>
    >>> logger = get_logger("gl-006-heatreclaim")
    >>> logger.info("Calculation started", agent_id="GL-006", calculation_type="pinch")
    >>>
    >>> # Output:
    >>> # {"timestamp": "2024-01-15T10:30:45.123Z", "level": "INFO",
    >>> #  "message": "Calculation started", "agent_id": "GL-006",
    >>> #  "calculation_type": "pinch", "correlation_id": "abc-123"}

Version: 1.0.0
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Pattern,
    Set,
    TextIO,
    Tuple,
    Union,
)

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for additional context
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


class LogLevel(Enum):
    """Log levels with numeric values."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, level: str) -> LogLevel:
        """Parse log level from string."""
        level_map = {
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "warn": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
            "fatal": cls.CRITICAL,
        }
        return level_map.get(level.lower(), cls.INFO)


@dataclass
class LogConfig:
    """
    Configuration for structured logging.

    Attributes:
        level: Minimum log level
        format: Output format (json, text)
        include_timestamp: Include ISO8601 timestamp
        include_level: Include log level
        include_logger: Include logger name
        include_location: Include file:line location
        include_correlation_id: Include correlation ID
        include_trace_context: Include OpenTelemetry trace context
        redact_patterns: Regex patterns for sensitive data redaction
        redact_fields: Field names to always redact
        output: Output stream (stdout, stderr, or file path)
        max_message_length: Maximum message length (0 for unlimited)
        pretty_print: Pretty print JSON (for development)
    """

    level: LogLevel = LogLevel.INFO
    format: str = "json"
    include_timestamp: bool = True
    include_level: bool = True
    include_logger: bool = True
    include_location: bool = False
    include_correlation_id: bool = True
    include_trace_context: bool = True
    redact_patterns: List[str] = field(default_factory=list)
    redact_fields: Set[str] = field(default_factory=set)
    output: str = "stdout"
    max_message_length: int = 0
    pretty_print: bool = False

    def __post_init__(self) -> None:
        """Initialize default redaction patterns."""
        if not self.redact_patterns:
            self.redact_patterns = [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
                r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",  # SSN
                r"\b[A-Za-z0-9]{32,}\b",  # API keys (long alphanumeric)
                r"(?i)(password|secret|token|api[_-]?key|auth)['\"]?\s*[:=]\s*['\"]?[^'\"\s]+",
            ]

        if not self.redact_fields:
            self.redact_fields = {
                "password",
                "secret",
                "token",
                "api_key",
                "apikey",
                "auth",
                "authorization",
                "credential",
                "private_key",
                "access_token",
                "refresh_token",
                "ssn",
                "social_security",
                "credit_card",
                "card_number",
            }

    @classmethod
    def from_env(cls) -> LogConfig:
        """Create configuration from environment variables."""
        return cls(
            level=LogLevel.from_string(os.getenv("LOG_LEVEL", "INFO")),
            format=os.getenv("LOG_FORMAT", "json"),
            include_timestamp=os.getenv("LOG_TIMESTAMP", "true").lower() == "true",
            include_level=os.getenv("LOG_INCLUDE_LEVEL", "true").lower() == "true",
            include_logger=os.getenv("LOG_INCLUDE_LOGGER", "true").lower() == "true",
            include_location=os.getenv("LOG_INCLUDE_LOCATION", "false").lower() == "true",
            include_correlation_id=os.getenv(
                "LOG_INCLUDE_CORRELATION_ID", "true"
            ).lower() == "true",
            output=os.getenv("LOG_OUTPUT", "stdout"),
            pretty_print=os.getenv("LOG_PRETTY", "false").lower() == "true",
        )


class SensitiveDataRedactor:
    """
    Redacts sensitive data from log messages and structured fields.

    Implements GDPR-compliant PII redaction with configurable patterns.
    """

    REDACTED_VALUE = "[REDACTED]"

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        fields: Optional[Set[str]] = None,
    ):
        """Initialize redactor with patterns and field names."""
        self.patterns: List[Pattern[str]] = []
        self.fields: Set[str] = fields or set()

        if patterns:
            for pattern in patterns:
                try:
                    self.patterns.append(re.compile(pattern))
                except re.error as e:
                    logging.warning(f"Invalid redaction pattern '{pattern}': {e}")

    def redact_string(self, value: str) -> str:
        """Redact sensitive patterns from a string."""
        result = value
        for pattern in self.patterns:
            result = pattern.sub(self.REDACTED_VALUE, result)
        return result

    def redact_dict(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Recursively redact sensitive fields from a dictionary."""
        if depth > 10:  # Prevent infinite recursion
            return data

        result: Dict[str, Any] = {}
        for key, value in data.items():
            lower_key = key.lower()

            # Check if field name should be redacted
            if lower_key in self.fields:
                result[key] = self.REDACTED_VALUE
            elif isinstance(value, str):
                result[key] = self.redact_string(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, depth + 1)
            elif isinstance(value, list):
                result[key] = self._redact_list(value, depth + 1)
            else:
                result[key] = value

        return result

    def _redact_list(self, data: List[Any], depth: int) -> List[Any]:
        """Redact sensitive data from a list."""
        result: List[Any] = []
        for item in data:
            if isinstance(item, str):
                result.append(self.redact_string(item))
            elif isinstance(item, dict):
                result.append(self.redact_dict(item, depth))
            elif isinstance(item, list):
                result.append(self._redact_list(item, depth))
            else:
                result.append(item)
        return result


class CorrelationContext:
    """
    Context manager for correlation ID propagation.

    Automatically generates or uses provided correlation ID for log tracing.

    Example:
        >>> with CorrelationContext("request-123") as ctx:
        ...     logger.info("Processing request")  # Includes correlation_id
        ...     do_work()
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize correlation context."""
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.extra_context = extra_context or {}
        self._token: Optional[contextvars.Token[Optional[str]]] = None
        self._context_token: Optional[contextvars.Token[Dict[str, Any]]] = None

    def __enter__(self) -> CorrelationContext:
        """Enter context and set correlation ID."""
        self._token = _correlation_id.set(self.correlation_id)

        current_context = _log_context.get().copy()
        current_context.update(self.extra_context)
        self._context_token = _log_context.set(current_context)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous correlation ID."""
        if self._token:
            _correlation_id.reset(self._token)
        if self._context_token:
            _log_context.reset(self._context_token)


def set_correlation_id(correlation_id: str) -> contextvars.Token[Optional[str]]:
    """Set the current correlation ID."""
    return _correlation_id.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def add_log_context(key: str, value: Any) -> None:
    """Add a key-value pair to the current log context."""
    current = _log_context.get().copy()
    current[key] = value
    _log_context.set(current)


def clear_log_context() -> None:
    """Clear the current log context."""
    _log_context.set({})


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces JSON-formatted log lines suitable for ELK, Loki, or other
    log aggregation systems.
    """

    def __init__(
        self,
        config: LogConfig,
        redactor: Optional[SensitiveDataRedactor] = None,
    ):
        """Initialize formatter."""
        super().__init__()
        self.config = config
        self.redactor = redactor or SensitiveDataRedactor(
            patterns=config.redact_patterns,
            fields=config.redact_fields,
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_entry: Dict[str, Any] = {}

        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.config.include_level:
            log_entry["level"] = record.levelname

        if self.config.include_logger:
            log_entry["logger"] = record.name

        # Add message
        message = record.getMessage()
        if self.config.max_message_length > 0:
            message = message[: self.config.max_message_length]
        log_entry["message"] = message

        if self.config.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add correlation ID
        if self.config.include_correlation_id:
            correlation_id = get_correlation_id()
            if correlation_id:
                log_entry["correlation_id"] = correlation_id

        # Add trace context if available
        if self.config.include_trace_context:
            try:
                from .tracing import get_current_trace_id, get_current_span

                trace_id = get_current_trace_id()
                span = get_current_span()
                if trace_id:
                    log_entry["trace_id"] = trace_id
                if span:
                    log_entry["span_id"] = span.span_id
            except ImportError:
                pass

        # Add context from contextvars
        context = _log_context.get()
        if context:
            log_entry.update(context)

        # Add extra fields from record
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "traceback": self.formatException(record.exc_info),
            }

        # Redact sensitive data
        log_entry = self.redactor.redact_dict(log_entry)

        # Serialize to JSON
        if self.config.pretty_print:
            return json.dumps(log_entry, indent=2, default=str)
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.

    Produces colored, readable log output for local development.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, config: LogConfig, use_colors: bool = True):
        """Initialize formatter."""
        super().__init__()
        self.config = config
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        parts: List[str] = []

        if self.config.include_timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            parts.append(f"[{timestamp}]")

        if self.config.include_level:
            level = record.levelname
            if self.use_colors:
                color = self.COLORS.get(level, "")
                parts.append(f"{color}[{level:8}]{self.RESET}")
            else:
                parts.append(f"[{level:8}]")

        if self.config.include_logger:
            parts.append(f"[{record.name}]")

        # Add correlation ID
        correlation_id = get_correlation_id()
        if self.config.include_correlation_id and correlation_id:
            parts.append(f"[{correlation_id[:8]}]")

        parts.append(record.getMessage())

        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            parts.append(f"| {extra_str}")

        # Add exception
        if record.exc_info:
            parts.append(f"\n{self.formatException(record.exc_info)}")

        return " ".join(parts)


class StructuredLogger:
    """
    Structured logger for GreenLang agents.

    Provides JSON-structured logging with context propagation,
    sensitive data redaction, and log aggregation support.

    Example:
        >>> logger = StructuredLogger("gl-006-heatreclaim")
        >>> logger.info("Calculation started", agent_id="GL-006", inputs_count=5)
        >>> logger.error("Calculation failed", error="Division by zero", exc_info=True)
    """

    def __init__(
        self,
        name: str,
        config: Optional[LogConfig] = None,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically service/agent name)
            config: Logging configuration
        """
        self.name = name
        self.config = config or LogConfig.from_env()
        self._logger = logging.getLogger(name)
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure the underlying Python logger."""
        self._logger.setLevel(self.config.level.value)
        self._logger.handlers.clear()

        # Create handler based on output config
        if self.config.output == "stdout":
            handler: logging.Handler = logging.StreamHandler(sys.stdout)
        elif self.config.output == "stderr":
            handler = logging.StreamHandler(sys.stderr)
        else:
            handler = logging.FileHandler(self.config.output)

        # Set formatter based on format config
        if self.config.format == "json":
            formatter: logging.Formatter = JSONFormatter(self.config)
        else:
            formatter = TextFormatter(self.config)

        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Prevent propagation to root logger
        self._logger.propagate = False

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Internal logging method."""
        # Create log record with extra fields
        extra = {"extra": kwargs}
        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(
        self,
        message: str,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(
        self,
        message: str,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, exc_info=True, **kwargs)

    def bind(self, **kwargs: Any) -> BoundLogger:
        """Create a bound logger with preset fields."""
        return BoundLogger(self, kwargs)

    def with_correlation_id(self, correlation_id: str) -> CorrelationContext:
        """Create a correlation context."""
        return CorrelationContext(correlation_id)


class BoundLogger:
    """
    Logger with preset fields bound to every log call.

    Example:
        >>> bound = logger.bind(agent_id="GL-006", request_id="abc")
        >>> bound.info("Processing")  # Includes agent_id and request_id
    """

    def __init__(
        self,
        parent: StructuredLogger,
        bound_fields: Dict[str, Any],
    ):
        """Initialize bound logger."""
        self._parent = parent
        self._bound_fields = bound_fields

    def _merge_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge bound fields with kwargs."""
        result = self._bound_fields.copy()
        result.update(kwargs)
        return result

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._parent.debug(message, **self._merge_kwargs(kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._parent.info(message, **self._merge_kwargs(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._parent.warning(message, **self._merge_kwargs(kwargs))

    def error(
        self,
        message: str,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        self._parent.error(message, exc_info=exc_info, **self._merge_kwargs(kwargs))

    def critical(
        self,
        message: str,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log critical message."""
        self._parent.critical(message, exc_info=exc_info, **self._merge_kwargs(kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._parent.exception(message, **self._merge_kwargs(kwargs))

    def bind(self, **kwargs: Any) -> BoundLogger:
        """Create a new bound logger with additional fields."""
        new_fields = self._bound_fields.copy()
        new_fields.update(kwargs)
        return BoundLogger(self._parent, new_fields)


# Global logger cache
_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()


def get_logger(
    name: str,
    config: Optional[LogConfig] = None,
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name
        config: Optional configuration (uses defaults if not provided)

    Returns:
        StructuredLogger instance
    """
    with _loggers_lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, config)
        return _loggers[name]


def configure_root_logger(config: Optional[LogConfig] = None) -> None:
    """
    Configure the root logger with structured logging.

    Affects all loggers that don't have explicit configuration.
    """
    config = config or LogConfig.from_env()
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level.value)
    root_logger.handlers.clear()

    if config.output == "stdout":
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
    elif config.output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(config.output)

    if config.format == "json":
        formatter: logging.Formatter = JSONFormatter(config)
    else:
        formatter = TextFormatter(config)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def redact_sensitive_data(
    data: Union[str, Dict[str, Any]],
    patterns: Optional[List[str]] = None,
    fields: Optional[Set[str]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Redact sensitive data from a string or dictionary.

    Args:
        data: String or dictionary to redact
        patterns: Regex patterns for sensitive data
        fields: Field names to always redact

    Returns:
        Redacted data
    """
    redactor = SensitiveDataRedactor(patterns=patterns, fields=fields)

    if isinstance(data, str):
        return redactor.redact_string(data)
    elif isinstance(data, dict):
        return redactor.redact_dict(data)
    else:
        return data
