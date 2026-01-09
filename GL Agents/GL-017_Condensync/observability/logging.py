# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Structured Logging Module
==============================================

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
- Condenser-specific context fields

Example:
    >>> from observability.logging import StructuredLogger, get_logger
    >>>
    >>> logger = get_logger("gl-017-condensync")
    >>> logger.info(
    ...     "Cleanliness factor calculated",
    ...     condenser_id="COND-001",
    ...     cf_value=0.85,
    ...     calculation_time_ms=12.5
    ... )
    >>>
    >>> # Output:
    >>> # {"timestamp": "2025-01-15T10:30:45.123Z", "level": "INFO",
    >>> #  "message": "Cleanliness factor calculated", "condenser_id": "COND-001",
    >>> #  "cf_value": 0.85, "calculation_time_ms": 12.5, "correlation_id": "abc-123"}

Author: GL-BackendDeveloper
Date: December 2025
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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Union,
)

# Context variables for correlation and logging context
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

_condenser_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "condenser_id", default=None
)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

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


class LogFormat(str, Enum):
    """Supported log output formats."""
    JSON = "json"
    TEXT = "text"


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
        include_condenser_id: Include condenser ID from context
        redact_patterns: Regex patterns for sensitive data redaction
        redact_fields: Field names to always redact
        output: Output stream (stdout, stderr, or file path)
        max_message_length: Maximum message length (0 for unlimited)
        pretty_print: Pretty print JSON (for development)
        service_name: Service name to include in logs
    """

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    include_timestamp: bool = True
    include_level: bool = True
    include_logger: bool = True
    include_location: bool = False
    include_correlation_id: bool = True
    include_trace_context: bool = True
    include_condenser_id: bool = True
    redact_patterns: List[str] = field(default_factory=list)
    redact_fields: Set[str] = field(default_factory=set)
    output: str = "stdout"
    max_message_length: int = 0
    pretty_print: bool = False
    service_name: str = "gl-017-condensync"

    def __post_init__(self) -> None:
        """Initialize default redaction patterns."""
        if not self.redact_patterns:
            self.redact_patterns = [
                # Email addresses
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                # Phone numbers
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                # SSN
                r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
                # API keys (long alphanumeric strings)
                r"\b[A-Za-z0-9]{32,}\b",
                # Password patterns
                r"(?i)(password|secret|token|api[_-]?key|auth)['\"]?\s*[:=]\s*['\"]?[^'\"\s]+",
                # Credit card patterns
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
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
                "opc_ua_password",
                "cmms_api_key",
                "pi_server_token",
            }

    @classmethod
    def from_env(cls) -> LogConfig:
        """Create configuration from environment variables."""
        return cls(
            level=LogLevel.from_string(os.getenv("LOG_LEVEL", "INFO")),
            format=LogFormat(os.getenv("LOG_FORMAT", "json").lower()),
            include_timestamp=os.getenv("LOG_TIMESTAMP", "true").lower() == "true",
            include_level=os.getenv("LOG_INCLUDE_LEVEL", "true").lower() == "true",
            include_logger=os.getenv("LOG_INCLUDE_LOGGER", "true").lower() == "true",
            include_location=os.getenv("LOG_INCLUDE_LOCATION", "false").lower() == "true",
            include_correlation_id=os.getenv(
                "LOG_INCLUDE_CORRELATION_ID", "true"
            ).lower() == "true",
            include_condenser_id=os.getenv(
                "LOG_INCLUDE_CONDENSER_ID", "true"
            ).lower() == "true",
            output=os.getenv("LOG_OUTPUT", "stdout"),
            pretty_print=os.getenv("LOG_PRETTY", "false").lower() == "true",
            service_name=os.getenv("SERVICE_NAME", "gl-017-condensync"),
        )


# =============================================================================
# SENSITIVE DATA REDACTION
# =============================================================================

class SensitiveDataRedactor:
    """
    Redacts sensitive data from log messages and structured fields.

    Implements GDPR-compliant PII redaction with configurable patterns.
    Thread-safe implementation with compiled regex caching.

    Example:
        >>> redactor = SensitiveDataRedactor()
        >>> redacted = redactor.redact_string("Email: user@example.com")
        >>> assert "user@example.com" not in redacted
    """

    REDACTED_VALUE = "[REDACTED]"

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        fields: Optional[Set[str]] = None,
    ):
        """
        Initialize redactor with patterns and field names.

        Args:
            patterns: Regex patterns for sensitive data
            fields: Field names to always redact
        """
        self.patterns: List[Pattern[str]] = []
        self.fields: Set[str] = fields or set()

        if patterns:
            for pattern in patterns:
                try:
                    self.patterns.append(re.compile(pattern))
                except re.error as e:
                    logging.warning(f"Invalid redaction pattern '{pattern}': {e}")

    def redact_string(self, value: str) -> str:
        """
        Redact sensitive patterns from a string.

        Args:
            value: String to redact

        Returns:
            Redacted string
        """
        result = value
        for pattern in self.patterns:
            result = pattern.sub(self.REDACTED_VALUE, result)
        return result

    def redact_dict(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """
        Recursively redact sensitive fields from a dictionary.

        Args:
            data: Dictionary to redact
            depth: Current recursion depth (for protection)

        Returns:
            Redacted dictionary
        """
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
        """
        Redact sensitive data from a list.

        Args:
            data: List to redact
            depth: Current recursion depth

        Returns:
            Redacted list
        """
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


# =============================================================================
# CORRELATION CONTEXT
# =============================================================================

class CorrelationContext:
    """
    Context manager for correlation ID propagation.

    Automatically generates or uses provided correlation ID for log tracing.
    Supports nesting and async context propagation.

    Example:
        >>> with CorrelationContext("request-123") as ctx:
        ...     logger.info("Processing request")  # Includes correlation_id
        ...     do_work()
        >>>
        >>> # With condenser context
        >>> with CorrelationContext(condenser_id="COND-001") as ctx:
        ...     logger.info("Processing condenser")
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        condenser_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize correlation context.

        Args:
            correlation_id: Correlation ID (generated if not provided)
            condenser_id: Optional condenser ID for context
            extra_context: Additional context key-value pairs
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.condenser_id = condenser_id
        self.extra_context = extra_context or {}

        self._corr_token: Optional[contextvars.Token[Optional[str]]] = None
        self._context_token: Optional[contextvars.Token[Dict[str, Any]]] = None
        self._condenser_token: Optional[contextvars.Token[Optional[str]]] = None

    def __enter__(self) -> CorrelationContext:
        """Enter context and set correlation ID."""
        self._corr_token = _correlation_id.set(self.correlation_id)

        current_context = _log_context.get().copy()
        current_context.update(self.extra_context)
        self._context_token = _log_context.set(current_context)

        if self.condenser_id:
            self._condenser_token = _condenser_context.set(self.condenser_id)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous values."""
        if self._corr_token:
            _correlation_id.reset(self._corr_token)
        if self._context_token:
            _log_context.reset(self._context_token)
        if self._condenser_token:
            _condenser_context.reset(self._condenser_token)


class CondenserLogContext:
    """
    Context manager specifically for condenser processing.

    Adds condenser-specific context to all logs within the context.

    Example:
        >>> with CondenserLogContext("COND-001", unit="Unit-1") as ctx:
        ...     logger.info("Calculating CF")  # Includes condenser_id and unit
    """

    def __init__(
        self,
        condenser_id: str,
        unit: Optional[str] = None,
        **extra_context: Any,
    ):
        """
        Initialize condenser log context.

        Args:
            condenser_id: Condenser identifier
            unit: Optional unit identifier
            **extra_context: Additional context fields
        """
        self.condenser_id = condenser_id
        self.unit = unit
        self.extra_context = extra_context

        self._condenser_token: Optional[contextvars.Token[Optional[str]]] = None
        self._context_token: Optional[contextvars.Token[Dict[str, Any]]] = None

    def __enter__(self) -> CondenserLogContext:
        """Enter context."""
        self._condenser_token = _condenser_context.set(self.condenser_id)

        current_context = _log_context.get().copy()
        current_context["condenser_id"] = self.condenser_id
        if self.unit:
            current_context["unit"] = self.unit
        current_context.update(self.extra_context)
        self._context_token = _log_context.set(current_context)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context."""
        if self._condenser_token:
            _condenser_context.reset(self._condenser_token)
        if self._context_token:
            _log_context.reset(self._context_token)


# =============================================================================
# CONTEXT FUNCTIONS
# =============================================================================

def set_correlation_id(correlation_id: str) -> contextvars.Token[Optional[str]]:
    """
    Set the current correlation ID.

    Args:
        correlation_id: Correlation ID to set

    Returns:
        Token for resetting
    """
    return _correlation_id.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_condenser_context(condenser_id: str) -> contextvars.Token[Optional[str]]:
    """
    Set the current condenser context.

    Args:
        condenser_id: Condenser ID to set

    Returns:
        Token for resetting
    """
    return _condenser_context.set(condenser_id)


def get_condenser_context() -> Optional[str]:
    """Get the current condenser ID from context."""
    return _condenser_context.get()


def add_log_context(key: str, value: Any) -> None:
    """
    Add a key-value pair to the current log context.

    Args:
        key: Context key
        value: Context value
    """
    current = _log_context.get().copy()
    current[key] = value
    _log_context.set(current)


def clear_log_context() -> None:
    """Clear the current log context."""
    _log_context.set({})


# =============================================================================
# JSON FORMATTER
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces JSON-formatted log lines suitable for ELK, Loki, or other
    log aggregation systems.

    Includes:
    - Timestamp
    - Log level
    - Logger name
    - Message
    - Correlation ID
    - Condenser ID
    - Trace context
    - Extra fields
    - Exception info
    """

    def __init__(
        self,
        config: LogConfig,
        redactor: Optional[SensitiveDataRedactor] = None,
    ):
        """
        Initialize formatter.

        Args:
            config: Log configuration
            redactor: Optional custom redactor
        """
        super().__init__()
        self.config = config
        self.redactor = redactor or SensitiveDataRedactor(
            patterns=config.redact_patterns,
            fields=config.redact_fields,
        )

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON-formatted log string
        """
        # Build base log entry
        log_entry: Dict[str, Any] = {}

        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.config.include_level:
            log_entry["level"] = record.levelname

        if self.config.include_logger:
            log_entry["logger"] = record.name

        # Add service name
        log_entry["service"] = self.config.service_name

        # Add agent ID
        log_entry["agent_id"] = "GL-017"

        # Add message
        message = record.getMessage()
        if self.config.max_message_length > 0:
            message = message[:self.config.max_message_length]
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

        # Add condenser ID from context
        if self.config.include_condenser_id:
            condenser_id = get_condenser_context()
            if condenser_id:
                log_entry["condenser_id"] = condenser_id

        # Add trace context if available
        if self.config.include_trace_context:
            try:
                # Try to get trace context from observability module
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
        if hasattr(record, "extra") and record.extra:
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
        return json.dumps(log_entry, default=str, separators=(",", ":"))


# =============================================================================
# TEXT FORMATTER
# =============================================================================

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
        """
        Initialize formatter.

        Args:
            config: Log configuration
            use_colors: Whether to use ANSI colors
        """
        super().__init__()
        self.config = config
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as human-readable text.

        Args:
            record: Log record

        Returns:
            Formatted log string
        """
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

        # Add condenser ID
        condenser_id = get_condenser_context()
        if self.config.include_condenser_id and condenser_id:
            parts.append(f"[{condenser_id}]")

        parts.append(record.getMessage())

        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            parts.append(f"| {extra_str}")

        # Add exception
        if record.exc_info:
            parts.append(f"\n{self.formatException(record.exc_info)}")

        return " ".join(parts)


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """
    Structured logger for GL-017 CONDENSYNC.

    Provides JSON-structured logging with context propagation,
    sensitive data redaction, and log aggregation support.

    Example:
        >>> logger = StructuredLogger("gl-017-condensync")
        >>> logger.info(
        ...     "Calculation completed",
        ...     condenser_id="COND-001",
        ...     cf_value=0.85,
        ...     calculation_time_ms=12.5
        ... )
        >>>
        >>> # Bound logger with preset fields
        >>> calc_logger = logger.bind(calculation_type="cleanliness_factor")
        >>> calc_logger.info("Starting calculation")
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
        if self.config.format == LogFormat.JSON:
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
        """
        Internal logging method.

        Args:
            level: Log level
            message: Log message
            exc_info: Include exception info
            **kwargs: Extra fields
        """
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
        """
        Create a bound logger with preset fields.

        Args:
            **kwargs: Fields to bind

        Returns:
            BoundLogger instance
        """
        return BoundLogger(self, kwargs)

    def with_correlation_id(
        self,
        correlation_id: Optional[str] = None,
    ) -> CorrelationContext:
        """
        Create a correlation context.

        Args:
            correlation_id: Correlation ID (generated if not provided)

        Returns:
            CorrelationContext for use with 'with' statement
        """
        return CorrelationContext(correlation_id)

    def with_condenser(
        self,
        condenser_id: str,
        unit: Optional[str] = None,
        **extra: Any,
    ) -> CondenserLogContext:
        """
        Create a condenser log context.

        Args:
            condenser_id: Condenser identifier
            unit: Optional unit identifier
            **extra: Additional context fields

        Returns:
            CondenserLogContext for use with 'with' statement
        """
        return CondenserLogContext(condenser_id, unit, **extra)


# =============================================================================
# BOUND LOGGER
# =============================================================================

class BoundLogger:
    """
    Logger with preset fields bound to every log call.

    Example:
        >>> bound = logger.bind(condenser_id="COND-001", calculation_type="cf")
        >>> bound.info("Processing")  # Includes condenser_id and calculation_type
    """

    def __init__(
        self,
        parent: StructuredLogger,
        bound_fields: Dict[str, Any],
    ):
        """
        Initialize bound logger.

        Args:
            parent: Parent StructuredLogger
            bound_fields: Fields to include in all logs
        """
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


# =============================================================================
# GLOBAL LOGGER MANAGEMENT
# =============================================================================

_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()


def get_logger(
    name: str,
    config: Optional[LogConfig] = None,
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Thread-safe singleton pattern per logger name.

    Args:
        name: Logger name
        config: Optional configuration

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

    Args:
        config: Optional configuration
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

    if config.format == LogFormat.JSON:
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


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "StructuredLogger",
    "BoundLogger",
    "LogConfig",
    # Context managers
    "CorrelationContext",
    "CondenserLogContext",
    # Enums
    "LogLevel",
    "LogFormat",
    # Functions
    "get_logger",
    "configure_root_logger",
    "set_correlation_id",
    "get_correlation_id",
    "set_condenser_context",
    "get_condenser_context",
    "add_log_context",
    "clear_log_context",
    "redact_sensitive_data",
    # Formatters
    "JSONFormatter",
    "TextFormatter",
    # Redactor
    "SensitiveDataRedactor",
]
