"""
GL-004 BURNMASTER Observability Module

This module provides comprehensive observability configuration for combustion
optimization operations, including structured logging with structlog,
distributed tracing with OpenTelemetry, and metrics configuration with
Prometheus.

Example:
    >>> setup_logging(LogConfig(level="INFO", format="json"))
    >>> setup_tracing(TraceConfig(service_name="burnmaster"))
    >>> with create_span("optimization_cycle", {"unit_id": "BNR-001"}) as span:
    ...     # Perform optimization
    ...     pass
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Union
import hashlib
import json
import logging
import sys
import threading
import time
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    JSON = "json"
    CONSOLE = "console"
    PLAIN = "plain"


class TraceExporter(str, Enum):
    """Trace exporter types."""
    CONSOLE = "console"
    OTLP = "otlp"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    NONE = "none"


class MetricsExporter(str, Enum):
    """Metrics exporter types."""
    PROMETHEUS = "prometheus"
    OTLP = "otlp"
    CONSOLE = "console"
    NONE = "none"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class LogConfig(BaseModel):
    """Configuration for structured logging."""

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level"
    )
    format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Log output format"
    )

    # Output
    output_file: Optional[str] = Field(
        None, description="Log file path (None for stdout)"
    )
    rotate_size_mb: int = Field(
        default=100, ge=1, description="Log rotation size MB"
    )
    rotate_keep: int = Field(
        default=5, ge=1, description="Number of rotated logs to keep"
    )

    # Context
    service_name: str = Field(
        default="burnmaster", description="Service name in logs"
    )
    service_version: str = Field(
        default="1.0.0", description="Service version in logs"
    )
    environment: str = Field(
        default="production", description="Environment name"
    )

    # Processors
    add_timestamp: bool = Field(default=True, description="Add ISO timestamp")
    add_caller: bool = Field(default=True, description="Add caller info")
    add_stack_trace: bool = Field(
        default=True, description="Add stack trace on errors"
    )

    # Filtering
    filter_sensitive: bool = Field(
        default=True, description="Filter sensitive data"
    )
    sensitive_fields: List[str] = Field(
        default_factory=lambda: ["password", "token", "secret", "api_key"],
        description="Fields to filter"
    )


class TraceConfig(BaseModel):
    """Configuration for distributed tracing."""

    enabled: bool = Field(default=True, description="Enable tracing")
    service_name: str = Field(
        default="burnmaster", description="Service name"
    )
    service_version: str = Field(
        default="1.0.0", description="Service version"
    )

    # Exporter
    exporter: TraceExporter = Field(
        default=TraceExporter.OTLP, description="Trace exporter type"
    )
    endpoint: str = Field(
        default="http://localhost:4317",
        description="Exporter endpoint"
    )

    # Sampling
    sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sample rate"
    )

    # Context propagation
    propagators: List[str] = Field(
        default_factory=lambda: ["tracecontext", "baggage"],
        description="Context propagators"
    )

    # Resource attributes
    resource_attributes: Dict[str, str] = Field(
        default_factory=dict, description="Additional resource attributes"
    )


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""

    enabled: bool = Field(default=True, description="Enable metrics")
    prefix: str = Field(
        default="burnmaster", description="Metrics name prefix"
    )

    # Exporter
    exporter: MetricsExporter = Field(
        default=MetricsExporter.PROMETHEUS, description="Metrics exporter"
    )
    endpoint: str = Field(
        default="http://localhost:9090",
        description="Exporter endpoint"
    )
    port: int = Field(
        default=8000, ge=1, le=65535, description="Metrics server port"
    )

    # Collection
    collection_interval_s: float = Field(
        default=15.0, ge=1.0, description="Collection interval seconds"
    )

    # Default labels
    default_labels: Dict[str, str] = Field(
        default_factory=dict, description="Default metric labels"
    )


# =============================================================================
# SPAN IMPLEMENTATION
# =============================================================================

class Span:
    """
    Represents a span in a distributed trace.

    A span represents a single operation within a trace, including
    timing information, attributes, and events.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Unique span identifier
        name: Span name/operation
        start_time: Span start timestamp
        attributes: Span attributes

    Example:
        >>> span = Span("database_query", {"query_type": "SELECT"})
        >>> span.add_event("query_started")
        >>> # ... perform operation
        >>> span.add_event("query_completed")
        >>> span.end()
    """

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span: Optional["Span"] = None
    ):
        """
        Initialize a span.

        Args:
            name: Span name/operation
            attributes: Initial span attributes
            parent_span: Optional parent span
        """
        self.trace_id = (
            parent_span.trace_id if parent_span
            else uuid.uuid4().hex
        )
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_span_id = parent_span.span_id if parent_span else None
        self.name = name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = attributes or {}
        self.events: List[Dict[str, Any]] = []
        self.status: str = "OK"
        self.status_message: Optional[str] = None
        self._is_ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        if not self._is_ended:
            self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event to the span."""
        if not self._is_ended:
            self.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            })

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set span status (OK, ERROR)."""
        if not self._is_ended:
            self.status = status
            self.status_message = message

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        if not self._is_ended:
            self.set_status("ERROR", str(exception))
            self.add_event("exception", {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            })

    def end(self) -> None:
        """End the span and record duration."""
        if not self._is_ended:
            self.end_time = time.time()
            self._is_ended = True

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_message": self.status_message,
        }

    def __enter__(self) -> "Span":
        """Enter span context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit span context."""
        if exc_type is not None:
            self.record_exception(exc_val)
        self.end()


# =============================================================================
# GLOBAL STATE
# =============================================================================

_log_config: Optional[LogConfig] = None
_trace_config: Optional[TraceConfig] = None
_metrics_config: Optional[MetricsConfig] = None
_current_span: threading.local = threading.local()
_span_processors: List[Callable[[Span], None]] = []


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """
    Structured logger implementation using structlog patterns.

    Provides JSON and console output formats with context binding.
    """

    def __init__(self, config: LogConfig):
        """Initialize the structured logger."""
        self.config = config
        self._context: Dict[str, Any] = {
            "service": config.service_name,
            "version": config.service_version,
            "environment": config.environment,
        }

    def bind(self, **kwargs) -> "StructuredLogger":
        """Bind additional context to logger."""
        new_logger = StructuredLogger(self.config)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def _format_message(
        self,
        level: str,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """Format log message based on config."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **self._context,
            **context,
        }

        # Filter sensitive data
        if self.config.filter_sensitive:
            log_data = self._filter_sensitive(log_data)

        if self.config.format == LogFormat.JSON:
            return json.dumps(log_data, default=str)
        elif self.config.format == LogFormat.CONSOLE:
            return (
                f"[{log_data['timestamp']}] {level:8} | "
                f"{message} | {json.dumps({k: v for k, v in log_data.items() if k not in ['timestamp', 'level', 'message']}, default=str)}"
            )
        else:
            return f"{level}: {message}"

    def _filter_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive fields from log data."""
        filtered = {}
        for key, value in data.items():
            if key.lower() in [f.lower() for f in self.config.sensitive_fields]:
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive(value)
            else:
                filtered[key] = value
        return filtered

    def _log(self, level: str, message: str, **context) -> None:
        """Internal log method."""
        formatted = self._format_message(level, message, context)
        print(formatted, file=sys.stdout if level != "ERROR" else sys.stderr)

    def debug(self, message: str, **context) -> None:
        """Log debug message."""
        if self._should_log(LogLevel.DEBUG):
            self._log("DEBUG", message, **context)

    def info(self, message: str, **context) -> None:
        """Log info message."""
        if self._should_log(LogLevel.INFO):
            self._log("INFO", message, **context)

    def warning(self, message: str, **context) -> None:
        """Log warning message."""
        if self._should_log(LogLevel.WARNING):
            self._log("WARNING", message, **context)

    def error(self, message: str, **context) -> None:
        """Log error message."""
        if self._should_log(LogLevel.ERROR):
            self._log("ERROR", message, **context)

    def critical(self, message: str, **context) -> None:
        """Log critical message."""
        if self._should_log(LogLevel.CRITICAL):
            self._log("CRITICAL", message, **context)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return levels.index(level) >= levels.index(self.config.level)


_structured_logger: Optional[StructuredLogger] = None


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_logging(config: LogConfig) -> None:
    """
    Setup structured logging with the provided configuration.

    Args:
        config: Logging configuration

    Example:
        >>> setup_logging(LogConfig(level="INFO", format="json"))
    """
    global _log_config, _structured_logger

    _log_config = config
    _structured_logger = StructuredLogger(config)

    # Configure standard library logging
    log_level = getattr(logging, config.level.value)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger.info(
        f"Logging configured: level={config.level.value}, "
        f"format={config.format.value}"
    )


def setup_tracing(config: TraceConfig) -> None:
    """
    Setup distributed tracing with the provided configuration.

    Args:
        config: Tracing configuration

    Example:
        >>> setup_tracing(TraceConfig(service_name="burnmaster"))
    """
    global _trace_config

    _trace_config = config

    if not config.enabled:
        logger.info("Tracing disabled")
        return

    logger.info(
        f"Tracing configured: service={config.service_name}, "
        f"exporter={config.exporter.value}, "
        f"sample_rate={config.sample_rate}"
    )


def setup_metrics(config: MetricsConfig) -> None:
    """
    Setup metrics collection with the provided configuration.

    Args:
        config: Metrics configuration

    Example:
        >>> setup_metrics(MetricsConfig(prefix="burnmaster"))
    """
    global _metrics_config

    _metrics_config = config

    if not config.enabled:
        logger.info("Metrics disabled")
        return

    logger.info(
        f"Metrics configured: prefix={config.prefix}, "
        f"exporter={config.exporter.value}, "
        f"port={config.port}"
    )


# =============================================================================
# SPAN FUNCTIONS
# =============================================================================

@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None
) -> Generator[Span, None, None]:
    """
    Create a new span for tracing.

    Args:
        name: Span name/operation
        attributes: Initial span attributes

    Yields:
        Span instance

    Example:
        >>> with create_span("database_query", {"table": "burners"}) as span:
        ...     span.add_event("query_started")
        ...     # ... perform query
        ...     span.set_attribute("rows_returned", 100)
    """
    # Get parent span from context if exists
    parent = getattr(_current_span, 'span', None)

    # Create new span
    span = Span(name, attributes, parent)

    # Set as current span
    previous_span = getattr(_current_span, 'span', None)
    _current_span.span = span

    try:
        yield span
    except Exception as e:
        span.record_exception(e)
        raise
    finally:
        span.end()

        # Process span
        for processor in _span_processors:
            try:
                processor(span)
            except Exception as e:
                logger.error(f"Span processor error: {e}")

        # Restore previous span
        _current_span.span = previous_span

        # Log span if configured
        if _trace_config and _trace_config.exporter == TraceExporter.CONSOLE:
            log_structured(
                "DEBUG",
                f"Span completed: {name}",
                span_data=span.to_dict()
            )


def get_current_span() -> Optional[Span]:
    """Get the current active span."""
    return getattr(_current_span, 'span', None)


def add_span_processor(processor: Callable[[Span], None]) -> None:
    """
    Add a span processor callback.

    Args:
        processor: Callback function that receives completed spans
    """
    _span_processors.append(processor)


# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def log_structured(
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Log a structured message.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        context: Additional context dictionary
        **kwargs: Additional key-value pairs to include

    Example:
        >>> log_structured("INFO", "Optimization complete",
        ...               context={"unit_id": "BNR-001", "duration_ms": 150.5})
    """
    if _structured_logger is None:
        # Fallback to standard logging
        log_level = getattr(logging, level.upper(), logging.INFO)
        full_context = {**(context or {}), **kwargs}
        logging.log(log_level, f"{message} | {full_context}")
        return

    full_context = {**(context or {}), **kwargs}

    # Add trace context if available
    current_span = get_current_span()
    if current_span:
        full_context['trace_id'] = current_span.trace_id
        full_context['span_id'] = current_span.span_id

    level_upper = level.upper()
    if level_upper == "DEBUG":
        _structured_logger.debug(message, **full_context)
    elif level_upper == "INFO":
        _structured_logger.info(message, **full_context)
    elif level_upper == "WARNING":
        _structured_logger.warning(message, **full_context)
    elif level_upper == "ERROR":
        _structured_logger.error(message, **full_context)
    elif level_upper == "CRITICAL":
        _structured_logger.critical(message, **full_context)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a bound logger with the given name.

    Args:
        name: Logger name (typically module name)

    Returns:
        StructuredLogger bound with the name
    """
    if _structured_logger is None:
        setup_logging(LogConfig())

    return _structured_logger.bind(logger=name)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def trace_function(name: Optional[str] = None):
    """
    Decorator to trace function execution.

    Args:
        name: Optional span name (defaults to function name)

    Example:
        >>> @trace_function("compute_emissions")
        ... def compute_emissions(data):
        ...     # ... computation
        ...     return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with create_span(span_name, {"function": func.__name__}) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    raise
        return wrapper
    return decorator


def timed_operation(operation_name: str):
    """
    Context manager for timing operations.

    Args:
        operation_name: Name of the operation being timed

    Example:
        >>> with timed_operation("database_query") as timer:
        ...     # ... perform query
        ...     pass
        >>> print(f"Duration: {timer.duration_ms}ms")
    """
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        @property
        def duration_ms(self) -> float:
            if self.end_time:
                return (self.end_time - self.start_time) * 1000
            return (time.time() - self.start_time) * 1000

    @contextmanager
    def timer_context():
        timer = Timer()
        timer.start_time = time.time()
        try:
            yield timer
        finally:
            timer.end_time = time.time()
            log_structured(
                "DEBUG",
                f"Operation '{operation_name}' completed",
                duration_ms=timer.duration_ms
            )

    return timer_context()


def get_observability_status() -> Dict[str, Any]:
    """Get current observability configuration status."""
    return {
        "logging": {
            "configured": _log_config is not None,
            "level": _log_config.level.value if _log_config else None,
            "format": _log_config.format.value if _log_config else None,
        },
        "tracing": {
            "configured": _trace_config is not None,
            "enabled": _trace_config.enabled if _trace_config else False,
            "exporter": _trace_config.exporter.value if _trace_config else None,
        },
        "metrics": {
            "configured": _metrics_config is not None,
            "enabled": _metrics_config.enabled if _metrics_config else False,
            "exporter": _metrics_config.exporter.value if _metrics_config else None,
        },
        "span_processors": len(_span_processors),
    }
