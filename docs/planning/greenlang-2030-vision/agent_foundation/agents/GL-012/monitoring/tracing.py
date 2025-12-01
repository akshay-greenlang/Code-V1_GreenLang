# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Steam Quality Controller Distributed Tracing
==============================================================

Production-ready distributed tracing support for the GL-012 STEAMQUAL
SteamQualityController agent. Provides OpenTelemetry-compatible tracing
for quality calculations, control actions, integration calls, and
agent coordination.

Features:
- OpenTelemetry-compatible span creation
- Automatic context propagation
- Exception recording with stack traces
- Event annotation support
- Configurable sampling and export
- Fallback stub implementation when OTEL not available

Span Categories:
1. Quality Calculations - Steam quality computation traces
2. Control Actions - Valve and desuperheater control traces
3. Integration Calls - SCADA and external system calls
4. Agent Coordination - Inter-agent communication traces

Usage:
    >>> from monitoring.tracing import TracingConfig, get_tracer
    >>>
    >>> # Initialize tracing
    >>> config = TracingConfig(
    ...     service_name="gl-012-steamqual",
    ...     endpoint="http://jaeger:14268/api/traces"
    ... )
    >>> config.initialize()
    >>>
    >>> # Create spans for operations
    >>> tracer = get_tracer()
    >>> with tracer.create_span("calculate_quality") as span:
    ...     span.add_event("started_calculation")
    ...     result = calculate_steam_quality()
    ...     span.set_attribute("dryness", result.dryness)

Author: GreenLang Team
License: Proprietary
"""

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Generator, Callable, Union
)
from uuid import uuid4

logger = logging.getLogger(__name__)


# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        Tracer,
        Span,
        SpanKind,
        Status,
        StatusCode,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.context import Context

    # Try to import OTLP exporter
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False

    # Try to import Jaeger exporter
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        JAEGER_AVAILABLE = True
    except ImportError:
        JAEGER_AVAILABLE = False

    OTEL_AVAILABLE = True

except ImportError:
    OTEL_AVAILABLE = False
    OTLP_AVAILABLE = False
    JAEGER_AVAILABLE = False
    logger.warning(
        "OpenTelemetry not installed. Tracing will use stub implementation."
    )


class SpanType(Enum):
    """Types of spans for steam quality operations."""
    QUALITY_CALCULATION = "quality_calculation"
    CONTROL_ACTION = "control_action"
    INTEGRATION_CALL = "integration_call"
    AGENT_COORDINATION = "agent_coordination"
    VALIDATION = "validation"
    DATA_PROCESSING = "data_processing"

    def __str__(self) -> str:
        return self.value


class ExporterType(Enum):
    """Supported trace exporters."""
    CONSOLE = "console"
    OTLP = "otlp"
    JAEGER = "jaeger"
    NONE = "none"


@dataclass
class SpanData:
    """
    Represents span data for the stub implementation.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Unique span identifier
        parent_span_id: Parent span ID (if child span)
        name: Span name
        span_type: Type of span
        start_time: Span start timestamp
        end_time: Span end timestamp (set on completion)
        status: Span status (ok, error)
        attributes: Span attributes
        events: Span events
        exception: Recorded exception (if any)
    """
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    name: str = ""
    span_type: SpanType = SpanType.QUALITY_CALCULATION
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    exception: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "start_time_iso": datetime.fromtimestamp(
                self.start_time, tz=timezone.utc
            ).isoformat(),
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "exception": self.exception,
        }


class StubSpan:
    """
    Stub span implementation for when OpenTelemetry is not available.

    Provides the same interface as OpenTelemetry spans but stores
    data locally for debugging and testing.
    """

    def __init__(
        self,
        name: str,
        span_type: SpanType = SpanType.QUALITY_CALCULATION,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span: Optional["StubSpan"] = None,
    ):
        """
        Initialize stub span.

        Args:
            name: Span name
            span_type: Type of span
            attributes: Initial attributes
            parent_span: Parent span (if child)
        """
        self.data = SpanData(
            name=name,
            span_type=span_type,
            attributes=attributes or {},
        )

        if parent_span:
            self.data.parent_span_id = parent_span.data.span_id
            self.data.trace_id = parent_span.data.trace_id

        self._is_recording = True

    def set_attribute(self, key: str, value: Any) -> "StubSpan":
        """
        Set a span attribute.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            Self for chaining
        """
        if self._is_recording:
            self.data.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "StubSpan":
        """
        Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes

        Returns:
            Self for chaining
        """
        if self._is_recording:
            self.data.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "StubSpan":
        """
        Add an event to the span.

        Args:
            name: Event name
            attributes: Event attributes

        Returns:
            Self for chaining
        """
        if self._is_recording:
            event = {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
            self.data.events.append(event)
        return self

    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "StubSpan":
        """
        Record an exception on the span.

        Args:
            exception: Exception to record
            attributes: Additional attributes

        Returns:
            Self for chaining
        """
        if self._is_recording:
            self.data.status = "error"
            self.data.exception = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
                "attributes": attributes or {},
            }
        return self

    def set_status(self, status: str, description: str = "") -> "StubSpan":
        """
        Set span status.

        Args:
            status: Status code (ok, error)
            description: Status description

        Returns:
            Self for chaining
        """
        if self._is_recording:
            self.data.status = status
            if description:
                self.data.attributes["status_description"] = description
        return self

    def end(self, end_time: Optional[float] = None) -> None:
        """
        End the span.

        Args:
            end_time: Optional custom end time
        """
        self.data.end_time = end_time or time.time()
        self._is_recording = False

    def is_recording(self) -> bool:
        """Check if span is still recording."""
        return self._is_recording

    def get_span_context(self) -> Dict[str, str]:
        """Get span context for propagation."""
        return {
            "trace_id": self.data.trace_id,
            "span_id": self.data.span_id,
        }

    def __enter__(self) -> "StubSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val:
            self.record_exception(exc_val)
        self.end()


class StubTracer:
    """
    Stub tracer implementation for when OpenTelemetry is not available.

    Provides the same interface as OpenTelemetry tracer but creates
    StubSpan instances.
    """

    def __init__(self, name: str = "gl-012-steamqual"):
        """
        Initialize stub tracer.

        Args:
            name: Tracer name
        """
        self.name = name
        self._spans: List[SpanData] = []
        self._current_span: Optional[StubSpan] = None

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.QUALITY_CALCULATION,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> StubSpan:
        """
        Start a new span.

        Args:
            name: Span name
            span_type: Type of span
            attributes: Initial attributes

        Returns:
            New StubSpan instance
        """
        span = StubSpan(
            name=name,
            span_type=span_type,
            attributes=attributes,
            parent_span=self._current_span,
        )
        return span

    @contextmanager
    def create_span(
        self,
        name: str,
        span_type: SpanType = SpanType.QUALITY_CALCULATION,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[StubSpan, None, None]:
        """
        Create a span as a context manager.

        Args:
            name: Span name
            span_type: Type of span
            attributes: Initial attributes

        Yields:
            StubSpan instance
        """
        span = self.start_span(name, span_type, attributes)
        previous_span = self._current_span
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._spans.append(span.data)
            self._current_span = previous_span

    def get_current_span(self) -> Optional[StubSpan]:
        """Get the current active span."""
        return self._current_span

    def get_recorded_spans(self) -> List[SpanData]:
        """Get all recorded spans (for testing/debugging)."""
        return self._spans.copy()

    def clear_spans(self) -> None:
        """Clear recorded spans."""
        self._spans.clear()


@dataclass
class TracingConfig:
    """
    Configuration for distributed tracing.

    Attributes:
        service_name: Service name for traces
        exporter_type: Type of exporter to use
        endpoint: Trace collector endpoint
        sampling_ratio: Sampling ratio (0.0-1.0)
        enabled: Whether tracing is enabled
        console_export: Also export to console (for debugging)
        attributes: Default attributes for all spans
    """
    service_name: str = "gl-012-steamqual"
    exporter_type: ExporterType = ExporterType.OTLP
    endpoint: Optional[str] = None
    sampling_ratio: float = 1.0
    enabled: bool = True
    console_export: bool = False
    attributes: Dict[str, str] = field(default_factory=dict)

    # Internal state
    _initialized: bool = field(default=False, init=False)
    _tracer: Any = field(default=None, init=False)

    def initialize(self) -> bool:
        """
        Initialize tracing with this configuration.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            logger.warning("Tracing already initialized")
            return True

        if not self.enabled:
            logger.info("Tracing disabled by configuration")
            self._tracer = StubTracer(self.service_name)
            self._initialized = True
            return True

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available, using stub tracer"
            )
            self._tracer = StubTracer(self.service_name)
            self._initialized = True
            return True

        try:
            # Create resource with service name
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                "agent.id": "GL-012",
                "agent.codename": "STEAMQUAL",
                "agent.domain": "steam_quality_control",
                **self.attributes,
            })

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add exporters based on configuration
            exporters_added = 0

            if self.exporter_type == ExporterType.CONSOLE or self.console_export:
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(
                    BatchSpanProcessor(console_exporter)
                )
                exporters_added += 1
                logger.info("Added console trace exporter")

            if self.exporter_type == ExporterType.OTLP:
                if OTLP_AVAILABLE and self.endpoint:
                    otlp_exporter = OTLPSpanExporter(endpoint=self.endpoint)
                    provider.add_span_processor(
                        BatchSpanProcessor(otlp_exporter)
                    )
                    exporters_added += 1
                    logger.info(f"Added OTLP exporter: {self.endpoint}")
                else:
                    logger.warning(
                        "OTLP exporter requested but not available or no endpoint"
                    )

            if self.exporter_type == ExporterType.JAEGER:
                if JAEGER_AVAILABLE and self.endpoint:
                    jaeger_exporter = JaegerExporter(
                        agent_host_name=self.endpoint.split(":")[0],
                        agent_port=int(self.endpoint.split(":")[1])
                        if ":" in self.endpoint else 6831,
                    )
                    provider.add_span_processor(
                        BatchSpanProcessor(jaeger_exporter)
                    )
                    exporters_added += 1
                    logger.info(f"Added Jaeger exporter: {self.endpoint}")
                else:
                    logger.warning(
                        "Jaeger exporter requested but not available or no endpoint"
                    )

            if exporters_added == 0:
                logger.warning("No trace exporters configured")

            # Set the global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self._tracer = trace.get_tracer(
                self.service_name,
                "1.0.0",
            )

            self._initialized = True
            logger.info(
                f"Tracing initialized: service={self.service_name}, "
                f"exporters={exporters_added}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}", exc_info=True)
            self._tracer = StubTracer(self.service_name)
            self._initialized = True
            return False

    def get_tracer(self) -> Union[StubTracer, Any]:
        """
        Get the configured tracer.

        Returns:
            Tracer instance (OpenTelemetry or Stub)
        """
        if not self._initialized:
            self.initialize()
        return self._tracer

    @contextmanager
    def create_span(
        self,
        name: str,
        span_type: SpanType = SpanType.QUALITY_CALCULATION,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """
        Create a span using this configuration.

        Args:
            name: Span name
            span_type: Type of span
            attributes: Span attributes

        Yields:
            Span instance
        """
        tracer = self.get_tracer()

        if isinstance(tracer, StubTracer):
            with tracer.create_span(name, span_type, attributes) as span:
                yield span
        else:
            # OpenTelemetry tracer
            span_attributes = {
                "span.type": span_type.value,
                "agent.id": "GL-012",
                **(attributes or {}),
            }
            with tracer.start_as_current_span(
                name,
                attributes=span_attributes,
            ) as span:
                yield span

    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an exception on the current span.

        Args:
            exception: Exception to record
            attributes: Additional attributes
        """
        tracer = self.get_tracer()

        if isinstance(tracer, StubTracer):
            current_span = tracer.get_current_span()
            if current_span:
                current_span.record_exception(exception, attributes)
        elif OTEL_AVAILABLE:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.record_exception(exception, attributes)
                current_span.set_status(
                    Status(StatusCode.ERROR, str(exception))
                )

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an event to the current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        tracer = self.get_tracer()

        if isinstance(tracer, StubTracer):
            current_span = tracer.get_current_span()
            if current_span:
                current_span.add_event(name, attributes)
        elif OTEL_AVAILABLE:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.add_event(name, attributes or {})


# Module-level tracer management

_default_config: Optional[TracingConfig] = None


def get_tracer() -> Union[StubTracer, Any]:
    """
    Get the default tracer.

    Returns:
        Tracer instance
    """
    global _default_config
    if _default_config is None:
        _default_config = TracingConfig()
        _default_config.initialize()
    return _default_config.get_tracer()


def init_tracing(**kwargs) -> TracingConfig:
    """
    Initialize tracing with the given configuration.

    Args:
        **kwargs: Arguments for TracingConfig

    Returns:
        Initialized TracingConfig
    """
    global _default_config
    _default_config = TracingConfig(**kwargs)
    _default_config.initialize()
    return _default_config


def create_span(
    name: str,
    span_type: SpanType = SpanType.QUALITY_CALCULATION,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Create a span using the default configuration.

    Args:
        name: Span name
        span_type: Type of span
        attributes: Span attributes

    Yields:
        Span instance
    """
    global _default_config
    if _default_config is None:
        _default_config = TracingConfig()
        _default_config.initialize()

    return _default_config.create_span(name, span_type, attributes)


def record_exception(
    exception: Exception,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an exception on the current span.

    Args:
        exception: Exception to record
        attributes: Additional attributes
    """
    global _default_config
    if _default_config is None:
        return

    _default_config.record_exception(exception, attributes)


def add_event(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    global _default_config
    if _default_config is None:
        return

    _default_config.add_event(name, attributes)


__all__ = [
    "OTEL_AVAILABLE",
    "OTLP_AVAILABLE",
    "JAEGER_AVAILABLE",
    "SpanType",
    "ExporterType",
    "SpanData",
    "StubSpan",
    "StubTracer",
    "TracingConfig",
    "get_tracer",
    "init_tracing",
    "create_span",
    "record_exception",
    "add_event",
]
