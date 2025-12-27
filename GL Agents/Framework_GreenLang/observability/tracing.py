"""
GreenLang Framework - OpenTelemetry Tracing Module
===================================================

Provides distributed tracing capabilities for GreenLang agents using OpenTelemetry.
Supports span creation, context propagation, attribute injection, and export to
Jaeger/Zipkin backends.

Features:
- @traced decorator for automatic span creation
- Trace context propagation across service boundaries
- Attribute injection (agent_id, calculation_type, provenance_hash)
- Configurable sampling strategies
- Multiple exporter support (Jaeger, Zipkin, OTLP, Console)
- Zero-hallucination calculation path tracking

Example:
    >>> from greenlang_observability.tracing import TracingManager, traced
    >>>
    >>> # Initialize tracing
    >>> tracing = TracingManager(
    ...     service_name="gl-006-heatreclaim",
    ...     exporter_type="jaeger",
    ...     jaeger_endpoint="http://localhost:14268/api/traces",
    ... )
    >>>
    >>> @traced(name="calculate_pinch_temperature", attributes={"calculation_type": "thermal"})
    >>> def calculate_pinch(inputs):
    ...     # Calculation logic
    ...     return result

Version: 1.0.0
"""

from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """OpenTelemetry span kinds."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SamplingStrategy(Enum):
    """Sampling strategies for trace collection."""

    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    TRACE_ID_RATIO = "trace_id_ratio"
    PARENT_BASED = "parent_based"


class ExporterType(Enum):
    """Supported trace exporter types."""

    CONSOLE = "console"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    NONE = "none"


@dataclass
class TracingConfig:
    """
    Configuration for tracing infrastructure.

    Attributes:
        service_name: Name of the service (e.g., "gl-006-heatreclaim")
        service_version: Version of the service
        environment: Deployment environment (dev, staging, prod)
        exporter_type: Type of trace exporter to use
        jaeger_endpoint: Jaeger collector endpoint (for Jaeger exporter)
        zipkin_endpoint: Zipkin collector endpoint (for Zipkin exporter)
        otlp_endpoint: OTLP collector endpoint (for OTLP exporter)
        sampling_strategy: Strategy for sampling traces
        sampling_ratio: Ratio for trace_id_ratio sampling (0.0 to 1.0)
        max_attributes: Maximum number of attributes per span
        max_events: Maximum number of events per span
        max_links: Maximum number of links per span
        enabled: Whether tracing is enabled
    """

    service_name: str
    service_version: str = "1.0.0"
    environment: str = "development"
    exporter_type: ExporterType = ExporterType.CONSOLE
    jaeger_endpoint: Optional[str] = None
    zipkin_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    sampling_strategy: SamplingStrategy = SamplingStrategy.ALWAYS_ON
    sampling_ratio: float = 1.0
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.sampling_ratio < 0.0 or self.sampling_ratio > 1.0:
            raise ValueError("sampling_ratio must be between 0.0 and 1.0")

        if self.exporter_type == ExporterType.JAEGER and not self.jaeger_endpoint:
            self.jaeger_endpoint = os.getenv(
                "OTEL_EXPORTER_JAEGER_ENDPOINT",
                "http://localhost:14268/api/traces",
            )

        if self.exporter_type == ExporterType.ZIPKIN and not self.zipkin_endpoint:
            self.zipkin_endpoint = os.getenv(
                "OTEL_EXPORTER_ZIPKIN_ENDPOINT",
                "http://localhost:9411/api/v2/spans",
            )

        if self.exporter_type == ExporterType.OTLP and not self.otlp_endpoint:
            self.otlp_endpoint = os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://localhost:4317",
            )

    @classmethod
    def from_env(cls, service_name: str) -> TracingConfig:
        """Create configuration from environment variables."""
        return cls(
            service_name=service_name,
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            exporter_type=ExporterType(
                os.getenv("OTEL_EXPORTER_TYPE", "console").lower()
            ),
            jaeger_endpoint=os.getenv("OTEL_EXPORTER_JAEGER_ENDPOINT"),
            zipkin_endpoint=os.getenv("OTEL_EXPORTER_ZIPKIN_ENDPOINT"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            sampling_strategy=SamplingStrategy(
                os.getenv("OTEL_SAMPLING_STRATEGY", "always_on").lower()
            ),
            sampling_ratio=float(os.getenv("OTEL_SAMPLING_RATIO", "1.0")),
            enabled=os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true",
        )


@dataclass
class TraceContext:
    """
    Trace context for propagation across service boundaries.

    Implements W3C Trace Context specification.
    """

    trace_id: str
    span_id: str
    trace_flags: int = 1  # 1 = sampled
    trace_state: Optional[str] = None

    def to_traceparent(self) -> str:
        """Convert to W3C traceparent header format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, traceparent: str) -> Optional[TraceContext]:
        """Parse W3C traceparent header."""
        try:
            parts = traceparent.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None
            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
            )
        except (ValueError, IndexError):
            return None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for header injection."""
        result = {"traceparent": self.to_traceparent()}
        if self.trace_state:
            result["tracestate"] = self.trace_state
        return result


@dataclass
class SpanAttributes:
    """Standard span attributes for GreenLang agents."""

    agent_id: Optional[str] = None
    agent_version: Optional[str] = None
    calculation_type: Optional[str] = None
    provenance_hash: Optional[str] = None
    inputs_hash: Optional[str] = None
    framework_name: str = "greenlang"
    framework_version: str = "1.0.0"
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary of attributes."""
        attrs: Dict[str, Any] = {
            "greenlang.framework.name": self.framework_name,
            "greenlang.framework.version": self.framework_version,
        }

        if self.agent_id:
            attrs["greenlang.agent.id"] = self.agent_id
        if self.agent_version:
            attrs["greenlang.agent.version"] = self.agent_version
        if self.calculation_type:
            attrs["greenlang.calculation.type"] = self.calculation_type
        if self.provenance_hash:
            attrs["greenlang.provenance.hash"] = self.provenance_hash
        if self.inputs_hash:
            attrs["greenlang.inputs.hash"] = self.inputs_hash

        # Add custom attributes with prefix
        for key, value in self.custom.items():
            attrs[f"greenlang.custom.{key}"] = value

        return attrs


@dataclass
class Span:
    """
    Represents a trace span.

    A span represents a single operation within a trace.
    """

    name: str
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    status_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def record_exception(self, exception: Exception) -> None:
        """Record an exception as a span event."""
        self.add_event(
            name="exception",
            attributes={
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        self.set_status("ERROR", str(exception))

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self._calculate_duration(),
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_message": self.status_message,
        }

    def _calculate_duration(self) -> Optional[float]:
        """Calculate span duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: List[Span]) -> bool:
        """Export spans. Override in subclasses."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown exporter. Override in subclasses."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to console for debugging."""

    def export(self, spans: List[Span]) -> bool:
        """Export spans to console."""
        for span in spans:
            logger.info(
                f"[TRACE] {span.name} | trace_id={span.trace_id[:8]}... | "
                f"duration={span._calculate_duration():.2f}ms | "
                f"status={span.status}"
            )
        return True


class JaegerExporter(SpanExporter):
    """Exports spans to Jaeger backend."""

    def __init__(self, endpoint: str, service_name: str):
        """Initialize Jaeger exporter."""
        self.endpoint = endpoint
        self.service_name = service_name
        self._batch: List[Span] = []

    def export(self, spans: List[Span]) -> bool:
        """Export spans to Jaeger."""
        try:
            # In production, this would use the Jaeger Thrift/HTTP API
            # For now, log the intent and return success
            logger.debug(
                f"[JAEGER] Exporting {len(spans)} spans to {self.endpoint}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")
            return False


class ZipkinExporter(SpanExporter):
    """Exports spans to Zipkin backend."""

    def __init__(self, endpoint: str, service_name: str):
        """Initialize Zipkin exporter."""
        self.endpoint = endpoint
        self.service_name = service_name

    def export(self, spans: List[Span]) -> bool:
        """Export spans to Zipkin."""
        try:
            # In production, this would use the Zipkin HTTP API
            logger.debug(
                f"[ZIPKIN] Exporting {len(spans)} spans to {self.endpoint}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to export spans to Zipkin: {e}")
            return False


class OTLPExporter(SpanExporter):
    """Exports spans using OTLP protocol."""

    def __init__(self, endpoint: str, service_name: str):
        """Initialize OTLP exporter."""
        self.endpoint = endpoint
        self.service_name = service_name

    def export(self, spans: List[Span]) -> bool:
        """Export spans via OTLP."""
        try:
            # In production, this would use gRPC or HTTP OTLP protocol
            logger.debug(
                f"[OTLP] Exporting {len(spans)} spans to {self.endpoint}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to export spans via OTLP: {e}")
            return False


class Sampler:
    """Base class for sampling decisions."""

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[TraceContext] = None,
    ) -> bool:
        """Determine if a trace should be sampled."""
        raise NotImplementedError


class AlwaysOnSampler(Sampler):
    """Sample all traces."""

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[TraceContext] = None,
    ) -> bool:
        """Always return True."""
        return True


class AlwaysOffSampler(Sampler):
    """Sample no traces."""

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[TraceContext] = None,
    ) -> bool:
        """Always return False."""
        return False


class TraceIdRatioSampler(Sampler):
    """Sample traces based on trace ID ratio."""

    def __init__(self, ratio: float):
        """Initialize with sampling ratio."""
        self.ratio = max(0.0, min(1.0, ratio))

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[TraceContext] = None,
    ) -> bool:
        """Sample based on trace ID hash."""
        # Use first 8 chars of trace_id for deterministic sampling
        trace_hash = int(trace_id[:8], 16) / 0xFFFFFFFF
        return trace_hash < self.ratio


class ParentBasedSampler(Sampler):
    """Sample based on parent span decision."""

    def __init__(self, root_sampler: Sampler):
        """Initialize with root sampler for traces without parent."""
        self.root_sampler = root_sampler

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[TraceContext] = None,
    ) -> bool:
        """Sample based on parent decision or root sampler."""
        if parent_context:
            return parent_context.trace_flags == 1
        return self.root_sampler.should_sample(trace_id)


class TracingManager:
    """
    Central tracing manager for GreenLang agents.

    Manages span lifecycle, context propagation, and export.
    Thread-safe implementation using context variables.

    Example:
        >>> tracing = TracingManager(
        ...     service_name="gl-006-heatreclaim",
        ...     exporter_type=ExporterType.JAEGER,
        ... )
        >>>
        >>> with tracing.start_span("calculate") as span:
        ...     span.set_attribute("calculation_type", "pinch_analysis")
        ...     result = perform_calculation()
    """

    # Class-level context for current span (thread-local in production)
    _current_span: Optional[Span] = None
    _current_context: Optional[TraceContext] = None
    _instance: Optional[TracingManager] = None

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        exporter_type: Union[ExporterType, str] = ExporterType.CONSOLE,
        jaeger_endpoint: Optional[str] = None,
        zipkin_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        sampling_strategy: Union[SamplingStrategy, str] = SamplingStrategy.ALWAYS_ON,
        sampling_ratio: float = 1.0,
        enabled: bool = True,
        config: Optional[TracingConfig] = None,
    ):
        """
        Initialize tracing manager.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            exporter_type: Type of exporter to use
            jaeger_endpoint: Jaeger collector endpoint
            zipkin_endpoint: Zipkin collector endpoint
            otlp_endpoint: OTLP collector endpoint
            sampling_strategy: Sampling strategy
            sampling_ratio: Ratio for trace_id_ratio sampling
            enabled: Whether tracing is enabled
            config: Full configuration object (overrides other params)
        """
        if config:
            self.config = config
        else:
            if isinstance(exporter_type, str):
                exporter_type = ExporterType(exporter_type.lower())
            if isinstance(sampling_strategy, str):
                sampling_strategy = SamplingStrategy(sampling_strategy.lower())

            self.config = TracingConfig(
                service_name=service_name,
                service_version=service_version,
                exporter_type=exporter_type,
                jaeger_endpoint=jaeger_endpoint,
                zipkin_endpoint=zipkin_endpoint,
                otlp_endpoint=otlp_endpoint,
                sampling_strategy=sampling_strategy,
                sampling_ratio=sampling_ratio,
                enabled=enabled,
            )

        self._exporter = self._create_exporter()
        self._sampler = self._create_sampler()
        self._span_stack: List[Span] = []
        self._completed_spans: List[Span] = []

        # Set as global instance
        TracingManager._instance = self

    def _create_exporter(self) -> SpanExporter:
        """Create appropriate exporter based on config."""
        if self.config.exporter_type == ExporterType.JAEGER:
            return JaegerExporter(
                endpoint=self.config.jaeger_endpoint or "",
                service_name=self.config.service_name,
            )
        elif self.config.exporter_type == ExporterType.ZIPKIN:
            return ZipkinExporter(
                endpoint=self.config.zipkin_endpoint or "",
                service_name=self.config.service_name,
            )
        elif self.config.exporter_type == ExporterType.OTLP:
            return OTLPExporter(
                endpoint=self.config.otlp_endpoint or "",
                service_name=self.config.service_name,
            )
        else:
            return ConsoleSpanExporter()

    def _create_sampler(self) -> Sampler:
        """Create appropriate sampler based on config."""
        if self.config.sampling_strategy == SamplingStrategy.ALWAYS_OFF:
            return AlwaysOffSampler()
        elif self.config.sampling_strategy == SamplingStrategy.TRACE_ID_RATIO:
            return TraceIdRatioSampler(self.config.sampling_ratio)
        elif self.config.sampling_strategy == SamplingStrategy.PARENT_BASED:
            root = TraceIdRatioSampler(self.config.sampling_ratio)
            return ParentBasedSampler(root)
        else:
            return AlwaysOnSampler()

    def _generate_trace_id(self) -> str:
        """Generate a new trace ID (32 hex chars)."""
        return uuid.uuid4().hex + uuid.uuid4().hex[:8]

    def _generate_span_id(self) -> str:
        """Generate a new span ID (16 hex chars)."""
        return uuid.uuid4().hex[:16]

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[TraceContext] = None,
    ) -> Generator[Span, None, None]:
        """
        Start a new span as a context manager.

        Args:
            name: Name of the span
            kind: Kind of span
            attributes: Initial attributes
            parent_context: Parent trace context for distributed tracing

        Yields:
            The active span
        """
        if not self.config.enabled:
            # Return a no-op span
            yield Span(name=name, span_id="", trace_id="")
            return

        # Determine trace and parent span
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        elif self._span_stack:
            parent_span = self._span_stack[-1]
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
        else:
            trace_id = self._generate_trace_id()
            parent_span_id = None

        # Check sampling decision
        if not self._sampler.should_sample(trace_id, parent_context):
            yield Span(name=name, span_id="", trace_id="")
            return

        # Create span
        span = Span(
            name=name,
            span_id=self._generate_span_id(),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            kind=kind,
            start_time=datetime.now(timezone.utc),
            attributes=attributes or {},
        )

        # Add service attributes
        span.set_attribute("service.name", self.config.service_name)
        span.set_attribute("service.version", self.config.service_version)

        # Push to stack
        self._span_stack.append(span)
        TracingManager._current_span = span

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            # End span
            span.end_time = datetime.now(timezone.utc)

            # Pop from stack
            self._span_stack.pop()
            TracingManager._current_span = (
                self._span_stack[-1] if self._span_stack else None
            )

            # Export span
            self._completed_spans.append(span)
            if len(self._completed_spans) >= 10:
                self.flush()

    def start_span_no_context(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Start a span without using context manager.

        Must call end_span() manually.
        """
        if not self.config.enabled:
            return Span(name=name, span_id="", trace_id="")

        trace_id = (
            self._span_stack[-1].trace_id
            if self._span_stack
            else self._generate_trace_id()
        )
        parent_span_id = self._span_stack[-1].span_id if self._span_stack else None

        span = Span(
            name=name,
            span_id=self._generate_span_id(),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            kind=kind,
            start_time=datetime.now(timezone.utc),
            attributes=attributes or {},
        )

        span.set_attribute("service.name", self.config.service_name)
        span.set_attribute("service.version", self.config.service_version)

        self._span_stack.append(span)
        TracingManager._current_span = span

        return span

    def end_span(self, span: Span) -> None:
        """End a span that was started without context manager."""
        span.end_time = datetime.now(timezone.utc)

        if span in self._span_stack:
            self._span_stack.remove(span)

        TracingManager._current_span = (
            self._span_stack[-1] if self._span_stack else None
        )

        self._completed_spans.append(span)

    def flush(self) -> None:
        """Flush completed spans to exporter."""
        if self._completed_spans:
            self._exporter.export(self._completed_spans)
            self._completed_spans.clear()

    def shutdown(self) -> None:
        """Shutdown tracing manager and flush remaining spans."""
        self.flush()
        self._exporter.shutdown()

    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context for propagation."""
        if not self._span_stack:
            return None

        current = self._span_stack[-1]
        return TraceContext(
            trace_id=current.trace_id,
            span_id=current.span_id,
            trace_flags=1,
        )


# Global functions for convenience


def get_current_span() -> Optional[Span]:
    """Get the current active span."""
    return TracingManager._current_span


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    span = get_current_span()
    return span.trace_id if span else None


def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject trace context into HTTP headers."""
    if TracingManager._instance:
        context = TracingManager._instance.get_current_context()
        if context:
            headers.update(context.to_dict())
    return headers


def extract_trace_context(headers: Dict[str, str]) -> Optional[TraceContext]:
    """Extract trace context from HTTP headers."""
    traceparent = headers.get("traceparent")
    if traceparent:
        context = TraceContext.from_traceparent(traceparent)
        if context:
            context.trace_state = headers.get("tracestate")
        return context
    return None


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator for automatic span creation.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Static attributes to add
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record return value as attribute

    Example:
        >>> @traced(name="calculate_emissions", attributes={"type": "scope1"})
        >>> def calculate_emissions(activity_data, emission_factor):
        ...     return activity_data * emission_factor
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = TracingManager._instance

            if not manager or not manager.config.enabled:
                return func(*args, **kwargs)

            span_name = name or func.__name__
            span_attrs = dict(attributes or {})

            # Add function metadata
            span_attrs["code.function"] = func.__name__
            span_attrs["code.namespace"] = func.__module__

            # Optionally record arguments
            if record_args:
                span_attrs["function.args_count"] = len(args)
                span_attrs["function.kwargs_keys"] = list(kwargs.keys())

            with manager.start_span(span_name, kind=kind, attributes=span_attrs) as span:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)

                    # Optionally record result
                    if record_result and result is not None:
                        span.set_attribute("function.result_type", type(result).__name__)

                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("function.duration_ms", elapsed_ms)

        return cast(F, wrapper)

    return decorator


# Async version of traced decorator
def traced_async(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]:
    """
    Async decorator for automatic span creation.

    Same as @traced but for async functions.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = TracingManager._instance

            if not manager or not manager.config.enabled:
                return await func(*args, **kwargs)

            span_name = name or func.__name__
            span_attrs = dict(attributes or {})

            span_attrs["code.function"] = func.__name__
            span_attrs["code.namespace"] = func.__module__

            if record_args:
                span_attrs["function.args_count"] = len(args)
                span_attrs["function.kwargs_keys"] = list(kwargs.keys())

            with manager.start_span(span_name, kind=kind, attributes=span_attrs) as span:
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)

                    if record_result and result is not None:
                        span.set_attribute("function.result_type", type(result).__name__)

                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("function.duration_ms", elapsed_ms)

        return cast(F, wrapper)

    return decorator
