# -*- coding: utf-8 -*-
"""
Distributed tracing for GreenLang using OpenTelemetry
"""

import functools
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import uuid
from greenlang.determinism import deterministic_random
from greenlang.determinism import deterministic_uuid, DeterministicClock

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

    # Mock classes for when OpenTelemetry is not available
    class MockTracer:
        def start_as_current_span(self, name, **kwargs):
            return MockSpan()

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def set_status(self, status):
            pass

        def add_event(self, name, attributes=None):
            pass

        def record_exception(self, exception):
            pass

    trace = type("trace", (), {"get_tracer": lambda name: MockTracer()})()
    Status = StatusCode = TraceContextTextMapPropagator = None

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Span kinds"""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanContext:
    """Span context information"""

    trace_id: str = field(default_factory=lambda: deterministic_uuid(__name__, str(DeterministicClock.now())).hex)
    span_id: str = field(default_factory=lambda: deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16])
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceConfig:
    """Tracing configuration"""

    service_name: str = "greenlang"
    service_version: str = "1.0.0"
    environment: str = "production"

    # Exporters
    console_export: bool = False
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None

    # Sampling
    sampling_rate: float = 1.0

    # Processing
    batch_size: int = 512
    export_timeout_ms: int = 30000
    max_queue_size: int = 2048


class TracingManager:
    """Manage distributed tracing"""

    def __init__(self, config: Optional[TraceConfig] = None):
        """
        Initialize tracing manager

        Args:
            config: Tracing configuration
        """
        self.config = config or TraceConfig()
        self.tracer = None
        self.propagator = None
        self.provider = None

        if OPENTELEMETRY_AVAILABLE:
            self._initialize_tracing()
        else:
            logger.warning("OpenTelemetry not available, tracing disabled")

    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing"""
        # Create resource
        resource = Resource.create(
            {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
            }
        )

        # Create provider
        self.provider = TracerProvider(resource=resource)

        # Add exporters
        if self.config.console_export:
            self.provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        if self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_endpoint.split(":")[0],
                agent_port=(
                    int(self.config.jaeger_endpoint.split(":")[1])
                    if ":" in self.config.jaeger_endpoint
                    else 6831
                ),
            )
            self.provider.add_span_processor(
                BatchSpanProcessor(
                    jaeger_exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.batch_size,
                )
            )

        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint, insecure=True
            )
            self.provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.batch_size,
                )
            )

        # Set global provider
        trace.set_tracer_provider(self.provider)

        # Get tracer
        self.tracer = trace.get_tracer(
            self.config.service_name, self.config.service_version
        )

        # Initialize propagator
        self.propagator = TraceContextTextMapPropagator()

        # Auto-instrument libraries
        RequestsInstrumentor().instrument()

        logger.info(f"Tracing initialized for service {self.config.service_name}")

    def get_tracer(self):
        """Get tracer instance"""
        if not self.tracer and OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(self.config.service_name)
        return self.tracer or MockTracer()

    @contextmanager
    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a new span

        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
        """
        tracer = self.get_tracer()

        with tracer.start_as_current_span(
            name,
            kind=(
                getattr(trace.SpanKind, kind.value.upper(), None)
                if OPENTELEMETRY_AVAILABLE
                else None
            ),
        ) as span:
            if attributes and hasattr(span, "set_attribute"):
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            yield span

    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """
        Inject trace context into carrier

        Args:
            carrier: Carrier dictionary

        Returns:
            Carrier with trace context
        """
        if self.propagator and OPENTELEMETRY_AVAILABLE:
            self.propagator.inject(carrier)
        return carrier

    def extract_context(self, carrier: Dict[str, str]):
        """
        Extract trace context from carrier

        Args:
            carrier: Carrier dictionary

        Returns:
            Context
        """
        if self.propagator and OPENTELEMETRY_AVAILABLE:
            return self.propagator.extract(carrier)
        return None

    def shutdown(self):
        """Shutdown tracing"""
        if self.provider and hasattr(self.provider, "shutdown"):
            self.provider.shutdown()


def trace_operation(operation: str, kind: SpanKind = SpanKind.INTERNAL):
    """
    Decorator to trace function execution

    Args:
        operation: Operation name
        kind: Span kind
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get tracer
            tracer = get_tracer()

            # Extract attributes from function arguments
            attributes = {
                "operation": operation,
                "function": func.__name__,
                "module": func.__module__,
            }

            # Add tenant_id if present
            if "tenant_id" in kwargs:
                attributes["tenant_id"] = kwargs["tenant_id"]

            # Create span
            with tracer.start_as_current_span(
                operation,
                kind=(
                    getattr(trace.SpanKind, kind.value.upper(), None)
                    if OPENTELEMETRY_AVAILABLE
                    else None
                ),
            ) as span:
                # Set attributes
                if hasattr(span, "set_attribute"):
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Set success status
                    if hasattr(span, "set_status") and OPENTELEMETRY_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    # Record exception
                    if hasattr(span, "record_exception"):
                        span.record_exception(e)

                    # Set error status
                    if hasattr(span, "set_status") and OPENTELEMETRY_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))

                    raise

        return wrapper

    return decorator


def add_span_attributes(**attributes):
    """
    Add attributes to current span

    Args:
        **attributes: Attributes to add
    """
    if OPENTELEMETRY_AVAILABLE:
        span = trace.get_current_span()
        if span and hasattr(span, "set_attribute"):
            for key, value in attributes.items():
                span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add event to current span

    Args:
        name: Event name
        attributes: Event attributes
    """
    if OPENTELEMETRY_AVAILABLE:
        span = trace.get_current_span()
        if span and hasattr(span, "add_event"):
            span.add_event(name, attributes=attributes)


def set_span_status(status: bool, message: Optional[str] = None):
    """
    Set status of current span

    Args:
        status: Success/failure
        message: Status message
    """
    if OPENTELEMETRY_AVAILABLE:
        span = trace.get_current_span()
        if span and hasattr(span, "set_status"):
            if status:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, message or "Operation failed"))


class TraceContextManager:
    """Manage trace context across async operations"""

    def __init__(self):
        """Initialize context manager"""
        self.contexts: Dict[str, SpanContext] = {}

    def save_context(self, operation_id: str) -> SpanContext:
        """
        Save current trace context

        Args:
            operation_id: Operation identifier

        Returns:
            Saved context
        """
        if OPENTELEMETRY_AVAILABLE:
            span = trace.get_current_span()
            if span:
                context = span.get_span_context()
                span_context = SpanContext(
                    trace_id=format(context.trace_id, "032x"),
                    span_id=format(context.span_id, "016x"),
                )
                self.contexts[operation_id] = span_context
                return span_context

        # Create mock context
        span_context = SpanContext()
        self.contexts[operation_id] = span_context
        return span_context

    def restore_context(self, operation_id: str) -> Optional[SpanContext]:
        """
        Restore trace context

        Args:
            operation_id: Operation identifier

        Returns:
            Restored context
        """
        return self.contexts.get(operation_id)

    def clear_context(self, operation_id: str):
        """
        Clear saved context

        Args:
            operation_id: Operation identifier
        """
        self.contexts.pop(operation_id, None)


class DistributedTracer:
    """Distributed tracing across services"""

    def __init__(self, service_name: str):
        """
        Initialize distributed tracer

        Args:
            service_name: Service name
        """
        self.service_name = service_name
        self.tracer = get_tracer()
        self.propagator = (
            TraceContextTextMapPropagator() if OPENTELEMETRY_AVAILABLE else None
        )

    def start_trace(
        self, operation: str, carrier: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Start a distributed trace

        Args:
            operation: Operation name
            carrier: Incoming trace context

        Returns:
            Outgoing trace context
        """
        # Extract parent context if provided
        parent_context = None
        if carrier and self.propagator:
            parent_context = self.propagator.extract(carrier)

        # Start span
        span = self.tracer.start_as_current_span(
            f"{self.service_name}.{operation}", context=parent_context
        )

        # Create outgoing context
        outgoing = {}
        if self.propagator:
            self.propagator.inject(outgoing)

        return outgoing

    def continue_trace(
        self, operation: str, trace_context: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Continue an existing trace

        Args:
            operation: Operation name
            trace_context: Incoming trace context

        Returns:
            Outgoing trace context
        """
        return self.start_trace(operation, trace_context)


# Sampling strategies
class SamplingStrategy:
    """Base sampling strategy"""

    def should_sample(self, operation: str, attributes: Dict[str, Any]) -> bool:
        """
        Determine if operation should be sampled

        Args:
            operation: Operation name
            attributes: Operation attributes

        Returns:
            True if should sample
        """
        return True


class RateSampler(SamplingStrategy):
    """Rate-based sampling"""

    def __init__(self, rate: float):
        """
        Initialize rate sampler

        Args:
            rate: Sampling rate (0.0 to 1.0)
        """
        self.rate = max(0.0, min(1.0, rate))

    def should_sample(self, operation: str, attributes: Dict[str, Any]) -> bool:
        """Check if should sample based on rate"""
        import random

        return deterministic_random().random() < self.rate


class ErrorSampler(SamplingStrategy):
    """Sample all errors"""

    def should_sample(self, operation: str, attributes: Dict[str, Any]) -> bool:
        """Sample if error attribute is present"""
        return attributes.get("error", False)


class CompositeSampler(SamplingStrategy):
    """Composite sampling strategy"""

    def __init__(self, strategies: List[SamplingStrategy]):
        """
        Initialize composite sampler

        Args:
            strategies: List of sampling strategies
        """
        self.strategies = strategies

    def should_sample(self, operation: str, attributes: Dict[str, Any]) -> bool:
        """Check if any strategy says to sample"""
        return any(s.should_sample(operation, attributes) for s in self.strategies)


# Global instances
_tracing_manager: Optional[TracingManager] = None
_trace_context_manager: Optional[TraceContextManager] = None


def get_tracing_manager() -> TracingManager:
    """Get global tracing manager"""
    global _tracing_manager
    if not _tracing_manager:
        _tracing_manager = TracingManager()
    return _tracing_manager


def get_tracer():
    """Get global tracer"""
    return get_tracing_manager().get_tracer()


def get_trace_context_manager() -> TraceContextManager:
    """Get global trace context manager"""
    global _trace_context_manager
    if not _trace_context_manager:
        _trace_context_manager = TraceContextManager()
    return _trace_context_manager


def create_span(name: str, **kwargs):
    """Create a new span"""
    return get_tracing_manager().create_span(name, **kwargs)
