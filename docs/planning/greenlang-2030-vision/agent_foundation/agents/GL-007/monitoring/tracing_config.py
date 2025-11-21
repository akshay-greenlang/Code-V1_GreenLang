# -*- coding: utf-8 -*-
"""
Distributed tracing configuration for GL-007 FurnacePerformanceMonitor.

Provides:
- OpenTelemetry integration
- Trace propagation
- Span creation and management
- Jaeger/Zipkin export
- Performance profiling
- Custom instrumentation
"""

import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for distributed tracing."""

    def __init__(
        self,
        service_name: str = "gl-007-furnace-monitor",
        service_version: str = "1.0.0",
        environment: str = "production",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        enable_console: bool = False,
        sample_rate: float = 1.0,
    ):
        """
        Initialize tracing configuration.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Environment (development, staging, production)
            jaeger_endpoint: Jaeger collector endpoint
            otlp_endpoint: OTLP collector endpoint
            enable_console: Enable console exporter for debugging
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.jaeger_endpoint = jaeger_endpoint or os.getenv('JAEGER_ENDPOINT')
        self.otlp_endpoint = otlp_endpoint or os.getenv('OTLP_ENDPOINT')
        self.enable_console = enable_console
        self.sample_rate = sample_rate

    def setup_tracing(self) -> trace.Tracer:
        """
        Setup OpenTelemetry tracing.

        Returns:
            Configured tracer instance
        """
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            "environment": self.environment,
            "agent.id": "GL-007",
            "agent.name": "FurnacePerformanceMonitor",
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add exporters
        if self.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.jaeger_endpoint.split(':')[0],
                agent_port=int(self.jaeger_endpoint.split(':')[1]) if ':' in self.jaeger_endpoint else 6831,
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            logger.info(f"Jaeger exporter configured: {self.jaeger_endpoint}")

        if self.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured: {self.otlp_endpoint}")

        if self.enable_console:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
            logger.info("Console exporter enabled")

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Instrument libraries
        RequestsInstrumentor().instrument()
        LoggingInstrumentor().instrument()

        logger.info(
            f"Tracing configured for {self.service_name} v{self.service_version} "
            f"in {self.environment} environment"
        )

        # Return tracer
        return trace.get_tracer(__name__)


# Global tracer instance
_tracer: Optional[trace.Tracer] = None


def get_tracer() -> trace.Tracer:
    """
    Get the global tracer instance.

    Returns:
        Tracer instance
    """
    global _tracer
    if _tracer is None:
        # Create default tracer
        config = TracingConfig()
        _tracer = config.setup_tracing()
    return _tracer


def traced(
    span_name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to automatically trace function execution.

    Usage:
        @traced("calculate_efficiency")
        async def calculate_efficiency(furnace_id: str):
            return efficiency

    Args:
        span_name: Name of the span (defaults to function name)
        kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)
        attributes: Additional attributes to add to span

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        actual_span_name = span_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                actual_span_name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                try:
                    # Add function info
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)

                    # Execute function
                    result = await func(*args, **kwargs)

                    # Mark as successful
                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                actual_span_name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                try:
                    # Add function info
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)

                    # Execute function
                    result = func(*args, **kwargs)

                    # Mark as successful
                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class TracingContext:
    """
    Context manager for creating custom spans.

    Usage:
        with TracingContext("process_furnace_data", furnace_id="F-001"):
            # ... processing logic ...
            pass
    """

    def __init__(
        self,
        span_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        **attributes
    ):
        """
        Initialize tracing context.

        Args:
            span_name: Name of the span
            kind: Span kind
            **attributes: Additional attributes
        """
        self.span_name = span_name
        self.kind = kind
        self.attributes = attributes
        self.span = None

    def __enter__(self):
        """Enter tracing context."""
        tracer = get_tracer()
        self.span = tracer.start_span(
            self.span_name,
            kind=self.kind,
            attributes=self.attributes
        )
        self.span.__enter__()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit tracing context."""
        if self.span:
            if exc_type:
                self.span.record_exception(exc_val)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self.span.set_status(Status(StatusCode.OK))
            self.span.__exit__(exc_type, exc_val, exc_tb)


def add_span_attributes(**attributes):
    """
    Add attributes to current span.

    Usage:
        add_span_attributes(furnace_id="F-001", temperature=1250.5)
    """
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add event to current span.

    Usage:
        add_span_event("calculation_completed", {"efficiency": 85.2})
    """
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes=attributes or {})


def record_exception(exception: Exception):
    """
    Record exception in current span.

    Usage:
        try:
            # ... code ...
        except Exception as e:
            record_exception(e)
            raise
    """
    span = trace.get_current_span()
    if span:
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))


# Tracing helpers for specific operations
class FurnaceTracing:
    """Helper class for furnace-specific tracing."""

    @staticmethod
    def trace_calculation(calculation_type: str, furnace_id: str) -> TracingContext:
        """
        Create tracing context for furnace calculation.

        Args:
            calculation_type: Type of calculation
            furnace_id: Furnace identifier

        Returns:
            Tracing context
        """
        return TracingContext(
            f"furnace.calculation.{calculation_type}",
            kind=SpanKind.INTERNAL,
            furnace_id=furnace_id,
            calculation_type=calculation_type,
        )

    @staticmethod
    def trace_scada_read(furnace_id: str, tag_name: str) -> TracingContext:
        """
        Create tracing context for SCADA read operation.

        Args:
            furnace_id: Furnace identifier
            tag_name: SCADA tag name

        Returns:
            Tracing context
        """
        return TracingContext(
            "scada.read",
            kind=SpanKind.CLIENT,
            furnace_id=furnace_id,
            tag_name=tag_name,
        )

    @staticmethod
    def trace_prediction(model_type: str, furnace_id: str) -> TracingContext:
        """
        Create tracing context for ML prediction.

        Args:
            model_type: Type of ML model
            furnace_id: Furnace identifier

        Returns:
            Tracing context
        """
        return TracingContext(
            f"ml.prediction.{model_type}",
            kind=SpanKind.INTERNAL,
            model_type=model_type,
            furnace_id=furnace_id,
        )

    @staticmethod
    def trace_maintenance_check(furnace_id: str, component: str) -> TracingContext:
        """
        Create tracing context for maintenance check.

        Args:
            furnace_id: Furnace identifier
            component: Component being checked

        Returns:
            Tracing context
        """
        return TracingContext(
            "maintenance.check",
            kind=SpanKind.INTERNAL,
            furnace_id=furnace_id,
            component=component,
        )


# HTTP request tracing middleware
class TracingMiddleware:
    """
    Middleware to trace HTTP requests.

    Usage:
        app.add_middleware(TracingMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self.propagator = TraceContextTextMapPropagator()

    async def __call__(self, scope, receive, send):
        """Process request with tracing."""
        if scope['type'] == 'http':
            tracer = get_tracer()

            # Extract trace context from headers
            headers = dict(scope.get('headers', []))
            context = self.propagator.extract(headers)

            # Start span
            with tracer.start_as_current_span(
                f"{scope.get('method')} {scope.get('path')}",
                kind=SpanKind.SERVER,
                context=context,
                attributes={
                    "http.method": scope.get('method'),
                    "http.url": scope.get('path'),
                    "http.scheme": scope.get('scheme'),
                    "http.host": dict(scope.get('headers', [])).get(b'host', b'').decode('utf-8'),
                }
            ) as span:
                # Process request
                await self.app(scope, receive, send)

                # Add response status
                span.set_attribute("http.status_code", 200)  # Would get from actual response

        else:
            await self.app(scope, receive, send)


# Configuration presets
TRACING_CONFIGS = {
    'development': {
        'environment': 'development',
        'enable_console': True,
        'sample_rate': 1.0,
    },
    'staging': {
        'environment': 'staging',
        'jaeger_endpoint': 'jaeger-collector.greenlang.svc:6831',
        'sample_rate': 0.5,
    },
    'production': {
        'environment': 'production',
        'jaeger_endpoint': 'jaeger-collector.greenlang.svc:6831',
        'otlp_endpoint': 'otel-collector.greenlang.svc:4317',
        'sample_rate': 0.1,
    },
}


def setup_tracing_for_environment(environment: str = 'production') -> trace.Tracer:
    """
    Setup tracing based on environment.

    Args:
        environment: Environment name (development, staging, production)

    Returns:
        Configured tracer
    """
    config_params = TRACING_CONFIGS.get(environment, TRACING_CONFIGS['production'])
    config = TracingConfig(**config_params)
    return config.setup_tracing()


# Example usage
if __name__ == '__main__':
    # Setup tracing
    tracer = setup_tracing_for_environment('development')

    # Example 1: Using decorator
    @traced("example_function")
    def example_function(furnace_id: str):
        print(f"Processing furnace {furnace_id}")
        add_span_attributes(furnace_id=furnace_id)
        add_span_event("processing_started")
        return {"status": "success"}

    # Example 2: Using context manager
    with TracingContext("manual_span", furnace_id="F-001"):
        print("Processing with manual span")
        add_span_event("step_completed", {"step": 1})

    # Example 3: Furnace-specific tracing
    with FurnaceTracing.trace_calculation("thermal_efficiency", "F-001"):
        print("Calculating thermal efficiency")
        add_span_attributes(result=85.2)

    print("Tracing examples completed")
