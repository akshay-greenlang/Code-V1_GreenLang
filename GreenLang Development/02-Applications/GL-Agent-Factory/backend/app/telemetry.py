"""
OpenTelemetry Distributed Tracing Configuration

This module provides comprehensive OpenTelemetry setup for the GreenLang Agent Factory,
including automatic instrumentation for FastAPI, httpx, SQLAlchemy, and Redis,
plus custom spans for agent execution tracing.

Configuration:
    Set GREENLANG_OTLP_ENDPOINT to configure the OTLP exporter endpoint.
    Set GREENLANG_TRACING_ENABLED=true to enable tracing.

Example:
    >>> from app.telemetry import init_telemetry, get_tracer
    >>> init_telemetry(service_name="greenlang-api")
    >>> tracer = get_tracer()
    >>> with tracer.start_as_current_span("my-operation") as span:
    ...     span.set_attribute("custom.key", "value")
"""

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import extract, inject, set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBasedTraceIdRatio
from opentelemetry.trace import Status, StatusCode, Span, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])

# Global tracer instance
_tracer: Optional[Tracer] = None
_tracer_provider: Optional[TracerProvider] = None
_initialized: bool = False


def init_telemetry(
    service_name: str = "greenlang-agent-factory",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
    enable_console_exporter: bool = False,
    environment: str = "development",
) -> TracerProvider:
    """
    Initialize OpenTelemetry distributed tracing.

    This function sets up:
    - TracerProvider with configurable sampling
    - OTLP exporter for Jaeger/Tempo
    - Automatic instrumentation for FastAPI, httpx
    - W3C TraceContext and B3 propagation

    Args:
        service_name: Name of the service for trace identification
        service_version: Version string for the service
        otlp_endpoint: OTLP collector endpoint (e.g., "http://jaeger:4317")
        sample_rate: Sampling rate (0.0 to 1.0), 1.0 = sample all traces
        enable_console_exporter: Enable console output for debugging
        environment: Deployment environment (development, staging, production)

    Returns:
        Configured TracerProvider instance

    Example:
        >>> provider = init_telemetry(
        ...     service_name="greenlang-api",
        ...     otlp_endpoint="http://jaeger:4317",
        ...     sample_rate=0.5,
        ... )
    """
    global _tracer, _tracer_provider, _initialized

    if _initialized:
        logger.warning("Telemetry already initialized, skipping re-initialization")
        return _tracer_provider

    # Get configuration from environment if not provided
    otlp_endpoint = otlp_endpoint or os.getenv("GREENLANG_OTLP_ENDPOINT", "http://jaeger:4317")
    tracing_enabled = os.getenv("GREENLANG_TRACING_ENABLED", "true").lower() == "true"

    if not tracing_enabled:
        logger.info("Tracing is disabled via GREENLANG_TRACING_ENABLED")
        _initialized = True
        return None

    logger.info(f"Initializing OpenTelemetry tracing for service: {service_name}")

    # Create resource with service information
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": environment,
            "service.namespace": "greenlang",
            "telemetry.sdk.language": "python",
        }
    )

    # Configure sampler based on sample rate
    if sample_rate >= 1.0:
        sampler = ParentBasedTraceIdRatio(1.0)
    else:
        sampler = ParentBasedTraceIdRatio(sample_rate)

    # Create TracerProvider
    _tracer_provider = TracerProvider(
        resource=resource,
        sampler=sampler,
    )

    # Add OTLP exporter for Jaeger/Tempo
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # Use TLS in production
        )
        otlp_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )
        _tracer_provider.add_span_processor(otlp_processor)
        logger.info(f"OTLP exporter configured for endpoint: {otlp_endpoint}")
    except Exception as e:
        logger.warning(f"Failed to configure OTLP exporter: {e}")

    # Add console exporter for debugging (development only)
    if enable_console_exporter or environment == "development":
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        _tracer_provider.add_span_processor(console_processor)
        logger.debug("Console span exporter enabled")

    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Configure propagators for distributed context
    # Support both W3C TraceContext and B3 for compatibility
    propagator = CompositePropagator(
        [
            TraceContextTextMapPropagator(),  # W3C TraceContext (primary)
            B3MultiFormat(),  # B3 multi-header (fallback)
        ]
    )
    set_global_textmap(propagator)

    # Get tracer instance
    _tracer = trace.get_tracer(service_name, service_version)

    _initialized = True
    logger.info("OpenTelemetry tracing initialized successfully")

    return _tracer_provider


def instrument_fastapi(app) -> None:
    """
    Add automatic instrumentation to FastAPI application.

    This adds tracing to all HTTP requests with:
    - Request/response attributes
    - Route information
    - HTTP status codes
    - Exception tracking

    Args:
        app: FastAPI application instance
    """
    if not _initialized:
        logger.warning("Telemetry not initialized, skipping FastAPI instrumentation")
        return

    try:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,metrics",  # Skip health checks
            tracer_provider=_tracer_provider,
        )
        logger.info("FastAPI instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI: {e}")


def instrument_httpx() -> None:
    """
    Add automatic instrumentation for httpx HTTP client.

    This traces all outgoing HTTP requests with:
    - Request URL and method
    - Response status
    - Request/response sizes
    """
    if not _initialized:
        logger.warning("Telemetry not initialized, skipping httpx instrumentation")
        return

    try:
        HTTPXClientInstrumentor().instrument(
            tracer_provider=_tracer_provider,
        )
        logger.info("httpx instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument httpx: {e}")


def instrument_sqlalchemy(engine) -> None:
    """
    Add automatic instrumentation for SQLAlchemy.

    This traces all database operations with:
    - SQL statements (sanitized)
    - Database connection info
    - Execution timing

    Args:
        engine: SQLAlchemy engine instance
    """
    if not _initialized:
        logger.warning("Telemetry not initialized, skipping SQLAlchemy instrumentation")
        return

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(
            engine=engine,
            tracer_provider=_tracer_provider,
            enable_commenter=True,  # Add trace context to SQL comments
        )
        logger.info("SQLAlchemy instrumentation enabled")
    except ImportError:
        logger.warning("SQLAlchemy instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to instrument SQLAlchemy: {e}")


def instrument_redis(client) -> None:
    """
    Add automatic instrumentation for Redis client.

    This traces all Redis operations with:
    - Command type
    - Key names (configurable)
    - Response times

    Args:
        client: Redis client instance
    """
    if not _initialized:
        logger.warning("Telemetry not initialized, skipping Redis instrumentation")
        return

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument(
            tracer_provider=_tracer_provider,
        )
        logger.info("Redis instrumentation enabled")
    except ImportError:
        logger.warning("Redis instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to instrument Redis: {e}")


def get_tracer(name: Optional[str] = None) -> Optional[Tracer]:
    """
    Get a tracer instance for manual instrumentation.

    Args:
        name: Optional tracer name (defaults to module name)

    Returns:
        Tracer instance or None if tracing is disabled

    Example:
        >>> tracer = get_tracer("my-module")
        >>> with tracer.start_as_current_span("operation") as span:
        ...     span.set_attribute("key", "value")
    """
    if not _initialized or _tracer_provider is None:
        return None

    return trace.get_tracer(name or __name__)


def get_current_span() -> Optional[Span]:
    """
    Get the current active span.

    Returns:
        Current span or None if no active span
    """
    return trace.get_current_span()


@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[Optional[Span], None, None]:
    """
    Context manager for creating custom spans.

    Args:
        name: Span name
        attributes: Initial span attributes
        kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)

    Yields:
        Span instance or None if tracing is disabled

    Example:
        >>> with create_span("process-data", {"item.count": 100}) as span:
        ...     result = process_data()
        ...     span.set_attribute("result.size", len(result))
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# =============================================================================
# Agent Execution Tracing
# =============================================================================


@contextmanager
def trace_agent_execution(
    agent_id: str,
    agent_name: str,
    tenant_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None,
) -> Generator[Optional[Span], None, None]:
    """
    Create a span for agent execution tracing.

    This is the primary tracing context for agent operations, providing:
    - Agent identification attributes
    - Execution tracking
    - Input/output recording
    - Error handling

    Args:
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
        tenant_id: Optional tenant identifier
        execution_id: Optional execution identifier
        input_data: Optional input parameters (sanitized)

    Yields:
        Span instance for adding custom attributes

    Example:
        >>> with trace_agent_execution(
        ...     agent_id="gl-001",
        ...     agent_name="Carbon Emissions Calculator",
        ...     execution_id="exec-12345",
        ... ) as span:
        ...     result = execute_agent()
        ...     span.set_attribute("agent.output.emissions_kg", result.emissions)
    """
    attributes = {
        "agent.id": agent_id,
        "agent.name": agent_name,
        "agent.execution.status": "running",
    }

    if tenant_id:
        attributes["tenant.id"] = tenant_id
    if execution_id:
        attributes["agent.execution.id"] = execution_id

    # Add sanitized input data (avoid sensitive info)
    if input_data:
        # Only include safe keys
        safe_keys = ["activity_type", "calculation_type", "scope", "region"]
        for key in safe_keys:
            if key in input_data:
                attributes[f"agent.input.{key}"] = str(input_data[key])

    with create_span(
        f"agent.execute.{agent_name}",
        attributes=attributes,
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        try:
            yield span
            if span:
                span.set_attribute("agent.execution.status", "completed")
        except Exception as e:
            if span:
                span.set_attribute("agent.execution.status", "failed")
                span.set_attribute("agent.execution.error", str(e))
            raise


@contextmanager
def trace_llm_call(
    model: str,
    provider: str = "anthropic",
    operation: str = "completion",
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
) -> Generator[Optional[Span], None, None]:
    """
    Create a span for LLM API calls.

    Traces interactions with LLM providers (Anthropic, OpenAI) including:
    - Model information
    - Token usage
    - Latency

    Args:
        model: Model identifier (e.g., "claude-3-opus")
        provider: LLM provider name
        operation: Operation type (completion, embedding)
        tokens_input: Input token count
        tokens_output: Output token count

    Yields:
        Span instance for recording token usage
    """
    attributes = {
        "llm.provider": provider,
        "llm.model": model,
        "llm.operation": operation,
    }

    if tokens_input is not None:
        attributes["llm.tokens.input"] = tokens_input
    if tokens_output is not None:
        attributes["llm.tokens.output"] = tokens_output

    with create_span(
        f"llm.{operation}",
        attributes=attributes,
        kind=trace.SpanKind.CLIENT,
    ) as span:
        yield span


@contextmanager
def trace_calculation(
    calculation_type: str,
    formula_id: Optional[str] = None,
    data_source: Optional[str] = None,
) -> Generator[Optional[Span], None, None]:
    """
    Create a span for zero-hallucination calculations.

    Traces calculation operations with provenance for audit:
    - Calculation type and formula
    - Data source
    - SHA-256 verification

    Args:
        calculation_type: Type of calculation (ghg, cbam, csrd)
        formula_id: Formula identifier
        data_source: Data source reference

    Yields:
        Span instance for recording calculation details
    """
    attributes = {
        "calculation.type": calculation_type,
        "calculation.verified": False,
    }

    if formula_id:
        attributes["calculation.formula_id"] = formula_id
    if data_source:
        attributes["calculation.data_source"] = data_source

    with create_span(
        f"calculation.{calculation_type}",
        attributes=attributes,
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        yield span


# =============================================================================
# Decorators for Tracing
# =============================================================================


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator for automatic function tracing.

    Args:
        name: Custom span name (defaults to function name)
        attributes: Static attributes to add to span

    Returns:
        Decorated function with automatic tracing

    Example:
        >>> @traced(attributes={"component": "data-processor"})
        ... def process_data(items):
        ...     return [transform(item) for item in items]
    """

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            with create_span(span_name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if span:
                        span.record_exception(e)
                    raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with create_span(span_name, attributes=attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    if span:
                        span.record_exception(e)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# Context Propagation
# =============================================================================


def extract_context(headers: Dict[str, str]) -> Context:
    """
    Extract trace context from incoming request headers.

    Supports W3C TraceContext and B3 headers.

    Args:
        headers: HTTP request headers

    Returns:
        Extracted OpenTelemetry context
    """
    return extract(headers)


def inject_context(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into outgoing request headers.

    Adds W3C TraceContext and B3 headers for propagation.

    Args:
        headers: HTTP request headers to modify

    Returns:
        Headers with trace context added
    """
    inject(headers)
    return headers


def get_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a hex string.

    Returns:
        Trace ID string or None if no active trace
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_span_id() -> Optional[str]:
    """
    Get the current span ID as a hex string.

    Returns:
        Span ID string or None if no active span
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, "016x")
    return None


# =============================================================================
# Shutdown
# =============================================================================


def shutdown_telemetry() -> None:
    """
    Gracefully shutdown telemetry and flush pending spans.

    Should be called during application shutdown.
    """
    global _tracer_provider, _initialized

    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry telemetry shut down successfully")
        except Exception as e:
            logger.warning(f"Error during telemetry shutdown: {e}")

    _initialized = False
    _tracer_provider = None
