"""
GreenLang Framework - Observability Module
==========================================

Comprehensive observability infrastructure for GreenLang agents.
Provides OpenTelemetry tracing, Prometheus metrics, structured logging,
and health check capabilities.

Components:
- Tracing: OpenTelemetry-based distributed tracing with Jaeger/Zipkin export
- Metrics: Prometheus metrics with counters, histograms, gauges, and summaries
- Logging: JSON-structured logging with correlation IDs and sensitive data redaction
- Health: Kubernetes-compatible health probes (liveness, readiness, startup)

Standards Compliance:
- OpenTelemetry Specification v1.0
- Prometheus Exposition Format
- Kubernetes Health Check Best Practices
- GDPR-compliant logging (sensitive data redaction)

Example:
    >>> from greenlang_observability import (
    ...     TracingManager,
    ...     MetricsRegistry,
    ...     StructuredLogger,
    ...     HealthCheckManager,
    ... )
    >>>
    >>> # Initialize tracing
    >>> tracing = TracingManager(service_name="gl-006-heatreclaim")
    >>>
    >>> # Initialize metrics
    >>> metrics = MetricsRegistry(namespace="greenlang")
    >>>
    >>> # Initialize logging
    >>> logger = StructuredLogger(service_name="gl-006-heatreclaim")
    >>>
    >>> # Initialize health checks
    >>> health = HealthCheckManager()

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Technologies"

from .tracing import (
    TracingManager,
    traced,
    SpanKind,
    TracingConfig,
    TraceContext,
    get_current_span,
    get_current_trace_id,
    inject_trace_context,
    extract_trace_context,
)

from .metrics import (
    MetricsRegistry,
    Counter,
    Histogram,
    Gauge,
    Summary,
    MetricsConfig,
    get_default_registry,
    calculation_counter,
    calculation_latency,
    queue_depth,
    active_tasks,
    response_size,
)

from .logging import (
    StructuredLogger,
    LogConfig,
    LogLevel,
    CorrelationContext,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    redact_sensitive_data,
)

from .health import (
    HealthCheckManager,
    HealthStatus,
    HealthCheckResult,
    DependencyCheck,
    LivenessCheck,
    ReadinessCheck,
    StartupCheck,
)

__all__ = [
    # Version
    "__version__",
    # Tracing
    "TracingManager",
    "traced",
    "SpanKind",
    "TracingConfig",
    "TraceContext",
    "get_current_span",
    "get_current_trace_id",
    "inject_trace_context",
    "extract_trace_context",
    # Metrics
    "MetricsRegistry",
    "Counter",
    "Histogram",
    "Gauge",
    "Summary",
    "MetricsConfig",
    "get_default_registry",
    "calculation_counter",
    "calculation_latency",
    "queue_depth",
    "active_tasks",
    "response_size",
    # Logging
    "StructuredLogger",
    "LogConfig",
    "LogLevel",
    "CorrelationContext",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "redact_sensitive_data",
    # Health
    "HealthCheckManager",
    "HealthStatus",
    "HealthCheckResult",
    "DependencyCheck",
    "LivenessCheck",
    "ReadinessCheck",
    "StartupCheck",
]
