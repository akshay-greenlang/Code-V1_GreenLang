# -*- coding: utf-8 -*-
"""
GreenLang Observability Package

This package provides comprehensive observability infrastructure including:
- Metrics collection (Prometheus)
- Structured logging (JSON logs)
- Distributed tracing (OpenTelemetry)
- Health checks (Kubernetes-compatible)
- Performance monitoring

Note: This package re-exports from greenlang.telemetry for backwards compatibility
while providing a more organized observability-focused interface.
"""

# Import from telemetry module (existing infrastructure)
from greenlang.telemetry.metrics import (
    MetricsCollector,
    MetricType,
    CustomMetric,
    MetricsAggregator,
    get_metrics_collector,
    get_metrics_aggregator,
    get_metrics_registry,
    track_execution,
    track_resource,
    track_api_request,
    track_cache,
    track_database_query,
    # Core metrics
    pipeline_runs,
    pipeline_duration,
    active_executions,
    resource_usage,
    cpu_usage,
    memory_usage,
    disk_usage,
    pack_operations,
    pack_size,
    api_requests,
    api_latency,
    errors,
    cache_hits,
    cache_misses,
    db_queries,
    db_query_duration,
    db_connections,
    system_info,
)

from greenlang.telemetry.logging import (
    LogLevel,
    LogContext,
    LogEntry,
    StructuredLogger,
    LogFormatter,
    LogAggregator,
    LogShipper,
    get_logger,
    get_log_aggregator,
    get_log_shipper,
    configure_logging,
)

from greenlang.telemetry.tracing import (
    SpanKind,
    SpanContext,
    TraceConfig,
    TracingManager,
    TraceContextManager,
    DistributedTracer,
    SamplingStrategy,
    RateSampler,
    ErrorSampler,
    CompositeSampler,
    get_tracing_manager,
    get_tracer,
    get_trace_context_manager,
    trace_operation,
    add_span_attributes,
    add_span_event,
    set_span_status,
    create_span,
)

from greenlang.telemetry.health import (
    HealthStatus,
    CheckType,
    HealthCheckResult,
    HealthReport,
    HealthCheck,
    LivenessCheck,
    ReadinessCheck,
    DatabaseHealthCheck,
    DiskSpaceHealthCheck,
    MemoryHealthCheck,
    CPUHealthCheck,
    ServiceHealthCheck,
    HealthChecker,
    get_health_checker,
    get_health_status,
    register_health_check,
    check_health,
    check_health_async,
)

from greenlang.telemetry.performance import (
    PerformanceMetric,
    PerformanceProfile,
    PerformanceMonitor,
    PerformanceAnalyzer,
    get_performance_monitor,
    get_performance_stats,
    profile_function,
    measure_latency,
    track_memory,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "MetricType",
    "CustomMetric",
    "MetricsAggregator",
    "get_metrics_collector",
    "get_metrics_aggregator",
    "get_metrics_registry",
    "track_execution",
    "track_resource",
    "track_api_request",
    "track_cache",
    "track_database_query",
    "pipeline_runs",
    "pipeline_duration",
    "active_executions",
    "resource_usage",
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "pack_operations",
    "pack_size",
    "api_requests",
    "api_latency",
    "errors",
    "cache_hits",
    "cache_misses",
    "db_queries",
    "db_query_duration",
    "db_connections",
    "system_info",
    # Logging
    "LogLevel",
    "LogContext",
    "LogEntry",
    "StructuredLogger",
    "LogFormatter",
    "LogAggregator",
    "LogShipper",
    "get_logger",
    "get_log_aggregator",
    "get_log_shipper",
    "configure_logging",
    # Tracing
    "SpanKind",
    "SpanContext",
    "TraceConfig",
    "TracingManager",
    "TraceContextManager",
    "DistributedTracer",
    "SamplingStrategy",
    "RateSampler",
    "ErrorSampler",
    "CompositeSampler",
    "get_tracing_manager",
    "get_tracer",
    "get_trace_context_manager",
    "trace_operation",
    "add_span_attributes",
    "add_span_event",
    "set_span_status",
    "create_span",
    # Health
    "HealthStatus",
    "CheckType",
    "HealthCheckResult",
    "HealthReport",
    "HealthCheck",
    "LivenessCheck",
    "ReadinessCheck",
    "DatabaseHealthCheck",
    "DiskSpaceHealthCheck",
    "MemoryHealthCheck",
    "CPUHealthCheck",
    "ServiceHealthCheck",
    "HealthChecker",
    "get_health_checker",
    "get_health_status",
    "register_health_check",
    "check_health",
    "check_health_async",
    # Performance
    "PerformanceMetric",
    "PerformanceProfile",
    "PerformanceMonitor",
    "PerformanceAnalyzer",
    "get_performance_monitor",
    "get_performance_stats",
    "profile_function",
    "measure_latency",
    "track_memory",
]

# Version
__version__ = "1.0.0"
