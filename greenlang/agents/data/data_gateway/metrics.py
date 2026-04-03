# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

12 Prometheus metrics for data gateway service monitoring with graceful
fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_data_gateway_operations_total (Counter, labels: type, tenant_id)
    2.  gl_data_gateway_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_data_gateway_validation_errors_total (Counter, labels: severity, type)
    4.  gl_data_gateway_batch_jobs_total (Counter, labels: status)
    5.  gl_data_gateway_active_jobs (Gauge)
    6.  gl_data_gateway_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_data_gateway_cache_hits_total (Counter, labels: source)
    8.  gl_data_gateway_cache_misses_total (Counter, labels: source)
    9.  gl_data_gateway_routing_decisions_total (Counter, labels: source, strategy)
    10. gl_data_gateway_aggregation_operations_total (Counter, labels: sources_count, status)
    11. gl_data_gateway_schema_translations_total (Counter, labels: source, direction)
    12. gl_data_gateway_response_size_bytes (Histogram, labels: source)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    DURATION_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_data_gateway",
    "Data Gateway",
    duration_buckets=(
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    ),
)

# Backward-compat aliases for the standard 6
data_gateway_queries_total = m.operations_total
data_gateway_query_duration_seconds = m.processing_duration
data_gateway_processing_errors_total = m.validation_errors_total
data_gateway_active_queries = m.active_jobs

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

data_gateway_cache_hits_total = m.create_custom_counter(
    "cache_hits_total",
    "Total data gateway cache hits",
    labelnames=["source"],
)

data_gateway_cache_misses_total = m.create_custom_counter(
    "cache_misses_total",
    "Total data gateway cache misses",
    labelnames=["source"],
)

data_gateway_routing_decisions_total = m.create_custom_counter(
    "routing_decisions_total",
    "Total query routing decisions made",
    labelnames=["source", "strategy"],
)

data_gateway_aggregation_operations_total = m.create_custom_counter(
    "aggregation_operations_total",
    "Total multi-source aggregation operations",
    labelnames=["sources_count", "status"],
)

data_gateway_schema_translations_total = m.create_custom_counter(
    "schema_translations_total",
    "Total schema translation operations",
    labelnames=["source", "direction"],
)

data_gateway_source_health = m.create_custom_gauge(
    "source_health",
    "Data source health status (1=healthy, 0.5=degraded, 0=unhealthy)",
    labelnames=["source", "status"],
)

data_gateway_connection_pool_size = m.create_custom_gauge(
    "connection_pool_size",
    "Connection pool size per data source",
    labelnames=["source"],
)

data_gateway_response_size_bytes = m.create_custom_histogram(
    "response_size_bytes",
    "Data gateway response size in bytes",
    buckets=(
        256, 1024, 4096, 16384, 65536, 262144,
        1048576, 4194304, 16777216, 67108864,
    ),
    labelnames=["source"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_query(
    source: str,
    operation: str,
    status: str,
    duration: float = 0.0,
) -> None:
    """Record a query execution event.

    Args:
        source: Data source identifier.
        operation: Operation type (parse, execute, batch).
        status: Result status (success, error).
        duration: Execution duration in seconds.
    """
    m.record_operation(duration, type=f"{source}:{operation}", tenant_id=status)


def record_cache_hit(source: str) -> None:
    """Record a cache hit event.

    Args:
        source: Data source identifier.
    """
    m.safe_inc(data_gateway_cache_hits_total, 1, source=source)


def record_cache_miss(source: str) -> None:
    """Record a cache miss event.

    Args:
        source: Data source identifier.
    """
    m.safe_inc(data_gateway_cache_misses_total, 1, source=source)


def record_routing_decision(source: str, strategy: str) -> None:
    """Record a query routing decision.

    Args:
        source: Target data source.
        strategy: Routing strategy (single, multi, cached).
    """
    m.safe_inc(data_gateway_routing_decisions_total, 1, source=source, strategy=strategy)


def record_aggregation(sources_count: str, status: str) -> None:
    """Record a multi-source aggregation operation.

    Args:
        sources_count: Number of sources aggregated (as string label).
        status: Operation status (success, partial_error, error).
    """
    m.safe_inc(
        data_gateway_aggregation_operations_total, 1,
        sources_count=sources_count, status=status,
    )


def record_schema_translation(source: str, direction: str) -> None:
    """Record a schema translation operation.

    Args:
        source: Source schema type.
        direction: Translation direction (e.g. "erp_to_canonical").
    """
    m.safe_inc(
        data_gateway_schema_translations_total, 1,
        source=source, direction=direction,
    )


def record_processing_error(source: str, error_type: str) -> None:
    """Record a processing error event.

    Args:
        source: Data source identifier.
        error_type: Type of error encountered.
    """
    m.record_validation_error(severity=source, type=error_type)


def update_active_queries(delta: int) -> None:
    """Update the active queries gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    m.update_active_jobs(delta)


def update_source_health(source: str, status: str) -> None:
    """Update the source health gauge.

    Args:
        source: Data source identifier.
        status: Health status (healthy, degraded, unhealthy).
    """
    health_value = {
        "healthy": 1.0,
        "degraded": 0.5,
        "unhealthy": 0.0,
        "unknown": -1.0,
    }.get(status, -1.0)
    m.safe_set(data_gateway_source_health, health_value, source=source, status=status)


def update_connection_pool(source: str, size: int) -> None:
    """Update the connection pool size gauge.

    Args:
        source: Data source identifier.
        size: Current pool size.
    """
    m.safe_set(data_gateway_connection_pool_size, size, source=source)


def record_response_size(source: str, size_bytes: int) -> None:
    """Record a response size observation.

    Args:
        source: Data source identifier.
        size_bytes: Response size in bytes.
    """
    m.safe_observe(data_gateway_response_size_bytes, size_bytes, source=source)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "data_gateway_queries_total",
    "data_gateway_query_duration_seconds",
    "data_gateway_cache_hits_total",
    "data_gateway_cache_misses_total",
    "data_gateway_routing_decisions_total",
    "data_gateway_aggregation_operations_total",
    "data_gateway_schema_translations_total",
    "data_gateway_active_queries",
    "data_gateway_source_health",
    "data_gateway_connection_pool_size",
    "data_gateway_processing_errors_total",
    "data_gateway_response_size_bytes",
    # Helper functions
    "record_query",
    "record_cache_hit",
    "record_cache_miss",
    "record_routing_decision",
    "record_aggregation",
    "record_schema_translation",
    "record_processing_error",
    "update_active_queries",
    "update_source_health",
    "update_connection_pool",
    "record_response_size",
]
