# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

12 Prometheus metrics for data gateway service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_data_gateway_queries_total (Counter, labels: source, operation, status)
    2.  gl_data_gateway_query_duration_seconds (Histogram, labels: source, operation)
    3.  gl_data_gateway_cache_hits_total (Counter, labels: source)
    4.  gl_data_gateway_cache_misses_total (Counter, labels: source)
    5.  gl_data_gateway_routing_decisions_total (Counter, labels: source, strategy)
    6.  gl_data_gateway_aggregation_operations_total (Counter, labels: sources_count, status)
    7.  gl_data_gateway_schema_translations_total (Counter, labels: source, direction)
    8.  gl_data_gateway_active_queries (Gauge)
    9.  gl_data_gateway_source_health (Gauge, labels: source, status)
    10. gl_data_gateway_connection_pool_size (Gauge, labels: source)
    11. gl_data_gateway_processing_errors_total (Counter, labels: source, error_type)
    12. gl_data_gateway_response_size_bytes (Histogram, labels: source)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; data gateway metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Total queries executed by source, operation, and status
    data_gateway_queries_total = Counter(
        "gl_data_gateway_queries_total",
        "Total data gateway queries executed",
        labelnames=["source", "operation", "status"],
    )

    # 2. Query duration histogram by source and operation
    data_gateway_query_duration_seconds = Histogram(
        "gl_data_gateway_query_duration_seconds",
        "Data gateway query duration in seconds",
        labelnames=["source", "operation"],
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
        ),
    )

    # 3. Cache hits by source
    data_gateway_cache_hits_total = Counter(
        "gl_data_gateway_cache_hits_total",
        "Total data gateway cache hits",
        labelnames=["source"],
    )

    # 4. Cache misses by source
    data_gateway_cache_misses_total = Counter(
        "gl_data_gateway_cache_misses_total",
        "Total data gateway cache misses",
        labelnames=["source"],
    )

    # 5. Routing decisions by source and strategy
    data_gateway_routing_decisions_total = Counter(
        "gl_data_gateway_routing_decisions_total",
        "Total query routing decisions made",
        labelnames=["source", "strategy"],
    )

    # 6. Aggregation operations by source count and status
    data_gateway_aggregation_operations_total = Counter(
        "gl_data_gateway_aggregation_operations_total",
        "Total multi-source aggregation operations",
        labelnames=["sources_count", "status"],
    )

    # 7. Schema translations by source and direction
    data_gateway_schema_translations_total = Counter(
        "gl_data_gateway_schema_translations_total",
        "Total schema translation operations",
        labelnames=["source", "direction"],
    )

    # 8. Currently active queries
    data_gateway_active_queries = Gauge(
        "gl_data_gateway_active_queries",
        "Number of currently active queries",
    )

    # 9. Source health status gauge
    data_gateway_source_health = Gauge(
        "gl_data_gateway_source_health",
        "Data source health status (1=healthy, 0.5=degraded, 0=unhealthy)",
        labelnames=["source", "status"],
    )

    # 10. Connection pool size per source
    data_gateway_connection_pool_size = Gauge(
        "gl_data_gateway_connection_pool_size",
        "Connection pool size per data source",
        labelnames=["source"],
    )

    # 11. Processing errors by source and error type
    data_gateway_processing_errors_total = Counter(
        "gl_data_gateway_processing_errors_total",
        "Total data gateway processing errors",
        labelnames=["source", "error_type"],
    )

    # 12. Response size histogram by source
    data_gateway_response_size_bytes = Histogram(
        "gl_data_gateway_response_size_bytes",
        "Data gateway response size in bytes",
        labelnames=["source"],
        buckets=(
            256, 1024, 4096, 16384, 65536, 262144,
            1048576, 4194304, 16777216, 67108864,
        ),
    )

else:
    # No-op placeholders
    data_gateway_queries_total = None  # type: ignore[assignment]
    data_gateway_query_duration_seconds = None  # type: ignore[assignment]
    data_gateway_cache_hits_total = None  # type: ignore[assignment]
    data_gateway_cache_misses_total = None  # type: ignore[assignment]
    data_gateway_routing_decisions_total = None  # type: ignore[assignment]
    data_gateway_aggregation_operations_total = None  # type: ignore[assignment]
    data_gateway_schema_translations_total = None  # type: ignore[assignment]
    data_gateway_active_queries = None  # type: ignore[assignment]
    data_gateway_source_health = None  # type: ignore[assignment]
    data_gateway_connection_pool_size = None  # type: ignore[assignment]
    data_gateway_processing_errors_total = None  # type: ignore[assignment]
    data_gateway_response_size_bytes = None  # type: ignore[assignment]


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
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_queries_total.labels(
        source=source, operation=operation, status=status,
    ).inc()
    if duration > 0:
        data_gateway_query_duration_seconds.labels(
            source=source, operation=operation,
        ).observe(duration)


def record_cache_hit(source: str) -> None:
    """Record a cache hit event.

    Args:
        source: Data source identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_cache_hits_total.labels(source=source).inc()


def record_cache_miss(source: str) -> None:
    """Record a cache miss event.

    Args:
        source: Data source identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_cache_misses_total.labels(source=source).inc()


def record_routing_decision(source: str, strategy: str) -> None:
    """Record a query routing decision.

    Args:
        source: Target data source.
        strategy: Routing strategy (single, multi, cached).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_routing_decisions_total.labels(
        source=source, strategy=strategy,
    ).inc()


def record_aggregation(sources_count: str, status: str) -> None:
    """Record a multi-source aggregation operation.

    Args:
        sources_count: Number of sources aggregated (as string label).
        status: Operation status (success, partial_error, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_aggregation_operations_total.labels(
        sources_count=sources_count, status=status,
    ).inc()


def record_schema_translation(source: str, direction: str) -> None:
    """Record a schema translation operation.

    Args:
        source: Source schema type.
        direction: Translation direction (e.g. "erp_to_canonical").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_schema_translations_total.labels(
        source=source, direction=direction,
    ).inc()


def record_processing_error(source: str, error_type: str) -> None:
    """Record a processing error event.

    Args:
        source: Data source identifier.
        error_type: Type of error encountered.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_processing_errors_total.labels(
        source=source, error_type=error_type,
    ).inc()


def update_active_queries(delta: int) -> None:
    """Update the active queries gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        data_gateway_active_queries.inc(delta)
    elif delta < 0:
        data_gateway_active_queries.dec(abs(delta))


def update_source_health(source: str, status: str) -> None:
    """Update the source health gauge.

    Args:
        source: Data source identifier.
        status: Health status (healthy, degraded, unhealthy).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    health_value = {
        "healthy": 1.0,
        "degraded": 0.5,
        "unhealthy": 0.0,
        "unknown": -1.0,
    }.get(status, -1.0)
    data_gateway_source_health.labels(
        source=source, status=status,
    ).set(health_value)


def update_connection_pool(source: str, size: int) -> None:
    """Update the connection pool size gauge.

    Args:
        source: Data source identifier.
        size: Current pool size.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_connection_pool_size.labels(source=source).set(size)


def record_response_size(source: str, size_bytes: int) -> None:
    """Record a response size observation.

    Args:
        source: Data source identifier.
        size_bytes: Response size in bytes.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    data_gateway_response_size_bytes.labels(source=source).observe(size_bytes)


__all__ = [
    "PROMETHEUS_AVAILABLE",
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
