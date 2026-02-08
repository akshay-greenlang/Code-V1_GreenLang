# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-007: Agent Registry & Service Catalog

12 Prometheus metrics for agent registry monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_agent_registry_operations_total (Counter)
    2.  gl_agent_registry_operation_duration_seconds (Histogram)
    3.  gl_agent_registry_registrations_total (Counter)
    4.  gl_agent_registry_unregistrations_total (Counter)
    5.  gl_agent_registry_queries_total (Counter)
    6.  gl_agent_registry_query_results_total (Counter)
    7.  gl_agent_registry_health_checks_total (Counter)
    8.  gl_agent_registry_agents_total (Gauge)
    9.  gl_agent_registry_agents_by_health (Gauge)
    10. gl_agent_registry_hot_reloads_total (Counter)
    11. gl_agent_registry_dependency_resolutions_total (Counter)
    12. gl_agent_registry_dependency_depth (Histogram)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
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
        "prometheus_client not installed; agent registry metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Operations count
    agent_registry_operations_total = Counter(
        "gl_agent_registry_operations_total",
        "Total agent registry operations performed",
        labelnames=["operation", "result"],
    )

    # 2. Operation duration
    agent_registry_operation_duration_seconds = Histogram(
        "gl_agent_registry_operation_duration_seconds",
        "Agent registry operation duration in seconds",
        labelnames=["operation"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    # 3. Registrations count
    agent_registry_registrations_total = Counter(
        "gl_agent_registry_registrations_total",
        "Total agent registrations performed",
        labelnames=["layer"],
    )

    # 4. Unregistrations count
    agent_registry_unregistrations_total = Counter(
        "gl_agent_registry_unregistrations_total",
        "Total agent unregistrations performed",
    )

    # 5. Queries count
    agent_registry_queries_total = Counter(
        "gl_agent_registry_queries_total",
        "Total agent registry queries performed",
        labelnames=["filter_type"],
    )

    # 6. Query results count
    agent_registry_query_results_total = Counter(
        "gl_agent_registry_query_results_total",
        "Total agent registry query results returned",
    )

    # 7. Health checks count
    agent_registry_health_checks_total = Counter(
        "gl_agent_registry_health_checks_total",
        "Total agent health checks performed",
        labelnames=["status"],
    )

    # 8. Total agents gauge
    agent_registry_agents_total = Gauge(
        "gl_agent_registry_agents_total",
        "Current number of agents in registry",
    )

    # 9. Agents by health status
    agent_registry_agents_by_health = Gauge(
        "gl_agent_registry_agents_by_health",
        "Number of agents by health status",
        labelnames=["health_status"],
    )

    # 10. Hot reloads count
    agent_registry_hot_reloads_total = Counter(
        "gl_agent_registry_hot_reloads_total",
        "Total agent hot-reloads performed",
    )

    # 11. Dependency resolutions count
    agent_registry_dependency_resolutions_total = Counter(
        "gl_agent_registry_dependency_resolutions_total",
        "Total dependency resolution operations",
        labelnames=["result"],
    )

    # 12. Dependency depth
    agent_registry_dependency_depth = Histogram(
        "gl_agent_registry_dependency_depth",
        "Depth of agent dependency chains",
        buckets=(1, 2, 3, 4, 5, 7, 10, 15, 20),
    )

else:
    # No-op placeholders
    agent_registry_operations_total = None  # type: ignore[assignment]
    agent_registry_operation_duration_seconds = None  # type: ignore[assignment]
    agent_registry_registrations_total = None  # type: ignore[assignment]
    agent_registry_unregistrations_total = None  # type: ignore[assignment]
    agent_registry_queries_total = None  # type: ignore[assignment]
    agent_registry_query_results_total = None  # type: ignore[assignment]
    agent_registry_health_checks_total = None  # type: ignore[assignment]
    agent_registry_agents_total = None  # type: ignore[assignment]
    agent_registry_agents_by_health = None  # type: ignore[assignment]
    agent_registry_hot_reloads_total = None  # type: ignore[assignment]
    agent_registry_dependency_resolutions_total = None  # type: ignore[assignment]
    agent_registry_dependency_depth = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_operation(operation: str, result: str, duration_seconds: float) -> None:
    """Record an agent registry operation.

    Args:
        operation: Operation name (register, query, health_check, etc.).
        result: Operation result ("success" or "error").
        duration_seconds: Operation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_operations_total.labels(operation=operation, result=result).inc()
    agent_registry_operation_duration_seconds.labels(operation=operation).observe(
        duration_seconds,
    )


def record_registration(layer: str) -> None:
    """Record an agent registration.

    Args:
        layer: The layer of the registered agent.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_registrations_total.labels(layer=layer).inc()


def record_unregistration() -> None:
    """Record an agent unregistration."""
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_unregistrations_total.inc()


def record_query(filter_type: str) -> None:
    """Record a registry query.

    Args:
        filter_type: Type of filter used (layer, sector, capability, text, all).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_queries_total.labels(filter_type=filter_type).inc()


def record_query_results(count: int) -> None:
    """Record query results count.

    Args:
        count: Number of results returned.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_query_results_total.inc(count)


def record_health_check(status: str) -> None:
    """Record a health check result.

    Args:
        status: Resulting health status.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_health_checks_total.labels(status=status).inc()


def update_agents_count(count: int) -> None:
    """Set the total agents gauge.

    Args:
        count: Current number of agents.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_agents_total.set(count)


def update_agents_by_health(health_status: str, count: int) -> None:
    """Set the agents-by-health gauge for a status.

    Args:
        health_status: The health status label.
        count: Number of agents with this status.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_agents_by_health.labels(health_status=health_status).set(count)


def record_hot_reload() -> None:
    """Record a hot-reload operation."""
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_hot_reloads_total.inc()


def record_dependency_resolution(result: str) -> None:
    """Record a dependency resolution operation.

    Args:
        result: Resolution result ("success" or "error").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_dependency_resolutions_total.labels(result=result).inc()


def record_dependency_depth(depth: int) -> None:
    """Record a dependency chain depth measurement.

    Args:
        depth: Depth of the dependency chain.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    agent_registry_dependency_depth.observe(depth)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "agent_registry_operations_total",
    "agent_registry_operation_duration_seconds",
    "agent_registry_registrations_total",
    "agent_registry_unregistrations_total",
    "agent_registry_queries_total",
    "agent_registry_query_results_total",
    "agent_registry_health_checks_total",
    "agent_registry_agents_total",
    "agent_registry_agents_by_health",
    "agent_registry_hot_reloads_total",
    "agent_registry_dependency_resolutions_total",
    "agent_registry_dependency_depth",
    # Helper functions
    "record_operation",
    "record_registration",
    "record_unregistration",
    "record_query",
    "record_query_results",
    "record_health_check",
    "update_agents_count",
    "update_agents_by_health",
    "record_hot_reload",
    "record_dependency_resolution",
    "record_dependency_depth",
]
