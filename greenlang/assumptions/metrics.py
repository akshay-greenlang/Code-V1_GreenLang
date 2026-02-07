# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-004: Assumptions Registry

12 Prometheus metrics for assumptions registry monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_assumptions_operations_total (Counter)
    2.  gl_assumptions_operation_duration_seconds (Histogram)
    3.  gl_assumptions_validations_total (Counter)
    4.  gl_assumptions_validation_failures_total (Counter)
    5.  gl_assumptions_scenario_accesses_total (Counter)
    6.  gl_assumptions_version_creates_total (Counter)
    7.  gl_assumptions_change_log_entries (Gauge)
    8.  gl_assumptions_total (Gauge)
    9.  gl_assumptions_scenarios_total (Gauge)
    10. gl_assumptions_cache_hits_total (Counter)
    11. gl_assumptions_cache_misses_total (Counter)
    12. gl_assumptions_dependency_depth (Histogram)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
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
        "prometheus_client not installed; assumptions metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Operations count
    assumptions_operations_total = Counter(
        "gl_assumptions_operations_total",
        "Total assumptions registry operations performed",
        labelnames=["operation", "result"],
    )

    # 2. Operation duration
    assumptions_operation_duration_seconds = Histogram(
        "gl_assumptions_operation_duration_seconds",
        "Assumptions registry operation duration in seconds",
        labelnames=["operation"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    # 3. Validations count
    assumptions_validations_total = Counter(
        "gl_assumptions_validations_total",
        "Total assumption validations performed",
        labelnames=["result"],
    )

    # 4. Validation failures by rule
    assumptions_validation_failures_total = Counter(
        "gl_assumptions_validation_failures_total",
        "Total assumption validation failures by rule",
        labelnames=["rule_id"],
    )

    # 5. Scenario accesses
    assumptions_scenario_accesses_total = Counter(
        "gl_assumptions_scenario_accesses_total",
        "Total scenario value accesses",
        labelnames=["scenario_type"],
    )

    # 6. Version creates
    assumptions_version_creates_total = Counter(
        "gl_assumptions_version_creates_total",
        "Total assumption version creates",
    )

    # 7. Change log entries gauge
    assumptions_change_log_entries = Gauge(
        "gl_assumptions_change_log_entries",
        "Current number of change log entries",
    )

    # 8. Total assumptions gauge
    assumptions_total = Gauge(
        "gl_assumptions_total",
        "Current number of assumptions in registry",
    )

    # 9. Total scenarios gauge
    assumptions_scenarios_total = Gauge(
        "gl_assumptions_scenarios_total",
        "Current number of scenarios in registry",
    )

    # 10. Cache hits
    assumptions_cache_hits_total = Counter(
        "gl_assumptions_cache_hits_total",
        "Total assumptions cache hits",
    )

    # 11. Cache misses
    assumptions_cache_misses_total = Counter(
        "gl_assumptions_cache_misses_total",
        "Total assumptions cache misses",
    )

    # 12. Dependency depth
    assumptions_dependency_depth = Histogram(
        "gl_assumptions_dependency_depth",
        "Depth of assumption dependency chains",
        buckets=(1, 2, 3, 4, 5, 7, 10, 15, 20),
    )

else:
    # No-op placeholders
    assumptions_operations_total = None  # type: ignore[assignment]
    assumptions_operation_duration_seconds = None  # type: ignore[assignment]
    assumptions_validations_total = None  # type: ignore[assignment]
    assumptions_validation_failures_total = None  # type: ignore[assignment]
    assumptions_scenario_accesses_total = None  # type: ignore[assignment]
    assumptions_version_creates_total = None  # type: ignore[assignment]
    assumptions_change_log_entries = None  # type: ignore[assignment]
    assumptions_total = None  # type: ignore[assignment]
    assumptions_scenarios_total = None  # type: ignore[assignment]
    assumptions_cache_hits_total = None  # type: ignore[assignment]
    assumptions_cache_misses_total = None  # type: ignore[assignment]
    assumptions_dependency_depth = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_operation(operation: str, result: str, duration_seconds: float) -> None:
    """Record an assumptions registry operation.

    Args:
        operation: Operation name (create, get, update, delete, etc.).
        result: Operation result ("success" or "error").
        duration_seconds: Operation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_operations_total.labels(operation=operation, result=result).inc()
    assumptions_operation_duration_seconds.labels(operation=operation).observe(
        duration_seconds,
    )


def record_validation(result: str) -> None:
    """Record a validation execution.

    Args:
        result: Validation result ("pass" or "fail").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_validations_total.labels(result=result).inc()


def record_validation_failure(rule_id: str) -> None:
    """Record a validation failure for a specific rule.

    Args:
        rule_id: The rule identifier that failed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_validation_failures_total.labels(rule_id=rule_id).inc()


def record_scenario_access(scenario_type: str) -> None:
    """Record a scenario value access.

    Args:
        scenario_type: Type of scenario accessed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_scenario_accesses_total.labels(scenario_type=scenario_type).inc()


def record_version_create() -> None:
    """Record a new assumption version creation."""
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_version_creates_total.inc()


def update_change_log_count(count: int) -> None:
    """Set the change log entries gauge.

    Args:
        count: Current number of change log entries.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_change_log_entries.set(count)


def update_assumptions_count(count: int) -> None:
    """Set the total assumptions gauge.

    Args:
        count: Current number of assumptions.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_total.set(count)


def update_scenarios_count(count: int) -> None:
    """Set the total scenarios gauge.

    Args:
        count: Current number of scenarios.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_scenarios_total.set(count)


def record_cache_hit() -> None:
    """Record an assumptions cache hit."""
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_cache_hits_total.inc()


def record_cache_miss() -> None:
    """Record an assumptions cache miss."""
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_cache_misses_total.inc()


def record_dependency_depth(depth: int) -> None:
    """Record a dependency chain depth measurement.

    Args:
        depth: Depth of the dependency chain.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    assumptions_dependency_depth.observe(depth)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "assumptions_operations_total",
    "assumptions_operation_duration_seconds",
    "assumptions_validations_total",
    "assumptions_validation_failures_total",
    "assumptions_scenario_accesses_total",
    "assumptions_version_creates_total",
    "assumptions_change_log_entries",
    "assumptions_total",
    "assumptions_scenarios_total",
    "assumptions_cache_hits_total",
    "assumptions_cache_misses_total",
    "assumptions_dependency_depth",
    # Helper functions
    "record_operation",
    "record_validation",
    "record_validation_failure",
    "record_scenario_access",
    "record_version_create",
    "update_change_log_count",
    "update_assumptions_count",
    "update_scenarios_count",
    "record_cache_hit",
    "record_cache_miss",
    "record_dependency_depth",
]
