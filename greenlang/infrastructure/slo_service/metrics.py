# -*- coding: utf-8 -*-
"""
SLO Service Metrics - OBS-005: SLO/SLI Definitions & Error Budget Management

Prometheus metrics for the SLO/SLI service covering SLO evaluations,
error budget tracking, burn rate calculations, alert activity,
recording rule generation, and compliance reporting.

All metrics use the ``gl_slo_`` prefix to align with the GreenLang
monitoring namespace. When ``prometheus_client`` is not installed the
helper functions degrade to no-ops.

Example:
    >>> from greenlang.infrastructure.slo_service.metrics import (
    ...     record_evaluation, record_budget_remaining,
    ... )
    >>> record_evaluation("api-gateway", "availability", "pass")
    >>> record_budget_remaining("api-availability-99.9", 75.5)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus client import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[assignment, misc]
    Gauge = None  # type: ignore[assignment, misc]
    Histogram = None  # type: ignore[assignment, misc]
    logger.debug(
        "prometheus_client not installed; SLO metrics operate in no-op mode"
    )


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    gl_slo_evaluations_total = Counter(
        "gl_slo_evaluations_total",
        "Total SLO evaluation cycles completed",
        ["service", "sli_type", "result"],
    )

    gl_slo_evaluation_duration_seconds = Histogram(
        "gl_slo_evaluation_duration_seconds",
        "SLO evaluation duration in seconds",
        ["service"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    gl_slo_error_budget_remaining_percent = Gauge(
        "gl_slo_error_budget_remaining_percent",
        "Error budget remaining as a percentage (0-100)",
        ["slo_id", "service"],
    )

    gl_slo_burn_rate = Gauge(
        "gl_slo_burn_rate",
        "Current burn rate for an SLO and window",
        ["slo_id", "window"],
    )

    gl_slo_definitions_total = Gauge(
        "gl_slo_definitions_total",
        "Total number of active SLO definitions",
        ["service"],
    )

    gl_slo_compliance_percent = Gauge(
        "gl_slo_compliance_percent",
        "Percentage of SLOs meeting their targets",
    )

    gl_slo_alerts_fired_total = Counter(
        "gl_slo_alerts_fired_total",
        "Total SLO-related alerts fired",
        ["slo_id", "severity", "alert_type"],
    )

    gl_slo_recording_rules_generated_total = Counter(
        "gl_slo_recording_rules_generated_total",
        "Total recording rule generation events",
    )

    gl_slo_budget_snapshots_total = Counter(
        "gl_slo_budget_snapshots_total",
        "Total error budget snapshots persisted",
        ["slo_id"],
    )

    gl_slo_reports_generated_total = Counter(
        "gl_slo_reports_generated_total",
        "Total compliance reports generated",
        ["report_type"],
    )
else:
    gl_slo_evaluations_total = None  # type: ignore[assignment]
    gl_slo_evaluation_duration_seconds = None  # type: ignore[assignment]
    gl_slo_error_budget_remaining_percent = None  # type: ignore[assignment]
    gl_slo_burn_rate = None  # type: ignore[assignment]
    gl_slo_definitions_total = None  # type: ignore[assignment]
    gl_slo_compliance_percent = None  # type: ignore[assignment]
    gl_slo_alerts_fired_total = None  # type: ignore[assignment]
    gl_slo_recording_rules_generated_total = None  # type: ignore[assignment]
    gl_slo_budget_snapshots_total = None  # type: ignore[assignment]
    gl_slo_reports_generated_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_evaluation(
    service: str,
    sli_type: str,
    result: str,
    duration_s: float = 0.0,
) -> None:
    """Record an SLO evaluation cycle.

    Args:
        service: Service name being evaluated.
        sli_type: SLI type (availability, latency, etc.).
        result: Evaluation result (pass, fail, error).
        duration_s: Evaluation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_evaluations_total.labels(
        service=service, sli_type=sli_type, result=result,
    ).inc()
    gl_slo_evaluation_duration_seconds.labels(service=service).observe(
        duration_s,
    )


def record_budget_remaining(
    slo_id: str,
    service: str,
    remaining_percent: float,
) -> None:
    """Update the error budget remaining gauge for an SLO.

    Args:
        slo_id: SLO identifier.
        service: Service name.
        remaining_percent: Budget remaining as percentage (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_error_budget_remaining_percent.labels(
        slo_id=slo_id, service=service,
    ).set(remaining_percent)


def record_burn_rate(slo_id: str, window: str, rate: float) -> None:
    """Update the burn rate gauge for an SLO and window.

    Args:
        slo_id: SLO identifier.
        window: Burn rate window (fast, medium, slow).
        rate: Current burn rate multiplier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_burn_rate.labels(slo_id=slo_id, window=window).set(rate)


def update_definitions_count(service: str, count: int) -> None:
    """Set the total number of active SLO definitions for a service.

    Args:
        service: Service name.
        count: Number of active SLO definitions.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_definitions_total.labels(service=service).set(count)


def update_compliance_percent(percent: float) -> None:
    """Set the overall SLO compliance percentage.

    Args:
        percent: Percentage of SLOs meeting targets (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_compliance_percent.set(percent)


def record_alert_fired(
    slo_id: str,
    severity: str,
    alert_type: str,
) -> None:
    """Record an SLO-related alert firing.

    Args:
        slo_id: SLO identifier.
        severity: Alert severity (critical, warning, info).
        alert_type: Alert type (burn_rate, budget_warning, budget_critical, budget_exhausted).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_alerts_fired_total.labels(
        slo_id=slo_id, severity=severity, alert_type=alert_type,
    ).inc()


def record_recording_rules_generated() -> None:
    """Record a recording rule generation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_recording_rules_generated_total.inc()


def record_budget_snapshot(slo_id: str) -> None:
    """Record a budget snapshot persistence event.

    Args:
        slo_id: SLO identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_budget_snapshots_total.labels(slo_id=slo_id).inc()


def record_report_generated(report_type: str) -> None:
    """Record a compliance report generation event.

    Args:
        report_type: Report type (weekly, monthly, quarterly).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_slo_reports_generated_total.labels(report_type=report_type).inc()
