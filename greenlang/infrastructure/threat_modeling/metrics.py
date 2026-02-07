# -*- coding: utf-8 -*-
"""
Threat Modeling Prometheus Metrics - SEC-010 Phase 2

Defines all Prometheus metrics for the GreenLang threat modeling system
and provides helper functions for recording threat model events, risk
scores, and coverage metrics.

Metrics are organized by subsystem:
    - Threat Models: Total models, status counts
    - Threats: By category and severity
    - Coverage: Service model coverage
    - Mitigations: Mitigation counts and status

All metrics use the ``gl_secops_`` prefix for security operations.

Example:
    >>> from greenlang.infrastructure.threat_modeling.metrics import (
    ...     record_threat_model_created, update_threat_counts,
    ... )
    >>> record_threat_model_created("payment-service")
    >>> update_threat_counts("payment-service", {"S": 5, "T": 3})

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[misc, assignment]
    Gauge = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:

    gl_secops_threat_models_total = Gauge(
        "gl_secops_threat_models_total",
        "Total number of threat models by status",
        labelnames=["status"],
    )
    """Gauge: Total threat models. Labels: status (draft, in_review, approved, archived)."""

    gl_secops_threats_by_category = Gauge(
        "gl_secops_threats_by_category",
        "Number of threats by STRIDE category",
        labelnames=["service", "category"],
    )
    """Gauge: Threats by category. Labels: service, category (S, T, R, I, D, E)."""

    gl_secops_threats_by_severity = Gauge(
        "gl_secops_threats_by_severity",
        "Number of threats by severity level",
        labelnames=["service", "severity"],
    )
    """Gauge: Threats by severity. Labels: service, severity (critical, high, medium, low)."""

    gl_secops_threat_model_coverage = Gauge(
        "gl_secops_threat_model_coverage",
        "Percentage of services with threat models",
        labelnames=["environment"],
    )
    """Gauge: Model coverage. Labels: environment."""

    gl_secops_threats_mitigated_total = Counter(
        "gl_secops_threats_mitigated_total",
        "Total number of threats that have been mitigated",
        labelnames=["service", "category"],
    )
    """Counter: Mitigated threats. Labels: service, category."""

    gl_secops_risk_score_average = Gauge(
        "gl_secops_risk_score_average",
        "Average risk score for a service",
        labelnames=["service"],
    )
    """Gauge: Average risk score. Labels: service."""

    gl_secops_threats_critical_total = Gauge(
        "gl_secops_threats_critical_total",
        "Total number of critical severity threats",
        labelnames=["service"],
    )
    """Gauge: Critical threats. Labels: service."""

    gl_secops_threats_unmitigated_total = Gauge(
        "gl_secops_threats_unmitigated_total",
        "Total number of unmitigated threats",
        labelnames=["service"],
    )
    """Gauge: Unmitigated threats. Labels: service."""

    gl_secops_threat_model_age_days = Gauge(
        "gl_secops_threat_model_age_days",
        "Age of threat model in days since last update",
        labelnames=["service"],
    )
    """Gauge: Model age in days. Labels: service."""

    gl_secops_control_coverage = Gauge(
        "gl_secops_control_coverage",
        "Control coverage percentage for a service",
        labelnames=["service"],
    )
    """Gauge: Control coverage. Labels: service."""

    gl_secops_attack_surface_score = Gauge(
        "gl_secops_attack_surface_score",
        "Attack surface exposure score for a service (0-10)",
        labelnames=["service"],
    )
    """Gauge: Attack surface score. Labels: service."""

    gl_secops_threat_model_analysis_duration_seconds = Histogram(
        "gl_secops_threat_model_analysis_duration_seconds",
        "Duration of threat model analysis in seconds",
        labelnames=["operation"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    """Histogram: Analysis duration. Labels: operation."""

else:
    # Provide no-op stubs when prometheus_client is not installed
    logger.info(
        "prometheus_client not available; threat modeling metrics will be no-ops"
    )

    class _NoOpMetric:
        """No-op metric stub when prometheus_client is not installed."""

        def labels(self, *args, **kwargs):  # noqa: ANN
            """Return self for chaining."""
            return self

        def inc(self, amount=1):  # noqa: ANN
            """No-op increment."""

        def dec(self, amount=1):  # noqa: ANN
            """No-op decrement."""

        def set(self, value):  # noqa: ANN
            """No-op set."""

        def observe(self, amount):  # noqa: ANN
            """No-op observe."""

    gl_secops_threat_models_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threats_by_category = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threats_by_severity = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threat_model_coverage = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threats_mitigated_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_risk_score_average = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threats_critical_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threats_unmitigated_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threat_model_age_days = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_control_coverage = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_attack_surface_score = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_threat_model_analysis_duration_seconds = _NoOpMetric()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_threat_model_created(service: str, status: str = "draft") -> None:
    """Record creation of a new threat model.

    Args:
        service: The service name.
        status: Initial status of the model.
    """
    try:
        gl_secops_threat_models_total.labels(status=status).inc()
    except Exception as exc:
        logger.debug("Failed to record threat model creation metric: %s", exc)


def update_threat_model_status(old_status: str, new_status: str) -> None:
    """Update threat model status counts.

    Args:
        old_status: Previous status.
        new_status: New status.
    """
    try:
        gl_secops_threat_models_total.labels(status=old_status).dec()
        gl_secops_threat_models_total.labels(status=new_status).inc()
    except Exception as exc:
        logger.debug("Failed to update threat model status metric: %s", exc)


def update_threat_counts(
    service: str,
    category_counts: Dict[str, int],
) -> None:
    """Update threat counts by category for a service.

    Args:
        service: The service name.
        category_counts: Dict mapping category (S, T, R, I, D, E) to count.
    """
    try:
        for category, count in category_counts.items():
            gl_secops_threats_by_category.labels(
                service=service,
                category=category,
            ).set(float(count))
    except Exception as exc:
        logger.debug("Failed to update threat count metrics: %s", exc)


def update_severity_counts(
    service: str,
    severity_counts: Dict[str, int],
) -> None:
    """Update threat counts by severity for a service.

    Args:
        service: The service name.
        severity_counts: Dict mapping severity to count.
    """
    try:
        for severity, count in severity_counts.items():
            gl_secops_threats_by_severity.labels(
                service=service,
                severity=severity,
            ).set(float(count))

        # Update critical total separately
        critical_count = severity_counts.get("critical", 0)
        gl_secops_threats_critical_total.labels(service=service).set(float(critical_count))
    except Exception as exc:
        logger.debug("Failed to update severity count metrics: %s", exc)


def update_threat_model_coverage(
    environment: str,
    coverage_percentage: float,
) -> None:
    """Update threat model coverage percentage.

    Args:
        environment: The deployment environment.
        coverage_percentage: Coverage as a percentage (0-100).
    """
    try:
        gl_secops_threat_model_coverage.labels(environment=environment).set(coverage_percentage)
    except Exception as exc:
        logger.debug("Failed to update coverage metric: %s", exc)


def record_threat_mitigated(
    service: str,
    category: str,
) -> None:
    """Record that a threat has been mitigated.

    Args:
        service: The service name.
        category: The STRIDE category.
    """
    try:
        gl_secops_threats_mitigated_total.labels(
            service=service,
            category=category,
        ).inc()
    except Exception as exc:
        logger.debug("Failed to record threat mitigation metric: %s", exc)


def update_risk_score(
    service: str,
    average_risk_score: float,
) -> None:
    """Update average risk score for a service.

    Args:
        service: The service name.
        average_risk_score: Average risk score (0-10).
    """
    try:
        gl_secops_risk_score_average.labels(service=service).set(average_risk_score)
    except Exception as exc:
        logger.debug("Failed to update risk score metric: %s", exc)


def update_unmitigated_count(
    service: str,
    count: int,
) -> None:
    """Update count of unmitigated threats for a service.

    Args:
        service: The service name.
        count: Number of unmitigated threats.
    """
    try:
        gl_secops_threats_unmitigated_total.labels(service=service).set(float(count))
    except Exception as exc:
        logger.debug("Failed to update unmitigated count metric: %s", exc)


def update_threat_model_age(
    service: str,
    age_days: float,
) -> None:
    """Update threat model age in days.

    Args:
        service: The service name.
        age_days: Age in days since last update.
    """
    try:
        gl_secops_threat_model_age_days.labels(service=service).set(age_days)
    except Exception as exc:
        logger.debug("Failed to update model age metric: %s", exc)


def update_control_coverage(
    service: str,
    coverage_percentage: float,
) -> None:
    """Update control coverage for a service.

    Args:
        service: The service name.
        coverage_percentage: Coverage percentage (0-100).
    """
    try:
        gl_secops_control_coverage.labels(service=service).set(coverage_percentage)
    except Exception as exc:
        logger.debug("Failed to update control coverage metric: %s", exc)


def update_attack_surface_score(
    service: str,
    score: float,
) -> None:
    """Update attack surface score for a service.

    Args:
        service: The service name.
        score: Attack surface score (0-10).
    """
    try:
        gl_secops_attack_surface_score.labels(service=service).set(score)
    except Exception as exc:
        logger.debug("Failed to update attack surface score metric: %s", exc)


def record_analysis_duration(
    operation: str,
    duration_seconds: float,
) -> None:
    """Record duration of a threat model analysis operation.

    Args:
        operation: The operation type (stride_analysis, dfd_validation, etc.).
        duration_seconds: Duration in seconds.
    """
    try:
        gl_secops_threat_model_analysis_duration_seconds.labels(
            operation=operation
        ).observe(duration_seconds)
    except Exception as exc:
        logger.debug("Failed to record analysis duration metric: %s", exc)


__all__ = [
    # Metrics
    "gl_secops_threat_models_total",
    "gl_secops_threats_by_category",
    "gl_secops_threats_by_severity",
    "gl_secops_threat_model_coverage",
    "gl_secops_threats_mitigated_total",
    "gl_secops_risk_score_average",
    "gl_secops_threats_critical_total",
    "gl_secops_threats_unmitigated_total",
    "gl_secops_threat_model_age_days",
    "gl_secops_control_coverage",
    "gl_secops_attack_surface_score",
    "gl_secops_threat_model_analysis_duration_seconds",
    # Helper functions
    "record_threat_model_created",
    "update_threat_model_status",
    "update_threat_counts",
    "update_severity_counts",
    "update_threat_model_coverage",
    "record_threat_mitigated",
    "update_risk_score",
    "update_unmitigated_count",
    "update_threat_model_age",
    "update_control_coverage",
    "update_attack_surface_score",
    "record_analysis_duration",
]
