# -*- coding: utf-8 -*-
"""
Security Training Prometheus Metrics - SEC-010

Defines all Prometheus metrics for the GreenLang security training platform
and provides helper functions for recording training events, phishing campaign
results, and security score updates.

Metrics are organized by subsystem:
    - Training: Completion rates, completions total, overdue count
    - Phishing: Campaign totals, click rates, report rates
    - Security Score: Average scores by team

All metrics use the ``gl_secops_`` prefix to align with the security operations
namespace.

Example:
    >>> from greenlang.infrastructure.security_training.metrics import (
    ...     record_training_completion, update_phishing_metrics,
    ...     update_security_score,
    ... )
    >>> record_training_completion("owasp_top_10", passed=True)
    >>> update_phishing_metrics("campaign-1", click_rate=0.05, report_rate=0.30)
    >>> update_security_score("team-1", average_score=78.5)
"""

from __future__ import annotations

import logging
from typing import Optional

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

    # Training Metrics
    gl_secops_training_completion_rate = Gauge(
        "gl_secops_training_completion_rate",
        "Training completion rate by course (0.0-1.0)",
        labelnames=["course"],
    )
    """Gauge: Completion rate per course. Labels: course."""

    gl_secops_training_completions_total = Counter(
        "gl_secops_training_completions_total",
        "Total training completions",
        labelnames=["course", "passed"],
    )
    """Counter: Total completions. Labels: course, passed (true/false)."""

    gl_secops_training_overdue_total = Gauge(
        "gl_secops_training_overdue_total",
        "Total overdue training assignments",
        labelnames=[],
    )
    """Gauge: Total overdue training count."""

    gl_secops_training_attempts_total = Counter(
        "gl_secops_training_attempts_total",
        "Total quiz attempts",
        labelnames=["course"],
    )
    """Counter: Quiz attempts per course. Labels: course."""

    gl_secops_training_score_histogram = Histogram(
        "gl_secops_training_score_histogram",
        "Distribution of training assessment scores",
        labelnames=["course"],
        buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    )
    """Histogram: Assessment score distribution. Labels: course."""

    # Phishing Campaign Metrics
    gl_secops_phishing_campaigns_total = Counter(
        "gl_secops_phishing_campaigns_total",
        "Total phishing campaigns by status",
        labelnames=["status"],
    )
    """Counter: Campaign count by status. Labels: status."""

    gl_secops_phishing_emails_sent_total = Counter(
        "gl_secops_phishing_emails_sent_total",
        "Total phishing simulation emails sent",
        labelnames=["campaign"],
    )
    """Counter: Emails sent per campaign. Labels: campaign."""

    gl_secops_phishing_click_rate = Gauge(
        "gl_secops_phishing_click_rate",
        "Phishing click rate by campaign (0.0-1.0)",
        labelnames=["campaign"],
    )
    """Gauge: Click rate per campaign. Labels: campaign."""

    gl_secops_phishing_report_rate = Gauge(
        "gl_secops_phishing_report_rate",
        "Phishing report rate by campaign (0.0-1.0)",
        labelnames=["campaign"],
    )
    """Gauge: Report rate per campaign. Labels: campaign."""

    gl_secops_phishing_credential_rate = Gauge(
        "gl_secops_phishing_credential_rate",
        "Credential submission rate by campaign (0.0-1.0)",
        labelnames=["campaign"],
    )
    """Gauge: Credential submission rate. Labels: campaign."""

    # Security Score Metrics
    gl_secops_security_score_average = Gauge(
        "gl_secops_security_score_average",
        "Average security score by team (0-100)",
        labelnames=["team"],
    )
    """Gauge: Average security score per team. Labels: team."""

    gl_secops_security_score_distribution = Histogram(
        "gl_secops_security_score_distribution",
        "Distribution of employee security scores",
        labelnames=[],
        buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    )
    """Histogram: Security score distribution across organization."""

    gl_secops_at_risk_users_total = Gauge(
        "gl_secops_at_risk_users_total",
        "Total at-risk users below score threshold",
        labelnames=["team"],
    )
    """Gauge: At-risk user count. Labels: team."""

    # Compliance Metrics
    gl_secops_compliance_rate = Gauge(
        "gl_secops_compliance_rate",
        "Training compliance rate (0.0-1.0)",
        labelnames=["scope"],
    )
    """Gauge: Compliance rate by scope. Labels: scope (org, team)."""

    gl_secops_certificates_issued_total = Counter(
        "gl_secops_certificates_issued_total",
        "Total certificates issued",
        labelnames=["course"],
    )
    """Counter: Certificates issued per course. Labels: course."""

    gl_secops_certificates_expiring_total = Gauge(
        "gl_secops_certificates_expiring_total",
        "Certificates expiring within 30 days",
        labelnames=[],
    )
    """Gauge: Certificates expiring soon."""

else:
    # No-op stubs when prometheus_client is not installed
    logger.info(
        "prometheus_client not available; security training metrics will be no-ops"
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

    gl_secops_training_completion_rate = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_training_completions_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_training_overdue_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_training_attempts_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_training_score_histogram = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_phishing_campaigns_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_phishing_emails_sent_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_phishing_click_rate = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_phishing_report_rate = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_phishing_credential_rate = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_security_score_average = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_security_score_distribution = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_at_risk_users_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_compliance_rate = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_certificates_issued_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_certificates_expiring_total = _NoOpMetric()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_training_completion(
    course: str,
    passed: bool,
    score: Optional[int] = None,
) -> None:
    """Record a training completion event.

    Updates the completion counter and optionally records the score.

    Args:
        course: Course identifier.
        passed: Whether the user passed.
        score: Optional assessment score.
    """
    try:
        gl_secops_training_completions_total.labels(
            course=course,
            passed="true" if passed else "false",
        ).inc()

        if score is not None:
            gl_secops_training_score_histogram.labels(course=course).observe(score)
    except Exception as exc:
        logger.debug("Failed to record training completion metric: %s", exc)


def record_training_attempt(course: str) -> None:
    """Record a quiz attempt.

    Args:
        course: Course identifier.
    """
    try:
        gl_secops_training_attempts_total.labels(course=course).inc()
    except Exception as exc:
        logger.debug("Failed to record training attempt metric: %s", exc)


def update_training_completion_rate(course: str, rate: float) -> None:
    """Update the completion rate gauge for a course.

    Args:
        course: Course identifier.
        rate: Completion rate (0.0-1.0).
    """
    try:
        gl_secops_training_completion_rate.labels(course=course).set(rate)
    except Exception as exc:
        logger.debug("Failed to update completion rate metric: %s", exc)


def update_overdue_count(count: int) -> None:
    """Update the overdue training count.

    Args:
        count: Number of overdue assignments.
    """
    try:
        gl_secops_training_overdue_total.set(count)
    except Exception as exc:
        logger.debug("Failed to update overdue count metric: %s", exc)


def record_campaign_status(status: str) -> None:
    """Record a phishing campaign status change.

    Args:
        status: Campaign status.
    """
    try:
        gl_secops_phishing_campaigns_total.labels(status=status).inc()
    except Exception as exc:
        logger.debug("Failed to record campaign status metric: %s", exc)


def record_phishing_emails_sent(campaign: str, count: int) -> None:
    """Record phishing emails sent.

    Args:
        campaign: Campaign identifier.
        count: Number of emails sent.
    """
    try:
        for _ in range(count):
            gl_secops_phishing_emails_sent_total.labels(campaign=campaign).inc()
    except Exception as exc:
        logger.debug("Failed to record phishing emails metric: %s", exc)


def update_phishing_metrics(
    campaign: str,
    click_rate: float,
    report_rate: float,
    credential_rate: float = 0.0,
) -> None:
    """Update phishing campaign rate metrics.

    Args:
        campaign: Campaign identifier.
        click_rate: Click rate (0.0-1.0).
        report_rate: Report rate (0.0-1.0).
        credential_rate: Credential submission rate (0.0-1.0).
    """
    try:
        gl_secops_phishing_click_rate.labels(campaign=campaign).set(click_rate)
        gl_secops_phishing_report_rate.labels(campaign=campaign).set(report_rate)
        gl_secops_phishing_credential_rate.labels(campaign=campaign).set(credential_rate)
    except Exception as exc:
        logger.debug("Failed to update phishing metrics: %s", exc)


def update_security_score(team: str, average_score: float) -> None:
    """Update average security score for a team.

    Args:
        team: Team identifier.
        average_score: Average score (0-100).
    """
    try:
        gl_secops_security_score_average.labels(team=team).set(average_score)
    except Exception as exc:
        logger.debug("Failed to update security score metric: %s", exc)


def record_security_score(score: int) -> None:
    """Record a security score for distribution tracking.

    Args:
        score: Security score (0-100).
    """
    try:
        gl_secops_security_score_distribution.observe(score)
    except Exception as exc:
        logger.debug("Failed to record security score distribution: %s", exc)


def update_at_risk_count(team: str, count: int) -> None:
    """Update at-risk user count for a team.

    Args:
        team: Team identifier.
        count: Number of at-risk users.
    """
    try:
        gl_secops_at_risk_users_total.labels(team=team).set(count)
    except Exception as exc:
        logger.debug("Failed to update at-risk count metric: %s", exc)


def update_compliance_rate(scope: str, rate: float) -> None:
    """Update compliance rate.

    Args:
        scope: Scope identifier (org, team-xxx).
        rate: Compliance rate (0.0-1.0).
    """
    try:
        gl_secops_compliance_rate.labels(scope=scope).set(rate)
    except Exception as exc:
        logger.debug("Failed to update compliance rate metric: %s", exc)


def record_certificate_issued(course: str) -> None:
    """Record a certificate issuance.

    Args:
        course: Course identifier.
    """
    try:
        gl_secops_certificates_issued_total.labels(course=course).inc()
    except Exception as exc:
        logger.debug("Failed to record certificate issued metric: %s", exc)


def update_certificates_expiring(count: int) -> None:
    """Update count of certificates expiring soon.

    Args:
        count: Number of certificates expiring within 30 days.
    """
    try:
        gl_secops_certificates_expiring_total.set(count)
    except Exception as exc:
        logger.debug("Failed to update certificates expiring metric: %s", exc)


__all__ = [
    # Training metrics
    "gl_secops_certificates_expiring_total",
    "gl_secops_certificates_issued_total",
    "gl_secops_compliance_rate",
    "gl_secops_training_attempts_total",
    "gl_secops_training_completion_rate",
    "gl_secops_training_completions_total",
    "gl_secops_training_overdue_total",
    "gl_secops_training_score_histogram",
    # Phishing metrics
    "gl_secops_phishing_campaigns_total",
    "gl_secops_phishing_click_rate",
    "gl_secops_phishing_credential_rate",
    "gl_secops_phishing_emails_sent_total",
    "gl_secops_phishing_report_rate",
    # Security score metrics
    "gl_secops_at_risk_users_total",
    "gl_secops_security_score_average",
    "gl_secops_security_score_distribution",
    # Helper functions
    "record_campaign_status",
    "record_certificate_issued",
    "record_phishing_emails_sent",
    "record_security_score",
    "record_training_attempt",
    "record_training_completion",
    "update_at_risk_count",
    "update_certificates_expiring",
    "update_compliance_rate",
    "update_overdue_count",
    "update_phishing_metrics",
    "update_security_score",
    "update_training_completion_rate",
]
