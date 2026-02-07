# -*- coding: utf-8 -*-
"""
Alerting Service Metrics - OBS-004: Unified Alerting Service

Prometheus metrics for the unified alerting service covering notification
delivery, MTTA/MTTR, escalation activity, deduplication effectiveness,
alert fatigue scoring, and on-call provider lookups.

All metrics use the ``gl_alert_`` prefix to align with the GreenLang
monitoring namespace. When ``prometheus_client`` is not installed the
helper functions degrade to no-ops.

Example:
    >>> from greenlang.infrastructure.alerting_service.metrics import (
    ...     record_notification,
    ... )
    >>> record_notification("slack", "warning", "sent", 0.123)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
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
        "prometheus_client not installed; alerting metrics operate in no-op mode"
    )


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    gl_alert_notifications_total = Counter(
        "gl_alert_notifications_total",
        "Total notification delivery attempts",
        ["channel", "severity", "status"],
    )

    gl_alert_notification_duration_seconds = Histogram(
        "gl_alert_notification_duration_seconds",
        "Notification delivery latency in seconds",
        ["channel"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    gl_alert_notification_failures_total = Counter(
        "gl_alert_notification_failures_total",
        "Total notification delivery failures",
        ["channel", "error_type"],
    )

    gl_alert_mtta_seconds = Histogram(
        "gl_alert_mtta_seconds",
        "Mean Time To Acknowledge in seconds",
        ["team", "severity"],
        buckets=(60, 120, 300, 600, 900, 1800, 3600, 7200),
    )

    gl_alert_mttr_seconds = Histogram(
        "gl_alert_mttr_seconds",
        "Mean Time To Resolve in seconds",
        ["team", "severity"],
        buckets=(300, 600, 1800, 3600, 7200, 14400, 28800, 86400),
    )

    gl_alert_active_total = Gauge(
        "gl_alert_active_total",
        "Currently active alerts",
        ["severity", "status"],
    )

    gl_alert_escalations_total = Counter(
        "gl_alert_escalations_total",
        "Total alert escalations",
        ["level", "policy"],
    )

    gl_alert_dedup_total = Counter(
        "gl_alert_dedup_total",
        "Total deduplicated (suppressed duplicate) alerts",
    )

    gl_alert_fatigue_score = Gauge(
        "gl_alert_fatigue_score",
        "Alert fatigue score (alerts per hour) per team",
        ["team"],
    )

    gl_alert_oncall_lookups_total = Counter(
        "gl_alert_oncall_lookups_total",
        "Total on-call schedule lookups",
        ["provider", "status"],
    )
else:
    gl_alert_notifications_total = None  # type: ignore[assignment]
    gl_alert_notification_duration_seconds = None  # type: ignore[assignment]
    gl_alert_notification_failures_total = None  # type: ignore[assignment]
    gl_alert_mtta_seconds = None  # type: ignore[assignment]
    gl_alert_mttr_seconds = None  # type: ignore[assignment]
    gl_alert_active_total = None  # type: ignore[assignment]
    gl_alert_escalations_total = None  # type: ignore[assignment]
    gl_alert_dedup_total = None  # type: ignore[assignment]
    gl_alert_fatigue_score = None  # type: ignore[assignment]
    gl_alert_oncall_lookups_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_notification(
    channel: str,
    severity: str,
    status: str,
    duration_s: float = 0.0,
) -> None:
    """Record a notification delivery attempt.

    Args:
        channel: Channel name (e.g. ``slack``).
        severity: Alert severity (e.g. ``critical``).
        status: Delivery status (e.g. ``sent``, ``failed``).
        duration_s: Delivery latency in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_notifications_total.labels(
        channel=channel, severity=severity, status=status,
    ).inc()
    gl_alert_notification_duration_seconds.labels(channel=channel).observe(
        duration_s,
    )


def record_notification_failure(channel: str, error_type: str) -> None:
    """Record a notification delivery failure.

    Args:
        channel: Channel name.
        error_type: Error classification (e.g. ``timeout``, ``auth``).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_notification_failures_total.labels(
        channel=channel, error_type=error_type,
    ).inc()


def record_mtta(team: str, severity: str, seconds: float) -> None:
    """Record Mean Time To Acknowledge for an alert.

    Args:
        team: Team that acknowledged.
        severity: Alert severity.
        seconds: Time from firing to acknowledgement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_mtta_seconds.labels(team=team, severity=severity).observe(seconds)


def record_mttr(team: str, severity: str, seconds: float) -> None:
    """Record Mean Time To Resolve for an alert.

    Args:
        team: Team that resolved.
        severity: Alert severity.
        seconds: Time from firing to resolution.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_mttr_seconds.labels(team=team, severity=severity).observe(seconds)


def record_escalation(level: str, policy: str) -> None:
    """Record an escalation event.

    Args:
        level: Escalation level (e.g. ``1``, ``2``).
        policy: Escalation policy name.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_escalations_total.labels(level=level, policy=policy).inc()


def record_dedup() -> None:
    """Record a deduplicated (suppressed duplicate) alert."""
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_dedup_total.inc()


def update_active_alerts(severity: str, status: str, value: int) -> None:
    """Set the gauge for currently active alerts.

    Args:
        severity: Alert severity.
        status: Alert status.
        value: Current count.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_active_total.labels(severity=severity, status=status).set(value)


def update_fatigue_score(team: str, score: float) -> None:
    """Update alert fatigue score for a team.

    Args:
        team: Team identifier.
        score: Alerts per hour.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_fatigue_score.labels(team=team).set(score)


def record_oncall_lookup(provider: str, status: str) -> None:
    """Record an on-call schedule lookup.

    Args:
        provider: On-call provider (``pagerduty`` or ``opsgenie``).
        status: Lookup outcome (``success`` or ``error``).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gl_alert_oncall_lookups_total.labels(
        provider=provider, status=status,
    ).inc()
