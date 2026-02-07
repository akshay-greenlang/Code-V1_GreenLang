# -*- coding: utf-8 -*-
"""
Incident Response Metrics - SEC-010

Prometheus metrics for incident response observability. Tracks incident
counts, MTTD/MTTR/MTTS, playbook executions, alerts, and escalations.

Metrics are lazily initialized on first use so that the module can be imported
even when prometheus_client is not installed (metrics become no-ops).

Registered metrics:
    - gl_secops_incidents_total (Counter): Total incidents by severity/type/source.
    - gl_secops_incident_mttd_seconds (Histogram): Mean Time to Detect.
    - gl_secops_incident_mttr_seconds (Histogram): Mean Time to Respond.
    - gl_secops_incident_mtts_seconds (Histogram): Mean Time to Start.
    - gl_secops_playbook_executions_total (Counter): Playbook executions.
    - gl_secops_alerts_total (Counter): Alerts by source/severity.
    - gl_secops_escalations_total (Counter): Escalations by level/reason.
    - gl_secops_active_incidents (Gauge): Currently active incidents.

Example:
    >>> from greenlang.infrastructure.incident_response.metrics import (
    ...     IncidentResponseMetrics,
    ...     get_metrics,
    ... )
    >>> metrics = get_metrics()
    >>> metrics.record_incident_created("P0", "data_breach", "guardduty")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: Lazy Prometheus Metric Handles
# ---------------------------------------------------------------------------


class _PrometheusHandles:
    """Lazy-initialized Prometheus metric objects.

    Metrics are created on first call to ensure_initialized().
    If prometheus_client is not installed, all handles remain None
    and recording methods become safe no-ops.
    """

    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()
    _available: bool = False

    # Counters
    incidents_total: Any = None
    playbook_executions_total: Any = None
    alerts_total: Any = None
    escalations_total: Any = None
    notifications_total: Any = None

    # Gauges
    active_incidents: Any = None
    pending_escalations: Any = None

    # Histograms
    incident_mttd_seconds: Any = None
    incident_mttr_seconds: Any = None
    incident_mtts_seconds: Any = None
    playbook_duration_seconds: Any = None
    alert_correlation_duration_seconds: Any = None

    @classmethod
    def ensure_initialized(cls) -> bool:
        """Create Prometheus metrics if the library is available.

        Thread-safe via a class-level lock.

        Returns:
            True if prometheus_client is available and metrics are
            registered, False otherwise.
        """
        if cls._initialized:
            return cls._available

        with cls._lock:
            if cls._initialized:
                return cls._available

            cls._initialized = True

            try:
                from prometheus_client import Counter, Gauge, Histogram
            except ImportError:
                logger.info(
                    "prometheus_client not installed; incident response metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_secops"

            # -- Counters --------------------------------------------------

            cls.incidents_total = Counter(
                f"{prefix}_incidents_total",
                "Total security incidents created",
                ["severity", "type", "source"],
            )

            cls.playbook_executions_total = Counter(
                f"{prefix}_playbook_executions_total",
                "Total playbook executions",
                ["playbook", "status"],
            )

            cls.alerts_total = Counter(
                f"{prefix}_alerts_total",
                "Total alerts received",
                ["source", "severity"],
            )

            cls.escalations_total = Counter(
                f"{prefix}_escalations_total",
                "Total escalations triggered",
                ["level", "reason"],
            )

            cls.notifications_total = Counter(
                f"{prefix}_notifications_total",
                "Total notifications sent",
                ["channel", "severity", "status"],
            )

            # -- Gauges ----------------------------------------------------

            cls.active_incidents = Gauge(
                f"{prefix}_active_incidents",
                "Currently active incidents",
                ["severity"],
            )

            cls.pending_escalations = Gauge(
                f"{prefix}_pending_escalations",
                "Incidents pending escalation",
                ["severity"],
            )

            # -- Histograms ------------------------------------------------

            cls.incident_mttd_seconds = Histogram(
                f"{prefix}_incident_mttd_seconds",
                "Mean Time to Detect (seconds)",
                ["severity"],
                buckets=(
                    30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400,
                ),
            )

            cls.incident_mttr_seconds = Histogram(
                f"{prefix}_incident_mttr_seconds",
                "Mean Time to Respond/Resolve (seconds)",
                ["severity"],
                buckets=(
                    60, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400,
                ),
            )

            cls.incident_mtts_seconds = Histogram(
                f"{prefix}_incident_mtts_seconds",
                "Mean Time to Start (seconds)",
                ["severity"],
                buckets=(
                    30, 60, 120, 300, 600, 900, 1800, 3600,
                ),
            )

            cls.playbook_duration_seconds = Histogram(
                f"{prefix}_playbook_duration_seconds",
                "Playbook execution duration (seconds)",
                ["playbook", "status"],
                buckets=(
                    10, 30, 60, 120, 300, 600, 900, 1800,
                ),
            )

            cls.alert_correlation_duration_seconds = Histogram(
                f"{prefix}_alert_correlation_duration_seconds",
                "Alert correlation processing time (seconds)",
                buckets=(
                    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                ),
            )

            cls._available = True
            logger.info("Incident response Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# IncidentResponseMetrics
# ---------------------------------------------------------------------------


class IncidentResponseMetrics:
    """Manages Prometheus metrics for incident response.

    All recording methods are safe no-ops when prometheus_client is
    not installed. Thread-safe.

    Example:
        >>> m = IncidentResponseMetrics()
        >>> m.record_incident_created("P0", "data_breach", "guardduty")
        >>> m.record_mttd("P0", 120.5)
    """

    def __init__(self, prefix: str = "gl_secops") -> None:
        """Initialize incident response metrics.

        Args:
            prefix: Metric name prefix (documentation only).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Incidents
    # ------------------------------------------------------------------

    def record_incident_created(
        self,
        severity: str,
        incident_type: str,
        source: str,
    ) -> None:
        """Record a new incident.

        Args:
            severity: Incident severity (P0, P1, P2, P3).
            incident_type: Type of incident.
            source: Alert source.
        """
        if not self._available:
            return

        _PrometheusHandles.incidents_total.labels(
            severity=severity,
            type=incident_type,
            source=source,
        ).inc()

    def set_active_incidents(
        self,
        severity: str,
        count: int,
    ) -> None:
        """Set the active incident count for a severity.

        Args:
            severity: Incident severity.
            count: Current count of active incidents.
        """
        if not self._available:
            return

        _PrometheusHandles.active_incidents.labels(
            severity=severity
        ).set(count)

    def record_mttd(
        self,
        severity: str,
        seconds: float,
    ) -> None:
        """Record Mean Time to Detect.

        Args:
            severity: Incident severity.
            seconds: MTTD in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.incident_mttd_seconds.labels(
            severity=severity
        ).observe(seconds)

    def record_mttr(
        self,
        severity: str,
        seconds: float,
    ) -> None:
        """Record Mean Time to Respond/Resolve.

        Args:
            severity: Incident severity.
            seconds: MTTR in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.incident_mttr_seconds.labels(
            severity=severity
        ).observe(seconds)

    def record_mtts(
        self,
        severity: str,
        seconds: float,
    ) -> None:
        """Record Mean Time to Start (remediation).

        Args:
            severity: Incident severity.
            seconds: MTTS in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.incident_mtts_seconds.labels(
            severity=severity
        ).observe(seconds)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def record_alert_received(
        self,
        source: str,
        severity: str,
    ) -> None:
        """Record an alert received.

        Args:
            source: Alert source (prometheus, loki, guardduty, etc.).
            severity: Alert severity.
        """
        if not self._available:
            return

        _PrometheusHandles.alerts_total.labels(
            source=source,
            severity=severity,
        ).inc()

    def record_alert_correlation_time(
        self,
        seconds: float,
    ) -> None:
        """Record alert correlation processing time.

        Args:
            seconds: Processing time in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.alert_correlation_duration_seconds.observe(seconds)

    # ------------------------------------------------------------------
    # Escalations
    # ------------------------------------------------------------------

    def record_escalation(
        self,
        level: str,
        reason: str,
    ) -> None:
        """Record an escalation.

        Args:
            level: Escalation level (1, 2, 3, etc.).
            reason: Reason for escalation.
        """
        if not self._available:
            return

        _PrometheusHandles.escalations_total.labels(
            level=str(level),
            reason=reason,
        ).inc()

    def set_pending_escalations(
        self,
        severity: str,
        count: int,
    ) -> None:
        """Set count of pending escalations.

        Args:
            severity: Incident severity.
            count: Number of pending escalations.
        """
        if not self._available:
            return

        _PrometheusHandles.pending_escalations.labels(
            severity=severity
        ).set(count)

    # ------------------------------------------------------------------
    # Playbooks
    # ------------------------------------------------------------------

    def record_playbook_execution(
        self,
        playbook: str,
        status: str,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a playbook execution.

        Args:
            playbook: Playbook ID.
            status: Execution status (completed, failed, cancelled).
            duration_seconds: Execution duration.
        """
        if not self._available:
            return

        _PrometheusHandles.playbook_executions_total.labels(
            playbook=playbook,
            status=status,
        ).inc()

        if duration_seconds is not None:
            _PrometheusHandles.playbook_duration_seconds.labels(
                playbook=playbook,
                status=status,
            ).observe(duration_seconds)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def record_notification_sent(
        self,
        channel: str,
        severity: str,
        success: bool,
    ) -> None:
        """Record a notification sent.

        Args:
            channel: Notification channel (pagerduty, slack, email, sms).
            severity: Incident severity.
            success: Whether notification was successful.
        """
        if not self._available:
            return

        _PrometheusHandles.notifications_total.labels(
            channel=channel,
            severity=severity,
            status="success" if success else "failure",
        ).inc()


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_metrics: Optional[IncidentResponseMetrics] = None


def get_metrics() -> IncidentResponseMetrics:
    """Get or create the global incident response metrics instance.

    Returns:
        The global IncidentResponseMetrics instance.
    """
    global _global_metrics

    if _global_metrics is None:
        _global_metrics = IncidentResponseMetrics()

    return _global_metrics


__all__ = [
    "IncidentResponseMetrics",
    "get_metrics",
]
