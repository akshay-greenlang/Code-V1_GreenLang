# -*- coding: utf-8 -*-
"""
Alert Lifecycle Manager - OBS-004: Unified Alerting Service

Manages the full lifecycle of alerts from firing through acknowledgement,
investigation, resolution, and suppression. Enforces a strict state-machine
with validated transitions and records timestamps at each stage.

Example:
    >>> from greenlang.infrastructure.alerting_service.lifecycle import AlertLifecycle
    >>> from greenlang.infrastructure.alerting_service.config import AlertingConfig
    >>> lc = AlertLifecycle(AlertingConfig())
    >>> alert = lc.fire({"source": "prometheus", "name": "HighCPU", ...})
    >>> alert = lc.acknowledge(alert.alert_id, "oncall-user")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.config import AlertingConfig
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Machine Transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: Dict[AlertStatus, List[AlertStatus]] = {
    AlertStatus.FIRING: [
        AlertStatus.ACKNOWLEDGED,
        AlertStatus.SUPPRESSED,
        AlertStatus.RESOLVED,
    ],
    AlertStatus.ACKNOWLEDGED: [
        AlertStatus.INVESTIGATING,
        AlertStatus.RESOLVED,
    ],
    AlertStatus.INVESTIGATING: [
        AlertStatus.RESOLVED,
    ],
    AlertStatus.RESOLVED: [
        AlertStatus.FIRING,  # re-fire
    ],
    AlertStatus.SUPPRESSED: [
        AlertStatus.FIRING,
        AlertStatus.RESOLVED,
    ],
}


# ---------------------------------------------------------------------------
# AlertLifecycle
# ---------------------------------------------------------------------------


class AlertLifecycle:
    """State-machine manager for the full alert lifecycle.

    Maintains an in-memory store of alerts keyed by ``alert_id`` and
    enforces valid transitions between lifecycle states.

    Attributes:
        config: AlertingConfig instance.
    """

    def __init__(self, config: AlertingConfig) -> None:
        """Initialize the lifecycle manager.

        Args:
            config: AlertingConfig for timeout thresholds.
        """
        self.config = config
        self._alerts: Dict[str, Alert] = {}
        self._fingerprint_index: Dict[str, str] = {}  # fingerprint -> alert_id
        self._lock = threading.RLock()
        logger.info("AlertLifecycle initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fire(self, alert_data: Dict[str, Any]) -> Alert:
        """Create or re-fire an alert.

        If an active alert with the same fingerprint already exists, the
        existing alert's ``notification_count`` is incremented.  Otherwise
        a new Alert is created in FIRING state.

        Args:
            alert_data: Dictionary matching Alert constructor kwargs.

        Returns:
            The new or existing Alert instance.
        """
        severity = alert_data.get("severity", "info")
        if isinstance(severity, str):
            severity = AlertSeverity(severity)
        alert_data["severity"] = severity

        fingerprint = Alert.generate_fingerprint(
            alert_data.get("source", "unknown"),
            alert_data.get("name", "unknown"),
            alert_data.get("labels", {}),
        )

        with self._lock:
            # Check for existing active alert by fingerprint
            existing_id = self._fingerprint_index.get(fingerprint)
            if existing_id and existing_id in self._alerts:
                existing = self._alerts[existing_id]
                if existing.status != AlertStatus.RESOLVED:
                    existing.notification_count += 1
                    logger.debug(
                        "Alert re-fired (dedup): alert_id=%s, count=%d",
                        existing.alert_id, existing.notification_count,
                    )
                    return existing

                # Resolved alert re-firing
                self._validate_transition(
                    existing.status, AlertStatus.FIRING,
                )
                existing.status = AlertStatus.FIRING
                existing.fired_at = datetime.now(timezone.utc)
                existing.resolved_at = None
                existing.resolved_by = ""
                existing.acknowledged_at = None
                existing.acknowledged_by = ""
                existing.escalation_level = 0
                existing.notification_count += 1
                logger.info(
                    "Alert re-fired from RESOLVED: alert_id=%s",
                    existing.alert_id,
                )
                return existing

            # New alert
            alert = Alert(**alert_data)
            alert.fingerprint = fingerprint
            alert.status = AlertStatus.FIRING
            self._alerts[alert.alert_id] = alert
            self._fingerprint_index[fingerprint] = alert.alert_id
            logger.info(
                "Alert fired: alert_id=%s, name=%s, severity=%s",
                alert.alert_id, alert.name, alert.severity.value,
            )
            return alert

    def acknowledge(self, alert_id: str, user: str) -> Alert:
        """Transition an alert to ACKNOWLEDGED.

        Args:
            alert_id: Alert identifier.
            user: User performing the acknowledgement.

        Returns:
            The updated Alert.

        Raises:
            KeyError: If alert_id not found.
            ValueError: If the transition is invalid.
        """
        with self._lock:
            alert = self._get_alert(alert_id)
            self._validate_transition(alert.status, AlertStatus.ACKNOWLEDGED)
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.acknowledged_by = user
            logger.info(
                "Alert acknowledged: alert_id=%s, by=%s", alert_id, user,
            )
            return alert

    def investigate(self, alert_id: str) -> Alert:
        """Transition an alert to INVESTIGATING.

        Args:
            alert_id: Alert identifier.

        Returns:
            The updated Alert.

        Raises:
            KeyError: If alert_id not found.
            ValueError: If the transition is invalid.
        """
        with self._lock:
            alert = self._get_alert(alert_id)
            self._validate_transition(alert.status, AlertStatus.INVESTIGATING)
            alert.status = AlertStatus.INVESTIGATING
            logger.info("Alert under investigation: alert_id=%s", alert_id)
            return alert

    def resolve(self, alert_id: str, user: str) -> Alert:
        """Transition an alert to RESOLVED.

        Args:
            alert_id: Alert identifier.
            user: User resolving the alert.

        Returns:
            The updated Alert.

        Raises:
            KeyError: If alert_id not found.
            ValueError: If the transition is invalid.
        """
        with self._lock:
            alert = self._get_alert(alert_id)
            self._validate_transition(alert.status, AlertStatus.RESOLVED)
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            alert.resolved_by = user
            logger.info(
                "Alert resolved: alert_id=%s, by=%s", alert_id, user,
            )
            return alert

    def suppress(
        self,
        alert_id: str,
        duration_minutes: int,
        reason: str = "",
    ) -> Alert:
        """Suppress an alert for a given duration.

        The alert transitions to SUPPRESSED.  A future caller should
        restore it to FIRING after ``duration_minutes`` if it is not
        explicitly resolved.

        Args:
            alert_id: Alert identifier.
            duration_minutes: How long to suppress.
            reason: Free-text reason for suppression.

        Returns:
            The updated Alert.

        Raises:
            KeyError: If alert_id not found.
            ValueError: If the transition is invalid.
        """
        with self._lock:
            alert = self._get_alert(alert_id)
            self._validate_transition(alert.status, AlertStatus.SUPPRESSED)
            alert.status = AlertStatus.SUPPRESSED
            alert.annotations["suppress_reason"] = reason
            alert.annotations["suppress_until"] = (
                datetime.now(timezone.utc)
                + timedelta(minutes=duration_minutes)
            ).isoformat()
            logger.info(
                "Alert suppressed: alert_id=%s, duration=%dm, reason=%s",
                alert_id, duration_minutes, reason,
            )
            return alert

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrieve an alert by its ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            Alert or None if not found.
        """
        return self._alerts.get(alert_id)

    def list_alerts(
        self,
        *,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        team: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Alert]:
        """List alerts with optional filters.

        Args:
            status: Filter by status.
            severity: Filter by severity.
            team: Filter by team name.
            service: Filter by service name.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Filtered list of alerts.
        """
        results: List[Alert] = []
        for alert in self._alerts.values():
            if status is not None and alert.status != status:
                continue
            if severity is not None and alert.severity != severity:
                continue
            if team and alert.team != team:
                continue
            if service and alert.service != service:
                continue
            results.append(alert)

        # Sort newest first
        results.sort(
            key=lambda a: a.fired_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return results[offset: offset + limit]

    def get_active_counts(self) -> Dict[str, Dict[str, int]]:
        """Return counts of active alerts grouped by severity and status.

        Returns:
            Nested dict ``{severity: {status: count}}``.
        """
        counts: Dict[str, Dict[str, int]] = {}
        for alert in self._alerts.values():
            sev = alert.severity.value
            st = alert.status.value
            counts.setdefault(sev, {})
            counts[sev][st] = counts[sev].get(st, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_alert(self, alert_id: str) -> Alert:
        """Fetch alert or raise KeyError."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert not found: {alert_id}")
        return alert

    @staticmethod
    def _validate_transition(
        current: AlertStatus,
        target: AlertStatus,
    ) -> None:
        """Validate a state transition.

        Args:
            current: Current alert status.
            target: Desired target status.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = _VALID_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"Invalid transition: {current.value} -> {target.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
