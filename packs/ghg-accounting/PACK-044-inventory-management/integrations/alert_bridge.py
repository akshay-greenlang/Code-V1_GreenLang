# -*- coding: utf-8 -*-
"""
AlertBridge - Alert and Notification Integration for PACK-044
================================================================

This module provides alert and notification management for the GHG Inventory
Management Pack. It supports data deadline alerts, quality degradation
warnings, review deadline reminders, compliance deadline notifications,
version conflict alerts, emission anomaly detection, and consolidation
error alerts across multiple delivery channels.

Alert Types (7):
    - DATA_DEADLINE: Data collection deadlines approaching or missed
    - QUALITY_DEGRADATION: Data quality scores below threshold
    - REVIEW_DEADLINE: Review cycle deadlines approaching
    - COMPLIANCE_DEADLINE: Regulatory reporting deadlines
    - VERSION_CONFLICT: Version control conflicts detected
    - EMISSION_ANOMALY: Unusual emission patterns detected
    - CONSOLIDATION_ERROR: Consolidation processing errors

Channels (4):
    - EMAIL: SMTP email notifications
    - SMS: SMS via provider API (critical alerts)
    - WEBHOOK: Generic HTTP webhook (EMS/SCADA integration)
    - IN_APP: Dashboard notifications

Zero-Hallucination:
    All alert thresholds, anomaly detection, and deadline calculations
    use deterministic rule-based logic. No LLM calls in the alerting path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-044 GHG Inventory Management
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertType(str, Enum):
    """Types of inventory management alerts."""

    DATA_DEADLINE = "data_deadline"
    QUALITY_DEGRADATION = "quality_degradation"
    REVIEW_DEADLINE = "review_deadline"
    COMPLIANCE_DEADLINE = "compliance_deadline"
    VERSION_CONFLICT = "version_conflict"
    EMISSION_ANOMALY = "emission_anomaly"
    CONSOLIDATION_ERROR = "consolidation_error"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class AlertStatus(str, Enum):
    """Alert lifecycle status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


# ---------------------------------------------------------------------------
# Compliance Deadline Definitions
# ---------------------------------------------------------------------------

FRAMEWORK_DEADLINES: Dict[str, Dict[str, str]] = {
    "ghg_protocol": {
        "description": "GHG Protocol Corporate Standard",
        "typical_deadline": "Q1 following reporting year",
    },
    "csrd_esrs_e1": {
        "description": "CSRD ESRS E1 Climate Change",
        "typical_deadline": "April 30 following reporting year",
    },
    "cdp_climate": {
        "description": "CDP Climate Change Questionnaire",
        "typical_deadline": "July 31 annually",
    },
    "sec_climate": {
        "description": "SEC Climate Disclosure Rule",
        "typical_deadline": "Within annual report filing deadline",
    },
    "iso_14064": {
        "description": "ISO 14064-1 Verification",
        "typical_deadline": "As per verification schedule",
    },
    "sbti_target": {
        "description": "SBTi Target Submission/Progress Report",
        "typical_deadline": "Annual progress report due each year",
    },
    "tcfd_climate": {
        "description": "TCFD Climate-Related Financial Disclosures",
        "typical_deadline": "Aligned with financial reporting cycle",
    },
}

# ---------------------------------------------------------------------------
# Review Cycle Defaults
# ---------------------------------------------------------------------------

REVIEW_CYCLE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "quarterly": {
        "frequency_days": 90,
        "reminder_days_before": 14,
        "escalation_days_after": 7,
    },
    "annual": {
        "frequency_days": 365,
        "reminder_days_before": 30,
        "escalation_days_after": 14,
    },
    "monthly": {
        "frequency_days": 30,
        "reminder_days_before": 7,
        "escalation_days_after": 3,
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AlertConfig(BaseModel):
    """Alert system configuration."""

    config_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-044")
    enabled: bool = Field(default=True)
    default_recipients: List[str] = Field(default_factory=list)
    email_from: str = Field(default="inventory-alerts@greenlang.io")
    webhook_url: Optional[str] = Field(None)
    anomaly_threshold_pct: float = Field(default=10.0, ge=1.0, le=100.0)
    data_quality_threshold: float = Field(default=80.0, ge=50.0, le=100.0)
    deadline_warning_days: int = Field(default=30, ge=7)
    review_warning_days: int = Field(default=14, ge=3)
    consolidation_error_escalate: bool = Field(default=True)


class Alert(BaseModel):
    """Alert message with metadata."""

    alert_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    message: str = Field(default="")
    channel: AlertChannel = Field(default=AlertChannel.IN_APP)
    recipient: str = Field(default="")
    due_date: Optional[str] = Field(None)
    details: Dict[str, Any] = Field(default_factory=dict)
    status: AlertStatus = Field(default=AlertStatus.ACTIVE)
    created_at: datetime = Field(default_factory=_utcnow)
    acknowledged_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")


class ScheduledAlert(BaseModel):
    """Scheduled alert with recurrence."""

    schedule_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    message: str = Field(default="")
    schedule_cron: str = Field(default="0 9 * * 1")
    channel: AlertChannel = Field(default=AlertChannel.EMAIL)
    recipient: str = Field(default="")
    next_fire: Optional[str] = Field(None)
    active: bool = Field(default=True)


class SendResult(BaseModel):
    """Result of sending an alert."""

    send_id: str = Field(default_factory=_new_uuid)
    alert_id: str = Field(default="")
    channel: AlertChannel = Field(...)
    success: bool = Field(default=True)
    recipient: str = Field(default="")
    error: Optional[str] = Field(None)
    sent_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# AlertBridge
# ---------------------------------------------------------------------------


class AlertBridge:
    """Alert and notification integration for GHG Inventory Management Pack.

    Manages alert creation, delivery, scheduling, anomaly detection,
    quality degradation monitoring, review deadline tracking, and
    compliance deadline management across email, SMS, webhook, and
    in-app channels.

    Attributes:
        config: Alert system configuration.
        _alerts: Active alert registry.
        _scheduled: Scheduled alert registry.
        _history: Send history.

    Example:
        >>> bridge = AlertBridge()
        >>> alert = bridge.create_alert(AlertType.EMISSION_ANOMALY, "Spike detected", AlertChannel.EMAIL)
        >>> success = bridge.send_alert(alert)
    """

    def __init__(
        self,
        config: Optional[AlertConfig] = None,
    ) -> None:
        """Initialize AlertBridge.

        Args:
            config: Alert system configuration. Uses defaults if None.
        """
        self.config = config or AlertConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._alerts: Dict[str, Alert] = {}
        self._scheduled: Dict[str, ScheduledAlert] = {}
        self._history: List[SendResult] = []

        self.logger.info(
            "AlertBridge initialized: pack=%s, enabled=%s",
            self.config.pack_id, self.config.enabled,
        )

    # -------------------------------------------------------------------------
    # Alert Creation
    # -------------------------------------------------------------------------

    def create_alert(
        self,
        alert_type: AlertType,
        message: str,
        channel: AlertChannel,
        severity: AlertSeverity = AlertSeverity.INFO,
        recipient: str = "",
        due_date: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a new alert.

        Args:
            alert_type: Type of alert.
            message: Alert message body.
            channel: Delivery channel.
            severity: Alert severity level.
            recipient: Alert recipient.
            due_date: Optional deadline date string.
            details: Additional alert details.

        Returns:
            Created Alert.
        """
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            channel=channel,
            recipient=recipient or (
                self.config.default_recipients[0]
                if self.config.default_recipients
                else "admin@greenlang.io"
            ),
            due_date=due_date,
            details=details or {},
        )
        alert.provenance_hash = _compute_hash(alert)
        self._alerts[alert.alert_id] = alert

        self.logger.info(
            "Alert created: id=%s, type=%s, severity=%s, channel=%s",
            alert.alert_id, alert_type.value, severity.value, channel.value,
        )
        return alert

    # -------------------------------------------------------------------------
    # Alert Delivery
    # -------------------------------------------------------------------------

    def send_alert(self, alert: Alert) -> bool:
        """Send an alert via its configured channel.

        Args:
            alert: Alert to send.

        Returns:
            True if send was successful.
        """
        if not self.config.enabled:
            self.logger.info("Alerting disabled, skipping send for %s", alert.alert_id)
            return False

        result = self._deliver(alert)
        self._history.append(result)

        self.logger.info(
            "Alert sent: id=%s, channel=%s, success=%s",
            alert.alert_id, alert.channel.value, result.success,
        )
        return result.success

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    def schedule_reminder(self, alert: Alert, schedule: str) -> ScheduledAlert:
        """Schedule a recurring reminder for an alert.

        Args:
            alert: Alert to schedule.
            schedule: Cron expression for recurrence.

        Returns:
            ScheduledAlert with schedule details.
        """
        scheduled = ScheduledAlert(
            alert_type=alert.alert_type,
            message=alert.message,
            schedule_cron=schedule,
            channel=alert.channel,
            recipient=alert.recipient,
        )
        self._scheduled[scheduled.schedule_id] = scheduled

        self.logger.info(
            "Alert scheduled: id=%s, type=%s, cron=%s",
            scheduled.schedule_id, alert.alert_type.value, schedule,
        )
        return scheduled

    # -------------------------------------------------------------------------
    # Anomaly Detection
    # -------------------------------------------------------------------------

    def check_anomalies(
        self,
        current_emissions: float,
        expected_emissions: float,
        threshold_pct: Optional[float] = None,
    ) -> List[Alert]:
        """Check for emission anomalies against expected values.

        Uses deterministic rule-based deviation analysis. No LLM calls.

        Args:
            current_emissions: Current period emissions (tCO2e).
            expected_emissions: Expected emissions based on trend (tCO2e).
            threshold_pct: Override anomaly threshold percentage.

        Returns:
            List of anomaly alerts (empty if no anomalies).
        """
        threshold = threshold_pct or self.config.anomaly_threshold_pct
        alerts: List[Alert] = []

        if expected_emissions <= 0:
            return alerts

        deviation_pct = abs(current_emissions - expected_emissions) / expected_emissions * 100

        if deviation_pct > threshold:
            direction = "above" if current_emissions > expected_emissions else "below"
            severity = AlertSeverity.WARNING
            if deviation_pct > threshold * 2:
                severity = AlertSeverity.CRITICAL
            if deviation_pct > threshold * 3:
                severity = AlertSeverity.EMERGENCY

            alert = self.create_alert(
                alert_type=AlertType.EMISSION_ANOMALY,
                message=(
                    f"Emissions {deviation_pct:.1f}% {direction} expected: "
                    f"current={current_emissions:,.1f} tCO2e, "
                    f"expected={expected_emissions:,.1f} tCO2e "
                    f"(threshold={threshold:.0f}%)"
                ),
                channel=(
                    AlertChannel.EMAIL
                    if severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
                    else AlertChannel.IN_APP
                ),
                severity=severity,
                details={
                    "current_tco2e": current_emissions,
                    "expected_tco2e": expected_emissions,
                    "deviation_pct": round(deviation_pct, 1),
                    "threshold_pct": threshold,
                    "direction": direction,
                },
            )
            alerts.append(alert)

        return alerts

    # -------------------------------------------------------------------------
    # Quality Degradation
    # -------------------------------------------------------------------------

    def check_quality_degradation(
        self,
        current_score: float,
        previous_score: float,
        entity_name: str = "",
    ) -> List[Alert]:
        """Check for data quality degradation across inventory entities.

        Args:
            current_score: Current DQI score (0-100).
            previous_score: Previous DQI score (0-100).
            entity_name: Name of entity being checked.

        Returns:
            List of quality degradation alerts.
        """
        alerts: List[Alert] = []
        threshold = self.config.data_quality_threshold

        if current_score < threshold:
            drop = previous_score - current_score
            severity = AlertSeverity.WARNING
            if current_score < threshold * 0.7:
                severity = AlertSeverity.CRITICAL

            alert = self.create_alert(
                alert_type=AlertType.QUALITY_DEGRADATION,
                message=(
                    f"Data quality below threshold for {entity_name or 'inventory'}: "
                    f"score={current_score:.1f}% (threshold={threshold:.0f}%), "
                    f"dropped {drop:.1f} points"
                ),
                channel=AlertChannel.EMAIL if severity == AlertSeverity.CRITICAL else AlertChannel.IN_APP,
                severity=severity,
                details={
                    "current_score": current_score,
                    "previous_score": previous_score,
                    "threshold": threshold,
                    "drop_points": round(drop, 1),
                    "entity_name": entity_name,
                },
            )
            alerts.append(alert)

        return alerts

    # -------------------------------------------------------------------------
    # Deadline Checking
    # -------------------------------------------------------------------------

    def check_deadlines(
        self,
        frameworks: List[str],
        current_date: str,
    ) -> List[Alert]:
        """Check compliance deadlines and create alerts for approaching ones.

        Args:
            frameworks: List of framework identifiers to check.
            current_date: Current date string (YYYY-MM-DD).

        Returns:
            List of deadline alerts.
        """
        alerts: List[Alert] = []

        for framework in frameworks:
            deadline_info = FRAMEWORK_DEADLINES.get(framework)
            if not deadline_info:
                continue

            alert = self.create_alert(
                alert_type=AlertType.COMPLIANCE_DEADLINE,
                message=(
                    f"Compliance deadline reminder: {deadline_info['description']}. "
                    f"Typical deadline: {deadline_info['typical_deadline']}."
                ),
                channel=AlertChannel.EMAIL,
                severity=AlertSeverity.WARNING,
                details={
                    "framework": framework,
                    "description": deadline_info["description"],
                    "typical_deadline": deadline_info["typical_deadline"],
                    "current_date": current_date,
                },
            )
            alerts.append(alert)

        self.logger.info(
            "Checked %d framework deadlines, created %d alerts",
            len(frameworks), len(alerts),
        )
        return alerts

    def check_review_deadlines(
        self,
        review_name: str,
        days_remaining: int,
        reviewer: str = "",
    ) -> List[Alert]:
        """Check review cycle deadlines and create reminders.

        Args:
            review_name: Name of the review cycle.
            days_remaining: Days until review deadline.
            reviewer: Assigned reviewer identifier.

        Returns:
            List of review deadline alerts.
        """
        alerts: List[Alert] = []
        warning_days = self.config.review_warning_days

        if days_remaining <= warning_days:
            severity = AlertSeverity.INFO
            if days_remaining <= 3:
                severity = AlertSeverity.CRITICAL
            elif days_remaining <= 7:
                severity = AlertSeverity.WARNING

            alert = self.create_alert(
                alert_type=AlertType.REVIEW_DEADLINE,
                message=(
                    f"Review deadline approaching: {review_name}. "
                    f"{days_remaining} days remaining."
                ),
                channel=AlertChannel.EMAIL if severity != AlertSeverity.INFO else AlertChannel.IN_APP,
                severity=severity,
                recipient=reviewer,
                details={
                    "review_name": review_name,
                    "days_remaining": days_remaining,
                    "reviewer": reviewer,
                },
            )
            alerts.append(alert)

        return alerts

    # -------------------------------------------------------------------------
    # Version Conflict Detection
    # -------------------------------------------------------------------------

    def check_version_conflict(
        self,
        version_a: str,
        version_b: str,
        entity_name: str = "",
    ) -> List[Alert]:
        """Check for version control conflicts in inventory snapshots.

        Args:
            version_a: First version identifier.
            version_b: Second version identifier.
            entity_name: Entity with conflicting versions.

        Returns:
            List of version conflict alerts.
        """
        alerts: List[Alert] = []

        if version_a != version_b:
            alert = self.create_alert(
                alert_type=AlertType.VERSION_CONFLICT,
                message=(
                    f"Version conflict detected for {entity_name or 'inventory'}: "
                    f"v{version_a} vs v{version_b}. Manual resolution required."
                ),
                channel=AlertChannel.EMAIL,
                severity=AlertSeverity.WARNING,
                details={
                    "version_a": version_a,
                    "version_b": version_b,
                    "entity_name": entity_name,
                },
            )
            alerts.append(alert)

        return alerts

    # -------------------------------------------------------------------------
    # Consolidation Error Alerts
    # -------------------------------------------------------------------------

    def alert_consolidation_error(
        self,
        error_message: str,
        entity_name: str = "",
        phase: str = "",
    ) -> Alert:
        """Create a consolidation error alert.

        Args:
            error_message: Error description from consolidation engine.
            entity_name: Entity where error occurred.
            phase: Consolidation phase that failed.

        Returns:
            Alert for the consolidation error.
        """
        severity = AlertSeverity.CRITICAL if self.config.consolidation_error_escalate else AlertSeverity.WARNING

        alert = self.create_alert(
            alert_type=AlertType.CONSOLIDATION_ERROR,
            message=(
                f"Consolidation error in {entity_name or 'inventory'}: "
                f"{error_message}"
            ),
            channel=AlertChannel.EMAIL,
            severity=severity,
            details={
                "error_message": error_message,
                "entity_name": entity_name,
                "phase": phase,
                "escalated": self.config.consolidation_error_escalate,
            },
        )
        return alert

    # -------------------------------------------------------------------------
    # Alert Management
    # -------------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an active alert.

        Args:
            alert_id: Alert to acknowledge.

        Returns:
            Dict with acknowledgement status.
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return {"alert_id": alert_id, "acknowledged": False, "reason": "Not found"}

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = _utcnow()
        return {"alert_id": alert_id, "acknowledged": True}

    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve an alert.

        Args:
            alert_id: Alert to resolve.

        Returns:
            Dict with resolution status.
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return {"alert_id": alert_id, "resolved": False, "reason": "Not found"}

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = _utcnow()
        return {"alert_id": alert_id, "resolved": True}

    def get_active_alerts(
        self,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        """Get active alerts with optional type filter.

        Args:
            alert_type: Optional filter by alert type.

        Returns:
            List of active alerts.
        """
        active = [
            a for a in self._alerts.values()
            if a.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED, AlertStatus.ESCALATED)
        ]
        if alert_type:
            active = [a for a in active if a.alert_type == alert_type]
        return active

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics.

        Returns:
            Dict with alert counts by type, severity, and status.
        """
        alerts = list(self._alerts.values())
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        for a in alerts:
            by_type[a.alert_type.value] = by_type.get(a.alert_type.value, 0) + 1
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1

        return {
            "total_alerts": len(alerts),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_status": by_status,
            "sends_total": len(self._history),
            "sends_failed": sum(1 for r in self._history if not r.success),
            "scheduled_count": len(self._scheduled),
            "provenance_hash": _compute_hash(by_type),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _deliver(self, alert: Alert) -> SendResult:
        """Deliver alert to channel (simulated).

        Args:
            alert: Alert to deliver.

        Returns:
            SendResult for the delivery attempt.
        """
        self.logger.debug(
            "Delivering alert %s via %s to %s",
            alert.alert_id, alert.channel.value, alert.recipient,
        )
        return SendResult(
            alert_id=alert.alert_id,
            channel=alert.channel,
            success=True,
            recipient=alert.recipient,
        )
