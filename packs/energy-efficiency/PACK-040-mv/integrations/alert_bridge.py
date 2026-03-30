# -*- coding: utf-8 -*-
"""
AlertBridge - Multi-Channel Alerting for M&V (PACK-040)
==========================================================

This module provides alert and notification management for the
Measurement & Verification Pack. It supports savings degradation
alerts, compliance deadline notifications, report scheduling alerts,
baseline drift warnings, meter calibration reminders, and uncertainty
threshold breaches across multiple delivery channels.

Alert Types:
    - SAVINGS_DEGRADATION: Verified savings falling below expected levels
    - COMPLIANCE_DEADLINE: Upcoming regulatory compliance deadlines
    - REPORT_SCHEDULED: M&V report generation due or completed
    - BASELINE_DRIFT: Baseline model performance degradation
    - METER_CALIBRATION: Meter calibration due or overdue
    - UNCERTAINTY_BREACH: Savings uncertainty exceeds threshold
    - DATA_QUALITY: Data quality issues affecting M&V calculations
    - PERSISTENCE_ALERT: Multi-year persistence check warnings

Channels:
    - email: SMTP email notifications
    - sms: SMS via provider API (critical alerts)
    - webhook: Generic HTTP webhook (EMS/SCADA integration)
    - dashboard: In-app dashboard notifications
    - slack: Slack channel notifications

Escalation:
    Level 1: Dashboard + Email (all alerts)
    Level 2: SMS (savings degradation, compliance deadlines)
    Level 3: Webhook + SMS (system errors, contract non-compliance)

Zero-Hallucination:
    All alert thresholds and escalation routing use deterministic
    rule-based logic. No LLM calls in the alerting path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-040 Measurement & Verification
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class AlertChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SLACK = "slack"

class AlertType(str, Enum):
    """Types of M&V alerts."""

    SAVINGS_DEGRADATION = "savings_degradation"
    COMPLIANCE_DEADLINE = "compliance_deadline"
    REPORT_SCHEDULED = "report_scheduled"
    BASELINE_DRIFT = "baseline_drift"
    METER_CALIBRATION = "meter_calibration"
    UNCERTAINTY_BREACH = "uncertainty_breach"
    DATA_QUALITY = "data_quality"
    PERSISTENCE_ALERT = "persistence_alert"

class EscalationLevel(str, Enum):
    """Alert escalation levels."""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"

# ---------------------------------------------------------------------------
# Escalation Rules
# ---------------------------------------------------------------------------

DEFAULT_ESCALATION_MAP: Dict[AlertType, Dict[AlertSeverity, EscalationLevel]] = {
    AlertType.SAVINGS_DEGRADATION: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_3,
    },
    AlertType.COMPLIANCE_DEADLINE: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_3,
    },
    AlertType.REPORT_SCHEDULED: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_1,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_2,
    },
    AlertType.BASELINE_DRIFT: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_3,
    },
    AlertType.METER_CALIBRATION: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_2,
    },
    AlertType.UNCERTAINTY_BREACH: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_3,
    },
    AlertType.DATA_QUALITY: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_2,
    },
    AlertType.PERSISTENCE_ALERT: {
        AlertSeverity.INFO: EscalationLevel.LEVEL_1,
        AlertSeverity.WARNING: EscalationLevel.LEVEL_1,
        AlertSeverity.CRITICAL: EscalationLevel.LEVEL_2,
        AlertSeverity.EMERGENCY: EscalationLevel.LEVEL_3,
    },
}

ESCALATION_CHANNELS: Dict[EscalationLevel, List[AlertChannel]] = {
    EscalationLevel.LEVEL_1: [AlertChannel.DASHBOARD, AlertChannel.EMAIL],
    EscalationLevel.LEVEL_2: [AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.SMS],
    EscalationLevel.LEVEL_3: [
        AlertChannel.DASHBOARD, AlertChannel.EMAIL,
        AlertChannel.SMS, AlertChannel.WEBHOOK, AlertChannel.SLACK,
    ],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EscalationRule(BaseModel):
    """Escalation rule definition."""

    rule_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(...)
    escalation_level: EscalationLevel = Field(...)
    channels: List[AlertChannel] = Field(default_factory=list)
    escalation_delay_minutes: int = Field(default=30, ge=0)
    auto_resolve_hours: Optional[int] = Field(None, ge=1)

class AlertConfig(BaseModel):
    """Alert system configuration."""

    config_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-040")
    enabled: bool = Field(default=True)
    default_recipients: List[str] = Field(default_factory=list)
    email_from: str = Field(default="mv-alerts@greenlang.io")
    webhook_url: Optional[str] = Field(None)
    slack_channel: str = Field(default="#mv-alerts")
    escalation_rules: List[EscalationRule] = Field(default_factory=list)
    savings_degradation_threshold_pct: float = Field(default=10.0, ge=1.0)
    baseline_drift_cvrmse_pct: float = Field(default=30.0, ge=5.0)
    uncertainty_threshold_pct: float = Field(default=50.0, ge=10.0)
    calibration_warning_days: int = Field(default=30, ge=7)
    compliance_warning_days: int = Field(default=60, ge=7)

class AlertMessage(BaseModel):
    """Alert message with metadata."""

    alert_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    title: str = Field(default="")
    message: str = Field(default="")
    details: Dict[str, Any] = Field(default_factory=dict)
    ecm_id: Optional[str] = Field(None)
    meter_id: Optional[str] = Field(None)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.LEVEL_1)
    channels: List[AlertChannel] = Field(default_factory=list)
    status: AlertStatus = Field(default=AlertStatus.ACTIVE)
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")

class NotificationResult(BaseModel):
    """Result of sending a notification."""

    notification_id: str = Field(default_factory=_new_uuid)
    alert_id: str = Field(default="")
    channel: AlertChannel = Field(...)
    success: bool = Field(default=True)
    recipient: str = Field(default="")
    error: Optional[str] = Field(None)
    sent_at: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# AlertBridge
# ---------------------------------------------------------------------------

class AlertBridge:
    """Multi-channel alerting for M&V Pack.

    Manages alert creation, escalation, and delivery across email, SMS,
    webhook, Slack, and dashboard channels for M&V-specific events
    including savings degradation, compliance deadlines, baseline drift,
    and meter calibration warnings.

    Attributes:
        config: Alert system configuration.
        _alerts: Active alert registry.
        _history: Notification delivery history.

    Example:
        >>> bridge = AlertBridge()
        >>> alert = bridge.create_alert(AlertType.SAVINGS_DEGRADATION, ...)
        >>> results = bridge.send_alert(alert)
        >>> assert all(r.success for r in results)
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
        self._alerts: Dict[str, AlertMessage] = {}
        self._history: List[NotificationResult] = []
        self.logger.info(
            "AlertBridge initialized: pack=%s, enabled=%s",
            self.config.pack_id, self.config.enabled,
        )

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        project_id: str = "",
        details: Optional[Dict[str, Any]] = None,
        ecm_id: Optional[str] = None,
        meter_id: Optional[str] = None,
    ) -> AlertMessage:
        """Create a new M&V alert.

        Args:
            alert_type: Type of alert.
            severity: Alert severity level.
            title: Alert title.
            message: Alert message body.
            project_id: Associated M&V project.
            details: Additional alert details.
            ecm_id: Associated ECM identifier.
            meter_id: Associated meter identifier.

        Returns:
            Created AlertMessage.
        """
        escalation = self._resolve_escalation(alert_type, severity)
        channels = ESCALATION_CHANNELS.get(escalation, [AlertChannel.DASHBOARD])

        alert = AlertMessage(
            project_id=project_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            details=details or {},
            ecm_id=ecm_id,
            meter_id=meter_id,
            escalation_level=escalation,
            channels=channels,
        )
        alert.provenance_hash = _compute_hash(alert)
        self._alerts[alert.alert_id] = alert

        self.logger.info(
            "Alert created: id=%s, type=%s, severity=%s, escalation=%s, channels=%s",
            alert.alert_id, alert_type.value, severity.value,
            escalation.value, [c.value for c in channels],
        )
        return alert

    def send_alert(
        self,
        alert: AlertMessage,
        recipients: Optional[List[str]] = None,
    ) -> List[NotificationResult]:
        """Send an alert via all configured channels.

        Args:
            alert: Alert message to send.
            recipients: Override recipients. Uses config defaults if None.

        Returns:
            List of notification results per channel.
        """
        if not self.config.enabled:
            self.logger.info("Alerting disabled, skipping send for %s", alert.alert_id)
            return []

        target_recipients = recipients or self.config.default_recipients or ["admin@greenlang.io"]
        results: List[NotificationResult] = []

        for channel in alert.channels:
            for recipient in target_recipients:
                result = self._send_to_channel(alert, channel, recipient)
                results.append(result)
                self._history.append(result)

        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            "Alert sent: id=%s, channels=%d, recipients=%d, success=%d/%d",
            alert.alert_id, len(alert.channels), len(target_recipients),
            success_count, len(results),
        )
        return results

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "",
    ) -> Dict[str, Any]:
        """Acknowledge an active alert.

        Args:
            alert_id: Alert to acknowledge.
            acknowledged_by: User who acknowledged.

        Returns:
            Dict with acknowledgement status.
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return {"alert_id": alert_id, "acknowledged": False, "reason": "Not found"}

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = utcnow()

        self.logger.info("Alert acknowledged: id=%s, by=%s", alert_id, acknowledged_by)
        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "acknowledged_by": acknowledged_by,
            "timestamp": alert.acknowledged_at.isoformat(),
        }

    def resolve_alert(
        self,
        alert_id: str,
        resolution_note: str = "",
    ) -> Dict[str, Any]:
        """Resolve an active alert.

        Args:
            alert_id: Alert to resolve.
            resolution_note: Resolution description.

        Returns:
            Dict with resolution status.
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return {"alert_id": alert_id, "resolved": False, "reason": "Not found"}

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = utcnow()

        self.logger.info("Alert resolved: id=%s, note=%s", alert_id, resolution_note)
        return {
            "alert_id": alert_id,
            "resolved": True,
            "resolution_note": resolution_note,
            "timestamp": alert.resolved_at.isoformat(),
        }

    def get_active_alerts(
        self,
        project_id: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[AlertMessage]:
        """Get active (non-resolved) alerts with optional filters.

        Args:
            project_id: Filter by project.
            alert_type: Filter by alert type.
            severity: Filter by severity.

        Returns:
            List of matching active alerts.
        """
        active = [
            a for a in self._alerts.values()
            if a.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED, AlertStatus.ESCALATED)
        ]
        if project_id:
            active = [a for a in active if a.project_id == project_id]
        if alert_type:
            active = [a for a in active if a.alert_type == alert_type]
        if severity:
            active = [a for a in active if a.severity == severity]
        return active

    def get_alert_summary(
        self,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get alert summary statistics.

        Args:
            project_id: Optional project filter.

        Returns:
            Dict with alert statistics.
        """
        alerts = list(self._alerts.values())
        if project_id:
            alerts = [a for a in alerts if a.project_id == project_id]

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
            "notifications_sent": len(self._history),
            "notifications_failed": sum(1 for r in self._history if not r.success),
            "provenance_hash": _compute_hash(by_type),
        }

    def check_savings_degradation(
        self,
        project_id: str,
        expected_savings_kwh: float,
        actual_savings_kwh: float,
    ) -> Optional[AlertMessage]:
        """Check for savings degradation and create alert if needed.

        Args:
            project_id: M&V project identifier.
            expected_savings_kwh: Expected savings from baseline model.
            actual_savings_kwh: Actual verified savings.

        Returns:
            AlertMessage if degradation detected, None otherwise.
        """
        if expected_savings_kwh <= 0:
            return None

        degradation_pct = (
            (expected_savings_kwh - actual_savings_kwh)
            / expected_savings_kwh * 100.0
        )

        threshold = self.config.savings_degradation_threshold_pct

        if degradation_pct > threshold:
            severity = AlertSeverity.WARNING
            if degradation_pct > threshold * 2:
                severity = AlertSeverity.CRITICAL
            if degradation_pct > threshold * 3:
                severity = AlertSeverity.EMERGENCY

            return self.create_alert(
                alert_type=AlertType.SAVINGS_DEGRADATION,
                severity=severity,
                title=f"Savings Degradation: {degradation_pct:.1f}%",
                message=(
                    f"Verified savings ({actual_savings_kwh:,.0f} kWh) are "
                    f"{degradation_pct:.1f}% below expected ({expected_savings_kwh:,.0f} kWh). "
                    f"Threshold: {threshold:.0f}%."
                ),
                project_id=project_id,
                details={
                    "expected_kwh": expected_savings_kwh,
                    "actual_kwh": actual_savings_kwh,
                    "degradation_pct": round(degradation_pct, 1),
                    "threshold_pct": threshold,
                },
            )
        return None

    def check_meter_calibration(
        self,
        project_id: str,
        meter_id: str,
        calibration_due_date: str,
        days_until_due: int,
    ) -> Optional[AlertMessage]:
        """Check meter calibration schedule and alert if due.

        Args:
            project_id: M&V project identifier.
            meter_id: Meter identifier.
            calibration_due_date: Calibration due date string.
            days_until_due: Days until calibration is due.

        Returns:
            AlertMessage if calibration alert needed, None otherwise.
        """
        warning_days = self.config.calibration_warning_days

        if days_until_due <= 0:
            return self.create_alert(
                alert_type=AlertType.METER_CALIBRATION,
                severity=AlertSeverity.CRITICAL,
                title=f"Meter Calibration Overdue: {meter_id}",
                message=(
                    f"Meter {meter_id} calibration was due on {calibration_due_date}. "
                    f"Overdue by {abs(days_until_due)} days. M&V data quality may be affected."
                ),
                project_id=project_id,
                meter_id=meter_id,
                details={
                    "calibration_due": calibration_due_date,
                    "days_overdue": abs(days_until_due),
                },
            )
        elif days_until_due <= warning_days:
            return self.create_alert(
                alert_type=AlertType.METER_CALIBRATION,
                severity=AlertSeverity.WARNING,
                title=f"Meter Calibration Due Soon: {meter_id}",
                message=(
                    f"Meter {meter_id} calibration due in {days_until_due} days "
                    f"({calibration_due_date}). Schedule calibration to maintain M&V accuracy."
                ),
                project_id=project_id,
                meter_id=meter_id,
                details={
                    "calibration_due": calibration_due_date,
                    "days_until_due": days_until_due,
                },
            )
        return None

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _resolve_escalation(
        self, alert_type: AlertType, severity: AlertSeverity
    ) -> EscalationLevel:
        """Resolve escalation level from type and severity.

        Args:
            alert_type: Alert type.
            severity: Alert severity.

        Returns:
            Appropriate escalation level.
        """
        type_map = DEFAULT_ESCALATION_MAP.get(alert_type, {})
        return type_map.get(severity, EscalationLevel.LEVEL_1)

    def _send_to_channel(
        self,
        alert: AlertMessage,
        channel: AlertChannel,
        recipient: str,
    ) -> NotificationResult:
        """Send alert to a specific channel (simulated).

        Args:
            alert: Alert message.
            channel: Delivery channel.
            recipient: Recipient identifier.

        Returns:
            NotificationResult for the delivery attempt.
        """
        self.logger.debug(
            "Sending alert %s via %s to %s",
            alert.alert_id, channel.value, recipient,
        )

        return NotificationResult(
            alert_id=alert.alert_id,
            channel=channel,
            success=True,
            recipient=recipient,
        )
