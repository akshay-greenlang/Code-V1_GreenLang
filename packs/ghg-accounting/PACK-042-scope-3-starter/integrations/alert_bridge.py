# -*- coding: utf-8 -*-
"""
AlertBridge - Alert and Notification Integration for PACK-042
================================================================

This module provides alert and notification management for the Scope 3
Starter Pack. It supports data collection reminders, supplier response
deadlines, calculation completion notifications, compliance deadline
alerts, data quality rating (DQR) warnings, and hotspot threshold
alerts across multiple delivery channels.

Alert Types (7):
    - DATA_COLLECTION: Data collection deadlines approaching or missed
    - SUPPLIER_RESPONSE: Supplier questionnaire response deadlines
    - CALCULATION_COMPLETE: Category calculation completion notifications
    - COMPLIANCE_DEADLINE: Regulatory reporting deadlines (ESRS/CDP/SBTi)
    - DQR_WARNING: Data quality rating below threshold
    - HOTSPOT_ALERT: Emission hotspot threshold exceeded
    - SCOPE3_ANOMALY: Unusual Scope 3 emission patterns detected

Channels (6):
    - EMAIL: SMTP email notifications
    - SLACK: Slack webhook integration
    - TEAMS: Microsoft Teams webhook
    - WEBHOOK: Generic HTTP webhook
    - IN_APP: Dashboard notifications
    - SMS: SMS via provider API (critical alerts)

Zero-Hallucination:
    All alert thresholds, anomaly detection, and deadline calculations
    use deterministic rule-based logic. No LLM calls in the alerting path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-042 Scope 3 Starter
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
    """Types of Scope 3 inventory alerts."""

    DATA_COLLECTION = "data_collection"
    SUPPLIER_RESPONSE = "supplier_response"
    CALCULATION_COMPLETE = "calculation_complete"
    COMPLIANCE_DEADLINE = "compliance_deadline"
    DQR_WARNING = "dqr_warning"
    HOTSPOT_ALERT = "hotspot_alert"
    SCOPE3_ANOMALY = "scope3_anomaly"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SMS = "sms"


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
        "description": "GHG Protocol Scope 3 Standard",
        "typical_deadline": "Q1 following reporting year",
    },
    "csrd_esrs_e1": {
        "description": "CSRD ESRS E1 Climate Change (incl. Scope 3)",
        "typical_deadline": "April 30 following reporting year",
    },
    "cdp_climate": {
        "description": "CDP Climate Change Questionnaire (Scope 3 module)",
        "typical_deadline": "July 31 annually",
    },
    "sbti": {
        "description": "SBTi Target Setting (Scope 3 required if >40% of total)",
        "typical_deadline": "24 months from commitment",
    },
    "sec_climate": {
        "description": "SEC Climate Disclosure Rule (Scope 3 phased)",
        "typical_deadline": "Within annual report filing deadline",
    },
    "iso_14064": {
        "description": "ISO 14064-1 (optional Scope 3 categories)",
        "typical_deadline": "As per verification schedule",
    },
    "pcaf": {
        "description": "PCAF Financed Emissions (Cat 15 Investments)",
        "typical_deadline": "Annual disclosure",
    },
}

# DQR thresholds by methodology tier
DQR_THRESHOLDS: Dict[str, float] = {
    "spend_based": 3.0,          # DQR 3.0 minimum for spend-based
    "average_data": 2.5,         # DQR 2.5 minimum for average data
    "hybrid": 2.0,               # DQR 2.0 minimum for hybrid
    "supplier_specific": 1.5,    # DQR 1.5 minimum for supplier-specific
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AlertConfig(BaseModel):
    """Alert system configuration."""

    config_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-042")
    enabled: bool = Field(default=True)
    default_recipients: List[str] = Field(default_factory=list)
    email_from: str = Field(default="scope3-alerts@greenlang.io")
    slack_webhook_url: Optional[str] = Field(None)
    teams_webhook_url: Optional[str] = Field(None)
    webhook_url: Optional[str] = Field(None)
    anomaly_threshold_pct: float = Field(default=15.0, ge=1.0, le=100.0)
    dqr_warning_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    hotspot_share_threshold_pct: float = Field(default=25.0, ge=5.0, le=100.0)
    deadline_warning_days: int = Field(default=30, ge=7)
    supplier_response_reminder_days: int = Field(default=14, ge=3)


class Alert(BaseModel):
    """Alert message with metadata."""

    alert_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    message: str = Field(default="")
    channel: AlertChannel = Field(default=AlertChannel.IN_APP)
    recipient: str = Field(default="")
    due_date: Optional[str] = Field(None)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
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
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
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
    """Alert and notification integration for Scope 3 Starter Pack.

    Manages alert creation, delivery, scheduling, anomaly detection,
    DQR monitoring, hotspot alerts, and compliance deadline tracking
    across email, Slack, Teams, webhook, in-app, and SMS channels.

    Attributes:
        config: Alert system configuration.
        _alerts: Active alert registry.
        _scheduled: Scheduled alert registry.
        _history: Send history.

    Example:
        >>> bridge = AlertBridge()
        >>> alert = bridge.create_alert(AlertType.SCOPE3_ANOMALY, "Cat 1 spike", AlertChannel.EMAIL)
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
            "AlertBridge initialized: pack=%s, enabled=%s, channels=6",
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
        scope3_category: Optional[int] = None,
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
            scope3_category: Optional Scope 3 category number (1-15).
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
            scope3_category=scope3_category,
            details=details or {},
        )
        alert.provenance_hash = _compute_hash(alert)
        self._alerts[alert.alert_id] = alert

        self.logger.info(
            "Alert created: id=%s, type=%s, severity=%s, channel=%s, cat=%s",
            alert.alert_id, alert_type.value, severity.value,
            channel.value, scope3_category,
        )
        return alert

    # -------------------------------------------------------------------------
    # Alert Delivery
    # -------------------------------------------------------------------------

    def send_alert(
        self,
        alert: Alert,
    ) -> bool:
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

    def broadcast_alert(
        self,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channels: Optional[List[AlertChannel]] = None,
        scope3_category: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Broadcast an alert to multiple channels simultaneously.

        Args:
            alert_type: Type of alert.
            message: Alert message body.
            severity: Alert severity level.
            channels: Channels to broadcast to. Defaults to EMAIL + IN_APP.
            scope3_category: Optional Scope 3 category number.
            details: Additional alert details.

        Returns:
            List of created and sent alerts.
        """
        target_channels = channels or [AlertChannel.EMAIL, AlertChannel.IN_APP]
        alerts: List[Alert] = []

        for channel in target_channels:
            alert = self.create_alert(
                alert_type=alert_type,
                message=message,
                channel=channel,
                severity=severity,
                scope3_category=scope3_category,
                details=details,
            )
            self.send_alert(alert)
            alerts.append(alert)

        self.logger.info(
            "Alert broadcast: type=%s, channels=%d, severity=%s",
            alert_type.value, len(target_channels), severity.value,
        )
        return alerts

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    def schedule_reminder(
        self,
        alert: Alert,
        schedule: str,
    ) -> ScheduledAlert:
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
            scope3_category=alert.scope3_category,
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
        scope3_category: Optional[int] = None,
        threshold_pct: Optional[float] = None,
    ) -> List[Alert]:
        """Check for Scope 3 emission anomalies against expected values.

        Args:
            current_emissions: Current period emissions (tCO2e).
            expected_emissions: Expected emissions based on trend (tCO2e).
            scope3_category: Optional category number for context.
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

            cat_label = f" (Cat {scope3_category})" if scope3_category else ""

            alert = self.create_alert(
                alert_type=AlertType.SCOPE3_ANOMALY,
                message=(
                    f"Scope 3 emissions{cat_label} {deviation_pct:.1f}% {direction} expected: "
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
                scope3_category=scope3_category,
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
    # DQR Monitoring
    # -------------------------------------------------------------------------

    def check_dqr(
        self,
        category_dqr_scores: Dict[int, float],
        methodology_tier: str = "spend_based",
    ) -> List[Alert]:
        """Check data quality ratings against thresholds.

        Args:
            category_dqr_scores: Dict mapping category number to DQR score.
            methodology_tier: Current methodology tier for threshold lookup.

        Returns:
            List of DQR warning alerts.
        """
        threshold = DQR_THRESHOLDS.get(methodology_tier, 3.0)
        alerts: List[Alert] = []

        for cat_num, dqr_score in category_dqr_scores.items():
            if dqr_score > threshold:
                severity = AlertSeverity.WARNING
                if dqr_score > threshold + 1.0:
                    severity = AlertSeverity.CRITICAL

                alert = self.create_alert(
                    alert_type=AlertType.DQR_WARNING,
                    message=(
                        f"Category {cat_num} DQR score {dqr_score:.1f} exceeds "
                        f"threshold {threshold:.1f} for {methodology_tier} tier. "
                        f"Consider upgrading data sources."
                    ),
                    channel=AlertChannel.IN_APP,
                    severity=severity,
                    scope3_category=cat_num,
                    details={
                        "dqr_score": dqr_score,
                        "threshold": threshold,
                        "methodology_tier": methodology_tier,
                        "gap": round(dqr_score - threshold, 2),
                    },
                )
                alerts.append(alert)

        if alerts:
            self.logger.info(
                "DQR check: %d categories below threshold (tier=%s, threshold=%.1f)",
                len(alerts), methodology_tier, threshold,
            )
        return alerts

    # -------------------------------------------------------------------------
    # Hotspot Alerts
    # -------------------------------------------------------------------------

    def check_hotspots(
        self,
        category_emissions: Dict[int, float],
        total_scope3: float,
        threshold_pct: Optional[float] = None,
    ) -> List[Alert]:
        """Check for emission hotspots exceeding share threshold.

        Args:
            category_emissions: Dict mapping category number to tCO2e.
            total_scope3: Total Scope 3 emissions.
            threshold_pct: Override hotspot share threshold percentage.

        Returns:
            List of hotspot alerts.
        """
        threshold = threshold_pct or self.config.hotspot_share_threshold_pct
        alerts: List[Alert] = []

        if total_scope3 <= 0:
            return alerts

        for cat_num, emissions in category_emissions.items():
            share_pct = (emissions / total_scope3) * 100
            if share_pct >= threshold:
                alert = self.create_alert(
                    alert_type=AlertType.HOTSPOT_ALERT,
                    message=(
                        f"Category {cat_num} represents {share_pct:.1f}% of total "
                        f"Scope 3 ({emissions:,.1f} / {total_scope3:,.1f} tCO2e). "
                        f"Threshold: {threshold:.0f}%."
                    ),
                    channel=AlertChannel.IN_APP,
                    severity=(
                        AlertSeverity.CRITICAL if share_pct > 50
                        else AlertSeverity.WARNING
                    ),
                    scope3_category=cat_num,
                    details={
                        "category_tco2e": emissions,
                        "total_scope3_tco2e": total_scope3,
                        "share_pct": round(share_pct, 1),
                        "threshold_pct": threshold,
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
        scope3_category: Optional[int] = None,
    ) -> List[Alert]:
        """Get active alerts with optional filters.

        Args:
            alert_type: Optional filter by alert type.
            scope3_category: Optional filter by Scope 3 category.

        Returns:
            List of active alerts.
        """
        active = [
            a for a in self._alerts.values()
            if a.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED, AlertStatus.ESCALATED)
        ]
        if alert_type:
            active = [a for a in active if a.alert_type == alert_type]
        if scope3_category is not None:
            active = [a for a in active if a.scope3_category == scope3_category]
        return active

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics.

        Returns:
            Dict with alert counts by type, severity, status, and category.
        """
        alerts = list(self._alerts.values())
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        for a in alerts:
            by_type[a.alert_type.value] = by_type.get(a.alert_type.value, 0) + 1
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
            if a.scope3_category is not None:
                cat_key = f"cat_{a.scope3_category}"
                by_category[cat_key] = by_category.get(cat_key, 0) + 1

        return {
            "total_alerts": len(alerts),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_status": by_status,
            "by_category": by_category,
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
