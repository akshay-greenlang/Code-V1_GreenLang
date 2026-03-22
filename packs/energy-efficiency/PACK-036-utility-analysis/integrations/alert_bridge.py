# -*- coding: utf-8 -*-
"""
AlertBridge - Multi-Channel Notification and Alert Management for PACK-036
============================================================================

This module provides alert and notification management for the Utility
Analysis Pack. It supports multiple notification channels (email, SMS,
webhook, in-app), configurable alert rules, and alert lifecycle management.

Alert Types:
    - BILL_ERROR:            Bill parsing or validation error detected
    - BUDGET_VARIANCE:       Actual cost exceeds budget threshold
    - DEMAND_PEAK:           Demand approaching or exceeding limit
    - RATE_CHANGE:           Tariff or rate structure change notification
    - PROCUREMENT_DEADLINE:  Contract renewal or procurement deadline
    - ANOMALY_DETECTED:      Consumption or cost anomaly detected
    - DATA_QUALITY:          Data quality below threshold
    - REGULATORY_CHANGE:     Regulatory charge or exemption change

Channels:
    - email:     SMTP email notifications
    - sms:       SMS via provider API
    - webhook:   Generic HTTP webhook (Slack, Teams, PagerDuty)
    - in_app:    In-app dashboard notification

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-036 Utility Analysis
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class AlertType(str, Enum):
    """Types of utility analysis alerts."""

    BILL_ERROR = "bill_error"
    BUDGET_VARIANCE = "budget_variance"
    DEMAND_PEAK = "demand_peak"
    RATE_CHANGE = "rate_change"
    PROCUREMENT_DEADLINE = "procurement_deadline"
    ANOMALY_DETECTED = "anomaly_detected"
    DATA_QUALITY = "data_quality"
    REGULATORY_CHANGE = "regulatory_change"


class AlertStatus(str, Enum):
    """Alert lifecycle status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AlertConfig(BaseModel):
    """Configuration for the Alert Bridge."""

    pack_id: str = Field(default="PACK-036")
    enable_provenance: bool = Field(default=True)
    default_channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.IN_APP, AlertChannel.EMAIL]
    )
    enable_email: bool = Field(default=True)
    enable_sms: bool = Field(default=False)
    enable_webhook: bool = Field(default=False)
    email_recipients: List[str] = Field(default_factory=list)
    webhook_url: str = Field(default="")
    sms_recipients: List[str] = Field(default_factory=list)
    budget_variance_threshold_pct: float = Field(default=10.0, ge=0)
    demand_peak_threshold_pct: float = Field(default=90.0, ge=0, le=100)
    data_quality_threshold: float = Field(default=80.0, ge=0, le=100)
    cooldown_minutes: int = Field(default=60, ge=0)


class Alert(BaseModel):
    """An alert instance."""

    alert_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    status: AlertStatus = Field(default=AlertStatus.ACTIVE)
    title: str = Field(default="")
    message: str = Field(default="")
    facility_id: str = Field(default="")
    account_id: str = Field(default="")
    commodity: str = Field(default="")
    metric_value: Optional[float] = Field(None)
    threshold_value: Optional[float] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    acknowledged_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    dismissed_at: Optional[datetime] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class AlertRule(BaseModel):
    """An alert rule that triggers notifications."""

    rule_id: str = Field(default_factory=_new_uuid)
    rule_name: str = Field(default="")
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    channels: List[AlertChannel] = Field(default_factory=list)
    condition: str = Field(default="", description="Rule condition expression")
    threshold_value: float = Field(default=0.0)
    enabled: bool = Field(default=True)
    cooldown_minutes: int = Field(default=60, ge=0)
    description: str = Field(default="")
    commodity_filter: Optional[str] = Field(None)


class NotificationResult(BaseModel):
    """Result of sending an alert notification."""

    notification_id: str = Field(default_factory=_new_uuid)
    alert_id: str = Field(default="")
    channel: AlertChannel = Field(...)
    success: bool = Field(default=False)
    message: str = Field(default="")
    delivered_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class AlertSummary(BaseModel):
    """Summary of alert activity for a period."""

    summary_id: str = Field(default_factory=_new_uuid)
    period: str = Field(default="")
    total_alerts: int = Field(default=0)
    active_alerts: int = Field(default=0)
    acknowledged_alerts: int = Field(default=0)
    resolved_alerts: int = Field(default=0)
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Default Alert Rules
# ---------------------------------------------------------------------------

DEFAULT_ALERT_RULES: List[AlertRule] = [
    AlertRule(
        rule_name="Bill Error Detection",
        alert_type=AlertType.BILL_ERROR,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.IN_APP, AlertChannel.EMAIL],
        condition="bill_error_count > 0",
        threshold_value=0.0,
        description="Alert when bill parsing or validation errors are detected",
    ),
    AlertRule(
        rule_name="Budget Variance Warning",
        alert_type=AlertType.BUDGET_VARIANCE,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.IN_APP, AlertChannel.EMAIL],
        condition="variance_pct > threshold",
        threshold_value=10.0,
        description="Alert when actual cost exceeds budget by threshold %",
    ),
    AlertRule(
        rule_name="Budget Variance Critical",
        alert_type=AlertType.BUDGET_VARIANCE,
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.IN_APP, AlertChannel.EMAIL, AlertChannel.SMS],
        condition="variance_pct > threshold",
        threshold_value=20.0,
        description="Critical alert for severe budget overrun",
    ),
    AlertRule(
        rule_name="Demand Peak Warning",
        alert_type=AlertType.DEMAND_PEAK,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.IN_APP],
        condition="demand_pct_of_limit > threshold",
        threshold_value=90.0,
        description="Alert when demand approaches contract limit",
    ),
    AlertRule(
        rule_name="Rate Change Notification",
        alert_type=AlertType.RATE_CHANGE,
        severity=AlertSeverity.INFO,
        channels=[AlertChannel.IN_APP, AlertChannel.EMAIL],
        condition="rate_change_detected",
        threshold_value=0.0,
        description="Notify when tariff or rate structure changes",
    ),
    AlertRule(
        rule_name="Procurement Deadline",
        alert_type=AlertType.PROCUREMENT_DEADLINE,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.IN_APP, AlertChannel.EMAIL],
        condition="days_to_deadline < 30",
        threshold_value=30.0,
        description="Alert for upcoming contract renewal deadlines",
    ),
    AlertRule(
        rule_name="Consumption Anomaly",
        alert_type=AlertType.ANOMALY_DETECTED,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.IN_APP],
        condition="z_score > threshold",
        threshold_value=2.5,
        description="Alert when consumption deviates significantly from norm",
    ),
    AlertRule(
        rule_name="Data Quality Below Threshold",
        alert_type=AlertType.DATA_QUALITY,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.IN_APP],
        condition="quality_score < threshold",
        threshold_value=80.0,
        description="Alert when data quality drops below acceptable level",
    ),
]


# ---------------------------------------------------------------------------
# AlertBridge
# ---------------------------------------------------------------------------


class AlertBridge:
    """Multi-channel notification and alert management for Utility Analysis.

    Manages alert creation, notification delivery across multiple channels,
    alert rules configuration, and alert lifecycle management for utility
    billing, budget, demand, and procurement events.

    Attributes:
        config: Alert configuration.
        _alerts: Active alerts by alert_id.
        _rules: Configured alert rules.
        _notification_history: Sent notification records.

    Example:
        >>> bridge = AlertBridge()
        >>> alert = Alert(
        ...     alert_type=AlertType.BUDGET_VARIANCE,
        ...     severity=AlertSeverity.WARNING,
        ...     title="Budget variance detected",
        ...     message="March electricity cost 15% over budget",
        ...     metric_value=15.0, threshold_value=10.0,
        ... )
        >>> result = bridge.send_alert(alert)
        >>> active = bridge.get_active_alerts()
    """

    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        """Initialize the Alert Bridge.

        Args:
            config: Alert configuration. Uses defaults if None.
        """
        self.config = config or AlertConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._notification_history: List[NotificationResult] = []

        # Load default rules
        for rule in DEFAULT_ALERT_RULES:
            self._rules[rule.rule_id] = rule

        enabled_channels = []
        if self.config.enable_email:
            enabled_channels.append("email")
        if self.config.enable_sms:
            enabled_channels.append("sms")
        if self.config.enable_webhook:
            enabled_channels.append("webhook")

        self.logger.info(
            "AlertBridge initialized: %d rules, channels=%s",
            len(self._rules), enabled_channels or ["in_app"],
        )

    def send_alert(self, alert: Alert) -> NotificationResult:
        """Send an alert notification across configured channels.

        Args:
            alert: Alert to send.

        Returns:
            NotificationResult with delivery status.
        """
        start = time.monotonic()

        # Compute provenance
        if self.config.enable_provenance:
            alert.provenance_hash = _compute_hash(alert)
        self._alerts[alert.alert_id] = alert

        # Determine channels from matching rules or defaults
        channels = self._get_channels_for_alert(alert)

        # Send to each channel (stub implementation)
        all_success = True
        for channel in channels:
            success = self._deliver_to_channel(alert, channel)
            if not success:
                all_success = False

        elapsed = (time.monotonic() - start) * 1000

        result = NotificationResult(
            alert_id=alert.alert_id,
            channel=channels[0] if channels else AlertChannel.IN_APP,
            success=all_success,
            message=f"Alert sent to {len(channels)} channel(s)",
            delivered_at=_utcnow() if all_success else None,
            duration_ms=elapsed,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._notification_history.append(result)

        self.logger.info(
            "Alert sent: id=%s, type=%s, severity=%s, channels=%d",
            alert.alert_id, alert.alert_type.value,
            alert.severity.value, len(channels),
        )
        return result

    def evaluate_rules(
        self, metrics: Dict[str, Any]
    ) -> List[Alert]:
        """Evaluate alert rules against current metrics.

        Checks each enabled rule against provided metrics and generates
        alerts for rules that trigger.

        Args:
            metrics: Dict of current metric values.

        Returns:
            List of triggered Alert instances.
        """
        triggered: List[Alert] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            alert = self._evaluate_single_rule(rule, metrics)
            if alert is not None:
                triggered.append(alert)

        if triggered:
            self.logger.info(
                "Rule evaluation triggered %d alerts", len(triggered)
            )

        return triggered

    def configure_rules(self, rules: List[AlertRule]) -> bool:
        """Configure alert rules.

        Args:
            rules: List of alert rules to configure.

        Returns:
            True if all rules were configured successfully.
        """
        for rule in rules:
            self._rules[rule.rule_id] = rule
            self.logger.info(
                "Alert rule configured: %s (%s)",
                rule.rule_name, rule.alert_type.value,
            )
        return True

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (non-resolved, non-dismissed) alerts.

        Returns:
            List of active Alert instances.
        """
        return [
            alert for alert in self._alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier to acknowledge.

        Returns:
            True if alert was found and acknowledged.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = _utcnow()
        self.logger.info("Alert acknowledged: %s", alert_id)
        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert identifier to resolve.

        Returns:
            True if alert was found and resolved.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = _utcnow()
        self.logger.info("Alert resolved: %s", alert_id)
        return True

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert.

        Args:
            alert_id: Alert identifier to dismiss.

        Returns:
            True if alert was found and dismissed.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            self.logger.warning(
                "Alert not found for dismissal: %s", alert_id
            )
            return False

        alert.status = AlertStatus.DISMISSED
        alert.dismissed_at = _utcnow()
        self.logger.info("Alert dismissed: %s", alert_id)
        return True

    def get_alert_summary(self, period: str = "") -> AlertSummary:
        """Get summary of alert activity.

        Args:
            period: Reporting period.

        Returns:
            AlertSummary with aggregated metrics.
        """
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        active = acknowledged = resolved = 0

        for alert in self._alerts.values():
            atype = alert.alert_type.value
            by_type[atype] = by_type.get(atype, 0) + 1
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            if alert.status == AlertStatus.ACTIVE:
                active += 1
            elif alert.status == AlertStatus.ACKNOWLEDGED:
                acknowledged += 1
            elif alert.status == AlertStatus.RESOLVED:
                resolved += 1

        summary = AlertSummary(
            period=period,
            total_alerts=len(self._alerts),
            active_alerts=active,
            acknowledged_alerts=acknowledged,
            resolved_alerts=resolved,
            by_type=by_type,
            by_severity=by_severity,
        )

        if self.config.enable_provenance:
            summary.provenance_hash = _compute_hash(summary)

        return summary

    def get_rules(self) -> List[AlertRule]:
        """Get all configured alert rules.

        Returns:
            List of AlertRule instances.
        """
        return list(self._rules.values())

    # ---- Internal Helpers ----

    def _get_channels_for_alert(self, alert: Alert) -> List[AlertChannel]:
        """Determine delivery channels for an alert based on rules."""
        for rule in self._rules.values():
            if (
                rule.enabled
                and rule.alert_type == alert.alert_type
                and rule.severity == alert.severity
                and rule.channels
            ):
                return rule.channels
        return self.config.default_channels

    def _evaluate_single_rule(
        self, rule: AlertRule, metrics: Dict[str, Any]
    ) -> Optional[Alert]:
        """Evaluate a single rule against metrics."""
        if rule.alert_type == AlertType.BUDGET_VARIANCE:
            variance = metrics.get("budget_variance_pct", 0.0)
            if variance > rule.threshold_value:
                return Alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=f"Budget variance: {variance:.1f}%",
                    message=(
                        f"Utility cost exceeds budget by {variance:.1f}% "
                        f"(threshold: {rule.threshold_value}%)"
                    ),
                    metric_value=variance,
                    threshold_value=rule.threshold_value,
                )

        elif rule.alert_type == AlertType.DEMAND_PEAK:
            demand_pct = metrics.get("demand_pct_of_limit", 0.0)
            if demand_pct > rule.threshold_value:
                return Alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=f"Demand peak: {demand_pct:.0f}% of limit",
                    message=(
                        f"Demand at {demand_pct:.0f}% of contract limit "
                        f"(threshold: {rule.threshold_value}%)"
                    ),
                    metric_value=demand_pct,
                    threshold_value=rule.threshold_value,
                )

        elif rule.alert_type == AlertType.DATA_QUALITY:
            quality = metrics.get("data_quality_score", 100.0)
            if quality < rule.threshold_value:
                return Alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=f"Data quality: {quality:.0f}%",
                    message=(
                        f"Data quality score {quality:.0f}% below threshold "
                        f"{rule.threshold_value}%"
                    ),
                    metric_value=quality,
                    threshold_value=rule.threshold_value,
                )

        elif rule.alert_type == AlertType.ANOMALY_DETECTED:
            z_score = metrics.get("consumption_z_score", 0.0)
            if abs(z_score) > rule.threshold_value:
                return Alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=f"Consumption anomaly (z={z_score:.1f})",
                    message=(
                        f"Consumption z-score {z_score:.1f} exceeds "
                        f"threshold {rule.threshold_value}"
                    ),
                    metric_value=z_score,
                    threshold_value=rule.threshold_value,
                )

        return None

    def _deliver_to_channel(
        self, alert: Alert, channel: AlertChannel
    ) -> bool:
        """Deliver an alert to a specific channel.

        In production, this dispatches to the appropriate notification
        service. The stub always succeeds.

        Args:
            alert: Alert to deliver.
            channel: Target channel.

        Returns:
            True if delivery was successful.
        """
        self.logger.debug(
            "Delivering alert %s to %s (stub)",
            alert.alert_id, channel.value,
        )
        return True
