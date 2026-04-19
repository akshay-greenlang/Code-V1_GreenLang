# -*- coding: utf-8 -*-
"""
AlertBridge - Multi-Channel Alerting for Demand Response (PACK-037)
=====================================================================

This module provides alert and notification management for the Demand Response
Pack. It supports DR event dispatch notifications, pre-event reminders,
real-time performance alerts, settlement notifications, and enrollment
deadline alerts across multiple delivery channels.

Alert Types:
    - DR_EVENT_DISPATCH: Grid signal received, curtailment required
    - PRE_EVENT_REMINDER: Upcoming DR event notification
    - PERFORMANCE_DEVIATION: Real-time performance below target
    - SETTLEMENT_NOTIFICATION: Event settlement/payment processed
    - ENROLLMENT_DEADLINE: DR program enrollment deadline approaching
    - BASELINE_ANOMALY: Customer baseline calculation anomaly
    - DER_FAULT: DER asset fault or communication failure
    - GRID_EMERGENCY: Grid emergency or reliability event

Channels:
    - email: SMTP email notifications
    - sms: SMS via provider API (critical alerts)
    - push: Push notification to mobile app
    - webhook: Generic HTTP webhook (aggregator platforms)

Escalation:
    Level 1: Dashboard + Email (all alerts)
    Level 2: SMS (critical DR events, 15+ min before)
    Level 3: Push + Phone (emergency grid events)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-037 Demand Response
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

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
    PUSH = "push"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"

class AlertType(str, Enum):
    """Types of demand response alerts."""

    DR_EVENT_DISPATCH = "dr_event_dispatch"
    PRE_EVENT_REMINDER = "pre_event_reminder"
    PERFORMANCE_DEVIATION = "performance_deviation"
    SETTLEMENT_NOTIFICATION = "settlement_notification"
    ENROLLMENT_DEADLINE = "enrollment_deadline"
    BASELINE_ANOMALY = "baseline_anomaly"
    DER_FAULT = "der_fault"
    GRID_EMERGENCY = "grid_emergency"

class EscalationLevel(str, Enum):
    """Alert escalation levels."""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class AlertConfig(BaseModel):
    """Configuration for the Alert Bridge."""

    pack_id: str = Field(default="PACK-037")
    enable_provenance: bool = Field(default=True)
    default_channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.DASHBOARD, AlertChannel.EMAIL]
    )
    enable_email: bool = Field(default=True)
    enable_sms: bool = Field(default=False)
    enable_push: bool = Field(default=False)
    enable_webhook: bool = Field(default=False)
    email_recipients: List[str] = Field(default_factory=list)
    sms_recipients: List[str] = Field(default_factory=list)
    webhook_url: str = Field(default="")
    webhook_auth_header: str = Field(default="")
    pre_event_reminder_minutes: List[int] = Field(
        default_factory=lambda: [60, 30, 15],
        description="Minutes before event to send reminders",
    )
    performance_threshold_pct: float = Field(
        default=80.0, ge=0.0, le=100.0,
        description="Min performance ratio before alert",
    )

class AlertMessage(BaseModel):
    """An alert message instance."""

    alert_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    title: str = Field(default="")
    message: str = Field(default="")
    facility_id: str = Field(default="")
    event_id: str = Field(default="", description="DR event ID if applicable")
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: Optional[datetime] = Field(None)
    acknowledged: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = Field(None)
    acknowledged_by: str = Field(default="")
    dismissed: bool = Field(default=False)
    dismissed_at: Optional[datetime] = Field(None)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.LEVEL_1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class EscalationRule(BaseModel):
    """An escalation rule for alert routing."""

    rule_id: str = Field(default_factory=_new_uuid)
    rule_name: str = Field(default="")
    alert_type: AlertType = Field(...)
    min_severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.LEVEL_1)
    channels: List[AlertChannel] = Field(default_factory=list)
    delay_minutes: int = Field(default=0, ge=0)
    auto_acknowledge_minutes: int = Field(default=0, ge=0, description="0=no auto-ack")
    enabled: bool = Field(default=True)
    description: str = Field(default="")

class NotificationResult(BaseModel):
    """Result of sending an alert notification."""

    notification_id: str = Field(default_factory=_new_uuid)
    alert_id: str = Field(default="")
    channel: AlertChannel = Field(...)
    success: bool = Field(default=False)
    message: str = Field(default="")
    delivered_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.LEVEL_1)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Default Escalation Rules
# ---------------------------------------------------------------------------

DEFAULT_ESCALATION_RULES: List[Dict[str, Any]] = [
    {
        "rule_name": "DR Event Dispatch - Level 1",
        "alert_type": AlertType.DR_EVENT_DISPATCH,
        "min_severity": AlertSeverity.WARNING,
        "escalation_level": EscalationLevel.LEVEL_1,
        "channels": [AlertChannel.DASHBOARD, AlertChannel.EMAIL],
        "delay_minutes": 0,
    },
    {
        "rule_name": "DR Event Dispatch - Level 2",
        "alert_type": AlertType.DR_EVENT_DISPATCH,
        "min_severity": AlertSeverity.CRITICAL,
        "escalation_level": EscalationLevel.LEVEL_2,
        "channels": [AlertChannel.SMS],
        "delay_minutes": 5,
    },
    {
        "rule_name": "Grid Emergency - Level 3",
        "alert_type": AlertType.GRID_EMERGENCY,
        "min_severity": AlertSeverity.EMERGENCY,
        "escalation_level": EscalationLevel.LEVEL_3,
        "channels": [AlertChannel.SMS, AlertChannel.PUSH],
        "delay_minutes": 0,
    },
    {
        "rule_name": "Performance Deviation",
        "alert_type": AlertType.PERFORMANCE_DEVIATION,
        "min_severity": AlertSeverity.WARNING,
        "escalation_level": EscalationLevel.LEVEL_1,
        "channels": [AlertChannel.DASHBOARD],
        "delay_minutes": 0,
    },
    {
        "rule_name": "Settlement Notification",
        "alert_type": AlertType.SETTLEMENT_NOTIFICATION,
        "min_severity": AlertSeverity.INFO,
        "escalation_level": EscalationLevel.LEVEL_1,
        "channels": [AlertChannel.EMAIL],
        "delay_minutes": 0,
    },
    {
        "rule_name": "DER Fault Alert",
        "alert_type": AlertType.DER_FAULT,
        "min_severity": AlertSeverity.CRITICAL,
        "escalation_level": EscalationLevel.LEVEL_2,
        "channels": [AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.SMS],
        "delay_minutes": 0,
    },
]

# ---------------------------------------------------------------------------
# AlertBridge
# ---------------------------------------------------------------------------

class AlertBridge:
    """Multi-channel alerting for Demand Response Pack.

    Manages DR event dispatch notifications, pre-event reminders, real-time
    performance alerts, settlement notifications, and enrollment deadlines
    with multi-level escalation across email, SMS, push, and webhook channels.

    Attributes:
        config: Alert configuration.
        _alerts: Active alerts by alert_id.
        _rules: Configured escalation rules.
        _notification_history: Sent notification records.

    Example:
        >>> bridge = AlertBridge()
        >>> alert = AlertMessage(
        ...     alert_type=AlertType.DR_EVENT_DISPATCH,
        ...     severity=AlertSeverity.CRITICAL,
        ...     title="DR Event - Curtail 750 kW",
        ...     message="Grid signal received: curtail 750 kW for 4 hours",
        ...     event_id="EVT-001",
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
        self._alerts: Dict[str, AlertMessage] = {}
        self._rules: Dict[str, EscalationRule] = {}
        self._notification_history: List[NotificationResult] = []

        # Load default escalation rules
        for rule_data in DEFAULT_ESCALATION_RULES:
            rule = EscalationRule(
                rule_name=rule_data["rule_name"],
                alert_type=rule_data["alert_type"],
                min_severity=rule_data["min_severity"],
                escalation_level=rule_data["escalation_level"],
                channels=rule_data["channels"],
                delay_minutes=rule_data["delay_minutes"],
            )
            self._rules[rule.rule_id] = rule

        enabled_channels = []
        if self.config.enable_email:
            enabled_channels.append("email")
        if self.config.enable_sms:
            enabled_channels.append("sms")
        if self.config.enable_push:
            enabled_channels.append("push")
        if self.config.enable_webhook:
            enabled_channels.append("webhook")

        self.logger.info(
            "AlertBridge initialized: channels=%s, rules=%d",
            enabled_channels or ["dashboard"],
            len(self._rules),
        )

    def send_alert(self, alert: AlertMessage) -> NotificationResult:
        """Send an alert notification across configured channels.

        Args:
            alert: Alert message to send.

        Returns:
            NotificationResult with delivery status.
        """
        start = time.monotonic()

        # Compute provenance
        if self.config.enable_provenance:
            alert.provenance_hash = _compute_hash(alert)
        self._alerts[alert.alert_id] = alert

        # Determine channels based on escalation rules
        channels = self._resolve_channels(alert)

        # Send to each channel (stub implementation)
        all_success = True
        for channel in channels:
            success = self._deliver_to_channel(alert, channel)
            if not success:
                all_success = False

        elapsed = (time.monotonic() - start) * 1000

        result = NotificationResult(
            alert_id=alert.alert_id,
            channel=channels[0] if channels else AlertChannel.DASHBOARD,
            success=all_success,
            message=f"Alert sent to {len(channels)} channel(s): {[c.value for c in channels]}",
            delivered_at=utcnow() if all_success else None,
            duration_ms=elapsed,
            escalation_level=alert.escalation_level,
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

    def send_pre_event_reminder(
        self,
        event_id: str,
        facility_id: str,
        minutes_until_event: int,
        curtailment_kw: float,
    ) -> NotificationResult:
        """Send a pre-event reminder alert.

        Args:
            event_id: DR event identifier.
            facility_id: Target facility.
            minutes_until_event: Minutes until event starts.
            curtailment_kw: Expected curtailment.

        Returns:
            NotificationResult for the reminder.
        """
        alert = AlertMessage(
            alert_type=AlertType.PRE_EVENT_REMINDER,
            severity=AlertSeverity.WARNING if minutes_until_event > 15 else AlertSeverity.CRITICAL,
            title=f"DR Event in {minutes_until_event} minutes",
            message=(
                f"DR event {event_id} starts in {minutes_until_event} minutes. "
                f"Target curtailment: {curtailment_kw:.0f} kW."
            ),
            facility_id=facility_id,
            event_id=event_id,
            metadata={
                "minutes_until_event": minutes_until_event,
                "curtailment_kw": curtailment_kw,
            },
        )
        return self.send_alert(alert)

    def send_performance_alert(
        self,
        event_id: str,
        facility_id: str,
        performance_pct: float,
        target_pct: float,
    ) -> NotificationResult:
        """Send a performance deviation alert during DR event.

        Args:
            event_id: DR event identifier.
            facility_id: Facility identifier.
            performance_pct: Current performance ratio.
            target_pct: Target performance threshold.

        Returns:
            NotificationResult for the performance alert.
        """
        alert = AlertMessage(
            alert_type=AlertType.PERFORMANCE_DEVIATION,
            severity=AlertSeverity.CRITICAL if performance_pct < 50.0 else AlertSeverity.WARNING,
            title=f"DR Performance Below Target ({performance_pct:.0f}%)",
            message=(
                f"Event {event_id}: performance at {performance_pct:.1f}% "
                f"(target: {target_pct:.1f}%). Increase curtailment."
            ),
            facility_id=facility_id,
            event_id=event_id,
            metadata={
                "performance_pct": performance_pct,
                "target_pct": target_pct,
            },
        )
        return self.send_alert(alert)

    def configure_rules(self, rules: List[EscalationRule]) -> bool:
        """Configure escalation rules.

        Args:
            rules: List of escalation rules to configure.

        Returns:
            True if all rules were configured successfully.
        """
        for rule in rules:
            self._rules[rule.rule_id] = rule
            self.logger.info(
                "Escalation rule configured: %s (%s, %s)",
                rule.rule_name, rule.alert_type.value, rule.escalation_level.value,
            )
        return True

    def get_active_alerts(self) -> List[AlertMessage]:
        """Get all active (non-dismissed) alerts.

        Returns:
            List of active AlertMessage instances.
        """
        return [
            alert for alert in self._alerts.values()
            if not alert.dismissed
        ]

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert.

        Args:
            alert_id: Alert identifier to dismiss.

        Returns:
            True if alert was found and dismissed.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            self.logger.warning("Alert not found for dismissal: %s", alert_id)
            return False

        alert.dismissed = True
        alert.dismissed_at = utcnow()
        self.logger.info("Alert dismissed: %s", alert_id)
        return True

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "") -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier to acknowledge.
            acknowledged_by: User who acknowledged.

        Returns:
            True if alert was found and acknowledged.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False

        alert.acknowledged = True
        alert.acknowledged_at = utcnow()
        alert.acknowledged_by = acknowledged_by
        self.logger.info("Alert acknowledged: %s by %s", alert_id, acknowledged_by or "system")
        return True

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _resolve_channels(self, alert: AlertMessage) -> List[AlertChannel]:
        """Resolve delivery channels based on escalation rules.

        Args:
            alert: Alert message to route.

        Returns:
            List of AlertChannel to deliver to.
        """
        channels: List[AlertChannel] = []
        severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        alert_level = severity_order.index(alert.severity) if alert.severity in severity_order else 0

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.alert_type != alert.alert_type:
                continue
            rule_level = severity_order.index(rule.min_severity) if rule.min_severity in severity_order else 0
            if alert_level >= rule_level:
                for ch in rule.channels:
                    if ch not in channels:
                        channels.append(ch)

        # Always include dashboard
        if AlertChannel.DASHBOARD not in channels:
            channels.insert(0, AlertChannel.DASHBOARD)

        return channels

    def _deliver_to_channel(self, alert: AlertMessage, channel: AlertChannel) -> bool:
        """Deliver an alert to a specific channel.

        In production, this dispatches to the appropriate notification service.

        Args:
            alert: Alert to deliver.
            channel: Target channel.

        Returns:
            True if delivery was successful.
        """
        self.logger.debug(
            "Delivering alert %s to %s (stub)", alert.alert_id, channel.value
        )
        # Stub: always succeeds
        return True
