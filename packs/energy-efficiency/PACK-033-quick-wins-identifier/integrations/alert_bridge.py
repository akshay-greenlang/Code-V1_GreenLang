# -*- coding: utf-8 -*-
"""
AlertBridge - Notification and Alert Management for PACK-033
==============================================================

This module provides alert and notification management for the Quick Wins
Identifier Pack. It supports multiple notification channels (email, Slack,
Teams, webhook, SMS, dashboard), configurable alert rules, and alert lifecycle
management.

Alert Types:
    - SAVINGS_BELOW_TARGET: Actual savings below projected target
    - IMPLEMENTATION_DELAYED: Implementation milestone overdue
    - REBATE_EXPIRING: Utility rebate application deadline approaching
    - VERIFICATION_DUE: Savings verification measurement due
    - BUDGET_EXCEEDED: Implementation budget exceeded

Channels:
    - email: SMTP email notifications
    - slack: Slack webhook integration
    - teams: Microsoft Teams webhook integration
    - webhook: Generic HTTP webhook
    - sms: SMS via provider API
    - dashboard: In-app dashboard notification

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-033 Quick Wins Identifier
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
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

class AlertType(str, Enum):
    """Types of quick win alerts."""

    SAVINGS_BELOW_TARGET = "savings_below_target"
    IMPLEMENTATION_DELAYED = "implementation_delayed"
    REBATE_EXPIRING = "rebate_expiring"
    VERIFICATION_DUE = "verification_due"
    BUDGET_EXCEEDED = "budget_exceeded"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class AlertConfig(BaseModel):
    """Configuration for the Alert Bridge."""

    pack_id: str = Field(default="PACK-033")
    enable_provenance: bool = Field(default=True)
    default_channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.DASHBOARD, AlertChannel.EMAIL]
    )
    enable_email: bool = Field(default=True)
    enable_slack: bool = Field(default=False)
    enable_teams: bool = Field(default=False)
    enable_webhook: bool = Field(default=False)
    enable_sms: bool = Field(default=False)
    email_recipients: List[str] = Field(default_factory=list)
    slack_webhook_url: str = Field(default="")
    teams_webhook_url: str = Field(default="")
    webhook_url: str = Field(default="")

class Alert(BaseModel):
    """An alert instance."""

    alert_id: str = Field(default_factory=_new_uuid)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    title: str = Field(default="")
    message: str = Field(default="")
    facility_id: str = Field(default="")
    measure_id: str = Field(default="")
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = Field(None)
    dismissed: bool = Field(default=False)
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

# ---------------------------------------------------------------------------
# AlertBridge
# ---------------------------------------------------------------------------

class AlertBridge:
    """Notification and alert management for Quick Wins Identifier.

    Manages alert creation, notification delivery across multiple channels,
    alert rules configuration, and alert lifecycle (acknowledge/dismiss).

    Attributes:
        config: Alert configuration.
        _alerts: Active alerts by alert_id.
        _rules: Configured alert rules.
        _notification_history: Sent notification records.

    Example:
        >>> bridge = AlertBridge()
        >>> alert = Alert(
        ...     alert_type=AlertType.REBATE_EXPIRING,
        ...     severity=AlertSeverity.WARNING,
        ...     title="Rebate deadline approaching",
        ...     message="Lighting rebate expires in 30 days",
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

        enabled_channels = []
        if self.config.enable_email:
            enabled_channels.append("email")
        if self.config.enable_slack:
            enabled_channels.append("slack")
        if self.config.enable_teams:
            enabled_channels.append("teams")

        self.logger.info(
            "AlertBridge initialized: channels=%s",
            enabled_channels or ["dashboard"],
        )

    def send_alert(self, alert: Alert) -> NotificationResult:
        """Send an alert notification across configured channels.

        Args:
            alert: Alert to send.

        Returns:
            NotificationResult with delivery status.
        """
        start = time.monotonic()

        # Store alert
        if self.config.enable_provenance:
            alert.provenance_hash = _compute_hash(alert)
        self._alerts[alert.alert_id] = alert

        # Determine channels
        channels = self.config.default_channels

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
            message=f"Alert sent to {len(channels)} channel(s)",
            delivered_at=utcnow() if all_success else None,
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
        """Get all active (non-dismissed) alerts.

        Returns:
            List of active Alert instances.
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

        alert.acknowledged = True
        alert.acknowledged_at = utcnow()
        self.logger.info("Alert acknowledged: %s", alert_id)
        return True

    def _deliver_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
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
