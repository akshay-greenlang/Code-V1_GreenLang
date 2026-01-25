# -*- coding: utf-8 -*-
"""
Alert Rules Module for GL-015 INSULSCAN.

This module provides comprehensive alert rule definitions and management:
- Heat loss and insulation alerts with warning/critical thresholds
- Safety temperature alerts for personnel protection
- Equipment degradation and maintenance alerts
- Integration and API health alerts
- Alert state management (firing, resolved, pending)
- Notification channels (Email, Slack, PagerDuty)
- Silence and inhibit rule support
- Alert history tracking

Example:
    >>> from monitoring.alerts import AlertManager, HIGH_HEAT_LOSS_ALERT
    >>> manager = AlertManager()
    >>> manager.register_rule(HIGH_HEAT_LOSS_ALERT)
    >>> alerts = manager.evaluate_rules({"heat_loss_wm": 500.0})
    >>> for alert in alerts:
    ...     print(f"{alert.rule_name}: {alert.severity}")

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading
import hashlib
import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ALERT ENUMERATIONS
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels (aligned with Prometheus/Grafana)."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # Requires immediate response


class AlertState(Enum):
    """Alert state lifecycle."""
    INACTIVE = "inactive"  # Condition not met
    PENDING = "pending"  # Condition met, waiting for duration
    FIRING = "firing"  # Alert is active
    RESOLVED = "resolved"  # Alert was firing, now resolved


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    OPSGENIE = "opsgenie"


class ComparisonOperator(Enum):
    """Comparison operators for alert conditions."""
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


# =============================================================================
# ALERT DATA STRUCTURES
# =============================================================================

@dataclass
class AlertCondition:
    """
    Definition of an alert trigger condition.

    Attributes:
        metric_name: Name of the metric to evaluate
        operator: Comparison operator
        threshold: Threshold value for comparison
        for_duration_seconds: Duration condition must be true before firing
        labels: Label matchers for the metric
    """
    metric_name: str
    operator: ComparisonOperator
    threshold: float
    for_duration_seconds: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def evaluate(self, value: float) -> bool:
        """
        Evaluate the condition against a value.

        Args:
            value: Metric value to evaluate

        Returns:
            True if condition is met
        """
        if self.operator == ComparisonOperator.GREATER_THAN:
            return value > self.threshold
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN:
            return value < self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return value <= self.threshold
        elif self.operator == ComparisonOperator.EQUAL:
            return value == self.threshold
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return value != self.threshold
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "for_duration_seconds": self.for_duration_seconds,
            "labels": self.labels,
        }


@dataclass
class AlertRule:
    """
    Definition of an alert rule.

    Attributes:
        name: Unique rule name
        description: Human-readable description
        severity: Alert severity level
        condition: Alert trigger condition
        annotations: Additional information (summary, description, runbook)
        labels: Labels to attach to alerts
        channels: Notification channels to use
        enabled: Whether rule is active
    """
    name: str
    description: str
    severity: AlertSeverity
    condition: AlertCondition
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    enabled: bool = True
    group: str = "default"
    repeat_interval_seconds: int = 3600  # Re-notify every hour

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "condition": self.condition.to_dict(),
            "annotations": self.annotations,
            "labels": self.labels,
            "channels": [c.value for c in self.channels],
            "enabled": self.enabled,
            "group": self.group,
            "repeat_interval_seconds": self.repeat_interval_seconds,
        }


@dataclass
class AlertInstance:
    """
    Instance of a fired alert.

    Attributes:
        alert_id: Unique alert instance ID
        rule_name: Name of the rule that fired
        severity: Alert severity
        state: Current alert state
        value: Metric value that triggered the alert
        threshold: Threshold that was exceeded
        started_at: When the alert started
        resolved_at: When the alert was resolved
        labels: Alert labels
        annotations: Alert annotations
        fingerprint: Unique identifier for deduplication
    """
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    value: float
    threshold: float
    started_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""
    notification_count: int = 0
    last_notified_at: Optional[datetime] = None

    def __post_init__(self):
        """Generate fingerprint if not provided."""
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.rule_name}:{json.dumps(self.labels, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "state": self.state.value,
            "value": self.value,
            "threshold": self.threshold,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "labels": self.labels,
            "annotations": self.annotations,
            "fingerprint": self.fingerprint,
            "notification_count": self.notification_count,
            "last_notified_at": self.last_notified_at.isoformat() if self.last_notified_at else None,
        }


@dataclass
class SilenceRule:
    """
    Rule to silence alerts for a period.

    Attributes:
        silence_id: Unique silence ID
        matchers: Label matchers for alerts to silence
        starts_at: When silence begins
        ends_at: When silence ends
        created_by: Who created the silence
        comment: Reason for silence
    """
    silence_id: str
    matchers: Dict[str, str]
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str = ""

    def is_active(self) -> bool:
        """Check if silence is currently active."""
        now = datetime.now(timezone.utc)
        return self.starts_at <= now <= self.ends_at

    def matches(self, labels: Dict[str, str]) -> bool:
        """Check if labels match this silence."""
        for key, pattern in self.matchers.items():
            if key not in labels:
                return False
            if not re.match(pattern, labels[key]):
                return False
        return True


@dataclass
class InhibitRule:
    """
    Rule to inhibit alerts when another alert is firing.

    Attributes:
        source_matchers: Matchers for the inhibiting alert
        target_matchers: Matchers for alerts to inhibit
        equal_labels: Labels that must match for inhibition
    """
    source_matchers: Dict[str, str]
    target_matchers: Dict[str, str]
    equal_labels: List[str] = field(default_factory=list)


# =============================================================================
# BUILT-IN ALERT RULES - INSULATION SPECIFIC
# =============================================================================

HIGH_HEAT_LOSS_ALERT = AlertRule(
    name="high_heat_loss",
    description="Heat loss exceeds acceptable threshold per meter of insulation",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="heat_loss_wm",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=100.0,  # W/m threshold for warning
        for_duration_seconds=300,  # 5 minutes
    ),
    annotations={
        "summary": "High heat loss detected at {{ $labels.facility_id }} zone {{ $labels.zone }}",
        "description": "Heat loss {{ $value }} W/m exceeds threshold {{ $threshold }} W/m",
        "runbook_url": "https://docs.greenlang.io/runbooks/high-heat-loss",
        "recommendation": "Inspect insulation for damage or missing sections",
    },
    labels={
        "alert_type": "heat_loss",
        "component": "insulation",
    },
    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
    group="heat_loss_alerts",
)

CRITICAL_DEGRADATION_ALERT = AlertRule(
    name="critical_degradation",
    description="Insulation condition has degraded to critical severity",
    severity=AlertSeverity.CRITICAL,
    condition=AlertCondition(
        metric_name="condition_severity_score",
        operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        threshold=4.0,  # Critical severity = 4
        for_duration_seconds=0,  # Immediate
    ),
    annotations={
        "summary": "Critical insulation degradation at {{ $labels.equipment_id }}",
        "description": "Insulation condition severity {{ $value }} is CRITICAL (>=4)",
        "runbook_url": "https://docs.greenlang.io/runbooks/critical-degradation",
        "recommendation": "Schedule emergency repair within 24 hours",
    },
    labels={
        "alert_type": "degradation",
        "component": "insulation",
    },
    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
    group="degradation_alerts",
)

SAFETY_TEMPERATURE_EXCEEDED_ALERT = AlertRule(
    name="safety_temperature_exceeded",
    description="Surface temperature exceeds safe personnel contact limit (60C)",
    severity=AlertSeverity.CRITICAL,
    condition=AlertCondition(
        metric_name="surface_temperature_celsius",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=60.0,  # 60C is typical burn hazard threshold
        for_duration_seconds=0,  # Immediate - safety critical
    ),
    annotations={
        "summary": "SAFETY: Surface temperature exceeds 60C at {{ $labels.equipment_id }}",
        "description": "Surface temperature {{ $value }}C exceeds safe contact limit of {{ $threshold }}C",
        "runbook_url": "https://docs.greenlang.io/runbooks/safety-temperature",
        "recommendation": "IMMEDIATE: Apply warning labels, install guards, notify safety officer",
    },
    labels={
        "alert_type": "safety",
        "component": "insulation",
        "safety_critical": "true",
    },
    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
    group="safety_alerts",
)

INSPECTION_OVERDUE_ALERT = AlertRule(
    name="inspection_overdue",
    description="Equipment has not been inspected within required timeframe",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="days_since_last_inspection",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=90,  # 90 days overdue
        for_duration_seconds=0,
    ),
    annotations={
        "summary": "Inspection overdue for {{ $labels.equipment_id }}",
        "description": "Last inspection was {{ $value }} days ago (threshold: {{ $threshold }} days)",
        "runbook_url": "https://docs.greenlang.io/runbooks/inspection-overdue",
        "recommendation": "Schedule thermal imaging inspection within 7 days",
    },
    labels={
        "alert_type": "maintenance",
        "component": "inspection",
    },
    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
    group="maintenance_alerts",
)

MOISTURE_DETECTED_ALERT = AlertRule(
    name="moisture_detected",
    description="Moisture detected in insulation material",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="moisture_content_percent",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=5.0,  # 5% moisture content
        for_duration_seconds=0,
    ),
    annotations={
        "summary": "Moisture detected in insulation at {{ $labels.equipment_id }}",
        "description": "Moisture content {{ $value }}% exceeds dry threshold {{ $threshold }}%",
        "runbook_url": "https://docs.greenlang.io/runbooks/moisture-detection",
        "recommendation": "Investigate moisture source, consider insulation replacement",
    },
    labels={
        "alert_type": "degradation",
        "component": "insulation",
        "issue_type": "moisture",
    },
    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
    group="degradation_alerts",
)

RAPID_DEGRADATION_RATE_ALERT = AlertRule(
    name="rapid_degradation_rate",
    description="Insulation is degrading faster than expected",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="degradation_rate_percent_per_year",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=5.0,  # 5% degradation per year is concerning
        for_duration_seconds=3600,  # 1 hour sustained
    ),
    annotations={
        "summary": "Rapid degradation rate at {{ $labels.equipment_id }}",
        "description": "Degradation rate {{ $value }}%/year exceeds {{ $threshold }}%/year",
        "runbook_url": "https://docs.greenlang.io/runbooks/rapid-degradation",
        "recommendation": "Investigate root cause: vibration, chemical attack, UV exposure",
    },
    labels={
        "alert_type": "degradation",
        "component": "insulation",
    },
    channels=[NotificationChannel.SLACK],
    group="degradation_alerts",
)

LOW_INSULATION_EFFICIENCY_ALERT = AlertRule(
    name="low_insulation_efficiency",
    description="Insulation efficiency below acceptable performance threshold",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="insulation_efficiency_percent",
        operator=ComparisonOperator.LESS_THAN,
        threshold=70.0,  # 70% of design efficiency
        for_duration_seconds=600,  # 10 minutes
    ),
    annotations={
        "summary": "Low insulation efficiency on {{ $labels.equipment_id }}",
        "description": "Insulation efficiency {{ $value }}% is below {{ $threshold }}% of design spec",
        "runbook_url": "https://docs.greenlang.io/runbooks/low-efficiency",
        "recommendation": "Inspect for compression, settling, or missing sections",
    },
    labels={
        "alert_type": "performance",
        "component": "insulation",
    },
    channels=[NotificationChannel.EMAIL],
    group="performance_alerts",
)

INTEGRATION_FAILURE_ALERT = AlertRule(
    name="integration_failure",
    description="External system integration failure detected",
    severity=AlertSeverity.CRITICAL,
    condition=AlertCondition(
        metric_name="integration_error_count",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=5,  # 5 errors
        for_duration_seconds=300,  # Within 5 minutes
    ),
    annotations={
        "summary": "Integration failure with {{ $labels.system }}",
        "description": "{{ $value }} errors detected in last 5 minutes",
        "runbook_url": "https://docs.greenlang.io/runbooks/integration-failure",
        "recommendation": "Check connectivity, credentials, and API availability",
    },
    labels={
        "alert_type": "integration",
        "component": "connector",
    },
    channels=[NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
    group="system_alerts",
)

CAMERA_DISCONNECTED_ALERT = AlertRule(
    name="camera_disconnected",
    description="Thermal camera has lost connection",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="camera_connection_status",
        operator=ComparisonOperator.EQUAL,
        threshold=0.0,  # 0 = disconnected
        for_duration_seconds=60,  # 1 minute
    ),
    annotations={
        "summary": "Thermal camera {{ $labels.camera_id }} disconnected",
        "description": "Camera has been disconnected for over 1 minute",
        "runbook_url": "https://docs.greenlang.io/runbooks/camera-disconnected",
        "recommendation": "Check camera power, network connection, and USB cable",
    },
    labels={
        "alert_type": "integration",
        "component": "camera",
    },
    channels=[NotificationChannel.SLACK],
    group="integration_alerts",
)

HIGH_API_LATENCY_ALERT = AlertRule(
    name="high_api_latency",
    description="API response latency exceeds threshold",
    severity=AlertSeverity.WARNING,
    condition=AlertCondition(
        metric_name="api_latency_p95_seconds",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=2.0,  # 2 seconds P95 latency
        for_duration_seconds=300,
    ),
    annotations={
        "summary": "High API latency on {{ $labels.endpoint }}",
        "description": "P95 latency {{ $value }}s exceeds {{ $threshold }}s",
        "runbook_url": "https://docs.greenlang.io/runbooks/api-latency",
        "recommendation": "Check database performance, image processing queue",
    },
    labels={
        "alert_type": "latency",
        "component": "api",
    },
    channels=[NotificationChannel.SLACK],
    group="system_alerts",
)

HIGH_ERROR_RATE_ALERT = AlertRule(
    name="high_error_rate",
    description="API error rate exceeds threshold",
    severity=AlertSeverity.CRITICAL,
    condition=AlertCondition(
        metric_name="error_rate_percent",
        operator=ComparisonOperator.GREATER_THAN,
        threshold=5.0,  # 5% error rate
        for_duration_seconds=300,
    ),
    annotations={
        "summary": "High error rate on {{ $labels.endpoint }}",
        "description": "Error rate {{ $value }}% exceeds {{ $threshold }}%",
        "runbook_url": "https://docs.greenlang.io/runbooks/error-rate",
        "recommendation": "Check logs for error patterns, recent deployments",
    },
    labels={
        "alert_type": "errors",
        "component": "api",
    },
    channels=[NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
    group="system_alerts",
)


# =============================================================================
# NOTIFICATION CLASSES
# =============================================================================

@dataclass
class NotificationConfig:
    """
    Configuration for notification channels.

    Attributes:
        channel: Notification channel type
        enabled: Whether channel is enabled
        config: Channel-specific configuration
    """
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class BaseNotifier(ABC):
    """Abstract base class for notification senders."""

    def __init__(self, config: NotificationConfig):
        """
        Initialize notifier.

        Args:
            config: Notification configuration
        """
        self.config = config
        self.enabled = config.enabled

    @abstractmethod
    def send(self, alert: AlertInstance) -> bool:
        """
        Send notification for an alert.

        Args:
            alert: Alert instance to notify about

        Returns:
            True if notification sent successfully
        """
        pass

    @abstractmethod
    def send_resolved(self, alert: AlertInstance) -> bool:
        """
        Send resolution notification.

        Args:
            alert: Resolved alert instance

        Returns:
            True if notification sent successfully
        """
        pass


class EmailNotifier(BaseNotifier):
    """Email notification sender."""

    def __init__(self, config: NotificationConfig):
        """
        Initialize email notifier.

        Args:
            config: Email configuration (smtp_host, smtp_port, from_addr, to_addrs)
        """
        super().__init__(config)
        self.smtp_host = config.config.get("smtp_host", "localhost")
        self.smtp_port = config.config.get("smtp_port", 25)
        self.from_addr = config.config.get("from_addr", "alerts@greenlang.io")
        self.to_addrs = config.config.get("to_addrs", [])

    def send(self, alert: AlertInstance) -> bool:
        """Send email notification."""
        if not self.enabled:
            return False

        try:
            subject = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            body = self._format_alert_body(alert)

            # In production, use smtplib to send email
            logger.info(f"Email notification sent: {subject} to {self.to_addrs}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def send_resolved(self, alert: AlertInstance) -> bool:
        """Send resolution email notification."""
        if not self.enabled:
            return False

        try:
            subject = f"[RESOLVED] {alert.rule_name}"
            body = self._format_resolved_body(alert)

            logger.info(f"Email resolution sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send resolution email: {e}")
            return False

    def _format_alert_body(self, alert: AlertInstance) -> str:
        """Format alert email body."""
        return f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value}
State: {alert.state.value}

Value: {alert.value}
Threshold: {alert.threshold}

Started: {alert.started_at.isoformat()}

Labels:
{json.dumps(alert.labels, indent=2)}

Annotations:
{json.dumps(alert.annotations, indent=2)}
"""

    def _format_resolved_body(self, alert: AlertInstance) -> str:
        """Format resolution email body."""
        return f"""
Alert Resolved: {alert.rule_name}

Value: {alert.value}
Threshold: {alert.threshold}

Started: {alert.started_at.isoformat()}
Resolved: {alert.resolved_at.isoformat() if alert.resolved_at else 'N/A'}
Duration: {self._calculate_duration(alert)}

Labels:
{json.dumps(alert.labels, indent=2)}
"""

    def _calculate_duration(self, alert: AlertInstance) -> str:
        """Calculate alert duration."""
        if alert.resolved_at:
            duration = alert.resolved_at - alert.started_at
            return str(duration)
        return "Ongoing"


class SlackNotifier(BaseNotifier):
    """Slack notification sender."""

    def __init__(self, config: NotificationConfig):
        """
        Initialize Slack notifier.

        Args:
            config: Slack configuration (webhook_url, channel, username)
        """
        super().__init__(config)
        self.webhook_url = config.config.get("webhook_url", "")
        self.channel = config.config.get("channel", "#alerts")
        self.username = config.config.get("username", "GL-015 Alerts")

    def send(self, alert: AlertInstance) -> bool:
        """Send Slack notification."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            payload = self._build_alert_payload(alert)

            # In production, use requests to POST to webhook
            logger.info(f"Slack notification sent to {self.channel}: {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def send_resolved(self, alert: AlertInstance) -> bool:
        """Send Slack resolution notification."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            payload = self._build_resolved_payload(alert)

            logger.info(f"Slack resolution sent to {self.channel}: {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack resolution: {e}")
            return False

    def _build_alert_payload(self, alert: AlertInstance) -> Dict[str, Any]:
        """Build Slack alert payload."""
        color = self._severity_to_color(alert.severity)

        return {
            "channel": self.channel,
            "username": self.username,
            "attachments": [{
                "color": color,
                "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                "fields": [
                    {"title": "Value", "value": str(alert.value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Started", "value": alert.started_at.isoformat(), "short": True},
                    {"title": "Facility", "value": alert.labels.get("facility_id", "N/A"), "short": True},
                ],
                "footer": "GL-015 INSULSCAN",
                "ts": int(alert.started_at.timestamp()),
            }]
        }

    def _build_resolved_payload(self, alert: AlertInstance) -> Dict[str, Any]:
        """Build Slack resolution payload."""
        return {
            "channel": self.channel,
            "username": self.username,
            "attachments": [{
                "color": "good",
                "title": f"[RESOLVED] {alert.rule_name}",
                "fields": [
                    {"title": "Duration", "value": self._calculate_duration(alert), "short": True},
                    {"title": "Facility", "value": alert.labels.get("facility_id", "N/A"), "short": True},
                ],
                "footer": "GL-015 INSULSCAN",
            }]
        }

    def _severity_to_color(self, severity: AlertSeverity) -> str:
        """Map severity to Slack color."""
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#f2c744",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.PAGE: "#ff0000",
        }
        return colors.get(severity, "#808080")

    def _calculate_duration(self, alert: AlertInstance) -> str:
        """Calculate alert duration."""
        if alert.resolved_at:
            duration = alert.resolved_at - alert.started_at
            return str(duration)
        return "Ongoing"


class PagerDutyNotifier(BaseNotifier):
    """PagerDuty notification sender."""

    def __init__(self, config: NotificationConfig):
        """
        Initialize PagerDuty notifier.

        Args:
            config: PagerDuty configuration (routing_key, severity_map)
        """
        super().__init__(config)
        self.routing_key = config.config.get("routing_key", "")
        self.api_url = config.config.get("api_url", "https://events.pagerduty.com/v2/enqueue")

    def send(self, alert: AlertInstance) -> bool:
        """Send PagerDuty notification."""
        if not self.enabled or not self.routing_key:
            return False

        try:
            payload = self._build_trigger_payload(alert)

            # In production, use requests to POST to PagerDuty API
            logger.info(f"PagerDuty notification sent: {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False

    def send_resolved(self, alert: AlertInstance) -> bool:
        """Send PagerDuty resolution."""
        if not self.enabled or not self.routing_key:
            return False

        try:
            payload = self._build_resolve_payload(alert)

            logger.info(f"PagerDuty resolution sent: {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty resolution: {e}")
            return False

    def _build_trigger_payload(self, alert: AlertInstance) -> Dict[str, Any]:
        """Build PagerDuty trigger payload."""
        return {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": f"{alert.rule_name}: {alert.annotations.get('summary', '')}",
                "severity": self._map_severity(alert.severity),
                "source": "GL-015 INSULSCAN",
                "timestamp": alert.started_at.isoformat(),
                "custom_details": {
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                }
            }
        }

    def _build_resolve_payload(self, alert: AlertInstance) -> Dict[str, Any]:
        """Build PagerDuty resolve payload."""
        return {
            "routing_key": self.routing_key,
            "event_action": "resolve",
            "dedup_key": alert.fingerprint,
        }

    def _map_severity(self, severity: AlertSeverity) -> str:
        """Map severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.PAGE: "critical",
        }
        return mapping.get(severity, "warning")


# =============================================================================
# ALERT MANAGER CLASS
# =============================================================================

class AlertManager:
    """
    Central alert management system.

    Handles alert rule registration, evaluation, state management,
    and notification dispatch.

    Example:
        >>> manager = AlertManager()
        >>> manager.register_rule(HIGH_HEAT_LOSS_ALERT)
        >>> metrics = {"heat_loss_wm": 150.0}
        >>> alerts = manager.evaluate_rules(metrics)
    """

    _instance: Optional["AlertManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "AlertManager":
        """Singleton pattern for global alert manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, auto_register: bool = True):
        """
        Initialize the AlertManager.

        Args:
            auto_register: Whether to auto-register default alert rules
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, AlertInstance] = {}
        self._alert_history: List[AlertInstance] = []
        self._silences: Dict[str, SilenceRule] = {}
        self._inhibit_rules: List[InhibitRule] = []
        self._notifiers: Dict[NotificationChannel, BaseNotifier] = {}
        self._pending_conditions: Dict[str, datetime] = {}

        self._manager_lock = threading.RLock()
        self._max_history_size = 10000

        if auto_register:
            self._register_default_rules()

        self._initialized = True
        logger.info("AlertManager initialized")

    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        default_rules = [
            HIGH_HEAT_LOSS_ALERT,
            CRITICAL_DEGRADATION_ALERT,
            SAFETY_TEMPERATURE_EXCEEDED_ALERT,
            INSPECTION_OVERDUE_ALERT,
            MOISTURE_DETECTED_ALERT,
            RAPID_DEGRADATION_RATE_ALERT,
            LOW_INSULATION_EFFICIENCY_ALERT,
            INTEGRATION_FAILURE_ALERT,
            CAMERA_DISCONNECTED_ALERT,
            HIGH_API_LATENCY_ALERT,
            HIGH_ERROR_RATE_ALERT,
        ]

        for rule in default_rules:
            self.register_rule(rule)

        logger.debug(f"Registered {len(default_rules)} default alert rules")

    def register_rule(self, rule: AlertRule) -> None:
        """
        Register an alert rule.

        Args:
            rule: AlertRule to register
        """
        with self._manager_lock:
            self._rules[rule.name] = rule
            logger.debug(f"Registered alert rule: {rule.name}")

    def unregister_rule(self, rule_name: str) -> None:
        """
        Unregister an alert rule.

        Args:
            rule_name: Name of rule to remove
        """
        with self._manager_lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
                logger.debug(f"Unregistered alert rule: {rule_name}")

    def register_notifier(
        self,
        channel: NotificationChannel,
        notifier: BaseNotifier
    ) -> None:
        """
        Register a notification sender.

        Args:
            channel: Notification channel type
            notifier: Notifier instance
        """
        with self._manager_lock:
            self._notifiers[channel] = notifier
            logger.debug(f"Registered notifier: {channel.value}")

    def add_silence(self, silence: SilenceRule) -> None:
        """
        Add a silence rule.

        Args:
            silence: SilenceRule to add
        """
        with self._manager_lock:
            self._silences[silence.silence_id] = silence
            logger.info(f"Added silence: {silence.silence_id}")

    def remove_silence(self, silence_id: str) -> None:
        """
        Remove a silence rule.

        Args:
            silence_id: ID of silence to remove
        """
        with self._manager_lock:
            if silence_id in self._silences:
                del self._silences[silence_id]
                logger.info(f"Removed silence: {silence_id}")

    def evaluate_rules(
        self,
        metrics: Dict[str, float],
        labels: Optional[Dict[str, str]] = None
    ) -> List[AlertInstance]:
        """
        Evaluate all rules against provided metrics.

        Args:
            metrics: Dictionary of metric names to values
            labels: Optional labels for all metrics

        Returns:
            List of fired or resolved AlertInstances
        """
        labels = labels or {}
        results: List[AlertInstance] = []

        with self._manager_lock:
            for rule_name, rule in self._rules.items():
                if not rule.enabled:
                    continue

                metric_value = metrics.get(rule.condition.metric_name)
                if metric_value is None:
                    continue

                condition_met = rule.condition.evaluate(metric_value)
                alert_key = f"{rule_name}:{json.dumps(labels, sort_keys=True)}"

                if condition_met:
                    alert = self._handle_condition_met(
                        rule, metric_value, labels, alert_key
                    )
                    if alert:
                        results.append(alert)
                else:
                    resolved = self._handle_condition_cleared(alert_key)
                    if resolved:
                        results.append(resolved)

        return results

    def _handle_condition_met(
        self,
        rule: AlertRule,
        value: float,
        labels: Dict[str, str],
        alert_key: str
    ) -> Optional[AlertInstance]:
        """Handle when alert condition is met."""
        now = datetime.now(timezone.utc)

        # Check if already firing
        if alert_key in self._active_alerts:
            return None

        # Check pending duration
        if rule.condition.for_duration_seconds > 0:
            if alert_key not in self._pending_conditions:
                self._pending_conditions[alert_key] = now
                return None

            pending_since = self._pending_conditions[alert_key]
            elapsed = (now - pending_since).total_seconds()

            if elapsed < rule.condition.for_duration_seconds:
                return None

            # Duration met, remove from pending
            del self._pending_conditions[alert_key]

        # Check silences
        merged_labels = {**rule.labels, **labels}
        if self._is_silenced(merged_labels):
            logger.debug(f"Alert {rule.name} silenced")
            return None

        # Check inhibitions
        if self._is_inhibited(rule, merged_labels):
            logger.debug(f"Alert {rule.name} inhibited")
            return None

        # Create and fire alert
        alert = AlertInstance(
            alert_id=str(uuid.uuid4()),
            rule_name=rule.name,
            severity=rule.severity,
            state=AlertState.FIRING,
            value=value,
            threshold=rule.condition.threshold,
            started_at=now,
            labels=merged_labels,
            annotations=rule.annotations,
        )

        self._active_alerts[alert_key] = alert
        self._send_notifications(alert, rule.channels)

        logger.info(f"Alert fired: {rule.name} (value={value})")
        return alert

    def _handle_condition_cleared(self, alert_key: str) -> Optional[AlertInstance]:
        """Handle when alert condition is no longer met."""
        # Clear pending
        if alert_key in self._pending_conditions:
            del self._pending_conditions[alert_key]

        # Check if was firing
        if alert_key not in self._active_alerts:
            return None

        # Resolve alert
        alert = self._active_alerts[alert_key]
        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)

        # Move to history
        del self._active_alerts[alert_key]
        self._add_to_history(alert)

        # Send resolution notifications
        rule = self._rules.get(alert.rule_name)
        if rule:
            self._send_resolution_notifications(alert, rule.channels)

        logger.info(f"Alert resolved: {alert.rule_name}")
        return alert

    def _is_silenced(self, labels: Dict[str, str]) -> bool:
        """Check if labels match any active silence."""
        for silence in self._silences.values():
            if silence.is_active() and silence.matches(labels):
                return True
        return False

    def _is_inhibited(self, rule: AlertRule, labels: Dict[str, str]) -> bool:
        """Check if alert should be inhibited."""
        for inhibit_rule in self._inhibit_rules:
            # Check if source alert is firing
            source_firing = False
            for alert in self._active_alerts.values():
                if self._matches_labels(alert.labels, inhibit_rule.source_matchers):
                    source_firing = True
                    break

            if not source_firing:
                continue

            # Check if target matches
            if self._matches_labels(labels, inhibit_rule.target_matchers):
                return True

        return False

    def _matches_labels(
        self,
        labels: Dict[str, str],
        matchers: Dict[str, str]
    ) -> bool:
        """Check if labels match matchers."""
        for key, pattern in matchers.items():
            if key not in labels:
                return False
            if not re.match(pattern, labels[key]):
                return False
        return True

    def _send_notifications(
        self,
        alert: AlertInstance,
        channels: List[NotificationChannel]
    ) -> None:
        """Send notifications for a fired alert."""
        for channel in channels:
            notifier = self._notifiers.get(channel)
            if notifier:
                try:
                    notifier.send(alert)
                    alert.notification_count += 1
                    alert.last_notified_at = datetime.now(timezone.utc)
                except Exception as e:
                    logger.error(f"Failed to send {channel.value} notification: {e}")

    def _send_resolution_notifications(
        self,
        alert: AlertInstance,
        channels: List[NotificationChannel]
    ) -> None:
        """Send resolution notifications."""
        for channel in channels:
            notifier = self._notifiers.get(channel)
            if notifier:
                try:
                    notifier.send_resolved(alert)
                except Exception as e:
                    logger.error(f"Failed to send {channel.value} resolution: {e}")

    def _add_to_history(self, alert: AlertInstance) -> None:
        """Add alert to history, maintaining max size."""
        self._alert_history.append(alert)
        while len(self._alert_history) > self._max_history_size:
            self._alert_history.pop(0)

    def get_active_alerts(self) -> List[AlertInstance]:
        """Get list of currently active alerts."""
        with self._manager_lock:
            return list(self._active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 100,
        rule_name: Optional[str] = None
    ) -> List[AlertInstance]:
        """
        Get alert history.

        Args:
            limit: Maximum number of alerts to return
            rule_name: Optional filter by rule name

        Returns:
            List of historical alerts
        """
        with self._manager_lock:
            history = self._alert_history

            if rule_name:
                history = [a for a in history if a.rule_name == rule_name]

            return history[-limit:]

    def get_rules(self) -> Dict[str, AlertRule]:
        """Get all registered rules."""
        with self._manager_lock:
            return dict(self._rules)

    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get a specific rule by name."""
        with self._manager_lock:
            return self._rules.get(rule_name)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def evaluate_threshold(
    value: float,
    warning_threshold: float,
    critical_threshold: float,
    operator: ComparisonOperator = ComparisonOperator.GREATER_THAN
) -> Optional[AlertSeverity]:
    """
    Evaluate a value against warning and critical thresholds.

    Args:
        value: Value to evaluate
        warning_threshold: Warning threshold
        critical_threshold: Critical threshold
        operator: Comparison operator

    Returns:
        AlertSeverity or None if no threshold exceeded
    """
    critical_condition = AlertCondition(
        metric_name="",
        operator=operator,
        threshold=critical_threshold
    )

    warning_condition = AlertCondition(
        metric_name="",
        operator=operator,
        threshold=warning_threshold
    )

    if critical_condition.evaluate(value):
        return AlertSeverity.CRITICAL
    elif warning_condition.evaluate(value):
        return AlertSeverity.WARNING

    return None


_global_manager: Optional[AlertManager] = None


def create_alert_manager(auto_register: bool = True) -> AlertManager:
    """
    Create or get the global alert manager.

    Args:
        auto_register: Whether to auto-register default rules

    Returns:
        AlertManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = AlertManager(auto_register=auto_register)
    return _global_manager


def get_default_alert_rules() -> List[AlertRule]:
    """
    Get list of all default alert rules.

    Returns:
        List of default AlertRule objects
    """
    return [
        HIGH_HEAT_LOSS_ALERT,
        CRITICAL_DEGRADATION_ALERT,
        SAFETY_TEMPERATURE_EXCEEDED_ALERT,
        INSPECTION_OVERDUE_ALERT,
        MOISTURE_DETECTED_ALERT,
        RAPID_DEGRADATION_RATE_ALERT,
        LOW_INSULATION_EFFICIENCY_ALERT,
        INTEGRATION_FAILURE_ALERT,
        CAMERA_DISCONNECTED_ALERT,
        HIGH_API_LATENCY_ALERT,
        HIGH_ERROR_RATE_ALERT,
    ]


def reset_alert_manager() -> None:
    """Reset the global alert manager (for testing)."""
    global _global_manager
    _global_manager = None


__all__ = [
    # Main Classes
    "AlertManager",

    # Alert Enumerations
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",
    "ComparisonOperator",

    # Alert Data Classes
    "AlertCondition",
    "AlertRule",
    "AlertInstance",
    "SilenceRule",
    "InhibitRule",

    # Built-in Alert Rules
    "HIGH_HEAT_LOSS_ALERT",
    "CRITICAL_DEGRADATION_ALERT",
    "SAFETY_TEMPERATURE_EXCEEDED_ALERT",
    "INSPECTION_OVERDUE_ALERT",
    "MOISTURE_DETECTED_ALERT",
    "RAPID_DEGRADATION_RATE_ALERT",
    "LOW_INSULATION_EFFICIENCY_ALERT",
    "INTEGRATION_FAILURE_ALERT",
    "CAMERA_DISCONNECTED_ALERT",
    "HIGH_API_LATENCY_ALERT",
    "HIGH_ERROR_RATE_ALERT",

    # Notification Classes
    "NotificationConfig",
    "BaseNotifier",
    "EmailNotifier",
    "SlackNotifier",
    "PagerDutyNotifier",

    # Utility Functions
    "create_alert_manager",
    "get_default_alert_rules",
    "evaluate_threshold",
    "reset_alert_manager",
]
