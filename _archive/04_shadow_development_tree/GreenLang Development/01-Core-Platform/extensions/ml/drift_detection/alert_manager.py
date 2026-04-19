"""
Drift Alert Manager - Alert Management for Drift Detection.

This module provides alert management capabilities for drift detection
in GreenLang Process Heat agents, including:
- Severity-based alerting (INFO, WARNING, CRITICAL)
- Integration hooks for PagerDuty and Slack
- Automatic model rollback triggers
- Alert throttling and cooldown

Example:
    >>> from greenlang.ml.drift_detection import DriftAlertManager, AlertSeverity
    >>> manager = DriftAlertManager()
    >>> alert = manager.create_alert(
    ...     agent_id="GL-001",
    ...     severity=AlertSeverity.WARNING,
    ...     message="Data drift detected in emission factors",
    ...     drift_score=0.25
    ... )
    >>> manager.dispatch_alert(alert)
"""

import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from collections import deque

import requests
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert notification channels."""

    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"


class AlertStatus(str, Enum):
    """Alert status."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class RemediationAction(str, Enum):
    """Automatic remediation actions."""

    NONE = "none"
    NOTIFY = "notify"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    PAUSE = "pause"


# =============================================================================
# Configuration Models
# =============================================================================

class SlackConfig(BaseModel):
    """Slack integration configuration."""

    enabled: bool = Field(default=False, description="Enable Slack notifications")
    webhook_url: Optional[str] = Field(None, description="Slack webhook URL")
    channel: Optional[str] = Field(None, description="Default channel override")
    username: str = Field(
        default="GreenLang Drift Monitor",
        description="Bot username"
    )
    icon_emoji: str = Field(default=":chart_with_upwards_trend:", description="Bot icon")

    # Severity channel mapping
    critical_channel: Optional[str] = Field(
        None, description="Channel for critical alerts"
    )
    warning_channel: Optional[str] = Field(
        None, description="Channel for warning alerts"
    )


class PagerDutyConfig(BaseModel):
    """PagerDuty integration configuration."""

    enabled: bool = Field(default=False, description="Enable PagerDuty integration")
    routing_key: Optional[str] = Field(None, description="PagerDuty routing/integration key")
    api_url: str = Field(
        default="https://events.pagerduty.com/v2/enqueue",
        description="PagerDuty events API URL"
    )

    # Service configuration
    service_name: str = Field(
        default="GreenLang Process Heat",
        description="Service name in PagerDuty"
    )

    # Severity mapping
    severity_mapping: Dict[str, str] = Field(
        default={
            "info": "info",
            "warning": "warning",
            "critical": "critical",
        },
        description="Mapping from GreenLang severity to PagerDuty severity"
    )


class EmailConfig(BaseModel):
    """Email notification configuration."""

    enabled: bool = Field(default=False, description="Enable email notifications")
    smtp_host: Optional[str] = Field(None, description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: Optional[str] = Field(None, description="SMTP username")
    smtp_password: Optional[str] = Field(None, description="SMTP password")
    from_address: Optional[str] = Field(None, description="From email address")
    to_addresses: List[str] = Field(
        default_factory=list, description="Recipient email addresses"
    )

    # Escalation emails
    critical_recipients: List[str] = Field(
        default_factory=list, description="Recipients for critical alerts"
    )


class WebhookConfig(BaseModel):
    """Custom webhook configuration."""

    enabled: bool = Field(default=False, description="Enable webhook notifications")
    url: Optional[str] = Field(None, description="Webhook URL")
    method: str = Field(default="POST", description="HTTP method")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Custom headers"
    )
    auth_token: Optional[str] = Field(None, description="Bearer token for authentication")


class AlertManagerConfig(BaseModel):
    """Configuration for the Drift Alert Manager."""

    # General settings
    storage_path: str = Field(
        default="./mlops_data/drift_alerts",
        description="Path for storing alert data"
    )

    # Throttling
    cooldown_minutes: int = Field(
        default=30, ge=1, description="Minimum minutes between same alerts"
    )
    max_alerts_per_hour: int = Field(
        default=20, ge=1, le=200, description="Maximum alerts per hour per agent"
    )

    # Auto-remediation
    auto_remediation_enabled: bool = Field(
        default=True, description="Enable automatic remediation actions"
    )
    rollback_on_critical: bool = Field(
        default=False, description="Trigger rollback on critical alerts"
    )
    retrain_on_high_drift: bool = Field(
        default=True, description="Trigger retraining on high drift"
    )

    # Escalation
    escalation_timeout_minutes: int = Field(
        default=60, ge=5, description="Minutes before unacknowledged alerts escalate"
    )

    # Integration configs
    slack: SlackConfig = Field(default_factory=SlackConfig)
    pagerduty: PagerDutyConfig = Field(default_factory=PagerDutyConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


# =============================================================================
# Alert Models
# =============================================================================

class DriftAlert(BaseModel):
    """Drift detection alert."""

    # Identification
    alert_id: str = Field(..., description="Unique alert identifier")
    agent_id: str = Field(..., description="Agent ID (GL-001 through GL-020)")
    model_name: str = Field(default="unknown", description="Model name")
    model_version: str = Field(default="unknown", description="Model version")

    # Alert details
    severity: AlertSeverity = Field(..., description="Alert severity level")
    status: AlertStatus = Field(
        default=AlertStatus.PENDING, description="Alert status"
    )
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Detailed alert message")

    # Drift information
    drift_type: str = Field(default="data", description="Type of drift detected")
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift score")
    drifted_features: List[str] = Field(
        default_factory=list, description="Features with drift"
    )

    # Recommendations and actions
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    remediation_action: RemediationAction = Field(
        default=RemediationAction.NONE, description="Automatic remediation action"
    )
    remediation_triggered: bool = Field(
        default=False, description="Whether remediation was triggered"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Alert creation time"
    )
    dispatched_at: Optional[datetime] = Field(
        None, description="When alert was dispatched"
    )
    acknowledged_at: Optional[datetime] = Field(
        None, description="When alert was acknowledged"
    )
    acknowledged_by: Optional[str] = Field(
        None, description="Who acknowledged the alert"
    )
    resolved_at: Optional[datetime] = Field(
        None, description="When alert was resolved"
    )

    # Tracking
    dispatch_channels: List[AlertChannel] = Field(
        default_factory=list, description="Channels where alert was sent"
    )
    retry_count: int = Field(default=0, description="Number of dispatch retries")

    # Provenance
    report_id: Optional[str] = Field(
        None, description="Associated drift report ID"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        alert_dict = self.model_dump(exclude={"provenance_hash"})
        alert_str = json.dumps(alert_dict, sort_keys=True, default=str)
        return hashlib.sha256(alert_str.encode()).hexdigest()

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Alert Handlers (Abstract Base)
# =============================================================================

class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @abstractmethod
    def send(self, alert: DriftAlert) -> bool:
        """
        Send an alert.

        Args:
            alert: The alert to send.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the alert service.

        Returns:
            True if connection is successful.
        """
        pass


class SlackHandler(AlertHandler):
    """Slack alert handler."""

    def __init__(self, config: SlackConfig):
        """Initialize Slack handler."""
        self.config = config

    def send(self, alert: DriftAlert) -> bool:
        """Send alert to Slack."""
        if not self.config.enabled or not self.config.webhook_url:
            return False

        try:
            # Determine channel based on severity
            channel = self.config.channel
            if alert.severity == AlertSeverity.CRITICAL and self.config.critical_channel:
                channel = self.config.critical_channel
            elif alert.severity == AlertSeverity.WARNING and self.config.warning_channel:
                channel = self.config.warning_channel

            # Build Slack message
            color = self._severity_to_color(alert.severity)

            payload = {
                "username": self.config.username,
                "icon_emoji": self.config.icon_emoji,
                "attachments": [
                    {
                        "color": color,
                        "title": f":warning: {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Agent",
                                "value": alert.agent_id,
                                "short": True,
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True,
                            },
                            {
                                "title": "Drift Score",
                                "value": f"{alert.drift_score:.3f}",
                                "short": True,
                            },
                            {
                                "title": "Drift Type",
                                "value": alert.drift_type,
                                "short": True,
                            },
                        ],
                        "footer": f"Alert ID: {alert.alert_id}",
                        "ts": int(alert.created_at.timestamp()),
                    }
                ],
            }

            if channel:
                payload["channel"] = channel

            if alert.drifted_features:
                payload["attachments"][0]["fields"].append({
                    "title": "Drifted Features",
                    "value": ", ".join(alert.drifted_features[:5]),
                    "short": False,
                })

            if alert.recommendations:
                payload["attachments"][0]["fields"].append({
                    "title": "Recommendations",
                    "value": "\n".join(f"- {r}" for r in alert.recommendations[:3]),
                    "short": False,
                })

            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"Slack alert sent for {alert.agent_id}: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _severity_to_color(self, severity: AlertSeverity) -> str:
        """Map severity to Slack attachment color."""
        return {
            AlertSeverity.INFO: "#2196F3",      # Blue
            AlertSeverity.WARNING: "#FF9800",    # Orange
            AlertSeverity.CRITICAL: "#F44336",   # Red
        }.get(severity, "#9E9E9E")

    def test_connection(self) -> bool:
        """Test Slack webhook connection."""
        if not self.config.enabled or not self.config.webhook_url:
            return False

        try:
            response = requests.post(
                self.config.webhook_url,
                json={
                    "text": "GreenLang Drift Monitor connection test",
                    "username": self.config.username,
                },
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False


class PagerDutyHandler(AlertHandler):
    """PagerDuty alert handler."""

    def __init__(self, config: PagerDutyConfig):
        """Initialize PagerDuty handler."""
        self.config = config

    def send(self, alert: DriftAlert) -> bool:
        """Send alert to PagerDuty."""
        if not self.config.enabled or not self.config.routing_key:
            return False

        try:
            # Map severity
            pd_severity = self.config.severity_mapping.get(
                alert.severity.value, "warning"
            )

            # Build PagerDuty event
            payload = {
                "routing_key": self.config.routing_key,
                "event_action": "trigger",
                "dedup_key": f"{alert.agent_id}_{alert.drift_type}_{alert.alert_id}",
                "payload": {
                    "summary": alert.title,
                    "severity": pd_severity,
                    "source": f"{self.config.service_name} - {alert.agent_id}",
                    "component": alert.agent_id,
                    "group": "drift_detection",
                    "class": alert.drift_type,
                    "custom_details": {
                        "agent_id": alert.agent_id,
                        "model_name": alert.model_name,
                        "model_version": alert.model_version,
                        "drift_score": alert.drift_score,
                        "drift_type": alert.drift_type,
                        "drifted_features": alert.drifted_features[:10],
                        "message": alert.message,
                        "recommendations": alert.recommendations[:5],
                        "alert_id": alert.alert_id,
                    },
                },
                "links": [],
                "images": [],
            }

            response = requests.post(
                self.config.api_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"PagerDuty alert sent for {alert.agent_id}: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test PagerDuty connection."""
        # PagerDuty doesn't have a simple test endpoint
        return self.config.enabled and bool(self.config.routing_key)


class WebhookHandler(AlertHandler):
    """Custom webhook alert handler."""

    def __init__(self, config: WebhookConfig):
        """Initialize webhook handler."""
        self.config = config

    def send(self, alert: DriftAlert) -> bool:
        """Send alert to webhook."""
        if not self.config.enabled or not self.config.url:
            return False

        try:
            headers = dict(self.config.headers)
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
            headers["Content-Type"] = "application/json"

            payload = alert.model_dump(mode="json")

            response = requests.request(
                method=self.config.method,
                url=self.config.url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"Webhook alert sent for {alert.agent_id}: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test webhook connection."""
        if not self.config.enabled or not self.config.url:
            return False

        try:
            response = requests.request(
                method="HEAD" if self.config.method == "POST" else self.config.method,
                url=self.config.url,
                timeout=5,
            )
            return response.status_code < 400
        except Exception:
            return False


class LogHandler(AlertHandler):
    """Log-based alert handler (always enabled)."""

    def send(self, alert: DriftAlert) -> bool:
        """Log the alert."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
        }.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            f"[DRIFT ALERT] {alert.severity.value.upper()} - {alert.agent_id}: "
            f"{alert.title} (score={alert.drift_score:.3f})"
        )
        return True

    def test_connection(self) -> bool:
        """Log handler is always available."""
        return True


# =============================================================================
# Drift Alert Manager
# =============================================================================

class DriftAlertManager:
    """
    Drift Alert Manager for GreenLang Process Heat agents.

    This class manages drift detection alerts, including:
    - Creating and dispatching alerts
    - Throttling and cooldown
    - Integration with Slack, PagerDuty, and webhooks
    - Automatic remediation triggers
    - Alert lifecycle management

    Attributes:
        config: Alert manager configuration
        handlers: Dictionary of alert handlers by channel

    Example:
        >>> manager = DriftAlertManager()
        >>> alert = manager.create_alert(
        ...     agent_id="GL-001",
        ...     severity=AlertSeverity.WARNING,
        ...     message="Data drift detected",
        ...     drift_score=0.25
        ... )
        >>> manager.dispatch_alert(alert)
    """

    def __init__(self, config: Optional[AlertManagerConfig] = None):
        """
        Initialize DriftAlertManager.

        Args:
            config: Alert manager configuration. If None, uses defaults.
        """
        self.config = config or AlertManagerConfig()
        self._lock = threading.RLock()

        # Storage
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Alert tracking
        self._active_alerts: Dict[str, List[DriftAlert]] = {}
        self._alert_history: Dict[str, deque] = {}  # For throttling
        self._cooldowns: Dict[str, datetime] = {}  # Cooldown tracking
        self._max_history = 1000

        # Initialize handlers
        self.handlers: Dict[AlertChannel, AlertHandler] = {
            AlertChannel.LOG: LogHandler(),
            AlertChannel.SLACK: SlackHandler(self.config.slack),
            AlertChannel.PAGERDUTY: PagerDutyHandler(self.config.pagerduty),
            AlertChannel.WEBHOOK: WebhookHandler(self.config.webhook),
        }

        # Remediation callbacks
        self._remediation_callbacks: Dict[RemediationAction, List[Callable]] = {
            RemediationAction.RETRAIN: [],
            RemediationAction.ROLLBACK: [],
            RemediationAction.PAUSE: [],
        }

        logger.info(
            f"DriftAlertManager initialized with storage at {self.storage_path}"
        )

    def _generate_alert_id(self, agent_id: str) -> str:
        """Generate unique alert identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        unique_str = f"{agent_id}_{timestamp}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    def create_alert(
        self,
        agent_id: str,
        severity: AlertSeverity,
        message: str,
        drift_score: float,
        title: Optional[str] = None,
        drift_type: str = "data",
        drifted_features: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        model_name: str = "unknown",
        model_version: str = "unknown",
        report_id: Optional[str] = None,
    ) -> DriftAlert:
        """
        Create a new drift alert.

        Args:
            agent_id: Agent identifier (GL-001 through GL-020).
            severity: Alert severity level.
            message: Detailed alert message.
            drift_score: Drift score (0-1).
            title: Optional alert title.
            drift_type: Type of drift detected.
            drifted_features: List of features with drift.
            recommendations: Recommended actions.
            model_name: Model name.
            model_version: Model version.
            report_id: Associated drift report ID.

        Returns:
            Created DriftAlert.
        """
        if title is None:
            title = f"{drift_type.title()} Drift Detected in {agent_id}"

        # Determine remediation action
        remediation_action = RemediationAction.NONE
        if self.config.auto_remediation_enabled:
            if severity == AlertSeverity.CRITICAL:
                if self.config.rollback_on_critical:
                    remediation_action = RemediationAction.ROLLBACK
                else:
                    remediation_action = RemediationAction.RETRAIN
            elif severity == AlertSeverity.WARNING and self.config.retrain_on_high_drift:
                if drift_score > 0.4:
                    remediation_action = RemediationAction.RETRAIN

        alert = DriftAlert(
            alert_id=self._generate_alert_id(agent_id),
            agent_id=agent_id,
            model_name=model_name,
            model_version=model_version,
            severity=severity,
            title=title,
            message=message,
            drift_type=drift_type,
            drift_score=drift_score,
            drifted_features=drifted_features or [],
            recommendations=recommendations or [],
            remediation_action=remediation_action,
            report_id=report_id,
        )

        # Calculate provenance hash
        alert.provenance_hash = alert.calculate_provenance_hash()

        logger.info(
            f"Created alert {alert.alert_id} for {agent_id}: "
            f"severity={severity.value}, score={drift_score:.3f}"
        )

        return alert

    def dispatch_alert(
        self,
        alert: DriftAlert,
        channels: Optional[List[AlertChannel]] = None,
    ) -> bool:
        """
        Dispatch an alert to notification channels.

        Args:
            alert: The alert to dispatch.
            channels: Channels to dispatch to. If None, uses severity defaults.

        Returns:
            True if at least one channel succeeded.
        """
        # Check throttling
        if self._is_throttled(alert):
            logger.info(
                f"Alert throttled for {alert.agent_id}: {alert.alert_id}"
            )
            return False

        # Determine channels
        if channels is None:
            channels = self._get_channels_for_severity(alert.severity)

        # Always include log
        if AlertChannel.LOG not in channels:
            channels.append(AlertChannel.LOG)

        # Dispatch to channels
        successful_channels = []
        for channel in channels:
            handler = self.handlers.get(channel)
            if handler:
                try:
                    if handler.send(alert):
                        successful_channels.append(channel)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel.value}: {e}")

        # Update alert
        if successful_channels:
            alert.status = AlertStatus.DISPATCHED
            alert.dispatched_at = datetime.utcnow()
            alert.dispatch_channels = successful_channels

            # Save alert
            self._save_alert(alert)

            # Track for throttling
            self._track_alert(alert)

            # Trigger remediation if needed
            if alert.remediation_action != RemediationAction.NONE:
                self._trigger_remediation(alert)

            return True

        return False

    def _is_throttled(self, alert: DriftAlert) -> bool:
        """Check if alert should be throttled."""
        with self._lock:
            # Check cooldown
            cooldown_key = f"{alert.agent_id}_{alert.drift_type}"
            if cooldown_key in self._cooldowns:
                last_alert_time = self._cooldowns[cooldown_key]
                cooldown_delta = timedelta(minutes=self.config.cooldown_minutes)
                if datetime.utcnow() - last_alert_time < cooldown_delta:
                    return True

            # Check rate limit
            agent_history = self._alert_history.get(alert.agent_id, deque())
            cutoff = datetime.utcnow() - timedelta(hours=1)
            recent_count = sum(
                1 for ts in agent_history if ts > cutoff
            )

            if recent_count >= self.config.max_alerts_per_hour:
                return True

        return False

    def _track_alert(self, alert: DriftAlert) -> None:
        """Track alert for throttling."""
        with self._lock:
            # Update cooldown
            cooldown_key = f"{alert.agent_id}_{alert.drift_type}"
            self._cooldowns[cooldown_key] = datetime.utcnow()

            # Update history
            if alert.agent_id not in self._alert_history:
                self._alert_history[alert.agent_id] = deque(maxlen=self._max_history)
            self._alert_history[alert.agent_id].append(datetime.utcnow())

            # Track active alerts
            if alert.agent_id not in self._active_alerts:
                self._active_alerts[alert.agent_id] = []
            self._active_alerts[alert.agent_id].append(alert)

    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Get default channels for a severity level."""
        channels = [AlertChannel.LOG]

        if severity == AlertSeverity.CRITICAL:
            if self.config.slack.enabled:
                channels.append(AlertChannel.SLACK)
            if self.config.pagerduty.enabled:
                channels.append(AlertChannel.PAGERDUTY)
            if self.config.webhook.enabled:
                channels.append(AlertChannel.WEBHOOK)

        elif severity == AlertSeverity.WARNING:
            if self.config.slack.enabled:
                channels.append(AlertChannel.SLACK)
            if self.config.webhook.enabled:
                channels.append(AlertChannel.WEBHOOK)

        return channels

    def _save_alert(self, alert: DriftAlert) -> None:
        """Save alert to storage."""
        alert_path = self.storage_path / f"alert_{alert.alert_id}.json"
        with open(alert_path, "w") as f:
            f.write(alert.model_dump_json(indent=2))

    def _trigger_remediation(self, alert: DriftAlert) -> None:
        """Trigger automatic remediation action."""
        if alert.remediation_action == RemediationAction.NONE:
            return

        logger.info(
            f"Triggering remediation {alert.remediation_action.value} "
            f"for {alert.agent_id}"
        )

        # Call registered callbacks
        callbacks = self._remediation_callbacks.get(alert.remediation_action, [])
        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Remediation callback failed: {e}")

        alert.remediation_triggered = True
        self._save_alert(alert)

    def register_remediation_callback(
        self,
        action: RemediationAction,
        callback: Callable[[DriftAlert], None],
    ) -> None:
        """
        Register a callback for automatic remediation.

        Args:
            action: Remediation action to register for.
            callback: Callback function taking DriftAlert.
        """
        if action not in self._remediation_callbacks:
            self._remediation_callbacks[action] = []
        self._remediation_callbacks[action].append(callback)

        logger.info(f"Registered remediation callback for {action.value}")

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier.
            acknowledged_by: User or system acknowledging.

        Returns:
            True if alert was found and acknowledged.
        """
        for agent_alerts in self._active_alerts.values():
            for alert in agent_alerts:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.utcnow()
                    alert.acknowledged_by = acknowledged_by
                    self._save_alert(alert)

                    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                    return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            True if alert was found and resolved.
        """
        for agent_id, agent_alerts in self._active_alerts.items():
            for i, alert in enumerate(agent_alerts):
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()
                    self._save_alert(alert)

                    # Remove from active
                    self._active_alerts[agent_id].pop(i)

                    logger.info(f"Alert {alert_id} resolved")
                    return True

        return False

    def get_active_alerts(
        self,
        agent_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[DriftAlert]:
        """
        Get active (unresolved) alerts.

        Args:
            agent_id: Filter by agent ID.
            severity: Filter by severity.

        Returns:
            List of active alerts.
        """
        alerts = []

        with self._lock:
            if agent_id:
                alerts = list(self._active_alerts.get(agent_id, []))
            else:
                for agent_alerts in self._active_alerts.values():
                    alerts.extend(agent_alerts)

        # Filter by severity
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Filter resolved
        alerts = [
            a for a in alerts
            if a.status not in [AlertStatus.RESOLVED]
        ]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_statistics(
        self,
        agent_id: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get alert statistics.

        Args:
            agent_id: Filter by agent ID.
            hours: Time window in hours.

        Returns:
            Dictionary with alert statistics.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Collect alerts from history
        with self._lock:
            if agent_id:
                history = list(self._alert_history.get(agent_id, []))
            else:
                history = []
                for agent_history in self._alert_history.values():
                    history.extend(agent_history)

        recent = [ts for ts in history if ts > cutoff]

        active = self.get_active_alerts(agent_id=agent_id)

        severity_counts = {s.value: 0 for s in AlertSeverity}
        for alert in active:
            severity_counts[alert.severity.value] += 1

        return {
            "agent_id": agent_id or "all",
            "hours": hours,
            "total_alerts": len(recent),
            "active_alerts": len(active),
            "severity_distribution": severity_counts,
            "critical_count": severity_counts.get("critical", 0),
            "warning_count": severity_counts.get("warning", 0),
            "info_count": severity_counts.get("info", 0),
            "alerts_per_hour": len(recent) / max(hours, 1),
        }

    def test_integrations(self) -> Dict[str, bool]:
        """
        Test all integration connections.

        Returns:
            Dictionary mapping channel to connection status.
        """
        results = {}
        for channel, handler in self.handlers.items():
            try:
                results[channel.value] = handler.test_connection()
            except Exception:
                results[channel.value] = False

        return results

    def cleanup_old_alerts(self, days: int = 30) -> int:
        """
        Clean up old alert files.

        Args:
            days: Days to retain.

        Returns:
            Number of alerts cleaned up.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        cleaned = 0

        for alert_file in self.storage_path.glob("alert_*.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(alert_file.stat().st_mtime)
                if mtime < cutoff:
                    alert_file.unlink()
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean up {alert_file}: {e}")

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old alert files")

        return cleaned
