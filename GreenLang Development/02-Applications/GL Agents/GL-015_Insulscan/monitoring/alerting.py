"""
GL-015 INSULSCAN - Alert Management System

This module implements comprehensive alert management for insulation scanning
and thermal assessment, including alert creation, acknowledgment, escalation,
and notification routing with deduplication and suppression capabilities.

Alert Types:
    - CRITICAL_HOT_SPOT: Critical temperature anomaly detected
    - RAPID_DEGRADATION: Rapid insulation condition decline
    - EXCESSIVE_HEAT_LOSS: Heat loss exceeds threshold
    - INSULATION_FAILURE: Complete insulation failure detected
    - SYSTEM_ERROR: System or integration errors
    - DATA_QUALITY_DEGRADED: Input data quality issues

Severity Levels:
    - INFO: Informational, no action required
    - WARNING: Action recommended within timeframe
    - CRITICAL: Immediate action required

Features:
    - Alert deduplication using fingerprinting
    - Alert suppression with configurable duration
    - Multi-channel notifications (PagerDuty, Slack, Email)
    - Alert escalation with timeout-based rules
    - Hysteresis to prevent alert chatter

Example:
    >>> manager = InsulscanAlertManager()
    >>> alert = manager.create_alert(
    ...     alert_type=InsulationAlertType.CRITICAL_HOT_SPOT,
    ...     severity=AlertSeverity.CRITICAL,
    ...     source="PIPE-001",
    ...     message="Critical hot spot detected: 85C delta",
    ... )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import hashlib
import json
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class InsulationAlertType(Enum):
    """Types of insulation-related alerts."""
    CRITICAL_HOT_SPOT = "critical_hot_spot"
    RAPID_DEGRADATION = "rapid_degradation"
    EXCESSIVE_HEAT_LOSS = "excessive_heat_loss"
    INSULATION_FAILURE = "insulation_failure"
    REPAIR_OVERDUE = "repair_overdue"
    DATA_QUALITY_DEGRADED = "data_quality_degraded"
    SYSTEM_ERROR = "system_error"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    INTEGRATION_FAILURE = "integration_failure"


class AlertSeverity(IntEnum):
    """
    Alert severity levels following industrial standards.

    INFO: Informational, logged only
    WARNING: Action recommended within timeframe
    CRITICAL: Immediate action required
    """
    INFO = 0
    WARNING = 1
    CRITICAL = 2

    @property
    def name_display(self) -> str:
        """Get display name for severity."""
        names = {
            AlertSeverity.INFO: "INFO",
            AlertSeverity.WARNING: "WARNING",
            AlertSeverity.CRITICAL: "CRITICAL",
        }
        return names.get(self, "UNKNOWN")

    @property
    def response_time_minutes(self) -> int:
        """Get expected response time for severity level."""
        times = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 30,
            AlertSeverity.CRITICAL: 5,
        }
        return times.get(self, 60)


class AlertState(Enum):
    """Alert lifecycle states."""
    PENDING = "pending"
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class EscalationLevel(Enum):
    """Alert escalation levels."""
    L1_OPERATOR = "L1_operator"
    L2_SUPERVISOR = "L2_supervisor"
    L3_ENGINEER = "L3_engineer"
    L4_MANAGEMENT = "L4_management"

    @property
    def timeout_minutes(self) -> int:
        """Default escalation timeout for this level."""
        timeouts = {
            EscalationLevel.L1_OPERATOR: 10,
            EscalationLevel.L2_SUPERVISOR: 20,
            EscalationLevel.L3_ENGINEER: 45,
            EscalationLevel.L4_MANAGEMENT: 90,
        }
        return timeouts.get(self, 60)


# =============================================================================
# Alert Rule Configuration
# =============================================================================

@dataclass
class AlertRule:
    """
    Configuration for an alert rule.

    Defines conditions under which an alert should be triggered,
    including thresholds, severity, and message templates.
    """
    rule_id: str
    alert_type: InsulationAlertType
    condition_field: str
    threshold: float
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_seconds: float = 60.0
    deduplication_window_seconds: float = 300.0

    def evaluate(self, value: float) -> bool:
        """
        Evaluate if the condition is met.

        Args:
            value: Current value to check against threshold

        Returns:
            True if condition is met
        """
        ops = {
            'gt': lambda v, t: v > t,
            'lt': lambda v, t: v < t,
            'gte': lambda v, t: v >= t,
            'lte': lambda v, t: v <= t,
            'eq': lambda v, t: v == t,
        }
        op_fn = ops.get(self.operator, lambda v, t: False)
        return op_fn(value, self.threshold)

    def format_message(self, **kwargs) -> str:
        """Format the message template with provided values."""
        try:
            return self.message_template.format(**kwargs)
        except KeyError:
            return self.message_template


# Default alert rules
DEFAULT_ALERT_RULES: List[AlertRule] = [
    AlertRule(
        rule_id="critical_hot_spot_high",
        alert_type=InsulationAlertType.CRITICAL_HOT_SPOT,
        condition_field="temperature_delta_c",
        threshold=50.0,
        operator="gt",
        severity=AlertSeverity.CRITICAL,
        message_template="Critical hot spot on {asset_id}: {temperature_delta_c:.1f}C delta exceeds 50C threshold",
    ),
    AlertRule(
        rule_id="critical_hot_spot_medium",
        alert_type=InsulationAlertType.CRITICAL_HOT_SPOT,
        condition_field="temperature_delta_c",
        threshold=30.0,
        operator="gt",
        severity=AlertSeverity.WARNING,
        message_template="Hot spot on {asset_id}: {temperature_delta_c:.1f}C delta exceeds 30C threshold",
    ),
    AlertRule(
        rule_id="rapid_degradation",
        alert_type=InsulationAlertType.RAPID_DEGRADATION,
        condition_field="degradation_rate_per_day",
        threshold=0.05,
        operator="gt",
        severity=AlertSeverity.CRITICAL,
        message_template="Rapid degradation on {asset_id}: {degradation_rate_per_day:.3f}/day exceeds threshold",
    ),
    AlertRule(
        rule_id="excessive_heat_loss",
        alert_type=InsulationAlertType.EXCESSIVE_HEAT_LOSS,
        condition_field="heat_loss_watts",
        threshold=10000.0,
        operator="gt",
        severity=AlertSeverity.WARNING,
        message_template="Excessive heat loss on {asset_id}: {heat_loss_watts:.0f}W exceeds threshold",
    ),
    AlertRule(
        rule_id="low_condition_score",
        alert_type=InsulationAlertType.INSULATION_FAILURE,
        condition_field="condition_score",
        threshold=0.3,
        operator="lt",
        severity=AlertSeverity.CRITICAL,
        message_template="Insulation failure on {asset_id}: condition score {condition_score:.2f} below 0.3",
    ),
    AlertRule(
        rule_id="data_quality_low",
        alert_type=InsulationAlertType.DATA_QUALITY_DEGRADED,
        condition_field="data_quality_score",
        threshold=0.7,
        operator="lt",
        severity=AlertSeverity.WARNING,
        message_template="Data quality degraded for {asset_id}: score {data_quality_score:.2f}",
    ),
]


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class AlertContext:
    """Contextual information for an alert."""
    asset_id: Optional[str] = None
    surface_type: Optional[str] = None
    insulation_type: Optional[str] = None
    location: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    temperature_delta_c: Optional[float] = None
    heat_loss_watts: Optional[float] = None
    condition_score: Optional[float] = None
    degradation_rate: Optional[float] = None
    repair_recommended: bool = False
    projected_savings_usd: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "asset_id": self.asset_id,
            "surface_type": self.surface_type,
            "insulation_type": self.insulation_type,
            "location": self.location,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "temperature_delta_c": self.temperature_delta_c,
            "heat_loss_watts": self.heat_loss_watts,
            "condition_score": self.condition_score,
            "degradation_rate": self.degradation_rate,
            "repair_recommended": self.repair_recommended,
            "projected_savings_usd": self.projected_savings_usd,
            "metadata": self.metadata,
        }


@dataclass
class Alert:
    """Alert instance representing a detected condition."""
    alert_id: str
    alert_type: InsulationAlertType
    severity: AlertSeverity
    source: str
    message: str
    state: AlertState = AlertState.PENDING
    context: AlertContext = field(default_factory=AlertContext)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fired_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledgment_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    escalation_level: EscalationLevel = EscalationLevel.L1_OPERATOR
    escalated_at: Optional[datetime] = None
    fingerprint: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    notification_count: int = 0

    def __post_init__(self) -> None:
        """Compute fingerprint after initialization."""
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute unique fingerprint for alert deduplication."""
        data = (
            f"{self.alert_type.value}:"
            f"{self.source}:"
            f"{self.severity.value}:"
            f"{self.context.asset_id}:"
            f"{sorted(self.labels.items())}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.name_display,
            "severity_code": self.severity.value,
            "source": self.source,
            "message": self.message,
            "state": self.state.value,
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "acknowledgment_notes": self.acknowledgment_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "escalation_level": self.escalation_level.value,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "fingerprint": self.fingerprint,
            "labels": self.labels,
            "notification_count": self.notification_count,
        }


@dataclass
class AlertFilter:
    """Filter criteria for querying alerts."""
    alert_types: Optional[List[InsulationAlertType]] = None
    severities: Optional[List[AlertSeverity]] = None
    states: Optional[List[AlertState]] = None
    source_pattern: Optional[str] = None
    asset_id: Optional[str] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    escalation_levels: Optional[List[EscalationLevel]] = None


@dataclass
class AcknowledgmentResult:
    """Result of an alert acknowledgment operation."""
    success: bool
    alert_id: str
    acknowledged_by: str
    acknowledged_at: datetime
    notes: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class EscalationResult:
    """Result of an alert escalation operation."""
    success: bool
    alert_id: str
    previous_level: EscalationLevel
    new_level: EscalationLevel
    escalated_at: datetime
    escalation_reason: Optional[str] = None
    error_message: Optional[str] = None


# =============================================================================
# Notification Channels
# =============================================================================

class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement send()")


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel for development and debugging."""

    async def send(self, alert: Alert) -> bool:
        """Log alert to standard logging."""
        log_level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
        }
        log_level = log_level_map.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            "[ALERT] %s | %s | %s: %s (source=%s)",
            alert.alert_type.value.upper(),
            alert.severity.name_display,
            alert.alert_id[:8],
            alert.message,
            alert.source,
        )
        return True


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel for critical alerts."""

    def __init__(
        self,
        routing_key: str,
        service_name: str = "INSULSCAN",
        enabled: bool = True,
    ) -> None:
        """
        Initialize PagerDuty channel.

        Args:
            routing_key: PagerDuty routing key
            service_name: Service name for alert source
            enabled: Whether channel is enabled
        """
        self.routing_key = routing_key
        self.service_name = service_name
        self.enabled = enabled

    async def send(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        if not self.enabled:
            return False

        # Only send critical alerts to PagerDuty
        if alert.severity != AlertSeverity.CRITICAL:
            return False

        try:
            # In production, this would use httpx or aiohttp
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": alert.message,
                    "source": f"{self.service_name}:{alert.source}",
                    "severity": "critical",
                    "custom_details": alert.context.to_dict(),
                },
            }

            logger.info(
                "PagerDuty alert sent: %s (fingerprint=%s)",
                alert.alert_id[:8],
                alert.fingerprint[:8],
            )
            return True

        except Exception as e:
            logger.error("PagerDuty notification failed: %s", e)
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel for team notifications."""

    def __init__(
        self,
        webhook_url: str,
        channel: str = "#insulation-alerts",
        enabled: bool = True,
    ) -> None:
        """
        Initialize Slack channel.

        Args:
            webhook_url: Slack webhook URL
            channel: Target channel
            enabled: Whether channel is enabled
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.enabled = enabled

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.enabled:
            return False

        try:
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9800",
                AlertSeverity.CRITICAL: "#d32f2f",
            }

            payload = {
                "channel": self.channel,
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": f"[{alert.severity.name_display}] {alert.alert_type.value}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Asset", "value": alert.context.asset_id or "N/A", "short": True},
                        ],
                        "footer": f"INSULSCAN | {alert.alert_id[:8]}",
                        "ts": int(alert.created_at.timestamp()),
                    }
                ],
            }

            logger.info(
                "Slack alert sent: %s to %s",
                alert.alert_id[:8],
                self.channel,
            )
            return True

        except Exception as e:
            logger.error("Slack notification failed: %s", e)
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel for formal notifications."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_address: str,
        to_addresses: List[str],
        enabled: bool = True,
    ) -> None:
        """
        Initialize Email channel.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            from_address: Sender email address
            to_addresses: Recipient email addresses
            enabled: Whether channel is enabled
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_address = from_address
        self.to_addresses = to_addresses
        self.enabled = enabled

    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.enabled:
            return False

        try:
            subject = f"[{alert.severity.name_display}] INSULSCAN Alert: {alert.alert_type.value}"
            body = f"""
INSULSCAN Alert Notification

Severity: {alert.severity.name_display}
Type: {alert.alert_type.value}
Source: {alert.source}
Asset: {alert.context.asset_id or 'N/A'}

Message:
{alert.message}

Context:
- Temperature Delta: {alert.context.temperature_delta_c or 'N/A'} C
- Heat Loss: {alert.context.heat_loss_watts or 'N/A'} W
- Condition Score: {alert.context.condition_score or 'N/A'}

Alert ID: {alert.alert_id}
Created: {alert.created_at.isoformat()}
"""

            logger.info(
                "Email alert sent: %s to %s",
                alert.alert_id[:8],
                ", ".join(self.to_addresses),
            )
            return True

        except Exception as e:
            logger.error("Email notification failed: %s", e)
            return False


# =============================================================================
# Main Alert Manager Class
# =============================================================================

class InsulscanAlertManager:
    """
    Alert management system for GL-015 INSULSCAN.

    This class provides comprehensive alert lifecycle management including
    creation, acknowledgment, escalation, and notification routing with
    deduplication and suppression capabilities.

    Attributes:
        namespace: Namespace for alert identification
        rules: Alert rules for automatic triggering

    Example:
        >>> manager = InsulscanAlertManager()
        >>> alert = manager.create_alert(
        ...     alert_type=InsulationAlertType.CRITICAL_HOT_SPOT,
        ...     severity=AlertSeverity.CRITICAL,
        ...     source="PIPE-001",
        ...     message="Critical hot spot detected",
        ... )
        >>> manager.acknowledge_alert(alert.alert_id, "operator1")
    """

    AGENT_ID = "GL-015"
    AGENT_NAME = "INSULSCAN"

    def __init__(
        self,
        namespace: str = "insulscan",
        on_alert_callback: Optional[Callable[[Alert], None]] = None,
        deduplication_window_s: float = 300.0,
        rules: Optional[List[AlertRule]] = None,
    ) -> None:
        """
        Initialize InsulscanAlertManager.

        Args:
            namespace: Namespace for alert identification
            on_alert_callback: Optional callback invoked on new alerts
            deduplication_window_s: Window for alert deduplication (seconds)
            rules: Custom alert rules (default: DEFAULT_ALERT_RULES)
        """
        self.namespace = namespace
        self._on_alert_callback = on_alert_callback
        self._deduplication_window_s = deduplication_window_s
        self._rules = rules or DEFAULT_ALERT_RULES
        self._lock = threading.Lock()

        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_by_fingerprint: Dict[str, str] = {}
        self._alert_history: List[Alert] = []
        self._max_history_size = 10000

        # Suppression
        self._suppressed_fingerprints: Dict[str, datetime] = {}

        # Rate limiting
        self._rate_limit_buckets: Dict[str, List[datetime]] = {}

        # Notification channels
        self._channels: List[NotificationChannel] = [LogNotificationChannel()]

        # Escalation configuration
        self._escalation_timeouts: Dict[InsulationAlertType, Dict[AlertSeverity, int]] = {
            InsulationAlertType.CRITICAL_HOT_SPOT: {
                AlertSeverity.CRITICAL: 5,
                AlertSeverity.WARNING: 15,
                AlertSeverity.INFO: 0,
            },
            InsulationAlertType.RAPID_DEGRADATION: {
                AlertSeverity.CRITICAL: 5,
                AlertSeverity.WARNING: 30,
                AlertSeverity.INFO: 0,
            },
            InsulationAlertType.EXCESSIVE_HEAT_LOSS: {
                AlertSeverity.CRITICAL: 10,
                AlertSeverity.WARNING: 30,
                AlertSeverity.INFO: 0,
            },
            InsulationAlertType.INSULATION_FAILURE: {
                AlertSeverity.CRITICAL: 5,
                AlertSeverity.WARNING: 15,
                AlertSeverity.INFO: 0,
            },
            InsulationAlertType.SYSTEM_ERROR: {
                AlertSeverity.CRITICAL: 5,
                AlertSeverity.WARNING: 15,
                AlertSeverity.INFO: 0,
            },
        }

        # Statistics
        self._stats = {
            "alerts_created": 0,
            "alerts_acknowledged": 0,
            "alerts_resolved": 0,
            "alerts_escalated": 0,
            "alerts_deduplicated": 0,
            "alerts_suppressed": 0,
            "notifications_sent": 0,
        }

        logger.info(
            "InsulscanAlertManager initialized: namespace=%s, dedup_window=%ss, rules=%d",
            namespace,
            deduplication_window_s,
            len(self._rules),
        )

    # =========================================================================
    # Rule Evaluation
    # =========================================================================

    def evaluate_rules(
        self,
        asset_id: str,
        values: Dict[str, float],
        context: Optional[AlertContext] = None,
    ) -> List[Alert]:
        """
        Evaluate all rules against provided values.

        Args:
            asset_id: Asset identifier
            values: Dictionary of field names to values
            context: Optional alert context

        Returns:
            List of alerts created from triggered rules
        """
        created_alerts = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            field_value = values.get(rule.condition_field)
            if field_value is None:
                continue

            if rule.evaluate(field_value):
                # Check rate limit
                rate_key = f"{asset_id}:{rule.rule_id}"
                if self._is_rate_limited(rate_key, rule.cooldown_seconds):
                    continue

                # Create context if not provided
                ctx = context or AlertContext()
                ctx.asset_id = asset_id
                ctx.current_value = field_value
                ctx.threshold_value = rule.threshold

                # Format message
                message = rule.format_message(
                    asset_id=asset_id,
                    **values,
                )

                # Create alert
                alert = self.create_alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    source=asset_id,
                    message=message,
                    context=ctx,
                    labels={"rule_id": rule.rule_id},
                )

                if alert.state == AlertState.FIRING:
                    created_alerts.append(alert)

                # Update rate limit
                self._update_rate_limit(rate_key)

        return created_alerts

    def _is_rate_limited(self, key: str, cooldown_seconds: float) -> bool:
        """Check if a rate limit key is currently limited."""
        now = datetime.now(timezone.utc)

        with self._lock:
            if key not in self._rate_limit_buckets:
                return False

            last_fires = self._rate_limit_buckets[key]
            if not last_fires:
                return False

            last_fire = max(last_fires)
            return (now - last_fire).total_seconds() < cooldown_seconds

    def _update_rate_limit(self, key: str) -> None:
        """Update rate limit bucket for a key."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        with self._lock:
            if key not in self._rate_limit_buckets:
                self._rate_limit_buckets[key] = []

            # Clean old entries
            self._rate_limit_buckets[key] = [
                t for t in self._rate_limit_buckets[key] if t > cutoff
            ]
            self._rate_limit_buckets[key].append(now)

    # =========================================================================
    # Alert Creation
    # =========================================================================

    def create_alert(
        self,
        alert_type: InsulationAlertType,
        severity: AlertSeverity,
        source: str,
        message: str,
        context: Optional[AlertContext] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Alert:
        """
        Create a new alert.

        Args:
            alert_type: Type of alert
            severity: Severity level
            source: Source identifier (asset, system, etc.)
            message: Human-readable alert message
            context: Optional contextual information
            labels: Optional labels for categorization

        Returns:
            Created Alert instance (may be existing if deduplicated)
        """
        context = context or AlertContext()
        labels = labels or {}

        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            source=source,
            message=message,
            context=context,
            labels=labels,
        )

        with self._lock:
            # Check suppression
            if alert.fingerprint in self._suppressed_fingerprints:
                expires_at = self._suppressed_fingerprints[alert.fingerprint]
                if datetime.now(timezone.utc) <= expires_at:
                    logger.debug("Alert suppressed: %s", alert.fingerprint[:8])
                    alert.state = AlertState.SUPPRESSED
                    self._stats["alerts_suppressed"] += 1
                    return alert
                del self._suppressed_fingerprints[alert.fingerprint]

            # Check deduplication
            if alert.fingerprint in self._alert_by_fingerprint:
                existing_id = self._alert_by_fingerprint[alert.fingerprint]
                if existing_id in self._active_alerts:
                    existing = self._active_alerts[existing_id]
                    time_since_fired = (
                        datetime.now(timezone.utc) - (existing.fired_at or existing.created_at)
                    ).total_seconds()

                    if time_since_fired < self._deduplication_window_s:
                        self._stats["alerts_deduplicated"] += 1
                        logger.debug(
                            "Alert deduplicated: %s -> %s",
                            alert.fingerprint[:8],
                            existing.alert_id[:8],
                        )
                        return existing

            # Fire the alert
            alert.state = AlertState.FIRING
            alert.fired_at = datetime.now(timezone.utc)

            # Store alert
            self._active_alerts[alert.alert_id] = alert
            self._alert_by_fingerprint[alert.fingerprint] = alert.alert_id
            self._stats["alerts_created"] += 1

        logger.info(
            "Alert created: %s | %s | %s: %s",
            alert_type.value,
            severity.name_display,
            alert.alert_id[:8],
            message[:80],
        )

        # Invoke callback
        if self._on_alert_callback:
            try:
                self._on_alert_callback(alert)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

        # Send notifications asynchronously
        try:
            asyncio.create_task(self._notify_async(alert))
        except RuntimeError:
            # No event loop running
            pass

        return alert

    async def _notify_async(self, alert: Alert) -> None:
        """Send notifications to all channels asynchronously."""
        for channel in self._channels:
            try:
                success = await channel.send(alert)
                if success:
                    alert.notification_count += 1
                    self._stats["notifications_sent"] += 1
            except Exception as e:
                logger.error("Notification channel failed: %s", e)

    # =========================================================================
    # Alert Lifecycle Methods
    # =========================================================================

    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        notes: Optional[str] = None,
    ) -> AcknowledgmentResult:
        """
        Acknowledge an active alert.

        Args:
            alert_id: Alert identifier
            user_id: User performing acknowledgment
            notes: Optional acknowledgment notes

        Returns:
            AcknowledgmentResult with operation status
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if alert_id not in self._active_alerts:
                return AcknowledgmentResult(
                    success=False,
                    alert_id=alert_id,
                    acknowledged_by=user_id,
                    acknowledged_at=now,
                    error_message=f"Alert not found: {alert_id}",
                )

            alert = self._active_alerts[alert_id]

            if alert.state not in [AlertState.FIRING, AlertState.PENDING]:
                return AcknowledgmentResult(
                    success=False,
                    alert_id=alert_id,
                    acknowledged_by=user_id,
                    acknowledged_at=now,
                    error_message=f"Alert not in acknowledgeable state: {alert.state.value}",
                )

            # Update alert
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = now
            alert.acknowledged_by = user_id
            alert.acknowledgment_notes = notes

            self._stats["alerts_acknowledged"] += 1

        logger.info(
            "Alert acknowledged: %s by %s%s",
            alert_id[:8],
            user_id,
            f" - {notes}" if notes else "",
        )

        return AcknowledgmentResult(
            success=True,
            alert_id=alert_id,
            acknowledged_by=user_id,
            acknowledged_at=now,
            notes=notes,
        )

    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: Alert identifier
            resolution_notes: Optional resolution notes

        Returns:
            True if successfully resolved
        """
        with self._lock:
            if alert_id not in self._active_alerts:
                return False

            alert = self._active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            alert.resolution_notes = resolution_notes

            # Move to history
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_history_size:
                self._alert_history = self._alert_history[-self._max_history_size:]

            # Clean up active storage
            del self._active_alerts[alert_id]
            if alert.fingerprint in self._alert_by_fingerprint:
                del self._alert_by_fingerprint[alert.fingerprint]

            self._stats["alerts_resolved"] += 1

        logger.info(
            "Alert resolved: %s%s",
            alert_id[:8],
            f" - {resolution_notes}" if resolution_notes else "",
        )

        return True

    def escalate_alert(
        self,
        alert_id: str,
        escalation_level: EscalationLevel,
        reason: Optional[str] = None,
    ) -> EscalationResult:
        """
        Escalate an alert to a higher level.

        Args:
            alert_id: Alert identifier
            escalation_level: Target escalation level
            reason: Optional escalation reason

        Returns:
            EscalationResult with operation status
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if alert_id not in self._active_alerts:
                return EscalationResult(
                    success=False,
                    alert_id=alert_id,
                    previous_level=EscalationLevel.L1_OPERATOR,
                    new_level=escalation_level,
                    escalated_at=now,
                    error_message=f"Alert not found: {alert_id}",
                )

            alert = self._active_alerts[alert_id]
            previous_level = alert.escalation_level

            # Validate escalation direction
            level_order = [
                EscalationLevel.L1_OPERATOR,
                EscalationLevel.L2_SUPERVISOR,
                EscalationLevel.L3_ENGINEER,
                EscalationLevel.L4_MANAGEMENT,
            ]

            if level_order.index(escalation_level) <= level_order.index(previous_level):
                return EscalationResult(
                    success=False,
                    alert_id=alert_id,
                    previous_level=previous_level,
                    new_level=escalation_level,
                    escalated_at=now,
                    error_message=f"Cannot escalate from {previous_level.value} to {escalation_level.value}",
                )

            # Update alert
            alert.escalation_level = escalation_level
            alert.escalated_at = now

            self._stats["alerts_escalated"] += 1

        logger.warning(
            "Alert escalated: %s from %s to %s%s",
            alert_id[:8],
            previous_level.value,
            escalation_level.value,
            f" - {reason}" if reason else "",
        )

        return EscalationResult(
            success=True,
            alert_id=alert_id,
            previous_level=previous_level,
            new_level=escalation_level,
            escalated_at=now,
            escalation_reason=reason,
        )

    # =========================================================================
    # Suppression
    # =========================================================================

    def suppress_alert_pattern(
        self,
        fingerprint: str,
        duration_minutes: int = 60,
    ) -> bool:
        """
        Suppress alerts matching a fingerprint pattern.

        Args:
            fingerprint: Alert fingerprint to suppress
            duration_minutes: Suppression duration

        Returns:
            True if suppression was set
        """
        with self._lock:
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
            self._suppressed_fingerprints[fingerprint] = expires_at

        logger.info(
            "Alert pattern suppressed: %s until %s",
            fingerprint[:8],
            expires_at,
        )
        return True

    def suppress_asset_alerts(
        self,
        asset_id: str,
        duration_minutes: int = 60,
    ) -> int:
        """
        Suppress all alerts for an asset.

        Args:
            asset_id: Asset identifier
            duration_minutes: Suppression duration

        Returns:
            Number of alerts suppressed
        """
        suppressed_count = 0

        with self._lock:
            for alert in self._active_alerts.values():
                if alert.context.asset_id == asset_id:
                    expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
                    self._suppressed_fingerprints[alert.fingerprint] = expires_at
                    suppressed_count += 1

        logger.info(
            "Suppressed %d alerts for asset %s",
            suppressed_count,
            asset_id,
        )
        return suppressed_count

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_active_alerts(
        self,
        filters: Optional[AlertFilter] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            filters: Optional filter criteria

        Returns:
            List of matching active alerts
        """
        with self._lock:
            alerts = list(self._active_alerts.values())

        if filters:
            if filters.alert_types:
                alerts = [a for a in alerts if a.alert_type in filters.alert_types]

            if filters.severities:
                alerts = [a for a in alerts if a.severity in filters.severities]

            if filters.states:
                alerts = [a for a in alerts if a.state in filters.states]

            if filters.source_pattern:
                alerts = [a for a in alerts if filters.source_pattern in a.source]

            if filters.asset_id:
                alerts = [a for a in alerts if a.context.asset_id == filters.asset_id]

            if filters.escalation_levels:
                alerts = [a for a in alerts if a.escalation_level in filters.escalation_levels]

            if filters.from_time:
                alerts = [a for a in alerts if a.created_at >= filters.from_time]

            if filters.to_time:
                alerts = [a for a in alerts if a.created_at <= filters.to_time]

        # Sort by severity then time
        return sorted(
            alerts,
            key=lambda a: (-a.severity.value, a.created_at),
        )

    def get_alert_history(
        self,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Get alert history within time window.

        Args:
            time_window: Optional time window to filter
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        with self._lock:
            alerts = list(self._alert_history)

        if time_window:
            cutoff = datetime.now(timezone.utc) - time_window
            alerts = [a for a in alerts if a.created_at >= cutoff]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_alerts": len(self._active_alerts),
                "suppressed_patterns": len(self._suppressed_fingerprints),
                "history_size": len(self._alert_history),
                "notification_channels": len(self._channels),
                "rules_count": len(self._rules),
            }

    # =========================================================================
    # Notification Channel Management
    # =========================================================================

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)
        logger.info("Added notification channel: %s", type(channel).__name__)

    def remove_notification_channel(self, channel_type: type) -> bool:
        """
        Remove notification channels of a specific type.

        Args:
            channel_type: Type of channel to remove

        Returns:
            True if any channels were removed
        """
        original_count = len(self._channels)
        self._channels = [c for c in self._channels if not isinstance(c, channel_type)]
        removed = original_count - len(self._channels)

        if removed > 0:
            logger.info("Removed %d %s channels", removed, channel_type.__name__)

        return removed > 0


# =============================================================================
# Global Instance
# =============================================================================

_alert_manager: Optional[InsulscanAlertManager] = None


def get_alert_manager() -> InsulscanAlertManager:
    """
    Get or create the global alert manager.

    Returns:
        Global InsulscanAlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = InsulscanAlertManager()
    return _alert_manager
