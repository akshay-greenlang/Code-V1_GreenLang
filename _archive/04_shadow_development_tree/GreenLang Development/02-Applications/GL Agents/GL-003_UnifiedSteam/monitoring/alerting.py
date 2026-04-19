"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Alert Management

This module implements comprehensive alert management for steam system optimization,
including alert creation, acknowledgment, escalation, and notification routing.

Alert Types:
    - STEAM_QUALITY: Steam quality degradation alerts
    - DESUPERHEATER: Desuperheater performance alerts
    - CONDENSATE: Condensate recovery issues
    - TRAP_FAILURE: Steam trap failure detection
    - BALANCE_DEVIATION: Enthalpy balance deviations
    - SAFETY_ENVELOPE: Safety boundary violations
    - OPTIMIZATION_OPPORTUNITY: Optimization recommendations

Features:
    - Alert deduplication using fingerprinting
    - Rate limiting to prevent alert storms
    - Multi-level escalation (L1-L4)
    - Alert acknowledgment with audit trail
    - Notification channel routing

Example:
    >>> manager = AlertManager()
    >>> alert = manager.create_alert(
    ...     alert_type=AlertType.TRAP_FAILURE,
    ...     severity=AlertSeverity.CRITICAL,
    ...     source="TRAP-001",
    ...     message="Steam trap blow-through detected",
    ...     context=AlertContext(trap_id="TRAP-001", loss_rate_kg_h=50.0)
    ... )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Steam system alert types."""
    STEAM_QUALITY = "steam_quality"
    DESUPERHEATER = "desuperheater"
    CONDENSATE = "condensate"
    TRAP_FAILURE = "trap_failure"
    BALANCE_DEVIATION = "balance_deviation"
    SAFETY_ENVELOPE = "safety_envelope"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


class AlertSeverity(Enum):
    """Alert severity levels following industrial standards."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    @property
    def priority(self) -> int:
        """Get numeric priority for sorting (lower is more urgent)."""
        priority_map = {
            AlertSeverity.EMERGENCY: 1,
            AlertSeverity.CRITICAL: 2,
            AlertSeverity.WARNING: 3,
            AlertSeverity.INFO: 4,
        }
        return priority_map.get(self, 5)


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
            EscalationLevel.L1_OPERATOR: 15,
            EscalationLevel.L2_SUPERVISOR: 30,
            EscalationLevel.L3_ENGINEER: 60,
            EscalationLevel.L4_MANAGEMENT: 120,
        }
        return timeouts.get(self, 60)


@dataclass
class AlertContext:
    """Contextual information for an alert."""
    # Location identifiers
    system_id: Optional[str] = None
    site_id: Optional[str] = None
    equipment_id: Optional[str] = None
    trap_id: Optional[str] = None
    header_id: Optional[str] = None

    # Measured values
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    deviation_percent: Optional[float] = None

    # Steam-specific context
    pressure_bar: Optional[float] = None
    temperature_c: Optional[float] = None
    steam_quality_percent: Optional[float] = None
    loss_rate_kg_h: Optional[float] = None
    energy_loss_kw: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "system_id": self.system_id,
            "site_id": self.site_id,
            "equipment_id": self.equipment_id,
            "trap_id": self.trap_id,
            "header_id": self.header_id,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "deviation_percent": self.deviation_percent,
            "pressure_bar": self.pressure_bar,
            "temperature_c": self.temperature_c,
            "steam_quality_percent": self.steam_quality_percent,
            "loss_rate_kg_h": self.loss_rate_kg_h,
            "energy_loss_kw": self.energy_loss_kw,
            "metadata": self.metadata,
        }


@dataclass
class Alert:
    """Alert instance representing a detected condition."""
    alert_id: str
    alert_type: AlertType
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
            f"{self.context.system_id}:"
            f"{self.context.equipment_id}:"
            f"{sorted(self.labels.items())}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
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
            "escalation_level": self.escalation_level.value,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "fingerprint": self.fingerprint,
            "labels": self.labels,
            "notification_count": self.notification_count,
        }


@dataclass
class AlertFilter:
    """Filter criteria for querying alerts."""
    alert_types: Optional[List[AlertType]] = None
    severities: Optional[List[AlertSeverity]] = None
    states: Optional[List[AlertState]] = None
    source_pattern: Optional[str] = None
    system_id: Optional[str] = None
    site_id: Optional[str] = None
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
            AlertSeverity.EMERGENCY: logging.CRITICAL,
        }
        log_level = log_level_map.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            "[ALERT] %s | %s | %s: %s (source=%s, escalation=%s)",
            alert.alert_type.value.upper(),
            alert.severity.value.upper(),
            alert.alert_id[:8],
            alert.message,
            alert.source,
            alert.escalation_level.value,
        )
        return True


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel for external integrations."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 10.0,
    ) -> None:
        """Initialize webhook channel."""
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout_s = timeout_s

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        # In production, use aiohttp or httpx:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         self.url,
        #         json=alert.to_dict(),
        #         headers=self.headers,
        #         timeout=aiohttp.ClientTimeout(total=self.timeout_s),
        #     ) as response:
        #         return response.status == 200

        logger.info("Webhook notification: %s -> %s", alert.alert_id[:8], self.url)
        return True


class AlertManager:
    """
    Alert management system for steam system optimization.

    This class provides comprehensive alert lifecycle management including
    creation, acknowledgment, escalation, and notification routing with
    deduplication and rate limiting capabilities.

    Attributes:
        namespace: Namespace for alert identification
        on_alert_callback: Optional callback for new alerts

    Example:
        >>> manager = AlertManager(namespace="steam_system")
        >>> alert = manager.create_alert(
        ...     alert_type=AlertType.TRAP_FAILURE,
        ...     severity=AlertSeverity.CRITICAL,
        ...     source="TRAP-001",
        ...     message="Blow-through detected",
        ...     context=AlertContext(trap_id="TRAP-001")
        ... )
        >>> result = manager.acknowledge_alert(alert.alert_id, "operator_1", "Investigating")
    """

    def __init__(
        self,
        namespace: str = "unifiedsteam",
        on_alert_callback: Optional[Callable[[Alert], None]] = None,
        deduplication_window_s: float = 300.0,
        rate_limit_per_minute: int = 60,
    ) -> None:
        """
        Initialize AlertManager.

        Args:
            namespace: Namespace for alert identification
            on_alert_callback: Optional callback invoked on new alerts
            deduplication_window_s: Window for alert deduplication (seconds)
            rate_limit_per_minute: Maximum alerts per minute per fingerprint
        """
        self.namespace = namespace
        self._on_alert_callback = on_alert_callback
        self._deduplication_window_s = deduplication_window_s
        self._rate_limit_per_minute = rate_limit_per_minute

        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self._alert_by_fingerprint: Dict[str, str] = {}  # fingerprint -> alert_id
        self._alert_history: List[Alert] = []
        self._max_history_size = 10000

        # Rate limiting
        self._rate_limit_buckets: Dict[str, List[datetime]] = {}

        # Suppression rules
        self._suppressed_fingerprints: Dict[str, datetime] = {}  # fingerprint -> expires_at

        # Notification channels
        self._channels: List[NotificationChannel] = [LogNotificationChannel()]

        # Escalation rules
        self._escalation_rules: Dict[AlertType, Dict[AlertSeverity, int]] = (
            self._default_escalation_rules()
        )

        # Statistics
        self._stats = {
            "alerts_created": 0,
            "alerts_acknowledged": 0,
            "alerts_resolved": 0,
            "alerts_escalated": 0,
            "alerts_deduplicated": 0,
            "alerts_rate_limited": 0,
            "notifications_sent": 0,
        }

        logger.info(
            "AlertManager initialized: namespace=%s, dedup_window=%ss",
            namespace,
            deduplication_window_s,
        )

    def _default_escalation_rules(self) -> Dict[AlertType, Dict[AlertSeverity, int]]:
        """Define default escalation timeout rules (minutes)."""
        return {
            AlertType.SAFETY_ENVELOPE: {
                AlertSeverity.EMERGENCY: 5,
                AlertSeverity.CRITICAL: 10,
                AlertSeverity.WARNING: 30,
                AlertSeverity.INFO: 60,
            },
            AlertType.TRAP_FAILURE: {
                AlertSeverity.EMERGENCY: 10,
                AlertSeverity.CRITICAL: 20,
                AlertSeverity.WARNING: 60,
                AlertSeverity.INFO: 120,
            },
            AlertType.STEAM_QUALITY: {
                AlertSeverity.EMERGENCY: 10,
                AlertSeverity.CRITICAL: 15,
                AlertSeverity.WARNING: 45,
                AlertSeverity.INFO: 90,
            },
            AlertType.DESUPERHEATER: {
                AlertSeverity.EMERGENCY: 10,
                AlertSeverity.CRITICAL: 20,
                AlertSeverity.WARNING: 60,
                AlertSeverity.INFO: 120,
            },
            AlertType.CONDENSATE: {
                AlertSeverity.EMERGENCY: 15,
                AlertSeverity.CRITICAL: 30,
                AlertSeverity.WARNING: 60,
                AlertSeverity.INFO: 120,
            },
            AlertType.BALANCE_DEVIATION: {
                AlertSeverity.EMERGENCY: 15,
                AlertSeverity.CRITICAL: 30,
                AlertSeverity.WARNING: 90,
                AlertSeverity.INFO: 180,
            },
            AlertType.OPTIMIZATION_OPPORTUNITY: {
                AlertSeverity.INFO: 240,
                AlertSeverity.WARNING: 180,
                AlertSeverity.CRITICAL: 120,
                AlertSeverity.EMERGENCY: 60,
            },
        }

    def _check_rate_limit(self, fingerprint: str) -> bool:
        """Check if alert is rate limited."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=1)

        if fingerprint not in self._rate_limit_buckets:
            self._rate_limit_buckets[fingerprint] = []

        # Clean old entries
        self._rate_limit_buckets[fingerprint] = [
            t for t in self._rate_limit_buckets[fingerprint] if t > cutoff
        ]

        # Check limit
        if len(self._rate_limit_buckets[fingerprint]) >= self._rate_limit_per_minute:
            self._stats["alerts_rate_limited"] += 1
            return True

        # Add current timestamp
        self._rate_limit_buckets[fingerprint].append(now)
        return False

    def _check_suppression(self, fingerprint: str) -> bool:
        """Check if alert fingerprint is suppressed."""
        if fingerprint not in self._suppressed_fingerprints:
            return False

        expires_at = self._suppressed_fingerprints[fingerprint]
        if datetime.now(timezone.utc) > expires_at:
            del self._suppressed_fingerprints[fingerprint]
            return False

        return True

    def _check_deduplication(self, fingerprint: str) -> Optional[Alert]:
        """Check for existing active alert with same fingerprint."""
        if fingerprint not in self._alert_by_fingerprint:
            return None

        alert_id = self._alert_by_fingerprint[fingerprint]
        if alert_id not in self._active_alerts:
            # Stale reference, clean up
            del self._alert_by_fingerprint[fingerprint]
            return None

        alert = self._active_alerts[alert_id]

        # Check if within deduplication window
        time_since_fired = (
            datetime.now(timezone.utc) - (alert.fired_at or alert.created_at)
        ).total_seconds()

        if time_since_fired < self._deduplication_window_s:
            self._stats["alerts_deduplicated"] += 1
            return alert

        return None

    def create_alert(
        self,
        alert_type: AlertType,
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
            source: Source identifier (equipment, sensor, etc.)
            message: Human-readable alert message
            context: Optional contextual information
            labels: Optional labels for categorization

        Returns:
            Created Alert instance (may be existing if deduplicated)
        """
        context = context or AlertContext()
        labels = labels or {}

        # Create alert with temporary fingerprint
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            source=source,
            message=message,
            context=context,
            labels=labels,
        )

        # Check suppression
        if self._check_suppression(alert.fingerprint):
            logger.debug("Alert suppressed: %s", alert.fingerprint[:8])
            alert.state = AlertState.SUPPRESSED
            return alert

        # Check rate limiting
        if self._check_rate_limit(alert.fingerprint):
            logger.warning("Alert rate limited: %s", alert.fingerprint[:8])
            alert.state = AlertState.SUPPRESSED
            return alert

        # Check deduplication
        existing = self._check_deduplication(alert.fingerprint)
        if existing:
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
            severity.value,
            alert.alert_id[:8],
            message,
        )

        # Invoke callback
        if self._on_alert_callback:
            try:
                self._on_alert_callback(alert)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

        # Send notifications asynchronously
        asyncio.create_task(self._notify_async(alert))

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
                error_message=f"Alert not in acknowledgeble state: {alert.state.value}",
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
                error_message=(
                    f"Cannot escalate from {previous_level.value} to {escalation_level.value}"
                ),
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

        # Send escalation notification
        asyncio.create_task(self._notify_async(alert))

        return EscalationResult(
            success=True,
            alert_id=alert_id,
            previous_level=previous_level,
            new_level=escalation_level,
            escalated_at=now,
            escalation_reason=reason,
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
        if alert_id not in self._active_alerts:
            return False

        alert = self._active_alerts[alert_id]
        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)

        # Move to history
        self._add_to_history(alert)

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

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history with size limit."""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history_size:
            self._alert_history = self._alert_history[-self._max_history_size:]

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
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self._suppressed_fingerprints[fingerprint] = expires_at
        logger.info("Alert pattern suppressed: %s until %s", fingerprint[:8], expires_at)
        return True

    def get_active_alerts(
        self,
        filters: Optional[AlertFilter] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            filters: Optional filter criteria

        Returns:
            List of matching active alerts sorted by severity and time
        """
        alerts = list(self._active_alerts.values())

        if filters:
            if filters.alert_types:
                alerts = [a for a in alerts if a.alert_type in filters.alert_types]

            if filters.severities:
                alerts = [a for a in alerts if a.severity in filters.severities]

            if filters.states:
                alerts = [a for a in alerts if a.state in filters.states]

            if filters.source_pattern:
                alerts = [
                    a for a in alerts if filters.source_pattern in a.source
                ]

            if filters.system_id:
                alerts = [
                    a for a in alerts if a.context.system_id == filters.system_id
                ]

            if filters.site_id:
                alerts = [
                    a for a in alerts if a.context.site_id == filters.site_id
                ]

            if filters.escalation_levels:
                alerts = [
                    a for a in alerts if a.escalation_level in filters.escalation_levels
                ]

            if filters.from_time:
                alerts = [
                    a for a in alerts
                    if a.created_at >= filters.from_time
                ]

            if filters.to_time:
                alerts = [
                    a for a in alerts
                    if a.created_at <= filters.to_time
                ]

        # Sort by severity (priority) then by created time (oldest first)
        return sorted(
            alerts,
            key=lambda a: (a.severity.priority, a.created_at),
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
        alerts = self._alert_history

        if time_window:
            cutoff = datetime.now(timezone.utc) - time_window
            alerts = [a for a in alerts if a.created_at >= cutoff]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:limit]

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)
        logger.info("Added notification channel: %s", type(channel).__name__)

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            **self._stats,
            "active_alerts": len(self._active_alerts),
            "suppressed_patterns": len(self._suppressed_fingerprints),
            "history_size": len(self._alert_history),
            "notification_channels": len(self._channels),
        }

    async def check_auto_escalations(self) -> List[EscalationResult]:
        """
        Check and perform automatic escalations for overdue alerts.

        Returns:
            List of escalation results
        """
        results: List[EscalationResult] = []
        now = datetime.now(timezone.utc)

        for alert in self._active_alerts.values():
            if alert.state not in [AlertState.FIRING, AlertState.PENDING]:
                continue

            # Get escalation timeout for this alert type/severity
            alert_rules = self._escalation_rules.get(alert.alert_type, {})
            timeout_minutes = alert_rules.get(alert.severity, 60)

            # Check if escalation is needed
            time_since_fired = (now - (alert.fired_at or alert.created_at)).total_seconds() / 60

            if time_since_fired < timeout_minutes:
                continue

            # Determine next escalation level
            level_order = [
                EscalationLevel.L1_OPERATOR,
                EscalationLevel.L2_SUPERVISOR,
                EscalationLevel.L3_ENGINEER,
                EscalationLevel.L4_MANAGEMENT,
            ]

            current_index = level_order.index(alert.escalation_level)
            if current_index >= len(level_order) - 1:
                continue  # Already at max level

            next_level = level_order[current_index + 1]
            result = self.escalate_alert(
                alert.alert_id,
                next_level,
                f"Auto-escalation: unacknowledged for {time_since_fired:.0f} minutes",
            )
            results.append(result)

        return results

    async def start_background_escalation_check(
        self,
        interval_s: float = 60.0,
    ) -> None:
        """Start background auto-escalation checking."""
        self._escalation_running = True

        while self._escalation_running:
            try:
                await self.check_auto_escalations()
            except Exception as e:
                logger.error("Auto-escalation check failed: %s", e)

            await asyncio.sleep(interval_s)

    def stop_background_escalation_check(self) -> None:
        """Stop background escalation checking."""
        self._escalation_running = False
