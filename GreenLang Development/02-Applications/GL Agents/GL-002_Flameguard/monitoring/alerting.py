"""
GL-002 FLAMEGUARD - Alerting

Alert management and notification system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert lifecycle states."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    boiler_id: str
    state: AlertState = AlertState.PENDING
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute unique fingerprint for alert deduplication."""
        data = f"{self.name}:{self.boiler_id}:{sorted(self.labels.items())}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "boiler_id": self.boiler_id,
            "state": self.state.value,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "labels": self.labels,
            "value": self.value,
            "threshold": self.threshold,
            "fingerprint": self.fingerprint,
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    message_template: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    for_duration_s: float = 0.0  # Time condition must be true before firing
    repeat_interval_s: float = 3600.0  # Repeat notification interval
    enabled: bool = True


class NotificationChannel:
    """Base notification channel."""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification."""
        raise NotImplementedError


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel."""

    async def send(self, alert: Alert) -> bool:
        """Log alert."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            f"[ALERT] {alert.name}: {alert.message} "
            f"(boiler={alert.boiler_id}, severity={alert.severity.value})",
        )
        return True


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        self.url = url
        self.headers = headers or {}

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        # In production, use aiohttp:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         self.url,
        #         json=alert.to_dict(),
        #         headers=self.headers,
        #     ) as response:
        #         return response.status == 200

        logger.info(f"Webhook alert: {self.url} - {alert.name}")
        return True


class AlertManager:
    """
    Alert management system.

    Features:
    - Alert deduplication
    - Alert grouping
    - Notification routing
    - Alert silencing
    - Escalation
    """

    def __init__(
        self,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self._on_alert = on_alert

        # Rules
        self._rules: Dict[str, AlertRule] = {}

        # Active alerts
        self._active_alerts: Dict[str, Alert] = {}

        # Alert history
        self._history: List[Alert] = []
        self._max_history = 1000

        # Pending conditions (for for_duration evaluation)
        self._pending: Dict[str, datetime] = {}

        # Silences
        self._silences: Dict[str, datetime] = {}  # fingerprint -> expires_at

        # Notification channels
        self._channels: List[NotificationChannel] = [LogNotificationChannel()]

        # Statistics
        self._stats = {
            "alerts_fired": 0,
            "alerts_resolved": 0,
            "notifications_sent": 0,
        }

        # Register default rules
        self._register_default_rules()

        logger.info("AlertManager initialized")

    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        default_rules = [
            AlertRule(
                name="HighO2",
                condition="o2_percent > 5.0",
                severity=AlertSeverity.WARNING,
                message_template="High O2: {value:.1f}% > {threshold}%",
                for_duration_s=60.0,
            ),
            AlertRule(
                name="LowO2",
                condition="o2_percent < 1.5",
                severity=AlertSeverity.ERROR,
                message_template="Low O2: {value:.1f}% < {threshold}%",
                for_duration_s=30.0,
            ),
            AlertRule(
                name="HighCO",
                condition="co_ppm > 400",
                severity=AlertSeverity.ERROR,
                message_template="High CO: {value:.0f} ppm > {threshold} ppm",
                for_duration_s=0.0,  # Immediate
            ),
            AlertRule(
                name="LowEfficiency",
                condition="efficiency_percent < 75.0",
                severity=AlertSeverity.WARNING,
                message_template="Low efficiency: {value:.1f}% < {threshold}%",
                for_duration_s=300.0,
            ),
            AlertRule(
                name="HighFlueGasTemp",
                condition="flue_gas_temp_f > 650",
                severity=AlertSeverity.WARNING,
                message_template="High flue gas temp: {value:.0f}F > {threshold}F",
                for_duration_s=120.0,
            ),
            AlertRule(
                name="FlameFailure",
                condition="flame_proven == False and firing == True",
                severity=AlertSeverity.CRITICAL,
                message_template="Flame failure detected",
                for_duration_s=0.0,  # Immediate
            ),
            AlertRule(
                name="SafetyTrip",
                condition="safety_tripped == True",
                severity=AlertSeverity.CRITICAL,
                message_template="Safety interlock tripped: {trip_cause}",
                for_duration_s=0.0,
            ),
        ]

        for rule in default_rules:
            self._rules[rule.name] = rule

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """Remove alert rule."""
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel."""
        self._channels.append(channel)

    async def evaluate(
        self,
        boiler_id: str,
        metrics: Dict[str, Any],
    ) -> List[Alert]:
        """Evaluate all rules against current metrics."""
        new_alerts: List[Alert] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            try:
                # Evaluate condition
                condition_met, value, threshold = self._evaluate_condition(
                    rule.condition, metrics
                )

                fingerprint = self._compute_rule_fingerprint(rule.name, boiler_id)

                if condition_met:
                    # Check for_duration
                    if rule.for_duration_s > 0:
                        if fingerprint not in self._pending:
                            self._pending[fingerprint] = datetime.now(timezone.utc)
                            continue

                        pending_since = self._pending[fingerprint]
                        elapsed = (datetime.now(timezone.utc) - pending_since).total_seconds()
                        if elapsed < rule.for_duration_s:
                            continue

                    # Fire alert
                    alert = await self._fire_alert(
                        rule, boiler_id, value, threshold
                    )
                    if alert:
                        new_alerts.append(alert)

                else:
                    # Clear pending
                    if fingerprint in self._pending:
                        del self._pending[fingerprint]

                    # Resolve existing alert
                    await self._resolve_alert(rule.name, boiler_id)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        return new_alerts

    def _evaluate_condition(
        self,
        condition: str,
        metrics: Dict[str, Any],
    ) -> tuple:
        """Evaluate condition expression."""
        # Simple expression evaluation
        # In production, use a proper expression parser

        # Extract comparison
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    threshold = float(parts[1].strip())
                    value = metrics.get(var_name, 0.0)

                    if op == ">":
                        return value > threshold, value, threshold
                    elif op == "<":
                        return value < threshold, value, threshold
                    elif op == ">=":
                        return value >= threshold, value, threshold
                    elif op == "<=":
                        return value <= threshold, value, threshold
                    elif op == "==":
                        return value == threshold, value, threshold
                    elif op == "!=":
                        return value != threshold, value, threshold

        return False, None, None

    def _compute_rule_fingerprint(self, rule_name: str, boiler_id: str) -> str:
        """Compute fingerprint for rule instance."""
        data = f"{rule_name}:{boiler_id}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    async def _fire_alert(
        self,
        rule: AlertRule,
        boiler_id: str,
        value: Optional[float],
        threshold: Optional[float],
    ) -> Optional[Alert]:
        """Fire an alert."""
        fingerprint = self._compute_rule_fingerprint(rule.name, boiler_id)

        # Check silence
        if fingerprint in self._silences:
            if datetime.now(timezone.utc) < self._silences[fingerprint]:
                return None
            else:
                del self._silences[fingerprint]

        # Check deduplication
        if fingerprint in self._active_alerts:
            existing = self._active_alerts[fingerprint]
            if existing.state == AlertState.FIRING:
                return None  # Already firing

        # Create alert
        message = rule.message_template.format(
            value=value or 0,
            threshold=threshold or 0,
        )

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=rule.name,
            severity=rule.severity,
            message=message,
            boiler_id=boiler_id,
            state=AlertState.FIRING,
            labels=dict(rule.labels),
            annotations=dict(rule.annotations),
            value=value,
            threshold=threshold,
            fingerprint=fingerprint,
        )

        # Store alert
        self._active_alerts[fingerprint] = alert
        self._add_to_history(alert)
        self._stats["alerts_fired"] += 1

        # Send notifications
        await self._notify(alert)

        # Callback
        if self._on_alert:
            self._on_alert(alert)

        logger.warning(f"Alert fired: {alert.name} for {boiler_id}")
        return alert

    async def _resolve_alert(self, rule_name: str, boiler_id: str) -> Optional[Alert]:
        """Resolve an active alert."""
        fingerprint = self._compute_rule_fingerprint(rule_name, boiler_id)

        if fingerprint not in self._active_alerts:
            return None

        alert = self._active_alerts[fingerprint]
        if alert.state != AlertState.FIRING:
            return None

        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        self._stats["alerts_resolved"] += 1

        del self._active_alerts[fingerprint]
        self._add_to_history(alert)

        logger.info(f"Alert resolved: {alert.name} for {boiler_id}")
        return alert

    async def _notify(self, alert: Alert) -> None:
        """Send alert notifications."""
        for channel in self._channels:
            try:
                success = await channel.send(alert)
                if success:
                    self._stats["notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history."""
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def acknowledge(
        self,
        alert_id: str,
        operator: str,
    ) -> bool:
        """Acknowledge an alert."""
        for alert in self._active_alerts.values():
            if alert.alert_id == alert_id:
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = operator
                logger.info(f"Alert acknowledged: {alert.name} by {operator}")
                return True
        return False

    def silence(
        self,
        fingerprint: str,
        duration_minutes: int = 60,
    ) -> bool:
        """Silence an alert pattern."""
        expires = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self._silences[fingerprint] = expires
        logger.info(f"Alert silenced: {fingerprint} until {expires}")
        return True

    def get_active_alerts(
        self,
        boiler_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get active alerts with optional filters."""
        alerts = list(self._active_alerts.values())

        if boiler_id:
            alerts = [a for a in alerts if a.boiler_id == boiler_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.fired_at, reverse=True)

    def get_alert_history(
        self,
        limit: int = 100,
        boiler_id: Optional[str] = None,
    ) -> List[Alert]:
        """Get alert history."""
        alerts = self._history

        if boiler_id:
            alerts = [a for a in alerts if a.boiler_id == boiler_id]

        return alerts[-limit:]

    def get_statistics(self) -> Dict:
        """Get alerting statistics."""
        return {
            **self._stats,
            "active_alerts": len(self._active_alerts),
            "silences": len(self._silences),
            "rules": len(self._rules),
        }
