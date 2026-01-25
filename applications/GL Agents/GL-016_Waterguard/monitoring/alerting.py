"""
Alert Manager for GL-016 Waterguard

This module provides alert generation and management for the Waterguard
boiler water chemistry optimization agent. Supports severity levels,
alert routing, and silencing.

Key Features:
    - Severity levels: info, warning, critical
    - Alert rules with thresholds
    - Operator routing
    - Alert silencing/inhibition
    - Alert history and metrics

Example:
    >>> manager = AlertManager()
    >>> manager.add_rule(AlertRule(
    ...     name="high_conductivity",
    ...     parameter="conductivity",
    ...     threshold=6000,
    ...     operator=">",
    ...     severity=AlertSeverity.WARNING
    ... ))
    >>> alert = manager.evaluate("boiler-001", "conductivity", 6500)
    >>> manager.route_alert(alert)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertState(str, Enum):
    """Alert state."""

    FIRING = "FIRING"
    RESOLVED = "RESOLVED"
    SILENCED = "SILENCED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class AlertRule(BaseModel):
    """Definition of an alert rule."""

    rule_id: str = Field(default_factory=lambda: f"rule-{uuid4().hex[:8]}")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    parameter: str = Field(..., description="Parameter to evaluate")
    threshold: float = Field(..., description="Threshold value")
    operator: str = Field(..., description="Comparison operator (>, <, >=, <=, ==)")
    severity: AlertSeverity = Field(..., description="Alert severity")
    duration_seconds: int = Field(0, ge=0, description="Duration condition must be met")
    labels: Dict[str, str] = Field(default_factory=dict, description="Additional labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    enabled: bool = Field(True, description="Whether rule is enabled")

    def evaluate(self, value: float) -> bool:
        """
        Evaluate the rule against a value.

        Args:
            value: Value to evaluate

        Returns:
            True if rule condition is met
        """
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return value == self.threshold
        elif self.operator == "!=":
            return value != self.threshold
        else:
            return False


class Alert(BaseModel):
    """An active or historical alert."""

    alert_id: str = Field(default_factory=lambda: f"alert-{uuid4().hex[:12]}")
    rule_id: str = Field(..., description="ID of rule that triggered alert")
    rule_name: str = Field(..., description="Name of rule")
    asset_id: str = Field(..., description="Asset identifier")
    parameter: str = Field(..., description="Parameter name")
    severity: AlertSeverity = Field(..., description="Alert severity")
    state: AlertState = Field(default=AlertState.FIRING, description="Alert state")

    # Values
    current_value: float = Field(..., description="Current parameter value")
    threshold: float = Field(..., description="Threshold that was breached")
    operator: str = Field(..., description="Comparison operator")

    # Timestamps
    firing_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When alert started firing"
    )
    resolved_time: Optional[datetime] = Field(None, description="When alert resolved")
    acknowledged_time: Optional[datetime] = Field(None, description="When acknowledged")

    # Routing
    routed_to: List[str] = Field(default_factory=list, description="Recipients")
    notification_sent: bool = Field(False, description="Whether notification was sent")

    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    fingerprint: str = Field(default="", description="Unique fingerprint for deduplication")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def __init__(self, **data):
        super().__init__(**data)
        if not self.fingerprint:
            self.fingerprint = f"{self.asset_id}:{self.rule_id}:{self.parameter}"


class AlertRoute(BaseModel):
    """Routing rule for alerts."""

    route_id: str = Field(default_factory=lambda: f"route-{uuid4().hex[:8]}")
    name: str = Field(..., description="Route name")
    match_labels: Dict[str, str] = Field(
        default_factory=dict, description="Labels to match"
    )
    match_severity: Optional[List[AlertSeverity]] = Field(
        None, description="Severities to match"
    )
    receivers: List[str] = Field(..., description="Receiver identifiers")
    continue_routing: bool = Field(False, description="Continue to next route")
    group_wait: int = Field(30, ge=0, description="Wait time before sending")
    group_interval: int = Field(300, ge=0, description="Interval between group notifications")
    repeat_interval: int = Field(3600, ge=0, description="Repeat notification interval")

    def matches(self, alert: Alert) -> bool:
        """Check if alert matches this route."""
        # Check severity
        if self.match_severity and alert.severity not in self.match_severity:
            return False

        # Check labels
        for key, value in self.match_labels.items():
            if alert.labels.get(key) != value:
                return False

        return True


class AlertSilence(BaseModel):
    """Silence rule for suppressing alerts."""

    silence_id: str = Field(default_factory=lambda: f"silence-{uuid4().hex[:8]}")
    created_by: str = Field(..., description="Creator of silence")
    comment: str = Field(..., description="Reason for silence")
    starts_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Silence start time"
    )
    ends_at: datetime = Field(..., description="Silence end time")
    matchers: Dict[str, str] = Field(
        default_factory=dict, description="Label matchers"
    )
    state: str = Field(default="active", description="Silence state")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def is_active(self) -> bool:
        """Check if silence is currently active."""
        now = datetime.now(timezone.utc)
        return self.starts_at <= now < self.ends_at and self.state == "active"

    def matches(self, alert: Alert) -> bool:
        """Check if alert is matched by this silence."""
        if not self.is_active():
            return False

        for key, value in self.matchers.items():
            if key == "asset_id" and alert.asset_id != value:
                return False
            elif key == "rule_name" and alert.rule_name != value:
                return False
            elif key == "parameter" and alert.parameter != value:
                return False
            elif alert.labels.get(key) != value:
                return False

        return True


class AlertManager:
    """
    Alert manager for Waterguard operational alerts.

    Manages alert rules, generates alerts, routes to operators,
    and tracks alert history.

    Attributes:
        rules: Alert rules
        routes: Alert routing configuration
        silences: Active silences
        active_alerts: Currently firing alerts

    Example:
        >>> manager = AlertManager()
        >>> manager.add_rule(AlertRule(
        ...     name="high_conductivity",
        ...     parameter="conductivity",
        ...     threshold=6000,
        ...     operator=">",
        ...     severity=AlertSeverity.WARNING
        ... ))
        >>> alert = manager.evaluate("boiler-001", "conductivity", 6500)
    """

    def __init__(self):
        """Initialize alert manager."""
        self._rules: Dict[str, AlertRule] = {}
        self._routes: List[AlertRoute] = []
        self._silences: Dict[str, AlertSilence] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._pending_conditions: Dict[str, datetime] = {}
        self._notification_handlers: Dict[str, Callable] = {}

        # Add default rules
        self._add_default_rules()

        # Add default routes
        self._add_default_routes()

        logger.info("AlertManager initialized")

    def _add_default_rules(self) -> None:
        """Add default alert rules for water chemistry."""
        default_rules = [
            # Conductivity rules
            AlertRule(
                name="high_conductivity_warning",
                description="Conductivity approaching limit",
                parameter="conductivity",
                threshold=5500,
                operator=">=",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="high_conductivity_critical",
                description="Conductivity exceeds limit",
                parameter="conductivity",
                threshold=6000,
                operator=">=",
                severity=AlertSeverity.CRITICAL,
            ),
            # pH rules
            AlertRule(
                name="low_ph_warning",
                description="pH below target range",
                parameter="ph",
                threshold=9.5,
                operator="<",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="high_ph_warning",
                description="pH above target range",
                parameter="ph",
                threshold=11.0,
                operator=">",
                severity=AlertSeverity.WARNING,
            ),
            # Silica rules
            AlertRule(
                name="high_silica_warning",
                description="Silica approaching limit",
                parameter="silica",
                threshold=120,
                operator=">=",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="high_silica_critical",
                description="Silica exceeds limit",
                parameter="silica",
                threshold=150,
                operator=">=",
                severity=AlertSeverity.CRITICAL,
            ),
            # Phosphate rules
            AlertRule(
                name="low_phosphate_warning",
                description="Phosphate below target",
                parameter="phosphate",
                threshold=5,
                operator="<",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="high_phosphate_warning",
                description="Phosphate above target",
                parameter="phosphate",
                threshold=25,
                operator=">",
                severity=AlertSeverity.WARNING,
            ),
            # Dissolved oxygen
            AlertRule(
                name="high_do_warning",
                description="Dissolved oxygen elevated",
                parameter="dissolved_oxygen",
                threshold=7,
                operator=">",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="high_do_critical",
                description="Dissolved oxygen critically high",
                parameter="dissolved_oxygen",
                threshold=10,
                operator=">",
                severity=AlertSeverity.CRITICAL,
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

    def _add_default_routes(self) -> None:
        """Add default alert routes."""
        default_routes = [
            AlertRoute(
                name="critical_to_all",
                match_severity=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY],
                receivers=["water_treatment_specialist", "shift_supervisor", "operations_manager"],
            ),
            AlertRoute(
                name="warning_to_operators",
                match_severity=[AlertSeverity.WARNING],
                receivers=["water_treatment_specialist", "shift_operator"],
            ),
            AlertRoute(
                name="info_to_log",
                match_severity=[AlertSeverity.INFO],
                receivers=["log_only"],
            ),
        ]

        self._routes = default_routes

    def add_rule(self, rule: AlertRule) -> str:
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add

        Returns:
            Rule ID
        """
        self._rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
        return rule.rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def add_route(self, route: AlertRoute) -> str:
        """Add an alert route."""
        self._routes.append(route)
        logger.info(f"Added alert route: {route.name}")
        return route.route_id

    def add_silence(self, silence: AlertSilence) -> str:
        """
        Add an alert silence.

        Args:
            silence: Silence to add

        Returns:
            Silence ID
        """
        self._silences[silence.silence_id] = silence
        logger.info(
            f"Added silence: {silence.silence_id}",
            extra={"created_by": silence.created_by, "ends_at": silence.ends_at.isoformat()}
        )
        return silence.silence_id

    def remove_silence(self, silence_id: str) -> bool:
        """Remove a silence."""
        if silence_id in self._silences:
            del self._silences[silence_id]
            return True
        return False

    def register_notification_handler(
        self,
        receiver: str,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Register a notification handler for a receiver.

        Args:
            receiver: Receiver identifier
            handler: Callable that receives an Alert
        """
        self._notification_handlers[receiver] = handler
        logger.info(f"Registered notification handler: {receiver}")

    def evaluate(
        self,
        asset_id: str,
        parameter: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Alert]:
        """
        Evaluate rules against a parameter value.

        Args:
            asset_id: Asset identifier
            parameter: Parameter name
            value: Current value
            labels: Additional labels

        Returns:
            Alert if any rule fires, None otherwise
        """
        labels = labels or {}
        highest_severity_alert: Optional[Alert] = None

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.parameter != parameter:
                continue

            if rule.evaluate(value):
                alert = Alert(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    asset_id=asset_id,
                    parameter=parameter,
                    severity=rule.severity,
                    current_value=value,
                    threshold=rule.threshold,
                    operator=rule.operator,
                    labels={**rule.labels, **labels},
                    annotations=rule.annotations,
                )

                # Check for silences
                for silence in self._silences.values():
                    if silence.matches(alert):
                        alert.state = AlertState.SILENCED
                        break

                # Track highest severity
                if highest_severity_alert is None:
                    highest_severity_alert = alert
                elif self._compare_severity(alert.severity, highest_severity_alert.severity) > 0:
                    highest_severity_alert = alert

        if highest_severity_alert:
            self._handle_alert(highest_severity_alert)
            return highest_severity_alert

        # Check if we should resolve any active alerts
        self._check_resolved(asset_id, parameter, value)

        return None

    def _compare_severity(self, s1: AlertSeverity, s2: AlertSeverity) -> int:
        """Compare two severities. Returns positive if s1 > s2."""
        order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
            AlertSeverity.EMERGENCY: 3,
        }
        return order[s1] - order[s2]

    def _handle_alert(self, alert: Alert) -> None:
        """Handle a new or existing alert."""
        fingerprint = alert.fingerprint

        if fingerprint in self._active_alerts:
            # Update existing alert
            existing = self._active_alerts[fingerprint]
            existing.current_value = alert.current_value
            if alert.state == AlertState.SILENCED:
                existing.state = AlertState.SILENCED
        else:
            # New alert
            self._active_alerts[fingerprint] = alert

            if alert.state != AlertState.SILENCED:
                self.route_alert(alert)
                self._alert_history.append(alert)

            logger.warning(
                f"Alert firing: {alert.rule_name}",
                extra={
                    "alert_id": alert.alert_id,
                    "asset_id": alert.asset_id,
                    "parameter": alert.parameter,
                    "value": alert.current_value,
                    "threshold": alert.threshold,
                    "severity": alert.severity.value,
                }
            )

    def _check_resolved(self, asset_id: str, parameter: str, value: float) -> None:
        """Check if any active alerts should be resolved."""
        to_remove = []

        for fingerprint, alert in self._active_alerts.items():
            if alert.asset_id != asset_id or alert.parameter != parameter:
                continue

            rule = self._rules.get(alert.rule_id)
            if rule and not rule.evaluate(value):
                # Alert condition no longer met
                alert.state = AlertState.RESOLVED
                alert.resolved_time = datetime.now(timezone.utc)
                to_remove.append(fingerprint)

                logger.info(
                    f"Alert resolved: {alert.rule_name}",
                    extra={
                        "alert_id": alert.alert_id,
                        "asset_id": alert.asset_id,
                        "duration_seconds": (
                            alert.resolved_time - alert.firing_time
                        ).total_seconds(),
                    }
                )

        for fingerprint in to_remove:
            del self._active_alerts[fingerprint]

    def route_alert(self, alert: Alert) -> List[str]:
        """
        Route an alert to appropriate receivers.

        Args:
            alert: Alert to route

        Returns:
            List of receivers notified
        """
        notified = []

        for route in self._routes:
            if route.matches(alert):
                for receiver in route.receivers:
                    if receiver not in notified:
                        self._notify(alert, receiver)
                        notified.append(receiver)
                        alert.routed_to.append(receiver)

                if not route.continue_routing:
                    break

        alert.notification_sent = len(notified) > 0
        return notified

    def _notify(self, alert: Alert, receiver: str) -> None:
        """Send notification to a receiver."""
        handler = self._notification_handlers.get(receiver)
        if handler:
            try:
                handler(alert)
                logger.info(f"Notified {receiver} about alert {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to notify {receiver}: {e}")
        else:
            # Log-only notification
            logger.info(
                f"Alert notification for {receiver}: {alert.rule_name} - "
                f"{alert.parameter}={alert.current_value} (threshold: {alert.threshold})"
            )

    def acknowledge(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: User who acknowledged

        Returns:
            True if acknowledged successfully
        """
        for alert in self._active_alerts.values():
            if alert.alert_id == alert_id:
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_time = datetime.now(timezone.utc)
                alert.labels["acknowledged_by"] = acknowledged_by
                logger.info(
                    f"Alert acknowledged: {alert_id} by {acknowledged_by}"
                )
                return True
        return False

    def get_active_alerts(
        self,
        asset_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get currently active alerts."""
        alerts = list(self._active_alerts.values())

        if asset_id:
            alerts = [a for a in alerts if a.asset_id == asset_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.firing_time, reverse=True)

    def get_alert_history(
        self,
        asset_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get historical alerts."""
        alerts = self._alert_history.copy()

        if asset_id:
            alerts = [a for a in alerts if a.asset_id == asset_id]

        if start_time:
            alerts = [a for a in alerts if a.firing_time >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.firing_time <= end_time]

        return sorted(alerts, key=lambda a: a.firing_time, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active = list(self._active_alerts.values())
        history = self._alert_history

        by_severity = {}
        for sev in AlertSeverity:
            by_severity[sev.value] = {
                "active": len([a for a in active if a.severity == sev]),
                "total": len([a for a in history if a.severity == sev]),
            }

        return {
            "active_alerts": len(active),
            "total_alerts": len(history),
            "rules_count": len(self._rules),
            "silences_count": len([s for s in self._silences.values() if s.is_active()]),
            "by_severity": by_severity,
        }

    def cleanup_expired_silences(self) -> int:
        """Remove expired silences."""
        now = datetime.now(timezone.utc)
        to_remove = [
            sid for sid, silence in self._silences.items()
            if silence.ends_at < now
        ]

        for sid in to_remove:
            del self._silences[sid]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} expired silences")

        return len(to_remove)
