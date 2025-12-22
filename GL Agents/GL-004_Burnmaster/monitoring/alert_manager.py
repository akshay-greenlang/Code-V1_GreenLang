"""
GL-004 BURNMASTER Alert Manager Module

This module provides comprehensive alert management for combustion optimization
operations, including alert rule definition, evaluation, notification dispatch,
acknowledgment, and escalation workflows.

Example:
    >>> manager = AlertManager()
    >>> rule = AlertRule(name="high_nox", condition="nox_ppm > 50")
    >>> rule_id = manager.define_alert_rule(rule)
    >>> alerts = manager.evaluate_alerts(burner_state)
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import logging
import re
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AlertLevel(str, Enum):
    """Alert severity levels for combustion monitoring."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertStatus(str, Enum):
    """Alert lifecycle status."""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"
    ESCALATED = "ESCALATED"


class NotificationChannel(str, Enum):
    """Available notification channels."""
    EMAIL = "EMAIL"
    SMS = "SMS"
    SLACK = "SLACK"
    PAGERDUTY = "PAGERDUTY"
    WEBHOOK = "WEBHOOK"
    OPC_ALARM = "OPC_ALARM"
    DCS_CONSOLE = "DCS_CONSOLE"


# =============================================================================
# DATA MODELS
# =============================================================================

class BurnerState(BaseModel):
    """Current state of a burner unit for alert evaluation."""

    unit_id: str = Field(..., description="Burner unit identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State timestamp"
    )

    # Combustion parameters
    air_fuel_ratio: float = Field(..., ge=0.5, le=3.0, description="Air-fuel ratio")
    o2_percent: float = Field(..., ge=0.0, le=21.0, description="Oxygen percentage")
    co_ppm: float = Field(..., ge=0.0, description="CO in parts per million")
    nox_ppm: float = Field(..., ge=0.0, description="NOx in parts per million")

    # Temperatures
    flame_temp_c: float = Field(..., ge=0.0, description="Flame temperature Celsius")
    stack_temp_c: float = Field(..., ge=0.0, description="Stack temperature Celsius")

    # Operational
    firing_rate_percent: float = Field(..., ge=0.0, le=100.0, description="Firing rate")
    burner_status: str = Field(..., description="Burner operational status")

    # Safety
    flame_detected: bool = Field(default=True, description="Flame detection status")
    safety_interlocks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Safety interlock states"
    )

    # Performance
    efficiency_percent: float = Field(..., ge=0.0, le=100.0, description="Combustion efficiency")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Flame stability 0-1")

    # Extended data
    extended_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional sensor data"
    )

    class Config:
        use_enum_values = True


class AlertRule(BaseModel):
    """Definition of an alert rule for combustion monitoring."""

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique rule identifier"
    )
    name: str = Field(..., min_length=1, max_length=100, description="Rule name")
    description: str = Field(default="", description="Rule description")

    # Condition
    condition: str = Field(..., description="Alert condition expression")
    level: AlertLevel = Field(..., description="Alert severity level")

    # Targeting
    unit_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific units (None = all)"
    )

    # Timing
    holdoff_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Seconds condition must persist"
    )
    repeat_interval_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="Minimum seconds between repeat alerts"
    )

    # Notifications
    notification_channels: List[NotificationChannel] = Field(
        default_factory=list,
        description="Channels to notify"
    )
    escalation_chain: List[str] = Field(
        default_factory=list,
        description="Escalation user/group chain"
    )

    # Metadata
    enabled: bool = Field(default=True, description="Rule enabled state")
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Rule tags for categorization"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rule creation timestamp"
    )
    created_by: str = Field(default="system", description="Rule creator")

    @validator('condition')
    def validate_condition(cls, v: str) -> str:
        """Validate condition is parseable."""
        # Basic validation - check for common operators
        valid_operators = ['>', '<', '>=', '<=', '==', '!=', 'and', 'or', 'not']
        tokens = v.lower().split()
        if not any(op in v.lower() for op in valid_operators):
            raise ValueError(f"Condition must contain comparison operator: {valid_operators}")
        return v

    class Config:
        use_enum_values = True


class Alert(BaseModel):
    """An active or historical alert instance."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique alert identifier"
    )
    rule_id: str = Field(..., description="Source rule ID")
    rule_name: str = Field(..., description="Source rule name")

    # Alert details
    level: AlertLevel = Field(..., description="Alert severity")
    status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="Alert status")
    message: str = Field(..., description="Alert message")

    # Context
    unit_id: str = Field(..., description="Affected burner unit")
    triggered_value: float = Field(..., description="Value that triggered alert")
    threshold_value: Optional[float] = Field(None, description="Threshold value")
    condition_expression: str = Field(..., description="Condition that triggered")

    # Timing
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert trigger time"
    )
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")

    # Acknowledgment
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    acknowledgment_notes: Optional[str] = Field(None, description="Acknowledgment notes")

    # Escalation
    escalation_level: int = Field(default=0, ge=0, description="Current escalation level")
    escalated_to: Optional[str] = Field(None, description="Current escalation target")

    # Audit
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    # Additional data
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert metadata"
    )

    def __init__(self, **data):
        """Initialize alert and compute provenance hash."""
        super().__init__(**data)
        if not self.provenance_hash:
            self.provenance_hash = self._compute_provenance()

    def _compute_provenance(self) -> str:
        """Compute SHA-256 provenance hash."""
        content = f"{self.rule_id}:{self.unit_id}:{self.triggered_at.isoformat()}"
        content += f":{self.triggered_value}:{self.condition_expression}"
        return hashlib.sha256(content.encode()).hexdigest()

    class Config:
        use_enum_values = True


class SendResult(BaseModel):
    """Result of sending an alert notification."""

    success: bool = Field(..., description="Overall success")
    alert_id: str = Field(..., description="Alert ID")
    channels_attempted: List[str] = Field(..., description="Channels attempted")
    channels_succeeded: List[str] = Field(..., description="Channels succeeded")
    channels_failed: List[str] = Field(..., description="Channels failed")
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Error messages per channel"
    )
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Send timestamp"
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Send latency")


class AckResult(BaseModel):
    """Result of acknowledging an alert."""

    success: bool = Field(..., description="Acknowledgment success")
    alert_id: str = Field(..., description="Alert ID")
    acknowledged_by: str = Field(..., description="User who acknowledged")
    acknowledged_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Acknowledgment timestamp"
    )
    previous_status: AlertStatus = Field(..., description="Previous alert status")
    new_status: AlertStatus = Field(..., description="New alert status")
    error: Optional[str] = Field(None, description="Error if failed")


class EscalationResult(BaseModel):
    """Result of escalating an alert."""

    success: bool = Field(..., description="Escalation success")
    alert_id: str = Field(..., description="Alert ID")
    previous_level: AlertLevel = Field(..., description="Previous alert level")
    new_level: AlertLevel = Field(..., description="New alert level")
    escalated_to: str = Field(..., description="Escalation target")
    escalated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Escalation timestamp"
    )
    notification_sent: bool = Field(default=False, description="Notification sent")
    error: Optional[str] = Field(None, description="Error if failed")


# =============================================================================
# CONDITION EVALUATOR
# =============================================================================

class ConditionEvaluator:
    """
    Safe evaluator for alert conditions.

    Supports basic comparisons and logical operators without eval().
    """

    OPERATORS = {
        '>': lambda a, b: a > b,
        '<': lambda a, b: a < b,
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }

    def __init__(self):
        """Initialize the condition evaluator."""
        self._compiled_conditions: Dict[str, Callable] = {}

    def evaluate(self, condition: str, state: BurnerState) -> tuple[bool, Optional[float]]:
        """
        Evaluate a condition against burner state.

        Args:
            condition: Condition expression (e.g., "nox_ppm > 50")
            state: Current burner state

        Returns:
            Tuple of (condition_met, triggered_value)
        """
        try:
            # Parse condition into parts
            # Support: field operator value (e.g., "nox_ppm > 50")
            # Support: field1 op1 val1 and field2 op2 val2

            state_dict = state.dict()

            # Handle simple conditions
            for op_str, op_func in sorted(
                self.OPERATORS.items(),
                key=lambda x: len(x[0]),
                reverse=True
            ):
                if op_str in condition:
                    parts = condition.split(op_str)
                    if len(parts) == 2:
                        field = parts[0].strip()
                        threshold = float(parts[1].strip())

                        # Get value from state
                        value = self._get_nested_value(state_dict, field)
                        if value is not None:
                            result = op_func(float(value), threshold)
                            return result, float(value)

            # Handle compound conditions (and/or)
            if ' and ' in condition.lower():
                sub_conditions = re.split(r'\s+and\s+', condition, flags=re.IGNORECASE)
                results = []
                triggered_val = None
                for sub in sub_conditions:
                    met, val = self.evaluate(sub.strip(), state)
                    results.append(met)
                    if met and triggered_val is None:
                        triggered_val = val
                return all(results), triggered_val

            if ' or ' in condition.lower():
                sub_conditions = re.split(r'\s+or\s+', condition, flags=re.IGNORECASE)
                for sub in sub_conditions:
                    met, val = self.evaluate(sub.strip(), state)
                    if met:
                        return True, val
                return False, None

            logger.warning(f"Could not parse condition: {condition}")
            return False, None

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False, None

    def _get_nested_value(self, data: Dict, field: str) -> Optional[Any]:
        """Get possibly nested value from dict."""
        parts = field.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current


# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """
    Comprehensive alert management for combustion optimization.

    Handles alert rule definition, evaluation against burner state,
    notification dispatch, acknowledgment, and escalation workflows.

    Attributes:
        rules: Dictionary of alert rules by rule_id
        active_alerts: Dictionary of active alerts by alert_id

    Example:
        >>> manager = AlertManager()
        >>> rule = AlertRule(
        ...     name="high_nox",
        ...     condition="nox_ppm > 50",
        ...     level=AlertLevel.WARNING
        ... )
        >>> rule_id = manager.define_alert_rule(rule)
        >>> alerts = manager.evaluate_alerts(burner_state)
    """

    def __init__(self):
        """Initialize the AlertManager."""
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._condition_evaluator = ConditionEvaluator()
        self._last_alert_times: Dict[str, datetime] = {}  # rule_id:unit_id -> last alert
        self._condition_start_times: Dict[str, datetime] = {}  # For holdoff tracking
        self._notification_handlers: Dict[NotificationChannel, Callable] = {}

        logger.info("AlertManager initialized")

    def define_alert_rule(self, rule: AlertRule) -> str:
        """
        Define a new alert rule.

        Args:
            rule: AlertRule configuration

        Returns:
            The rule_id of the defined rule

        Raises:
            ValueError: If rule with same ID already exists
        """
        if rule.rule_id in self._rules:
            raise ValueError(f"Rule with ID {rule.rule_id} already exists")

        self._rules[rule.rule_id] = rule
        logger.info(
            f"Defined alert rule: {rule.name} (ID: {rule.rule_id}), "
            f"level={rule.level}, condition='{rule.condition}'"
        )
        return rule.rule_id

    def evaluate_alerts(self, state: BurnerState) -> List[Alert]:
        """
        Evaluate all enabled rules against current burner state.

        Args:
            state: Current burner state to evaluate

        Returns:
            List of newly triggered alerts
        """
        triggered_alerts: List[Alert] = []

        for rule_id, rule in self._rules.items():
            if not rule.enabled:
                continue

            # Check if rule applies to this unit
            if rule.unit_ids and state.unit_id not in rule.unit_ids:
                continue

            # Evaluate condition
            condition_met, triggered_value = self._condition_evaluator.evaluate(
                rule.condition, state
            )

            if condition_met:
                # Check holdoff
                holdoff_key = f"{rule_id}:{state.unit_id}"
                now = datetime.now(timezone.utc)

                if holdoff_key not in self._condition_start_times:
                    self._condition_start_times[holdoff_key] = now
                    continue  # Start holdoff timer

                elapsed = (now - self._condition_start_times[holdoff_key]).total_seconds()
                if elapsed < rule.holdoff_seconds:
                    continue  # Still in holdoff

                # Check repeat interval
                last_alert_key = f"{rule_id}:{state.unit_id}"
                if last_alert_key in self._last_alert_times:
                    since_last = (now - self._last_alert_times[last_alert_key]).total_seconds()
                    if since_last < rule.repeat_interval_seconds:
                        continue  # Too soon to repeat

                # Create alert
                alert = self._create_alert(rule, state, triggered_value)
                self._active_alerts[alert.alert_id] = alert
                self._last_alert_times[last_alert_key] = now
                triggered_alerts.append(alert)

                logger.warning(
                    f"Alert triggered: {alert.rule_name} for unit {alert.unit_id}, "
                    f"value={triggered_value}, level={alert.level}"
                )
            else:
                # Clear holdoff timer if condition not met
                holdoff_key = f"{rule_id}:{state.unit_id}"
                self._condition_start_times.pop(holdoff_key, None)

        return triggered_alerts

    def _create_alert(
        self,
        rule: AlertRule,
        state: BurnerState,
        triggered_value: Optional[float]
    ) -> Alert:
        """Create an Alert instance from a triggered rule."""
        # Extract threshold from condition for display
        threshold = self._extract_threshold(rule.condition)

        message = (
            f"{rule.name}: {rule.condition} triggered for unit {state.unit_id}. "
            f"Current value: {triggered_value}"
        )

        return Alert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            level=rule.level,
            message=message,
            unit_id=state.unit_id,
            triggered_value=triggered_value or 0.0,
            threshold_value=threshold,
            condition_expression=rule.condition,
            metadata={
                "burner_status": state.burner_status,
                "firing_rate": state.firing_rate_percent,
                "efficiency": state.efficiency_percent,
            }
        )

    def _extract_threshold(self, condition: str) -> Optional[float]:
        """Extract numeric threshold from condition string."""
        # Find numbers in condition
        numbers = re.findall(r'[-+]?\d*\.?\d+', condition)
        if numbers:
            return float(numbers[-1])
        return None

    def send_alert(
        self,
        alert: Alert,
        channels: List[str]
    ) -> SendResult:
        """
        Send alert notification to specified channels.

        Args:
            alert: Alert to send
            channels: List of channel names to notify

        Returns:
            SendResult with success/failure details
        """
        start_time = datetime.now(timezone.utc)
        channels_attempted = channels
        channels_succeeded = []
        channels_failed = []
        errors: Dict[str, str] = {}

        for channel_name in channels:
            try:
                channel = NotificationChannel(channel_name.upper())

                # Check for registered handler
                if channel in self._notification_handlers:
                    handler = self._notification_handlers[channel]
                    handler(alert)
                    channels_succeeded.append(channel_name)
                else:
                    # Log notification (default behavior)
                    logger.info(
                        f"Alert notification [{channel_name}]: "
                        f"{alert.level} - {alert.message}"
                    )
                    channels_succeeded.append(channel_name)

            except Exception as e:
                channels_failed.append(channel_name)
                errors[channel_name] = str(e)
                logger.error(f"Failed to send alert via {channel_name}: {e}")

        latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return SendResult(
            success=len(channels_failed) == 0,
            alert_id=alert.alert_id,
            channels_attempted=channels_attempted,
            channels_succeeded=channels_succeeded,
            channels_failed=channels_failed,
            errors=errors,
            latency_ms=latency
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        user: str,
        notes: Optional[str] = None
    ) -> AckResult:
        """
        Acknowledge an active alert.

        Args:
            alert_id: ID of alert to acknowledge
            user: User performing acknowledgment
            notes: Optional acknowledgment notes

        Returns:
            AckResult with success/failure details
        """
        if alert_id not in self._active_alerts:
            return AckResult(
                success=False,
                alert_id=alert_id,
                acknowledged_by=user,
                previous_status=AlertStatus.ACTIVE,
                new_status=AlertStatus.ACTIVE,
                error=f"Alert {alert_id} not found in active alerts"
            )

        alert = self._active_alerts[alert_id]
        previous_status = alert.status

        # Update alert
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = user
        if notes:
            alert.acknowledgment_notes = notes

        logger.info(
            f"Alert {alert_id} acknowledged by {user}. "
            f"Previous status: {previous_status}, Notes: {notes}"
        )

        return AckResult(
            success=True,
            alert_id=alert_id,
            acknowledged_by=user,
            acknowledged_at=alert.acknowledged_at,
            previous_status=previous_status,
            new_status=AlertStatus.ACKNOWLEDGED
        )

    def escalate_alert(
        self,
        alert_id: str,
        level: AlertLevel
    ) -> EscalationResult:
        """
        Escalate an alert to a higher severity level.

        Args:
            alert_id: ID of alert to escalate
            level: New alert level

        Returns:
            EscalationResult with success/failure details
        """
        if alert_id not in self._active_alerts:
            return EscalationResult(
                success=False,
                alert_id=alert_id,
                previous_level=AlertLevel.INFO,
                new_level=level,
                escalated_to="",
                error=f"Alert {alert_id} not found"
            )

        alert = self._active_alerts[alert_id]
        previous_level = alert.level

        # Get rule for escalation chain
        rule = self._rules.get(alert.rule_id)
        escalation_target = ""

        if rule and rule.escalation_chain:
            next_idx = min(alert.escalation_level, len(rule.escalation_chain) - 1)
            escalation_target = rule.escalation_chain[next_idx]

        # Update alert
        alert.level = level
        alert.status = AlertStatus.ESCALATED
        alert.escalation_level += 1
        alert.escalated_to = escalation_target

        # Send escalation notification
        notification_sent = False
        if rule and rule.notification_channels:
            send_result = self.send_alert(
                alert,
                [ch.value for ch in rule.notification_channels]
            )
            notification_sent = send_result.success

        logger.warning(
            f"Alert {alert_id} escalated from {previous_level} to {level}, "
            f"target: {escalation_target}"
        )

        return EscalationResult(
            success=True,
            alert_id=alert_id,
            previous_level=previous_level,
            new_level=level,
            escalated_to=escalation_target,
            escalated_at=datetime.now(timezone.utc),
            notification_sent=notification_sent
        )

    def get_active_alerts(self, unit_id: str) -> List[Alert]:
        """
        Get all active alerts for a specific unit.

        Args:
            unit_id: Unit identifier to filter by

        Returns:
            List of active alerts for the unit
        """
        return [
            alert for alert in self._active_alerts.values()
            if alert.unit_id == unit_id and alert.status in [
                AlertStatus.ACTIVE,
                AlertStatus.ACKNOWLEDGED,
                AlertStatus.ESCALATED
            ]
        ]

    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: ID of alert to resolve
            user: User resolving the alert

        Returns:
            True if resolved successfully
        """
        if alert_id not in self._active_alerts:
            return False

        alert = self._active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)

        # Move to history
        self._alert_history.append(alert)
        del self._active_alerts[alert_id]

        logger.info(f"Alert {alert_id} resolved by {user}")
        return True

    def register_notification_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Alert], None]
    ) -> None:
        """
        Register a custom notification handler for a channel.

        Args:
            channel: Notification channel
            handler: Callback function for sending notifications
        """
        self._notification_handlers[channel] = handler
        logger.info(f"Registered notification handler for {channel.value}")

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> List[AlertRule]:
        """Get all defined alert rules."""
        return list(self._rules.values())

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
            return True
        return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics summary."""
        active_by_level = {}
        for alert in self._active_alerts.values():
            level = alert.level.value if isinstance(alert.level, AlertLevel) else alert.level
            active_by_level[level] = active_by_level.get(level, 0) + 1

        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "active_alerts": len(self._active_alerts),
            "active_by_level": active_by_level,
            "total_historical": len(self._alert_history),
        }
