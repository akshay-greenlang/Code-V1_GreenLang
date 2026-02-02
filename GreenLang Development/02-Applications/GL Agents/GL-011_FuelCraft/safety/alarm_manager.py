# -*- coding: utf-8 -*-
"""
AlarmManager - Alarm handling and escalation for GL-011 FuelCraft.

This module implements alarm management including severity classification,
operator review triggers, recommendation blocking policies, and escalation rules.

Reference Standards:
    - ISA-18.2: Management of Alarm Systems
    - IEC 62682: Management of Alarm Systems
    - EEMUA 191: Alarm Systems - A Guide to Design, Management and Procurement

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
from threading import Lock
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class AlarmSeverity(str, Enum):
    """Alarm severity levels per ISA-18.2."""
    CRITICAL = "critical"     # Immediate action required
    HIGH = "high"             # Prompt action required
    MEDIUM = "medium"         # Action required
    LOW = "low"               # Awareness only
    DIAGNOSTIC = "diagnostic" # Information only


class AlarmState(str, Enum):
    """Alarm states per ISA-18.2."""
    ACTIVE_UNACKNOWLEDGED = "active_unack"
    ACTIVE_ACKNOWLEDGED = "active_ack"
    CLEARED_UNACKNOWLEDGED = "cleared_unack"
    CLEARED_ACKNOWLEDGED = "cleared_ack"
    SUPPRESSED = "suppressed"
    SHELVED = "shelved"


class AlarmPriority(int, Enum):
    """Alarm priority for processing order."""
    P1 = 1  # Highest - Immediate
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5  # Lowest - Information


class EscalationLevel(str, Enum):
    """Escalation levels."""
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    MANAGER = "manager"
    EMERGENCY = "emergency"


class AlarmDefinition(BaseModel):
    """Definition of an alarm type."""
    alarm_id: str = Field(...)
    alarm_name: str = Field(...)
    description: str = Field(...)
    severity: AlarmSeverity = Field(...)
    priority: AlarmPriority = Field(...)
    equipment_id: Optional[str] = Field(None)
    process_area: Optional[str] = Field(None)
    parameter: str = Field(...)
    setpoint_high: Optional[float] = Field(None)
    setpoint_low: Optional[float] = Field(None)
    deadband: float = Field(0.0, ge=0)
    delay_seconds: float = Field(0.0, ge=0)
    requires_acknowledgement: bool = Field(True)
    blocks_recommendations: bool = Field(False)
    reference_standard: Optional[str] = Field(None)


class AlarmInstance(BaseModel):
    """Instance of an active or historical alarm."""
    instance_id: str = Field(...)
    alarm_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: AlarmState = Field(AlarmState.ACTIVE_UNACKNOWLEDGED)
    severity: AlarmSeverity = Field(...)
    priority: AlarmPriority = Field(...)
    current_value: float = Field(...)
    setpoint: float = Field(...)
    deviation: float = Field(...)
    message: str = Field(...)
    acknowledged_by: Optional[str] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)
    cleared_at: Optional[datetime] = Field(None)
    escalation_level: EscalationLevel = Field(EscalationLevel.OPERATOR)
    provenance_hash: str = Field(...)


class EscalationRule(BaseModel):
    """Rule for alarm escalation."""
    rule_id: str = Field(...)
    alarm_severity: AlarmSeverity = Field(...)
    time_to_escalate_minutes: int = Field(..., ge=1)
    from_level: EscalationLevel = Field(...)
    to_level: EscalationLevel = Field(...)
    notification_method: str = Field("email")


class AlarmStatistics(BaseModel):
    """Alarm system statistics per EEMUA 191."""
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    total_alarms: int = Field(0)
    alarms_per_hour: float = Field(0.0)
    critical_alarms: int = Field(0)
    high_alarms: int = Field(0)
    unacknowledged_count: int = Field(0)
    average_response_time_seconds: Optional[float] = Field(None)
    standing_alarms: int = Field(0)
    chattering_alarms: List[str] = Field(default_factory=list)
    flood_periods: int = Field(0)


class OperatorReviewTrigger(BaseModel):
    """Trigger configuration for operator review requirement."""
    trigger_id: str = Field(...)
    trigger_name: str = Field(...)
    description: str = Field(...)
    condition_type: str = Field(...)  # alarm_count, severity, parameter_deviation
    threshold: float = Field(...)
    time_window_minutes: int = Field(5)
    blocks_until_reviewed: bool = Field(True)
    severity_filter: Optional[AlarmSeverity] = Field(None)


class RecommendationBlockPolicy(BaseModel):
    """Policy for blocking recommendations based on alarm conditions."""
    policy_id: str = Field(...)
    policy_name: str = Field(...)
    description: str = Field(...)
    enabled: bool = Field(True)

    # Block conditions
    block_on_critical_alarm: bool = Field(True)
    block_on_high_alarm: bool = Field(True)
    block_on_unack_count: Optional[int] = Field(3)
    block_on_alarm_flood: bool = Field(True)

    # Release conditions
    release_after_ack: bool = Field(True)
    release_after_cleared: bool = Field(False)
    require_supervisor_override: bool = Field(False)


class AlarmManager:
    """
    Alarm management system per ISA-18.2 and EEMUA 191.

    Provides alarm creation, acknowledgement, escalation, and analytics.
    Integrates with recommendation blocking policy for safety.
    """

    def __init__(
        self,
        recommendation_block_policy: Optional[RecommendationBlockPolicy] = None,
        escalation_rules: Optional[List[EscalationRule]] = None
    ):
        """Initialize alarm manager."""
        self._lock = Lock()
        self._alarm_definitions: Dict[str, AlarmDefinition] = {}
        self._active_alarms: Dict[str, AlarmInstance] = {}
        self._alarm_history: deque = deque(maxlen=10000)
        self._escalation_rules = escalation_rules or self._default_escalation_rules()
        self._block_policy = recommendation_block_policy or self._default_block_policy()
        self._operator_triggers: Dict[str, OperatorReviewTrigger] = {}
        self._review_pending: bool = False
        self._review_pending_reason: Optional[str] = None

        logger.info("AlarmManager initialized per ISA-18.2")

    def register_alarm(self, definition: AlarmDefinition) -> None:
        """Register an alarm definition."""
        with self._lock:
            self._alarm_definitions[definition.alarm_id] = definition
            logger.info(f"Registered alarm: {definition.alarm_id} ({definition.severity.value})")

    def register_operator_trigger(self, trigger: OperatorReviewTrigger) -> None:
        """Register an operator review trigger."""
        with self._lock:
            self._operator_triggers[trigger.trigger_id] = trigger
            logger.info(f"Registered operator trigger: {trigger.trigger_id}")

    def raise_alarm(
        self,
        alarm_id: str,
        current_value: float,
        message: Optional[str] = None
    ) -> Optional[AlarmInstance]:
        """
        Raise an alarm instance.

        Returns the alarm instance if created, None if alarm is already active.
        """
        with self._lock:
            definition = self._alarm_definitions.get(alarm_id)
            if definition is None:
                logger.error(f"Unknown alarm ID: {alarm_id}")
                return None

            if alarm_id in self._active_alarms:
                existing = self._active_alarms[alarm_id]
                existing_dict = existing.model_dump()
                existing_dict['current_value'] = current_value
                self._active_alarms[alarm_id] = AlarmInstance(**existing_dict)
                return None

            setpoint = definition.setpoint_high or definition.setpoint_low or 0.0
            deviation = abs(current_value - setpoint)

            instance_id = hashlib.sha256(
                f"{alarm_id}|{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:16]

            alarm_message = message or f"{definition.alarm_name}: value={current_value}, setpoint={setpoint}"

            instance = AlarmInstance(
                instance_id=instance_id,
                alarm_id=alarm_id,
                severity=definition.severity,
                priority=definition.priority,
                current_value=current_value,
                setpoint=setpoint,
                deviation=deviation,
                message=alarm_message,
                provenance_hash=hashlib.sha256(
                    json.dumps({
                        "instance_id": instance_id, "alarm_id": alarm_id,
                        "value": current_value, "severity": definition.severity.value
                    }, sort_keys=True).encode()
                ).hexdigest()
            )

            self._active_alarms[alarm_id] = instance
            self._alarm_history.append(instance)

            self._check_operator_triggers()

            log_level = logging.CRITICAL if definition.severity == AlarmSeverity.CRITICAL else (
                logging.WARNING if definition.severity in [AlarmSeverity.HIGH, AlarmSeverity.MEDIUM] else logging.INFO
            )
            logger.log(
                log_level,
                f"[ALARM] {definition.severity.value.upper()}: {alarm_message} "
                f"(ID: {instance_id})"
            )

            return instance

    def acknowledge_alarm(
        self,
        alarm_id: str,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an active alarm."""
        with self._lock:
            if alarm_id not in self._active_alarms:
                return False

            alarm = self._active_alarms[alarm_id]
            if alarm.state not in [AlarmState.ACTIVE_UNACKNOWLEDGED, AlarmState.CLEARED_UNACKNOWLEDGED]:
                return False

            new_state = (
                AlarmState.ACTIVE_ACKNOWLEDGED
                if alarm.state == AlarmState.ACTIVE_UNACKNOWLEDGED
                else AlarmState.CLEARED_ACKNOWLEDGED
            )

            alarm_dict = alarm.model_dump()
            alarm_dict['state'] = new_state
            alarm_dict['acknowledged_by'] = acknowledged_by
            alarm_dict['acknowledged_at'] = datetime.now(timezone.utc)

            self._active_alarms[alarm_id] = AlarmInstance(**alarm_dict)

            self._check_review_release()

            logger.info(f"[ALARM] Acknowledged: {alarm_id} by {acknowledged_by}")
            return True

    def clear_alarm(self, alarm_id: str) -> bool:
        """Clear an alarm when condition returns to normal."""
        with self._lock:
            if alarm_id not in self._active_alarms:
                return False

            alarm = self._active_alarms[alarm_id]
            definition = self._alarm_definitions.get(alarm_id)

            if definition and definition.requires_acknowledgement:
                if alarm.state == AlarmState.ACTIVE_UNACKNOWLEDGED:
                    alarm_dict = alarm.model_dump()
                    alarm_dict['state'] = AlarmState.CLEARED_UNACKNOWLEDGED
                    alarm_dict['cleared_at'] = datetime.now(timezone.utc)
                    self._active_alarms[alarm_id] = AlarmInstance(**alarm_dict)
                elif alarm.state == AlarmState.ACTIVE_ACKNOWLEDGED:
                    alarm_dict = alarm.model_dump()
                    alarm_dict['state'] = AlarmState.CLEARED_ACKNOWLEDGED
                    alarm_dict['cleared_at'] = datetime.now(timezone.utc)
                    del self._active_alarms[alarm_id]
            else:
                del self._active_alarms[alarm_id]

            logger.info(f"[ALARM] Cleared: {alarm_id}")
            return True

    def shelve_alarm(
        self,
        alarm_id: str,
        shelved_by: str,
        duration_minutes: int,
        reason: str
    ) -> bool:
        """Shelve an alarm temporarily."""
        with self._lock:
            if alarm_id not in self._active_alarms:
                return False

            alarm = self._active_alarms[alarm_id]
            alarm_dict = alarm.model_dump()
            alarm_dict['state'] = AlarmState.SHELVED

            self._active_alarms[alarm_id] = AlarmInstance(**alarm_dict)

            logger.warning(
                f"[ALARM] Shelved: {alarm_id} by {shelved_by} for {duration_minutes}min: {reason}"
            )
            return True

    def is_recommendations_blocked(self) -> tuple[bool, Optional[str]]:
        """
        Check if recommendations are blocked due to alarm conditions.

        Returns (is_blocked, reason).
        """
        with self._lock:
            if not self._block_policy.enabled:
                return False, None

            if self._review_pending:
                return True, self._review_pending_reason

            for alarm in self._active_alarms.values():
                if alarm.state == AlarmState.SHELVED:
                    continue

                if self._block_policy.block_on_critical_alarm and alarm.severity == AlarmSeverity.CRITICAL:
                    if alarm.state in [AlarmState.ACTIVE_UNACKNOWLEDGED, AlarmState.ACTIVE_ACKNOWLEDGED]:
                        return True, f"Critical alarm active: {alarm.alarm_id}"

                if self._block_policy.block_on_high_alarm and alarm.severity == AlarmSeverity.HIGH:
                    if alarm.state == AlarmState.ACTIVE_UNACKNOWLEDGED:
                        return True, f"High alarm unacknowledged: {alarm.alarm_id}"

            unack_count = sum(
                1 for a in self._active_alarms.values()
                if a.state == AlarmState.ACTIVE_UNACKNOWLEDGED and a.state != AlarmState.SHELVED
            )

            if self._block_policy.block_on_unack_count and unack_count >= self._block_policy.block_on_unack_count:
                return True, f"Too many unacknowledged alarms: {unack_count}"

            return False, None

    def override_block(self, authorized_by: str, reason: str) -> bool:
        """Override recommendation block (supervisor only)."""
        if not self._block_policy.require_supervisor_override:
            logger.warning(
                f"[ALARM] Block override by {authorized_by}: {reason}"
            )
            return True

        logger.warning(
            f"[ALARM] Block override REQUIRES SUPERVISOR - requested by {authorized_by}: {reason}"
        )
        return False

    def get_active_alarms(self, severity: Optional[AlarmSeverity] = None) -> List[AlarmInstance]:
        """Get list of active alarms, optionally filtered by severity."""
        with self._lock:
            alarms = list(self._active_alarms.values())
            if severity:
                alarms = [a for a in alarms if a.severity == severity]
            return sorted(alarms, key=lambda a: (a.priority.value, a.timestamp))

    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alarms."""
        with self._lock:
            return sum(
                1 for a in self._active_alarms.values()
                if a.state in [AlarmState.ACTIVE_UNACKNOWLEDGED, AlarmState.CLEARED_UNACKNOWLEDGED]
            )

    def get_statistics(self, period_minutes: int = 60) -> AlarmStatistics:
        """Get alarm statistics for the specified period."""
        with self._lock:
            now = datetime.now(timezone.utc)
            period_start = now - timedelta(minutes=period_minutes)

            recent_alarms = [
                a for a in self._alarm_history
                if a.timestamp >= period_start
            ]

            critical_count = sum(1 for a in recent_alarms if a.severity == AlarmSeverity.CRITICAL)
            high_count = sum(1 for a in recent_alarms if a.severity == AlarmSeverity.HIGH)

            hours = period_minutes / 60.0
            alarms_per_hour = len(recent_alarms) / hours if hours > 0 else 0.0

            unack = sum(
                1 for a in self._active_alarms.values()
                if a.state in [AlarmState.ACTIVE_UNACKNOWLEDGED, AlarmState.CLEARED_UNACKNOWLEDGED]
            )

            standing = sum(
                1 for a in self._active_alarms.values()
                if a.state in [AlarmState.ACTIVE_UNACKNOWLEDGED, AlarmState.ACTIVE_ACKNOWLEDGED]
            )

            return AlarmStatistics(
                period_start=period_start,
                period_end=now,
                total_alarms=len(recent_alarms),
                alarms_per_hour=alarms_per_hour,
                critical_alarms=critical_count,
                high_alarms=high_count,
                unacknowledged_count=unack,
                standing_alarms=standing
            )

    def _check_operator_triggers(self) -> None:
        """Check if any operator review triggers are activated."""
        for trigger in self._operator_triggers.values():
            if self._evaluate_trigger(trigger):
                self._review_pending = True
                self._review_pending_reason = f"Trigger: {trigger.trigger_name}"
                logger.warning(f"[ALARM] Operator review required: {trigger.trigger_name}")
                break

    def _evaluate_trigger(self, trigger: OperatorReviewTrigger) -> bool:
        """Evaluate a single operator trigger."""
        if trigger.condition_type == "alarm_count":
            count = len([
                a for a in self._active_alarms.values()
                if trigger.severity_filter is None or a.severity == trigger.severity_filter
            ])
            return count >= trigger.threshold
        return False

    def _check_review_release(self) -> None:
        """Check if operator review requirement can be released."""
        if not self._review_pending:
            return

        unack = self.get_unacknowledged_count()
        if unack == 0 and self._block_policy.release_after_ack:
            self._review_pending = False
            self._review_pending_reason = None
            logger.info("[ALARM] Operator review requirement released")

    def _default_escalation_rules(self) -> List[EscalationRule]:
        """Get default escalation rules."""
        return [
            EscalationRule(
                rule_id="ESC_CRIT_5", alarm_severity=AlarmSeverity.CRITICAL,
                time_to_escalate_minutes=5, from_level=EscalationLevel.OPERATOR,
                to_level=EscalationLevel.SUPERVISOR, notification_method="phone"
            ),
            EscalationRule(
                rule_id="ESC_CRIT_15", alarm_severity=AlarmSeverity.CRITICAL,
                time_to_escalate_minutes=15, from_level=EscalationLevel.SUPERVISOR,
                to_level=EscalationLevel.EMERGENCY, notification_method="phone"
            ),
            EscalationRule(
                rule_id="ESC_HIGH_15", alarm_severity=AlarmSeverity.HIGH,
                time_to_escalate_minutes=15, from_level=EscalationLevel.OPERATOR,
                to_level=EscalationLevel.SUPERVISOR, notification_method="email"
            ),
        ]

    def _default_block_policy(self) -> RecommendationBlockPolicy:
        """Get default recommendation block policy."""
        return RecommendationBlockPolicy(
            policy_id="DEFAULT_BLOCK",
            policy_name="Default Block Policy",
            description="Block recommendations on critical alarms or alarm flood",
            block_on_critical_alarm=True,
            block_on_high_alarm=True,
            block_on_unack_count=3,
            block_on_alarm_flood=True,
            release_after_ack=True,
            require_supervisor_override=False
        )
