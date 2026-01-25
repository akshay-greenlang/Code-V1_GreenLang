# -*- coding: utf-8 -*-
"""
Alarm Threshold Management for GL-005 CombustionSense
=====================================================

Provides comprehensive alarm management including:
    - Multi-level alarm thresholds
    - Deadband/hysteresis logic
    - Alarm prioritization (ISA-18.2)
    - Alarm shelving and suppression
    - Alarm rationalization support

Reference Standards:
    - ISA-18.2: Management of Alarm Systems
    - IEC 62682: Management of Alarm Systems
    - EEMUA 191: Alarm Systems Guide

Author: GL-ControlEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS (ISA-18.2 ALIGNED)
# =============================================================================

class AlarmPriority(Enum):
    """Alarm priority levels per ISA-18.2."""
    EMERGENCY = 1     # Immediate action (<1 minute)
    HIGH = 2          # Prompt action (1-5 minutes)
    MEDIUM = 3        # Timely action (5-30 minutes)
    LOW = 4           # Awareness (>30 minutes)
    DIAGNOSTIC = 5    # Maintenance/diagnostic


class AlarmState(Enum):
    """Alarm states per ISA-18.2."""
    NORMAL = "normal"
    UNACKNOWLEDGED = "unacknowledged"
    ACKNOWLEDGED = "acknowledged"
    RETURNED_UNACKNOWLEDGED = "returned_unacknowledged"
    SHELVED = "shelved"
    SUPPRESSED = "suppressed"
    OUT_OF_SERVICE = "out_of_service"


class AlarmType(Enum):
    """Types of alarms."""
    HIGH = "high"
    HIGH_HIGH = "high_high"
    LOW = "low"
    LOW_LOW = "low_low"
    RATE_OF_CHANGE = "rate_of_change"
    DEVIATION = "deviation"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AlarmThreshold:
    """Definition of an alarm threshold."""
    alarm_id: str
    parameter: str
    alarm_type: AlarmType
    setpoint: float
    deadband: float          # Hysteresis
    priority: AlarmPriority
    delay_seconds: float = 0.0
    description: str = ""
    consequence: str = ""    # What happens if ignored
    response_action: str = ""  # Required operator action


@dataclass
class ActiveAlarm:
    """Active alarm instance."""
    alarm_id: str
    threshold: AlarmThreshold
    state: AlarmState
    triggered_at: datetime
    current_value: float
    peak_value: float
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    shelved_until: Optional[datetime] = None
    provenance_hash: str = ""


@dataclass
class AlarmEvent:
    """Alarm event for logging."""
    event_id: str
    alarm_id: str
    event_type: str          # ACTIVATED, ACKNOWLEDGED, CLEARED, etc.
    timestamp: datetime
    value: float
    operator_id: Optional[str]
    details: Dict[str, Any]


# =============================================================================
# ALARM MANAGER
# =============================================================================

class AlarmManager:
    """
    Comprehensive alarm management system.

    Features:
        - Threshold monitoring with deadband
        - ISA-18.2 compliant state machine
        - Alarm prioritization
        - Shelving and suppression
        - Complete audit trail
    """

    def __init__(self, equipment_id: str = "CombustionSense"):
        self.equipment_id = equipment_id
        self.thresholds: Dict[str, AlarmThreshold] = {}
        self.active_alarms: Dict[str, ActiveAlarm] = {}
        self.event_log: List[AlarmEvent] = []
        self.alarm_callbacks: List[Callable] = []
        self.last_values: Dict[str, float] = {}

        # Alarm rate limiting
        self.alarm_flood_threshold = 10  # Alarms per minute
        self.alarm_counts: Dict[str, List[datetime]] = {}

    def register_threshold(self, threshold: AlarmThreshold) -> None:
        """
        Register an alarm threshold.

        Args:
            threshold: Alarm threshold definition
        """
        self.thresholds[threshold.alarm_id] = threshold
        self.alarm_counts[threshold.alarm_id] = []

    def process_value(
        self,
        parameter: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> List[ActiveAlarm]:
        """
        Process a value against all thresholds.

        Args:
            parameter: Parameter name
            value: Current value
            timestamp: Value timestamp (default: now)

        Returns:
            List of newly activated alarms
        """
        if timestamp is None:
            timestamp = datetime.now()

        new_alarms = []

        # Check all thresholds for this parameter
        for alarm_id, threshold in self.thresholds.items():
            if threshold.parameter != parameter:
                continue

            is_violated = self._check_threshold(value, threshold)
            is_active = alarm_id in self.active_alarms

            if is_violated and not is_active:
                # New alarm
                alarm = self._activate_alarm(threshold, value, timestamp)
                if alarm:
                    new_alarms.append(alarm)

            elif not is_violated and is_active:
                # Check deadband before clearing
                if self._check_return_to_normal(value, threshold):
                    self._clear_alarm(alarm_id, value, timestamp)

            elif is_violated and is_active:
                # Update peak value
                alarm = self.active_alarms[alarm_id]
                if threshold.alarm_type in [AlarmType.HIGH, AlarmType.HIGH_HIGH]:
                    alarm.peak_value = max(alarm.peak_value, value)
                else:
                    alarm.peak_value = min(alarm.peak_value, value)
                alarm.current_value = value

        self.last_values[parameter] = value

        return new_alarms

    def _check_threshold(self, value: float, threshold: AlarmThreshold) -> bool:
        """Check if value violates threshold."""
        if threshold.alarm_type in [AlarmType.HIGH, AlarmType.HIGH_HIGH]:
            return value >= threshold.setpoint
        elif threshold.alarm_type in [AlarmType.LOW, AlarmType.LOW_LOW]:
            return value <= threshold.setpoint
        return False

    def _check_return_to_normal(self, value: float, threshold: AlarmThreshold) -> bool:
        """Check if value has returned to normal with deadband."""
        if threshold.alarm_type in [AlarmType.HIGH, AlarmType.HIGH_HIGH]:
            return value < (threshold.setpoint - threshold.deadband)
        elif threshold.alarm_type in [AlarmType.LOW, AlarmType.LOW_LOW]:
            return value > (threshold.setpoint + threshold.deadband)
        return True

    def _activate_alarm(
        self,
        threshold: AlarmThreshold,
        value: float,
        timestamp: datetime
    ) -> Optional[ActiveAlarm]:
        """Activate a new alarm."""
        # Check for alarm flooding
        if self._is_alarm_flooding(threshold.alarm_id):
            logger.warning(f"Alarm flooding detected for {threshold.alarm_id}")
            return None

        alarm = ActiveAlarm(
            alarm_id=threshold.alarm_id,
            threshold=threshold,
            state=AlarmState.UNACKNOWLEDGED,
            triggered_at=timestamp,
            current_value=value,
            peak_value=value,
            provenance_hash=self._calculate_hash(threshold.alarm_id, value, timestamp),
        )

        self.active_alarms[threshold.alarm_id] = alarm
        self.alarm_counts[threshold.alarm_id].append(timestamp)

        # Log event
        self._log_event(
            alarm_id=threshold.alarm_id,
            event_type="ACTIVATED",
            timestamp=timestamp,
            value=value,
            operator_id=None,
            details={"setpoint": threshold.setpoint},
        )

        # Execute callbacks
        for callback in self.alarm_callbacks:
            try:
                callback("ACTIVATED", alarm)
            except Exception as e:
                logger.error(f"Alarm callback error: {e}")

        logger.info(f"Alarm activated: {threshold.alarm_id} ({threshold.priority.name})")

        return alarm

    def _clear_alarm(
        self,
        alarm_id: str,
        value: float,
        timestamp: datetime
    ) -> None:
        """Clear an alarm."""
        if alarm_id not in self.active_alarms:
            return

        alarm = self.active_alarms[alarm_id]

        # ISA-18.2: If acknowledged, clear immediately
        # If unacknowledged, go to RETURNED_UNACKNOWLEDGED
        if alarm.state == AlarmState.ACKNOWLEDGED:
            del self.active_alarms[alarm_id]
            event_type = "CLEARED"
        else:
            alarm.state = AlarmState.RETURNED_UNACKNOWLEDGED
            event_type = "RETURNED_TO_NORMAL"

        self._log_event(
            alarm_id=alarm_id,
            event_type=event_type,
            timestamp=timestamp,
            value=value,
            operator_id=None,
            details={},
        )

        logger.info(f"Alarm {event_type.lower()}: {alarm_id}")

    def acknowledge_alarm(
        self,
        alarm_id: str,
        operator_id: str
    ) -> bool:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: ID of alarm to acknowledge
            operator_id: ID of operator acknowledging

        Returns:
            True if acknowledged successfully
        """
        if alarm_id not in self.active_alarms:
            return False

        alarm = self.active_alarms[alarm_id]

        if alarm.state == AlarmState.UNACKNOWLEDGED:
            alarm.state = AlarmState.ACKNOWLEDGED
        elif alarm.state == AlarmState.RETURNED_UNACKNOWLEDGED:
            # Clear alarm completely
            del self.active_alarms[alarm_id]

        alarm.acknowledged_by = operator_id
        alarm.acknowledged_at = datetime.now()

        self._log_event(
            alarm_id=alarm_id,
            event_type="ACKNOWLEDGED",
            timestamp=datetime.now(),
            value=alarm.current_value,
            operator_id=operator_id,
            details={},
        )

        return True

    def shelve_alarm(
        self,
        alarm_id: str,
        operator_id: str,
        duration_minutes: int = 60
    ) -> bool:
        """
        Shelve an alarm temporarily.

        Args:
            alarm_id: ID of alarm to shelve
            operator_id: ID of operator shelving
            duration_minutes: Shelving duration

        Returns:
            True if shelved successfully
        """
        if alarm_id not in self.thresholds:
            return False

        # Create or update shelved state
        if alarm_id in self.active_alarms:
            alarm = self.active_alarms[alarm_id]
            alarm.state = AlarmState.SHELVED
            alarm.shelved_until = datetime.now() + timedelta(minutes=duration_minutes)

        self._log_event(
            alarm_id=alarm_id,
            event_type="SHELVED",
            timestamp=datetime.now(),
            value=0.0,
            operator_id=operator_id,
            details={"duration_minutes": duration_minutes},
        )

        return True

    def _is_alarm_flooding(self, alarm_id: str) -> bool:
        """Check if alarm is flooding."""
        counts = self.alarm_counts.get(alarm_id, [])
        one_minute_ago = datetime.now() - timedelta(minutes=1)

        # Clean old counts
        counts = [t for t in counts if t > one_minute_ago]
        self.alarm_counts[alarm_id] = counts

        return len(counts) >= self.alarm_flood_threshold

    def get_alarm_summary(self) -> Dict[str, Any]:
        """Get summary of current alarms."""
        priority_counts = {p.name: 0 for p in AlarmPriority}
        state_counts = {s.value: 0 for s in AlarmState}

        for alarm in self.active_alarms.values():
            priority_counts[alarm.threshold.priority.name] += 1
            state_counts[alarm.state.value] += 1

        return {
            "total_active": len(self.active_alarms),
            "by_priority": priority_counts,
            "by_state": state_counts,
            "unacknowledged_count": sum(
                1 for a in self.active_alarms.values()
                if a.state == AlarmState.UNACKNOWLEDGED
            ),
        }

    def get_alarms_by_priority(self, priority: AlarmPriority) -> List[ActiveAlarm]:
        """Get all active alarms of specified priority."""
        return [
            alarm for alarm in self.active_alarms.values()
            if alarm.threshold.priority == priority
        ]

    def register_callback(
        self,
        callback: Callable[[str, ActiveAlarm], None]
    ) -> None:
        """Register callback for alarm events."""
        self.alarm_callbacks.append(callback)

    def _log_event(
        self,
        alarm_id: str,
        event_type: str,
        timestamp: datetime,
        value: float,
        operator_id: Optional[str],
        details: Dict[str, Any]
    ) -> None:
        """Log an alarm event."""
        event = AlarmEvent(
            event_id=f"{self.equipment_id}:{alarm_id}:{timestamp.strftime('%Y%m%d%H%M%S%f')}",
            alarm_id=alarm_id,
            event_type=event_type,
            timestamp=timestamp,
            value=value,
            operator_id=operator_id,
            details=details,
        )
        self.event_log.append(event)

        if len(self.event_log) > 10000:
            self.event_log = self.event_log[-10000:]

    def _calculate_hash(
        self,
        alarm_id: str,
        value: float,
        timestamp: datetime
    ) -> str:
        """Calculate provenance hash."""
        data = f"{alarm_id}:{value}:{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_combustion_alarm_thresholds() -> List[AlarmThreshold]:
    """Create standard combustion alarm thresholds."""
    return [
        # O2 Alarms
        AlarmThreshold(
            alarm_id="O2_LOW",
            parameter="O2",
            alarm_type=AlarmType.LOW,
            setpoint=2.0,
            deadband=0.3,
            priority=AlarmPriority.HIGH,
            delay_seconds=5.0,
            description="O2 low - risk of incomplete combustion",
            consequence="CO formation, efficiency loss",
            response_action="Increase combustion air",
        ),
        AlarmThreshold(
            alarm_id="O2_LOW_LOW",
            parameter="O2",
            alarm_type=AlarmType.LOW_LOW,
            setpoint=1.0,
            deadband=0.2,
            priority=AlarmPriority.EMERGENCY,
            delay_seconds=0.0,
            description="O2 critically low - incomplete combustion",
            consequence="High CO, safety hazard",
            response_action="Reduce firing rate immediately",
        ),

        # CO Alarms
        AlarmThreshold(
            alarm_id="CO_HIGH",
            parameter="CO",
            alarm_type=AlarmType.HIGH,
            setpoint=400.0,
            deadband=50.0,
            priority=AlarmPriority.HIGH,
            delay_seconds=10.0,
            description="CO high - incomplete combustion",
            consequence="Efficiency loss, emissions violation",
            response_action="Check O2 and air-fuel ratio",
        ),
        AlarmThreshold(
            alarm_id="CO_HIGH_HIGH",
            parameter="CO",
            alarm_type=AlarmType.HIGH_HIGH,
            setpoint=1000.0,
            deadband=100.0,
            priority=AlarmPriority.EMERGENCY,
            delay_seconds=5.0,
            description="CO critically high",
            consequence="Safety hazard, potential trip",
            response_action="Reduce firing immediately",
        ),

        # Flame Alarms
        AlarmThreshold(
            alarm_id="FLAME_LOW",
            parameter="flame_signal",
            alarm_type=AlarmType.LOW,
            setpoint=3.0,
            deadband=0.5,
            priority=AlarmPriority.HIGH,
            delay_seconds=2.0,
            description="Flame signal weak",
            consequence="Potential flame loss",
            response_action="Check flame scanner and burner",
        ),

        # Furnace Pressure Alarms
        AlarmThreshold(
            alarm_id="PRESSURE_HIGH",
            parameter="furnace_pressure",
            alarm_type=AlarmType.HIGH,
            setpoint=0.5,
            deadband=0.1,
            priority=AlarmPriority.MEDIUM,
            delay_seconds=3.0,
            description="Furnace pressure high",
            consequence="Structural stress, puffback risk",
            response_action="Check ID fan and dampers",
        ),
    ]


if __name__ == "__main__":
    # Example usage
    manager = AlarmManager()

    for threshold in create_combustion_alarm_thresholds():
        manager.register_threshold(threshold)

    # Simulate low O2 condition
    new_alarms = manager.process_value("O2", 1.5)
    print(f"New alarms: {[a.alarm_id for a in new_alarms]}")

    # Get summary
    print(manager.get_alarm_summary())

    # Acknowledge
    if new_alarms:
        manager.acknowledge_alarm(new_alarms[0].alarm_id, "OPERATOR-1")

    # Value returns to normal
    manager.process_value("O2", 3.5)
    print(f"Active alarms after recovery: {list(manager.active_alarms.keys())}")
