"""
Safety Tests: Alarm and Emergency Stop Handling

Tests alarm management and emergency stop behavior including:
- Alarm storm detection and handling
- Alarm prioritization
- Emergency stop sequence
- Safe state transitions

Reference: GL-001 Specification Section 11.4
Target Coverage: 85%+
"""

import pytest
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum
from collections import deque


# =============================================================================
# Alarm Handling Classes (Simulated Production Code)
# =============================================================================

class AlarmPriority(Enum):
    """Alarm priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AlarmState(Enum):
    """Alarm states."""
    INACTIVE = 0
    ACTIVE_UNACKED = 1
    ACTIVE_ACKED = 2
    CLEARED_UNACKED = 3


@dataclass
class Alarm:
    """Alarm definition and current state."""
    alarm_id: str
    description: str
    priority: AlarmPriority
    source: str
    state: AlarmState = AlarmState.INACTIVE
    timestamp: Optional[datetime] = None
    ack_timestamp: Optional[datetime] = None
    clear_timestamp: Optional[datetime] = None
    value: Optional[float] = None
    limit: Optional[float] = None


class AlarmStormDetector:
    """Detects and handles alarm storms."""

    def __init__(self, window_seconds: float = 60.0, threshold: int = 10):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.alarm_timestamps: deque = deque()
        self.storm_active = False
        self.storm_start: Optional[datetime] = None

    def record_alarm(self, timestamp: datetime) -> bool:
        """Record an alarm occurrence and check for storm.

        Returns True if storm is detected.
        """
        # Remove old timestamps outside window
        cutoff = timestamp - timedelta(seconds=self.window_seconds)
        while self.alarm_timestamps and self.alarm_timestamps[0] < cutoff:
            self.alarm_timestamps.popleft()

        # Add new timestamp
        self.alarm_timestamps.append(timestamp)

        # Check for storm
        if len(self.alarm_timestamps) >= self.threshold:
            if not self.storm_active:
                self.storm_active = True
                self.storm_start = timestamp
            return True

        return False

    def is_storm_active(self) -> bool:
        """Check if alarm storm is currently active."""
        return self.storm_active

    def get_alarm_rate(self) -> float:
        """Get current alarm rate (alarms per minute)."""
        if not self.alarm_timestamps:
            return 0.0

        if len(self.alarm_timestamps) < 2:
            return 0.0

        duration = (self.alarm_timestamps[-1] - self.alarm_timestamps[0]).total_seconds()
        if duration <= 0:
            return 0.0

        return (len(self.alarm_timestamps) - 1) / (duration / 60.0)

    def clear_storm(self):
        """Clear storm condition."""
        self.storm_active = False
        self.storm_start = None


class AlarmManager:
    """Manages alarm lifecycle and prioritization."""

    def __init__(self):
        self.alarms: Dict[str, Alarm] = {}
        self.storm_detector = AlarmStormDetector()
        self.suppressed_alarms: List[str] = []

    def register_alarm(self, alarm: Alarm):
        """Register an alarm definition."""
        self.alarms[alarm.alarm_id] = alarm

    def activate_alarm(self, alarm_id: str, value: float = None,
                      timestamp: datetime = None) -> bool:
        """Activate an alarm."""
        if alarm_id not in self.alarms:
            return False

        alarm = self.alarms[alarm_id]
        timestamp = timestamp or datetime.now()

        # Check for storm
        is_storm = self.storm_detector.record_alarm(timestamp)

        # Suppress low priority alarms during storm
        if is_storm and alarm.priority.value < AlarmPriority.HIGH.value:
            self.suppressed_alarms.append(alarm_id)
            return False

        alarm.state = AlarmState.ACTIVE_UNACKED
        alarm.timestamp = timestamp
        alarm.value = value

        return True

    def acknowledge_alarm(self, alarm_id: str, timestamp: datetime = None) -> bool:
        """Acknowledge an alarm."""
        if alarm_id not in self.alarms:
            return False

        alarm = self.alarms[alarm_id]
        if alarm.state != AlarmState.ACTIVE_UNACKED:
            return False

        alarm.state = AlarmState.ACTIVE_ACKED
        alarm.ack_timestamp = timestamp or datetime.now()

        return True

    def clear_alarm(self, alarm_id: str, timestamp: datetime = None) -> bool:
        """Clear an alarm (condition no longer present)."""
        if alarm_id not in self.alarms:
            return False

        alarm = self.alarms[alarm_id]

        if alarm.state == AlarmState.ACTIVE_ACKED:
            alarm.state = AlarmState.INACTIVE
        elif alarm.state == AlarmState.ACTIVE_UNACKED:
            alarm.state = AlarmState.CLEARED_UNACKED

        alarm.clear_timestamp = timestamp or datetime.now()

        return True

    def get_active_alarms(self, min_priority: AlarmPriority = None) -> List[Alarm]:
        """Get list of active alarms, optionally filtered by priority."""
        active = [a for a in self.alarms.values()
                  if a.state in (AlarmState.ACTIVE_UNACKED, AlarmState.ACTIVE_ACKED)]

        if min_priority:
            active = [a for a in active if a.priority.value >= min_priority.value]

        return sorted(active, key=lambda a: a.priority.value, reverse=True)

    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alarms."""
        return sum(1 for a in self.alarms.values()
                   if a.state in (AlarmState.ACTIVE_UNACKED, AlarmState.CLEARED_UNACKED))


class EmergencyStopController:
    """Controls emergency stop sequence."""

    def __init__(self):
        self.estop_active = False
        self.estop_timestamp: Optional[datetime] = None
        self.estop_reason: Optional[str] = None
        self.equipment_states: Dict[str, str] = {}

    def trigger_estop(self, reason: str, timestamp: datetime = None) -> bool:
        """Trigger emergency stop."""
        self.estop_active = True
        self.estop_timestamp = timestamp or datetime.now()
        self.estop_reason = reason

        # Execute emergency shutdown sequence
        self._execute_shutdown_sequence()

        return True

    def _execute_shutdown_sequence(self):
        """Execute emergency shutdown sequence."""
        # Close fuel valves
        self.equipment_states["fuel_valve_main"] = "CLOSED"
        self.equipment_states["fuel_valve_pilot"] = "CLOSED"

        # Stop combustion air
        self.equipment_states["combustion_air_fan"] = "STOPPED"

        # Open vent valves
        self.equipment_states["vent_valve"] = "OPEN"

        # Trip boilers
        self.equipment_states["boiler_01"] = "TRIPPED"
        self.equipment_states["boiler_02"] = "TRIPPED"

    def reset_estop(self) -> bool:
        """Reset emergency stop (requires all conditions safe)."""
        if not self.estop_active:
            return True

        # Check if safe to reset (simplified)
        if self._all_conditions_safe():
            self.estop_active = False
            return True

        return False

    def _all_conditions_safe(self) -> bool:
        """Check if all conditions are safe for reset."""
        # In real implementation, would check actual safety conditions
        return True

    def get_equipment_state(self, equipment_id: str) -> Optional[str]:
        """Get current state of equipment."""
        return self.equipment_states.get(equipment_id)


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.safety
class TestAlarmStormDetection:
    """Test alarm storm detection."""

    @pytest.fixture
    def detector(self):
        """Create alarm storm detector."""
        return AlarmStormDetector(window_seconds=60.0, threshold=10)

    def test_no_storm_below_threshold(self, detector):
        """Test no storm detected below threshold."""
        base_time = datetime.now()

        for i in range(5):
            is_storm = detector.record_alarm(base_time + timedelta(seconds=i))

        assert is_storm == False
        assert detector.is_storm_active() == False

    def test_storm_detected_at_threshold(self, detector):
        """Test storm detected when threshold reached."""
        base_time = datetime.now()

        for i in range(10):
            is_storm = detector.record_alarm(base_time + timedelta(seconds=i))

        assert is_storm == True
        assert detector.is_storm_active() == True

    def test_storm_clears_when_cleared(self, detector):
        """Test storm condition clears."""
        base_time = datetime.now()

        # Trigger storm
        for i in range(10):
            detector.record_alarm(base_time + timedelta(seconds=i))

        detector.clear_storm()

        assert detector.is_storm_active() == False

    def test_old_alarms_expire(self, detector):
        """Test old alarms outside window are not counted."""
        # Alarms 2 minutes ago
        old_time = datetime.now() - timedelta(minutes=2)
        for i in range(5):
            detector.record_alarm(old_time + timedelta(seconds=i))

        # Recent alarms
        now = datetime.now()
        for i in range(4):  # Total would be 9 if old counted
            is_storm = detector.record_alarm(now + timedelta(seconds=i))

        # Should not be storm because old alarms expired
        assert is_storm == False

    def test_alarm_rate_calculation(self, detector):
        """Test alarm rate calculation."""
        base_time = datetime.now()

        # 10 alarms over 1 minute = 9 alarms/minute
        for i in range(10):
            detector.record_alarm(base_time + timedelta(seconds=i * 6))

        rate = detector.get_alarm_rate()

        # Should be approximately 10/minute
        assert rate > 8 and rate < 12


@pytest.mark.safety
class TestAlarmManager:
    """Test alarm management."""

    @pytest.fixture
    def manager(self):
        """Create alarm manager with registered alarms."""
        manager = AlarmManager()
        manager.register_alarm(Alarm(
            alarm_id="TEMP_HIGH",
            description="Temperature high",
            priority=AlarmPriority.HIGH,
            source="BOILER_001"
        ))
        manager.register_alarm(Alarm(
            alarm_id="PRESS_LOW",
            description="Pressure low",
            priority=AlarmPriority.MEDIUM,
            source="BOILER_001"
        ))
        manager.register_alarm(Alarm(
            alarm_id="TRIP_EMERGENCY",
            description="Emergency trip",
            priority=AlarmPriority.EMERGENCY,
            source="SIS"
        ))
        return manager

    def test_activate_alarm(self, manager):
        """Test alarm activation."""
        result = manager.activate_alarm("TEMP_HIGH", value=550.0)

        assert result == True
        assert manager.alarms["TEMP_HIGH"].state == AlarmState.ACTIVE_UNACKED

    def test_acknowledge_alarm(self, manager):
        """Test alarm acknowledgment."""
        manager.activate_alarm("TEMP_HIGH")
        result = manager.acknowledge_alarm("TEMP_HIGH")

        assert result == True
        assert manager.alarms["TEMP_HIGH"].state == AlarmState.ACTIVE_ACKED

    def test_clear_acked_alarm(self, manager):
        """Test clearing acknowledged alarm."""
        manager.activate_alarm("TEMP_HIGH")
        manager.acknowledge_alarm("TEMP_HIGH")
        result = manager.clear_alarm("TEMP_HIGH")

        assert result == True
        assert manager.alarms["TEMP_HIGH"].state == AlarmState.INACTIVE

    def test_clear_unacked_alarm(self, manager):
        """Test clearing unacknowledged alarm."""
        manager.activate_alarm("TEMP_HIGH")
        result = manager.clear_alarm("TEMP_HIGH")

        assert result == True
        assert manager.alarms["TEMP_HIGH"].state == AlarmState.CLEARED_UNACKED

    def test_get_active_alarms_sorted_by_priority(self, manager):
        """Test active alarms are sorted by priority."""
        manager.activate_alarm("TEMP_HIGH")
        manager.activate_alarm("PRESS_LOW")
        manager.activate_alarm("TRIP_EMERGENCY")

        active = manager.get_active_alarms()

        assert len(active) == 3
        assert active[0].alarm_id == "TRIP_EMERGENCY"  # Highest priority first
        assert active[1].alarm_id == "TEMP_HIGH"
        assert active[2].alarm_id == "PRESS_LOW"

    def test_filter_active_alarms_by_priority(self, manager):
        """Test filtering active alarms by minimum priority."""
        manager.activate_alarm("TEMP_HIGH")
        manager.activate_alarm("PRESS_LOW")
        manager.activate_alarm("TRIP_EMERGENCY")

        active = manager.get_active_alarms(min_priority=AlarmPriority.HIGH)

        assert len(active) == 2
        assert all(a.priority.value >= AlarmPriority.HIGH.value for a in active)

    def test_unacknowledged_count(self, manager):
        """Test unacknowledged alarm count."""
        manager.activate_alarm("TEMP_HIGH")
        manager.activate_alarm("PRESS_LOW")
        manager.acknowledge_alarm("TEMP_HIGH")

        count = manager.get_unacknowledged_count()

        assert count == 1


@pytest.mark.safety
class TestEmergencyStop:
    """Test emergency stop functionality."""

    @pytest.fixture
    def estop(self):
        """Create emergency stop controller."""
        return EmergencyStopController()

    def test_trigger_estop(self, estop):
        """Test triggering emergency stop."""
        result = estop.trigger_estop("High temperature trip")

        assert result == True
        assert estop.estop_active == True
        assert estop.estop_reason == "High temperature trip"

    def test_estop_closes_fuel_valves(self, estop):
        """Test E-stop closes fuel valves."""
        estop.trigger_estop("Test")

        assert estop.get_equipment_state("fuel_valve_main") == "CLOSED"
        assert estop.get_equipment_state("fuel_valve_pilot") == "CLOSED"

    def test_estop_stops_combustion_air(self, estop):
        """Test E-stop stops combustion air fan."""
        estop.trigger_estop("Test")

        assert estop.get_equipment_state("combustion_air_fan") == "STOPPED"

    def test_estop_opens_vent_valves(self, estop):
        """Test E-stop opens vent valves."""
        estop.trigger_estop("Test")

        assert estop.get_equipment_state("vent_valve") == "OPEN"

    def test_estop_trips_boilers(self, estop):
        """Test E-stop trips all boilers."""
        estop.trigger_estop("Test")

        assert estop.get_equipment_state("boiler_01") == "TRIPPED"
        assert estop.get_equipment_state("boiler_02") == "TRIPPED"

    def test_estop_reset(self, estop):
        """Test E-stop reset when conditions safe."""
        estop.trigger_estop("Test")
        result = estop.reset_estop()

        assert result == True
        assert estop.estop_active == False

    def test_estop_records_timestamp(self, estop):
        """Test E-stop records timestamp."""
        before = datetime.now()
        estop.trigger_estop("Test")
        after = datetime.now()

        assert estop.estop_timestamp >= before
        assert estop.estop_timestamp <= after


@pytest.mark.safety
class TestAlarmStormSuppression:
    """Test alarm suppression during storm conditions."""

    def test_low_priority_suppressed_during_storm(self):
        """Test low priority alarms suppressed during storm."""
        manager = AlarmManager()
        manager.register_alarm(Alarm(
            alarm_id="LOW_PRIO",
            description="Low priority alarm",
            priority=AlarmPriority.LOW,
            source="SENSOR"
        ))

        # Trigger storm by activating many alarms
        base_time = datetime.now()
        for i in range(15):
            manager.storm_detector.record_alarm(base_time + timedelta(seconds=i))

        # Try to activate low priority alarm during storm
        result = manager.activate_alarm("LOW_PRIO", timestamp=base_time)

        assert result == False
        assert "LOW_PRIO" in manager.suppressed_alarms

    def test_high_priority_not_suppressed_during_storm(self):
        """Test high priority alarms not suppressed during storm."""
        manager = AlarmManager()
        manager.register_alarm(Alarm(
            alarm_id="HIGH_PRIO",
            description="High priority alarm",
            priority=AlarmPriority.HIGH,
            source="SIS"
        ))

        # Trigger storm
        base_time = datetime.now()
        for i in range(15):
            manager.storm_detector.record_alarm(base_time + timedelta(seconds=i))

        # High priority should still activate
        result = manager.activate_alarm("HIGH_PRIO", timestamp=base_time)

        assert result == True
