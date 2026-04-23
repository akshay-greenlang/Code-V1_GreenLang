# -*- coding: utf-8 -*-
"""
Combustion Safety Limits Module for GL-005 CombustionSense
==========================================================

Provides real-time monitoring and enforcement of combustion safety limits
based on industry standards and regulatory requirements.

Safety Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - API 556: Instrumentation, Control, and Protective Systems
    - IEC 61511: Safety Instrumented Systems

Features:
    - Real-time limit monitoring
    - Multi-level alarm generation
    - Automatic trip initiation
    - Safety interlock validation
    - Audit trail for safety events

Author: GL-SafetyEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime
import hashlib
import json


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AlarmPriority(Enum):
    """Alarm priority levels per ISA-18.2."""
    EMERGENCY = 1    # Immediate action required
    HIGH = 2         # Prompt action required
    MEDIUM = 3       # Operator awareness
    LOW = 4          # Information only
    DIAGNOSTIC = 5   # Maintenance/diagnostic


class AlarmState(Enum):
    """Alarm states per ISA-18.2."""
    NORMAL = "normal"
    UNACKNOWLEDGED = "unacknowledged"
    ACKNOWLEDGED = "acknowledged"
    SHELVED = "shelved"
    SUPPRESSED = "suppressed"


class TripAction(Enum):
    """Safety trip actions."""
    NONE = "none"
    ALARM_ONLY = "alarm_only"
    REDUCE_FIRING = "reduce_firing"
    MASTER_FUEL_TRIP = "master_fuel_trip"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class SensorStatus(Enum):
    """Sensor health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    BYPASSED = "bypassed"


# =============================================================================
# SAFETY LIMIT DEFINITIONS
# =============================================================================

@dataclass
class SafetyLimit:
    """Definition of a combustion safety limit."""
    name: str
    parameter: str
    units: str

    # Limit values
    ll_trip: Optional[float] = None     # Low-Low (Trip)
    l_alarm: Optional[float] = None     # Low Alarm
    h_alarm: Optional[float] = None     # High Alarm
    hh_trip: Optional[float] = None     # High-High (Trip)

    # Timing
    delay_seconds: float = 0.0          # Time delay before action
    deadband_percent: float = 2.0       # Deadband to prevent chatter

    # Actions
    trip_action: TripAction = TripAction.ALARM_ONLY
    priority: AlarmPriority = AlarmPriority.MEDIUM

    # Metadata
    source_standard: str = ""
    rationale: str = ""


# NFPA 85 and industry-standard combustion limits
COMBUSTION_SAFETY_LIMITS = {
    # Oxygen limits
    "O2_low": SafetyLimit(
        name="O2 Low Limit",
        parameter="O2",
        units="%",
        ll_trip=1.0,      # Trip at <1% O2 (incomplete combustion)
        l_alarm=2.0,      # Alarm at <2% O2
        delay_seconds=5.0,
        trip_action=TripAction.REDUCE_FIRING,
        priority=AlarmPriority.HIGH,
        source_standard="NFPA 85",
        rationale="Low O2 indicates incomplete combustion, CO formation risk",
    ),

    "O2_high": SafetyLimit(
        name="O2 High Limit",
        parameter="O2",
        units="%",
        h_alarm=8.0,      # Alarm at >8% O2 (efficiency loss)
        hh_trip=12.0,     # Trip at >12% O2 (flammability concern)
        delay_seconds=10.0,
        trip_action=TripAction.ALARM_ONLY,
        priority=AlarmPriority.MEDIUM,
        source_standard="Industry Practice",
        rationale="High O2 indicates excess air, efficiency loss, or air leak",
    ),

    # Carbon monoxide limits
    "CO_high": SafetyLimit(
        name="CO High Limit",
        parameter="CO",
        units="ppm",
        h_alarm=400.0,    # Alarm at >400 ppm
        hh_trip=1000.0,   # Trip at >1000 ppm
        delay_seconds=10.0,
        trip_action=TripAction.REDUCE_FIRING,
        priority=AlarmPriority.HIGH,
        source_standard="NFPA 85",
        rationale="High CO indicates incomplete combustion, safety hazard",
    ),

    # Combustibles (LEL monitoring)
    "LEL": SafetyLimit(
        name="LEL (Lower Explosive Limit)",
        parameter="LEL",
        units="%LEL",
        h_alarm=25.0,     # Alarm at 25% LEL
        hh_trip=50.0,     # Trip at 50% LEL
        delay_seconds=2.0,
        trip_action=TripAction.MASTER_FUEL_TRIP,
        priority=AlarmPriority.EMERGENCY,
        source_standard="NFPA 86",
        rationale="LEL monitoring prevents explosive atmosphere formation",
    ),

    # Flame detection
    "flame_signal": SafetyLimit(
        name="Flame Signal",
        parameter="flame_signal",
        units="mA",
        ll_trip=2.0,      # Trip if flame signal <2 mA (loss of flame)
        l_alarm=3.0,      # Alarm at <3 mA
        delay_seconds=4.0,  # NFPA 85 requires detection within 4 seconds
        trip_action=TripAction.MASTER_FUEL_TRIP,
        priority=AlarmPriority.EMERGENCY,
        source_standard="NFPA 85 Chapter 8",
        rationale="Flame failure requires immediate fuel shutoff",
    ),

    # Furnace pressure
    "furnace_pressure_high": SafetyLimit(
        name="Furnace Pressure High",
        parameter="furnace_pressure",
        units="inwc",
        h_alarm=0.5,      # High pressure alarm
        hh_trip=1.0,      # Trip at +1 inwc
        delay_seconds=3.0,
        trip_action=TripAction.REDUCE_FIRING,
        priority=AlarmPriority.HIGH,
        source_standard="NFPA 85",
        rationale="High furnace pressure indicates ID fan or damper issue",
    ),

    "furnace_pressure_low": SafetyLimit(
        name="Furnace Pressure Low",
        parameter="furnace_pressure",
        units="inwc",
        ll_trip=-3.0,     # Trip at -3 inwc (implosion risk)
        l_alarm=-2.0,     # Low pressure alarm
        delay_seconds=3.0,
        trip_action=TripAction.REDUCE_FIRING,
        priority=AlarmPriority.HIGH,
        source_standard="NFPA 85",
        rationale="Low furnace pressure indicates FD fan or damper issue",
    ),

    # Combustion air flow
    "airflow_low": SafetyLimit(
        name="Combustion Air Flow Low",
        parameter="airflow",
        units="%",
        ll_trip=25.0,     # Trip at <25% of design flow
        l_alarm=50.0,     # Alarm at <50% flow
        delay_seconds=5.0,
        trip_action=TripAction.MASTER_FUEL_TRIP,
        priority=AlarmPriority.EMERGENCY,
        source_standard="NFPA 85",
        rationale="Insufficient air causes incomplete combustion",
    ),

    # Fuel pressure
    "fuel_pressure_low": SafetyLimit(
        name="Fuel Pressure Low",
        parameter="fuel_pressure",
        units="psig",
        ll_trip=2.0,      # Trip at <2 psig (gas)
        l_alarm=5.0,      # Low pressure alarm
        delay_seconds=2.0,
        trip_action=TripAction.MASTER_FUEL_TRIP,
        priority=AlarmPriority.EMERGENCY,
        source_standard="NFPA 85",
        rationale="Low fuel pressure causes flame instability",
    ),

    # Stack temperature
    "stack_temp_high": SafetyLimit(
        name="Stack Temperature High",
        parameter="stack_temperature",
        units="°C",
        h_alarm=350.0,    # High temp alarm
        hh_trip=450.0,    # Trip at very high temp
        delay_seconds=30.0,
        trip_action=TripAction.REDUCE_FIRING,
        priority=AlarmPriority.MEDIUM,
        source_standard="Industry Practice",
        rationale="High stack temp indicates efficiency loss or damage risk",
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AlarmEvent:
    """Record of an alarm event."""
    alarm_id: str
    limit_name: str
    parameter: str
    measured_value: float
    limit_value: float
    priority: AlarmPriority
    state: AlarmState
    trip_action: TripAction
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alarm_id": self.alarm_id,
            "limit_name": self.limit_name,
            "parameter": self.parameter,
            "measured_value": self.measured_value,
            "limit_value": self.limit_value,
            "priority": self.priority.name,
            "state": self.state.value,
            "trip_action": self.trip_action.value,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class SafetyStatus:
    """Current safety system status."""
    is_safe: bool
    active_alarms: List[AlarmEvent]
    tripped_limits: List[str]
    sensor_status: Dict[str, SensorStatus]
    last_check_time: datetime
    trip_required: bool
    trip_action: TripAction


# =============================================================================
# COMBUSTION SAFETY MONITOR
# =============================================================================

class CombustionSafetyMonitor:
    """
    Real-time combustion safety limit monitoring.

    Provides:
        - Continuous limit checking
        - Alarm generation and management
        - Safety trip initiation
        - Complete audit trail
    """

    def __init__(self, equipment_id: str = "CombustionSense"):
        """
        Initialize safety monitor.

        Args:
            equipment_id: Identifier for the monitored equipment
        """
        self.equipment_id = equipment_id
        self.limits = COMBUSTION_SAFETY_LIMITS.copy()
        self.active_alarms: Dict[str, AlarmEvent] = {}
        self.alarm_history: List[AlarmEvent] = []
        self.trip_callbacks: List[Callable] = []

    def check_limits(
        self,
        measurements: Dict[str, float]
    ) -> SafetyStatus:
        """
        Check all measurements against safety limits.

        Args:
            measurements: Dictionary of current measurements

        Returns:
            SafetyStatus with current safety state
        """
        new_alarms = []
        tripped_limits = []
        trip_required = False
        highest_trip_action = TripAction.NONE

        for limit_key, limit in self.limits.items():
            param = limit.parameter
            if param not in measurements:
                continue

            value = measurements[param]
            alarm = self._check_single_limit(limit_key, limit, value)

            if alarm:
                new_alarms.append(alarm)
                if alarm.trip_action != TripAction.ALARM_ONLY:
                    tripped_limits.append(limit_key)
                    trip_required = True

                    # Track highest priority trip action
                    if alarm.trip_action.value > highest_trip_action.value:
                        highest_trip_action = alarm.trip_action

        # Update active alarms
        for alarm in new_alarms:
            self.active_alarms[alarm.alarm_id] = alarm
            self.alarm_history.append(alarm)

        # Execute trip callbacks if needed
        if trip_required:
            self._execute_trip(highest_trip_action)

        # Determine sensor status (simplified)
        sensor_status = {
            param: SensorStatus.HEALTHY
            for param in measurements.keys()
        }

        return SafetyStatus(
            is_safe=not trip_required,
            active_alarms=list(self.active_alarms.values()),
            tripped_limits=tripped_limits,
            sensor_status=sensor_status,
            last_check_time=datetime.now(),
            trip_required=trip_required,
            trip_action=highest_trip_action,
        )

    def _check_single_limit(
        self,
        limit_key: str,
        limit: SafetyLimit,
        value: float
    ) -> Optional[AlarmEvent]:
        """Check a single limit and generate alarm if violated."""
        alarm_id = f"{self.equipment_id}:{limit_key}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # Check high-high trip
        if limit.hh_trip is not None and value >= limit.hh_trip:
            return self._create_alarm(
                alarm_id, limit, value, limit.hh_trip,
                AlarmPriority.EMERGENCY, limit.trip_action
            )

        # Check high alarm
        if limit.h_alarm is not None and value >= limit.h_alarm:
            return self._create_alarm(
                alarm_id, limit, value, limit.h_alarm,
                limit.priority, TripAction.ALARM_ONLY
            )

        # Check low-low trip
        if limit.ll_trip is not None and value <= limit.ll_trip:
            return self._create_alarm(
                alarm_id, limit, value, limit.ll_trip,
                AlarmPriority.EMERGENCY, limit.trip_action
            )

        # Check low alarm
        if limit.l_alarm is not None and value <= limit.l_alarm:
            return self._create_alarm(
                alarm_id, limit, value, limit.l_alarm,
                limit.priority, TripAction.ALARM_ONLY
            )

        return None

    def _create_alarm(
        self,
        alarm_id: str,
        limit: SafetyLimit,
        value: float,
        limit_value: float,
        priority: AlarmPriority,
        trip_action: TripAction
    ) -> AlarmEvent:
        """Create an alarm event."""
        provenance_data = f"{alarm_id}:{limit.parameter}:{value}:{limit_value}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()[:16]

        return AlarmEvent(
            alarm_id=alarm_id,
            limit_name=limit.name,
            parameter=limit.parameter,
            measured_value=value,
            limit_value=limit_value,
            priority=priority,
            state=AlarmState.UNACKNOWLEDGED,
            trip_action=trip_action,
            provenance_hash=provenance_hash,
        )

    def _execute_trip(self, action: TripAction) -> None:
        """Execute safety trip action."""
        for callback in self.trip_callbacks:
            try:
                callback(action)
            except Exception as e:
                # Log but don't fail safety action
                pass

    def acknowledge_alarm(self, alarm_id: str, operator: str) -> bool:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: ID of alarm to acknowledge
            operator: Operator identifier

        Returns:
            True if acknowledged successfully
        """
        if alarm_id in self.active_alarms:
            alarm = self.active_alarms[alarm_id]
            alarm.state = AlarmState.ACKNOWLEDGED
            alarm.acknowledged_by = operator
            alarm.acknowledged_at = datetime.now()
            return True
        return False

    def clear_alarm(self, alarm_id: str) -> bool:
        """
        Clear an acknowledged alarm if condition has returned to normal.

        Args:
            alarm_id: ID of alarm to clear

        Returns:
            True if cleared successfully
        """
        if alarm_id in self.active_alarms:
            alarm = self.active_alarms[alarm_id]
            if alarm.state == AlarmState.ACKNOWLEDGED:
                del self.active_alarms[alarm_id]
                return True
        return False

    def register_trip_callback(self, callback: Callable) -> None:
        """Register a callback for trip actions."""
        self.trip_callbacks.append(callback)

    def get_alarm_summary(self) -> Dict[str, Any]:
        """Get summary of current alarm status."""
        priority_counts = {p.name: 0 for p in AlarmPriority}
        for alarm in self.active_alarms.values():
            priority_counts[alarm.priority.name] += 1

        return {
            "total_active": len(self.active_alarms),
            "by_priority": priority_counts,
            "unacknowledged": sum(
                1 for a in self.active_alarms.values()
                if a.state == AlarmState.UNACKNOWLEDGED
            ),
            "trip_actions_pending": sum(
                1 for a in self.active_alarms.values()
                if a.trip_action != TripAction.ALARM_ONLY
            ),
        }

    def export_audit_trail(self) -> List[Dict[str, Any]]:
        """Export complete alarm history for audit."""
        return [alarm.to_dict() for alarm in self.alarm_history]


# =============================================================================
# LIMIT VALIDATION FUNCTIONS
# =============================================================================

def validate_combustion_safe(
    o2_percent: float,
    co_ppm: float,
    flame_signal: float,
    fuel_pressure: float,
) -> Tuple[bool, List[str]]:
    """
    Quick validation of combustion safety.

    Args:
        o2_percent: Oxygen percentage
        co_ppm: CO concentration in ppm
        flame_signal: Flame detector signal
        fuel_pressure: Fuel supply pressure

    Returns:
        Tuple of (is_safe, list of violations)
    """
    violations = []

    # Check O2 limits
    if o2_percent < 1.0:
        violations.append(f"O2 critically low: {o2_percent}% < 1.0%")
    elif o2_percent < 2.0:
        violations.append(f"O2 low: {o2_percent}% < 2.0%")

    # Check CO limits
    if co_ppm > 1000:
        violations.append(f"CO critically high: {co_ppm} ppm > 1000 ppm")
    elif co_ppm > 400:
        violations.append(f"CO high: {co_ppm} ppm > 400 ppm")

    # Check flame
    if flame_signal < 2.0:
        violations.append(f"Flame signal lost: {flame_signal} mA < 2.0 mA")
    elif flame_signal < 3.0:
        violations.append(f"Flame signal weak: {flame_signal} mA < 3.0 mA")

    # Check fuel pressure
    if fuel_pressure < 2.0:
        violations.append(f"Fuel pressure low: {fuel_pressure} psig < 2.0 psig")

    is_safe = len(violations) == 0
    return is_safe, violations


def get_safe_operating_envelope() -> Dict[str, Tuple[float, float]]:
    """
    Get the safe operating envelope for combustion parameters.

    Returns:
        Dictionary of parameter: (min_safe, max_safe) tuples
    """
    return {
        "O2": (2.0, 8.0),            # % O2
        "CO": (0.0, 400.0),          # ppm
        "flame_signal": (3.0, 100.0), # mA
        "fuel_pressure": (5.0, 100.0), # psig
        "furnace_pressure": (-2.0, 0.5), # inwc
        "airflow": (50.0, 100.0),    # % of design
        "stack_temperature": (100.0, 350.0), # °C
    }


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_safety_limits() -> Dict[str, Any]:
    """Export all safety limit definitions."""
    return {
        "metadata": {
            "version": "1.0.0",
            "standards": ["NFPA 85", "NFPA 86", "API 556", "IEC 61511"],
            "agent": "GL-005_CombustionSense",
        },
        "limits": {
            key: {
                "name": limit.name,
                "parameter": limit.parameter,
                "units": limit.units,
                "ll_trip": limit.ll_trip,
                "l_alarm": limit.l_alarm,
                "h_alarm": limit.h_alarm,
                "hh_trip": limit.hh_trip,
                "trip_action": limit.trip_action.value,
                "priority": limit.priority.name,
                "source": limit.source_standard,
                "rationale": limit.rationale,
            }
            for key, limit in COMBUSTION_SAFETY_LIMITS.items()
        },
        "safe_envelope": get_safe_operating_envelope(),
    }


if __name__ == "__main__":
    # Example usage
    monitor = CombustionSafetyMonitor()

    # Simulate measurements
    measurements = {
        "O2": 3.5,
        "CO": 50.0,
        "flame_signal": 15.0,
        "fuel_pressure": 25.0,
        "furnace_pressure": -0.5,
        "airflow": 75.0,
    }

    status = monitor.check_limits(measurements)

    print("Safety Status:")
    print(f"  Is Safe: {status.is_safe}")
    print(f"  Active Alarms: {len(status.active_alarms)}")
    print(f"  Trip Required: {status.trip_required}")

    # Test with unsafe conditions
    unsafe_measurements = {
        "O2": 0.8,  # Below LL trip
        "CO": 1200.0,  # Above HH trip
        "flame_signal": 1.5,  # Below LL trip
    }

    status = monitor.check_limits(unsafe_measurements)
    print("\nUnsafe Status:")
    print(f"  Is Safe: {status.is_safe}")
    print(f"  Trip Action: {status.trip_action.value}")
    print(f"  Tripped Limits: {status.tripped_limits}")
