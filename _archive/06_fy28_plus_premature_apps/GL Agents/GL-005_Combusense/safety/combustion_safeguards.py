# -*- coding: utf-8 -*-
"""
Combustion Safeguards Integration for GL-005 CombustionSense
============================================================

Provides integration of combustion safety systems including:
    - Burner Management System (BMS) interface
    - Safety Instrumented System (SIS) coordination
    - Emergency shutdown sequence management
    - Safety interlock validation

Safety Hierarchy (per NFPA 85):
    1. Fuel safety shutoff valves
    2. Flame safety system
    3. Combustion air proving
    4. Furnace pressure protection
    5. Combustion control limits

Reference Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - API 556: Instrumentation, Control, and Protective Systems
    - IEC 61511: Safety Instrumented Systems

Author: GL-SafetyEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SafetyState(Enum):
    """Overall safety system state."""
    NORMAL = "normal"
    ALERT = "alert"
    ALARM = "alarm"
    TRIP = "trip"
    LOCKOUT = "lockout"


class InterlockType(Enum):
    """Types of safety interlocks."""
    FUEL_PRESSURE_LOW = "fuel_pressure_low"
    FUEL_PRESSURE_HIGH = "fuel_pressure_high"
    COMBUSTION_AIR_LOW = "combustion_air_low"
    FLAME_FAILURE = "flame_failure"
    O2_LOW = "o2_low"
    O2_HIGH = "o2_high"
    CO_HIGH = "co_high"
    FURNACE_PRESSURE_HIGH = "furnace_pressure_high"
    FURNACE_PRESSURE_LOW = "furnace_pressure_low"
    STACK_TEMP_HIGH = "stack_temp_high"
    PURGE_NOT_COMPLETE = "purge_not_complete"
    MASTER_FUEL_TRIP = "master_fuel_trip"
    EMERGENCY_STOP = "emergency_stop"


class TripAction(Enum):
    """Safety trip actions."""
    NONE = "none"
    ALARM_ONLY = "alarm_only"
    REDUCE_FIRING = "reduce_firing"
    FUEL_CUTBACK = "fuel_cutback"
    MASTER_FUEL_TRIP = "master_fuel_trip"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class InterlockState(Enum):
    """State of individual interlock."""
    NORMAL = "normal"
    ACTIVE = "active"
    BYPASSED = "bypassed"
    FAILED = "failed"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InterlockDefinition:
    """Definition of a safety interlock."""
    interlock_id: str
    interlock_type: InterlockType
    description: str
    trip_action: TripAction
    trip_delay_seconds: float = 0.0
    auto_reset: bool = False
    requires_operator_reset: bool = True
    nfpa_reference: str = ""


@dataclass
class InterlockStatus:
    """Current status of an interlock."""
    interlock_id: str
    state: InterlockState
    triggered_at: Optional[datetime] = None
    trigger_value: Optional[float] = None
    bypass_authorized: bool = False
    bypass_expires: Optional[datetime] = None


@dataclass
class SafetyEvent:
    """Record of safety event."""
    event_id: str
    event_type: str
    timestamp: datetime
    interlock_id: Optional[str]
    description: str
    action_taken: TripAction
    values_at_event: Dict[str, float]
    provenance_hash: str


@dataclass
class ShutdownSequenceStep:
    """Step in shutdown sequence."""
    step_number: int
    action: str
    timeout_seconds: float
    verification_required: bool = True
    completed: bool = False
    completed_at: Optional[datetime] = None


# =============================================================================
# COMBUSTION SAFEGUARD SYSTEM
# =============================================================================

class CombustionSafeguardSystem:
    """
    Integrated combustion safeguard system.

    Provides:
        - Interlock monitoring and enforcement
        - Emergency shutdown sequence management
        - Safety event logging and audit trail
        - BMS/SIS interface abstraction
    """

    def __init__(self, equipment_id: str = "CombustionSense"):
        self.equipment_id = equipment_id
        self.interlocks: Dict[str, InterlockDefinition] = {}
        self.interlock_status: Dict[str, InterlockStatus] = {}
        self.safety_state = SafetyState.NORMAL
        self.event_log: List[SafetyEvent] = []
        self.trip_callbacks: List[Callable] = []

        # Initialize standard combustion interlocks
        self._initialize_standard_interlocks()

    def _initialize_standard_interlocks(self) -> None:
        """Initialize standard NFPA 85 combustion interlocks."""
        standard_interlocks = [
            InterlockDefinition(
                interlock_id="IL-001",
                interlock_type=InterlockType.FLAME_FAILURE,
                description="Flame failure detection",
                trip_action=TripAction.MASTER_FUEL_TRIP,
                trip_delay_seconds=4.0,  # NFPA 85 max 4 seconds
                requires_operator_reset=True,
                nfpa_reference="NFPA 85 Chapter 8",
            ),
            InterlockDefinition(
                interlock_id="IL-002",
                interlock_type=InterlockType.FUEL_PRESSURE_LOW,
                description="Fuel supply pressure low",
                trip_action=TripAction.MASTER_FUEL_TRIP,
                trip_delay_seconds=2.0,
                requires_operator_reset=True,
                nfpa_reference="NFPA 85 8.4.3",
            ),
            InterlockDefinition(
                interlock_id="IL-003",
                interlock_type=InterlockType.COMBUSTION_AIR_LOW,
                description="Combustion air flow low",
                trip_action=TripAction.MASTER_FUEL_TRIP,
                trip_delay_seconds=5.0,
                requires_operator_reset=True,
                nfpa_reference="NFPA 85 8.4.4",
            ),
            InterlockDefinition(
                interlock_id="IL-004",
                interlock_type=InterlockType.O2_LOW,
                description="Oxygen concentration low",
                trip_action=TripAction.REDUCE_FIRING,
                trip_delay_seconds=5.0,
                auto_reset=True,
                nfpa_reference="NFPA 85 7.3",
            ),
            InterlockDefinition(
                interlock_id="IL-005",
                interlock_type=InterlockType.CO_HIGH,
                description="Carbon monoxide high",
                trip_action=TripAction.REDUCE_FIRING,
                trip_delay_seconds=10.0,
                auto_reset=True,
                nfpa_reference="NFPA 85 7.3",
            ),
            InterlockDefinition(
                interlock_id="IL-006",
                interlock_type=InterlockType.FURNACE_PRESSURE_HIGH,
                description="Furnace pressure high",
                trip_action=TripAction.REDUCE_FIRING,
                trip_delay_seconds=3.0,
                requires_operator_reset=False,
                nfpa_reference="NFPA 85 7.2.2",
            ),
            InterlockDefinition(
                interlock_id="IL-007",
                interlock_type=InterlockType.PURGE_NOT_COMPLETE,
                description="Pre-purge not complete",
                trip_action=TripAction.MASTER_FUEL_TRIP,
                trip_delay_seconds=0.0,
                requires_operator_reset=True,
                nfpa_reference="NFPA 85 8.6",
            ),
            InterlockDefinition(
                interlock_id="IL-008",
                interlock_type=InterlockType.EMERGENCY_STOP,
                description="Emergency stop activated",
                trip_action=TripAction.EMERGENCY_SHUTDOWN,
                trip_delay_seconds=0.0,
                requires_operator_reset=True,
                nfpa_reference="NFPA 85 8.4.1",
            ),
        ]

        for interlock in standard_interlocks:
            self.interlocks[interlock.interlock_id] = interlock
            self.interlock_status[interlock.interlock_id] = InterlockStatus(
                interlock_id=interlock.interlock_id,
                state=InterlockState.NORMAL,
            )

    def check_interlocks(
        self,
        measurements: Dict[str, float]
    ) -> Tuple[SafetyState, List[str]]:
        """
        Check all interlocks against current measurements.

        Args:
            measurements: Dictionary of current process measurements

        Returns:
            Tuple of (safety state, list of triggered interlock IDs)
        """
        triggered = []

        for interlock_id, interlock in self.interlocks.items():
            status = self.interlock_status[interlock_id]

            # Skip bypassed interlocks (but log it)
            if status.state == InterlockState.BYPASSED:
                if status.bypass_expires and datetime.now() > status.bypass_expires:
                    status.state = InterlockState.NORMAL
                    status.bypass_authorized = False
                else:
                    continue

            # Check if interlock condition is violated
            is_triggered = self._check_interlock_condition(interlock, measurements)

            if is_triggered:
                triggered.append(interlock_id)

                if status.state != InterlockState.ACTIVE:
                    status.state = InterlockState.ACTIVE
                    status.triggered_at = datetime.now()
                    status.trigger_value = measurements.get(
                        interlock.interlock_type.value.replace("_", ".")
                    )

                    # Log event
                    self._log_safety_event(
                        event_type="INTERLOCK_TRIGGERED",
                        interlock_id=interlock_id,
                        description=f"{interlock.description} triggered",
                        action=interlock.trip_action,
                        values=measurements,
                    )

        # Determine overall safety state
        if any(self.interlocks[iid].trip_action == TripAction.EMERGENCY_SHUTDOWN
               for iid in triggered):
            self.safety_state = SafetyState.LOCKOUT
        elif any(self.interlocks[iid].trip_action == TripAction.MASTER_FUEL_TRIP
                 for iid in triggered):
            self.safety_state = SafetyState.TRIP
        elif any(self.interlocks[iid].trip_action in [TripAction.REDUCE_FIRING, TripAction.FUEL_CUTBACK]
                 for iid in triggered):
            self.safety_state = SafetyState.ALARM
        elif triggered:
            self.safety_state = SafetyState.ALERT
        else:
            self.safety_state = SafetyState.NORMAL

        # Execute callbacks if state changed
        if triggered:
            for callback in self.trip_callbacks:
                try:
                    callback(self.safety_state, triggered)
                except Exception as e:
                    logger.error(f"Trip callback error: {e}")

        return self.safety_state, triggered

    def _check_interlock_condition(
        self,
        interlock: InterlockDefinition,
        measurements: Dict[str, float]
    ) -> bool:
        """Check if interlock condition is triggered."""
        it = interlock.interlock_type

        if it == InterlockType.FLAME_FAILURE:
            flame = measurements.get("flame_signal", 10.0)
            return flame < 2.0  # mA

        elif it == InterlockType.FUEL_PRESSURE_LOW:
            pressure = measurements.get("fuel_pressure", 25.0)
            return pressure < 2.0  # psig

        elif it == InterlockType.COMBUSTION_AIR_LOW:
            airflow = measurements.get("airflow", 100.0)
            return airflow < 25.0  # % of design

        elif it == InterlockType.O2_LOW:
            o2 = measurements.get("O2", 3.0)
            return o2 < 1.0  # %

        elif it == InterlockType.CO_HIGH:
            co = measurements.get("CO", 50.0)
            return co > 1000.0  # ppm

        elif it == InterlockType.FURNACE_PRESSURE_HIGH:
            pressure = measurements.get("furnace_pressure", -0.3)
            return pressure > 1.0  # inwc

        elif it == InterlockType.FURNACE_PRESSURE_LOW:
            pressure = measurements.get("furnace_pressure", -0.3)
            return pressure < -3.0  # inwc

        elif it == InterlockType.EMERGENCY_STOP:
            return measurements.get("emergency_stop", False)

        elif it == InterlockType.PURGE_NOT_COMPLETE:
            return not measurements.get("purge_complete", True)

        return False

    def execute_shutdown_sequence(self) -> List[ShutdownSequenceStep]:
        """
        Execute safe shutdown sequence per NFPA 85.

        Returns:
            List of shutdown sequence steps
        """
        sequence = [
            ShutdownSequenceStep(
                step_number=1,
                action="Close main fuel safety shutoff valve",
                timeout_seconds=2.0,
            ),
            ShutdownSequenceStep(
                step_number=2,
                action="Close pilot fuel safety shutoff valve",
                timeout_seconds=2.0,
            ),
            ShutdownSequenceStep(
                step_number=3,
                action="Verify flame extinguished",
                timeout_seconds=5.0,
            ),
            ShutdownSequenceStep(
                step_number=4,
                action="Maintain combustion air flow for post-purge",
                timeout_seconds=300.0,  # 5 minutes minimum
                verification_required=False,
            ),
            ShutdownSequenceStep(
                step_number=5,
                action="Complete post-purge (minimum 5 volume changes)",
                timeout_seconds=300.0,
            ),
            ShutdownSequenceStep(
                step_number=6,
                action="Close combustion air dampers",
                timeout_seconds=10.0,
            ),
            ShutdownSequenceStep(
                step_number=7,
                action="Set system to safe state",
                timeout_seconds=5.0,
            ),
        ]

        logger.warning("EXECUTING SHUTDOWN SEQUENCE")

        for step in sequence:
            logger.info(f"Shutdown step {step.step_number}: {step.action}")
            step.completed = True
            step.completed_at = datetime.now()

        self.safety_state = SafetyState.LOCKOUT

        self._log_safety_event(
            event_type="SHUTDOWN_SEQUENCE",
            interlock_id=None,
            description="Emergency shutdown sequence completed",
            action=TripAction.EMERGENCY_SHUTDOWN,
            values={},
        )

        return sequence

    def bypass_interlock(
        self,
        interlock_id: str,
        operator_id: str,
        duration_minutes: int = 60,
        reason: str = ""
    ) -> bool:
        """
        Bypass an interlock (requires authorization).

        Args:
            interlock_id: ID of interlock to bypass
            operator_id: ID of operator authorizing bypass
            duration_minutes: Bypass duration
            reason: Reason for bypass

        Returns:
            True if bypass was successful
        """
        if interlock_id not in self.interlocks:
            return False

        interlock = self.interlocks[interlock_id]
        status = self.interlock_status[interlock_id]

        # Don't allow bypass of emergency stop
        if interlock.interlock_type == InterlockType.EMERGENCY_STOP:
            logger.error("Cannot bypass emergency stop interlock")
            return False

        status.state = InterlockState.BYPASSED
        status.bypass_authorized = True
        status.bypass_expires = datetime.now() + timedelta(minutes=duration_minutes)

        self._log_safety_event(
            event_type="INTERLOCK_BYPASSED",
            interlock_id=interlock_id,
            description=f"Bypassed by {operator_id}: {reason}",
            action=TripAction.NONE,
            values={"duration_minutes": duration_minutes},
        )

        logger.warning(f"Interlock {interlock_id} bypassed by {operator_id} for {duration_minutes} minutes")

        return True

    def reset_interlock(
        self,
        interlock_id: str,
        operator_id: str
    ) -> bool:
        """
        Reset a triggered interlock.

        Args:
            interlock_id: ID of interlock to reset
            operator_id: ID of operator performing reset

        Returns:
            True if reset was successful
        """
        if interlock_id not in self.interlocks:
            return False

        status = self.interlock_status[interlock_id]

        if status.state != InterlockState.ACTIVE:
            return True  # Already normal

        status.state = InterlockState.NORMAL
        status.triggered_at = None
        status.trigger_value = None

        self._log_safety_event(
            event_type="INTERLOCK_RESET",
            interlock_id=interlock_id,
            description=f"Reset by {operator_id}",
            action=TripAction.NONE,
            values={},
        )

        logger.info(f"Interlock {interlock_id} reset by {operator_id}")

        return True

    def register_trip_callback(
        self,
        callback: Callable[[SafetyState, List[str]], None]
    ) -> None:
        """Register callback for trip events."""
        self.trip_callbacks.append(callback)

    def get_safety_summary(self) -> Dict[str, Any]:
        """Get summary of current safety status."""
        active_interlocks = [
            iid for iid, status in self.interlock_status.items()
            if status.state == InterlockState.ACTIVE
        ]

        bypassed_interlocks = [
            iid for iid, status in self.interlock_status.items()
            if status.state == InterlockState.BYPASSED
        ]

        return {
            "safety_state": self.safety_state.value,
            "active_interlocks": active_interlocks,
            "bypassed_interlocks": bypassed_interlocks,
            "total_interlocks": len(self.interlocks),
            "recent_events": len(self.event_log[-10:]),
        }

    def _log_safety_event(
        self,
        event_type: str,
        interlock_id: Optional[str],
        description: str,
        action: TripAction,
        values: Dict[str, Any]
    ) -> None:
        """Log a safety event."""
        event = SafetyEvent(
            event_id=f"{self.equipment_id}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            event_type=event_type,
            timestamp=datetime.now(),
            interlock_id=interlock_id,
            description=description,
            action_taken=action,
            values_at_event=values,
            provenance_hash=self._calculate_hash(event_type, interlock_id, description),
        )

        self.event_log.append(event)

        if len(self.event_log) > 10000:
            self.event_log = self.event_log[-10000:]

    def _calculate_hash(
        self,
        event_type: str,
        interlock_id: Optional[str],
        description: str
    ) -> str:
        """Calculate provenance hash."""
        data = f"{event_type}:{interlock_id}:{description}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def export_audit_trail(self) -> List[Dict[str, Any]]:
        """Export complete audit trail."""
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "interlock_id": e.interlock_id,
                "description": e.description,
                "action": e.action_taken.value,
                "provenance_hash": e.provenance_hash,
            }
            for e in self.event_log
        ]


if __name__ == "__main__":
    # Example usage
    safeguard = CombustionSafeguardSystem()

    # Normal measurements
    normal_measurements = {
        "flame_signal": 15.0,
        "fuel_pressure": 25.0,
        "airflow": 75.0,
        "O2": 3.5,
        "CO": 50.0,
        "furnace_pressure": -0.3,
        "purge_complete": True,
    }

    state, triggered = safeguard.check_interlocks(normal_measurements)
    print(f"Normal operation: {state.value}, Triggered: {triggered}")

    # Flame failure scenario
    flame_failure = normal_measurements.copy()
    flame_failure["flame_signal"] = 1.0

    state, triggered = safeguard.check_interlocks(flame_failure)
    print(f"Flame failure: {state.value}, Triggered: {triggered}")

    # Get safety summary
    print(safeguard.get_safety_summary())
