# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Safety Interlock Matrix

Defines all safety interlocks and their conditions per NFPA 85 and IEC 61511.

Interlock Matrix Structure:
- Interlock ID and description
- Trip condition (cause)
- Protected states
- SIL rating (Safety Integrity Level)
- Response action
- Reset requirements

Reference: NFPA 85-2023, IEC 61511
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import logging

logger = logging.getLogger(__name__)


class SafetyIntegrityLevel(str, Enum):
    """IEC 61511 Safety Integrity Levels."""
    SIL_1 = "SIL-1"  # Low demand: 10^-2 to 10^-1 PFD
    SIL_2 = "SIL-2"  # Medium: 10^-3 to 10^-2 PFD
    SIL_3 = "SIL-3"  # High: 10^-4 to 10^-3 PFD
    SIL_4 = "SIL-4"  # Very high: 10^-5 to 10^-4 PFD (rarely used in process)


class InterlockCategory(str, Enum):
    """Interlock categories per NFPA 85."""
    COMBUSTION = "combustion"
    FUEL = "fuel"
    AIR = "air"
    WATER_STEAM = "water_steam"
    FLAME = "flame"
    PURGE = "purge"
    GENERAL = "general"


class ResetType(str, Enum):
    """Interlock reset requirements."""
    AUTOMATIC = "automatic"  # Auto-reset when condition clears
    MANUAL = "manual"  # Requires operator action
    MAINTENANCE = "maintenance"  # Requires maintenance personnel
    ENGINEERING = "engineering"  # Requires engineering review


class InterlockState(str, Enum):
    """Current state of an interlock."""
    NORMAL = "normal"  # No trip, permissive satisfied
    TRIPPED = "tripped"  # Trip condition active
    BYPASSED = "bypassed"  # Interlock bypassed (with authorization)
    FAILED = "failed"  # Interlock sensor/actuator failure


@dataclass
class InterlockCondition:
    """Defines a single interlock trip condition."""
    parameter: str  # e.g., "drum_level", "flame_signal"
    operator: str  # "lt", "gt", "eq", "ne", "range"
    setpoint: float
    deadband: float = 0.0
    delay_s: float = 0.0  # Time delay before trip
    units: str = ""


@dataclass
class InterlockAction:
    """Action to take when interlock trips."""
    action_type: str  # "close_valve", "open_valve", "trip_burner", etc.
    target: str  # Equipment identifier
    timeout_s: float = 0.0  # Max time to complete action
    confirmation_required: bool = True


@dataclass
class Interlock:
    """
    Complete interlock definition.

    Represents a safety interlock per NFPA 85 and IEC 61511.
    """
    interlock_id: str
    description: str
    category: InterlockCategory
    sil_rating: SafetyIntegrityLevel

    # Trip condition(s)
    trip_conditions: List[InterlockCondition]
    logic: str = "AND"  # "AND" or "OR" for multiple conditions

    # Protected states (burner states where interlock is active)
    protected_states: List[str] = field(default_factory=list)

    # Actions on trip
    trip_actions: List[InterlockAction] = field(default_factory=list)

    # Reset requirements
    reset_type: ResetType = ResetType.MANUAL

    # Current state
    state: InterlockState = InterlockState.NORMAL
    last_trip_time: Optional[datetime] = None
    trip_count: int = 0

    # Bypass management
    bypass_allowed: bool = False
    bypass_duration_max_s: float = 3600.0  # 1 hour default
    bypassed_by: Optional[str] = None
    bypass_expires: Optional[datetime] = None

    # Provenance
    nfpa85_reference: str = ""
    iec61511_reference: str = ""


class InterlockMatrix:
    """
    Safety Interlock Matrix Manager.

    Manages all safety interlocks for a boiler system.
    """

    VERSION = "1.0.0"

    def __init__(self, boiler_id: str) -> None:
        """Initialize interlock matrix for a boiler."""
        self.boiler_id = boiler_id
        self._interlocks: Dict[str, Interlock] = {}
        self._trip_history: List[Dict[str, Any]] = []
        self._initialize_standard_interlocks()
        logger.info(f"InterlockMatrix initialized for {boiler_id}")

    def _initialize_standard_interlocks(self) -> None:
        """Initialize standard NFPA 85 interlocks."""

        # IL-001: Low Drum Level
        self._interlocks["IL-001"] = Interlock(
            interlock_id="IL-001",
            description="Low Drum Level Cutoff",
            category=InterlockCategory.WATER_STEAM,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="drum_level_percent",
                    operator="lt",
                    setpoint=-20.0,  # -20% from normal water level
                    deadband=2.0,
                    delay_s=3.0,
                    units="%",
                )
            ],
            protected_states=["PRE_PURGE", "PILOT_LIGHT_TRIAL", "FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="trip_burner",
                    target="main_burner",
                    timeout_s=0.5,
                ),
                InterlockAction(
                    action_type="close_valve",
                    target="main_fuel_valve",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MANUAL,
            nfpa85_reference="5.3.1.4",
        )

        # IL-002: High Steam Pressure
        self._interlocks["IL-002"] = Interlock(
            interlock_id="IL-002",
            description="High Steam Pressure Cutoff",
            category=InterlockCategory.WATER_STEAM,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="steam_pressure_psig",
                    operator="gt",
                    setpoint=175.0,  # Example: 175 psig trip
                    deadband=5.0,
                    units="psig",
                )
            ],
            protected_states=["FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="trip_burner",
                    target="main_burner",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.AUTOMATIC,  # Auto-reset when pressure drops
            nfpa85_reference="5.3.1.5",
        )

        # IL-003: Loss of Flame
        self._interlocks["IL-003"] = Interlock(
            interlock_id="IL-003",
            description="Flame Failure Safety Shutdown",
            category=InterlockCategory.FLAME,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="flame_signal_percent",
                    operator="lt",
                    setpoint=10.0,  # Below 10% = no flame
                    deadband=2.0,
                    delay_s=4.0,  # NFPA 85 max 4 seconds
                    units="%",
                )
            ],
            protected_states=["FIRING", "MAIN_FLAME_PROVEN"],
            trip_actions=[
                InterlockAction(
                    action_type="close_valve",
                    target="main_fuel_valve",
                    timeout_s=0.5,
                ),
                InterlockAction(
                    action_type="close_valve",
                    target="pilot_fuel_valve",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MANUAL,
            nfpa85_reference="5.3.5.2",
        )

        # IL-004: Loss of Combustion Air
        self._interlocks["IL-004"] = Interlock(
            interlock_id="IL-004",
            description="Combustion Air Failure",
            category=InterlockCategory.AIR,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="fd_fan_status",
                    operator="eq",
                    setpoint=0.0,  # Fan stopped
                    units="",
                ),
                InterlockCondition(
                    parameter="air_pressure_wc",
                    operator="lt",
                    setpoint=1.0,  # Minimum air pressure
                    units="in. WC",
                ),
            ],
            logic="OR",  # Trip on either condition
            protected_states=["PRE_PURGE", "PILOT_LIGHT_TRIAL", "FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="close_valve",
                    target="main_fuel_valve",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MANUAL,
            nfpa85_reference="5.3.1.1",
        )

        # IL-005: Low Fuel Pressure
        self._interlocks["IL-005"] = Interlock(
            interlock_id="IL-005",
            description="Low Fuel Gas Pressure",
            category=InterlockCategory.FUEL,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="fuel_pressure_psig",
                    operator="lt",
                    setpoint=3.0,  # Minimum fuel pressure
                    deadband=0.5,
                    delay_s=2.0,
                    units="psig",
                )
            ],
            protected_states=["PILOT_LIGHT_TRIAL", "FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="close_valve",
                    target="main_fuel_valve",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MANUAL,
            nfpa85_reference="5.3.1.2",
        )

        # IL-006: High Fuel Pressure
        self._interlocks["IL-006"] = Interlock(
            interlock_id="IL-006",
            description="High Fuel Gas Pressure",
            category=InterlockCategory.FUEL,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="fuel_pressure_psig",
                    operator="gt",
                    setpoint=15.0,  # Maximum fuel pressure
                    deadband=0.5,
                    units="psig",
                )
            ],
            protected_states=["PILOT_LIGHT_TRIAL", "FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="close_valve",
                    target="main_fuel_valve",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MANUAL,
            nfpa85_reference="5.3.1.3",
        )

        # IL-007: Purge Interlock
        self._interlocks["IL-007"] = Interlock(
            interlock_id="IL-007",
            description="Purge Not Complete",
            category=InterlockCategory.PURGE,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="purge_complete",
                    operator="eq",
                    setpoint=0.0,  # Purge not complete
                    units="",
                )
            ],
            protected_states=["PILOT_LIGHT_TRIAL"],
            trip_actions=[
                InterlockAction(
                    action_type="prevent_ignition",
                    target="igniter",
                ),
            ],
            reset_type=ResetType.AUTOMATIC,
            nfpa85_reference="5.6.4",
        )

        # IL-008: Flame Scanner Failure
        self._interlocks["IL-008"] = Interlock(
            interlock_id="IL-008",
            description="Flame Scanner Self-Test Failure",
            category=InterlockCategory.FLAME,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="scanner_self_test",
                    operator="eq",
                    setpoint=0.0,  # Self-test failed
                    units="",
                )
            ],
            protected_states=["PILOT_LIGHT_TRIAL", "FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="trip_burner",
                    target="main_burner",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MAINTENANCE,
            nfpa85_reference="5.3.3",
        )

        # IL-009: Emergency Stop
        self._interlocks["IL-009"] = Interlock(
            interlock_id="IL-009",
            description="Emergency Stop Button",
            category=InterlockCategory.GENERAL,
            sil_rating=SafetyIntegrityLevel.SIL_2,
            trip_conditions=[
                InterlockCondition(
                    parameter="e_stop_pressed",
                    operator="eq",
                    setpoint=1.0,
                    units="",
                )
            ],
            protected_states=["PRE_PURGE", "PILOT_LIGHT_TRIAL", "FIRING", "POST_PURGE"],
            trip_actions=[
                InterlockAction(
                    action_type="close_valve",
                    target="main_fuel_valve",
                    timeout_s=0.5,
                ),
                InterlockAction(
                    action_type="trip_burner",
                    target="main_burner",
                    timeout_s=0.5,
                ),
            ],
            reset_type=ResetType.MANUAL,
            nfpa85_reference="5.3.4",
        )

        # IL-010: High CO in Flue Gas
        self._interlocks["IL-010"] = Interlock(
            interlock_id="IL-010",
            description="High Carbon Monoxide in Flue Gas",
            category=InterlockCategory.COMBUSTION,
            sil_rating=SafetyIntegrityLevel.SIL_1,
            trip_conditions=[
                InterlockCondition(
                    parameter="flue_gas_co_ppm",
                    operator="gt",
                    setpoint=400.0,  # High CO indicates incomplete combustion
                    deadband=20.0,
                    delay_s=30.0,  # Delay to avoid spurious trips
                    units="ppm",
                )
            ],
            protected_states=["FIRING"],
            trip_actions=[
                InterlockAction(
                    action_type="alarm",
                    target="control_room",
                ),
                InterlockAction(
                    action_type="reduce_firing_rate",
                    target="burner_controller",
                ),
            ],
            reset_type=ResetType.AUTOMATIC,
            bypass_allowed=True,
            bypass_duration_max_s=1800.0,  # 30 minute max bypass
        )

    def get_interlock(self, interlock_id: str) -> Optional[Interlock]:
        """Get interlock by ID."""
        return self._interlocks.get(interlock_id)

    def get_all_interlocks(self) -> Dict[str, Interlock]:
        """Get all interlocks."""
        return self._interlocks.copy()

    def get_interlocks_by_category(
        self,
        category: InterlockCategory,
    ) -> List[Interlock]:
        """Get interlocks by category."""
        return [
            il for il in self._interlocks.values()
            if il.category == category
        ]

    def get_interlocks_for_state(self, state: str) -> List[Interlock]:
        """Get interlocks active for a given burner state."""
        return [
            il for il in self._interlocks.values()
            if state in il.protected_states
        ]

    def check_interlock(
        self,
        interlock_id: str,
        process_values: Dict[str, float],
    ) -> bool:
        """
        Check if an interlock is tripped.

        Returns True if tripped (condition violated).
        """
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if interlock.state == InterlockState.BYPASSED:
            return False  # Bypassed interlocks don't trip

        conditions_met = []
        for condition in interlock.trip_conditions:
            value = process_values.get(condition.parameter, 0.0)

            if condition.operator == "lt":
                met = value < condition.setpoint
            elif condition.operator == "gt":
                met = value > condition.setpoint
            elif condition.operator == "eq":
                met = abs(value - condition.setpoint) < 0.001
            elif condition.operator == "ne":
                met = abs(value - condition.setpoint) >= 0.001
            else:
                met = False

            conditions_met.append(met)

        # Apply logic
        if interlock.logic == "AND":
            tripped = all(conditions_met)
        else:  # OR
            tripped = any(conditions_met)

        return tripped

    def trip_interlock(
        self,
        interlock_id: str,
        reason: str = "",
    ) -> bool:
        """Trip an interlock and record the event."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        interlock.state = InterlockState.TRIPPED
        interlock.last_trip_time = datetime.now(timezone.utc)
        interlock.trip_count += 1

        self._trip_history.append({
            "interlock_id": interlock_id,
            "timestamp": datetime.now(timezone.utc),
            "reason": reason,
            "description": interlock.description,
        })

        logger.warning(f"Interlock {interlock_id} tripped: {interlock.description} - {reason}")
        return True

    def reset_interlock(
        self,
        interlock_id: str,
        operator: str,
    ) -> bool:
        """Reset a tripped interlock."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if interlock.state != InterlockState.TRIPPED:
            return False

        # Check reset requirements
        if interlock.reset_type == ResetType.AUTOMATIC:
            interlock.state = InterlockState.NORMAL
        elif interlock.reset_type == ResetType.MANUAL:
            interlock.state = InterlockState.NORMAL
        elif interlock.reset_type in [ResetType.MAINTENANCE, ResetType.ENGINEERING]:
            # Would require additional authorization checks
            interlock.state = InterlockState.NORMAL

        logger.info(f"Interlock {interlock_id} reset by {operator}")
        return True

    def bypass_interlock(
        self,
        interlock_id: str,
        operator: str,
        duration_s: float,
        reason: str,
    ) -> bool:
        """Bypass an interlock with authorization."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if not interlock.bypass_allowed:
            logger.error(f"Interlock {interlock_id} bypass not allowed")
            return False

        if duration_s > interlock.bypass_duration_max_s:
            logger.error(f"Bypass duration {duration_s}s exceeds max {interlock.bypass_duration_max_s}s")
            return False

        interlock.state = InterlockState.BYPASSED
        interlock.bypassed_by = operator
        interlock.bypass_expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

        logger.warning(
            f"Interlock {interlock_id} bypassed by {operator} for {duration_s}s: {reason}"
        )
        return True

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of interlock matrix status."""
        normal = sum(1 for il in self._interlocks.values() if il.state == InterlockState.NORMAL)
        tripped = sum(1 for il in self._interlocks.values() if il.state == InterlockState.TRIPPED)
        bypassed = sum(1 for il in self._interlocks.values() if il.state == InterlockState.BYPASSED)

        return {
            "boiler_id": self.boiler_id,
            "total_interlocks": len(self._interlocks),
            "normal": normal,
            "tripped": tripped,
            "bypassed": bypassed,
            "trip_history_count": len(self._trip_history),
            "tripped_interlocks": [
                il.interlock_id for il in self._interlocks.values()
                if il.state == InterlockState.TRIPPED
            ],
            "bypassed_interlocks": [
                il.interlock_id for il in self._interlocks.values()
                if il.state == InterlockState.BYPASSED
            ],
        }

    def export_matrix(self) -> List[Dict[str, Any]]:
        """Export interlock matrix as list of dicts for documentation."""
        result = []
        for il in self._interlocks.values():
            result.append({
                "interlock_id": il.interlock_id,
                "description": il.description,
                "category": il.category.value,
                "sil_rating": il.sil_rating.value,
                "protected_states": il.protected_states,
                "reset_type": il.reset_type.value,
                "nfpa85_reference": il.nfpa85_reference,
                "bypass_allowed": il.bypass_allowed,
                "current_state": il.state.value,
            })
        return result


__all__ = [
    "SafetyIntegrityLevel",
    "InterlockCategory",
    "ResetType",
    "InterlockState",
    "InterlockCondition",
    "InterlockAction",
    "Interlock",
    "InterlockMatrix",
]
