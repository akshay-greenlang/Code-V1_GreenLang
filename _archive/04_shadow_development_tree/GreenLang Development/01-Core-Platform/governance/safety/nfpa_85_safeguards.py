"""
NFPA 85 Combustion Safeguards for GL-018 UNIFIEDCOMBUSTION

This module implements comprehensive NFPA 85 "Boiler and Combustion Systems
Hazards Code" requirements for the GreenLang Process Heat system.

Key NFPA 85 Requirements Implemented:
- Flame failure response (<4 seconds per Section 8.5.2.2)
- Prepurge timing (minimum 4 volume changes per Section 8.4.2)
- Pilot/main flame trial timing (10 seconds max per Section 8.5.4)
- Safety interlocks (low/high fuel pressure, low air, flame failure,
  high steam pressure, low water level per Section 8.6.3)
- Burner Management System (BMS) state machine per Section 8.4

Reference: NFPA 85-2019, Boiler and Combustion Systems Hazards Code

Example:
    >>> from greenlang.safety.nfpa_85_safeguards import (
    ...     NFPA85SafeguardManager,
    ...     BMSStateMachine,
    ...     CombustionSafetyInterlock
    ... )
    >>> manager = NFPA85SafeguardManager(equipment_id="BLR-001")
    >>> status = manager.get_safety_status()

Author: GreenLang Safety Engineering Team
Version: 2.0
Date: 2025-12-06
Classification: Safety Critical Code - SIL 2
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from threading import Lock, Timer
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# NFPA 85 Timing Constants
# =============================================================================

class NFPA85TimingRequirements:
    """
    NFPA 85 mandated timing requirements.

    These values are derived directly from NFPA 85-2019 and SHALL NOT
    be modified without formal Management of Change (MOC) approval.
    """

    # Section 8.5.2.2 - Flame Failure Response
    FLAME_FAILURE_RESPONSE_MAX_SECONDS: float = 4.0

    # Section 8.4.2 - Prepurge Requirements
    PREPURGE_MIN_VOLUME_CHANGES: int = 4
    PREPURGE_MIN_AIRFLOW_PERCENT: float = 25.0
    PREPURGE_MIN_TIME_SECONDS: float = 15.0  # Minimum absolute time

    # Section 8.5.4 - Trial for Ignition
    PILOT_TRIAL_MAX_SECONDS: float = 10.0
    MAIN_FLAME_TRIAL_MAX_SECONDS: float = 10.0
    TOTAL_TRIAL_MAX_SECONDS: float = 15.0  # Pilot + Main combined max

    # Section 8.4.3 - Postpurge Requirements
    POSTPURGE_MIN_TIME_SECONDS: float = 15.0

    # Safety Shutdown Timing
    FUEL_VALVE_CLOSURE_MAX_SECONDS: float = 3.0
    SAFETY_SHUTDOWN_COMPLETE_MAX_SECONDS: float = 5.0

    # Interlock Response Times
    LOW_FUEL_PRESSURE_RESPONSE_SECONDS: float = 3.0
    HIGH_FUEL_PRESSURE_RESPONSE_SECONDS: float = 3.0
    LOW_AIR_FLOW_RESPONSE_SECONDS: float = 3.0
    HIGH_STEAM_PRESSURE_RESPONSE_SECONDS: float = 5.0
    LOW_WATER_LEVEL_RESPONSE_SECONDS: float = 3.0


# =============================================================================
# Enumerations
# =============================================================================

class BMSState(Enum):
    """
    Burner Management System states per NFPA 85 Section 8.4.

    State transitions follow strict sequencing requirements.
    """
    IDLE = auto()              # System idle, fuel valves proven closed
    PERMISSIVE_CHECK = auto()  # Verifying all permissives
    PREPURGE = auto()          # Prepurge in progress (4 volume changes)
    PILOT_TRIAL = auto()       # Pilot ignition trial (max 10 sec)
    MAIN_TRIAL = auto()        # Main flame trial (max 10 sec)
    RUN = auto()               # Normal modulating operation
    POSTPURGE = auto()         # Postpurge after shutdown
    SAFETY_SHUTDOWN = auto()   # Safety shutdown in progress
    LOCKOUT = auto()           # Lockout requiring manual reset
    MAINTENANCE = auto()       # Maintenance mode with bypasses


class InterlockType(Enum):
    """Safety interlock types per NFPA 85 Section 8.6.3."""
    LOW_FUEL_PRESSURE = "low_fuel_pressure"
    HIGH_FUEL_PRESSURE = "high_fuel_pressure"
    LOW_COMBUSTION_AIR = "low_combustion_air"
    FLAME_FAILURE = "flame_failure"
    HIGH_STEAM_PRESSURE = "high_steam_pressure"
    LOW_WATER_LEVEL = "low_water_level"
    HIGH_FUEL_GAS_PRESSURE = "high_fuel_gas_pressure"
    LOW_FUEL_GAS_PRESSURE = "low_fuel_gas_pressure"
    LOW_ATOMIZING_MEDIA = "low_atomizing_media"
    HIGH_FURNACE_PRESSURE = "high_furnace_pressure"
    LOW_FURNACE_PRESSURE = "low_furnace_pressure"
    MANUAL_ESD = "manual_esd"
    EXTERNAL_ESD = "external_esd"


class InterlockStatus(Enum):
    """Interlock status states."""
    NORMAL = auto()            # Interlock healthy, not tripped
    TRIPPED = auto()           # Interlock has tripped
    BYPASSED = auto()          # Interlock bypassed (time-limited)
    FAULT = auto()             # Interlock circuit fault detected
    UNKNOWN = auto()           # Status cannot be determined


class FlameStatus(Enum):
    """Flame detection status per NFPA 85 Section 8.5."""
    FLAME_ON = auto()          # Stable flame detected
    FLAME_OFF = auto()         # No flame detected
    FLAME_UNSTABLE = auto()    # Flame signal intermittent
    SCANNER_FAULT = auto()     # Flame scanner malfunction
    UNKNOWN = auto()           # Status unknown


class ShutdownType(Enum):
    """Types of safety shutdown per NFPA 85."""
    NORMAL = auto()            # Operator-initiated normal shutdown
    SAFETY = auto()            # Safety interlock trip
    EMERGENCY = auto()         # Emergency shutdown (ESD)
    FLAME_FAILURE = auto()     # Flame failure shutdown


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FlameReading:
    """Flame scanner reading with metadata."""
    signal_strength: float     # 0-100%
    scanner_id: str
    timestamp: datetime
    is_valid: bool
    diagnostic_status: str = "OK"

    def __post_init__(self):
        """Validate flame reading."""
        if not 0 <= self.signal_strength <= 100:
            raise ValueError(f"Signal strength must be 0-100%, got {self.signal_strength}")


@dataclass
class InterlockReading:
    """Safety interlock sensor reading."""
    interlock_type: InterlockType
    value: float
    unit: str
    setpoint: float
    status: InterlockStatus
    timestamp: datetime
    sensor_id: str
    is_bypassed: bool = False
    bypass_expires: Optional[datetime] = None


@dataclass
class PurgeStatus:
    """Prepurge/postpurge status tracking."""
    is_active: bool
    start_time: Optional[datetime]
    target_duration_seconds: float
    elapsed_seconds: float
    airflow_percent: float
    volume_changes_completed: float
    is_complete: bool

    @property
    def remaining_seconds(self) -> float:
        """Calculate remaining purge time."""
        if self.is_complete:
            return 0.0
        return max(0.0, self.target_duration_seconds - self.elapsed_seconds)


@dataclass
class SafetyEvent:
    """Safety event record for audit trail."""
    event_id: str
    event_type: str
    timestamp: datetime
    equipment_id: str
    description: str
    severity: str  # INFO, WARNING, ALARM, TRIP
    interlock_type: Optional[InterlockType] = None
    bms_state: Optional[BMSState] = None
    response_time_ms: Optional[float] = None
    provenance_hash: str = ""

    def __post_init__(self):
        """Generate provenance hash."""
        data = f"{self.event_id}|{self.event_type}|{self.timestamp.isoformat()}|{self.equipment_id}"
        self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()


@dataclass
class BypassRecord:
    """Bypass authorization record per NFPA 85 Section 8.6.5."""
    bypass_id: str
    interlock_type: InterlockType
    authorized_by: str
    authorization_level: str  # OPERATOR, SUPERVISOR, SAFETY_ENGINEER
    start_time: datetime
    max_duration_hours: float
    reason: str
    equipment_id: str
    is_active: bool = True

    @property
    def expires_at(self) -> datetime:
        """Calculate bypass expiration time."""
        return self.start_time + timedelta(hours=self.max_duration_hours)

    @property
    def is_expired(self) -> bool:
        """Check if bypass has expired."""
        return datetime.utcnow() > self.expires_at


# =============================================================================
# Interlock Class
# =============================================================================

class CombustionSafetyInterlock:
    """
    Individual safety interlock implementation per NFPA 85 Section 8.6.3.

    Each interlock monitors a specific process condition and initiates
    safety shutdown when the condition exceeds safe limits.

    Attributes:
        interlock_type: Type of interlock from InterlockType enum
        setpoint: Trip setpoint value
        deadband: Deadband for reset (prevents hunting)
        response_time_limit: Maximum allowed response time in seconds
        requires_manual_reset: Whether manual reset is required after trip
    """

    def __init__(
        self,
        interlock_type: InterlockType,
        setpoint: float,
        unit: str,
        is_high_trip: bool = True,
        deadband: float = 0.0,
        response_time_limit: float = 3.0,
        requires_manual_reset: bool = True,
        nfpa_clause: str = "8.6.3"
    ):
        """
        Initialize safety interlock.

        Args:
            interlock_type: Type of safety interlock
            setpoint: Trip setpoint value
            unit: Engineering unit for setpoint
            is_high_trip: True if trips on high value, False for low
            deadband: Reset deadband (value must be setpoint +/- deadband to reset)
            response_time_limit: Maximum response time in seconds
            requires_manual_reset: If True, requires manual reset after trip
            nfpa_clause: NFPA 85 clause reference
        """
        self.interlock_type = interlock_type
        self.setpoint = setpoint
        self.unit = unit
        self.is_high_trip = is_high_trip
        self.deadband = deadband
        self.response_time_limit = response_time_limit
        self.requires_manual_reset = requires_manual_reset
        self.nfpa_clause = nfpa_clause

        self._status = InterlockStatus.NORMAL
        self._current_value: Optional[float] = None
        self._trip_time: Optional[datetime] = None
        self._last_reading_time: Optional[datetime] = None
        self._is_bypassed = False
        self._bypass_record: Optional[BypassRecord] = None
        self._trip_count = 0
        self._lock = Lock()

        logger.info(
            f"Initialized interlock {interlock_type.value}: "
            f"setpoint={setpoint} {unit}, "
            f"{'HIGH' if is_high_trip else 'LOW'} trip"
        )

    @property
    def status(self) -> InterlockStatus:
        """Get current interlock status."""
        with self._lock:
            return self._status

    @property
    def is_tripped(self) -> bool:
        """Check if interlock is currently tripped."""
        with self._lock:
            return self._status == InterlockStatus.TRIPPED

    @property
    def is_bypassed(self) -> bool:
        """Check if interlock is currently bypassed."""
        with self._lock:
            if self._bypass_record and self._bypass_record.is_expired:
                self._clear_bypass()
            return self._is_bypassed

    @property
    def is_healthy(self) -> bool:
        """Check if interlock is in healthy state (not tripped, not faulted)."""
        with self._lock:
            return self._status in (InterlockStatus.NORMAL, InterlockStatus.BYPASSED)

    def process_reading(self, value: float, timestamp: datetime) -> Tuple[bool, Optional[SafetyEvent]]:
        """
        Process a sensor reading and determine if trip is required.

        Args:
            value: Current sensor value
            timestamp: Timestamp of reading

        Returns:
            Tuple of (trip_required, safety_event if trip occurred)
        """
        with self._lock:
            self._current_value = value
            self._last_reading_time = timestamp

            # Check for trip condition
            trip_required = False
            if self.is_high_trip:
                trip_required = value >= self.setpoint
            else:
                trip_required = value <= self.setpoint

            # If bypassed, log but don't trip
            if self._is_bypassed and trip_required:
                logger.warning(
                    f"Interlock {self.interlock_type.value} would trip "
                    f"(value={value} {self.unit}) but is BYPASSED"
                )
                return False, None

            # Process trip
            if trip_required and self._status != InterlockStatus.TRIPPED:
                self._status = InterlockStatus.TRIPPED
                self._trip_time = timestamp
                self._trip_count += 1

                event = SafetyEvent(
                    event_id=f"TRIP-{uuid.uuid4().hex[:8].upper()}",
                    event_type="INTERLOCK_TRIP",
                    timestamp=timestamp,
                    equipment_id="",  # Set by caller
                    description=(
                        f"{self.interlock_type.value} trip: "
                        f"value={value:.2f} {self.unit}, "
                        f"setpoint={self.setpoint:.2f} {self.unit}"
                    ),
                    severity="TRIP",
                    interlock_type=self.interlock_type
                )

                logger.critical(
                    f"INTERLOCK TRIP: {self.interlock_type.value} - "
                    f"{value:.2f} {self.unit} {'>' if self.is_high_trip else '<'} "
                    f"{self.setpoint:.2f} {self.unit}"
                )

                return True, event

            return False, None

    def check_reset_conditions(self, current_value: float) -> bool:
        """
        Check if reset conditions are met.

        Args:
            current_value: Current sensor value

        Returns:
            True if reset conditions are satisfied
        """
        with self._lock:
            if self._status != InterlockStatus.TRIPPED:
                return True

            # Check value is within safe range with deadband
            if self.is_high_trip:
                safe = current_value < (self.setpoint - self.deadband)
            else:
                safe = current_value > (self.setpoint + self.deadband)

            return safe

    def reset(self, authorized_by: str) -> bool:
        """
        Attempt to reset the interlock.

        Args:
            authorized_by: ID of person authorizing reset

        Returns:
            True if reset successful
        """
        with self._lock:
            if self._status != InterlockStatus.TRIPPED:
                return True

            if self._current_value is not None:
                if not self.check_reset_conditions(self._current_value):
                    logger.warning(
                        f"Cannot reset {self.interlock_type.value}: "
                        f"process value {self._current_value} still in trip range"
                    )
                    return False

            self._status = InterlockStatus.NORMAL
            self._trip_time = None

            logger.info(
                f"Interlock {self.interlock_type.value} reset by {authorized_by}"
            )
            return True

    def set_bypass(self, bypass_record: BypassRecord) -> bool:
        """
        Set interlock bypass with authorization.

        Args:
            bypass_record: Bypass authorization record

        Returns:
            True if bypass set successfully
        """
        with self._lock:
            if self._status == InterlockStatus.TRIPPED:
                logger.error(
                    f"Cannot bypass tripped interlock {self.interlock_type.value}"
                )
                return False

            self._is_bypassed = True
            self._bypass_record = bypass_record
            self._status = InterlockStatus.BYPASSED

            logger.warning(
                f"BYPASS SET: {self.interlock_type.value} "
                f"by {bypass_record.authorized_by} "
                f"for {bypass_record.max_duration_hours} hours - "
                f"Reason: {bypass_record.reason}"
            )
            return True

    def clear_bypass(self) -> None:
        """Clear interlock bypass."""
        with self._lock:
            self._clear_bypass()

    def _clear_bypass(self) -> None:
        """Internal bypass clear (must hold lock)."""
        if self._is_bypassed:
            logger.info(f"Bypass cleared for {self.interlock_type.value}")
        self._is_bypassed = False
        self._bypass_record = None
        if self._status == InterlockStatus.BYPASSED:
            self._status = InterlockStatus.NORMAL

    def get_reading(self) -> InterlockReading:
        """Get current interlock reading."""
        with self._lock:
            return InterlockReading(
                interlock_type=self.interlock_type,
                value=self._current_value or 0.0,
                unit=self.unit,
                setpoint=self.setpoint,
                status=self._status,
                timestamp=self._last_reading_time or datetime.utcnow(),
                sensor_id=f"INTLK-{self.interlock_type.value}",
                is_bypassed=self._is_bypassed,
                bypass_expires=(
                    self._bypass_record.expires_at
                    if self._bypass_record else None
                )
            )


# =============================================================================
# BMS State Machine
# =============================================================================

class BMSStateMachine:
    """
    Burner Management System state machine per NFPA 85 Section 8.4.

    Implements the complete BMS startup, run, and shutdown sequences
    with all required permissive checks and timing requirements.

    The state machine enforces:
    - Prepurge requirements (4 volume changes, minimum airflow)
    - Pilot trial timing (max 10 seconds)
    - Main flame trial timing (max 10 seconds)
    - Flame failure response (< 4 seconds)
    - Safe shutdown sequencing
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        BMSState.IDLE: {BMSState.PERMISSIVE_CHECK, BMSState.MAINTENANCE},
        BMSState.PERMISSIVE_CHECK: {BMSState.PREPURGE, BMSState.IDLE, BMSState.LOCKOUT},
        BMSState.PREPURGE: {BMSState.PILOT_TRIAL, BMSState.SAFETY_SHUTDOWN, BMSState.LOCKOUT},
        BMSState.PILOT_TRIAL: {BMSState.MAIN_TRIAL, BMSState.SAFETY_SHUTDOWN, BMSState.LOCKOUT},
        BMSState.MAIN_TRIAL: {BMSState.RUN, BMSState.SAFETY_SHUTDOWN, BMSState.LOCKOUT},
        BMSState.RUN: {BMSState.POSTPURGE, BMSState.SAFETY_SHUTDOWN, BMSState.LOCKOUT},
        BMSState.POSTPURGE: {BMSState.IDLE, BMSState.LOCKOUT},
        BMSState.SAFETY_SHUTDOWN: {BMSState.POSTPURGE, BMSState.LOCKOUT},
        BMSState.LOCKOUT: {BMSState.IDLE},  # Requires manual reset
        BMSState.MAINTENANCE: {BMSState.IDLE},
    }

    def __init__(
        self,
        equipment_id: str,
        furnace_volume_cubic_feet: float,
        max_airflow_cfm: float,
        on_state_change: Optional[Callable[[BMSState, BMSState], None]] = None,
        on_safety_event: Optional[Callable[[SafetyEvent], None]] = None
    ):
        """
        Initialize BMS state machine.

        Args:
            equipment_id: Equipment identifier
            furnace_volume_cubic_feet: Furnace volume for purge calculation
            max_airflow_cfm: Maximum airflow rate in CFM
            on_state_change: Callback for state changes
            on_safety_event: Callback for safety events
        """
        self.equipment_id = equipment_id
        self.furnace_volume = furnace_volume_cubic_feet
        self.max_airflow_cfm = max_airflow_cfm
        self.on_state_change = on_state_change
        self.on_safety_event = on_safety_event

        self._state = BMSState.IDLE
        self._previous_state = BMSState.IDLE
        self._state_entry_time = datetime.utcnow()
        self._lock = Lock()

        # Timing trackers
        self._purge_start_time: Optional[datetime] = None
        self._pilot_trial_start: Optional[datetime] = None
        self._main_trial_start: Optional[datetime] = None
        self._flame_loss_time: Optional[datetime] = None

        # Status flags
        self._flame_proven = False
        self._pilot_proven = False
        self._all_permissives_ok = False

        # Event history
        self._event_history: List[SafetyEvent] = []
        self._state_history: List[Tuple[datetime, BMSState, BMSState]] = []

        logger.info(
            f"BMS State Machine initialized for {equipment_id}: "
            f"volume={furnace_volume_cubic_feet} ft3, "
            f"max_airflow={max_airflow_cfm} CFM"
        )

    @property
    def state(self) -> BMSState:
        """Get current BMS state."""
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Check if burner is in RUN state."""
        with self._lock:
            return self._state == BMSState.RUN

    @property
    def is_safe(self) -> bool:
        """Check if system is in a safe state."""
        with self._lock:
            return self._state in (
                BMSState.IDLE,
                BMSState.POSTPURGE,
                BMSState.LOCKOUT
            )

    @property
    def time_in_state(self) -> float:
        """Get time in current state in seconds."""
        with self._lock:
            return (datetime.utcnow() - self._state_entry_time).total_seconds()

    def calculate_prepurge_time(self, airflow_percent: float) -> float:
        """
        Calculate required prepurge time for given airflow.

        Per NFPA 85 Section 8.4.2, prepurge must provide minimum
        4 furnace volume changes at not less than 25% airflow.

        Args:
            airflow_percent: Current airflow as percentage of max

        Returns:
            Required prepurge time in seconds
        """
        if airflow_percent < NFPA85TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT:
            raise ValueError(
                f"Airflow {airflow_percent}% below minimum "
                f"{NFPA85TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT}%"
            )

        # Calculate actual airflow in CFM
        actual_cfm = self.max_airflow_cfm * (airflow_percent / 100.0)

        # Calculate time for 4 volume changes
        # Time = (4 * Volume) / Airflow
        time_for_4_changes = (
            NFPA85TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES
            * self.furnace_volume / actual_cfm * 60  # Convert to seconds
        )

        # Return maximum of calculated time and minimum time
        return max(
            time_for_4_changes,
            NFPA85TimingRequirements.PREPURGE_MIN_TIME_SECONDS
        )

    def transition_to(self, new_state: BMSState, reason: str = "") -> bool:
        """
        Attempt state transition.

        Args:
            new_state: Target state
            reason: Reason for transition

        Returns:
            True if transition successful
        """
        with self._lock:
            # Validate transition
            if new_state not in self.VALID_TRANSITIONS.get(self._state, set()):
                logger.error(
                    f"Invalid BMS transition: {self._state.name} -> {new_state.name}"
                )
                return False

            # Record transition
            old_state = self._state
            self._previous_state = old_state
            self._state = new_state
            self._state_entry_time = datetime.utcnow()

            self._state_history.append((
                datetime.utcnow(),
                old_state,
                new_state
            ))

            # Log transition
            logger.info(
                f"BMS State: {old_state.name} -> {new_state.name} "
                f"({reason or 'No reason provided'})"
            )

            # Create safety event
            event = SafetyEvent(
                event_id=f"STATE-{uuid.uuid4().hex[:8].upper()}",
                event_type="BMS_STATE_CHANGE",
                timestamp=datetime.utcnow(),
                equipment_id=self.equipment_id,
                description=f"BMS state change: {old_state.name} -> {new_state.name}. {reason}",
                severity="INFO",
                bms_state=new_state
            )
            self._event_history.append(event)

            # Callbacks
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
            if self.on_safety_event:
                self.on_safety_event(event)

            return True

    def request_start(self) -> Tuple[bool, str]:
        """
        Request burner startup sequence.

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.IDLE:
                return False, f"Cannot start from state {self._state.name}"

        if self.transition_to(BMSState.PERMISSIVE_CHECK, "Startup requested"):
            return True, "Starting permissive check"
        return False, "Transition failed"

    def permissives_satisfied(self, all_ok: bool) -> Tuple[bool, str]:
        """
        Report permissive check results.

        Args:
            all_ok: True if all permissives are satisfied

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.PERMISSIVE_CHECK:
                return False, f"Not in PERMISSIVE_CHECK state"

            self._all_permissives_ok = all_ok

            if not all_ok:
                self.transition_to(BMSState.IDLE, "Permissives not satisfied")
                return False, "Permissives not satisfied"

        # Start prepurge
        self._purge_start_time = datetime.utcnow()
        if self.transition_to(BMSState.PREPURGE, "Permissives OK, starting prepurge"):
            return True, "Starting prepurge"
        return False, "Transition to prepurge failed"

    def prepurge_complete(self, volume_changes: float, elapsed_seconds: float) -> Tuple[bool, str]:
        """
        Report prepurge completion.

        Args:
            volume_changes: Number of volume changes completed
            elapsed_seconds: Time elapsed during prepurge

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.PREPURGE:
                return False, f"Not in PREPURGE state"

            # Validate prepurge requirements
            if volume_changes < NFPA85TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES:
                return False, (
                    f"Insufficient volume changes: {volume_changes:.1f} "
                    f"(min {NFPA85TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES})"
                )

            if elapsed_seconds < NFPA85TimingRequirements.PREPURGE_MIN_TIME_SECONDS:
                return False, (
                    f"Prepurge too short: {elapsed_seconds:.1f}s "
                    f"(min {NFPA85TimingRequirements.PREPURGE_MIN_TIME_SECONDS}s)"
                )

        # Start pilot trial
        self._pilot_trial_start = datetime.utcnow()
        if self.transition_to(BMSState.PILOT_TRIAL, "Prepurge complete"):
            return True, "Starting pilot trial"
        return False, "Transition to pilot trial failed"

    def pilot_proven(self, flame_signal: float) -> Tuple[bool, str]:
        """
        Report pilot flame proven.

        Args:
            flame_signal: Flame scanner signal strength (0-100%)

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.PILOT_TRIAL:
                return False, f"Not in PILOT_TRIAL state"

            # Check trial timing
            if self._pilot_trial_start:
                elapsed = (datetime.utcnow() - self._pilot_trial_start).total_seconds()
                if elapsed > NFPA85TimingRequirements.PILOT_TRIAL_MAX_SECONDS:
                    self._initiate_safety_shutdown("Pilot trial timeout")
                    return False, "Pilot trial timeout - safety shutdown"

            if flame_signal < 50.0:  # Minimum signal threshold
                return False, f"Pilot flame signal too low: {flame_signal}%"

            self._pilot_proven = True

        # Start main flame trial
        self._main_trial_start = datetime.utcnow()
        if self.transition_to(BMSState.MAIN_TRIAL, f"Pilot proven at {flame_signal}%"):
            return True, "Starting main flame trial"
        return False, "Transition to main trial failed"

    def main_flame_proven(self, flame_signal: float) -> Tuple[bool, str]:
        """
        Report main flame proven.

        Args:
            flame_signal: Flame scanner signal strength (0-100%)

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.MAIN_TRIAL:
                return False, f"Not in MAIN_TRIAL state"

            # Check trial timing
            if self._main_trial_start:
                elapsed = (datetime.utcnow() - self._main_trial_start).total_seconds()
                if elapsed > NFPA85TimingRequirements.MAIN_FLAME_TRIAL_MAX_SECONDS:
                    self._initiate_safety_shutdown("Main flame trial timeout")
                    return False, "Main flame trial timeout - safety shutdown"

            if flame_signal < 50.0:
                return False, f"Main flame signal too low: {flame_signal}%"

            self._flame_proven = True

        # Enter RUN state
        if self.transition_to(BMSState.RUN, f"Main flame proven at {flame_signal}%"):
            return True, "Burner running"
        return False, "Transition to RUN failed"

    def process_flame_status(self, flame_status: FlameStatus, flame_signal: float) -> Tuple[bool, Optional[SafetyEvent]]:
        """
        Process flame status during operation.

        Per NFPA 85 Section 8.5.2.2, flame failure must be detected
        and fuel shut off within 4 seconds.

        Args:
            flame_status: Current flame status
            flame_signal: Flame scanner signal strength

        Returns:
            Tuple of (is_ok, safety_event if trip occurred)
        """
        with self._lock:
            if self._state != BMSState.RUN:
                return True, None

            if flame_status == FlameStatus.FLAME_ON and flame_signal >= 30.0:
                # Flame OK
                self._flame_loss_time = None
                return True, None

            # Flame lost or unstable
            now = datetime.utcnow()

            if self._flame_loss_time is None:
                self._flame_loss_time = now
                logger.warning(
                    f"Flame loss detected: status={flame_status.name}, "
                    f"signal={flame_signal}%"
                )

            # Check if response time exceeded
            elapsed = (now - self._flame_loss_time).total_seconds()

            if elapsed >= NFPA85TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS:
                # IMMEDIATE SAFETY SHUTDOWN
                event = SafetyEvent(
                    event_id=f"FLAME-{uuid.uuid4().hex[:8].upper()}",
                    event_type="FLAME_FAILURE_SHUTDOWN",
                    timestamp=now,
                    equipment_id=self.equipment_id,
                    description=(
                        f"Flame failure shutdown: {flame_status.name}, "
                        f"signal={flame_signal}%, "
                        f"response_time={elapsed:.2f}s"
                    ),
                    severity="TRIP",
                    response_time_ms=elapsed * 1000
                )

                self._event_history.append(event)

                logger.critical(
                    f"FLAME FAILURE SHUTDOWN: {self.equipment_id} - "
                    f"Flame loss for {elapsed:.2f}s exceeds 4s limit"
                )

                # Initiate safety shutdown
                self._initiate_safety_shutdown(
                    f"Flame failure - {elapsed:.2f}s response time"
                )

                if self.on_safety_event:
                    self.on_safety_event(event)

                return False, event

            return True, None

    def request_normal_shutdown(self) -> Tuple[bool, str]:
        """
        Request normal burner shutdown.

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.RUN:
                return False, f"Cannot shutdown from state {self._state.name}"

        if self.transition_to(BMSState.POSTPURGE, "Normal shutdown requested"):
            return True, "Starting postpurge"
        return False, "Transition failed"

    def initiate_safety_shutdown(self, reason: str) -> SafetyEvent:
        """
        Initiate safety shutdown (public interface).

        Args:
            reason: Reason for shutdown

        Returns:
            Safety event record
        """
        with self._lock:
            return self._initiate_safety_shutdown(reason)

    def _initiate_safety_shutdown(self, reason: str) -> SafetyEvent:
        """
        Internal safety shutdown (must hold lock).

        Args:
            reason: Reason for shutdown

        Returns:
            Safety event record
        """
        event = SafetyEvent(
            event_id=f"SSHUT-{uuid.uuid4().hex[:8].upper()}",
            event_type="SAFETY_SHUTDOWN",
            timestamp=datetime.utcnow(),
            equipment_id=self.equipment_id,
            description=f"Safety shutdown initiated: {reason}",
            severity="TRIP",
            bms_state=self._state
        )

        self._event_history.append(event)

        logger.critical(
            f"SAFETY SHUTDOWN: {self.equipment_id} - {reason}"
        )

        # Force state transition
        self._previous_state = self._state
        self._state = BMSState.SAFETY_SHUTDOWN
        self._state_entry_time = datetime.utcnow()

        self._flame_proven = False
        self._pilot_proven = False

        if self.on_state_change:
            self.on_state_change(self._previous_state, BMSState.SAFETY_SHUTDOWN)
        if self.on_safety_event:
            self.on_safety_event(event)

        return event

    def initiate_lockout(self, reason: str) -> SafetyEvent:
        """
        Initiate lockout requiring manual reset.

        Args:
            reason: Reason for lockout

        Returns:
            Safety event record
        """
        with self._lock:
            event = SafetyEvent(
                event_id=f"LOCK-{uuid.uuid4().hex[:8].upper()}",
                event_type="BMS_LOCKOUT",
                timestamp=datetime.utcnow(),
                equipment_id=self.equipment_id,
                description=f"BMS lockout: {reason}",
                severity="ALARM",
                bms_state=self._state
            )

            self._event_history.append(event)

            logger.critical(
                f"BMS LOCKOUT: {self.equipment_id} - {reason} - "
                "Manual reset required"
            )

            self._previous_state = self._state
            self._state = BMSState.LOCKOUT
            self._state_entry_time = datetime.utcnow()

            if self.on_state_change:
                self.on_state_change(self._previous_state, BMSState.LOCKOUT)
            if self.on_safety_event:
                self.on_safety_event(event)

            return event

    def manual_reset(self, authorized_by: str, all_conditions_clear: bool) -> Tuple[bool, str]:
        """
        Perform manual reset from lockout.

        Args:
            authorized_by: ID of person performing reset
            all_conditions_clear: Confirm all trip conditions are cleared

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.LOCKOUT:
                return False, f"Not in LOCKOUT state"

            if not all_conditions_clear:
                return False, "All trip conditions must be cleared before reset"

        logger.info(f"BMS manual reset by {authorized_by}")

        if self.transition_to(BMSState.IDLE, f"Manual reset by {authorized_by}"):
            return True, "Reset successful"
        return False, "Reset failed"

    def postpurge_complete(self) -> Tuple[bool, str]:
        """
        Report postpurge completion.

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._state != BMSState.POSTPURGE:
                return False, f"Not in POSTPURGE state"

        if self.transition_to(BMSState.IDLE, "Postpurge complete"):
            return True, "Shutdown complete"
        return False, "Transition failed"

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive BMS status."""
        with self._lock:
            return {
                "equipment_id": self.equipment_id,
                "state": self._state.name,
                "previous_state": self._previous_state.name,
                "time_in_state_seconds": self.time_in_state,
                "flame_proven": self._flame_proven,
                "pilot_proven": self._pilot_proven,
                "all_permissives_ok": self._all_permissives_ok,
                "is_safe": self.is_safe,
                "is_running": self.is_running,
                "event_count": len(self._event_history),
                "last_event": (
                    self._event_history[-1].description
                    if self._event_history else None
                )
            }


# =============================================================================
# Main Safeguard Manager
# =============================================================================

class NFPA85SafeguardManager:
    """
    Main NFPA 85 Combustion Safeguard Manager for GL-018 UNIFIEDCOMBUSTION.

    This class provides the top-level interface for managing all NFPA 85
    combustion safety requirements including:
    - Safety interlock monitoring and response
    - BMS state machine coordination
    - Timing validation (prepurge, trial, flame failure)
    - Bypass management with time limits
    - Safety event logging and provenance

    Example:
        >>> manager = NFPA85SafeguardManager(
        ...     equipment_id="BLR-001",
        ...     furnace_volume_cubic_feet=500.0,
        ...     max_airflow_cfm=2000.0
        ... )
        >>> manager.add_interlock(
        ...     InterlockType.LOW_FUEL_PRESSURE,
        ...     setpoint=5.0, unit="psig", is_high_trip=False
        ... )
        >>> status = manager.get_safety_status()
    """

    # Maximum bypass duration per NFPA 85 guidance
    MAX_BYPASS_HOURS = 8.0

    # Maximum simultaneous bypasses
    MAX_SIMULTANEOUS_BYPASSES = 1

    def __init__(
        self,
        equipment_id: str,
        furnace_volume_cubic_feet: float = 500.0,
        max_airflow_cfm: float = 2000.0,
        on_safety_event: Optional[Callable[[SafetyEvent], None]] = None
    ):
        """
        Initialize NFPA 85 Safeguard Manager.

        Args:
            equipment_id: Equipment identifier
            furnace_volume_cubic_feet: Furnace volume for purge calculations
            max_airflow_cfm: Maximum airflow rate in CFM
            on_safety_event: Callback for safety events
        """
        self.equipment_id = equipment_id
        self.on_safety_event = on_safety_event

        # Initialize BMS state machine
        self.bms = BMSStateMachine(
            equipment_id=equipment_id,
            furnace_volume_cubic_feet=furnace_volume_cubic_feet,
            max_airflow_cfm=max_airflow_cfm,
            on_state_change=self._on_bms_state_change,
            on_safety_event=self._on_bms_event
        )

        # Initialize interlocks
        self._interlocks: Dict[InterlockType, CombustionSafetyInterlock] = {}
        self._lock = Lock()

        # Bypass tracking
        self._active_bypasses: List[BypassRecord] = []

        # Event history
        self._safety_events: List[SafetyEvent] = []

        # Initialize standard interlocks per NFPA 85 Section 8.6.3
        self._initialize_standard_interlocks()

        logger.info(
            f"NFPA85SafeguardManager initialized for {equipment_id}"
        )

    def _initialize_standard_interlocks(self) -> None:
        """Initialize standard NFPA 85 required interlocks."""
        standard_interlocks = [
            (InterlockType.LOW_FUEL_PRESSURE, 5.0, "psig", False, "8.6.3.1"),
            (InterlockType.HIGH_FUEL_PRESSURE, 25.0, "psig", True, "8.6.3.2"),
            (InterlockType.LOW_COMBUSTION_AIR, 20.0, "%", False, "8.6.3.3"),
            (InterlockType.FLAME_FAILURE, 30.0, "%", False, "8.6.3.4"),
            (InterlockType.HIGH_STEAM_PRESSURE, 175.0, "psig", True, "8.6.3.5"),
            (InterlockType.LOW_WATER_LEVEL, 25.0, "%", False, "8.6.3.6"),
        ]

        for itype, setpoint, unit, is_high, clause in standard_interlocks:
            self.add_interlock(
                interlock_type=itype,
                setpoint=setpoint,
                unit=unit,
                is_high_trip=is_high,
                nfpa_clause=clause
            )

    def add_interlock(
        self,
        interlock_type: InterlockType,
        setpoint: float,
        unit: str,
        is_high_trip: bool = True,
        deadband: float = 0.0,
        response_time_limit: float = 3.0,
        nfpa_clause: str = "8.6.3"
    ) -> CombustionSafetyInterlock:
        """
        Add a safety interlock.

        Args:
            interlock_type: Type of safety interlock
            setpoint: Trip setpoint
            unit: Engineering unit
            is_high_trip: True for high trip, False for low
            deadband: Reset deadband
            response_time_limit: Maximum response time
            nfpa_clause: NFPA 85 clause reference

        Returns:
            Created interlock instance
        """
        interlock = CombustionSafetyInterlock(
            interlock_type=interlock_type,
            setpoint=setpoint,
            unit=unit,
            is_high_trip=is_high_trip,
            deadband=deadband,
            response_time_limit=response_time_limit,
            nfpa_clause=nfpa_clause
        )

        with self._lock:
            self._interlocks[interlock_type] = interlock

        return interlock

    def process_sensor_reading(
        self,
        interlock_type: InterlockType,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Optional[SafetyEvent]]:
        """
        Process a sensor reading for an interlock.

        Args:
            interlock_type: Type of interlock
            value: Sensor value
            timestamp: Reading timestamp (defaults to now)

        Returns:
            Tuple of (trip_occurred, safety_event if trip)
        """
        timestamp = timestamp or datetime.utcnow()

        with self._lock:
            if interlock_type not in self._interlocks:
                logger.error(f"Unknown interlock type: {interlock_type}")
                return False, None

            interlock = self._interlocks[interlock_type]
            trip_required, event = interlock.process_reading(value, timestamp)

            if trip_required and event:
                event.equipment_id = self.equipment_id
                self._safety_events.append(event)

                # Initiate safety shutdown
                self.bms.initiate_safety_shutdown(
                    f"Interlock trip: {interlock_type.value}"
                )

                if self.on_safety_event:
                    self.on_safety_event(event)

            return trip_required, event

    def process_flame_reading(
        self,
        flame_readings: List[FlameReading]
    ) -> Tuple[FlameStatus, float, Optional[SafetyEvent]]:
        """
        Process flame scanner readings with voting.

        Per NFPA 85, flame detection should use redundant scanners.
        This method implements 1oo2 voting (any scanner detects flame = flame on).

        Args:
            flame_readings: List of flame scanner readings

        Returns:
            Tuple of (status, voted_signal, event if trip)
        """
        if not flame_readings:
            return FlameStatus.UNKNOWN, 0.0, None

        # Filter valid readings
        valid_readings = [r for r in flame_readings if r.is_valid]

        if not valid_readings:
            return FlameStatus.SCANNER_FAULT, 0.0, None

        # 1oo2 voting: use maximum signal (any scanner seeing flame = flame on)
        max_signal = max(r.signal_strength for r in valid_readings)

        # Determine flame status
        if max_signal >= 70.0:
            status = FlameStatus.FLAME_ON
        elif max_signal >= 30.0:
            status = FlameStatus.FLAME_UNSTABLE
        else:
            status = FlameStatus.FLAME_OFF

        # Process through BMS
        is_ok, event = self.bms.process_flame_status(status, max_signal)

        return status, max_signal, event

    def authorize_bypass(
        self,
        interlock_type: InterlockType,
        authorized_by: str,
        authorization_level: str,
        reason: str,
        duration_hours: float = 8.0
    ) -> Tuple[bool, str, Optional[BypassRecord]]:
        """
        Authorize an interlock bypass.

        Per NFPA 85 Section 8.6.5, bypasses must be:
        - Time limited
        - Authorized by qualified personnel
        - Logged and tracked

        Args:
            interlock_type: Interlock to bypass
            authorized_by: Person authorizing bypass
            authorization_level: Authorization level (OPERATOR, SUPERVISOR, SAFETY_ENGINEER)
            reason: Reason for bypass
            duration_hours: Bypass duration (max 8 hours)

        Returns:
            Tuple of (success, message, bypass_record if success)
        """
        with self._lock:
            # Check bypass limit
            active_count = sum(1 for b in self._active_bypasses if b.is_active)
            if active_count >= self.MAX_SIMULTANEOUS_BYPASSES:
                return False, f"Maximum bypasses ({self.MAX_SIMULTANEOUS_BYPASSES}) already active", None

            # Check duration
            if duration_hours > self.MAX_BYPASS_HOURS:
                duration_hours = self.MAX_BYPASS_HOURS
                logger.warning(
                    f"Bypass duration limited to {self.MAX_BYPASS_HOURS} hours"
                )

            # Check interlock exists
            if interlock_type not in self._interlocks:
                return False, f"Unknown interlock: {interlock_type.value}", None

            # Create bypass record
            bypass = BypassRecord(
                bypass_id=f"BYP-{uuid.uuid4().hex[:8].upper()}",
                interlock_type=interlock_type,
                authorized_by=authorized_by,
                authorization_level=authorization_level,
                start_time=datetime.utcnow(),
                max_duration_hours=duration_hours,
                reason=reason,
                equipment_id=self.equipment_id
            )

            # Apply bypass
            interlock = self._interlocks[interlock_type]
            if not interlock.set_bypass(bypass):
                return False, "Cannot bypass tripped interlock", None

            self._active_bypasses.append(bypass)

            # Log safety event
            event = SafetyEvent(
                event_id=bypass.bypass_id,
                event_type="INTERLOCK_BYPASS",
                timestamp=datetime.utcnow(),
                equipment_id=self.equipment_id,
                description=(
                    f"Bypass authorized: {interlock_type.value} "
                    f"by {authorized_by} ({authorization_level}) "
                    f"for {duration_hours}h - Reason: {reason}"
                ),
                severity="WARNING",
                interlock_type=interlock_type
            )
            self._safety_events.append(event)

            if self.on_safety_event:
                self.on_safety_event(event)

            return True, f"Bypass active until {bypass.expires_at}", bypass

    def clear_bypass(self, interlock_type: InterlockType) -> bool:
        """
        Clear an active bypass.

        Args:
            interlock_type: Interlock to clear bypass for

        Returns:
            True if bypass cleared
        """
        with self._lock:
            if interlock_type not in self._interlocks:
                return False

            self._interlocks[interlock_type].clear_bypass()

            # Update bypass records
            for bypass in self._active_bypasses:
                if bypass.interlock_type == interlock_type and bypass.is_active:
                    bypass.is_active = False

            return True

    def check_bypass_expirations(self) -> List[InterlockType]:
        """
        Check and clear expired bypasses.

        Returns:
            List of interlocks with cleared bypasses
        """
        cleared = []

        with self._lock:
            for bypass in self._active_bypasses:
                if bypass.is_active and bypass.is_expired:
                    bypass.is_active = False
                    if bypass.interlock_type in self._interlocks:
                        self._interlocks[bypass.interlock_type].clear_bypass()
                        cleared.append(bypass.interlock_type)

                        logger.warning(
                            f"Bypass expired: {bypass.interlock_type.value}"
                        )

        return cleared

    def validate_prepurge_timing(
        self,
        airflow_percent: float,
        elapsed_seconds: float,
        volume_changes: float
    ) -> Tuple[bool, str]:
        """
        Validate prepurge timing requirements.

        Args:
            airflow_percent: Current airflow percentage
            elapsed_seconds: Elapsed purge time
            volume_changes: Completed volume changes

        Returns:
            Tuple of (is_valid, message)
        """
        # Check minimum airflow
        if airflow_percent < NFPA85TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT:
            return False, (
                f"Airflow {airflow_percent:.1f}% below minimum "
                f"{NFPA85TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT}%"
            )

        # Check volume changes
        if volume_changes < NFPA85TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES:
            return False, (
                f"Volume changes {volume_changes:.1f} below minimum "
                f"{NFPA85TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES}"
            )

        # Check minimum time
        if elapsed_seconds < NFPA85TimingRequirements.PREPURGE_MIN_TIME_SECONDS:
            return False, (
                f"Purge time {elapsed_seconds:.1f}s below minimum "
                f"{NFPA85TimingRequirements.PREPURGE_MIN_TIME_SECONDS}s"
            )

        return True, "Prepurge requirements satisfied"

    def validate_flame_failure_response(
        self,
        detection_time: datetime,
        response_time: datetime
    ) -> Tuple[bool, float, str]:
        """
        Validate flame failure response time per NFPA 85.

        Args:
            detection_time: Time flame loss detected
            response_time: Time fuel shutoff completed

        Returns:
            Tuple of (is_compliant, response_seconds, message)
        """
        response_seconds = (response_time - detection_time).total_seconds()

        is_compliant = (
            response_seconds <= NFPA85TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS
        )

        if is_compliant:
            message = (
                f"Flame failure response {response_seconds:.2f}s "
                f"within {NFPA85TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS}s limit"
            )
        else:
            message = (
                f"VIOLATION: Flame failure response {response_seconds:.2f}s "
                f"exceeds {NFPA85TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS}s limit"
            )
            logger.critical(message)

        return is_compliant, response_seconds, message

    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get comprehensive safety status.

        Returns:
            Dictionary with complete safety status
        """
        with self._lock:
            # Check bypass expirations
            self.check_bypass_expirations()

            interlock_status = {}
            all_healthy = True
            tripped_count = 0
            bypassed_count = 0

            for itype, interlock in self._interlocks.items():
                reading = interlock.get_reading()
                interlock_status[itype.value] = {
                    "value": reading.value,
                    "unit": reading.unit,
                    "setpoint": reading.setpoint,
                    "status": reading.status.name,
                    "is_tripped": interlock.is_tripped,
                    "is_bypassed": interlock.is_bypassed
                }

                if interlock.is_tripped:
                    all_healthy = False
                    tripped_count += 1
                if interlock.is_bypassed:
                    bypassed_count += 1

            return {
                "equipment_id": self.equipment_id,
                "timestamp": datetime.utcnow().isoformat(),
                "bms_status": self.bms.get_status(),
                "interlocks": interlock_status,
                "summary": {
                    "all_interlocks_healthy": all_healthy,
                    "tripped_count": tripped_count,
                    "bypassed_count": bypassed_count,
                    "total_interlocks": len(self._interlocks),
                    "is_safe_to_operate": all_healthy and self.bms.is_safe
                },
                "active_bypasses": [
                    {
                        "interlock": b.interlock_type.value,
                        "authorized_by": b.authorized_by,
                        "expires": b.expires_at.isoformat(),
                        "reason": b.reason
                    }
                    for b in self._active_bypasses if b.is_active
                ],
                "recent_events": [
                    {
                        "id": e.event_id,
                        "type": e.event_type,
                        "timestamp": e.timestamp.isoformat(),
                        "description": e.description,
                        "severity": e.severity
                    }
                    for e in self._safety_events[-10:]
                ],
                "timing_requirements": {
                    "flame_failure_response_max_sec": NFPA85TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS,
                    "prepurge_min_volume_changes": NFPA85TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES,
                    "prepurge_min_airflow_pct": NFPA85TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT,
                    "pilot_trial_max_sec": NFPA85TimingRequirements.PILOT_TRIAL_MAX_SECONDS,
                    "main_trial_max_sec": NFPA85TimingRequirements.MAIN_FLAME_TRIAL_MAX_SECONDS
                }
            }

    def _on_bms_state_change(self, old_state: BMSState, new_state: BMSState) -> None:
        """Handle BMS state change callback."""
        logger.info(
            f"BMS state change: {old_state.name} -> {new_state.name}"
        )

    def _on_bms_event(self, event: SafetyEvent) -> None:
        """Handle BMS safety event callback."""
        with self._lock:
            self._safety_events.append(event)

        if self.on_safety_event:
            self.on_safety_event(event)

    def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate NFPA 85 compliance report.

        Returns:
            Compliance report dictionary
        """
        return {
            "report_id": f"NFPA85-{uuid.uuid4().hex[:8].upper()}",
            "generated_at": datetime.utcnow().isoformat(),
            "equipment_id": self.equipment_id,
            "standard": "NFPA 85-2019",
            "requirements": {
                "section_8_4_2_prepurge": {
                    "requirement": "Minimum 4 furnace volume changes at >=25% airflow",
                    "implemented": True,
                    "validation_method": "calculate_prepurge_time()"
                },
                "section_8_5_2_2_flame_failure": {
                    "requirement": "Flame failure response within 4 seconds",
                    "implemented": True,
                    "validation_method": "process_flame_status()"
                },
                "section_8_5_4_ignition_trials": {
                    "requirement": "Pilot trial max 10s, Main trial max 10s",
                    "implemented": True,
                    "validation_method": "BMSStateMachine.pilot_proven(), main_flame_proven()"
                },
                "section_8_6_3_interlocks": {
                    "requirement": "Required safety interlocks",
                    "implemented": True,
                    "interlocks_configured": [i.value for i in self._interlocks.keys()]
                },
                "section_8_6_5_bypasses": {
                    "requirement": "Bypass management with time limits",
                    "implemented": True,
                    "max_duration_hours": self.MAX_BYPASS_HOURS,
                    "max_simultaneous": self.MAX_SIMULTANEOUS_BYPASSES
                }
            },
            "audit_trail": {
                "total_events": len(self._safety_events),
                "trips": sum(1 for e in self._safety_events if e.severity == "TRIP"),
                "bypasses": sum(1 for e in self._safety_events if e.event_type == "INTERLOCK_BYPASS")
            }
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    'NFPA85SafeguardManager',
    'BMSStateMachine',
    'CombustionSafetyInterlock',

    # Enumerations
    'BMSState',
    'InterlockType',
    'InterlockStatus',
    'FlameStatus',
    'ShutdownType',

    # Data classes
    'FlameReading',
    'InterlockReading',
    'PurgeStatus',
    'SafetyEvent',
    'BypassRecord',

    # Constants
    'NFPA85TimingRequirements',
]
