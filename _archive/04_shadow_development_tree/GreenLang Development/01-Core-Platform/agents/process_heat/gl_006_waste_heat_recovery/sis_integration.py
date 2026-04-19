# -*- coding: utf-8 -*-
"""
GL-006 WasteHeatRecovery Agent - SIS Integration Module
========================================================

Complete IEC 61511 SIL 2 Safety Instrumented System integration for
Waste Heat Recovery systems. Implements safety interlocks for heat exchangers,
process streams, and corrosion protection.

Key Features:
    - IEC 61511 SIL 2 compliant interlock definitions
    - 2oo3, 1oo2, 2oo2 voting logic implementations
    - Temperature protection for heat exchangers
    - Acid dew point corrosion protection
    - Pressure relief interlocks
    - Flow loss detection
    - Thermal stress protection
    - Comprehensive audit trail

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - IEC 61508-6:2010 PFD Calculations
    - ASME B31.3 Process Piping
    - API 660 Shell-and-Tube Heat Exchangers

Example:
    >>> from greenlang.agents.process_heat.gl_006_waste_heat_recovery.sis_integration import (
    ...     WHRSISManager, WHRInterlock, VotingType, SensorConfig
    ... )
    >>>
    >>> manager = WHRSISManager(sil_level=2)
    >>> interlock = create_acid_dew_point_interlock(
    ...     name="ADP_PROTECTION_HX-001",
    ...     tag_prefix="TT-001",
    ...     acid_dew_point_f=250.0,
    ...     margin_f=25.0
    ... )
    >>> manager.register_interlock(interlock)

Author: GreenLang Safety Engineering Team
Version: 1.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class VotingType(str, Enum):
    """Voting architecture types per IEC 61511."""
    ONE_OO_ONE = "1oo1"      # Single channel
    ONE_OO_TWO = "1oo2"      # Any one of two triggers
    TWO_OO_TWO = "2oo2"      # Both must agree
    TWO_OO_THREE = "2oo3"    # Two of three (TMR)
    ONE_OO_THREE = "1oo3"    # Any one of three
    THREE_OO_THREE = "3oo3"  # All three must agree


class SafeStateAction(str, Enum):
    """Safe state actions for waste heat recovery interlocks."""
    CLOSE_HOT_SIDE_VALVE = "close_hot_side_valve"
    CLOSE_COLD_SIDE_VALVE = "close_cold_side_valve"
    OPEN_BYPASS_VALVE = "open_bypass_valve"
    ISOLATE_HEAT_EXCHANGER = "isolate_heat_exchanger"
    ACTIVATE_QUENCH = "activate_quench"
    EMERGENCY_COOLDOWN = "emergency_cooldown"
    REDUCE_HOT_FLOW = "reduce_hot_flow"
    OPEN_RELIEF_VALVE = "open_relief_valve"
    TRIP_PROCESS = "trip_process"
    ALARM_ONLY = "alarm_only"
    CUSTOM = "custom"


class SensorType(str, Enum):
    """Sensor types for SIS inputs."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    FLOW = "flow"
    LEVEL = "level"
    PH = "ph"
    CONDUCTIVITY = "conductivity"
    CORROSION_RATE = "corrosion_rate"


class SensorStatus(str, Enum):
    """Health status of a sensor channel."""
    NORMAL = "normal"
    FAULT = "fault"
    OUT_OF_RANGE = "out_of_range"
    BYPASSED = "bypassed"
    CALIBRATING = "calibrating"
    UNKNOWN = "unknown"


class InterlockStatus(str, Enum):
    """Status of a safety interlock."""
    ARMED = "armed"
    TRIPPED = "tripped"
    BYPASSED = "bypassed"
    TESTING = "testing"
    FAULT = "fault"
    RESET_REQUIRED = "reset_required"


class BypassReason(str, Enum):
    """Authorized reasons for interlock bypass."""
    MAINTENANCE = "maintenance"
    PROOF_TEST = "proof_test"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    CALIBRATION = "calibration"
    EMERGENCY_OPERATION = "emergency_operation"


class WHRInterlockType(str, Enum):
    """Types of waste heat recovery interlocks."""
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    ACID_DEW_POINT = "acid_dew_point"
    HIGH_PRESSURE = "high_pressure"
    LOW_FLOW = "low_flow"
    HIGH_DIFFERENTIAL_PRESSURE = "high_differential_pressure"
    THERMAL_STRESS = "thermal_stress"
    TUBE_RUPTURE = "tube_rupture"


# =============================================================================
# DATA MODELS
# =============================================================================

class SensorConfig(BaseModel):
    """Configuration for a single SIS sensor channel."""
    sensor_id: str = Field(..., description="Unique sensor identifier")
    channel: str = Field(..., description="Channel identifier (A, B, C)")
    sensor_type: SensorType = Field(..., description="Type of sensor")
    tag_name: str = Field(..., description="Process tag name")
    engineering_units: str = Field(..., description="Engineering units")
    range_low: float = Field(..., description="Low end of calibrated range")
    range_high: float = Field(..., description="High end of calibrated range")
    accuracy_percent: float = Field(default=0.5, ge=0, le=10)
    response_time_ms: float = Field(default=100.0, ge=0)
    fail_direction: str = Field(default="high", description="Fail direction")
    lambda_du: float = Field(default=1e-6, ge=0, description="Dangerous undetected failure rate")
    diagnostics_enabled: bool = Field(default=True)
    status: SensorStatus = Field(default=SensorStatus.NORMAL)
    last_calibration: Optional[datetime] = Field(default=None)
    calibration_interval_days: int = Field(default=365, ge=1)

    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Validate channel identifier."""
        valid_channels = {'A', 'B', 'C', 'D', '1', '2', '3', '4'}
        if v.upper() not in valid_channels:
            raise ValueError(f"Invalid channel: {v}")
        return v.upper()


class SensorReading(BaseModel):
    """Real-time reading from a sensor channel."""
    sensor_id: str = Field(..., description="Sensor identifier")
    channel: str = Field(..., description="Channel identifier")
    value: float = Field(..., description="Current value")
    quality: float = Field(default=1.0, ge=0, le=1)
    status: SensorStatus = Field(default=SensorStatus.NORMAL)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VotingResult(BaseModel):
    """Result of voting logic evaluation."""
    voting_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    voting_type: VotingType = Field(...)
    trip_decision: bool = Field(...)
    channels_voting_trip: int = Field(...)
    channels_total: int = Field(...)
    channels_required: int = Field(...)
    channels_healthy: int = Field(...)
    channels_bypassed: int = Field(default=0)
    channels_faulted: int = Field(default=0)
    degraded_mode: bool = Field(default=False)
    effective_voting: str = Field(default="")
    channel_details: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_time_ms: float = Field(default=0.0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")


class WHRInterlock(BaseModel):
    """
    Waste Heat Recovery Safety Interlock definition.

    Represents a complete safety interlock including sensors,
    voting logic, setpoints, and safe state actions.
    """
    interlock_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Interlock name")
    interlock_type: WHRInterlockType = Field(..., description="Type of interlock")
    voting_logic: VotingType = Field(..., description="Voting architecture")
    sensors: List[SensorConfig] = Field(..., min_length=1)
    trip_setpoint: float = Field(..., description="Trip setpoint value")
    trip_direction: str = Field(default="high", description="Trip on high or low")
    deadband: float = Field(default=0.0, ge=0)
    safe_state: SafeStateAction = Field(..., description="Safe state action")
    custom_action: Optional[str] = Field(default=None)
    response_time_ms: int = Field(..., ge=0, le=500)
    sil_level: int = Field(default=2, ge=1, le=4)
    pfd_target: float = Field(default=0.005, ge=1e-5, le=0.1)
    proof_test_interval_hours: float = Field(default=8760.0, ge=168)
    status: InterlockStatus = Field(default=InterlockStatus.ARMED)
    bypass_active: bool = Field(default=False)
    bypass_reason: Optional[BypassReason] = Field(default=None)
    bypass_authorized_by: Optional[str] = Field(default=None)
    bypass_expiry: Optional[datetime] = Field(default=None)
    last_trip_time: Optional[datetime] = Field(default=None)
    trip_count: int = Field(default=0, ge=0)
    last_proof_test: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # WHR-specific fields
    heat_exchanger_id: Optional[str] = Field(default=None, description="Associated HX ID")
    process_stream: Optional[str] = Field(default=None, description="hot or cold")
    acid_dew_point_f: Optional[float] = Field(default=None, description="Acid dew point if applicable")

    @field_validator('response_time_ms')
    @classmethod
    def validate_response_time(cls, v: int) -> int:
        """Validate response time meets SIL 2 requirements."""
        if v > 500:
            logger.warning(f"Response time {v}ms exceeds 500ms limit for SIL 2")
        return v


class InterlockTrip(BaseModel):
    """Record of an interlock trip event."""
    trip_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interlock_id: str = Field(...)
    interlock_name: str = Field(...)
    trip_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trip_value: float = Field(...)
    setpoint: float = Field(...)
    voting_result: VotingResult = Field(...)
    safe_state_action: SafeStateAction = Field(...)
    response_time_actual_ms: float = Field(..., ge=0)
    sensor_readings: List[SensorReading] = Field(default_factory=list)
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(default=None)
    reset_time: Optional[datetime] = Field(default=None)
    reset_by: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.interlock_id}|{self.trip_time.isoformat()}|"
                f"{self.trip_value}|{self.setpoint}|{self.response_time_actual_ms}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()


class ProofTestSchedule(BaseModel):
    """Proof test schedule for an interlock."""
    schedule_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    interlock_id: str = Field(...)
    test_interval_hours: float = Field(..., ge=168)
    last_test_date: Optional[datetime] = Field(default=None)
    next_test_date: Optional[datetime] = Field(default=None)
    test_procedure_id: str = Field(default="")
    assigned_technician: Optional[str] = Field(default=None)


# =============================================================================
# VOTING LOGIC ENGINE
# =============================================================================

class WHRVotingEngine:
    """
    Voting logic engine for WHR SIS channels.

    Implements IEC 61511 voting architectures with support for
    degraded mode operation and fault handling.
    """

    VOTING_CONFIG: Dict[VotingType, Tuple[int, int]] = {
        VotingType.ONE_OO_ONE: (1, 1),
        VotingType.ONE_OO_TWO: (1, 2),
        VotingType.TWO_OO_TWO: (2, 2),
        VotingType.TWO_OO_THREE: (2, 3),
        VotingType.ONE_OO_THREE: (1, 3),
        VotingType.THREE_OO_THREE: (3, 3),
    }

    def __init__(self, fail_safe_on_fault: bool = True) -> None:
        """Initialize voting engine."""
        self.fail_safe_on_fault = fail_safe_on_fault
        logger.info("WHRVotingEngine initialized (fail_safe_on_fault=%s)", fail_safe_on_fault)

    def evaluate(
        self,
        voting_type: VotingType,
        readings: List[SensorReading],
        setpoint: float,
        trip_direction: str = "high"
    ) -> VotingResult:
        """Evaluate voting logic with sensor readings."""
        start_time = datetime.now(timezone.utc)

        required, expected = self.VOTING_CONFIG[voting_type]

        channels_trip = 0
        channels_healthy = 0
        channels_bypassed = 0
        channels_faulted = 0
        channel_details = []

        for reading in readings:
            detail = {
                "sensor_id": reading.sensor_id,
                "channel": reading.channel,
                "value": reading.value,
                "status": reading.status.value,
                "quality": reading.quality,
            }

            if reading.status == SensorStatus.BYPASSED:
                channels_bypassed += 1
                detail["vote"] = "bypassed"
            elif reading.status == SensorStatus.FAULT:
                channels_faulted += 1
                if self.fail_safe_on_fault:
                    channels_trip += 1
                    detail["vote"] = "trip_on_fault"
                else:
                    detail["vote"] = "ignored"
            elif reading.status == SensorStatus.UNKNOWN:
                if self.fail_safe_on_fault:
                    channels_trip += 1
                    detail["vote"] = "trip_on_unknown"
                else:
                    detail["vote"] = "ignored"
            else:
                channels_healthy += 1
                if trip_direction == "high":
                    trips = reading.value >= setpoint
                else:
                    trips = reading.value <= setpoint

                if trips:
                    channels_trip += 1
                    detail["vote"] = "trip"
                else:
                    detail["vote"] = "no_trip"

            channel_details.append(detail)

        effective_channels = len(readings) - channels_bypassed
        degraded_mode = (
            channels_bypassed > 0 or
            channels_faulted > 0 or
            len(readings) != expected
        )

        effective_voting, effective_required = self._get_degraded_voting(
            voting_type, effective_channels, channels_bypassed, channels_faulted
        )

        trip_decision = channels_trip >= effective_required

        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        result = VotingResult(
            voting_type=voting_type,
            trip_decision=trip_decision,
            channels_voting_trip=channels_trip,
            channels_total=len(readings),
            channels_required=effective_required,
            channels_healthy=channels_healthy,
            channels_bypassed=channels_bypassed,
            channels_faulted=channels_faulted,
            degraded_mode=degraded_mode,
            effective_voting=effective_voting,
            channel_details=channel_details,
            evaluation_time_ms=eval_time,
        )

        result.provenance_hash = self._calculate_provenance(result)

        logger.debug(
            "Voting %s: trip=%s (%d/%d channels)",
            voting_type.value, trip_decision, channels_trip, effective_required
        )

        return result

    def _get_degraded_voting(
        self,
        voting_type: VotingType,
        healthy_channels: int,
        bypassed: int,
        faulted: int
    ) -> Tuple[str, int]:
        """Determine effective voting in degraded mode."""
        if voting_type == VotingType.TWO_OO_THREE:
            if healthy_channels == 3:
                return "2oo3", 2
            elif healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        elif voting_type == VotingType.ONE_OO_TWO:
            if healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        elif voting_type == VotingType.TWO_OO_TWO:
            if healthy_channels == 2:
                return "2oo2", 2
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        required, _ = self.VOTING_CONFIG[voting_type]
        return voting_type.value, required

    def _calculate_provenance(self, result: VotingResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.voting_type.value}|{result.trip_decision}|"
            f"{result.channels_voting_trip}|{result.channels_total}|"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# WHR SIS MANAGER
# =============================================================================

class WHRSISManager:
    """
    Safety Instrumented System Manager for Waste Heat Recovery.

    Central manager for SIS interlocks in waste heat recovery systems.
    Handles interlock registration, evaluation, bypass management,
    and comprehensive audit logging.

    Example:
        >>> manager = WHRSISManager(sil_level=2)
        >>> manager.register_interlock(acid_dew_point_interlock)
        >>> result = manager.evaluate_interlock("ADP_PROTECTION", readings)
    """

    def __init__(
        self,
        sil_level: int = 2,
        fail_safe_on_fault: bool = True,
        max_bypass_hours: float = 8.0
    ) -> None:
        """Initialize WHR SIS Manager."""
        if sil_level < 1 or sil_level > 4:
            raise ValueError(f"Invalid SIL level: {sil_level}")

        self.sil_level = sil_level
        self.max_bypass_hours = max_bypass_hours

        self._voting_engine = WHRVotingEngine(fail_safe_on_fault=fail_safe_on_fault)
        self._interlocks: Dict[str, WHRInterlock] = {}
        self._trip_history: List[InterlockTrip] = []
        self._proof_test_schedules: Dict[str, ProofTestSchedule] = {}
        self._trip_callbacks: Dict[SafeStateAction, Callable] = {}
        self._audit_log: List[Dict[str, Any]] = []

        logger.info(
            "WHRSISManager initialized (SIL-%d, fail_safe=%s, max_bypass=%sh)",
            sil_level, fail_safe_on_fault, max_bypass_hours
        )

    # =========================================================================
    # INTERLOCK MANAGEMENT
    # =========================================================================

    def register_interlock(self, interlock: WHRInterlock) -> bool:
        """Register a safety interlock."""
        if interlock.interlock_id in self._interlocks:
            logger.warning("Interlock %s already registered", interlock.interlock_id)
            return False

        if interlock.sil_level > self.sil_level:
            logger.warning(
                "Interlock %s requires SIL-%d but manager configured for SIL-%d",
                interlock.name, interlock.sil_level, self.sil_level
            )

        if self.sil_level >= 2 and interlock.response_time_ms > 500:
            raise ValueError(
                f"Interlock {interlock.name} response time {interlock.response_time_ms}ms "
                f"exceeds 500ms limit for SIL-{self.sil_level}"
            )

        self._interlocks[interlock.interlock_id] = interlock

        schedule = ProofTestSchedule(
            interlock_id=interlock.interlock_id,
            test_interval_hours=interlock.proof_test_interval_hours,
            last_test_date=interlock.last_proof_test,
        )
        if schedule.last_test_date:
            schedule.next_test_date = schedule.last_test_date + timedelta(
                hours=interlock.proof_test_interval_hours
            )
        self._proof_test_schedules[interlock.interlock_id] = schedule

        self._log_audit(
            "INTERLOCK_REGISTERED",
            interlock_id=interlock.interlock_id,
            name=interlock.name,
            type=interlock.interlock_type.value,
            voting=interlock.voting_logic.value,
            sil_level=interlock.sil_level,
        )

        logger.info(
            "Interlock registered: %s (%s, %s, SIL-%d)",
            interlock.name, interlock.interlock_id,
            interlock.voting_logic.value, interlock.sil_level
        )
        return True

    def get_interlock(self, interlock_id: str) -> Optional[WHRInterlock]:
        """Get interlock by ID."""
        return self._interlocks.get(interlock_id)

    def get_all_interlocks(self) -> List[WHRInterlock]:
        """Get all registered interlocks."""
        return list(self._interlocks.values())

    def get_interlocks_by_heat_exchanger(self, hx_id: str) -> List[WHRInterlock]:
        """Get all interlocks for a specific heat exchanger."""
        return [
            interlock for interlock in self._interlocks.values()
            if interlock.heat_exchanger_id == hx_id
        ]

    # =========================================================================
    # INTERLOCK EVALUATION
    # =========================================================================

    def evaluate_interlock(
        self,
        interlock_id: str,
        readings: List[SensorReading]
    ) -> VotingResult:
        """Evaluate an interlock with current sensor readings."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            raise ValueError(f"Interlock not found: {interlock_id}")

        if interlock.bypass_active:
            if interlock.bypass_expiry and datetime.now(timezone.utc) > interlock.bypass_expiry:
                self._clear_bypass(interlock_id)
            else:
                logger.warning("Interlock %s is bypassed", interlock.name)
                return VotingResult(
                    voting_type=interlock.voting_logic,
                    trip_decision=False,
                    channels_voting_trip=0,
                    channels_total=len(readings),
                    channels_required=0,
                    channels_healthy=len(readings),
                    effective_voting="BYPASSED",
                )

        result = self._voting_engine.evaluate(
            voting_type=interlock.voting_logic,
            readings=readings,
            setpoint=interlock.trip_setpoint,
            trip_direction=interlock.trip_direction,
        )

        return result

    def evaluate_all_interlocks(
        self,
        readings_by_interlock: Dict[str, List[SensorReading]]
    ) -> Dict[str, VotingResult]:
        """Evaluate all interlocks with their respective readings."""
        results = {}
        for interlock_id, readings in readings_by_interlock.items():
            try:
                results[interlock_id] = self.evaluate_interlock(interlock_id, readings)
            except ValueError as e:
                logger.error("Failed to evaluate %s: %s", interlock_id, e)
        return results

    async def execute_trip(
        self,
        interlock_id: str,
        voting_result: VotingResult,
        readings: List[SensorReading]
    ) -> InterlockTrip:
        """Execute interlock trip action."""
        start_time = datetime.now(timezone.utc)

        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            raise ValueError(f"Interlock not found: {interlock_id}")

        # Get trip value
        healthy_readings = [r for r in readings if r.status == SensorStatus.NORMAL]
        if healthy_readings:
            if interlock.trip_direction == "high":
                trip_value = max(r.value for r in healthy_readings)
            else:
                trip_value = min(r.value for r in healthy_readings)
        else:
            trip_value = interlock.trip_setpoint

        # Execute safe state action
        await self._execute_safe_state_action(interlock.safe_state, interlock_id)

        response_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        interlock.status = InterlockStatus.TRIPPED
        interlock.last_trip_time = datetime.now(timezone.utc)
        interlock.trip_count += 1

        trip_record = InterlockTrip(
            interlock_id=interlock_id,
            interlock_name=interlock.name,
            trip_value=trip_value,
            setpoint=interlock.trip_setpoint,
            voting_result=voting_result,
            safe_state_action=interlock.safe_state,
            response_time_actual_ms=response_time_ms,
            sensor_readings=readings,
        )

        self._trip_history.append(trip_record)

        self._log_audit(
            "INTERLOCK_TRIP",
            interlock_id=interlock_id,
            name=interlock.name,
            type=interlock.interlock_type.value,
            trip_value=trip_value,
            setpoint=interlock.trip_setpoint,
            response_time_ms=response_time_ms,
            action=interlock.safe_state.value,
        )

        if response_time_ms > interlock.response_time_ms:
            logger.error(
                "RESPONSE TIME EXCEEDED: %s actual %.1fms > required %dms",
                interlock.name, response_time_ms, interlock.response_time_ms
            )

        logger.critical(
            "WHR INTERLOCK TRIPPED: %s (value=%.1f, setpoint=%.1f, action=%s)",
            interlock.name, trip_value, interlock.trip_setpoint, interlock.safe_state.value
        )

        return trip_record

    async def _execute_safe_state_action(
        self,
        action: SafeStateAction,
        interlock_id: str
    ) -> None:
        """Execute the safe state action."""
        logger.info("Executing safe state action: %s for %s", action.value, interlock_id)

        if action in self._trip_callbacks:
            try:
                callback = self._trip_callbacks[action]
                if asyncio.iscoroutinefunction(callback):
                    await callback(interlock_id)
                else:
                    callback(interlock_id)
            except Exception as e:
                logger.error("Safe state callback failed: %s", e, exc_info=True)

        await asyncio.sleep(0.01)  # Simulate I/O

    def register_trip_callback(
        self,
        action: SafeStateAction,
        callback: Callable
    ) -> None:
        """Register a callback for a safe state action."""
        self._trip_callbacks[action] = callback
        logger.info("Trip callback registered for action: %s", action.value)

    # =========================================================================
    # BYPASS MANAGEMENT
    # =========================================================================

    def request_bypass(
        self,
        interlock_id: str,
        reason: BypassReason,
        authorized_by: str,
        duration_hours: float = 8.0
    ) -> bool:
        """Request bypass for an interlock."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if duration_hours > self.max_bypass_hours:
            logger.warning(
                "Bypass duration %.1fh exceeds maximum %.1fh",
                duration_hours, self.max_bypass_hours
            )
            duration_hours = self.max_bypass_hours

        interlock.bypass_active = True
        interlock.bypass_reason = reason
        interlock.bypass_authorized_by = authorized_by
        interlock.bypass_expiry = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
        interlock.status = InterlockStatus.BYPASSED

        self._log_audit(
            "INTERLOCK_BYPASS_ACTIVATED",
            interlock_id=interlock_id,
            name=interlock.name,
            reason=reason.value,
            authorized_by=authorized_by,
            duration_hours=duration_hours,
        )

        logger.warning(
            "WHR BYPASS ACTIVATED: %s by %s for %s",
            interlock.name, authorized_by, reason.value
        )
        return True

    def clear_bypass(self, interlock_id: str, cleared_by: str) -> bool:
        """Clear an active bypass."""
        return self._clear_bypass(interlock_id, cleared_by)

    def _clear_bypass(self, interlock_id: str, cleared_by: str = "SYSTEM") -> bool:
        """Internal bypass clear."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock or not interlock.bypass_active:
            return False

        interlock.bypass_active = False
        interlock.bypass_reason = None
        interlock.bypass_authorized_by = None
        interlock.bypass_expiry = None
        interlock.status = InterlockStatus.ARMED

        self._log_audit(
            "INTERLOCK_BYPASS_CLEARED",
            interlock_id=interlock_id,
            name=interlock.name,
            cleared_by=cleared_by,
        )

        logger.info("WHR Bypass cleared: %s by %s", interlock.name, cleared_by)
        return True

    # =========================================================================
    # RESET OPERATIONS
    # =========================================================================

    async def reset_interlock(
        self,
        interlock_id: str,
        reset_by: str,
        force: bool = False
    ) -> bool:
        """Reset a tripped interlock."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if interlock.status != InterlockStatus.TRIPPED:
            logger.warning("Cannot reset %s: not in tripped state", interlock.name)
            return False

        interlock.status = InterlockStatus.ARMED

        if self._trip_history:
            last_trip = next(
                (t for t in reversed(self._trip_history) if t.interlock_id == interlock_id),
                None
            )
            if last_trip:
                last_trip.reset_time = datetime.now(timezone.utc)
                last_trip.reset_by = reset_by

        self._log_audit(
            "INTERLOCK_RESET",
            interlock_id=interlock_id,
            name=interlock.name,
            reset_by=reset_by,
            force=force,
        )

        logger.info("WHR Interlock reset: %s by %s", interlock.name, reset_by)
        return True

    # =========================================================================
    # STATUS AND MONITORING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get overall SIS status."""
        tripped = [i for i in self._interlocks.values() if i.status == InterlockStatus.TRIPPED]
        bypassed = [i for i in self._interlocks.values() if i.bypass_active]
        faulted = [i for i in self._interlocks.values() if i.status == InterlockStatus.FAULT]

        return {
            "sil_level": self.sil_level,
            "total_interlocks": len(self._interlocks),
            "armed": len([i for i in self._interlocks.values() if i.status == InterlockStatus.ARMED]),
            "tripped": len(tripped),
            "tripped_names": [i.name for i in tripped],
            "bypassed": len(bypassed),
            "bypassed_names": [i.name for i in bypassed],
            "faulted": len(faulted),
            "total_trip_count": sum(i.trip_count for i in self._interlocks.values()),
        }

    def get_trip_history(
        self,
        interlock_id: Optional[str] = None,
        limit: int = 100
    ) -> List[InterlockTrip]:
        """Get trip history."""
        history = self._trip_history
        if interlock_id:
            history = [t for t in history if t.interlock_id == interlock_id]
        return list(reversed(history[-limit:]))

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return list(reversed(self._audit_log[-limit:]))

    def _log_audit(self, event_type: str, **kwargs: Any) -> None:
        """Log an audit event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **kwargs
        }
        hash_str = f"{entry['timestamp']}|{event_type}|{str(kwargs)}"
        entry["provenance_hash"] = hashlib.sha256(hash_str.encode()).hexdigest()[:16]
        self._audit_log.append(entry)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_acid_dew_point_interlock(
    name: str,
    tag_prefix: str,
    acid_dew_point_f: float,
    margin_f: float = 25.0,
    heat_exchanger_id: Optional[str] = None,
    response_time_ms: int = 250
) -> WHRInterlock:
    """
    Factory function to create an acid dew point protection interlock.

    Protects heat exchangers from corrosion by ensuring exhaust gas
    temperature stays above the acid dew point plus a safety margin.

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix (e.g., "TT-001")
        acid_dew_point_f: Acid dew point temperature in F
        margin_f: Safety margin above acid dew point
        heat_exchanger_id: Associated heat exchanger ID
        response_time_ms: Required response time

    Returns:
        Configured WHRInterlock
    """
    setpoint = acid_dew_point_f + margin_f

    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.TEMPERATURE,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="degF",
            range_low=100.0,
            range_high=800.0,
            accuracy_percent=0.5,
            response_time_ms=50.0,
            fail_direction="low",  # Fail low to trip on temperature drop
        )
        for ch in ["A", "B", "C"]
    ]

    return WHRInterlock(
        name=name,
        interlock_type=WHRInterlockType.ACID_DEW_POINT,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=setpoint,
        trip_direction="low",
        deadband=5.0,
        safe_state=SafeStateAction.OPEN_BYPASS_VALVE,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.005,
        proof_test_interval_hours=8760.0,
        heat_exchanger_id=heat_exchanger_id,
        process_stream="hot",
        acid_dew_point_f=acid_dew_point_f,
    )


def create_high_temperature_interlock(
    name: str,
    tag_prefix: str,
    setpoint_f: float,
    heat_exchanger_id: Optional[str] = None,
    process_stream: str = "cold",
    response_time_ms: int = 250
) -> WHRInterlock:
    """
    Factory function to create a high temperature protection interlock.

    Protects process equipment from over-temperature conditions.

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix
        setpoint_f: Trip setpoint in degrees F
        heat_exchanger_id: Associated heat exchanger ID
        process_stream: "hot" or "cold"
        response_time_ms: Required response time

    Returns:
        Configured WHRInterlock
    """
    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.TEMPERATURE,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="degF",
            range_low=0.0,
            range_high=1000.0,
            accuracy_percent=0.5,
            response_time_ms=50.0,
            fail_direction="high",
        )
        for ch in ["A", "B", "C"]
    ]

    return WHRInterlock(
        name=name,
        interlock_type=WHRInterlockType.HIGH_TEMPERATURE,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=setpoint_f,
        trip_direction="high",
        deadband=5.0,
        safe_state=SafeStateAction.CLOSE_HOT_SIDE_VALVE,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.005,
        proof_test_interval_hours=8760.0,
        heat_exchanger_id=heat_exchanger_id,
        process_stream=process_stream,
    )


def create_high_pressure_interlock(
    name: str,
    tag_prefix: str,
    setpoint_psig: float,
    heat_exchanger_id: Optional[str] = None,
    process_stream: str = "cold",
    response_time_ms: int = 200
) -> WHRInterlock:
    """
    Factory function to create a high pressure interlock.

    Protects heat exchangers from over-pressure conditions.

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix
        setpoint_psig: Trip setpoint in psig
        heat_exchanger_id: Associated heat exchanger ID
        process_stream: "hot" or "cold"
        response_time_ms: Required response time

    Returns:
        Configured WHRInterlock
    """
    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.PRESSURE,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="psig",
            range_low=0.0,
            range_high=500.0,
            accuracy_percent=0.25,
            response_time_ms=25.0,
            fail_direction="high",
        )
        for ch in ["A", "B", "C"]
    ]

    return WHRInterlock(
        name=name,
        interlock_type=WHRInterlockType.HIGH_PRESSURE,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=setpoint_psig,
        trip_direction="high",
        deadband=2.0,
        safe_state=SafeStateAction.OPEN_RELIEF_VALVE,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.003,
        proof_test_interval_hours=8760.0,
        heat_exchanger_id=heat_exchanger_id,
        process_stream=process_stream,
    )


def create_low_flow_interlock(
    name: str,
    tag_prefix: str,
    setpoint_gpm: float,
    heat_exchanger_id: Optional[str] = None,
    process_stream: str = "cold",
    response_time_ms: int = 300
) -> WHRInterlock:
    """
    Factory function to create a low flow protection interlock.

    Protects heat exchangers from thermal damage due to loss of cooling flow.

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix
        setpoint_gpm: Trip setpoint in GPM (or % of design)
        heat_exchanger_id: Associated heat exchanger ID
        process_stream: "hot" or "cold"
        response_time_ms: Required response time

    Returns:
        Configured WHRInterlock
    """
    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.FLOW,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="gpm",
            range_low=0.0,
            range_high=1000.0,
            accuracy_percent=1.0,
            response_time_ms=100.0,
            fail_direction="low",
        )
        for ch in ["A", "B", "C"]
    ]

    return WHRInterlock(
        name=name,
        interlock_type=WHRInterlockType.LOW_FLOW,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=setpoint_gpm,
        trip_direction="low",
        deadband=5.0,
        safe_state=SafeStateAction.CLOSE_HOT_SIDE_VALVE,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.005,
        proof_test_interval_hours=8760.0,
        heat_exchanger_id=heat_exchanger_id,
        process_stream=process_stream,
    )


def create_tube_rupture_interlock(
    name: str,
    tag_prefix: str,
    max_dp_psig: float,
    heat_exchanger_id: Optional[str] = None,
    response_time_ms: int = 150
) -> WHRInterlock:
    """
    Factory function to create a tube rupture detection interlock.

    Detects sudden pressure differential changes indicating tube failure.

    Args:
        name: Interlock name
        tag_prefix: Differential pressure sensor tag prefix
        max_dp_psig: Maximum differential pressure indicating rupture
        heat_exchanger_id: Associated heat exchanger ID
        response_time_ms: Required response time

    Returns:
        Configured WHRInterlock
    """
    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.DIFFERENTIAL_PRESSURE,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="psid",
            range_low=-50.0,
            range_high=50.0,
            accuracy_percent=0.5,
            response_time_ms=25.0,
            fail_direction="high",
        )
        for ch in ["A", "B", "C"]
    ]

    return WHRInterlock(
        name=name,
        interlock_type=WHRInterlockType.TUBE_RUPTURE,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=max_dp_psig,
        trip_direction="high",
        deadband=1.0,
        safe_state=SafeStateAction.ISOLATE_HEAT_EXCHANGER,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.003,
        proof_test_interval_hours=8760.0,
        heat_exchanger_id=heat_exchanger_id,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    "VotingType",
    "SafeStateAction",
    "SensorType",
    "SensorStatus",
    "InterlockStatus",
    "BypassReason",
    "WHRInterlockType",
    # Data Models
    "SensorConfig",
    "SensorReading",
    "VotingResult",
    "WHRInterlock",
    "InterlockTrip",
    "ProofTestSchedule",
    # Classes
    "WHRVotingEngine",
    "WHRSISManager",
    # Factory Functions
    "create_acid_dew_point_interlock",
    "create_high_temperature_interlock",
    "create_high_pressure_interlock",
    "create_low_flow_interlock",
    "create_tube_rupture_interlock",
]
