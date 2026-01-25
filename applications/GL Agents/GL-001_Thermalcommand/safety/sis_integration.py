"""
GL-001 ThermalCommand Orchestrator - SIS Integration Module

Complete IEC 61511 SIL 2 Safety Instrumented System integration for
the ThermalCommand Orchestrator. Implements 2oo3 voting logic, interlock
definitions, and safety function management.

Key Features:
    - IEC 61511 SIL 2 compliant interlock definitions
    - 2oo3, 1oo2, 2oo2 voting logic implementations
    - Sensor configuration and validation
    - Safe state action definitions
    - Response time monitoring (< 500ms for SIL 2)
    - Proof test scheduling and tracking
    - Comprehensive audit trail

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - IEC 61508-6:2010 PFD Calculations
    - OSHA 29 CFR 1910.119 PSM

Example:
    >>> from greenlang.agents.process_heat.gl_001_thermal_command.sis_integration import (
    ...     SISManager, SISInterlock, VotingType, SensorConfig
    ... )
    >>>
    >>> manager = SISManager(sil_level=2)
    >>> interlock = SISInterlock(
    ...     name="HIGH_TEMP_SHUTDOWN",
    ...     voting_logic=VotingType.TWO_OO_THREE,
    ...     sensors=[sensor_a, sensor_b, sensor_c],
    ...     trip_setpoint=500.0,
    ...     safe_state=SafeStateAction.CLOSE_FUEL_VALVE,
    ...     response_time_ms=250
    ... )
    >>> manager.register_interlock(interlock)
    >>> result = manager.evaluate_interlock("HIGH_TEMP_SHUTDOWN", [480.0, 510.0, 505.0])

Author: GreenLang Safety Engineering Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
    """
    Voting architecture types per IEC 61511.

    Defines how multiple sensor channels are combined to make
    trip decisions for safety interlocks.
    """
    ONE_OO_ONE = "1oo1"      # Single channel (any trip triggers)
    ONE_OO_TWO = "1oo2"      # Any one of two triggers
    TWO_OO_TWO = "2oo2"      # Both must agree (high availability)
    TWO_OO_THREE = "2oo3"    # Two of three must agree (TMR)
    ONE_OO_THREE = "1oo3"    # Any one of three triggers
    THREE_OO_THREE = "3oo3"  # All three must agree
    TWO_OO_FOUR = "2oo4"     # Two of four must agree


class SafeStateAction(str, Enum):
    """
    Safe state actions for interlocks.

    Defines the action taken when an interlock trips.
    All actions must be fail-safe (de-energize to trip).
    """
    CLOSE_FUEL_VALVE = "close_fuel_valve"
    OPEN_VENT_VALVE = "open_vent_valve"
    STOP_COMBUSTION_AIR = "stop_combustion_air"
    TRIP_BURNER = "trip_burner"
    ISOLATE_FUEL_TRAIN = "isolate_fuel_train"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    REDUCE_FIRING_RATE = "reduce_firing_rate"
    ACTIVATE_QUENCH = "activate_quench"
    CLOSE_STEAM_VALVE = "close_steam_valve"
    OPEN_RELIEF_VALVE = "open_relief_valve"
    STOP_FEED_PUMP = "stop_feed_pump"
    INITIATE_BLOWDOWN = "initiate_blowdown"
    CUSTOM = "custom"


class SensorType(str, Enum):
    """Sensor types for SIS inputs."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    LEVEL = "level"
    FLOW = "flow"
    FLAME = "flame"
    COMBUSTIBLES = "combustibles"
    POSITION = "position"
    VIBRATION = "vibration"
    SPEED = "speed"


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


# =============================================================================
# DATA MODELS
# =============================================================================

class SensorConfig(BaseModel):
    """
    Configuration for a single SIS sensor channel.

    Defines the sensor parameters including calibration,
    response characteristics, and failure modes.
    """
    sensor_id: str = Field(
        ...,
        description="Unique sensor identifier (tag number)"
    )
    channel: str = Field(
        ...,
        description="Channel identifier (A, B, C for TMR)"
    )
    sensor_type: SensorType = Field(
        ...,
        description="Type of sensor"
    )
    tag_name: str = Field(
        ...,
        description="Process tag name"
    )
    engineering_units: str = Field(
        ...,
        description="Engineering units (e.g., degF, psig)"
    )
    range_low: float = Field(
        ...,
        description="Low end of calibrated range"
    )
    range_high: float = Field(
        ...,
        description="High end of calibrated range"
    )
    accuracy_percent: float = Field(
        default=0.5,
        ge=0,
        le=10,
        description="Sensor accuracy as percent of span"
    )
    response_time_ms: float = Field(
        default=100.0,
        ge=0,
        description="Sensor response time in milliseconds"
    )
    fail_direction: str = Field(
        default="high",
        description="Fail direction (high/low) for fail-safe"
    )
    lambda_du: float = Field(
        default=1e-6,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    diagnostics_enabled: bool = Field(
        default=True,
        description="Whether online diagnostics are enabled"
    )
    status: SensorStatus = Field(
        default=SensorStatus.NORMAL,
        description="Current sensor status"
    )
    last_calibration: Optional[datetime] = Field(
        default=None,
        description="Last calibration date"
    )
    calibration_interval_days: int = Field(
        default=365,
        ge=1,
        description="Required calibration interval"
    )

    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Validate channel identifier."""
        valid_channels = {'A', 'B', 'C', 'D', '1', '2', '3', '4'}
        if v.upper() not in valid_channels:
            raise ValueError(f"Invalid channel: {v}. Must be one of {valid_channels}")
        return v.upper()


class SensorReading(BaseModel):
    """Real-time reading from a sensor channel."""
    sensor_id: str = Field(..., description="Sensor identifier")
    channel: str = Field(..., description="Channel identifier")
    value: float = Field(..., description="Current value in engineering units")
    quality: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Signal quality (0-1)"
    )
    status: SensorStatus = Field(
        default=SensorStatus.NORMAL,
        description="Sensor status"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )


class VotingResult(BaseModel):
    """Result of voting logic evaluation."""
    voting_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Voting evaluation ID"
    )
    voting_type: VotingType = Field(
        ...,
        description="Voting architecture used"
    )
    trip_decision: bool = Field(
        ...,
        description="Final trip decision"
    )
    channels_voting_trip: int = Field(
        ...,
        description="Number of channels voting for trip"
    )
    channels_total: int = Field(
        ...,
        description="Total number of channels"
    )
    channels_required: int = Field(
        ...,
        description="Channels required for trip"
    )
    channels_healthy: int = Field(
        ...,
        description="Number of healthy channels"
    )
    channels_bypassed: int = Field(
        default=0,
        description="Number of bypassed channels"
    )
    channels_faulted: int = Field(
        default=0,
        description="Number of faulted channels"
    )
    degraded_mode: bool = Field(
        default=False,
        description="Is system in degraded mode?"
    )
    effective_voting: str = Field(
        default="",
        description="Effective voting after degradation"
    )
    channel_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Details of each channel"
    )
    evaluation_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Evaluation time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Evaluation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )


class SISInterlock(BaseModel):
    """
    Safety Instrumented System Interlock definition.

    Represents a complete safety interlock including sensors,
    voting logic, setpoints, and safe state actions.

    Per IEC 61511, response time must be < 500ms for SIL 2.
    """
    interlock_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique interlock identifier"
    )
    name: str = Field(
        ...,
        description="Interlock name/description"
    )
    voting_logic: VotingType = Field(
        ...,
        description="Voting architecture"
    )
    sensors: List[SensorConfig] = Field(
        ...,
        min_length=1,
        description="Sensor configurations"
    )
    trip_setpoint: float = Field(
        ...,
        description="Trip setpoint value"
    )
    trip_direction: str = Field(
        default="high",
        description="Trip on high or low"
    )
    deadband: float = Field(
        default=0.0,
        ge=0,
        description="Deadband for reset"
    )
    safe_state: SafeStateAction = Field(
        ...,
        description="Safe state action on trip"
    )
    custom_action: Optional[str] = Field(
        default=None,
        description="Custom action if safe_state is CUSTOM"
    )
    response_time_ms: int = Field(
        ...,
        ge=0,
        le=500,
        description="Required response time (< 500ms for SIL 2)"
    )
    sil_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Safety Integrity Level"
    )
    pfd_target: float = Field(
        default=0.005,
        ge=1e-5,
        le=0.1,
        description="Target PFD average"
    )
    proof_test_interval_hours: float = Field(
        default=8760.0,
        ge=168,
        description="Proof test interval in hours"
    )
    status: InterlockStatus = Field(
        default=InterlockStatus.ARMED,
        description="Current interlock status"
    )
    bypass_active: bool = Field(
        default=False,
        description="Is bypass currently active?"
    )
    bypass_reason: Optional[BypassReason] = Field(
        default=None,
        description="Reason for bypass if active"
    )
    bypass_authorized_by: Optional[str] = Field(
        default=None,
        description="Person who authorized bypass"
    )
    bypass_expiry: Optional[datetime] = Field(
        default=None,
        description="Bypass expiration time"
    )
    last_trip_time: Optional[datetime] = Field(
        default=None,
        description="Last trip timestamp"
    )
    trip_count: int = Field(
        default=0,
        ge=0,
        description="Total trip count"
    )
    last_proof_test: Optional[datetime] = Field(
        default=None,
        description="Last proof test date"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    @field_validator('response_time_ms')
    @classmethod
    def validate_response_time(cls, v: int) -> int:
        """Validate response time meets SIL 2 requirements."""
        if v > 500:
            logger.warning(
                f"Response time {v}ms exceeds 500ms limit for SIL 2. "
                "Consider using faster components."
            )
        return v

    @field_validator('sensors')
    @classmethod
    def validate_sensors_match_voting(cls, v: List[SensorConfig], info) -> List[SensorConfig]:
        """Validate sensor count matches voting architecture."""
        voting = info.data.get('voting_logic')
        if voting:
            required_counts = {
                VotingType.ONE_OO_ONE: 1,
                VotingType.ONE_OO_TWO: 2,
                VotingType.TWO_OO_TWO: 2,
                VotingType.TWO_OO_THREE: 3,
                VotingType.ONE_OO_THREE: 3,
                VotingType.THREE_OO_THREE: 3,
                VotingType.TWO_OO_FOUR: 4,
            }
            required = required_counts.get(voting, 1)
            if len(v) < required:
                raise ValueError(
                    f"Voting {voting.value} requires at least {required} sensors, "
                    f"but only {len(v)} provided"
                )
        return v


class InterlockTrip(BaseModel):
    """Record of an interlock trip event."""
    trip_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Trip event ID"
    )
    interlock_id: str = Field(..., description="Interlock that tripped")
    interlock_name: str = Field(..., description="Interlock name")
    trip_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Trip timestamp"
    )
    trip_value: float = Field(..., description="Value that caused trip")
    setpoint: float = Field(..., description="Trip setpoint")
    voting_result: VotingResult = Field(..., description="Voting evaluation")
    safe_state_action: SafeStateAction = Field(..., description="Action taken")
    response_time_actual_ms: float = Field(
        ...,
        ge=0,
        description="Actual response time in milliseconds"
    )
    sensor_readings: List[SensorReading] = Field(
        default_factory=list,
        description="Sensor values at trip"
    )
    acknowledged: bool = Field(default=False, description="Trip acknowledged?")
    acknowledged_by: Optional[str] = Field(default=None, description="Acknowledger")
    acknowledged_time: Optional[datetime] = Field(default=None, description="Ack time")
    reset_time: Optional[datetime] = Field(default=None, description="Reset time")
    reset_by: Optional[str] = Field(default=None, description="Reset operator")
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{self.interlock_id}|{self.trip_time.isoformat()}|"
            f"{self.trip_value}|{self.setpoint}|{self.response_time_actual_ms}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


class ProofTestSchedule(BaseModel):
    """Proof test schedule for an interlock."""
    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Schedule ID"
    )
    interlock_id: str = Field(..., description="Interlock ID")
    test_interval_hours: float = Field(..., ge=168, description="Test interval")
    last_test_date: Optional[datetime] = Field(default=None, description="Last test")
    next_test_date: Optional[datetime] = Field(default=None, description="Next test due")
    test_procedure_id: str = Field(default="", description="Procedure document ID")
    assigned_technician: Optional[str] = Field(default=None, description="Assigned tech")


# =============================================================================
# VOTING LOGIC ENGINE
# =============================================================================

class VotingEngine:
    """
    Voting logic engine for SIS channels.

    Implements IEC 61511 voting architectures with support for:
    - Standard voting evaluation
    - Degraded mode operation
    - Bypassed channel handling
    - Fault detection integration
    - Response time tracking
    """

    # Voting configuration: (required_channels, total_channels)
    VOTING_CONFIG: Dict[VotingType, Tuple[int, int]] = {
        VotingType.ONE_OO_ONE: (1, 1),
        VotingType.ONE_OO_TWO: (1, 2),
        VotingType.TWO_OO_TWO: (2, 2),
        VotingType.TWO_OO_THREE: (2, 3),
        VotingType.ONE_OO_THREE: (1, 3),
        VotingType.THREE_OO_THREE: (3, 3),
        VotingType.TWO_OO_FOUR: (2, 4),
    }

    def __init__(self, fail_safe_on_fault: bool = True) -> None:
        """
        Initialize voting engine.

        Args:
            fail_safe_on_fault: If True, faulted channels vote for trip
        """
        self.fail_safe_on_fault = fail_safe_on_fault
        logger.info("VotingEngine initialized (fail_safe_on_fault=%s)", fail_safe_on_fault)

    def evaluate(
        self,
        voting_type: VotingType,
        readings: List[SensorReading],
        setpoint: float,
        trip_direction: str = "high"
    ) -> VotingResult:
        """
        Evaluate voting logic with sensor readings.

        Args:
            voting_type: Voting architecture to use
            readings: List of sensor readings
            setpoint: Trip setpoint
            trip_direction: "high" or "low"

        Returns:
            VotingResult with trip decision
        """
        start_time = datetime.now(timezone.utc)

        required, expected = self.VOTING_CONFIG[voting_type]

        # Classify channels
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

            # Determine if channel votes for trip
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
                # Compare to setpoint
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

        # Check for degraded mode
        effective_channels = len(readings) - channels_bypassed
        degraded_mode = (
            channels_bypassed > 0 or
            channels_faulted > 0 or
            len(readings) != expected
        )

        # Determine effective voting in degraded mode
        effective_voting, effective_required = self._get_degraded_voting(
            voting_type, effective_channels, channels_bypassed, channels_faulted
        )

        # Make trip decision
        trip_decision = channels_trip >= effective_required

        # Calculate evaluation time
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

        # Calculate provenance
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
        """
        Determine effective voting in degraded mode.

        Per IEC 61511, when channels fail, the architecture
        degrades to a more conservative configuration.
        """
        # 2oo3 degradation logic
        if voting_type == VotingType.TWO_OO_THREE:
            if healthy_channels == 3:
                return "2oo3", 2
            elif healthy_channels == 2:
                return "1oo2", 1  # More conservative
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0  # All failed - force trip

        # 1oo2 degradation
        elif voting_type == VotingType.ONE_OO_TWO:
            if healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        # 2oo2 degradation
        elif voting_type == VotingType.TWO_OO_TWO:
            if healthy_channels == 2:
                return "2oo2", 2
            elif healthy_channels == 1:
                return "1oo1", 1  # Single channel remains
            else:
                return "0oo0", 0

        # 2oo4 degradation
        elif voting_type == VotingType.TWO_OO_FOUR:
            if healthy_channels == 4:
                return "2oo4", 2
            elif healthy_channels == 3:
                return "2oo3", 2
            elif healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        # Default - use original
        required, _ = self.VOTING_CONFIG[voting_type]
        return voting_type.value, required

    def _calculate_provenance(self, result: VotingResult) -> str:
        """Calculate SHA-256 provenance hash for voting result."""
        provenance_str = (
            f"{result.voting_type.value}|{result.trip_decision}|"
            f"{result.channels_voting_trip}|{result.channels_total}|"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# SIS MANAGER
# =============================================================================

class SISManager:
    """
    Safety Instrumented System Manager.

    Central manager for SIS interlocks in the ThermalCommand Orchestrator.
    Handles interlock registration, evaluation, bypass management, and
    comprehensive audit logging.

    Features:
        - IEC 61511 SIL 2 compliance
        - 2oo3 voting logic (TMR)
        - Response time monitoring
        - Bypass authorization and tracking
        - Proof test scheduling
        - Comprehensive audit trail

    Example:
        >>> manager = SISManager(sil_level=2)
        >>> manager.register_interlock(high_temp_interlock)
        >>> result = manager.evaluate_interlock(
        ...     "HIGH_TEMP_SHUTDOWN",
        ...     [SensorReading(sensor_id="TT-101A", channel="A", value=510.0), ...]
        ... )
        >>> if result.trip_decision:
        ...     await manager.execute_trip("HIGH_TEMP_SHUTDOWN", result)
    """

    def __init__(
        self,
        sil_level: int = 2,
        fail_safe_on_fault: bool = True,
        max_bypass_hours: float = 8.0
    ) -> None:
        """
        Initialize SIS Manager.

        Args:
            sil_level: Target Safety Integrity Level (1-4)
            fail_safe_on_fault: Trip on sensor fault
            max_bypass_hours: Maximum bypass duration
        """
        if sil_level < 1 or sil_level > 4:
            raise ValueError(f"Invalid SIL level: {sil_level}. Must be 1-4.")

        self.sil_level = sil_level
        self.max_bypass_hours = max_bypass_hours

        # Core components
        self._voting_engine = VotingEngine(fail_safe_on_fault=fail_safe_on_fault)
        self._interlocks: Dict[str, SISInterlock] = {}
        self._trip_history: List[InterlockTrip] = []
        self._proof_test_schedules: Dict[str, ProofTestSchedule] = {}

        # Callbacks for trip actions
        self._trip_callbacks: Dict[SafeStateAction, Callable] = {}

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        logger.info(
            "SISManager initialized (SIL-%d, fail_safe=%s, max_bypass=%sh)",
            sil_level, fail_safe_on_fault, max_bypass_hours
        )

    # =========================================================================
    # INTERLOCK MANAGEMENT
    # =========================================================================

    def register_interlock(self, interlock: SISInterlock) -> bool:
        """
        Register a safety interlock.

        Args:
            interlock: Interlock configuration

        Returns:
            True if registered successfully
        """
        if interlock.interlock_id in self._interlocks:
            logger.warning(
                "Interlock %s already registered", interlock.interlock_id
            )
            return False

        # Validate SIL level
        if interlock.sil_level > self.sil_level:
            logger.warning(
                "Interlock %s requires SIL-%d but manager configured for SIL-%d",
                interlock.name, interlock.sil_level, self.sil_level
            )

        # Validate response time for SIL 2
        if self.sil_level >= 2 and interlock.response_time_ms > 500:
            raise ValueError(
                f"Interlock {interlock.name} response time {interlock.response_time_ms}ms "
                f"exceeds 500ms limit for SIL-{self.sil_level}"
            )

        self._interlocks[interlock.interlock_id] = interlock

        # Create proof test schedule
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
            voting=interlock.voting_logic.value,
            sil_level=interlock.sil_level,
        )

        logger.info(
            "Interlock registered: %s (%s, %s, SIL-%d)",
            interlock.name, interlock.interlock_id,
            interlock.voting_logic.value, interlock.sil_level
        )
        return True

    def deregister_interlock(self, interlock_id: str) -> bool:
        """Deregister an interlock."""
        if interlock_id not in self._interlocks:
            return False

        interlock = self._interlocks[interlock_id]
        del self._interlocks[interlock_id]
        self._proof_test_schedules.pop(interlock_id, None)

        self._log_audit(
            "INTERLOCK_DEREGISTERED",
            interlock_id=interlock_id,
            name=interlock.name,
        )

        logger.info("Interlock deregistered: %s", interlock_id)
        return True

    def get_interlock(self, interlock_id: str) -> Optional[SISInterlock]:
        """Get interlock by ID."""
        return self._interlocks.get(interlock_id)

    def get_all_interlocks(self) -> List[SISInterlock]:
        """Get all registered interlocks."""
        return list(self._interlocks.values())

    # =========================================================================
    # INTERLOCK EVALUATION
    # =========================================================================

    def evaluate_interlock(
        self,
        interlock_id: str,
        readings: List[SensorReading]
    ) -> VotingResult:
        """
        Evaluate an interlock with current sensor readings.

        Args:
            interlock_id: Interlock to evaluate
            readings: Current sensor readings

        Returns:
            VotingResult with trip decision

        Raises:
            ValueError: If interlock not found
        """
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            raise ValueError(f"Interlock not found: {interlock_id}")

        # Check if bypassed
        if interlock.bypass_active:
            # Check if bypass expired
            if interlock.bypass_expiry and datetime.now(timezone.utc) > interlock.bypass_expiry:
                self._clear_bypass(interlock_id)
            else:
                logger.warning(
                    "Interlock %s is bypassed - returning no-trip",
                    interlock.name
                )
                return VotingResult(
                    voting_type=interlock.voting_logic,
                    trip_decision=False,
                    channels_voting_trip=0,
                    channels_total=len(readings),
                    channels_required=0,
                    channels_healthy=len(readings),
                    effective_voting="BYPASSED",
                )

        # Evaluate voting logic
        result = self._voting_engine.evaluate(
            voting_type=interlock.voting_logic,
            readings=readings,
            setpoint=interlock.trip_setpoint,
            trip_direction=interlock.trip_direction,
        )

        return result

    async def execute_trip(
        self,
        interlock_id: str,
        voting_result: VotingResult,
        readings: List[SensorReading]
    ) -> InterlockTrip:
        """
        Execute interlock trip action.

        Args:
            interlock_id: Interlock that tripped
            voting_result: Voting evaluation result
            readings: Sensor readings at trip

        Returns:
            InterlockTrip record
        """
        start_time = datetime.now(timezone.utc)

        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            raise ValueError(f"Interlock not found: {interlock_id}")

        # Get trip value (highest/lowest depending on direction)
        if interlock.trip_direction == "high":
            trip_value = max(r.value for r in readings if r.status == SensorStatus.NORMAL)
        else:
            trip_value = min(r.value for r in readings if r.status == SensorStatus.NORMAL)

        # Execute safe state action
        await self._execute_safe_state_action(interlock.safe_state, interlock_id)

        # Calculate actual response time
        response_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        # Update interlock state
        interlock.status = InterlockStatus.TRIPPED
        interlock.last_trip_time = datetime.now(timezone.utc)
        interlock.trip_count += 1

        # Create trip record
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

        # Log audit
        self._log_audit(
            "INTERLOCK_TRIP",
            interlock_id=interlock_id,
            name=interlock.name,
            trip_value=trip_value,
            setpoint=interlock.trip_setpoint,
            response_time_ms=response_time_ms,
            action=interlock.safe_state.value,
        )

        # Check response time against requirement
        if response_time_ms > interlock.response_time_ms:
            logger.error(
                "RESPONSE TIME EXCEEDED: %s actual %.1fms > required %dms",
                interlock.name, response_time_ms, interlock.response_time_ms
            )

        logger.critical(
            "INTERLOCK TRIPPED: %s (value=%.1f > setpoint=%.1f, action=%s, response=%.1fms)",
            interlock.name, trip_value, interlock.trip_setpoint,
            interlock.safe_state.value, response_time_ms
        )

        return trip_record

    async def _execute_safe_state_action(
        self,
        action: SafeStateAction,
        interlock_id: str
    ) -> None:
        """Execute the safe state action."""
        logger.info("Executing safe state action: %s for %s", action.value, interlock_id)

        # Check for registered callback
        if action in self._trip_callbacks:
            try:
                callback = self._trip_callbacks[action]
                if asyncio.iscoroutinefunction(callback):
                    await callback(interlock_id)
                else:
                    callback(interlock_id)
            except Exception as e:
                logger.error("Safe state callback failed: %s", e, exc_info=True)

        # Simulated action execution (in production, sends to DCS/PLC)
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
    # RESET OPERATIONS
    # =========================================================================

    async def reset_interlock(
        self,
        interlock_id: str,
        reset_by: str,
        force: bool = False
    ) -> bool:
        """
        Reset a tripped interlock.

        Args:
            interlock_id: Interlock to reset
            reset_by: Person authorizing reset
            force: Force reset even if conditions not met

        Returns:
            True if reset successful
        """
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if interlock.status != InterlockStatus.TRIPPED:
            logger.warning("Cannot reset %s: not in tripped state", interlock.name)
            return False

        # Update state
        interlock.status = InterlockStatus.ARMED

        # Update trip record
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

        logger.info(
            "Interlock reset: %s by %s",
            interlock.name, reset_by
        )
        return True

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
        """
        Request bypass for an interlock.

        Args:
            interlock_id: Interlock to bypass
            reason: Authorized bypass reason
            authorized_by: Person authorizing bypass
            duration_hours: Bypass duration

        Returns:
            True if bypass approved
        """
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        # Enforce maximum bypass duration
        if duration_hours > self.max_bypass_hours:
            logger.warning(
                "Bypass duration %.1fh exceeds maximum %.1fh",
                duration_hours, self.max_bypass_hours
            )
            duration_hours = self.max_bypass_hours

        # Activate bypass
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
            expiry=interlock.bypass_expiry.isoformat(),
        )

        logger.warning(
            "BYPASS ACTIVATED: %s by %s for %s (expires %s)",
            interlock.name, authorized_by, reason.value,
            interlock.bypass_expiry.strftime("%Y-%m-%d %H:%M")
        )
        return True

    def clear_bypass(self, interlock_id: str, cleared_by: str) -> bool:
        """Clear an active bypass."""
        return self._clear_bypass(interlock_id, cleared_by)

    def _clear_bypass(
        self,
        interlock_id: str,
        cleared_by: str = "SYSTEM"
    ) -> bool:
        """Internal bypass clear."""
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return False

        if not interlock.bypass_active:
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

        logger.info("Bypass cleared: %s by %s", interlock.name, cleared_by)
        return True

    def get_active_bypasses(self) -> List[SISInterlock]:
        """Get all interlocks with active bypasses."""
        return [
            interlock for interlock in self._interlocks.values()
            if interlock.bypass_active
        ]

    # =========================================================================
    # PROOF TEST MANAGEMENT
    # =========================================================================

    def get_overdue_proof_tests(self) -> List[ProofTestSchedule]:
        """Get interlocks with overdue proof tests."""
        now = datetime.now(timezone.utc)
        overdue = []

        for schedule in self._proof_test_schedules.values():
            if schedule.next_test_date and now > schedule.next_test_date:
                overdue.append(schedule)

        return overdue

    def record_proof_test(
        self,
        interlock_id: str,
        test_result: str,
        performed_by: str,
        notes: str = ""
    ) -> bool:
        """
        Record completion of a proof test.

        Args:
            interlock_id: Interlock tested
            test_result: "pass" or "fail"
            performed_by: Technician who performed test
            notes: Additional notes

        Returns:
            True if recorded successfully
        """
        interlock = self._interlocks.get(interlock_id)
        schedule = self._proof_test_schedules.get(interlock_id)

        if not interlock or not schedule:
            return False

        now = datetime.now(timezone.utc)
        interlock.last_proof_test = now
        schedule.last_test_date = now
        schedule.next_test_date = now + timedelta(hours=schedule.test_interval_hours)

        self._log_audit(
            "PROOF_TEST_COMPLETED",
            interlock_id=interlock_id,
            name=interlock.name,
            result=test_result,
            performed_by=performed_by,
            notes=notes,
        )

        logger.info(
            "Proof test recorded: %s - %s by %s",
            interlock.name, test_result, performed_by
        )
        return True

    # =========================================================================
    # STATUS AND MONITORING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get overall SIS status."""
        now = datetime.now(timezone.utc)

        tripped = [i for i in self._interlocks.values() if i.status == InterlockStatus.TRIPPED]
        bypassed = [i for i in self._interlocks.values() if i.bypass_active]
        faulted = [i for i in self._interlocks.values() if i.status == InterlockStatus.FAULT]
        overdue_tests = self.get_overdue_proof_tests()

        return {
            "sil_level": self.sil_level,
            "total_interlocks": len(self._interlocks),
            "armed": len([i for i in self._interlocks.values() if i.status == InterlockStatus.ARMED]),
            "tripped": len(tripped),
            "tripped_names": [i.name for i in tripped],
            "bypassed": len(bypassed),
            "bypassed_names": [i.name for i in bypassed],
            "faulted": len(faulted),
            "faulted_names": [i.name for i in faulted],
            "overdue_proof_tests": len(overdue_tests),
            "total_trip_count": sum(i.trip_count for i in self._interlocks.values()),
            "last_trip_time": max(
                (i.last_trip_time for i in self._interlocks.values() if i.last_trip_time),
                default=None
            ),
        }

    def get_trip_history(
        self,
        interlock_id: Optional[str] = None,
        limit: int = 100
    ) -> List[InterlockTrip]:
        """Get trip history, optionally filtered by interlock."""
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

        # Calculate provenance hash
        hash_str = f"{entry['timestamp']}|{event_type}|{str(kwargs)}"
        entry["provenance_hash"] = hashlib.sha256(hash_str.encode()).hexdigest()[:16]

        self._audit_log.append(entry)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_high_temperature_interlock(
    name: str,
    tag_prefix: str,
    setpoint_f: float,
    response_time_ms: int = 250
) -> SISInterlock:
    """
    Factory function to create a standard high temperature interlock.

    Creates a 2oo3 voting interlock for high temperature shutdown
    with three redundant temperature sensors.

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix (e.g., "TT-101")
        setpoint_f: Trip setpoint in degrees F
        response_time_ms: Required response time

    Returns:
        Configured SISInterlock
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

    return SISInterlock(
        name=name,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=setpoint_f,
        trip_direction="high",
        deadband=5.0,
        safe_state=SafeStateAction.TRIP_BURNER,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.005,
        proof_test_interval_hours=8760.0,
    )


def create_high_pressure_interlock(
    name: str,
    tag_prefix: str,
    setpoint_psig: float,
    response_time_ms: int = 200
) -> SISInterlock:
    """
    Factory function to create a high pressure interlock.

    Creates a 2oo3 voting interlock for high pressure shutdown.

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix (e.g., "PT-201")
        setpoint_psig: Trip setpoint in psig
        response_time_ms: Required response time

    Returns:
        Configured SISInterlock
    """
    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.PRESSURE,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="psig",
            range_low=0.0,
            range_high=300.0,
            accuracy_percent=0.25,
            response_time_ms=25.0,
            fail_direction="high",
        )
        for ch in ["A", "B", "C"]
    ]

    return SISInterlock(
        name=name,
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
    )


def create_low_level_interlock(
    name: str,
    tag_prefix: str,
    setpoint_percent: float,
    response_time_ms: int = 300
) -> SISInterlock:
    """
    Factory function to create a low level interlock.

    Creates a 2oo3 voting interlock for low level shutdown
    (e.g., boiler drum level protection).

    Args:
        name: Interlock name
        tag_prefix: Sensor tag prefix (e.g., "LT-301")
        setpoint_percent: Trip setpoint as percent of range
        response_time_ms: Required response time

    Returns:
        Configured SISInterlock
    """
    sensors = [
        SensorConfig(
            sensor_id=f"{tag_prefix}{ch}",
            channel=ch,
            sensor_type=SensorType.LEVEL,
            tag_name=f"{tag_prefix}{ch}",
            engineering_units="%",
            range_low=0.0,
            range_high=100.0,
            accuracy_percent=0.5,
            response_time_ms=100.0,
            fail_direction="low",
        )
        for ch in ["A", "B", "C"]
    ]

    return SISInterlock(
        name=name,
        voting_logic=VotingType.TWO_OO_THREE,
        sensors=sensors,
        trip_setpoint=setpoint_percent,
        trip_direction="low",
        deadband=2.0,
        safe_state=SafeStateAction.TRIP_BURNER,
        response_time_ms=response_time_ms,
        sil_level=2,
        pfd_target=0.005,
        proof_test_interval_hours=8760.0,
    )
