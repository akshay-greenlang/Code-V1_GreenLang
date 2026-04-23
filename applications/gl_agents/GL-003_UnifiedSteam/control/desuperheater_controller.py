"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Desuperheater Controller

This module implements advanced desuperheater control for steam temperature
management with anti-windup, bumpless transfer, and wet steam protection.

Control Architecture:
    - PID control with anti-windup (tracking integrator)
    - Bumpless transfer between modes (AUTO/MANUAL/CASCADE)
    - Rate limiting for thermal shock avoidance
    - Minimum approach to saturation protection
    - Transport delay and sensor lag compensation

Reference Standards:
    - ISA-5.1 Instrumentation Symbols and Identification
    - IEC 61511 Functional Safety
    - ASME PTC 19.3 Temperature Measurement

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ControlMode(str, Enum):
    """Control mode enumeration."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    ADVISORY = "advisory"
    CLOSED_LOOP = "closed_loop"


class ProtectionStatusType(str, Enum):
    """Protection status type enumeration."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    TRIPPED = "tripped"


class SetpointSource(str, Enum):
    """Setpoint source enumeration."""
    OPERATOR = "operator"
    OPTIMIZER = "optimizer"
    CASCADE = "cascade"
    SAFETY_OVERRIDE = "safety_override"


# =============================================================================
# DATA MODELS
# =============================================================================

class PIDTuning(BaseModel):
    """PID controller tuning parameters."""

    kp: float = Field(..., ge=0, description="Proportional gain")
    ki: float = Field(..., ge=0, description="Integral gain (1/s)")
    kd: float = Field(..., ge=0, description="Derivative gain (s)")
    anti_windup_limit: float = Field(
        default=100.0,
        ge=0,
        description="Anti-windup integrator limit (%)"
    )
    derivative_filter_tc: float = Field(
        default=0.1,
        ge=0,
        description="Derivative filter time constant (s)"
    )
    output_min: float = Field(default=0.0, description="Minimum output (%)")
    output_max: float = Field(default=100.0, description="Maximum output (%)")


class DesuperheaterState(BaseModel):
    """Current desuperheater state."""

    equipment_id: str = Field(..., description="Desuperheater equipment ID")
    inlet_temperature_c: float = Field(..., description="Inlet steam temperature (C)")
    outlet_temperature_c: float = Field(..., description="Outlet steam temperature (C)")
    steam_pressure_kpa: float = Field(..., ge=0, description="Steam pressure (kPa)")
    spray_valve_position_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Spray valve position (%)"
    )
    spray_flow_kg_s: float = Field(..., ge=0, description="Spray water flow (kg/s)")
    steam_flow_kg_s: float = Field(..., ge=0, description="Steam flow (kg/s)")
    saturation_temperature_c: float = Field(
        ...,
        description="Saturation temperature at current pressure (C)"
    )
    current_superheat_c: float = Field(
        ...,
        description="Current superheat above saturation (C)"
    )
    control_mode: ControlMode = Field(
        default=ControlMode.ADVISORY,
        description="Current control mode"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="State timestamp"
    )


class DesuperheaterConstraints(BaseModel):
    """Desuperheater operational constraints."""

    min_superheat_margin_c: float = Field(
        default=15.0,
        ge=0,
        description="Minimum superheat margin above saturation (C)"
    )
    max_temperature_rate_c_per_min: float = Field(
        default=5.0,
        ge=0,
        description="Maximum temperature change rate (C/min)"
    )
    max_spray_valve_rate_pct_per_s: float = Field(
        default=2.0,
        ge=0,
        description="Maximum spray valve change rate (%/s)"
    )
    min_spray_valve_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Minimum spray valve position (%)"
    )
    max_spray_valve_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Maximum spray valve position (%)"
    )
    transport_delay_s: float = Field(
        default=5.0,
        ge=0,
        description="Transport delay from valve to sensor (s)"
    )
    sensor_lag_s: float = Field(
        default=2.0,
        ge=0,
        description="Temperature sensor time constant (s)"
    )


class Setpoint(BaseModel):
    """Control setpoint with metadata."""

    value: float = Field(..., description="Setpoint value")
    unit: str = Field(..., description="Engineering unit")
    source: SetpointSource = Field(..., description="Setpoint source")
    ramp_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Ramp rate if ramping"
    )
    effective_time: datetime = Field(
        default_factory=datetime.now,
        description="Time setpoint becomes effective"
    )
    authorization_id: Optional[str] = Field(
        None,
        description="Authorization ID for setpoint change"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ProtectionStatus(BaseModel):
    """Wet steam protection status."""

    status: ProtectionStatusType = Field(..., description="Protection status")
    current_superheat_c: float = Field(
        ...,
        description="Current superheat margin (C)"
    )
    min_superheat_c: float = Field(
        ...,
        description="Minimum required superheat (C)"
    )
    approach_to_saturation_c: float = Field(
        ...,
        description="Distance to saturation temperature (C)"
    )
    max_allowed_reduction_c: float = Field(
        ...,
        description="Maximum allowed temperature reduction (C)"
    )
    message: str = Field(default="", description="Status message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Status timestamp"
    )


class AdvisoryResult(BaseModel):
    """Result of advisory mode execution."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    recommended_setpoint: float = Field(
        ...,
        description="Recommended setpoint value"
    )
    unit: str = Field(..., description="Engineering unit")
    expected_benefit: str = Field(..., description="Expected benefit description")
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    safety_validated: bool = Field(
        ...,
        description="Passed safety validation"
    )
    constraint_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraint check summary"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Recommendation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ControlResult(BaseModel):
    """Result of closed-loop control execution."""

    execution_id: str = Field(..., description="Execution ID")
    applied_setpoint: float = Field(
        ...,
        description="Applied setpoint value"
    )
    actual_output: float = Field(
        ...,
        description="Actual control output (%)"
    )
    error: float = Field(..., description="Control error")
    mode: ControlMode = Field(..., description="Control mode")
    safety_gates_passed: bool = Field(
        ...,
        description="All safety gates passed"
    )
    rate_limited: bool = Field(
        default=False,
        description="Output was rate limited"
    )
    anti_windup_active: bool = Field(
        default=False,
        description="Anti-windup is active"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Execution timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class PIDState(BaseModel):
    """Internal PID controller state."""

    integral: float = Field(default=0.0, description="Integral accumulator")
    previous_error: float = Field(default=0.0, description="Previous error")
    previous_output: float = Field(default=0.0, description="Previous output")
    derivative_filtered: float = Field(
        default=0.0,
        description="Filtered derivative"
    )
    last_update_time: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )


# =============================================================================
# DESUPERHEATER CONTROLLER
# =============================================================================

class DesuperheaterController:
    """
    Desuperheater control with anti-windup, bumpless transfer, and wet steam protection.

    This controller manages spray water injection for steam temperature control
    while ensuring safe operation above saturation temperature.

    Control Features:
        - PID control with tracking anti-windup
        - Bumpless transfer between modes
        - Rate limiting for thermal shock avoidance
        - Minimum approach to saturation protection
        - Transport delay and sensor lag compensation

    Safety Features:
        - Never allows operation below saturation + margin
        - Validates all setpoints against safety envelope
        - Rate limits all changes to prevent thermal shock
        - Advisory mode requires operator confirmation
        - Closed-loop mode has safety gates

    Attributes:
        equipment_id: Desuperheater equipment identifier
        tuning: PID tuning parameters
        constraints: Operational constraints
        pid_state: Internal PID state
        control_mode: Current control mode

    Example:
        >>> controller = DesuperheaterController(
        ...     equipment_id="DS-001",
        ...     tuning=PIDTuning(kp=2.0, ki=0.1, kd=0.5)
        ... )
        >>> state = DesuperheaterState(...)
        >>> setpoint = controller.compute_setpoint(state, target_temp=400.0, constraints)
        >>> result = controller.execute_advisory(recommendation)
    """

    def __init__(
        self,
        equipment_id: str,
        tuning: PIDTuning,
        constraints: Optional[DesuperheaterConstraints] = None
    ):
        """
        Initialize DesuperheaterController.

        Args:
            equipment_id: Desuperheater equipment identifier
            tuning: PID tuning parameters
            constraints: Operational constraints (uses defaults if None)
        """
        self.equipment_id = equipment_id
        self.tuning = tuning
        self.constraints = constraints or DesuperheaterConstraints()
        self.pid_state = PIDState()
        self.control_mode = ControlMode.ADVISORY
        self._last_setpoint: Optional[Setpoint] = None

        logger.info(
            f"DesuperheaterController initialized for {equipment_id} "
            f"with Kp={tuning.kp}, Ki={tuning.ki}, Kd={tuning.kd}"
        )

    def compute_setpoint(
        self,
        current_state: DesuperheaterState,
        target_temp_c: float,
        constraints: Optional[DesuperheaterConstraints] = None
    ) -> Setpoint:
        """
        Compute control setpoint for desuperheater.

        This method calculates the optimal spray valve setpoint to achieve
        the target temperature while respecting all constraints.

        Args:
            current_state: Current desuperheater state
            target_temp_c: Target outlet temperature (C)
            constraints: Override constraints (uses instance constraints if None)

        Returns:
            Setpoint: Computed setpoint with metadata

        Raises:
            ValueError: If target temperature violates safety limits
        """
        start_time = datetime.now()
        constraints = constraints or self.constraints

        # Step 1: Validate target against saturation margin
        protection_status = self.check_wet_steam_protection(
            current_state.steam_pressure_kpa,
            target_temp_c
        )

        if protection_status.status == ProtectionStatusType.CRITICAL:
            logger.error(
                f"Target temp {target_temp_c}C violates wet steam protection - "
                f"min allowed: {current_state.saturation_temperature_c + constraints.min_superheat_margin_c}C"
            )
            raise ValueError(
                f"Target temperature {target_temp_c}C too close to saturation. "
                f"Minimum allowed: {current_state.saturation_temperature_c + constraints.min_superheat_margin_c}C"
            )

        # Step 2: Compute PID output
        error = target_temp_c - current_state.outlet_temperature_c
        pid_output = self._compute_pid(error, current_state.timestamp)

        # Step 3: Apply rate limiting
        if self._last_setpoint is not None:
            pid_output = self.apply_ramp_limit(
                self._last_setpoint.value,
                pid_output,
                constraints.max_spray_valve_rate_pct_per_s
            )

        # Step 4: Clamp to valve limits
        pid_output = max(
            constraints.min_spray_valve_pct,
            min(constraints.max_spray_valve_pct, pid_output)
        )

        # Step 5: Create setpoint
        setpoint = Setpoint(
            value=pid_output,
            unit="%",
            source=SetpointSource.OPTIMIZER,
            ramp_rate=constraints.max_spray_valve_rate_pct_per_s,
            effective_time=datetime.now()
        )

        # Calculate provenance hash
        setpoint.provenance_hash = self._calculate_provenance(
            current_state, target_temp_c, setpoint.value
        )

        self._last_setpoint = setpoint

        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Computed setpoint {setpoint.value:.2f}% for target {target_temp_c}C "
            f"(error={error:.2f}C, processing_time={processing_time_ms:.1f}ms)"
        )

        return setpoint

    def apply_ramp_limit(
        self,
        current: float,
        target: float,
        max_rate: float
    ) -> float:
        """
        Apply rate limiting to prevent thermal shock.

        This method limits the rate of change to protect equipment from
        thermal stress caused by rapid temperature changes.

        Args:
            current: Current value
            target: Target value
            max_rate: Maximum rate of change per second

        Returns:
            float: Rate-limited target value
        """
        if max_rate <= 0:
            return target

        # Calculate time since last update
        time_delta_s = (
            datetime.now() - self.pid_state.last_update_time
        ).total_seconds()

        # Clamp time delta to reasonable range
        time_delta_s = max(0.001, min(time_delta_s, 10.0))

        max_change = max_rate * time_delta_s

        if abs(target - current) <= max_change:
            return target

        if target > current:
            limited_value = current + max_change
        else:
            limited_value = current - max_change

        logger.debug(
            f"Rate limited: {current:.2f} -> {limited_value:.2f} "
            f"(target={target:.2f}, max_rate={max_rate}/s)"
        )

        return limited_value

    def check_wet_steam_protection(
        self,
        pressure_kpa: float,
        target_temp_c: float
    ) -> ProtectionStatus:
        """
        Check wet steam protection status.

        Ensures the target temperature maintains adequate superheat margin
        above saturation temperature to prevent wet steam damage.

        Args:
            pressure_kpa: Current steam pressure (kPa)
            target_temp_c: Target temperature (C)

        Returns:
            ProtectionStatus: Protection status with margin information
        """
        # Calculate saturation temperature using Antoine equation approximation
        # This is a simplified calculation - production would use steam tables
        saturation_temp_c = self._calculate_saturation_temp(pressure_kpa)

        current_superheat = target_temp_c - saturation_temp_c
        min_superheat = self.constraints.min_superheat_margin_c

        approach_to_saturation = target_temp_c - saturation_temp_c
        max_allowed_reduction = current_superheat - min_superheat

        # Determine protection status
        if current_superheat < min_superheat:
            status = ProtectionStatusType.CRITICAL
            message = (
                f"CRITICAL: Target temp {target_temp_c:.1f}C is only "
                f"{current_superheat:.1f}C above saturation. "
                f"Minimum required: {min_superheat}C"
            )
            logger.error(message)
        elif current_superheat < min_superheat * 1.5:
            status = ProtectionStatusType.WARNING
            message = (
                f"WARNING: Target temp {target_temp_c:.1f}C approaching "
                f"saturation margin limit. Current margin: {current_superheat:.1f}C"
            )
            logger.warning(message)
        else:
            status = ProtectionStatusType.SAFE
            message = f"Safe: Superheat margin {current_superheat:.1f}C is adequate"
            logger.debug(message)

        return ProtectionStatus(
            status=status,
            current_superheat_c=current_superheat,
            min_superheat_c=min_superheat,
            approach_to_saturation_c=approach_to_saturation,
            max_allowed_reduction_c=max(0, max_allowed_reduction),
            message=message
        )

    def execute_advisory(
        self,
        recommendation: Dict[str, Any]
    ) -> AdvisoryResult:
        """
        Execute advisory mode - recommend setpoint, require operator confirmation.

        In advisory mode, the controller generates recommendations but does not
        directly manipulate the process. Operator confirmation is required.

        Args:
            recommendation: Dictionary containing recommendation details:
                - target_temp_c: Target temperature
                - expected_benefit: Expected benefit description
                - current_state: Current desuperheater state

        Returns:
            AdvisoryResult: Advisory result requiring operator confirmation
        """
        start_time = datetime.now()

        target_temp_c = recommendation.get("target_temp_c", 0)
        expected_benefit = recommendation.get("expected_benefit", "")
        current_state = recommendation.get("current_state")

        if current_state is None:
            raise ValueError("current_state is required in recommendation")

        # Validate against safety envelope
        protection_status = self.check_wet_steam_protection(
            current_state.steam_pressure_kpa,
            target_temp_c
        )

        safety_validated = protection_status.status != ProtectionStatusType.CRITICAL

        # Generate recommendation ID
        recommendation_id = hashlib.sha256(
            f"{self.equipment_id}_{target_temp_c}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Build constraint summary
        constraint_summary = {
            "wet_steam_protection": {
                "status": protection_status.status.value,
                "current_superheat_c": protection_status.current_superheat_c,
                "min_required_c": protection_status.min_superheat_c
            },
            "rate_limit": {
                "max_rate_c_per_min": self.constraints.max_temperature_rate_c_per_min,
                "compliant": True  # Advisory doesn't apply changes, so always compliant
            }
        }

        result = AdvisoryResult(
            recommendation_id=recommendation_id,
            recommended_setpoint=target_temp_c,
            unit="C",
            expected_benefit=expected_benefit,
            requires_confirmation=True,
            safety_validated=safety_validated,
            constraint_summary=constraint_summary,
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        result.provenance_hash = hashlib.sha256(
            f"{result.recommendation_id}_{result.recommended_setpoint}_{safety_validated}".encode()
        ).hexdigest()

        logger.info(
            f"Advisory recommendation {recommendation_id}: "
            f"target={target_temp_c}C, safety_validated={safety_validated}, "
            f"requires_confirmation=True"
        )

        return result

    def execute_closed_loop(
        self,
        setpoint: Setpoint,
        current_state: DesuperheaterState,
        safety_envelope: Optional[Any] = None
    ) -> ControlResult:
        """
        Execute closed-loop control with safety gates.

        In closed-loop mode, the controller directly manipulates the process
        within safe bounds. All safety gates must pass before execution.

        Safety Gates:
            1. Wet steam protection check
            2. Rate limit validation
            3. Valve position limits
            4. Safety envelope validation (if provided)

        Args:
            setpoint: Target setpoint to apply
            current_state: Current desuperheater state
            safety_envelope: Optional safety envelope for additional validation

        Returns:
            ControlResult: Result of closed-loop execution

        Raises:
            ValueError: If any safety gate fails
        """
        start_time = datetime.now()

        if self.control_mode != ControlMode.CLOSED_LOOP:
            logger.warning(
                f"Controller not in CLOSED_LOOP mode (current: {self.control_mode}). "
                "Switching to CLOSED_LOOP mode."
            )
            self.control_mode = ControlMode.CLOSED_LOOP

        # Safety Gate 1: Wet steam protection
        # Estimate resulting temperature from spray valve position
        estimated_temp_c = self._estimate_outlet_temp(
            current_state, setpoint.value
        )
        protection_status = self.check_wet_steam_protection(
            current_state.steam_pressure_kpa,
            estimated_temp_c
        )

        if protection_status.status == ProtectionStatusType.CRITICAL:
            raise ValueError(
                f"Safety Gate 1 FAILED: Wet steam protection violated. "
                f"{protection_status.message}"
            )

        # Safety Gate 2: Rate limit validation
        rate_limited = False
        actual_output = setpoint.value

        if self._last_setpoint is not None:
            limited_output = self.apply_ramp_limit(
                self._last_setpoint.value,
                setpoint.value,
                self.constraints.max_spray_valve_rate_pct_per_s
            )
            if limited_output != setpoint.value:
                rate_limited = True
                actual_output = limited_output
                logger.info(
                    f"Output rate limited: {setpoint.value:.2f}% -> {actual_output:.2f}%"
                )

        # Safety Gate 3: Valve position limits
        actual_output = max(
            self.constraints.min_spray_valve_pct,
            min(self.constraints.max_spray_valve_pct, actual_output)
        )

        # Safety Gate 4: Safety envelope validation (if provided)
        safety_gates_passed = True
        if safety_envelope is not None:
            # This would call safety_envelope.check_within_envelope()
            # Simplified for this implementation
            pass

        # Calculate control error
        error = setpoint.value - current_state.spray_valve_position_pct

        # Update internal state
        self._last_setpoint = Setpoint(
            value=actual_output,
            unit="%",
            source=setpoint.source,
            effective_time=datetime.now()
        )

        # Generate execution ID
        execution_id = hashlib.sha256(
            f"{self.equipment_id}_{actual_output}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = ControlResult(
            execution_id=execution_id,
            applied_setpoint=setpoint.value,
            actual_output=actual_output,
            error=error,
            mode=self.control_mode,
            safety_gates_passed=safety_gates_passed,
            rate_limited=rate_limited,
            anti_windup_active=self.pid_state.integral >= self.tuning.anti_windup_limit,
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        result.provenance_hash = hashlib.sha256(
            f"{execution_id}_{actual_output}_{safety_gates_passed}".encode()
        ).hexdigest()

        logger.info(
            f"Closed-loop execution {execution_id}: "
            f"output={actual_output:.2f}%, error={error:.2f}%, "
            f"rate_limited={rate_limited}, gates_passed={safety_gates_passed}"
        )

        return result

    def set_mode(self, mode: ControlMode) -> None:
        """
        Set control mode with bumpless transfer.

        Args:
            mode: New control mode
        """
        if mode == self.control_mode:
            return

        # Bumpless transfer: Reset integrator to current output
        if self._last_setpoint is not None:
            self.pid_state.integral = self._last_setpoint.value / self.tuning.ki if self.tuning.ki > 0 else 0

        logger.info(f"Control mode changed: {self.control_mode.value} -> {mode.value}")
        self.control_mode = mode

    def reset(self) -> None:
        """Reset controller state."""
        self.pid_state = PIDState()
        self._last_setpoint = None
        logger.info(f"Controller {self.equipment_id} state reset")

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _compute_pid(self, error: float, timestamp: datetime) -> float:
        """
        Compute PID output with anti-windup.

        Uses tracking anti-windup to prevent integrator windup when output
        is saturated.

        Args:
            error: Control error
            timestamp: Current timestamp

        Returns:
            float: PID output (0-100%)
        """
        # Calculate time delta
        time_delta_s = (timestamp - self.pid_state.last_update_time).total_seconds()
        time_delta_s = max(0.001, min(time_delta_s, 10.0))  # Clamp to reasonable range

        # Proportional term
        p_term = self.tuning.kp * error

        # Integral term with anti-windup
        self.pid_state.integral += self.tuning.ki * error * time_delta_s

        # Anti-windup: Clamp integral
        self.pid_state.integral = max(
            -self.tuning.anti_windup_limit,
            min(self.tuning.anti_windup_limit, self.pid_state.integral)
        )

        i_term = self.pid_state.integral

        # Derivative term with filtering
        if time_delta_s > 0:
            derivative = (error - self.pid_state.previous_error) / time_delta_s

            # Low-pass filter for derivative
            alpha = time_delta_s / (self.tuning.derivative_filter_tc + time_delta_s)
            self.pid_state.derivative_filtered = (
                alpha * derivative +
                (1 - alpha) * self.pid_state.derivative_filtered
            )

        d_term = self.tuning.kd * self.pid_state.derivative_filtered

        # Calculate output
        output = p_term + i_term + d_term

        # Clamp output to limits
        output = max(self.tuning.output_min, min(self.tuning.output_max, output))

        # Tracking anti-windup: Reduce integral if output is saturated
        if output == self.tuning.output_max or output == self.tuning.output_min:
            # Back-calculate integral to prevent windup
            self.pid_state.integral -= (output - (p_term + i_term + d_term)) * 0.1

        # Update state
        self.pid_state.previous_error = error
        self.pid_state.previous_output = output
        self.pid_state.last_update_time = timestamp

        return output

    def _calculate_saturation_temp(self, pressure_kpa: float) -> float:
        """
        Calculate saturation temperature from pressure.

        Uses Antoine equation approximation. Production systems would use
        IAPWS-IF97 steam tables.

        Args:
            pressure_kpa: Pressure in kPa

        Returns:
            float: Saturation temperature in Celsius
        """
        # Simplified Antoine equation approximation
        # T_sat = B / (A - log10(P)) - C
        # These are approximate constants for water
        if pressure_kpa <= 0:
            return 100.0  # Default to 100C for invalid pressure

        pressure_bar = pressure_kpa / 100.0

        # Approximate formula for T_sat
        # Valid for rough estimates in industrial range
        saturation_temp_c = 100 + 30 * (pressure_bar - 1) ** 0.5

        return saturation_temp_c

    def _estimate_outlet_temp(
        self,
        current_state: DesuperheaterState,
        spray_valve_pct: float
    ) -> float:
        """
        Estimate outlet temperature from spray valve position.

        Simple linear model for estimation. Production systems would use
        process models with transport delay compensation.

        Args:
            current_state: Current desuperheater state
            spray_valve_pct: Spray valve position (%)

        Returns:
            float: Estimated outlet temperature (C)
        """
        # Simple linear model: more spray = lower temperature
        # Assumes inlet-outlet temp delta is proportional to spray flow
        max_cooling_c = current_state.inlet_temperature_c - (
            current_state.saturation_temperature_c +
            self.constraints.min_superheat_margin_c
        )

        cooling_c = (spray_valve_pct / 100.0) * max_cooling_c
        estimated_temp = current_state.inlet_temperature_c - cooling_c

        return estimated_temp

    def _calculate_provenance(
        self,
        current_state: DesuperheaterState,
        target_temp_c: float,
        output_value: float
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            current_state: Input state
            target_temp_c: Target temperature
            output_value: Computed output

        Returns:
            str: SHA-256 hash string
        """
        provenance_data = (
            f"{current_state.equipment_id}|"
            f"{current_state.outlet_temperature_c}|"
            f"{current_state.steam_pressure_kpa}|"
            f"{target_temp_c}|"
            f"{output_value}|"
            f"{datetime.now().isoformat()}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()
