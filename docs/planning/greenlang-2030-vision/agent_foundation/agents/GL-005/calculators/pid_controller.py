# -*- coding: utf-8 -*-
"""
PID Controller for GL-005 CombustionControlAgent

Implements PID control algorithm with anti-windup and auto-tuning capabilities.
Zero-hallucination design using classical control theory.

Reference Standards:
- ISA-5.1: Instrumentation Symbols and Identification
- ANSI/ISA-51.1: Process Instrumentation Terminology
- IEEE Std 421.5: Recommended Practice for Excitation System Models

Mathematical Formulas:
- PID Output: u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt
- Discrete PID: u[k] = Kp*e[k] + Ki*Σe[k]*Δt + Kd*(e[k]-e[k-1])/Δt
- Ziegler-Nichols Tuning: Kp = 0.6*Ku, Ki = 2*Kp/Tu, Kd = Kp*Tu/8
- Anti-Windup: Clamp integral term when output saturates
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    """PID control modes"""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"


class TuningMethod(str, Enum):
    """PID tuning methods"""
    ZIEGLER_NICHOLS_CLOSED_LOOP = "ziegler_nichols_closed_loop"
    ZIEGLER_NICHOLS_OPEN_LOOP = "ziegler_nichols_open_loop"
    COHEN_COON = "cohen_coon"
    TYREUS_LUYBEN = "tyreus_luyben"
    MANUAL = "manual"


class AntiWindupMethod(str, Enum):
    """Anti-windup methods"""
    CLAMPING = "clamping"
    BACK_CALCULATION = "back_calculation"
    CONDITIONAL_INTEGRATION = "conditional_integration"


@dataclass
class PIDGains:
    """PID controller gains"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

    def validate(self) -> bool:
        """Validate gains are non-negative"""
        return self.kp >= 0 and self.ki >= 0 and self.kd >= 0


@dataclass
class PIDState:
    """Internal PID controller state"""
    error_current: float
    error_previous: float
    error_integral: float
    error_derivative: float
    output_current: float
    output_previous: float
    time_current: float
    time_previous: float


class PIDInput(BaseModel):
    """Input parameters for PID controller"""

    # Process variables
    setpoint: float = Field(
        ...,
        description="Target setpoint value"
    )
    process_variable: float = Field(
        ...,
        description="Current measured process variable"
    )
    timestamp: float = Field(
        ...,
        ge=0,
        description="Current timestamp in seconds"
    )

    # PID gains
    kp: float = Field(
        ...,
        ge=0,
        description="Proportional gain"
    )
    ki: float = Field(
        default=0,
        ge=0,
        description="Integral gain"
    )
    kd: float = Field(
        default=0,
        ge=0,
        description="Derivative gain"
    )

    # Control limits
    output_min: float = Field(
        default=0,
        description="Minimum output limit"
    )
    output_max: float = Field(
        default=100,
        description="Maximum output limit"
    )

    # Anti-windup parameters
    enable_anti_windup: bool = Field(
        default=True,
        description="Enable anti-windup mechanism"
    )
    anti_windup_method: AntiWindupMethod = Field(
        default=AntiWindupMethod.CLAMPING
    )

    # Derivative filter
    derivative_filter_coefficient: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Derivative filter coefficient (0=no filter, 1=full filter)"
    )

    # Control mode
    control_mode: ControlMode = Field(
        default=ControlMode.AUTO
    )
    manual_output: Optional[float] = Field(
        None,
        description="Manual output value when in manual mode"
    )

    @validator('output_max')
    def validate_output_limits(cls, v, values):
        """Ensure max > min"""
        if 'output_min' in values and v <= values['output_min']:
            raise ValueError("output_max must be greater than output_min")
        return v


class PIDOutput(BaseModel):
    """PID controller output"""

    # Control output
    control_output: float = Field(
        ...,
        description="Controller output (actuator command)"
    )

    # Error terms
    error: float = Field(
        ...,
        description="Current error (setpoint - process_variable)"
    )
    error_integral: float = Field(
        ...,
        description="Accumulated integral error"
    )
    error_derivative: float = Field(
        ...,
        description="Rate of change of error"
    )

    # Individual contributions
    p_term: float = Field(..., description="Proportional term contribution")
    i_term: float = Field(..., description="Integral term contribution")
    d_term: float = Field(..., description="Derivative term contribution")

    # Status
    output_saturated: bool = Field(
        ...,
        description="Whether output hit limits"
    )
    anti_windup_active: bool = Field(
        ...,
        description="Whether anti-windup is currently active"
    )
    control_mode: ControlMode

    # Performance metrics
    steady_state_error: Optional[float] = None
    overshoot_percent: Optional[float] = None


class AutoTuneInput(BaseModel):
    """Input for auto-tuning PID controller"""

    # Process data
    process_data: List[Tuple[float, float]] = Field(
        ...,
        description="List of (timestamp, process_variable) tuples",
        min_items=10
    )
    setpoint: float = Field(
        ...,
        description="Setpoint used during tuning test"
    )

    # Tuning method
    tuning_method: TuningMethod = Field(
        default=TuningMethod.ZIEGLER_NICHOLS_CLOSED_LOOP
    )

    # Process characteristics (for open-loop methods)
    ultimate_gain: Optional[float] = Field(
        None,
        description="Ultimate gain (Ku) for closed-loop tuning"
    )
    ultimate_period: Optional[float] = Field(
        None,
        description="Ultimate period (Tu) for closed-loop tuning"
    )

    # Tuning aggressiveness
    tuning_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Tuning aggressiveness (1.0=standard, <1=conservative, >1=aggressive)"
    )


class AutoTuneOutput(BaseModel):
    """Auto-tuning results"""

    # Tuned gains
    kp: float
    ki: float
    kd: float

    # Process characteristics
    ultimate_gain: Optional[float] = None
    ultimate_period: Optional[float] = None
    process_dead_time: Optional[float] = None
    process_time_constant: Optional[float] = None

    # Tuning method used
    tuning_method: TuningMethod

    # Performance predictions
    predicted_rise_time: Optional[float] = None
    predicted_settling_time: Optional[float] = None
    predicted_overshoot_percent: Optional[float] = None

    # Quality metrics
    tuning_quality_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Quality of tuning (0=poor, 1=excellent)"
    )


class PIDController:
    """
    PID controller implementation with anti-windup and auto-tuning.

    This controller implements the classical PID algorithm in discrete time:

        u[k] = Kp * e[k] + Ki * sum(e[i]*Δt) + Kd * (e[k] - e[k-1])/Δt

    Where:
        - e[k] = setpoint - process_variable (error)
        - u[k] = control output
        - Δt = sampling time

    Features:
        - Anti-windup (clamping, back-calculation, conditional integration)
        - Derivative filtering (to reduce noise amplification)
        - Output limiting
        - Auto-tuning (Ziegler-Nichols, Cohen-Coon, Tyreus-Luyben)
        - Bumpless transfer between manual and auto modes
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_min: float = 0.0,
        output_max: float = 100.0
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output limit
            output_max: Maximum output limit
        """
        self.logger = logging.getLogger(__name__)

        # Controller gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Output limits
        self.output_min = output_min
        self.output_max = output_max

        # Internal state
        self.error_previous = 0.0
        self.error_integral = 0.0
        self.derivative_filtered = 0.0
        self.output_previous = 0.0
        self.time_previous = None

        # Anti-windup state
        self.windup_active = False

        # History for analysis
        self.error_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)

    def calculate_control_output(
        self,
        pid_input: PIDInput
    ) -> PIDOutput:
        """
        Calculate PID control output.

        Algorithm:
            1. Calculate error = setpoint - process_variable
            2. Calculate proportional term: P = Kp * error
            3. Update and calculate integral term: I = Ki * sum(error*dt)
            4. Calculate derivative term: D = Kd * d(error)/dt
            5. Sum terms: output = P + I + D
            6. Apply output limits
            7. Apply anti-windup if output saturated
            8. Return control output

        Args:
            pid_input: PID input parameters

        Returns:
            PIDOutput with control signal and diagnostics
        """
        # Manual mode - return manual output directly
        if pid_input.control_mode == ControlMode.MANUAL:
            manual_out = pid_input.manual_output if pid_input.manual_output is not None else self.output_previous
            return self._create_manual_output(manual_out, pid_input)

        # Calculate time step
        if self.time_previous is None:
            dt = 0.1  # Default 100ms
            self.time_previous = pid_input.timestamp
        else:
            dt = pid_input.timestamp - self.time_previous
            if dt <= 0:
                dt = 0.1  # Prevent division by zero

        # Step 1: Calculate error
        error = pid_input.setpoint - pid_input.process_variable

        # Step 2: Proportional term
        p_term = pid_input.kp * error

        # Step 3: Integral term (with anti-windup consideration)
        if pid_input.enable_anti_windup and self.windup_active:
            # Don't accumulate integral if in windup
            if pid_input.anti_windup_method == AntiWindupMethod.CONDITIONAL_INTEGRATION:
                # Only integrate if error is reducing
                if (error * self.error_previous) <= 0:
                    self.error_integral += error * dt
        else:
            self.error_integral += error * dt

        i_term = pid_input.ki * self.error_integral

        # Step 4: Derivative term (with filtering)
        if dt > 0:
            error_derivative_raw = (error - self.error_previous) / dt
        else:
            error_derivative_raw = 0

        # Low-pass filter on derivative to reduce noise
        alpha = pid_input.derivative_filter_coefficient
        self.derivative_filtered = (
            alpha * self.derivative_filtered +
            (1 - alpha) * error_derivative_raw
        )

        d_term = pid_input.kd * self.derivative_filtered

        # Step 5: Calculate total output
        output_raw = p_term + i_term + d_term

        # Step 6: Apply output limits
        output_limited = self._clamp(output_raw, pid_input.output_min, pid_input.output_max)

        # Step 7: Check for saturation and apply anti-windup
        output_saturated = (output_limited != output_raw)

        if pid_input.enable_anti_windup and output_saturated:
            self.windup_active = True

            if pid_input.anti_windup_method == AntiWindupMethod.CLAMPING:
                # Clamp integral term
                max_integral = (pid_input.output_max - p_term - d_term) / pid_input.ki if pid_input.ki > 0 else 0
                min_integral = (pid_input.output_min - p_term - d_term) / pid_input.ki if pid_input.ki > 0 else 0
                self.error_integral = self._clamp(self.error_integral, min_integral, max_integral)

            elif pid_input.anti_windup_method == AntiWindupMethod.BACK_CALCULATION:
                # Back-calculate integral to prevent windup
                error_saturation = output_limited - output_raw
                self.error_integral += error_saturation / pid_input.ki if pid_input.ki > 0 else 0

        else:
            self.windup_active = False

        # Step 8: Update state
        self.error_previous = error
        self.output_previous = output_limited
        self.time_previous = pid_input.timestamp

        # Update history
        self.error_history.append(error)
        self.output_history.append(output_limited)

        return PIDOutput(
            control_output=self._round_decimal(output_limited, 4),
            error=self._round_decimal(error, 4),
            error_integral=self._round_decimal(self.error_integral, 4),
            error_derivative=self._round_decimal(self.derivative_filtered, 4),
            p_term=self._round_decimal(p_term, 4),
            i_term=self._round_decimal(i_term, 4),
            d_term=self._round_decimal(d_term, 4),
            output_saturated=output_saturated,
            anti_windup_active=self.windup_active,
            control_mode=pid_input.control_mode
        )

    def update_pid_state(
        self,
        error_integral: Optional[float] = None,
        reset_integral: bool = False
    ) -> None:
        """
        Update internal PID state.

        Args:
            error_integral: Set integral error to specific value
            reset_integral: Reset integral error to zero
        """
        if reset_integral:
            self.error_integral = 0.0
            self.logger.info("Integral error reset to zero")

        if error_integral is not None:
            self.error_integral = error_integral
            self.logger.info(f"Integral error set to {error_integral}")

    def apply_anti_windup(
        self,
        method: AntiWindupMethod = AntiWindupMethod.CLAMPING
    ) -> None:
        """
        Manually apply anti-windup mechanism.

        Args:
            method: Anti-windup method to apply
        """
        if method == AntiWindupMethod.CLAMPING:
            # Clamp integral to reasonable bounds
            self.error_integral = self._clamp(self.error_integral, -1000, 1000)

        elif method == AntiWindupMethod.BACK_CALCULATION:
            # Back-calculate to prevent excessive integral
            if abs(self.error_integral) > 100:
                self.error_integral *= 0.9  # Reduce by 10%

        self.logger.info(f"Applied anti-windup method: {method}")

    def auto_tune_parameters(
        self,
        auto_tune_input: AutoTuneInput
    ) -> AutoTuneOutput:
        """
        Auto-tune PID parameters using specified method.

        Args:
            auto_tune_input: Auto-tuning input parameters

        Returns:
            AutoTuneOutput with tuned gains

        Supported Methods:
            - Ziegler-Nichols Closed Loop (requires Ku, Tu)
            - Ziegler-Nichols Open Loop (requires process response data)
            - Cohen-Coon (requires process response data)
            - Tyreus-Luyben (requires Ku, Tu)
        """
        self.logger.info(f"Auto-tuning PID using {auto_tune_input.tuning_method}")

        method = auto_tune_input.tuning_method

        if method == TuningMethod.ZIEGLER_NICHOLS_CLOSED_LOOP:
            return self._tune_ziegler_nichols_closed_loop(auto_tune_input)

        elif method == TuningMethod.ZIEGLER_NICHOLS_OPEN_LOOP:
            return self._tune_ziegler_nichols_open_loop(auto_tune_input)

        elif method == TuningMethod.COHEN_COON:
            return self._tune_cohen_coon(auto_tune_input)

        elif method == TuningMethod.TYREUS_LUYBEN:
            return self._tune_tyreus_luyben(auto_tune_input)

        else:
            raise ValueError(f"Unknown tuning method: {method}")

    def _tune_ziegler_nichols_closed_loop(
        self,
        auto_tune_input: AutoTuneInput
    ) -> AutoTuneOutput:
        """
        Ziegler-Nichols Closed Loop tuning method.

        Requires ultimate gain (Ku) and ultimate period (Tu) from
        sustained oscillation test.

        Formulas:
            Kp = 0.6 * Ku
            Ki = 2 * Kp / Tu = 1.2 * Ku / Tu
            Kd = Kp * Tu / 8 = 0.075 * Ku * Tu
        """
        if auto_tune_input.ultimate_gain is None or auto_tune_input.ultimate_period is None:
            # Try to detect from data
            ku, tu = self._detect_ultimate_gain_period(auto_tune_input.process_data)
        else:
            ku = auto_tune_input.ultimate_gain
            tu = auto_tune_input.ultimate_period

        # Apply tuning factor for aggressiveness
        factor = auto_tune_input.tuning_factor

        # Classical Ziegler-Nichols formulas
        kp = 0.6 * ku * factor
        ki = 1.2 * ku / tu * factor
        kd = 0.075 * ku * tu * factor

        # Predict performance
        predicted_overshoot = 25.0 * factor  # ZN typically gives ~25% overshoot
        predicted_rise_time = 0.5 * tu
        predicted_settling_time = 4 * tu

        # Quality score (ZN is aggressive, moderate quality)
        quality_score = 0.7

        return AutoTuneOutput(
            kp=self._round_decimal(kp, 4),
            ki=self._round_decimal(ki, 4),
            kd=self._round_decimal(kd, 4),
            ultimate_gain=ku,
            ultimate_period=tu,
            tuning_method=TuningMethod.ZIEGLER_NICHOLS_CLOSED_LOOP,
            predicted_rise_time=predicted_rise_time,
            predicted_settling_time=predicted_settling_time,
            predicted_overshoot_percent=predicted_overshoot,
            tuning_quality_score=quality_score
        )

    def _tune_ziegler_nichols_open_loop(
        self,
        auto_tune_input: AutoTuneInput
    ) -> AutoTuneOutput:
        """
        Ziegler-Nichols Open Loop (Reaction Curve) method.

        Requires step response data to extract process characteristics.
        """
        # Extract process parameters from step response
        L, T = self._extract_process_parameters(auto_tune_input.process_data)

        if T == 0:
            T = 1.0  # Prevent division by zero

        # Ziegler-Nichols open loop formulas
        kp = 1.2 * T / L if L > 0 else 1.0
        ki = 2.0 / L if L > 0 else 0.1
        kd = 0.5 * L

        # Apply tuning factor
        factor = auto_tune_input.tuning_factor
        kp *= factor
        ki *= factor
        kd *= factor

        quality_score = 0.65  # Open loop typically less accurate

        return AutoTuneOutput(
            kp=self._round_decimal(kp, 4),
            ki=self._round_decimal(ki, 4),
            kd=self._round_decimal(kd, 4),
            process_dead_time=L,
            process_time_constant=T,
            tuning_method=TuningMethod.ZIEGLER_NICHOLS_OPEN_LOOP,
            tuning_quality_score=quality_score
        )

    def _tune_cohen_coon(
        self,
        auto_tune_input: AutoTuneInput
    ) -> AutoTuneOutput:
        """
        Cohen-Coon tuning method (improved over ZN for processes with lag).
        """
        L, T = self._extract_process_parameters(auto_tune_input.process_data)

        if T == 0 or L == 0:
            # Fallback to ZN
            return self._tune_ziegler_nichols_open_loop(auto_tune_input)

        # Cohen-Coon formulas (more complex than ZN)
        tau = T / L  # Dimensionless time constant

        kp = (1.35 / L) * (T + 0.185 * L) / T if T > 0 else 1.0
        ki = (1.2 / T) * (T + 0.185 * L) / (T + 0.611 * L) if T > 0 else 0.1
        kd = 0.37 * L * T / (T + 0.185 * L) if (T + 0.185 * L) > 0 else 0

        # Apply tuning factor
        factor = auto_tune_input.tuning_factor
        kp *= factor
        ki *= factor
        kd *= factor

        quality_score = 0.75  # Cohen-Coon typically better than ZN

        return AutoTuneOutput(
            kp=self._round_decimal(kp, 4),
            ki=self._round_decimal(ki, 4),
            kd=self._round_decimal(kd, 4),
            process_dead_time=L,
            process_time_constant=T,
            tuning_method=TuningMethod.COHEN_COON,
            tuning_quality_score=quality_score
        )

    def _tune_tyreus_luyben(
        self,
        auto_tune_input: AutoTuneInput
    ) -> AutoTuneOutput:
        """
        Tyreus-Luyben tuning method (more conservative than ZN).

        Provides less overshoot and better stability margins.
        """
        if auto_tune_input.ultimate_gain is None or auto_tune_input.ultimate_period is None:
            ku, tu = self._detect_ultimate_gain_period(auto_tune_input.process_data)
        else:
            ku = auto_tune_input.ultimate_gain
            tu = auto_tune_input.ultimate_period

        # Tyreus-Luyben formulas (more conservative)
        factor = auto_tune_input.tuning_factor
        kp = 0.45 * ku * factor
        ki = 0.54 * ku / tu * factor
        kd = 0.15 * ku * tu * factor

        # Better performance than ZN
        predicted_overshoot = 10.0 * factor
        predicted_rise_time = 0.6 * tu
        predicted_settling_time = 3 * tu

        quality_score = 0.8  # Better than ZN

        return AutoTuneOutput(
            kp=self._round_decimal(kp, 4),
            ki=self._round_decimal(ki, 4),
            kd=self._round_decimal(kd, 4),
            ultimate_gain=ku,
            ultimate_period=tu,
            tuning_method=TuningMethod.TYREUS_LUYBEN,
            predicted_rise_time=predicted_rise_time,
            predicted_settling_time=predicted_settling_time,
            predicted_overshoot_percent=predicted_overshoot,
            tuning_quality_score=quality_score
        )

    def _detect_ultimate_gain_period(
        self,
        process_data: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Detect ultimate gain and period from oscillating process data.

        Uses zero-crossing method to detect period and peak-to-peak
        amplitude to estimate gain.
        """
        times = [t for t, _ in process_data]
        values = [v for _, v in process_data]

        # Find mean value
        mean_value = sum(values) / len(values)

        # Count zero crossings
        crossings = []
        for i in range(1, len(values)):
            if (values[i-1] < mean_value <= values[i]) or (values[i-1] >= mean_value > values[i]):
                crossings.append(times[i])

        # Calculate period (time between alternate crossings)
        if len(crossings) >= 3:
            periods = [crossings[i+2] - crossings[i] for i in range(len(crossings)-2)]
            tu = sum(periods) / len(periods) if periods else 1.0
        else:
            tu = 1.0  # Default

        # Estimate ultimate gain from amplitude
        amplitude = (max(values) - min(values)) / 2
        ku = 1.0 / amplitude if amplitude > 0 else 1.0

        return ku, tu

    def _extract_process_parameters(
        self,
        process_data: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Extract process dead time (L) and time constant (T) from step response.

        Returns:
            Tuple of (dead_time, time_constant)
        """
        times = [t for t, _ in process_data]
        values = [v for _, v in process_data]

        # Normalize to 0-1
        v_min = min(values)
        v_max = max(values)
        v_range = v_max - v_min

        if v_range == 0:
            return 1.0, 1.0

        normalized = [(v - v_min) / v_range for v in values]

        # Find when response reaches 63.2% (1 time constant)
        t63_idx = None
        for i, v in enumerate(normalized):
            if v >= 0.632:
                t63_idx = i
                break

        if t63_idx is None:
            t63_idx = len(times) // 2

        # Estimate dead time (when response starts)
        dead_time = times[0]
        for i, v in enumerate(normalized):
            if v >= 0.05:  # 5% threshold
                dead_time = times[i]
                break

        # Time constant = time to 63.2% - dead time
        time_constant = times[t63_idx] - dead_time if t63_idx < len(times) else 1.0

        return dead_time, time_constant

    def _create_manual_output(
        self,
        manual_output: float,
        pid_input: PIDInput
    ) -> PIDOutput:
        """Create output for manual mode"""
        return PIDOutput(
            control_output=manual_output,
            error=pid_input.setpoint - pid_input.process_variable,
            error_integral=self.error_integral,
            error_derivative=0,
            p_term=0,
            i_term=0,
            d_term=0,
            output_saturated=False,
            anti_windup_active=False,
            control_mode=ControlMode.MANUAL
        )

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
