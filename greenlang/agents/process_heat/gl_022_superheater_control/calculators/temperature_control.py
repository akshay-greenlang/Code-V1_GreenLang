"""
GL-022 SUPERHEATER CONTROL - Temperature Control Calculator Module

This module provides temperature control calculations including:
- PID controller implementation with anti-windup
- Cascade control logic (primary/secondary)
- Feedforward control (load-based)
- Rate limiters for thermal protection
- Model predictive control elements

All calculations are ZERO-HALLUCINATION deterministic with complete provenance tracking.

Standards Reference:
    - ISA-77.43: Fossil Fuel Power Plant Desuperheater Controls
    - ISA-5.1: Instrumentation Symbols and Identification
    - ASME PTC 4.2: Steam Generating Units

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.calculators.temperature_control import (
    ...     PIDController,
    ...     CascadeController,
    ...     FeedforwardController,
    ... )
    >>>
    >>> pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
    >>> output = pid.calculate(setpoint=850.0, process_value=860.0, dt=1.0)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import deque
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - CONTROL SYSTEM PARAMETERS
# =============================================================================

class ControlSystemConstants:
    """
    Constants for temperature control system design.

    Based on ISA-77.43 guidelines for fossil fuel power plant controls.
    """

    # PID tuning default values for superheater temperature control
    DEFAULT_KP = 2.0  # Proportional gain (%/F)
    DEFAULT_KI = 0.05  # Integral gain (%/F-sec)
    DEFAULT_KD = 10.0  # Derivative gain (%-sec/F)

    # Output limits
    DEFAULT_OUTPUT_MIN = 0.0  # Minimum output (%)
    DEFAULT_OUTPUT_MAX = 100.0  # Maximum output (%)

    # Rate limits for thermal protection
    DEFAULT_RATE_LIMIT_UP = 10.0  # %/sec increase rate
    DEFAULT_RATE_LIMIT_DOWN = 20.0  # %/sec decrease rate (faster for safety)

    # Deadband for setpoint tracking
    DEFAULT_DEADBAND_F = 2.0  # Degrees F deadband

    # Anti-windup
    ANTI_WINDUP_BACK_CALCULATION_GAIN = 0.1

    # Cascade control parameters
    CASCADE_PRIMARY_SETPOINT_RANGE_F = 50.0  # Max adjustment from master
    CASCADE_HANDOFF_THRESHOLD = 5.0  # Error threshold for cascade handoff

    # Feedforward parameters
    FEEDFORWARD_GAIN_DEFAULT = 0.8  # Default feedforward gain
    FEEDFORWARD_LAG_SECONDS = 30.0  # Typical process lag

    # MPC parameters
    MPC_PREDICTION_HORIZON = 60  # seconds
    MPC_CONTROL_HORIZON = 10  # steps
    MPC_SAMPLE_TIME = 1.0  # seconds


class ThermalProtectionConstants:
    """Constants for thermal protection calculations."""

    # Temperature rate limits (F/min)
    MAX_TEMP_RATE_NORMAL = 5.0
    MAX_TEMP_RATE_STARTUP = 15.0
    MAX_TEMP_RATE_SHUTDOWN = 10.0

    # Temperature limits
    HIGH_TEMP_ALARM_OFFSET_F = 25.0
    HIGH_TEMP_TRIP_OFFSET_F = 50.0
    LOW_TEMP_ALARM_OFFSET_F = 50.0

    # Spray valve travel limits
    SPRAY_VALVE_MAX_RATE_PCT_SEC = 5.0  # %/sec
    SPRAY_VALVE_MIN_POSITION_PCT = 0.0
    SPRAY_VALVE_MAX_POSITION_PCT = 100.0


# =============================================================================
# DATA CLASSES FOR CONTROL RESULTS
# =============================================================================

@dataclass
class PIDControlResult:
    """Result of PID controller calculation."""
    output: float
    proportional_term: float
    integral_term: float
    derivative_term: float
    error: float
    error_rate: float
    is_saturated: bool
    anti_windup_active: bool
    output_before_limits: float
    calculation_method: str
    provenance_hash: str


@dataclass
class CascadeControlResult:
    """Result of cascade controller calculation."""
    primary_output: float
    secondary_setpoint: float
    secondary_output: float
    primary_error: float
    secondary_error: float
    cascade_active: bool
    cascade_tracking: bool
    calculation_method: str
    provenance_hash: str


@dataclass
class FeedforwardResult:
    """Result of feedforward calculation."""
    feedforward_output: float
    feedforward_contribution: float
    load_change_detected: bool
    disturbance_magnitude: float
    lag_compensated: bool
    calculation_method: str
    provenance_hash: str


@dataclass
class RateLimiterResult:
    """Result of rate limiter calculation."""
    limited_value: float
    original_value: float
    was_limited: bool
    rate_of_change: float
    limit_applied: float
    direction: str  # "up", "down", "none"
    calculation_method: str
    provenance_hash: str


@dataclass
class ControllerState:
    """State of a controller for persistence."""
    integral_term: float
    previous_error: float
    previous_output: float
    previous_pv: float
    timestamp: datetime
    mode: str  # "auto", "manual", "cascade"


# =============================================================================
# PID CONTROLLER
# =============================================================================

class PIDController:
    """
    PID controller with anti-windup for superheater temperature control.

    Implements the standard PID control algorithm:
        u(t) = Kp * e(t) + Ki * integral(e(t)) + Kd * de(t)/dt

    Features:
    - Proportional, Integral, Derivative control
    - Anti-windup using back-calculation method
    - Output rate limiting
    - Derivative filtering
    - Bumpless transfer

    Formula Reference:
        Proportional: P = Kp * error
        Integral: I = Ki * integral(error * dt)
        Derivative: D = Kd * d(error)/dt
        Output: u = P + I + D

    Example:
        >>> pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
        >>> result = pid.calculate(setpoint=850.0, process_value=860.0, dt=1.0)
        >>> print(f"Output: {result.output:.1f}%")
    """

    def __init__(
        self,
        kp: float = ControlSystemConstants.DEFAULT_KP,
        ki: float = ControlSystemConstants.DEFAULT_KI,
        kd: float = ControlSystemConstants.DEFAULT_KD,
        output_min: float = ControlSystemConstants.DEFAULT_OUTPUT_MIN,
        output_max: float = ControlSystemConstants.DEFAULT_OUTPUT_MAX,
        setpoint: float = 0.0,
        derivative_filter_coeff: float = 0.1,
    ) -> None:
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            setpoint: Initial setpoint
            derivative_filter_coeff: Derivative filter coefficient (0-1)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.setpoint = setpoint
        self.derivative_filter_coeff = derivative_filter_coeff

        # Internal state
        self._integral_term = 0.0
        self._previous_error = 0.0
        self._previous_derivative = 0.0
        self._previous_output = 0.0
        self._previous_pv = None
        self._anti_windup_gain = ControlSystemConstants.ANTI_WINDUP_BACK_CALCULATION_GAIN

        self._calculation_count = 0

        logger.debug(f"PIDController initialized: Kp={kp}, Ki={ki}, Kd={kd}")

    def calculate(
        self,
        setpoint: float,
        process_value: float,
        dt: float,
        feedforward: float = 0.0,
    ) -> PIDControlResult:
        """
        Calculate PID controller output - DETERMINISTIC.

        Args:
            setpoint: Desired setpoint value
            process_value: Current process value
            dt: Time step (seconds)
            feedforward: Feedforward contribution (optional)

        Returns:
            PIDControlResult with output and component breakdown
        """
        self._calculation_count += 1
        self.setpoint = setpoint

        # Calculate error (reverse acting for temperature control)
        # Positive error = temperature too high = need more spray = positive output
        error = process_value - setpoint

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral_term += self.ki * error * dt

        # Derivative term (on PV to avoid derivative kick on SP change)
        if self._previous_pv is not None and dt > 0:
            dPV = (process_value - self._previous_pv) / dt
            # Apply derivative filtering
            d_term_raw = self.kd * dPV
            d_term = (self.derivative_filter_coeff * d_term_raw +
                      (1 - self.derivative_filter_coeff) * self._previous_derivative)
        else:
            d_term = 0.0
            dPV = 0.0

        # Store filtered derivative
        self._previous_derivative = d_term

        # Calculate output before limits
        output_before_limits = p_term + self._integral_term + d_term + feedforward

        # Apply output limits
        output = max(self.output_min, min(self.output_max, output_before_limits))

        # Check saturation
        is_saturated = output != output_before_limits

        # Anti-windup: back-calculation method
        anti_windup_active = False
        if is_saturated:
            # Prevent integral from winding up
            integral_correction = self._anti_windup_gain * (output - output_before_limits)
            self._integral_term += integral_correction
            anti_windup_active = True

        # Store state for next iteration
        self._previous_error = error
        self._previous_output = output
        self._previous_pv = process_value

        # Error rate
        error_rate = dPV

        # Provenance hash
        provenance_data = {
            "setpoint": setpoint,
            "process_value": process_value,
            "dt": dt,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "p_term": p_term,
            "i_term": self._integral_term,
            "d_term": d_term,
            "output": output,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return PIDControlResult(
            output=round(output, 2),
            proportional_term=round(p_term, 3),
            integral_term=round(self._integral_term, 3),
            derivative_term=round(d_term, 3),
            error=round(error, 2),
            error_rate=round(error_rate, 3),
            is_saturated=is_saturated,
            anti_windup_active=anti_windup_active,
            output_before_limits=round(output_before_limits, 2),
            calculation_method="pid_back_calculation_antiwindup",
            provenance_hash=provenance_hash,
        )

    def reset(self) -> None:
        """Reset controller state."""
        self._integral_term = 0.0
        self._previous_error = 0.0
        self._previous_derivative = 0.0
        self._previous_output = 0.0
        self._previous_pv = None
        logger.debug("PID controller reset")

    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """Set output limits."""
        self.output_min = min_output
        self.output_max = max_output

    def set_tuning(self, kp: float, ki: float, kd: float) -> None:
        """Update PID tuning parameters."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        logger.info(f"PID tuning updated: Kp={kp}, Ki={ki}, Kd={kd}")

    def get_state(self) -> ControllerState:
        """Get current controller state."""
        return ControllerState(
            integral_term=self._integral_term,
            previous_error=self._previous_error,
            previous_output=self._previous_output,
            previous_pv=self._previous_pv if self._previous_pv else 0.0,
            timestamp=datetime.now(timezone.utc),
            mode="auto",
        )

    def set_state(self, state: ControllerState) -> None:
        """Restore controller state."""
        self._integral_term = state.integral_term
        self._previous_error = state.previous_error
        self._previous_output = state.previous_output
        self._previous_pv = state.previous_pv


# =============================================================================
# CASCADE CONTROLLER
# =============================================================================

class CascadeController:
    """
    Cascade controller for superheater temperature control.

    Implements master-slave cascade control where:
    - Primary (Master): Final outlet temperature controller
    - Secondary (Slave): Spray water or intermediate temperature controller

    The primary controller output becomes the setpoint for the secondary controller.

    Benefits:
    - Faster disturbance rejection
    - Better control of intermediate processes
    - Improved stability

    Example:
        >>> cascade = CascadeController(
        ...     primary_kp=1.5, primary_ki=0.05,
        ...     secondary_kp=3.0, secondary_ki=0.1,
        ... )
        >>> result = cascade.calculate(
        ...     primary_setpoint=850.0,
        ...     primary_pv=855.0,
        ...     secondary_pv=870.0,
        ...     dt=1.0,
        ... )
    """

    def __init__(
        self,
        primary_kp: float = 1.5,
        primary_ki: float = 0.05,
        primary_kd: float = 5.0,
        secondary_kp: float = 3.0,
        secondary_ki: float = 0.1,
        secondary_kd: float = 1.0,
        secondary_setpoint_min: float = 750.0,
        secondary_setpoint_max: float = 950.0,
        output_min: float = 0.0,
        output_max: float = 100.0,
    ) -> None:
        """
        Initialize cascade controller.

        Args:
            primary_kp: Primary (master) proportional gain
            primary_ki: Primary integral gain
            primary_kd: Primary derivative gain
            secondary_kp: Secondary (slave) proportional gain
            secondary_ki: Secondary integral gain
            secondary_kd: Secondary derivative gain
            secondary_setpoint_min: Minimum secondary setpoint
            secondary_setpoint_max: Maximum secondary setpoint
            output_min: Minimum final output
            output_max: Maximum final output
        """
        # Primary controller (output is secondary setpoint bias)
        self.primary = PIDController(
            kp=primary_kp,
            ki=primary_ki,
            kd=primary_kd,
            output_min=-ControlSystemConstants.CASCADE_PRIMARY_SETPOINT_RANGE_F,
            output_max=ControlSystemConstants.CASCADE_PRIMARY_SETPOINT_RANGE_F,
        )

        # Secondary controller (output is final control output)
        self.secondary = PIDController(
            kp=secondary_kp,
            ki=secondary_ki,
            kd=secondary_kd,
            output_min=output_min,
            output_max=output_max,
        )

        self.secondary_setpoint_min = secondary_setpoint_min
        self.secondary_setpoint_max = secondary_setpoint_max
        self._base_secondary_setpoint = 0.0
        self._cascade_active = True

        logger.debug("CascadeController initialized")

    def calculate(
        self,
        primary_setpoint: float,
        primary_pv: float,
        secondary_pv: float,
        dt: float,
        feedforward: float = 0.0,
    ) -> CascadeControlResult:
        """
        Calculate cascade controller output - DETERMINISTIC.

        Args:
            primary_setpoint: Primary (master) setpoint
            primary_pv: Primary process value (final temperature)
            secondary_pv: Secondary process value (intermediate temp or spray)
            dt: Time step (seconds)
            feedforward: Feedforward contribution (optional)

        Returns:
            CascadeControlResult with primary and secondary outputs
        """
        # Primary controller calculates secondary setpoint adjustment
        primary_result = self.primary.calculate(
            setpoint=primary_setpoint,
            process_value=primary_pv,
            dt=dt,
        )

        # Secondary setpoint = base + primary output
        # For superheater: higher primary error (temp too high) -> lower secondary SP
        secondary_setpoint = primary_setpoint + primary_result.output

        # Clamp secondary setpoint
        secondary_setpoint = max(
            self.secondary_setpoint_min,
            min(self.secondary_setpoint_max, secondary_setpoint)
        )

        # Secondary controller calculates final output
        secondary_result = self.secondary.calculate(
            setpoint=secondary_setpoint,
            process_value=secondary_pv,
            dt=dt,
            feedforward=feedforward,
        )

        # Cascade tracking (for bumpless transfer)
        cascade_tracking = abs(primary_result.error) < ControlSystemConstants.CASCADE_HANDOFF_THRESHOLD

        # Provenance hash
        provenance_data = {
            "primary_setpoint": primary_setpoint,
            "primary_pv": primary_pv,
            "secondary_pv": secondary_pv,
            "dt": dt,
            "primary_output": primary_result.output,
            "secondary_setpoint": secondary_setpoint,
            "secondary_output": secondary_result.output,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return CascadeControlResult(
            primary_output=primary_result.output,
            secondary_setpoint=round(secondary_setpoint, 1),
            secondary_output=secondary_result.output,
            primary_error=primary_result.error,
            secondary_error=secondary_result.error,
            cascade_active=self._cascade_active,
            cascade_tracking=cascade_tracking,
            calculation_method="cascade_pid",
            provenance_hash=provenance_hash,
        )

    def set_cascade_mode(self, active: bool) -> None:
        """Enable or disable cascade mode."""
        self._cascade_active = active
        if not active:
            self.primary.reset()
        logger.info(f"Cascade mode {'enabled' if active else 'disabled'}")

    def reset(self) -> None:
        """Reset both controllers."""
        self.primary.reset()
        self.secondary.reset()


# =============================================================================
# FEEDFORWARD CONTROLLER
# =============================================================================

class FeedforwardController:
    """
    Feedforward controller for load-based disturbance rejection.

    Implements feedforward control to compensate for measurable
    disturbances before they affect the controlled variable.

    For superheater temperature control:
    - Load changes affect steam flow and temperature
    - Feedforward anticipates these changes
    - Reduces temperature deviations during load transients

    Formula:
        FF = K_ff * (Load_change / Load_nominal) * FF_bias

    Example:
        >>> ff = FeedforwardController(gain=0.8, lag_time=30.0)
        >>> result = ff.calculate(
        ...     current_load=80.0,
        ...     previous_load=75.0,
        ...     nominal_load=100.0,
        ...     dt=1.0,
        ... )
    """

    def __init__(
        self,
        gain: float = ControlSystemConstants.FEEDFORWARD_GAIN_DEFAULT,
        lag_time: float = ControlSystemConstants.FEEDFORWARD_LAG_SECONDS,
        load_deadband_pct: float = 1.0,
        max_output: float = 20.0,
    ) -> None:
        """
        Initialize feedforward controller.

        Args:
            gain: Feedforward gain
            lag_time: First-order lag time constant (seconds)
            load_deadband_pct: Load change deadband (%)
            max_output: Maximum feedforward output
        """
        self.gain = gain
        self.lag_time = lag_time
        self.load_deadband_pct = load_deadband_pct
        self.max_output = max_output

        # State for lag filter
        self._filtered_load = 0.0
        self._previous_load = 0.0

        logger.debug(f"FeedforwardController initialized: gain={gain}, lag={lag_time}s")

    def calculate(
        self,
        current_load: float,
        previous_load: float,
        nominal_load: float,
        dt: float,
    ) -> FeedforwardResult:
        """
        Calculate feedforward output - DETERMINISTIC.

        Args:
            current_load: Current plant load (%)
            previous_load: Previous load (%)
            nominal_load: Nominal (design) load (%)
            dt: Time step (seconds)

        Returns:
            FeedforwardResult with feedforward contribution
        """
        # Calculate load change rate
        load_change = current_load - previous_load
        load_change_rate = load_change / dt if dt > 0 else 0.0

        # Check if load change exceeds deadband
        load_change_detected = abs(load_change) > self.load_deadband_pct

        # Calculate feedforward based on load change rate
        if nominal_load > 0 and load_change_detected:
            # Feedforward proportional to rate of load change
            ff_raw = self.gain * (load_change / nominal_load) * 100
        else:
            ff_raw = 0.0

        # Apply first-order lag filter
        if self.lag_time > 0 and dt > 0:
            alpha = dt / (self.lag_time + dt)
            ff_filtered = alpha * ff_raw + (1 - alpha) * self._filtered_load
        else:
            ff_filtered = ff_raw

        # Store filtered value
        self._filtered_load = ff_filtered

        # Apply limits
        ff_output = max(-self.max_output, min(self.max_output, ff_filtered))

        # Disturbance magnitude
        disturbance = abs(load_change_rate) * nominal_load / 100

        # Provenance hash
        provenance_data = {
            "current_load": current_load,
            "previous_load": previous_load,
            "nominal_load": nominal_load,
            "dt": dt,
            "gain": self.gain,
            "ff_raw": ff_raw,
            "ff_filtered": ff_filtered,
            "ff_output": ff_output,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return FeedforwardResult(
            feedforward_output=round(ff_output, 2),
            feedforward_contribution=round(ff_filtered, 2),
            load_change_detected=load_change_detected,
            disturbance_magnitude=round(disturbance, 2),
            lag_compensated=self.lag_time > 0,
            calculation_method="first_order_lag",
            provenance_hash=provenance_hash,
        )

    def reset(self) -> None:
        """Reset feedforward filter state."""
        self._filtered_load = 0.0
        self._previous_load = 0.0


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Rate limiter for thermal protection.

    Limits the rate of change of control signals to protect
    superheater tubes from thermal shock.

    Formula:
        delta_max = rate_limit * dt
        If |delta| > delta_max: output = previous + sign(delta) * delta_max

    Example:
        >>> limiter = RateLimiter(rate_up=5.0, rate_down=10.0)
        >>> result = limiter.limit(
        ...     desired_value=50.0,
        ...     previous_value=40.0,
        ...     dt=1.0,
        ... )
    """

    def __init__(
        self,
        rate_up: float = ControlSystemConstants.DEFAULT_RATE_LIMIT_UP,
        rate_down: float = ControlSystemConstants.DEFAULT_RATE_LIMIT_DOWN,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            rate_up: Maximum rate of increase (units/sec)
            rate_down: Maximum rate of decrease (units/sec)
        """
        self.rate_up = rate_up
        self.rate_down = rate_down
        self._previous_value = None

        logger.debug(f"RateLimiter initialized: up={rate_up}/s, down={rate_down}/s")

    def limit(
        self,
        desired_value: float,
        previous_value: Optional[float] = None,
        dt: float = 1.0,
    ) -> RateLimiterResult:
        """
        Apply rate limiting to a value - DETERMINISTIC.

        Args:
            desired_value: Desired (unlimited) value
            previous_value: Previous output (uses internal state if None)
            dt: Time step (seconds)

        Returns:
            RateLimiterResult with limited value
        """
        # Use provided previous value or internal state
        if previous_value is not None:
            prev = previous_value
        elif self._previous_value is not None:
            prev = self._previous_value
        else:
            # First call - no limiting
            self._previous_value = desired_value
            return self._create_unlimited_result(desired_value)

        # Calculate change
        delta = desired_value - prev

        # Determine direction and limit
        if delta > 0:
            direction = "up"
            max_delta = self.rate_up * dt
            limit_applied = self.rate_up
        elif delta < 0:
            direction = "down"
            max_delta = self.rate_down * dt
            limit_applied = self.rate_down
        else:
            direction = "none"
            max_delta = 0.0
            limit_applied = 0.0

        # Apply rate limit
        if abs(delta) > max_delta and max_delta > 0:
            limited_value = prev + math.copysign(max_delta, delta)
            was_limited = True
        else:
            limited_value = desired_value
            was_limited = False

        # Store for next iteration
        self._previous_value = limited_value

        # Calculate actual rate
        rate_of_change = (limited_value - prev) / dt if dt > 0 else 0.0

        # Provenance hash
        provenance_data = {
            "desired_value": desired_value,
            "previous_value": prev,
            "dt": dt,
            "rate_up": self.rate_up,
            "rate_down": self.rate_down,
            "limited_value": limited_value,
            "was_limited": was_limited,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return RateLimiterResult(
            limited_value=round(limited_value, 2),
            original_value=round(desired_value, 2),
            was_limited=was_limited,
            rate_of_change=round(rate_of_change, 3),
            limit_applied=limit_applied,
            direction=direction,
            calculation_method="symmetric_rate_limit",
            provenance_hash=provenance_hash,
        )

    def _create_unlimited_result(self, value: float) -> RateLimiterResult:
        """Create result for first call (no limiting)."""
        return RateLimiterResult(
            limited_value=value,
            original_value=value,
            was_limited=False,
            rate_of_change=0.0,
            limit_applied=0.0,
            direction="none",
            calculation_method="first_call",
            provenance_hash=hashlib.sha256(str(value).encode()).hexdigest(),
        )

    def reset(self) -> None:
        """Reset rate limiter state."""
        self._previous_value = None


# =============================================================================
# TEMPERATURE RATE LIMITER
# =============================================================================

class TemperatureRateLimiter:
    """
    Temperature rate limiter for thermal stress protection.

    Specifically designed for superheater temperature control to prevent
    excessive thermal stresses during transients.

    Limits both:
    - Temperature setpoint rate of change
    - Actual temperature rate (via control action adjustment)

    Example:
        >>> limiter = TemperatureRateLimiter(
        ...     max_rate_normal=5.0,  # F/min
        ...     max_rate_startup=15.0,
        ... )
        >>> result = limiter.limit_setpoint(
        ...     desired_setpoint=900.0,
        ...     current_setpoint=850.0,
        ...     dt=60.0,  # 1 minute
        ...     operation_mode="normal",
        ... )
    """

    def __init__(
        self,
        max_rate_normal: float = ThermalProtectionConstants.MAX_TEMP_RATE_NORMAL,
        max_rate_startup: float = ThermalProtectionConstants.MAX_TEMP_RATE_STARTUP,
        max_rate_shutdown: float = ThermalProtectionConstants.MAX_TEMP_RATE_SHUTDOWN,
    ) -> None:
        """
        Initialize temperature rate limiter.

        Args:
            max_rate_normal: Maximum rate during normal operation (F/min)
            max_rate_startup: Maximum rate during startup (F/min)
            max_rate_shutdown: Maximum rate during shutdown (F/min)
        """
        self.max_rate_normal = max_rate_normal
        self.max_rate_startup = max_rate_startup
        self.max_rate_shutdown = max_rate_shutdown

        self._previous_setpoint = None
        self._previous_temperature = None

        logger.debug(f"TemperatureRateLimiter initialized")

    def limit_setpoint(
        self,
        desired_setpoint: float,
        current_setpoint: float,
        dt: float,
        operation_mode: str = "normal",
    ) -> RateLimiterResult:
        """
        Limit temperature setpoint rate of change - DETERMINISTIC.

        Args:
            desired_setpoint: Desired setpoint (F)
            current_setpoint: Current setpoint (F)
            dt: Time step (seconds)
            operation_mode: "normal", "startup", or "shutdown"

        Returns:
            RateLimiterResult with limited setpoint
        """
        # Select rate limit based on operation mode
        if operation_mode == "startup":
            rate_limit = self.max_rate_startup
        elif operation_mode == "shutdown":
            rate_limit = self.max_rate_shutdown
        else:
            rate_limit = self.max_rate_normal

        # Convert to per-second
        rate_limit_per_sec = rate_limit / 60.0

        # Calculate maximum change
        max_change = rate_limit_per_sec * dt
        delta = desired_setpoint - current_setpoint

        # Apply limit
        if abs(delta) > max_change:
            limited_setpoint = current_setpoint + math.copysign(max_change, delta)
            was_limited = True
        else:
            limited_setpoint = desired_setpoint
            was_limited = False

        # Determine direction
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
        else:
            direction = "none"

        # Actual rate (F/min)
        actual_rate = (limited_setpoint - current_setpoint) / dt * 60 if dt > 0 else 0

        # Provenance hash
        provenance_data = {
            "desired_setpoint": desired_setpoint,
            "current_setpoint": current_setpoint,
            "dt": dt,
            "operation_mode": operation_mode,
            "rate_limit": rate_limit,
            "limited_setpoint": limited_setpoint,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return RateLimiterResult(
            limited_value=round(limited_setpoint, 1),
            original_value=round(desired_setpoint, 1),
            was_limited=was_limited,
            rate_of_change=round(actual_rate, 2),
            limit_applied=rate_limit,
            direction=direction,
            calculation_method="temperature_rate_limit",
            provenance_hash=provenance_hash,
        )

    def check_temperature_rate(
        self,
        current_temperature: float,
        previous_temperature: float,
        dt: float,
        operation_mode: str = "normal",
    ) -> Tuple[float, bool, str]:
        """
        Check if temperature rate of change exceeds limits - DETERMINISTIC.

        Args:
            current_temperature: Current temperature (F)
            previous_temperature: Previous temperature (F)
            dt: Time step (seconds)
            operation_mode: Operation mode

        Returns:
            Tuple of (actual_rate_f_min, exceeds_limit, warning_message)
        """
        # Calculate actual rate (F/min)
        if dt > 0:
            actual_rate = (current_temperature - previous_temperature) / dt * 60
        else:
            actual_rate = 0.0

        # Get limit for mode
        if operation_mode == "startup":
            limit = self.max_rate_startup
        elif operation_mode == "shutdown":
            limit = self.max_rate_shutdown
        else:
            limit = self.max_rate_normal

        # Check if exceeded
        exceeds_limit = abs(actual_rate) > limit

        if exceeds_limit:
            warning = (
                f"Temperature rate {abs(actual_rate):.1f} F/min exceeds "
                f"{operation_mode} limit of {limit:.1f} F/min"
            )
        else:
            warning = ""

        return round(actual_rate, 2), exceeds_limit, warning


# =============================================================================
# MODEL PREDICTIVE CONTROL ELEMENTS
# =============================================================================

class SimpleMPC:
    """
    Simplified Model Predictive Control elements for superheater control.

    Implements basic MPC concepts:
    - Prediction horizon
    - Control horizon
    - Constraint handling
    - Reference trajectory tracking

    This is a simplified implementation suitable for real-time control.
    For full MPC, consider using specialized libraries.

    Example:
        >>> mpc = SimpleMPC(
        ...     prediction_horizon=60,
        ...     control_horizon=10,
        ...     model_gain=0.5,
        ...     model_time_constant=30.0,
        ... )
        >>> output = mpc.calculate(
        ...     setpoint=850.0,
        ...     current_pv=860.0,
        ...     current_output=45.0,
        ...     dt=1.0,
        ... )
    """

    def __init__(
        self,
        prediction_horizon: int = ControlSystemConstants.MPC_PREDICTION_HORIZON,
        control_horizon: int = ControlSystemConstants.MPC_CONTROL_HORIZON,
        model_gain: float = 0.5,
        model_time_constant: float = 30.0,
        output_min: float = 0.0,
        output_max: float = 100.0,
        output_rate_limit: float = 5.0,
    ) -> None:
        """
        Initialize simplified MPC.

        Args:
            prediction_horizon: Prediction horizon (seconds)
            control_horizon: Control horizon (steps)
            model_gain: Process model gain
            model_time_constant: Process time constant (seconds)
            output_min: Minimum output
            output_max: Maximum output
            output_rate_limit: Maximum output rate (%/sec)
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.model_gain = model_gain
        self.model_time_constant = model_time_constant
        self.output_min = output_min
        self.output_max = output_max
        self.output_rate_limit = output_rate_limit

        self._previous_output = 0.0

        logger.debug("SimpleMPC initialized")

    def calculate(
        self,
        setpoint: float,
        current_pv: float,
        current_output: float,
        dt: float,
    ) -> Dict[str, Any]:
        """
        Calculate MPC output - DETERMINISTIC.

        Uses simplified one-step ahead prediction with constraints.

        Args:
            setpoint: Target setpoint
            current_pv: Current process value
            current_output: Current control output
            dt: Time step (seconds)

        Returns:
            Dictionary with MPC output and predictions
        """
        # Predict future PV based on simple first-order model
        # delta_PV = (K * u - PV) * dt / tau
        predictions = []
        pv_pred = current_pv

        for i in range(self.control_horizon):
            # Simple first-order dynamics
            delta_pv = (self.model_gain * current_output - pv_pred) * dt / self.model_time_constant
            pv_pred += delta_pv
            predictions.append(round(pv_pred, 1))

        # Calculate optimal output to minimize predicted error
        # Simplified: use proportional correction based on predicted error
        predicted_error = setpoint - predictions[-1] if predictions else setpoint - current_pv

        # Optimal output adjustment
        optimal_delta = predicted_error / (self.model_gain * self.control_horizon) if self.model_gain > 0 else 0

        # Apply constraints
        desired_output = current_output + optimal_delta

        # Rate limit
        max_change = self.output_rate_limit * dt
        delta_output = desired_output - self._previous_output
        if abs(delta_output) > max_change:
            desired_output = self._previous_output + math.copysign(max_change, delta_output)

        # Output limits
        output = max(self.output_min, min(self.output_max, desired_output))

        self._previous_output = output

        # Provenance hash
        provenance_data = {
            "setpoint": setpoint,
            "current_pv": current_pv,
            "current_output": current_output,
            "dt": dt,
            "predictions": predictions,
            "output": output,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "output": round(output, 2),
            "predictions": predictions,
            "predicted_error": round(predicted_error, 2),
            "optimal_delta": round(optimal_delta, 2),
            "calculation_method": "simplified_mpc",
            "provenance_hash": provenance_hash,
        }


# =============================================================================
# INTEGRATED SUPERHEATER TEMPERATURE CONTROLLER
# =============================================================================

class SuperheaterTemperatureController:
    """
    Integrated superheater temperature controller combining:
    - Cascade PID control (primary/secondary)
    - Feedforward control (load-based)
    - Rate limiting (thermal protection)
    - Anti-windup

    This is the main controller class for GL-022 Superheater Control Agent.

    Example:
        >>> controller = SuperheaterTemperatureController(
        ...     primary_setpoint=850.0,
        ...     secondary_setpoint_min=750.0,
        ...     secondary_setpoint_max=950.0,
        ... )
        >>> result = controller.calculate(
        ...     primary_pv=855.0,
        ...     secondary_pv=870.0,
        ...     load_pct=80.0,
        ...     dt=1.0,
        ... )
    """

    def __init__(
        self,
        primary_setpoint: float = 850.0,
        secondary_setpoint_min: float = 750.0,
        secondary_setpoint_max: float = 950.0,
        use_feedforward: bool = True,
        use_cascade: bool = True,
    ) -> None:
        """
        Initialize integrated temperature controller.

        Args:
            primary_setpoint: Primary temperature setpoint (F)
            secondary_setpoint_min: Minimum secondary setpoint (F)
            secondary_setpoint_max: Maximum secondary setpoint (F)
            use_feedforward: Enable feedforward control
            use_cascade: Enable cascade control
        """
        self.primary_setpoint = primary_setpoint
        self.use_feedforward = use_feedforward
        self.use_cascade = use_cascade

        # Cascade controller
        self.cascade = CascadeController(
            secondary_setpoint_min=secondary_setpoint_min,
            secondary_setpoint_max=secondary_setpoint_max,
        )

        # Single PID for non-cascade mode
        self.pid = PIDController()

        # Feedforward controller
        self.feedforward = FeedforwardController()

        # Rate limiters
        self.output_rate_limiter = RateLimiter(
            rate_up=ControlSystemConstants.DEFAULT_RATE_LIMIT_UP,
            rate_down=ControlSystemConstants.DEFAULT_RATE_LIMIT_DOWN,
        )
        self.setpoint_rate_limiter = TemperatureRateLimiter()

        # State tracking
        self._previous_load = 100.0
        self._mode = "auto"

        logger.info("SuperheaterTemperatureController initialized")

    def calculate(
        self,
        primary_pv: float,
        secondary_pv: float,
        load_pct: float,
        dt: float,
        setpoint_override: Optional[float] = None,
        operation_mode: str = "normal",
    ) -> Dict[str, Any]:
        """
        Calculate controller output - DETERMINISTIC.

        Args:
            primary_pv: Primary (final) temperature (F)
            secondary_pv: Secondary temperature (F)
            load_pct: Plant load (%)
            dt: Time step (seconds)
            setpoint_override: Override primary setpoint (optional)
            operation_mode: "normal", "startup", or "shutdown"

        Returns:
            Dictionary with control output and diagnostics
        """
        # Use override setpoint if provided
        setpoint = setpoint_override if setpoint_override is not None else self.primary_setpoint

        # Apply setpoint rate limiting
        sp_result = self.setpoint_rate_limiter.limit_setpoint(
            desired_setpoint=setpoint,
            current_setpoint=self.primary_setpoint,
            dt=dt,
            operation_mode=operation_mode,
        )
        effective_setpoint = sp_result.limited_value

        # Calculate feedforward
        ff_result = None
        ff_output = 0.0
        if self.use_feedforward:
            ff_result = self.feedforward.calculate(
                current_load=load_pct,
                previous_load=self._previous_load,
                nominal_load=100.0,
                dt=dt,
            )
            ff_output = ff_result.feedforward_output

        # Calculate control output
        if self.use_cascade:
            cascade_result = self.cascade.calculate(
                primary_setpoint=effective_setpoint,
                primary_pv=primary_pv,
                secondary_pv=secondary_pv,
                dt=dt,
                feedforward=ff_output,
            )
            raw_output = cascade_result.secondary_output
            control_result = cascade_result
        else:
            pid_result = self.pid.calculate(
                setpoint=effective_setpoint,
                process_value=primary_pv,
                dt=dt,
                feedforward=ff_output,
            )
            raw_output = pid_result.output
            control_result = pid_result

        # Apply output rate limiting
        rate_result = self.output_rate_limiter.limit(
            desired_value=raw_output,
            dt=dt,
        )
        final_output = rate_result.limited_value

        # Update state
        self._previous_load = load_pct

        # Provenance hash
        provenance_data = {
            "primary_pv": primary_pv,
            "secondary_pv": secondary_pv,
            "setpoint": setpoint,
            "effective_setpoint": effective_setpoint,
            "load_pct": load_pct,
            "dt": dt,
            "ff_output": ff_output,
            "raw_output": raw_output,
            "final_output": final_output,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "output": final_output,
            "setpoint": effective_setpoint,
            "error": round(primary_pv - effective_setpoint, 2),
            "feedforward_output": round(ff_output, 2),
            "raw_output": round(raw_output, 2),
            "output_rate_limited": rate_result.was_limited,
            "setpoint_rate_limited": sp_result.was_limited,
            "mode": self._mode,
            "cascade_active": self.use_cascade,
            "feedforward_active": self.use_feedforward,
            "calculation_method": "integrated_cascade_ff",
            "provenance_hash": provenance_hash,
        }

    def set_setpoint(self, setpoint: float) -> None:
        """Set primary temperature setpoint."""
        self.primary_setpoint = setpoint
        logger.info(f"Setpoint changed to {setpoint}F")

    def reset(self) -> None:
        """Reset all controllers."""
        self.cascade.reset()
        self.pid.reset()
        self.feedforward.reset()
        self.output_rate_limiter.reset()
        logger.info("Controller reset")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_pid_controller(
    kp: float = ControlSystemConstants.DEFAULT_KP,
    ki: float = ControlSystemConstants.DEFAULT_KI,
    kd: float = ControlSystemConstants.DEFAULT_KD,
) -> PIDController:
    """Factory function to create PIDController."""
    return PIDController(kp=kp, ki=ki, kd=kd)


def create_cascade_controller(
    primary_kp: float = 1.5,
    primary_ki: float = 0.05,
    secondary_kp: float = 3.0,
    secondary_ki: float = 0.1,
) -> CascadeController:
    """Factory function to create CascadeController."""
    return CascadeController(
        primary_kp=primary_kp,
        primary_ki=primary_ki,
        secondary_kp=secondary_kp,
        secondary_ki=secondary_ki,
    )


def create_feedforward_controller(
    gain: float = ControlSystemConstants.FEEDFORWARD_GAIN_DEFAULT,
    lag_time: float = ControlSystemConstants.FEEDFORWARD_LAG_SECONDS,
) -> FeedforwardController:
    """Factory function to create FeedforwardController."""
    return FeedforwardController(gain=gain, lag_time=lag_time)


def create_superheater_temperature_controller(
    primary_setpoint: float = 850.0,
    use_feedforward: bool = True,
    use_cascade: bool = True,
) -> SuperheaterTemperatureController:
    """Factory function to create SuperheaterTemperatureController."""
    return SuperheaterTemperatureController(
        primary_setpoint=primary_setpoint,
        use_feedforward=use_feedforward,
        use_cascade=use_cascade,
    )
