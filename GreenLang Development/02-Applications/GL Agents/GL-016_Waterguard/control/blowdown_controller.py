# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Blowdown Controller Module

This module implements the BlowdownController for continuous and intermittent
blowdown control in boiler water treatment systems. It manages conductivity
and silica levels through precise valve control with thermal shock prevention.

Key Features:
    - Temperature-compensated conductivity control
    - Silica-based control with conservative setpoint selection
    - Continuous and intermittent blowdown modes
    - Ramp rate limiting for thermal shock prevention
    - Valve position-to-flow calibration
    - Full audit trail with SHA-256 provenance hashing

Compliance:
    - ISA-18.2 (Alarm Management)
    - ISA-95 (Enterprise-Control Integration)
    - IEC 62443 (Industrial Cybersecurity)

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import hashlib
import logging
import threading
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class BlowdownMode(str, Enum):
    """Blowdown operating mode enumeration."""
    CONTINUOUS = "continuous"
    INTERMITTENT = "intermittent"
    COMBINED = "combined"
    MANUAL = "manual"
    DISABLED = "disabled"


class ControlVariable(str, Enum):
    """Primary control variable selection."""
    CONDUCTIVITY = "conductivity"
    SILICA = "silica"
    BOTH_CONSERVATIVE = "both_conservative"


class ValveCharacteristic(str, Enum):
    """Valve flow characteristic curves."""
    LINEAR = "linear"
    EQUAL_PERCENTAGE = "equal_percentage"
    QUICK_OPENING = "quick_opening"


class AlertPriority(str, Enum):
    """ISA-18.2 compliant alert priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DIAGNOSTIC = "diagnostic"


# =============================================================================
# Data Models - Readings
# =============================================================================

class ConductivityReading(BaseModel):
    """
    Conductivity measurement with temperature compensation.

    Temperature compensation follows standard formula:
        compensated = raw / (1 + alpha * (temp - ref_temp))
    where alpha is typically 0.02 (2% per degree C).
    """
    raw_value: float = Field(..., ge=0, description="Raw conductivity in uS/cm")
    temperature: float = Field(..., ge=0, le=200, description="Temperature in Celsius")
    reference_temperature: float = Field(default=25.0, description="Reference temp for compensation")
    temperature_coefficient: float = Field(default=0.02, ge=0.01, le=0.03, description="Temp coefficient (alpha)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")
    quality: str = Field(default="GOOD", description="Data quality indicator")

    @property
    def compensated_value(self) -> float:
        """Calculate temperature-compensated conductivity."""
        temp_factor = 1.0 + self.temperature_coefficient * (self.temperature - self.reference_temperature)
        if temp_factor <= 0:
            logger.warning(f"Invalid temp factor {temp_factor}, using raw value")
            return self.raw_value
        return self.raw_value / temp_factor

    @validator("quality")
    def validate_quality(cls, v):
        """Validate quality indicator."""
        valid_qualities = {"GOOD", "UNCERTAIN", "BAD", "NOT_AVAILABLE"}
        if v not in valid_qualities:
            raise ValueError(f"Quality must be one of {valid_qualities}")
        return v


class SilicaReading(BaseModel):
    """Silica concentration measurement."""
    value: float = Field(..., ge=0, description="Silica concentration in ppb or ppm")
    unit: str = Field(default="ppb", description="Measurement unit (ppb or ppm)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")
    quality: str = Field(default="GOOD", description="Data quality indicator")
    analyzer_status: str = Field(default="NORMAL", description="Analyzer operational status")

    @property
    def value_in_ppb(self) -> float:
        """Convert to ppb for consistent comparison."""
        if self.unit.lower() == "ppm":
            return self.value * 1000.0
        return self.value

    @validator("unit")
    def validate_unit(cls, v):
        """Validate unit."""
        if v.lower() not in {"ppb", "ppm"}:
            raise ValueError("Unit must be ppb or ppm")
        return v.lower()


# =============================================================================
# Data Models - Configuration
# =============================================================================

class BlowdownConfig(BaseModel):
    """Configuration for BlowdownController."""

    # Control variable settings
    control_variable: ControlVariable = Field(
        default=ControlVariable.BOTH_CONSERVATIVE,
        description="Primary control variable selection"
    )

    # Conductivity limits
    conductivity_setpoint: float = Field(
        default=2500.0, ge=100, le=10000,
        description="Target conductivity setpoint in uS/cm"
    )
    conductivity_high_limit: float = Field(
        default=3000.0, ge=100, le=12000,
        description="High conductivity alarm limit in uS/cm"
    )
    conductivity_high_high_limit: float = Field(
        default=3500.0, ge=100, le=15000,
        description="High-high conductivity trip limit in uS/cm"
    )
    conductivity_deadband: float = Field(
        default=50.0, ge=0, le=500,
        description="Conductivity control deadband in uS/cm"
    )

    # Silica limits
    silica_setpoint: float = Field(
        default=150.0, ge=0, le=1000,
        description="Target silica setpoint in ppb"
    )
    silica_high_limit: float = Field(
        default=200.0, ge=0, le=1500,
        description="High silica alarm limit in ppb"
    )
    silica_high_high_limit: float = Field(
        default=250.0, ge=0, le=2000,
        description="High-high silica trip limit in ppb"
    )
    silica_deadband: float = Field(
        default=10.0, ge=0, le=100,
        description="Silica control deadband in ppb"
    )

    # PI controller tuning
    kp: float = Field(default=2.0, ge=0, le=20, description="Proportional gain")
    ki: float = Field(default=0.1, ge=0, le=5, description="Integral gain")
    anti_windup_limit: float = Field(
        default=100.0, ge=0, le=200,
        description="Anti-windup limit for integral term"
    )

    # Valve settings
    min_valve_position: float = Field(
        default=0.0, ge=0, le=100,
        description="Minimum valve position %"
    )
    max_valve_position: float = Field(
        default=100.0, ge=0, le=100,
        description="Maximum valve position %"
    )
    valve_characteristic: ValveCharacteristic = Field(
        default=ValveCharacteristic.EQUAL_PERCENTAGE,
        description="Valve flow characteristic"
    )

    # Ramp rate limits
    max_ramp_rate_per_second: float = Field(
        default=1.0, ge=0.1, le=10,
        description="Maximum valve change rate per second (%/s)"
    )
    max_ramp_rate_per_minute: float = Field(
        default=20.0, ge=1, le=100,
        description="Maximum valve change rate per minute (%/min)"
    )

    # Intermittent blowdown settings
    intermittent_duration: float = Field(
        default=30.0, ge=5, le=300,
        description="Intermittent blowdown duration in seconds"
    )
    intermittent_interval: float = Field(
        default=3600.0, ge=300, le=86400,
        description="Minimum interval between intermittent blowdowns in seconds"
    )
    intermittent_valve_position: float = Field(
        default=100.0, ge=0, le=100,
        description="Valve position during intermittent blowdown %"
    )

    # Rapid rise detection
    rapid_rise_threshold: float = Field(
        default=500.0, ge=10, le=2000,
        description="Conductivity rise rate threshold for rapid response (uS/cm per minute)"
    )
    rapid_rise_response_position: float = Field(
        default=80.0, ge=0, le=100,
        description="Valve position for rapid rise response %"
    )

    # Timing
    control_interval_seconds: float = Field(
        default=1.0, ge=0.1, le=60,
        description="Control loop execution interval in seconds"
    )

    @validator("conductivity_high_limit")
    def validate_cond_high(cls, v, values):
        """Ensure high limit > setpoint."""
        if "conductivity_setpoint" in values and v <= values["conductivity_setpoint"]:
            raise ValueError("High limit must be greater than setpoint")
        return v

    @validator("conductivity_high_high_limit")
    def validate_cond_high_high(cls, v, values):
        """Ensure high-high limit > high limit."""
        if "conductivity_high_limit" in values and v <= values["conductivity_high_limit"]:
            raise ValueError("High-high limit must be greater than high limit")
        return v


# =============================================================================
# Data Models - State and Output
# =============================================================================

class BlowdownState(BaseModel):
    """Internal state of the BlowdownController."""
    mode: BlowdownMode = Field(default=BlowdownMode.DISABLED, description="Current operating mode")
    current_valve_position: float = Field(default=0.0, ge=0, le=100, description="Current valve position %")
    target_valve_position: float = Field(default=0.0, ge=0, le=100, description="Target valve position %")
    integral_term: float = Field(default=0.0, description="PI controller integral accumulator")
    last_conductivity: Optional[float] = Field(default=None, description="Last conductivity reading")
    last_silica: Optional[float] = Field(default=None, description="Last silica reading")
    last_intermittent_time: Optional[datetime] = Field(default=None, description="Last intermittent blowdown time")
    intermittent_active: bool = Field(default=False, description="Intermittent blowdown in progress")
    intermittent_end_time: Optional[datetime] = Field(default=None, description="When intermittent ends")
    rapid_rise_active: bool = Field(default=False, description="Rapid rise response active")
    total_blowdown_seconds: float = Field(default=0.0, ge=0, description="Total blowdown time today")
    last_update: datetime = Field(default_factory=datetime.now, description="Last state update time")


class BlowdownOutput(BaseModel):
    """Output from BlowdownController calculation."""
    valve_position_command: float = Field(..., ge=0, le=100, description="Commanded valve position %")
    valve_flow_rate: float = Field(..., ge=0, description="Estimated flow rate")
    control_error: float = Field(..., description="Current control error")
    control_variable_used: ControlVariable = Field(..., description="Which CV was used")
    mode: BlowdownMode = Field(..., description="Current operating mode")
    intermittent_active: bool = Field(default=False, description="Intermittent in progress")
    rapid_rise_active: bool = Field(default=False, description="Rapid rise response active")
    timestamp: datetime = Field(default_factory=datetime.now, description="Output timestamp")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    reason_code: str = Field(default="NORMAL", description="Reason for current output")


class BlowdownAlert(BaseModel):
    """Alert generated by BlowdownController."""
    priority: AlertPriority = Field(..., description="ISA-18.2 priority level")
    message: str = Field(..., description="Alert message")
    source: str = Field(default="BlowdownController", description="Alert source")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")
    value: Optional[float] = Field(default=None, description="Associated value")
    limit: Optional[float] = Field(default=None, description="Violated limit")
    acknowledged: bool = Field(default=False, description="Acknowledgement status")


# =============================================================================
# Valve Model Class
# =============================================================================

class BlowdownValveModel:
    """
    Valve position-to-flow calibration model.

    Supports multiple valve characteristics:
        - Linear: flow = position
        - Equal Percentage: flow = R^(position - 1), R typically 50
        - Quick Opening: flow = sqrt(position)

    Attributes:
        characteristic: Valve flow characteristic type
        rangeability: Valve rangeability for equal percentage (default 50)
        max_flow: Maximum flow rate at 100% position
        calibration_points: Optional calibration curve points
    """

    def __init__(
        self,
        characteristic: ValveCharacteristic = ValveCharacteristic.EQUAL_PERCENTAGE,
        rangeability: float = 50.0,
        max_flow: float = 100.0,
        calibration_points: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize valve model.

        Args:
            characteristic: Flow characteristic curve type
            rangeability: Rangeability R for equal percentage valves
            max_flow: Maximum flow at 100% position
            calibration_points: Optional list of (position, flow) tuples for custom curve
        """
        self.characteristic = characteristic
        self.rangeability = rangeability
        self.max_flow = max_flow
        self.calibration_points = calibration_points
        self._lock = threading.Lock()

        logger.info(
            f"BlowdownValveModel initialized: characteristic={characteristic.value}, "
            f"rangeability={rangeability}, max_flow={max_flow}"
        )

    def position_to_flow(self, position: float) -> float:
        """
        Convert valve position to flow rate.

        Args:
            position: Valve position in percent (0-100)

        Returns:
            Flow rate (0 to max_flow)
        """
        # Clamp position to valid range
        position = max(0.0, min(100.0, position))

        # Use calibration points if available
        if self.calibration_points and len(self.calibration_points) >= 2:
            return self._interpolate_calibration(position)

        # Normalize position to 0-1
        pos_norm = position / 100.0

        if self.characteristic == ValveCharacteristic.LINEAR:
            flow_norm = pos_norm
        elif self.characteristic == ValveCharacteristic.EQUAL_PERCENTAGE:
            # Equal percentage: flow = R^(x-1) where x is 0-1, R is rangeability
            if pos_norm <= 0:
                flow_norm = 0.0
            else:
                flow_norm = pow(self.rangeability, pos_norm - 1)
        elif self.characteristic == ValveCharacteristic.QUICK_OPENING:
            # Quick opening: flow = sqrt(position)
            flow_norm = math.sqrt(pos_norm)
        else:
            flow_norm = pos_norm

        return flow_norm * self.max_flow

    def flow_to_position(self, flow: float) -> float:
        """
        Convert flow rate to valve position (inverse calculation).

        Args:
            flow: Desired flow rate

        Returns:
            Required valve position in percent (0-100)
        """
        # Clamp flow to valid range
        flow = max(0.0, min(self.max_flow, flow))

        if self.max_flow <= 0:
            return 0.0

        # Normalize flow to 0-1
        flow_norm = flow / self.max_flow

        if self.characteristic == ValveCharacteristic.LINEAR:
            pos_norm = flow_norm
        elif self.characteristic == ValveCharacteristic.EQUAL_PERCENTAGE:
            # Inverse: position = 1 + log_R(flow)
            if flow_norm <= 0:
                pos_norm = 0.0
            else:
                pos_norm = 1.0 + math.log(flow_norm) / math.log(self.rangeability)
                pos_norm = max(0.0, pos_norm)
        elif self.characteristic == ValveCharacteristic.QUICK_OPENING:
            # Inverse: position = flow^2
            pos_norm = flow_norm * flow_norm
        else:
            pos_norm = flow_norm

        return pos_norm * 100.0

    def _interpolate_calibration(self, position: float) -> float:
        """Linear interpolation through calibration points."""
        points = sorted(self.calibration_points, key=lambda x: x[0])

        # Handle edge cases
        if position <= points[0][0]:
            return points[0][1]
        if position >= points[-1][0]:
            return points[-1][1]

        # Find bracketing points
        for i in range(len(points) - 1):
            if points[i][0] <= position <= points[i + 1][0]:
                # Linear interpolation
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                if x1 == x0:
                    return y0
                return y0 + (y1 - y0) * (position - x0) / (x1 - x0)

        return 0.0


# =============================================================================
# Ramp Rate Limiter Class
# =============================================================================

class RampRateLimiter:
    """
    Ramp rate limiter for thermal shock prevention.

    Enforces both per-second and per-minute rate limits to prevent
    thermal shock to the boiler and downstream piping.

    Attributes:
        max_rate_per_second: Maximum change rate per second (%/s)
        max_rate_per_minute: Maximum change rate per minute (%/min)
    """

    def __init__(
        self,
        max_rate_per_second: float = 1.0,
        max_rate_per_minute: float = 20.0
    ):
        """
        Initialize ramp rate limiter.

        Args:
            max_rate_per_second: Maximum rate per second (%/s)
            max_rate_per_minute: Maximum rate per minute (%/min)
        """
        self.max_rate_per_second = max_rate_per_second
        self.max_rate_per_minute = max_rate_per_minute

        # Track minute budget usage
        self._minute_history: List[Tuple[datetime, float]] = []
        self._lock = threading.Lock()

        logger.info(
            f"RampRateLimiter initialized: per_second={max_rate_per_second}%/s, "
            f"per_minute={max_rate_per_minute}%/min"
        )

    def apply_limit(
        self,
        current_position: float,
        target_position: float,
        dt_seconds: float
    ) -> float:
        """
        Apply ramp rate limiting to a position change.

        Args:
            current_position: Current valve position %
            target_position: Desired target position %
            dt_seconds: Time step in seconds

        Returns:
            Limited position that respects both rate constraints
        """
        with self._lock:
            # Calculate desired change
            desired_change = target_position - current_position

            if abs(desired_change) < 0.001:
                return target_position

            # Per-second limit
            max_change_per_second = self.max_rate_per_second * dt_seconds

            # Check minute budget
            available_minute_budget = self._get_available_minute_budget()

            # Use the more restrictive limit
            max_allowed = min(max_change_per_second, available_minute_budget)

            # Apply limit
            if abs(desired_change) <= max_allowed:
                limited_change = desired_change
            else:
                limited_change = max_allowed if desired_change > 0 else -max_allowed

            # Record the change
            self._record_change(abs(limited_change))

            new_position = current_position + limited_change

            if abs(limited_change) < abs(desired_change):
                logger.debug(
                    f"Ramp rate limited: desired={desired_change:.2f}%, "
                    f"allowed={limited_change:.2f}%"
                )

            return max(0.0, min(100.0, new_position))

    def _get_available_minute_budget(self) -> float:
        """Calculate remaining rate budget for the current minute."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=60)

        # Clean old entries
        self._minute_history = [
            (ts, change) for ts, change in self._minute_history
            if ts > cutoff
        ]

        # Sum recent changes
        used_budget = sum(change for _, change in self._minute_history)
        available = max(0.0, self.max_rate_per_minute - used_budget)

        return available

    def _record_change(self, change: float) -> None:
        """Record a position change for minute budget tracking."""
        self._minute_history.append((datetime.now(), change))

    def reset(self) -> None:
        """Reset the minute budget history."""
        with self._lock:
            self._minute_history.clear()
            logger.info("RampRateLimiter budget reset")


# =============================================================================
# Blowdown Controller Class
# =============================================================================

class BlowdownController:
    """
    Blowdown Controller for boiler water treatment.

    Implements PI control with anti-windup for continuous blowdown,
    timed intermittent blowdown cycles, and rapid rise response.
    Supports both conductivity and silica-based control with
    conservative setpoint selection.

    Features:
        - Temperature-compensated conductivity control
        - Silica-based control with conservative selection
        - PI control with anti-windup
        - Ramp rate limiting for thermal shock prevention
        - Intermittent blowdown scheduling
        - Rapid conductivity rise detection and response
        - Full audit trail with SHA-256 provenance hashing

    Example:
        >>> config = BlowdownConfig(conductivity_setpoint=2500.0)
        >>> controller = BlowdownController(config)
        >>> output = controller.calculate(conductivity_reading, silica_reading)
        >>> print(f"Valve command: {output.valve_position_command}%")

    Attributes:
        config: Controller configuration
        valve_model: Valve position-to-flow model
        ramp_limiter: Ramp rate limiter for thermal shock prevention
        state: Current controller state
    """

    def __init__(
        self,
        config: BlowdownConfig,
        valve_model: Optional[BlowdownValveModel] = None,
        alert_callback: Optional[Callable[[BlowdownAlert], None]] = None
    ):
        """
        Initialize BlowdownController.

        Args:
            config: Controller configuration
            valve_model: Optional custom valve model
            alert_callback: Optional callback for alerts
        """
        self.config = config
        self.valve_model = valve_model or BlowdownValveModel(
            characteristic=config.valve_characteristic
        )
        self.ramp_limiter = RampRateLimiter(
            max_rate_per_second=config.max_ramp_rate_per_second,
            max_rate_per_minute=config.max_ramp_rate_per_minute
        )
        self._state = BlowdownState()
        self._alert_callback = alert_callback
        self._lock = threading.RLock()
        self._last_calc_time: Optional[datetime] = None
        self._conductivity_history: List[Tuple[datetime, float]] = []

        logger.info(
            f"BlowdownController initialized: mode={config.control_variable.value}, "
            f"conductivity_sp={config.conductivity_setpoint}, silica_sp={config.silica_setpoint}"
        )

    def calculate(
        self,
        conductivity: Optional[ConductivityReading] = None,
        silica: Optional[SilicaReading] = None
    ) -> BlowdownOutput:
        """
        Calculate blowdown valve position command.

        This is the main control loop method. It evaluates conductivity
        and/or silica readings, applies PI control, and returns the
        valve position command with full provenance tracking.

        Args:
            conductivity: Current conductivity reading
            silica: Current silica reading

        Returns:
            BlowdownOutput with valve command and metadata

        Raises:
            ValueError: If required readings are missing based on control mode
        """
        with self._lock:
            now = datetime.now()

            # Calculate time step
            if self._last_calc_time:
                dt = (now - self._last_calc_time).total_seconds()
            else:
                dt = self.config.control_interval_seconds
            self._last_calc_time = now

            # Handle disabled mode
            if self._state.mode == BlowdownMode.DISABLED:
                return self._create_output(
                    valve_position=0.0,
                    error=0.0,
                    control_variable=self.config.control_variable,
                    reason_code="DISABLED"
                )

            # Check for intermittent blowdown
            if self._check_intermittent_active(now):
                return self._create_output(
                    valve_position=self.config.intermittent_valve_position,
                    error=0.0,
                    control_variable=self.config.control_variable,
                    reason_code="INTERMITTENT_ACTIVE"
                )

            # Get process values
            cond_value = conductivity.compensated_value if conductivity else None
            silica_value = silica.value_in_ppb if silica else None

            # Check data quality
            if conductivity and conductivity.quality != "GOOD":
                logger.warning(f"Conductivity quality is {conductivity.quality}")
            if silica and silica.quality != "GOOD":
                logger.warning(f"Silica quality is {silica.quality}")

            # Update history for rapid rise detection
            if cond_value is not None:
                self._conductivity_history.append((now, cond_value))
                self._trim_history()
                self._state.last_conductivity = cond_value

            if silica_value is not None:
                self._state.last_silica = silica_value

            # Check for rapid rise
            if self._check_rapid_rise():
                self._state.rapid_rise_active = True
                self._raise_alert(
                    AlertPriority.HIGH,
                    "Rapid conductivity rise detected",
                    value=cond_value
                )
                return self._create_output(
                    valve_position=self.config.rapid_rise_response_position,
                    error=0.0,
                    control_variable=ControlVariable.CONDUCTIVITY,
                    reason_code="RAPID_RISE_RESPONSE"
                )
            else:
                self._state.rapid_rise_active = False

            # Check limits and raise alerts
            self._check_limits(cond_value, silica_value)

            # Calculate setpoint and error based on control variable
            error, cv_used = self._calculate_error(cond_value, silica_value)

            # PI control calculation
            target_position = self._calculate_pi_output(error, dt)

            # Apply ramp rate limiting
            limited_position = self.ramp_limiter.apply_limit(
                self._state.current_valve_position,
                target_position,
                dt
            )

            # Update state
            self._state.current_valve_position = limited_position
            self._state.target_valve_position = target_position
            self._state.last_update = now

            return self._create_output(
                valve_position=limited_position,
                error=error,
                control_variable=cv_used,
                reason_code="NORMAL"
            )

    def _calculate_error(
        self,
        conductivity: Optional[float],
        silica: Optional[float]
    ) -> Tuple[float, ControlVariable]:
        """
        Calculate control error based on configured control variable.

        Args:
            conductivity: Compensated conductivity value
            silica: Silica value in ppb

        Returns:
            Tuple of (error, control_variable_used)
        """
        cv = self.config.control_variable

        if cv == ControlVariable.CONDUCTIVITY:
            if conductivity is None:
                return 0.0, cv
            error = conductivity - self.config.conductivity_setpoint
            return error, cv

        elif cv == ControlVariable.SILICA:
            if silica is None:
                return 0.0, cv
            error = silica - self.config.silica_setpoint
            return error, cv

        elif cv == ControlVariable.BOTH_CONSERVATIVE:
            # Use whichever requires MORE blowdown (conservative approach)
            cond_error = 0.0
            silica_error = 0.0

            if conductivity is not None:
                cond_error = (conductivity - self.config.conductivity_setpoint) / self.config.conductivity_setpoint

            if silica is not None:
                silica_error = (silica - self.config.silica_setpoint) / self.config.silica_setpoint

            # Normalize errors for comparison
            if abs(cond_error) >= abs(silica_error):
                if conductivity is not None:
                    return conductivity - self.config.conductivity_setpoint, ControlVariable.CONDUCTIVITY
                return 0.0, ControlVariable.CONDUCTIVITY
            else:
                if silica is not None:
                    return silica - self.config.silica_setpoint, ControlVariable.SILICA
                return 0.0, ControlVariable.SILICA

        return 0.0, cv

    def _calculate_pi_output(self, error: float, dt: float) -> float:
        """
        Calculate PI controller output.

        Implements PI control with anti-windup:
            output = Kp * error + Ki * integral

        Args:
            error: Current control error
            dt: Time step in seconds

        Returns:
            Target valve position (0-100%)
        """
        # Proportional term
        p_term = self.config.kp * error

        # Integral term with anti-windup
        self._state.integral_term += self.config.ki * error * dt

        # Anti-windup clamping
        if self._state.integral_term > self.config.anti_windup_limit:
            self._state.integral_term = self.config.anti_windup_limit
        elif self._state.integral_term < -self.config.anti_windup_limit:
            self._state.integral_term = -self.config.anti_windup_limit

        i_term = self._state.integral_term

        # Calculate output
        output = p_term + i_term

        # Clamp to valid range
        output = max(self.config.min_valve_position, min(self.config.max_valve_position, output))

        return output

    def _check_intermittent_active(self, now: datetime) -> bool:
        """Check if intermittent blowdown is currently active."""
        if not self._state.intermittent_active:
            return False

        if self._state.intermittent_end_time and now >= self._state.intermittent_end_time:
            # Intermittent cycle complete
            self._state.intermittent_active = False
            self._state.intermittent_end_time = None
            logger.info("Intermittent blowdown cycle complete")
            return False

        return True

    def _check_rapid_rise(self) -> bool:
        """Check for rapid conductivity rise condition."""
        if len(self._conductivity_history) < 2:
            return False

        # Get readings from last minute
        now = datetime.now()
        cutoff = now - timedelta(seconds=60)
        recent = [(ts, val) for ts, val in self._conductivity_history if ts > cutoff]

        if len(recent) < 2:
            return False

        # Calculate rate of change
        oldest = recent[0]
        newest = recent[-1]
        time_diff = (newest[0] - oldest[0]).total_seconds()

        if time_diff < 10:  # Need at least 10 seconds of data
            return False

        rate_per_minute = (newest[1] - oldest[1]) / (time_diff / 60.0)

        return rate_per_minute > self.config.rapid_rise_threshold

    def _check_limits(
        self,
        conductivity: Optional[float],
        silica: Optional[float]
    ) -> None:
        """Check process values against limits and raise alerts."""
        if conductivity is not None:
            if conductivity >= self.config.conductivity_high_high_limit:
                self._raise_alert(
                    AlertPriority.CRITICAL,
                    "Conductivity HIGH-HIGH limit exceeded",
                    value=conductivity,
                    limit=self.config.conductivity_high_high_limit
                )
            elif conductivity >= self.config.conductivity_high_limit:
                self._raise_alert(
                    AlertPriority.HIGH,
                    "Conductivity HIGH limit exceeded",
                    value=conductivity,
                    limit=self.config.conductivity_high_limit
                )

        if silica is not None:
            if silica >= self.config.silica_high_high_limit:
                self._raise_alert(
                    AlertPriority.CRITICAL,
                    "Silica HIGH-HIGH limit exceeded",
                    value=silica,
                    limit=self.config.silica_high_high_limit
                )
            elif silica >= self.config.silica_high_limit:
                self._raise_alert(
                    AlertPriority.HIGH,
                    "Silica HIGH limit exceeded",
                    value=silica,
                    limit=self.config.silica_high_limit
                )

    def _raise_alert(
        self,
        priority: AlertPriority,
        message: str,
        value: Optional[float] = None,
        limit: Optional[float] = None
    ) -> None:
        """Raise an alert via callback."""
        alert = BlowdownAlert(
            priority=priority,
            message=message,
            value=value,
            limit=limit
        )

        logger.warning(f"Alert [{priority.value}]: {message}")

        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _create_output(
        self,
        valve_position: float,
        error: float,
        control_variable: ControlVariable,
        reason_code: str
    ) -> BlowdownOutput:
        """Create BlowdownOutput with provenance hash."""
        flow_rate = self.valve_model.position_to_flow(valve_position)

        provenance_data = {
            "valve_position": valve_position,
            "error": error,
            "cv": control_variable.value,
            "mode": self._state.mode.value,
            "timestamp": datetime.now().isoformat()
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        return BlowdownOutput(
            valve_position_command=valve_position,
            valve_flow_rate=flow_rate,
            control_error=error,
            control_variable_used=control_variable,
            mode=self._state.mode,
            intermittent_active=self._state.intermittent_active,
            rapid_rise_active=self._state.rapid_rise_active,
            provenance_hash=provenance_hash,
            reason_code=reason_code
        )

    def _trim_history(self) -> None:
        """Trim conductivity history to last 5 minutes."""
        cutoff = datetime.now() - timedelta(minutes=5)
        self._conductivity_history = [
            (ts, val) for ts, val in self._conductivity_history
            if ts > cutoff
        ]

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def set_mode(self, mode: BlowdownMode) -> bool:
        """
        Set the operating mode.

        Args:
            mode: New operating mode

        Returns:
            True if mode was changed successfully
        """
        with self._lock:
            old_mode = self._state.mode
            self._state.mode = mode

            if mode == BlowdownMode.DISABLED:
                self._state.current_valve_position = 0.0
                self._state.target_valve_position = 0.0
                self._state.integral_term = 0.0

            logger.info(f"Blowdown mode changed: {old_mode.value} -> {mode.value}")
            return True

    def get_mode(self) -> BlowdownMode:
        """Get current operating mode."""
        return self._state.mode

    def trigger_intermittent_blowdown(self) -> bool:
        """
        Manually trigger an intermittent blowdown cycle.

        Returns:
            True if cycle was started, False if too soon after last cycle
        """
        with self._lock:
            now = datetime.now()

            # Check minimum interval
            if self._state.last_intermittent_time:
                elapsed = (now - self._state.last_intermittent_time).total_seconds()
                if elapsed < self.config.intermittent_interval:
                    logger.warning(
                        f"Intermittent blowdown too soon, wait {self.config.intermittent_interval - elapsed:.0f}s"
                    )
                    return False

            # Start intermittent cycle
            self._state.intermittent_active = True
            self._state.last_intermittent_time = now
            self._state.intermittent_end_time = now + timedelta(
                seconds=self.config.intermittent_duration
            )

            logger.info(
                f"Intermittent blowdown started, duration={self.config.intermittent_duration}s"
            )
            return True

    def reset_integral(self) -> None:
        """Reset the PI controller integral term."""
        with self._lock:
            self._state.integral_term = 0.0
            logger.info("PI integral term reset")

    def get_state(self) -> BlowdownState:
        """Get a copy of the current state."""
        with self._lock:
            return self._state.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get controller diagnostics.

        Returns:
            Dictionary with diagnostic information
        """
        with self._lock:
            return {
                "mode": self._state.mode.value,
                "current_valve_position": self._state.current_valve_position,
                "target_valve_position": self._state.target_valve_position,
                "integral_term": self._state.integral_term,
                "last_conductivity": self._state.last_conductivity,
                "last_silica": self._state.last_silica,
                "intermittent_active": self._state.intermittent_active,
                "rapid_rise_active": self._state.rapid_rise_active,
                "history_points": len(self._conductivity_history),
                "last_update": self._state.last_update.isoformat()
            }