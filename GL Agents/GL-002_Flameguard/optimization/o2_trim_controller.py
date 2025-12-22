"""
GL-002 FLAMEGUARD - O2 Trim Controller

Provides advanced O2 trim control with:
- Adaptive PID tuning
- CO cross-limiting
- Load-based setpoint curves
- Combustion air temperature compensation
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class TrimSetpoint:
    """O2 trim setpoint with metadata."""

    o2_setpoint: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    load_percent: float = 75.0
    trim_output: float = 0.0  # -100 to +100%
    co_override_active: bool = False
    confidence: float = 1.0
    source: str = "curve"  # curve, ai, operator


@dataclass
class COBreakthroughEvent:
    """CO breakthrough detection event."""

    timestamp: datetime
    co_ppm: float
    threshold_ppm: float
    duration_s: float
    response_action: str
    o2_adjustment: float


class PIDController:
    """
    PID controller for O2 trim.

    Implements:
    - Proportional-Integral-Derivative control
    - Anti-windup
    - Bumpless transfer
    - Output limiting
    """

    def __init__(
        self,
        kp: float = 2.0,
        ki: float = 0.008,
        kd: float = 0.0,
        output_min: float = -10.0,
        output_max: float = 10.0,
        deadband: float = 0.2,
    ) -> None:
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain (1/Ti in seconds)
            kd: Derivative gain
            output_min: Minimum output
            output_max: Maximum output
            deadband: Error deadband
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.deadband = deadband

        # State
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None
        self._output = 0.0

        # Mode
        self._auto = True
        self._manual_output = 0.0

    def compute(
        self,
        setpoint: float,
        process_value: float,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Compute PID output.

        Args:
            setpoint: Target value
            process_value: Current value
            timestamp: Current time (seconds)

        Returns:
            Controller output
        """
        if not self._auto:
            return self._manual_output

        if timestamp is None:
            timestamp = time.time()

        error = setpoint - process_value

        # Apply deadband
        if abs(error) < self.deadband:
            error = 0.0

        # Calculate dt
        if self._prev_time is not None:
            dt = timestamp - self._prev_time
        else:
            dt = 1.0

        dt = max(0.001, min(60.0, dt))  # Bound dt

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        i_term = self.ki * self._integral

        # Derivative term (on error)
        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0

        # Total output
        output = p_term + i_term + d_term

        # Output limiting
        if output > self.output_max:
            output = self.output_max
            # Anti-windup: back-calculate integral
            self._integral = (output - p_term - d_term) / self.ki if self.ki != 0 else 0
        elif output < self.output_min:
            output = self.output_min
            self._integral = (output - p_term - d_term) / self.ki if self.ki != 0 else 0

        # Update state
        self._prev_error = error
        self._prev_time = timestamp
        self._output = output

        return output

    def set_manual(self, output: float) -> None:
        """Switch to manual mode with specified output."""
        self._auto = False
        self._manual_output = max(self.output_min, min(self.output_max, output))
        self._output = self._manual_output

    def set_auto(self, bumpless: bool = True) -> None:
        """Switch to auto mode."""
        if bumpless and not self._auto:
            # Bumpless transfer: set integral to match current output
            self._integral = self._output / self.ki if self.ki != 0 else 0
        self._auto = True

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._output = 0.0

    @property
    def is_auto(self) -> bool:
        """Check if controller is in auto mode."""
        return self._auto

    @property
    def output(self) -> float:
        """Get current output."""
        return self._output

    def get_state(self) -> Dict[str, Any]:
        """Get controller state."""
        return {
            "auto": self._auto,
            "output": self._output,
            "integral": self._integral,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
        }


class O2TrimController:
    """
    Advanced O2 trim controller with CO cross-limiting.

    Features:
    - Load-based O2 setpoint curves
    - Adaptive gain scheduling
    - CO cross-limiting override
    - Air temperature compensation
    - Combustion quality monitoring
    """

    def __init__(
        self,
        boiler_id: str,
        o2_setpoint_curve: Optional[Dict[float, float]] = None,
        kp: float = 2.0,
        ki: float = 0.008,
        kd: float = 0.0,
        output_limit: float = 10.0,
        co_limit_ppm: float = 400.0,
        co_response_gain: float = 0.5,
    ) -> None:
        """
        Initialize O2 trim controller.

        Args:
            boiler_id: Boiler identifier
            o2_setpoint_curve: Load vs O2 setpoint curve
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limit: Maximum trim output (%)
            co_limit_ppm: CO limit for cross-limiting
            co_response_gain: CO response gain
        """
        self.boiler_id = boiler_id
        self.co_limit_ppm = co_limit_ppm
        self.co_response_gain = co_response_gain

        # Default O2 setpoint curve
        if o2_setpoint_curve is None:
            o2_setpoint_curve = {
                0.25: 5.0,
                0.50: 3.5,
                0.75: 3.0,
                1.00: 2.5,
            }
        self.o2_setpoint_curve = o2_setpoint_curve

        # PID controller
        self._pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            output_min=-output_limit,
            output_max=output_limit,
        )

        # State
        self._current_setpoint = 3.0
        self._current_o2 = 3.0
        self._current_co = 0.0
        self._current_load = 75.0
        self._trim_output = 0.0
        self._co_override_active = False
        self._co_override_amount = 0.0

        # CO breakthrough tracking
        self._co_high_start: Optional[float] = None
        self._co_breakthrough_events: List[COBreakthroughEvent] = []

        # Air temperature compensation
        self._reference_air_temp = 80.0  # °F
        self._current_air_temp = 80.0

        # History
        self._setpoint_history: List[TrimSetpoint] = []
        self._max_history = 1000

        logger.info(f"O2TrimController initialized for {boiler_id}")

    def compute(
        self,
        o2_measured: float,
        co_measured: float,
        load_percent: float,
        air_temp: Optional[float] = None,
    ) -> TrimSetpoint:
        """
        Compute O2 trim output.

        Args:
            o2_measured: Measured O2 (%)
            co_measured: Measured CO (ppm)
            load_percent: Current load (%)
            air_temp: Combustion air temperature (°F)

        Returns:
            TrimSetpoint with output and metadata
        """
        # Update state
        self._current_o2 = o2_measured
        self._current_co = co_measured
        self._current_load = load_percent
        if air_temp is not None:
            self._current_air_temp = air_temp

        # Get base setpoint from curve
        base_setpoint = self._interpolate_setpoint(load_percent)

        # Apply air temperature compensation
        temp_compensation = self._calculate_temp_compensation()
        compensated_setpoint = base_setpoint + temp_compensation

        # Check CO cross-limiting
        co_adjustment, co_override = self._check_co_crosslimit(co_measured)
        final_setpoint = compensated_setpoint + co_adjustment

        # Bound setpoint
        final_setpoint = max(1.5, min(8.0, final_setpoint))
        self._current_setpoint = final_setpoint

        # Compute PID output
        timestamp = time.time()
        trim_output = self._pid.compute(final_setpoint, o2_measured, timestamp)
        self._trim_output = trim_output
        self._co_override_active = co_override
        self._co_override_amount = co_adjustment

        # Create setpoint record
        setpoint = TrimSetpoint(
            o2_setpoint=final_setpoint,
            load_percent=load_percent,
            trim_output=trim_output,
            co_override_active=co_override,
            confidence=0.95 if not co_override else 0.85,
            source="curve" if not co_override else "co_override",
        )

        # Store history
        self._setpoint_history.append(setpoint)
        if len(self._setpoint_history) > self._max_history:
            self._setpoint_history = self._setpoint_history[-self._max_history:]

        return setpoint

    def _interpolate_setpoint(self, load_percent: float) -> float:
        """Interpolate O2 setpoint from load curve."""
        load_fraction = load_percent / 100.0
        loads = sorted(self.o2_setpoint_curve.keys())

        if load_fraction <= loads[0]:
            return self.o2_setpoint_curve[loads[0]]
        if load_fraction >= loads[-1]:
            return self.o2_setpoint_curve[loads[-1]]

        for i in range(len(loads) - 1):
            if loads[i] <= load_fraction <= loads[i + 1]:
                t = (load_fraction - loads[i]) / (loads[i + 1] - loads[i])
                return (
                    self.o2_setpoint_curve[loads[i]] +
                    t * (self.o2_setpoint_curve[loads[i + 1]] -
                         self.o2_setpoint_curve[loads[i]])
                )

        return 3.0  # Default

    def _calculate_temp_compensation(self) -> float:
        """
        Calculate air temperature compensation.

        Warmer air = less dense = need more O2 setpoint.
        """
        temp_diff = self._current_air_temp - self._reference_air_temp
        # ~0.01% O2 adjustment per °F
        return temp_diff * 0.01

    def _check_co_crosslimit(self, co_ppm: float) -> Tuple[float, bool]:
        """
        Check CO cross-limiting condition.

        If CO is high, increase O2 setpoint.

        Args:
            co_ppm: Current CO reading

        Returns:
            Tuple of (O2 adjustment, override active)
        """
        timestamp = time.time()

        if co_ppm > self.co_limit_ppm:
            # High CO - need more air
            if self._co_high_start is None:
                self._co_high_start = timestamp

            duration = timestamp - self._co_high_start
            exceedance = co_ppm - self.co_limit_ppm

            # Progressive response
            adjustment = min(2.0, exceedance / 200 * self.co_response_gain)

            # Check for breakthrough event
            if duration > 30:  # 30 seconds
                event = COBreakthroughEvent(
                    timestamp=datetime.now(timezone.utc),
                    co_ppm=co_ppm,
                    threshold_ppm=self.co_limit_ppm,
                    duration_s=duration,
                    response_action="increase_air",
                    o2_adjustment=adjustment,
                )
                self._co_breakthrough_events.append(event)
                logger.warning(
                    f"CO breakthrough for {self.boiler_id}: "
                    f"{co_ppm:.0f} ppm for {duration:.0f}s"
                )

            return adjustment, True
        else:
            # CO is OK
            self._co_high_start = None
            return 0.0, False

    def set_manual_output(self, output: float) -> None:
        """Set controller to manual mode."""
        self._pid.set_manual(output)
        logger.info(f"O2 trim {self.boiler_id} switched to manual: {output:.1f}%")

    def set_auto(self) -> None:
        """Switch to auto mode."""
        self._pid.set_auto(bumpless=True)
        logger.info(f"O2 trim {self.boiler_id} switched to auto")

    def update_tuning(
        self,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> None:
        """Update PID tuning parameters."""
        if kp is not None:
            self._pid.kp = kp
        if ki is not None:
            self._pid.ki = ki
        if kd is not None:
            self._pid.kd = kd
        logger.info(
            f"O2 trim {self.boiler_id} tuning updated: "
            f"Kp={self._pid.kp}, Ki={self._pid.ki}, Kd={self._pid.kd}"
        )

    def update_setpoint_curve(
        self,
        curve: Dict[float, float],
    ) -> None:
        """Update O2 setpoint curve."""
        self.o2_setpoint_curve = curve
        logger.info(f"O2 trim {self.boiler_id} setpoint curve updated")

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "boiler_id": self.boiler_id,
            "mode": "auto" if self._pid.is_auto else "manual",
            "current_setpoint": self._current_setpoint,
            "current_o2": self._current_o2,
            "current_co": self._current_co,
            "current_load": self._current_load,
            "trim_output": self._trim_output,
            "co_override_active": self._co_override_active,
            "co_override_amount": self._co_override_amount,
            "pid_state": self._pid.get_state(),
            "breakthrough_events": len(self._co_breakthrough_events),
        }

    def get_breakthrough_events(
        self,
        limit: int = 10,
    ) -> List[COBreakthroughEvent]:
        """Get recent CO breakthrough events."""
        return self._co_breakthrough_events[-limit:]

    def reset(self) -> None:
        """Reset controller state."""
        self._pid.reset()
        self._co_high_start = None
        logger.info(f"O2 trim {self.boiler_id} reset")
