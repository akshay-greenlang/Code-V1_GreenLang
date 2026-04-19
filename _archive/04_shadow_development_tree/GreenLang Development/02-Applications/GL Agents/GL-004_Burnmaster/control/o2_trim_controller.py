"""
GL-004 BURNMASTER - O2 Trim Controller

Stack O2 trim control for fine-tuning combustion air.
Provides bias adjustments to air-fuel ratio based on stack O2 measurements.

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrimMode(str, Enum):
    """O2 trim controller modes."""
    OFF = "off"                 # No trimming
    MANUAL = "manual"           # Manual bias
    AUTO = "auto"               # Automatic trimming
    TRACKING = "tracking"       # Tracking (no output)


class TrimStatus(str, Enum):
    """O2 trim controller status."""
    ACTIVE = "active"
    SATURATED_HIGH = "saturated_high"
    SATURATED_LOW = "saturated_low"
    ALARM = "alarm"
    INHIBITED = "inhibited"


class O2AnalyzerQuality(str, Enum):
    """O2 analyzer data quality."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    CALIBRATING = "calibrating"


class TrimOutput(BaseModel):
    """O2 trim controller output."""
    output_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    unit_id: str = Field(..., description="Combustion unit ID")

    # Process values
    measured_o2: float = Field(..., description="Measured O2 (%)")
    o2_setpoint: float = Field(..., description="O2 setpoint (%)")
    error: float = Field(..., description="Error (SP - PV)")

    # Controller output
    trim_bias: float = Field(..., description="Trim bias output (%)")
    trim_bias_change: float = Field(default=0.0, description="Change from last output")

    # Status
    mode: TrimMode = Field(default=TrimMode.AUTO)
    status: TrimStatus = Field(default=TrimStatus.ACTIVE)
    analyzer_quality: O2AnalyzerQuality = Field(default=O2AnalyzerQuality.GOOD)

    # Internals (for diagnostics)
    proportional_term: float = Field(default=0.0)
    integral_term: float = Field(default=0.0)
    derivative_term: float = Field(default=0.0)

    # Constraints
    output_limited: bool = Field(default=False)
    limiting_factor: Optional[str] = Field(None)


class TrimConfig(BaseModel):
    """O2 trim controller configuration."""
    unit_id: str = Field(..., description="Combustion unit ID")

    # Setpoint
    o2_setpoint: float = Field(default=3.0, ge=0, le=21)
    setpoint_source: str = Field(default="fixed")  # fixed, characterizer, optimizer

    # PID tuning
    kp: float = Field(default=0.5, description="Proportional gain")
    ki: float = Field(default=0.1, description="Integral gain (1/min)")
    kd: float = Field(default=0.0, description="Derivative gain (min)")

    # Output limits
    output_high_limit: float = Field(default=10.0, description="Max trim bias (%)")
    output_low_limit: float = Field(default=-10.0, description="Min trim bias (%)")
    output_rate_limit: float = Field(default=2.0, description="Max rate (%/min)")

    # Anti-windup
    anti_windup_enabled: bool = Field(default=True)
    integral_high_limit: float = Field(default=15.0)
    integral_low_limit: float = Field(default=-15.0)

    # Deadband
    deadband: float = Field(default=0.1, description="Deadband around SP")

    # Analyzer validation
    max_stale_seconds: float = Field(default=60.0)
    o2_low_alarm: float = Field(default=1.0)
    o2_high_alarm: float = Field(default=8.0)

    # Inhibit conditions
    inhibit_on_load_change: bool = Field(default=True)
    load_change_threshold: float = Field(default=5.0, description="% load change/min")


class O2TrimController:
    """
    O2 trim controller for combustion optimization.

    Provides fine-tuning of combustion air based on stack O2 measurements.
    Implements PID control with anti-windup and rate limiting.

    Example:
        >>> controller = O2TrimController(config)
        >>> output = controller.execute(
        ...     measured_o2=3.5,
        ...     timestamp=datetime.now(timezone.utc)
        ... )
    """

    def __init__(self, config: TrimConfig):
        """Initialize O2 trim controller."""
        self.config = config
        self._mode = TrimMode.AUTO
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._last_timestamp: Optional[datetime] = None
        self._last_o2: Optional[float] = None
        self._last_load: float = 0.0
        self._error_history: deque = deque(maxlen=100)
        self._inhibit_until: Optional[datetime] = None
        logger.info(f"O2TrimController initialized for {config.unit_id}")

    @property
    def mode(self) -> TrimMode:
        return self._mode

    def set_mode(self, mode: TrimMode) -> None:
        """Set controller mode."""
        old_mode = self._mode
        self._mode = mode

        if mode == TrimMode.AUTO and old_mode != TrimMode.AUTO:
            # Bumpless transfer - initialize integral for current output
            pass

        logger.info(f"O2TrimController mode changed: {old_mode.value} -> {mode.value}")

    def set_setpoint(self, setpoint: float) -> None:
        """Update O2 setpoint."""
        self.config.o2_setpoint = max(0, min(21, setpoint))
        logger.info(f"O2 setpoint updated to {self.config.o2_setpoint}%")

    def execute(
        self,
        measured_o2: float,
        timestamp: Optional[datetime] = None,
        analyzer_quality: O2AnalyzerQuality = O2AnalyzerQuality.GOOD,
        current_load: float = 0.0,
        manual_bias: Optional[float] = None,
    ) -> TrimOutput:
        """
        Execute one cycle of O2 trim control.

        Args:
            measured_o2: Measured stack O2 (%)
            timestamp: Measurement timestamp
            analyzer_quality: Data quality indicator
            current_load: Current load (% for load change detection)
            manual_bias: Manual bias override (if in manual mode)

        Returns:
            TrimOutput with trim bias and diagnostics
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        status = TrimStatus.ACTIVE
        limiting_factor = None
        output_limited = False

        # Calculate dt
        if self._last_timestamp:
            dt_seconds = (timestamp - self._last_timestamp).total_seconds()
            dt_minutes = dt_seconds / 60.0
        else:
            dt_minutes = 1.0 / 60.0  # Assume 1 second for first cycle

        # Check analyzer quality
        if analyzer_quality in [O2AnalyzerQuality.BAD, O2AnalyzerQuality.STALE]:
            status = TrimStatus.ALARM
            # Hold last output
            trim_bias = self._last_output
        elif self._mode == TrimMode.OFF:
            trim_bias = 0.0
        elif self._mode == TrimMode.MANUAL:
            trim_bias = manual_bias if manual_bias is not None else self._last_output
        elif self._mode == TrimMode.TRACKING:
            # Track process but don't output
            trim_bias = 0.0
        else:
            # AUTO mode - execute PID
            # Check for inhibit conditions
            inhibited = self._check_inhibit_conditions(current_load, timestamp)

            if inhibited:
                status = TrimStatus.INHIBITED
                trim_bias = self._last_output
            else:
                # Calculate error
                error = self.config.o2_setpoint - measured_o2

                # Apply deadband
                if abs(error) < self.config.deadband:
                    error = 0.0

                # Proportional term
                p_term = self.config.kp * error

                # Integral term (with anti-windup)
                if self.config.anti_windup_enabled:
                    # Only integrate if not saturated in the direction of error
                    if not (
                        (self._last_output >= self.config.output_high_limit and error > 0) or
                        (self._last_output <= self.config.output_low_limit and error < 0)
                    ):
                        self._integral += self.config.ki * error * dt_minutes
                else:
                    self._integral += self.config.ki * error * dt_minutes

                # Clamp integral
                self._integral = max(
                    self.config.integral_low_limit,
                    min(self.config.integral_high_limit, self._integral)
                )
                i_term = self._integral

                # Derivative term (on measurement, not error, to avoid derivative kick)
                if self._last_o2 is not None and dt_minutes > 0:
                    d_measurement = (measured_o2 - self._last_o2) / dt_minutes
                    d_term = -self.config.kd * d_measurement  # Negative because acting on PV
                else:
                    d_term = 0.0

                # Calculate raw output
                raw_output = p_term + i_term + d_term

                # Apply rate limit
                if self._last_output is not None:
                    max_change = self.config.output_rate_limit * dt_minutes
                    output_change = raw_output - self._last_output
                    if abs(output_change) > max_change:
                        raw_output = self._last_output + max_change * (1 if output_change > 0 else -1)
                        output_limited = True
                        limiting_factor = "rate_limit"

                # Apply output limits
                if raw_output > self.config.output_high_limit:
                    trim_bias = self.config.output_high_limit
                    output_limited = True
                    limiting_factor = "high_limit"
                    status = TrimStatus.SATURATED_HIGH
                elif raw_output < self.config.output_low_limit:
                    trim_bias = self.config.output_low_limit
                    output_limited = True
                    limiting_factor = "low_limit"
                    status = TrimStatus.SATURATED_LOW
                else:
                    trim_bias = raw_output

                # Check O2 alarms
                if measured_o2 < self.config.o2_low_alarm:
                    status = TrimStatus.ALARM
                    # Force positive bias to increase air
                    trim_bias = max(trim_bias, self.config.output_high_limit * 0.5)
                elif measured_o2 > self.config.o2_high_alarm:
                    status = TrimStatus.ALARM

                # Store error history
                self._error_history.append({
                    "timestamp": timestamp,
                    "error": error,
                    "output": trim_bias,
                })

        # Calculate change
        trim_bias_change = trim_bias - self._last_output if self._last_output else 0.0

        # Update state
        self._last_output = trim_bias
        self._last_timestamp = timestamp
        self._last_o2 = measured_o2
        self._last_error = self.config.o2_setpoint - measured_o2
        self._last_load = current_load

        return TrimOutput(
            unit_id=self.config.unit_id,
            measured_o2=round(measured_o2, 2),
            o2_setpoint=self.config.o2_setpoint,
            error=round(self._last_error, 3),
            trim_bias=round(trim_bias, 2),
            trim_bias_change=round(trim_bias_change, 3),
            mode=self._mode,
            status=status,
            analyzer_quality=analyzer_quality,
            proportional_term=round(self.config.kp * self._last_error, 3) if self._mode == TrimMode.AUTO else 0,
            integral_term=round(self._integral, 3),
            derivative_term=0.0,  # Would need to track
            output_limited=output_limited,
            limiting_factor=limiting_factor,
        )

    def _check_inhibit_conditions(
        self,
        current_load: float,
        timestamp: datetime
    ) -> bool:
        """Check if controller should be inhibited."""
        # Check explicit inhibit timer
        if self._inhibit_until and timestamp < self._inhibit_until:
            return True

        # Check load change rate
        if self.config.inhibit_on_load_change:
            if self._last_load > 0:
                load_change = abs(current_load - self._last_load)
                if load_change > self.config.load_change_threshold:
                    # Inhibit for 30 seconds on load change
                    self._inhibit_until = timestamp + timedelta(seconds=30)
                    logger.info(f"O2 trim inhibited due to load change: {load_change:.1f}%")
                    return True

        return False

    def reset_integral(self) -> None:
        """Reset integral term to zero."""
        self._integral = 0.0
        logger.info("O2 trim integral reset")

    def set_manual_output(self, bias: float) -> None:
        """Set manual output and switch to manual mode."""
        self._mode = TrimMode.MANUAL
        self._last_output = max(
            self.config.output_low_limit,
            min(self.config.output_high_limit, bias)
        )

    def bumpless_transfer_to_auto(self) -> None:
        """Perform bumpless transfer from manual to auto."""
        # Set integral to maintain current output
        self._integral = self._last_output - self.config.kp * self._last_error
        self._mode = TrimMode.AUTO
        logger.info("Bumpless transfer to AUTO mode")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get controller diagnostics."""
        return {
            "mode": self._mode.value,
            "integral": round(self._integral, 3),
            "last_output": round(self._last_output, 3),
            "last_error": round(self._last_error, 3),
            "error_history_size": len(self._error_history),
            "config": self.config.model_dump(),
        }
