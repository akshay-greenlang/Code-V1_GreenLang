"""
GL-004 BURNMASTER - Damper Position Controller

Air damper position control for combustion optimization.
Manages FD fan dampers, ID fan dampers, and air register positions.

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DamperType(str, Enum):
    """Types of combustion dampers."""
    FD_FAN = "fd_fan"           # Forced draft fan discharge damper
    ID_FAN = "id_fan"           # Induced draft fan inlet damper
    AIR_REGISTER = "air_register"  # Burner air register
    SECONDARY_AIR = "secondary_air"
    OVERFIRE_AIR = "overfire_air"
    FGR_DAMPER = "fgr_damper"   # Flue gas recirculation


class DamperMode(str, Enum):
    """Damper control modes."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"     # Follows master controller
    CHARACTERIZER = "characterizer"  # Position vs load curve


class DamperStatus(str, Enum):
    """Damper operational status."""
    NORMAL = "normal"
    TRACKING = "tracking"
    SATURATED_OPEN = "saturated_open"
    SATURATED_CLOSED = "saturated_closed"
    FAULT = "fault"
    CALIBRATING = "calibrating"


class DamperOutput(BaseModel):
    """Damper controller output."""
    output_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    damper_id: str = Field(..., description="Damper identifier")
    damper_type: DamperType = Field(...)

    # Positions
    position_setpoint: float = Field(..., ge=0, le=100, description="Position setpoint (%)")
    position_feedback: float = Field(..., ge=0, le=100, description="Actual position (%)")
    position_error: float = Field(..., description="SP - PV error")

    # Flow/pressure effect
    estimated_air_flow_pct: float = Field(default=0.0, description="Est. air flow (%)")

    # Status
    mode: DamperMode = Field(default=DamperMode.AUTO)
    status: DamperStatus = Field(default=DamperStatus.NORMAL)

    # Movement
    last_move_direction: str = Field(default="none")  # opening, closing, none
    travel_time_seconds: float = Field(default=0.0)

    # Diagnostics
    actuator_healthy: bool = Field(default=True)
    position_at_limit: bool = Field(default=False)
    limit_type: Optional[str] = Field(None)


class DamperConfig(BaseModel):
    """Damper controller configuration."""
    damper_id: str = Field(..., description="Damper identifier")
    damper_type: DamperType = Field(...)
    unit_id: str = Field(..., description="Associated combustion unit")

    # Position limits
    position_min: float = Field(default=0.0, ge=0, le=100)
    position_max: float = Field(default=100.0, ge=0, le=100)

    # Rate limits
    open_rate_pct_per_sec: float = Field(default=5.0, description="Opening rate")
    close_rate_pct_per_sec: float = Field(default=5.0, description="Closing rate")

    # Deadband
    position_deadband: float = Field(default=0.5, description="Position deadband (%)")

    # Characterization curve (position vs load)
    # List of (load%, position%) tuples
    characterization_curve: List[Tuple[float, float]] = Field(
        default=[(0, 10), (25, 25), (50, 50), (75, 75), (100, 100)]
    )

    # Safety
    fail_position: float = Field(default=100.0, description="Position on failure (%)")
    min_position_interlock: float = Field(default=5.0, description="Minimum for flame safety")


class DamperPositionController:
    """
    Damper position controller for combustion air management.

    Controls air damper positions based on load characterization,
    O2 trim bias, or direct setpoint from optimizer.

    Example:
        >>> controller = DamperPositionController(config)
        >>> output = controller.calculate_position(
        ...     load_percent=75,
        ...     o2_trim_bias=2.0,
        ...     feedback_position=70.0
        ... )
    """

    def __init__(self, config: DamperConfig):
        """Initialize damper controller."""
        self.config = config
        self._mode = DamperMode.CHARACTERIZER
        self._manual_setpoint = 50.0
        self._current_setpoint = 50.0
        self._last_position = 50.0
        self._last_timestamp: Optional[datetime] = None
        self._position_history: deque = deque(maxlen=100)
        logger.info(f"DamperPositionController initialized: {config.damper_id}")

    def calculate_position(
        self,
        load_percent: float,
        o2_trim_bias: float = 0.0,
        feedback_position: float = 50.0,
        optimizer_setpoint: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> DamperOutput:
        """
        Calculate damper position setpoint.

        Args:
            load_percent: Current load (% of design)
            o2_trim_bias: Bias from O2 trim controller (%)
            feedback_position: Actual damper position feedback
            optimizer_setpoint: Direct setpoint from optimizer (if cascade)
            timestamp: Calculation timestamp

        Returns:
            DamperOutput with position setpoint and status
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        status = DamperStatus.NORMAL
        limit_type = None

        # Calculate base position based on mode
        if self._mode == DamperMode.MANUAL:
            base_position = self._manual_setpoint

        elif self._mode == DamperMode.CASCADE and optimizer_setpoint is not None:
            base_position = optimizer_setpoint

        elif self._mode in [DamperMode.AUTO, DamperMode.CHARACTERIZER]:
            # Get characterized position for load
            base_position = self._get_characterized_position(load_percent)

        else:
            base_position = self._current_setpoint

        # Apply O2 trim bias
        biased_position = base_position + o2_trim_bias

        # Apply rate limiting
        if self._last_timestamp:
            dt_seconds = (timestamp - self._last_timestamp).total_seconds()
            position_setpoint = self._apply_rate_limit(
                biased_position, self._current_setpoint, dt_seconds
            )
        else:
            position_setpoint = biased_position

        # Apply position limits
        position_setpoint, at_limit, limit_type = self._apply_limits(position_setpoint)

        if at_limit:
            if position_setpoint >= self.config.position_max:
                status = DamperStatus.SATURATED_OPEN
            else:
                status = DamperStatus.SATURATED_CLOSED

        # Calculate error
        position_error = position_setpoint - feedback_position

        # Determine move direction
        if abs(position_setpoint - self._current_setpoint) < self.config.position_deadband:
            move_direction = "none"
        elif position_setpoint > self._current_setpoint:
            move_direction = "opening"
        else:
            move_direction = "closing"

        # Estimate air flow (simplified linear assumption)
        estimated_air_flow = self._estimate_air_flow(feedback_position)

        # Calculate travel time
        travel_time = self._calculate_travel_time(
            feedback_position, position_setpoint
        )

        # Update state
        self._current_setpoint = position_setpoint
        self._last_position = feedback_position
        self._last_timestamp = timestamp

        # Record history
        self._position_history.append({
            "timestamp": timestamp,
            "setpoint": position_setpoint,
            "feedback": feedback_position,
            "load": load_percent,
        })

        return DamperOutput(
            damper_id=self.config.damper_id,
            damper_type=self.config.damper_type,
            position_setpoint=round(position_setpoint, 1),
            position_feedback=round(feedback_position, 1),
            position_error=round(position_error, 1),
            estimated_air_flow_pct=round(estimated_air_flow, 1),
            mode=self._mode,
            status=status,
            last_move_direction=move_direction,
            travel_time_seconds=round(travel_time, 1),
            actuator_healthy=True,
            position_at_limit=at_limit,
            limit_type=limit_type,
        )

    def _get_characterized_position(self, load_percent: float) -> float:
        """Get characterized damper position for given load."""
        curve = self.config.characterization_curve

        if not curve:
            return load_percent  # Linear default

        # Sort by load
        curve = sorted(curve, key=lambda x: x[0])

        # Handle out of range
        if load_percent <= curve[0][0]:
            return curve[0][1]
        if load_percent >= curve[-1][0]:
            return curve[-1][1]

        # Linear interpolation
        for i in range(len(curve) - 1):
            load_low, pos_low = curve[i]
            load_high, pos_high = curve[i + 1]

            if load_low <= load_percent <= load_high:
                factor = (load_percent - load_low) / (load_high - load_low)
                return pos_low + factor * (pos_high - pos_low)

        return load_percent  # Fallback

    def _apply_rate_limit(
        self,
        target: float,
        current: float,
        dt_seconds: float
    ) -> float:
        """Apply damper travel rate limit."""
        if dt_seconds <= 0:
            return current

        delta = target - current

        if delta > 0:
            max_delta = self.config.open_rate_pct_per_sec * dt_seconds
        else:
            max_delta = -self.config.close_rate_pct_per_sec * dt_seconds

        if abs(delta) > abs(max_delta):
            return current + max_delta
        return target

    def _apply_limits(self, position: float) -> Tuple[float, bool, Optional[str]]:
        """Apply position limits."""
        at_limit = False
        limit_type = None

        # Safety interlock minimum
        if position < self.config.min_position_interlock:
            position = self.config.min_position_interlock
            at_limit = True
            limit_type = "safety_min"

        # Configured limits
        if position < self.config.position_min:
            position = self.config.position_min
            at_limit = True
            limit_type = "config_min"
        elif position > self.config.position_max:
            position = self.config.position_max
            at_limit = True
            limit_type = "config_max"

        return position, at_limit, limit_type

    def _estimate_air_flow(self, position: float) -> float:
        """Estimate air flow percentage from damper position."""
        # Non-linear damper characteristic (typical butterfly)
        # Flow ~ sin^2(position * pi/200) for butterfly
        # Simplified: use square relationship
        flow = (position / 100) ** 0.5 * 100
        return min(100, max(0, flow))

    def _calculate_travel_time(self, current: float, target: float) -> float:
        """Calculate estimated travel time in seconds."""
        delta = abs(target - current)
        if delta == 0:
            return 0.0

        if target > current:
            rate = self.config.open_rate_pct_per_sec
        else:
            rate = self.config.close_rate_pct_per_sec

        if rate <= 0:
            return 0.0

        return delta / rate

    def set_mode(self, mode: DamperMode) -> None:
        """Set controller mode."""
        old_mode = self._mode
        self._mode = mode
        logger.info(f"Damper {self.config.damper_id} mode: {old_mode.value} -> {mode.value}")

    def set_manual_position(self, position: float) -> None:
        """Set manual position setpoint."""
        self._mode = DamperMode.MANUAL
        self._manual_setpoint = max(
            self.config.position_min,
            min(self.config.position_max, position)
        )
        logger.info(f"Damper {self.config.damper_id} manual position: {self._manual_setpoint}%")

    def update_characterization(
        self,
        curve: List[Tuple[float, float]]
    ) -> None:
        """Update characterization curve."""
        self.config.characterization_curve = curve
        logger.info(f"Damper {self.config.damper_id} characterization updated")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get controller diagnostics."""
        return {
            "damper_id": self.config.damper_id,
            "mode": self._mode.value,
            "current_setpoint": self._current_setpoint,
            "last_position": self._last_position,
            "history_size": len(self._position_history),
            "config": self.config.model_dump(),
        }
