"""
GL-004 BURNMASTER - Actuator Rate Limiter

This module implements rate limiting for actuator movements in the burner
management control system. It prevents excessive actuator wear and hunting
while ensuring smooth process transitions.

Key Features:
    - Rate limiting for all actuators
    - Hunting detection and prevention
    - Actuator wear tracking
    - Configurable limits per actuator
    - Complete audit trail

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid
from collections import deque

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActuatorType(str, Enum):
    """Types of actuators in the burner management system."""
    DAMPER = "damper"
    VALVE = "valve"
    MOTOR = "motor"
    VFD = "vfd"
    POSITIONER = "positioner"


class LimitStatus(str, Enum):
    """Status of rate limiting check."""
    ALLOWED = "allowed"
    LIMITED = "limited"
    BLOCKED = "blocked"
    HUNTING_DETECTED = "hunting_detected"


class HuntingStatus(str, Enum):
    """Status of hunting detection."""
    NORMAL = "normal"
    WARNING = "warning"
    HUNTING = "hunting"
    BLOCKED = "blocked"


class ActuatorLimits(BaseModel):
    """Rate limits configuration for an actuator."""
    actuator_id: str
    actuator_type: ActuatorType
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=100.0)
    rate_limit_per_second: float = Field(default=1.0, gt=0)
    rate_limit_per_minute: float = Field(default=30.0, gt=0)
    max_moves_per_minute: int = Field(default=10, ge=1)
    min_move_interval_seconds: float = Field(default=1.0, ge=0)
    deadband: float = Field(default=0.5, ge=0)
    hunting_threshold: int = Field(default=5, ge=2)
    hunting_window_seconds: float = Field(default=60.0, gt=0)
    unit: str = Field(default="%")


class HuntingDetection(BaseModel):
    """Hunting detection result."""
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    actuator_id: str
    status: HuntingStatus
    reversal_count: int = Field(default=0, ge=0)
    window_seconds: float = Field(default=60.0)
    threshold: int = Field(default=5)
    detected_at: Optional[datetime] = None
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recommendation: str = Field(default="")


class RateLimitResult(BaseModel):
    """Result of rate limiting a command."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    actuator_id: str
    requested_value: float
    allowed_value: float
    original_value: float
    status: LimitStatus
    limited_by_rate: bool = Field(default=False)
    limited_by_deadband: bool = Field(default=False)
    hunting_detected: bool = Field(default=False)
    rate_applied: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.result_id}|{self.actuator_id}|{self.requested_value}|{self.allowed_value}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class ActuatorState(BaseModel):
    """Current state of an actuator."""
    actuator_id: str
    current_value: float
    target_value: float
    last_move_time: Optional[datetime] = None
    move_count_last_minute: int = Field(default=0, ge=0)
    total_travel: float = Field(default=0.0, ge=0)
    reversal_count: int = Field(default=0, ge=0)
    hunting_status: HuntingStatus = Field(default=HuntingStatus.NORMAL)


class MoveRecord(BaseModel):
    """Record of an actuator movement."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    actuator_id: str
    from_value: float
    to_value: float
    move_size: float
    direction: str  # "up" or "down"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    was_limited: bool = Field(default=False)
    was_reversal: bool = Field(default=False)


class ActuatorRateLimiter:
    """
    Rate limiter for actuator movements.

    This class provides:
    - Rate limiting per actuator
    - Hunting detection and prevention
    - Move tracking and statistics
    - Wear monitoring

    Example:
        >>> limiter = ActuatorRateLimiter()
        >>> limiter.configure_actuator("damper_position", ActuatorLimits(...))
        >>> result = limiter.limit_rate("damper_position", current=50.0, target=60.0)
        >>> if result.status == LimitStatus.ALLOWED:
        ...     apply_setpoint(result.allowed_value)
    """

    def __init__(self) -> None:
        """Initialize the actuator rate limiter."""
        self._limits: Dict[str, ActuatorLimits] = {}
        self._states: Dict[str, ActuatorState] = {}
        self._move_history: Dict[str, deque] = {}
        self._hunting_status: Dict[str, HuntingDetection] = {}
        self._audit_log: List[Dict[str, Any]] = []

        # Statistics
        self._total_commands = 0
        self._limited_commands = 0
        self._blocked_commands = 0
        self._hunting_events = 0

        # Initialize default actuators
        self._initialize_default_actuators()

        logger.info("ActuatorRateLimiter initialized")

    def _initialize_default_actuators(self) -> None:
        """Initialize default actuator configurations."""
        default_actuators = [
            ActuatorLimits(
                actuator_id="damper_position",
                actuator_type=ActuatorType.DAMPER,
                rate_limit_per_second=2.0,
                rate_limit_per_minute=60.0,
                max_moves_per_minute=15,
                deadband=0.5,
                hunting_threshold=6
            ),
            ActuatorLimits(
                actuator_id="fuel_valve_position",
                actuator_type=ActuatorType.VALVE,
                rate_limit_per_second=1.5,
                rate_limit_per_minute=45.0,
                max_moves_per_minute=12,
                deadband=0.3,
                hunting_threshold=5
            ),
            ActuatorLimits(
                actuator_id="fgr_damper",
                actuator_type=ActuatorType.DAMPER,
                max_value=30.0,
                rate_limit_per_second=0.5,
                rate_limit_per_minute=15.0,
                max_moves_per_minute=8,
                deadband=0.5,
                hunting_threshold=5
            ),
            ActuatorLimits(
                actuator_id="combustion_air_fan",
                actuator_type=ActuatorType.VFD,
                rate_limit_per_second=1.0,
                rate_limit_per_minute=30.0,
                max_moves_per_minute=10,
                deadband=1.0,
                hunting_threshold=4
            ),
        ]

        for limits in default_actuators:
            self.configure_actuator(limits.actuator_id, limits)

    def configure_actuator(
        self,
        actuator_id: str,
        limits: ActuatorLimits
    ) -> None:
        """
        Configure rate limits for an actuator.

        Args:
            actuator_id: Unique identifier for the actuator
            limits: Rate limit configuration
        """
        self._limits[actuator_id] = limits
        self._states[actuator_id] = ActuatorState(
            actuator_id=actuator_id,
            current_value=0.0,
            target_value=0.0
        )
        self._move_history[actuator_id] = deque(maxlen=100)
        self._hunting_status[actuator_id] = HuntingDetection(
            actuator_id=actuator_id,
            status=HuntingStatus.NORMAL,
            threshold=limits.hunting_threshold,
            window_seconds=limits.hunting_window_seconds
        )

        self._log_event("ACTUATOR_CONFIGURED", limits)
        logger.info(f"Actuator configured: {actuator_id}")

    def limit_rate(
        self,
        actuator_id: str,
        current_value: float,
        target_value: float,
        dt_seconds: float = 1.0
    ) -> RateLimitResult:
        """
        Apply rate limiting to an actuator command.

        Args:
            actuator_id: ID of the actuator
            current_value: Current position value
            target_value: Requested target value
            dt_seconds: Time since last command (for rate calculation)

        Returns:
            RateLimitResult with allowed value and status
        """
        self._total_commands += 1

        limits = self._limits.get(actuator_id)
        if not limits:
            # No limits configured, allow as-is
            return RateLimitResult(
                actuator_id=actuator_id,
                requested_value=target_value,
                allowed_value=target_value,
                original_value=current_value,
                status=LimitStatus.ALLOWED
            )

        state = self._states.get(actuator_id)
        if state:
            state.current_value = current_value
            state.target_value = target_value

        # Calculate requested change
        requested_change = target_value - current_value

        # Check deadband
        if abs(requested_change) < limits.deadband:
            return RateLimitResult(
                actuator_id=actuator_id,
                requested_value=target_value,
                allowed_value=current_value,  # No change
                original_value=current_value,
                status=LimitStatus.ALLOWED,
                limited_by_deadband=True
            )

        # Check hunting
        hunting = self.detect_hunting(actuator_id)
        if hunting.status == HuntingStatus.BLOCKED:
            self._blocked_commands += 1
            return RateLimitResult(
                actuator_id=actuator_id,
                requested_value=target_value,
                allowed_value=current_value,
                original_value=current_value,
                status=LimitStatus.HUNTING_DETECTED,
                hunting_detected=True
            )

        # Apply rate limiting
        max_change = limits.rate_limit_per_second * dt_seconds
        limited_by_rate = False

        if abs(requested_change) > max_change:
            # Limit the change
            direction = 1 if requested_change > 0 else -1
            allowed_change = direction * max_change
            allowed_value = current_value + allowed_change
            limited_by_rate = True
            self._limited_commands += 1
        else:
            allowed_value = target_value

        # Enforce min/max bounds
        allowed_value = max(limits.min_value, min(limits.max_value, allowed_value))

        # Determine status
        if limited_by_rate:
            status = LimitStatus.LIMITED
        elif hunting.status == HuntingStatus.WARNING:
            status = LimitStatus.ALLOWED  # Allowed but with warning
        else:
            status = LimitStatus.ALLOWED

        # Record the move
        actual_change = allowed_value - current_value
        if abs(actual_change) >= limits.deadband:
            self._record_move(actuator_id, current_value, allowed_value, limited_by_rate)

        result = RateLimitResult(
            actuator_id=actuator_id,
            requested_value=target_value,
            allowed_value=allowed_value,
            original_value=current_value,
            status=status,
            limited_by_rate=limited_by_rate,
            hunting_detected=hunting.status in [HuntingStatus.WARNING, HuntingStatus.HUNTING],
            rate_applied=abs(actual_change) / dt_seconds if dt_seconds > 0 else 0.0
        )

        return result

    def detect_hunting(self, actuator_id: str) -> HuntingDetection:
        """
        Detect hunting behavior in actuator movements.

        Args:
            actuator_id: ID of the actuator

        Returns:
            HuntingDetection result
        """
        limits = self._limits.get(actuator_id)
        history = self._move_history.get(actuator_id)
        hunting = self._hunting_status.get(actuator_id)

        if not limits or not history or not hunting:
            return HuntingDetection(
                actuator_id=actuator_id,
                status=HuntingStatus.NORMAL
            )

        # Count reversals in the window
        window_start = datetime.now(timezone.utc) - timedelta(seconds=limits.hunting_window_seconds)
        recent_moves = [m for m in history if m.timestamp > window_start]

        reversal_count = sum(1 for m in recent_moves if m.was_reversal)

        # Update hunting status
        if reversal_count >= limits.hunting_threshold:
            hunting.status = HuntingStatus.HUNTING
            hunting.detected_at = datetime.now(timezone.utc)
            hunting.recommendation = "Reduce control action or increase deadband"
            self._hunting_events += 1
            logger.warning(f"Hunting detected on {actuator_id}: {reversal_count} reversals")

            # If severe, block further moves temporarily
            if reversal_count >= limits.hunting_threshold * 1.5:
                hunting.status = HuntingStatus.BLOCKED
                hunting.recommendation = "Actuator blocked due to excessive hunting"
        elif reversal_count >= limits.hunting_threshold * 0.7:
            hunting.status = HuntingStatus.WARNING
            hunting.recommendation = "Approaching hunting threshold"
        else:
            hunting.status = HuntingStatus.NORMAL
            hunting.detected_at = None
            hunting.recommendation = ""

        hunting.reversal_count = reversal_count
        hunting.last_check = datetime.now(timezone.utc)

        self._hunting_status[actuator_id] = hunting
        return hunting

    def reset_hunting(self, actuator_id: str) -> bool:
        """
        Reset hunting detection for an actuator.

        Args:
            actuator_id: ID of the actuator

        Returns:
            True if reset successful
        """
        if actuator_id not in self._hunting_status:
            return False

        hunting = self._hunting_status[actuator_id]
        hunting.status = HuntingStatus.NORMAL
        hunting.reversal_count = 0
        hunting.detected_at = None
        hunting.recommendation = ""

        # Clear recent move history
        if actuator_id in self._move_history:
            self._move_history[actuator_id].clear()

        self._log_event("HUNTING_RESET", {"actuator_id": actuator_id})
        logger.info(f"Hunting reset for {actuator_id}")

        return True

    def get_actuator_limits(self, actuator_id: str) -> Optional[ActuatorLimits]:
        """Get limits for an actuator."""
        return self._limits.get(actuator_id)

    def get_actuator_state(self, actuator_id: str) -> Optional[ActuatorState]:
        """Get current state of an actuator."""
        return self._states.get(actuator_id)

    def get_hunting_status(self, actuator_id: str) -> Optional[HuntingDetection]:
        """Get hunting status for an actuator."""
        return self._hunting_status.get(actuator_id)

    def get_move_history(
        self,
        actuator_id: str,
        limit: int = 50
    ) -> List[MoveRecord]:
        """Get recent move history for an actuator."""
        history = self._move_history.get(actuator_id)
        if not history:
            return []
        return list(history)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_commands": self._total_commands,
            "limited_commands": self._limited_commands,
            "blocked_commands": self._blocked_commands,
            "hunting_events": self._hunting_events,
            "limit_rate": (
                self._limited_commands / self._total_commands
                if self._total_commands > 0 else 0.0
            ),
            "actuator_count": len(self._limits)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        actuator_statuses = {}
        for actuator_id in self._limits:
            state = self._states.get(actuator_id)
            hunting = self._hunting_status.get(actuator_id)
            actuator_statuses[actuator_id] = {
                "current_value": state.current_value if state else 0.0,
                "hunting_status": hunting.status.value if hunting else "unknown"
            }

        return {
            "statistics": self.get_statistics(),
            "actuators": actuator_statuses
        }

    def _record_move(
        self,
        actuator_id: str,
        from_value: float,
        to_value: float,
        was_limited: bool
    ) -> None:
        """Record an actuator movement."""
        history = self._move_history.get(actuator_id)
        if not history:
            return

        move_size = abs(to_value - from_value)
        direction = "up" if to_value > from_value else "down"

        # Check if this is a reversal
        was_reversal = False
        if len(history) > 0:
            last_move = history[-1]
            if last_move.direction != direction:
                was_reversal = True

        record = MoveRecord(
            actuator_id=actuator_id,
            from_value=from_value,
            to_value=to_value,
            move_size=move_size,
            direction=direction,
            was_limited=was_limited,
            was_reversal=was_reversal
        )

        history.append(record)

        # Update state
        state = self._states.get(actuator_id)
        if state:
            state.current_value = to_value
            state.last_move_time = datetime.now(timezone.utc)
            state.total_travel += move_size
            if was_reversal:
                state.reversal_count += 1

    def _log_event(self, event_type: str, data: Any) -> None:
        """Log an event to the audit trail."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data.model_dump() if hasattr(data, 'model_dump') else data
        }
        self._audit_log.append(audit_entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return list(reversed(self._audit_log[-limit:]))
