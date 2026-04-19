# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Insulation Velocity Limiter

Limits the rate of change for insulation condition score assessments to prevent:
- Condition score oscillation (flip-flopping between good/bad ratings)
- Operator confusion from rapidly changing guidance
- System instability from over-reactive repair recommendations

The velocity limiter implements:
1. Maximum delta per time period constraints
2. Cooldown periods between significant changes
3. Exponential moving average smoothing
4. Ramp rate limiting for gradual transitions
5. State tracking with full provenance

Safety Principles:
- Smooth transitions prevent operational disruption
- Cooldown periods allow operator assessment
- All state changes are auditable
- Conservative defaults favor stability

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ConditionPriority(str, Enum):
    """Priority level for insulation repair recommendations."""

    EXCELLENT = "excellent"  # No repair needed
    GOOD = "good"  # Monitor condition
    FAIR = "fair"  # Schedule inspection
    POOR = "poor"  # Schedule repair
    CRITICAL = "critical"  # Immediate repair required
    FAILED = "failed"  # Emergency replacement needed


class VelocityViolationType(str, Enum):
    """Type of velocity constraint violation."""

    NONE = "none"
    MAX_DELTA_EXCEEDED = "max_delta_exceeded"
    RAMP_RATE_EXCEEDED = "ramp_rate_exceeded"
    COOLDOWN_ACTIVE = "cooldown_active"
    OSCILLATION_DETECTED = "oscillation_detected"
    SMOOTHING_APPLIED = "smoothing_applied"
    COMBINED = "combined"


class ConstraintAction(str, Enum):
    """Action taken by the velocity limiter."""

    PASSTHROUGH = "passthrough"  # No constraint applied
    CLAMPED = "clamped"  # Value clamped to max delta
    SMOOTHED = "smoothed"  # EMA smoothing applied
    RAMPED = "ramped"  # Ramp rate limit applied
    HELD = "held"  # Held at previous value (cooldown)
    DAMPENED = "dampened"  # Oscillation dampening applied


class CooldownReason(str, Enum):
    """Reason for cooldown activation."""

    PRIORITY_CHANGE = "priority_change"  # Significant priority change
    DIRECTION_CHANGE = "direction_change"  # Condition trend reversed
    MANUAL_OVERRIDE = "manual_override"  # Operator overrode assessment
    OSCILLATION = "oscillation"  # Oscillation detected
    SYSTEM_INITIATED = "system_initiated"  # System-level cooldown


# =============================================================================
# CONFIGURATION
# =============================================================================


class InsulationVelocityLimiterConfig(BaseModel):
    """
    Configuration for the insulation condition velocity limiter.

    Default values are tuned for insulation condition assessments,
    balancing responsiveness with stability.

    Attributes:
        max_delta_per_hour: Maximum condition score change per hour
        smoothing_window_hours: EMA window for smoothing
        ramp_rate_per_hour: Maximum ramp rate for condition score
        cooldown_duration_hours: Cooldown period after significant changes
        oscillation_window_count: Number of changes to detect oscillation
        oscillation_threshold: Threshold for oscillation detection
        priority_change_cooldown_hours: Cooldown after priority level change
        history_size: Number of historical entries to retain
        violation_threshold: Consecutive violations before alert
        enable_clamping: Enable max delta clamping
        enable_smoothing: Enable EMA smoothing
        enable_ramping: Enable ramp rate limiting
        enable_cooldown: Enable cooldown periods
        enable_oscillation_detection: Enable oscillation detection
    """

    max_delta_per_hour: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Maximum condition score change per hour"
    )
    smoothing_window_hours: float = Field(
        default=8.0,
        ge=1.0,
        le=48.0,
        description="EMA smoothing window in hours"
    )
    ramp_rate_per_hour: float = Field(
        default=0.08,
        ge=0.01,
        le=0.5,
        description="Maximum ramp rate per hour"
    )
    cooldown_duration_hours: float = Field(
        default=4.0,
        ge=1.0,
        le=48.0,
        description="Cooldown period after significant changes"
    )
    oscillation_window_count: int = Field(
        default=6,
        ge=3,
        le=20,
        description="Number of changes to evaluate for oscillation"
    )
    oscillation_threshold: float = Field(
        default=0.35,
        ge=0.1,
        le=0.8,
        description="Cumulative directional change threshold for oscillation"
    )
    priority_change_cooldown_hours: float = Field(
        default=8.0,
        ge=2.0,
        le=72.0,
        description="Cooldown after priority level change"
    )
    history_size: int = Field(
        default=336,
        ge=48,
        le=1000,
        description="History entries (default 14 days hourly)"
    )
    violation_threshold: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Consecutive violations before alert"
    )
    enable_clamping: bool = Field(
        default=True,
        description="Enable max delta clamping"
    )
    enable_smoothing: bool = Field(
        default=True,
        description="Enable EMA smoothing"
    )
    enable_ramping: bool = Field(
        default=True,
        description="Enable ramp rate limiting"
    )
    enable_cooldown: bool = Field(
        default=True,
        description="Enable cooldown periods"
    )
    enable_oscillation_detection: bool = Field(
        default=True,
        description="Enable oscillation detection and dampening"
    )
    min_condition_score: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Minimum allowed condition score (0=failed)"
    )
    max_condition_score: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum allowed condition score (1=excellent)"
    )

    @field_validator("max_delta_per_hour")
    @classmethod
    def validate_max_delta(cls, v: float) -> float:
        """Validate max delta is reasonable."""
        if v > 0.3:
            logger.warning(
                f"max_delta_per_hour={v} is high, may allow rapid changes"
            )
        return v


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ConditionState:
    """
    State tracking for a single insulation asset's condition assessments.

    Attributes:
        asset_id: Insulation asset identifier
        current_score: Current condition score [0, 1]
        previous_score: Previous condition score
        current_priority: Current priority level
        previous_priority: Previous priority level
        ema_score: Exponentially smoothed score
        ramp_target: Target value for ramping
        last_update: Timestamp of last update
        cooldown_until: When cooldown expires (if active)
        cooldown_reason: Why cooldown was activated
        history: Historical (timestamp, raw, constrained) values
        violation_count: Consecutive velocity violations
        oscillation_score: Current oscillation score
    """

    asset_id: str
    current_score: float = 0.5
    previous_score: float = 0.5
    current_priority: ConditionPriority = ConditionPriority.FAIR
    previous_priority: ConditionPriority = ConditionPriority.FAIR
    ema_score: float = 0.5
    ramp_target: float = 0.5
    last_update: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    cooldown_reason: Optional[CooldownReason] = None
    history: Deque[Tuple[datetime, float, float]] = field(
        default_factory=lambda: deque(maxlen=336)
    )
    violation_count: int = 0
    oscillation_score: float = 0.0

    def is_in_cooldown(self, now: Optional[datetime] = None) -> bool:
        """Check if currently in cooldown period."""
        if self.cooldown_until is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now < self.cooldown_until


@dataclass
class VelocityCheckResult:
    """
    Result of applying velocity constraints to a condition assessment.

    Provides complete audit trail for the constraint decision.

    Attributes:
        asset_id: Insulation asset identifier
        original_score: Raw condition score before constraints
        constrained_score: Condition score after constraints
        original_priority: Raw priority level
        constrained_priority: Priority after constraints
        was_constrained: Whether any constraint was applied
        violation_type: Type of velocity violation (if any)
        action_taken: Constraint action applied
        delta_requested: Change requested by new assessment
        delta_allowed: Change actually allowed
        delta_per_hour: Rate of change per hour
        smoothed_value: EMA smoothed value (if smoothing enabled)
        in_cooldown: Whether cooldown was active
        cooldown_remaining_hours: Hours until cooldown expires
        oscillation_detected: Whether oscillation was detected
        constraint_ratio: Ratio of allowed to requested change
        timestamp: When this check was performed
        elapsed_hours: Time since last update
        provenance_hash: SHA-256 hash for audit trail
    """

    asset_id: str
    original_score: float
    constrained_score: float
    original_priority: ConditionPriority
    constrained_priority: ConditionPriority
    was_constrained: bool
    violation_type: VelocityViolationType
    action_taken: ConstraintAction
    delta_requested: float
    delta_allowed: float
    delta_per_hour: float
    smoothed_value: Optional[float] = None
    in_cooldown: bool = False
    cooldown_remaining_hours: float = 0.0
    oscillation_detected: bool = False
    constraint_ratio: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_hours: float = 0.0
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            content = (
                f"{self.asset_id}|{self.original_score:.6f}|"
                f"{self.constrained_score:.6f}|{self.action_taken.value}|"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class CooldownEvent:
    """Record of a cooldown activation."""

    asset_id: str
    reason: CooldownReason
    started_at: datetime
    expires_at: datetime
    trigger_value: float
    previous_value: float
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.asset_id}|{self.reason.value}|"
                f"{self.started_at.isoformat()}|{self.trigger_value:.6f}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# INSULATION VELOCITY LIMITER
# =============================================================================


class InsulationVelocityLimiter:
    """
    Limits the rate of change for insulation condition assessments.

    Prevents condition score oscillation and provides smooth transitions
    for operator-facing repair recommendations. Implements multiple layers
    of velocity control:

    1. **Max Delta Clamping**: Limits maximum change per time period
    2. **EMA Smoothing**: Exponential moving average for noise reduction
    3. **Ramp Rate Limiting**: Gradual transitions to new values
    4. **Cooldown Periods**: Forced stability after significant changes
    5. **Oscillation Detection**: Detects and dampens flip-flopping

    Thread Safety:
        All public methods are thread-safe via internal locking.

    Example:
        >>> config = InsulationVelocityLimiterConfig(max_delta_per_hour=0.10)
        >>> limiter = InsulationVelocityLimiter(config)
        >>>
        >>> # Apply velocity limits to new assessment
        >>> result = limiter.apply(
        ...     asset_id="INS-PIPE-101",
        ...     new_score=0.35,
        ...     new_priority=ConditionPriority.POOR,
        ... )
        >>>
        >>> if result.was_constrained:
        ...     print(f"Score limited: {result.original_score} -> {result.constrained_score}")

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[InsulationVelocityLimiterConfig] = None,
        violation_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Initialize the insulation velocity limiter.

        Args:
            config: Velocity limiter configuration
            violation_callback: Optional callback for violation alerts
        """
        self.config = config or InsulationVelocityLimiterConfig()
        self._states: Dict[str, ConditionState] = {}
        self._lock = threading.RLock()
        self._violation_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._cooldown_events: List[CooldownEvent] = []

        if violation_callback:
            self._violation_callbacks.append(violation_callback)

        logger.info(
            f"InsulationVelocityLimiter initialized: "
            f"max_delta={self.config.max_delta_per_hour}/hr, "
            f"smoothing={self.config.smoothing_window_hours}hr, "
            f"cooldown={self.config.cooldown_duration_hours}hr"
        )

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _get_state_key(self, asset_id: str) -> str:
        """Generate state storage key."""
        return asset_id

    def _get_or_create_state(
        self,
        asset_id: str,
        initial_score: float = 0.5,
    ) -> ConditionState:
        """Get existing state or create new one."""
        key = self._get_state_key(asset_id)

        if key not in self._states:
            self._states[key] = ConditionState(
                asset_id=asset_id,
                current_score=initial_score,
                previous_score=initial_score,
                ema_score=initial_score,
                ramp_target=initial_score,
                history=deque(maxlen=self.config.history_size),
            )
            logger.debug(f"Created new state for asset {asset_id}")

        return self._states[key]

    def get_state(self, asset_id: str) -> Optional[ConditionState]:
        """Get current state for an asset."""
        with self._lock:
            return self._states.get(self._get_state_key(asset_id))

    # =========================================================================
    # PRIORITY CONVERSION
    # =========================================================================

    def _score_to_priority(self, score: float) -> ConditionPriority:
        """Convert condition score to priority level."""
        if score >= 0.9:
            return ConditionPriority.EXCELLENT
        elif score >= 0.7:
            return ConditionPriority.GOOD
        elif score >= 0.5:
            return ConditionPriority.FAIR
        elif score >= 0.3:
            return ConditionPriority.POOR
        elif score >= 0.1:
            return ConditionPriority.CRITICAL
        else:
            return ConditionPriority.FAILED

    def _priority_to_score_min(self, priority: ConditionPriority) -> float:
        """Get minimum score for a priority level."""
        mapping = {
            ConditionPriority.EXCELLENT: 0.9,
            ConditionPriority.GOOD: 0.7,
            ConditionPriority.FAIR: 0.5,
            ConditionPriority.POOR: 0.3,
            ConditionPriority.CRITICAL: 0.1,
            ConditionPriority.FAILED: 0.0,
        }
        return mapping.get(priority, 0.5)

    # =========================================================================
    # OSCILLATION DETECTION
    # =========================================================================

    def _detect_oscillation(self, state: ConditionState) -> Tuple[bool, float]:
        """
        Detect oscillation in condition assessment history.

        Returns:
            Tuple of (is_oscillating, oscillation_score)
        """
        if len(state.history) < self.config.oscillation_window_count:
            return False, 0.0

        # Get recent changes
        recent = list(state.history)[-self.config.oscillation_window_count:]
        changes = []

        for i in range(1, len(recent)):
            _, _, prev_constrained = recent[i - 1]
            _, _, curr_constrained = recent[i]
            changes.append(curr_constrained - prev_constrained)

        if not changes:
            return False, 0.0

        # Count direction changes
        direction_changes = 0
        for i in range(1, len(changes)):
            if (changes[i] > 0 and changes[i - 1] < 0) or \
               (changes[i] < 0 and changes[i - 1] > 0):
                direction_changes += 1

        # Calculate oscillation score
        max_changes = len(changes) - 1
        if max_changes > 0:
            oscillation_score = direction_changes / max_changes
        else:
            oscillation_score = 0.0

        # Also consider magnitude of oscillations
        total_abs_change = sum(abs(c) for c in changes)
        if total_abs_change > 0:
            net_change = abs(sum(changes))
            oscillation_magnitude = 1.0 - (net_change / total_abs_change)
            oscillation_score = (oscillation_score + oscillation_magnitude) / 2

        is_oscillating = oscillation_score >= self.config.oscillation_threshold

        if is_oscillating:
            logger.warning(
                f"Oscillation detected for {state.asset_id}: "
                f"score={oscillation_score:.2f}, threshold={self.config.oscillation_threshold}"
            )

        return is_oscillating, oscillation_score

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def _activate_cooldown(
        self,
        state: ConditionState,
        reason: CooldownReason,
        duration_hours: Optional[float] = None,
        trigger_value: float = 0.0,
    ) -> CooldownEvent:
        """Activate cooldown period for a state."""
        now = datetime.now(timezone.utc)
        duration = duration_hours or self.config.cooldown_duration_hours
        expires_at = now + timedelta(hours=duration)

        state.cooldown_until = expires_at
        state.cooldown_reason = reason

        event = CooldownEvent(
            asset_id=state.asset_id,
            reason=reason,
            started_at=now,
            expires_at=expires_at,
            trigger_value=trigger_value,
            previous_value=state.current_score,
        )
        self._cooldown_events.append(event)

        logger.info(
            f"Cooldown activated for {state.asset_id}: "
            f"reason={reason.value}, expires={expires_at.isoformat()}"
        )

        return event

    def _check_priority_change(
        self,
        old_priority: ConditionPriority,
        new_priority: ConditionPriority,
    ) -> bool:
        """Check if priority change is significant (more than 1 level)."""
        priority_order = [
            ConditionPriority.FAILED,
            ConditionPriority.CRITICAL,
            ConditionPriority.POOR,
            ConditionPriority.FAIR,
            ConditionPriority.GOOD,
            ConditionPriority.EXCELLENT,
        ]
        try:
            old_idx = priority_order.index(old_priority)
            new_idx = priority_order.index(new_priority)
            return abs(new_idx - old_idx) > 1
        except ValueError:
            return False

    # =========================================================================
    # VELOCITY CONSTRAINT APPLICATION
    # =========================================================================

    def apply(
        self,
        asset_id: str,
        new_score: float,
        new_priority: Optional[ConditionPriority] = None,
        timestamp: Optional[datetime] = None,
        force_update: bool = False,
    ) -> VelocityCheckResult:
        """
        Apply velocity constraints to a new condition assessment.

        Args:
            asset_id: Insulation asset identifier
            new_score: New condition score [0, 1]
            new_priority: New priority level (derived from score if not provided)
            timestamp: Timestamp for this update (defaults to now)
            force_update: If True, bypass velocity limits (for initialization)

        Returns:
            VelocityCheckResult with constrained values and audit trail

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            timestamp = timestamp or datetime.now(timezone.utc)

            # Validate and clamp input score to valid range
            new_score = max(
                self.config.min_condition_score,
                min(self.config.max_condition_score, new_score)
            )

            # Derive priority if not provided
            if new_priority is None:
                new_priority = self._score_to_priority(new_score)

            # Get or create state
            state = self._get_or_create_state(asset_id, new_score)

            # First update - just record and return
            if state.last_update is None or force_update:
                state.current_score = new_score
                state.previous_score = new_score
                state.ema_score = new_score
                state.ramp_target = new_score
                state.current_priority = new_priority
                state.previous_priority = new_priority
                state.last_update = timestamp
                state.history.append((timestamp, new_score, new_score))

                return VelocityCheckResult(
                    asset_id=asset_id,
                    original_score=new_score,
                    constrained_score=new_score,
                    original_priority=new_priority,
                    constrained_priority=new_priority,
                    was_constrained=False,
                    violation_type=VelocityViolationType.NONE,
                    action_taken=ConstraintAction.PASSTHROUGH,
                    delta_requested=0.0,
                    delta_allowed=0.0,
                    delta_per_hour=0.0,
                    smoothed_value=new_score,
                    timestamp=timestamp,
                    elapsed_hours=0.0,
                )

            # Calculate elapsed time
            elapsed_seconds = max(
                (timestamp - state.last_update).total_seconds(),
                0.001
            )
            elapsed_hours = elapsed_seconds / 3600.0

            # Store previous values
            previous_score = state.current_score
            previous_priority = state.current_priority

            # Calculate requested change
            delta_requested = new_score - previous_score
            delta_per_hour = delta_requested / elapsed_hours if elapsed_hours > 0 else 0.0

            # Initialize constrained values
            constrained_score = new_score
            was_constrained = False
            violation_type = VelocityViolationType.NONE
            action_taken = ConstraintAction.PASSTHROUGH
            smoothed_value: Optional[float] = None
            in_cooldown = False
            cooldown_remaining_hours = 0.0
            oscillation_detected = False

            # Check cooldown first
            if self.config.enable_cooldown and state.is_in_cooldown(timestamp):
                in_cooldown = True
                cooldown_remaining_hours = (
                    state.cooldown_until - timestamp
                ).total_seconds() / 3600.0

                # During cooldown, hold at previous value
                constrained_score = previous_score
                was_constrained = True
                violation_type = VelocityViolationType.COOLDOWN_ACTIVE
                action_taken = ConstraintAction.HELD

                logger.debug(
                    f"Cooldown active for {asset_id}: "
                    f"holding at {previous_score:.3f}, "
                    f"remaining={cooldown_remaining_hours:.2f}hr"
                )

            else:
                # Apply velocity constraints in order

                # 1. Oscillation detection
                if self.config.enable_oscillation_detection:
                    is_oscillating, osc_score = self._detect_oscillation(state)
                    state.oscillation_score = osc_score

                    if is_oscillating:
                        oscillation_detected = True
                        # Dampen the change
                        dampening_factor = 1.0 - osc_score
                        constrained_score = (
                            previous_score +
                            delta_requested * dampening_factor
                        )
                        was_constrained = True
                        violation_type = VelocityViolationType.OSCILLATION_DETECTED
                        action_taken = ConstraintAction.DAMPENED

                        # Activate cooldown for oscillation
                        self._activate_cooldown(
                            state,
                            CooldownReason.OSCILLATION,
                            duration_hours=self.config.cooldown_duration_hours * 1.5,
                            trigger_value=new_score,
                        )

                # 2. Max delta clamping
                if self.config.enable_clamping and not in_cooldown:
                    max_delta = self.config.max_delta_per_hour * elapsed_hours

                    if abs(constrained_score - previous_score) > max_delta:
                        if constrained_score > previous_score:
                            constrained_score = previous_score + max_delta
                        else:
                            constrained_score = previous_score - max_delta

                        was_constrained = True
                        if violation_type == VelocityViolationType.NONE:
                            violation_type = VelocityViolationType.MAX_DELTA_EXCEEDED
                            action_taken = ConstraintAction.CLAMPED
                        else:
                            violation_type = VelocityViolationType.COMBINED

                # 3. EMA smoothing
                if self.config.enable_smoothing and not in_cooldown:
                    # Calculate alpha for exponential smoothing
                    alpha = min(
                        max(
                            1.0 - math.exp(-elapsed_hours / self.config.smoothing_window_hours),
                            0.01
                        ),
                        1.0
                    )
                    smoothed_value = alpha * constrained_score + (1 - alpha) * state.ema_score

                    # Only apply if it reduces the change
                    if abs(smoothed_value - previous_score) < abs(constrained_score - previous_score):
                        if not was_constrained:
                            action_taken = ConstraintAction.SMOOTHED
                            violation_type = VelocityViolationType.SMOOTHING_APPLIED

                        constrained_score = smoothed_value
                        was_constrained = True

                # 4. Ramp rate limiting
                if self.config.enable_ramping and not in_cooldown:
                    max_ramp_step = self.config.ramp_rate_per_hour * elapsed_hours
                    current_ramp_delta = constrained_score - previous_score

                    if abs(current_ramp_delta) > max_ramp_step:
                        if current_ramp_delta > 0:
                            constrained_score = previous_score + max_ramp_step
                        else:
                            constrained_score = previous_score - max_ramp_step

                        was_constrained = True
                        if violation_type == VelocityViolationType.NONE:
                            violation_type = VelocityViolationType.RAMP_RATE_EXCEEDED
                            action_taken = ConstraintAction.RAMPED
                        elif violation_type not in (
                            VelocityViolationType.COMBINED,
                            VelocityViolationType.RAMP_RATE_EXCEEDED
                        ):
                            violation_type = VelocityViolationType.COMBINED

            # Clamp final value to valid range
            constrained_score = max(
                self.config.min_condition_score,
                min(self.config.max_condition_score, constrained_score)
            )

            # Derive constrained priority
            constrained_priority = self._score_to_priority(constrained_score)

            # Check for significant priority change -> activate cooldown
            if (
                self.config.enable_cooldown and
                not in_cooldown and
                self._check_priority_change(previous_priority, new_priority)
            ):
                self._activate_cooldown(
                    state,
                    CooldownReason.PRIORITY_CHANGE,
                    duration_hours=self.config.priority_change_cooldown_hours,
                    trigger_value=new_score,
                )

            # Calculate allowed delta
            delta_allowed = constrained_score - previous_score
            constraint_ratio = (
                abs(delta_allowed) / abs(delta_requested)
                if abs(delta_requested) > 1e-10
                else 1.0
            )

            # Update state
            state.previous_score = previous_score
            state.current_score = constrained_score
            state.previous_priority = previous_priority
            state.current_priority = constrained_priority
            state.ema_score = smoothed_value if smoothed_value else constrained_score
            state.ramp_target = new_score  # Track intended target
            state.last_update = timestamp
            state.history.append((timestamp, new_score, constrained_score))

            # Track violations
            if was_constrained and violation_type in (
                VelocityViolationType.MAX_DELTA_EXCEEDED,
                VelocityViolationType.RAMP_RATE_EXCEEDED,
                VelocityViolationType.OSCILLATION_DETECTED,
            ):
                state.violation_count += 1
                if state.violation_count >= self.config.violation_threshold:
                    self._trigger_violation_alert(asset_id, state, violation_type)
            else:
                state.violation_count = 0

            # Build result
            result = VelocityCheckResult(
                asset_id=asset_id,
                original_score=new_score,
                constrained_score=constrained_score,
                original_priority=new_priority,
                constrained_priority=constrained_priority,
                was_constrained=was_constrained,
                violation_type=violation_type,
                action_taken=action_taken,
                delta_requested=delta_requested,
                delta_allowed=delta_allowed,
                delta_per_hour=delta_per_hour,
                smoothed_value=smoothed_value,
                in_cooldown=in_cooldown,
                cooldown_remaining_hours=cooldown_remaining_hours,
                oscillation_detected=oscillation_detected,
                constraint_ratio=constraint_ratio,
                timestamp=timestamp,
                elapsed_hours=elapsed_hours,
            )

            if was_constrained:
                logger.debug(
                    f"Velocity constrained for {asset_id}: "
                    f"{new_score:.3f} -> {constrained_score:.3f}, "
                    f"action={action_taken.value}, violation={violation_type.value}"
                )

            return result

    def apply_batch(
        self,
        assessments: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, VelocityCheckResult]:
        """
        Apply velocity constraints to multiple condition assessments.

        Args:
            assessments: Dict of asset_id -> condition score
            timestamp: Timestamp for all updates

        Returns:
            Dict of asset_id -> VelocityCheckResult
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        results = {}

        for asset_id, score in assessments.items():
            results[asset_id] = self.apply(
                asset_id=asset_id,
                new_score=score,
                timestamp=timestamp,
            )

        return results

    # =========================================================================
    # CALLBACKS AND ALERTS
    # =========================================================================

    def register_violation_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a callback for velocity violation alerts."""
        if callable(callback):
            self._violation_callbacks.append(callback)
            logger.debug("Registered velocity violation callback")

    def _trigger_violation_alert(
        self,
        asset_id: str,
        state: ConditionState,
        violation_type: VelocityViolationType,
    ) -> None:
        """Trigger violation alert callbacks."""
        alert_data = {
            "asset_id": asset_id,
            "violation_type": violation_type.value,
            "violation_count": state.violation_count,
            "current_score": state.current_score,
            "ema_score": state.ema_score,
            "oscillation_score": state.oscillation_score,
            "threshold": self.config.violation_threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.warning(
            f"Velocity violation threshold reached for {asset_id}: "
            f"type={violation_type.value}, count={state.violation_count}"
        )

        for callback in self._violation_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Violation callback failed: {e}")

    # =========================================================================
    # HISTORY AND STATISTICS
    # =========================================================================

    def get_history(
        self,
        asset_id: str,
        limit: Optional[int] = None,
    ) -> List[Tuple[datetime, float, float]]:
        """
        Get condition assessment history for an asset.

        Args:
            asset_id: Insulation asset identifier
            limit: Maximum entries to return

        Returns:
            List of (timestamp, raw_score, constrained_score) tuples
        """
        with self._lock:
            state = self.get_state(asset_id)
            if state is None:
                return []

            history = list(state.history)
            if limit:
                return history[-limit:]
            return history

    def get_cooldown_events(
        self,
        asset_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CooldownEvent]:
        """
        Get cooldown event history.

        Args:
            asset_id: Filter by asset (optional)
            limit: Maximum events to return

        Returns:
            List of CooldownEvent records
        """
        with self._lock:
            events = self._cooldown_events

            if asset_id:
                events = [e for e in events if e.asset_id == asset_id]

            return list(reversed(events[-limit:]))

    def get_statistics(
        self,
        asset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get velocity limiter statistics.

        Args:
            asset_id: Filter by asset (optional)

        Returns:
            Statistics dictionary
        """
        with self._lock:
            if asset_id:
                states = {
                    k: v for k, v in self._states.items()
                    if v.asset_id == asset_id
                }
            else:
                states = self._states

            # Calculate statistics
            total_tracked = len(states)
            in_cooldown = sum(1 for s in states.values() if s.is_in_cooldown())
            with_violations = sum(1 for s in states.values() if s.violation_count > 0)
            oscillating = sum(
                1 for s in states.values()
                if s.oscillation_score >= self.config.oscillation_threshold
            )
            total_history = sum(len(s.history) for s in states.values())

            # Average condition score
            if states:
                avg_score = sum(s.current_score for s in states.values()) / len(states)
            else:
                avg_score = 0.0

            return {
                "total_tracked_assets": total_tracked,
                "assets_in_cooldown": in_cooldown,
                "assets_with_violations": with_violations,
                "assets_oscillating": oscillating,
                "total_history_entries": total_history,
                "average_current_score": avg_score,
                "total_cooldown_events": len(self._cooldown_events),
                "config": {
                    "max_delta_per_hour": self.config.max_delta_per_hour,
                    "smoothing_window_hours": self.config.smoothing_window_hours,
                    "ramp_rate_per_hour": self.config.ramp_rate_per_hour,
                    "cooldown_duration_hours": self.config.cooldown_duration_hours,
                    "oscillation_threshold": self.config.oscillation_threshold,
                    "violation_threshold": self.config.violation_threshold,
                },
            }

    # =========================================================================
    # RESET AND MANAGEMENT
    # =========================================================================

    def reset(
        self,
        asset_id: Optional[str] = None,
    ) -> int:
        """
        Reset state for one or all assets.

        Args:
            asset_id: Specific asset to reset (None = all)

        Returns:
            Number of states reset
        """
        with self._lock:
            if asset_id is None:
                count = len(self._states)
                self._states.clear()
                logger.info(f"Reset all {count} velocity limiter states")
                return count

            key = self._get_state_key(asset_id)
            if key in self._states:
                del self._states[key]
                logger.info(f"Reset velocity limiter state for {asset_id}")
                return 1

            return 0

    def clear_cooldown(
        self,
        asset_id: str,
        reason: str = "manual_clear",
    ) -> bool:
        """
        Manually clear cooldown for an asset.

        Args:
            asset_id: Insulation asset identifier
            reason: Reason for clearing cooldown

        Returns:
            True if cooldown was cleared, False if not in cooldown
        """
        with self._lock:
            state = self.get_state(asset_id)
            if state is None or not state.is_in_cooldown():
                return False

            state.cooldown_until = None
            state.cooldown_reason = None

            logger.info(
                f"Cooldown cleared for {asset_id}: reason={reason}"
            )
            return True


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def check_condition_velocity(
    current_score: float,
    previous_score: float,
    elapsed_hours: float,
    max_delta_per_hour: float = 0.10,
) -> Tuple[bool, float, float]:
    """
    Quick check if a condition change exceeds velocity limits.

    Args:
        current_score: New condition score
        previous_score: Previous condition score
        elapsed_hours: Time elapsed in hours
        max_delta_per_hour: Maximum allowed change per hour

    Returns:
        Tuple of (is_violation, delta_per_hour, max_allowed_delta)
    """
    if elapsed_hours <= 0:
        elapsed_hours = 0.001

    delta = current_score - previous_score
    delta_per_hour = delta / elapsed_hours
    max_allowed_delta = max_delta_per_hour * elapsed_hours

    is_violation = abs(delta) > max_allowed_delta

    return is_violation, delta_per_hour, max_allowed_delta


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ConditionPriority",
    "VelocityViolationType",
    "ConstraintAction",
    "CooldownReason",
    # Config
    "InsulationVelocityLimiterConfig",
    # Data models
    "ConditionState",
    "VelocityCheckResult",
    "CooldownEvent",
    # Main class
    "InsulationVelocityLimiter",
    # Convenience function
    "check_condition_velocity",
]
