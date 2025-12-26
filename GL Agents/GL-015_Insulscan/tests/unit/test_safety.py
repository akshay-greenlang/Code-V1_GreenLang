# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Safety Mechanism Unit Tests

Tests for safety mechanisms including:
- Velocity limiter for recommendation changes
- Circuit breaker for service protection
- Constraint validation
- Edge cases and error handling

Safety Reference:
- IEC 61508: Functional Safety
- Netflix Hystrix patterns for circuit breakers

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import asyncio
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch
from collections import deque


# =============================================================================
# ENUMERATIONS FOR SAFETY MECHANISMS
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing fast
    HALF_OPEN = "half_open"    # Testing recovery


class VelocityViolationType(Enum):
    """Types of velocity constraint violations."""
    NONE = "none"
    MAX_DELTA_EXCEEDED = "max_delta_exceeded"
    RAMP_RATE_EXCEEDED = "ramp_rate_exceeded"
    COOLDOWN_ACTIVE = "cooldown_active"


class ConstraintAction(Enum):
    """Actions taken by constraint validators."""
    PASSTHROUGH = "passthrough"
    CLAMPED = "clamped"
    REJECTED = "rejected"
    HELD = "held"


# =============================================================================
# VELOCITY LIMITER IMPLEMENTATION
# =============================================================================

@dataclass
class VelocityLimiterConfig:
    """Configuration for velocity limiter."""
    max_delta_per_hour: float = 10.0  # Maximum score change per hour
    ramp_rate_per_hour: float = 5.0   # Maximum ramp rate
    cooldown_hours: float = 1.0       # Cooldown after significant change
    min_value: float = 0.0
    max_value: float = 100.0


@dataclass
class VelocityLimiterState:
    """State tracking for velocity limiter."""
    asset_id: str
    current_value: float = 0.0
    previous_value: float = 0.0
    last_update: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    history: deque = field(default_factory=lambda: deque(maxlen=100))

    def is_in_cooldown(self, now: Optional[datetime] = None) -> bool:
        """Check if currently in cooldown."""
        if self.cooldown_until is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now < self.cooldown_until


@dataclass
class VelocityCheckResult:
    """Result of velocity check."""
    original_value: float
    constrained_value: float
    was_constrained: bool
    violation_type: VelocityViolationType
    action_taken: ConstraintAction
    delta_requested: float
    delta_allowed: float


class VelocityLimiter:
    """
    Limits the rate of change for condition scores and recommendations.

    Prevents rapid oscillations that could confuse operators or trigger
    unnecessary maintenance actions.
    """

    def __init__(self, config: Optional[VelocityLimiterConfig] = None):
        self.config = config or VelocityLimiterConfig()
        self._states: Dict[str, VelocityLimiterState] = {}

    def _get_or_create_state(self, asset_id: str) -> VelocityLimiterState:
        """Get or create state for an asset."""
        if asset_id not in self._states:
            self._states[asset_id] = VelocityLimiterState(asset_id=asset_id)
        return self._states[asset_id]

    def apply(
        self,
        asset_id: str,
        new_value: float,
        timestamp: Optional[datetime] = None,
        force: bool = False,
    ) -> VelocityCheckResult:
        """
        Apply velocity constraints to a new value.

        Args:
            asset_id: Asset identifier
            new_value: New value to apply
            timestamp: Timestamp for this update
            force: Force update without constraints

        Returns:
            VelocityCheckResult with constrained value and details
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        state = self._get_or_create_state(asset_id)

        # Clamp to valid range
        new_value = max(self.config.min_value, min(self.config.max_value, new_value))

        # First update or force - just record
        if state.last_update is None or force:
            state.current_value = new_value
            state.previous_value = new_value
            state.last_update = timestamp
            state.history.append((timestamp, new_value, new_value))

            return VelocityCheckResult(
                original_value=new_value,
                constrained_value=new_value,
                was_constrained=False,
                violation_type=VelocityViolationType.NONE,
                action_taken=ConstraintAction.PASSTHROUGH,
                delta_requested=0.0,
                delta_allowed=0.0,
            )

        # Calculate time elapsed
        elapsed_seconds = max((timestamp - state.last_update).total_seconds(), 0.001)
        elapsed_hours = elapsed_seconds / 3600.0

        # Store previous values
        previous_value = state.current_value
        delta_requested = new_value - previous_value

        # Initialize result
        constrained_value = new_value
        was_constrained = False
        violation_type = VelocityViolationType.NONE
        action_taken = ConstraintAction.PASSTHROUGH

        # Check cooldown
        if state.is_in_cooldown(timestamp):
            constrained_value = previous_value
            was_constrained = True
            violation_type = VelocityViolationType.COOLDOWN_ACTIVE
            action_taken = ConstraintAction.HELD

        else:
            # Apply max delta constraint
            max_delta = self.config.max_delta_per_hour * elapsed_hours

            if abs(delta_requested) > max_delta:
                if delta_requested > 0:
                    constrained_value = previous_value + max_delta
                else:
                    constrained_value = previous_value - max_delta

                was_constrained = True
                violation_type = VelocityViolationType.MAX_DELTA_EXCEEDED
                action_taken = ConstraintAction.CLAMPED

            # Apply ramp rate constraint
            max_ramp = self.config.ramp_rate_per_hour * elapsed_hours
            current_delta = constrained_value - previous_value

            if abs(current_delta) > max_ramp:
                if current_delta > 0:
                    constrained_value = previous_value + max_ramp
                else:
                    constrained_value = previous_value - max_ramp

                was_constrained = True
                if violation_type == VelocityViolationType.NONE:
                    violation_type = VelocityViolationType.RAMP_RATE_EXCEEDED
                action_taken = ConstraintAction.CLAMPED

        # Update state
        delta_allowed = constrained_value - previous_value
        state.previous_value = previous_value
        state.current_value = constrained_value
        state.last_update = timestamp
        state.history.append((timestamp, new_value, constrained_value))

        # Activate cooldown for significant changes
        if abs(delta_allowed) > self.config.max_delta_per_hour * 0.5:
            state.cooldown_until = timestamp + timedelta(hours=self.config.cooldown_hours)

        return VelocityCheckResult(
            original_value=new_value,
            constrained_value=constrained_value,
            was_constrained=was_constrained,
            violation_type=violation_type,
            action_taken=action_taken,
            delta_requested=delta_requested,
            delta_allowed=delta_allowed,
        )

    def reset(self, asset_id: Optional[str] = None) -> int:
        """Reset state for one or all assets."""
        if asset_id is None:
            count = len(self._states)
            self._states.clear()
            return count

        if asset_id in self._states:
            del self._states[asset_id]
            return 1
        return 0


# =============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 3           # Successes before closing
    timeout_seconds: float = 30.0        # Time in open state
    half_open_max_calls: int = 3         # Max calls in half-open


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    half_open_calls: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Protects against cascading failures by failing fast when
    a service is unhealthy.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._callbacks: List = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_timeout()
        return self._state.state

    def _check_timeout(self) -> None:
        """Check if timeout has expired for state transition."""
        if self._state.state != CircuitState.OPEN:
            return

        if self._state.last_state_change is None:
            return

        now = datetime.now(timezone.utc)
        elapsed = (now - self._state.last_state_change).total_seconds()

        if elapsed >= self.config.timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_state_change = datetime.now(timezone.utc)

        if new_state == CircuitState.HALF_OPEN:
            self._state.half_open_calls = 0
            self._state.success_count = 0

        if new_state == CircuitState.CLOSED:
            self._state.failure_count = 0

        for callback in self._callbacks:
            try:
                callback(old_state, new_state)
            except Exception:
                pass

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        state = self.state  # This checks timeout

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        if state == CircuitState.HALF_OPEN:
            return self._state.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state.state == CircuitState.HALF_OPEN:
            self._state.success_count += 1
            self._state.half_open_calls += 1

            if self._state.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

        elif self._state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self._state.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        now = datetime.now(timezone.utc)
        self._state.last_failure_time = now

        if self._state.state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens circuit
            self._transition_to(CircuitState.OPEN)

        elif self._state.state == CircuitState.CLOSED:
            self._state.failure_count += 1

            if self._state.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitBreakerState()

    def register_callback(self, callback) -> None:
        """Register state change callback."""
        self._callbacks.append(callback)


# =============================================================================
# CONSTRAINT VALIDATOR IMPLEMENTATION
# =============================================================================

@dataclass
class ConstraintValidatorConfig:
    """Configuration for constraint validator."""
    min_temperature_C: float = -273.15  # Absolute zero
    max_temperature_C: float = 1000.0   # Practical max
    min_thickness_m: float = 0.001      # 1mm minimum
    max_thickness_m: float = 1.0        # 1m maximum
    min_pipe_diameter_m: float = 0.01   # 10mm minimum
    max_pipe_diameter_m: float = 3.0    # 3m maximum
    min_k_value: float = 0.001          # Practical minimum
    max_k_value: float = 10.0           # Practical maximum


class ConstraintValidationError(Exception):
    """Raised when constraint validation fails."""
    def __init__(self, field: str, value: Any, constraint: str):
        self.field = field
        self.value = value
        self.constraint = constraint
        super().__init__(f"Constraint violation: {field}={value} - {constraint}")


class ConstraintValidator:
    """
    Validates physical constraints for insulation calculations.

    Ensures all input values are physically meaningful and within
    safe operating ranges.
    """

    def __init__(self, config: Optional[ConstraintValidatorConfig] = None):
        self.config = config or ConstraintValidatorConfig()

    def validate_temperature(self, temp_C: float, field_name: str = "temperature") -> float:
        """Validate temperature is within physical limits."""
        if math.isnan(temp_C) or math.isinf(temp_C):
            raise ConstraintValidationError(
                field_name, temp_C, "must be a finite number"
            )

        if temp_C < self.config.min_temperature_C:
            raise ConstraintValidationError(
                field_name, temp_C,
                f"cannot be below absolute zero ({self.config.min_temperature_C}C)"
            )

        if temp_C > self.config.max_temperature_C:
            raise ConstraintValidationError(
                field_name, temp_C,
                f"exceeds maximum ({self.config.max_temperature_C}C)"
            )

        return temp_C

    def validate_thickness(self, thickness_m: float, field_name: str = "thickness") -> float:
        """Validate insulation thickness."""
        if math.isnan(thickness_m) or math.isinf(thickness_m):
            raise ConstraintValidationError(
                field_name, thickness_m, "must be a finite number"
            )

        if thickness_m < self.config.min_thickness_m:
            raise ConstraintValidationError(
                field_name, thickness_m,
                f"must be at least {self.config.min_thickness_m}m"
            )

        if thickness_m > self.config.max_thickness_m:
            raise ConstraintValidationError(
                field_name, thickness_m,
                f"cannot exceed {self.config.max_thickness_m}m"
            )

        return thickness_m

    def validate_pipe_diameter(self, diameter_m: float, field_name: str = "diameter") -> float:
        """Validate pipe diameter."""
        if math.isnan(diameter_m) or math.isinf(diameter_m):
            raise ConstraintValidationError(
                field_name, diameter_m, "must be a finite number"
            )

        if diameter_m < self.config.min_pipe_diameter_m:
            raise ConstraintValidationError(
                field_name, diameter_m,
                f"must be at least {self.config.min_pipe_diameter_m}m"
            )

        if diameter_m > self.config.max_pipe_diameter_m:
            raise ConstraintValidationError(
                field_name, diameter_m,
                f"cannot exceed {self.config.max_pipe_diameter_m}m"
            )

        return diameter_m

    def validate_thermal_conductivity(self, k: float, field_name: str = "k") -> float:
        """Validate thermal conductivity."""
        if math.isnan(k) or math.isinf(k):
            raise ConstraintValidationError(
                field_name, k, "must be a finite number"
            )

        if k < self.config.min_k_value:
            raise ConstraintValidationError(
                field_name, k,
                f"must be at least {self.config.min_k_value} W/m-K"
            )

        if k > self.config.max_k_value:
            raise ConstraintValidationError(
                field_name, k,
                f"cannot exceed {self.config.max_k_value} W/m-K"
            )

        return k

    def validate_all(
        self,
        process_temp_C: float,
        ambient_temp_C: float,
        pipe_diameter_m: float,
        insulation_thickness_m: float,
        k_insulation: float,
    ) -> Dict[str, float]:
        """Validate all parameters for heat loss calculation."""
        return {
            "process_temp_C": self.validate_temperature(process_temp_C, "process_temp_C"),
            "ambient_temp_C": self.validate_temperature(ambient_temp_C, "ambient_temp_C"),
            "pipe_diameter_m": self.validate_pipe_diameter(pipe_diameter_m, "pipe_diameter_m"),
            "insulation_thickness_m": self.validate_thickness(insulation_thickness_m, "insulation_thickness_m"),
            "k_insulation": self.validate_thermal_conductivity(k_insulation, "k_insulation"),
        }


# =============================================================================
# TEST CLASS: VELOCITY LIMITER
# =============================================================================

class TestVelocityLimiter:
    """Test suite for velocity limiter behavior."""

    @pytest.fixture
    def limiter(self) -> VelocityLimiter:
        """Create velocity limiter with default config."""
        return VelocityLimiter(VelocityLimiterConfig(
            max_delta_per_hour=10.0,
            ramp_rate_per_hour=5.0,
            cooldown_hours=1.0,
        ))

    @pytest.mark.unit
    def test_first_value_passthrough(self, limiter):
        """Test first value passes through without constraints."""
        result = limiter.apply("ASSET-001", 50.0)

        assert not result.was_constrained
        assert result.constrained_value == 50.0
        assert result.violation_type == VelocityViolationType.NONE
        assert result.action_taken == ConstraintAction.PASSTHROUGH

    @pytest.mark.unit
    def test_small_change_allowed(self, limiter):
        """Test small changes within limits are allowed."""
        now = datetime.now(timezone.utc)

        # Initial value
        limiter.apply("ASSET-001", 50.0, now)

        # Small change 1 hour later (well within 10/hour limit)
        later = now + timedelta(hours=1)
        result = limiter.apply("ASSET-001", 55.0, later)

        assert not result.was_constrained
        assert result.constrained_value == 55.0

    @pytest.mark.unit
    def test_large_change_clamped(self, limiter):
        """Test large changes exceeding limits are clamped."""
        now = datetime.now(timezone.utc)

        # Initial value
        limiter.apply("ASSET-001", 50.0, now)

        # Large change 1 hour later (exceeds 10/hour limit)
        later = now + timedelta(hours=1)
        result = limiter.apply("ASSET-001", 75.0, later)  # 25 point change

        assert result.was_constrained
        assert result.constrained_value == 60.0  # Clamped to +10
        assert result.violation_type == VelocityViolationType.MAX_DELTA_EXCEEDED
        assert result.action_taken == ConstraintAction.CLAMPED

    @pytest.mark.unit
    def test_negative_change_clamped(self, limiter):
        """Test negative changes are also clamped."""
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 80.0, now)

        later = now + timedelta(hours=1)
        result = limiter.apply("ASSET-001", 50.0, later)  # -30 point change

        assert result.was_constrained
        assert result.constrained_value == 70.0  # Clamped to -10
        assert result.violation_type == VelocityViolationType.MAX_DELTA_EXCEEDED

    @pytest.mark.unit
    def test_proportional_to_time(self, limiter):
        """Test constraints are proportional to elapsed time."""
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 50.0, now)

        # Half hour later - max delta is 5
        later = now + timedelta(minutes=30)
        result = limiter.apply("ASSET-001", 70.0, later)

        assert result.was_constrained
        assert result.constrained_value == pytest.approx(55.0, rel=0.01)

    @pytest.mark.unit
    def test_cooldown_period(self, limiter):
        """Test cooldown period holds values."""
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 50.0, now)

        # Make a significant change that triggers cooldown
        t1 = now + timedelta(hours=1)
        limiter.apply("ASSET-001", 60.0, t1)  # +10, triggers cooldown

        # Immediate attempt should be held
        t2 = t1 + timedelta(minutes=1)
        result = limiter.apply("ASSET-001", 80.0, t2)

        assert result.was_constrained
        assert result.constrained_value == 60.0  # Held at previous
        assert result.violation_type == VelocityViolationType.COOLDOWN_ACTIVE
        assert result.action_taken == ConstraintAction.HELD

    @pytest.mark.unit
    def test_cooldown_expires(self, limiter):
        """Test changes allowed after cooldown expires."""
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 50.0, now)

        t1 = now + timedelta(hours=1)
        limiter.apply("ASSET-001", 60.0, t1)

        # Wait for cooldown to expire (1 hour)
        t2 = t1 + timedelta(hours=1.5)
        result = limiter.apply("ASSET-001", 65.0, t2)

        # Should not be in cooldown anymore
        assert result.violation_type != VelocityViolationType.COOLDOWN_ACTIVE

    @pytest.mark.unit
    def test_force_bypasses_constraints(self, limiter):
        """Test force flag bypasses all constraints."""
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 50.0, now)

        later = now + timedelta(minutes=1)
        result = limiter.apply("ASSET-001", 100.0, later, force=True)

        assert not result.was_constrained
        assert result.constrained_value == 100.0

    @pytest.mark.unit
    def test_value_clamped_to_range(self, limiter):
        """Test values are clamped to valid range."""
        result1 = limiter.apply("ASSET-001", -50.0)
        assert result1.constrained_value == 0.0

        result2 = limiter.apply("ASSET-002", 150.0)
        assert result2.constrained_value == 100.0

    @pytest.mark.unit
    def test_reset_clears_state(self, limiter):
        """Test reset clears asset state."""
        limiter.apply("ASSET-001", 50.0)
        limiter.apply("ASSET-002", 60.0)

        count = limiter.reset("ASSET-001")
        assert count == 1

        # ASSET-001 should be fresh
        result = limiter.apply("ASSET-001", 90.0)
        assert not result.was_constrained  # First value always passes

    @pytest.mark.unit
    def test_reset_all(self, limiter):
        """Test reset all clears all states."""
        limiter.apply("ASSET-001", 50.0)
        limiter.apply("ASSET-002", 60.0)
        limiter.apply("ASSET-003", 70.0)

        count = limiter.reset()
        assert count == 3

    @pytest.mark.unit
    def test_multiple_assets_independent(self, limiter):
        """Test different assets are tracked independently."""
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 50.0, now)
        limiter.apply("ASSET-002", 30.0, now)

        later = now + timedelta(hours=1)

        result1 = limiter.apply("ASSET-001", 70.0, later)  # +20
        result2 = limiter.apply("ASSET-002", 35.0, later)  # +5

        assert result1.was_constrained  # Exceeds limit
        assert not result2.was_constrained  # Within limit


# =============================================================================
# TEST CLASS: CIRCUIT BREAKER
# =============================================================================

class TestCircuitBreaker:
    """Test suite for circuit breaker state transitions."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create circuit breaker with test config."""
        return CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=5.0,
            half_open_max_calls=2,
        ))

    @pytest.mark.unit
    def test_initial_state_closed(self, breaker):
        """Test initial state is closed."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute()

    @pytest.mark.unit
    def test_stays_closed_on_success(self, breaker):
        """Test stays closed with successful calls."""
        for _ in range(10):
            assert breaker.can_execute()
            breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.unit
    def test_opens_after_failure_threshold(self, breaker):
        """Test opens after reaching failure threshold."""
        for i in range(3):  # threshold is 3
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert not breaker.can_execute()

    @pytest.mark.unit
    def test_failure_count_resets_on_success(self, breaker):
        """Test failure count resets after success."""
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()  # Reset count
        breaker.record_failure()
        breaker.record_failure()

        # Still closed - count was reset
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.unit
    def test_transitions_to_half_open_after_timeout(self, breaker):
        """Test transitions to half-open after timeout."""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Simulate timeout
        breaker._state.last_state_change = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        )

        # Should now be half-open
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute()

    @pytest.mark.unit
    def test_half_open_closes_on_success(self, breaker):
        """Test half-open closes after success threshold."""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        # Simulate timeout to half-open
        breaker._state.last_state_change = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        )

        # Trigger state check
        _ = breaker.state

        # Record successes
        for _ in range(2):  # threshold is 2
            breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.unit
    def test_half_open_reopens_on_failure(self, breaker):
        """Test half-open reopens on any failure."""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        # Simulate timeout
        breaker._state.last_state_change = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        )
        _ = breaker.state

        assert breaker.state == CircuitState.HALF_OPEN

        # One failure reopens
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_half_open_limits_calls(self, breaker):
        """Test half-open limits concurrent calls."""
        # Open and transition to half-open
        for _ in range(3):
            breaker.record_failure()

        breaker._state.last_state_change = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        )
        _ = breaker.state

        # First two calls allowed
        assert breaker.can_execute()
        breaker._state.half_open_calls += 1
        assert breaker.can_execute()
        breaker._state.half_open_calls += 1

        # Third call blocked
        assert not breaker.can_execute()

    @pytest.mark.unit
    def test_reset_returns_to_closed(self, breaker):
        """Test reset returns to closed state."""
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute()

    @pytest.mark.unit
    def test_callback_on_state_change(self, breaker):
        """Test callbacks are invoked on state changes."""
        state_changes = []

        def callback(old_state, new_state):
            state_changes.append((old_state, new_state))

        breaker.register_callback(callback)

        # Trigger open
        for _ in range(3):
            breaker.record_failure()

        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)


# =============================================================================
# TEST CLASS: CONSTRAINT VALIDATOR
# =============================================================================

class TestConstraintValidator:
    """Test suite for constraint validation."""

    @pytest.fixture
    def validator(self) -> ConstraintValidator:
        """Create constraint validator with default config."""
        return ConstraintValidator()

    @pytest.mark.unit
    def test_valid_temperature(self, validator):
        """Test valid temperatures pass validation."""
        assert validator.validate_temperature(25.0) == 25.0
        assert validator.validate_temperature(-40.0) == -40.0
        assert validator.validate_temperature(500.0) == 500.0

    @pytest.mark.unit
    def test_temperature_below_absolute_zero(self, validator):
        """Test temperature below absolute zero is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_temperature(-300.0)

        assert "absolute zero" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_temperature_nan(self, validator):
        """Test NaN temperature is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_temperature(float('nan'))

        assert "finite" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_temperature_infinity(self, validator):
        """Test infinite temperature is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_temperature(float('inf'))

        assert "finite" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_valid_thickness(self, validator):
        """Test valid thicknesses pass validation."""
        assert validator.validate_thickness(0.05) == 0.05
        assert validator.validate_thickness(0.1) == 0.1
        assert validator.validate_thickness(0.5) == 0.5

    @pytest.mark.unit
    def test_thickness_too_small(self, validator):
        """Test too small thickness is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_thickness(0.0001)

        assert "at least" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_thickness_too_large(self, validator):
        """Test too large thickness is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_thickness(2.0)

        assert "exceed" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_valid_pipe_diameter(self, validator):
        """Test valid pipe diameters pass validation."""
        assert validator.validate_pipe_diameter(0.1) == 0.1
        assert validator.validate_pipe_diameter(0.5) == 0.5
        assert validator.validate_pipe_diameter(1.0) == 1.0

    @pytest.mark.unit
    def test_pipe_diameter_too_small(self, validator):
        """Test too small pipe diameter is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_pipe_diameter(0.005)

        assert "at least" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_valid_thermal_conductivity(self, validator):
        """Test valid k values pass validation."""
        assert validator.validate_thermal_conductivity(0.04) == 0.04
        assert validator.validate_thermal_conductivity(0.1) == 0.1
        assert validator.validate_thermal_conductivity(1.0) == 1.0

    @pytest.mark.unit
    def test_k_too_small(self, validator):
        """Test too small k is rejected."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_thermal_conductivity(0.0001)

        assert "at least" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_validate_all_valid(self, validator):
        """Test validate_all with valid parameters."""
        result = validator.validate_all(
            process_temp_C=175.0,
            ambient_temp_C=25.0,
            pipe_diameter_m=0.2,
            insulation_thickness_m=0.05,
            k_insulation=0.04,
        )

        assert result["process_temp_C"] == 175.0
        assert result["ambient_temp_C"] == 25.0
        assert result["pipe_diameter_m"] == 0.2
        assert result["insulation_thickness_m"] == 0.05
        assert result["k_insulation"] == 0.04

    @pytest.mark.unit
    def test_validate_all_fails_on_first_error(self, validator):
        """Test validate_all fails on first invalid parameter."""
        with pytest.raises(ConstraintValidationError) as exc_info:
            validator.validate_all(
                process_temp_C=-300.0,  # Invalid
                ambient_temp_C=25.0,
                pipe_diameter_m=0.2,
                insulation_thickness_m=0.05,
                k_insulation=0.04,
            )

        assert "process_temp_C" in str(exc_info.value)


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling in safety mechanisms."""

    @pytest.mark.unit
    def test_velocity_limiter_rapid_updates(self):
        """Test velocity limiter with very rapid updates."""
        limiter = VelocityLimiter()
        now = datetime.now(timezone.utc)

        limiter.apply("ASSET-001", 50.0, now)

        # Rapid updates (1 second apart)
        for i in range(10):
            t = now + timedelta(seconds=i + 1)
            result = limiter.apply("ASSET-001", 50.0 + i * 5, t)
            # Changes should be heavily constrained
            assert result.constrained_value < 60.0

    @pytest.mark.unit
    def test_circuit_breaker_concurrent_access(self):
        """Test circuit breaker behavior approximation for concurrent access."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        # Simulate concurrent failures
        for _ in range(10):
            if breaker.can_execute():
                breaker.record_failure()

        # Should have opened
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_constraint_validator_boundary_values(self):
        """Test constraint validator at boundary values."""
        validator = ConstraintValidator()

        # Test exact boundary values
        assert validator.validate_temperature(-273.15) == -273.15  # Absolute zero
        assert validator.validate_temperature(1000.0) == 1000.0    # Max

        assert validator.validate_thickness(0.001) == 0.001  # Min
        assert validator.validate_thickness(1.0) == 1.0      # Max

    @pytest.mark.unit
    def test_velocity_limiter_state_isolation(self):
        """Test velocity limiter states are isolated between assets."""
        limiter = VelocityLimiter()
        now = datetime.now(timezone.utc)

        # Set up different states
        limiter.apply("ASSET-001", 10.0, now)
        limiter.apply("ASSET-002", 90.0, now)

        # Verify independent
        state1 = limiter._get_or_create_state("ASSET-001")
        state2 = limiter._get_or_create_state("ASSET-002")

        assert state1.current_value == 10.0
        assert state2.current_value == 90.0


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestVelocityLimiter",
    "TestCircuitBreaker",
    "TestConstraintValidator",
    "TestEdgeCases",
    "VelocityLimiter",
    "CircuitBreaker",
    "ConstraintValidator",
    "ConstraintValidationError",
]
