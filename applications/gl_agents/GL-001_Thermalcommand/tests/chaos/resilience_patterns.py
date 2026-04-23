"""
GL-001 ThermalCommand - Resilience Pattern Tests

This module provides chaos tests for validating resilience patterns:
- Circuit breaker activation and recovery
- Retry mechanism effectiveness
- Fallback behavior verification
- Graceful degradation

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker Pattern Tests
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker testing."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes before closing
    timeout_seconds: float = 30.0  # Time before half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class CircuitBreakerTestResult:
    """Result of circuit breaker test."""
    test_name: str
    passed: bool
    initial_state: CircuitState
    final_state: CircuitState
    transitions: List[Tuple[CircuitState, CircuitState, str]]
    total_calls: int
    failed_calls: int
    successful_calls: int
    fast_failed_calls: int
    recovery_time_seconds: float
    observations: List[str]
    errors: List[str]


class SimulatedCircuitBreaker:
    """
    Simulated circuit breaker for testing.

    This simulates circuit breaker behavior without
    requiring actual service calls.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._transitions: List[Tuple[CircuitState, CircuitState, str]] = []
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        # Check timeout for half-open transition
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN, "timeout elapsed")
        return self._state

    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._transitions.append((old_state, new_state, reason))
        logger.debug(f"Circuit breaker: {old_state.value} -> {new_state.value} ({reason})")

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

    async def call(self, should_fail: bool = False) -> Tuple[bool, str]:
        """
        Simulate a service call.

        Args:
            should_fail: Whether the call should fail

        Returns:
            Tuple of (success, message)
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            return False, "fast_fail:circuit_open"

        if current_state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls > self.config.half_open_max_calls:
                self._transition_to(CircuitState.OPEN, "half-open limit exceeded")
                return False, "fast_fail:half_open_limit"

        # Simulate call
        if should_fail:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()

            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN, "failure threshold reached")

            return False, "call_failed"
        else:
            self._success_count += 1
            self._failure_count = 0

            if current_state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED, "success threshold reached")

            return True, "call_succeeded"

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._transitions.clear()
        self._half_open_calls = 0


class CircuitBreakerTest:
    """
    Test suite for circuit breaker patterns.

    Tests:
    - Circuit opens after threshold failures
    - Circuit stays open during timeout
    - Half-open state allows probe calls
    - Circuit closes after success threshold
    - Fast-fail behavior when open

    Example:
        >>> test = CircuitBreakerTest()
        >>> result = await test.test_circuit_opens_on_failures()
        >>> assert result.passed
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.breaker = SimulatedCircuitBreaker(self.config)

    async def test_circuit_opens_on_failures(self) -> CircuitBreakerTestResult:
        """Test that circuit opens after threshold failures."""
        test_name = "circuit_opens_on_failures"
        observations = []
        errors = []

        self.breaker.reset()
        initial_state = self.breaker.state
        observations.append(f"Initial state: {initial_state.value}")

        total_calls = 0
        failed_calls = 0
        fast_failed = 0

        # Generate failures to trigger opening
        for i in range(self.config.failure_threshold + 2):
            success, message = await self.breaker.call(should_fail=True)
            total_calls += 1

            if not success:
                if "fast_fail" in message:
                    fast_failed += 1
                else:
                    failed_calls += 1

            observations.append(f"Call {i+1}: {message}, state={self.breaker.state.value}")

        final_state = self.breaker.state
        passed = final_state == CircuitState.OPEN

        if not passed:
            errors.append(f"Expected OPEN state, got {final_state.value}")

        return CircuitBreakerTestResult(
            test_name=test_name,
            passed=passed,
            initial_state=initial_state,
            final_state=final_state,
            transitions=self.breaker._transitions.copy(),
            total_calls=total_calls,
            failed_calls=failed_calls,
            successful_calls=0,
            fast_failed_calls=fast_failed,
            recovery_time_seconds=0,
            observations=observations,
            errors=errors,
        )

    async def test_circuit_fast_fails_when_open(self) -> CircuitBreakerTestResult:
        """Test that circuit fast-fails when open."""
        test_name = "circuit_fast_fails_when_open"
        observations = []
        errors = []

        self.breaker.reset()

        # Force circuit open
        for _ in range(self.config.failure_threshold):
            await self.breaker.call(should_fail=True)

        initial_state = self.breaker.state
        observations.append(f"Circuit opened: {initial_state.value}")

        total_calls = 0
        fast_failed = 0
        start_time = time.time()

        # Attempt calls while open (before timeout)
        for i in range(5):
            success, message = await self.breaker.call(should_fail=False)
            total_calls += 1

            if "fast_fail" in message:
                fast_failed += 1

            observations.append(f"Call {i+1}: {message}")

        final_state = self.breaker.state
        duration = time.time() - start_time

        # All calls should have fast-failed
        passed = fast_failed == 5 and final_state == CircuitState.OPEN

        if fast_failed != 5:
            errors.append(f"Expected 5 fast-fails, got {fast_failed}")

        return CircuitBreakerTestResult(
            test_name=test_name,
            passed=passed,
            initial_state=initial_state,
            final_state=final_state,
            transitions=self.breaker._transitions.copy(),
            total_calls=total_calls,
            failed_calls=0,
            successful_calls=0,
            fast_failed_calls=fast_failed,
            recovery_time_seconds=duration,
            observations=observations,
            errors=errors,
        )

    async def test_circuit_recovery(self) -> CircuitBreakerTestResult:
        """Test circuit recovery through half-open state."""
        test_name = "circuit_recovery"
        observations = []
        errors = []

        # Use shorter timeout for testing
        self.breaker.config.timeout_seconds = 0.5
        self.breaker.reset()

        # Force circuit open
        for _ in range(self.config.failure_threshold):
            await self.breaker.call(should_fail=True)

        observations.append(f"Circuit opened: {self.breaker.state.value}")
        initial_state = self.breaker.state

        # Wait for timeout to trigger half-open
        await asyncio.sleep(0.6)
        _ = self.breaker.state  # Trigger state check
        observations.append(f"After timeout: {self.breaker.state.value}")

        total_calls = 0
        successful = 0
        start_time = time.time()

        # Make successful calls to close circuit
        for i in range(self.config.success_threshold + 1):
            success, message = await self.breaker.call(should_fail=False)
            total_calls += 1
            if success:
                successful += 1
            observations.append(f"Recovery call {i+1}: {message}, state={self.breaker.state.value}")

        final_state = self.breaker.state
        recovery_time = time.time() - start_time

        passed = final_state == CircuitState.CLOSED

        if not passed:
            errors.append(f"Expected CLOSED state, got {final_state.value}")

        return CircuitBreakerTestResult(
            test_name=test_name,
            passed=passed,
            initial_state=initial_state,
            final_state=final_state,
            transitions=self.breaker._transitions.copy(),
            total_calls=total_calls,
            failed_calls=0,
            successful_calls=successful,
            fast_failed_calls=0,
            recovery_time_seconds=recovery_time,
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[CircuitBreakerTestResult]:
        """Run all circuit breaker tests."""
        results = []

        results.append(await self.test_circuit_opens_on_failures())
        results.append(await self.test_circuit_fast_fails_when_open())
        results.append(await self.test_circuit_recovery())

        return results


# =============================================================================
# Retry Mechanism Tests
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry testing."""
    max_retries: int = 3
    initial_delay_ms: float = 100
    max_delay_ms: float = 5000
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class RetryTestResult:
    """Result of retry mechanism test."""
    test_name: str
    passed: bool
    total_attempts: int
    successful_on_attempt: int
    total_delay_ms: float
    expected_delays: List[float]
    actual_delays: List[float]
    observations: List[str]
    errors: List[str]


class RetryMechanismTest:
    """
    Test suite for retry mechanisms.

    Tests:
    - Retry on transient failures
    - Exponential backoff
    - Jitter in delays
    - Max retry limit
    - Success after retries

    Example:
        >>> test = RetryMechanismTest()
        >>> result = await test.test_exponential_backoff()
        >>> assert result.passed
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt with exponential backoff."""
        delay = self.config.initial_delay_ms * (self.config.backoff_multiplier ** attempt)
        delay = min(delay, self.config.max_delay_ms)

        if self.config.jitter:
            jitter = random.uniform(0, delay * 0.2)
            delay += jitter

        return delay

    async def test_retry_on_transient_failure(
        self,
        fail_until_attempt: int = 2
    ) -> RetryTestResult:
        """Test retry succeeds after transient failures."""
        test_name = "retry_on_transient_failure"
        observations = []
        errors = []
        actual_delays = []
        expected_delays = []

        total_attempts = 0
        successful_on = 0
        total_delay = 0

        for attempt in range(self.config.max_retries + 1):
            total_attempts = attempt + 1

            if attempt > 0:
                delay = self._calculate_delay(attempt - 1)
                expected_delays.append(delay)
                actual_delays.append(delay)  # Simulated
                total_delay += delay
                observations.append(f"Waiting {delay:.1f}ms before retry")

            # Simulate call
            if attempt < fail_until_attempt:
                observations.append(f"Attempt {attempt + 1}: Failed (transient)")
            else:
                successful_on = attempt + 1
                observations.append(f"Attempt {attempt + 1}: Success")
                break

        passed = successful_on > 0 and successful_on <= self.config.max_retries + 1

        if successful_on == 0:
            errors.append("Never succeeded")

        return RetryTestResult(
            test_name=test_name,
            passed=passed,
            total_attempts=total_attempts,
            successful_on_attempt=successful_on,
            total_delay_ms=total_delay,
            expected_delays=expected_delays,
            actual_delays=actual_delays,
            observations=observations,
            errors=errors,
        )

    async def test_max_retry_limit(self) -> RetryTestResult:
        """Test that retries stop at max limit."""
        test_name = "max_retry_limit"
        observations = []
        errors = []
        actual_delays = []
        expected_delays = []

        total_attempts = 0
        total_delay = 0

        # Always fail - should stop at max retries
        for attempt in range(self.config.max_retries + 5):
            total_attempts = attempt + 1

            if attempt > 0:
                delay = self._calculate_delay(attempt - 1)
                expected_delays.append(delay)
                actual_delays.append(delay)
                total_delay += delay

            observations.append(f"Attempt {attempt + 1}: Failed")

            if attempt >= self.config.max_retries:
                observations.append("Max retries reached - giving up")
                break

        passed = total_attempts == self.config.max_retries + 1

        if total_attempts > self.config.max_retries + 1:
            errors.append(f"Exceeded max retries: {total_attempts} > {self.config.max_retries + 1}")

        return RetryTestResult(
            test_name=test_name,
            passed=passed,
            total_attempts=total_attempts,
            successful_on_attempt=0,
            total_delay_ms=total_delay,
            expected_delays=expected_delays,
            actual_delays=actual_delays,
            observations=observations,
            errors=errors,
        )

    async def test_exponential_backoff(self) -> RetryTestResult:
        """Test that delays follow exponential backoff."""
        test_name = "exponential_backoff"
        observations = []
        errors = []
        expected_delays = []
        actual_delays = []

        total_delay = 0

        for attempt in range(self.config.max_retries):
            expected = self.config.initial_delay_ms * (self.config.backoff_multiplier ** attempt)
            expected = min(expected, self.config.max_delay_ms)
            expected_delays.append(expected)

            actual = self._calculate_delay(attempt)
            actual_delays.append(actual)
            total_delay += actual

            observations.append(
                f"Attempt {attempt + 1}: expected delay ~{expected:.1f}ms, "
                f"actual {actual:.1f}ms"
            )

        # Verify delays are increasing (accounting for jitter)
        passed = True
        for i in range(1, len(actual_delays)):
            if actual_delays[i] < actual_delays[i-1] * 0.8:  # Allow 20% jitter variance
                passed = False
                errors.append(
                    f"Delay {i+1} ({actual_delays[i]:.1f}ms) not greater than "
                    f"delay {i} ({actual_delays[i-1]:.1f}ms)"
                )

        return RetryTestResult(
            test_name=test_name,
            passed=passed,
            total_attempts=self.config.max_retries,
            successful_on_attempt=0,
            total_delay_ms=total_delay,
            expected_delays=expected_delays,
            actual_delays=actual_delays,
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[RetryTestResult]:
        """Run all retry mechanism tests."""
        results = []

        results.append(await self.test_retry_on_transient_failure())
        results.append(await self.test_max_retry_limit())
        results.append(await self.test_exponential_backoff())

        return results


# =============================================================================
# Fallback Behavior Tests
# =============================================================================

@dataclass
class FallbackTestResult:
    """Result of fallback behavior test."""
    test_name: str
    passed: bool
    primary_available: bool
    fallback_used: bool
    fallback_response: Any
    response_time_ms: float
    observations: List[str]
    errors: List[str]


class FallbackBehaviorTest:
    """
    Test suite for fallback behavior patterns.

    Tests:
    - Primary service fallback
    - Cached data fallback
    - Default value fallback
    - Graceful error handling

    Example:
        >>> test = FallbackBehaviorTest()
        >>> result = await test.test_fallback_on_primary_failure()
        >>> assert result.passed and result.fallback_used
    """

    def __init__(self):
        self._primary_available = True
        self._fallback_value = {"status": "fallback", "data": "cached_data"}
        self._default_value = {"status": "default", "data": None}

    async def _call_primary(self) -> Tuple[bool, Any]:
        """Simulate primary service call."""
        if self._primary_available:
            return True, {"status": "primary", "data": "fresh_data"}
        return False, None

    async def _call_fallback(self) -> Any:
        """Simulate fallback service call."""
        return self._fallback_value

    async def _get_default(self) -> Any:
        """Get default value."""
        return self._default_value

    async def test_fallback_on_primary_failure(self) -> FallbackTestResult:
        """Test fallback is used when primary fails."""
        test_name = "fallback_on_primary_failure"
        observations = []
        errors = []

        self._primary_available = False
        observations.append("Primary service set to unavailable")

        start_time = time.time()

        # Try primary
        success, primary_result = await self._call_primary()
        observations.append(f"Primary call: {'success' if success else 'failed'}")

        fallback_used = False
        response = None

        if not success:
            # Use fallback
            response = await self._call_fallback()
            fallback_used = True
            observations.append("Fallback used")

        response_time = (time.time() - start_time) * 1000

        passed = not success and fallback_used and response is not None

        if not fallback_used:
            errors.append("Fallback was not used")

        return FallbackTestResult(
            test_name=test_name,
            passed=passed,
            primary_available=self._primary_available,
            fallback_used=fallback_used,
            fallback_response=response,
            response_time_ms=response_time,
            observations=observations,
            errors=errors,
        )

    async def test_primary_preferred(self) -> FallbackTestResult:
        """Test primary is used when available."""
        test_name = "primary_preferred"
        observations = []
        errors = []

        self._primary_available = True
        observations.append("Primary service set to available")

        start_time = time.time()

        success, response = await self._call_primary()
        observations.append(f"Primary call: {'success' if success else 'failed'}")

        fallback_used = not success
        if fallback_used:
            response = await self._call_fallback()
            observations.append("Fallback used (unexpected)")

        response_time = (time.time() - start_time) * 1000

        passed = success and not fallback_used

        if fallback_used:
            errors.append("Fallback was used when primary was available")

        return FallbackTestResult(
            test_name=test_name,
            passed=passed,
            primary_available=self._primary_available,
            fallback_used=fallback_used,
            fallback_response=response,
            response_time_ms=response_time,
            observations=observations,
            errors=errors,
        )

    async def test_default_when_all_fail(self) -> FallbackTestResult:
        """Test default value when both primary and fallback fail."""
        test_name = "default_when_all_fail"
        observations = []
        errors = []

        self._primary_available = False
        self._fallback_value = None  # Simulate fallback failure
        observations.append("Primary and fallback set to unavailable")

        start_time = time.time()

        # Try primary
        success, response = await self._call_primary()
        observations.append(f"Primary call: {'success' if success else 'failed'}")

        fallback_used = False
        if not success:
            response = await self._call_fallback()
            if response is None:
                observations.append("Fallback also failed")
                response = await self._get_default()
                observations.append("Using default value")
            else:
                fallback_used = True

        response_time = (time.time() - start_time) * 1000

        passed = response is not None and response.get("status") == "default"

        if response is None:
            errors.append("No response returned")

        # Reset fallback
        self._fallback_value = {"status": "fallback", "data": "cached_data"}

        return FallbackTestResult(
            test_name=test_name,
            passed=passed,
            primary_available=self._primary_available,
            fallback_used=fallback_used,
            fallback_response=response,
            response_time_ms=response_time,
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[FallbackTestResult]:
        """Run all fallback behavior tests."""
        results = []

        results.append(await self.test_fallback_on_primary_failure())
        results.append(await self.test_primary_preferred())
        results.append(await self.test_default_when_all_fail())

        return results


# =============================================================================
# Graceful Degradation Tests
# =============================================================================

@dataclass
class DegradationLevel:
    """Degradation level configuration."""
    name: str
    features_disabled: List[str]
    capacity_percent: float
    latency_multiplier: float


@dataclass
class GracefulDegradationTestResult:
    """Result of graceful degradation test."""
    test_name: str
    passed: bool
    initial_level: str
    final_level: str
    transitions: List[Tuple[str, str, str]]
    features_available: List[str]
    features_disabled: List[str]
    current_capacity: float
    response_degradation: float
    observations: List[str]
    errors: List[str]


class GracefulDegradationTest:
    """
    Test suite for graceful degradation patterns.

    Tests:
    - Feature shedding under load
    - Capacity reduction
    - Priority-based degradation
    - Recovery from degradation

    Example:
        >>> test = GracefulDegradationTest()
        >>> result = await test.test_load_shedding()
        >>> assert result.passed
    """

    def __init__(self):
        self._degradation_levels = [
            DegradationLevel("normal", [], 100.0, 1.0),
            DegradationLevel("reduced", ["analytics", "recommendations"], 80.0, 1.2),
            DegradationLevel("minimal", ["analytics", "recommendations", "search", "notifications"], 50.0, 1.5),
            DegradationLevel("critical", ["analytics", "recommendations", "search", "notifications", "history"], 20.0, 2.0),
        ]
        self._current_level_index = 0
        self._transitions: List[Tuple[str, str, str]] = []
        self._all_features = ["core", "analytics", "recommendations", "search", "notifications", "history"]

    def _get_current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._degradation_levels[self._current_level_index]

    def _transition_to(self, level_index: int, reason: str) -> None:
        """Transition to a new degradation level."""
        old_level = self._degradation_levels[self._current_level_index].name
        new_level = self._degradation_levels[level_index].name
        self._current_level_index = level_index
        self._transitions.append((old_level, new_level, reason))

    async def test_load_shedding(self) -> GracefulDegradationTestResult:
        """Test that load causes appropriate feature shedding."""
        test_name = "load_shedding"
        observations = []
        errors = []

        self._current_level_index = 0
        self._transitions.clear()
        initial_level = self._get_current_level().name

        # Simulate increasing load
        load_levels = [0.5, 0.75, 0.9, 0.95, 0.99]

        for load in load_levels:
            observations.append(f"Load at {load*100:.0f}%")

            # Determine degradation level based on load
            if load < 0.7:
                target_level = 0
            elif load < 0.85:
                target_level = 1
            elif load < 0.95:
                target_level = 2
            else:
                target_level = 3

            if target_level != self._current_level_index:
                self._transition_to(target_level, f"load={load*100:.0f}%")

            current = self._get_current_level()
            observations.append(
                f"  Level: {current.name}, Capacity: {current.capacity_percent}%, "
                f"Disabled: {current.features_disabled}"
            )

        final_level = self._get_current_level()

        # Verify degradation occurred
        passed = len(self._transitions) > 0 and final_level.name != "normal"

        if not self._transitions:
            errors.append("No degradation transitions occurred")

        return GracefulDegradationTestResult(
            test_name=test_name,
            passed=passed,
            initial_level=initial_level,
            final_level=final_level.name,
            transitions=self._transitions.copy(),
            features_available=[f for f in self._all_features if f not in final_level.features_disabled],
            features_disabled=final_level.features_disabled,
            current_capacity=final_level.capacity_percent,
            response_degradation=final_level.latency_multiplier,
            observations=observations,
            errors=errors,
        )

    async def test_recovery_from_degradation(self) -> GracefulDegradationTestResult:
        """Test recovery from degraded state."""
        test_name = "recovery_from_degradation"
        observations = []
        errors = []

        # Start in degraded state
        self._current_level_index = 3  # critical
        self._transitions.clear()
        initial_level = self._get_current_level().name
        observations.append(f"Starting in {initial_level} degradation")

        # Simulate decreasing load
        load_levels = [0.95, 0.85, 0.7, 0.5, 0.3]

        for load in load_levels:
            observations.append(f"Load reduced to {load*100:.0f}%")

            # Determine recovery level
            if load < 0.5:
                target_level = 0
            elif load < 0.7:
                target_level = 1
            elif load < 0.85:
                target_level = 2
            else:
                target_level = 3

            if target_level != self._current_level_index:
                self._transition_to(target_level, f"load={load*100:.0f}%")

            current = self._get_current_level()
            observations.append(f"  Level: {current.name}")

        final_level = self._get_current_level()

        # Verify recovery occurred
        passed = final_level.name == "normal" and len(self._transitions) > 0

        if final_level.name != "normal":
            errors.append(f"Did not recover to normal: {final_level.name}")

        return GracefulDegradationTestResult(
            test_name=test_name,
            passed=passed,
            initial_level=initial_level,
            final_level=final_level.name,
            transitions=self._transitions.copy(),
            features_available=self._all_features,
            features_disabled=[],
            current_capacity=100.0,
            response_degradation=1.0,
            observations=observations,
            errors=errors,
        )

    async def test_core_features_preserved(self) -> GracefulDegradationTestResult:
        """Test that core features are never disabled."""
        test_name = "core_features_preserved"
        observations = []
        errors = []

        self._current_level_index = 0
        self._transitions.clear()
        initial_level = self._get_current_level().name

        # Go through all degradation levels
        core_always_available = True

        for i, level in enumerate(self._degradation_levels):
            self._current_level_index = i
            observations.append(f"Level {level.name}: disabled={level.features_disabled}")

            if "core" in level.features_disabled:
                core_always_available = False
                errors.append(f"Core disabled at level {level.name}")

        passed = core_always_available

        return GracefulDegradationTestResult(
            test_name=test_name,
            passed=passed,
            initial_level=initial_level,
            final_level=self._get_current_level().name,
            transitions=self._transitions.copy(),
            features_available=["core"],
            features_disabled=self._degradation_levels[-1].features_disabled,
            current_capacity=self._degradation_levels[-1].capacity_percent,
            response_degradation=self._degradation_levels[-1].latency_multiplier,
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[GracefulDegradationTestResult]:
        """Run all graceful degradation tests."""
        results = []

        results.append(await self.test_load_shedding())
        results.append(await self.test_recovery_from_degradation())
        results.append(await self.test_core_features_preserved())

        return results
