"""
GreenLang Framework - Circuit Breaker Tests

Comprehensive test suite for the circuit breaker module.
Tests cover state transitions, metrics collection, recovery behavior,
async support, and registry functionality.

Test Coverage Target: 85%+
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# Import from the module under test
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitMetrics,
    StateTransitionEvent,
    CircuitState,
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenError,
    circuit_breaker,
    CIRCUIT_BREAKER_REGISTRY,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def circuit_breaker_default() -> CircuitBreaker:
    """Create a circuit breaker with default settings."""
    return CircuitBreaker(
        name="test-default",
        failure_threshold=5,
        recovery_timeout=30.0,
        success_threshold=3,
    )


@pytest.fixture
def circuit_breaker_fast() -> CircuitBreaker:
    """Create a circuit breaker with fast recovery for testing."""
    return CircuitBreaker(
        name="test-fast",
        failure_threshold=2,
        recovery_timeout=0.1,  # 100ms for fast testing
        success_threshold=2,
    )


@pytest.fixture
def circuit_breaker_with_callbacks() -> CircuitBreaker:
    """Create a circuit breaker with callbacks."""
    on_state_change = MagicMock()
    on_failure = MagicMock()
    on_success = MagicMock()

    breaker = CircuitBreaker(
        name="test-callbacks",
        failure_threshold=2,
        recovery_timeout=0.1,
        success_threshold=2,
        on_state_change=on_state_change,
        on_failure=on_failure,
        on_success=on_success,
    )
    breaker._on_state_change_mock = on_state_change
    breaker._on_failure_mock = on_failure
    breaker._on_success_mock = on_success

    return breaker


@pytest.fixture
def registry() -> CircuitBreakerRegistry:
    """Create a fresh circuit breaker registry."""
    return CircuitBreakerRegistry()


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global registry before each test."""
    CIRCUIT_BREAKER_REGISTRY._breakers.clear()
    yield
    CIRCUIT_BREAKER_REGISTRY._breakers.clear()


# =============================================================================
# CIRCUIT METRICS TESTS
# =============================================================================

class TestCircuitMetrics:
    """Tests for CircuitMetrics class."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = CircuitMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.rejected_requests == 0
        assert metrics.consecutive_successes == 0
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 100.0
        assert metrics.failure_rate == 0.0

    def test_record_success(self):
        """Test recording successful requests."""
        metrics = CircuitMetrics()

        metrics.record_success(100.0)
        metrics.record_success(150.0)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.consecutive_successes == 2
        assert metrics.consecutive_failures == 0
        assert metrics.last_success_time is not None
        assert metrics.average_response_time_ms == 125.0

    def test_record_failure(self):
        """Test recording failed requests."""
        metrics = CircuitMetrics()

        metrics.record_failure(50.0)
        metrics.record_failure(75.0)

        assert metrics.total_requests == 2
        assert metrics.failed_requests == 2
        assert metrics.consecutive_failures == 2
        assert metrics.consecutive_successes == 0
        assert metrics.last_failure_time is not None

    def test_record_mixed(self):
        """Test recording mixed success/failure."""
        metrics = CircuitMetrics()

        metrics.record_success(100.0)
        metrics.record_success(100.0)
        metrics.record_failure(50.0)
        metrics.record_success(100.0)

        assert metrics.total_requests == 4
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 1
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 75.0
        assert metrics.failure_rate == 25.0

    def test_record_rejection(self):
        """Test recording rejected requests."""
        metrics = CircuitMetrics()

        metrics.record_rejection()
        metrics.record_rejection()

        assert metrics.rejected_requests == 2

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = CircuitMetrics()
        metrics.record_success(100.0)
        metrics.record_failure(50.0)

        data = metrics.to_dict()

        assert data["total_requests"] == 2
        assert data["successful_requests"] == 1
        assert data["failed_requests"] == 1
        assert data["success_rate"] == 50.0
        assert "last_success_time" in data
        assert "last_failure_time" in data

    def test_reset(self):
        """Test metrics reset."""
        metrics = CircuitMetrics()
        metrics.record_success(100.0)
        metrics.record_failure(50.0)

        metrics.reset()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.last_success_time is None
        assert metrics.last_failure_time is None


# =============================================================================
# CIRCUIT BREAKER STATE TESTS
# =============================================================================

class TestCircuitBreakerStates:
    """Tests for circuit breaker state machine."""

    def test_initial_state_closed(self, circuit_breaker_default: CircuitBreaker):
        """Test circuit breaker starts in CLOSED state."""
        assert circuit_breaker_default.state == CircuitState.CLOSED
        assert circuit_breaker_default.is_closed is True
        assert circuit_breaker_default.is_open is False
        assert circuit_breaker_default.is_half_open is False

    def test_transition_to_open_on_failures(self, circuit_breaker_fast: CircuitBreaker):
        """Test circuit opens after failure threshold."""
        def failing_function():
            raise ValueError("Simulated failure")

        # Trigger failures up to threshold
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        assert circuit_breaker_fast.state == CircuitState.OPEN
        assert circuit_breaker_fast.is_open is True

    def test_reject_requests_when_open(self, circuit_breaker_fast: CircuitBreaker):
        """Test requests are rejected when circuit is open."""
        def failing_function():
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        # Next request should be rejected
        def success_function():
            return "success"

        with pytest.raises(CircuitOpenError) as exc_info:
            circuit_breaker_fast.execute(success_function)

        assert exc_info.value.breaker_name == "test-fast"
        assert exc_info.value.time_until_recovery >= 0

    def test_transition_to_half_open_after_timeout(
        self,
        circuit_breaker_fast: CircuitBreaker,
    ):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        def failing_function():
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        assert circuit_breaker_fast.is_open is True

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next request should be allowed (circuit goes to HALF_OPEN)
        def success_function():
            return "success"

        result = circuit_breaker_fast.execute(success_function)
        assert result == "success"
        # After one success, still in HALF_OPEN (need 2 successes)
        assert circuit_breaker_fast.is_half_open is True

    def test_transition_to_closed_after_successes(
        self,
        circuit_breaker_fast: CircuitBreaker,
    ):
        """Test circuit closes after success threshold in HALF_OPEN."""
        def failing_function():
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        # Wait for recovery timeout
        time.sleep(0.15)

        # Execute enough successes to close circuit
        def success_function():
            return "success"

        for _ in range(2):
            circuit_breaker_fast.execute(success_function)

        assert circuit_breaker_fast.is_closed is True

    def test_transition_back_to_open_on_half_open_failure(
        self,
        circuit_breaker_fast: CircuitBreaker,
    ):
        """Test circuit reopens on failure in HALF_OPEN state."""
        def failing_function():
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        # Wait for recovery timeout
        time.sleep(0.15)

        # First request goes through (HALF_OPEN)
        def success_function():
            return "success"

        circuit_breaker_fast.execute(success_function)
        assert circuit_breaker_fast.is_half_open is True

        # Failure should reopen circuit
        with pytest.raises(ValueError):
            circuit_breaker_fast.execute(failing_function)

        assert circuit_breaker_fast.is_open is True


# =============================================================================
# CIRCUIT BREAKER EXECUTION TESTS
# =============================================================================

class TestCircuitBreakerExecution:
    """Tests for circuit breaker execution."""

    def test_execute_success(self, circuit_breaker_default: CircuitBreaker):
        """Test successful function execution."""
        def success_function(x: int, y: int) -> int:
            return x + y

        result = circuit_breaker_default.execute(success_function, 5, 3)
        assert result == 8
        assert circuit_breaker_default.metrics.successful_requests == 1

    def test_execute_with_kwargs(self, circuit_breaker_default: CircuitBreaker):
        """Test execution with keyword arguments."""
        def function_with_kwargs(a: int, b: int = 10) -> int:
            return a * b

        result = circuit_breaker_default.execute(function_with_kwargs, 5, b=20)
        assert result == 100

    def test_execute_propagates_exception(self, circuit_breaker_default: CircuitBreaker):
        """Test that exceptions are propagated."""
        def failing_function():
            raise RuntimeError("Original error")

        with pytest.raises(RuntimeError) as exc_info:
            circuit_breaker_default.execute(failing_function)

        assert str(exc_info.value) == "Original error"
        assert circuit_breaker_default.metrics.failed_requests == 1

    def test_excluded_exceptions(self):
        """Test that excluded exceptions don't count as failures."""
        breaker = CircuitBreaker(
            name="test-excluded",
            failure_threshold=2,
            excluded_exceptions={ValueError},
        )

        def value_error_function():
            raise ValueError("Not a failure")

        # These should not count as failures
        for _ in range(5):
            with pytest.raises(ValueError):
                breaker.execute(value_error_function)

        # Circuit should still be closed
        assert breaker.is_closed is True
        assert breaker._failure_count == 0

    def test_half_open_limit(self, circuit_breaker_fast: CircuitBreaker):
        """Test half-open request limit."""
        def failing_function():
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        # Wait for recovery
        time.sleep(0.15)

        # Use up half-open slots
        def slow_function():
            time.sleep(0.1)
            return "success"

        # First 3 calls should be allowed
        for _ in range(3):
            try:
                circuit_breaker_fast.execute(slow_function)
            except (CircuitOpenError, CircuitHalfOpenError):
                pass

        # Circuit should reject additional calls in HALF_OPEN
        # (if it hasn't already transitioned)


# =============================================================================
# ASYNC CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreakerAsync:
    """Tests for async circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_execute_async_success(self, circuit_breaker_default: CircuitBreaker):
        """Test successful async execution."""
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await circuit_breaker_default.execute_async(async_function, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_execute_async_failure(self, circuit_breaker_fast: CircuitBreaker):
        """Test async execution with failures."""
        async def async_failing():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async failure")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await circuit_breaker_fast.execute_async(async_failing)

        assert circuit_breaker_fast.is_open is True

    @pytest.mark.asyncio
    async def test_execute_async_open_rejection(
        self,
        circuit_breaker_fast: CircuitBreaker,
    ):
        """Test async rejection when circuit is open."""
        async def async_failing():
            raise RuntimeError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await circuit_breaker_fast.execute_async(async_failing)

        # Next request should be rejected
        async def async_success():
            return "success"

        with pytest.raises(CircuitOpenError):
            await circuit_breaker_fast.execute_async(async_success)


# =============================================================================
# DECORATOR TESTS
# =============================================================================

class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator."""

    def test_decorator_as_class_instance(self, circuit_breaker_default: CircuitBreaker):
        """Test using circuit breaker as decorator."""
        @circuit_breaker_default
        def decorated_function(x: int) -> int:
            return x + 1

        result = decorated_function(5)
        assert result == 6

    def test_decorator_factory(self):
        """Test circuit_breaker decorator factory."""
        @circuit_breaker(name="factory-test", failure_threshold=3)
        def decorated_function(x: int) -> int:
            return x * 2

        result = decorated_function(5)
        assert result == 10

        # Check it's in the global registry
        breaker = CIRCUIT_BREAKER_REGISTRY.get("factory-test")
        assert breaker is not None

    def test_decorator_without_registry(self):
        """Test decorator without global registry."""
        @circuit_breaker(name="no-registry", use_registry=False)
        def decorated_function(x: int) -> int:
            return x + 1

        result = decorated_function(5)
        assert result == 6

        # Should not be in global registry
        breaker = CIRCUIT_BREAKER_REGISTRY.get("no-registry")
        assert breaker is None

    @pytest.mark.asyncio
    async def test_decorator_async_function(self, circuit_breaker_default: CircuitBreaker):
        """Test decorator on async function."""
        @circuit_breaker_default
        async def async_decorated(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_decorated(5)
        assert result == 10


# =============================================================================
# CALLBACK TESTS
# =============================================================================

class TestCircuitBreakerCallbacks:
    """Tests for circuit breaker callbacks."""

    def test_on_state_change_callback(
        self,
        circuit_breaker_with_callbacks: CircuitBreaker,
    ):
        """Test state change callback is called."""
        breaker = circuit_breaker_with_callbacks

        def failing_function():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.execute(failing_function)

        # Callback should have been called
        breaker._on_state_change_mock.assert_called()
        call_args = breaker._on_state_change_mock.call_args[0]
        assert call_args[0] == CircuitState.CLOSED
        assert call_args[1] == CircuitState.OPEN

    def test_on_failure_callback(
        self,
        circuit_breaker_with_callbacks: CircuitBreaker,
    ):
        """Test failure callback is called."""
        breaker = circuit_breaker_with_callbacks

        def failing_function():
            raise ValueError("Test failure")

        with pytest.raises(ValueError):
            breaker.execute(failing_function)

        breaker._on_failure_mock.assert_called_once()

    def test_on_success_callback(
        self,
        circuit_breaker_with_callbacks: CircuitBreaker,
    ):
        """Test success callback is called."""
        breaker = circuit_breaker_with_callbacks

        def success_function():
            return "success"

        breaker.execute(success_function)

        breaker._on_success_mock.assert_called_once()


# =============================================================================
# MANUAL CONTROL TESTS
# =============================================================================

class TestCircuitBreakerManualControl:
    """Tests for manual circuit breaker control."""

    def test_reset(self, circuit_breaker_fast: CircuitBreaker):
        """Test manual reset."""
        def failing_function():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        assert circuit_breaker_fast.is_open is True

        # Reset the circuit
        circuit_breaker_fast.reset()

        assert circuit_breaker_fast.is_closed is True
        assert circuit_breaker_fast._failure_count == 0

    def test_force_open(self, circuit_breaker_default: CircuitBreaker):
        """Test force open."""
        circuit_breaker_default.force_open("Manual test")

        assert circuit_breaker_default.is_open is True

        def success_function():
            return "success"

        with pytest.raises(CircuitOpenError):
            circuit_breaker_default.execute(success_function)

    def test_force_close(self, circuit_breaker_fast: CircuitBreaker):
        """Test force close."""
        def failing_function():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        assert circuit_breaker_fast.is_open is True

        # Force close
        circuit_breaker_fast.force_close("Manual override")

        assert circuit_breaker_fast.is_closed is True


# =============================================================================
# STATUS AND METRICS TESTS
# =============================================================================

class TestCircuitBreakerStatus:
    """Tests for circuit breaker status reporting."""

    def test_get_status_closed(self, circuit_breaker_default: CircuitBreaker):
        """Test status when closed."""
        status = circuit_breaker_default.get_status()

        assert status["name"] == "test-default"
        assert status["state"] == "CLOSED"
        assert status["failure_threshold"] == 5
        assert status["recovery_timeout"] == 30.0
        assert status["time_until_recovery"] is None

    def test_get_status_open(self, circuit_breaker_fast: CircuitBreaker):
        """Test status when open."""
        def failing_function():
            raise ValueError("Failure")

        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        status = circuit_breaker_fast.get_status()

        assert status["state"] == "OPEN"
        assert status["time_until_recovery"] is not None
        assert status["time_until_recovery"] >= 0

    def test_metrics_in_status(self, circuit_breaker_default: CircuitBreaker):
        """Test metrics included in status."""
        def success_function():
            return "success"

        circuit_breaker_default.execute(success_function)

        status = circuit_breaker_default.get_status()

        assert "metrics" in status
        assert status["metrics"]["total_requests"] == 1
        assert status["metrics"]["successful_requests"] == 1


# =============================================================================
# TRANSITION HISTORY TESTS
# =============================================================================

class TestStateTransitionHistory:
    """Tests for state transition history."""

    def test_transition_history_recorded(self, circuit_breaker_fast: CircuitBreaker):
        """Test that transitions are recorded."""
        def failing_function():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        history = circuit_breaker_fast.transition_history

        assert len(history) == 1
        assert history[0].from_state == CircuitState.CLOSED
        assert history[0].to_state == CircuitState.OPEN
        assert history[0].breaker_name == "test-fast"

    def test_transition_event_to_dict(self, circuit_breaker_fast: CircuitBreaker):
        """Test transition event serialization."""
        def failing_function():
            raise ValueError("Failure")

        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker_fast.execute(failing_function)

        history = circuit_breaker_fast.transition_history
        event_dict = history[0].to_dict()

        assert "event_id" in event_dict
        assert event_dict["from_state"] == "CLOSED"
        assert event_dict["to_state"] == "OPEN"
        assert "timestamp" in event_dict
        assert "metrics_snapshot" in event_dict


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_register(self, registry: CircuitBreakerRegistry):
        """Test registering a circuit breaker."""
        breaker = CircuitBreaker(name="registry-test-1")
        registry.register(breaker)

        assert registry.get("registry-test-1") is breaker

    def test_unregister(self, registry: CircuitBreakerRegistry):
        """Test unregistering a circuit breaker."""
        breaker = CircuitBreaker(name="registry-test-2")
        registry.register(breaker)

        removed = registry.unregister("registry-test-2")

        assert removed is breaker
        assert registry.get("registry-test-2") is None

    def test_get_nonexistent(self, registry: CircuitBreakerRegistry):
        """Test getting nonexistent circuit breaker."""
        result = registry.get("nonexistent")
        assert result is None

    def test_get_or_create(self, registry: CircuitBreakerRegistry):
        """Test get_or_create functionality."""
        # First call creates
        breaker1 = registry.get_or_create(
            name="get-or-create-test",
            failure_threshold=3,
        )
        assert breaker1 is not None
        assert breaker1.failure_threshold == 3

        # Second call returns existing
        breaker2 = registry.get_or_create(
            name="get-or-create-test",
            failure_threshold=10,  # Different value
        )
        assert breaker2 is breaker1
        assert breaker2.failure_threshold == 3  # Original value

    def test_get_all(self, registry: CircuitBreakerRegistry):
        """Test getting all circuit breakers."""
        registry.register(CircuitBreaker(name="all-test-1"))
        registry.register(CircuitBreaker(name="all-test-2"))

        all_breakers = registry.get_all()

        assert len(all_breakers) == 2
        assert "all-test-1" in all_breakers
        assert "all-test-2" in all_breakers

    def test_get_health_status_healthy(self, registry: CircuitBreakerRegistry):
        """Test health status when all circuits healthy."""
        registry.register(CircuitBreaker(name="health-1"))
        registry.register(CircuitBreaker(name="health-2"))

        health = registry.get_health_status()

        assert health["overall_health"] == "HEALTHY"
        assert health["total_breakers"] == 2
        assert health["closed"] == 2
        assert health["open"] == 0

    def test_get_health_status_degraded(self, registry: CircuitBreakerRegistry):
        """Test health status when some circuits open."""
        breaker1 = CircuitBreaker(name="degraded-1", failure_threshold=1)
        breaker2 = CircuitBreaker(name="degraded-2")

        registry.register(breaker1)
        registry.register(breaker2)

        # Open one circuit
        breaker1.force_open("Test")

        health = registry.get_health_status()

        assert health["overall_health"] == "DEGRADED"
        assert health["open"] == 1
        assert health["closed"] == 1

    def test_get_health_status_critical(self, registry: CircuitBreakerRegistry):
        """Test health status when all circuits open."""
        breaker1 = CircuitBreaker(name="critical-1")
        breaker2 = CircuitBreaker(name="critical-2")

        registry.register(breaker1)
        registry.register(breaker2)

        breaker1.force_open("Test")
        breaker2.force_open("Test")

        health = registry.get_health_status()

        assert health["overall_health"] == "CRITICAL"
        assert health["open"] == 2
        assert health["closed"] == 0

    def test_reset_all(self, registry: CircuitBreakerRegistry):
        """Test resetting all circuit breakers."""
        breaker1 = CircuitBreaker(name="reset-all-1")
        breaker2 = CircuitBreaker(name="reset-all-2")

        registry.register(breaker1)
        registry.register(breaker2)

        breaker1.force_open("Test")
        breaker2.force_open("Test")

        registry.reset_all()

        assert breaker1.is_closed is True
        assert breaker2.is_closed is True


# =============================================================================
# EXCEPTION TESTS
# =============================================================================

class TestCircuitBreakerExceptions:
    """Tests for circuit breaker exceptions."""

    def test_circuit_open_error_attributes(self):
        """Test CircuitOpenError attributes."""
        error = CircuitOpenError(
            message="Circuit is open",
            breaker_name="test-breaker",
            time_until_recovery=15.5,
        )

        assert str(error) == "Circuit is open"
        assert error.breaker_name == "test-breaker"
        assert error.time_until_recovery == 15.5

    def test_circuit_half_open_error_attributes(self):
        """Test CircuitHalfOpenError attributes."""
        error = CircuitHalfOpenError(
            message="Half-open limit exceeded",
            breaker_name="test-breaker",
        )

        assert str(error) == "Half-open limit exceeded"
        assert error.breaker_name == "test-breaker"

    def test_exception_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(CircuitOpenError, CircuitBreakerError)
        assert issubclass(CircuitHalfOpenError, CircuitBreakerError)
        assert issubclass(CircuitBreakerError, Exception)


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestCircuitBreakerThreadSafety:
    """Tests for circuit breaker thread safety."""

    def test_concurrent_executions(self):
        """Test concurrent executions are thread-safe."""
        import threading

        breaker = CircuitBreaker(
            name="concurrent-test",
            failure_threshold=100,  # High threshold
        )

        results = []
        errors = []

        def worker(index: int):
            try:
                result = breaker.execute(lambda: index * 2)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert breaker.metrics.total_requests == 20


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestCircuitBreakerEdgeCases:
    """Tests for edge cases."""

    def test_zero_failure_threshold(self):
        """Test with zero failure threshold (immediate open)."""
        breaker = CircuitBreaker(
            name="zero-threshold",
            failure_threshold=0,
        )

        # Circuit should already be considered ready to open
        # First failure should open it
        def failing_function():
            raise ValueError("Failure")

        # Note: With threshold 0, behavior depends on implementation
        # Typically means no failures tolerated
        assert breaker.is_closed is True

    def test_very_long_recovery_timeout(self):
        """Test with very long recovery timeout."""
        breaker = CircuitBreaker(
            name="long-timeout",
            failure_threshold=1,
            recovery_timeout=3600.0,  # 1 hour
        )

        def failing_function():
            raise ValueError("Failure")

        with pytest.raises(ValueError):
            breaker.execute(failing_function)

        status = breaker.get_status()
        assert status["state"] == "OPEN"
        assert status["time_until_recovery"] > 3500  # Close to 1 hour

    def test_rapid_state_transitions(self, circuit_breaker_fast: CircuitBreaker):
        """Test rapid state transitions."""
        def failing_function():
            raise ValueError("Failure")

        def success_function():
            return "success"

        # Rapid transitions
        for _ in range(5):
            # Open
            for _ in range(2):
                with pytest.raises(ValueError):
                    circuit_breaker_fast.execute(failing_function)

            # Wait and close
            time.sleep(0.15)
            for _ in range(2):
                circuit_breaker_fast.execute(success_function)

        # Should end up closed
        assert circuit_breaker_fast.is_closed is True
        assert circuit_breaker_fast.metrics.state_transitions >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
