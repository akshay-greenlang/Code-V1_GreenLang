"""
GL-002 FLAMEGUARD - Circuit Breaker Unit Tests

Comprehensive test suite for circuit breaker implementation covering:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Timeout behavior
- Recovery patterns
- Metrics collection
- Configuration validation
- Edge cases

Test Coverage Target: 85%+
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
    CircuitBreakerState,
    CircuitState,
    CircuitOpenError,
    CircuitHalfOpenError,
    circuit_breaker,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> CircuitBreakerConfig:
    """Default circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_s=1.0,  # Short for testing
        half_open_max_calls=2,
        success_threshold=2,
    )


@pytest.fixture
def breaker(default_config) -> CircuitBreaker:
    """Create a circuit breaker for testing."""
    return CircuitBreaker("test_breaker", config=default_config)


@pytest.fixture
def registry() -> CircuitBreakerRegistry:
    """Create a fresh registry for testing."""
    CircuitBreakerRegistry.reset_instance()
    return CircuitBreakerRegistry()


@pytest.fixture
async def mock_success_func() -> AsyncMock:
    """Mock async function that always succeeds."""
    mock = AsyncMock(return_value="success")
    return mock


@pytest.fixture
async def mock_failure_func() -> AsyncMock:
    """Mock async function that always fails."""
    mock = AsyncMock(side_effect=Exception("test failure"))
    return mock


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout_s == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.failure_rate_threshold == 0.5

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout_s=60.0,
            half_open_max_calls=5,
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout_s == 60.0
        assert config.half_open_max_calls == 5

    def test_invalid_failure_threshold(self):
        """Test validation of failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_recovery_timeout(self):
        """Test validation of recovery_timeout_s."""
        with pytest.raises(ValueError, match="recovery_timeout_s must be > 0"):
            CircuitBreakerConfig(recovery_timeout_s=-1.0)

    def test_invalid_failure_rate_threshold(self):
        """Test validation of failure_rate_threshold."""
        with pytest.raises(ValueError, match="failure_rate_threshold"):
            CircuitBreakerConfig(failure_rate_threshold=1.5)


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================

class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_is_closed(self, breaker):
        """Test that initial state is CLOSED."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_half_open

    @pytest.mark.asyncio
    async def test_transition_to_open_on_failures(self, breaker, mock_failure_func):
        """Test transition from CLOSED to OPEN after reaching failure threshold."""
        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception, match="test failure"):
                await breaker.call(mock_failure_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_open_circuit_raises_error(self, breaker, mock_failure_func, mock_success_func):
        """Test that open circuit raises CircuitOpenError."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_failure_func)

        # Try to make a call - should raise CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(mock_success_func)

        assert exc_info.value.breaker_name == "test_breaker"
        assert exc_info.value.time_until_retry > 0

    @pytest.mark.asyncio
    async def test_transition_to_half_open_after_timeout(self, breaker, mock_failure_func):
        """Test transition from OPEN to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_failure_func)

        assert breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # The next call should be allowed (transitions to HALF_OPEN)
        mock_success = AsyncMock(return_value="success")
        result = await breaker.call(mock_success)

        assert result == "success"
        # After success, should still be in HALF_OPEN or CLOSED
        assert breaker.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]

    @pytest.mark.asyncio
    async def test_transition_to_closed_on_recovery(self, breaker, mock_failure_func):
        """Test transition from HALF_OPEN to CLOSED on successful recovery."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_failure_func)

        await asyncio.sleep(1.1)

        # Make successful calls in half-open
        mock_success = AsyncMock(return_value="success")
        for _ in range(2):  # Need 2 successes to close
            await breaker.call(mock_success)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_reopen_on_failure_in_half_open(self, breaker, mock_failure_func):
        """Test that failure in HALF_OPEN reopens the circuit."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_failure_func)

        await asyncio.sleep(1.1)

        # Fail in half-open
        with pytest.raises(Exception):
            await breaker.call(mock_failure_func)

        assert breaker.state == CircuitState.OPEN


# =============================================================================
# TIMEOUT BEHAVIOR TESTS
# =============================================================================

class TestCircuitBreakerTimeouts:
    """Tests for timeout behavior."""

    @pytest.mark.asyncio
    async def test_recovery_timeout_respected(self, default_config):
        """Test that recovery timeout is respected."""
        breaker = CircuitBreaker(
            "timeout_test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_s=0.5,
            ),
        )

        # Open the circuit
        mock_fail = AsyncMock(side_effect=Exception("fail"))
        with pytest.raises(Exception):
            await breaker.call(mock_fail)

        assert breaker.is_open

        # Immediate call should fail
        mock_success = AsyncMock(return_value="ok")
        with pytest.raises(CircuitOpenError):
            await breaker.call(mock_success)

        # Wait partial timeout
        await asyncio.sleep(0.3)
        with pytest.raises(CircuitOpenError):
            await breaker.call(mock_success)

        # Wait remaining timeout
        await asyncio.sleep(0.3)
        result = await breaker.call(mock_success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_time_until_retry_accuracy(self, breaker, mock_failure_func):
        """Test time_until_retry calculation accuracy."""
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_failure_func)

        assert breaker.is_open

        # Get initial time until retry
        status = breaker.get_status()
        initial_time = status["time_until_retry_s"]
        assert 0 < initial_time <= 1.0

        # Wait a bit
        await asyncio.sleep(0.3)

        # Time should decrease
        status = breaker.get_status()
        new_time = status["time_until_retry_s"]
        assert new_time < initial_time


# =============================================================================
# RECOVERY PATTERN TESTS
# =============================================================================

class TestCircuitBreakerRecovery:
    """Tests for recovery patterns."""

    @pytest.mark.asyncio
    async def test_gradual_recovery_pattern(self):
        """Test gradual recovery with limited half-open calls."""
        breaker = CircuitBreaker(
            "recovery_test",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout_s=0.1,
                half_open_max_calls=2,
                success_threshold=2,
            ),
        )

        # Open circuit
        mock_fail = AsyncMock(side_effect=Exception("fail"))
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_fail)

        await asyncio.sleep(0.15)

        # Make successful calls in half-open
        mock_success = AsyncMock(return_value="ok")
        await breaker.call(mock_success)
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.call(mock_success)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self):
        """Test that half-open max calls limit is enforced."""
        breaker = CircuitBreaker(
            "half_open_limit",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_s=0.1,
                half_open_max_calls=1,
                success_threshold=2,
            ),
        )

        # Open circuit
        mock_fail = AsyncMock(side_effect=Exception("fail"))
        with pytest.raises(Exception):
            await breaker.call(mock_fail)

        await asyncio.sleep(0.15)

        # First call allowed
        mock_success = AsyncMock(return_value="ok")
        await breaker.call(mock_success)

        # Second call should be rejected (max calls reached, success threshold not met)
        with pytest.raises(CircuitHalfOpenError):
            await breaker.call(mock_success)


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestCircuitBreakerMetrics:
    """Tests for metrics collection."""

    def test_metrics_initial_state(self):
        """Test initial metrics state."""
        metrics = CircuitBreakerMetrics()
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.rejected_calls == 0

    def test_record_success(self):
        """Test recording successful calls."""
        metrics = CircuitBreakerMetrics()
        metrics.record_success(0.1)

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.consecutive_successes == 1
        assert metrics.last_success_time is not None

    def test_record_failure(self):
        """Test recording failed calls."""
        metrics = CircuitBreakerMetrics()
        metrics.record_failure(0.05)

        assert metrics.total_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.consecutive_failures == 1
        assert metrics.last_failure_time is not None

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        metrics = CircuitBreakerMetrics()

        # 3 successes, 2 failures = 40% failure rate
        metrics.record_success(0.1)
        metrics.record_success(0.1)
        metrics.record_failure(0.1)
        metrics.record_success(0.1)
        metrics.record_failure(0.1)

        rate = metrics.get_failure_rate(window_size=5)
        assert rate == 0.4

    @pytest.mark.asyncio
    async def test_breaker_metrics_integration(self, breaker):
        """Test metrics are collected during breaker operation."""
        mock_success = AsyncMock(return_value="ok")

        await breaker.call(mock_success)
        await breaker.call(mock_success)

        assert breaker.metrics.total_calls == 2
        assert breaker.metrics.successful_calls == 2

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = CircuitBreakerMetrics()
        metrics.record_success(0.1)
        metrics.record_failure(0.2)

        result = metrics.to_dict()

        assert "total_calls" in result
        assert "successful_calls" in result
        assert "failed_calls" in result
        assert "failure_rate" in result
        assert result["total_calls"] == 2


# =============================================================================
# DECORATOR AND CONTEXT MANAGER TESTS
# =============================================================================

class TestCircuitBreakerDecorator:
    """Tests for decorator usage."""

    @pytest.mark.asyncio
    async def test_decorator_basic_usage(self, breaker):
        """Test basic decorator usage."""
        @breaker
        async def protected_func():
            return "result"

        result = await protected_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_with_arguments(self, breaker):
        """Test decorator with function arguments."""
        @breaker
        async def protected_func(a, b, key=None):
            return f"{a}-{b}-{key}"

        result = await protected_func("x", "y", key="z")
        assert result == "x-y-z"

    @pytest.mark.asyncio
    async def test_decorator_factory(self):
        """Test circuit_breaker decorator factory."""
        @circuit_breaker("factory_test", failure_threshold=2)
        async def protected_func():
            return "ok"

        result = await protected_func()
        assert result == "ok"

        # Verify breaker was registered
        registry = CircuitBreakerRegistry()
        assert "factory_test" in registry._breakers


class TestCircuitBreakerContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, breaker):
        """Test context manager with successful operation."""
        async with breaker:
            result = "success"

        assert result == "success"
        assert breaker.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, breaker):
        """Test context manager with failed operation."""
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker.metrics.failed_calls == 1

    @pytest.mark.asyncio
    async def test_context_manager_circuit_open(self, breaker, mock_failure_func):
        """Test context manager when circuit is open."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_failure_func)

        with pytest.raises(CircuitOpenError):
            async with breaker:
                pass


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        CircuitBreakerRegistry.reset_instance()
        reg1 = CircuitBreakerRegistry()
        reg2 = CircuitBreakerRegistry()
        assert reg1 is reg2

    def test_register_and_get(self, registry):
        """Test registering and getting breakers."""
        breaker = CircuitBreaker("test")
        registry.register(breaker)

        retrieved = registry.get("test")
        assert retrieved is breaker

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent breaker."""
        result = registry.get("nonexistent")
        assert result is None

    def test_get_or_create(self, registry):
        """Test get_or_create functionality."""
        breaker1 = registry.get_or_create("new_breaker")
        breaker2 = registry.get_or_create("new_breaker")

        assert breaker1 is breaker2

    def test_remove(self, registry):
        """Test removing a breaker."""
        breaker = CircuitBreaker("to_remove")
        registry.register(breaker)
        registry.remove("to_remove")

        assert registry.get("to_remove") is None

    def test_get_all_status(self, registry):
        """Test getting status of all breakers."""
        registry.get_or_create("breaker1")
        registry.get_or_create("breaker2")

        status = registry.get_all_status()

        assert "breaker1" in status
        assert "breaker2" in status

    def test_get_open_breakers(self, registry):
        """Test getting open breakers."""
        breaker = registry.get_or_create(
            "open_test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        breaker.force_open()

        open_breakers = registry.get_open_breakers()
        assert "open_test" in open_breakers

    def test_health_summary(self, registry):
        """Test health summary."""
        registry.get_or_create("healthy")
        unhealthy = registry.get_or_create(
            "unhealthy",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        unhealthy.force_open()

        summary = registry.get_health_summary()

        assert summary["total_breakers"] == 2
        assert summary["closed"] == 1
        assert summary["open"] == 1

    def test_reset_all(self, registry):
        """Test resetting all breakers."""
        breaker = registry.get_or_create(
            "to_reset",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        breaker.force_open()

        registry.reset_all()

        assert breaker.is_closed


# =============================================================================
# CALLBACK TESTS
# =============================================================================

class TestCircuitBreakerCallbacks:
    """Tests for callback functionality."""

    @pytest.mark.asyncio
    async def test_on_state_change_callback(self):
        """Test state change callback is invoked."""
        state_changes = []

        def on_change(name, old, new):
            state_changes.append((name, old, new))

        breaker = CircuitBreaker(
            "callback_test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.1),
            on_state_change=on_change,
        )

        mock_fail = AsyncMock(side_effect=Exception("fail"))
        with pytest.raises(Exception):
            await breaker.call(mock_fail)

        assert len(state_changes) == 1
        assert state_changes[0][0] == "callback_test"
        assert state_changes[0][2] == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_on_failure_callback(self):
        """Test failure callback is invoked."""
        failures = []

        def on_failure(name, exc):
            failures.append((name, str(exc)))

        breaker = CircuitBreaker(
            "failure_callback",
            on_failure=on_failure,
        )

        mock_fail = AsyncMock(side_effect=ValueError("test error"))
        with pytest.raises(ValueError):
            await breaker.call(mock_fail)

        assert len(failures) == 1
        assert failures[0][0] == "failure_callback"
        assert "test error" in failures[0][1]

    @pytest.mark.asyncio
    async def test_on_success_callback(self):
        """Test success callback is invoked."""
        successes = []

        def on_success(name):
            successes.append(name)

        breaker = CircuitBreaker(
            "success_callback",
            on_success=on_success,
        )

        mock_success = AsyncMock(return_value="ok")
        await breaker.call(mock_success)

        assert len(successes) == 1
        assert successes[0] == "success_callback"


# =============================================================================
# FORCE STATE TESTS
# =============================================================================

class TestCircuitBreakerForceState:
    """Tests for force state methods."""

    def test_force_open(self, breaker):
        """Test force_open method."""
        assert breaker.is_closed
        breaker.force_open()
        assert breaker.is_open

    def test_force_close(self, breaker, mock_failure_func):
        """Test force_close method."""
        breaker.force_open()
        assert breaker.is_open
        breaker.force_close()
        assert breaker.is_closed

    def test_reset(self, breaker):
        """Test reset method."""
        breaker.force_open()
        breaker.metrics.total_calls = 100
        breaker.reset()

        assert breaker.is_closed
        assert breaker.metrics.total_calls == 0


# =============================================================================
# STATUS AND PROVENANCE TESTS
# =============================================================================

class TestCircuitBreakerStatus:
    """Tests for status and provenance methods."""

    def test_get_status(self, breaker):
        """Test get_status method."""
        status = breaker.get_status()

        assert status["name"] == "test_breaker"
        assert status["state"] == "closed"
        assert "config" in status
        assert "metrics" in status

    def test_provenance_hash(self, breaker):
        """Test provenance hash calculation."""
        hash1 = breaker.get_provenance_hash()
        assert len(hash1) == 64  # SHA-256 hex length

        # Hash should change with state
        breaker.force_open()
        hash2 = breaker.get_provenance_hash()
        assert hash1 != hash2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestCircuitBreakerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_calls(self, breaker):
        """Test concurrent calls through breaker."""
        mock_success = AsyncMock(return_value="ok")

        # Make multiple concurrent calls
        results = await asyncio.gather(*[
            breaker.call(mock_success)
            for _ in range(10)
        ])

        assert all(r == "ok" for r in results)
        assert breaker.metrics.total_calls == 10

    @pytest.mark.asyncio
    async def test_ignored_exceptions(self):
        """Test that ignored exceptions don't count as failures."""
        class IgnoredError(Exception):
            pass

        breaker = CircuitBreaker(
            "ignore_test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                ignore_exceptions=(IgnoredError,),
            ),
        )

        mock_fail = AsyncMock(side_effect=IgnoredError("ignored"))

        with pytest.raises(IgnoredError):
            await breaker.call(mock_fail)

        # Circuit should still be closed
        assert breaker.is_closed
        assert breaker.metrics.failed_calls == 0

    @pytest.mark.asyncio
    async def test_slow_calls_tracking(self):
        """Test that slow calls are tracked."""
        breaker = CircuitBreaker(
            "slow_test",
            config=CircuitBreakerConfig(
                slow_call_duration_threshold_s=0.1,
            ),
        )

        async def slow_func():
            await asyncio.sleep(0.15)
            return "ok"

        await breaker.call(slow_func)

        assert breaker.metrics.slow_calls == 1

    @pytest.mark.asyncio
    async def test_failure_rate_based_opening(self):
        """Test circuit opens based on failure rate threshold."""
        breaker = CircuitBreaker(
            "rate_test",
            config=CircuitBreakerConfig(
                failure_threshold=100,  # High threshold
                failure_rate_threshold=0.3,  # Low rate threshold
                sliding_window_size=10,
            ),
        )

        mock_success = AsyncMock(return_value="ok")
        mock_fail = AsyncMock(side_effect=Exception("fail"))

        # 6 successes, 4 failures = 40% failure rate (> 30% threshold)
        for _ in range(6):
            await breaker.call(mock_success)

        for _ in range(4):
            with pytest.raises(Exception):
                await breaker.call(mock_fail)

        assert breaker.is_open

    def test_metrics_cleanup(self):
        """Test that old metrics records are cleaned up."""
        metrics = CircuitBreakerMetrics()

        # Add some old records
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        metrics._call_history.append((old_time, True, 0.1))
        metrics._failure_history.append(old_time)

        # Add recent records
        metrics.record_success(0.1)

        # Cleanup with 1 hour max age
        metrics.cleanup_old_records(max_age_s=3600)

        assert len(metrics._call_history) == 1
        assert len(metrics._failure_history) == 0
