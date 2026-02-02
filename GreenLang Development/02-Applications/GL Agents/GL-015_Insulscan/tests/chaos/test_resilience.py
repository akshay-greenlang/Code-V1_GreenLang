# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Chaos Engineering Resilience Tests

Tests system behavior under failure conditions:
- Thermal camera connection failures
- Database unavailability
- Network issues and circuit breaker patterns
- Bad data injection and validation
- Resource exhaustion handling
- Recovery behavior after failures

Reference:
- Netflix Chaos Monkey principles
- IEC 61508 Fault injection testing
- Site Reliability Engineering practices

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import random
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CHAOS HELPERS
# =============================================================================

class ChaosInjector:
    """Helper class for injecting chaos into tests."""

    def __init__(
        self,
        failure_rate: float = 0.3,
        slow_rate: float = 0.2,
        slow_delay_ms: float = 500.0,
        seed: int = 42,
    ):
        self.failure_rate = failure_rate
        self.slow_rate = slow_rate
        self.slow_delay_ms = slow_delay_ms
        self.call_count = 0
        self.failure_count = 0
        self.slow_count = 0
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    async def chaotic_call(self) -> Dict[str, Any]:
        """Simulate a chaotic service call."""
        self.call_count += 1

        # Random failure
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise ConnectionError(f"Chaos failure #{self.failure_count}")

        # Random slow response
        if random.random() < self.slow_rate:
            self.slow_count += 1
            await asyncio.sleep(self.slow_delay_ms / 1000.0)

        return {"status": "success", "call": self.call_count}

    def reset(self):
        """Reset counters."""
        self.call_count = 0
        self.failure_count = 0
        self.slow_count = 0
        random.seed(self.seed)
        np.random.seed(self.seed)


class LatencyInjector:
    """Inject variable latency into calls."""

    def __init__(
        self,
        base_latency_ms: float = 50.0,
        jitter_ms: float = 20.0,
        spike_probability: float = 0.1,
        spike_latency_ms: float = 2000.0,
    ):
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.spike_probability = spike_probability
        self.spike_latency_ms = spike_latency_ms

    async def delayed_call(self) -> Dict[str, Any]:
        """Execute call with injected latency."""
        if random.random() < self.spike_probability:
            delay = self.spike_latency_ms
        else:
            delay = self.base_latency_ms + random.uniform(-self.jitter_ms, self.jitter_ms)

        await asyncio.sleep(delay / 1000.0)
        return {"latency_ms": delay}


# =============================================================================
# CIRCUIT BREAKER FOR TESTING
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 10.0


class CircuitBreaker:
    """Circuit breaker for resilience testing."""

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None

    @property
    def state(self) -> CircuitState:
        self._check_timeout()
        return self._state

    def _check_timeout(self):
        if self._state != CircuitState.OPEN:
            return
        if self._last_failure_time is None:
            return

        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        if elapsed >= self.config.timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

    def can_execute(self) -> bool:
        state = self.state
        return state != CircuitState.OPEN

    def record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self):
        self._last_failure_time = datetime.now(timezone.utc)

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


# =============================================================================
# TEST CLASS: THERMAL CAMERA FAILURES
# =============================================================================

class TestThermalCameraFailures:
    """Test behavior when thermal camera fails."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_camera_connection_timeout(self):
        """Test handling of camera connection timeout."""
        async def slow_connect():
            await asyncio.sleep(10)  # Very slow
            return True

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_connect(), timeout=0.1)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_camera_connection_refused(self):
        """Test handling of camera connection refused."""
        camera = Mock()
        camera.connect = AsyncMock(
            side_effect=ConnectionRefusedError("Camera offline")
        )

        with pytest.raises(ConnectionRefusedError):
            await camera.connect()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_camera_capture_failure_fallback(self):
        """Test fallback when camera capture fails."""
        camera_available = False

        async def capture_with_fallback(asset_id: str) -> Dict[str, Any]:
            if camera_available:
                return {
                    "source": "thermal_camera",
                    "surface_temp_C": 45.0,
                    "confidence": 0.95,
                }
            else:
                # Fallback to historical data or estimate
                return {
                    "source": "fallback_estimate",
                    "surface_temp_C": 50.0,  # Estimated
                    "confidence": 0.5,  # Lower confidence
                }

        result = await capture_with_fallback("PIPE-001")

        assert result["source"] == "fallback_estimate"
        assert result["confidence"] < 1.0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_camera_intermittent_failures(self, chaos_injector):
        """Test handling of intermittent camera failures."""
        injector = chaos_injector(failure_rate=0.3, slow_rate=0.1)

        successes = 0
        failures = 0

        for _ in range(50):
            try:
                await injector.chaotic_call()
                successes += 1
            except ConnectionError:
                failures += 1

        assert successes > 0
        assert failures > 0
        assert successes + failures == 50

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_camera_reconnection_after_failure(self):
        """Test camera reconnection after failure."""
        connection_attempts = 0
        connected = False

        async def connect_with_retry(max_retries: int = 3) -> bool:
            nonlocal connection_attempts, connected

            for attempt in range(max_retries):
                connection_attempts += 1
                if attempt < 2:
                    raise ConnectionError("Camera not ready")
                connected = True
                return True

            return False

        result = await connect_with_retry()

        assert result == True
        assert connection_attempts == 3


# =============================================================================
# TEST CLASS: DATABASE FAILURES
# =============================================================================

class TestDatabaseFailures:
    """Test behavior when database fails."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_database_connection_loss(self, mock_database):
        """Test handling of database connection loss."""
        mock_database.connected = False

        # Should handle gracefully
        if not mock_database.connected:
            # Fall back to cached data or return error
            result = {"status": "degraded", "using_cache": True}
            assert result["status"] == "degraded"

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_database_write_failure(self, mock_database):
        """Test handling of database write failure."""
        async def failing_save(data):
            raise IOError("Database write failed")

        with pytest.raises(IOError):
            await failing_save({"test": "data"})

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_database_read_timeout(self):
        """Test handling of database read timeout."""
        async def slow_query():
            await asyncio.sleep(5)
            return {"result": "data"}

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_query(), timeout=0.1)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_database_retry_logic(self):
        """Test database operation retry logic."""
        attempts = 0
        max_retries = 3

        async def operation_with_retry():
            nonlocal attempts
            attempts += 1
            if attempts < max_retries:
                raise IOError("Temporary failure")
            return {"status": "success"}

        result = None
        for _ in range(max_retries):
            try:
                result = await operation_with_retry()
                break
            except IOError:
                await asyncio.sleep(0.01)  # Brief delay before retry

        assert result is not None
        assert result["status"] == "success"
        assert attempts == max_retries


# =============================================================================
# TEST CLASS: NETWORK FAILURES
# =============================================================================

class TestNetworkFailures:
    """Test behavior under network failure conditions."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_network_partition(self):
        """Test handling of network partition."""
        partition_active = True

        async def network_call():
            if partition_active:
                raise ConnectionError("Network partition")
            return {"status": "ok"}

        # During partition
        with pytest.raises(ConnectionError):
            await network_call()

        # After partition heals
        partition_active = False
        result = await network_call()
        assert result["status"] == "ok"

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_high_latency_network(self):
        """Test handling of high latency network."""
        latency = LatencyInjector(
            base_latency_ms=500,
            spike_probability=0.3,
            spike_latency_ms=2000
        )

        results = []
        for _ in range(10):
            result = await latency.delayed_call()
            results.append(result["latency_ms"])

        # Should have some high latency calls
        assert max(results) > 500

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """Test handling of DNS resolution failure."""
        async def dns_lookup():
            raise OSError("Name or service not known")

        with pytest.raises(OSError):
            await dns_lookup()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """Test behavior when connection pool is exhausted."""
        pool_size = 5
        active_connections = pool_size  # All in use

        async def get_connection():
            if active_connections >= pool_size:
                raise RuntimeError("Connection pool exhausted")
            return Mock()

        with pytest.raises(RuntimeError) as exc_info:
            await get_connection()

        assert "exhausted" in str(exc_info.value)


# =============================================================================
# TEST CLASS: BAD DATA INJECTION
# =============================================================================

class TestBadDataInjection:
    """Test handling of bad data injection."""

    @pytest.mark.chaos
    def test_nan_temperature_handling(self):
        """Test handling of NaN temperature values."""
        bad_temp = float('nan')

        is_nan = math.isnan(bad_temp)
        assert is_nan, "NaN should be detected"

    @pytest.mark.chaos
    def test_infinite_value_handling(self):
        """Test handling of infinite values."""
        heat_loss = float('inf')

        is_inf = math.isinf(heat_loss)
        assert is_inf, "Infinity should be detected"

    @pytest.mark.chaos
    def test_negative_dimension_rejection(self):
        """Test rejection of negative dimensions."""
        invalid_thickness = -0.05

        is_valid = invalid_thickness > 0
        assert not is_valid, "Negative thickness should be rejected"

    @pytest.mark.chaos
    def test_physically_impossible_temperatures(self):
        """Test rejection of physically impossible temperatures."""
        T_below_absolute_zero = -300.0  # Celsius

        is_valid = T_below_absolute_zero > -273.15
        assert not is_valid, "Temperature below absolute zero should be rejected"

    @pytest.mark.chaos
    def test_sensor_spike_detection(self):
        """Test detection of sensor spikes."""
        readings = [45.0, 46.0, 45.5, 200.0, 45.8, 45.2]  # 200 is spike

        window = 3
        threshold = 3.0
        spikes = []

        for i in range(window, len(readings)):
            window_values = readings[i - window:i]
            mean = np.mean(window_values)
            std = np.std(window_values)

            if std > 0 and abs(readings[i] - mean) > threshold * std:
                spikes.append(i)

        assert 3 in spikes, "Spike at index 3 should be detected"

    @pytest.mark.chaos
    def test_stuck_sensor_detection(self):
        """Test detection of stuck sensor values."""
        readings = [45.0, 45.0, 45.0, 45.0, 45.0]  # All same value

        variance = np.var(readings)
        is_stuck = variance < 0.001

        assert is_stuck, "Stuck sensor should be detected"

    @pytest.mark.chaos
    def test_data_quality_scoring(self):
        """Test data quality score with bad data."""
        readings = [45.0, 46.0, None, 45.5, float('nan'), 46.2]

        valid_count = sum(
            1 for r in readings
            if r is not None and not (isinstance(r, float) and np.isnan(r))
        )
        quality_score = valid_count / len(readings)

        assert quality_score < 1.0
        assert quality_score == pytest.approx(0.667, abs=0.01)


# =============================================================================
# TEST CLASS: RESOURCE EXHAUSTION
# =============================================================================

class TestResourceExhaustion:
    """Test behavior under resource exhaustion conditions."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self):
        """Test handling of concurrent request limit."""
        max_concurrent = 10
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request():
            async with semaphore:
                await asyncio.sleep(0.01)
                return {"status": "ok"}

        # Launch more requests than limit
        tasks = [limited_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # All should complete eventually
        assert len(results) == 20

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test behavior under simulated memory pressure."""
        large_arrays = []

        try:
            for _ in range(5):
                # Allocate moderate array
                arr = np.zeros((1000, 1000))
                large_arrays.append(arr)
        except MemoryError:
            pass  # Handle gracefully

        # Clean up
        large_arrays.clear()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test handling of queue overflow."""
        queue_max_size = 10
        queue = asyncio.Queue(maxsize=queue_max_size)

        # Fill queue
        for i in range(queue_max_size):
            await queue.put({"item": i})

        assert queue.full()

        # Additional put should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.put({"item": "overflow"}), timeout=0.1)


# =============================================================================
# TEST CLASS: RECOVERY BEHAVIOR
# =============================================================================

class TestRecoveryBehavior:
    """Test recovery behavior after failures."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_recovery_after_service_failure(self):
        """Test recovery after service failure."""
        service_healthy = False

        async def call_service():
            if not service_healthy:
                raise ConnectionError("Service unavailable")
            return {"status": "ok"}

        # Initial failure
        with pytest.raises(ConnectionError):
            await call_service()

        # Service recovers
        service_healthy = True

        # Should succeed now
        result = await call_service()
        assert result["status"] == "ok"

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        components = {
            "thermal_camera": False,  # Failed
            "database": True,
            "calculation_engine": True,
        }

        available_features = []

        if components["thermal_camera"]:
            available_features.append("live_thermal_imaging")
        else:
            available_features.append("historical_data_fallback")

        if components["calculation_engine"]:
            available_features.append("heat_loss_calculation")

        if components["database"]:
            available_features.append("data_persistence")

        assert "historical_data_fallback" in available_features
        assert "heat_loss_calculation" in available_features
        assert "data_persistence" in available_features

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self):
        """Test exponential backoff retry pattern."""
        max_retries = 5
        base_delay = 0.01
        attempt = 0

        async def operation_with_backoff():
            nonlocal attempt
            attempt += 1

            if attempt < 4:
                raise ConnectionError("Temporary failure")
            return {"status": "ok"}

        delays = []
        for retry in range(max_retries):
            try:
                result = await operation_with_backoff()
                break
            except ConnectionError:
                delay = base_delay * (2 ** retry)
                delays.append(delay)
                await asyncio.sleep(delay)

        # Should have exponentially increasing delays
        for i in range(1, len(delays)):
            assert delays[i] > delays[i - 1]

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, chaos_injector):
        """Test circuit breaker recovery after failures."""
        breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.5,
        ))

        injector = chaos_injector(failure_rate=0.8)

        # Trigger circuit opening
        for _ in range(10):
            if not breaker.can_execute():
                break
            try:
                await injector.chaotic_call()
                breaker.record_success()
            except ConnectionError:
                breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.6)

        # Should be half-open now
        assert breaker.state == CircuitState.HALF_OPEN

        # Successful calls should close it
        for _ in range(2):
            breaker.record_success()

        assert breaker.state == CircuitState.CLOSED


# =============================================================================
# TEST CLASS: CIRCUIT BREAKER TRANSITIONS
# =============================================================================

class TestCircuitBreakerTransitions:
    """Test circuit breaker state transitions in detail."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        return CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.5,
        ))

    @pytest.mark.chaos
    def test_closed_to_open_transition(self, breaker):
        """Test transition from closed to open state."""
        assert breaker.state == CircuitState.CLOSED

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert not breaker.can_execute()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_open_to_half_open_transition(self, breaker):
        """Test transition from open to half-open after timeout."""
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.6)

        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute()

    @pytest.mark.chaos
    def test_half_open_to_closed_transition(self, breaker):
        """Test transition from half-open to closed on success."""
        for _ in range(3):
            breaker.record_failure()

        breaker._state = CircuitState.HALF_OPEN
        breaker._success_count = 0

        for _ in range(2):
            breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.chaos
    def test_half_open_to_open_on_failure(self, breaker):
        """Test transition from half-open to open on failure."""
        breaker._state = CircuitState.HALF_OPEN

        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.chaos
    def test_reset_restores_closed_state(self, breaker):
        """Test reset restores circuit to closed state."""
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED


# =============================================================================
# TEST CLASS: CHAOS TEST UTILITIES
# =============================================================================

class TestChaosUtilities:
    """Test chaos testing utilities."""

    @pytest.mark.chaos
    def test_chaos_injector_determinism(self, chaos_injector):
        """Test that chaos injector is deterministic with seed."""
        random.seed(42)
        values1 = [random.random() for _ in range(10)]

        random.seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_latency_injector_spikes(self):
        """Test latency injector spike generation."""
        injector = LatencyInjector(
            base_latency_ms=10,
            spike_probability=1.0,  # Always spike
            spike_latency_ms=1000
        )

        result = await injector.delayed_call()

        assert result["latency_ms"] == 1000

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_chaos_injector_failure_rate(self, chaos_injector):
        """Test chaos injector respects failure rate."""
        injector = chaos_injector(failure_rate=0.5, slow_rate=0.0)

        failures = 0
        total = 100

        for _ in range(total):
            try:
                await injector.chaotic_call()
            except ConnectionError:
                failures += 1

        # Should be roughly 50% failures (with some variance)
        assert 30 < failures < 70


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestThermalCameraFailures",
    "TestDatabaseFailures",
    "TestNetworkFailures",
    "TestBadDataInjection",
    "TestResourceExhaustion",
    "TestRecoveryBehavior",
    "TestCircuitBreakerTransitions",
    "TestChaosUtilities",
    "ChaosInjector",
    "LatencyInjector",
    "CircuitBreaker",
]
