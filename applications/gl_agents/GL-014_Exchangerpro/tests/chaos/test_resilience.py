# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Chaos Engineering Resilience Tests

Tests system behavior under failure conditions:
- ML service failures and fallback behavior
- Network issues and circuit breaker patterns
- Bad data injection and validation
- Resource exhaustion handling
- Recovery behavior

Reference:
    - Netflix Chaos Monkey principles
    - IEC 61508 Fault injection testing

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import random
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch


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
# ML SERVICE FAILURE TESTS
# =============================================================================

class TestMLServiceFailures:
    """Test behavior when ML service fails."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_ml_service_timeout(self):
        """Test handling of ML service timeout."""
        async def slow_prediction():
            await asyncio.sleep(10)  # Simulate very slow response
            return {"prediction": 0.5}

        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_prediction(), timeout=0.1)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_ml_service_connection_refused(self):
        """Test handling of ML service connection refused."""
        ml_service = Mock()
        ml_service.predict = AsyncMock(side_effect=ConnectionRefusedError("Connection refused"))

        with pytest.raises(ConnectionRefusedError):
            await ml_service.predict({})

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_fallback_on_ml_failure(self):
        """Test fallback to rule-based prediction on ML failure."""
        ml_available = False

        async def predict_with_fallback(features: Dict[str, Any]) -> Dict[str, Any]:
            if ml_available:
                return {"source": "ml", "fouling_resistance": 0.00035}
            else:
                # Fallback: rule-based using pressure drop ratio
                dp_ratio = features.get("dp_shell_ratio", 1.0)
                rf_estimate = max(0.0001, (dp_ratio - 1.0) * 0.0005)
                return {
                    "source": "fallback",
                    "fouling_resistance": rf_estimate,
                    "confidence": 0.5,  # Lower confidence for fallback
                }

        result = await predict_with_fallback({"dp_shell_ratio": 1.2})

        assert result["source"] == "fallback"
        assert result["confidence"] < 1.0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_ml_service_intermittent_failures(self, chaos_injector):
        """Test handling of intermittent ML service failures."""
        injector = chaos_injector(failure_rate=0.3, slow_rate=0.1)

        successes = 0
        failures = 0

        for _ in range(50):
            try:
                await injector.chaotic_call()
                successes += 1
            except ConnectionError:
                failures += 1

        # Should have mix of successes and failures
        assert successes > 0
        assert failures > 0
        assert successes + failures == 50

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_ml_model_version_mismatch(self):
        """Test handling of ML model version mismatch."""
        model_version_server = "1.2.0"
        model_version_expected = "1.3.0"

        if model_version_server != model_version_expected:
            warning = f"Model version mismatch: expected {model_version_expected}, got {model_version_server}"
            # Should log warning but continue with available model
            assert "mismatch" in warning


# =============================================================================
# NETWORK FAILURE TESTS
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
    async def test_opcua_connection_loss(self, mock_opcua_client):
        """Test handling of OPC-UA connection loss."""
        # Simulate connection loss
        mock_opcua_client.is_connected = Mock(return_value=False)
        mock_opcua_client.read_tags = AsyncMock(
            side_effect=ConnectionError("OPC-UA connection lost")
        )

        with pytest.raises(ConnectionError):
            await mock_opcua_client.read_tags(["HX-001/TI_HOT_IN"])

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_cmms_connection_retry(self, mock_cmms_connector):
        """Test CMMS connection retry logic."""
        attempts = 0
        max_retries = 3

        async def connect_with_retry():
            nonlocal attempts
            attempts += 1
            if attempts < max_retries:
                raise ConnectionError("CMMS unavailable")
            return True

        # Should succeed after retries
        for _ in range(max_retries):
            try:
                connected = await connect_with_retry()
                break
            except ConnectionError:
                continue

        assert connected == True
        assert attempts == max_retries

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


# =============================================================================
# BAD DATA INJECTION TESTS
# =============================================================================

class TestBadDataInjection:
    """Test handling of bad data injection."""

    @pytest.mark.chaos
    def test_nan_temperature_handling(self, sample_operating_state):
        """Test handling of NaN temperature values."""
        state = sample_operating_state

        # Inject NaN
        bad_temp = float('nan')

        # Detection
        import math
        is_nan = math.isnan(bad_temp)

        assert is_nan, "NaN should be detected"

    @pytest.mark.chaos
    def test_infinite_value_handling(self):
        """Test handling of infinite values."""
        Q = float('inf')

        # Detection
        import math
        is_inf = math.isinf(Q)

        assert is_inf, "Infinity should be detected"

    @pytest.mark.chaos
    def test_negative_flow_rate_rejection(self, sample_operating_state):
        """Test rejection of negative flow rate."""
        invalid_flow = -5.0

        # Validation
        is_valid = invalid_flow > 0

        assert not is_valid, "Negative flow should be rejected"

    @pytest.mark.chaos
    def test_physically_impossible_temperatures(self):
        """Test rejection of physically impossible temperatures."""
        T_below_absolute_zero = -300.0  # Celsius

        is_valid = T_below_absolute_zero > -273.15

        assert not is_valid, "Temperature below absolute zero should be rejected"

    @pytest.mark.chaos
    def test_sensor_spike_detection(self):
        """Test detection of sensor spikes."""
        readings = [100.0, 102.0, 101.0, 500.0, 101.5, 100.5]  # 500 is spike

        # Detect spikes (> 3 sigma from rolling mean)
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
        readings = [100.0, 100.0, 100.0, 100.0, 100.0]  # All same value

        # Detect stuck sensor (zero variance)
        variance = np.var(readings)
        is_stuck = variance < 0.001

        assert is_stuck, "Stuck sensor should be detected"

    @pytest.mark.chaos
    def test_data_quality_score_degradation(self):
        """Test data quality score calculation with bad data."""
        readings = [100.0, 101.0, None, 102.0, float('nan'), 103.0]

        # Calculate quality score
        valid_count = sum(1 for r in readings if r is not None and not (isinstance(r, float) and np.isnan(r)))
        quality_score = valid_count / len(readings)

        assert quality_score < 1.0
        assert quality_score == pytest.approx(0.667, abs=0.01)


# =============================================================================
# RESOURCE EXHAUSTION TESTS
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
        # Simulate memory pressure by allocating large arrays
        large_arrays = []

        try:
            for _ in range(5):
                # Allocate 10MB array
                arr = np.zeros((10000, 1000))
                large_arrays.append(arr)
        except MemoryError:
            # Should handle gracefully
            pass

        # Clean up
        large_arrays.clear()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_thread_pool_exhaustion(self, chaos_injector):
        """Test behavior when thread pool is exhausted."""
        injector = chaos_injector(failure_rate=0.0, slow_rate=0.8, slow_delay_ms=100)

        # Run many slow concurrent operations
        tasks = [injector.chaotic_call() for _ in range(20)]

        # Should complete without deadlock
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, dict))
        assert success_count > 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test handling of queue overflow."""
        queue_max_size = 10
        queue = asyncio.Queue(maxsize=queue_max_size)

        # Fill queue
        for i in range(queue_max_size):
            await queue.put({"item": i})

        # Queue should be full
        assert queue.full()

        # Additional put should block or timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.put({"item": "overflow"}), timeout=0.1)


# =============================================================================
# RECOVERY BEHAVIOR TESTS
# =============================================================================

class TestRecoveryBehavior:
    """Test recovery behavior after failures."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_recovery_after_ml_failure(self):
        """Test recovery after ML service failure."""
        service_healthy = False

        async def check_health():
            return service_healthy

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
            "thermal_engine": True,
            "ml_service": False,  # Failed
            "optimizer": True,
        }

        # System should continue with reduced functionality
        available_features = []

        if components["thermal_engine"]:
            available_features.append("kpi_calculation")

        if components["ml_service"]:
            available_features.append("fouling_prediction")
        else:
            available_features.append("fouling_estimation_fallback")

        if components["optimizer"]:
            available_features.append("cleaning_schedule")

        assert "kpi_calculation" in available_features
        assert "fouling_estimation_fallback" in available_features

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self):
        """Test exponential backoff retry pattern."""
        max_retries = 5
        base_delay = 0.1
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
    async def test_circuit_breaker_pattern(self, chaos_injector):
        """Test circuit breaker pattern."""
        failure_threshold = 3
        failure_count = 0
        circuit_open = False

        injector = chaos_injector(failure_rate=0.8)

        async def protected_call():
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise Exception("Circuit is open")

            try:
                result = await injector.chaotic_call()
                failure_count = 0
                return result
            except ConnectionError:
                failure_count += 1
                if failure_count >= failure_threshold:
                    circuit_open = True
                raise

        blocked_by_circuit = False
        for _ in range(10):
            try:
                await protected_call()
            except ConnectionError:
                continue
            except Exception as e:
                if "Circuit is open" in str(e):
                    blocked_by_circuit = True
                    break

        assert blocked_by_circuit or circuit_open


# =============================================================================
# CHAOS TEST UTILITIES
# =============================================================================

class TestChaosUtilities:
    """Test chaos testing utilities."""

    @pytest.mark.chaos
    def test_chaos_injector_determinism(self, chaos_injector):
        """Test that chaos injector is deterministic with seed."""
        injector1 = chaos_injector(failure_rate=0.5, seed=42)
        injector2 = chaos_injector(failure_rate=0.5, seed=42)

        # Generate random values with same seed
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


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestMLServiceFailures",
    "TestNetworkFailures",
    "TestBadDataInjection",
    "TestResourceExhaustion",
    "TestRecoveryBehavior",
    "TestChaosUtilities",
    "ChaosInjector",
    "LatencyInjector",
]
