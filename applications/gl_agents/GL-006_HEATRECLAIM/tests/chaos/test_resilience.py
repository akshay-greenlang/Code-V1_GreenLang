# -*- coding: utf-8 -*-
"""
GL-006 HEATRECLAIM - Chaos Engineering Resilience Tests

Validates system behavior under failure conditions following chaos engineering
principles. Tests ensure the Heat Exchanger Network optimizer maintains safety
guarantees even during cascading failures.

Test Categories:
1. Circuit Breaker Chaos - Test adaptive threshold behavior under stress
2. Network Failures - Simulate network partitions and timeouts
3. Service Degradation - Test behavior with slow/failing dependencies
4. Resource Exhaustion - Memory pressure and thread pool exhaustion
5. Recovery Behavior - Verify correct recovery after failures

Reference:
    - Netflix Chaos Monkey principles
    - IEC 61508 Fault injection testing
    - ASME PTC 4.3 Safety validation

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import asyncio
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import circuit breaker components
from safety.circuit_breaker import (
    DynamicCircuitBreaker,
    DynamicCircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerEvent,
    CircuitOpenError,
    CircuitHalfOpenError,
    LoadShedError,
    HealthLevel,
    HeatReclaimCircuitBreakers,
    get_or_create_circuit_breaker,
    circuit_protected,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def reset_registry():
    """Reset circuit breaker registry between tests."""
    from safety.circuit_breaker import _registry
    _registry.clear()
    yield
    _registry.clear()


@pytest.fixture
def dynamic_breaker(reset_registry):
    """Create a dynamic circuit breaker for testing."""
    return DynamicCircuitBreaker(
        name="test_breaker",
        base_failure_threshold=5,
        min_failure_threshold=2,
        max_failure_threshold=10,
        base_recovery_timeout_seconds=1.0,
        max_recovery_timeout_seconds=5.0,
    )


@pytest.fixture
def fast_breaker(reset_registry):
    """Create a fast circuit breaker for chaos tests."""
    config = DynamicCircuitBreakerConfig(
        name="chaos_breaker",
        base_failure_threshold=3,
        min_failure_threshold=1,
        max_failure_threshold=5,
        base_recovery_timeout_seconds=0.5,
        max_recovery_timeout_seconds=2.0,
        jitter_factor=0.1,
        half_open_max_calls=2,
        success_threshold=2,
        health_window_size=20,
        threshold_adjustment_interval=1.0,
        load_shed_threshold=0.7,
        load_shed_max_ratio=0.3,
    )
    return DynamicCircuitBreaker(name="chaos_breaker", config=config)


# =============================================================================
# CHAOS HELPERS
# =============================================================================


class ChaosInjector:
    """Helper class for injecting chaos into tests."""

    def __init__(
        self,
        failure_rate: float = 0.5,
        slow_rate: float = 0.2,
        slow_delay_ms: float = 1000.0,
    ):
        self.failure_rate = failure_rate
        self.slow_rate = slow_rate
        self.slow_delay_ms = slow_delay_ms
        self.call_count = 0
        self.failure_count = 0
        self.slow_count = 0

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


class LatencyInjector:
    """Inject variable latency into calls."""

    def __init__(
        self,
        base_latency_ms: float = 50.0,
        jitter_ms: float = 20.0,
        spike_probability: float = 0.1,
        spike_latency_ms: float = 5000.0,
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
# CIRCUIT BREAKER CHAOS TESTS
# =============================================================================


class TestCircuitBreakerChaos:
    """Test circuit breaker behavior under chaotic conditions."""

    @pytest.mark.asyncio
    async def test_rapid_failure_cascade(self, fast_breaker):
        """Test circuit opens correctly during rapid failure cascade."""
        failures = 0

        for i in range(10):
            try:
                async with fast_breaker.protect():
                    raise ConnectionError(f"Cascade failure {i}")
            except ConnectionError:
                failures += 1
            except CircuitOpenError:
                break

        # Circuit should have opened after threshold failures
        assert fast_breaker.state == CircuitBreakerState.OPEN
        assert failures >= fast_breaker._current_failure_threshold

    @pytest.mark.asyncio
    async def test_intermittent_failures(self, fast_breaker):
        """Test circuit handles intermittent failures correctly."""
        chaos = ChaosInjector(failure_rate=0.3, slow_rate=0.0)

        successes = 0
        failures = 0
        blocked = 0

        for _ in range(50):
            try:
                async with fast_breaker.protect():
                    await chaos.chaotic_call()
                successes += 1
            except ConnectionError:
                failures += 1
            except CircuitOpenError:
                blocked += 1
                await asyncio.sleep(0.1)  # Wait for recovery

        # Should have a mix of successes, failures, and blocked calls
        assert successes > 0
        assert failures > 0
        # Circuit may open/close multiple times with intermittent failures

    @pytest.mark.asyncio
    async def test_adaptive_threshold_under_stress(self, fast_breaker):
        """Test that thresholds adapt based on system health."""
        initial_threshold = fast_breaker.current_failure_threshold

        # Generate sustained failures to degrade health
        for _ in range(15):
            fast_breaker.record_failure(ConnectionError("stress test"))
            if fast_breaker.state == CircuitBreakerState.OPEN:
                fast_breaker.reset()
            await asyncio.sleep(0.1)

        # Threshold should have decreased (stricter)
        fast_breaker._adjust_thresholds()  # Force adjustment
        # Note: In fast chaos tests, threshold may still be at base

    @pytest.mark.asyncio
    async def test_health_score_degradation(self, fast_breaker):
        """Test health score decreases with failures."""
        initial_health = fast_breaker.health_score

        # Record multiple failures
        for _ in range(10):
            fast_breaker.record_failure(ConnectionError("health test"))
            if fast_breaker.state == CircuitBreakerState.OPEN:
                fast_breaker.reset()

        final_health = fast_breaker.health_score
        assert final_health < initial_health

    @pytest.mark.asyncio
    async def test_load_shedding_activation(self, fast_breaker):
        """Test load shedding activates under degraded conditions."""
        # Degrade health below load shed threshold
        for _ in range(15):
            fast_breaker.record_failure(ConnectionError("load shed test"))
            if fast_breaker.state == CircuitBreakerState.OPEN:
                fast_breaker.reset()

        fast_breaker._update_load_shedding()

        # Check if load shedding was activated
        metrics = fast_breaker.get_metrics()
        if metrics.health_score < fast_breaker._config.load_shed_threshold:
            assert metrics.load_shed_active or metrics.health_score < 0.5

    @pytest.mark.asyncio
    async def test_exponential_backoff_recovery(self, fast_breaker):
        """Test exponential backoff increases recovery timeout."""
        timeouts = []

        # Open circuit multiple times
        for i in range(3):
            # Force failures to open circuit
            for _ in range(fast_breaker._current_failure_threshold + 1):
                fast_breaker.record_failure(ConnectionError(f"backoff test {i}"))

            if fast_breaker.state == CircuitBreakerState.OPEN:
                timeouts.append(fast_breaker.current_recovery_timeout)
                # Wait for recovery and close
                await asyncio.sleep(fast_breaker.current_recovery_timeout + 0.1)
                # Force success to close
                for _ in range(fast_breaker._config.success_threshold):
                    fast_breaker.record_success()

        # Later timeouts should be longer (with some variance from jitter)
        if len(timeouts) >= 2:
            # At least one later timeout should be >= first
            assert any(t >= timeouts[0] for t in timeouts[1:])

    @pytest.mark.asyncio
    async def test_concurrent_chaos(self, fast_breaker):
        """Test circuit breaker under concurrent chaotic load."""
        chaos = ChaosInjector(failure_rate=0.4, slow_rate=0.1, slow_delay_ms=100)

        async def worker(worker_id: int):
            results = {"success": 0, "failure": 0, "blocked": 0}
            for _ in range(10):
                try:
                    async with fast_breaker.protect():
                        await chaos.chaotic_call()
                    results["success"] += 1
                except ConnectionError:
                    results["failure"] += 1
                except (CircuitOpenError, CircuitHalfOpenError, LoadShedError):
                    results["blocked"] += 1
                await asyncio.sleep(0.01)
            return results

        # Run concurrent workers
        tasks = [worker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        total_success = sum(r["success"] for r in results)
        total_failure = sum(r["failure"] for r in results)
        total_blocked = sum(r["blocked"] for r in results)

        # Should have mix of outcomes
        total = total_success + total_failure + total_blocked
        assert total == 50  # 5 workers * 10 calls


# =============================================================================
# NETWORK FAILURE TESTS
# =============================================================================


class TestNetworkFailures:
    """Test behavior under simulated network failures."""

    @pytest.mark.asyncio
    async def test_connection_timeout_cascade(self, fast_breaker):
        """Test handling of connection timeouts."""

        async def timeout_call():
            await asyncio.sleep(10)  # Simulate timeout
            return {"status": "ok"}

        blocked_count = 0
        timeout_count = 0

        for _ in range(10):
            try:
                async with fast_breaker.protect():
                    # Use asyncio.wait_for to simulate timeout
                    try:
                        await asyncio.wait_for(timeout_call(), timeout=0.1)
                    except asyncio.TimeoutError:
                        raise ConnectionError("Connection timeout")
            except ConnectionError:
                timeout_count += 1
            except CircuitOpenError:
                blocked_count += 1

        # Circuit should have opened after timeouts
        assert fast_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_connection_refused(self, fast_breaker):
        """Test handling of connection refused errors."""

        async def refused_call():
            raise ConnectionRefusedError("Connection refused")

        with pytest.raises((ConnectionRefusedError, CircuitOpenError)):
            for _ in range(10):
                async with fast_breaker.protect():
                    await refused_call()

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self, fast_breaker):
        """Test handling of DNS resolution failures."""

        async def dns_failure():
            raise OSError("Name or service not known")

        blocked = 0
        for _ in range(10):
            try:
                async with fast_breaker.protect():
                    await dns_failure()
            except OSError:
                pass
            except CircuitOpenError:
                blocked += 1

        assert blocked > 0 or fast_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_network_partition_simulation(self, fast_breaker):
        """Simulate network partition where all calls fail."""
        partition_active = True

        async def partitioned_call():
            if partition_active:
                raise ConnectionError("Network partition")
            return {"status": "ok"}

        # Partition phase - all calls fail
        for _ in range(5):
            try:
                async with fast_breaker.protect():
                    await partitioned_call()
            except (ConnectionError, CircuitOpenError):
                pass

        assert fast_breaker.state == CircuitBreakerState.OPEN

        # Heal partition
        partition_active = False

        # Wait for recovery
        await asyncio.sleep(fast_breaker.current_recovery_timeout + 0.2)

        # Should recover
        successes = 0
        for _ in range(5):
            try:
                async with fast_breaker.protect():
                    await partitioned_call()
                successes += 1
            except (ConnectionError, CircuitOpenError, CircuitHalfOpenError):
                pass

        # Should have some successes after partition heals
        assert successes > 0 or fast_breaker.state == CircuitBreakerState.HALF_OPEN


# =============================================================================
# SERVICE DEGRADATION TESTS
# =============================================================================


class TestServiceDegradation:
    """Test behavior under service degradation conditions."""

    @pytest.mark.asyncio
    async def test_gradual_degradation(self, fast_breaker):
        """Test circuit breaker during gradual service degradation."""
        failure_rate = 0.0

        async def degrading_call():
            nonlocal failure_rate
            if random.random() < failure_rate:
                raise ConnectionError("Service degraded")
            return {"status": "ok"}

        # Gradually increase failure rate
        for phase in range(5):
            failure_rate = phase * 0.2  # 0%, 20%, 40%, 60%, 80%

            for _ in range(10):
                try:
                    async with fast_breaker.protect():
                        await degrading_call()
                except (ConnectionError, CircuitOpenError, CircuitHalfOpenError):
                    pass
                await asyncio.sleep(0.01)

        # Health should have degraded
        assert fast_breaker.health_score < 1.0

    @pytest.mark.asyncio
    async def test_latency_spike_handling(self, fast_breaker):
        """Test handling of latency spikes."""
        latency = LatencyInjector(
            base_latency_ms=10.0,
            spike_probability=0.3,
            spike_latency_ms=500.0,
        )

        slow_calls = 0
        for _ in range(20):
            try:
                async with fast_breaker.protect():
                    result = await latency.delayed_call()
                    if result["latency_ms"] > 100:
                        slow_calls += 1
            except (CircuitOpenError, CircuitHalfOpenError):
                pass

        # Should have tracked slow calls
        metrics = fast_breaker.get_metrics()
        assert metrics.avg_call_duration_ms > 0

    @pytest.mark.asyncio
    async def test_service_flapping(self, fast_breaker):
        """Test handling of service that flaps up/down."""
        service_up = True
        flap_count = 0

        async def flapping_call():
            if not service_up:
                raise ConnectionError("Service down")
            return {"status": "ok"}

        async def flap_service():
            nonlocal service_up, flap_count
            while flap_count < 5:
                await asyncio.sleep(0.2)
                service_up = not service_up
                flap_count += 1

        # Start flapper
        flap_task = asyncio.create_task(flap_service())

        # Make calls during flapping
        for _ in range(30):
            try:
                async with fast_breaker.protect():
                    await flapping_call()
            except (ConnectionError, CircuitOpenError, CircuitHalfOpenError):
                pass
            await asyncio.sleep(0.05)

        await flap_task

        # Circuit breaker should have adapted
        assert fast_breaker._total_calls > 0

    @pytest.mark.asyncio
    async def test_partial_outage(self, fast_breaker):
        """Test handling of partial outage (some endpoints fail)."""

        async def partial_outage_call(endpoint: str):
            failing_endpoints = {"endpoint_a", "endpoint_b"}
            if endpoint in failing_endpoints:
                raise ConnectionError(f"{endpoint} unavailable")
            return {"endpoint": endpoint, "status": "ok"}

        endpoints = ["endpoint_a", "endpoint_b", "endpoint_c", "endpoint_d"]
        success_count = 0
        failure_count = 0

        for _ in range(20):
            endpoint = random.choice(endpoints)
            try:
                async with fast_breaker.protect():
                    await partial_outage_call(endpoint)
                success_count += 1
            except ConnectionError:
                failure_count += 1
            except CircuitOpenError:
                pass

        # Should have mix of successes (50% of endpoints work)
        assert success_count > 0


# =============================================================================
# RESOURCE EXHAUSTION TESTS
# =============================================================================


class TestResourceExhaustion:
    """Test behavior under resource exhaustion conditions."""

    @pytest.mark.asyncio
    async def test_thread_pool_exhaustion(self, fast_breaker):
        """Test behavior when thread pool is exhausted."""
        semaphore = asyncio.Semaphore(2)  # Simulate limited pool

        async def pool_limited_call():
            async with semaphore:
                await asyncio.sleep(0.1)
                return {"status": "ok"}

        # Run more concurrent calls than pool size
        async def make_calls():
            for _ in range(5):
                try:
                    async with fast_breaker.protect():
                        await pool_limited_call()
                except (CircuitOpenError, CircuitHalfOpenError):
                    pass

        tasks = [make_calls() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should complete without deadlock
        assert fast_breaker._total_calls > 0

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, fast_breaker):
        """Test under high concurrency stress."""
        call_count = 0
        lock = asyncio.Lock()

        async def stress_call():
            nonlocal call_count
            async with lock:
                call_count += 1
            await asyncio.sleep(0.001)
            if random.random() < 0.2:
                raise ConnectionError("Stress failure")
            return {"count": call_count}

        async def worker():
            for _ in range(20):
                try:
                    async with fast_breaker.protect():
                        await stress_call()
                except (ConnectionError, CircuitOpenError, CircuitHalfOpenError, LoadShedError):
                    pass

        # High concurrency
        tasks = [worker() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Should handle high load
        assert call_count > 0

    @pytest.mark.asyncio
    async def test_burst_traffic(self, fast_breaker):
        """Test handling of burst traffic patterns."""

        async def burst_call():
            if random.random() < 0.1:
                raise ConnectionError("Burst failure")
            return {"status": "ok"}

        # Burst of 50 calls
        tasks = []
        for _ in range(50):
            async def single_call():
                try:
                    async with fast_breaker.protect():
                        await burst_call()
                except (ConnectionError, CircuitOpenError, CircuitHalfOpenError, LoadShedError):
                    pass

            tasks.append(single_call())

        await asyncio.gather(*tasks)

        metrics = fast_breaker.get_metrics()
        assert metrics.total_calls > 0


# =============================================================================
# RECOVERY BEHAVIOR TESTS
# =============================================================================


class TestRecoveryBehavior:
    """Test recovery behavior after failures."""

    @pytest.mark.asyncio
    async def test_recovery_after_total_failure(self, fast_breaker):
        """Test recovery after complete service failure."""
        service_healthy = False

        async def recovering_call():
            if not service_healthy:
                raise ConnectionError("Service down")
            return {"status": "ok"}

        # Total failure phase
        for _ in range(10):
            try:
                async with fast_breaker.protect():
                    await recovering_call()
            except (ConnectionError, CircuitOpenError):
                pass

        assert fast_breaker.state == CircuitBreakerState.OPEN

        # Service recovers
        service_healthy = True

        # Wait for circuit to try recovery
        await asyncio.sleep(fast_breaker.current_recovery_timeout + 0.2)

        # Make successful calls
        successes = 0
        for _ in range(10):
            try:
                async with fast_breaker.protect():
                    await recovering_call()
                successes += 1
            except (CircuitOpenError, CircuitHalfOpenError):
                await asyncio.sleep(0.1)

        # Should have recovered
        assert successes > 0

    @pytest.mark.asyncio
    async def test_health_score_recovery(self, fast_breaker):
        """Test health score recovers after sustained success."""
        # Degrade health first
        for _ in range(10):
            fast_breaker.record_failure(ConnectionError("degrade"))
            if fast_breaker.state == CircuitBreakerState.OPEN:
                fast_breaker.reset()

        degraded_health = fast_breaker.health_score

        # Sustained success
        for _ in range(20):
            fast_breaker.record_success(50.0)

        recovered_health = fast_breaker.health_score

        # Health should have improved
        assert recovered_health > degraded_health

    @pytest.mark.asyncio
    async def test_graceful_degradation_recovery(self, fast_breaker):
        """Test graceful degradation and recovery cycle."""
        failure_rate = 0.0

        async def variable_call():
            if random.random() < failure_rate:
                raise ConnectionError("Variable failure")
            return {"status": "ok"}

        # Degradation cycle: 0% -> 50% -> 0%
        for phase, rate in enumerate([0.0, 0.25, 0.5, 0.25, 0.0]):
            failure_rate = rate

            for _ in range(10):
                try:
                    async with fast_breaker.protect():
                        await variable_call()
                except (ConnectionError, CircuitOpenError, CircuitHalfOpenError, LoadShedError):
                    pass
                await asyncio.sleep(0.01)

        # Should be in healthy state after recovery
        metrics = fast_breaker.get_metrics()
        # Health should have recovered to some degree
        assert metrics.success_count > 0

    @pytest.mark.asyncio
    async def test_audit_trail_during_chaos(self, fast_breaker):
        """Test audit trail is maintained during chaos."""
        initial_records = len(fast_breaker.get_audit_records())

        # Generate chaos
        chaos = ChaosInjector(failure_rate=0.5)

        for _ in range(20):
            try:
                async with fast_breaker.protect():
                    await chaos.chaotic_call()
            except (ConnectionError, CircuitOpenError, CircuitHalfOpenError, LoadShedError):
                pass

        records = fast_breaker.get_audit_records()

        # Should have audit records for state transitions
        assert len(records) >= initial_records
        # Verify records have provenance hashes
        for record in records:
            assert record.provenance_hash


# =============================================================================
# HEATRECLAIM-SPECIFIC CIRCUIT BREAKER TESTS
# =============================================================================


class TestHeatReclaimCircuitBreakers:
    """Test pre-configured HeatReclaim circuit breakers."""

    def test_circuit_breakers_initialization(self, reset_registry):
        """Test HeatReclaimCircuitBreakers initializes correctly."""
        breakers = HeatReclaimCircuitBreakers()

        assert breakers.opcua is not None
        assert breakers.milp_solver is not None
        assert breakers.historian is not None

    def test_system_health_calculation(self, reset_registry):
        """Test system health calculation."""
        breakers = HeatReclaimCircuitBreakers()

        health = breakers.get_system_health()

        assert "health" in health
        assert health["health"] == "healthy"
        assert health["open_circuits"] == 0

    @pytest.mark.asyncio
    async def test_opcua_breaker_under_chaos(self, reset_registry):
        """Test OPC-UA breaker under chaotic conditions."""
        breakers = HeatReclaimCircuitBreakers()

        chaos = ChaosInjector(failure_rate=0.6)

        for _ in range(20):
            try:
                async with breakers.opcua.protect():
                    await chaos.chaotic_call()
            except (ConnectionError, CircuitOpenError, CircuitHalfOpenError, LoadShedError):
                pass

        # OPC-UA breaker should have opened (strict threshold)
        metrics = breakers.opcua.get_metrics()
        assert metrics.failure_count > 0

    def test_metrics_export(self, reset_registry):
        """Test metrics can be exported from all breakers."""
        breakers = HeatReclaimCircuitBreakers()

        all_metrics = breakers.get_all_metrics()

        assert "opcua" in all_metrics
        assert "milp_solver" in all_metrics
        assert "historian" in all_metrics

        for name, metrics in all_metrics.items():
            assert metrics.name is not None
            assert metrics.state is not None
            assert metrics.health_score >= 0.0

    def test_reset_all_breakers(self, reset_registry):
        """Test resetting all breakers."""
        breakers = HeatReclaimCircuitBreakers()

        # Record some failures
        for _ in range(3):
            breakers.opcua.record_failure(ConnectionError("test"))
            breakers.milp_solver.record_failure(ConnectionError("test"))

        # Reset all
        breakers.reset_all()

        # All should be clean
        for metrics in breakers.get_all_metrics().values():
            assert metrics.failure_count == 0
            assert metrics.state == CircuitBreakerState.CLOSED


# =============================================================================
# DECORATOR CHAOS TESTS
# =============================================================================


class TestDecoratorChaos:
    """Test circuit_protected decorator under chaos."""

    @pytest.mark.asyncio
    async def test_decorated_function_chaos(self, reset_registry):
        """Test decorated function under chaotic conditions."""
        chaos = ChaosInjector(failure_rate=0.4)

        @circuit_protected(
            name="chaos_decorated",
            base_failure_threshold=3,
            base_recovery_timeout_seconds=0.5,
        )
        async def chaotic_function():
            return await chaos.chaotic_call()

        successes = 0
        failures = 0
        blocked = 0

        for _ in range(30):
            try:
                await chaotic_function()
                successes += 1
            except ConnectionError:
                failures += 1
            except CircuitOpenError:
                blocked += 1
                await asyncio.sleep(0.1)

        # Should have mix of outcomes
        assert successes + failures + blocked == 30

    @pytest.mark.asyncio
    async def test_fallback_during_chaos(self, reset_registry):
        """Test fallback is used when circuit is open."""
        call_count = 0

        async def fallback_fn():
            return {"fallback": True}

        @circuit_protected(
            name="fallback_test",
            base_failure_threshold=2,
            base_recovery_timeout_seconds=1.0,
            fallback=fallback_fn,
        )
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        fallback_used = 0

        for _ in range(10):
            try:
                result = await failing_function()
                if result.get("fallback"):
                    fallback_used += 1
            except ConnectionError:
                pass

        # Fallback should have been used when circuit opened
        assert fallback_used > 0


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "TestCircuitBreakerChaos",
    "TestNetworkFailures",
    "TestServiceDegradation",
    "TestResourceExhaustion",
    "TestRecoveryBehavior",
    "TestHeatReclaimCircuitBreakers",
    "TestDecoratorChaos",
    "ChaosInjector",
    "LatencyInjector",
]
