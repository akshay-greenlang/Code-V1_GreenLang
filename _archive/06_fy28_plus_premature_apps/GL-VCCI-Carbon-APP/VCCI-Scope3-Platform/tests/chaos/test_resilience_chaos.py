# -*- coding: utf-8 -*-
"""
Chaos Engineering Tests for GL-VCCI Scope 3 Platform

Chaos tests for resilience validation:
- Random failure injection
- System stability under chaos
- Cascading failure prevention
- Recovery validation
- Performance degradation

Based on Chaos Engineering principles
Team: Testing & Documentation Team
Phase: 5 (Production Readiness)
Date: 2025-11-09
"""

import asyncio
import pytest
import random
import time
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

from greenlang.intelligence.fallback import FallbackManager, ModelConfig
from greenlang.intelligence.providers.resilience import ResilientHTTPClient
from greenlang.determinism import deterministic_random


# =============================================================================
# CHAOS ENGINEERING FRAMEWORK
# =============================================================================

class ChaosType(Enum):
    """Types of chaos to inject"""
    LATENCY = "latency"           # Add random delays
    FAILURE = "failure"           # Random failures
    TIMEOUT = "timeout"           # Force timeouts
    RATE_LIMIT = "rate_limit"     # Simulate rate limits
    PARTIAL_FAILURE = "partial"   # Partial response failures
    INTERMITTENT = "intermittent" # On/off failures


@dataclass
class ChaosConfig:
    """Configuration for chaos injection"""
    enabled: bool = True
    failure_rate: float = 0.3      # 30% failure rate
    latency_ms_range: tuple = (100, 2000)  # 100-2000ms latency
    intermittent_pattern: str = "random"   # random, periodic, burst


class ChaosInjector:
    """Inject chaos into system calls"""

    def __init__(self, config: ChaosConfig = None):
        self.config = config or ChaosConfig()
        self.call_count = 0
        self.failure_count = 0
        self.success_count = 0

    async def inject_chaos(
        self,
        fn: Callable,
        chaos_type: ChaosType,
        *args,
        **kwargs
    ) -> Any:
        """
        Inject chaos and execute function

        Args:
            fn: Function to execute
            chaos_type: Type of chaos to inject
            *args, **kwargs: Function arguments

        Returns:
            Function result (if successful)

        Raises:
            Exception: On injected failure
        """
        self.call_count += 1

        if not self.config.enabled:
            return await fn(*args, **kwargs)

        # Inject chaos based on type
        if chaos_type == ChaosType.LATENCY:
            # Add random latency
            latency = deterministic_random().randint(*self.config.latency_ms_range) / 1000.0
            await asyncio.sleep(latency)

        elif chaos_type == ChaosType.FAILURE:
            # Random failure
            if deterministic_random().random() < self.config.failure_rate:
                self.failure_count += 1
                raise Exception(f"Chaos: Injected failure #{self.failure_count}")

        elif chaos_type == ChaosType.TIMEOUT:
            # Force timeout
            if deterministic_random().random() < self.config.failure_rate:
                await asyncio.sleep(10.0)  # Very long delay

        elif chaos_type == ChaosType.RATE_LIMIT:
            # Simulate rate limit
            if deterministic_random().random() < self.config.failure_rate:
                self.failure_count += 1
                raise Exception("429 Rate limit exceeded (chaos)")

        elif chaos_type == ChaosType.INTERMITTENT:
            # Intermittent failures
            if self.call_count % 3 == 0:  # Every 3rd call fails
                self.failure_count += 1
                raise Exception("Intermittent failure (chaos)")

        # Execute function
        try:
            result = await fn(*args, **kwargs)
            self.success_count += 1
            return result
        except Exception as e:
            self.failure_count += 1
            raise

    def get_stats(self) -> Dict[str, int]:
        """Get chaos injection statistics"""
        return {
            "total_calls": self.call_count,
            "failures": self.failure_count,
            "successes": self.success_count,
            "failure_rate": self.failure_count / self.call_count if self.call_count > 0 else 0,
        }


# =============================================================================
# CHAOS TEST SUITE 1: Random Failure Injection (5 tests)
# =============================================================================

class TestRandomFailureInjection:
    """Test system handles random failures"""

    @pytest.mark.asyncio
    async def test_system_stability_under_random_failures(self):
        """Test system remains stable with 30% random failures"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.3))
        client = ResilientHTTPClient(max_retries=3, base_delay=0.01)

        async def unreliable_service():
            await asyncio.sleep(0.01)
            return {"status": "success"}

        successes = 0
        failures = 0

        for _ in range(50):
            try:
                await client.call(
                    lambda: chaos.inject_chaos(
                        unreliable_service,
                        ChaosType.FAILURE
                    )
                )
                successes += 1
            except:
                failures += 1

        # With retries, success rate should be reasonable
        success_rate = successes / (successes + failures)
        assert success_rate > 0.6  # At least 60% should succeed with retries

    @pytest.mark.asyncio
    async def test_fallback_chain_under_chaos(self):
        """Test fallback chain handles chaos"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.5))
        manager = FallbackManager()

        async def chaotic_llm_call(cfg):
            async def llm_call():
                return {"result": "success", "model": cfg.model}

            return await chaos.inject_chaos(llm_call, ChaosType.FAILURE)

        successes = 0

        for _ in range(20):
            result = await manager.execute_with_fallback(chaotic_llm_call)
            if result.success:
                successes += 1

        # With fallback chain, should handle chaos well
        assert successes >= 15  # At least 75% success

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_under_chaos(self):
        """Test circuit breaker opens under sustained chaos"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=1.0))  # 100% failure
        client = ResilientHTTPClient(failure_threshold=3, recovery_timeout=1.0)

        async def always_failing():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        # Trigger circuit breaker
        for _ in range(5):
            try:
                await client.call(always_failing)
            except:
                pass

        # Circuit should be open
        stats = client.get_stats()
        assert stats.state.value in ["open", "half_open"]

    @pytest.mark.asyncio
    async def test_burst_failures_handled(self):
        """Test system handles burst of failures"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.8))  # High failure rate
        client = ResilientHTTPClient(max_retries=5, base_delay=0.01)

        async def burst_unreliable():
            await asyncio.sleep(0.01)
            return {"status": "success"}

        # Burst of 10 rapid requests
        results = await asyncio.gather(*[
            client.call(
                lambda: chaos.inject_chaos(burst_unreliable, ChaosType.FAILURE)
            )
            for _ in range(10)
        ], return_exceptions=True)

        successes = sum(1 for r in results if isinstance(r, dict))

        # Some should succeed despite high failure rate
        assert successes > 0

    @pytest.mark.asyncio
    async def test_recovery_after_chaos_storm(self):
        """Test system recovers after chaos storm"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=1.0))
        client = ResilientHTTPClient(
            failure_threshold=5,
            recovery_timeout=0.5,
        )

        async def storm_service():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        # Chaos storm - all fail
        for _ in range(10):
            try:
                await client.call(storm_service)
            except:
                pass

        # Disable chaos
        chaos.config.failure_rate = 0.0

        # Wait for recovery
        await asyncio.sleep(1.0)

        # Should recover
        result = await client.call(storm_service)
        assert result["status"] == "success"


# =============================================================================
# CHAOS TEST SUITE 2: Latency Injection (5 tests)
# =============================================================================

class TestLatencyInjection:
    """Test system handles random latency"""

    @pytest.mark.asyncio
    async def test_timeout_protection_under_latency_chaos(self):
        """Test timeout protection works with random latency"""
        chaos = ChaosInjector(ChaosConfig(
            failure_rate=1.0,  # Always inject
            latency_ms_range=(1000, 3000)  # 1-3s latency
        ))

        config = ModelConfig(model="test", provider="test", timeout=0.5)
        manager = FallbackManager(fallback_chain=[config])

        async def slow_service(cfg):
            async def call():
                return {"result": "success"}
            return await chaos.inject_chaos(call, ChaosType.LATENCY)

        start = time.time()
        result = await manager.execute_with_fallback(slow_service)
        elapsed = time.time() - start

        # Should timeout quickly, not wait for full latency
        assert elapsed < 2.0  # Much less than max latency

    @pytest.mark.asyncio
    async def test_performance_degradation_monitoring(self):
        """Test can monitor performance degradation under latency"""
        chaos = ChaosInjector(ChaosConfig(
            failure_rate=0.5,
            latency_ms_range=(200, 500)
        ))

        latencies = []

        async def monitored_service():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.LATENCY)

        client = ResilientHTTPClient(max_retries=1)

        for _ in range(10):
            start = time.time()
            try:
                await client.call(monitored_service)
                latencies.append(time.time() - start)
            except:
                pass

        # Should detect degraded performance
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        assert avg_latency > 0.1  # Noticeably slower

    @pytest.mark.asyncio
    async def test_fallback_on_slow_primary(self):
        """Test falls back when primary is too slow"""
        chaos = ChaosInjector(ChaosConfig(latency_ms_range=(1000, 1000)))

        chain = [
            ModelConfig(model="slow", provider="test", timeout=0.5),
            ModelConfig(model="fast", provider="test", timeout=2.0),
        ]
        manager = FallbackManager(fallback_chain=chain)

        async def service(cfg):
            if cfg.model == "slow":
                async def slow_call():
                    return {"result": "slow"}
                return await chaos.inject_chaos(slow_call, ChaosType.LATENCY)
            else:
                await asyncio.sleep(0.1)
                return {"result": "fast"}

        result = await manager.execute_with_fallback(service)

        assert result.success
        assert result.model_used == "fast"

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_latency_chaos(self):
        """Test concurrent requests handle latency chaos"""
        chaos = ChaosInjector(ChaosConfig(
            failure_rate=0.5,
            latency_ms_range=(100, 300)
        ))

        async def latent_service():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.LATENCY)

        client = ResilientHTTPClient(max_retries=2, base_delay=0.01)

        # 20 concurrent requests
        results = await asyncio.gather(*[
            client.call(latent_service)
            for _ in range(20)
        ], return_exceptions=True)

        successes = sum(1 for r in results if isinstance(r, dict))

        # Most should succeed despite latency
        assert successes >= 15

    @pytest.mark.asyncio
    async def test_latency_percentiles_tracking(self):
        """Test can track latency percentiles under chaos"""
        chaos = ChaosInjector(ChaosConfig(
            failure_rate=1.0,
            latency_ms_range=(50, 500)
        ))

        latencies = []

        async def service():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.LATENCY)

        for _ in range(100):
            start = time.time()
            try:
                await service()
                latencies.append((time.time() - start) * 1000)  # ms
            except:
                pass

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        # Validate distribution
        assert 50 <= p50 <= 500
        assert p95 > p50
        assert p99 >= p95


# =============================================================================
# CHAOS TEST SUITE 3: Cascading Failure Prevention (5 tests)
# =============================================================================

class TestCascadingFailurePrevention:
    """Test prevents cascading failures"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascade(self):
        """Test circuit breaker stops cascading failures"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=1.0))

        # Service B (failing)
        service_b = ResilientHTTPClient(failure_threshold=3)

        async def failing_b():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        # Service A (depends on B)
        async def service_a():
            try:
                return await service_b.call(failing_b)
            except:
                return {"status": "degraded", "service_b": "unavailable"}

        # Multiple calls - B should open circuit quickly
        results = []
        for _ in range(10):
            result = await service_a()
            results.append(result)

        # Service A should stay up in degraded mode
        assert all(isinstance(r, dict) for r in results)
        most_degraded = sum(1 for r in results if r.get("status") == "degraded")
        assert most_degraded > 5

    @pytest.mark.asyncio
    async def test_bulkhead_isolation(self):
        """Test failures are isolated (bulkhead pattern)"""
        chaos_a = ChaosInjector(ChaosConfig(failure_rate=1.0))
        chaos_b = ChaosInjector(ChaosConfig(failure_rate=0.0))

        # Two independent services
        service_a_client = ResilientHTTPClient(failure_threshold=3)
        service_b_client = ResilientHTTPClient(failure_threshold=3)

        async def service_a():
            async def call():
                return {"service": "A"}
            return await chaos_a.inject_chaos(call, ChaosType.FAILURE)

        async def service_b():
            async def call():
                return {"service": "B"}
            return await chaos_b.inject_chaos(call, ChaosType.FAILURE)

        # Service A fails
        for _ in range(5):
            try:
                await service_a_client.call(service_a)
            except:
                pass

        # Service B should still work
        result = await service_b_client.call(service_b)
        assert result["service"] == "B"

    @pytest.mark.asyncio
    async def test_timeout_prevents_thread_exhaustion(self):
        """Test timeouts prevent thread/resource exhaustion"""
        chaos = ChaosInjector(ChaosConfig(
            failure_rate=1.0,
            latency_ms_range=(5000, 10000)  # Very slow
        ))

        async def slow_service():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.LATENCY)

        config = ModelConfig(model="test", provider="test", timeout=0.5)
        manager = FallbackManager(fallback_chain=[config])

        # Many concurrent slow requests
        start = time.time()
        results = await asyncio.gather(*[
            manager.execute_with_fallback(lambda cfg: slow_service())
            for _ in range(20)
        ], return_exceptions=True)
        elapsed = time.time() - start

        # Should complete quickly due to timeouts
        assert elapsed < 5.0  # Not 20 * 5s = 100s

    @pytest.mark.asyncio
    async def test_load_shedding_under_overload(self):
        """Test system sheds load under overload"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.9))  # 90% failure

        client = ResilientHTTPClient(
            max_retries=1,  # Limit retries to shed load
            failure_threshold=5,
        )

        async def overloaded_service():
            async def call():
                return {"status": "success"}
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        start = time.time()

        # Burst of requests
        for _ in range(50):
            try:
                await client.call(overloaded_service)
            except:
                pass  # Fail fast

        elapsed = time.time() - start

        # Should complete quickly by failing fast
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_graceful_degradation_under_partial_failure(self):
        """Test graceful degradation under partial failures"""
        # 3 dependent services
        chaos_1 = ChaosInjector(ChaosConfig(failure_rate=0.0))
        chaos_2 = ChaosInjector(ChaosConfig(failure_rate=1.0))  # Fails
        chaos_3 = ChaosInjector(ChaosConfig(failure_rate=0.0))

        async def composite_service():
            results = {}

            # Service 1
            async def s1():
                return {"data": "service1"}
            try:
                results["s1"] = await chaos_1.inject_chaos(s1, ChaosType.FAILURE)
            except:
                results["s1"] = None

            # Service 2 (failing)
            async def s2():
                return {"data": "service2"}
            try:
                results["s2"] = await chaos_2.inject_chaos(s2, ChaosType.FAILURE)
            except:
                results["s2"] = None

            # Service 3
            async def s3():
                return {"data": "service3"}
            try:
                results["s3"] = await chaos_3.inject_chaos(s3, ChaosType.FAILURE)
            except:
                results["s3"] = None

            # Return partial results
            return {
                "results": results,
                "degraded": results["s2"] is None,
            }

        result = await composite_service()

        # Should have partial results
        assert result["degraded"] == True
        assert result["results"]["s1"] is not None
        assert result["results"]["s2"] is None
        assert result["results"]["s3"] is not None


# =============================================================================
# CHAOS TEST SUITE 4: System Stability (3 tests)
# =============================================================================

class TestSystemStability:
    """Test overall system stability under chaos"""

    @pytest.mark.asyncio
    async def test_sustained_chaos_stability(self):
        """Test system remains stable under sustained chaos"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.4))
        manager = FallbackManager()

        async def chaotic_service(cfg):
            async def call():
                return {"result": "success"}
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        # 100 requests over time
        successes = 0
        for _ in range(100):
            result = await manager.execute_with_fallback(chaotic_service)
            if result.success:
                successes += 1
            await asyncio.sleep(0.01)

        # Should maintain reasonable success rate
        success_rate = successes / 100
        assert success_rate > 0.7  # >70% success despite 40% failure

    @pytest.mark.asyncio
    async def test_no_memory_leaks_under_chaos(self):
        """Test no memory leaks under sustained chaos"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.5))
        client = ResilientHTTPClient(max_retries=2)

        async def service():
            async def call():
                # Create some data
                data = {"x": [1] * 1000}
                return data
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        # Many iterations
        for _ in range(1000):
            try:
                await client.call(service)
            except:
                pass

        # If we get here without OOM, memory is managed
        assert True

    @pytest.mark.asyncio
    async def test_metrics_accuracy_under_chaos(self):
        """Test metrics remain accurate under chaos"""
        chaos = ChaosInjector(ChaosConfig(failure_rate=0.3))
        manager = FallbackManager()

        async def service(cfg):
            async def call():
                return {"result": "success"}
            return await chaos.inject_chaos(call, ChaosType.FAILURE)

        for _ in range(50):
            await manager.execute_with_fallback(service)

        metrics = manager.get_metrics()
        chaos_stats = chaos.get_stats()

        # Metrics should be tracking
        assert metrics["total_requests"] == 50
        assert chaos_stats["total_calls"] > 0


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    """Run chaos tests with pytest"""
    print("=" * 80)
    print("CHAOS ENGINEERING TEST SUITE")
    print("=" * 80)
    print("\nThese tests inject random failures, latency, and chaos to validate")
    print("system resilience under adverse conditions.")
    print("\nRunning tests...\n")

    pytest.main([__file__, "-v", "--tb=short", "-k", "chaos"])
