# -*- coding: utf-8 -*-
"""
Integration Tests for LLM Performance and Load Testing.

Tests concurrent requests, throughput, latency benchmarks, memory usage,
and scalability of the LLM system.

Test Coverage:
- Concurrent requests (100+ simultaneous)
- Throughput measurement (requests/second)
- Latency benchmarks (P50, P95, P99)
- Memory usage under load
- Rate limiter performance
- Cost tracker performance
- Stress testing (breaking points)
"""

import pytest
import asyncio
import time
import psutil
import os
from typing import List
from statistics import mean, median

from llm.llm_router import LLMRouter, RoutingStrategy
from llm.providers.base_provider import GenerationRequest, GenerationResponse
from llm.rate_limiter import RateLimiter, RateLimitExceededError
from llm.cost_tracker import CostTracker


# ============================================================================
# Concurrent Request Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    @pytest.mark.asyncio
    async def test_10_concurrent_requests(self, mock_router, simple_request):
        """Test handling 10 concurrent requests."""
        # Make 10 concurrent requests
        tasks = [mock_router.generate(simple_request) for _ in range(10)]

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # All should succeed
        assert len(responses) == 10
        assert all(isinstance(r, GenerationResponse) for r in responses)

        # Calculate throughput
        throughput = len(responses) / elapsed

        print(f"\n[Concurrent 10] Completed in {elapsed:.2f}s")
        print(f"[Concurrent 10] Throughput: {throughput:.2f} req/s")

    @pytest.mark.asyncio
    async def test_50_concurrent_requests(self, mock_router, simple_request):
        """Test handling 50 concurrent requests."""
        tasks = [mock_router.generate(simple_request) for _ in range(50)]

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        assert len(responses) == 50
        throughput = len(responses) / elapsed

        print(f"\n[Concurrent 50] Completed in {elapsed:.2f}s")
        print(f"[Concurrent 50] Throughput: {throughput:.2f} req/s")

    @pytest.mark.asyncio
    async def test_100_concurrent_requests(self, mock_router, simple_request):
        """Test handling 100 concurrent requests."""
        tasks = [mock_router.generate(simple_request) for _ in range(100)]

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        assert len(responses) == 100
        throughput = len(responses) / elapsed

        print(f"\n[Concurrent 100] Completed in {elapsed:.2f}s")
        print(f"[Concurrent 100] Throughput: {throughput:.2f} req/s")

        # Performance assertion
        assert throughput > 10, f"Throughput {throughput:.2f} below target 10 req/s"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_500_concurrent_requests_stress(self, mock_router, simple_request):
        """Stress test with 500 concurrent requests."""
        tasks = [mock_router.generate(simple_request) for _ in range(500)]

        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Count successes and failures
        successes = [r for r in responses if isinstance(r, GenerationResponse)]
        failures = [r for r in responses if isinstance(r, Exception)]

        success_rate = len(successes) / len(responses) * 100
        throughput = len(successes) / elapsed

        print(f"\n[Stress 500] Completed in {elapsed:.2f}s")
        print(f"[Stress 500] Success rate: {success_rate:.1f}%")
        print(f"[Stress 500] Throughput: {throughput:.2f} req/s")
        print(f"[Stress 500] Failures: {len(failures)}")

        # Should maintain high success rate
        assert success_rate > 95, f"Success rate {success_rate:.1f}% below 95% target"


# ============================================================================
# Latency Benchmark Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestLatencyBenchmarks:
    """Test latency measurements and targets."""

    @pytest.mark.asyncio
    async def test_latency_distribution(self, mock_router, simple_request):
        """Measure latency distribution (P50, P95, P99)."""
        latencies = []

        # Run 100 requests to get good distribution
        for _ in range(100):
            response = await mock_router.generate(simple_request)
            latencies.append(response.generation_time_ms)

        # Sort for percentile calculation
        latencies.sort()

        # Calculate percentiles
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        p_mean = mean(latencies)

        print(f"\n[Latency Distribution]")
        print(f"  Mean: {p_mean:.0f}ms")
        print(f"  P50 (median): {p50:.0f}ms")
        print(f"  P95: {p95:.0f}ms")
        print(f"  P99: {p99:.0f}ms")
        print(f"  Min: {latencies[0]:.0f}ms")
        print(f"  Max: {latencies[-1]:.0f}ms")

        # Validate P95 target (<2000ms)
        assert p95 < 2000, f"P95 latency {p95:.0f}ms exceeds 2000ms target"

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_real_api_latency(self, real_router, simple_request):
        """Measure real API latency."""
        if not real_router:
            pytest.skip("No real providers available")

        latencies = []

        # Run 20 requests (cost-effective for real APIs)
        for i in range(20):
            response = await real_router.generate(simple_request)
            latencies.append(response.generation_time_ms)
            if i < 19:  # Don't wait after last request
                await asyncio.sleep(1)  # Rate limit friendly

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\n[Real API Latency]")
        print(f"  P50: {p50:.0f}ms")
        print(f"  P95: {p95:.0f}ms")

        # Validate target
        assert p95 < 2000, f"Real API P95 {p95:.0f}ms exceeds target"


# ============================================================================
# Throughput Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestThroughput:
    """Test system throughput capabilities."""

    @pytest.mark.asyncio
    async def test_sustained_throughput_1_minute(self, mock_router, simple_request):
        """Test sustained throughput over 1 minute."""
        start_time = time.time()
        end_time = start_time + 60  # 1 minute

        request_count = 0
        errors = 0

        while time.time() < end_time:
            try:
                await mock_router.generate(simple_request)
                request_count += 1
            except Exception as e:
                errors += 1
                print(f"Error during throughput test: {e}")

        elapsed = time.time() - start_time
        throughput = request_count / elapsed

        print(f"\n[Sustained Throughput]")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Requests: {request_count}")
        print(f"  Errors: {errors}")
        print(f"  Throughput: {throughput:.2f} req/s")

        assert throughput > 5, f"Throughput {throughput:.2f} below 5 req/s target"

    @pytest.mark.asyncio
    async def test_burst_throughput(self, mock_router, simple_request):
        """Test burst throughput (as fast as possible for 10s)."""
        start_time = time.time()
        duration = 10  # seconds

        tasks = []
        request_count = 0

        while time.time() - start_time < duration:
            task = asyncio.create_task(mock_router.generate(simple_request))
            tasks.append(task)
            request_count += 1

        # Wait for all to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successes = [r for r in responses if isinstance(r, GenerationResponse)]
        throughput = len(successes) / elapsed

        print(f"\n[Burst Throughput]")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Requests launched: {request_count}")
        print(f"  Successful: {len(successes)}")
        print(f"  Throughput: {throughput:.2f} req/s")


# ============================================================================
# Memory Usage Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage under load."""

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_router, simple_request):
        """Test memory usage doesn't leak during heavy load."""
        process = psutil.Process(os.getpid())

        # Measure initial memory
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        # Run 1000 requests
        for i in range(10):  # 10 batches of 100
            tasks = [mock_router.generate(simple_request) for _ in range(100)]
            await asyncio.gather(*tasks)

            if i % 2 == 0:  # Sample every other batch
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory_mb - initial_memory_mb
                print(f"\n[Memory] After {(i+1)*100} requests: "
                      f"{current_memory_mb:.1f}MB (+{memory_increase:.1f}MB)")

        # Measure final memory
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory_mb - initial_memory_mb

        print(f"\n[Memory Final]")
        print(f"  Initial: {initial_memory_mb:.1f}MB")
        print(f"  Final: {final_memory_mb:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")

        # Memory increase should be reasonable (<500MB for 1000 requests)
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB (>500MB threshold)"


# ============================================================================
# Rate Limiter Performance Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestRateLimiterPerformance:
    """Test rate limiter performance."""

    @pytest.mark.asyncio
    async def test_rate_limiter_accuracy(self, rate_limiter):
        """Test rate limiter enforces limits accurately."""
        # Rate limiter is configured for 60 req/min = 1 req/s

        request_times = []
        start_time = time.time()

        # Make 10 requests
        for _ in range(10):
            async with rate_limiter.acquire():
                request_times.append(time.time())

        elapsed = time.time() - start_time

        # Should take approximately 9 seconds (10 requests at 1/s = 9 intervals)
        print(f"\n[Rate Limiter] 10 requests took {elapsed:.2f}s (expected ~9s)")

        # Allow some tolerance
        assert 8.0 < elapsed < 11.0, f"Expected ~9s, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_rate_limiter_queue_handling(self):
        """Test rate limiter handles queued requests correctly."""
        # Fast rate limiter for testing
        limiter = RateLimiter(
            requests_per_minute=120,  # 2 per second
            tokens_per_minute=10000,
            enable_queuing=True,
            max_wait_time=10.0,
            name="queue-test"
        )

        start_time = time.time()

        # Launch 20 requests concurrently
        async def make_request():
            async with limiter.acquire():
                pass

        tasks = [make_request() for _ in range(20)]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Should take approximately 10 seconds (20 requests at 2/s = 10s)
        print(f"\n[Rate Limiter Queue] 20 concurrent requests took {elapsed:.2f}s (expected ~10s)")

        assert 8.0 < elapsed < 12.0


# ============================================================================
# Cost Tracker Performance Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestCostTrackerPerformance:
    """Test cost tracker performance."""

    @pytest.mark.asyncio
    async def test_cost_tracker_high_volume(self, cost_tracker):
        """Test cost tracker handles high volume of records."""
        from llm.providers.base_provider import TokenUsage

        # Track 10,000 usage records
        start_time = time.time()

        for i in range(10000):
            usage = TokenUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost_usd=0.0001,
                output_cost_usd=0.00005,
                total_cost_usd=0.00015
            )

            cost_tracker.track_usage(
                provider="test-provider",
                tenant_id="test-tenant",
                agent_id=f"agent-{i % 10}",  # 10 different agents
                model_id="test-model",
                usage=usage,
                request_id=f"req-{i}"
            )

        elapsed = time.time() - start_time

        # Get summary (this should also be fast)
        summary_start = time.time()
        summary = cost_tracker.get_summary()
        summary_time = time.time() - summary_start

        print(f"\n[Cost Tracker Volume]")
        print(f"  Tracked 10,000 records in {elapsed:.2f}s")
        print(f"  Rate: {10000/elapsed:.0f} records/s")
        print(f"  Summary generation: {summary_time*1000:.0f}ms")
        print(f"  Total cost: ${summary.total_cost_usd:.2f}")
        print(f"  Total tokens: {summary.total_tokens:,}")

        assert elapsed < 5.0, f"Tracking 10K records took {elapsed:.2f}s (>5s threshold)"
        assert summary_time < 1.0, f"Summary generation took {summary_time:.2f}s (>1s threshold)"


# ============================================================================
# Scalability Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
class TestScalability:
    """Test system scalability."""

    @pytest.mark.asyncio
    async def test_scalability_increasing_load(self, mock_router, simple_request):
        """Test system behavior under increasing load."""
        load_levels = [10, 50, 100, 200, 500]
        results = []

        for load in load_levels:
            start_time = time.time()

            tasks = [mock_router.generate(simple_request) for _ in range(load)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            elapsed = time.time() - start_time
            successes = [r for r in responses if isinstance(r, GenerationResponse)]
            success_rate = len(successes) / len(responses) * 100
            throughput = len(successes) / elapsed

            results.append({
                "load": load,
                "time": elapsed,
                "success_rate": success_rate,
                "throughput": throughput
            })

            print(f"\n[Scalability Load={load}]")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Throughput: {throughput:.2f} req/s")

        # Check that system maintains performance
        for result in results:
            assert result["success_rate"] > 95, \
                f"Success rate dropped to {result['success_rate']:.1f}% at load={result['load']}"


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary information."""
    print("\n" + "=" * 80)
    print("LLM Performance and Load Testing")
    print("=" * 80)
    print("\nTest Coverage:")
    print("  - Concurrent requests (10, 50, 100, 500)")
    print("  - Latency distribution (P50, P95, P99)")
    print("  - Throughput measurement (sustained & burst)")
    print("  - Memory usage under load")
    print("  - Rate limiter accuracy and queue handling")
    print("  - Cost tracker high-volume performance")
    print("  - Scalability under increasing load")
    print("\nPerformance Targets:")
    print("  - P95 latency: <2000ms")
    print("  - Throughput: >10 req/s")
    print("  - Success rate: >95%")
    print("  - Memory increase: <500MB per 1000 requests")
    print("=" * 80)
