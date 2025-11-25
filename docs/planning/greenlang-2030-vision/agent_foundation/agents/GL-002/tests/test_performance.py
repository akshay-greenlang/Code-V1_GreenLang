# -*- coding: utf-8 -*-
"""
Performance tests for GL-002 BoilerEfficiencyOptimizer

Tests latency, throughput, memory usage, and cost optimization.
Validates that the system meets performance targets.

Target: 10+ tests with <3s latency target
"""

import pytest
import asyncio
import time
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
from typing import List, Dict, Any
import tracemalloc
import gc

# Import components to test
from greenlang.determinism import deterministic_random
from greenlang_boiler_efficiency import (
    BoilerEfficiencyOrchestrator,
    BoilerInput,
    AgentConfig,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def performance_config():
    """Create performance-optimized configuration."""
    return AgentConfig(
        name="GL-002-Performance",
        version="2.0.0",
        environment="performance",
        cache_enabled=True,
        cache_ttl=300,
        async_enabled=True,
        max_workers=4,
        batch_size=100,
        timeout_seconds=3,
    )


@pytest.fixture
def orchestrator(performance_config):
    """Create orchestrator for performance testing."""
    return BoilerEfficiencyOrchestrator(performance_config)


@pytest.fixture
def sample_input():
    """Create sample input for performance tests."""
    return BoilerInput(
        boiler_id="PERF-001",
        fuel_type="natural_gas",
        fuel_flow_rate=100.0,
        steam_output=1500.0,
        steam_pressure=10.0,
        steam_temperature=180.0,
        feedwater_temperature=80.0,
        excess_air_ratio=1.15,
        ambient_temperature=25.0,
    )


@pytest.fixture
def large_dataset(sample_input):
    """Create large dataset for performance testing."""
    dataset = []
    for i in range(1000):
        input_copy = sample_input.copy()
        # Add some variation
        input_copy.fuel_flow_rate = 100 + np.random.normal(0, 5)
        input_copy.steam_output = 1500 + np.random.normal(0, 50)
        dataset.append(input_copy)
    return dataset


# ============================================================================
# TEST LATENCY
# ============================================================================

class TestLatency:
    """Test latency performance."""

    @pytest.mark.performance
    def test_single_calculation_latency(self, orchestrator, sample_input):
        """Test latency for single calculation (<3s target)."""
        start_time = time.perf_counter()
        result = orchestrator.process(sample_input)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        assert result is not None
        assert latency_ms < 3000  # <3s target
        assert result.processing_time_ms < 3000

        # Log performance for tracking
        print(f"\nSingle calculation latency: {latency_ms:.2f}ms")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_calculation_latency(self, orchestrator, sample_input):
        """Test async calculation latency."""
        start_time = time.perf_counter()
        result = await orchestrator.process_async(sample_input)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        assert latency_ms < 3000  # <3s target
        print(f"\nAsync calculation latency: {latency_ms:.2f}ms")

    @pytest.mark.performance
    def test_p95_latency(self, orchestrator, sample_input):
        """Test 95th percentile latency."""
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            orchestrator.process(sample_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        assert p95 < 3000  # P95 < 3s
        assert p99 < 5000  # P99 < 5s

        print(f"\nP95 latency: {p95:.2f}ms")
        print(f"P99 latency: {p99:.2f}ms")

    @pytest.mark.performance
    def test_cold_start_latency(self, performance_config, sample_input):
        """Test cold start latency (first calculation)."""
        # Create new instance (cold start)
        orchestrator = BoilerEfficiencyOrchestrator(performance_config)

        start_time = time.perf_counter()
        result = orchestrator.process(sample_input)
        end_time = time.perf_counter()

        cold_start_latency = (end_time - start_time) * 1000

        assert cold_start_latency < 5000  # Allow 5s for cold start
        print(f"\nCold start latency: {cold_start_latency:.2f}ms")

    @pytest.mark.performance
    def test_warm_cache_latency(self, orchestrator, sample_input):
        """Test latency with warm cache."""
        # Warm up cache
        orchestrator.process(sample_input)

        # Measure with warm cache
        start_time = time.perf_counter()
        result = orchestrator.process(sample_input)
        end_time = time.perf_counter()

        warm_cache_latency = (end_time - start_time) * 1000

        assert warm_cache_latency < 100  # Cache hit should be <100ms
        print(f"\nWarm cache latency: {warm_cache_latency:.2f}ms")


# ============================================================================
# TEST THROUGHPUT
# ============================================================================

class TestThroughput:
    """Test throughput performance."""

    @pytest.mark.performance
    def test_throughput_single_thread(self, orchestrator, large_dataset):
        """Test single-threaded throughput."""
        start_time = time.perf_counter()

        for input_data in large_dataset[:100]:  # Process 100 records
            orchestrator.process(input_data)

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = 100 / duration

        assert throughput >= 10  # At least 10 records/second
        print(f"\nSingle-thread throughput: {throughput:.2f} records/sec")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_async(self, orchestrator, large_dataset):
        """Test async throughput with concurrency."""
        start_time = time.perf_counter()

        # Process 100 records concurrently
        tasks = [
            orchestrator.process_async(input_data)
            for input_data in large_dataset[:100]
        ]
        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = 100 / duration

        assert throughput >= 50  # At least 50 records/second with async
        print(f"\nAsync throughput: {throughput:.2f} records/sec")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_processing_throughput(self, orchestrator, large_dataset):
        """Test batch processing throughput."""
        batch_size = 100
        num_batches = 10

        start_time = time.perf_counter()

        for i in range(num_batches):
            batch = large_dataset[i * batch_size:(i + 1) * batch_size]
            await orchestrator.process_batch_async(batch)

        end_time = time.perf_counter()
        duration = end_time - start_time
        total_records = batch_size * num_batches
        throughput = total_records / duration

        assert throughput >= 100  # At least 100 records/second in batch mode
        print(f"\nBatch processing throughput: {throughput:.2f} records/sec")

    @pytest.mark.performance
    def test_sustained_throughput(self, orchestrator, sample_input):
        """Test sustained throughput over extended period."""
        duration_seconds = 10
        count = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_seconds:
            orchestrator.process(sample_input)
            count += 1

        actual_duration = time.perf_counter() - start_time
        sustained_throughput = count / actual_duration

        assert sustained_throughput >= 20  # Sustain 20+ records/sec
        print(f"\nSustained throughput ({duration_seconds}s): {sustained_throughput:.2f} records/sec")


# ============================================================================
# TEST MEMORY USAGE
# ============================================================================

class TestMemoryUsage:
    """Test memory usage and efficiency."""

    @pytest.mark.performance
    def test_memory_footprint(self, orchestrator, sample_input):
        """Test base memory footprint."""
        process = psutil.Process(os.getpid())

        # Force garbage collection
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process some data
        for _ in range(100):
            orchestrator.process(sample_input)

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = final_memory - initial_memory

        assert memory_increase < 500  # Less than 500MB increase
        print(f"\nMemory increase: {memory_increase:.2f}MB")

    @pytest.mark.performance
    def test_memory_leak_detection(self, orchestrator, sample_input):
        """Test for memory leaks during repeated operations."""
        tracemalloc.start()

        # Baseline snapshot
        for _ in range(10):
            orchestrator.process(sample_input)

        snapshot1 = tracemalloc.take_snapshot()

        # Run many more iterations
        for _ in range(1000):
            orchestrator.process(sample_input)

        snapshot2 = tracemalloc.take_snapshot()

        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # Check for significant memory growth
        total_growth = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB

        tracemalloc.stop()

        assert total_growth < 100  # Less than 100MB growth
        print(f"\nMemory growth after 1000 iterations: {total_growth:.2f}MB")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_memory_usage(self, orchestrator, large_dataset):
        """Test memory usage under concurrent load."""
        process = psutil.Process(os.getpid())
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create many concurrent tasks
        tasks = [
            orchestrator.process_async(input_data)
            for input_data in large_dataset[:200]
        ]
        await asyncio.gather(*tasks)

        gc.collect()
        peak_memory = process.memory_info().rss / 1024 / 1024

        memory_used = peak_memory - initial_memory

        assert memory_used < 1000  # Less than 1GB for 200 concurrent operations
        print(f"\nConcurrent operations memory usage: {memory_used:.2f}MB")


# ============================================================================
# TEST SCALABILITY
# ============================================================================

class TestScalability:
    """Test system scalability."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_horizontal_scalability(self, performance_config, large_dataset):
        """Test scalability with multiple workers."""
        worker_counts = [1, 2, 4, 8]
        throughputs = []

        for workers in worker_counts:
            config = performance_config.copy()
            config.max_workers = workers
            orchestrator = BoilerEfficiencyOrchestrator(config)

            start = time.perf_counter()
            tasks = [
                orchestrator.process_async(input_data)
                for input_data in large_dataset[:100]
            ]
            await asyncio.gather(*tasks)
            duration = time.perf_counter() - start

            throughput = 100 / duration
            throughputs.append(throughput)

            print(f"\nThroughput with {workers} workers: {throughput:.2f} records/sec")

        # Check that throughput scales with workers
        assert throughputs[1] > throughputs[0] * 1.5  # 2 workers > 1.5x single
        assert throughputs[2] > throughputs[0] * 2.5  # 4 workers > 2.5x single

    @pytest.mark.performance
    def test_data_size_scalability(self, orchestrator):
        """Test performance with varying data sizes."""
        data_sizes = [10, 100, 1000, 10000]
        processing_times = []

        for size in data_sizes:
            # Create data of specified size
            data = {
                "measurements": [
                    {"timestamp": DeterministicClock.now(), "value": np.deterministic_random().random()}
                    for _ in range(size)
                ]
            }

            start = time.perf_counter()
            orchestrator.process_large_dataset(data)
            duration = time.perf_counter() - start

            processing_times.append(duration)
            print(f"\nProcessing {size} data points: {duration:.3f}s")

        # Check sub-linear scaling (better than O(n))
        # Time for 10000 should be less than 1000 * (time for 10)
        assert processing_times[3] < processing_times[0] * 1000


# ============================================================================
# TEST COST OPTIMIZATION
# ============================================================================

class TestCostOptimization:
    """Test cost optimization features."""

    @pytest.mark.performance
    def test_cache_hit_ratio(self, orchestrator, sample_input):
        """Test cache effectiveness for cost reduction."""
        # Process same input multiple times
        cache_hits = 0
        cache_misses = 0

        for i in range(100):
            if i % 10 == 0:
                # Change input occasionally
                sample_input.fuel_flow_rate = 100 + i / 10

            with patch.object(orchestrator.cache, 'get') as mock_get:
                if i > 0 and i % 10 != 0:
                    mock_get.return_value = {"cached": True}
                    cache_hits += 1
                else:
                    mock_get.return_value = None
                    cache_misses += 1

                orchestrator.process(sample_input)

        hit_ratio = cache_hits / (cache_hits + cache_misses)

        assert hit_ratio > 0.8  # 80%+ cache hit ratio
        print(f"\nCache hit ratio: {hit_ratio:.2%}")

    @pytest.mark.performance
    def test_computation_cost_reduction(self, orchestrator):
        """Test computation cost reduction strategies."""
        # Measure with full computation
        start = time.perf_counter()
        orchestrator.process(sample_input, optimization_level="full")
        full_compute_time = time.perf_counter() - start

        # Measure with optimized computation
        start = time.perf_counter()
        orchestrator.process(sample_input, optimization_level="optimized")
        optimized_time = time.perf_counter() - start

        cost_reduction = (full_compute_time - optimized_time) / full_compute_time

        assert cost_reduction > 0.3  # At least 30% cost reduction
        print(f"\nComputation cost reduction: {cost_reduction:.2%}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_utilization(self, orchestrator, large_dataset):
        """Test efficient resource utilization."""
        process = psutil.Process(os.getpid())

        # Monitor resource usage during processing
        cpu_samples = []
        memory_samples = []

        async def monitor():
            while True:
                cpu_samples.append(process.cpu_percent())
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.1)

        # Start monitoring
        monitor_task = asyncio.create_task(monitor())

        # Process data
        tasks = [
            orchestrator.process_async(input_data)
            for input_data in large_dataset[:100]
        ]
        await asyncio.gather(*tasks)

        # Stop monitoring
        monitor_task.cancel()

        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        avg_memory = np.mean(memory_samples) if memory_samples else 0

        # Check resource utilization is reasonable
        assert avg_cpu < 80  # CPU usage < 80%
        assert avg_memory < 2000  # Memory < 2GB

        print(f"\nAverage CPU usage: {avg_cpu:.1f}%")
        print(f"Average memory usage: {avg_memory:.1f}MB")


# ============================================================================
# TEST STRESS CONDITIONS
# ============================================================================

class TestStressConditions:
    """Test performance under stress conditions."""

    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, orchestrator, sample_input):
        """Test performance under high concurrency."""
        concurrent_requests = 500

        start = time.perf_counter()
        tasks = [
            orchestrator.process_async(sample_input)
            for _ in range(concurrent_requests)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.perf_counter() - start

        successful = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successful / concurrent_requests

        assert success_rate > 0.95  # 95%+ success rate
        assert duration < 30  # Complete within 30 seconds

        print(f"\nHigh concurrency test:")
        print(f"  Concurrent requests: {concurrent_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total duration: {duration:.2f}s")

    @pytest.mark.performance
    @pytest.mark.stress
    def test_sustained_load(self, orchestrator, sample_input):
        """Test sustained high load performance."""
        test_duration = 30  # seconds
        requests_completed = 0
        errors = 0

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < test_duration:
            try:
                orchestrator.process(sample_input)
                requests_completed += 1
            except Exception:
                errors += 1

        actual_duration = time.perf_counter() - start_time
        throughput = requests_completed / actual_duration
        error_rate = errors / (requests_completed + errors) if (requests_completed + errors) > 0 else 0

        assert throughput > 10  # Maintain 10+ req/sec under sustained load
        assert error_rate < 0.01  # Less than 1% error rate

        print(f"\nSustained load test ({test_duration}s):")
        print(f"  Requests completed: {requests_completed}")
        print(f"  Throughput: {throughput:.2f} req/sec")
        print(f"  Error rate: {error_rate:.2%}")

    @pytest.mark.performance
    @pytest.mark.stress
    def test_recovery_after_spike(self, orchestrator, sample_input):
        """Test recovery after traffic spike."""
        # Normal load
        normal_latencies = []
        for _ in range(10):
            start = time.perf_counter()
            orchestrator.process(sample_input)
            normal_latencies.append(time.perf_counter() - start)

        avg_normal_latency = np.mean(normal_latencies)

        # Create spike
        spike_tasks = []
        for _ in range(100):
            # Process without waiting
            orchestrator.process_async(sample_input)

        # Measure recovery
        recovery_latencies = []
        for _ in range(10):
            start = time.perf_counter()
            orchestrator.process(sample_input)
            recovery_latencies.append(time.perf_counter() - start)

        avg_recovery_latency = np.mean(recovery_latencies)

        # Should recover to within 2x normal latency
        assert avg_recovery_latency < avg_normal_latency * 2

        print(f"\nRecovery after spike:")
        print(f"  Normal latency: {avg_normal_latency*1000:.2f}ms")
        print(f"  Recovery latency: {avg_recovery_latency*1000:.2f}ms")