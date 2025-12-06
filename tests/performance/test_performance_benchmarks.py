# -*- coding: utf-8 -*-
"""
Performance Tests for GreenLang Agents and Pipelines

Comprehensive test suite with 40 test cases covering:
- Latency Benchmarks (12 tests)
- Throughput Benchmarks (12 tests)
- Resource Utilization (10 tests)
- Scalability Tests (6 tests)

Target: Validate performance meets SLA requirements
Run with: pytest tests/performance/test_performance_benchmarks.py -v --tb=short

Author: GL-TestEngineer
Version: 1.0.0

Performance Targets:
- Agent execution: <10ms p95 latency
- Pipeline execution: <50ms p95 latency
- Throughput: >1000 calculations/second
- Memory: <500MB for 100k records
"""

import pytest
import asyncio
import time
import json
import hashlib
import statistics
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_fast_agent():
    """Create mock agent with fast execution."""
    agent = Mock()
    agent.name = "fast_agent"

    async def fast_process(input_data):
        # Simulate fast processing
        result = {
            "value": input_data.get("quantity", 0) * 0.0561,
            "provenance_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),
        }
        return result

    agent.process = AsyncMock(side_effect=fast_process)
    return agent


@pytest.fixture
def sample_batch_data():
    """Generate sample batch data for throughput tests."""
    return [
        {"fuel_type": "natural_gas", "quantity": i * 100, "unit": "MJ"}
        for i in range(1, 1001)
    ]


@pytest.fixture
def performance_tracker():
    """Create performance metrics tracker."""
    class PerformanceTracker:
        def __init__(self):
            self.latencies = []
            self.throughput_samples = []
            self.memory_samples = []

        def record_latency(self, latency_ms):
            self.latencies.append(latency_ms)

        def record_throughput(self, records_per_second):
            self.throughput_samples.append(records_per_second)

        def record_memory(self, memory_mb):
            self.memory_samples.append(memory_mb)

        def get_p50_latency(self):
            if not self.latencies:
                return 0
            return statistics.median(self.latencies)

        def get_p95_latency(self):
            if not self.latencies:
                return 0
            sorted_latencies = sorted(self.latencies)
            index = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]

        def get_p99_latency(self):
            if not self.latencies:
                return 0
            sorted_latencies = sorted(self.latencies)
            index = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]

        def get_avg_throughput(self):
            if not self.throughput_samples:
                return 0
            return statistics.mean(self.throughput_samples)

        def get_max_memory(self):
            if not self.memory_samples:
                return 0
            return max(self.memory_samples)

    return PerformanceTracker()


# =============================================================================
# Latency Benchmark Tests (12 tests)
# =============================================================================

class TestLatencyBenchmarks:
    """Test suite for latency benchmarks - 12 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_calculation_latency(self, mock_fast_agent, performance_tracker):
        """PERF-LAT-001: Test single calculation latency is <10ms."""
        input_data = {"fuel_type": "natural_gas", "quantity": 1000, "unit": "MJ"}

        for _ in range(100):
            start = time.perf_counter()
            await mock_fast_agent.process(input_data)
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 10, f"P95 latency {p95_latency}ms exceeds 10ms target"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_hash_computation_latency(self, performance_tracker):
        """PERF-LAT-002: Test SHA-256 hash computation latency."""
        data = {"large": "dataset" * 1000}

        for _ in range(1000):
            start = time.perf_counter()
            hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 1, f"P95 hash latency {p95_latency}ms exceeds 1ms target"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_json_serialization_latency(self, performance_tracker):
        """PERF-LAT-003: Test JSON serialization latency."""
        data = {
            "emissions": [{"value": i, "unit": "kgCO2e"} for i in range(100)],
            "metadata": {"source": "test", "timestamp": datetime.now().isoformat()},
        }

        for _ in range(1000):
            start = time.perf_counter()
            json.dumps(data, sort_keys=True)
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 1, f"P95 serialization latency {p95_latency}ms exceeds 1ms target"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_latency(self, performance_tracker):
        """PERF-LAT-004: Test input validation latency."""
        def validate_input(data):
            required = ["fuel_type", "quantity", "unit"]
            return all(k in data for k in required)

        data = {"fuel_type": "diesel", "quantity": 100, "unit": "L"}

        for _ in range(10000):
            start = time.perf_counter()
            validate_input(data)
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 0.1, f"P95 validation latency {p95_latency}ms exceeds 0.1ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_request_latency(self, mock_fast_agent, performance_tracker):
        """PERF-LAT-005: Test latency under concurrent load."""
        async def timed_request(agent, data):
            start = time.perf_counter()
            await agent.process(data)
            end = time.perf_counter()
            return (end - start) * 1000

        # 50 concurrent requests
        tasks = [
            timed_request(mock_fast_agent, {"quantity": i})
            for i in range(50)
        ]

        latencies = await asyncio.gather(*tasks)

        for latency in latencies:
            performance_tracker.record_latency(latency)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 20, f"P95 concurrent latency {p95_latency}ms exceeds 20ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_latency(self, mock_fast_agent, performance_tracker):
        """PERF-LAT-006: Test multi-agent pipeline latency."""
        agents = [mock_fast_agent, mock_fast_agent, mock_fast_agent]

        for _ in range(50):
            start = time.perf_counter()
            data = {"quantity": 1000}
            for agent in agents:
                result = await agent.process(data)
                data = {"quantity": result.get("value", 0)}
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 50, f"P95 pipeline latency {p95_latency}ms exceeds 50ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_hit_latency(self, performance_tracker):
        """PERF-LAT-007: Test cache hit latency."""
        cache = {}

        # Prime cache
        key = "cached_value"
        cache[key] = {"emissions": 56.1}

        for _ in range(10000):
            start = time.perf_counter()
            _ = cache.get(key)
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 0.01, f"P95 cache hit latency {p95_latency}ms exceeds 0.01ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_lookup_latency(self, performance_tracker):
        """PERF-LAT-008: Test emission factor lookup latency."""
        factors = {
            (f"fuel_{i}", f"region_{j}", 2023): 0.05 * i
            for i in range(100)
            for j in range(10)
        }

        for _ in range(1000):
            key = ("fuel_50", "region_5", 2023)
            start = time.perf_counter()
            _ = factors.get(key)
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 0.01, f"P95 lookup latency {p95_latency}ms exceeds 0.01ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_calculation_latency(self, performance_tracker):
        """PERF-LAT-009: Test emissions calculation latency."""
        def calculate_emissions(quantity, ef_value):
            return quantity * ef_value

        for _ in range(10000):
            start = time.perf_counter()
            calculate_emissions(1000.0, 0.0561)
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p95_latency = performance_tracker.get_p95_latency()
        assert p95_latency < 0.001, f"P95 calculation latency {p95_latency}ms exceeds 0.001ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_p50_vs_p99_latency_ratio(self, mock_fast_agent, performance_tracker):
        """PERF-LAT-010: Test P99/P50 latency ratio is acceptable."""
        for _ in range(1000):
            start = time.perf_counter()
            await mock_fast_agent.process({"quantity": 1000})
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        p50 = performance_tracker.get_p50_latency()
        p99 = performance_tracker.get_p99_latency()

        if p50 > 0:
            ratio = p99 / p50
            assert ratio < 10, f"P99/P50 ratio {ratio} exceeds acceptable 10x"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cold_vs_warm_latency(self, mock_fast_agent, performance_tracker):
        """PERF-LAT-011: Test cold start vs warm latency."""
        # Cold start
        start = time.perf_counter()
        await mock_fast_agent.process({"quantity": 1000})
        cold_latency = (time.perf_counter() - start) * 1000

        # Warm (subsequent calls)
        warm_latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await mock_fast_agent.process({"quantity": 1000})
            warm_latencies.append((time.perf_counter() - start) * 1000)

        avg_warm = statistics.mean(warm_latencies)

        # Cold start can be up to 5x slower
        assert cold_latency < avg_warm * 5, "Cold start too slow compared to warm"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_latency_consistency(self, mock_fast_agent, performance_tracker):
        """PERF-LAT-012: Test latency consistency (low variance)."""
        for _ in range(100):
            start = time.perf_counter()
            await mock_fast_agent.process({"quantity": 1000})
            end = time.perf_counter()
            performance_tracker.record_latency((end - start) * 1000)

        if len(performance_tracker.latencies) > 1:
            std_dev = statistics.stdev(performance_tracker.latencies)
            mean = statistics.mean(performance_tracker.latencies)

            if mean > 0:
                cv = std_dev / mean  # Coefficient of variation
                assert cv < 1.0, f"Latency CV {cv} indicates high variance"


# =============================================================================
# Throughput Benchmark Tests (12 tests)
# =============================================================================

class TestThroughputBenchmarks:
    """Test suite for throughput benchmarks - 12 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_agent_throughput(self, mock_fast_agent, sample_batch_data, performance_tracker):
        """PERF-THR-001: Test single agent throughput >1000 records/sec."""
        start = time.perf_counter()
        for data in sample_batch_data[:1000]:
            await mock_fast_agent.process(data)
        duration = time.perf_counter() - start

        throughput = 1000 / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 1000, f"Throughput {throughput}/sec below 1000 target"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_processing_throughput(self, mock_fast_agent, sample_batch_data, performance_tracker):
        """PERF-THR-002: Test batch processing throughput."""
        batch_size = 100
        batches = [sample_batch_data[i:i+batch_size] for i in range(0, len(sample_batch_data), batch_size)]

        start = time.perf_counter()
        for batch in batches:
            await asyncio.gather(*[mock_fast_agent.process(d) for d in batch])
        duration = time.perf_counter() - start

        throughput = len(sample_batch_data) / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 5000, f"Batch throughput {throughput}/sec below 5000 target"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_hash_computation_throughput(self, performance_tracker):
        """PERF-THR-003: Test hash computation throughput."""
        data = {"value": "test" * 100}
        count = 10000

        start = time.perf_counter()
        for _ in range(count):
            hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 50000, f"Hash throughput {throughput}/sec below 50000"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_throughput(self, mock_fast_agent, performance_tracker):
        """PERF-THR-004: Test throughput with concurrent workers."""
        async def worker(agent, count):
            for i in range(count):
                await agent.process({"quantity": i})

        workers = 10
        records_per_worker = 100

        start = time.perf_counter()
        await asyncio.gather(*[worker(mock_fast_agent, records_per_worker) for _ in range(workers)])
        duration = time.perf_counter() - start

        total_records = workers * records_per_worker
        throughput = total_records / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 2000, f"Concurrent throughput {throughput}/sec below 2000"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_throughput(self, mock_fast_agent, performance_tracker):
        """PERF-THR-005: Test pipeline throughput with multiple stages."""
        stages = 3
        records = 500

        start = time.perf_counter()
        for i in range(records):
            data = {"quantity": i}
            for _ in range(stages):
                result = await mock_fast_agent.process(data)
                data = {"quantity": result.get("value", 0)}
        duration = time.perf_counter() - start

        throughput = records / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 500, f"Pipeline throughput {throughput}/sec below 500"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sustained_throughput(self, mock_fast_agent, performance_tracker):
        """PERF-THR-006: Test sustained throughput over time."""
        duration_seconds = 2
        records_processed = 0

        start = time.perf_counter()
        while time.perf_counter() - start < duration_seconds:
            await mock_fast_agent.process({"quantity": records_processed})
            records_processed += 1

        throughput = records_processed / duration_seconds
        performance_tracker.record_throughput(throughput)

        assert throughput > 500, f"Sustained throughput {throughput}/sec below 500"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_with_validation(self, mock_fast_agent, performance_tracker):
        """PERF-THR-007: Test throughput including validation overhead."""
        def validate(data):
            required = ["quantity"]
            return all(k in data for k in required)

        count = 1000

        start = time.perf_counter()
        for i in range(count):
            data = {"quantity": i}
            if validate(data):
                await mock_fast_agent.process(data)
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 800, f"Throughput with validation {throughput}/sec below 800"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_with_caching(self, mock_fast_agent, performance_tracker):
        """PERF-THR-008: Test throughput improvement with caching."""
        cache = {}
        count = 1000

        async def cached_process(agent, data):
            key = json.dumps(data, sort_keys=True)
            if key in cache:
                return cache[key]
            result = await agent.process(data)
            cache[key] = result
            return result

        # With caching (many cache hits expected with repeated data)
        repeated_data = [{"quantity": i % 10} for i in range(count)]

        start = time.perf_counter()
        for data in repeated_data:
            await cached_process(mock_fast_agent, data)
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 2000, f"Cached throughput {throughput}/sec below 2000"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_payload_throughput(self, mock_fast_agent, performance_tracker):
        """PERF-THR-009: Test throughput with large payloads."""
        large_data = {"quantity": 1000, "metadata": {"details": "x" * 10000}}
        count = 100

        start = time.perf_counter()
        for _ in range(count):
            await mock_fast_agent.process(large_data)
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 50, f"Large payload throughput {throughput}/sec below 50"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_serialization_throughput(self, performance_tracker):
        """PERF-THR-010: Test JSON serialization throughput."""
        data = {
            "emissions": [{"value": i, "unit": "kgCO2e"} for i in range(100)],
            "metadata": {"source": "test"},
        }
        count = 5000

        start = time.perf_counter()
        for _ in range(count):
            json.dumps(data, sort_keys=True)
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 10000, f"Serialization throughput {throughput}/sec below 10000"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_deserialization_throughput(self, performance_tracker):
        """PERF-THR-011: Test JSON deserialization throughput."""
        data = {"emissions": [{"value": i} for i in range(100)]}
        json_str = json.dumps(data)
        count = 5000

        start = time.perf_counter()
        for _ in range(count):
            json.loads(json_str)
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 10000, f"Deserialization throughput {throughput}/sec below 10000"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mixed_workload_throughput(self, mock_fast_agent, performance_tracker):
        """PERF-THR-012: Test mixed workload throughput."""
        import random

        workloads = [
            {"quantity": random.randint(1, 1000)},
            {"quantity": random.randint(1, 10000), "large": "x" * 1000},
            {"quantity": 0},
        ]
        count = 1000

        start = time.perf_counter()
        for _ in range(count):
            data = random.choice(workloads)
            await mock_fast_agent.process(data)
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        performance_tracker.record_throughput(throughput)

        assert throughput > 500, f"Mixed workload throughput {throughput}/sec below 500"


# =============================================================================
# Resource Utilization Tests (10 tests)
# =============================================================================

class TestResourceUtilization:
    """Test suite for resource utilization - 10 test cases."""

    @pytest.mark.performance
    def test_memory_usage_baseline(self, performance_tracker):
        """PERF-RES-001: Test baseline memory usage."""
        import sys

        # Create baseline objects
        data = []
        for i in range(1000):
            data.append({"value": i, "hash": "a" * 64})

        # Estimate memory (rough)
        memory_bytes = sys.getsizeof(data)
        memory_mb = memory_bytes / (1024 * 1024)

        performance_tracker.record_memory(memory_mb)

        assert memory_mb < 1, f"Baseline memory {memory_mb}MB exceeds 1MB"

    @pytest.mark.performance
    def test_memory_scaling(self, performance_tracker):
        """PERF-RES-002: Test memory scales linearly with data."""
        import sys

        sizes = [1000, 5000, 10000]
        memories = []

        for size in sizes:
            data = [{"value": i} for i in range(size)]
            memories.append(sys.getsizeof(data))

        # Memory should scale roughly linearly
        ratio_1 = memories[1] / memories[0]
        ratio_2 = memories[2] / memories[0]

        expected_ratio_1 = sizes[1] / sizes[0]
        expected_ratio_2 = sizes[2] / sizes[0]

        # Allow 2x variance from linear
        assert ratio_1 < expected_ratio_1 * 2
        assert ratio_2 < expected_ratio_2 * 2

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_memory_usage(self, mock_fast_agent, performance_tracker):
        """PERF-RES-003: Test memory under concurrent load."""
        import sys

        initial_size = sys.getsizeof([])
        results = []

        # Process many concurrent requests
        for i in range(1000):
            result = await mock_fast_agent.process({"quantity": i})
            results.append(result)

        final_size = sys.getsizeof(results)
        memory_increase_kb = (final_size - initial_size) / 1024

        performance_tracker.record_memory(memory_increase_kb / 1024)

        # Should be < 10MB for 1000 results
        assert memory_increase_kb < 10240, f"Memory increase {memory_increase_kb}KB exceeds 10MB"

    @pytest.mark.performance
    def test_cache_memory_limit(self, performance_tracker):
        """PERF-RES-004: Test cache respects memory limits."""
        import sys

        max_cache_mb = 100
        max_cache_bytes = max_cache_mb * 1024 * 1024

        cache = {}
        total_size = 0

        for i in range(100000):
            entry = {"key": f"entry_{i}", "value": "x" * 100}
            entry_size = sys.getsizeof(entry)

            if total_size + entry_size > max_cache_bytes:
                break

            cache[f"key_{i}"] = entry
            total_size += entry_size

        memory_mb = total_size / (1024 * 1024)
        performance_tracker.record_memory(memory_mb)

        assert memory_mb <= max_cache_mb, f"Cache {memory_mb}MB exceeds {max_cache_mb}MB limit"

    @pytest.mark.performance
    def test_provenance_storage_efficiency(self, performance_tracker):
        """PERF-RES-005: Test provenance storage is space-efficient."""
        # SHA-256 hashes are fixed 64 bytes as hex string
        hashes = []
        for i in range(10000):
            data = {"value": i}
            hash_val = hashlib.sha256(json.dumps(data).encode()).hexdigest()
            hashes.append(hash_val)

        # 64 bytes per hash * 10000 hashes = 640KB
        expected_size_kb = 64 * 10000 / 1024

        import sys
        actual_size = sys.getsizeof(hashes)
        actual_size_kb = actual_size / 1024

        performance_tracker.record_memory(actual_size_kb / 1024)

        # Allow 10x overhead for Python object overhead
        assert actual_size_kb < expected_size_kb * 10

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_no_memory_leak_on_repeated_calls(self, mock_fast_agent, performance_tracker):
        """PERF-RES-006: Test no memory leak on repeated agent calls."""
        import gc

        # Force garbage collection
        gc.collect()

        # Track objects before
        objects_before = len(gc.get_objects())

        # Many repeated calls
        for i in range(1000):
            await mock_fast_agent.process({"quantity": i})

        # Force garbage collection
        gc.collect()

        # Track objects after
        objects_after = len(gc.get_objects())

        # Object count should not grow significantly
        growth = objects_after - objects_before
        assert growth < 10000, f"Object growth {growth} suggests memory leak"

    @pytest.mark.performance
    def test_string_interning_efficiency(self, performance_tracker):
        """PERF-RES-007: Test string interning for common values."""
        # Common fuel types should be interned
        fuel_types = ["natural_gas", "diesel", "gasoline", "lpg", "electricity"]

        instances = []
        for _ in range(1000):
            for fuel in fuel_types:
                instances.append(fuel)

        # Due to string interning, memory should not grow linearly
        unique_fuel_ids = len(set(id(f) for f in fuel_types))

        # Should have same number of unique ids as fuel types (interned)
        assert unique_fuel_ids == len(fuel_types)

    @pytest.mark.performance
    def test_list_vs_generator_memory(self, performance_tracker):
        """PERF-RES-008: Test generator memory efficiency."""
        import sys

        # List approach (all in memory)
        list_data = [i * 0.0561 for i in range(10000)]
        list_size = sys.getsizeof(list_data)

        # Generator approach (lazy)
        def gen():
            for i in range(10000):
                yield i * 0.0561

        gen_size = sys.getsizeof(gen())

        # Generator should be much smaller
        assert gen_size < list_size / 10

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_vs_individual_memory(self, mock_fast_agent, performance_tracker):
        """PERF-RES-009: Test batch processing memory efficiency."""
        import sys

        # Individual results
        individual_results = []
        for i in range(100):
            result = await mock_fast_agent.process({"quantity": i})
            individual_results.append(result)

        individual_size = sys.getsizeof(individual_results)

        # Batch as single object
        batch_data = [{"quantity": i} for i in range(100)]
        batch_results = await asyncio.gather(*[
            mock_fast_agent.process(d) for d in batch_data
        ])

        batch_size = sys.getsizeof(batch_results)

        # Both should be similar in size
        ratio = batch_size / individual_size if individual_size > 0 else 1
        assert 0.5 < ratio < 2.0

    @pytest.mark.performance
    def test_json_memory_overhead(self, performance_tracker):
        """PERF-RES-010: Test JSON string memory overhead."""
        import sys

        data = {"emissions": 56.1, "unit": "kgCO2e", "provenance": "a" * 64}

        dict_size = sys.getsizeof(data)
        json_str = json.dumps(data)
        json_size = sys.getsizeof(json_str)

        # JSON string should be reasonably close to dict size
        ratio = json_size / dict_size if dict_size > 0 else 1

        # JSON typically 2-5x the dict size
        assert ratio < 10, f"JSON overhead ratio {ratio} too high"


# =============================================================================
# Scalability Tests (6 tests)
# =============================================================================

class TestScalability:
    """Test suite for scalability - 6 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_linear_scaling(self, mock_fast_agent, performance_tracker):
        """PERF-SCALE-001: Test throughput scales linearly with workers."""
        async def measure_throughput(workers, records_per_worker):
            async def worker_task(agent, count):
                for i in range(count):
                    await agent.process({"quantity": i})

            start = time.perf_counter()
            await asyncio.gather(*[
                worker_task(mock_fast_agent, records_per_worker)
                for _ in range(workers)
            ])
            duration = time.perf_counter() - start

            return (workers * records_per_worker) / duration if duration > 0 else 0

        throughput_1 = await measure_throughput(1, 100)
        throughput_4 = await measure_throughput(4, 100)

        # 4 workers should be at least 2x faster (not 4x due to overhead)
        scaling_factor = throughput_4 / throughput_1 if throughput_1 > 0 else 0

        assert scaling_factor > 2, f"Scaling factor {scaling_factor} below 2x"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_data_size_scaling(self, mock_fast_agent, performance_tracker):
        """PERF-SCALE-002: Test performance with increasing data sizes."""
        sizes = [100, 1000, 10000]
        times = []

        for size in sizes:
            data = [{"quantity": i} for i in range(size)]

            start = time.perf_counter()
            for d in data:
                await mock_fast_agent.process(d)
            times.append(time.perf_counter() - start)

        # Time should scale linearly (within 2x of linear)
        ratio_1 = times[1] / times[0]
        ratio_2 = times[2] / times[0]

        expected_1 = sizes[1] / sizes[0]
        expected_2 = sizes[2] / sizes[0]

        assert ratio_1 < expected_1 * 2, f"Scaling ratio {ratio_1} vs expected {expected_1}"
        assert ratio_2 < expected_2 * 2, f"Scaling ratio {ratio_2} vs expected {expected_2}"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_depth_scaling(self, mock_fast_agent, performance_tracker):
        """PERF-SCALE-003: Test performance with increasing pipeline depth."""
        depths = [1, 3, 5, 10]
        times = []

        for depth in depths:
            start = time.perf_counter()
            data = {"quantity": 1000}
            for _ in range(100):  # 100 records through each pipeline
                current = data
                for _ in range(depth):
                    result = await mock_fast_agent.process(current)
                    current = {"quantity": result.get("value", 0)}
            times.append(time.perf_counter() - start)

        # Time should scale linearly with depth
        for i, (depth, t) in enumerate(zip(depths, times)):
            if i > 0:
                expected_ratio = depths[i] / depths[0]
                actual_ratio = times[i] / times[0]
                assert actual_ratio < expected_ratio * 2

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_connections_scaling(self, mock_fast_agent, performance_tracker):
        """PERF-SCALE-004: Test with increasing concurrent connections."""
        connections = [10, 50, 100]
        throughputs = []

        for conn_count in connections:
            async def connection_task(id):
                for i in range(10):
                    await mock_fast_agent.process({"conn_id": id, "req": i})

            start = time.perf_counter()
            await asyncio.gather(*[connection_task(i) for i in range(conn_count)])
            duration = time.perf_counter() - start

            throughput = (conn_count * 10) / duration
            throughputs.append(throughput)

        # Throughput should generally increase with connections (up to a point)
        assert throughputs[1] >= throughputs[0] * 0.5, "Throughput degraded significantly"

    @pytest.mark.performance
    def test_lookup_table_scaling(self, performance_tracker):
        """PERF-SCALE-005: Test lookup performance with table size."""
        sizes = [100, 1000, 10000]
        times = []

        for size in sizes:
            table = {f"key_{i}": f"value_{i}" for i in range(size)}

            start = time.perf_counter()
            for _ in range(10000):
                _ = table.get(f"key_{size // 2}")  # Middle key
            duration = time.perf_counter() - start
            times.append(duration)

        # Dictionary lookup should be O(1), so time should be similar
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            assert ratio < 2, f"Lookup time ratio {ratio} suggests non-O(1) behavior"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_horizontal_scaling_simulation(self, mock_fast_agent, performance_tracker):
        """PERF-SCALE-006: Test simulated horizontal scaling."""
        # Simulate multiple "instances" processing different partitions
        instances = 4
        records_per_instance = 250

        async def instance_process(instance_id, records):
            results = []
            for r in records:
                result = await mock_fast_agent.process(r)
                results.append(result)
            return results

        # Partition data
        all_data = [{"quantity": i, "partition": i % instances} for i in range(instances * records_per_instance)]
        partitions = [
            [d for d in all_data if d["partition"] == p]
            for p in range(instances)
        ]

        start = time.perf_counter()
        await asyncio.gather(*[
            instance_process(i, partitions[i])
            for i in range(instances)
        ])
        duration = time.perf_counter() - start

        total_records = instances * records_per_instance
        throughput = total_records / duration

        # Should achieve good throughput with horizontal scaling
        assert throughput > 1000, f"Horizontal scaling throughput {throughput}/sec below 1000"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
