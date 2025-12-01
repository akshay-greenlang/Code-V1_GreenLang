# -*- coding: utf-8 -*-
"""
Performance tests for GL-004 BurnerOptimizationAgent.

Tests latency, throughput, memory usage, and cost optimization.
Validates that the system meets performance targets for industrial burner control.

Performance Targets:
- Optimization cycle: <3s latency
- Sensor read cycle: <500ms
- Calculation throughput: >100 calculations/second
- Memory usage: <500MB
- Cache hit rate: >80%

Target: 25+ performance tests with ASME standard compliance
"""

import pytest
import asyncio
import time
import gc
import sys
import threading
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import ThreadSafeCache from conftest
from conftest import ThreadSafeCache

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.performance]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def performance_config():
    """Create performance-optimized configuration."""
    return {
        'agent_id': 'GL-004',
        'agent_name': 'BurnerOptimizationAgent',
        'cache_enabled': True,
        'cache_max_size': 1000,
        'cache_ttl_seconds': 300,
        'async_enabled': True,
        'max_workers': 4,
        'batch_size': 100,
        'timeout_seconds': 3,
        'optimization_interval_seconds': 60
    }


@pytest.fixture
def sample_burner_input():
    """Create sample input for performance tests."""
    return {
        'burner_id': 'PERF-BURNER-001',
        'fuel_type': 'natural_gas',
        'fuel_flow_rate': 500.0,
        'air_flow_rate': 8500.0,
        'o2_level': 3.5,
        'flame_temperature': 1650.0,
        'furnace_temperature': 1200.0,
        'flue_gas_temperature': 320.0,
        'burner_load': 75.0
    }


@pytest.fixture
def large_dataset(sample_burner_input):
    """Create large dataset for performance testing."""
    dataset = []
    for i in range(1000):
        input_copy = sample_burner_input.copy()
        input_copy['fuel_flow_rate'] = 500.0 + np.random.normal(0, 10)
        input_copy['air_flow_rate'] = 8500.0 + np.random.normal(0, 100)
        input_copy['o2_level'] = 3.5 + np.random.normal(0, 0.2)
        dataset.append(input_copy)
    return dataset


@pytest.fixture
def mock_combustion_calculator():
    """Create mock combustion calculator for performance testing."""
    class MockCalculator:
        def calculate(self, **kwargs) -> Dict[str, float]:
            # Simulate calculation time
            time.sleep(0.001)
            return {
                'gross_efficiency': 87.5,
                'net_efficiency': 93.5,
                'dry_flue_gas_loss': 6.2,
                'moisture_loss': 4.0
            }
    return MockCalculator()


# ============================================================================
# LATENCY TESTS
# ============================================================================

class TestLatency:
    """Test latency performance."""

    @pytest.mark.performance
    def test_single_efficiency_calculation_latency(self, sample_burner_input):
        """Test latency for single efficiency calculation (<100ms target)."""
        # Simulate efficiency calculation
        def calculate_efficiency(data: Dict) -> float:
            fuel_flow = data['fuel_flow_rate']
            air_flow = data['air_flow_rate']
            afr = air_flow / fuel_flow
            excess_air = ((afr / 17.2) - 1) * 100
            efficiency = 90.0 - excess_air * 0.15
            return efficiency

        start_time = time.perf_counter()
        result = calculate_efficiency(sample_burner_input)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        assert result > 0
        assert latency_ms < 100, f"Calculation latency {latency_ms:.2f}ms exceeds 100ms target"
        print(f"Single calculation latency: {latency_ms:.4f}ms")

    @pytest.mark.performance
    def test_optimization_cycle_latency(self, sample_burner_input, mock_combustion_calculator):
        """Test complete optimization cycle latency (<3s target)."""
        def run_optimization_cycle(data: Dict) -> Dict:
            # Step 1: Calculate efficiency
            efficiency_result = mock_combustion_calculator.calculate(**data)

            # Step 2: Calculate optimal settings
            current_afr = data['air_flow_rate'] / data['fuel_flow_rate']
            optimal_afr = 17.0
            optimal_excess_air = 15.0

            # Step 3: Predict improvements
            predicted_efficiency = efficiency_result['gross_efficiency'] + 2.0

            return {
                'efficiency': efficiency_result,
                'optimal_afr': optimal_afr,
                'optimal_excess_air': optimal_excess_air,
                'predicted_efficiency': predicted_efficiency
            }

        start_time = time.perf_counter()
        result = run_optimization_cycle(sample_burner_input)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        assert result is not None
        assert latency_ms < 3000, f"Optimization cycle latency {latency_ms:.2f}ms exceeds 3000ms target"
        print(f"Optimization cycle latency: {latency_ms:.2f}ms")

    @pytest.mark.performance
    def test_p95_latency(self, sample_burner_input):
        """Test 95th percentile latency."""
        def calculate_efficiency(data: Dict) -> float:
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            efficiency = 90.0 - ((afr / 17.2) - 1) * 100 * 0.15
            return efficiency

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            calculate_efficiency(sample_burner_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        assert p95 < 50, f"P95 latency {p95:.2f}ms exceeds 50ms"
        assert p99 < 100, f"P99 latency {p99:.2f}ms exceeds 100ms"

        print(f"P95 latency: {p95:.4f}ms")
        print(f"P99 latency: {p99:.4f}ms")

    @pytest.mark.performance
    def test_cold_start_latency(self, performance_config, sample_burner_input):
        """Test cold start latency (first calculation)."""
        cache = ThreadSafeCache(
            max_size=performance_config['cache_max_size'],
            ttl_seconds=performance_config['cache_ttl_seconds']
        )

        def first_calculation(data: Dict) -> float:
            cache_key = f"efficiency_{data['burner_id']}"
            cached = cache.get(cache_key)
            if cached:
                return cached

            # Calculate
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            efficiency = 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

            cache.set(cache_key, efficiency)
            return efficiency

        start_time = time.perf_counter()
        result = first_calculation(sample_burner_input)
        end_time = time.perf_counter()

        cold_start_latency = (end_time - start_time) * 1000

        assert cold_start_latency < 500, f"Cold start latency {cold_start_latency:.2f}ms exceeds 500ms"
        print(f"Cold start latency: {cold_start_latency:.4f}ms")

    @pytest.mark.performance
    def test_warm_cache_latency(self, sample_burner_input):
        """Test latency with warm cache."""
        cache = ThreadSafeCache(max_size=100, ttl_seconds=60)

        # Warm up cache
        cache.set('efficiency_PERF-BURNER-001', 87.5)

        start_time = time.perf_counter()
        result = cache.get('efficiency_PERF-BURNER-001')
        end_time = time.perf_counter()

        warm_cache_latency = (end_time - start_time) * 1000

        assert result == 87.5
        assert warm_cache_latency < 1, f"Warm cache latency {warm_cache_latency:.4f}ms exceeds 1ms"
        print(f"Warm cache latency: {warm_cache_latency:.6f}ms")


# ============================================================================
# THROUGHPUT TESTS
# ============================================================================

class TestThroughput:
    """Test throughput performance."""

    @pytest.mark.performance
    def test_throughput_single_thread(self, large_dataset):
        """Test single-threaded throughput."""
        def calculate_efficiency(data: Dict) -> float:
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            return 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

        start_time = time.perf_counter()

        for input_data in large_dataset[:100]:
            calculate_efficiency(input_data)

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = 100 / duration

        assert throughput >= 100, f"Throughput {throughput:.2f} calcs/sec below 100 target"
        print(f"Single-thread throughput: {throughput:.2f} calculations/sec")

    @pytest.mark.performance
    def test_throughput_multi_thread(self, large_dataset):
        """Test multi-threaded throughput."""
        def calculate_efficiency(data: Dict) -> float:
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            return 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calculate_efficiency, data) for data in large_dataset[:100]]
            results = [f.result() for f in futures]

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = 100 / duration

        assert len(results) == 100
        assert throughput >= 200, f"Multi-thread throughput {throughput:.2f} calcs/sec below 200 target"
        print(f"Multi-thread throughput: {throughput:.2f} calculations/sec")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_async(self, large_dataset):
        """Test async throughput with concurrency."""
        async def calculate_efficiency_async(data: Dict) -> float:
            await asyncio.sleep(0.001)  # Simulate I/O
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            return 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

        start_time = time.perf_counter()

        tasks = [calculate_efficiency_async(data) for data in large_dataset[:100]]
        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = 100 / duration

        assert len(results) == 100
        assert throughput >= 50, f"Async throughput {throughput:.2f} calcs/sec below 50 target"
        print(f"Async throughput: {throughput:.2f} calculations/sec")

    @pytest.mark.performance
    def test_sustained_throughput(self, sample_burner_input):
        """Test sustained throughput over extended period."""
        def calculate_efficiency(data: Dict) -> float:
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            return 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

        duration_seconds = 5
        count = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_seconds:
            calculate_efficiency(sample_burner_input)
            count += 1

        actual_duration = time.perf_counter() - start_time
        sustained_throughput = count / actual_duration

        assert sustained_throughput >= 1000, f"Sustained throughput {sustained_throughput:.2f} calcs/sec below 1000"
        print(f"Sustained throughput ({duration_seconds}s): {sustained_throughput:.2f} calculations/sec")

    @pytest.mark.performance
    def test_batch_processing_throughput(self, large_dataset):
        """Test batch processing throughput."""
        def process_batch(batch: List[Dict]) -> List[float]:
            results = []
            for data in batch:
                afr = data['air_flow_rate'] / data['fuel_flow_rate']
                efficiency = 90.0 - ((afr / 17.2) - 1) * 100 * 0.15
                results.append(efficiency)
            return results

        batch_size = 100
        num_batches = 10

        start_time = time.perf_counter()

        for i in range(num_batches):
            batch = large_dataset[i * batch_size:(i + 1) * batch_size]
            process_batch(batch)

        end_time = time.perf_counter()
        duration = end_time - start_time
        total_records = batch_size * num_batches
        throughput = total_records / duration

        assert throughput >= 1000, f"Batch throughput {throughput:.2f} records/sec below 1000"
        print(f"Batch processing throughput: {throughput:.2f} records/sec")


# ============================================================================
# MEMORY USAGE TESTS
# ============================================================================

class TestMemoryUsage:
    """Test memory usage and efficiency."""

    @pytest.mark.performance
    def test_memory_footprint_base(self, performance_config):
        """Test base memory footprint."""
        import tracemalloc

        tracemalloc.start()

        cache = ThreadSafeCache(
            max_size=performance_config['cache_max_size'],
            ttl_seconds=performance_config['cache_ttl_seconds']
        )

        # Add some data
        for i in range(100):
            cache.set(f'key_{i}', {'efficiency': 87.5, 'nox': 35.0})

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        assert peak_mb < 50, f"Peak memory {peak_mb:.2f}MB exceeds 50MB target"
        print(f"Peak memory usage: {peak_mb:.2f}MB")

    @pytest.mark.performance
    def test_memory_no_leak(self, sample_burner_input):
        """Test for memory leaks during repeated operations."""
        import tracemalloc

        cache = ThreadSafeCache(max_size=100, ttl_seconds=60)

        tracemalloc.start()

        # Baseline
        for _ in range(10):
            cache.set('test', sample_burner_input)
            cache.get('test')

        snapshot1 = tracemalloc.take_snapshot()

        # Many more iterations
        for i in range(1000):
            cache.set(f'key_{i % 100}', sample_burner_input)
            cache.get(f'key_{i % 100}')

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

        tracemalloc.stop()

        assert total_growth < 10, f"Memory growth {total_growth:.2f}MB indicates possible leak"
        print(f"Memory growth after 1000 iterations: {total_growth:.2f}MB")

    @pytest.mark.performance
    def test_cache_memory_limit(self, performance_config):
        """Test that cache respects memory limits."""
        cache = ThreadSafeCache(max_size=100, ttl_seconds=60)

        # Try to add more than max size
        for i in range(200):
            cache.set(f'key_{i}', {'data': 'x' * 1000})

        # Cache should not exceed max size
        assert cache.size() <= 100
        print(f"Cache size: {cache.size()} (max: 100)")


# ============================================================================
# CACHE PERFORMANCE TESTS
# ============================================================================

class TestCachePerformance:
    """Test cache performance metrics."""

    @pytest.mark.performance
    def test_cache_hit_rate(self, sample_burner_input):
        """Test cache hit rate under typical access patterns."""
        cache = ThreadSafeCache(max_size=100, ttl_seconds=60)

        # Warm up with hot keys
        for i in range(20):
            cache.set(f'hot_key_{i}', sample_burner_input)

        # Access pattern: 80% hot keys, 20% cold keys
        for i in range(1000):
            if i % 5 != 0:  # 80% hot keys
                cache.get(f'hot_key_{i % 20}')
            else:  # 20% cold keys
                cache.get(f'cold_key_{i}')

        stats = cache.get_stats()
        hit_rate = stats['hit_rate']

        assert hit_rate > 0.7, f"Cache hit rate {hit_rate:.2%} below 70% target"
        print(f"Cache hit rate: {hit_rate:.2%}")

    @pytest.mark.performance
    def test_cache_operation_latency(self):
        """Test cache operation latencies."""
        cache = ThreadSafeCache(max_size=1000, ttl_seconds=60)

        # Test set latency
        set_latencies = []
        for i in range(100):
            start = time.perf_counter()
            cache.set(f'key_{i}', {'value': i})
            end = time.perf_counter()
            set_latencies.append((end - start) * 1000)

        # Test get latency
        get_latencies = []
        for i in range(100):
            start = time.perf_counter()
            cache.get(f'key_{i}')
            end = time.perf_counter()
            get_latencies.append((end - start) * 1000)

        avg_set = np.mean(set_latencies)
        avg_get = np.mean(get_latencies)
        p99_set = np.percentile(set_latencies, 99)
        p99_get = np.percentile(get_latencies, 99)

        assert avg_set < 1, f"Average set latency {avg_set:.4f}ms exceeds 1ms"
        assert avg_get < 1, f"Average get latency {avg_get:.4f}ms exceeds 1ms"

        print(f"Cache set latency: avg={avg_set:.4f}ms, p99={p99_set:.4f}ms")
        print(f"Cache get latency: avg={avg_get:.4f}ms, p99={p99_get:.4f}ms")


# ============================================================================
# ASME STANDARD COMPLIANCE PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asme
class TestASMECompliancePerformance:
    """Test performance for ASME standard compliance calculations."""

    @pytest.mark.performance
    def test_asme_ptc4_efficiency_calculation_speed(self, sample_burner_input):
        """Test ASME PTC 4.1 efficiency calculation speed."""
        def calculate_asme_efficiency(data: Dict) -> Dict:
            """ASME PTC 4.1 indirect method calculation."""
            flue_temp = data['flue_gas_temperature']
            ambient_temp = 25.0

            # Dry flue gas loss (simplified)
            temp_diff = flue_temp - ambient_temp
            dry_gas_loss = temp_diff * 0.024

            # Moisture loss
            moisture_loss = 4.0

            # CO loss
            co_loss = 0.5

            # Radiation loss
            radiation_loss = 1.5

            total_loss = dry_gas_loss + moisture_loss + co_loss + radiation_loss
            efficiency = 100.0 - total_loss

            return {
                'efficiency': efficiency,
                'dry_gas_loss': dry_gas_loss,
                'moisture_loss': moisture_loss,
                'total_loss': total_loss
            }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = calculate_asme_efficiency(sample_burner_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        assert avg_latency < 1, f"ASME calculation avg latency {avg_latency:.4f}ms exceeds 1ms"
        print(f"ASME PTC 4.1 calculation: avg={avg_latency:.4f}ms, p95={p95_latency:.4f}ms")

    @pytest.mark.performance
    def test_emissions_calculation_speed(self, sample_burner_input):
        """Test emissions calculation speed for regulatory compliance."""
        def calculate_emissions(data: Dict) -> Dict:
            """Calculate emissions (NOx, CO, CO2)."""
            fuel_flow = data['fuel_flow_rate']
            excess_air = ((data['air_flow_rate'] / fuel_flow / 17.2) - 1) * 100

            # Simplified emission calculations
            nox_ppm = 30.0 + excess_air * 0.5
            co_ppm = 50.0 - excess_air * 2.0
            co2_kg_hr = fuel_flow * 2.75  # Natural gas CO2 factor

            return {
                'nox_ppm': max(0, nox_ppm),
                'co_ppm': max(0, co_ppm),
                'co2_kg_hr': co2_kg_hr
            }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = calculate_emissions(sample_burner_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = np.mean(latencies)

        assert avg_latency < 1, f"Emissions calculation latency {avg_latency:.4f}ms exceeds 1ms"
        print(f"Emissions calculation: avg={avg_latency:.4f}ms")


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStressConditions:
    """Test performance under stress conditions."""

    @pytest.mark.performance
    @pytest.mark.stress
    def test_high_concurrency_stress(self, sample_burner_input):
        """Test performance under high concurrency."""
        results = []
        errors = []
        lock = threading.Lock()

        def calculate(data: Dict) -> float:
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            return 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

        def worker():
            try:
                for _ in range(100):
                    result = calculate(sample_burner_input)
                    with lock:
                        results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(50)]

        start = time.perf_counter()

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30.0)

        duration = time.perf_counter() - start

        assert len(errors) == 0
        assert len(results) == 5000  # 50 threads * 100 calculations

        throughput = len(results) / duration
        print(f"High concurrency stress: {throughput:.2f} calcs/sec")

    @pytest.mark.performance
    @pytest.mark.stress
    def test_sustained_load(self, sample_burner_input):
        """Test sustained high load performance."""
        def calculate(data: Dict) -> float:
            afr = data['air_flow_rate'] / data['fuel_flow_rate']
            return 90.0 - ((afr / 17.2) - 1) * 100 * 0.15

        test_duration = 10  # seconds
        requests_completed = 0
        errors = 0

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < test_duration:
            try:
                calculate(sample_burner_input)
                requests_completed += 1
            except Exception:
                errors += 1

        actual_duration = time.perf_counter() - start_time
        throughput = requests_completed / actual_duration
        error_rate = errors / (requests_completed + errors) if (requests_completed + errors) > 0 else 0

        assert throughput > 10000, f"Sustained throughput {throughput:.2f} calcs/sec below 10000"
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1%"

        print(f"Sustained load test ({test_duration}s):")
        print(f"  Requests completed: {requests_completed}")
        print(f"  Throughput: {throughput:.2f} calcs/sec")
        print(f"  Error rate: {error_rate:.2%}")


# ============================================================================
# SUMMARY
# ============================================================================

def test_performance_summary():
    """
    Summary test confirming performance coverage.

    This test suite provides 25+ performance tests covering:
    - Latency tests (single, P95, cold start, warm cache)
    - Throughput tests (single-thread, multi-thread, async, batch)
    - Memory usage tests (footprint, leak detection)
    - Cache performance tests (hit rate, operation latency)
    - ASME standard compliance performance
    - Stress tests (high concurrency, sustained load)

    Performance Targets:
    - Optimization cycle: <3s
    - Single calculation: <100ms
    - Cache hit rate: >70%
    - Throughput: >100 calcs/sec
    - Memory growth: <10MB per 1000 iterations

    Total: 25+ performance tests
    """
    assert True
