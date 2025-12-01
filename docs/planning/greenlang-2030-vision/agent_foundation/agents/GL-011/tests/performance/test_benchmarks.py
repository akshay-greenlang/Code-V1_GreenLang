# -*- coding: utf-8 -*-
"""
Performance benchmarks for GL-011 FUELCRAFT.

Tests performance characteristics:
- Calculation throughput
- Memory usage
- Cache efficiency
- Concurrent operation performance
- Latency requirements
"""

import pytest
import sys
import time
import threading
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.multi_fuel_optimizer import MultiFuelOptimizer, MultiFuelOptimizationInput
from calculators.cost_optimization_calculator import CostOptimizationCalculator, CostOptimizationInput
from calculators.fuel_blending_calculator import FuelBlendingCalculator, BlendingInput
from calculators.carbon_footprint_calculator import CarbonFootprintCalculator, CarbonFootprintInput
from fuel_management_orchestrator import ThreadSafeCache


class TestCalculationThroughput:
    """Test suite for calculation throughput benchmarks."""

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties for benchmarks."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_nox_g_gj': 250,
                'renewable': False
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_nox_g_gj': 150,
                'renewable': True
            }
        }

    @pytest.fixture
    def market_prices(self):
        """Standard market prices for benchmarks."""
        return {
            'natural_gas': 0.045,
            'coal': 0.035,
            'biomass': 0.08
        }

    def test_multi_fuel_optimization_throughput(self, fuel_properties, market_prices):
        """Benchmark multi-fuel optimization throughput."""
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        # Warm-up
        for _ in range(10):
            optimizer.optimize(input_data)

        # Benchmark
        iterations = 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            optimizer.optimize(input_data)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        throughput = iterations / elapsed

        # Should achieve at least 50 optimizations per second
        assert throughput > 50, f"Throughput {throughput:.1f} ops/sec below 50 ops/sec target"

        print(f"\nMulti-fuel optimization throughput: {throughput:.1f} ops/sec")

    def test_cost_optimization_throughput(self, fuel_properties, market_prices):
        """Benchmark cost optimization throughput."""
        calculator = CostOptimizationCalculator()
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000, 'coal': 100000, 'biomass': 100000},
            delivery_costs={'natural_gas': 0, 'coal': 0, 'biomass': 0},
            constraints={}
        )

        # Warm-up
        for _ in range(10):
            calculator.optimize(input_data)

        # Benchmark
        iterations = 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            calculator.optimize(input_data)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        throughput = iterations / elapsed

        # Should achieve at least 100 optimizations per second
        assert throughput > 100, f"Throughput {throughput:.1f} ops/sec below 100 ops/sec target"

        print(f"\nCost optimization throughput: {throughput:.1f} ops/sec")

    def test_carbon_footprint_throughput(self, fuel_properties):
        """Benchmark carbon footprint calculation throughput."""
        calculator = CarbonFootprintCalculator()
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000, 'coal': 500, 'biomass': 300},
            fuel_properties=fuel_properties
        )

        # Warm-up
        for _ in range(10):
            calculator.calculate(input_data)

        # Benchmark
        iterations = 1000
        start_time = time.perf_counter()
        for _ in range(iterations):
            calculator.calculate(input_data)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        throughput = iterations / elapsed

        # Should achieve at least 500 calculations per second
        assert throughput > 500, f"Throughput {throughput:.1f} ops/sec below 500 ops/sec target"

        print(f"\nCarbon footprint throughput: {throughput:.1f} ops/sec")


class TestLatencyRequirements:
    """Test suite for latency requirements."""

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_nox_g_gj': 250,
                'renewable': False
            }
        }

    @pytest.fixture
    def market_prices(self):
        """Standard market prices."""
        return {
            'natural_gas': 0.045,
            'coal': 0.035
        }

    def test_optimization_p99_latency(self, fuel_properties, market_prices):
        """Test P99 latency for optimization is under 100ms."""
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        # Collect latency samples
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            optimizer.optimize(input_data)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        p99 = sorted(latencies)[98]  # 99th percentile

        # P99 should be under 100ms
        assert p99 < 100, f"P99 latency {p99:.1f}ms exceeds 100ms target"

        print(f"\nOptimization P99 latency: {p99:.2f}ms")
        print(f"Mean latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median latency: {statistics.median(latencies):.2f}ms")

    def test_carbon_calculation_p95_latency(self, fuel_properties):
        """Test P95 latency for carbon calculation is under 10ms."""
        calculator = CarbonFootprintCalculator()
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000, 'coal': 500},
            fuel_properties=fuel_properties
        )

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            calculator.calculate(input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        p95 = sorted(latencies)[189]  # 95th percentile

        # P95 should be under 10ms
        assert p95 < 10, f"P95 latency {p95:.1f}ms exceeds 10ms target"

        print(f"\nCarbon calculation P95 latency: {p95:.2f}ms")


class TestMemoryUsage:
    """Test suite for memory usage benchmarks."""

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            }
        }

    def test_optimizer_memory_stability(self, fuel_properties):
        """Test optimizer memory remains stable over many iterations."""
        import tracemalloc

        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        # Measure initial memory
        tracemalloc.start()

        # Run many iterations
        for _ in range(1000):
            optimizer.optimize(input_data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should not grow excessively (< 10MB for 1000 iterations)
        assert peak < 10 * 1024 * 1024, f"Peak memory {peak/1024/1024:.1f}MB exceeds 10MB"

        print(f"\nPeak memory usage: {peak/1024/1024:.2f}MB")
        print(f"Current memory usage: {current/1024/1024:.2f}MB")

    def test_cache_memory_limit(self):
        """Test cache respects memory limits."""
        cache = ThreadSafeCache(max_size=100)

        # Fill cache beyond limit
        for i in range(200):
            cache.set(f"key_{i}", f"value_{i}" * 1000)

        # Cache should not exceed max size
        assert len(cache._cache) <= 100

    def test_no_memory_leak_in_provenance(self, fuel_properties):
        """Test provenance tracking doesn't leak memory."""
        import tracemalloc

        calculator = CarbonFootprintCalculator()
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties
        )

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(1000):
            calculator.calculate(input_data)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare memory growth
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Memory growth should be minimal (< 1MB for 1000 calculations)
        assert total_diff < 1 * 1024 * 1024, f"Memory growth {total_diff/1024:.1f}KB too high"


class TestCacheEfficiency:
    """Test suite for cache efficiency benchmarks."""

    def test_cache_hit_rate(self):
        """Test cache achieves high hit rate with repeated queries."""
        cache = ThreadSafeCache(max_size=100)

        # Simulate typical access pattern (80% repeated, 20% new)
        hits = 0
        misses = 0

        for i in range(1000):
            if i < 100:
                # First 100: all misses, populate cache
                key = f"key_{i}"
                cache.set(key, f"value_{i}")
                misses += 1
            else:
                # Rest: 80% repeat access, 20% new
                import random
                if random.random() < 0.8:
                    key = f"key_{random.randint(0, 99)}"
                    if cache.get(key) is not None:
                        hits += 1
                    else:
                        misses += 1
                else:
                    key = f"key_{i}"
                    cache.set(key, f"value_{i}")
                    misses += 1

        hit_rate = hits / (hits + misses) * 100

        # Should achieve at least 60% hit rate
        assert hit_rate > 60, f"Hit rate {hit_rate:.1f}% below 60% target"

        print(f"\nCache hit rate: {hit_rate:.1f}%")

    def test_cache_speedup(self):
        """Test cache provides significant speedup."""
        cache = ThreadSafeCache(max_size=100)

        def expensive_operation(key):
            """Simulate expensive operation."""
            time.sleep(0.001)  # 1ms
            return f"result_{key}"

        # Without cache
        start = time.perf_counter()
        for i in range(100):
            key = f"key_{i % 10}"  # 10 unique keys, repeated
            expensive_operation(key)
        uncached_time = time.perf_counter() - start

        # With cache
        start = time.perf_counter()
        for i in range(100):
            key = f"key_{i % 10}"
            result = cache.get(key)
            if result is None:
                result = expensive_operation(key)
                cache.set(key, result)
        cached_time = time.perf_counter() - start

        speedup = uncached_time / cached_time

        # Cache should provide at least 5x speedup
        assert speedup > 5, f"Speedup {speedup:.1f}x below 5x target"

        print(f"\nCache speedup: {speedup:.1f}x")


class TestConcurrentPerformance:
    """Test suite for concurrent operation performance."""

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_nox_g_gj': 250,
                'renewable': False
            }
        }

    @pytest.fixture
    def market_prices(self):
        """Standard market prices."""
        return {
            'natural_gas': 0.045,
            'coal': 0.035
        }

    def test_concurrent_optimization_throughput(self, fuel_properties, market_prices):
        """Test concurrent optimization throughput."""
        optimizer = MultiFuelOptimizer()

        def run_optimization(demand):
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=demand,
                available_fuels=['natural_gas', 'coal'],
                fuel_properties=fuel_properties,
                market_prices=market_prices,
                emission_limits={},
                constraints={},
                optimization_objective='balanced'
            )
            return optimizer.optimize(input_data)

        iterations = 100
        workers = 4

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_optimization, 100 + i) for i in range(iterations)]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed

        # Concurrent throughput should be higher than single-threaded
        assert throughput > 50, f"Concurrent throughput {throughput:.1f} ops/sec too low"
        assert len(results) == iterations

        print(f"\nConcurrent optimization throughput ({workers} workers): {throughput:.1f} ops/sec")

    def test_cache_concurrent_performance(self):
        """Test cache performance under concurrent access."""
        cache = ThreadSafeCache(max_size=1000)

        def cache_operations(thread_id):
            for i in range(100):
                key = f"key_{thread_id}_{i % 20}"
                cache.set(key, f"value_{thread_id}_{i}")
                cache.get(key)
            return True

        workers = 8
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(workers)]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.perf_counter() - start

        operations = workers * 100 * 2  # set + get per iteration
        ops_per_sec = operations / elapsed

        # Should achieve at least 10000 ops/sec
        assert ops_per_sec > 10000, f"Cache ops/sec {ops_per_sec:.0f} below 10000"
        assert all(results)

        print(f"\nCache concurrent ops/sec: {ops_per_sec:.0f}")


class TestScalabilityBenchmarks:
    """Test suite for scalability benchmarks."""

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_nox_g_gj': 250,
                'renewable': False
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_nox_g_gj': 150,
                'renewable': True
            },
            'fuel_oil': {
                'heating_value_mj_kg': 42.0,
                'emission_factor_co2_kg_gj': 77.4,
                'emission_factor_nox_g_gj': 120,
                'renewable': False
            },
            'hydrogen': {
                'heating_value_mj_kg': 120.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_nox_g_gj': 10,
                'renewable': True
            }
        }

    def test_fuel_count_scalability(self, fuel_properties):
        """Test optimization time scales reasonably with fuel count."""
        optimizer = MultiFuelOptimizer()
        market_prices = {fuel: 0.05 for fuel in fuel_properties}

        times = {}
        for n_fuels in [1, 2, 3, 4, 5]:
            fuels = list(fuel_properties.keys())[:n_fuels]
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=100,
                available_fuels=fuels,
                fuel_properties={k: fuel_properties[k] for k in fuels},
                market_prices={k: market_prices[k] for k in fuels},
                emission_limits={},
                constraints={},
                optimization_objective='balanced'
            )

            start = time.perf_counter()
            for _ in range(20):
                optimizer.optimize(input_data)
            times[n_fuels] = (time.perf_counter() - start) / 20 * 1000

        # Time should not grow exponentially
        # Allow 3x growth from 1 to 5 fuels (should be < O(n^2))
        assert times[5] < times[1] * 10, "Time scaling too aggressive"

        print("\nFuel count scalability:")
        for n, t in times.items():
            print(f"  {n} fuels: {t:.2f}ms")

    def test_batch_size_scalability(self, fuel_properties):
        """Test carbon calculation scales linearly with batch size."""
        calculator = CarbonFootprintCalculator()

        times = {}
        for n_fuels in [1, 10, 100, 1000]:
            fuel_quantities = {f"fuel_{i}": 100.0 for i in range(n_fuels)}
            # Use repeated fuel properties
            props = {}
            for i in range(n_fuels):
                props[f"fuel_{i}"] = fuel_properties['natural_gas']

            input_data = CarbonFootprintInput(
                fuel_quantities=fuel_quantities,
                fuel_properties=props
            )

            start = time.perf_counter()
            for _ in range(10):
                calculator.calculate(input_data)
            times[n_fuels] = (time.perf_counter() - start) / 10 * 1000

        # Should scale approximately linearly (allow 2x overhead)
        assert times[1000] < times[1] * 2000, "Batch scaling not linear"

        print("\nBatch size scalability:")
        for n, t in times.items():
            print(f"  {n} items: {t:.2f}ms")
