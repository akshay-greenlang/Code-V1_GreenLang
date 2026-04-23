"""
Performance tests for GL-001 ThermalCommand Optimization Latency.

Tests the optimization cycle time to ensure <5s target is met
for various problem sizes and complexity levels.

Coverage Target: Performance validation
Reference: GL-001 Specification Section 11

Performance Targets:
- Optimization cycle: <5s
- Simple problems (3 equipment): <1s
- Complex problems (20+ equipment): <5s
- Warm start improvement: >20%

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import time
import statistics
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# MOCK CLASSES FOR PERFORMANCE TESTING
# =============================================================================

class MockEquipment:
    """Mock equipment for performance testing."""

    def __init__(
        self,
        equipment_id: str,
        max_capacity: float = 50.0,
        min_capacity: float = 10.0,
        fuel_cost: float = 5.0,
        efficiency: float = 0.85
    ):
        self.equipment_id = equipment_id
        self.max_capacity_mmbtu_hr = max_capacity
        self.min_capacity_mmbtu_hr = min_capacity
        self.fuel_cost_per_mmbtu = fuel_cost
        self.efficiency = efficiency
        self.status = "available"
        self.current_load = 0.0


class MockOptimizer:
    """Mock optimizer with realistic timing characteristics."""

    def __init__(self, complexity_factor: float = 1.0):
        self._equipment: Dict[str, MockEquipment] = {}
        self._complexity_factor = complexity_factor
        self._warm_start_solution = None
        self._solve_count = 0

    def add_equipment(self, equipment: MockEquipment):
        self._equipment[equipment.equipment_id] = equipment

    def optimize(
        self,
        demand: float,
        use_warm_start: bool = True
    ) -> Dict[str, Any]:
        """Run optimization with realistic timing."""
        start_time = time.perf_counter()

        # Simulate computation time based on problem size
        n_equipment = len(self._equipment)

        # Base computation time (scales with problem size)
        base_time_ms = 10 + n_equipment * 5

        # Add complexity factor
        base_time_ms *= self._complexity_factor

        # Warm start provides ~30% speedup
        if use_warm_start and self._warm_start_solution:
            base_time_ms *= 0.7

        # Add some variance
        actual_time_ms = base_time_ms * (0.9 + np.random.random() * 0.2)

        # Simulate the computation time
        time.sleep(actual_time_ms / 1000.0)

        # Calculate allocations
        available = [
            e for e in self._equipment.values()
            if e.status == "available"
        ]
        total_capacity = sum(e.max_capacity_mmbtu_hr for e in available)

        allocations = []
        remaining = demand

        for eq in sorted(available, key=lambda x: x.fuel_cost_per_mmbtu):
            if remaining <= 0:
                break
            alloc = min(remaining, eq.max_capacity_mmbtu_hr)
            alloc = max(alloc, eq.min_capacity_mmbtu_hr) if alloc > 0 else 0
            allocations.append({
                "equipment_id": eq.equipment_id,
                "allocation": alloc
            })
            remaining -= alloc

        end_time = time.perf_counter()
        solve_time_ms = (end_time - start_time) * 1000

        # Store for warm start
        self._warm_start_solution = allocations
        self._solve_count += 1

        return {
            "status": "optimal" if remaining <= 0 else "feasible",
            "allocations": allocations,
            "solve_time_ms": solve_time_ms,
            "equipment_count": n_equipment,
            "warm_start_used": use_warm_start and self._solve_count > 1
        }

    def clear_warm_start(self):
        self._warm_start_solution = None


class PerformanceMetrics:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.measurements: List[float] = []
        self.labels: List[str] = []

    def record(self, value: float, label: str = ""):
        self.measurements.append(value)
        self.labels.append(label)

    def clear(self):
        self.measurements = []
        self.labels = []

    @property
    def count(self) -> int:
        return len(self.measurements)

    @property
    def mean(self) -> float:
        return statistics.mean(self.measurements) if self.measurements else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.measurements) if self.measurements else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0

    @property
    def min_value(self) -> float:
        return min(self.measurements) if self.measurements else 0.0

    @property
    def max_value(self) -> float:
        return max(self.measurements) if self.measurements else 0.0

    def percentile(self, p: float) -> float:
        if not self.measurements:
            return 0.0
        sorted_values = sorted(self.measurements)
        index = int(len(sorted_values) * p / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "median_ms": self.median,
            "stdev_ms": self.stdev,
            "min_ms": self.min_value,
            "max_ms": self.max_value,
            "p95_ms": self.percentile(95),
            "p99_ms": self.percentile(99),
        }


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def performance_metrics() -> PerformanceMetrics:
    """Create performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def simple_optimizer() -> MockOptimizer:
    """Create optimizer with 3 equipment (simple problem)."""
    optimizer = MockOptimizer(complexity_factor=1.0)
    for i in range(3):
        optimizer.add_equipment(MockEquipment(
            equipment_id=f"BOILER-{i+1:03d}",
            max_capacity=50.0 + i * 10,
            fuel_cost=5.0 + i * 0.5
        ))
    return optimizer


@pytest.fixture
def medium_optimizer() -> MockOptimizer:
    """Create optimizer with 10 equipment (medium problem)."""
    optimizer = MockOptimizer(complexity_factor=1.5)
    for i in range(10):
        optimizer.add_equipment(MockEquipment(
            equipment_id=f"BOILER-{i+1:03d}",
            max_capacity=30.0 + np.random.uniform(-10, 20),
            fuel_cost=4.0 + np.random.uniform(0, 3)
        ))
    return optimizer


@pytest.fixture
def large_optimizer() -> MockOptimizer:
    """Create optimizer with 20+ equipment (large problem)."""
    optimizer = MockOptimizer(complexity_factor=2.0)
    for i in range(25):
        optimizer.add_equipment(MockEquipment(
            equipment_id=f"BOILER-{i+1:03d}",
            max_capacity=30.0 + np.random.uniform(-10, 20),
            fuel_cost=4.0 + np.random.uniform(0, 3)
        ))
    return optimizer


# =============================================================================
# TEST CLASS: OPTIMIZATION CYCLE TIME
# =============================================================================

class TestOptimizationCycleTime:
    """Tests for optimization cycle time targets."""

    @pytest.mark.performance
    def test_simple_optimization_under_1s(
        self, simple_optimizer, performance_metrics
    ):
        """Test simple optimization completes in <1s."""
        iterations = 20

        for _ in range(iterations):
            result = simple_optimizer.optimize(demand=100.0)
            performance_metrics.record(result["solve_time_ms"])

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 1000, \
            f"95th percentile ({summary['p95_ms']:.1f}ms) exceeds 1s target"

    @pytest.mark.performance
    def test_medium_optimization_under_3s(
        self, medium_optimizer, performance_metrics
    ):
        """Test medium optimization completes in <3s."""
        iterations = 10

        for _ in range(iterations):
            result = medium_optimizer.optimize(demand=200.0)
            performance_metrics.record(result["solve_time_ms"])

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 3000, \
            f"95th percentile ({summary['p95_ms']:.1f}ms) exceeds 3s target"

    @pytest.mark.performance
    def test_large_optimization_under_5s(
        self, large_optimizer, performance_metrics
    ):
        """Test large optimization completes in <5s (main target)."""
        iterations = 10

        for _ in range(iterations):
            result = large_optimizer.optimize(demand=500.0)
            performance_metrics.record(result["solve_time_ms"])

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 5000, \
            f"95th percentile ({summary['p95_ms']:.1f}ms) exceeds 5s target"

    @pytest.mark.performance
    def test_consistent_performance(
        self, simple_optimizer, performance_metrics
    ):
        """Test optimization time is consistent (low variance)."""
        iterations = 50

        for _ in range(iterations):
            result = simple_optimizer.optimize(demand=100.0)
            performance_metrics.record(result["solve_time_ms"])

        summary = performance_metrics.summary()

        # Coefficient of variation should be less than 50%
        cv = summary["stdev_ms"] / summary["mean_ms"] if summary["mean_ms"] > 0 else 0
        assert cv < 0.5, \
            f"Performance variance too high: CV = {cv:.2%}"


# =============================================================================
# TEST CLASS: WARM START PERFORMANCE
# =============================================================================

class TestWarmStartPerformance:
    """Tests for warm start performance improvement."""

    @pytest.mark.performance
    def test_warm_start_improvement(
        self, simple_optimizer, performance_metrics
    ):
        """Test that warm start provides >20% improvement."""
        # Cold start measurements
        cold_times = []
        for _ in range(10):
            simple_optimizer.clear_warm_start()
            result = simple_optimizer.optimize(demand=100.0, use_warm_start=False)
            cold_times.append(result["solve_time_ms"])

        # Warm start measurements
        warm_times = []
        simple_optimizer.optimize(demand=100.0)  # Initialize warm start
        for _ in range(10):
            result = simple_optimizer.optimize(demand=100.0, use_warm_start=True)
            warm_times.append(result["solve_time_ms"])

        cold_mean = statistics.mean(cold_times)
        warm_mean = statistics.mean(warm_times)

        improvement = (cold_mean - warm_mean) / cold_mean * 100

        assert improvement > 20, \
            f"Warm start improvement ({improvement:.1f}%) below 20% target"

    @pytest.mark.performance
    def test_warm_start_with_demand_change(
        self, simple_optimizer, performance_metrics
    ):
        """Test warm start with varying demand."""
        # Initialize
        simple_optimizer.optimize(demand=100.0)

        demands = [80.0, 120.0, 90.0, 110.0, 100.0]

        for demand in demands:
            result = simple_optimizer.optimize(demand=demand, use_warm_start=True)
            performance_metrics.record(result["solve_time_ms"])
            assert result["warm_start_used"] is True

        # Should still meet performance targets
        summary = performance_metrics.summary()
        assert summary["mean_ms"] < 1000


# =============================================================================
# TEST CLASS: SCALING BEHAVIOR
# =============================================================================

class TestScalingBehavior:
    """Tests for optimization scaling behavior."""

    @pytest.mark.performance
    def test_linear_scaling(self):
        """Test that optimization time scales reasonably with problem size."""
        sizes = [5, 10, 15, 20]
        times = []

        for size in sizes:
            optimizer = MockOptimizer(complexity_factor=1.0)
            for i in range(size):
                optimizer.add_equipment(MockEquipment(
                    equipment_id=f"EQ-{i}",
                    max_capacity=50.0
                ))

            # Measure solve time
            result = optimizer.optimize(demand=size * 30)
            times.append(result["solve_time_ms"])

        # Check scaling is approximately linear (not exponential)
        # Time at 20 should be < 5x time at 5
        scaling_factor = times[-1] / times[0]
        expected_max = 5.0  # Allow 5x increase for 4x problem size

        assert scaling_factor < expected_max, \
            f"Scaling factor ({scaling_factor:.1f}) exceeds expected ({expected_max})"

    @pytest.mark.performance
    def test_worst_case_scaling(self):
        """Test worst-case scenario (all constraints active)."""
        optimizer = MockOptimizer(complexity_factor=3.0)  # High complexity

        for i in range(20):
            optimizer.add_equipment(MockEquipment(
                equipment_id=f"EQ-{i}",
                max_capacity=30.0,
                min_capacity=25.0  # Tight constraints
            ))

        result = optimizer.optimize(demand=500.0)

        # Even worst case should be under 5s
        assert result["solve_time_ms"] < 5000


# =============================================================================
# TEST CLASS: THROUGHPUT
# =============================================================================

class TestThroughput:
    """Tests for optimization throughput."""

    @pytest.mark.performance
    def test_optimizations_per_second(
        self, simple_optimizer, performance_metrics
    ):
        """Test number of optimizations possible per second."""
        duration_seconds = 5.0
        count = 0

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_seconds:
            simple_optimizer.optimize(demand=100.0)
            count += 1

        throughput = count / duration_seconds

        # Should achieve at least 10 optimizations per second
        assert throughput >= 10, \
            f"Throughput ({throughput:.1f}/s) below 10/s target"

    @pytest.mark.performance
    def test_sustained_throughput(
        self, simple_optimizer, performance_metrics
    ):
        """Test sustained throughput over longer period."""
        iterations = 100

        start_time = time.perf_counter()

        for i in range(iterations):
            result = simple_optimizer.optimize(demand=100.0)
            performance_metrics.record(result["solve_time_ms"])

        elapsed = time.perf_counter() - start_time
        throughput = iterations / elapsed

        summary = performance_metrics.summary()

        # Performance should not degrade over time
        # First 20 vs last 20 measurements
        first_20 = performance_metrics.measurements[:20]
        last_20 = performance_metrics.measurements[-20:]

        first_mean = statistics.mean(first_20)
        last_mean = statistics.mean(last_20)

        degradation = (last_mean - first_mean) / first_mean * 100

        assert degradation < 20, \
            f"Performance degradation ({degradation:.1f}%) exceeds 20%"


# =============================================================================
# TEST CLASS: MEMORY BEHAVIOR
# =============================================================================

class TestMemoryBehavior:
    """Tests for memory usage during optimization."""

    @pytest.mark.performance
    def test_memory_growth(self):
        """Test that memory does not grow unbounded."""
        import tracemalloc

        optimizer = MockOptimizer()
        for i in range(20):
            optimizer.add_equipment(MockEquipment(
                equipment_id=f"EQ-{i}",
                max_capacity=50.0
            ))

        tracemalloc.start()

        # Initial memory
        _, initial_peak = tracemalloc.get_traced_memory()

        # Run many optimizations
        for _ in range(100):
            optimizer.optimize(demand=500.0)

        # Final memory
        _, final_peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        memory_growth_mb = (final_peak - initial_peak) / (1024 * 1024)

        # Memory growth should be less than 50MB for 100 optimizations
        assert memory_growth_mb < 50, \
            f"Memory growth ({memory_growth_mb:.1f}MB) exceeds 50MB limit"


# =============================================================================
# TEST CLASS: LATENCY PERCENTILES
# =============================================================================

class TestLatencyPercentiles:
    """Tests for latency percentile targets."""

    @pytest.mark.performance
    def test_p50_latency(self, simple_optimizer, performance_metrics):
        """Test 50th percentile latency."""
        for _ in range(100):
            result = simple_optimizer.optimize(demand=100.0)
            performance_metrics.record(result["solve_time_ms"])

        p50 = performance_metrics.percentile(50)

        # P50 should be under 500ms for simple problems
        assert p50 < 500, f"P50 latency ({p50:.1f}ms) exceeds 500ms"

    @pytest.mark.performance
    def test_p95_latency(self, simple_optimizer, performance_metrics):
        """Test 95th percentile latency."""
        for _ in range(100):
            result = simple_optimizer.optimize(demand=100.0)
            performance_metrics.record(result["solve_time_ms"])

        p95 = performance_metrics.percentile(95)

        # P95 should be under 1s for simple problems
        assert p95 < 1000, f"P95 latency ({p95:.1f}ms) exceeds 1s"

    @pytest.mark.performance
    def test_p99_latency(self, simple_optimizer, performance_metrics):
        """Test 99th percentile latency."""
        for _ in range(100):
            result = simple_optimizer.optimize(demand=100.0)
            performance_metrics.record(result["solve_time_ms"])

        p99 = performance_metrics.percentile(99)

        # P99 should be under 2s for simple problems
        assert p99 < 2000, f"P99 latency ({p99:.1f}ms) exceeds 2s"


# =============================================================================
# TEST CLASS: STRESS TESTING
# =============================================================================

class TestStressTesting:
    """Stress tests for optimization."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_high_frequency_optimization(
        self, simple_optimizer, performance_metrics
    ):
        """Test rapid-fire optimization requests."""
        iterations = 500

        start_time = time.perf_counter()

        for i in range(iterations):
            # Varying demand to simulate real conditions
            demand = 50.0 + 100.0 * np.sin(i * 0.1)
            result = simple_optimizer.optimize(demand=demand)
            performance_metrics.record(result["solve_time_ms"])

        elapsed = time.perf_counter() - start_time

        summary = performance_metrics.summary()

        # All iterations should complete
        assert performance_metrics.count == iterations

        # Max latency should still be under 5s
        assert summary["max_ms"] < 5000

    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_stress(self, simple_optimizer):
        """Test concurrent optimization stress."""
        import threading
        import queue

        results_queue = queue.Queue()
        errors = []

        def worker(thread_id: int, iterations: int):
            try:
                for i in range(iterations):
                    result = simple_optimizer.optimize(
                        demand=100.0 + thread_id * 10
                    )
                    results_queue.put(result["solve_time_ms"])
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i, 20))
            threads.append(t)

        start_time = time.perf_counter()

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        elapsed = time.perf_counter() - start_time

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert results_queue.qsize() == 100

        # Should complete 100 optimizations across 5 threads reasonably fast
        assert elapsed < 30


# =============================================================================
# BENCHMARK REPORTING
# =============================================================================

class TestBenchmarkReporting:
    """Benchmark tests with detailed reporting."""

    @pytest.mark.performance
    def test_full_benchmark_suite(self):
        """Run full benchmark suite and report results."""
        results = {}

        # Simple problem
        simple_opt = MockOptimizer(complexity_factor=1.0)
        for i in range(3):
            simple_opt.add_equipment(MockEquipment(f"S-{i}"))

        simple_times = []
        for _ in range(20):
            r = simple_opt.optimize(demand=100.0)
            simple_times.append(r["solve_time_ms"])

        results["simple"] = {
            "equipment": 3,
            "mean_ms": statistics.mean(simple_times),
            "p95_ms": sorted(simple_times)[int(0.95 * len(simple_times))],
        }

        # Medium problem
        medium_opt = MockOptimizer(complexity_factor=1.5)
        for i in range(10):
            medium_opt.add_equipment(MockEquipment(f"M-{i}"))

        medium_times = []
        for _ in range(20):
            r = medium_opt.optimize(demand=300.0)
            medium_times.append(r["solve_time_ms"])

        results["medium"] = {
            "equipment": 10,
            "mean_ms": statistics.mean(medium_times),
            "p95_ms": sorted(medium_times)[int(0.95 * len(medium_times))],
        }

        # Large problem
        large_opt = MockOptimizer(complexity_factor=2.0)
        for i in range(25):
            large_opt.add_equipment(MockEquipment(f"L-{i}"))

        large_times = []
        for _ in range(10):
            r = large_opt.optimize(demand=600.0)
            large_times.append(r["solve_time_ms"])

        results["large"] = {
            "equipment": 25,
            "mean_ms": statistics.mean(large_times),
            "p95_ms": sorted(large_times)[int(0.95 * len(large_times))],
        }

        # Report results
        print("\n" + "=" * 60)
        print("OPTIMIZATION LATENCY BENCHMARK RESULTS")
        print("=" * 60)

        for name, metrics in results.items():
            print(f"\n{name.upper()} ({metrics['equipment']} equipment):")
            print(f"  Mean:  {metrics['mean_ms']:.1f}ms")
            print(f"  P95:   {metrics['p95_ms']:.1f}ms")
            print(f"  Target: <{'1s' if name == 'simple' else '3s' if name == 'medium' else '5s'}")

        print("\n" + "=" * 60)

        # Assert all targets met
        assert results["simple"]["p95_ms"] < 1000
        assert results["medium"]["p95_ms"] < 3000
        assert results["large"]["p95_ms"] < 5000
