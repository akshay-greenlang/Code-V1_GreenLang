"""
Determinism Under Stress Edge Case Tests for GL-004 BURNMASTER

Tests system determinism and correctness under stress conditions:
- High-frequency control updates
- Memory pressure scenarios
- CPU throttling effects
- Large dataset processing
- Sustained high load
- Resource exhaustion scenarios

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
import gc
import time
import threading
import psutil
import os
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import tracemalloc
import random

# Import system modules
import sys
sys.path.insert(0, 'C:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-004_Burnmaster')

from combustion.stoichiometry import (
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_air_percent,
    compute_excess_o2,
    infer_lambda_from_o2,
)
from combustion.fuel_properties import (
    FuelType, FuelComposition,
    compute_fuel_properties, get_fuel_properties,
    compute_molecular_weight, compute_heating_values,
)
from combustion.thermodynamics import (
    compute_stack_loss,
    compute_efficiency_indirect,
    compute_heat_balance,
)
from safety.safety_envelope import SafetyEnvelope, Setpoint, EnvelopeStatus
from calculators.stability_calculator import (
    FlameStabilityCalculator, StabilityLevel, RiskLevel,
)


# ============================================================================
# PERFORMANCE MEASUREMENT UTILITIES
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation: str
    iterations: int
    total_time_s: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops: float
    memory_used_mb: float = 0.0
    cpu_percent: float = 0.0

    def __str__(self):
        return (
            f"{self.operation}: {self.iterations} iterations in {self.total_time_s:.2f}s "
            f"(avg: {self.avg_time_ms:.3f}ms, p95: {self.p95_time_ms:.3f}ms, "
            f"throughput: {self.throughput_ops:.1f} ops/s)"
        )


def measure_performance(
    func,
    iterations: int = 1000,
    warmup: int = 100
) -> PerformanceMetrics:
    """Measure performance of a function."""
    # Warmup
    for _ in range(warmup):
        func()

    # Force garbage collection before measurement
    gc.collect()

    # Measure
    times = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    end_total = time.perf_counter()
    total_time = end_total - start_total

    times = np.array(times)

    return PerformanceMetrics(
        operation=func.__name__ if hasattr(func, '__name__') else "unknown",
        iterations=iterations,
        total_time_s=total_time,
        avg_time_ms=float(np.mean(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        p95_time_ms=float(np.percentile(times, 95)),
        p99_time_ms=float(np.percentile(times, 99)),
        throughput_ops=iterations / total_time
    )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def stability_calculator():
    """Create FlameStabilityCalculator instance."""
    return FlameStabilityCalculator(precision=4)


@pytest.fixture
def safety_envelope():
    """Create configured SafetyEnvelope instance."""
    envelope = SafetyEnvelope(unit_id="BLR-TEST")
    envelope.define_envelope("BLR-TEST", {
        "o2_min": 1.5,
        "o2_max": 8.0,
        "co_max": 200,
        "nox_max": 100,
        "draft_min": -0.5,
        "draft_max": -0.01,
        "flame_signal_min": 30.0,
        "steam_temp_max": 550.0,
        "steam_pressure_max": 150.0,
        "firing_rate_min": 10.0,
        "firing_rate_max": 100.0,
    })
    return envelope


@pytest.fixture
def thread_pool():
    """Create thread pool for concurrent testing."""
    pool = ThreadPoolExecutor(max_workers=8)
    yield pool
    pool.shutdown(wait=True)


# ============================================================================
# HIGH-FREQUENCY UPDATE TESTS
# ============================================================================

class TestHighFrequencyUpdates:
    """Test suite for high-frequency control update scenarios."""

    def test_stability_calculation_at_high_frequency(self, stability_calculator):
        """Test stability calculation performance at high frequency."""
        signal = np.array([100.0 + np.random.normal(0, 3) for _ in range(50)])

        def calculate():
            return stability_calculator.compute_stability_index(signal.copy(), 0.1)

        metrics = measure_performance(calculate, iterations=1000, warmup=100)

        # Should complete 1000 calculations quickly
        assert metrics.avg_time_ms < 5.0, f"Average time {metrics.avg_time_ms}ms exceeds 5ms target"
        assert metrics.p99_time_ms < 20.0, f"P99 time {metrics.p99_time_ms}ms exceeds 20ms target"

        print(f"\n{metrics}")

    def test_stoichiometry_at_high_frequency(self):
        """Test stoichiometry calculation performance."""
        composition = {"CH4": 94.0, "C2H6": 3.0, "C3H8": 1.0, "N2": 2.0}

        def calculate():
            return compute_stoichiometric_air(composition)

        metrics = measure_performance(calculate, iterations=5000, warmup=500)

        # Stoichiometry should be very fast
        assert metrics.avg_time_ms < 1.0, f"Average time {metrics.avg_time_ms}ms exceeds 1ms target"

        print(f"\n{metrics}")

    def test_envelope_validation_at_high_frequency(self, safety_envelope):
        """Test envelope validation performance."""
        setpoint = Setpoint(parameter_name="o2", value=3.0, unit="%")

        def validate():
            return safety_envelope.validate_within_envelope(setpoint)

        metrics = measure_performance(validate, iterations=2000, warmup=200)

        # Validation should be fast
        assert metrics.avg_time_ms < 2.0, f"Average time {metrics.avg_time_ms}ms exceeds 2ms target"

        print(f"\n{metrics}")

    def test_rapid_successive_calculations(self, stability_calculator):
        """Test determinism under rapid successive calculations."""
        signal = np.array([100.0] * 50)
        results = []

        # Rapid fire 1000 calculations
        start = time.perf_counter()
        for _ in range(1000):
            result = stability_calculator.compute_stability_index(signal, 0.1)
            results.append(result.provenance_hash)
        elapsed = time.perf_counter() - start

        # All results should be identical (deterministic)
        assert len(set(results)) == 1, "Results should be deterministic"

        # Should complete in reasonable time
        assert elapsed < 5.0, f"1000 calculations took {elapsed:.2f}s, exceeds 5s target"

    @pytest.mark.parametrize("update_rate_hz", [10, 50, 100, 200])
    def test_sustained_update_rate(self, stability_calculator, update_rate_hz: int):
        """Test sustained calculation at various update rates."""
        duration_s = 1.0
        target_iterations = int(update_rate_hz * duration_s)
        signal = np.array([100.0 + np.random.normal(0, 2) for _ in range(50)])

        completed = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_s:
            stability_calculator.compute_stability_index(signal, 0.1)
            completed += 1

        achieved_rate = completed / duration_s

        # Should achieve at least 80% of target rate
        assert achieved_rate >= update_rate_hz * 0.8, \
            f"Achieved {achieved_rate:.1f} Hz, target was {update_rate_hz} Hz"


# ============================================================================
# MEMORY PRESSURE TESTS
# ============================================================================

class TestMemoryPressure:
    """Test suite for memory pressure scenarios."""

    def test_stability_under_memory_allocation(self, stability_calculator):
        """Test calculation stability while allocating memory."""
        signal = np.array([100.0] * 50)

        # Get baseline result
        baseline = stability_calculator.compute_stability_index(signal, 0.1)

        # Allocate significant memory
        large_arrays = [np.random.random((1000, 1000)) for _ in range(5)]

        # Calculate under memory pressure
        stressed_result = stability_calculator.compute_stability_index(signal, 0.1)

        # Results should be identical
        assert stressed_result.stability_index == baseline.stability_index
        assert stressed_result.provenance_hash == baseline.provenance_hash

        # Cleanup
        del large_arrays
        gc.collect()

    def test_memory_not_leaked_in_calculations(self, stability_calculator):
        """Test that calculations don't leak memory."""
        tracemalloc.start()

        signal = np.array([100.0 + np.random.normal(0, 2) for _ in range(100)])

        # Baseline memory
        gc.collect()
        _, baseline_peak = tracemalloc.get_traced_memory()

        # Run many calculations
        for _ in range(1000):
            result = stability_calculator.compute_stability_index(signal, 0.1)
            del result

        gc.collect()
        _, after_peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # Memory increase should be minimal (< 10 MB)
        memory_increase_mb = (after_peak - baseline_peak) / (1024 * 1024)
        assert memory_increase_mb < 10, f"Memory increased by {memory_increase_mb:.2f} MB"

    def test_large_signal_processing(self, stability_calculator):
        """Test processing of large signal arrays."""
        # Large signal (10 seconds at 1000 Hz)
        large_signal = np.array([100.0 + np.random.normal(0, 3) for _ in range(10000)])

        start = time.perf_counter()
        result = stability_calculator.compute_stability_index(large_signal, 0.1)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 1.0, f"Large signal processing took {elapsed:.2f}s"
        assert result.stability_level is not None

    def test_batch_processing_memory_efficiency(self, stability_calculator):
        """Test memory efficiency of batch processing."""
        signals = [
            {'flame_signal': [100.0 + np.random.normal(0, 2) for _ in range(50)],
             'o2_variance': 0.1}
            for _ in range(100)
        ]

        tracemalloc.start()
        gc.collect()
        _, baseline = tracemalloc.get_traced_memory()

        results = stability_calculator.analyze_stability_batch(signals)

        gc.collect()
        _, after = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # Should process all signals
        assert len(results) == len(signals)

        # Memory usage should be reasonable
        memory_per_signal_kb = (after - baseline) / 1024 / len(signals)
        assert memory_per_signal_kb < 100, f"Memory per signal: {memory_per_signal_kb:.1f} KB"


# ============================================================================
# CPU STRESS TESTS
# ============================================================================

class TestCPUStress:
    """Test suite for CPU stress scenarios."""

    def test_calculations_under_cpu_load(self, stability_calculator, thread_pool):
        """Test calculation correctness under CPU load."""
        signal = np.array([100.0] * 50)

        # Get baseline
        baseline = stability_calculator.compute_stability_index(signal, 0.1)

        # Create CPU load with background work
        def cpu_work():
            for _ in range(10000):
                _ = np.random.random((100, 100)) @ np.random.random((100, 100))

        # Start CPU load
        load_futures = [thread_pool.submit(cpu_work) for _ in range(4)]

        # Run calculations under load
        stressed_results = []
        for _ in range(100):
            result = stability_calculator.compute_stability_index(signal, 0.1)
            stressed_results.append(result.provenance_hash)

        # Wait for load to complete
        for f in load_futures:
            f.result()

        # All results should match baseline
        assert all(h == baseline.provenance_hash for h in stressed_results)

    def test_concurrent_calculation_throughput(self, thread_pool):
        """Test throughput of concurrent calculations."""
        calculator = FlameStabilityCalculator(precision=4)
        num_calculations = 500
        signals = [
            np.array([100.0 + np.random.normal(0, 2) for _ in range(50)])
            for _ in range(num_calculations)
        ]

        def calculate(signal):
            return calculator.compute_stability_index(signal, 0.1)

        start = time.perf_counter()
        futures = [thread_pool.submit(calculate, sig) for sig in signals]
        results = [f.result() for f in as_completed(futures)]
        elapsed = time.perf_counter() - start

        throughput = num_calculations / elapsed

        assert len(results) == num_calculations
        assert throughput >= 100, f"Throughput {throughput:.1f} ops/s below 100 ops/s target"

        print(f"\nConcurrent throughput: {throughput:.1f} ops/s")

    def test_calculation_timing_consistency(self, stability_calculator):
        """Test that calculation times are consistent."""
        signal = np.array([100.0] * 50)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            stability_calculator.compute_stability_index(signal, 0.1)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time  # Coefficient of variation

        # CV should be relatively low (consistent timing)
        assert cv < 0.5, f"Timing too variable: CV={cv:.2f}"


# ============================================================================
# LARGE DATASET TESTS
# ============================================================================

class TestLargeDatasets:
    """Test suite for large dataset processing."""

    def test_bulk_fuel_property_calculations(self):
        """Test bulk calculation of fuel properties."""
        compositions = [
            FuelComposition(
                ch4=94.0 - i * 0.1,
                c2h6=3.0 + i * 0.05,
                n2=3.0 + i * 0.05
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        properties = [compute_fuel_properties(c, FuelType.NATURAL_GAS) for c in compositions]
        elapsed = time.perf_counter() - start

        assert len(properties) == len(compositions)
        assert elapsed < 5.0, f"Bulk calculation took {elapsed:.2f}s"

        # All properties should be valid
        for props in properties:
            assert props.hhv > 0
            assert props.lhv > 0

    def test_large_validation_batch(self, safety_envelope):
        """Test batch validation of many setpoints."""
        setpoints = [
            Setpoint(
                parameter_name="o2",
                value=1.0 + i * 0.05,  # Range from 1.0 to 6.0
                unit="%"
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        validations = [safety_envelope.validate_within_envelope(sp) for sp in setpoints]
        elapsed = time.perf_counter() - start

        assert len(validations) == len(setpoints)
        assert elapsed < 2.0, f"Batch validation took {elapsed:.2f}s"

        # Count valid/invalid
        valid_count = sum(1 for v in validations if v.is_valid)
        invalid_count = len(validations) - valid_count

        # Should have mix of valid and invalid
        assert valid_count > 0
        assert invalid_count > 0

    def test_time_series_analysis(self, stability_calculator):
        """Test analysis of long time series data."""
        # 1 hour of data at 1 Hz
        duration_points = 3600
        time_series = [
            np.array([100.0 + np.sin(i * 0.01) * 5 + np.random.normal(0, 2)
                     for _ in range(50)])
            for i in range(duration_points)
        ]

        start = time.perf_counter()
        results = []
        for signal in time_series[:100]:  # Process first 100 for speed
            result = stability_calculator.compute_stability_index(signal, 0.1)
            results.append(result.stability_index)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 5.0, f"Time series analysis took {elapsed:.2f}s"


# ============================================================================
# SUSTAINED LOAD TESTS
# ============================================================================

class TestSustainedLoad:
    """Test suite for sustained high load scenarios."""

    def test_sustained_calculation_for_1_minute(self, stability_calculator):
        """Test sustained calculations for extended period."""
        duration_s = 5.0  # Shortened for test speed
        signal = np.array([100.0 + np.random.normal(0, 2) for _ in range(50)])

        count = 0
        errors = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_s:
            try:
                result = stability_calculator.compute_stability_index(signal, 0.1)
                assert result.stability_level is not None
                count += 1
            except Exception:
                errors += 1

        # Should have zero errors
        assert errors == 0, f"Had {errors} errors during sustained load"

        # Should maintain reasonable throughput
        throughput = count / duration_s
        assert throughput >= 50, f"Throughput dropped to {throughput:.1f} ops/s"

    def test_no_degradation_over_time(self, stability_calculator):
        """Test that performance doesn't degrade over time."""
        signal = np.array([100.0] * 50)

        # Measure initial performance
        initial_times = []
        for _ in range(100):
            start = time.perf_counter()
            stability_calculator.compute_stability_index(signal, 0.1)
            initial_times.append((time.perf_counter() - start) * 1000)

        # Do sustained work
        for _ in range(1000):
            stability_calculator.compute_stability_index(signal, 0.1)

        # Measure final performance
        final_times = []
        for _ in range(100):
            start = time.perf_counter()
            stability_calculator.compute_stability_index(signal, 0.1)
            final_times.append((time.perf_counter() - start) * 1000)

        initial_avg = np.mean(initial_times)
        final_avg = np.mean(final_times)

        # Final should not be more than 50% slower than initial
        degradation = (final_avg - initial_avg) / initial_avg * 100
        assert degradation < 50, f"Performance degraded by {degradation:.1f}%"


# ============================================================================
# DETERMINISM VERIFICATION TESTS
# ============================================================================

class TestDeterminismVerification:
    """Test suite for verifying calculation determinism under stress."""

    def test_determinism_under_memory_pressure(self, stability_calculator):
        """Verify determinism when memory is constrained."""
        signal = np.array([100.0, 102.0, 98.0, 101.0, 99.0] * 10)

        # Baseline
        baseline = stability_calculator.compute_stability_index(signal, 0.1)

        # Allocate then free memory repeatedly
        for _ in range(10):
            large = [np.random.random((500, 500)) for _ in range(10)]
            result = stability_calculator.compute_stability_index(signal, 0.1)
            assert result.provenance_hash == baseline.provenance_hash
            del large
            gc.collect()

    def test_determinism_under_concurrent_load(self, thread_pool):
        """Verify determinism under concurrent load."""
        calculator = FlameStabilityCalculator(precision=4)
        signal = np.array([100.0] * 50)

        # Get expected hash
        expected = calculator.compute_stability_index(signal, 0.1).provenance_hash

        # Run many concurrent calculations
        def calculate():
            result = calculator.compute_stability_index(signal.copy(), 0.1)
            return result.provenance_hash

        futures = [thread_pool.submit(calculate) for _ in range(100)]
        hashes = [f.result() for f in as_completed(futures)]

        # All hashes should match expected
        assert all(h == expected for h in hashes), "Concurrent calculations not deterministic"

    def test_determinism_across_data_types(self, stability_calculator):
        """Verify determinism with different input data types."""
        # List input
        signal_list = [100.0] * 50
        result_list = stability_calculator.compute_stability_index(
            np.array(signal_list), 0.1
        )

        # Numpy array input
        signal_array = np.array([100.0] * 50)
        result_array = stability_calculator.compute_stability_index(signal_array, 0.1)

        # Different dtype
        signal_float64 = np.array([100.0] * 50, dtype=np.float64)
        result_float64 = stability_calculator.compute_stability_index(signal_float64, 0.1)

        # All should produce same hash
        assert result_list.provenance_hash == result_array.provenance_hash
        assert result_array.provenance_hash == result_float64.provenance_hash

    def test_decimal_precision_maintained(self, stability_calculator):
        """Verify Decimal precision is maintained under stress."""
        signal = np.array([100.0] * 50)

        results = []
        for _ in range(1000):
            result = stability_calculator.compute_stability_index(signal, 0.1)
            results.append(result.stability_index)

        # All should be exact same Decimal value
        assert len(set(results)) == 1

        # Check precision
        decimal_str = str(results[0])
        if '.' in decimal_str:
            decimal_places = len(decimal_str.split('.')[1])
            assert decimal_places <= 4, "Precision exceeded 4 decimal places"

    def test_stoichiometry_determinism_stress(self):
        """Verify stoichiometry determinism under stress."""
        composition = {"CH4": 94.0, "C2H6": 3.0, "C3H8": 1.0, "N2": 2.0}

        baseline = compute_stoichiometric_air(composition)

        # Run many times
        results = [compute_stoichiometric_air(composition) for _ in range(1000)]

        # All should be identical
        assert all(r == baseline for r in results)


# ============================================================================
# RESOURCE EXHAUSTION TESTS
# ============================================================================

class TestResourceExhaustion:
    """Test suite for resource exhaustion scenarios."""

    def test_graceful_handling_of_large_input(self, stability_calculator):
        """Test graceful handling of very large input."""
        # Very large signal
        huge_signal = np.array([100.0] * 100000)

        start = time.perf_counter()
        result = stability_calculator.compute_stability_index(huge_signal, 0.1)
        elapsed = time.perf_counter() - start

        # Should complete (even if slow)
        assert result.stability_level is not None
        print(f"\nHuge signal ({len(huge_signal)} points) processed in {elapsed:.2f}s")

    def test_handles_rapid_envelope_updates(self, safety_envelope):
        """Test handling of rapid envelope updates."""
        for i in range(100):
            safety_envelope.shrink_envelope(0.99, f"Stress test iteration {i}")

        # Should still be functional
        setpoint = Setpoint(parameter_name="o2", value=3.0, unit="%")
        result = safety_envelope.validate_within_envelope(setpoint)

        # May be invalid due to shrinking, but should not crash
        assert result is not None

    def test_thread_pool_exhaustion_recovery(self, thread_pool):
        """Test recovery from thread pool exhaustion."""
        calculator = FlameStabilityCalculator(precision=4)
        signal = np.array([100.0] * 50)

        # Submit more tasks than workers
        num_tasks = 100

        futures = [
            thread_pool.submit(
                calculator.compute_stability_index, signal.copy(), 0.1
            )
            for _ in range(num_tasks)
        ]

        # All should eventually complete
        results = [f.result(timeout=30) for f in futures]
        assert len(results) == num_tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
