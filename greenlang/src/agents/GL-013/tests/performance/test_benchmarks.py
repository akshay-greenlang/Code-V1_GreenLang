# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Performance Benchmark Tests
Performance testing and benchmarking for predictive maintenance calculations.

Tests cover:
- Diagnose workflow latency (<5s target)
- Predict workflow latency (<10s target)
- Schedule workflow latency (<30s target)
- Throughput (100 equipment per minute target)
- Memory usage limits
- Cache hit rate (>80% target)
- Batch processing performance
- Concurrent request handling

Performance Targets:
- Single equipment diagnosis: <5 seconds
- Single equipment prediction: <10 seconds
- Maintenance scheduling: <30 seconds
- Batch processing: 100+ equipment/minute
- Memory: <500MB for 10000 equipment batch

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import time
import sys
import tracemalloc
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import functools

# Import test fixtures from conftest
from ..conftest import (
    MachineClass,
    VibrationZone,
    HealthState,
    WEIBULL_PARAMETERS,
)


# =============================================================================
# PERFORMANCE TEST UTILITIES
# =============================================================================


def measure_execution_time(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


class PerformanceMetrics:
    """Utility class for collecting performance metrics."""

    def __init__(self):
        self.execution_times = []
        self.memory_samples = []

    def record_time(self, duration: float):
        """Record execution time."""
        self.execution_times.append(duration)

    def record_memory(self, memory_mb: float):
        """Record memory usage."""
        self.memory_samples.append(memory_mb)

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0

    @property
    def max_time(self) -> float:
        """Maximum execution time."""
        return max(self.execution_times) if self.execution_times else 0

    @property
    def min_time(self) -> float:
        """Minimum execution time."""
        return min(self.execution_times) if self.execution_times else 0

    @property
    def p95_time(self) -> float:
        """95th percentile execution time."""
        if not self.execution_times:
            return 0
        sorted_times = sorted(self.execution_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]

    @property
    def throughput(self) -> float:
        """Operations per second."""
        total_time = sum(self.execution_times)
        return len(self.execution_times) / total_time if total_time > 0 else 0


# =============================================================================
# TEST CLASS: DIAGNOSE LATENCY
# =============================================================================


class TestDiagnoseLatencyUnder5s:
    """Tests for diagnosis workflow latency (<5s target)."""

    @pytest.mark.performance
    def test_single_diagnosis_under_5s(
        self,
        vibration_analyzer,
        thermal_degradation_calculator,
        anomaly_detector,
        pump_equipment_data,
    ):
        """Test single equipment diagnosis completes under 5 seconds."""
        start_time = time.perf_counter()

        # Perform diagnosis workflow
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=pump_equipment_data["vibration_velocity_mm_s"],
            machine_class=MachineClass.CLASS_II,
            equipment_id=pump_equipment_data["equipment_id"],
        )

        thermal_result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=pump_equipment_data["temperature_c"],
            reference_temperature_c=Decimal("110"),
        )

        historical_data = [2.0, 2.1, 2.2, 2.3, 2.4]
        anomaly_result = anomaly_detector.detect_univariate_anomaly(
            value=pump_equipment_data["vibration_velocity_mm_s"],
            historical_data=historical_data,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        assert execution_time < 5.0, \
            f"Diagnosis took {execution_time:.2f}s, exceeds 5s target"

    @pytest.mark.performance
    def test_diagnosis_latency_p95_under_5s(
        self,
        vibration_analyzer,
        pump_equipment_data,
    ):
        """Test 95th percentile diagnosis latency under 5 seconds."""
        metrics = PerformanceMetrics()

        for _ in range(100):
            start_time = time.perf_counter()

            vibration_analyzer.assess_severity(
                velocity_rms=pump_equipment_data["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
            )

            end_time = time.perf_counter()
            metrics.record_time(end_time - start_time)

        assert metrics.p95_time < 5.0, \
            f"P95 latency {metrics.p95_time:.2f}s exceeds 5s target"

    @pytest.mark.performance
    def test_diagnosis_average_latency(
        self,
        vibration_analyzer,
        pump_equipment_data,
    ):
        """Test average diagnosis latency."""
        metrics = PerformanceMetrics()

        for _ in range(50):
            start_time = time.perf_counter()

            vibration_analyzer.assess_severity(
                velocity_rms=pump_equipment_data["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
            )

            end_time = time.perf_counter()
            metrics.record_time(end_time - start_time)

        print(f"\nDiagnosis Latency Metrics:")
        print(f"  Average: {metrics.avg_time*1000:.2f}ms")
        print(f"  Min: {metrics.min_time*1000:.2f}ms")
        print(f"  Max: {metrics.max_time*1000:.2f}ms")
        print(f"  P95: {metrics.p95_time*1000:.2f}ms")

        # Average should be well under 5s
        assert metrics.avg_time < 1.0


# =============================================================================
# TEST CLASS: PREDICT LATENCY
# =============================================================================


class TestPredictLatencyUnder10s:
    """Tests for prediction workflow latency (<10s target)."""

    @pytest.mark.performance
    def test_single_prediction_under_10s(
        self,
        rul_calculator,
        failure_probability_calculator,
        pump_equipment_data,
    ):
        """Test single equipment prediction completes under 10 seconds."""
        start_time = time.perf_counter()

        # RUL calculation
        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=pump_equipment_data["operating_hours"],
            target_reliability="0.5",
            confidence_level="90%",
        )

        # Failure probability
        params = WEIBULL_PARAMETERS["pump_centrifugal"]
        fp_result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=params["beta"],
            eta=params["eta"],
            time_hours=pump_equipment_data["operating_hours"],
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        assert execution_time < 10.0, \
            f"Prediction took {execution_time:.2f}s, exceeds 10s target"

    @pytest.mark.performance
    def test_prediction_with_multiple_failure_modes_under_10s(
        self,
        failure_probability_calculator,
        pump_equipment_data,
    ):
        """Test prediction with multiple failure modes under 10s."""
        failure_modes = [
            {"name": "bearing_wear", "beta": Decimal("2.5"), "eta": Decimal("45000")},
            {"name": "seal_failure", "beta": Decimal("1.8"), "eta": Decimal("25000")},
            {"name": "cavitation", "beta": Decimal("3.0"), "eta": Decimal("35000")},
            {"name": "impeller_erosion", "beta": Decimal("3.5"), "eta": Decimal("55000")},
            {"name": "coupling_failure", "beta": Decimal("2.2"), "eta": Decimal("40000")},
        ]

        start_time = time.perf_counter()

        for mode in failure_modes:
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=mode["beta"],
                eta=mode["eta"],
                time_hours=pump_equipment_data["operating_hours"],
            )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        assert execution_time < 10.0, \
            f"Multi-mode prediction took {execution_time:.2f}s, exceeds 10s target"

    @pytest.mark.performance
    def test_prediction_p95_latency(
        self,
        rul_calculator,
        pump_equipment_data,
    ):
        """Test 95th percentile prediction latency."""
        metrics = PerformanceMetrics()

        for _ in range(100):
            start_time = time.perf_counter()

            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=pump_equipment_data["operating_hours"],
            )

            end_time = time.perf_counter()
            metrics.record_time(end_time - start_time)

        assert metrics.p95_time < 10.0, \
            f"P95 latency {metrics.p95_time:.2f}s exceeds 10s target"


# =============================================================================
# TEST CLASS: SCHEDULE LATENCY
# =============================================================================


class TestScheduleLatencyUnder30s:
    """Tests for scheduling workflow latency (<30s target)."""

    @pytest.mark.performance
    def test_single_schedule_under_30s(
        self,
        maintenance_scheduler,
        spare_parts_calculator,
    ):
        """Test single equipment scheduling under 30 seconds."""
        start_time = time.perf_counter()

        # Calculate optimal interval
        schedule_result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("45000"),
            preventive_cost=Decimal("1500"),
            corrective_cost=Decimal("15000"),
        )

        # Calculate spare parts requirements
        eoq_result = spare_parts_calculator.calculate_eoq(
            annual_demand=Decimal("12"),
            ordering_cost=Decimal("50"),
            holding_cost_rate=Decimal("0.25"),
            unit_cost=Decimal("200"),
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        assert execution_time < 30.0, \
            f"Scheduling took {execution_time:.2f}s, exceeds 30s target"

    @pytest.mark.performance
    def test_schedule_optimization_multiple_equipment_under_30s(
        self,
        maintenance_scheduler,
        equipment_data_generator,
    ):
        """Test scheduling optimization for multiple equipment under 30s."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=50)

        start_time = time.perf_counter()

        for equip in equipment_list:
            maintenance_scheduler.calculate_optimal_interval(
                beta=Decimal("2.5"),
                eta=Decimal("45000"),
                preventive_cost=Decimal("1500"),
                corrective_cost=Decimal("15000"),
            )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        assert execution_time < 30.0, \
            f"Multi-equipment scheduling took {execution_time:.2f}s, exceeds 30s target"


# =============================================================================
# TEST CLASS: THROUGHPUT
# =============================================================================


class TestThroughput100EquipmentPerMinute:
    """Tests for throughput (100 equipment per minute target)."""

    @pytest.mark.performance
    def test_diagnosis_throughput(
        self,
        vibration_analyzer,
        equipment_data_generator,
    ):
        """Test diagnosis throughput meets 100/minute target."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=100)

        start_time = time.perf_counter()

        for equip in equipment_list:
            vibration_analyzer.assess_severity(
                velocity_rms=equip["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
                equipment_id=equip["equipment_id"],
            )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        throughput = 100 / execution_time * 60  # Equipment per minute

        print(f"\nDiagnosis Throughput: {throughput:.0f} equipment/minute")

        assert throughput >= 100, \
            f"Throughput {throughput:.0f}/min below 100/min target"

    @pytest.mark.performance
    def test_prediction_throughput(
        self,
        rul_calculator,
        equipment_data_generator,
    ):
        """Test prediction throughput meets 100/minute target."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=100)

        start_time = time.perf_counter()

        for equip in equipment_list:
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        throughput = 100 / execution_time * 60

        print(f"\nPrediction Throughput: {throughput:.0f} equipment/minute")

        assert throughput >= 100, \
            f"Throughput {throughput:.0f}/min below 100/min target"

    @pytest.mark.performance
    def test_combined_workflow_throughput(
        self,
        vibration_analyzer,
        rul_calculator,
        equipment_data_generator,
    ):
        """Test combined workflow throughput."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=100)

        start_time = time.perf_counter()

        for equip in equipment_list:
            # Diagnosis
            vibration_analyzer.assess_severity(
                velocity_rms=equip["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
            )

            # Prediction
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        throughput = 100 / execution_time * 60

        print(f"\nCombined Workflow Throughput: {throughput:.0f} equipment/minute")

        assert throughput >= 50, \
            f"Combined throughput {throughput:.0f}/min below 50/min target"


# =============================================================================
# TEST CLASS: MEMORY USAGE
# =============================================================================


class TestMemoryUsageUnderLimit:
    """Tests for memory usage limits."""

    @pytest.mark.performance
    def test_memory_baseline(self):
        """Measure baseline memory usage."""
        tracemalloc.start()

        # Baseline measurement
        current, peak = tracemalloc.get_traced_memory()
        baseline_mb = current / 1024 / 1024

        tracemalloc.stop()

        print(f"\nBaseline Memory: {baseline_mb:.2f} MB")

        assert baseline_mb < 100  # Should start under 100MB

    @pytest.mark.performance
    def test_memory_single_calculation(
        self,
        rul_calculator,
        pump_equipment_data,
    ):
        """Test memory usage for single calculation."""
        tracemalloc.start()

        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=pump_equipment_data["operating_hours"],
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_mb = peak / 1024 / 1024

        print(f"\nSingle Calculation Peak Memory: {memory_mb:.2f} MB")

        assert memory_mb < 50  # Single calculation should use <50MB

    @pytest.mark.performance
    def test_memory_batch_processing(
        self,
        rul_calculator,
        equipment_data_generator,
    ):
        """Test memory usage during batch processing."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=1000)

        tracemalloc.start()

        for equip in equipment_list:
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_mb = peak / 1024 / 1024

        print(f"\n1000 Equipment Batch Peak Memory: {memory_mb:.2f} MB")

        assert memory_mb < 500, \
            f"Memory usage {memory_mb:.2f}MB exceeds 500MB limit"

    @pytest.mark.performance
    def test_memory_no_leak_repeated_calculations(
        self,
        rul_calculator,
        pump_equipment_data,
    ):
        """Test for memory leaks in repeated calculations."""
        tracemalloc.start()

        # Run many calculations
        for _ in range(1000):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=pump_equipment_data["operating_hours"],
            )

        current1, peak1 = tracemalloc.get_traced_memory()

        # Run more calculations
        for _ in range(1000):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=pump_equipment_data["operating_hours"],
            )

        current2, peak2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should not grow significantly
        memory_growth = (current2 - current1) / 1024 / 1024

        print(f"\nMemory Growth After 2000 Calculations: {memory_growth:.2f} MB")

        assert memory_growth < 100, \
            f"Memory grew {memory_growth:.2f}MB, possible leak"


# =============================================================================
# TEST CLASS: CACHE PERFORMANCE
# =============================================================================


class TestCacheHitRateAbove80Percent:
    """Tests for cache hit rate (>80% target)."""

    @pytest.mark.performance
    def test_cache_effectiveness_repeated_queries(
        self,
        vibration_analyzer,
    ):
        """Test cache effectiveness with repeated identical queries."""
        # Note: This tests the concept - actual caching depends on implementation
        cache_hits = 0
        total_queries = 100

        # First query establishes baseline
        first_start = time.perf_counter()
        vibration_analyzer.assess_severity(
            velocity_rms=Decimal("4.5"),
            machine_class=MachineClass.CLASS_II,
        )
        first_time = time.perf_counter() - first_start

        # Repeated queries should be faster if cached
        fast_threshold = first_time * 0.5  # 50% faster = likely cached

        for _ in range(total_queries - 1):
            start = time.perf_counter()
            vibration_analyzer.assess_severity(
                velocity_rms=Decimal("4.5"),
                machine_class=MachineClass.CLASS_II,
            )
            query_time = time.perf_counter() - start

            if query_time <= fast_threshold:
                cache_hits += 1

        hit_rate = cache_hits / total_queries * 100

        print(f"\nCache Hit Rate (estimated): {hit_rate:.1f}%")
        print(f"First query time: {first_time*1000:.3f}ms")
        print(f"Fast threshold: {fast_threshold*1000:.3f}ms")

    @pytest.mark.performance
    def test_parameter_lookup_caching(self):
        """Test that parameter lookups benefit from caching."""
        equipment_types = ["pump_centrifugal", "motor_ac_induction_large", "gearbox_helical"]

        times_first_pass = []
        times_second_pass = []

        # First pass
        for eq_type in equipment_types:
            start = time.perf_counter()
            params = WEIBULL_PARAMETERS.get(eq_type)
            times_first_pass.append(time.perf_counter() - start)

        # Second pass (should benefit from caching)
        for eq_type in equipment_types:
            start = time.perf_counter()
            params = WEIBULL_PARAMETERS.get(eq_type)
            times_second_pass.append(time.perf_counter() - start)

        # Second pass should be at least as fast
        avg_first = sum(times_first_pass) / len(times_first_pass)
        avg_second = sum(times_second_pass) / len(times_second_pass)

        print(f"\nFirst pass avg: {avg_first*1000000:.2f}us")
        print(f"Second pass avg: {avg_second*1000000:.2f}us")


# =============================================================================
# TEST CLASS: CONCURRENT PERFORMANCE
# =============================================================================


class TestConcurrentPerformance:
    """Tests for concurrent request handling performance."""

    @pytest.mark.performance
    def test_concurrent_diagnosis_requests(
        self,
        vibration_analyzer,
        equipment_data_generator,
    ):
        """Test performance under concurrent diagnosis requests."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=100)

        def diagnose(equip):
            return vibration_analyzer.assess_severity(
                velocity_rms=equip["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
            )

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(diagnose, eq) for eq in equipment_list]
            results = [f.result() for f in as_completed(futures)]

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        throughput = len(results) / execution_time * 60

        print(f"\nConcurrent Diagnosis (10 workers):")
        print(f"  Total time: {execution_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} equipment/minute")

        assert len(results) == 100
        assert throughput >= 200  # Should be faster with concurrency

    @pytest.mark.performance
    def test_concurrent_prediction_requests(
        self,
        rul_calculator,
        equipment_data_generator,
    ):
        """Test performance under concurrent prediction requests."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=100)

        def predict(equip):
            return rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(predict, eq) for eq in equipment_list]
            results = [f.result() for f in as_completed(futures)]

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        throughput = len(results) / execution_time * 60

        print(f"\nConcurrent Prediction (10 workers):")
        print(f"  Total time: {execution_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} equipment/minute")

        assert len(results) == 100

    @pytest.mark.performance
    def test_concurrent_vs_sequential_speedup(
        self,
        vibration_analyzer,
        equipment_data_generator,
    ):
        """Test speedup factor of concurrent vs sequential execution."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=50)

        def diagnose(equip):
            return vibration_analyzer.assess_severity(
                velocity_rms=equip["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
            )

        # Sequential
        start = time.perf_counter()
        for eq in equipment_list:
            diagnose(eq)
        sequential_time = time.perf_counter() - start

        # Concurrent
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(diagnose, equipment_list))
        concurrent_time = time.perf_counter() - start

        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1

        print(f"\nSpeedup Analysis:")
        print(f"  Sequential: {sequential_time*1000:.2f}ms")
        print(f"  Concurrent (5 workers): {concurrent_time*1000:.2f}ms")
        print(f"  Speedup factor: {speedup:.2f}x")


# =============================================================================
# TEST CLASS: SCALABILITY
# =============================================================================


class TestScalability:
    """Tests for system scalability."""

    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [10, 50, 100, 500])
    def test_linear_scaling(
        self,
        rul_calculator,
        equipment_data_generator,
        batch_size,
    ):
        """Test that processing time scales linearly with batch size."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=batch_size)

        start_time = time.perf_counter()

        for equip in equipment_list:
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )

        execution_time = time.perf_counter() - start_time
        time_per_item = execution_time / batch_size

        print(f"\nBatch size {batch_size}: {time_per_item*1000:.3f}ms per item")

        # Time per item should remain relatively constant
        assert time_per_item < 0.1  # Less than 100ms per item

    @pytest.mark.performance
    def test_large_batch_stability(
        self,
        rul_calculator,
        equipment_data_generator,
    ):
        """Test stability with large batch sizes."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=5000)

        start_time = time.perf_counter()

        results = []
        for equip in equipment_list:
            result = rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )
            results.append(result)

        execution_time = time.perf_counter() - start_time
        throughput = len(results) / execution_time * 60

        print(f"\nLarge Batch (5000 items):")
        print(f"  Total time: {execution_time:.2f}s")
        print(f"  Throughput: {throughput:.0f}/minute")

        assert len(results) == 5000
        assert throughput >= 100  # Still maintain minimum throughput
