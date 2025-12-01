# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests for GL-014 EXCHANGER-PRO.

Tests performance characteristics including:
- Calculation latency (<5ms target)
- API throughput
- Memory usage
- Concurrent request handling

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# Import test utilities
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    KernSeatonInput,
    EbertPanchalInput,
    FoulingSeverityInput,
    FluidType,
    FoulingMechanism,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    ROIInput,
    CarbonImpactInput,
    FuelType,
)


# =============================================================================
# Test Class: Calculation Latency
# =============================================================================

@pytest.mark.performance
class TestCalculationLatency:
    """Tests for calculation latency requirements."""

    TARGET_LATENCY_MS = 5.0  # Target: <5ms per calculation

    def test_calculation_latency(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test fouling calculation meets latency target."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act: Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            fouling_calculator.calculate_fouling_resistance(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[94]
        max_latency = max(latencies)

        # Assert
        assert avg_latency < self.TARGET_LATENCY_MS, (
            f"Average latency {avg_latency:.2f}ms exceeds target {self.TARGET_LATENCY_MS}ms"
        )
        assert p95_latency < self.TARGET_LATENCY_MS * 2, (
            f"P95 latency {p95_latency:.2f}ms exceeds 2x target"
        )

    def test_kern_seaton_latency(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test Kern-Seaton calculation meets latency target."""
        # Arrange
        input_data = KernSeatonInput(
            r_f_max_m2_k_w=0.0005,
            time_constant_hours=500.0,
            time_hours=200.0,
        )

        # Act
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            fouling_calculator.calculate_kern_seaton(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)

        # Assert
        assert avg_latency < self.TARGET_LATENCY_MS

    def test_economic_calculation_latency(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test economic calculation meets latency target."""
        # Arrange
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1500"),
            actual_duty_kw=Decimal("1275"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )

        # Act
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            economic_calculator.calculate_energy_loss_cost(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)

        # Assert
        assert avg_latency < self.TARGET_LATENCY_MS

    def test_roi_calculation_latency(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test ROI calculation meets latency target."""
        # Arrange: ROI with 15-year analysis is more complex
        input_data = ROIInput(
            investment_cost=Decimal("50000"),
            annual_savings=Decimal("25000"),
            analysis_period_years=15,
        )

        # Act
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            economic_calculator.perform_roi_analysis(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        # Assert: Allow slightly more for complex ROI calculations
        assert avg_latency < self.TARGET_LATENCY_MS * 2, (
            f"ROI calculation avg latency {avg_latency:.2f}ms exceeds target"
        )


# =============================================================================
# Test Class: API Throughput
# =============================================================================

@pytest.mark.performance
class TestAPIThroughput:
    """Tests for API throughput requirements."""

    TARGET_THROUGHPUT = 1000  # Target: 1000 requests/second

    def test_api_throughput(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test calculation throughput meets target."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )
        num_requests = 10000

        # Act: Measure total time for batch
        start = time.perf_counter()
        for _ in range(num_requests):
            fouling_calculator.calculate_fouling_resistance(input_data)
        end = time.perf_counter()

        elapsed_seconds = end - start
        throughput = num_requests / elapsed_seconds

        # Assert
        assert throughput > self.TARGET_THROUGHPUT, (
            f"Throughput {throughput:.0f}/s below target {self.TARGET_THROUGHPUT}/s"
        )

    def test_batch_processing_throughput(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test batch processing throughput."""
        # Arrange: Generate batch of inputs
        batch_size = 1000
        inputs = [
            FoulingResistanceInput(
                u_clean_w_m2_k=500.0 + i * 0.1,
                u_fouled_w_m2_k=420.0 + i * 0.05,
            )
            for i in range(batch_size)
        ]

        # Act
        start = time.perf_counter()
        results = [
            fouling_calculator.calculate_fouling_resistance(inp)
            for inp in inputs
        ]
        end = time.perf_counter()

        elapsed_seconds = end - start
        throughput = batch_size / elapsed_seconds

        # Assert
        assert len(results) == batch_size
        assert throughput > 500, f"Batch throughput {throughput:.0f}/s too low"


# =============================================================================
# Test Class: Memory Usage
# =============================================================================

@pytest.mark.performance
class TestMemoryUsage:
    """Tests for memory usage requirements."""

    def test_memory_usage(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test memory usage stays within bounds during batch processing."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Force garbage collection and get baseline
        gc.collect()
        import tracemalloc
        tracemalloc.start()

        # Act: Process large batch
        num_calculations = 100000
        for _ in range(num_calculations):
            result = fouling_calculator.calculate_fouling_resistance(input_data)
            # Don't store results to test for memory leaks

        # Check memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        # Assert: Peak memory should be reasonable (<100MB)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeds limit"

    def test_no_memory_leak(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test no memory leak during repeated calculations."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Get initial memory
        gc.collect()
        import tracemalloc
        tracemalloc.start()

        # Run first batch
        for _ in range(10000):
            fouling_calculator.calculate_fouling_resistance(input_data)

        first_snapshot = tracemalloc.take_snapshot()

        # Run second batch
        for _ in range(10000):
            fouling_calculator.calculate_fouling_resistance(input_data)

        second_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare memory growth
        top_stats = second_snapshot.compare_to(first_snapshot, 'lineno')

        # Calculate total growth
        total_growth = sum(stat.size_diff for stat in top_stats[:10])
        growth_kb = total_growth / 1024

        # Assert: Memory growth should be minimal (<1MB between batches)
        assert growth_kb < 1024, f"Memory grew by {growth_kb:.1f}KB between batches"


# =============================================================================
# Test Class: Concurrent Requests
# =============================================================================

@pytest.mark.performance
class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    def test_concurrent_requests(self):
        """Test handling of concurrent calculation requests."""
        # Arrange: Create multiple calculator instances
        num_threads = 10
        requests_per_thread = 100

        def worker(thread_id: int) -> List[float]:
            calculator = FoulingCalculator()
            latencies = []
            for i in range(requests_per_thread):
                input_data = FoulingResistanceInput(
                    u_clean_w_m2_k=500.0 + thread_id,
                    u_fouled_w_m2_k=420.0 + thread_id,
                )
                start = time.perf_counter()
                calculator.calculate_fouling_resistance(input_data)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            return latencies

        # Act: Run concurrent requests
        all_latencies = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                all_latencies.extend(future.result())

        # Calculate statistics
        avg_latency = statistics.mean(all_latencies)
        p99_latency = sorted(all_latencies)[int(len(all_latencies) * 0.99)]

        # Assert
        assert len(all_latencies) == num_threads * requests_per_thread
        assert avg_latency < 10, f"Concurrent avg latency {avg_latency:.2f}ms too high"

    def test_thread_safety(self):
        """Test thread safety of calculators."""
        # Arrange
        calculator = FoulingCalculator()
        num_threads = 20
        requests_per_thread = 50
        results = []
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for _ in range(requests_per_thread):
                    input_data = FoulingResistanceInput(
                        u_clean_w_m2_k=500.0,
                        u_fouled_w_m2_k=420.0,
                    )
                    result = calculator.calculate_fouling_resistance(input_data)
                    results.append(result.fouling_resistance_m2_k_w)
            except Exception as e:
                errors.append(str(e))

        # Act
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                pass

        # Assert: No errors and all results consistent
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        assert len(results) == num_threads * requests_per_thread

        # All results should be identical (deterministic)
        unique_results = set(results)
        assert len(unique_results) == 1, "Results should be identical across threads"


# =============================================================================
# Test Class: Scalability
# =============================================================================

@pytest.mark.performance
class TestScalability:
    """Tests for calculation scalability."""

    def test_linear_scaling_batch_size(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test calculation time scales linearly with batch size."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        batch_sizes = [100, 1000, 10000]
        times = []

        # Act: Measure time for each batch size
        for batch_size in batch_sizes:
            start = time.perf_counter()
            for _ in range(batch_size):
                fouling_calculator.calculate_fouling_resistance(input_data)
            end = time.perf_counter()
            times.append(end - start)

        # Assert: Time should scale approximately linearly
        # ratio of times should be roughly equal to ratio of batch sizes
        ratio_1_2 = times[1] / times[0]
        ratio_2_3 = times[2] / times[1]
        expected_ratio_1_2 = batch_sizes[1] / batch_sizes[0]
        expected_ratio_2_3 = batch_sizes[2] / batch_sizes[1]

        # Allow 50% tolerance for linear scaling
        assert 0.5 < ratio_1_2 / expected_ratio_1_2 < 2.0
        assert 0.5 < ratio_2_3 / expected_ratio_2_3 < 2.0

    def test_complexity_analysis(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test ROI calculation complexity with analysis period."""
        # Arrange: Different analysis periods
        periods = [5, 10, 15, 20]
        times = []

        for period in periods:
            input_data = ROIInput(
                investment_cost=Decimal("50000"),
                annual_savings=Decimal("25000"),
                analysis_period_years=period,
            )

            start = time.perf_counter()
            for _ in range(100):
                economic_calculator.perform_roi_analysis(input_data)
            end = time.perf_counter()
            times.append(end - start)

        # Assert: Time should increase with period but not exponentially
        # Ratio of longest to shortest should be reasonable
        ratio = times[-1] / times[0]
        period_ratio = periods[-1] / periods[0]

        # Allow for linear or slightly superlinear scaling
        assert ratio < period_ratio * 3, (
            f"Calculation time ratio {ratio:.1f} too high for period ratio {period_ratio}"
        )


# =============================================================================
# Test Class: Warm-up Performance
# =============================================================================

@pytest.mark.performance
class TestWarmupPerformance:
    """Tests for warm-up effects on performance."""

    def test_warmup_effect(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test first calculation is not significantly slower than subsequent."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act: First calculation
        start = time.perf_counter()
        fouling_calculator.calculate_fouling_resistance(input_data)
        first_time = (time.perf_counter() - start) * 1000

        # Subsequent calculations
        subsequent_times = []
        for _ in range(100):
            start = time.perf_counter()
            fouling_calculator.calculate_fouling_resistance(input_data)
            subsequent_times.append((time.perf_counter() - start) * 1000)

        avg_subsequent = statistics.mean(subsequent_times)

        # Assert: First should not be more than 10x slower
        assert first_time < avg_subsequent * 10, (
            f"First calculation {first_time:.2f}ms much slower than avg {avg_subsequent:.2f}ms"
        )
