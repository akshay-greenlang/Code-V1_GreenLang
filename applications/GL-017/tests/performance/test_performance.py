# -*- coding: utf-8 -*-
"""
Performance tests for GL-017 CONDENSYNC.

Tests latency targets, throughput capabilities, and memory usage
for condenser optimization calculations and agent operations.

Author: GL-017 Test Engineering Team
Target Latency: <5ms for calculations, <100ms for agent operations
Target Throughput: 1000+ calculations/second
"""

import pytest
import sys
import time
import gc
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Mock Classes for Performance Testing
# ============================================================================

@dataclass
class CondenserInput:
    """Condenser input for performance testing."""
    vacuum_pressure_mbar: float = 50.0
    hotwell_temperature_c: float = 33.2
    cooling_water_inlet_temp_c: float = 25.0
    cooling_water_outlet_temp_c: float = 32.0
    cooling_water_flow_rate_m3_hr: float = 45000.0
    heat_duty_mw: float = 180.0
    surface_area_m2: float = 17500.0
    cleanliness_factor: float = 0.85


class PerformanceCalculator:
    """Calculator optimized for performance testing."""

    def __init__(self):
        self._cache = {}

    def calculate_lmtd(self, t_hot_in: float, t_hot_out: float,
                       t_cold_in: float, t_cold_out: float) -> float:
        """Calculate LMTD."""
        import math

        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in

        if abs(delta_t1 - delta_t2) < 0.001:
            return delta_t1

        return (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

    def calculate_overall_htc(self, heat_duty_mw: float, lmtd_c: float,
                              surface_area_m2: float) -> float:
        """Calculate overall HTC."""
        heat_duty_w = heat_duty_mw * 1e6
        return heat_duty_w / (surface_area_m2 * lmtd_c)

    def calculate_vacuum_efficiency(self, current_vacuum: float,
                                    design_vacuum: float) -> float:
        """Calculate vacuum efficiency."""
        return min((design_vacuum / current_vacuum) * 100, 100.0)

    def calculate_fouling_rate(self, cleanliness_factor: float,
                               tds_ppm: float, velocity: float) -> float:
        """Calculate fouling rate."""
        base_rate = 0.01
        tds_factor = 1 + (tds_ppm - 1000) / 5000
        velocity_factor = 2.0 / velocity if velocity > 0 else 2.0
        return base_rate * tds_factor * velocity_factor

    def comprehensive_analysis(self, input_data: CondenserInput) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        lmtd = self.calculate_lmtd(
            input_data.hotwell_temperature_c + 2,
            input_data.hotwell_temperature_c,
            input_data.cooling_water_inlet_temp_c,
            input_data.cooling_water_outlet_temp_c
        )

        htc = self.calculate_overall_htc(
            input_data.heat_duty_mw,
            lmtd,
            input_data.surface_area_m2
        )

        vacuum_eff = self.calculate_vacuum_efficiency(
            input_data.vacuum_pressure_mbar,
            45.0
        )

        fouling = self.calculate_fouling_rate(
            input_data.cleanliness_factor,
            1500.0,
            2.0
        )

        return {
            'lmtd_c': lmtd,
            'overall_htc': htc,
            'vacuum_efficiency': vacuum_eff,
            'fouling_rate': fouling
        }


class MockCondenserAgent:
    """Mock agent for performance testing."""

    def __init__(self):
        self.calculator = PerformanceCalculator()
        self._cache = {}

    def orchestrate(self, input_data: CondenserInput) -> Dict[str, Any]:
        """Orchestrate analysis (mock)."""
        analysis = self.calculator.comprehensive_analysis(input_data)

        return {
            'performance_score': 85.0,
            'vacuum_efficiency': analysis['vacuum_efficiency'],
            'heat_transfer_efficiency': (analysis['overall_htc'] / 3200) * 100,
            'recommendations': ['Maintain current operating parameters'],
            'provenance_hash': 'abc123' * 10 + 'abcd'
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def calculator():
    """Create performance calculator."""
    return PerformanceCalculator()


@pytest.fixture
def agent():
    """Create mock agent."""
    return MockCondenserAgent()


@pytest.fixture
def standard_input():
    """Standard condenser input."""
    return CondenserInput()


@pytest.fixture
def batch_inputs():
    """Batch of condenser inputs for throughput testing."""
    inputs = []
    for i in range(1000):
        inputs.append(CondenserInput(
            vacuum_pressure_mbar=45.0 + (i % 40),
            cooling_water_inlet_temp_c=20.0 + (i % 15),
            cleanliness_factor=0.60 + (i % 40) * 0.01
        ))
    return inputs


# ============================================================================
# Latency Tests
# ============================================================================

class TestLatency:
    """Tests for latency targets."""

    @pytest.mark.performance
    def test_lmtd_calculation_latency(self, calculator):
        """Test LMTD calculation meets latency target (<1ms)."""
        latencies = []

        for _ in range(1000):
            start = time.perf_counter()
            calculator.calculate_lmtd(33.0, 32.5, 25.0, 32.0)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}ms exceeds 1ms target"
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.3f}ms exceeds 2ms target"

    @pytest.mark.performance
    def test_htc_calculation_latency(self, calculator):
        """Test HTC calculation meets latency target (<1ms)."""
        latencies = []

        for _ in range(1000):
            start = time.perf_counter()
            calculator.calculate_overall_htc(180.0, 10.5, 17500.0)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        assert avg_latency < 1.0

    @pytest.mark.performance
    def test_vacuum_efficiency_latency(self, calculator):
        """Test vacuum efficiency calculation meets latency target (<1ms)."""
        latencies = []

        for _ in range(1000):
            start = time.perf_counter()
            calculator.calculate_vacuum_efficiency(50.0, 45.0)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        assert avg_latency < 1.0

    @pytest.mark.performance
    def test_comprehensive_analysis_latency(self, calculator, standard_input):
        """Test comprehensive analysis meets latency target (<5ms)."""
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            calculator.comprehensive_analysis(standard_input)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_latency < 5.0, f"Average latency {avg_latency:.3f}ms exceeds 5ms target"
        assert p95_latency < 10.0, f"P95 latency {p95_latency:.3f}ms exceeds 10ms target"

    @pytest.mark.performance
    def test_agent_orchestrate_latency(self, agent, standard_input):
        """Test agent orchestration meets latency target (<100ms)."""
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            agent.orchestrate(standard_input)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_latency < 100.0, f"Average latency {avg_latency:.3f}ms exceeds 100ms target"
        assert p95_latency < 200.0, f"P95 latency {p95_latency:.3f}ms exceeds 200ms target"


# ============================================================================
# Throughput Tests
# ============================================================================

class TestThroughput:
    """Tests for throughput capabilities."""

    @pytest.mark.performance
    def test_lmtd_throughput(self, calculator):
        """Test LMTD calculation throughput (>10000/sec)."""
        count = 10000

        start = time.perf_counter()
        for _ in range(count):
            calculator.calculate_lmtd(33.0, 32.5, 25.0, 32.0)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed

        assert throughput > 10000, f"Throughput {throughput:.0f}/sec below 10000/sec target"

    @pytest.mark.performance
    def test_comprehensive_analysis_throughput(self, calculator, standard_input):
        """Test comprehensive analysis throughput (>1000/sec)."""
        count = 1000

        start = time.perf_counter()
        for _ in range(count):
            calculator.comprehensive_analysis(standard_input)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed

        assert throughput > 1000, f"Throughput {throughput:.0f}/sec below 1000/sec target"

    @pytest.mark.performance
    def test_batch_processing_throughput(self, calculator, batch_inputs):
        """Test batch processing throughput."""
        start = time.perf_counter()

        results = []
        for input_data in batch_inputs:
            results.append(calculator.comprehensive_analysis(input_data))

        elapsed = time.perf_counter() - start
        throughput = len(batch_inputs) / elapsed

        assert len(results) == len(batch_inputs)
        assert throughput > 500, f"Batch throughput {throughput:.0f}/sec below 500/sec target"

    @pytest.mark.performance
    def test_agent_throughput(self, agent, batch_inputs):
        """Test agent orchestration throughput."""
        count = 100
        inputs = batch_inputs[:count]

        start = time.perf_counter()
        results = []
        for input_data in inputs:
            results.append(agent.orchestrate(input_data))
        elapsed = time.perf_counter() - start

        throughput = count / elapsed

        assert len(results) == count
        assert throughput > 50, f"Agent throughput {throughput:.0f}/sec below 50/sec target"


# ============================================================================
# Memory Usage Tests
# ============================================================================

class TestMemoryUsage:
    """Tests for memory usage."""

    @pytest.mark.performance
    def test_calculation_memory_stability(self, calculator, standard_input):
        """Test memory usage is stable during calculations."""
        import sys

        # Force garbage collection
        gc.collect()

        # Measure initial memory
        initial_objects = len(gc.get_objects())

        # Perform many calculations
        for _ in range(10000):
            calculator.comprehensive_analysis(standard_input)

        # Force garbage collection
        gc.collect()

        # Measure final memory
        final_objects = len(gc.get_objects())

        # Allow some growth but not excessive
        growth = final_objects - initial_objects
        assert growth < 1000, f"Object count grew by {growth}, possible memory leak"

    @pytest.mark.performance
    def test_batch_memory_usage(self, calculator, batch_inputs):
        """Test memory usage during batch processing."""
        gc.collect()

        # Process in batches to check memory stability
        batch_size = 100
        memory_samples = []

        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]

            # Process batch
            results = [calculator.comprehensive_analysis(inp) for inp in batch]

            gc.collect()
            memory_samples.append(len(gc.get_objects()))

        # Check memory doesn't grow significantly across batches
        if len(memory_samples) > 1:
            max_growth = max(memory_samples) - min(memory_samples)
            avg_objects = statistics.mean(memory_samples)

            # Allow 10% variation
            assert max_growth < avg_objects * 0.1, f"Memory growth {max_growth} exceeds 10% threshold"

    @pytest.mark.performance
    def test_large_dataset_memory(self, calculator):
        """Test memory handling with large dataset."""
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create and process large number of inputs
        for i in range(5000):
            input_data = CondenserInput(
                vacuum_pressure_mbar=45.0 + (i % 40),
                cooling_water_inlet_temp_c=20.0 + (i % 15)
            )
            result = calculator.comprehensive_analysis(input_data)

            # Ensure results aren't being accumulated
            del result

        gc.collect()
        final_objects = len(gc.get_objects())

        growth = final_objects - initial_objects
        assert growth < 5000, f"Memory grew by {growth} objects"


# ============================================================================
# Scalability Tests
# ============================================================================

class TestScalability:
    """Tests for scalability characteristics."""

    @pytest.mark.performance
    def test_linear_scaling(self, calculator, standard_input):
        """Test calculation time scales linearly with load."""
        times = []

        for multiplier in [100, 200, 500, 1000]:
            start = time.perf_counter()
            for _ in range(multiplier):
                calculator.comprehensive_analysis(standard_input)
            elapsed = time.perf_counter() - start
            times.append((multiplier, elapsed))

        # Check roughly linear scaling (within 2x)
        for i in range(1, len(times)):
            expected_ratio = times[i][0] / times[0][0]
            actual_ratio = times[i][1] / times[0][1]

            assert actual_ratio < expected_ratio * 2, \
                f"Non-linear scaling detected at {times[i][0]} iterations"

    @pytest.mark.performance
    def test_concurrent_performance(self, calculator, standard_input):
        """Test performance with concurrent operations (simulated)."""
        import concurrent.futures

        def process_input(inp):
            return calculator.comprehensive_analysis(inp)

        inputs = [standard_input] * 100

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_input, inputs))
        elapsed = time.perf_counter() - start

        assert len(results) == len(inputs)
        throughput = len(inputs) / elapsed

        assert throughput > 200, f"Concurrent throughput {throughput:.0f}/sec below target"


# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for extreme conditions."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_load(self, calculator, standard_input):
        """Test sustained load over time."""
        duration_seconds = 5
        start = time.perf_counter()
        count = 0

        while time.perf_counter() - start < duration_seconds:
            calculator.comprehensive_analysis(standard_input)
            count += 1

        throughput = count / duration_seconds

        # Ensure throughput doesn't degrade over time
        assert throughput > 500, f"Sustained throughput {throughput:.0f}/sec below target"

    @pytest.mark.performance
    def test_burst_load(self, calculator, standard_input):
        """Test handling of burst load."""
        burst_size = 1000
        bursts = 5
        burst_times = []

        for _ in range(bursts):
            start = time.perf_counter()
            for _ in range(burst_size):
                calculator.comprehensive_analysis(standard_input)
            elapsed = time.perf_counter() - start
            burst_times.append(elapsed)

            # Small pause between bursts
            time.sleep(0.01)

        # All bursts should complete in similar time
        avg_time = statistics.mean(burst_times)
        max_deviation = max(abs(t - avg_time) for t in burst_times)

        assert max_deviation < avg_time * 0.5, \
            f"Burst performance variation {max_deviation:.3f}s exceeds threshold"

    @pytest.mark.performance
    def test_varying_input_sizes(self, calculator):
        """Test performance with varying input complexity."""
        times = []

        # Test with different cleanliness factors (affects fouling calculation)
        for cf in [0.50, 0.65, 0.75, 0.85, 0.95]:
            input_data = CondenserInput(cleanliness_factor=cf)

            start = time.perf_counter()
            for _ in range(1000):
                calculator.comprehensive_analysis(input_data)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        # All should complete in similar time regardless of input
        max_time = max(times)
        min_time = min(times)
        ratio = max_time / min_time if min_time > 0 else float('inf')

        assert ratio < 1.5, f"Performance varies too much with input ({ratio:.2f}x)"


# ============================================================================
# Benchmark Tests
# ============================================================================

class TestBenchmarks:
    """Benchmark tests for baseline establishment."""

    @pytest.mark.performance
    def test_calculation_benchmark(self, calculator, standard_input):
        """Benchmark comprehensive calculation."""
        iterations = 10000

        times = []
        for _ in range(10):  # Run 10 batches for statistics
            start = time.perf_counter()
            for _ in range(iterations // 10):
                calculator.comprehensive_analysis(standard_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_batch_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        # Report benchmark results
        throughput = (iterations // 10) / avg_batch_time

        # Basic assertions
        assert throughput > 100, f"Benchmark throughput {throughput:.0f}/sec too low"

        # Low standard deviation indicates consistent performance
        cv = std_dev / avg_batch_time if avg_batch_time > 0 else 0
        assert cv < 0.2, f"Performance too variable (CV={cv:.2f})"

    @pytest.mark.performance
    def test_latency_percentiles(self, calculator, standard_input):
        """Benchmark latency percentiles."""
        latencies = []

        for _ in range(10000):
            start = time.perf_counter()
            calculator.comprehensive_analysis(standard_input)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        sorted_latencies = sorted(latencies)

        p50 = sorted_latencies[int(len(latencies) * 0.50)]
        p90 = sorted_latencies[int(len(latencies) * 0.90)]
        p95 = sorted_latencies[int(len(latencies) * 0.95)]
        p99 = sorted_latencies[int(len(latencies) * 0.99)]

        # Verify latency targets
        assert p50 < 1.0, f"P50 latency {p50:.3f}ms exceeds 1ms"
        assert p90 < 2.0, f"P90 latency {p90:.3f}ms exceeds 2ms"
        assert p95 < 5.0, f"P95 latency {p95:.3f}ms exceeds 5ms"
        assert p99 < 10.0, f"P99 latency {p99:.3f}ms exceeds 10ms"
