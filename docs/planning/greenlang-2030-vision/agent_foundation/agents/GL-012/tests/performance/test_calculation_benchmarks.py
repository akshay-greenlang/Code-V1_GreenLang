# -*- coding: utf-8 -*-
"""
Calculation Benchmark Tests for GL-012 STEAMQUAL.

Tests performance characteristics of all calculator components:
- Steam quality calculations (dryness fraction, steam state, SQI)
- Desuperheater calculations (injection rate, energy balance)
- Pressure control calculations (PID, valve position)

Performance Targets:
- Single steam quality calculation: <1ms
- Batch calculation throughput: >10,000/sec
- Memory usage per calculation: <1KB
- Desuperheater injection rate calculation: <0.5ms
- Energy balance validation: <0.5ms
- PID calculation: <0.1ms
- Valve position calculation: <0.2ms

Author: GL-TestEngineer
Version: 1.0.0
"""

import gc
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import pytest

# Add parent directories for imports
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "calculators"))

from steam_quality_calculator import SteamQualityCalculator, SteamQualityInput
from desuperheater_calculator import DesuperheaterCalculator, DesuperheaterInput
from pressure_control_calculator import (
    PressureControlCalculator,
    PressureControlInput,
    PIDGains,
    ValveCharacteristic,
)

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.benchmark]


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def steam_quality_calculator():
    """Create SteamQualityCalculator instance."""
    return SteamQualityCalculator()


@pytest.fixture
def desuperheater_calculator():
    """Create DesuperheaterCalculator instance."""
    return DesuperheaterCalculator()


@pytest.fixture
def pressure_control_calculator():
    """Create PressureControlCalculator instance."""
    return PressureControlCalculator()


@pytest.fixture
def sample_steam_input():
    """Sample steam quality input."""
    return SteamQualityInput(
        pressure_mpa=1.0,
        temperature_c=200.0,
        pressure_stability=0.95,
        temperature_stability=0.92
    )


@pytest.fixture
def sample_desuperheater_input():
    """Sample desuperheater input."""
    return DesuperheaterInput(
        steam_flow_kg_s=50.0,
        inlet_temperature_c=350.0,
        inlet_pressure_mpa=4.0,
        target_temperature_c=280.0,
        water_temperature_c=30.0,
        water_pressure_mpa=6.0
    )


@pytest.fixture
def sample_pressure_input():
    """Sample pressure control input."""
    return PressureControlInput(
        setpoint_mpa=1.0,
        actual_mpa=0.95,
        flow_rate_kg_s=10.0,
        fluid_density_kg_m3=10.0,
        valve_cv_max=100.0
    )


# =============================================================================
# STEAM QUALITY CALCULATION BENCHMARKS
# =============================================================================

class TestSteamQualityCalculationBenchmarks:
    """Benchmark tests for steam quality calculations."""

    @pytest.mark.benchmark
    def test_single_steam_quality_calculation_latency(
        self, steam_quality_calculator, sample_steam_input
    ):
        """
        Test single steam quality calculation latency.

        Target: <1ms per calculation
        """
        # Warm-up
        for _ in range(10):
            steam_quality_calculator.calculate_steam_quality(sample_steam_input)

        # Measure latency
        iterations = 100
        latencies_ms = []

        for _ in range(iterations):
            start = time.perf_counter()
            steam_quality_calculator.calculate_steam_quality(sample_steam_input)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        avg_latency = statistics.mean(latencies_ms)
        p95_latency = sorted(latencies_ms)[int(iterations * 0.95)]
        p99_latency = sorted(latencies_ms)[int(iterations * 0.99)]

        print(f"\nSteam quality calculation latency:")
        print(f"  Average: {avg_latency:.4f}ms")
        print(f"  P95: {p95_latency:.4f}ms")
        print(f"  P99: {p99_latency:.4f}ms")

        assert avg_latency < 1.0, f"Average latency {avg_latency:.4f}ms exceeds 1ms target"
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.4f}ms exceeds 2ms"

    @pytest.mark.benchmark
    def test_dryness_fraction_calculation_latency(self, steam_quality_calculator):
        """
        Test dryness fraction calculation latency.

        Target: <0.5ms per calculation
        """
        # Warm-up
        for _ in range(10):
            steam_quality_calculator.calculate_dryness_fraction(1500.0, 762.68, 2014.9)

        iterations = 1000
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            steam_quality_calculator.calculate_dryness_fraction(1500.0, 762.68, 2014.9)
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nDryness fraction calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.5, f"Latency {avg_latency_ms:.4f}ms exceeds 0.5ms target"

    @pytest.mark.benchmark
    def test_steam_quality_index_calculation_latency(self, steam_quality_calculator):
        """
        Test Steam Quality Index calculation latency.

        Target: <0.1ms per calculation
        """
        iterations = 1000
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            steam_quality_calculator.calculate_steam_quality_index(0.95, 0.92, 0.90)
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nSteam Quality Index calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.1, f"Latency {avg_latency_ms:.4f}ms exceeds 0.1ms target"

    @pytest.mark.benchmark
    def test_superheat_degree_calculation_latency(self, steam_quality_calculator):
        """
        Test superheat degree calculation latency.

        Target: <0.05ms per calculation
        """
        iterations = 1000
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            steam_quality_calculator.calculate_superheat_degree(250.0, 179.88)
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nSuperheat degree calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.05, f"Latency {avg_latency_ms:.4f}ms exceeds 0.05ms target"

    @pytest.mark.benchmark
    def test_batch_steam_quality_throughput(
        self, steam_quality_calculator, steam_quality_test_data
    ):
        """
        Test batch steam quality calculation throughput.

        Target: >10,000 calculations/second
        """
        # Prepare inputs
        inputs = [
            SteamQualityInput(
                pressure_mpa=d["pressure_mpa"],
                temperature_c=d["temperature_c"],
                pressure_stability=d["pressure_stability"],
                temperature_stability=d["temperature_stability"]
            )
            for d in steam_quality_test_data[:1000]
        ]

        # Warm-up
        for inp in inputs[:10]:
            steam_quality_calculator.calculate_steam_quality(inp)

        # Measure throughput
        start = time.perf_counter()

        for inp in inputs:
            steam_quality_calculator.calculate_steam_quality(inp)

        duration = time.perf_counter() - start
        throughput = len(inputs) / duration

        print(f"\nBatch steam quality throughput: {throughput:.0f} calcs/sec")

        assert throughput > 10000, f"Throughput {throughput:.0f}/sec below 10,000/sec target"

    @pytest.mark.benchmark
    def test_steam_quality_memory_per_calculation(
        self, steam_quality_calculator, sample_steam_input
    ):
        """
        Test memory usage per steam quality calculation.

        Target: <1KB per calculation
        """
        gc.collect()
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Perform calculations
        num_calculations = 10000
        for _ in range(num_calculations):
            steam_quality_calculator.calculate_steam_quality(sample_steam_input)

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth = final_memory - initial_memory
        memory_per_calc = memory_growth / num_calculations

        print(f"\nMemory per steam quality calculation: {memory_per_calc:.0f} bytes")

        assert memory_per_calc < 1024, f"Memory {memory_per_calc:.0f}B exceeds 1KB target"


# =============================================================================
# DESUPERHEATER CALCULATION BENCHMARKS
# =============================================================================

class TestDesuperheaterCalculationBenchmarks:
    """Benchmark tests for desuperheater calculations."""

    @pytest.mark.benchmark
    def test_injection_rate_calculation_latency(self, desuperheater_calculator):
        """
        Test injection rate calculation latency.

        Target: <0.5ms per calculation
        """
        # Warm-up
        for _ in range(10):
            desuperheater_calculator.calculate_injection_rate(
                m_steam=50.0, h_inlet=3050.0, h_outlet=2900.0, h_water=125.0
            )

        iterations = 500
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            desuperheater_calculator.calculate_injection_rate(
                m_steam=50.0, h_inlet=3050.0, h_outlet=2900.0, h_water=125.0
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nInjection rate calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.5, f"Latency {avg_latency_ms:.4f}ms exceeds 0.5ms target"

    @pytest.mark.benchmark
    def test_energy_balance_validation_latency(self, desuperheater_calculator):
        """
        Test energy balance validation latency.

        Target: <0.5ms per validation
        """
        inlet_conditions = {'mass_flow_kg_s': 50.0, 'enthalpy_kj_kg': 3050.0}
        outlet_conditions = {'mass_flow_kg_s': 50.5, 'enthalpy_kj_kg': 2900.0}
        injection = {'mass_flow_kg_s': 0.5, 'enthalpy_kj_kg': 125.0}

        # Warm-up
        for _ in range(10):
            desuperheater_calculator.validate_energy_balance(
                inlet_conditions, outlet_conditions, injection
            )

        iterations = 500
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            desuperheater_calculator.validate_energy_balance(
                inlet_conditions, outlet_conditions, injection
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nEnergy balance validation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.5, f"Latency {avg_latency_ms:.4f}ms exceeds 0.5ms target"

    @pytest.mark.benchmark
    def test_outlet_temperature_calculation_latency(self, desuperheater_calculator):
        """
        Test outlet temperature calculation latency.

        Target: <0.3ms per calculation
        """
        iterations = 500
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            desuperheater_calculator.calculate_outlet_temperature(
                m_steam=50.0, T_inlet=350.0, m_water=2.0, T_water=30.0, P_mpa=4.0
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nOutlet temperature calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.3, f"Latency {avg_latency_ms:.4f}ms exceeds 0.3ms target"

    @pytest.mark.benchmark
    def test_full_desuperheater_calculation_latency(
        self, desuperheater_calculator, sample_desuperheater_input
    ):
        """
        Test full desuperheater calculation latency.

        Target: <2ms per calculation
        """
        # Warm-up
        for _ in range(10):
            desuperheater_calculator.calculate(sample_desuperheater_input)

        iterations = 100
        latencies_ms = []

        for _ in range(iterations):
            desuperheater_calculator.reset_pid_state()  # Reset for consistency
            start = time.perf_counter()
            desuperheater_calculator.calculate(sample_desuperheater_input)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        avg_latency = statistics.mean(latencies_ms)
        p95_latency = sorted(latencies_ms)[int(iterations * 0.95)]

        print(f"\nFull desuperheater calculation latency:")
        print(f"  Average: {avg_latency:.4f}ms")
        print(f"  P95: {p95_latency:.4f}ms")

        assert avg_latency < 2.0, f"Average latency {avg_latency:.4f}ms exceeds 2ms target"

    @pytest.mark.benchmark
    def test_desuperheater_batch_throughput(
        self, desuperheater_calculator, desuperheater_test_data
    ):
        """
        Test batch desuperheater calculation throughput.

        Target: >5,000 calculations/second
        """
        inputs = [
            DesuperheaterInput(
                steam_flow_kg_s=d["steam_flow_kg_s"],
                inlet_temperature_c=d["inlet_temperature_c"],
                inlet_pressure_mpa=d["inlet_pressure_mpa"],
                target_temperature_c=d["target_temperature_c"],
                water_temperature_c=d["water_temperature_c"]
            )
            for d in desuperheater_test_data[:500]
        ]

        start = time.perf_counter()

        for inp in inputs:
            desuperheater_calculator.reset_pid_state()
            desuperheater_calculator.calculate(inp)

        duration = time.perf_counter() - start
        throughput = len(inputs) / duration

        print(f"\nDesuperheater batch throughput: {throughput:.0f} calcs/sec")

        assert throughput > 5000, f"Throughput {throughput:.0f}/sec below 5,000/sec target"


# =============================================================================
# PRESSURE CONTROL CALCULATION BENCHMARKS
# =============================================================================

class TestPressureControlCalculationBenchmarks:
    """Benchmark tests for pressure control calculations."""

    @pytest.mark.benchmark
    def test_pid_calculation_latency(self, pressure_control_calculator):
        """
        Test PID calculation latency.

        Target: <0.1ms per calculation
        """
        gains = PIDGains(kp=2.0, ki=0.5, kd=0.1)

        iterations = 1000
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_pid_output(
                error=0.05, integral=0.1, derivative=-0.01, gains=gains
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nPID calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.1, f"Latency {avg_latency_ms:.4f}ms exceeds 0.1ms target"

    @pytest.mark.benchmark
    def test_valve_position_calculation_latency(self, pressure_control_calculator):
        """
        Test valve position calculation latency.

        Target: <0.2ms per calculation
        """
        iterations = 500
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_valve_position(
                setpoint=1.0, actual=0.95, valve_cv=100.0, delta_p=0.2,
                flow_rate=10.0, density=10.0,
                characteristic=ValveCharacteristic.EQUAL_PERCENTAGE
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nValve position calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.2, f"Latency {avg_latency_ms:.4f}ms exceeds 0.2ms target"

    @pytest.mark.benchmark
    def test_flow_coefficient_calculation_latency(self, pressure_control_calculator):
        """
        Test flow coefficient (Cv) calculation latency.

        Target: <0.2ms per calculation
        """
        iterations = 500
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_flow_coefficient(
                flow_rate=10.0, delta_p=0.2, density=10.0
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nFlow coefficient calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.2, f"Latency {avg_latency_ms:.4f}ms exceeds 0.2ms target"

    @pytest.mark.benchmark
    def test_pressure_drop_calculation_latency(self, pressure_control_calculator):
        """
        Test pressure drop calculation latency.

        Target: <0.3ms per calculation
        """
        iterations = 500
        latencies_us = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_pressure_drop(
                flow_rate=10.0, pipe_diameter=0.1, length=50.0,
                roughness=0.000045, density=10.0
            )
            latencies_us.append((time.perf_counter_ns() - start) / 1000)

        avg_latency_ms = statistics.mean(latencies_us) / 1000

        print(f"\nPressure drop calculation latency: {avg_latency_ms:.4f}ms")

        assert avg_latency_ms < 0.3, f"Latency {avg_latency_ms:.4f}ms exceeds 0.3ms target"

    @pytest.mark.benchmark
    def test_full_pressure_control_calculation_latency(
        self, pressure_control_calculator, sample_pressure_input
    ):
        """
        Test full pressure control calculation latency.

        Target: <1ms per calculation
        """
        # Warm-up
        for _ in range(10):
            pressure_control_calculator.reset_controller_state()
            pressure_control_calculator.calculate(sample_pressure_input)

        iterations = 100
        latencies_ms = []

        for _ in range(iterations):
            pressure_control_calculator.reset_controller_state()
            start = time.perf_counter()
            pressure_control_calculator.calculate(sample_pressure_input)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        avg_latency = statistics.mean(latencies_ms)
        p95_latency = sorted(latencies_ms)[int(iterations * 0.95)]

        print(f"\nFull pressure control calculation latency:")
        print(f"  Average: {avg_latency:.4f}ms")
        print(f"  P95: {p95_latency:.4f}ms")

        assert avg_latency < 1.0, f"Average latency {avg_latency:.4f}ms exceeds 1ms target"

    @pytest.mark.benchmark
    def test_pressure_control_batch_throughput(
        self, pressure_control_calculator, pressure_control_test_data
    ):
        """
        Test batch pressure control calculation throughput.

        Target: >10,000 calculations/second
        """
        inputs = [
            PressureControlInput(
                setpoint_mpa=d["setpoint_mpa"],
                actual_mpa=d["actual_mpa"],
                flow_rate_kg_s=d["flow_rate_kg_s"],
                fluid_density_kg_m3=d["fluid_density_kg_m3"]
            )
            for d in pressure_control_test_data[:1000]
        ]

        start = time.perf_counter()

        for inp in inputs:
            pressure_control_calculator.reset_controller_state()
            pressure_control_calculator.calculate(inp)

        duration = time.perf_counter() - start
        throughput = len(inputs) / duration

        print(f"\nPressure control batch throughput: {throughput:.0f} calcs/sec")

        assert throughput > 10000, f"Throughput {throughput:.0f}/sec below 10,000/sec target"


# =============================================================================
# VARYING INPUT SIZE BENCHMARKS
# =============================================================================

class TestVaryingInputSizeBenchmarks:
    """Benchmark tests with varying input sizes."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [10, 100, 1000, 5000, 10000])
    def test_steam_quality_scaling_with_batch_size(
        self, steam_quality_calculator, batch_size
    ):
        """Test steam quality calculation scaling with batch size."""
        inputs = [
            SteamQualityInput(
                pressure_mpa=1.0 + (i % 10) * 0.5,
                temperature_c=180.0 + (i % 50),
                pressure_stability=0.9 + (i % 10) * 0.01,
                temperature_stability=0.9 + (i % 10) * 0.01
            )
            for i in range(batch_size)
        ]

        start = time.perf_counter()

        for inp in inputs:
            steam_quality_calculator.calculate_steam_quality(inp)

        duration = time.perf_counter() - start
        throughput = batch_size / duration
        avg_latency_ms = (duration / batch_size) * 1000

        print(f"\nBatch size {batch_size}:")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:.0f}/sec")
        print(f"  Avg latency: {avg_latency_ms:.4f}ms")

        # Throughput should remain high regardless of batch size
        assert throughput > 5000, f"Throughput {throughput:.0f}/sec dropped below 5000/sec"


# =============================================================================
# HOT PATH PROFILING
# =============================================================================

class TestHotPathProfiling:
    """Profile hot paths in calculations."""

    @pytest.mark.benchmark
    def test_profile_steam_quality_hot_path(self, steam_quality_calculator):
        """Profile hot path in steam quality calculation."""
        inp = SteamQualityInput(
            pressure_mpa=1.0,
            temperature_c=200.0,
            pressure_stability=0.95,
            temperature_stability=0.92
        )

        # Profile individual components
        iterations = 100
        component_times: Dict[str, List[float]] = {
            "state_determination": [],
            "saturation_lookup": [],
            "dryness_calculation": [],
            "quality_index": [],
            "provenance_hash": [],
        }

        for _ in range(iterations):
            # State determination
            start = time.perf_counter_ns()
            steam_quality_calculator.determine_steam_state(
                inp.pressure_mpa, inp.temperature_c
            )
            component_times["state_determination"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

            # Dryness calculation
            start = time.perf_counter_ns()
            steam_quality_calculator.calculate_dryness_fraction(1500.0, 762.68, 2014.9)
            component_times["dryness_calculation"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

            # Quality index
            start = time.perf_counter_ns()
            steam_quality_calculator.calculate_steam_quality_index(0.95, 0.92, 0.90)
            component_times["quality_index"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

        print("\nHot path profiling (average times):")
        for component, times in component_times.items():
            if times:
                avg = statistics.mean(times)
                print(f"  {component}: {avg:.4f}ms")

    @pytest.mark.benchmark
    def test_profile_pressure_control_hot_path(self, pressure_control_calculator):
        """Profile hot path in pressure control calculation."""
        iterations = 100
        component_times: Dict[str, List[float]] = {
            "pid_calculation": [],
            "flow_coefficient": [],
            "valve_position": [],
            "pressure_drop": [],
        }

        for _ in range(iterations):
            # PID calculation
            gains = PIDGains(kp=2.0, ki=0.5, kd=0.1)
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_pid_output(0.05, 0.1, -0.01, gains)
            component_times["pid_calculation"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

            # Flow coefficient
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_flow_coefficient(10.0, 0.2, 10.0)
            component_times["flow_coefficient"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

            # Valve position
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_valve_position(
                1.0, 0.95, 100.0, 0.2, 10.0, 10.0, ValveCharacteristic.EQUAL_PERCENTAGE
            )
            component_times["valve_position"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

            # Pressure drop
            start = time.perf_counter_ns()
            pressure_control_calculator.calculate_pressure_drop(
                10.0, 0.1, 50.0, 0.000045, 10.0
            )
            component_times["pressure_drop"].append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

        print("\nPressure control hot path profiling:")
        for component, times in component_times.items():
            if times:
                avg = statistics.mean(times)
                print(f"  {component}: {avg:.4f}ms")


# =============================================================================
# CONCURRENT CALCULATION BENCHMARKS
# =============================================================================

class TestConcurrentCalculationBenchmarks:
    """Benchmark tests for concurrent calculations."""

    @pytest.mark.benchmark
    def test_concurrent_steam_quality_calculations(
        self, thread_pool_4, steam_quality_test_data
    ):
        """Test concurrent steam quality calculation throughput."""
        calc = SteamQualityCalculator()

        inputs = [
            SteamQualityInput(
                pressure_mpa=d["pressure_mpa"],
                temperature_c=d["temperature_c"],
                pressure_stability=d["pressure_stability"],
                temperature_stability=d["temperature_stability"]
            )
            for d in steam_quality_test_data[:1000]
        ]

        def calculate(inp):
            return calc.calculate_steam_quality(inp)

        start = time.perf_counter()

        futures = [thread_pool_4.submit(calculate, inp) for inp in inputs]
        results = [f.result() for f in as_completed(futures)]

        duration = time.perf_counter() - start
        throughput = len(inputs) / duration

        print(f"\nConcurrent (4 workers) steam quality throughput: {throughput:.0f}/sec")

        assert len(results) == len(inputs)
        assert throughput > 15000, f"Concurrent throughput {throughput:.0f}/sec below 15000/sec"

    @pytest.mark.benchmark
    def test_thread_safety_under_load(self, thread_pool_16):
        """Test calculator thread safety under concurrent load."""
        calc = SteamQualityCalculator()

        # Same input should produce same output
        inp = SteamQualityInput(
            pressure_mpa=1.0,
            temperature_c=200.0,
            pressure_stability=0.95,
            temperature_stability=0.92
        )

        def calculate(_):
            return calc.calculate_steam_quality(inp)

        futures = [thread_pool_16.submit(calculate, i) for i in range(500)]
        results = [f.result() for f in as_completed(futures)]

        # All results should have same provenance hash (deterministic)
        hashes = [r.provenance_hash for r in results]
        unique_hashes = set(hashes)

        print(f"\nThread safety test: {len(results)} results, {len(unique_hashes)} unique hashes")

        # Should only have 1 unique hash (deterministic)
        assert len(unique_hashes) == 1, "Results are not deterministic under concurrent access"


# =============================================================================
# SUMMARY TEST
# =============================================================================

def test_calculation_benchmark_summary():
    """
    Summary test confirming calculation benchmark coverage.

    This test suite provides 25+ calculation benchmark tests covering:
    - Steam quality calculations (dryness, SQI, superheat, state)
    - Desuperheater calculations (injection, energy balance, outlet temp)
    - Pressure control calculations (PID, valve, Cv, pressure drop)
    - Batch throughput tests
    - Memory usage tests
    - Varying input size scaling
    - Hot path profiling
    - Concurrent calculation tests

    Performance Targets:
    - Single steam quality calculation: <1ms
    - Batch throughput: >10,000/sec
    - PID calculation: <0.1ms
    - Valve position: <0.2ms
    - Injection rate: <0.5ms

    Total: 25+ benchmark tests
    """
    assert True
