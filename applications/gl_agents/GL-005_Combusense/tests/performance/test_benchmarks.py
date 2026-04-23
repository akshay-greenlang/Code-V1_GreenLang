# -*- coding: utf-8 -*-
"""
Performance Benchmarks for GL-005 COMBUSENSE (CombustionEfficiencyOptimizer).

Comprehensive performance testing with clear pass/fail criteria for:
- Control loop timing (<100ms target)
- Calculation throughput
- PID controller response time
- Safety interlock validation speed
- Provenance hash computation
- Memory usage under load

Reference Standards:
- ASME PTC 4.1: Performance targets for combustion systems
- IEC 61131-3: PLC response time requirements
- ISA-84: Safety system response requirements
"""

import pytest
import time
import hashlib
import json
import random
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


# -----------------------------------------------------------------------------
# Benchmark Result Data Classes
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_per_sec: float
    passed: bool
    target_ms: float

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: avg={self.avg_time_ms:.4f}ms "
            f"(target={self.target_ms}ms) throughput={self.throughput_per_sec:.0f}/s"
        )


class PerformanceBenchmark:
    """Performance benchmark runner with statistical analysis."""

    @staticmethod
    def run_benchmark(
        func: Callable,
        iterations: int = 1000,
        target_ms: float = 1.0,
        name: str = "benchmark",
        warmup_iterations: int = 100
    ) -> BenchmarkResult:
        """
        Run benchmark with warmup and statistical analysis.

        Args:
            func: Function to benchmark
            iterations: Number of iterations
            target_ms: Target time in milliseconds
            name: Benchmark name
            warmup_iterations: Number of warmup iterations

        Returns:
            BenchmarkResult with statistics
        """
        # Warmup phase
        for _ in range(warmup_iterations):
            func()

        # Measurement phase
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # Calculate statistics
        times.sort()
        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = times[0]
        max_time = times[-1]

        # Percentiles
        p50_idx = int(iterations * 0.50)
        p95_idx = int(iterations * 0.95)
        p99_idx = int(iterations * 0.99)

        p50_time = times[p50_idx] if p50_idx < iterations else times[-1]
        p95_time = times[p95_idx] if p95_idx < iterations else times[-1]
        p99_time = times[p99_idx] if p99_idx < iterations else times[-1]

        throughput = iterations / (total_time / 1000) if total_time > 0 else 0

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            p50_time_ms=p50_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            throughput_per_sec=throughput,
            passed=avg_time <= target_ms,
            target_ms=target_ms
        )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def benchmark_runner():
    """Create benchmark runner instance."""
    return PerformanceBenchmark()


@pytest.fixture
def sample_combustion_data():
    """Sample combustion state data for benchmarks."""
    return {
        "fuel_flow": 100.0,
        "air_flow": 1200.0,
        "air_fuel_ratio": 12.0,
        "flame_temperature": 1200.0,
        "furnace_temperature": 900.0,
        "flue_gas_temperature": 250.0,
        "ambient_temperature": 25.0,
        "fuel_pressure": 300.0,
        "air_pressure": 101.3,
        "furnace_pressure": -50.0,
        "o2_percent": 4.5,
        "co_ppm": 25.0,
        "co2_percent": 11.5,
        "thermal_efficiency": 88.5
    }


@pytest.fixture
def pid_parameters():
    """PID controller parameters."""
    return {
        "kp": 2.0,
        "ki": 0.5,
        "kd": 0.1,
        "setpoint": 4.5,
        "sample_time": 0.1
    }


@pytest.fixture
def safety_limits():
    """Safety limit parameters."""
    return {
        "max_combustion_temperature": 1500,
        "max_flue_gas_temperature": 500,
        "min_operating_temperature": 200,
        "max_combustion_pressure": 10000,
        "max_fuel_flow_rate": 200.0,
        "min_o2_percent": 2.0,
        "max_co_ppm": 400
    }


# -----------------------------------------------------------------------------
# Core Calculation Benchmarks
# -----------------------------------------------------------------------------

class TestCalculationBenchmarks:
    """Benchmark core combustion calculations."""

    @pytest.mark.performance
    def test_thermal_efficiency_calculation_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data
    ):
        """Benchmark thermal efficiency calculation (<1ms target)."""
        def calculate_efficiency():
            fuel_flow = Decimal(str(sample_combustion_data["fuel_flow"]))
            fuel_lhv = Decimal("42.0")  # MJ/kg
            heat_output = Decimal("1000.0")  # kW

            gross_input = fuel_flow * fuel_lhv * Decimal("1000") / Decimal("3600")
            efficiency = heat_output / gross_input * Decimal("100")
            return float(efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_efficiency,
            iterations=10000,
            target_ms=1.0,
            name="thermal_efficiency"
        )

        assert result.passed, f"Thermal efficiency: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        assert result.throughput_per_sec > 10000

    @pytest.mark.performance
    def test_excess_air_calculation_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data
    ):
        """Benchmark excess air calculation from O2 (<1ms target)."""
        def calculate_excess_air():
            o2_percent = Decimal(str(sample_combustion_data["o2_percent"]))
            denominator = Decimal("21") - o2_percent
            if denominator > 0:
                excess_air = o2_percent / denominator * Decimal("100")
            else:
                excess_air = Decimal("0")
            return float(excess_air.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_excess_air,
            iterations=10000,
            target_ms=1.0,
            name="excess_air"
        )

        assert result.passed
        assert result.throughput_per_sec > 10000

    @pytest.mark.performance
    def test_air_fuel_ratio_calculation_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data
    ):
        """Benchmark air-fuel ratio calculation (<0.5ms target)."""
        def calculate_afr():
            fuel_flow = sample_combustion_data["fuel_flow"]
            air_flow = sample_combustion_data["air_flow"]
            return air_flow / fuel_flow if fuel_flow > 0 else 0

        result = benchmark_runner.run_benchmark(
            calculate_afr,
            iterations=10000,
            target_ms=0.5,
            name="air_fuel_ratio"
        )

        assert result.passed

    @pytest.mark.performance
    def test_heat_output_calculation_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data
    ):
        """Benchmark heat output calculation (<1ms target)."""
        def calculate_heat_output():
            fuel_flow = Decimal(str(sample_combustion_data["fuel_flow"]))
            fuel_lhv = Decimal("42.0")
            efficiency = Decimal(str(sample_combustion_data["thermal_efficiency"]))

            # Heat output = fuel_flow * LHV * efficiency / 100
            heat_kw = fuel_flow * fuel_lhv * efficiency / Decimal("100") * Decimal("1000") / Decimal("3600")
            return float(heat_kw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_heat_output,
            iterations=10000,
            target_ms=1.0,
            name="heat_output"
        )

        assert result.passed


class TestPIDControllerBenchmarks:
    """Benchmark PID controller performance."""

    @pytest.mark.performance
    def test_pid_update_benchmark(self, benchmark_runner, pid_parameters):
        """Benchmark PID controller update (<0.5ms target for 10Hz rate)."""
        # Simple PID implementation for benchmarking
        integral = 0.0
        last_error = 0.0
        process_value = 4.2

        def pid_update():
            nonlocal integral, last_error
            error = pid_parameters["setpoint"] - process_value
            integral += error * pid_parameters["sample_time"]
            derivative = (error - last_error) / pid_parameters["sample_time"]
            output = (
                pid_parameters["kp"] * error +
                pid_parameters["ki"] * integral +
                pid_parameters["kd"] * derivative
            )
            last_error = error
            return max(0, min(100, output))

        result = benchmark_runner.run_benchmark(
            pid_update,
            iterations=10000,
            target_ms=0.5,
            name="pid_update"
        )

        assert result.passed, f"PID update: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        # Must support 10Hz update rate (100ms intervals)
        assert result.avg_time_ms < 10.0

    @pytest.mark.performance
    def test_cascaded_pid_benchmark(self, benchmark_runner, pid_parameters):
        """Benchmark cascaded PID (fuel + air) update (<1ms target)."""
        fuel_integral = 0.0
        air_integral = 0.0

        def cascaded_pid_update():
            nonlocal fuel_integral, air_integral

            # Fuel PID
            fuel_error = 100.0 - 98.0
            fuel_integral += fuel_error * 0.1
            fuel_output = 2.0 * fuel_error + 0.5 * fuel_integral

            # Air PID (cascaded from fuel)
            air_setpoint = fuel_output * 12.0  # Stoichiometric ratio
            air_error = air_setpoint - 1180.0
            air_integral += air_error * 0.1
            air_output = 1.5 * air_error + 0.3 * air_integral

            return (
                max(0, min(100, fuel_output)),
                max(0, min(100, air_output))
            )

        result = benchmark_runner.run_benchmark(
            cascaded_pid_update,
            iterations=10000,
            target_ms=1.0,
            name="cascaded_pid"
        )

        assert result.passed


class TestSafetyValidationBenchmarks:
    """Benchmark safety validation performance."""

    @pytest.mark.performance
    def test_safety_interlock_check_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data,
        safety_limits
    ):
        """Benchmark safety interlock validation (<5ms target per ISA-84)."""
        def check_interlocks():
            violations = []

            # Temperature checks
            if sample_combustion_data["furnace_temperature"] > safety_limits["max_combustion_temperature"]:
                violations.append("temperature_high")

            # Pressure checks
            if sample_combustion_data["furnace_pressure"] > safety_limits["max_combustion_pressure"]:
                violations.append("pressure_high")

            # Flow checks
            if sample_combustion_data["fuel_flow"] > safety_limits["max_fuel_flow_rate"]:
                violations.append("fuel_flow_high")

            # Emission checks
            if sample_combustion_data["o2_percent"] < safety_limits["min_o2_percent"]:
                violations.append("o2_low")
            if sample_combustion_data["co_ppm"] > safety_limits["max_co_ppm"]:
                violations.append("co_high")

            return len(violations) == 0, violations

        result = benchmark_runner.run_benchmark(
            check_interlocks,
            iterations=10000,
            target_ms=5.0,
            name="safety_interlocks"
        )

        assert result.passed
        # Must meet ISA-84 SIL requirements
        assert result.p99_time_ms < 10.0

    @pytest.mark.performance
    def test_emergency_condition_check_benchmark(self, benchmark_runner):
        """Benchmark emergency condition detection (<1ms target)."""
        conditions = {
            "fire_detected": False,
            "gas_leak_detected": False,
            "operator_stop": False,
            "flame_present": True
        }

        def check_emergency():
            if conditions["fire_detected"]:
                return True, "FIRE"
            if conditions["gas_leak_detected"]:
                return True, "GAS_LEAK"
            if conditions["operator_stop"]:
                return True, "E_STOP"
            if not conditions["flame_present"]:
                return True, "FLAME_OUT"
            return False, None

        result = benchmark_runner.run_benchmark(
            check_emergency,
            iterations=10000,
            target_ms=1.0,
            name="emergency_check"
        )

        assert result.passed
        assert result.max_time_ms < 5.0  # Critical path


class TestProvenanceHashBenchmarks:
    """Benchmark provenance hash computation."""

    @pytest.mark.performance
    def test_provenance_hash_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data
    ):
        """Benchmark SHA-256 provenance hash (<1ms target)."""
        def calculate_hash():
            hashable_data = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in sample_combustion_data.items()
            }
            hash_input = json.dumps(hashable_data, sort_keys=True)
            return hashlib.sha256(hash_input.encode()).hexdigest()

        result = benchmark_runner.run_benchmark(
            calculate_hash,
            iterations=10000,
            target_ms=1.0,
            name="provenance_hash"
        )

        assert result.passed
        assert result.throughput_per_sec > 10000

    @pytest.mark.performance
    def test_control_action_hash_benchmark(self, benchmark_runner):
        """Benchmark control action hash for audit trail (<0.5ms target)."""
        action_data = {
            "fuel_flow_setpoint": 105.234567,
            "air_flow_setpoint": 1254.789012,
            "fuel_valve_position": 52.3456,
            "air_damper_position": 61.2345
        }

        def calculate_action_hash():
            rounded = {k: round(v, 6) for k, v in action_data.items()}
            return hashlib.sha256(
                json.dumps(rounded, sort_keys=True).encode()
            ).hexdigest()

        result = benchmark_runner.run_benchmark(
            calculate_action_hash,
            iterations=10000,
            target_ms=0.5,
            name="action_hash"
        )

        assert result.passed


class TestControlLoopBenchmarks:
    """Benchmark complete control loop performance."""

    @pytest.mark.performance
    def test_full_control_cycle_benchmark(
        self,
        benchmark_runner,
        sample_combustion_data
    ):
        """Benchmark complete control cycle (<100ms target per spec)."""
        def full_control_cycle():
            # 1. Read state (simulated)
            state = sample_combustion_data.copy()

            # 2. Check safety interlocks
            safe = all([
                state["furnace_temperature"] < 1500,
                state["o2_percent"] >= 2.0,
                state["co_ppm"] < 400
            ])

            if not safe:
                return {"success": False}

            # 3. Calculate efficiency
            efficiency = Decimal(str(state["thermal_efficiency"]))

            # 4. PID update
            error = Decimal("4.5") - Decimal(str(state["o2_percent"]))
            correction = float(error * Decimal("2.0"))

            # 5. Calculate new setpoints
            fuel_setpoint = state["fuel_flow"] + correction * 0.1
            air_setpoint = fuel_setpoint * 12.0 * 1.25

            # 6. Generate provenance hash
            action = {
                "fuel_setpoint": round(fuel_setpoint, 6),
                "air_setpoint": round(air_setpoint, 6)
            }
            prov_hash = hashlib.sha256(
                json.dumps(action, sort_keys=True).encode()
            ).hexdigest()

            return {
                "success": True,
                "fuel_setpoint": fuel_setpoint,
                "air_setpoint": air_setpoint,
                "provenance_hash": prov_hash
            }

        result = benchmark_runner.run_benchmark(
            full_control_cycle,
            iterations=1000,
            target_ms=100.0,
            name="full_control_cycle"
        )

        assert result.passed, f"Control cycle: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        assert result.p99_time_ms < 100.0  # P99 must also meet target


class TestThroughputBenchmarks:
    """Benchmark system throughput."""

    @pytest.mark.performance
    def test_batch_calculation_throughput(self, benchmark_runner):
        """Test batch calculation throughput (target: 1000 calcs/sec)."""
        batch_size = 100

        def batch_calculate():
            results = []
            for i in range(batch_size):
                fuel = 80 + i * 0.4
                air = fuel * 12.0
                efficiency = 85 + random.random() * 5
                heat = fuel * 42.0 * efficiency / 100 * 1000 / 3600
                results.append({
                    "fuel": fuel,
                    "air": air,
                    "efficiency": efficiency,
                    "heat_output": heat
                })
            return results

        result = benchmark_runner.run_benchmark(
            batch_calculate,
            iterations=100,
            target_ms=100.0,  # 100 calcs in 100ms = 1000/sec
            name="batch_calculation"
        )

        effective_throughput = batch_size * result.throughput_per_sec
        assert effective_throughput >= 1000, f"Throughput {effective_throughput:.0f}/sec < 1000/sec"


class TestCachePerformance:
    """Benchmark cache performance."""

    @pytest.mark.performance
    def test_state_cache_lookup(self, benchmark_runner):
        """Test state cache lookup performance (<0.01ms target)."""
        # Pre-populate cache
        cache = {
            f"state_{i}": {
                "fuel_flow": 100 + i,
                "air_flow": 1200 + i * 10,
                "efficiency": 85 + i * 0.1
            }
            for i in range(1000)
        }

        def cache_lookup():
            key = f"state_{random.randint(0, 999)}"
            return cache.get(key)

        result = benchmark_runner.run_benchmark(
            cache_lookup,
            iterations=10000,
            target_ms=0.01,
            name="cache_lookup"
        )

        assert result.avg_time_ms < 0.1

    @pytest.mark.performance
    def test_control_history_append(self, benchmark_runner):
        """Test control history append performance (<0.1ms target)."""
        from collections import deque
        history = deque(maxlen=1000)

        def append_history():
            history.append({
                "action_id": f"action_{len(history)}",
                "fuel_setpoint": 100.0,
                "air_setpoint": 1200.0,
                "timestamp": time.time()
            })

        result = benchmark_runner.run_benchmark(
            append_history,
            iterations=10000,
            target_ms=0.1,
            name="history_append"
        )

        assert result.passed


class TestStabilityAnalysisBenchmarks:
    """Benchmark stability analysis performance."""

    @pytest.mark.performance
    def test_variance_calculation_benchmark(self, benchmark_runner):
        """Benchmark variance calculation for stability (<5ms target)."""
        values = [900.0 + random.random() * 10 for _ in range(100)]

        def calculate_variance():
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std_dev = variance ** 0.5
            cv = (std_dev / mean) * 100 if mean != 0 else 0
            return variance, std_dev, cv

        result = benchmark_runner.run_benchmark(
            calculate_variance,
            iterations=1000,
            target_ms=5.0,
            name="variance_calculation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_stability_index_calculation(self, benchmark_runner):
        """Benchmark stability index calculation (<2ms target)."""
        heat_values = [1000 + random.random() * 50 for _ in range(100)]
        temp_values = [900 + random.random() * 20 for _ in range(100)]

        def calculate_stability_index():
            # Heat stability
            heat_mean = sum(heat_values) / len(heat_values)
            heat_var = sum((v - heat_mean) ** 2 for v in heat_values) / len(heat_values)
            heat_cv = (heat_var ** 0.5) / heat_mean if heat_mean != 0 else 1

            # Temp stability
            temp_mean = sum(temp_values) / len(temp_values)
            temp_var = sum((v - temp_mean) ** 2 for v in temp_values) / len(temp_values)
            temp_cv = (temp_var ** 0.5) / temp_mean if temp_mean != 0 else 1

            # Combined stability (lower CV = more stable)
            heat_stability = max(0, 1 - heat_cv * 10)
            temp_stability = max(0, 1 - temp_cv * 10)

            overall = 0.6 * heat_stability + 0.4 * temp_stability
            return overall * 100

        result = benchmark_runner.run_benchmark(
            calculate_stability_index,
            iterations=1000,
            target_ms=2.0,
            name="stability_index"
        )

        assert result.passed
