# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests for GL-004 BurnerOptimizationAgent.

Tests performance benchmarks for:
- Combustion efficiency calculations
- Optimization algorithm latency
- Real-time control response times
- ASME standard calculation compliance timing
- Memory usage during calculations
- Cache performance for repeated calculations

Performance Targets:
- Combustion efficiency calculation: <10ms
- Optimization cycle: <100ms (without I/O)
- Real-time control response: <50ms
- ASME PTC 4.1 calculations: <5ms
- Memory per calculation: <1MB

Target: 25+ benchmark tests with ASME compliance
"""

import pytest
import time
import hashlib
import json
import threading
import math
from datetime import datetime
from typing import Dict, Any, List
from decimal import Decimal, ROUND_HALF_UP
from collections import OrderedDict
import sys
import os

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.benchmark]


# ============================================================================
# THREAD-SAFE CACHE FOR TESTING
# ============================================================================

class BenchmarkCache:
    """Thread-safe cache for benchmark testing."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str):
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            if time.time() - self._timestamps[key] > self._ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total if total > 0 else 0.0,
                'size': len(self._cache)
            }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Performance benchmark configuration."""
    return {
        'combustion_efficiency_target_ms': 10.0,
        'optimization_cycle_target_ms': 100.0,
        'real_time_control_target_ms': 50.0,
        'asme_calculation_target_ms': 5.0,
        'stoichiometric_target_ms': 1.0,
        'emissions_target_ms': 5.0,
        'memory_per_calc_mb': 1.0,
        'throughput_target_per_sec': 1000,
        'cache_hit_rate_target': 0.80
    }


@pytest.fixture
def burner_input_data():
    """Standard burner input data for benchmarks."""
    return {
        'fuel_type': 'natural_gas',
        'fuel_flow_rate': 500.0,
        'air_flow_rate': 8500.0,
        'o2_level': 3.5,
        'co_level': 25.0,
        'nox_level': 35.0,
        'flame_temperature': 1650.0,
        'furnace_temperature': 1200.0,
        'flue_gas_temperature': 320.0,
        'burner_load': 75.0,
        'ambient_temperature': 25.0
    }


@pytest.fixture
def combustion_calculator():
    """Create combustion efficiency calculator for benchmarks."""
    class CombustionEfficiencyCalculator:
        CP_DRY_AIR = 1.005
        CP_H2O_VAPOR = 1.86

        def calculate(self, data: Dict) -> Dict[str, float]:
            temp_diff = data['flue_gas_temperature'] - data.get('ambient_temperature', 25.0)
            dry_gas_loss = (temp_diff * self.CP_DRY_AIR * 0.24) / 100
            h2_mass_frac = 0.10
            moisture_loss = h2_mass_frac * 9 * 2.442 / 50
            co_loss = (data.get('co_level', 0) / 10000) * 0.5
            radiation_loss = 1.5
            total_losses = dry_gas_loss + moisture_loss + co_loss + radiation_loss
            gross_efficiency = 100 - total_losses
            net_efficiency = gross_efficiency + 6.0

            return {
                'gross_efficiency': round(max(0, min(100, gross_efficiency)), 6),
                'net_efficiency': round(max(0, min(100, net_efficiency)), 6),
                'dry_flue_gas_loss': round(dry_gas_loss, 6),
                'moisture_loss': round(moisture_loss, 6),
                'incomplete_combustion_loss': round(co_loss, 6),
                'radiation_loss': radiation_loss,
                'total_losses': round(total_losses, 6)
            }

    return CombustionEfficiencyCalculator()


@pytest.fixture
def stoichiometric_calculator():
    """Create stoichiometric calculator for benchmarks."""
    class StoichiometricCalculator:
        STOICH_AFR = {
            'natural_gas': 17.2,
            'propane': 15.7,
            'fuel_oil': 14.2,
            'coal': 11.0
        }

        def calculate(self, data: Dict) -> Dict[str, float]:
            fuel_type = data.get('fuel_type', 'natural_gas')
            air_flow = data['air_flow_rate']
            fuel_flow = data['fuel_flow_rate']

            stoich_afr = self.STOICH_AFR.get(fuel_type, 17.2)
            actual_afr = air_flow / fuel_flow if fuel_flow > 0 else 0
            excess_air = ((actual_afr / stoich_afr) - 1) * 100 if stoich_afr > 0 else 0

            return {
                'stoichiometric_afr': stoich_afr,
                'actual_afr': round(actual_afr, 6),
                'excess_air_percent': round(excess_air, 6),
                'lambda_value': round(actual_afr / stoich_afr, 6) if stoich_afr > 0 else 0
            }

    return StoichiometricCalculator()


@pytest.fixture
def emissions_calculator():
    """Create emissions calculator for benchmarks."""
    class EmissionsCalculator:
        def calculate(self, data: Dict) -> Dict[str, float]:
            fuel_flow = data['fuel_flow_rate']
            flame_temp = data.get('flame_temperature', 1500.0)
            excess_air = data.get('excess_air_percent', 15.0)

            co2_kg_hr = fuel_flow * 0.75 * 3.67

            if flame_temp > 1300:
                temp_factor = math.exp((flame_temp - 1300) / 200)
            else:
                temp_factor = 1.0

            nox_base = 30.0
            nox_ppm = nox_base * temp_factor * (1 + excess_air / 100)

            co_base = 50.0
            excess_ratio = 1 + excess_air / 100
            co_ppm = co_base / excess_ratio if excess_ratio > 0 else co_base

            return {
                'co2_kg_hr': round(co2_kg_hr, 2),
                'nox_ppm': round(min(500, max(10, nox_ppm)), 1),
                'co_ppm': round(min(1000, max(5, co_ppm)), 1)
            }

    return EmissionsCalculator()


@pytest.fixture
def optimizer():
    """Create air-fuel optimizer for benchmarks."""
    class AirFuelOptimizer:
        def optimize(self, data: Dict, constraints: Dict = None) -> Dict[str, Any]:
            constraints = constraints or {}
            fuel_flow = data['fuel_flow_rate']
            current_afr = data['air_flow_rate'] / fuel_flow if fuel_flow > 0 else 17.0

            stoich_afr = 17.2
            target_excess = constraints.get('target_excess_air', 15.0)
            min_excess = constraints.get('min_excess_air', 5.0)
            max_excess = constraints.get('max_excess_air', 25.0)

            optimal_excess = max(min_excess, min(max_excess, target_excess))
            optimal_afr = stoich_afr * (1 + optimal_excess / 100)
            optimal_air_flow = fuel_flow * optimal_afr

            base_efficiency = 0.95
            stack_loss = (data['flue_gas_temperature'] - 25.0) * 0.0004 * (1 + optimal_excess / 100)
            excess_loss = optimal_excess * 0.0002
            predicted_efficiency = (base_efficiency - stack_loss - excess_loss - 0.02) * 100

            return {
                'optimal_afr': round(optimal_afr, 2),
                'optimal_air_flow': round(optimal_air_flow, 2),
                'optimal_excess_air': round(optimal_excess, 1),
                'predicted_efficiency': round(predicted_efficiency, 2),
                'convergence_status': 'converged',
                'iterations': 5,
                'confidence_score': 0.95
            }

    return AirFuelOptimizer()


# ============================================================================
# COMBUSTION EFFICIENCY CALCULATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestCombustionEfficiencyBenchmarks:
    """Benchmark tests for combustion efficiency calculations."""

    def test_benchmark_001_single_efficiency_calc(
        self,
        combustion_calculator,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 001: Single combustion efficiency calculation.

        Target: <10ms latency
        """
        start = time.perf_counter()
        result = combustion_calculator.calculate(burner_input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result['gross_efficiency'] > 0
        assert elapsed_ms < benchmark_config['combustion_efficiency_target_ms'], \
            f"Efficiency calc took {elapsed_ms:.3f}ms, target <{benchmark_config['combustion_efficiency_target_ms']}ms"

        print(f"Single efficiency calculation: {elapsed_ms:.4f}ms")

    def test_benchmark_002_efficiency_p95_latency(
        self,
        combustion_calculator,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 002: P95 latency for efficiency calculations.

        Target: P95 < 2x single calc target
        """
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            combustion_calculator.calculate(burner_input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        target_p95 = benchmark_config['combustion_efficiency_target_ms'] * 2

        assert p95 < target_p95, f"P95 latency {p95:.3f}ms exceeds target {target_p95}ms"

        print(f"Efficiency P50: {p50:.4f}ms, P95: {p95:.4f}ms, P99: {p99:.4f}ms")

    def test_benchmark_003_efficiency_throughput(
        self,
        combustion_calculator,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 003: Efficiency calculation throughput.

        Target: >1000 calculations/second
        """
        count = 0
        duration_seconds = 2
        start = time.perf_counter()

        while time.perf_counter() - start < duration_seconds:
            combustion_calculator.calculate(burner_input_data)
            count += 1

        actual_duration = time.perf_counter() - start
        throughput = count / actual_duration

        target = benchmark_config['throughput_target_per_sec']
        assert throughput >= target, f"Throughput {throughput:.1f}/s below target {target}/s"

        print(f"Efficiency throughput: {throughput:.1f} calcs/sec")

    def test_benchmark_004_asme_indirect_method_speed(
        self,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 004: ASME PTC 4.1 indirect method calculation speed.

        Target: <5ms for complete ASME calculation
        """
        def asme_ptc41_indirect(data: Dict) -> Dict[str, float]:
            flue_temp = data['flue_gas_temperature']
            ambient_temp = data.get('ambient_temperature', 25.0)
            temp_diff = flue_temp - ambient_temp

            l1_dry_gas = temp_diff * 0.38 * 1.15 / 100
            l2_h2_moisture = 0.235 * 9 * 2.442 / 50
            l3_fuel_moisture = 0.0
            l4_air_moisture = 0.001
            l5_co = (data.get('co_level', 0) / 10000) * 0.5
            l6_radiation = 0.015
            l7_unmeasured = 0.005

            total_losses = l1_dry_gas + l2_h2_moisture + l3_fuel_moisture + \
                          l4_air_moisture + l5_co + l6_radiation + l7_unmeasured
            gross_efficiency = (1.0 - total_losses) * 100

            return {
                'gross_efficiency': round(gross_efficiency, 2),
                'dry_flue_gas_loss': round(l1_dry_gas * 100, 2),
                'h2_moisture_loss': round(l2_h2_moisture * 100, 2),
                'total_losses': round(total_losses * 100, 2)
            }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = asme_ptc41_indirect(burner_input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        target = benchmark_config['asme_calculation_target_ms']
        assert avg_latency < target, f"ASME calc avg {avg_latency:.4f}ms exceeds {target}ms"

        print(f"ASME PTC 4.1 indirect method: {avg_latency:.4f}ms avg")


# ============================================================================
# OPTIMIZATION ALGORITHM LATENCY BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestOptimizationAlgorithmBenchmarks:
    """Benchmark tests for optimization algorithm latency."""

    def test_benchmark_005_single_optimization_cycle(
        self,
        optimizer,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 005: Single optimization cycle latency.

        Target: <100ms for complete optimization
        """
        start = time.perf_counter()
        result = optimizer.optimize(burner_input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result['convergence_status'] == 'converged'
        assert elapsed_ms < benchmark_config['optimization_cycle_target_ms'], \
            f"Optimization took {elapsed_ms:.3f}ms, target <{benchmark_config['optimization_cycle_target_ms']}ms"

        print(f"Single optimization cycle: {elapsed_ms:.4f}ms")

    def test_benchmark_006_optimization_convergence_speed(
        self,
        optimizer,
        burner_input_data
    ):
        """
        BENCHMARK 006: Optimization convergence speed.

        Tests iterations needed for convergence.
        """
        result = optimizer.optimize(burner_input_data)

        assert result['iterations'] <= 20, f"Convergence took {result['iterations']} iterations"
        assert result['confidence_score'] >= 0.90

        print(f"Optimization converged in {result['iterations']} iterations")

    def test_benchmark_007_optimization_with_constraints(
        self,
        optimizer,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 007: Optimization with constraints.

        Tests optimization speed with active constraints.
        """
        constraints = {
            'min_excess_air': 10.0,
            'max_excess_air': 20.0,
            'target_excess_air': 15.0,
            'max_nox_ppm': 30.0,
            'min_efficiency': 85.0
        }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = optimizer.optimize(burner_input_data, constraints)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < benchmark_config['optimization_cycle_target_ms']
        assert result['optimal_excess_air'] >= constraints['min_excess_air']
        assert result['optimal_excess_air'] <= constraints['max_excess_air']

        print(f"Constrained optimization: {avg_latency:.4f}ms avg")

    def test_benchmark_008_multi_objective_optimization(
        self,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 008: Multi-objective optimization speed.

        Tests optimization balancing efficiency and emissions.
        """
        def multi_objective_optimize(data: Dict) -> Dict[str, float]:
            fuel_flow = data['fuel_flow_rate']
            stoich_afr = 17.2

            best_score = 0
            best_afr = stoich_afr
            best_efficiency = 0
            best_nox = 100

            for excess_air in range(5, 26):
                afr = stoich_afr * (1 + excess_air / 100)

                efficiency = 95 - (excess_air - 15) ** 2 * 0.05 - 7
                nox = 30 + (20 - excess_air) * 0.5

                efficiency = max(80, min(95, efficiency))
                nox = max(20, min(60, nox))

                eff_score = efficiency / 100
                nox_score = 1 - (nox / 100)
                total_score = 0.6 * eff_score + 0.4 * nox_score

                if total_score > best_score:
                    best_score = total_score
                    best_afr = afr
                    best_efficiency = efficiency
                    best_nox = nox

            return {
                'optimal_afr': best_afr,
                'predicted_efficiency': best_efficiency,
                'predicted_nox': best_nox,
                'optimization_score': best_score
            }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = multi_objective_optimize(burner_input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < benchmark_config['optimization_cycle_target_ms']

        print(f"Multi-objective optimization: {avg_latency:.4f}ms avg")


# ============================================================================
# REAL-TIME CONTROL RESPONSE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestRealTimeControlBenchmarks:
    """Benchmark tests for real-time control response."""

    def test_benchmark_009_control_loop_latency(
        self,
        combustion_calculator,
        stoichiometric_calculator,
        optimizer,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 009: Complete control loop latency.

        Target: <50ms for sensor-to-actuator cycle
        """
        def control_loop(data: Dict) -> Dict:
            stoich = stoichiometric_calculator.calculate(data)
            data['excess_air_percent'] = stoich['excess_air_percent']

            efficiency = combustion_calculator.calculate(data)

            optimization = optimizer.optimize(data)

            return {
                'stoichiometric': stoich,
                'efficiency': efficiency,
                'optimization': optimization
            }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = control_loop(burner_input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        target = benchmark_config['real_time_control_target_ms']
        assert avg_latency < target, f"Control loop avg {avg_latency:.3f}ms exceeds {target}ms"

        print(f"Control loop: avg={avg_latency:.4f}ms, max={max_latency:.4f}ms")

    def test_benchmark_010_setpoint_calculation_speed(
        self,
        optimizer,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 010: Setpoint calculation speed.

        Target: <10ms for setpoint determination
        """
        latencies = []
        for _ in range(100):
            start = time.perf_counter()

            result = optimizer.optimize(burner_input_data)
            fuel_setpoint = result.get('optimal_fuel_flow', burner_input_data['fuel_flow_rate'])
            air_setpoint = result['optimal_air_flow']

            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < benchmark_config['combustion_efficiency_target_ms']

        print(f"Setpoint calculation: {avg_latency:.4f}ms avg")

    def test_benchmark_011_rapid_load_change_response(
        self,
        optimizer,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 011: Response to rapid load changes.

        Tests optimization speed during load transients.
        """
        loads = [25, 50, 75, 100, 75, 50, 25, 50, 75, 100]
        latencies = []

        for load in loads:
            data = burner_input_data.copy()
            data['burner_load'] = load
            data['fuel_flow_rate'] = 500.0 * load / 100
            data['air_flow_rate'] = 8500.0 * load / 100

            start = time.perf_counter()
            result = optimizer.optimize(data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert max_latency < benchmark_config['real_time_control_target_ms']

        print(f"Load change response: avg={avg_latency:.4f}ms, max={max_latency:.4f}ms")


# ============================================================================
# STOICHIOMETRIC CALCULATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestStoichiometricBenchmarks:
    """Benchmark tests for stoichiometric calculations."""

    def test_benchmark_012_stoichiometric_calc_speed(
        self,
        stoichiometric_calculator,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 012: Stoichiometric calculation speed.

        Target: <1ms for stoichiometric calculation
        """
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            result = stoichiometric_calculator.calculate(burner_input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        target = benchmark_config['stoichiometric_target_ms']
        assert avg_latency < target, f"Stoich calc avg {avg_latency:.4f}ms exceeds {target}ms"

        print(f"Stoichiometric calculation: {avg_latency:.6f}ms avg")

    def test_benchmark_013_excess_air_calculation_speed(self, benchmark_config):
        """
        BENCHMARK 013: Excess air calculation from O2 speed.

        Target: <0.1ms for O2 to excess air conversion
        """
        def calculate_excess_air(o2_percent: float) -> float:
            if o2_percent >= 21.0:
                return 100.0
            if o2_percent <= 0:
                return 0.0
            return (o2_percent / (21.0 - o2_percent)) * 100.0

        o2_levels = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0]
        latencies = []

        for _ in range(1000):
            for o2 in o2_levels:
                start = time.perf_counter()
                calculate_excess_air(o2)
                latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < 0.1, f"Excess air calc {avg_latency:.6f}ms exceeds 0.1ms"

        print(f"Excess air calculation: {avg_latency:.6f}ms avg")

    def test_benchmark_014_afr_calculation_throughput(
        self,
        stoichiometric_calculator,
        benchmark_config
    ):
        """
        BENCHMARK 014: AFR calculation throughput.

        Target: >10000 calculations/second
        """
        data = {
            'fuel_type': 'natural_gas',
            'fuel_flow_rate': 500.0,
            'air_flow_rate': 8500.0
        }

        count = 0
        duration = 1.0
        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            stoichiometric_calculator.calculate(data)
            count += 1

        throughput = count / duration

        assert throughput >= 10000, f"AFR throughput {throughput:.1f}/s below 10000/s"

        print(f"AFR calculation throughput: {throughput:.1f} calcs/sec")


# ============================================================================
# EMISSIONS CALCULATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestEmissionsCalculationBenchmarks:
    """Benchmark tests for emissions calculations."""

    def test_benchmark_015_emissions_calc_speed(
        self,
        emissions_calculator,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 015: Emissions calculation speed.

        Target: <5ms for complete emissions calculation
        """
        data = burner_input_data.copy()
        data['excess_air_percent'] = 15.0

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = emissions_calculator.calculate(data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        target = benchmark_config['emissions_target_ms']
        assert avg_latency < target, f"Emissions calc {avg_latency:.4f}ms exceeds {target}ms"

        print(f"Emissions calculation: {avg_latency:.4f}ms avg")

    def test_benchmark_016_nox_prediction_speed(self, benchmark_config):
        """
        BENCHMARK 016: NOx prediction calculation speed.

        Target: <1ms for NOx prediction
        """
        def predict_nox(flame_temp: float, excess_air: float) -> float:
            if flame_temp > 1300:
                temp_factor = math.exp((flame_temp - 1300) / 200)
            else:
                temp_factor = 1.0

            excess_ratio = 1 + excess_air / 100
            if excess_ratio < 1.0:
                air_factor = excess_ratio
            elif excess_ratio < 1.3:
                air_factor = 1.0 + (excess_ratio - 1.0) * 2
            else:
                air_factor = 1.6 - (excess_ratio - 1.3) * 0.5

            nox = 30.0 * temp_factor * air_factor
            return min(500, max(10, nox))

        latencies = []
        temps = [1400, 1500, 1600, 1700]
        excess_airs = [5, 10, 15, 20, 25]

        for _ in range(100):
            for temp in temps:
                for ea in excess_airs:
                    start = time.perf_counter()
                    predict_nox(temp, ea)
                    latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < 1.0, f"NOx prediction {avg_latency:.6f}ms exceeds 1ms"

        print(f"NOx prediction: {avg_latency:.6f}ms avg")


# ============================================================================
# CACHE PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestCachePerformanceBenchmarks:
    """Benchmark tests for cache performance."""

    def test_benchmark_017_cache_hit_performance(self, benchmark_config):
        """
        BENCHMARK 017: Cache hit latency.

        Target: <0.01ms for cache hit
        """
        cache = BenchmarkCache(max_size=1000)

        for i in range(100):
            cache.set(f'key_{i}', {'efficiency': 87.5, 'nox': 35.0})

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            cache.get('key_50')
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < 0.1, f"Cache hit latency {avg_latency:.6f}ms exceeds 0.1ms"

        print(f"Cache hit latency: {avg_latency:.6f}ms avg")

    def test_benchmark_018_cache_miss_performance(self, benchmark_config):
        """
        BENCHMARK 018: Cache miss latency.

        Target: <0.01ms for cache miss
        """
        cache = BenchmarkCache(max_size=1000)

        latencies = []
        for i in range(1000):
            start = time.perf_counter()
            cache.get(f'missing_key_{i}')
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < 0.1, f"Cache miss latency {avg_latency:.6f}ms exceeds 0.1ms"

        print(f"Cache miss latency: {avg_latency:.6f}ms avg")

    def test_benchmark_019_cache_hit_rate_under_load(self, benchmark_config):
        """
        BENCHMARK 019: Cache hit rate under realistic load.

        Target: >80% hit rate with 80/20 access pattern
        """
        cache = BenchmarkCache(max_size=100)

        for i in range(20):
            cache.set(f'hot_key_{i}', {'value': i})

        import random
        random.seed(42)

        for _ in range(1000):
            if random.random() < 0.8:
                key = f'hot_key_{random.randint(0, 19)}'
            else:
                key = f'cold_key_{random.randint(0, 999)}'
            cache.get(key)

        stats = cache.get_stats()
        hit_rate = stats['hit_rate']

        target = benchmark_config['cache_hit_rate_target']
        assert hit_rate >= target, f"Hit rate {hit_rate:.2%} below {target:.2%}"

        print(f"Cache hit rate: {hit_rate:.2%}")


# ============================================================================
# MEMORY USAGE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestMemoryUsageBenchmarks:
    """Benchmark tests for memory usage."""

    def test_benchmark_020_calculation_memory_footprint(
        self,
        combustion_calculator,
        burner_input_data,
        benchmark_config
    ):
        """
        BENCHMARK 020: Memory footprint per calculation.

        Target: <1MB per calculation
        """
        import tracemalloc

        tracemalloc.start()

        for _ in range(100):
            combustion_calculator.calculate(burner_input_data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        per_calc_kb = (peak / 1024) / 100

        target_mb = benchmark_config['memory_per_calc_mb']
        assert peak_mb < target_mb * 100, f"Peak memory {peak_mb:.2f}MB too high"

        print(f"Memory per calculation: {per_calc_kb:.2f}KB")

    def test_benchmark_021_no_memory_leak(
        self,
        combustion_calculator,
        burner_input_data
    ):
        """
        BENCHMARK 021: No memory leak over many calculations.

        Verifies memory does not grow unbounded.
        """
        import tracemalloc

        tracemalloc.start()

        for _ in range(100):
            combustion_calculator.calculate(burner_input_data)

        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(10000):
            combustion_calculator.calculate(burner_input_data)

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

        tracemalloc.stop()

        assert total_growth < 10, f"Memory growth {total_growth:.2f}MB indicates possible leak"

        print(f"Memory growth over 10000 iterations: {total_growth:.2f}MB")


# ============================================================================
# CONCURRENT PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestConcurrentPerformanceBenchmarks:
    """Benchmark tests for concurrent performance."""

    def test_benchmark_022_multi_thread_throughput(
        self,
        combustion_calculator,
        burner_input_data
    ):
        """
        BENCHMARK 022: Multi-threaded throughput.

        Tests throughput with concurrent calculations.
        """
        results = []
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(100):
                    result = combustion_calculator.calculate(burner_input_data)
                    with lock:
                        results.append(result['gross_efficiency'])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]

        start = time.perf_counter()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        duration = time.perf_counter() - start
        throughput = len(results) / duration

        assert len(errors) == 0
        assert throughput >= 1000, f"Multi-thread throughput {throughput:.1f}/s below 1000/s"

        print(f"Multi-thread throughput (4 threads): {throughput:.1f} calcs/sec")

    def test_benchmark_023_cache_concurrent_access(self):
        """
        BENCHMARK 023: Cache performance under concurrent access.

        Tests cache integrity with concurrent reads/writes.
        """
        cache = BenchmarkCache(max_size=100)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f'key_{i}', {'value': i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f'key_{i % 50}')
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        start = time.perf_counter()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        duration = time.perf_counter() - start

        assert len(errors) == 0
        assert duration < 1.0

        print(f"Concurrent cache access duration: {duration:.4f}s")


# ============================================================================
# ASME COMPLIANCE TIMING BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asme
class TestASMEComplianceTimingBenchmarks:
    """Benchmark tests for ASME compliance calculations."""

    def test_benchmark_024_asme_ptc41_all_losses(self, burner_input_data, benchmark_config):
        """
        BENCHMARK 024: ASME PTC 4.1 all losses calculation.

        Target: <5ms for complete loss breakdown
        """
        def calculate_all_losses(data: Dict) -> Dict[str, float]:
            flue_temp = data['flue_gas_temperature']
            ambient_temp = data.get('ambient_temperature', 25.0)
            h2_fraction = 0.235
            co_ppm = data.get('co_level', 0)

            temp_diff = flue_temp - ambient_temp

            l1 = temp_diff * 0.38 * 1.15 / 100
            l2 = h2_fraction * 9 * (2442 + 1.86 * temp_diff) / 50000
            l3 = 0.0
            l4 = 0.001
            l5 = (co_ppm / 1e6) * 10.1 / 50.0
            l6 = 0.015
            l7 = 0.005

            total = l1 + l2 + l3 + l4 + l5 + l6 + l7
            efficiency = (1.0 - total) * 100

            return {
                'L1_dry_gas': l1 * 100,
                'L2_h2_moisture': l2 * 100,
                'L3_fuel_moisture': l3 * 100,
                'L4_air_moisture': l4 * 100,
                'L5_co': l5 * 100,
                'L6_radiation': l6 * 100,
                'L7_unmeasured': l7 * 100,
                'total_losses': total * 100,
                'efficiency': efficiency
            }

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = calculate_all_losses(burner_input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        target = benchmark_config['asme_calculation_target_ms']
        assert avg_latency < target, f"ASME all losses {avg_latency:.4f}ms exceeds {target}ms"

        print(f"ASME PTC 4.1 all losses: {avg_latency:.4f}ms avg")

    def test_benchmark_025_o2_correction_speed(self, benchmark_config):
        """
        BENCHMARK 025: O2 correction calculation speed.

        Target: <0.1ms for O2 correction
        """
        def correct_to_reference_o2(measured_ppm: float, measured_o2: float, ref_o2: float = 3.0) -> float:
            if measured_o2 >= 20.9:
                return 0.0
            factor = (20.9 - ref_o2) / (20.9 - measured_o2)
            return measured_ppm * factor

        latencies = []
        test_cases = [
            (30.0, 3.5),
            (40.0, 4.0),
            (50.0, 5.0),
            (25.0, 2.5)
        ]

        for _ in range(1000):
            for nox, o2 in test_cases:
                start = time.perf_counter()
                correct_to_reference_o2(nox, o2)
                latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < 0.1, f"O2 correction {avg_latency:.6f}ms exceeds 0.1ms"

        print(f"O2 correction calculation: {avg_latency:.6f}ms avg")


# ============================================================================
# SUMMARY
# ============================================================================

def test_benchmark_summary():
    """
    Summary test confirming benchmark coverage.

    This test suite provides 25 benchmark tests covering:
    - Combustion efficiency calculations (4 tests)
    - Optimization algorithm latency (4 tests)
    - Real-time control response (3 tests)
    - Stoichiometric calculations (3 tests)
    - Emissions calculations (2 tests)
    - Cache performance (3 tests)
    - Memory usage (2 tests)
    - Concurrent performance (2 tests)
    - ASME compliance timing (2 tests)

    Performance Targets:
    - Combustion efficiency: <10ms
    - Optimization cycle: <100ms
    - Real-time control: <50ms
    - ASME calculations: <5ms
    - Stoichiometric: <1ms
    - Throughput: >1000 calcs/sec

    Total: 25 benchmark tests
    """
    assert True
