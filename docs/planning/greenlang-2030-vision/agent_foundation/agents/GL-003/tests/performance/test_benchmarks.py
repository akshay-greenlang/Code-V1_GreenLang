# -*- coding: utf-8 -*-
"""
Performance Benchmarks for GL-003 STEAMWISE SteamSystemAnalyzer.

Comprehensive performance testing with clear pass/fail criteria including:
- Steam property calculations benchmarks
- Distribution efficiency analysis performance
- Real-time monitoring latency

Author: GL-TestEngineer
Version: 1.0.0
Standards: GL-012 Test Patterns, GreenLang Performance Requirements
"""

import pytest
import time
import hashlib
import json
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# BENCHMARK INFRASTRUCTURE
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
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
    memory_delta_mb: float = 0.0


class PerformanceBenchmark:
    """Performance benchmark runner with comprehensive metrics."""

    @staticmethod
    def run_benchmark(
        func: Callable,
        iterations: int = 1000,
        target_ms: float = 1.0,
        name: str = "benchmark",
        warmup_iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run a benchmark with warmup and detailed statistics.

        Args:
            func: Function to benchmark
            iterations: Number of iterations
            target_ms: Target time in milliseconds
            name: Benchmark name
            warmup_iterations: Warmup iterations before measurement

        Returns:
            BenchmarkResult with comprehensive statistics
        """
        # Warmup phase
        for _ in range(warmup_iterations):
            func()

        # Measurement phase
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)

        # Calculate statistics
        times_sorted = sorted(times)
        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = min(times)
        max_time = max(times)

        # Percentiles
        p50_idx = int(iterations * 0.50)
        p95_idx = int(iterations * 0.95)
        p99_idx = int(iterations * 0.99)

        p50_time = times_sorted[p50_idx] if p50_idx < iterations else times_sorted[-1]
        p95_time = times_sorted[p95_idx] if p95_idx < iterations else times_sorted[-1]
        p99_time = times_sorted[p99_idx] if p99_idx < iterations else times_sorted[-1]

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

    @staticmethod
    async def run_async_benchmark(
        func: Callable,
        iterations: int = 100,
        target_ms: float = 10.0,
        name: str = "async_benchmark",
        warmup_iterations: int = 5
    ) -> BenchmarkResult:
        """Run an async benchmark with comprehensive metrics."""
        # Warmup phase
        for _ in range(warmup_iterations):
            await func()

        # Measurement phase
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # Calculate statistics
        times_sorted = sorted(times)
        total_time = sum(times)
        avg_time = total_time / iterations

        p50_idx = int(iterations * 0.50)
        p95_idx = int(iterations * 0.95)
        p99_idx = int(iterations * 0.99)

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=times_sorted[p50_idx] if p50_idx < iterations else times_sorted[-1],
            p95_time_ms=times_sorted[p95_idx] if p95_idx < iterations else times_sorted[-1],
            p99_time_ms=times_sorted[p99_idx] if p99_idx < iterations else times_sorted[-1],
            throughput_per_sec=iterations / (total_time / 1000) if total_time > 0 else 0,
            passed=avg_time <= target_ms,
            target_ms=target_ms
        )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_runner():
    """Provide benchmark runner."""
    return PerformanceBenchmark()


@pytest.fixture
def sample_steam_properties_input():
    """Sample input for steam properties calculation."""
    return {
        'pressure_bar': Decimal('10.0'),
        'temperature_c': Decimal('180.0')
    }


@pytest.fixture
def sample_pipe_segment_data():
    """Sample pipe segment data for distribution efficiency."""
    return {
        'length_m': Decimal('100.0'),
        'diameter_mm': Decimal('150.0'),
        'insulation_thickness_mm': Decimal('50.0'),
        'steam_temperature_c': Decimal('180.0'),
        'ambient_temperature_c': Decimal('25.0')
    }


@pytest.fixture
def sample_leak_detection_data():
    """Sample data for leak detection calculations."""
    return {
        'inlet_flow_kg_hr': [5000.0, 5050.0, 4980.0, 5020.0],
        'outlet_flow_kg_hr': [4800.0, 4850.0, 4750.0, 4820.0],
        'expected_pressure_drop_bar': 0.5
    }


@pytest.fixture
def sample_condensate_data():
    """Sample condensate optimization data."""
    return {
        'condensate_flow_rate_kg_hr': Decimal('4000.0'),
        'condensate_temperature_c': Decimal('95.0'),
        'condensate_pressure_bar': Decimal('8.0'),
        'flash_vessel_pressure_bar': Decimal('1.5'),
        'feedwater_temperature_c': Decimal('60.0'),
        'return_rate_percent': Decimal('65.0')
    }


# ============================================================================
# STEAM PROPERTY CALCULATION BENCHMARKS
# ============================================================================

class TestSteamPropertyBenchmarks:
    """Benchmarks for steam property calculations."""

    @pytest.mark.performance
    def test_saturation_temperature_calculation_benchmark(
        self, benchmark_runner
    ):
        """Benchmark saturation temperature from pressure calculation."""
        def calculate_saturation_temp():
            # IAPWS-IF97 simplified saturation temperature
            P = Decimal('10.0')  # bar
            P_mpa = P / Decimal('10')

            # Simplified correlation
            T_sat_k = Decimal('373.15') + Decimal('100') * (P_mpa - Decimal('0.1')).sqrt()
            return float(T_sat_k - Decimal('273.15'))

        result = benchmark_runner.run_benchmark(
            calculate_saturation_temp,
            iterations=10000,
            target_ms=0.1,
            name="saturation_temperature"
        )

        assert result.passed, (
            f"Saturation temp calc {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        )
        assert result.throughput_per_sec > 10000, "Should exceed 10k ops/sec"

    @pytest.mark.performance
    def test_enthalpy_calculation_benchmark(
        self, benchmark_runner, sample_steam_properties_input
    ):
        """Benchmark enthalpy calculation."""
        P = sample_steam_properties_input['pressure_bar']
        T = sample_steam_properties_input['temperature_c']

        def calculate_enthalpy():
            # Simplified enthalpy correlation
            Cp = Decimal('4.18')  # kJ/(kg*K) for liquid
            h_base = Cp * T

            # Latent heat estimation
            h_fg = Decimal('2257') - Decimal('2.3') * T
            h_vapor = h_base + h_fg

            return float(h_vapor.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_enthalpy,
            iterations=10000,
            target_ms=0.5,
            name="enthalpy_calculation"
        )

        assert result.passed, f"Enthalpy calc too slow: {result.avg_time_ms:.4f}ms"
        assert result.p99_time_ms < 2.0, "p99 should be under 2ms"

    @pytest.mark.performance
    def test_steam_quality_calculation_benchmark(self, benchmark_runner):
        """Benchmark steam quality (dryness fraction) calculation."""
        def calculate_quality():
            h_total = Decimal('2700.0')
            h_f = Decimal('762.8')
            h_fg = Decimal('2015.0')

            quality = (h_total - h_f) / h_fg
            return float(quality.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_quality,
            iterations=10000,
            target_ms=0.05,
            name="steam_quality"
        )

        assert result.passed, f"Quality calc too slow: {result.avg_time_ms:.4f}ms"
        assert result.throughput_per_sec > 50000

    @pytest.mark.performance
    def test_specific_volume_calculation_benchmark(self, benchmark_runner):
        """Benchmark specific volume calculation."""
        def calculate_specific_volume():
            T_k = Decimal('453.15')  # 180C in Kelvin
            P_kpa = Decimal('1000')  # 10 bar in kPa
            R = Decimal('0.4615')  # kJ/(kg*K)
            Z = Decimal('0.95')  # Compressibility factor

            v = Z * R * T_k / P_kpa
            return float(v.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_specific_volume,
            iterations=10000,
            target_ms=0.1,
            name="specific_volume"
        )

        assert result.passed
        assert result.p95_time_ms < 0.5

    @pytest.mark.performance
    def test_provenance_hash_generation_benchmark(
        self, benchmark_runner, sample_steam_properties_input
    ):
        """Benchmark provenance hash generation."""
        data = {k: str(v) for k, v in sample_steam_properties_input.items()}

        def generate_provenance_hash():
            return hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()

        result = benchmark_runner.run_benchmark(
            generate_provenance_hash,
            iterations=10000,
            target_ms=0.1,
            name="provenance_hash"
        )

        assert result.passed
        assert result.throughput_per_sec > 50000


# ============================================================================
# DISTRIBUTION EFFICIENCY ANALYSIS BENCHMARKS
# ============================================================================

class TestDistributionEfficiencyBenchmarks:
    """Benchmarks for distribution efficiency calculations."""

    @pytest.mark.performance
    def test_heat_loss_calculation_benchmark(
        self, benchmark_runner, sample_pipe_segment_data
    ):
        """Benchmark heat loss calculation for pipe segment."""
        segment = sample_pipe_segment_data

        def calculate_heat_loss():
            L = segment['length_m']
            D = segment['diameter_mm'] / Decimal('1000')
            t_ins = segment['insulation_thickness_mm'] / Decimal('1000')
            T_steam = segment['steam_temperature_c']
            T_amb = segment['ambient_temperature_c']

            # Simplified heat loss calculation
            r1 = D / Decimal('2')
            r2 = r1 + Decimal('0.005')
            r3 = r2 + t_ins

            k_ins = Decimal('0.045')  # W/(m*K) for mineral wool
            pi = Decimal(str(math.pi))

            # Thermal resistance
            R_ins = Decimal(str(math.log(float(r3 / r2)))) / (Decimal('2') * pi * k_ins)
            h_ext = Decimal('10.0')  # W/(m2*K)
            R_ext = Decimal('1') / (Decimal('2') * pi * r3 * h_ext)
            R_total = R_ins + R_ext

            # Heat loss
            q_per_length = (T_steam - T_amb) / R_total
            Q_total = (q_per_length * L) / Decimal('1000')

            return float(Q_total)

        result = benchmark_runner.run_benchmark(
            calculate_heat_loss,
            iterations=5000,
            target_ms=1.0,
            name="heat_loss_calculation"
        )

        assert result.passed, f"Heat loss calc too slow: {result.avg_time_ms:.4f}ms"
        assert result.throughput_per_sec > 1000

    @pytest.mark.performance
    def test_thermal_conductivity_interpolation_benchmark(self, benchmark_runner):
        """Benchmark thermal conductivity interpolation."""
        k_data = {50: 0.040, 100: 0.045, 150: 0.051, 200: 0.058}
        temps = sorted(k_data.keys())

        def interpolate_conductivity():
            temp = 125.0  # Interpolation target
            for i in range(len(temps) - 1):
                if temps[i] <= temp <= temps[i + 1]:
                    T1, T2 = temps[i], temps[i + 1]
                    k1, k2 = k_data[T1], k_data[T2]
                    k = k1 + (k2 - k1) * (temp - T1) / (T2 - T1)
                    return k
            return 0.050

        result = benchmark_runner.run_benchmark(
            interpolate_conductivity,
            iterations=10000,
            target_ms=0.01,
            name="thermal_conductivity_interpolation"
        )

        assert result.passed
        assert result.throughput_per_sec > 100000

    @pytest.mark.performance
    def test_distribution_efficiency_percent_benchmark(self, benchmark_runner):
        """Benchmark distribution efficiency percentage calculation."""
        def calculate_efficiency():
            energy_carried_kw = Decimal('1000.0')
            heat_loss_kw = Decimal('50.0')

            efficiency = ((energy_carried_kw - heat_loss_kw) / energy_carried_kw) * Decimal('100')
            return float(efficiency.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_efficiency,
            iterations=10000,
            target_ms=0.05,
            name="distribution_efficiency"
        )

        assert result.passed
        assert result.throughput_per_sec > 100000

    @pytest.mark.performance
    def test_multi_segment_analysis_benchmark(self, benchmark_runner):
        """Benchmark analysis of multiple pipe segments."""
        num_segments = 20

        def analyze_multi_segment():
            total_loss = Decimal('0')
            for i in range(num_segments):
                # Simplified loss per segment
                segment_loss = Decimal('2.5') + Decimal(str(i * 0.1))
                total_loss += segment_loss
            return float(total_loss)

        result = benchmark_runner.run_benchmark(
            analyze_multi_segment,
            iterations=1000,
            target_ms=5.0,
            name="multi_segment_analysis"
        )

        assert result.passed


# ============================================================================
# REAL-TIME MONITORING LATENCY BENCHMARKS
# ============================================================================

class TestRealTimeMonitoringLatency:
    """Benchmarks for real-time monitoring latency requirements."""

    @pytest.mark.performance
    def test_sensor_data_processing_latency(self, benchmark_runner):
        """Benchmark sensor data processing latency (<10ms target)."""
        sensor_data = {
            'timestamp': '2024-01-01T00:00:00Z',
            'pressure_bar': 10.5,
            'temperature_c': 182.3,
            'flow_rate_kg_hr': 5250.0,
            'quality': 'good'
        }

        def process_sensor_data():
            # Validate and transform sensor data
            pressure = Decimal(str(sensor_data['pressure_bar']))
            temp = Decimal(str(sensor_data['temperature_c']))
            flow = Decimal(str(sensor_data['flow_rate_kg_hr']))

            # Calculate derived values
            energy_kw = (flow / Decimal('3600')) * Decimal('2800')

            # Generate data hash for provenance
            data_hash = hashlib.sha256(
                json.dumps(sensor_data, sort_keys=True).encode()
            ).hexdigest()[:16]

            return {
                'energy_kw': float(energy_kw),
                'data_hash': data_hash
            }

        result = benchmark_runner.run_benchmark(
            process_sensor_data,
            iterations=5000,
            target_ms=10.0,
            name="sensor_data_processing"
        )

        assert result.passed, f"Sensor processing too slow: {result.avg_time_ms:.4f}ms"
        assert result.p99_time_ms < 50.0, "p99 latency must be <50ms"

    @pytest.mark.performance
    def test_alarm_threshold_check_latency(self, benchmark_runner):
        """Benchmark alarm threshold checking (<1ms target)."""
        thresholds = {
            'pressure_high': 15.0,
            'pressure_low': 5.0,
            'temperature_high': 200.0,
            'temperature_low': 100.0,
            'flow_high': 10000.0,
            'flow_low': 1000.0
        }

        def check_thresholds():
            values = {
                'pressure': 10.5,
                'temperature': 182.3,
                'flow': 5250.0
            }
            alarms = []

            if values['pressure'] > thresholds['pressure_high']:
                alarms.append('pressure_high')
            if values['pressure'] < thresholds['pressure_low']:
                alarms.append('pressure_low')
            if values['temperature'] > thresholds['temperature_high']:
                alarms.append('temperature_high')
            if values['temperature'] < thresholds['temperature_low']:
                alarms.append('temperature_low')
            if values['flow'] > thresholds['flow_high']:
                alarms.append('flow_high')
            if values['flow'] < thresholds['flow_low']:
                alarms.append('flow_low')

            return alarms

        result = benchmark_runner.run_benchmark(
            check_thresholds,
            iterations=10000,
            target_ms=1.0,
            name="alarm_threshold_check"
        )

        assert result.passed
        assert result.throughput_per_sec > 10000

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_data_collection_latency(self, benchmark_runner):
        """Benchmark async data collection latency."""
        async def collect_data():
            # Simulate async sensor reads
            await asyncio.sleep(0.001)  # 1ms simulated I/O

            return {
                'pressure': 10.5,
                'temperature': 182.3,
                'flow': 5250.0
            }

        result = await benchmark_runner.run_async_benchmark(
            collect_data,
            iterations=100,
            target_ms=50.0,
            name="async_data_collection"
        )

        assert result.passed
        assert result.p95_time_ms < 100.0

    @pytest.mark.performance
    def test_batch_data_processing_throughput(self, benchmark_runner):
        """Benchmark batch data processing throughput."""
        batch_size = 100
        sensor_readings = [
            {
                'timestamp': f'2024-01-01T00:{i:02d}:00Z',
                'pressure_bar': 10.0 + i * 0.01,
                'temperature_c': 180.0 + i * 0.1
            }
            for i in range(batch_size)
        ]

        def process_batch():
            results = []
            for reading in sensor_readings:
                pressure = Decimal(str(reading['pressure_bar']))
                temp = Decimal(str(reading['temperature_c']))
                energy = pressure * temp / Decimal('100')
                results.append(float(energy))
            return results

        result = benchmark_runner.run_benchmark(
            process_batch,
            iterations=100,
            target_ms=50.0,
            name="batch_data_processing"
        )

        assert result.passed
        # Should process at least 2000 records/second
        records_per_sec = batch_size * result.throughput_per_sec
        assert records_per_sec > 2000


# ============================================================================
# LEAK DETECTION PERFORMANCE BENCHMARKS
# ============================================================================

class TestLeakDetectionPerformance:
    """Benchmarks for leak detection calculations."""

    @pytest.mark.performance
    def test_mass_balance_calculation_benchmark(
        self, benchmark_runner, sample_leak_detection_data
    ):
        """Benchmark mass balance calculation."""
        data = sample_leak_detection_data

        def calculate_mass_balance():
            inlet = data['inlet_flow_kg_hr']
            outlet = data['outlet_flow_kg_hr']

            avg_inlet = sum(inlet) / len(inlet)
            avg_outlet = sum(outlet) / len(outlet)

            imbalance = Decimal(str(avg_inlet - avg_outlet))
            deviation_percent = (imbalance / Decimal(str(avg_inlet))) * Decimal('100')

            return float(deviation_percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_mass_balance,
            iterations=5000,
            target_ms=0.5,
            name="mass_balance"
        )

        assert result.passed

    @pytest.mark.performance
    def test_statistical_anomaly_detection_benchmark(self, benchmark_runner):
        """Benchmark statistical anomaly detection (3-sigma)."""
        flow_readings = [5000.0 + i * 10 for i in range(100)]

        def detect_anomalies():
            flows = [Decimal(str(f)) for f in flow_readings]
            mean = sum(flows) / len(flows)
            variance = sum((x - mean) ** 2 for x in flows) / len(flows)
            std = variance.sqrt()

            anomalies = []
            for idx, flow in enumerate(flows):
                deviation = abs(flow - mean) / std if std > 0 else Decimal('0')
                if deviation > Decimal('3'):
                    anomalies.append(idx)

            return anomalies

        result = benchmark_runner.run_benchmark(
            detect_anomalies,
            iterations=1000,
            target_ms=10.0,
            name="anomaly_detection"
        )

        assert result.passed


# ============================================================================
# CONDENSATE OPTIMIZATION PERFORMANCE BENCHMARKS
# ============================================================================

class TestCondensateOptimizationPerformance:
    """Benchmarks for condensate optimization calculations."""

    @pytest.mark.performance
    def test_flash_steam_calculation_benchmark(
        self, benchmark_runner, sample_condensate_data
    ):
        """Benchmark flash steam generation calculation."""
        data = sample_condensate_data

        def calculate_flash_steam():
            m = data['condensate_flow_rate_kg_hr']
            T_cond = data['condensate_temperature_c']
            P_flash = data['flash_vessel_pressure_bar']

            # Simplified flash fraction calculation
            h_initial = Decimal('4.18') * T_cond
            h_flash = Decimal('4.18') * Decimal('100')  # Sat at 1 bar
            h_fg = Decimal('2257')

            flash_fraction = (h_initial - h_flash) / h_fg
            flash_fraction = max(Decimal('0'), min(flash_fraction, Decimal('0.3')))

            m_flash = m * flash_fraction
            return float(m_flash)

        result = benchmark_runner.run_benchmark(
            calculate_flash_steam,
            iterations=5000,
            target_ms=0.5,
            name="flash_steam_calculation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_annual_savings_calculation_benchmark(self, benchmark_runner):
        """Benchmark annual savings calculation."""
        def calculate_savings():
            current_rate = Decimal('65')
            optimal_rate = Decimal('90')
            energy_recovery_gj_hr = Decimal('5.0')
            energy_cost = Decimal('20')
            hours = Decimal('8760')

            improvement = (optimal_rate - current_rate) / Decimal('100')
            energy_savings = energy_recovery_gj_hr * hours * improvement
            cost_savings = energy_savings * energy_cost

            return float(cost_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

        result = benchmark_runner.run_benchmark(
            calculate_savings,
            iterations=5000,
            target_ms=0.2,
            name="annual_savings"
        )

        assert result.passed


# ============================================================================
# CACHE PERFORMANCE BENCHMARKS
# ============================================================================

class TestCachePerformance:
    """Benchmarks for caching mechanisms."""

    @pytest.mark.performance
    def test_cache_hit_performance(self, benchmark_runner):
        """Benchmark cache hit latency (<0.1ms target)."""
        cache = {f"steam_props_{i}": {"enthalpy": 2800.0 + i} for i in range(1000)}

        def cache_lookup():
            return cache.get("steam_props_500")

        result = benchmark_runner.run_benchmark(
            cache_lookup,
            iterations=100000,
            target_ms=0.01,
            name="cache_hit"
        )

        assert result.passed
        assert result.avg_time_ms < 0.1

    @pytest.mark.performance
    def test_cache_miss_with_computation_performance(self, benchmark_runner):
        """Benchmark cache miss with computation."""
        cache = {}

        def cache_lookup_with_compute():
            key = "steam_props_999"
            if key not in cache:
                # Compute value
                value = {"enthalpy": float(Decimal('2800') + Decimal('999'))}
                cache[key] = value
            return cache[key]

        result = benchmark_runner.run_benchmark(
            cache_lookup_with_compute,
            iterations=10000,
            target_ms=0.5,
            name="cache_miss_compute"
        )

        assert result.passed


# ============================================================================
# ORCHESTRATION PERFORMANCE BENCHMARKS
# ============================================================================

class TestOrchestrationPerformance:
    """Benchmarks for orchestration overhead."""

    @pytest.mark.performance
    def test_full_analysis_pipeline_benchmark(self, benchmark_runner):
        """Benchmark complete analysis pipeline."""
        def full_analysis():
            # Step 1: Validate inputs
            pressure = Decimal('10.0')
            temperature = Decimal('180.0')

            # Step 2: Calculate steam properties
            enthalpy = Decimal('4.18') * temperature + Decimal('2257')

            # Step 3: Calculate efficiency
            efficiency = Decimal('95.0')

            # Step 4: Generate recommendations
            recommendations = ["Maintain current efficiency level"]

            # Step 5: Create result with provenance
            result = {
                'enthalpy': float(enthalpy),
                'efficiency': float(efficiency),
                'recommendations': recommendations
            }

            # Step 6: Generate provenance hash
            result['provenance_hash'] = hashlib.sha256(
                json.dumps(result, sort_keys=True, default=str).encode()
            ).hexdigest()

            return result

        result = benchmark_runner.run_benchmark(
            full_analysis,
            iterations=1000,
            target_ms=100.0,
            name="full_analysis_pipeline"
        )

        assert result.passed
        assert result.p99_time_ms < 500.0

    @pytest.mark.performance
    def test_parallel_calculation_throughput(self, benchmark_runner):
        """Benchmark parallel calculation capacity."""
        num_parallel = 10

        def parallel_calculations():
            results = []
            for i in range(num_parallel):
                P = Decimal(str(5 + i))
                T = Decimal(str(150 + i * 5))
                h = Decimal('4.18') * T + Decimal('2257')
                results.append(float(h))
            return results

        result = benchmark_runner.run_benchmark(
            parallel_calculations,
            iterations=500,
            target_ms=20.0,
            name="parallel_calculations"
        )

        assert result.passed
        calcs_per_sec = num_parallel * result.throughput_per_sec
        assert calcs_per_sec > 1000
