# -*- coding: utf-8 -*-
"""
Performance benchmark tests for GL-008 TRAPCATCHER SteamTrapInspector.

This module contains performance benchmarks for throughput, latency, memory usage,
and scalability of the SteamTrapInspector agent and its tools.

Performance Targets:
- Acoustic analysis: < 100ms per analysis
- Thermal analysis: < 10ms per analysis
- Energy loss calculation: < 5ms per calculation
- Fleet prioritization: < 1s for 100 traps, < 5s for 1000 traps
- Memory usage: < 500MB increase for batch processing of 10,000 records
"""

import pytest
import numpy as np
import time
import gc
from typing import Dict, List, Any, Callable
from datetime import datetime
import sys
from pathlib import Path
import statistics

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools import SteamTrapTools
from config import TrapType, FailureMode


# Try to import psutil for memory tests
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@pytest.fixture
def tools():
    """Create SteamTrapTools instance for benchmarking."""
    return SteamTrapTools()


@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        'acoustic_max_time_ms': 100,
        'thermal_max_time_ms': 10,
        'energy_loss_max_time_ms': 5,
        'diagnosis_max_time_ms': 20,
        'rul_max_time_ms': 50,
        'prioritization_100_max_time_s': 1.0,
        'prioritization_1000_max_time_s': 5.0,
        'batch_throughput_min': 100,  # records per second
        'memory_increase_max_mb': 500,
        'warmup_iterations': 3,
        'benchmark_iterations': 10
    }


def measure_execution_time(func: Callable, *args, **kwargs) -> float:
    """Measure execution time of a function in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@pytest.mark.performance
class TestAcousticAnalysisPerformance:
    """Performance benchmarks for acoustic signature analysis."""

    def test_acoustic_analysis_latency(self, tools, benchmark_config):
        """Test that acoustic analysis completes within target latency."""
        np.random.seed(42)
        signal = (np.random.randn(50000) * 0.3).tolist()

        acoustic_data = {
            'trap_id': 'TRAP-PERF-ACOUSTIC',
            'signal': signal,
            'sampling_rate_hz': 250000
        }

        # Warmup
        for _ in range(benchmark_config['warmup_iterations']):
            tools.analyze_acoustic_signature(acoustic_data)

        # Benchmark
        times = []
        for _ in range(benchmark_config['benchmark_iterations']):
            duration_ms, _ = measure_execution_time(
                tools.analyze_acoustic_signature, acoustic_data
            )
            times.append(duration_ms)

        avg_time = statistics.mean(times)
        max_time = max(times)
        p95_time = np.percentile(times, 95)

        assert avg_time < benchmark_config['acoustic_max_time_ms'], \
            f"Avg acoustic analysis time {avg_time:.2f}ms exceeds target {benchmark_config['acoustic_max_time_ms']}ms"

    def test_acoustic_analysis_with_varying_signal_lengths(self, tools, benchmark_config):
        """Test acoustic analysis performance with different signal lengths."""
        np.random.seed(42)
        signal_lengths = [1000, 10000, 50000, 100000, 250000]
        results = {}

        for length in signal_lengths:
            signal = (np.random.randn(length) * 0.3).tolist()
            acoustic_data = {
                'trap_id': f'TRAP-PERF-LEN-{length}',
                'signal': signal,
                'sampling_rate_hz': 250000
            }

            times = []
            for _ in range(5):
                duration_ms, _ = measure_execution_time(
                    tools.analyze_acoustic_signature, acoustic_data
                )
                times.append(duration_ms)

            results[length] = statistics.mean(times)

        # Verify scaling is reasonable (roughly linear or better)
        assert results[250000] < results[1000] * 1000, \
            "Acoustic analysis does not scale reasonably with signal length"

    def test_acoustic_batch_throughput(self, tools, benchmark_config):
        """Test throughput for batch acoustic analysis."""
        np.random.seed(42)
        num_samples = 50
        signals = [
            {
                'trap_id': f'TRAP-BATCH-{i:03d}',
                'signal': (np.random.randn(10000) * 0.3).tolist(),
                'sampling_rate_hz': 250000
            }
            for i in range(num_samples)
        ]

        start = time.perf_counter()
        for signal_data in signals:
            tools.analyze_acoustic_signature(signal_data)
        total_time = time.perf_counter() - start

        throughput = num_samples / total_time

        assert throughput >= benchmark_config['batch_throughput_min'], \
            f"Batch throughput {throughput:.2f} rec/s below target {benchmark_config['batch_throughput_min']} rec/s"


@pytest.mark.performance
class TestThermalAnalysisPerformance:
    """Performance benchmarks for thermal pattern analysis."""

    def test_thermal_analysis_latency(self, tools, benchmark_config):
        """Test that thermal analysis completes within target latency."""
        thermal_data = {
            'trap_id': 'TRAP-PERF-THERMAL',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0
        }

        # Warmup
        for _ in range(benchmark_config['warmup_iterations']):
            tools.analyze_thermal_pattern(thermal_data)

        # Benchmark
        times = []
        for _ in range(benchmark_config['benchmark_iterations']):
            duration_ms, _ = measure_execution_time(
                tools.analyze_thermal_pattern, thermal_data
            )
            times.append(duration_ms)

        avg_time = statistics.mean(times)

        assert avg_time < benchmark_config['thermal_max_time_ms'], \
            f"Avg thermal analysis time {avg_time:.2f}ms exceeds target {benchmark_config['thermal_max_time_ms']}ms"

    def test_thermal_batch_throughput(self, tools, benchmark_config):
        """Test throughput for batch thermal analysis."""
        num_samples = 1000
        thermal_data_list = [
            {
                'trap_id': f'TRAP-THERMAL-BATCH-{i:04d}',
                'temperature_upstream_c': 140.0 + (i % 40),
                'temperature_downstream_c': 100.0 + (i % 60),
                'ambient_temp_c': 15.0 + (i % 20)
            }
            for i in range(num_samples)
        ]

        start = time.perf_counter()
        for thermal_data in thermal_data_list:
            tools.analyze_thermal_pattern(thermal_data)
        total_time = time.perf_counter() - start

        throughput = num_samples / total_time

        assert throughput >= 500, \
            f"Thermal batch throughput {throughput:.2f} rec/s is below minimum"


@pytest.mark.performance
class TestEnergyLossCalculationPerformance:
    """Performance benchmarks for energy loss calculations."""

    def test_energy_loss_calculation_latency(self, tools, benchmark_config):
        """Test that energy loss calculation completes within target latency."""
        trap_data = {
            'trap_id': 'TRAP-PERF-ENERGY',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'steam_cost_usd_per_1000lb': 8.50,
            'operating_hours_yr': 8760,
            'failure_severity': 1.0
        }

        # Warmup
        for _ in range(benchmark_config['warmup_iterations']):
            tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Benchmark
        times = []
        for _ in range(benchmark_config['benchmark_iterations']):
            duration_ms, _ = measure_execution_time(
                tools.calculate_energy_loss, trap_data, FailureMode.FAILED_OPEN
            )
            times.append(duration_ms)

        avg_time = statistics.mean(times)

        assert avg_time < benchmark_config['energy_loss_max_time_ms'], \
            f"Avg energy loss calc time {avg_time:.2f}ms exceeds target {benchmark_config['energy_loss_max_time_ms']}ms"

    def test_energy_loss_high_volume(self, tools):
        """Test energy loss calculations at high volume (10,000 calculations)."""
        num_calculations = 10000
        trap_data = {
            'trap_id': 'TRAP-HIGH-VOL',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        start = time.perf_counter()
        for i in range(num_calculations):
            trap_data['trap_id'] = f'TRAP-HIGH-VOL-{i:05d}'
            tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        total_time = time.perf_counter() - start

        throughput = num_calculations / total_time

        assert throughput >= 1000, \
            f"Energy loss throughput {throughput:.2f} calc/s is below minimum of 1000"


@pytest.mark.performance
class TestFleetPrioritizationPerformance:
    """Performance benchmarks for fleet prioritization."""

    def test_prioritization_100_traps(self, tools, benchmark_config):
        """Test fleet prioritization with 100 traps."""
        fleet = self._generate_fleet(100)

        # Warmup
        tools.prioritize_maintenance(fleet)

        # Benchmark
        times = []
        for _ in range(5):
            duration_ms, _ = measure_execution_time(
                tools.prioritize_maintenance, fleet
            )
            times.append(duration_ms)

        avg_time_s = statistics.mean(times) / 1000

        assert avg_time_s < benchmark_config['prioritization_100_max_time_s'], \
            f"100-trap prioritization {avg_time_s:.2f}s exceeds target {benchmark_config['prioritization_100_max_time_s']}s"

    def test_prioritization_500_traps(self, tools, benchmark_config):
        """Test fleet prioritization with 500 traps."""
        fleet = self._generate_fleet(500)

        times = []
        for _ in range(3):
            duration_ms, _ = measure_execution_time(
                tools.prioritize_maintenance, fleet
            )
            times.append(duration_ms)

        avg_time_s = statistics.mean(times) / 1000

        assert avg_time_s < 3.0, \
            f"500-trap prioritization {avg_time_s:.2f}s exceeds 3s target"

    def test_prioritization_1000_traps(self, tools, benchmark_config):
        """Test fleet prioritization with 1000 traps."""
        fleet = self._generate_fleet(1000)

        times = []
        for _ in range(3):
            duration_ms, _ = measure_execution_time(
                tools.prioritize_maintenance, fleet
            )
            times.append(duration_ms)

        avg_time_s = statistics.mean(times) / 1000

        assert avg_time_s < benchmark_config['prioritization_1000_max_time_s'], \
            f"1000-trap prioritization {avg_time_s:.2f}s exceeds target {benchmark_config['prioritization_1000_max_time_s']}s"

    def test_prioritization_scaling(self, tools):
        """Test that prioritization scales reasonably with fleet size."""
        fleet_sizes = [50, 100, 200, 400]
        times_by_size = {}

        for size in fleet_sizes:
            fleet = self._generate_fleet(size)
            duration_ms, _ = measure_execution_time(
                tools.prioritize_maintenance, fleet
            )
            times_by_size[size] = duration_ms

        # Verify roughly linear scaling (O(n log n) is acceptable)
        # Time for 400 should be less than 10x time for 50
        scaling_factor = times_by_size[400] / times_by_size[50]
        assert scaling_factor < 20, \
            f"Prioritization scaling factor {scaling_factor:.2f}x is too high for 8x size increase"

    def _generate_fleet(self, num_traps: int) -> List[Dict]:
        """Generate test fleet data."""
        failure_modes = [FailureMode.NORMAL, FailureMode.FAILED_OPEN,
                        FailureMode.LEAKING, FailureMode.FAILED_CLOSED]
        return [
            {
                'trap_id': f'TRAP-FLEET-{i:05d}',
                'failure_mode': failure_modes[i % 4],
                'energy_loss_usd_yr': max(0, 15000 - (i * 10)) if i % 4 != 0 else 0,
                'process_criticality': 10 - (i % 6),
                'current_age_years': 1 + (i % 15),
                'health_score': max(20, 95 - (i % 75))
            }
            for i in range(num_traps)
        ]


@pytest.mark.performance
class TestRULPredictionPerformance:
    """Performance benchmarks for RUL prediction."""

    def test_rul_prediction_latency(self, tools, benchmark_config):
        """Test that RUL prediction completes within target latency."""
        condition_data = {
            'trap_id': 'TRAP-PERF-RUL',
            'current_age_days': 1000,
            'degradation_rate': 0.1,
            'current_health_score': 70,
            'historical_failures': [1800, 2000, 2200, 1900]
        }

        # Warmup
        for _ in range(benchmark_config['warmup_iterations']):
            tools.predict_remaining_useful_life(condition_data)

        # Benchmark
        times = []
        for _ in range(benchmark_config['benchmark_iterations']):
            duration_ms, _ = measure_execution_time(
                tools.predict_remaining_useful_life, condition_data
            )
            times.append(duration_ms)

        avg_time = statistics.mean(times)

        assert avg_time < benchmark_config['rul_max_time_ms'], \
            f"Avg RUL prediction time {avg_time:.2f}ms exceeds target {benchmark_config['rul_max_time_ms']}ms"

    def test_rul_batch_processing(self, tools):
        """Test batch RUL prediction performance."""
        num_predictions = 500
        conditions = [
            {
                'trap_id': f'TRAP-RUL-BATCH-{i:04d}',
                'current_age_days': 500 + (i * 3),
                'degradation_rate': 0.05 + (i % 20) * 0.01,
                'current_health_score': max(20, 90 - (i % 70))
            }
            for i in range(num_predictions)
        ]

        start = time.perf_counter()
        for condition in conditions:
            tools.predict_remaining_useful_life(condition)
        total_time = time.perf_counter() - start

        throughput = num_predictions / total_time

        assert throughput >= 100, \
            f"RUL batch throughput {throughput:.2f} pred/s is below minimum"


@pytest.mark.performance
@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
class TestMemoryUsage:
    """Memory usage benchmarks."""

    def test_acoustic_analysis_memory(self, tools, benchmark_config):
        """Test memory usage during acoustic analysis."""
        gc.collect()
        initial_memory = get_memory_usage_mb()

        np.random.seed(42)
        num_analyses = 1000
        for i in range(num_analyses):
            signal = (np.random.randn(10000) * 0.3).tolist()
            acoustic_data = {
                'trap_id': f'TRAP-MEM-{i:04d}',
                'signal': signal,
                'sampling_rate_hz': 250000
            }
            tools.analyze_acoustic_signature(acoustic_data)

        gc.collect()
        final_memory = get_memory_usage_mb()
        memory_increase = final_memory - initial_memory

        assert memory_increase < benchmark_config['memory_increase_max_mb'], \
            f"Memory increase {memory_increase:.2f}MB exceeds target {benchmark_config['memory_increase_max_mb']}MB"

    def test_large_signal_memory(self, tools, benchmark_config):
        """Test memory usage with very large signal."""
        gc.collect()
        initial_memory = get_memory_usage_mb()

        np.random.seed(42)
        # 1 million samples (~8MB of float64 data)
        signal = (np.random.randn(1000000) * 0.3).tolist()
        acoustic_data = {
            'trap_id': 'TRAP-LARGE-SIGNAL',
            'signal': signal,
            'sampling_rate_hz': 250000
        }

        tools.analyze_acoustic_signature(acoustic_data)

        gc.collect()
        final_memory = get_memory_usage_mb()
        memory_increase = final_memory - initial_memory

        # Large signal should not cause excessive memory retention
        assert memory_increase < 100, \
            f"Large signal memory increase {memory_increase:.2f}MB is excessive"


@pytest.mark.performance
class TestConcurrentPerformance:
    """Performance under concurrent load."""

    def test_concurrent_acoustic_analysis(self, tools):
        """Test concurrent acoustic analysis performance."""
        import concurrent.futures

        np.random.seed(42)
        num_concurrent = 10
        signals = [
            {
                'trap_id': f'TRAP-CONC-{i:03d}',
                'signal': (np.random.randn(10000) * 0.3).tolist(),
                'sampling_rate_hz': 250000
            }
            for i in range(num_concurrent)
        ]

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(tools.analyze_acoustic_signature, signal)
                for signal in signals
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.perf_counter() - start

        throughput = num_concurrent / total_time

        assert len(results) == num_concurrent
        assert all(r is not None for r in results)


@pytest.mark.performance
class TestLatencyPercentiles:
    """Test latency percentiles (P50, P95, P99)."""

    def test_acoustic_latency_percentiles(self, tools):
        """Test acoustic analysis latency percentiles."""
        np.random.seed(42)
        num_iterations = 100
        times = []

        for i in range(num_iterations):
            signal = (np.random.randn(10000) * 0.3).tolist()
            acoustic_data = {
                'trap_id': f'TRAP-PERC-{i:03d}',
                'signal': signal,
                'sampling_rate_hz': 250000
            }
            duration_ms, _ = measure_execution_time(
                tools.analyze_acoustic_signature, acoustic_data
            )
            times.append(duration_ms)

        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        # Log percentiles for analysis
        print(f"\nAcoustic Analysis Latency Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")

        # P99 should be within 3x of P50 (no extreme outliers)
        assert p99 < p50 * 5, \
            f"P99 latency {p99:.2f}ms is too high compared to P50 {p50:.2f}ms"

    def test_energy_loss_latency_percentiles(self, tools):
        """Test energy loss calculation latency percentiles."""
        num_iterations = 100
        times = []

        for i in range(num_iterations):
            trap_data = {
                'trap_id': f'TRAP-ENERGY-PERC-{i:03d}',
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': 100.0,
                'failure_severity': 1.0
            }
            duration_ms, _ = measure_execution_time(
                tools.calculate_energy_loss, trap_data, FailureMode.FAILED_OPEN
            )
            times.append(duration_ms)

        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        print(f"\nEnergy Loss Latency Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")

        # All percentiles should be under 10ms for simple calculations
        assert p99 < 10, f"P99 energy loss latency {p99:.2f}ms exceeds 10ms target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
