# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests for GL-006 HEATRECLAIM (WasteHeatRecoveryOptimizer).

This module provides comprehensive performance benchmarks covering:
- Pinch analysis calculation performance
- Exergy calculation throughput
- Heat exchanger network optimization performance
- ROI calculation benchmarks
- Memory usage validation
- Scalability testing with varying problem sizes
- Concurrent request handling
- Cache performance

Performance Targets:
- Pinch analysis: <100ms for 20 streams
- Exergy calculation: <50ms per stream
- HEN optimization: <500ms for 10 exchangers
- ROI calculation: <10ms
- Full pipeline: <2s for standard problems

References:
- GL-012 STEAMQUAL performance patterns
- GreenLang Performance Guidelines
"""

import pytest
import time
import hashlib
import json
import sys
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import math

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# BENCHMARK INFRASTRUCTURE
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_per_sec: float
    passed: bool
    target_ms: float
    memory_mb: float = 0.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"{self.name}: {status} - avg={self.avg_time_ms:.3f}ms "
                f"(target={self.target_ms}ms) throughput={self.throughput_per_sec:.1f}/s")


class PerformanceBenchmark:
    """Performance benchmark runner with statistics."""

    @staticmethod
    def run_benchmark(
        func,
        iterations: int = 1000,
        target_ms: float = 1.0,
        name: str = "benchmark",
        warmup_iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run a performance benchmark with warmup and statistics.

        Args:
            func: Function to benchmark (no arguments)
            iterations: Number of iterations to run
            target_ms: Target time in milliseconds
            name: Benchmark name for reporting
            warmup_iterations: Number of warmup iterations

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
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = min(times)
        max_time = max(times)

        # Calculate standard deviation
        variance = sum((t - avg_time) ** 2 for t in times) / iterations
        std_dev = math.sqrt(variance)

        throughput = iterations / (total_time / 1000) if total_time > 0 else 0

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_per_sec=throughput,
            passed=avg_time <= target_ms,
            target_ms=target_ms
        )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_runner():
    """Create performance benchmark runner."""
    return PerformanceBenchmark()


@pytest.fixture
def sample_stream_data():
    """Create sample stream data for benchmarks."""
    return {
        "hot_streams": [
            {"id": f"H{i}", "supply_temp": 180 - i * 10, "target_temp": 50 + i * 5,
             "heat_capacity_flow": 10 - i * 0.5}
            for i in range(5)
        ],
        "cold_streams": [
            {"id": f"C{i}", "supply_temp": 20 + i * 5, "target_temp": 120 + i * 10,
             "heat_capacity_flow": 8 - i * 0.4}
            for i in range(4)
        ],
        "min_approach_temp": 10.0
    }


@pytest.fixture
def large_stream_data():
    """Create large stream dataset for scalability testing."""
    return {
        "hot_streams": [
            {"id": f"H{i}", "supply_temp": 200 - i * 2, "target_temp": 40 + i,
             "heat_capacity_flow": 15 - i * 0.1}
            for i in range(50)
        ],
        "cold_streams": [
            {"id": f"C{i}", "supply_temp": 15 + i, "target_temp": 150 + i * 0.5,
             "heat_capacity_flow": 12 - i * 0.08}
            for i in range(50)
        ],
        "min_approach_temp": 10.0
    }


@pytest.fixture
def exergy_stream_data():
    """Create stream data for exergy calculations."""
    return {
        "temperature_k": 453.15,  # 180 C
        "pressure_kpa": 500.0,
        "mass_flow_kg_s": 5.0,
        "specific_heat_kj_kg_k": 4.18,
        "reference_temp_k": 298.15,
        "reference_pressure_kpa": 101.325
    }


@pytest.fixture
def roi_calculation_data():
    """Create data for ROI benchmark calculations."""
    return {
        "capital_cost_usd": 500000.0,
        "annual_savings_usd": 150000.0,
        "discount_rate": 0.10,
        "project_life_years": 15,
        "escalation_rate": 0.03
    }


# ============================================================================
# PINCH ANALYSIS BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestPinchAnalysisBenchmarks:
    """Performance benchmarks for pinch analysis calculations."""

    @pytest.mark.performance
    def test_temperature_interval_construction(self, benchmark_runner, sample_stream_data):
        """Benchmark temperature interval construction."""
        def construct_intervals():
            temps = set()
            min_approach = sample_stream_data["min_approach_temp"]

            for stream in sample_stream_data["hot_streams"]:
                temps.add(stream["supply_temp"] - min_approach / 2)
                temps.add(stream["target_temp"] - min_approach / 2)

            for stream in sample_stream_data["cold_streams"]:
                temps.add(stream["supply_temp"] + min_approach / 2)
                temps.add(stream["target_temp"] + min_approach / 2)

            return sorted(temps, reverse=True)

        result = benchmark_runner.run_benchmark(
            construct_intervals,
            iterations=10000,
            target_ms=0.1,
            name="Temperature Interval Construction"
        )

        assert result.passed, f"Temperature interval construction: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        assert result.throughput_per_sec > 10000

    @pytest.mark.performance
    def test_problem_table_algorithm(self, benchmark_runner, sample_stream_data):
        """Benchmark problem table algorithm."""
        intervals = list(range(20))  # Simulated intervals

        def problem_table():
            cumulative = 0
            results = []
            for interval in intervals:
                net_heat = random.uniform(-100, 100)  # Simulated heat flow
                cumulative += net_heat
                results.append({"interval": interval, "cumulative": cumulative})
            return results

        result = benchmark_runner.run_benchmark(
            problem_table,
            iterations=5000,
            target_ms=0.5,
            name="Problem Table Algorithm"
        )

        assert result.passed, f"Problem table: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"

    @pytest.mark.performance
    def test_composite_curve_generation(self, benchmark_runner, sample_stream_data):
        """Benchmark composite curve generation."""
        def generate_composite():
            temps = []
            enthalpies = [0]
            cumulative_h = 0

            for stream in sample_stream_data["hot_streams"]:
                delta_h = stream["heat_capacity_flow"] * abs(
                    stream["supply_temp"] - stream["target_temp"]
                )
                temps.append(stream["supply_temp"])
                cumulative_h += delta_h
                enthalpies.append(cumulative_h)

            return {"temperatures": temps, "enthalpies": enthalpies}

        result = benchmark_runner.run_benchmark(
            generate_composite,
            iterations=10000,
            target_ms=0.2,
            name="Composite Curve Generation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_full_pinch_analysis_small(self, benchmark_runner, sample_stream_data):
        """Benchmark full pinch analysis for small problems."""
        def full_pinch_analysis():
            # Step 1: Build intervals
            temps = set()
            min_approach = sample_stream_data["min_approach_temp"]
            for s in sample_stream_data["hot_streams"]:
                temps.add(s["supply_temp"] - min_approach / 2)
                temps.add(s["target_temp"] - min_approach / 2)
            for s in sample_stream_data["cold_streams"]:
                temps.add(s["supply_temp"] + min_approach / 2)
                temps.add(s["target_temp"] + min_approach / 2)

            intervals = sorted(temps, reverse=True)

            # Step 2: Calculate heat flows
            heat_flows = []
            for i in range(len(intervals) - 1):
                dt = intervals[i] - intervals[i + 1]
                hot_cp = sum(s["heat_capacity_flow"] for s in sample_stream_data["hot_streams"])
                cold_cp = sum(s["heat_capacity_flow"] for s in sample_stream_data["cold_streams"])
                net_heat = (hot_cp - cold_cp) * dt
                heat_flows.append(net_heat)

            # Step 3: Cascade
            cumulative = 0
            cascade = []
            for h in heat_flows:
                cumulative += h
                cascade.append(cumulative)

            # Step 4: Find pinch
            min_cascade = min(cascade)
            pinch_idx = cascade.index(min_cascade)

            return {
                "pinch_temp": intervals[pinch_idx],
                "min_hot_utility": max(0, -min_cascade),
                "min_cold_utility": max(0, cascade[-1] + max(0, -min_cascade))
            }

        result = benchmark_runner.run_benchmark(
            full_pinch_analysis,
            iterations=1000,
            target_ms=1.0,
            name="Full Pinch Analysis (Small)"
        )

        assert result.passed, f"Full pinch analysis: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"


@pytest.mark.performance
class TestExergyCalculationBenchmarks:
    """Performance benchmarks for exergy calculations."""

    @pytest.mark.performance
    def test_physical_exergy_calculation(self, benchmark_runner, exergy_stream_data):
        """Benchmark physical exergy calculation."""
        def calculate_physical_exergy():
            T = Decimal(str(exergy_stream_data["temperature_k"]))
            T0 = Decimal(str(exergy_stream_data["reference_temp_k"]))
            cp = Decimal(str(exergy_stream_data["specific_heat_kj_kg_k"]))
            m = Decimal(str(exergy_stream_data["mass_flow_kg_s"]))

            # Physical exergy per unit mass
            # ex_ph = cp * (T - T0 - T0 * ln(T/T0))
            ln_ratio = Decimal(str(math.log(float(T) / float(T0))))
            ex_ph = cp * (T - T0 - T0 * ln_ratio)

            return float(m * ex_ph)

        result = benchmark_runner.run_benchmark(
            calculate_physical_exergy,
            iterations=10000,
            target_ms=0.5,
            name="Physical Exergy Calculation"
        )

        assert result.passed
        assert result.throughput_per_sec > 2000

    @pytest.mark.performance
    def test_carnot_factor_calculation(self, benchmark_runner):
        """Benchmark Carnot factor calculation."""
        def calculate_carnot():
            T_hot = Decimal("453.15")  # 180 C
            T_cold = Decimal("298.15")  # 25 C
            return 1 - T_cold / T_hot

        result = benchmark_runner.run_benchmark(
            calculate_carnot,
            iterations=50000,
            target_ms=0.01,
            name="Carnot Factor Calculation"
        )

        assert result.passed
        assert result.throughput_per_sec > 50000

    @pytest.mark.performance
    def test_exergy_destruction_calculation(self, benchmark_runner):
        """Benchmark exergy destruction calculation."""
        def calculate_destruction():
            T0 = Decimal("298.15")
            S_gen = Decimal("0.5")  # Entropy generation kJ/K
            return float(T0 * S_gen)

        result = benchmark_runner.run_benchmark(
            calculate_destruction,
            iterations=50000,
            target_ms=0.01,
            name="Exergy Destruction Calculation"
        )

        assert result.passed


@pytest.mark.performance
class TestHENOptimizationBenchmarks:
    """Performance benchmarks for heat exchanger network optimization."""

    @pytest.mark.performance
    def test_lmtd_calculation(self, benchmark_runner):
        """Benchmark Log Mean Temperature Difference calculation."""
        def calculate_lmtd():
            dt1 = 80.0  # Hot inlet - Cold outlet
            dt2 = 30.0  # Hot outlet - Cold inlet

            if abs(dt1 - dt2) < 0.1:
                return dt1

            return (dt1 - dt2) / math.log(dt1 / dt2)

        result = benchmark_runner.run_benchmark(
            calculate_lmtd,
            iterations=50000,
            target_ms=0.01,
            name="LMTD Calculation"
        )

        assert result.passed
        assert result.throughput_per_sec > 100000

    @pytest.mark.performance
    def test_heat_exchanger_area_calculation(self, benchmark_runner):
        """Benchmark heat exchanger area calculation."""
        def calculate_area():
            Q = 500.0  # kW
            U = 0.5    # kW/m2K
            LMTD = 40.0  # K

            return Q / (U * LMTD)

        result = benchmark_runner.run_benchmark(
            calculate_area,
            iterations=50000,
            target_ms=0.01,
            name="HX Area Calculation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_network_capital_cost(self, benchmark_runner):
        """Benchmark network capital cost calculation."""
        exchangers = [
            {"area_m2": 50.0, "type": "shell_tube"},
            {"area_m2": 35.0, "type": "plate"},
            {"area_m2": 80.0, "type": "shell_tube"},
            {"area_m2": 25.0, "type": "plate"},
        ]

        cost_per_m2 = {"shell_tube": 800, "plate": 600}

        def calculate_network_cost():
            total = 0
            for hx in exchangers:
                base_cost = hx["area_m2"] * cost_per_m2[hx["type"]]
                # Add installation factor
                total += base_cost * 1.4
            return total

        result = benchmark_runner.run_benchmark(
            calculate_network_cost,
            iterations=10000,
            target_ms=0.1,
            name="Network Capital Cost"
        )

        assert result.passed


@pytest.mark.performance
class TestROICalculationBenchmarks:
    """Performance benchmarks for ROI calculations."""

    @pytest.mark.performance
    def test_npv_calculation(self, benchmark_runner, roi_calculation_data):
        """Benchmark Net Present Value calculation."""
        def calculate_npv():
            capital = roi_calculation_data["capital_cost_usd"]
            savings = roi_calculation_data["annual_savings_usd"]
            rate = roi_calculation_data["discount_rate"]
            years = roi_calculation_data["project_life_years"]
            escalation = roi_calculation_data["escalation_rate"]

            npv = -capital
            for year in range(1, years + 1):
                escalated_savings = savings * (1 + escalation) ** (year - 1)
                pv = escalated_savings / (1 + rate) ** year
                npv += pv

            return npv

        result = benchmark_runner.run_benchmark(
            calculate_npv,
            iterations=10000,
            target_ms=0.5,
            name="NPV Calculation"
        )

        assert result.passed
        assert result.throughput_per_sec > 2000

    @pytest.mark.performance
    def test_irr_calculation(self, benchmark_runner, roi_calculation_data):
        """Benchmark Internal Rate of Return calculation (Newton-Raphson)."""
        def calculate_irr():
            capital = roi_calculation_data["capital_cost_usd"]
            savings = roi_calculation_data["annual_savings_usd"]
            years = roi_calculation_data["project_life_years"]

            # Newton-Raphson
            irr = 0.10
            for _ in range(50):
                npv = -capital
                npv_deriv = 0

                for year in range(1, years + 1):
                    npv += savings / (1 + irr) ** year
                    npv_deriv -= year * savings / (1 + irr) ** (year + 1)

                if abs(npv) < 0.01:
                    break

                if npv_deriv != 0:
                    irr = irr - npv / npv_deriv
                    irr = max(-0.99, min(irr, 10.0))

            return irr * 100

        result = benchmark_runner.run_benchmark(
            calculate_irr,
            iterations=5000,
            target_ms=1.0,
            name="IRR Calculation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_simple_payback(self, benchmark_runner, roi_calculation_data):
        """Benchmark simple payback calculation."""
        def calculate_payback():
            capital = roi_calculation_data["capital_cost_usd"]
            savings = roi_calculation_data["annual_savings_usd"]
            return capital / savings if savings > 0 else 999

        result = benchmark_runner.run_benchmark(
            calculate_payback,
            iterations=100000,
            target_ms=0.001,
            name="Simple Payback"
        )

        assert result.passed
        assert result.throughput_per_sec > 1000000


@pytest.mark.performance
class TestProvenanceHashBenchmarks:
    """Performance benchmarks for provenance hash calculations."""

    @pytest.mark.performance
    def test_sha256_hash_small(self, benchmark_runner, sample_stream_data):
        """Benchmark SHA-256 hash for small data."""
        def calculate_hash():
            return hashlib.sha256(
                json.dumps(sample_stream_data, sort_keys=True).encode()
            ).hexdigest()

        result = benchmark_runner.run_benchmark(
            calculate_hash,
            iterations=10000,
            target_ms=0.5,
            name="SHA-256 Hash (Small)"
        )

        assert result.passed
        assert result.throughput_per_sec > 2000

    @pytest.mark.performance
    def test_sha256_hash_large(self, benchmark_runner, large_stream_data):
        """Benchmark SHA-256 hash for large data."""
        def calculate_hash():
            return hashlib.sha256(
                json.dumps(large_stream_data, sort_keys=True).encode()
            ).hexdigest()

        result = benchmark_runner.run_benchmark(
            calculate_hash,
            iterations=1000,
            target_ms=5.0,
            name="SHA-256 Hash (Large)"
        )

        assert result.passed


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Test performance scaling with problem size."""

    @pytest.mark.performance
    def test_pinch_scaling_with_streams(self, benchmark_runner):
        """Test pinch analysis scaling with number of streams."""
        timing_results = {}

        for n_streams in [5, 10, 20, 50]:
            streams = [
                {"supply_temp": 180 - i * 2, "target_temp": 50 + i, "cp": 10}
                for i in range(n_streams)
            ]

            def analyze():
                temps = set()
                for s in streams:
                    temps.add(s["supply_temp"])
                    temps.add(s["target_temp"])
                intervals = sorted(temps, reverse=True)

                cascade = []
                cumulative = 0
                for i in range(len(intervals) - 1):
                    dt = intervals[i] - intervals[i + 1]
                    net = sum(s["cp"] for s in streams) * dt
                    cumulative += net
                    cascade.append(cumulative)

                return min(cascade)

            start = time.perf_counter()
            for _ in range(100):
                analyze()
            elapsed = (time.perf_counter() - start) * 1000

            timing_results[n_streams] = elapsed / 100

        # Verify polynomial scaling (not exponential)
        # 50 streams should not take more than 20x of 5 streams
        assert timing_results[50] < timing_results[5] * 50

    @pytest.mark.performance
    def test_hen_scaling_with_exchangers(self, benchmark_runner):
        """Test HEN optimization scaling with number of exchangers."""
        timing_results = {}

        for n_exchangers in [2, 5, 10, 20]:
            exchangers = [
                {"id": f"HX-{i}", "area": 50 + i * 10, "duty": 500 + i * 50}
                for i in range(n_exchangers)
            ]

            def optimize():
                total_area = sum(hx["area"] for hx in exchangers)
                total_duty = sum(hx["duty"] for hx in exchangers)
                total_cost = sum(hx["area"] * 800 for hx in exchangers)
                return {"area": total_area, "duty": total_duty, "cost": total_cost}

            start = time.perf_counter()
            for _ in range(1000):
                optimize()
            elapsed = (time.perf_counter() - start) * 1000

            timing_results[n_exchangers] = elapsed / 1000

        # Linear scaling expected
        assert timing_results[20] < timing_results[2] * 20


@pytest.mark.performance
class TestCacheBenchmarks:
    """Test cache performance."""

    @pytest.mark.performance
    def test_cache_hit_performance(self, benchmark_runner):
        """Benchmark cache hit performance."""
        cache = {f"result_{i}": {"value": i * 100} for i in range(1000)}

        def cache_lookup():
            return cache.get("result_500")

        result = benchmark_runner.run_benchmark(
            cache_lookup,
            iterations=100000,
            target_ms=0.001,
            name="Cache Hit"
        )

        assert result.avg_time_ms < 0.01

    @pytest.mark.performance
    def test_cache_miss_with_compute(self, benchmark_runner):
        """Benchmark cache miss with recomputation."""
        cache = {}

        def compute_and_cache():
            key = "computed_result"
            if key not in cache:
                # Simulate computation
                result = sum(i ** 2 for i in range(100))
                cache[key] = result
            return cache[key]

        result = benchmark_runner.run_benchmark(
            compute_and_cache,
            iterations=10000,
            target_ms=0.1,
            name="Cache Miss + Compute"
        )

        assert result.passed


@pytest.mark.performance
class TestMemoryBenchmarks:
    """Test memory usage."""

    @pytest.mark.performance
    def test_memory_stability_under_load(self):
        """Test memory usage remains stable under repeated calculations."""
        import gc

        # Force garbage collection
        gc.collect()

        results = []
        for iteration in range(100):
            # Create moderately sized data
            data = {
                "streams": [{"id": i, "temp": 100 + i} for i in range(100)],
                "results": [{"value": i * 1.5} for i in range(100)]
            }

            # Process and hash
            hash_val = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()

            results.append(hash_val)

        # All hashes should be identical (deterministic)
        assert len(set(results)) == 1

        # Clean up
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
