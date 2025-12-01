# -*- coding: utf-8 -*-
"""
Performance Test Fixtures and Configuration for GL-002 FLAMEGUARD.

Provides fixtures for:
- Benchmark runner utilities
- Performance timing and measurement
- Memory profiling helpers
- Throughput calculation utilities

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import time
import psutil
import os
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from decimal import Decimal


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with performance-specific markers."""
    markers = [
        "performance: Performance benchmark tests",
        "benchmark: Micro-benchmark tests",
        "throughput: Throughput measurement tests",
        "latency: Latency measurement tests",
        "memory: Memory usage tests",
        "slow: Slow-running performance tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
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


@dataclass
class MemoryResult:
    """Result of memory measurement."""
    initial_mb: float
    peak_mb: float
    final_mb: float
    increase_mb: float
    passed: bool
    limit_mb: float


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class PerformanceBenchmark:
    """Performance benchmark runner with detailed statistics."""

    @staticmethod
    def run_benchmark(
        func: Callable,
        iterations: int = 1000,
        target_ms: float = 1.0,
        name: str = "benchmark",
        warmup_iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run benchmark with warmup and detailed statistics.

        Args:
            func: Function to benchmark
            iterations: Number of iterations to run
            target_ms: Target time in milliseconds
            name: Name of the benchmark
            warmup_iterations: Number of warmup iterations

        Returns:
            BenchmarkResult with detailed statistics
        """
        # Warmup
        for _ in range(warmup_iterations):
            func()

        # Actual benchmark
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
        std_dev = variance ** 0.5

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

    @staticmethod
    def measure_memory(func: Callable, limit_mb: float = 100.0) -> MemoryResult:
        """
        Measure memory usage of a function.

        Args:
            func: Function to measure
            limit_mb: Maximum allowed memory increase in MB

        Returns:
            MemoryResult with memory statistics
        """
        import gc
        gc.collect()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        func()

        peak_memory = process.memory_info().rss / 1024 / 1024
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024

        increase = peak_memory - initial_memory

        return MemoryResult(
            initial_mb=initial_memory,
            peak_mb=peak_memory,
            final_mb=final_memory,
            increase_mb=increase,
            passed=increase <= limit_mb,
            limit_mb=limit_mb
        )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_runner():
    """Provide benchmark runner."""
    return PerformanceBenchmark()


@pytest.fixture
def sample_boiler_data():
    """Sample boiler data for benchmarks."""
    return {
        "boiler_id": "BOILER-001",
        "load_percent": 75.0,
        "steam_flow_kg_hr": 20000.0,
        "fuel_flow_kg_hr": 1500.0,
        "steam_pressure_bar": 35.0,
        "steam_temperature_c": 400.0,
        "o2_percent": 4.5,
        "co_ppm": 15.0,
        "nox_ppm": 22.0,
        "flue_gas_temp_c": 180.0,
        "feedwater_temp_c": 100.0,
        "ambient_temp_c": 25.0
    }


@pytest.fixture
def large_dataset():
    """Large dataset for throughput tests."""
    return [
        {
            "boiler_id": f"BOILER-{i:04d}",
            "load_percent": 50 + (i % 50),
            "steam_flow_kg_hr": 15000 + (i * 100) % 30000,
            "fuel_flow_kg_hr": 1000 + (i * 10) % 2000,
            "efficiency_percent": 80 + (i % 15),
            "o2_percent": 3.0 + (i % 30) / 10,
            "timestamp": f"2025-01-01T{i%24:02d}:00:00Z"
        }
        for i in range(10000)
    ]


@pytest.fixture
def performance_targets():
    """Performance target thresholds."""
    return {
        "efficiency_calculation_ms": 100.0,
        "combustion_analysis_ms": 150.0,
        "emissions_calculation_ms": 80.0,
        "optimization_cycle_ms": 3000.0,
        "hash_calculation_ms": 1.0,
        "data_validation_ms": 50.0,
        "throughput_min_rps": 100.0,
        "memory_increase_max_mb": 100.0
    }
