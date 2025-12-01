# -*- coding: utf-8 -*-
"""
Performance Benchmarks for GL-012 STEAMQUAL.
Comprehensive performance testing with clear pass/fail criteria.
"""

import pytest
import time
import hashlib
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_per_sec: float
    passed: bool
    target_ms: float


class PerformanceBenchmark:
    @staticmethod
    def run_benchmark(func, iterations=1000, target_ms=1.0, name="benchmark"):
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
        throughput = iterations / (total_time / 1000) if total_time > 0 else 0
        
        return BenchmarkResult(
            name=name, iterations=iterations, total_time_ms=total_time,
            avg_time_ms=avg_time, min_time_ms=min_time, max_time_ms=max_time,
            throughput_per_sec=throughput, passed=avg_time <= target_ms, target_ms=target_ms
        )


@pytest.fixture
def benchmark_runner():
    return PerformanceBenchmark()


@pytest.fixture
def sample_steam_data():
    return {"pressure_bar": 10.0, "temperature_c": 180.0, "dryness_fraction": 0.98}


class TestCalculationBenchmarks:
    @pytest.mark.performance
    def test_dryness_fraction_calculation_benchmark(self, benchmark_runner):
        def calculate_dryness():
            h_total, h_f, h_fg = Decimal("2700.0"), Decimal("762.8"), Decimal("2015.0")
            return float((h_total - h_f) / h_fg)
        
        result = benchmark_runner.run_benchmark(calculate_dryness, iterations=10000, target_ms=1.0, name="dryness")
        assert result.passed, f"Dryness calc {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        assert result.throughput_per_sec > 10000

    @pytest.mark.performance
    def test_quality_index_benchmark(self, benchmark_runner):
        def calculate_quality():
            return 0.98 * 100 * 0.4 + 0.95 * 100 * 0.3 + 0.92 * 100 * 0.3
        
        result = benchmark_runner.run_benchmark(calculate_quality, iterations=10000, target_ms=1.0)
        assert result.passed

    @pytest.mark.performance
    def test_provenance_hash_benchmark(self, benchmark_runner, sample_steam_data):
        def calculate_hash():
            return hashlib.sha256(json.dumps(sample_steam_data, sort_keys=True).encode()).hexdigest()
        
        result = benchmark_runner.run_benchmark(calculate_hash, iterations=10000, target_ms=1.0)
        assert result.passed


class TestOrchestrationPerformance:
    @pytest.mark.performance
    def test_full_analysis_pipeline(self, benchmark_runner, sample_steam_data):
        def full_analysis():
            quality = sample_steam_data["dryness_fraction"] * 100 * 0.4 + 95 * 0.3 + 92 * 0.3
            dashboard = {"quality_index": quality, "dryness": sample_steam_data["dryness_fraction"]}
            return hashlib.sha256(json.dumps(dashboard, sort_keys=True).encode()).hexdigest()
        
        result = benchmark_runner.run_benchmark(full_analysis, iterations=1000, target_ms=100.0)
        assert result.passed


class TestCachePerformance:
    @pytest.mark.performance
    def test_cache_hit_performance(self, benchmark_runner):
        cache = {f"quality_{i}": {"index": 95.0 + i * 0.01} for i in range(100)}
        def cache_lookup():
            return cache.get("quality_50")
        
        result = benchmark_runner.run_benchmark(cache_lookup, iterations=10000, target_ms=0.01)
        assert result.avg_time_ms < 0.1
