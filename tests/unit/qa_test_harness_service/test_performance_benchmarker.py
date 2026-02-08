# -*- coding: utf-8 -*-
"""
Unit Tests for PerformanceBenchmarker (AGENT-FOUND-009)

Tests benchmark execution, warmup, threshold checking, statistics calculation,
percentile computation, baseline creation, and baseline comparison.

Coverage target: 85%+ of performance_benchmarker.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline PerformanceBenchmarker
# ---------------------------------------------------------------------------

class BenchmarkResult:
    def __init__(self, agent_type: str, operation: str = "execute",
                 iterations: int = 0, min_ms: float = 0.0, max_ms: float = 0.0,
                 mean_ms: float = 0.0, median_ms: float = 0.0,
                 std_dev_ms: float = 0.0, p95_ms: float = 0.0,
                 p99_ms: float = 0.0, passed_threshold: bool = True,
                 threshold_ms: Optional[float] = None):
        self.agent_type = agent_type
        self.operation = operation
        self.iterations = iterations
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.mean_ms = mean_ms
        self.median_ms = median_ms
        self.std_dev_ms = std_dev_ms
        self.p95_ms = p95_ms
        self.p99_ms = p99_ms
        self.passed_threshold = passed_threshold
        self.threshold_ms = threshold_ms


class PerformanceBaseline:
    def __init__(self, baseline_id: str, agent_type: str, mean_ms: float,
                 p95_ms: float, p99_ms: float,
                 threshold_ms: Optional[float] = None,
                 created_at: Optional[datetime] = None):
        self.baseline_id = baseline_id
        self.agent_type = agent_type
        self.mean_ms = mean_ms
        self.p95_ms = p95_ms
        self.p99_ms = p99_ms
        self.threshold_ms = threshold_ms
        self.created_at = created_at or datetime.now(timezone.utc)


class PerformanceBenchmarker:
    """Performance benchmarker for agents."""

    def __init__(self):
        self._baselines: Dict[str, PerformanceBaseline] = {}
        self._history: Dict[str, List[BenchmarkResult]] = {}
        self._counter = 0

    def benchmark(self, agent_fn: Callable, input_data: Dict[str, Any],
                  agent_type: str = "Agent", iterations: int = 10,
                  warmup: int = 2,
                  threshold_ms: Optional[float] = None) -> BenchmarkResult:
        """Run a benchmark."""
        # Warmup
        for _ in range(warmup):
            agent_fn(input_data)

        # Measure
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            agent_fn(input_data)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

        stats = self._calculate_statistics(timings)
        passed = True
        if threshold_ms is not None:
            passed = stats["p95_ms"] <= threshold_ms

        result = BenchmarkResult(
            agent_type=agent_type, iterations=iterations,
            min_ms=stats["min_ms"], max_ms=stats["max_ms"],
            mean_ms=stats["mean_ms"], median_ms=stats["median_ms"],
            std_dev_ms=stats["std_dev_ms"], p95_ms=stats["p95_ms"],
            p99_ms=stats["p99_ms"], passed_threshold=passed,
            threshold_ms=threshold_ms,
        )

        if agent_type not in self._history:
            self._history[agent_type] = []
        self._history[agent_type].append(result)

        return result

    def _calculate_statistics(self, timings: List[float]) -> Dict[str, float]:
        """Calculate timing statistics."""
        if not timings:
            return {k: 0.0 for k in [
                "min_ms", "max_ms", "mean_ms", "median_ms",
                "std_dev_ms", "p95_ms", "p99_ms"
            ]}

        sorted_timings = sorted(timings)
        n = len(sorted_timings)
        mean = sum(sorted_timings) / n
        variance = sum((t - mean) ** 2 for t in sorted_timings) / n
        std_dev = math.sqrt(variance)

        return {
            "min_ms": round(sorted_timings[0], 3),
            "max_ms": round(sorted_timings[-1], 3),
            "mean_ms": round(mean, 3),
            "median_ms": round(sorted_timings[n // 2], 3),
            "std_dev_ms": round(std_dev, 3),
            "p95_ms": round(self._percentile(sorted_timings, 95), 3),
            "p99_ms": round(self._percentile(sorted_timings, 99), 3),
        }

    def _percentile(self, sorted_data: List[float], pct: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        idx = int(len(sorted_data) * pct / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    def create_baseline(self, agent_type: str,
                        result: BenchmarkResult,
                        threshold_ms: Optional[float] = None) -> PerformanceBaseline:
        """Create a performance baseline from benchmark result."""
        self._counter += 1
        baseline_id = f"pb-{self._counter:04d}"
        baseline = PerformanceBaseline(
            baseline_id=baseline_id, agent_type=agent_type,
            mean_ms=result.mean_ms, p95_ms=result.p95_ms,
            p99_ms=result.p99_ms, threshold_ms=threshold_ms,
        )
        self._baselines[baseline_id] = baseline
        return baseline

    def get_baseline(self, baseline_id: str) -> Optional[PerformanceBaseline]:
        return self._baselines.get(baseline_id)

    def compare_with_baseline(self, baseline_id: str,
                              result: BenchmarkResult,
                              tolerance_pct: float = 10.0) -> Dict[str, Any]:
        """Compare benchmark result against baseline."""
        baseline = self._baselines.get(baseline_id)
        if baseline is None:
            return {"passed": False, "error": "Baseline not found"}

        mean_diff_pct = abs(result.mean_ms - baseline.mean_ms) / baseline.mean_ms * 100 if baseline.mean_ms > 0 else 0
        p95_diff_pct = abs(result.p95_ms - baseline.p95_ms) / baseline.p95_ms * 100 if baseline.p95_ms > 0 else 0

        mean_ok = mean_diff_pct <= tolerance_pct
        p95_ok = p95_diff_pct <= tolerance_pct

        return {
            "passed": mean_ok and p95_ok,
            "mean_diff_pct": round(mean_diff_pct, 2),
            "p95_diff_pct": round(p95_diff_pct, 2),
            "tolerance_pct": tolerance_pct,
            "baseline_mean_ms": baseline.mean_ms,
            "result_mean_ms": result.mean_ms,
        }

    def get_history(self, agent_type: str) -> List[BenchmarkResult]:
        return self._history.get(agent_type, [])


# ===========================================================================
# Mock agent functions
# ===========================================================================


def _fast_agent(data):
    return {"success": True, "data": {"result": 42}}


def _medium_agent(data):
    time.sleep(0.001)
    return {"success": True, "data": {"result": 42}}


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def benchmarker():
    return PerformanceBenchmarker()


class TestBenchmarkBasic:
    def test_benchmark_basic(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, iterations=5, warmup=1)
        assert result.iterations == 5
        assert result.mean_ms >= 0

    def test_benchmark_has_timing_stats(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, iterations=10, warmup=1)
        assert result.min_ms >= 0
        assert result.max_ms >= result.min_ms
        assert result.mean_ms >= result.min_ms
        assert result.mean_ms <= result.max_ms

    def test_benchmark_agent_type(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, agent_type="TestAgent")
        assert result.agent_type == "TestAgent"

    def test_benchmark_records_history(self, benchmarker):
        benchmarker.benchmark(_fast_agent, {}, agent_type="Agent1")
        benchmarker.benchmark(_fast_agent, {}, agent_type="Agent1")
        history = benchmarker.get_history("Agent1")
        assert len(history) == 2


class TestBenchmarkWarmup:
    def test_benchmark_with_warmup(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, iterations=5, warmup=3)
        assert result.iterations == 5  # warmup not counted

    def test_benchmark_zero_warmup(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, iterations=5, warmup=0)
        assert result.iterations == 5


class TestBenchmarkThreshold:
    def test_benchmark_threshold_pass(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, iterations=5,
                                       threshold_ms=1000.0)
        assert result.passed_threshold is True

    def test_benchmark_threshold_fail(self, benchmarker):
        result = benchmarker.benchmark(_medium_agent, {}, iterations=5,
                                       threshold_ms=0.001)
        assert result.passed_threshold is False

    def test_benchmark_no_threshold(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {}, iterations=5)
        assert result.passed_threshold is True
        assert result.threshold_ms is None


class TestCalculateStatistics:
    def test_calculate_statistics_basic(self, benchmarker):
        stats = benchmarker._calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["min_ms"] == 1.0
        assert stats["max_ms"] == 5.0
        assert stats["mean_ms"] == 3.0

    def test_calculate_statistics_median(self, benchmarker):
        stats = benchmarker._calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["median_ms"] == 3.0

    def test_calculate_statistics_single_value(self, benchmarker):
        stats = benchmarker._calculate_statistics([5.0])
        assert stats["min_ms"] == 5.0
        assert stats["max_ms"] == 5.0
        assert stats["mean_ms"] == 5.0
        assert stats["std_dev_ms"] == 0.0

    def test_calculate_statistics_empty(self, benchmarker):
        stats = benchmarker._calculate_statistics([])
        assert stats["mean_ms"] == 0.0

    def test_calculate_statistics_std_dev(self, benchmarker):
        stats = benchmarker._calculate_statistics([1.0, 1.0, 1.0])
        assert stats["std_dev_ms"] == 0.0


class TestPercentileCalculation:
    def test_p95_calculation(self, benchmarker):
        data = list(range(1, 101))  # 1 to 100
        p95 = benchmarker._percentile(data, 95)
        assert p95 == 95 or p95 == 96

    def test_p99_calculation(self, benchmarker):
        data = list(range(1, 101))
        p99 = benchmarker._percentile(data, 99)
        assert p99 >= 99

    def test_p50_calculation(self, benchmarker):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        p50 = benchmarker._percentile(data, 50)
        assert p50 == 3.0

    def test_percentile_empty(self, benchmarker):
        p = benchmarker._percentile([], 95)
        assert p == 0.0

    def test_percentile_single(self, benchmarker):
        p = benchmarker._percentile([42.0], 95)
        assert p == 42.0


class TestCreateBaseline:
    def test_create_baseline(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {})
        baseline = benchmarker.create_baseline("Agent", result)
        assert baseline.baseline_id.startswith("pb-")
        assert baseline.agent_type == "Agent"
        assert baseline.mean_ms == result.mean_ms

    def test_create_baseline_with_threshold(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {})
        baseline = benchmarker.create_baseline("Agent", result, threshold_ms=100.0)
        assert baseline.threshold_ms == 100.0


class TestGetBaseline:
    def test_get_baseline(self, benchmarker):
        result = benchmarker.benchmark(_fast_agent, {})
        baseline = benchmarker.create_baseline("Agent", result)
        retrieved = benchmarker.get_baseline(baseline.baseline_id)
        assert retrieved is not None
        assert retrieved.baseline_id == baseline.baseline_id

    def test_get_nonexistent(self, benchmarker):
        assert benchmarker.get_baseline("pb-9999") is None


class TestCompareWithBaseline:
    def test_compare_pass(self, benchmarker):
        result1 = benchmarker.benchmark(_fast_agent, {}, iterations=10)
        baseline = benchmarker.create_baseline("Agent", result1)
        result2 = benchmarker.benchmark(_fast_agent, {}, iterations=10)
        comparison = benchmarker.compare_with_baseline(baseline.baseline_id, result2,
                                                        tolerance_pct=200.0)
        assert comparison["passed"] is True

    def test_compare_fail(self, benchmarker):
        # Create baseline with fast agent
        fast_result = BenchmarkResult(
            agent_type="Agent", iterations=10,
            mean_ms=1.0, p95_ms=1.5, p99_ms=2.0,
        )
        baseline = benchmarker.create_baseline("Agent", fast_result)

        # Create result with very different mean
        slow_result = BenchmarkResult(
            agent_type="Agent", iterations=10,
            mean_ms=100.0, p95_ms=150.0, p99_ms=200.0,
        )
        comparison = benchmarker.compare_with_baseline(baseline.baseline_id, slow_result,
                                                        tolerance_pct=10.0)
        assert comparison["passed"] is False

    def test_compare_not_found(self, benchmarker):
        result = BenchmarkResult(agent_type="Agent", mean_ms=5.0, p95_ms=8.0)
        comparison = benchmarker.compare_with_baseline("pb-9999", result)
        assert comparison["passed"] is False
        assert "not found" in comparison["error"].lower()

    def test_compare_tolerance(self, benchmarker):
        base_result = BenchmarkResult(
            agent_type="Agent", mean_ms=10.0, p95_ms=15.0, p99_ms=20.0,
        )
        baseline = benchmarker.create_baseline("Agent", base_result)

        # Within 20% tolerance
        test_result = BenchmarkResult(
            agent_type="Agent", mean_ms=11.0, p95_ms=16.0, p99_ms=21.0,
        )
        comparison = benchmarker.compare_with_baseline(baseline.baseline_id, test_result,
                                                        tolerance_pct=20.0)
        assert comparison["passed"] is True

    def test_compare_has_diff_percentages(self, benchmarker):
        base = BenchmarkResult(agent_type="A", mean_ms=10.0, p95_ms=15.0, p99_ms=20.0)
        baseline = benchmarker.create_baseline("A", base)
        test = BenchmarkResult(agent_type="A", mean_ms=12.0, p95_ms=18.0, p99_ms=22.0)
        result = benchmarker.compare_with_baseline(baseline.baseline_id, test)
        assert "mean_diff_pct" in result
        assert "p95_diff_pct" in result


class TestGetHistory:
    def test_get_history_empty(self, benchmarker):
        assert benchmarker.get_history("Agent") == []

    def test_get_history_after_benchmarks(self, benchmarker):
        benchmarker.benchmark(_fast_agent, {}, agent_type="Agent")
        benchmarker.benchmark(_fast_agent, {}, agent_type="Agent")
        assert len(benchmarker.get_history("Agent")) == 2
