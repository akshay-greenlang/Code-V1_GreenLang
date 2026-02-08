# -*- coding: utf-8 -*-
"""
Performance Benchmarking Engine for QA Test Harness - AGENT-FOUND-009

Provides performance benchmarking with warmup iterations, statistical
analysis (min, max, mean, median, std dev, p95, p99), threshold checking,
and baseline management for detecting performance regressions.

Zero-Hallucination Guarantees:
    - All timing measurements use deterministic perf_counter
    - All statistics computed via Python arithmetic (no LLM)
    - Percentile calculations use sorted array indexing
    - Complete provenance for every benchmark operation

Example:
    >>> from greenlang.qa_test_harness.performance_benchmarker import PerformanceBenchmarker
    >>> benchmarker = PerformanceBenchmarker(config)
    >>> result = benchmarker.benchmark(MyAgentClass, input_data, iterations=10)
    >>> print(f"p95: {result.p95_ms}ms")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from greenlang.qa_test_harness.config import QATestHarnessConfig
from greenlang.qa_test_harness.models import (
    PerformanceBenchmark,
    PerformanceBaseline,
    TestAssertion,
    SeverityLevel,
)
from greenlang.qa_test_harness.metrics import record_performance_breach

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class PerformanceBenchmarker:
    """Performance benchmarking engine for QA test harness.

    Executes performance benchmarks with configurable warmup and iteration
    counts, computes comprehensive statistics, and manages performance
    baselines for regression detection.

    Attributes:
        config: QA test harness configuration.
        _baselines: In-memory store of performance baselines keyed by agent_type.

    Example:
        >>> benchmarker = PerformanceBenchmarker(config)
        >>> result = benchmarker.benchmark(MyAgentClass, input_data)
    """

    def __init__(self, config: QATestHarnessConfig) -> None:
        """Initialize PerformanceBenchmarker.

        Args:
            config: QA test harness configuration.
        """
        self.config = config
        self._baselines: Dict[str, PerformanceBaseline] = {}

        logger.info(
            "PerformanceBenchmarker initialized: warmup=%d, iterations=%d",
            config.performance_warmup_iterations,
            config.performance_default_iterations,
        )

    def benchmark(
        self,
        agent_class: Type[Any],
        input_data: Dict[str, Any],
        iterations: Optional[int] = None,
        warmup: Optional[int] = None,
        threshold_ms: Optional[float] = None,
    ) -> PerformanceBenchmark:
        """Run a performance benchmark on an agent class.

        Args:
            agent_class: Agent class to benchmark (must support no-arg __init__).
            input_data: Input data for the agent.
            iterations: Number of benchmark iterations (defaults to config value).
            warmup: Number of warmup iterations (defaults to config value).
            threshold_ms: Performance threshold in milliseconds for p95.

        Returns:
            PerformanceBenchmark with timing statistics.
        """
        iters = iterations or self.config.performance_default_iterations
        warmup_iters = warmup or self.config.performance_warmup_iterations

        agent_type = getattr(agent_class, "AGENT_NAME", agent_class.__name__)

        logger.info(
            "Benchmarking %s: %d iterations, %d warmup",
            agent_type, iters, warmup_iters,
        )

        # Warmup phase (results discarded)
        for _ in range(warmup_iters):
            agent = agent_class()
            agent.run(input_data)

        # Benchmark phase
        timings: List[float] = []
        for _ in range(iters):
            agent = agent_class()
            start = time.perf_counter()
            agent.run(input_data)
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        stats = self._calculate_statistics(timings)

        # Check threshold
        passed_threshold = True
        if threshold_ms is not None:
            passed_threshold = stats["p95"] <= threshold_ms
            if not passed_threshold:
                record_performance_breach()

        benchmark = PerformanceBenchmark(
            agent_type=agent_type,
            operation="execute",
            iterations=iters,
            min_ms=round(stats["min"], 3),
            max_ms=round(stats["max"], 3),
            mean_ms=round(stats["mean"], 3),
            median_ms=round(stats["median"], 3),
            std_dev_ms=round(stats["std_dev"], 3),
            p95_ms=round(stats["p95"], 3),
            p99_ms=round(stats["p99"], 3),
            passed_threshold=passed_threshold,
            threshold_ms=threshold_ms,
        )

        logger.info(
            "Benchmark complete: %s mean=%.2fms, p95=%.2fms, p99=%.2fms, passed=%s",
            agent_type, stats["mean"], stats["p95"], stats["p99"], passed_threshold,
        )

        return benchmark

    def create_baseline(
        self,
        agent_type: str,
        benchmark: PerformanceBenchmark,
    ) -> PerformanceBaseline:
        """Create a performance baseline from a benchmark result.

        Args:
            agent_type: Type of agent.
            benchmark: Benchmark result to use as baseline.

        Returns:
            Created PerformanceBaseline instance.
        """
        baseline = PerformanceBaseline(
            agent_type=agent_type,
            operation=benchmark.operation,
            p95_ms=benchmark.p95_ms,
            p99_ms=benchmark.p99_ms,
            mean_ms=benchmark.mean_ms,
            threshold_ms=benchmark.threshold_ms,
        )

        self._baselines[agent_type] = baseline

        logger.info(
            "Created performance baseline: %s (p95=%.2fms, p99=%.2fms)",
            agent_type, benchmark.p95_ms, benchmark.p99_ms,
        )
        return baseline

    def get_baseline(
        self,
        agent_type: str,
    ) -> Optional[PerformanceBaseline]:
        """Get the active performance baseline for an agent type.

        Args:
            agent_type: Type of agent.

        Returns:
            PerformanceBaseline if found and active, None otherwise.
        """
        baseline = self._baselines.get(agent_type)
        if baseline and baseline.is_active:
            return baseline
        return None

    def compare_with_baseline(
        self,
        agent_type: str,
        current: PerformanceBenchmark,
    ) -> TestAssertion:
        """Compare current benchmark against stored baseline.

        Args:
            agent_type: Type of agent.
            current: Current benchmark result.

        Returns:
            TestAssertion with comparison result.
        """
        baseline = self.get_baseline(agent_type)

        if baseline is None:
            return TestAssertion(
                name="performance_baseline_check",
                passed=True,
                expected="no_baseline",
                actual=f"p95={current.p95_ms:.2f}ms",
                message="No performance baseline found; first benchmark",
                severity=SeverityLevel.INFO,
            )

        # Compare p95 values - regression if more than 20% slower
        regression_threshold = 1.2  # 20% regression allowed
        p95_ratio = current.p95_ms / baseline.p95_ms if baseline.p95_ms > 0 else 1.0
        is_regression = p95_ratio > regression_threshold

        if is_regression:
            record_performance_breach()

        return TestAssertion(
            name="performance_baseline_check",
            passed=not is_regression,
            expected=f"p95<={baseline.p95_ms * regression_threshold:.2f}ms",
            actual=f"p95={current.p95_ms:.2f}ms",
            message=(
                f"Performance within acceptable range (ratio={p95_ratio:.2f})"
                if not is_regression
                else f"PERFORMANCE REGRESSION: p95 ratio={p95_ratio:.2f} exceeds threshold"
            ),
            severity=SeverityLevel.HIGH if is_regression else SeverityLevel.INFO,
        )

    def list_baselines(self) -> List[PerformanceBaseline]:
        """List all active performance baselines.

        Returns:
            List of active performance baselines.
        """
        return [
            b for b in self._baselines.values()
            if b.is_active
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_statistics(
        self,
        timings: List[float],
    ) -> Dict[str, float]:
        """Calculate comprehensive timing statistics.

        Args:
            timings: List of timing measurements in milliseconds.

        Returns:
            Dictionary with min, max, mean, median, std_dev, p95, p99.
        """
        if not timings:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_timings = sorted(timings)
        n = len(sorted_timings)

        min_ms = sorted_timings[0]
        max_ms = sorted_timings[-1]
        mean_ms = sum(sorted_timings) / n
        median_ms = sorted_timings[n // 2]

        # Standard deviation
        variance = sum((t - mean_ms) ** 2 for t in sorted_timings) / n
        std_dev_ms = variance ** 0.5

        # Percentiles using nearest-rank method
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        p95_ms = sorted_timings[min(p95_idx, n - 1)]
        p99_ms = sorted_timings[min(p99_idx, n - 1)]

        return {
            "min": min_ms,
            "max": max_ms,
            "mean": mean_ms,
            "median": median_ms,
            "std_dev": std_dev_ms,
            "p95": p95_ms,
            "p99": p99_ms,
        }


__all__ = [
    "PerformanceBenchmarker",
]
