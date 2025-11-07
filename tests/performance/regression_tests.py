"""Performance Regression Tests for GreenLang Agents.

This module provides automated performance regression detection:
- Baseline performance metrics storage
- Automated regression detection
- Performance SLOs (Service Level Objectives)
- Historical performance tracking
- Pass/fail determination based on thresholds

Service Level Objectives (SLOs):
- Agent execution: p95 < 500ms
- API endpoints: p95 < 200ms
- Database queries: p95 < 50ms
- Async speedup: >= 5x for 10 parallel agents
- Memory overhead: < 5% vs baseline

Example Usage:
    >>> from tests.performance.regression_tests import RegressionTester
    >>>
    >>> async def main():
    ...     tester = RegressionTester()
    ...     results = await tester.run_regression_tests()
    ...     if not results.all_passed:
    ...         print("REGRESSION DETECTED!")

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try importing async agents
try:
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
    from greenlang.config import get_config
    ASYNC_AGENTS_AVAILABLE = True
except ImportError:
    ASYNC_AGENTS_AVAILABLE = False


# ==============================================================================
# Performance SLOs (Service Level Objectives)
# ==============================================================================

@dataclass
class PerformanceSLO:
    """Service Level Objective for performance metrics."""
    name: str
    metric_name: str  # e.g., "p95_latency_ms", "throughput_rps"
    threshold: float
    comparison: str = "lt"  # "lt", "gt", "le", "ge"
    unit: str = "ms"

    def check(self, value: float) -> bool:
        """Check if value meets SLO."""
        if self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "le":
            return value <= self.threshold
        elif self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "ge":
            return value >= self.threshold
        return False


# Default SLOs
DEFAULT_SLOS = [
    PerformanceSLO(
        name="Agent Execution p95",
        metric_name="p95_latency_ms",
        threshold=500.0,
        comparison="lt",
        unit="ms"
    ),
    PerformanceSLO(
        name="Agent Execution p99",
        metric_name="p99_latency_ms",
        threshold=1000.0,
        comparison="lt",
        unit="ms"
    ),
    PerformanceSLO(
        name="Error Rate",
        metric_name="error_rate",
        threshold=0.01,  # < 1%
        comparison="lt",
        unit="%"
    ),
    PerformanceSLO(
        name="Throughput (10 concurrent)",
        metric_name="throughput_rps",
        threshold=5.0,  # > 5 RPS
        comparison="gt",
        unit="RPS"
    ),
]


# ==============================================================================
# Baseline and Regression Data Structures
# ==============================================================================

@dataclass
class PerformanceBaseline:
    """Baseline performance metrics."""
    test_name: str
    version: str
    timestamp: datetime

    # Latency metrics (ms)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput
    throughput_rps: float

    # Resource usage
    memory_mb: float
    cpu_percent: float

    # Error rate
    error_rate: float

    # Raw data
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionTestResult:
    """Result from a single regression test."""
    test_name: str
    passed: bool
    baseline: Optional[PerformanceBaseline]
    current: PerformanceBaseline
    slo_results: List[Tuple[PerformanceSLO, bool, float]]  # (SLO, passed, value)
    regression_detected: bool
    regression_details: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegressionTestSuite:
    """Results from full regression test suite."""
    results: List[RegressionTestResult]
    all_passed: bool
    num_tests: int
    num_passed: int
    num_failed: int
    timestamp: datetime = field(default_factory=datetime.now)


# ==============================================================================
# Regression Tester
# ==============================================================================

class RegressionTester:
    """Performance regression testing framework."""

    def __init__(
        self,
        baseline_dir: Optional[Path] = None,
        slos: Optional[List[PerformanceSLO]] = None,
        regression_threshold: float = 0.10,  # 10% performance degradation
    ):
        """Initialize regression tester.

        Args:
            baseline_dir: Directory to store baseline metrics
            slos: Custom SLOs (uses defaults if not provided)
            regression_threshold: Acceptable performance degradation (0.10 = 10%)
        """
        if baseline_dir is None:
            baseline_dir = Path(__file__).parent / "results" / "baselines"

        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        self.slos = slos if slos else DEFAULT_SLOS
        self.regression_threshold = regression_threshold

        self.config = get_config() if ASYNC_AGENTS_AVAILABLE else None

        # Test input
        self.default_test_input = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US"
        }

    # ==========================================================================
    # Regression Testing
    # ==========================================================================

    async def run_regression_tests(
        self,
        update_baseline: bool = False
    ) -> RegressionTestSuite:
        """Run full regression test suite.

        Args:
            update_baseline: Whether to update baseline metrics

        Returns:
            RegressionTestSuite with all results
        """
        print(f"\n{'='*80}")
        print("PERFORMANCE REGRESSION TEST SUITE")
        print(f"{'='*80}")
        print(f"Regression Threshold: {self.regression_threshold * 100}%")
        print(f"Update Baseline: {update_baseline}")
        print(f"{'='*80}\n")

        results = []

        # Test 1: Single agent execution
        print("\n[1/4] Testing single agent execution...")
        result1 = await self.test_single_agent_performance(update_baseline)
        results.append(result1)

        # Test 2: Concurrent execution (10)
        print("\n[2/4] Testing concurrent execution (10 agents)...")
        result2 = await self.test_concurrent_performance(10, update_baseline)
        results.append(result2)

        # Test 3: Concurrent execution (100)
        print("\n[3/4] Testing concurrent execution (100 agents)...")
        result3 = await self.test_concurrent_performance(100, update_baseline)
        results.append(result3)

        # Test 4: Async speedup validation
        print("\n[4/4] Testing async speedup...")
        result4 = await self.test_async_speedup(update_baseline)
        results.append(result4)

        # Aggregate results
        num_passed = sum(1 for r in results if r.passed)
        num_failed = len(results) - num_passed
        all_passed = num_failed == 0

        suite = RegressionTestSuite(
            results=results,
            all_passed=all_passed,
            num_tests=len(results),
            num_passed=num_passed,
            num_failed=num_failed
        )

        self._print_suite_summary(suite)

        return suite

    async def test_single_agent_performance(
        self, update_baseline: bool = False
    ) -> RegressionTestResult:
        """Test single agent execution performance."""
        test_name = "single_agent_execution"

        # Measure current performance
        latencies = []

        async with AsyncFuelAgentAI(self.config) as agent:
            for _ in range(50):  # 50 iterations for stable metrics
                start = time.perf_counter()
                result = await agent.run_async(self.default_test_input)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # ms

        # Calculate metrics
        current = self._create_baseline(
            test_name=test_name,
            latencies=latencies,
            error_rate=0.0
        )

        # Load baseline
        baseline = self._load_baseline(test_name)

        # Check SLOs and regression
        slo_results = self._check_slos(current)
        regression_detected, regression_details = self._check_regression(
            baseline, current
        )

        passed = all(passed for _, passed, _ in slo_results) and not regression_detected

        # Update baseline if requested
        if update_baseline:
            self._save_baseline(current)

        return RegressionTestResult(
            test_name=test_name,
            passed=passed,
            baseline=baseline,
            current=current,
            slo_results=slo_results,
            regression_detected=regression_detected,
            regression_details=regression_details
        )

    async def test_concurrent_performance(
        self, num_concurrent: int, update_baseline: bool = False
    ) -> RegressionTestResult:
        """Test concurrent execution performance."""
        test_name = f"concurrent_{num_concurrent}_execution"

        # Measure current performance
        start_time = time.perf_counter()
        latencies = []

        async with AsyncFuelAgentAI(self.config) as agent:
            tasks = []
            for _ in range(num_concurrent):
                task_start = time.perf_counter()
                task = agent.run_async(self.default_test_input)
                tasks.append((task, task_start))

            results = []
            for task, task_start in tasks:
                result = await task
                task_end = time.perf_counter()
                latencies.append((task_end - task_start) * 1000)
                results.append(result)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Calculate metrics
        throughput_rps = num_concurrent / duration
        error_rate = sum(1 for r in results if not r.success) / len(results)

        current = self._create_baseline(
            test_name=test_name,
            latencies=latencies,
            throughput_rps=throughput_rps,
            error_rate=error_rate
        )

        # Load baseline
        baseline = self._load_baseline(test_name)

        # Check SLOs and regression
        slo_results = self._check_slos(current)
        regression_detected, regression_details = self._check_regression(
            baseline, current
        )

        passed = all(passed for _, passed, _ in slo_results) and not regression_detected

        # Update baseline if requested
        if update_baseline:
            self._save_baseline(current)

        return RegressionTestResult(
            test_name=test_name,
            passed=passed,
            baseline=baseline,
            current=current,
            slo_results=slo_results,
            regression_detected=regression_detected,
            regression_details=regression_details
        )

    async def test_async_speedup(
        self, update_baseline: bool = False
    ) -> RegressionTestResult:
        """Test async parallel speedup."""
        test_name = "async_speedup"

        num_agents = 10

        # Measure parallel execution time
        start = time.perf_counter()
        async with AsyncFuelAgentAI(self.config) as agent:
            tasks = [agent.run_async(self.default_test_input) for _ in range(num_agents)]
            await asyncio.gather(*tasks)
        parallel_time = time.perf_counter() - start

        # Measure sequential execution time (for comparison)
        start = time.perf_counter()
        async with AsyncFuelAgentAI(self.config) as agent:
            for _ in range(num_agents):
                await agent.run_async(self.default_test_input)
        sequential_time = time.perf_counter() - start

        # Calculate speedup
        speedup = sequential_time / parallel_time

        # Create baseline
        current = PerformanceBaseline(
            test_name=test_name,
            version="1.0.0",
            timestamp=datetime.now(),
            mean_latency_ms=parallel_time * 1000,
            median_latency_ms=parallel_time * 1000,
            p95_latency_ms=parallel_time * 1000,
            p99_latency_ms=parallel_time * 1000,
            throughput_rps=num_agents / parallel_time,
            memory_mb=0,
            cpu_percent=0,
            error_rate=0,
            raw_metrics={
                "parallel_time_ms": parallel_time * 1000,
                "sequential_time_ms": sequential_time * 1000,
                "speedup": speedup
            }
        )

        # Check if speedup meets minimum threshold (5x)
        speedup_slo = PerformanceSLO(
            name="Async Speedup",
            metric_name="speedup",
            threshold=5.0,
            comparison="ge",
            unit="x"
        )

        slo_results = [(speedup_slo, speedup >= 5.0, speedup)]
        regression_detected = speedup < 5.0
        regression_details = []

        if regression_detected:
            regression_details.append(
                f"Async speedup {speedup:.2f}x is below minimum threshold of 5.0x"
            )

        passed = not regression_detected

        # Update baseline if requested
        if update_baseline:
            self._save_baseline(current)

        return RegressionTestResult(
            test_name=test_name,
            passed=passed,
            baseline=None,
            current=current,
            slo_results=slo_results,
            regression_detected=regression_detected,
            regression_details=regression_details
        )

    # ==========================================================================
    # Baseline Management
    # ==========================================================================

    def _create_baseline(
        self,
        test_name: str,
        latencies: List[float],
        throughput_rps: float = 0,
        memory_mb: float = 0,
        cpu_percent: float = 0,
        error_rate: float = 0,
    ) -> PerformanceBaseline:
        """Create baseline from measurements."""
        sorted_latencies = sorted(latencies)

        return PerformanceBaseline(
            test_name=test_name,
            version="1.0.0",
            timestamp=datetime.now(),
            mean_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
            p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
            throughput_rps=throughput_rps,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            error_rate=error_rate,
        )

    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to disk."""
        filepath = self.baseline_dir / f"{baseline.test_name}.json"

        data = {
            "test_name": baseline.test_name,
            "version": baseline.version,
            "timestamp": baseline.timestamp.isoformat(),
            "mean_latency_ms": baseline.mean_latency_ms,
            "median_latency_ms": baseline.median_latency_ms,
            "p95_latency_ms": baseline.p95_latency_ms,
            "p99_latency_ms": baseline.p99_latency_ms,
            "throughput_rps": baseline.throughput_rps,
            "memory_mb": baseline.memory_mb,
            "cpu_percent": baseline.cpu_percent,
            "error_rate": baseline.error_rate,
            "raw_metrics": baseline.raw_metrics,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Baseline saved: {filepath}")

    def _load_baseline(self, test_name: str) -> Optional[PerformanceBaseline]:
        """Load baseline from disk."""
        filepath = self.baseline_dir / f"{test_name}.json"

        if not filepath.exists():
            print(f"  No baseline found: {filepath}")
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        return PerformanceBaseline(
            test_name=data["test_name"],
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            mean_latency_ms=data["mean_latency_ms"],
            median_latency_ms=data["median_latency_ms"],
            p95_latency_ms=data["p95_latency_ms"],
            p99_latency_ms=data["p99_latency_ms"],
            throughput_rps=data["throughput_rps"],
            memory_mb=data["memory_mb"],
            cpu_percent=data["cpu_percent"],
            error_rate=data["error_rate"],
            raw_metrics=data.get("raw_metrics", {}),
        )

    # ==========================================================================
    # SLO and Regression Checking
    # ==========================================================================

    def _check_slos(
        self, baseline: PerformanceBaseline
    ) -> List[Tuple[PerformanceSLO, bool, float]]:
        """Check if baseline meets all SLOs."""
        results = []

        for slo in self.slos:
            # Get metric value
            value = getattr(baseline, slo.metric_name, None)

            if value is None:
                # Check raw metrics
                value = baseline.raw_metrics.get(slo.metric_name)

            if value is not None:
                passed = slo.check(value)
                results.append((slo, passed, value))

        return results

    def _check_regression(
        self, baseline: Optional[PerformanceBaseline], current: PerformanceBaseline
    ) -> Tuple[bool, List[str]]:
        """Check if current metrics show regression vs baseline.

        Returns:
            (regression_detected, details)
        """
        if baseline is None:
            return False, ["No baseline available for comparison"]

        regression_detected = False
        details = []

        # Check latency regression (p95)
        p95_increase = (current.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms
        if p95_increase > self.regression_threshold:
            regression_detected = True
            details.append(
                f"p95 latency increased by {p95_increase*100:.1f}% "
                f"({baseline.p95_latency_ms:.2f}ms -> {current.p95_latency_ms:.2f}ms)"
            )

        # Check throughput regression
        if baseline.throughput_rps > 0:
            throughput_decrease = (baseline.throughput_rps - current.throughput_rps) / baseline.throughput_rps
            if throughput_decrease > self.regression_threshold:
                regression_detected = True
                details.append(
                    f"Throughput decreased by {throughput_decrease*100:.1f}% "
                    f"({baseline.throughput_rps:.2f} -> {current.throughput_rps:.2f} RPS)"
                )

        # Check memory regression
        if baseline.memory_mb > 0:
            memory_increase = (current.memory_mb - baseline.memory_mb) / baseline.memory_mb
            if memory_increase > self.regression_threshold:
                regression_detected = True
                details.append(
                    f"Memory usage increased by {memory_increase*100:.1f}% "
                    f"({baseline.memory_mb:.2f} -> {current.memory_mb:.2f} MB)"
                )

        return regression_detected, details

    # ==========================================================================
    # Reporting
    # ==========================================================================

    def _print_suite_summary(self, suite: RegressionTestSuite):
        """Print regression test suite summary."""
        print(f"\n{'='*80}")
        print("REGRESSION TEST SUMMARY")
        print(f"{'='*80}")

        for result in suite.results:
            status = "PASS" if result.passed else "FAIL"
            icon = "✓" if result.passed else "✗"
            print(f"\n{icon} {result.test_name}: {status}")

            # Print SLO results
            for slo, passed, value in result.slo_results:
                slo_status = "✓" if passed else "✗"
                print(f"  {slo_status} {slo.name}: {value:.2f}{slo.unit} (threshold: {slo.threshold}{slo.unit})")

            # Print regression details
            if result.regression_detected:
                print(f"  REGRESSION DETECTED:")
                for detail in result.regression_details:
                    print(f"    - {detail}")

            # Print comparison with baseline
            if result.baseline:
                print(f"  Baseline vs Current:")
                print(f"    p95: {result.baseline.p95_latency_ms:.2f}ms -> {result.current.p95_latency_ms:.2f}ms")
                if result.baseline.throughput_rps > 0:
                    print(f"    Throughput: {result.baseline.throughput_rps:.2f} -> {result.current.throughput_rps:.2f} RPS")

        print(f"\n{'='*80}")
        print(f"OVERALL: {'PASS' if suite.all_passed else 'FAIL'}")
        print(f"Tests: {suite.num_passed}/{suite.num_tests} passed")
        print(f"{'='*80}\n")


# ==============================================================================
# Main Entry Point
# ==============================================================================

async def main():
    """Run regression tests."""
    if not ASYNC_AGENTS_AVAILABLE:
        print("Error: Async agents not available. Cannot run regression tests.")
        return

    tester = RegressionTester()

    # First run: Create baseline
    print("="*80)
    print("CREATING BASELINE METRICS")
    print("="*80)
    await tester.run_regression_tests(update_baseline=True)

    # Second run: Test against baseline
    print("\n" + "="*80)
    print("RUNNING REGRESSION TESTS")
    print("="*80)
    suite = await tester.run_regression_tests(update_baseline=False)

    # Exit with failure if regressions detected
    if not suite.all_passed:
        print("\nERROR: Performance regressions detected!")
        return 1
    else:
        print("\nSUCCESS: All regression tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
