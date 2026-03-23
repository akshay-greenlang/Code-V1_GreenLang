# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-009: QA Test Harness

12 Prometheus metrics for QA test harness service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_qa_test_runs_total (Counter, labels: status, category)
    2.  gl_qa_test_duration_seconds (Histogram)
    3.  gl_qa_test_assertions_total (Counter, labels: result)
    4.  gl_qa_test_pass_rate (Gauge)
    5.  gl_qa_test_failures_total (Counter, labels: severity)
    6.  gl_qa_test_regressions_total (Counter)
    7.  gl_qa_golden_file_mismatches_total (Counter)
    8.  gl_qa_performance_threshold_breaches_total (Counter)
    9.  gl_qa_coverage_percent (Gauge, labels: agent_type)
    10. gl_qa_suites_total (Counter)
    11. gl_qa_cache_hits_total (Counter)
    12. gl_qa_cache_misses_total (Counter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; QA test harness metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Test runs count by status and category
    qa_test_runs_total = Counter(
        "gl_qa_test_runs_total",
        "Total QA test runs executed",
        labelnames=["status", "category"],
    )

    # 2. Test duration
    qa_test_duration_seconds = Histogram(
        "gl_qa_test_duration_seconds",
        "QA test execution duration in seconds",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # 3. Assertions by result
    qa_test_assertions_total = Counter(
        "gl_qa_test_assertions_total",
        "Total QA test assertions evaluated",
        labelnames=["result"],
    )

    # 4. Pass rate gauge
    qa_test_pass_rate = Gauge(
        "gl_qa_test_pass_rate",
        "Current QA test pass rate percentage",
    )

    # 5. Failures by severity
    qa_test_failures_total = Counter(
        "gl_qa_test_failures_total",
        "Total QA test failures by severity",
        labelnames=["severity"],
    )

    # 6. Regressions detected
    qa_test_regressions_total = Counter(
        "gl_qa_test_regressions_total",
        "Total regressions detected by QA test harness",
    )

    # 7. Golden file mismatches
    qa_golden_file_mismatches_total = Counter(
        "gl_qa_golden_file_mismatches_total",
        "Total golden file mismatches detected",
    )

    # 8. Performance threshold breaches
    qa_performance_threshold_breaches_total = Counter(
        "gl_qa_performance_threshold_breaches_total",
        "Total performance threshold breaches detected",
    )

    # 9. Coverage percent by agent type
    qa_coverage_percent = Gauge(
        "gl_qa_coverage_percent",
        "Current test coverage percentage per agent type",
        labelnames=["agent_type"],
    )

    # 10. Suites executed
    qa_suites_total = Counter(
        "gl_qa_suites_total",
        "Total test suites executed",
    )

    # 11. Cache hits
    qa_cache_hits_total = Counter(
        "gl_qa_cache_hits_total",
        "Total QA test harness cache hits",
    )

    # 12. Cache misses
    qa_cache_misses_total = Counter(
        "gl_qa_cache_misses_total",
        "Total QA test harness cache misses",
    )

else:
    # No-op placeholders
    qa_test_runs_total = None  # type: ignore[assignment]
    qa_test_duration_seconds = None  # type: ignore[assignment]
    qa_test_assertions_total = None  # type: ignore[assignment]
    qa_test_pass_rate = None  # type: ignore[assignment]
    qa_test_failures_total = None  # type: ignore[assignment]
    qa_test_regressions_total = None  # type: ignore[assignment]
    qa_golden_file_mismatches_total = None  # type: ignore[assignment]
    qa_performance_threshold_breaches_total = None  # type: ignore[assignment]
    qa_coverage_percent = None  # type: ignore[assignment]
    qa_suites_total = None  # type: ignore[assignment]
    qa_cache_hits_total = None  # type: ignore[assignment]
    qa_cache_misses_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_test_run(status: str, category: str, duration_seconds: float) -> None:
    """Record a test run with its status, category, and duration.

    Args:
        status: Test status (passed, failed, error, skipped, timeout).
        category: Test category (zero_hallucination, determinism, etc.).
        duration_seconds: Test duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qa_test_runs_total.labels(status=status, category=category).inc()
    qa_test_duration_seconds.observe(duration_seconds)


def record_assertion(result: str) -> None:
    """Record an assertion evaluation result.

    Args:
        result: Assertion result (passed, failed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qa_test_assertions_total.labels(result=result).inc()


def record_failure(severity: str) -> None:
    """Record a test failure by severity.

    Args:
        severity: Failure severity (critical, high, medium, low, info).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qa_test_failures_total.labels(severity=severity).inc()


def record_regression() -> None:
    """Record a regression detection."""
    if not PROMETHEUS_AVAILABLE:
        return
    qa_test_regressions_total.inc()


def record_golden_file_mismatch() -> None:
    """Record a golden file mismatch detection."""
    if not PROMETHEUS_AVAILABLE:
        return
    qa_golden_file_mismatches_total.inc()


def record_performance_breach() -> None:
    """Record a performance threshold breach."""
    if not PROMETHEUS_AVAILABLE:
        return
    qa_performance_threshold_breaches_total.inc()


def update_coverage(agent_type: str, coverage_pct: float) -> None:
    """Update the coverage percentage gauge for an agent type.

    Args:
        agent_type: Type of agent.
        coverage_pct: Coverage percentage (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qa_coverage_percent.labels(agent_type=agent_type).set(coverage_pct)


def update_pass_rate(pass_rate: float) -> None:
    """Update the global pass rate gauge.

    Args:
        pass_rate: Pass rate percentage (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qa_test_pass_rate.set(pass_rate)


def record_suite() -> None:
    """Record a test suite execution."""
    if not PROMETHEUS_AVAILABLE:
        return
    qa_suites_total.inc()


def record_cache_hit() -> None:
    """Record a cache hit."""
    if not PROMETHEUS_AVAILABLE:
        return
    qa_cache_hits_total.inc()


def record_cache_miss() -> None:
    """Record a cache miss."""
    if not PROMETHEUS_AVAILABLE:
        return
    qa_cache_misses_total.inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "qa_test_runs_total",
    "qa_test_duration_seconds",
    "qa_test_assertions_total",
    "qa_test_pass_rate",
    "qa_test_failures_total",
    "qa_test_regressions_total",
    "qa_golden_file_mismatches_total",
    "qa_performance_threshold_breaches_total",
    "qa_coverage_percent",
    "qa_suites_total",
    "qa_cache_hits_total",
    "qa_cache_misses_total",
    # Helper functions
    "record_test_run",
    "record_assertion",
    "record_failure",
    "record_regression",
    "record_golden_file_mismatch",
    "record_performance_breach",
    "update_coverage",
    "update_pass_rate",
    "record_suite",
    "record_cache_hit",
    "record_cache_miss",
]
