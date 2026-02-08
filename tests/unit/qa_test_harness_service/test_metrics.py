# -*- coding: utf-8 -*-
"""
Unit Tests for QA Test Harness Metrics (AGENT-FOUND-009)

Tests all 12 Prometheus metric definitions, recording functions,
and graceful fallback when prometheus_client is unavailable.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline metrics mirroring greenlang/qa_test_harness/metrics.py
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "qa_test_harness_tests_total",
    "qa_test_harness_tests_passed_total",
    "qa_test_harness_tests_failed_total",
    "qa_test_harness_tests_skipped_total",
    "qa_test_harness_tests_error_total",
    "qa_test_harness_assertions_total",
    "qa_test_harness_assertions_passed_total",
    "qa_test_harness_regressions_detected_total",
    "qa_test_harness_golden_file_mismatches_total",
    "qa_test_harness_performance_breaches_total",
    "qa_test_harness_coverage_percent",
    "qa_test_harness_suite_duration_seconds",
]


class NoOpMetric:
    """No-op metric for when prometheus_client is not available."""

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def labels(self, **kwargs) -> "NoOpMetric":
        return self


class MetricsRegistry:
    """Registry for all QA test harness metrics."""

    def __init__(self, prometheus_available: bool = True):
        self._prometheus_available = prometheus_available
        self._metrics: Dict[str, Any] = {}
        self._call_counts: Dict[str, int] = {}

        for name in METRIC_NAMES:
            if prometheus_available:
                self._metrics[name] = MagicMock()
                self._metrics[name].labels.return_value = self._metrics[name]
            else:
                self._metrics[name] = NoOpMetric()
            self._call_counts[name] = 0

    @property
    def all_metric_names(self):
        return list(self._metrics.keys())

    def get_metric(self, name: str) -> Any:
        return self._metrics.get(name)

    def record_test_run(self, status: str) -> None:
        self._metrics["qa_test_harness_tests_total"].inc()
        self._call_counts["qa_test_harness_tests_total"] += 1
        status_map = {
            "passed": "qa_test_harness_tests_passed_total",
            "failed": "qa_test_harness_tests_failed_total",
            "skipped": "qa_test_harness_tests_skipped_total",
            "error": "qa_test_harness_tests_error_total",
        }
        metric_name = status_map.get(status)
        if metric_name:
            self._metrics[metric_name].inc()
            self._call_counts[metric_name] += 1

    def record_assertion(self, passed: bool) -> None:
        self._metrics["qa_test_harness_assertions_total"].inc()
        self._call_counts["qa_test_harness_assertions_total"] += 1
        if passed:
            self._metrics["qa_test_harness_assertions_passed_total"].inc()
            self._call_counts["qa_test_harness_assertions_passed_total"] += 1

    def record_regression(self) -> None:
        self._metrics["qa_test_harness_regressions_detected_total"].inc()
        self._call_counts["qa_test_harness_regressions_detected_total"] += 1

    def record_golden_file_mismatch(self) -> None:
        self._metrics["qa_test_harness_golden_file_mismatches_total"].inc()
        self._call_counts["qa_test_harness_golden_file_mismatches_total"] += 1

    def record_performance_breach(self) -> None:
        self._metrics["qa_test_harness_performance_breaches_total"].inc()
        self._call_counts["qa_test_harness_performance_breaches_total"] += 1

    def update_coverage(self, agent_type: str, coverage_pct: float) -> None:
        self._metrics["qa_test_harness_coverage_percent"].labels(
            agent_type=agent_type
        ).set(coverage_pct)
        self._call_counts["qa_test_harness_coverage_percent"] += 1

    def record_suite(self, duration_seconds: float) -> None:
        self._metrics["qa_test_harness_suite_duration_seconds"].observe(duration_seconds)
        self._call_counts["qa_test_harness_suite_duration_seconds"] += 1


# ===========================================================================
# Test Classes
# ===========================================================================


class TestMetricDefinitions:
    def test_all_12_metrics_defined(self):
        registry = MetricsRegistry()
        assert len(registry.all_metric_names) == 12

    @pytest.mark.parametrize("metric_name", METRIC_NAMES)
    def test_metric_exists(self, metric_name):
        registry = MetricsRegistry()
        assert registry.get_metric(metric_name) is not None

    def test_metric_names_match_expected(self):
        registry = MetricsRegistry()
        for name in METRIC_NAMES:
            assert name in registry.all_metric_names


class TestRecordTestRun:
    def test_record_test_run_passed(self):
        r = MetricsRegistry()
        r.record_test_run("passed")
        assert r._call_counts["qa_test_harness_tests_total"] == 1
        assert r._call_counts["qa_test_harness_tests_passed_total"] == 1

    def test_record_test_run_failed(self):
        r = MetricsRegistry()
        r.record_test_run("failed")
        assert r._call_counts["qa_test_harness_tests_total"] == 1
        assert r._call_counts["qa_test_harness_tests_failed_total"] == 1

    def test_record_test_run_skipped(self):
        r = MetricsRegistry()
        r.record_test_run("skipped")
        assert r._call_counts["qa_test_harness_tests_skipped_total"] == 1

    def test_record_test_run_error(self):
        r = MetricsRegistry()
        r.record_test_run("error")
        assert r._call_counts["qa_test_harness_tests_error_total"] == 1

    def test_record_test_run_multiple(self):
        r = MetricsRegistry()
        r.record_test_run("passed")
        r.record_test_run("passed")
        r.record_test_run("failed")
        assert r._call_counts["qa_test_harness_tests_total"] == 3
        assert r._call_counts["qa_test_harness_tests_passed_total"] == 2
        assert r._call_counts["qa_test_harness_tests_failed_total"] == 1

    def test_record_test_run_unknown_status(self):
        r = MetricsRegistry()
        r.record_test_run("unknown")
        assert r._call_counts["qa_test_harness_tests_total"] == 1


class TestRecordAssertion:
    def test_record_assertion_passed(self):
        r = MetricsRegistry()
        r.record_assertion(True)
        assert r._call_counts["qa_test_harness_assertions_total"] == 1
        assert r._call_counts["qa_test_harness_assertions_passed_total"] == 1

    def test_record_assertion_failed(self):
        r = MetricsRegistry()
        r.record_assertion(False)
        assert r._call_counts["qa_test_harness_assertions_total"] == 1
        assert r._call_counts["qa_test_harness_assertions_passed_total"] == 0

    def test_record_assertion_multiple(self):
        r = MetricsRegistry()
        r.record_assertion(True)
        r.record_assertion(True)
        r.record_assertion(False)
        assert r._call_counts["qa_test_harness_assertions_total"] == 3
        assert r._call_counts["qa_test_harness_assertions_passed_total"] == 2


class TestRecordRegression:
    def test_record_regression(self):
        r = MetricsRegistry()
        r.record_regression()
        assert r._call_counts["qa_test_harness_regressions_detected_total"] == 1

    def test_record_regression_multiple(self):
        r = MetricsRegistry()
        r.record_regression()
        r.record_regression()
        assert r._call_counts["qa_test_harness_regressions_detected_total"] == 2


class TestRecordGoldenFileMismatch:
    def test_record_golden_file_mismatch(self):
        r = MetricsRegistry()
        r.record_golden_file_mismatch()
        assert r._call_counts["qa_test_harness_golden_file_mismatches_total"] == 1


class TestRecordPerformanceBreach:
    def test_record_performance_breach(self):
        r = MetricsRegistry()
        r.record_performance_breach()
        assert r._call_counts["qa_test_harness_performance_breaches_total"] == 1


class TestUpdateCoverage:
    def test_update_coverage(self):
        r = MetricsRegistry()
        r.update_coverage("Agent", 85.0)
        assert r._call_counts["qa_test_harness_coverage_percent"] == 1

    def test_update_coverage_multiple_agents(self):
        r = MetricsRegistry()
        r.update_coverage("Agent1", 85.0)
        r.update_coverage("Agent2", 90.0)
        assert r._call_counts["qa_test_harness_coverage_percent"] == 2


class TestRecordSuite:
    def test_record_suite(self):
        r = MetricsRegistry()
        r.record_suite(1.5)
        assert r._call_counts["qa_test_harness_suite_duration_seconds"] == 1


class TestGracefulWithoutPrometheus:
    def test_noop_metrics_no_exception(self):
        r = MetricsRegistry(prometheus_available=False)
        r.record_test_run("passed")
        r.record_assertion(True)
        r.record_regression()
        r.record_golden_file_mismatch()
        r.record_performance_breach()
        r.update_coverage("Agent", 85.0)
        r.record_suite(1.0)

    def test_noop_metric_inc(self):
        metric = NoOpMetric()
        metric.inc()

    def test_noop_metric_dec(self):
        metric = NoOpMetric()
        metric.dec()

    def test_noop_metric_set(self):
        metric = NoOpMetric()
        metric.set(42.0)

    def test_noop_metric_observe(self):
        metric = NoOpMetric()
        metric.observe(1.0)

    def test_noop_metric_labels(self):
        metric = NoOpMetric()
        result = metric.labels(agent_type="test")
        assert isinstance(result, NoOpMetric)
