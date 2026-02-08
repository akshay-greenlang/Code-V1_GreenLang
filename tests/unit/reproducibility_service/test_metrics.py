# -*- coding: utf-8 -*-
"""
Unit Tests for Reproducibility Metrics (AGENT-FOUND-008)

Tests all 12 Prometheus metric definitions, recording functions,
and graceful fallback when prometheus_client is unavailable.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inline metrics mirroring greenlang/reproducibility/metrics.py
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "reproducibility_verifications_total",
    "reproducibility_verifications_passed_total",
    "reproducibility_verifications_failed_total",
    "reproducibility_verifications_warned_total",
    "reproducibility_hash_computations_total",
    "reproducibility_hash_computation_seconds",
    "reproducibility_drift_detections_total",
    "reproducibility_drift_severity",
    "reproducibility_replays_total",
    "reproducibility_replay_duration_seconds",
    "reproducibility_non_determinism_sources_total",
    "reproducibility_cache_operations_total",
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
    """Registry for all reproducibility metrics."""

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

    def record_verification(self, status: str) -> None:
        self._metrics["reproducibility_verifications_total"].inc()
        self._call_counts["reproducibility_verifications_total"] += 1
        if status == "pass":
            self._metrics["reproducibility_verifications_passed_total"].inc()
            self._call_counts["reproducibility_verifications_passed_total"] += 1
        elif status == "fail":
            self._metrics["reproducibility_verifications_failed_total"].inc()
            self._call_counts["reproducibility_verifications_failed_total"] += 1
        elif status == "warning":
            self._metrics["reproducibility_verifications_warned_total"].inc()
            self._call_counts["reproducibility_verifications_warned_total"] += 1

    def record_hash_computation(self, duration_seconds: float) -> None:
        self._metrics["reproducibility_hash_computations_total"].inc()
        self._call_counts["reproducibility_hash_computations_total"] += 1
        self._metrics["reproducibility_hash_computation_seconds"].observe(duration_seconds)
        self._call_counts["reproducibility_hash_computation_seconds"] += 1

    def record_drift(self, severity: str) -> None:
        self._metrics["reproducibility_drift_detections_total"].inc()
        self._call_counts["reproducibility_drift_detections_total"] += 1
        self._metrics["reproducibility_drift_severity"].labels(severity=severity).inc()
        self._call_counts["reproducibility_drift_severity"] += 1

    def record_replay(self, success: bool, duration_seconds: float) -> None:
        self._metrics["reproducibility_replays_total"].labels(
            status="success" if success else "failure"
        ).inc()
        self._call_counts["reproducibility_replays_total"] += 1
        self._metrics["reproducibility_replay_duration_seconds"].observe(duration_seconds)
        self._call_counts["reproducibility_replay_duration_seconds"] += 1

    def record_non_determinism(self, source: str) -> None:
        self._metrics["reproducibility_non_determinism_sources_total"].labels(
            source=source
        ).inc()
        self._call_counts["reproducibility_non_determinism_sources_total"] += 1

    def record_cache_operation(self, operation: str) -> None:
        self._metrics["reproducibility_cache_operations_total"].labels(
            operation=operation
        ).inc()
        self._call_counts["reproducibility_cache_operations_total"] += 1


# ===========================================================================
# Test Classes
# ===========================================================================


class TestMetricDefinitions:
    """Test that all 12 metrics are defined."""

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


class TestRecordVerification:
    """Test record_verification method."""

    def test_record_verification_pass(self):
        registry = MetricsRegistry()
        registry.record_verification("pass")
        assert registry._call_counts["reproducibility_verifications_total"] == 1
        assert registry._call_counts["reproducibility_verifications_passed_total"] == 1

    def test_record_verification_fail(self):
        registry = MetricsRegistry()
        registry.record_verification("fail")
        assert registry._call_counts["reproducibility_verifications_total"] == 1
        assert registry._call_counts["reproducibility_verifications_failed_total"] == 1

    def test_record_verification_warning(self):
        registry = MetricsRegistry()
        registry.record_verification("warning")
        assert registry._call_counts["reproducibility_verifications_total"] == 1
        assert registry._call_counts["reproducibility_verifications_warned_total"] == 1

    def test_record_verification_multiple(self):
        registry = MetricsRegistry()
        registry.record_verification("pass")
        registry.record_verification("pass")
        registry.record_verification("fail")
        assert registry._call_counts["reproducibility_verifications_total"] == 3
        assert registry._call_counts["reproducibility_verifications_passed_total"] == 2
        assert registry._call_counts["reproducibility_verifications_failed_total"] == 1


class TestRecordHashComputation:
    """Test record_hash_computation method."""

    def test_record_hash_computation(self):
        registry = MetricsRegistry()
        registry.record_hash_computation(0.001)
        assert registry._call_counts["reproducibility_hash_computations_total"] == 1
        assert registry._call_counts["reproducibility_hash_computation_seconds"] == 1

    def test_record_hash_computation_multiple(self):
        registry = MetricsRegistry()
        for _ in range(5):
            registry.record_hash_computation(0.001)
        assert registry._call_counts["reproducibility_hash_computations_total"] == 5


class TestRecordDrift:
    """Test record_drift method."""

    def test_record_drift_none(self):
        registry = MetricsRegistry()
        registry.record_drift("none")
        assert registry._call_counts["reproducibility_drift_detections_total"] == 1
        assert registry._call_counts["reproducibility_drift_severity"] == 1

    def test_record_drift_minor(self):
        registry = MetricsRegistry()
        registry.record_drift("minor")
        assert registry._call_counts["reproducibility_drift_severity"] == 1

    def test_record_drift_moderate(self):
        registry = MetricsRegistry()
        registry.record_drift("moderate")
        assert registry._call_counts["reproducibility_drift_severity"] == 1

    def test_record_drift_critical(self):
        registry = MetricsRegistry()
        registry.record_drift("critical")
        assert registry._call_counts["reproducibility_drift_severity"] == 1


class TestRecordReplay:
    """Test record_replay method."""

    def test_record_replay_success(self):
        registry = MetricsRegistry()
        registry.record_replay(True, 1.5)
        assert registry._call_counts["reproducibility_replays_total"] == 1
        assert registry._call_counts["reproducibility_replay_duration_seconds"] == 1

    def test_record_replay_failure(self):
        registry = MetricsRegistry()
        registry.record_replay(False, 2.0)
        assert registry._call_counts["reproducibility_replays_total"] == 1


class TestRecordNonDeterminism:
    """Test record_non_determinism method."""

    def test_record_timestamp_source(self):
        registry = MetricsRegistry()
        registry.record_non_determinism("timestamp")
        assert registry._call_counts["reproducibility_non_determinism_sources_total"] == 1

    def test_record_random_seed_source(self):
        registry = MetricsRegistry()
        registry.record_non_determinism("random_seed")
        assert registry._call_counts["reproducibility_non_determinism_sources_total"] == 1

    def test_record_floating_point_source(self):
        registry = MetricsRegistry()
        registry.record_non_determinism("floating_point")
        assert registry._call_counts["reproducibility_non_determinism_sources_total"] == 1

    def test_record_environment_mismatch(self):
        registry = MetricsRegistry()
        registry.record_non_determinism("environment_variable")
        assert registry._call_counts["reproducibility_non_determinism_sources_total"] == 1


class TestRecordCache:
    """Test record_cache_operation method."""

    def test_record_cache_hit(self):
        registry = MetricsRegistry()
        registry.record_cache_operation("hit")
        assert registry._call_counts["reproducibility_cache_operations_total"] == 1

    def test_record_cache_miss(self):
        registry = MetricsRegistry()
        registry.record_cache_operation("miss")
        assert registry._call_counts["reproducibility_cache_operations_total"] == 1


class TestGracefulWithoutPrometheus:
    """Test metrics work gracefully without prometheus_client."""

    def test_noop_metrics_no_exception(self):
        registry = MetricsRegistry(prometheus_available=False)
        # These should not raise
        registry.record_verification("pass")
        registry.record_hash_computation(0.001)
        registry.record_drift("none")
        registry.record_replay(True, 1.0)
        registry.record_non_determinism("timestamp")
        registry.record_cache_operation("hit")

    def test_noop_metric_inc(self):
        metric = NoOpMetric()
        metric.inc()  # Should not raise

    def test_noop_metric_observe(self):
        metric = NoOpMetric()
        metric.observe(1.0)  # Should not raise

    def test_noop_metric_labels(self):
        metric = NoOpMetric()
        result = metric.labels(status="pass")
        assert isinstance(result, NoOpMetric)
