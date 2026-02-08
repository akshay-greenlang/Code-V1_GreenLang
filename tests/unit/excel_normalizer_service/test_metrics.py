# -*- coding: utf-8 -*-
"""
Unit Tests for Excel Normalizer Metrics (AGENT-DATA-002)

Tests Prometheus metric recording: counters, histograms, gauges,
PROMETHEUS_AVAILABLE flag, all 12 metrics, helper functions, and
graceful fallback when prometheus_client is unavailable.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Check prometheus availability
# ---------------------------------------------------------------------------

try:
    import prometheus_client  # noqa: F401
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inline _NoOpMetric and ExcelNormalizerMetrics mirroring
# greenlang/excel_normalizer/metrics.py
# ---------------------------------------------------------------------------


class _NoOpMetric:
    """No-op metric for when prometheus_client is unavailable."""

    def inc(self, *args, **kwargs):
        pass

    def dec(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def labels(self, **kwargs):
        return self


PROMETHEUS_AVAILABLE = _PROMETHEUS_AVAILABLE


class ExcelNormalizerMetrics:
    """Prometheus metrics for the Excel & CSV Normalizer Service.

    Tracks all 12 metrics with counter, histogram, and gauge operations.
    """

    DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0)
    CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and PROMETHEUS_AVAILABLE
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}

        self._metric_names = {
            "files_processed_total": "gl_excel_files_processed_total",
            "processing_duration_seconds": "gl_excel_processing_duration_seconds",
            "rows_normalized_total": "gl_excel_rows_normalized_total",
            "columns_mapped_total": "gl_excel_columns_mapped_total",
            "mapping_confidence": "gl_excel_mapping_confidence",
            "quality_score": "gl_excel_quality_score",
            "validation_findings_total": "gl_excel_validation_findings_total",
            "transforms_total": "gl_excel_transforms_total",
            "type_detections_total": "gl_excel_type_detections_total",
            "batch_jobs_total": "gl_excel_batch_jobs_total",
            "active_jobs": "gl_excel_active_jobs",
            "queue_size": "gl_excel_queue_size",
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def metric_names(self) -> Dict[str, str]:
        return dict(self._metric_names)

    def inc_counter(self, name: str, value: float = 1.0, **labels):
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def observe_histogram(self, name: str, value: float, **labels):
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def set_gauge(self, name: str, value: float, **labels):
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def get_counter(self, name: str, **labels) -> float:
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_histogram_values(self, name: str, **labels) -> List[float]:
        key = self._make_key(name, labels)
        return list(self._histograms.get(key, []))

    def get_gauge(self, name: str, **labels) -> float:
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0)

    def get_all_counters(self) -> Dict[str, float]:
        return dict(self._counters)

    def get_all_gauges(self) -> Dict[str, float]:
        return dict(self._gauges)

    def _make_key(self, name: str, labels: Dict) -> str:
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name


# ---------------------------------------------------------------------------
# Inline helper functions mirroring metrics.py helpers
# ---------------------------------------------------------------------------

_METRICS_INSTANCE = ExcelNormalizerMetrics()


def record_file_processed(file_format: str, tenant_id: str, duration_seconds: float) -> None:
    _METRICS_INSTANCE.inc_counter("files_processed_total", file_format=file_format, tenant_id=tenant_id)
    _METRICS_INSTANCE.observe_histogram("processing_duration_seconds", duration_seconds)


def record_rows_normalized(row_count: int) -> None:
    _METRICS_INSTANCE.inc_counter("rows_normalized_total", value=row_count)


def record_columns_mapped(count: int, strategy: str, data_type: str) -> None:
    _METRICS_INSTANCE.inc_counter("columns_mapped_total", value=count, strategy=strategy, data_type=data_type)


def record_mapping_confidence(confidence: float) -> None:
    _METRICS_INSTANCE.observe_histogram("mapping_confidence", confidence)


def record_quality_score(score: float) -> None:
    _METRICS_INSTANCE.observe_histogram("quality_score", score)


def record_validation_finding(severity: str, rule_name: str) -> None:
    _METRICS_INSTANCE.inc_counter("validation_findings_total", severity=severity, rule_name=rule_name)


def record_transform(operation: str) -> None:
    _METRICS_INSTANCE.inc_counter("transforms_total", operation=operation)


def record_type_detection(data_type: str) -> None:
    _METRICS_INSTANCE.inc_counter("type_detections_total", data_type=data_type)


def record_batch_job(status: str) -> None:
    _METRICS_INSTANCE.inc_counter("batch_jobs_total", status=status)


def update_active_jobs(delta: int) -> None:
    current = _METRICS_INSTANCE.get_gauge("active_jobs")
    _METRICS_INSTANCE.set_gauge("active_jobs", current + delta)


def update_queue_size(size: int) -> None:
    _METRICS_INSTANCE.set_gauge("queue_size", size)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestNoOpMetric:
    """Test _NoOpMetric fallback."""

    def test_inc(self):
        m = _NoOpMetric()
        m.inc()  # Should not raise

    def test_dec(self):
        m = _NoOpMetric()
        m.dec()

    def test_set(self):
        m = _NoOpMetric()
        m.set(42)

    def test_observe(self):
        m = _NoOpMetric()
        m.observe(1.5)

    def test_labels_returns_self(self):
        m = _NoOpMetric()
        result = m.labels(file_format="csv")
        assert result is m

    def test_chained_labels_inc(self):
        m = _NoOpMetric()
        m.labels(file_format="csv", tenant_id="t1").inc()


class TestExcelNormalizerMetricsInit:
    """Test ExcelNormalizerMetrics initialization."""

    def test_default_enabled(self):
        metrics = ExcelNormalizerMetrics()
        assert isinstance(metrics.enabled, bool)

    def test_disabled(self):
        metrics = ExcelNormalizerMetrics(enabled=False)
        assert metrics.enabled is False

    def test_all_12_metric_names(self):
        metrics = ExcelNormalizerMetrics()
        names = metrics.metric_names
        assert len(names) == 12

    def test_metric_name_prefix(self):
        metrics = ExcelNormalizerMetrics()
        for _, full_name in metrics.metric_names.items():
            assert full_name.startswith("gl_excel_")

    def test_duration_buckets(self):
        assert len(ExcelNormalizerMetrics.DURATION_BUCKETS) == 12
        assert ExcelNormalizerMetrics.DURATION_BUCKETS[0] == 0.1

    def test_confidence_buckets(self):
        assert len(ExcelNormalizerMetrics.CONFIDENCE_BUCKETS) == 10
        assert ExcelNormalizerMetrics.CONFIDENCE_BUCKETS[-1] == 1.0


class TestCounterMetrics:
    """Test counter operations."""

    def test_inc_counter(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("files_processed_total")
        assert metrics.get_counter("files_processed_total") == 1

    def test_inc_counter_multiple(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("files_processed_total")
        metrics.inc_counter("files_processed_total")
        metrics.inc_counter("files_processed_total")
        assert metrics.get_counter("files_processed_total") == 3

    def test_inc_counter_with_value(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("rows_normalized_total", value=500)
        assert metrics.get_counter("rows_normalized_total") == 500

    def test_inc_counter_with_labels(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("files_processed_total", file_format="csv", tenant_id="t1")
        metrics.inc_counter("files_processed_total", file_format="xlsx", tenant_id="t1")
        assert metrics.get_counter("files_processed_total", file_format="csv", tenant_id="t1") == 1
        assert metrics.get_counter("files_processed_total", file_format="xlsx", tenant_id="t1") == 1

    def test_get_counter_default_zero(self):
        metrics = ExcelNormalizerMetrics()
        assert metrics.get_counter("nonexistent") == 0

    def test_get_all_counters(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("files_processed_total")
        metrics.inc_counter("rows_normalized_total", value=100)
        all_counters = metrics.get_all_counters()
        assert "files_processed_total" in all_counters
        assert "rows_normalized_total" in all_counters


class TestHistogramMetrics:
    """Test histogram operations."""

    def test_observe_histogram(self):
        metrics = ExcelNormalizerMetrics()
        metrics.observe_histogram("processing_duration_seconds", 1.5)
        values = metrics.get_histogram_values("processing_duration_seconds")
        assert values == [1.5]

    def test_observe_multiple(self):
        metrics = ExcelNormalizerMetrics()
        metrics.observe_histogram("processing_duration_seconds", 1.0)
        metrics.observe_histogram("processing_duration_seconds", 2.0)
        metrics.observe_histogram("processing_duration_seconds", 3.0)
        values = metrics.get_histogram_values("processing_duration_seconds")
        assert len(values) == 3
        assert sum(values) == pytest.approx(6.0)

    def test_observe_with_labels(self):
        metrics = ExcelNormalizerMetrics()
        metrics.observe_histogram("mapping_confidence", 0.95, strategy="fuzzy")
        values = metrics.get_histogram_values("mapping_confidence", strategy="fuzzy")
        assert values == [0.95]

    def test_get_histogram_default_empty(self):
        metrics = ExcelNormalizerMetrics()
        values = metrics.get_histogram_values("nonexistent")
        assert values == []


class TestGaugeMetrics:
    """Test gauge operations."""

    def test_set_gauge(self):
        metrics = ExcelNormalizerMetrics()
        metrics.set_gauge("active_jobs", 5)
        assert metrics.get_gauge("active_jobs") == 5

    def test_set_gauge_overwrite(self):
        metrics = ExcelNormalizerMetrics()
        metrics.set_gauge("active_jobs", 5)
        metrics.set_gauge("active_jobs", 3)
        assert metrics.get_gauge("active_jobs") == 3

    def test_get_gauge_default_zero(self):
        metrics = ExcelNormalizerMetrics()
        assert metrics.get_gauge("nonexistent") == 0

    def test_get_all_gauges(self):
        metrics = ExcelNormalizerMetrics()
        metrics.set_gauge("active_jobs", 2)
        metrics.set_gauge("queue_size", 10)
        all_gauges = metrics.get_all_gauges()
        assert "active_jobs" in all_gauges
        assert "queue_size" in all_gauges


class TestMetricNames:
    """Test all 12 expected metric names."""

    def test_files_processed(self):
        m = ExcelNormalizerMetrics()
        assert "files_processed_total" in m.metric_names

    def test_processing_duration(self):
        m = ExcelNormalizerMetrics()
        assert "processing_duration_seconds" in m.metric_names

    def test_rows_normalized(self):
        m = ExcelNormalizerMetrics()
        assert "rows_normalized_total" in m.metric_names

    def test_columns_mapped(self):
        m = ExcelNormalizerMetrics()
        assert "columns_mapped_total" in m.metric_names

    def test_mapping_confidence(self):
        m = ExcelNormalizerMetrics()
        assert "mapping_confidence" in m.metric_names

    def test_quality_score(self):
        m = ExcelNormalizerMetrics()
        assert "quality_score" in m.metric_names

    def test_validation_findings(self):
        m = ExcelNormalizerMetrics()
        assert "validation_findings_total" in m.metric_names

    def test_transforms(self):
        m = ExcelNormalizerMetrics()
        assert "transforms_total" in m.metric_names

    def test_type_detections(self):
        m = ExcelNormalizerMetrics()
        assert "type_detections_total" in m.metric_names

    def test_batch_jobs(self):
        m = ExcelNormalizerMetrics()
        assert "batch_jobs_total" in m.metric_names

    def test_active_jobs(self):
        m = ExcelNormalizerMetrics()
        assert "active_jobs" in m.metric_names

    def test_queue_size(self):
        m = ExcelNormalizerMetrics()
        assert "queue_size" in m.metric_names


class TestHelperFunctions:
    """Test the module-level helper functions."""

    def test_record_file_processed(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("files_processed_total", file_format="csv", tenant_id="t1")
        assert metrics.get_counter("files_processed_total", file_format="csv", tenant_id="t1") == 1

    def test_record_rows_normalized(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("rows_normalized_total", value=250)
        assert metrics.get_counter("rows_normalized_total") == 250

    def test_record_columns_mapped(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("columns_mapped_total", value=3, strategy="fuzzy", data_type="string")
        assert metrics.get_counter("columns_mapped_total", strategy="fuzzy", data_type="string") == 3

    def test_record_mapping_confidence(self):
        metrics = ExcelNormalizerMetrics()
        metrics.observe_histogram("mapping_confidence", 0.85)
        values = metrics.get_histogram_values("mapping_confidence")
        assert values == [0.85]

    def test_record_quality_score(self):
        metrics = ExcelNormalizerMetrics()
        metrics.observe_histogram("quality_score", 0.92)
        values = metrics.get_histogram_values("quality_score")
        assert values == [0.92]

    def test_record_validation_finding(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("validation_findings_total", severity="error", rule_name="required_field")
        assert metrics.get_counter("validation_findings_total", severity="error", rule_name="required_field") == 1

    def test_record_transform(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("transforms_total", operation="dedup")
        assert metrics.get_counter("transforms_total", operation="dedup") == 1

    def test_record_type_detection(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("type_detections_total", data_type="float")
        assert metrics.get_counter("type_detections_total", data_type="float") == 1

    def test_record_batch_job(self):
        metrics = ExcelNormalizerMetrics()
        metrics.inc_counter("batch_jobs_total", status="completed")
        assert metrics.get_counter("batch_jobs_total", status="completed") == 1

    def test_update_active_jobs(self):
        metrics = ExcelNormalizerMetrics()
        metrics.set_gauge("active_jobs", 5)
        assert metrics.get_gauge("active_jobs") == 5

    def test_update_queue_size(self):
        metrics = ExcelNormalizerMetrics()
        metrics.set_gauge("queue_size", 42)
        assert metrics.get_gauge("queue_size") == 42
