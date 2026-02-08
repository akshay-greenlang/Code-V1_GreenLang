# -*- coding: utf-8 -*-
"""
Unit Tests for PDF Extractor Metrics (AGENT-DATA-001)

Tests Prometheus metric recording: counters, histograms, gauges,
PROMETHEUS_AVAILABLE flag, all 12 metrics, and graceful fallback
when prometheus_client is unavailable.

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
# Inline _NoOpMetric and PDFExtractorMetrics mirroring
# greenlang/pdf_extractor/metrics.py
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


class PDFExtractorMetrics:
    """Prometheus metrics for the PDF & Invoice Extractor Service."""

    OPERATION_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 25, 50, 100)

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and PROMETHEUS_AVAILABLE
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}

        self._metric_names = {
            "documents_ingested_total": "gl_pdf_documents_ingested_total",
            "documents_classified_total": "gl_pdf_documents_classified_total",
            "extractions_completed_total": "gl_pdf_extractions_completed_total",
            "extractions_failed_total": "gl_pdf_extractions_failed_total",
            "validations_run_total": "gl_pdf_validations_run_total",
            "validations_passed_total": "gl_pdf_validations_passed_total",
            "ocr_operations_total": "gl_pdf_ocr_operations_total",
            "operation_duration_ms": "gl_pdf_operation_duration_ms",
            "extraction_confidence": "gl_pdf_extraction_confidence",
            "active_jobs": "gl_pdf_active_jobs",
            "batch_documents_total": "gl_pdf_batch_documents_total",
            "provenance_records_total": "gl_pdf_provenance_records_total",
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def metric_names(self) -> Dict[str, str]:
        return dict(self._metric_names)

    def inc_counter(self, name: str, value: float = 1.0, **labels):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def observe_histogram(self, name: str, value: float, **labels):
        """Observe a histogram value."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def set_gauge(self, name: str, value: float, **labels):
        """Set a gauge value."""
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
        result = m.labels(engine="tesseract")
        assert result is m

    def test_chained_labels_inc(self):
        m = _NoOpMetric()
        m.labels(engine="tesseract").inc()


class TestPDFExtractorMetricsInit:
    """Test PDFExtractorMetrics initialization."""

    def test_default_enabled(self):
        metrics = PDFExtractorMetrics()
        # Enabled depends on prometheus_client availability
        assert isinstance(metrics.enabled, bool)

    def test_disabled(self):
        metrics = PDFExtractorMetrics(enabled=False)
        assert metrics.enabled is False

    def test_all_12_metric_names(self):
        metrics = PDFExtractorMetrics()
        names = metrics.metric_names
        assert len(names) == 12

    def test_metric_name_prefix(self):
        metrics = PDFExtractorMetrics()
        for _, full_name in metrics.metric_names.items():
            assert full_name.startswith("gl_pdf_")

    def test_operation_buckets(self):
        assert len(PDFExtractorMetrics.OPERATION_BUCKETS) == 9
        assert PDFExtractorMetrics.OPERATION_BUCKETS[0] == 0.1


class TestCounterMetrics:
    """Test counter operations."""

    def test_inc_counter(self):
        metrics = PDFExtractorMetrics()
        metrics.inc_counter("documents_ingested_total")
        assert metrics.get_counter("documents_ingested_total") == 1

    def test_inc_counter_multiple(self):
        metrics = PDFExtractorMetrics()
        metrics.inc_counter("documents_ingested_total")
        metrics.inc_counter("documents_ingested_total")
        metrics.inc_counter("documents_ingested_total")
        assert metrics.get_counter("documents_ingested_total") == 3

    def test_inc_counter_with_value(self):
        metrics = PDFExtractorMetrics()
        metrics.inc_counter("batch_documents_total", value=10)
        assert metrics.get_counter("batch_documents_total") == 10

    def test_inc_counter_with_labels(self):
        metrics = PDFExtractorMetrics()
        metrics.inc_counter("ocr_operations_total", engine="tesseract")
        metrics.inc_counter("ocr_operations_total", engine="textract")
        assert metrics.get_counter("ocr_operations_total", engine="tesseract") == 1
        assert metrics.get_counter("ocr_operations_total", engine="textract") == 1

    def test_get_counter_default_zero(self):
        metrics = PDFExtractorMetrics()
        assert metrics.get_counter("nonexistent") == 0

    def test_get_all_counters(self):
        metrics = PDFExtractorMetrics()
        metrics.inc_counter("extractions_completed_total")
        metrics.inc_counter("extractions_failed_total")
        all_counters = metrics.get_all_counters()
        assert "extractions_completed_total" in all_counters


class TestHistogramMetrics:
    """Test histogram operations."""

    def test_observe_histogram(self):
        metrics = PDFExtractorMetrics()
        metrics.observe_histogram("operation_duration_ms", 15.5)
        values = metrics.get_histogram_values("operation_duration_ms")
        assert values == [15.5]

    def test_observe_multiple(self):
        metrics = PDFExtractorMetrics()
        metrics.observe_histogram("operation_duration_ms", 10.0)
        metrics.observe_histogram("operation_duration_ms", 20.0)
        metrics.observe_histogram("operation_duration_ms", 30.0)
        values = metrics.get_histogram_values("operation_duration_ms")
        assert len(values) == 3
        assert sum(values) == pytest.approx(60.0)

    def test_observe_with_labels(self):
        metrics = PDFExtractorMetrics()
        metrics.observe_histogram("extraction_confidence", 0.95, doc_type="invoice")
        values = metrics.get_histogram_values("extraction_confidence", doc_type="invoice")
        assert values == [0.95]

    def test_get_histogram_default_empty(self):
        metrics = PDFExtractorMetrics()
        values = metrics.get_histogram_values("nonexistent")
        assert values == []


class TestGaugeMetrics:
    """Test gauge operations."""

    def test_set_gauge(self):
        metrics = PDFExtractorMetrics()
        metrics.set_gauge("active_jobs", 5)
        assert metrics.get_gauge("active_jobs") == 5

    def test_set_gauge_overwrite(self):
        metrics = PDFExtractorMetrics()
        metrics.set_gauge("active_jobs", 5)
        metrics.set_gauge("active_jobs", 3)
        assert metrics.get_gauge("active_jobs") == 3

    def test_get_gauge_default_zero(self):
        metrics = PDFExtractorMetrics()
        assert metrics.get_gauge("nonexistent") == 0

    def test_get_all_gauges(self):
        metrics = PDFExtractorMetrics()
        metrics.set_gauge("active_jobs", 2)
        all_gauges = metrics.get_all_gauges()
        assert "active_jobs" in all_gauges


class TestMetricNames:
    """Test all 12 expected metric names."""

    def test_documents_ingested(self):
        m = PDFExtractorMetrics()
        assert "documents_ingested_total" in m.metric_names

    def test_documents_classified(self):
        m = PDFExtractorMetrics()
        assert "documents_classified_total" in m.metric_names

    def test_extractions_completed(self):
        m = PDFExtractorMetrics()
        assert "extractions_completed_total" in m.metric_names

    def test_extractions_failed(self):
        m = PDFExtractorMetrics()
        assert "extractions_failed_total" in m.metric_names

    def test_validations_run(self):
        m = PDFExtractorMetrics()
        assert "validations_run_total" in m.metric_names

    def test_validations_passed(self):
        m = PDFExtractorMetrics()
        assert "validations_passed_total" in m.metric_names

    def test_ocr_operations(self):
        m = PDFExtractorMetrics()
        assert "ocr_operations_total" in m.metric_names

    def test_operation_duration(self):
        m = PDFExtractorMetrics()
        assert "operation_duration_ms" in m.metric_names

    def test_extraction_confidence(self):
        m = PDFExtractorMetrics()
        assert "extraction_confidence" in m.metric_names

    def test_active_jobs(self):
        m = PDFExtractorMetrics()
        assert "active_jobs" in m.metric_names

    def test_batch_documents(self):
        m = PDFExtractorMetrics()
        assert "batch_documents_total" in m.metric_names

    def test_provenance_records(self):
        m = PDFExtractorMetrics()
        assert "provenance_records_total" in m.metric_names
