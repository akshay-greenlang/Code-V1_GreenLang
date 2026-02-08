# -*- coding: utf-8 -*-
"""
Unit Tests for MetricsCollector (AGENT-FOUND-010)

Tests metric registration, recording (counter/gauge/histogram/summary),
query operations, Prometheus export, provenance hashing, statistics,
and error handling.

Coverage target: 85%+ of metrics_collector.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from greenlang.observability_agent.metrics_collector import (
    DEFAULT_HISTOGRAM_BUCKETS,
    MetricDefinition,
    MetricRecording,
    MetricsCollector,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    """Minimal config stub for MetricsCollector."""
    max_series: int = 50000
    metric_retention_seconds: int = 86400


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def collector(config):
    return MetricsCollector(config)


# ==========================================================================
# Registration Tests
# ==========================================================================

class TestMetricsCollectorRegistration:
    """Tests for register_metric."""

    def test_register_counter(self, collector):
        defn = collector.register_metric("requests_total", "counter", "Total requests")
        assert defn.name == "requests_total"
        assert defn.metric_type == "counter"
        assert defn.description == "Total requests"

    def test_register_gauge(self, collector):
        defn = collector.register_metric("temperature", "gauge")
        assert defn.metric_type == "gauge"

    def test_register_histogram(self, collector):
        defn = collector.register_metric("latency", "histogram")
        assert defn.metric_type == "histogram"
        assert defn.buckets == DEFAULT_HISTOGRAM_BUCKETS

    def test_register_summary(self, collector):
        defn = collector.register_metric("duration", "summary")
        assert defn.metric_type == "summary"

    def test_register_with_labels(self, collector):
        defn = collector.register_metric(
            "http_requests", "counter", labels=["method", "status"],
        )
        assert defn.labels == ["method", "status"]

    def test_register_with_custom_buckets(self, collector):
        custom = (0.1, 0.5, 1.0, 5.0)
        defn = collector.register_metric("custom_hist", "histogram", buckets=custom)
        assert defn.buckets == custom

    def test_register_duplicate_raises(self, collector):
        collector.register_metric("dup", "counter")
        with pytest.raises(ValueError, match="already registered"):
            collector.register_metric("dup", "counter")

    def test_register_empty_name_raises(self, collector):
        with pytest.raises(ValueError, match="non-empty"):
            collector.register_metric("", "counter")

    def test_register_whitespace_name_raises(self, collector):
        with pytest.raises(ValueError, match="non-empty"):
            collector.register_metric("   ", "counter")

    def test_register_invalid_type_raises(self, collector):
        with pytest.raises(ValueError, match="Invalid metric_type"):
            collector.register_metric("bad", "invalid_type")

    def test_register_returns_metric_definition(self, collector):
        defn = collector.register_metric("m", "counter")
        assert isinstance(defn, MetricDefinition)
        assert defn.metric_id  # UUID generated


# ==========================================================================
# Counter Recording Tests
# ==========================================================================

class TestMetricsCollectorCounter:
    """Tests for counter metric operations."""

    def test_increment_counter(self, collector):
        collector.register_metric("req", "counter")
        rec = collector.increment("req")
        assert isinstance(rec, MetricRecording)
        assert rec.metric_name == "req"
        assert rec.value == 1.0

    def test_increment_counter_accumulates(self, collector):
        collector.register_metric("count", "counter")
        collector.increment("count")
        collector.increment("count")
        collector.increment("count", amount=5.0)
        val = collector.get_metric_value("count")
        assert val == pytest.approx(7.0)

    def test_increment_with_labels(self, collector):
        collector.register_metric("http", "counter", labels=["method"])
        collector.increment("http", labels={"method": "GET"})
        collector.increment("http", labels={"method": "POST"})
        val_get = collector.get_metric_value("http", {"method": "GET"})
        val_post = collector.get_metric_value("http", {"method": "POST"})
        assert val_get == pytest.approx(1.0)
        assert val_post == pytest.approx(1.0)

    def test_increment_negative_amount_raises(self, collector):
        collector.register_metric("cnt", "counter")
        with pytest.raises(ValueError, match="non-negative"):
            collector.increment("cnt", amount=-1.0)

    def test_increment_non_counter_raises(self, collector):
        collector.register_metric("g", "gauge")
        with pytest.raises(ValueError, match="counter"):
            collector.increment("g")

    def test_increment_unregistered_raises(self, collector):
        with pytest.raises(ValueError, match="not registered"):
            collector.increment("nonexistent")


# ==========================================================================
# Gauge Recording Tests
# ==========================================================================

class TestMetricsCollectorGauge:
    """Tests for gauge metric operations."""

    def test_set_gauge(self, collector):
        collector.register_metric("temp", "gauge")
        rec = collector.set_gauge("temp", 42.0)
        assert rec.metric_name == "temp"
        assert rec.value == 42.0

    def test_set_gauge_overwrite(self, collector):
        collector.register_metric("g", "gauge")
        collector.set_gauge("g", 10.0)
        collector.set_gauge("g", 20.0)
        val = collector.get_metric_value("g")
        assert val == pytest.approx(20.0)

    def test_set_gauge_non_gauge_raises(self, collector):
        collector.register_metric("c", "counter")
        with pytest.raises(ValueError, match="gauge"):
            collector.set_gauge("c", 1.0)

    def test_set_gauge_with_labels(self, collector):
        collector.register_metric("cpu", "gauge", labels=["host"])
        collector.set_gauge("cpu", 0.5, {"host": "web1"})
        collector.set_gauge("cpu", 0.7, {"host": "web2"})
        assert collector.get_metric_value("cpu", {"host": "web1"}) == pytest.approx(0.5)
        assert collector.get_metric_value("cpu", {"host": "web2"}) == pytest.approx(0.7)


# ==========================================================================
# Histogram Recording Tests
# ==========================================================================

class TestMetricsCollectorHistogram:
    """Tests for histogram metric operations."""

    def test_observe_histogram(self, collector):
        collector.register_metric("lat", "histogram")
        rec = collector.observe_histogram("lat", 0.35)
        assert rec.metric_name == "lat"
        assert rec.value == pytest.approx(0.35)

    def test_observe_histogram_non_histogram_raises(self, collector):
        collector.register_metric("cnt", "counter")
        with pytest.raises(ValueError, match="histogram"):
            collector.observe_histogram("cnt", 1.0)

    def test_histogram_bucket_data(self, collector):
        collector.register_metric("h", "histogram")
        collector.observe_histogram("h", 0.003)
        collector.observe_histogram("h", 0.05)
        collector.observe_histogram("h", 1.5)
        data = collector.get_histogram_data("h")
        assert data is not None
        assert data["count"] == 3
        assert data["sum"] == pytest.approx(1.553)

    def test_histogram_data_nonexistent(self, collector):
        collector.register_metric("h", "histogram")
        data = collector.get_histogram_data("h", {"missing": "label"})
        assert data is None


# ==========================================================================
# General Record Tests
# ==========================================================================

class TestMetricsCollectorRecord:
    """Tests for the general record() method."""

    def test_record_returns_metric_recording(self, collector):
        collector.register_metric("m", "counter")
        rec = collector.record("m", 5.0)
        assert isinstance(rec, MetricRecording)
        assert rec.provenance_hash
        assert len(rec.provenance_hash) == 64  # SHA-256 hex

    def test_record_unregistered_raises(self, collector):
        with pytest.raises(ValueError, match="not registered"):
            collector.record("ghost", 1.0)

    def test_record_unexpected_labels_raises(self, collector):
        collector.register_metric("strict", "counter", labels=["env"])
        with pytest.raises(ValueError, match="Unexpected labels"):
            collector.record("strict", 1.0, {"env": "prod", "extra": "bad"})


# ==========================================================================
# Query Tests
# ==========================================================================

class TestMetricsCollectorQuery:
    """Tests for query operations."""

    def test_get_metric_existing(self, collector):
        collector.register_metric("q", "counter")
        defn = collector.get_metric("q")
        assert defn is not None
        assert defn.name == "q"

    def test_get_metric_nonexistent(self, collector):
        assert collector.get_metric("nope") is None

    def test_get_metric_value_no_series(self, collector):
        collector.register_metric("v", "counter")
        assert collector.get_metric_value("v") is None

    def test_list_metrics_all(self, collector):
        collector.register_metric("b_metric", "counter")
        collector.register_metric("a_metric", "gauge")
        result = collector.list_metrics()
        assert len(result) == 2
        assert result[0].name == "a_metric"  # sorted

    def test_list_metrics_with_prefix_filter(self, collector):
        collector.register_metric("http_req", "counter")
        collector.register_metric("http_err", "counter")
        collector.register_metric("db_conn", "gauge")
        result = collector.list_metrics(prefix_filter="http_")
        assert len(result) == 2

    def test_get_metric_series(self, collector):
        collector.register_metric("ms", "counter", labels=["env"])
        collector.record("ms", 1.0, {"env": "prod"})
        collector.record("ms", 2.0, {"env": "dev"})
        series = collector.get_metric_series("ms")
        assert len(series) == 2


# ==========================================================================
# Prometheus Export Tests
# ==========================================================================

class TestMetricsCollectorExport:
    """Tests for Prometheus text export."""

    def test_export_empty(self, collector):
        output = collector.export_prometheus()
        assert isinstance(output, str)

    def test_export_counter_format(self, collector):
        collector.register_metric("cnt", "counter", "A counter")
        collector.record("cnt", 5.0)
        output = collector.export_prometheus()
        assert "# HELP cnt A counter" in output
        assert "# TYPE cnt counter" in output
        assert "cnt 5" in output

    def test_export_gauge_with_labels(self, collector):
        collector.register_metric("g", "gauge", labels=["host"])
        collector.set_gauge("g", 42.0, {"host": "h1"})
        output = collector.export_prometheus()
        assert 'host="h1"' in output

    def test_export_histogram_format(self, collector):
        collector.register_metric("hist", "histogram", "Latency hist")
        collector.observe_histogram("hist", 0.1)
        output = collector.export_prometheus()
        assert "hist_bucket" in output
        assert "hist_sum" in output
        assert "hist_count" in output
        assert "# TYPE hist histogram" in output


# ==========================================================================
# Provenance Tests
# ==========================================================================

class TestMetricsCollectorProvenance:
    """Tests for provenance hashing."""

    def test_provenance_hash_is_sha256(self, collector):
        collector.register_metric("p", "counter")
        rec = collector.record("p", 1.0)
        assert len(rec.provenance_hash) == 64

    def test_deterministic_label_key(self, collector):
        collector.register_metric("det", "counter", labels=["a", "b"])
        rec1 = collector.record("det", 1.0, {"b": "2", "a": "1"})
        rec2 = collector.record("det", 1.0, {"a": "1", "b": "2"})
        # Same labels different order should produce same label key
        val = collector.get_metric_value("det", {"a": "1", "b": "2"})
        assert val == pytest.approx(2.0)  # accumulated counter


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestMetricsCollectorStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, collector):
        stats = collector.get_statistics()
        assert stats["total_metrics"] == 0
        assert stats["total_series"] == 0
        assert stats["total_recordings"] == 0

    def test_statistics_after_recordings(self, collector):
        collector.register_metric("s1", "counter")
        collector.register_metric("s2", "gauge")
        collector.record("s1", 1.0)
        collector.set_gauge("s2", 5.0)
        stats = collector.get_statistics()
        assert stats["total_metrics"] == 2
        assert stats["total_series"] == 2
        assert stats["total_recordings"] == 2
        assert "counter" in stats["metrics_by_type"]
        assert "gauge" in stats["metrics_by_type"]


# ==========================================================================
# Maintenance Tests
# ==========================================================================

class TestMetricsCollectorMaintenance:
    """Tests for unregister and reset."""

    def test_unregister_metric(self, collector):
        collector.register_metric("rm", "counter")
        collector.record("rm", 1.0)
        result = collector.unregister_metric("rm")
        assert result is True
        assert collector.get_metric("rm") is None

    def test_unregister_nonexistent(self, collector):
        result = collector.unregister_metric("ghost")
        assert result is False

    def test_reset_metric(self, collector):
        collector.register_metric("res", "counter")
        collector.record("res", 10.0)
        count = collector.reset_metric("res")
        assert count >= 1
        assert collector.get_metric_value("res") == pytest.approx(0.0)

    def test_reset_unregistered_raises(self, collector):
        with pytest.raises(ValueError, match="not registered"):
            collector.reset_metric("ghost")


# ==========================================================================
# Series Limit Tests
# ==========================================================================

class TestMetricsCollectorSeriesLimit:
    """Tests for max series enforcement."""

    def test_max_series_limit(self):
        cfg = _StubConfig(max_series=2)
        coll = MetricsCollector(cfg)
        coll.register_metric("m", "counter", labels=["id"])
        coll.record("m", 1.0, {"id": "1"})
        coll.record("m", 1.0, {"id": "2"})
        with pytest.raises(ValueError, match="Maximum series limit"):
            coll.record("m", 1.0, {"id": "3"})


# ==========================================================================
# Multiple Label Combinations
# ==========================================================================

class TestMetricsCollectorMultiLabel:
    """Tests for multiple label combinations."""

    def test_multiple_label_combinations_independent(self, collector):
        collector.register_metric("req", "counter", labels=["method", "status"])
        collector.record("req", 1.0, {"method": "GET", "status": "200"})
        collector.record("req", 3.0, {"method": "POST", "status": "201"})
        collector.record("req", 2.0, {"method": "GET", "status": "200"})

        val_get = collector.get_metric_value("req", {"method": "GET", "status": "200"})
        val_post = collector.get_metric_value("req", {"method": "POST", "status": "201"})
        assert val_get == pytest.approx(3.0)
        assert val_post == pytest.approx(3.0)
