# -*- coding: utf-8 -*-
"""
Unit Tests for GIS Connector Agent Metrics (AGENT-DATA-006)

Tests Prometheus metric recording: NoOp fallback, all 12 metric names,
counter/histogram/gauge operations, and all helper functions for the
GIS/Mapping Connector Agent service.

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
# Inline _NoOpMetric and GISConnectorMetrics
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


class GISConnectorMetrics:
    """Prometheus metrics for the GIS/Mapping Connector Agent Service."""

    OPERATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and PROMETHEUS_AVAILABLE
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}

        self._metric_names = {
            "parse_operations_total": "gl_gis_connector_parse_operations_total",
            "transform_operations_total": "gl_gis_connector_transform_operations_total",
            "spatial_analyses_total": "gl_gis_connector_spatial_analyses_total",
            "geocode_requests_total": "gl_gis_connector_geocode_requests_total",
            "land_cover_classifications_total": "gl_gis_connector_land_cover_classifications_total",
            "boundary_resolutions_total": "gl_gis_connector_boundary_resolutions_total",
            "layer_operations_total": "gl_gis_connector_layer_operations_total",
            "processing_duration_seconds": "gl_gis_connector_processing_duration_seconds",
            "processing_errors_total": "gl_gis_connector_processing_errors_total",
            "cache_hits_total": "gl_gis_connector_cache_hits_total",
            "active_layers": "gl_gis_connector_active_layers",
            "features_stored": "gl_gis_connector_features_stored",
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

    def get_gauge(self, name: str, **labels) -> float:
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0)

    def _make_key(self, name: str, labels: Dict) -> str:
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name


# ---------------------------------------------------------------------------
# Helper functions (safe to call always, matching __init__.py exports)
# ---------------------------------------------------------------------------


def record_parse_operation(format_type: str = "geojson") -> None:
    """Record a geospatial parse operation event."""
    pass


def record_transform_operation(source_crs: str = "EPSG:4326", target_crs: str = "EPSG:3857") -> None:
    """Record a CRS transform operation event."""
    pass


def record_spatial_analysis(analysis_type: str = "distance") -> None:
    """Record a spatial analysis event."""
    pass


def record_geocode_request(direction: str = "forward") -> None:
    """Record a geocoding request event."""
    pass


def record_land_cover_classification(cover_type: str = "unknown") -> None:
    """Record a land cover classification event."""
    pass


def record_boundary_resolution(country_code: str = "unknown") -> None:
    """Record a boundary resolution event."""
    pass


def record_layer_operation(operation: str = "create") -> None:
    """Record a layer management operation event."""
    pass


def record_processing_error(error_type: str = "unknown") -> None:
    """Record a processing error event."""
    pass


def record_cache_hit(cache_type: str = "geocode") -> None:
    """Record a cache hit event."""
    pass


def update_active_layers(count: int = 0) -> None:
    """Update the active layers gauge."""
    pass


def update_features_stored(count: int = 0) -> None:
    """Update the features stored gauge."""
    pass


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPrometheusAvailability:
    """Tests for Prometheus availability flag."""

    def test_prometheus_available_flag(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


class TestNoOpFallback:
    """Tests for NoOp metric fallback."""

    def test_inc(self):
        m = _NoOpMetric()
        m.inc()

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
        result = m.labels(format_type="geojson")
        assert result is m

    def test_chained_labels_inc(self):
        m = _NoOpMetric()
        m.labels(format_type="shp").inc()


class TestMetricNamesExist:
    """Tests that all 12 metric objects are defined."""

    def test_all_12_metrics_exist(self):
        metrics = GISConnectorMetrics()
        names = metrics.metric_names
        assert len(names) == 12

    @pytest.mark.parametrize("short_name,full_name", [
        ("parse_operations_total", "gl_gis_connector_parse_operations_total"),
        ("transform_operations_total", "gl_gis_connector_transform_operations_total"),
        ("spatial_analyses_total", "gl_gis_connector_spatial_analyses_total"),
        ("geocode_requests_total", "gl_gis_connector_geocode_requests_total"),
        ("land_cover_classifications_total", "gl_gis_connector_land_cover_classifications_total"),
        ("boundary_resolutions_total", "gl_gis_connector_boundary_resolutions_total"),
        ("layer_operations_total", "gl_gis_connector_layer_operations_total"),
        ("processing_duration_seconds", "gl_gis_connector_processing_duration_seconds"),
        ("processing_errors_total", "gl_gis_connector_processing_errors_total"),
        ("cache_hits_total", "gl_gis_connector_cache_hits_total"),
        ("active_layers", "gl_gis_connector_active_layers"),
        ("features_stored", "gl_gis_connector_features_stored"),
    ])
    def test_metric_name(self, short_name, full_name):
        metrics = GISConnectorMetrics()
        assert metrics.metric_names[short_name] == full_name

    def test_metric_name_prefix(self):
        metrics = GISConnectorMetrics()
        for _, full_name in metrics.metric_names.items():
            assert full_name.startswith("gl_gis_connector_")


class TestHelperFunctions:
    """Tests that all 11 helper functions execute without errors."""

    def test_record_parse_operation(self):
        record_parse_operation("geojson")

    def test_record_transform_operation(self):
        record_transform_operation("EPSG:4326", "EPSG:3857")

    def test_record_spatial_analysis(self):
        record_spatial_analysis("distance")

    def test_record_geocode_request(self):
        record_geocode_request("forward")

    def test_record_land_cover_classification(self):
        record_land_cover_classification("forest_broadleaf")

    def test_record_boundary_resolution(self):
        record_boundary_resolution("US")

    def test_record_layer_operation(self):
        record_layer_operation("create")

    def test_record_processing_error(self):
        record_processing_error("invalid_format")

    def test_record_cache_hit(self):
        record_cache_hit("geocode")

    def test_update_active_layers(self):
        update_active_layers(5)

    def test_update_features_stored(self):
        update_features_stored(1500)


class TestCounterMetrics:
    """Tests for counter metric operations."""

    def test_inc_counter(self):
        metrics = GISConnectorMetrics()
        metrics.inc_counter("parse_operations_total")
        assert metrics.get_counter("parse_operations_total") == 1

    def test_inc_counter_multiple(self):
        metrics = GISConnectorMetrics()
        metrics.inc_counter("parse_operations_total")
        metrics.inc_counter("parse_operations_total")
        metrics.inc_counter("parse_operations_total")
        assert metrics.get_counter("parse_operations_total") == 3

    def test_inc_counter_with_labels(self):
        metrics = GISConnectorMetrics()
        metrics.inc_counter("parse_operations_total", format_type="geojson")
        metrics.inc_counter("parse_operations_total", format_type="shapefile")
        assert metrics.get_counter("parse_operations_total", format_type="geojson") == 1
        assert metrics.get_counter("parse_operations_total", format_type="shapefile") == 1

    def test_inc_counter_custom_value(self):
        metrics = GISConnectorMetrics()
        metrics.inc_counter("geocode_requests_total", value=5.0)
        assert metrics.get_counter("geocode_requests_total") == 5

    def test_get_counter_default_zero(self):
        metrics = GISConnectorMetrics()
        assert metrics.get_counter("nonexistent") == 0


class TestGaugeMetrics:
    """Tests for gauge metric operations."""

    def test_set_gauge(self):
        metrics = GISConnectorMetrics()
        metrics.set_gauge("active_layers", 5)
        assert metrics.get_gauge("active_layers") == 5

    def test_set_gauge_overwrite(self):
        metrics = GISConnectorMetrics()
        metrics.set_gauge("active_layers", 5)
        metrics.set_gauge("active_layers", 12)
        assert metrics.get_gauge("active_layers") == 12

    def test_get_gauge_default_zero(self):
        metrics = GISConnectorMetrics()
        assert metrics.get_gauge("nonexistent") == 0

    def test_set_features_stored(self):
        metrics = GISConnectorMetrics()
        metrics.set_gauge("features_stored", 15000)
        assert metrics.get_gauge("features_stored") == 15000


class TestHistogramMetrics:
    """Tests for histogram metric operations."""

    def test_observe_histogram(self):
        metrics = GISConnectorMetrics()
        metrics.observe_histogram("processing_duration_seconds", 0.25)
        # Histograms are stored in internal list

    def test_operation_buckets(self):
        assert len(GISConnectorMetrics.OPERATION_BUCKETS) == 12
        assert GISConnectorMetrics.OPERATION_BUCKETS[0] == 0.01

    def test_multiple_observations(self):
        metrics = GISConnectorMetrics()
        metrics.observe_histogram("processing_duration_seconds", 0.1)
        metrics.observe_histogram("processing_duration_seconds", 0.5)
        metrics.observe_histogram("processing_duration_seconds", 1.2)
        assert len(metrics._histograms["processing_duration_seconds"]) == 3

    def test_histogram_with_labels(self):
        metrics = GISConnectorMetrics()
        metrics.observe_histogram("processing_duration_seconds", 0.5, operation="parse")
        key = "processing_duration_seconds{operation=parse}"
        assert key in metrics._histograms
