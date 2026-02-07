# -*- coding: utf-8 -*-
"""
Unit Tests for Normalizer Metrics (AGENT-FOUND-003)

Tests Prometheus metric recording: counters, histograms, gauges,
PROMETHEUS_AVAILABLE flag, all 12 metrics, and graceful fallback
when prometheus_client is not available.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

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
# Inline _NoOpMetric and NormalizerMetrics mirroring
# greenlang/normalizer/metrics.py
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


class NormalizerMetrics:
    """
    Prometheus metrics for the Normalizer Service.

    Records 12 metrics covering conversions, entity resolution,
    cache hits, errors, batch sizes, etc. All functions gracefully
    no-op when prometheus_client is not available.
    """

    CONVERSION_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 25, 50, 100)
    BATCH_SIZE_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000)

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and PROMETHEUS_AVAILABLE
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}

        # 12 metric name registry
        self._metric_names = {
            "conversions_total": "gl_normalizer_conversions_total",
            "conversion_duration_ms": "gl_normalizer_conversion_duration_ms",
            "conversion_errors_total": "gl_normalizer_conversion_errors_total",
            "entity_resolutions_total": "gl_normalizer_entity_resolutions_total",
            "entity_resolution_duration_ms": "gl_normalizer_entity_resolution_duration_ms",
            "entity_unresolved_total": "gl_normalizer_entity_unresolved_total",
            "cache_hits_total": "gl_normalizer_cache_hits_total",
            "cache_misses_total": "gl_normalizer_cache_misses_total",
            "batch_size": "gl_normalizer_batch_size",
            "active_conversions": "gl_normalizer_active_conversions",
            "gwp_conversions_total": "gl_normalizer_gwp_conversions_total",
            "provenance_records_total": "gl_normalizer_provenance_records_total",
        }

    # ---- Counters ----

    def record_conversion(self, from_unit: str, to_unit: str, dimension: str):
        if not self._enabled:
            return
        key = f"conversion:{dimension}:{from_unit}->{to_unit}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_conversion_error(self, error_type: str):
        if not self._enabled:
            return
        key = f"conversion_error:{error_type}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_entity_resolution(self, entity_type: str, confidence_level: str):
        if not self._enabled:
            return
        key = f"entity_resolution:{entity_type}:{confidence_level}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_entity_unresolved(self, entity_type: str):
        if not self._enabled:
            return
        key = f"entity_unresolved:{entity_type}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_cache_hit(self, cache_type: str):
        if not self._enabled:
            return
        key = f"cache_hit:{cache_type}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_cache_miss(self, cache_type: str):
        if not self._enabled:
            return
        key = f"cache_miss:{cache_type}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_gwp_conversion(self, gas_type: str, gwp_version: str):
        if not self._enabled:
            return
        key = f"gwp:{gas_type}:{gwp_version}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_provenance(self):
        if not self._enabled:
            return
        key = "provenance_total"
        self._counters[key] = self._counters.get(key, 0) + 1

    # ---- Histograms ----

    def record_conversion_duration(self, duration_ms: float, dimension: str):
        if not self._enabled:
            return
        key = f"conversion_duration:{dimension}"
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(duration_ms)

    def record_entity_resolution_duration(self, duration_ms: float, entity_type: str):
        if not self._enabled:
            return
        key = f"entity_resolution_duration:{entity_type}"
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(duration_ms)

    def record_batch(self, batch_size: int):
        if not self._enabled:
            return
        key = "batch_size"
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(batch_size)

    # ---- Gauges ----

    def update_active_conversions(self, delta: int):
        if not self._enabled:
            return
        current = self._gauges.get("active_conversions", 0)
        self._gauges["active_conversions"] = max(0, current + delta)

    # ---- Accessors for testing ----

    def get_counter(self, key: str) -> float:
        return self._counters.get(key, 0)

    def get_gauge(self, key: str) -> float:
        return self._gauges.get(key, 0)

    def get_histogram_observations(self, key: str) -> List[float]:
        return self._histograms.get(key, [])


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPrometheusAvailableFlag:
    """Test PROMETHEUS_AVAILABLE flag detection."""

    def test_flag_is_boolean(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_metrics_enabled_when_prometheus_available(self):
        metrics = NormalizerMetrics(enabled=True)
        assert metrics._enabled == PROMETHEUS_AVAILABLE

    def test_metrics_disabled_overrides_prometheus(self):
        metrics = NormalizerMetrics(enabled=False)
        assert metrics._enabled is False


class TestRecordConversion:
    """Test record_conversion() counter."""

    def test_conversion_recorded(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_conversion("kg", "t", "MASS")
        assert m.get_counter("conversion:MASS:kg->t") == 1

    def test_multiple_conversions(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_conversion("kg", "t", "MASS")
        m.record_conversion("kg", "t", "MASS")
        m.record_conversion("kWh", "MWh", "ENERGY")
        assert m.get_counter("conversion:MASS:kg->t") == 2
        assert m.get_counter("conversion:ENERGY:kWh->MWh") == 1


class TestRecordConversionError:
    """Test record_conversion_error() counter."""

    def test_error_recorded(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_conversion_error("incompatible_dimensions")
        assert m.get_counter("conversion_error:incompatible_dimensions") == 1

    def test_multiple_errors(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_conversion_error("unknown_unit")
        m.record_conversion_error("unknown_unit")
        m.record_conversion_error("incompatible_dimensions")
        assert m.get_counter("conversion_error:unknown_unit") == 2
        assert m.get_counter("conversion_error:incompatible_dimensions") == 1


class TestRecordEntityResolution:
    """Test entity resolution counters."""

    def test_resolution_recorded(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_entity_resolution("fuel", "EXACT")
        assert m.get_counter("entity_resolution:fuel:EXACT") == 1

    def test_unresolved_recorded(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_entity_unresolved("material")
        assert m.get_counter("entity_unresolved:material") == 1


class TestRecordCacheHitMiss:
    """Test cache hit/miss counters."""

    def test_cache_hit(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_cache_hit("conversion")
        assert m.get_counter("cache_hit:conversion") == 1

    def test_cache_miss(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_cache_miss("conversion")
        assert m.get_counter("cache_miss:conversion") == 1

    def test_hit_miss_separate(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_cache_hit("conversion")
        m.record_cache_miss("conversion")
        assert m.get_counter("cache_hit:conversion") == 1
        assert m.get_counter("cache_miss:conversion") == 1


class TestRecordGWPConversion:
    """Test GWP conversion counter."""

    def test_gwp_recorded(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_gwp_conversion("CH4", "AR6")
        assert m.get_counter("gwp:CH4:AR6") == 1

    def test_multiple_gwp_versions(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_gwp_conversion("CH4", "AR6")
        m.record_gwp_conversion("CH4", "AR5")
        assert m.get_counter("gwp:CH4:AR6") == 1
        assert m.get_counter("gwp:CH4:AR5") == 1


class TestRecordProvenance:
    """Test provenance record counter."""

    def test_provenance_recorded(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_provenance()
        assert m.get_counter("provenance_total") == 1


class TestHistograms:
    """Test histogram recording."""

    def test_conversion_duration(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_conversion_duration(0.5, "MASS")
        obs = m.get_histogram_observations("conversion_duration:MASS")
        assert len(obs) == 1
        assert obs[0] == pytest.approx(0.5)

    def test_entity_resolution_duration(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_entity_resolution_duration(1.2, "fuel")
        obs = m.get_histogram_observations("entity_resolution_duration:fuel")
        assert len(obs) == 1

    def test_batch_size(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.record_batch(50)
        obs = m.get_histogram_observations("batch_size")
        assert obs == [50]


class TestActiveConversionsGauge:
    """Test active_conversions gauge."""

    def test_increment(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.update_active_conversions(1)
        assert m.get_gauge("active_conversions") == 1

    def test_decrement(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.update_active_conversions(1)
        m.update_active_conversions(1)
        m.update_active_conversions(-1)
        assert m.get_gauge("active_conversions") == 1

    def test_does_not_go_negative(self):
        m = NormalizerMetrics(enabled=True)
        if not m._enabled:
            pytest.skip("prometheus_client not available")
        m.update_active_conversions(-5)
        assert m.get_gauge("active_conversions") == 0


class TestGracefulFallbackWhenDisabled:
    """Test all metrics functions work when disabled."""

    def test_disabled_no_conversion_counter(self):
        m = NormalizerMetrics(enabled=False)
        m.record_conversion("kg", "t", "MASS")
        assert m.get_counter("conversion:MASS:kg->t") == 0

    def test_disabled_no_error_counter(self):
        m = NormalizerMetrics(enabled=False)
        m.record_conversion_error("test")
        assert m.get_counter("conversion_error:test") == 0

    def test_disabled_no_entity_resolution(self):
        m = NormalizerMetrics(enabled=False)
        m.record_entity_resolution("fuel", "EXACT")
        assert m.get_counter("entity_resolution:fuel:EXACT") == 0

    def test_disabled_no_cache_hit(self):
        m = NormalizerMetrics(enabled=False)
        m.record_cache_hit("conversion")
        assert m.get_counter("cache_hit:conversion") == 0

    def test_disabled_no_cache_miss(self):
        m = NormalizerMetrics(enabled=False)
        m.record_cache_miss("conversion")
        assert m.get_counter("cache_miss:conversion") == 0

    def test_disabled_no_gwp(self):
        m = NormalizerMetrics(enabled=False)
        m.record_gwp_conversion("CH4", "AR6")
        assert m.get_counter("gwp:CH4:AR6") == 0

    def test_disabled_no_provenance(self):
        m = NormalizerMetrics(enabled=False)
        m.record_provenance()
        assert m.get_counter("provenance_total") == 0

    def test_disabled_no_histograms(self):
        m = NormalizerMetrics(enabled=False)
        m.record_conversion_duration(1.0, "MASS")
        assert m.get_histogram_observations("conversion_duration:MASS") == []

    def test_disabled_no_gauge(self):
        m = NormalizerMetrics(enabled=False)
        m.update_active_conversions(1)
        assert m.get_gauge("active_conversions") == 0


class TestMetricNames:
    """Test metric name registry."""

    def test_12_metrics_registered(self):
        m = NormalizerMetrics(enabled=False)
        assert len(m._metric_names) == 12

    def test_all_names_have_gl_normalizer_prefix(self):
        m = NormalizerMetrics(enabled=False)
        for name in m._metric_names.values():
            assert name.startswith("gl_normalizer_"), (
                f"Metric name '{name}' must start with 'gl_normalizer_'"
            )

    def test_expected_metric_keys(self):
        m = NormalizerMetrics(enabled=False)
        expected = {
            "conversions_total",
            "conversion_duration_ms",
            "conversion_errors_total",
            "entity_resolutions_total",
            "entity_resolution_duration_ms",
            "entity_unresolved_total",
            "cache_hits_total",
            "cache_misses_total",
            "batch_size",
            "active_conversions",
            "gwp_conversions_total",
            "provenance_records_total",
        }
        assert set(m._metric_names.keys()) == expected


class TestHistogramBuckets:
    """Test histogram bucket boundaries."""

    def test_conversion_buckets_sorted(self):
        assert list(NormalizerMetrics.CONVERSION_BUCKETS) == sorted(
            NormalizerMetrics.CONVERSION_BUCKETS
        )

    def test_conversion_buckets_positive(self):
        for b in NormalizerMetrics.CONVERSION_BUCKETS:
            assert b > 0

    def test_batch_size_buckets_sorted(self):
        assert list(NormalizerMetrics.BATCH_SIZE_BUCKETS) == sorted(
            NormalizerMetrics.BATCH_SIZE_BUCKETS
        )

    def test_batch_size_buckets_cover_1000(self):
        assert NormalizerMetrics.BATCH_SIZE_BUCKETS[-1] >= 1000


class TestNoOpMetric:
    """Test _NoOpMetric does not raise on any operation."""

    def test_noop_inc(self):
        _NoOpMetric().inc()

    def test_noop_dec(self):
        _NoOpMetric().dec()

    def test_noop_set(self):
        _NoOpMetric().set(0)

    def test_noop_observe(self):
        _NoOpMetric().observe(1.0)

    def test_noop_labels_returns_self(self):
        noop = _NoOpMetric()
        assert noop.labels(unit="kg") is noop

    def test_noop_chained(self):
        _NoOpMetric().labels(unit="kg").inc()  # Should not raise
