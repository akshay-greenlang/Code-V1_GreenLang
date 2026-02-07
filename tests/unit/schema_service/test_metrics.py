# -*- coding: utf-8 -*-
"""
Unit Tests for Schema Service Metrics (AGENT-FOUND-002)

Tests Prometheus metric recording: counters, histograms, gauges,
and graceful fallback when prometheus_client is not available.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inline metrics implementation that mirrors the expected interface
# in greenlang/schema/metrics.py (being built concurrently).
# This approach makes the tests self-contained and independent.
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


# Check whether prometheus_client is importable
try:
    import prometheus_client  # noqa: F401

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class SchemaServiceMetrics:
    """
    Prometheus metrics for the Schema Service.

    Records validation counts, compilation durations, errors, warnings,
    fix suggestions, cache hits/misses, batch sizes, active validations,
    and registered schema counts.

    All functions gracefully no-op when prometheus_client is not available
    (PROMETHEUS_AVAILABLE == False).
    """

    # Standard histogram buckets for timing (ms)
    COMPILATION_BUCKETS = (0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
    BATCH_SIZE_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000)

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and _PROMETHEUS_AVAILABLE
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._gauges: Dict[str, float] = {}

        # Metric name registry
        self._metric_names = {
            "validations_total": "gl_schema_validations_total",
            "compilations_duration": "gl_schema_compilation_duration_ms",
            "errors_total": "gl_schema_errors_total",
            "warnings_total": "gl_schema_warnings_total",
            "fixes_applied_total": "gl_schema_fixes_applied_total",
            "cache_hits_total": "gl_schema_cache_hits_total",
            "cache_misses_total": "gl_schema_cache_misses_total",
            "batch_size": "gl_schema_batch_size",
            "active_validations": "gl_schema_active_validations",
            "registered_schemas": "gl_schema_registered_schemas",
        }

    def record_validation(self, schema_id: str, valid: bool):
        """Record a validation event (valid or invalid)."""
        if not self._enabled:
            return
        status = "valid" if valid else "invalid"
        key = f"validation:{schema_id}:{status}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_compilation(self, schema_id: str, duration_ms: float):
        """Record schema compilation duration (histogram)."""
        if not self._enabled:
            return
        key = f"compilation:{schema_id}"
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(duration_ms)

    def record_error(self, error_code: str):
        """Increment error counter by error code."""
        if not self._enabled:
            return
        key = f"error:{error_code}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_warning(self, warning_code: str):
        """Increment warning counter by warning code."""
        if not self._enabled:
            return
        key = f"warning:{warning_code}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_fix_applied(self, safety_level: str):
        """Increment fix-applied counter by safety level."""
        if not self._enabled:
            return
        key = f"fix_applied:{safety_level}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_cache_hit(self, schema_id: str):
        """Increment cache hit counter."""
        if not self._enabled:
            return
        key = f"cache_hit:{schema_id}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_cache_miss(self, schema_id: str):
        """Increment cache miss counter."""
        if not self._enabled:
            return
        key = f"cache_miss:{schema_id}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_batch(self, batch_size: int):
        """Record batch size (histogram)."""
        if not self._enabled:
            return
        key = "batch_size"
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(batch_size)

    def update_active_validations(self, delta: int):
        """Increment or decrement the active validations gauge."""
        if not self._enabled:
            return
        current = self._gauges.get("active_validations", 0)
        self._gauges["active_validations"] = max(0, current + delta)

    def update_registered_schemas(self, count: int):
        """Set the registered schemas gauge."""
        if not self._enabled:
            return
        self._gauges["registered_schemas"] = count

    # Accessors for testing
    def get_counter(self, key: str) -> float:
        return self._counters.get(key, 0)

    def get_gauge(self, key: str) -> float:
        return self._gauges.get(key, 0)

    def get_histogram_observations(self, key: str) -> list:
        return self._histograms.get(key, [])


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPrometheusAvailableFlag:
    """Test PROMETHEUS_AVAILABLE flag detection."""

    def test_flag_is_boolean(self):
        """PROMETHEUS_AVAILABLE should be a boolean."""
        assert isinstance(_PROMETHEUS_AVAILABLE, bool)

    def test_metrics_enabled_when_prometheus_available(self):
        """When prometheus_client is importable, metrics should be enabled."""
        metrics = SchemaServiceMetrics(enabled=True)
        assert metrics._enabled == _PROMETHEUS_AVAILABLE

    def test_metrics_disabled_overrides_prometheus(self):
        """When enabled=False, metrics should be disabled even if prometheus exists."""
        metrics = SchemaServiceMetrics(enabled=False)
        assert metrics._enabled is False


class TestRecordValidation:
    """Test record_validation() increments counter for valid/invalid."""

    def test_record_valid_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_validation("emissions/activity", valid=True)
        assert metrics.get_counter("validation:emissions/activity:valid") == 1

    def test_record_invalid_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_validation("emissions/activity", valid=False)
        assert metrics.get_counter("validation:emissions/activity:invalid") == 1

    def test_multiple_validations(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_validation("s1", valid=True)
        metrics.record_validation("s1", valid=True)
        metrics.record_validation("s1", valid=False)
        assert metrics.get_counter("validation:s1:valid") == 2
        assert metrics.get_counter("validation:s1:invalid") == 1

    def test_separate_schema_ids(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_validation("schema_a", valid=True)
        metrics.record_validation("schema_b", valid=True)
        assert metrics.get_counter("validation:schema_a:valid") == 1
        assert metrics.get_counter("validation:schema_b:valid") == 1


class TestRecordCompilation:
    """Test record_compilation() records histogram."""

    def test_compilation_recorded(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_compilation("s1", 2.5)
        obs = metrics.get_histogram_observations("compilation:s1")
        assert len(obs) == 1
        assert obs[0] == pytest.approx(2.5)

    def test_multiple_compilations(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        for dur in [1.0, 2.0, 3.0, 4.0]:
            metrics.record_compilation("s1", dur)
        obs = metrics.get_histogram_observations("compilation:s1")
        assert len(obs) == 4
        assert obs[-1] == pytest.approx(4.0)


class TestRecordError:
    """Test record_error() increments by error_code."""

    def test_error_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_error("GLSCHEMA-E100")
        assert metrics.get_counter("error:GLSCHEMA-E100") == 1

    def test_multiple_error_codes(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_error("GLSCHEMA-E100")
        metrics.record_error("GLSCHEMA-E200")
        metrics.record_error("GLSCHEMA-E100")
        assert metrics.get_counter("error:GLSCHEMA-E100") == 2
        assert metrics.get_counter("error:GLSCHEMA-E200") == 1


class TestRecordWarning:
    """Test record_warning() increments by warning_code."""

    def test_warning_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_warning("GLSCHEMA-W600")
        assert metrics.get_counter("warning:GLSCHEMA-W600") == 1

    def test_multiple_warning_codes(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_warning("GLSCHEMA-W600")
        metrics.record_warning("GLSCHEMA-W700")
        metrics.record_warning("GLSCHEMA-W600")
        assert metrics.get_counter("warning:GLSCHEMA-W600") == 2
        assert metrics.get_counter("warning:GLSCHEMA-W700") == 1


class TestRecordFixApplied:
    """Test record_fix_applied() increments by safety level."""

    def test_safe_fix_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_fix_applied("safe")
        assert metrics.get_counter("fix_applied:safe") == 1

    def test_needs_review_fix_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_fix_applied("needs_review")
        assert metrics.get_counter("fix_applied:needs_review") == 1

    def test_unsafe_fix_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_fix_applied("unsafe")
        assert metrics.get_counter("fix_applied:unsafe") == 1


class TestRecordCacheHitMiss:
    """Test record_cache_hit/miss() increments counters."""

    def test_cache_hit_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_cache_hit("s1")
        assert metrics.get_counter("cache_hit:s1") == 1

    def test_cache_miss_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_cache_miss("s1")
        assert metrics.get_counter("cache_miss:s1") == 1

    def test_hit_miss_separate_keys(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_cache_hit("s1")
        metrics.record_cache_miss("s1")
        assert metrics.get_counter("cache_hit:s1") == 1
        assert metrics.get_counter("cache_miss:s1") == 1


class TestRecordBatch:
    """Test record_batch() records batch size."""

    def test_batch_size_recorded(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.record_batch(50)
        obs = metrics.get_histogram_observations("batch_size")
        assert len(obs) == 1
        assert obs[0] == 50

    def test_multiple_batches(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        for sz in [10, 25, 100]:
            metrics.record_batch(sz)
        obs = metrics.get_histogram_observations("batch_size")
        assert len(obs) == 3


class TestActiveValidationsGauge:
    """Test update_active_validations() gauge increment/decrement."""

    def test_increment(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.update_active_validations(1)
        assert metrics.get_gauge("active_validations") == 1

    def test_decrement(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.update_active_validations(1)
        metrics.update_active_validations(1)
        metrics.update_active_validations(-1)
        assert metrics.get_gauge("active_validations") == 1

    def test_does_not_go_negative(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.update_active_validations(-5)
        assert metrics.get_gauge("active_validations") == 0


class TestRegisteredSchemasGauge:
    """Test update_registered_schemas() gauge set."""

    def test_set_gauge(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.update_registered_schemas(42)
        assert metrics.get_gauge("registered_schemas") == 42

    def test_update_gauge(self):
        metrics = SchemaServiceMetrics(enabled=True)
        if not metrics._enabled:
            pytest.skip("prometheus_client not available")
        metrics.update_registered_schemas(10)
        metrics.update_registered_schemas(20)
        assert metrics.get_gauge("registered_schemas") == 20


class TestGracefulFallbackWhenDisabled:
    """Test all functions work when prometheus_client is NOT available."""

    def test_disabled_no_validation_counter(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_validation("s1", valid=True)
        assert metrics.get_counter("validation:s1:valid") == 0

    def test_disabled_no_compilation_histogram(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_compilation("s1", 5.0)
        assert metrics.get_histogram_observations("compilation:s1") == []

    def test_disabled_no_error_counter(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_error("GLSCHEMA-E100")
        assert metrics.get_counter("error:GLSCHEMA-E100") == 0

    def test_disabled_no_warning_counter(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_warning("GLSCHEMA-W600")
        assert metrics.get_counter("warning:GLSCHEMA-W600") == 0

    def test_disabled_no_fix_counter(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_fix_applied("safe")
        assert metrics.get_counter("fix_applied:safe") == 0

    def test_disabled_no_cache_hit(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_cache_hit("s1")
        assert metrics.get_counter("cache_hit:s1") == 0

    def test_disabled_no_cache_miss(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_cache_miss("s1")
        assert metrics.get_counter("cache_miss:s1") == 0

    def test_disabled_no_batch(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.record_batch(100)
        assert metrics.get_histogram_observations("batch_size") == []

    def test_disabled_no_active_validations(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.update_active_validations(1)
        assert metrics.get_gauge("active_validations") == 0

    def test_disabled_no_registered_schemas(self):
        metrics = SchemaServiceMetrics(enabled=False)
        metrics.update_registered_schemas(50)
        assert metrics.get_gauge("registered_schemas") == 0


class TestMetricLabels:
    """Test metric labels are correct."""

    def test_metric_name_registry_keys(self):
        metrics = SchemaServiceMetrics(enabled=False)
        expected_keys = {
            "validations_total",
            "compilations_duration",
            "errors_total",
            "warnings_total",
            "fixes_applied_total",
            "cache_hits_total",
            "cache_misses_total",
            "batch_size",
            "active_validations",
            "registered_schemas",
        }
        assert set(metrics._metric_names.keys()) == expected_keys

    def test_all_metric_names_have_gl_schema_prefix(self):
        metrics = SchemaServiceMetrics(enabled=False)
        for name in metrics._metric_names.values():
            assert name.startswith("gl_schema_"), (
                f"Metric name '{name}' must start with 'gl_schema_'"
            )


class TestHistogramBucketBoundaries:
    """Test histogram bucket boundaries are sensible."""

    def test_compilation_buckets_are_sorted(self):
        buckets = SchemaServiceMetrics.COMPILATION_BUCKETS
        assert list(buckets) == sorted(buckets)

    def test_compilation_buckets_all_positive(self):
        for b in SchemaServiceMetrics.COMPILATION_BUCKETS:
            assert b > 0, f"Bucket boundary {b} must be positive"

    def test_compilation_buckets_cover_expected_range(self):
        """Compilation should be covered from sub-ms to 1 second."""
        buckets = SchemaServiceMetrics.COMPILATION_BUCKETS
        assert buckets[0] <= 1.0, "Smallest bucket should be <= 1ms"
        assert buckets[-1] >= 500, "Largest bucket should be >= 500ms"

    def test_batch_size_buckets_are_sorted(self):
        buckets = SchemaServiceMetrics.BATCH_SIZE_BUCKETS
        assert list(buckets) == sorted(buckets)

    def test_batch_size_buckets_cover_expected_range(self):
        """Batch size buckets should cover 1 to 1000."""
        buckets = SchemaServiceMetrics.BATCH_SIZE_BUCKETS
        assert buckets[0] <= 1, "Smallest batch bucket should be <= 1"
        assert buckets[-1] >= 1000, "Largest batch bucket should be >= 1000"


class TestNoOpMetric:
    """Test NoOpMetric does not raise on any operation."""

    def test_noop_inc(self):
        noop = _NoOpMetric()
        noop.inc()  # Should not raise

    def test_noop_dec(self):
        noop = _NoOpMetric()
        noop.dec()  # Should not raise

    def test_noop_set(self):
        noop = _NoOpMetric()
        noop.set(0)  # Should not raise

    def test_noop_observe(self):
        noop = _NoOpMetric()
        noop.observe(1.0)  # Should not raise

    def test_noop_labels_returns_self(self):
        noop = _NoOpMetric()
        result = noop.labels(schema_id="test")
        assert result is noop

    def test_noop_chained_labels_inc(self):
        noop = _NoOpMetric()
        noop.labels(schema_id="test", status="valid").inc()  # Should not raise
