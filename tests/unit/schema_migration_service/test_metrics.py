# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus Metrics - AGENT-DATA-017

Tests all 12 metric definitions, 12 helper functions, label validation,
histogram bucket configurations, graceful fallback when prometheus_client
is unavailable, and module exports.
Target: 60+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

# Import the metrics module and its exports
from greenlang.schema_migration.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    sm_schemas_registered_total,
    sm_versions_created_total,
    sm_changes_detected_total,
    sm_compatibility_checks_total,
    sm_migrations_planned_total,
    sm_migrations_executed_total,
    sm_rollbacks_total,
    sm_drift_detected_total,
    sm_migration_duration_seconds,
    sm_records_migrated,
    sm_processing_duration_seconds,
    sm_active_migrations,
    # Helper functions
    record_schema_registered,
    record_version_created,
    record_change_detected,
    record_compatibility_check,
    record_migration_planned,
    record_migration_executed,
    record_rollback,
    record_drift_detected,
    observe_migration_duration,
    observe_records_migrated,
    observe_processing_duration,
    set_active_migrations,
)


# =============================================================================
# TestMetricsAvailability
# =============================================================================


class TestMetricsAvailability:
    """Verify prometheus_client availability detection."""

    def test_prometheus_available_is_bool(self):
        """PROMETHEUS_AVAILABLE should be a boolean value."""
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_reflects_import(self):
        """PROMETHEUS_AVAILABLE should match whether prometheus_client is importable."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            assert PROMETHEUS_AVAILABLE is False

    def test_metric_objects_not_none_when_available(self):
        """When PROMETHEUS_AVAILABLE is True, all metric objects should be non-None."""
        if PROMETHEUS_AVAILABLE:
            metric_objects = [
                sm_schemas_registered_total,
                sm_versions_created_total,
                sm_changes_detected_total,
                sm_compatibility_checks_total,
                sm_migrations_planned_total,
                sm_migrations_executed_total,
                sm_rollbacks_total,
                sm_drift_detected_total,
                sm_migration_duration_seconds,
                sm_records_migrated,
                sm_processing_duration_seconds,
                sm_active_migrations,
            ]
            for metric in metric_objects:
                assert metric is not None
        else:
            pytest.skip("prometheus_client not installed")


# =============================================================================
# TestMetricObjects
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricObjects:
    """Verify each metric object exists and has the correct type."""

    def test_schemas_registered_is_counter(self):
        """sm_schemas_registered_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_schemas_registered_total, Counter)

    def test_versions_created_is_counter(self):
        """sm_versions_created_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_versions_created_total, Counter)

    def test_changes_detected_is_counter(self):
        """sm_changes_detected_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_changes_detected_total, Counter)

    def test_compatibility_checks_is_counter(self):
        """sm_compatibility_checks_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_compatibility_checks_total, Counter)

    def test_migrations_planned_is_counter(self):
        """sm_migrations_planned_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_migrations_planned_total, Counter)

    def test_migrations_executed_is_counter(self):
        """sm_migrations_executed_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_migrations_executed_total, Counter)

    def test_rollbacks_is_counter(self):
        """sm_rollbacks_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_rollbacks_total, Counter)

    def test_drift_detected_is_counter(self):
        """sm_drift_detected_total should be a Counter."""
        from prometheus_client import Counter
        assert isinstance(sm_drift_detected_total, Counter)

    def test_migration_duration_is_histogram(self):
        """sm_migration_duration_seconds should be a Histogram."""
        from prometheus_client import Histogram
        assert isinstance(sm_migration_duration_seconds, Histogram)

    def test_records_migrated_is_histogram(self):
        """sm_records_migrated should be a Histogram."""
        from prometheus_client import Histogram
        assert isinstance(sm_records_migrated, Histogram)

    def test_processing_duration_is_histogram(self):
        """sm_processing_duration_seconds should be a Histogram."""
        from prometheus_client import Histogram
        assert isinstance(sm_processing_duration_seconds, Histogram)

    def test_active_migrations_is_gauge(self):
        """sm_active_migrations should be a Gauge."""
        from prometheus_client import Gauge
        assert isinstance(sm_active_migrations, Gauge)


# =============================================================================
# TestMetricNames
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricNames:
    """Verify metric names follow the gl_sm_ naming convention.

    Note: prometheus_client Counter stores _name without the _total suffix
    because it automatically appends _total on export. Histograms and Gauges
    store the full name in _name.
    """

    def test_schemas_registered_metric_name(self):
        assert "gl_sm_schemas_registered" in sm_schemas_registered_total._name

    def test_versions_created_metric_name(self):
        assert "gl_sm_versions_created" in sm_versions_created_total._name

    def test_changes_detected_metric_name(self):
        assert "gl_sm_changes_detected" in sm_changes_detected_total._name

    def test_compatibility_checks_metric_name(self):
        assert "gl_sm_compatibility_checks" in sm_compatibility_checks_total._name

    def test_migrations_planned_metric_name(self):
        assert "gl_sm_migrations_planned" in sm_migrations_planned_total._name

    def test_migrations_executed_metric_name(self):
        assert "gl_sm_migrations_executed" in sm_migrations_executed_total._name

    def test_rollbacks_metric_name(self):
        assert "gl_sm_rollbacks" in sm_rollbacks_total._name

    def test_drift_detected_metric_name(self):
        assert "gl_sm_drift_detected" in sm_drift_detected_total._name

    def test_migration_duration_metric_name(self):
        assert sm_migration_duration_seconds._name == "gl_sm_migration_duration_seconds"

    def test_records_migrated_metric_name(self):
        assert sm_records_migrated._name == "gl_sm_records_migrated"

    def test_processing_duration_metric_name(self):
        assert sm_processing_duration_seconds._name == "gl_sm_processing_duration_seconds"

    def test_active_migrations_metric_name(self):
        assert sm_active_migrations._name == "gl_sm_active_migrations"


# =============================================================================
# TestHelperFunctions
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestHelperFunctions:
    """Test all 12 helper functions work correctly when Prometheus is available."""

    def test_record_schema_registered_json_schema(self):
        """record_schema_registered should not raise for json_schema type."""
        record_schema_registered("json_schema", "emissions")

    def test_record_schema_registered_avro(self):
        record_schema_registered("avro", "suppliers")

    def test_record_schema_registered_protobuf(self):
        record_schema_registered("protobuf", "global")

    def test_record_version_created_major(self):
        """record_version_created should not raise for major bump."""
        record_version_created("major")

    def test_record_version_created_minor(self):
        record_version_created("minor")

    def test_record_version_created_patch(self):
        record_version_created("patch")

    def test_record_change_detected_field_added(self):
        """record_change_detected should not raise for field_added."""
        record_change_detected("field_added", "non_breaking")

    def test_record_change_detected_field_removed(self):
        record_change_detected("field_removed", "breaking")

    def test_record_change_detected_type_changed(self):
        record_change_detected("type_changed", "breaking")

    def test_record_compatibility_check_compatible(self):
        """record_compatibility_check should not raise for compatible."""
        record_compatibility_check("compatible")

    def test_record_compatibility_check_incompatible(self):
        record_compatibility_check("incompatible")

    def test_record_compatibility_check_warning(self):
        record_compatibility_check("warning")

    def test_record_migration_planned_success(self):
        """record_migration_planned should not raise for success."""
        record_migration_planned("success")

    def test_record_migration_planned_failed(self):
        record_migration_planned("failed")

    def test_record_migration_executed_success(self):
        """record_migration_executed should not raise for success."""
        record_migration_executed("success")

    def test_record_migration_executed_failed(self):
        record_migration_executed("failed")

    def test_record_migration_executed_rolled_back(self):
        record_migration_executed("rolled_back")

    def test_record_rollback_automatic_success(self):
        """record_rollback should not raise for automatic/success."""
        record_rollback("automatic", "success")

    def test_record_rollback_manual_failed(self):
        record_rollback("manual", "failed")

    def test_record_rollback_emergency(self):
        record_rollback("emergency", "partial")

    def test_record_drift_detected_structural_critical(self):
        """record_drift_detected should not raise for structural/critical."""
        record_drift_detected("structural", "critical")

    def test_record_drift_detected_data_type_medium(self):
        record_drift_detected("data_type", "medium")

    def test_record_drift_detected_constraint_low(self):
        record_drift_detected("constraint", "low")

    def test_observe_migration_duration_small(self):
        """observe_migration_duration should not raise for small value."""
        observe_migration_duration(0.5)

    def test_observe_migration_duration_large(self):
        observe_migration_duration(3600.0)

    def test_observe_migration_duration_zero(self):
        observe_migration_duration(0.0)

    def test_observe_records_migrated_small(self):
        """observe_records_migrated should not raise for small count."""
        observe_records_migrated(10)

    def test_observe_records_migrated_large(self):
        observe_records_migrated(100000)

    def test_observe_records_migrated_zero(self):
        observe_records_migrated(0)

    def test_observe_processing_duration_register(self):
        """observe_processing_duration should not raise for schema_register."""
        observe_processing_duration("schema_register", 0.05)

    def test_observe_processing_duration_execute(self):
        observe_processing_duration("migration_execute", 5.0)

    def test_observe_processing_duration_rollback(self):
        observe_processing_duration("rollback", 1.5)

    def test_set_active_migrations_zero(self):
        """set_active_migrations should not raise for 0."""
        set_active_migrations(0)

    def test_set_active_migrations_positive(self):
        set_active_migrations(5)

    def test_set_active_migrations_large(self):
        set_active_migrations(100)


# =============================================================================
# TestHelperFunctionsWithoutPrometheus
# =============================================================================


class TestHelperFunctionsWithoutPrometheus:
    """Verify helpers are no-ops when PROMETHEUS_AVAILABLE is False."""

    def test_record_schema_registered_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_schema_registered("json_schema", "emissions")  # Should not raise

    def test_record_version_created_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_version_created("major")

    def test_record_change_detected_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_change_detected("field_added", "non_breaking")

    def test_record_compatibility_check_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_compatibility_check("compatible")

    def test_record_migration_planned_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_migration_planned("success")

    def test_record_migration_executed_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_migration_executed("success")

    def test_record_rollback_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_rollback("automatic", "success")

    def test_record_drift_detected_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            record_drift_detected("structural", "critical")

    def test_observe_migration_duration_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            observe_migration_duration(10.0)

    def test_observe_records_migrated_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            observe_records_migrated(1000)

    def test_observe_processing_duration_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            observe_processing_duration("schema_register", 0.05)

    def test_set_active_migrations_no_op(self):
        with patch("greenlang.schema_migration.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_migrations(5)


# =============================================================================
# TestMetricLabels
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricLabels:
    """Verify metrics expose the correct label names."""

    def test_schemas_registered_labels(self):
        """sm_schemas_registered_total should have schema_type and namespace labels."""
        assert "schema_type" in sm_schemas_registered_total._labelnames
        assert "namespace" in sm_schemas_registered_total._labelnames

    def test_versions_created_labels(self):
        """sm_versions_created_total should have bump_type label."""
        assert "bump_type" in sm_versions_created_total._labelnames

    def test_changes_detected_labels(self):
        """sm_changes_detected_total should have change_type and severity labels."""
        assert "change_type" in sm_changes_detected_total._labelnames
        assert "severity" in sm_changes_detected_total._labelnames

    def test_compatibility_checks_labels(self):
        """sm_compatibility_checks_total should have result label."""
        assert "result" in sm_compatibility_checks_total._labelnames

    def test_migrations_planned_labels(self):
        """sm_migrations_planned_total should have status label."""
        assert "status" in sm_migrations_planned_total._labelnames

    def test_migrations_executed_labels(self):
        """sm_migrations_executed_total should have status label."""
        assert "status" in sm_migrations_executed_total._labelnames

    def test_rollbacks_labels(self):
        """sm_rollbacks_total should have rollback_type and status labels."""
        assert "rollback_type" in sm_rollbacks_total._labelnames
        assert "status" in sm_rollbacks_total._labelnames

    def test_drift_detected_labels(self):
        """sm_drift_detected_total should have drift_type and severity labels."""
        assert "drift_type" in sm_drift_detected_total._labelnames
        assert "severity" in sm_drift_detected_total._labelnames

    def test_migration_duration_no_labels(self):
        """sm_migration_duration_seconds should have no labels."""
        assert sm_migration_duration_seconds._labelnames == ()

    def test_records_migrated_no_labels(self):
        """sm_records_migrated should have no labels."""
        assert sm_records_migrated._labelnames == ()

    def test_processing_duration_labels(self):
        """sm_processing_duration_seconds should have operation label."""
        assert "operation" in sm_processing_duration_seconds._labelnames

    def test_active_migrations_no_labels(self):
        """sm_active_migrations should have no labels."""
        assert sm_active_migrations._labelnames == ()


# =============================================================================
# TestHistogramBuckets
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestHistogramBuckets:
    """Verify histogram bucket configurations."""

    def test_migration_duration_buckets(self):
        """sm_migration_duration_seconds should have migration-scale buckets."""
        expected_buckets = (1, 5, 10, 30, 60, 300, 600, 1800, 3600)
        # _upper_bounds includes the user-defined buckets plus float('inf')
        upper_bounds = sm_migration_duration_seconds._upper_bounds
        for bucket in expected_buckets:
            assert float(bucket) in upper_bounds

    def test_migration_duration_has_inf_bucket(self):
        """sm_migration_duration_seconds should have +Inf as final bucket."""
        upper_bounds = sm_migration_duration_seconds._upper_bounds
        assert float("inf") in upper_bounds

    def test_records_migrated_buckets(self):
        """sm_records_migrated should have record-count-scale buckets."""
        expected_buckets = (10, 100, 1000, 5000, 10000, 50000, 100000)
        upper_bounds = sm_records_migrated._upper_bounds
        for bucket in expected_buckets:
            assert float(bucket) in upper_bounds

    def test_records_migrated_has_inf_bucket(self):
        """sm_records_migrated should have +Inf as final bucket."""
        upper_bounds = sm_records_migrated._upper_bounds
        assert float("inf") in upper_bounds

    def test_processing_duration_buckets(self):
        """sm_processing_duration_seconds should have sub-second scale buckets."""
        expected_buckets = (0.01, 0.05, 0.1, 0.5, 1, 5, 10)
        # For labeled histograms, buckets are configured at the metric level
        # Check _kwargs or use _upper_bounds on an instantiated label
        labeled = sm_processing_duration_seconds.labels(operation="test_bucket_check")
        upper_bounds = labeled._upper_bounds
        for bucket in expected_buckets:
            assert float(bucket) in upper_bounds

    def test_processing_duration_has_inf_bucket(self):
        """sm_processing_duration_seconds should have +Inf as final bucket."""
        labeled = sm_processing_duration_seconds.labels(operation="test_bucket_inf")
        upper_bounds = labeled._upper_bounds
        assert float("inf") in upper_bounds

    def test_migration_duration_bucket_count(self):
        """sm_migration_duration_seconds should have 9 user buckets + inf."""
        upper_bounds = sm_migration_duration_seconds._upper_bounds
        assert len(upper_bounds) == 10  # 9 user + inf

    def test_records_migrated_bucket_count(self):
        """sm_records_migrated should have 7 user buckets + inf."""
        upper_bounds = sm_records_migrated._upper_bounds
        assert len(upper_bounds) == 8  # 7 user + inf


# =============================================================================
# TestMetricsExports
# =============================================================================


class TestMetricsExports:
    """Verify the metrics module exports all expected names via __all__."""

    def test_all_metric_objects_exported(self):
        """All 12 metric objects should be in __all__."""
        import greenlang.schema_migration.metrics as mod
        metric_names = [
            "sm_schemas_registered_total",
            "sm_versions_created_total",
            "sm_changes_detected_total",
            "sm_compatibility_checks_total",
            "sm_migrations_planned_total",
            "sm_migrations_executed_total",
            "sm_rollbacks_total",
            "sm_drift_detected_total",
            "sm_migration_duration_seconds",
            "sm_records_migrated",
            "sm_processing_duration_seconds",
            "sm_active_migrations",
        ]
        for name in metric_names:
            assert name in mod.__all__, f"{name} missing from __all__"

    def test_all_helper_functions_exported(self):
        """All 12 helper functions should be in __all__."""
        import greenlang.schema_migration.metrics as mod
        helper_names = [
            "record_schema_registered",
            "record_version_created",
            "record_change_detected",
            "record_compatibility_check",
            "record_migration_planned",
            "record_migration_executed",
            "record_rollback",
            "record_drift_detected",
            "observe_migration_duration",
            "observe_records_migrated",
            "observe_processing_duration",
            "set_active_migrations",
        ]
        for name in helper_names:
            assert name in mod.__all__, f"{name} missing from __all__"

    def test_prometheus_available_exported(self):
        """PROMETHEUS_AVAILABLE should be in __all__."""
        import greenlang.schema_migration.metrics as mod
        assert "PROMETHEUS_AVAILABLE" in mod.__all__

    def test_total_exports_count(self):
        """__all__ should have exactly 25 entries (1 flag + 12 metrics + 12 helpers)."""
        import greenlang.schema_migration.metrics as mod
        assert len(mod.__all__) == 25
