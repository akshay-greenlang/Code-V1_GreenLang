# -*- coding: utf-8 -*-
"""
Unit Tests for Data Lineage Tracker Prometheus Metrics - AGENT-DATA-018

Tests the 12 Prometheus metrics and their helper functions:
  - PROMETHEUS_AVAILABLE flag
  - All 12 metric definitions (names and types)
  - All 12 helper functions (counter, histogram, gauge operations)
  - Graceful no-op behavior when prometheus_client is not installed

40+ test cases.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.data_lineage_tracker import metrics as dlt_metrics
from greenlang.data_lineage_tracker.metrics import (
    PROMETHEUS_AVAILABLE,
    dlt_assets_registered_total,
    dlt_change_events_total,
    dlt_edges_created_total,
    dlt_graph_edge_count,
    dlt_graph_node_count,
    dlt_graph_traversal_duration_seconds,
    dlt_impact_analyses_total,
    dlt_processing_duration_seconds,
    dlt_quality_scores_computed_total,
    dlt_reports_generated_total,
    dlt_transformations_captured_total,
    dlt_validations_total,
    observe_graph_traversal_duration,
    observe_processing_duration,
    record_asset_registered,
    record_change_event,
    record_edge_created,
    record_impact_analysis,
    record_quality_score,
    record_report_generated,
    record_transformation_captured,
    record_validation,
    set_graph_edge_count,
    set_graph_node_count,
)


# ============================================================================
# TestPrometheusAvailability
# ============================================================================


class TestPrometheusAvailability:
    """Tests for the PROMETHEUS_AVAILABLE flag."""

    def test_prometheus_available_flag_is_bool(self):
        """PROMETHEUS_AVAILABLE is a boolean."""
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_when_installed(self):
        """PROMETHEUS_AVAILABLE is True when prometheus_client is installed."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_metrics_module_has_all_exports(self):
        """metrics module __all__ includes all 24 expected names."""
        expected = {
            "PROMETHEUS_AVAILABLE",
            "dlt_assets_registered_total",
            "dlt_transformations_captured_total",
            "dlt_edges_created_total",
            "dlt_impact_analyses_total",
            "dlt_validations_total",
            "dlt_reports_generated_total",
            "dlt_change_events_total",
            "dlt_quality_scores_computed_total",
            "dlt_graph_traversal_duration_seconds",
            "dlt_processing_duration_seconds",
            "dlt_graph_node_count",
            "dlt_graph_edge_count",
            "record_asset_registered",
            "record_transformation_captured",
            "record_edge_created",
            "record_impact_analysis",
            "record_validation",
            "record_report_generated",
            "record_change_event",
            "record_quality_score",
            "observe_graph_traversal_duration",
            "observe_processing_duration",
            "set_graph_node_count",
            "set_graph_edge_count",
        }
        actual = set(dlt_metrics.__all__)
        assert expected == actual


# ============================================================================
# TestMetricDefinitions
# ============================================================================


class TestMetricDefinitions:
    """Tests for all 12 metric object definitions."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_1_assets_registered_is_counter(self):
        """gl_dlt_assets_registered_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_assets_registered_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_2_transformations_captured_is_counter(self):
        """gl_dlt_transformations_captured_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_transformations_captured_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_3_edges_created_is_counter(self):
        """gl_dlt_edges_created_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_edges_created_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_4_impact_analyses_is_counter(self):
        """gl_dlt_impact_analyses_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_impact_analyses_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_5_validations_is_counter(self):
        """gl_dlt_validations_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_validations_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_6_reports_generated_is_counter(self):
        """gl_dlt_reports_generated_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_reports_generated_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_7_change_events_is_counter(self):
        """gl_dlt_change_events_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_change_events_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_8_quality_scores_is_counter(self):
        """gl_dlt_quality_scores_computed_total is a Counter."""
        from prometheus_client import Counter
        assert isinstance(dlt_quality_scores_computed_total, Counter)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_9_graph_traversal_duration_is_histogram(self):
        """gl_dlt_graph_traversal_duration_seconds is a Histogram."""
        from prometheus_client import Histogram
        assert isinstance(dlt_graph_traversal_duration_seconds, Histogram)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_10_processing_duration_is_histogram(self):
        """gl_dlt_processing_duration_seconds is a Histogram."""
        from prometheus_client import Histogram
        assert isinstance(dlt_processing_duration_seconds, Histogram)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_11_graph_node_count_is_gauge(self):
        """gl_dlt_graph_node_count is a Gauge."""
        from prometheus_client import Gauge
        assert isinstance(dlt_graph_node_count, Gauge)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_12_graph_edge_count_is_gauge(self):
        """gl_dlt_graph_edge_count is a Gauge."""
        from prometheus_client import Gauge
        assert isinstance(dlt_graph_edge_count, Gauge)

    def test_all_12_metrics_defined(self):
        """All 12 metric objects are defined (not None when prometheus available)."""
        metric_objects = [
            dlt_assets_registered_total,
            dlt_transformations_captured_total,
            dlt_edges_created_total,
            dlt_impact_analyses_total,
            dlt_validations_total,
            dlt_reports_generated_total,
            dlt_change_events_total,
            dlt_quality_scores_computed_total,
            dlt_graph_traversal_duration_seconds,
            dlt_processing_duration_seconds,
            dlt_graph_node_count,
            dlt_graph_edge_count,
        ]
        if PROMETHEUS_AVAILABLE:
            for metric in metric_objects:
                assert metric is not None
        else:
            for metric in metric_objects:
                assert metric is None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metric_names_use_gl_dlt_prefix(self):
        """All metric names start with gl_dlt_ prefix."""
        metrics_with_describe = [
            dlt_assets_registered_total,
            dlt_transformations_captured_total,
            dlt_edges_created_total,
            dlt_impact_analyses_total,
            dlt_validations_total,
            dlt_reports_generated_total,
            dlt_change_events_total,
            dlt_quality_scores_computed_total,
            dlt_graph_traversal_duration_seconds,
            dlt_processing_duration_seconds,
            dlt_graph_node_count,
            dlt_graph_edge_count,
        ]
        for metric in metrics_with_describe:
            desc = metric.describe()
            for sample_family in desc:
                assert sample_family.name.startswith("gl_dlt_")


# ============================================================================
# TestHelperFunctionsWithPrometheus
# ============================================================================


class TestHelperFunctionsWithPrometheus:
    """Tests for helper functions when prometheus_client IS available."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_asset_registered(self):
        """record_asset_registered() increments counter without error."""
        record_asset_registered("dataset", "internal")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_asset_registered_various_types(self):
        """record_asset_registered() accepts all asset type / classification combos."""
        for asset_type in ("dataset", "field", "agent", "pipeline", "report", "metric"):
            for classification in ("public", "internal", "confidential", "restricted"):
                record_asset_registered(asset_type, classification)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_transformation_captured(self):
        """record_transformation_captured() increments counter without error."""
        record_transformation_captured("filter", "data-quality-profiler")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_edge_created(self):
        """record_edge_created() increments counter without error."""
        record_edge_created("dataset_level")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_edge_created_column_level(self):
        """record_edge_created() accepts column_level edge type."""
        record_edge_created("column_level")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_impact_analysis(self):
        """record_impact_analysis() increments counter without error."""
        record_impact_analysis("forward", "high")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_impact_analysis_backward(self):
        """record_impact_analysis() accepts backward direction."""
        record_impact_analysis("backward", "medium")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_validation(self):
        """record_validation() increments counter without error."""
        record_validation("pass")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_validation_fail(self):
        """record_validation() accepts fail result."""
        record_validation("fail")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_report_generated(self):
        """record_report_generated() increments counter without error."""
        record_report_generated("full_lineage", "json")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_change_event(self):
        """record_change_event() increments counter without error."""
        record_change_event("node_added", "low")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_quality_score(self):
        """record_quality_score() increments counter without error."""
        record_quality_score("excellent")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_quality_score_all_tiers(self):
        """record_quality_score() accepts all 5 tiers."""
        for tier in ("excellent", "good", "fair", "poor", "critical"):
            record_quality_score(tier)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_graph_traversal_duration(self):
        """observe_graph_traversal_duration() records histogram without error."""
        observe_graph_traversal_duration(0.25)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_graph_traversal_duration_large(self):
        """observe_graph_traversal_duration() handles large values."""
        observe_graph_traversal_duration(29.5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_processing_duration(self):
        """observe_processing_duration() records histogram without error."""
        observe_processing_duration("asset_register", 0.05)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_processing_duration_various_ops(self):
        """observe_processing_duration() accepts various operation labels."""
        for op in ("asset_register", "transformation_capture", "edge_create",
                    "impact_analyze", "validate", "report_generate"):
            observe_processing_duration(op, 0.01)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_graph_node_count(self):
        """set_graph_node_count() sets gauge without error."""
        set_graph_node_count(42)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_graph_node_count_zero(self):
        """set_graph_node_count() accepts zero."""
        set_graph_node_count(0)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_graph_edge_count(self):
        """set_graph_edge_count() sets gauge without error."""
        set_graph_edge_count(100)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_graph_edge_count_zero(self):
        """set_graph_edge_count() accepts zero."""
        set_graph_edge_count(0)


# ============================================================================
# TestHelperFunctionsWithoutPrometheus
# ============================================================================


class TestHelperFunctionsWithoutPrometheus:
    """Tests for helper functions when prometheus_client is NOT available.

    Verifies that all helpers are no-ops and do not crash.
    """

    def test_no_crash_record_asset_registered(self):
        """record_asset_registered() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_asset_registered("dataset", "internal")

    def test_no_crash_record_transformation_captured(self):
        """record_transformation_captured() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_transformation_captured("filter", "agent-1")

    def test_no_crash_record_edge_created(self):
        """record_edge_created() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_edge_created("dataset_level")

    def test_no_crash_record_impact_analysis(self):
        """record_impact_analysis() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_impact_analysis("forward", "medium")

    def test_no_crash_record_validation(self):
        """record_validation() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_validation("pass")

    def test_no_crash_record_report_generated(self):
        """record_report_generated() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_report_generated("custom", "json")

    def test_no_crash_record_change_event(self):
        """record_change_event() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_change_event("edge_added", "low")

    def test_no_crash_record_quality_score(self):
        """record_quality_score() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            record_quality_score("good")

    def test_no_crash_observe_graph_traversal_duration(self):
        """observe_graph_traversal_duration() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            observe_graph_traversal_duration(1.5)

    def test_no_crash_observe_processing_duration(self):
        """observe_processing_duration() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            observe_processing_duration("validate", 0.3)

    def test_no_crash_set_graph_node_count(self):
        """set_graph_node_count() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            set_graph_node_count(0)

    def test_no_crash_set_graph_edge_count(self):
        """set_graph_edge_count() does not crash without prometheus."""
        with patch.object(dlt_metrics, "PROMETHEUS_AVAILABLE", False):
            set_graph_edge_count(0)
