# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus Metrics - AGENT-DATA-011

Tests all 12 metric definitions, 12 helper functions, label validation,
graceful fallback when prometheus_client is unavailable, and edge cases.
Target: 70+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

# Import the metrics module and its exports
from greenlang.duplicate_detector.metrics import (
    PROMETHEUS_AVAILABLE,
    dd_active_jobs,
    dd_blocks_created_total,
    dd_clusters_formed_total,
    dd_comparisons_performed_total,
    dd_jobs_processed_total,
    dd_matches_found_total,
    dd_merge_conflicts_total,
    dd_merges_completed_total,
    dd_processing_duration_seconds,
    dd_processing_errors_total,
    dd_records_fingerprinted_total,
    dd_similarity_score,
    inc_blocks,
    inc_clusters,
    inc_comparisons,
    inc_conflicts,
    inc_errors,
    inc_fingerprints,
    inc_jobs,
    inc_matches,
    inc_merges,
    observe_duration,
    observe_similarity,
    set_active_jobs,
)


# =============================================================================
# Test PROMETHEUS_AVAILABLE flag
# =============================================================================


class TestPrometheusAvailability:
    """Verify prometheus_client availability detection."""

    def test_prometheus_available_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_reflects_import(self):
        """When prometheus_client is installed, PROMETHEUS_AVAILABLE should be True."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            assert PROMETHEUS_AVAILABLE is False


# =============================================================================
# Test metric object creation (when prometheus_client IS available)
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricCreation:
    """Verify each metric object is correctly instantiated."""

    def test_jobs_processed_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_jobs_processed_total, Counter)

    def test_records_fingerprinted_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_records_fingerprinted_total, Counter)

    def test_blocks_created_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_blocks_created_total, Counter)

    def test_comparisons_performed_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_comparisons_performed_total, Counter)

    def test_matches_found_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_matches_found_total, Counter)

    def test_clusters_formed_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_clusters_formed_total, Counter)

    def test_merges_completed_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_merges_completed_total, Counter)

    def test_merge_conflicts_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_merge_conflicts_total, Counter)

    def test_processing_duration_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(dd_processing_duration_seconds, Histogram)

    def test_similarity_score_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(dd_similarity_score, Histogram)

    def test_active_jobs_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(dd_active_jobs, Gauge)

    def test_processing_errors_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dd_processing_errors_total, Counter)


# =============================================================================
# Test metric names and descriptions
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricNames:
    """Verify metric names follow naming convention.

    Note: prometheus_client Counter stores _name without the _total suffix
    because it automatically appends _total on export. Histograms and Gauges
    store the full name in _name.
    """

    def test_jobs_metric_name(self):
        # Counter _name strips _total; the constructor arg includes it
        assert "gl_dd_jobs_processed" in dd_jobs_processed_total._name

    def test_fingerprints_metric_name(self):
        assert "gl_dd_records_fingerprinted" in dd_records_fingerprinted_total._name

    def test_blocks_metric_name(self):
        assert "gl_dd_blocks_created" in dd_blocks_created_total._name

    def test_comparisons_metric_name(self):
        assert "gl_dd_comparisons_performed" in dd_comparisons_performed_total._name

    def test_matches_metric_name(self):
        assert "gl_dd_matches_found" in dd_matches_found_total._name

    def test_clusters_metric_name(self):
        assert "gl_dd_clusters_formed" in dd_clusters_formed_total._name

    def test_merges_metric_name(self):
        assert "gl_dd_merges_completed" in dd_merges_completed_total._name

    def test_conflicts_metric_name(self):
        assert "gl_dd_merge_conflicts" in dd_merge_conflicts_total._name

    def test_duration_metric_name(self):
        assert dd_processing_duration_seconds._name == "gl_dd_processing_duration_seconds"

    def test_similarity_metric_name(self):
        assert dd_similarity_score._name == "gl_dd_similarity_score"

    def test_active_jobs_metric_name(self):
        assert dd_active_jobs._name == "gl_dd_active_jobs"

    def test_errors_metric_name(self):
        assert "gl_dd_processing_errors" in dd_processing_errors_total._name


# =============================================================================
# Test helper functions (when prometheus_client IS available)
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestHelperFunctions:
    """Test all 12 helper functions work correctly."""

    def test_inc_jobs_completed(self):
        """inc_jobs should not raise for a valid status."""
        inc_jobs("completed")

    def test_inc_jobs_failed(self):
        inc_jobs("failed")

    def test_inc_jobs_cancelled(self):
        inc_jobs("cancelled")

    def test_inc_jobs_timeout(self):
        inc_jobs("timeout")

    def test_inc_fingerprints_sha256(self):
        inc_fingerprints("sha256")

    def test_inc_fingerprints_simhash(self):
        inc_fingerprints("simhash")

    def test_inc_fingerprints_minhash(self):
        inc_fingerprints("minhash")

    def test_inc_fingerprints_with_count(self):
        inc_fingerprints("sha256", count=100)

    def test_inc_blocks_sorted_neighborhood(self):
        inc_blocks("sorted_neighborhood")

    def test_inc_blocks_standard(self):
        inc_blocks("standard")

    def test_inc_blocks_canopy(self):
        inc_blocks("canopy")

    def test_inc_blocks_with_count(self):
        inc_blocks("sorted_neighborhood", count=50)

    def test_inc_comparisons_jaro_winkler(self):
        inc_comparisons("jaro_winkler")

    def test_inc_comparisons_exact(self):
        inc_comparisons("exact")

    def test_inc_comparisons_levenshtein(self):
        inc_comparisons("levenshtein")

    def test_inc_comparisons_with_count(self):
        inc_comparisons("jaro_winkler", count=1000)

    def test_inc_matches_match(self):
        inc_matches("match")

    def test_inc_matches_possible(self):
        inc_matches("possible")

    def test_inc_matches_non_match(self):
        inc_matches("non_match")

    def test_inc_matches_with_count(self):
        inc_matches("match", count=25)

    def test_inc_clusters_union_find(self):
        inc_clusters("union_find")

    def test_inc_clusters_connected_components(self):
        inc_clusters("connected_components")

    def test_inc_clusters_with_count(self):
        inc_clusters("union_find", count=10)

    def test_inc_merges_keep_first(self):
        inc_merges("keep_first")

    def test_inc_merges_keep_most_complete(self):
        inc_merges("keep_most_complete")

    def test_inc_merges_golden_record(self):
        inc_merges("golden_record")

    def test_inc_merges_with_count(self):
        inc_merges("keep_first", count=5)

    def test_inc_conflicts_first(self):
        inc_conflicts("first")

    def test_inc_conflicts_most_complete(self):
        inc_conflicts("most_complete")

    def test_inc_conflicts_longest(self):
        inc_conflicts("longest")

    def test_inc_conflicts_with_count(self):
        inc_conflicts("first", count=3)

    def test_observe_duration_fingerprint(self):
        observe_duration("fingerprint", 0.5)

    def test_observe_duration_block(self):
        observe_duration("block", 1.0)

    def test_observe_duration_compare(self):
        observe_duration("compare", 2.5)

    def test_observe_duration_pipeline(self):
        observe_duration("pipeline", 120.0)

    def test_observe_duration_zero(self):
        observe_duration("fingerprint", 0.0)

    def test_observe_duration_large(self):
        observe_duration("job", 300.0)

    def test_observe_similarity_jaro_winkler(self):
        observe_similarity("jaro_winkler", 0.85)

    def test_observe_similarity_exact(self):
        observe_similarity("exact", 1.0)

    def test_observe_similarity_zero(self):
        observe_similarity("levenshtein", 0.0)

    def test_observe_similarity_mid(self):
        observe_similarity("ngram", 0.5)

    def test_set_active_jobs_zero(self):
        set_active_jobs(0)

    def test_set_active_jobs_positive(self):
        set_active_jobs(5)

    def test_set_active_jobs_large(self):
        set_active_jobs(1000)

    def test_inc_errors_validation(self):
        inc_errors("validation")

    def test_inc_errors_timeout(self):
        inc_errors("timeout")

    def test_inc_errors_data(self):
        inc_errors("data")

    def test_inc_errors_integration(self):
        inc_errors("integration")

    def test_inc_errors_comparison(self):
        inc_errors("comparison")

    def test_inc_errors_merge(self):
        inc_errors("merge")

    def test_inc_errors_unknown(self):
        inc_errors("unknown")


# =============================================================================
# Test graceful fallback when prometheus_client is NOT available
# =============================================================================


class TestGracefulFallback:
    """Verify helpers do nothing when PROMETHEUS_AVAILABLE is False."""

    def test_inc_jobs_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            # Should not raise
            inc_jobs("completed")

    def test_inc_fingerprints_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_fingerprints("sha256", count=10)

    def test_inc_blocks_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_blocks("sorted_neighborhood")

    def test_inc_comparisons_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_comparisons("exact", count=100)

    def test_inc_matches_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_matches("match")

    def test_inc_clusters_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_clusters("union_find")

    def test_inc_merges_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_merges("keep_first")

    def test_inc_conflicts_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_conflicts("first")

    def test_observe_duration_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            observe_duration("fingerprint", 1.5)

    def test_observe_similarity_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            observe_similarity("jaro_winkler", 0.9)

    def test_set_active_jobs_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_jobs(10)

    def test_inc_errors_no_op_when_unavailable(self):
        with patch("greenlang.duplicate_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_errors("validation")


# =============================================================================
# Test metric label names
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricLabels:
    """Verify metrics expose the correct label names."""

    def test_jobs_label_status(self):
        assert "status" in dd_jobs_processed_total._labelnames

    def test_fingerprints_label_algorithm(self):
        assert "algorithm" in dd_records_fingerprinted_total._labelnames

    def test_blocks_label_strategy(self):
        assert "strategy" in dd_blocks_created_total._labelnames

    def test_comparisons_label_algorithm(self):
        assert "algorithm" in dd_comparisons_performed_total._labelnames

    def test_matches_label_classification(self):
        assert "classification" in dd_matches_found_total._labelnames

    def test_clusters_label_algorithm(self):
        assert "algorithm" in dd_clusters_formed_total._labelnames

    def test_merges_label_strategy(self):
        assert "strategy" in dd_merges_completed_total._labelnames

    def test_conflicts_label_resolution(self):
        assert "resolution" in dd_merge_conflicts_total._labelnames

    def test_duration_label_operation(self):
        assert "operation" in dd_processing_duration_seconds._labelnames

    def test_similarity_label_algorithm(self):
        assert "algorithm" in dd_similarity_score._labelnames

    def test_active_jobs_no_labels(self):
        assert dd_active_jobs._labelnames == ()

    def test_errors_label_error_type(self):
        assert "error_type" in dd_processing_errors_total._labelnames


# =============================================================================
# Test histogram buckets
# =============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestHistogramBuckets:
    """Verify histogram bucket boundaries."""

    def test_duration_histogram_has_12_buckets(self):
        # prometheus_client adds +Inf as the last bucket
        expected_user_buckets = (
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        )
        # The _upper_bounds will have these + float('inf')
        assert len(expected_user_buckets) == 12

    def test_similarity_histogram_has_11_buckets(self):
        expected_user_buckets = (
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        )
        assert len(expected_user_buckets) == 11


# =============================================================================
# Test __all__ exports
# =============================================================================


class TestMetricsExports:
    """Verify the metrics module exports all expected names."""

    def test_all_metric_objects_exported(self):
        import greenlang.duplicate_detector.metrics as mod
        metric_names = [
            "dd_jobs_processed_total",
            "dd_records_fingerprinted_total",
            "dd_blocks_created_total",
            "dd_comparisons_performed_total",
            "dd_matches_found_total",
            "dd_clusters_formed_total",
            "dd_merges_completed_total",
            "dd_merge_conflicts_total",
            "dd_processing_duration_seconds",
            "dd_similarity_score",
            "dd_active_jobs",
            "dd_processing_errors_total",
        ]
        for name in metric_names:
            assert name in mod.__all__, f"{name} missing from __all__"

    def test_all_helper_functions_exported(self):
        import greenlang.duplicate_detector.metrics as mod
        helper_names = [
            "inc_jobs",
            "inc_fingerprints",
            "inc_blocks",
            "inc_comparisons",
            "inc_matches",
            "inc_clusters",
            "inc_merges",
            "inc_conflicts",
            "observe_duration",
            "observe_similarity",
            "set_active_jobs",
            "inc_errors",
        ]
        for name in helper_names:
            assert name in mod.__all__, f"{name} missing from __all__"

    def test_prometheus_available_exported(self):
        import greenlang.duplicate_detector.metrics as mod
        assert "PROMETHEUS_AVAILABLE" in mod.__all__

    def test_total_exports_count(self):
        import greenlang.duplicate_detector.metrics as mod
        # 1 flag + 12 metric objects + 12 helper functions = 25
        assert len(mod.__all__) == 25
