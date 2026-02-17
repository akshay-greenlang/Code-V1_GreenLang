# -*- coding: utf-8 -*-
"""
Unit tests for Time Series Gap Filler Prometheus metrics - AGENT-DATA-014

Tests all 12 metrics, 12 helper functions, and PROMETHEUS_AVAILABLE flag.
Target: 20+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.time_series_gap_filler.metrics import (
    PROMETHEUS_AVAILABLE,
    inc_jobs_processed,
    inc_gaps_detected,
    inc_gaps_filled,
    inc_validations,
    inc_frequencies,
    inc_strategies,
    observe_confidence,
    observe_duration,
    observe_gap_duration,
    set_active_jobs,
    set_gaps_open,
    inc_errors,
    tsgf_jobs_processed_total,
    tsgf_gaps_detected_total,
    tsgf_gaps_filled_total,
    tsgf_validations_passed_total,
    tsgf_frequencies_detected_total,
    tsgf_strategies_selected_total,
    tsgf_fill_confidence,
    tsgf_processing_duration_seconds,
    tsgf_gap_duration_seconds,
    tsgf_active_jobs,
    tsgf_total_gaps_open,
    tsgf_processing_errors_total,
)


# -----------------------------------------------------------------------
# 1. PROMETHEUS_AVAILABLE flag
# -----------------------------------------------------------------------


class TestPrometheusAvailable:
    """Test the PROMETHEUS_AVAILABLE flag."""

    def test_flag_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_flag_matches_import_success(self):
        """If we got here, prometheus_client imported or did not; flag is consistent."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            assert PROMETHEUS_AVAILABLE is False


# -----------------------------------------------------------------------
# 2. Metric object definitions
# -----------------------------------------------------------------------


class TestMetricObjects:
    """Test metric objects are defined (may be None if prometheus not installed)."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_counter_objects_not_none(self):
        assert tsgf_jobs_processed_total is not None
        assert tsgf_gaps_detected_total is not None
        assert tsgf_gaps_filled_total is not None
        assert tsgf_validations_passed_total is not None
        assert tsgf_frequencies_detected_total is not None
        assert tsgf_strategies_selected_total is not None
        assert tsgf_processing_errors_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_histogram_objects_not_none(self):
        assert tsgf_fill_confidence is not None
        assert tsgf_processing_duration_seconds is not None
        assert tsgf_gap_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_gauge_objects_not_none(self):
        assert tsgf_active_jobs is not None
        assert tsgf_total_gaps_open is not None

    @pytest.mark.skipif(PROMETHEUS_AVAILABLE, reason="prometheus_client is installed")
    def test_objects_none_when_unavailable(self):
        """When prometheus_client is absent, metric objects are None."""
        assert tsgf_jobs_processed_total is None
        assert tsgf_fill_confidence is None
        assert tsgf_active_jobs is None


# -----------------------------------------------------------------------
# 3. Helper functions execute without errors
# -----------------------------------------------------------------------


class TestHelperFunctions:
    """Test all 12 helper functions execute without error."""

    def test_inc_jobs_processed_no_error(self):
        inc_jobs_processed("completed")

    def test_inc_gaps_detected_no_error(self):
        inc_gaps_detected("missing", 5)

    def test_inc_gaps_filled_no_error(self):
        inc_gaps_filled("linear", 3)

    def test_inc_validations_no_error(self):
        inc_validations("passed", 2)

    def test_inc_frequencies_no_error(self):
        inc_frequencies("daily", 1)

    def test_inc_strategies_no_error(self):
        inc_strategies("interpolation", 4)

    def test_observe_confidence_no_error(self):
        observe_confidence(0.85)

    def test_observe_duration_no_error(self):
        observe_duration("detect_gaps", 1.5)

    def test_observe_gap_duration_no_error(self):
        observe_gap_duration(3600.0)

    def test_set_active_jobs_no_error(self):
        set_active_jobs(3)

    def test_set_gaps_open_no_error(self):
        set_gaps_open(42)

    def test_inc_errors_no_error(self):
        inc_errors("validation")


# -----------------------------------------------------------------------
# 4. Multiple calls accumulate correctly
# -----------------------------------------------------------------------


class TestAccumulation:
    """Test that multiple calls accumulate counters correctly."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_inc_jobs_processed_accumulates(self):
        """Calling inc_jobs_processed multiple times does not raise."""
        for status in ("completed", "failed", "cancelled", "timeout", "partial"):
            inc_jobs_processed(status)
        # No assertion on exact value because Prometheus collectors are global;
        # we simply verify no exception on repeated calls.

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_inc_gaps_detected_with_count(self):
        """Incrementing with count > 1 does not raise."""
        inc_gaps_detected("null", 10)
        inc_gaps_detected("null", 20)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_inc_gaps_filled_accumulates_multiple_methods(self):
        for method in ("linear", "spline", "forward_fill", "seasonal", "kalman"):
            inc_gaps_filled(method, 2)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_inc_validations_accumulates(self):
        inc_validations("passed", 5)
        inc_validations("failed", 3)
        inc_validations("warning", 1)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_inc_frequencies_accumulates(self):
        for level in ("minutely", "hourly", "daily", "weekly", "monthly"):
            inc_frequencies(level)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_inc_strategies_accumulates(self):
        for strategy in ("interpolation", "extrapolation", "imputation", "hybrid"):
            inc_strategies(strategy, 3)


# -----------------------------------------------------------------------
# 5. Stub mode - safe when prometheus_client not installed
# -----------------------------------------------------------------------


class TestStubModeWithoutPrometheus:
    """Test helpers are safe when prometheus_client is unavailable (stub mode)."""

    def test_inc_jobs_processed_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_jobs_processed("failed")  # Should not raise

    def test_inc_gaps_detected_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_gaps_detected("missing", 10)

    def test_inc_gaps_filled_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_gaps_filled("spline", 5)

    def test_inc_validations_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_validations("skipped", 2)

    def test_inc_frequencies_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_frequencies("quarterly")

    def test_inc_strategies_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_strategies("model_based")

    def test_observe_confidence_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            observe_confidence(0.75)

    def test_observe_duration_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            observe_duration("fill_gaps", 2.0)

    def test_observe_gap_duration_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            observe_gap_duration(86400.0)

    def test_set_active_jobs_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_jobs(0)

    def test_set_gaps_open_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            set_gaps_open(0)

    def test_inc_errors_when_unavailable(self):
        with patch("greenlang.time_series_gap_filler.metrics.PROMETHEUS_AVAILABLE", False):
            inc_errors("timeout")


# -----------------------------------------------------------------------
# 6. Default count parameters
# -----------------------------------------------------------------------


class TestDefaultCountParameters:
    """Test that count parameters default to 1."""

    def test_inc_gaps_detected_default_count(self):
        inc_gaps_detected("irregular")  # count defaults to 1

    def test_inc_gaps_filled_default_count(self):
        inc_gaps_filled("mean")  # count defaults to 1

    def test_inc_validations_default_count(self):
        inc_validations("passed")  # count defaults to 1

    def test_inc_frequencies_default_count(self):
        inc_frequencies("yearly")  # count defaults to 1

    def test_inc_strategies_default_count(self):
        inc_strategies("rule_based")  # count defaults to 1


# -----------------------------------------------------------------------
# 7. Boundary values
# -----------------------------------------------------------------------


class TestBoundaryValues:
    """Test histogram helpers with boundary values."""

    def test_observe_confidence_zero(self):
        observe_confidence(0.0)

    def test_observe_confidence_one(self):
        observe_confidence(1.0)

    def test_observe_confidence_mid(self):
        observe_confidence(0.5)

    def test_observe_duration_very_small(self):
        observe_duration("validate", 0.001)

    def test_observe_duration_large(self):
        observe_duration("pipeline", 300.0)

    def test_observe_gap_duration_one_minute(self):
        observe_gap_duration(60.0)

    def test_observe_gap_duration_one_month(self):
        observe_gap_duration(2592000.0)

    def test_set_active_jobs_zero(self):
        set_active_jobs(0)

    def test_set_gaps_open_zero(self):
        set_gaps_open(0)

    def test_set_active_jobs_high(self):
        set_active_jobs(10000)

    def test_set_gaps_open_high(self):
        set_gaps_open(99999)


# -----------------------------------------------------------------------
# 8. Prometheus export format
# -----------------------------------------------------------------------


class TestPrometheusExport:
    """Test Prometheus export integration (when prometheus_client is available)."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_generate_latest_includes_tsgf_metrics(self):
        """Verify generate_latest produces output containing our metric names."""
        from prometheus_client import generate_latest

        # Trigger at least one observation per metric type
        inc_jobs_processed("completed")
        observe_confidence(0.9)
        set_active_jobs(1)

        output = generate_latest().decode("utf-8")
        assert "gl_tsgf_jobs_processed_total" in output
        assert "gl_tsgf_fill_confidence" in output
        assert "gl_tsgf_active_jobs" in output

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_generate_latest_includes_all_counter_names(self):
        """Verify all counter metric names appear in Prometheus output."""
        from prometheus_client import generate_latest

        inc_gaps_detected("block")
        inc_gaps_filled("ensemble")
        inc_validations("failed")
        inc_frequencies("hourly")
        inc_strategies("seasonal_decomposition")
        inc_errors("data")

        output = generate_latest().decode("utf-8")
        assert "gl_tsgf_gaps_detected_total" in output
        assert "gl_tsgf_gaps_filled_total" in output
        assert "gl_tsgf_validations_passed_total" in output
        assert "gl_tsgf_frequencies_detected_total" in output
        assert "gl_tsgf_strategies_selected_total" in output
        assert "gl_tsgf_processing_errors_total" in output

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_generate_latest_includes_histograms_and_gauges(self):
        """Verify histogram and gauge metric names appear in Prometheus output."""
        from prometheus_client import generate_latest

        observe_duration("batch", 5.0)
        observe_gap_duration(7200.0)
        set_gaps_open(10)

        output = generate_latest().decode("utf-8")
        assert "gl_tsgf_processing_duration_seconds" in output
        assert "gl_tsgf_gap_duration_seconds" in output
        assert "gl_tsgf_total_gaps_open" in output


# -----------------------------------------------------------------------
# 9. All label values from docstrings
# -----------------------------------------------------------------------


class TestAllLabelValues:
    """Test that every documented label value is accepted without error."""

    def test_all_job_statuses(self):
        for status in ("completed", "failed", "cancelled", "timeout", "partial"):
            inc_jobs_processed(status)

    def test_all_gap_types(self):
        for gap_type in ("missing", "null", "irregular", "duplicated",
                         "truncated", "sparse", "block"):
            inc_gaps_detected(gap_type)

    def test_all_fill_methods(self):
        for method in ("linear", "spline", "forward_fill", "backward_fill",
                       "mean", "median", "seasonal", "kalman", "regression",
                       "ensemble", "custom"):
            inc_gaps_filled(method)

    def test_all_validation_results(self):
        for result in ("passed", "failed", "warning", "skipped"):
            inc_validations(result)

    def test_all_frequency_levels(self):
        for level in ("sub_minute", "minutely", "hourly", "daily", "weekly",
                      "monthly", "quarterly", "yearly", "irregular", "unknown"):
            inc_frequencies(level)

    def test_all_strategies(self):
        for strategy in ("interpolation", "extrapolation", "imputation",
                         "seasonal_decomposition", "model_based",
                         "rule_based", "hybrid", "manual"):
            inc_strategies(strategy)

    def test_all_operation_types(self):
        for operation in ("detect_gaps", "detect_frequency", "select_strategy",
                          "fill_gaps", "validate", "report", "pipeline",
                          "batch", "export"):
            observe_duration(operation, 0.1)

    def test_all_error_types(self):
        for error_type in ("validation", "timeout", "data", "integration",
                           "detection", "frequency", "strategy", "fill",
                           "interpolation", "extrapolation", "unknown"):
            inc_errors(error_type)
