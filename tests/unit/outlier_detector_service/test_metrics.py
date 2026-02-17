# -*- coding: utf-8 -*-
"""
Unit tests for Outlier Detection Prometheus metrics - AGENT-DATA-013

Tests all 12 metrics, 12 helper functions, and PROMETHEUS_AVAILABLE flag.
Target: 20+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.outlier_detector.metrics import (
    PROMETHEUS_AVAILABLE,
    inc_jobs,
    inc_outliers_detected,
    inc_outliers_classified,
    inc_treatments,
    inc_thresholds,
    inc_feedback,
    inc_errors,
    observe_ensemble_score,
    observe_duration,
    observe_confidence,
    set_active_jobs,
    set_total_outliers_flagged,
    od_jobs_processed_total,
    od_outliers_detected_total,
    od_outliers_classified_total,
    od_treatments_applied_total,
    od_thresholds_evaluated_total,
    od_feedback_received_total,
    od_processing_errors_total,
    od_ensemble_score,
    od_processing_duration_seconds,
    od_detection_confidence,
    od_active_jobs,
    od_total_outliers_flagged,
)


class TestPrometheusAvailable:
    """Test the PROMETHEUS_AVAILABLE flag."""

    def test_flag_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


class TestMetricObjects:
    """Test metric objects are defined (may be None if prometheus not installed)."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_counter_objects_not_none(self):
        assert od_jobs_processed_total is not None
        assert od_outliers_detected_total is not None
        assert od_outliers_classified_total is not None
        assert od_treatments_applied_total is not None
        assert od_thresholds_evaluated_total is not None
        assert od_feedback_received_total is not None
        assert od_processing_errors_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_histogram_objects_not_none(self):
        assert od_ensemble_score is not None
        assert od_processing_duration_seconds is not None
        assert od_detection_confidence is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_gauge_objects_not_none(self):
        assert od_active_jobs is not None
        assert od_total_outliers_flagged is not None


class TestHelperFunctions:
    """Test all 12 helper functions execute without error."""

    def test_inc_jobs_no_error(self):
        inc_jobs("completed")

    def test_inc_outliers_detected_no_error(self):
        inc_outliers_detected("iqr", 5)

    def test_inc_outliers_classified_no_error(self):
        inc_outliers_classified("error", 3)

    def test_inc_treatments_no_error(self):
        inc_treatments("cap", 2)

    def test_inc_thresholds_no_error(self):
        inc_thresholds("domain", 1)

    def test_inc_feedback_no_error(self):
        inc_feedback("confirmed_outlier")

    def test_inc_errors_no_error(self):
        inc_errors("validation")

    def test_observe_ensemble_score_no_error(self):
        observe_ensemble_score("weighted_average", 0.85)

    def test_observe_duration_no_error(self):
        observe_duration("detect", 1.5)

    def test_observe_confidence_no_error(self):
        observe_confidence("iqr", 0.9)

    def test_set_active_jobs_no_error(self):
        set_active_jobs(3)

    def test_set_total_outliers_flagged_no_error(self):
        set_total_outliers_flagged(42)


class TestHelperFunctionsWithoutPrometheus:
    """Test helpers are safe when prometheus_client is unavailable."""

    def test_inc_jobs_when_unavailable(self):
        with patch("greenlang.outlier_detector.metrics.PROMETHEUS_AVAILABLE", False):
            inc_jobs("failed")  # Should not raise

    def test_observe_duration_when_unavailable(self):
        with patch("greenlang.outlier_detector.metrics.PROMETHEUS_AVAILABLE", False):
            observe_duration("classify", 0.5)  # Should not raise

    def test_set_active_jobs_when_unavailable(self):
        with patch("greenlang.outlier_detector.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_jobs(0)  # Should not raise

    def test_inc_outliers_detected_default_count(self):
        inc_outliers_detected("zscore")  # count defaults to 1, should not raise

    def test_inc_treatments_default_count(self):
        inc_treatments("flag")  # count defaults to 1

    def test_inc_thresholds_default_count(self):
        inc_thresholds("statistical")  # count defaults to 1

    def test_observe_ensemble_score_boundary(self):
        observe_ensemble_score("mean_score", 0.0)
        observe_ensemble_score("mean_score", 1.0)

    def test_observe_confidence_boundary(self):
        observe_confidence("mahalanobis", 0.0)
        observe_confidence("mahalanobis", 1.0)
