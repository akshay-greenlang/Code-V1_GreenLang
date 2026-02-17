# -*- coding: utf-8 -*-
"""
Unit tests for Missing Value Imputer Prometheus metrics - AGENT-DATA-012

Tests all 12 metrics, 12 helper functions, and PROMETHEUS_AVAILABLE flag.
Target: 20+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.missing_value_imputer.metrics import (
    PROMETHEUS_AVAILABLE,
    inc_jobs,
    inc_values_imputed,
    inc_analyses,
    inc_validations,
    inc_rules_evaluated,
    inc_strategies_selected,
    inc_errors,
    observe_confidence,
    observe_duration,
    observe_completeness_improvement,
    set_active_jobs,
    set_total_missing_detected,
    mvi_jobs_processed_total,
    mvi_values_imputed_total,
    mvi_analyses_completed_total,
    mvi_validations_passed_total,
    mvi_rules_evaluated_total,
    mvi_strategies_selected_total,
    mvi_processing_errors_total,
    mvi_confidence_score,
    mvi_processing_duration_seconds,
    mvi_completeness_improvement,
    mvi_active_jobs,
    mvi_total_missing_detected,
)


class TestPrometheusAvailable:
    """Test the PROMETHEUS_AVAILABLE flag."""

    def test_flag_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


class TestMetricObjects:
    """Test metric objects are defined (may be None if prometheus not installed)."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_counter_objects_not_none(self):
        assert mvi_jobs_processed_total is not None
        assert mvi_values_imputed_total is not None
        assert mvi_analyses_completed_total is not None
        assert mvi_validations_passed_total is not None
        assert mvi_rules_evaluated_total is not None
        assert mvi_strategies_selected_total is not None
        assert mvi_processing_errors_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_histogram_objects_not_none(self):
        assert mvi_confidence_score is not None
        assert mvi_processing_duration_seconds is not None
        assert mvi_completeness_improvement is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_gauge_objects_not_none(self):
        assert mvi_active_jobs is not None
        assert mvi_total_missing_detected is not None


class TestHelperFunctions:
    """Test all 12 helper functions execute without error."""

    def test_inc_jobs_no_error(self):
        inc_jobs("completed")

    def test_inc_values_imputed_no_error(self):
        inc_values_imputed("mean", 5)

    def test_inc_analyses_no_error(self):
        inc_analyses("mcar")

    def test_inc_validations_no_error(self):
        inc_validations("ks_test")

    def test_inc_rules_evaluated_no_error(self):
        inc_rules_evaluated("high", 3)

    def test_inc_strategies_selected_no_error(self):
        inc_strategies_selected("median")

    def test_inc_errors_no_error(self):
        inc_errors("validation")

    def test_observe_confidence_no_error(self):
        observe_confidence("knn", 0.85)

    def test_observe_duration_no_error(self):
        observe_duration("analyze", 1.5)

    def test_observe_completeness_improvement_no_error(self):
        observe_completeness_improvement("mean", 0.15)

    def test_set_active_jobs_no_error(self):
        set_active_jobs(3)

    def test_set_total_missing_detected_no_error(self):
        set_total_missing_detected(42)


class TestHelperFunctionsWithoutPrometheus:
    """Test helpers are safe when prometheus_client is unavailable."""

    def test_inc_jobs_when_unavailable(self):
        with patch("greenlang.missing_value_imputer.metrics.PROMETHEUS_AVAILABLE", False):
            inc_jobs("failed")  # Should not raise

    def test_observe_duration_when_unavailable(self):
        with patch("greenlang.missing_value_imputer.metrics.PROMETHEUS_AVAILABLE", False):
            observe_duration("impute", 0.5)  # Should not raise

    def test_set_active_jobs_when_unavailable(self):
        with patch("greenlang.missing_value_imputer.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_jobs(0)  # Should not raise

    def test_inc_values_imputed_default_count(self):
        inc_values_imputed("mode")  # count defaults to 1, should not raise
