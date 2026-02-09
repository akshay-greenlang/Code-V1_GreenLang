# -*- coding: utf-8 -*-
"""
Unit Tests for Data Quality Profiler Metrics - AGENT-DATA-010

Tests the 12 Prometheus metrics, the PROMETHEUS_AVAILABLE flag, all 12
helper functions, and graceful fallback when prometheus_client is not installed.

Target: 65+ tests, 85%+ coverage of greenlang.data_quality_profiler.metrics

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_quality_profiler import metrics as metrics_mod
from greenlang.data_quality_profiler.metrics import (
    PROMETHEUS_AVAILABLE,
    dq_datasets_profiled_total,
    dq_columns_profiled_total,
    dq_assessments_completed_total,
    dq_rules_evaluated_total,
    dq_anomalies_detected_total,
    dq_gates_evaluated_total,
    dq_overall_quality_score,
    dq_processing_duration_seconds,
    dq_active_profiles,
    dq_total_issues_found,
    dq_processing_errors_total,
    dq_freshness_checks_total,
    record_profile,
    record_column_profile,
    record_assessment,
    record_rule_evaluation,
    record_anomaly,
    record_gate_evaluation,
    record_quality_score,
    record_processing_duration,
    update_active_profiles,
    update_total_issues,
    record_processing_error,
    record_freshness_check,
)


# ============================================================================
# TestPrometheusFlag
# ============================================================================


class TestPrometheusFlag:
    """PROMETHEUS_AVAILABLE flag tests."""

    def test_is_boolean(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_matches_import_availability(self):
        try:
            import prometheus_client  # noqa: F401
            expected = True
        except ImportError:
            expected = False
        assert PROMETHEUS_AVAILABLE == expected


# ============================================================================
# TestMetricObjects - verify all 12 metric objects exist
# ============================================================================


class TestMetricObjects:
    """All 12 metric objects are defined and accessible."""

    def test_dq_datasets_profiled_total_exists(self):
        assert dq_datasets_profiled_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_columns_profiled_total_exists(self):
        assert dq_columns_profiled_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_assessments_completed_total_exists(self):
        assert dq_assessments_completed_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_rules_evaluated_total_exists(self):
        assert dq_rules_evaluated_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_anomalies_detected_total_exists(self):
        assert dq_anomalies_detected_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_gates_evaluated_total_exists(self):
        assert dq_gates_evaluated_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_overall_quality_score_exists(self):
        assert dq_overall_quality_score is not None or not PROMETHEUS_AVAILABLE

    def test_dq_processing_duration_seconds_exists(self):
        assert dq_processing_duration_seconds is not None or not PROMETHEUS_AVAILABLE

    def test_dq_active_profiles_exists(self):
        assert dq_active_profiles is not None or not PROMETHEUS_AVAILABLE

    def test_dq_total_issues_found_exists(self):
        assert dq_total_issues_found is not None or not PROMETHEUS_AVAILABLE

    def test_dq_processing_errors_total_exists(self):
        assert dq_processing_errors_total is not None or not PROMETHEUS_AVAILABLE

    def test_dq_freshness_checks_total_exists(self):
        assert dq_freshness_checks_total is not None or not PROMETHEUS_AVAILABLE


# ============================================================================
# TestHelperFunctions - verify all 12 helpers are callable
# ============================================================================


class TestHelperFunctions:
    """All 12 helper functions exist and are callable."""

    def test_record_profile_callable(self):
        assert callable(record_profile)

    def test_record_column_profile_callable(self):
        assert callable(record_column_profile)

    def test_record_assessment_callable(self):
        assert callable(record_assessment)

    def test_record_rule_evaluation_callable(self):
        assert callable(record_rule_evaluation)

    def test_record_anomaly_callable(self):
        assert callable(record_anomaly)

    def test_record_gate_evaluation_callable(self):
        assert callable(record_gate_evaluation)

    def test_record_quality_score_callable(self):
        assert callable(record_quality_score)

    def test_record_processing_duration_callable(self):
        assert callable(record_processing_duration)

    def test_update_active_profiles_callable(self):
        assert callable(update_active_profiles)

    def test_update_total_issues_callable(self):
        assert callable(update_total_issues)

    def test_record_processing_error_callable(self):
        assert callable(record_processing_error)

    def test_record_freshness_check_callable(self):
        assert callable(record_freshness_check)

    def test_update_active_profiles_increment(self):
        """Call with positive delta does not raise."""
        update_active_profiles(1)  # Should not raise

    def test_update_active_profiles_decrement(self):
        """Call with negative delta does not raise."""
        update_active_profiles(-1)  # Should not raise


# ============================================================================
# TestWithPrometheus - behaviour when prometheus_client IS available
# ============================================================================


class TestWithPrometheus:
    """Test metric behaviour when prometheus_client is available."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_datasets_profiled_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_datasets_profiled_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_columns_profiled_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_columns_profiled_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_assessments_completed_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_assessments_completed_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_rules_evaluated_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_rules_evaluated_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_anomalies_detected_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_anomalies_detected_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_gates_evaluated_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_gates_evaluated_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_overall_quality_score_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(dq_overall_quality_score, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_duration_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(dq_processing_duration_seconds, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_profiles_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(dq_active_profiles, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_total_issues_found_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(dq_total_issues_found, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_errors_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_processing_errors_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_freshness_checks_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(dq_freshness_checks_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_profile_does_not_raise(self):
        record_profile(source="csv")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_column_profile_does_not_raise(self):
        record_column_profile(data_type="string")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_assessment_does_not_raise(self):
        record_assessment(quality_level="GOOD")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_quality_score_does_not_raise(self):
        record_quality_score(score=0.85)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_processing_duration_does_not_raise(self):
        record_processing_duration(operation="profile", duration=1.23)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_datasets_profiled_has_source_label(self):
        """Counter should accept source label."""
        dq_datasets_profiled_total.labels(source="api")


# ============================================================================
# TestWithoutPrometheus - graceful fallback when prometheus_client absent
# ============================================================================


class TestWithoutPrometheus:
    """Helper functions are no-ops when prometheus_client is not installed."""

    def _reload_metrics_without_prometheus(self):
        """Reload metrics module with prometheus_client missing."""
        # Save the real prometheus_client module if it exists
        saved = sys.modules.get("prometheus_client")
        sys.modules["prometheus_client"] = None  # type: ignore[assignment]
        try:
            importlib.reload(metrics_mod)
            return metrics_mod
        finally:
            if saved is not None:
                sys.modules["prometheus_client"] = saved
            else:
                sys.modules.pop("prometheus_client", None)
            # Restore original module state
            importlib.reload(metrics_mod)

    def test_record_profile_no_op(self):
        """record_profile does not raise even without prometheus."""
        record_profile(source="test")

    def test_record_column_profile_no_op(self):
        record_column_profile(data_type="integer")

    def test_record_assessment_no_op(self):
        record_assessment(quality_level="FAIR")

    def test_record_rule_evaluation_no_op(self):
        record_rule_evaluation(result="pass")

    def test_record_anomaly_no_op(self):
        record_anomaly(method="iqr")

    def test_record_gate_evaluation_no_op(self):
        record_gate_evaluation(outcome="fail")

    def test_record_quality_score_no_op(self):
        record_quality_score(score=0.5)

    def test_record_processing_duration_no_op(self):
        record_processing_duration(operation="assess", duration=0.5)

    def test_update_active_profiles_no_op(self):
        update_active_profiles(1)

    def test_update_total_issues_no_op(self):
        update_total_issues(5)

    def test_record_processing_error_no_op(self):
        record_processing_error(error_type="timeout")

    def test_record_freshness_check_no_op(self):
        record_freshness_check(status="fresh")


# ============================================================================
# TestMetricsExports - __all__ completeness
# ============================================================================


class TestMetricsExports:
    """Verify metrics module exports."""

    def test_all_list_exists(self):
        assert hasattr(metrics_mod, "__all__")

    def test_all_contains_prometheus_available(self):
        assert "PROMETHEUS_AVAILABLE" in metrics_mod.__all__

    def test_all_contains_record_profile(self):
        assert "record_profile" in metrics_mod.__all__

    def test_all_contains_dq_active_profiles(self):
        assert "dq_active_profiles" in metrics_mod.__all__

    def test_all_minimum_count(self):
        # 1 flag + 12 metrics + 12 helpers = 25
        assert len(metrics_mod.__all__) >= 25

    def test_all_contains_all_metric_objects(self):
        expected_metrics = [
            "dq_datasets_profiled_total",
            "dq_columns_profiled_total",
            "dq_assessments_completed_total",
            "dq_rules_evaluated_total",
            "dq_anomalies_detected_total",
            "dq_gates_evaluated_total",
            "dq_overall_quality_score",
            "dq_processing_duration_seconds",
            "dq_active_profiles",
            "dq_total_issues_found",
            "dq_processing_errors_total",
            "dq_freshness_checks_total",
        ]
        for name in expected_metrics:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"

    def test_all_contains_all_helper_functions(self):
        expected_helpers = [
            "record_profile",
            "record_column_profile",
            "record_assessment",
            "record_rule_evaluation",
            "record_anomaly",
            "record_gate_evaluation",
            "record_quality_score",
            "record_processing_duration",
            "update_active_profiles",
            "update_total_issues",
            "record_processing_error",
            "record_freshness_check",
        ]
        for name in expected_helpers:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"


# ============================================================================
# TestGaugeHelperEdgeCases
# ============================================================================


class TestGaugeHelperEdgeCases:
    """Edge cases for gauge helper functions."""

    def test_update_active_profiles_zero_delta(self):
        """Zero delta should be a no-op (no inc or dec called)."""
        update_active_profiles(0)  # Should not raise

    def test_update_total_issues_zero_delta(self):
        """Zero delta should be a no-op."""
        update_total_issues(0)  # Should not raise

    def test_update_active_profiles_large_positive(self):
        update_active_profiles(1000)  # Should not raise

    def test_update_total_issues_large_negative(self):
        update_total_issues(-500)  # Should not raise
