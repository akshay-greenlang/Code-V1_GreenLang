# -*- coding: utf-8 -*-
"""
Unit Tests for Data Freshness Monitor Metrics - AGENT-DATA-016

Tests the 12 Prometheus metrics, the PROMETHEUS_AVAILABLE flag, all 12
helper functions, DummyCounter/DummyHistogram/DummyGauge fallback classes,
metric labels, metric descriptions, and graceful fallback when
prometheus_client is not installed.

Target: 80+ tests, 85%+ coverage of greenlang.data_freshness_monitor.metrics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from greenlang.data_freshness_monitor import metrics as metrics_mod
from greenlang.data_freshness_monitor.metrics import (
    PROMETHEUS_AVAILABLE,
    DummyCounter,
    DummyGauge,
    DummyHistogram,
    _DummyLabeled,
    dfm_active_breaches,
    dfm_alerts_sent_total,
    dfm_checks_performed_total,
    dfm_data_age_hours,
    dfm_datasets_registered_total,
    dfm_freshness_score,
    dfm_monitored_datasets,
    dfm_predictions_made_total,
    dfm_processing_duration_seconds,
    dfm_processing_errors_total,
    dfm_refresh_events_total,
    dfm_sla_breaches_total,
    observe_data_age,
    observe_duration,
    observe_freshness_score,
    record_alert,
    record_breach,
    record_check,
    record_dataset_registered,
    record_error,
    record_prediction,
    record_refresh_event,
    set_active_breaches,
    set_monitored_datasets,
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

    def test_flag_is_module_level(self):
        assert hasattr(metrics_mod, "PROMETHEUS_AVAILABLE")

    def test_flag_in_all_exports(self):
        assert "PROMETHEUS_AVAILABLE" in metrics_mod.__all__


# ============================================================================
# TestMetricObjectsExist - verify all 12 metric objects exist
# ============================================================================


class TestMetricObjectsExist:
    """All 12 metric objects are defined and accessible."""

    def test_dfm_checks_performed_total_exists(self):
        assert dfm_checks_performed_total is not None

    def test_dfm_sla_breaches_total_exists(self):
        assert dfm_sla_breaches_total is not None

    def test_dfm_alerts_sent_total_exists(self):
        assert dfm_alerts_sent_total is not None

    def test_dfm_datasets_registered_total_exists(self):
        assert dfm_datasets_registered_total is not None

    def test_dfm_refresh_events_total_exists(self):
        assert dfm_refresh_events_total is not None

    def test_dfm_predictions_made_total_exists(self):
        assert dfm_predictions_made_total is not None

    def test_dfm_freshness_score_exists(self):
        assert dfm_freshness_score is not None

    def test_dfm_data_age_hours_exists(self):
        assert dfm_data_age_hours is not None

    def test_dfm_processing_duration_seconds_exists(self):
        assert dfm_processing_duration_seconds is not None

    def test_dfm_active_breaches_exists(self):
        assert dfm_active_breaches is not None

    def test_dfm_monitored_datasets_exists(self):
        assert dfm_monitored_datasets is not None

    def test_dfm_processing_errors_total_exists(self):
        assert dfm_processing_errors_total is not None


# ============================================================================
# TestMetricTypes - verify correct types when prometheus_client IS available
# ============================================================================


class TestMetricTypes:
    """Test metric types when prometheus_client is available."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_checks_performed_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_checks_performed_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sla_breaches_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_sla_breaches_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_alerts_sent_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_alerts_sent_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_datasets_registered_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_datasets_registered_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_refresh_events_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_refresh_events_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_predictions_made_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_predictions_made_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_freshness_score_is_histogram(self):
        from prometheus_client import Histogram

        assert isinstance(dfm_freshness_score, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_data_age_hours_is_histogram(self):
        from prometheus_client import Histogram

        assert isinstance(dfm_data_age_hours, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_duration_is_histogram(self):
        from prometheus_client import Histogram

        assert isinstance(dfm_processing_duration_seconds, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_breaches_is_gauge(self):
        from prometheus_client import Gauge

        assert isinstance(dfm_active_breaches, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_monitored_datasets_is_gauge(self):
        from prometheus_client import Gauge

        assert isinstance(dfm_monitored_datasets, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_errors_is_counter(self):
        from prometheus_client import Counter

        assert isinstance(dfm_processing_errors_total, Counter)


# ============================================================================
# TestMetricLabels - verify label acceptance when prometheus_client available
# ============================================================================


class TestMetricLabels:
    """Verify metric label names are accepted."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_checks_performed_accepts_dataset_result_labels(self):
        dfm_checks_performed_total.labels(dataset="test_ds", result="fresh")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sla_breaches_accepts_severity_label(self):
        dfm_sla_breaches_total.labels(severity="critical")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_alerts_sent_accepts_channel_severity_labels(self):
        dfm_alerts_sent_total.labels(channel="slack", severity="high")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_datasets_registered_accepts_status_label(self):
        dfm_datasets_registered_total.labels(status="active")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_refresh_events_accepts_source_label(self):
        dfm_refresh_events_total.labels(source="erp")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_predictions_made_accepts_status_label(self):
        dfm_predictions_made_total.labels(status="accurate")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_freshness_score_accepts_dataset_label(self):
        dfm_freshness_score.labels(dataset="test_ds")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_data_age_hours_accepts_dataset_label(self):
        dfm_data_age_hours.labels(dataset="test_ds")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_duration_accepts_operation_label(self):
        dfm_processing_duration_seconds.labels(operation="check_freshness")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_errors_accepts_error_type_label(self):
        dfm_processing_errors_total.labels(error_type="timeout")


# ============================================================================
# TestMetricDescriptions - verify descriptions are non-empty
# ============================================================================


class TestMetricDescriptions:
    """Verify metric descriptions are non-empty when prometheus_client is available."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_checks_performed_description(self):
        assert dfm_checks_performed_total._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sla_breaches_description(self):
        assert dfm_sla_breaches_total._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_alerts_sent_description(self):
        assert dfm_alerts_sent_total._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_datasets_registered_description(self):
        assert dfm_datasets_registered_total._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_refresh_events_description(self):
        assert dfm_refresh_events_total._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_predictions_made_description(self):
        assert dfm_predictions_made_total._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_freshness_score_description(self):
        assert dfm_freshness_score._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_data_age_hours_description(self):
        assert dfm_data_age_hours._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_duration_description(self):
        assert dfm_processing_duration_seconds._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_breaches_description(self):
        assert dfm_active_breaches._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_monitored_datasets_description(self):
        assert dfm_monitored_datasets._documentation != ""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_errors_description(self):
        assert dfm_processing_errors_total._documentation != ""


# ============================================================================
# TestHelperFunctionsCallable - verify all 12 helpers are callable
# ============================================================================


class TestHelperFunctionsCallable:
    """All 12 helper functions exist and are callable."""

    def test_record_check_callable(self):
        assert callable(record_check)

    def test_record_breach_callable(self):
        assert callable(record_breach)

    def test_record_alert_callable(self):
        assert callable(record_alert)

    def test_record_dataset_registered_callable(self):
        assert callable(record_dataset_registered)

    def test_record_refresh_event_callable(self):
        assert callable(record_refresh_event)

    def test_record_prediction_callable(self):
        assert callable(record_prediction)

    def test_observe_freshness_score_callable(self):
        assert callable(observe_freshness_score)

    def test_observe_data_age_callable(self):
        assert callable(observe_data_age)

    def test_observe_duration_callable(self):
        assert callable(observe_duration)

    def test_set_active_breaches_callable(self):
        assert callable(set_active_breaches)

    def test_set_monitored_datasets_callable(self):
        assert callable(set_monitored_datasets)

    def test_record_error_callable(self):
        assert callable(record_error)


# ============================================================================
# TestHelperFunctionsInvocation - call helpers, verify no exceptions
# ============================================================================


class TestHelperFunctionsInvocation:
    """All 12 helper functions can be called without raising."""

    def test_record_check_does_not_raise(self):
        record_check(dataset="test_ds", result="fresh")

    def test_record_check_stale_result(self):
        record_check(dataset="test_ds", result="stale")

    def test_record_check_warning_result(self):
        record_check(dataset="test_ds", result="warning")

    def test_record_check_critical_result(self):
        record_check(dataset="test_ds", result="critical")

    def test_record_check_unknown_result(self):
        record_check(dataset="test_ds", result="unknown")

    def test_record_check_skipped_result(self):
        record_check(dataset="test_ds", result="skipped")

    def test_record_breach_does_not_raise(self):
        record_breach(severity="critical")

    def test_record_breach_with_count(self):
        record_breach(severity="high", count=3)

    def test_record_breach_default_count(self):
        record_breach(severity="low")

    def test_record_alert_does_not_raise(self):
        record_alert(channel="slack", severity="high")

    def test_record_alert_email_channel(self):
        record_alert(channel="email", severity="medium")

    def test_record_alert_pagerduty_channel(self):
        record_alert(channel="pagerduty", severity="critical")

    def test_record_dataset_registered_does_not_raise(self):
        record_dataset_registered(status="active")

    def test_record_dataset_registered_with_count(self):
        record_dataset_registered(status="inactive", count=5)

    def test_record_refresh_event_does_not_raise(self):
        record_refresh_event(source="erp")

    def test_record_refresh_event_with_count(self):
        record_refresh_event(source="api", count=10)

    def test_record_prediction_does_not_raise(self):
        record_prediction(status="accurate")

    def test_record_prediction_inaccurate(self):
        record_prediction(status="inaccurate")

    def test_observe_freshness_score_does_not_raise(self):
        observe_freshness_score(dataset="test_ds", score=0.95)

    def test_observe_freshness_score_zero(self):
        observe_freshness_score(dataset="test_ds", score=0.0)

    def test_observe_freshness_score_one(self):
        observe_freshness_score(dataset="test_ds", score=1.0)

    def test_observe_data_age_does_not_raise(self):
        observe_data_age(dataset="test_ds", age_hours=24.5)

    def test_observe_data_age_zero(self):
        observe_data_age(dataset="test_ds", age_hours=0.0)

    def test_observe_duration_does_not_raise(self):
        observe_duration(operation="check_freshness", duration=0.123)

    def test_observe_duration_large_value(self):
        observe_duration(operation="evaluate_sla", duration=59.99)

    def test_set_active_breaches_does_not_raise(self):
        set_active_breaches(count=5)

    def test_set_active_breaches_zero(self):
        set_active_breaches(count=0)

    def test_set_monitored_datasets_does_not_raise(self):
        set_monitored_datasets(count=42)

    def test_set_monitored_datasets_zero(self):
        set_monitored_datasets(count=0)

    def test_record_error_does_not_raise(self):
        record_error(error_type="timeout")

    def test_record_error_validation_type(self):
        record_error(error_type="validation")

    def test_record_error_unknown_type(self):
        record_error(error_type="unknown")


# ============================================================================
# TestDummyLabeled - _DummyLabeled class
# ============================================================================


class TestDummyLabeled:
    """Tests for _DummyLabeled no-op labeled metric."""

    def test_inc_no_op(self):
        labeled = _DummyLabeled()
        result = labeled.inc()
        assert result is None

    def test_inc_with_amount(self):
        labeled = _DummyLabeled()
        result = labeled.inc(amount=5.0)
        assert result is None

    def test_observe_no_op(self):
        labeled = _DummyLabeled()
        result = labeled.observe(amount=1.23)
        assert result is None

    def test_set_no_op(self):
        labeled = _DummyLabeled()
        result = labeled.set(value=42.0)
        assert result is None


# ============================================================================
# TestDummyCounter - DummyCounter fallback class
# ============================================================================


class TestDummyCounter:
    """Tests for DummyCounter fallback class."""

    def test_instantiation(self):
        counter = DummyCounter()
        assert counter is not None

    def test_labels_returns_dummy_labeled(self):
        counter = DummyCounter()
        labeled = counter.labels(dataset="ds1", result="ok")
        assert isinstance(labeled, _DummyLabeled)

    def test_labels_with_no_kwargs(self):
        counter = DummyCounter()
        labeled = counter.labels()
        assert isinstance(labeled, _DummyLabeled)

    def test_inc_no_op(self):
        counter = DummyCounter()
        result = counter.inc()
        assert result is None

    def test_inc_with_amount(self):
        counter = DummyCounter()
        result = counter.inc(amount=10.0)
        assert result is None

    def test_labeled_inc(self):
        counter = DummyCounter()
        labeled = counter.labels(key="val")
        result = labeled.inc()
        assert result is None

    def test_labeled_inc_with_amount(self):
        counter = DummyCounter()
        labeled = counter.labels(key="val")
        result = labeled.inc(amount=5.0)
        assert result is None


# ============================================================================
# TestDummyHistogram - DummyHistogram fallback class
# ============================================================================


class TestDummyHistogram:
    """Tests for DummyHistogram fallback class."""

    def test_instantiation(self):
        histogram = DummyHistogram()
        assert histogram is not None

    def test_labels_returns_dummy_labeled(self):
        histogram = DummyHistogram()
        labeled = histogram.labels(dataset="ds1")
        assert isinstance(labeled, _DummyLabeled)

    def test_labels_with_no_kwargs(self):
        histogram = DummyHistogram()
        labeled = histogram.labels()
        assert isinstance(labeled, _DummyLabeled)

    def test_observe_no_op(self):
        histogram = DummyHistogram()
        result = histogram.observe(amount=0.5)
        assert result is None

    def test_labeled_observe(self):
        histogram = DummyHistogram()
        labeled = histogram.labels(dataset="ds1")
        result = labeled.observe(amount=1.23)
        assert result is None


# ============================================================================
# TestDummyGauge - DummyGauge fallback class
# ============================================================================


class TestDummyGauge:
    """Tests for DummyGauge fallback class."""

    def test_instantiation(self):
        gauge = DummyGauge()
        assert gauge is not None

    def test_labels_returns_dummy_labeled(self):
        gauge = DummyGauge()
        labeled = gauge.labels(key="val")
        assert isinstance(labeled, _DummyLabeled)

    def test_labels_with_no_kwargs(self):
        gauge = DummyGauge()
        labeled = gauge.labels()
        assert isinstance(labeled, _DummyLabeled)

    def test_set_no_op(self):
        gauge = DummyGauge()
        result = gauge.set(value=42.0)
        assert result is None

    def test_inc_no_op(self):
        gauge = DummyGauge()
        result = gauge.inc()
        assert result is None

    def test_inc_with_amount(self):
        gauge = DummyGauge()
        result = gauge.inc(amount=5.0)
        assert result is None

    def test_dec_no_op(self):
        gauge = DummyGauge()
        result = gauge.dec()
        assert result is None

    def test_dec_with_amount(self):
        gauge = DummyGauge()
        result = gauge.dec(amount=3.0)
        assert result is None

    def test_labeled_set(self):
        gauge = DummyGauge()
        labeled = gauge.labels(key="val")
        result = labeled.set(value=10.0)
        assert result is None


# ============================================================================
# TestHelperFunctionsWithoutPrometheus - no-op when flag is False
# ============================================================================


class TestHelperFunctionsWithoutPrometheus:
    """Helper functions are no-ops when PROMETHEUS_AVAILABLE is False."""

    def test_record_check_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_check(dataset="ds", result="fresh")

    def test_record_breach_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_breach(severity="critical")

    def test_record_alert_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_alert(channel="email", severity="high")

    def test_record_dataset_registered_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_dataset_registered(status="active")

    def test_record_refresh_event_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_refresh_event(source="erp")

    def test_record_prediction_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_prediction(status="accurate")

    def test_observe_freshness_score_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            observe_freshness_score(dataset="ds", score=0.9)

    def test_observe_data_age_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            observe_data_age(dataset="ds", age_hours=10.0)

    def test_observe_duration_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            observe_duration(operation="check", duration=0.5)

    def test_set_active_breaches_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            set_active_breaches(count=0)

    def test_set_monitored_datasets_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            set_monitored_datasets(count=0)

    def test_record_error_no_op_when_unavailable(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_error(error_type="timeout")


# ============================================================================
# TestMetricsExports - __all__ completeness
# ============================================================================


class TestMetricsExports:
    """Verify metrics module exports."""

    def test_all_list_exists(self):
        assert hasattr(metrics_mod, "__all__")

    def test_all_contains_prometheus_available(self):
        assert "PROMETHEUS_AVAILABLE" in metrics_mod.__all__

    def test_all_minimum_count(self):
        # 1 flag + 12 metrics + 12 helpers + 3 dummy classes = 28
        assert len(metrics_mod.__all__) >= 28

    def test_all_contains_all_metric_objects(self):
        expected_metrics = [
            "dfm_checks_performed_total",
            "dfm_sla_breaches_total",
            "dfm_alerts_sent_total",
            "dfm_datasets_registered_total",
            "dfm_refresh_events_total",
            "dfm_predictions_made_total",
            "dfm_freshness_score",
            "dfm_data_age_hours",
            "dfm_processing_duration_seconds",
            "dfm_active_breaches",
            "dfm_monitored_datasets",
            "dfm_processing_errors_total",
        ]
        for name in expected_metrics:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"

    def test_all_contains_all_helper_functions(self):
        expected_helpers = [
            "record_check",
            "record_breach",
            "record_alert",
            "record_dataset_registered",
            "record_refresh_event",
            "record_prediction",
            "observe_freshness_score",
            "observe_data_age",
            "observe_duration",
            "set_active_breaches",
            "set_monitored_datasets",
            "record_error",
        ]
        for name in expected_helpers:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"

    def test_all_contains_dummy_counter(self):
        assert "DummyCounter" in metrics_mod.__all__

    def test_all_contains_dummy_histogram(self):
        assert "DummyHistogram" in metrics_mod.__all__

    def test_all_contains_dummy_gauge(self):
        assert "DummyGauge" in metrics_mod.__all__

    def test_all_entries_are_resolvable(self):
        for name in metrics_mod.__all__:
            assert hasattr(metrics_mod, name), f"{name} in __all__ but not on module"


# ============================================================================
# TestMetricNaming - verify naming conventions
# ============================================================================


class TestMetricNaming:
    """Verify metric names follow gl_dfm_ naming convention."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_checks_performed_name(self):
        # prometheus_client Counter strips _total suffix from _name
        assert dfm_checks_performed_total._name == "gl_dfm_checks_performed"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sla_breaches_name(self):
        assert dfm_sla_breaches_total._name == "gl_dfm_sla_breaches"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_alerts_sent_name(self):
        assert dfm_alerts_sent_total._name == "gl_dfm_alerts_sent"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_datasets_registered_name(self):
        assert dfm_datasets_registered_total._name == "gl_dfm_datasets_registered"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_refresh_events_name(self):
        assert dfm_refresh_events_total._name == "gl_dfm_refresh_events"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_predictions_made_name(self):
        assert dfm_predictions_made_total._name == "gl_dfm_predictions_made"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_freshness_score_name(self):
        assert dfm_freshness_score._name == "gl_dfm_freshness_score"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_data_age_hours_name(self):
        assert dfm_data_age_hours._name == "gl_dfm_data_age_hours"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_duration_name(self):
        assert dfm_processing_duration_seconds._name == "gl_dfm_processing_duration_seconds"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_breaches_name(self):
        assert dfm_active_breaches._name == "gl_dfm_active_breaches"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_monitored_datasets_name(self):
        assert dfm_monitored_datasets._name == "gl_dfm_monitored_datasets"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_processing_errors_name(self):
        # prometheus_client Counter strips _total suffix from _name
        assert dfm_processing_errors_total._name == "gl_dfm_processing_errors"
