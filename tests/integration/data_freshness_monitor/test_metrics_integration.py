# -*- coding: utf-8 -*-
"""
Metrics integration tests for AGENT-DATA-016 Data Freshness Monitor.

Tests that all 12 Prometheus metric helpers work correctly in an
integration context (using either real prometheus_client or the
graceful Dummy fallbacks).

10+ tests covering:
- All 12 metric helper functions
- Metric recording during freshness checks
- Metric recording during breach detection
- Metric recording during alert generation
- Gauge set operations for active breaches and monitored datasets

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

import pytest

from greenlang.data_freshness_monitor.metrics import (
    PROMETHEUS_AVAILABLE,
    DummyCounter,
    DummyGauge,
    DummyHistogram,
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


# ===================================================================
# Metric Helper Function Tests
# ===================================================================


class TestMetricHelpers:
    """Test all 12 metric helper functions execute without errors."""

    def test_record_check_helper(self):
        """record_check() should not raise for valid parameters."""
        record_check(dataset="test-ds", result="fresh")
        record_check(dataset="test-ds", result="stale")
        record_check(dataset="test-ds", result="warning")
        record_check(dataset="test-ds", result="critical")
        record_check(dataset="test-ds", result="unknown")
        record_check(dataset="test-ds", result="skipped")

    def test_record_breach_helper(self):
        """record_breach() should not raise for valid severity levels."""
        record_breach(severity="critical")
        record_breach(severity="high")
        record_breach(severity="medium")
        record_breach(severity="low")
        record_breach(severity="info")
        record_breach(severity="critical", count=5)

    def test_record_alert_helper(self):
        """record_alert() should not raise for valid channels and severities."""
        record_alert(channel="slack", severity="critical")
        record_alert(channel="pagerduty", severity="high")
        record_alert(channel="email", severity="medium")
        record_alert(channel="opsgenie", severity="low")
        record_alert(channel="teams", severity="info")
        record_alert(channel="webhook", severity="warning")

    def test_record_dataset_registered_helper(self):
        """record_dataset_registered() should not raise for valid statuses."""
        record_dataset_registered(status="active")
        record_dataset_registered(status="inactive")
        record_dataset_registered(status="paused")
        record_dataset_registered(status="deregistered")
        record_dataset_registered(status="pending")
        record_dataset_registered(status="active", count=10)

    def test_record_refresh_event_helper(self):
        """record_refresh_event() should not raise for valid sources."""
        record_refresh_event(source="erp")
        record_refresh_event(source="api")
        record_refresh_event(source="file_upload")
        record_refresh_event(source="scheduled_etl")
        record_refresh_event(source="manual")
        record_refresh_event(source="streaming")
        record_refresh_event(source="webhook")
        record_refresh_event(source="erp", count=3)

    def test_record_prediction_helper(self):
        """record_prediction() should not raise for valid statuses."""
        record_prediction(status="accurate")
        record_prediction(status="inaccurate")
        record_prediction(status="pending")
        record_prediction(status="expired")
        record_prediction(status="failed")

    def test_observe_freshness_score_helper(self):
        """observe_freshness_score() should not raise for valid scores."""
        observe_freshness_score(dataset="test-ds", score=1.0)
        observe_freshness_score(dataset="test-ds", score=0.8)
        observe_freshness_score(dataset="test-ds", score=0.6)
        observe_freshness_score(dataset="test-ds", score=0.4)
        observe_freshness_score(dataset="test-ds", score=0.2)
        observe_freshness_score(dataset="test-ds", score=0.0)

    def test_observe_data_age_helper(self):
        """observe_data_age() should not raise for valid age values."""
        observe_data_age(dataset="test-ds", age_hours=0.1)
        observe_data_age(dataset="test-ds", age_hours=1.0)
        observe_data_age(dataset="test-ds", age_hours=24.0)
        observe_data_age(dataset="test-ds", age_hours=72.0)
        observe_data_age(dataset="test-ds", age_hours=720.0)

    def test_observe_duration_helper(self):
        """observe_duration() should not raise for valid operations."""
        observe_duration(operation="check_freshness", duration=0.001)
        observe_duration(operation="evaluate_sla", duration=0.005)
        observe_duration(operation="detect_breach", duration=0.01)
        observe_duration(operation="send_alert", duration=0.1)
        observe_duration(operation="predict_refresh", duration=0.5)
        observe_duration(operation="register_dataset", duration=0.002)
        observe_duration(operation="record_refresh", duration=0.003)

    def test_set_active_breaches_helper(self):
        """set_active_breaches() should not raise for valid counts."""
        set_active_breaches(count=0)
        set_active_breaches(count=5)
        set_active_breaches(count=100)

    def test_set_monitored_datasets_helper(self):
        """set_monitored_datasets() should not raise for valid counts."""
        set_monitored_datasets(count=0)
        set_monitored_datasets(count=50)
        set_monitored_datasets(count=10000)

    def test_record_error_helper(self):
        """record_error() should not raise for valid error types."""
        record_error(error_type="validation")
        record_error(error_type="timeout")
        record_error(error_type="data")
        record_error(error_type="integration")
        record_error(error_type="sla_evaluation")
        record_error(error_type="prediction")
        record_error(error_type="alerting")
        record_error(error_type="registration")
        record_error(error_type="refresh_tracking")
        record_error(error_type="unknown")


# ===================================================================
# Dummy Fallback Tests
# ===================================================================


class TestDummyFallbacks:
    """Test that the Dummy metric classes work as no-op replacements."""

    def test_dummy_counter_labels_inc(self):
        """DummyCounter.labels().inc() should silently no-op."""
        counter = DummyCounter()
        counter.labels(dataset="test", result="fresh").inc()
        counter.labels(dataset="test", result="fresh").inc(5)
        counter.inc()
        counter.inc(10)

    def test_dummy_histogram_labels_observe(self):
        """DummyHistogram.labels().observe() should silently no-op."""
        histogram = DummyHistogram()
        histogram.labels(dataset="test").observe(1.5)
        histogram.observe(0.5)

    def test_dummy_gauge_labels_set_inc_dec(self):
        """DummyGauge.labels().set/inc/dec() should silently no-op."""
        gauge = DummyGauge()
        gauge.labels(name="test").set(42.0)
        gauge.labels(name="test").inc(1)
        gauge.set(100)
        gauge.inc(5)
        gauge.dec(3)


# ===================================================================
# Metric Objects Are Initialized
# ===================================================================


class TestMetricObjectsInitialized:
    """Verify all 12 metric objects are initialized (either real or dummy)."""

    def test_checks_performed_total_exists(self):
        """dfm_checks_performed_total is not None."""
        assert dfm_checks_performed_total is not None

    def test_sla_breaches_total_exists(self):
        """dfm_sla_breaches_total is not None."""
        assert dfm_sla_breaches_total is not None

    def test_alerts_sent_total_exists(self):
        """dfm_alerts_sent_total is not None."""
        assert dfm_alerts_sent_total is not None

    def test_datasets_registered_total_exists(self):
        """dfm_datasets_registered_total is not None."""
        assert dfm_datasets_registered_total is not None

    def test_refresh_events_total_exists(self):
        """dfm_refresh_events_total is not None."""
        assert dfm_refresh_events_total is not None

    def test_predictions_made_total_exists(self):
        """dfm_predictions_made_total is not None."""
        assert dfm_predictions_made_total is not None

    def test_freshness_score_exists(self):
        """dfm_freshness_score is not None."""
        assert dfm_freshness_score is not None

    def test_data_age_hours_exists(self):
        """dfm_data_age_hours is not None."""
        assert dfm_data_age_hours is not None

    def test_processing_duration_seconds_exists(self):
        """dfm_processing_duration_seconds is not None."""
        assert dfm_processing_duration_seconds is not None

    def test_active_breaches_exists(self):
        """dfm_active_breaches is not None."""
        assert dfm_active_breaches is not None

    def test_monitored_datasets_exists(self):
        """dfm_monitored_datasets is not None."""
        assert dfm_monitored_datasets is not None

    def test_processing_errors_total_exists(self):
        """dfm_processing_errors_total is not None."""
        assert dfm_processing_errors_total is not None


# ===================================================================
# Metrics During Service Operations
# ===================================================================


class TestMetricsDuringServiceOperations:
    """Test that metric helpers can be called during actual service operations
    without interfering with service behavior."""

    def test_metrics_during_registration(self, service):
        """Call metric helpers during dataset registration and verify
        the registration still succeeds."""
        record_dataset_registered(status="active")
        set_monitored_datasets(count=1)

        ds = service.register_dataset(
            name="Metrics Registration DS",
            source="test",
        )
        assert ds["dataset_id"]

        record_dataset_registered(status="active")
        set_monitored_datasets(count=2)

    def test_metrics_during_freshness_check(self, service):
        """Call metric helpers during a freshness check and verify
        the check still produces correct results."""
        from datetime import timedelta

        ds = service.register_dataset(name="Metrics Check DS", source="test")
        dataset_id = ds["dataset_id"]

        fresh_ts = (_utcnow() - timedelta(hours=2)).isoformat()

        observe_duration(operation="check_freshness", duration=0.001)
        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=fresh_ts,
        )
        observe_freshness_score(dataset=ds["name"], score=check["freshness_score"])
        observe_data_age(dataset=ds["name"], age_hours=check["age_hours"])
        record_check(dataset=ds["name"], result=check["freshness_level"])
        observe_duration(operation="check_freshness", duration=0.002)

        assert check["freshness_level"] == "good"

    def test_metrics_during_breach_detection(self, service):
        """Call metric helpers when a breach is detected and verify
        both the breach and the metrics are handled correctly."""
        from datetime import timedelta

        ds = service.register_dataset(name="Metrics Breach DS", source="test")
        dataset_id = ds["dataset_id"]

        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()
        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=stale_ts,
        )

        assert check["sla_breach"] is not None
        record_breach(severity=check["sla_breach"]["severity"])
        set_active_breaches(count=1)
        record_check(dataset=ds["name"], result="critical")

    def test_metrics_error_recording(self, service):
        """Call record_error during a failed operation and verify
        the error metric helper does not raise."""
        try:
            service.run_check(dataset_id="nonexistent-id")
        except ValueError:
            record_error(error_type="validation")

        # Should not raise
        record_error(error_type="unknown")


def _utcnow():
    """Return current UTC datetime with microseconds zeroed."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0)
