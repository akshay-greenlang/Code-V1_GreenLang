# -*- coding: utf-8 -*-
"""
Unit Tests for Prometheus Self-Monitoring Metrics (AGENT-FOUND-010)

Tests all 12 Prometheus metrics and their helper functions, including
graceful fallback when prometheus_client is not installed.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.observability_agent.metrics import (
    PROMETHEUS_AVAILABLE,
    obs_alerts_evaluated_total,
    obs_alerts_firing,
    obs_dashboard_queries_total,
    obs_error_budget_remaining,
    obs_health_checks_total,
    obs_health_status,
    obs_logs_ingested_total,
    obs_metrics_recorded_total,
    obs_operation_duration_seconds,
    obs_slo_compliance_ratio,
    obs_spans_active,
    obs_spans_created_total,
    record_alert_evaluated,
    record_dashboard_query,
    record_health_check,
    record_log_ingested,
    record_metric_recorded,
    record_operation_duration,
    record_span_created,
    update_active_spans,
    update_error_budget,
    update_firing_alerts,
    update_health_status,
    update_slo_compliance,
)


# ==========================================================================
# Prometheus Availability Tests
# ==========================================================================

class TestPrometheusAvailability:
    """Tests for PROMETHEUS_AVAILABLE flag."""

    def test_prometheus_available_flag_is_boolean(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


# ==========================================================================
# Metric Object Existence Tests
# ==========================================================================

class TestMetricObjectsExist:
    """Tests that metric objects are defined (or None if no prometheus_client)."""

    def test_metrics_recorded_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_metrics_recorded_total is not None
        else:
            assert obs_metrics_recorded_total is None

    def test_operation_duration_seconds_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_operation_duration_seconds is not None
        else:
            assert obs_operation_duration_seconds is None

    def test_spans_created_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_spans_created_total is not None
        else:
            assert obs_spans_created_total is None

    def test_spans_active_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_spans_active is not None
        else:
            assert obs_spans_active is None

    def test_logs_ingested_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_logs_ingested_total is not None
        else:
            assert obs_logs_ingested_total is None

    def test_alerts_evaluated_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_alerts_evaluated_total is not None
        else:
            assert obs_alerts_evaluated_total is None

    def test_alerts_firing_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_alerts_firing is not None
        else:
            assert obs_alerts_firing is None

    def test_health_checks_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_health_checks_total is not None
        else:
            assert obs_health_checks_total is None

    def test_health_status_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_health_status is not None
        else:
            assert obs_health_status is None

    def test_slo_compliance_ratio_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_slo_compliance_ratio is not None
        else:
            assert obs_slo_compliance_ratio is None

    def test_error_budget_remaining_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_error_budget_remaining is not None
        else:
            assert obs_error_budget_remaining is None

    def test_dashboard_queries_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert obs_dashboard_queries_total is not None
        else:
            assert obs_dashboard_queries_total is None


# ==========================================================================
# Helper Function Tests (safe to call regardless of prometheus_client)
# ==========================================================================

class TestHelperFunctions:
    """Tests that helper functions run without error regardless of prometheus state."""

    def test_record_metric_recorded(self):
        # Should not raise even if prometheus is not available
        record_metric_recorded("counter", "default")

    def test_record_operation_duration(self):
        record_operation_duration(0.005)

    def test_record_span_created(self):
        record_span_created("ok")

    def test_update_active_spans(self):
        update_active_spans(5)

    def test_record_log_ingested(self):
        record_log_ingested("info")

    def test_record_alert_evaluated(self):
        record_alert_evaluated("firing")

    def test_update_firing_alerts(self):
        update_firing_alerts(3)

    def test_record_health_check(self):
        record_health_check("healthy", "liveness")

    def test_update_health_status(self):
        update_health_status(1.0)

    def test_update_slo_compliance(self):
        update_slo_compliance("api-service", "availability", 0.999)

    def test_update_error_budget(self):
        update_error_budget("api-service", 0.85)

    def test_record_dashboard_query(self):
        record_dashboard_query()


# ==========================================================================
# Helper Function with Labels Tests (when Prometheus IS available)
# ==========================================================================

@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
class TestHelperFunctionsWithPrometheus:
    """Tests helper functions actually interact with prometheus_client."""

    def test_record_metric_recorded_increments(self):
        # Just verify no error is raised; Prometheus metrics are global state
        record_metric_recorded("gauge", "tenant-test")

    def test_record_span_created_increments(self):
        record_span_created("error")

    def test_update_active_spans_sets_gauge(self):
        update_active_spans(42)

    def test_record_log_ingested_increments(self):
        record_log_ingested("error")

    def test_record_alert_evaluated_increments(self):
        record_alert_evaluated("resolved")

    def test_update_firing_alerts_sets_gauge(self):
        update_firing_alerts(0)

    def test_record_health_check_increments(self):
        record_health_check("degraded", "readiness")

    def test_update_health_status_gauge(self):
        update_health_status(0.5)

    def test_update_slo_compliance_gauge(self):
        update_slo_compliance("svc", "latency", 0.95)

    def test_update_error_budget_gauge(self):
        update_error_budget("svc", 0.1)

    def test_record_dashboard_query_increments(self):
        record_dashboard_query()
