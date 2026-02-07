# -*- coding: utf-8 -*-
"""
Unit tests for Alerting Metrics (OBS-004)

Tests Prometheus metric recording helpers including notification counters,
MTTA/MTTR histograms, escalation counters, dedup counters, active alert
gauges, fatigue scores, and graceful no-op when prometheus_client is
unavailable.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service import metrics as metrics_mod
from greenlang.infrastructure.alerting_service.metrics import (
    record_notification,
    record_mtta,
    record_mttr,
    record_escalation,
    record_dedup,
    update_active_alerts,
    record_oncall_lookup,
    update_fatigue_score,
)


# ============================================================================
# Tests
# ============================================================================


class TestAlertingMetrics:
    """Test suite for alerting metrics helper functions."""

    def test_record_notification(self):
        """Counter incremented with correct labels."""
        mock_counter = MagicMock()
        mock_histogram = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_notifications_total", mock_counter), \
             patch.object(metrics_mod, "gl_alert_notification_duration_seconds", mock_histogram):
            record_notification("slack", "critical", "sent", 0.15)

        mock_counter.labels.assert_called_once_with(
            channel="slack", severity="critical", status="sent",
        )
        mock_counter.labels().inc.assert_called_once()
        mock_histogram.labels.assert_called_once_with(channel="slack")
        mock_histogram.labels().observe.assert_called_once_with(0.15)

    def test_record_mtta(self):
        """MTTA histogram observed with correct labels."""
        mock_histogram = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_mtta_seconds", mock_histogram):
            record_mtta("platform", "critical", 300.0)

        mock_histogram.labels.assert_called_once_with(
            team="platform", severity="critical",
        )
        mock_histogram.labels().observe.assert_called_once_with(300.0)

    def test_record_mttr(self):
        """MTTR histogram observed with correct labels."""
        mock_histogram = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_mttr_seconds", mock_histogram):
            record_mttr("data-platform", "warning", 7200.0)

        mock_histogram.labels.assert_called_once_with(
            team="data-platform", severity="warning",
        )
        mock_histogram.labels().observe.assert_called_once_with(7200.0)

    def test_record_escalation(self):
        """Escalation counter incremented with correct labels."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_escalations_total", mock_counter):
            record_escalation("1", "critical_default")

        mock_counter.labels.assert_called_once_with(
            level="1", policy="critical_default",
        )
        mock_counter.labels().inc.assert_called_once()

    def test_record_dedup(self):
        """Dedup counter incremented."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_dedup_total", mock_counter):
            record_dedup()

        mock_counter.inc.assert_called_once()

    def test_update_active_alerts(self):
        """Active alerts gauge set with correct labels."""
        mock_gauge = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_active_total", mock_gauge):
            update_active_alerts("critical", "firing", 5)

        mock_gauge.labels.assert_called_once_with(
            severity="critical", status="firing",
        )
        mock_gauge.labels().set.assert_called_once_with(5)

    def test_record_oncall_lookup(self):
        """On-call lookup counter incremented."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_alert_oncall_lookups_total", mock_counter):
            record_oncall_lookup("pagerduty", "success")

        mock_counter.labels.assert_called_once_with(
            provider="pagerduty", status="success",
        )
        mock_counter.labels().inc.assert_called_once()

    def test_metrics_no_prometheus(self):
        """Graceful no-op when prometheus_client is unavailable."""
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            # None of these should raise
            record_notification("slack", "critical", "sent", 0.1)
            record_mtta("team", "critical", 300.0)
            record_mttr("team", "warning", 7200.0)
            record_escalation("1", "policy")
            record_dedup()
            update_active_alerts("critical", "firing", 5)
            update_fatigue_score("team", 10.0)
            record_oncall_lookup("pagerduty", "success")
