# -*- coding: utf-8 -*-
"""
Unit tests for SLO Service Metrics (OBS-005)

Tests Prometheus metric recording helpers and graceful degradation
when prometheus_client is unavailable.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.slo_service import metrics as metrics_mod
from greenlang.infrastructure.slo_service.metrics import (
    record_alert_fired,
    record_budget_remaining,
    record_budget_snapshot,
    record_burn_rate,
    record_evaluation,
    record_recording_rules_generated,
    record_report_generated,
    update_compliance_percent,
    update_definitions_count,
)


class TestSLOMetrics:
    """Test suite for SLO metrics helper functions."""

    def test_evaluation_counter_increments(self):
        """Evaluation counter incremented with correct labels."""
        mock_counter = MagicMock()
        mock_histogram = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_evaluations_total", mock_counter), \
             patch.object(metrics_mod, "gl_slo_evaluation_duration_seconds", mock_histogram):
            record_evaluation("api-gateway", "availability", "pass", 0.5)

        mock_counter.labels.assert_called_once_with(
            service="api-gateway", sli_type="availability", result="pass",
        )
        mock_counter.labels().inc.assert_called_once()

    def test_evaluation_duration_histogram(self):
        """Evaluation duration histogram observed."""
        mock_counter = MagicMock()
        mock_histogram = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_evaluations_total", mock_counter), \
             patch.object(metrics_mod, "gl_slo_evaluation_duration_seconds", mock_histogram):
            record_evaluation("api-gateway", "availability", "pass", 1.23)

        mock_histogram.labels.assert_called_once_with(service="api-gateway")
        mock_histogram.labels().observe.assert_called_once_with(1.23)

    def test_budget_remaining_gauge(self):
        """Budget remaining gauge set with correct labels."""
        mock_gauge = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_error_budget_remaining_percent", mock_gauge):
            record_budget_remaining("slo-1", "api-gateway", 75.5)

        mock_gauge.labels.assert_called_once_with(
            slo_id="slo-1", service="api-gateway",
        )
        mock_gauge.labels().set.assert_called_once_with(75.5)

    def test_burn_rate_gauge(self):
        """Burn rate gauge set with correct labels."""
        mock_gauge = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_burn_rate", mock_gauge):
            record_burn_rate("slo-1", "fast", 14.4)

        mock_gauge.labels.assert_called_once_with(
            slo_id="slo-1", window="fast",
        )
        mock_gauge.labels().set.assert_called_once_with(14.4)

    def test_definitions_total_gauge(self):
        """Definitions total gauge set correctly."""
        mock_gauge = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_definitions_total", mock_gauge):
            update_definitions_count("api-gateway", 5)

        mock_gauge.labels.assert_called_once_with(service="api-gateway")
        mock_gauge.labels().set.assert_called_once_with(5)

    def test_compliance_percent_gauge(self):
        """Compliance percentage gauge set correctly."""
        mock_gauge = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_compliance_percent", mock_gauge):
            update_compliance_percent(95.0)

        mock_gauge.set.assert_called_once_with(95.0)

    def test_alerts_fired_counter(self):
        """Alerts fired counter incremented with correct labels."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_alerts_fired_total", mock_counter):
            record_alert_fired("slo-1", "critical", "burn_rate")

        mock_counter.labels.assert_called_once_with(
            slo_id="slo-1", severity="critical", alert_type="burn_rate",
        )
        mock_counter.labels().inc.assert_called_once()

    def test_recording_rules_generated_counter(self):
        """Recording rules generated counter incremented."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_recording_rules_generated_total", mock_counter):
            record_recording_rules_generated()

        mock_counter.inc.assert_called_once()

    def test_budget_snapshot_counter(self):
        """Budget snapshot counter incremented with correct labels."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_budget_snapshots_total", mock_counter):
            record_budget_snapshot("slo-42")

        mock_counter.labels.assert_called_once_with(slo_id="slo-42")
        mock_counter.labels().inc.assert_called_once()

    def test_report_generated_counter(self):
        """Report generated counter incremented with correct labels."""
        mock_counter = MagicMock()

        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", True), \
             patch.object(metrics_mod, "gl_slo_reports_generated_total", mock_counter):
            record_report_generated("monthly")

        mock_counter.labels.assert_called_once_with(report_type="monthly")
        mock_counter.labels().inc.assert_called_once()

    def test_prometheus_not_available_graceful(self):
        """All metrics degrade to no-ops when prometheus_client unavailable."""
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            record_evaluation("svc", "availability", "pass", 0.1)
            record_budget_remaining("slo-1", "svc", 75.0)
            record_burn_rate("slo-1", "fast", 10.0)
            update_definitions_count("svc", 5)
            update_compliance_percent(95.0)
            record_alert_fired("slo-1", "critical", "burn_rate")
            record_recording_rules_generated()
            record_budget_snapshot("slo-1")
            record_report_generated("weekly")
