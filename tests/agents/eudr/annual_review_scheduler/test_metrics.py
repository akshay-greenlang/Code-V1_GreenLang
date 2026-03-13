# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-034

Tests each of the metric helper functions and graceful degradation
when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.agents.eudr.annual_review_scheduler import metrics


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_cycle_created_no_error(self):
        """Calling record_cycle_created should not raise."""
        metrics.record_cycle_created("coffee", "annual")

    def test_record_cycle_completed_no_error(self):
        metrics.record_cycle_completed("cocoa", "success")

    def test_record_phase_advanced_no_error(self):
        metrics.record_phase_advanced("cyc-001", "data_collection")

    def test_record_deadline_created_no_error(self):
        metrics.record_deadline_created("data_collection")

    def test_record_deadline_met_no_error(self):
        metrics.record_deadline_met("analysis")

    def test_record_deadline_missed_no_error(self):
        metrics.record_deadline_missed("remediation")

    def test_record_checklist_item_completed_no_error(self):
        metrics.record_checklist_item_completed("data_collection")

    def test_record_notification_sent_no_error(self):
        metrics.record_notification_sent("email", "normal")

    def test_record_notification_failed_no_error(self):
        metrics.record_notification_failed("sms", "gateway_error")

    def test_record_year_comparison_completed_no_error(self):
        metrics.record_year_comparison_completed("coffee")

    def test_record_entity_assigned_no_error(self):
        metrics.record_entity_assigned("reviewer")

    def test_record_api_error_no_error(self):
        metrics.record_api_error("create_cycle")


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_cycle_creation_duration_no_error(self):
        metrics.observe_cycle_creation_duration("coffee", 0.5)

    def test_observe_phase_advancement_duration_no_error(self):
        metrics.observe_phase_advancement_duration("data_collection", 0.1)

    def test_observe_checklist_generation_duration_no_error(self):
        metrics.observe_checklist_generation_duration("coffee", 0.25)

    def test_observe_year_comparison_duration_no_error(self):
        metrics.observe_year_comparison_duration("cocoa", 1.5)

    def test_observe_notification_delivery_duration_no_error(self):
        metrics.observe_notification_delivery_duration("email", 0.3)

    def test_observe_calendar_sync_duration_no_error(self):
        metrics.observe_calendar_sync_duration(0.8)


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_review_cycles_no_error(self):
        metrics.set_active_review_cycles(5)

    def test_set_overdue_deadlines_no_error(self):
        metrics.set_overdue_deadlines(2)

    def test_set_pending_checklist_items_no_error(self):
        metrics.set_pending_checklist_items(15)

    def test_set_pending_notifications_no_error(self):
        metrics.set_pending_notifications(8)

    def test_set_active_entities_no_error(self):
        metrics.set_active_entities(12)

    def test_set_compliance_rate_gauge_no_error(self):
        metrics.set_compliance_rate_gauge(95.5)


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counter_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.record_cycle_created("coffee", "annual")
            metrics.record_cycle_completed("cocoa", "success")
            metrics.record_phase_advanced("cyc-001", "analysis")
            metrics.record_deadline_created("preparation")
            metrics.record_deadline_met("data_collection")
            metrics.record_deadline_missed("remediation")
            metrics.record_checklist_item_completed("sign_off")
            metrics.record_notification_sent("email", "high")
            metrics.record_notification_failed("webhook", "timeout")
            metrics.record_year_comparison_completed("wood")
            metrics.record_entity_assigned("analyst")
            metrics.record_api_error("create_cycle")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histogram_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_cycle_creation_duration("coffee", 0.5)
            metrics.observe_phase_advancement_duration("analysis", 0.1)
            metrics.observe_checklist_generation_duration("cocoa", 0.3)
            metrics.observe_year_comparison_duration("wood", 1.0)
            metrics.observe_notification_delivery_duration("slack", 0.2)
            metrics.observe_calendar_sync_duration(0.5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauge_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_review_cycles(10)
            metrics.set_overdue_deadlines(0)
            metrics.set_pending_checklist_items(20)
            metrics.set_pending_notifications(5)
            metrics.set_active_entities(8)
            metrics.set_compliance_rate_gauge(97.0)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original
