# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-035

Tests each of the metric helper functions and graceful degradation
when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.agents.eudr.improvement_plan_creator import metrics


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_finding_aggregated_no_error(self):
        """Calling record_finding_aggregated should not raise."""
        metrics.record_finding_aggregated("audit")

    def test_record_gap_identified_no_error(self):
        metrics.record_gap_identified("critical")

    def test_record_action_generated_no_error(self):
        metrics.record_action_generated("corrective")

    def test_record_action_completed_no_error(self):
        metrics.record_action_completed()

    def test_record_action_verified_no_error(self):
        metrics.record_action_verified()

    def test_record_root_cause_mapped_no_error(self):
        metrics.record_root_cause_mapped("process")

    def test_record_action_prioritized_no_error(self):
        metrics.record_action_prioritized("do_first")

    def test_record_progress_snapshot_no_error(self):
        metrics.record_progress_snapshot()

    def test_record_stakeholder_assigned_no_error(self):
        metrics.record_stakeholder_assigned("responsible")

    def test_record_notification_sent_no_error(self):
        metrics.record_notification_sent("email")

    def test_record_plan_created_no_error(self):
        metrics.record_plan_created("draft")

    def test_record_plan_approved_no_error(self):
        metrics.record_plan_approved()

    def test_record_report_generated_no_error(self):
        metrics.record_report_generated("json")

    def test_record_escalation_triggered_no_error(self):
        metrics.record_escalation_triggered("manager")

    def test_record_duplicates_removed_no_error(self):
        metrics.record_duplicates_removed(3)

    def test_record_smart_validation_no_error(self):
        metrics.record_smart_validation("pass")


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_finding_aggregation_duration_no_error(self):
        metrics.observe_finding_aggregation_duration(0.5)

    def test_observe_gap_analysis_duration_no_error(self):
        metrics.observe_gap_analysis_duration(1.2)

    def test_observe_action_generation_duration_no_error(self):
        metrics.observe_action_generation_duration(0.8)

    def test_observe_root_cause_mapping_duration_no_error(self):
        metrics.observe_root_cause_mapping_duration(2.0)

    def test_observe_prioritization_duration_no_error(self):
        metrics.observe_prioritization_duration(0.3)

    def test_observe_progress_tracking_duration_no_error(self):
        metrics.observe_progress_tracking_duration(0.15)

    def test_observe_stakeholder_coord_duration_no_error(self):
        metrics.observe_stakeholder_coord_duration(0.6)

    def test_observe_plan_creation_duration_no_error(self):
        metrics.observe_plan_creation_duration(3.5)

    def test_observe_notification_dispatch_duration_no_error(self):
        metrics.observe_notification_dispatch_duration(0.1)

    def test_observe_fishbone_analysis_duration_no_error(self):
        metrics.observe_fishbone_analysis_duration(1.5)

    def test_observe_five_whys_duration_no_error(self):
        metrics.observe_five_whys_duration(1.2)

    def test_observe_report_generation_duration_no_error(self):
        metrics.observe_report_generation_duration(2.0)

    def test_observe_effectiveness_review_duration_no_error(self):
        metrics.observe_effectiveness_review_duration(1.8)

    def test_observe_raci_validation_duration_no_error(self):
        metrics.observe_raci_validation_duration(0.5)


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_plans_no_error(self):
        metrics.set_active_plans(3)

    def test_set_pending_actions_no_error(self):
        metrics.set_pending_actions(8)

    def test_set_overdue_actions_no_error(self):
        metrics.set_overdue_actions(2)

    def test_set_critical_gaps_open_no_error(self):
        metrics.set_critical_gaps_open(2)

    def test_set_high_gaps_open_no_error(self):
        metrics.set_high_gaps_open(5)

    def test_set_overall_progress_no_error(self):
        metrics.set_overall_progress(65.5)

    def test_set_avg_effectiveness_no_error(self):
        metrics.set_avg_effectiveness(75.0)

    def test_set_stakeholders_pending_ack_no_error(self):
        metrics.set_stakeholders_pending_ack(3)

    def test_set_on_track_plans_no_error(self):
        metrics.set_on_track_plans(2)

    def test_set_off_track_plans_no_error(self):
        metrics.set_off_track_plans(1)

    def test_set_actions_on_hold_no_error(self):
        metrics.set_actions_on_hold(1)

    def test_set_systemic_root_causes_no_error(self):
        metrics.set_systemic_root_causes(4)


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counter_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.record_finding_aggregated("audit")
            metrics.record_gap_identified("critical")
            metrics.record_action_generated("corrective")
            metrics.record_action_completed()
            metrics.record_action_verified()
            metrics.record_root_cause_mapped("people")
            metrics.record_action_prioritized("do_first")
            metrics.record_progress_snapshot()
            metrics.record_stakeholder_assigned("responsible")
            metrics.record_notification_sent("email")
            metrics.record_plan_created("active")
            metrics.record_plan_approved()
            metrics.record_report_generated("json")
            metrics.record_escalation_triggered("manager")
            metrics.record_duplicates_removed(2)
            metrics.record_smart_validation("pass")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histogram_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_finding_aggregation_duration(0.5)
            metrics.observe_gap_analysis_duration(1.2)
            metrics.observe_action_generation_duration(0.8)
            metrics.observe_root_cause_mapping_duration(2.0)
            metrics.observe_prioritization_duration(0.3)
            metrics.observe_progress_tracking_duration(0.15)
            metrics.observe_stakeholder_coord_duration(0.6)
            metrics.observe_plan_creation_duration(3.5)
            metrics.observe_notification_dispatch_duration(0.1)
            metrics.observe_fishbone_analysis_duration(1.5)
            metrics.observe_five_whys_duration(1.2)
            metrics.observe_report_generation_duration(2.0)
            metrics.observe_effectiveness_review_duration(1.8)
            metrics.observe_raci_validation_duration(0.5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauge_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_plans(2)
            metrics.set_pending_actions(6)
            metrics.set_overdue_actions(1)
            metrics.set_critical_gaps_open(1)
            metrics.set_high_gaps_open(4)
            metrics.set_overall_progress(72.0)
            metrics.set_avg_effectiveness(75.0)
            metrics.set_stakeholders_pending_ack(3)
            metrics.set_on_track_plans(2)
            metrics.set_off_track_plans(0)
            metrics.set_actions_on_hold(0)
            metrics.set_systemic_root_causes(5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original
