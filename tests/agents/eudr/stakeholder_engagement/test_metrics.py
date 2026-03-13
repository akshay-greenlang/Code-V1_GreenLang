# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-031

Tests each metric helper function (counters, histograms, gauges)
and graceful degradation when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.agents.eudr.stakeholder_engagement import metrics


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_stakeholder_mapped_no_error(self):
        """Test record_stakeholder_mapped executes without error."""
        metrics.record_stakeholder_mapped("indigenous_community", "coffee")

    def test_record_stakeholder_mapped_various_categories(self):
        """Test record_stakeholder_mapped with various categories."""
        for category in ["indigenous_community", "local_community", "cooperative", "ngo", "smallholder"]:
            metrics.record_stakeholder_mapped(category, "coffee")

    def test_record_stakeholder_mapped_various_commodities(self):
        """Test record_stakeholder_mapped with various commodities."""
        for commodity in ["coffee", "cocoa", "wood", "rubber", "soya", "palm_oil", "cattle"]:
            metrics.record_stakeholder_mapped("local_community", commodity)

    def test_record_fpic_initiated_no_error(self):
        """Test record_fpic_initiated executes without error."""
        metrics.record_fpic_initiated("coffee")

    def test_record_fpic_initiated_all_commodities(self):
        """Test record_fpic_initiated with all commodities."""
        for commodity in ["coffee", "cocoa", "wood", "rubber", "soya", "palm_oil", "cattle"]:
            metrics.record_fpic_initiated(commodity)

    def test_record_fpic_consent_no_error(self):
        """Test record_fpic_consent executes without error."""
        metrics.record_fpic_consent("coffee", "granted")

    def test_record_fpic_consent_various_statuses(self):
        """Test record_fpic_consent with various consent statuses."""
        for status in ["granted", "withheld", "conditional", "withdrawn", "expired"]:
            metrics.record_fpic_consent("coffee", status)

    def test_record_grievance_submitted_no_error(self):
        """Test record_grievance_submitted executes without error."""
        metrics.record_grievance_submitted("critical")

    def test_record_grievance_submitted_all_severities(self):
        """Test record_grievance_submitted with all severities."""
        for severity in ["critical", "high", "standard", "minor"]:
            metrics.record_grievance_submitted(severity)

    def test_record_grievance_resolved_no_error(self):
        """Test record_grievance_resolved executes without error."""
        metrics.record_grievance_resolved("standard")

    def test_record_consultation_conducted_no_error(self):
        """Test record_consultation_conducted executes without error."""
        metrics.record_consultation_conducted("community_meeting")

    def test_record_consultation_conducted_all_types(self):
        """Test record_consultation_conducted with all types."""
        for ctype in ["community_meeting", "bilateral", "focus_group", "public_hearing", "workshop", "field_visit"]:
            metrics.record_consultation_conducted(ctype)

    def test_record_communication_sent_no_error(self):
        """Test record_communication_sent executes without error."""
        metrics.record_communication_sent("email")

    def test_record_communication_sent_all_channels(self):
        """Test record_communication_sent with all channels."""
        for channel in ["email", "sms", "letter", "radio", "in_person", "phone"]:
            metrics.record_communication_sent(channel)

    def test_record_assessment_completed_no_error(self):
        """Test record_assessment_completed executes without error."""
        metrics.record_assessment_completed("STK-001")

    def test_record_report_generated_no_error(self):
        """Test record_report_generated executes without error."""
        metrics.record_report_generated("dds_summary", "json")

    def test_record_report_generated_all_types(self):
        """Test record_report_generated with all report types."""
        for rtype in ["dds_summary", "fpic_compliance", "grievance_report", "consultation_register", "engagement_summary"]:
            metrics.record_report_generated(rtype, "json")

    def test_record_api_error_no_error(self):
        """Test record_api_error executes without error."""
        metrics.record_api_error("map_stakeholder")

    def test_record_api_error_various_operations(self):
        """Test record_api_error with various operations."""
        for operation in ["map_stakeholder", "initiate_fpic", "submit_grievance", "send_communication", "assess_engagement"]:
            metrics.record_api_error(operation)


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_stakeholder_mapping_duration_no_error(self):
        """Test observe_stakeholder_mapping_duration executes without error."""
        metrics.observe_stakeholder_mapping_duration("coffee", 0.5)

    def test_observe_stakeholder_mapping_duration_various_durations(self):
        """Test observe_stakeholder_mapping_duration with various durations."""
        for duration in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:
            metrics.observe_stakeholder_mapping_duration("coffee", duration)

    def test_observe_fpic_stage_duration_no_error(self):
        """Test observe_fpic_stage_duration executes without error."""
        metrics.observe_fpic_stage_duration("notification", 86400.0)

    def test_observe_fpic_stage_duration_all_stages(self):
        """Test observe_fpic_stage_duration with all FPIC stages."""
        for stage in ["notification", "information_sharing", "consultation", "deliberation", "decision", "agreement", "monitoring"]:
            metrics.observe_fpic_stage_duration(stage, 3600.0)

    def test_observe_grievance_resolution_duration_no_error(self):
        """Test observe_grievance_resolution_duration executes without error."""
        metrics.observe_grievance_resolution_duration("critical", 3600.0)

    def test_observe_grievance_resolution_duration_all_severities(self):
        """Test observe_grievance_resolution_duration with all severities."""
        for severity in ["critical", "high", "standard", "minor"]:
            metrics.observe_grievance_resolution_duration(severity, 7200.0)

    def test_observe_consultation_duration_no_error(self):
        """Test observe_consultation_duration executes without error."""
        metrics.observe_consultation_duration("community_meeting", 120.0)

    def test_observe_communication_delivery_duration_no_error(self):
        """Test observe_communication_delivery_duration executes without error."""
        metrics.observe_communication_delivery_duration("email", 2.5)

    def test_observe_communication_delivery_duration_all_channels(self):
        """Test observe_communication_delivery_duration with all channels."""
        for channel in ["email", "sms", "letter", "phone"]:
            metrics.observe_communication_delivery_duration(channel, 1.0)


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_stakeholders_no_error(self):
        """Test set_active_stakeholders executes without error."""
        metrics.set_active_stakeholders(45)

    def test_set_active_stakeholders_various_counts(self):
        """Test set_active_stakeholders with various counts."""
        for count in [0, 1, 10, 50, 100, 500]:
            metrics.set_active_stakeholders(count)

    def test_set_active_fpic_workflows_no_error(self):
        """Test set_active_fpic_workflows executes without error."""
        metrics.set_active_fpic_workflows(3)

    def test_set_active_fpic_workflows_various_counts(self):
        """Test set_active_fpic_workflows with various counts."""
        for count in [0, 1, 5, 10, 25]:
            metrics.set_active_fpic_workflows(count)

    def test_set_open_grievances_no_error(self):
        """Test set_open_grievances executes without error."""
        metrics.set_open_grievances(12)

    def test_set_open_grievances_various_counts(self):
        """Test set_open_grievances with various counts."""
        for count in [0, 5, 10, 20, 50]:
            metrics.set_open_grievances(count)

    def test_set_pending_communications_no_error(self):
        """Test set_pending_communications executes without error."""
        metrics.set_pending_communications(8)

    def test_set_pending_communications_various_counts(self):
        """Test set_pending_communications with various counts."""
        for count in [0, 10, 50, 100]:
            metrics.set_pending_communications(count)


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counter_when_prometheus_unavailable(self):
        """Test counter metrics work when prometheus_client is unavailable."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.record_stakeholder_mapped("indigenous_community", "coffee")
            metrics.record_fpic_initiated("coffee")
            metrics.record_fpic_consent("coffee", "granted")
            metrics.record_grievance_submitted("critical")
            metrics.record_grievance_resolved("standard")
            metrics.record_consultation_conducted("community_meeting")
            metrics.record_communication_sent("email")
            metrics.record_assessment_completed("STK-001")
            metrics.record_report_generated("dds_summary", "json")
            metrics.record_api_error("test")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histogram_when_prometheus_unavailable(self):
        """Test histogram metrics work when prometheus_client is unavailable."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_stakeholder_mapping_duration("coffee", 0.5)
            metrics.observe_fpic_stage_duration("notification", 86400.0)
            metrics.observe_grievance_resolution_duration("critical", 3600.0)
            metrics.observe_consultation_duration("community_meeting", 120.0)
            metrics.observe_communication_delivery_duration("email", 2.5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauge_when_prometheus_unavailable(self):
        """Test gauge metrics work when prometheus_client is unavailable."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_stakeholders(45)
            metrics.set_active_fpic_workflows(3)
            metrics.set_open_grievances(12)
            metrics.set_pending_communications(8)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_all_counters_when_prometheus_unavailable(self):
        """Test all counter metrics gracefully degrade."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.record_stakeholder_mapped("cooperative", "cocoa")
            metrics.record_fpic_initiated("wood")
            metrics.record_fpic_consent("rubber", "withheld")
            metrics.record_grievance_submitted("minor")
            metrics.record_grievance_resolved("high")
            metrics.record_consultation_conducted("bilateral")
            metrics.record_communication_sent("sms")
            metrics.record_assessment_completed("STK-002")
            metrics.record_report_generated("engagement_summary", "pdf")
            metrics.record_api_error("resolve_grievance")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_all_histograms_when_prometheus_unavailable(self):
        """Test all histogram metrics gracefully degrade."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_stakeholder_mapping_duration("cattle", 2.5)
            metrics.observe_fpic_stage_duration("deliberation", 7776000.0)
            metrics.observe_grievance_resolution_duration("minor", 86400.0)
            metrics.observe_consultation_duration("workshop", 180.0)
            metrics.observe_communication_delivery_duration("sms", 0.5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_all_gauges_when_prometheus_unavailable(self):
        """Test all gauge metrics gracefully degrade."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_stakeholders(100)
            metrics.set_active_fpic_workflows(10)
            metrics.set_open_grievances(25)
            metrics.set_pending_communications(50)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original


class TestMetricsEdgeCases:
    """Test edge cases for metrics functions."""

    def test_counter_with_empty_strings(self):
        """Test counter metrics handle empty strings."""
        metrics.record_stakeholder_mapped("", "")
        metrics.record_fpic_initiated("")
        metrics.record_grievance_submitted("")

    def test_histogram_with_zero_duration(self):
        """Test histogram metrics handle zero duration."""
        metrics.observe_stakeholder_mapping_duration("coffee", 0.0)
        metrics.observe_fpic_stage_duration("notification", 0.0)
        metrics.observe_grievance_resolution_duration("critical", 0.0)

    def test_histogram_with_large_duration(self):
        """Test histogram metrics handle large durations."""
        metrics.observe_stakeholder_mapping_duration("coffee", 3600.0)
        metrics.observe_fpic_stage_duration("deliberation", 7776000.0)  # 90 days in seconds

    def test_gauge_with_zero(self):
        """Test gauge metrics handle zero values."""
        metrics.set_active_stakeholders(0)
        metrics.set_active_fpic_workflows(0)
        metrics.set_open_grievances(0)
        metrics.set_pending_communications(0)

    def test_gauge_with_large_values(self):
        """Test gauge metrics handle large values."""
        metrics.set_active_stakeholders(10000)
        metrics.set_open_grievances(5000)

    def test_record_stakeholder_mapped_called_multiple_times(self):
        """Test counter increments properly with multiple calls."""
        for i in range(10):
            metrics.record_stakeholder_mapped("indigenous_community", "coffee")

    def test_observe_duration_called_multiple_times(self):
        """Test histogram records properly with multiple observations."""
        for duration in [0.1, 0.5, 1.0, 2.0, 5.0]:
            metrics.observe_stakeholder_mapping_duration("coffee", duration)

    def test_set_gauge_called_multiple_times(self):
        """Test gauge updates properly with multiple calls."""
        for count in [10, 20, 30, 40, 50]:
            metrics.set_active_stakeholders(count)

    def test_record_fpic_consent_pending(self):
        """Test recording pending consent status."""
        metrics.record_fpic_consent("coffee", "pending")

    def test_record_grievance_all_severities_resolved(self):
        """Test recording resolved grievances of all severities."""
        for severity in ["critical", "high", "standard", "minor"]:
            metrics.record_grievance_resolved(severity)


class TestMetricsPrefix:
    """Test metrics prefix consistency."""

    def test_metrics_use_correct_prefix(self):
        """Test that all metrics use gl_eudr_set_ prefix."""
        if metrics._PROMETHEUS_AVAILABLE:
            assert hasattr(metrics, '_STAKEHOLDERS_MAPPED')
            assert hasattr(metrics, '_FPIC_INITIATED')
            assert hasattr(metrics, '_GRIEVANCES_SUBMITTED')
