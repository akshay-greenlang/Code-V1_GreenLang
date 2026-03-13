# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-036

Tests each of the 18 metric helper functions and graceful degradation
when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.eu_information_system_interface.metrics import (
    record_dds_submitted,
    record_dds_accepted,
    record_dds_rejected,
    record_operator_registered,
    record_package_assembled,
    record_status_check,
    record_api_call,
    record_api_error,
    observe_submission_duration,
    observe_geolocation_format_duration,
    observe_package_assembly_duration,
    observe_api_call_duration,
    observe_status_check_duration,
    set_active_submissions,
    set_pending_dds,
    set_registered_operators,
    set_eu_api_health,
    set_audit_records_count,
)


class TestCounterMetrics:
    """Test counter metric functions."""

    def test_record_dds_submitted(self):
        record_dds_submitted(commodity="cocoa", dds_type="placing")

    def test_record_dds_submitted_export(self):
        record_dds_submitted(commodity="coffee", dds_type="export")

    def test_record_dds_accepted(self):
        record_dds_accepted(commodity="cocoa")

    def test_record_dds_accepted_rubber(self):
        record_dds_accepted(commodity="rubber")

    def test_record_dds_rejected(self):
        record_dds_rejected(commodity="cocoa", reason="validation_error")

    def test_record_dds_rejected_missing_geolocation(self):
        record_dds_rejected(commodity="palm_oil", reason="missing_geolocation")

    def test_record_operator_registered(self):
        record_operator_registered(member_state="DE")

    def test_record_operator_registered_france(self):
        record_operator_registered(member_state="FR")

    def test_record_package_assembled(self):
        record_package_assembled(commodity="cocoa")

    def test_record_status_check(self):
        record_status_check(result="accepted")

    def test_record_status_check_pending(self):
        record_status_check(result="pending")

    def test_record_api_call(self):
        record_api_call(method="POST", endpoint="/dds/submit")

    def test_record_api_call_get(self):
        record_api_call(method="GET", endpoint="/dds/status")

    def test_record_api_error(self):
        record_api_error(operation="submit_dds")

    def test_record_api_error_with_type(self):
        record_api_error(operation="submit_dds", error_type="timeout")


class TestHistogramMetrics:
    """Test histogram metric functions."""

    def test_observe_submission_duration(self):
        observe_submission_duration(commodity="cocoa", duration=1.5)

    def test_observe_geolocation_format_duration(self):
        observe_geolocation_format_duration(duration=0.025)

    def test_observe_package_assembly_duration(self):
        observe_package_assembly_duration(commodity="coffee", duration=0.35)

    def test_observe_api_call_duration(self):
        observe_api_call_duration(endpoint="/dds/submit", duration=0.2)

    def test_observe_api_call_duration_status(self):
        observe_api_call_duration(endpoint="/dds/status", duration=0.15)

    def test_observe_status_check_duration(self):
        observe_status_check_duration(duration=0.1)


class TestGaugeMetrics:
    """Test gauge metric functions."""

    def test_set_active_submissions(self):
        set_active_submissions(5)

    def test_set_pending_dds(self):
        set_pending_dds(10)

    def test_set_registered_operators(self):
        set_registered_operators(25)

    def test_set_eu_api_health_up(self):
        set_eu_api_health(True)

    def test_set_eu_api_health_down(self):
        set_eu_api_health(False)

    def test_set_audit_records_count(self):
        set_audit_records_count(1000)


class TestGracefulDegradation:
    """Test that metrics work without prometheus_client installed."""

    def test_all_counters_no_error(self):
        """All counter functions should run without error regardless."""
        record_dds_submitted(commodity="cocoa", dds_type="placing")
        record_dds_accepted(commodity="cocoa")
        record_dds_rejected(commodity="cocoa", reason="validation_error")
        record_operator_registered(member_state="DE")
        record_package_assembled(commodity="cocoa")
        record_status_check(result="accepted")
        record_api_call(method="POST", endpoint="/dds/submit")
        record_api_error(operation="submit_dds")

    def test_all_histograms_no_error(self):
        """All histogram functions should run without error."""
        observe_submission_duration(commodity="cocoa", duration=1.0)
        observe_geolocation_format_duration(duration=0.01)
        observe_package_assembly_duration(commodity="cocoa", duration=0.5)
        observe_api_call_duration(endpoint="/dds/submit", duration=0.2)
        observe_status_check_duration(duration=0.075)

    def test_all_gauges_no_error(self):
        """All gauge functions should run without error."""
        set_active_submissions(0)
        set_pending_dds(0)
        set_registered_operators(0)
        set_eu_api_health(True)
        set_audit_records_count(0)
