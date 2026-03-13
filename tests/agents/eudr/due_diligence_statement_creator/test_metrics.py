# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-037

Tests all 40 metric helper functions (14 counters, 11 histograms, 15 gauges)
and graceful degradation when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator import metrics


# ====================================================================
# Counter Metric Tests (14 counters)
# ====================================================================


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_statement_created(self):
        metrics.record_statement_created("placing")

    def test_record_statement_created_making_available(self):
        metrics.record_statement_created("making_available")

    def test_record_statement_submitted(self):
        metrics.record_statement_submitted("submitted")

    def test_record_statement_submitted_accepted(self):
        metrics.record_statement_submitted("accepted")

    def test_record_amendment_created(self):
        metrics.record_amendment_created("correction_of_error")

    def test_record_amendment_created_additional(self):
        metrics.record_amendment_created("additional_information")

    def test_record_validation_passed(self):
        metrics.record_validation_passed()

    def test_record_validation_failed(self):
        metrics.record_validation_failed("operator_name")

    def test_record_validation_failed_geolocation(self):
        metrics.record_validation_failed("geolocation")

    def test_record_document_packaged(self):
        metrics.record_document_packaged("certificate_of_origin")

    def test_record_document_packaged_satellite(self):
        metrics.record_document_packaged("satellite_imagery")

    def test_record_signature_applied(self):
        metrics.record_signature_applied("qualified_electronic")

    def test_record_signature_applied_advanced(self):
        metrics.record_signature_applied("advanced_electronic")

    def test_record_geolocation_formatted(self):
        metrics.record_geolocation_formatted("gps_field_survey")

    def test_record_geolocation_formatted_satellite(self):
        metrics.record_geolocation_formatted("satellite_derived")

    def test_record_risk_integration(self):
        metrics.record_risk_integration("EUDR-016")

    def test_record_risk_integration_supplier(self):
        metrics.record_risk_integration("EUDR-017")

    def test_record_supply_chain_compilation(self):
        metrics.record_supply_chain_compilation()

    def test_record_version_created(self):
        metrics.record_version_created()

    def test_record_withdrawal(self):
        metrics.record_withdrawal()

    def test_record_translation(self):
        metrics.record_translation("fr")

    def test_record_translation_de(self):
        metrics.record_translation("de")

    def test_record_batch_operation(self):
        metrics.record_batch_operation()


# ====================================================================
# Histogram Metric Tests (11 histograms)
# ====================================================================


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_statement_generation_duration(self):
        metrics.observe_statement_generation_duration(0.5)

    def test_observe_statement_generation_duration_zero(self):
        metrics.observe_statement_generation_duration(0.0)

    def test_observe_validation_duration(self):
        metrics.observe_validation_duration(1.2)

    def test_observe_validation_duration_large(self):
        metrics.observe_validation_duration(120.0)

    def test_observe_geolocation_formatting_duration(self):
        metrics.observe_geolocation_formatting_duration(0.8)

    def test_observe_geolocation_formatting_duration_small(self):
        metrics.observe_geolocation_formatting_duration(0.001)

    def test_observe_risk_integration_duration(self):
        metrics.observe_risk_integration_duration(2.5)

    def test_observe_supply_chain_compilation_duration(self):
        metrics.observe_supply_chain_compilation_duration(3.0)

    def test_observe_document_packaging_duration(self):
        metrics.observe_document_packaging_duration(1.5)

    def test_observe_signing_duration(self):
        metrics.observe_signing_duration(0.3)

    def test_observe_submission_duration(self):
        metrics.observe_submission_duration(5.0)

    def test_observe_submission_duration_large(self):
        metrics.observe_submission_duration(60.0)

    def test_observe_amendment_duration(self):
        metrics.observe_amendment_duration(1.0)

    def test_observe_translation_duration(self):
        metrics.observe_translation_duration("fr", 2.0)

    def test_observe_version_creation_duration(self):
        metrics.observe_version_creation_duration(0.5)


# ====================================================================
# Gauge Metric Tests (15 gauges)
# ====================================================================


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_statements(self):
        metrics.set_active_statements(10)

    def test_set_active_statements_zero(self):
        metrics.set_active_statements(0)

    def test_set_pending_submissions(self):
        metrics.set_pending_submissions(5)

    def test_set_pending_submissions_zero(self):
        metrics.set_pending_submissions(0)

    def test_set_failed_validations(self):
        metrics.set_failed_validations(3)

    def test_set_failed_validations_zero(self):
        metrics.set_failed_validations(0)

    def test_set_total_commodity_volume(self):
        metrics.set_total_commodity_volume(12345.67)

    def test_set_draft_statements(self):
        metrics.set_draft_statements(8)

    def test_set_draft_statements_zero(self):
        metrics.set_draft_statements(0)

    def test_set_validated_statements(self):
        metrics.set_validated_statements(7)

    def test_set_signed_statements(self):
        metrics.set_signed_statements(6)

    def test_set_submitted_statements(self):
        metrics.set_submitted_statements(5)

    def test_set_accepted_statements(self):
        metrics.set_accepted_statements(4)

    def test_set_rejected_statements(self):
        metrics.set_rejected_statements(2)

    def test_set_rejected_statements_zero(self):
        metrics.set_rejected_statements(0)

    def test_set_amended_statements(self):
        metrics.set_amended_statements(1)

    def test_set_withdrawn_statements(self):
        metrics.set_withdrawn_statements(0)

    def test_set_total_documents(self):
        metrics.set_total_documents(50)

    def test_set_total_geolocations(self):
        metrics.set_total_geolocations(120)

    def test_set_average_risk_score(self):
        metrics.set_average_risk_score(45.5)


# ====================================================================
# Graceful Degradation Tests
# ====================================================================


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counters_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.record_statement_created("placing")
            metrics.record_statement_submitted("submitted")
            metrics.record_amendment_created("correction_of_error")
            metrics.record_validation_passed()
            metrics.record_validation_failed("field_x")
            metrics.record_document_packaged("pdf")
            metrics.record_signature_applied("qualified_electronic")
            metrics.record_geolocation_formatted("gps")
            metrics.record_risk_integration("EUDR-016")
            metrics.record_supply_chain_compilation()
            metrics.record_version_created()
            metrics.record_withdrawal()
            metrics.record_translation("en")
            metrics.record_batch_operation()
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histograms_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_statement_generation_duration(0.5)
            metrics.observe_validation_duration(1.2)
            metrics.observe_geolocation_formatting_duration(0.8)
            metrics.observe_risk_integration_duration(2.5)
            metrics.observe_supply_chain_compilation_duration(3.0)
            metrics.observe_document_packaging_duration(1.5)
            metrics.observe_signing_duration(0.3)
            metrics.observe_submission_duration(5.0)
            metrics.observe_amendment_duration(1.0)
            metrics.observe_translation_duration("fr", 2.0)
            metrics.observe_version_creation_duration(0.5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauges_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_statements(10)
            metrics.set_pending_submissions(5)
            metrics.set_failed_validations(3)
            metrics.set_total_commodity_volume(100.0)
            metrics.set_draft_statements(8)
            metrics.set_validated_statements(7)
            metrics.set_signed_statements(6)
            metrics.set_submitted_statements(5)
            metrics.set_accepted_statements(4)
            metrics.set_rejected_statements(2)
            metrics.set_amended_statements(1)
            metrics.set_withdrawn_statements(0)
            metrics.set_total_documents(50)
            metrics.set_total_geolocations(120)
            metrics.set_average_risk_score(45.5)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original
