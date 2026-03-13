# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-039

Tests all metric helper functions (counters, histograms, gauges)
and graceful degradation when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.customs_declaration_support import metrics


# ====================================================================
# Counter Metric Tests
# ====================================================================


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_declaration_created(self):
        metrics.record_declaration_created("import")

    def test_record_declaration_created_export(self):
        metrics.record_declaration_created("export")

    def test_record_declaration_created_transit(self):
        metrics.record_declaration_created("transit")

    def test_record_declaration_submitted(self):
        metrics.record_declaration_submitted("ncts")

    def test_record_declaration_submitted_ais(self):
        metrics.record_declaration_submitted("ais")

    def test_record_declaration_cleared(self):
        metrics.record_declaration_cleared()

    def test_record_declaration_rejected(self):
        metrics.record_declaration_rejected("missing_dds_reference")

    def test_record_declaration_rejected_invalid_cn(self):
        metrics.record_declaration_rejected("invalid_cn_code")

    def test_record_cn_code_mapped(self):
        metrics.record_cn_code_mapped("cocoa")

    def test_record_cn_code_mapped_coffee(self):
        metrics.record_cn_code_mapped("coffee")

    def test_record_cn_code_mapped_wood(self):
        metrics.record_cn_code_mapped("wood")

    def test_record_hs_code_validated(self):
        metrics.record_hs_code_validated("180100")

    def test_record_hs_code_validated_coffee(self):
        metrics.record_hs_code_validated("090111")

    def test_record_tariff_calculated(self):
        metrics.record_tariff_calculated()

    def test_record_origin_verified(self):
        metrics.record_origin_verified("verified")

    def test_record_origin_verified_mismatch(self):
        metrics.record_origin_verified("mismatch")

    def test_record_compliance_check(self):
        metrics.record_compliance_check("pass")

    def test_record_compliance_check_fail(self):
        metrics.record_compliance_check("fail")

    def test_record_sad_form_generated(self):
        metrics.record_sad_form_generated()

    def test_record_mrn_generated(self):
        metrics.record_mrn_generated()

    def test_record_customs_submission(self):
        metrics.record_customs_submission("ncts")

    def test_record_customs_submission_ais(self):
        metrics.record_customs_submission("ais")

    def test_record_batch_operation(self):
        metrics.record_batch_operation()

    def test_record_currency_conversion(self):
        metrics.record_currency_conversion("USD", "EUR")


# ====================================================================
# Histogram Metric Tests
# ====================================================================


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_declaration_generation_duration(self):
        metrics.observe_declaration_generation_duration(0.5)

    def test_observe_declaration_generation_duration_zero(self):
        metrics.observe_declaration_generation_duration(0.0)

    def test_observe_cn_code_mapping_duration(self):
        metrics.observe_cn_code_mapping_duration(0.1)

    def test_observe_hs_code_validation_duration(self):
        metrics.observe_hs_code_validation_duration(0.05)

    def test_observe_tariff_calculation_duration(self):
        metrics.observe_tariff_calculation_duration(1.2)

    def test_observe_tariff_calculation_duration_large(self):
        metrics.observe_tariff_calculation_duration(30.0)

    def test_observe_origin_verification_duration(self):
        metrics.observe_origin_verification_duration(0.8)

    def test_observe_compliance_check_duration(self):
        metrics.observe_compliance_check_duration(2.5)

    def test_observe_sad_form_generation_duration(self):
        metrics.observe_sad_form_generation_duration(3.0)

    def test_observe_customs_submission_duration(self):
        metrics.observe_customs_submission_duration(5.0)

    def test_observe_customs_submission_duration_large(self):
        metrics.observe_customs_submission_duration(60.0)

    def test_observe_currency_conversion_duration(self):
        metrics.observe_currency_conversion_duration(0.3)

    def test_observe_value_calculation_duration(self):
        metrics.observe_value_calculation_duration(0.15)

    def test_observe_mrn_generation_duration(self):
        metrics.observe_mrn_generation_duration(0.02)

    def test_observe_batch_processing_duration(self):
        metrics.observe_batch_processing_duration(10.0)


# ====================================================================
# Gauge Metric Tests
# ====================================================================


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_declarations(self):
        metrics.set_active_declarations(10)

    def test_set_active_declarations_zero(self):
        metrics.set_active_declarations(0)

    def test_set_pending_declarations(self):
        metrics.set_pending_declarations(5)

    def test_set_pending_declarations_zero(self):
        metrics.set_pending_declarations(0)

    def test_set_submitted_declarations(self):
        metrics.set_submitted_declarations(8)

    def test_set_cleared_declarations(self):
        metrics.set_cleared_declarations(15)

    def test_set_rejected_declarations(self):
        metrics.set_rejected_declarations(2)

    def test_set_rejected_declarations_zero(self):
        metrics.set_rejected_declarations(0)

    def test_set_total_customs_value(self):
        metrics.set_total_customs_value(500000.00)

    def test_set_total_duty_collected(self):
        metrics.set_total_duty_collected(25000.00)

    def test_set_total_vat_collected(self):
        metrics.set_total_vat_collected(105000.00)

    def test_set_average_processing_time(self):
        metrics.set_average_processing_time(2.5)

    def test_set_compliance_pass_rate(self):
        metrics.set_compliance_pass_rate(95.5)

    def test_set_origin_verification_rate(self):
        metrics.set_origin_verification_rate(88.0)

    def test_set_ncts_queue_depth(self):
        metrics.set_ncts_queue_depth(3)

    def test_set_ais_queue_depth(self):
        metrics.set_ais_queue_depth(7)


# ====================================================================
# Graceful Degradation Tests
# ====================================================================


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counters_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.record_declaration_created("import")
            metrics.record_declaration_submitted("ncts")
            metrics.record_declaration_cleared()
            metrics.record_declaration_rejected("missing_dds")
            metrics.record_cn_code_mapped("cocoa")
            metrics.record_hs_code_validated("180100")
            metrics.record_tariff_calculated()
            metrics.record_origin_verified("verified")
            metrics.record_compliance_check("pass")
            metrics.record_sad_form_generated()
            metrics.record_mrn_generated()
            metrics.record_customs_submission("ncts")
            metrics.record_batch_operation()
            metrics.record_currency_conversion("USD", "EUR")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histograms_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_declaration_generation_duration(0.5)
            metrics.observe_cn_code_mapping_duration(0.1)
            metrics.observe_hs_code_validation_duration(0.05)
            metrics.observe_tariff_calculation_duration(1.2)
            metrics.observe_origin_verification_duration(0.8)
            metrics.observe_compliance_check_duration(2.5)
            metrics.observe_sad_form_generation_duration(3.0)
            metrics.observe_customs_submission_duration(5.0)
            metrics.observe_currency_conversion_duration(0.3)
            metrics.observe_value_calculation_duration(0.15)
            metrics.observe_mrn_generation_duration(0.02)
            metrics.observe_batch_processing_duration(10.0)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauges_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_declarations(10)
            metrics.set_pending_declarations(5)
            metrics.set_submitted_declarations(8)
            metrics.set_cleared_declarations(15)
            metrics.set_rejected_declarations(2)
            metrics.set_total_customs_value(500000.00)
            metrics.set_total_duty_collected(25000.00)
            metrics.set_total_vat_collected(105000.00)
            metrics.set_average_processing_time(2.5)
            metrics.set_compliance_pass_rate(95.5)
            metrics.set_origin_verification_rate(88.0)
            metrics.set_ncts_queue_depth(3)
            metrics.set_ais_queue_depth(7)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original
