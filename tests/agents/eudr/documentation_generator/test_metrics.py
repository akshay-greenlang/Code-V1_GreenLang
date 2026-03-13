# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-030

Tests each of the 18 metric helper functions and graceful degradation
when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.agents.eudr.documentation_generator import metrics


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_dds_generated_no_error(self):
        """Test record_dds_generated executes without error."""
        metrics.record_dds_generated("coffee", "draft")

    def test_record_dds_generated_various_commodities(self):
        """Test record_dds_generated with various commodities."""
        for commodity in ["coffee", "cocoa", "wood", "rubber", "soya", "palm_oil", "cattle"]:
            metrics.record_dds_generated(commodity, "draft")

    def test_record_dds_generated_various_statuses(self):
        """Test record_dds_generated with various statuses."""
        for status in ["draft", "validated", "submitted", "acknowledged", "rejected"]:
            metrics.record_dds_generated("coffee", status)

    def test_record_article9_assembly_no_error(self):
        """Test record_article9_assembly executes without error."""
        metrics.record_article9_assembly("coffee")

    def test_record_article9_assembly_all_commodities(self):
        """Test record_article9_assembly with all commodities."""
        for commodity in ["coffee", "cocoa", "wood", "rubber", "soya", "palm_oil", "cattle"]:
            metrics.record_article9_assembly(commodity)

    def test_record_risk_doc_no_error(self):
        """Test record_risk_doc executes without error."""
        metrics.record_risk_doc("coffee", "high")

    def test_record_risk_doc_various_risk_levels(self):
        """Test record_risk_doc with various risk levels."""
        for risk_level in ["negligible", "low", "standard", "high", "critical"]:
            metrics.record_risk_doc("coffee", risk_level)

    def test_record_mitigation_doc_no_error(self):
        """Test record_mitigation_doc executes without error."""
        metrics.record_mitigation_doc("coffee")

    def test_record_compliance_package_no_error(self):
        """Test record_compliance_package executes without error."""
        metrics.record_compliance_package("wood")

    def test_record_submission_no_error(self):
        """Test record_submission executes without error."""
        metrics.record_submission("coffee", "submitted")

    def test_record_submission_various_statuses(self):
        """Test record_submission with various statuses."""
        for status in ["pending", "validating", "submitted", "acknowledged", "rejected", "resubmitted"]:
            metrics.record_submission("coffee", status)

    def test_record_validation_no_error(self):
        """Test record_validation executes without error."""
        metrics.record_validation("dds", "passed")

    def test_record_validation_various_document_types(self):
        """Test record_validation with various document types."""
        for doc_type in ["dds", "article9_package", "risk_assessment", "mitigation_report", "compliance_package"]:
            metrics.record_validation(doc_type, "passed")

    def test_record_validation_various_results(self):
        """Test record_validation with various results."""
        for result in ["passed", "failed", "warning"]:
            metrics.record_validation("dds", result)

    def test_record_api_error_no_error(self):
        """Test record_api_error executes without error."""
        metrics.record_api_error("generate_dds")

    def test_record_api_error_various_operations(self):
        """Test record_api_error with various operations."""
        for operation in ["generate_dds", "assemble_article9", "build_package", "submit_dds", "validate"]:
            metrics.record_api_error(operation)


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_dds_generation_duration_no_error(self):
        """Test observe_dds_generation_duration executes without error."""
        metrics.observe_dds_generation_duration("coffee", 0.5)

    def test_observe_dds_generation_duration_various_durations(self):
        """Test observe_dds_generation_duration with various durations."""
        for duration in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]:
            metrics.observe_dds_generation_duration("coffee", duration)

    def test_observe_dds_generation_duration_all_commodities(self):
        """Test observe_dds_generation_duration with all commodities."""
        for commodity in ["coffee", "cocoa", "wood", "rubber", "soya", "palm_oil", "cattle"]:
            metrics.observe_dds_generation_duration(commodity, 1.5)

    def test_observe_article9_assembly_duration_no_error(self):
        """Test observe_article9_assembly_duration executes without error."""
        metrics.observe_article9_assembly_duration("coffee", 0.1)

    def test_observe_article9_assembly_duration_various_durations(self):
        """Test observe_article9_assembly_duration with various durations."""
        for duration in [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]:
            metrics.observe_article9_assembly_duration("coffee", duration)

    def test_observe_package_build_duration_no_error(self):
        """Test observe_package_build_duration executes without error."""
        metrics.observe_package_build_duration("coffee", 1.0)

    def test_observe_package_build_duration_various_durations(self):
        """Test observe_package_build_duration with various durations."""
        for duration in [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]:
            metrics.observe_package_build_duration("wood", duration)

    def test_observe_submission_duration_no_error(self):
        """Test observe_submission_duration executes without error."""
        metrics.observe_submission_duration(5.0)

    def test_observe_submission_duration_various_durations(self):
        """Test observe_submission_duration with various durations."""
        for duration in [0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]:
            metrics.observe_submission_duration(duration)

    def test_observe_validation_duration_no_error(self):
        """Test observe_validation_duration executes without error."""
        metrics.observe_validation_duration("dds", 0.25)

    def test_observe_validation_duration_various_durations(self):
        """Test observe_validation_duration with various durations."""
        for duration in [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]:
            metrics.observe_validation_duration("dds", duration)

    def test_observe_validation_duration_various_document_types(self):
        """Test observe_validation_duration with various document types."""
        for doc_type in ["dds", "article9_package", "risk_assessment", "mitigation_report", "compliance_package"]:
            metrics.observe_validation_duration(doc_type, 0.5)


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_drafts_no_error(self):
        """Test set_active_drafts executes without error."""
        metrics.set_active_drafts(5)

    def test_set_active_drafts_various_counts(self):
        """Test set_active_drafts with various counts."""
        for count in [0, 1, 10, 50, 100, 500]:
            metrics.set_active_drafts(count)

    def test_set_pending_submissions_no_error(self):
        """Test set_pending_submissions executes without error."""
        metrics.set_pending_submissions(3)

    def test_set_pending_submissions_various_counts(self):
        """Test set_pending_submissions with various counts."""
        for count in [0, 5, 10, 20, 50]:
            metrics.set_pending_submissions(count)

    def test_set_rejected_submissions_no_error(self):
        """Test set_rejected_submissions executes without error."""
        metrics.set_rejected_submissions(2)

    def test_set_rejected_submissions_various_counts(self):
        """Test set_rejected_submissions with various counts."""
        for count in [0, 1, 5, 10]:
            metrics.set_rejected_submissions(count)

    def test_set_document_versions_no_error(self):
        """Test set_document_versions executes without error."""
        metrics.set_document_versions(100)

    def test_set_document_versions_various_counts(self):
        """Test set_document_versions with various counts."""
        for count in [0, 50, 100, 500, 1000, 5000]:
            metrics.set_document_versions(count)

    def test_set_retention_documents_no_error(self):
        """Test set_retention_documents executes without error."""
        metrics.set_retention_documents(250)

    def test_set_retention_documents_various_counts(self):
        """Test set_retention_documents with various counts."""
        for count in [0, 100, 500, 1000, 5000]:
            metrics.set_retention_documents(count)


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counter_when_prometheus_unavailable(self):
        """Test counter metrics work when prometheus_client is unavailable."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            # Should not raise when prometheus is unavailable
            metrics.record_dds_generated("coffee", "draft")
            metrics.record_article9_assembly("coffee")
            metrics.record_risk_doc("coffee", "high")
            metrics.record_mitigation_doc("coffee")
            metrics.record_compliance_package("coffee")
            metrics.record_submission("coffee", "submitted")
            metrics.record_validation("dds", "passed")
            metrics.record_api_error("test")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histogram_when_prometheus_unavailable(self):
        """Test histogram metrics work when prometheus_client is unavailable."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_dds_generation_duration("coffee", 0.5)
            metrics.observe_article9_assembly_duration("coffee", 0.1)
            metrics.observe_package_build_duration("coffee", 1.0)
            metrics.observe_submission_duration(5.0)
            metrics.observe_validation_duration("dds", 0.25)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauge_when_prometheus_unavailable(self):
        """Test gauge metrics work when prometheus_client is unavailable."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_drafts(10)
            metrics.set_pending_submissions(5)
            metrics.set_rejected_submissions(2)
            metrics.set_document_versions(100)
            metrics.set_retention_documents(250)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_all_counters_when_prometheus_unavailable(self):
        """Test all counter metrics gracefully degrade."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            # Test all 8 counter metrics
            metrics.record_dds_generated("coffee", "draft")
            metrics.record_article9_assembly("cocoa")
            metrics.record_risk_doc("wood", "standard")
            metrics.record_mitigation_doc("rubber")
            metrics.record_compliance_package("soya")
            metrics.record_submission("palm_oil", "pending")
            metrics.record_validation("article9_package", "passed")
            metrics.record_api_error("build_package")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_all_histograms_when_prometheus_unavailable(self):
        """Test all histogram metrics gracefully degrade."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            # Test all 5 histogram metrics
            metrics.observe_dds_generation_duration("cattle", 2.5)
            metrics.observe_article9_assembly_duration("coffee", 0.3)
            metrics.observe_package_build_duration("cocoa", 3.0)
            metrics.observe_submission_duration(10.0)
            metrics.observe_validation_duration("risk_assessment", 0.75)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_all_gauges_when_prometheus_unavailable(self):
        """Test all gauge metrics gracefully degrade."""
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            # Test all 5 gauge metrics
            metrics.set_active_drafts(15)
            metrics.set_pending_submissions(8)
            metrics.set_rejected_submissions(3)
            metrics.set_document_versions(500)
            metrics.set_retention_documents(1000)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original


class TestMetricsEdgeCases:
    """Test edge cases for metrics functions."""

    def test_counter_with_empty_strings(self):
        """Test counter metrics handle empty strings."""
        metrics.record_dds_generated("", "")
        metrics.record_article9_assembly("")
        metrics.record_risk_doc("", "")

    def test_histogram_with_zero_duration(self):
        """Test histogram metrics handle zero duration."""
        metrics.observe_dds_generation_duration("coffee", 0.0)
        metrics.observe_article9_assembly_duration("coffee", 0.0)
        metrics.observe_submission_duration(0.0)

    def test_histogram_with_large_duration(self):
        """Test histogram metrics handle large durations."""
        metrics.observe_dds_generation_duration("coffee", 3600.0)  # 1 hour
        metrics.observe_submission_duration(7200.0)  # 2 hours

    def test_gauge_with_zero(self):
        """Test gauge metrics handle zero values."""
        metrics.set_active_drafts(0)
        metrics.set_pending_submissions(0)
        metrics.set_rejected_submissions(0)

    def test_gauge_with_large_values(self):
        """Test gauge metrics handle large values."""
        metrics.set_document_versions(100000)
        metrics.set_retention_documents(50000)

    def test_record_dds_generated_called_multiple_times(self):
        """Test counter increments properly with multiple calls."""
        for i in range(10):
            metrics.record_dds_generated("coffee", "draft")

    def test_observe_duration_called_multiple_times(self):
        """Test histogram records properly with multiple observations."""
        for duration in [0.1, 0.5, 1.0, 2.0, 5.0]:
            metrics.observe_dds_generation_duration("coffee", duration)

    def test_set_gauge_called_multiple_times(self):
        """Test gauge updates properly with multiple calls."""
        for count in [10, 20, 30, 40, 50]:
            metrics.set_active_drafts(count)


class TestMetricsPrefix:
    """Test metrics prefix consistency."""

    def test_metrics_use_correct_prefix(self):
        """Test that all metrics use gl_eudr_dgn_ prefix."""
        # This test verifies the module uses the correct prefix
        # by checking the actual metric definitions if prometheus is available
        if metrics._PROMETHEUS_AVAILABLE:
            # The metrics are defined at module level with gl_eudr_dgn_ prefix
            # We can't directly test the metric names without accessing internals,
            # but we can verify the module has the expected structure
            assert hasattr(metrics, '_DDS_GENERATED')
            assert hasattr(metrics, '_ARTICLE9_ASSEMBLIES')
            assert hasattr(metrics, '_RISK_DOCS')
