# -*- coding: utf-8 -*-
"""
Unit tests for Spend Data Categorizer Metrics (AGENT-DATA-009)

Tests the PROMETHEUS_AVAILABLE flag, all 12 metric objects, all 12 helper
functions, metric type assertions with prometheus_client installed, graceful
no-op behaviour when prometheus_client is absent, and __all__ export list.

Target: 55+ tests for comprehensive metrics coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from greenlang.spend_categorizer import metrics as metrics_module
from greenlang.spend_categorizer.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    spend_cat_records_ingested_total,
    spend_cat_records_classified_total,
    spend_cat_scope3_mapped_total,
    spend_cat_emissions_calculated_total,
    spend_cat_rules_evaluated_total,
    spend_cat_reports_generated_total,
    spend_cat_classification_confidence,
    spend_cat_processing_duration_seconds,
    spend_cat_active_batches,
    spend_cat_total_spend_usd,
    spend_cat_processing_errors_total,
    spend_cat_emission_factor_lookups_total,
    # Helper functions
    record_ingestion,
    record_classification,
    record_scope3_mapping,
    record_emission_calculation,
    record_rule_evaluation,
    record_report_generation,
    record_classification_confidence,
    record_processing_duration,
    update_active_batches,
    update_total_spend,
    record_processing_error,
    record_factor_lookup,
)


# ============================================================================
# PROMETHEUS_AVAILABLE flag tests
# ============================================================================


class TestPrometheusFlag:
    """Test the PROMETHEUS_AVAILABLE flag."""

    def test_flag_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_flag_reflects_import(self):
        """PROMETHEUS_AVAILABLE should be True if prometheus_client is importable."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            assert PROMETHEUS_AVAILABLE is False


# ============================================================================
# Metric object existence tests
# ============================================================================


class TestMetricObjects:
    """Test that all 12 metric objects exist in the module."""

    def test_records_ingested_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_records_ingested_total")

    def test_records_classified_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_records_classified_total")

    def test_scope3_mapped_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_scope3_mapped_total")

    def test_emissions_calculated_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_emissions_calculated_total")

    def test_rules_evaluated_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_rules_evaluated_total")

    def test_reports_generated_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_reports_generated_total")

    def test_classification_confidence_exists(self):
        assert hasattr(metrics_module, "spend_cat_classification_confidence")

    def test_processing_duration_seconds_exists(self):
        assert hasattr(metrics_module, "spend_cat_processing_duration_seconds")

    def test_active_batches_exists(self):
        assert hasattr(metrics_module, "spend_cat_active_batches")

    def test_total_spend_usd_exists(self):
        assert hasattr(metrics_module, "spend_cat_total_spend_usd")

    def test_processing_errors_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_processing_errors_total")

    def test_emission_factor_lookups_total_exists(self):
        assert hasattr(metrics_module, "spend_cat_emission_factor_lookups_total")


# ============================================================================
# Helper function callable tests
# ============================================================================


class TestHelperFunctions:
    """Test that all 12 helper functions are callable with correct args."""

    def test_record_ingestion_callable(self):
        # Should not raise
        record_ingestion("csv")

    def test_record_classification_callable(self):
        record_classification("unspsc")

    def test_record_scope3_mapping_callable(self):
        record_scope3_mapping("cat1")

    def test_record_emission_calculation_callable(self):
        record_emission_calculation("eeio")

    def test_record_rule_evaluation_callable(self):
        record_rule_evaluation("match")

    def test_record_report_generation_callable(self):
        record_report_generation("json")

    def test_record_classification_confidence_callable(self):
        record_classification_confidence(0.85)

    def test_record_processing_duration_callable(self):
        record_processing_duration("ingest", 1.5)

    def test_update_active_batches_increment(self):
        update_active_batches(1)

    def test_update_active_batches_decrement(self):
        update_active_batches(-1)

    def test_update_active_batches_zero(self):
        # delta=0 should be a no-op
        update_active_batches(0)

    def test_update_total_spend_callable(self):
        update_total_spend(50000.0)

    def test_record_processing_error_callable(self):
        record_processing_error("validation")

    def test_record_factor_lookup_callable(self):
        record_factor_lookup("exiobase")


# ============================================================================
# Tests WITH prometheus_client installed
# ============================================================================


@pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestWithPrometheus:
    """Test metric types and labels when prometheus_client is installed."""

    def test_records_ingested_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_records_ingested_total, Counter)

    def test_records_classified_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_records_classified_total, Counter)

    def test_scope3_mapped_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_scope3_mapped_total, Counter)

    def test_emissions_calculated_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_emissions_calculated_total, Counter)

    def test_rules_evaluated_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_rules_evaluated_total, Counter)

    def test_reports_generated_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_reports_generated_total, Counter)

    def test_classification_confidence_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(spend_cat_classification_confidence, Histogram)

    def test_processing_duration_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(spend_cat_processing_duration_seconds, Histogram)

    def test_active_batches_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(spend_cat_active_batches, Gauge)

    def test_total_spend_usd_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(spend_cat_total_spend_usd, Gauge)

    def test_processing_errors_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_processing_errors_total, Counter)

    def test_emission_factor_lookups_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(spend_cat_emission_factor_lookups_total, Counter)

    def test_records_ingested_has_source_label(self):
        assert "source" in spend_cat_records_ingested_total._labelnames

    def test_records_classified_has_taxonomy_label(self):
        assert "taxonomy" in spend_cat_records_classified_total._labelnames

    def test_scope3_mapped_has_category_label(self):
        assert "category" in spend_cat_scope3_mapped_total._labelnames

    def test_emissions_calculated_has_source_label(self):
        assert "source" in spend_cat_emissions_calculated_total._labelnames

    def test_rules_evaluated_has_result_label(self):
        assert "result" in spend_cat_rules_evaluated_total._labelnames

    def test_reports_generated_has_format_label(self):
        assert "format" in spend_cat_reports_generated_total._labelnames

    def test_processing_duration_has_operation_label(self):
        assert "operation" in spend_cat_processing_duration_seconds._labelnames

    def test_processing_errors_has_error_type_label(self):
        assert "error_type" in spend_cat_processing_errors_total._labelnames

    def test_emission_factor_lookups_has_source_label(self):
        assert "source" in spend_cat_emission_factor_lookups_total._labelnames


# ============================================================================
# Tests WITHOUT prometheus_client (simulated)
# ============================================================================


class TestWithoutPrometheus:
    """Test graceful no-op when prometheus_client is not installed.

    Simulates the absence of prometheus_client by patching PROMETHEUS_AVAILABLE
    in the metrics module to False and setting metrics to None.
    """

    def test_record_ingestion_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            # Should not raise
            metrics_module.record_ingestion("csv")

    def test_record_classification_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_classification("naics")

    def test_record_scope3_mapping_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_scope3_mapping("cat3")

    def test_record_emission_calculation_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_emission_calculation("defra")

    def test_record_rule_evaluation_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_rule_evaluation("no_match")

    def test_record_report_generation_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_report_generation("csv")

    def test_record_classification_confidence_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_classification_confidence(0.75)

    def test_record_processing_duration_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_processing_duration("classify", 2.0)

    def test_update_active_batches_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.update_active_batches(1)
            metrics_module.update_active_batches(-1)

    def test_update_total_spend_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.update_total_spend(10000.0)

    def test_record_processing_error_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_processing_error("timeout")

    def test_record_factor_lookup_noop(self):
        with patch.object(metrics_module, "PROMETHEUS_AVAILABLE", False):
            metrics_module.record_factor_lookup("ecoinvent")


# ============================================================================
# __all__ export tests
# ============================================================================


class TestMetricsExports:
    """Test __all__ export list completeness."""

    def test_all_is_list(self):
        assert isinstance(metrics_module.__all__, list)

    def test_all_count(self):
        # 1 flag + 12 metric objects + 12 helper functions = 25
        assert len(metrics_module.__all__) == 25, (
            f"Expected 25 exports, got {len(metrics_module.__all__)}"
        )

    def test_prometheus_available_in_all(self):
        assert "PROMETHEUS_AVAILABLE" in metrics_module.__all__

    def test_metric_objects_in_all(self):
        metric_names = [
            "spend_cat_records_ingested_total",
            "spend_cat_records_classified_total",
            "spend_cat_scope3_mapped_total",
            "spend_cat_emissions_calculated_total",
            "spend_cat_rules_evaluated_total",
            "spend_cat_reports_generated_total",
            "spend_cat_classification_confidence",
            "spend_cat_processing_duration_seconds",
            "spend_cat_active_batches",
            "spend_cat_total_spend_usd",
            "spend_cat_processing_errors_total",
            "spend_cat_emission_factor_lookups_total",
        ]
        for name in metric_names:
            assert name in metrics_module.__all__, f"{name} missing from __all__"

    def test_helper_functions_in_all(self):
        helpers = [
            "record_ingestion",
            "record_classification",
            "record_scope3_mapping",
            "record_emission_calculation",
            "record_rule_evaluation",
            "record_report_generation",
            "record_classification_confidence",
            "record_processing_duration",
            "update_active_batches",
            "update_total_spend",
            "record_processing_error",
            "record_factor_lookup",
        ]
        for name in helpers:
            assert name in metrics_module.__all__, f"{name} missing from __all__"

    def test_all_entries_are_accessible(self):
        for name in metrics_module.__all__:
            assert hasattr(metrics_module, name), (
                f"__all__ entry '{name}' is not accessible on metrics module"
            )
