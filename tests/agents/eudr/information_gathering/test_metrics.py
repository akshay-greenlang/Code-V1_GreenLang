# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus Metrics helpers - AGENT-EUDR-027

Tests all 18 metric helper functions. Each function should not raise
regardless of whether prometheus_client is installed or not. This test
validates the no-op fallback behavior when metrics are disabled.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (GL-EUDR-IGA-027)
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.information_gathering.metrics import (
    observe_aggregation_duration,
    observe_certification_duration,
    observe_external_query_duration,
    observe_harvest_duration,
    observe_package_assembly_duration,
    record_api_error,
    record_certification_verified,
    record_completeness_validation,
    record_external_query,
    record_gathering_operation,
    record_normalization_error,
    record_package_assembled,
    record_public_data_harvest,
    record_supplier_aggregated,
    set_active_operations,
    set_cache_hit_ratio,
    set_expiring_certificates,
    set_stale_data_sources,
)


class TestCounterMetrics:
    """Test counter metric helper functions (8 counters)."""

    def test_record_gathering_operation(self):
        # Should not raise
        record_gathering_operation("coffee", "completed")

    def test_record_external_query(self):
        record_external_query("eu_traces", "success")

    def test_record_certification_verified(self):
        record_certification_verified("fsc", "valid")

    def test_record_public_data_harvest(self):
        record_public_data_harvest("fao_stat")

    def test_record_supplier_aggregated(self):
        record_supplier_aggregated("cocoa")

    def test_record_completeness_validation(self):
        record_completeness_validation("complete")

    def test_record_package_assembled(self):
        record_package_assembled("wood")

    def test_record_api_error(self):
        record_api_error("external_query")


class TestHistogramMetrics:
    """Test histogram metric helper functions (5 histograms)."""

    def test_observe_external_query_duration(self):
        observe_external_query_duration("eu_traces", 0.5)

    def test_observe_certification_duration(self):
        observe_certification_duration("fsc", 0.25)

    def test_observe_harvest_duration(self):
        observe_harvest_duration("fao_stat", 15.0)

    def test_observe_aggregation_duration(self):
        observe_aggregation_duration(1.2)

    def test_observe_package_assembly_duration(self):
        observe_package_assembly_duration("coffee", 30.0)


class TestGaugeMetrics:
    """Test gauge metric helper functions (5 gauges)."""

    def test_set_active_operations(self):
        set_active_operations(5)

    def test_set_stale_data_sources(self):
        set_stale_data_sources(2)

    def test_set_expiring_certificates(self):
        set_expiring_certificates(10)

    def test_set_cache_hit_ratio(self):
        set_cache_hit_ratio(0.85)

    def test_record_normalization_error(self):
        record_normalization_error("coordinate")


class TestMetricsEdgeCases:
    """Test edge cases for metric helpers."""

    def test_record_gathering_operation_empty_strings(self):
        record_gathering_operation("", "")

    def test_observe_zero_duration(self):
        observe_external_query_duration("eu_traces", 0.0)

    def test_set_gauge_zero(self):
        set_active_operations(0)
        set_stale_data_sources(0)
        set_expiring_certificates(0)
        set_cache_hit_ratio(0.0)
