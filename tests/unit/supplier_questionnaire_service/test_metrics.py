# -*- coding: utf-8 -*-
"""
Unit tests for Supplier Questionnaire Prometheus Metrics (AGENT-DATA-008)

Tests that all 12 metric objects exist, all record_* helper functions
are callable with correct arguments, and that the module gracefully
handles the absence of prometheus_client.

Target: 30+ tests.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from greenlang.supplier_questionnaire import metrics as metrics_mod


# ============================================================================
# PROMETHEUS_AVAILABLE flag tests
# ============================================================================


class TestPrometheusAvailableFlag:
    def test_prometheus_available_is_bool(self):
        assert isinstance(metrics_mod.PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_matches_import_success(self):
        try:
            import prometheus_client  # noqa: F401
            expected = True
        except ImportError:
            expected = False
        assert metrics_mod.PROMETHEUS_AVAILABLE == expected


# ============================================================================
# Metric object existence tests
# ============================================================================


class TestMetricObjectsExist:
    """All 12 metric objects should be defined (as real metrics or None)."""

    def test_supplier_quest_templates_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_templates_total")

    def test_supplier_quest_distributions_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_distributions_total")

    def test_supplier_quest_responses_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_responses_total")

    def test_supplier_quest_validations_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_validations_total")

    def test_supplier_quest_scores_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_scores_total")

    def test_supplier_quest_followups_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_followups_total")

    def test_supplier_quest_response_rate_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_response_rate")

    def test_supplier_quest_processing_duration_seconds_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_processing_duration_seconds")

    def test_supplier_quest_active_campaigns_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_active_campaigns")

    def test_supplier_quest_pending_responses_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_pending_responses")

    def test_supplier_quest_processing_errors_total_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_processing_errors_total")

    def test_supplier_quest_data_quality_score_exists(self):
        assert hasattr(metrics_mod, "supplier_quest_data_quality_score")


# ============================================================================
# Helper function callable tests (should never raise)
# ============================================================================


class TestHelperFunctionsCallable:
    """All record_* / update_* helpers must be safely callable."""

    def test_record_template_callable(self):
        metrics_mod.record_template("cdp_climate", "created")

    def test_record_distribution_callable(self):
        metrics_mod.record_distribution("email", "sent")

    def test_record_response_callable(self):
        metrics_mod.record_response("portal", "submitted")

    def test_record_validation_callable(self):
        metrics_mod.record_validation("completeness", "pass")

    def test_record_score_callable(self):
        metrics_mod.record_score("ecovadis", "leader")

    def test_record_followup_callable(self):
        metrics_mod.record_followup("reminder", "sent")

    def test_update_response_rate_callable(self):
        metrics_mod.update_response_rate("camp-001", 75.0)

    def test_record_processing_duration_callable(self):
        metrics_mod.record_processing_duration("validate", 1.23)

    def test_update_active_campaigns_increment(self):
        metrics_mod.update_active_campaigns(1)

    def test_update_active_campaigns_decrement(self):
        metrics_mod.update_active_campaigns(-1)

    def test_update_active_campaigns_zero_noop(self):
        # delta=0 should not crash
        metrics_mod.update_active_campaigns(0)

    def test_update_pending_responses_increment(self):
        metrics_mod.update_pending_responses(5)

    def test_update_pending_responses_decrement(self):
        metrics_mod.update_pending_responses(-3)

    def test_update_pending_responses_zero_noop(self):
        metrics_mod.update_pending_responses(0)

    def test_record_processing_error_callable(self):
        metrics_mod.record_processing_error("template", "validation")

    def test_record_data_quality_callable(self):
        metrics_mod.record_data_quality("gri", 85.5)


# ============================================================================
# Prometheus-available path tests (if prometheus_client IS installed)
# ============================================================================


@pytest.mark.skipif(
    not metrics_mod.PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestMetricsWithPrometheus:
    """If prometheus_client is available, verify metrics are real objects."""

    def test_templates_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_templates_total, Counter)

    def test_distributions_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_distributions_total, Counter)

    def test_responses_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_responses_total, Counter)

    def test_validations_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_validations_total, Counter)

    def test_scores_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_scores_total, Counter)

    def test_followups_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_followups_total, Counter)

    def test_response_rate_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(metrics_mod.supplier_quest_response_rate, Gauge)

    def test_processing_duration_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(metrics_mod.supplier_quest_processing_duration_seconds, Histogram)

    def test_active_campaigns_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(metrics_mod.supplier_quest_active_campaigns, Gauge)

    def test_pending_responses_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(metrics_mod.supplier_quest_pending_responses, Gauge)

    def test_processing_errors_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(metrics_mod.supplier_quest_processing_errors_total, Counter)

    def test_data_quality_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(metrics_mod.supplier_quest_data_quality_score, Histogram)

    def test_templates_total_label_names(self):
        assert metrics_mod.supplier_quest_templates_total._labelnames == ("framework", "status")

    def test_distributions_total_label_names(self):
        assert metrics_mod.supplier_quest_distributions_total._labelnames == ("channel", "status")

    def test_processing_duration_label_names(self):
        assert metrics_mod.supplier_quest_processing_duration_seconds._labelnames == ("operation",)

    def test_processing_errors_label_names(self):
        assert metrics_mod.supplier_quest_processing_errors_total._labelnames == ("engine", "error_type")


# ============================================================================
# Prometheus-unavailable path tests (simulated)
# ============================================================================


class TestMetricsWithoutPrometheus:
    """Simulate prometheus_client absence and verify graceful fallback."""

    def test_record_template_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            # Should not raise
            metrics_mod.record_template("custom", "created")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_distribution_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_distribution("email", "sent")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_response_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_response("api", "submitted")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_validation_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_validation("structural", "pass")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_score_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_score("custom", "developing")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_followup_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_followup("reminder", "scheduled")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_update_response_rate_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.update_response_rate("camp-1", 50.0)
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_processing_duration_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_processing_duration("score", 0.5)
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_update_active_campaigns_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.update_active_campaigns(1)
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_update_pending_responses_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.update_pending_responses(-2)
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_processing_error_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_processing_error("validation", "timeout")
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original

    def test_record_data_quality_noop_when_unavailable(self):
        original = metrics_mod.PROMETHEUS_AVAILABLE
        try:
            metrics_mod.PROMETHEUS_AVAILABLE = False
            metrics_mod.record_data_quality("tcfd", 72.0)
        finally:
            metrics_mod.PROMETHEUS_AVAILABLE = original


# ============================================================================
# __all__ export tests
# ============================================================================


class TestMetricsExports:
    def test_all_exports_defined(self):
        expected = [
            "PROMETHEUS_AVAILABLE",
            "supplier_quest_templates_total",
            "supplier_quest_distributions_total",
            "supplier_quest_responses_total",
            "supplier_quest_validations_total",
            "supplier_quest_scores_total",
            "supplier_quest_followups_total",
            "supplier_quest_response_rate",
            "supplier_quest_processing_duration_seconds",
            "supplier_quest_active_campaigns",
            "supplier_quest_pending_responses",
            "supplier_quest_processing_errors_total",
            "supplier_quest_data_quality_score",
            "record_template",
            "record_distribution",
            "record_response",
            "record_validation",
            "record_score",
            "record_followup",
            "update_response_rate",
            "record_processing_duration",
            "update_active_campaigns",
            "update_pending_responses",
            "record_processing_error",
            "record_data_quality",
        ]
        for name in expected:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"

    def test_all_count_matches(self):
        # 1 flag + 12 metrics + 12 helpers = 25
        assert len(metrics_mod.__all__) == 25
