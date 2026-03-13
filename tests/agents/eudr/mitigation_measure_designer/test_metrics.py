# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-029

Tests each of the 18 metric helper functions and graceful degradation
when prometheus_client is not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.agents.eudr.mitigation_measure_designer import metrics


class TestCounterMetrics:
    """Test counter metric helper functions."""

    def test_record_strategy_designed_no_error(self):
        """Calling record_strategy_designed should not raise."""
        metrics.record_strategy_designed("coffee", "high")

    def test_record_measure_proposed_no_error(self):
        metrics.record_measure_proposed("independent_audit", "supplier")

    def test_record_measure_approved_no_error(self):
        metrics.record_measure_approved("additional_info")

    def test_record_measure_completed_no_error(self):
        metrics.record_measure_completed("other_measures", "success")

    def test_record_verification_no_error(self):
        metrics.record_verification("sufficient")

    def test_record_report_generated_no_error(self):
        metrics.record_report_generated("wood")

    def test_record_workflow_closed_no_error(self):
        metrics.record_workflow_closed("cocoa", "success")

    def test_record_api_error_no_error(self):
        metrics.record_api_error("design_strategy")


class TestHistogramMetrics:
    """Test histogram metric helper functions."""

    def test_observe_strategy_design_duration_no_error(self):
        metrics.observe_strategy_design_duration("coffee", 0.5)

    def test_observe_effectiveness_estimation_duration_no_error(self):
        metrics.observe_effectiveness_estimation_duration(0.1)

    def test_observe_verification_duration_no_error(self):
        metrics.observe_verification_duration("soya", 1.0)

    def test_observe_report_generation_duration_no_error(self):
        metrics.observe_report_generation_duration("cattle", 0.25)

    def test_observe_workflow_duration_no_error(self):
        metrics.observe_workflow_duration("rubber", 3600.0)


class TestGaugeMetrics:
    """Test gauge metric helper functions."""

    def test_set_active_workflows_no_error(self):
        metrics.set_active_workflows(5)

    def test_set_overdue_measures_no_error(self):
        metrics.set_overdue_measures(2)

    def test_set_average_risk_reduction_no_error(self):
        metrics.set_average_risk_reduction(25.5)

    def test_set_pending_approvals_no_error(self):
        metrics.set_pending_approvals(3)

    def test_set_template_library_size_no_error(self):
        metrics.set_template_library_size(50)


class TestMetricsGracefulDegradation:
    """Test that metrics work when prometheus_client is not available."""

    def test_counter_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            # Should not raise when prometheus is unavailable
            metrics.record_strategy_designed("coffee", "high")
            metrics.record_measure_proposed("audit", "country")
            metrics.record_measure_approved("info")
            metrics.record_measure_completed("other", "ok")
            metrics.record_verification("sufficient")
            metrics.record_report_generated("wood")
            metrics.record_workflow_closed("soya", "success")
            metrics.record_api_error("test")
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_histogram_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.observe_strategy_design_duration("coffee", 0.5)
            metrics.observe_effectiveness_estimation_duration(0.1)
            metrics.observe_verification_duration("wood", 1.0)
            metrics.observe_report_generation_duration("soya", 0.3)
            metrics.observe_workflow_duration("cattle", 100.0)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original

    def test_gauge_when_prometheus_unavailable(self):
        original = metrics._PROMETHEUS_AVAILABLE
        try:
            metrics._PROMETHEUS_AVAILABLE = False
            metrics.set_active_workflows(10)
            metrics.set_overdue_measures(0)
            metrics.set_average_risk_reduction(30.0)
            metrics.set_pending_approvals(5)
            metrics.set_template_library_size(50)
        finally:
            metrics._PROMETHEUS_AVAILABLE = original
