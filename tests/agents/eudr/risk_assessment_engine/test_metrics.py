# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 Prometheus metrics helpers.

Each test verifies that the helper function can be called without raising
an exception, regardless of whether prometheus_client is installed.
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.risk_assessment_engine import metrics


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------


class TestCounterMetrics:
    """Verify counter metric helpers do not raise."""

    def test_record_assessment(self):
        metrics.record_assessment("cocoa", "completed")

    def test_record_factor_aggregation(self):
        metrics.record_factor_aggregation("country", "success")

    def test_record_benchmark_applied(self):
        metrics.record_benchmark_applied("BR", "high")

    def test_record_criteria_evaluation(self):
        metrics.record_criteria_evaluation("prevalence_of_deforestation", "pass")

    def test_record_classification(self):
        metrics.record_classification("standard")

    def test_record_report_generated(self):
        metrics.record_report_generated("cocoa")

    def test_record_override_applied(self):
        metrics.record_override_applied("expert_judgment")

    def test_record_api_error(self):
        metrics.record_api_error("assess_risk")


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------


class TestHistogramMetrics:
    """Verify histogram metric helpers do not raise."""

    def test_observe_assessment_duration(self):
        metrics.observe_assessment_duration("cocoa", 0.123)

    def test_observe_aggregation_duration(self):
        metrics.observe_aggregation_duration("country", 0.045)

    def test_observe_classification_duration(self):
        metrics.observe_classification_duration(0.010)

    def test_observe_report_generation_duration(self):
        metrics.observe_report_generation_duration("coffee", 1.5)

    def test_observe_criteria_evaluation_duration(self):
        metrics.observe_criteria_evaluation_duration(0.250)


# ---------------------------------------------------------------------------
# Gauge helpers
# ---------------------------------------------------------------------------


class TestGaugeMetrics:
    """Verify gauge metric helpers do not raise."""

    def test_set_active_assessments(self):
        metrics.set_active_assessments(5)

    def test_set_high_risk_operators(self):
        metrics.set_high_risk_operators(3)

    def test_set_average_risk_score(self):
        metrics.set_average_risk_score(42.5)

    def test_set_override_count(self):
        metrics.set_override_count(2)

    def test_set_trend_data_points(self):
        metrics.set_trend_data_points(100)
