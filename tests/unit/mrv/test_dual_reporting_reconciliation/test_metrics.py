# -*- coding: utf-8 -*-
"""
Unit tests for Dual Reporting Reconciliation metrics.

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 20 tests covering DualReportingReconciliationMetrics singleton.
"""

from __future__ import annotations

import pytest

from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
    DualReportingReconciliationMetrics,
    get_metrics,
)


# ===========================================================================
# 1. Singleton Tests
# ===========================================================================


class TestMetricsSingleton:
    """Test DualReportingReconciliationMetrics singleton pattern."""

    def test_get_metrics_returns_instance(self):
        m = get_metrics()
        assert m is not None

    def test_singleton_returns_same_instance(self):
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_metrics_is_correct_type(self):
        m = get_metrics()
        assert isinstance(m, DualReportingReconciliationMetrics)


# ===========================================================================
# 2. Recording Tests
# ===========================================================================


class TestMetricsRecording:
    """Test metrics recording methods."""

    def test_record_reconciliation(self):
        m = get_metrics()
        m.record_reconciliation(
            tenant_id="tenant-001",
            status="completed",
            energy_type="electricity",
            duration_s=0.5,
            discrepancy_pct=12.5,
            pif=0.44,
        )

    def test_record_discrepancy(self):
        m = get_metrics()
        m.record_discrepancy(
            discrepancy_type="rec_go_impact",
            materiality="material",
            direction="market_lower",
        )

    def test_record_quality_score(self):
        m = get_metrics()
        m.record_quality_score(
            dimension="completeness",
            score=0.85,
            grade="B",
        )

    def test_record_report_generated(self):
        m = get_metrics()
        m.record_report_generated(framework="ghg_protocol")

    def test_record_trend_analysis(self):
        m = get_metrics()
        m.record_trend_analysis(tenant_id="tenant-001")

    def test_record_compliance_check(self):
        m = get_metrics()
        m.record_compliance_check(framework="csrd_esrs", status="compliant")

    def test_record_batch(self):
        m = get_metrics()
        m.record_batch(tenant_id="tenant-001", batch_size=10)

    def test_record_error(self):
        m = get_metrics()
        m.record_error(error_type="validation_failure", operation="reconcile")


# ===========================================================================
# 3. Statistics Tests
# ===========================================================================


class TestMetricsStatistics:
    """Test metrics statistics retrieval."""

    def test_get_metrics_summary(self):
        m = get_metrics()
        summary = m.get_metrics_summary()
        assert isinstance(summary, dict)

    def test_reset_stats(self):
        m = get_metrics()
        m.record_reconciliation(
            tenant_id="tenant-001",
            status="completed",
            energy_type="electricity",
            duration_s=1.0,
            discrepancy_pct=5.0,
        )
        m.reset_stats()
        summary = m.get_metrics_summary()
        assert isinstance(summary, dict)
