# -*- coding: utf-8 -*-
"""
Unit tests for YearComparatorEngine - AGENT-EUDR-034

Tests multi-year comparison logic, metric snapshots, trend analysis,
dimension comparison, compliance rate tracking, risk score evolution,
statistical calculations, and report generation.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 5: Year Comparator)
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.year_comparator import (
    YearComparatorEngine,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    ComparisonDimension,
    ComparisonMetric,
    ComparisonResult,
    EUDRCommodity,
    YearComparison,
    YearComparisonStatus,
    YearMetricSnapshot,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def comparator(config):
    return YearComparatorEngine(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def snapshot_2024():
    return YearMetricSnapshot(
        snapshot_id="snap-2024-001",
        operator_id="operator-001",
        year=2024,
        commodity=EUDRCommodity.COFFEE,
        total_suppliers=10,
        compliant_suppliers=7,
        compliance_rate=Decimal("70.00"),
        average_risk_score=Decimal("42.00"),
        total_shipments=80,
        deforestation_free_rate=Decimal("92.50"),
        dds_submitted=75,
        dds_approved=70,
        audit_findings=5,
        remediation_actions=4,
        provenance_hash="x" * 64,
    )


@pytest.fixture
def snapshot_2025():
    return YearMetricSnapshot(
        snapshot_id="snap-2025-001",
        operator_id="operator-001",
        year=2025,
        commodity=EUDRCommodity.COFFEE,
        total_suppliers=12,
        compliant_suppliers=10,
        compliance_rate=Decimal("83.33"),
        average_risk_score=Decimal("35.20"),
        total_shipments=95,
        deforestation_free_rate=Decimal("97.50"),
        dds_submitted=90,
        dds_approved=85,
        audit_findings=3,
        remediation_actions=2,
        provenance_hash="y" * 64,
    )


@pytest.fixture
def snapshot_2026():
    return YearMetricSnapshot(
        snapshot_id="snap-2026-001",
        operator_id="operator-001",
        year=2026,
        commodity=EUDRCommodity.COFFEE,
        total_suppliers=15,
        compliant_suppliers=14,
        compliance_rate=Decimal("93.33"),
        average_risk_score=Decimal("28.50"),
        total_shipments=120,
        deforestation_free_rate=Decimal("99.10"),
        dds_submitted=118,
        dds_approved=115,
        audit_findings=1,
        remediation_actions=1,
        provenance_hash="z" * 64,
    )


# ---------------------------------------------------------------------------
# Snapshot Registration
# ---------------------------------------------------------------------------

class TestSnapshotRegistration:
    """Test year metric snapshot registration."""

    @pytest.mark.asyncio
    async def test_register_snapshot(self, comparator, snapshot_2025):
        result = await comparator.register_snapshot(snapshot_2025)
        assert result is True

    @pytest.mark.asyncio
    async def test_register_multiple_snapshots(self, comparator, snapshot_2024, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2024)
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        snapshots = await comparator.list_snapshots(operator_id="operator-001")
        assert len(snapshots) == 3

    @pytest.mark.asyncio
    async def test_get_snapshot_by_year(self, comparator, snapshot_2025):
        await comparator.register_snapshot(snapshot_2025)
        retrieved = await comparator.get_snapshot(
            operator_id="operator-001", year=2025, commodity=EUDRCommodity.COFFEE,
        )
        assert retrieved.year == 2025

    @pytest.mark.asyncio
    async def test_get_nonexistent_snapshot_raises(self, comparator):
        with pytest.raises(ValueError, match="not found"):
            await comparator.get_snapshot(
                operator_id="operator-999", year=2025,
                commodity=EUDRCommodity.COFFEE,
            )

    @pytest.mark.asyncio
    async def test_update_existing_snapshot(self, comparator, snapshot_2025):
        await comparator.register_snapshot(snapshot_2025)
        snapshot_2025.compliance_rate = Decimal("85.00")
        await comparator.register_snapshot(snapshot_2025)
        retrieved = await comparator.get_snapshot(
            operator_id="operator-001", year=2025, commodity=EUDRCommodity.COFFEE,
        )
        assert retrieved.compliance_rate == Decimal("85.00")


# ---------------------------------------------------------------------------
# Two-Year Comparison
# ---------------------------------------------------------------------------

class TestTwoYearComparison:
    """Test year-over-year comparison."""

    @pytest.mark.asyncio
    async def test_compare_two_years(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        assert isinstance(comparison, YearComparison)
        assert comparison.base_year == 2025
        assert comparison.compare_year == 2026

    @pytest.mark.asyncio
    async def test_comparison_has_metrics(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        assert len(comparison.metrics) > 0

    @pytest.mark.asyncio
    async def test_comparison_status_completed(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        assert comparison.status == YearComparisonStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_comparison_provenance_hash(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        assert len(comparison.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_comparison_detects_improving_trend(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        assert comparison.overall_trend == "improving"

    @pytest.mark.asyncio
    async def test_comparison_missing_base_year_raises(self, comparator, snapshot_2026):
        await comparator.register_snapshot(snapshot_2026)
        with pytest.raises(ValueError, match="not found"):
            await comparator.compare_years(
                operator_id="operator-001",
                commodity=EUDRCommodity.COFFEE,
                base_year=2020,
                compare_year=2026,
            )

    @pytest.mark.asyncio
    async def test_compare_same_year_raises(self, comparator, snapshot_2025):
        await comparator.register_snapshot(snapshot_2025)
        with pytest.raises(ValueError, match="same year"):
            await comparator.compare_years(
                operator_id="operator-001",
                commodity=EUDRCommodity.COFFEE,
                base_year=2025,
                compare_year=2025,
            )


# ---------------------------------------------------------------------------
# Metric Calculations
# ---------------------------------------------------------------------------

class TestMetricCalculations:
    """Test individual metric comparison calculations."""

    @pytest.mark.asyncio
    async def test_compliance_rate_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        compliance_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.COMPLIANCE_RATE),
            None,
        )
        assert compliance_metric is not None
        assert compliance_metric.change > Decimal("0")

    @pytest.mark.asyncio
    async def test_risk_score_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        risk_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.RISK_SCORE),
            None,
        )
        assert risk_metric is not None
        # Risk score decreased = improvement
        assert risk_metric.change < Decimal("0")

    @pytest.mark.asyncio
    async def test_supplier_count_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        supplier_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.SUPPLIER_COUNT),
            None,
        )
        assert supplier_metric is not None
        assert supplier_metric.change > Decimal("0")

    @pytest.mark.asyncio
    async def test_deforestation_rate_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        def_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.DEFORESTATION_RATE),
            None,
        )
        assert def_metric is not None

    @pytest.mark.asyncio
    async def test_audit_findings_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        audit_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.AUDIT_FINDINGS),
            None,
        )
        assert audit_metric is not None
        # Findings decreased = improvement
        assert audit_metric.change < Decimal("0")

    @pytest.mark.asyncio
    async def test_dds_approval_rate_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        dds_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.DDS_APPROVAL_RATE),
            None,
        )
        assert dds_metric is not None


# ---------------------------------------------------------------------------
# Multi-Year Trend
# ---------------------------------------------------------------------------

class TestMultiYearTrend:
    """Test multi-year trend analysis."""

    @pytest.mark.asyncio
    async def test_three_year_trend(self, comparator, snapshot_2024, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2024)
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        trend = await comparator.analyze_multi_year_trend(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            years=[2024, 2025, 2026],
        )
        assert trend is not None
        assert trend["overall_direction"] in ("improving", "stable", "declining")

    @pytest.mark.asyncio
    async def test_trend_with_single_year_raises(self, comparator, snapshot_2025):
        await comparator.register_snapshot(snapshot_2025)
        with pytest.raises(ValueError, match="at least two years"):
            await comparator.analyze_multi_year_trend(
                operator_id="operator-001",
                commodity=EUDRCommodity.COFFEE,
                years=[2025],
            )

    @pytest.mark.asyncio
    async def test_trend_includes_per_year_metrics(self, comparator, snapshot_2024, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2024)
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        trend = await comparator.analyze_multi_year_trend(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            years=[2024, 2025, 2026],
        )
        assert "yearly_data" in trend
        assert len(trend["yearly_data"]) == 3

    @pytest.mark.asyncio
    async def test_trend_compliance_rate_improving(self, comparator, snapshot_2024, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2024)
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        trend = await comparator.analyze_multi_year_trend(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            years=[2024, 2025, 2026],
        )
        rates = [y["compliance_rate"] for y in trend["yearly_data"]]
        assert rates == sorted(rates)  # Ascending = improving


# ---------------------------------------------------------------------------
# Percentage Change
# ---------------------------------------------------------------------------

class TestPercentageChange:
    """Test percentage change calculations."""

    @pytest.mark.asyncio
    async def test_positive_change(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        compliance_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.COMPLIANCE_RATE),
            None,
        )
        assert compliance_metric is not None
        assert compliance_metric.percentage_change > Decimal("0")

    @pytest.mark.asyncio
    async def test_negative_change_risk_score(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        risk_metric = next(
            (m for m in comparison.metrics
             if m.dimension == ComparisonDimension.RISK_SCORE),
            None,
        )
        assert risk_metric is not None
        assert risk_metric.percentage_change < Decimal("0")

    @pytest.mark.asyncio
    async def test_zero_base_value_handles_division(self, comparator):
        snap_zero = YearMetricSnapshot(
            snapshot_id="snap-zero",
            operator_id="operator-002",
            year=2024,
            commodity=EUDRCommodity.WOOD,
            audit_findings=0,
            provenance_hash="0" * 64,
        )
        snap_nonzero = YearMetricSnapshot(
            snapshot_id="snap-nonzero",
            operator_id="operator-002",
            year=2025,
            commodity=EUDRCommodity.WOOD,
            audit_findings=3,
            provenance_hash="1" * 64,
        )
        await comparator.register_snapshot(snap_zero)
        await comparator.register_snapshot(snap_nonzero)
        comparison = await comparator.compare_years(
            operator_id="operator-002",
            commodity=EUDRCommodity.WOOD,
            base_year=2024,
            compare_year=2025,
        )
        assert comparison.status == YearComparisonStatus.COMPLETED


# ---------------------------------------------------------------------------
# List Comparisons
# ---------------------------------------------------------------------------

class TestListComparisons:
    """Test comparison listing."""

    @pytest.mark.asyncio
    async def test_list_comparisons_by_operator(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        comparisons = await comparator.list_comparisons(operator_id="operator-001")
        assert len(comparisons) >= 1

    @pytest.mark.asyncio
    async def test_list_comparisons_by_commodity(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        comparisons = await comparator.list_comparisons(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
        )
        assert len(comparisons) >= 1

    @pytest.mark.asyncio
    async def test_get_comparison_by_id(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        comparison = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        retrieved = await comparator.get_comparison(comparison.comparison_id)
        assert retrieved.comparison_id == comparison.comparison_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_comparison_raises(self, comparator):
        with pytest.raises(ValueError, match="not found"):
            await comparator.get_comparison("cmp-nonexistent")


# ---------------------------------------------------------------------------
# Deterministic Comparison
# ---------------------------------------------------------------------------

class TestDeterministicComparison:
    """Test that comparisons are deterministic."""

    @pytest.mark.asyncio
    async def test_same_input_same_provenance(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        c1 = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        c2 = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        assert c1.provenance_hash == c2.provenance_hash

    @pytest.mark.asyncio
    async def test_same_input_same_metrics(self, comparator, snapshot_2025, snapshot_2026):
        await comparator.register_snapshot(snapshot_2025)
        await comparator.register_snapshot(snapshot_2026)
        c1 = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        c2 = await comparator.compare_years(
            operator_id="operator-001",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
        )
        for m1, m2 in zip(c1.metrics, c2.metrics):
            assert m1.change == m2.change
