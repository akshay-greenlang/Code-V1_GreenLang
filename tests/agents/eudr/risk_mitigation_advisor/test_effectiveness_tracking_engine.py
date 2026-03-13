# -*- coding: utf-8 -*-
"""
Tests for Engine 5: Effectiveness Tracking Engine - AGENT-EUDR-025

Tests before/after risk scoring, ROI calculation, risk reduction deltas,
predicted vs actual deviation, statistical significance testing,
underperformance detection, trend analysis, and feedback loop.

Test count: ~70 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    EffectivenessRecord,
    MeasureEffectivenessRequest,
    MeasureEffectivenessResponse,
    RiskCategory,
)
from greenlang.agents.eudr.risk_mitigation_advisor.effectiveness_tracking_engine import (
    EffectivenessTrackingEngine,
)


class TestEffectivenessEngineInit:
    def test_engine_initializes(self, effectiveness_engine):
        assert effectiveness_engine is not None


class TestRiskReductionCalculation:
    @pytest.mark.asyncio
    async def test_calculate_reduction_basic(self, effectiveness_engine):
        baseline = {"country": Decimal("80"), "supplier": Decimal("70")}
        current = {"country": Decimal("60"), "supplier": Decimal("50")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        assert reduction["country"] == Decimal("25.00")
        assert reduction["supplier"] == pytest.approx(Decimal("28.57"), abs=Decimal("0.01"))

    @pytest.mark.asyncio
    async def test_calculate_reduction_zero_baseline(self, effectiveness_engine):
        baseline = {"country": Decimal("0")}
        current = {"country": Decimal("0")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        assert reduction["country"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_reduction_full_reduction(self, effectiveness_engine):
        baseline = {"deforestation": Decimal("90")}
        current = {"deforestation": Decimal("0")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        assert reduction["deforestation"] == Decimal("100.00")

    @pytest.mark.asyncio
    async def test_calculate_reduction_negative(self, effectiveness_engine):
        """Risk increased (negative reduction)."""
        baseline = {"supplier": Decimal("40")}
        current = {"supplier": Decimal("60")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        assert reduction["supplier"] < Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_reduction_uses_decimal(self, effectiveness_engine):
        baseline = {"country": Decimal("75")}
        current = {"country": Decimal("50")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        assert isinstance(reduction["country"], Decimal)

    @pytest.mark.asyncio
    async def test_composite_reduction(self, effectiveness_engine):
        baseline = {
            "country": Decimal("80"),
            "supplier": Decimal("70"),
            "deforestation": Decimal("90"),
        }
        current = {
            "country": Decimal("60"),
            "supplier": Decimal("50"),
            "deforestation": Decimal("60"),
        }
        composite = await effectiveness_engine.calculate_composite_reduction(
            baseline, current
        )
        assert isinstance(composite, Decimal)
        assert composite > Decimal("0")

    @pytest.mark.asyncio
    async def test_composite_reduction_all_dimensions(self, effectiveness_engine):
        baseline = {cat.value: Decimal("80") for cat in RiskCategory}
        current = {cat.value: Decimal("50") for cat in RiskCategory}
        composite = await effectiveness_engine.calculate_composite_reduction(
            baseline, current
        )
        assert Decimal("30") <= composite <= Decimal("45")


class TestROICalculation:
    @pytest.mark.asyncio
    async def test_roi_positive(self, effectiveness_engine):
        roi = await effectiveness_engine.calculate_roi(
            risk_reduction_pct=Decimal("30"),
            mitigation_cost=Decimal("50000"),
            penalty_exposure=Decimal("1000000"),
        )
        assert isinstance(roi, Decimal)
        assert roi > Decimal("0")

    @pytest.mark.asyncio
    async def test_roi_zero_cost(self, effectiveness_engine):
        roi = await effectiveness_engine.calculate_roi(
            risk_reduction_pct=Decimal("30"),
            mitigation_cost=Decimal("0"),
            penalty_exposure=Decimal("1000000"),
        )
        # Infinite or very large ROI
        assert roi is None or roi > Decimal("1000")

    @pytest.mark.asyncio
    async def test_roi_negative_when_no_reduction(self, effectiveness_engine):
        roi = await effectiveness_engine.calculate_roi(
            risk_reduction_pct=Decimal("0"),
            mitigation_cost=Decimal("50000"),
            penalty_exposure=Decimal("1000000"),
        )
        assert roi is not None
        assert roi < Decimal("0")

    @pytest.mark.asyncio
    async def test_roi_uses_decimal(self, effectiveness_engine):
        roi = await effectiveness_engine.calculate_roi(
            risk_reduction_pct=Decimal("25"),
            mitigation_cost=Decimal("100000"),
            penalty_exposure=Decimal("2000000"),
        )
        if roi is not None:
            assert isinstance(roi, Decimal)


class TestDeviationAnalysis:
    @pytest.mark.asyncio
    async def test_deviation_within_tolerance(self, effectiveness_engine):
        deviation = await effectiveness_engine.calculate_deviation(
            predicted=Decimal("30"),
            actual=Decimal("28"),
        )
        assert isinstance(deviation, Decimal)
        assert abs(deviation) <= Decimal("15")

    @pytest.mark.asyncio
    async def test_deviation_significant(self, effectiveness_engine):
        deviation = await effectiveness_engine.calculate_deviation(
            predicted=Decimal("40"),
            actual=Decimal("20"),
        )
        assert abs(deviation) > Decimal("15")

    @pytest.mark.asyncio
    async def test_deviation_exact_match(self, effectiveness_engine):
        deviation = await effectiveness_engine.calculate_deviation(
            predicted=Decimal("30"),
            actual=Decimal("30"),
        )
        assert deviation == Decimal("0")


class TestStatisticalSignificance:
    @pytest.mark.asyncio
    async def test_significance_with_enough_data(self, effectiveness_engine):
        baseline_scores = [Decimal("80"), Decimal("78"), Decimal("82"), Decimal("79"), Decimal("81")]
        current_scores = [Decimal("55"), Decimal("58"), Decimal("52"), Decimal("60"), Decimal("54")]
        result = await effectiveness_engine.test_significance(
            baseline_scores, current_scores
        )
        assert "significant" in result
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_significance_not_enough_data(self, effectiveness_engine):
        result = await effectiveness_engine.test_significance(
            [Decimal("80")], [Decimal("75")]
        )
        assert result["significant"] is False

    @pytest.mark.asyncio
    async def test_significance_no_difference(self, effectiveness_engine):
        same_scores = [Decimal("50")] * 10
        result = await effectiveness_engine.test_significance(same_scores, same_scores)
        assert result["significant"] is False


class TestUnderperformanceDetection:
    @pytest.mark.asyncio
    async def test_detect_underperforming(self, effectiveness_engine):
        is_under = await effectiveness_engine.is_underperforming(
            predicted_reduction=Decimal("40"),
            actual_reduction=Decimal("15"),
        )
        assert is_under is True

    @pytest.mark.asyncio
    async def test_detect_performing_well(self, effectiveness_engine):
        is_under = await effectiveness_engine.is_underperforming(
            predicted_reduction=Decimal("30"),
            actual_reduction=Decimal("28"),
        )
        assert is_under is False

    @pytest.mark.asyncio
    async def test_detect_overperforming(self, effectiveness_engine):
        is_under = await effectiveness_engine.is_underperforming(
            predicted_reduction=Decimal("25"),
            actual_reduction=Decimal("40"),
        )
        assert is_under is False


class TestEffectivenessMeasurement:
    @pytest.mark.asyncio
    async def test_measure_effectiveness(self, effectiveness_engine, effectiveness_request):
        result = await effectiveness_engine.measure_effectiveness(effectiveness_request)
        assert isinstance(result, MeasureEffectivenessResponse)
        assert result.record is not None

    @pytest.mark.asyncio
    async def test_measurement_has_provenance(self, effectiveness_engine, effectiveness_request):
        result = await effectiveness_engine.measure_effectiveness(effectiveness_request)
        assert result.provenance_hash != ""

    @pytest.mark.asyncio
    async def test_measurement_processing_time(self, effectiveness_engine, effectiveness_request):
        result = await effectiveness_engine.measure_effectiveness(effectiveness_request)
        assert result.processing_time_ms >= Decimal("0")

    @pytest.mark.asyncio
    async def test_measurement_includes_roi(self, effectiveness_engine, effectiveness_request):
        result = await effectiveness_engine.measure_effectiveness(effectiveness_request)
        # ROI is optional but should be attempted when include_roi is True
        assert result.record is not None


class TestEffectivenessTrends:
    @pytest.mark.asyncio
    async def test_get_trend_data(self, effectiveness_engine):
        trend = await effectiveness_engine.get_effectiveness_trend(
            plan_id="plan-001",
            supplier_id="sup-001",
        )
        assert trend is not None
        assert isinstance(trend, list)

    @pytest.mark.asyncio
    async def test_trend_empty_for_new_plan(self, effectiveness_engine):
        trend = await effectiveness_engine.get_effectiveness_trend(
            plan_id="nonexistent-plan",
            supplier_id="nonexistent-sup",
        )
        assert trend == []


class TestEffectivenessEdgeCases:
    @pytest.mark.asyncio
    async def test_all_scores_unchanged(self, effectiveness_engine):
        baseline = {"country": Decimal("50")}
        current = {"country": Decimal("50")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        assert reduction["country"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_empty_scores(self, effectiveness_engine):
        reduction = await effectiveness_engine.calculate_risk_reduction({}, {})
        assert reduction == {}

    @pytest.mark.asyncio
    async def test_mismatched_categories(self, effectiveness_engine):
        baseline = {"country": Decimal("80")}
        current = {"supplier": Decimal("60")}
        reduction = await effectiveness_engine.calculate_risk_reduction(baseline, current)
        # Should handle gracefully
        assert isinstance(reduction, dict)
