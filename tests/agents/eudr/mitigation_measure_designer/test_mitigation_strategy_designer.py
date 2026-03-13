# -*- coding: utf-8 -*-
"""
Unit tests for MitigationStrategyDesigner - AGENT-EUDR-029

Tests strategy design, dimension identification, measure selection,
cumulative reduction, feasibility validation, priority assignment,
and timeline calculation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.mitigation_strategy_designer import (
    MitigationStrategyDesigner,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    Article11Category,
    EUDRCommodity,
    MeasurePriority,
    MeasureStatus,
    MeasureTemplate,
    MitigationMeasure,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    RiskTrigger,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return MitigationMeasureDesignerConfig()


@pytest.fixture
def designer(config):
    return MitigationStrategyDesigner(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def templates_for_supplier() -> List[MeasureTemplate]:
    """Templates applicable to supplier dimension."""
    return [
        MeasureTemplate(
            template_id="T-SUP-001",
            title="Supplier Audit",
            article11_category=Article11Category.INDEPENDENT_AUDIT,
            applicable_dimensions=[RiskDimension.SUPPLIER],
            base_effectiveness=Decimal("30"),
        ),
        MeasureTemplate(
            template_id="T-SUP-002",
            title="Supplier Questionnaire",
            article11_category=Article11Category.ADDITIONAL_INFO,
            applicable_dimensions=[RiskDimension.SUPPLIER],
            base_effectiveness=Decimal("18"),
        ),
        MeasureTemplate(
            template_id="T-SUP-003",
            title="Supplier Training",
            article11_category=Article11Category.OTHER_MEASURES,
            applicable_dimensions=[RiskDimension.SUPPLIER],
            base_effectiveness=Decimal("22"),
        ),
    ]


@pytest.fixture
def templates_mixed() -> List[MeasureTemplate]:
    """Templates spanning multiple dimensions."""
    return [
        MeasureTemplate(
            template_id="T-01",
            title="Country Assessment",
            article11_category=Article11Category.INDEPENDENT_AUDIT,
            applicable_dimensions=[RiskDimension.COUNTRY],
            base_effectiveness=Decimal("20"),
        ),
        MeasureTemplate(
            template_id="T-02",
            title="Supplier Audit",
            article11_category=Article11Category.INDEPENDENT_AUDIT,
            applicable_dimensions=[RiskDimension.SUPPLIER],
            base_effectiveness=Decimal("30"),
        ),
        MeasureTemplate(
            template_id="T-03",
            title="Satellite Monitoring",
            article11_category=Article11Category.OTHER_MEASURES,
            applicable_dimensions=[RiskDimension.DEFORESTATION],
            base_effectiveness=Decimal("28"),
        ),
        MeasureTemplate(
            template_id="T-04",
            title="Corruption Audit",
            article11_category=Article11Category.INDEPENDENT_AUDIT,
            applicable_dimensions=[RiskDimension.CORRUPTION],
            base_effectiveness=Decimal("25"),
        ),
    ]


class TestDesignStrategy:
    """Test design_strategy with various risk triggers."""

    @pytest.mark.asyncio
    async def test_design_returns_strategy(self, designer, sample_risk_trigger, templates_mixed):
        strategy = await designer.design_strategy(sample_risk_trigger, templates_mixed)
        assert isinstance(strategy, MitigationStrategy)
        assert strategy.strategy_id.startswith("stg-")
        assert strategy.pre_mitigation_score == Decimal("72")
        assert strategy.target_score == Decimal("30")

    @pytest.mark.asyncio
    async def test_design_strategy_has_measures(self, designer, sample_risk_trigger, templates_mixed):
        strategy = await designer.design_strategy(sample_risk_trigger, templates_mixed)
        assert len(strategy.measures) > 0

    @pytest.mark.asyncio
    async def test_design_strategy_has_provenance_hash(self, designer, sample_risk_trigger, templates_mixed):
        strategy = await designer.design_strategy(sample_risk_trigger, templates_mixed)
        assert len(strategy.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_design_strategy_all_measures_proposed(self, designer, sample_risk_trigger, templates_mixed):
        strategy = await designer.design_strategy(sample_risk_trigger, templates_mixed)
        for m in strategy.measures:
            assert m.status == MeasureStatus.PROPOSED


class TestIdentifyElevatedDimensions:
    """Test _identify_elevated_dimensions."""

    def test_identifies_dimensions_above_threshold(self, designer, sample_risk_trigger):
        elevated = designer._identify_elevated_dimensions(sample_risk_trigger)
        dim_names = [d.value for d, _ in elevated]
        # Supplier(80), Country(65), Deforestation(55), Corruption(45) all > 30
        assert "supplier" in dim_names
        assert "country" in dim_names
        assert "deforestation" in dim_names
        assert "corruption" in dim_names
        # Commodity(20) should NOT be elevated
        assert "commodity" not in dim_names

    def test_no_elevated_when_all_below(self, designer):
        trigger = RiskTrigger(
            assessment_id="a",
            operator_id="o",
            commodity=EUDRCommodity.WOOD,
            composite_score=Decimal("15"),
            risk_level=RiskLevel.NEGLIGIBLE,
            risk_dimensions={
                RiskDimension.COUNTRY: Decimal("10"),
                RiskDimension.SUPPLIER: Decimal("20"),
            },
        )
        elevated = designer._identify_elevated_dimensions(trigger)
        assert len(elevated) == 0

    def test_all_elevated(self, designer):
        trigger = RiskTrigger(
            assessment_id="a",
            operator_id="o",
            commodity=EUDRCommodity.SOYA,
            composite_score=Decimal("90"),
            risk_level=RiskLevel.CRITICAL,
            risk_dimensions={
                dim: Decimal("85") for dim in RiskDimension
            },
        )
        elevated = designer._identify_elevated_dimensions(trigger)
        assert len(elevated) == len(RiskDimension)

    def test_sorted_by_weighted_impact_descending(self, designer, sample_risk_trigger):
        elevated = designer._identify_elevated_dimensions(sample_risk_trigger)
        if len(elevated) >= 2:
            from greenlang.agents.eudr.mitigation_measure_designer.models import DEFAULT_RISK_WEIGHTS
            weighted = [
                s * DEFAULT_RISK_WEIGHTS.get(d, Decimal("0.10"))
                for d, s in elevated
            ]
            for i in range(len(weighted) - 1):
                assert weighted[i] >= weighted[i + 1]


class TestMeasureSelection:
    """Test measure selection per dimension."""

    def test_selects_measures_for_dimension(self, designer, templates_for_supplier):
        measures = designer._select_measures_for_dimension(
            dimension=RiskDimension.SUPPLIER,
            score=Decimal("80"),
            commodity=EUDRCommodity.COFFEE,
            templates=templates_for_supplier,
        )
        assert len(measures) > 0
        for m in measures:
            assert m.target_dimension == RiskDimension.SUPPLIER

    def test_high_score_selects_more_measures(self, designer, templates_for_supplier):
        measures_high = designer._select_measures_for_dimension(
            dimension=RiskDimension.SUPPLIER,
            score=Decimal("80"),
            commodity=EUDRCommodity.COFFEE,
            templates=templates_for_supplier,
        )
        measures_low = designer._select_measures_for_dimension(
            dimension=RiskDimension.SUPPLIER,
            score=Decimal("40"),
            commodity=EUDRCommodity.COFFEE,
            templates=templates_for_supplier,
        )
        # Score >= 60 -> max 3, Score < 60 -> max 2
        assert len(measures_high) == 3
        assert len(measures_low) == 2

    def test_creates_generic_when_no_templates(self, designer):
        measures = designer._select_measures_for_dimension(
            dimension=RiskDimension.COUNTRY,
            score=Decimal("70"),
            commodity=EUDRCommodity.COFFEE,
            templates=[],  # No templates
        )
        assert len(measures) == 1
        assert measures[0].article11_category == Article11Category.OTHER_MEASURES


class TestCumulativeReduction:
    """Test cumulative reduction with diminishing returns."""

    def test_empty_measures_returns_zero(self, designer):
        result = designer._estimate_cumulative_reduction([])
        assert result == Decimal("0")

    def test_single_measure_reduction(self, designer):
        measures = [
            MitigationMeasure(
                measure_id="m1", strategy_id="s1", title="T",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                expected_risk_reduction=Decimal("30"),
            ),
        ]
        result = designer._estimate_cumulative_reduction(measures)
        assert result == Decimal("30.00")

    def test_diminishing_returns(self, designer):
        """Three 30% measures should NOT give 90% total."""
        measures = [
            MitigationMeasure(
                measure_id=f"m{i}", strategy_id="s1", title=f"T{i}",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                expected_risk_reduction=Decimal("30"),
            )
            for i in range(3)
        ]
        result = designer._estimate_cumulative_reduction(measures)
        # 1 - (0.7 * 0.7 * 0.7) = 1 - 0.343 = 0.657 -> 65.70%
        assert result == Decimal("65.70")
        assert result < Decimal("90")

    def test_capped_at_max_effectiveness(self, designer):
        """Reduction should not exceed max_effectiveness_cap (80)."""
        measures = [
            MitigationMeasure(
                measure_id=f"m{i}", strategy_id="s1", title=f"T{i}",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                expected_risk_reduction=Decimal("50"),
            )
            for i in range(5)
        ]
        result = designer._estimate_cumulative_reduction(measures)
        assert result <= Decimal("80")


class TestFeasibilityValidation:
    """Test strategy feasibility validation."""

    def test_feasible_when_reduction_sufficient(self, designer):
        result = designer._validate_strategy_feasibility_score(
            pre_score=Decimal("70"),
            target_score=Decimal("30"),
            estimated_reduction=Decimal("60"),
        )
        assert result is True

    def test_not_feasible_when_reduction_insufficient(self, designer):
        result = designer._validate_strategy_feasibility_score(
            pre_score=Decimal("70"),
            target_score=Decimal("30"),
            estimated_reduction=Decimal("10"),
        )
        assert result is False

    def test_feasible_when_already_at_target(self, designer):
        result = designer._validate_strategy_feasibility_score(
            pre_score=Decimal("25"),
            target_score=Decimal("30"),
            estimated_reduction=Decimal("0"),
        )
        assert result is True


class TestPriorityAssignment:
    """Test priority assignment based on risk scores."""

    def test_critical_priority_for_score_80_plus(self, designer, sample_risk_trigger):
        measures = [
            MitigationMeasure(
                measure_id="m1", strategy_id="s1", title="T",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.SUPPLIER,  # Score = 80
                expected_risk_reduction=Decimal("25"),
            ),
        ]
        result = designer._assign_priorities(measures, sample_risk_trigger)
        assert result[0].priority == MeasurePriority.CRITICAL

    def test_high_priority_for_score_60_to_79(self, designer, sample_risk_trigger):
        measures = [
            MitigationMeasure(
                measure_id="m1", strategy_id="s1", title="T",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,  # Score = 65
                expected_risk_reduction=Decimal("20"),
            ),
        ]
        result = designer._assign_priorities(measures, sample_risk_trigger)
        assert result[0].priority == MeasurePriority.HIGH


class TestTimelineCalculation:
    """Test timeline estimation."""

    def test_empty_measures_zero_timeline(self, designer):
        assert designer._calculate_timeline([]) == 0

    def test_single_priority_group(self, designer):
        measures = [
            MitigationMeasure(
                measure_id="m1", strategy_id="s1", title="T",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                priority=MeasurePriority.HIGH,
            ),
        ]
        timeline = designer._calculate_timeline(measures)
        assert timeline == 30  # default_deadline_days

    def test_multiple_priority_groups_sequential(self, designer):
        measures = [
            MitigationMeasure(
                measure_id="m1", strategy_id="s1", title="T1",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                priority=MeasurePriority.HIGH,
            ),
            MitigationMeasure(
                measure_id="m2", strategy_id="s1", title="T2",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.SUPPLIER,
                priority=MeasurePriority.MEDIUM,
            ),
        ]
        timeline = designer._calculate_timeline(measures)
        # 2 priority groups * 30 days each = 60
        assert timeline == 60
