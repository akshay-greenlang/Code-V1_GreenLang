# -*- coding: utf-8 -*-
"""
Unit tests for EffectivenessEstimator - AGENT-EUDR-029

Tests single-measure effectiveness estimation, strategy-level estimation,
cumulative reduction with diminishing returns, feasibility validation,
applicability factor, quality factor, and scenario bounds.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.effectiveness_estimator import (
    EffectivenessEstimator,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    Article11Category,
    EffectivenessEstimate,
    EUDRCommodity,
    MeasurePriority,
    MeasureStatus,
    MeasureTemplate,
    MitigationMeasure,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    RiskTrigger,
    WorkflowStatus,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return MitigationMeasureDesignerConfig()


@pytest.fixture
def estimator(config):
    return EffectivenessEstimator(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def test_template():
    return MeasureTemplate(
        template_id="T-TEST",
        title="Test Template",
        article11_category=Article11Category.INDEPENDENT_AUDIT,
        applicable_dimensions=[RiskDimension.SUPPLIER],
        base_effectiveness=Decimal("30"),
    )


@pytest.fixture
def test_measure():
    return MitigationMeasure(
        measure_id="msr-eff-001",
        strategy_id="stg-001",
        template_id="T-TEST",
        title="Test Measure",
        article11_category=Article11Category.INDEPENDENT_AUDIT,
        target_dimension=RiskDimension.SUPPLIER,
        expected_risk_reduction=Decimal("30"),
    )


@pytest.fixture
def test_measure_with_evidence():
    return MitigationMeasure(
        measure_id="msr-eff-002",
        strategy_id="stg-001",
        template_id="T-TEST",
        title="Test Measure with Evidence",
        article11_category=Article11Category.INDEPENDENT_AUDIT,
        target_dimension=RiskDimension.SUPPLIER,
        expected_risk_reduction=Decimal("30"),
        assigned_to="auditor@test.com",
        evidence_ids=["evd-001", "evd-002"],
    )


class TestSingleMeasureEffectiveness:
    """Test estimate_measure_effectiveness for single measures."""

    def test_returns_effectiveness_estimate(self, estimator, test_measure, test_template, sample_risk_trigger):
        est = estimator.estimate_measure_effectiveness(
            measure=test_measure,
            template=test_template,
            risk_trigger=sample_risk_trigger,
        )
        assert isinstance(est, EffectivenessEstimate)
        assert est.estimate_id.startswith("eff-")
        assert est.measure_id == "msr-eff-001"

    def test_conservative_le_moderate_le_optimistic(self, estimator, test_measure, test_template, sample_risk_trigger):
        est = estimator.estimate_measure_effectiveness(
            measure=test_measure,
            template=test_template,
            risk_trigger=sample_risk_trigger,
        )
        assert est.conservative <= est.moderate <= est.optimistic

    def test_estimates_within_bounds(self, estimator, test_measure, test_template, sample_risk_trigger):
        est = estimator.estimate_measure_effectiveness(
            measure=test_measure,
            template=test_template,
            risk_trigger=sample_risk_trigger,
        )
        min_thresh = estimator._config.min_effectiveness_threshold
        max_cap = estimator._config.max_effectiveness_cap
        assert est.conservative >= min_thresh
        assert est.moderate >= min_thresh
        assert est.optimistic <= max_cap

    def test_applicability_factor_in_range(self, estimator, test_measure, test_template, sample_risk_trigger):
        est = estimator.estimate_measure_effectiveness(
            measure=test_measure,
            template=test_template,
            risk_trigger=sample_risk_trigger,
        )
        assert Decimal("0.50") <= est.applicability_factor <= Decimal("1.00")

    def test_confidence_in_range(self, estimator, test_measure, test_template, sample_risk_trigger):
        est = estimator.estimate_measure_effectiveness(
            measure=test_measure,
            template=test_template,
            risk_trigger=sample_risk_trigger,
        )
        assert Decimal("0") <= est.confidence <= Decimal("1")


class TestStrategyEffectiveness:
    """Test estimate_strategy_effectiveness for strategies."""

    def test_returns_estimates_for_all_measures(self, estimator, sample_risk_trigger):
        measure1 = MitigationMeasure(
            measure_id="m1", strategy_id="s1", template_id="T-TEST", title="M1",
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=RiskDimension.SUPPLIER,
            expected_risk_reduction=Decimal("20"),
        )
        measure2 = MitigationMeasure(
            measure_id="m2", strategy_id="s1", template_id=None, title="M2",
            article11_category=Article11Category.ADDITIONAL_INFO,
            target_dimension=RiskDimension.COUNTRY,
            expected_risk_reduction=Decimal("15"),
        )
        strategy = MitigationStrategy(
            strategy_id="s1", workflow_id="w1",
            risk_trigger=sample_risk_trigger,
            measures=[measure1, measure2],
            pre_mitigation_score=Decimal("72"),
            target_score=Decimal("30"),
        )
        templates = {
            "T-TEST": MeasureTemplate(
                template_id="T-TEST", title="T",
                article11_category=Article11Category.OTHER_MEASURES,
                applicable_dimensions=[RiskDimension.SUPPLIER],
                base_effectiveness=Decimal("20"),
            ),
        }
        estimates = estimator.estimate_strategy_effectiveness(strategy, templates)
        assert len(estimates) == 2
        assert "m1" in estimates
        assert "m2" in estimates

    def test_synthetic_template_for_missing(self, estimator, sample_risk_trigger):
        """When template_id is not in templates dict, a synthetic is created."""
        measure = MitigationMeasure(
            measure_id="m-syn", strategy_id="s1", template_id=None, title="Synthetic",
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=RiskDimension.COUNTRY,
            expected_risk_reduction=Decimal("15"),
        )
        strategy = MitigationStrategy(
            strategy_id="s1", workflow_id="w1",
            risk_trigger=sample_risk_trigger,
            measures=[measure],
            pre_mitigation_score=Decimal("72"),
            target_score=Decimal("30"),
        )
        estimates = estimator.estimate_strategy_effectiveness(strategy, {})
        assert "m-syn" in estimates


class TestCumulativeReduction:
    """Test calculate_cumulative_reduction with diminishing returns."""

    def test_empty_estimates(self, estimator):
        c, m, o = estimator.calculate_cumulative_reduction([])
        assert c == Decimal("0")
        assert m == Decimal("0")
        assert o == Decimal("0")

    def test_single_estimate(self, estimator):
        estimates = [
            EffectivenessEstimate(
                estimate_id="e1", measure_id="m1",
                conservative=Decimal("20"), moderate=Decimal("25"),
                optimistic=Decimal("30"),
            )
        ]
        c, m, o = estimator.calculate_cumulative_reduction(estimates)
        assert c == Decimal("20.00")
        assert m == Decimal("25.00")
        assert o == Decimal("30.00")

    def test_diminishing_returns_multiple_estimates(self, estimator):
        estimates = [
            EffectivenessEstimate(
                estimate_id=f"e{i}", measure_id=f"m{i}",
                conservative=Decimal("20"), moderate=Decimal("25"),
                optimistic=Decimal("30"),
            )
            for i in range(3)
        ]
        c, m, o = estimator.calculate_cumulative_reduction(estimates)
        # conservative: 1 - 0.8^3 = 0.488 -> 48.80%
        assert c == Decimal("48.80")
        # Results should be less than 3 * individual value
        assert c < Decimal("60")
        assert m < Decimal("75")

    def test_capped_at_max_effectiveness(self, estimator):
        estimates = [
            EffectivenessEstimate(
                estimate_id=f"e{i}", measure_id=f"m{i}",
                conservative=Decimal("50"), moderate=Decimal("60"),
                optimistic=Decimal("70"),
            )
            for i in range(5)
        ]
        c, m, o = estimator.calculate_cumulative_reduction(estimates)
        cap = estimator._config.max_effectiveness_cap
        assert c <= cap
        assert m <= cap
        assert o <= cap


class TestFeasibilityValidation:
    """Test validate_feasibility."""

    def test_feasible_strategy(self, estimator):
        estimates = [
            EffectivenessEstimate(
                estimate_id="e1", measure_id="m1",
                conservative=Decimal("40"), moderate=Decimal("60"),
                optimistic=Decimal("75"),
            ),
        ]
        result = estimator.validate_feasibility(
            pre_score=Decimal("70"),
            target_score=Decimal("30"),
            estimates=estimates,
        )
        assert result is True

    def test_infeasible_strategy(self, estimator):
        estimates = [
            EffectivenessEstimate(
                estimate_id="e1", measure_id="m1",
                conservative=Decimal("5"), moderate=Decimal("10"),
                optimistic=Decimal("15"),
            ),
        ]
        result = estimator.validate_feasibility(
            pre_score=Decimal("80"),
            target_score=Decimal("30"),
            estimates=estimates,
        )
        assert result is False

    def test_already_at_target(self, estimator):
        result = estimator.validate_feasibility(
            pre_score=Decimal("25"),
            target_score=Decimal("30"),
            estimates=[],
        )
        assert result is True


class TestApplicabilityFactor:
    """Test _calculate_applicability_factor."""

    def test_low_score_gives_low_factor(self, estimator):
        factor = estimator._calculate_applicability_factor(
            dimension=RiskDimension.COUNTRY, risk_score=Decimal("0"),
        )
        assert factor == Decimal("0.50")

    def test_high_score_gives_high_factor(self, estimator):
        factor = estimator._calculate_applicability_factor(
            dimension=RiskDimension.COUNTRY, risk_score=Decimal("100"),
        )
        assert factor == Decimal("1.00")

    def test_mid_score_gives_mid_factor(self, estimator):
        factor = estimator._calculate_applicability_factor(
            dimension=RiskDimension.COUNTRY, risk_score=Decimal("50"),
        )
        assert factor == Decimal("0.75")


class TestQualityFactor:
    """Test _calculate_quality_factor."""

    def test_base_quality_for_minimal_measure(self, estimator):
        measure = MitigationMeasure(
            measure_id="m1", strategy_id="s1", title="T",
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=RiskDimension.COUNTRY,
        )
        factor = estimator._calculate_quality_factor(measure)
        assert factor == Decimal("1.00")

    def test_template_based_increases_quality(self, estimator):
        measure = MitigationMeasure(
            measure_id="m1", strategy_id="s1", title="T",
            template_id="T-001",
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=RiskDimension.COUNTRY,
        )
        factor = estimator._calculate_quality_factor(measure)
        assert factor > Decimal("1.00")

    def test_assigned_measure_increases_quality(self, estimator):
        measure = MitigationMeasure(
            measure_id="m1", strategy_id="s1", title="T",
            assigned_to="user@test.com",
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=RiskDimension.COUNTRY,
        )
        factor = estimator._calculate_quality_factor(measure)
        assert factor > Decimal("1.00")
