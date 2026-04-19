# -*- coding: utf-8 -*-
"""
Tests for AdverseImpactEngine - PACK-019 CSDDD Readiness Pack
==============================================================

Validates risk matrix calculation, impact prioritisation, summary
statistics, and filtering per CSDDD Articles 6 and 7.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-019 CSDDD Readiness Pack
"""

import sys
from pathlib import Path

import pytest
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import _load_engine


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_mod = _load_engine("adverse_impact")

AdverseImpactType = getattr(_mod, "AdverseImpactType")
ImpactSeverity = getattr(_mod, "ImpactSeverity")
ImpactLikelihood = getattr(_mod, "ImpactLikelihood")
ImpactStatus = getattr(_mod, "ImpactStatus")
ValueChainPosition = getattr(_mod, "ValueChainPosition")
HumanRightsCategory = getattr(_mod, "HumanRightsCategory")
EnvironmentalCategory = getattr(_mod, "EnvironmentalCategory")
RiskLevel = getattr(_mod, "RiskLevel")
AdverseImpact = getattr(_mod, "AdverseImpact")
RiskMatrix = getattr(_mod, "RiskMatrix")
SummaryStatistics = getattr(_mod, "SummaryStatistics")
ImpactAssessmentResult = getattr(_mod, "ImpactAssessmentResult")
AdverseImpactEngine = getattr(_mod, "AdverseImpactEngine")
SEVERITY_SCORES = getattr(_mod, "SEVERITY_SCORES")
LIKELIHOOD_SCORES = getattr(_mod, "LIKELIHOOD_SCORES")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_impact(
    impact_id="TEST-001",
    impact_type=None,
    category="forced_labour",
    severity=None,
    likelihood=None,
    status=None,
    vc_position=None,
    country="MM",
):
    return AdverseImpact(
        impact_id=impact_id,
        impact_type=impact_type or AdverseImpactType.HUMAN_RIGHTS,
        category=category,
        severity=severity or ImpactSeverity.CRITICAL,
        likelihood=likelihood or ImpactLikelihood.LIKELY,
        status=status or ImpactStatus.POTENTIAL,
        value_chain_position=vc_position or ValueChainPosition.UPSTREAM_INDIRECT,
        country=country,
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:

    def test_adverse_impact_type(self):
        assert AdverseImpactType.HUMAN_RIGHTS.value == "human_rights"
        assert AdverseImpactType.ENVIRONMENTAL.value == "environmental"

    def test_impact_severity(self):
        assert len(list(ImpactSeverity)) == 4
        assert ImpactSeverity.CRITICAL.value == "critical"
        assert ImpactSeverity.LOW.value == "low"

    def test_impact_likelihood(self):
        assert len(list(ImpactLikelihood)) == 5
        assert ImpactLikelihood.VERY_LIKELY.value == "very_likely"
        assert ImpactLikelihood.RARE.value == "rare"

    def test_risk_level(self):
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.VERY_LOW.value == "very_low"

    def test_value_chain_positions(self):
        assert len(list(ValueChainPosition)) == 5
        assert ValueChainPosition.OWN_OPERATIONS.value == "own_operations"

    def test_human_rights_categories(self):
        assert HumanRightsCategory.FORCED_LABOUR.value == "forced_labour"
        assert HumanRightsCategory.CHILD_LABOUR.value == "child_labour"

    def test_environmental_categories(self):
        assert EnvironmentalCategory.DEFORESTATION.value == "deforestation"
        assert EnvironmentalCategory.WATER_POLLUTION.value == "water_pollution"


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestAdverseImpactModel:

    def test_minimal_creation(self):
        ai = _make_impact()
        assert ai.impact_id == "TEST-001"
        assert ai.severity == ImpactSeverity.CRITICAL

    def test_country_uppercased(self):
        ai = AdverseImpact(
            impact_type=AdverseImpactType.HUMAN_RIGHTS,
            category="forced_labour",
            severity=ImpactSeverity.HIGH,
            likelihood=ImpactLikelihood.LIKELY,
            status=ImpactStatus.ACTUAL,
            value_chain_position=ValueChainPosition.OWN_OPERATIONS,
            country="de",
        )
        assert ai.country == "DE"

    def test_defaults(self):
        ai = _make_impact()
        assert ai.affected_stakeholder_count == 0
        assert ai.is_salient is False
        assert ai.irremediable_character is False


class TestRiskMatrixModel:

    def test_instantiation(self):
        rm = RiskMatrix(
            severity_score=4,
            likelihood_score=5,
            risk_score=20,
            risk_level=RiskLevel.CRITICAL,
        )
        assert rm.risk_score == 20


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestAdverseImpactEngine:

    @pytest.fixture
    def engine(self):
        return AdverseImpactEngine()

    @pytest.fixture
    def sample_impacts(self):
        return [
            _make_impact("AI-001", severity=ImpactSeverity.CRITICAL,
                         likelihood=ImpactLikelihood.LIKELY),
            _make_impact("AI-002", impact_type=AdverseImpactType.ENVIRONMENTAL,
                         category="water_pollution",
                         severity=ImpactSeverity.HIGH,
                         likelihood=ImpactLikelihood.VERY_LIKELY,
                         status=ImpactStatus.ACTUAL,
                         vc_position=ValueChainPosition.UPSTREAM_DIRECT,
                         country="CN"),
            _make_impact("AI-003", severity=ImpactSeverity.MEDIUM,
                         likelihood=ImpactLikelihood.POSSIBLE,
                         vc_position=ValueChainPosition.OWN_OPERATIONS,
                         country="DE"),
        ]

    # -- Risk matrix --

    @pytest.mark.parametrize("severity,likelihood,expected_score", [
        (ImpactSeverity.CRITICAL, ImpactLikelihood.VERY_LIKELY, 20),
        (ImpactSeverity.CRITICAL, ImpactLikelihood.LIKELY, 16),
        (ImpactSeverity.HIGH, ImpactLikelihood.VERY_LIKELY, 15),
        (ImpactSeverity.MEDIUM, ImpactLikelihood.POSSIBLE, 6),
        (ImpactSeverity.LOW, ImpactLikelihood.RARE, 1),
    ])
    def test_risk_score_calculation(self, engine, severity, likelihood, expected_score):
        impact = _make_impact(severity=severity, likelihood=likelihood)
        rm = engine.calculate_risk_matrix(impact)
        assert rm.risk_score == expected_score

    @pytest.mark.parametrize("risk_score,expected_level", [
        (20, RiskLevel.CRITICAL),
        (16, RiskLevel.CRITICAL),
        (15, RiskLevel.HIGH),
        (12, RiskLevel.HIGH),
        (11, RiskLevel.MEDIUM),
        (6, RiskLevel.MEDIUM),
        (5, RiskLevel.LOW),
        (3, RiskLevel.LOW),
        (2, RiskLevel.VERY_LOW),
        (1, RiskLevel.VERY_LOW),
    ])
    def test_risk_level_mapping(self, engine, risk_score, expected_level):
        assert engine._score_to_risk_level(risk_score) == expected_level

    # -- Prioritisation --

    def test_prioritize_impacts_ordering(self, engine, sample_impacts):
        matrices = [engine.calculate_risk_matrix(i) for i in sample_impacts]
        priority_ids, sorted_matrices = engine.prioritize_impacts(
            sample_impacts, matrices
        )
        # AI-001 CRITICAL(4)*LIKELY(4)=16 > AI-002 HIGH(3)*VERY_LIKELY(5)=15
        assert priority_ids[0] == "AI-001"
        assert priority_ids[1] == "AI-002"

    def test_prioritize_assigns_ranks(self, engine, sample_impacts):
        matrices = [engine.calculate_risk_matrix(i) for i in sample_impacts]
        _, sorted_matrices = engine.prioritize_impacts(sample_impacts, matrices)
        ranks = [rm.priority_rank for rm in sorted_matrices]
        assert ranks == [1, 2, 3]

    def test_prioritize_empty(self, engine):
        ids, matrices = engine.prioritize_impacts([], [])
        assert ids == []
        assert matrices == []

    # -- Summary statistics --

    def test_summary_stats_counts(self, engine, sample_impacts):
        matrices = [engine.calculate_risk_matrix(i) for i in sample_impacts]
        stats = engine.get_summary_statistics(sample_impacts, matrices)
        assert stats.total_impacts == 3
        assert stats.human_rights_count == 2
        assert stats.environmental_count == 1

    def test_summary_stats_status_counts(self, engine, sample_impacts):
        matrices = [engine.calculate_risk_matrix(i) for i in sample_impacts]
        stats = engine.get_summary_statistics(sample_impacts, matrices)
        assert stats.actual_count == 1
        assert stats.potential_count == 2

    def test_summary_stats_empty(self, engine):
        stats = engine.get_summary_statistics([], [])
        assert stats.total_impacts == 0
        assert stats.average_risk_score == Decimal("0")

    def test_summary_average_risk(self, engine, sample_impacts):
        matrices = [engine.calculate_risk_matrix(i) for i in sample_impacts]
        stats = engine.get_summary_statistics(sample_impacts, matrices)
        assert stats.average_risk_score > Decimal("0")

    # -- Full assessment --

    def test_assess_impacts_returns_result(self, engine, sample_impacts):
        result = engine.assess_impacts(sample_impacts)
        assert isinstance(result, ImpactAssessmentResult)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_impacts_empty(self, engine):
        result = engine.assess_impacts([])
        assert result.summary_stats.total_impacts == 0
        assert result.provenance_hash != ""

    def test_assess_impacts_top_risks(self, engine, sample_impacts):
        result = engine.assess_impacts(sample_impacts)
        assert len(result.top_risks) <= 10
        assert len(result.top_risks) == 3

    def test_assess_impacts_category_summary(self, engine, sample_impacts):
        result = engine.assess_impacts(sample_impacts)
        assert len(result.category_risk_summary) > 0

    def test_assess_impacts_vc_summary(self, engine, sample_impacts):
        result = engine.assess_impacts(sample_impacts)
        assert len(result.value_chain_risk_summary) > 0

    # -- Single impact --

    def test_assess_single_impact(self, engine):
        impact = _make_impact()
        result = engine.assess_single_impact(impact)
        assert "risk_score" in result
        assert "provenance_hash" in result
        assert result["risk_level"] == "critical"

    # -- Filters --

    def test_filter_by_type(self, engine, sample_impacts):
        hr = AdverseImpactEngine.filter_impacts_by_type(
            sample_impacts, AdverseImpactType.HUMAN_RIGHTS
        )
        assert len(hr) == 2

    def test_filter_by_value_chain(self, engine, sample_impacts):
        own = AdverseImpactEngine.filter_impacts_by_value_chain(
            sample_impacts, ValueChainPosition.OWN_OPERATIONS
        )
        assert len(own) == 1

    def test_filter_by_risk_level(self, engine, sample_impacts):
        matrices = [engine.calculate_risk_matrix(i) for i in sample_impacts]
        high_plus = AdverseImpactEngine.filter_impacts_by_risk_level(
            sample_impacts, matrices, RiskLevel.HIGH
        )
        assert len(high_plus) >= 1

    # -- Constants --

    def test_severity_scores(self):
        assert SEVERITY_SCORES["critical"] == 4
        assert SEVERITY_SCORES["low"] == 1

    def test_likelihood_scores(self):
        assert LIKELIHOOD_SCORES["very_likely"] == 5
        assert LIKELIHOOD_SCORES["rare"] == 1
