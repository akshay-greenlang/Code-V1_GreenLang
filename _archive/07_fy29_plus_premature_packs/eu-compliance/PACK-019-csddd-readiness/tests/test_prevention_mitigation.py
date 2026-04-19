# -*- coding: utf-8 -*-
"""
Tests for PreventionMitigationEngine - PACK-019 CSDDD Readiness Pack
=====================================================================

Validates effectiveness calculations, budget summaries, coverage
analysis, and gap identification per CSDDD Articles 8 and 9.

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

_mod = _load_engine("prevention_mitigation")

MeasureType = getattr(_mod, "MeasureType")
MeasureStatus = getattr(_mod, "MeasureStatus")
EffectivenessRating = getattr(_mod, "EffectivenessRating")
MeasureCategory = getattr(_mod, "MeasureCategory")
ImpactDomain = getattr(_mod, "ImpactDomain")
PreventionMeasure = getattr(_mod, "PreventionMeasure")
MeasureEffectiveness = getattr(_mod, "MeasureEffectiveness")
BudgetSummary = getattr(_mod, "BudgetSummary")
CoverageAnalysis = getattr(_mod, "CoverageAnalysis")
GapAnalysis = getattr(_mod, "GapAnalysis")
EffectivenessSummary = getattr(_mod, "EffectivenessSummary")
PreventionResult = getattr(_mod, "PreventionResult")
PreventionMitigationEngine = getattr(_mod, "PreventionMitigationEngine")
STATUS_PROGRESS_WEIGHTS = getattr(_mod, "STATUS_PROGRESS_WEIGHTS")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_measure(
    measure_id="PM-001",
    measure_type=None,
    status=None,
    target_ids=None,
    budget=Decimal("50000"),
    effectiveness=Decimal("70"),
):
    return PreventionMeasure(
        measure_id=measure_id,
        measure_type=measure_type or MeasureType.PREVENTION,
        status=status or MeasureStatus.IN_PROGRESS,
        target_impact_ids=target_ids or ["AI-001"],
        budget_eur=budget,
        effectiveness_score=effectiveness,
        responsible_person="Test Lead",
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:

    def test_measure_type(self):
        assert MeasureType.PREVENTION.value == "prevention"
        assert MeasureType.CESSATION.value == "cessation"
        assert len(list(MeasureType)) == 4

    def test_measure_status(self):
        assert MeasureStatus.PLANNED.value == "planned"
        assert MeasureStatus.OVERDUE.value == "overdue"
        assert len(list(MeasureStatus)) == 5

    def test_effectiveness_rating(self):
        assert EffectivenessRating.HIGHLY_EFFECTIVE.value == "highly_effective"
        assert EffectivenessRating.NOT_ASSESSED.value == "not_assessed"
        assert len(list(EffectivenessRating)) == 5

    def test_measure_category(self):
        assert MeasureCategory.ACTION_PLAN.value == "action_plan"
        assert MeasureCategory.OTHER.value == "other"
        assert len(list(MeasureCategory)) == 10


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestPreventionMeasureModel:

    def test_minimal_creation(self):
        m = _make_measure()
        assert m.measure_id == "PM-001"
        assert m.budget_eur == Decimal("50000")

    def test_defaults(self):
        m = PreventionMeasure(
            measure_type=MeasureType.PREVENTION,
        )
        assert m.status == MeasureStatus.PLANNED
        assert m.budget_eur == Decimal("0")
        assert m.is_verified is False
        assert m.kpis == []


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestPreventionMitigationEngine:

    @pytest.fixture
    def engine(self):
        return PreventionMitigationEngine()

    @pytest.fixture
    def sample_measures(self):
        return [
            _make_measure("PM-001", MeasureType.PREVENTION,
                          MeasureStatus.IN_PROGRESS, ["AI-001"],
                          Decimal("50000"), Decimal("70")),
            _make_measure("PM-002", MeasureType.MITIGATION,
                          MeasureStatus.COMPLETED, ["AI-002"],
                          Decimal("200000"), Decimal("85")),
            _make_measure("PM-003", MeasureType.PREVENTION,
                          MeasureStatus.PLANNED, ["AI-003"],
                          Decimal("150000"), Decimal("0")),
        ]

    @pytest.fixture
    def impact_ids(self):
        return ["AI-001", "AI-002", "AI-003", "AI-004", "AI-005"]

    # -- Effectiveness --

    @pytest.mark.parametrize("score,expected_rating", [
        (Decimal("95"), EffectivenessRating.HIGHLY_EFFECTIVE),
        (Decimal("90"), EffectivenessRating.HIGHLY_EFFECTIVE),
        (Decimal("75"), EffectivenessRating.EFFECTIVE),
        (Decimal("50"), EffectivenessRating.PARTIALLY_EFFECTIVE),
        (Decimal("20"), EffectivenessRating.INEFFECTIVE),
    ])
    def test_effectiveness_rating(self, engine, score, expected_rating):
        m = _make_measure(effectiveness=score, status=MeasureStatus.COMPLETED)
        eff = engine.calculate_effectiveness(m)
        assert eff.rating == expected_rating

    def test_effectiveness_cancelled_is_not_assessed(self, engine):
        m = _make_measure(status=MeasureStatus.CANCELLED)
        eff = engine.calculate_effectiveness(m)
        assert eff.rating == EffectivenessRating.NOT_ASSESSED

    def test_effectiveness_zero_planned_is_not_assessed(self, engine):
        m = _make_measure(effectiveness=Decimal("0"), status=MeasureStatus.PLANNED)
        eff = engine.calculate_effectiveness(m)
        assert eff.rating == EffectivenessRating.NOT_ASSESSED

    # -- Budget --

    def test_budget_summary(self, engine, sample_measures):
        bs = engine.calculate_budget_summary(sample_measures)
        assert bs.total_budget_eur == Decimal("400000")
        assert bs.measures_with_budget == 3

    def test_budget_empty(self, engine):
        bs = engine.calculate_budget_summary([])
        assert bs.total_budget_eur == Decimal("0")

    def test_budget_over_budget_detection(self, engine):
        m = _make_measure(budget=Decimal("100"))
        m.actual_cost_eur = Decimal("150")
        bs = engine.calculate_budget_summary([m])
        assert bs.measures_over_budget == 1
        assert bs.over_budget_amount_eur == Decimal("50")

    # -- Coverage --

    def test_coverage_analysis(self, engine, sample_measures, impact_ids):
        ca = engine.analyze_coverage(sample_measures, impact_ids)
        assert ca.total_impacts == 5
        assert ca.covered_impacts == 3
        assert ca.uncovered_impacts == 2
        assert ca.coverage_pct > Decimal("0")

    def test_coverage_empty_impacts(self, engine, sample_measures):
        ca = engine.analyze_coverage(sample_measures, [])
        assert ca.total_impacts == 0

    def test_coverage_cancelled_excluded(self, engine, impact_ids):
        m = _make_measure(status=MeasureStatus.CANCELLED, target_ids=["AI-001"])
        ca = engine.analyze_coverage([m], impact_ids)
        assert ca.covered_impacts == 0

    # -- Gap analysis --

    def test_gap_analysis_uncovered(self, engine, sample_measures, impact_ids):
        ga = engine.identify_gaps(sample_measures, impact_ids)
        assert "AI-004" in ga.uncovered_impact_ids
        assert "AI-005" in ga.uncovered_impact_ids

    def test_gap_analysis_missing_kpis(self, engine, impact_ids):
        m = _make_measure()
        # kpis default is empty list
        ga = engine.identify_gaps([m], impact_ids)
        assert ga.missing_kpis >= 1

    def test_gap_analysis_recommendations(self, engine, sample_measures, impact_ids):
        ga = engine.identify_gaps(sample_measures, impact_ids)
        assert len(ga.recommendations) > 0

    # -- Overall progress --

    def test_overall_progress_all_completed(self, engine):
        m1 = _make_measure("M1", status=MeasureStatus.COMPLETED)
        m2 = _make_measure("M2", status=MeasureStatus.COMPLETED)
        score = engine._calculate_overall_progress([m1, m2])
        assert score == Decimal("100.0")

    def test_overall_progress_all_planned(self, engine):
        m = _make_measure(status=MeasureStatus.PLANNED)
        score = engine._calculate_overall_progress([m])
        assert score == Decimal("10.0")

    def test_overall_progress_empty(self, engine):
        score = engine._calculate_overall_progress([])
        assert score == Decimal("0")

    # -- Verification rate --

    def test_verification_rate_none_verified(self, engine):
        m = _make_measure(status=MeasureStatus.COMPLETED)
        rate = engine._calculate_verification_rate([m])
        assert rate == Decimal("0.0")

    def test_verification_rate_all_verified(self, engine):
        m = _make_measure(status=MeasureStatus.COMPLETED)
        m.is_verified = True
        rate = engine._calculate_verification_rate([m])
        assert rate == Decimal("100.0")

    # -- Full assessment --

    def test_assess_returns_result(self, engine, sample_measures, impact_ids):
        result = engine.assess_prevention_measures(sample_measures, impact_ids)
        assert isinstance(result, PreventionResult)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_empty_measures(self, engine, impact_ids):
        result = engine.assess_prevention_measures([], impact_ids)
        assert result.total_measures == 0
        assert result.coverage_analysis.uncovered_impacts == 5
        assert result.provenance_hash != ""

    def test_assess_total_measures(self, engine, sample_measures, impact_ids):
        result = engine.assess_prevention_measures(sample_measures, impact_ids)
        assert result.total_measures == 3

    def test_assess_measures_by_type(self, engine, sample_measures, impact_ids):
        result = engine.assess_prevention_measures(sample_measures, impact_ids)
        assert result.measures_by_type["prevention"] == 2
        assert result.measures_by_type["mitigation"] == 1

    def test_assess_processing_time(self, engine, sample_measures, impact_ids):
        result = engine.assess_prevention_measures(sample_measures, impact_ids)
        assert result.processing_time_ms >= 0

    # -- Constants --

    def test_status_progress_weights(self):
        assert STATUS_PROGRESS_WEIGHTS["completed"] == Decimal("1.0")
        assert STATUS_PROGRESS_WEIGHTS["cancelled"] == Decimal("0.0")
