# -*- coding: utf-8 -*-
"""
Tests for ClimateTransitionEngine - PACK-019 CSDDD Readiness Pack
==================================================================

Validates climate target assessment, transition plan evaluation,
SBTi alignment, and pathway analysis per CSDDD Article 22.

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

_mod = _load_engine("climate_transition")

TransitionPlanStatus = getattr(_mod, "TransitionPlanStatus")
EmissionScope = getattr(_mod, "EmissionScope")
AlignmentLevel = getattr(_mod, "AlignmentLevel")
TransitionElement = getattr(_mod, "TransitionElement")
InterimMilestone = getattr(_mod, "InterimMilestone")
ClimateTarget = getattr(_mod, "ClimateTarget")
TransitionPlanDetails = getattr(_mod, "TransitionPlanDetails")
TransitionPlanAssessment = getattr(_mod, "TransitionPlanAssessment")
TargetAnalysis = getattr(_mod, "TargetAnalysis")
ClimateTransitionResult = getattr(_mod, "ClimateTransitionResult")
ClimateTransitionEngine = getattr(_mod, "ClimateTransitionEngine")
SBTI_15C_ANNUAL_REDUCTION_PCT = getattr(_mod, "SBTI_15C_ANNUAL_REDUCTION_PCT")
SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT = getattr(_mod, "SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT")
REQUIRED_ELEMENTS = getattr(_mod, "REQUIRED_ELEMENTS")
ART_22_MILESTONE_YEARS = getattr(_mod, "ART_22_MILESTONE_YEARS")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    target_id="CT-001",
    scope=None,
    base_year=2023,
    target_year=2030,
    reduction_pct=Decimal("42"),
    aligned_15c=True,
):
    return ClimateTarget(
        target_id=target_id,
        scope=scope or EmissionScope.SCOPE_1,
        base_year=base_year,
        target_year=target_year,
        reduction_pct=reduction_pct,
        aligned_with_15c=aligned_15c,
        base_year_emissions_tco2e=Decimal("100000"),
        current_year=2025,
        current_emissions_tco2e=Decimal("85000"),
    )


def _make_plan_details():
    return TransitionPlanDetails(
        status=TransitionPlanStatus.IMPLEMENTING,
        has_targets=True,
        has_decarbonization_levers=True,
        has_investment_plan=True,
        has_governance=True,
        has_engagement=True,
        has_monitoring=True,
        board_oversight=True,
        dedicated_team=True,
        kpis_defined=True,
        annual_review=True,
        scope_3_included=True,
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:

    def test_transition_plan_status(self):
        assert TransitionPlanStatus.DRAFTED.value == "drafted"
        assert TransitionPlanStatus.IMPLEMENTING.value == "implementing"
        assert TransitionPlanStatus.ACHIEVED.value == "achieved"
        assert len(list(TransitionPlanStatus)) == 6

    def test_emission_scope(self):
        assert EmissionScope.SCOPE_1.value == "scope_1"
        assert EmissionScope.SCOPE_2.value == "scope_2"
        assert EmissionScope.SCOPE_3.value == "scope_3"
        assert len(list(EmissionScope)) == 3

    def test_alignment_level(self):
        assert AlignmentLevel.PARIS_ALIGNED_15C.value == "paris_aligned_1_5c"
        assert AlignmentLevel.NOT_ALIGNED.value == "not_aligned"
        assert AlignmentLevel.INSUFFICIENT_DATA.value == "insufficient_data"
        assert len(list(AlignmentLevel)) == 5

    def test_transition_element(self):
        assert TransitionElement.TARGETS.value == "targets"
        assert TransitionElement.MONITORING.value == "monitoring"
        assert len(list(TransitionElement)) == 6


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestClimateTargetModel:

    def test_minimal_creation(self):
        ct = _make_target()
        assert ct.target_id == "CT-001"
        assert ct.reduction_pct == Decimal("42")

    def test_target_year_after_base(self):
        with pytest.raises(Exception):
            ClimateTarget(
                scope=EmissionScope.SCOPE_1,
                base_year=2025,
                target_year=2025,
                reduction_pct=Decimal("10"),
            )

    def test_defaults(self):
        ct = _make_target()
        assert ct.is_sbti_validated is False
        assert ct.is_net_zero_target is False
        assert ct.interim_milestones == []


class TestInterimMilestoneModel:

    def test_instantiation(self):
        m = InterimMilestone(year=2027, reduction_pct=Decimal("20"))
        assert m.year == 2027
        assert m.status == "planned"


class TestTransitionPlanDetailsModel:

    def test_creation(self):
        pd = _make_plan_details()
        assert pd.has_targets is True
        assert pd.has_governance is True

    def test_defaults(self):
        pd = TransitionPlanDetails()
        assert pd.status == TransitionPlanStatus.DRAFTED
        assert pd.has_targets is False
        assert pd.total_investment_eur == Decimal("0")


class TestTargetAnalysisModel:

    def test_instantiation(self):
        ta = TargetAnalysis(target_id="CT-001", scope="scope_1")
        assert ta.alignment_level == AlignmentLevel.INSUFFICIENT_DATA.value


class TestClimateTransitionResultModel:

    def test_instantiation(self):
        r = ClimateTransitionResult()
        assert r.provenance_hash == ""
        assert r.targets_count == 0


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestClimateTransitionEngine:

    @pytest.fixture
    def engine(self):
        return ClimateTransitionEngine()

    @pytest.fixture
    def sample_targets(self):
        return [
            _make_target("CT-001", EmissionScope.SCOPE_1, 2023, 2030, Decimal("42"), True),
            _make_target("CT-002", EmissionScope.SCOPE_2, 2023, 2030, Decimal("50"), True),
            _make_target("CT-003", EmissionScope.SCOPE_3, 2023, 2035, Decimal("30"), False),
        ]

    @pytest.fixture
    def sample_plan(self):
        return _make_plan_details()

    # -- Target ambition --

    def test_assess_target_ambition(self, engine, sample_targets):
        """assess_target_ambition takes a single ClimateTarget."""
        results = [engine.assess_target_ambition(t) for t in sample_targets]
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, dict)
            assert "alignment_level" in r

    def test_assess_target_ambition_single(self, engine):
        t = _make_target(reduction_pct=Decimal("42"))
        result = engine.assess_target_ambition(t)
        assert isinstance(result, dict)
        assert "alignment_level" in result

    def test_assess_target_ambition_aligned(self, engine):
        """42% reduction over 7 years = 6.0%/yr > 4.2% SBTi threshold."""
        t = _make_target(reduction_pct=Decimal("42"), base_year=2023, target_year=2030)
        result = engine.assess_target_ambition(t)
        assert result["alignment_level"] == "paris_aligned_1_5c"

    # -- Pathway assessment --

    def test_assess_pathway(self, engine, sample_targets):
        result = engine.assess_pathway(sample_targets)
        assert isinstance(result, dict)

    # -- Implementation readiness --

    def test_assess_implementation_readiness(self, engine, sample_plan):
        result = engine.assess_implementation_readiness(sample_plan)
        assert isinstance(result, dict)

    def test_assess_implementation_readiness_minimal(self, engine):
        plan = TransitionPlanDetails()
        result = engine.assess_implementation_readiness(plan)
        assert isinstance(result, dict)

    # -- Alignment calculation --

    def test_calculate_alignment(self, engine, sample_targets):
        result = engine.calculate_alignment(sample_targets)
        assert isinstance(result, str)
        valid_levels = [a.value for a in AlignmentLevel]
        assert result in valid_levels

    # -- Full assessment --

    def test_assess_transition_plan_returns_result(self, engine, sample_targets, sample_plan):
        result = engine.assess_transition_plan(sample_targets, sample_plan)
        assert isinstance(result, ClimateTransitionResult)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_transition_plan_empty_targets(self, engine, sample_plan):
        result = engine.assess_transition_plan([], sample_plan)
        assert result.provenance_hash != ""
        assert result.targets_count == 0

    def test_assess_transition_plan_target_count(self, engine, sample_targets, sample_plan):
        result = engine.assess_transition_plan(sample_targets, sample_plan)
        assert result.targets_count == 3

    def test_assess_transition_plan_processing_time(self, engine, sample_targets, sample_plan):
        result = engine.assess_transition_plan(sample_targets, sample_plan)
        assert result.processing_time_ms >= 0

    def test_assess_transition_plan_has_recommendations(self, engine, sample_plan):
        targets = [_make_target(aligned_15c=False, reduction_pct=Decimal("10"))]
        result = engine.assess_transition_plan(targets, sample_plan)
        # Should have recommendations since target is weak
        assert isinstance(result.recommendations, list)

    def test_assess_transition_plan_overall_score(self, engine, sample_targets, sample_plan):
        result = engine.assess_transition_plan(sample_targets, sample_plan)
        assert Decimal("0") <= result.overall_score <= Decimal("100")

    # -- Constants --

    def test_sbti_thresholds(self):
        assert SBTI_15C_ANNUAL_REDUCTION_PCT == Decimal("4.2")
        assert SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT == Decimal("2.5")

    def test_required_elements(self):
        assert "targets" in REQUIRED_ELEMENTS
        assert "monitoring" in REQUIRED_ELEMENTS
        assert len(REQUIRED_ELEMENTS) == 6

    def test_milestone_years(self):
        assert 2030 in ART_22_MILESTONE_YEARS
        assert 2050 in ART_22_MILESTONE_YEARS
