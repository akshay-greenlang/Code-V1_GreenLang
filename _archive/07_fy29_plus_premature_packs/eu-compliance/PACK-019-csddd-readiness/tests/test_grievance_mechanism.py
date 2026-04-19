# -*- coding: utf-8 -*-
"""
Tests for GrievanceMechanismEngine - PACK-019 CSDDD Readiness Pack
===================================================================

Validates grievance case handling, UNGP effectiveness criteria
assessment, resolution statistics, and response time analysis
per CSDDD Article 12.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-019 CSDDD Readiness Pack
"""

import sys
from pathlib import Path

import pytest
from decimal import Decimal
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import _load_engine


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_mod = _load_engine("grievance_mechanism")

GrievanceStatus = getattr(_mod, "GrievanceStatus")
StakeholderGroup = getattr(_mod, "StakeholderGroup")
GrievanceChannel = getattr(_mod, "GrievanceChannel")
MechanismCriteria = getattr(_mod, "MechanismCriteria")
GrievanceCase = getattr(_mod, "GrievanceCase")
MechanismConfig = getattr(_mod, "MechanismConfig")
MechanismAssessment = getattr(_mod, "MechanismAssessment")
ResolutionStats = getattr(_mod, "ResolutionStats")
ResponseTimeStats = getattr(_mod, "ResponseTimeStats")
GrievanceResult = getattr(_mod, "GrievanceResult")
GrievanceMechanismEngine = getattr(_mod, "GrievanceMechanismEngine")
CRITERIA_WEIGHTS = getattr(_mod, "CRITERIA_WEIGHTS")
ACCESSIBILITY_SUBCRITERIA = getattr(_mod, "ACCESSIBILITY_SUBCRITERIA")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(
    case_id="GC-001",
    status=None,
    group=None,
    channel=None,
    days_to_resolve=None,
    days_to_acknowledge=None,
):
    return GrievanceCase(
        case_id=case_id,
        status=status or GrievanceStatus.RESOLVED,
        stakeholder_group=group or StakeholderGroup.WORKERS,
        channel=channel or GrievanceChannel.WEB_PORTAL,
        description="Test grievance",
        days_to_resolve=days_to_resolve,
        days_to_acknowledge=days_to_acknowledge,
    )


def _make_config():
    return MechanismConfig(
        channels_available=[GrievanceChannel.WEB_PORTAL, GrievanceChannel.EMAIL],
        languages_supported=["en", "de"],
        anonymous_submission_allowed=True,
        disability_accessible=True,
        available_to_all_groups=True,
        publicised_to_stakeholders=True,
        geographically_accessible=True,
        has_written_procedures=True,
        has_defined_timeframes=True,
        has_escalation_process=True,
        has_appeal_mechanism=True,
        provides_progress_updates=True,
        outcomes_communicated=True,
        independent_oversight=True,
        rights_based_outcomes=True,
        lessons_learned_process=True,
        stakeholder_input_on_design=True,
        regular_effectiveness_review=True,
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:

    def test_grievance_status(self):
        assert GrievanceStatus.RECEIVED.value == "received"
        assert GrievanceStatus.RESOLVED.value == "resolved"
        assert GrievanceStatus.ESCALATED.value == "escalated"
        assert len(list(GrievanceStatus)) == 6

    def test_stakeholder_group(self):
        assert StakeholderGroup.WORKERS.value == "workers"
        assert StakeholderGroup.NGOS.value == "ngos"
        assert StakeholderGroup.REGULATORS.value == "regulators"
        assert len(list(StakeholderGroup)) == 8

    def test_grievance_channel(self):
        assert GrievanceChannel.HOTLINE.value == "hotline"
        assert GrievanceChannel.TRADE_UNION_REP.value == "trade_union_rep"
        assert len(list(GrievanceChannel)) == 7

    def test_mechanism_criteria(self):
        assert MechanismCriteria.LEGITIMATE.value == "legitimate"
        assert MechanismCriteria.ACCESSIBLE.value == "accessible"
        assert MechanismCriteria.BASED_ON_ENGAGEMENT_DIALOGUE.value == "based_on_engagement_dialogue"
        assert len(list(MechanismCriteria)) == 8


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestGrievanceCaseModel:

    def test_minimal_creation(self):
        gc = _make_case()
        assert gc.case_id == "GC-001"
        assert gc.status == GrievanceStatus.RESOLVED

    def test_defaults(self):
        gc = GrievanceCase(
            status=GrievanceStatus.RECEIVED,
            stakeholder_group=StakeholderGroup.COMMUNITIES,
        )
        assert gc.is_anonymous is False
        assert gc.complainant_satisfied is None
        assert gc.days_to_resolve is None
        assert gc.channel == GrievanceChannel.WEB_PORTAL


class TestMechanismConfigModel:

    def test_creation(self):
        mc = _make_config()
        assert len(mc.channels_available) == 2
        assert mc.anonymous_submission_allowed is True

    def test_defaults(self):
        mc = MechanismConfig()
        assert mc.no_cost_to_complainant is True
        assert mc.anonymous_submission_allowed is False


class TestResolutionStatsModel:

    def test_instantiation(self):
        rs = ResolutionStats(total_cases=10, resolved_count=7)
        assert rs.total_cases == 10
        assert rs.resolution_rate_pct == Decimal("0.0")


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestGrievanceMechanismEngine:

    @pytest.fixture
    def engine(self):
        return GrievanceMechanismEngine()

    @pytest.fixture
    def sample_cases(self):
        return [
            _make_case("GC-001", GrievanceStatus.RESOLVED,
                        StakeholderGroup.WORKERS,
                        days_to_resolve=45, days_to_acknowledge=2),
            _make_case("GC-002", GrievanceStatus.INVESTIGATING,
                        StakeholderGroup.COMMUNITIES,
                        days_to_acknowledge=3),
            _make_case("GC-003", GrievanceStatus.RECEIVED,
                        StakeholderGroup.NGOS),
        ]

    @pytest.fixture
    def sample_config(self):
        return _make_config()

    # -- Mechanism criteria assessment --

    def test_assess_mechanism_criteria(self, engine, sample_config):
        assessments = engine.assess_mechanism_criteria(sample_config)
        assert isinstance(assessments, list)
        assert len(assessments) == 8  # 8 UNGP criteria

    def test_assess_mechanism_criteria_types(self, engine, sample_config):
        assessments = engine.assess_mechanism_criteria(sample_config)
        for a in assessments:
            assert isinstance(a, dict)
            assert Decimal("0") <= Decimal(str(a["score"])) <= Decimal("100")

    # -- Resolution stats --

    def test_resolution_stats(self, engine, sample_cases):
        stats = engine.calculate_resolution_stats(sample_cases)
        assert isinstance(stats, dict)
        assert stats["total_cases"] == 3
        assert stats["resolved_count"] >= 1

    def test_resolution_stats_empty(self, engine):
        stats = engine.calculate_resolution_stats([])
        assert stats["total_cases"] == 0

    def test_resolution_stats_by_group(self, engine, sample_cases):
        stats = engine.calculate_resolution_stats(sample_cases)
        assert "workers" in stats["by_stakeholder_group"]

    # -- Response times --

    def test_response_times(self, engine, sample_cases):
        rt = engine.calculate_response_times(sample_cases)
        assert isinstance(rt, dict)

    def test_response_times_with_data(self, engine):
        cases = [
            _make_case("G1", GrievanceStatus.RESOLVED,
                        days_to_resolve=30, days_to_acknowledge=2),
            _make_case("G2", GrievanceStatus.RESOLVED,
                        days_to_resolve=60, days_to_acknowledge=4),
        ]
        rt = engine.calculate_response_times(cases)
        assert rt["cases_with_resolution_data"] == 2

    def test_response_times_empty(self, engine):
        rt = engine.calculate_response_times([])
        assert rt["cases_with_resolution_data"] == 0

    # -- Accessibility --

    def test_accessibility_assessment(self, engine, sample_config):
        score = engine.assess_accessibility(sample_config)
        assert isinstance(score, Decimal)
        assert Decimal("0") <= score <= Decimal("100")

    # -- Full assessment --

    def test_assess_mechanism_returns_result(self, engine, sample_cases, sample_config):
        result = engine.assess_grievance_mechanism(sample_cases, sample_config)
        assert isinstance(result, GrievanceResult)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_mechanism_empty_cases(self, engine, sample_config):
        result = engine.assess_grievance_mechanism([], sample_config)
        assert result.provenance_hash != ""

    def test_assess_mechanism_processing_time(self, engine, sample_cases, sample_config):
        result = engine.assess_grievance_mechanism(sample_cases, sample_config)
        assert result.processing_time_ms >= 0

    # -- Constants --

    def test_criteria_weights_sum_to_1(self):
        total = sum(CRITERIA_WEIGHTS.values())
        assert total == Decimal("1.000")

    def test_accessibility_subcriteria_count(self):
        assert len(ACCESSIBILITY_SUBCRITERIA) == 8
