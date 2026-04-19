# -*- coding: utf-8 -*-
"""
Tests for RemediationTrackingEngine - PACK-019 CSDDD Readiness Pack
====================================================================

Validates remediation action tracking, financial analysis, victim
engagement assessment, and timeline compliance per CSDDD Article 10.

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

_mod = _load_engine("remediation_tracking")

RemediationStatus = getattr(_mod, "RemediationStatus")
RemediationType = getattr(_mod, "RemediationType")
CompanyContribution = getattr(_mod, "CompanyContribution")
VictimEngagementLevel = getattr(_mod, "VictimEngagementLevel")
ImpactDomain = getattr(_mod, "ImpactDomain")
RemediationAction = getattr(_mod, "RemediationAction")
TimelineAnalysis = getattr(_mod, "TimelineAnalysis")
FinancialAnalysis = getattr(_mod, "FinancialAnalysis")
VictimEngagementAnalysis = getattr(_mod, "VictimEngagementAnalysis")
RemediationTrackingEngine = getattr(_mod, "RemediationTrackingEngine")
ENGAGEMENT_SCORES = getattr(_mod, "ENGAGEMENT_SCORES")
CONTRIBUTION_WEIGHTS = getattr(_mod, "CONTRIBUTION_WEIGHTS")

# Result model -- check both possible names
RemediationResult = getattr(_mod, "RemediationResult", None)
if RemediationResult is None:
    RemediationResult = getattr(_mod, "RemediationTrackingResult", None)
CompletenessAssessment = getattr(_mod, "CompletenessAssessment", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action(
    action_id="RA-001",
    adverse_impact_id="AI-002",
    rem_type=None,
    status=None,
    provision=Decimal("200000"),
    engagement=None,
    victim_count=100,
):
    return RemediationAction(
        action_id=action_id,
        adverse_impact_id=adverse_impact_id,
        remediation_type=rem_type or RemediationType.FINANCIAL_COMPENSATION,
        completion_status=status or RemediationStatus.IN_PROGRESS,
        financial_provision_eur=provision,
        victim_engagement=engagement or VictimEngagementLevel.CONSULTED,
        victim_count=victim_count,
        victims_reached=50,
        responsible_person="Remediation Lead",
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:

    def test_remediation_status(self):
        assert RemediationStatus.NOT_STARTED.value == "not_started"
        assert RemediationStatus.VERIFIED.value == "verified"
        assert RemediationStatus.FAILED.value == "failed"
        assert len(list(RemediationStatus)) == 5

    def test_remediation_type(self):
        assert RemediationType.FINANCIAL_COMPENSATION.value == "financial_compensation"
        assert RemediationType.RESTITUTION.value == "restitution"
        assert RemediationType.REHABILITATION.value == "rehabilitation"
        assert RemediationType.GUARANTEE_NON_REPETITION.value == "guarantee_non_repetition"
        assert RemediationType.APOLOGY.value == "apology"
        assert RemediationType.OPERATIONAL_CHANGE.value == "operational_change"
        assert len(list(RemediationType)) == 6

    def test_company_contribution(self):
        assert CompanyContribution.CAUSED.value == "caused"
        assert CompanyContribution.DIRECTLY_LINKED.value == "directly_linked"
        assert len(list(CompanyContribution)) == 4

    def test_victim_engagement_level(self):
        assert VictimEngagementLevel.NONE.value == "none"
        assert VictimEngagementLevel.CO_DESIGNED.value == "co_designed"
        assert len(list(VictimEngagementLevel)) == 5

    def test_impact_domain(self):
        assert ImpactDomain.HUMAN_RIGHTS.value == "human_rights"
        assert ImpactDomain.BOTH.value == "both"


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestRemediationActionModel:

    def test_minimal_creation(self):
        ra = _make_action()
        assert ra.action_id == "RA-001"
        assert ra.financial_provision_eur == Decimal("200000")

    def test_defaults(self):
        ra = RemediationAction(
            adverse_impact_id="AI-X",
            remediation_type=RemediationType.APOLOGY,
        )
        assert ra.completion_status == RemediationStatus.NOT_STARTED
        assert ra.financial_provision_eur == Decimal("0")
        assert ra.victim_engagement == VictimEngagementLevel.NONE
        assert ra.grievance_mechanism_used is False

    def test_country_uppercased(self):
        ra = RemediationAction(
            adverse_impact_id="AI-X",
            remediation_type=RemediationType.RESTITUTION,
            country="de",
        )
        assert ra.country == "DE"


class TestTimelineAnalysisModel:

    def test_instantiation(self):
        ta = TimelineAnalysis(total_actions=5)
        assert ta.total_actions == 5
        assert ta.on_time_rate_pct == Decimal("0.0")


class TestFinancialAnalysisModel:

    def test_instantiation(self):
        fa = FinancialAnalysis(
            total_financial_provision_eur=Decimal("500000"),
        )
        assert fa.total_financial_provision_eur == Decimal("500000")


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestRemediationTrackingEngine:

    @pytest.fixture
    def engine(self):
        return RemediationTrackingEngine()

    @pytest.fixture
    def sample_actions(self):
        return [
            _make_action("RA-001", "AI-002",
                         RemediationType.FINANCIAL_COMPENSATION,
                         RemediationStatus.IN_PROGRESS,
                         Decimal("200000"),
                         VictimEngagementLevel.ACTIVELY_ENGAGED, 100),
            _make_action("RA-002", "AI-005",
                         RemediationType.OPERATIONAL_CHANGE,
                         RemediationStatus.COMPLETED,
                         Decimal("75000"),
                         VictimEngagementLevel.CONSULTED, 50),
        ]

    # -- Completion rate --

    def test_completion_rate(self, engine, sample_actions):
        rate = engine.calculate_completion_rate(sample_actions)
        # 1 out of 2 completed
        assert rate == Decimal("50.0")

    def test_completion_rate_empty(self, engine):
        rate = engine.calculate_completion_rate([])
        assert rate == Decimal("0.0")

    def test_completion_rate_all_completed(self, engine):
        a1 = _make_action(status=RemediationStatus.COMPLETED)
        a2 = _make_action("RA-X", status=RemediationStatus.VERIFIED)
        rate = engine.calculate_completion_rate([a1, a2])
        assert rate >= Decimal("50.0")

    # -- Victim engagement --

    def test_victim_engagement_assessment(self, engine, sample_actions):
        analysis = engine.assess_victim_engagement(sample_actions)
        assert isinstance(analysis, VictimEngagementAnalysis)
        assert analysis.total_actions == 2
        assert analysis.actions_with_engagement >= 1

    def test_victim_engagement_none(self, engine):
        a = _make_action(engagement=VictimEngagementLevel.NONE)
        analysis = engine.assess_victim_engagement([a])
        assert analysis.actions_with_engagement == 0

    # -- Financial analysis --

    def test_financial_analysis(self, engine, sample_actions):
        fa = engine.analyze_financial_provisions(sample_actions)
        assert isinstance(fa, FinancialAnalysis)
        assert fa.total_financial_provision_eur == Decimal("275000")

    def test_financial_analysis_empty(self, engine):
        fa = engine.analyze_financial_provisions([])
        assert fa.total_financial_provision_eur == Decimal("0")

    # -- Timeline --

    def test_timeline_compliance(self, engine, sample_actions):
        ta = engine.assess_timeline_compliance(sample_actions)
        assert isinstance(ta, TimelineAnalysis)
        assert ta.total_actions == 2

    # -- Full assessment --

    def test_assess_remediation_returns_result(self, engine, sample_actions):
        result = engine.assess_remediation(sample_actions)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_remediation_empty(self, engine):
        result = engine.assess_remediation([])
        assert result.provenance_hash != ""

    def test_assess_remediation_processing_time(self, engine, sample_actions):
        result = engine.assess_remediation(sample_actions)
        assert result.processing_time_ms >= 0

    # -- Constants --

    def test_engagement_scores(self):
        assert ENGAGEMENT_SCORES["none"] == Decimal("0")
        assert ENGAGEMENT_SCORES["co_designed"] == Decimal("100")

    def test_contribution_weights(self):
        assert CONTRIBUTION_WEIGHTS["caused"] == Decimal("1.0")
        assert CONTRIBUTION_WEIGHTS["directly_linked"] == Decimal("0.25")
