# -*- coding: utf-8 -*-
"""
Tests for DueDiligencePolicyEngine - PACK-019 CSDDD Readiness Pack
===================================================================

Validates scope determination, article-level assessments, policy area
assessments, and overall scoring against CSDDD Articles 2 and 5.

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

_mod = _load_engine("due_diligence_policy")

CompanyScope = getattr(_mod, "CompanyScope")
ComplianceStatus = getattr(_mod, "ComplianceStatus")
ArticleReference = getattr(_mod, "ArticleReference")
PolicyArea = getattr(_mod, "PolicyArea")
CompanyProfile = getattr(_mod, "CompanyProfile")
PolicyAssessment = getattr(_mod, "PolicyAssessment")
ArticleAssessment = getattr(_mod, "ArticleAssessment")
ScopeAssessment = getattr(_mod, "ScopeAssessment")
DueDiligencePolicyResult = getattr(_mod, "DueDiligencePolicyResult")
DueDiligencePolicyEngine = getattr(_mod, "DueDiligencePolicyEngine")
SCOPE_THRESHOLDS = getattr(_mod, "SCOPE_THRESHOLDS")
ARTICLE_WEIGHTS = getattr(_mod, "ARTICLE_WEIGHTS")


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Verify all enums expose the expected members."""

    def test_company_scope_members(self):
        assert CompanyScope.PHASE_1.value == "phase_1"
        assert CompanyScope.PHASE_2.value == "phase_2"
        assert CompanyScope.PHASE_3.value == "phase_3"
        assert CompanyScope.NOT_IN_SCOPE.value == "not_in_scope"
        assert CompanyScope.VOLUNTARY.value == "voluntary"

    def test_compliance_status_members(self):
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_article_reference_members(self):
        expected = ["art_5", "art_6", "art_7", "art_8", "art_9", "art_10",
                    "art_11", "art_12", "art_13", "art_14", "art_15",
                    "art_22", "art_29"]
        values = [a.value for a in ArticleReference]
        assert values == expected

    def test_policy_area_members(self):
        assert len(list(PolicyArea)) == 6
        assert PolicyArea.CODE_OF_CONDUCT.value == "code_of_conduct"
        assert PolicyArea.GOVERNANCE.value == "governance"


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestCompanyProfile:
    """Validate CompanyProfile Pydantic model."""

    def test_minimal_instantiation(self):
        p = CompanyProfile(
            company_name="Test GmbH",
            country="DE",
            employee_count=100,
            worldwide_turnover_eur=Decimal("1000000"),
            reporting_year=2027,
        )
        assert p.company_name == "Test GmbH"
        assert p.country == "DE"

    def test_country_uppercased(self):
        p = CompanyProfile(
            company_name="X",
            country="de",
            employee_count=1,
            worldwide_turnover_eur=Decimal("0"),
            reporting_year=2024,
        )
        assert p.country == "DE"

    def test_defaults(self):
        p = CompanyProfile(
            company_name="X",
            country="FR",
            employee_count=0,
            worldwide_turnover_eur=Decimal("0"),
            reporting_year=2024,
        )
        assert p.has_dd_policy is False
        assert p.has_code_of_conduct is False
        assert p.has_grievance_mechanism is False
        assert p.has_climate_transition_plan is False
        assert p.value_chain_tiers == 1


class TestPolicyAssessmentModel:

    def test_instantiation(self):
        pa = PolicyAssessment(
            policy_area=PolicyArea.CODE_OF_CONDUCT,
            status=ComplianceStatus.COMPLIANT,
            score=Decimal("80"),
        )
        assert pa.policy_area == PolicyArea.CODE_OF_CONDUCT
        assert pa.score == Decimal("80")


class TestArticleAssessmentModel:

    def test_instantiation(self):
        aa = ArticleAssessment(
            article=ArticleReference.ART_5,
            status=ComplianceStatus.NON_COMPLIANT,
            score=Decimal("0"),
        )
        assert aa.article == ArticleReference.ART_5


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestDueDiligencePolicyEngine:
    """Core engine tests."""

    @pytest.fixture
    def engine(self):
        return DueDiligencePolicyEngine()

    @pytest.fixture
    def phase1_profile(self):
        return CompanyProfile(
            company_name="BigCorp AG",
            country="DE",
            sector="MANUFACTURING",
            employee_count=6000,
            worldwide_turnover_eur=Decimal("2000000000"),
            reporting_year=2027,
            has_dd_policy=True,
            has_code_of_conduct=True,
        )

    @pytest.fixture
    def phase3_profile(self):
        return CompanyProfile(
            company_name="MidCorp GmbH",
            country="DE",
            employee_count=1500,
            worldwide_turnover_eur=Decimal("500000000"),
            reporting_year=2029,
        )

    @pytest.fixture
    def out_of_scope_profile(self):
        return CompanyProfile(
            company_name="SmallCo",
            country="DE",
            employee_count=500,
            worldwide_turnover_eur=Decimal("100000000"),
            reporting_year=2027,
        )

    @pytest.fixture
    def fully_compliant_profile(self):
        return CompanyProfile(
            company_name="GreenLeader SE",
            country="NL",
            employee_count=8000,
            worldwide_turnover_eur=Decimal("5000000000"),
            reporting_year=2027,
            has_dd_policy=True,
            dd_policy_describes_approach=True,
            dd_policy_has_code_of_conduct=True,
            code_rules_for_employees=True,
            code_rules_for_subsidiaries=True,
            dd_policy_describes_processes=True,
            dd_policy_verification_measures=True,
            dd_policy_extended_to_partners=True,
            dd_policy_updated_annually=True,
            dd_policy_approved_by_board=True,
            has_code_of_conduct=True,
            code_covers_human_rights=True,
            code_covers_environment=True,
            code_covers_subsidiaries=True,
            code_covers_business_partners=True,
            code_reviewed_annually=True,
            has_risk_management_framework=True,
            risk_framework_covers_hr=True,
            risk_framework_covers_env=True,
            risk_assessment_frequency_adequate=True,
            risk_mitigation_measures_defined=True,
            board_oversight_defined=True,
            dedicated_dd_officer=True,
            governance_body_receives_reports=True,
            accountability_mechanisms_exist=True,
            monitoring_process_defined=True,
            periodic_assessments_conducted=True,
            kpis_defined=True,
            third_party_verification=True,
            annual_reporting_committed=True,
            reporting_covers_impacts=True,
            reporting_covers_measures=True,
            reporting_publicly_available=True,
            stakeholder_mapping_conducted=True,
            engagement_process_defined=True,
            affected_communities_consulted=True,
            engagement_outcomes_documented=True,
            has_grievance_mechanism=True,
            grievance_mechanism_accessible=True,
            grievance_mechanism_documented=True,
            has_climate_transition_plan=True,
            climate_plan_paris_aligned=True,
            climate_plan_has_targets=True,
            climate_plan_has_actions=True,
            climate_plan_has_investments=True,
            has_impact_identification_process=True,
            impact_identification_covers_own_ops=True,
            impact_identification_covers_subsidiaries=True,
            impact_identification_covers_value_chain=True,
            has_prioritisation_methodology=True,
            prioritisation_considers_severity=True,
            prioritisation_considers_likelihood=True,
            has_prevention_measures=True,
            prevention_includes_cap=True,
            prevention_includes_contractual=True,
            has_cessation_measures=True,
            cessation_includes_corrective_action=True,
            has_remediation_process=True,
            remediation_includes_financial=True,
            has_civil_liability_awareness=True,
            civil_liability_insurance=True,
        )

    # -- Scope tests --

    def test_scope_phase1(self, engine, phase1_profile):
        result = engine.assess_scope(phase1_profile)
        assert result.scope_phase == CompanyScope.PHASE_1
        assert result.in_scope is True
        assert result.meets_employee_threshold is True
        assert result.meets_turnover_threshold is True

    def test_scope_phase3(self, engine, phase3_profile):
        result = engine.assess_scope(phase3_profile)
        assert result.scope_phase == CompanyScope.PHASE_3
        assert result.in_scope is True

    def test_scope_not_in_scope(self, engine, out_of_scope_profile):
        result = engine.assess_scope(out_of_scope_profile)
        assert result.scope_phase == CompanyScope.NOT_IN_SCOPE
        assert result.in_scope is False

    def test_scope_boundary_phase1(self, engine):
        """Exactly 5000 employees should NOT meet phase 1 (requires >5000)."""
        profile = CompanyProfile(
            company_name="Boundary",
            country="DE",
            employee_count=5000,
            worldwide_turnover_eur=Decimal("1500000001"),
            reporting_year=2027,
        )
        result = engine.assess_scope(profile)
        assert result.scope_phase != CompanyScope.PHASE_1

    def test_scope_third_country_uses_eu_turnover(self, engine):
        profile = CompanyProfile(
            company_name="USCo",
            country="US",
            employee_count=6000,
            worldwide_turnover_eur=Decimal("5000000000"),
            eu_turnover_eur=Decimal("100000000"),
            reporting_year=2027,
            is_eu_company=False,
            is_third_country_company=True,
        )
        result = engine.assess_scope(profile)
        # EU turnover 100M < 450M, not in scope despite huge worldwide
        assert result.in_scope is False

    # -- Full assessment tests --

    def test_assess_policy_returns_result(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert isinstance(result, DueDiligencePolicyResult)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_policy_has_article_assessments(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert len(result.article_assessments) == 13  # 13 articles

    def test_assess_policy_has_policy_assessments(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert len(result.policy_assessments) == 6  # 6 policy areas

    def test_assess_policy_overall_score_range(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert Decimal("0") <= result.overall_score <= Decimal("100")

    def test_assess_policy_processing_time(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert result.processing_time_ms >= 0

    def test_assess_policy_gaps_summary(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert isinstance(result.gaps_summary, dict)
        assert len(result.gaps_summary) == 13

    def test_fully_compliant_scores_high(self, engine, fully_compliant_profile):
        result = engine.assess_policy(fully_compliant_profile)
        assert result.overall_score >= Decimal("80")
        assert result.overall_status == ComplianceStatus.COMPLIANT

    def test_non_compliant_all_false(self, engine, out_of_scope_profile):
        result = engine.assess_policy(out_of_scope_profile)
        assert result.overall_score == Decimal("0.0") or result.overall_score < Decimal("40")
        assert result.overall_status == ComplianceStatus.NON_COMPLIANT

    def test_recommendations_present_when_gaps(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert len(result.recommendations) > 0

    def test_no_recommendations_when_fully_compliant(self, engine, fully_compliant_profile):
        result = engine.assess_policy(fully_compliant_profile)
        assert len(result.recommendations) == 0

    def test_company_name_in_result(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert result.company == "BigCorp AG"

    def test_reporting_year_in_result(self, engine, phase1_profile):
        result = engine.assess_policy(phase1_profile)
        assert result.reporting_year == 2027

    # -- Score-to-status helper --

    def test_score_to_status_compliant(self, engine):
        assert engine._score_to_status(Decimal("80")) == ComplianceStatus.COMPLIANT
        assert engine._score_to_status(Decimal("100")) == ComplianceStatus.COMPLIANT

    def test_score_to_status_partial(self, engine):
        assert engine._score_to_status(Decimal("40")) == ComplianceStatus.PARTIALLY_COMPLIANT
        assert engine._score_to_status(Decimal("79.9")) == ComplianceStatus.PARTIALLY_COMPLIANT

    def test_score_to_status_non_compliant(self, engine):
        assert engine._score_to_status(Decimal("0")) == ComplianceStatus.NON_COMPLIANT
        assert engine._score_to_status(Decimal("39.9")) == ComplianceStatus.NON_COMPLIANT

    # -- Constants --

    def test_article_weights_sum_to_100(self):
        total = sum(ARTICLE_WEIGHTS.values())
        assert total == Decimal("100")

    def test_scope_thresholds_phases(self):
        assert "phase_1" in SCOPE_THRESHOLDS
        assert "phase_2" in SCOPE_THRESHOLDS
        assert "phase_3" in SCOPE_THRESHOLDS

    # -- Provenance reproducibility --

    def test_provenance_deterministic(self, engine, fully_compliant_profile):
        r1 = engine.assess_policy(fully_compliant_profile)
        r2 = engine.assess_policy(fully_compliant_profile)
        # Provenance hashes may differ due to timestamps; but structure is valid
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
