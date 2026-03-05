# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Governance Engine.

Tests board oversight assessment, management role tracking, maturity
scoring across 8 dimensions, competency matrix evaluation, incentive
linkage validation, and governance disclosure text generation
with 28+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from services.config import (
    MaturityLevel,
    MATURITY_SCORES,
    TCFDPillar,
    TCFD_DISCLOSURES,
    PILLAR_NAMES,
)
from services.models import (
    GovernanceAssessment,
    GovernanceRole,
    _new_id,
    _sha256,
)


# ===========================================================================
# Board Oversight Assessment
# ===========================================================================

class TestBoardOversightAssessment:
    """Test board oversight evaluation."""

    def test_board_oversight_score_range(self, sample_governance_assessment):
        assert 1 <= sample_governance_assessment.board_oversight_score <= 5

    def test_board_committees_tracked(self, sample_governance_assessment):
        assert len(sample_governance_assessment.board_committees) >= 1
        assert "Sustainability Committee" in sample_governance_assessment.board_committees

    def test_meeting_frequency_positive(self, sample_governance_assessment):
        assert sample_governance_assessment.meeting_frequency > 0

    def test_high_oversight_score(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            board_oversight_score=5,
            board_committees=["Sustainability", "Audit", "Risk"],
            meeting_frequency=12,
            climate_competency_score=5,
            incentive_linkage=True,
            incentive_pct=Decimal("25.0"),
            overall_maturity=MaturityLevel.OPTIMIZED,
        )
        assert assessment.board_oversight_score == 5
        assert assessment.overall_maturity == MaturityLevel.OPTIMIZED

    def test_low_oversight_score(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            board_oversight_score=1,
            meeting_frequency=0,
            climate_competency_score=1,
            incentive_linkage=False,
            overall_maturity=MaturityLevel.INITIAL,
        )
        assert assessment.board_oversight_score == 1
        assert assessment.overall_maturity == MaturityLevel.INITIAL

    def test_board_oversight_score_boundary_low(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            board_oversight_score=1,
        )
        assert assessment.board_oversight_score == 1

    def test_board_oversight_score_boundary_high(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            board_oversight_score=5,
        )
        assert assessment.board_oversight_score == 5


# ===========================================================================
# Management Role Tracking
# ===========================================================================

class TestManagementRoleTracking:
    """Test management climate role tracking."""

    def test_role_creation(self, sample_governance_role):
        assert sample_governance_role.role_title == "Chief Sustainability Officer"
        assert sample_governance_role.climate_accountability is True

    def test_role_reporting_line(self, sample_governance_role):
        assert sample_governance_role.reporting_line == "CEO"

    def test_role_competency_areas(self, sample_governance_role):
        assert "climate science" in sample_governance_role.competency_areas

    def test_multiple_management_roles(self, sample_governance_assessment):
        roles = sample_governance_assessment.management_roles
        assert isinstance(roles, list)

    def test_role_without_accountability(self):
        role = GovernanceRole(
            org_id=_new_id(),
            role_title="External Auditor",
            climate_accountability=False,
        )
        assert role.climate_accountability is False

    def test_role_with_all_fields(self):
        role = GovernanceRole(
            org_id=_new_id(),
            role_title="VP Sustainability",
            person_name="John Doe",
            responsibility_description="Lead climate strategy implementation",
            climate_accountability=True,
            reporting_line="CSO",
            competency_areas=["GHG accounting", "scenario analysis", "TCFD reporting"],
        )
        assert role.person_name == "John Doe"
        assert len(role.competency_areas) == 3


# ===========================================================================
# Maturity Scoring
# ===========================================================================

class TestMaturityScoring:
    """Test governance maturity scoring across all dimensions."""

    def test_maturity_level_mapping(self):
        assert MATURITY_SCORES[MaturityLevel.INITIAL] == 1
        assert MATURITY_SCORES[MaturityLevel.OPTIMIZED] == 5

    def test_maturity_scores_eight_dimensions(self, sample_governance_assessment):
        scores = sample_governance_assessment.maturity_scores
        expected_dims = [
            "board_oversight", "management_roles", "climate_competency",
            "meeting_frequency", "reporting_structure", "incentive_alignment",
            "risk_integration", "strategy_integration",
        ]
        for dim in expected_dims:
            assert dim in scores
            assert 1 <= scores[dim] <= 5

    def test_overall_maturity_from_scores(self, sample_governance_assessment):
        scores = sample_governance_assessment.maturity_scores
        avg = sum(scores.values()) / len(scores)
        assert avg >= 2  # DEVELOPING or above

    @pytest.mark.parametrize("level,expected_score", [
        (MaturityLevel.INITIAL, 1),
        (MaturityLevel.DEVELOPING, 2),
        (MaturityLevel.DEFINED, 3),
        (MaturityLevel.MANAGED, 4),
        (MaturityLevel.OPTIMIZED, 5),
    ])
    def test_maturity_level_numeric(self, level, expected_score):
        assert MATURITY_SCORES[level] == expected_score


# ===========================================================================
# Competency Matrix
# ===========================================================================

class TestCompetencyMatrix:
    """Test climate competency matrix evaluation."""

    def test_competency_score_range(self, sample_governance_assessment):
        assert 1 <= sample_governance_assessment.climate_competency_score <= 5

    def test_competency_areas_populated(self, sample_governance_role):
        assert len(sample_governance_role.competency_areas) > 0

    def test_competency_score_high(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            climate_competency_score=5,
        )
        assert assessment.climate_competency_score == 5

    def test_competency_score_low(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            climate_competency_score=1,
        )
        assert assessment.climate_competency_score == 1


# ===========================================================================
# Incentive Linkage
# ===========================================================================

class TestIncentiveLinkage:
    """Test incentive linkage tracking."""

    def test_incentive_linked(self, sample_governance_assessment):
        assert sample_governance_assessment.incentive_linkage is True
        assert sample_governance_assessment.incentive_pct == Decimal("15.0")

    def test_no_incentive_linkage(self):
        assessment = GovernanceAssessment(
            org_id=_new_id(),
            incentive_linkage=False,
            incentive_pct=Decimal("0"),
        )
        assert assessment.incentive_linkage is False
        assert assessment.incentive_pct == Decimal("0")


# ===========================================================================
# Disclosure Generation
# ===========================================================================

class TestGovernanceDisclosureGeneration:
    """Test governance disclosure text generation."""

    def test_governance_pillar_exists(self):
        assert TCFDPillar.GOVERNANCE.value == "governance"

    def test_governance_disclosures_defined(self):
        assert "gov_a" in TCFD_DISCLOSURES
        assert "gov_b" in TCFD_DISCLOSURES

    def test_gov_a_board_oversight(self):
        disclosure = TCFD_DISCLOSURES["gov_a"]
        assert disclosure["pillar"] == "governance"
        assert "board" in disclosure["title"].lower() or "board" in disclosure["description"].lower()

    def test_gov_b_management_role(self):
        disclosure = TCFD_DISCLOSURES["gov_b"]
        assert disclosure["pillar"] == "governance"
        assert "management" in disclosure["description"].lower()

    def test_pillar_display_name(self):
        assert PILLAR_NAMES[TCFDPillar.GOVERNANCE] == "Governance"

    def test_assessment_provenance_deterministic(self):
        org_id = _new_id()
        a1 = GovernanceAssessment(
            org_id=org_id,
            assessment_date=date(2025, 12, 31),
            board_oversight_score=4,
            climate_competency_score=3,
        )
        a2 = GovernanceAssessment(
            org_id=org_id,
            assessment_date=date(2025, 12, 31),
            board_oversight_score=4,
            climate_competency_score=3,
        )
        assert a1.provenance_hash == a2.provenance_hash
