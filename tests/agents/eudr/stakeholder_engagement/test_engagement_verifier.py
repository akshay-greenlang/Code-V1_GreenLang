# -*- coding: utf-8 -*-
"""
Unit tests for EngagementVerifier Engine - AGENT-EUDR-031

Tests engagement assessment, dimension scoring, composite score
calculation, and recommendation generation across all six
engagement quality dimensions.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.engagement_verifier import (
    EngagementVerifier,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    EngagementAssessment,
    EngagementDimension,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def verifier(config):
    return EngagementVerifier(config=config)


# ---------------------------------------------------------------------------
# Test: AssessEngagement
# ---------------------------------------------------------------------------

class TestAssessEngagement:
    """Test overall engagement assessment."""

    @pytest.mark.asyncio
    async def test_assess_engagement_success(self, verifier):
        """Test successful engagement assessment."""
        assessment = await verifier.assess_engagement(
            operator_id="OP-001",
            stakeholder_id="STK-IND-001",
            engagement_data={
                "consultations_held": 5,
                "grievances_resolved": 3,
                "fpic_status": "granted",
                "communication_frequency": "monthly",
                "cultural_practices_respected": True,
                "rights_impact_assessed": True,
            },
        )
        assert isinstance(assessment, EngagementAssessment)
        assert assessment.operator_id == "OP-001"
        assert assessment.stakeholder_id == "STK-IND-001"

    @pytest.mark.asyncio
    async def test_assess_engagement_returns_scores(self, verifier):
        """Test assessment returns dimension scores."""
        assessment = await verifier.assess_engagement(
            operator_id="OP-001",
            stakeholder_id="STK-IND-001",
            engagement_data={"consultations_held": 3, "grievances_resolved": 1},
        )
        assert len(assessment.dimension_scores) == 6
        for dim in EngagementDimension:
            assert dim in assessment.dimension_scores

    @pytest.mark.asyncio
    async def test_assess_engagement_composite_score(self, verifier):
        """Test assessment calculates composite score."""
        assessment = await verifier.assess_engagement(
            operator_id="OP-001",
            stakeholder_id="STK-001",
            engagement_data={"consultations_held": 10, "grievances_resolved": 5},
        )
        assert Decimal("0") <= assessment.composite_score <= Decimal("100")

    @pytest.mark.asyncio
    async def test_assess_engagement_missing_operator_raises(self, verifier):
        """Test assessment with missing operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await verifier.assess_engagement("", "STK-001", {})

    @pytest.mark.asyncio
    async def test_assess_engagement_missing_stakeholder_raises(self, verifier):
        """Test assessment with missing stakeholder raises error."""
        with pytest.raises(ValueError, match="stakeholder_id is required"):
            await verifier.assess_engagement("OP-001", "", {})

    @pytest.mark.asyncio
    async def test_assess_engagement_high_performance(self, verifier):
        """Test assessment for high-performing engagement."""
        assessment = await verifier.assess_engagement(
            operator_id="OP-001",
            stakeholder_id="STK-001",
            engagement_data={
                "consultations_held": 20,
                "grievances_resolved": 10,
                "grievance_satisfaction_avg": 95,
                "fpic_status": "granted",
                "communication_frequency": "weekly",
                "cultural_practices_respected": True,
                "rights_impact_assessed": True,
                "indigenous_protocols_followed": True,
            },
        )
        assert assessment.composite_score >= Decimal("70")

    @pytest.mark.asyncio
    async def test_assess_engagement_low_performance(self, verifier):
        """Test assessment for low-performing engagement."""
        assessment = await verifier.assess_engagement(
            operator_id="OP-001",
            stakeholder_id="STK-001",
            engagement_data={
                "consultations_held": 0,
                "grievances_resolved": 0,
                "grievances_total": 5,
                "fpic_status": "pending",
                "communication_frequency": "none",
            },
        )
        assert assessment.composite_score < Decimal("50")

    @pytest.mark.asyncio
    async def test_assess_engagement_sets_date(self, verifier):
        """Test assessment sets assessment date."""
        assessment = await verifier.assess_engagement(
            "OP-001", "STK-001", {"consultations_held": 1},
        )
        assert isinstance(assessment.assessment_date, datetime)


# ---------------------------------------------------------------------------
# Test: ScoreDimensions (all 6 dimensions)
# ---------------------------------------------------------------------------

class TestScoreDimensions:
    """Test individual dimension scoring."""

    def test_score_inclusiveness(self, verifier):
        """Test inclusiveness dimension scoring."""
        data = {
            "consultations_held": 5,
            "stakeholder_groups_represented": 4,
            "vulnerable_groups_included": True,
        }
        score = verifier.score_dimension(EngagementDimension.INCLUSIVENESS, data)
        assert Decimal("0") <= score <= Decimal("100")

    def test_score_inclusiveness_high(self, verifier):
        """Test high inclusiveness score."""
        data = {
            "consultations_held": 20,
            "stakeholder_groups_represented": 8,
            "vulnerable_groups_included": True,
            "women_represented": True,
            "youth_represented": True,
        }
        score = verifier.score_dimension(EngagementDimension.INCLUSIVENESS, data)
        assert score >= Decimal("60")

    def test_score_transparency(self, verifier):
        """Test transparency dimension scoring."""
        data = {
            "information_shared_publicly": True,
            "reports_published": 4,
            "languages_supported": 2,
        }
        score = verifier.score_dimension(EngagementDimension.TRANSPARENCY, data)
        assert Decimal("0") <= score <= Decimal("100")

    def test_score_transparency_low(self, verifier):
        """Test low transparency score."""
        data = {
            "information_shared_publicly": False,
            "reports_published": 0,
        }
        score = verifier.score_dimension(EngagementDimension.TRANSPARENCY, data)
        assert score < Decimal("50")

    def test_score_responsiveness(self, verifier):
        """Test responsiveness dimension scoring."""
        data = {
            "avg_grievance_response_hours": 24,
            "grievances_resolved_within_sla": 8,
            "grievances_total": 10,
        }
        score = verifier.score_dimension(EngagementDimension.RESPONSIVENESS, data)
        assert Decimal("0") <= score <= Decimal("100")

    def test_score_responsiveness_fast(self, verifier):
        """Test high responsiveness with fast response time."""
        data = {
            "avg_grievance_response_hours": 4,
            "grievances_resolved_within_sla": 10,
            "grievances_total": 10,
        }
        score = verifier.score_dimension(EngagementDimension.RESPONSIVENESS, data)
        assert score >= Decimal("70")

    def test_score_accountability(self, verifier):
        """Test accountability dimension scoring."""
        data = {
            "commitments_fulfilled": 8,
            "commitments_total": 10,
            "independent_audits": 2,
        }
        score = verifier.score_dimension(EngagementDimension.ACCOUNTABILITY, data)
        assert Decimal("0") <= score <= Decimal("100")

    def test_score_accountability_low(self, verifier):
        """Test low accountability score."""
        data = {
            "commitments_fulfilled": 1,
            "commitments_total": 10,
            "independent_audits": 0,
        }
        score = verifier.score_dimension(EngagementDimension.ACCOUNTABILITY, data)
        assert score < Decimal("40")

    def test_score_cultural_sensitivity(self, verifier):
        """Test cultural sensitivity dimension scoring."""
        data = {
            "cultural_practices_respected": True,
            "local_language_used": True,
            "indigenous_protocols_followed": True,
        }
        score = verifier.score_dimension(EngagementDimension.CULTURAL_SENSITIVITY, data)
        assert Decimal("0") <= score <= Decimal("100")

    def test_score_cultural_sensitivity_high(self, verifier):
        """Test high cultural sensitivity score."""
        data = {
            "cultural_practices_respected": True,
            "local_language_used": True,
            "indigenous_protocols_followed": True,
            "community_elders_consulted": True,
            "traditional_decision_making_respected": True,
        }
        score = verifier.score_dimension(EngagementDimension.CULTURAL_SENSITIVITY, data)
        assert score >= Decimal("70")

    def test_score_rights_respect(self, verifier):
        """Test rights respect dimension scoring."""
        data = {
            "fpic_obtained": True,
            "land_rights_respected": True,
            "rights_impact_assessed": True,
        }
        score = verifier.score_dimension(EngagementDimension.RIGHTS_RESPECT, data)
        assert Decimal("0") <= score <= Decimal("100")

    def test_score_rights_respect_no_fpic(self, verifier):
        """Test rights respect score without FPIC."""
        data = {
            "fpic_obtained": False,
            "fpic_required": True,
            "land_rights_respected": False,
        }
        score = verifier.score_dimension(EngagementDimension.RIGHTS_RESPECT, data)
        assert score < Decimal("30")


# ---------------------------------------------------------------------------
# Test: CalculateComposite
# ---------------------------------------------------------------------------

class TestCalculateComposite:
    """Test composite score calculation."""

    def test_calculate_composite_equal_scores(self, verifier):
        """Test composite with equal dimension scores."""
        scores = {dim: Decimal("75") for dim in EngagementDimension}
        composite = verifier.calculate_composite(scores)
        assert composite == Decimal("75")

    def test_calculate_composite_mixed_scores(self, verifier):
        """Test composite with mixed dimension scores."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("90"),
            EngagementDimension.TRANSPARENCY: Decimal("60"),
            EngagementDimension.RESPONSIVENESS: Decimal("80"),
            EngagementDimension.ACCOUNTABILITY: Decimal("70"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("85"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("95"),
        }
        composite = verifier.calculate_composite(scores)
        assert Decimal("60") <= composite <= Decimal("95")

    def test_calculate_composite_all_zeros(self, verifier):
        """Test composite with all zero scores."""
        scores = {dim: Decimal("0") for dim in EngagementDimension}
        composite = verifier.calculate_composite(scores)
        assert composite == Decimal("0")

    def test_calculate_composite_all_hundred(self, verifier):
        """Test composite with all perfect scores."""
        scores = {dim: Decimal("100") for dim in EngagementDimension}
        composite = verifier.calculate_composite(scores)
        assert composite == Decimal("100")

    def test_calculate_composite_returns_decimal(self, verifier):
        """Test composite returns Decimal."""
        scores = {dim: Decimal("50") for dim in EngagementDimension}
        composite = verifier.calculate_composite(scores)
        assert isinstance(composite, Decimal)

    def test_calculate_composite_bounded(self, verifier):
        """Test composite is bounded between 0 and 100."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("10"),
            EngagementDimension.TRANSPARENCY: Decimal("95"),
            EngagementDimension.RESPONSIVENESS: Decimal("5"),
            EngagementDimension.ACCOUNTABILITY: Decimal("99"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("50"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("75"),
        }
        composite = verifier.calculate_composite(scores)
        assert Decimal("0") <= composite <= Decimal("100")

    def test_calculate_composite_missing_dimension_raises(self, verifier):
        """Test composite with missing dimension raises error."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("50"),
            EngagementDimension.TRANSPARENCY: Decimal("50"),
            # Missing 4 dimensions
        }
        with pytest.raises(ValueError, match="all 6 dimensions required"):
            verifier.calculate_composite(scores)

    def test_calculate_composite_weighted(self, verifier):
        """Test composite respects dimension weights if applicable."""
        # Rights respect and cultural sensitivity may be weighted higher
        high_rights = {
            EngagementDimension.INCLUSIVENESS: Decimal("50"),
            EngagementDimension.TRANSPARENCY: Decimal("50"),
            EngagementDimension.RESPONSIVENESS: Decimal("50"),
            EngagementDimension.ACCOUNTABILITY: Decimal("50"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("100"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("100"),
        }
        low_rights = {
            EngagementDimension.INCLUSIVENESS: Decimal("100"),
            EngagementDimension.TRANSPARENCY: Decimal("100"),
            EngagementDimension.RESPONSIVENESS: Decimal("50"),
            EngagementDimension.ACCOUNTABILITY: Decimal("50"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("0"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("0"),
        }
        composite_hr = verifier.calculate_composite(high_rights)
        composite_lr = verifier.calculate_composite(low_rights)
        # Both should be valid composites
        assert Decimal("0") <= composite_hr <= Decimal("100")
        assert Decimal("0") <= composite_lr <= Decimal("100")


# ---------------------------------------------------------------------------
# Test: GenerateRecommendations
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    """Test recommendation generation based on assessment results."""

    def test_recommendations_for_low_inclusiveness(self, verifier):
        """Test recommendations generated for low inclusiveness."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("20"),
            EngagementDimension.TRANSPARENCY: Decimal("80"),
            EngagementDimension.RESPONSIVENESS: Decimal("80"),
            EngagementDimension.ACCOUNTABILITY: Decimal("80"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("80"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("80"),
        }
        recs = verifier.generate_recommendations(scores)
        assert isinstance(recs, list)
        assert len(recs) >= 1
        assert any("inclus" in r.lower() for r in recs)

    def test_recommendations_for_low_transparency(self, verifier):
        """Test recommendations generated for low transparency."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("80"),
            EngagementDimension.TRANSPARENCY: Decimal("15"),
            EngagementDimension.RESPONSIVENESS: Decimal("80"),
            EngagementDimension.ACCOUNTABILITY: Decimal("80"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("80"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("80"),
        }
        recs = verifier.generate_recommendations(scores)
        assert any("transparen" in r.lower() for r in recs)

    def test_recommendations_for_low_responsiveness(self, verifier):
        """Test recommendations generated for low responsiveness."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("80"),
            EngagementDimension.TRANSPARENCY: Decimal("80"),
            EngagementDimension.RESPONSIVENESS: Decimal("10"),
            EngagementDimension.ACCOUNTABILITY: Decimal("80"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("80"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("80"),
        }
        recs = verifier.generate_recommendations(scores)
        assert any("respons" in r.lower() for r in recs)

    def test_recommendations_for_low_accountability(self, verifier):
        """Test recommendations generated for low accountability."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("80"),
            EngagementDimension.TRANSPARENCY: Decimal("80"),
            EngagementDimension.RESPONSIVENESS: Decimal("80"),
            EngagementDimension.ACCOUNTABILITY: Decimal("15"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("80"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("80"),
        }
        recs = verifier.generate_recommendations(scores)
        assert any("accountab" in r.lower() for r in recs)

    def test_recommendations_for_low_cultural_sensitivity(self, verifier):
        """Test recommendations generated for low cultural sensitivity."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("80"),
            EngagementDimension.TRANSPARENCY: Decimal("80"),
            EngagementDimension.RESPONSIVENESS: Decimal("80"),
            EngagementDimension.ACCOUNTABILITY: Decimal("80"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("10"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("80"),
        }
        recs = verifier.generate_recommendations(scores)
        assert any("cultur" in r.lower() for r in recs)

    def test_recommendations_for_low_rights_respect(self, verifier):
        """Test recommendations generated for low rights respect."""
        scores = {
            EngagementDimension.INCLUSIVENESS: Decimal("80"),
            EngagementDimension.TRANSPARENCY: Decimal("80"),
            EngagementDimension.RESPONSIVENESS: Decimal("80"),
            EngagementDimension.ACCOUNTABILITY: Decimal("80"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("80"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("10"),
        }
        recs = verifier.generate_recommendations(scores)
        assert any("right" in r.lower() for r in recs)

    def test_recommendations_all_high_scores(self, verifier):
        """Test no critical recommendations for all high scores."""
        scores = {dim: Decimal("95") for dim in EngagementDimension}
        recs = verifier.generate_recommendations(scores)
        assert isinstance(recs, list)
        # High scores should yield fewer or no recommendations
        assert len(recs) <= 2

    def test_recommendations_all_low_scores(self, verifier):
        """Test many recommendations for all low scores."""
        scores = {dim: Decimal("10") for dim in EngagementDimension}
        recs = verifier.generate_recommendations(scores)
        assert len(recs) >= 6  # At least one per dimension

    def test_recommendations_returns_list(self, verifier):
        """Test recommendations returns a list."""
        scores = {dim: Decimal("50") for dim in EngagementDimension}
        recs = verifier.generate_recommendations(scores)
        assert isinstance(recs, list)

    def test_recommendations_are_strings(self, verifier):
        """Test each recommendation is a string."""
        scores = {dim: Decimal("30") for dim in EngagementDimension}
        recs = verifier.generate_recommendations(scores)
        for rec in recs:
            assert isinstance(rec, str)
            assert len(rec) > 10  # Meaningful recommendation

    def test_recommendations_empty_scores_raises(self, verifier):
        """Test recommendations with empty scores raises error."""
        with pytest.raises(ValueError):
            verifier.generate_recommendations({})

    def test_recommendations_border_threshold(self, verifier):
        """Test recommendations at passing threshold border."""
        scores = {dim: Decimal("60") for dim in EngagementDimension}
        recs = verifier.generate_recommendations(scores)
        assert isinstance(recs, list)
