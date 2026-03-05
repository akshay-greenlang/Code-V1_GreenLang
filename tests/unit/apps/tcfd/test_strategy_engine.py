# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Strategy Engine.

Tests risk identification by sector, opportunity identification, time
horizon categorization, business model impact assessment, value chain
mapping, and strategic response with 28+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    RiskType,
    OpportunityCategory,
    TimeHorizon,
    RiskLikelihood,
    RiskImpact,
    SectorType,
    SECTOR_TRANSITION_PROFILES,
    TIME_HORIZON_YEARS,
    LIKELIHOOD_SCORES,
    IMPACT_SCORES,
    RISK_MATRIX_THRESHOLDS,
    TCFD_DISCLOSURES,
)
from services.models import (
    ClimateRisk,
    ClimateOpportunity,
    _new_id,
)


# ===========================================================================
# Risk Identification by Sector
# ===========================================================================

class TestRiskIdentificationBySector:
    """Test climate risk identification by sector."""

    def test_energy_sector_high_transition_exposure(self):
        profile = SECTOR_TRANSITION_PROFILES[SectorType.ENERGY]
        assert profile["transition_exposure"] == "very_high"

    def test_technology_sector_low_transition_exposure(self):
        profile = SECTOR_TRANSITION_PROFILES[SectorType.TECHNOLOGY]
        assert profile["transition_exposure"] == "low"

    @pytest.mark.parametrize("sector", list(SectorType))
    def test_all_sectors_have_profiles(self, sector):
        assert sector in SECTOR_TRANSITION_PROFILES
        profile = SECTOR_TRANSITION_PROFILES[sector]
        assert "transition_exposure" in profile
        assert "stranding_risk" in profile
        assert "key_drivers" in profile

    def test_physical_risk_creation(self, sample_climate_risk):
        assert sample_climate_risk.risk_type == RiskType.PHYSICAL_ACUTE
        assert sample_climate_risk.name == "Coastal Flooding"

    def test_transition_risk_creation(self, sample_transition_risk):
        assert sample_transition_risk.risk_type == RiskType.TRANSITION_POLICY
        assert sample_transition_risk.name == "Carbon Tax"

    @pytest.mark.parametrize("risk_type", list(RiskType))
    def test_all_risk_types_valid(self, risk_type):
        risk = ClimateRisk(
            org_id=_new_id(),
            risk_type=risk_type,
            name=f"Test {risk_type.value}",
        )
        assert risk.risk_type == risk_type


# ===========================================================================
# Opportunity Identification
# ===========================================================================

class TestOpportunityIdentification:
    """Test climate opportunity identification."""

    def test_energy_source_opportunity(self, sample_climate_opportunity):
        assert sample_climate_opportunity.category == OpportunityCategory.ENERGY_SOURCE

    @pytest.mark.parametrize("category", list(OpportunityCategory))
    def test_all_opportunity_categories(self, category):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=category,
            name=f"Test {category.value}",
        )
        assert opp.category == category

    def test_opportunity_financial_sizing(self, sample_climate_opportunity):
        assert sample_climate_opportunity.revenue_potential == Decimal("100000000")
        assert sample_climate_opportunity.cost_savings == Decimal("15000000")
        assert sample_climate_opportunity.investment_required == Decimal("50000000")

    def test_opportunity_roi_positive(self, sample_climate_opportunity):
        assert sample_climate_opportunity.roi_estimate > Decimal("0")


# ===========================================================================
# Time Horizon Categorization
# ===========================================================================

class TestTimeHorizonCategorization:
    """Test time horizon categorization."""

    def test_short_term_range(self):
        st = TIME_HORIZON_YEARS[TimeHorizon.SHORT_TERM]
        assert st["min_years"] == 0
        assert st["max_years"] == 3

    def test_medium_term_range(self):
        mt = TIME_HORIZON_YEARS[TimeHorizon.MEDIUM_TERM]
        assert mt["min_years"] == 3
        assert mt["max_years"] == 10

    def test_long_term_range(self):
        lt = TIME_HORIZON_YEARS[TimeHorizon.LONG_TERM]
        assert lt["min_years"] == 10
        assert lt["max_years"] == 30

    def test_risk_time_horizon_assignment(self, sample_climate_risk):
        assert sample_climate_risk.time_horizon == TimeHorizon.MEDIUM_TERM

    def test_transition_risk_short_term(self, sample_transition_risk):
        assert sample_transition_risk.time_horizon == TimeHorizon.SHORT_TERM


# ===========================================================================
# Business Model Impact
# ===========================================================================

class TestBusinessModelImpact:
    """Test business model impact assessment."""

    def test_financial_impact_estimate(self, sample_climate_risk):
        assert sample_climate_risk.financial_impact_estimate == Decimal("25000000")

    def test_zero_financial_impact(self):
        risk = ClimateRisk(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_REPUTATION,
            name="Brand Risk",
            financial_impact_estimate=Decimal("0"),
        )
        assert risk.financial_impact_estimate == Decimal("0")

    def test_high_financial_impact(self):
        risk = ClimateRisk(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_POLICY,
            name="Carbon Tax",
            financial_impact_estimate=Decimal("1000000000"),
        )
        assert risk.financial_impact_estimate == Decimal("1000000000")

    def test_strategy_disclosures_defined(self):
        assert "str_a" in TCFD_DISCLOSURES
        assert "str_b" in TCFD_DISCLOSURES
        assert "str_c" in TCFD_DISCLOSURES


# ===========================================================================
# Value Chain Mapping
# ===========================================================================

class TestValueChainMapping:
    """Test value chain impact mapping."""

    def test_affected_assets(self, sample_climate_risk):
        assert len(sample_climate_risk.affected_assets) >= 1

    def test_empty_affected_assets(self):
        risk = ClimateRisk(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_MARKET,
            name="Market Shift",
        )
        assert risk.affected_assets == []


# ===========================================================================
# Strategic Response
# ===========================================================================

class TestStrategicResponse:
    """Test strategic response definition."""

    def test_response_strategy(self, sample_climate_risk):
        assert len(sample_climate_risk.response_strategy) > 0

    def test_risk_status_active(self, sample_climate_risk):
        assert sample_climate_risk.status == "active"

    def test_risk_owner(self, sample_climate_risk):
        assert sample_climate_risk.owner == "VP Operations"


# ===========================================================================
# Risk Scoring
# ===========================================================================

class TestRiskScoring:
    """Test likelihood x impact risk scoring."""

    @pytest.mark.parametrize("likelihood,expected_score", [
        (RiskLikelihood.RARE, 1),
        (RiskLikelihood.UNLIKELY, 2),
        (RiskLikelihood.POSSIBLE, 3),
        (RiskLikelihood.LIKELY, 4),
        (RiskLikelihood.ALMOST_CERTAIN, 5),
    ])
    def test_likelihood_scores(self, likelihood, expected_score):
        assert LIKELIHOOD_SCORES[likelihood] == expected_score

    @pytest.mark.parametrize("impact,expected_score", [
        (RiskImpact.INSIGNIFICANT, 1),
        (RiskImpact.MINOR, 2),
        (RiskImpact.MODERATE, 3),
        (RiskImpact.MAJOR, 4),
        (RiskImpact.CATASTROPHIC, 5),
    ])
    def test_impact_scores(self, impact, expected_score):
        assert IMPACT_SCORES[impact] == expected_score

    def test_risk_matrix_thresholds_complete(self):
        assert "low" in RISK_MATRIX_THRESHOLDS
        assert "medium" in RISK_MATRIX_THRESHOLDS
        assert "high" in RISK_MATRIX_THRESHOLDS
        assert "critical" in RISK_MATRIX_THRESHOLDS

    def test_critical_threshold(self):
        critical = RISK_MATRIX_THRESHOLDS["critical"]
        assert critical["min"] == 20
        assert critical["max"] == 25
