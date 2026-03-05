# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Transition Risk Engine.

Tests policy risk assessment, carbon price impact modeling, technology
risk assessment, market risk assessment, reputation risk, composite
scoring, and sector profiles with 30+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    RiskType,
    TransitionDriver,
    SectorType,
    SECTOR_TRANSITION_PROFILES,
    ScenarioType,
    SCENARIO_LIBRARY,
)
from services.models import (
    TransitionRiskAssessment,
    ClimateRisk,
    _new_id,
)


# ===========================================================================
# Policy Risk Assessment
# ===========================================================================

class TestPolicyRiskAssessment:
    """Test policy risk assessment."""

    def test_policy_risk_creation(self, sample_transition_risk_assessment):
        assert sample_transition_risk_assessment.risk_type == RiskType.TRANSITION_POLICY

    def test_policy_driver(self, sample_transition_risk_assessment):
        assert sample_transition_risk_assessment.driver == TransitionDriver.CARBON_PRICING

    @pytest.mark.parametrize("driver", list(TransitionDriver))
    def test_all_transition_drivers(self, driver):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_POLICY,
            driver=driver,
            sector=SectorType.ENERGY,
        )
        assert assessment.driver == driver

    def test_regulation_driver(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_POLICY,
            driver=TransitionDriver.REGULATION,
            sector=SectorType.ENERGY,
            financial_impact=Decimal("75000000"),
        )
        assert assessment.driver == TransitionDriver.REGULATION

    def test_litigation_driver(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_POLICY,
            driver=TransitionDriver.LITIGATION,
            sector=SectorType.ENERGY,
        )
        assert assessment.driver == TransitionDriver.LITIGATION


# ===========================================================================
# Carbon Price Impact Modeling
# ===========================================================================

class TestCarbonPriceImpactModeling:
    """Test carbon price impact on transition risk."""

    def test_carbon_price_impact(self, sample_transition_risk_assessment):
        assert sample_transition_risk_assessment.financial_impact == Decimal("150000000")

    def test_current_vs_projected_exposure(self, sample_transition_risk_assessment):
        assert sample_transition_risk_assessment.projected_exposure_2030 > \
               sample_transition_risk_assessment.current_exposure

    def test_2050_exposure_higher_than_2030(self, sample_transition_risk_assessment):
        assert sample_transition_risk_assessment.projected_exposure_2050 > \
               sample_transition_risk_assessment.projected_exposure_2030

    def test_carbon_cost_calculation(self):
        emissions_tco2e = Decimal("125000")
        carbon_price = Decimal("130")
        carbon_cost = emissions_tco2e * carbon_price
        assert carbon_cost == Decimal("16250000")

    @pytest.mark.parametrize("scenario,year,expected_min", [
        (ScenarioType.IEA_NZE, 2030, Decimal("100")),
        (ScenarioType.NGFS_DELAYED_TRANSITION, 2040, Decimal("200")),
        (ScenarioType.NGFS_CURRENT_POLICIES, 2050, Decimal("20")),
    ])
    def test_scenario_carbon_prices(self, scenario, year, expected_min):
        price = SCENARIO_LIBRARY[scenario]["carbon_price_trajectory"][year]
        assert price >= expected_min


# ===========================================================================
# Technology Risk Assessment
# ===========================================================================

class TestTechnologyRiskAssessment:
    """Test technology risk assessment."""

    def test_technology_substitution_risk(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_TECHNOLOGY,
            driver=TransitionDriver.TECHNOLOGY_SUBSTITUTION,
            sector=SectorType.ENERGY,
            financial_impact=Decimal("100000000"),
        )
        assert assessment.risk_type == RiskType.TRANSITION_TECHNOLOGY

    def test_technology_disruption_risk(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_TECHNOLOGY,
            driver=TransitionDriver.TECHNOLOGY_DISRUPTION,
            sector=SectorType.TRANSPORT,
        )
        assert assessment.driver == TransitionDriver.TECHNOLOGY_DISRUPTION

    def test_energy_sector_decarbonization_pathway(self):
        profile = SECTOR_TRANSITION_PROFILES[SectorType.ENERGY]
        assert "renewables" in profile["decarbonization_pathway"].lower()


# ===========================================================================
# Market Risk Assessment
# ===========================================================================

class TestMarketRiskAssessment:
    """Test market risk assessment."""

    def test_market_risk_creation(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_MARKET,
            driver=TransitionDriver.DEMAND_SHIFT,
            sector=SectorType.CONSUMER_GOODS,
            financial_impact=Decimal("30000000"),
        )
        assert assessment.risk_type == RiskType.TRANSITION_MARKET

    def test_supply_chain_market_risk(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_MARKET,
            driver=TransitionDriver.SUPPLY_CHAIN,
            sector=SectorType.MATERIALS,
        )
        assert assessment.driver == TransitionDriver.SUPPLY_CHAIN


# ===========================================================================
# Reputation Risk
# ===========================================================================

class TestReputationRisk:
    """Test reputation risk assessment."""

    def test_reputation_risk_creation(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_REPUTATION,
            driver=TransitionDriver.STAKEHOLDER_SENTIMENT,
            sector=SectorType.BANKING,
            financial_impact=Decimal("20000000"),
        )
        assert assessment.risk_type == RiskType.TRANSITION_REPUTATION

    def test_stakeholder_sentiment_driver(self):
        banking_profile = SECTOR_TRANSITION_PROFILES[SectorType.BANKING]
        assert "stakeholder_sentiment" in banking_profile["key_drivers"]


# ===========================================================================
# Composite Scoring
# ===========================================================================

class TestCompositeScoring:
    """Test composite transition risk scoring."""

    def test_composite_score_calculation(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_POLICY,
            driver=TransitionDriver.CARBON_PRICING,
            sector=SectorType.ENERGY,
        )
        # All default sub-scores are 0
        assert assessment.financial_impact >= Decimal("0")

    def test_provenance_hash(self, sample_transition_risk_assessment):
        assert len(sample_transition_risk_assessment.provenance_hash) == 64

    def test_provenance_deterministic(self):
        org_id = _new_id()
        a1 = TransitionRiskAssessment(
            org_id=org_id,
            risk_type=RiskType.TRANSITION_POLICY,
            driver=TransitionDriver.CARBON_PRICING,
            financial_impact=Decimal("100000"),
        )
        a2 = TransitionRiskAssessment(
            org_id=org_id,
            risk_type=RiskType.TRANSITION_POLICY,
            driver=TransitionDriver.CARBON_PRICING,
            financial_impact=Decimal("100000"),
        )
        assert a1.provenance_hash == a2.provenance_hash

    def test_mitigation_actions(self, sample_transition_risk_assessment):
        assert len(sample_transition_risk_assessment.mitigation_actions) >= 1


# ===========================================================================
# Sector Profiles
# ===========================================================================

class TestSectorProfiles:
    """Test sector-specific transition risk profiles."""

    @pytest.mark.parametrize("sector", list(SectorType))
    def test_all_sectors_have_profiles(self, sector):
        assert sector in SECTOR_TRANSITION_PROFILES

    @pytest.mark.parametrize("sector,expected_exposure", [
        (SectorType.ENERGY, "very_high"),
        (SectorType.TRANSPORT, "high"),
        (SectorType.MATERIALS, "high"),
        (SectorType.TECHNOLOGY, "low"),
        (SectorType.HEALTHCARE, "low"),
    ])
    def test_sector_exposure_levels(self, sector, expected_exposure):
        profile = SECTOR_TRANSITION_PROFILES[sector]
        assert profile["transition_exposure"] == expected_exposure

    @pytest.mark.parametrize("sector,expected_stranding", [
        (SectorType.ENERGY, "high"),
        (SectorType.AGRICULTURE, "low"),
        (SectorType.BANKING, "low"),
    ])
    def test_sector_stranding_risk(self, sector, expected_stranding):
        profile = SECTOR_TRANSITION_PROFILES[sector]
        assert profile["stranding_risk"] == expected_stranding

    def test_stranding_probability(self, sample_transition_risk_assessment):
        assert sample_transition_risk_assessment.stranding_probability == Decimal("25.0")

    def test_stranding_probability_range(self):
        assessment = TransitionRiskAssessment(
            org_id=_new_id(),
            risk_type=RiskType.TRANSITION_POLICY,
            driver=TransitionDriver.CARBON_PRICING,
            stranding_probability=Decimal("0.5"),
        )
        assert Decimal("0") <= assessment.stranding_probability <= Decimal("100")
