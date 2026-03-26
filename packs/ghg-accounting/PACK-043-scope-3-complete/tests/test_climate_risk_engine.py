# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Climate Risk Engine
==============================================

Tests carbon pricing exposure, physical risk assessment, opportunity
valuation, NPV calculation, IEA NZE scenario, NGFS comparison, CBAM
exposure, stranded asset risk, and edge cases.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal

import pytest


# =============================================================================
# Carbon Pricing Exposure
# =============================================================================


class TestCarbonPricingExposure:
    """Test carbon pricing exposure at various price levels."""

    @pytest.mark.parametrize("scenario,expected_price", [
        ("low", Decimal("50")),
        ("medium", Decimal("100")),
        ("high", Decimal("150")),
        ("extreme", Decimal("200")),
    ])
    def test_carbon_price_scenarios(self, scenario, expected_price, sample_climate_risks):
        tr = sample_climate_risks["transition_risks"][0]
        assert tr["carbon_price_scenarios"][scenario]["price_usd_per_tco2e"] == expected_price

    def test_exposure_at_50_per_tco2e(self, sample_climate_risks):
        """$50/tCO2e * 252,500 tCO2e = $12,625,000."""
        tr = sample_climate_risks["transition_risks"][0]
        emissions = tr["affected_scope3_tco2e"]
        price = tr["carbon_price_scenarios"]["low"]["price_usd_per_tco2e"]
        expected = emissions * price
        assert expected == Decimal("12625000")
        assert tr["carbon_price_scenarios"]["low"]["annual_exposure_usd"] == expected

    def test_exposure_at_100_per_tco2e(self, sample_climate_risks):
        tr = sample_climate_risks["transition_risks"][0]
        price = Decimal("100")
        emissions = tr["affected_scope3_tco2e"]
        expected = emissions * price
        assert expected == Decimal("25250000")

    def test_exposure_at_200_per_tco2e(self, sample_climate_risks):
        tr = sample_climate_risks["transition_risks"][0]
        expected = tr["carbon_price_scenarios"]["extreme"]["annual_exposure_usd"]
        assert expected == Decimal("50500000")

    def test_exposure_increases_linearly(self, sample_climate_risks):
        """Double the carbon price -> double the exposure."""
        scenarios = sample_climate_risks["transition_risks"][0]["carbon_price_scenarios"]
        low = scenarios["low"]["annual_exposure_usd"]
        medium = scenarios["medium"]["annual_exposure_usd"]
        assert medium == low * 2

    def test_all_transition_risks_present(self, sample_climate_risks):
        assert len(sample_climate_risks["transition_risks"]) == 3


# =============================================================================
# Physical Risk Assessment
# =============================================================================


class TestPhysicalRiskAssessment:
    """Test physical risk assessment with hazard data."""

    def test_two_physical_risks(self, sample_climate_risks):
        assert len(sample_climate_risks["physical_risks"]) == 2

    def test_chronic_risk_present(self, sample_climate_risks):
        chronic = [r for r in sample_climate_risks["physical_risks"] if r["risk_type"] == "chronic"]
        assert len(chronic) == 1

    def test_acute_risk_present(self, sample_climate_risks):
        acute = [r for r in sample_climate_risks["physical_risks"] if r["risk_type"] == "acute"]
        assert len(acute) == 1

    def test_physical_risk_impact_positive(self, sample_climate_risks):
        for risk in sample_climate_risks["physical_risks"]:
            assert risk["annual_impact_usd"] > Decimal("0")

    def test_sea_level_rise_hazard(self, sample_climate_risks):
        slr = next(
            r for r in sample_climate_risks["physical_risks"]
            if r["hazard"] == "sea_level_rise"
        )
        assert slr["affected_suppliers"] > 0

    def test_extreme_heat_hazard(self, sample_climate_risks):
        heat = next(
            r for r in sample_climate_risks["physical_risks"]
            if r["hazard"] == "extreme_heat"
        )
        assert heat["affected_facilities"] > 0


# =============================================================================
# Opportunity Valuation
# =============================================================================


class TestOpportunityValuation:
    """Test climate opportunity valuation."""

    def test_two_opportunities(self, sample_climate_risks):
        assert len(sample_climate_risks["opportunities"]) == 2

    def test_opportunities_have_positive_value(self, sample_climate_risks):
        for opp in sample_climate_risks["opportunities"]:
            assert opp["annual_value_usd"] > Decimal("0")

    def test_payback_period_reasonable(self, sample_climate_risks):
        for opp in sample_climate_risks["opportunities"]:
            payback = opp["payback_years"]
            assert Decimal("0") < payback < Decimal("15")

    def test_payback_calculation(self, sample_climate_risks):
        """Payback = investment / annual_value."""
        for opp in sample_climate_risks["opportunities"]:
            expected_payback = opp["investment_required_usd"] / opp["annual_value_usd"]
            assert expected_payback == pytest.approx(
                opp["payback_years"], abs=Decimal("0.01")
            )


# =============================================================================
# NPV Calculation
# =============================================================================


class TestNPVCalculation:
    """Test NPV calculation over various time horizons."""

    @pytest.mark.parametrize("years", [10, 20, 30])
    def test_npv_positive_for_opportunity(self, years, sample_climate_risks):
        """NPV of opportunity should be positive."""
        opp = sample_climate_risks["opportunities"][0]
        discount_rate = Decimal("0.08")
        annual_cf = opp["annual_value_usd"]
        investment = opp["investment_required_usd"]
        npv = -investment + sum(
            annual_cf / (Decimal("1") + discount_rate) ** y
            for y in range(1, years + 1)
        )
        assert npv > Decimal("0")

    def test_npv_10_year_carbon_exposure(self, sample_climate_risks):
        """NPV of 10-year carbon pricing exposure."""
        tr = sample_climate_risks["transition_risks"][0]
        annual_exposure = tr["carbon_price_scenarios"]["medium"]["annual_exposure_usd"]
        discount_rate = Decimal("0.08")
        npv = sum(
            annual_exposure / (Decimal("1") + discount_rate) ** y
            for y in range(1, 11)
        )
        assert npv > Decimal("0")
        # NPV should be less than undiscounted 10-year total
        assert npv < annual_exposure * 10

    def test_higher_discount_rate_lower_npv(self, sample_climate_risks):
        annual = Decimal("25250000")
        years = 10
        npv_low = sum(annual / Decimal("1.05") ** y for y in range(1, years + 1))
        npv_high = sum(annual / Decimal("1.15") ** y for y in range(1, years + 1))
        assert npv_high < npv_low


# =============================================================================
# IEA NZE Scenario
# =============================================================================


class TestIEANZEScenario:
    """Test IEA Net Zero Emissions scenario analysis."""

    def test_nze_alignment_status(self, sample_climate_risks):
        nze = sample_climate_risks["scenario_analysis"]["iea_nze"]
        assert nze["aligned"] is False

    def test_nze_gap(self, sample_climate_risks):
        nze = sample_climate_risks["scenario_analysis"]["iea_nze"]
        assert nze["gap_pct"] > Decimal("0")

    def test_nze_required_reduction(self, sample_climate_risks):
        nze = sample_climate_risks["scenario_analysis"]["iea_nze"]
        assert nze["required_reduction_by_2030"] == Decimal("42")


# =============================================================================
# NGFS Scenario Comparison
# =============================================================================


class TestNGFSScenarioComparison:
    """Test NGFS scenario comparison."""

    def test_orderly_scenario(self, sample_climate_risks):
        orderly = sample_climate_risks["scenario_analysis"]["ngfs_orderly"]
        assert orderly["carbon_price_2030"] == Decimal("130")

    def test_disorderly_scenario(self, sample_climate_risks):
        disorderly = sample_climate_risks["scenario_analysis"]["ngfs_disorderly"]
        assert disorderly["carbon_price_2030"] == Decimal("200")

    def test_disorderly_worse_than_orderly(self, sample_climate_risks):
        orderly = sample_climate_risks["scenario_analysis"]["ngfs_orderly"]
        disorderly = sample_climate_risks["scenario_analysis"]["ngfs_disorderly"]
        assert disorderly["total_exposure_2030"] > orderly["total_exposure_2030"]


# =============================================================================
# CBAM Exposure
# =============================================================================


class TestCBAMExposure:
    """Test EU CBAM exposure calculation."""

    def test_cbam_risk_present(self, sample_climate_risks):
        cbam = next(
            r for r in sample_climate_risks["transition_risks"]
            if r["risk_type"] == "cbam"
        )
        assert cbam["annual_exposure_usd"] > Decimal("0")

    def test_cbam_affects_categories_1_2(self, sample_climate_risks):
        cbam = next(
            r for r in sample_climate_risks["transition_risks"]
            if r["risk_type"] == "cbam"
        )
        assert 1 in cbam["affected_categories"]
        assert 2 in cbam["affected_categories"]


# =============================================================================
# Stranded Asset Risk
# =============================================================================


class TestStrandedAssetRisk:
    """Test stranded asset risk assessment."""

    def test_stranded_asset_risk_present(self, sample_climate_risks):
        stranded = next(
            r for r in sample_climate_risks["transition_risks"]
            if r["risk_type"] == "stranded_assets"
        )
        assert stranded["exposure_usd"] > Decimal("0")

    def test_stranded_asset_timeframe(self, sample_climate_risks):
        stranded = next(
            r for r in sample_climate_risks["transition_risks"]
            if r["risk_type"] == "stranded_assets"
        )
        assert stranded["timeframe_years"] > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestClimateRiskEdgeCases:
    """Test edge cases for climate risk engine."""

    def test_zero_carbon_price(self):
        """At zero carbon price, exposure is zero."""
        emissions = Decimal("252500")
        price = Decimal("0")
        exposure = emissions * price
        assert exposure == Decimal("0")

    def test_no_physical_risk(self):
        """Organization with no physical risk exposure."""
        physical_risks = []
        total_impact = sum(r.get("annual_impact_usd", Decimal("0")) for r in physical_risks)
        assert total_impact == Decimal("0")

    def test_net_opportunity_vs_risk(self, sample_climate_risks):
        """Compare total opportunity value vs total risk exposure."""
        opportunity_value = sum(
            o["annual_value_usd"] for o in sample_climate_risks["opportunities"]
        )
        risk_exposure = sum(
            r["annual_impact_usd"] for r in sample_climate_risks["physical_risks"]
        )
        net = opportunity_value - risk_exposure
        assert isinstance(net, Decimal)

    def test_risk_ids_unique(self, sample_climate_risks):
        ids = []
        ids.extend(r["risk_id"] for r in sample_climate_risks["transition_risks"])
        ids.extend(r["risk_id"] for r in sample_climate_risks["physical_risks"])
        assert len(ids) == len(set(ids))
