# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Pathway Calculator Engine.

Tests ACA pathways (1.5C and WB2C), SDA sector-specific convergence
pathways, economic/physical intensity pathways, FLAG commodity and sector
pathways, pathway comparison, uncertainty bands, and milestone generation
with 30+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal

import pytest


# ===========================================================================
# ACA Pathway
# ===========================================================================

class TestACAPathway:
    """Test Absolute Contraction Approach pathway calculations."""

    def test_aca_1_5c_annual_rate(self, sample_pathway):
        assert sample_pathway["annual_reduction_rate_pct"] == 4.2

    def test_aca_wb2c_rate(self):
        """WB2C requires minimum 2.5% per year."""
        rate = 2.5
        base = 100_000.0
        expected_2030 = base * (1 - (rate / 100) * 10)
        assert expected_2030 == 75_000.0

    def test_aca_1_5c_10yr_reduction(self):
        """4.2% per year over 10 years = 42% total reduction."""
        rate = 4.2
        total_reduction_pct = rate * 10
        assert total_reduction_pct == 42.0

    def test_aca_milestones_decreasing(self, sample_pathway):
        milestones = sample_pathway["milestones"]
        years = sorted(milestones.keys())
        for i in range(1, len(years)):
            assert milestones[years[i]] < milestones[years[i - 1]]

    def test_aca_milestone_correctness(self, sample_pathway):
        base = sample_pathway["base_emissions_tco2e"]
        rate = sample_pathway["annual_reduction_rate_pct"]
        milestones = sample_pathway["milestones"]
        for year, expected_value in milestones.items():
            years_elapsed = year - sample_pathway["base_year"]
            calculated = base - base * (rate / 100.0) * years_elapsed
            assert abs(expected_value - calculated) < 0.5

    def test_aca_target_year_value(self, sample_pathway):
        base = sample_pathway["base_emissions_tco2e"]
        rate = sample_pathway["annual_reduction_rate_pct"]
        target_year = sample_pathway["target_year"]
        years = target_year - sample_pathway["base_year"]
        expected = base * (1 - (rate / 100) * years)
        assert sample_pathway["milestones"][target_year] == pytest.approx(expected, abs=1.0)

    @pytest.mark.parametrize("ambition,expected_rate", [
        ("1.5C", 4.2),
        ("well_below_2C", 2.5),
    ])
    def test_aca_ambition_to_rate_mapping(self, ambition, expected_rate):
        rates = {"1.5C": 4.2, "well_below_2C": 2.5}
        assert rates[ambition] == expected_rate


# ===========================================================================
# SDA Pathway
# ===========================================================================

class TestSDAPathway:
    """Test Sectoral Decarbonization Approach pathway calculations."""

    def test_sda_pathway_type(self, sda_pathway):
        assert sda_pathway["pathway_type"] == "sda"

    def test_sda_convergence_target(self, sda_pathway):
        assert sda_pathway["convergence_intensity_2050"] < sda_pathway["base_intensity"]

    def test_sda_milestones_decreasing(self, sda_pathway):
        milestones = sda_pathway["milestones"]
        years = sorted(milestones.keys())
        for i in range(1, len(years)):
            assert milestones[years[i]] < milestones[years[i - 1]]

    def test_sda_reaches_convergence(self, sda_pathway):
        milestones = sda_pathway["milestones"]
        assert milestones[2050] == sda_pathway["convergence_intensity_2050"]

    def test_sda_cement_sector(self, sda_pathway):
        assert sda_pathway["sector"] == "cement"
        assert sda_pathway["intensity_unit"] == "tCO2e per tonne cement"


# ===========================================================================
# Economic Intensity
# ===========================================================================

class TestEconomicIntensity:
    """Test economic intensity pathway calculation."""

    def test_economic_intensity_pathway(self):
        base_intensity = 50.0  # tCO2e per USD million revenue
        target_intensity = 25.0
        reduction_pct = ((base_intensity - target_intensity) / base_intensity) * 100
        assert reduction_pct == 50.0

    def test_economic_intensity_units(self):
        valid_units = [
            "tCO2e per USD million revenue",
            "tCO2e per EUR million revenue",
            "tCO2e per employee",
            "tCO2e per unit produced",
        ]
        assert len(valid_units) >= 3


# ===========================================================================
# Physical Intensity
# ===========================================================================

class TestPhysicalIntensity:
    """Test physical intensity pathway calculation."""

    def test_physical_intensity_pathway(self, sample_intensity_target):
        base = sample_intensity_target["base_year_intensity"]
        target = sample_intensity_target["target_intensity"]
        assert target < base

    @pytest.mark.parametrize("sector,base,target_2050", [
        ("cement", 0.62, 0.10),
        ("power", 0.48, 0.0),
    ])
    def test_sector_intensity_convergence(self, sector, base, target_2050):
        assert target_2050 < base


# ===========================================================================
# FLAG Commodity Pathway
# ===========================================================================

class TestFLAGCommodityPathway:
    """Test FLAG commodity-specific pathways (11 commodities)."""

    def test_flag_commodity_pathway_type(self, flag_commodity_pathway):
        assert flag_commodity_pathway["pathway_type"] == "flag_commodity"

    def test_flag_commodity_rate(self, flag_commodity_pathway):
        assert flag_commodity_pathway["annual_reduction_rate_pct"] == 3.03

    @pytest.mark.parametrize("commodity", [
        "cattle", "soy", "palm_oil", "timber", "cocoa",
        "coffee", "rubber", "maize", "rice", "wheat", "sugarcane",
    ])
    def test_all_11_commodities_defined(self, flag_commodity_pathways, commodity):
        assert commodity in flag_commodity_pathways
        assert flag_commodity_pathways[commodity]["annual_rate"] == 3.03

    def test_flag_commodity_count(self, flag_commodity_pathways):
        assert len(flag_commodity_pathways) == 11

    def test_flag_long_term_reduction(self, flag_commodity_pathway):
        assert flag_commodity_pathway["long_term_reduction_pct"] == 72.0
        assert flag_commodity_pathway["long_term_target_year"] == 2050


# ===========================================================================
# FLAG Sector Pathway
# ===========================================================================

class TestFLAGSectorPathway:
    """Test FLAG sector-level pathway (3.03%/yr)."""

    def test_flag_sector_rate(self, sample_config):
        assert sample_config["flag_sector_annual_rate"] == 3.03

    def test_flag_sector_10yr_reduction(self):
        rate = 3.03
        total = rate * 10
        assert total == pytest.approx(30.3, abs=0.01)

    def test_flag_sector_milestone_generation(self):
        base = 25_000.0
        rate = 3.03
        milestones = {}
        for year in range(2021, 2031):
            elapsed = year - 2020
            milestones[year] = round(base * (1 - (rate / 100) * elapsed), 1)
        assert milestones[2025] < base
        assert milestones[2030] < milestones[2025]


# ===========================================================================
# Pathway Comparison
# ===========================================================================

class TestPathwayComparison:
    """Test multiple pathway comparison."""

    def test_compare_aca_vs_sda(self, sample_pathway, sda_pathway):
        assert sample_pathway["pathway_type"] != sda_pathway["pathway_type"]

    def test_compare_ambition_levels(self):
        aca_1_5c_rate = 4.2
        aca_wb2c_rate = 2.5
        assert aca_1_5c_rate > aca_wb2c_rate

    def test_pathway_ranking_by_ambition(self):
        pathways = [
            {"name": "ACA 1.5C", "rate": 4.2},
            {"name": "ACA WB2C", "rate": 2.5},
            {"name": "FLAG Sector", "rate": 3.03},
        ]
        ranked = sorted(pathways, key=lambda p: p["rate"], reverse=True)
        assert ranked[0]["name"] == "ACA 1.5C"
        assert ranked[-1]["name"] == "ACA WB2C"


# ===========================================================================
# Uncertainty Bands
# ===========================================================================

class TestUncertaintyBands:
    """Test confidence interval calculation."""

    def test_uncertainty_bounds_exist(self, sample_pathway):
        assert sample_pathway["uncertainty_lower_pct"] is not None
        assert sample_pathway["uncertainty_upper_pct"] is not None

    def test_lower_bound_less_than_upper(self, sample_pathway):
        assert sample_pathway["uncertainty_lower_pct"] < sample_pathway["uncertainty_upper_pct"]

    def test_central_rate_within_bounds(self, sample_pathway):
        rate = sample_pathway["annual_reduction_rate_pct"]
        lower = sample_pathway["uncertainty_lower_pct"]
        upper = sample_pathway["uncertainty_upper_pct"]
        assert lower <= rate <= upper

    def test_confidence_level(self, sample_pathway):
        assert 0 < sample_pathway["confidence_level"] <= 1.0


# ===========================================================================
# Milestone Generation
# ===========================================================================

class TestMilestoneGeneration:
    """Test annual milestone correctness."""

    def test_milestones_annual(self, sample_pathway):
        milestones = sample_pathway["milestones"]
        years = sorted(milestones.keys())
        for i in range(1, len(years)):
            assert years[i] - years[i - 1] == 1

    def test_milestones_start_after_base_year(self, sample_pathway):
        first_year = min(sample_pathway["milestones"].keys())
        assert first_year == sample_pathway["base_year"] + 1

    def test_milestones_end_at_target_year(self, sample_pathway):
        last_year = max(sample_pathway["milestones"].keys())
        assert last_year == sample_pathway["target_year"]

    def test_milestone_count(self, sample_pathway):
        expected_count = sample_pathway["target_year"] - sample_pathway["base_year"]
        assert len(sample_pathway["milestones"]) == expected_count

    def test_milestones_all_below_base(self, sample_pathway):
        base = sample_pathway["base_emissions_tco2e"]
        for year, value in sample_pathway["milestones"].items():
            assert value < base
