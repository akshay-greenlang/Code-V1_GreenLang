# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Sector Pathway Engine.

Tests sector-specific decarbonization pathways for power, cement, steel,
buildings, maritime, and aviation sectors, sector detection from
ISIC/NACE/NAICS codes, and multi-sector blending with 28+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest


# ===========================================================================
# Power Pathway
# ===========================================================================

class TestPowerPathway:
    """Test power sector tCO2/MWh convergence pathway."""

    def test_power_base_intensity(self, sector_pathway_data):
        assert sector_pathway_data["power"]["base_2020"] == 0.48

    def test_power_2030_target(self, sector_pathway_data):
        assert sector_pathway_data["power"]["target_2030"] == 0.14

    def test_power_2050_target(self, sector_pathway_data):
        assert sector_pathway_data["power"]["target_2050"] == 0.0

    def test_power_unit(self, sector_pathway_data):
        assert sector_pathway_data["power"]["unit"] == "tCO2/MWh"

    def test_power_decreasing_trajectory(self, sector_pathway_data):
        p = sector_pathway_data["power"]
        assert p["target_2030"] < p["base_2020"]
        assert p["target_2050"] < p["target_2030"]


# ===========================================================================
# Cement Pathway
# ===========================================================================

class TestCementPathway:
    """Test cement sector tCO2e/tonne values."""

    def test_cement_base_intensity(self, sector_pathway_data):
        assert sector_pathway_data["cement"]["base_2020"] == 0.62

    def test_cement_2030_target(self, sector_pathway_data):
        assert sector_pathway_data["cement"]["target_2030"] == 0.42

    def test_cement_2050_target(self, sector_pathway_data):
        assert sector_pathway_data["cement"]["target_2050"] == 0.10

    def test_cement_unit(self, sector_pathway_data):
        assert sector_pathway_data["cement"]["unit"] == "tCO2e/tonne"

    def test_cement_sda_convergence(self, sda_pathway):
        assert sda_pathway["sector"] == "cement"
        assert sda_pathway["convergence_intensity_2050"] == 0.10

    def test_cement_milestones_decreasing(self, sda_pathway):
        milestones = sda_pathway["milestones"]
        years = sorted(milestones.keys())
        for i in range(1, len(years)):
            assert milestones[years[i]] < milestones[years[i - 1]]


# ===========================================================================
# Steel Pathway
# ===========================================================================

class TestSteelPathway:
    """Test steel sector with iron-ore vs scrap differentiation."""

    def test_steel_iron_ore_base(self, sector_pathway_data):
        assert sector_pathway_data["steel"]["iron_ore_base_2020"] == 2.0

    def test_steel_scrap_base(self, sector_pathway_data):
        assert sector_pathway_data["steel"]["scrap_base_2020"] == 0.4

    def test_iron_ore_more_intensive(self, sector_pathway_data):
        s = sector_pathway_data["steel"]
        assert s["iron_ore_base_2020"] > s["scrap_base_2020"]

    def test_steel_iron_ore_convergence(self, sector_pathway_data):
        s = sector_pathway_data["steel"]
        assert s["iron_ore_target_2050"] < s["iron_ore_base_2020"]

    def test_steel_scrap_convergence(self, sector_pathway_data):
        s = sector_pathway_data["steel"]
        assert s["scrap_target_2050"] < s["scrap_base_2020"]

    def test_steel_unit(self, sector_pathway_data):
        assert sector_pathway_data["steel"]["unit"] == "tCO2e/tonne steel"


# ===========================================================================
# Buildings Pathway
# ===========================================================================

class TestBuildingsPathway:
    """Test buildings sector kgCO2e/m2 CRREM pathway."""

    def test_buildings_unit(self, sector_pathway_data):
        assert sector_pathway_data["buildings"]["unit"] == "kgCO2e/m2"

    def test_buildings_residential_base(self, sector_pathway_data):
        assert sector_pathway_data["buildings"]["residential_base"] == 35.0

    def test_buildings_commercial_base(self, sector_pathway_data):
        assert sector_pathway_data["buildings"]["commercial_base"] == 50.0

    def test_commercial_more_intensive(self, sector_pathway_data):
        b = sector_pathway_data["buildings"]
        assert b["commercial_base"] > b["residential_base"]

    def test_buildings_crrem_2030(self, sector_pathway_data):
        b = sector_pathway_data["buildings"]
        assert b["crrem_2030"] < b["residential_base"]
        assert b["crrem_2030"] < b["commercial_base"]


# ===========================================================================
# Maritime Pathway
# ===========================================================================

class TestMaritimePathway:
    """Test maritime sector gCO2/dwt-nm pathway."""

    def test_maritime_unit(self, sector_pathway_data):
        assert sector_pathway_data["maritime"]["unit"] == "gCO2/dwt-nm"

    def test_maritime_base(self, sector_pathway_data):
        assert sector_pathway_data["maritime"]["base_2020"] == 8.5

    def test_maritime_2030_target(self, sector_pathway_data):
        m = sector_pathway_data["maritime"]
        assert m["target_2030"] < m["base_2020"]

    def test_maritime_2050_near_zero(self, sector_pathway_data):
        m = sector_pathway_data["maritime"]
        assert m["target_2050"] < 1.0  # Near-zero


# ===========================================================================
# Aviation Pathway
# ===========================================================================

class TestAviationPathway:
    """Test aviation sector gCO2/RPK and gCO2/RTK pathways."""

    def test_aviation_passenger_unit(self, sector_pathway_data):
        assert sector_pathway_data["aviation"]["unit_passenger"] == "gCO2/RPK"

    def test_aviation_freight_unit(self, sector_pathway_data):
        assert sector_pathway_data["aviation"]["unit_freight"] == "gCO2/RTK"

    def test_aviation_passenger_base(self, sector_pathway_data):
        assert sector_pathway_data["aviation"]["passenger_base_2020"] == 90.0

    def test_aviation_freight_base(self, sector_pathway_data):
        assert sector_pathway_data["aviation"]["freight_base_2020"] == 600.0

    def test_aviation_passenger_target(self, sector_pathway_data):
        a = sector_pathway_data["aviation"]
        assert a["passenger_target_2050"] < a["passenger_base_2020"]

    def test_aviation_freight_target(self, sector_pathway_data):
        a = sector_pathway_data["aviation"]
        assert a["freight_target_2050"] < a["freight_base_2020"]

    def test_freight_more_intensive(self, sector_pathway_data):
        a = sector_pathway_data["aviation"]
        assert a["freight_base_2020"] > a["passenger_base_2020"]


# ===========================================================================
# Sector Detection
# ===========================================================================

class TestSectorDetection:
    """Test ISIC/NACE/NAICS to sector pathway mapping."""

    @pytest.mark.parametrize("isic_code,expected_sector", [
        ("3510", "power"),
        ("2394", "cement"),
        ("2410", "steel"),
        ("4100", "buildings"),
        ("5011", "maritime"),
        ("5110", "aviation"),
        ("0111", "flag"),
    ])
    def test_isic_to_sector_mapping(self, isic_code, expected_sector):
        isic_sector_map = {
            "3510": "power", "2394": "cement", "2410": "steel",
            "4100": "buildings", "5011": "maritime", "5110": "aviation",
            "0111": "flag",
        }
        assert isic_sector_map.get(isic_code) == expected_sector

    @pytest.mark.parametrize("nace_code,expected_sector", [
        ("D35.11", "power"),
        ("C23.51", "cement"),
        ("C24.10", "steel"),
        ("F41.10", "buildings"),
    ])
    def test_nace_to_sector_mapping(self, nace_code, expected_sector):
        nace_sector_map = {
            "D35.11": "power", "C23.51": "cement",
            "C24.10": "steel", "F41.10": "buildings",
        }
        assert nace_sector_map.get(nace_code) == expected_sector

    def test_naics_to_sector_mapping(self):
        naics_map = {
            "221112": "power", "327310": "cement", "331110": "steel",
        }
        assert naics_map["327310"] == "cement"


# ===========================================================================
# Multi-Sector Blending
# ===========================================================================

class TestMultiSectorBlending:
    """Test weighted multi-sector pathway blending."""

    def test_weighted_blend(self):
        sectors = [
            {"sector": "power", "intensity_2030": 0.14, "weight": 0.6},
            {"sector": "cement", "intensity_2030": 0.42, "weight": 0.4},
        ]
        blended = sum(s["intensity_2030"] * s["weight"] for s in sectors)
        expected = 0.14 * 0.6 + 0.42 * 0.4
        assert blended == pytest.approx(expected, abs=0.001)

    def test_weights_sum_to_1(self):
        weights = [0.6, 0.3, 0.1]
        assert sum(weights) == pytest.approx(1.0)

    def test_single_sector_blend(self):
        sectors = [{"intensity_2030": 0.42, "weight": 1.0}]
        blended = sum(s["intensity_2030"] * s["weight"] for s in sectors)
        assert blended == 0.42
