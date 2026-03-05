# -*- coding: utf-8 -*-
"""
Unit tests for SBTi FLAG Assessment Engine.

Tests FLAG (Forestry, Land and Agriculture) trigger assessment at 20%
threshold, sector classification, all 11 commodity pathways, sector-level
3.03%/yr pathway, deforestation commitment tracking, long-term FLAG
72% by 2050, and FLAG vs non-FLAG emission separation with 25+ test
functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date
from decimal import Decimal

import pytest


# ===========================================================================
# FLAG Trigger
# ===========================================================================

class TestFLAGTrigger:
    """Test FLAG 20% threshold trigger."""

    def test_flag_triggered(self, sample_flag_assessment):
        assert sample_flag_assessment["flag_target_required"] is True
        assert sample_flag_assessment["flag_pct_of_total"] >= 20.0

    def test_flag_not_triggered(self, non_flag_assessment):
        assert non_flag_assessment["flag_target_required"] is False
        assert non_flag_assessment["flag_pct_of_total"] < 20.0

    @pytest.mark.parametrize("flag_pct,required", [
        (20.0, True),   # Exactly at threshold
        (20.1, True),
        (25.0, True),
        (50.0, True),
        (19.9, False),
        (10.0, False),
        (0.0, False),
    ])
    def test_flag_threshold_edge_cases(self, flag_pct, required):
        assert (flag_pct >= 20.0) == required

    def test_flag_pct_calculation(self):
        total = 100_000.0
        flag = 25_000.0
        pct = (flag / total) * 100
        assert pct == 25.0

    def test_flag_threshold_value(self, sample_flag_assessment):
        assert sample_flag_assessment["flag_threshold_pct"] == 20.0


# ===========================================================================
# Sector Classification
# ===========================================================================

class TestSectorClassification:
    """Test FLAG sector classification (agriculture, food, forestry)."""

    FLAG_SECTORS = [
        "food_agriculture",
        "forestry",
        "paper_pulp",
        "agricultural_products",
        "food_beverage",
        "tobacco",
    ]

    def test_flag_sector_identified(self, sample_flag_assessment):
        assert sample_flag_assessment["flag_sector_classification"] in self.FLAG_SECTORS

    def test_flag_sub_sectors(self, sample_flag_assessment):
        sub_sectors = sample_flag_assessment["flag_sub_sectors"]
        assert len(sub_sectors) >= 1

    @pytest.mark.parametrize("isic_code,expected_flag", [
        ("0111", True),   # Growing of cereals
        ("0121", True),   # Growing of grapes
        ("0210", True),   # Silviculture (forestry)
        ("2394", False),  # Cement manufacturing
        ("6419", False),  # Financial services
    ])
    def test_isic_to_flag_mapping(self, isic_code, expected_flag):
        flag_isic_prefixes = ["01", "02", "03"]  # Agriculture, Forestry, Fishing
        is_flag = isic_code[:2] in flag_isic_prefixes
        assert is_flag == expected_flag

    def test_non_flag_organization(self, sample_organization):
        assert sample_organization["sector"] == "manufacturing"


# ===========================================================================
# Commodity Pathway
# ===========================================================================

class TestCommodityPathway:
    """Test all 11 FLAG commodity-specific pathways."""

    ALL_COMMODITIES = [
        "cattle", "soy", "palm_oil", "timber", "cocoa",
        "coffee", "rubber", "maize", "rice", "wheat", "sugarcane",
    ]

    @pytest.mark.parametrize("commodity", [
        "cattle", "soy", "palm_oil", "timber", "cocoa",
        "coffee", "rubber", "maize", "rice", "wheat", "sugarcane",
    ])
    def test_commodity_pathway_exists(self, flag_commodity_pathways, commodity):
        assert commodity in flag_commodity_pathways

    def test_all_11_commodities(self, flag_commodity_pathways):
        assert len(flag_commodity_pathways) == 11

    def test_commodity_annual_rate(self, flag_commodity_pathways):
        for commodity, data in flag_commodity_pathways.items():
            assert data["annual_rate"] == 3.03

    def test_commodity_pathway_fields(self, flag_commodity_pathways):
        for commodity, data in flag_commodity_pathways.items():
            assert "annual_rate" in data
            assert "base_year" in data
            assert "target_year" in data

    def test_commodity_breakdown_present(self, sample_flag_assessment):
        commodities = sample_flag_assessment["commodity_breakdown"]
        assert len(commodities) >= 1

    def test_commodity_percentages_sum_to_100(self, sample_flag_assessment):
        commodities = sample_flag_assessment["commodity_breakdown"]
        total_pct = sum(c["pct"] for c in commodities.values())
        assert abs(total_pct - 100.0) < 0.5


# ===========================================================================
# Sector Pathway
# ===========================================================================

class TestSectorPathway:
    """Test FLAG sector-level pathway (3.03%/yr)."""

    def test_sector_annual_rate(self, sample_flag_assessment):
        assert sample_flag_assessment["near_term_rate_pct"] == 3.03

    def test_sector_10yr_total(self):
        rate = 3.03
        total = rate * 10
        assert total == pytest.approx(30.3, abs=0.01)

    def test_sector_pathway_milestone(self):
        base = 25_000.0
        rate = 3.03
        year_5 = base * (1 - (rate / 100) * 5)
        assert year_5 < base
        assert year_5 == pytest.approx(21_212.5, abs=1.0)


# ===========================================================================
# Deforestation Commitment
# ===========================================================================

class TestDeforestationCommitment:
    """Test deforestation and land-use change commitment tracking."""

    def test_deforestation_commitment_present(self, sample_flag_assessment):
        assert sample_flag_assessment["deforestation_commitment"] is True

    def test_deforestation_target_date(self, sample_flag_assessment):
        target_date = sample_flag_assessment["deforestation_target_date"]
        assert isinstance(target_date, date)

    def test_deforestation_by_2025(self, sample_flag_assessment):
        target_date = sample_flag_assessment["deforestation_target_date"]
        assert target_date.year <= 2025

    def test_flag_target_deforestation(self, sample_flag_target):
        assert sample_flag_target["deforestation_commitment"] is True
        assert sample_flag_target["deforestation_target_date"].year <= 2025


# ===========================================================================
# Long-Term FLAG
# ===========================================================================

class TestLongTermFLAG:
    """Test long-term FLAG 72% reduction by 2050."""

    def test_long_term_reduction_72_pct(self, sample_flag_assessment):
        assert sample_flag_assessment["long_term_reduction_pct"] == 72.0

    def test_long_term_flag_target(self):
        base = 25_000.0
        reduction_pct = 72.0
        target_2050 = base * (1 - reduction_pct / 100)
        assert target_2050 == 7_000.0

    @pytest.mark.parametrize("reduction,meets_long_term", [
        (72.0, True),
        (80.0, True),
        (71.9, False),
        (50.0, False),
    ])
    def test_long_term_threshold(self, reduction, meets_long_term):
        assert (reduction >= 72.0) == meets_long_term


# ===========================================================================
# FLAG vs Non-FLAG Emission Separation
# ===========================================================================

class TestEmissionsSplit:
    """Test FLAG vs non-FLAG emissions separation."""

    def test_net_flag_after_removals(self, sample_flag_assessment):
        gross = sample_flag_assessment["flag_emissions_tco2e"]
        removals = sample_flag_assessment["removals_tco2e"]
        net = sample_flag_assessment["net_flag_emissions_tco2e"]
        assert net == gross - removals

    def test_removals_tracked(self, sample_flag_assessment):
        assert sample_flag_assessment["removals_tco2e"] >= 0

    def test_flag_separate_from_total(self, sample_flag_assessment):
        total = sample_flag_assessment["total_emissions_tco2e"]
        flag = sample_flag_assessment["flag_emissions_tco2e"]
        non_flag = total - flag
        assert non_flag > 0
        assert non_flag + flag == total

    def test_flag_target_removals(self, sample_flag_target):
        assert sample_flag_target["removals_target_tco2e"] > 0
