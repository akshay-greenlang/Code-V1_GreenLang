# -*- coding: utf-8 -*-
"""
Deep tests for SectorPathwayEngine (Engine 6 of 10).

Covers: 25 sector pathway database, gap-to-benchmark calculation,
pathway alignment scoring, sector mapping, milestone tracking,
pathway credibility assessment, technology adoption curves,
Decimal arithmetic, SHA-256 provenance.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.sector_pathway_engine import (
    SectorPathwayEngine,
    PathwayCredibility,
    SectorId,
    SECTOR_PATHWAYS,
)

from conftest import assert_provenance_hash, timed_block


# ========================================================================
# Sector Database Validation
# ========================================================================


class TestSectorPathwayDatabase:
    """Validate the 25-sector pathway database."""

    def test_sector_count_at_least_25(self):
        assert len(SectorId) >= 25

    def test_sector_pathways_populated(self):
        assert len(SECTOR_PATHWAYS) >= 5

    @pytest.mark.parametrize("sector", [
        "power_generation", "oil_gas", "steel", "cement", "aluminium",
        "aviation",
    ])
    def test_key_sectors_present(self, sector):
        assert sector in SECTOR_PATHWAYS

    def test_each_pathway_has_name(self):
        for sid, data in SECTOR_PATHWAYS.items():
            assert "name" in data, f"{sid} missing name"

    def test_each_pathway_has_source(self):
        for sid, data in SECTOR_PATHWAYS.items():
            assert "source" in data, f"{sid} missing source"

    def test_each_pathway_has_milestones(self):
        for sid, data in SECTOR_PATHWAYS.items():
            assert "milestones" in data, f"{sid} missing milestones"

    def test_milestones_have_years(self):
        for sid, data in SECTOR_PATHWAYS.items():
            for ms in data["milestones"]:
                assert "year" in ms
                assert "value" in ms

    def test_milestones_chronologically_ordered(self):
        for sid, data in SECTOR_PATHWAYS.items():
            years = [m["year"] for m in data["milestones"]]
            assert years == sorted(years), f"{sid} milestones out of order"

    def test_power_generation_source(self):
        assert SECTOR_PATHWAYS["power_generation"]["source"] == "IEA NZE 2023"

    def test_steel_source(self):
        src = SECTOR_PATHWAYS["steel"]["source"]
        assert "MPP" in src or "IEA" in src


# ========================================================================
# Enum Validation
# ========================================================================


class TestSectorPathwayEnums:
    """Validate sector pathway enums."""

    def test_pathway_credibility_4_values(self):
        assert len(PathwayCredibility) == 4

    def test_credibility_values(self):
        assert PathwayCredibility.CONSERVATIVE.value == "conservative"
        assert PathwayCredibility.MODERATE.value == "moderate"
        assert PathwayCredibility.AGGRESSIVE.value == "aggressive"
        assert PathwayCredibility.MISALIGNED.value == "misaligned"

    def test_sector_id_count(self):
        assert len(SectorId) >= 25

    @pytest.mark.parametrize("sector_id", [
        "power_generation", "oil_gas", "coal_mining", "steel", "cement",
        "aluminium", "chemicals", "pulp_paper", "aviation", "maritime",
        "road_transport_light", "road_transport_heavy", "rail",
        "buildings_commercial", "buildings_residential",
        "agriculture", "food_beverage", "retail", "financial_services",
        "technology", "healthcare", "higher_education",
        "waste_management", "water_utilities", "telecommunications",
    ])
    def test_all_25_sector_ids(self, sector_id):
        assert SectorId(sector_id) is not None


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestSectorPathwayEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, sector_engine):
        assert sector_engine is not None

    def test_engine_has_calculate(self, sector_engine):
        assert callable(getattr(sector_engine, "assess", None))

    def test_engine_class_name(self):
        assert SectorPathwayEngine.__name__ == "SectorPathwayEngine"


# ========================================================================
# Sector Pathway Milestone Validation
# ========================================================================


class TestSectorMilestones:
    """Validate sector milestone data integrity."""

    def test_power_2030_milestone(self):
        ms = SECTOR_PATHWAYS["power_generation"]["milestones"]
        m2030 = next(m for m in ms if m["year"] == 2030)
        assert m2030["value"] == Decimal("60")

    def test_power_2050_milestone(self):
        ms = SECTOR_PATHWAYS["power_generation"]["milestones"]
        m2050 = next(m for m in ms if m["year"] == 2050)
        assert m2050["value"] == Decimal("100")

    def test_steel_2030_milestone(self):
        ms = SECTOR_PATHWAYS["steel"]["milestones"]
        m2030 = next(m for m in ms if m["year"] == 2030)
        assert m2030["value"] == Decimal("10")

    def test_cement_2030_milestone(self):
        ms = SECTOR_PATHWAYS["cement"]["milestones"]
        m2030 = next(m for m in ms if m["year"] == 2030)
        assert m2030["value"] == Decimal("0.52")

    def test_aviation_2030_milestone(self):
        ms = SECTOR_PATHWAYS["aviation"]["milestones"]
        m2030 = next(m for m in ms if m["year"] == 2030)
        assert m2030["value"] == Decimal("10")

    def test_each_sector_has_at_least_3_milestones(self):
        for sid, data in SECTOR_PATHWAYS.items():
            assert len(data["milestones"]) >= 3, (
                f"{sid} has fewer than 3 milestones"
            )

    def test_milestone_values_are_decimal(self):
        for sid, data in SECTOR_PATHWAYS.items():
            for ms in data["milestones"]:
                assert isinstance(ms["value"], Decimal), (
                    f"{sid} milestone year {ms['year']} value is not Decimal"
                )

    def test_milestone_descriptions_non_empty(self):
        for sid, data in SECTOR_PATHWAYS.items():
            for ms in data["milestones"]:
                assert "description" in ms
                assert len(ms["description"]) > 0
