# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Technology Roadmap Engine.

Tests technology adoption curves, CapEx phasing, cost projections,
milestone tracking, and sector-specific technology roadmaps.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  5 of 8 - technology_roadmap_engine.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.technology_roadmap_engine import (
    TechnologyRoadmapEngine,
    TechnologyRoadmapInput,
    TechnologyRoadmapResult,
    TechnologyAdoptionCurve,
    CapExPhase,
    CostProjection,
    MilestoneTrackingResult,
    TechnologyDependency,
    CurrentTechnologyStatus,
    TechnologyReadinessLevel,
    TechnologyCategory,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_provenance_hash,
    assert_processing_time,
    SDA_SECTORS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(sector="power_generation", entity_name="TestCo",
                base_year=2024, target_year=2050,
                current_technologies=None, include_milestones=True,
                include_capex=True, include_cost_projections=True,
                annual_capex_budget_eur=None, total_capacity=None):
    kw = dict(
        entity_name=entity_name,
        sector=sector,
        base_year=base_year,
        target_year=target_year,
        include_milestone_tracking=include_milestones,
        include_capex_phasing=include_capex,
        include_cost_projections=include_cost_projections,
    )
    if current_technologies is not None:
        kw["current_technologies"] = current_technologies
    if annual_capex_budget_eur is not None:
        kw["annual_capex_budget_eur"] = annual_capex_budget_eur
    if total_capacity is not None:
        kw["total_capacity"] = total_capacity
    return TechnologyRoadmapInput(**kw)


def _make_tech(name, penetration=Decimal("10"), trl="trl_9"):
    return CurrentTechnologyStatus(
        technology_name=name,
        current_penetration_pct=penetration,
        trl=TechnologyReadinessLevel(trl),
    )


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestTechRoadmapInstantiation:
    """Engine instantiation and metadata tests."""

    def test_engine_instantiates(self):
        engine = TechnologyRoadmapEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = TechnologyRoadmapEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = TechnologyRoadmapEngine()
        assert engine.engine_version == "1.0.0"

    def test_get_sector_technologies(self):
        engine = TechnologyRoadmapEngine()
        techs = engine.get_sector_technologies("power_generation")
        assert len(techs) > 0

    def test_get_milestone_count(self):
        engine = TechnologyRoadmapEngine()
        count = engine.get_milestone_count()
        assert count >= 0


# ===========================================================================
# Basic Roadmap Generation
# ===========================================================================


class TestBasicRoadmapGeneration:
    """Test basic technology roadmap generation."""

    def test_power_sector_roadmap(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(sector="power_generation"))
        assert result.total_technologies > 0
        assert len(result.adoption_curves) > 0

    def test_steel_sector_roadmap(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(sector="steel"))
        assert result.total_technologies > 0

    def test_cement_sector_roadmap(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(sector="cement"))
        assert result.total_technologies > 0

    def test_roadmap_has_adoption_curves(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert len(result.adoption_curves) > 0
        curve = result.adoption_curves[0]
        assert isinstance(curve, TechnologyAdoptionCurve)

    def test_roadmap_has_capex_phasing(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(include_capex=True))
        assert len(result.capex_phasing) > 0

    def test_roadmap_has_cost_projections(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(include_cost_projections=True))
        assert len(result.cost_projections) > 0

    def test_roadmap_has_milestones(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(include_milestones=True))
        assert result.milestone_tracking is not None

    def test_roadmap_has_dependencies(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.dependencies, list)


# ===========================================================================
# Custom Technologies
# ===========================================================================


class TestCustomTechnologies:
    """Test with user-provided technology data."""

    def test_custom_tech_list(self):
        engine = TechnologyRoadmapEngine()
        techs = [
            _make_tech("Solar PV", Decimal("10"), "trl_9"),
            _make_tech("Wind", Decimal("15"), "trl_9"),
        ]
        result = engine.calculate(_make_input(current_technologies=techs))
        assert result.total_technologies > 0

    def test_custom_tech_with_low_trl(self):
        engine = TechnologyRoadmapEngine()
        techs = [
            _make_tech("Green H2", Decimal("0"), "trl_6"),
        ]
        result = engine.calculate(_make_input(current_technologies=techs))
        assert result is not None

    def test_custom_tech_zero_penetration(self):
        engine = TechnologyRoadmapEngine()
        techs = [
            _make_tech("CCS", Decimal("0"), "trl_5"),
        ]
        result = engine.calculate(_make_input(current_technologies=techs))
        assert result.total_technologies > 0


# ===========================================================================
# Sector Parametrized Tests
# ===========================================================================


class TestSectorRoadmaps:
    """Test roadmap generation across sectors."""

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement",
                                         "aviation", "shipping"])
    def test_sector_roadmap_with_techs(self, sector):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(sector=sector))
        assert result.sector == sector
        assert result.total_technologies > 0

    @pytest.mark.parametrize("sector", ["aluminum", "buildings_commercial",
                                         "chemicals", "road_transport"])
    def test_sector_roadmap_no_builtin_techs(self, sector):
        """Sectors without built-in technologies return zero total."""
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(sector=sector))
        assert result.sector == sector
        assert result.total_technologies == 0

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_sector_has_technologies(self, sector):
        engine = TechnologyRoadmapEngine()
        techs = engine.get_sector_technologies(sector)
        assert len(techs) > 0


# ===========================================================================
# CapEx & Cost Analysis
# ===========================================================================


class TestCapExAnalysis:
    """Test CapEx phasing and cost projections."""

    def test_total_capex_calculated(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert result.total_capex_required_eur >= Decimal("0")

    def test_capex_by_year(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.total_capex_by_year, dict)

    def test_capex_with_budget(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(
            annual_capex_budget_eur=Decimal("100000000")))
        assert result is not None


# ===========================================================================
# Result Structure & Provenance
# ===========================================================================


class TestTechRoadmapResultStructure:
    """Test result structure and provenance."""

    def test_result_has_provenance(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_result_has_processing_time(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert_processing_time(result)

    def test_result_entity_name(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(entity_name="TechCo"))
        assert result.entity_name == "TechCo"

    def test_result_has_recommendations(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.recommendations, list)

    def test_result_deterministic(self):
        engine = TechnologyRoadmapEngine()
        inp = _make_input()
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.total_technologies == r2.total_technologies


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestTechRoadmapEdgeCases:
    """Edge case tests."""

    def test_no_custom_technologies(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(current_technologies=[]))
        assert result.total_technologies > 0

    def test_high_capex_budget(self):
        engine = TechnologyRoadmapEngine()
        result = engine.calculate(_make_input(
            annual_capex_budget_eur=Decimal("999999999999")))
        assert result is not None


# ===========================================================================
# Performance Tests
# ===========================================================================


class TestTechRoadmapPerformance:
    """Performance tests."""

    def test_single_roadmap_under_200ms(self):
        engine = TechnologyRoadmapEngine()
        with timed_block("single_roadmap", max_seconds=0.2):
            engine.calculate(_make_input())

    def test_50_roadmaps_under_5s(self):
        engine = TechnologyRoadmapEngine()
        with timed_block("50_roadmaps", max_seconds=5.0):
            for _ in range(50):
                engine.calculate(_make_input())

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement",
                                         "aviation", "shipping"])
    def test_each_sector_under_200ms(self, sector):
        engine = TechnologyRoadmapEngine()
        with timed_block(f"roadmap_{sector}", max_seconds=0.2):
            engine.calculate(_make_input(sector=sector))
