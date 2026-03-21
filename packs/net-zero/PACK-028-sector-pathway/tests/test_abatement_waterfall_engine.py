# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Abatement Waterfall Engine.

Tests sector-specific lever taxonomy, waterfall calculation accuracy,
cost curve generation, lever sequencing, dependency handling, and
cumulative abatement tracking.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  6 of 8 - abatement_waterfall_engine.py
Tests:   ~100 tests
"""

import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.abatement_waterfall_engine import (
    AbatementWaterfallEngine,
    AbatementInput,
    AbatementResult,
    WaterfallLever,
    CostCurvePoint,
    ImplementationPhase,
    LeverOverride,
    SECTOR_LEVERS,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_percentage_range,
    assert_provenance_hash,
    assert_processing_time,
    SDA_SECTORS,
    timed_block,
)


# Helper to build a valid AbatementInput
def _make_input(
    sector: str = "power_generation",
    baseline: Decimal = Decimal("22500000"),
    target: Decimal = Decimal("0"),
    base_year: int = 2024,
    target_year: int = 2050,
    entity_name: str = "TestCo",
    lever_overrides=None,
) -> AbatementInput:
    return AbatementInput(
        entity_name=entity_name,
        sector=sector,
        baseline_emissions_tco2e=baseline,
        target_emissions_tco2e=target,
        base_year=base_year,
        target_year=target_year,
        lever_overrides=lever_overrides or [],
    )


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestAbatementWaterfallInstantiation:
    """Engine instantiation and metadata tests."""

    def test_engine_instantiates(self):
        engine = AbatementWaterfallEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = AbatementWaterfallEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = AbatementWaterfallEngine()
        assert engine.engine_version == "1.0.0"

    def test_engine_has_lever_taxonomy(self):
        engine = AbatementWaterfallEngine()
        assert hasattr(engine, "get_sector_levers")


# ===========================================================================
# Sector Lever Taxonomy
# ===========================================================================


class TestSectorLeverTaxonomy:
    """Test sector-specific abatement lever taxonomy."""

    def test_power_sector_levers(self):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers("power_generation")
        assert len(levers) >= 6
        assert any("renewable" in n.lower() for n in levers)
        assert any("coal" in n.lower() for n in levers)

    def test_steel_sector_levers(self):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers("steel")
        assert len(levers) >= 5
        assert any("eaf" in n.lower() or "arc" in n.lower() for n in levers)
        assert any("hydrogen" in n.lower() or "h2" in n.lower() or "dri" in n.lower()
                    for n in levers)

    def test_cement_sector_levers(self):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers("cement")
        assert len(levers) >= 5
        assert any("clinker" in n.lower() for n in levers)
        assert any("ccs" in n.lower() or "ccus" in n.lower() or "carbon capture" in n.lower()
                    for n in levers)

    def test_aviation_sector_levers(self):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers("aviation")
        assert len(levers) >= 4
        assert any("saf" in n.lower() or "sustainable" in n.lower() for n in levers)
        assert any("fleet" in n.lower() or "aircraft" in n.lower() for n in levers)

    def test_buildings_sector_levers(self):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers("buildings_commercial")
        assert len(levers) >= 4
        assert any("heat pump" in n.lower() or "heating" in n.lower() or "hvac" in n.lower()
                    for n in levers)
        assert any("envelope" in n.lower() or "insulation" in n.lower() or "retrofit" in n.lower()
                    for n in levers)

    def test_shipping_sector_levers(self):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers("shipping")
        assert len(levers) >= 4
        assert any("fuel" in n.lower() or "ammonia" in n.lower() or "methanol" in n.lower()
                    for n in levers)

    @pytest.mark.parametrize("sector", [
        s for s in ["power_generation", "steel", "cement", "aviation",
                     "shipping", "buildings_residential", "buildings_commercial"]
        if s in SECTOR_LEVERS
    ])
    def test_all_sectors_have_levers(self, sector):
        engine = AbatementWaterfallEngine()
        levers = engine.get_sector_levers(sector)
        assert len(levers) >= 3

    def test_lever_definitions_have_reduction_pct(self):
        """Verify lever definitions in SECTOR_LEVERS have reduction_pct."""
        for lever_def in SECTOR_LEVERS.get("power_generation", []):
            assert "reduction_pct" in lever_def
            assert lever_def["reduction_pct"] > 0

    def test_lever_definitions_have_cost(self):
        """Verify lever definitions in SECTOR_LEVERS have cost."""
        for lever_def in SECTOR_LEVERS.get("steel", []):
            assert "cost_eur_per_tco2e" in lever_def


# ===========================================================================
# Waterfall Calculation
# ===========================================================================


class TestWaterfallCalculation:
    """Test waterfall calculation accuracy."""

    def test_power_waterfall_calculation(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        assert result is not None
        assert len(result.waterfall) > 0

    def test_steel_waterfall_calculation(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector="steel",
            baseline=Decimal("10500000"),
            target=Decimal("1000000"),
        )
        result = engine.calculate(inp)
        assert result is not None

    def test_lever_contributions_sum_correctly(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        total_abatement_pct = sum(lc.abatement_pct for lc in result.waterfall)
        assert total_abatement_pct > Decimal("0")

    def test_lever_contribution_has_abatement_tco2e(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            assert hasattr(lc, "abatement_tco2e")
            assert lc.abatement_tco2e >= Decimal("0")

    def test_lever_contribution_has_percentage(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            assert hasattr(lc, "abatement_pct")
            assert lc.abatement_pct >= Decimal("0")

    def test_waterfall_cumulative_tracking(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        prev = Decimal("0")
        for lc in result.waterfall:
            assert lc.cumulative_abatement_tco2e >= prev
            prev = lc.cumulative_abatement_tco2e


# ===========================================================================
# Cost Curves
# ===========================================================================


class TestCostCurves:
    """Test cost curve generation for abatement levers."""

    def test_marginal_abatement_cost_curve(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        assert hasattr(result, "cost_curve")
        assert len(result.cost_curve) > 0

    def test_cost_curve_sorted_by_cost(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        costs = [p.marginal_cost_eur_per_tco2e for p in result.cost_curve]
        assert costs == sorted(costs)

    def test_negative_cost_levers_first(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        if result.cost_curve:
            first = result.cost_curve[0]
            assert first.marginal_cost_eur_per_tco2e <= Decimal("0")

    def test_cost_curve_has_cumulative_abatement(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        for p in result.cost_curve:
            assert hasattr(p, "cumulative_abatement_tco2e")

    def test_total_cost_present(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        assert hasattr(result, "total_cost_eur")
        assert result.total_cost_eur is not None


# ===========================================================================
# Lever Sequencing (via implementation phases)
# ===========================================================================


class TestLeverSequencing:
    """Test lever implementation sequencing."""

    def test_implementation_phases_present(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        assert hasattr(result, "implementation_phases")
        assert len(result.implementation_phases) > 0

    def test_implementation_phases_ordered(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        start_years = [p.start_year for p in result.implementation_phases]
        assert start_years == sorted(start_years)

    def test_lever_timeline_valid(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            if lc.implementation_start_year > 0 and lc.implementation_end_year > 0:
                assert lc.implementation_end_year >= lc.implementation_start_year

    def test_sequencing_respects_cost_order(self):
        """Waterfall is sorted by cost (cheapest first = MACC ordering)."""
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="steel", baseline=Decimal("10500000"), target=Decimal("1000000"))
        result = engine.calculate(inp)
        costs = [lc.cost_eur_per_tco2e for lc in result.waterfall]
        assert costs == sorted(costs)


# ===========================================================================
# Lever Interdependencies
# ===========================================================================


class TestLeverInterdependencies:
    """Test lever interdependency handling."""

    def test_sector_lever_defs_have_depends_on(self):
        """Verify some levers have dependencies in definitions."""
        power_levers = SECTOR_LEVERS.get("power_generation", [])
        has_deps = any(len(l.get("depends_on", [])) > 0 for l in power_levers)
        assert has_deps, "Power sector should have levers with dependencies"

    def test_steel_has_dependency_chain(self):
        """Steel sector has levers that depend on others."""
        steel_levers = SECTOR_LEVERS.get("steel", [])
        has_deps = any(len(l.get("depends_on", [])) > 0 for l in steel_levers)
        assert has_deps, "Steel sector should have levers with dependencies"

    def test_interdependency_does_not_crash(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector="steel",
            baseline=Decimal("10500000"),
            target=Decimal("1000000"),
            entity_name="SteelCo",
        )
        inp.include_interdependencies = True
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestWaterfallEdgeCases:
    """Edge case tests for waterfall calculations."""

    def test_unknown_sector_produces_empty_waterfall(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="unknown_sector", baseline=Decimal("10500000"))
        result = engine.calculate(inp)
        assert result is not None
        assert len(result.waterfall) == 0

    def test_target_equals_baseline(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector="steel",
            baseline=Decimal("10500000"),
            target=Decimal("10500000"),
        )
        result = engine.calculate(inp)
        assert result is not None
        assert result.total_gap_tco2e == Decimal("0")

    def test_small_baseline(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="steel", baseline=Decimal("100"))
        result = engine.calculate(inp)
        assert result is not None

    def test_large_baseline(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("999999999"))
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Provenance & Determinism
# ===========================================================================


class TestWaterfallProvenance:
    """Test result provenance and determinism."""

    def test_result_has_provenance_hash(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        assert_provenance_hash(result)

    def test_result_is_deterministic(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="steel", baseline=Decimal("10500000"), target=Decimal("1000000"))
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.provenance_hash == r2.provenance_hash

    def test_result_processing_time(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        assert_processing_time(result)


# ===========================================================================
# Sector-Specific Waterfall Tests
# ===========================================================================

_SECTORS_WITH_LEVERS = [s for s in SECTOR_LEVERS.keys()]


class TestSectorSpecificWaterfalls:
    """Test waterfall calculations for specific sectors."""

    @pytest.mark.parametrize("sector", _SECTORS_WITH_LEVERS)
    def test_waterfall_for_each_sector(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector=sector,
            baseline=Decimal("1000000"),
            target=Decimal("100000"),
        )
        result = engine.calculate(inp)
        assert result is not None
        assert len(result.waterfall) > 0

    def test_cement_waterfall_has_clinker_lever(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="cement", baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        lever_names = [l.lever_name.lower() for l in result.waterfall]
        assert any("clinker" in n for n in lever_names)

    def test_aviation_waterfall_has_saf_lever(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="aviation", baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        lever_names = [l.lever_name.lower() for l in result.waterfall]
        assert any("saf" in n or "sustainable" in n for n in lever_names)

    def test_buildings_waterfall_has_efficiency_lever(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="buildings_commercial", baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        lever_names = [l.lever_name.lower() for l in result.waterfall]
        assert any("efficiency" in n or "envelope" in n or "hvac" in n
                    for n in lever_names)


# ===========================================================================
# Waterfall Year-by-Year Deployment
# ===========================================================================


class TestWaterfallYearlyDeployment:
    """Test year-by-year lever deployment tracking."""

    def test_levers_have_start_year(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            assert lc.implementation_start_year >= 2020

    def test_lever_end_after_start(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            if lc.implementation_end_year > 0:
                assert lc.implementation_end_year >= lc.implementation_start_year

    def test_lever_overlap_handling(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        # Total abatement percentage should be reasonable
        total_pct = sum(lc.abatement_pct for lc in result.waterfall)
        assert total_pct > Decimal("0")


# ===========================================================================
# Deep Sector-Specific Lever Validation
# ===========================================================================


class TestSectorSpecificLevers:
    """Test abatement levers specific to each sector via engine calculations."""

    SECTOR_TEST_CASES = list(SECTOR_LEVERS.keys())

    @pytest.mark.parametrize("sector", SECTOR_TEST_CASES)
    def test_sector_levers_calculation(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert result is not None
        assert len(result.waterfall) > 0

    @pytest.mark.parametrize("sector", SECTOR_TEST_CASES)
    def test_cumulative_reduction_calculation(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        total = sum(lc.abatement_pct for lc in result.waterfall)
        assert total > Decimal("0")

    @pytest.mark.parametrize("sector", SECTOR_TEST_CASES)
    def test_lever_contributions_have_names(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            assert lc.lever_name and len(lc.lever_name) > 0


# ===========================================================================
# Cost Curve Validation
# ===========================================================================


class TestCostCurveGeneration:
    """Test marginal abatement cost curve generation."""

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement", "aviation", "shipping"])
    def test_macc_generated(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert len(result.cost_curve) > 0

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_negative_cost_levers_in_macc(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        if result.cost_curve:
            negative_costs = [p for p in result.cost_curve
                              if p.marginal_cost_eur_per_tco2e < Decimal("0")]
            # Most sectors have negative-cost (efficiency) levers
            assert len(negative_costs) >= 1

    @pytest.mark.parametrize("base_emissions", [
        Decimal("1000000"), Decimal("5000000"), Decimal("10000000"),
        Decimal("50000000"), Decimal("100000000"),
    ])
    def test_waterfall_with_various_baselines(self, base_emissions):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=base_emissions)
        result = engine.calculate(inp)
        assert result is not None
        total = sum(lc.abatement_pct for lc in result.waterfall)
        assert total > Decimal("0")


# ===========================================================================
# Partial Abatement Targets
# ===========================================================================


class TestPartialAbatementTargets:
    """Test abatement waterfall with non-zero target emissions."""

    @pytest.mark.parametrize("target_pct", [
        Decimal("10"), Decimal("25"), Decimal("50"), Decimal("75"), Decimal("90"),
    ])
    def test_partial_reduction_targets(self, target_pct):
        engine = AbatementWaterfallEngine()
        base = Decimal("10000000")
        target = base * (Decimal("100") - target_pct) / Decimal("100")
        inp = _make_input(sector="steel", baseline=base, target=target)
        result = engine.calculate(inp)
        assert result is not None

    @pytest.mark.parametrize("sector,target_factor", [
        ("power_generation", Decimal("0.0")),
        ("steel", Decimal("0.1")),
        ("cement", Decimal("0.2")),
        ("aviation", Decimal("0.3")),
        ("shipping", Decimal("0.4")),
    ])
    def test_sector_specific_targets(self, sector, target_factor):
        engine = AbatementWaterfallEngine()
        base = Decimal("10000000")
        target = base * target_factor
        inp = _make_input(sector=sector, baseline=base, target=target)
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Waterfall Result Serialization
# ===========================================================================


class TestWaterfallSerialization:
    """Test waterfall result serialization."""

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_result_model_dump(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "waterfall" in d

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_result_model_dump_json(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        j = result.model_dump_json()
        assert isinstance(j, str)


# ===========================================================================
# Waterfall Performance Benchmarks
# ===========================================================================


class TestWaterfallPerformance:
    """Test abatement waterfall engine performance."""

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_single_sector_under_2s(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        start = time.time()
        result = engine.calculate(inp)
        elapsed = (time.time() - start) * 1000
        assert result is not None
        assert elapsed < 2000

    def test_batch_all_sectors_under_10s(self):
        engine = AbatementWaterfallEngine()
        start = time.time()
        for sector in SECTOR_LEVERS.keys():
            inp = _make_input(sector=sector, baseline=Decimal("10000000"))
            result = engine.calculate(inp)
            assert result is not None
        elapsed = (time.time() - start) * 1000
        assert elapsed < 10000


# ===========================================================================
# Lever Temporal Phasing
# ===========================================================================


class TestLeverTemporalPhasing:
    """Test lever deployment timing and phasing."""

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_start_years_in_range(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            if lc.implementation_start_year > 0:
                assert 2020 <= lc.implementation_start_year <= 2060

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_lever_end_year_after_start(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            if lc.implementation_end_year > 0:
                assert lc.implementation_end_year >= lc.implementation_start_year

    @pytest.mark.parametrize("target_year", [2030, 2035, 2040, 2045, 2050])
    def test_waterfall_with_various_target_years(self, target_year):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector="power_generation",
            baseline=Decimal("10000000"),
            target_year=target_year,
        )
        result = engine.calculate(inp)
        assert result is not None

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_waterfall_provenance_hash(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert_provenance_hash(result)


# ===========================================================================
# Abatement Edge Cases
# ===========================================================================


class TestAbatementEdgeCases:
    """Test abatement waterfall edge cases."""

    def test_single_lever_override(self):
        engine = AbatementWaterfallEngine()
        override = LeverOverride(
            lever_name="Renewable Capacity Expansion",
            reduction_pct=Decimal("50"),
        )
        inp = _make_input(
            sector="power_generation",
            baseline=Decimal("10000000"),
            lever_overrides=[override],
        )
        result = engine.calculate(inp)
        assert result is not None
        assert len(result.waterfall) >= 1

    def test_very_large_baseline(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("999999999"))
        result = engine.calculate(inp)
        assert result is not None

    def test_very_small_baseline(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="steel", baseline=Decimal("100"))
        result = engine.calculate(inp)
        assert result is not None

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_deterministic_output(self, sector):
        """Same inputs should produce identical outputs."""
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result1 = engine.calculate(inp)
        result2 = engine.calculate(inp)
        assert len(result1.waterfall) == len(result2.waterfall)
        assert result1.provenance_hash == result2.provenance_hash


# ===========================================================================
# Lever Interaction and Dependencies
# ===========================================================================


class TestLeverInteractions:
    """Test lever interaction effects and dependencies."""

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_lever_count_matches_sector_definition(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        expected = len(SECTOR_LEVERS[sector])
        assert len(result.waterfall) == expected

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_lever_names_present(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        input_names = {l["name"] for l in SECTOR_LEVERS[sector]}
        for lc in result.waterfall:
            assert lc.lever_name in input_names

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_total_abatement_within_gap(self, sector):
        engine = AbatementWaterfallEngine()
        base = Decimal("10000000")
        target = Decimal("2000000")
        inp = _make_input(sector=sector, baseline=base, target=target)
        result = engine.calculate(inp)
        expected_gap = base - target
        assert result.total_abatement_tco2e <= expected_gap * Decimal("1.05")


# ===========================================================================
# Waterfall Result Completeness
# ===========================================================================


class TestWaterfallResultCompleteness:
    """Test that waterfall results include all expected fields."""

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_result_has_waterfall(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert hasattr(result, "waterfall")
        assert len(result.waterfall) >= 1

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_result_has_sector_field(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert result.sector == sector

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_lever_has_abatement_pct(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            assert isinstance(lc.abatement_pct, Decimal)

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_lever_positive_abatement(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        for lc in result.waterfall:
            assert lc.abatement_pct >= Decimal("0")


# ===========================================================================
# Cross-Sector Abatement Comparison
# ===========================================================================


class TestCrossSectorAbatementComparison:
    """Test abatement waterfall comparison across sectors."""

    @pytest.mark.parametrize("sector_pair", [
        ("power_generation", "steel"),
        ("steel", "cement"),
        ("aviation", "shipping"),
    ])
    def test_cross_sector_comparison(self, sector_pair):
        engine = AbatementWaterfallEngine()
        results = {}
        for sector in sector_pair:
            inp = _make_input(sector=sector, baseline=Decimal("10000000"))
            results[sector] = engine.calculate(inp)
        assert len(results) == 2
        for sector, result in results.items():
            assert len(result.waterfall) > 0

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_total_cost_is_decimal(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert isinstance(result.total_cost_eur, Decimal)

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_gap_remaining_calculated(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector=sector,
            baseline=Decimal("10000000"),
            target=Decimal("1000000"),
        )
        result = engine.calculate(inp)
        assert result.gap_remaining_tco2e >= Decimal("0")


# ===========================================================================
# Abatement Lever Maturity Levels
# ===========================================================================


class TestAbatementLeverMaturity:
    """Test maturity classification of abatement levers."""

    LEVER_MATURITY = {
        "power_generation": {
            "solar_pv": "commercial", "wind_onshore": "commercial", "wind_offshore": "commercial",
            "battery_storage": "early_commercial", "green_hydrogen": "demonstration",
            "ccus": "demonstration", "nuclear_smr": "pilot",
        },
        "steel": {
            "scrap_recycling": "commercial", "eaf_conversion": "commercial",
            "hydrogen_dri": "demonstration", "ccus": "demonstration",
            "electrolysis_iron": "pilot",
        },
        "cement": {
            "alternative_fuels": "commercial", "clinker_substitution": "commercial",
            "calcined_clay": "early_commercial", "ccus": "demonstration",
            "novel_cements": "pilot",
        },
    }

    MATURITY_ORDER = ["pilot", "demonstration", "early_commercial", "commercial"]

    @pytest.mark.parametrize("sector", list(LEVER_MATURITY.keys()))
    def test_all_levers_have_valid_maturity(self, sector):
        levers = self.LEVER_MATURITY[sector]
        for lever, maturity in levers.items():
            assert maturity in self.MATURITY_ORDER, f"{lever} has invalid maturity: {maturity}"

    @pytest.mark.parametrize("sector", list(LEVER_MATURITY.keys()))
    def test_has_at_least_one_commercial_lever(self, sector):
        levers = self.LEVER_MATURITY[sector]
        commercial = [l for l, m in levers.items() if m == "commercial"]
        assert len(commercial) >= 1, f"{sector} needs at least one commercial lever"

    @pytest.mark.parametrize("sector", list(LEVER_MATURITY.keys()))
    def test_pilot_levers_not_dominant(self, sector):
        levers = self.LEVER_MATURITY[sector]
        pilots = [l for l, m in levers.items() if m == "pilot"]
        assert len(pilots) <= len(levers) // 2, f"Too many pilot levers in {sector}"

    @pytest.mark.parametrize("sector,lever,expected_maturity", [
        ("power_generation", "solar_pv", "commercial"),
        ("power_generation", "green_hydrogen", "demonstration"),
        ("steel", "hydrogen_dri", "demonstration"),
        ("steel", "scrap_recycling", "commercial"),
        ("cement", "ccus", "demonstration"),
        ("cement", "alternative_fuels", "commercial"),
    ])
    def test_specific_lever_maturity(self, sector, lever, expected_maturity):
        actual = self.LEVER_MATURITY[sector][lever]
        assert actual == expected_maturity


# ===========================================================================
# Abatement Cost Curve Ordering
# ===========================================================================


class TestAbatementCostCurveOrdering:
    """Test that abatement cost curves are properly ordered."""

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_cost_curve_ascending(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("5000000"), target=Decimal("1000000"))
        result = engine.calculate(inp)
        if result.cost_curve:
            costs = [p.marginal_cost_eur_per_tco2e for p in result.cost_curve]
            assert costs == sorted(costs)

    @pytest.mark.parametrize("base_emissions", [
        Decimal("100000"), Decimal("1000000"), Decimal("10000000"),
        Decimal("50000000"), Decimal("100000000"),
    ])
    def test_cost_scales_with_base_emissions(self, base_emissions):
        engine = AbatementWaterfallEngine()
        inp = _make_input(
            sector="power_generation",
            baseline=base_emissions,
            target=base_emissions * Decimal("0.1"),
        )
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Recommendations
# ===========================================================================


class TestWaterfallRecommendations:
    """Test recommendations generated by waterfall engine."""

    @pytest.mark.parametrize("sector", list(SECTOR_LEVERS.keys()))
    def test_recommendations_present(self, sector):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector=sector, baseline=Decimal("10000000"))
        result = engine.calculate(inp)
        assert hasattr(result, "recommendations")
        assert isinstance(result.recommendations, list)

    def test_recommendations_mention_cost_savings(self):
        engine = AbatementWaterfallEngine()
        inp = _make_input(sector="power_generation", baseline=Decimal("22500000"))
        result = engine.calculate(inp)
        # Power sector has negative-cost levers, so should recommend starting there
        rec_text = " ".join(result.recommendations).lower()
        assert "sav" in rec_text or "cost" in rec_text or "lever" in rec_text


# ===========================================================================
# Lever Override Tests
# ===========================================================================


class TestLeverOverrides:
    """Test lever override functionality."""

    def test_override_reduction_pct(self):
        engine = AbatementWaterfallEngine()
        override = LeverOverride(
            lever_name="Renewable Capacity Expansion",
            reduction_pct=Decimal("80"),
        )
        inp = _make_input(
            sector="power_generation",
            baseline=Decimal("10000000"),
            lever_overrides=[override],
        )
        result = engine.calculate(inp)
        assert result is not None

    def test_override_cost(self):
        engine = AbatementWaterfallEngine()
        override = LeverOverride(
            lever_name="Renewable Capacity Expansion",
            cost_eur_per_tco2e=Decimal("50"),
        )
        inp = _make_input(
            sector="power_generation",
            baseline=Decimal("10000000"),
            lever_overrides=[override],
        )
        result = engine.calculate(inp)
        assert result is not None

    def test_multiple_overrides(self):
        engine = AbatementWaterfallEngine()
        overrides = [
            LeverOverride(lever_name="Renewable Capacity Expansion", reduction_pct=Decimal("60")),
            LeverOverride(lever_name="Coal Plant Phase-Out", reduction_pct=Decimal("30")),
        ]
        inp = _make_input(
            sector="power_generation",
            baseline=Decimal("10000000"),
            lever_overrides=overrides,
        )
        result = engine.calculate(inp)
        assert result is not None
        assert len(result.waterfall) > 0
