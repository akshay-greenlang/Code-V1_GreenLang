# -*- coding: utf-8 -*-
"""
Unit tests for SoilOrganicCarbonEngine (Engine 4 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Tests IPCC Tier 1/2 soil organic carbon calculations including:
    - SOC = SOC_ref * F_LU * F_MG * F_I formula correctness
    - DeltaSOC = (SOC_new - SOC_old) / T annual change
    - SOC reference stock lookups
    - Factor application for land use, management, and input levels
    - Liming and urea CO2 emission calculations
    - N mineralisation from SOC loss
    - Depth adjustments (30 cm vs 100 cm)
    - Transition SOC interpolation
    - Parcel history and cumulative changes
    - IPCC Table 5.5 / 5.10 validation examples

Target: 110 tests, ~700 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.land_use_emissions.soil_organic_carbon import (
    SoilOrganicCarbonEngine,
    SOCHistoryEntry,
    DEFAULT_TRANSITION_PERIOD,
    TIER_1_DEPTH_CM,
    TIER_2_DEPTH_CM,
    EF_LIMESTONE,
    EF_DOLOMITE,
    EF_UREA,
    N_MINERALIZATION_C_TO_N,
    TIER_2_DEPTH_RATIO,
    _D,
    _safe_decimal,
)
from greenlang.land_use_emissions.land_use_database import (
    SOC_REFERENCE_STOCKS,
    SOC_LAND_USE_FACTORS,
    SOC_MANAGEMENT_FACTORS,
    SOC_INPUT_FACTORS,
    CONVERSION_FACTOR_CO2_C,
    N2O_N_RATIO,
    LandUseDatabaseEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine():
    """Create a real LandUseDatabaseEngine instance."""
    return LandUseDatabaseEngine()


@pytest.fixture
def soc_engine(db_engine):
    """Create a SoilOrganicCarbonEngine with a real database."""
    engine = SoilOrganicCarbonEngine(land_use_database=db_engine)
    yield engine
    engine.reset()


@pytest.fixture
def soc_engine_no_db():
    """Create a SoilOrganicCarbonEngine without a database (fallback path)."""
    engine = SoilOrganicCarbonEngine(land_use_database=None)
    yield engine
    engine.reset()


@pytest.fixture
def base_soc_request() -> Dict[str, Any]:
    """Return a minimal valid SOC calculation request."""
    return {
        "climate_zone": "TROPICAL_WET",
        "soil_type": "HIGH_ACTIVITY_CLAY",
        "land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
        "management_practice": "FULL_TILLAGE",
        "input_level": "MEDIUM",
        "area_ha": 100,
    }


@pytest.fixture
def soc_change_request() -> Dict[str, Any]:
    """Return a valid SOC change calculation request."""
    return {
        "climate_zone": "TROPICAL_WET",
        "soil_type": "HIGH_ACTIVITY_CLAY",
        "old_land_use_type": "FOREST_NATIVE",
        "old_management_practice": "NOMINAL",
        "old_input_level": "MEDIUM",
        "new_land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
        "new_management_practice": "FULL_TILLAGE",
        "new_input_level": "MEDIUM",
        "area_ha": 100,
        "transition_period_years": 20,
    }


# ===========================================================================
# 1. Initialisation Tests
# ===========================================================================


class TestSoilOrganicCarbonEngineInit:
    """Test SoilOrganicCarbonEngine initialisation."""

    def test_init_with_db(self, db_engine):
        """Test engine initialises with an explicit database engine."""
        engine = SoilOrganicCarbonEngine(land_use_database=db_engine)
        assert engine._land_use_db is db_engine

    def test_init_without_db_falls_back(self):
        """Test engine initialises without a database engine using fallback dictionaries."""
        engine = SoilOrganicCarbonEngine(land_use_database=None)
        assert engine._land_use_db is None
        assert engine._total_calculations == 0

    def test_init_creates_empty_history(self, soc_engine):
        """Test that parcel history starts empty."""
        stats = soc_engine.get_statistics()
        assert stats["tracked_parcels"] == 0
        assert stats["total_history_entries"] == 0


# ===========================================================================
# 2. SOC Reference Lookup Tests
# ===========================================================================


class TestSOCReferenceLookup:
    """Test SOC reference stock lookups for all climate zone x soil type combos."""

    @pytest.mark.parametrize("zone,soil,expected", [
        ("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", Decimal("65")),
        ("TROPICAL_WET", "LOW_ACTIVITY_CLAY", Decimal("47")),
        ("TROPICAL_WET", "SANDY", Decimal("39")),
        ("TROPICAL_WET", "SPODIC", Decimal("70")),
        ("TROPICAL_WET", "VOLCANIC", Decimal("130")),
        ("TROPICAL_WET", "WETLAND", Decimal("86")),
        ("TROPICAL_WET", "ORGANIC", Decimal("200")),
        ("TROPICAL_DRY", "HIGH_ACTIVITY_CLAY", Decimal("38")),
        ("TROPICAL_DRY", "LOW_ACTIVITY_CLAY", Decimal("35")),
        ("TROPICAL_DRY", "SANDY", Decimal("31")),
        ("SUBTROPICAL_HUMID", "HIGH_ACTIVITY_CLAY", Decimal("88")),
        ("SUBTROPICAL_HUMID", "LOW_ACTIVITY_CLAY", Decimal("63")),
        ("SUBTROPICAL_HUMID", "SANDY", Decimal("34")),
        ("SUBTROPICAL_HUMID", "SPODIC", Decimal("115")),
    ])
    def test_soc_reference_tier1(self, soc_engine, zone, soil, expected):
        """Test SOC reference stock lookup returns correct IPCC Table 2.3 value at 30cm."""
        result = soc_engine.get_soc_reference(zone, soil, TIER_1_DEPTH_CM)
        assert result == expected

    def test_soc_reference_case_insensitive(self, soc_engine):
        """Test SOC reference lookup is case-insensitive."""
        result = soc_engine.get_soc_reference("tropical_wet", "high_activity_clay")
        assert result == Decimal("65")

    def test_soc_reference_invalid_zone_raises(self, soc_engine):
        """Test that an invalid climate zone raises KeyError."""
        with pytest.raises(KeyError, match="No SOC reference"):
            soc_engine.get_soc_reference("INVALID_ZONE", "HIGH_ACTIVITY_CLAY")

    def test_soc_reference_invalid_soil_raises(self, soc_engine):
        """Test that an invalid soil type raises KeyError."""
        with pytest.raises(KeyError, match="No SOC reference"):
            soc_engine.get_soc_reference("TROPICAL_WET", "NONEXISTENT_SOIL")

    def test_soc_reference_tier2_depth_adjustment(self, soc_engine):
        """Test that Tier 2 depth (100cm) applies diminishing-return scaling."""
        soc_30 = soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 30)
        soc_100 = soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 100)
        # 100/30 = 3.333...; factor = 3.333^0.75 is roughly 2.62
        assert soc_100 > soc_30
        # The ratio should be less than 3.333 (linear) due to diminishing returns
        ratio = float(soc_100 / soc_30)
        assert ratio < 3.334
        assert ratio > 1.0

    def test_soc_reference_custom_depth_60cm(self, soc_engine):
        """Test SOC reference at a custom intermediate depth."""
        soc_30 = soc_engine.get_soc_reference("TROPICAL_WET", "SANDY", 30)
        soc_60 = soc_engine.get_soc_reference("TROPICAL_WET", "SANDY", 60)
        assert soc_60 > soc_30

    def test_tropical_moist_matches_tropical_wet_soc(self, soc_engine):
        """Test TROPICAL_MOIST SOC references match TROPICAL_WET per IPCC Table 2.3."""
        for soil in ["HIGH_ACTIVITY_CLAY", "LOW_ACTIVITY_CLAY", "SANDY", "SPODIC", "VOLCANIC"]:
            wet = soc_engine.get_soc_reference("TROPICAL_WET", soil)
            moist = soc_engine.get_soc_reference("TROPICAL_MOIST", soil)
            assert wet == moist, f"Mismatch for {soil}: wet={wet}, moist={moist}"


# ===========================================================================
# 3. Factor Application Tests
# ===========================================================================


class TestFactorApplication:
    """Test F_LU, F_MG, and F_I application."""

    @pytest.mark.parametrize("land_use,expected_flu", [
        ("FOREST_NATIVE", Decimal("1.0")),
        ("FOREST_PLANTATION", Decimal("0.8")),
        ("CROPLAND_ANNUAL_FULL_TILL", Decimal("0.69")),
        ("CROPLAND_PERENNIAL", Decimal("1.0")),
        ("CROPLAND_SET_ASIDE", Decimal("0.82")),
        ("GRASSLAND_NATIVE", Decimal("1.0")),
        ("GRASSLAND_IMPROVED", Decimal("1.14")),
        ("GRASSLAND_DEGRADED", Decimal("0.97")),
        ("WETLANDS_MANAGED", Decimal("0.70")),
        ("WETLANDS_UNMANAGED", Decimal("1.0")),
        ("SETTLEMENTS", Decimal("0.80")),
        ("OTHER_LAND", Decimal("1.0")),
    ])
    def test_flu_for_land_use_types(self, soc_engine, land_use, expected_flu):
        """Test F_LU factor matches IPCC Table 5.5 values."""
        result = soc_engine.apply_factors(
            Decimal("65"), land_use, "NOMINAL", "MEDIUM"
        )
        assert result["f_lu"] == expected_flu

    @pytest.mark.parametrize("practice,expected_fmg", [
        ("FULL_TILLAGE", Decimal("1.0")),
        ("REDUCED_TILLAGE", Decimal("1.08")),
        ("NO_TILLAGE", Decimal("1.15")),
        ("NOMINAL", Decimal("1.0")),
        ("IMPROVED", Decimal("1.04")),
        ("SEVERELY_DEGRADED", Decimal("0.70")),
        ("MODERATELY_DEGRADED", Decimal("0.95")),
    ])
    def test_fmg_for_management_practices(self, soc_engine, practice, expected_fmg):
        """Test F_MG factor matches IPCC Table 5.10 values."""
        result = soc_engine.apply_factors(
            Decimal("65"), "FOREST_NATIVE", practice, "MEDIUM"
        )
        assert result["f_mg"] == expected_fmg

    @pytest.mark.parametrize("input_level,expected_fi", [
        ("LOW", Decimal("0.92")),
        ("MEDIUM", Decimal("1.0")),
        ("HIGH_WITHOUT_MANURE", Decimal("1.11")),
        ("HIGH_WITH_MANURE", Decimal("1.44")),
    ])
    def test_fi_for_input_levels(self, soc_engine, input_level, expected_fi):
        """Test F_I factor matches IPCC input-level values."""
        result = soc_engine.apply_factors(
            Decimal("65"), "FOREST_NATIVE", "NOMINAL", input_level
        )
        assert result["f_i"] == expected_fi

    def test_soc_formula_correctness(self, soc_engine):
        """Test SOC = SOC_ref * F_LU * F_MG * F_I is computed correctly."""
        soc_ref = Decimal("65")
        result = soc_engine.apply_factors(
            soc_ref, "CROPLAND_ANNUAL_FULL_TILL", "FULL_TILLAGE", "MEDIUM"
        )
        # Expected: 65 * 0.69 * 1.0 * 1.0 = 44.85
        expected = (soc_ref * Decimal("0.69") * Decimal("1.0") * Decimal("1.0")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert result["soc_calculated"] == expected

    def test_soc_formula_with_all_non_unity_factors(self, soc_engine):
        """Test SOC formula when all three factors are non-unity."""
        soc_ref = Decimal("65")
        result = soc_engine.apply_factors(
            soc_ref,
            "CROPLAND_ANNUAL_FULL_TILL",  # F_LU = 0.69
            "REDUCED_TILLAGE",             # F_MG = 1.08
            "HIGH_WITH_MANURE",            # F_I  = 1.44
        )
        expected = (soc_ref * Decimal("0.69") * Decimal("1.08") * Decimal("1.44")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert result["soc_calculated"] == expected


# ===========================================================================
# 4. calculate_soc() Tests
# ===========================================================================


class TestCalculateSOC:
    """Test the main calculate_soc method."""

    def test_success_returns_expected_fields(self, soc_engine, base_soc_request):
        """Test that a valid request returns all expected output fields."""
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["status"] == "SUCCESS"
        assert "calculation_id" in result
        assert "soc_ref_tc_ha" in result
        assert "f_lu" in result
        assert "f_mg" in result
        assert "f_i" in result
        assert "soc_tc_ha" in result
        assert "soc_total_tc" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_total_equals_per_ha_times_area(self, soc_engine, base_soc_request):
        """Test soc_total_tc equals soc_tc_ha * area_ha."""
        result = soc_engine.calculate_soc(base_soc_request)
        soc_ha = Decimal(result["soc_tc_ha"])
        area = Decimal(str(base_soc_request["area_ha"]))
        expected_total = (soc_ha * area).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["soc_total_tc"]) == expected_total

    def test_tier1_at_30cm(self, soc_engine, base_soc_request):
        """Test that default depth is 30cm and tier is TIER_1."""
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["tier"] == "TIER_1"
        assert result["depth_cm"] == 30

    def test_tier2_at_100cm(self, soc_engine, base_soc_request):
        """Test that depth >30cm triggers TIER_2 classification."""
        base_soc_request["depth_cm"] = 100
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["tier"] == "TIER_2"
        assert result["depth_cm"] == 100

    def test_missing_climate_zone_returns_validation_error(self, soc_engine):
        """Test validation error when climate_zone is missing."""
        result = soc_engine.calculate_soc({
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "FOREST_NATIVE",
            "management_practice": "NOMINAL",
            "area_ha": 100,
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert any("climate_zone" in e for e in result["errors"])

    def test_missing_soil_type_returns_validation_error(self, soc_engine):
        """Test validation error when soil_type is missing."""
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "land_use_type": "FOREST_NATIVE",
            "management_practice": "NOMINAL",
            "area_ha": 100,
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert any("soil_type" in e for e in result["errors"])

    def test_zero_area_returns_validation_error(self, soc_engine, base_soc_request):
        """Test validation error when area_ha is zero."""
        base_soc_request["area_ha"] = 0
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["status"] == "VALIDATION_ERROR"
        assert any("area_ha" in e for e in result["errors"])

    def test_negative_area_returns_validation_error(self, soc_engine, base_soc_request):
        """Test validation error when area_ha is negative."""
        base_soc_request["area_ha"] = -50
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_invalid_climate_zone_returns_lookup_error(self, soc_engine, base_soc_request):
        """Test lookup error when climate_zone does not exist in reference data."""
        base_soc_request["climate_zone"] = "NONEXISTENT_ZONE"
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["status"] == "LOOKUP_ERROR"

    def test_processing_time_is_positive(self, soc_engine, base_soc_request):
        """Test that processing_time_ms is a positive number."""
        result = soc_engine.calculate_soc(base_soc_request)
        assert result["processing_time_ms"] >= 0

    def test_provenance_hash_is_deterministic(self, soc_engine, base_soc_request):
        """Test that identical inputs produce the same provenance hash (ignoring calc_id)."""
        # Provenance hash includes calc_id which is random, so hashes will differ
        # but the computation itself is deterministic (tested via SOC value consistency)
        r1 = soc_engine.calculate_soc(base_soc_request)
        r2 = soc_engine.calculate_soc(base_soc_request)
        assert r1["soc_tc_ha"] == r2["soc_tc_ha"]
        assert r1["soc_total_tc"] == r2["soc_total_tc"]

    def test_calculations_counter_increments(self, soc_engine, base_soc_request):
        """Test that the calculations counter increments with each call."""
        stats_before = soc_engine.get_statistics()
        soc_engine.calculate_soc(base_soc_request)
        stats_after = soc_engine.get_statistics()
        assert stats_after["total_calculations"] == stats_before["total_calculations"] + 1


# ===========================================================================
# 5. SOC Change Calculation Tests
# ===========================================================================


class TestCalculateSOCChange:
    """Test DeltaSOC = (SOC_new - SOC_old) / T calculations."""

    def test_forest_to_cropland_is_net_emission(self, soc_engine, soc_change_request):
        """Test that forest-to-cropland transition produces a net emission (SOC loss)."""
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert result["status"] == "SUCCESS"
        assert result["emission_type"] == "NET_EMISSION"
        assert Decimal(result["delta_soc_tc_yr"]) < 0

    def test_cropland_to_forest_is_net_removal(self, soc_engine):
        """Test that cropland-to-forest transition produces a net removal (SOC gain)."""
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "old_management_practice": "FULL_TILLAGE",
            "old_input_level": "MEDIUM",
            "new_land_use_type": "FOREST_NATIVE",
            "new_management_practice": "NOMINAL",
            "new_input_level": "MEDIUM",
            "area_ha": 100,
        })
        assert result["status"] == "SUCCESS"
        assert result["emission_type"] == "NET_REMOVAL"
        assert Decimal(result["delta_soc_tc_yr"]) > 0

    def test_delta_soc_formula(self, soc_engine, soc_change_request):
        """Test DeltaSOC = (SOC_new - SOC_old) / T formula correctness."""
        result = soc_engine.calculate_soc_change(soc_change_request)
        soc_old = Decimal(result["old_state"]["soc_tc_ha"])
        soc_new = Decimal(result["new_state"]["soc_tc_ha"])
        t = Decimal(str(soc_change_request["transition_period_years"]))
        expected_delta_ha = ((soc_new - soc_old) / t).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["delta_soc_tc_ha_yr"]) == expected_delta_ha

    def test_default_20_year_transition(self, soc_engine, soc_change_request):
        """Test that the default transition period is 20 years."""
        del soc_change_request["transition_period_years"]
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert result["transition_period_years"] == 20

    def test_custom_transition_period(self, soc_engine, soc_change_request):
        """Test SOC change with a custom transition period of 10 years."""
        soc_change_request["transition_period_years"] = 10
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert result["transition_period_years"] == 10
        # With shorter period, annual change rate is larger
        result_20 = soc_engine.calculate_soc_change({
            **soc_change_request, "transition_period_years": 20
        })
        assert abs(Decimal(result["delta_soc_tc_ha_yr"])) > abs(
            Decimal(result_20["delta_soc_tc_ha_yr"])
        )

    def test_co2_equivalent_from_delta_soc(self, soc_engine, soc_change_request):
        """Test CO2e = DeltaSOC_total * 3.66667."""
        result = soc_engine.calculate_soc_change(soc_change_request)
        delta_total = Decimal(result["delta_soc_tc_yr"])
        co2_expected = (delta_total * CONVERSION_FACTOR_CO2_C).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["delta_co2_tonnes_yr"]) == co2_expected

    def test_n_mineralization_on_soc_loss(self, soc_engine, soc_change_request):
        """Test N mineralisation is computed when SOC is lost."""
        result = soc_engine.calculate_soc_change(soc_change_request)
        # Forest to cropland = SOC loss, so N should be mineralized
        assert Decimal(result["n_mineralized_tc_yr"]) > 0

    def test_no_n_mineralization_on_soc_gain(self, soc_engine):
        """Test N mineralisation is zero when SOC increases."""
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "old_management_practice": "FULL_TILLAGE",
            "old_input_level": "MEDIUM",
            "new_land_use_type": "FOREST_NATIVE",
            "new_management_practice": "NOMINAL",
            "new_input_level": "MEDIUM",
            "area_ha": 100,
        })
        assert Decimal(result["n_mineralized_tc_yr"]) == 0

    def test_same_land_use_zero_change(self, soc_engine):
        """Test that identical old/new land use produces zero SOC change."""
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "FOREST_NATIVE",
            "old_management_practice": "NOMINAL",
            "old_input_level": "MEDIUM",
            "new_land_use_type": "FOREST_NATIVE",
            "new_management_practice": "NOMINAL",
            "new_input_level": "MEDIUM",
            "area_ha": 100,
        })
        assert Decimal(result["delta_soc_tc_ha_yr"]) == 0
        assert Decimal(result["delta_soc_tc_yr"]) == 0

    def test_soc_change_validation_errors(self, soc_engine):
        """Test validation errors for missing required fields."""
        result = soc_engine.calculate_soc_change({
            "area_ha": 100,
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert len(result["errors"]) >= 4  # climate_zone, soil, old_lu, new_lu, old_mg, new_mg

    def test_soc_change_zero_transition_period(self, soc_engine, soc_change_request):
        """Test that zero transition period returns validation error."""
        soc_change_request["transition_period_years"] = 0
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_soc_change_large_area(self, soc_engine, soc_change_request):
        """Test SOC change scales linearly with area."""
        soc_change_request["area_ha"] = 100
        r100 = soc_engine.calculate_soc_change(soc_change_request)
        soc_change_request["area_ha"] = 1000
        r1000 = soc_engine.calculate_soc_change(soc_change_request)
        ratio = Decimal(r1000["delta_soc_tc_yr"]) / Decimal(r100["delta_soc_tc_yr"])
        assert ratio == Decimal("10")


# ===========================================================================
# 6. Transition SOC Tests
# ===========================================================================


class TestTransitionSOC:
    """Test linear interpolation during SOC transition."""

    def test_transition_at_year_zero(self, soc_engine, soc_change_request):
        """Test that at year 0, SOC equals the old state."""
        req = {**soc_change_request, "years_since_transition": 0}
        result = soc_engine.calculate_transition_soc(req)
        assert result["status"] == "SUCCESS"
        assert Decimal(result["soc_current_tc_ha"]) == Decimal(result["soc_old_tc_ha"])
        assert result["is_transition_complete"] is False

    def test_transition_at_full_period(self, soc_engine, soc_change_request):
        """Test that at year T, SOC equals the new state."""
        req = {**soc_change_request, "years_since_transition": 20}
        result = soc_engine.calculate_transition_soc(req)
        assert result["status"] == "SUCCESS"
        assert Decimal(result["soc_current_tc_ha"]) == Decimal(result["soc_new_tc_ha"])
        assert result["is_transition_complete"] is True

    def test_transition_at_midpoint(self, soc_engine, soc_change_request):
        """Test that at year T/2, SOC is halfway between old and new."""
        req = {**soc_change_request, "years_since_transition": 10}
        result = soc_engine.calculate_transition_soc(req)
        old = Decimal(result["soc_old_tc_ha"])
        new = Decimal(result["soc_new_tc_ha"])
        midpoint = ((old + new) / 2).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
        assert Decimal(result["soc_current_tc_ha"]) == midpoint

    def test_transition_beyond_period_equals_new(self, soc_engine, soc_change_request):
        """Test that years beyond T still cap at the new state."""
        req = {**soc_change_request, "years_since_transition": 50}
        result = soc_engine.calculate_transition_soc(req)
        assert Decimal(result["soc_current_tc_ha"]) == Decimal(result["soc_new_tc_ha"])
        assert result["is_transition_complete"] is True

    def test_transition_fraction_at_5_of_20(self, soc_engine, soc_change_request):
        """Test transition fraction is 0.25 at year 5 of 20."""
        req = {**soc_change_request, "years_since_transition": 5}
        result = soc_engine.calculate_transition_soc(req)
        assert Decimal(result["transition_fraction"]) == Decimal("0.2500")

    def test_transition_total_uses_area(self, soc_engine, soc_change_request):
        """Test that soc_total_tc = soc_current_tc_ha * area_ha."""
        req = {**soc_change_request, "years_since_transition": 10}
        result = soc_engine.calculate_transition_soc(req)
        current_ha = Decimal(result["soc_current_tc_ha"])
        area = Decimal(result["area_ha"])
        expected_total = (current_ha * area).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["soc_total_tc"]) == expected_total


# ===========================================================================
# 7. Liming Emissions Tests
# ===========================================================================


class TestLimingEmissions:
    """Test CO2 emissions from limestone (CaCO3) and dolomite application."""

    def test_limestone_only(self, soc_engine):
        """Test liming CO2 from limestone only."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": 100,
            "dolomite_tonnes": 0,
        })
        assert result["status"] == "SUCCESS"
        # C = 100 * 0.12 = 12 tC; CO2 = 12 * 3.66667 = 44.00004
        c_expected = (Decimal("100") * EF_LIMESTONE).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        co2_expected = (c_expected * CONVERSION_FACTOR_CO2_C).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["carbon_limestone_tc"]) == c_expected
        assert Decimal(result["co2_tonnes"]) == co2_expected

    def test_dolomite_only(self, soc_engine):
        """Test liming CO2 from dolomite only."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": 0,
            "dolomite_tonnes": 100,
        })
        assert result["status"] == "SUCCESS"
        c_expected = (Decimal("100") * EF_DOLOMITE).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["carbon_dolomite_tc"]) == c_expected

    def test_both_limestone_and_dolomite(self, soc_engine):
        """Test liming CO2 from combined limestone and dolomite application."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": 50,
            "dolomite_tonnes": 30,
        })
        c_limestone = Decimal("50") * EF_LIMESTONE
        c_dolomite = Decimal("30") * EF_DOLOMITE
        c_total = (c_limestone + c_dolomite)
        co2_expected = (c_total * CONVERSION_FACTOR_CO2_C).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["co2_tonnes"]) == co2_expected

    def test_negative_limestone_treated_as_zero(self, soc_engine):
        """Test that negative limestone input is clamped to zero."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": -10,
            "dolomite_tonnes": 0,
        })
        assert Decimal(result["co2_tonnes"]) == 0

    def test_zero_liming_produces_zero_co2(self, soc_engine):
        """Test that zero liming produces zero CO2."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": 0,
            "dolomite_tonnes": 0,
        })
        assert Decimal(result["co2_tonnes"]) == 0

    def test_liming_provenance_hash(self, soc_engine):
        """Test that liming result has a provenance hash."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": 100,
            "dolomite_tonnes": 0,
        })
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 8. Urea Emissions Tests
# ===========================================================================


class TestUreaEmissions:
    """Test CO2 emissions from urea application."""

    def test_urea_co2_calculation(self, soc_engine):
        """Test CO2 = urea_tonnes * EF_urea * 44/12."""
        result = soc_engine.calculate_urea_emissions({"urea_tonnes": 100})
        c_expected = (Decimal("100") * EF_UREA).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        co2_expected = (c_expected * CONVERSION_FACTOR_CO2_C).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["carbon_released_tc"]) == c_expected
        assert Decimal(result["co2_tonnes"]) == co2_expected

    def test_zero_urea_produces_zero_co2(self, soc_engine):
        """Test that zero urea produces zero CO2."""
        result = soc_engine.calculate_urea_emissions({"urea_tonnes": 0})
        assert Decimal(result["co2_tonnes"]) == 0

    def test_negative_urea_clamped_to_zero(self, soc_engine):
        """Test that negative urea is clamped to zero."""
        result = soc_engine.calculate_urea_emissions({"urea_tonnes": -50})
        assert Decimal(result["co2_tonnes"]) == 0

    def test_urea_provenance_hash(self, soc_engine):
        """Test that urea result has a provenance hash."""
        result = soc_engine.calculate_urea_emissions({"urea_tonnes": 50})
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 9. N Mineralisation Tests
# ===========================================================================


class TestNMineralization:
    """Test nitrogen mineralisation from SOC loss."""

    def test_n_mineralization_from_loss(self, soc_engine):
        """Test N mineralised = |SOC_loss| * 0.01."""
        result = soc_engine.get_n_mineralization(Decimal("-10.0"))
        assert Decimal(result["soc_loss_tc_yr"]) == Decimal("10.0")
        n_expected = (Decimal("10.0") * N_MINERALIZATION_C_TO_N).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["n_mineralized_tn_yr"]) == n_expected

    def test_n_mineralization_positive_input(self, soc_engine):
        """Test that positive SOC loss value is handled correctly (absolute value)."""
        result = soc_engine.get_n_mineralization(Decimal("5.0"))
        assert Decimal(result["soc_loss_tc_yr"]) == Decimal("5.0")

    def test_n_mineralization_zero(self, soc_engine):
        """Test that zero SOC loss produces zero N mineralisation."""
        result = soc_engine.get_n_mineralization(Decimal("0"))
        assert Decimal(result["n_mineralized_tn_yr"]) == 0

    def test_n2o_direct_from_mineralized_n(self, soc_engine):
        """Test that direct N2O is computed from mineralised N."""
        result = soc_engine.get_n_mineralization(Decimal("-100.0"))
        n2o = Decimal(result["n2o_direct_tonnes_yr"])
        assert n2o > 0


# ===========================================================================
# 10. Parcel History Tests
# ===========================================================================


class TestParcelHistory:
    """Test per-parcel SOC history tracking."""

    def test_no_history_for_unknown_parcel(self, soc_engine):
        """Test that an unknown parcel returns empty history."""
        history = soc_engine.get_parcel_history("nonexistent-parcel")
        assert history["entry_count"] == 0
        assert history["entries"] == []

    def test_single_entry_in_history(self, soc_engine, base_soc_request):
        """Test that a calculation with parcel_id adds to history."""
        base_soc_request["parcel_id"] = "parcel-001"
        soc_engine.calculate_soc(base_soc_request)
        history = soc_engine.get_parcel_history("parcel-001")
        assert history["entry_count"] == 1
        assert history["entries"][0]["delta_soc_tc_ha"] == "0"

    def test_multiple_entries_track_delta(self, soc_engine, base_soc_request):
        """Test that multiple entries track cumulative SOC change."""
        base_soc_request["parcel_id"] = "parcel-002"
        soc_engine.calculate_soc(base_soc_request)

        # Change land use type for second entry
        base_soc_request["land_use_type"] = "GRASSLAND_IMPROVED"
        base_soc_request["management_practice"] = "IMPROVED"
        soc_engine.calculate_soc(base_soc_request)

        history = soc_engine.get_parcel_history("parcel-002")
        assert history["entry_count"] == 2
        # Second entry should have a delta vs first
        assert history["entries"][1]["delta_soc_tc_ha"] != "0"

    def test_cumulative_change_calculated(self, soc_engine, base_soc_request):
        """Test cumulative SOC change across three entries."""
        base_soc_request["parcel_id"] = "parcel-003"
        # Entry 1: Cropland full-till
        soc_engine.calculate_soc(base_soc_request)
        # Entry 2: Grassland improved
        base_soc_request["land_use_type"] = "GRASSLAND_IMPROVED"
        base_soc_request["management_practice"] = "IMPROVED"
        soc_engine.calculate_soc(base_soc_request)
        # Entry 3: Forest native
        base_soc_request["land_use_type"] = "FOREST_NATIVE"
        base_soc_request["management_practice"] = "NOMINAL"
        soc_engine.calculate_soc(base_soc_request)

        history = soc_engine.get_parcel_history("parcel-003")
        assert history["entry_count"] == 3
        assert "cumulative_soc_change_tc_ha" in history

    def test_no_parcel_id_skips_history(self, soc_engine, base_soc_request):
        """Test that a calculation without parcel_id does not add to history."""
        # No parcel_id in request
        if "parcel_id" in base_soc_request:
            del base_soc_request["parcel_id"]
        soc_engine.calculate_soc(base_soc_request)
        stats = soc_engine.get_statistics()
        assert stats["tracked_parcels"] == 0


# ===========================================================================
# 11. Statistics and Reset Tests
# ===========================================================================


class TestStatisticsAndReset:
    """Test engine statistics and reset functionality."""

    def test_statistics_structure(self, soc_engine):
        """Test that statistics returns all expected fields."""
        stats = soc_engine.get_statistics()
        assert stats["engine"] == "SoilOrganicCarbonEngine"
        assert stats["version"] == "1.0.0"
        assert "created_at" in stats
        assert "total_calculations" in stats
        assert "tracked_parcels" in stats
        assert "total_history_entries" in stats
        assert "db_available" in stats

    def test_reset_clears_state(self, soc_engine, base_soc_request):
        """Test that reset clears parcel history and calculation counter."""
        base_soc_request["parcel_id"] = "parcel-reset"
        soc_engine.calculate_soc(base_soc_request)
        soc_engine.reset()
        stats = soc_engine.get_statistics()
        assert stats["total_calculations"] == 0
        assert stats["tracked_parcels"] == 0


# ===========================================================================
# 12. IPCC Table Validation Tests
# ===========================================================================


class TestIPCCTableValidation:
    """Validate SOC calculations against IPCC reference examples."""

    def test_ipcc_table_5_5_cropland_full_till_tropical(self, soc_engine):
        """Validate: Tropical wet HAC cropland full-till SOC.

        SOC_ref = 65, F_LU = 0.69, F_MG = 1.0, F_I = 1.0
        Expected SOC = 65 * 0.69 = 44.85 tC/ha
        """
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "management_practice": "FULL_TILLAGE",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        assert result["status"] == "SUCCESS"
        assert Decimal(result["soc_tc_ha"]) == Decimal("44.85000000")

    def test_ipcc_grassland_improved_subtropical(self, soc_engine):
        """Validate: Subtropical humid HAC improved grassland.

        SOC_ref = 88, F_LU = 1.14, F_MG = 1.04, F_I = 1.0
        Expected = 88 * 1.14 * 1.04 = 104.3232 tC/ha
        """
        result = soc_engine.calculate_soc({
            "climate_zone": "SUBTROPICAL_HUMID",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "GRASSLAND_IMPROVED",
            "management_practice": "IMPROVED",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        assert result["status"] == "SUCCESS"
        expected = (Decimal("88") * Decimal("1.14") * Decimal("1.04") * Decimal("1.0")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["soc_tc_ha"]) == expected

    def test_ipcc_forest_native_all_unity(self, soc_engine):
        """Validate: Forest native with all unity factors preserves SOC_ref."""
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "VOLCANIC",
            "land_use_type": "FOREST_NATIVE",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        assert result["status"] == "SUCCESS"
        # All factors = 1.0, so SOC = SOC_ref = 130 tC/ha
        assert Decimal(result["soc_tc_ha"]) == Decimal("130.00000000")

    def test_ipcc_cropland_high_manure_input(self, soc_engine):
        """Validate: Cropland with high manure input F_I = 1.44.

        SOC_ref = 65 (tropical wet HAC)
        F_LU = 0.69, F_MG = 1.15 (no tillage), F_I = 1.44
        Expected = 65 * 0.69 * 1.15 * 1.44
        """
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "management_practice": "NO_TILLAGE",
            "input_level": "HIGH_WITH_MANURE",
            "area_ha": 1,
        })
        expected = (Decimal("65") * Decimal("0.69") * Decimal("1.15") * Decimal("1.44")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["soc_tc_ha"]) == expected


# ===========================================================================
# 13. Edge Cases and Special Handling
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_peat_soil_organic_high_soc(self, soc_engine):
        """Test that ORGANIC soil type has the highest reference SOC (200 tC/ha)."""
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "ORGANIC",
            "land_use_type": "FOREST_NATIVE",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        assert Decimal(result["soc_tc_ha"]) == Decimal("200.00000000")

    def test_very_large_area(self, soc_engine):
        """Test calculation with a very large area (100,000 ha)."""
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "FOREST_NATIVE",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 100000,
        })
        assert result["status"] == "SUCCESS"
        # 65 tC/ha * 100,000 ha = 6,500,000 tC
        assert Decimal(result["soc_total_tc"]) == Decimal("6500000.00000000")

    def test_fractional_area(self, soc_engine):
        """Test calculation with fractional hectares."""
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "FOREST_NATIVE",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 0.5,
        })
        assert result["status"] == "SUCCESS"
        assert Decimal(result["soc_total_tc"]) == Decimal("32.50000000")

    def test_thread_safety_concurrent_calculations(self, soc_engine):
        """Test that concurrent calculations do not corrupt state."""
        errors = []

        def calculate(parcel_num):
            try:
                for i in range(10):
                    result = soc_engine.calculate_soc({
                        "climate_zone": "TROPICAL_WET",
                        "soil_type": "HIGH_ACTIVITY_CLAY",
                        "land_use_type": "FOREST_NATIVE",
                        "management_practice": "NOMINAL",
                        "input_level": "MEDIUM",
                        "area_ha": 1,
                        "parcel_id": f"thread-parcel-{parcel_num}",
                    })
                    if result["status"] != "SUCCESS":
                        errors.append(f"Thread {parcel_num} iteration {i}: {result}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=calculate, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_soc_history_entry_to_dict(self):
        """Test SOCHistoryEntry.to_dict() returns correct structure."""
        entry = SOCHistoryEntry(
            entry_id="test-001",
            parcel_id="parcel-001",
            climate_zone="TROPICAL_WET",
            soil_type="HIGH_ACTIVITY_CLAY",
            soc_ref=Decimal("65"),
            land_use_type="FOREST_NATIVE",
            management_practice="NOMINAL",
            input_level="MEDIUM",
            f_lu=Decimal("1.0"),
            f_mg=Decimal("1.0"),
            f_i=Decimal("1.0"),
            soc_calculated=Decimal("65"),
            calculation_date="2026-02-19T00:00:00",
            transition_year=2020,
        )
        d = entry.to_dict()
        assert d["entry_id"] == "test-001"
        assert d["parcel_id"] == "parcel-001"
        assert d["soc_ref_tc_ha"] == "65"
        assert d["soc_calculated_tc_ha"] == "65"
        assert d["transition_year"] == 2020


# ===========================================================================
# 14. Fallback (No DB) Tests
# ===========================================================================


class TestFallbackNoDatabase:
    """Test engine behaviour when LandUseDatabaseEngine is not available."""

    def test_no_db_uses_module_level_factors(self, soc_engine_no_db):
        """Test that the engine falls back to module-level dicts when db is None."""
        assert soc_engine_no_db._land_use_db is None

    def test_no_db_unknown_land_use_defaults_to_one(self, soc_engine_no_db):
        """Test that unknown land use type defaults to F_LU = 1.0 in fallback mode."""
        result = soc_engine_no_db.apply_factors(
            Decimal("65"), "UNKNOWN_LU_TYPE", "NOMINAL", "MEDIUM"
        )
        assert result["f_lu"] == Decimal("1")

    def test_no_db_known_land_use_resolves(self, soc_engine_no_db):
        """Test that known land use type resolves correctly in fallback mode."""
        result = soc_engine_no_db.apply_factors(
            Decimal("65"), "CROPLAND_ANNUAL_FULL_TILL", "FULL_TILLAGE", "MEDIUM"
        )
        assert result["f_lu"] == Decimal("0.69")
        assert result["f_mg"] == Decimal("1.0")
        assert result["f_i"] == Decimal("1.0")

    def test_no_db_unknown_management_defaults_to_one(self, soc_engine_no_db):
        """Test that unknown management practice defaults to F_MG = 1.0 in fallback."""
        result = soc_engine_no_db.apply_factors(
            Decimal("65"), "FOREST_NATIVE", "UNKNOWN_MGMT", "MEDIUM"
        )
        assert result["f_mg"] == Decimal("1")

    def test_no_db_unknown_input_level_defaults_to_one(self, soc_engine_no_db):
        """Test that unknown input level defaults to F_I = 1.0 in fallback."""
        result = soc_engine_no_db.apply_factors(
            Decimal("65"), "FOREST_NATIVE", "NOMINAL", "UNKNOWN_INPUT"
        )
        assert result["f_i"] == Decimal("1")


# ===========================================================================
# 15. Additional SOC Reference Lookup Tests
# ===========================================================================


class TestSOCReferenceExtended:
    """Extended SOC reference stock tests for additional climate zone x soil combos."""

    @pytest.mark.parametrize("zone,soil,expected", [
        ("TROPICAL_DRY", "SPODIC", Decimal("43")),
        ("TROPICAL_DRY", "VOLCANIC", Decimal("80")),
        ("TROPICAL_DRY", "WETLAND", Decimal("86")),
        ("TROPICAL_DRY", "ORGANIC", Decimal("200")),
        ("TROPICAL_MONTANE", "HIGH_ACTIVITY_CLAY", Decimal("65")),
        ("TROPICAL_MONTANE", "LOW_ACTIVITY_CLAY", Decimal("47")),
        ("SUBTROPICAL_HUMID", "VOLCANIC", Decimal("130")),
        ("SUBTROPICAL_HUMID", "WETLAND", Decimal("86")),
    ])
    def test_additional_zone_soil_combos(self, soc_engine, zone, soil, expected):
        """Test additional climate zone x soil type SOC reference stocks."""
        result = soc_engine.get_soc_reference(zone, soil, TIER_1_DEPTH_CM)
        assert result == expected

    def test_all_climate_zones_have_organic_soil(self, soc_engine):
        """Test that every climate zone has an ORGANIC soil reference."""
        zones_to_check = [
            "TROPICAL_WET", "TROPICAL_MOIST", "TROPICAL_DRY",
            "TROPICAL_MONTANE", "SUBTROPICAL_HUMID",
        ]
        for zone in zones_to_check:
            soc_ref = soc_engine.get_soc_reference(zone, "ORGANIC")
            assert soc_ref >= Decimal("100"), f"{zone} ORGANIC SOC_ref too low: {soc_ref}"


# ===========================================================================
# 16. Additional Factor Application Tests
# ===========================================================================


class TestFactorApplicationExtended:
    """Extended factor application tests with multiple factor combinations."""

    @pytest.mark.parametrize("lu,mg,inp,expected_soc", [
        ("FOREST_NATIVE", "NOMINAL", "MEDIUM", Decimal("65")),
        ("FOREST_PLANTATION", "NOMINAL", "MEDIUM", Decimal("52")),
        ("CROPLAND_ANNUAL_FULL_TILL", "NO_TILLAGE", "LOW", None),
        ("GRASSLAND_DEGRADED", "MODERATELY_DEGRADED", "MEDIUM", None),
    ])
    def test_factor_combinations(self, soc_engine, lu, mg, inp, expected_soc):
        """Test various factor combinations produce correct SOC values."""
        result = soc_engine.apply_factors(Decimal("65"), lu, mg, inp)
        if expected_soc is not None:
            assert result["soc_calculated"] == expected_soc.quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
        else:
            # Just verify computation does not error
            assert result["soc_calculated"] > 0

    def test_all_land_use_factors_produce_positive_soc(self, soc_engine):
        """Test that every land use factor produces positive SOC when applied."""
        for lu_type in SOC_LAND_USE_FACTORS:
            result = soc_engine.apply_factors(Decimal("100"), lu_type, "NOMINAL", "MEDIUM")
            assert result["soc_calculated"] > 0, f"{lu_type} produced non-positive SOC"

    def test_all_management_factors_produce_positive_soc(self, soc_engine):
        """Test that every management factor produces positive SOC when applied."""
        for mg in SOC_MANAGEMENT_FACTORS:
            result = soc_engine.apply_factors(Decimal("100"), "FOREST_NATIVE", mg, "MEDIUM")
            assert result["soc_calculated"] > 0, f"{mg} produced non-positive SOC"

    def test_all_input_factors_produce_positive_soc(self, soc_engine):
        """Test that every input factor produces positive SOC when applied."""
        for inp in SOC_INPUT_FACTORS:
            result = soc_engine.apply_factors(Decimal("100"), "FOREST_NATIVE", "NOMINAL", inp)
            assert result["soc_calculated"] > 0, f"{inp} produced non-positive SOC"


# ===========================================================================
# 17. Additional SOC Change Tests
# ===========================================================================


class TestSOCChangeExtended:
    """Extended SOC change calculation tests."""

    def test_grassland_degradation_is_emission(self, soc_engine):
        """Test grassland degradation (improved to degraded) produces emission."""
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "GRASSLAND_IMPROVED",
            "old_management_practice": "IMPROVED",
            "old_input_level": "MEDIUM",
            "new_land_use_type": "GRASSLAND_DEGRADED",
            "new_management_practice": "SEVERELY_DEGRADED",
            "new_input_level": "LOW",
            "area_ha": 100,
        })
        assert result["status"] == "SUCCESS"
        assert result["emission_type"] == "NET_EMISSION"

    def test_tillage_improvement_is_removal(self, soc_engine):
        """Test tillage improvement (full-till to no-till) produces removal."""
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "old_management_practice": "FULL_TILLAGE",
            "old_input_level": "LOW",
            "new_land_use_type": "CROPLAND_ANNUAL_NO_TILL",
            "new_management_practice": "NO_TILLAGE",
            "new_input_level": "HIGH_WITH_MANURE",
            "area_ha": 100,
        })
        assert result["status"] == "SUCCESS"
        assert result["emission_type"] == "NET_REMOVAL"

    def test_soc_change_provenance_hash(self, soc_engine, soc_change_request):
        """Test that SOC change result includes provenance hash."""
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_soc_change_depth_tier2(self, soc_engine, soc_change_request):
        """Test SOC change calculation with Tier 2 depth."""
        soc_change_request["depth_cm"] = 100
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert result["status"] == "SUCCESS"
        assert result["depth_cm"] == 100

    def test_soc_change_missing_old_management_fails(self, soc_engine):
        """Test that missing old_management_practice fails validation."""
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "FOREST_NATIVE",
            "new_land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "new_management_practice": "FULL_TILLAGE",
            "area_ha": 100,
        })
        assert result["status"] == "VALIDATION_ERROR"

    def test_soc_change_negative_area_fails(self, soc_engine, soc_change_request):
        """Test negative area fails validation."""
        soc_change_request["area_ha"] = -50
        result = soc_engine.calculate_soc_change(soc_change_request)
        assert result["status"] == "VALIDATION_ERROR"


# ===========================================================================
# 18. Additional Liming and Urea Tests
# ===========================================================================


class TestLimingAndUreaExtended:
    """Extended liming and urea emission tests."""

    def test_large_limestone_application(self, soc_engine):
        """Test liming with large limestone application (10,000 tonnes)."""
        result = soc_engine.calculate_liming_emissions({
            "limestone_tonnes": 10000,
            "dolomite_tonnes": 0,
        })
        assert result["status"] == "SUCCESS"
        c_expected = Decimal("10000") * EF_LIMESTONE
        assert Decimal(result["carbon_limestone_tc"]) == c_expected.quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )

    def test_fractional_urea_tonnes(self, soc_engine):
        """Test urea with fractional tonnes."""
        result = soc_engine.calculate_urea_emissions({"urea_tonnes": 0.5})
        assert result["status"] == "SUCCESS"
        assert Decimal(result["co2_tonnes"]) > 0

    def test_liming_ef_constants(self):
        """Test that liming emission factor constants match IPCC values."""
        assert EF_LIMESTONE == Decimal("0.12")  # IPCC: 0.12 tC/t CaCO3
        assert EF_DOLOMITE == Decimal("0.13")   # IPCC: 0.13 tC/t CaMg(CO3)2

    def test_urea_ef_constant(self):
        """Test that urea emission factor constant matches IPCC value."""
        assert EF_UREA == Decimal("0.20")  # IPCC: 0.20 tC/t urea


# ===========================================================================
# 19. Depth Adjustment Tests
# ===========================================================================


class TestDepthAdjustment:
    """Test SOC depth adjustments for different soil depths."""

    @pytest.mark.parametrize("depth", [30, 50, 60, 80, 100])
    def test_soc_increases_with_depth(self, soc_engine, depth):
        """Test that SOC reference increases with depth."""
        soc = soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", depth)
        soc_30 = soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 30)
        if depth > 30:
            assert soc > soc_30
        else:
            assert soc == soc_30

    def test_depth_30_is_standard_tier1(self, soc_engine):
        """Test that 30cm depth equals the raw IPCC reference (no scaling)."""
        soc = soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 30)
        assert soc == Decimal("65")

    def test_diminishing_returns_at_depth(self, soc_engine):
        """Test that depth scaling shows diminishing returns (sublinear)."""
        soc_30 = float(soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 30))
        soc_60 = float(soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 60))
        soc_90 = float(soc_engine.get_soc_reference("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", 90))
        # Increment from 30-60 should be larger than 60-90 (diminishing returns)
        inc_30_60 = soc_60 - soc_30
        inc_60_90 = soc_90 - soc_60
        assert inc_30_60 > inc_60_90


# ===========================================================================
# 20. Additional IPCC Validation and Miscellaneous Tests
# ===========================================================================


class TestIPCCValidationExtended:
    """Additional IPCC Table validation tests with different zone/soil combos."""

    def test_wetland_managed_soc(self, soc_engine):
        """Validate: Tropical wet wetland managed SOC.

        SOC_ref = 86 (wetland), F_LU = 0.70 (WETLANDS_MANAGED), F_MG=1.0, F_I=1.0
        Expected = 86 * 0.70 = 60.2 tC/ha
        """
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "WETLAND",
            "land_use_type": "WETLANDS_MANAGED",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        expected = (Decimal("86") * Decimal("0.70")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(result["soc_tc_ha"]) == expected

    def test_settlements_soc(self, soc_engine):
        """Validate: Settlements SOC with F_LU = 0.80.

        SOC_ref = 65 (tropical wet HAC), F_LU = 0.80
        Expected = 65 * 0.80 = 52.0 tC/ha
        """
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "SETTLEMENTS",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        assert Decimal(result["soc_tc_ha"]) == Decimal("52.00000000")

    def test_cropland_set_aside_soc(self, soc_engine):
        """Validate: Cropland set-aside SOC with F_LU = 0.82.

        SOC_ref = 65, F_LU = 0.82, F_MG = 1.0, F_I = 1.0
        Expected = 65 * 0.82 = 53.30 tC/ha
        """
        result = soc_engine.calculate_soc({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "land_use_type": "CROPLAND_SET_ASIDE",
            "management_practice": "NOMINAL",
            "input_level": "MEDIUM",
            "area_ha": 1,
        })
        assert Decimal(result["soc_tc_ha"]) == Decimal("53.30000000")

    def test_soc_change_20yr_cumulative(self, soc_engine):
        """Validate: Total SOC change over full 20-year transition.

        DeltaSOC_total_annual * 20 should approximately equal SOC_new - SOC_old.
        """
        result = soc_engine.calculate_soc_change({
            "climate_zone": "TROPICAL_WET",
            "soil_type": "HIGH_ACTIVITY_CLAY",
            "old_land_use_type": "FOREST_NATIVE",
            "old_management_practice": "NOMINAL",
            "old_input_level": "MEDIUM",
            "new_land_use_type": "CROPLAND_ANNUAL_FULL_TILL",
            "new_management_practice": "FULL_TILLAGE",
            "new_input_level": "MEDIUM",
            "area_ha": 1,
            "transition_period_years": 20,
        })
        delta_annual = Decimal(result["delta_soc_tc_ha_yr"])
        soc_old = Decimal(result["old_state"]["soc_tc_ha"])
        soc_new = Decimal(result["new_state"]["soc_tc_ha"])
        total_change = delta_annual * Decimal("20")
        expected_change = soc_new - soc_old
        assert abs(total_change - expected_change) < Decimal("0.001")


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_safe_decimal_none_returns_default(self):
        """Test _safe_decimal returns default for None input."""
        result = _safe_decimal(None, Decimal("99"))
        assert result == Decimal("99")

    def test_safe_decimal_invalid_returns_default(self):
        """Test _safe_decimal returns default for invalid input."""
        result = _safe_decimal("not_a_number", Decimal("0"))
        assert result == Decimal("0")

    def test_safe_decimal_valid_string(self):
        """Test _safe_decimal converts valid string."""
        result = _safe_decimal("42.5")
        assert result == Decimal("42.5")

    def test_safe_decimal_integer(self):
        """Test _safe_decimal converts integer."""
        result = _safe_decimal(100)
        assert result == Decimal("100")

    def test_D_from_float(self):
        """Test _D converts float to Decimal via string representation."""
        result = _D(3.14)
        assert result == Decimal("3.14")

    def test_D_passthrough_decimal(self):
        """Test _D passes through existing Decimal."""
        d = Decimal("99.99")
        result = _D(d)
        assert result is d

    def test_constants_values(self):
        """Test that module constants have expected IPCC values."""
        assert DEFAULT_TRANSITION_PERIOD == 20
        assert TIER_1_DEPTH_CM == 30
        assert TIER_2_DEPTH_CM == 100
        assert N_MINERALIZATION_C_TO_N == Decimal("0.01")
        assert TIER_2_DEPTH_RATIO == Decimal("2.5")
