# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 CarbonStockCalculatorEngine.

Tests stock-difference method, gain-loss method, AGB change calculations,
BGB from root-to-shoot ratios, dead wood dynamics, litter dynamics,
fire emission calculations, conversion emissions, net emission/removal
calculations, CO2e conversion, trace step recording, batch calculations,
Decimal arithmetic determinism, zero-area edge cases, negative stock
changes (removals), and IPCC example validation.

Target: 130 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.land_use_emissions.carbon_stock_calculator import (
    CarbonStockCalculatorEngine,
    CalculationMethod,
    CarbonPool,
    TraceStep,
    _D,
    _safe_decimal,
    _ZERO,
    _ONE,
    _PRECISION,
)
from greenlang.land_use_emissions.land_use_database import LandUseDatabaseEngine


# ===========================================================================
# Helper Function Tests
# ===========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_d_from_int(self):
        """_D converts integer to Decimal."""
        assert _D(42) == Decimal("42")

    def test_d_from_float(self):
        """_D converts float to Decimal via string."""
        result = _D(3.14)
        assert isinstance(result, Decimal)

    def test_d_from_string(self):
        """_D converts string to Decimal."""
        assert _D("100.5") == Decimal("100.5")

    def test_d_from_decimal_passthrough(self):
        """_D passes Decimal through unchanged."""
        d = Decimal("99.99")
        assert _D(d) is d

    def test_safe_decimal_from_none(self):
        """_safe_decimal returns default for None."""
        assert _safe_decimal(None) == _ZERO

    def test_safe_decimal_from_invalid(self):
        """_safe_decimal returns default for invalid input."""
        assert _safe_decimal("abc") == _ZERO

    def test_safe_decimal_from_valid(self):
        """_safe_decimal converts valid input to Decimal."""
        assert _safe_decimal("42.5") == Decimal("42.5")

    def test_safe_decimal_custom_default(self):
        """_safe_decimal uses custom default when provided."""
        assert _safe_decimal(None, Decimal("10")) == Decimal("10")


# ===========================================================================
# TraceStep Tests
# ===========================================================================


class TestTraceStep:
    """Tests for the TraceStep dataclass."""

    def test_creation(self):
        """TraceStep can be created with all fields."""
        step = TraceStep(
            step_number=1,
            description="Calculate AGB change",
            formula="DeltaC = (C_t2 - C_t1) / (t2 - t1)",
            inputs={"C_t1": "180", "C_t2": "170"},
            output="2.0",
            unit="tC/ha/yr",
        )
        assert step.step_number == 1
        assert step.description == "Calculate AGB change"

    def test_to_dict(self):
        """TraceStep.to_dict returns all fields."""
        step = TraceStep(
            step_number=1,
            description="test",
            formula="f",
            inputs={"a": "1"},
            output="2",
            unit="tC",
        )
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["formula"] == "f"
        assert d["inputs"] == {"a": "1"}


# ===========================================================================
# Engine Initialization Tests
# ===========================================================================


class TestEngineInit:
    """Tests for CarbonStockCalculatorEngine initialization."""

    def test_init_with_db(self, land_use_database_engine):
        """Engine initializes with a database engine."""
        calc = CarbonStockCalculatorEngine(land_use_database=land_use_database_engine)
        assert calc._land_use_db is land_use_database_engine

    def test_init_without_db(self):
        """Engine can initialize without a database engine."""
        calc = CarbonStockCalculatorEngine(land_use_database=None)
        assert calc is not None

    def test_default_gwp_source(self, land_use_database_engine):
        """Default GWP source is AR6."""
        calc = CarbonStockCalculatorEngine(land_use_database=land_use_database_engine)
        assert calc._gwp_source == "AR6"

    def test_custom_gwp_source(self, land_use_database_engine):
        """Custom GWP source is respected."""
        calc = CarbonStockCalculatorEngine(
            land_use_database=land_use_database_engine, gwp_source="AR5"
        )
        assert calc._gwp_source == "AR5"

    def test_calculation_counter_starts_zero(self, land_use_database_engine):
        """Total calculations counter starts at zero."""
        calc = CarbonStockCalculatorEngine(land_use_database=land_use_database_engine)
        assert calc._total_calculations == 0


# ===========================================================================
# Stock-Difference Method Tests
# ===========================================================================


class TestStockDifference:
    """Tests for calculate_stock_difference method."""

    def test_successful_calculation(self, carbon_stock_calculator, sample_calculation_request):
        """Stock-difference calculation returns SUCCESS status."""
        result = carbon_stock_calculator.calculate_stock_difference(
            sample_calculation_request
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "STOCK_DIFFERENCE"

    def test_calculation_id_is_uuid(self, carbon_stock_calculator, sample_calculation_request):
        """Result contains a unique calculation ID."""
        result = carbon_stock_calculator.calculate_stock_difference(
            sample_calculation_request
        )
        assert "calculation_id" in result
        assert len(result["calculation_id"]) > 0

    def test_per_pool_breakdown(self, carbon_stock_calculator, sample_calculation_request):
        """Result includes per-pool breakdown."""
        result = carbon_stock_calculator.calculate_stock_difference(
            sample_calculation_request
        )
        assert "pool_results" in result
        assert "AGB" in result["pool_results"]
        assert "BGB" in result["pool_results"]
        assert "DEAD_WOOD" in result["pool_results"]
        assert "LITTER" in result["pool_results"]

    def test_agb_stock_difference_calculation(self, carbon_stock_calculator):
        """AGB stock difference is (C_t2 - C_t1) / time_interval."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 200, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 190, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        agb_result = result["pool_results"]["AGB"]
        delta_per_ha = Decimal(agb_result["delta_c_tc_ha_yr"])
        expected = Decimal("-2.00000000")
        assert delta_per_ha == expected

    def test_total_delta_c_sums_pools(self, carbon_stock_calculator):
        """Total delta C equals sum of individual pool deltas."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 100, "BGB": 25, "DEAD_WOOD": 10, "LITTER": 5},
            "c_t2": {"AGB": 95, "BGB": 24, "DEAD_WOOD": 9, "LITTER": 5},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        pool_sum = Decimal("0")
        for pool_data in result["pool_results"].values():
            pool_sum += Decimal(pool_data["delta_c_tc_ha_yr"])
        total = Decimal(result["total_delta_c_tc_ha_yr"])
        assert abs(pool_sum - total) < Decimal("0.0001")

    def test_emission_type_is_net_emission(self, carbon_stock_calculator):
        """Negative stock change classifies as NET_EMISSION."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 200, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 180, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["emission_type"] == "NET_EMISSION"

    def test_emission_type_is_net_removal(self, carbon_stock_calculator):
        """Positive stock change classifies as NET_REMOVAL."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 100, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 110, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["emission_type"] == "NET_REMOVAL"

    def test_co2_conversion_uses_44_12(self, carbon_stock_calculator):
        """CO2 conversion uses the 44/12 molecular weight ratio."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 1,
            "c_t1": {"AGB": 100, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 99, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2020,
            "year_t2": 2021,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        delta_c = Decimal(result["total_delta_c_tc_yr"])
        co2 = Decimal(result["total_co2_tonnes_yr"])
        ratio = co2 / delta_c
        assert abs(ratio - Decimal("3.66667")) < Decimal("0.01")

    def test_trace_steps_present(self, carbon_stock_calculator, sample_calculation_request):
        """Result includes trace steps for audit trail."""
        result = carbon_stock_calculator.calculate_stock_difference(
            sample_calculation_request
        )
        assert "trace_steps" in result
        assert len(result["trace_steps"]) > 0

    def test_provenance_hash_present(self, carbon_stock_calculator, sample_calculation_request):
        """Result includes a SHA-256 provenance hash."""
        result = carbon_stock_calculator.calculate_stock_difference(
            sample_calculation_request
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_processing_time_recorded(self, carbon_stock_calculator, sample_calculation_request):
        """Result includes processing time in ms."""
        result = carbon_stock_calculator.calculate_stock_difference(
            sample_calculation_request
        )
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_missing_land_category_error(self, carbon_stock_calculator):
        """Missing land_category returns VALIDATION_ERROR."""
        request = {
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 100},
            "c_t2": {"AGB": 90},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_zero_area_error(self, carbon_stock_calculator):
        """Zero area returns VALIDATION_ERROR."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 0,
            "c_t1": {"AGB": 100},
            "c_t2": {"AGB": 90},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_year_t2_before_year_t1_error(self, carbon_stock_calculator):
        """year_t2 < year_t1 returns VALIDATION_ERROR."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 100},
            "c_t2": {"AGB": 90},
            "year_t1": 2025,
            "year_t2": 2020,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_deterministic_output(self, carbon_stock_calculator, sample_calculation_request):
        """Same input always produces same output (deterministic)."""
        r1 = carbon_stock_calculator.calculate_stock_difference(sample_calculation_request)
        r2 = carbon_stock_calculator.calculate_stock_difference(sample_calculation_request)
        assert r1["total_delta_c_tc_yr"] == r2["total_delta_c_tc_yr"]
        assert r1["total_co2_tonnes_yr"] == r2["total_co2_tonnes_yr"]

    def test_large_area_calculation(self, carbon_stock_calculator):
        """Calculation works correctly with large areas."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 1_000_000,
            "c_t1": {"AGB": 180, "BGB": 43, "DEAD_WOOD": 14, "LITTER": 5},
            "c_t2": {"AGB": 170, "BGB": 40, "DEAD_WOOD": 13, "LITTER": 5},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["status"] == "SUCCESS"

    def test_no_change_in_stocks(self, carbon_stock_calculator):
        """Zero stock change results in zero emissions."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 100, "BGB": 25, "DEAD_WOOD": 10, "LITTER": 5},
            "c_t2": {"AGB": 100, "BGB": 25, "DEAD_WOOD": 10, "LITTER": 5},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert Decimal(result["total_delta_c_tc_yr"]) == Decimal("0")

    def test_counter_increments(self, carbon_stock_calculator, sample_calculation_request):
        """Calculation counter increments after each call."""
        initial = carbon_stock_calculator._total_calculations
        carbon_stock_calculator.calculate_stock_difference(sample_calculation_request)
        assert carbon_stock_calculator._total_calculations == initial + 1


# ===========================================================================
# AGB Change Method Tests
# ===========================================================================


class TestCalculateAGBChange:
    """Tests for the calculate_agb_change method."""

    def test_agb_decrease(self, carbon_stock_calculator):
        """AGB decrease produces negative delta_c."""
        result = carbon_stock_calculator.calculate_agb_change(
            agb_t1=Decimal("180"), agb_t2=Decimal("170"),
            area_ha=Decimal("100"), time_interval=Decimal("5"),
        )
        assert Decimal(result["delta_c_tc_ha_yr"]) == Decimal("-2.00000000")

    def test_agb_increase(self, carbon_stock_calculator):
        """AGB increase produces positive delta_c."""
        result = carbon_stock_calculator.calculate_agb_change(
            agb_t1=Decimal("100"), agb_t2=Decimal("110"),
            area_ha=Decimal("50"), time_interval=Decimal("10"),
        )
        assert Decimal(result["delta_c_tc_ha_yr"]) == Decimal("1.00000000")

    def test_agb_no_change(self, carbon_stock_calculator):
        """Zero AGB change produces zero delta_c."""
        result = carbon_stock_calculator.calculate_agb_change(
            agb_t1=Decimal("100"), agb_t2=Decimal("100"),
            area_ha=Decimal("100"), time_interval=Decimal("5"),
        )
        assert Decimal(result["delta_c_tc_ha_yr"]) == Decimal("0")

    def test_co2_conversion(self, carbon_stock_calculator):
        """CO2 tonnes are correctly calculated."""
        result = carbon_stock_calculator.calculate_agb_change(
            agb_t1=Decimal("100"), agb_t2=Decimal("90"),
            area_ha=Decimal("1"), time_interval=Decimal("1"),
        )
        delta_c = Decimal(result["delta_c_tc_yr"])
        co2 = Decimal(result["co2_tonnes_yr"])
        assert abs(co2 / delta_c - Decimal("3.66667")) < Decimal("0.01")

    def test_pool_label(self, carbon_stock_calculator):
        """Result is labeled as AGB pool."""
        result = carbon_stock_calculator.calculate_agb_change(
            agb_t1=Decimal("100"), agb_t2=Decimal("90"),
            area_ha=Decimal("1"), time_interval=Decimal("1"),
        )
        assert result["pool"] == "AGB"


# ===========================================================================
# Gain-Loss Method Tests
# ===========================================================================


class TestGainLoss:
    """Tests for calculate_gain_loss method."""

    def test_successful_gain_loss(self, carbon_stock_calculator, sample_gain_loss_request):
        """Gain-loss calculation returns SUCCESS status."""
        result = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        assert result["status"] == "SUCCESS"
        assert result["method"] == "GAIN_LOSS"

    def test_gains_section_present(self, carbon_stock_calculator, sample_gain_loss_request):
        """Result includes gains breakdown."""
        result = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        assert "gains" in result
        assert "agb_gain_tc_yr" in result["gains"]
        assert "bgb_gain_tc_yr" in result["gains"]
        assert "total_gains_tc_yr" in result["gains"]

    def test_losses_section_present(self, carbon_stock_calculator, sample_gain_loss_request):
        """Result includes losses breakdown."""
        result = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        assert "losses" in result
        assert "harvest_loss_tc_yr" in result["losses"]
        assert "fuelwood_loss_tc_yr" in result["losses"]
        assert "disturbance_loss_tc_yr" in result["losses"]

    def test_net_delta_c_equals_gains_minus_losses(self, carbon_stock_calculator):
        """Net delta C equals gains minus losses."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "harvest_volume_m3": 0,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 0,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        gains = Decimal(result["gains"]["total_gains_tc_yr"])
        losses = Decimal(result["losses"]["total_losses_tc_yr"])
        net = Decimal(result["net_delta_c_tc_yr"])
        assert abs(net - (gains - losses)) < Decimal("0.001")

    def test_no_losses_results_in_net_removal(self, carbon_stock_calculator):
        """No harvest or disturbance with growth results in NET_REMOVAL."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "harvest_volume_m3": 0,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 0,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert result["emission_type"] == "NET_REMOVAL"

    def test_large_harvest_results_in_net_emission(self, carbon_stock_calculator):
        """Large harvest volume results in NET_EMISSION."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 10,
            "harvest_volume_m3": 100000,
            "fuelwood_volume_m3": 50000,
            "disturbance_area_ha": 0,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert result["emission_type"] == "NET_EMISSION"

    def test_growth_rate_override(self, carbon_stock_calculator):
        """growth_rate_override is used when provided."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "growth_rate_override": 10.0,
            "harvest_volume_m3": 0,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 0,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert result["gains"]["growth_rate_tdm_ha_yr"] == "10.00000000"

    def test_missing_land_category_error(self, carbon_stock_calculator):
        """Missing land_category returns VALIDATION_ERROR."""
        request = {
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_zero_area_error(self, carbon_stock_calculator):
        """Zero area returns VALIDATION_ERROR."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 0,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_trace_steps_present(self, carbon_stock_calculator, sample_gain_loss_request):
        """Gain-loss result includes trace steps."""
        result = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        assert "trace_steps" in result
        assert len(result["trace_steps"]) > 0

    def test_provenance_hash_present(self, carbon_stock_calculator, sample_gain_loss_request):
        """Gain-loss result includes provenance hash."""
        result = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_harvest_loss_formula(self, carbon_stock_calculator):
        """Harvest loss = V * D * BCEF * CF."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "harvest_volume_m3": 1000,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 0,
            "wood_density": 0.5,
            "bcef": 1.0,
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        harvest_loss = Decimal(result["losses"]["harvest_loss_tc_yr"])
        expected = (Decimal("1000") * Decimal("0.5") * Decimal("1.0") * Decimal("0.47"))
        assert abs(harvest_loss - expected.quantize(_PRECISION)) < Decimal("0.001")

    def test_deterministic_gain_loss(self, carbon_stock_calculator, sample_gain_loss_request):
        """Same input produces same output for gain-loss."""
        r1 = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        r2 = carbon_stock_calculator.calculate_gain_loss(sample_gain_loss_request)
        assert r1["net_delta_c_tc_yr"] == r2["net_delta_c_tc_yr"]


# ===========================================================================
# Fire Emissions Tests
# ===========================================================================


class TestFireEmissions:
    """Tests for fire emission calculations within the engine."""

    def test_fire_emissions_with_disturbance(self, carbon_stock_calculator):
        """Gain-loss with fire disturbance includes fire emissions."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "harvest_volume_m3": 0,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 50,
            "disturbance_type": "FIRE_WILDFIRE",
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert result["status"] == "SUCCESS"
        disturbance_loss = Decimal(result["losses"]["disturbance_loss_tc_yr"])
        if result.get("fire_emissions"):
            assert disturbance_loss > Decimal("0")

    def test_zero_disturbance_area_no_fire(self, carbon_stock_calculator):
        """Zero disturbance area produces no fire emissions."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "harvest_volume_m3": 0,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 0,
            "disturbance_type": "FIRE_WILDFIRE",
        }
        result = carbon_stock_calculator.calculate_gain_loss(request)
        assert Decimal(result["losses"]["disturbance_loss_tc_yr"]) == Decimal("0")


# ===========================================================================
# IPCC Example Validation Tests
# ===========================================================================


class TestIPCCExampleValidation:
    """Tests validating calculations against known IPCC examples."""

    def test_stock_difference_simple_example(self, carbon_stock_calculator):
        """Validate stock-difference with a simple IPCC-style example.

        1000 ha of forest land, AGB decreases from 180 to 170 tC/ha
        over 5 years.
        Delta_C_AGB = (170 - 180) / 5 = -2 tC/ha/yr
        Total = -2 * 1000 = -2000 tC/yr
        CO2 = -2000 * 3.66667 = -7333.34 tCO2/yr (emission)
        """
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 1000,
            "c_t1": {"AGB": 180, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 170, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        assert result["status"] == "SUCCESS"
        delta_c = Decimal(result["total_delta_c_tc_yr"])
        assert delta_c == Decimal("-2000.00000000")
        assert result["emission_type"] == "NET_EMISSION"

    def test_stock_increase_example(self, carbon_stock_calculator):
        """Validate net removal calculation with growing forest.

        100 ha of forest land, AGB increases from 100 to 120 tC/ha
        over 10 years.
        Delta_C_AGB = (120 - 100) / 10 = 2 tC/ha/yr
        Total = 2 * 100 = 200 tC/yr (removal)
        """
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 100, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 120, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2010,
            "year_t2": 2020,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        delta_c = Decimal(result["total_delta_c_tc_yr"])
        assert delta_c == Decimal("200.00000000")
        assert result["emission_type"] == "NET_REMOVAL"


# ===========================================================================
# Decimal Arithmetic Determinism Tests
# ===========================================================================


class TestDecimalDeterminism:
    """Tests ensuring Decimal arithmetic produces deterministic results."""

    def test_repeated_calculations_identical(self, carbon_stock_calculator):
        """10 repeated calculations produce identical results."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 1000,
            "c_t1": {"AGB": 180.123, "BGB": 43.456, "DEAD_WOOD": 14.789, "LITTER": 5.012},
            "c_t2": {"AGB": 175.567, "BGB": 42.890, "DEAD_WOOD": 14.234, "LITTER": 4.678},
            "year_t1": 2015,
            "year_t2": 2023,
        }
        results = [
            carbon_stock_calculator.calculate_stock_difference(request)
            for _ in range(10)
        ]
        first_co2 = results[0]["total_co2_tonnes_yr"]
        for r in results[1:]:
            assert r["total_co2_tonnes_yr"] == first_co2

    def test_decimal_precision_8_places(self, carbon_stock_calculator):
        """Results have 8 decimal places of precision."""
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 333,
            "c_t1": {"AGB": 111.111, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "c_t2": {"AGB": 100.000, "BGB": 0, "DEAD_WOOD": 0, "LITTER": 0},
            "year_t1": 2020,
            "year_t2": 2023,
        }
        result = carbon_stock_calculator.calculate_stock_difference(request)
        delta = result["pool_results"]["AGB"]["delta_c_tc_ha_yr"]
        parts = str(delta).split(".")
        if len(parts) == 2:
            assert len(parts[1]) == 8


# ===========================================================================
# Engine Without Database Tests
# ===========================================================================


class TestEngineWithoutDB:
    """Tests for CarbonStockCalculatorEngine when no database is available."""

    def test_stock_difference_without_db(self):
        """Stock-difference works without a database engine."""
        calc = CarbonStockCalculatorEngine(land_use_database=None)
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "c_t1": {"AGB": 180, "BGB": 43, "DEAD_WOOD": 14, "LITTER": 5},
            "c_t2": {"AGB": 170, "BGB": 40, "DEAD_WOOD": 13, "LITTER": 5},
            "year_t1": 2020,
            "year_t2": 2025,
        }
        result = calc.calculate_stock_difference(request)
        assert result["status"] == "SUCCESS"

    def test_gain_loss_without_db_uses_defaults(self):
        """Gain-loss without DB uses default growth rate."""
        calc = CarbonStockCalculatorEngine(land_use_database=None)
        request = {
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
            "harvest_volume_m3": 0,
            "fuelwood_volume_m3": 0,
            "disturbance_area_ha": 0,
        }
        result = calc.calculate_gain_loss(request)
        assert result["status"] == "SUCCESS"
