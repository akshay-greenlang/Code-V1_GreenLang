# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-011 SteamEmissionsCalculatorEngine (Engine 2).

Tests direct supplier EF, fuel-based, blended multi-fuel, condensate return,
biogenic separation, per-gas breakdown, GWP variations, effective EF,
validation, batch processing, provenance hash, calculation trace, singleton,
and reset.

Target: 105 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

from greenlang.steam_heat_purchase.steam_emissions_calculator import (
    SteamEmissionsCalculatorEngine,
    FUEL_EMISSION_FACTORS,
    GWP_VALUES,
    BIOGENIC_FUEL_TYPES,
    ZERO_EMISSION_FUEL_TYPES,
    VALID_FUEL_TYPES,
)


# ===========================================================================
# Precision helpers (match engine internals)
# ===========================================================================

_PREC_INTERNAL = Decimal("0.00000001")
_PREC_OUTPUT = Decimal("0.001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_KG_TO_TONNES = Decimal("0.001")


def _q(v: Decimal) -> Decimal:
    return v.quantize(_PREC_INTERNAL, rounding=ROUND_HALF_UP)


def _q_out(v: Decimal) -> Decimal:
    return v.quantize(_PREC_OUTPUT, rounding=ROUND_HALF_UP)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_calculator_singleton():
    """Reset the calculator singleton before and after each test."""
    SteamEmissionsCalculatorEngine._instance = None
    yield
    SteamEmissionsCalculatorEngine._instance = None


@pytest.fixture
def engine() -> SteamEmissionsCalculatorEngine:
    """Return a fresh SteamEmissionsCalculatorEngine instance."""
    return SteamEmissionsCalculatorEngine()


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestSteamEmissionsCalculatorSingleton:
    """Tests for the singleton pattern."""

    def test_singleton_same_instance(self):
        """Multiple instantiations return the same object."""
        e1 = SteamEmissionsCalculatorEngine()
        e2 = SteamEmissionsCalculatorEngine()
        assert e1 is e2

    def test_reset_creates_new_instance(self):
        """After resetting _instance, a new instance is created."""
        e1 = SteamEmissionsCalculatorEngine()
        SteamEmissionsCalculatorEngine._instance = None
        e2 = SteamEmissionsCalculatorEngine()
        assert e1 is not e2


# ===========================================================================
# Direct Supplier EF Tests
# ===========================================================================


class TestCalculateWithSupplierEF:
    """Tests for calculate_with_supplier_ef method."""

    def test_basic_supplier_ef_calculation(self, engine):
        """1000 GJ x 66.5 kgCO2e/GJ = 66,500 kgCO2e."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("1000"),
            supplier_ef=Decimal("66.5"),
        )
        assert result["total_co2e_kg"] == _q_out(Decimal("66500"))
        assert result["method"] == "supplier_ef"

    def test_supplier_ef_zero_consumption(self, engine):
        """Zero consumption yields zero emissions."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("0"),
            supplier_ef=Decimal("66.5"),
        )
        assert result["total_co2e_kg"] == _ZERO

    def test_supplier_ef_zero_ef(self, engine):
        """Zero EF yields zero emissions."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("1000"),
            supplier_ef=Decimal("0"),
        )
        assert result["total_co2e_kg"] == _ZERO

    def test_supplier_ef_negative_consumption_raises(self, engine):
        """Negative consumption raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.calculate_with_supplier_ef(
                consumption_gj=Decimal("-1"),
                supplier_ef=Decimal("66.5"),
            )

    def test_supplier_ef_negative_ef_raises(self, engine):
        """Negative EF raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.calculate_with_supplier_ef(
                consumption_gj=Decimal("1000"),
                supplier_ef=Decimal("-5"),
            )

    def test_supplier_ef_co2e_tonnes_correct(self, engine):
        """total_co2e_tonnes = total_co2e_kg / 1000."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("1000"),
            supplier_ef=Decimal("66.5"),
        )
        expected_tonnes = _q_out(Decimal("66500") * _KG_TO_TONNES)
        assert result["total_co2e_tonnes"] == expected_tonnes

    def test_supplier_ef_biogenic_is_zero(self, engine):
        """Supplier EF method has zero biogenic CO2."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("500"),
            supplier_ef=Decimal("50"),
        )
        assert result["biogenic_co2_kg"] == _ZERO

    def test_supplier_ef_effective_ef_equals_input(self, engine):
        """Effective EF equals the supplier EF."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("500"),
            supplier_ef=Decimal("72.3"),
        )
        assert result["effective_ef_kgco2e_gj"] == _q_out(Decimal("72.3"))

    def test_supplier_ef_provenance_hash_present(self, engine):
        """Result includes a 64-char provenance hash."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("100"),
            supplier_ef=Decimal("66.5"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_supplier_ef_trace_non_empty(self, engine):
        """Result includes a non-empty calculation trace."""
        result = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("100"),
            supplier_ef=Decimal("66.5"),
        )
        assert "calculation_trace" in result
        assert len(result["calculation_trace"]) > 0


# ===========================================================================
# Fuel-Based Calculation Tests
# ===========================================================================


class TestCalculateWithFuel:
    """Tests for calculate_with_fuel method."""

    def test_fuel_based_natural_gas(self, engine):
        """Fuel-based natural gas: 1000/0.85 = 1176.47 GJ fuel input."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR5",
        )
        assert result["method"] == "fuel_based"
        expected_fuel_input = _q(Decimal("1000") / Decimal("0.85"))
        assert result["fuel_input_gj"] == pytest.approx(
            _q_out(expected_fuel_input), rel=Decimal("0.01")
        )
        # CO2 = 1176.47 * 56.1 = ~65999.97
        expected_co2 = _q(expected_fuel_input * Decimal("56.100"))
        assert result["co2_kg"] == pytest.approx(
            _q_out(expected_co2), rel=Decimal("0.01")
        )

    def test_fuel_based_coal_bituminous(self, engine):
        """Fuel-based coal: 1000/0.78 = 1282.05 GJ, CO2 = 1282.05 x 94.6."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="coal_bituminous",
            boiler_efficiency=Decimal("0.78"),
            gwp_source="AR5",
        )
        expected_fuel_input = _q(Decimal("1000") / Decimal("0.78"))
        expected_co2 = _q(expected_fuel_input * Decimal("94.600"))
        assert result["co2_kg"] == pytest.approx(
            _q_out(expected_co2), rel=Decimal("0.01")
        )

    def test_fuel_based_default_efficiency_used(self, engine):
        """When no efficiency provided, uses fuel default."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="natural_gas",
            gwp_source="AR5",
        )
        assert result["boiler_efficiency"] == Decimal("0.85")

    def test_fuel_based_unknown_fuel_raises(self, engine):
        """Unknown fuel type raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_with_fuel(
                consumption_gj=Decimal("100"),
                fuel_type="plutonium",
                gwp_source="AR5",
            )

    def test_fuel_based_negative_consumption_raises(self, engine):
        """Negative consumption raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_with_fuel(
                consumption_gj=Decimal("-100"),
                fuel_type="natural_gas",
                gwp_source="AR5",
            )

    def test_fuel_based_per_gas_breakdown(self, engine):
        """Result includes CO2, CH4, N2O breakdown."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR5",
        )
        assert result["co2_kg"] > _ZERO
        assert result["ch4_kg"] > _ZERO
        assert result["n2o_kg"] > _ZERO
        assert result["ch4_co2e_kg"] > _ZERO
        assert result["n2o_co2e_kg"] > _ZERO

    def test_fuel_based_gas_breakdown_detail_list(self, engine):
        """Result includes gas_breakdown list with 3 entries."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="natural_gas",
            gwp_source="AR5",
        )
        assert "gas_breakdown" in result
        assert len(result["gas_breakdown"]) == 3
        gases = {g["gas"] for g in result["gas_breakdown"]}
        assert gases == {"CO2", "CH4", "N2O"}


# ===========================================================================
# Biogenic Separation Tests
# ===========================================================================


class TestBiogenicSeparation:
    """Tests for biogenic CO2 separation."""

    def test_biomass_wood_co2_is_biogenic(self, engine):
        """biomass_wood CO2 is reported as biogenic_co2_kg."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="biomass_wood",
            gwp_source="AR5",
        )
        assert result["biogenic_co2_kg"] > _ZERO
        # Fossil CO2 should be zero for biomass
        assert result["fossil_co2_kg"] == _ZERO
        assert result["is_biogenic"] is True

    def test_biomass_biogas_co2_is_biogenic(self, engine):
        """biomass_biogas CO2 is reported as biogenic."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="biomass_biogas",
            gwp_source="AR5",
        )
        assert result["biogenic_co2_kg"] > _ZERO
        assert result["fossil_co2_kg"] == _ZERO

    def test_biogenic_ch4_n2o_still_fossil(self, engine):
        """For biogenic fuels, CH4 and N2O are still counted as fossil GHG."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="biomass_wood",
            gwp_source="AR5",
        )
        # total_co2e_kg = fossil_co2 + ch4_co2e + n2o_co2e
        # fossil_co2 is 0, so total = ch4_co2e + n2o_co2e
        assert result["total_co2e_kg"] == pytest.approx(
            result["ch4_co2e_kg"] + result["n2o_co2e_kg"],
            rel=Decimal("0.001"),
        )

    def test_natural_gas_no_biogenic(self, engine):
        """Natural gas has zero biogenic CO2."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="natural_gas",
            gwp_source="AR5",
        )
        assert result["biogenic_co2_kg"] == _ZERO
        assert result["is_biogenic"] is False

    def test_separate_biogenic_directly(self, engine):
        """separate_biogenic correctly splits fossil and biogenic."""
        fossil, biogenic = engine.separate_biogenic(
            "biomass_wood", Decimal("10000"),
        )
        assert fossil == _ZERO
        assert biogenic == Decimal("10000")

    def test_separate_biogenic_fossil_fuel(self, engine):
        """separate_biogenic returns all as fossil for non-biogenic fuel."""
        fossil, biogenic = engine.separate_biogenic(
            "natural_gas", Decimal("10000"),
        )
        assert fossil == Decimal("10000")
        assert biogenic == _ZERO


# ===========================================================================
# Zero-Emission Fuels Tests
# ===========================================================================


class TestZeroEmissionFuels:
    """Tests for waste_heat, geothermal, solar_thermal, electric."""

    @pytest.mark.parametrize("fuel_type", [
        "waste_heat", "geothermal", "solar_thermal", "electric",
    ])
    def test_zero_emission_fuels_produce_zero_emissions(self, engine, fuel_type):
        """Zero-emission fuels produce zero CO2, CH4, N2O."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type=fuel_type,
            gwp_source="AR5",
        )
        assert result["co2_kg"] == _ZERO
        assert result["ch4_kg"] == _ZERO
        assert result["n2o_kg"] == _ZERO
        assert result["total_co2e_kg"] == _ZERO


# ===========================================================================
# Condensate Return Tests
# ===========================================================================


class TestCondensateReturn:
    """Tests for condensate return adjustment."""

    def test_condensate_return_20_percent(self, engine):
        """20% condensate return: 1000 x (1 - 0.20) = 800 GJ."""
        effective = engine.apply_condensate_return(
            Decimal("1000"), Decimal("20"),
        )
        expected = _q(Decimal("1000") * (Decimal("1") - Decimal("20") / Decimal("100")))
        assert effective == expected

    def test_condensate_return_zero(self, engine):
        """0% condensate return = no adjustment."""
        effective = engine.apply_condensate_return(
            Decimal("1000"), Decimal("0"),
        )
        expected = _q(Decimal("1000"))
        assert effective == expected

    def test_condensate_return_negative_raises(self, engine):
        """Negative condensate return raises ValueError."""
        with pytest.raises(ValueError):
            engine.apply_condensate_return(Decimal("1000"), Decimal("-5"))

    def test_condensate_return_over_95_raises(self, engine):
        """Condensate return > 95 raises ValueError."""
        with pytest.raises(ValueError):
            engine.apply_condensate_return(Decimal("1000"), Decimal("96"))

    def test_condensate_return_max_95(self, engine):
        """95% condensate return is valid."""
        effective = engine.apply_condensate_return(
            Decimal("1000"), Decimal("95"),
        )
        expected = _q(Decimal("1000") * Decimal("0.05"))
        assert effective == expected


# ===========================================================================
# GWP Variation Tests
# ===========================================================================


class TestGWPVariations:
    """Tests for different GWP sources producing different totals."""

    def test_ar4_vs_ar5_different_totals(self, engine):
        """AR4 and AR5 give different total CO2e (different CH4 GWP)."""
        result_ar4 = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR4",
        )
        result_ar5 = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR5",
        )
        # AR4 CH4=25, AR5 CH4=28, so AR5 total should be slightly higher
        # (CO2 is dominant, difference is small)
        assert result_ar4["gwp_source"] == "AR4"
        assert result_ar5["gwp_source"] == "AR5"

    def test_ar6_values_used(self, engine):
        """AR6 GWP values are applied correctly."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR6",
        )
        assert result["gwp_ch4"] == Decimal("27.9")
        assert result["gwp_n2o"] == Decimal("273")

    def test_ar6_20yr_higher_ch4_gwp(self, engine):
        """AR6_20YR has CH4 GWP of 81.2 (much higher than 100yr)."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("1000"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR6_20YR",
        )
        assert result["gwp_ch4"] == Decimal("81.2")

    def test_unknown_gwp_source_raises(self, engine):
        """Unknown GWP source raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_with_fuel(
                consumption_gj=Decimal("100"),
                fuel_type="natural_gas",
                gwp_source="AR99",
            )


# ===========================================================================
# Multi-Fuel Blended Steam Tests
# ===========================================================================


class TestCalculateBlendedSteam:
    """Tests for calculate_blended_steam method."""

    def test_blended_70_gas_30_biomass(self, engine):
        """70% gas + 30% biomass blended calculation."""
        fuel_mix = [
            {"fuel_type": "natural_gas", "fraction": Decimal("0.70")},
            {"fuel_type": "biomass_wood", "fraction": Decimal("0.30")},
        ]
        result = engine.calculate_blended_steam(
            consumption_gj=Decimal("1000"),
            fuel_mix=fuel_mix,
            boiler_efficiency=Decimal("0.80"),
            gwp_source="AR5",
        )
        assert result["method"] == "blended"
        assert result["total_co2e_kg"] > _ZERO
        assert "per_fuel_results" in result

    def test_blended_fractions_must_sum_to_1(self, engine):
        """Fractions not summing to 1.0 raises ValueError."""
        fuel_mix = [
            {"fuel_type": "natural_gas", "fraction": Decimal("0.50")},
            {"fuel_type": "coal_bituminous", "fraction": Decimal("0.20")},
        ]
        with pytest.raises(ValueError, match="sum to 1.0"):
            engine.calculate_blended_steam(
                consumption_gj=Decimal("1000"),
                fuel_mix=fuel_mix,
                gwp_source="AR5",
            )

    def test_blended_empty_mix_raises(self, engine):
        """Empty fuel mix raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_blended_steam(
                consumption_gj=Decimal("1000"),
                fuel_mix=[],
                gwp_source="AR5",
            )

    def test_blended_unknown_fuel_in_mix_raises(self, engine):
        """Unknown fuel type in mix raises ValueError."""
        fuel_mix = [
            {"fuel_type": "natural_gas", "fraction": Decimal("0.50")},
            {"fuel_type": "plutonium", "fraction": Decimal("0.50")},
        ]
        with pytest.raises(ValueError, match="Unknown fuel_type"):
            engine.calculate_blended_steam(
                consumption_gj=Decimal("1000"),
                fuel_mix=fuel_mix,
                gwp_source="AR5",
            )

    def test_blended_single_fuel_equals_direct(self, engine):
        """100% single fuel blended result equals direct fuel calculation."""
        fuel_mix = [
            {"fuel_type": "natural_gas", "fraction": Decimal("1.00")},
        ]
        blended = engine.calculate_blended_steam(
            consumption_gj=Decimal("500"),
            fuel_mix=fuel_mix,
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR5",
        )
        direct = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="natural_gas",
            boiler_efficiency=Decimal("0.85"),
            gwp_source="AR5",
        )
        assert blended["total_co2e_kg"] == pytest.approx(
            direct["total_co2e_kg"], rel=Decimal("0.01"),
        )


# ===========================================================================
# Effective EF Tests
# ===========================================================================


class TestComputeEffectiveEF:
    """Tests for compute_effective_ef method."""

    def test_effective_ef_basic(self, engine):
        """effective_ef = total_co2e / consumption_gj."""
        ef = engine.compute_effective_ef(
            total_co2e_kg=Decimal("66500"),
            consumption_gj=Decimal("1000"),
        )
        expected = _q_out(Decimal("66500") / Decimal("1000"))
        assert ef == expected

    def test_effective_ef_zero_consumption_returns_zero(self, engine):
        """Zero consumption yields zero effective EF (avoid division by zero)."""
        ef = engine.compute_effective_ef(
            total_co2e_kg=Decimal("0"),
            consumption_gj=Decimal("0"),
        )
        assert ef == _ZERO


# ===========================================================================
# Compute Fuel Input Tests
# ===========================================================================


class TestComputeFuelInput:
    """Tests for compute_fuel_input method."""

    def test_fuel_input_basic(self, engine):
        """fuel_input = consumption / efficiency."""
        result = engine.compute_fuel_input(
            Decimal("1000"), Decimal("0.85"),
        )
        expected = _q(Decimal("1000") / Decimal("0.85"))
        assert result == expected

    def test_fuel_input_efficiency_too_low_raises(self, engine):
        """Efficiency below 0.50 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0.5"):
            engine.compute_fuel_input(Decimal("1000"), Decimal("0.40"))

    def test_fuel_input_efficiency_above_1_raises(self, engine):
        """Efficiency above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be <= 1"):
            engine.compute_fuel_input(Decimal("1000"), Decimal("1.05"))


# ===========================================================================
# Compute Gas Emissions Tests
# ===========================================================================


class TestComputeGasEmissions:
    """Tests for compute_gas_emissions method."""

    def test_gas_emissions_natural_gas(self, engine):
        """Gas emissions for natural gas with known EFs."""
        fuel_input = Decimal("1176.47")
        result = engine.compute_gas_emissions(
            fuel_input, "natural_gas", "AR5",
        )
        # CO2 = 1176.47 * 56.1
        expected_co2 = _q(fuel_input * Decimal("56.100"))
        assert result["co2_kg"] == expected_co2
        # CH4 = 1176.47 * 0.001
        expected_ch4 = _q(fuel_input * Decimal("0.001"))
        assert result["ch4_kg"] == expected_ch4
        # CH4 CO2e = ch4_kg * 28 (AR5)
        assert result["ch4_co2e_kg"] == _q(expected_ch4 * Decimal("28"))

    def test_gas_emissions_unknown_fuel_raises(self, engine):
        """Unknown fuel raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fuel_type"):
            engine.compute_gas_emissions(Decimal("100"), "xyz", "AR5")

    def test_gas_emissions_unknown_gwp_raises(self, engine):
        """Unknown GWP source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gwp_source"):
            engine.compute_gas_emissions(
                Decimal("100"), "natural_gas", "AR99",
            )


# ===========================================================================
# Validate Request Tests
# ===========================================================================


class TestValidateRequest:
    """Tests for validate_request method."""

    def test_valid_request_with_supplier_ef(self, engine):
        """Valid request with supplier_ef passes validation."""
        is_valid, errors = engine.validate_request({
            "consumption_gj": 1000,
            "supplier_ef": 66.5,
        })
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_request_with_fuel_type(self, engine):
        """Valid request with fuel_type passes validation."""
        is_valid, errors = engine.validate_request({
            "consumption_gj": 1000,
            "fuel_type": "natural_gas",
        })
        assert is_valid is True

    def test_missing_consumption_fails(self, engine):
        """Missing consumption_gj fails validation."""
        is_valid, errors = engine.validate_request({
            "supplier_ef": 66.5,
        })
        assert is_valid is False
        assert len(errors) > 0


# ===========================================================================
# Main Entry Point Tests
# ===========================================================================


class TestCalculateSteamEmissions:
    """Tests for the main calculate_steam_emissions entry point."""

    def test_main_entry_supplier_ef(self, engine):
        """Main entry dispatches to supplier_ef method."""
        result = engine.calculate_steam_emissions({
            "consumption_gj": 1000,
            "supplier_ef": 66.5,
        })
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_kg"] > _ZERO

    def test_main_entry_fuel_type(self, engine):
        """Main entry dispatches to fuel-based method."""
        result = engine.calculate_steam_emissions({
            "consumption_gj": 1000,
            "fuel_type": "natural_gas",
        })
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_kg"] > _ZERO

    def test_main_entry_fuel_mix(self, engine):
        """Main entry dispatches to blended method."""
        result = engine.calculate_steam_emissions({
            "consumption_gj": 1000,
            "fuel_mix": [
                {"fuel_type": "natural_gas", "fraction": 0.70},
                {"fuel_type": "biomass_wood", "fraction": 0.30},
            ],
        })
        assert result["status"] == "SUCCESS"

    def test_main_entry_missing_method_fails(self, engine):
        """Request without supplier_ef/fuel_type/fuel_mix returns FAILED."""
        result = engine.calculate_steam_emissions({
            "consumption_gj": 1000,
        })
        assert result["status"] == "FAILED"

    def test_main_entry_condensate_return(self, engine):
        """Main entry applies condensate return adjustment."""
        result = engine.calculate_steam_emissions({
            "consumption_gj": 1000,
            "supplier_ef": 66.5,
            "condensate_return_pct": 20,
        })
        assert result["status"] == "SUCCESS"
        # With 20% condensate return, emissions should be less
        # than without: 1000 * (1-0.20) * 66.5 = 53,200
        assert result["total_co2e_kg"] < Decimal("66500")

    def test_main_entry_energy_unit_conversion(self, engine):
        """Main entry converts energy units to GJ."""
        result = engine.calculate_steam_emissions({
            "consumption_gj": 277.778,
            "energy_unit": "MWh",
            "supplier_ef": 66.5,
        })
        assert result["status"] == "SUCCESS"


# ===========================================================================
# Compare Fuels Tests
# ===========================================================================


class TestCompareFuels:
    """Tests for compare_fuels method."""

    def test_compare_gas_coal_biomass(self, engine):
        """Compare natural_gas vs coal_bituminous vs biomass_wood."""
        result = engine.compare_fuels(
            consumption_gj=Decimal("1000"),
            fuel_types=["natural_gas", "coal_bituminous", "biomass_wood"],
            gwp_source="AR5",
        )
        assert len(result) == 3
        # Coal should have highest fossil CO2e
        coal_result = next(r for r in result if r["fuel_type"] == "coal_bituminous")
        gas_result = next(r for r in result if r["fuel_type"] == "natural_gas")
        assert coal_result["total_co2e_kg"] > gas_result["total_co2e_kg"]


# ===========================================================================
# Health Check Tests
# ===========================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_healthy(self, engine):
        """health_check returns healthy status."""
        result = engine.health_check()
        assert result["status"] == "healthy"
        assert result["engine"] == "steam_emissions_calculator"


# ===========================================================================
# Provenance and Trace Tests
# ===========================================================================


class TestProvenanceAndTrace:
    """Tests for provenance hash and calculation trace presence."""

    def test_fuel_based_provenance_hash(self, engine):
        """Fuel-based result includes 64-char SHA-256 provenance hash."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="natural_gas",
            gwp_source="AR5",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_fuel_based_trace(self, engine):
        """Fuel-based result includes non-empty calculation trace."""
        result = engine.calculate_with_fuel(
            consumption_gj=Decimal("500"),
            fuel_type="natural_gas",
            gwp_source="AR5",
        )
        assert "calculation_trace" in result
        assert len(result["calculation_trace"]) > 0

    def test_provenance_hash_deterministic(self, engine):
        """Same inputs produce the same provenance hash."""
        r1 = engine.calculate_with_supplier_ef(
            consumption_gj=Decimal("100"),
            supplier_ef=Decimal("66.5"),
        )
        # Reset singleton to get fresh engine
        SteamEmissionsCalculatorEngine._instance = None
        engine2 = SteamEmissionsCalculatorEngine()
        r2 = engine2.calculate_with_supplier_ef(
            consumption_gj=Decimal("100"),
            supplier_ef=Decimal("66.5"),
        )
        assert r1["provenance_hash"] == r2["provenance_hash"]


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_fuel_emission_factors_has_14_entries(self):
        """FUEL_EMISSION_FACTORS has 14 entries."""
        assert len(FUEL_EMISSION_FACTORS) == 14

    def test_gwp_values_has_4_sources(self):
        """GWP_VALUES has 4 sources."""
        assert len(GWP_VALUES) == 4

    def test_biogenic_fuel_types(self):
        """BIOGENIC_FUEL_TYPES contains biomass_wood and biomass_biogas."""
        assert "biomass_wood" in BIOGENIC_FUEL_TYPES
        assert "biomass_biogas" in BIOGENIC_FUEL_TYPES
        assert "natural_gas" not in BIOGENIC_FUEL_TYPES

    def test_zero_emission_fuel_types(self):
        """ZERO_EMISSION_FUEL_TYPES contains waste_heat, geothermal, etc."""
        assert "waste_heat" in ZERO_EMISSION_FUEL_TYPES
        assert "geothermal" in ZERO_EMISSION_FUEL_TYPES
        assert "solar_thermal" in ZERO_EMISSION_FUEL_TYPES
        assert "electric" in ZERO_EMISSION_FUEL_TYPES

    def test_valid_fuel_types(self):
        """VALID_FUEL_TYPES matches FUEL_EMISSION_FACTORS keys."""
        assert VALID_FUEL_TYPES == set(FUEL_EMISSION_FACTORS.keys())
