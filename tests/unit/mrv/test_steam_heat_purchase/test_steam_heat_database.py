# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-011 SteamHeatDatabaseEngine (Engine 1).

Tests all 14 fuel types, 13 district heating regions, 9 cooling technologies,
5 CHP fuel types, 4 GWP sources, unit conversions, blended EF calculation,
search, custom factors, health check, singleton, and reset.

Target: 80 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.mrv.steam_heat_purchase.steam_heat_database import (
    SteamHeatDatabaseEngine,
    FUEL_EMISSION_FACTORS,
    DISTRICT_HEATING_FACTORS,
    COOLING_SYSTEM_FACTORS,
    COOLING_ENERGY_SOURCE,
    CHP_DEFAULT_EFFICIENCIES,
    GWP_VALUES,
    UNIT_CONVERSIONS,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_database_singleton():
    """Reset the database engine singleton before and after each test."""
    SteamHeatDatabaseEngine.reset_singleton()
    yield
    SteamHeatDatabaseEngine.reset_singleton()


@pytest.fixture
def db() -> SteamHeatDatabaseEngine:
    """Return a fresh SteamHeatDatabaseEngine instance."""
    return SteamHeatDatabaseEngine()


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestSteamHeatDatabaseSingleton:
    """Tests for the singleton pattern."""

    def test_singleton_same_instance(self):
        """Multiple instantiations return the same object."""
        db1 = SteamHeatDatabaseEngine()
        db2 = SteamHeatDatabaseEngine()
        assert db1 is db2

    def test_reset_singleton_creates_new_instance(self):
        """After reset_singleton, a new instance is created."""
        db1 = SteamHeatDatabaseEngine()
        SteamHeatDatabaseEngine.reset_singleton()
        db2 = SteamHeatDatabaseEngine()
        assert db1 is not db2

    def test_engine_id_and_version(self, db):
        """ENGINE_ID and ENGINE_VERSION are set correctly."""
        assert db.ENGINE_ID == "steam_heat_database"
        assert db.ENGINE_VERSION == "1.0.0"


# ===========================================================================
# Fuel Emission Factors Tests
# ===========================================================================


ALL_14_FUEL_TYPES = [
    "natural_gas", "fuel_oil_2", "fuel_oil_6",
    "coal_bituminous", "coal_subbituminous", "coal_lignite",
    "lpg", "biomass_wood", "biomass_biogas",
    "municipal_waste", "waste_heat", "geothermal",
    "solar_thermal", "electric",
]


class TestFuelEmissionFactors:
    """Tests for fuel EF lookups."""

    @pytest.mark.parametrize("fuel_type", ALL_14_FUEL_TYPES)
    def test_get_fuel_ef_all_14_types(self, db, fuel_type):
        """get_fuel_ef returns valid data for each of the 14 fuel types."""
        result = db.get_fuel_ef(fuel_type)
        assert result["fuel_type"] == fuel_type
        assert "co2_ef" in result
        assert "ch4_ef" in result
        assert "n2o_ef" in result
        assert "default_efficiency" in result
        assert "provenance_hash" in result
        assert isinstance(result["co2_ef"], Decimal)

    def test_get_fuel_ef_returns_correct_structure(self, db):
        """get_fuel_ef for natural_gas returns expected structure."""
        result = db.get_fuel_ef("natural_gas")
        assert result["co2_ef"] == Decimal("56.100")
        assert result["ch4_ef"] == Decimal("0.001")
        assert result["n2o_ef"] == Decimal("0.0001")
        assert result["default_efficiency"] == Decimal("0.85")
        assert result["source"] == "builtin"

    def test_get_fuel_ef_case_insensitive(self, db):
        """get_fuel_ef is case-insensitive."""
        result = db.get_fuel_ef("Natural_Gas")
        assert result["fuel_type"] == "natural_gas"

    def test_get_fuel_ef_unknown_raises(self, db):
        """get_fuel_ef raises ValueError for unknown fuel type."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.get_fuel_ef("unknown_fuel")

    def test_is_biogenic_true_for_biomass_wood(self, db):
        """is_biogenic_fuel returns True for biomass_wood."""
        assert db.is_biogenic_fuel("biomass_wood") is True

    def test_is_biogenic_true_for_biomass_biogas(self, db):
        """is_biogenic_fuel returns True for biomass_biogas."""
        assert db.is_biogenic_fuel("biomass_biogas") is True

    def test_is_biogenic_false_for_natural_gas(self, db):
        """is_biogenic_fuel returns False for natural_gas."""
        assert db.is_biogenic_fuel("natural_gas") is False

    def test_is_biogenic_false_for_coal(self, db):
        """is_biogenic_fuel returns False for coal_bituminous."""
        assert db.is_biogenic_fuel("coal_bituminous") is False

    def test_is_biogenic_unknown_raises(self, db):
        """is_biogenic_fuel raises ValueError for unknown fuel."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.is_biogenic_fuel("fake_fuel")

    @pytest.mark.parametrize("fuel_type", ALL_14_FUEL_TYPES)
    def test_get_default_efficiency_all_types(self, db, fuel_type):
        """get_default_efficiency returns Decimal for all fuel types."""
        eff = db.get_default_efficiency(fuel_type)
        assert isinstance(eff, Decimal)
        assert Decimal("0") < eff <= Decimal("1")

    def test_get_default_efficiency_natural_gas(self, db):
        """get_default_efficiency for natural_gas returns 0.85."""
        assert db.get_default_efficiency("natural_gas") == Decimal("0.85")

    def test_get_default_efficiency_unknown_raises(self, db):
        """get_default_efficiency raises ValueError for unknown fuel."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.get_default_efficiency("unknown_fuel")

    def test_get_all_fuel_efs_returns_14_entries(self, db):
        """get_all_fuel_efs returns all 14 fuel types."""
        all_efs = db.get_all_fuel_efs()
        assert len(all_efs) >= 14
        for fuel_type in ALL_14_FUEL_TYPES:
            assert fuel_type in all_efs

    def test_get_fuel_types_returns_sorted_list(self, db):
        """get_fuel_types returns sorted list of fuel type IDs."""
        types = db.get_fuel_types()
        assert isinstance(types, list)
        assert types == sorted(types)
        assert len(types) >= 14


# ===========================================================================
# District Heating Factors Tests
# ===========================================================================


ALL_13_DH_REGIONS = [
    "denmark", "sweden", "finland", "germany", "poland",
    "netherlands", "france", "uk", "us", "china",
    "japan", "south_korea", "global_default",
]


class TestDistrictHeatingFactors:
    """Tests for district heating factor lookups."""

    @pytest.mark.parametrize("region", ALL_13_DH_REGIONS)
    def test_get_dh_factor_all_13_regions(self, db, region):
        """get_dh_factor returns valid data for each of the 13 regions."""
        result = db.get_dh_factor(region)
        assert result["region"] == region
        assert "ef_kgco2e_per_gj" in result
        assert "distribution_loss_pct" in result
        assert "provenance_hash" in result
        assert isinstance(result["ef_kgco2e_per_gj"], Decimal)

    def test_get_dh_factor_returns_ef_and_loss(self, db):
        """get_dh_factor for germany returns correct EF and loss."""
        result = db.get_dh_factor("germany")
        assert result["ef_kgco2e_per_gj"] == Decimal("72.0")
        assert result["distribution_loss_pct"] == Decimal("0.12")

    def test_get_dh_factor_global_default_fallback(self, db):
        """get_dh_factor falls back to global_default for unknown regions."""
        result = db.get_dh_factor("mars")
        assert result["ef_kgco2e_per_gj"] == Decimal("70.0")
        assert result["source"] == "global_default_fallback"

    def test_get_dh_factor_case_insensitive(self, db):
        """get_dh_factor is case-insensitive."""
        result = db.get_dh_factor("Denmark")
        assert result["region"] == "denmark"

    def test_get_all_dh_factors_returns_13(self, db):
        """get_all_dh_factors returns all 13 regions."""
        all_factors = db.get_all_dh_factors()
        assert len(all_factors) >= 13

    def test_get_dh_regions_sorted(self, db):
        """get_dh_regions returns sorted list."""
        regions = db.get_dh_regions()
        assert isinstance(regions, list)
        assert regions == sorted(regions)
        assert len(regions) >= 13

    def test_get_distribution_loss_pct_known_region(self, db):
        """get_distribution_loss_pct returns correct value for known region."""
        loss = db.get_distribution_loss_pct("denmark")
        assert loss == Decimal("0.10")

    def test_get_distribution_loss_pct_unknown_region_uses_fallback(self, db):
        """get_distribution_loss_pct uses global_default for unknown."""
        loss = db.get_distribution_loss_pct("antarctica")
        assert loss == Decimal("0.12")


# ===========================================================================
# Cooling System Factors Tests
# ===========================================================================


ALL_9_COOLING_TECHS = [
    "centrifugal_chiller", "screw_chiller", "reciprocating_chiller",
    "absorption_single", "absorption_double", "absorption_triple",
    "free_cooling", "ice_storage", "thermal_storage",
]


class TestCoolingSystemFactors:
    """Tests for cooling system factor lookups."""

    @pytest.mark.parametrize("technology", ALL_9_COOLING_TECHS)
    def test_get_cooling_factor_all_9_technologies(self, db, technology):
        """get_cooling_factor returns valid data for each of 9 techs."""
        result = db.get_cooling_factor(technology)
        assert result["technology"] == technology
        assert "cop_min" in result
        assert "cop_max" in result
        assert "cop_default" in result
        assert "energy_source" in result
        assert "provenance_hash" in result

    @pytest.mark.parametrize("technology", ALL_9_COOLING_TECHS)
    def test_get_cop_returns_decimal(self, db, technology):
        """get_cop returns a Decimal for all 9 technologies."""
        cop = db.get_cop(technology)
        assert isinstance(cop, Decimal)
        assert cop > Decimal("0")

    def test_get_cop_centrifugal_chiller(self, db):
        """get_cop for centrifugal_chiller returns 6.0."""
        assert db.get_cop("centrifugal_chiller") == Decimal("6.0")

    def test_get_cop_absorption_double(self, db):
        """get_cop for absorption_double returns 1.2."""
        assert db.get_cop("absorption_double") == Decimal("1.2")

    def test_get_cop_free_cooling(self, db):
        """get_cop for free_cooling returns 20.0."""
        assert db.get_cop("free_cooling") == Decimal("20.0")

    def test_get_cop_unknown_raises(self, db):
        """get_cop raises ValueError for unknown technology."""
        with pytest.raises(ValueError, match="Unknown cooling technology"):
            db.get_cop("laser_cooling")

    def test_get_cooling_energy_source_electricity(self, db):
        """Electric chillers return 'electricity' as energy source."""
        assert db.get_cooling_energy_source("centrifugal_chiller") == "electricity"
        assert db.get_cooling_energy_source("screw_chiller") == "electricity"

    def test_get_cooling_energy_source_heat(self, db):
        """Absorption chillers return 'heat' as energy source."""
        assert db.get_cooling_energy_source("absorption_single") == "heat"
        assert db.get_cooling_energy_source("absorption_double") == "heat"
        assert db.get_cooling_energy_source("absorption_triple") == "heat"

    def test_get_cooling_energy_source_free(self, db):
        """Free cooling returns 'electricity' as energy source."""
        assert db.get_cooling_energy_source("free_cooling") == "electricity"

    def test_get_cooling_energy_source_unknown_raises(self, db):
        """get_cooling_energy_source raises ValueError for unknown."""
        with pytest.raises(ValueError, match="Unknown cooling technology"):
            db.get_cooling_energy_source("plasma_cooling")

    def test_get_all_cooling_factors_returns_9(self, db):
        """get_all_cooling_factors returns all 9 technologies."""
        all_factors = db.get_all_cooling_factors()
        assert len(all_factors) >= 9

    def test_get_cooling_technologies_sorted(self, db):
        """get_cooling_technologies returns sorted list."""
        techs = db.get_cooling_technologies()
        assert techs == sorted(techs)
        assert len(techs) >= 9


# ===========================================================================
# CHP Default Efficiencies Tests
# ===========================================================================


ALL_5_CHP_FUELS = list(CHP_DEFAULT_EFFICIENCIES.keys())


class TestCHPDefaults:
    """Tests for CHP default efficiency lookups."""

    @pytest.mark.parametrize("fuel_type", ALL_5_CHP_FUELS)
    def test_get_chp_defaults_all_5_fuels(self, db, fuel_type):
        """get_chp_defaults returns valid data for each CHP fuel type."""
        result = db.get_chp_defaults(fuel_type)
        assert result["fuel_type"] == fuel_type
        assert "electrical_efficiency" in result
        assert "thermal_efficiency" in result
        assert "overall_efficiency" in result
        assert "provenance_hash" in result

    @pytest.mark.parametrize("fuel_type", ALL_5_CHP_FUELS)
    def test_chp_efficiency_values_within_bounds(self, db, fuel_type):
        """CHP efficiency values are within [0, 1]."""
        result = db.get_chp_defaults(fuel_type)
        for key in ["electrical_efficiency", "thermal_efficiency", "overall_efficiency"]:
            val = result[key]
            assert isinstance(val, Decimal)
            assert Decimal("0") <= val <= Decimal("1"), (
                f"{key}={val} out of bounds for {fuel_type}"
            )

    def test_get_chp_defaults_unknown_raises(self, db):
        """get_chp_defaults raises ValueError for unknown CHP fuel type."""
        with pytest.raises(ValueError, match="Unknown CHP fuel type"):
            db.get_chp_defaults("hydrogen")

    def test_get_all_chp_defaults_returns_5(self, db):
        """get_all_chp_defaults returns all 5 CHP fuel types."""
        all_chp = db.get_all_chp_defaults()
        assert len(all_chp) >= 5


# ===========================================================================
# GWP Values Tests
# ===========================================================================


ALL_4_GWP_SOURCES = ["AR4", "AR5", "AR6", "AR6_20YR"]


class TestGWPValues:
    """Tests for GWP value lookups."""

    @pytest.mark.parametrize("source", ALL_4_GWP_SOURCES)
    def test_get_gwp_values_all_4_sources(self, db, source):
        """get_gwp_values returns valid data for each of 4 sources."""
        result = db.get_gwp_values(source)
        assert result["source"] == source
        assert result["co2_gwp"] == Decimal("1")
        assert isinstance(result["ch4_gwp"], Decimal)
        assert isinstance(result["n2o_gwp"], Decimal)
        assert "provenance_hash" in result

    def test_get_gwp_values_ar5_ch4(self, db):
        """AR5 CH4 GWP is 28."""
        result = db.get_gwp_values("AR5")
        assert result["ch4_gwp"] == Decimal("28")

    def test_get_gwp_values_ar6_ch4(self, db):
        """AR6 CH4 GWP is 27.9."""
        result = db.get_gwp_values("AR6")
        assert result["ch4_gwp"] == Decimal("27.9")

    def test_get_gwp_values_unknown_raises(self, db):
        """get_gwp_values raises ValueError for unknown source."""
        with pytest.raises(ValueError, match="Unknown GWP source"):
            db.get_gwp_values("AR99")

    def test_get_gwp_individual_gas(self, db):
        """get_gwp returns correct value for individual gas."""
        assert db.get_gwp("CO2", "AR5") == Decimal("1")
        assert db.get_gwp("CH4", "AR5") == Decimal("28")
        assert db.get_gwp("N2O", "AR5") == Decimal("265")

    def test_get_gwp_unknown_gas_raises(self, db):
        """get_gwp raises ValueError for unknown gas."""
        with pytest.raises(ValueError, match="Unknown gas"):
            db.get_gwp("SF6", "AR5")


# ===========================================================================
# Unit Conversion Tests
# ===========================================================================


class TestUnitConversions:
    """Tests for energy unit conversions."""

    def test_gj_to_mwh_conversion(self, db):
        """1 GJ = 0.277778 MWh."""
        result = db.convert_energy(Decimal("1"), "gj", "mwh")
        assert result == pytest.approx(Decimal("0.277778"), rel=Decimal("1e-4"))

    def test_mwh_to_gj_conversion(self, db):
        """1 MWh = 3.6 GJ."""
        result = db.convert_energy(Decimal("1"), "mwh", "gj")
        assert result == pytest.approx(Decimal("3.6"), rel=Decimal("1e-4"))

    def test_roundtrip_gj_mwh_gj(self, db):
        """GJ -> MWh -> GJ roundtrip preserves value."""
        original = Decimal("100")
        mwh = db.convert_energy(original, "gj", "mwh")
        back = db.convert_energy(mwh, "mwh", "gj")
        assert back == pytest.approx(original, rel=Decimal("0.001"))

    def test_same_unit_no_conversion(self, db):
        """Converting GJ to GJ returns the same value."""
        assert db.convert_energy(Decimal("42"), "gj", "gj") == Decimal("42")

    def test_negative_value_raises(self, db):
        """Negative energy value raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            db.convert_energy(Decimal("-1"), "gj", "mwh")

    def test_unknown_from_unit_raises(self, db):
        """Unknown source unit raises ValueError."""
        with pytest.raises(ValueError, match="Unknown energy unit"):
            db.convert_energy(Decimal("1"), "gallons", "gj")

    def test_unknown_to_unit_raises(self, db):
        """Unknown target unit raises ValueError."""
        with pytest.raises(ValueError, match="Unknown energy unit"):
            db.convert_energy(Decimal("1"), "gj", "gallons")

    def test_get_conversion_factor_mwh_to_gj(self, db):
        """get_conversion_factor returns correct mwh_to_gj factor."""
        factor = db.get_conversion_factor("mwh_to_gj")
        assert factor == Decimal("3.6")

    def test_get_conversion_factor_unknown_raises(self, db):
        """get_conversion_factor raises ValueError for unknown."""
        with pytest.raises(ValueError, match="Unknown conversion factor"):
            db.get_conversion_factor("lightyears_to_gj")


# ===========================================================================
# Blended Emission Factor Tests
# ===========================================================================


class TestBlendedEmissionFactor:
    """Tests for blended multi-fuel EF calculation."""

    def test_blended_ef_70_gas_30_biomass(self, db):
        """70% natural_gas + 30% biomass_wood blended EF is weighted average."""
        result = db.get_blended_ef({
            "natural_gas": Decimal("0.70"),
            "biomass_wood": Decimal("0.30"),
        })
        # Expected: 0.70 * 56.1 + 0.30 * 112.0 = 39.27 + 33.6 = 72.87
        expected_co2 = Decimal("0.70") * Decimal("56.100") + Decimal("0.30") * Decimal("112.000")
        assert result["blended_co2_ef"] == pytest.approx(expected_co2, rel=Decimal("0.01"))
        assert "provenance_hash" in result

    def test_blended_ef_biogenic_fraction(self, db):
        """Blended EF correctly reports biogenic fraction."""
        result = db.get_blended_ef({
            "natural_gas": Decimal("0.60"),
            "biomass_wood": Decimal("0.40"),
        })
        assert result["biogenic_fraction"] == pytest.approx(Decimal("0.40"), rel=Decimal("0.01"))

    def test_blended_ef_empty_mix_raises(self, db):
        """Empty fuel mix raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            db.get_blended_ef({})

    def test_blended_ef_fractions_not_summing_to_1_raises(self, db):
        """Fractions not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must sum to"):
            db.get_blended_ef({
                "natural_gas": Decimal("0.50"),
                "coal_bituminous": Decimal("0.20"),
            })


# ===========================================================================
# Search Tests
# ===========================================================================


class TestSearch:
    """Tests for search_factors method."""

    def test_search_finds_natural_gas(self, db):
        """Search for 'natural' finds natural_gas."""
        results = db.search_factors("natural")
        assert len(results) > 0
        found_fuel = any(r.get("fuel_type") == "natural_gas" for r in results)
        assert found_fuel is True

    def test_search_across_all_tables(self, db):
        """Search for 'coal' finds coal fuel types."""
        results = db.search_factors("coal")
        assert len(results) > 0


# ===========================================================================
# Custom Factors Tests
# ===========================================================================


class TestCustomFactors:
    """Tests for set/remove custom fuel emission factors."""

    def test_set_custom_fuel_ef(self, db):
        """Custom fuel EF overrides built-in data."""
        db.set_custom_fuel_ef("natural_gas", {
            "co2_ef": Decimal("60.000"),
            "ch4_ef": Decimal("0.002"),
            "n2o_ef": Decimal("0.0002"),
            "default_efficiency": Decimal("0.90"),
            "is_biogenic": False,
        })
        result = db.get_fuel_ef("natural_gas")
        assert result["co2_ef"] == Decimal("60.000")
        assert result["source"] == "custom"

    def test_remove_custom_fuel_ef(self, db):
        """Removing custom EF reverts to built-in data."""
        db.set_custom_fuel_ef("natural_gas", {
            "co2_ef": Decimal("60.000"),
            "ch4_ef": Decimal("0.002"),
            "n2o_ef": Decimal("0.0002"),
            "default_efficiency": Decimal("0.90"),
        })
        db.remove_custom_fuel_ef("natural_gas")
        result = db.get_fuel_ef("natural_gas")
        assert result["co2_ef"] == Decimal("56.100")
        assert result["source"] == "builtin"


# ===========================================================================
# Health Check Tests
# ===========================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_healthy(self, db):
        """health_check returns healthy status."""
        result = db.health_check()
        assert result["status"] == "healthy"
        assert result["engine"] == "steam_heat_database"
        assert "fuel_types" in result
        assert "dh_regions" in result
        assert "cooling_technologies" in result


# ===========================================================================
# Reset Tests
# ===========================================================================


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_custom_factors(self, db):
        """reset clears custom factors and counters."""
        db.set_custom_fuel_ef("natural_gas", {
            "co2_ef": Decimal("99"),
            "ch4_ef": Decimal("0"),
            "n2o_ef": Decimal("0"),
            "default_efficiency": Decimal("0.5"),
        })
        db.reset()
        result = db.get_fuel_ef("natural_gas")
        assert result["co2_ef"] == Decimal("56.100")
