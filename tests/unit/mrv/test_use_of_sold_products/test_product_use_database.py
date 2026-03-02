# -*- coding: utf-8 -*-
"""
Unit tests for ProductUseDatabaseEngine -- AGENT-MRV-024

Tests all lookup methods of ProductUseDatabaseEngine including product profiles,
fuel emission factors, refrigerant GWPs (AR5 and AR6), grid emission factors
(16 regions), lifetime defaults, degradation rates, steam/cooling factors,
chemical EFs, and feedstock properties.

Target: 35+ tests with exhaustive parametrize coverage.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pytest

try:
    from greenlang.use_of_sold_products.product_use_database import (
        ProductUseDatabaseEngine,
        get_database_engine,
        calculate_provenance_hash,
        ProductCategory,
        ProductSubcategory,
        FuelType,
        RefrigerantType,
        GridRegion,
        PRODUCT_PROFILES,
        FUEL_EMISSION_FACTORS,
        REFRIGERANT_GWPS,
        GRID_EMISSION_FACTORS,
        LIFETIME_DEFAULTS,
        DEGRADATION_RATES,
        STEAM_COOLING_EFS,
        CHEMICAL_EFS,
        FEEDSTOCK_PROPERTIES,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ProductUseDatabaseEngine not available")
pytestmark = _SKIP


# ============================================================================
# HELPERS
# ============================================================================

def _is_valid_hash(h: str) -> bool:
    """Return True if h is a 64-char lowercase hex string."""
    return len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# ============================================================================
# TEST: Product Profiles
# ============================================================================


class TestProductProfiles:
    """Test get_product_profile for all product subcategories."""

    @pytest.mark.parametrize(
        "subcategory,expected_lifetime",
        [
            ("passenger_car", 15),
            ("refrigerator", 15),
            ("air_conditioner", 12),
            ("laptop", 5),
            ("generator", 20),
            ("led_bulb", 25),
            ("furnace", 20),
            ("washing_machine", 12),
            ("heat_pump", 15),
            ("server", 7),
            ("electric_oven", 15),
            ("commercial_chiller", 20),
        ],
    )
    def test_product_profile_lifetime(self, subcategory, expected_lifetime):
        """Test product profile returns correct default lifetime."""
        engine = ProductUseDatabaseEngine()
        profile = engine.get_product_profile(subcategory)
        assert profile is not None
        assert profile["default_lifetime_years"] == expected_lifetime

    @pytest.mark.parametrize(
        "subcategory",
        [
            "passenger_car", "refrigerator", "air_conditioner", "laptop",
            "generator", "led_bulb", "furnace", "washing_machine",
            "heat_pump", "server", "electric_oven", "commercial_chiller",
        ],
    )
    def test_product_profile_has_degradation_rate(self, subcategory):
        """Test all product profiles include degradation_rate."""
        engine = ProductUseDatabaseEngine()
        profile = engine.get_product_profile(subcategory)
        assert "degradation_rate" in profile

    def test_product_profile_unknown_returns_none(self):
        """Test unknown subcategory returns None."""
        engine = ProductUseDatabaseEngine()
        profile = engine.get_product_profile("unknown_product")
        assert profile is None

    def test_product_profile_case_insensitive(self):
        """Test profile lookup is case-insensitive."""
        engine = ProductUseDatabaseEngine()
        lower = engine.get_product_profile("passenger_car")
        upper = engine.get_product_profile("PASSENGER_CAR")
        assert lower is not None
        assert upper is not None
        assert lower["default_lifetime_years"] == upper["default_lifetime_years"]

    def test_all_profiles_count(self):
        """Test at least 12 product profiles exist."""
        assert len(PRODUCT_PROFILES) >= 12

    @pytest.mark.parametrize(
        "subcategory,has_refrigerant",
        [
            ("air_conditioner", True),
            ("heat_pump", True),
            ("commercial_chiller", True),
            ("passenger_car", False),
            ("laptop", False),
            ("led_bulb", False),
        ],
    )
    def test_profiles_refrigerant_flag(self, subcategory, has_refrigerant):
        """Test which profiles include refrigerant charge data."""
        engine = ProductUseDatabaseEngine()
        profile = engine.get_product_profile(subcategory)
        if has_refrigerant:
            assert "refrigerant_charge_kg" in profile
            assert "annual_leak_rate" in profile
        else:
            assert "refrigerant_charge_kg" not in profile


# ============================================================================
# TEST: Fuel Emission Factors
# ============================================================================


class TestFuelEmissionFactors:
    """Test get_fuel_ef for all 15 fuel types."""

    @pytest.mark.parametrize(
        "fuel_type,expected_ef",
        [
            ("gasoline", Decimal("2.315")),
            ("diesel", Decimal("2.680")),
            ("lpg", Decimal("1.553")),
            ("kerosene", Decimal("2.540")),
            ("heating_oil", Decimal("2.960")),
            ("propane", Decimal("1.510")),
            ("ethanol_e85", Decimal("1.610")),
            ("biodiesel_b20", Decimal("2.144")),
            ("lng", Decimal("1.180")),
            ("wood_pellets", Decimal("0.039")),
        ],
    )
    def test_fuel_ef_lookup(self, fuel_type, expected_ef):
        """Test fuel emission factor lookup for common fuels."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_fuel_ef(fuel_type)
        assert "ef_kg_per_litre" in result or "ef_kg_per_m3" in result or "ef_kg_per_kg" in result

    def test_fuel_ef_unknown_raises(self):
        """Test unknown fuel type raises ValueError."""
        engine = ProductUseDatabaseEngine()
        with pytest.raises(ValueError):
            engine.get_fuel_ef("unknown_fuel")

    def test_all_15_fuel_types(self):
        """Test all 15 fuel types are present in database."""
        assert len(FUEL_EMISSION_FACTORS) >= 15

    def test_hydrogen_zero_direct_co2(self):
        """Test hydrogen has zero direct CO2 emissions."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_fuel_ef("hydrogen")
        assert result["ef_kg_per_kg"] == Decimal("0.000")


# ============================================================================
# TEST: Refrigerant GWPs
# ============================================================================


class TestRefrigerantGWPs:
    """Test get_refrigerant_gwp for all 10 refrigerants."""

    @pytest.mark.parametrize(
        "refrigerant,gwp_version,expected_gwp",
        [
            ("R-134a", "AR5", Decimal("1430")),
            ("R-134a", "AR6", Decimal("1530")),
            ("R-410A", "AR5", Decimal("2088")),
            ("R-410A", "AR6", Decimal("2256")),
            ("R-32", "AR5", Decimal("675")),
            ("R-32", "AR6", Decimal("771")),
            ("R-404A", "AR5", Decimal("3922")),
            ("R-290", "AR5", Decimal("3")),
            ("R-744", "AR5", Decimal("1")),
            ("R-1234yf", "AR5", Decimal("4")),
        ],
    )
    def test_refrigerant_gwp_lookup(self, refrigerant, gwp_version, expected_gwp):
        """Test refrigerant GWP lookup returns correct value."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_refrigerant_gwp(refrigerant, gwp_version)
        assert result == expected_gwp

    def test_refrigerant_gwp_unknown_raises(self):
        """Test unknown refrigerant raises ValueError."""
        engine = ProductUseDatabaseEngine()
        with pytest.raises(ValueError):
            engine.get_refrigerant_gwp("R-INVALID", "AR5")

    def test_all_10_refrigerants(self):
        """Test all 10 refrigerants are present."""
        assert len(REFRIGERANT_GWPS) >= 10

    def test_ar6_generally_higher_for_hfcs(self):
        """Test AR6 GWPs are generally higher for HFC refrigerants."""
        for ref in ["R-134a", "R-410A", "R-404A"]:
            ar5 = REFRIGERANT_GWPS[RefrigerantType(ref)]["gwp_ar5"]
            ar6 = REFRIGERANT_GWPS[RefrigerantType(ref)]["gwp_ar6"]
            assert ar6 >= ar5, f"{ref}: AR6 ({ar6}) should be >= AR5 ({ar5})"


# ============================================================================
# TEST: Grid Emission Factors
# ============================================================================


class TestGridEmissionFactors:
    """Test get_grid_ef for all 16 regions."""

    @pytest.mark.parametrize(
        "region,expected_ef",
        [
            ("US", Decimal("0.417")),
            ("US_CAMX", Decimal("0.275")),
            ("US_RFCW", Decimal("0.520")),
            ("US_SRMW", Decimal("0.680")),
            ("DE", Decimal("0.350")),
            ("CN", Decimal("0.580")),
            ("GB", Decimal("0.230")),
            ("JP", Decimal("0.470")),
            ("IN", Decimal("0.710")),
            ("BR", Decimal("0.080")),
            ("FR", Decimal("0.060")),
            ("AU", Decimal("0.630")),
            ("CA", Decimal("0.130")),
            ("KR", Decimal("0.460")),
            ("ZA", Decimal("0.920")),
            ("GLOBAL", Decimal("0.440")),
        ],
    )
    def test_grid_ef_all_regions(self, region, expected_ef):
        """Test grid emission factor lookup for each of 16 regions."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_grid_ef(region)
        assert result["ef_kg_per_kwh"] == expected_ef

    def test_grid_ef_unknown_falls_back_to_global(self):
        """Test unknown region falls back to GLOBAL."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_grid_ef("ZZ")
        assert result["ef_kg_per_kwh"] == Decimal("0.440")


# ============================================================================
# TEST: Lifetime Defaults
# ============================================================================


class TestLifetimeDefaults:
    """Test get_default_lifetime for all categories."""

    @pytest.mark.parametrize(
        "category,expected_lifetime",
        [
            ("vehicles", 15),
            ("appliances", 15),
            ("hvac", 12),
            ("lighting", 25),
            ("it_equipment", 5),
            ("industrial_equipment", 20),
            ("building_products", 20),
            ("medical_devices", 12),
        ],
    )
    def test_default_lifetime_by_category(self, category, expected_lifetime):
        """Test default lifetime for each product category."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_default_lifetime(category)
        assert result == expected_lifetime


# ============================================================================
# TEST: Degradation Rates
# ============================================================================


class TestDegradationRates:
    """Test get_degradation_rate for product categories."""

    def test_vehicles_degradation_rate(self):
        """Test vehicles degradation rate is 0.005."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_degradation_rate("vehicles")
        assert result == Decimal("0.005")

    def test_it_equipment_degradation_rate(self):
        """Test IT equipment degradation rate is 0.02."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_degradation_rate("it_equipment")
        assert result == Decimal("0.02")

    def test_unknown_degradation_zero(self):
        """Test unknown category returns zero degradation."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_degradation_rate("unknown")
        assert result == Decimal("0.0")


# ============================================================================
# TEST: Steam/Cooling EFs
# ============================================================================


class TestSteamCoolingEFs:
    """Test get_steam_cooling_ef for all system types."""

    @pytest.mark.parametrize(
        "system_type,expected_ef",
        [
            ("steam_boiler_gas", Decimal("0.200")),
            ("steam_boiler_oil", Decimal("0.270")),
            ("steam_chp", Decimal("0.150")),
            ("cooling_electric_chiller", Decimal("0.140")),
            ("cooling_absorption", Decimal("0.090")),
            ("district_heating", Decimal("0.180")),
            ("district_cooling", Decimal("0.120")),
        ],
    )
    def test_steam_cooling_ef_lookup(self, system_type, expected_ef):
        """Test steam/cooling EF lookup for each system type."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_steam_cooling_ef(system_type)
        assert result["ef_kg_per_kwh"] == expected_ef


# ============================================================================
# TEST: Chemical EFs
# ============================================================================


class TestChemicalEFs:
    """Test get_chemical_ef for chemical product types."""

    def test_hfc134a_gwp(self):
        """Test HFC-134a GWP is 1430."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_chemical_ef("HFC-134a")
        assert result["gwp"] == Decimal("1430")

    def test_sf6_gwp(self):
        """Test SF6 GWP is 22800."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_chemical_ef("sf6")
        assert result["gwp"] == Decimal("22800")


# ============================================================================
# TEST: Feedstock Properties
# ============================================================================


class TestFeedstockProperties:
    """Test get_feedstock_properties for feedstock types."""

    @pytest.mark.parametrize(
        "feedstock,expected_carbon",
        [
            ("naphtha", Decimal("0.836")),
            ("ethane", Decimal("0.799")),
            ("natural_gas_liquid", Decimal("0.830")),
            ("coal_feedstock", Decimal("0.710")),
        ],
    )
    def test_feedstock_carbon_content(self, feedstock, expected_carbon):
        """Test feedstock carbon content lookup."""
        engine = ProductUseDatabaseEngine()
        result = engine.get_feedstock_properties(feedstock)
        assert result["carbon_content"] == expected_carbon


# ============================================================================
# TEST: Provenance Hash
# ============================================================================


class TestDatabaseProvenance:
    """Test provenance hash generation from database lookups."""

    def test_provenance_hash_64_chars(self):
        """Test provenance hash is 64-char hex."""
        h = calculate_provenance_hash("gasoline", "US", "2024")
        assert _is_valid_hash(h)

    def test_provenance_hash_deterministic(self):
        """Test same inputs produce same hash."""
        h1 = calculate_provenance_hash("gasoline", "US", "2024")
        h2 = calculate_provenance_hash("gasoline", "US", "2024")
        assert h1 == h2

    def test_provenance_hash_different_inputs(self):
        """Test different inputs produce different hashes."""
        h1 = calculate_provenance_hash("gasoline", "US")
        h2 = calculate_provenance_hash("diesel", "DE")
        assert h1 != h2


# ============================================================================
# TEST: Database Summary
# ============================================================================


class TestDatabaseSummary:
    """Test get_database_summary."""

    def test_summary_contains_counts(self):
        """Test database summary includes all count fields."""
        engine = ProductUseDatabaseEngine()
        summary = engine.get_database_summary()
        assert "categories" in summary
        assert "fuel_types" in summary
        assert "regions" in summary
        assert "refrigerants" in summary
        assert summary["categories"] >= 10
        assert summary["fuel_types"] >= 15
        assert summary["regions"] >= 16
        assert summary["refrigerants"] >= 10
