# -*- coding: utf-8 -*-
"""
Unit tests for VehicleDatabaseEngine (Engine 1) - AGENT-MRV-003 Mobile Combustion.

Tests all public methods with 95+ test functions covering:
- Initialization, vehicle types, fuel types, emission factors
- CH4/N2O factors by model year and control technology
- Distance emission factors, GWP values, custom factors
- Search, statistics, edge cases, provenance tracking

Author: GreenLang QA Team
"""

import threading
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.mobile_combustion.vehicle_database import (
    CONTROL_TECHNOLOGIES,
    FUEL_TYPES,
    VEHICLE_CATEGORIES,
    VEHICLE_TYPES,
    VehicleDatabaseEngine,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def db():
    """Create a default VehicleDatabaseEngine instance."""
    return VehicleDatabaseEngine()


@pytest.fixture
def db_no_provenance():
    """Create a VehicleDatabaseEngine with provenance disabled."""
    return VehicleDatabaseEngine(config={"enable_provenance": False})


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test VehicleDatabaseEngine initialization."""

    def test_default_init(self, db):
        """Engine initializes with default configuration."""
        assert db._config == {}
        assert db._enable_provenance is True
        assert db._custom_factors == {}

    def test_init_with_config(self):
        """Engine initializes with custom configuration."""
        config = {"enable_provenance": False, "custom_key": "value"}
        engine = VehicleDatabaseEngine(config=config)
        assert engine._enable_provenance is False
        assert engine._config["custom_key"] == "value"

    def test_init_creates_lock(self, db):
        """Engine creates a threading lock on initialization."""
        assert isinstance(db._lock, type(threading.Lock()))

    def test_init_loads_18_vehicle_types(self):
        """Engine loads exactly 18 vehicle types."""
        assert len(VEHICLE_TYPES) == 18

    def test_init_loads_15_fuel_types(self):
        """Engine loads exactly 15 fuel types."""
        assert len(FUEL_TYPES) == 15

    def test_init_loads_11_control_technologies(self):
        """Engine loads exactly 11 control technologies."""
        assert len(CONTROL_TECHNOLOGIES) == 11

    def test_init_loads_5_categories(self):
        """Engine loads exactly 5 vehicle categories."""
        assert len(VEHICLE_CATEGORIES) == 5


# ===========================================================================
# TestVehicleTypes
# ===========================================================================


class TestVehicleTypes:
    """Test vehicle type lookup and listing."""

    def test_get_known_vehicle_type(self, db):
        """Get a known vehicle type returns correct data."""
        result = db.get_vehicle_type("HEAVY_DUTY_TRUCK")
        assert result["category"] == "ON_ROAD"
        assert result["vehicle_type"] == "HEAVY_DUTY_TRUCK"
        assert result["default_fuel"] == "DIESEL"
        assert result["default_fuel_economy_km_per_l"] == Decimal("2.8")

    def test_get_vehicle_type_case_insensitive(self, db):
        """Vehicle type lookup is case-insensitive."""
        result = db.get_vehicle_type("passenger_car_gasoline")
        assert result["vehicle_type"] == "PASSENGER_CAR_GASOLINE"

    def test_get_vehicle_type_strips_whitespace(self, db):
        """Vehicle type lookup strips whitespace."""
        result = db.get_vehicle_type("  MOTORCYCLE  ")
        assert result["vehicle_type"] == "MOTORCYCLE"

    def test_get_vehicle_type_includes_provenance(self, db):
        """Vehicle type lookup includes provenance hash when enabled."""
        result = db.get_vehicle_type("BUS_DIESEL")
        assert "_provenance_hash" in result
        assert len(result["_provenance_hash"]) == 64

    def test_get_vehicle_type_no_provenance_when_disabled(self, db_no_provenance):
        """Vehicle type lookup omits provenance when disabled."""
        result = db_no_provenance.get_vehicle_type("BUS_DIESEL")
        assert "_provenance_hash" not in result

    def test_get_unknown_vehicle_type_raises(self, db):
        """Unknown vehicle type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vehicle type"):
            db.get_vehicle_type("FLYING_CAR")

    def test_list_all_vehicle_types(self, db):
        """Listing without category returns all 18 vehicle types."""
        result = db.list_vehicle_types()
        assert len(result) == 18
        types = {v["vehicle_type"] for v in result}
        assert "HEAVY_DUTY_TRUCK" in types
        assert "DIESEL_LOCOMOTIVE" in types

    def test_list_vehicle_types_on_road(self, db):
        """Listing ON_ROAD category returns 11 vehicle types."""
        result = db.list_vehicle_types(category="ON_ROAD")
        assert len(result) == 11
        for v in result:
            assert v["category"] == "ON_ROAD"

    def test_list_vehicle_types_off_road(self, db):
        """Listing OFF_ROAD category returns 5 vehicle types."""
        result = db.list_vehicle_types(category="OFF_ROAD")
        assert len(result) == 5

    def test_list_vehicle_types_marine(self, db):
        """Listing MARINE category returns 3 vehicle types."""
        result = db.list_vehicle_types(category="MARINE")
        assert len(result) == 3

    def test_list_vehicle_types_aviation(self, db):
        """Listing AVIATION category returns 3 vehicle types."""
        result = db.list_vehicle_types(category="AVIATION")
        assert len(result) == 3

    def test_list_vehicle_types_rail(self, db):
        """Listing RAIL category returns 1 vehicle type."""
        result = db.list_vehicle_types(category="RAIL")
        assert len(result) == 1
        assert result[0]["vehicle_type"] == "DIESEL_LOCOMOTIVE"

    def test_list_vehicle_types_invalid_category_raises(self, db):
        """Listing with invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vehicle category"):
            db.list_vehicle_types(category="SPACESHIP")


# ===========================================================================
# TestFuelTypes
# ===========================================================================


class TestFuelTypes:
    """Test fuel type lookup and listing."""

    def test_get_diesel(self, db):
        """Get DIESEL fuel returns correct CO2 emission factor."""
        result = db.get_fuel_type("DIESEL")
        assert result["fuel_type"] == "DIESEL"
        assert result["co2_ef_kg_per_l"] == Decimal("2.68")
        assert result["density_kg_per_l"] == Decimal("0.832")
        assert result["biofuel_fraction"] == Decimal("0.0")

    def test_get_gasoline(self, db):
        """Get GASOLINE fuel returns correct properties."""
        result = db.get_fuel_type("GASOLINE")
        assert result["co2_ef_kg_per_l"] == Decimal("2.31")
        assert result["is_biofuel"] is False

    def test_get_biodiesel_b100(self, db):
        """Get BIODIESEL_B100 shows 100% biofuel fraction."""
        result = db.get_fuel_type("BIODIESEL_B100")
        assert result["biofuel_fraction"] == Decimal("1.00")
        assert result["is_biofuel"] is True

    def test_get_ethanol_e85_biofuel_fraction(self, db):
        """Ethanol E85 has 85% biofuel fraction."""
        result = db.get_fuel_type("ETHANOL_E85")
        assert result["biofuel_fraction"] == Decimal("0.85")

    def test_get_cng_special_units(self, db):
        """CNG uses m3-based units and has no per-liter CO2 EF."""
        result = db.get_fuel_type("CNG")
        assert result["unit"] == "m3"
        assert result["co2_ef_kg_per_l"] is None
        assert result["co2_ef_kg_per_m3"] == Decimal("2.02")

    def test_get_fuel_type_case_insensitive(self, db):
        """Fuel type lookup is case-insensitive."""
        result = db.get_fuel_type("jet_fuel_a")
        assert result["fuel_type"] == "JET_FUEL_A"

    def test_get_unknown_fuel_type_raises(self, db):
        """Unknown fuel type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.get_fuel_type("HYDROGEN")

    def test_list_fuel_types_returns_15(self, db):
        """Listing fuel types returns all 15."""
        result = db.list_fuel_types()
        assert len(result) == 15
        fuel_keys = {f["fuel_type"] for f in result}
        assert "GASOLINE" in fuel_keys
        assert "HFO" in fuel_keys
        assert "SAF" in fuel_keys

    def test_fuel_type_includes_provenance(self, db):
        """Fuel type lookup includes provenance hash."""
        result = db.get_fuel_type("DIESEL")
        assert "_provenance_hash" in result
        assert len(result["_provenance_hash"]) == 64


# ===========================================================================
# TestEmissionFactors
# ===========================================================================


class TestEmissionFactors:
    """Test emission factor lookups for CO2, CH4, N2O."""

    def test_co2_factor_diesel(self, db):
        """CO2 factor for DIESEL is 2.68 kg/L."""
        result = db.get_emission_factor("HEAVY_DUTY_TRUCK", "DIESEL", "CO2")
        assert result["value"] == Decimal("2.68")
        assert result["unit"] == "kg CO2/L"
        assert result["gas"] == "CO2"

    def test_ch4_factor_diesel(self, db):
        """CH4 factor for DIESEL is 3.9 kg/TJ."""
        result = db.get_emission_factor("HEAVY_DUTY_TRUCK", "DIESEL", "CH4")
        assert result["value"] == Decimal("3.9")
        assert result["unit"] == "kg CH4/TJ"

    def test_n2o_factor_gasoline(self, db):
        """N2O factor for GASOLINE is 0.6 kg/TJ."""
        result = db.get_emission_factor("PASSENGER_CAR_GASOLINE", "GASOLINE", "N2O")
        assert result["value"] == Decimal("0.6")

    def test_co2_factor_cng_uses_m3(self, db):
        """CNG CO2 factor uses kg CO2/m3 since per-L is None."""
        result = db.get_emission_factor("BUS_CNG", "CNG", "CO2")
        assert result["value"] == Decimal("2.02")
        assert result["unit"] == "kg CO2/m3"

    def test_emission_factor_unknown_vehicle_raises(self, db):
        """Unknown vehicle type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vehicle type"):
            db.get_emission_factor("FLYING_CAR", "DIESEL", "CO2")

    def test_emission_factor_unknown_fuel_raises(self, db):
        """Unknown fuel type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.get_emission_factor("HEAVY_DUTY_TRUCK", "HYDROGEN", "CO2")

    def test_emission_factor_unknown_gas_raises(self, db):
        """Unknown gas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gas"):
            db.get_emission_factor("HEAVY_DUTY_TRUCK", "DIESEL", "SF6")

    def test_emission_factor_provenance(self, db):
        """Emission factor includes provenance hash."""
        result = db.get_emission_factor("HEAVY_DUTY_TRUCK", "DIESEL", "CO2")
        assert "_provenance_hash" in result
        assert len(result["_provenance_hash"]) == 64


# ===========================================================================
# TestCH4N2OFactors
# ===========================================================================


class TestCH4N2OFactors:
    """Test CH4/N2O factor lookups by model year and control technology."""

    def test_heavy_duty_fixed_factors(self, db):
        """Heavy-duty trucks have fixed CH4/N2O factors regardless of year."""
        result = db.get_ch4_n2o_factors("HEAVY_DUTY_TRUCK")
        assert result["ch4_value"] == Decimal("0.0251")
        assert result["n2o_value"] == Decimal("0.0200")
        assert result["unit"] == "g/km"
        assert result["model_year_range"] == "ALL_YEARS"

    def test_bus_cng_high_ch4(self, db):
        """CNG bus has very high CH4 factor (1.9660 g/km)."""
        result = db.get_ch4_n2o_factors("BUS_CNG")
        assert result["ch4_value"] == Decimal("1.9660")

    def test_gasoline_car_2006_plus(self, db):
        """Gasoline car model year 2010 uses 2006_PLUS range."""
        result = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", model_year=2010)
        assert result["model_year_range"] == "2006_PLUS"
        assert result["ch4_value"] == Decimal("0.0113")
        assert result["n2o_value"] == Decimal("0.0132")

    def test_gasoline_car_1990(self, db):
        """Gasoline car model year 1990 uses 1985_1995 range."""
        result = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", model_year=1990)
        assert result["model_year_range"] == "1985_1995"
        assert result["ch4_value"] == Decimal("0.0394")

    def test_gasoline_car_2000(self, db):
        """Gasoline car model year 2000 uses 1996_2005 range."""
        result = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", model_year=2000)
        assert result["model_year_range"] == "1996_2005"

    def test_gasoline_car_pre_1985(self, db):
        """Gasoline car model year 1980 uses PRE_1985 range."""
        result = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", model_year=1980)
        assert result["model_year_range"] == "PRE_1985"
        assert result["ch4_value"] == Decimal("0.0602")

    def test_diesel_car_factors(self, db):
        """Diesel car uses diesel-specific CH4/N2O table."""
        result = db.get_ch4_n2o_factors("PASSENGER_CAR_DIESEL", model_year=2010)
        assert result["ch4_value"] == Decimal("0.0005")
        assert result["n2o_value"] == Decimal("0.0014")

    def test_model_year_none_defaults_to_2006_plus(self, db):
        """When model_year is None, defaults to 2006_PLUS range."""
        result = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE")
        assert result["model_year_range"] == "2006_PLUS"

    def test_offroad_factors_g_per_kg(self, db):
        """Off-road equipment uses g/kg-fuel units."""
        result = db.get_ch4_n2o_factors("CONSTRUCTION_EQUIPMENT")
        assert result["unit"] == "g/kg-fuel"
        assert result["ch4_value"] == Decimal("0.17")
        assert result["n2o_value"] == Decimal("0.12")

    def test_marine_factors(self, db):
        """Marine vessels use g/kg-fuel units."""
        result = db.get_ch4_n2o_factors("MARINE_OCEAN")
        assert result["unit"] == "g/kg-fuel"
        assert result["ch4_value"] == Decimal("0.30")

    def test_aviation_factors(self, db):
        """Aviation uses g/kg-fuel units."""
        result = db.get_ch4_n2o_factors("CORPORATE_JET")
        assert result["unit"] == "g/kg-fuel"
        assert result["ch4_value"] == Decimal("0.02")

    def test_rail_factors(self, db):
        """Rail uses g/kg-fuel units."""
        result = db.get_ch4_n2o_factors("DIESEL_LOCOMOTIVE")
        assert result["unit"] == "g/kg-fuel"
        assert result["ch4_value"] == Decimal("0.25")
        assert result["n2o_value"] == Decimal("0.30")


# ===========================================================================
# TestControlTechnology
# ===========================================================================


class TestControlTechnology:
    """Test control technology lookups and adjustments."""

    def test_three_way_catalyst_adjustment(self, db):
        """Three-way catalyst reduces CH4 (0.3x) but increases N2O (1.5x)."""
        result = db.get_ch4_n2o_factors(
            "PASSENGER_CAR_GASOLINE",
            model_year=2010,
            control_technology="THREE_WAY_CATALYST",
        )
        base = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", model_year=2010)
        expected_ch4 = (base["ch4_value"] * Decimal("0.3")).quantize(Decimal("0.00000001"))
        expected_n2o = (base["n2o_value"] * Decimal("1.5")).quantize(Decimal("0.00000001"))
        assert result["ch4_value"] == expected_ch4
        assert result["n2o_value"] == expected_n2o
        assert result["control_technology"] == "THREE_WAY_CATALYST"

    def test_euro_6_low_multipliers(self, db):
        """Euro 6 technology has low CH4 (0.10) and N2O (0.40) multipliers."""
        result = db.get_ch4_n2o_factors(
            "PASSENGER_CAR_GASOLINE",
            model_year=2020,
            control_technology="EURO_6",
        )
        base = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", model_year=2020)
        expected_ch4 = (base["ch4_value"] * Decimal("0.10")).quantize(Decimal("0.00000001"))
        assert result["ch4_value"] == expected_ch4

    def test_uncontrolled_no_change(self, db):
        """UNCONTROLLED technology leaves factors unchanged (1.0x)."""
        result = db.get_ch4_n2o_factors(
            "HEAVY_DUTY_TRUCK",
            control_technology="UNCONTROLLED",
        )
        assert result["ch4_value"] == Decimal("0.02510000")
        assert result["n2o_value"] == Decimal("0.02000000")

    def test_invalid_control_technology_raises(self, db):
        """Invalid control technology raises ValueError."""
        with pytest.raises(ValueError, match="Unknown control technology"):
            db.get_ch4_n2o_factors(
                "PASSENGER_CAR_GASOLINE",
                control_technology="FLUX_CAPACITOR",
            )

    def test_get_control_technology_details(self, db):
        """get_control_technology returns full details."""
        result = db.get_control_technology("ADVANCED_CATALYST")
        assert result["technology_id"] == "ADVANCED_CATALYST"
        assert result["ch4_multiplier"] == Decimal("0.1")
        assert result["n2o_multiplier"] == Decimal("0.5")

    def test_list_control_technologies(self, db):
        """list_control_technologies returns all 11 entries."""
        result = db.list_control_technologies()
        assert len(result) == 11
        ids = {t["technology_id"] for t in result}
        assert "EURO_1" in ids
        assert "EURO_6" in ids

    def test_get_unknown_control_technology_raises(self, db):
        """Getting unknown control technology raises ValueError."""
        with pytest.raises(ValueError, match="Unknown control technology"):
            db.get_control_technology("MAGIC_FILTER")


# ===========================================================================
# TestDistanceFactors
# ===========================================================================


class TestDistanceFactors:
    """Test distance-based emission factor lookups."""

    def test_passenger_car_gasoline_factor(self, db):
        """Gasoline car distance factor is 192 g CO2e/km."""
        result = db.get_distance_emission_factor("PASSENGER_CAR_GASOLINE", "GASOLINE")
        assert result == Decimal("192.0")

    def test_heavy_duty_truck_diesel_factor(self, db):
        """Heavy-duty truck distance factor is 951 g CO2e/km."""
        result = db.get_distance_emission_factor("HEAVY_DUTY_TRUCK", "DIESEL")
        assert result == Decimal("951.0")

    def test_hybrid_car_gasoline_factor(self, db):
        """Hybrid car distance factor is 110 g CO2e/km."""
        result = db.get_distance_emission_factor("PASSENGER_CAR_HYBRID", "GASOLINE")
        assert result == Decimal("110.0")

    def test_offroad_has_no_distance_factor(self, db):
        """Off-road equipment has no distance factor."""
        with pytest.raises(ValueError, match="No distance-based emission factors"):
            db.get_distance_emission_factor("CONSTRUCTION_EQUIPMENT", "DIESEL")

    def test_invalid_fuel_for_vehicle(self, db):
        """Invalid fuel for a given vehicle raises ValueError."""
        with pytest.raises(ValueError, match="No distance-based emission factor"):
            db.get_distance_emission_factor("PASSENGER_CAR_GASOLINE", "DIESEL")

    def test_unknown_vehicle_for_distance_raises(self, db):
        """Unknown vehicle type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vehicle type"):
            db.get_distance_emission_factor("FLYING_CAR", "DIESEL")


# ===========================================================================
# TestGWP
# ===========================================================================


class TestGWP:
    """Test GWP value lookups."""

    def test_ar6_co2_gwp(self, db):
        """AR6 CO2 GWP is 1."""
        assert db.get_gwp("CO2", "AR6") == Decimal("1")

    def test_ar6_ch4_gwp(self, db):
        """AR6 CH4 GWP is 29.8."""
        assert db.get_gwp("CH4", "AR6") == Decimal("29.8")

    def test_ar6_n2o_gwp(self, db):
        """AR6 N2O GWP is 273."""
        assert db.get_gwp("N2O", "AR6") == Decimal("273")

    def test_ar4_ch4_gwp(self, db):
        """AR4 CH4 GWP is 25."""
        assert db.get_gwp("CH4", "AR4") == Decimal("25")

    def test_ar5_n2o_gwp(self, db):
        """AR5 N2O GWP is 265."""
        assert db.get_gwp("N2O", "AR5") == Decimal("265")

    def test_ar6_20yr_ch4_gwp(self, db):
        """AR6_20YR CH4 GWP is 82.5."""
        assert db.get_gwp("CH4", "AR6_20YR") == Decimal("82.5")

    def test_invalid_gwp_source_raises(self, db):
        """Invalid GWP source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown GWP source"):
            db.get_gwp("CO2", "AR99")

    def test_invalid_gas_raises(self, db):
        """Invalid gas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gas"):
            db.get_gwp("SF6", "AR6")


# ===========================================================================
# TestSearch
# ===========================================================================


class TestSearch:
    """Test emission factor search and filtering."""

    def test_search_no_filters(self, db):
        """Search with no filters returns non-empty results."""
        results = db.search_factors()
        assert len(results) > 0

    def test_search_by_vehicle_type(self, db):
        """Search filtered by vehicle type returns only that type."""
        results = db.search_factors({"vehicle_type": "MOTORCYCLE"})
        for r in results:
            assert r["vehicle_type"] == "MOTORCYCLE"

    def test_search_by_category(self, db):
        """Search filtered by category returns only that category."""
        results = db.search_factors({"category": "MARINE"})
        for r in results:
            assert r["category"] == "MARINE"

    def test_search_by_gas(self, db):
        """Search filtered by gas returns only that gas."""
        results = db.search_factors({"gas": "CH4"})
        for r in results:
            assert r["gas"] == "CH4"

    def test_search_biofuel_filter(self, db):
        """Search filtered by is_biofuel returns only biofuels."""
        results = db.search_factors({"is_biofuel": True, "fuel_type": "BIODIESEL_B100"})
        for r in results:
            assert r["is_biofuel"] is True

    def test_search_co2_threshold(self, db):
        """Search with min_co2_ef filter excludes low factors."""
        results = db.search_factors({"min_co2_ef": Decimal("3.0"), "fuel_type": "HFO"})
        for r in results:
            if r["gas"] == "CO2":
                assert r["value"] >= Decimal("3.0")

    def test_factor_count(self, db):
        """Factor count equals vehicle_types x fuel_types x 3 gases."""
        count = db.get_factor_count()
        expected = len(VEHICLE_TYPES) * len(FUEL_TYPES) * 3
        assert count == expected

    def test_factor_count_with_custom(self, db):
        """Factor count includes custom factors."""
        db.register_custom_factor(
            "test_cf", "MOTORCYCLE", "GASOLINE", "CO2",
            Decimal("2.5"), "kg CO2/L"
        )
        count = db.get_factor_count()
        expected = len(VEHICLE_TYPES) * len(FUEL_TYPES) * 3 + 1
        assert count == expected


# ===========================================================================
# TestCustomFactors
# ===========================================================================


class TestCustomFactors:
    """Test custom emission factor registration and management."""

    def test_register_custom_factor(self, db):
        """Register a custom factor and retrieve it."""
        fid = db.register_custom_factor(
            "my_factor", "MOTORCYCLE", "GASOLINE", "CO2",
            Decimal("2.50"), "kg CO2/L", "CUSTOM", {"note": "test"}
        )
        assert fid == "my_factor"
        cf = db.get_custom_factor("my_factor")
        assert cf["vehicle_type"] == "MOTORCYCLE"
        assert cf["fuel_type"] == "GASOLINE"
        assert cf["gas"] == "CO2"
        assert cf["value"] == Decimal("2.50")

    def test_list_custom_factors(self, db):
        """List custom factors returns registered entries."""
        db.register_custom_factor(
            "cf_a", "MOTORCYCLE", "GASOLINE", "CH4",
            Decimal("0.001"), "kg CH4/L"
        )
        db.register_custom_factor(
            "cf_b", "BUS_DIESEL", "DIESEL", "N2O",
            Decimal("0.002"), "kg N2O/L"
        )
        result = db.list_custom_factors()
        assert len(result) == 2
        ids = {f["factor_id"] for f in result}
        assert "cf_a" in ids
        assert "cf_b" in ids

    def test_delete_custom_factor(self, db):
        """Delete a custom factor removes it."""
        db.register_custom_factor(
            "del_me", "MOTORCYCLE", "GASOLINE", "CO2",
            Decimal("1.0"), "kg CO2/L"
        )
        assert db.delete_custom_factor("del_me") is True
        assert db.delete_custom_factor("del_me") is False

    def test_get_nonexistent_custom_factor_raises(self, db):
        """Getting a nonexistent custom factor raises ValueError."""
        with pytest.raises(ValueError, match="Custom factor not found"):
            db.get_custom_factor("nonexistent")

    def test_register_invalid_vehicle_type_raises(self, db):
        """Registering with invalid vehicle type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vehicle type"):
            db.register_custom_factor(
                "bad", "FLYING_CAR", "GASOLINE", "CO2",
                Decimal("1.0"), "kg CO2/L"
            )

    def test_register_invalid_fuel_type_raises(self, db):
        """Registering with invalid fuel type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.register_custom_factor(
                "bad2", "MOTORCYCLE", "HYDROGEN", "CO2",
                Decimal("1.0"), "kg CO2/L"
            )

    def test_register_invalid_gas_raises(self, db):
        """Registering with invalid gas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gas"):
            db.register_custom_factor(
                "bad3", "MOTORCYCLE", "GASOLINE", "SF6",
                Decimal("1.0"), "kg SF6/L"
            )


# ===========================================================================
# TestFuelProperties
# ===========================================================================


class TestFuelProperties:
    """Test fuel density, heating value, and biofuel fraction lookups."""

    def test_diesel_density(self, db):
        """DIESEL density is 0.832 kg/L."""
        assert db.get_fuel_density("DIESEL") == Decimal("0.832")

    def test_gasoline_density(self, db):
        """GASOLINE density is 0.745 kg/L."""
        assert db.get_fuel_density("GASOLINE") == Decimal("0.745")

    def test_unknown_fuel_density_raises(self, db):
        """Unknown fuel type raises ValueError for density."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.get_fuel_density("UNKNOWN")

    def test_diesel_hhv(self, db):
        """DIESEL HHV is 38.6 MJ/L."""
        assert db.get_heating_value("DIESEL", "HHV") == Decimal("38.6")

    def test_diesel_ncv(self, db):
        """DIESEL NCV is 36.4 MJ/L."""
        assert db.get_heating_value("DIESEL", "NCV") == Decimal("36.4")

    def test_cng_hhv_uses_m3(self, db):
        """CNG HHV returns value in MJ/m3."""
        assert db.get_heating_value("CNG", "HHV") == Decimal("39.0")

    def test_invalid_heating_basis_raises(self, db):
        """Invalid heating value basis raises ValueError."""
        with pytest.raises(ValueError, match="Invalid heating value basis"):
            db.get_heating_value("DIESEL", "LHV")

    def test_biofuel_fraction_diesel(self, db):
        """DIESEL biofuel fraction is 0.0."""
        assert db.get_biofuel_fraction("DIESEL") == Decimal("0.0")

    def test_biofuel_fraction_b20(self, db):
        """BIODIESEL_B20 biofuel fraction is 0.20."""
        assert db.get_biofuel_fraction("BIODIESEL_B20") == Decimal("0.20")

    def test_biofuel_fraction_saf(self, db):
        """SAF biofuel fraction is 0.50."""
        assert db.get_biofuel_fraction("SAF") == Decimal("0.50")

    def test_biofuel_fraction_unknown_raises(self, db):
        """Unknown fuel type raises ValueError for biofuel fraction."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            db.get_biofuel_fraction("PLUTONIUM")


# ===========================================================================
# TestVehicleCategories
# ===========================================================================


class TestVehicleCategories:
    """Test vehicle category lookups."""

    def test_get_on_road_category(self, db):
        """Get ON_ROAD category returns correct data."""
        result = db.get_vehicle_category("ON_ROAD")
        assert result["category"] == "ON_ROAD"
        assert result["display_name"] == "On-Road Vehicles"

    def test_get_unknown_category_raises(self, db):
        """Unknown category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vehicle category"):
            db.get_vehicle_category("UNDERWATER")

    def test_list_vehicle_categories(self, db):
        """List categories returns all 5 with vehicle counts."""
        result = db.list_vehicle_categories()
        assert len(result) == 5
        cat_names = {c["category"] for c in result}
        assert cat_names == {"ON_ROAD", "OFF_ROAD", "MARINE", "AVIATION", "RAIL"}
        for cat in result:
            assert "vehicle_count" in cat
            assert cat["vehicle_count"] > 0


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and thread safety."""

    def test_thread_safety_custom_factors(self, db):
        """Custom factor operations are thread-safe."""
        errors = []

        def register_factors(prefix, count):
            try:
                for i in range(count):
                    db.register_custom_factor(
                        f"{prefix}_{i}", "MOTORCYCLE", "GASOLINE", "CO2",
                        Decimal("1.0"), "kg CO2/L"
                    )
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=register_factors, args=(f"t{t}", 10))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(db.list_custom_factors()) == 50

    def test_statistics(self, db):
        """get_statistics returns correct structure."""
        stats = db.get_statistics()
        assert stats["vehicle_types"] == 18
        assert stats["vehicle_categories"] == 5
        assert stats["fuel_types"] == 15
        assert stats["control_technologies"] == 11
        assert stats["custom_factors"] == 0
        assert "AR4" in stats["gwp_sources"]
        assert "AR6" in stats["gwp_sources"]

    def test_statistics_total_factors(self, db):
        """Total emission factors matches expected count."""
        stats = db.get_statistics()
        assert stats["total_emission_factors"] == 18 * 15 * 3

    def test_off_road_no_fuel_economy(self, db):
        """Off-road types have no default_fuel_economy_km_per_l."""
        result = db.get_vehicle_type("CONSTRUCTION_EQUIPMENT")
        assert result["default_fuel_economy_km_per_l"] is None

    def test_all_vehicle_types_have_category(self, db):
        """Every vehicle type has a valid category."""
        for vtype in VEHICLE_TYPES:
            data = db.get_vehicle_type(vtype)
            assert data["category"] in VEHICLE_CATEGORIES

    def test_all_fuel_types_have_density(self, db):
        """Every fuel type has a density value."""
        for ftype in FUEL_TYPES:
            data = db.get_fuel_type(ftype)
            has_density = (
                data.get("density_kg_per_l") is not None
                or data.get("density_kg_per_m3") is not None
            )
            assert has_density, f"{ftype} has no density"
