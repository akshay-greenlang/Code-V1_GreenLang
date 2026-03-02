# -*- coding: utf-8 -*-
"""
Unit tests for EmployeeCommutingDatabaseEngine (Engine 1).

Tests the thread-safe singleton database engine that provides emission factor
lookups for all commuting modes, vehicle types, transit types, telework energy,
EEIO spend-based factors, working days, currency conversion, and CPI deflation.

Target: ~50 tests covering all lookup methods, fallback logic, and error handling.

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: fresh database engine (reset singleton between tests)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the database engine singleton before each test."""
    from greenlang.employee_commuting.employee_commuting_database import (
        EmployeeCommutingDatabaseEngine,
    )
    EmployeeCommutingDatabaseEngine._instance = None
    yield
    EmployeeCommutingDatabaseEngine._instance = None


@pytest.fixture
def engine():
    """Create a fresh EmployeeCommutingDatabaseEngine instance."""
    from greenlang.employee_commuting.employee_commuting_database import (
        EmployeeCommutingDatabaseEngine,
    )
    return EmployeeCommutingDatabaseEngine()


# ===========================================================================
# 1. SINGLETON PATTERN TESTS
# ===========================================================================

class TestSingletonPattern:
    """Tests for the thread-safe singleton pattern."""

    def test_singleton_returns_same_instance(self, engine):
        """Two constructions return the same object."""
        from greenlang.employee_commuting.employee_commuting_database import (
            EmployeeCommutingDatabaseEngine,
        )
        second = EmployeeCommutingDatabaseEngine()
        assert engine is second

    def test_singleton_thread_safety(self):
        """Concurrent construction from multiple threads returns the same instance."""
        from greenlang.employee_commuting.employee_commuting_database import (
            EmployeeCommutingDatabaseEngine,
        )
        instances = []

        def _create():
            instances.append(EmployeeCommutingDatabaseEngine())

        threads = [threading.Thread(target=_create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)

    def test_engine_has_expected_attributes(self, engine):
        """Engine has ENGINE_ID, ENGINE_VERSION, and lookup_count."""
        assert engine.ENGINE_ID == "employee_commuting_database_engine"
        assert engine.ENGINE_VERSION == "1.0.0"
        assert engine.get_lookup_count() == 0


# ===========================================================================
# 2. VEHICLE EMISSION FACTOR TESTS
# ===========================================================================

class TestGetVehicleEmissionFactor:
    """Tests for get_vehicle_emission_factor()."""

    @pytest.mark.parametrize("vehicle_type,fuel_type,age,expected_co2e", [
        ("small_car", "gasoline", "new_0_3yr", Decimal("0.14520")),
        ("small_car", "diesel", "mid_4_7yr", Decimal("0.13890")),
        ("medium_car", "gasoline", "mid_4_7yr", Decimal("0.19228")),
        ("medium_car", "hybrid", "new_0_3yr", Decimal("0.12050")),
        ("large_car", "gasoline", "old_8_plus", Decimal("0.24920")),
        ("large_car", "electric", "new_0_3yr", Decimal("0.06170")),
        ("suv", "plugin_hybrid", "mid_4_7yr", Decimal("0.10500")),
        ("motorcycle_small", "gasoline", "new_0_3yr", Decimal("0.07850")),
        ("motorcycle_large", "gasoline", "mid_4_7yr", Decimal("0.12994")),
    ])
    def test_vehicle_ef_known_values(self, engine, vehicle_type, fuel_type, age, expected_co2e):
        """Verify known emission factor values for specific vehicle/fuel/age combos."""
        result = engine.get_vehicle_emission_factor(vehicle_type, fuel_type, age)
        assert result["co2e_per_km"] == expected_co2e.quantize(Decimal("0.00000001"))

    def test_vehicle_ef_default_age_is_mid(self, engine):
        """When no age is provided, defaults to mid_4_7yr."""
        result = engine.get_vehicle_emission_factor("medium_car", "gasoline")
        expected = Decimal("0.19228000")
        assert result["co2e_per_km"] == expected

    def test_vehicle_ef_returns_all_fields(self, engine):
        """Result dict contains all expected keys."""
        result = engine.get_vehicle_emission_factor("small_car", "gasoline", "new_0_3yr")
        expected_keys = {"co2e_per_km", "co2_per_km", "ch4_per_km", "n2o_per_km", "wtt_factor", "source"}
        assert set(result.keys()) == expected_keys

    def test_vehicle_ef_fallback_for_invalid_type(self, engine):
        """Unknown vehicle type falls back to medium_car/gasoline/mid_4_7yr."""
        result = engine.get_vehicle_emission_factor("flying_car", "gasoline")
        assert "(fallback)" in result["source"]
        assert result["co2e_per_km"] == Decimal("0.19228000")

    def test_vehicle_ef_fallback_for_invalid_fuel(self, engine):
        """Unknown fuel type falls back to medium_car/gasoline/mid_4_7yr."""
        result = engine.get_vehicle_emission_factor("small_car", "hydrogen")
        assert "(fallback)" in result["source"]

    def test_vehicle_ef_case_insensitive(self, engine):
        """Lookup is case-insensitive."""
        result = engine.get_vehicle_emission_factor("MEDIUM_CAR", "GASOLINE")
        assert result["co2e_per_km"] == Decimal("0.19228000")

    def test_vehicle_ef_increments_lookup_count(self, engine):
        """Each lookup increments the counter."""
        engine.get_vehicle_emission_factor("medium_car", "gasoline")
        engine.get_vehicle_emission_factor("small_car", "diesel")
        assert engine.get_lookup_count() == 2


# ===========================================================================
# 3. TRANSIT EMISSION FACTOR TESTS
# ===========================================================================

class TestGetTransitEmissionFactor:
    """Tests for get_transit_emission_factor()."""

    @pytest.mark.parametrize("transit_type,expected_co2e", [
        ("local_bus", Decimal("0.10312")),
        ("express_bus", Decimal("0.08956")),
        ("coach", Decimal("0.02732")),
        ("commuter_rail", Decimal("0.04115")),
        ("subway_metro", Decimal("0.03071")),
        ("light_rail", Decimal("0.02904")),
        ("tram_streetcar", Decimal("0.02940")),
        ("ferry_boat", Decimal("0.11318")),
        ("water_taxi", Decimal("0.14782")),
    ])
    def test_transit_ef_known_values(self, engine, transit_type, expected_co2e):
        """Verify known transit emission factors from DEFRA 2024."""
        result = engine.get_transit_emission_factor(transit_type)
        assert result["co2e_per_pkm"] == expected_co2e.quantize(Decimal("0.00000001"))

    def test_transit_ef_returns_required_keys(self, engine):
        """Result dict has co2e_per_pkm, wtt_per_pkm, source."""
        result = engine.get_transit_emission_factor("commuter_rail")
        assert "co2e_per_pkm" in result
        assert "wtt_per_pkm" in result
        assert "source" in result

    def test_transit_ef_fallback_for_unknown_type(self, engine):
        """Unknown transit type falls back to local_bus."""
        result = engine.get_transit_emission_factor("hyperloop")
        assert "(fallback)" in result["source"]
        assert result["co2e_per_pkm"] == Decimal("0.10312000")


# ===========================================================================
# 4. GRID EMISSION FACTOR TESTS
# ===========================================================================

class TestGetGridEmissionFactor:
    """Tests for get_grid_emission_factor()."""

    @pytest.mark.parametrize("country,expected_co2e", [
        ("US", Decimal("0.37170")),
        ("GB", Decimal("0.20707")),
        ("DE", Decimal("0.33800")),
        ("FR", Decimal("0.05100")),
        ("JP", Decimal("0.43400")),
        ("CA", Decimal("0.12000")),
        ("AU", Decimal("0.65600")),
        ("IN", Decimal("0.70800")),
        ("CN", Decimal("0.53700")),
        ("BR", Decimal("0.07400")),
        ("GLOBAL", Decimal("0.43600")),
    ])
    def test_grid_ef_country_level(self, engine, country, expected_co2e):
        """Verify country-level grid emission factors from IEA 2024."""
        result = engine.get_grid_emission_factor(country)
        assert result["co2e_per_kwh"] == expected_co2e.quantize(Decimal("0.00000001"))

    def test_grid_ef_egrid_subregion_overrides_country(self, engine):
        """eGRID sub-region value takes priority when provided."""
        result = engine.get_grid_emission_factor("US", region="CAMX")
        assert result["co2e_per_kwh"] == Decimal("0.22800000")
        assert "eGRID" in result["source"]

    def test_grid_ef_falls_back_to_global_for_unknown_country(self, engine):
        """Unknown country falls back to GLOBAL."""
        result = engine.get_grid_emission_factor("XX")
        assert result["co2e_per_kwh"] == Decimal("0.43600000")
        assert "global fallback" in result["source"].lower()


# ===========================================================================
# 5. WORKING DAYS TESTS
# ===========================================================================

class TestGetWorkingDays:
    """Tests for get_working_days()."""

    @pytest.mark.parametrize("country,expected_days", [
        ("US", 225),
        ("GB", 212),
        ("DE", 200),
        ("FR", 209),
        ("JP", 219),
        ("CA", 220),
        ("AU", 218),
        ("IN", 233),
        ("CN", 240),
        ("BR", 217),
        ("KR", 222),
    ])
    def test_working_days_by_country(self, engine, country, expected_days):
        """Verify known net working days for each country."""
        assert engine.get_working_days(country) == expected_days

    def test_working_days_fallback_to_global(self, engine):
        """Unknown country falls back to GLOBAL default (230)."""
        result = engine.get_working_days("ZZ")
        assert result == 230


# ===========================================================================
# 6. AVERAGE COMMUTE DISTANCE TESTS
# ===========================================================================

class TestGetAverageCommuteDistance:
    """Tests for get_average_commute_distance()."""

    def test_us_average_distance(self, engine):
        """US average commute distance is 21.7 km."""
        result = engine.get_average_commute_distance("US")
        assert result["avg_distance_km"] == Decimal("21.70000000")

    def test_global_average_distance(self, engine):
        """GLOBAL average commute distance is 15.0 km."""
        result = engine.get_average_commute_distance("GLOBAL")
        assert result["avg_distance_km"] == Decimal("15.00000000")

    def test_unknown_country_falls_back_to_global(self, engine):
        """Unknown country code falls back to GLOBAL."""
        result = engine.get_average_commute_distance("ZZ")
        assert result["avg_distance_km"] == Decimal("15.00000000")

    def test_result_contains_mode_split(self, engine):
        """Result includes a mode_split dict."""
        result = engine.get_average_commute_distance("US")
        assert "mode_split" in result
        assert "car" in result["mode_split"]


# ===========================================================================
# 7. TELEWORK FACTOR TESTS
# ===========================================================================

class TestGetTeleworkFactor:
    """Tests for get_telework_factor()."""

    @pytest.mark.parametrize("zone,expected_total", [
        ("tropical", Decimal("5.00")),
        ("arid", Decimal("7.00")),
        ("temperate", Decimal("6.00")),
        ("continental", Decimal("7.50")),
        ("polar", Decimal("8.60")),
    ])
    def test_telework_energy_defaults_by_zone(self, engine, zone, expected_total):
        """Verify total daily kWh for each climate zone."""
        result = engine.get_telework_factor(zone)
        assert result["total_kwh_per_day"] == expected_total.quantize(Decimal("0.00000001"))

    def test_telework_factor_fallback_for_unknown_zone(self, engine):
        """Unknown climate zone falls back to temperate."""
        result = engine.get_telework_factor("martian")
        assert result["total_kwh_per_day"] == Decimal("6.00000000")


# ===========================================================================
# 8. EEIO FACTOR TESTS
# ===========================================================================

class TestGetEeioFactor:
    """Tests for get_eeio_factor()."""

    @pytest.mark.parametrize("naics,expected_co2e,expected_desc", [
        ("485000", Decimal("0.26000"), "Ground passenger transport"),
        ("485110", Decimal("0.22000"), "Mixed mode transit systems"),
        ("447110", Decimal("0.63000"), "Gasoline stations with convenience stores"),
    ])
    def test_eeio_factor_known_values(self, engine, naics, expected_co2e, expected_desc):
        """Verify known EEIO factors by NAICS code."""
        result = engine.get_eeio_factor(naics)
        assert result["co2e_per_usd"] == expected_co2e.quantize(Decimal("0.00000001"))
        assert result["description"] == expected_desc

    def test_eeio_factor_raises_for_unknown_code(self, engine):
        """ValueError is raised for an unknown NAICS code."""
        with pytest.raises(ValueError, match="not found"):
            engine.get_eeio_factor("999999")


# ===========================================================================
# 9. HEATING FUEL FACTOR TESTS
# ===========================================================================

class TestGetHeatingFuelFactor:
    """Tests for get_heating_fuel_factor()."""

    def test_natural_gas_factor(self, engine):
        """Natural gas heating fuel factor value is correct."""
        result = engine.get_heating_fuel_factor("natural_gas")
        assert result["co2e_per_kwh"] == Decimal("0.18316000")

    def test_heating_oil_factor(self, engine):
        """Heating oil factor value is correct."""
        result = engine.get_heating_fuel_factor("heating_oil")
        assert result["co2e_per_kwh"] == Decimal("0.24674000")

    def test_raises_for_unknown_fuel(self, engine):
        """ValueError is raised for an unknown fuel type."""
        with pytest.raises(ValueError, match="not found"):
            engine.get_heating_fuel_factor("nuclear_fusion")


# ===========================================================================
# 10. CURRENCY AND CPI TESTS
# ===========================================================================

class TestCurrencyAndCpi:
    """Tests for get_currency_rate() and get_cpi_deflator()."""

    def test_usd_rate_is_one(self, engine):
        """USD rate to itself is 1.0."""
        assert engine.get_currency_rate("USD") == Decimal("1.00000000")

    def test_gbp_rate(self, engine):
        """GBP exchange rate is correct."""
        assert engine.get_currency_rate("GBP") == Decimal("1.26500000")

    def test_currency_rate_raises_for_unknown(self, engine):
        """ValueError for unknown currency code."""
        with pytest.raises(ValueError, match="not found"):
            engine.get_currency_rate("XYZ")

    def test_cpi_base_year_is_one(self, engine):
        """CPI deflator for base year 2021 is 1.0."""
        assert engine.get_cpi_deflator(2021) == Decimal("1.00000000")

    def test_cpi_2024(self, engine):
        """CPI deflator for 2024 is 1.14900000."""
        assert engine.get_cpi_deflator(2024) == Decimal("1.14900000")

    def test_cpi_raises_for_unknown_year(self, engine):
        """ValueError for an unsupported year."""
        with pytest.raises(ValueError, match="not available"):
            engine.get_cpi_deflator(1999)


# ===========================================================================
# 11. CARPOOL OCCUPANCY TESTS
# ===========================================================================

class TestGetCarpoolOccupancy:
    """Tests for get_carpool_default_occupancy()."""

    def test_medium_car_occupancy(self, engine):
        """Medium car default occupancy is 2.30."""
        assert engine.get_carpool_default_occupancy("medium_car") == Decimal("2.30000000")

    def test_minibus_occupancy(self, engine):
        """Minibus default occupancy is 15.00."""
        assert engine.get_carpool_default_occupancy("minibus") == Decimal("15.00000000")

    def test_unknown_type_falls_back(self, engine):
        """Unknown vehicle type falls back to 2.30."""
        assert engine.get_carpool_default_occupancy("tank") == Decimal("2.30000000")


# ===========================================================================
# 12. ENUMERATION METHODS TESTS
# ===========================================================================

class TestEnumerationMethods:
    """Tests for get_all_* methods and get_database_summary."""

    def test_get_all_vehicle_types_non_empty(self, engine):
        """get_all_vehicle_types returns a non-empty sorted list."""
        types = engine.get_all_vehicle_types()
        assert len(types) > 0
        assert types == sorted(types)
        assert "medium_car" in types

    def test_get_all_fuel_types_includes_gasoline(self, engine):
        """get_all_fuel_types includes gasoline and electric."""
        fuels = engine.get_all_fuel_types()
        assert "gasoline" in fuels
        assert "electric" in fuels

    def test_get_all_transit_types_has_nine(self, engine):
        """get_all_transit_types returns 9 transit types."""
        types = engine.get_all_transit_types()
        assert len(types) == 9
        assert "subway_metro" in types

    def test_get_all_commute_modes_includes_sov(self, engine):
        """get_all_commute_modes includes sov and telework."""
        modes = engine.get_all_commute_modes()
        assert "sov" in modes
        assert "telework" in modes

    def test_get_all_countries_includes_us(self, engine):
        """get_all_countries includes US and GLOBAL."""
        countries = engine.get_all_countries()
        assert "US" in countries
        assert "GLOBAL" in countries

    def test_get_database_summary_structure(self, engine):
        """get_database_summary returns expected keys."""
        summary = engine.get_database_summary()
        assert "engine_id" in summary
        assert "vehicle_types" in summary


# ===========================================================================
# 13. SEARCH AND MODE SHARES TESTS
# ===========================================================================

class TestSearchEmissionFactors:
    """Tests for search_emission_factors()."""

    def test_search_vehicle_diesel(self, engine):
        """Searching vehicle factors for 'diesel' returns results."""
        results = engine.search_emission_factors("vehicle", "diesel")
        assert len(results) > 0
        assert all("diesel" in r.get("fuel_type", "") for r in results)

    def test_search_transit_bus(self, engine):
        """Searching transit factors for 'bus' returns results."""
        results = engine.search_emission_factors("transit", "bus")
        assert len(results) > 0

    def test_search_empty_query_returns_empty(self, engine):
        """An empty query string returns no results."""
        results = engine.search_emission_factors("vehicle", "")
        assert results == []

    def test_search_unknown_mode_returns_empty(self, engine):
        """An unknown search mode returns no results."""
        results = engine.search_emission_factors("quantum", "test")
        assert results == []


# ===========================================================================
# 14. METRICS RECORDING TEST
# ===========================================================================

class TestMetricsRecording:
    """Test that metrics recording does not break lookups."""

    def test_lookup_count_increments(self, engine):
        """Lookup count tracks correctly across multiple methods."""
        engine.get_vehicle_emission_factor("medium_car", "gasoline")
        engine.get_transit_emission_factor("local_bus")
        engine.get_grid_emission_factor("US")
        engine.get_working_days("US")
        engine.get_average_commute_distance("US")
        engine.get_telework_factor("temperate")
        engine.get_eeio_factor("485000")
        engine.get_currency_rate("USD")
        engine.get_cpi_deflator(2021)
        engine.get_carpool_default_occupancy("medium_car")
        engine.search_emission_factors("vehicle", "gas")
        assert engine.get_lookup_count() == 11
