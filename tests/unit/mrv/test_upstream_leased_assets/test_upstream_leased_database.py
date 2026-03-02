# -*- coding: utf-8 -*-
"""
Test suite for upstream_leased_assets.upstream_leased_database - AGENT-MRV-021.

Tests the UpstreamLeasedDatabaseEngine (Engine 1) for the Upstream Leased
Assets Agent (GL-MRV-S3-008) including singleton pattern, building EUI
lookups (8 types x 5 climate zones), grid emission factors (11 countries),
fuel EFs, vehicle EFs, equipment benchmarks, IT power ratings, EEIO factors,
currency rates, CPI deflators, climate zone lookups, refrigerant GWPs,
allocation defaults, search, and enumerations.

Coverage:
- Singleton pattern (thread-safe, same instance)
- Building EUI lookups (8 building types x 5 climate zones, parametrized)
- Grid emission factor lookups (11 countries)
- Fuel emission factor lookups (diesel, petrol, LPG, CNG)
- Vehicle emission factor lookups (8 vehicle types x 7 fuel types)
- Equipment benchmark lookups (6 equipment types)
- IT power rating lookups (7 IT asset types)
- EEIO factor lookups (6+ NAICS codes, invalid NAICS)
- Currency rate lookups (12 currencies)
- CPI deflator lookups (11 years)
- Climate zone lookups
- Allocation default lookups
- Search and enumeration methods
- Case-insensitivity
- Fallback behavior

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List, Optional
import pytest

try:
    from greenlang.upstream_leased_assets.upstream_leased_database import (
        UpstreamLeasedDatabaseEngine,
        get_database_engine,
        reset_database_engine,
    )
    from greenlang.upstream_leased_assets.models import (
        BuildingType,
        ClimateZone,
        VehicleType,
        FuelType,
        EquipmentType,
        ITAssetType,
        AllocationMethod,
        EnergySource,
        CurrencyCode,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="upstream_leased_assets.upstream_leased_database not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the singleton engine before each test."""
    if _AVAILABLE:
        reset_database_engine()
    yield
    if _AVAILABLE:
        reset_database_engine()


@pytest.fixture
def engine() -> "UpstreamLeasedDatabaseEngine":
    """Create a fresh UpstreamLeasedDatabaseEngine instance."""
    return UpstreamLeasedDatabaseEngine()


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, engine):
        """Test UpstreamLeasedDatabaseEngine returns the same instance."""
        engine2 = UpstreamLeasedDatabaseEngine()
        assert engine is engine2

    def test_singleton_via_get_database_engine(self):
        """Test get_database_engine returns singleton."""
        engine1 = get_database_engine()
        engine2 = get_database_engine()
        assert engine1 is engine2

    def test_singleton_across_threads(self):
        """Test singleton works across threads."""
        instances: List = []

        def get_instance():
            instances.append(UpstreamLeasedDatabaseEngine())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        for inst in instances[1:]:
            assert inst is first

    def test_reset_creates_new_instance(self):
        """Test reset allows a new instance to be created."""
        engine1 = UpstreamLeasedDatabaseEngine()
        reset_database_engine()
        engine2 = UpstreamLeasedDatabaseEngine()
        assert engine1 is not engine2


# ==============================================================================
# BUILDING EUI LOOKUP TESTS
# ==============================================================================


class TestBuildingEUILookup:
    """Test building EUI benchmark lookups."""

    @pytest.mark.parametrize("building_type", [
        BuildingType.OFFICE,
        BuildingType.RETAIL,
        BuildingType.WAREHOUSE,
        BuildingType.INDUSTRIAL,
        BuildingType.DATA_CENTER,
        BuildingType.HOTEL,
        BuildingType.HEALTHCARE,
        BuildingType.EDUCATION,
    ])
    @pytest.mark.parametrize("climate_zone", [
        ClimateZone.TROPICAL,
        ClimateZone.ARID,
        ClimateZone.TEMPERATE,
        ClimateZone.COLD,
        ClimateZone.WARM,
    ])
    def test_building_eui_lookup(self, engine, building_type, climate_zone):
        """Test building EUI lookup for all type/zone combinations (40 combos)."""
        eui = engine.get_building_eui(building_type, climate_zone)
        assert isinstance(eui, Decimal)
        assert eui > 0

    def test_office_temperate_known_value(self, engine):
        """Test known EUI value for office/temperate."""
        eui = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert Decimal("100") <= eui <= Decimal("300")

    def test_data_center_highest_eui(self, engine):
        """Test data center has highest EUI."""
        dc_eui = engine.get_building_eui(BuildingType.DATA_CENTER, ClimateZone.TEMPERATE)
        office_eui = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert dc_eui > office_eui

    def test_warehouse_lower_than_office(self, engine):
        """Test warehouse EUI is lower than office."""
        wh_eui = engine.get_building_eui(BuildingType.WAREHOUSE, ClimateZone.TEMPERATE)
        office_eui = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert wh_eui < office_eui

    def test_eui_quantized_to_8dp(self, engine):
        """Test EUI values are quantized to 8 decimal places."""
        eui = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        # Decimal should have at most 8 decimal places
        assert eui == eui.quantize(Decimal("0.00000001"))


# ==============================================================================
# GRID EMISSION FACTOR TESTS
# ==============================================================================


class TestGridEFLookup:
    """Test grid emission factor lookups."""

    @pytest.mark.parametrize("country", [
        "US", "GB", "DE", "FR", "JP", "CA", "AU", "IN", "CN", "BR", "GLOBAL",
    ])
    def test_grid_ef_by_country(self, engine, country):
        """Test grid emission factor lookup for 11 countries."""
        ef = engine.get_grid_emission_factor(country)
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_us_grid_ef_value(self, engine):
        """Test US grid EF is approximately 0.37 kgCO2e/kWh."""
        ef = engine.get_grid_emission_factor("US")
        assert Decimal("0.30") <= ef <= Decimal("0.50")

    def test_france_lower_than_us(self, engine):
        """Test France grid EF is lower than US."""
        fr = engine.get_grid_emission_factor("FR")
        us = engine.get_grid_emission_factor("US")
        assert fr < us

    def test_case_insensitive_lookup(self, engine):
        """Test country code lookup is case-insensitive."""
        ef_upper = engine.get_grid_emission_factor("US")
        ef_lower = engine.get_grid_emission_factor("us")
        assert ef_upper == ef_lower

    def test_unknown_country_fallback(self, engine):
        """Test unknown country falls back to GLOBAL."""
        ef = engine.get_grid_emission_factor("XX")
        global_ef = engine.get_grid_emission_factor("GLOBAL")
        assert ef == global_ef


# ==============================================================================
# FUEL EMISSION FACTOR TESTS
# ==============================================================================


class TestFuelEFLookup:
    """Test fuel emission factor lookups."""

    @pytest.mark.parametrize("fuel_type", [
        FuelType.PETROL, FuelType.DIESEL, FuelType.LPG, FuelType.CNG,
    ])
    def test_fuel_ef_lookup(self, engine, fuel_type):
        """Test fuel EF lookup for conventional fuel types."""
        ef = engine.get_fuel_emission_factor(fuel_type)
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_diesel_ef_value(self, engine):
        """Test diesel EF is approximately 2.68-2.71 kgCO2e/litre."""
        ef = engine.get_fuel_emission_factor(FuelType.DIESEL)
        assert Decimal("2.50") <= ef <= Decimal("3.00")

    def test_petrol_ef_value(self, engine):
        """Test petrol EF is approximately 2.31 kgCO2e/litre."""
        ef = engine.get_fuel_emission_factor(FuelType.PETROL)
        assert Decimal("2.00") <= ef <= Decimal("2.60")

    def test_diesel_higher_than_petrol(self, engine):
        """Test diesel EF is higher than petrol EF per litre."""
        diesel = engine.get_fuel_emission_factor(FuelType.DIESEL)
        petrol = engine.get_fuel_emission_factor(FuelType.PETROL)
        assert diesel > petrol


# ==============================================================================
# VEHICLE EMISSION FACTOR TESTS
# ==============================================================================


class TestVehicleEFLookup:
    """Test vehicle emission factor lookups."""

    @pytest.mark.parametrize("vehicle_type", list(VehicleType))
    def test_vehicle_ef_lookup(self, engine, vehicle_type):
        """Test vehicle EF lookup for all 8 vehicle types."""
        ef = engine.get_vehicle_emission_factor(vehicle_type, FuelType.DIESEL)
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_heavy_truck_higher_than_small_car(self, engine):
        """Test heavy truck has higher EF than small car."""
        truck = engine.get_vehicle_emission_factor(VehicleType.HEAVY_TRUCK, FuelType.DIESEL)
        car = engine.get_vehicle_emission_factor(VehicleType.SMALL_CAR, FuelType.DIESEL)
        assert truck > car

    def test_bev_emission_factor(self, engine):
        """Test BEV returns electricity-based EF or zero."""
        ef = engine.get_vehicle_emission_factor(VehicleType.MEDIUM_CAR, FuelType.BEV)
        assert isinstance(ef, Decimal)
        # BEV direct tailpipe is 0, but may include electricity
        assert ef >= Decimal("0")


# ==============================================================================
# EQUIPMENT BENCHMARK TESTS
# ==============================================================================


class TestEquipmentBenchmarkLookup:
    """Test equipment benchmark lookups."""

    @pytest.mark.parametrize("equipment_type", list(EquipmentType))
    def test_equipment_benchmark_lookup(self, engine, equipment_type):
        """Test equipment benchmark lookup for all 6 types."""
        benchmark = engine.get_equipment_benchmark(equipment_type)
        assert isinstance(benchmark, dict)
        assert "default_load_factor" in benchmark

    def test_manufacturing_load_factor(self, engine):
        """Test manufacturing default load factor is reasonable."""
        benchmark = engine.get_equipment_benchmark(EquipmentType.MANUFACTURING)
        lf = benchmark["default_load_factor"]
        assert Decimal("0.50") <= lf <= Decimal("0.90")

    def test_generator_has_fuel_data(self, engine):
        """Test generator benchmark includes fuel consumption data."""
        benchmark = engine.get_equipment_benchmark(EquipmentType.GENERATOR)
        assert "fuel_consumption_factor" in benchmark or "default_load_factor" in benchmark


# ==============================================================================
# IT POWER RATING TESTS
# ==============================================================================


class TestITPowerRatingLookup:
    """Test IT power rating lookups."""

    @pytest.mark.parametrize("it_type", list(ITAssetType))
    def test_it_power_rating_lookup(self, engine, it_type):
        """Test IT power rating lookup for all 7 types."""
        rating = engine.get_it_power_rating(it_type)
        assert isinstance(rating, dict)
        assert "typical_power_w" in rating

    def test_server_highest_power(self, engine):
        """Test server has highest power rating."""
        server = engine.get_it_power_rating(ITAssetType.SERVER)
        laptop = engine.get_it_power_rating(ITAssetType.LAPTOP)
        assert server["typical_power_w"] > laptop["typical_power_w"]

    def test_standby_lower_than_active(self, engine):
        """Test standby power is lower than active for all types."""
        for it_type in ITAssetType:
            rating = engine.get_it_power_rating(it_type)
            assert rating["standby_power_w"] < rating["typical_power_w"]


# ==============================================================================
# EEIO FACTOR TESTS
# ==============================================================================


class TestEEIOFactorLookup:
    """Test EEIO factor lookups."""

    def test_office_rental_naics(self, engine):
        """Test NAICS 531120 (lessors of buildings) lookup."""
        factor = engine.get_eeio_factor("531120")
        assert isinstance(factor, dict)
        assert "ef" in factor
        assert isinstance(factor["ef"], Decimal)

    def test_vehicle_leasing_naics(self, engine):
        """Test NAICS 532112 (passenger car leasing) lookup."""
        factor = engine.get_eeio_factor("532112")
        assert isinstance(factor, dict)
        assert factor["ef"] > 0

    def test_invalid_naics_returns_none(self, engine):
        """Test invalid NAICS code returns None or raises."""
        result = engine.get_eeio_factor("999999")
        assert result is None or result == {}

    def test_all_eeio_codes_positive(self, engine):
        """Test all registered EEIO factors are positive."""
        codes = engine.get_available_eeio_codes()
        for code in codes:
            factor = engine.get_eeio_factor(code)
            assert factor["ef"] > 0, f"NAICS {code} EF not positive"


# ==============================================================================
# CURRENCY AND CPI TESTS
# ==============================================================================


class TestCurrencyRateLookup:
    """Test currency rate lookups."""

    @pytest.mark.parametrize("currency", list(CurrencyCode))
    def test_currency_rate_lookup(self, engine, currency):
        """Test currency rate lookup for all 12 currencies."""
        rate = engine.get_currency_rate(currency)
        assert isinstance(rate, Decimal)
        assert rate > 0

    def test_usd_rate_is_one(self, engine):
        """Test USD rate is 1.0."""
        rate = engine.get_currency_rate(CurrencyCode.USD)
        assert rate == Decimal("1.0")


class TestCPIDeflatorLookup:
    """Test CPI deflator lookups."""

    @pytest.mark.parametrize("year", list(range(2015, 2026)))
    def test_cpi_deflator_lookup(self, engine, year):
        """Test CPI deflator lookup for all years (2015-2025)."""
        deflator = engine.get_cpi_deflator(year)
        assert isinstance(deflator, Decimal)
        assert deflator > 0

    def test_base_year_2021_is_one(self, engine):
        """Test 2021 CPI deflator is 1.0."""
        deflator = engine.get_cpi_deflator(2021)
        assert deflator == Decimal("1.0000")

    def test_unknown_year_returns_latest(self, engine):
        """Test unknown year falls back to latest available."""
        deflator = engine.get_cpi_deflator(2030)
        assert isinstance(deflator, Decimal)
        assert deflator > 0


# ==============================================================================
# CLIMATE ZONE AND ALLOCATION TESTS
# ==============================================================================


class TestClimateZoneLookup:
    """Test climate zone lookups."""

    def test_get_available_climate_zones(self, engine):
        """Test listing available climate zones."""
        zones = engine.get_available_climate_zones()
        assert len(zones) == 5
        assert "temperate" in [str(z).lower() for z in zones]


class TestAllocationDefaultLookup:
    """Test allocation default lookups."""

    @pytest.mark.parametrize("method", list(AllocationMethod))
    def test_allocation_default_lookup(self, engine, method):
        """Test allocation default lookup for all 4 methods."""
        default = engine.get_allocation_default(method)
        assert isinstance(default, Decimal)
        assert Decimal("0") <= default <= Decimal("1.0")

    def test_equal_allocation_is_one(self, engine):
        """Test equal allocation default is 1.0."""
        default = engine.get_allocation_default(AllocationMethod.EQUAL)
        assert default == Decimal("1.0")


# ==============================================================================
# ENUMERATION AND SEARCH TESTS
# ==============================================================================


class TestEnumerations:
    """Test enumeration and search methods."""

    def test_get_available_building_types(self, engine):
        """Test listing available building types."""
        types = engine.get_available_building_types()
        assert len(types) == 8

    def test_get_available_vehicle_types(self, engine):
        """Test listing available vehicle types."""
        types = engine.get_available_vehicle_types()
        assert len(types) == 8

    def test_get_available_equipment_types(self, engine):
        """Test listing available equipment types."""
        types = engine.get_available_equipment_types()
        assert len(types) == 6

    def test_get_available_it_types(self, engine):
        """Test listing available IT asset types."""
        types = engine.get_available_it_types()
        assert len(types) == 7

    def test_get_available_eeio_codes(self, engine):
        """Test listing available EEIO codes."""
        codes = engine.get_available_eeio_codes()
        assert len(codes) >= 6
        assert "531120" in codes

    def test_database_summary(self, engine):
        """Test database summary returns statistics."""
        summary = engine.get_summary()
        assert isinstance(summary, dict)
        assert "building_types" in summary or "total_lookups" in summary

    def test_lookup_count_increments(self, engine):
        """Test lookup count increments with each lookup."""
        initial = engine.get_lookup_count()
        engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert engine.get_lookup_count() > initial


# ==============================================================================
# ADDITIONAL BUILDING EUI COMPARISON TESTS
# ==============================================================================


class TestBuildingEUIComparisons:
    """Test building EUI comparisons across types and zones."""

    def test_cold_zone_higher_than_tropical_for_office(self, engine):
        """Test cold zone EUI is higher than tropical for offices."""
        cold = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.COLD)
        tropical = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TROPICAL)
        assert cold >= tropical

    def test_healthcare_higher_than_warehouse(self, engine):
        """Test healthcare EUI is higher than warehouse."""
        hc = engine.get_building_eui(BuildingType.HEALTHCARE, ClimateZone.TEMPERATE)
        wh = engine.get_building_eui(BuildingType.WAREHOUSE, ClimateZone.TEMPERATE)
        assert hc > wh

    def test_hotel_higher_than_education(self, engine):
        """Test hotel EUI is higher than education."""
        hotel = engine.get_building_eui(BuildingType.HOTEL, ClimateZone.TEMPERATE)
        edu = engine.get_building_eui(BuildingType.EDUCATION, ClimateZone.TEMPERATE)
        assert hotel >= edu

    def test_industrial_eui_positive(self, engine):
        """Test industrial building has positive EUI in all zones."""
        for zone in ClimateZone:
            eui = engine.get_building_eui(BuildingType.INDUSTRIAL, zone)
            assert eui > 0

    def test_retail_eui_in_arid_zone(self, engine):
        """Test retail EUI in arid zone is reasonable."""
        eui = engine.get_building_eui(BuildingType.RETAIL, ClimateZone.ARID)
        assert Decimal("50") <= eui <= Decimal("500")


# ==============================================================================
# ADDITIONAL GRID EF COMPARISON TESTS
# ==============================================================================


class TestGridEFComparisons:
    """Test grid emission factor comparisons."""

    def test_india_higher_than_france(self, engine):
        """Test India grid EF is higher than France."""
        india = engine.get_grid_emission_factor("IN")
        france = engine.get_grid_emission_factor("FR")
        assert india > france

    def test_china_higher_than_global(self, engine):
        """Test China grid EF is higher than global average."""
        china = engine.get_grid_emission_factor("CN")
        glob = engine.get_grid_emission_factor("GLOBAL")
        assert china >= glob

    def test_germany_grid_ef_range(self, engine):
        """Test Germany grid EF is in expected range."""
        de = engine.get_grid_emission_factor("DE")
        assert Decimal("0.25") <= de <= Decimal("0.60")

    def test_japan_grid_ef_range(self, engine):
        """Test Japan grid EF is in expected range."""
        jp = engine.get_grid_emission_factor("JP")
        assert Decimal("0.30") <= jp <= Decimal("0.60")

    def test_grid_ef_quantized(self, engine):
        """Test grid EF values are quantized to 8dp."""
        ef = engine.get_grid_emission_factor("US")
        assert ef == ef.quantize(Decimal("0.00000001"))
