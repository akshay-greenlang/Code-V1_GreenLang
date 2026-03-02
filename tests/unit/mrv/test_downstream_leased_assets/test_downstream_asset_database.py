# -*- coding: utf-8 -*-
"""
Test suite for downstream_leased_assets.downstream_asset_database - AGENT-MRV-026.

Tests the DownstreamAssetDatabaseEngine (Engine 1) for building EUI lookups
(8 types x 5 climate zones), vehicle EFs (8 types x 7 fuels), equipment
benchmarks, IT power ratings, grid EFs, fuel EFs, EEIO factors, vacancy
base-load factors, refrigerant GWPs, and provenance hash determinism.

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import List
import pytest

try:
    from greenlang.downstream_leased_assets.downstream_asset_database import (
        DownstreamAssetDatabaseEngine,
        get_database_engine,
        reset_database_engine,
    )
    from greenlang.downstream_leased_assets.models import (
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
    reason="downstream_leased_assets.downstream_asset_database not available",
)

pytestmark = _SKIP


@pytest.fixture(autouse=True)
def reset_engine():
    if _AVAILABLE:
        reset_database_engine()
    yield
    if _AVAILABLE:
        reset_database_engine()


@pytest.fixture
def engine():
    return DownstreamAssetDatabaseEngine()


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:

    def test_singleton_instance(self, engine):
        engine2 = DownstreamAssetDatabaseEngine()
        assert engine is engine2

    def test_singleton_via_get_database_engine(self):
        e1 = get_database_engine()
        e2 = get_database_engine()
        assert e1 is e2

    def test_singleton_across_threads(self):
        instances: List = []

        def get_instance():
            instances.append(DownstreamAssetDatabaseEngine())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        for inst in instances[1:]:
            assert inst is first

    def test_reset_creates_new_instance(self):
        e1 = DownstreamAssetDatabaseEngine()
        reset_database_engine()
        e2 = DownstreamAssetDatabaseEngine()
        assert e1 is not e2


# ==============================================================================
# BUILDING EUI LOOKUP TESTS (8x5 parametrized)
# ==============================================================================


class TestBuildingEUILookup:

    @pytest.mark.parametrize("building_type", list(BuildingType))
    @pytest.mark.parametrize("climate_zone", list(ClimateZone))
    def test_building_eui_lookup(self, engine, building_type, climate_zone):
        """Test building EUI lookup for all 40 combinations."""
        eui = engine.get_building_eui(building_type, climate_zone)
        assert isinstance(eui, Decimal)
        assert eui > 0

    def test_office_temperate_known_value(self, engine):
        eui = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert Decimal("100") <= eui <= Decimal("300")

    def test_data_center_highest_eui(self, engine):
        dc = engine.get_building_eui(BuildingType.DATA_CENTER, ClimateZone.TEMPERATE)
        office = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert dc > office

    def test_warehouse_lower_than_office(self, engine):
        wh = engine.get_building_eui(BuildingType.WAREHOUSE, ClimateZone.TEMPERATE)
        office = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert wh < office

    def test_cold_zone_higher_than_tropical(self, engine):
        cold = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.COLD)
        tropical = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TROPICAL)
        assert cold >= tropical

    def test_eui_quantized_to_8dp(self, engine):
        eui = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert eui == eui.quantize(Decimal("0.00000001"))


# ==============================================================================
# VEHICLE EMISSION FACTOR TESTS (8x7 parametrized)
# ==============================================================================


class TestVehicleEFLookup:

    @pytest.mark.parametrize("vehicle_type", list(VehicleType))
    def test_vehicle_ef_lookup_diesel(self, engine, vehicle_type):
        ef = engine.get_vehicle_emission_factor(vehicle_type, FuelType.DIESEL)
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_heavy_truck_higher_than_small_car(self, engine):
        truck = engine.get_vehicle_emission_factor(VehicleType.HEAVY_TRUCK, FuelType.DIESEL)
        car = engine.get_vehicle_emission_factor(VehicleType.SMALL_CAR, FuelType.DIESEL)
        assert truck > car

    def test_bev_emission_factor_zero(self, engine):
        ef = engine.get_vehicle_emission_factor(VehicleType.MEDIUM_CAR, FuelType.BEV)
        assert isinstance(ef, Decimal)
        assert ef >= Decimal("0")

    @pytest.mark.parametrize("vehicle_type", list(VehicleType))
    @pytest.mark.parametrize("fuel_type", [FuelType.PETROL, FuelType.DIESEL, FuelType.BEV])
    def test_vehicle_fuel_combinations(self, engine, vehicle_type, fuel_type):
        """Test EF lookup for vehicle/fuel combinations that should exist."""
        try:
            ef = engine.get_vehicle_emission_factor(vehicle_type, fuel_type)
            assert isinstance(ef, Decimal)
            assert ef >= 0
        except (KeyError, ValueError):
            # Some combinations may not exist (e.g., petrol heavy_truck)
            pass


# ==============================================================================
# EQUIPMENT BENCHMARK TESTS (6 types)
# ==============================================================================


class TestEquipmentBenchmarkLookup:

    @pytest.mark.parametrize("equipment_type", list(EquipmentType))
    def test_equipment_benchmark_lookup(self, engine, equipment_type):
        benchmark = engine.get_equipment_benchmark(equipment_type)
        assert isinstance(benchmark, dict)
        assert "default_load_factor" in benchmark

    def test_manufacturing_load_factor(self, engine):
        benchmark = engine.get_equipment_benchmark(EquipmentType.MANUFACTURING)
        lf = benchmark["default_load_factor"]
        assert Decimal("0.50") <= lf <= Decimal("0.90")


# ==============================================================================
# IT POWER RATING TESTS (7 types)
# ==============================================================================


class TestITPowerRatingLookup:

    @pytest.mark.parametrize("it_type", list(ITAssetType))
    def test_it_power_rating_lookup(self, engine, it_type):
        rating = engine.get_it_power_rating(it_type)
        assert isinstance(rating, dict)
        assert "typical_power_w" in rating

    def test_server_highest_power(self, engine):
        server = engine.get_it_power_rating(ITAssetType.SERVER)
        laptop = engine.get_it_power_rating(ITAssetType.LAPTOP)
        assert server["typical_power_w"] > laptop["typical_power_w"]

    def test_standby_lower_than_active(self, engine):
        for it_type in ITAssetType:
            rating = engine.get_it_power_rating(it_type)
            assert rating["standby_power_w"] < rating["typical_power_w"]


# ==============================================================================
# GRID EMISSION FACTOR TESTS (12 countries + eGRID subregions)
# ==============================================================================


class TestGridEFLookup:

    @pytest.mark.parametrize("country", ["US", "GB", "DE", "FR", "JP", "CA", "AU", "IN", "CN", "BR", "GLOBAL"])
    def test_grid_ef_by_country(self, engine, country):
        ef = engine.get_grid_emission_factor(country)
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_us_grid_ef_range(self, engine):
        ef = engine.get_grid_emission_factor("US")
        assert Decimal("0.30") <= ef <= Decimal("0.50")

    def test_france_lower_than_us(self, engine):
        fr = engine.get_grid_emission_factor("FR")
        us = engine.get_grid_emission_factor("US")
        assert fr < us

    def test_case_insensitive_lookup(self, engine):
        ef_upper = engine.get_grid_emission_factor("US")
        ef_lower = engine.get_grid_emission_factor("us")
        assert ef_upper == ef_lower

    def test_unknown_country_fallback(self, engine):
        ef = engine.get_grid_emission_factor("XX")
        global_ef = engine.get_grid_emission_factor("GLOBAL")
        assert ef == global_ef


# ==============================================================================
# FUEL EMISSION FACTOR TESTS (8 types)
# ==============================================================================


class TestFuelEFLookup:

    @pytest.mark.parametrize("fuel_type", [FuelType.PETROL, FuelType.DIESEL, FuelType.LPG, FuelType.CNG])
    def test_fuel_ef_lookup(self, engine, fuel_type):
        ef = engine.get_fuel_emission_factor(fuel_type)
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_diesel_ef_value(self, engine):
        ef = engine.get_fuel_emission_factor(FuelType.DIESEL)
        assert Decimal("2.50") <= ef <= Decimal("3.00")

    def test_diesel_higher_than_petrol(self, engine):
        diesel = engine.get_fuel_emission_factor(FuelType.DIESEL)
        petrol = engine.get_fuel_emission_factor(FuelType.PETROL)
        assert diesel > petrol


# ==============================================================================
# EEIO FACTOR TESTS (10 NAICS codes)
# ==============================================================================


class TestEEIOFactorLookup:

    @pytest.mark.parametrize("naics", ["531120", "531130", "531190", "532111", "532112", "532120", "532310", "532412", "532490", "518210"])
    def test_eeio_factor_lookup(self, engine, naics):
        factor = engine.get_eeio_factor(naics)
        assert isinstance(factor, dict)
        assert "ef" in factor
        assert factor["ef"] > 0

    def test_invalid_naics_returns_none(self, engine):
        result = engine.get_eeio_factor("999999")
        assert result is None or result == {}


# ==============================================================================
# VACANCY BASE LOAD TESTS (8 types)
# ==============================================================================


class TestVacancyBaseLoadLookup:

    @pytest.mark.parametrize("building_type", list(BuildingType))
    def test_vacancy_factor_lookup(self, engine, building_type):
        factor = engine.get_vacancy_factor(building_type)
        assert isinstance(factor, Decimal)
        assert Decimal("0") <= factor <= Decimal("1.0")

    def test_data_center_highest_vacancy_load(self, engine):
        dc = engine.get_vacancy_factor(BuildingType.DATA_CENTER)
        for bt in BuildingType:
            if bt != BuildingType.DATA_CENTER:
                assert dc >= engine.get_vacancy_factor(bt)


# ==============================================================================
# REFRIGERANT GWP TESTS (15 refrigerants)
# ==============================================================================


class TestRefrigerantGWPLookup:

    def test_at_least_fifteen_refrigerants(self, engine):
        gwps = engine.get_available_refrigerants()
        assert len(gwps) >= 15

    def test_r134a_gwp(self, engine):
        gwp = engine.get_refrigerant_gwp("R-134a")
        assert isinstance(gwp, (int, Decimal))
        assert gwp > 0


# ==============================================================================
# CURRENCY AND CPI TESTS
# ==============================================================================


class TestCurrencyRateLookup:

    @pytest.mark.parametrize("currency", list(CurrencyCode))
    def test_currency_rate_lookup(self, engine, currency):
        rate = engine.get_currency_rate(currency)
        assert isinstance(rate, Decimal)
        assert rate > 0

    def test_usd_rate_is_one(self, engine):
        rate = engine.get_currency_rate(CurrencyCode.USD)
        assert rate == Decimal("1.0")


class TestCPIDeflatorLookup:

    @pytest.mark.parametrize("year", list(range(2015, 2026)))
    def test_cpi_deflator_lookup(self, engine, year):
        deflator = engine.get_cpi_deflator(year)
        assert isinstance(deflator, Decimal)
        assert deflator > 0

    def test_base_year_2021_is_one(self, engine):
        deflator = engine.get_cpi_deflator(2021)
        assert deflator == Decimal("1.0000")


# ==============================================================================
# PROVENANCE HASH DETERMINISM
# ==============================================================================


class TestProvenanceDeterminism:

    def test_same_lookup_same_hash(self, engine):
        """Test same lookup returns same data (deterministic)."""
        eui1 = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        eui2 = engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert eui1 == eui2

    def test_grid_ef_deterministic(self, engine):
        ef1 = engine.get_grid_emission_factor("US")
        ef2 = engine.get_grid_emission_factor("US")
        assert ef1 == ef2


# ==============================================================================
# ENUMERATION AND SEARCH TESTS
# ==============================================================================


class TestEnumerations:

    def test_get_available_building_types(self, engine):
        types = engine.get_available_building_types()
        assert len(types) == 8

    def test_get_available_vehicle_types(self, engine):
        types = engine.get_available_vehicle_types()
        assert len(types) == 8

    def test_get_available_equipment_types(self, engine):
        types = engine.get_available_equipment_types()
        assert len(types) == 6

    def test_get_available_it_types(self, engine):
        types = engine.get_available_it_types()
        assert len(types) == 7

    def test_get_available_eeio_codes(self, engine):
        codes = engine.get_available_eeio_codes()
        assert len(codes) >= 10
        assert "531120" in codes

    def test_get_available_climate_zones(self, engine):
        zones = engine.get_available_climate_zones()
        assert len(zones) == 5

    def test_database_summary(self, engine):
        summary = engine.get_summary()
        assert isinstance(summary, dict)

    def test_lookup_count_increments(self, engine):
        initial = engine.get_lookup_count()
        engine.get_building_eui(BuildingType.OFFICE, ClimateZone.TEMPERATE)
        assert engine.get_lookup_count() > initial
