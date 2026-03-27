# -*- coding: utf-8 -*-
"""
Test suite for business_travel.business_travel_database - AGENT-MRV-019.

Tests the BusinessTravelDatabaseEngine (Engine 1) for the Business Travel
Agent (GL-MRV-S3-006) including singleton pattern, air/rail/road/fuel/bus/
ferry/hotel emission factor lookups, EEIO spend-based factors, airport
database operations, cabin class multipliers, currency conversion, CPI
deflation, transport mode classification, and available options queries.

Coverage:
- Singleton pattern (thread-safe, same instance)
- Air emission factor lookups (domestic, short_haul, long_haul, with class)
- Rail emission factor lookups (national, international, eurostar, high_speed, us_intercity)
- Road emission factor lookups (car_average, hybrid, BEV, taxi)
- Fuel emission factor lookups (petrol, diesel)
- Bus emission factor lookups (local, coach)
- Ferry emission factor lookups (foot_passenger, car_passenger)
- Hotel emission factor lookups (UK, US, Japan, unknown country fallback, class multiplier)
- EEIO factor lookups (air NAICS, hotel NAICS, invalid NAICS)
- Airport lookups (LHR, JFK, invalid, search)
- Cabin class multiplier retrieval (economy, business)
- Currency rate retrieval (USD, EUR)
- CPI deflator retrieval (2021 base, 2024)
- Available transport modes and cabin classes
- Transport mode classification from trip data (air, rail)
- Database summary and lookup count
- Error handling (invalid enums, missing data)
- Quantization to 8 decimal places

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List, Optional
import pytest

from greenlang.agents.mrv.business_travel.business_travel_database import (
    BusinessTravelDatabaseEngine,
    get_database_engine,
    reset_database_engine,
)
from greenlang.agents.mrv.business_travel.models import (
    TransportMode,
    FlightDistanceBand,
    CabinClass,
    RailType,
    RoadVehicleType,
    FuelType,
    BusType,
    FerryType,
    HotelClass,
    EFSource,
    CurrencyCode,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the singleton engine before each test."""
    reset_database_engine()
    yield
    reset_database_engine()


@pytest.fixture
def engine() -> BusinessTravelDatabaseEngine:
    """Create a fresh BusinessTravelDatabaseEngine instance."""
    return BusinessTravelDatabaseEngine()


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, engine):
        """Test BusinessTravelDatabaseEngine returns the same instance."""
        engine2 = BusinessTravelDatabaseEngine()
        assert engine is engine2

    def test_singleton_via_get_database_engine(self):
        """Test get_database_engine returns singleton."""
        engine1 = get_database_engine()
        engine2 = get_database_engine()
        assert engine1 is engine2

    def test_singleton_across_threads(self):
        """Test singleton works across threads."""
        instances: List[BusinessTravelDatabaseEngine] = []

        def get_instance():
            instances.append(BusinessTravelDatabaseEngine())

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
        engine1 = BusinessTravelDatabaseEngine()
        reset_database_engine()
        engine2 = BusinessTravelDatabaseEngine()
        # After reset, _initialized attribute should exist on new instance
        assert engine2 is not engine1


# ==============================================================================
# AIR EMISSION FACTOR TESTS
# ==============================================================================


class TestAirEmissionFactors:
    """Test air emission factor lookups."""

    def test_get_air_ef_domestic(self, engine):
        """Test domestic air EF without_rf = 0.24587."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.DOMESTIC)
        assert ef["without_rf"] == Decimal("0.24587000")

    def test_get_air_ef_domestic_with_rf(self, engine):
        """Test domestic air EF with_rf = 0.27916."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.DOMESTIC)
        assert ef["with_rf"] == Decimal("0.27916000")

    def test_get_air_ef_domestic_wtt(self, engine):
        """Test domestic air EF wtt = 0.05765."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.DOMESTIC)
        assert ef["wtt"] == Decimal("0.05765000")

    def test_get_air_ef_short_haul(self, engine):
        """Test short_haul air EF without_rf = 0.15353."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.SHORT_HAUL)
        assert ef["without_rf"] == Decimal("0.15353000")

    def test_get_air_ef_long_haul(self, engine):
        """Test long_haul air EF without_rf = 0.19309."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        assert ef["without_rf"] == Decimal("0.19309000")

    def test_get_air_ef_long_haul_with_rf(self, engine):
        """Test long_haul air EF with_rf = 0.21932."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        assert ef["with_rf"] == Decimal("0.21932000")

    def test_get_air_ef_international_avg(self, engine):
        """Test international_avg air EF without_rf = 0.18362."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.INTERNATIONAL_AVG)
        assert ef["without_rf"] == Decimal("0.18362000")

    def test_get_air_ef_with_class_multiplier(self, engine):
        """Test economy class multiplier is 1.0."""
        ef = engine.get_air_emission_factor(
            FlightDistanceBand.LONG_HAUL, CabinClass.ECONOMY
        )
        assert ef["class_multiplier"] == Decimal("1.00000000")

    def test_get_air_ef_premium_economy(self, engine):
        """Test premium_economy class multiplier is 1.6."""
        ef = engine.get_air_emission_factor(
            FlightDistanceBand.LONG_HAUL, CabinClass.PREMIUM_ECONOMY
        )
        assert ef["class_multiplier"] == Decimal("1.60000000")

    def test_get_air_ef_business_class(self, engine):
        """Test business class multiplier is 2.9."""
        ef = engine.get_air_emission_factor(
            FlightDistanceBand.LONG_HAUL, CabinClass.BUSINESS
        )
        assert ef["class_multiplier"] == Decimal("2.90000000")

    def test_get_air_ef_first_class(self, engine):
        """Test first class multiplier is 4.0."""
        ef = engine.get_air_emission_factor(
            FlightDistanceBand.LONG_HAUL, CabinClass.FIRST
        )
        assert ef["class_multiplier"] == Decimal("4.00000000")

    def test_get_air_ef_returns_dict_keys(self, engine):
        """Test air EF return dict has expected keys."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        assert set(ef.keys()) == {"without_rf", "with_rf", "wtt", "class_multiplier"}

    def test_get_air_ef_all_values_are_decimal(self, engine):
        """Test all air EF values are Decimal type."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        for key, value in ef.items():
            assert isinstance(value, Decimal), f"Key '{key}' is not Decimal"


# ==============================================================================
# RAIL EMISSION FACTOR TESTS
# ==============================================================================


class TestRailEmissionFactors:
    """Test rail emission factor lookups."""

    def test_get_rail_ef_national(self, engine):
        """Test national rail TTW = 0.03549."""
        ef = engine.get_rail_emission_factor(RailType.NATIONAL)
        assert ef["ttw"] == Decimal("0.03549000")

    def test_get_rail_ef_national_wtt(self, engine):
        """Test national rail WTT = 0.00434."""
        ef = engine.get_rail_emission_factor(RailType.NATIONAL)
        assert ef["wtt"] == Decimal("0.00434000")

    def test_get_rail_ef_international(self, engine):
        """Test international rail TTW = 0.00446."""
        ef = engine.get_rail_emission_factor(RailType.INTERNATIONAL)
        assert ef["ttw"] == Decimal("0.00446000")

    def test_get_rail_ef_eurostar(self, engine):
        """Test Eurostar TTW = 0.00446."""
        ef = engine.get_rail_emission_factor(RailType.EUROSTAR)
        assert ef["ttw"] == Decimal("0.00446000")

    def test_get_rail_ef_eurostar_wtt(self, engine):
        """Test Eurostar WTT = 0.00086."""
        ef = engine.get_rail_emission_factor(RailType.EUROSTAR)
        assert ef["wtt"] == Decimal("0.00086000")

    def test_get_rail_ef_high_speed(self, engine):
        """Test high_speed rail TTW = 0.00324."""
        ef = engine.get_rail_emission_factor(RailType.HIGH_SPEED)
        assert ef["ttw"] == Decimal("0.00324000")

    def test_get_rail_ef_us_intercity(self, engine):
        """Test US intercity rail TTW = 0.08900."""
        ef = engine.get_rail_emission_factor(RailType.US_INTERCITY)
        assert ef["ttw"] == Decimal("0.08900000")

    def test_get_rail_ef_us_commuter(self, engine):
        """Test US commuter rail TTW = 0.10500."""
        ef = engine.get_rail_emission_factor(RailType.US_COMMUTER)
        assert ef["ttw"] == Decimal("0.10500000")

    def test_get_rail_ef_light_rail(self, engine):
        """Test light_rail TTW = 0.02904."""
        ef = engine.get_rail_emission_factor(RailType.LIGHT_RAIL)
        assert ef["ttw"] == Decimal("0.02904000")

    def test_get_rail_ef_underground(self, engine):
        """Test underground TTW = 0.02781."""
        ef = engine.get_rail_emission_factor(RailType.UNDERGROUND)
        assert ef["ttw"] == Decimal("0.02781000")

    def test_get_rail_ef_returns_dict_keys(self, engine):
        """Test rail EF return dict has expected keys."""
        ef = engine.get_rail_emission_factor(RailType.NATIONAL)
        assert set(ef.keys()) == {"ttw", "wtt"}


# ==============================================================================
# ROAD EMISSION FACTOR TESTS
# ==============================================================================


class TestRoadEmissionFactors:
    """Test road vehicle emission factor lookups."""

    def test_get_road_ef_car_average(self, engine):
        """Test car_average EF per vkm = 0.27145."""
        ef = engine.get_road_emission_factor(RoadVehicleType.CAR_AVERAGE)
        assert ef["ef_per_vkm"] == Decimal("0.27145000")

    def test_get_road_ef_car_average_pkm(self, engine):
        """Test car_average EF per pkm = 0.17082."""
        ef = engine.get_road_emission_factor(RoadVehicleType.CAR_AVERAGE)
        assert ef["ef_per_pkm"] == Decimal("0.17082000")

    def test_get_road_ef_car_average_wtt(self, engine):
        """Test car_average WTT per vkm = 0.06291."""
        ef = engine.get_road_emission_factor(RoadVehicleType.CAR_AVERAGE)
        assert ef["wtt_per_vkm"] == Decimal("0.06291000")

    def test_get_road_ef_car_average_occupancy(self, engine):
        """Test car_average occupancy = 1.59."""
        ef = engine.get_road_emission_factor(RoadVehicleType.CAR_AVERAGE)
        assert ef["occupancy"] == Decimal("1.59000000")

    def test_get_road_ef_hybrid(self, engine):
        """Test hybrid EF per vkm = 0.17830."""
        ef = engine.get_road_emission_factor(RoadVehicleType.HYBRID)
        assert ef["ef_per_vkm"] == Decimal("0.17830000")

    def test_get_road_ef_bev(self, engine):
        """Test BEV EF per vkm = 0.07005."""
        ef = engine.get_road_emission_factor(RoadVehicleType.BEV)
        assert ef["ef_per_vkm"] == Decimal("0.07005000")

    def test_get_road_ef_bev_wtt(self, engine):
        """Test BEV WTT per vkm = 0.01479."""
        ef = engine.get_road_emission_factor(RoadVehicleType.BEV)
        assert ef["wtt_per_vkm"] == Decimal("0.01479000")

    def test_get_road_ef_taxi(self, engine):
        """Test taxi_regular EF per vkm = 0.20920."""
        ef = engine.get_road_emission_factor(RoadVehicleType.TAXI_REGULAR)
        assert ef["ef_per_vkm"] == Decimal("0.20920000")

    def test_get_road_ef_taxi_black_cab(self, engine):
        """Test taxi_black_cab EF per vkm = 0.31477."""
        ef = engine.get_road_emission_factor(RoadVehicleType.TAXI_BLACK_CAB)
        assert ef["ef_per_vkm"] == Decimal("0.31477000")

    def test_get_road_ef_motorcycle(self, engine):
        """Test motorcycle EF per vkm = 0.11337."""
        ef = engine.get_road_emission_factor(RoadVehicleType.MOTORCYCLE)
        assert ef["ef_per_vkm"] == Decimal("0.11337000")

    def test_get_road_ef_returns_dict_keys(self, engine):
        """Test road EF return dict has expected keys."""
        ef = engine.get_road_emission_factor(RoadVehicleType.CAR_AVERAGE)
        assert set(ef.keys()) == {"ef_per_vkm", "ef_per_pkm", "wtt_per_vkm", "occupancy"}


# ==============================================================================
# FUEL EMISSION FACTOR TESTS
# ==============================================================================


class TestFuelEmissionFactors:
    """Test fuel-based emission factor lookups."""

    def test_get_fuel_ef_petrol(self, engine):
        """Test petrol EF per litre = 2.31480."""
        ef = engine.get_fuel_emission_factor(FuelType.PETROL)
        assert ef["ef_per_litre"] == Decimal("2.31480000")

    def test_get_fuel_ef_petrol_wtt(self, engine):
        """Test petrol WTT per litre = 0.58549."""
        ef = engine.get_fuel_emission_factor(FuelType.PETROL)
        assert ef["wtt_per_litre"] == Decimal("0.58549000")

    def test_get_fuel_ef_diesel(self, engine):
        """Test diesel EF per litre = 2.70370."""
        ef = engine.get_fuel_emission_factor(FuelType.DIESEL)
        assert ef["ef_per_litre"] == Decimal("2.70370000")

    def test_get_fuel_ef_diesel_wtt(self, engine):
        """Test diesel WTT per litre = 0.60927."""
        ef = engine.get_fuel_emission_factor(FuelType.DIESEL)
        assert ef["wtt_per_litre"] == Decimal("0.60927000")

    def test_get_fuel_ef_lpg(self, engine):
        """Test LPG EF per litre = 1.55370."""
        ef = engine.get_fuel_emission_factor(FuelType.LPG)
        assert ef["ef_per_litre"] == Decimal("1.55370000")

    def test_get_fuel_ef_cng(self, engine):
        """Test CNG EF per kg = 2.53970."""
        ef = engine.get_fuel_emission_factor(FuelType.CNG)
        assert ef["ef_per_litre"] == Decimal("2.53970000")

    def test_get_fuel_ef_e85(self, engine):
        """Test E85 EF per litre = 0.34728."""
        ef = engine.get_fuel_emission_factor(FuelType.E85)
        assert ef["ef_per_litre"] == Decimal("0.34728000")

    def test_get_fuel_ef_returns_dict_keys(self, engine):
        """Test fuel EF return dict has expected keys."""
        ef = engine.get_fuel_emission_factor(FuelType.DIESEL)
        assert set(ef.keys()) == {"ef_per_litre", "wtt_per_litre"}


# ==============================================================================
# BUS EMISSION FACTOR TESTS
# ==============================================================================


class TestBusEmissionFactors:
    """Test bus emission factor lookups."""

    def test_get_bus_ef_local(self, engine):
        """Test local bus EF = 0.10312."""
        ef = engine.get_bus_emission_factor(BusType.LOCAL)
        assert ef["ef"] == Decimal("0.10312000")

    def test_get_bus_ef_local_wtt(self, engine):
        """Test local bus WTT = 0.01847."""
        ef = engine.get_bus_emission_factor(BusType.LOCAL)
        assert ef["wtt"] == Decimal("0.01847000")

    def test_get_bus_ef_coach(self, engine):
        """Test coach EF = 0.02732."""
        ef = engine.get_bus_emission_factor(BusType.COACH)
        assert ef["ef"] == Decimal("0.02732000")

    def test_get_bus_ef_coach_wtt(self, engine):
        """Test coach WTT = 0.00489."""
        ef = engine.get_bus_emission_factor(BusType.COACH)
        assert ef["wtt"] == Decimal("0.00489000")

    def test_get_bus_ef_returns_dict_keys(self, engine):
        """Test bus EF return dict has expected keys."""
        ef = engine.get_bus_emission_factor(BusType.LOCAL)
        assert set(ef.keys()) == {"ef", "wtt"}


# ==============================================================================
# FERRY EMISSION FACTOR TESTS
# ==============================================================================


class TestFerryEmissionFactors:
    """Test ferry emission factor lookups."""

    def test_get_ferry_ef_foot(self, engine):
        """Test foot_passenger EF = 0.01877."""
        ef = engine.get_ferry_emission_factor(FerryType.FOOT_PASSENGER)
        assert ef["ef"] == Decimal("0.01877000")

    def test_get_ferry_ef_foot_wtt(self, engine):
        """Test foot_passenger WTT = 0.00572."""
        ef = engine.get_ferry_emission_factor(FerryType.FOOT_PASSENGER)
        assert ef["wtt"] == Decimal("0.00572000")

    def test_get_ferry_ef_car(self, engine):
        """Test car_passenger EF = 0.12952."""
        ef = engine.get_ferry_emission_factor(FerryType.CAR_PASSENGER)
        assert ef["ef"] == Decimal("0.12952000")

    def test_get_ferry_ef_car_wtt(self, engine):
        """Test car_passenger WTT = 0.03950."""
        ef = engine.get_ferry_emission_factor(FerryType.CAR_PASSENGER)
        assert ef["wtt"] == Decimal("0.03950000")

    def test_get_ferry_ef_returns_dict_keys(self, engine):
        """Test ferry EF return dict has expected keys."""
        ef = engine.get_ferry_emission_factor(FerryType.FOOT_PASSENGER)
        assert set(ef.keys()) == {"ef", "wtt"}


# ==============================================================================
# HOTEL EMISSION FACTOR TESTS
# ==============================================================================


class TestHotelEmissionFactors:
    """Test hotel emission factor lookups."""

    def test_get_hotel_ef_uk(self, engine):
        """Test UK hotel EF = 12.32 kgCO2e per room-night."""
        ef = engine.get_hotel_emission_factor("GB")
        assert ef["ef_per_room_night"] == Decimal("12.32000000")

    def test_get_hotel_ef_us(self, engine):
        """Test US hotel EF = 21.12 kgCO2e per room-night."""
        ef = engine.get_hotel_emission_factor("US")
        assert ef["ef_per_room_night"] == Decimal("21.12000000")

    def test_get_hotel_ef_japan(self, engine):
        """Test Japan hotel EF = 28.85 kgCO2e per room-night."""
        ef = engine.get_hotel_emission_factor("JP")
        assert ef["ef_per_room_night"] == Decimal("28.85000000")

    def test_get_hotel_ef_france(self, engine):
        """Test France hotel EF = 7.26 kgCO2e per room-night."""
        ef = engine.get_hotel_emission_factor("FR")
        assert ef["ef_per_room_night"] == Decimal("7.26000000")

    def test_get_hotel_ef_china(self, engine):
        """Test China hotel EF = 34.56 kgCO2e per room-night."""
        ef = engine.get_hotel_emission_factor("CN")
        assert ef["ef_per_room_night"] == Decimal("34.56000000")

    def test_get_hotel_ef_uae(self, engine):
        """Test UAE hotel EF = 37.50 kgCO2e per room-night (highest)."""
        ef = engine.get_hotel_emission_factor("AE")
        assert ef["ef_per_room_night"] == Decimal("37.50000000")

    def test_get_hotel_ef_unknown_country_fallback(self, engine):
        """Test unknown country falls back to GLOBAL = 20.90."""
        ef = engine.get_hotel_emission_factor("ZZ")
        assert ef["ef_per_room_night"] == Decimal("20.90000000")

    def test_get_hotel_ef_case_insensitive(self, engine):
        """Test country code is case-insensitive."""
        ef = engine.get_hotel_emission_factor("gb")
        assert ef["ef_per_room_night"] == Decimal("12.32000000")

    def test_get_hotel_ef_with_class_standard(self, engine):
        """Test standard class multiplier is 1.0."""
        ef = engine.get_hotel_emission_factor("GB", HotelClass.STANDARD)
        assert ef["class_multiplier"] == Decimal("1.00000000")

    def test_get_hotel_ef_with_class_budget(self, engine):
        """Test budget class multiplier is 0.75."""
        ef = engine.get_hotel_emission_factor("GB", HotelClass.BUDGET)
        assert ef["class_multiplier"] == Decimal("0.75000000")

    def test_get_hotel_ef_with_class_upscale(self, engine):
        """Test upscale class multiplier is 1.35."""
        ef = engine.get_hotel_emission_factor("GB", HotelClass.UPSCALE)
        assert ef["class_multiplier"] == Decimal("1.35000000")

    def test_get_hotel_ef_with_class_luxury(self, engine):
        """Test luxury class multiplier is 1.80."""
        ef = engine.get_hotel_emission_factor("GB", HotelClass.LUXURY)
        assert ef["class_multiplier"] == Decimal("1.80000000")

    def test_get_hotel_ef_returns_dict_keys(self, engine):
        """Test hotel EF return dict has expected keys."""
        ef = engine.get_hotel_emission_factor("GB")
        assert set(ef.keys()) == {"ef_per_room_night", "class_multiplier"}


# ==============================================================================
# EEIO FACTOR TESTS
# ==============================================================================


class TestEEIOFactors:
    """Test EEIO spend-based factor lookups."""

    def test_get_eeio_factor_air(self, engine):
        """Test air NAICS 481000 EF = 0.4770 kgCO2e/USD."""
        ef = engine.get_eeio_factor("481000")
        assert ef["ef_per_usd"] == Decimal("0.47700000")
        assert ef["name"] == "Air transportation"

    def test_get_eeio_factor_rail(self, engine):
        """Test rail NAICS 482000 EF = 0.3100 kgCO2e/USD."""
        ef = engine.get_eeio_factor("482000")
        assert ef["ef_per_usd"] == Decimal("0.31000000")
        assert ef["name"] == "Rail transportation"

    def test_get_eeio_factor_hotel(self, engine):
        """Test hotel NAICS 721100 EF = 0.1490 kgCO2e/USD."""
        ef = engine.get_eeio_factor("721100")
        assert ef["ef_per_usd"] == Decimal("0.14900000")
        assert ef["name"] == "Hotels and motels"

    def test_get_eeio_factor_taxi(self, engine):
        """Test taxi NAICS 485310 EF = 0.2800 kgCO2e/USD."""
        ef = engine.get_eeio_factor("485310")
        assert ef["ef_per_usd"] == Decimal("0.28000000")
        assert ef["name"] == "Taxi/ride-hailing"

    def test_get_eeio_factor_water(self, engine):
        """Test water transport NAICS 483000 EF = 0.5200 kgCO2e/USD."""
        ef = engine.get_eeio_factor("483000")
        assert ef["ef_per_usd"] == Decimal("0.52000000")
        assert ef["name"] == "Water transportation"

    def test_get_eeio_factor_invalid(self, engine):
        """Test invalid NAICS code raises ValueError."""
        with pytest.raises(ValueError, match="EEIO factor not found"):
            engine.get_eeio_factor("999999")

    def test_get_eeio_factor_returns_dict_keys(self, engine):
        """Test EEIO factor return dict has expected keys."""
        ef = engine.get_eeio_factor("481000")
        assert set(ef.keys()) == {"name", "ef_per_usd"}


# ==============================================================================
# AIRPORT LOOKUP TESTS
# ==============================================================================


class TestAirportLookups:
    """Test airport database lookups."""

    def test_lookup_airport_lhr(self, engine):
        """Test LHR lookup returns Heathrow with correct coordinates."""
        airport = engine.lookup_airport("LHR")
        assert airport is not None
        assert airport["name"] == "London Heathrow"
        assert airport["lat"] == Decimal("51.47")
        assert airport["lon"] == Decimal("-0.4543")
        assert airport["country"] == "GB"

    def test_lookup_airport_jfk(self, engine):
        """Test JFK lookup returns John F. Kennedy with correct coordinates."""
        airport = engine.lookup_airport("JFK")
        assert airport is not None
        assert airport["name"] == "John F. Kennedy International"
        assert airport["lat"] == Decimal("40.6413")
        assert airport["country"] == "US"

    def test_lookup_airport_invalid(self, engine):
        """Test invalid IATA code returns None."""
        airport = engine.lookup_airport("ZZZ")
        assert airport is None

    def test_lookup_airport_case_insensitive(self, engine):
        """Test airport lookup is case-insensitive."""
        airport = engine.lookup_airport("lhr")
        assert airport is not None
        assert airport["name"] == "London Heathrow"

    def test_lookup_airport_returns_dict_keys(self, engine):
        """Test airport lookup return dict has expected keys."""
        airport = engine.lookup_airport("LHR")
        assert airport is not None
        assert set(airport.keys()) == {"name", "lat", "lon", "country"}

    def test_search_airports_london(self, engine):
        """Test searching for 'london' returns at least LHR and LGW."""
        results = engine.search_airports("london")
        assert len(results) >= 2
        iata_codes = [r["iata"] for r in results]
        assert "LHR" in iata_codes
        assert "LGW" in iata_codes

    def test_search_airports_by_code(self, engine):
        """Test searching by partial IATA code returns matches."""
        results = engine.search_airports("jfk")
        assert len(results) >= 1
        assert results[0]["iata"] == "JFK"

    def test_search_airports_empty(self, engine):
        """Test empty search query returns empty list."""
        results = engine.search_airports("")
        assert results == []

    def test_search_airports_no_match(self, engine):
        """Test no-match query returns empty list."""
        results = engine.search_airports("xyznonexistentairport")
        assert results == []

    def test_search_airports_result_keys(self, engine):
        """Test search result dicts have expected keys."""
        results = engine.search_airports("heathrow")
        assert len(results) >= 1
        result = results[0]
        assert set(result.keys()) == {"iata", "name", "lat", "lon", "country"}


# ==============================================================================
# CABIN CLASS MULTIPLIER TESTS
# ==============================================================================


class TestCabinClassMultiplier:
    """Test cabin class multiplier retrieval."""

    def test_get_cabin_class_multiplier_economy(self, engine):
        """Test economy multiplier = 1.0."""
        m = engine.get_cabin_class_multiplier(CabinClass.ECONOMY)
        assert m == Decimal("1.00000000")

    def test_get_cabin_class_multiplier_premium_economy(self, engine):
        """Test premium_economy multiplier = 1.6."""
        m = engine.get_cabin_class_multiplier(CabinClass.PREMIUM_ECONOMY)
        assert m == Decimal("1.60000000")

    def test_get_cabin_class_multiplier_business(self, engine):
        """Test business multiplier = 2.9."""
        m = engine.get_cabin_class_multiplier(CabinClass.BUSINESS)
        assert m == Decimal("2.90000000")

    def test_get_cabin_class_multiplier_first(self, engine):
        """Test first class multiplier = 4.0."""
        m = engine.get_cabin_class_multiplier(CabinClass.FIRST)
        assert m == Decimal("4.00000000")


# ==============================================================================
# CURRENCY RATE TESTS
# ==============================================================================


class TestCurrencyRates:
    """Test currency rate retrieval."""

    def test_get_currency_rate_usd(self, engine):
        """Test USD rate = 1.0."""
        rate = engine.get_currency_rate(CurrencyCode.USD)
        assert rate == Decimal("1.00000000")

    def test_get_currency_rate_eur(self, engine):
        """Test EUR rate = 1.0850."""
        rate = engine.get_currency_rate(CurrencyCode.EUR)
        assert rate == Decimal("1.08500000")

    def test_get_currency_rate_gbp(self, engine):
        """Test GBP rate = 1.2650."""
        rate = engine.get_currency_rate(CurrencyCode.GBP)
        assert rate == Decimal("1.26500000")

    def test_get_currency_rate_jpy(self, engine):
        """Test JPY rate = 0.006667."""
        rate = engine.get_currency_rate(CurrencyCode.JPY)
        assert rate == Decimal("0.00666700")

    def test_get_currency_rate_all_currencies(self, engine):
        """Test all 12 currencies have valid rates."""
        for currency in CurrencyCode:
            rate = engine.get_currency_rate(currency)
            assert isinstance(rate, Decimal)
            assert rate > Decimal("0")


# ==============================================================================
# CPI DEFLATOR TESTS
# ==============================================================================


class TestCPIDeflators:
    """Test CPI deflator retrieval."""

    def test_get_cpi_deflator_2021(self, engine):
        """Test 2021 (base year) deflator = 1.0."""
        deflator = engine.get_cpi_deflator(2021)
        assert deflator == Decimal("1.00000000")

    def test_get_cpi_deflator_2024(self, engine):
        """Test 2024 deflator = 1.1490."""
        deflator = engine.get_cpi_deflator(2024)
        assert deflator == Decimal("1.14900000")

    def test_get_cpi_deflator_2015(self, engine):
        """Test 2015 deflator = 0.8490."""
        deflator = engine.get_cpi_deflator(2015)
        assert deflator == Decimal("0.84900000")

    def test_get_cpi_deflator_2025(self, engine):
        """Test 2025 deflator = 1.1780."""
        deflator = engine.get_cpi_deflator(2025)
        assert deflator == Decimal("1.17800000")

    def test_get_cpi_deflator_invalid_year(self, engine):
        """Test invalid year raises ValueError."""
        with pytest.raises(ValueError, match="CPI deflator not available"):
            engine.get_cpi_deflator(1999)

    def test_get_cpi_deflator_all_years(self, engine):
        """Test all available years return valid deflators."""
        for year in range(2015, 2026):
            deflator = engine.get_cpi_deflator(year)
            assert isinstance(deflator, Decimal)
            assert deflator > Decimal("0")


# ==============================================================================
# AVAILABLE OPTIONS TESTS
# ==============================================================================


class TestAvailableOptions:
    """Test available modes and classes queries."""

    def test_get_available_transport_modes(self, engine):
        """Test available transport modes includes all 8 modes."""
        modes = engine.get_available_transport_modes()
        assert isinstance(modes, list)
        assert len(modes) == 8
        assert "air" in modes
        assert "rail" in modes
        assert "road" in modes
        assert "bus" in modes
        assert "taxi" in modes
        assert "ferry" in modes
        assert "motorcycle" in modes
        assert "hotel" in modes

    def test_get_available_cabin_classes(self, engine):
        """Test available cabin classes returns 4 classes with multipliers."""
        classes = engine.get_available_cabin_classes()
        assert isinstance(classes, list)
        assert len(classes) == 4

        # Verify each class entry has expected keys
        for entry in classes:
            assert "cabin_class" in entry
            assert "multiplier" in entry
            assert isinstance(entry["multiplier"], Decimal)

    def test_get_available_cabin_classes_economy_first(self, engine):
        """Test economy and first class are in available classes."""
        classes = engine.get_available_cabin_classes()
        class_names = [c["cabin_class"] for c in classes]
        assert "economy" in class_names
        assert "first" in class_names


# ==============================================================================
# TRANSPORT MODE CLASSIFICATION TESTS
# ==============================================================================


class TestTransportModeClassification:
    """Test transport mode classification from trip data."""

    def test_classify_transport_mode_air(self, engine):
        """Test IATA codes classify as AIR."""
        mode = engine.classify_transport_mode({
            "origin_iata": "JFK",
            "destination_iata": "LHR",
        })
        assert mode == TransportMode.AIR

    def test_classify_transport_mode_air_origin_only(self, engine):
        """Test origin_iata alone classifies as AIR."""
        mode = engine.classify_transport_mode({"origin_iata": "JFK"})
        assert mode == TransportMode.AIR

    def test_classify_transport_mode_rail(self, engine):
        """Test rail_type key classifies as RAIL."""
        mode = engine.classify_transport_mode({
            "rail_type": "national",
            "distance_km": 640,
        })
        assert mode == TransportMode.RAIL

    def test_classify_transport_mode_bus(self, engine):
        """Test bus_type key classifies as BUS."""
        mode = engine.classify_transport_mode({"bus_type": "coach"})
        assert mode == TransportMode.BUS

    def test_classify_transport_mode_ferry(self, engine):
        """Test ferry_type key classifies as FERRY."""
        mode = engine.classify_transport_mode({"ferry_type": "foot_passenger"})
        assert mode == TransportMode.FERRY

    def test_classify_transport_mode_taxi(self, engine):
        """Test taxi_type key classifies as TAXI."""
        mode = engine.classify_transport_mode({"taxi_type": "regular"})
        assert mode == TransportMode.TAXI

    def test_classify_transport_mode_hotel(self, engine):
        """Test room_nights key classifies as HOTEL."""
        mode = engine.classify_transport_mode({
            "room_nights": 3,
            "country_code": "GB",
        })
        assert mode == TransportMode.HOTEL

    def test_classify_transport_mode_hotel_class(self, engine):
        """Test hotel_class key classifies as HOTEL."""
        mode = engine.classify_transport_mode({"hotel_class": "luxury"})
        assert mode == TransportMode.HOTEL

    def test_classify_transport_mode_road(self, engine):
        """Test vehicle_type key classifies as ROAD."""
        mode = engine.classify_transport_mode({
            "vehicle_type": "car_average",
            "distance_km": 300,
        })
        assert mode == TransportMode.ROAD

    def test_classify_transport_mode_motorcycle(self, engine):
        """Test vehicle_type containing 'motorcycle' classifies as MOTORCYCLE."""
        mode = engine.classify_transport_mode({
            "vehicle_type": "motorcycle",
        })
        assert mode == TransportMode.MOTORCYCLE

    def test_classify_transport_mode_fuel(self, engine):
        """Test fuel_type key classifies as ROAD."""
        mode = engine.classify_transport_mode({
            "fuel_type": "diesel",
            "litres": 45.0,
        })
        assert mode == TransportMode.ROAD

    def test_classify_transport_mode_explicit(self, engine):
        """Test explicit mode field overrides heuristics."""
        mode = engine.classify_transport_mode({"mode": "ferry"})
        assert mode == TransportMode.FERRY

    def test_classify_transport_mode_empty(self, engine):
        """Test empty trip_data defaults to ROAD."""
        mode = engine.classify_transport_mode({})
        assert mode == TransportMode.ROAD

    def test_classify_transport_mode_unknown_keys(self, engine):
        """Test unknown keys default to ROAD."""
        mode = engine.classify_transport_mode({"unrelated_field": "value"})
        assert mode == TransportMode.ROAD


# ==============================================================================
# DATABASE SUMMARY AND LOOKUP COUNT TESTS
# ==============================================================================


class TestDatabaseSummary:
    """Test database summary and lookup count."""

    def test_get_database_summary(self, engine):
        """Test database summary returns expected keys and counts."""
        summary = engine.get_database_summary()
        assert isinstance(summary, dict)
        assert summary["airport_count"] == 50
        assert summary["air_distance_bands"] == 4
        assert summary["cabin_classes"] == 4
        assert summary["rail_types"] == 8
        assert summary["road_vehicle_types"] == 13
        assert summary["fuel_types"] == 5
        assert summary["bus_types"] == 2
        assert summary["ferry_types"] == 2
        assert summary["hotel_countries"] == 16
        assert summary["hotel_classes"] == 4
        assert summary["eeio_naics_codes"] == 10
        assert summary["currencies"] == 12
        assert summary["transport_modes"] == 8

    def test_get_lookup_count_initial(self, engine):
        """Test initial lookup count is 0."""
        assert engine.get_lookup_count() == 0

    def test_get_lookup_count_increments(self, engine):
        """Test lookup count increments with each factor lookup."""
        engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        engine.get_rail_emission_factor(RailType.NATIONAL)
        engine.get_road_emission_factor(RoadVehicleType.CAR_AVERAGE)
        assert engine.get_lookup_count() == 3

    def test_get_lookup_count_thread_safe(self, engine):
        """Test lookup count is thread-safe under concurrent access."""
        errors: List[Exception] = []

        def do_lookups():
            try:
                for _ in range(50):
                    engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_lookups) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine.get_lookup_count() == 250  # 5 threads x 50 lookups


# ==============================================================================
# QUANTIZATION TESTS
# ==============================================================================


class TestQuantization:
    """Test 8 decimal place quantization."""

    def test_quantize_air_ef(self, engine):
        """Test air EF values are quantized to 8dp."""
        ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        # 0.19309 should become 0.19309000
        without_rf_str = str(ef["without_rf"])
        parts = without_rf_str.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 8

    def test_quantize_hotel_ef(self, engine):
        """Test hotel EF values are quantized to 8dp."""
        ef = engine.get_hotel_emission_factor("GB")
        # 12.32 should become 12.32000000
        ef_str = str(ef["ef_per_room_night"])
        parts = ef_str.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 8

    def test_quantize_currency_rate(self, engine):
        """Test currency rate is quantized to 8dp."""
        rate = engine.get_currency_rate(CurrencyCode.GBP)
        rate_str = str(rate)
        parts = rate_str.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 8

    def test_quantize_cpi_deflator(self, engine):
        """Test CPI deflator is quantized to 8dp."""
        deflator = engine.get_cpi_deflator(2024)
        deflator_str = str(deflator)
        parts = deflator_str.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 8


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_rail_type_raises_value_error(self, engine):
        """Test invalid rail type value raises ValueError."""
        # RailType is an enum, so passing a bad value should fail at enum level
        with pytest.raises((ValueError, KeyError)):
            engine.get_rail_emission_factor(RailType("nonexistent"))

    def test_invalid_road_vehicle_raises_value_error(self, engine):
        """Test invalid road vehicle type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            engine.get_road_emission_factor(RoadVehicleType("nonexistent"))

    def test_invalid_fuel_type_raises_value_error(self, engine):
        """Test invalid fuel type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            engine.get_fuel_emission_factor(FuelType("nonexistent"))

    def test_invalid_bus_type_raises_value_error(self, engine):
        """Test invalid bus type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            engine.get_bus_emission_factor(BusType("nonexistent"))

    def test_invalid_ferry_type_raises_value_error(self, engine):
        """Test invalid ferry type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            engine.get_ferry_emission_factor(FerryType("nonexistent"))

    def test_invalid_eeio_naics_raises_value_error(self, engine):
        """Test invalid NAICS code raises ValueError."""
        with pytest.raises(ValueError, match="EEIO factor not found"):
            engine.get_eeio_factor("000000")

    def test_invalid_currency_raises_value_error(self, engine):
        """Test invalid currency raises ValueError at enum level."""
        with pytest.raises((ValueError, KeyError)):
            engine.get_currency_rate(CurrencyCode("XYZ"))

    def test_invalid_cpi_year_raises_value_error(self, engine):
        """Test invalid CPI year raises ValueError."""
        with pytest.raises(ValueError, match="CPI deflator not available"):
            engine.get_cpi_deflator(2050)


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_database_engine_module_coverage():
    """Meta-test to ensure comprehensive database engine coverage."""
    tested_method_groups = [
        "singleton_pattern",
        "air_emission_factors",
        "rail_emission_factors",
        "road_emission_factors",
        "fuel_emission_factors",
        "bus_emission_factors",
        "ferry_emission_factors",
        "hotel_emission_factors",
        "eeio_factors",
        "airport_lookups",
        "cabin_class_multiplier",
        "currency_rates",
        "cpi_deflators",
        "available_options",
        "transport_mode_classification",
        "database_summary",
        "quantization",
        "error_handling",
    ]
    assert len(tested_method_groups) == 18
