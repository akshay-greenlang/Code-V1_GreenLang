# -*- coding: utf-8 -*-
"""
Unit tests for GroundTransportCalculatorEngine (Engine 3) - AGENT-MRV-019

Tests the complete ground transport emissions calculation pipeline for:
- Rail (8 types: national, international, eurostar, high-speed, etc.)
- Road distance-based (13 vehicle types: car average, hybrid, BEV, etc.)
- Road fuel-based (5 fuel types: petrol, diesel, LPG, CNG, E85)
- Taxi (regular and black cab)
- Bus (local and coach)
- Ferry (foot and car passenger)
- Motorcycle
- Unit conversions (miles-to-km, gallons-to-litres)
- Batch processing and thread safety

70 tests total across 9 test classes.

All expected emissions values are computed from DEFRA 2024 emission
factors using deterministic Decimal arithmetic matching the engine
formulae:
    Per-pkm: co2e = distance_km x passengers x ef
    Per-vkm: co2e = distance_km x ef_per_vkm
    Fuel:    co2e = litres x ef_per_litre

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.mrv.business_travel.models import (
    RailInput,
    RailResult,
    RoadDistanceInput,
    RoadDistanceResult,
    RoadFuelInput,
    RoadFuelResult,
    TaxiInput,
    BusInput,
    FerryInput,
    RailType,
    RoadVehicleType,
    FuelType,
    BusType,
    FerryType,
    EFSource,
    RAIL_EMISSION_FACTORS,
    ROAD_VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    BUS_EMISSION_FACTORS,
    FERRY_EMISSION_FACTORS,
)

# ---------------------------------------------------------------------------
# Module-level constants for expected values
# ---------------------------------------------------------------------------

_Q8 = Decimal("0.00000001")


def _q(v: Decimal) -> Decimal:
    """Local quantize helper matching engine precision."""
    return v.quantize(_Q8, rounding=ROUND_HALF_UP)


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """
    Reset the GroundTransportCalculatorEngine singleton before and
    after each test so that calculation counts and state do not
    leak between tests.
    """
    from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
        GroundTransportCalculatorEngine,
    )

    GroundTransportCalculatorEngine.reset_instance()
    yield
    GroundTransportCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """
    Create a fresh GroundTransportCalculatorEngine with mocked
    dependencies (config, metrics, provenance).
    """
    with patch(
        "greenlang.agents.mrv.business_travel.ground_transport_calculator.get_config"
    ) as mock_config, patch(
        "greenlang.agents.mrv.business_travel.ground_transport_calculator.get_metrics"
    ) as mock_metrics, patch(
        "greenlang.agents.mrv.business_travel.ground_transport_calculator.get_provenance_tracker"
    ) as mock_prov:
        mock_config.return_value = MagicMock()
        metrics = MagicMock()
        mock_metrics.return_value = metrics
        mock_prov.return_value = MagicMock()

        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        eng = GroundTransportCalculatorEngine(metrics=metrics)
        yield eng


# ===========================================================================
# 1. RAIL CALCULATION TESTS (15 tests)
# ===========================================================================


class TestRailCalculation:
    """Validate rail emissions using per-pkm factors (DEFRA 2024)."""

    # 1
    def test_national_rail_640km_co2e(self, engine):
        """National rail 640 km x 1 pax: co2e = 640 * 0.03549 = 22.71360."""
        inp = RailInput(
            rail_type=RailType.NATIONAL,
            distance_km=Decimal("640"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected_co2e = _q(Decimal("640") * Decimal("0.03549"))
        assert result.co2e == expected_co2e

    # 2
    def test_national_rail_640km_wtt(self, engine):
        """National rail 640 km x 1 pax: wtt = 640 * 0.00434 = 2.77760."""
        inp = RailInput(
            rail_type=RailType.NATIONAL,
            distance_km=Decimal("640"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected_wtt = _q(Decimal("640") * Decimal("0.00434"))
        assert result.wtt_co2e == expected_wtt

    # 3
    def test_national_rail_total_equals_co2e_plus_wtt(self, engine):
        """Total = co2e + wtt_co2e."""
        inp = RailInput(
            rail_type=RailType.NATIONAL,
            distance_km=Decimal("640"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 4
    def test_eurostar_340km_2pax_co2e(self, engine):
        """Eurostar 340 km x 2 pax: co2e = 340 * 2 * 0.00446 = 3.03280."""
        inp = RailInput(
            rail_type=RailType.EUROSTAR,
            distance_km=Decimal("340"),
            passengers=2,
        )
        result = engine.calculate_rail(inp)
        expected_co2e = _q(Decimal("340") * Decimal("2") * Decimal("0.00446"))
        assert result.co2e == expected_co2e

    # 5
    def test_eurostar_340km_2pax_wtt(self, engine):
        """Eurostar 340 km x 2 pax: wtt = 340 * 2 * 0.00086 = 0.58480."""
        inp = RailInput(
            rail_type=RailType.EUROSTAR,
            distance_km=Decimal("340"),
            passengers=2,
        )
        result = engine.calculate_rail(inp)
        expected_wtt = _q(Decimal("340") * Decimal("2") * Decimal("0.00086"))
        assert result.wtt_co2e == expected_wtt

    # 6
    def test_us_intercity_co2e(self, engine):
        """US Intercity 500 km: co2e = 500 * 0.08900 = 44.50000."""
        inp = RailInput(
            rail_type=RailType.US_INTERCITY,
            distance_km=Decimal("500"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected = _q(Decimal("500") * Decimal("0.08900"))
        assert result.co2e == expected

    # 7
    def test_high_speed_co2e(self, engine):
        """High speed 800 km: co2e = 800 * 0.00324 = 2.59200."""
        inp = RailInput(
            rail_type=RailType.HIGH_SPEED,
            distance_km=Decimal("800"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected = _q(Decimal("800") * Decimal("0.00324"))
        assert result.co2e == expected

    # 8
    def test_light_rail_co2e(self, engine):
        """Light rail 20 km: co2e = 20 * 0.02904 = 0.58080."""
        inp = RailInput(
            rail_type=RailType.LIGHT_RAIL,
            distance_km=Decimal("20"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected = _q(Decimal("20") * Decimal("0.02904"))
        assert result.co2e == expected

    # 9
    def test_underground_co2e(self, engine):
        """Underground 10 km: co2e = 10 * 0.02781 = 0.27810."""
        inp = RailInput(
            rail_type=RailType.UNDERGROUND,
            distance_km=Decimal("10"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected = _q(Decimal("10") * Decimal("0.02781"))
        assert result.co2e == expected

    # 10
    def test_us_commuter_co2e(self, engine):
        """US commuter 30 km: co2e = 30 * 0.10500 = 3.15000."""
        inp = RailInput(
            rail_type=RailType.US_COMMUTER,
            distance_km=Decimal("30"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        expected = _q(Decimal("30") * Decimal("0.10500"))
        assert result.co2e == expected

    # 11
    def test_rail_result_type(self, engine):
        """Result type must be RailResult."""
        inp = RailInput(
            rail_type=RailType.NATIONAL,
            distance_km=Decimal("100"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        assert isinstance(result, RailResult)

    # 12
    def test_rail_ef_source_defra(self, engine):
        """EF source must be DEFRA."""
        inp = RailInput(
            rail_type=RailType.NATIONAL,
            distance_km=Decimal("100"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        assert result.ef_source == EFSource.DEFRA

    # 13
    def test_rail_provenance_hash_present(self, engine):
        """Provenance hash is a 64-char hex string."""
        inp = RailInput(
            rail_type=RailType.NATIONAL,
            distance_km=Decimal("100"),
            passengers=1,
        )
        result = engine.calculate_rail(inp)
        assert len(result.provenance_hash) == 64

    # 14
    def test_rail_provenance_deterministic(self, engine):
        """Same input produces same provenance hash."""
        inp = RailInput(
            rail_type=RailType.EUROSTAR,
            distance_km=Decimal("340"),
            passengers=1,
        )
        r1 = engine.calculate_rail(inp)
        r2 = engine.calculate_rail(inp)
        assert r1.provenance_hash == r2.provenance_hash

    # 15
    def test_rail_multi_passenger_scales_linearly(self, engine):
        """5 passengers = 5x single-passenger emissions."""
        one = engine.calculate_rail(
            RailInput(rail_type=RailType.NATIONAL, distance_km=Decimal("200"), passengers=1)
        )
        five = engine.calculate_rail(
            RailInput(rail_type=RailType.NATIONAL, distance_km=Decimal("200"), passengers=5)
        )
        assert five.total_co2e == _q(one.total_co2e * Decimal("5"))


# ===========================================================================
# 2. ROAD DISTANCE-BASED TESTS (15 tests)
# ===========================================================================


class TestRoadDistanceCalculation:
    """Validate road vehicle distance-based emissions (per-vkm)."""

    # 16
    def test_car_average_300km_co2e(self, engine):
        """Car average 300 km: co2e = 300 * 0.27145 = 81.43500."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("300"),
        )
        result = engine.calculate_road_distance(inp)
        expected = _q(Decimal("300") * Decimal("0.27145"))
        assert result.co2e == expected

    # 17
    def test_car_average_300km_wtt(self, engine):
        """Car average 300 km: wtt = 300 * 0.06291 = 18.87300."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("300"),
        )
        result = engine.calculate_road_distance(inp)
        expected = _q(Decimal("300") * Decimal("0.06291"))
        assert result.wtt_co2e == expected

    # 18
    def test_car_average_total_equals_sum(self, engine):
        """Total = co2e + wtt_co2e."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("300"),
        )
        result = engine.calculate_road_distance(inp)
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 19
    def test_bev_100km_co2e(self, engine):
        """BEV 100 km: co2e = 100 * 0.07005 = 7.00500."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.BEV,
            distance_km=Decimal("100"),
        )
        result = engine.calculate_road_distance(inp)
        expected = _q(Decimal("100") * Decimal("0.07005"))
        assert result.co2e == expected

    # 20
    def test_bev_100km_wtt(self, engine):
        """BEV 100 km: wtt = 100 * 0.01479 = 1.47900."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.BEV,
            distance_km=Decimal("100"),
        )
        result = engine.calculate_road_distance(inp)
        expected = _q(Decimal("100") * Decimal("0.01479"))
        assert result.wtt_co2e == expected

    # 21
    def test_hybrid_lower_than_petrol_medium(self, engine):
        """Hybrid emissions lower than medium petrol for same distance."""
        dist = Decimal("200")
        hybrid = engine.calculate_road_distance(
            RoadDistanceInput(vehicle_type=RoadVehicleType.HYBRID, distance_km=dist)
        )
        petrol = engine.calculate_road_distance(
            RoadDistanceInput(vehicle_type=RoadVehicleType.CAR_MEDIUM_PETROL, distance_km=dist)
        )
        assert hybrid.total_co2e < petrol.total_co2e

    # 22
    def test_bev_lower_than_hybrid(self, engine):
        """BEV emissions lower than hybrid for same distance."""
        dist = Decimal("200")
        bev = engine.calculate_road_distance(
            RoadDistanceInput(vehicle_type=RoadVehicleType.BEV, distance_km=dist)
        )
        hybrid = engine.calculate_road_distance(
            RoadDistanceInput(vehicle_type=RoadVehicleType.HYBRID, distance_km=dist)
        )
        assert bev.total_co2e < hybrid.total_co2e

    # 23
    def test_large_petrol_highest_among_petrol(self, engine):
        """Large petrol has highest EF among petrol sizes."""
        dist = Decimal("100")
        small = engine.calculate_road_distance(
            RoadDistanceInput(vehicle_type=RoadVehicleType.CAR_SMALL_PETROL, distance_km=dist)
        )
        large = engine.calculate_road_distance(
            RoadDistanceInput(vehicle_type=RoadVehicleType.CAR_LARGE_PETROL, distance_km=dist)
        )
        assert large.total_co2e > small.total_co2e

    # 24
    def test_plugin_hybrid_co2e(self, engine):
        """Plugin hybrid 150 km: co2e = 150 * 0.10250 = 15.37500."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.PLUGIN_HYBRID,
            distance_km=Decimal("150"),
        )
        result = engine.calculate_road_distance(inp)
        expected = _q(Decimal("150") * Decimal("0.10250"))
        assert result.co2e == expected

    # 25
    def test_road_result_type(self, engine):
        """Result type must be RoadDistanceResult."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("100"),
        )
        result = engine.calculate_road_distance(inp)
        assert isinstance(result, RoadDistanceResult)

    # 26
    def test_road_ef_source_defra(self, engine):
        """EF source must be DEFRA."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("100"),
        )
        result = engine.calculate_road_distance(inp)
        assert result.ef_source == EFSource.DEFRA

    # 27
    def test_road_provenance_hash_64_chars(self, engine):
        """Provenance hash is 64 hex characters."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("100"),
        )
        result = engine.calculate_road_distance(inp)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    # 28
    def test_road_vehicle_type_stored(self, engine):
        """Vehicle type in result matches input."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_SMALL_DIESEL,
            distance_km=Decimal("100"),
        )
        result = engine.calculate_road_distance(inp)
        assert result.vehicle_type == RoadVehicleType.CAR_SMALL_DIESEL

    # 29
    def test_road_distance_stored(self, engine):
        """Distance in result matches input."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_AVERAGE,
            distance_km=Decimal("123.45"),
        )
        result = engine.calculate_road_distance(inp)
        assert result.distance_km == Decimal("123.45")

    # 30
    def test_road_small_diesel_co2e(self, engine):
        """Small diesel 200 km: co2e = 200 * 0.19290."""
        inp = RoadDistanceInput(
            vehicle_type=RoadVehicleType.CAR_SMALL_DIESEL,
            distance_km=Decimal("200"),
        )
        result = engine.calculate_road_distance(inp)
        expected = _q(Decimal("200") * Decimal("0.19290"))
        assert result.co2e == expected


# ===========================================================================
# 3. ROAD FUEL-BASED TESTS (10 tests)
# ===========================================================================


class TestRoadFuelCalculation:
    """Validate fuel-based road emissions (per-litre/per-kg)."""

    # 31
    def test_diesel_40l_co2e(self, engine):
        """Diesel 40 L: co2e = 40 * 2.70370 = 108.14800."""
        inp = RoadFuelInput(fuel_type=FuelType.DIESEL, litres=Decimal("40"))
        result = engine.calculate_road_fuel(inp)
        expected = _q(Decimal("40") * Decimal("2.70370"))
        assert result.co2e == expected

    # 32
    def test_diesel_40l_wtt(self, engine):
        """Diesel 40 L: wtt = 40 * 0.60927 = 24.37080."""
        inp = RoadFuelInput(fuel_type=FuelType.DIESEL, litres=Decimal("40"))
        result = engine.calculate_road_fuel(inp)
        expected = _q(Decimal("40") * Decimal("0.60927"))
        assert result.wtt_co2e == expected

    # 33
    def test_diesel_total_equals_sum(self, engine):
        """Total = co2e + wtt_co2e."""
        inp = RoadFuelInput(fuel_type=FuelType.DIESEL, litres=Decimal("40"))
        result = engine.calculate_road_fuel(inp)
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 34
    def test_petrol_50l_co2e(self, engine):
        """Petrol 50 L: co2e = 50 * 2.31480 = 115.74000."""
        inp = RoadFuelInput(fuel_type=FuelType.PETROL, litres=Decimal("50"))
        result = engine.calculate_road_fuel(inp)
        expected = _q(Decimal("50") * Decimal("2.31480"))
        assert result.co2e == expected

    # 35
    def test_lpg_30l_co2e(self, engine):
        """LPG 30 L: co2e = 30 * 1.55370 = 46.61100."""
        inp = RoadFuelInput(fuel_type=FuelType.LPG, litres=Decimal("30"))
        result = engine.calculate_road_fuel(inp)
        expected = _q(Decimal("30") * Decimal("1.55370"))
        assert result.co2e == expected

    # 36
    def test_cng_20kg_co2e(self, engine):
        """CNG 20 kg: co2e = 20 * 2.53970 = 50.79400."""
        inp = RoadFuelInput(fuel_type=FuelType.CNG, litres=Decimal("20"))
        result = engine.calculate_road_fuel(inp)
        expected = _q(Decimal("20") * Decimal("2.53970"))
        assert result.co2e == expected

    # 37
    def test_e85_25l_co2e(self, engine):
        """E85 25 L: co2e = 25 * 0.34728 = 8.68200."""
        inp = RoadFuelInput(fuel_type=FuelType.E85, litres=Decimal("25"))
        result = engine.calculate_road_fuel(inp)
        expected = _q(Decimal("25") * Decimal("0.34728"))
        assert result.co2e == expected

    # 38
    def test_fuel_result_type(self, engine):
        """Result type must be RoadFuelResult."""
        inp = RoadFuelInput(fuel_type=FuelType.DIESEL, litres=Decimal("10"))
        result = engine.calculate_road_fuel(inp)
        assert isinstance(result, RoadFuelResult)

    # 39
    def test_fuel_provenance_hash(self, engine):
        """Provenance hash is 64 hex characters."""
        inp = RoadFuelInput(fuel_type=FuelType.PETROL, litres=Decimal("10"))
        result = engine.calculate_road_fuel(inp)
        assert len(result.provenance_hash) == 64

    # 40
    def test_e85_lower_than_diesel(self, engine):
        """E85 has significantly lower EF than diesel per litre."""
        amount = Decimal("50")
        e85 = engine.calculate_road_fuel(
            RoadFuelInput(fuel_type=FuelType.E85, litres=amount)
        )
        diesel = engine.calculate_road_fuel(
            RoadFuelInput(fuel_type=FuelType.DIESEL, litres=amount)
        )
        assert e85.total_co2e < diesel.total_co2e


# ===========================================================================
# 4. TAXI TESTS (5 tests)
# ===========================================================================


class TestTaxiCalculation:
    """Validate taxi / ride-hailing emissions (per-vkm)."""

    # 41
    def test_regular_taxi_25km_co2e(self, engine):
        """Regular taxi 25 km: co2e = 25 * 0.20920 = 5.23000."""
        inp = TaxiInput(
            taxi_type=RoadVehicleType.TAXI_REGULAR,
            distance_km=Decimal("25"),
        )
        result = engine.calculate_taxi(inp)
        expected = _q(Decimal("25") * Decimal("0.20920"))
        assert result.co2e == expected

    # 42
    def test_regular_taxi_25km_wtt(self, engine):
        """Regular taxi 25 km: wtt = 25 * 0.04710 = 1.17750."""
        inp = TaxiInput(
            taxi_type=RoadVehicleType.TAXI_REGULAR,
            distance_km=Decimal("25"),
        )
        result = engine.calculate_taxi(inp)
        expected = _q(Decimal("25") * Decimal("0.04710"))
        assert result.wtt_co2e == expected

    # 43
    def test_black_cab_higher_than_regular(self, engine):
        """Black cab emissions higher than regular taxi."""
        dist = Decimal("15")
        regular = engine.calculate_taxi(
            TaxiInput(taxi_type=RoadVehicleType.TAXI_REGULAR, distance_km=dist)
        )
        black = engine.calculate_taxi(
            TaxiInput(taxi_type=RoadVehicleType.TAXI_BLACK_CAB, distance_km=dist)
        )
        assert black.total_co2e > regular.total_co2e

    # 44
    def test_taxi_total_equals_sum(self, engine):
        """Total = co2e + wtt_co2e."""
        inp = TaxiInput(
            taxi_type=RoadVehicleType.TAXI_REGULAR,
            distance_km=Decimal("10"),
        )
        result = engine.calculate_taxi(inp)
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 45
    def test_taxi_result_type(self, engine):
        """Result type must be RoadDistanceResult."""
        inp = TaxiInput(
            taxi_type=RoadVehicleType.TAXI_REGULAR,
            distance_km=Decimal("10"),
        )
        result = engine.calculate_taxi(inp)
        assert isinstance(result, RoadDistanceResult)


# ===========================================================================
# 5. BUS TESTS (5 tests)
# ===========================================================================


class TestBusCalculation:
    """Validate bus emissions using per-pkm factors."""

    # 46
    def test_coach_100km_co2e(self, engine):
        """Coach 100 km x 1 pax: co2e = 100 * 0.02732 = 2.73200."""
        inp = BusInput(
            bus_type=BusType.COACH,
            distance_km=Decimal("100"),
            passengers=1,
        )
        result = engine.calculate_bus(inp)
        expected = _q(Decimal("100") * Decimal("1") * Decimal("0.02732"))
        assert result.co2e == expected

    # 47
    def test_coach_100km_wtt(self, engine):
        """Coach 100 km x 1 pax: wtt = 100 * 0.00489 = 0.48900."""
        inp = BusInput(
            bus_type=BusType.COACH,
            distance_km=Decimal("100"),
            passengers=1,
        )
        result = engine.calculate_bus(inp)
        expected = _q(Decimal("100") * Decimal("1") * Decimal("0.00489"))
        assert result.wtt_co2e == expected

    # 48
    def test_local_bus_higher_than_coach(self, engine):
        """Local bus emissions higher than coach per pkm."""
        dist = Decimal("50")
        local = engine.calculate_bus(
            BusInput(bus_type=BusType.LOCAL, distance_km=dist, passengers=1)
        )
        coach = engine.calculate_bus(
            BusInput(bus_type=BusType.COACH, distance_km=dist, passengers=1)
        )
        assert local.total_co2e > coach.total_co2e

    # 49
    def test_bus_total_equals_sum(self, engine):
        """Total = co2e + wtt_co2e."""
        inp = BusInput(
            bus_type=BusType.COACH,
            distance_km=Decimal("50"),
            passengers=1,
        )
        result = engine.calculate_bus(inp)
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 50
    def test_bus_multi_passenger(self, engine):
        """3 passengers = 3x single-passenger emissions."""
        one = engine.calculate_bus(
            BusInput(bus_type=BusType.COACH, distance_km=Decimal("100"), passengers=1)
        )
        three = engine.calculate_bus(
            BusInput(bus_type=BusType.COACH, distance_km=Decimal("100"), passengers=3)
        )
        assert three.total_co2e == _q(one.total_co2e * Decimal("3"))


# ===========================================================================
# 6. FERRY TESTS (5 tests)
# ===========================================================================


class TestFerryCalculation:
    """Validate ferry emissions using per-pkm factors."""

    # 51
    def test_foot_passenger_50km_co2e(self, engine):
        """Foot passenger 50 km: co2e = 50 * 0.01877 = 0.93850."""
        inp = FerryInput(
            ferry_type=FerryType.FOOT_PASSENGER,
            distance_km=Decimal("50"),
            passengers=1,
        )
        result = engine.calculate_ferry(inp)
        expected = _q(Decimal("50") * Decimal("1") * Decimal("0.01877"))
        assert result.co2e == expected

    # 52
    def test_car_passenger_50km_co2e(self, engine):
        """Car passenger 50 km: co2e = 50 * 0.12952 = 6.47600."""
        inp = FerryInput(
            ferry_type=FerryType.CAR_PASSENGER,
            distance_km=Decimal("50"),
            passengers=1,
        )
        result = engine.calculate_ferry(inp)
        expected = _q(Decimal("50") * Decimal("1") * Decimal("0.12952"))
        assert result.co2e == expected

    # 53
    def test_car_passenger_higher_than_foot(self, engine):
        """Car passenger emissions much higher than foot passenger."""
        dist = Decimal("50")
        foot = engine.calculate_ferry(
            FerryInput(ferry_type=FerryType.FOOT_PASSENGER, distance_km=dist, passengers=1)
        )
        car = engine.calculate_ferry(
            FerryInput(ferry_type=FerryType.CAR_PASSENGER, distance_km=dist, passengers=1)
        )
        assert car.total_co2e > foot.total_co2e

    # 54
    def test_ferry_total_equals_sum(self, engine):
        """Total = co2e + wtt_co2e."""
        inp = FerryInput(
            ferry_type=FerryType.FOOT_PASSENGER,
            distance_km=Decimal("30"),
            passengers=1,
        )
        result = engine.calculate_ferry(inp)
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 55
    def test_ferry_provenance_hash(self, engine):
        """Provenance hash is 64 hex characters."""
        inp = FerryInput(
            ferry_type=FerryType.FOOT_PASSENGER,
            distance_km=Decimal("30"),
            passengers=1,
        )
        result = engine.calculate_ferry(inp)
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 7. MOTORCYCLE TESTS (5 tests)
# ===========================================================================


class TestMotorcycleCalculation:
    """Validate motorcycle emissions (per-vkm, occupancy 1.0)."""

    # 56
    def test_motorcycle_100km_co2e(self, engine):
        """Motorcycle 100 km: co2e = 100 * 0.11337 = 11.33700."""
        result = engine.calculate_motorcycle(Decimal("100"))
        expected = _q(Decimal("100") * Decimal("0.11337"))
        assert result.co2e == expected

    # 57
    def test_motorcycle_100km_wtt(self, engine):
        """Motorcycle 100 km: wtt = 100 * 0.02867 = 2.86700."""
        result = engine.calculate_motorcycle(Decimal("100"))
        expected = _q(Decimal("100") * Decimal("0.02867"))
        assert result.wtt_co2e == expected

    # 58
    def test_motorcycle_total_equals_sum(self, engine):
        """Total = co2e + wtt_co2e."""
        result = engine.calculate_motorcycle(Decimal("100"))
        assert result.total_co2e == _q(result.co2e + result.wtt_co2e)

    # 59
    def test_motorcycle_vehicle_type(self, engine):
        """Vehicle type must be MOTORCYCLE."""
        result = engine.calculate_motorcycle(Decimal("50"))
        assert result.vehicle_type == RoadVehicleType.MOTORCYCLE

    # 60
    def test_motorcycle_negative_distance_raises(self, engine):
        """Negative distance raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            engine.calculate_motorcycle(Decimal("-10"))


# ===========================================================================
# 8. UNIT CONVERSION TESTS (5 tests)
# ===========================================================================


class TestUnitConversions:
    """Validate miles-to-km and gallons-to-litres conversions."""

    # 61
    def test_miles_to_km_100(self):
        """100 miles = 160.93400000 km."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        result = GroundTransportCalculatorEngine.convert_miles_to_km(Decimal("100"))
        assert result == _q(Decimal("100") * Decimal("1.60934"))

    # 62
    def test_miles_to_km_zero(self):
        """0 miles = 0 km."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        result = GroundTransportCalculatorEngine.convert_miles_to_km(Decimal("0"))
        assert result == _q(Decimal("0"))

    # 63
    def test_miles_to_km_negative_raises(self):
        """Negative miles raises ValueError."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        with pytest.raises(ValueError, match="non-negative"):
            GroundTransportCalculatorEngine.convert_miles_to_km(Decimal("-5"))

    # 64
    def test_gallons_to_litres_10(self):
        """10 US gallons = 37.85410000 litres."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        result = GroundTransportCalculatorEngine.convert_gallons_to_litres(Decimal("10"))
        assert result == _q(Decimal("10") * Decimal("3.78541"))

    # 65
    def test_gallons_to_litres_negative_raises(self):
        """Negative gallons raises ValueError."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        with pytest.raises(ValueError, match="non-negative"):
            GroundTransportCalculatorEngine.convert_gallons_to_litres(Decimal("-1"))


# ===========================================================================
# 9. BATCH AND INFRASTRUCTURE TESTS (5 tests)
# ===========================================================================


class TestBatchAndInfrastructure:
    """Validate batch processing, calculation counter, and thread safety."""

    # 66
    def test_batch_mixed_modes(self, engine):
        """Batch with rail and motorcycle returns 2 results."""
        inputs = [
            {
                "mode": "rail",
                "rail_type": "eurostar",
                "distance_km": "340",
                "passengers": 2,
            },
            {
                "mode": "motorcycle",
                "distance_km": "50",
            },
        ]
        results = engine.calculate_batch_ground(inputs)
        assert len(results) == 2
        success_count = sum(1 for r in results if r["status"] == "success")
        assert success_count == 2

    # 67
    def test_batch_with_error_continues(self, engine):
        """Batch with invalid item still processes valid items."""
        inputs = [
            {
                "mode": "rail",
                "rail_type": "national",
                "distance_km": "100",
            },
            {
                "mode": "invalid_mode",
            },
            {
                "mode": "motorcycle",
                "distance_km": "25",
            },
        ]
        results = engine.calculate_batch_ground(inputs)
        assert len(results) == 3
        success = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]
        assert len(success) == 2
        assert len(errors) == 1

    # 68
    def test_calculation_count_increments(self, engine):
        """Calculation count increments after each calculation."""
        initial = engine.calculation_count
        engine.calculate_rail(
            RailInput(rail_type=RailType.NATIONAL, distance_km=Decimal("100"), passengers=1)
        )
        assert engine.calculation_count == initial + 1
        engine.calculate_motorcycle(Decimal("50"))
        assert engine.calculation_count == initial + 2

    # 69
    def test_singleton_reset(self):
        """reset_instance clears singleton so next call creates new engine."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
            _instance,
        )

        GroundTransportCalculatorEngine.reset_instance()
        # After reset, module-level _instance should be None
        import greenlang.agents.mrv.business_travel.ground_transport_calculator as mod

        assert mod._instance is None

    # 70
    def test_ef_accessor_rail(self, engine):
        """get_rail_emission_factors returns dict with all rail types."""
        from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
            GroundTransportCalculatorEngine,
        )

        efs = GroundTransportCalculatorEngine.get_rail_emission_factors()
        assert len(efs) == len(RAIL_EMISSION_FACTORS)
        assert "national" in efs
        assert "eurostar" in efs
        assert "ttw" in efs["national"]
        assert "wtt" in efs["national"]
