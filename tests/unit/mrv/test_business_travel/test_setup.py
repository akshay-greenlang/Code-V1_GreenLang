# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-019 Business Travel Agent - BusinessTravelService (setup.py).

Tests the service facade including initialization, engine access, single/batch
calculation, flight/rail/road/hotel/spend delegation, compliance checking,
emission factor retrieval, airport search, transport modes, cabin classes,
aggregation, singleton pattern, and thread safety.

Target: 20 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.business_travel.setup import (
        BusinessTravelService,
        get_service,
        get_router,
        TripCalculationRequest,
        BatchTripCalculationRequest,
        FlightCalculationRequest,
        RailCalculationRequest,
        RoadCalculationRequest,
        HotelCalculationRequest,
        SpendCalculationRequest,
        ComplianceCheckRequest,
        UncertaintyRequest,
        HotSpotRequest,
        AggregationRequest,
        TripCalculationResponse,
        BatchTripResponse,
        ComplianceCheckResponse,
        UncertaintyResponse,
        HotSpotResponse,
        EmissionFactorListResponse,
        AggregationResponse,
        HealthResponse,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="BusinessTravelService not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service():
    """Create a fresh BusinessTravelService instance."""
    return BusinessTravelService()


@pytest.fixture
def flight_request():
    """FlightCalculationRequest for JFK -> LHR economy one-way."""
    return FlightCalculationRequest(
        origin_iata="JFK",
        destination_iata="LHR",
        cabin_class="economy",
        passengers=1,
        round_trip=False,
        rf_option="with_rf",
    )


@pytest.fixture
def rail_request():
    """RailCalculationRequest for national rail 500 km."""
    return RailCalculationRequest(
        rail_type="national",
        distance_km=500.0,
        passengers=1,
    )


@pytest.fixture
def road_request():
    """RoadCalculationRequest for car_average 200 km."""
    return RoadCalculationRequest(
        vehicle_type="car_average",
        distance_km=200.0,
    )


@pytest.fixture
def hotel_request():
    """HotelCalculationRequest for US standard 2 nights."""
    return HotelCalculationRequest(
        country_code="US",
        room_nights=2,
        hotel_class="standard",
    )


@pytest.fixture
def spend_request():
    """SpendCalculationRequest for air transportation $3000 USD."""
    return SpendCalculationRequest(
        naics_code="481000",
        amount=3000.0,
        currency="USD",
        reporting_year=2024,
    )


@pytest.fixture
def trip_request():
    """Generic TripCalculationRequest for rail."""
    return TripCalculationRequest(
        mode="rail",
        trip_data={
            "rail_type": "national",
            "distance_km": 300,
            "passengers": 1,
        },
    )


# ===========================================================================
# Tests (20)
# ===========================================================================


@_SKIP
class TestBusinessTravelService:
    """Test suite for BusinessTravelService facade."""

    def test_service_creation(self):
        """BusinessTravelService can be instantiated."""
        service = BusinessTravelService()
        assert service is not None
        assert service._initialized is True

    def test_service_has_all_engines(self, service):
        """Service exposes all 7 engine attributes."""
        engine_attrs = [
            "_database_engine",
            "_air_engine",
            "_ground_engine",
            "_hotel_engine",
            "_spend_engine",
            "_compliance_engine",
            "_pipeline_engine",
        ]
        for attr in engine_attrs:
            assert hasattr(service, attr), f"Missing engine attribute: {attr}"

    def test_service_calculate_flight(self, service, flight_request):
        """calculate_flight delegates to the pipeline and returns a response."""
        response = service.calculate_flight(flight_request)
        assert isinstance(response, TripCalculationResponse)
        assert response.mode == "air"
        assert response.total_co2e_kg > 0.0

    def test_service_calculate_rail(self, service, rail_request):
        """calculate_rail returns a valid response with positive emissions."""
        response = service.calculate_rail(rail_request)
        assert isinstance(response, TripCalculationResponse)
        assert response.mode == "rail"
        assert response.total_co2e_kg > 0.0

    def test_service_calculate_road(self, service, road_request):
        """calculate_road returns a valid response with positive emissions."""
        response = service.calculate_road(road_request)
        assert isinstance(response, TripCalculationResponse)
        assert response.mode == "road"
        assert response.total_co2e_kg > 0.0

    def test_service_calculate_hotel(self, service, hotel_request):
        """calculate_hotel returns a valid response with positive emissions."""
        response = service.calculate_hotel(hotel_request)
        assert isinstance(response, TripCalculationResponse)
        assert response.mode == "hotel"
        assert response.total_co2e_kg > 0.0

    def test_service_calculate_spend(self, service, spend_request):
        """calculate_spend returns a valid response."""
        response = service.calculate_spend(spend_request)
        assert isinstance(response, TripCalculationResponse)
        # Spend routes through air mode with NAICS data
        assert response.total_co2e_kg >= 0.0

    def test_service_check_compliance(self, service, flight_request):
        """check_compliance returns a ComplianceCheckResponse."""
        # First create a calculation to get an ID
        calc_response = service.calculate_flight(flight_request)
        calc_id = calc_response.calculation_id

        request = ComplianceCheckRequest(
            calculation_id=calc_id,
            frameworks=["GHG_PROTOCOL"],
        )
        response = service.check_compliance(request)
        assert isinstance(response, ComplianceCheckResponse)
        assert response.success is True

    def test_service_get_emission_factors(self, service):
        """get_emission_factors returns air emission factors."""
        response = service.get_emission_factors("air")
        assert isinstance(response, EmissionFactorListResponse)
        assert response.success is True
        assert response.total_count > 0

    def test_service_get_airports(self, service):
        """get_airports returns airports matching a query."""
        airports = service.get_airports("LHR")
        assert len(airports) >= 1
        assert any(a.iata_code == "LHR" for a in airports)

    def test_service_get_transport_modes(self, service):
        """get_transport_modes returns a list with 8 modes."""
        modes = service.get_transport_modes()
        assert len(modes) == 8
        mode_names = [m.mode for m in modes]
        assert "air" in mode_names
        assert "rail" in mode_names
        assert "road" in mode_names
        assert "hotel" in mode_names

    def test_service_get_cabin_classes(self, service):
        """get_cabin_classes returns a list with 4 classes."""
        classes = service.get_cabin_classes()
        assert len(classes) == 4
        class_names = [c.cabin_class for c in classes]
        assert "economy" in class_names
        assert "business" in class_names
        assert "first" in class_names
        assert "premium_economy" in class_names

    def test_get_service_singleton(self):
        """get_service returns the same instance on repeated calls."""
        # Reset the module-level singleton for this test
        import greenlang.business_travel.setup as setup_mod
        original = setup_mod._service_instance
        setup_mod._service_instance = None
        try:
            s1 = get_service()
            s2 = get_service()
            assert s1 is s2
        finally:
            setup_mod._service_instance = original

    def test_get_service_thread_safety(self):
        """get_service is thread-safe under concurrent access."""
        import greenlang.business_travel.setup as setup_mod
        original = setup_mod._service_instance
        setup_mod._service_instance = None
        try:
            services = []
            errors = []

            def fetch():
                try:
                    s = get_service()
                    services.append(id(s))
                except Exception as ex:
                    errors.append(str(ex))

            threads = [threading.Thread(target=fetch) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(set(services)) == 1
        finally:
            setup_mod._service_instance = original

    def test_get_router_returns_router(self):
        """get_router returns a FastAPI APIRouter instance."""
        try:
            from fastapi import APIRouter
            rtr = get_router()
            assert isinstance(rtr, APIRouter)
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_router_has_prefix(self):
        """Router has the expected /api/v1/business-travel prefix."""
        try:
            rtr = get_router()
            assert rtr.prefix == "/api/v1/business-travel"
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_service_all_methods_exist(self, service):
        """Service has all 19 expected public methods."""
        expected_methods = [
            "calculate",
            "calculate_batch",
            "calculate_flight",
            "calculate_rail",
            "calculate_road",
            "calculate_hotel",
            "calculate_spend",
            "check_compliance",
            "analyze_uncertainty",
            "analyze_hot_spots",
            "get_emission_factors",
            "get_calculation",
            "list_calculations",
            "delete_calculation",
            "get_airports",
            "get_transport_modes",
            "get_cabin_classes",
            "get_aggregations",
            "get_provenance",
        ]
        for method_name in expected_methods:
            assert hasattr(service, method_name), f"Missing method: {method_name}"
            assert callable(getattr(service, method_name)), f"Not callable: {method_name}"

    def test_service_calculate_returns_result(self, service, trip_request):
        """Generic calculate method returns a TripCalculationResponse."""
        response = service.calculate(trip_request)
        assert isinstance(response, TripCalculationResponse)
        assert response.calculation_id is not None

    def test_service_batch_returns_result(self, service, trip_request):
        """calculate_batch returns a BatchTripResponse."""
        batch = BatchTripCalculationRequest(
            trips=[trip_request, trip_request],
            reporting_period="2024-Q2",
        )
        response = service.calculate_batch(batch)
        assert isinstance(response, BatchTripResponse)
        assert response.total_trips == 2

    def test_service_get_aggregations(self, service, trip_request):
        """get_aggregations returns an AggregationResponse."""
        # Seed a calculation
        service.calculate(trip_request)
        request = AggregationRequest(
            reporting_period="2024-Q2",
        )
        response = service.get_aggregations(request)
        assert isinstance(response, AggregationResponse)
        assert response.success is True
