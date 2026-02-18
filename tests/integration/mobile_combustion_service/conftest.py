# -*- coding: utf-8 -*-
"""
Integration test fixtures for Mobile Combustion Agent - AGENT-MRV-003

Provides shared fixtures for end-to-end and full pipeline integration
tests: fresh and pre-populated service instances, sample vehicle
registrations, sample trips, and calculation input generators.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator, List

import pytest

from greenlang.mobile_combustion.setup import MobileCombustionService


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that are not relevant to
# mobile combustion tests and may cause import/patching errors.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent mock_agents fixture (no-op for mobile combustion).

    The parent ``tests/integration/conftest.py`` defines an autouse
    ``mock_agents`` fixture that patches ``greenlang.agents.registry.get_agent``.
    That fixture is irrelevant here and may fail if the module path does not
    exist.  Defining the same fixture name in a child conftest overrides it.
    """
    yield {}


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_MOBILE_COMBUSTION_"


def _strip_env() -> None:
    """Remove all GL_MOBILE_COMBUSTION_* environment variables."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIX):
            del os.environ[key]


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove mobile combustion env vars and reset singletons."""
    _strip_env()

    # Reset singleton state
    import greenlang.mobile_combustion.setup as setup_mod
    setup_mod._singleton_instance = None
    setup_mod._service = None

    try:
        from greenlang.mobile_combustion.config import reset_config
        reset_config()
    except ImportError:
        pass

    try:
        from greenlang.mobile_combustion.provenance import reset_provenance_tracker
        reset_provenance_tracker()
    except ImportError:
        pass

    yield

    _strip_env()
    setup_mod._singleton_instance = None
    setup_mod._service = None

    try:
        from greenlang.mobile_combustion.config import reset_config
        reset_config()
    except ImportError:
        pass

    try:
        from greenlang.mobile_combustion.provenance import reset_provenance_tracker
        reset_provenance_tracker()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> MobileCombustionService:
    """Create a fresh MobileCombustionService instance.

    Returns:
        Empty MobileCombustionService with no vehicles, trips, or
        calculations.
    """
    return MobileCombustionService()


@pytest.fixture
def sample_vehicles() -> List[Dict[str, Any]]:
    """List of 5 vehicle registrations for integration testing.

    Covers: passenger car, heavy truck, bus, forklift (off-road),
    and corporate jet (aircraft).
    """
    return [
        {
            "vehicle_id": "v-car-001",
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "name": "Executive Sedan",
            "make": "Toyota",
            "model": "Camry",
            "year": 2024,
            "facility_id": "hq-london",
            "fleet_id": "fleet-exec",
            "fuel_economy": 8.5,
            "fuel_economy_unit": "L_PER_100KM",
            "odometer_km": 15000.0,
        },
        {
            "vehicle_id": "v-truck-001",
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "name": "Delivery Truck",
            "make": "Volvo",
            "model": "FH16",
            "year": 2023,
            "facility_id": "warehouse-birmingham",
            "fleet_id": "fleet-logistics",
            "fuel_economy": 32.0,
            "fuel_economy_unit": "L_PER_100KM",
            "odometer_km": 120000.0,
        },
        {
            "vehicle_id": "v-bus-001",
            "vehicle_type": "BUS_DIESEL",
            "fuel_type": "DIESEL",
            "name": "Employee Shuttle",
            "make": "Mercedes",
            "model": "Citaro",
            "year": 2022,
            "facility_id": "hq-london",
            "fleet_id": "fleet-transport",
            "fuel_economy": 28.0,
            "fuel_economy_unit": "L_PER_100KM",
            "odometer_km": 85000.0,
        },
        {
            "vehicle_id": "v-forklift-001",
            "vehicle_type": "OFF_ROAD_VEHICLE",
            "fuel_type": "LPG",
            "name": "Warehouse Forklift",
            "make": "Toyota",
            "model": "8FGU25",
            "year": 2021,
            "facility_id": "warehouse-birmingham",
            "fleet_id": "fleet-logistics",
            "odometer_km": 5000.0,
        },
        {
            "vehicle_id": "v-jet-001",
            "vehicle_type": "AIRCRAFT",
            "fuel_type": "JET_FUEL",
            "name": "Corporate Jet",
            "make": "Gulfstream",
            "model": "G650",
            "year": 2020,
            "facility_id": "hangar-luton",
            "fleet_id": "fleet-exec",
            "odometer_km": 500000.0,
        },
    ]


@pytest.fixture
def sample_trips() -> List[Dict[str, Any]]:
    """List of trips for the sample vehicles.

    Covers a variety of distances, purposes, and fuel quantities.
    """
    return [
        {
            "trip_id": "t-car-001",
            "vehicle_id": "v-car-001",
            "distance_km": 200.0,
            "distance_unit": "KM",
            "fuel_quantity": 17.0,
            "fuel_unit": "LITERS",
            "start_date": "2026-01-10T08:00:00Z",
            "end_date": "2026-01-10T11:00:00Z",
            "origin": "London HQ",
            "destination": "Oxford Office",
            "purpose": "client_meeting",
        },
        {
            "trip_id": "t-truck-001",
            "vehicle_id": "v-truck-001",
            "distance_km": 450.0,
            "distance_unit": "KM",
            "fuel_quantity": 144.0,
            "fuel_unit": "LITERS",
            "start_date": "2026-01-11T06:00:00Z",
            "end_date": "2026-01-11T14:00:00Z",
            "origin": "Birmingham Warehouse",
            "destination": "Edinburgh Distribution",
            "purpose": "delivery",
        },
        {
            "trip_id": "t-bus-001",
            "vehicle_id": "v-bus-001",
            "distance_km": 50.0,
            "distance_unit": "KM",
            "fuel_quantity": 14.0,
            "fuel_unit": "LITERS",
            "start_date": "2026-01-12T07:30:00Z",
            "end_date": "2026-01-12T08:30:00Z",
            "origin": "London HQ",
            "destination": "London Office",
            "purpose": "employee_transport",
        },
        {
            "trip_id": "t-forklift-001",
            "vehicle_id": "v-forklift-001",
            "distance_km": 5.0,
            "distance_unit": "KM",
            "fuel_quantity": 10.0,
            "fuel_unit": "LITERS",
            "start_date": "2026-01-13T09:00:00Z",
            "end_date": "2026-01-13T17:00:00Z",
            "origin": "Warehouse Bay A",
            "destination": "Warehouse Bay D",
            "purpose": "warehouse_operations",
        },
        {
            "trip_id": "t-jet-001",
            "vehicle_id": "v-jet-001",
            "distance_km": 5500.0,
            "distance_unit": "KM",
            "fuel_quantity": 4500.0,
            "fuel_unit": "LITERS",
            "start_date": "2026-01-15T10:00:00Z",
            "end_date": "2026-01-15T17:00:00Z",
            "origin": "Luton Airport",
            "destination": "JFK New York",
            "purpose": "business_travel",
        },
    ]


@pytest.fixture
def populated_service(
    service: MobileCombustionService,
    sample_vehicles: List[Dict[str, Any]],
    sample_trips: List[Dict[str, Any]],
) -> MobileCombustionService:
    """Service pre-populated with 5 vehicles and 5 trips.

    Args:
        service: Fresh MobileCombustionService.
        sample_vehicles: List of 5 vehicle registrations.
        sample_trips: List of 5 trip records.

    Returns:
        MobileCombustionService with vehicles and trips registered.
    """
    for vehicle in sample_vehicles:
        service.register_vehicle(registration=vehicle)

    for trip in sample_trips:
        service.log_trip(trip=trip)

    return service


# ---------------------------------------------------------------------------
# Calculation input generators
# ---------------------------------------------------------------------------


@pytest.fixture
def fuel_based_gasoline() -> Dict[str, Any]:
    """Fuel-based gasoline calculation input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 100.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def fuel_based_diesel() -> Dict[str, Any]:
    """Fuel-based diesel calculation input for heavy truck."""
    return {
        "vehicle_type": "HEAVY_TRUCK_DIESEL",
        "fuel_type": "DIESEL",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 500.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def distance_based_diesel() -> Dict[str, Any]:
    """Distance-based diesel calculation input."""
    return {
        "vehicle_type": "HEAVY_TRUCK_DIESEL",
        "fuel_type": "DIESEL",
        "calculation_method": "DISTANCE_BASED",
        "distance": 1000.0,
        "distance_unit": "KM",
        "gwp_source": "AR6",
    }


@pytest.fixture
def spend_based_gasoline() -> Dict[str, Any]:
    """Spend-based gasoline calculation input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "calculation_method": "SPEND_BASED",
        "spend_amount": 500.0,
        "spend_currency": "USD",
        "gwp_source": "AR6",
    }


@pytest.fixture
def biofuel_e10_input() -> Dict[str, Any]:
    """E10 biofuel blend input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "E10",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 100.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def jet_fuel_input() -> Dict[str, Any]:
    """Jet fuel calculation input for aircraft."""
    return {
        "vehicle_type": "AIRCRAFT",
        "fuel_type": "JET_FUEL",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 5000.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }
