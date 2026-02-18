# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for Mobile Combustion Agent unit tests.

Provides reusable fixtures for configuration, provenance tracking,
sample models, and test cleanup across all test modules in this
test package.

AGENT-MRV-003: Mobile Combustion Agent (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List

import pytest

from greenlang.mobile_combustion.config import (
    MobileCombustionConfig,
    get_config,
    reset_config,
    set_config,
)
from greenlang.mobile_combustion.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
)


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------

_ENV_VARS = [
    "GL_MOBILE_COMBUSTION_DATABASE_URL",
    "GL_MOBILE_COMBUSTION_REDIS_URL",
    "GL_MOBILE_COMBUSTION_LOG_LEVEL",
    "GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE",
    "GL_MOBILE_COMBUSTION_DEFAULT_CALCULATION_METHOD",
    "GL_MOBILE_COMBUSTION_MONTE_CARLO_ITERATIONS",
    "GL_MOBILE_COMBUSTION_BATCH_SIZE",
    "GL_MOBILE_COMBUSTION_MAX_BATCH_SIZE",
    "GL_MOBILE_COMBUSTION_CACHE_TTL_SECONDS",
    "GL_MOBILE_COMBUSTION_ENABLE_BIOGENIC_TRACKING",
    "GL_MOBILE_COMBUSTION_ENABLE_UNCERTAINTY",
    "GL_MOBILE_COMBUSTION_ENABLE_COMPLIANCE",
    "GL_MOBILE_COMBUSTION_ENABLE_FLEET_MANAGEMENT",
    "GL_MOBILE_COMBUSTION_DECIMAL_PRECISION",
    "GL_MOBILE_COMBUSTION_DEFAULT_VEHICLE_TYPE",
    "GL_MOBILE_COMBUSTION_DEFAULT_FUEL_TYPE",
    "GL_MOBILE_COMBUSTION_DEFAULT_DISTANCE_UNIT",
    "GL_MOBILE_COMBUSTION_DEFAULT_FUEL_ECONOMY_UNIT",
    "GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_90",
    "GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_95",
    "GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_99",
    "GL_MOBILE_COMBUSTION_DEFAULT_REGULATORY_FRAMEWORK",
    "GL_MOBILE_COMBUSTION_MAX_VEHICLES_PER_FLEET",
    "GL_MOBILE_COMBUSTION_MAX_TRIPS_PER_QUERY",
    "GL_MOBILE_COMBUSTION_CALCULATION_TIMEOUT_SECONDS",
    "GL_MOBILE_COMBUSTION_ENABLE_METRICS",
    "GL_MOBILE_COMBUSTION_ENABLE_TRACING",
    "GL_MOBILE_COMBUSTION_ENABLE_PROVENANCE",
    "GL_MOBILE_COMBUSTION_GENESIS_HASH",
    "GL_MOBILE_COMBUSTION_POOL_SIZE",
    "GL_MOBILE_COMBUSTION_RATE_LIMIT",
]


def _strip_env() -> None:
    """Remove all GL_MOBILE_COMBUSTION_* environment variables."""
    for var in _ENV_VARS:
        os.environ.pop(var, None)
    # Also remove any extra env vars that match the prefix
    prefix = "GL_MOBILE_COMBUSTION_"
    for key in list(os.environ):
        if key.startswith(prefix):
            del os.environ[key]


# ---------------------------------------------------------------------------
# Autouse cleanup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove all GL_MOBILE_COMBUSTION_* env vars, reset singletons.

    Runs before and after every test to prevent environment leakage
    between tests.
    """
    _strip_env()
    reset_config()
    reset_provenance_tracker()
    yield
    _strip_env()
    reset_config()
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> MobileCombustionConfig:
    """Return a MobileCombustionConfig with all defaults."""
    return MobileCombustionConfig()


@pytest.fixture
def custom_config() -> MobileCombustionConfig:
    """Return a MobileCombustionConfig with non-default values.

    Uses AR5, DISTANCE_BASED, reduced batch sizes and iterations,
    disabled fleet management, and different vehicle/fuel defaults.
    """
    return MobileCombustionConfig(
        database_url="postgresql://testhost:5432/testdb",
        redis_url="redis://testhost:6379/1",
        log_level="DEBUG",
        default_gwp_source="AR5",
        default_calculation_method="DISTANCE_BASED",
        monte_carlo_iterations=1_000,
        batch_size=50,
        max_batch_size=500,
        cache_ttl_seconds=1_800,
        enable_biogenic_tracking=False,
        enable_uncertainty=False,
        enable_compliance=False,
        enable_fleet_management=False,
        decimal_precision=4,
        default_vehicle_type="HEAVY_DUTY_TRUCK",
        default_fuel_type="DIESEL",
        default_distance_unit="MILES",
        default_fuel_economy_unit="MPG_US",
        confidence_level_90=0.90,
        confidence_level_95=0.95,
        confidence_level_99=0.99,
        default_regulatory_framework="ISO_14064",
        max_vehicles_per_fleet=5_000,
        max_trips_per_query=2_000,
        calculation_timeout_seconds=60,
        enable_metrics=False,
        enable_tracing=False,
        enable_provenance=False,
        genesis_hash="TEST-MOBILE-COMBUSTION-GENESIS",
        pool_size=5,
        rate_limit=500,
    )


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> ProvenanceTracker:
    """Return a fresh ProvenanceTracker with default genesis hash."""
    return ProvenanceTracker()


@pytest.fixture
def custom_tracker() -> ProvenanceTracker:
    """Return a ProvenanceTracker with a custom genesis hash."""
    return ProvenanceTracker(genesis_hash="CUSTOM-MOBILE-COMBUSTION-GENESIS-TEST")


@pytest.fixture
def populated_tracker() -> ProvenanceTracker:
    """Return a ProvenanceTracker pre-populated with 7+ entries.

    Entries cover a variety of entity types and actions for
    testing filtering, chain verification, and audit trail export.
    """
    t = ProvenanceTracker(genesis_hash="POPULATED-TEST-GENESIS")
    t.record("vehicle", "register", "veh_001", data={"make": "Toyota", "model": "Camry"})
    t.record("vehicle", "register", "veh_002", data={"make": "Ford", "model": "F-150"})
    t.record("trip", "create", "trip_001", data={"distance": 150.0, "unit": "KM"})
    t.record("factor", "read", "ef_001", data={"value": 2.31, "gas": "CO2"})
    t.record("calculation", "calculate", "calc_001", data={"total_co2e": 12.5})
    t.record("batch", "create", "batch_001", data={"count": 10})
    t.record("fleet", "aggregate", "fleet_001", data={"vehicles": 25})
    t.record("compliance", "check", "comp_001", data={"framework": "GHG_PROTOCOL"})
    t.record("uncertainty", "analyze", "unc_001", data={"iterations": 5000})
    return t


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def utcnow() -> datetime:
    """Return a fixed UTC datetime for deterministic testing."""
    return datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_vehicle_registration(utcnow: datetime) -> Dict[str, Any]:
    """Return a dictionary suitable for VehicleRegistration construction."""
    return {
        "vehicle_id": "veh_test_001",
        "vin": "1HGCM82633A123456",
        "make": "Toyota",
        "model": "Camry Hybrid",
        "model_year": 2024,
        "vehicle_type": "PASSENGER_CAR_HYBRID",
        "fuel_type": "GASOLINE",
        "emission_control": "THREE_WAY_CATALYST",
        "department": "Corporate Fleet",
        "fleet_id": "fleet_hq_001",
        "status": "ACTIVE",
        "odometer_km": 15000.5,
        "registration_date": utcnow,
        "disposal_date": None,
        "notes": "Test vehicle for unit testing",
    }


@pytest.fixture
def sample_trip_record(utcnow: datetime) -> Dict[str, Any]:
    """Return a dictionary suitable for TripRecord construction."""
    return {
        "trip_id": "trip_test_001",
        "vehicle_id": "veh_test_001",
        "distance_value": 245.7,
        "distance_unit": "KM",
        "fuel_consumed_liters": 18.5,
        "fuel_type": "GASOLINE",
        "start_time": utcnow,
        "end_time": utcnow + timedelta(hours=3),
        "start_location": "Headquarters, London",
        "end_location": "Warehouse, Birmingham",
        "route_description": "M40 motorway via Oxford",
        "purpose": "Business delivery",
        "status": "COMPLETED",
        "driver_id": "drv_001",
        "cargo_weight_kg": 500.0,
        "passengers": 2,
    }


@pytest.fixture
def sample_calculation_input(utcnow: datetime) -> Dict[str, Any]:
    """Return a dictionary suitable for CalculationInput construction."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "quantity": 50.0,
        "unit": "LITERS",
        "calculation_method": "FUEL_BASED",
        "fuel_economy_value": None,
        "fuel_economy_unit": "L_PER_100KM",
        "vehicle_id": "veh_test_001",
        "trip_id": "trip_test_001",
        "fleet_id": "fleet_hq_001",
        "period_start": utcnow,
        "period_end": utcnow + timedelta(days=30),
        "emission_control": "THREE_WAY_CATALYST",
        "model_year": 2024,
        "geography": "GB",
        "tier": "TIER_2",
        "gwp_source": "AR6",
        "custom_emission_factor_co2": None,
        "custom_emission_factor_ch4": None,
        "custom_emission_factor_n2o": None,
    }
