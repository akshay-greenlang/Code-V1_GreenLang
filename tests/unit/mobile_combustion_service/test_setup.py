# -*- coding: utf-8 -*-
"""
Unit tests for MobileCombustionService facade (setup.py) - AGENT-MRV-003

Tests the unified service facade that aggregates all seven engines
through a single entry point with convenience methods for calculations,
vehicle/trip management, fleet aggregation, uncertainty analysis,
compliance checking, and health monitoring.

Target: 62+ tests across 14 test classes.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.mobile_combustion.setup import (
    MobileCombustionService,
    CalculateResponse,
    BatchCalculateResponse,
    VehicleResponse,
    TripResponse,
    AggregationResponse,
    UncertaintyResponse,
    ComplianceResponse,
    HealthResponse,
    StatsResponse,
    _compute_hash,
    get_service,
    get_router,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def service() -> MobileCombustionService:
    """Create a fresh MobileCombustionService instance."""
    return MobileCombustionService()


@pytest.fixture
def fuel_based_input() -> Dict[str, Any]:
    """Standard fuel-based calculation input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 100.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def distance_based_input() -> Dict[str, Any]:
    """Standard distance-based calculation input."""
    return {
        "vehicle_type": "HEAVY_TRUCK_DIESEL",
        "fuel_type": "DIESEL",
        "calculation_method": "DISTANCE_BASED",
        "distance": 500.0,
        "distance_unit": "MILES",
        "gwp_source": "AR6",
    }


@pytest.fixture
def spend_based_input() -> Dict[str, Any]:
    """Standard spend-based calculation input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "calculation_method": "SPEND_BASED",
        "spend_amount": 1000.0,
        "spend_currency": "USD",
        "gwp_source": "AR6",
    }


@pytest.fixture
def sample_vehicle() -> Dict[str, Any]:
    """Sample vehicle registration data."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "name": "Company Car #1",
        "make": "Toyota",
        "model": "Camry",
        "year": 2024,
        "facility_id": "facility-001",
        "fleet_id": "fleet-hq",
        "fuel_economy": 8.9,
        "fuel_economy_unit": "L_PER_100KM",
        "odometer_km": 15000.0,
    }


@pytest.fixture
def sample_trip() -> Dict[str, Any]:
    """Sample trip data."""
    return {
        "vehicle_id": "veh-001",
        "distance_km": 200.0,
        "distance_unit": "KM",
        "fuel_quantity": 15.0,
        "fuel_unit": "LITERS",
        "start_date": "2026-01-15T08:00:00Z",
        "end_date": "2026-01-15T11:00:00Z",
        "origin": "London HQ",
        "destination": "Birmingham Warehouse",
        "purpose": "delivery",
    }


# ===================================================================
# TestServiceInit (5 tests)
# ===================================================================


class TestServiceInit:
    """Test MobileCombustionService initialization."""

    def test_service_creation(self):
        """Service can be created with no arguments."""
        svc = MobileCombustionService()
        assert svc is not None

    def test_service_has_pipeline_engine(self, service):
        """Service initializes with a pipeline engine."""
        assert service.pipeline_engine is not None

    def test_service_stores_are_empty(self, service):
        """In-memory stores are empty on creation."""
        assert len(service._calculations) == 0
        assert len(service._vehicles) == 0
        assert len(service._trips) == 0
        assert len(service._fuels) == 0
        assert len(service._aggregations) == 0

    def test_service_counters_at_zero(self, service):
        """Statistics counters start at zero."""
        assert service._total_calculations == 0
        assert service._total_batch_runs == 0
        assert service._total_compliance_checks == 0
        assert service._total_uncertainty_runs == 0

    def test_service_config_stored(self):
        """Config passed to service is stored."""
        mock_config = MagicMock()
        mock_config.genesis_hash = "TEST-GENESIS"
        svc = MobileCombustionService(config=mock_config)
        assert svc.config is mock_config


# ===================================================================
# TestCalculate (10 tests)
# ===================================================================


class TestCalculate:
    """Test single calculation via pipeline."""

    def test_fuel_based_gasoline(self, service, fuel_based_input):
        """Fuel-based gasoline calculation returns valid result."""
        result = service.calculate(input_data=fuel_based_input)
        assert isinstance(result, dict)
        assert result.get("total_co2e_kg", 0) > 0
        assert result.get("total_co2e_tonnes", 0) > 0

    def test_distance_based_diesel(self, service, distance_based_input):
        """Distance-based diesel calculation returns valid result."""
        result = service.calculate(input_data=distance_based_input)
        assert result.get("total_co2e_kg", 0) > 0

    def test_spend_based_calculation(self, service, spend_based_input):
        """Spend-based calculation returns valid result."""
        result = service.calculate(input_data=spend_based_input)
        assert result.get("total_co2e_kg", 0) > 0

    def test_calculation_cached(self, service, fuel_based_input):
        """Calculation result is cached in _calculations."""
        result = service.calculate(input_data=fuel_based_input)
        calc_id = result.get("calculation_id", "")
        assert calc_id in service._calculations

    def test_calculation_has_processing_time(self, service, fuel_based_input):
        """Result includes processing_time_ms."""
        result = service.calculate(input_data=fuel_based_input)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] > 0.0

    def test_calculation_has_provenance_hash(self, service, fuel_based_input):
        """Result includes a SHA-256 provenance hash."""
        result = service.calculate(input_data=fuel_based_input)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_calculation_counter_increments(self, service, fuel_based_input):
        """_total_calculations increments after each calculation."""
        assert service._total_calculations == 0
        service.calculate(input_data=fuel_based_input)
        assert service._total_calculations == 1
        service.calculate(input_data=fuel_based_input)
        assert service._total_calculations == 2

    def test_heavy_truck_diesel(self, service):
        """Heavy truck diesel calculation produces expected range."""
        inp = {
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 500.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        # 500 gal * 10.180 kg/gal = ~5090 kg CO2
        assert result.get("co2_kg", 0) > 5000

    def test_motorcycle_gasoline(self, service):
        """Motorcycle gasoline calculation succeeds."""
        inp = {
            "vehicle_type": "MOTORCYCLE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 5.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        assert result.get("total_co2e_kg", 0) > 0

    def test_bus_cng_calculation(self, service):
        """CNG bus calculation succeeds with non-zero emissions."""
        inp = {
            "vehicle_type": "BUS_CNG",
            "fuel_type": "CNG",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 1000.0,
            "fuel_unit": "CUBIC_FEET",
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        assert result.get("total_co2e_kg", 0) > 0


# ===================================================================
# TestBatchCalculate (5 tests)
# ===================================================================


class TestBatchCalculate:
    """Test batch calculation."""

    def test_batch_two_inputs(self, service, fuel_based_input, distance_based_input):
        """Batch with two inputs returns results for both."""
        result = service.calculate_batch(
            inputs=[fuel_based_input, distance_based_input],
        )
        assert "results" in result
        assert len(result["results"]) == 2

    def test_batch_aggregated_totals(self, service, fuel_based_input):
        """Batch summary contains aggregated totals."""
        result = service.calculate_batch(
            inputs=[fuel_based_input, fuel_based_input],
        )
        assert result.get("total_co2e_kg", 0) > 0
        assert result.get("total_co2e_tonnes", 0) > 0

    def test_batch_counter_increments(self, service, fuel_based_input):
        """_total_batch_runs increments after batch."""
        assert service._total_batch_runs == 0
        service.calculate_batch(inputs=[fuel_based_input])
        assert service._total_batch_runs == 1

    def test_batch_has_batch_id(self, service, fuel_based_input):
        """Batch result includes batch_id."""
        result = service.calculate_batch(inputs=[fuel_based_input])
        assert "batch_id" in result
        assert len(result["batch_id"]) > 0

    def test_batch_has_provenance_hash(self, service, fuel_based_input):
        """Batch result includes provenance_hash."""
        result = service.calculate_batch(inputs=[fuel_based_input])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===================================================================
# TestVehicleManagement (8 tests)
# ===================================================================


class TestVehicleManagement:
    """Test vehicle registration and retrieval."""

    def test_register_vehicle(self, service, sample_vehicle):
        """Registering a vehicle returns a vehicle_id."""
        vid = service.register_vehicle(registration=sample_vehicle)
        assert isinstance(vid, str)
        assert len(vid) > 0

    def test_register_vehicle_cached(self, service, sample_vehicle):
        """Registered vehicle is stored in _vehicles."""
        vid = service.register_vehicle(registration=sample_vehicle)
        assert vid in service._vehicles

    def test_get_registered_vehicle(self, service, sample_vehicle):
        """Registered vehicle can be retrieved by ID."""
        vid = service.register_vehicle(registration=sample_vehicle)
        vehicle = service.get_vehicle(vid)
        assert vehicle.get("vehicle_type") == "PASSENGER_CAR_GASOLINE"
        assert vehicle.get("fuel_type") == "GASOLINE"
        assert vehicle.get("make") == "Toyota"

    def test_get_nonexistent_vehicle(self, service):
        """Getting a nonexistent vehicle returns error."""
        vehicle = service.get_vehicle("no-such-vehicle")
        assert vehicle.get("error") is not None

    def test_list_vehicles_empty(self, service):
        """Listing vehicles on empty store returns empty list."""
        vehicles = service.list_vehicles()
        assert vehicles == []

    def test_list_vehicles_populated(self, service, sample_vehicle):
        """Listing vehicles after registration returns registered vehicles."""
        service.register_vehicle(registration=sample_vehicle)
        vehicles = service.list_vehicles()
        assert len(vehicles) == 1

    def test_list_vehicles_filter_by_type(self, service, sample_vehicle):
        """Listing vehicles with vehicle_type filter works."""
        service.register_vehicle(registration=sample_vehicle)
        diesel_reg = dict(sample_vehicle)
        diesel_reg["vehicle_type"] = "HEAVY_TRUCK_DIESEL"
        diesel_reg["fuel_type"] = "DIESEL"
        diesel_reg.pop("vehicle_id", None)
        service.register_vehicle(registration=diesel_reg)

        filtered = service.list_vehicles(
            filters={"vehicle_type": "HEAVY_TRUCK_DIESEL"},
        )
        assert len(filtered) == 1
        assert filtered[0]["vehicle_type"] == "HEAVY_TRUCK_DIESEL"

    def test_register_vehicle_has_provenance(self, service, sample_vehicle):
        """Registered vehicle has a provenance_hash."""
        vid = service.register_vehicle(registration=sample_vehicle)
        vehicle = service.get_vehicle(vid)
        assert "provenance_hash" in vehicle
        assert len(vehicle["provenance_hash"]) == 64


# ===================================================================
# TestTripManagement (6 tests)
# ===================================================================


class TestTripManagement:
    """Test trip logging and retrieval."""

    def test_log_trip(self, service, sample_trip):
        """Logging a trip returns a trip_id."""
        tid = service.log_trip(trip=sample_trip)
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_log_trip_cached(self, service, sample_trip):
        """Logged trip is stored in _trips."""
        tid = service.log_trip(trip=sample_trip)
        assert tid in service._trips

    def test_trip_has_correct_distance(self, service, sample_trip):
        """Logged trip preserves distance_km."""
        tid = service.log_trip(trip=sample_trip)
        trip = service._trips[tid]
        assert trip["distance_km"] == pytest.approx(200.0)

    def test_trip_has_vehicle_id(self, service, sample_trip):
        """Logged trip preserves vehicle_id."""
        tid = service.log_trip(trip=sample_trip)
        trip = service._trips[tid]
        assert trip["vehicle_id"] == "veh-001"

    def test_trip_has_provenance(self, service, sample_trip):
        """Logged trip has a provenance_hash."""
        tid = service.log_trip(trip=sample_trip)
        trip = service._trips[tid]
        assert "provenance_hash" in trip
        assert len(trip["provenance_hash"]) == 64

    def test_multiple_trips(self, service, sample_trip):
        """Multiple trips can be logged."""
        tid1 = service.log_trip(trip=sample_trip)
        trip2 = dict(sample_trip)
        trip2["distance_km"] = 500.0
        trip2.pop("trip_id", None)
        tid2 = service.log_trip(trip=trip2)
        assert tid1 != tid2
        assert len(service._trips) == 2


# ===================================================================
# TestFleetAggregation (6 tests)
# ===================================================================


class TestFleetAggregation:
    """Test fleet aggregation methods."""

    def test_aggregate_empty_fleet(self, service):
        """Aggregating with no calculations returns zero totals."""
        agg = service.aggregate_fleet(period="2025-Q1")
        assert agg["total_co2e_tonnes"] == 0.0
        assert agg["period"] == "2025-Q1"

    def test_aggregate_with_calculations(self, service, fuel_based_input):
        """Aggregation includes prior calculation results."""
        service.calculate(input_data=fuel_based_input)
        agg = service.aggregate_fleet(period="2025-Q1")
        assert agg["total_co2e_tonnes"] > 0.0

    def test_aggregate_by_vehicle_type(self, service, fuel_based_input, distance_based_input):
        """Aggregation groups by vehicle type."""
        service.calculate(input_data=fuel_based_input)
        service.calculate(input_data=distance_based_input)
        agg = service.aggregate_fleet(period="2025-Q1")
        assert "by_vehicle_type" in agg
        assert len(agg["by_vehicle_type"]) >= 1

    def test_aggregate_by_fuel_type(self, service, fuel_based_input):
        """Aggregation groups by fuel type."""
        service.calculate(input_data=fuel_based_input)
        agg = service.aggregate_fleet(period="2025-Q1")
        assert "by_fuel_type" in agg
        assert "GASOLINE" in agg["by_fuel_type"]

    def test_aggregate_has_provenance(self, service, fuel_based_input):
        """Aggregation result includes provenance_hash."""
        service.calculate(input_data=fuel_based_input)
        agg = service.aggregate_fleet(period="2025-Q1")
        assert "provenance_hash" in agg
        assert len(agg["provenance_hash"]) == 64

    def test_aggregate_cached(self, service, fuel_based_input):
        """Aggregation result is cached in _aggregations."""
        service.calculate(input_data=fuel_based_input)
        agg = service.aggregate_fleet(period="2025-Q1")
        agg_id = agg["aggregation_id"]
        assert agg_id in service._aggregations


# ===================================================================
# TestUncertainty (4 tests)
# ===================================================================


class TestUncertainty:
    """Test uncertainty analysis via service."""

    def test_uncertainty_on_calculation(self, service, fuel_based_input):
        """Uncertainty analysis on an existing calculation."""
        calc = service.calculate(input_data=fuel_based_input)
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(input_data={"calculation_id": calc_id})
        assert unc.get("mean_co2e_kg", 0) > 0

    def test_uncertainty_on_missing_calculation(self, service):
        """Uncertainty on nonexistent calculation returns error."""
        unc = service.run_uncertainty(
            input_data={"calculation_id": "no-such-id"},
        )
        assert "error" in unc

    def test_uncertainty_counter_increments(self, service, fuel_based_input):
        """_total_uncertainty_runs increments."""
        calc = service.calculate(input_data=fuel_based_input)
        calc_id = calc["calculation_id"]
        service.run_uncertainty(input_data={"calculation_id": calc_id})
        assert service._total_uncertainty_runs >= 1

    def test_uncertainty_has_percentiles(self, service, fuel_based_input):
        """Uncertainty result has p5 and p95 percentiles."""
        calc = service.calculate(input_data=fuel_based_input)
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(input_data={"calculation_id": calc_id})
        assert "p5_co2e_kg" in unc
        assert "p95_co2e_kg" in unc


# ===================================================================
# TestCompliance (4 tests)
# ===================================================================


class TestCompliance:
    """Test compliance checking via service."""

    def test_compliance_check_default_framework(self, service, fuel_based_input):
        """Compliance check with default framework (GHG_PROTOCOL)."""
        service.calculate(input_data=fuel_based_input)
        comp = service.check_compliance()
        assert comp is not None
        assert "framework" in comp or "compliant" in comp

    def test_compliance_check_iso_14064(self, service, fuel_based_input):
        """Compliance check with ISO 14064 framework."""
        service.calculate(input_data=fuel_based_input)
        comp = service.check_compliance(framework="ISO_14064")
        assert comp is not None

    def test_compliance_counter_increments(self, service, fuel_based_input):
        """_total_compliance_checks increments."""
        service.calculate(input_data=fuel_based_input)
        service.check_compliance()
        assert service._total_compliance_checks >= 1

    def test_compliance_with_specific_results(self, service, fuel_based_input):
        """Compliance check with explicitly passed results."""
        result = service.calculate(input_data=fuel_based_input)
        calc_id = result["calculation_id"]
        comp = service.check_compliance(
            results=[service._calculations[calc_id]],
            framework="GHG_PROTOCOL",
        )
        assert comp is not None


# ===================================================================
# TestFuelTypes (3 tests)
# ===================================================================


class TestFuelTypes:
    """Test fuel type listing."""

    def test_get_fuel_types_returns_list(self, service):
        """get_fuel_types returns a non-empty list."""
        fuels = service.get_fuel_types()
        assert isinstance(fuels, list)
        assert len(fuels) > 0

    def test_fuel_types_include_gasoline(self, service):
        """Default fuel types include GASOLINE."""
        fuels = service.get_fuel_types()
        fuel_names = [f["fuel_type"] for f in fuels]
        assert "GASOLINE" in fuel_names

    def test_fuel_types_include_biofuels(self, service):
        """Default fuel types include biofuel blends."""
        fuels = service.get_fuel_types()
        fuel_names = [f["fuel_type"] for f in fuels]
        assert "E10" in fuel_names
        assert "E85" in fuel_names
        assert "B20" in fuel_names


# ===================================================================
# TestVehicleTypes (3 tests)
# ===================================================================


class TestVehicleTypes:
    """Test vehicle type listing."""

    def test_get_vehicle_types_returns_list(self, service):
        """get_vehicle_types returns a non-empty list."""
        types = service.get_vehicle_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_vehicle_types_include_car(self, service):
        """Default vehicle types include PASSENGER_CAR_GASOLINE."""
        types = service.get_vehicle_types()
        type_names = [t["vehicle_type"] for t in types]
        assert "PASSENGER_CAR_GASOLINE" in type_names

    def test_vehicle_types_include_heavy_truck(self, service):
        """Default vehicle types include HEAVY_TRUCK_DIESEL."""
        types = service.get_vehicle_types()
        type_names = [t["vehicle_type"] for t in types]
        assert "HEAVY_TRUCK_DIESEL" in type_names


# ===================================================================
# TestEmissionFactors (3 tests)
# ===================================================================


class TestEmissionFactors:
    """Test emission factor retrieval."""

    def test_get_emission_factors_returns_list(self, service):
        """get_emission_factors returns a non-empty list."""
        factors = service.get_emission_factors()
        assert isinstance(factors, list)
        assert len(factors) > 0

    def test_factors_have_fuel_type(self, service):
        """Each factor record has a fuel_type field."""
        factors = service.get_emission_factors()
        for f in factors:
            assert "fuel_type" in f

    def test_factors_filter_by_fuel_type(self, service):
        """Factors can be filtered by fuel_type."""
        factors = service.get_emission_factors(
            filters={"fuel_type": "DIESEL"},
        )
        for f in factors:
            assert f["fuel_type"] == "DIESEL"


# ===================================================================
# TestHealthCheck (2 tests)
# ===================================================================


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_status(self, service):
        """Health check returns a status field."""
        health = service.health_check()
        assert "status" in health
        assert health["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_check_has_engines(self, service):
        """Health check includes engine availability."""
        health = service.health_check()
        assert "engines" in health
        assert "engines_available" in health
        assert "engines_total" in health


# ===================================================================
# TestStats (2 tests)
# ===================================================================


class TestStats:
    """Test service statistics."""

    def test_get_stats_returns_dict(self, service):
        """get_stats returns a dictionary with expected keys."""
        stats = service.get_stats()
        assert "total_calculations" in stats
        assert "total_batch_runs" in stats
        assert "total_vehicles" in stats
        assert "total_trips" in stats
        assert "total_fuel_types" in stats

    def test_stats_reflect_calculations(self, service, fuel_based_input):
        """Stats reflect the number of calculations performed."""
        service.calculate(input_data=fuel_based_input)
        service.calculate(input_data=fuel_based_input)
        stats = service.get_stats()
        assert stats["total_calculations"] == 2


# ===================================================================
# TestModuleFunctions (4 tests)
# ===================================================================


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_service_returns_instance(self):
        """get_service returns a MobileCombustionService."""
        # Reset singleton state
        import greenlang.mobile_combustion.setup as setup_mod
        setup_mod._singleton_instance = None
        setup_mod._service = None
        svc = get_service()
        assert isinstance(svc, MobileCombustionService)
        # Clean up singleton
        setup_mod._singleton_instance = None
        setup_mod._service = None

    def test_get_router_returns_router(self):
        """get_router returns the API router or None."""
        router = get_router()
        # Should return a router if FastAPI is available
        if router is not None:
            assert hasattr(router, "routes")

    def test_configure_mobile_combustion(self):
        """configure_mobile_combustion creates service and mounts router."""
        import asyncio
        from greenlang.mobile_combustion.setup import configure_mobile_combustion
        import greenlang.mobile_combustion.setup as setup_mod

        # Reset singleton
        setup_mod._singleton_instance = None
        setup_mod._service = None

        try:
            from fastapi import FastAPI
            app = FastAPI()
            svc = asyncio.run(configure_mobile_combustion(app))
            assert isinstance(svc, MobileCombustionService)
            assert svc._started is True
            assert hasattr(app.state, "mobile_combustion_service")
        except ImportError:
            pytest.skip("FastAPI not available")
        finally:
            setup_mod._singleton_instance = None
            setup_mod._service = None

    def test_compute_hash_deterministic(self):
        """_compute_hash produces deterministic SHA-256 output."""
        data = {"vehicle_type": "PASSENGER_CAR", "fuel": "GASOLINE"}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64
