# -*- coding: utf-8 -*-
"""
End-to-end integration tests for Mobile Combustion Agent - AGENT-MRV-003

Tests complete workflows: register vehicle -> log trip -> calculate
emissions -> aggregate fleet -> check compliance -> run uncertainty.
Each workflow exercises the full service facade including provenance
tracking, biofuel adjustment, and audit trail generation.

Target: 33+ tests across 8 test classes.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.mobile_combustion.setup import MobileCombustionService
from greenlang.mobile_combustion.mobile_combustion_pipeline import (
    BIOFUEL_FOSSIL_FRACTION,
    FUEL_CO2_FACTORS_KG_PER_GALLON,
    SPEND_BASED_FACTORS_KG_CO2E_PER_USD,
)


# ===================================================================
# TestFuelBasedWorkflow (6 tests)
# ===================================================================


class TestFuelBasedWorkflow:
    """End-to-end fuel-based calculation workflow."""

    def test_register_vehicle_then_calculate(self, service, fuel_based_gasoline):
        """Register a vehicle, then calculate emissions."""
        vid = service.register_vehicle(registration={
            "vehicle_id": "v-e2e-001",
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "name": "E2E Test Car",
        })
        assert vid == "v-e2e-001"

        result = service.calculate(input_data=fuel_based_gasoline)
        assert result.get("total_co2e_kg", 0) > 0
        assert result.get("status") in ("SUCCESS", "PARTIAL")

    def test_log_trip_then_calculate(self, service, fuel_based_gasoline):
        """Log a trip, then calculate emissions."""
        service.register_vehicle(registration={
            "vehicle_id": "v-e2e-002",
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
        })
        tid = service.log_trip(trip={
            "vehicle_id": "v-e2e-002",
            "distance_km": 300.0,
            "fuel_quantity": 25.0,
            "fuel_unit": "LITERS",
            "purpose": "delivery",
        })
        assert tid in service._trips

        result = service.calculate(input_data=fuel_based_gasoline)
        assert result.get("total_co2e_kg", 0) > 0

    def test_calculate_then_aggregate(self, service, fuel_based_gasoline):
        """Calculate, then aggregate fleet emissions."""
        service.calculate(input_data=fuel_based_gasoline)
        agg = service.aggregate_fleet(period="2025-Q1")
        assert agg["total_co2e_tonnes"] > 0.0
        assert agg["calculation_count"] == 1

    def test_full_workflow_vehicle_trip_calc_agg(self, populated_service, fuel_based_gasoline):
        """Full workflow: vehicles + trips + calculation + aggregation."""
        result = populated_service.calculate(input_data=fuel_based_gasoline)
        assert result.get("total_co2e_kg", 0) > 0

        agg = populated_service.aggregate_fleet(period="2025-FY")
        assert agg["total_co2e_tonnes"] > 0.0
        assert agg["vehicle_count"] == 5
        assert agg["trip_count"] == 5

    def test_multiple_calculations_aggregate(self, service):
        """Multiple calculations aggregate correctly."""
        inputs = [
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
                "gwp_source": "AR6",
            },
            {
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 200.0,
                "fuel_unit": "GALLONS",
                "gwp_source": "AR6",
            },
        ]
        results = [service.calculate(input_data=inp) for inp in inputs]
        total_from_calcs = sum(r.get("total_co2e_tonnes", 0) for r in results)

        agg = service.aggregate_fleet(period="2025-Q2")
        assert agg["total_co2e_tonnes"] == pytest.approx(total_from_calcs, rel=1e-3)

    def test_calculation_stored_retrievable(self, service, fuel_based_gasoline):
        """Calculation result can be retrieved from _calculations."""
        result = service.calculate(input_data=fuel_based_gasoline)
        calc_id = result["calculation_id"]
        stored = service._calculations.get(calc_id)
        assert stored is not None
        assert stored.get("total_co2e_kg") == result.get("total_co2e_kg")


# ===================================================================
# TestDistanceBasedWorkflow (5 tests)
# ===================================================================


class TestDistanceBasedWorkflow:
    """End-to-end distance-based calculation workflow."""

    def test_distance_based_calculates_emissions(self, service, distance_based_diesel):
        """Distance-based calculation returns valid emissions."""
        result = service.calculate(input_data=distance_based_diesel)
        assert result.get("total_co2e_kg", 0) > 0

    def test_distance_miles_conversion(self, service):
        """Distance in miles is correctly converted to km."""
        inp = {
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "DISTANCE_BASED",
            "distance": 100.0,
            "distance_unit": "MILES",
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        assert result.get("distance_km", 0) > 0
        # 100 miles * 1.60934 ~ 160.934 km
        assert result.get("distance_km", 0) == pytest.approx(160.934, rel=0.01)

    def test_distance_based_has_fuel_estimate(self, service, distance_based_diesel):
        """Distance-based calc estimates fuel consumption."""
        result = service.calculate(input_data=distance_based_diesel)
        assert result.get("fuel_quantity_gallons", 0) > 0

    def test_distance_based_aggregate(self, service, distance_based_diesel):
        """Distance-based calculation is included in aggregation."""
        service.calculate(input_data=distance_based_diesel)
        agg = service.aggregate_fleet(period="2025-Q3")
        assert agg["total_co2e_tonnes"] > 0.0
        assert "HEAVY_TRUCK_DIESEL" in agg.get("by_vehicle_type", {})

    def test_distance_based_provenance(self, service, distance_based_diesel):
        """Distance-based calculation has provenance hash."""
        result = service.calculate(input_data=distance_based_diesel)
        assert len(result.get("provenance_hash", "")) == 64


# ===================================================================
# TestSpendBasedWorkflow (3 tests)
# ===================================================================


class TestSpendBasedWorkflow:
    """End-to-end spend-based (screening) workflow."""

    def test_spend_based_calculates_emissions(self, service, spend_based_gasoline):
        """Spend-based calculation returns valid emissions."""
        result = service.calculate(input_data=spend_based_gasoline)
        expected_co2e = 500.0 * SPEND_BASED_FACTORS_KG_CO2E_PER_USD["GASOLINE"]
        assert result.get("total_co2e_kg", 0) == pytest.approx(
            expected_co2e, rel=0.01,
        )

    def test_spend_based_with_diesel(self, service):
        """Spend-based calculation with diesel fuel type."""
        inp = {
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "SPEND_BASED",
            "spend_amount": 1000.0,
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        expected = 1000.0 * SPEND_BASED_FACTORS_KG_CO2E_PER_USD["DIESEL"]
        assert result.get("total_co2e_kg", 0) == pytest.approx(expected, rel=0.01)

    def test_spend_based_aggregate(self, service, spend_based_gasoline):
        """Spend-based calculation included in fleet aggregation."""
        service.calculate(input_data=spend_based_gasoline)
        agg = service.aggregate_fleet(period="2025-Q4")
        assert agg["total_co2e_tonnes"] > 0.0


# ===================================================================
# TestBiofuelWorkflow (4 tests)
# ===================================================================


class TestBiofuelWorkflow:
    """End-to-end biofuel blending and biogenic tracking."""

    def test_e10_biogenic_tracking(self, service, biofuel_e10_input):
        """E10 calculation tracks biogenic CO2 separately."""
        result = service.calculate(input_data=biofuel_e10_input)
        assert result.get("biogenic_co2_kg", 0) > 0
        assert result.get("fossil_co2_kg", 0) > 0

    def test_b100_fully_biogenic(self, service):
        """B100 biodiesel: all CO2 is biogenic."""
        inp = {
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "B100",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        assert result.get("fossil_co2_kg", 0) == pytest.approx(0.0, abs=0.01)
        assert result.get("biogenic_co2_kg", 0) > 0

    def test_e85_high_biogenic_fraction(self, service):
        """E85 has high biogenic fraction (~78.5%)."""
        inp = {
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "E85",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 50.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        }
        result = service.calculate(input_data=inp)
        bio_co2 = result.get("biogenic_co2_kg", 0)
        fossil_co2 = result.get("fossil_co2_kg", 0)
        # Biogenic > fossil for E85
        assert bio_co2 > fossil_co2

    def test_biofuel_in_aggregation(self, service, biofuel_e10_input):
        """Biofuel calculations are properly aggregated."""
        service.calculate(input_data=biofuel_e10_input)
        agg = service.aggregate_fleet(period="2025-FY")
        assert agg["total_co2e_tonnes"] > 0.0


# ===================================================================
# TestMultiVehicleWorkflow (5 tests)
# ===================================================================


class TestMultiVehicleWorkflow:
    """End-to-end workflow with multiple vehicle types."""

    def test_mixed_fleet_calculations(self, populated_service):
        """Calculate emissions for different vehicle types in a fleet."""
        inputs = [
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
                "gwp_source": "AR6",
                "facility_id": "hq-london",
            },
            {
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 200.0,
                "fuel_unit": "GALLONS",
                "gwp_source": "AR6",
                "facility_id": "warehouse-birmingham",
            },
            {
                "vehicle_type": "AIRCRAFT",
                "fuel_type": "JET_FUEL",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 3000.0,
                "fuel_unit": "GALLONS",
                "gwp_source": "AR6",
                "facility_id": "hangar-luton",
            },
        ]
        results = [populated_service.calculate(input_data=inp) for inp in inputs]
        assert all(r.get("total_co2e_kg", 0) > 0 for r in results)

    def test_mixed_fleet_aggregation_by_type(self, service):
        """Aggregation breaks down by vehicle type."""
        service.calculate(input_data={
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 50.0,
            "fuel_unit": "GALLONS",
        })
        service.calculate(input_data={
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
        })
        agg = service.aggregate_fleet(period="2025-FY")
        assert len(agg["by_vehicle_type"]) == 2
        assert "PASSENGER_CAR_GASOLINE" in agg["by_vehicle_type"]
        assert "HEAVY_TRUCK_DIESEL" in agg["by_vehicle_type"]

    def test_mixed_fleet_aggregation_by_fuel(self, service):
        """Aggregation breaks down by fuel type."""
        service.calculate(input_data={
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 50.0,
            "fuel_unit": "GALLONS",
        })
        service.calculate(input_data={
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
        })
        agg = service.aggregate_fleet(period="2025-FY")
        assert "GASOLINE" in agg["by_fuel_type"]
        assert "DIESEL" in agg["by_fuel_type"]

    def test_batch_mixed_fleet(self, service):
        """Batch calculation with mixed vehicle types."""
        inputs = [
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 40.0,
                "fuel_unit": "GALLONS",
            },
            {
                "vehicle_type": "BUS_DIESEL",
                "fuel_type": "DIESEL",
                "calculation_method": "DISTANCE_BASED",
                "distance": 300.0,
                "distance_unit": "KM",
            },
        ]
        batch = service.calculate_batch(inputs=inputs)
        assert batch.get("total_co2e_kg", 0) > 0
        assert len(batch.get("results", [])) == 2

    def test_fleet_vehicle_count_in_aggregation(self, populated_service):
        """Aggregation reflects registered vehicle count."""
        populated_service.calculate(input_data={
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 10.0,
            "fuel_unit": "GALLONS",
        })
        agg = populated_service.aggregate_fleet(period="2025-Q1")
        assert agg["vehicle_count"] == 5


# ===================================================================
# TestComplianceWorkflow (4 tests)
# ===================================================================


class TestComplianceWorkflow:
    """End-to-end: calculate + compliance check."""

    def test_calculate_then_compliance(self, service, fuel_based_gasoline):
        """Calculate, then run compliance check."""
        service.calculate(input_data=fuel_based_gasoline)
        comp = service.check_compliance(framework="GHG_PROTOCOL")
        assert comp is not None
        assert "compliant" in comp or "framework" in comp

    def test_compliance_iso_14064(self, service, fuel_based_gasoline):
        """Compliance check against ISO 14064."""
        service.calculate(input_data=fuel_based_gasoline)
        comp = service.check_compliance(framework="ISO_14064")
        assert comp is not None

    def test_compliance_epa(self, service, fuel_based_gasoline):
        """Compliance check against EPA 40CFR98."""
        service.calculate(input_data=fuel_based_gasoline)
        comp = service.check_compliance(framework="EPA_40CFR98")
        assert comp is not None

    def test_compliance_counter(self, service, fuel_based_gasoline):
        """Compliance counter increments with each check."""
        service.calculate(input_data=fuel_based_gasoline)
        service.check_compliance()
        service.check_compliance(framework="ISO_14064")
        assert service._total_compliance_checks >= 2


# ===================================================================
# TestUncertaintyWorkflow (3 tests)
# ===================================================================


class TestUncertaintyWorkflow:
    """End-to-end: calculate + uncertainty analysis."""

    def test_calculate_then_uncertainty(self, service, fuel_based_gasoline):
        """Calculate, then run uncertainty analysis."""
        calc = service.calculate(input_data=fuel_based_gasoline)
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(input_data={"calculation_id": calc_id})
        assert unc.get("mean_co2e_kg", 0) > 0

    def test_uncertainty_percentiles_bracket_mean(self, service, fuel_based_gasoline):
        """p5 <= mean <= p95 in uncertainty results."""
        calc = service.calculate(input_data=fuel_based_gasoline)
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(input_data={"calculation_id": calc_id})
        assert unc["p5_co2e_kg"] <= unc["mean_co2e_kg"]
        assert unc["mean_co2e_kg"] <= unc["p95_co2e_kg"]

    def test_uncertainty_counter(self, service, fuel_based_gasoline):
        """Uncertainty counter increments."""
        calc = service.calculate(input_data=fuel_based_gasoline)
        calc_id = calc["calculation_id"]
        service.run_uncertainty(input_data={"calculation_id": calc_id})
        assert service._total_uncertainty_runs >= 1


# ===================================================================
# TestHealthWorkflow (3 tests)
# ===================================================================


class TestHealthWorkflow:
    """Health check after various operations."""

    def test_health_after_startup(self, service):
        """Health check on fresh service."""
        health = service.health_check()
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert health["vehicle_count"] == 0

    def test_health_after_calculations(self, service, fuel_based_gasoline):
        """Health check after performing calculations."""
        service.calculate(input_data=fuel_based_gasoline)
        health = service.health_check()
        assert health["statistics"]["total_calculations"] == 1

    def test_health_after_fleet_populated(self, populated_service):
        """Health check on populated service shows vehicle and trip counts."""
        health = populated_service.health_check()
        assert health["vehicle_count"] == 5
        assert health["trip_count"] == 5
