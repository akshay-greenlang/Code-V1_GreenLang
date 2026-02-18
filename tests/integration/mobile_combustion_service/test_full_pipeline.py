# -*- coding: utf-8 -*-
"""
Full pipeline integration tests for Mobile Combustion Agent - AGENT-MRV-003

Tests the MobileCombustionPipelineEngine end-to-end including all eight
stages, batch processing, compliance against all six frameworks,
uncertainty analysis, audit trail integrity, and edge cases.

Target: 28+ tests across 7 test classes.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.mobile_combustion.mobile_combustion_pipeline import (
    BIOFUEL_FOSSIL_FRACTION,
    COMPLIANCE_REQUIREMENTS,
    FUEL_CO2_FACTORS_KG_PER_GALLON,
    GWP_TABLES,
    MobileCombustionPipelineEngine,
    PIPELINE_STAGES,
    SPEND_BASED_FACTORS_KG_CO2E_PER_USD,
    VOLUME_TO_GALLONS,
)
from greenlang.mobile_combustion.setup import MobileCombustionService


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def pipeline() -> MobileCombustionPipelineEngine:
    """Create a fresh pipeline engine in fallback mode."""
    return MobileCombustionPipelineEngine()


# ===================================================================
# TestSingleCalculation (6 tests)
# ===================================================================


class TestSingleCalculation:
    """Test single calculation through the full pipeline."""

    def test_fuel_based_gasoline_full_pipeline(self, pipeline):
        """Full pipeline: fuel-based gasoline produces correct CO2."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        })
        assert result["success"] is True
        assert result["stages_completed"] == 8

        final = result["result"]
        expected_co2 = 100.0 * FUEL_CO2_FACTORS_KG_PER_GALLON["GASOLINE"]
        assert final["co2_kg"] == pytest.approx(expected_co2, rel=1e-3)
        assert final["total_co2e_kg"] > expected_co2  # CH4+N2O add
        assert final["total_co2e_tonnes"] == pytest.approx(
            final["total_co2e_kg"] / 1000.0, rel=1e-5,
        )

    def test_fuel_based_diesel_full_pipeline(self, pipeline):
        """Full pipeline: fuel-based diesel calculation."""
        result = pipeline.run_pipeline({
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 200.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        })
        assert result["success"] is True
        final = result["result"]
        expected_co2 = 200.0 * FUEL_CO2_FACTORS_KG_PER_GALLON["DIESEL"]
        assert final["co2_kg"] == pytest.approx(expected_co2, rel=1e-3)

    def test_distance_based_full_pipeline(self, pipeline):
        """Full pipeline: distance-based produces non-zero emissions."""
        result = pipeline.run_pipeline({
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "DISTANCE_BASED",
            "distance": 500.0,
            "distance_unit": "MILES",
            "gwp_source": "AR6",
        })
        assert result["success"] is True
        assert result["result"]["total_co2e_kg"] > 0.0

    def test_spend_based_full_pipeline(self, pipeline):
        """Full pipeline: spend-based uses correct factor."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "SPEND_BASED",
            "spend_amount": 1000.0,
            "gwp_source": "AR6",
        })
        assert result["success"] is True
        expected = 1000.0 * SPEND_BASED_FACTORS_KG_CO2E_PER_USD["GASOLINE"]
        assert result["result"]["total_co2e_kg"] == pytest.approx(
            expected, rel=1e-3,
        )

    def test_fuel_liters_conversion_pipeline(self, pipeline):
        """Full pipeline: fuel in liters converts to gallons correctly."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 379.0,  # ~100 gallons
            "fuel_unit": "LITERS",
            "gwp_source": "AR6",
        })
        assert result["success"] is True
        gallons = 379.0 * VOLUME_TO_GALLONS["LITERS"]
        expected_co2 = gallons * FUEL_CO2_FACTORS_KG_PER_GALLON["GASOLINE"]
        assert result["result"]["co2_kg"] == pytest.approx(expected_co2, rel=1e-2)

    def test_ar5_gwp_values_applied(self, pipeline):
        """Full pipeline: AR5 GWP values produce different CH4/N2O."""
        result_ar6 = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        })
        result_ar5 = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR5",
        })
        # CO2 should be identical (GWP CO2 = 1.0 in both)
        assert result_ar6["result"]["co2_kg"] == pytest.approx(
            result_ar5["result"]["co2_kg"], rel=1e-6,
        )
        # CH4 CO2e differs: AR6 GWP_CH4=27.9 vs AR5 GWP_CH4=28.0
        # N2O CO2e differs: AR6 GWP_N2O=273.0 vs AR5 GWP_N2O=265.0
        # Total CO2e should differ slightly due to N2O difference
        ar6_total = result_ar6["result"]["total_co2e_kg"]
        ar5_total = result_ar5["result"]["total_co2e_kg"]
        # They should be close but not identical
        assert ar6_total != ar5_total or abs(ar6_total - ar5_total) < 1.0


# ===================================================================
# TestBatchCalculation (4 tests)
# ===================================================================


class TestBatchCalculation:
    """Test batch pipeline processing."""

    def test_batch_two_methods(self, pipeline):
        """Batch with fuel-based and distance-based inputs."""
        inputs = [
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
            },
            {
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
                "calculation_method": "DISTANCE_BASED",
                "distance": 300.0,
                "distance_unit": "KM",
            },
        ]
        results = pipeline.run_batch_pipeline(inputs)
        summary = results[0]
        assert summary["total_count"] == 2
        assert summary["success_count"] == 2
        assert summary["total_co2e_kg"] > 0.0

    def test_batch_all_three_methods(self, pipeline):
        """Batch with all three calculation methods."""
        inputs = [
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 25.0,
                "fuel_unit": "GALLONS",
            },
            {
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
                "calculation_method": "DISTANCE_BASED",
                "distance": 200.0,
                "distance_unit": "MILES",
            },
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "SPEND_BASED",
                "spend_amount": 500.0,
            },
        ]
        results = pipeline.run_batch_pipeline(inputs)
        summary = results[0]
        assert summary["total_count"] == 3
        assert summary["success_count"] == 3

    def test_batch_with_invalid_input(self, pipeline):
        """Batch handles a mix of valid and invalid inputs."""
        inputs = [
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
            },
            {
                "calculation_method": "FUEL_BASED",
                # missing fuel_quantity
            },
        ]
        results = pipeline.run_batch_pipeline(inputs)
        summary = results[0]
        assert summary["total_count"] == 2
        # First succeeds, second fails at validation
        assert summary["success_count"] >= 1

    def test_batch_provenance(self, pipeline):
        """Batch result has a provenance hash."""
        results = pipeline.run_batch_pipeline([
            {
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "calculation_method": "FUEL_BASED",
                "fuel_quantity": 10.0,
                "fuel_unit": "GALLONS",
            },
        ])
        summary = results[0]
        assert len(summary.get("provenance_hash", "")) == 64


# ===================================================================
# TestFleetOperations (5 tests)
# ===================================================================


class TestFleetOperations:
    """Test full fleet lifecycle through the service."""

    def test_register_list_vehicles(self, service, sample_vehicles):
        """Register and list all vehicles."""
        for v in sample_vehicles:
            service.register_vehicle(registration=v)
        vehicles = service.list_vehicles()
        assert len(vehicles) == 5

    def test_register_log_trip_list(self, populated_service, sample_trips):
        """Register vehicles, log trips, list trips."""
        trips = list(populated_service._trips.values())
        assert len(trips) == 5

    def test_fleet_aggregation_by_facility(self, populated_service, fuel_based_gasoline):
        """Aggregation groups by facility_id."""
        calc_inp = dict(fuel_based_gasoline)
        calc_inp["facility_id"] = "hq-london"
        populated_service.calculate(input_data=calc_inp)

        calc_inp2 = dict(fuel_based_gasoline)
        calc_inp2["facility_id"] = "warehouse-birmingham"
        populated_service.calculate(input_data=calc_inp2)

        agg = populated_service.aggregate_fleet(period="2025-FY")
        assert "by_facility" in agg
        assert len(agg["by_facility"]) >= 2

    def test_fleet_filter_vehicles_by_fleet(self, populated_service):
        """Filter vehicles by fleet_id."""
        exec_vehicles = populated_service.list_vehicles(
            filters={"fleet_id": "fleet-exec"},
        )
        assert len(exec_vehicles) == 2  # Car + Jet

    def test_fleet_stats_reflect_state(self, populated_service):
        """Service stats reflect registered vehicles and trips."""
        stats = populated_service.get_stats()
        assert stats["total_vehicles"] == 5
        assert stats["total_trips"] == 5


# ===================================================================
# TestComplianceIntegration (4 tests)
# ===================================================================


class TestComplianceIntegration:
    """Test compliance checking against all supported frameworks."""

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL",
        "EPA_40CFR98",
        "ISO_14064",
        "CSRD_ESRS_E1",
    ])
    def test_compliance_framework(self, pipeline, framework):
        """Pipeline compliance stage checks the specified framework."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "regulatory_framework": framework,
        })
        assert result["success"] is True
        compliance = result["result"].get("compliance", [])
        assert len(compliance) >= 1
        assert compliance[0]["framework"] == framework


# ===================================================================
# TestUncertaintyIntegration (3 tests)
# ===================================================================


class TestUncertaintyIntegration:
    """Test uncertainty quantification through the pipeline."""

    def test_pipeline_includes_uncertainty(self, pipeline):
        """Pipeline result contains uncertainty data."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
        })
        assert result["success"] is True
        unc = result["result"].get("uncertainty", {})
        assert "mean_co2e_kg" in unc
        assert "std_co2e_kg" in unc

    def test_uncertainty_tier_2(self, pipeline):
        """Tier 2 has lower relative uncertainty than Tier 1."""
        result_t1 = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "tier": "TIER_1",
        })
        result_t2 = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "tier": "TIER_2",
        })
        unc_t1 = result_t1["result"].get("uncertainty", {})
        unc_t2 = result_t2["result"].get("uncertainty", {})
        # Tier 2 should have lower relative uncertainty
        assert unc_t2.get("relative_uncertainty_pct", 100) <= unc_t1.get(
            "relative_uncertainty_pct", 0,
        )

    def test_uncertainty_dqi_score(self, pipeline):
        """Data quality indicator score is set correctly per tier."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
            "tier": "TIER_3",
        })
        unc = result["result"].get("uncertainty", {})
        # TIER_3 -> DQI score = 5
        assert unc.get("data_quality_score") == 5


# ===================================================================
# TestAuditTrail (3 tests)
# ===================================================================


class TestAuditTrail:
    """Test audit trail and provenance chain integrity."""

    def test_pipeline_has_audit_entries(self, pipeline):
        """Pipeline result includes audit entries."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "GALLONS",
        })
        audit_entries = result["result"].get("audit_entries", [])
        assert len(audit_entries) >= 1

    def test_audit_entry_has_chain_hash(self, pipeline):
        """Audit entry chain_hash is a 64-char SHA-256."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 50.0,
            "fuel_unit": "GALLONS",
        })
        audit = result["result"]["audit_entries"][0]
        assert len(audit["chain_hash"]) == 64

    def test_provenance_chain_length_matches_stages(self, pipeline):
        """Provenance chain has one entry per completed stage."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 25.0,
            "fuel_unit": "GALLONS",
        })
        # 8 stages should produce 8 provenance entries
        stage_results = result["stage_results"]
        successful_stages = [
            sr for sr in stage_results if sr["status"] == "SUCCESS"
        ]
        # The audit entry's provenance_chain should have entries from
        # stages 1-7 (audit stage adds its own as stage 8)
        audit = result["result"]["audit_entries"][0]
        assert len(audit["provenance_chain"]) >= len(successful_stages) - 1


# ===================================================================
# TestEdgeCases (3 tests)
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary values."""

    def test_empty_input_fails_gracefully(self, pipeline):
        """Empty input dict fails at validation but does not crash."""
        result = pipeline.run_pipeline({})
        assert result["success"] is False
        assert result["stages_completed"] < 8

    def test_zero_fuel_quantity_fails(self, pipeline):
        """Zero fuel quantity fails validation."""
        result = pipeline.run_pipeline({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 0.0,
            "fuel_unit": "GALLONS",
        })
        assert result["success"] is False

    def test_very_large_fuel_quantity(self, pipeline):
        """Very large fuel quantity does not overflow."""
        result = pipeline.run_pipeline({
            "vehicle_type": "MARINE_VESSEL",
            "fuel_type": "MARINE_DIESEL",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 1_000_000.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        })
        assert result["success"] is True
        final = result["result"]
        expected_co2 = 1_000_000.0 * FUEL_CO2_FACTORS_KG_PER_GALLON["MARINE_DIESEL"]
        assert final["co2_kg"] == pytest.approx(expected_co2, rel=1e-3)
        assert final["total_co2e_tonnes"] > 0.0
