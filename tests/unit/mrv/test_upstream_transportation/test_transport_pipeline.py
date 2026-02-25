"""
Unit tests for TransportPipelineEngine.

Tests 10-stage upstream transportation emission calculation pipeline:
1. Validate input
2. Classify transport type
3. Normalize units
4. Resolve emission factors
5. Calculate legs
6. Calculate hubs
7. Apply allocation
8. Check compliance
9. Aggregate results
10. Seal with provenance

Tests:
- Full pipeline execution (distance-based, fuel-based, spend-based, supplier, multi-leg)
- Individual stage execution
- Stage validation
- Error handling and retry
- Provenance chain validation
- Metrics recording
- Batch processing
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any

from greenlang.mrv.upstream_transportation.engines.transport_pipeline import (
    TransportPipelineEngine,
    PipelineInput,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from greenlang.mrv.upstream_transportation.models import (
    TransportMode,
    VehicleType,
    FuelType,
    EmissionScope,
    DataQualityTier,
    AllocationMethod,
)


@pytest.fixture
def engine():
    """Create TransportPipelineEngine instance."""
    return TransportPipelineEngine()


@pytest.fixture
def distance_based_road_input():
    """Distance-based road transport input."""
    return PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("20.0"),
        origin="Warehouse A",
        destination="Customer B",
    )


@pytest.fixture
def fuel_based_input():
    """Fuel-based calculation input."""
    return PipelineInput(
        calculation_type="fuel_based",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        fuel_type=FuelType.DIESEL,
        fuel_consumed_liters=Decimal("850"),
        distance_km=Decimal("3000"),
    )


@pytest.fixture
def spend_based_input():
    """Spend-based calculation input."""
    return PipelineInput(
        calculation_type="spend_based",
        spend_amount=Decimal("15000.00"),
        spend_currency="USD",
        spend_year=2023,
        naics_code="484121",  # Trucking
        transport_mode=TransportMode.ROAD,
    )


@pytest.fixture
def supplier_specific_input():
    """Supplier-specific data input."""
    return PipelineInput(
        calculation_type="supplier_specific",
        mode=TransportMode.MARITIME,
        supplier_data={
            "methodology": "GLEC",
            "total_co2e_kg": Decimal("8500"),
            "distance_km": Decimal("12000"),
            "certification": "GLEC_ACCREDITED",
        },
    )


@pytest.fixture
def multi_leg_chain_input():
    """Multi-leg transport chain input."""
    return PipelineInput(
        calculation_type="multi_leg",
        transport_chain={
            "chain_id": "CHAIN-001",
            "legs": [
                {
                    "mode": TransportMode.ROAD,
                    "vehicle_type": VehicleType.TRUCK_RIGID_GT17T,
                    "distance_km": Decimal("100"),
                    "mass_tonnes": Decimal("15.0"),
                },
                {
                    "mode": TransportMode.MARITIME,
                    "vehicle_type": VehicleType.CONTAINER_SHIP_2000_8000TEU,
                    "distance_km": Decimal("8000"),
                    "mass_tonnes": Decimal("15.0"),
                },
                {
                    "mode": TransportMode.ROAD,
                    "vehicle_type": VehicleType.TRUCK_ARTICULATED_GT33T,
                    "distance_km": Decimal("150"),
                    "mass_tonnes": Decimal("15.0"),
                },
            ],
            "hubs": [
                {"type": "port", "location": "Port A"},
                {"type": "port", "location": "Port B"},
            ],
        },
    )


# ============================================================================
# Full Pipeline Execution
# ============================================================================


def test_execute_distance_based_road(engine, distance_based_road_input):
    """Test full pipeline execution for distance-based road transport."""
    result = engine.execute(distance_based_road_input)

    assert isinstance(result, PipelineResult)
    assert result.status == PipelineStatus.SUCCESS
    assert result.co2e_kg > Decimal("0")
    assert len(result.stages_completed) == 10  # All 10 stages
    assert result.provenance_hash is not None
    assert result.data_quality_tier in [DataQualityTier.TIER_1, DataQualityTier.TIER_2]


def test_execute_fuel_based(engine, fuel_based_input):
    """Test full pipeline execution for fuel-based calculation."""
    result = engine.execute(fuel_based_input)

    assert result.status == PipelineStatus.SUCCESS
    assert result.co2e_kg > Decimal("0")
    assert result.calculation_type == "fuel_based"
    assert result.data_quality_tier == DataQualityTier.TIER_1  # Fuel is Tier 1


def test_execute_spend_based(engine, spend_based_input):
    """Test full pipeline execution for spend-based calculation."""
    result = engine.execute(spend_based_input)

    assert result.status == PipelineStatus.SUCCESS
    assert result.co2e_kg > Decimal("0")
    assert result.calculation_type == "spend_based"
    assert result.data_quality_tier == DataQualityTier.TIER_3  # Spend is Tier 3


def test_execute_supplier_specific(engine, supplier_specific_input):
    """Test full pipeline execution for supplier-specific data."""
    result = engine.execute(supplier_specific_input)

    assert result.status == PipelineStatus.SUCCESS
    assert result.co2e_kg == Decimal("8500")  # Uses supplier value
    assert result.calculation_type == "supplier_specific"
    assert result.data_quality_tier == DataQualityTier.TIER_1  # GLEC is Tier 1


def test_execute_multi_leg_chain(engine, multi_leg_chain_input):
    """Test full pipeline execution for multi-leg transport chain."""
    result = engine.execute(multi_leg_chain_input)

    assert result.status == PipelineStatus.SUCCESS
    assert result.co2e_kg > Decimal("0")
    assert result.calculation_type == "multi_leg"
    assert len(result.leg_results) == 3
    assert len(result.hub_results) == 2


def test_execute_with_reefer(engine):
    """Test pipeline with refrigerated transport."""
    reefer_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_REFRIGERATED,
        distance_km=Decimal("800"),
        mass_tonnes=Decimal("18.0"),
        temperature_celsius=Decimal("-18"),  # Frozen
    )

    result = engine.execute(reefer_input)

    assert result.status == PipelineStatus.SUCCESS
    # Reefer should have additional emissions
    assert result.reefer_emissions_kg > Decimal("0")


def test_execute_with_warehousing(engine):
    """Test pipeline with warehousing emissions."""
    warehouse_input = PipelineInput(
        calculation_type="multi_leg",
        transport_chain={
            "legs": [
                {
                    "mode": TransportMode.ROAD,
                    "distance_km": Decimal("200"),
                    "mass_tonnes": Decimal("10.0"),
                }
            ],
            "hubs": [
                {
                    "type": "warehouse",
                    "storage_duration_days": 30,
                    "throughput_tonnes": Decimal("10.0"),
                }
            ],
        },
    )

    result = engine.execute(warehouse_input)

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.hub_results) == 1
    assert result.hub_results[0].co2e_kg > Decimal("0")


def test_execute_batch(engine, distance_based_road_input, fuel_based_input):
    """Test batch pipeline execution."""
    inputs = [distance_based_road_input, fuel_based_input]

    results = engine.execute_batch(inputs)

    assert len(results) == 2
    assert all(isinstance(r, PipelineResult) for r in results)
    assert all(r.status == PipelineStatus.SUCCESS for r in results)


# ============================================================================
# 10-Stage Pipeline
# ============================================================================


def test_10_stage_pipeline_all_stages_executed(engine, distance_based_road_input):
    """Test all 10 pipeline stages are executed."""
    result = engine.execute(distance_based_road_input)

    expected_stages = [
        PipelineStage.VALIDATE,
        PipelineStage.CLASSIFY,
        PipelineStage.NORMALIZE,
        PipelineStage.RESOLVE_EFS,
        PipelineStage.CALCULATE_LEGS,
        PipelineStage.CALCULATE_HUBS,
        PipelineStage.ALLOCATE,
        PipelineStage.COMPLIANCE,
        PipelineStage.AGGREGATE,
        PipelineStage.SEAL,
    ]

    assert len(result.stages_completed) == 10
    assert all(stage in result.stages_completed for stage in expected_stages)


# ============================================================================
# Individual Stage Execution
# ============================================================================


def test_validate_stage(engine, distance_based_road_input):
    """Test Stage 1: Validate input."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.VALIDATE,
        input_data=distance_based_road_input,
    )

    assert stage_result["is_valid"] is True
    assert len(stage_result["errors"]) == 0


def test_classify_stage(engine, distance_based_road_input):
    """Test Stage 2: Classify transport type."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.CLASSIFY,
        input_data=distance_based_road_input,
    )

    assert stage_result["calculation_type"] == "distance_based"
    assert stage_result["mode"] == TransportMode.ROAD
    assert stage_result["vehicle_type"] == VehicleType.TRUCK_ARTICULATED_GT33T


def test_normalize_stage(engine, distance_based_road_input):
    """Test Stage 3: Normalize units."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.NORMALIZE,
        input_data=distance_based_road_input,
    )

    # Distance normalized to km
    assert stage_result["distance_km"] == Decimal("500")
    # Mass normalized to tonnes
    assert stage_result["mass_tonnes"] == Decimal("20.0")


def test_resolve_efs_stage(engine, distance_based_road_input):
    """Test Stage 4: Resolve emission factors."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.RESOLVE_EFS,
        input_data=distance_based_road_input,
    )

    assert "emission_factor" in stage_result
    assert stage_result["emission_factor"]["kg_co2e_per_tonne_km"] > Decimal("0")
    assert "ef_source" in stage_result


def test_calculate_legs_stage(engine, distance_based_road_input):
    """Test Stage 5: Calculate leg emissions."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.CALCULATE_LEGS,
        input_data=distance_based_road_input,
    )

    assert "leg_results" in stage_result
    assert len(stage_result["leg_results"]) >= 1
    assert stage_result["total_leg_emissions_kg"] > Decimal("0")


def test_calculate_hubs_stage(engine, multi_leg_chain_input):
    """Test Stage 6: Calculate hub emissions."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.CALCULATE_HUBS,
        input_data=multi_leg_chain_input,
    )

    assert "hub_results" in stage_result
    assert len(stage_result["hub_results"]) == 2  # Two hubs
    assert stage_result["total_hub_emissions_kg"] > Decimal("0")


def test_allocate_stage(engine):
    """Test Stage 7: Apply allocation."""
    allocation_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("10.0"),  # Shipment
        total_load_mass_tonnes=Decimal("20.0"),  # Full load
        allocation_method=AllocationMethod.MASS,
    )

    stage_result = engine.execute_stage(
        stage=PipelineStage.ALLOCATE,
        input_data=allocation_input,
    )

    # 10/20 = 50% allocation
    assert stage_result["allocation_factor"] == Decimal("0.5")
    assert "allocated_emissions_kg" in stage_result


def test_compliance_stage(engine, distance_based_road_input):
    """Test Stage 8: Check compliance."""
    distance_based_road_input.framework = "GHG_PROTOCOL"
    distance_based_road_input.scope = EmissionScope.WTW

    stage_result = engine.execute_stage(
        stage=PipelineStage.COMPLIANCE,
        input_data=distance_based_road_input,
    )

    assert "compliance_result" in stage_result
    assert stage_result["compliance_result"]["framework"] == "GHG_PROTOCOL"


def test_aggregate_stage(engine, multi_leg_chain_input):
    """Test Stage 9: Aggregate results."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.AGGREGATE,
        input_data=multi_leg_chain_input,
    )

    assert "total_co2e_kg" in stage_result
    assert "by_mode" in stage_result
    assert "by_leg" in stage_result


def test_seal_stage_provenance_hash(engine, distance_based_road_input):
    """Test Stage 10: Seal with provenance hash."""
    stage_result = engine.execute_stage(
        stage=PipelineStage.SEAL,
        input_data=distance_based_road_input,
    )

    assert "provenance_hash" in stage_result
    assert len(stage_result["provenance_hash"]) == 64  # SHA-256 hex
    assert "timestamp" in stage_result


# ============================================================================
# Pipeline Status
# ============================================================================


def test_pipeline_status(engine, distance_based_road_input):
    """Test pipeline status tracking."""
    result = engine.execute(distance_based_road_input)

    assert result.status == PipelineStatus.SUCCESS
    assert result.start_time is not None
    assert result.end_time is not None
    assert result.processing_time_ms > 0


# ============================================================================
# Error Handling
# ============================================================================


def test_stage_error_partial_result(engine):
    """Test stage error produces partial result."""
    invalid_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        distance_km=Decimal("-100"),  # Invalid negative distance
        mass_tonnes=Decimal("20.0"),
    )

    result = engine.execute(invalid_input)

    assert result.status == PipelineStatus.FAILED
    assert len(result.errors) > 0
    # Should complete validate stage before failing
    assert PipelineStage.VALIDATE in result.stages_completed


# ============================================================================
# Retry Logic
# ============================================================================


def test_retry_from_stage(engine, distance_based_road_input):
    """Test retry from specific stage."""
    # Execute full pipeline first
    result1 = engine.execute(distance_based_road_input)

    # Retry from compliance stage (e.g., to check different framework)
    distance_based_road_input.framework = "ISO_14083"
    result2 = engine.retry_from_stage(
        input_data=distance_based_road_input,
        from_stage=PipelineStage.COMPLIANCE,
        previous_result=result1,
    )

    # Should reuse earlier stages
    assert result2.co2e_kg == result1.co2e_kg  # Same emissions
    # But different compliance result
    assert result2.compliance_result["framework"] == "ISO_14083"


# ============================================================================
# Provenance Chain
# ============================================================================


def test_provenance_chain_valid(engine, distance_based_road_input):
    """Test provenance chain is valid and reproducible."""
    result1 = engine.execute(distance_based_road_input)
    result2 = engine.execute(distance_based_road_input)

    # Same input → same provenance hash
    assert result1.provenance_hash == result2.provenance_hash


# ============================================================================
# Metrics Recording
# ============================================================================


def test_metrics_recorded(engine, distance_based_road_input):
    """Test pipeline metrics are recorded."""
    result = engine.execute(distance_based_road_input)

    assert result.metrics is not None
    assert "total_processing_time_ms" in result.metrics
    assert "stage_times_ms" in result.metrics
    assert len(result.metrics["stage_times_ms"]) == 10


# ============================================================================
# Edge Cases
# ============================================================================


def test_invalid_calculation_type_raises(engine):
    """Test invalid calculation type raises error."""
    invalid_input = PipelineInput(
        calculation_type="invalid_type",
        mode=TransportMode.ROAD,
    )

    with pytest.raises(ValueError, match="invalid calculation type"):
        engine.execute(invalid_input)


def test_missing_required_field_fails_validation(engine):
    """Test missing required field fails validation stage."""
    missing_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        # Missing distance_km
        mass_tonnes=Decimal("20.0"),
    )

    result = engine.execute(missing_input)

    assert result.status == PipelineStatus.FAILED
    assert PipelineStage.VALIDATE in result.stages_completed
    assert len(result.errors) > 0


def test_zero_distance_raises(engine):
    """Test zero distance raises error."""
    zero_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        distance_km=Decimal("0"),  # Zero
        mass_tonnes=Decimal("20.0"),
    )

    result = engine.execute(zero_input)

    assert result.status == PipelineStatus.FAILED


def test_negative_mass_raises(engine):
    """Test negative mass raises error."""
    negative_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("-10.0"),  # Negative
    )

    result = engine.execute(negative_input)

    assert result.status == PipelineStatus.FAILED


def test_supplier_data_skips_calculation_stages(engine, supplier_specific_input):
    """Test supplier-specific data skips some calculation stages."""
    result = engine.execute(supplier_specific_input)

    assert result.status == PipelineStatus.SUCCESS
    # Should skip resolve_efs and calculate_legs (uses supplier value)
    # But still completes all 10 stages (skipped stages marked as bypassed)
    assert len(result.stages_completed) == 10


def test_allocation_without_total_load_uses_100pct(engine):
    """Test allocation defaults to 100% without total load."""
    no_allocation_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("20.0"),
        # No total_load_mass_tonnes
        allocation_method=AllocationMethod.MASS,
    )

    result = engine.execute(no_allocation_input)

    # Should default to 100% allocation (no sharing)
    assert result.allocation_factor == Decimal("1.0")


def test_multi_leg_without_hubs_succeeds(engine):
    """Test multi-leg chain without hubs succeeds."""
    no_hubs_input = PipelineInput(
        calculation_type="multi_leg",
        transport_chain={
            "legs": [
                {
                    "mode": TransportMode.ROAD,
                    "distance_km": Decimal("100"),
                    "mass_tonnes": Decimal("10.0"),
                },
                {
                    "mode": TransportMode.ROAD,
                    "distance_km": Decimal("200"),
                    "mass_tonnes": Decimal("10.0"),
                },
            ],
            "hubs": [],  # No hubs
        },
    )

    result = engine.execute(no_hubs_input)

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.leg_results) == 2
    assert len(result.hub_results) == 0


def test_compliance_optional(engine, distance_based_road_input):
    """Test compliance check is optional."""
    # Don't specify framework
    distance_based_road_input.framework = None

    result = engine.execute(distance_based_road_input)

    # Should still succeed
    assert result.status == PipelineStatus.SUCCESS
    # Compliance stage skipped
    assert result.compliance_result is None


def test_very_long_chain_succeeds(engine):
    """Test very long chain (20+ legs) succeeds."""
    long_chain_input = PipelineInput(
        calculation_type="multi_leg",
        transport_chain={
            "legs": [
                {
                    "mode": TransportMode.ROAD,
                    "distance_km": Decimal("50"),
                    "mass_tonnes": Decimal("10.0"),
                }
                for _ in range(25)  # 25 legs
            ],
            "hubs": [],
        },
    )

    result = engine.execute(long_chain_input)

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.leg_results) == 25


def test_mixed_units_normalized(engine):
    """Test mixed units are normalized."""
    mixed_input = PipelineInput(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        distance_km=Decimal("500"),  # km
        distance_miles=Decimal("310.7"),  # Also provide miles (should be ignored)
        mass_tonnes=Decimal("20.0"),  # tonnes
        mass_kg=Decimal("20000"),  # Also provide kg (should be ignored)
    )

    result = engine.execute(mixed_input)

    # Should use km and tonnes
    assert result.distance_km == Decimal("500")
    assert result.mass_tonnes == Decimal("20.0")


def test_fuel_based_requires_fuel_type(engine):
    """Test fuel-based calculation requires fuel type."""
    no_fuel_type_input = PipelineInput(
        calculation_type="fuel_based",
        fuel_consumed_liters=Decimal("500"),
        # Missing fuel_type
    )

    result = engine.execute(no_fuel_type_input)

    assert result.status == PipelineStatus.FAILED
    assert any("fuel_type" in e.lower() for e in result.errors)


def test_spend_based_requires_spend_amount(engine):
    """Test spend-based calculation requires spend amount."""
    no_spend_input = PipelineInput(
        calculation_type="spend_based",
        naics_code="484121",
        # Missing spend_amount
    )

    result = engine.execute(no_spend_input)

    assert result.status == PipelineStatus.FAILED


def test_pipeline_execution_order_matters(engine, distance_based_road_input):
    """Test pipeline stages execute in correct order."""
    result = engine.execute(distance_based_road_input)

    stage_order = [
        PipelineStage.VALIDATE,
        PipelineStage.CLASSIFY,
        PipelineStage.NORMALIZE,
        PipelineStage.RESOLVE_EFS,
        PipelineStage.CALCULATE_LEGS,
        PipelineStage.CALCULATE_HUBS,
        PipelineStage.ALLOCATE,
        PipelineStage.COMPLIANCE,
        PipelineStage.AGGREGATE,
        PipelineStage.SEAL,
    ]

    # Check stages completed in order
    for i, stage in enumerate(stage_order):
        assert result.stages_completed[i] == stage


def test_decimal_precision_maintained(engine, distance_based_road_input):
    """Test Decimal precision maintained through pipeline."""
    result = engine.execute(distance_based_road_input)

    assert isinstance(result.co2e_kg, Decimal)
    assert isinstance(result.distance_km, Decimal)
    assert isinstance(result.mass_tonnes, Decimal)
