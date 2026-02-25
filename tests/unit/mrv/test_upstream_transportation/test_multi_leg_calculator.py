"""
Unit tests for MultiLegCalculatorEngine.

Tests multi-leg transportation chain calculation, hub/warehouse emissions,
supplier-specific data processing, allocation methods, refrigerated transport,
and complex supply chain scenarios.

Tests:
- Chain calculations (single-leg, multi-leg, with hubs)
- Individual leg calculations (road, maritime, air)
- Hub/warehouse emissions
- Supplier data processing (GLEC, SmartWay)
- Supplier verification and certification
- Allocation methods (mass, volume, pallet, TEU, revenue, chargeable weight)
- Refrigerated transport (reefer emissions, TRU fuel, refrigerant leakage)
- Aggregations
- Data quality weighting
- Batch processing
- Edge cases
"""

import pytest
from decimal import Decimal
from datetime import date
from typing import Dict, List, Any

from greenlang.mrv.upstream_transportation.engines.multi_leg_calculator import (
    MultiLegCalculatorEngine,
    TransportChain,
    TransportLeg,
    TransportHub,
    ChainResult,
    LegResult,
    HubResult,
    AllocationMethod,
    SupplierMethodology,
)
from greenlang.mrv.upstream_transportation.models import (
    TransportMode,
    VehicleType,
    FuelType,
    EmissionScope,
    DataQualityTier,
)


@pytest.fixture
def engine():
    """Create MultiLegCalculatorEngine instance."""
    return MultiLegCalculatorEngine()


@pytest.fixture
def single_leg_road_chain():
    """Single-leg road transport chain."""
    return TransportChain(
        chain_id="CHAIN-001",
        legs=[
            TransportLeg(
                leg_id="LEG-001",
                mode=TransportMode.ROAD,
                vehicle_type=VehicleType.TRUCK_RIGID_GT17T,
                distance_km=Decimal("250"),
                mass_tonnes=Decimal("15.0"),
                origin="Warehouse A",
                destination="Distribution Center B",
            )
        ],
        hubs=[],
    )


@pytest.fixture
def multi_leg_truck_ship_truck_chain():
    """Multi-leg chain: truck → ship → truck."""
    return TransportChain(
        chain_id="CHAIN-002",
        legs=[
            TransportLeg(
                leg_id="LEG-001",
                mode=TransportMode.ROAD,
                vehicle_type=VehicleType.TRUCK_RIGID_GT17T,
                distance_km=Decimal("100"),
                mass_tonnes=Decimal("20.0"),
                origin="Factory",
                destination="Port A",
            ),
            TransportLeg(
                leg_id="LEG-002",
                mode=TransportMode.MARITIME,
                vehicle_type=VehicleType.CONTAINER_SHIP_2000_8000TEU,
                distance_km=Decimal("5000"),
                mass_tonnes=Decimal("20.0"),
                origin="Port A",
                destination="Port B",
            ),
            TransportLeg(
                leg_id="LEG-003",
                mode=TransportMode.ROAD,
                vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
                distance_km=Decimal("150"),
                mass_tonnes=Decimal("20.0"),
                origin="Port B",
                destination="Customer",
            ),
        ],
        hubs=[
            TransportHub(
                hub_id="HUB-001",
                hub_type="port",
                location="Port A",
                operations=["container_handling", "customs"],
            ),
            TransportHub(
                hub_id="HUB-002",
                hub_type="port",
                location="Port B",
                operations=["container_handling"],
            ),
        ],
    )


@pytest.fixture
def shanghai_to_munich_chain():
    """Complex chain: Shanghai to Munich (truck→ship→rail→truck)."""
    return TransportChain(
        chain_id="CHAIN-003",
        legs=[
            TransportLeg(
                leg_id="LEG-001",
                mode=TransportMode.ROAD,
                vehicle_type=VehicleType.TRUCK_RIGID_GT17T,
                distance_km=Decimal("50"),
                mass_tonnes=Decimal("18.0"),
                origin="Factory Shanghai",
                destination="Port Shanghai",
            ),
            TransportLeg(
                leg_id="LEG-002",
                mode=TransportMode.MARITIME,
                vehicle_type=VehicleType.CONTAINER_SHIP_GT14500TEU,
                distance_km=Decimal("18500"),
                mass_tonnes=Decimal("18.0"),
                origin="Port Shanghai",
                destination="Port Hamburg",
            ),
            TransportLeg(
                leg_id="LEG-003",
                mode=TransportMode.RAIL,
                vehicle_type=VehicleType.FREIGHT_TRAIN,
                distance_km=Decimal("800"),
                mass_tonnes=Decimal("18.0"),
                origin="Port Hamburg",
                destination="Rail Terminal Munich",
            ),
            TransportLeg(
                leg_id="LEG-004",
                mode=TransportMode.ROAD,
                vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
                distance_km=Decimal("30"),
                mass_tonnes=Decimal("18.0"),
                origin="Rail Terminal Munich",
                destination="Distribution Center Munich",
            ),
        ],
        hubs=[
            TransportHub(
                hub_id="HUB-001",
                hub_type="port",
                location="Port Shanghai",
                operations=["container_loading", "customs_export"],
            ),
            TransportHub(
                hub_id="HUB-002",
                hub_type="port",
                location="Port Hamburg",
                operations=["container_unloading", "customs_import"],
            ),
            TransportHub(
                hub_id="HUB-003",
                hub_type="rail_terminal",
                location="Rail Terminal Munich",
                operations=["container_transfer"],
            ),
        ],
    )


# ============================================================================
# Chain Calculations
# ============================================================================


def test_calculate_chain_single_leg_road(engine, single_leg_road_chain):
    """Test single-leg road chain calculation."""
    result = engine.calculate_chain(single_leg_road_chain)

    assert isinstance(result, ChainResult)
    assert result.chain_id == "CHAIN-001"
    assert len(result.leg_results) == 1
    assert len(result.hub_results) == 0
    assert result.total_co2e_kg > Decimal("0")
    assert result.total_distance_km == Decimal("250")


def test_calculate_chain_multi_leg_truck_ship_truck(
    engine, multi_leg_truck_ship_truck_chain
):
    """Test multi-leg truck→ship→truck chain."""
    result = engine.calculate_chain(multi_leg_truck_ship_truck_chain)

    assert result.chain_id == "CHAIN-002"
    assert len(result.leg_results) == 3
    assert len(result.hub_results) == 2
    assert result.total_co2e_kg > Decimal("0")
    assert result.total_distance_km == Decimal("5250")  # 100+5000+150

    # Maritime leg should dominate distance
    maritime_leg = next(
        lr for lr in result.leg_results if lr.mode == TransportMode.MARITIME
    )
    assert maritime_leg.distance_km == Decimal("5000")


def test_calculate_chain_with_hubs(engine, multi_leg_truck_ship_truck_chain):
    """Test chain calculation includes hub emissions."""
    result = engine.calculate_chain(multi_leg_truck_ship_truck_chain)

    # Hub emissions should be non-zero
    total_hub_emissions = sum(hr.co2e_kg for hr in result.hub_results)
    assert total_hub_emissions > Decimal("0")

    # Hub emissions should be small compared to transport
    total_leg_emissions = sum(lr.co2e_kg for lr in result.leg_results)
    assert total_hub_emissions < total_leg_emissions * Decimal("0.1")  # <10%


def test_calculate_chain_shanghai_to_munich(engine, shanghai_to_munich_chain):
    """Test complex Shanghai→Munich chain (truck→ship→rail→truck)."""
    result = engine.calculate_chain(shanghai_to_munich_chain)

    assert result.chain_id == "CHAIN-003"
    assert len(result.leg_results) == 4
    assert len(result.hub_results) == 3

    # Maritime leg dominates emissions
    maritime_leg = next(
        lr for lr in result.leg_results if lr.mode == TransportMode.MARITIME
    )
    rail_leg = next(
        lr for lr in result.leg_results if lr.mode == TransportMode.RAIL
    )

    assert maritime_leg.co2e_kg > rail_leg.co2e_kg * Decimal("10")  # Much higher


# ============================================================================
# Individual Leg Calculations
# ============================================================================


def test_calculate_leg_road(engine):
    """Test road leg calculation."""
    leg = TransportLeg(
        leg_id="LEG-001",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        distance_km=Decimal("300"),
        mass_tonnes=Decimal("22.0"),
    )

    result = engine.calculate_leg(leg)

    assert isinstance(result, LegResult)
    assert result.leg_id == "LEG-001"
    assert result.co2e_kg > Decimal("0")
    assert result.mode == TransportMode.ROAD


def test_calculate_leg_maritime(engine):
    """Test maritime leg calculation."""
    leg = TransportLeg(
        leg_id="LEG-002",
        mode=TransportMode.MARITIME,
        vehicle_type=VehicleType.CONTAINER_SHIP_2000_8000TEU,
        distance_km=Decimal("8000"),
        mass_tonnes=Decimal("15.0"),
    )

    result = engine.calculate_leg(leg)

    assert result.mode == TransportMode.MARITIME
    assert result.co2e_kg > Decimal("0")
    # Maritime has low intensity per tonne-km
    intensity = result.co2e_kg / (leg.distance_km * leg.mass_tonnes)
    assert intensity < Decimal("0.05")  # Low g CO2e/tonne-km


def test_calculate_leg_air(engine):
    """Test air leg calculation."""
    leg = TransportLeg(
        leg_id="LEG-003",
        mode=TransportMode.AIR,
        vehicle_type=VehicleType.FREIGHT_AIRCRAFT,
        distance_km=Decimal("5000"),
        mass_tonnes=Decimal("5.0"),
    )

    result = engine.calculate_leg(leg)

    assert result.mode == TransportMode.AIR
    # Air has very high intensity
    intensity = result.co2e_kg / (leg.distance_km * leg.mass_tonnes)
    assert intensity > Decimal("0.5")  # High g CO2e/tonne-km


# ============================================================================
# Hub Calculations
# ============================================================================


def test_calculate_hub_logistics(engine):
    """Test logistics hub emissions calculation."""
    hub = TransportHub(
        hub_id="HUB-001",
        hub_type="logistics_center",
        operations=["sorting", "cross_docking"],
        throughput_tonnes=Decimal("100.0"),
    )

    result = engine.calculate_hub(hub)

    assert isinstance(result, HubResult)
    assert result.hub_id == "HUB-001"
    assert result.co2e_kg > Decimal("0")


def test_calculate_hub_container_terminal(engine):
    """Test container terminal emissions."""
    hub = TransportHub(
        hub_id="HUB-002",
        hub_type="container_terminal",
        operations=["container_handling", "storage"],
        throughput_teu=Decimal("50.0"),  # TEU throughput
    )

    result = engine.calculate_hub(hub)

    assert result.co2e_kg > Decimal("0")


# ============================================================================
# Warehouse Calculations
# ============================================================================


def test_calculate_warehouse_standard(engine):
    """Test standard warehouse emissions."""
    warehouse = TransportHub(
        hub_id="WH-001",
        hub_type="warehouse",
        operations=["storage", "handling"],
        storage_duration_days=30,
        throughput_tonnes=Decimal("50.0"),
    )

    result = engine.calculate_hub(warehouse)

    assert result.co2e_kg > Decimal("0")
    # Warehouse emissions are low
    assert result.co2e_kg < Decimal("500")


def test_calculate_warehouse_cold_storage(engine):
    """Test cold storage warehouse (higher emissions)."""
    cold_warehouse = TransportHub(
        hub_id="WH-002",
        hub_type="warehouse_cold_storage",
        operations=["refrigerated_storage", "handling"],
        storage_duration_days=30,
        throughput_tonnes=Decimal("50.0"),
        temperature_celsius=Decimal("-18"),  # Frozen
    )

    result = engine.calculate_hub(cold_warehouse)

    # Cold storage has 3-5x higher emissions
    assert result.co2e_kg > Decimal("100")


# ============================================================================
# Supplier Data Processing
# ============================================================================


def test_process_supplier_data_glec(engine):
    """Test processing GLEC Framework supplier data."""
    supplier_data = {
        "methodology": SupplierMethodology.GLEC,
        "total_co2e_kg": Decimal("5000.0"),
        "distance_km": Decimal("8000"),
        "mass_tonnes": Decimal("20.0"),
        "certification": "GLEC_ACCREDITED",
        "verification_status": "third_party_verified",
    }

    processed = engine.process_supplier_data(supplier_data)

    assert processed["co2e_kg"] == Decimal("5000.0")
    assert processed["data_quality_tier"] == DataQualityTier.TIER_1  # High quality
    assert processed["verification_score"] > 0.9


def test_process_supplier_data_smartway(engine):
    """Test processing SmartWay supplier data."""
    supplier_data = {
        "methodology": SupplierMethodology.SMARTWAY,
        "total_co2e_kg": Decimal("3000.0"),
        "distance_km": Decimal("5000"),
        "certification": "SMARTWAY_CERTIFIED",
    }

    processed = engine.process_supplier_data(supplier_data)

    assert processed["co2e_kg"] == Decimal("3000.0")
    assert processed["data_quality_tier"] == DataQualityTier.TIER_1


# ============================================================================
# Supplier Verification
# ============================================================================


def test_validate_supplier_methodology_glec_valid(engine):
    """Test GLEC methodology validation passes."""
    is_valid, errors = engine.validate_supplier_methodology(
        methodology=SupplierMethodology.GLEC,
        data={
            "total_co2e_kg": Decimal("1000"),
            "distance_km": Decimal("2000"),
            "mass_tonnes": Decimal("15"),
            "fuel_consumption_liters": Decimal("500"),
        },
    )

    assert is_valid is True
    assert len(errors) == 0


def test_validate_supplier_methodology_custom_review(engine):
    """Test custom methodology requires review."""
    is_valid, warnings = engine.validate_supplier_methodology(
        methodology=SupplierMethodology.CUSTOM,
        data={"total_co2e_kg": Decimal("1000")},
    )

    assert is_valid is True
    assert len(warnings) > 0
    assert any("review" in w.lower() for w in warnings)


def test_validate_supplier_certification_smartway(engine):
    """Test SmartWay certification validation."""
    is_valid, score = engine.validate_supplier_certification(
        certification="SMARTWAY_CERTIFIED",
        carrier_id="CARRIER-123",
    )

    assert is_valid is True
    assert score > 0.8


def test_validate_supplier_certification_glec_accredited(engine):
    """Test GLEC accreditation validation."""
    is_valid, score = engine.validate_supplier_certification(
        certification="GLEC_ACCREDITED",
        verifier="DNV_GL",
    )

    assert is_valid is True
    assert score > 0.9


def test_apply_supplier_verification_score(engine):
    """Test supplier verification score adjustment."""
    base_result = LegResult(
        leg_id="LEG-001",
        co2e_kg=Decimal("1000"),
        data_quality_tier=DataQualityTier.TIER_1,
    )

    adjusted = engine.apply_supplier_verification_score(
        result=base_result,
        verification_score=0.95,
    )

    # High verification doesn't change emissions, only quality score
    assert adjusted.co2e_kg == base_result.co2e_kg
    assert adjusted.data_quality_score > base_result.data_quality_score


# ============================================================================
# Allocation Methods
# ============================================================================


def test_allocate_by_mass(engine):
    """Test mass-based allocation."""
    total_emissions = Decimal("1000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.MASS,
        shipment_mass_tonnes=Decimal("5.0"),
        total_load_mass_tonnes=Decimal("20.0"),
    )

    # 5/20 = 25%
    expected = total_emissions * Decimal("0.25")
    assert allocated == expected


def test_allocate_by_volume(engine):
    """Test volume-based allocation."""
    total_emissions = Decimal("2000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.VOLUME,
        shipment_volume_m3=Decimal("10.0"),
        total_load_volume_m3=Decimal("50.0"),
    )

    # 10/50 = 20%
    expected = total_emissions * Decimal("0.20")
    assert allocated == expected


def test_allocate_by_pallet_positions(engine):
    """Test pallet position allocation."""
    total_emissions = Decimal("1500")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.PALLET_POSITIONS,
        shipment_pallets=8,
        total_load_pallets=33,  # Standard truck capacity
    )

    # 8/33 ≈ 24.2%
    expected = total_emissions * Decimal("8") / Decimal("33")
    assert abs(allocated - expected) < Decimal("0.01")


def test_allocate_by_teu(engine):
    """Test TEU-based allocation (containers)."""
    total_emissions = Decimal("10000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.TEU,
        shipment_teu=Decimal("5.0"),  # 5 TEU
        total_load_teu=Decimal("100.0"),  # Ship capacity
    )

    # 5/100 = 5%
    expected = total_emissions * Decimal("0.05")
    assert allocated == expected


def test_allocate_by_revenue(engine):
    """Test revenue-based allocation."""
    total_emissions = Decimal("5000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.REVENUE,
        shipment_revenue=Decimal("25000.00"),
        total_load_revenue=Decimal("100000.00"),
    )

    # 25000/100000 = 25%
    expected = total_emissions * Decimal("0.25")
    assert allocated == expected


def test_allocate_by_chargeable_weight(engine):
    """Test chargeable weight allocation (air freight)."""
    total_emissions = Decimal("8000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.CHARGEABLE_WEIGHT,
        shipment_chargeable_weight_kg=Decimal("500"),
        total_load_chargeable_weight_kg=Decimal("2000"),
    )

    # 500/2000 = 25%
    expected = total_emissions * Decimal("0.25")
    assert allocated == expected


# ============================================================================
# Chargeable Weight Calculations
# ============================================================================


def test_calculate_chargeable_weight_air(engine):
    """Test air freight chargeable weight calculation."""
    # Actual mass
    actual_mass_kg = Decimal("100")
    # Volumetric weight (L×W×H / 6000 for air)
    volume_m3 = Decimal("0.5")  # 0.5 m³
    volumetric_weight_kg = volume_m3 * Decimal("167")  # Air: 167 kg/m³

    chargeable = engine.calculate_chargeable_weight(
        mode=TransportMode.AIR,
        actual_mass_kg=actual_mass_kg,
        volume_m3=volume_m3,
    )

    # Higher of actual (100) or volumetric (~83.5)
    assert chargeable == actual_mass_kg


def test_calculate_chargeable_weight_road(engine):
    """Test road freight chargeable weight (usually actual mass)."""
    chargeable = engine.calculate_chargeable_weight(
        mode=TransportMode.ROAD,
        actual_mass_kg=Decimal("5000"),
        volume_m3=Decimal("30"),
    )

    # Road typically uses actual mass
    assert chargeable == Decimal("5000")


# ============================================================================
# Refrigerated Transport
# ============================================================================


def test_calculate_reefer_emissions_road(engine):
    """Test refrigerated truck (reefer) emissions."""
    reefer_result = engine.calculate_reefer_emissions(
        mode=TransportMode.ROAD,
        distance_km=Decimal("500"),
        temperature_celsius=Decimal("-18"),  # Frozen
        vehicle_type=VehicleType.TRUCK_REFRIGERATED,
    )

    assert reefer_result["tru_fuel_co2e_kg"] > Decimal("0")
    assert reefer_result["refrigerant_leakage_co2e_kg"] >= Decimal("0")
    assert reefer_result["total_co2e_kg"] > Decimal("0")


def test_calculate_reefer_emissions_maritime(engine):
    """Test refrigerated container (reefer) maritime emissions."""
    reefer_result = engine.calculate_reefer_emissions(
        mode=TransportMode.MARITIME,
        distance_km=Decimal("10000"),
        temperature_celsius=Decimal("2"),  # Chilled
        vehicle_type=VehicleType.CONTAINER_SHIP_REEFER,
    )

    assert reefer_result["total_co2e_kg"] > Decimal("0")


def test_calculate_refrigerant_leakage(engine):
    """Test refrigerant leakage emissions."""
    leakage = engine.calculate_refrigerant_leakage(
        refrigerant_type="R-404A",
        charge_kg=Decimal("10.0"),
        annual_leakage_rate=Decimal("0.15"),  # 15%
        duration_days=30,
    )

    assert leakage > Decimal("0")
    # R-404A has high GWP (~3900)
    assert leakage > Decimal("100")  # Significant CO2e


def test_calculate_tru_fuel(engine):
    """Test Transport Refrigeration Unit (TRU) fuel consumption."""
    tru_fuel = engine.calculate_tru_fuel_consumption(
        mode=TransportMode.ROAD,
        distance_km=Decimal("1000"),
        temperature_celsius=Decimal("-25"),  # Deep frozen
        ambient_temperature_celsius=Decimal("30"),  # Hot climate
    )

    assert tru_fuel["fuel_liters"] > Decimal("0")
    assert tru_fuel["co2e_kg"] > Decimal("0")


# ============================================================================
# Aggregations
# ============================================================================


def test_aggregate_chain_results(engine, multi_leg_truck_ship_truck_chain):
    """Test aggregation of chain results."""
    result = engine.calculate_chain(multi_leg_truck_ship_truck_chain)

    aggregated = engine.aggregate_chain_results([result])

    assert aggregated["total_co2e_kg"] == result.total_co2e_kg
    assert aggregated["total_distance_km"] == result.total_distance_km
    assert "by_mode" in aggregated
    assert "by_leg" in aggregated
    assert "by_hub" in aggregated


# ============================================================================
# Consistency Checks
# ============================================================================


def test_chain_total_equals_sum_of_legs_and_hubs(
    engine, multi_leg_truck_ship_truck_chain
):
    """Test chain total emissions = sum(legs) + sum(hubs)."""
    result = engine.calculate_chain(multi_leg_truck_ship_truck_chain)

    total_legs = sum(lr.co2e_kg for lr in result.leg_results)
    total_hubs = sum(hr.co2e_kg for hr in result.hub_results)

    assert result.total_co2e_kg == total_legs + total_hubs


def test_allocation_reduces_emissions(engine):
    """Test allocation reduces allocated emissions vs. total."""
    total_emissions = Decimal("1000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.MASS,
        shipment_mass_tonnes=Decimal("5.0"),
        total_load_mass_tonnes=Decimal("20.0"),
    )

    # Allocated should be 25% of total
    assert allocated < total_emissions
    assert allocated == total_emissions * Decimal("0.25")


def test_reefer_increases_emissions(engine):
    """Test reefer transport has higher emissions than standard."""
    standard_leg = TransportLeg(
        leg_id="LEG-001",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("20.0"),
    )

    reefer_leg = TransportLeg(
        leg_id="LEG-002",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_REFRIGERATED,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("20.0"),
        temperature_celsius=Decimal("-18"),
    )

    standard_result = engine.calculate_leg(standard_leg)
    reefer_result = engine.calculate_leg(reefer_leg)

    # Reefer should have higher emissions (TRU + refrigerant)
    assert reefer_result.co2e_kg > standard_result.co2e_kg


def test_hub_emissions_positive(engine):
    """Test hub emissions are always positive."""
    hub = TransportHub(
        hub_id="HUB-001",
        hub_type="warehouse",
        operations=["storage"],
        throughput_tonnes=Decimal("100"),
    )

    result = engine.calculate_hub(hub)

    assert result.co2e_kg > Decimal("0")


# ============================================================================
# Data Quality
# ============================================================================


def test_data_quality_weighted(engine, multi_leg_truck_ship_truck_chain):
    """Test chain data quality is weighted average of legs."""
    result = engine.calculate_chain(multi_leg_truck_ship_truck_chain)

    # Calculate expected weighted average
    total_emissions = sum(lr.co2e_kg for lr in result.leg_results)
    weighted_score = sum(
        lr.co2e_kg * lr.data_quality_score for lr in result.leg_results
    ) / total_emissions

    assert abs(result.data_quality_score - weighted_score) < 0.01


# ============================================================================
# Batch Processing
# ============================================================================


def test_batch_chains(engine, single_leg_road_chain, multi_leg_truck_ship_truck_chain):
    """Test batch chain processing."""
    chains = [single_leg_road_chain, multi_leg_truck_ship_truck_chain]

    results = engine.batch_calculate_chains(chains)

    assert len(results) == 2
    assert all(isinstance(r, ChainResult) for r in results)


# ============================================================================
# Precision
# ============================================================================


def test_decimal_precision(engine, single_leg_road_chain):
    """Test Decimal precision maintained."""
    result = engine.calculate_chain(single_leg_road_chain)

    assert isinstance(result.total_co2e_kg, Decimal)
    assert isinstance(result.total_distance_km, Decimal)


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_chain_raises(engine):
    """Test empty chain raises error."""
    empty_chain = TransportChain(
        chain_id="EMPTY",
        legs=[],
        hubs=[],
    )

    with pytest.raises(ValueError, match="chain must have at least one leg"):
        engine.calculate_chain(empty_chain)


def test_negative_distance_raises(engine):
    """Test negative distance raises error."""
    invalid_leg = TransportLeg(
        leg_id="LEG-001",
        mode=TransportMode.ROAD,
        distance_km=Decimal("-100"),  # Negative
        mass_tonnes=Decimal("10"),
    )

    with pytest.raises(ValueError, match="distance must be positive"):
        engine.calculate_leg(invalid_leg)


def test_zero_mass_raises(engine):
    """Test zero mass raises error."""
    zero_mass_leg = TransportLeg(
        leg_id="LEG-001",
        mode=TransportMode.ROAD,
        distance_km=Decimal("100"),
        mass_tonnes=Decimal("0"),  # Zero
    )

    with pytest.raises(ValueError, match="mass must be positive"):
        engine.calculate_leg(zero_mass_leg)


def test_allocation_full_load_equals_total(engine):
    """Test allocation with full load equals total emissions."""
    total_emissions = Decimal("1000")
    allocated = engine.allocate_emissions(
        total_co2e_kg=total_emissions,
        allocation_method=AllocationMethod.MASS,
        shipment_mass_tonnes=Decimal("20.0"),
        total_load_mass_tonnes=Decimal("20.0"),  # Full load
    )

    # 100% allocation
    assert allocated == total_emissions


def test_very_long_chain(engine):
    """Test chain with many legs."""
    legs = [
        TransportLeg(
            leg_id=f"LEG-{i:03d}",
            mode=TransportMode.ROAD,
            distance_km=Decimal("50"),
            mass_tonnes=Decimal("10"),
        )
        for i in range(20)  # 20 legs
    ]

    long_chain = TransportChain(
        chain_id="LONG-CHAIN",
        legs=legs,
        hubs=[],
    )

    result = engine.calculate_chain(long_chain)

    assert len(result.leg_results) == 20
    assert result.total_distance_km == Decimal("1000")  # 20 × 50


def test_supplier_data_overrides_calculation(engine):
    """Test supplier-specific data overrides default calculation."""
    leg_with_supplier = TransportLeg(
        leg_id="LEG-001",
        mode=TransportMode.ROAD,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("20"),
        supplier_data={
            "methodology": SupplierMethodology.GLEC,
            "total_co2e_kg": Decimal("1500"),  # Supplier-provided
            "verification_status": "third_party_verified",
        },
    )

    result = engine.calculate_leg(leg_with_supplier)

    # Should use supplier value
    assert result.co2e_kg == Decimal("1500")
    assert result.data_quality_tier == DataQualityTier.TIER_1


def test_intermodal_container_tracked(engine, multi_leg_truck_ship_truck_chain):
    """Test intermodal container is tracked across legs."""
    # Add container ID
    for leg in multi_leg_truck_ship_truck_chain.legs:
        leg.container_id = "CONT-123456"

    result = engine.calculate_chain(multi_leg_truck_ship_truck_chain)

    # All legs should reference same container
    assert all(lr.container_id == "CONT-123456" for lr in result.leg_results)
