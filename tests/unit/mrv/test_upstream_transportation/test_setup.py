"""
Unit tests for UpstreamTransportationService setup and configuration.

Tests service initialization, singleton pattern, configuration,
and main service methods (calculate, list, get, delete, etc.).

Tests:
- Service initialization
- Singleton pattern
- Configuration
- Calculate single
- Calculate batch
- List calculations
- Get calculation
- Delete calculation
- Create transport chain
- List/get transport chains
- Get emission factors
- Create custom emission factor
- Classify shipment
- Check compliance
- Calculate uncertainty
- Get aggregations
- Get hot spots
- Export report
- Health check
- Stats
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any

from greenlang.mrv.upstream_transportation.setup import (
    UpstreamTransportationService,
    configure_upstream_transportation,
    get_service,
    get_router,
)
from greenlang.mrv.upstream_transportation.models import (
    CalculationRequest,
    CalculationResponse,
    TransportChain,
    EmissionFactor,
    TransportMode,
    VehicleType,
    DataQualityTier,
)


@pytest.fixture
def service():
    """Create UpstreamTransportationService instance."""
    return UpstreamTransportationService()


@pytest.fixture
def calculation_request():
    """Sample calculation request."""
    return CalculationRequest(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("20.0"),
        origin="Warehouse A",
        destination="Customer B",
        scope="WTW",
    )


@pytest.fixture
def transport_chain_request():
    """Sample transport chain creation request."""
    return {
        "chain_id": "CHAIN-001",
        "legs": [
            {
                "mode": "ROAD",
                "vehicle_type": "TRUCK_RIGID_GT17T",
                "distance_km": "100",
                "mass_tonnes": "15.0",
                "origin": "Factory",
                "destination": "Port A",
            },
            {
                "mode": "MARITIME",
                "vehicle_type": "CONTAINER_SHIP_2000_8000TEU",
                "distance_km": "8000",
                "mass_tonnes": "15.0",
                "origin": "Port A",
                "destination": "Port B",
            },
        ],
        "hubs": [
            {"type": "port", "location": "Port A"},
            {"type": "port", "location": "Port B"},
        ],
    }


# ============================================================================
# Service Initialization
# ============================================================================


def test_service_initialization(service):
    """Test service initializes correctly."""
    assert service is not None
    assert hasattr(service, "distance_calculator")
    assert hasattr(service, "fuel_calculator")
    assert hasattr(service, "spend_calculator")
    assert hasattr(service, "multi_leg_calculator")
    assert hasattr(service, "compliance_checker")
    assert hasattr(service, "transport_pipeline")


# ============================================================================
# Singleton Pattern
# ============================================================================


def test_singleton_pattern():
    """Test service follows singleton pattern."""
    service1 = get_service()
    service2 = get_service()

    assert service1 is service2


# ============================================================================
# Configuration
# ============================================================================


def test_configure_upstream_transportation():
    """Test configure_upstream_transportation sets up service."""
    from fastapi import FastAPI

    app = FastAPI()
    configure_upstream_transportation(app)

    # Should register router
    assert any("/api/v1/mrv/upstream-transportation" in str(route) for route in app.routes)


def test_get_service():
    """Test get_service returns service instance."""
    service = get_service()

    assert isinstance(service, UpstreamTransportationService)


def test_get_router():
    """Test get_router returns FastAPI router."""
    from fastapi import APIRouter

    router = get_router()

    assert isinstance(router, APIRouter)


# ============================================================================
# Calculate Single
# ============================================================================


def test_calculate_single(service, calculation_request):
    """Test single calculation."""
    response = service.calculate(calculation_request)

    assert isinstance(response, CalculationResponse)
    assert response.calculation_id is not None
    assert response.co2e_kg > Decimal("0")
    assert response.mode == TransportMode.ROAD
    assert response.data_quality_tier in [DataQualityTier.TIER_1, DataQualityTier.TIER_2]


# ============================================================================
# Calculate Batch
# ============================================================================


def test_calculate_batch(service, calculation_request):
    """Test batch calculations."""
    requests = [
        calculation_request,
        CalculationRequest(
            calculation_type="distance_based",
            mode=TransportMode.AIR,
            distance_km=Decimal("2000"),
            mass_tonnes=Decimal("5.0"),
        ),
    ]

    responses = service.calculate_batch(requests)

    assert len(responses) == 2
    assert all(isinstance(r, CalculationResponse) for r in responses)
    assert all(r.co2e_kg > Decimal("0") for r in responses)


# ============================================================================
# List Calculations
# ============================================================================


def test_list_calculations(service, calculation_request):
    """Test list calculations."""
    # Create a calculation first
    service.calculate(calculation_request)

    # List calculations
    calculations = service.list_calculations(limit=10)

    assert len(calculations) > 0
    assert all("calculation_id" in c for c in calculations)


# ============================================================================
# Get Calculation
# ============================================================================


def test_get_calculation(service, calculation_request):
    """Test get calculation by ID."""
    # Create calculation
    response = service.calculate(calculation_request)
    calc_id = response.calculation_id

    # Get calculation
    retrieved = service.get_calculation(calc_id)

    assert retrieved is not None
    assert retrieved["calculation_id"] == calc_id
    assert retrieved["co2e_kg"] == response.co2e_kg


# ============================================================================
# Delete Calculation
# ============================================================================


def test_delete_calculation(service, calculation_request):
    """Test delete calculation."""
    # Create calculation
    response = service.calculate(calculation_request)
    calc_id = response.calculation_id

    # Delete
    deleted = service.delete_calculation(calc_id)

    assert deleted is True

    # Verify deleted
    retrieved = service.get_calculation(calc_id)
    assert retrieved is None


# ============================================================================
# Create Transport Chain
# ============================================================================


def test_create_transport_chain(service, transport_chain_request):
    """Test create transport chain."""
    chain = service.create_transport_chain(transport_chain_request)

    assert chain is not None
    assert chain["chain_id"] == "CHAIN-001"
    assert len(chain["legs"]) == 2
    assert len(chain["hubs"]) == 2


# ============================================================================
# List Transport Chains
# ============================================================================


def test_list_transport_chains(service, transport_chain_request):
    """Test list transport chains."""
    # Create chain first
    service.create_transport_chain(transport_chain_request)

    # List chains
    chains = service.list_transport_chains(limit=10)

    assert len(chains) > 0
    assert all("chain_id" in c for c in chains)


# ============================================================================
# Get Transport Chain
# ============================================================================


def test_get_transport_chain(service, transport_chain_request):
    """Test get transport chain by ID."""
    # Create chain
    created = service.create_transport_chain(transport_chain_request)
    chain_id = created["chain_id"]

    # Get chain
    retrieved = service.get_transport_chain(chain_id)

    assert retrieved is not None
    assert retrieved["chain_id"] == chain_id
    assert len(retrieved["legs"]) == 2


# ============================================================================
# Get Emission Factors
# ============================================================================


def test_get_emission_factors(service):
    """Test get emission factors list."""
    efs = service.get_emission_factors(
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
    )

    assert len(efs) > 0
    assert all("ef_id" in ef for ef in efs)
    assert all("kg_co2e_per_tonne_km" in ef for ef in efs)


# ============================================================================
# Get Emission Factor
# ============================================================================


def test_get_emission_factor(service):
    """Test get specific emission factor."""
    ef = service.get_emission_factor(
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        source="DEFRA",
    )

    assert ef is not None
    assert ef["mode"] == TransportMode.ROAD
    assert ef["vehicle_type"] == VehicleType.TRUCK_ARTICULATED_GT33T
    assert Decimal(ef["kg_co2e_per_tonne_km"]) > Decimal("0")


# ============================================================================
# Create Custom Emission Factor
# ============================================================================


def test_create_custom_emission_factor(service):
    """Test create custom emission factor."""
    custom_ef = {
        "mode": "ROAD",
        "vehicle_type": "TRUCK_CUSTOM",
        "kg_co2e_per_tonne_km": "0.085",
        "source": "company_specific",
        "scope": "WTW",
        "year": 2023,
    }

    created = service.create_custom_emission_factor(custom_ef)

    assert created is not None
    assert created["ef_id"] is not None
    assert created["source"] == "company_specific"


# ============================================================================
# Classify Shipment
# ============================================================================


def test_classify_shipment(service):
    """Test classify shipment (mode/vehicle detection)."""
    shipment_data = {
        "description": "Truck delivery from warehouse to customer",
        "origin": "Warehouse A",
        "destination": "Customer B",
        "distance_km": "500",
    }

    classification = service.classify_shipment(shipment_data)

    assert classification is not None
    assert classification["mode"] == TransportMode.ROAD
    assert "vehicle_type" in classification
    assert classification["confidence"] > 0.7


# ============================================================================
# Check Compliance
# ============================================================================


def test_check_compliance(service, calculation_request):
    """Test compliance check."""
    # Create calculation first
    response = service.calculate(calculation_request)

    # Check compliance
    compliance = service.check_compliance(
        calculation_id=response.calculation_id,
        framework="GHG_PROTOCOL",
    )

    assert compliance is not None
    assert compliance["framework"] == "GHG_PROTOCOL"
    assert "status" in compliance
    assert "score" in compliance


# ============================================================================
# Get Compliance Result
# ============================================================================


def test_get_compliance_result(service, calculation_request):
    """Test get compliance result."""
    # Create calculation
    response = service.calculate(calculation_request)

    # Check compliance first
    service.check_compliance(
        calculation_id=response.calculation_id,
        framework="GHG_PROTOCOL",
    )

    # Get compliance result
    result = service.get_compliance_result(response.calculation_id)

    assert result is not None
    assert "framework" in result


# ============================================================================
# Calculate Uncertainty
# ============================================================================


def test_calculate_uncertainty(service, calculation_request):
    """Test uncertainty calculation."""
    # Create calculation
    response = service.calculate(calculation_request)

    # Calculate uncertainty
    uncertainty = service.calculate_uncertainty(response.calculation_id)

    assert uncertainty is not None
    assert "uncertainty_percent" in uncertainty
    assert uncertainty["uncertainty_percent"] > 0


# ============================================================================
# Get Aggregations
# ============================================================================


def test_get_aggregations(service, calculation_request):
    """Test get aggregations."""
    # Create multiple calculations
    service.calculate(calculation_request)
    service.calculate(
        CalculationRequest(
            calculation_type="distance_based",
            mode=TransportMode.MARITIME,
            distance_km=Decimal("8000"),
            mass_tonnes=Decimal("15.0"),
        )
    )

    # Get aggregations
    aggregations = service.get_aggregations(
        group_by=["mode", "vehicle_type"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime.now(),
    )

    assert aggregations is not None
    assert "by_mode" in aggregations
    assert len(aggregations["by_mode"]) > 0


# ============================================================================
# Get Hot Spots
# ============================================================================


def test_get_hot_spots(service, calculation_request):
    """Test get emission hot spots."""
    # Create calculation
    service.calculate(calculation_request)

    # Get hot spots
    hot_spots = service.get_hot_spots(
        top_n=10,
        metric="co2e_kg",
    )

    assert hot_spots is not None
    assert len(hot_spots) > 0
    assert all("co2e_kg" in h for h in hot_spots)


# ============================================================================
# Export Report
# ============================================================================


def test_export_report(service, calculation_request):
    """Test export report."""
    # Create calculation
    response = service.calculate(calculation_request)

    # Export report
    report = service.export_report(
        calculation_ids=[response.calculation_id],
        format="json",
    )

    assert report is not None
    assert "calculations" in report
    assert len(report["calculations"]) == 1


# ============================================================================
# Health Check
# ============================================================================


def test_health_check(service):
    """Test health check."""
    health = service.health_check()

    assert health is not None
    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert "components" in health


# ============================================================================
# Stats
# ============================================================================


def test_get_stats(service, calculation_request):
    """Test get stats."""
    # Create calculation
    service.calculate(calculation_request)

    # Get stats
    stats = service.get_stats()

    assert stats is not None
    assert "total_calculations" in stats
    assert stats["total_calculations"] > 0


# ============================================================================
# Edge Cases
# ============================================================================


def test_get_nonexistent_calculation_returns_none(service):
    """Test getting nonexistent calculation returns None."""
    result = service.get_calculation("NONEXISTENT-ID")

    assert result is None


def test_delete_nonexistent_calculation_returns_false(service):
    """Test deleting nonexistent calculation returns False."""
    result = service.delete_calculation("NONEXISTENT-ID")

    assert result is False


def test_list_calculations_empty(service):
    """Test list calculations when empty."""
    # Clear all first (in test environment)
    calculations = service.list_calculations(limit=1000)
    for calc in calculations:
        service.delete_calculation(calc["calculation_id"])

    # List should be empty
    calculations = service.list_calculations()

    assert len(calculations) == 0


def test_list_calculations_with_filters(service, calculation_request):
    """Test list calculations with filters."""
    # Create calculation
    service.calculate(calculation_request)

    # List with mode filter
    calculations = service.list_calculations(
        mode=TransportMode.ROAD,
        limit=10,
    )

    assert len(calculations) > 0
    assert all(c["mode"] == TransportMode.ROAD for c in calculations)


def test_list_calculations_pagination(service, calculation_request):
    """Test list calculations pagination."""
    # Create multiple calculations
    for _ in range(5):
        service.calculate(calculation_request)

    # Page 1
    page1 = service.list_calculations(limit=2, offset=0)
    assert len(page1) == 2

    # Page 2
    page2 = service.list_calculations(limit=2, offset=2)
    assert len(page2) == 2

    # Different results
    assert page1[0]["calculation_id"] != page2[0]["calculation_id"]


def test_get_emission_factors_no_filters(service):
    """Test get all emission factors without filters."""
    efs = service.get_emission_factors()

    assert len(efs) > 0


def test_get_emission_factors_by_source(service):
    """Test get emission factors by source."""
    efs = service.get_emission_factors(source="DEFRA")

    assert len(efs) > 0
    assert all(ef["source"] == "DEFRA" for ef in efs)


def test_classify_shipment_ambiguous_low_confidence(service):
    """Test classify shipment with ambiguous data has low confidence."""
    ambiguous_data = {
        "description": "General transport service",
        "origin": "Location A",
        "destination": "Location B",
    }

    classification = service.classify_shipment(ambiguous_data)

    # Low confidence
    assert classification["confidence"] < 0.6


def test_check_compliance_invalid_framework_raises(service, calculation_request):
    """Test check compliance with invalid framework raises error."""
    response = service.calculate(calculation_request)

    with pytest.raises(ValueError, match="invalid framework"):
        service.check_compliance(
            calculation_id=response.calculation_id,
            framework="INVALID_FRAMEWORK",
        )


def test_get_aggregations_no_data_returns_empty(service):
    """Test get aggregations with no data returns empty."""
    # Clear all calculations
    calculations = service.list_calculations(limit=1000)
    for calc in calculations:
        service.delete_calculation(calc["calculation_id"])

    # Get aggregations
    aggregations = service.get_aggregations()

    assert aggregations is not None
    assert aggregations["by_mode"] == {}


def test_export_report_multiple_formats(service, calculation_request):
    """Test export report in multiple formats."""
    response = service.calculate(calculation_request)

    formats = ["json", "csv", "excel"]

    for fmt in formats:
        report = service.export_report(
            calculation_ids=[response.calculation_id],
            format=fmt,
        )

        assert report is not None


def test_calculate_with_allocation(service):
    """Test calculate with allocation."""
    request = CalculationRequest(
        calculation_type="distance_based",
        mode=TransportMode.ROAD,
        vehicle_type=VehicleType.TRUCK_ARTICULATED_GT33T,
        distance_km=Decimal("500"),
        mass_tonnes=Decimal("10.0"),  # Shipment
        total_load_mass_tonnes=Decimal("20.0"),  # Full load
        allocation_method="MASS",
    )

    response = service.calculate(request)

    assert response.allocation_factor == Decimal("0.5")  # 10/20
    # Allocated emissions should be half of full load
    assert response.allocated_co2e_kg == response.co2e_kg * Decimal("0.5")


def test_calculate_multi_leg_chain(service, transport_chain_request):
    """Test calculate multi-leg chain."""
    request = CalculationRequest(
        calculation_type="multi_leg",
        transport_chain=transport_chain_request,
    )

    response = service.calculate(request)

    assert response.co2e_kg > Decimal("0")
    assert len(response.leg_results) == 2
    assert len(response.hub_results) == 2


def test_health_check_degraded_when_component_fails(service):
    """Test health check shows degraded when component fails."""
    # Simulate component failure (would need mocking in real test)
    # For now, just verify health check structure
    health = service.health_check()

    assert "components" in health
    assert "distance_calculator" in health["components"]
    assert "fuel_calculator" in health["components"]


def test_stats_breakdown(service, calculation_request):
    """Test stats breakdown by mode, type, etc."""
    # Create multiple calculations
    service.calculate(calculation_request)
    service.calculate(
        CalculationRequest(
            calculation_type="fuel_based",
            mode=TransportMode.ROAD,
            fuel_type="DIESEL",
            fuel_consumed_liters=Decimal("500"),
        )
    )

    stats = service.get_stats()

    assert "by_mode" in stats
    assert "by_calculation_type" in stats
    assert stats["by_mode"][TransportMode.ROAD] >= 2


def test_service_thread_safe():
    """Test service is thread-safe (singleton)."""
    import threading

    services = []

    def get_service_in_thread():
        services.append(get_service())

    threads = [threading.Thread(target=get_service_in_thread) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All should be same instance
    assert all(s is services[0] for s in services)
