"""
Tests for Category 9: Downstream Transportation & Distribution Calculator
GL-VCCI Scope 3 Platform

Comprehensive test suite with 30+ tests covering:
- ISO 14083 compliance (similar to Category 4)
- All transport modes
- LLM route analysis and carrier selection
- Last-mile delivery estimation
- Load factor optimization
- Edge cases and validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from services.agents.calculator.categories.category_9 import (
    Category9Calculator,
    Category9Input,
)
from services.agents.calculator.config import TierType, TransportMode
from services.agents.calculator.exceptions import (
    DataValidationError,
    TransportModeError,
    CalculationError,
)


@pytest.fixture
def mock_factor_broker():
    """Mock FactorBroker for testing."""
    broker = Mock()
    broker.resolve = AsyncMock()
    return broker


@pytest.fixture
def mock_uncertainty_engine():
    """Mock UncertaintyEngine for testing."""
    engine = Mock()
    engine.propagate_logistics = AsyncMock(return_value=None)
    return engine


@pytest.fixture
def mock_provenance_builder():
    """Mock ProvenanceChainBuilder for testing."""
    from services.agents.calculator.models import ProvenanceChain, DataQualityInfo

    builder = Mock()
    builder.build = AsyncMock(return_value=ProvenanceChain(
        calculation_id="test-calc-123",
        category=9,
        tier=TierType.TIER_2,
        input_data_hash="abc123",
        calculation={},
        data_quality=DataQualityInfo(
            dqi_score=70.0,
            tier=TierType.TIER_2,
            rating="good",
            pedigree_score=3.5
        ),
        provenance_chain=[]
    ))
    return builder


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient for testing."""
    client = Mock()
    return client


@pytest.fixture
def calculator(mock_factor_broker, mock_uncertainty_engine, mock_provenance_builder, mock_llm_client):
    """Create Category9Calculator with mocks."""
    return Category9Calculator(
        factor_broker=mock_factor_broker,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        llm_client=mock_llm_client
    )


# =============================================================================
# TIER 2 TESTS: Detailed Shipping Data (ISO 14083)
# =============================================================================

@pytest.mark.asyncio
async def test_road_truck_medium_tier2(calculator):
    """Test medium truck road transport (ISO 14083)."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        load_factor=1.0
    )

    result = await calculator.calculate(input_data)

    # 100 km × 5 t × 0.110 kgCO2e/t-km
    expected = 100.0 * 5.0 * 0.110
    assert abs(result.emissions_kgco2e - expected) < 0.01
    assert result.category == 9
    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "iso_14083_downstream_logistics"
    assert result.metadata["iso_14083_compliant"] is True


@pytest.mark.asyncio
async def test_road_truck_heavy_tier2(calculator):
    """Test heavy truck has lower emissions per tonne-km."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_HEAVY,
        distance_km=200.0,
        weight_tonnes=15.0,
        load_factor=1.0
    )

    result = await calculator.calculate(input_data)

    # 200 km × 15 t × 0.062 kgCO2e/t-km
    expected = 200.0 * 15.0 * 0.062
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_road_van_tier2(calculator):
    """Test van transport (last-mile delivery)."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_VAN,
        distance_km=50.0,
        weight_tonnes=0.5,
        load_factor=1.0,
        is_last_mile=True
    )

    result = await calculator.calculate(input_data)

    # 50 km × 0.5 t × 0.250 kgCO2e/t-km
    expected = 50.0 * 0.5 * 0.250
    assert abs(result.emissions_kgco2e - expected) < 0.01
    assert result.metadata["is_last_mile"] is True


@pytest.mark.asyncio
async def test_sea_container_tier2(calculator):
    """Test sea container shipping (low emissions)."""
    input_data = Category9Input(
        transport_mode=TransportMode.SEA_CONTAINER,
        distance_km=5000.0,
        weight_tonnes=20.0,
        load_factor=1.0
    )

    result = await calculator.calculate(input_data)

    # 5000 km × 20 t × 0.012 kgCO2e/t-km
    expected = 5000.0 * 20.0 * 0.012
    assert abs(result.emissions_kgco2e - expected) < 0.1


@pytest.mark.asyncio
async def test_air_cargo_tier2(calculator):
    """Test air cargo (high emissions)."""
    input_data = Category9Input(
        transport_mode=TransportMode.AIR_CARGO,
        distance_km=1000.0,
        weight_tonnes=2.0,
        load_factor=1.0
    )

    result = await calculator.calculate(input_data)

    # 1000 km × 2 t × 0.680 kgCO2e/t-km
    expected = 1000.0 * 2.0 * 0.680
    assert abs(result.emissions_kgco2e - expected) < 0.1


@pytest.mark.asyncio
async def test_rail_freight_tier2(calculator):
    """Test rail freight (efficient long-distance)."""
    input_data = Category9Input(
        transport_mode=TransportMode.RAIL_FREIGHT,
        distance_km=800.0,
        weight_tonnes=50.0,
        load_factor=1.0
    )

    result = await calculator.calculate(input_data)

    # 800 km × 50 t × 0.022 kgCO2e/t-km
    expected = 800.0 * 50.0 * 0.022
    assert abs(result.emissions_kgco2e - expected) < 0.1


@pytest.mark.asyncio
async def test_load_factor_adjustment(calculator):
    """Test load factor reduces emissions per tonne."""
    input_full = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        load_factor=1.0  # Full load
    )

    input_half = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        load_factor=0.5  # Half load
    )

    result_full = await calculator.calculate(input_full)
    result_half = await calculator.calculate(input_half)

    # Half load should be 2x emissions per tonne
    assert result_half.emissions_kgco2e == pytest.approx(result_full.emissions_kgco2e * 2, rel=0.01)


@pytest.mark.asyncio
async def test_low_load_factor_warning(calculator):
    """Test warning for low load factor."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        load_factor=0.6  # Below 70%
    )

    result = await calculator.calculate(input_data)

    warnings_text = " ".join(result.warnings)
    assert "load factor" in warnings_text.lower()


@pytest.mark.asyncio
async def test_custom_emission_factor(calculator):
    """Test using custom emission factor."""
    custom_ef = 0.200

    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        emission_factor=custom_ef,
        load_factor=1.0
    )

    result = await calculator.calculate(input_data)

    # Should use custom EF
    expected = 100.0 * 5.0 * custom_ef
    assert abs(result.emissions_kgco2e - expected) < 0.01


# =============================================================================
# TRANSPORT MODE COMPARISON TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_air_vs_sea_emissions(calculator):
    """Test air cargo has much higher emissions than sea freight."""
    distance = 3000.0
    weight = 10.0

    input_air = Category9Input(
        transport_mode=TransportMode.AIR_CARGO,
        distance_km=distance,
        weight_tonnes=weight
    )

    input_sea = Category9Input(
        transport_mode=TransportMode.SEA_CONTAINER,
        distance_km=distance,
        weight_tonnes=weight
    )

    result_air = await calculator.calculate(input_air)
    result_sea = await calculator.calculate(input_sea)

    # Air should be 50+ times more emissions
    assert result_air.emissions_kgco2e > result_sea.emissions_kgco2e * 50


@pytest.mark.asyncio
async def test_rail_vs_truck_emissions(calculator):
    """Test rail freight has lower emissions than road truck."""
    distance = 500.0
    weight = 20.0

    input_truck = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=distance,
        weight_tonnes=weight
    )

    input_rail = Category9Input(
        transport_mode=TransportMode.RAIL_FREIGHT,
        distance_km=distance,
        weight_tonnes=weight
    )

    result_truck = await calculator.calculate(input_truck)
    result_rail = await calculator.calculate(input_rail)

    assert result_rail.emissions_kgco2e < result_truck.emissions_kgco2e


# =============================================================================
# TIER 3 TESTS: LLM Route Analysis
# =============================================================================

@pytest.mark.asyncio
async def test_llm_route_analysis_tier3(calculator):
    """Test LLM analysis of delivery route."""
    input_data = Category9Input(
        warehouse_address="123 Distribution Center, City A",
        customer_address="456 Customer St, City B",
        weight_tonnes=2.0,
        product_description="Electronics equipment"
    )

    with patch.object(calculator, '_analyze_delivery_route') as mock_analyze:
        mock_analyze.return_value = {
            "distance_km": 85.0,
            "transport_mode": "road_truck_medium",
            "is_last_mile": False,
            "load_factor": 0.75,
            "carrier": "Regional freight",
            "confidence": 0.80,
            "reasoning": "Regional delivery via truck"
        }

        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3
        assert result.calculation_method == "llm_route_analysis"
        assert result.metadata["llm_analyzed"] is True
        assert result.metadata["llm_confidence"] == 0.80
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_llm_last_mile_detection(calculator):
    """Test LLM detects last-mile delivery."""
    input_data = Category9Input(
        warehouse_address="Local Distribution Center",
        customer_address="123 Residential St, Apartment 4B",
        weight_tonnes=0.05,  # Small package
        delivery_instructions="Doorstep delivery"
    )

    with patch.object(calculator, '_analyze_delivery_route') as mock_analyze:
        mock_analyze.return_value = {
            "distance_km": 15.0,
            "transport_mode": "road_van",
            "is_last_mile": True,
            "load_factor": 0.60,
            "carrier": "Local courier",
            "confidence": 0.85
        }

        result = await calculator.calculate(input_data)

        assert result.metadata["is_last_mile"] is True
        assert "last-mile" in " ".join(result.warnings).lower()


@pytest.mark.asyncio
async def test_llm_carrier_recommendation(calculator):
    """Test LLM recommends appropriate carrier."""
    input_data = Category9Input(
        warehouse_address="Port of LA",
        customer_address="New York customer",
        weight_tonnes=15.0,
        product_description="Industrial equipment"
    )

    with patch.object(calculator, '_analyze_delivery_route') as mock_analyze:
        mock_analyze.return_value = {
            "distance_km": 4500.0,
            "transport_mode": "rail_freight",
            "is_last_mile": False,
            "load_factor": 0.85,
            "carrier": "Transcontinental rail",
            "confidence": 0.75,
            "reasoning": "Long-haul, heavy freight best by rail"
        }

        result = await calculator.calculate(input_data)

        assert "reasoning" in result.metadata.get("llm_reasoning", "").lower()


# =============================================================================
# TIER 3 TESTS: Aggregate Shipping
# =============================================================================

@pytest.mark.asyncio
async def test_aggregate_shipping_tier3(calculator):
    """Test aggregate shipping statistics calculation."""
    input_data = Category9Input(
        total_shipments=500,
        average_distance_km=120.0,
        average_weight_tonnes=1.5
    )

    result = await calculator.calculate(input_data)

    # 500 × 120 × 1.5 × 0.110 / 0.75
    expected = 500 * 120.0 * 1.5 * 0.110 / 0.75
    assert abs(result.emissions_kgco2e - expected) < 50.0
    assert result.tier == TierType.TIER_3
    assert result.calculation_method == "aggregate_shipping_statistics"


@pytest.mark.asyncio
async def test_aggregate_low_data_quality(calculator):
    """Test aggregate method has lower data quality."""
    input_data = Category9Input(
        total_shipments=100,
        average_distance_km=100.0,
        average_weight_tonnes=2.0
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.dqi_score < 50.0
    assert result.data_quality.rating in ["fair", "poor"]
    assert len(result.warnings) > 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_negative_distance_validation(calculator):
    """Test negative distance is rejected."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=-100.0,
        weight_tonnes=5.0
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "distance" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_negative_weight_validation(calculator):
    """Test negative weight is rejected."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=-5.0
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "weight" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_invalid_load_factor(calculator):
    """Test invalid load factor is rejected."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        load_factor=1.5  # > 1.0
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "load_factor" in str(exc_info.value)


@pytest.mark.asyncio
async def test_no_valid_input_data(calculator):
    """Test completely insufficient input data."""
    input_data = Category9Input(
        shipment_id="test-123"
        # No mode/distance/weight, no addresses, no aggregate
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "insufficient" in str(exc_info.value).lower()


# =============================================================================
# EDGE CASES & BOUNDARY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_very_short_distance(calculator):
    """Test very short delivery (5km local)."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_VAN,
        distance_km=5.0,
        weight_tonnes=0.1
    )

    result = await calculator.calculate(input_data)

    expected = 5.0 * 0.1 * 0.250
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_very_long_distance(calculator):
    """Test intercontinental shipping (10,000km)."""
    input_data = Category9Input(
        transport_mode=TransportMode.SEA_CONTAINER,
        distance_km=10000.0,
        weight_tonnes=50.0
    )

    result = await calculator.calculate(input_data)

    # Should complete without error
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_very_small_package(calculator):
    """Test very small package (0.01 tonnes = 10kg)."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_VAN,
        distance_km=20.0,
        weight_tonnes=0.01
    )

    result = await calculator.calculate(input_data)

    expected = 20.0 * 0.01 * 0.250
    assert abs(result.emissions_kgco2e - expected) < 0.001


@pytest.mark.asyncio
async def test_very_heavy_shipment(calculator):
    """Test very heavy bulk shipment (100 tonnes)."""
    input_data = Category9Input(
        transport_mode=TransportMode.SEA_BULK,
        distance_km=5000.0,
        weight_tonnes=100.0
    )

    result = await calculator.calculate(input_data)

    expected = 5000.0 * 100.0 * 0.008
    assert abs(result.emissions_kgco2e - expected) < 10.0


@pytest.mark.asyncio
async def test_inland_waterway(calculator):
    """Test inland waterway transport."""
    input_data = Category9Input(
        transport_mode=TransportMode.INLAND_WATERWAY,
        distance_km=300.0,
        weight_tonnes=20.0
    )

    result = await calculator.calculate(input_data)

    expected = 300.0 * 20.0 * 0.031
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_electric_rail_freight(calculator):
    """Test electric rail has lower emissions than diesel rail."""
    input_diesel = Category9Input(
        transport_mode=TransportMode.RAIL_FREIGHT_DIESEL,
        distance_km=500.0,
        weight_tonnes=30.0
    )

    input_electric = Category9Input(
        transport_mode=TransportMode.RAIL_FREIGHT_ELECTRIC,
        distance_km=500.0,
        weight_tonnes=30.0
    )

    result_diesel = await calculator.calculate(input_diesel)
    result_electric = await calculator.calculate(input_electric)

    assert result_electric.emissions_kgco2e < result_diesel.emissions_kgco2e


# =============================================================================
# METADATA & QUALITY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_metadata_includes_tonne_km(calculator):
    """Test result metadata includes tonne-km."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=150.0,
        weight_tonnes=8.0
    )

    result = await calculator.calculate(input_data)

    assert "tonne_km" in result.metadata
    expected_tkm = 150.0 * 8.0
    assert result.metadata["tonne_km"] == expected_tkm


@pytest.mark.asyncio
async def test_metadata_includes_transport_mode(calculator):
    """Test result metadata includes transport mode."""
    input_data = Category9Input(
        transport_mode=TransportMode.SEA_CONTAINER,
        distance_km=3000.0,
        weight_tonnes=20.0
    )

    result = await calculator.calculate(input_data)

    assert "transport_mode" in result.metadata
    assert result.metadata["transport_mode"] == "sea_container"


@pytest.mark.asyncio
async def test_iso14083_compliance_flag(calculator):
    """Test ISO 14083 compliance flag in metadata."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_HEAVY,
        distance_km=200.0,
        weight_tonnes=10.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata.get("iso_14083_compliant") is True


@pytest.mark.asyncio
async def test_emissions_kg_to_tonnes_conversion(calculator):
    """Test emissions are correctly converted to tonnes."""
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_tco2e == result.emissions_kgco2e / 1000


# =============================================================================
# INTEGRATION WITH CATEGORY 4 TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_same_formula_as_category_4(calculator):
    """Test Category 9 uses same ISO 14083 formula as Category 4."""
    # This is conceptual - both should use: distance × weight × EF / load_factor
    input_data = Category9Input(
        transport_mode=TransportMode.ROAD_TRUCK_MEDIUM,
        distance_km=100.0,
        weight_tonnes=5.0,
        load_factor=0.8
    )

    result = await calculator.calculate(input_data)

    # Manual calculation with ISO 14083 formula
    expected = 100.0 * 5.0 * 0.110 / 0.8
    assert abs(result.emissions_kgco2e - expected) < 0.01
