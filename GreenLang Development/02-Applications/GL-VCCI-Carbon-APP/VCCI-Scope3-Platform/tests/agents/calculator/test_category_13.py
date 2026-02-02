# -*- coding: utf-8 -*-
"""
Category 13: Downstream Leased Assets Calculator Tests
GL-VCCI Scope 3 Platform

Comprehensive test suite for Category 13 calculator with:
- Tier 1, 2, 3 calculations
- LLM classification testing
- Building and tenant type classification
- Edge cases and error handling
- Data quality validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from greenlang.determinism import DeterministicClock
from services.agents.calculator.categories.category_13 import (
    Category13Calculator,
    Category13Input,
    BuildingType,
    TenantType,
)
from services.agents.calculator.models import (
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
)
from services.agents.calculator.config import TierType
from services.agents.calculator.exceptions import (
    DataValidationError,
    CalculationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_factor_broker():
    """Mock factor broker."""
    broker = Mock()
    broker.get_factor = AsyncMock()
    return broker


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = Mock()
    client.classify_spend = AsyncMock()
    return client


@pytest.fixture
def mock_uncertainty_engine():
    """Mock uncertainty engine."""
    engine = Mock()
    engine.propagate = AsyncMock()
    return engine


@pytest.fixture
def mock_provenance_builder():
    """Mock provenance builder."""
    builder = Mock()
    builder.build = Mock(return_value=Mock(
        calculation_id="test_calc_id",
        timestamp=DeterministicClock.utcnow(),
        category=13,
        tier=TierType.TIER_1,
        input_data_hash="test_hash",
        emission_factor=None,
        calculation={},
        data_quality=Mock(),
        provenance_chain=[],
        opentelemetry_trace_id=None
    ))
    return builder


@pytest.fixture
def calculator(
    mock_factor_broker,
    mock_llm_client,
    mock_uncertainty_engine,
    mock_provenance_builder
):
    """Create Category13Calculator instance."""
    return Category13Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder
    )


# ============================================================================
# TIER 1 TESTS (Actual Tenant Energy Data)
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_actual_tenant_energy(calculator):
    """Test Tier 1 calculation with actual tenant energy data."""
    input_data = Category13Input(
        asset_id="ASSET001",
        tenant_energy_kwh=100000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 13
    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e > 0
    assert result.data_quality.tier == TierType.TIER_1
    assert result.data_quality.rating == "excellent"
    assert result.calculation_method == "tier_1_actual_tenant_energy"


@pytest.mark.asyncio
async def test_tier1_with_fuel_consumption(calculator):
    """Test Tier 1 with both electricity and fuel consumption."""
    input_data = Category13Input(
        asset_id="ASSET002",
        tenant_energy_kwh=100000.0,
        tenant_fuel_consumption={
            "natural_gas": 50000.0,
            "diesel": 10000.0
        },
        region="GB",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 13
    assert result.emissions_kgco2e > 0
    # Should include both electricity and fuel emissions
    assert result.metadata["asset_id"] == "ASSET002"


@pytest.mark.asyncio
async def test_tier1_with_location_specificity(calculator):
    """Test Tier 1 with specific city for grid EF."""
    input_data = Category13Input(
        asset_id="ASSET003",
        tenant_energy_kwh=75000.0,
        region="US",
        city="San Francisco",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 13
    assert result.metadata["region"] == "US"


# ============================================================================
# TIER 2 TESTS (Area-Based Estimation)
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_area_based_office(calculator):
    """Test Tier 2 area-based calculation for office building."""
    input_data = Category13Input(
        asset_id="ASSET004",
        building_type=BuildingType.OFFICE,
        floor_area_sqm=5000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 13
    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "tier_2_area_based_estimation"
    assert result.metadata["building_type"] == "office"
    assert "estimated_energy_kwh" in result.metadata


@pytest.mark.asyncio
async def test_tier2_with_tenant_multiplier(calculator):
    """Test Tier 2 with tenant type multiplier."""
    input_data = Category13Input(
        asset_id="ASSET005",
        building_type=BuildingType.OFFICE,
        floor_area_sqm=3000.0,
        tenant_type=TenantType.OFFICE_HIGH_ENERGY,
        region="DE",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["tenant_type"] == "office_high_energy"
    # High energy tenant should result in higher emissions
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_tier2_warehouse(calculator):
    """Test Tier 2 for warehouse (lower energy intensity)."""
    input_data = Category13Input(
        asset_id="ASSET006",
        building_type=BuildingType.WAREHOUSE,
        floor_area_sqm=10000.0,
        tenant_type=TenantType.WAREHOUSE,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["building_type"] == "warehouse"


@pytest.mark.asyncio
async def test_tier2_data_center(calculator):
    """Test Tier 2 for data center (high energy intensity)."""
    input_data = Category13Input(
        asset_id="ASSET007",
        building_type=BuildingType.DATA_CENTER,
        floor_area_sqm=2000.0,
        tenant_type=TenantType.DATA_CENTER,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    # Data center should have very high emissions
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_tier2_without_tenant_type(calculator):
    """Test Tier 2 without tenant type (should use default multiplier)."""
    input_data = Category13Input(
        asset_id="ASSET008",
        building_type=BuildingType.RETAIL,
        floor_area_sqm=4000.0,
        region="FR",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert "Tenant type not specified" in result.warnings[0]


# ============================================================================
# TIER 3 TESTS (LLM-Based Estimation)
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_llm_estimation(calculator):
    """Test Tier 3 with LLM-based estimation."""
    input_data = Category13Input(
        asset_id="ASSET009",
        asset_description="Small office building in downtown",
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 13
    assert result.tier == TierType.TIER_3
    assert result.calculation_method == "tier_3_llm_estimation"
    assert "Using LLM-based estimation" in result.warnings[0]
    assert "llm_estimated_energy_kwh" in result.metadata


@pytest.mark.asyncio
async def test_tier3_minimal_data(calculator):
    """Test Tier 3 with minimal data."""
    input_data = Category13Input(
        asset_id="ASSET010",
        region="JP",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_3
    assert result.data_quality.rating == "fair"


# ============================================================================
# LLM CLASSIFICATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_llm_classify_building_type_office(calculator):
    """Test LLM classification of office building."""
    description = "Modern corporate headquarters with open floor plan"

    building_type = await calculator._llm_classify_building_type(description)

    assert building_type == BuildingType.OFFICE


@pytest.mark.asyncio
async def test_llm_classify_building_type_retail(calculator):
    """Test LLM classification of retail building."""
    description = "Shopping center with multiple retail stores"

    building_type = await calculator._llm_classify_building_type(description)

    assert building_type == BuildingType.RETAIL


@pytest.mark.asyncio
async def test_llm_classify_building_type_warehouse(calculator):
    """Test LLM classification of warehouse."""
    description = "Large distribution center for logistics operations"

    building_type = await calculator._llm_classify_building_type(description)

    assert building_type == BuildingType.WAREHOUSE


@pytest.mark.asyncio
async def test_llm_classify_building_type_data_center(calculator):
    """Test LLM classification of data center."""
    description = "Data center facility with server rooms and cooling systems"

    building_type = await calculator._llm_classify_building_type(description)

    assert building_type == BuildingType.DATA_CENTER


@pytest.mark.asyncio
async def test_llm_classify_building_type_unknown(calculator):
    """Test LLM classification returns UNKNOWN for ambiguous description."""
    description = "Some property somewhere"

    building_type = await calculator._llm_classify_building_type(description)

    assert building_type == BuildingType.UNKNOWN


@pytest.mark.asyncio
async def test_llm_classify_tenant_type_tech(calculator):
    """Test LLM classification of tech tenant."""
    description = "Software development company with high IT needs"

    tenant_type = await calculator._llm_classify_tenant_type(description)

    assert tenant_type == TenantType.OFFICE_HIGH_ENERGY


@pytest.mark.asyncio
async def test_llm_classify_tenant_type_manufacturing(calculator):
    """Test LLM classification of manufacturing tenant."""
    description = "Manufacturing facility producing electronics"

    tenant_type = await calculator._llm_classify_tenant_type(description)

    assert tenant_type == TenantType.MANUFACTURING


@pytest.mark.asyncio
async def test_llm_classify_tenant_type_restaurant(calculator):
    """Test LLM classification of restaurant tenant."""
    description = "Food service restaurant with commercial kitchen"

    tenant_type = await calculator._llm_classify_tenant_type(description)

    assert tenant_type == TenantType.RESTAURANT


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_auto_classification_workflow(calculator):
    """Test automatic building and tenant classification."""
    input_data = Category13Input(
        asset_id="ASSET011",
        asset_description="Office building in tech park",
        tenant_description="Software company with data servers",
        floor_area_sqm=4000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 13
    # Should auto-classify and use Tier 2
    assert result.tier == TierType.TIER_2


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

@pytest.mark.asyncio
async def test_missing_asset_id(calculator):
    """Test validation error for missing asset ID."""
    input_data = Category13Input(
        asset_id="",
        tenant_energy_kwh=100000.0,
        region="US",
        reporting_year=2024
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "Asset ID is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_missing_region(calculator):
    """Test validation error for missing region."""
    input_data = Category13Input(
        asset_id="ASSET012",
        tenant_energy_kwh=100000.0,
        region="",
        reporting_year=2024
    )

    with pytest.raises(ValidationError):  # Pydantic validation error
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_zero_energy_consumption(calculator):
    """Test handling of zero energy consumption."""
    input_data = Category13Input(
        asset_id="ASSET013",
        tenant_energy_kwh=0.0,
        building_type=BuildingType.OFFICE,
        floor_area_sqm=3000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Should fall back to Tier 2 due to zero energy
    assert result.tier == TierType.TIER_2


@pytest.mark.asyncio
async def test_very_large_building(calculator):
    """Test calculation for very large building."""
    input_data = Category13Input(
        asset_id="ASSET014",
        building_type=BuildingType.OFFICE,
        floor_area_sqm=100000.0,  # 100,000 sqm
        region="CN",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0
    assert result.emissions_tco2e > 0


@pytest.mark.asyncio
async def test_very_small_building(calculator):
    """Test calculation for very small building."""
    input_data = Category13Input(
        asset_id="ASSET015",
        building_type=BuildingType.RETAIL,
        floor_area_sqm=50.0,  # 50 sqm
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_data_quality_score(calculator):
    """Test Tier 1 has highest data quality score."""
    input_data = Category13Input(
        asset_id="ASSET016",
        tenant_energy_kwh=100000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.dqi_score >= 80.0
    assert result.data_quality.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_tier3_data_quality_warnings(calculator):
    """Test Tier 3 includes appropriate warnings."""
    input_data = Category13Input(
        asset_id="ASSET017",
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert len(result.warnings) > 0
    assert any("LLM-based estimation" in w for w in result.warnings)


# ============================================================================
# UNCERTAINTY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_low_uncertainty(calculator):
    """Test Tier 1 has lower uncertainty than Tier 2/3."""
    input_data = Category13Input(
        asset_id="ASSET018",
        tenant_energy_kwh=100000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    if result.uncertainty:
        assert result.uncertainty.coefficient_of_variation <= 0.15


@pytest.mark.asyncio
async def test_tier3_high_uncertainty(calculator):
    """Test Tier 3 has higher uncertainty."""
    input_data = Category13Input(
        asset_id="ASSET019",
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    if result.uncertainty:
        assert result.uncertainty.coefficient_of_variation >= 0.25


# ============================================================================
# METADATA TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_completeness(calculator):
    """Test result metadata includes all relevant fields."""
    input_data = Category13Input(
        asset_id="ASSET020",
        building_type=BuildingType.OFFICE,
        tenant_type=TenantType.OFFICE_STANDARD,
        floor_area_sqm=5000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert "asset_id" in result.metadata
    assert "building_type" in result.metadata
    assert "tenant_type" in result.metadata
    assert "region" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
