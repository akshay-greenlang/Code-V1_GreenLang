"""
Category 14: Franchises Calculator Tests
GL-VCCI Scope 3 Platform

Comprehensive test suite for Category 14 calculator with:
- Tier 1, 2, 3 calculations
- LLM franchise type classification
- Operational control determination
- Multi-location scenarios
- Edge cases and error handling

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from services.agents.calculator.categories.category_14 import (
    Category14Calculator,
    Category14Input,
    FranchiseType,
    OperationalControl,
)
from services.agents.calculator.models import CalculationResult
from services.agents.calculator.config import TierType
from services.agents.calculator.exceptions import DataValidationError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_factor_broker():
    broker = Mock()
    broker.get_factor = AsyncMock()
    return broker


@pytest.fixture
def mock_llm_client():
    client = Mock()
    client.classify_spend = AsyncMock()
    return client


@pytest.fixture
def mock_uncertainty_engine():
    engine = Mock()
    engine.propagate = AsyncMock()
    return engine


@pytest.fixture
def mock_provenance_builder():
    builder = Mock()
    builder.build = Mock(return_value=Mock(
        calculation_id="test_calc_id",
        timestamp=datetime.utcnow(),
        category=14,
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
    return Category14Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder
    )


# ============================================================================
# TIER 1 TESTS (Actual Energy Data)
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_total_energy(calculator):
    """Test Tier 1 with total energy data."""
    input_data = Category14Input(
        franchise_id="FRAN001",
        franchise_name="FastBurger",
        franchise_type=FranchiseType.FAST_FOOD,
        num_locations=50,
        total_energy_kwh=5000000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 14
    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e > 0
    assert result.calculation_method == "tier_1_actual_franchise_energy"
    assert result.metadata["num_locations"] == 50


@pytest.mark.asyncio
async def test_tier1_avg_energy_per_location(calculator):
    """Test Tier 1 with average energy per location."""
    input_data = Category14Input(
        franchise_id="FRAN002",
        franchise_name="CoffeeHub",
        franchise_type=FranchiseType.COFFEE_SHOP,
        num_locations=30,
        avg_energy_per_location_kwh=80000.0,
        region="GB",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    assert result.metadata["total_energy_kwh"] == 80000.0 * 30


@pytest.mark.asyncio
async def test_tier1_multi_region(calculator):
    """Test Tier 1 with locations across multiple regions."""
    input_data = Category14Input(
        franchise_id="FRAN003",
        franchise_name="GlobalRetail",
        franchise_type=FranchiseType.RETAIL_STORE,
        num_locations=100,
        total_energy_kwh=10000000.0,
        locations_by_region={
            "US": 50,
            "GB": 30,
            "DE": 20
        },
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e > 0


# ============================================================================
# TIER 2 TESTS (Revenue & Area-Based)
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_revenue_total(calculator):
    """Test Tier 2 revenue-based with total revenue."""
    input_data = Category14Input(
        franchise_id="FRAN004",
        franchise_name="QuickEats",
        franchise_type=FranchiseType.QUICK_SERVICE_RESTAURANT,
        num_locations=40,
        total_revenue_usd=20000000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "tier_2_revenue_based_estimation"
    assert "revenue_intensity" in result.metadata


@pytest.mark.asyncio
async def test_tier2_revenue_avg(calculator):
    """Test Tier 2 with average revenue per location."""
    input_data = Category14Input(
        franchise_id="FRAN005",
        franchise_name="DineWell",
        franchise_type=FranchiseType.CASUAL_DINING,
        num_locations=25,
        avg_revenue_per_location_usd=800000.0,
        region="CA",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["total_revenue_usd"] == 800000.0 * 25


@pytest.mark.asyncio
async def test_tier2_area_based(calculator):
    """Test Tier 2 area-based estimation."""
    input_data = Category14Input(
        franchise_id="FRAN006",
        franchise_name="FitLife Gym",
        franchise_type=FranchiseType.GYM_FITNESS,
        num_locations=15,
        avg_floor_area_sqm=600.0,
        region="AU",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "tier_2_area_based_estimation"
    assert "estimated_energy_kwh" in result.metadata


@pytest.mark.asyncio
async def test_tier2_convenience_store(calculator):
    """Test Tier 2 for convenience store franchise."""
    input_data = Category14Input(
        franchise_id="FRAN007",
        franchise_type=FranchiseType.CONVENIENCE_STORE,
        num_locations=60,
        avg_floor_area_sqm=130.0,
        region="JP",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["franchise_type"] == "convenience_store"


@pytest.mark.asyncio
async def test_tier2_hotel_franchise(calculator):
    """Test Tier 2 for hotel franchise."""
    input_data = Category14Input(
        franchise_id="FRAN008",
        franchise_type=FranchiseType.HOTEL,
        num_locations=10,
        avg_floor_area_sqm=2000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    # Hotels should have significant emissions
    assert result.emissions_kgco2e > 0


# ============================================================================
# TIER 3 TESTS (Benchmark Estimation)
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_benchmark_estimation(calculator):
    """Test Tier 3 with industry benchmark estimation."""
    input_data = Category14Input(
        franchise_id="FRAN009",
        franchise_name="Beauty Plus",
        franchise_type=FranchiseType.BEAUTY_SALON,
        num_locations=20,
        region="FR",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_3
    assert result.calculation_method == "tier_3_benchmark_estimation"
    assert "Using industry benchmark" in result.warnings[0]


@pytest.mark.asyncio
async def test_tier3_minimal_data(calculator):
    """Test Tier 3 with minimal data."""
    input_data = Category14Input(
        franchise_id="FRAN010",
        num_locations=5,
        region="MX",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_3
    assert result.data_quality.rating == "fair"


# ============================================================================
# LLM CLASSIFICATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_llm_classify_fast_food(calculator):
    """Test LLM classification of fast food franchise."""
    description = "Burger and fries restaurant chain"

    franchise_type = await calculator._llm_classify_franchise_type(description)

    assert franchise_type == FranchiseType.FAST_FOOD


@pytest.mark.asyncio
async def test_llm_classify_coffee_shop(calculator):
    """Test LLM classification of coffee shop."""
    description = "Coffee and pastries cafe franchise"

    franchise_type = await calculator._llm_classify_franchise_type(description)

    assert franchise_type == FranchiseType.COFFEE_SHOP


@pytest.mark.asyncio
async def test_llm_classify_gym(calculator):
    """Test LLM classification of gym franchise."""
    description = "Fitness center and health club"

    franchise_type = await calculator._llm_classify_franchise_type(description)

    assert franchise_type == FranchiseType.GYM_FITNESS


@pytest.mark.asyncio
async def test_llm_classify_auto_service(calculator):
    """Test LLM classification of auto service franchise."""
    description = "Vehicle repair and tire service"

    franchise_type = await calculator._llm_classify_franchise_type(description)

    assert franchise_type == FranchiseType.AUTO_SERVICE


@pytest.mark.asyncio
async def test_llm_operational_control_franchisee(calculator):
    """Test LLM determining franchisee full control."""
    description = "Independently operated franchise locations"

    control = await calculator._llm_determine_operational_control(description)

    assert control == OperationalControl.FRANCHISEE_FULL


@pytest.mark.asyncio
async def test_llm_operational_control_franchisor(calculator):
    """Test LLM determining franchisor full control."""
    description = "Company owned and operated locations"

    control = await calculator._llm_determine_operational_control(description)

    assert control == OperationalControl.FRANCHISOR_FULL


@pytest.mark.asyncio
async def test_llm_operational_control_partial(calculator):
    """Test LLM determining partial control."""
    description = "Mixed ownership franchise model"

    control = await calculator._llm_determine_operational_control(description)

    assert control == OperationalControl.FRANCHISOR_PARTIAL


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_auto_classification_workflow(calculator):
    """Test automatic franchise type and control classification."""
    input_data = Category14Input(
        franchise_id="FRAN011",
        franchise_description="Pizza delivery and dine-in restaurant",
        num_locations=35,
        avg_revenue_per_location_usd=650000.0,
        region="IT",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 14
    # Should auto-classify and calculate


@pytest.mark.asyncio
async def test_large_franchise_network(calculator):
    """Test calculation for large franchise network."""
    input_data = Category14Input(
        franchise_id="FRAN012",
        franchise_type=FranchiseType.QUICK_SERVICE_RESTAURANT,
        num_locations=500,
        total_energy_kwh=50000000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_tco2e > 1000.0  # Large network


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

@pytest.mark.asyncio
async def test_missing_franchise_id(calculator):
    """Test validation error for missing franchise ID."""
    input_data = Category14Input(
        franchise_id="",
        num_locations=10,
        region="US",
        reporting_year=2024
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "Franchise ID is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zero_locations(calculator):
    """Test validation error for zero locations."""
    input_data = Category14Input(
        franchise_id="FRAN013",
        num_locations=0,
        region="US",
        reporting_year=2024
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "at least 1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_single_location_franchise(calculator):
    """Test calculation for single location franchise."""
    input_data = Category14Input(
        franchise_id="FRAN014",
        franchise_type=FranchiseType.BEAUTY_SALON,
        num_locations=1,
        avg_floor_area_sqm=100.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0
    assert result.metadata["num_locations"] == 1


@pytest.mark.asyncio
async def test_very_large_franchise(calculator):
    """Test calculation for very large franchise network."""
    input_data = Category14Input(
        franchise_id="FRAN015",
        franchise_type=FranchiseType.FAST_FOOD,
        num_locations=10000,
        avg_energy_per_location_kwh=150000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_tco2e > 10000.0


# ============================================================================
# FRANCHISE TYPE SPECIFIC TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_qsr_high_intensity(calculator):
    """Test QSR has high energy intensity."""
    input_data = Category14Input(
        franchise_id="FRAN016",
        franchise_type=FranchiseType.QUICK_SERVICE_RESTAURANT,
        num_locations=30,
        avg_floor_area_sqm=180.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # QSR should have high emissions per sqm
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_cleaning_service_low_intensity(calculator):
    """Test cleaning service has lower energy intensity."""
    input_data = Category14Input(
        franchise_id="FRAN017",
        franchise_type=FranchiseType.CLEANING_SERVICE,
        num_locations=40,
        avg_floor_area_sqm=50.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Cleaning services should have lower emissions
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_education_franchise(calculator):
    """Test education franchise calculation."""
    input_data = Category14Input(
        franchise_id="FRAN018",
        franchise_type=FranchiseType.EDUCATION,
        num_locations=20,
        avg_floor_area_sqm=300.0,
        region="CA",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["franchise_type"] == "education"


@pytest.mark.asyncio
async def test_healthcare_franchise(calculator):
    """Test healthcare franchise calculation."""
    input_data = Category14Input(
        franchise_id="FRAN019",
        franchise_type=FranchiseType.HEALTHCARE,
        num_locations=12,
        avg_floor_area_sqm=250.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["franchise_type"] == "healthcare"


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_excellent_quality(calculator):
    """Test Tier 1 has excellent data quality."""
    input_data = Category14Input(
        franchise_id="FRAN020",
        num_locations=25,
        total_energy_kwh=2500000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.rating == "excellent"
    assert result.data_quality.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_tier3_fair_quality(calculator):
    """Test Tier 3 has fair data quality with warnings."""
    input_data = Category14Input(
        franchise_id="FRAN021",
        num_locations=15,
        region="DE",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.rating == "fair"
    assert len(result.warnings) > 0


# ============================================================================
# METADATA TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_completeness(calculator):
    """Test result metadata includes all relevant fields."""
    input_data = Category14Input(
        franchise_id="FRAN022",
        franchise_name="TestFranchise",
        franchise_type=FranchiseType.RETAIL_STORE,
        num_locations=40,
        total_energy_kwh=4000000.0,
        region="GB",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert "franchise_id" in result.metadata
    assert "franchise_type" in result.metadata
    assert "num_locations" in result.metadata
    assert "region" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
