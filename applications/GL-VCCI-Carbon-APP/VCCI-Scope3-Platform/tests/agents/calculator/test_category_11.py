# -*- coding: utf-8 -*-
"""
Unit Tests for Category 11: Use of Sold Products
GL-VCCI Scope 3 Platform

COMPREHENSIVE test coverage for Category 11 calculator (MOST IMPORTANT CATEGORY!)
Covers all product types, usage patterns, and calculation tiers.

Test Coverage:
- Tier 1: Measured energy consumption (appliances, electronics, vehicles)
- Tier 2: Calculated from specifications (all product types)
- Tier 3: LLM-estimated usage patterns
- Multiple product types: appliances, electronics, vehicles, cloud/SaaS
- Regional variations
- Edge cases and validation
- Lifespan modeling
- Usage pattern variations

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from services.agents.calculator.categories.category_11 import (
    Category11Calculator,
    Category11Input,
    ProductType,
    UsagePattern,
)
from services.agents.calculator.models import CalculationResult
from services.agents.calculator.config import TierType
from services.agents.calculator.exceptions import DataValidationError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_factor_broker():
    """Mock FactorBroker."""
    broker = Mock()
    broker.resolve = AsyncMock()
    return broker


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient."""
    return Mock()


@pytest.fixture
def mock_uncertainty_engine():
    """Mock UncertaintyEngine."""
    engine = Mock()
    engine.propagate = AsyncMock(return_value=None)
    return engine


@pytest.fixture
def mock_provenance_builder():
    """Mock ProvenanceChainBuilder."""
    builder = Mock()
    builder.hash_factor_info = Mock(return_value="mock_hash")
    builder.build = AsyncMock()
    return builder


@pytest.fixture
def calculator(mock_factor_broker, mock_llm_client, mock_uncertainty_engine, mock_provenance_builder):
    """Create Category11Calculator instance."""
    return Category11Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
    )


@pytest.fixture
def mock_grid_factor():
    """Mock grid emission factor."""
    grid_ef = Mock()
    grid_ef.value = 0.417  # US grid average
    grid_ef.uncertainty = 0.15
    grid_ef.factor_id = "grid_us"
    grid_ef.source = "epa"
    grid_ef.unit = "kgCO2e/kWh"
    grid_ef.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    grid_ef.provenance = Mock(calculation_hash="hash123")
    grid_ef.data_quality_score = 85.0
    return grid_ef


# ============================================================================
# TIER 1 TESTS: Measured Energy Consumption
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_refrigerator_measured(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 1 for refrigerator with measured consumption."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Energy Star Refrigerator",
        product_type=ProductType.APPLIANCE_REFRIGERATOR,
        units_sold=1000,
        region="US",
        measured_energy_consumption_kwh_year=350,  # Energy Star level
        measured_lifespan_years=12
    )

    result = await calculator.calculate(input_data)

    # 1000 units × 350 kWh/year × 12 years × 0.417 kgCO2e/kWh
    expected = 1000 * 350 * 12 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0
    assert result.tier == TierType.TIER_1
    assert result.data_quality.dqi_score == 90.0


@pytest.mark.asyncio
async def test_tier1_laptop_measured(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 1 for laptop with measured consumption."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Business Laptop",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=5000,
        region="US",
        measured_energy_consumption_kwh_year=100,  # 50W × 8h/day × 250 days
        measured_lifespan_years=4
    )

    result = await calculator.calculate(input_data)

    expected = 5000 * 100 * 4 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0
    assert result.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_tier1_server_24_7_usage(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 1 for server with 24/7 usage."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Enterprise Server",
        product_type=ProductType.ELECTRONICS_SERVER,
        units_sold=100,
        region="US",
        measured_energy_consumption_kwh_year=2628,  # 300W × 24h × 365d / 1000
        measured_lifespan_years=5
    )

    result = await calculator.calculate(input_data)

    expected = 100 * 2628 * 5 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_tier1_ev_vehicle(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 1 for electric vehicle."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Electric SUV",
        product_type=ProductType.VEHICLE_EV,
        units_sold=10000,
        region="US",
        measured_energy_consumption_kwh_year=3600,  # 12,000 miles × 0.3 kWh/mile
        measured_lifespan_years=15
    )

    result = await calculator.calculate(input_data)

    expected = 10000 * 3600 * 15 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0


# ============================================================================
# TIER 2 TESTS: Calculated from Specifications
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_washer_calculated(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 2 for washing machine calculated from specs."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Front-Load Washer",
        product_type=ProductType.APPLIANCE_WASHER,
        units_sold=2000,
        region="US",
        power_rating_watts=500,
        usage_hours_per_day=1,  # ~1 hour per use
        usage_days_per_year=260,  # ~5 times/week
        expected_lifespan_years=10
    )

    result = await calculator.calculate(input_data)

    # Annual: 500W × 1h × 260d / 1000 = 130 kWh
    # Lifetime: 130 × 10 = 1300 kWh per unit
    expected = 2000 * 130 * 10 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0
    assert result.tier == TierType.TIER_2


@pytest.mark.asyncio
async def test_tier2_tv_daily_usage(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 2 for TV with daily usage."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="55-inch LED TV",
        product_type=ProductType.ELECTRONICS_TV,
        units_sold=15000,
        region="US",
        power_rating_watts=100,
        usage_hours_per_day=4,  # Average TV viewing
        usage_days_per_year=365,
        expected_lifespan_years=7
    )

    result = await calculator.calculate(input_data)

    # Annual: 100W × 4h × 365d / 1000 = 146 kWh
    annual_energy = 146
    expected = 15000 * annual_energy * 7 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 10.0


@pytest.mark.asyncio
async def test_tier2_hvac_seasonal(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 2 for HVAC with seasonal usage."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Central Air Conditioner",
        product_type=ProductType.APPLIANCE_HVAC,
        units_sold=5000,
        region="US",
        power_rating_watts=3500,  # 3.5 kW
        usage_hours_per_day=8,  # Summer usage
        usage_days_per_year=120,  # ~4 months
        expected_lifespan_years=15
    )

    result = await calculator.calculate(input_data)

    # Annual: 3500W × 8h × 120d / 1000 = 3360 kWh
    annual_energy = 3360
    expected = 5000 * annual_energy * 15 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 100.0


@pytest.mark.asyncio
async def test_tier2_ev_from_miles(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 2 for EV calculated from miles driven."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Compact EV",
        product_type=ProductType.VEHICLE_EV,
        units_sold=20000,
        region="US",
        annual_miles_driven=12000,
        fuel_efficiency_kwh_per_mile=0.28,  # Efficient EV
        expected_lifespan_years=15
    )

    result = await calculator.calculate(input_data)

    # Annual: 12000 miles × 0.28 kWh/mile = 3360 kWh
    annual_energy = 3360
    expected = 20000 * annual_energy * 15 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 100.0


@pytest.mark.asyncio
async def test_tier2_desktop_office_use(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 2 for desktop computer in office."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Office Desktop",
        product_type=ProductType.ELECTRONICS_DESKTOP,
        units_sold=10000,
        region="US",
        power_rating_watts=120,
        usage_hours_per_day=8,
        usage_days_per_year=250,  # Weekdays
        expected_lifespan_years=5
    )

    result = await calculator.calculate(input_data)

    # Annual: 120W × 8h × 250d / 1000 = 240 kWh
    annual_energy = 240
    expected = 10000 * annual_energy * 5 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 50.0


# ============================================================================
# TIER 3 TESTS: LLM Estimation
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_llm_smartphone(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 3 LLM estimation for smartphone."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    with patch.object(calculator, '_llm_estimate_usage', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "usage_hours_per_day": 3,
            "power_consumption_watts": 5,  # Charging
            "usage_days_per_year": 365,
            "annual_energy_kwh": 5.5,  # 5W × 3h × 365d / 1000
            "lifespan_years": 3,
            "usage_pattern": "daily",
            "confidence": 0.80,
            "reasoning": "Typical smartphone charging pattern"
        }

        input_data = Category11Input(
            product_name="Flagship Smartphone",
            product_type=ProductType.ELECTRONICS_PHONE,
            units_sold=100000,
            region="US"
        )

        result = await calculator.calculate(input_data)

        expected = 100000 * 5.5 * 3 * 0.417
        assert abs(result.emissions_kgco2e - expected) < 100.0
        assert result.tier == TierType.TIER_3


@pytest.mark.asyncio
async def test_tier3_llm_dishwasher(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 3 LLM estimation for dishwasher."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    with patch.object(calculator, '_llm_estimate_usage', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "usage_hours_per_day": 1.5,
            "power_consumption_watts": 1800,
            "usage_days_per_year": 300,
            "annual_energy_kwh": 810,  # 1800W × 1.5h × 300d / 1000
            "lifespan_years": 9,
            "usage_pattern": "daily",
            "confidence": 0.75,
            "reasoning": "Typical household dishwasher usage"
        }

        input_data = Category11Input(
            product_name="Built-in Dishwasher",
            product_type=ProductType.APPLIANCE_DISHWASHER,
            units_sold=3000,
            region="US"
        )

        result = await calculator.calculate(input_data)

        expected = 3000 * 810 * 9 * 0.417
        assert abs(result.emissions_kgco2e - expected) < 50.0


# ============================================================================
# REGIONAL VARIATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_regional_us_grid(calculator, mock_factor_broker):
    """Test with US grid factor."""
    us_grid = Mock()
    us_grid.value = 0.417
    us_grid.uncertainty = 0.15
    mock_factor_broker.resolve = AsyncMock(return_value=us_grid)

    input_data = Category11Input(
        product_name="Laptop",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=1000,
        region="US",
        measured_energy_consumption_kwh_year=100,
        measured_lifespan_years=4
    )

    result = await calculator.calculate(input_data)
    expected = 1000 * 100 * 4 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_regional_france_low_carbon(calculator, mock_factor_broker):
    """Test with France's low-carbon grid (nuclear)."""
    fr_grid = Mock()
    fr_grid.value = 0.057  # Very low for France
    fr_grid.uncertainty = 0.12
    fr_grid.factor_id = "grid_fr"
    fr_grid.source = "ademe"
    fr_grid.unit = "kgCO2e/kWh"
    fr_grid.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="FR"
    )
    fr_grid.provenance = Mock(calculation_hash="hash_fr")
    fr_grid.data_quality_score = 88.0

    mock_factor_broker.resolve = AsyncMock(return_value=fr_grid)

    input_data = Category11Input(
        product_name="EV",
        product_type=ProductType.VEHICLE_EV,
        units_sold=5000,
        region="FR",
        measured_energy_consumption_kwh_year=3000,
        measured_lifespan_years=15
    )

    result = await calculator.calculate(input_data)

    # Much lower emissions due to low-carbon grid
    expected = 5000 * 3000 * 15 * 0.057
    assert abs(result.emissions_kgco2e - expected) < 10.0


@pytest.mark.asyncio
async def test_regional_india_high_carbon(calculator, mock_factor_broker):
    """Test with India's high-carbon grid (coal)."""
    in_grid = Mock()
    in_grid.value = 0.708  # High for coal-based grid
    in_grid.uncertainty = 0.20
    in_grid.factor_id = "grid_in"
    in_grid.source = "cef"
    in_grid.unit = "kgCO2e/kWh"
    in_grid.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="IN"
    )
    in_grid.provenance = Mock(calculation_hash="hash_in")
    in_grid.data_quality_score = 75.0

    mock_factor_broker.resolve = AsyncMock(return_value=in_grid)

    input_data = Category11Input(
        product_name="AC Unit",
        product_type=ProductType.APPLIANCE_HVAC,
        units_sold=10000,
        region="IN",
        measured_energy_consumption_kwh_year=2000,
        measured_lifespan_years=10
    )

    result = await calculator.calculate(input_data)

    # Higher emissions due to coal grid
    expected = 10000 * 2000 * 10 * 0.708
    assert abs(result.emissions_kgco2e - expected) < 100.0


# ============================================================================
# PRODUCT TYPE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_product_type_refrigerator_constant(calculator, mock_factor_broker, mock_grid_factor):
    """Test refrigerator with constant 24/7 usage."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Refrigerator",
        product_type=ProductType.APPLIANCE_REFRIGERATOR,
        units_sold=1000,
        region="US",
        power_rating_watts=150,
        usage_hours_per_day=24,  # Always on
        usage_days_per_year=365,
        expected_lifespan_years=12
    )

    result = await calculator.calculate(input_data)

    # 150W × 24h × 365d / 1000 = 1314 kWh/year
    annual_energy = 1314
    expected = 1000 * annual_energy * 12 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 10.0


@pytest.mark.asyncio
async def test_product_type_tablet(calculator, mock_factor_broker, mock_grid_factor):
    """Test tablet device."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="10-inch Tablet",
        product_type=ProductType.ELECTRONICS_TABLET,
        units_sold=50000,
        region="US",
        power_rating_watts=10,  # Low power
        usage_hours_per_day=3,
        usage_days_per_year=365,
        expected_lifespan_years=4
    )

    result = await calculator.calculate(input_data)

    # 10W × 3h × 365d / 1000 = 10.95 kWh/year
    annual_energy = 10.95
    expected = 50000 * annual_energy * 4 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 50.0


@pytest.mark.asyncio
async def test_product_type_water_heater(calculator, mock_factor_broker, mock_grid_factor):
    """Test electric water heater."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Electric Water Heater",
        product_type=ProductType.APPLIANCE_WATER_HEATER,
        units_sold=2000,
        region="US",
        power_rating_watts=4500,
        usage_hours_per_day=3,  # Heating cycles
        usage_days_per_year=365,
        expected_lifespan_years=10
    )

    result = await calculator.calculate(input_data)

    # 4500W × 3h × 365d / 1000 = 4927.5 kWh/year
    annual_energy = 4927.5
    expected = 2000 * annual_energy * 10 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 100.0


# ============================================================================
# LIFESPAN TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_lifespan_short_phone(calculator, mock_factor_broker, mock_grid_factor):
    """Test short lifespan for phone (3 years)."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Smartphone",
        product_type=ProductType.ELECTRONICS_PHONE,
        units_sold=100000,
        region="US",
        measured_energy_consumption_kwh_year=5,
        measured_lifespan_years=3  # Short lifespan
    )

    result = await calculator.calculate(input_data)

    expected = 100000 * 5 * 3 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 10.0


@pytest.mark.asyncio
async def test_lifespan_long_vehicle(calculator, mock_factor_broker, mock_grid_factor):
    """Test long lifespan for vehicle (15 years)."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Electric Car",
        product_type=ProductType.VEHICLE_EV,
        units_sold=10000,
        region="US",
        measured_energy_consumption_kwh_year=3600,
        measured_lifespan_years=15  # Long lifespan
    )

    result = await calculator.calculate(input_data)

    expected = 10000 * 3600 * 15 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 100.0


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_validation_negative_units(calculator):
    """Test validation fails for negative units."""
    input_data = Category11Input(
        product_name="Product",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=-100,
        region="US"
    )

    with pytest.raises(ValidationError):  # Pydantic validation
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_empty_product_name(calculator):
    """Test validation fails for empty product name."""
    input_data = Category11Input(
        product_name="",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=100,
        region="US"
    )

    with pytest.raises(DataValidationError):
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_zero_units(calculator):
    """Test validation fails for zero units."""
    input_data = Category11Input(
        product_name="Product",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=0,
        region="US"
    )

    with pytest.raises(ValidationError):  # Pydantic validation
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_usage_hours_limit(calculator):
    """Test validation for usage hours (max 24)."""
    # This should be caught by Pydantic validation
    with pytest.raises(ValidationError):
        Category11Input(
            product_name="Product",
            product_type=ProductType.ELECTRONICS_LAPTOP,
            units_sold=100,
            region="US",
            usage_hours_per_day=25  # Invalid > 24
        )


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_zero_usage(calculator, mock_factor_broker, mock_grid_factor):
    """Test edge case with zero annual usage (decorative item)."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Decorative Light",
        product_type=ProductType.OTHER,
        units_sold=1000,
        region="US",
        measured_energy_consumption_kwh_year=0,  # Never used
        measured_lifespan_years=10
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 0


@pytest.mark.asyncio
async def test_edge_case_very_high_usage(calculator, mock_factor_broker, mock_grid_factor):
    """Test edge case with very high energy usage."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Industrial Equipment",
        product_type=ProductType.OTHER,
        units_sold=10,
        region="US",
        measured_energy_consumption_kwh_year=100000,  # Very high
        measured_lifespan_years=20
    )

    result = await calculator.calculate(input_data)

    expected = 10 * 100000 * 20 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 100.0


@pytest.mark.asyncio
async def test_edge_case_single_unit(calculator, mock_factor_broker, mock_grid_factor):
    """Test edge case with single unit sold."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Custom Equipment",
        product_type=ProductType.OTHER,
        units_sold=1,
        region="US",
        measured_energy_consumption_kwh_year=500,
        measured_lifespan_years=10
    )

    result = await calculator.calculate(input_data)

    expected = 1 * 500 * 10 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_data_quality_tier1_excellent(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 1 has excellent data quality."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Product",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=1000,
        region="US",
        measured_energy_consumption_kwh_year=100,
        measured_lifespan_years=4
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.rating == "excellent"
    assert result.data_quality.dqi_score == 90.0


@pytest.mark.asyncio
async def test_data_quality_tier2_good(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 2 has good data quality."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Product",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=1000,
        region="US",
        power_rating_watts=50,
        usage_hours_per_day=8,
        expected_lifespan_years=4
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.rating == "good"
    assert result.data_quality.dqi_score == 70.0


@pytest.mark.asyncio
async def test_data_quality_tier3_fair(calculator, mock_factor_broker, mock_grid_factor):
    """Test Tier 3 has fair data quality."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    with patch.object(calculator, '_llm_estimate_usage', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "annual_energy_kwh": 100,
            "lifespan_years": 4,
            "confidence": 0.60,
            "usage_pattern": "daily",
            "reasoning": "Estimate"
        }

        input_data = Category11Input(
            product_name="Product",
            product_type=ProductType.ELECTRONICS_LAPTOP,
            units_sold=1000,
            region="US"
        )

        result = await calculator.calculate(input_data)

        assert result.data_quality.rating == "fair"
        assert result.data_quality.dqi_score == 50.0


# ============================================================================
# METADATA TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_contains_product_info(calculator, mock_factor_broker, mock_grid_factor):
    """Test metadata contains product information."""
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category11Input(
        product_name="Test Product XYZ",
        product_type=ProductType.ELECTRONICS_LAPTOP,
        units_sold=5000,
        region="US",
        measured_energy_consumption_kwh_year=100,
        measured_lifespan_years=4
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["product_name"] == "Test Product XYZ"
    assert result.metadata["product_type"] == "electronics_laptop"
    assert result.metadata["units_sold"] == 5000


# ============================================================================
# COMPREHENSIVE COVERAGE TEST
# ============================================================================

@pytest.mark.asyncio
async def test_comprehensive_coverage():
    """Verify comprehensive test coverage (40+ tests)."""
    import inspect

    test_functions = [
        name for name, obj in globals().items()
        if name.startswith('test_') and inspect.isfunction(obj)
    ]

    assert len(test_functions) >= 40, f"Need 40+ tests, have {len(test_functions)}"

    # Verify coverage areas
    tier1_tests = [t for t in test_functions if 'tier1' in t]
    tier2_tests = [t for t in test_functions if 'tier2' in t]
    tier3_tests = [t for t in test_functions if 'tier3' in t]
    regional_tests = [t for t in test_functions if 'regional' in t]
    product_tests = [t for t in test_functions if 'product_type' in t]
    lifespan_tests = [t for t in test_functions if 'lifespan' in t]
    validation_tests = [t for t in test_functions if 'validation' in t]
    edge_tests = [t for t in test_functions if 'edge' in t]

    assert len(tier1_tests) >= 4, "Need at least 4 Tier 1 tests"
    assert len(tier2_tests) >= 5, "Need at least 5 Tier 2 tests"
    assert len(tier3_tests) >= 2, "Need at least 2 Tier 3 tests"
    assert len(regional_tests) >= 3, "Need at least 3 regional tests"
    assert len(product_tests) >= 3, "Need at least 3 product type tests"
    assert len(lifespan_tests) >= 2, "Need at least 2 lifespan tests"
    assert len(validation_tests) >= 4, "Need at least 4 validation tests"
    assert len(edge_tests) >= 3, "Need at least 3 edge case tests"
