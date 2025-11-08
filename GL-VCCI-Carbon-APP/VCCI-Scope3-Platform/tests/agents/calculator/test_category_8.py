"""
Tests for Category 8: Upstream Leased Assets Calculator
GL-VCCI Scope 3 Platform

Comprehensive test suite with 20+ tests covering:
- Energy consumption calculations
- Floor area intensity methods
- LLM contract analysis
- Different lease types
- Edge cases and validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from services.agents.calculator.categories.category_8 import (
    Category8Calculator,
    Category8Input,
    LeaseType,
    EnergyType,
)
from services.agents.calculator.config import TierType
from services.agents.calculator.exceptions import (
    DataValidationError,
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
    engine.propagate = AsyncMock(return_value=None)
    return engine


@pytest.fixture
def mock_provenance_builder():
    """Mock ProvenanceChainBuilder for testing."""
    from services.agents.calculator.models import ProvenanceChain, DataQualityInfo

    builder = Mock()
    builder.build = AsyncMock(return_value=ProvenanceChain(
        calculation_id="test-calc-123",
        category=8,
        tier=TierType.TIER_2,
        input_data_hash="abc123",
        calculation={},
        data_quality=DataQualityInfo(
            dqi_score=75.0,
            tier=TierType.TIER_2,
            rating="good",
            pedigree_score=3.75
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
    """Create Category8Calculator with mocks."""
    return Category8Calculator(
        factor_broker=mock_factor_broker,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        llm_client=mock_llm_client
    )


# =============================================================================
# TIER 2 TESTS: Energy Consumption Data
# =============================================================================

@pytest.mark.asyncio
async def test_electricity_only_consumption(calculator):
    """Test calculation with only electricity consumption."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        electricity_kwh=10000.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    # 10000 kWh × 0.417 kgCO2e/kWh (US grid)
    expected = 10000.0 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 1.0
    assert result.category == 8
    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "actual_energy_consumption"


@pytest.mark.asyncio
async def test_multiple_energy_sources(calculator):
    """Test calculation with multiple energy sources."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        electricity_kwh=8000.0,
        natural_gas_kwh=3000.0,
        heating_kwh=1500.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    # Electricity: 8000 × 0.417 = 3336
    # Gas: 3000 × 0.184 = 552
    # Heating: 1500 × 0.110 = 165
    # Total: 4053
    expected = (8000.0 * 0.417) + (3000.0 * 0.184) + (1500.0 * 0.110)
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_natural_gas_only(calculator):
    """Test calculation with only natural gas."""
    input_data = Category8Input(
        lease_type=LeaseType.WAREHOUSE,
        natural_gas_kwh=5000.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    # 5000 kWh × 0.184 kgCO2e/kWh
    expected = 5000.0 * 0.184
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_fuel_oil_consumption(calculator):
    """Test calculation with fuel oil consumption."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        fuel_oil_liters=1000.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    # 1000 liters × 2.96 kgCO2e/liter
    expected = 1000.0 * 2.96
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_district_heating_and_cooling(calculator):
    """Test district heating and cooling emissions."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        heating_kwh=2000.0,
        cooling_kwh=1500.0,
        region="EU"
    )

    result = await calculator.calculate(input_data)

    # Heating: 2000 × 0.110 = 220
    # Cooling: 1500 × 0.095 = 142.5
    expected = (2000.0 * 0.110) + (1500.0 * 0.095)
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_different_grid_regions(calculator):
    """Test electricity factors vary by region."""
    input_us = Category8Input(
        electricity_kwh=5000.0,
        region="US"
    )

    input_gb = Category8Input(
        electricity_kwh=5000.0,
        region="GB"
    )

    result_us = await calculator.calculate(input_us)
    result_gb = await calculator.calculate(input_gb)

    # US grid is more carbon intensive than GB
    assert result_us.emissions_kgco2e > result_gb.emissions_kgco2e


# =============================================================================
# TIER 2 TESTS: Floor Area Intensity
# =============================================================================

@pytest.mark.asyncio
async def test_office_building_intensity(calculator):
    """Test office building floor area intensity calculation."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=1000.0,
        region="US",
        lease_duration_months=12
    )

    result = await calculator.calculate(input_data)

    # 1000 m² × 200 kWh/m²/year × 1 year × 0.417 kgCO2e/kWh
    estimated_energy = 1000.0 * 200.0 * 1.0
    expected = estimated_energy * 0.417
    assert abs(result.emissions_kgco2e - expected) < 10.0
    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "floor_area_intensity"


@pytest.mark.asyncio
async def test_warehouse_lower_intensity(calculator):
    """Test warehouse has lower energy intensity than office."""
    input_office = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=1000.0,
        region="US"
    )

    input_warehouse = Category8Input(
        lease_type=LeaseType.WAREHOUSE,
        floor_area_m2=1000.0,
        region="US"
    )

    result_office = await calculator.calculate(input_office)
    result_warehouse = await calculator.calculate(input_warehouse)

    # Warehouse (120 kWh/m²) < Office (200 kWh/m²)
    assert result_warehouse.emissions_kgco2e < result_office.emissions_kgco2e


@pytest.mark.asyncio
async def test_data_center_high_intensity(calculator):
    """Test data center has very high energy intensity."""
    input_data = Category8Input(
        lease_type=LeaseType.DATA_CENTER,
        floor_area_m2=500.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    # 500 m² × 1500 kWh/m²/year × 0.417
    expected = 500.0 * 1500.0 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 50.0


@pytest.mark.asyncio
async def test_partial_year_lease(calculator):
    """Test lease duration less than full year."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=1000.0,
        region="US",
        lease_duration_months=6  # Half year
    )

    result = await calculator.calculate(input_data)

    # Should be half of full year
    expected = 1000.0 * 200.0 * 0.5 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 10.0


@pytest.mark.asyncio
async def test_multi_year_lease(calculator):
    """Test lease duration longer than one year."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=800.0,
        region="US",
        lease_duration_months=24  # 2 years
    )

    result = await calculator.calculate(input_data)

    # Should be 2x annual emissions
    expected = 800.0 * 200.0 * 2.0 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 20.0


@pytest.mark.asyncio
async def test_retail_space_intensity(calculator):
    """Test retail space energy intensity."""
    input_data = Category8Input(
        lease_type=LeaseType.RETAIL_SPACE,
        floor_area_m2=500.0,
        region="EU"
    )

    result = await calculator.calculate(input_data)

    # 500 m² × 250 kWh/m²/year × 0.295 (EU grid)
    expected = 500.0 * 250.0 * 0.295
    assert abs(result.emissions_kgco2e - expected) < 10.0


# =============================================================================
# TIER 3 TESTS: LLM Contract Analysis
# =============================================================================

@pytest.mark.asyncio
async def test_llm_contract_analysis_tier3(calculator):
    """Test LLM analysis of lease contract."""
    input_data = Category8Input(
        contract_data="Lease agreement for 1200 sqm office space at 123 Main St, monthly rent $5000",
        region="US"
    )

    with patch.object(calculator, '_analyze_lease_contract') as mock_analyze:
        mock_analyze.return_value = {
            "is_lease": True,
            "lease_type": "office_building",
            "floor_area_m2": 1200.0,
            "lease_duration_months": 24,
            "address": "123 Main St",
            "confidence": 0.85
        }

        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3
        assert result.calculation_method == "llm_contract_analysis"
        assert result.metadata["llm_analyzed"] is True
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_llm_detects_not_a_lease(calculator):
    """Test LLM correctly identifies owned asset (not leased)."""
    input_data = Category8Input(
        contract_data="Purchase agreement for office building, owned by company",
        region="US"
    )

    with patch.object(calculator, '_analyze_lease_contract') as mock_analyze:
        mock_analyze.return_value = {
            "is_lease": False,
            "lease_type": "office_building",
            "confidence": 0.90
        }

        with pytest.raises(DataValidationError) as exc_info:
            await calculator.calculate(input_data)

        assert "not a leased asset" in str(exc_info.value)


@pytest.mark.asyncio
async def test_llm_extracts_building_type(calculator):
    """Test LLM correctly classifies building type from contract."""
    input_data = Category8Input(
        contract_data="Warehouse lease, 2000 square meters, industrial zone",
        region="US"
    )

    with patch.object(calculator, '_analyze_lease_contract') as mock_analyze:
        mock_analyze.return_value = {
            "is_lease": True,
            "lease_type": "warehouse",
            "floor_area_m2": 2000.0,
            "lease_duration_months": 12,
            "confidence": 0.80
        }

        result = await calculator.calculate(input_data)

        # Should use warehouse intensity
        assert result.emissions_kgco2e > 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_negative_floor_area_validation(calculator):
    """Test negative floor area is rejected."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=-100.0,
        region="US"
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "floor_area" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_negative_electricity_validation(calculator):
    """Test negative electricity consumption is rejected."""
    input_data = Category8Input(
        electricity_kwh=-5000.0,
        region="US"
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "electricity" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_zero_lease_duration(calculator):
    """Test zero lease duration is rejected."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=1000.0,
        lease_duration_months=0,
        region="US"
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "lease_duration" in str(exc_info.value)


@pytest.mark.asyncio
async def test_no_valid_input_data(calculator):
    """Test completely insufficient input data."""
    input_data = Category8Input(
        asset_id="test-asset"
        # No energy, no floor area, no contract
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "insufficient" in str(exc_info.value).lower()


# =============================================================================
# EDGE CASES & BOUNDARY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_very_small_office(calculator):
    """Test very small leased space (50 m²)."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=50.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    expected = 50.0 * 200.0 * 0.417
    assert abs(result.emissions_kgco2e - expected) < 5.0


@pytest.mark.asyncio
async def test_very_large_facility(calculator):
    """Test very large leased facility (10,000 m²)."""
    input_data = Category8Input(
        lease_type=LeaseType.MANUFACTURING_FACILITY,
        floor_area_m2=10000.0,
        region="CN"
    )

    result = await calculator.calculate(input_data)

    # Should complete without error
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_zero_electricity_consumption(calculator):
    """Test zero electricity (only gas)."""
    input_data = Category8Input(
        lease_type=LeaseType.WAREHOUSE,
        electricity_kwh=0.0,
        natural_gas_kwh=2000.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    # Should only have gas emissions
    expected = 2000.0 * 0.184
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_global_grid_factor_fallback(calculator):
    """Test unknown region uses global grid factor."""
    input_data = Category8Input(
        electricity_kwh=5000.0,
        region="XX"  # Unknown region code
    )

    result = await calculator.calculate(input_data)

    # Should use global average (0.475)
    expected = 5000.0 * 0.475
    assert abs(result.emissions_kgco2e - expected) < 10.0


# =============================================================================
# METADATA & QUALITY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_energy_method_higher_quality_than_intensity(calculator):
    """Test actual energy data has higher quality than intensity estimate."""
    input_energy = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        electricity_kwh=8000.0,
        region="US"
    )

    input_intensity = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=1000.0,
        region="US"
    )

    result_energy = await calculator.calculate(input_energy)
    result_intensity = await calculator.calculate(input_intensity)

    assert result_energy.data_quality.dqi_score > result_intensity.data_quality.dqi_score


@pytest.mark.asyncio
async def test_metadata_includes_building_details(calculator):
    """Test result metadata includes building details."""
    input_data = Category8Input(
        lease_type=LeaseType.OFFICE_BUILDING,
        floor_area_m2=1500.0,
        region="GB",
        building_name="Downtown Office"
    )

    result = await calculator.calculate(input_data)

    assert "lease_type" in result.metadata
    assert result.metadata["lease_type"] == "office_building"
    assert "floor_area_m2" in result.metadata


@pytest.mark.asyncio
async def test_warning_for_missing_lease_type(calculator):
    """Test warning when lease type not specified with energy data."""
    input_data = Category8Input(
        # No lease_type specified
        electricity_kwh=5000.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    warnings_text = " ".join(result.warnings)
    assert "lease type" in warnings_text.lower() or "not specified" in warnings_text.lower()


@pytest.mark.asyncio
async def test_emissions_kg_to_tonnes_conversion(calculator):
    """Test emissions are correctly converted to tonnes."""
    input_data = Category8Input(
        electricity_kwh=10000.0,
        region="US"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_tco2e == result.emissions_kgco2e / 1000
