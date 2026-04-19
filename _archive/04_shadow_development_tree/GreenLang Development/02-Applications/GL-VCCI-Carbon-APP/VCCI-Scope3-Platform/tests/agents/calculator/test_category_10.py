# -*- coding: utf-8 -*-
"""
Unit Tests for Category 10: Processing of Sold Products
GL-VCCI Scope 3 Platform

Comprehensive test coverage for Category 10 calculator including:
- Tier 1: Customer-provided processing data
- Tier 2: Industry-specific factors
- Tier 3: LLM-estimated processing
- Edge cases and validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from services.agents.calculator.categories.category_10 import (
    Category10Calculator,
    Category10Input,
)
from services.agents.calculator.models import (
    CalculationResult,
    DataQualityInfo,
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
    """Mock FactorBroker."""
    broker = Mock()
    broker.resolve = AsyncMock()
    return broker


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient."""
    client = Mock()
    return client


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
    """Create Category10Calculator instance."""
    return Category10Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
    )


# ============================================================================
# TIER 1 TESTS: Customer-Provided Processing Data
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_direct_processing_emissions(calculator):
    """Test Tier 1 with direct processing emissions per unit."""
    input_data = Category10Input(
        product_name="Steel components",
        quantity=1000,
        quantity_unit="kg",
        region="US",
        processing_emissions_per_unit=2.5,  # kgCO2e/kg
        customer_name="Automotive Corp"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 1000 * 2.5
    assert result.emissions_tco2e == 2.5
    assert result.tier == TierType.TIER_1
    assert result.data_quality.dqi_score == 85.0


@pytest.mark.asyncio
async def test_tier1_energy_based(calculator, mock_factor_broker):
    """Test Tier 1 with energy-based calculation."""
    # Mock grid factor
    mock_grid_factor = Mock()
    mock_grid_factor.value = 0.417  # kgCO2e/kWh
    mock_grid_factor.uncertainty = 0.10
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category10Input(
        product_name="Electronic components",
        quantity=500,
        quantity_unit="units",
        region="US",
        energy_per_unit=0.5,  # kWh per unit
        customer_name="Electronics Inc"
    )

    result = await calculator.calculate(input_data)

    expected_emissions = 500 * 0.5 * 0.417  # 104.25 kgCO2e
    assert abs(result.emissions_kgco2e - expected_emissions) < 0.01
    assert result.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_tier1_large_quantity(calculator):
    """Test Tier 1 with large quantities."""
    input_data = Category10Input(
        product_name="Raw materials",
        quantity=100000,
        quantity_unit="kg",
        region="CN",
        processing_emissions_per_unit=1.2,
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 100000 * 1.2
    assert result.emissions_tco2e == 120.0


# ============================================================================
# TIER 2 TESTS: Industry-Specific Factors
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_industry_specific(calculator, mock_factor_broker):
    """Test Tier 2 with industry-specific processing factors."""
    # Mock industry processing factor
    mock_processing_factor = Mock()
    mock_processing_factor.value = 3.5  # kgCO2e/kg
    mock_processing_factor.uncertainty = 0.20
    mock_processing_factor.factor_id = "automotive_processing"
    mock_processing_factor.source = "industry_database"
    mock_processing_factor.unit = "kgCO2e/kg"
    mock_processing_factor.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    mock_processing_factor.provenance = Mock(calculation_hash="hash123")
    mock_processing_factor.data_quality_score = 70.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_processing_factor)

    input_data = Category10Input(
        product_name="Metal parts",
        quantity=2000,
        quantity_unit="kg",
        region="US",
        industry_sector="automotive",
        processing_type="machining"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 2000 * 3.5
    assert result.tier == TierType.TIER_2
    assert result.data_quality.dqi_score == 65.0


@pytest.mark.asyncio
async def test_tier2_electronics_industry(calculator, mock_factor_broker):
    """Test Tier 2 for electronics industry."""
    mock_processing_factor = Mock()
    mock_processing_factor.value = 2.8
    mock_processing_factor.uncertainty = 0.25
    mock_processing_factor.factor_id = "electronics_assembly"
    mock_processing_factor.source = "ecoinvent"
    mock_processing_factor.unit = "kgCO2e/unit"
    mock_processing_factor.metadata = Mock(
        source_version="3.9",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="GB"
    )
    mock_processing_factor.provenance = Mock(calculation_hash="hash456")
    mock_processing_factor.data_quality_score = 75.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_processing_factor)

    input_data = Category10Input(
        product_name="Circuit boards",
        quantity=1000,
        quantity_unit="units",
        region="GB",
        industry_sector="electronics",
        processing_type="assembly"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 1000 * 2.8


# ============================================================================
# TIER 3 TESTS: LLM Estimation
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_llm_estimation(calculator, mock_factor_broker):
    """Test Tier 3 with LLM-estimated processing."""
    # Mock grid factor
    mock_grid_factor = Mock()
    mock_grid_factor.value = 0.348  # Germany
    mock_grid_factor.uncertainty = 0.15
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    # Mock LLM estimation in calculator
    with patch.object(calculator, '_llm_estimate_processing', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "energy_per_unit": 1.5,  # kWh
            "processing_type": "chemical_processing",
            "confidence": 0.75,
            "reasoning": "Chemical processing typically requires moderate energy",
            "typical_processes": ["mixing", "heating", "cooling"]
        }

        input_data = Category10Input(
            product_name="Chemical intermediates",
            quantity=500,
            quantity_unit="kg",
            region="DE",
            product_description="Chemical compound for pharmaceutical industry"
        )

        result = await calculator.calculate(input_data)

        expected_emissions = 500 * 1.5 * 0.348
        assert abs(result.emissions_kgco2e - expected_emissions) < 0.01
        assert result.tier == TierType.TIER_3
        assert result.data_quality.dqi_score == 45.0


@pytest.mark.asyncio
async def test_tier3_low_confidence(calculator, mock_factor_broker):
    """Test Tier 3 with low LLM confidence."""
    mock_grid_factor = Mock()
    mock_grid_factor.value = 0.417
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    with patch.object(calculator, '_llm_estimate_processing', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "energy_per_unit": 2.0,
            "processing_type": "unknown",
            "confidence": 0.40,
            "reasoning": "Insufficient information for accurate estimate"
        }

        input_data = Category10Input(
            product_name="Unknown product",
            quantity=100,
            quantity_unit="units",
            region="US"
        )

        result = await calculator.calculate(input_data)

        assert "confidence" in result.metadata
        assert result.metadata["llm_confidence"] == 0.40


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_validation_negative_quantity(calculator):
    """Test validation fails for negative quantity."""
    input_data = Category10Input(
        product_name="Test product",
        quantity=-100,
        quantity_unit="kg",
        region="US"
    )

    with pytest.raises(ValidationError):  # Pydantic validation
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_empty_product_name(calculator):
    """Test validation fails for empty product name."""
    input_data = Category10Input(
        product_name="",
        quantity=100,
        quantity_unit="kg",
        region="US"
    )

    with pytest.raises(DataValidationError):
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_zero_quantity(calculator):
    """Test validation fails for zero quantity."""
    input_data = Category10Input(
        product_name="Test product",
        quantity=0,
        quantity_unit="kg",
        region="US"
    )

    with pytest.raises(ValidationError):  # Pydantic validation
        await calculator.calculate(input_data)


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_very_small_quantity(calculator):
    """Test with very small quantities."""
    input_data = Category10Input(
        product_name="Precision components",
        quantity=0.001,
        quantity_unit="kg",
        region="US",
        processing_emissions_per_unit=100.0
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 0.001 * 100.0
    assert result.emissions_kgco2e == 0.1


@pytest.mark.asyncio
async def test_edge_case_high_emission_factor(calculator):
    """Test with unusually high emission factor."""
    input_data = Category10Input(
        product_name="Energy-intensive processing",
        quantity=100,
        quantity_unit="kg",
        region="US",
        processing_emissions_per_unit=50.0  # Very high
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 100 * 50.0
    assert result.emissions_tco2e == 5.0


@pytest.mark.asyncio
async def test_edge_case_multiple_regions(calculator, mock_factor_broker):
    """Test calculations for different regions."""
    regions = ["US", "GB", "DE", "CN", "IN"]

    for region in regions:
        mock_grid_factor = Mock()
        mock_grid_factor.value = 0.5  # Simplified
        mock_grid_factor.uncertainty = 0.15
        mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

        input_data = Category10Input(
            product_name="Global product",
            quantity=1000,
            quantity_unit="kg",
            region=region,
            energy_per_unit=1.0
        )

        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e > 0


# ============================================================================
# TIER FALLBACK TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tier_fallback_1_to_2(calculator, mock_factor_broker):
    """Test fallback from Tier 1 to Tier 2."""
    # Tier 1 data incomplete (only quantity, no emissions)
    # Should fall back to Tier 2

    mock_processing_factor = Mock()
    mock_processing_factor.value = 2.0
    mock_processing_factor.uncertainty = 0.20
    mock_processing_factor.factor_id = "industry_processing"
    mock_processing_factor.source = "database"
    mock_processing_factor.unit = "kgCO2e/kg"
    mock_processing_factor.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    mock_processing_factor.provenance = Mock(calculation_hash="hash")
    mock_processing_factor.data_quality_score = 65.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_processing_factor)

    input_data = Category10Input(
        product_name="Components",
        quantity=500,
        quantity_unit="kg",
        region="US",
        industry_sector="manufacturing"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2


@pytest.mark.asyncio
async def test_tier_fallback_2_to_3(calculator, mock_factor_broker):
    """Test fallback from Tier 2 to Tier 3 when industry factor not found."""
    # Mock Tier 2 failure
    mock_factor_broker.resolve = AsyncMock(return_value=None)

    # Mock Tier 3 LLM success
    mock_grid_factor = Mock()
    mock_grid_factor.value = 0.417
    mock_grid_factor.uncertainty = 0.15

    call_count = [0]

    async def mock_resolve(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return None  # Industry factor not found
        else:
            return mock_grid_factor  # Grid factor found

    mock_factor_broker.resolve = mock_resolve

    with patch.object(calculator, '_llm_estimate_processing', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "energy_per_unit": 2.5,
            "processing_type": "assembly",
            "confidence": 0.65,
            "reasoning": "Estimated based on product type"
        }

        input_data = Category10Input(
            product_name="Unknown product",
            quantity=200,
            quantity_unit="units",
            region="US",
            product_description="Some product description"
        )

        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_data_quality_tier1_excellent(calculator):
    """Test that Tier 1 has excellent data quality."""
    input_data = Category10Input(
        product_name="Components",
        quantity=1000,
        quantity_unit="kg",
        region="US",
        processing_emissions_per_unit=2.0,
        customer_name="Customer A"
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.rating == "excellent"
    assert result.data_quality.dqi_score >= 80.0


@pytest.mark.asyncio
async def test_data_quality_tier3_fair(calculator, mock_factor_broker):
    """Test that Tier 3 has fair data quality."""
    mock_grid_factor = Mock()
    mock_grid_factor.value = 0.417
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    with patch.object(calculator, '_llm_estimate_processing', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "energy_per_unit": 1.5,
            "processing_type": "assembly",
            "confidence": 0.60,
            "reasoning": "Estimate"
        }

        input_data = Category10Input(
            product_name="Product",
            quantity=100,
            quantity_unit="units",
            region="US"
        )

        result = await calculator.calculate(input_data)

        assert result.data_quality.rating == "fair"
        assert result.data_quality.dqi_score <= 50.0


# ============================================================================
# METADATA TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_customer_name(calculator):
    """Test that customer name is preserved in metadata."""
    input_data = Category10Input(
        product_name="Parts",
        quantity=500,
        quantity_unit="kg",
        region="US",
        processing_emissions_per_unit=1.5,
        customer_name="ABC Corporation"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["customer_name"] == "ABC Corporation"


@pytest.mark.asyncio
async def test_metadata_product_name(calculator):
    """Test that product name is preserved in metadata."""
    input_data = Category10Input(
        product_name="Custom Components XYZ",
        quantity=1000,
        quantity_unit="units",
        region="US",
        processing_emissions_per_unit=2.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["product_name"] == "Custom Components XYZ"


# ============================================================================
# CALCULATION METHOD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_calculation_method_tier1_direct(calculator):
    """Test calculation method for Tier 1 direct emissions."""
    input_data = Category10Input(
        product_name="Product",
        quantity=100,
        quantity_unit="kg",
        region="US",
        processing_emissions_per_unit=1.0
    )

    result = await calculator.calculate(input_data)

    assert result.calculation_method == "tier_1_direct_processing_emissions"


@pytest.mark.asyncio
async def test_calculation_method_tier1_energy(calculator, mock_factor_broker):
    """Test calculation method for Tier 1 energy-based."""
    mock_grid_factor = Mock()
    mock_grid_factor.value = 0.417
    mock_grid_factor.uncertainty = 0.10
    mock_factor_broker.resolve = AsyncMock(return_value=mock_grid_factor)

    input_data = Category10Input(
        product_name="Product",
        quantity=100,
        quantity_unit="units",
        region="US",
        energy_per_unit=0.5
    )

    result = await calculator.calculate(input_data)

    assert result.calculation_method == "tier_1_energy_based"


@pytest.mark.asyncio
async def test_calculation_method_tier2(calculator, mock_factor_broker):
    """Test calculation method for Tier 2."""
    mock_processing_factor = Mock()
    mock_processing_factor.value = 2.0
    mock_processing_factor.uncertainty = 0.20
    mock_processing_factor.factor_id = "industry"
    mock_processing_factor.source = "db"
    mock_processing_factor.unit = "kgCO2e/kg"
    mock_processing_factor.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    mock_processing_factor.provenance = Mock(calculation_hash="hash")
    mock_processing_factor.data_quality_score = 65.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_processing_factor)

    input_data = Category10Input(
        product_name="Product",
        quantity=100,
        quantity_unit="kg",
        region="US",
        industry_sector="manufacturing"
    )

    result = await calculator.calculate(input_data)

    assert result.calculation_method == "tier_2_industry_specific"


# ============================================================================
# INDUSTRY SECTOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_industry_automotive(calculator, mock_factor_broker):
    """Test automotive industry processing."""
    mock_factor = Mock()
    mock_factor.value = 4.0
    mock_factor.uncertainty = 0.18
    mock_factor.factor_id = "automotive"
    mock_factor.source = "db"
    mock_factor.unit = "kgCO2e/kg"
    mock_factor.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    mock_factor.provenance = Mock(calculation_hash="hash")
    mock_factor.data_quality_score = 70.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

    input_data = Category10Input(
        product_name="Steel parts",
        quantity=1000,
        quantity_unit="kg",
        region="US",
        industry_sector="automotive",
        processing_type="stamping"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 1000 * 4.0


@pytest.mark.asyncio
async def test_industry_pharmaceutical(calculator, mock_factor_broker):
    """Test pharmaceutical industry processing."""
    mock_factor = Mock()
    mock_factor.value = 6.5  # Higher for pharmaceutical
    mock_factor.uncertainty = 0.25
    mock_factor.factor_id = "pharmaceutical"
    mock_factor.source = "db"
    mock_factor.unit = "kgCO2e/kg"
    mock_factor.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    mock_factor.provenance = Mock(calculation_hash="hash")
    mock_factor.data_quality_score = 68.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

    input_data = Category10Input(
        product_name="Active pharmaceutical ingredient",
        quantity=100,
        quantity_unit="kg",
        region="US",
        industry_sector="pharmaceutical",
        processing_type="synthesis"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 100 * 6.5


# ============================================================================
# SUMMARY TEST
# ============================================================================

@pytest.mark.asyncio
async def test_comprehensive_coverage():
    """Test that we have comprehensive coverage."""
    # This test just verifies the test file structure
    import inspect

    # Get all test functions
    test_functions = [
        name for name, obj in globals().items()
        if name.startswith('test_') and inspect.isfunction(obj)
    ]

    # Verify we have 25+ tests
    assert len(test_functions) >= 25, f"Need 25+ tests, have {len(test_functions)}"

    # Verify coverage areas
    tier1_tests = [t for t in test_functions if 'tier1' in t or 'tier_1' in t]
    tier2_tests = [t for t in test_functions if 'tier2' in t or 'tier_2' in t]
    tier3_tests = [t for t in test_functions if 'tier3' in t or 'tier_3' in t]
    validation_tests = [t for t in test_functions if 'validation' in t]
    edge_tests = [t for t in test_functions if 'edge' in t]

    assert len(tier1_tests) >= 3, "Need at least 3 Tier 1 tests"
    assert len(tier2_tests) >= 2, "Need at least 2 Tier 2 tests"
    assert len(tier3_tests) >= 2, "Need at least 2 Tier 3 tests"
    assert len(validation_tests) >= 3, "Need at least 3 validation tests"
    assert len(edge_tests) >= 3, "Need at least 3 edge case tests"
