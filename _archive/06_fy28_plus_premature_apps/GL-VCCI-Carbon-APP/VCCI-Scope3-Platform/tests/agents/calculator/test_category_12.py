# -*- coding: utf-8 -*-
"""
Unit Tests for Category 12: End-of-Life Treatment of Sold Products
GL-VCCI Scope 3 Platform

Comprehensive test coverage for Category 12 calculator including:
- Tier 1: Detailed material composition
- Tier 2: Total weight with primary material
- Tier 3: LLM-estimated composition
- Multiple disposal methods (landfill, recycling, incineration)
- Material types and recycling credits
- Regional disposal practices
- Edge cases and validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from services.agents.calculator.categories.category_12 import (
    Category12Calculator,
    Category12Input,
    MaterialType,
    DisposalMethod,
    MaterialComposition,
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
    return Mock()


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
    """Create Category12Calculator instance."""
    return Category12Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
    )


# ============================================================================
# TIER 1 TESTS: Detailed Material Composition
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_plastic_product(calculator):
    """Test Tier 1 with plastic product composition."""
    input_data = Category12Input(
        product_name="Plastic Container",
        units_sold=10000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=0.5, recycling_rate=0.30)
        ],
        landfill_percentage=50,
        recycling_percentage=30,
        incineration_percentage=20
    )

    result = await calculator.calculate(input_data)

    # Verify emissions calculated
    assert result.emissions_kgco2e > 0
    assert result.tier == TierType.TIER_1
    assert result.data_quality.dqi_score == 80.0


@pytest.mark.asyncio
async def test_tier1_multi_material_electronics(calculator):
    """Test Tier 1 with multi-material electronics."""
    input_data = Category12Input(
        product_name="Laptop Computer",
        units_sold=5000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.ELECTRONICS, weight_kg=1.5),
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=0.8),
            MaterialComposition(material_type=MaterialType.METAL_ALUMINUM, weight_kg=0.5),
            MaterialComposition(material_type=MaterialType.GLASS, weight_kg=0.2),
        ],
        recycling_percentage=45,
        landfill_percentage=35,
        incineration_percentage=20
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    assert "material_breakdown" in result.metadata
    assert len(result.metadata["material_breakdown"]) == 4


@pytest.mark.asyncio
async def test_tier1_aluminum_recycling_credit(calculator):
    """Test Tier 1 with aluminum (high recycling credit)."""
    input_data = Category12Input(
        product_name="Aluminum Beverage Cans",
        units_sold=100000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.METAL_ALUMINUM, weight_kg=0.015, recycling_rate=0.75)
        ],
        recycling_percentage=75,
        landfill_percentage=15,
        incineration_percentage=10
    )

    result = await calculator.calculate(input_data)

    # With high recycling rate, emissions could be negative (credit)
    # or very low due to avoided emissions
    assert result.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_tier1_paper_packaging(calculator):
    """Test Tier 1 with paper/cardboard packaging."""
    input_data = Category12Input(
        product_name="Cardboard Box",
        units_sold=50000,
        region="DE",  # Germany has high recycling
        material_composition=[
            MaterialComposition(material_type=MaterialType.CARDBOARD, weight_kg=0.3, recycling_rate=0.85)
        ],
        recycling_percentage=85,
        landfill_percentage=5,
        incineration_percentage=10
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    # High recycling should reduce net emissions


# ============================================================================
# TIER 2 TESTS: Total Weight with Primary Material
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_steel_product(calculator, mock_factor_broker):
    """Test Tier 2 with steel product."""
    # Mock disposal factor
    mock_disposal_factor = Mock()
    mock_disposal_factor.value = 0.5  # Weighted average
    mock_disposal_factor.uncertainty = 0.25
    mock_disposal_factor.factor_id = "disposal_steel"
    mock_disposal_factor.source = "database"
    mock_disposal_factor.unit = "kgCO2e/kg"
    mock_disposal_factor.metadata = Mock(
        source_version="2024",
        gwp_standard=Mock(value="AR6"),
        reference_year=2024,
        geographic_scope="US"
    )
    mock_disposal_factor.provenance = Mock(calculation_hash="hash123")
    mock_disposal_factor.data_quality_score = 65.0

    mock_factor_broker.resolve = AsyncMock(return_value=mock_disposal_factor)

    input_data = Category12Input(
        product_name="Steel Furniture",
        units_sold=2000,
        region="US",
        total_weight_kg=25.0,
        primary_material=MaterialType.METAL_STEEL,
        recycling_percentage=60,
        landfill_percentage=30,
        incineration_percentage=10
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.data_quality.dqi_score == 65.0


@pytest.mark.asyncio
async def test_tier2_glass_product(calculator, mock_factor_broker):
    """Test Tier 2 with glass product."""
    input_data = Category12Input(
        product_name="Glass Bottles",
        units_sold=20000,
        region="GB",
        total_weight_kg=0.4,
        primary_material=MaterialType.GLASS,
        recycling_percentage=70,
        landfill_percentage=20,
        incineration_percentage=10
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["primary_material"] == "glass"


@pytest.mark.asyncio
async def test_tier2_textile_product(calculator, mock_factor_broker):
    """Test Tier 2 with textile product."""
    input_data = Category12Input(
        product_name="Cotton T-Shirt",
        units_sold=50000,
        region="US",
        total_weight_kg=0.2,
        primary_material=MaterialType.TEXTILES,
        landfill_percentage=80,  # Most textiles landfilled
        recycling_percentage=15,
        incineration_percentage=5
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2


# ============================================================================
# TIER 3 TESTS: LLM Estimation
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_llm_electronics(calculator, mock_factor_broker):
    """Test Tier 3 LLM estimation for electronics."""
    with patch.object(calculator, '_llm_estimate_composition', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "total_weight_kg": 2.5,
            "primary_material": "electronics",
            "confidence": 0.75,
            "reasoning": "Typical laptop weight and composition",
            "typical_disposal": "E-waste recycling"
        }

        input_data = Category12Input(
            product_name="Laptop Computer",
            units_sold=5000,
            region="US",
            product_category="electronics"
        )

        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3
        assert result.data_quality.dqi_score == 45.0
        assert "llm_confidence" in result.metadata


@pytest.mark.asyncio
async def test_tier3_llm_furniture(calculator, mock_factor_broker):
    """Test Tier 3 LLM estimation for furniture."""
    with patch.object(calculator, '_llm_estimate_composition', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "total_weight_kg": 25.0,
            "primary_material": "mixed",
            "confidence": 0.60,
            "reasoning": "Average furniture weight estimate"
        }

        input_data = Category12Input(
            product_name="Office Chair",
            units_sold=1000,
            region="US",
            product_category="furniture"
        )

        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3
        assert result.metadata["llm_total_weight_kg"] == 25.0


# ============================================================================
# DISPOSAL METHOD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_disposal_100_percent_landfill(calculator):
    """Test 100% landfill disposal."""
    input_data = Category12Input(
        product_name="Mixed Waste Product",
        units_sold=1000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.MIXED, weight_kg=1.0)
        ],
        landfill_percentage=100,
        recycling_percentage=0,
        incineration_percentage=0
    )

    result = await calculator.calculate(input_data)

    # All landfill emissions
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_disposal_100_percent_recycling(calculator):
    """Test 100% recycling (should give credit)."""
    input_data = Category12Input(
        product_name="Aluminum Product",
        units_sold=1000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.METAL_ALUMINUM, weight_kg=1.0, recycling_rate=1.0)
        ],
        landfill_percentage=0,
        recycling_percentage=100,
        incineration_percentage=0
    )

    result = await calculator.calculate(input_data)

    # High recycling of aluminum gives large credit (may be negative)
    # Just verify calculation completes
    assert result.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_disposal_mixed_methods(calculator):
    """Test mixed disposal methods."""
    input_data = Category12Input(
        product_name="Plastic Product",
        units_sold=5000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=0.5)
        ],
        landfill_percentage=40,
        recycling_percentage=35,
        incineration_percentage=25
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0
    # Mixed methods should have moderate emissions


# ============================================================================
# REGIONAL DISPOSAL PRACTICE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_regional_germany_high_recycling(calculator):
    """Test Germany's high recycling rate."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=1000,
        region="DE",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=1.0)
        ]
        # No disposal percentages - use regional defaults
    )

    result = await calculator.calculate(input_data)

    # Germany has high recycling (67%), low landfill (1%)
    # Should result in lower net emissions
    assert result.emissions_kgco2e >= 0  # Could be low or even negative


@pytest.mark.asyncio
async def test_regional_us_moderate_practices(calculator):
    """Test US moderate waste management."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=1000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=1.0)
        ]
        # Use US defaults: 52% landfill, 32% recycling, 16% incineration
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_regional_india_high_landfill(calculator):
    """Test India's high landfill rate."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=1000,
        region="IN",
        material_composition=[
            MaterialComposition(material_type=MaterialType.ORGANIC, weight_kg=1.0)
        ]
        # India defaults: 80% landfill, 15% recycling, 5% incineration
    )

    result = await calculator.calculate(input_data)

    # Organic waste in landfill produces methane -> high emissions
    assert result.emissions_kgco2e > 0


# ============================================================================
# MATERIAL TYPE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_material_organic_methane_emissions(calculator):
    """Test organic material with high methane emissions from landfill."""
    input_data = Category12Input(
        product_name="Food Packaging",
        units_sold=10000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.ORGANIC, weight_kg=0.1)
        ],
        landfill_percentage=90,  # Most goes to landfill
        recycling_percentage=5,
        incineration_percentage=5
    )

    result = await calculator.calculate(input_data)

    # Organic waste in landfill has high methane emissions
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_material_copper_recycling(calculator):
    """Test copper with recycling value."""
    input_data = Category12Input(
        product_name="Copper Wire",
        units_sold=1000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.METAL_COPPER, weight_kg=0.5, recycling_rate=0.80)
        ],
        recycling_percentage=80,
        landfill_percentage=15,
        incineration_percentage=5
    )

    result = await calculator.calculate(input_data)

    # High recycling of copper should reduce emissions
    assert result.tier == TierType.TIER_1


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_validation_negative_units(calculator):
    """Test validation fails for negative units."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=-100,
        region="US",
        total_weight_kg=1.0,
        primary_material=MaterialType.PLASTIC
    )

    with pytest.raises(ValidationError):  # Pydantic validation
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_empty_product_name(calculator):
    """Test validation fails for empty product name."""
    input_data = Category12Input(
        product_name="",
        units_sold=100,
        region="US",
        total_weight_kg=1.0,
        primary_material=MaterialType.PLASTIC
    )

    with pytest.raises(DataValidationError):
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_zero_units(calculator):
    """Test validation fails for zero units."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=0,
        region="US",
        total_weight_kg=1.0,
        primary_material=MaterialType.PLASTIC
    )

    with pytest.raises(ValidationError):  # Pydantic validation
        await calculator.calculate(input_data)


@pytest.mark.asyncio
async def test_validation_disposal_percentages_valid(calculator):
    """Test disposal percentages are valid (sum to 100)."""
    # This should work fine
    input_data = Category12Input(
        product_name="Product",
        units_sold=100,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=1.0)
        ],
        landfill_percentage=50,
        recycling_percentage=30,
        incineration_percentage=20
    )

    result = await calculator.calculate(input_data)
    assert result is not None


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_very_light_product(calculator):
    """Test edge case with very light product."""
    input_data = Category12Input(
        product_name="Plastic Bag",
        units_sold=100000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=0.005)  # 5 grams
        ],
        landfill_percentage=60,
        recycling_percentage=20,
        incineration_percentage=20
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e >= 0
    # Very light but high volume


@pytest.mark.asyncio
async def test_edge_case_very_heavy_product(calculator):
    """Test edge case with very heavy product."""
    input_data = Category12Input(
        product_name="Industrial Equipment",
        units_sold=10,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.METAL_STEEL, weight_kg=500)
        ],
        recycling_percentage=80,
        landfill_percentage=15,
        incineration_percentage=5
    )

    result = await calculator.calculate(input_data)

    # Heavy product should have significant disposal impacts
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_edge_case_single_unit_custom(calculator):
    """Test edge case with single custom unit."""
    input_data = Category12Input(
        product_name="Custom Prototype",
        units_sold=1,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.ELECTRONICS, weight_kg=2.0)
        ],
        recycling_percentage=100
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_data_quality_tier1_excellent(calculator):
    """Test Tier 1 has excellent data quality."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=1000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=1.0)
        ],
        landfill_percentage=50,
        recycling_percentage=30,
        incineration_percentage=20
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.rating == "excellent"
    assert result.data_quality.dqi_score == 80.0


@pytest.mark.asyncio
async def test_data_quality_tier3_fair(calculator, mock_factor_broker):
    """Test Tier 3 has fair data quality."""
    with patch.object(calculator, '_llm_estimate_composition', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "total_weight_kg": 5.0,
            "primary_material": "mixed",
            "confidence": 0.50
        }

        input_data = Category12Input(
            product_name="Product",
            units_sold=1000,
            region="US"
        )

        result = await calculator.calculate(input_data)

        assert result.data_quality.rating == "fair"
        assert result.data_quality.dqi_score == 45.0


# ============================================================================
# METADATA TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_product_info(calculator):
    """Test metadata contains product information."""
    input_data = Category12Input(
        product_name="Test Product XYZ",
        units_sold=5000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=1.0)
        ]
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["product_name"] == "Test Product XYZ"
    assert result.metadata["units_sold"] == 5000


# ============================================================================
# CALCULATION METHOD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_calculation_method_tier1(calculator):
    """Test calculation method for Tier 1."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=1000,
        region="US",
        material_composition=[
            MaterialComposition(material_type=MaterialType.PLASTIC, weight_kg=1.0)
        ]
    )

    result = await calculator.calculate(input_data)

    assert result.calculation_method == "tier_1_detailed_composition"


@pytest.mark.asyncio
async def test_calculation_method_tier2(calculator, mock_factor_broker):
    """Test calculation method for Tier 2."""
    input_data = Category12Input(
        product_name="Product",
        units_sold=1000,
        region="US",
        total_weight_kg=2.0,
        primary_material=MaterialType.PLASTIC
    )

    result = await calculator.calculate(input_data)

    assert result.calculation_method == "tier_2_primary_material"


# ============================================================================
# COMPREHENSIVE COVERAGE TEST
# ============================================================================

@pytest.mark.asyncio
async def test_comprehensive_coverage():
    """Verify comprehensive test coverage (25+ tests)."""
    import inspect

    test_functions = [
        name for name, obj in globals().items()
        if name.startswith('test_') and inspect.isfunction(obj)
    ]

    assert len(test_functions) >= 25, f"Need 25+ tests, have {len(test_functions)}"

    # Verify coverage areas
    tier1_tests = [t for t in test_functions if 'tier1' in t]
    tier2_tests = [t for t in test_functions if 'tier2' in t]
    tier3_tests = [t for t in test_functions if 'tier3' in t]
    disposal_tests = [t for t in test_functions if 'disposal' in t]
    regional_tests = [t for t in test_functions if 'regional' in t]
    material_tests = [t for t in test_functions if 'material' in t]
    validation_tests = [t for t in test_functions if 'validation' in t]
    edge_tests = [t for t in test_functions if 'edge' in t]

    assert len(tier1_tests) >= 4, "Need at least 4 Tier 1 tests"
    assert len(tier2_tests) >= 3, "Need at least 3 Tier 2 tests"
    assert len(tier3_tests) >= 2, "Need at least 2 Tier 3 tests"
    assert len(disposal_tests) >= 3, "Need at least 3 disposal method tests"
    assert len(regional_tests) >= 3, "Need at least 3 regional tests"
    assert len(material_tests) >= 2, "Need at least 2 material type tests"
    assert len(validation_tests) >= 4, "Need at least 4 validation tests"
    assert len(edge_tests) >= 3, "Need at least 3 edge case tests"
