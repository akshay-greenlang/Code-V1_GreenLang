"""
Tests for Category 2: Capital Goods Calculator
GL-VCCI Scope 3 Platform

Comprehensive test suite covering:
- Happy path scenarios
- Edge cases
- LLM integration (mocked)
- Tier fallback logic
- Data validation
- Integration with Factor Broker

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from services.agents.calculator.categories.category_2 import (
    Category2Calculator,
    ASSET_CATEGORIES,
    ASSET_EMISSION_FACTORS,
)
from services.agents.calculator.models import (
    Category2Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
)
from services.agents.calculator.config import TierType, get_config
from services.agents.calculator.exceptions import (
    DataValidationError,
    TierFallbackError,
    CalculationError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock calculator configuration."""
    config = get_config()
    config.enable_monte_carlo = False  # Disable for faster tests
    return config


@pytest.fixture
def mock_factor_broker():
    """Mock Factor Broker."""
    broker = AsyncMock()
    return broker


@pytest.fixture
def mock_llm_client():
    """Mock LLM Client."""
    client = AsyncMock()
    # Default LLM response
    client.complete = AsyncMock(return_value=json.dumps({
        "category": "machinery",
        "useful_life_years": 10,
        "confidence": 0.85,
        "reasoning": "Industrial manufacturing equipment"
    }))
    return client


@pytest.fixture
def mock_uncertainty_engine():
    """Mock Uncertainty Engine."""
    engine = AsyncMock()
    engine.propagate = AsyncMock(return_value=None)
    return engine


@pytest.fixture
def mock_provenance_builder():
    """Mock Provenance Builder."""
    builder = AsyncMock()
    builder.hash_factor_info = Mock(return_value="test_hash_123")
    builder.build = AsyncMock(return_value=ProvenanceChain(
        calculation_id="test_calc_123",
        timestamp=datetime.utcnow(),
        category=2,
        tier=TierType.TIER_2,
        input_data_hash="input_hash_123",
        calculation={},
        data_quality=DataQualityInfo(
            dqi_score=70.0,
            tier=TierType.TIER_2,
            rating="good",
            warnings=[]
        ),
        provenance_chain=[]
    ))
    return builder


@pytest.fixture
def calculator(mock_factor_broker, mock_llm_client, mock_uncertainty_engine, mock_provenance_builder, mock_config):
    """Create Category2Calculator instance."""
    return Category2Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        config=mock_config
    )


# ============================================================================
# Test Data
# ============================================================================

def create_category2_input(**kwargs) -> Category2Input:
    """Helper to create Category2Input with defaults."""
    defaults = {
        "asset_description": "Industrial CNC machinery for metal fabrication",
        "capex_amount": 500000.0,
        "region": "US",
    }
    defaults.update(kwargs)
    return Category2Input(**defaults)


# ============================================================================
# Tier 1 Tests (Supplier-Specific PCF)
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_supplier_pcf_success(calculator):
    """Test successful Tier 1 calculation with supplier PCF."""
    input_data = create_category2_input(
        supplier_pcf=250000.0,  # kgCO2e total
        supplier_pcf_uncertainty=0.10,
        supplier_name="ACME Manufacturing"
    )

    result = await calculator.calculate(input_data)

    assert result.category == 2
    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e == 250000.0
    assert result.emissions_tco2e == 250.0
    assert result.data_quality.rating == "excellent"
    assert result.calculation_method == "tier_1_supplier_pcf"


@pytest.mark.asyncio
async def test_tier1_with_zero_pcf_falls_back(calculator, mock_llm_client):
    """Test that zero supplier PCF triggers fallback to Tier 2."""
    input_data = create_category2_input(
        supplier_pcf=0.0,  # Zero PCF should trigger fallback
        asset_category="machinery"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2  # Should fall back
    assert mock_llm_client.complete.called  # LLM should be invoked


@pytest.mark.asyncio
async def test_tier1_with_negative_pcf_falls_back(calculator):
    """Test that negative supplier PCF triggers fallback."""
    input_data = create_category2_input(
        supplier_pcf=-100.0,  # Invalid
        asset_category="vehicles"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2  # Should fall back


# ============================================================================
# Tier 2 Tests (Asset-Specific with Amortization)
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_with_llm_classification(calculator, mock_llm_client):
    """Test Tier 2 with LLM asset classification."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "machinery",
        "useful_life_years": 12,
        "confidence": 0.90,
        "reasoning": "Heavy industrial equipment based on description and value"
    })

    input_data = create_category2_input(
        capex_amount=500000.0,
        industry="manufacturing"
    )

    result = await calculator.calculate(input_data)

    assert result.category == 2
    assert result.tier == TierType.TIER_2
    assert result.metadata["asset_category"] == "machinery"
    assert result.metadata["useful_life_years"] == 12
    assert "LLM" in result.metadata["llm_classification"]

    # Verify amortization: (500000 × 0.48) / 12 = 20000 kgCO2e/year
    expected_emissions = (500000.0 * ASSET_EMISSION_FACTORS["machinery"]) / 12
    assert abs(result.emissions_kgco2e - expected_emissions) < 1.0


@pytest.mark.asyncio
async def test_tier2_buildings_classification(calculator, mock_llm_client):
    """Test classification of building assets."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "buildings",
        "useful_life_years": 30,
        "confidence": 0.95,
        "reasoning": "Commercial building construction"
    })

    input_data = create_category2_input(
        asset_description="New office building construction",
        capex_amount=5000000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["asset_category"] == "buildings"
    assert result.metadata["useful_life_years"] == 30

    # Buildings have 30 year life, EF = 0.42
    expected = (5000000.0 * 0.42) / 30
    assert abs(result.emissions_kgco2e - expected) < 1.0


@pytest.mark.asyncio
async def test_tier2_it_equipment_classification(calculator, mock_llm_client):
    """Test classification of IT equipment."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "it_equipment",
        "useful_life_years": 4,
        "confidence": 0.88,
        "reasoning": "Server hardware and data center equipment"
    })

    input_data = create_category2_input(
        asset_description="Data center server infrastructure",
        capex_amount=200000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["asset_category"] == "it_equipment"
    assert result.metadata["useful_life_years"] == 4


@pytest.mark.asyncio
async def test_tier2_vehicles_classification(calculator, mock_llm_client):
    """Test classification of vehicle fleet."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "vehicles",
        "useful_life_years": 8,
        "confidence": 0.92,
        "reasoning": "Commercial delivery trucks"
    })

    input_data = create_category2_input(
        asset_description="Fleet of 10 delivery trucks",
        capex_amount=800000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["asset_category"] == "vehicles"
    assert result.metadata["useful_life_years"] == 8


@pytest.mark.asyncio
async def test_tier2_with_manual_asset_category(calculator):
    """Test Tier 2 with manually specified asset category."""
    input_data = create_category2_input(
        asset_category="machinery",
        useful_life_years=15,
        capex_amount=300000.0
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["useful_life_years"] == 15
    # Should use manual values, not LLM


@pytest.mark.asyncio
async def test_tier2_llm_failure_fallback_to_keywords(calculator, mock_llm_client):
    """Test fallback to keyword classification when LLM fails."""
    mock_llm_client.complete.side_effect = Exception("LLM API error")

    input_data = create_category2_input(
        asset_description="Industrial manufacturing machine for production",
        capex_amount=400000.0
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    # Should still work with keyword fallback
    assert "machinery" in result.metadata["asset_category"]


@pytest.mark.asyncio
async def test_tier2_keyword_classification_machinery(calculator, mock_llm_client):
    """Test keyword-based classification for machinery."""
    mock_llm_client.complete.side_effect = Exception("No LLM")

    input_data = create_category2_input(
        asset_description="Production equipment and machinery assembly line",
        capex_amount=600000.0
    )

    result = await calculator.calculate(input_data)

    # Keywords should match machinery
    assert result.metadata["asset_category"] == "machinery"


@pytest.mark.asyncio
async def test_tier2_keyword_classification_building(calculator, mock_llm_client):
    """Test keyword-based classification for buildings."""
    mock_llm_client.complete.side_effect = Exception("No LLM")

    input_data = create_category2_input(
        asset_description="Office building construction and facility upgrade",
        capex_amount=3000000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["asset_category"] == "buildings"


@pytest.mark.asyncio
async def test_tier2_useful_life_clamping(calculator, mock_llm_client):
    """Test that useful life is clamped to reasonable ranges."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "machinery",
        "useful_life_years": 100,  # Unreasonably high
        "confidence": 0.80,
        "reasoning": "Test"
    })

    input_data = create_category2_input(capex_amount=500000.0)

    result = await calculator.calculate(input_data)

    # Should be clamped to max range for machinery (20 years)
    assert result.metadata["useful_life_years"] <= 20


# ============================================================================
# Tier 3 Tests (Spend-Based)
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_spend_based(calculator, mock_factor_broker):
    """Test Tier 3 spend-based calculation."""
    from services.factor_broker.models import FactorResponse, FactorMetadata, ProvenanceInfo, SourceType

    # Mock factor broker response
    mock_factor_broker.resolve.return_value = FactorResponse(
        factor_id="eio_capital_goods_US",
        value=0.45,  # kgCO2e/USD
        unit="kgCO2e/USD",
        uncertainty=0.50,
        metadata=FactorMetadata(
            source=SourceType.EEIO,
            source_version="2024",
            gwp_standard="AR6",
            reference_year=2024,
            geographic_scope="US",
        ),
        provenance=ProvenanceInfo()
    )

    input_data = create_category2_input(
        capex_amount=100000.0,
        economic_sector="capital_goods_average"
    )

    # Remove all higher tier data to force Tier 3
    input_data.supplier_pcf = None
    input_data.asset_category = None

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_3
    assert result.emissions_kgco2e == 100000.0 * 0.45
    assert "spend" in result.calculation_method.lower()


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_zero_capex_validation_error(calculator):
    """Test that zero capex raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category2_input(capex_amount=0.0)
        await calculator.calculate(input_data)

    assert "capex_amount" in str(exc_info.value)


@pytest.mark.asyncio
async def test_negative_capex_validation_error(calculator):
    """Test that negative capex raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category2_input(capex_amount=-50000.0)
        await calculator.calculate(input_data)

    assert "positive" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_empty_asset_description_validation_error(calculator):
    """Test that empty asset description raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category2_input(asset_description="")
        await calculator.calculate(input_data)

    assert "asset_description" in str(exc_info.value)


@pytest.mark.asyncio
async def test_negative_useful_life_validation_error(calculator):
    """Test that negative useful life raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category2_input(
            useful_life_years=-5.0,
            asset_category="machinery"
        )
        await calculator.calculate(input_data)

    assert "useful_life" in str(exc_info.value)


@pytest.mark.asyncio
async def test_very_small_capex(calculator, mock_llm_client):
    """Test calculation with very small capex amount."""
    input_data = create_category2_input(
        capex_amount=100.0,  # $100
        asset_category="other_equipment"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0
    assert result.emissions_kgco2e < 100  # Should be small


@pytest.mark.asyncio
async def test_very_large_capex(calculator, mock_llm_client):
    """Test calculation with very large capex amount."""
    input_data = create_category2_input(
        capex_amount=100000000.0,  # $100M
        asset_category="buildings"
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 1000000  # Should be large


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_tier_fallback_sequence(calculator, mock_llm_client):
    """Test complete tier fallback from 1 -> 2 -> 3."""
    input_data = create_category2_input(
        supplier_pcf=None,  # No Tier 1
        capex_amount=250000.0
    )

    # First attempt Tier 2 with LLM
    result = await calculator.calculate(input_data)

    # Should succeed at Tier 2
    assert result.tier in [TierType.TIER_2, TierType.TIER_3]
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_data_quality_warnings(calculator, mock_llm_client):
    """Test that appropriate data quality warnings are generated."""
    input_data = create_category2_input(
        capex_amount=300000.0,
        industry="tech"
    )

    result = await calculator.calculate(input_data)

    # Should have warnings about LLM classification
    assert len(result.warnings) > 0 or len(result.data_quality.warnings) > 0


@pytest.mark.asyncio
async def test_metadata_completeness(calculator, mock_llm_client):
    """Test that result metadata is complete and accurate."""
    input_data = create_category2_input(
        asset_description="Test machinery",
        capex_amount=400000.0,
        supplier_name="Test Supplier",
        industry="manufacturing"
    )

    result = await calculator.calculate(input_data)

    # Check metadata completeness
    assert "asset_description" in result.metadata
    assert "asset_category" in result.metadata
    assert "capex_amount" in result.metadata
    assert "useful_life_years" in result.metadata


@pytest.mark.asyncio
async def test_provenance_chain_creation(calculator, mock_provenance_builder):
    """Test that provenance chain is properly created."""
    input_data = create_category2_input(
        capex_amount=500000.0,
        asset_category="machinery"
    )

    result = await calculator.calculate(input_data)

    # Verify provenance builder was called
    assert mock_provenance_builder.build.called
    call_args = mock_provenance_builder.build.call_args[1]
    assert call_args["category"] == 2


@pytest.mark.asyncio
async def test_uncertainty_propagation_disabled(calculator):
    """Test calculation with uncertainty propagation disabled."""
    calculator.config.enable_monte_carlo = False

    input_data = create_category2_input(
        supplier_pcf=100000.0,
        supplier_pcf_uncertainty=0.15
    )

    result = await calculator.calculate(input_data)

    # Uncertainty should be None when disabled
    assert result.uncertainty is None


# ============================================================================
# LLM Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_llm_classification_confidence_threshold(calculator, mock_llm_client):
    """Test handling of low-confidence LLM classifications."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "other_equipment",
        "useful_life_years": 10,
        "confidence": 0.45,  # Low confidence
        "reasoning": "Uncertain classification"
    })

    input_data = create_category2_input(capex_amount=200000.0)

    result = await calculator.calculate(input_data)

    # Should still work, but may have warnings
    assert result.tier == TierType.TIER_2
    assert result.metadata["asset_category"] == "other_equipment"


@pytest.mark.asyncio
async def test_llm_invalid_json_fallback(calculator, mock_llm_client):
    """Test fallback when LLM returns invalid JSON."""
    mock_llm_client.complete.return_value = "This is not JSON"

    input_data = create_category2_input(
        asset_description="Machine with production keyword",
        capex_amount=300000.0
    )

    result = await calculator.calculate(input_data)

    # Should fall back to keyword classification
    assert result.tier == TierType.TIER_2
    assert "machinery" in result.metadata["asset_category"]


@pytest.mark.asyncio
async def test_llm_invalid_category_fallback(calculator, mock_llm_client):
    """Test fallback when LLM returns invalid category."""
    mock_llm_client.complete.return_value = json.dumps({
        "category": "invalid_category_xyz",
        "useful_life_years": 10,
        "confidence": 0.90,
        "reasoning": "Test"
    })

    input_data = create_category2_input(
        asset_description="Office furniture purchase",
        capex_amount=50000.0
    )

    result = await calculator.calculate(input_data)

    # Should default to other_equipment
    assert result.metadata["asset_category"] in ASSET_CATEGORIES.keys()


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_batch_calculation_performance(calculator):
    """Test performance with multiple calculations."""
    inputs = [
        create_category2_input(
            asset_description=f"Asset {i}",
            capex_amount=100000.0 * (i + 1),
            asset_category="machinery"
        )
        for i in range(10)
    ]

    results = []
    for input_data in inputs:
        result = await calculator.calculate(input_data)
        results.append(result)

    assert len(results) == 10
    assert all(r.emissions_kgco2e > 0 for r in results)


@pytest.mark.asyncio
async def test_concurrent_calculations(calculator):
    """Test concurrent calculation handling."""
    import asyncio

    inputs = [
        create_category2_input(
            asset_description=f"Asset {i}",
            capex_amount=50000.0,
            asset_category="it_equipment"
        )
        for i in range(5)
    ]

    # Run calculations concurrently
    results = await asyncio.gather(*[calculator.calculate(inp) for inp in inputs])

    assert len(results) == 5
    assert all(r.category == 2 for r in results)
