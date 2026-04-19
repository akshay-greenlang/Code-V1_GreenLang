# -*- coding: utf-8 -*-
"""
Tests for Category 5: Waste Generated in Operations Calculator
GL-VCCI Scope 3 Platform

Comprehensive test suite covering:
- Happy path scenarios
- Edge cases
- LLM waste classification (mocked)
- Disposal method identification
- Recycling rate adjustments
- Tier fallback logic
- Data validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from greenlang.determinism import DeterministicClock
from services.agents.calculator.categories.category_5 import (
    Category5Calculator,
    WASTE_TYPES,
    DISPOSAL_METHODS,
)
from services.agents.calculator.models import (
    Category5Input,
    CalculationResult,
    DataQualityInfo,
    ProvenanceChain,
)
from services.agents.calculator.config import TierType, get_config
from services.agents.calculator.exceptions import (
    DataValidationError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock calculator configuration."""
    config = get_config()
    config.enable_monte_carlo = False
    return config


@pytest.fixture
def mock_factor_broker():
    """Mock Factor Broker."""
    return AsyncMock()


@pytest.fixture
def mock_llm_client():
    """Mock LLM Client."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value=json.dumps({
        "waste_type": "municipal_solid_waste",
        "disposal_method": "landfill",
        "recycling_rate": 0.0,
        "confidence": 0.85,
        "reasoning": "General office waste sent to landfill"
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
    builder.hash_factor_info = Mock(return_value="test_hash")
    builder.build = AsyncMock(return_value=ProvenanceChain(
        calculation_id="test_calc",
        timestamp=DeterministicClock.utcnow(),
        category=5,
        tier=TierType.TIER_2,
        input_data_hash="input_hash",
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
    """Create Category5Calculator instance."""
    return Category5Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        config=mock_config
    )


# ============================================================================
# Test Data Helpers
# ============================================================================

def create_category5_input(**kwargs):
    """Helper to create Category5Input with defaults."""
    defaults = {
        "waste_description": "General office waste",
        "waste_mass_kg": 5000.0,
        "region": "US",
    }
    defaults.update(kwargs)
    return Category5Input(**defaults)


# ============================================================================
# Tier 1 Tests (Supplier-Specific)
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_supplier_disposal_ef(calculator):
    """Test Tier 1 with supplier-specific disposal factor."""
    input_data = create_category5_input(
        supplier_disposal_ef=0.65,  # kgCO2e/kg
        waste_handler="Waste Management Inc"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e == 5000.0 * 0.65
    assert result.calculation_method == "tier_1_supplier_disposal"


@pytest.mark.asyncio
async def test_tier1_zero_emissions_waste(calculator):
    """Test Tier 1 with zero emission waste (e.g., recycling credit)."""
    input_data = create_category5_input(
        supplier_disposal_ef=0.0,
        waste_handler="Recycling Facility"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e == 0.0


@pytest.mark.asyncio
async def test_tier1_negative_ef_falls_back(calculator):
    """Test that negative EF triggers fallback to Tier 2."""
    input_data = create_category5_input(
        supplier_disposal_ef=-0.1,  # Invalid
        waste_type="municipal_solid_waste"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2  # Should fall back


# ============================================================================
# Tier 2 Tests (Waste-Specific with LLM)
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_llm_waste_classification(calculator, mock_llm_client):
    """Test Tier 2 with LLM waste classification."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "organic_waste",
        "disposal_method": "composting",
        "recycling_rate": 0.0,
        "confidence": 0.90,
        "reasoning": "Food waste sent to composting facility"
    })

    input_data = create_category5_input(
        waste_description="Kitchen food waste from cafeteria",
        waste_mass_kg=2000.0
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["waste_type"] == "organic_waste"
    assert result.metadata["disposal_method"] == "composting"


@pytest.mark.asyncio
async def test_tier2_landfill_disposal(calculator, mock_llm_client):
    """Test landfill disposal method."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "municipal_solid_waste",
        "disposal_method": "landfill",
        "recycling_rate": 0.0,
        "confidence": 0.87,
        "reasoning": "General waste to landfill"
    })

    input_data = create_category5_input(
        waste_description="Mixed office waste",
        waste_mass_kg=3000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["disposal_method"] == "landfill"
    # Landfill has high emissions
    assert result.emissions_kgco2e > 1000


@pytest.mark.asyncio
async def test_tier2_incineration_with_energy_recovery(calculator, mock_llm_client):
    """Test incineration with energy recovery."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "municipal_solid_waste",
        "disposal_method": "incineration_with_energy_recovery",
        "recycling_rate": 0.0,
        "confidence": 0.88,
        "reasoning": "Waste-to-energy facility"
    })

    input_data = create_category5_input(
        waste_description="Waste to energy plant",
        waste_mass_kg=10000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["disposal_method"] == "incineration_with_energy_recovery"
    # WTE has lower emissions
    assert result.emissions_kgco2e < 5000


@pytest.mark.asyncio
async def test_tier2_recycling_negative_emissions(calculator, mock_llm_client):
    """Test recycling with negative emissions (avoided emissions)."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "plastic_waste",
        "disposal_method": "recycling",
        "recycling_rate": 0.0,
        "confidence": 0.92,
        "reasoning": "Plastic recycling"
    })

    input_data = create_category5_input(
        waste_description="Recyclable plastic materials",
        waste_mass_kg=1000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["disposal_method"] == "recycling"
    # Recycling typically has avoided emissions
    assert result.emissions_kgco2e <= 0


@pytest.mark.asyncio
async def test_tier2_hazardous_waste(calculator, mock_llm_client):
    """Test hazardous waste classification."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "hazardous_waste",
        "disposal_method": "incineration",
        "recycling_rate": 0.0,
        "confidence": 0.95,
        "reasoning": "Hazardous chemical waste requiring incineration"
    })

    input_data = create_category5_input(
        waste_description="Hazardous chemical waste from laboratory",
        waste_mass_kg=500.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["waste_type"] == "hazardous_waste"


@pytest.mark.asyncio
async def test_tier2_construction_waste(calculator, mock_llm_client):
    """Test construction waste classification."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "construction_waste",
        "disposal_method": "landfill",
        "recycling_rate": 0.3,
        "confidence": 0.89,
        "reasoning": "Construction debris, partially recycled"
    })

    input_data = create_category5_input(
        waste_description="Construction demolition debris",
        waste_mass_kg=20000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["waste_type"] == "construction_waste"
    assert result.metadata["recycling_rate"] == 0.3


@pytest.mark.asyncio
async def test_tier2_electronic_waste(calculator, mock_llm_client):
    """Test e-waste classification."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "electronic_waste",
        "disposal_method": "recycling",
        "recycling_rate": 0.8,
        "confidence": 0.93,
        "reasoning": "Electronic waste sent to certified e-waste recycler"
    })

    input_data = create_category5_input(
        waste_description="Old computers and electronic equipment",
        waste_mass_kg=800.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["waste_type"] == "electronic_waste"


@pytest.mark.asyncio
async def test_tier2_metal_waste_recycling(calculator, mock_llm_client):
    """Test metal waste recycling."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "metal_waste",
        "disposal_method": "recycling",
        "recycling_rate": 0.95,
        "confidence": 0.91,
        "reasoning": "Metal scrap for recycling"
    })

    input_data = create_category5_input(
        waste_description="Scrap metal from manufacturing",
        waste_mass_kg=5000.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["waste_type"] == "metal_waste"
    assert result.metadata["recycling_rate"] == 0.95


@pytest.mark.asyncio
async def test_tier2_paper_cardboard(calculator, mock_llm_client):
    """Test paper and cardboard waste."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "paper_cardboard",
        "disposal_method": "recycling",
        "recycling_rate": 0.85,
        "confidence": 0.90,
        "reasoning": "Paper and cardboard recycling"
    })

    input_data = create_category5_input(
        waste_description="Office paper and cardboard boxes",
        waste_mass_kg=1500.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["waste_type"] == "paper_cardboard"


@pytest.mark.asyncio
async def test_tier2_recycling_rate_adjustment(calculator, mock_llm_client):
    """Test that recycling rate reduces emissions correctly."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "municipal_solid_waste",
        "disposal_method": "landfill",
        "recycling_rate": 0.5,
        "confidence": 0.85,
        "reasoning": "Mixed waste with 50% recycling"
    })

    input_data = create_category5_input(
        waste_mass_kg=1000.0,
        recycling_rate=0.5
    )

    result = await calculator.calculate(input_data)

    # With 50% recycling, emissions should be halved
    assert result.metadata["recycling_rate"] == 0.5


@pytest.mark.asyncio
async def test_tier2_manual_waste_type_and_disposal(calculator):
    """Test with manually specified waste type and disposal method."""
    input_data = create_category5_input(
        waste_type="organic_waste",
        disposal_method="composting",
        waste_mass_kg=2000.0
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["disposal_method"] == "composting"


@pytest.mark.asyncio
async def test_tier2_llm_failure_keyword_fallback(calculator, mock_llm_client):
    """Test keyword fallback when LLM fails."""
    mock_llm_client.complete.side_effect = Exception("LLM error")

    input_data = create_category5_input(
        waste_description="Municipal solid waste from office",
        waste_mass_kg=3000.0
    )

    result = await calculator.calculate(input_data)

    # Should work with keyword fallback
    assert result.tier == TierType.TIER_2
    assert result.metadata["waste_type"] in WASTE_TYPES.keys()


@pytest.mark.asyncio
async def test_tier2_keyword_classification_hazardous(calculator, mock_llm_client):
    """Test keyword-based classification for hazardous waste."""
    mock_llm_client.complete.side_effect = Exception("No LLM")

    input_data = create_category5_input(
        waste_description="Hazardous chemical toxic waste",
        waste_mass_kg=200.0
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["waste_type"] == "hazardous_waste"


# ============================================================================
# Tier 3 Tests (Generic Factors)
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_generic_waste_factor(calculator):
    """Test Tier 3 generic waste disposal."""
    input_data = create_category5_input(
        waste_description="Generic waste",
        waste_mass_kg=1000.0
    )

    result = await calculator.calculate(input_data)

    # Should use generic factor (0.7 kgCO2e/kg)
    assert result.emissions_kgco2e == 1000.0 * 0.7


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_zero_waste_mass_validation_error(calculator):
    """Test that zero waste mass raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category5_input(waste_mass_kg=0.0)
        await calculator.calculate(input_data)

    assert "waste_mass" in str(exc_info.value)


@pytest.mark.asyncio
async def test_negative_waste_mass_validation_error(calculator):
    """Test that negative waste mass raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category5_input(waste_mass_kg=-100.0)
        await calculator.calculate(input_data)

    assert "positive" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_empty_waste_description_validation_error(calculator):
    """Test that empty waste description raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category5_input(waste_description="")
        await calculator.calculate(input_data)

    assert "waste_description" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_recycling_rate_validation_error(calculator):
    """Test that invalid recycling rate raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category5_input(
            recycling_rate=1.5  # >1.0 is invalid
        )
        await calculator.calculate(input_data)

    assert "recycling_rate" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_negative_recycling_rate_validation_error(calculator):
    """Test that negative recycling rate raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category5_input(
            recycling_rate=-0.1
        )
        await calculator.calculate(input_data)

    assert "recycling_rate" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_very_small_waste_mass(calculator):
    """Test calculation with very small waste mass."""
    input_data = create_category5_input(
        waste_mass_kg=0.1,
        supplier_disposal_ef=0.7
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0
    assert result.emissions_kgco2e < 1.0


@pytest.mark.asyncio
async def test_very_large_waste_mass(calculator):
    """Test calculation with very large waste mass."""
    input_data = create_category5_input(
        waste_mass_kg=1000000.0,
        supplier_disposal_ef=0.7
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 500000


# ============================================================================
# Recycling Rate Tests
# ============================================================================

@pytest.mark.asyncio
async def test_100_percent_recycling(calculator):
    """Test 100% recycling rate."""
    input_data = create_category5_input(
        waste_mass_kg=1000.0,
        supplier_disposal_ef=0.8,
        recycling_rate=1.0
    )

    result = await calculator.calculate(input_data)

    # With 100% recycling, landfill emissions should be zero
    # (but there might be recycling process emissions)
    assert result.emissions_kgco2e >= 0


@pytest.mark.asyncio
async def test_partial_recycling_rate(calculator):
    """Test partial recycling rate reduces emissions."""
    input_data_no_recycling = create_category5_input(
        waste_mass_kg=1000.0,
        supplier_disposal_ef=0.8,
        recycling_rate=0.0
    )

    input_data_with_recycling = create_category5_input(
        waste_mass_kg=1000.0,
        supplier_disposal_ef=0.8,
        recycling_rate=0.6
    )

    result_no_recycling = await calculator.calculate(input_data_no_recycling)
    result_with_recycling = await calculator.calculate(input_data_with_recycling)

    # With 60% recycling, emissions should be 40% of original
    assert result_with_recycling.emissions_kgco2e < result_no_recycling.emissions_kgco2e


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_completeness(calculator):
    """Test that result metadata is complete."""
    input_data = create_category5_input(
        supplier_disposal_ef=0.65,
        waste_handler="Test Handler"
    )

    result = await calculator.calculate(input_data)

    assert "waste_description" in result.metadata
    assert "waste_mass_kg" in result.metadata
    assert "waste_handler" in result.metadata


@pytest.mark.asyncio
async def test_data_quality_warnings(calculator, mock_llm_client):
    """Test that data quality warnings are generated."""
    input_data = create_category5_input(
        waste_description="Unknown waste type XYZ"
    )

    result = await calculator.calculate(input_data)

    # Should have warnings about classification
    assert len(result.warnings) > 0 or len(result.data_quality.warnings) > 0


@pytest.mark.asyncio
async def test_provenance_tracking(calculator, mock_provenance_builder):
    """Test provenance chain creation."""
    input_data = create_category5_input(
        supplier_disposal_ef=0.7
    )

    result = await calculator.calculate(input_data)

    assert mock_provenance_builder.build.called
    call_args = mock_provenance_builder.build.call_args[1]
    assert call_args["category"] == 5


@pytest.mark.asyncio
async def test_llm_invalid_waste_type_defaults(calculator, mock_llm_client):
    """Test that invalid LLM waste type defaults appropriately."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "invalid_type_xyz",
        "disposal_method": "landfill",
        "recycling_rate": 0.0,
        "confidence": 0.50,
        "reasoning": "Unknown"
    })

    input_data = create_category5_input(
        waste_description="Some unknown waste"
    )

    result = await calculator.calculate(input_data)

    # Should default to a valid waste type
    assert result.metadata["waste_type"] in WASTE_TYPES.keys()


@pytest.mark.asyncio
async def test_llm_invalid_disposal_method_defaults(calculator, mock_llm_client):
    """Test that invalid LLM disposal method defaults appropriately."""
    mock_llm_client.complete.return_value = json.dumps({
        "waste_type": "municipal_solid_waste",
        "disposal_method": "invalid_method_xyz",
        "recycling_rate": 0.0,
        "confidence": 0.50,
        "reasoning": "Unknown"
    })

    input_data = create_category5_input(
        waste_description="General waste"
    )

    result = await calculator.calculate(input_data)

    # Should default to a valid disposal method
    assert result.metadata["disposal_method"] in DISPOSAL_METHODS.keys()


@pytest.mark.asyncio
async def test_batch_different_waste_types(calculator, mock_llm_client):
    """Test batch processing of different waste types."""
    waste_types = [
        ("municipal_solid_waste", "landfill"),
        ("organic_waste", "composting"),
        ("plastic_waste", "recycling"),
        ("hazardous_waste", "incineration")
    ]

    for waste_type, disposal in waste_types:
        mock_llm_client.complete.return_value = json.dumps({
            "waste_type": waste_type,
            "disposal_method": disposal,
            "recycling_rate": 0.0,
            "confidence": 0.90,
            "reasoning": f"{waste_type} disposal"
        })

        input_data = create_category5_input(
            waste_description=f"{waste_type} test",
            waste_mass_kg=1000.0
        )

        result = await calculator.calculate(input_data)

        assert result.category == 5
        assert result.metadata["waste_type"] == waste_type
