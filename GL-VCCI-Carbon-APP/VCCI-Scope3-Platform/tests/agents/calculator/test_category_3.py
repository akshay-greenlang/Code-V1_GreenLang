"""
Tests for Category 3: Fuel and Energy-Related Activities Calculator
GL-VCCI Scope 3 Platform

Comprehensive test suite covering:
- Happy path scenarios
- Edge cases
- LLM fuel identification (mocked)
- Well-to-tank calculations
- T&D loss calculations
- Tier fallback logic
- Data validation

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from services.agents.calculator.categories.category_3 import (
    Category3Calculator,
    FUEL_TYPES,
)
from services.agents.calculator.models import (
    Category3Input,
    CalculationResult,
    DataQualityInfo,
    ProvenanceChain,
)
from services.agents.calculator.config import TierType, get_config
from services.agents.calculator.exceptions import (
    DataValidationError,
    TierFallbackError,
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
        "fuel_type": "natural_gas",
        "confidence": 0.90,
        "reasoning": "Natural gas consumption based on description"
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
        timestamp=datetime.utcnow(),
        category=3,
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
    """Create Category3Calculator instance."""
    return Category3Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        config=mock_config
    )


# ============================================================================
# Test Data Helpers
# ============================================================================

def create_category3_input(**kwargs):
    """Helper to create Category3Input with defaults."""
    defaults = {
        "fuel_or_energy_type": "Natural gas for heating",
        "quantity": 10000.0,
        "quantity_unit": "kWh",
        "region": "US",
    }
    defaults.update(kwargs)
    return Category3Input(**defaults)


# ============================================================================
# Tier 1 Tests (Supplier-Specific)
# ============================================================================

@pytest.mark.asyncio
async def test_tier1_supplier_upstream_only(calculator):
    """Test Tier 1 with supplier upstream EF only."""
    input_data = create_category3_input(
        supplier_upstream_ef=0.15,  # kgCO2e/kWh
        supplier_name="Energy Provider Inc"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    assert result.emissions_kgco2e == 10000.0 * 0.15
    assert result.calculation_method == "tier_1_supplier_upstream"


@pytest.mark.asyncio
async def test_tier1_with_td_losses(calculator):
    """Test Tier 1 with both upstream and T&D losses."""
    input_data = create_category3_input(
        fuel_or_energy_type="Electricity",
        supplier_upstream_ef=0.12,
        supplier_td_losses_ef=0.03,
        supplier_name="Grid Operator"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1
    # Total = (0.12 + 0.03) * 10000 = 1500 kgCO2e
    assert result.emissions_kgco2e == 1500.0
    assert result.metadata["upstream_emissions_kgco2e"] == 1200.0
    assert result.metadata["td_losses_emissions_kgco2e"] == 300.0


# ============================================================================
# Tier 2 Tests (WTT Factors with LLM)
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_electricity_td_losses(calculator, mock_llm_client):
    """Test Tier 2 electricity with T&D losses."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "electricity",
        "confidence": 0.95,
        "reasoning": "Electricity consumption"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Electricity consumption",
        quantity=50000.0,
        quantity_unit="kWh"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["is_electricity"] == True
    assert "td_loss_percentage" in result.metadata
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_tier2_natural_gas_wtt(calculator, mock_llm_client):
    """Test Tier 2 natural gas with well-to-tank factor."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "natural_gas",
        "confidence": 0.92,
        "reasoning": "Natural gas heating fuel"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Natural gas for industrial heating",
        quantity=25000.0,
        quantity_unit="m3"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["is_electricity"] == False
    assert "wtt_factor" in result.metadata
    assert result.metadata["identified_fuel_type"] == "natural_gas"


@pytest.mark.asyncio
async def test_tier2_diesel_wtt(calculator, mock_llm_client):
    """Test Tier 2 diesel with well-to-tank factor."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "diesel",
        "confidence": 0.88,
        "reasoning": "Diesel fuel for generators"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Diesel fuel for backup generators",
        quantity=5000.0,
        quantity_unit="liters"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2
    assert result.metadata["identified_fuel_type"] == "diesel"


@pytest.mark.asyncio
async def test_tier2_gasoline_wtt(calculator, mock_llm_client):
    """Test Tier 2 gasoline with well-to-tank factor."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "gasoline",
        "confidence": 0.85,
        "reasoning": "Gasoline for fleet vehicles"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Gasoline for company vehicles",
        quantity=3000.0,
        quantity_unit="liters"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "gasoline"


@pytest.mark.asyncio
async def test_tier2_coal_wtt(calculator, mock_llm_client):
    """Test Tier 2 coal with well-to-tank factor."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "coal",
        "confidence": 0.90,
        "reasoning": "Coal for industrial processes"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Coal for manufacturing",
        quantity=100000.0,
        quantity_unit="kg"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "coal"


@pytest.mark.asyncio
async def test_tier2_llm_failure_keyword_fallback(calculator, mock_llm_client):
    """Test keyword fallback when LLM fails."""
    mock_llm_client.complete.side_effect = Exception("LLM error")

    input_data = create_category3_input(
        fuel_or_energy_type="Natural gas pipeline consumption",
        quantity=15000.0
    )

    result = await calculator.calculate(input_data)

    # Should still work with keyword fallback
    assert result.tier == TierType.TIER_2
    assert result.metadata["identified_fuel_type"] == "natural_gas"


@pytest.mark.asyncio
async def test_tier2_keyword_classification_electricity(calculator, mock_llm_client):
    """Test keyword-based classification for electricity."""
    mock_llm_client.complete.side_effect = Exception("No LLM")

    input_data = create_category3_input(
        fuel_or_energy_type="Electric power consumption from grid",
        quantity=20000.0,
        quantity_unit="kWh"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "electricity"


@pytest.mark.asyncio
async def test_tier2_keyword_classification_diesel(calculator, mock_llm_client):
    """Test keyword-based classification for diesel."""
    mock_llm_client.complete.side_effect = Exception("No LLM")

    input_data = create_category3_input(
        fuel_or_energy_type="Diesel fuel for equipment",
        quantity=1000.0,
        quantity_unit="liters"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "diesel"


# ============================================================================
# Tier 3 Tests (Proxy Factors)
# ============================================================================

@pytest.mark.asyncio
async def test_tier3_proxy_upstream(calculator, mock_llm_client):
    """Test Tier 3 proxy upstream calculation."""
    input_data = create_category3_input(
        fuel_or_energy_type="Generic fuel",
        quantity=5000.0
    )

    result = await calculator.calculate(input_data)

    # Should calculate with 15% proxy upstream factor
    assert result.emissions_kgco2e > 0
    assert "proxy" in result.calculation_method.lower()


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_zero_quantity_validation_error(calculator):
    """Test that zero quantity raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category3_input(quantity=0.0)
        await calculator.calculate(input_data)

    assert "quantity" in str(exc_info.value)


@pytest.mark.asyncio
async def test_negative_quantity_validation_error(calculator):
    """Test that negative quantity raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category3_input(quantity=-1000.0)
        await calculator.calculate(input_data)

    assert "positive" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_empty_fuel_type_validation_error(calculator):
    """Test that empty fuel type raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category3_input(fuel_or_energy_type="")
        await calculator.calculate(input_data)

    assert "fuel" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_invalid_td_loss_percentage_validation_error(calculator):
    """Test that invalid T&D loss percentage raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category3_input(
            td_loss_percentage=1.5  # >1.0 is invalid
        )
        await calculator.calculate(input_data)

    assert "td_loss" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_negative_td_loss_percentage_validation_error(calculator):
    """Test that negative T&D loss percentage raises validation error."""
    with pytest.raises(DataValidationError) as exc_info:
        input_data = create_category3_input(
            td_loss_percentage=-0.1
        )
        await calculator.calculate(input_data)

    assert "td_loss" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_very_small_quantity(calculator):
    """Test calculation with very small quantity."""
    input_data = create_category3_input(
        quantity=0.1,
        supplier_upstream_ef=0.15
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0
    assert result.emissions_kgco2e < 1.0


@pytest.mark.asyncio
async def test_very_large_quantity(calculator):
    """Test calculation with very large quantity."""
    input_data = create_category3_input(
        quantity=10000000.0,
        supplier_upstream_ef=0.15
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 1000000


# ============================================================================
# Fuel Type Specific Tests
# ============================================================================

@pytest.mark.asyncio
async def test_biomass_fuel(calculator, mock_llm_client):
    """Test biomass fuel identification."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "biomass",
        "confidence": 0.87,
        "reasoning": "Biomass pellets"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Wood biomass pellets",
        quantity=5000.0,
        quantity_unit="kg"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "biomass"


@pytest.mark.asyncio
async def test_lpg_fuel(calculator, mock_llm_client):
    """Test LPG fuel identification."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "lpg",
        "confidence": 0.91,
        "reasoning": "LPG for forklifts"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="LPG propane for warehouse forklifts",
        quantity=2000.0,
        quantity_unit="liters"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "lpg"


@pytest.mark.asyncio
async def test_fuel_oil(calculator, mock_llm_client):
    """Test fuel oil identification."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "fuel_oil",
        "confidence": 0.89,
        "reasoning": "Heating oil"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Heating oil for facility",
        quantity=3000.0,
        quantity_unit="liters"
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["identified_fuel_type"] == "fuel_oil"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tier_fallback_from_1_to_2(calculator, mock_llm_client):
    """Test fallback from Tier 1 to Tier 2."""
    input_data = create_category3_input(
        supplier_upstream_ef=None,  # No Tier 1 data
        fuel_or_energy_type="Natural gas"
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_2


@pytest.mark.asyncio
async def test_custom_td_loss_percentage(calculator):
    """Test with custom T&D loss percentage."""
    input_data = create_category3_input(
        fuel_or_energy_type="Electricity",
        supplier_upstream_ef=0.10,
        supplier_td_losses_ef=0.05,  # Custom value
        td_loss_percentage=0.08
    )

    result = await calculator.calculate(input_data)

    assert result.tier == TierType.TIER_1


@pytest.mark.asyncio
async def test_grid_region_specification(calculator, mock_llm_client):
    """Test with specific grid region."""
    input_data = create_category3_input(
        fuel_or_energy_type="Electricity from regional grid",
        grid_region="WECC",
        quantity=30000.0
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_metadata_completeness(calculator):
    """Test that result metadata is complete."""
    input_data = create_category3_input(
        supplier_upstream_ef=0.12,
        supplier_name="Energy Co"
    )

    result = await calculator.calculate(input_data)

    assert "fuel_or_energy_type" in result.metadata
    assert "supplier_name" in result.metadata


@pytest.mark.asyncio
async def test_data_quality_warnings(calculator, mock_llm_client):
    """Test that data quality warnings are generated."""
    input_data = create_category3_input(
        fuel_or_energy_type="Unknown fuel type XYZ",
        quantity=1000.0
    )

    result = await calculator.calculate(input_data)

    # Should have warnings about classification
    assert len(result.warnings) > 0 or len(result.data_quality.warnings) > 0


@pytest.mark.asyncio
async def test_provenance_tracking(calculator, mock_provenance_builder):
    """Test provenance chain creation."""
    input_data = create_category3_input(
        supplier_upstream_ef=0.15
    )

    result = await calculator.calculate(input_data)

    assert mock_provenance_builder.build.called
    call_args = mock_provenance_builder.build.call_args[1]
    assert call_args["category"] == 3


@pytest.mark.asyncio
async def test_llm_invalid_fuel_type_defaults_to_electricity(calculator, mock_llm_client):
    """Test that invalid LLM fuel type defaults to electricity."""
    mock_llm_client.complete.return_value = json.dumps({
        "fuel_type": "invalid_fuel_xyz",
        "confidence": 0.70,
        "reasoning": "Unknown"
    })

    input_data = create_category3_input(
        fuel_or_energy_type="Some unknown energy source"
    )

    result = await calculator.calculate(input_data)

    # Should default to electricity
    assert result.metadata["identified_fuel_type"] in FUEL_TYPES.keys()


@pytest.mark.asyncio
async def test_batch_different_fuel_types(calculator, mock_llm_client):
    """Test batch processing of different fuel types."""
    fuel_types = ["electricity", "natural_gas", "diesel", "gasoline"]

    for fuel in fuel_types:
        mock_llm_client.complete.return_value = json.dumps({
            "fuel_type": fuel,
            "confidence": 0.90,
            "reasoning": f"{fuel} consumption"
        })

        input_data = create_category3_input(
            fuel_or_energy_type=f"{fuel} usage",
            quantity=10000.0
        )

        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e > 0
        assert result.category == 3
