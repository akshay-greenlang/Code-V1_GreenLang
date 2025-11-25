# -*- coding: utf-8 -*-
"""
Category 15: Investments Calculator Tests (PCAF Standard)
GL-VCCI Scope 3 Platform

Comprehensive test suite for Category 15 calculator with:
- PCAF Score 1-5 calculations
- Multiple attribution methods
- Asset class variations
- LLM sector classification
- Portfolio aggregation
- Edge cases and error handling

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from greenlang.determinism import DeterministicClock
from services.agents.calculator.categories.category_15 import (
    Category15Calculator,
    Category15Input,
    AssetClass,
    AttributionMethod,
    PCAFDataQuality,
    IndustrySector,
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
        timestamp=DeterministicClock.utcnow(),
        category=15,
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
    return Category15Calculator(
        factor_broker=mock_factor_broker,
        llm_client=mock_llm_client,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder
    )


# ============================================================================
# PCAF SCORE 1 TESTS (Verified Reported Emissions)
# ============================================================================

@pytest.mark.asyncio
async def test_pcaf_score1_verified_emissions(calculator):
    """Test PCAF Score 1 with verified reported emissions."""
    input_data = Category15Input(
        investment_id="INV001",
        portfolio_company_name="TechCorp Inc",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=10_000_000.0,
        company_value_evic=100_000_000.0,
        company_emissions_scope1_tco2e=50000.0,
        company_emissions_scope2_tco2e=30000.0,
        emissions_verified=True,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.category == 15
    assert result.metadata["pcaf_score"] == 1
    assert result.data_quality.rating == "excellent"
    assert result.metadata["attribution_factor"] == 0.1
    # Financed emissions = 80000 * 0.1 = 8000 tCO2e
    assert result.emissions_tco2e == pytest.approx(8000.0, rel=0.01)


@pytest.mark.asyncio
async def test_pcaf_score1_with_scope3(calculator):
    """Test PCAF Score 1 including Scope 3 emissions."""
    input_data = Category15Input(
        investment_id="INV002",
        portfolio_company_name="EnergyPower LLC",
        asset_class=AssetClass.CORPORATE_BONDS,
        outstanding_amount=5_000_000.0,
        company_value_evic=50_000_000.0,
        company_emissions_scope1_tco2e=100000.0,
        company_emissions_scope2_tco2e=20000.0,
        company_emissions_scope3_tco2e=50000.0,
        emissions_verified=True,
        region="GB",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 1
    # Total = 170000, attribution = 0.1, financed = 17000 tCO2e
    assert result.emissions_tco2e == pytest.approx(17000.0, rel=0.01)


# ============================================================================
# PCAF SCORE 2 TESTS (Unverified Reported Emissions)
# ============================================================================

@pytest.mark.asyncio
async def test_pcaf_score2_unverified_emissions(calculator):
    """Test PCAF Score 2 with unverified reported emissions."""
    input_data = Category15Input(
        investment_id="INV003",
        portfolio_company_name="RetailChain Co",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=8_000_000.0,
        company_total_assets=80_000_000.0,
        company_emissions_scope1_tco2e=15000.0,
        company_emissions_scope2_tco2e=10000.0,
        emissions_verified=False,
        region="DE",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 2
    assert result.data_quality.rating == "good"
    # Attribution = 8M / 80M = 0.1
    # Financed = 25000 * 0.1 = 2500 tCO2e
    assert result.emissions_tco2e == pytest.approx(2500.0, rel=0.01)


# ============================================================================
# PCAF SCORE 3 TESTS (Physical Activity, Primary)
# ============================================================================

@pytest.mark.asyncio
async def test_pcaf_score3_physical_activity(calculator):
    """Test PCAF Score 3 with physical activity data."""
    input_data = Category15Input(
        investment_id="INV004",
        portfolio_company_name="BuildingManagement Inc",
        asset_class=AssetClass.COMMERCIAL_REAL_ESTATE,
        outstanding_amount=20_000_000.0,
        company_total_assets=200_000_000.0,
        physical_activity_data={
            "floor_area_sqm": 50000,
            "energy_kwh": 5_000_000
        },
        industry_sector=IndustrySector.REAL_ESTATE,
        company_revenue=50_000_000.0,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 3
    assert result.data_quality.rating == "good"


# ============================================================================
# PCAF SCORE 4 TESTS (Physical Activity, Estimated)
# ============================================================================

@pytest.mark.asyncio
async def test_pcaf_score4_real_estate(calculator):
    """Test PCAF Score 4 for commercial real estate."""
    input_data = Category15Input(
        investment_id="INV005",
        portfolio_company_name="Property Holdings",
        asset_class=AssetClass.COMMERCIAL_REAL_ESTATE,
        outstanding_amount=15_000_000.0,
        company_total_assets=150_000_000.0,
        company_revenue=30_000_000.0,
        industry_sector=IndustrySector.REAL_ESTATE,
        region="FR",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 4
    assert result.data_quality.rating == "fair"


@pytest.mark.asyncio
async def test_pcaf_score4_mortgages(calculator):
    """Test PCAF Score 4 for mortgages."""
    input_data = Category15Input(
        investment_id="INV006",
        portfolio_company_name="Residential Property",
        asset_class=AssetClass.MORTGAGES,
        outstanding_amount=500_000.0,
        company_total_assets=500_000.0,
        region="CA",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 4


# ============================================================================
# PCAF SCORE 5 TESTS (Economic Activity)
# ============================================================================

@pytest.mark.asyncio
async def test_pcaf_score5_economic_activity(calculator):
    """Test PCAF Score 5 with economic activity estimation."""
    input_data = Category15Input(
        investment_id="INV007",
        portfolio_company_name="Manufacturing Corp",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=12_000_000.0,
        company_revenue=120_000_000.0,
        industry_sector=IndustrySector.MANUFACTURING,
        region="JP",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 5
    assert result.data_quality.rating == "fair"
    assert "economic activity" in result.warnings[0].lower()


@pytest.mark.asyncio
async def test_pcaf_score5_high_carbon_sector(calculator):
    """Test PCAF Score 5 for high-carbon sector (Energy)."""
    input_data = Category15Input(
        investment_id="INV008",
        portfolio_company_name="Oil & Gas Co",
        asset_class=AssetClass.CORPORATE_BONDS,
        outstanding_amount=25_000_000.0,
        company_revenue=500_000_000.0,
        industry_sector=IndustrySector.ENERGY,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["pcaf_score"] == 5
    assert result.metadata["sector"] == "energy"
    # Energy sector should have high emissions
    assert result.emissions_tco2e > 1000.0


# ============================================================================
# ATTRIBUTION METHOD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_attribution_equity_share(calculator):
    """Test equity share attribution method."""
    input_data = Category15Input(
        investment_id="INV009",
        portfolio_company_name="Public Company",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=20_000_000.0,
        company_value_evic=200_000_000.0,
        company_emissions_scope1_tco2e=100000.0,
        emissions_verified=True,
        attribution_method=AttributionMethod.EQUITY_SHARE,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Attribution = 20M / 200M = 0.1
    assert result.metadata["attribution_factor"] == 0.1


@pytest.mark.asyncio
async def test_attribution_asset_based(calculator):
    """Test asset-based attribution method."""
    input_data = Category15Input(
        investment_id="INV010",
        portfolio_company_name="Asset Heavy Corp",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=30_000_000.0,
        company_total_assets=300_000_000.0,
        company_emissions_scope1_tco2e=50000.0,
        attribution_method=AttributionMethod.ASSET_BASED,
        region="DE",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Attribution = 30M / 300M = 0.1
    assert result.metadata["attribution_factor"] == 0.1


@pytest.mark.asyncio
async def test_attribution_project_specific(calculator):
    """Test project-specific attribution (100%)."""
    input_data = Category15Input(
        investment_id="INV011",
        portfolio_company_name="Solar Farm Project",
        asset_class=AssetClass.PROJECT_FINANCE,
        outstanding_amount=50_000_000.0,
        company_emissions_scope1_tco2e=5000.0,
        attribution_method=AttributionMethod.PROJECT_SPECIFIC,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Project finance = 100% attribution
    assert result.metadata["attribution_factor"] == 1.0
    assert result.emissions_tco2e == pytest.approx(5000.0, rel=0.01)


# ============================================================================
# ASSET CLASS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_asset_class_corporate_bonds(calculator):
    """Test corporate bonds asset class."""
    input_data = Category15Input(
        investment_id="INV012",
        portfolio_company_name="Bond Issuer Inc",
        asset_class=AssetClass.CORPORATE_BONDS,
        outstanding_amount=15_000_000.0,
        company_value_evic=150_000_000.0,
        company_emissions_scope1_tco2e=30000.0,
        region="GB",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["asset_class"] == "corporate_bonds"


@pytest.mark.asyncio
async def test_asset_class_sovereign_debt(calculator):
    """Test sovereign debt asset class."""
    input_data = Category15Input(
        investment_id="INV013",
        portfolio_company_name="Country Government",
        asset_class=AssetClass.SOVEREIGN_DEBT,
        outstanding_amount=100_000_000.0,
        company_value_evic=1_000_000_000.0,
        region="FR",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.metadata["asset_class"] == "sovereign_debt"


# ============================================================================
# LLM SECTOR CLASSIFICATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_llm_classify_financial_services(calculator):
    """Test LLM classification of financial services."""
    description = "Leading investment bank providing financial advisory"

    sector = await calculator._llm_classify_sector(description, "FinBank Corp")

    assert sector == IndustrySector.FINANCIAL_SERVICES


@pytest.mark.asyncio
async def test_llm_classify_technology(calculator):
    """Test LLM classification of technology sector."""
    description = "Software company developing cloud computing solutions"

    sector = await calculator._llm_classify_sector(description, "CloudTech Inc")

    assert sector == IndustrySector.TECHNOLOGY


@pytest.mark.asyncio
async def test_llm_classify_energy(calculator):
    """Test LLM classification of energy sector."""
    description = "Oil and gas exploration and production company"

    sector = await calculator._llm_classify_sector(description, "PetroEnergy")

    assert sector == IndustrySector.ENERGY


@pytest.mark.asyncio
async def test_llm_classify_utilities(calculator):
    """Test LLM classification of utilities sector."""
    description = "Electric power utility serving residential customers"

    sector = await calculator._llm_classify_sector(description, "PowerGrid Co")

    assert sector == IndustrySector.UTILITIES


@pytest.mark.asyncio
async def test_llm_classify_manufacturing(calculator):
    """Test LLM classification of manufacturing sector."""
    description = "Automotive parts manufacturing facility"

    sector = await calculator._llm_classify_sector(description, "AutoParts Inc")

    assert sector == IndustrySector.MANUFACTURING


# ============================================================================
# PORTFOLIO AGGREGATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_portfolio_calculation(calculator):
    """Test portfolio-level aggregation."""
    investments = [
        Category15Input(
            investment_id=f"INV{i}",
            portfolio_company_name=f"Company {i}",
            asset_class=AssetClass.LISTED_EQUITY,
            outstanding_amount=10_000_000.0,
            company_value_evic=100_000_000.0,
            company_emissions_scope1_tco2e=50000.0,
            emissions_verified=True,
            region="US",
            reporting_year=2024
        )
        for i in range(1, 6)
    ]

    portfolio_result = await calculator.calculate_portfolio(investments)

    assert portfolio_result["total_investments"] == 5
    assert portfolio_result["successful_calculations"] == 5
    assert portfolio_result["total_financed_emissions_tco2e"] > 0
    assert "by_asset_class" in portfolio_result
    assert "by_sector" in portfolio_result


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

@pytest.mark.asyncio
async def test_missing_investment_id(calculator):
    """Test validation error for missing investment ID."""
    input_data = Category15Input(
        investment_id="",
        portfolio_company_name="Test Company",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=1_000_000.0,
        region="US",
        reporting_year=2024
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "Investment ID is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_missing_company_name(calculator):
    """Test validation error for missing company name."""
    input_data = Category15Input(
        investment_id="INV001",
        portfolio_company_name="",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=1_000_000.0,
        region="US",
        reporting_year=2024
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "Portfolio company name is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zero_outstanding_amount(calculator):
    """Test validation error for zero outstanding amount."""
    input_data = Category15Input(
        investment_id="INV001",
        portfolio_company_name="Test Company",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=0.0,
        region="US",
        reporting_year=2024
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "greater than 0" in str(exc_info.value)


@pytest.mark.asyncio
async def test_minimal_data_fallback(calculator):
    """Test fallback estimation with minimal data."""
    input_data = Category15Input(
        investment_id="INV014",
        portfolio_company_name="Unknown Company",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=5_000_000.0,
        region="CN",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Should default to PCAF Score 5
    assert result.metadata["pcaf_score"] == 5
    assert result.emissions_tco2e > 0


@pytest.mark.asyncio
async def test_very_large_investment(calculator):
    """Test calculation for very large investment."""
    input_data = Category15Input(
        investment_id="INV015",
        portfolio_company_name="Mega Corp",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=1_000_000_000.0,
        company_value_evic=10_000_000_000.0,
        company_emissions_scope1_tco2e=5_000_000.0,
        emissions_verified=True,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Attribution = 1B / 10B = 0.1
    # Financed = 5M * 0.1 = 500k tCO2e
    assert result.emissions_tco2e == pytest.approx(500_000.0, rel=0.01)


# ============================================================================
# SECTOR-SPECIFIC TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_sector_utilities_high_intensity(calculator):
    """Test utilities sector has high carbon intensity."""
    input_data = Category15Input(
        investment_id="INV016",
        portfolio_company_name="PowerPlant Co",
        asset_class=AssetClass.CORPORATE_BONDS,
        outstanding_amount=20_000_000.0,
        company_revenue=200_000_000.0,
        industry_sector=IndustrySector.UTILITIES,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Utilities should have very high emissions
    assert result.emissions_tco2e > 5000.0


@pytest.mark.asyncio
async def test_sector_financial_low_intensity(calculator):
    """Test financial services sector has low carbon intensity."""
    input_data = Category15Input(
        investment_id="INV017",
        portfolio_company_name="Investment Bank",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=30_000_000.0,
        company_revenue=300_000_000.0,
        industry_sector=IndustrySector.FINANCIAL_SERVICES,
        region="GB",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    # Financial services should have lower emissions
    assert result.metadata["sector"] == "financial_services"


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_score1_excellent_quality(calculator):
    """Test PCAF Score 1 has excellent data quality."""
    input_data = Category15Input(
        investment_id="INV018",
        portfolio_company_name="Verified Corp",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=10_000_000.0,
        company_value_evic=100_000_000.0,
        company_emissions_scope1_tco2e=50000.0,
        emissions_verified=True,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.dqi_score >= 90.0
    assert result.data_quality.rating == "excellent"


@pytest.mark.asyncio
async def test_score5_lower_quality(calculator):
    """Test PCAF Score 5 has lower data quality."""
    input_data = Category15Input(
        investment_id="INV019",
        portfolio_company_name="Estimated Corp",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=5_000_000.0,
        company_revenue=50_000_000.0,
        region="FR",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.dqi_score < 50.0
    assert len(result.warnings) > 0


# ============================================================================
# UNCERTAINTY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_score1_low_uncertainty(calculator):
    """Test PCAF Score 1 has lowest uncertainty."""
    input_data = Category15Input(
        investment_id="INV020",
        portfolio_company_name="Certain Corp",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=10_000_000.0,
        company_value_evic=100_000_000.0,
        company_emissions_scope1_tco2e=50000.0,
        emissions_verified=True,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    if result.uncertainty:
        assert result.uncertainty.coefficient_of_variation <= 0.15


@pytest.mark.asyncio
async def test_score5_high_uncertainty(calculator):
    """Test PCAF Score 5 has highest uncertainty."""
    input_data = Category15Input(
        investment_id="INV021",
        portfolio_company_name="Uncertain Corp",
        asset_class=AssetClass.BUSINESS_LOANS,
        outstanding_amount=5_000_000.0,
        company_revenue=50_000_000.0,
        region="JP",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    if result.uncertainty:
        assert result.uncertainty.coefficient_of_variation >= 0.50


# ============================================================================
# METADATA TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metadata_completeness(calculator):
    """Test result metadata includes all PCAF-required fields."""
    input_data = Category15Input(
        investment_id="INV022",
        portfolio_company_name="Complete Data Corp",
        asset_class=AssetClass.LISTED_EQUITY,
        outstanding_amount=15_000_000.0,
        company_value_evic=150_000_000.0,
        company_emissions_scope1_tco2e=75000.0,
        emissions_verified=True,
        industry_sector=IndustrySector.TECHNOLOGY,
        region="US",
        reporting_year=2024
    )

    result = await calculator.calculate(input_data)

    assert "investment_id" in result.metadata
    assert "portfolio_company" in result.metadata
    assert "asset_class" in result.metadata
    assert "pcaf_score" in result.metadata
    assert "attribution_factor" in result.metadata
    assert "company_emissions_tco2e" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
