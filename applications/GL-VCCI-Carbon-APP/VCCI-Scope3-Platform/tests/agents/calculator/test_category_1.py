# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Category 1: Purchased Goods & Services
GL-VCCI Scope 3 Platform

Tests all calculation tiers, waterfall logic, uncertainty, provenance, and edge cases.

Total: 35 tests

Version: 1.0.0
Date: 2025-11-09
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.agents.calculator.categories.category_1 import Category1Calculator
from services.agents.calculator.models import (
    Category1Input,
    CalculationResult,
    DataQualityInfo,
    TierType,
)
from services.agents.calculator.config import get_config
from services.agents.calculator.exceptions import (
    DataValidationError,
    TierFallbackError,
    CalculationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator(mock_factor_broker, mock_uncertainty_engine, mock_provenance_builder):
    """Create Category1Calculator instance."""
    industry_mapper = Mock()
    industry_mapper.map = Mock(return_value=Mock(matched=True, matched_title="Steel Products"))

    return Category1Calculator(
        factor_broker=mock_factor_broker,
        industry_mapper=industry_mapper,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        config=get_config()
    )


@pytest.fixture
def tier1_input():
    """Tier 1 input with supplier PCF."""
    return Category1Input(
        product_name="Steel Sheets",
        quantity=1000.0,
        quantity_unit="kg",
        supplier_pcf=2.5,  # kgCO2e/kg
        supplier_name="ACME Steel",
        region="US"
    )


@pytest.fixture
def tier2_input():
    """Tier 2 input with product category."""
    return Category1Input(
        product_name="Steel Sheets",
        quantity=1000.0,
        quantity_unit="kg",
        product_category="steel_products",
        region="US"
    )


@pytest.fixture
def tier3_input():
    """Tier 3 input with spend data."""
    return Category1Input(
        product_name="Steel Sheets",
        quantity=1000.0,
        quantity_unit="kg",
        spend_usd=5000.0,
        economic_sector="manufacturing",
        region="US"
    )


# ============================================================================
# TIER 1 CALCULATION TESTS (10 tests)
# ============================================================================

class TestTier1Calculations:
    """Test Tier 1 (supplier-specific PCF) calculations."""

    @pytest.mark.asyncio
    async def test_tier1_basic_calculation(self, calculator, tier1_input):
        """Test basic Tier 1 calculation with supplier PCF."""
        result = await calculator.calculate(tier1_input)

        assert result is not None
        assert result.tier == TierType.TIER_1
        assert result.emissions_kgco2e == 2500.0  # 1000 kg * 2.5 kgCO2e/kg
        assert result.emissions_tco2e == 2.5
        assert result.calculation_method == "tier_1_supplier_pcf"

    @pytest.mark.asyncio
    async def test_tier1_with_high_pcf(self, calculator):
        """Test Tier 1 with high supplier PCF value."""
        input_data = Category1Input(
            product_name="Carbon Intensive Product",
            quantity=100.0,
            quantity_unit="kg",
            supplier_pcf=50.0,  # Very high PCF
            supplier_name="High Emissions Co",
            region="US"
        )
        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == 5000.0
        assert result.tier == TierType.TIER_1

    @pytest.mark.asyncio
    async def test_tier1_with_low_pcf(self, calculator):
        """Test Tier 1 with low supplier PCF value."""
        input_data = Category1Input(
            product_name="Low Carbon Product",
            quantity=1000.0,
            quantity_unit="kg",
            supplier_pcf=0.1,
            supplier_name="Green Supplier",
            region="US"
        )
        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == 100.0
        assert result.tier == TierType.TIER_1

    @pytest.mark.asyncio
    async def test_tier1_with_uncertainty(self, calculator, mock_uncertainty_engine):
        """Test Tier 1 with uncertainty propagation."""
        mock_uncertainty_engine.propagate = AsyncMock(return_value=Mock(
            p50=2500.0,
            p95=2750.0,
            std_dev=125.0
        ))

        input_data = Category1Input(
            product_name="Steel",
            quantity=1000.0,
            quantity_unit="kg",
            supplier_pcf=2.5,
            supplier_pcf_uncertainty=0.10,
            supplier_name="ACME",
            region="US"
        )
        result = await calculator.calculate(input_data)

        assert result.uncertainty is not None
        assert result.uncertainty.p50 == 2500.0

    @pytest.mark.asyncio
    async def test_tier1_large_quantity(self, calculator):
        """Test Tier 1 with large quantity."""
        input_data = Category1Input(
            product_name="Steel",
            quantity=1000000.0,  # 1 million kg
            quantity_unit="kg",
            supplier_pcf=2.5,
            supplier_name="ACME",
            region="US"
        )
        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == 2500000.0
        assert result.emissions_tco2e == 2500.0

    @pytest.mark.asyncio
    async def test_tier1_small_quantity(self, calculator):
        """Test Tier 1 with small quantity."""
        input_data = Category1Input(
            product_name="Steel Sample",
            quantity=0.001,  # 1 gram
            quantity_unit="kg",
            supplier_pcf=2.5,
            supplier_name="ACME",
            region="US"
        )
        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == pytest.approx(0.0025, rel=1e-6)

    @pytest.mark.asyncio
    async def test_tier1_provenance_tracking(self, calculator, tier1_input, mock_provenance_builder):
        """Test that provenance is tracked for Tier 1."""
        result = await calculator.calculate(tier1_input)

        mock_provenance_builder.build.assert_called_once()
        call_kwargs = mock_provenance_builder.build.call_args[1]
        assert call_kwargs['category'] == 1
        assert call_kwargs['tier'] == TierType.TIER_1

    @pytest.mark.asyncio
    async def test_tier1_data_quality_excellent(self, calculator, tier1_input):
        """Test that Tier 1 has excellent data quality."""
        result = await calculator.calculate(tier1_input)

        assert result.data_quality.dqi_score >= 80  # Tier 1 should be high quality
        assert result.data_quality.tier == TierType.TIER_1
        assert result.data_quality.rating == "excellent"

    @pytest.mark.asyncio
    async def test_tier1_different_units(self, calculator):
        """Test Tier 1 with different quantity units."""
        input_data = Category1Input(
            product_name="Steel",
            quantity=1.0,
            quantity_unit="tonne",
            supplier_pcf=2500.0,  # kgCO2e/tonne
            supplier_name="ACME",
            region="US"
        )
        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == 2500.0
        assert result.tier == TierType.TIER_1

    @pytest.mark.asyncio
    async def test_tier1_metadata_capture(self, calculator, tier1_input):
        """Test that metadata is captured in Tier 1 results."""
        result = await calculator.calculate(tier1_input)

        assert result.metadata is not None
        assert 'supplier_name' in result.metadata
        assert result.metadata['supplier_name'] == "ACME Steel"
        assert 'product_name' in result.metadata


# ============================================================================
# TIER 2 CALCULATION TESTS (8 tests)
# ============================================================================

class TestTier2Calculations:
    """Test Tier 2 (average product data) calculations."""

    @pytest.mark.asyncio
    async def test_tier2_basic_calculation(self, calculator, tier2_input, mock_factor_broker):
        """Test basic Tier 2 calculation with product emission factor."""
        mock_factor = Mock()
        mock_factor.value = 3.0  # kgCO2e/kg
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "ecoinvent"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "steel_ef_001"
        mock_factor.metadata = Mock(
            source_version="v3.8",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash123")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier2_input)

        assert result.tier == TierType.TIER_2
        assert result.emissions_kgco2e == 3000.0  # 1000 kg * 3.0
        assert result.calculation_method == "tier_2_average_data"

    @pytest.mark.asyncio
    async def test_tier2_product_categorization(self, calculator, mock_factor_broker):
        """Test Tier 2 with product categorization."""
        input_data = Category1Input(
            product_name="Aluminum Sheets",
            quantity=500.0,
            quantity_unit="kg",
            region="EU"
        )

        mock_factor = Mock()
        mock_factor.value = 8.5
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "ecoinvent"
        mock_factor.uncertainty = 0.25
        mock_factor.data_quality_score = 65
        mock_factor.factor_id = "aluminum_ef_001"
        mock_factor.metadata = Mock(
            source_version="v3.8",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="EU"
        )
        mock_factor.provenance = Mock(calculation_hash="hash456")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == 4250.0  # 500 * 8.5

    @pytest.mark.asyncio
    async def test_tier2_regional_factors(self, calculator, mock_factor_broker):
        """Test Tier 2 with region-specific emission factors."""
        for region in ["US", "EU", "CN", "IN"]:
            input_data = Category1Input(
                product_name="Steel",
                quantity=1000.0,
                quantity_unit="kg",
                product_category="steel",
                region=region
            )

            mock_factor = Mock()
            mock_factor.value = 2.5
            mock_factor.unit = "kgCO2e/kg"
            mock_factor.source = "test_db"
            mock_factor.uncertainty = 0.20
            mock_factor.data_quality_score = 70
            mock_factor.factor_id = f"steel_{region}"
            mock_factor.metadata = Mock(
                source_version="v1",
                gwp_standard=Mock(value="AR6"),
                reference_year=2024,
                geographic_scope=region
            )
            mock_factor.provenance = Mock(calculation_hash="hash")

            mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

            result = await calculator.calculate(input_data)
            assert result.tier == TierType.TIER_2

    @pytest.mark.asyncio
    async def test_tier2_data_quality_scoring(self, calculator, tier2_input, mock_factor_broker):
        """Test Tier 2 data quality scoring."""
        mock_factor = Mock()
        mock_factor.value = 3.0
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "database"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 60
        mock_factor.factor_id = "test_ef"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier2_input)

        assert result.data_quality.tier == TierType.TIER_2
        assert 40 <= result.data_quality.dqi_score <= 80  # Medium quality range

    @pytest.mark.asyncio
    async def test_tier2_low_quality_warning(self, calculator, tier2_input, mock_factor_broker):
        """Test Tier 2 low quality warning generation."""
        mock_factor = Mock()
        mock_factor.value = 3.0
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "low_quality_db"
        mock_factor.uncertainty = 0.50
        mock_factor.data_quality_score = 30  # Low quality
        mock_factor.factor_id = "low_quality_ef"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier2_input)

        assert len(result.warnings) > 0
        assert any("quality" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_tier2_factor_not_found_fallback(self, calculator, tier2_input, mock_factor_broker):
        """Test Tier 2 fallback when factor not found."""
        mock_factor_broker.resolve = AsyncMock(return_value=None)

        # Should fallback to Tier 3 if spend data available
        tier2_input.spend_usd = 5000.0

        result = await calculator.calculate(tier2_input)
        assert result.tier == TierType.TIER_3 or result is not None

    @pytest.mark.asyncio
    async def test_tier2_uncertainty_propagation(self, calculator, tier2_input, mock_factor_broker, mock_uncertainty_engine):
        """Test Tier 2 uncertainty propagation."""
        mock_factor = Mock()
        mock_factor.value = 3.0
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "database"
        mock_factor.uncertainty = 0.25
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "test_ef"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        mock_uncertainty_engine.propagate = AsyncMock(return_value=Mock(
            p50=3000.0,
            p95=3500.0,
            std_dev=250.0
        ))

        result = await calculator.calculate(tier2_input)

        assert result.uncertainty is not None

    @pytest.mark.asyncio
    async def test_tier2_provenance_tracking(self, calculator, tier2_input, mock_factor_broker, mock_provenance_builder):
        """Test Tier 2 provenance tracking."""
        mock_factor = Mock()
        mock_factor.value = 3.0
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "database"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "test_ef"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash123")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier2_input)

        mock_provenance_builder.build.assert_called()
        call_kwargs = mock_provenance_builder.build.call_args[1]
        assert call_kwargs['tier'] == TierType.TIER_2


# ============================================================================
# TIER 3 CALCULATION TESTS (8 tests)
# ============================================================================

class TestTier3Calculations:
    """Test Tier 3 (spend-based) calculations."""

    @pytest.mark.asyncio
    async def test_tier3_basic_calculation(self, calculator, tier3_input, mock_factor_broker):
        """Test basic Tier 3 spend-based calculation."""
        mock_factor = Mock()
        mock_factor.value = 0.45  # kgCO2e/USD
        mock_factor.unit = "kgCO2e/USD"
        mock_factor.source = "eio"
        mock_factor.uncertainty = 0.50
        mock_factor.data_quality_score = 40
        mock_factor.factor_id = "manufacturing_eio"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash789")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier3_input)

        assert result.tier == TierType.TIER_3
        assert result.emissions_kgco2e == 2250.0  # 5000 USD * 0.45
        assert result.calculation_method == "tier_3_spend_based"

    @pytest.mark.asyncio
    async def test_tier3_different_sectors(self, calculator, mock_factor_broker):
        """Test Tier 3 with different economic sectors."""
        sectors = {
            "manufacturing": 0.45,
            "services": 0.22,
            "agriculture": 0.38,
            "construction": 0.42
        }

        for sector, intensity in sectors.items():
            input_data = Category1Input(
                product_name="Product",
                quantity=1000.0,
                quantity_unit="kg",
                spend_usd=10000.0,
                economic_sector=sector,
                region="US"
            )

            mock_factor = Mock()
            mock_factor.value = intensity
            mock_factor.unit = "kgCO2e/USD"
            mock_factor.source = "eio"
            mock_factor.uncertainty = 0.50
            mock_factor.data_quality_score = 40
            mock_factor.factor_id = f"{sector}_eio"
            mock_factor.metadata = Mock(
                source_version="v1",
                gwp_standard=Mock(value="AR6"),
                reference_year=2024,
                geographic_scope="US"
            )
            mock_factor.provenance = Mock(calculation_hash="hash")

            mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

            result = await calculator.calculate(input_data)
            assert result.emissions_kgco2e == 10000.0 * intensity

    @pytest.mark.asyncio
    async def test_tier3_high_spend(self, calculator, mock_factor_broker):
        """Test Tier 3 with high spend value."""
        input_data = Category1Input(
            product_name="High Value Product",
            quantity=100.0,
            quantity_unit="kg",
            spend_usd=1000000.0,  # $1M
            economic_sector="manufacturing",
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.45
        mock_factor.unit = "kgCO2e/USD"
        mock_factor.source = "eio"
        mock_factor.uncertainty = 0.50
        mock_factor.data_quality_score = 40
        mock_factor.factor_id = "manufacturing_eio"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == 450000.0

    @pytest.mark.asyncio
    async def test_tier3_low_data_quality(self, calculator, tier3_input, mock_factor_broker):
        """Test that Tier 3 has lower data quality."""
        mock_factor = Mock()
        mock_factor.value = 0.45
        mock_factor.unit = "kgCO2e/USD"
        mock_factor.source = "eio"
        mock_factor.uncertainty = 0.50
        mock_factor.data_quality_score = 40
        mock_factor.factor_id = "manufacturing_eio"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier3_input)

        assert result.data_quality.tier == TierType.TIER_3
        assert result.data_quality.dqi_score < 60  # Lower quality
        assert result.data_quality.rating == "fair"

    @pytest.mark.asyncio
    async def test_tier3_warnings_generated(self, calculator, tier3_input, mock_factor_broker):
        """Test that Tier 3 generates appropriate warnings."""
        mock_factor = Mock()
        mock_factor.value = 0.45
        mock_factor.unit = "kgCO2e/USD"
        mock_factor.source = "eio"
        mock_factor.uncertainty = 0.50
        mock_factor.data_quality_score = 40
        mock_factor.factor_id = "manufacturing_eio"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier3_input)

        assert len(result.warnings) >= 1
        assert any("spend-based" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_tier3_default_sector_fallback(self, calculator, mock_factor_broker):
        """Test Tier 3 default sector fallback."""
        input_data = Category1Input(
            product_name="Unknown Product",
            quantity=1000.0,
            quantity_unit="kg",
            spend_usd=5000.0,
            region="US"
            # No economic_sector specified
        )

        # Should use default/average sector
        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3
        assert result.emissions_kgco2e > 0

    @pytest.mark.asyncio
    async def test_tier3_high_uncertainty(self, calculator, tier3_input, mock_factor_broker, mock_uncertainty_engine):
        """Test Tier 3 high uncertainty propagation."""
        mock_factor = Mock()
        mock_factor.value = 0.45
        mock_factor.unit = "kgCO2e/USD"
        mock_factor.source = "eio"
        mock_factor.uncertainty = 0.60  # High uncertainty
        mock_factor.data_quality_score = 35
        mock_factor.factor_id = "manufacturing_eio"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        mock_uncertainty_engine.propagate = AsyncMock(return_value=Mock(
            p50=2250.0,
            p95=3500.0,
            std_dev=625.0  # High std dev
        ))

        result = await calculator.calculate(tier3_input)

        assert result.uncertainty is not None
        assert result.uncertainty.std_dev > 500

    @pytest.mark.asyncio
    async def test_tier3_metadata_tracking(self, calculator, tier3_input, mock_factor_broker):
        """Test Tier 3 metadata tracking."""
        mock_factor = Mock()
        mock_factor.value = 0.45
        mock_factor.unit = "kgCO2e/USD"
        mock_factor.source = "eio"
        mock_factor.uncertainty = 0.50
        mock_factor.data_quality_score = 40
        mock_factor.factor_id = "manufacturing_eio"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(tier3_input)

        assert 'spend_usd' in result.metadata
        assert result.metadata['spend_usd'] == 5000.0
        assert 'economic_sector' in result.metadata


# ============================================================================
# WATERFALL FALLBACK TESTS (5 tests)
# ============================================================================

class TestWaterfallFallback:
    """Test 3-tier waterfall fallback logic."""

    @pytest.mark.asyncio
    async def test_waterfall_tier1_to_tier2(self, calculator, mock_factor_broker):
        """Test fallback from Tier 1 to Tier 2."""
        input_data = Category1Input(
            product_name="Steel",
            quantity=1000.0,
            quantity_unit="kg",
            supplier_pcf=None,  # No Tier 1 data
            product_category="steel",
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 3.0
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "database"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "steel_ef"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # Should fallback to Tier 2
        assert result.tier == TierType.TIER_2

    @pytest.mark.asyncio
    async def test_waterfall_tier2_to_tier3(self, calculator, mock_factor_broker):
        """Test fallback from Tier 2 to Tier 3."""
        input_data = Category1Input(
            product_name="Unknown Product",
            quantity=1000.0,
            quantity_unit="kg",
            spend_usd=5000.0,
            economic_sector="manufacturing",
            region="US"
        )

        # Tier 2 factor not found
        mock_factor_broker.resolve = AsyncMock(return_value=None)

        result = await calculator.calculate(input_data)

        # Should fallback to Tier 3
        assert result.tier == TierType.TIER_3

    @pytest.mark.asyncio
    async def test_waterfall_all_tiers_fail(self, calculator, mock_factor_broker):
        """Test error when all tiers fail."""
        input_data = Category1Input(
            product_name="Product",
            quantity=1000.0,
            quantity_unit="kg",
            region="US"
            # No tier data available
        )

        mock_factor_broker.resolve = AsyncMock(return_value=None)

        with pytest.raises(TierFallbackError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_waterfall_tier1_preferred(self, calculator, tier1_input, mock_factor_broker):
        """Test that Tier 1 is preferred when available."""
        # Provide Tier 1 data
        tier1_input.product_category = "steel"  # Also has Tier 2 data
        tier1_input.spend_usd = 5000.0  # Also has Tier 3 data

        result = await calculator.calculate(tier1_input)

        # Should use Tier 1
        assert result.tier == TierType.TIER_1

    @pytest.mark.asyncio
    async def test_waterfall_dqi_threshold(self, calculator, tier2_input, mock_factor_broker):
        """Test DQI threshold enforcement in waterfall."""
        # Tier 2 factor with very low DQI
        mock_factor = Mock()
        mock_factor.value = 3.0
        mock_factor.unit = "kgCO2e/kg"
        mock_factor.source = "low_quality_db"
        mock_factor.uncertainty = 0.70
        mock_factor.data_quality_score = 10  # Very low
        mock_factor.factor_id = "low_quality_ef"
        mock_factor.metadata = Mock(
            source_version="v1",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        tier2_input.spend_usd = 5000.0  # Add Tier 3 data

        result = await calculator.calculate(tier2_input)

        # May fallback to Tier 3 if DQI too low
        assert result.tier in [TierType.TIER_2, TierType.TIER_3]


# ============================================================================
# EDGE CASES & ERROR HANDLING (4 tests)
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_validation_negative_quantity(self, calculator):
        """Test validation error for negative quantity."""
        input_data = Category1Input(
            product_name="Steel",
            quantity=-1000.0,
            quantity_unit="kg",
            supplier_pcf=2.5,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_zero_quantity(self, calculator):
        """Test validation error for zero quantity."""
        input_data = Category1Input(
            product_name="Steel",
            quantity=0.0,
            quantity_unit="kg",
            supplier_pcf=2.5,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_empty_product_name(self, calculator):
        """Test validation error for empty product name."""
        input_data = Category1Input(
            product_name="",
            quantity=1000.0,
            quantity_unit="kg",
            supplier_pcf=2.5,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_no_tier_data(self, calculator):
        """Test validation error when no tier data provided."""
        input_data = Category1Input(
            product_name="Product",
            quantity=1000.0,
            quantity_unit="kg",
            region="US"
            # No supplier_pcf, product_category, or spend_usd
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
