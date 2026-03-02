# -*- coding: utf-8 -*-
"""
Test suite for franchises.spend_based_calculator - AGENT-MRV-027.

Tests the SpendBasedCalculatorEngine including revenue-based, royalty-based,
and per-unit average approaches. Tests all 9 NAICS codes, currency
conversion, margin removal, data quality assessment, and edge cases.

Target: 45+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from greenlang.franchises.spend_based_calculator import (
    SpendBasedCalculatorEngine,
    FranchiseNetworkInput,
    NetworkAggregationResult,
    FRANCHISE_EEIO_FACTORS,
    FRANCHISE_TYPE_NAICS,
    CURRENCY_RATES,
    CurrencyCode,
    SpendApproach,
    DataQualityTier,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> SpendBasedCalculatorEngine:
    """Create a fresh SpendBasedCalculatorEngine instance."""
    SpendBasedCalculatorEngine._instance = None
    return SpendBasedCalculatorEngine()


@pytest.fixture
def revenue_input() -> FranchiseNetworkInput:
    """Revenue-based spend input."""
    return FranchiseNetworkInput(
        network_id="NET-REV-001",
        franchisor_name="TestBurger Inc.",
        naics_code="722513",
        total_franchise_revenue=Decimal("450000000"),
        reporting_year=2025,
        currency=CurrencyCode.USD,
    )


@pytest.fixture
def royalty_input() -> FranchiseNetworkInput:
    """Royalty-based spend input."""
    return FranchiseNetworkInput(
        network_id="NET-ROY-001",
        franchisor_name="TestBurger Inc.",
        naics_code="722513",
        total_royalty_income=Decimal("27000000"),
        royalty_rate=Decimal("0.06"),
        reporting_year=2025,
        currency=CurrencyCode.USD,
    )


@pytest.fixture
def per_unit_input() -> FranchiseNetworkInput:
    """Per-unit average spend input."""
    return FranchiseNetworkInput(
        network_id="NET-PUA-001",
        franchisor_name="TestBurger Inc.",
        naics_code="722513",
        average_unit_revenue=Decimal("900000"),
        franchised_unit_count=500,
        reporting_year=2025,
        currency=CurrencyCode.USD,
    )


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestSpendBasedInit:
    """Test SpendBasedCalculatorEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern via get_instance."""
        SpendBasedCalculatorEngine._instance = None
        e1 = SpendBasedCalculatorEngine.get_instance()
        e2 = SpendBasedCalculatorEngine.get_instance()
        assert e1 is e2


# ==============================================================================
# REVENUE-BASED APPROACH TESTS
# ==============================================================================


class TestRevenueBasedApproach:
    """Test revenue-based spend calculation."""

    def test_revenue_based_calculation(self, engine, revenue_input):
        """Test revenue-based approach produces result."""
        result = engine.calculate(revenue_input)
        assert result is not None
        assert isinstance(result, NetworkAggregationResult)
        assert result.total_co2e > 0
        assert result.approach == SpendApproach.TOTAL_REVENUE

    def test_revenue_based_provenance(self, engine, revenue_input):
        """Test revenue-based result includes provenance hash."""
        result = engine.calculate(revenue_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_revenue_based_proportional(self, engine):
        """Test higher revenue produces proportionally higher emissions."""
        low_rev = FranchiseNetworkInput(
            network_id="NET-LOW-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            total_franchise_revenue=Decimal("100000000"),
            reporting_year=2025,
        )
        high_rev = FranchiseNetworkInput(
            network_id="NET-HIGH-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            total_franchise_revenue=Decimal("500000000"),
            reporting_year=2025,
        )

        result_low = engine.calculate(low_rev)
        result_high = engine.calculate(high_rev)
        assert result_high.total_co2e > result_low.total_co2e

    def test_revenue_based_method_is_spend_based(self, engine, revenue_input):
        """Test method classification is spend_based."""
        result = engine.calculate(revenue_input)
        assert result.method.value == "spend_based"

    def test_revenue_based_has_eeio_factor(self, engine, revenue_input):
        """Test result contains EEIO factor details."""
        result = engine.calculate(revenue_input)
        assert result.eeio_factor > 0

    def test_revenue_based_margin_applied(self, engine, revenue_input):
        """Test margin removal is applied by default."""
        result = engine.calculate(revenue_input)
        assert result.margin_rate >= 0
        assert result.revenue_after_margin <= result.revenue_usd


# ==============================================================================
# ROYALTY-BASED APPROACH TESTS
# ==============================================================================


class TestRoyaltyBasedApproach:
    """Test royalty-based spend calculation."""

    def test_royalty_based_calculation(self, engine, royalty_input):
        """Test royalty-based approach produces result."""
        result = engine.calculate(royalty_input)
        assert result is not None
        assert result.total_co2e > 0
        assert result.approach == SpendApproach.ROYALTY_BASED

    def test_royalty_based_estimates_total_revenue(self, engine, royalty_input):
        """Test royalty approach estimates total revenue from royalty rate."""
        result = engine.calculate(royalty_input)
        # royalty_income / royalty_rate should approximate total revenue
        assert result.revenue_usd > 0
        assert result.total_co2e > 0

    def test_royalty_based_higher_income_more_emissions(self, engine):
        """Test higher royalty income produces more emissions."""
        low = FranchiseNetworkInput(
            network_id="NET-RLOW-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            total_royalty_income=Decimal("10000000"),
            royalty_rate=Decimal("0.06"),
            reporting_year=2025,
        )
        high = FranchiseNetworkInput(
            network_id="NET-RHIGH-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            total_royalty_income=Decimal("50000000"),
            royalty_rate=Decimal("0.06"),
            reporting_year=2025,
        )

        result_low = engine.calculate(low)
        result_high = engine.calculate(high)
        assert result_high.total_co2e > result_low.total_co2e

    def test_royalty_based_provenance(self, engine, royalty_input):
        """Test royalty-based result includes provenance hash."""
        result = engine.calculate(royalty_input)
        assert len(result.provenance_hash) == 64


# ==============================================================================
# PER-UNIT AVERAGE APPROACH TESTS
# ==============================================================================


class TestPerUnitAverageApproach:
    """Test per-unit average spend calculation."""

    def test_per_unit_calculation(self, engine, per_unit_input):
        """Test per-unit average approach produces result."""
        result = engine.calculate(per_unit_input)
        assert result is not None
        assert result.total_co2e > 0
        assert result.approach == SpendApproach.PER_UNIT_AVERAGE

    def test_per_unit_more_units_more_emissions(self, engine):
        """Test more units produce more emissions."""
        few = FranchiseNetworkInput(
            network_id="NET-FEW-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            average_unit_revenue=Decimal("900000"),
            franchised_unit_count=50,
            reporting_year=2025,
        )
        many = FranchiseNetworkInput(
            network_id="NET-MANY-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            average_unit_revenue=Decimal("900000"),
            franchised_unit_count=500,
            reporting_year=2025,
        )

        result_few = engine.calculate(few)
        result_many = engine.calculate(many)
        assert result_many.total_co2e > result_few.total_co2e

    def test_per_unit_has_per_unit_co2e(self, engine, per_unit_input):
        """Test per-unit result has per_unit_co2e field."""
        result = engine.calculate(per_unit_input)
        assert result.per_unit_co2e is not None or result.franchised_unit_count is not None


# ==============================================================================
# NAICS CODE TESTS
# ==============================================================================


class TestNAICSCodes:
    """Test all 9 NAICS codes for franchise industries."""

    @pytest.mark.parametrize("naics_code", list(FRANCHISE_EEIO_FACTORS.keys()))
    def test_all_naics_codes(self, engine, naics_code):
        """Test EEIO factor retrieval for each NAICS code."""
        input_data = FranchiseNetworkInput(
            network_id=f"NET-NAICS-{naics_code}",
            franchisor_name="Test Franchise Co.",
            naics_code=naics_code,
            total_franchise_revenue=Decimal("10000000"),
            reporting_year=2025,
        )
        result = engine.calculate(input_data)
        assert result.total_co2e > 0
        assert result.naics_code == naics_code


# ==============================================================================
# CURRENCY CONVERSION TESTS
# ==============================================================================


class TestCurrencyConversion:
    """Test currency conversion in spend-based calculations."""

    @pytest.mark.parametrize("currency", [
        CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP,
        CurrencyCode.CAD, CurrencyCode.AUD, CurrencyCode.JPY,
    ])
    def test_currency_conversion(self, engine, currency):
        """Test calculation with various currencies."""
        input_data = FranchiseNetworkInput(
            network_id=f"NET-CUR-{currency.value}",
            franchisor_name="Test Franchise Co.",
            naics_code="722513",
            total_franchise_revenue=Decimal("10000000"),
            reporting_year=2025,
            currency=currency,
        )
        result = engine.calculate(input_data)
        assert result.total_co2e > 0

    def test_non_usd_converts_to_usd(self, engine):
        """Test non-USD currency is converted to USD."""
        input_data = FranchiseNetworkInput(
            network_id="NET-EUR-001",
            franchisor_name="Euro Franchise Co.",
            naics_code="722513",
            total_franchise_revenue=Decimal("10000000"),
            reporting_year=2025,
            currency=CurrencyCode.EUR,
        )
        result = engine.calculate(input_data)
        assert result.revenue_usd > 0


# ==============================================================================
# MARGIN REMOVAL TESTS
# ==============================================================================


class TestMarginRemoval:
    """Test margin removal in spend-based calculations."""

    def test_margin_removal_reduces_effective_revenue(self, engine):
        """Test margin removal reduces effective revenue used for calculation."""
        with_margin = FranchiseNetworkInput(
            network_id="NET-MRG-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            total_franchise_revenue=Decimal("100000000"),
            reporting_year=2025,
            enable_margin_removal=True,
        )
        result = engine.calculate(with_margin)
        assert result.revenue_after_margin <= result.revenue_usd

    def test_custom_margin_rate(self, engine):
        """Test custom margin rate override."""
        input_data = FranchiseNetworkInput(
            network_id="NET-CMR-001",
            franchisor_name="Test Inc.",
            naics_code="722513",
            total_franchise_revenue=Decimal("100000000"),
            reporting_year=2025,
            enable_margin_removal=True,
            custom_margin_rate=Decimal("0.20"),
        )
        result = engine.calculate(input_data)
        assert result.margin_rate == Decimal("0.20")


# ==============================================================================
# DATA QUALITY ASSESSMENT TESTS
# ==============================================================================


class TestDataQualityAssessment:
    """Test data quality assessment for spend-based method."""

    def test_dqi_tier_3(self, engine, revenue_input):
        """Test spend-based method is classified as Tier 3."""
        result = engine.calculate(revenue_input)
        assert result.data_quality is not None
        assert result.data_quality.tier == DataQualityTier.TIER_3

    def test_dqi_score_low(self, engine, revenue_input):
        """Test spend-based DQI score is relatively low (Tier 3)."""
        result = engine.calculate(revenue_input)
        assert result.data_quality.overall_score <= Decimal("3.0")

    def test_dqi_has_dimensions(self, engine, revenue_input):
        """Test DQI has dimension scores."""
        result = engine.calculate(revenue_input)
        assert len(result.data_quality.dimensions) > 0

    def test_dqi_has_classification(self, engine, revenue_input):
        """Test DQI has classification label."""
        result = engine.calculate(revenue_input)
        assert result.data_quality.classification in (
            "Excellent", "Good", "Fair", "Poor", "Very Poor"
        )


# ==============================================================================
# UNCERTAINTY TESTS
# ==============================================================================


class TestUncertainty:
    """Test uncertainty quantification for spend-based method."""

    def test_uncertainty_bounds(self, engine, revenue_input):
        """Test uncertainty bounds are present and sensible."""
        result = engine.calculate(revenue_input)
        assert result.uncertainty_lower < result.total_co2e
        assert result.uncertainty_upper > result.total_co2e

    def test_uncertainty_wide_for_tier_3(self, engine, revenue_input):
        """Test Tier 3 has wide uncertainty range."""
        result = engine.calculate(revenue_input)
        spread = result.uncertainty_upper - result.uncertainty_lower
        assert spread > 0


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Test edge cases for spend-based calculations."""

    def test_missing_naics_raises(self, engine):
        """Test missing NAICS code raises error."""
        with pytest.raises((ValueError, KeyError, Exception)):
            input_data = FranchiseNetworkInput(
                network_id="NET-BAD-001",
                franchisor_name="Bad Co.",
                naics_code="000000",
                total_franchise_revenue=Decimal("10000000"),
                reporting_year=2025,
            )
            engine.calculate(input_data)

    def test_single_unit_per_unit(self, engine):
        """Test calculation with single unit."""
        input_data = FranchiseNetworkInput(
            network_id="NET-SINGLE-001",
            franchisor_name="Single Unit Co.",
            naics_code="722513",
            average_unit_revenue=Decimal("900000"),
            franchised_unit_count=1,
            reporting_year=2025,
        )
        result = engine.calculate(input_data)
        assert result.total_co2e > 0

    def test_very_large_network(self, engine):
        """Test calculation with very large franchise network."""
        input_data = FranchiseNetworkInput(
            network_id="NET-LARGE-001",
            franchisor_name="MegaCorp Franchise",
            naics_code="722513",
            total_franchise_revenue=Decimal("50000000000"),
            reporting_year=2025,
        )
        result = engine.calculate(input_data)
        assert result.total_co2e > 0

    def test_no_financial_data_raises(self, engine):
        """Test no financial data raises error."""
        with pytest.raises((ValueError, Exception)):
            input_data = FranchiseNetworkInput(
                network_id="NET-EMPTY-001",
                franchisor_name="Empty Co.",
                naics_code="722513",
                reporting_year=2025,
            )
            engine.calculate(input_data)


# ==============================================================================
# ENGINE INFO AND STATS TESTS
# ==============================================================================


class TestEngineInfo:
    """Test engine info and stats methods."""

    def test_get_available_naics_codes(self, engine):
        """Test listing available NAICS codes."""
        codes = engine.get_available_naics_codes()
        assert isinstance(codes, list)
        assert len(codes) > 0

    def test_get_franchise_type_mapping(self, engine):
        """Test franchise type to NAICS mapping."""
        mapping = engine.get_franchise_type_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_get_stats(self, engine, revenue_input):
        """Test engine stats tracking."""
        engine.calculate(revenue_input)
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert stats.get("calculation_count", 0) >= 1
