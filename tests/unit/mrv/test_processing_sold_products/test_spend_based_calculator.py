# -*- coding: utf-8 -*-
"""
Unit tests for SpendBasedCalculatorEngine -- AGENT-MRV-023

Tests the spend-based EEIO calculation method including revenue-to-emissions
conversion, currency conversion for 12 currencies, CPI deflation for 11 years,
margin adjustment for 12 NAICS sectors, DQI scoring, and batch processing.

Target: 25+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.agents.mrv.processing_sold_products.spend_based_calculator import (
        SpendBasedCalculatorEngine,
        SpendCalculationResult,
        SpendBreakdown,
        SpendItem,
        SpendDataQualityScore,
        SpendUncertaintyResult,
        EEIO_SECTOR_FACTORS,
        CURRENCY_RATES,
        CPI_INDEX,
        SECTOR_MARGINS,
        NAICSSector,
        CurrencyCode,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="SpendBasedCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a SpendBasedCalculatorEngine instance."""
    SpendBasedCalculatorEngine.reset()
    return SpendBasedCalculatorEngine()


# ============================================================================
# TEST: EEIO Calculation -- E = Rev * EF * (1 - margin)
# ============================================================================


class TestEEIOCalculation:
    """Test the core EEIO emissions calculation."""

    def test_known_value_naics_331_1m_usd(self, engine):
        """Known-value: $1,000,000 x 0.82 x (1 - 0.08) = $754,400 kgCO2e."""
        result = engine.calculate(
            revenue=Decimal("1000000"),
            currency="USD",
            sector="331",
            year=2024,
            org_id="ORG-001",
            reporting_year=2024,
        )
        # Revenue adjusted = 1,000,000 * (1 - 0.08) = 920,000
        # Emissions = 920,000 * 0.82 = 754,400
        assert result.total_emissions_kgco2e == Decimal("754400").quantize(_Q8)

    def test_calculate_eeio_direct(self, engine):
        """Test calculate_eeio with pre-converted USD revenue."""
        emissions = engine.calculate_eeio(Decimal("1000000"), "331")
        # 1,000,000 * (1 - 0.08) * 0.82 = 754,400
        assert emissions == Decimal("754400").quantize(_Q8)

    @pytest.mark.parametrize(
        "sector,expected_ef,expected_margin",
        [
            ("331", Decimal("0.82"), Decimal("0.08")),
            ("332", Decimal("0.45"), Decimal("0.10")),
            ("325", Decimal("0.65"), Decimal("0.12")),
            ("326", Decimal("0.52"), Decimal("0.10")),
            ("311", Decimal("0.38"), Decimal("0.08")),
            ("313", Decimal("0.42"), Decimal("0.10")),
            ("334", Decimal("0.28"), Decimal("0.15")),
            ("327", Decimal("0.72"), Decimal("0.08")),
            ("321", Decimal("0.35"), Decimal("0.10")),
            ("322", Decimal("0.48"), Decimal("0.10")),
            ("336", Decimal("0.40"), Decimal("0.12")),
            ("335", Decimal("0.32"), Decimal("0.12")),
        ],
    )
    def test_all_12_sectors(self, engine, sector, expected_ef, expected_margin):
        """Test EEIO calculation for all 12 NAICS sectors."""
        revenue = Decimal("100000")
        result = engine.calculate(
            revenue=revenue,
            currency="USD",
            sector=sector,
            year=2024,
            org_id="ORG-001",
            reporting_year=2024,
        )
        adjusted = revenue * (Decimal("1") - expected_margin)
        expected = (adjusted * expected_ef).quantize(_Q8)
        assert result.total_emissions_kgco2e == expected

    def test_unknown_sector_raises(self, engine):
        """Test that unknown sector raises ValueError."""
        with pytest.raises(ValueError, match="Unknown NAICS"):
            engine.calculate(
                revenue=Decimal("100000"),
                currency="USD",
                sector="999",
                year=2024,
                org_id="ORG-001",
                reporting_year=2024,
            )


# ============================================================================
# TEST: Currency Conversion
# ============================================================================


class TestCurrencyConversion:
    """Test currency conversion to USD."""

    @pytest.mark.parametrize(
        "currency,rate",
        [
            ("USD", Decimal("1.000")),
            ("EUR", Decimal("1.085")),
            ("GBP", Decimal("1.268")),
            ("JPY", Decimal("0.0067")),
            ("CNY", Decimal("0.138")),
            ("INR", Decimal("0.012")),
            ("CAD", Decimal("0.742")),
            ("AUD", Decimal("0.651")),
            ("KRW", Decimal("0.00075")),
            ("BRL", Decimal("0.198")),
            ("MXN", Decimal("0.058")),
            ("CHF", Decimal("1.122")),
        ],
    )
    def test_convert_to_usd_all_12_currencies(self, engine, currency, rate):
        """Test conversion for all 12 currencies."""
        amount = Decimal("10000")
        result = engine.convert_to_usd(amount, currency)
        expected = (amount * rate).quantize(_Q8)
        assert result == expected

    def test_usd_unchanged(self, engine):
        """Test that USD conversion returns the same amount."""
        result = engine.convert_to_usd(Decimal("5000"), "USD")
        assert result == Decimal("5000").quantize(_Q8)

    def test_unknown_currency_raises(self, engine):
        """Test that unknown currency raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported currency"):
            engine.convert_to_usd(Decimal("1000"), "ZZZ")

    def test_negative_amount_raises(self, engine):
        """Test that negative amount raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            engine.convert_to_usd(Decimal("-100"), "USD")


# ============================================================================
# TEST: CPI Deflation
# ============================================================================


class TestCPIDeflation:
    """Test CPI deflation to base year."""

    @pytest.mark.parametrize(
        "year,cpi_value",
        [
            (2015, Decimal("76.5")),
            (2016, Decimal("77.5")),
            (2017, Decimal("79.1")),
            (2018, Decimal("81.0")),
            (2019, Decimal("82.5")),
            (2020, Decimal("83.5")),
            (2021, Decimal("87.3")),
            (2022, Decimal("94.1")),
            (2023, Decimal("97.8")),
            (2024, Decimal("100.0")),
            (2025, Decimal("102.4")),
        ],
    )
    def test_deflation_all_11_years(self, engine, year, cpi_value):
        """Test CPI deflation from each of 11 years to base year 2024."""
        amount = Decimal("10000")
        result = engine.deflate_to_base(amount, year)
        cpi_ratio = (Decimal("100.0") / cpi_value).quantize(_Q8)
        expected = (amount * cpi_ratio).quantize(_Q8)
        assert result == expected

    def test_deflation_base_year_unchanged(self, engine):
        """Test that deflation from base year returns the same amount."""
        result = engine.deflate_to_base(Decimal("10000"), 2024)
        assert result == Decimal("10000").quantize(_Q8)

    def test_deflation_unavailable_year_raises(self, engine):
        """Test that unavailable year raises ValueError."""
        with pytest.raises(ValueError, match="CPI data not available"):
            engine.deflate_to_base(Decimal("10000"), 2010)


# ============================================================================
# TEST: Margin Adjustment
# ============================================================================


class TestMarginAdjustment:
    """Test margin removal from revenue."""

    def test_margin_adjustment_331(self, engine):
        """Test margin adjustment for sector 331 (8% margin)."""
        result = engine.apply_margin_adjustment(Decimal("100000"), "331")
        expected = (Decimal("100000") * Decimal("0.92")).quantize(_Q8)
        assert result == expected

    def test_margin_adjustment_334(self, engine):
        """Test margin adjustment for sector 334 (15% margin)."""
        result = engine.apply_margin_adjustment(Decimal("100000"), "334")
        expected = (Decimal("100000") * Decimal("0.85")).quantize(_Q8)
        assert result == expected


# ============================================================================
# TEST: DQI Scoring
# ============================================================================


class TestSpendDQI:
    """Test DQI scoring for spend-based method."""

    def test_dqi_composite_score_30(self, engine):
        """Test that spend-based DQI composite score is 30."""
        dqi = engine.compute_dqi_score()
        assert dqi.composite == 30

    def test_dqi_method_is_spend_based(self, engine):
        """Test that DQI method field is 'spend_based'."""
        dqi = engine.compute_dqi_score()
        assert dqi.method == "spend_based"

    def test_dqi_dimensions_present(self, engine):
        """Test that all 5 DQI dimensions are present."""
        dqi = engine.compute_dqi_score()
        d = dqi.to_dict()
        for dim in ["reliability", "completeness", "temporal", "geographical", "technological"]:
            assert dim in d


# ============================================================================
# TEST: Batch Calculation
# ============================================================================


class TestBatchCalculation:
    """Test batch processing of multiple spend items."""

    def test_batch_two_items(self, engine):
        """Test batch calculation with two items."""
        items = [
            {"item_id": "I1", "product_name": "Steel", "revenue": "500000",
             "currency": "USD", "sector": "331", "year": 2024},
            {"item_id": "I2", "product_name": "Plastic", "revenue": "300000",
             "currency": "USD", "sector": "326", "year": 2024},
        ]
        result = engine.calculate_batch(items, "ORG-001", 2024)
        assert result.item_count == 2
        assert result.total_emissions_kgco2e > Decimal("0")
        assert len(result.breakdowns) == 2

    def test_batch_empty_raises(self, engine):
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_batch([], "ORG-001", 2024)


# ============================================================================
# TEST: Uncertainty
# ============================================================================


class TestSpendUncertainty:
    """Test uncertainty quantification for spend-based method."""

    def test_uncertainty_50_pct_default(self, engine):
        """Test default spend-based uncertainty of +/-50%."""
        unc = engine.compute_uncertainty(Decimal("100000"))
        assert unc.lower_bound_kgco2e == Decimal("50000").quantize(_Q8)
        assert unc.upper_bound_kgco2e == Decimal("150000").quantize(_Q8)
        assert unc.uncertainty_pct == Decimal("50")
        assert unc.confidence_level == 95


# ============================================================================
# TEST: Health Check and Singleton
# ============================================================================


class TestSpendEngineStatus:
    """Test engine health check and singleton behavior."""

    def test_health_check_status(self, engine):
        """Test that health check returns healthy status."""
        status = engine.health_check()
        assert status["status"] == "healthy"
        assert status["engine"] == "SpendBasedCalculatorEngine"
        assert status["sectors"] == 12
        assert status["currencies"] == 12
        assert status["cpi_years"] == 11

    def test_singleton_identity(self, engine):
        """Test that two instantiations return the same object."""
        engine2 = SpendBasedCalculatorEngine()
        assert engine is engine2

    def test_list_sectors(self, engine):
        """Test list_sectors returns all 12 sectors."""
        sectors = engine.list_sectors()
        assert len(sectors) == 12

    def test_list_currencies(self, engine):
        """Test list_currencies returns all 12 currencies."""
        currencies = engine.list_currencies()
        assert len(currencies) == 12

    def test_list_cpi_years(self, engine):
        """Test list_cpi_years returns all 11 years."""
        years = engine.list_cpi_years()
        assert len(years) == 11
        assert 2015 in years
        assert 2025 in years


# ============================================================================
# TEST: Provenance
# ============================================================================


class TestSpendProvenance:
    """Test provenance hashing in spend-based calculations."""

    def test_provenance_hash_64_char(self, engine):
        """Test that result provenance_hash is a 64-char hex string."""
        result = engine.calculate(
            revenue=Decimal("100000"),
            currency="USD",
            sector="331",
            year=2024,
            org_id="ORG-001",
            reporting_year=2024,
        )
        h = result.provenance_hash
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
