# -*- coding: utf-8 -*-
"""
Unit tests for SpendBasedCalculatorEngine (AGENT-MRV-019, Engine 5)

35 tests covering:
- NAICS code EEIO factor lookups (air, hotel, taxi, rail, car rental, ground, water, restaurant)
- Multi-currency conversion (EUR, GBP, JPY)
- CPI deflation for reporting years 2020-2024
- Margin removal (enabled/disabled)
- Expense classification (keyword matching)
- Batch processing
- Input validation (invalid NAICS, zero/negative amount)
- Provenance hashing
- Singleton pattern
- Decimal precision

Spend-based formula:
    spend_usd       = amount x currency_rate
    deflated_spend  = spend_usd / cpi_deflator(year)
    adjusted_spend  = deflated_spend x (1 - margin_rate) if enabled, else deflated_spend
    co2e            = adjusted_spend x eeio_factor

Author: GL-TestEngineer
Date: February 2026
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch, MagicMock, PropertyMock

from greenlang.business_travel.spend_based_calculator import (
    SpendBasedCalculatorEngine,
)
from greenlang.business_travel.models import (
    SpendInput,
    SpendResult,
    CurrencyCode,
    EFSource,
    EEIO_FACTORS,
    CURRENCY_RATES,
    CPI_DEFLATORS,
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")


def _q(value: Decimal) -> Decimal:
    """Quantize to 8 decimal places."""
    return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test for isolation."""
    SpendBasedCalculatorEngine.reset_instance()
    yield
    SpendBasedCalculatorEngine.reset_instance()


def _make_mock_config(
    enable_cpi_deflation: bool = True,
    enable_margin_removal: bool = False,
    default_margin_rate: Decimal = Decimal("0.15"),
    base_year: int = 2021,
    default_currency: str = "USD",
):
    """Create a mock config with configurable spend settings."""
    mock_cfg = MagicMock()
    mock_cfg.spend.enable_cpi_deflation = enable_cpi_deflation
    mock_cfg.spend.enable_margin_removal = enable_margin_removal
    mock_cfg.spend.default_margin_rate = default_margin_rate
    mock_cfg.spend.base_year = base_year
    mock_cfg.spend.default_currency = default_currency
    return mock_cfg


@pytest.fixture
def engine():
    """Create a fresh SpendBasedCalculatorEngine with mocked dependencies."""
    with patch(
        "greenlang.business_travel.spend_based_calculator.get_config"
    ) as mock_config, patch(
        "greenlang.business_travel.spend_based_calculator.get_metrics"
    ) as mock_metrics:
        mock_config.return_value = _make_mock_config()
        mock_metrics.return_value = MagicMock()
        eng = SpendBasedCalculatorEngine()
        yield eng


@pytest.fixture
def engine_with_margin():
    """Engine with margin removal enabled (15% margin)."""
    with patch(
        "greenlang.business_travel.spend_based_calculator.get_config"
    ) as mock_config, patch(
        "greenlang.business_travel.spend_based_calculator.get_metrics"
    ) as mock_metrics:
        mock_config.return_value = _make_mock_config(enable_margin_removal=True)
        mock_metrics.return_value = MagicMock()
        eng = SpendBasedCalculatorEngine()
        yield eng


# ==============================================================================
# NAICS EEIO FACTOR TESTS
# ==============================================================================


class TestNAICSFactors:
    """Tests for NAICS code EEIO factor lookups and spend calculations."""

    def test_air_5000usd_2024(self, engine):
        """Air (481000): 5000 USD, 2024. Deflated = 5000/1.149, x 0.477."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("5000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        result = engine.calculate(inp)
        # Manual: 5000/1.149 = 4351.6101... x 0.477 = 2075.72 (approx)
        assert result.co2e > Decimal("2000")
        assert result.co2e < Decimal("2200")
        assert result.naics_code == "481000"

    def test_hotel_3000usd(self, engine):
        """Hotel (721100): ef = 0.149."""
        inp = SpendInput(
            naics_code="721100",
            amount=Decimal("3000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.1490")
        assert result.co2e > Decimal("0")

    def test_taxi_500usd(self, engine):
        """Taxi (485310): ef = 0.280."""
        inp = SpendInput(
            naics_code="485310",
            amount=Decimal("500.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.2800")

    def test_rail_1000usd(self, engine):
        """Rail (482000): ef = 0.310."""
        inp = SpendInput(
            naics_code="482000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.3100")

    def test_car_rental_2000usd(self, engine):
        """Car rental (532100): ef = 0.195."""
        inp = SpendInput(
            naics_code="532100",
            amount=Decimal("2000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.1950")

    def test_ground_transport(self, engine):
        """Ground passenger transport (485000): ef = 0.260."""
        inp = SpendInput(
            naics_code="485000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.2600")

    def test_water_transport(self, engine):
        """Water transportation (483000): ef = 0.520."""
        inp = SpendInput(
            naics_code="483000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.5200")

    def test_restaurant(self, engine):
        """Restaurant (722500): ef = 0.205."""
        inp = SpendInput(
            naics_code="722500",
            amount=Decimal("500.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.2050")


# ==============================================================================
# CURRENCY CONVERSION TESTS
# ==============================================================================


class TestCurrencyConversion:
    """Tests for multi-currency conversion to USD."""

    def test_eur_conversion(self, engine):
        """EUR 1000 -> 1000 x 1.085 = 1085 USD."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.EUR,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        expected_usd = _q(Decimal("1000.00") * Decimal("1.0850"))
        assert result.spend_usd == expected_usd

    def test_gbp_conversion(self, engine):
        """GBP 1000 -> 1000 x 1.265 = 1265 USD."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.GBP,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        expected_usd = _q(Decimal("1000.00") * Decimal("1.2650"))
        assert result.spend_usd == expected_usd

    def test_jpy_conversion(self, engine):
        """JPY uses a small conversion rate (0.006667)."""
        inp = SpendInput(
            naics_code="721100",
            amount=Decimal("100000"),
            currency=CurrencyCode.JPY,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        expected_usd = _q(Decimal("100000") * Decimal("0.006667"))
        assert result.spend_usd == expected_usd


# ==============================================================================
# CPI DEFLATION TESTS
# ==============================================================================


class TestCPIDeflation:
    """Tests for CPI deflation normalization to base year 2021."""

    def test_cpi_deflation_2024(self, engine):
        """2024 deflator = 1.149: spend_usd / 1.149."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("5000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        result = engine.calculate(inp)
        assert result.cpi_deflator == Decimal("1.1490")

    def test_cpi_deflation_2021_no_change(self, engine):
        """2021 deflator = 1.0: no deflation applied."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.cpi_deflator == Decimal("1.0000")
        # No deflation: co2e = 1000 x 0.477 = 477.0
        expected = _q(Decimal("1000.00") * Decimal("0.4770"))
        assert result.co2e == expected

    def test_cpi_deflation_2020(self, engine):
        """2020 deflator = 0.9271."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2020,
        )
        result = engine.calculate(inp)
        assert result.cpi_deflator == Decimal("0.9271")


# ==============================================================================
# MARGIN REMOVAL TESTS
# ==============================================================================


class TestMarginRemoval:
    """Tests for profit margin removal from spend amounts."""

    def test_no_margin_removal_default(self, engine):
        """Margin removal disabled by default -- full spend used."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        # No deflation (2021), no margin: co2e = 1000 x 0.477
        expected = _q(Decimal("1000") * Decimal("0.4770"))
        assert result.co2e == expected

    def test_margin_removal_when_enabled(self, engine_with_margin):
        """Margin removal enabled: spend x (1 - 0.15) = spend x 0.85."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine_with_margin.calculate(inp)
        # With margin: 1000 x 0.85 x 0.477 = 405.45
        adjusted = _q(Decimal("1000") * Decimal("0.85"))
        expected = _q(adjusted * Decimal("0.4770"))
        assert result.co2e == expected


# ==============================================================================
# RESULT STRUCTURE & PROVENANCE TESTS
# ==============================================================================


class TestResultStructure:
    """Tests for result type, provenance, and stored fields."""

    def test_provenance_hash_present(self, engine):
        """Result contains a 64-char SHA-256 provenance hash."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_result_is_spend_result(self, engine):
        """Calculate returns a SpendResult instance."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert isinstance(result, SpendResult)

    def test_ef_source_eeio(self, engine):
        """EF source should be EEIO for spend-based calculations."""
        inp = SpendInput(
            naics_code="721100",
            amount=Decimal("2000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.ef_source == EFSource.EEIO

    def test_spend_usd_stored(self, engine):
        """spend_usd field should be populated after currency conversion."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.EUR,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.spend_usd > Decimal("0")

    def test_eeio_factor_stored(self, engine):
        """eeio_factor field should match the NAICS lookup."""
        inp = SpendInput(
            naics_code="532100",
            amount=Decimal("500.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        result = engine.calculate(inp)
        assert result.eeio_factor == Decimal("0.1950")

    def test_decimal_precision(self, engine):
        """co2e should have 8 decimal places."""
        inp = SpendInput(
            naics_code="481000",
            amount=Decimal("3333.33"),
            currency=CurrencyCode.USD,
            reporting_year=2023,
        )
        result = engine.calculate(inp)
        co2e_str = str(result.co2e)
        if "." in co2e_str:
            assert len(co2e_str.split(".")[1]) == 8


# ==============================================================================
# INPUT VALIDATION TESTS
# ==============================================================================


class TestInputValidation:
    """Tests for invalid input handling."""

    def test_invalid_naics_raises(self, engine):
        """Invalid NAICS code raises ValueError."""
        inp = SpendInput(
            naics_code="999999",
            amount=Decimal("1000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2021,
        )
        with pytest.raises(ValueError, match="NAICS code"):
            engine.calculate(inp)

    def test_zero_amount_raises(self):
        """Zero amount should raise during model validation."""
        with pytest.raises(Exception):
            SpendInput(
                naics_code="481000",
                amount=Decimal("0"),
                currency=CurrencyCode.USD,
                reporting_year=2021,
            )

    def test_negative_amount_raises(self):
        """Negative amount should raise during model validation."""
        with pytest.raises(Exception):
            SpendInput(
                naics_code="481000",
                amount=Decimal("-100"),
                currency=CurrencyCode.USD,
                reporting_year=2021,
            )


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:
    """Tests for batch spend calculations."""

    def test_batch_3_expenses(self, engine):
        """Batch of 3 valid inputs returns 3 results."""
        inputs = [
            SpendInput(naics_code="481000", amount=Decimal("1000"), currency=CurrencyCode.USD, reporting_year=2021),
            SpendInput(naics_code="721100", amount=Decimal("2000"), currency=CurrencyCode.USD, reporting_year=2021),
            SpendInput(naics_code="485310", amount=Decimal("500"), currency=CurrencyCode.USD, reporting_year=2021),
        ]
        results = engine.calculate_batch(inputs)
        assert len(results) == 3

    def test_batch_empty(self, engine):
        """Empty batch raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            engine.calculate_batch([])

    def test_available_naics_10_codes(self, engine):
        """get_available_naics_codes returns exactly 10 NAICS codes."""
        codes = engine.get_available_naics_codes()
        assert len(codes) == 10


# ==============================================================================
# EXPENSE CLASSIFICATION TESTS
# ==============================================================================


class TestExpenseClassification:
    """Tests for keyword-based expense classification."""

    def test_classify_expense_flight(self, engine):
        """Flight keywords classify to NAICS 481000."""
        assert engine.classify_expense_category("Delta flight to Chicago") == "481000"

    def test_classify_expense_hotel(self, engine):
        """Hotel keywords classify to NAICS 721100."""
        assert engine.classify_expense_category("Marriott hotel stay") == "721100"

    def test_classify_expense_uber(self, engine):
        """Uber/taxi keywords classify to NAICS 485310."""
        assert engine.classify_expense_category("Uber ride to airport") == "485310"

    def test_classify_expense_hertz(self, engine):
        """Hertz/rental keywords classify to NAICS 532100."""
        assert engine.classify_expense_category("Hertz car rental") == "532100"

    def test_classify_expense_train(self, engine):
        """Train keywords classify to NAICS 482000."""
        assert engine.classify_expense_category("Amtrak train Boston to NYC") == "482000"

    def test_classify_expense_unknown(self, engine):
        """Unknown description returns None."""
        assert engine.classify_expense_category("office supplies purchase") is None


# ==============================================================================
# SINGLETON PATTERN TEST
# ==============================================================================


class TestSingleton:
    """Tests for singleton instance management."""

    def test_singleton_pattern(self):
        """get_instance returns the same object on repeated calls."""
        with patch(
            "greenlang.business_travel.spend_based_calculator.get_config"
        ) as mock_config, patch(
            "greenlang.business_travel.spend_based_calculator.get_metrics"
        ) as mock_metrics:
            mock_config.return_value = _make_mock_config()
            mock_metrics.return_value = MagicMock()
            inst1 = SpendBasedCalculatorEngine.get_instance()
            inst2 = SpendBasedCalculatorEngine.get_instance()
            assert inst1 is inst2
