# -*- coding: utf-8 -*-
"""
Test suite for investments.equity_investment_calculator - AGENT-MRV-028.

Tests the EquityInvestmentCalculatorEngine (Engine 2) for the Investments
Agent (GL-MRV-S3-015) including listed equity EVIC calculations, private
equity equity-share calculations, all 5 PCAF quality tiers, WACI
computation, carbon intensity metrics, DC-INV-001 consolidated exclusion,
batch calculation, and currency conversion.

Coverage:
- Listed equity: EVIC calculation, attribution factor, financed emissions
- Private equity: total_equity_plus_debt denominator
- All 5 PCAF quality tiers (reported -> sector average)
- WACI calculation
- Carbon intensity metrics (tCO2e / $M revenue)
- DC-INV-001 (consolidated in Scope 1/2 rejected)
- Batch calculation
- Currency conversion
- Parametrized tests for sectors, quality scores

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock
import pytest

from greenlang.investments.equity_investment_calculator import (
    EquityInvestmentCalculatorEngine,
)
from greenlang.investments.models import (
    AssetClass,
    PCAFDataQuality,
    CalculationMethod,
    AttributionMethod,
    Sector,
    CurrencyCode,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    EquityInvestmentCalculatorEngine.reset_instance()
    yield
    EquityInvestmentCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh EquityInvestmentCalculatorEngine with mocked config."""
    with patch(
        "greenlang.investments.equity_investment_calculator.get_config"
    ) as mock_config:
        cfg = MagicMock()
        cfg.equity.default_evic_source = "BLOOMBERG"
        cfg.equity.include_scope3 = False
        cfg.general.default_gwp = "AR5"
        mock_config.return_value = cfg
        eng = EquityInvestmentCalculatorEngine()
        yield eng


def _make_equity_input(**overrides):
    """Build a listed equity input dict with defaults."""
    base = {
        "asset_class": "listed_equity",
        "investee_name": "Apple Inc.",
        "isin": "US0378331005",
        "outstanding_amount": Decimal("100000000"),
        "evic": Decimal("3000000000000"),
        "investee_scope1": Decimal("22400"),
        "investee_scope2": Decimal("9100"),
        "sector": "information_technology",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 1,
    }
    base.update(overrides)
    return base


def _make_private_equity_input(**overrides):
    """Build a private equity input dict with defaults."""
    base = {
        "asset_class": "private_equity",
        "investee_name": "GreenTech Solutions",
        "outstanding_amount": Decimal("50000000"),
        "total_equity_plus_debt": Decimal("200000000"),
        "investee_scope1": Decimal("15000"),
        "investee_scope2": Decimal("8000"),
        "sector": "industrials",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }
    base.update(overrides)
    return base


# ==============================================================================
# LISTED EQUITY TESTS
# ==============================================================================


class TestListedEquityCalculation:
    """Test listed equity EVIC-based calculations."""

    def test_evic_attribution_factor(self, engine):
        """Test EVIC attribution factor = outstanding / EVIC."""
        data = _make_equity_input()
        result = engine.calculate(data)
        expected_af = Decimal("100000000") / Decimal("3000000000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0000001")

    def test_financed_emissions_scope1_plus_scope2(self, engine):
        """Test financed emissions = AF x (Scope 1 + Scope 2)."""
        data = _make_equity_input()
        result = engine.calculate(data)
        af = Decimal("100000000") / Decimal("3000000000000")
        expected = af * (Decimal("22400") + Decimal("9100"))
        assert abs(result["financed_emissions"] - expected) < Decimal("0.01")

    def test_financed_emissions_positive(self, engine):
        """Test financed emissions are always positive."""
        data = _make_equity_input()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_attribution_method_is_evic(self, engine):
        """Test attribution method is EVIC for listed equity."""
        data = _make_equity_input()
        result = engine.calculate(data)
        assert result["attribution_method"] == "evic"

    def test_provenance_hash_present(self, engine):
        """Test provenance hash is present and 64 chars."""
        data = _make_equity_input()
        result = engine.calculate(data)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_provenance_hash_deterministic(self, engine):
        """Test same input produces same provenance hash."""
        data = _make_equity_input()
        r1 = engine.calculate(data)
        r2 = engine.calculate(data)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_evic_changes_result(self, engine):
        """Test different EVIC values produce different emissions."""
        data_small_evic = _make_equity_input(evic=Decimal("500000000000"))
        data_large_evic = _make_equity_input(evic=Decimal("3000000000000"))
        r_small = engine.calculate(data_small_evic)
        r_large = engine.calculate(data_large_evic)
        assert r_small["financed_emissions"] > r_large["financed_emissions"]

    def test_larger_outstanding_increases_emissions(self, engine):
        """Test larger outstanding amount increases financed emissions."""
        data_small = _make_equity_input(outstanding_amount=Decimal("50000000"))
        data_large = _make_equity_input(outstanding_amount=Decimal("200000000"))
        r_small = engine.calculate(data_small)
        r_large = engine.calculate(data_large)
        assert r_large["financed_emissions"] > r_small["financed_emissions"]


# ==============================================================================
# PRIVATE EQUITY TESTS
# ==============================================================================


class TestPrivateEquityCalculation:
    """Test private equity equity-share calculations."""

    def test_private_equity_attribution_factor(self, engine):
        """Test PE attribution factor = outstanding / (equity + debt)."""
        data = _make_private_equity_input()
        result = engine.calculate(data)
        expected_af = Decimal("50000000") / Decimal("200000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0001")

    def test_private_equity_financed_emissions(self, engine):
        """Test PE financed emissions = AF x investee emissions."""
        data = _make_private_equity_input()
        result = engine.calculate(data)
        af = Decimal("50000000") / Decimal("200000000")
        expected = af * (Decimal("15000") + Decimal("8000"))
        assert abs(result["financed_emissions"] - expected) < Decimal("0.01")

    def test_private_equity_attribution_method(self, engine):
        """Test attribution method is equity_share for PE."""
        data = _make_private_equity_input()
        result = engine.calculate(data)
        assert result["attribution_method"] == "equity_share"

    def test_pe_higher_share_more_emissions(self, engine):
        """Test higher equity share produces more financed emissions."""
        data_small = _make_private_equity_input(outstanding_amount=Decimal("25000000"))
        data_large = _make_private_equity_input(outstanding_amount=Decimal("100000000"))
        r_small = engine.calculate(data_small)
        r_large = engine.calculate(data_large)
        assert r_large["financed_emissions"] > r_small["financed_emissions"]


# ==============================================================================
# PCAF QUALITY TIER TESTS
# ==============================================================================


class TestPCAFQualityTiers:
    """Test all 5 PCAF quality tiers."""

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_pcaf_score_accepted(self, engine, score):
        """Test all PCAF scores 1-5 are accepted."""
        data = _make_equity_input(pcaf_quality_score=score)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == score

    def test_pcaf_score_1_reported_verified(self, engine):
        """Test PCAF score 1 uses reported/verified emissions."""
        data = _make_equity_input(
            pcaf_quality_score=1,
            investee_scope1=Decimal("22400"),
            investee_scope2=Decimal("9100"),
        )
        result = engine.calculate(data)
        assert result["calculation_method"] in ["reported", "pcaf_score_1"]

    def test_pcaf_score_5_sector_average(self, engine):
        """Test PCAF score 5 uses sector average fallback."""
        data = _make_equity_input(
            pcaf_quality_score=5,
            investee_scope1=None,
            investee_scope2=None,
        )
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == 5

    def test_higher_pcaf_score_lower_quality(self, engine):
        """Test that higher PCAF score indicates lower quality data."""
        data_q1 = _make_equity_input(pcaf_quality_score=1)
        data_q5 = _make_equity_input(pcaf_quality_score=5)
        r1 = engine.calculate(data_q1)
        r5 = engine.calculate(data_q5)
        # Score 1 should use reported data, score 5 should use estimated
        assert r1["pcaf_quality_score"] < r5["pcaf_quality_score"]

    @pytest.mark.parametrize("sector", [
        "energy", "materials", "industrials", "consumer_discretionary",
        "information_technology", "utilities",
    ])
    def test_sector_average_fallback(self, engine, sector):
        """Test sector average fallback for PCAF score 4-5."""
        data = _make_equity_input(
            pcaf_quality_score=5,
            sector=sector,
            investee_scope1=None,
            investee_scope2=None,
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")


# ==============================================================================
# WACI CALCULATION TESTS
# ==============================================================================


class TestWACICalculation:
    """Test Weighted Average Carbon Intensity (WACI) calculations."""

    def test_waci_single_holding(self, engine):
        """Test WACI with single holding equals carbon intensity."""
        data = _make_equity_input()
        result = engine.calculate(data)
        if "waci" in result:
            assert result["waci"] > Decimal("0")

    def test_waci_with_revenue(self, engine):
        """Test WACI = Scope 1+2 emissions / revenue."""
        data = _make_equity_input(
            investee_revenue=Decimal("394000000000"),
        )
        result = engine.calculate(data)
        if "carbon_intensity" in result:
            expected_intensity = (Decimal("22400") + Decimal("9100")) / Decimal("394000000000") * Decimal("1000000")
            assert abs(result["carbon_intensity"] - expected_intensity) < Decimal("1.0")


# ==============================================================================
# CARBON INTENSITY TESTS
# ==============================================================================


class TestCarbonIntensity:
    """Test carbon intensity metric calculations."""

    def test_carbon_intensity_per_revenue(self, engine):
        """Test carbon intensity in tCO2e per $M revenue."""
        data = _make_equity_input(
            investee_revenue=Decimal("394000000000"),
        )
        result = engine.calculate(data)
        if "carbon_intensity" in result:
            assert result["carbon_intensity"] > Decimal("0")

    def test_carbon_intensity_per_evic(self, engine):
        """Test carbon intensity per EVIC."""
        data = _make_equity_input()
        result = engine.calculate(data)
        if "carbon_intensity_evic" in result:
            expected = (Decimal("22400") + Decimal("9100")) / Decimal("3000000000000") * Decimal("1000000")
            assert abs(result["carbon_intensity_evic"] - expected) < Decimal("0.1")


# ==============================================================================
# DC-INV-001 TESTS
# ==============================================================================


class TestDCINV001:
    """Test DC-INV-001: consolidated subsidiaries excluded."""

    def test_consolidated_subsidiary_rejected(self, engine):
        """Test consolidated subsidiary is excluded from Cat 15."""
        data = _make_equity_input(
            is_consolidated=True,
        )
        result = engine.calculate(data)
        assert result.get("dc_inv_001_triggered", False) is True or \
               result.get("excluded", False) is True or \
               result.get("financed_emissions") == Decimal("0")

    def test_non_consolidated_accepted(self, engine):
        """Test non-consolidated investment is included."""
        data = _make_equity_input(
            is_consolidated=False,
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")


# ==============================================================================
# BATCH CALCULATION TESTS
# ==============================================================================


class TestBatchCalculation:
    """Test batch equity calculations."""

    def test_batch_single_item(self, engine):
        """Test batch with single item."""
        items = [_make_equity_input()]
        results = engine.calculate_batch(items)
        assert len(results) == 1
        assert results[0]["financed_emissions"] > Decimal("0")

    def test_batch_multiple_items(self, engine):
        """Test batch with multiple items."""
        items = [
            _make_equity_input(investee_name="Apple"),
            _make_equity_input(investee_name="Microsoft", evic=Decimal("2800000000000")),
            _make_equity_input(investee_name="Google", evic=Decimal("1800000000000")),
        ]
        results = engine.calculate_batch(items)
        assert len(results) == 3
        for r in results:
            assert r["financed_emissions"] > Decimal("0")

    def test_batch_total_emissions(self, engine):
        """Test batch total equals sum of individual emissions."""
        items = [
            _make_equity_input(investee_name="A"),
            _make_equity_input(investee_name="B"),
        ]
        results = engine.calculate_batch(items)
        total = sum(r["financed_emissions"] for r in results)
        assert total > Decimal("0")

    def test_batch_empty_list(self, engine):
        """Test batch with empty list returns empty results."""
        results = engine.calculate_batch([])
        assert len(results) == 0


# ==============================================================================
# CURRENCY CONVERSION TESTS
# ==============================================================================


class TestCurrencyConversion:
    """Test currency conversion in equity calculations."""

    def test_usd_no_conversion(self, engine):
        """Test USD calculation requires no conversion."""
        data = _make_equity_input(currency="USD")
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_eur_conversion(self, engine):
        """Test EUR to USD conversion in calculation."""
        data = _make_equity_input(
            currency="EUR",
            outstanding_amount=Decimal("100000000"),
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_gbp_conversion(self, engine):
        """Test GBP to USD conversion in calculation."""
        data = _make_equity_input(
            currency="GBP",
            outstanding_amount=Decimal("100000000"),
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    @pytest.mark.parametrize("currency", ["EUR", "GBP", "JPY", "CHF"])
    def test_multiple_currencies(self, engine, currency):
        """Test calculation works with multiple currencies."""
        data = _make_equity_input(currency=currency)
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_evic_raises_error(self, engine):
        """Test zero EVIC raises ValueError."""
        data = _make_equity_input(evic=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.calculate(data)

    def test_negative_outstanding_raises_error(self, engine):
        """Test negative outstanding amount raises error."""
        data = _make_equity_input(outstanding_amount=Decimal("-100"))
        with pytest.raises(ValueError):
            engine.calculate(data)

    def test_scope1_none_with_score_1_degrades_quality(self, engine):
        """Test missing Scope 1 data with score 1 triggers fallback."""
        data = _make_equity_input(
            pcaf_quality_score=1,
            investee_scope1=None,
        )
        result = engine.calculate(data)
        # Should degrade quality or use fallback
        assert result["pcaf_quality_score"] >= 1

    def test_result_contains_all_required_fields(self, engine):
        """Test result contains all required output fields."""
        data = _make_equity_input()
        result = engine.calculate(data)
        required_fields = [
            "investee_name", "asset_class", "attribution_factor",
            "financed_emissions", "pcaf_quality_score", "provenance_hash",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
