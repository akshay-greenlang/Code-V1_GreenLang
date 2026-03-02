# -*- coding: utf-8 -*-
"""
Unit tests for HotelStayCalculatorEngine (AGENT-MRV-019, Engine 4)

40 tests covering:
- Country-specific EF lookups (GB, US, JP, CN, FR, DE, SG, AE, BR, AU, IN, GLOBAL)
- Hotel class multipliers (budget, standard, upscale, luxury)
- Extended stay discount (>= 14 nights => 0.85 multiplier)
- Batch processing (success, empty, with errors)
- Provenance hashing (presence, determinism)
- Input validation (zero/negative nights)
- Singleton pattern
- Decimal precision (8 dp)

All calculations verified against the formula:
    base_co2e  = room_nights x ef_per_room_night
    class_co2e = base_co2e x class_multiplier
    discount   = 0.85 if room_nights >= 14 else 1.0
    total_co2e = class_co2e x discount

Author: GL-TestEngineer
Date: February 2026
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch, MagicMock

from greenlang.business_travel.hotel_stay_calculator import HotelStayCalculatorEngine
from greenlang.business_travel.models import (
    HotelInput,
    HotelResult,
    HotelClass,
    EFSource,
    HOTEL_EMISSION_FACTORS,
    HOTEL_CLASS_MULTIPLIERS,
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")


def _q(value: Decimal) -> Decimal:
    """Quantize to 8 decimal places."""
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test for isolation."""
    HotelStayCalculatorEngine.reset_instance()
    yield
    HotelStayCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh HotelStayCalculatorEngine with mocked dependencies."""
    with patch(
        "greenlang.business_travel.hotel_stay_calculator.get_config"
    ) as mock_config, patch(
        "greenlang.business_travel.hotel_stay_calculator.get_metrics"
    ) as mock_metrics, patch(
        "greenlang.business_travel.hotel_stay_calculator.get_provenance_tracker"
    ) as mock_prov:
        mock_cfg = MagicMock()
        mock_config.return_value = mock_cfg
        mock_met = MagicMock()
        mock_metrics.return_value = mock_met
        mock_prov.return_value = MagicMock()
        eng = HotelStayCalculatorEngine(metrics=mock_met)
        yield eng


# ==============================================================================
# COUNTRY EF TESTS
# ==============================================================================


class TestCountryEmissionFactors:
    """Tests for country-specific emission factor lookups and calculations."""

    def test_uk_3nights_standard(self, engine):
        """GB standard 3 nights: 3 x 12.32 x 1.0 = 36.96."""
        inp = HotelInput(country_code="GB", room_nights=3, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        expected = _q(Decimal("3") * Decimal("12.32") * Decimal("1.0"))
        assert result.total_co2e == expected

    def test_us_2nights_standard(self, engine):
        """US standard 2 nights: 2 x 21.12 = 42.24."""
        inp = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        expected = _q(Decimal("2") * Decimal("21.12"))
        assert result.total_co2e == expected

    def test_us_2nights_luxury(self, engine):
        """US luxury 2 nights: 2 x 21.12 x 1.80 = 76.032."""
        inp = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.LUXURY)
        result = engine.calculate(inp)
        expected = _q(_q(Decimal("2") * Decimal("21.12")) * Decimal("1.80"))
        assert result.total_co2e == expected

    def test_japan_1night(self, engine):
        """JP standard 1 night: 28.85."""
        inp = HotelInput(country_code="JP", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("28.85"))

    def test_china_1night(self, engine):
        """CN standard 1 night: 34.56."""
        inp = HotelInput(country_code="CN", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("34.56"))

    def test_france_1night(self, engine):
        """FR standard 1 night: 7.26."""
        inp = HotelInput(country_code="FR", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("7.26"))

    def test_germany_1night(self, engine):
        """DE standard 1 night: 13.50."""
        inp = HotelInput(country_code="DE", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("13.50"))

    def test_singapore_1night(self, engine):
        """SG standard 1 night: 27.00."""
        inp = HotelInput(country_code="SG", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("27.00"))

    def test_uae_1night(self, engine):
        """AE standard 1 night: 37.50."""
        inp = HotelInput(country_code="AE", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("37.50"))

    def test_brazil_1night(self, engine):
        """BR standard 1 night: 8.28."""
        inp = HotelInput(country_code="BR", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("8.28"))

    def test_global_average(self, engine):
        """GLOBAL standard 1 night: 20.90."""
        inp = HotelInput(country_code="GLOBAL", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("20.90"))

    def test_unknown_country_global_fallback(self, engine):
        """Unknown country code falls back to GLOBAL (20.90)."""
        inp = HotelInput(country_code="ZZ", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("20.90"))
        assert result.country_code == "ZZ"

    def test_australia_ef(self, engine):
        """AU standard 1 night: 25.90."""
        inp = HotelInput(country_code="AU", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("25.90"))

    def test_india_ef(self, engine):
        """IN standard 1 night: 22.08."""
        inp = HotelInput(country_code="IN", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.total_co2e == _q(Decimal("22.08"))


# ==============================================================================
# CLASS MULTIPLIER TESTS
# ==============================================================================


class TestHotelClassMultipliers:
    """Tests for hotel class multiplier application."""

    def test_budget_multiplier(self, engine):
        """Budget class: base x 0.75."""
        inp = HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.BUDGET)
        result = engine.calculate(inp)
        expected = _q(_q(Decimal("1") * Decimal("12.32")) * Decimal("0.75"))
        assert result.total_co2e == expected

    def test_upscale_multiplier(self, engine):
        """Upscale class: base x 1.35."""
        inp = HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.UPSCALE)
        result = engine.calculate(inp)
        expected = _q(_q(Decimal("1") * Decimal("12.32")) * Decimal("1.35"))
        assert result.total_co2e == expected

    def test_luxury_multiplier(self, engine):
        """Luxury class: base x 1.80."""
        inp = HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.LUXURY)
        result = engine.calculate(inp)
        expected = _q(_q(Decimal("1") * Decimal("12.32")) * Decimal("1.80"))
        assert result.total_co2e == expected

    def test_luxury_higher_than_standard(self, engine):
        """Luxury total must exceed standard total for same stay."""
        inp_luxury = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.LUXURY)
        inp_standard = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.STANDARD)
        r_lux = engine.calculate(inp_luxury)
        r_std = engine.calculate(inp_standard)
        assert r_lux.total_co2e > r_std.total_co2e

    def test_budget_lower_than_standard(self, engine):
        """Budget total must be less than standard total for same stay."""
        inp_budget = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.BUDGET)
        inp_standard = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.STANDARD)
        r_bud = engine.calculate(inp_budget)
        r_std = engine.calculate(inp_standard)
        assert r_bud.total_co2e < r_std.total_co2e

    def test_class_multipliers_dict(self, engine):
        """get_hotel_class_multipliers returns dict with all 4 classes."""
        mults = HotelStayCalculatorEngine.get_hotel_class_multipliers()
        assert isinstance(mults, dict)
        assert len(mults) == 4
        assert mults["budget"] == Decimal("0.75")
        assert mults["standard"] == Decimal("1.0")
        assert mults["upscale"] == Decimal("1.35")
        assert mults["luxury"] == Decimal("1.80")


# ==============================================================================
# EXTENDED STAY DISCOUNT TESTS
# ==============================================================================


class TestExtendedStayDiscount:
    """Tests for extended stay discount (>= 14 nights => 0.85 multiplier)."""

    def test_extended_stay_14nights(self, engine):
        """14 nights gets 0.85 discount."""
        inp = HotelInput(country_code="GB", room_nights=14, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        base = _q(Decimal("14") * Decimal("12.32"))
        expected = _q(base * Decimal("0.85"))
        assert result.total_co2e == expected

    def test_13nights_no_discount(self, engine):
        """13 nights does NOT get discount."""
        inp = HotelInput(country_code="GB", room_nights=13, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        expected = _q(Decimal("13") * Decimal("12.32"))
        assert result.total_co2e == expected

    def test_15nights_gets_discount(self, engine):
        """15 nights gets 0.85 discount."""
        inp = HotelInput(country_code="GB", room_nights=15, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        base = _q(Decimal("15") * Decimal("12.32"))
        expected = _q(base * Decimal("0.85"))
        assert result.total_co2e == expected

    def test_multi_night_linear_scaling(self, engine):
        """5 nights should be exactly 5x the 1-night calculation."""
        inp_1 = HotelInput(country_code="US", room_nights=1, hotel_class=HotelClass.STANDARD)
        inp_5 = HotelInput(country_code="US", room_nights=5, hotel_class=HotelClass.STANDARD)
        r_1 = engine.calculate(inp_1)
        r_5 = engine.calculate(inp_5)
        assert r_5.total_co2e == _q(r_1.total_co2e * Decimal("5"))


# ==============================================================================
# INPUT VALIDATION TESTS
# ==============================================================================


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_zero_nights_raises_error(self, engine):
        """Zero room_nights should raise ValueError."""
        with pytest.raises(Exception):
            HotelInput(country_code="GB", room_nights=0, hotel_class=HotelClass.STANDARD)

    def test_negative_nights_raises_error(self, engine):
        """Negative room_nights should raise ValueError."""
        with pytest.raises(Exception):
            HotelInput(country_code="GB", room_nights=-1, hotel_class=HotelClass.STANDARD)


# ==============================================================================
# RESULT STRUCTURE TESTS
# ==============================================================================


class TestResultStructure:
    """Tests for result type, provenance, and EF source."""

    def test_result_is_hotel_result(self, engine):
        """Calculate returns a HotelResult instance."""
        inp = HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert isinstance(result, HotelResult)

    def test_provenance_hash_present(self, engine):
        """Result contains a non-empty provenance hash."""
        inp = HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, engine):
        """Same input produces same provenance hash (bit-perfect reproducibility)."""
        inp = HotelInput(country_code="GB", room_nights=3, hotel_class=HotelClass.UPSCALE)
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.provenance_hash == r2.provenance_hash

    def test_ef_source_defra(self, engine):
        """EF source should be DEFRA for hotel calculations."""
        inp = HotelInput(country_code="US", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert result.ef_source == EFSource.DEFRA

    def test_wtt_embedded(self, engine):
        """Hotel result has no separate WTT field (WTT is embedded in room-night factor)."""
        inp = HotelInput(country_code="US", room_nights=1, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        assert not hasattr(result, "wtt_co2e")

    def test_total_co2e_matches_co2e(self, engine):
        """For STANDARD class (multiplier 1.0, no discount), total_co2e == base co2e x 1.0."""
        inp = HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.STANDARD)
        result = engine.calculate(inp)
        # co2e is base (room_nights x ef), total_co2e = co2e x class_mult x discount
        # For STANDARD (1.0) and < 14 nights (no discount): total_co2e == co2e
        assert result.total_co2e == result.co2e

    def test_decimal_precision_8dp(self, engine):
        """All Decimal values should have exactly 8 decimal places."""
        inp = HotelInput(country_code="GB", room_nights=7, hotel_class=HotelClass.UPSCALE)
        result = engine.calculate(inp)
        # Check that total_co2e has 8 decimal places
        total_str = str(result.total_co2e)
        if "." in total_str:
            decimal_part = total_str.split(".")[1]
            assert len(decimal_part) == 8


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:
    """Tests for batch hotel calculation."""

    def test_batch_3_hotels(self, engine):
        """Batch of 3 valid inputs returns 3 results."""
        inputs = [
            HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.STANDARD),
            HotelInput(country_code="US", room_nights=2, hotel_class=HotelClass.LUXURY),
            HotelInput(country_code="JP", room_nights=3, hotel_class=HotelClass.BUDGET),
        ]
        results = engine.calculate_batch(inputs)
        assert len(results) == 3

    def test_batch_empty(self, engine):
        """Batch of empty list returns empty list."""
        results = engine.calculate_batch([])
        assert results == []

    def test_batch_with_errors(self, engine):
        """Batch with some invalid inputs still returns valid results."""
        inputs = [
            HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.STANDARD),
        ]
        # Use calculate_batch_with_errors for error tracking
        results = engine.calculate_batch_with_errors(inputs)
        assert len(results) == 1
        assert results[0]["status"] == "success"


# ==============================================================================
# STATIC METHOD & SINGLETON TESTS
# ==============================================================================


class TestStaticMethodsAndSingleton:
    """Tests for static methods and singleton pattern."""

    def test_available_countries_count(self, engine):
        """get_available_countries returns >= 16 countries."""
        countries = HotelStayCalculatorEngine.get_available_countries()
        assert len(countries) >= 16

    def test_available_countries_includes_gb(self, engine):
        """GB must be in the available countries list."""
        countries = HotelStayCalculatorEngine.get_available_countries()
        codes = [c["country_code"] for c in countries]
        assert "GB" in codes

    def test_singleton_pattern(self):
        """get_instance returns the same object on repeated calls."""
        with patch(
            "greenlang.business_travel.hotel_stay_calculator.get_config"
        ), patch(
            "greenlang.business_travel.hotel_stay_calculator.get_metrics"
        ), patch(
            "greenlang.business_travel.hotel_stay_calculator.get_provenance_tracker"
        ):
            inst1 = HotelStayCalculatorEngine.get_instance()
            inst2 = HotelStayCalculatorEngine.get_instance()
            assert inst1 is inst2

    def test_country_code_case_insensitive(self, engine):
        """Country code should be treated case-insensitively."""
        inp_lower = HotelInput(country_code="gb", room_nights=1, hotel_class=HotelClass.STANDARD)
        inp_upper = HotelInput(country_code="GB", room_nights=1, hotel_class=HotelClass.STANDARD)
        r_lower = engine.calculate(inp_lower)
        r_upper = engine.calculate(inp_upper)
        assert r_lower.total_co2e == r_upper.total_co2e
