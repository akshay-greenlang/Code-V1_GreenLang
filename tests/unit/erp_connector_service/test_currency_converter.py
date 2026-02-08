# -*- coding: utf-8 -*-
"""
Unit Tests for CurrencyConverter (AGENT-DATA-003)

Tests single and batch currency conversion, rate management, default rates,
supported currencies, round-trip conversion, and unknown currency handling.

Coverage target: 85%+ of currency conversion logic

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline CurrencyConverter
# ---------------------------------------------------------------------------

DEFAULT_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CHF": 1.13,
    "CAD": 0.74,
    "AUD": 0.65,
    "CNY": 0.14,
    "INR": 0.012,
    "BRL": 0.20,
}


class CurrencyConverter:
    """Converts amounts between currencies using configurable exchange rates."""

    def __init__(self, rates: Optional[Dict[str, float]] = None):
        self._rates = dict(rates or DEFAULT_RATES_TO_USD)

    def convert(self, amount: float, from_currency: str, to_currency: str = "USD") -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if from_currency == to_currency:
            return round(amount, 2)
        from_rate = self._rates.get(from_currency)
        to_rate = self._rates.get(to_currency)
        if from_rate is None:
            raise ValueError(f"Unsupported currency: {from_currency}")
        if to_rate is None:
            raise ValueError(f"Unsupported currency: {to_currency}")
        usd_amount = amount * from_rate
        result = usd_amount / to_rate
        return round(result, 2)

    def convert_records(self, records: List[Dict[str, Any]],
                        to_currency: str = "USD") -> List[Dict[str, Any]]:
        results = []
        for r in records:
            amount = r.get("amount", 0.0)
            from_currency = r.get("currency", "USD")
            converted = self.convert(amount, from_currency, to_currency)
            results.append({**r, "amount_converted": converted, "converted_currency": to_currency})
        return results

    def set_rate(self, currency: str, rate_to_usd: float):
        self._rates[currency.upper()] = rate_to_usd

    def get_rate(self, currency: str) -> Optional[float]:
        return self._rates.get(currency.upper())

    def list_rates(self) -> Dict[str, float]:
        return dict(self._rates)

    def supported_currencies(self) -> List[str]:
        return sorted(self._rates.keys())

    def remove_rate(self, currency: str) -> bool:
        currency = currency.upper()
        if currency in self._rates and currency != "USD":
            del self._rates[currency]
            return True
        return False


# ===========================================================================
# Test Classes
# ===========================================================================


class TestConvertSingle:
    def test_eur_to_usd(self):
        converter = CurrencyConverter()
        result = converter.convert(1000.0, "EUR", "USD")
        assert result == pytest.approx(1080.0)

    def test_gbp_to_usd(self):
        converter = CurrencyConverter()
        result = converter.convert(1000.0, "GBP", "USD")
        assert result == pytest.approx(1270.0)

    def test_usd_to_usd(self):
        converter = CurrencyConverter()
        result = converter.convert(1000.0, "USD", "USD")
        assert result == 1000.0

    def test_eur_to_gbp(self):
        converter = CurrencyConverter()
        result = converter.convert(1000.0, "EUR", "GBP")
        expected = (1000.0 * 1.08) / 1.27
        assert result == pytest.approx(expected, rel=0.01)

    def test_jpy_to_usd(self):
        converter = CurrencyConverter()
        result = converter.convert(100000.0, "JPY", "USD")
        assert result == pytest.approx(670.0)

    def test_case_insensitive(self):
        converter = CurrencyConverter()
        result1 = converter.convert(1000.0, "eur", "usd")
        result2 = converter.convert(1000.0, "EUR", "USD")
        assert result1 == result2

    def test_zero_amount(self):
        converter = CurrencyConverter()
        result = converter.convert(0.0, "EUR", "USD")
        assert result == 0.0


class TestConvertRecords:
    def test_convert_batch(self):
        converter = CurrencyConverter()
        records = [
            {"record_id": "SPD-001", "amount": 125000.0, "currency": "EUR"},
            {"record_id": "SPD-002", "amount": 78500.0, "currency": "USD"},
            {"record_id": "SPD-003", "amount": 12300.0, "currency": "GBP"},
        ]
        results = converter.convert_records(records, "USD")
        assert len(results) == 3
        for r in results:
            assert "amount_converted" in r
            assert r["converted_currency"] == "USD"

    def test_convert_batch_preserves_fields(self):
        converter = CurrencyConverter()
        records = [{"record_id": "SPD-001", "amount": 1000.0, "currency": "EUR"}]
        results = converter.convert_records(records, "USD")
        assert results[0]["record_id"] == "SPD-001"

    def test_convert_batch_to_eur(self):
        converter = CurrencyConverter()
        records = [{"amount": 1000.0, "currency": "USD"}]
        results = converter.convert_records(records, "EUR")
        expected = 1000.0 / 1.08
        assert results[0]["amount_converted"] == pytest.approx(expected, rel=0.01)


class TestSetRate:
    def test_set_new_rate(self):
        converter = CurrencyConverter()
        converter.set_rate("KRW", 0.00075)
        result = converter.convert(1000000.0, "KRW", "USD")
        assert result == pytest.approx(750.0)

    def test_override_existing_rate(self):
        converter = CurrencyConverter()
        converter.set_rate("EUR", 1.10)
        result = converter.convert(1000.0, "EUR", "USD")
        assert result == pytest.approx(1100.0)


class TestGetRate:
    def test_get_existing_rate(self):
        converter = CurrencyConverter()
        rate = converter.get_rate("EUR")
        assert rate == 1.08

    def test_get_nonexistent_rate(self):
        converter = CurrencyConverter()
        rate = converter.get_rate("ZZZ")
        assert rate is None

    def test_get_rate_case_insensitive(self):
        converter = CurrencyConverter()
        rate = converter.get_rate("eur")
        assert rate == 1.08


class TestListRates:
    def test_list_all_rates(self):
        converter = CurrencyConverter()
        rates = converter.list_rates()
        assert "USD" in rates
        assert "EUR" in rates
        assert len(rates) == 10


class TestSupportedCurrencies:
    def test_supported_currencies(self):
        converter = CurrencyConverter()
        currencies = converter.supported_currencies()
        assert "USD" in currencies
        assert "EUR" in currencies
        assert currencies == sorted(currencies)

    def test_supported_count(self):
        converter = CurrencyConverter()
        assert len(converter.supported_currencies()) == 10


class TestRoundTripConversion:
    def test_round_trip_eur_usd(self):
        converter = CurrencyConverter()
        amount = 1000.0
        usd = converter.convert(amount, "EUR", "USD")
        back = converter.convert(usd, "USD", "EUR")
        assert back == pytest.approx(amount, rel=0.01)

    def test_round_trip_gbp_jpy(self):
        converter = CurrencyConverter()
        amount = 5000.0
        jpy = converter.convert(amount, "GBP", "JPY")
        back = converter.convert(jpy, "JPY", "GBP")
        assert back == pytest.approx(amount, rel=0.02)


class TestUnknownCurrency:
    def test_unknown_from_currency_raises(self):
        converter = CurrencyConverter()
        with pytest.raises(ValueError, match="Unsupported currency"):
            converter.convert(1000.0, "ZZZ", "USD")

    def test_unknown_to_currency_raises(self):
        converter = CurrencyConverter()
        with pytest.raises(ValueError, match="Unsupported currency"):
            converter.convert(1000.0, "USD", "ZZZ")


class TestRemoveRate:
    def test_remove_existing(self):
        converter = CurrencyConverter()
        result = converter.remove_rate("BRL")
        assert result is True
        assert converter.get_rate("BRL") is None

    def test_remove_usd_not_allowed(self):
        converter = CurrencyConverter()
        result = converter.remove_rate("USD")
        assert result is False

    def test_remove_nonexistent(self):
        converter = CurrencyConverter()
        result = converter.remove_rate("ZZZ")
        assert result is False
