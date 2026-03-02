# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 ETS Price Service

Tests ETS price service:
- get_current_price (valid, no data)
- get_weekly_price (specific dates)
- get_quarterly_average (volume-weighted)
- get_price_history (date range)
- set_price (manual entry)
- Price trend analysis
- Built-in reference data integrity

Target: 35+ tests
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from copy import deepcopy


# ---------------------------------------------------------------------------
# Inline ETS price service for self-contained tests
# ---------------------------------------------------------------------------

class ETSPriceError(Exception):
    pass


class ETSPriceService:
    """EU ETS carbon price service."""

    # Built-in reference prices (EUR/tCO2) -- illustrative data
    REFERENCE_PRICES = {
        "2025-Q4": Decimal("72.50"),
        "2026-Q1": Decimal("75.00"),
        "2026-Q2": Decimal("78.50"),
        "2026-Q3": Decimal("80.00"),
        "2026-Q4": Decimal("82.50"),
        "2027-Q1": Decimal("85.00"),
    }

    def __init__(self):
        self._daily_prices = {}
        self._weekly_prices = {}

    def get_current_price(self, ref_date=None):
        ref_date = ref_date or date.today()
        if ref_date in self._daily_prices:
            return self._daily_prices[ref_date]
        quarter_key = self._date_to_quarter_key(ref_date)
        if quarter_key in self.REFERENCE_PRICES:
            return self.REFERENCE_PRICES[quarter_key]
        raise ETSPriceError(f"No price data available for {ref_date}")

    def _date_to_quarter_key(self, d):
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"

    def get_weekly_price(self, week_start_date):
        if week_start_date in self._weekly_prices:
            return self._weekly_prices[week_start_date]
        return self.get_current_price(week_start_date)

    def get_quarterly_average(self, quarter, volume_weights=None):
        """Volume-weighted quarterly average."""
        quarter_key = quarter.replace("Q", "-Q")
        if quarter_key in self.REFERENCE_PRICES and not volume_weights:
            return self.REFERENCE_PRICES[quarter_key]

        if volume_weights:
            total_value = Decimal("0")
            total_volume = Decimal("0")
            for price, volume in volume_weights:
                total_value += Decimal(str(price)) * Decimal(str(volume))
                total_volume += Decimal(str(volume))
            if total_volume == 0:
                raise ETSPriceError("Total volume is zero")
            return (total_value / total_volume).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        if quarter_key not in self.REFERENCE_PRICES:
            raise ETSPriceError(f"No price data for quarter: {quarter}")
        return self.REFERENCE_PRICES[quarter_key]

    def get_price_history(self, start_date, end_date):
        history = []
        current = start_date
        while current <= end_date:
            try:
                price = self.get_current_price(current)
                history.append({"date": current.isoformat(), "price_eur": str(price)})
            except ETSPriceError:
                pass
            current += timedelta(days=1)
        return history

    def set_price(self, price_date, price_eur, source="manual"):
        if price_eur < 0:
            raise ETSPriceError("Price cannot be negative")
        self._daily_prices[price_date] = Decimal(str(price_eur))
        return {
            "date": price_date.isoformat(),
            "price_eur": str(Decimal(str(price_eur))),
            "source": source,
        }

    def get_price_trend(self, quarters):
        prices = []
        for q in quarters:
            try:
                price = self.get_quarterly_average(q)
                prices.append({"quarter": q, "price_eur": price})
            except ETSPriceError:
                continue
        if len(prices) < 2:
            return {"trend": "insufficient_data", "prices": prices}

        first = prices[0]["price_eur"]
        last = prices[-1]["price_eur"]
        change_pct = ((last - first) / first * 100).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        if change_pct > 0:
            trend = "increasing"
        elif change_pct < 0:
            trend = "decreasing"
        else:
            trend = "stable"
        return {
            "trend": trend,
            "change_pct": change_pct,
            "first_price": first,
            "last_price": last,
            "prices": prices,
        }


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def price_service():
    return ETSPriceService()


# ===========================================================================
# TEST CLASS -- get_current_price
# ===========================================================================

class TestGetCurrentPrice:
    """Tests for get_current_price."""

    def test_known_quarter_price(self, price_service):
        price = price_service.get_current_price(date(2026, 1, 15))
        assert price == Decimal("75.00")

    def test_different_quarter(self, price_service):
        price = price_service.get_current_price(date(2026, 5, 1))
        assert price == Decimal("78.50")

    def test_no_data_raises(self, price_service):
        with pytest.raises(ETSPriceError):
            price_service.get_current_price(date(2030, 1, 1))

    def test_manual_price_overrides(self, price_service):
        price_service.set_price(date(2026, 1, 15), 76.00)
        price = price_service.get_current_price(date(2026, 1, 15))
        assert price == Decimal("76.00")

    def test_q4_2025(self, price_service):
        price = price_service.get_current_price(date(2025, 11, 1))
        assert price == Decimal("72.50")


# ===========================================================================
# TEST CLASS -- get_weekly_price
# ===========================================================================

class TestGetWeeklyPrice:
    """Tests for get_weekly_price."""

    def test_falls_back_to_current(self, price_service):
        price = price_service.get_weekly_price(date(2026, 2, 3))
        assert price == Decimal("75.00")

    def test_manual_weekly_price(self, price_service):
        price_service._weekly_prices[date(2026, 2, 3)] = Decimal("76.50")
        price = price_service.get_weekly_price(date(2026, 2, 3))
        assert price == Decimal("76.50")


# ===========================================================================
# TEST CLASS -- get_quarterly_average
# ===========================================================================

class TestGetQuarterlyAverage:
    """Tests for get_quarterly_average."""

    def test_reference_quarter(self, price_service):
        avg = price_service.get_quarterly_average("2026Q1")
        assert avg == Decimal("75.00")

    def test_volume_weighted(self, price_service):
        weights = [(70, 100), (80, 200)]
        avg = price_service.get_quarterly_average("2026Q1", volume_weights=weights)
        expected = (70 * 100 + 80 * 200) / 300
        assert avg == Decimal(str(expected)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def test_zero_volume_raises(self, price_service):
        with pytest.raises(ETSPriceError, match="zero"):
            price_service.get_quarterly_average("2026Q1", volume_weights=[(70, 0)])

    def test_unknown_quarter_raises(self, price_service):
        with pytest.raises(ETSPriceError):
            price_service.get_quarterly_average("2030Q1")


# ===========================================================================
# TEST CLASS -- get_price_history
# ===========================================================================

class TestGetPriceHistory:
    """Tests for get_price_history."""

    def test_range_within_quarter(self, price_service):
        history = price_service.get_price_history(
            date(2026, 1, 1), date(2026, 1, 5)
        )
        assert len(history) == 5
        assert all(h["price_eur"] == "75.00" for h in history)

    def test_empty_range(self, price_service):
        history = price_service.get_price_history(
            date(2030, 1, 1), date(2030, 1, 5)
        )
        assert len(history) == 0

    def test_single_day(self, price_service):
        history = price_service.get_price_history(
            date(2026, 1, 1), date(2026, 1, 1)
        )
        assert len(history) == 1

    def test_cross_quarter(self, price_service):
        history = price_service.get_price_history(
            date(2026, 3, 30), date(2026, 4, 2)
        )
        prices = [Decimal(h["price_eur"]) for h in history]
        assert Decimal("75.00") in prices
        assert Decimal("78.50") in prices


# ===========================================================================
# TEST CLASS -- set_price
# ===========================================================================

class TestSetPrice:
    """Tests for set_price."""

    def test_set_valid_price(self, price_service):
        result = price_service.set_price(date(2026, 1, 20), 77.25)
        assert result["price_eur"] == "77.25"

    def test_negative_price_raises(self, price_service):
        with pytest.raises(ETSPriceError, match="negative"):
            price_service.set_price(date(2026, 1, 20), -10)

    def test_zero_price_accepted(self, price_service):
        result = price_service.set_price(date(2026, 1, 20), 0)
        assert result["price_eur"] == "0"

    def test_source_recorded(self, price_service):
        result = price_service.set_price(
            date(2026, 1, 20), 77.25, source="ICE ECX"
        )
        assert result["source"] == "ICE ECX"


# ===========================================================================
# TEST CLASS -- Price trend analysis
# ===========================================================================

class TestPriceTrend:
    """Tests for get_price_trend."""

    def test_increasing_trend(self, price_service):
        result = price_service.get_price_trend(["2026Q1", "2026Q2", "2026Q3"])
        assert result["trend"] == "increasing"
        assert result["change_pct"] > 0

    def test_insufficient_data(self, price_service):
        result = price_service.get_price_trend(["2026Q1"])
        assert result["trend"] == "insufficient_data"

    def test_full_year_trend(self, price_service):
        result = price_service.get_price_trend(
            ["2026Q1", "2026Q2", "2026Q3", "2026Q4"]
        )
        assert result["trend"] == "increasing"
        assert result["first_price"] == Decimal("75.00")
        assert result["last_price"] == Decimal("82.50")


# ===========================================================================
# TEST CLASS -- Reference data integrity
# ===========================================================================

class TestReferenceDataIntegrity:
    """Tests for built-in reference data."""

    def test_all_reference_prices_positive(self, price_service):
        for key, price in price_service.REFERENCE_PRICES.items():
            assert price > 0, f"Reference price for {key} must be positive"

    def test_reference_prices_reasonable_range(self, price_service):
        for key, price in price_service.REFERENCE_PRICES.items():
            assert Decimal("10") <= price <= Decimal("200"), \
                f"Price {price} for {key} outside reasonable range"

    def test_reference_quarters_format(self, price_service):
        import re
        for key in price_service.REFERENCE_PRICES.keys():
            assert re.match(r'^\d{4}-Q[1-4]$', key), \
                f"Invalid quarter format: {key}"
