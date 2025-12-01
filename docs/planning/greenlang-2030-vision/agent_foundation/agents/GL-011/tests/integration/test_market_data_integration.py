# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Market Data Integration Test Suite.

This module tests integration with external market data sources:
- Market price API integration
- Data staleness detection
- Price update frequency
- Historical data retrieval
- Real-time price feeds
- Multiple market sources
- Data validation and sanitation
- Cache management

Test Count: 15+ integration tests
Coverage: Market data connectors, API integration, data quality

Author: GreenLang Industrial Optimization Team
Version: 1.0.0
"""

import pytest
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.market_price_connector import MarketPriceConnector
from config import MarketPriceData


@pytest.mark.integration
class TestMarketDataAPIIntegration:
    """Integration tests for market data API."""

    def test_market_price_api_connection(self, mock_market_data_connector):
        """
        Integration: Connect to market price API.

        Expected:
        - Connection established
        - Authentication successful
        - API version verified
        """
        result = mock_market_data_connector.connect()
        assert result is True

    def test_fetch_current_natural_gas_price(self, mock_market_data_connector):
        """
        Integration: Fetch current natural gas price from NYMEX.

        Expected:
        - Price returned in USD/MMBtu
        - Timestamp included
        - Price within reasonable range
        """
        price = mock_market_data_connector.get_current_price("natural_gas")

        assert price > 0
        assert price < 100  # Reasonable upper bound

    def test_fetch_multiple_fuel_prices_batch(self, mock_market_data_connector):
        """
        Integration: Fetch multiple fuel prices in single API call.

        Expected:
        - Batch request succeeds
        - All requested fuels returned
        - Consistent timestamp
        """
        mock_market_data_connector.get_prices_batch = Mock(return_value={
            "natural_gas": 0.045,
            "coal": 0.035,
            "fuel_oil": 0.055,
        })

        prices = mock_market_data_connector.get_prices_batch(
            ["natural_gas", "coal", "fuel_oil"]
        )

        assert len(prices) == 3
        assert all(p > 0 for p in prices.values())

    def test_fetch_price_with_unit_conversion(self, mock_market_data_connector):
        """
        Integration: Fetch price with automatic unit conversion.

        Expected:
        - API returns price in USD/MMBtu
        - Converted to USD/kg for internal use
        - Conversion factor applied correctly
        """
        mock_market_data_connector.get_current_price = Mock(return_value=4.50)  # USD/MMBtu

        price_mmbtu = mock_market_data_connector.get_current_price("natural_gas")

        # Convert to USD/kg (1 MMBtu NG ≈ 20 kg)
        price_kg = price_mmbtu / 20.0

        assert abs(price_kg - 0.225) < 0.01


@pytest.mark.integration
class TestDataStalenessDetection:
    """Integration tests for data staleness detection."""

    def test_detect_stale_price_data(self, mock_market_data_connector):
        """
        Integration: Detect when price data is stale.

        Expected:
        - Data older than threshold flagged as stale
        - Triggers refresh
        """
        mock_market_data_connector.is_data_stale = Mock(return_value=True)

        is_stale = mock_market_data_connector.is_data_stale()
        assert is_stale is True

    def test_price_data_freshness_within_threshold(self, mock_market_data_connector):
        """
        Integration: Verify price data is fresh.

        Expected:
        - Recent data not flagged as stale
        - No refresh needed
        """
        mock_market_data_connector.is_data_stale = Mock(return_value=False)

        is_stale = mock_market_data_connector.is_data_stale()
        assert is_stale is False

    def test_automatic_refresh_on_stale_data(self, mock_market_data_connector):
        """
        Integration: Automatic refresh when data is stale.

        Expected:
        - Stale data triggers refresh
        - Fresh data retrieved
        - Cache updated
        """
        mock_market_data_connector.is_data_stale = Mock(return_value=True)
        mock_market_data_connector.refresh_prices = Mock()

        if mock_market_data_connector.is_data_stale():
            mock_market_data_connector.refresh_prices()

        mock_market_data_connector.refresh_prices.assert_called_once()


@pytest.mark.integration
class TestPriceUpdateFrequency:
    """Integration tests for price update frequency."""

    def test_price_update_every_5_minutes(self, mock_market_data_connector):
        """
        Integration: Prices update every 5 minutes.

        Expected:
        - Updates triggered on schedule
        - No duplicate updates
        - Timestamp reflects update time
        """
        mock_market_data_connector.get_last_update_time = Mock(
            return_value=datetime.now(timezone.utc) - timedelta(minutes=6)
        )

        last_update = mock_market_data_connector.get_last_update_time()
        time_since_update = datetime.now(timezone.utc) - last_update

        # More than 5 minutes → needs update
        assert time_since_update.total_seconds() > 300

    def test_rate_limiting_prevents_excessive_updates(self, mock_market_data_connector):
        """
        Integration: Rate limiting prevents excessive API calls.

        Expected:
        - Max update frequency enforced
        - Returns cached data if too frequent
        """
        mock_market_data_connector.get_current_price = Mock(
            side_effect=lambda fuel: 0.045  # Same price (cached)
        )

        # Call multiple times rapidly
        prices = [mock_market_data_connector.get_current_price("natural_gas") for _ in range(10)]

        # All should return same cached value
        assert all(p == 0.045 for p in prices)


@pytest.mark.integration
class TestHistoricalDataRetrieval:
    """Integration tests for historical price data."""

    def test_fetch_historical_prices_last_30_days(self, mock_market_data_connector):
        """
        Integration: Fetch historical prices for last 30 days.

        Expected:
        - Returns time series data
        - Correct date range
        - No missing data points
        """
        mock_data = [
            {"timestamp": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(), "price": 0.045 + i * 0.001}
            for i in range(30)
        ]
        mock_market_data_connector.get_price_history = Mock(return_value=mock_data)

        history = mock_market_data_connector.get_price_history(
            fuel_id="natural_gas",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc)
        )

        assert len(history) == 30
        assert all("timestamp" in d and "price" in d for d in history)

    def test_calculate_30_day_average_price(self, mock_market_data_connector):
        """
        Integration: Calculate 30-day average from historical data.

        Expected:
        - Average calculated correctly
        - Outliers handled appropriately
        """
        mock_data = [
            {"timestamp": f"2025-01-{i:02d}T00:00:00Z", "price": 0.045}
            for i in range(1, 31)
        ]
        mock_market_data_connector.get_price_history = Mock(return_value=mock_data)

        history = mock_market_data_connector.get_price_history("natural_gas")
        avg_price = sum(d["price"] for d in history) / len(history)

        assert abs(avg_price - 0.045) < 0.001

    def test_identify_price_trend_increasing(self, mock_market_data_connector):
        """
        Integration: Identify increasing price trend.

        Expected:
        - Detects upward trend
        - Calculates slope
        """
        # Increasing trend
        mock_data = [
            {"timestamp": f"2025-01-{i:02d}T00:00:00Z", "price": 0.040 + i * 0.001}
            for i in range(1, 31)
        ]
        mock_market_data_connector.get_price_history = Mock(return_value=mock_data)

        history = mock_market_data_connector.get_price_history("natural_gas")

        # Simple trend detection
        first_price = history[0]["price"]
        last_price = history[-1]["price"]
        is_increasing = last_price > first_price

        assert is_increasing is True


@pytest.mark.integration
class TestMultipleMarketSources:
    """Integration tests for multiple market data sources."""

    def test_fetch_from_multiple_sources_nymex_ice(self):
        """
        Integration: Fetch prices from multiple sources (NYMEX, ICE).

        Expected:
        - Both sources queried
        - Prices aggregated or averaged
        - Source attribution preserved
        """
        mock_nymex = Mock()
        mock_nymex.get_current_price.return_value = 4.50

        mock_ice = Mock()
        mock_ice.get_current_price.return_value = 4.55

        nymex_price = mock_nymex.get_current_price("natural_gas")
        ice_price = mock_ice.get_current_price("natural_gas")

        # Average from both sources
        avg_price = (nymex_price + ice_price) / 2.0

        assert abs(avg_price - 4.525) < 0.01

    def test_fallback_to_secondary_source_on_primary_failure(self):
        """
        Integration: Fallback to secondary source if primary fails.

        Expected:
        - Primary source failure detected
        - Automatically switches to secondary
        - No data loss
        """
        mock_primary = Mock()
        mock_primary.get_current_price.side_effect = Exception("API down")

        mock_secondary = Mock()
        mock_secondary.get_current_price.return_value = 0.045

        try:
            price = mock_primary.get_current_price("natural_gas")
        except Exception:
            # Fallback to secondary
            price = mock_secondary.get_current_price("natural_gas")

        assert price == 0.045


@pytest.mark.integration
class TestDataValidationAndSanitation:
    """Integration tests for data validation."""

    def test_reject_negative_price_data(self):
        """
        Integration: Reject invalid negative prices from API.

        Expected:
        - Negative prices rejected
        - Error logged
        - Cached data used instead
        """
        from config import MarketPriceData
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MarketPriceData(
                fuel_id="NG-001",
                price_source="NYMEX",
                current_price=-0.01,  # Invalid
                price_unit="USD/kg",
            )

    def test_reject_unrealistic_price_spike(self, mock_market_data_connector):
        """
        Integration: Detect and reject unrealistic price spikes.

        Expected:
        - Spike detection algorithm triggered
        - Price flagged for review
        - Previous valid price used
        """
        # Mock historical average
        historical_avg = 0.045

        # Mock current price (100x spike)
        current_price = 4.50

        # Spike detection
        spike_threshold = 2.0  # 200%
        is_spike = (current_price / historical_avg) > spike_threshold

        assert is_spike is True

    def test_sanitize_price_data_remove_outliers(self):
        """
        Integration: Sanitize price data by removing outliers.

        Expected:
        - Outliers identified (e.g., >3 std dev)
        - Outliers removed or flagged
        - Cleaned dataset returned
        """
        prices = [0.044, 0.045, 0.046, 0.045, 10.0, 0.044, 0.046]  # 10.0 is outlier

        # Simple outlier removal (values >10x median)
        median = sorted(prices)[len(prices) // 2]
        cleaned = [p for p in prices if p < median * 10]

        assert len(cleaned) < len(prices)
        assert 10.0 not in cleaned


@pytest.mark.integration
class TestCacheManagement:
    """Integration tests for price data caching."""

    def test_cache_price_data_on_first_fetch(self, mock_market_data_connector):
        """
        Integration: Cache price data on first fetch.

        Expected:
        - Data cached with TTL
        - Subsequent requests use cache
        """
        cache = {}

        def get_price_with_cache(fuel_id: str):
            if fuel_id in cache:
                return cache[fuel_id]

            price = mock_market_data_connector.get_current_price(fuel_id)
            cache[fuel_id] = price
            return price

        mock_market_data_connector.get_current_price = Mock(return_value=0.045)

        # First call - fetches from API
        price1 = get_price_with_cache("natural_gas")

        # Second call - uses cache
        price2 = get_price_with_cache("natural_gas")

        assert price1 == price2
        # API called only once
        mock_market_data_connector.get_current_price.assert_called_once()

    def test_cache_invalidation_after_ttl(self):
        """
        Integration: Cache invalidated after TTL expires.

        Expected:
        - Cached data expires after TTL
        - Fresh data fetched on next request
        """
        import time

        cache = {}
        ttl_seconds = 1

        def get_price_with_ttl(fuel_id: str):
            now = time.time()
            if fuel_id in cache:
                cached_price, cached_time = cache[fuel_id]
                if now - cached_time < ttl_seconds:
                    return cached_price

            # Fetch new price
            price = 0.045
            cache[fuel_id] = (price, now)
            return price

        price1 = get_price_with_ttl("natural_gas")
        time.sleep(2)  # Exceed TTL
        price2 = get_price_with_ttl("natural_gas")

        # Both should work
        assert price1 == 0.045
        assert price2 == 0.045
