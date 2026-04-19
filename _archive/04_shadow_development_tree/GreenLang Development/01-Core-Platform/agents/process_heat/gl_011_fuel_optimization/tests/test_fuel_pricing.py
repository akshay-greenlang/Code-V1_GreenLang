"""
GL-011 FUELCRAFT - Fuel Pricing Service Tests

Unit tests for FuelPricingService including price fetching,
caching, history tracking, and forecasting.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import (
    FuelPricingService,
    PriceQuote,
    PriceHistory,
    PriceForecast,
    PriceCache,
    CachedPrice,
    REGIONAL_BASIS_DIFFERENTIALS,
    EMISSION_FACTORS,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    FuelPricingConfig,
    PriceSource,
)


class TestPriceCache:
    """Tests for PriceCache class."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = PriceCache(default_ttl_seconds=300)

        assert cache._default_ttl == 300
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_set_and_get(self):
        """Test setting and getting cached values."""
        cache = PriceCache(default_ttl_seconds=300)

        quote = PriceQuote(
            fuel_type="natural_gas",
            commodity_price=3.00,
            total_price=3.50,
            source="henry_hub",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        cache.set("test_key", quote)
        result = cache.get("test_key")

        assert result is not None
        assert result.fuel_type == "natural_gas"
        assert result.total_price == 3.50

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = PriceCache()

        result = cache.get("nonexistent_key")

        assert result is None
        assert cache._misses == 1

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = PriceCache(default_ttl_seconds=1)

        quote = PriceQuote(
            fuel_type="test",
            commodity_price=3.00,
            total_price=3.50,
            source="test",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        cache.set("test_key", quote, ttl_seconds=0)

        # Entry should be expired immediately
        result = cache.get("test_key")
        assert result is None

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = PriceCache()

        quote = PriceQuote(
            fuel_type="test",
            commodity_price=3.00,
            total_price=3.50,
            source="test",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        cache.set("key1", quote)

        # 2 hits
        cache.get("key1")
        cache.get("key1")

        # 1 miss
        cache.get("key2")

        assert cache.hit_rate == pytest.approx(2/3, rel=0.01)

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = PriceCache()

        quote = PriceQuote(
            fuel_type="test",
            commodity_price=3.00,
            total_price=3.50,
            source="test",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        cache.set("key1", quote)
        cache.set("key2", quote)

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestFuelPricingService:
    """Tests for FuelPricingService class."""

    def test_service_initialization(self, fuel_pricing_config):
        """Test service initialization."""
        service = FuelPricingService(
            config=fuel_pricing_config,
            carbon_price_usd_ton=50.0,
        )

        assert service.config == fuel_pricing_config
        assert service.carbon_price_usd_ton == 50.0

    def test_get_current_price_natural_gas(self, fuel_pricing_service):
        """Test getting current natural gas price."""
        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert quote.fuel_type == "natural_gas"
        assert quote.total_price > 0
        assert quote.commodity_price > 0
        assert quote.unit == "USD/MMBTU"
        assert quote.timestamp is not None
        assert quote.valid_until > quote.timestamp

    def test_get_current_price_with_region(self, fuel_pricing_service):
        """Test price with regional basis differential."""
        quote_hh = fuel_pricing_service.get_current_price(
            "natural_gas",
            region="henry_hub",
        )

        quote_socal = fuel_pricing_service.get_current_price(
            "natural_gas",
            region="socal_citygate",
        )

        # SoCal should have positive basis differential
        assert quote_socal.basis_differential > quote_hh.basis_differential

    def test_get_current_price_with_carbon(self, fuel_pricing_service):
        """Test price includes carbon cost."""
        quote_with = fuel_pricing_service.get_current_price(
            "natural_gas",
            include_carbon=True,
        )

        quote_without = fuel_pricing_service.get_current_price(
            "natural_gas",
            include_carbon=False,
        )

        assert quote_with.carbon_cost > 0
        assert quote_without.carbon_cost == 0
        assert quote_with.total_price > quote_without.total_price

    def test_price_caching(self, fuel_pricing_service):
        """Test price caching behavior."""
        # First call - cache miss
        quote1 = fuel_pricing_service.get_current_price("natural_gas")

        # Second call - should be cached
        quote2 = fuel_pricing_service.get_current_price("natural_gas")

        # Prices should be identical (from cache)
        assert quote1.total_price == quote2.total_price
        assert quote1.timestamp == quote2.timestamp

    def test_get_all_current_prices(self, fuel_pricing_service):
        """Test getting prices for multiple fuels."""
        fuel_types = ["natural_gas", "no2_fuel_oil", "lpg_propane"]

        prices = fuel_pricing_service.get_all_current_prices(fuel_types)

        assert len(prices) == 3
        assert "natural_gas" in prices
        assert "no2_fuel_oil" in prices
        assert "lpg_propane" in prices

        for fuel, quote in prices.items():
            assert quote.total_price > 0

    def test_set_manual_price(self, fuel_pricing_service):
        """Test setting manual price."""
        fuel_pricing_service.set_manual_price("natural_gas", 5.00)

        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert quote.commodity_price == 5.00

    def test_set_contract_price(self, fuel_pricing_service):
        """Test setting contract price."""
        now = datetime.now(timezone.utc)
        fuel_pricing_service.set_contract_price(
            fuel_type="natural_gas",
            price=4.00,
            contract_id="CONTRACT-001",
            valid_from=now - timedelta(days=30),
            valid_until=now + timedelta(days=30),
        )

        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert quote.commodity_price == 4.00


class TestPriceHistory:
    """Tests for price history tracking."""

    def test_get_price_history_empty(self, fuel_pricing_service):
        """Test getting history with no data."""
        history = fuel_pricing_service.get_price_history("natural_gas", days=30)

        assert history.fuel_type == "natural_gas"
        assert len(history.prices) == 0
        assert history.period_days == 30

    def test_price_history_statistics(self, fuel_pricing_service):
        """Test price history statistics calculation."""
        # Generate some history by making multiple calls
        for i in range(5):
            fuel_pricing_service.get_current_price("natural_gas")

        history = fuel_pricing_service.get_price_history("natural_gas", days=30)

        # Should have statistics even with limited data
        assert history.avg_price >= 0
        assert history.min_price >= 0
        assert history.max_price >= history.min_price

    def test_price_history_trend_calculation(self, fuel_pricing_service):
        """Test trend direction calculation."""
        # Get some price data
        for i in range(3):
            fuel_pricing_service.get_current_price("natural_gas")

        history = fuel_pricing_service.get_price_history("natural_gas")

        assert history.trend_direction in ["up", "down", "stable"]


class TestPriceForecast:
    """Tests for price forecasting."""

    def test_get_price_forecast(self, fuel_pricing_service):
        """Test getting price forecast."""
        forecast = fuel_pricing_service.get_price_forecast(
            "natural_gas",
            horizon_days=30,
        )

        assert forecast.fuel_type == "natural_gas"
        assert forecast.horizon_days == 30
        assert forecast.current_price > 0
        assert len(forecast.forecasts) == 30

    def test_forecast_structure(self, fuel_pricing_service):
        """Test forecast data structure."""
        forecast = fuel_pricing_service.get_price_forecast("natural_gas", 7)

        for day_forecast in forecast.forecasts:
            date, low, mid, high = day_forecast
            assert isinstance(date, datetime)
            assert low <= mid <= high

    def test_forecast_confidence(self, fuel_pricing_service):
        """Test forecast confidence level."""
        forecast = fuel_pricing_service.get_price_forecast("natural_gas")

        assert 0.0 <= forecast.confidence_level <= 1.0


class TestBasisDifferentials:
    """Tests for regional basis differentials."""

    def test_henry_hub_zero_differential(self, fuel_pricing_service):
        """Test Henry Hub has zero basis differential."""
        quote = fuel_pricing_service.get_current_price(
            "natural_gas",
            region="henry_hub",
        )

        assert quote.basis_differential == 0.0

    def test_regional_differentials_exist(self):
        """Test regional differentials are defined."""
        assert "henry_hub" in REGIONAL_BASIS_DIFFERENTIALS
        assert "socal_citygate" in REGIONAL_BASIS_DIFFERENTIALS
        assert "algonquin_citygate" in REGIONAL_BASIS_DIFFERENTIALS

    def test_socal_positive_differential(self):
        """Test SoCal has positive differential."""
        assert REGIONAL_BASIS_DIFFERENTIALS["socal_citygate"] > 0

    def test_aeco_negative_differential(self):
        """Test AECO has negative differential."""
        assert REGIONAL_BASIS_DIFFERENTIALS["aeco_alberta"] < 0

    def test_non_gas_no_differential(self, fuel_pricing_service):
        """Test non-gas fuels have no basis differential."""
        quote = fuel_pricing_service.get_current_price(
            "no2_fuel_oil",
            region="socal_citygate",
        )

        assert quote.basis_differential == 0.0


class TestEmissionFactors:
    """Tests for emission factors."""

    def test_natural_gas_emission_factor(self):
        """Test natural gas emission factor."""
        assert EMISSION_FACTORS["natural_gas"] == pytest.approx(53.06, rel=0.01)

    def test_coal_higher_than_gas(self):
        """Test coal has higher emissions than gas."""
        assert EMISSION_FACTORS["coal_bituminous"] > EMISSION_FACTORS["natural_gas"]

    def test_zero_carbon_fuels(self):
        """Test zero-carbon fuels have zero emission factor."""
        assert EMISSION_FACTORS["biomass_wood"] == 0.0
        assert EMISSION_FACTORS["biogas"] == 0.0
        assert EMISSION_FACTORS["hydrogen"] == 0.0
        assert EMISSION_FACTORS["rng"] == 0.0


class TestCarbonCostCalculation:
    """Tests for carbon cost calculation."""

    def test_carbon_cost_calculation(self, fuel_pricing_service):
        """Test carbon cost is correctly calculated."""
        quote = fuel_pricing_service.get_current_price(
            "natural_gas",
            include_carbon=True,
        )

        # Carbon cost = emission_factor * carbon_price / 1000
        # Natural gas: 53.06 kg/MMBTU * $50/ton / 1000 = ~$2.65/MMBTU
        expected_carbon_cost = 53.06 * 50.0 / 1000.0
        assert quote.carbon_cost == pytest.approx(expected_carbon_cost, rel=0.01)

    def test_zero_carbon_fuel_no_cost(self, fuel_pricing_service):
        """Test zero-carbon fuel has no carbon cost."""
        # Manually set hydrogen price
        fuel_pricing_service.set_manual_price("hydrogen", 15.00)

        quote = fuel_pricing_service.get_current_price(
            "hydrogen",
            include_carbon=True,
        )

        assert quote.carbon_cost == 0.0


class TestFallbackPrices:
    """Tests for fallback price handling."""

    def test_unknown_fuel_uses_fallback(self, fuel_pricing_service):
        """Test unknown fuel uses fallback price."""
        quote = fuel_pricing_service.get_current_price("unknown_fuel_type")

        # Should return a fallback price
        assert quote.total_price > 0

    def test_fallback_prices_defined(self, fuel_pricing_service):
        """Test common fuels have fallback prices."""
        common_fuels = [
            "natural_gas", "no2_fuel_oil", "lpg_propane",
            "coal_bituminous", "hydrogen",
        ]

        for fuel in common_fuels:
            quote = fuel_pricing_service.get_current_price(fuel)
            assert quote.total_price > 0


class TestPriceQuoteFields:
    """Tests for PriceQuote field validation."""

    def test_quote_confidence_field(self, fuel_pricing_service):
        """Test quote confidence field."""
        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert 0.0 <= quote.confidence <= 1.0

    def test_quote_valid_until_field(self, fuel_pricing_service):
        """Test quote valid_until is in the future."""
        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert quote.valid_until > quote.timestamp

    def test_quote_region_field(self, fuel_pricing_service):
        """Test quote region field is set."""
        quote = fuel_pricing_service.get_current_price(
            "natural_gas",
            region="socal_citygate",
        )

        assert quote.region == "socal_citygate"


class TestTransportCosts:
    """Tests for transport cost calculation."""

    def test_transport_cost_included(self, fuel_pricing_service):
        """Test transport cost is included in total."""
        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert quote.transport_cost >= 0
        # Total should include transport
        assert quote.total_price >= quote.commodity_price + quote.transport_cost

    def test_different_fuels_different_transport(self, fuel_pricing_service):
        """Test different fuels have different transport costs."""
        gas_quote = fuel_pricing_service.get_current_price("natural_gas")
        oil_quote = fuel_pricing_service.get_current_price("no2_fuel_oil")

        # Fuel oil typically has higher transport cost
        assert oil_quote.transport_cost >= gas_quote.transport_cost


class TestTaxCalculation:
    """Tests for tax calculation."""

    def test_taxes_included(self, fuel_pricing_service):
        """Test taxes are included in total."""
        quote = fuel_pricing_service.get_current_price("natural_gas")

        assert quote.taxes >= 0

    def test_taxes_based_on_commodity(self, fuel_pricing_config):
        """Test taxes are based on commodity price."""
        service = FuelPricingService(fuel_pricing_config)

        quote = service.get_current_price("natural_gas")

        # Taxes should be ~5% of commodity price
        expected_taxes = quote.commodity_price * 0.05
        assert quote.taxes == pytest.approx(expected_taxes, rel=0.1)
