"""
GL-011 FUELCRAFT - Fuel Pricing Service

This module provides real-time fuel price integration for the FuelOptimizationAgent.
Supports multiple price sources including Henry Hub, Brent, WTI, and regional markets.

Features:
    - Real-time price fetching from commodity APIs
    - Price history and trend analysis
    - Regional basis differential adjustments
    - Transport and tax cost calculations
    - Price forecasting for optimization
    - Caching for performance (configurable TTL)

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import (
    ...     FuelPricingService,
    ...     PriceQuote,
    ... )
    >>>
    >>> service = FuelPricingService(config)
    >>> quote = service.get_current_price("natural_gas")
    >>> print(f"Price: ${quote.total_price:.2f}/MMBTU")
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import lru_cache
import asyncio
import hashlib
import logging
import os

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    FuelPricingConfig,
    PriceSource,
    FuelType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Reference Prices and Factors
# =============================================================================

# Default basis differentials ($/MMBTU) from Henry Hub
REGIONAL_BASIS_DIFFERENTIALS = {
    "henry_hub": 0.00,
    "chicago_citygate": -0.15,
    "socal_citygate": 0.35,
    "pge_citygate": 0.40,
    "algonquin_citygate": 0.75,
    "transco_z6_ny": 0.60,
    "nymex_hub": 0.05,
    "dawn_ontario": -0.25,
    "aeco_alberta": -0.50,
    "waha_texas": -0.35,
}

# Emission factors for carbon cost calculation (kg CO2/MMBTU)
EMISSION_FACTORS = {
    "natural_gas": 53.06,
    "no2_fuel_oil": 73.16,
    "no6_fuel_oil": 75.10,
    "lpg_propane": 62.87,
    "lpg_butane": 64.77,
    "coal_bituminous": 93.28,
    "coal_sub_bituminous": 97.17,
    "biomass_wood": 0.0,  # Biogenic
    "biogas": 0.0,  # Biogenic
    "hydrogen": 0.0,  # Green H2
    "rng": 0.0,  # Renewable natural gas
}

# Unit conversion factors
UNIT_CONVERSIONS = {
    "usd_per_mmbtu_to_usd_per_therm": 0.1,
    "usd_per_mmbtu_to_usd_per_mcf": 1.028,
    "usd_per_bbl_to_usd_per_mmbtu_oil": 0.172,  # ~5.8 MMBTU/bbl
    "usd_per_ton_to_usd_per_mmbtu_coal": 0.04,  # ~25 MMBTU/ton
}


# =============================================================================
# DATA MODELS
# =============================================================================

class PriceQuote(BaseModel):
    """Current fuel price quote."""

    fuel_type: str = Field(..., description="Fuel type identifier")
    commodity_price: float = Field(..., ge=0, description="Base commodity price")
    basis_differential: float = Field(default=0.0, description="Regional basis")
    transport_cost: float = Field(default=0.0, ge=0, description="Transport cost")
    taxes: float = Field(default=0.0, ge=0, description="Applicable taxes")
    carbon_cost: float = Field(default=0.0, ge=0, description="Carbon cost")
    total_price: float = Field(..., ge=0, description="Total delivered price")

    unit: str = Field(default="USD/MMBTU", description="Price unit")
    source: str = Field(..., description="Price source")
    region: str = Field(default="", description="Pricing region")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Quote timestamp"
    )
    valid_until: datetime = Field(
        ...,
        description="Quote validity end"
    )

    # Quality indicators
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Price confidence"
    )
    is_stale: bool = Field(default=False, description="Price data is stale")
    fallback_used: bool = Field(
        default=False,
        description="Fallback price source used"
    )


class PriceHistory(BaseModel):
    """Historical price data."""

    fuel_type: str = Field(..., description="Fuel type identifier")
    prices: List[Tuple[datetime, float]] = Field(
        ...,
        description="List of (timestamp, price) tuples"
    )
    unit: str = Field(default="USD/MMBTU", description="Price unit")
    source: str = Field(..., description="Price source")

    # Statistics
    avg_price: float = Field(..., description="Average price")
    min_price: float = Field(..., description="Minimum price")
    max_price: float = Field(..., description="Maximum price")
    std_dev: float = Field(..., description="Standard deviation")
    volatility_pct: float = Field(..., description="Price volatility (%)")

    # Trend
    trend_direction: str = Field(
        default="stable",
        description="Trend direction (up, down, stable)"
    )
    trend_slope: float = Field(default=0.0, description="Trend slope ($/day)")
    period_days: int = Field(..., description="History period in days")


class PriceForecast(BaseModel):
    """Price forecast data."""

    fuel_type: str = Field(..., description="Fuel type identifier")
    current_price: float = Field(..., description="Current price")
    forecasts: List[Tuple[datetime, float, float, float]] = Field(
        ...,
        description="List of (date, low, mid, high) forecasts"
    )
    unit: str = Field(default="USD/MMBTU", description="Price unit")
    horizon_days: int = Field(..., description="Forecast horizon")

    # Confidence
    model_type: str = Field(
        default="trend_extrapolation",
        description="Forecast model used"
    )
    confidence_level: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Forecast confidence"
    )

    # Key dates
    expected_peak_date: Optional[datetime] = Field(
        default=None,
        description="Expected price peak date"
    )
    expected_trough_date: Optional[datetime] = Field(
        default=None,
        description="Expected price trough date"
    )


# =============================================================================
# PRICE CACHE
# =============================================================================

@dataclass
class CachedPrice:
    """Cached price entry."""
    quote: PriceQuote
    cached_at: datetime
    ttl_seconds: int

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now(timezone.utc) - self.cached_at).total_seconds()
        return age > self.ttl_seconds


class PriceCache:
    """Thread-safe price cache with TTL."""

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        """Initialize cache."""
        self._cache: Dict[str, CachedPrice] = {}
        self._default_ttl = default_ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[PriceQuote]:
        """Get cached price if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired:
                self._hits += 1
                return entry.quote
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def set(
        self,
        key: str,
        quote: PriceQuote,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set cached price."""
        self._cache[key] = CachedPrice(
            quote=quote,
            cached_at=datetime.now(timezone.utc),
            ttl_seconds=ttl_seconds or self._default_ttl,
        )

    def clear(self) -> None:
        """Clear all cached prices."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# =============================================================================
# FUEL PRICING SERVICE
# =============================================================================

class FuelPricingService:
    """
    Fuel pricing service with real-time price integration.

    This service provides current and historical fuel prices from
    multiple sources including commodity exchanges and regional markets.

    Features:
        - Real-time price fetching with caching
        - Regional basis differential adjustments
        - Transport and tax cost calculations
        - Carbon cost integration
        - Price history and forecasting

    Example:
        >>> service = FuelPricingService(config)
        >>> quote = service.get_current_price("natural_gas", region="socal")
        >>> print(f"Total price: ${quote.total_price:.2f}/MMBTU")
    """

    def __init__(
        self,
        config: FuelPricingConfig,
        carbon_price_usd_ton: float = 50.0,
    ) -> None:
        """
        Initialize the fuel pricing service.

        Args:
            config: Pricing configuration
            carbon_price_usd_ton: Carbon price for emissions cost calculation
        """
        self.config = config
        self.carbon_price_usd_ton = carbon_price_usd_ton
        self._cache = PriceCache(
            default_ttl_seconds=config.update_interval_minutes * 60
        )

        # Price fetchers by source
        self._fetchers: Dict[PriceSource, Callable] = {
            PriceSource.HENRY_HUB: self._fetch_henry_hub,
            PriceSource.BRENT: self._fetch_brent,
            PriceSource.WTI: self._fetch_wti,
            PriceSource.REGIONAL_HUB: self._fetch_regional,
            PriceSource.CONTRACT: self._fetch_contract,
            PriceSource.MANUAL: self._fetch_manual,
        }

        # Manual prices (set via API or config)
        self._manual_prices: Dict[str, float] = {}

        # Contract prices (from agreements)
        self._contract_prices: Dict[str, Dict] = {}

        # Price history storage
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}

        logger.info(
            f"FuelPricingService initialized "
            f"(source: {config.primary_source}, TTL: {config.update_interval_minutes}min)"
        )

    def get_current_price(
        self,
        fuel_type: str,
        region: str = "henry_hub",
        include_carbon: bool = True,
    ) -> PriceQuote:
        """
        Get current fuel price quote.

        Args:
            fuel_type: Fuel type identifier
            region: Pricing region for basis differential
            include_carbon: Include carbon cost in total

        Returns:
            PriceQuote with complete price breakdown
        """
        cache_key = f"{fuel_type}:{region}:{include_carbon}"

        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached

        logger.debug(f"Fetching price for {fuel_type} in {region}")

        # Fetch base commodity price
        commodity_price = self._get_commodity_price(fuel_type)

        # Apply regional basis differential
        basis_differential = self._get_basis_differential(region, fuel_type)

        # Calculate transport cost
        transport_cost = self._calculate_transport_cost(fuel_type, region)

        # Calculate taxes
        taxes = self._calculate_taxes(commodity_price, fuel_type)

        # Calculate carbon cost
        carbon_cost = 0.0
        if include_carbon:
            carbon_cost = self._calculate_carbon_cost(fuel_type)

        # Total delivered price
        total_price = (
            commodity_price +
            basis_differential +
            transport_cost +
            taxes +
            carbon_cost
        )

        # Create quote
        now = datetime.now(timezone.utc)
        quote = PriceQuote(
            fuel_type=fuel_type,
            commodity_price=round(commodity_price, 4),
            basis_differential=round(basis_differential, 4),
            transport_cost=round(transport_cost, 4),
            taxes=round(taxes, 4),
            carbon_cost=round(carbon_cost, 4),
            total_price=round(total_price, 4),
            unit="USD/MMBTU",
            source=self.config.primary_source.value,
            region=region,
            timestamp=now,
            valid_until=now + timedelta(minutes=self.config.update_interval_minutes),
        )

        # Cache the quote
        self._cache.set(cache_key, quote)

        # Store in history
        self._add_to_history(fuel_type, now, total_price)

        return quote

    def get_all_current_prices(
        self,
        fuel_types: List[str],
        region: str = "henry_hub",
    ) -> Dict[str, PriceQuote]:
        """
        Get current prices for multiple fuel types.

        Args:
            fuel_types: List of fuel type identifiers
            region: Pricing region

        Returns:
            Dictionary of fuel type to PriceQuote
        """
        return {
            fuel_type: self.get_current_price(fuel_type, region)
            for fuel_type in fuel_types
        }

    def get_price_history(
        self,
        fuel_type: str,
        days: int = 30,
    ) -> PriceHistory:
        """
        Get historical prices for a fuel type.

        Args:
            fuel_type: Fuel type identifier
            days: Number of days of history

        Returns:
            PriceHistory with statistics
        """
        history = self._price_history.get(fuel_type, [])

        # Filter to requested period
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered = [(ts, price) for ts, price in history if ts >= cutoff]

        if not filtered:
            # Return empty history
            return PriceHistory(
                fuel_type=fuel_type,
                prices=[],
                source=self.config.primary_source.value,
                avg_price=0.0,
                min_price=0.0,
                max_price=0.0,
                std_dev=0.0,
                volatility_pct=0.0,
                period_days=days,
            )

        # Calculate statistics
        prices = [p for _, p in filtered]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)

        # Standard deviation
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5

        # Volatility (coefficient of variation)
        volatility_pct = (std_dev / avg_price * 100) if avg_price > 0 else 0.0

        # Trend analysis
        trend_direction, trend_slope = self._calculate_trend(filtered)

        return PriceHistory(
            fuel_type=fuel_type,
            prices=filtered,
            source=self.config.primary_source.value,
            avg_price=round(avg_price, 4),
            min_price=round(min_price, 4),
            max_price=round(max_price, 4),
            std_dev=round(std_dev, 4),
            volatility_pct=round(volatility_pct, 2),
            trend_direction=trend_direction,
            trend_slope=round(trend_slope, 4),
            period_days=days,
        )

    def get_price_forecast(
        self,
        fuel_type: str,
        horizon_days: int = 30,
    ) -> PriceForecast:
        """
        Get price forecast for a fuel type.

        Uses trend extrapolation with uncertainty bounds.
        In production, could integrate with ML forecasting.

        Args:
            fuel_type: Fuel type identifier
            horizon_days: Forecast horizon in days

        Returns:
            PriceForecast with low/mid/high projections
        """
        # Get current price and history
        current_quote = self.get_current_price(fuel_type)
        history = self.get_price_history(fuel_type, days=90)

        current_price = current_quote.total_price

        # Calculate forecast based on trend
        forecasts = []
        now = datetime.now(timezone.utc)

        for day in range(1, horizon_days + 1):
            forecast_date = now + timedelta(days=day)

            # Mid forecast: current + trend
            mid = current_price + history.trend_slope * day

            # Low/high: add uncertainty based on volatility
            volatility_factor = 1 + (history.volatility_pct / 100)
            low = mid / volatility_factor
            high = mid * volatility_factor

            forecasts.append((forecast_date, round(low, 4), round(mid, 4), round(high, 4)))

        return PriceForecast(
            fuel_type=fuel_type,
            current_price=current_price,
            forecasts=forecasts,
            unit="USD/MMBTU",
            horizon_days=horizon_days,
            model_type="trend_extrapolation",
            confidence_level=0.8,
        )

    def set_manual_price(
        self,
        fuel_type: str,
        price: float,
    ) -> None:
        """
        Set manual price for a fuel type.

        Args:
            fuel_type: Fuel type identifier
            price: Price in $/MMBTU
        """
        self._manual_prices[fuel_type] = price
        logger.info(f"Manual price set for {fuel_type}: ${price:.4f}/MMBTU")

        # Clear cache for this fuel type
        self._cache.clear()

    def set_contract_price(
        self,
        fuel_type: str,
        price: float,
        contract_id: str,
        valid_from: datetime,
        valid_until: datetime,
    ) -> None:
        """
        Set contract price for a fuel type.

        Args:
            fuel_type: Fuel type identifier
            price: Contract price in $/MMBTU
            contract_id: Contract identifier
            valid_from: Contract start date
            valid_until: Contract end date
        """
        self._contract_prices[fuel_type] = {
            "price": price,
            "contract_id": contract_id,
            "valid_from": valid_from,
            "valid_until": valid_until,
        }
        logger.info(
            f"Contract price set for {fuel_type}: ${price:.4f}/MMBTU "
            f"(contract: {contract_id})"
        )

    def _get_commodity_price(self, fuel_type: str) -> float:
        """Get base commodity price for fuel type."""
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")

        # Check for manual price first
        if fuel_key in self._manual_prices:
            return self._manual_prices[fuel_key]

        # Check for valid contract price
        if fuel_key in self._contract_prices:
            contract = self._contract_prices[fuel_key]
            now = datetime.now(timezone.utc)
            if contract["valid_from"] <= now <= contract["valid_until"]:
                return contract["price"]

        # Fetch from primary source
        try:
            return self._fetch_price_from_source(
                fuel_key,
                self.config.primary_source
            )
        except Exception as e:
            logger.warning(
                f"Primary source failed for {fuel_type}: {e}, "
                "trying secondary"
            )

        # Try secondary source
        if self.config.secondary_source:
            try:
                return self._fetch_price_from_source(
                    fuel_key,
                    self.config.secondary_source
                )
            except Exception as e:
                logger.warning(f"Secondary source also failed: {e}")

        # Return fallback prices
        return self._get_fallback_price(fuel_key)

    def _fetch_price_from_source(
        self,
        fuel_type: str,
        source: PriceSource,
    ) -> float:
        """Fetch price from specified source."""
        fetcher = self._fetchers.get(source)
        if fetcher:
            return fetcher(fuel_type)
        return self._get_fallback_price(fuel_type)

    def _fetch_henry_hub(self, fuel_type: str) -> float:
        """Fetch Henry Hub natural gas price."""
        # In production, this would call an actual API
        # Returning representative prices for simulation
        return 2.85  # Typical Henry Hub price ($/MMBTU)

    def _fetch_brent(self, fuel_type: str) -> float:
        """Fetch Brent crude price and convert."""
        # Typical Brent price ~$75/bbl
        brent_per_bbl = 75.0
        # Convert to $/MMBTU for oil products
        return brent_per_bbl * UNIT_CONVERSIONS["usd_per_bbl_to_usd_per_mmbtu_oil"]

    def _fetch_wti(self, fuel_type: str) -> float:
        """Fetch WTI crude price and convert."""
        # Typical WTI price ~$72/bbl
        wti_per_bbl = 72.0
        return wti_per_bbl * UNIT_CONVERSIONS["usd_per_bbl_to_usd_per_mmbtu_oil"]

    def _fetch_regional(self, fuel_type: str) -> float:
        """Fetch regional hub price."""
        # Base on Henry Hub with typical differential
        return 2.85 + 0.25

    def _fetch_contract(self, fuel_type: str) -> float:
        """Get contract price."""
        if fuel_type in self._contract_prices:
            return self._contract_prices[fuel_type]["price"]
        raise ValueError(f"No contract price for {fuel_type}")

    def _fetch_manual(self, fuel_type: str) -> float:
        """Get manual price."""
        if fuel_type in self._manual_prices:
            return self._manual_prices[fuel_type]
        raise ValueError(f"No manual price for {fuel_type}")

    def _get_fallback_price(self, fuel_type: str) -> float:
        """Get fallback price for fuel type."""
        fallback_prices = {
            "natural_gas": 3.00,
            "no2_fuel_oil": 15.00,
            "no6_fuel_oil": 12.00,
            "lpg_propane": 8.00,
            "lpg_butane": 7.50,
            "coal_bituminous": 2.50,
            "coal_sub_bituminous": 2.00,
            "biomass_wood": 4.00,
            "biomass_pellets": 5.00,
            "biogas": 6.00,
            "hydrogen": 15.00,
            "rng": 8.00,
        }
        return fallback_prices.get(fuel_type, 5.00)

    def _get_basis_differential(self, region: str, fuel_type: str) -> float:
        """Get regional basis differential."""
        if not self.config.basis_differential_enabled:
            return 0.0

        # Only applies to natural gas
        if "gas" not in fuel_type.lower() and "rng" not in fuel_type.lower():
            return 0.0

        region_key = region.lower().replace(" ", "_").replace("-", "_")
        return REGIONAL_BASIS_DIFFERENTIALS.get(region_key, 0.0)

    def _calculate_transport_cost(self, fuel_type: str, region: str) -> float:
        """Calculate transportation cost."""
        if not self.config.transport_cost_enabled:
            return 0.0

        # Simplified transport cost model
        # In production, this would use actual logistics data
        transport_costs = {
            "natural_gas": 0.15,  # Pipeline
            "no2_fuel_oil": 0.50,  # Truck
            "no6_fuel_oil": 0.40,  # Ship/rail
            "lpg_propane": 0.35,
            "coal_bituminous": 0.30,  # Rail
            "biomass_wood": 0.60,  # Truck
        }

        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")
        return transport_costs.get(fuel_key, 0.20)

    def _calculate_taxes(self, commodity_price: float, fuel_type: str) -> float:
        """Calculate applicable taxes."""
        if not self.config.taxes_included:
            return 0.0

        # Simplified tax model (sales tax on fuel)
        tax_rate = 0.05  # 5% default
        return commodity_price * tax_rate

    def _calculate_carbon_cost(self, fuel_type: str) -> float:
        """Calculate carbon cost based on emission factor."""
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")
        emission_factor = EMISSION_FACTORS.get(fuel_key, 53.0)

        # Convert $/ton to $/kg
        carbon_price_per_kg = self.carbon_price_usd_ton / 1000.0

        # Carbon cost per MMBTU
        return emission_factor * carbon_price_per_kg

    def _add_to_history(
        self,
        fuel_type: str,
        timestamp: datetime,
        price: float,
    ) -> None:
        """Add price to history."""
        if fuel_type not in self._price_history:
            self._price_history[fuel_type] = []

        self._price_history[fuel_type].append((timestamp, price))

        # Trim old history
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.price_history_days
        )
        self._price_history[fuel_type] = [
            (ts, p) for ts, p in self._price_history[fuel_type]
            if ts >= cutoff
        ]

    def _calculate_trend(
        self,
        history: List[Tuple[datetime, float]],
    ) -> Tuple[str, float]:
        """Calculate price trend from history."""
        if len(history) < 2:
            return "stable", 0.0

        # Simple linear regression
        n = len(history)
        base_time = history[0][0]

        x = [(ts - base_time).total_seconds() / 86400 for ts, _ in history]  # Days
        y = [p for _, p in history]

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable", 0.0

        slope = numerator / denominator

        if slope > 0.01:
            direction = "up"
        elif slope < -0.01:
            direction = "down"
        else:
            direction = "stable"

        return direction, slope

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        return self._cache.hit_rate
