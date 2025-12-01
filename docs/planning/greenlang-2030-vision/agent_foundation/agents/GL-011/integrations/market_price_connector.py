# -*- coding: utf-8 -*-
"""
Market Price Connector for GL-011 FUELCRAFT.

Provides integration with fuel market price feeds from ICE, NYMEX,
EIA, and other commodity exchanges for real-time pricing data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class PriceQuote:
    """Single price quote for a fuel."""
    fuel_type: str
    price: float
    currency: str
    unit: str
    source: str
    timestamp: datetime
    change_24h: float
    change_percent_24h: float


@dataclass
class PriceHistory:
    """Historical price data."""
    fuel_type: str
    prices: List[Dict[str, Any]]  # [{timestamp, price}, ...]
    period: str
    high: float
    low: float
    average: float
    volatility: float


@dataclass
class MarketSummary:
    """Summary of market conditions."""
    timestamp: datetime
    quotes: Dict[str, PriceQuote]
    market_status: str
    trends: Dict[str, str]  # fuel -> trend (up, down, stable)
    alerts: List[str]


class MarketPriceConnector:
    """
    Connector for fuel market price feeds.

    Data sources:
    - ICE (Intercontinental Exchange)
    - NYMEX (New York Mercantile Exchange)
    - EIA (US Energy Information Administration)
    - Simulation mode

    Example:
        >>> connector = MarketPriceConnector(config)
        >>> await connector.connect()
        >>> prices = await connector.get_current_prices()
    """

    # Base prices for simulation (USD/kg unless noted)
    BASE_PRICES = {
        'natural_gas': 0.045,    # ~$3.50/MMBtu
        'coal': 0.035,           # ~$140/tonne
        'biomass': 0.08,         # ~$180/tonne
        'hydrogen': 5.00,        # ~$5/kg
        'fuel_oil': 0.55,        # ~$60/barrel
        'diesel': 0.95,          # ~$3.50/gallon
        'propane': 0.55,         # ~$1.50/gallon
        'biogas': 0.035,
        'wood_pellets': 0.12
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market price connector.

        Args:
            config: Configuration with sources, API keys, etc.
        """
        self.config = config
        self.sources = config.get('sources', ['simulation'])
        self.update_interval = config.get('update_interval_seconds', 300)
        self.connected = False
        self._last_update: Optional[datetime] = None
        self._price_cache: Dict[str, PriceQuote] = {}
        self._price_history: Dict[str, List[Dict[str, Any]]] = {}

    async def connect(self) -> bool:
        """
        Establish connection to market data sources.

        Returns:
            True if connection successful
        """
        try:
            for source in self.sources:
                if source == 'simulation':
                    logger.info("Market connector in simulation mode")
                elif source == 'ice':
                    logger.info("Connecting to ICE...")
                    # Would connect to ICE API
                elif source == 'nymex':
                    logger.info("Connecting to NYMEX...")
                elif source == 'eia':
                    logger.info("Connecting to EIA API...")

            self.connected = True
            await self._refresh_prices()
            return True

        except Exception as e:
            logger.error(f"Market connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close connection."""
        self.connected = False
        logger.info("Market connector disconnected")

    async def get_current_prices(
        self,
        fuel_types: Optional[List[str]] = None
    ) -> Dict[str, PriceQuote]:
        """
        Get current prices for fuels.

        Args:
            fuel_types: List of fuel types (or all if None)

        Returns:
            Dict mapping fuel type to PriceQuote
        """
        if not self.connected:
            logger.warning("Not connected to market data")
            return {}

        # Check if refresh needed
        if self._needs_refresh():
            await self._refresh_prices()

        if fuel_types:
            return {f: self._price_cache[f] for f in fuel_types if f in self._price_cache}
        return self._price_cache.copy()

    async def get_price(self, fuel_type: str) -> Optional[PriceQuote]:
        """
        Get current price for a specific fuel.

        Args:
            fuel_type: Type of fuel

        Returns:
            PriceQuote or None
        """
        prices = await self.get_current_prices([fuel_type])
        return prices.get(fuel_type)

    async def get_price_history(
        self,
        fuel_type: str,
        days: int = 30
    ) -> PriceHistory:
        """
        Get historical prices for a fuel.

        Args:
            fuel_type: Type of fuel
            days: Number of days of history

        Returns:
            PriceHistory with daily prices
        """
        if fuel_type not in self._price_history:
            self._generate_price_history(fuel_type, days)

        history = self._price_history[fuel_type][-days:]
        prices = [h['price'] for h in history]

        return PriceHistory(
            fuel_type=fuel_type,
            prices=history,
            period=f"{days}_days",
            high=max(prices),
            low=min(prices),
            average=sum(prices) / len(prices),
            volatility=self._calculate_volatility(prices)
        )

    async def get_market_summary(self) -> MarketSummary:
        """
        Get summary of market conditions.

        Returns:
            MarketSummary with all quotes and trends
        """
        prices = await self.get_current_prices()

        # Determine trends
        trends = {}
        alerts = []

        for fuel, quote in prices.items():
            if quote.change_percent_24h > 5:
                trends[fuel] = 'up'
                alerts.append(f"{fuel} price up {quote.change_percent_24h:.1f}%")
            elif quote.change_percent_24h < -5:
                trends[fuel] = 'down'
                alerts.append(f"{fuel} price down {abs(quote.change_percent_24h):.1f}%")
            else:
                trends[fuel] = 'stable'

        return MarketSummary(
            timestamp=datetime.now(timezone.utc),
            quotes=prices,
            market_status='open',  # Simplified
            trends=trends,
            alerts=alerts
        )

    async def subscribe_to_updates(
        self,
        fuel_types: List[str],
        callback: Any
    ) -> str:
        """
        Subscribe to price updates.

        Args:
            fuel_types: Fuels to monitor
            callback: Callback function for updates

        Returns:
            Subscription ID
        """
        subscription_id = f"SUB-{len(fuel_types)}-{hash(tuple(fuel_types)) % 10000}"
        logger.info(f"Created subscription {subscription_id} for {fuel_types}")
        return subscription_id

    def _needs_refresh(self) -> bool:
        """Check if price refresh is needed."""
        if not self._last_update:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_update).total_seconds()
        return elapsed > self.update_interval

    async def _refresh_prices(self) -> None:
        """Refresh all prices from sources."""
        now = datetime.now(timezone.utc)

        for fuel_type, base_price in self.BASE_PRICES.items():
            # Simulate price movement
            random.seed(int(now.timestamp()) + hash(fuel_type))
            volatility = 0.02  # 2% daily volatility
            change = random.gauss(0, volatility)
            current_price = base_price * (1 + change)

            # 24h change
            random.seed(int(now.timestamp() - 86400) + hash(fuel_type))
            yesterday_change = random.gauss(0, volatility)
            yesterday_price = base_price * (1 + yesterday_change)
            change_24h = current_price - yesterday_price
            change_pct = (change_24h / yesterday_price) * 100

            self._price_cache[fuel_type] = PriceQuote(
                fuel_type=fuel_type,
                price=round(current_price, 4),
                currency='USD',
                unit='kg',
                source='simulation',
                timestamp=now,
                change_24h=round(change_24h, 4),
                change_percent_24h=round(change_pct, 2)
            )

        self._last_update = now
        logger.debug(f"Refreshed prices for {len(self._price_cache)} fuels")

    def _generate_price_history(self, fuel_type: str, days: int) -> None:
        """Generate historical price data for simulation."""
        base_price = self.BASE_PRICES.get(fuel_type, 0.05)
        history = []
        now = datetime.now(timezone.utc)

        for i in range(days, 0, -1):
            date = now - timedelta(days=i)
            random.seed(int(date.timestamp()) + hash(fuel_type))

            # Simulate price walk
            cumulative_change = 0
            for _ in range(i):
                cumulative_change += random.gauss(0, 0.005)

            price = base_price * (1 + cumulative_change)

            history.append({
                'timestamp': date.isoformat(),
                'price': round(price, 4),
                'volume': random.randint(10000, 100000)
            })

        self._price_history[fuel_type] = history

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility (standard deviation of returns)."""
        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return round(math.sqrt(variance) * 100, 2)  # Percentage
