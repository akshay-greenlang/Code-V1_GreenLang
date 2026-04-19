# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Market Data Providers

Market data integration for carbon trading.

Author: GreenLang GL-010 EmissionsGuardian
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import logging

from .schemas import CarbonMarket, Currency, MarketPrice

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_current_price(
        self,
        instrument: str,
        vintage: Optional[int] = None
    ) -> Optional[MarketPrice]:
        """Get current price for an instrument."""
        pass

    @abstractmethod
    def get_historical_prices(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        vintage: Optional[int] = None
    ) -> List[MarketPrice]:
        """Get historical prices."""
        pass

    @abstractmethod
    def subscribe(self, instrument: str, callback) -> str:
        """Subscribe to real-time updates."""
        pass

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from updates."""
        pass


class ICEMarketProvider(MarketDataProvider):
    """ICE (Intercontinental Exchange) market data provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.market = CarbonMarket.EU_ETS
        self._cache: Dict[str, MarketPrice] = {}
        logger.info("ICE Market Provider initialized")

    def get_current_price(
        self,
        instrument: str,
        vintage: Optional[int] = None
    ) -> Optional[MarketPrice]:
        """Get current EU ETS price."""
        # Simulated price data
        return MarketPrice(
            market=self.market,
            instrument=instrument,
            vintage=vintage,
            bid=Decimal("85.50"),
            ask=Decimal("86.00"),
            last=Decimal("85.75"),
            volume=Decimal("10000"),
            currency=Currency.EUR,
            source="ICE"
        )

    def get_historical_prices(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        vintage: Optional[int] = None
    ) -> List[MarketPrice]:
        """Get historical prices."""
        return []

    def subscribe(self, instrument: str, callback) -> str:
        """Subscribe to updates."""
        return f"ICE-{instrument}-{datetime.utcnow().timestamp()}"

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe."""
        return True


class CMEMarketProvider(MarketDataProvider):
    """CME Group market data provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.market = CarbonMarket.RGGI
        logger.info("CME Market Provider initialized")

    def get_current_price(
        self,
        instrument: str,
        vintage: Optional[int] = None
    ) -> Optional[MarketPrice]:
        """Get current RGGI price."""
        return MarketPrice(
            market=self.market,
            instrument=instrument,
            vintage=vintage,
            bid=Decimal("14.50"),
            ask=Decimal("14.75"),
            last=Decimal("14.60"),
            volume=Decimal("5000"),
            currency=Currency.USD,
            source="CME"
        )

    def get_historical_prices(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        vintage: Optional[int] = None
    ) -> List[MarketPrice]:
        return []

    def subscribe(self, instrument: str, callback) -> str:
        return f"CME-{instrument}-{datetime.utcnow().timestamp()}"

    def unsubscribe(self, subscription_id: str) -> bool:
        return True


class CBLMarketProvider(MarketDataProvider):
    """CBL Markets (voluntary carbon) provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.market = CarbonMarket.VOLUNTARY
        logger.info("CBL Market Provider initialized")

    def get_current_price(
        self,
        instrument: str,
        vintage: Optional[int] = None
    ) -> Optional[MarketPrice]:
        """Get current voluntary market price."""
        return MarketPrice(
            market=self.market,
            instrument=instrument,
            vintage=vintage,
            bid=Decimal("12.00"),
            ask=Decimal("13.00"),
            last=Decimal("12.50"),
            volume=Decimal("2000"),
            currency=Currency.USD,
            source="CBL"
        )

    def get_historical_prices(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        vintage: Optional[int] = None
    ) -> List[MarketPrice]:
        return []

    def subscribe(self, instrument: str, callback) -> str:
        return f"CBL-{instrument}-{datetime.utcnow().timestamp()}"

    def unsubscribe(self, subscription_id: str) -> bool:
        return True


class MarketDataAggregator:
    """Aggregates market data from multiple providers."""

    def __init__(self):
        self._providers: Dict[CarbonMarket, MarketDataProvider] = {}
        self._price_cache: Dict[str, MarketPrice] = {}
        logger.info("MarketDataAggregator initialized")

    def register_provider(
        self,
        market: CarbonMarket,
        provider: MarketDataProvider
    ) -> None:
        """Register a market data provider."""
        self._providers[market] = provider
        logger.info(f"Registered provider for {market.value}")

    def get_price(
        self,
        market: CarbonMarket,
        instrument: str,
        vintage: Optional[int] = None
    ) -> Optional[MarketPrice]:
        """Get current price from appropriate provider."""
        provider = self._providers.get(market)
        if not provider:
            logger.warning(f"No provider for market: {market.value}")
            return None

        return provider.get_current_price(instrument, vintage)

    def get_all_prices(self) -> Dict[str, MarketPrice]:
        """Get latest prices from all providers."""
        prices = {}
        for market, provider in self._providers.items():
            try:
                price = provider.get_current_price("default")
                if price:
                    prices[market.value] = price
            except Exception as e:
                logger.error(f"Error getting price from {market.value}: {e}")
        return prices


__all__ = [
    "MarketDataProvider",
    "ICEMarketProvider",
    "CMEMarketProvider",
    "CBLMarketProvider",
    "MarketDataAggregator",
]
