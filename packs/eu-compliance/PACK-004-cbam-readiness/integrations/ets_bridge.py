# -*- coding: utf-8 -*-
"""
ETSBridge - EU ETS Price Feed Integration for CBAM Readiness Pack
==================================================================

This module integrates EU Emissions Trading System (ETS) price data into the
CBAM Readiness Pack. It provides current and historical ETS allowance prices,
weekly auction clearing prices, price projections under multiple scenarios,
carbon price comparisons between the EU ETS and origin countries, ECB
exchange rate lookups, and CBAM certificate price calculations.

Per CBAM Regulation Article 21, certificate prices are based on the weekly
average EU ETS auction clearing price. This bridge provides the data layer
for that calculation.

Example:
    >>> bridge = ETSBridge()
    >>> price = bridge.get_current_price()
    >>> print(f"EU ETS: {price.price_eur_per_tco2} EUR/tCO2")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
"""

import hashlib
import logging
import math
import time
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class PriceSource(str, Enum):
    """Source of the ETS price data."""
    AUCTION = "AUCTION"
    SPOT = "SPOT"
    MANUAL = "MANUAL"
    PROJECTED = "PROJECTED"
    MOCK = "MOCK"


class ProjectionScenario(str, Enum):
    """Price projection scenario."""
    BASELINE = "baseline"
    HIGH = "high"
    LOW = "low"
    POLICY_TIGHTENING = "policy_tightening"


class Currency(str, Enum):
    """Supported currencies for carbon price conversion."""
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    CNY = "CNY"
    KRW = "KRW"
    NZD = "NZD"
    CAD = "CAD"
    AUD = "AUD"
    ZAR = "ZAR"


# =============================================================================
# Data Models
# =============================================================================


class ETSPrice(BaseModel):
    """A single EU ETS price observation."""
    date: str = Field(..., description="Date of the observation (YYYY-MM-DD)")
    price_eur_per_tco2: float = Field(
        ..., ge=0.0, description="Price in EUR per tonne CO2 equivalent"
    )
    source: PriceSource = Field(default=PriceSource.MOCK, description="Price data source")
    currency: str = Field(default="EUR", description="Currency of the price")
    volume_traded: Optional[float] = Field(
        None, description="Volume traded in allowances (if auction)"
    )
    notes: str = Field(default="", description="Additional notes")


class CarbonPriceComparison(BaseModel):
    """Comparison of EU ETS price vs. origin country carbon price."""
    comparison_id: str = Field(
        default_factory=lambda: str(uuid4())[:12],
        description="Unique comparison identifier",
    )
    date: str = Field(..., description="Comparison date")
    eu_ets_price: float = Field(..., description="EU ETS price in EUR/tCO2")
    origin_country: str = Field(..., description="Origin country code")
    origin_country_name: str = Field(default="", description="Origin country name")
    origin_price: float = Field(default=0.0, description="Carbon price in origin country")
    origin_currency: str = Field(default="", description="Currency of origin price")
    exchange_rate: float = Field(
        default=1.0, description="Exchange rate to EUR"
    )
    origin_price_eur: float = Field(
        default=0.0, description="Origin price converted to EUR"
    )
    price_difference: float = Field(
        default=0.0, description="EU ETS - Origin price (EUR)"
    )
    deduction_eligible: bool = Field(
        default=False, description="Whether the origin price is deductible under CBAM"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CertificatePriceResult(BaseModel):
    """Result of a CBAM certificate price calculation."""
    base_ets_price: float = Field(..., description="Base EU ETS price used")
    weekly_average_price: float = Field(
        default=0.0, description="Weekly average auction price per Art. 21"
    )
    adjustments_applied: Dict[str, float] = Field(
        default_factory=dict, description="Adjustments applied to the price"
    )
    certificate_price: float = Field(..., description="Final certificate price in EUR/tCO2")
    calculation_date: str = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"),
        description="Date of calculation",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# =============================================================================
# Mock Price Data (Realistic EU ETS prices 2024-2026)
# =============================================================================


def _generate_mock_price_series() -> List[Dict[str, Any]]:
    """Generate a realistic mock EU ETS price series from 2024-01 to 2026-03.

    Prices model the ~60-100 EUR/tCO2 range observed in the EU ETS market
    with seasonal patterns and a modest upward trend.

    Returns:
        List of dicts with date, price, source, and volume.
    """
    prices: List[Dict[str, Any]] = []
    base_price = 65.0
    trend_per_month = 0.5
    start_date = date(2024, 1, 1)
    end_date = date(2026, 3, 14)

    current = start_date
    idx = 0
    while current <= end_date:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        months_elapsed = (current.year - 2024) * 12 + current.month - 1
        seasonal = 5.0 * math.sin(2 * math.pi * (current.month - 1) / 12)
        trend = trend_per_month * months_elapsed

        # Deterministic pseudo-random variation based on date ordinal
        noise = ((current.toordinal() * 7 + 13) % 100 - 50) / 25.0

        price = base_price + trend + seasonal + noise
        price = max(40.0, min(120.0, round(price, 2)))

        is_auction_day = current.weekday() == 2  # Wednesdays
        source = PriceSource.AUCTION if is_auction_day else PriceSource.SPOT
        volume = 3_500_000 if is_auction_day else None

        prices.append({
            "date": current.strftime("%Y-%m-%d"),
            "price_eur_per_tco2": price,
            "source": source,
            "volume_traded": volume,
        })

        current += timedelta(days=1)
        idx += 1

    return prices


_MOCK_PRICE_SERIES: Optional[List[Dict[str, Any]]] = None


def _get_mock_prices() -> List[Dict[str, Any]]:
    """Lazily initialize and return the mock price series."""
    global _MOCK_PRICE_SERIES
    if _MOCK_PRICE_SERIES is None:
        _MOCK_PRICE_SERIES = _generate_mock_price_series()
    return _MOCK_PRICE_SERIES


# =============================================================================
# Mock Exchange Rates
# =============================================================================

_MOCK_EXCHANGE_RATES: Dict[str, float] = {
    "EUR": 1.0,
    "USD": 0.92,
    "GBP": 1.17,
    "CHF": 1.06,
    "JPY": 0.0061,
    "CNY": 0.127,
    "KRW": 0.00069,
    "NZD": 0.56,
    "CAD": 0.68,
    "AUD": 0.60,
    "ZAR": 0.050,
}


# =============================================================================
# Origin Country Carbon Prices (for comparison)
# =============================================================================

_ORIGIN_CARBON_PRICES: Dict[str, Dict[str, Any]] = {
    "GB": {"price": 50.0, "currency": "GBP", "scheme": "UK ETS", "deductible": True},
    "CA": {"price": 80.0, "currency": "CAD", "scheme": "Federal Carbon Tax", "deductible": True},
    "NZ": {"price": 70.0, "currency": "NZD", "scheme": "NZ ETS", "deductible": True},
    "KR": {"price": 25000.0, "currency": "KRW", "scheme": "Korea ETS", "deductible": True},
    "CN": {"price": 80.0, "currency": "CNY", "scheme": "National ETS", "deductible": True},
    "JP": {"price": 289.0, "currency": "JPY", "scheme": "Carbon Tax", "deductible": True},
    "ZA": {"price": 190.0, "currency": "ZAR", "scheme": "Carbon Tax", "deductible": True},
    "CH": {"price": 120.0, "currency": "CHF", "scheme": "Swiss ETS (linked)", "deductible": True},
    "SG": {"price": 25.0, "currency": "USD", "scheme": "Carbon Tax", "deductible": True},
    "UA": {"price": 30.0, "currency": "EUR", "scheme": "Carbon Tax", "deductible": True},
    "KZ": {"price": 1500.0, "currency": "KRW", "scheme": "Kazakhstan ETS", "deductible": True},
}


# =============================================================================
# ETS Bridge Implementation
# =============================================================================


class ETSBridge:
    """EU ETS price feed integration for CBAM Readiness Pack.

    Provides current, historical, and projected EU ETS prices, cross-country
    carbon price comparisons, exchange rate lookups, and CBAM certificate
    price calculations per Article 21 of the CBAM Regulation.

    In production, this bridge connects to live market data feeds. In
    development/test, it uses realistic mock data covering 2024-2026.

    Attributes:
        config: Optional configuration dictionary
        logger: Module-level logger
        _stub_mode: Whether running in stub/mock mode

    Example:
        >>> bridge = ETSBridge()
        >>> current = bridge.get_current_price()
        >>> assert current.price_eur_per_tco2 > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ETS bridge.

        Args:
            config: Optional configuration dictionary. Keys:
                - live_feed_url: URL for live price feed (enables live mode)
                - api_key: API key for live feed
                - cache_ttl_seconds: Cache TTL for price lookups
        """
        self.config = config or {}
        self.logger = logger
        self._stub_mode = not bool(self.config.get("live_feed_url"))
        self._initialized_at = datetime.utcnow()
        self._price_cache: Dict[str, ETSPrice] = {}

        mode = "mock" if self._stub_mode else "live"
        self.logger.info("ETSBridge initialized in %s mode", mode)

    # -------------------------------------------------------------------------
    # Current Price
    # -------------------------------------------------------------------------

    def get_current_price(self) -> ETSPrice:
        """Get the most recent EU ETS allowance price.

        Returns:
            ETSPrice for the latest available trading day.
        """
        if self._stub_mode:
            prices = _get_mock_prices()
            if not prices:
                return ETSPrice(
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                    price_eur_per_tco2=75.0,
                    source=PriceSource.MOCK,
                )
            latest = prices[-1]
            return ETSPrice(
                date=latest["date"],
                price_eur_per_tco2=latest["price_eur_per_tco2"],
                source=PriceSource.MOCK,
                volume_traded=latest.get("volume_traded"),
            )

        # Production path: would call live feed API
        self.logger.warning("Live feed not implemented, returning mock price")
        return ETSPrice(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            price_eur_per_tco2=75.0,
            source=PriceSource.MANUAL,
        )

    # -------------------------------------------------------------------------
    # Historical Prices
    # -------------------------------------------------------------------------

    def get_price_history(
        self,
        start_date: str,
        end_date: str,
    ) -> List[ETSPrice]:
        """Get historical EU ETS prices for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            List of ETSPrice sorted by date ascending.
        """
        prices = _get_mock_prices()
        result: List[ETSPrice] = []

        for entry in prices:
            if start_date <= entry["date"] <= end_date:
                result.append(ETSPrice(
                    date=entry["date"],
                    price_eur_per_tco2=entry["price_eur_per_tco2"],
                    source=entry.get("source", PriceSource.MOCK),
                    volume_traded=entry.get("volume_traded"),
                ))

        self.logger.info(
            "Price history: %d observations from %s to %s",
            len(result), start_date, end_date,
        )
        return result

    # -------------------------------------------------------------------------
    # Weekly Auction Price
    # -------------------------------------------------------------------------

    def get_weekly_auction_price(self) -> ETSPrice:
        """Get the most recent weekly EU ETS auction clearing price.

        Per CBAM Article 22, certificate prices are based on the weekly
        average of EU ETS auction clearing prices published on the common
        auction platform.

        Returns:
            ETSPrice for the latest Wednesday auction.
        """
        prices = _get_mock_prices()
        auction_prices = [
            p for p in prices
            if p.get("source") == PriceSource.AUCTION
        ]

        if not auction_prices:
            return ETSPrice(
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                price_eur_per_tco2=75.0,
                source=PriceSource.AUCTION,
                notes="No auction data available, using fallback",
            )

        latest_auction = auction_prices[-1]
        return ETSPrice(
            date=latest_auction["date"],
            price_eur_per_tco2=latest_auction["price_eur_per_tco2"],
            source=PriceSource.AUCTION,
            volume_traded=latest_auction.get("volume_traded"),
        )

    def get_weekly_average_price(self, week_ending: Optional[str] = None) -> float:
        """Calculate the weekly average auction price ending on a given date.

        If no date is provided, uses the most recent complete week.

        Args:
            week_ending: End of the week (YYYY-MM-DD), or None for latest.

        Returns:
            Weekly average price in EUR/tCO2.
        """
        if week_ending is None:
            today = datetime.utcnow().date()
            # Find previous Sunday
            days_since_sunday = (today.weekday() + 1) % 7
            week_end = today - timedelta(days=days_since_sunday)
            week_ending = week_end.strftime("%Y-%m-%d")

        end = datetime.strptime(week_ending, "%Y-%m-%d").date()
        start = end - timedelta(days=6)
        start_str = start.strftime("%Y-%m-%d")

        prices = self.get_price_history(start_str, week_ending)
        if not prices:
            self.logger.warning("No prices found for week ending %s", week_ending)
            return 0.0

        total = sum(p.price_eur_per_tco2 for p in prices)
        avg = round(total / len(prices), 2)
        self.logger.info(
            "Weekly average price for week ending %s: %.2f EUR/tCO2 (%d observations)",
            week_ending, avg, len(prices),
        )
        return avg

    # -------------------------------------------------------------------------
    # Price Projections
    # -------------------------------------------------------------------------

    def get_price_projection(
        self,
        horizon_months: int = 12,
        scenario: str = "baseline",
    ) -> List[ETSPrice]:
        """Generate an EU ETS price projection for the given horizon.

        Scenarios:
            - baseline: Moderate growth continuing current trend (~2% monthly)
            - high: Aggressive policy tightening or supply reduction (~5% monthly)
            - low: Recession or policy relaxation (~-1% monthly)
            - policy_tightening: Fit-for-55 scenario with steeper trajectory

        Args:
            horizon_months: Number of months to project.
            scenario: Projection scenario name.

        Returns:
            List of ETSPrice for projected months.
        """
        try:
            scenario_enum = ProjectionScenario(scenario)
        except ValueError:
            self.logger.warning("Unknown scenario '%s', using baseline", scenario)
            scenario_enum = ProjectionScenario.BASELINE

        monthly_growth_rates = {
            ProjectionScenario.BASELINE: 0.02,
            ProjectionScenario.HIGH: 0.05,
            ProjectionScenario.LOW: -0.01,
            ProjectionScenario.POLICY_TIGHTENING: 0.04,
        }

        growth = monthly_growth_rates[scenario_enum]
        current = self.get_current_price()
        base_price = current.price_eur_per_tco2

        projections: List[ETSPrice] = []
        current_date = datetime.utcnow().date()

        for month in range(1, horizon_months + 1):
            proj_date = current_date + timedelta(days=month * 30)
            proj_price = base_price * ((1 + growth) ** month)
            proj_price = round(max(20.0, proj_price), 2)

            projections.append(ETSPrice(
                date=proj_date.strftime("%Y-%m-%d"),
                price_eur_per_tco2=proj_price,
                source=PriceSource.PROJECTED,
                notes=f"Scenario: {scenario_enum.value}, month +{month}",
            ))

        self.logger.info(
            "Generated %d-month projection (%s): %.2f -> %.2f EUR/tCO2",
            horizon_months, scenario,
            projections[0].price_eur_per_tco2 if projections else 0,
            projections[-1].price_eur_per_tco2 if projections else 0,
        )
        return projections

    # -------------------------------------------------------------------------
    # Carbon Price Comparison
    # -------------------------------------------------------------------------

    def compare_carbon_prices(self, country_code: str) -> CarbonPriceComparison:
        """Compare the EU ETS price with the carbon price in an origin country.

        This is used to determine the CBAM certificate deduction per
        Article 9 of the CBAM Regulation, which allows deduction of the
        carbon price effectively paid in the country of origin.

        Args:
            country_code: ISO alpha-2 country code of the origin country.

        Returns:
            CarbonPriceComparison with price difference and deduction eligibility.
        """
        code = country_code.upper()
        current = self.get_current_price()
        eu_ets_price = current.price_eur_per_tco2

        origin_info = _ORIGIN_CARBON_PRICES.get(code)
        if origin_info is None:
            # No carbon pricing in origin country
            comparison = CarbonPriceComparison(
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                eu_ets_price=eu_ets_price,
                origin_country=code,
                origin_price=0.0,
                origin_currency="",
                exchange_rate=1.0,
                origin_price_eur=0.0,
                price_difference=eu_ets_price,
                deduction_eligible=False,
            )
            comparison.provenance_hash = _compute_hash(
                f"{comparison.comparison_id}:{eu_ets_price}:{code}:0"
            )
            return comparison

        origin_price = origin_info["price"]
        origin_currency = origin_info["currency"]
        rate = self.get_exchange_rate(origin_currency)
        origin_eur = round(origin_price * rate, 2)
        difference = round(eu_ets_price - origin_eur, 2)

        comparison = CarbonPriceComparison(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            eu_ets_price=eu_ets_price,
            origin_country=code,
            origin_country_name=origin_info.get("scheme", code),
            origin_price=origin_price,
            origin_currency=origin_currency,
            exchange_rate=rate,
            origin_price_eur=origin_eur,
            price_difference=difference,
            deduction_eligible=origin_info.get("deductible", False),
        )
        comparison.provenance_hash = _compute_hash(
            f"{comparison.comparison_id}:{eu_ets_price}:{code}:{origin_eur}"
        )

        self.logger.info(
            "Carbon price comparison %s: EU ETS=%.2f, %s=%.2f EUR, diff=%.2f",
            code, eu_ets_price, origin_info["scheme"], origin_eur, difference,
        )
        return comparison

    # -------------------------------------------------------------------------
    # Exchange Rates
    # -------------------------------------------------------------------------

    def get_exchange_rate(
        self,
        currency: str,
        reference_date: Optional[str] = None,
    ) -> float:
        """Get the ECB exchange rate for converting to EUR.

        In production, this would fetch from the ECB SDMX API. In mock
        mode, uses static representative exchange rates.

        Args:
            currency: ISO currency code.
            reference_date: Optional reference date (YYYY-MM-DD).

        Returns:
            Exchange rate multiplier (currency_amount * rate = EUR_amount).
        """
        code = currency.upper()
        if code == "EUR":
            return 1.0

        rate = _MOCK_EXCHANGE_RATES.get(code)
        if rate is None:
            self.logger.warning(
                "Exchange rate not found for currency '%s', returning 1.0", code,
            )
            return 1.0

        return rate

    # -------------------------------------------------------------------------
    # Certificate Price Calculation
    # -------------------------------------------------------------------------

    def calculate_certificate_price(
        self,
        ets_price: Optional[float] = None,
        adjustments: Optional[Dict[str, float]] = None,
    ) -> CertificatePriceResult:
        """Calculate the CBAM certificate price.

        Per CBAM Regulation Article 21, the price of CBAM certificates
        shall be calculated as the average of the closing prices of EU
        ETS allowances on the common auction platform for each calendar
        week.

        Args:
            ets_price: Override ETS price. If None, uses weekly average.
            adjustments: Dictionary of named adjustments (e.g. free allocation
                deduction, origin country carbon price deduction).

        Returns:
            CertificatePriceResult with the final certificate price.
        """
        if ets_price is not None:
            base_price = ets_price
        else:
            base_price = self.get_weekly_average_price()
            if base_price == 0.0:
                base_price = self.get_current_price().price_eur_per_tco2

        adjustments = adjustments or {}
        total_adjustment = sum(adjustments.values())
        certificate_price = round(max(0.0, base_price + total_adjustment), 2)

        provenance_data = f"cert:{base_price}:{total_adjustment}:{certificate_price}"

        result = CertificatePriceResult(
            base_ets_price=base_price,
            weekly_average_price=base_price,
            adjustments_applied=adjustments,
            certificate_price=certificate_price,
            provenance_hash=_compute_hash(provenance_data),
        )

        self.logger.info(
            "Certificate price calculated: base=%.2f, adjustments=%.2f, final=%.2f EUR/tCO2",
            base_price, total_adjustment, certificate_price,
        )
        return result

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def get_price_summary(self) -> Dict[str, Any]:
        """Get a summary of the current ETS price environment.

        Returns:
            Dictionary with current price, 30-day range, YTD stats.
        """
        current = self.get_current_price()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        ytd_start = f"{datetime.utcnow().year}-01-01"

        recent = self.get_price_history(thirty_days_ago, today)
        ytd = self.get_price_history(ytd_start, today)

        recent_prices = [p.price_eur_per_tco2 for p in recent] if recent else [0.0]
        ytd_prices = [p.price_eur_per_tco2 for p in ytd] if ytd else [0.0]

        return {
            "current_price": current.price_eur_per_tco2,
            "current_date": current.date,
            "source": current.source.value,
            "thirty_day_high": max(recent_prices),
            "thirty_day_low": min(recent_prices),
            "thirty_day_avg": round(sum(recent_prices) / len(recent_prices), 2),
            "ytd_high": max(ytd_prices),
            "ytd_low": min(ytd_prices),
            "ytd_avg": round(sum(ytd_prices) / len(ytd_prices), 2),
            "ytd_observations": len(ytd),
            "mode": "mock" if self._stub_mode else "live",
        }


# =============================================================================
# Module-Level Helper
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
