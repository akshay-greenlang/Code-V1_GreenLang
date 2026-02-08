# -*- coding: utf-8 -*-
"""
Currency Converter - AGENT-DATA-003: ERP/Finance Connector
============================================================

Multi-currency normalisation engine for spend records. Provides
deterministic exchange-rate lookups, single-amount conversion,
batch record conversion, and rate management. Ships with built-in
default rates for 15+ major currencies.

Supports:
    - Single-amount currency conversion
    - Batch spend-record conversion to target currency
    - Manual exchange-rate setting with source and effective date
    - Rate lookup between any two supported currencies
    - Default rates for 15+ major currencies (hardcoded, deterministic)
    - Rate listing and supported-currency enumeration
    - Thread-safe statistics counters

Zero-Hallucination Guarantees:
    - All conversions use deterministic exchange rates
    - No external API calls (all rates are hardcoded or manually set)
    - Cross-rate calculation via base-currency triangulation
    - No LLM or ML model in conversion path

Example:
    >>> from greenlang.erp_connector.currency_converter import CurrencyConverter
    >>> converter = CurrencyConverter()
    >>> amount_usd, rate = converter.convert(1000.0, "EUR", "USD")
    >>> print(f"{amount_usd} USD at rate {rate.rate}")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Layer 1 imports
from greenlang.agents.data.erp_connector_agent import SpendRecord

logger = logging.getLogger(__name__)

__all__ = [
    "CurrencyRate",
    "CurrencyConverter",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today() -> date:
    """Return today's date in UTC."""
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CurrencyRate(BaseModel):
    """Exchange rate between two currencies.

    Stores the rate such that:
        1 unit of from_currency = rate units of to_currency
    """

    from_currency: str = Field(
        ..., min_length=3, max_length=3,
        description="Source currency ISO 4217 code",
    )
    to_currency: str = Field(
        ..., min_length=3, max_length=3,
        description="Target currency ISO 4217 code",
    )
    rate: float = Field(
        ..., gt=0, description="Exchange rate (1 from = rate to)",
    )
    source: str = Field(
        default="default", description="Rate source",
    )
    effective_date: date = Field(
        default_factory=_today, description="Rate effective date",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Default exchange rates (to USD)
# ---------------------------------------------------------------------------

# Rates expressed as: 1 USD = X foreign currency
# To convert FROM foreign TO USD: amount_foreign / rate
# To convert FROM USD TO foreign: amount_usd * rate
_DEFAULT_RATES_TO_USD: Dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,      # 1 USD = 0.92 EUR
    "GBP": 0.79,      # 1 USD = 0.79 GBP
    "JPY": 149.50,    # 1 USD = 149.50 JPY
    "CAD": 1.36,      # 1 USD = 1.36 CAD
    "AUD": 1.53,      # 1 USD = 1.53 AUD
    "CHF": 0.88,      # 1 USD = 0.88 CHF
    "CNY": 7.24,      # 1 USD = 7.24 CNY
    "INR": 83.10,     # 1 USD = 83.10 INR
    "MXN": 17.15,     # 1 USD = 17.15 MXN
    "BRL": 4.97,      # 1 USD = 4.97 BRL
    "KRW": 1325.00,   # 1 USD = 1325 KRW
    "SGD": 1.34,      # 1 USD = 1.34 SGD
    "HKD": 7.82,      # 1 USD = 7.82 HKD
    "SEK": 10.45,     # 1 USD = 10.45 SEK
    "NOK": 10.62,     # 1 USD = 10.62 NOK
    "DKK": 6.88,      # 1 USD = 6.88 DKK
    "ZAR": 18.85,     # 1 USD = 18.85 ZAR
    "THB": 35.50,     # 1 USD = 35.50 THB
    "TWD": 31.50,     # 1 USD = 31.50 TWD
    "PLN": 4.05,      # 1 USD = 4.05 PLN
    "NZD": 1.63,      # 1 USD = 1.63 NZD
}


# ---------------------------------------------------------------------------
# CurrencyConverter
# ---------------------------------------------------------------------------


class CurrencyConverter:
    """Multi-currency conversion engine with deterministic rates.

    Converts amounts between currencies using hardcoded default rates
    or manually set rates. All conversions are deterministic with
    no external API calls. Cross-rates are computed via USD
    triangulation.

    Attributes:
        _base_currency: Base currency for triangulation (default USD).
        _rates: Dictionary of (from, to) -> CurrencyRate.
        _rates_to_usd: Rates expressed as 1 USD = X foreign.
        _config: Configuration dictionary.
        _lock: Threading lock for statistics.
        _stats: Conversion statistics counters.

    Example:
        >>> converter = CurrencyConverter()
        >>> amount, rate = converter.convert(1000.0, "EUR", "USD")
        >>> print(f"{amount:.2f} USD")
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        base_currency: str = "USD",
    ) -> None:
        """Initialize CurrencyConverter.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``auto_load_defaults``: bool (default True)
            base_currency: Base currency for triangulation (default "USD").
        """
        self._config = config or {}
        self._base_currency = base_currency.upper()
        self._rates_to_usd: Dict[str, float] = {}
        self._custom_rates: Dict[Tuple[str, str], CurrencyRate] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "conversions_performed": 0,
            "by_currency_pair": {},
            "errors": 0,
        }

        # Auto-load defaults
        if self._config.get("auto_load_defaults", True):
            self.load_default_rates()

        logger.info(
            "CurrencyConverter initialised: base_currency=%s, "
            "currencies=%d",
            self._base_currency,
            len(self._rates_to_usd),
        )

    # ------------------------------------------------------------------
    # Public API - Conversion
    # ------------------------------------------------------------------

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: Optional[str] = None,
    ) -> Tuple[float, CurrencyRate]:
        """Convert an amount from one currency to another.

        Uses direct rate if available, otherwise triangulates through
        the base currency (USD).

        Args:
            amount: Amount to convert.
            from_currency: Source currency ISO 4217 code.
            to_currency: Target currency code (defaults to
                base_currency).

        Returns:
            Tuple of (converted_amount, CurrencyRate used).

        Raises:
            ValueError: If either currency is not supported.
        """
        from_cur = from_currency.upper()
        to_cur = (to_currency or self._base_currency).upper()

        if from_cur == to_cur:
            rate_obj = CurrencyRate(
                from_currency=from_cur,
                to_currency=to_cur,
                rate=1.0,
                source="identity",
            )
            return amount, rate_obj

        # Check for direct custom rate
        custom_key = (from_cur, to_cur)
        if custom_key in self._custom_rates:
            rate_obj = self._custom_rates[custom_key]
            converted = round(amount * rate_obj.rate, 2)
            self._record_conversion(from_cur, to_cur)
            return converted, rate_obj

        # Triangulate via base currency
        rate = self._compute_cross_rate(from_cur, to_cur)

        rate_obj = CurrencyRate(
            from_currency=from_cur,
            to_currency=to_cur,
            rate=rate,
            source="cross_rate_via_usd",
        )

        converted = round(amount * rate, 2)
        self._record_conversion(from_cur, to_cur)

        return converted, rate_obj

    def convert_records(
        self,
        records: List[SpendRecord],
        target_currency: str = "USD",
    ) -> List[SpendRecord]:
        """Convert spend record amounts to a target currency.

        Updates the amount_usd field on each record. If the record
        currency matches the target, the amount is used directly.

        Args:
            records: List of SpendRecord objects.
            target_currency: Target currency (default "USD").

        Returns:
            The same list of SpendRecord objects with amount_usd
            updated.
        """
        start = time.monotonic()
        target_cur = target_currency.upper()

        for record in records:
            if record.currency.upper() == target_cur:
                record.amount_usd = record.amount
                continue

            converted, _rate = self.convert(
                record.amount,
                record.currency,
                target_cur,
            )
            record.amount_usd = converted

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Converted %d records to %s (%.1f ms)",
            len(records), target_cur, elapsed_ms,
        )
        return records

    # ------------------------------------------------------------------
    # Public API - Rate Management
    # ------------------------------------------------------------------

    def set_rate(
        self,
        from_currency: str,
        to_currency: str,
        rate: float,
        source: str = "manual",
        effective_date: Optional[date] = None,
    ) -> CurrencyRate:
        """Set a custom exchange rate.

        Args:
            from_currency: Source currency code.
            to_currency: Target currency code.
            rate: Exchange rate (1 from = rate to).
            source: Rate source description (default "manual").
            effective_date: Rate effective date (defaults to today).

        Returns:
            CurrencyRate object.

        Raises:
            ValueError: If rate is not positive.
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")

        from_cur = from_currency.upper()
        to_cur = to_currency.upper()

        rate_obj = CurrencyRate(
            from_currency=from_cur,
            to_currency=to_cur,
            rate=rate,
            source=source,
            effective_date=effective_date or _today(),
        )

        self._custom_rates[(from_cur, to_cur)] = rate_obj

        # Also update the to-USD rates if applicable
        if to_cur == "USD":
            # 1 from_cur = rate USD, so 1 USD = 1/rate from_cur
            self._rates_to_usd[from_cur] = round(1.0 / rate, 6)
        elif from_cur == "USD":
            # 1 USD = rate to_cur
            self._rates_to_usd[to_cur] = rate

        logger.info(
            "Set rate: 1 %s = %.6f %s (source=%s)",
            from_cur, rate, to_cur, source,
        )
        return rate_obj

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> CurrencyRate:
        """Get the current exchange rate between two currencies.

        Args:
            from_currency: Source currency code.
            to_currency: Target currency code.

        Returns:
            CurrencyRate object.

        Raises:
            ValueError: If either currency is not supported.
        """
        from_cur = from_currency.upper()
        to_cur = to_currency.upper()

        if from_cur == to_cur:
            return CurrencyRate(
                from_currency=from_cur,
                to_currency=to_cur,
                rate=1.0,
                source="identity",
            )

        # Check custom rates
        custom_key = (from_cur, to_cur)
        if custom_key in self._custom_rates:
            return self._custom_rates[custom_key]

        # Compute cross rate
        rate = self._compute_cross_rate(from_cur, to_cur)
        return CurrencyRate(
            from_currency=from_cur,
            to_currency=to_cur,
            rate=rate,
            source="cross_rate_via_usd",
        )

    def list_rates(self) -> List[CurrencyRate]:
        """List all loaded exchange rates.

        Returns rates from the base-to-USD table and any custom rates.

        Returns:
            List of CurrencyRate objects.
        """
        rates: List[CurrencyRate] = []

        # Base rates (expressed as from_currency -> USD)
        for currency, to_usd_rate in sorted(self._rates_to_usd.items()):
            if currency == "USD":
                continue
            # Rate: 1 currency = (1/to_usd_rate) USD
            rate_to_usd = round(1.0 / to_usd_rate, 6) if to_usd_rate != 0 else 0.0
            rates.append(CurrencyRate(
                from_currency=currency,
                to_currency="USD",
                rate=rate_to_usd,
                source="default",
            ))

        # Custom rates
        for _key, rate_obj in self._custom_rates.items():
            rates.append(rate_obj)

        return rates

    def load_default_rates(self) -> None:
        """Load built-in default exchange rates.

        Populates the internal rate table with hardcoded rates for
        15+ major currencies. All rates are deterministic and do not
        rely on external API calls.
        """
        self._rates_to_usd = dict(_DEFAULT_RATES_TO_USD)
        logger.info(
            "Loaded %d default currency rates",
            len(self._rates_to_usd),
        )

    def get_supported_currencies(self) -> List[str]:
        """List all supported currency codes.

        Returns:
            Sorted list of ISO 4217 currency codes.
        """
        currencies = set(self._rates_to_usd.keys())

        # Also include currencies from custom rates
        for (from_cur, to_cur) in self._custom_rates:
            currencies.add(from_cur)
            currencies.add(to_cur)

        return sorted(currencies)

    def get_statistics(self) -> Dict[str, Any]:
        """Return conversion statistics.

        Returns:
            Dictionary of counter values and pair breakdowns.
        """
        with self._lock:
            return {
                "conversions_performed": self._stats[
                    "conversions_performed"
                ],
                "by_currency_pair": dict(
                    self._stats["by_currency_pair"],
                ),
                "supported_currencies": len(
                    self.get_supported_currencies(),
                ),
                "custom_rates_count": len(self._custom_rates),
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_cross_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> float:
        """Compute cross rate via USD triangulation.

        Converts: from_currency -> USD -> to_currency

        Args:
            from_currency: Source currency code.
            to_currency: Target currency code.

        Returns:
            Cross rate as a float.

        Raises:
            ValueError: If either currency is not in the rate table.
        """
        if from_currency not in self._rates_to_usd:
            raise ValueError(
                f"Unsupported currency: {from_currency}. "
                f"Supported: {', '.join(sorted(self._rates_to_usd.keys()))}"
            )
        if to_currency not in self._rates_to_usd:
            raise ValueError(
                f"Unsupported currency: {to_currency}. "
                f"Supported: {', '.join(sorted(self._rates_to_usd.keys()))}"
            )

        # 1 USD = from_rate units of from_currency
        # So 1 from_currency = 1/from_rate USD
        from_rate = self._rates_to_usd[from_currency]

        # 1 USD = to_rate units of to_currency
        to_rate = self._rates_to_usd[to_currency]

        # 1 from_currency = (to_rate / from_rate) to_currency
        if from_rate == 0:
            raise ValueError(
                f"Zero rate for {from_currency}, cannot compute cross rate"
            )

        cross_rate = to_rate / from_rate
        return round(cross_rate, 6)

    def _record_conversion(
        self,
        from_currency: str,
        to_currency: str,
    ) -> None:
        """Record a conversion in statistics.

        Args:
            from_currency: Source currency code.
            to_currency: Target currency code.
        """
        pair_key = f"{from_currency}->{to_currency}"
        with self._lock:
            self._stats["conversions_performed"] += 1
            pair_counts = self._stats["by_currency_pair"]
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
