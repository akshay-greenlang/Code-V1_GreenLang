# -*- coding: utf-8 -*-
"""
SpendBasedCalculatorEngine - AGENT-MRV-019 Engine 5

GHG Protocol Scope 3 Category 6 spend-based emissions calculator using
Environmentally Extended Input-Output (EEIO) factors.

This engine provides spend-based emissions estimation as a fallback when
distance-based or supplier-specific data is unavailable. It implements:

1. **EEIO Factor Lookup**: Maps NAICS industry codes to kgCO2e/USD factors
   covering 10 travel-related sectors (air, rail, bus, taxi, car rental,
   hotel, campground, restaurant, ferry, scenic transport).

2. **Multi-Currency Conversion**: Converts 12 currencies to USD using
   stored mid-market exchange rates before applying EEIO factors.

3. **CPI Deflation**: Adjusts nominal spend to base-year (2021) real USD
   using US BLS CPI-U deflators, ensuring consistent EEIO factor application
   across reporting years 2015-2025.

4. **Margin Removal**: Optionally strips profit margins from spend data
   (default 15%) to isolate the cost of the service (fuel, energy, labor)
   from the supplier's profit, improving emission estimate accuracy.

5. **Expense Classification**: Simple keyword-based classifier that maps
   expense descriptions (e.g., "flight to London", "Marriott hotel") to
   NAICS codes, reducing manual categorization effort.

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern for concurrent pipeline use.

GHG Protocol Hierarchy Note:
    Spend-based is the lowest-accuracy method (Tier 3 data quality).
    Organizations should prioritize supplier-specific > distance-based >
    average-data > spend-based whenever data is available.

References:
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 6
    - US EPA EEIO (USEEIO v2.0) Supply Chain Emission Factors
    - US BLS CPI-U (Consumer Price Index for All Urban Consumers)
    - OECD Purchasing Power Parities (PPP)

Example:
    >>> engine = get_spend_based_calculator()
    >>> result = engine.calculate(SpendInput(
    ...     naics_code="481000",
    ...     amount=Decimal("5000.00"),
    ...     currency=CurrencyCode.USD,
    ...     reporting_year=2024
    ... ))
    >>> result.co2e
    Decimal('2077.72930800')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-006
"""

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.mrv.business_travel.models import (
    SpendInput,
    SpendResult,
    CurrencyCode,
    EFSource,
    EEIO_FACTORS,
    CURRENCY_RATES,
    CPI_DEFLATORS,
    calculate_provenance_hash,
    convert_currency_to_usd,
    get_cpi_deflator,
)
from greenlang.agents.mrv.business_travel.config import get_config
from greenlang.agents.mrv.business_travel.metrics import get_metrics

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "spend_based_calculator_engine"
ENGINE_VERSION: str = "1.0.0"

# Decimal precision for rounding (8 decimal places for sub-cent accuracy)
PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_2DP: Decimal = Decimal("0.01")

# Keyword mappings for expense classification
# Each key is a frozenset of keywords that map to a NAICS code
_EXPENSE_KEYWORDS: Dict[str, List[str]] = {
    "481000": [
        "flight", "air", "airline", "airfare", "aviation",
        "plane", "jet", "delta", "united", "american airlines",
        "southwest", "british airways", "lufthansa", "emirates",
    ],
    "482000": [
        "train", "rail", "amtrak", "eurostar", "railway",
        "high-speed rail", "intercity rail", "sncf", "deutsche bahn",
    ],
    "485310": [
        "taxi", "uber", "lyft", "rideshare", "ride-hailing",
        "cab", "ride share", "grab", "bolt", "didi",
    ],
    "532100": [
        "rental", "car rental", "hertz", "avis", "enterprise",
        "budget car", "sixt", "alamo", "national car",
    ],
    "721100": [
        "hotel", "accommodation", "marriott", "hilton", "hyatt",
        "intercontinental", "sheraton", "holiday inn", "best western",
        "radisson", "wyndham", "accor", "four seasons", "ritz",
        "lodging", "motel", "room night",
    ],
    "485000": [
        "bus", "coach", "transit", "greyhound", "megabus",
        "flixbus", "national express",
    ],
    "483000": [
        "ferry", "boat", "cruise", "water taxi", "ship",
        "vessel", "maritime",
    ],
    "722500": [
        "meal", "restaurant", "dining", "food", "catering",
        "per diem", "breakfast", "lunch", "dinner",
    ],
}


# ==============================================================================
# SpendBasedCalculatorEngine
# ==============================================================================


class SpendBasedCalculatorEngine:
    """
    Spend-based emissions calculator using EEIO factors.

    Implements the spend-based calculation method for GHG Protocol Scope 3
    Category 6 (Business Travel). This is the fallback method when activity
    data (distance, fuel, supplier-specific) is unavailable.

    Calculation Steps:
        1. Validate input (amount > 0, NAICS code in EEIO_FACTORS)
        2. Convert currency to USD using stored exchange rates
        3. Apply CPI deflation to normalize to base year (2021)
        4. Optionally remove profit margin to isolate service cost
        5. Look up EEIO factor (kgCO2e/USD) by NAICS code
        6. Calculate emissions: co2e = adjusted_spend x eeio_factor
        7. Record provenance hash for audit trail
        8. Record Prometheus metrics

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.
        All state mutation is protected by the lock.

    Data Quality:
        Spend-based estimates are Tier 3 (lowest accuracy). The GHG Protocol
        recommends using supplier-specific or distance-based methods when
        possible and limiting spend-based to < 30% of total Category 6.

    Attributes:
        _config: Business travel configuration (spend section)
        _metrics: Prometheus metrics tracker
        _calculation_count: Running count of calculations performed

    Example:
        >>> engine = SpendBasedCalculatorEngine.get_instance()
        >>> result = engine.calculate(SpendInput(
        ...     naics_code="481000",
        ...     amount=Decimal("5000.00"),
        ...     currency=CurrencyCode.USD,
        ...     reporting_year=2024
        ... ))
        >>> result.co2e > Decimal("0")
        True
    """

    _instance: Optional["SpendBasedCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize SpendBasedCalculatorEngine with config and metrics."""
        self._config = get_config()
        self._metrics = get_metrics()
        self._calculation_count: int = 0
        self._batch_count: int = 0

        logger.info(
            "SpendBasedCalculatorEngine initialized: version=%s, "
            "base_year=%d, margin_removal=%s, cpi_deflation=%s",
            ENGINE_VERSION,
            self._config.spend.base_year,
            self._config.spend.enable_margin_removal,
            self._config.spend.enable_cpi_deflation,
        )

    @classmethod
    def get_instance(cls) -> "SpendBasedCalculatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            SpendBasedCalculatorEngine singleton instance.

        Example:
            >>> engine = SpendBasedCalculatorEngine.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level lock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("SpendBasedCalculatorEngine singleton reset")

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def calculate(self, spend_input: SpendInput) -> SpendResult:
        """
        Calculate spend-based emissions using EEIO factors.

        Steps:
            1. Validate input (amount > 0, NAICS code exists)
            2. Convert currency to USD
            3. Apply CPI deflation (if enabled)
            4. Apply margin removal (if enabled)
            5. Multiply adjusted spend by EEIO factor
            6. Compute provenance hash
            7. Record metrics

        Args:
            spend_input: Validated spend input with NAICS code, amount,
                currency, and reporting year.

        Returns:
            SpendResult with emissions in kgCO2e, provenance hash,
            and intermediate calculation values.

        Raises:
            ValueError: If NAICS code not found in EEIO_FACTORS.
            ValueError: If currency conversion or CPI deflation fails.

        Example:
            >>> result = engine.calculate(SpendInput(
            ...     naics_code="721100",
            ...     amount=Decimal("2000.00"),
            ...     currency=CurrencyCode.GBP,
            ...     reporting_year=2024
            ... ))
            >>> result.co2e > Decimal("0")
            True
        """
        start_time = time.monotonic()

        # Step 1: Validate NAICS code
        self._validate_naics_code(spend_input.naics_code)

        # Step 2: Convert currency to USD
        spend_usd = self._convert_to_usd(
            spend_input.amount, spend_input.currency
        )

        logger.debug(
            "Currency conversion: %s %s -> %s USD",
            spend_input.amount,
            spend_input.currency.value,
            spend_usd,
        )

        # Step 3: Apply CPI deflation to normalize to base year
        cpi_deflator = self._get_cpi_deflator(spend_input.reporting_year)
        deflated_spend = self._apply_cpi_deflation(spend_usd, cpi_deflator)

        logger.debug(
            "CPI deflation: %s USD (nominal) / %s (deflator) = %s USD (real %d)",
            spend_usd,
            cpi_deflator,
            deflated_spend,
            self._config.spend.base_year,
        )

        # Step 4: Apply margin removal (if enabled)
        adjusted_spend = self._apply_margin_removal(deflated_spend)

        if self._config.spend.enable_margin_removal:
            logger.debug(
                "Margin removal: %s USD x (1 - %s) = %s USD",
                deflated_spend,
                self._config.spend.default_margin_rate,
                adjusted_spend,
            )

        # Step 5: Look up EEIO factor
        eeio_factor = self._lookup_eeio_factor(spend_input.naics_code)

        # Step 6: Calculate emissions (kgCO2e)
        co2e = (adjusted_spend * eeio_factor).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        logger.debug(
            "Emissions: %s USD x %s kgCO2e/USD = %s kgCO2e",
            adjusted_spend,
            eeio_factor,
            co2e,
        )

        # Step 7: Compute provenance hash
        provenance_hash = calculate_provenance_hash(
            spend_input,
            spend_usd,
            cpi_deflator,
            adjusted_spend,
            eeio_factor,
            co2e,
        )

        # Step 8: Record metrics
        duration = time.monotonic() - start_time
        self._record_metrics(
            naics_code=spend_input.naics_code,
            co2e=co2e,
            duration=duration,
            status="success",
        )

        # Increment calculation count
        self._calculation_count += 1

        result = SpendResult(
            naics_code=spend_input.naics_code,
            spend_usd=spend_usd,
            cpi_deflator=cpi_deflator,
            eeio_factor=eeio_factor,
            co2e=co2e,
            ef_source=EFSource.EEIO,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Spend-based calculation complete: NAICS=%s, amount=%s %s, "
            "co2e=%s kgCO2e, duration=%.4fs",
            spend_input.naics_code,
            spend_input.amount,
            spend_input.currency.value,
            co2e,
            duration,
        )

        return result

    def calculate_batch(
        self, inputs: List[SpendInput]
    ) -> List[SpendResult]:
        """
        Calculate spend-based emissions for a batch of inputs.

        Processes each input sequentially, collecting results and logging
        any per-record errors without aborting the entire batch.

        Args:
            inputs: List of SpendInput records.

        Returns:
            List of SpendResult objects. Failed records are excluded
            from results and logged at ERROR level.

        Raises:
            ValueError: If inputs list is empty.

        Example:
            >>> results = engine.calculate_batch([
            ...     SpendInput(naics_code="481000", amount=Decimal("5000"), ...),
            ...     SpendInput(naics_code="721100", amount=Decimal("2000"), ...),
            ... ])
            >>> len(results) == 2
            True
        """
        if not inputs:
            raise ValueError("Batch inputs list cannot be empty")

        start_time = time.monotonic()
        results: List[SpendResult] = []
        error_count = 0

        logger.info(
            "Starting spend-based batch calculation: %d records", len(inputs)
        )

        for idx, spend_input in enumerate(inputs):
            try:
                result = self.calculate(spend_input)
                results.append(result)
            except (ValueError, InvalidOperation) as e:
                error_count += 1
                logger.error(
                    "Batch record %d failed: %s (NAICS=%s, amount=%s %s)",
                    idx,
                    str(e),
                    spend_input.naics_code,
                    spend_input.amount,
                    spend_input.currency.value,
                )

        duration = time.monotonic() - start_time
        self._batch_count += 1

        logger.info(
            "Spend-based batch complete: %d/%d succeeded, %d failed, "
            "duration=%.4fs",
            len(results),
            len(inputs),
            error_count,
            duration,
        )

        return results

    def get_available_naics_codes(self) -> List[Dict[str, Any]]:
        """
        Return all available NAICS codes with category names and EEIO factors.

        Returns:
            List of dictionaries, each containing naics_code, name, and ef.

        Example:
            >>> codes = engine.get_available_naics_codes()
            >>> len(codes) == 10
            True
            >>> codes[0]["naics_code"]
            '481000'
        """
        result = []
        for naics_code, data in sorted(EEIO_FACTORS.items()):
            result.append({
                "naics_code": naics_code,
                "name": data["name"],
                "ef": float(data["ef"]),
                "ef_unit": "kgCO2e/USD",
                "ef_source": EFSource.EEIO.value,
            })

        return result

    def classify_expense_category(
        self, description: str
    ) -> Optional[str]:
        """
        Classify an expense description to a NAICS code using keyword matching.

        This is a simple rule-based classifier that checks the lowercase
        description against known keywords for each NAICS code. Returns
        the first matching NAICS code or None if no match is found.

        Priority order: air > rail > taxi > rental > hotel > bus > ferry > meal

        Note:
            This is a convenience method for initial categorization. For
            production use, descriptions should be manually verified or
            processed through a more sophisticated NLP pipeline.

        Args:
            description: Expense description text (e.g., "United Airlines
                flight to London", "Marriott hotel 3 nights").

        Returns:
            Matching NAICS code string, or None if no keywords matched.

        Example:
            >>> engine.classify_expense_category("Delta flight to Chicago")
            '481000'
            >>> engine.classify_expense_category("Marriott hotel stay")
            '721100'
            >>> engine.classify_expense_category("office supplies")
            None
        """
        if not description:
            return None

        desc_lower = description.lower().strip()

        # Check each NAICS code's keywords in priority order
        priority_order = [
            "481000",  # Air transportation
            "482000",  # Rail transportation
            "485310",  # Taxi / ride-hailing
            "532100",  # Car rental
            "721100",  # Hotels
            "485000",  # Bus / coach / transit
            "483000",  # Ferry / water
            "722500",  # Restaurant / meals
        ]

        for naics_code in priority_order:
            keywords = _EXPENSE_KEYWORDS.get(naics_code, [])
            for keyword in keywords:
                if keyword in desc_lower:
                    logger.debug(
                        "Expense classified: '%s' -> NAICS %s (keyword: '%s')",
                        description[:50],
                        naics_code,
                        keyword,
                    )
                    return naics_code

        logger.debug(
            "Expense not classified: '%s' (no keyword match)",
            description[:50],
        )
        return None

    def get_calculation_stats(self) -> Dict[str, Any]:
        """
        Return engine calculation statistics.

        Returns:
            Dictionary with calculation_count, batch_count, engine_id,
            and engine_version.

        Example:
            >>> stats = engine.get_calculation_stats()
            >>> stats["engine_id"]
            'spend_based_calculator_engine'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "config": {
                "base_year": self._config.spend.base_year,
                "enable_cpi_deflation": self._config.spend.enable_cpi_deflation,
                "enable_margin_removal": self._config.spend.enable_margin_removal,
                "default_margin_rate": str(
                    self._config.spend.default_margin_rate
                ),
                "default_currency": self._config.spend.default_currency,
            },
        }

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _validate_naics_code(self, naics_code: str) -> None:
        """
        Validate that a NAICS code exists in the EEIO factor table.

        Args:
            naics_code: NAICS industry code string.

        Raises:
            ValueError: If naics_code is not found in EEIO_FACTORS.
        """
        if naics_code not in EEIO_FACTORS:
            available = sorted(EEIO_FACTORS.keys())
            raise ValueError(
                f"NAICS code '{naics_code}' not found in EEIO_FACTORS. "
                f"Available codes: {available}"
            )

    def _convert_to_usd(
        self, amount: Decimal, currency: CurrencyCode
    ) -> Decimal:
        """
        Convert spend amount from source currency to USD.

        Args:
            amount: Spend amount in source currency.
            currency: Source currency code.

        Returns:
            Amount in USD, quantized to 8 decimal places.

        Raises:
            ValueError: If currency is not supported.
        """
        return convert_currency_to_usd(amount, currency)

    def _get_cpi_deflator(self, reporting_year: int) -> Decimal:
        """
        Get CPI deflator for the reporting year.

        If CPI deflation is disabled in config, returns Decimal("1.0")
        (no adjustment).

        Args:
            reporting_year: Year the spend was incurred.

        Returns:
            CPI deflator value (base year = 1.0).

        Raises:
            ValueError: If year not in CPI_DEFLATORS and deflation is enabled.
        """
        if not self._config.spend.enable_cpi_deflation:
            return Decimal("1.00000000")

        return get_cpi_deflator(
            reporting_year, base_year=self._config.spend.base_year
        )

    def _apply_cpi_deflation(
        self, spend_usd: Decimal, cpi_deflator: Decimal
    ) -> Decimal:
        """
        Apply CPI deflation to convert nominal USD to real (base-year) USD.

        Formula: real_usd = nominal_usd / cpi_deflator

        The EEIO factors are denominated in base-year (2021) dollars,
        so nominal spend must be deflated before multiplication.

        Args:
            spend_usd: Nominal spend in USD.
            cpi_deflator: CPI deflator for the reporting year.

        Returns:
            Real (deflated) spend in base-year USD.
        """
        if cpi_deflator == Decimal("0"):
            raise ValueError("CPI deflator cannot be zero")

        deflated = (spend_usd / cpi_deflator).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        return deflated

    def _apply_margin_removal(self, spend_usd: Decimal) -> Decimal:
        """
        Optionally remove profit margin from spend amount.

        If margin removal is enabled in config, applies:
            adjusted = spend x (1 - margin_rate)

        This isolates the cost of the service (fuel, energy, labor) from
        the supplier's profit margin, improving emission estimate accuracy.

        Args:
            spend_usd: Spend amount in USD (possibly already deflated).

        Returns:
            Adjusted spend with margin removed (or unchanged if disabled).
        """
        if not self._config.spend.enable_margin_removal:
            return spend_usd

        margin_rate = self._config.spend.default_margin_rate
        adjusted = (
            spend_usd * (Decimal("1") - margin_rate)
        ).quantize(_QUANT_8DP, rounding=ROUNDING)
        return adjusted

    def _lookup_eeio_factor(self, naics_code: str) -> Decimal:
        """
        Look up EEIO emission factor by NAICS code.

        Args:
            naics_code: NAICS industry code (already validated).

        Returns:
            EEIO factor in kgCO2e/USD.
        """
        entry = EEIO_FACTORS[naics_code]
        return entry["ef"]

    def _record_metrics(
        self,
        naics_code: str,
        co2e: Decimal,
        duration: float,
        status: str,
    ) -> None:
        """
        Record Prometheus metrics for the calculation.

        Args:
            naics_code: NAICS code used.
            co2e: Calculated emissions (kgCO2e).
            duration: Calculation duration in seconds.
            status: Calculation status ("success" or "error").
        """
        try:
            self._metrics.record_calculation(
                method="spend_based",
                mode="spend",
                status=status,
                duration=duration,
                co2e=float(co2e),
            )
        except Exception as e:
            # Metrics recording should never break the calculation
            logger.warning(
                "Failed to record metrics: %s", str(e)
            )


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_spend_based_calculator() -> SpendBasedCalculatorEngine:
    """
    Get the SpendBasedCalculatorEngine singleton instance.

    Convenience function that delegates to the class-level get_instance().

    Returns:
        SpendBasedCalculatorEngine singleton.

    Example:
        >>> engine = get_spend_based_calculator()
        >>> engine.get_calculation_stats()["engine_id"]
        'spend_based_calculator_engine'
    """
    return SpendBasedCalculatorEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "SpendBasedCalculatorEngine",
    "get_spend_based_calculator",
]
