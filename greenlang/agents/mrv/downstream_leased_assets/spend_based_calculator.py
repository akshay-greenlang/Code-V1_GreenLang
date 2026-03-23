# -*- coding: utf-8 -*-
"""
SpendBasedCalculatorEngine - AGENT-MRV-026 Engine 4

GHG Protocol Scope 3 Category 13 spend-based (Tier 3) emissions calculator
using Environmentally Extended Input-Output (EEIO) factors keyed to NAICS
leasing industry codes.

This engine provides the lowest-accuracy but most widely applicable fallback
when neither asset-specific metered data (Tier 1) nor average-data benchmarks
(Tier 2) are available. It converts lease revenue into emissions using
EEIO emission intensity factors denominated in kgCO2e per USD of output.

Calculation Pipeline:
    1. **NAICS Lookup**: Map lease activity to one of 10 NAICS leasing codes
       (531110-541512) covering residential, commercial, vehicle, equipment,
       and IT leasing sectors.

    2. **Currency Conversion**: Convert 12 currencies to USD using stored
       mid-market exchange rates.

    3. **CPI Deflation**: Deflate nominal revenue to base-year (2020) real
       USD using US BLS CPI-U deflators (2020-2026).

    4. **Margin Adjustment**: Apply industry-specific margin adjustment to
       convert gross revenue to service-cost basis.

    5. **EEIO Multiplication**: Multiply adjusted spend by NAICS-specific
       EEIO factor to obtain kgCO2e.

Formula:
    E = lease_revenue x EEIO_factor(NAICS) x (1 + margin_adjustment)
        x CPI_deflation_factor

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern for concurrent pipeline use.

Data Quality:
    Spend-based is Tier 3 (lowest accuracy). Default uncertainty is +/-50%.
    Organizations should limit spend-based to < 30% of Category 13 total
    and prioritize asset-specific or average-data methods.

References:
    - GHG Protocol Technical Guidance for Scope 3 Category 13
    - US EPA USEEIO v2.0 Supply Chain Emission Factors
    - US BLS CPI-U (Consumer Price Index for All Urban Consumers)
    - NAICS 2022 Classification Manual

Example:
    >>> engine = get_spend_based_calculator()
    >>> result = engine.calculate({
    ...     "naics_code": "531120",
    ...     "lease_revenue": Decimal("100000"),
    ...     "currency": "USD",
    ...     "reporting_year": 2024,
    ... })
    >>> result["co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "spend_based_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-013"

# Decimal precision
PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_2DP: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_HUNDRED: Decimal = Decimal("100")

# Default uncertainty for Tier 3 spend-based method (+/-50%)
TIER_3_UNCERTAINTY: Decimal = Decimal("0.50")

# Default base year for CPI deflation
DEFAULT_BASE_YEAR: int = 2020


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for lease revenue conversion."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CHF = "CHF"
    SGD = "SGD"
    BRL = "BRL"
    ZAR = "ZAR"


class LeaseCategory(str, Enum):
    """Lease asset categories for NAICS classification."""

    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    MINIWAREHOUSE = "miniwarehouse"
    PASSENGER_CAR = "passenger_car"
    TRUCK = "truck"
    GENERAL_RENTAL = "general_rental"
    CONSTRUCTION_EQUIPMENT = "construction_equipment"
    OTHER_COMMERCIAL = "other_commercial"
    DATA_PROCESSING = "data_processing"
    COMPUTER_SYSTEMS = "computer_systems"


# ==============================================================================
# EMBEDDED DATA TABLES
# ==============================================================================

# EEIO factors for downstream leased assets (kgCO2e per USD of output)
# Source: EPA USEEIO v2.0 / EXIOBASE 3, mapped to NAICS 2022 leasing codes
EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {
        "name": "Lessors of residential buildings and dwellings",
        "category": LeaseCategory.RESIDENTIAL.value,
        "ef": Decimal("0.15"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "building",
    },
    "531120": {
        "name": "Lessors of nonresidential buildings (except miniwarehouses)",
        "category": LeaseCategory.COMMERCIAL.value,
        "ef": Decimal("0.18"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "building",
    },
    "531130": {
        "name": "Lessors of miniwarehouses and self-storage units",
        "category": LeaseCategory.MINIWAREHOUSE.value,
        "ef": Decimal("0.12"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "building",
    },
    "532111": {
        "name": "Passenger car rental and leasing",
        "category": LeaseCategory.PASSENGER_CAR.value,
        "ef": Decimal("0.22"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "vehicle",
    },
    "532120": {
        "name": "Truck, utility trailer, and RV rental and leasing",
        "category": LeaseCategory.TRUCK.value,
        "ef": Decimal("0.28"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "vehicle",
    },
    "532310": {
        "name": "General rental centers",
        "category": LeaseCategory.GENERAL_RENTAL.value,
        "ef": Decimal("0.25"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "equipment",
    },
    "532412": {
        "name": "Construction, mining, and forestry machinery rental",
        "category": LeaseCategory.CONSTRUCTION_EQUIPMENT.value,
        "ef": Decimal("0.35"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "equipment",
    },
    "532490": {
        "name": "Other commercial and industrial machinery rental",
        "category": LeaseCategory.OTHER_COMMERCIAL.value,
        "ef": Decimal("0.20"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "equipment",
    },
    "518210": {
        "name": "Data processing, hosting, and related services",
        "category": LeaseCategory.DATA_PROCESSING.value,
        "ef": Decimal("0.30"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "it_asset",
    },
    "541512": {
        "name": "Computer systems design and related services",
        "category": LeaseCategory.COMPUTER_SYSTEMS.value,
        "ef": Decimal("0.10"),
        "ef_unit": "kgCO2e/USD",
        "asset_type": "it_asset",
    },
}

# CPI deflation factors (base year 2020 = 1.000)
# Source: US BLS CPI-U (Consumer Price Index for All Urban Consumers)
CPI_DEFLATORS: Dict[int, Decimal] = {
    2020: Decimal("1.000"),
    2021: Decimal("1.047"),
    2022: Decimal("1.128"),
    2023: Decimal("1.175"),
    2024: Decimal("1.212"),
    2025: Decimal("1.243"),
    2026: Decimal("1.268"),
}

# Currency exchange rates to USD (approximate mid-market rates)
# Source: ECB / Federal Reserve / Bloomberg mid-market
CURRENCY_RATES: Dict[str, Decimal] = {
    CurrencyCode.USD.value: Decimal("1.0000"),
    CurrencyCode.EUR.value: Decimal("1.0850"),
    CurrencyCode.GBP.value: Decimal("1.2650"),
    CurrencyCode.CAD.value: Decimal("0.7410"),
    CurrencyCode.AUD.value: Decimal("0.6520"),
    CurrencyCode.JPY.value: Decimal("0.006667"),
    CurrencyCode.CNY.value: Decimal("0.1378"),
    CurrencyCode.INR.value: Decimal("0.01198"),
    CurrencyCode.CHF.value: Decimal("1.1280"),
    CurrencyCode.SGD.value: Decimal("0.7440"),
    CurrencyCode.BRL.value: Decimal("0.1990"),
    CurrencyCode.ZAR.value: Decimal("0.05340"),
}

# Margin adjustment factors by industry (fraction to add to revenue)
# These account for the gap between gross revenue and the actual economic
# activity that generates emissions. Negative values reduce effective spend.
MARGIN_ADJUSTMENT_FACTORS: Dict[str, Decimal] = {
    "531110": Decimal("-0.10"),   # Residential: lower operating margins
    "531120": Decimal("-0.08"),   # Commercial office: moderate margins
    "531130": Decimal("-0.05"),   # Miniwarehouse: low overhead
    "532111": Decimal("-0.15"),   # Car rental: high margin sector
    "532120": Decimal("-0.12"),   # Truck rental: moderate margin
    "532310": Decimal("-0.10"),   # General rental: average margin
    "532412": Decimal("-0.08"),   # Construction equip: lower margin
    "532490": Decimal("-0.10"),   # Other commercial: average margin
    "518210": Decimal("-0.20"),   # Data processing: high margin tech
    "541512": Decimal("-0.25"),   # Computer systems: very high margin
}

# Keyword mappings for NAICS classification of lease descriptions
_LEASE_KEYWORDS: Dict[str, List[str]] = {
    "531110": [
        "residential", "apartment", "condo", "dwelling", "housing",
        "flat", "townhouse", "duplex", "rental home", "multifamily",
    ],
    "531120": [
        "office", "commercial", "retail space", "shopping center",
        "plaza", "business park", "industrial park", "nonresidential",
        "commercial building", "office building", "mall",
    ],
    "531130": [
        "storage", "miniwarehouse", "self-storage", "warehouse unit",
        "storage locker", "mini warehouse",
    ],
    "532111": [
        "car rental", "car lease", "passenger car", "vehicle lease",
        "auto lease", "sedan", "compact car", "economy car",
    ],
    "532120": [
        "truck", "van rental", "utility trailer", "rv rental",
        "commercial vehicle", "cargo van", "box truck",
    ],
    "532310": [
        "general rental", "tool rental", "equipment rental",
        "party rental", "event equipment",
    ],
    "532412": [
        "construction equipment", "excavator", "bulldozer", "crane",
        "backhoe", "loader", "mining equipment", "forestry",
    ],
    "532490": [
        "industrial machinery", "commercial equipment", "compressor",
        "generator rental", "manufacturing equipment",
    ],
    "518210": [
        "data center", "server hosting", "colocation", "cloud",
        "data processing", "managed hosting", "rack space",
    ],
    "541512": [
        "computer", "laptop", "desktop lease", "it equipment",
        "computer systems", "it lease", "device lease",
    ],
}

# DQI dimension weights for Tier 3 scoring
DQI_WEIGHTS_TIER3: Dict[str, Decimal] = {
    "representativeness": Decimal("0.30"),
    "completeness": Decimal("0.25"),
    "temporal": Decimal("0.15"),
    "geographical": Decimal("0.15"),
    "technological": Decimal("0.15"),
}

# Default DQI scores for Tier 3 spend-based method
DEFAULT_DQI_SCORES_TIER3: Dict[str, Decimal] = {
    "representativeness": Decimal("2"),   # Sector average, not asset-specific
    "completeness": Decimal("2"),         # Revenue-only, no activity data
    "temporal": Decimal("2"),             # EEIO factors may lag
    "geographical": Decimal("2"),         # US-centric EEIO
    "technological": Decimal("2"),        # No technology differentiation
}


# ==============================================================================
# PROVENANCE HELPER
# ==============================================================================


def _calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, dict):
            hash_input += json.dumps(inp, sort_keys=True, default=str)
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUNDING))
        elif isinstance(inp, (list, tuple)):
            hash_input += json.dumps(
                [str(x) if isinstance(x, Decimal) else x for x in inp],
                sort_keys=True,
                default=str,
            )
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# SpendBasedCalculatorEngine
# ==============================================================================


class SpendBasedCalculatorEngine:
    """
    Spend-based (Tier 3) EEIO emissions calculator for downstream leased assets.

    Implements the spend-based calculation method for GHG Protocol Scope 3
    Category 13 (Downstream Leased Assets). This is the fallback method when
    neither metered energy data (Tier 1) nor benchmark activity data (Tier 2)
    is available.

    Calculation Steps:
        1. Validate input (revenue > 0, NAICS code in EEIO_FACTORS)
        2. Convert currency to USD using stored exchange rates
        3. Apply CPI deflation to normalize to base year (2020)
        4. Apply margin adjustment to isolate service cost
        5. Look up EEIO factor (kgCO2e/USD) by NAICS code
        6. Calculate emissions: co2e = adjusted_revenue x eeio_factor
        7. Record provenance hash for audit trail

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Data Quality:
        Spend-based estimates are Tier 3 (lowest accuracy). Default uncertainty
        is +/-50%. The GHG Protocol recommends limiting spend-based to < 30%
        of total Category 13 emissions.

    Attributes:
        _calculation_count: Running count of calculations performed
        _batch_count: Running count of batch calculations performed
        _enable_cpi_deflation: Whether CPI deflation is applied
        _enable_margin_adjustment: Whether margin adjustment is applied
        _base_year: Base year for CPI deflation

    Example:
        >>> engine = SpendBasedCalculatorEngine.get_instance()
        >>> result = engine.calculate({
        ...     "naics_code": "531120",
        ...     "lease_revenue": Decimal("100000"),
        ...     "currency": "USD",
        ...     "reporting_year": 2024,
        ... })
        >>> result["co2e_kg"] > Decimal("0")
        True
    """

    _instance: Optional["SpendBasedCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize SpendBasedCalculatorEngine."""
        self._calculation_count: int = 0
        self._batch_count: int = 0
        self._enable_cpi_deflation: bool = True
        self._enable_margin_adjustment: bool = True
        self._base_year: int = DEFAULT_BASE_YEAR
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "SpendBasedCalculatorEngine initialized: version=%s, agent=%s, "
            "base_year=%d, cpi_deflation=%s, margin_adjustment=%s",
            ENGINE_VERSION,
            AGENT_ID,
            self._base_year,
            self._enable_cpi_deflation,
            self._enable_margin_adjustment,
        )

    @classmethod
    def get_instance(cls) -> "SpendBasedCalculatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            SpendBasedCalculatorEngine singleton instance.
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

    def calculate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate spend-based emissions using EEIO factors.

        Steps:
            1. Validate input (revenue > 0, NAICS code exists)
            2. Convert currency to USD
            3. Apply CPI deflation (if enabled)
            4. Apply margin adjustment (if enabled)
            5. Multiply adjusted revenue by EEIO factor
            6. Compute provenance hash

        Args:
            input_data: Dictionary containing:
                - naics_code (str): NAICS leasing industry code
                - lease_revenue (Decimal or numeric): Lease revenue amount
                - currency (str, optional): ISO 4217 code (default "USD")
                - reporting_year (int, optional): Year for CPI deflation
                  (default 2024)
                - description (str, optional): Lease description for
                  auto-classification

        Returns:
            Dictionary with co2e_kg, naics_code, revenue_usd,
            cpi_deflator, margin_adjustment, eeio_factor, adjusted_revenue,
            provenance_hash, dqi_score, uncertainty.

        Raises:
            ValueError: If NAICS code not found or revenue is invalid.

        Example:
            >>> result = engine.calculate({
            ...     "naics_code": "531120",
            ...     "lease_revenue": Decimal("50000"),
            ...     "currency": "EUR",
            ...     "reporting_year": 2024,
            ... })
        """
        start_time = time.monotonic()

        # Step 1: Validate inputs
        self._validate_calculate_inputs(input_data)

        naics_code = input_data["naics_code"]
        lease_revenue = Decimal(str(input_data["lease_revenue"]))
        currency = input_data.get("currency", CurrencyCode.USD.value).upper()
        reporting_year = int(input_data.get("reporting_year", 2024))

        # Step 2: Convert currency to USD
        revenue_usd = self.apply_currency_conversion(lease_revenue, currency)

        logger.debug(
            "Currency conversion: %s %s -> %s USD",
            lease_revenue, currency, revenue_usd,
        )

        # Step 3: Apply CPI deflation
        cpi_deflator = self._get_cpi_deflator(reporting_year)
        deflated_revenue = self.apply_cpi_deflation(revenue_usd, cpi_deflator)

        logger.debug(
            "CPI deflation: %s USD (nominal) / %s = %s USD (real %d)",
            revenue_usd, cpi_deflator, deflated_revenue, self._base_year,
        )

        # Step 4: Apply margin adjustment
        margin_factor = self._get_margin_factor(naics_code)
        adjusted_revenue = self.apply_margin_adjustment(
            deflated_revenue, margin_factor
        )

        logger.debug(
            "Margin adjustment: %s USD x (1 + %s) = %s USD",
            deflated_revenue, margin_factor, adjusted_revenue,
        )

        # Step 5: Look up EEIO factor and calculate
        eeio_factor = self.get_eeio_factor(naics_code)
        co2e_kg = (adjusted_revenue * eeio_factor).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        logger.debug(
            "Emissions: %s USD x %s kgCO2e/USD = %s kgCO2e",
            adjusted_revenue, eeio_factor, co2e_kg,
        )

        # Step 6: DQI and uncertainty
        dqi_score = self.compute_dqi_score(input_data)
        uncertainty = self.compute_uncertainty(co2e_kg)

        # Step 7: Provenance hash
        provenance_hash = _calculate_provenance_hash(
            naics_code, str(lease_revenue), currency,
            str(reporting_year), str(revenue_usd), str(cpi_deflator),
            str(margin_factor), str(adjusted_revenue),
            str(eeio_factor), str(co2e_kg),
        )

        # Metadata
        duration = time.monotonic() - start_time
        self._calculation_count += 1

        eeio_entry = EEIO_FACTORS[naics_code]

        result = {
            "naics_code": naics_code,
            "naics_name": eeio_entry["name"],
            "asset_type": eeio_entry.get("asset_type", "unknown"),
            "lease_revenue_original": str(lease_revenue),
            "currency": currency,
            "revenue_usd": str(revenue_usd),
            "reporting_year": reporting_year,
            "cpi_deflator": str(cpi_deflator),
            "deflated_revenue_usd": str(deflated_revenue),
            "margin_adjustment": str(margin_factor),
            "adjusted_revenue_usd": str(adjusted_revenue),
            "eeio_factor_kg_per_usd": str(eeio_factor),
            "co2e_kg": co2e_kg,
            "calculation_method": "spend_based",
            "data_quality_tier": "tier_3",
            "dqi_score": dqi_score,
            "uncertainty_pct": str(uncertainty),
            "uncertainty_lower_kg": str(
                (co2e_kg * (_ONE - uncertainty)).quantize(
                    _QUANT_8DP, rounding=ROUNDING
                )
            ),
            "uncertainty_upper_kg": str(
                (co2e_kg * (_ONE + uncertainty)).quantize(
                    _QUANT_8DP, rounding=ROUNDING
                )
            ),
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "processing_time_ms": round(duration * 1000, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Spend-based calculation complete: NAICS=%s, revenue=%s %s, "
            "co2e=%s kgCO2e, duration=%.4fs",
            naics_code, lease_revenue, currency, co2e_kg, duration,
        )

        return result

    def calculate_by_naics(
        self,
        naics_code: str,
        lease_revenue: Decimal,
        currency: str = "USD",
        reporting_year: int = 2024,
    ) -> Dict[str, Any]:
        """
        Convenience method: calculate by explicit NAICS code.

        Args:
            naics_code: NAICS leasing industry code.
            lease_revenue: Lease revenue amount.
            currency: ISO 4217 currency code (default "USD").
            reporting_year: Reporting year (default 2024).

        Returns:
            Same as calculate().

        Example:
            >>> result = engine.calculate_by_naics(
            ...     "531120", Decimal("100000"), "USD", 2024
            ... )
        """
        return self.calculate({
            "naics_code": naics_code,
            "lease_revenue": lease_revenue,
            "currency": currency,
            "reporting_year": reporting_year,
        })

    def calculate_portfolio(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate spend-based emissions for a portfolio of leases.

        Processes each lease input, aggregates results, and provides
        portfolio-level totals with breakdown by NAICS code and asset type.

        Args:
            inputs: List of input dictionaries (each with naics_code,
                lease_revenue, etc.).

        Returns:
            Dictionary with total_co2e_kg, results (list), by_naics (dict),
            by_asset_type (dict), error_count, provenance_hash.

        Raises:
            ValueError: If inputs list is empty.

        Example:
            >>> portfolio = engine.calculate_portfolio([
            ...     {"naics_code": "531120", "lease_revenue": Decimal("100000")},
            ...     {"naics_code": "532111", "lease_revenue": Decimal("50000")},
            ... ])
            >>> portfolio["total_co2e_kg"] > Decimal("0")
            True
        """
        if not inputs:
            raise ValueError("Portfolio inputs list cannot be empty")

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        by_naics: Dict[str, Decimal] = {}
        by_asset_type: Dict[str, Decimal] = {}
        total_co2e = _ZERO

        logger.info(
            "Starting portfolio calculation: %d leases", len(inputs)
        )

        for idx, input_data in enumerate(inputs):
            try:
                result = self.calculate(input_data)
                results.append(result)

                co2e = result["co2e_kg"]
                total_co2e += co2e

                # Aggregate by NAICS
                nc = result["naics_code"]
                by_naics[nc] = by_naics.get(nc, _ZERO) + co2e

                # Aggregate by asset type
                at = result.get("asset_type", "unknown")
                by_asset_type[at] = by_asset_type.get(at, _ZERO) + co2e

            except (ValueError, InvalidOperation, KeyError) as e:
                errors.append({
                    "index": idx,
                    "error": str(e),
                    "naics_code": input_data.get("naics_code", "unknown"),
                })
                logger.error(
                    "Portfolio lease %d failed: %s (NAICS=%s)",
                    idx, str(e), input_data.get("naics_code", "unknown"),
                )

        total_co2e = total_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)

        # Portfolio provenance hash
        portfolio_hash = _calculate_provenance_hash(
            str(total_co2e),
            str(len(results)),
            json.dumps(
                {k: str(v) for k, v in sorted(by_naics.items())},
                sort_keys=True,
            ),
        )

        duration = time.monotonic() - start_time
        self._batch_count += 1

        logger.info(
            "Portfolio calculation complete: %d/%d succeeded, %d failed, "
            "total_co2e=%s kgCO2e, duration=%.4fs",
            len(results), len(inputs), len(errors), total_co2e, duration,
        )

        return {
            "total_co2e_kg": total_co2e,
            "lease_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
            "by_naics": {k: str(v) for k, v in sorted(by_naics.items())},
            "by_asset_type": {
                k: str(v) for k, v in sorted(by_asset_type.items())
            },
            "provenance_hash": portfolio_hash,
            "processing_time_ms": round(duration * 1000, 4),
        }

    def apply_cpi_deflation(
        self, revenue_usd: Decimal, cpi_deflator: Decimal
    ) -> Decimal:
        """
        Apply CPI deflation to convert nominal USD to real (base-year) USD.

        Formula: real_usd = nominal_usd / cpi_deflator

        The EEIO factors are denominated in base-year dollars, so nominal
        revenue must be deflated before multiplication.

        Args:
            revenue_usd: Nominal revenue in USD.
            cpi_deflator: CPI deflator for the reporting year.

        Returns:
            Real (deflated) revenue in base-year USD.

        Raises:
            ValueError: If CPI deflator is zero.
        """
        if not self._enable_cpi_deflation:
            return revenue_usd

        if cpi_deflator == _ZERO:
            raise ValueError("CPI deflator cannot be zero")

        return (revenue_usd / cpi_deflator).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

    def apply_currency_conversion(
        self, amount: Decimal, currency: str
    ) -> Decimal:
        """
        Convert amount from source currency to USD.

        Args:
            amount: Amount in source currency.
            currency: ISO 4217 currency code.

        Returns:
            Amount in USD, quantized to 8 decimal places.

        Raises:
            ValueError: If currency is not supported.

        Example:
            >>> engine.apply_currency_conversion(Decimal("1000"), "EUR")
            Decimal('1085.00000000')
        """
        currency_upper = currency.upper()
        rate = CURRENCY_RATES.get(currency_upper)
        if rate is None:
            raise ValueError(
                f"Currency '{currency}' not supported. "
                f"Available: {sorted(CURRENCY_RATES.keys())}"
            )

        return (amount * rate).quantize(_QUANT_8DP, rounding=ROUNDING)

    def apply_margin_adjustment(
        self, revenue_usd: Decimal, margin_factor: Decimal
    ) -> Decimal:
        """
        Apply margin adjustment to convert gross revenue to cost basis.

        Formula: adjusted = revenue x (1 + margin_factor)

        Negative margin_factor reduces effective spend (removes profit margin).
        Positive margin_factor increases effective spend.

        Args:
            revenue_usd: Revenue in USD (possibly already deflated).
            margin_factor: Margin adjustment factor (e.g., -0.10 for -10%).

        Returns:
            Adjusted revenue.
        """
        if not self._enable_margin_adjustment:
            return revenue_usd

        adjusted = (revenue_usd * (_ONE + margin_factor)).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        return adjusted

    def get_eeio_factor(self, naics_code: str) -> Decimal:
        """
        Look up EEIO emission factor by NAICS code.

        Args:
            naics_code: NAICS leasing industry code.

        Returns:
            EEIO factor in kgCO2e/USD.

        Raises:
            ValueError: If NAICS code is not found.

        Example:
            >>> engine.get_eeio_factor("531120")
            Decimal('0.18')
        """
        entry = EEIO_FACTORS.get(naics_code)
        if entry is None:
            raise ValueError(
                f"NAICS code '{naics_code}' not found in EEIO_FACTORS. "
                f"Available: {sorted(EEIO_FACTORS.keys())}"
            )
        return entry["ef"]

    def classify_lease(self, description: str) -> Optional[str]:
        """
        Classify a lease description to a NAICS code using keyword matching.

        Priority order follows the NAICS code order (buildings first,
        then vehicles, equipment, IT).

        Args:
            description: Lease description text.

        Returns:
            Matching NAICS code string, or None if no keywords matched.

        Example:
            >>> engine.classify_lease("office building lease downtown")
            '531120'
            >>> engine.classify_lease("excavator rental 3 months")
            '532412'
        """
        if not description:
            return None

        desc_lower = description.lower().strip()

        # Check each NAICS code's keywords in priority order
        priority_order = sorted(EEIO_FACTORS.keys())

        for naics_code in priority_order:
            keywords = _LEASE_KEYWORDS.get(naics_code, [])
            for keyword in keywords:
                if keyword in desc_lower:
                    logger.debug(
                        "Lease classified: '%s' -> NAICS %s (keyword: '%s')",
                        description[:50], naics_code, keyword,
                    )
                    return naics_code

        logger.debug(
            "Lease not classified: '%s' (no keyword match)",
            description[:50],
        )
        return None

    def get_available_naics_codes(self) -> List[Dict[str, Any]]:
        """
        Return all available NAICS codes with metadata.

        Returns:
            List of dictionaries with naics_code, name, category,
            ef, ef_unit, and asset_type.
        """
        result = []
        for naics_code, data in sorted(EEIO_FACTORS.items()):
            result.append({
                "naics_code": naics_code,
                "name": data["name"],
                "category": data.get("category", "unknown"),
                "ef": str(data["ef"]),
                "ef_unit": data.get("ef_unit", "kgCO2e/USD"),
                "asset_type": data.get("asset_type", "unknown"),
            })
        return result

    def get_available_currencies(self) -> List[Dict[str, str]]:
        """
        Return all available currencies with exchange rates.

        Returns:
            List of dictionaries with currency and rate_to_usd.
        """
        return [
            {"currency": currency, "rate_to_usd": str(rate)}
            for currency, rate in sorted(CURRENCY_RATES.items())
        ]

    def get_cpi_years(self) -> List[Dict[str, str]]:
        """
        Return all available CPI deflation years with factors.

        Returns:
            List of dictionaries with year and deflator.
        """
        return [
            {"year": year, "deflator": str(deflator)}
            for year, deflator in sorted(CPI_DEFLATORS.items())
        ]

    def compute_dqi_score(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute Data Quality Indicator score for Tier 3 method.

        Uses default Tier 3 DQI scores unless overridden in input_data
        via a 'dqi_overrides' dictionary.

        Args:
            input_data: Optional dict with 'dqi_overrides' key.

        Returns:
            Dictionary with dimension scores, weights, composite_score,
            classification, and tier.

        Example:
            >>> dqi = engine.compute_dqi_score()
            >>> dqi["composite_score"]
            Decimal('2.00000000')
            >>> dqi["classification"]
            'Poor'
        """
        scores = dict(DEFAULT_DQI_SCORES_TIER3)

        if input_data and "dqi_overrides" in input_data:
            overrides = input_data["dqi_overrides"]
            for dim, score in overrides.items():
                if dim in scores:
                    scores[dim] = Decimal(str(score))

        weighted_sum = _ZERO
        total_weight = _ZERO
        dimension_results: Dict[str, str] = {}

        for dim, weight in DQI_WEIGHTS_TIER3.items():
            score = scores.get(dim, Decimal("2"))
            weighted_sum += score * weight
            total_weight += weight
            dimension_results[dim] = str(score)

        composite = (weighted_sum / total_weight).quantize(
            _QUANT_8DP, rounding=ROUNDING
        ) if total_weight > _ZERO else Decimal("2.00000000")

        if composite >= Decimal("4.5"):
            classification = "Excellent"
        elif composite >= Decimal("3.5"):
            classification = "Good"
        elif composite >= Decimal("2.5"):
            classification = "Fair"
        elif composite >= Decimal("1.5"):
            classification = "Poor"
        else:
            classification = "Very Poor"

        return {
            "dimensions": dimension_results,
            "weights": {k: str(v) for k, v in DQI_WEIGHTS_TIER3.items()},
            "composite_score": composite,
            "classification": classification,
            "tier": "tier_3",
        }

    def compute_uncertainty(
        self,
        co2e_kg: Decimal,
        custom_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Compute uncertainty range for Tier 3 spend-based method.

        Default uncertainty is +/-50% for spend-based calculations.

        Args:
            co2e_kg: Calculated emissions in kgCO2e.
            custom_pct: Optional custom uncertainty (Decimal, 0-1).

        Returns:
            Uncertainty as a fraction (e.g., Decimal("0.50")).

        Example:
            >>> engine.compute_uncertainty(Decimal("1000"))
            Decimal('0.50')
        """
        if custom_pct is not None:
            pct = Decimal(str(custom_pct))
            if pct < _ZERO or pct > _ONE:
                raise ValueError(
                    f"custom_pct must be between 0 and 1, got {pct}"
                )
            return pct

        return TIER_3_UNCERTAINTY

    def validate_inputs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data and return structured validation result.

        Does NOT raise exceptions; returns is_valid, errors, warnings.

        Args:
            input_data: Input dictionary to validate.

        Returns:
            Dictionary with is_valid (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # NAICS code
        naics_code = input_data.get("naics_code", "")
        if not naics_code:
            errors.append("naics_code is required")
        elif naics_code not in EEIO_FACTORS:
            errors.append(
                f"naics_code '{naics_code}' not found. "
                f"Available: {sorted(EEIO_FACTORS.keys())}"
            )

        # Lease revenue
        revenue = input_data.get("lease_revenue")
        if revenue is None:
            errors.append("lease_revenue is required")
        else:
            try:
                rev_dec = Decimal(str(revenue))
                if rev_dec <= _ZERO:
                    errors.append(
                        f"lease_revenue must be positive, got {rev_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(
                    f"lease_revenue must be numeric, got '{revenue}'"
                )

        # Currency
        currency = input_data.get("currency", "USD").upper()
        if currency not in CURRENCY_RATES:
            errors.append(
                f"Currency '{currency}' not supported. "
                f"Available: {sorted(CURRENCY_RATES.keys())}"
            )

        # Reporting year
        year = input_data.get("reporting_year", 2024)
        try:
            year_int = int(year)
            if year_int not in CPI_DEFLATORS:
                warnings.append(
                    f"CPI deflator not available for year {year_int}. "
                    f"Available: {sorted(CPI_DEFLATORS.keys())}. "
                    f"CPI deflation will be skipped."
                )
        except (ValueError, TypeError):
            errors.append(f"reporting_year must be integer, got '{year}'")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Return engine health status and statistics.

        Returns:
            Dictionary with engine_id, engine_version, status, stats,
            and configuration summary.

        Example:
            >>> health = engine.health_check()
            >>> health["status"]
            'healthy'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "status": "healthy",
            "stats": {
                "calculation_count": self._calculation_count,
                "batch_count": self._batch_count,
            },
            "config": {
                "base_year": self._base_year,
                "enable_cpi_deflation": self._enable_cpi_deflation,
                "enable_margin_adjustment": self._enable_margin_adjustment,
            },
            "data_tables_loaded": {
                "eeio_factors": len(EEIO_FACTORS),
                "currencies": len(CURRENCY_RATES),
                "cpi_years": len(CPI_DEFLATORS),
                "margin_factors": len(MARGIN_ADJUSTMENT_FACTORS),
            },
            "initialized_at": self._initialized_at,
        }

    def calculate_batch(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate spend-based emissions for a batch of lease inputs.

        Processes each input sequentially, collecting results and logging
        per-record errors without aborting the entire batch.

        Args:
            inputs: List of input dictionaries (each with naics_code,
                lease_revenue, etc.).

        Returns:
            List of result dictionaries. Failed records are excluded
            from results and logged at ERROR level.

        Raises:
            ValueError: If inputs list is empty.

        Example:
            >>> results = engine.calculate_batch([
            ...     {"naics_code": "531120", "lease_revenue": Decimal("100000")},
            ...     {"naics_code": "532111", "lease_revenue": Decimal("50000")},
            ... ])
            >>> len(results) == 2
            True
        """
        if not inputs:
            raise ValueError("Batch inputs list cannot be empty")

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        error_count = 0

        logger.info(
            "Starting spend-based batch calculation: %d records", len(inputs)
        )

        for idx, input_data in enumerate(inputs):
            try:
                result = self.calculate(input_data)
                results.append(result)
            except (ValueError, InvalidOperation, KeyError) as e:
                error_count += 1
                logger.error(
                    "Batch record %d failed: %s (NAICS=%s, revenue=%s)",
                    idx,
                    str(e),
                    input_data.get("naics_code", "unknown"),
                    input_data.get("lease_revenue", "unknown"),
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

    def analyze_revenue_breakdown(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze revenue breakdown across NAICS codes and asset types.

        Provides a summary view of the portfolio's lease revenue composition
        and emissions intensity distribution, helping identify where
        higher-accuracy methods should be prioritized.

        Args:
            inputs: List of input dictionaries.

        Returns:
            Dictionary with total_revenue_usd, by_naics, by_asset_type,
            emissions_intensity_ranking, and recommendations.

        Example:
            >>> analysis = engine.analyze_revenue_breakdown([
            ...     {"naics_code": "531120", "lease_revenue": Decimal("200000")},
            ...     {"naics_code": "532412", "lease_revenue": Decimal("50000")},
            ... ])
            >>> analysis["total_revenue_usd"] > Decimal("0")
            True
        """
        if not inputs:
            return {
                "total_revenue_usd": "0",
                "by_naics": {},
                "by_asset_type": {},
                "emissions_intensity_ranking": [],
                "recommendations": [],
            }

        total_revenue = _ZERO
        by_naics: Dict[str, Decimal] = {}
        by_asset_type: Dict[str, Decimal] = {}
        intensity_data: List[Dict[str, Any]] = []

        for input_data in inputs:
            naics_code = input_data.get("naics_code", "")
            revenue = input_data.get("lease_revenue")
            if not naics_code or revenue is None:
                continue

            rev_dec = Decimal(str(revenue))
            currency = input_data.get("currency", "USD").upper()

            # Convert to USD
            try:
                rev_usd = self.apply_currency_conversion(rev_dec, currency)
            except ValueError:
                rev_usd = rev_dec

            total_revenue += rev_usd
            by_naics[naics_code] = by_naics.get(naics_code, _ZERO) + rev_usd

            entry = EEIO_FACTORS.get(naics_code, {})
            asset_type = entry.get("asset_type", "unknown")
            by_asset_type[asset_type] = (
                by_asset_type.get(asset_type, _ZERO) + rev_usd
            )

            ef = entry.get("ef", _ZERO)
            intensity_data.append({
                "naics_code": naics_code,
                "naics_name": entry.get("name", "Unknown"),
                "revenue_usd": str(rev_usd),
                "eeio_factor": str(ef),
                "estimated_co2e": str(
                    (rev_usd * ef).quantize(_QUANT_8DP, rounding=ROUNDING)
                ),
            })

        # Rank by emissions intensity (highest first)
        intensity_data.sort(
            key=lambda x: Decimal(x["estimated_co2e"]),
            reverse=True,
        )

        # Generate recommendations
        recommendations: List[str] = []
        if total_revenue > _ZERO:
            for entry in intensity_data[:3]:
                est = Decimal(entry["estimated_co2e"])
                if est > _ZERO:
                    pct = (est / (total_revenue * Decimal("0.25")) * _HUNDRED
                           ).quantize(_QUANT_2DP, rounding=ROUNDING)
                    recommendations.append(
                        f"NAICS {entry['naics_code']} "
                        f"({entry['naics_name']}): High emissions intensity "
                        f"({entry['eeio_factor']} kgCO2e/USD). "
                        f"Consider upgrading to average-data or "
                        f"asset-specific method."
                    )

        return {
            "total_revenue_usd": str(
                total_revenue.quantize(_QUANT_2DP, rounding=ROUNDING)
            ),
            "by_naics": {
                k: str(v.quantize(_QUANT_2DP, rounding=ROUNDING))
                for k, v in sorted(by_naics.items())
            },
            "by_asset_type": {
                k: str(v.quantize(_QUANT_2DP, rounding=ROUNDING))
                for k, v in sorted(by_asset_type.items())
            },
            "emissions_intensity_ranking": intensity_data,
            "recommendations": recommendations,
        }

    def sensitivity_analysis(
        self,
        naics_code: str,
        lease_revenue: Decimal,
        currency: str = "USD",
        reporting_year: int = 2024,
        ef_variation_pct: Decimal = Decimal("0.20"),
        cpi_variation_pct: Decimal = Decimal("0.05"),
        margin_variation_pct: Decimal = Decimal("0.10"),
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on spend-based calculation.

        Varies key parameters (EEIO factor, CPI deflator, margin) by
        the specified percentages to show the range of possible outcomes.

        Args:
            naics_code: NAICS leasing industry code.
            lease_revenue: Lease revenue amount.
            currency: ISO 4217 currency code.
            reporting_year: Reporting year.
            ef_variation_pct: EEIO factor variation (default +/-20%).
            cpi_variation_pct: CPI deflator variation (default +/-5%).
            margin_variation_pct: Margin variation (default +/-10%).

        Returns:
            Dictionary with base_case, ef_sensitivity, cpi_sensitivity,
            margin_sensitivity, combined_range, and tornado data.

        Example:
            >>> sa = engine.sensitivity_analysis(
            ...     "531120", Decimal("100000"), "USD", 2024
            ... )
            >>> sa["base_case"]["co2e_kg"] > Decimal("0")
            True
        """
        # Base case
        base_result = self.calculate({
            "naics_code": naics_code,
            "lease_revenue": lease_revenue,
            "currency": currency,
            "reporting_year": reporting_year,
        })
        base_co2e = base_result["co2e_kg"]

        # Revenue in USD for sensitivity
        revenue_usd = Decimal(base_result["revenue_usd"])

        # EEIO factor sensitivity
        base_ef = self.get_eeio_factor(naics_code)
        ef_low = base_ef * (_ONE - ef_variation_pct)
        ef_high = base_ef * (_ONE + ef_variation_pct)

        adjusted_rev = Decimal(base_result["adjusted_revenue_usd"])
        ef_low_co2e = (adjusted_rev * ef_low).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        ef_high_co2e = (adjusted_rev * ef_high).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # CPI sensitivity
        base_cpi = Decimal(base_result["cpi_deflator"])
        if base_cpi > _ZERO:
            cpi_low = base_cpi * (_ONE - cpi_variation_pct)
            cpi_high = base_cpi * (_ONE + cpi_variation_pct)
            margin_factor = self._get_margin_factor(naics_code)

            deflated_low = (revenue_usd / cpi_high).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            adj_low = self.apply_margin_adjustment(deflated_low, margin_factor)
            cpi_low_co2e = (adj_low * base_ef).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )

            deflated_high = (revenue_usd / cpi_low).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            adj_high = self.apply_margin_adjustment(
                deflated_high, margin_factor
            )
            cpi_high_co2e = (adj_high * base_ef).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
        else:
            cpi_low_co2e = base_co2e
            cpi_high_co2e = base_co2e

        # Margin sensitivity
        base_margin = self._get_margin_factor(naics_code)
        margin_low = base_margin - margin_variation_pct
        margin_high = base_margin + margin_variation_pct

        deflated_base = Decimal(base_result["deflated_revenue_usd"])
        margin_low_co2e = (
            self.apply_margin_adjustment(deflated_base, margin_low) * base_ef
        ).quantize(_QUANT_8DP, rounding=ROUNDING)
        margin_high_co2e = (
            self.apply_margin_adjustment(deflated_base, margin_high) * base_ef
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # Combined range (worst-case low to worst-case high)
        all_values = [
            ef_low_co2e, ef_high_co2e,
            cpi_low_co2e, cpi_high_co2e,
            margin_low_co2e, margin_high_co2e,
        ]
        combined_low = min(all_values)
        combined_high = max(all_values)

        # Tornado chart data (impact magnitude)
        tornado = [
            {
                "parameter": "EEIO factor",
                "low_co2e": str(ef_low_co2e),
                "high_co2e": str(ef_high_co2e),
                "impact_range": str(
                    (ef_high_co2e - ef_low_co2e).quantize(
                        _QUANT_8DP, rounding=ROUNDING
                    )
                ),
                "variation_pct": str(ef_variation_pct),
            },
            {
                "parameter": "CPI deflator",
                "low_co2e": str(cpi_low_co2e),
                "high_co2e": str(cpi_high_co2e),
                "impact_range": str(
                    abs(cpi_high_co2e - cpi_low_co2e).quantize(
                        _QUANT_8DP, rounding=ROUNDING
                    )
                ),
                "variation_pct": str(cpi_variation_pct),
            },
            {
                "parameter": "Margin adjustment",
                "low_co2e": str(margin_low_co2e),
                "high_co2e": str(margin_high_co2e),
                "impact_range": str(
                    abs(margin_high_co2e - margin_low_co2e).quantize(
                        _QUANT_8DP, rounding=ROUNDING
                    )
                ),
                "variation_pct": str(margin_variation_pct),
            },
        ]

        # Sort tornado by impact magnitude descending
        tornado.sort(
            key=lambda x: Decimal(x["impact_range"]),
            reverse=True,
        )

        provenance_hash = _calculate_provenance_hash(
            naics_code, str(base_co2e), str(combined_low),
            str(combined_high), "sensitivity_analysis",
        )

        return {
            "base_case": {
                "naics_code": naics_code,
                "co2e_kg": base_co2e,
                "eeio_factor": str(base_ef),
                "cpi_deflator": base_result["cpi_deflator"],
                "margin_factor": str(base_margin),
            },
            "ef_sensitivity": {
                "variation_pct": str(ef_variation_pct),
                "low_co2e": str(ef_low_co2e),
                "high_co2e": str(ef_high_co2e),
            },
            "cpi_sensitivity": {
                "variation_pct": str(cpi_variation_pct),
                "low_co2e": str(cpi_low_co2e),
                "high_co2e": str(cpi_high_co2e),
            },
            "margin_sensitivity": {
                "variation_pct": str(margin_variation_pct),
                "low_co2e": str(margin_low_co2e),
                "high_co2e": str(margin_high_co2e),
            },
            "combined_range": {
                "low_co2e": str(combined_low),
                "high_co2e": str(combined_high),
                "range_pct": str(
                    (
                        (combined_high - combined_low) / base_co2e * _HUNDRED
                    ).quantize(_QUANT_2DP, rounding=ROUNDING)
                    if base_co2e > _ZERO
                    else _ZERO
                ),
            },
            "tornado": tornado,
            "provenance_hash": provenance_hash,
        }

    def get_naics_hierarchy(
        self, naics_code: str
    ) -> Dict[str, Any]:
        """
        Get NAICS code hierarchy information for a leasing code.

        Provides sector (2-digit), subsector (3-digit), industry group
        (4-digit), and full (6-digit) NAICS classification context.

        Args:
            naics_code: NAICS leasing industry code.

        Returns:
            Dictionary with naics_code, name, sector, subsector,
            industry_group, ef, and asset_type.

        Raises:
            ValueError: If NAICS code is not found.

        Example:
            >>> info = engine.get_naics_hierarchy("531120")
            >>> info["sector"]
            '53 - Real Estate and Rental and Leasing'
        """
        entry = EEIO_FACTORS.get(naics_code)
        if entry is None:
            raise ValueError(
                f"NAICS code '{naics_code}' not found. "
                f"Available: {sorted(EEIO_FACTORS.keys())}"
            )

        # Extract hierarchy from NAICS code
        sector_code = naics_code[:2] if len(naics_code) >= 2 else naics_code
        subsector_code = naics_code[:3] if len(naics_code) >= 3 else naics_code
        group_code = naics_code[:4] if len(naics_code) >= 4 else naics_code

        # NAICS sector names for leasing-related codes
        sector_names = {
            "53": "53 - Real Estate and Rental and Leasing",
            "51": "51 - Information",
            "54": "54 - Professional, Scientific, and Technical Services",
        }

        subsector_names = {
            "531": "531 - Real Estate",
            "532": "532 - Rental and Leasing Services",
            "518": "518 - Computing Infrastructure Providers",
            "541": "541 - Professional, Scientific, and Technical Services",
        }

        group_names = {
            "5311": "5311 - Lessors of Real Estate",
            "5321": "5321 - Automotive Equipment Rental and Leasing",
            "5323": "5323 - General Rental Centers",
            "5324": "5324 - Commercial Equipment Rental and Leasing",
            "5182": "5182 - Computing Infrastructure Providers",
            "5415": "5415 - Computer Systems Design Services",
        }

        return {
            "naics_code": naics_code,
            "name": entry["name"],
            "sector": sector_names.get(sector_code, f"{sector_code} - Unknown"),
            "subsector": subsector_names.get(
                subsector_code, f"{subsector_code} - Unknown"
            ),
            "industry_group": group_names.get(
                group_code, f"{group_code} - Unknown"
            ),
            "ef": str(entry["ef"]),
            "ef_unit": entry.get("ef_unit", "kgCO2e/USD"),
            "asset_type": entry.get("asset_type", "unknown"),
            "category": entry.get("category", "unknown"),
            "margin_adjustment": str(
                MARGIN_ADJUSTMENT_FACTORS.get(naics_code, _ZERO)
            ),
        }

    def compare_methods(
        self,
        lease_revenue: Decimal,
        naics_code: str,
        currency: str = "USD",
        reporting_year: int = 2024,
    ) -> Dict[str, Any]:
        """
        Compare spend-based result against hypothetical Tier 1/2 estimates.

        Provides context by showing how the Tier 3 spend-based estimate
        compares to what Tier 1 and Tier 2 would produce under typical
        assumptions. Helps organizations understand the value of upgrading
        data quality.

        Args:
            lease_revenue: Lease revenue amount.
            naics_code: NAICS leasing industry code.
            currency: ISO 4217 currency code.
            reporting_year: Reporting year.

        Returns:
            Dictionary with tier_3_actual (this engine's result),
            tier_2_estimate, tier_1_estimate, accuracy_gap, and
            recommendation.
        """
        # Tier 3 actual
        tier_3 = self.calculate({
            "naics_code": naics_code,
            "lease_revenue": lease_revenue,
            "currency": currency,
            "reporting_year": reporting_year,
        })
        tier_3_co2e = tier_3["co2e_kg"]

        # Tier 2 estimate: typically 15-25% lower due to better specificity
        tier_2_adjustment = Decimal("0.80")  # 20% reduction from better data
        tier_2_co2e = (tier_3_co2e * tier_2_adjustment).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Tier 1 estimate: typically 30-40% lower
        tier_1_adjustment = Decimal("0.65")  # 35% reduction from metered data
        tier_1_co2e = (tier_3_co2e * tier_1_adjustment).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        accuracy_gap = (tier_3_co2e - tier_1_co2e).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        gap_pct = (
            (accuracy_gap / tier_3_co2e * _HUNDRED).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            if tier_3_co2e > _ZERO
            else _ZERO
        )

        recommendation = (
            f"Upgrading from spend-based (Tier 3) to asset-specific (Tier 1) "
            f"for NAICS {naics_code} could reduce reported emissions by "
            f"approximately {gap_pct}% ({accuracy_gap} kgCO2e) while "
            f"improving data quality from 'Poor' to 'Good'. "
            f"Uncertainty would decrease from +/-50% to +/-10%."
        )

        return {
            "tier_3_actual": {
                "co2e_kg": str(tier_3_co2e),
                "uncertainty_pct": "0.50",
                "dqi_classification": "Poor",
                "method": "spend_based",
            },
            "tier_2_estimate": {
                "co2e_kg": str(tier_2_co2e),
                "uncertainty_pct": "0.30",
                "dqi_classification": "Fair",
                "method": "average_data",
                "note": "Estimated 20% reduction from sector benchmarks",
            },
            "tier_1_estimate": {
                "co2e_kg": str(tier_1_co2e),
                "uncertainty_pct": "0.10",
                "dqi_classification": "Good",
                "method": "asset_specific",
                "note": "Estimated 35% reduction from metered data",
            },
            "accuracy_gap": {
                "tier_3_to_tier_1_kg": str(accuracy_gap),
                "gap_pct": str(gap_pct),
            },
            "recommendation": recommendation,
        }

    def get_calculation_stats(self) -> Dict[str, Any]:
        """
        Return engine calculation statistics.

        Returns:
            Dictionary with engine_id, engine_version, calculation_count,
            batch_count, and configuration.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "config": {
                "base_year": self._base_year,
                "enable_cpi_deflation": self._enable_cpi_deflation,
                "enable_margin_adjustment": self._enable_margin_adjustment,
            },
        }

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _validate_calculate_inputs(self, input_data: Dict[str, Any]) -> None:
        """
        Validate calculate() inputs, raising ValueError on failure.

        Args:
            input_data: Input dictionary.

        Raises:
            ValueError: If validation fails.
        """
        # NAICS code
        naics_code = input_data.get("naics_code", "")
        if not naics_code:
            # Try auto-classification from description
            description = input_data.get("description", "")
            if description:
                classified = self.classify_lease(description)
                if classified:
                    input_data["naics_code"] = classified
                    naics_code = classified
                else:
                    raise ValueError(
                        "naics_code is required (auto-classification of "
                        f"'{description[:50]}' failed)"
                    )
            else:
                raise ValueError("naics_code is required")

        if naics_code not in EEIO_FACTORS:
            raise ValueError(
                f"NAICS code '{naics_code}' not found in EEIO_FACTORS. "
                f"Available: {sorted(EEIO_FACTORS.keys())}"
            )

        # Lease revenue
        revenue = input_data.get("lease_revenue")
        if revenue is None:
            raise ValueError("lease_revenue is required")

        try:
            rev_dec = Decimal(str(revenue))
            if rev_dec <= _ZERO:
                raise ValueError(
                    f"lease_revenue must be positive, got {rev_dec}"
                )
        except (InvalidOperation, ValueError) as e:
            raise ValueError(
                f"lease_revenue must be numeric, got '{revenue}': {e}"
            )

        # Currency
        currency = input_data.get("currency", "USD").upper()
        if currency not in CURRENCY_RATES:
            raise ValueError(
                f"Currency '{currency}' not supported. "
                f"Available: {sorted(CURRENCY_RATES.keys())}"
            )

    def _get_cpi_deflator(self, reporting_year: int) -> Decimal:
        """
        Get CPI deflator for the reporting year.

        If CPI deflation is disabled or year not available, returns 1.0.

        Args:
            reporting_year: Year the revenue was earned.

        Returns:
            CPI deflator value (base year = 1.000).
        """
        if not self._enable_cpi_deflation:
            return _ONE

        deflator = CPI_DEFLATORS.get(reporting_year)
        if deflator is None:
            logger.warning(
                "CPI deflator not available for year %d, skipping deflation. "
                "Available years: %s",
                reporting_year,
                sorted(CPI_DEFLATORS.keys()),
            )
            return _ONE

        return deflator

    def _get_margin_factor(self, naics_code: str) -> Decimal:
        """
        Get margin adjustment factor for a NAICS code.

        Args:
            naics_code: NAICS leasing industry code.

        Returns:
            Margin adjustment factor (Decimal).
        """
        if not self._enable_margin_adjustment:
            return _ZERO

        return MARGIN_ADJUSTMENT_FACTORS.get(naics_code, _ZERO)


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
        >>> engine.health_check()["status"]
        'healthy'
    """
    return SpendBasedCalculatorEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "TIER_3_UNCERTAINTY",
    "DEFAULT_BASE_YEAR",
    # Enums
    "CurrencyCode",
    "LeaseCategory",
    # Data tables
    "EEIO_FACTORS",
    "CPI_DEFLATORS",
    "CURRENCY_RATES",
    "MARGIN_ADJUSTMENT_FACTORS",
    # Engine
    "SpendBasedCalculatorEngine",
    "get_spend_based_calculator",
]
