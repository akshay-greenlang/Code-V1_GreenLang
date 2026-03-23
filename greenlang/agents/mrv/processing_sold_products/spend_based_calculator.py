# -*- coding: utf-8 -*-
"""
SpendBasedCalculatorEngine -- AGENT-MRV-023 Engine 4 of 7

This module implements the SpendBasedCalculatorEngine for the Processing of
Sold Products Agent (GL-MRV-S3-010).  The engine calculates GHG emissions
using the *spend-based EEIO method* from the GHG Protocol Scope 3 Technical
Guidance, Chapter 6 (Category 10), where emissions are estimated from
revenue of intermediate products sold by applying downstream-sector
EEIO emission factors.

Core Formula (Formula F -- EEIO)::

    E_cat10 = Sum_i( Rev_i x EF_eeio_sector_i x (1 - margin_i) )

    where:
        Rev_i       = revenue from intermediate product i (USD)
        EF_eeio_i   = EEIO factor for downstream sector (kgCO2e/USD)
        margin_i    = profit margin adjustment (0.05-0.15)

Pipeline (per item)::

    Revenue_local
        -> (/ FX_rate) -> Revenue_USD
        -> (* CPI_base / CPI_current) -> Revenue_deflated
        -> (* (1 - margin_rate)) -> Revenue_adjusted
        -> (* EEIO_factor) -> Emissions_kgCO2e
        -> (/ 1000) -> Emissions_tCO2e

The engine supports:

- **Single-item calculation** via ``calculate``.
- **EEIO factor application** via ``calculate_eeio``.
- **Batch processing** via ``calculate_batch``.
- **Currency conversion** for 12 currencies via ``convert_to_usd``.
- **CPI deflation** across 11 years (2015-2025) via ``deflate_to_base``.
- **Margin adjustment** per NAICS sector via ``apply_margin_adjustment``.
- **DQI scoring** with lowest-tier default scores.
- **Uncertainty estimation** at +/-50% default.
- **SHA-256 provenance tracking** for complete audit trails.

NAICS Sectors (12):
    331, 332, 325, 326, 311, 313, 334, 327, 321, 322, 336, 335

Currencies (12):
    USD, EUR, GBP, JPY, CNY, INR, CAD, AUD, KRW, BRL, MXN, CHF

CPI Years (11):
    2015-2025 with 2024 as base year (index 100.0)

Thread Safety:
    The engine is implemented as a thread-safe singleton using
    ``threading.RLock`` with double-checked locking.

Zero-Hallucination Guarantee:
    All numeric operations use ``Decimal`` with explicit quantization.
    No LLM calls are made for any numeric calculation.  EEIO factors
    are sourced exclusively from the deterministic constant tables
    in this module, derived from PRD-AGENT-MRV-023 Section 5.3 and 5.7.

Data Sources:
    - EPA USEEIO v1.2/v1.3 (sector-level factors)
    - EXIOBASE 3.8 (supplementary)
    - US Bureau of Labor Statistics CPI data
    - Federal Reserve exchange rates

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-023 Processing of Sold Products (GL-MRV-S3-010)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "SpendBasedCalculatorEngine",
    "EEIO_SECTOR_FACTORS",
    "CURRENCY_RATES",
    "CPI_INDEX",
    "SECTOR_MARGINS",
    "SpendItem",
    "SpendCalculationResult",
    "SpendBreakdown",
    "SpendDataQualityScore",
    "SpendUncertaintyResult",
]

# ---------------------------------------------------------------------------
# Agent-level constants
# ---------------------------------------------------------------------------

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"

# ---------------------------------------------------------------------------
# Decimal constants
# ---------------------------------------------------------------------------

DECIMAL_PLACES: int = 8
ZERO: Decimal = Decimal("0")
ONE: Decimal = Decimal("1")
ONE_HUNDRED: Decimal = Decimal("100")
ONE_THOUSAND: Decimal = Decimal("1000")
_PRECISION: Decimal = Decimal(10) ** -DECIMAL_PLACES

# ---------------------------------------------------------------------------
# Default DQI composite score for spend-based method
# PRD Section 5.8: Lowest quality tier (score 20-40)
# ---------------------------------------------------------------------------

_DQI_COMPOSITE_SPEND: int = 30

# Default uncertainty percentage for spend-based (PRD Section 5.9)
_DEFAULT_UNCERTAINTY_PCT: Decimal = Decimal("50")
_MIN_UNCERTAINTY_PCT: Decimal = Decimal("30")
_MAX_UNCERTAINTY_PCT: Decimal = Decimal("100")

# Maximum batch size
_MAX_BATCH_SIZE: int = 100_000

# CPI base year
_CPI_BASE_YEAR: int = 2024


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NAICSSector(str, Enum):
    """NAICS sector codes for downstream processing industries.

    These 12 sectors represent the primary downstream industries
    that process intermediate products. Each sector has an associated
    EEIO factor (kgCO2e/USD) and margin percentage.
    """

    NAICS_331 = "331"
    NAICS_332 = "332"
    NAICS_325 = "325"
    NAICS_326 = "326"
    NAICS_311 = "311"
    NAICS_313 = "313"
    NAICS_334 = "334"
    NAICS_327 = "327"
    NAICS_321 = "321"
    NAICS_322 = "322"
    NAICS_336 = "336"
    NAICS_335 = "335"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes supported for conversion.

    The engine supports 12 major currencies with fixed exchange
    rates to USD. For production use, rates should be updated
    periodically from a live feed.
    """

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CAD = "CAD"
    AUD = "AUD"
    KRW = "KRW"
    BRL = "BRL"
    MXN = "MXN"
    CHF = "CHF"


# ---------------------------------------------------------------------------
# EEIO Sector Factors (kgCO2e/USD) -- PRD Section 5.3
# ---------------------------------------------------------------------------

EEIO_SECTOR_FACTORS: Dict[str, Decimal] = {
    NAICSSector.NAICS_331.value: Decimal("0.82"),
    NAICSSector.NAICS_332.value: Decimal("0.45"),
    NAICSSector.NAICS_325.value: Decimal("0.65"),
    NAICSSector.NAICS_326.value: Decimal("0.52"),
    NAICSSector.NAICS_311.value: Decimal("0.38"),
    NAICSSector.NAICS_313.value: Decimal("0.42"),
    NAICSSector.NAICS_334.value: Decimal("0.28"),
    NAICSSector.NAICS_327.value: Decimal("0.72"),
    NAICSSector.NAICS_321.value: Decimal("0.35"),
    NAICSSector.NAICS_322.value: Decimal("0.48"),
    NAICSSector.NAICS_336.value: Decimal("0.40"),
    NAICSSector.NAICS_335.value: Decimal("0.32"),
}


# Sector margin percentages (as decimal fractions, e.g. 0.08 = 8%)
# PRD Section 5.3
SECTOR_MARGINS: Dict[str, Decimal] = {
    NAICSSector.NAICS_331.value: Decimal("0.08"),
    NAICSSector.NAICS_332.value: Decimal("0.10"),
    NAICSSector.NAICS_325.value: Decimal("0.12"),
    NAICSSector.NAICS_326.value: Decimal("0.10"),
    NAICSSector.NAICS_311.value: Decimal("0.08"),
    NAICSSector.NAICS_313.value: Decimal("0.10"),
    NAICSSector.NAICS_334.value: Decimal("0.15"),
    NAICSSector.NAICS_327.value: Decimal("0.08"),
    NAICSSector.NAICS_321.value: Decimal("0.10"),
    NAICSSector.NAICS_322.value: Decimal("0.10"),
    NAICSSector.NAICS_336.value: Decimal("0.12"),
    NAICSSector.NAICS_335.value: Decimal("0.12"),
}

#: Default margin rate when sector is not in the table
_DEFAULT_MARGIN: Decimal = Decimal("0.10")


# Sector display names for reporting
SECTOR_NAMES: Dict[str, str] = {
    NAICSSector.NAICS_331.value: "Primary Metal Manufacturing",
    NAICSSector.NAICS_332.value: "Fabricated Metal Products",
    NAICSSector.NAICS_325.value: "Chemical Manufacturing",
    NAICSSector.NAICS_326.value: "Plastics & Rubber Products",
    NAICSSector.NAICS_311.value: "Food Manufacturing",
    NAICSSector.NAICS_313.value: "Textile Mills",
    NAICSSector.NAICS_334.value: "Computer & Electronic Products",
    NAICSSector.NAICS_327.value: "Nonmetallic Mineral Products",
    NAICSSector.NAICS_321.value: "Wood Product Manufacturing",
    NAICSSector.NAICS_322.value: "Paper Manufacturing",
    NAICSSector.NAICS_336.value: "Transportation Equipment",
    NAICSSector.NAICS_335.value: "Electrical Equipment",
}


# ---------------------------------------------------------------------------
# Currency Exchange Rates to USD -- PRD Section 5.7
# ---------------------------------------------------------------------------

CURRENCY_RATES: Dict[str, Decimal] = {
    CurrencyCode.USD.value: Decimal("1.000"),
    CurrencyCode.EUR.value: Decimal("1.085"),
    CurrencyCode.GBP.value: Decimal("1.268"),
    CurrencyCode.JPY.value: Decimal("0.0067"),
    CurrencyCode.CNY.value: Decimal("0.138"),
    CurrencyCode.INR.value: Decimal("0.012"),
    CurrencyCode.CAD.value: Decimal("0.742"),
    CurrencyCode.AUD.value: Decimal("0.651"),
    CurrencyCode.KRW.value: Decimal("0.00075"),
    CurrencyCode.BRL.value: Decimal("0.198"),
    CurrencyCode.MXN.value: Decimal("0.058"),
    CurrencyCode.CHF.value: Decimal("1.122"),
}


# ---------------------------------------------------------------------------
# CPI Deflation Index -- PRD Section 5.7
# Base year 2024 = 100.0
# ---------------------------------------------------------------------------

CPI_INDEX: Dict[int, Decimal] = {
    2015: Decimal("76.5"),
    2016: Decimal("77.5"),
    2017: Decimal("79.1"),
    2018: Decimal("81.0"),
    2019: Decimal("82.5"),
    2020: Decimal("83.5"),
    2021: Decimal("87.3"),
    2022: Decimal("94.1"),
    2023: Decimal("97.8"),
    2024: Decimal("100.0"),
    2025: Decimal("102.4"),
}


# Default DQI dimension scores for spend-based method (scale 1-5, lower=better)
_DQI_SPEND: Dict[str, Decimal] = {
    "reliability": Decimal("4"),
    "completeness": Decimal("3"),
    "temporal": Decimal("4"),
    "geographical": Decimal("4"),
    "technological": Decimal("4"),
}


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


class SpendDataQualityScore:
    """Data quality indicator score for a spend-based calculation.

    Attributes:
        reliability: Reliability dimension score (1-5).
        completeness: Completeness dimension score (1-5).
        temporal: Temporal correlation score (1-5).
        geographical: Geographical correlation score (1-5).
        technological: Technological correlation score (1-5).
        composite: Weighted composite score (0-100).
        method: Calculation method (always 'spend_based').
    """

    __slots__ = (
        "reliability", "completeness", "temporal",
        "geographical", "technological", "composite",
        "method",
    )

    def __init__(
        self,
        reliability: Decimal,
        completeness: Decimal,
        temporal: Decimal,
        geographical: Decimal,
        technological: Decimal,
        composite: int,
        method: str = "spend_based",
    ) -> None:
        """Initialize SpendDataQualityScore.

        Args:
            reliability: Reliability dimension (1-5).
            completeness: Completeness dimension (1-5).
            temporal: Temporal dimension (1-5).
            geographical: Geographical dimension (1-5).
            technological: Technological dimension (1-5).
            composite: Composite score 0-100.
            method: Method identifier.
        """
        self.reliability = reliability
        self.completeness = completeness
        self.temporal = temporal
        self.geographical = geographical
        self.technological = technological
        self.composite = composite
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary with all DQI scores.
        """
        return {
            "reliability": str(self.reliability),
            "completeness": str(self.completeness),
            "temporal": str(self.temporal),
            "geographical": str(self.geographical),
            "technological": str(self.technological),
            "composite": self.composite,
            "method": self.method,
        }


class SpendUncertaintyResult:
    """Uncertainty quantification result for spend-based emissions.

    Attributes:
        emissions_kgco2e: Central estimate in kgCO2e.
        lower_bound_kgco2e: Lower bound at 95% CI.
        upper_bound_kgco2e: Upper bound at 95% CI.
        uncertainty_pct: Uncertainty percentage.
        confidence_level: Confidence level (default 95).
        method: Method identifier.
    """

    __slots__ = (
        "emissions_kgco2e", "lower_bound_kgco2e", "upper_bound_kgco2e",
        "uncertainty_pct", "confidence_level", "method",
    )

    def __init__(
        self,
        emissions_kgco2e: Decimal,
        lower_bound_kgco2e: Decimal,
        upper_bound_kgco2e: Decimal,
        uncertainty_pct: Decimal,
        confidence_level: int = 95,
        method: str = "spend_based",
    ) -> None:
        """Initialize SpendUncertaintyResult.

        Args:
            emissions_kgco2e: Central emission estimate.
            lower_bound_kgco2e: Lower bound.
            upper_bound_kgco2e: Upper bound.
            uncertainty_pct: Uncertainty percentage.
            confidence_level: Confidence level.
            method: Method identifier.
        """
        self.emissions_kgco2e = emissions_kgco2e
        self.lower_bound_kgco2e = lower_bound_kgco2e
        self.upper_bound_kgco2e = upper_bound_kgco2e
        self.uncertainty_pct = uncertainty_pct
        self.confidence_level = confidence_level
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "emissions_kgco2e": str(self.emissions_kgco2e),
            "lower_bound_kgco2e": str(self.lower_bound_kgco2e),
            "upper_bound_kgco2e": str(self.upper_bound_kgco2e),
            "uncertainty_pct": str(self.uncertainty_pct),
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


class SpendItem:
    """Input data for a single spend-based calculation item.

    Attributes:
        item_id: Unique item identifier.
        product_name: Human-readable product name.
        revenue: Revenue amount in source currency.
        currency: ISO 4217 currency code.
        sector: NAICS 3-digit sector code.
        year: Revenue year for CPI deflation.
    """

    __slots__ = (
        "item_id", "product_name", "revenue",
        "currency", "sector", "year",
    )

    def __init__(
        self,
        item_id: str,
        product_name: str,
        revenue: Decimal,
        currency: str,
        sector: str,
        year: int,
    ) -> None:
        """Initialize SpendItem.

        Args:
            item_id: Unique identifier.
            product_name: Product name.
            revenue: Revenue amount.
            currency: Currency code.
            sector: NAICS sector code.
            year: Revenue year.
        """
        self.item_id = item_id
        self.product_name = product_name
        self.revenue = revenue
        self.currency = currency
        self.sector = sector
        self.year = year

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "item_id": self.item_id,
            "product_name": self.product_name,
            "revenue": str(self.revenue),
            "currency": self.currency,
            "sector": self.sector,
            "year": self.year,
        }


class SpendBreakdown:
    """Per-item breakdown from spend-based calculation.

    Attributes:
        item_id: Item identifier.
        product_name: Product name.
        sector: NAICS sector code.
        sector_name: Sector display name.
        revenue_original: Original revenue in source currency.
        currency: Source currency code.
        revenue_usd: Revenue converted to USD.
        revenue_deflated: Revenue after CPI deflation.
        revenue_adjusted: Revenue after margin removal.
        eeio_factor: EEIO factor applied (kgCO2e/USD).
        margin_rate: Margin rate applied.
        cpi_ratio: CPI ratio used for deflation.
        fx_rate: Exchange rate used.
        emissions_kgco2e: Emissions in kgCO2e.
        emissions_tco2e: Emissions in tCO2e.
        dqi: Data quality score.
        uncertainty: Uncertainty range.
        provenance_hash: SHA-256 provenance hash.
        calculated_at: Calculation timestamp.
    """

    __slots__ = (
        "item_id", "product_name", "sector", "sector_name",
        "revenue_original", "currency", "revenue_usd",
        "revenue_deflated", "revenue_adjusted",
        "eeio_factor", "margin_rate", "cpi_ratio", "fx_rate",
        "emissions_kgco2e", "emissions_tco2e",
        "dqi", "uncertainty",
        "provenance_hash", "calculated_at",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SpendBreakdown from keyword arguments.

        Args:
            **kwargs: Field values matching __slots__.
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        result: Dict[str, Any] = {}
        for slot in self.__slots__:
            val = getattr(self, slot, None)
            if val is not None:
                if isinstance(val, Decimal):
                    result[slot] = str(val)
                elif isinstance(val, datetime):
                    result[slot] = val.isoformat()
                elif hasattr(val, "to_dict"):
                    result[slot] = val.to_dict()
                else:
                    result[slot] = val
        return result


class SpendCalculationResult:
    """Aggregated result from spend-based calculations.

    Attributes:
        calculation_id: Unique calculation identifier.
        org_id: Organization identifier.
        reporting_year: Reporting period year.
        method: Always 'spend_based'.
        total_emissions_kgco2e: Total emissions in kgCO2e.
        total_emissions_tco2e: Total emissions in tCO2e.
        total_revenue_usd: Total revenue processed in USD.
        item_count: Number of items processed.
        breakdowns: Per-item breakdowns.
        dqi: Portfolio-level DQI score.
        uncertainty: Portfolio-level uncertainty.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Calculation duration in ms.
        calculated_at: Calculation timestamp.
        warnings: Warnings generated.
        errors: Errors encountered.
    """

    __slots__ = (
        "calculation_id", "org_id", "reporting_year",
        "method", "total_emissions_kgco2e",
        "total_emissions_tco2e", "total_revenue_usd",
        "item_count", "breakdowns",
        "dqi", "uncertainty",
        "provenance_hash", "processing_time_ms",
        "calculated_at", "warnings", "errors",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SpendCalculationResult from keyword arguments.

        Args:
            **kwargs: Field values matching __slots__.
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        result: Dict[str, Any] = {}
        for slot in self.__slots__:
            val = getattr(self, slot, None)
            if val is not None:
                if isinstance(val, Decimal):
                    result[slot] = str(val)
                elif isinstance(val, datetime):
                    result[slot] = val.isoformat()
                elif hasattr(val, "to_dict"):
                    result[slot] = val.to_dict()
                elif isinstance(val, list):
                    result[slot] = [
                        item.to_dict() if hasattr(item, "to_dict") else item
                        for item in val
                    ]
                else:
                    result[slot] = val
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed.

    Returns:
        datetime: Current UTC time with microsecond=0.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the configured precision.

    Args:
        value: Raw Decimal value.

    Returns:
        Decimal quantized to DECIMAL_PLACES places using ROUND_HALF_UP.
    """
    try:
        return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)
    except (InvalidOperation, OverflowError):
        logger.warning("Quantize failed for value=%s, returning ZERO", value)
        return ZERO


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when denominator is zero.

    Returns:
        Quantized quotient or default.
    """
    if denominator == ZERO or denominator is None:
        return default
    return _quantize(numerator / denominator)


def _compute_sha256(data: Any) -> str:
    """Compute SHA-256 hex digest for arbitrary data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    try:
        canonical = json.dumps(
            data,
            sort_keys=True,
            default=str,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        logger.warning("SHA-256 hashing failed: %s", exc)
        return hashlib.sha256(b"fallback").hexdigest()


def _validate_positive_decimal(
    value: Any,
    field_name: str = "value",
) -> Decimal:
    """Validate and coerce to a positive Decimal.

    Args:
        value: Value to validate.
        field_name: Field name for error messages.

    Returns:
        Positive Decimal.

    Raises:
        ValueError: If value is None, negative, or non-numeric.
    """
    if value is None:
        raise ValueError(f"{field_name} must not be None")
    try:
        dec_val = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a valid number, got {value!r}"
        ) from exc
    if dec_val < ZERO:
        raise ValueError(
            f"{field_name} must be non-negative, got {dec_val}"
        )
    return dec_val


def _resolve_sector(sector: Any) -> str:
    """Resolve a NAICS sector code to its 3-digit key.

    Accepts enum values, strings, and integer codes. Extracts the
    first 3 digits from longer NAICS codes.

    Args:
        sector: NAICS sector code (string, int, or NAICSSector).

    Returns:
        3-digit NAICS sector code string.

    Raises:
        ValueError: If sector cannot be resolved.
    """
    if isinstance(sector, NAICSSector):
        return sector.value
    sector_str = str(sector).strip()
    # Extract first 3 digits
    if len(sector_str) >= 3:
        code_3 = sector_str[:3]
    else:
        code_3 = sector_str
    # Validate against known sectors
    if code_3 in EEIO_SECTOR_FACTORS:
        return code_3
    raise ValueError(
        f"Unknown NAICS sector: {sector!r}. "
        f"Valid sectors: {list(EEIO_SECTOR_FACTORS.keys())}"
    )


def _resolve_currency(currency: Any) -> str:
    """Resolve a currency code to its uppercase key.

    Args:
        currency: Currency code (string or CurrencyCode).

    Returns:
        Uppercase ISO 4217 currency code.

    Raises:
        ValueError: If currency is not supported.
    """
    if isinstance(currency, CurrencyCode):
        return currency.value
    code = str(currency).upper().strip()
    if code in CURRENCY_RATES:
        return code
    raise ValueError(
        f"Unsupported currency: {currency!r}. "
        f"Supported: {list(CURRENCY_RATES.keys())}"
    )


# ===========================================================================
# SpendBasedCalculatorEngine -- Thread-Safe Singleton
# ===========================================================================


class SpendBasedCalculatorEngine:
    """Engine 4: Spend-based EEIO emission calculator for Processing of Sold Products.

    Implements the spend-based EEIO calculation method from the GHG Protocol
    Scope 3 Technical Guidance Chapter 6 (Category 10). Estimates downstream
    processing emissions from revenue of intermediate products sold using
    sector-level EEIO factors with currency conversion, CPI deflation,
    and margin adjustment.

    This is the broadest-coverage but lowest-accuracy method (DQI=30),
    suitable as a screening tool for product lines where site-specific
    or average-data methods are not feasible.

    Pipeline (per item):
        1. Currency conversion: Revenue_local -> Revenue_USD
        2. CPI deflation: Revenue_USD -> Revenue_deflated (base year 2024)
        3. Margin removal: Revenue_deflated x (1 - margin) -> Revenue_adjusted
        4. EEIO factor: Revenue_adjusted x EF_eeio -> Emissions_kgCO2e
        5. Unit conversion: kgCO2e / 1000 -> tCO2e
        6. DQI scoring: Default spend-based scores
        7. Provenance: SHA-256 hash of all intermediates

    Thread Safety:
        Singleton via ``__new__`` with ``threading.RLock()``.

    Zero-Hallucination:
        All arithmetic uses ``Decimal``.  No LLM involvement.

    Attributes:
        _calculation_count: Total calculations performed.
        _total_emissions_kgco2e: Cumulative emissions.
        _total_revenue_usd: Cumulative revenue processed.
        _error_count: Total errors encountered.
        _last_calculation_time: Timestamp of last calculation.

    Example:
        >>> engine = SpendBasedCalculatorEngine()
        >>> result = engine.calculate(
        ...     revenue=Decimal("500000"),
        ...     currency="USD",
        ...     sector="331",
        ...     year=2024,
        ...     org_id="ORG-1",
        ...     reporting_year=2024,
        ... )
        >>> result.total_emissions_kgco2e
        Decimal('376200.00000000')
    """

    _instance: Optional[SpendBasedCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    def __new__(cls) -> SpendBasedCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with ``threading.RLock``.

        Returns:
            The singleton SpendBasedCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine with internal counters.

        Guarded by ``_initialized`` flag to prevent re-initialization.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._calculation_count: int = 0
            self._batch_count: int = 0
            self._total_emissions_kgco2e: Decimal = ZERO
            self._total_revenue_usd: Decimal = ZERO
            self._error_count: int = 0
            self._last_calculation_time: Optional[datetime] = None
            self.__class__._initialized = True
            logger.info(
                "SpendBasedCalculatorEngine initialized "
                "(agent=%s, version=%s, precision=%d, "
                "sectors=%d, currencies=%d, cpi_years=%d)",
                AGENT_ID,
                VERSION,
                DECIMAL_PLACES,
                len(EEIO_SECTOR_FACTORS),
                len(CURRENCY_RATES),
                len(CPI_INDEX),
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for testing purposes.

        Clears the singleton so next instantiation creates a fresh engine.
        For unit tests only.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.info("SpendBasedCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Public API: Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and counters.
        """
        return {
            "engine": "SpendBasedCalculatorEngine",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "status": "healthy",
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "total_emissions_kgco2e": str(self._total_emissions_kgco2e),
            "total_revenue_usd": str(self._total_revenue_usd),
            "error_count": self._error_count,
            "last_calculation_time": (
                self._last_calculation_time.isoformat()
                if self._last_calculation_time else None
            ),
            "sectors": len(EEIO_SECTOR_FACTORS),
            "currencies": len(CURRENCY_RATES),
            "cpi_years": len(CPI_INDEX),
        }

    # ------------------------------------------------------------------
    # Public API: Single Item Calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        revenue: Any,
        currency: str,
        sector: str,
        year: int,
        org_id: str,
        reporting_year: int,
        product_name: str = "",
        item_id: Optional[str] = None,
    ) -> SpendCalculationResult:
        """Calculate spend-based emissions for a single revenue item.

        Executes the full pipeline: currency conversion, CPI deflation,
        margin removal, EEIO factor application, and provenance hashing.

        Args:
            revenue: Revenue amount in source currency.
            currency: ISO 4217 currency code.
            sector: NAICS 3-digit sector code.
            year: Revenue year for CPI deflation.
            org_id: Organization identifier.
            reporting_year: Reporting period year.
            product_name: Optional product name for reporting.
            item_id: Optional item identifier.

        Returns:
            SpendCalculationResult with emissions and breakdown.

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())
        actual_item_id = item_id or str(uuid.uuid4())

        try:
            # Validate inputs
            revenue_dec = _validate_positive_decimal(revenue, "revenue")
            sector_code = _resolve_sector(sector)
            currency_code = _resolve_currency(currency)

            # Execute pipeline
            breakdown = self._calculate_single_item(
                item_id=actual_item_id,
                product_name=product_name,
                revenue=revenue_dec,
                currency=currency_code,
                sector=sector_code,
                year=year,
            )

            total_kgco2e = breakdown.emissions_kgco2e
            total_tco2e = breakdown.emissions_tco2e

            dqi = self.compute_dqi_score()
            uncertainty = self.compute_uncertainty(total_kgco2e)

            provenance_hash = self._build_provenance(
                inputs={
                    "revenue": str(revenue_dec),
                    "currency": currency_code,
                    "sector": sector_code,
                    "year": year,
                },
                result={
                    "total_kgco2e": str(total_kgco2e),
                    "total_tco2e": str(total_tco2e),
                },
            )

            elapsed_ms = _quantize(
                Decimal(str((time.monotonic() - start_time) * 1000))
            )

            result = SpendCalculationResult(
                calculation_id=calculation_id,
                org_id=org_id,
                reporting_year=reporting_year,
                method="spend_based",
                total_emissions_kgco2e=total_kgco2e,
                total_emissions_tco2e=total_tco2e,
                total_revenue_usd=breakdown.revenue_usd,
                item_count=1,
                breakdowns=[breakdown],
                dqi=dqi,
                uncertainty=uncertainty,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
                calculated_at=_utcnow(),
                warnings=[],
                errors=[],
            )

            self._update_counters(total_kgco2e, breakdown.revenue_usd, 1)

            logger.info(
                "calculate completed (calc_id=%s, sector=%s, "
                "revenue_usd=%s, emissions_kgco2e=%s, elapsed_ms=%s)",
                calculation_id,
                sector_code,
                str(breakdown.revenue_usd),
                str(total_kgco2e),
                str(elapsed_ms),
            )

            return result

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "calculate failed: %s", str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API: EEIO Factor Calculation
    # ------------------------------------------------------------------

    def calculate_eeio(
        self,
        revenue_usd: Any,
        sector: str,
    ) -> Decimal:
        """Calculate EEIO emissions from USD revenue and sector.

        E = Rev x EF_eeio x (1 - margin)

        This is the core EEIO calculation without currency conversion
        or CPI deflation. Revenue must already be in USD.

        Args:
            revenue_usd: Revenue in USD (pre-converted).
            sector: NAICS 3-digit sector code.

        Returns:
            Emissions in kgCO2e.

        Raises:
            ValueError: If revenue or sector is invalid.
        """
        rev = _validate_positive_decimal(revenue_usd, "revenue_usd")
        sector_code = _resolve_sector(sector)

        eeio_factor, margin = self._resolve_eeio_factor(sector_code)
        adjusted = _quantize(rev * (ONE - margin))
        emissions = _quantize(adjusted * eeio_factor)

        logger.debug(
            "calculate_eeio: revenue_usd=%s, sector=%s, "
            "eeio_factor=%s, margin=%s, adjusted=%s, emissions=%s",
            str(rev), sector_code, str(eeio_factor),
            str(margin), str(adjusted), str(emissions),
        )

        return emissions

    # ------------------------------------------------------------------
    # Public API: Batch Calculation
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> SpendCalculationResult:
        """Calculate spend-based emissions for a batch of items.

        Each item in the list should be a dictionary with keys:
        item_id, product_name, revenue, currency, sector, year.

        Args:
            items: List of item dictionaries.
            org_id: Organization identifier.
            reporting_year: Reporting period year.

        Returns:
            SpendCalculationResult with aggregated emissions.

        Raises:
            ValueError: If items list is empty or exceeds max size.
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())

        try:
            self._validate_batch_input(items)

            breakdowns: List[SpendBreakdown] = []
            total_kgco2e = ZERO
            total_revenue_usd = ZERO
            warnings: List[str] = []
            errors: List[str] = []

            for item in items:
                try:
                    bd = self._calculate_from_dict(item)
                    breakdowns.append(bd)
                    total_kgco2e += bd.emissions_kgco2e
                    total_revenue_usd += bd.revenue_usd
                except Exception as exc:
                    error_msg = (
                        f"Item {item.get('item_id', 'unknown')}: "
                        f"{str(exc)}"
                    )
                    errors.append(error_msg)
                    logger.warning("Batch item failed: %s", error_msg)

            total_tco2e = _quantize(total_kgco2e / ONE_THOUSAND)
            total_kgco2e = _quantize(total_kgco2e)
            total_revenue_usd = _quantize(total_revenue_usd)

            dqi = self.compute_dqi_score()
            uncertainty = self.compute_uncertainty(total_kgco2e)

            provenance_hash = self._build_provenance(
                inputs={
                    "item_count": len(items),
                    "item_ids": [i.get("item_id") for i in items],
                },
                result={
                    "total_kgco2e": str(total_kgco2e),
                    "total_revenue_usd": str(total_revenue_usd),
                },
            )

            elapsed_ms = _quantize(
                Decimal(str((time.monotonic() - start_time) * 1000))
            )

            result = SpendCalculationResult(
                calculation_id=calculation_id,
                org_id=org_id,
                reporting_year=reporting_year,
                method="spend_based",
                total_emissions_kgco2e=total_kgco2e,
                total_emissions_tco2e=total_tco2e,
                total_revenue_usd=total_revenue_usd,
                item_count=len(breakdowns),
                breakdowns=breakdowns,
                dqi=dqi,
                uncertainty=uncertainty,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
                calculated_at=_utcnow(),
                warnings=warnings,
                errors=errors,
            )

            self._update_counters(
                total_kgco2e, total_revenue_usd, len(breakdowns),
            )

            logger.info(
                "calculate_batch completed (calc_id=%s, items=%d, "
                "total_kgco2e=%s, total_revenue_usd=%s, elapsed_ms=%s)",
                calculation_id,
                len(breakdowns),
                str(total_kgco2e),
                str(total_revenue_usd),
                str(elapsed_ms),
            )

            return result

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "calculate_batch failed: %s", str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API: Currency Conversion
    # ------------------------------------------------------------------

    def convert_to_usd(
        self,
        amount: Any,
        currency: str,
    ) -> Decimal:
        """Convert an amount from source currency to USD.

        Uses the fixed exchange rate table from CURRENCY_RATES.
        For USD, returns the amount unchanged. For other currencies,
        multiplies by the rate-to-USD factor.

        Args:
            amount: Amount in source currency.
            currency: ISO 4217 currency code.

        Returns:
            Amount in USD, quantized.

        Raises:
            ValueError: If amount is invalid or currency unsupported.
        """
        dec_amount = _validate_positive_decimal(amount, "amount")
        code = _resolve_currency(currency)

        if code == CurrencyCode.USD.value:
            return _quantize(dec_amount)

        rate = CURRENCY_RATES.get(code)
        if rate is None or rate == ZERO:
            raise ValueError(
                f"No exchange rate for currency: {code}"
            )

        usd_amount = _quantize(dec_amount * rate)

        logger.debug(
            "convert_to_usd: %s %s -> %s USD (rate=%s)",
            str(dec_amount), code, str(usd_amount), str(rate),
        )

        return usd_amount

    # ------------------------------------------------------------------
    # Public API: CPI Deflation
    # ------------------------------------------------------------------

    def deflate_to_base(
        self,
        amount: Any,
        from_year: int,
        base_year: int = _CPI_BASE_YEAR,
    ) -> Decimal:
        """Deflate an amount from a given year to the base year.

        Uses the CPI index table to adjust for inflation/deflation.
        Formula: amount_deflated = amount x (CPI_base / CPI_from)

        Args:
            amount: Amount to deflate.
            from_year: Year of the original amount.
            base_year: Target base year (default 2024).

        Returns:
            Deflated amount, quantized.

        Raises:
            ValueError: If amount is invalid or years not in CPI table.
        """
        dec_amount = _validate_positive_decimal(amount, "amount")

        cpi_from = CPI_INDEX.get(from_year)
        cpi_base = CPI_INDEX.get(base_year)

        if cpi_from is None:
            raise ValueError(
                f"CPI data not available for year {from_year}. "
                f"Available years: {sorted(CPI_INDEX.keys())}"
            )
        if cpi_base is None:
            raise ValueError(
                f"CPI data not available for base year {base_year}. "
                f"Available years: {sorted(CPI_INDEX.keys())}"
            )

        if cpi_from == ZERO:
            return dec_amount

        cpi_ratio = _quantize(cpi_base / cpi_from)
        deflated = _quantize(dec_amount * cpi_ratio)

        logger.debug(
            "deflate_to_base: %s from %d to %d (CPI ratio=%s) -> %s",
            str(dec_amount), from_year, base_year,
            str(cpi_ratio), str(deflated),
        )

        return deflated

    # ------------------------------------------------------------------
    # Public API: Margin Adjustment
    # ------------------------------------------------------------------

    def apply_margin_adjustment(
        self,
        revenue: Any,
        sector: str,
    ) -> Decimal:
        """Apply margin adjustment to remove profit margin from revenue.

        The margin adjustment converts purchaser-price revenue to
        producer-price revenue by removing the profit margin.
        Formula: revenue_adjusted = revenue x (1 - margin_rate)

        Args:
            revenue: Revenue amount in USD.
            sector: NAICS 3-digit sector code.

        Returns:
            Margin-adjusted revenue, quantized.

        Raises:
            ValueError: If revenue or sector is invalid.
        """
        dec_revenue = _validate_positive_decimal(revenue, "revenue")
        sector_code = _resolve_sector(sector)

        margin = SECTOR_MARGINS.get(sector_code, _DEFAULT_MARGIN)
        adjusted = _quantize(dec_revenue * (ONE - margin))

        logger.debug(
            "apply_margin_adjustment: revenue=%s, sector=%s, "
            "margin=%s, adjusted=%s",
            str(dec_revenue), sector_code, str(margin), str(adjusted),
        )

        return adjusted

    # ------------------------------------------------------------------
    # Public API: DQI Scoring
    # ------------------------------------------------------------------

    def compute_dqi_score(self) -> SpendDataQualityScore:
        """Compute the data quality indicator score for spend-based method.

        Returns the lowest-quality default DQI scores reflecting the
        inherent limitations of EEIO factors. Composite score: 30.

        Returns:
            SpendDataQualityScore with spend-based defaults.
        """
        return SpendDataQualityScore(
            reliability=_DQI_SPEND["reliability"],
            completeness=_DQI_SPEND["completeness"],
            temporal=_DQI_SPEND["temporal"],
            geographical=_DQI_SPEND["geographical"],
            technological=_DQI_SPEND["technological"],
            composite=_DQI_COMPOSITE_SPEND,
            method="spend_based",
        )

    # ------------------------------------------------------------------
    # Public API: Uncertainty
    # ------------------------------------------------------------------

    def compute_uncertainty(
        self,
        emissions_kgco2e: Decimal,
    ) -> SpendUncertaintyResult:
        """Compute uncertainty range for spend-based emissions.

        Default uncertainty: +/-50% at 95% confidence.

        Args:
            emissions_kgco2e: Central emission estimate in kgCO2e.

        Returns:
            SpendUncertaintyResult with bounds and percentage.
        """
        fraction = _DEFAULT_UNCERTAINTY_PCT / ONE_HUNDRED
        lower = _quantize(emissions_kgco2e * (ONE - fraction))
        upper = _quantize(emissions_kgco2e * (ONE + fraction))

        return SpendUncertaintyResult(
            emissions_kgco2e=emissions_kgco2e,
            lower_bound_kgco2e=lower,
            upper_bound_kgco2e=upper,
            uncertainty_pct=_DEFAULT_UNCERTAINTY_PCT,
            confidence_level=95,
            method="spend_based",
        )

    # ------------------------------------------------------------------
    # Internal: EEIO Factor Resolution
    # ------------------------------------------------------------------

    def _resolve_eeio_factor(
        self,
        sector: str,
    ) -> Tuple[Decimal, Decimal]:
        """Resolve the EEIO factor and margin for a sector.

        Args:
            sector: NAICS 3-digit sector code.

        Returns:
            Tuple of (eeio_factor, margin_rate).

        Raises:
            ValueError: If sector is not in the EEIO table.
        """
        eeio_factor = EEIO_SECTOR_FACTORS.get(sector)
        if eeio_factor is None:
            raise ValueError(
                f"No EEIO factor for sector: {sector!r}. "
                f"Available: {list(EEIO_SECTOR_FACTORS.keys())}"
            )
        margin = SECTOR_MARGINS.get(sector, _DEFAULT_MARGIN)
        return eeio_factor, margin

    # ------------------------------------------------------------------
    # Internal: Single Item Pipeline
    # ------------------------------------------------------------------

    def _calculate_single_item(
        self,
        item_id: str,
        product_name: str,
        revenue: Decimal,
        currency: str,
        sector: str,
        year: int,
    ) -> SpendBreakdown:
        """Execute the full spend-based pipeline for a single item.

        Args:
            item_id: Item identifier.
            product_name: Product name.
            revenue: Revenue in source currency.
            currency: Currency code.
            sector: NAICS sector code.
            year: Revenue year.

        Returns:
            SpendBreakdown with all intermediate values.
        """
        # Step 1: Currency conversion
        fx_rate = CURRENCY_RATES.get(currency, ONE)
        revenue_usd = self.convert_to_usd(revenue, currency)

        # Step 2: CPI deflation
        if year in CPI_INDEX and year != _CPI_BASE_YEAR:
            revenue_deflated = self.deflate_to_base(revenue_usd, year)
            cpi_from = CPI_INDEX.get(year, ONE_HUNDRED)
            cpi_base = CPI_INDEX.get(_CPI_BASE_YEAR, ONE_HUNDRED)
            cpi_ratio = _quantize(cpi_base / cpi_from) if cpi_from != ZERO else ONE
        else:
            revenue_deflated = revenue_usd
            cpi_ratio = ONE

        # Step 3: Margin adjustment
        eeio_factor, margin = self._resolve_eeio_factor(sector)
        revenue_adjusted = _quantize(revenue_deflated * (ONE - margin))

        # Step 4: EEIO emission calculation
        emissions_kgco2e = _quantize(revenue_adjusted * eeio_factor)
        emissions_tco2e = _quantize(emissions_kgco2e / ONE_THOUSAND)

        # Step 5: DQI and uncertainty
        dqi = self.compute_dqi_score()
        uncertainty = self.compute_uncertainty(emissions_kgco2e)

        # Step 6: Provenance hash
        provenance_hash = _compute_sha256({
            "engine": "SpendBasedCalculatorEngine",
            "item_id": item_id,
            "revenue_original": str(revenue),
            "currency": currency,
            "revenue_usd": str(revenue_usd),
            "revenue_deflated": str(revenue_deflated),
            "revenue_adjusted": str(revenue_adjusted),
            "sector": sector,
            "eeio_factor": str(eeio_factor),
            "margin": str(margin),
            "fx_rate": str(fx_rate),
            "cpi_ratio": str(cpi_ratio),
            "emissions_kgco2e": str(emissions_kgco2e),
            "agent_id": AGENT_ID,
            "version": VERSION,
        })

        sector_name = SECTOR_NAMES.get(sector, "Unknown Sector")

        return SpendBreakdown(
            item_id=item_id,
            product_name=product_name,
            sector=sector,
            sector_name=sector_name,
            revenue_original=revenue,
            currency=currency,
            revenue_usd=revenue_usd,
            revenue_deflated=revenue_deflated,
            revenue_adjusted=revenue_adjusted,
            eeio_factor=eeio_factor,
            margin_rate=margin,
            cpi_ratio=cpi_ratio,
            fx_rate=fx_rate,
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_tco2e,
            dqi=dqi,
            uncertainty=uncertainty,
            provenance_hash=provenance_hash,
            calculated_at=_utcnow(),
        )

    def _calculate_from_dict(
        self,
        item: Dict[str, Any],
    ) -> SpendBreakdown:
        """Calculate spend-based emissions from a dictionary item.

        Args:
            item: Dictionary with item_id, product_name, revenue,
                currency, sector, year fields.

        Returns:
            SpendBreakdown.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        item_id = str(item.get("item_id", str(uuid.uuid4())))
        product_name = str(item.get("product_name", ""))
        revenue = _validate_positive_decimal(
            item.get("revenue"), "revenue",
        )
        currency = _resolve_currency(item.get("currency", "USD"))
        sector = _resolve_sector(item.get("sector"))
        year = int(item.get("year", _CPI_BASE_YEAR))

        return self._calculate_single_item(
            item_id=item_id,
            product_name=product_name,
            revenue=revenue,
            currency=currency,
            sector=sector,
            year=year,
        )

    # ------------------------------------------------------------------
    # Internal: Provenance
    # ------------------------------------------------------------------

    def _build_provenance(
        self,
        inputs: Dict[str, Any],
        result: Dict[str, Any],
    ) -> str:
        """Build a SHA-256 provenance hash.

        Args:
            inputs: Input parameters dictionary.
            result: Output values dictionary.

        Returns:
            64-character hex SHA-256 digest.
        """
        provenance_data = {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "engine": "SpendBasedCalculatorEngine",
            "method": "spend_based",
            "inputs": inputs,
            "result": result,
            "timestamp": _utcnow().isoformat(),
        }
        return _compute_sha256(provenance_data)

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_batch_input(
        self,
        items: List[Dict[str, Any]],
    ) -> None:
        """Validate the batch items list.

        Args:
            items: List of item dictionaries.

        Raises:
            ValueError: If list is None, empty, or too large.
            TypeError: If items is not a list.
        """
        if items is None:
            raise ValueError("Items list must not be None")
        if not isinstance(items, list):
            raise TypeError(
                f"Items must be a list, got {type(items).__name__}"
            )
        if len(items) == 0:
            raise ValueError("Items list must not be empty")
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Items list exceeds maximum of {_MAX_BATCH_SIZE}, "
                f"got {len(items)}"
            )

    # ------------------------------------------------------------------
    # Internal: Counter Updates
    # ------------------------------------------------------------------

    def _update_counters(
        self,
        emissions_kgco2e: Decimal,
        revenue_usd: Decimal,
        item_count: int,
    ) -> None:
        """Update internal counters thread-safely.

        Args:
            emissions_kgco2e: Emissions to add.
            revenue_usd: Revenue to add.
            item_count: Number of items processed.
        """
        with self._lock:
            self._calculation_count += 1
            self._batch_count += item_count
            self._total_emissions_kgco2e += emissions_kgco2e
            self._total_revenue_usd += revenue_usd
            self._last_calculation_time = _utcnow()

    # ------------------------------------------------------------------
    # Public API: Aggregation Helpers
    # ------------------------------------------------------------------

    def aggregate_by_sector(
        self,
        breakdowns: List[SpendBreakdown],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by NAICS sector.

        Args:
            breakdowns: List of SpendBreakdown objects.

        Returns:
            Dictionary mapping sector code to total kgCO2e.
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            sector = getattr(bd, "sector", None)
            emissions = getattr(bd, "emissions_kgco2e", ZERO)
            if sector and emissions:
                result[sector] = _quantize(result[sector] + emissions)
        return dict(result)

    def aggregate_by_currency(
        self,
        breakdowns: List[SpendBreakdown],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by source currency.

        Args:
            breakdowns: List of SpendBreakdown objects.

        Returns:
            Dictionary mapping currency to total kgCO2e.
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            currency = getattr(bd, "currency", None)
            emissions = getattr(bd, "emissions_kgco2e", ZERO)
            if currency and emissions:
                result[currency] = _quantize(result[currency] + emissions)
        return dict(result)

    def get_eeio_factor(self, sector: str) -> Optional[Decimal]:
        """Get the EEIO factor for a sector.

        Args:
            sector: NAICS sector code.

        Returns:
            EEIO factor or None.
        """
        try:
            code = _resolve_sector(sector)
            return EEIO_SECTOR_FACTORS.get(code)
        except ValueError:
            return None

    def get_margin(self, sector: str) -> Optional[Decimal]:
        """Get the margin rate for a sector.

        Args:
            sector: NAICS sector code.

        Returns:
            Margin rate or None.
        """
        try:
            code = _resolve_sector(sector)
            return SECTOR_MARGINS.get(code)
        except ValueError:
            return None

    def get_fx_rate(self, currency: str) -> Optional[Decimal]:
        """Get the exchange rate for a currency.

        Args:
            currency: Currency code.

        Returns:
            Exchange rate to USD or None.
        """
        try:
            code = _resolve_currency(currency)
            return CURRENCY_RATES.get(code)
        except ValueError:
            return None

    def get_cpi(self, year: int) -> Optional[Decimal]:
        """Get the CPI index for a year.

        Args:
            year: Calendar year.

        Returns:
            CPI index value or None.
        """
        return CPI_INDEX.get(year)

    def list_sectors(self) -> List[str]:
        """List all available NAICS sectors.

        Returns:
            List of sector code strings.
        """
        return list(EEIO_SECTOR_FACTORS.keys())

    def list_currencies(self) -> List[str]:
        """List all supported currencies.

        Returns:
            List of currency code strings.
        """
        return list(CURRENCY_RATES.keys())

    def list_cpi_years(self) -> List[int]:
        """List all available CPI years.

        Returns:
            Sorted list of years.
        """
        return sorted(CPI_INDEX.keys())
