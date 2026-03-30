# -*- coding: utf-8 -*-
"""
SpendBasedCalculatorEngine - Engine 2: Purchased Goods & Services Agent (AGENT-MRV-014)

Core calculation engine implementing the GHG Protocol Scope 3 Category 1
spend-based method using Environmentally-Extended Input-Output (EEIO)
emission factors. Converts procurement spend data into greenhouse gas
emission estimates via a deterministic pipeline of currency conversion,
inflation deflation, margin removal, NAICS sector resolution, and EEIO
factor application.

The spend-based method is the broadest-coverage but lowest-accuracy
approach defined in the GHG Protocol Corporate Value Chain (Scope 3)
Standard. It is suitable as a screening method and for procurement
categories where physical quantity or supplier-specific data is
unavailable. Typical uncertainty range is +/- 50-100%.

Pipeline:
    Spend_local
        -> (/ FX_rate) -> Spend_USD
        -> (* CPI_base / CPI_current) -> Spend_deflated
        -> (* (1 - margin_rate / 100)) -> Spend_producer
        -> (* EEIO_factor) -> Emissions_kgCO2e
        -> (/ 1000) -> Emissions_tCO2e

Supported EEIO databases:
    - EPA USEEIO v1.2/v1.3 (1,016 US NAICS-6 commodities)
    - EXIOBASE 3.8 (163 product groups x 49 regions)
    - WIOD 2016 (43 countries x 56 sectors)
    - GTAP 11 (141 regions x 65 sectors)
    - DEFRA EEIO (UK sector-level factors)

Supported currencies: 20 ISO 4217 codes (USD, EUR, GBP, JPY, CNY,
INR, CAD, AUD, CHF, KRW, BRL, MXN, SGD, HKD, SEK, NOK, DKK, PLN,
CZK, ZAR).

Zero-Hallucination Guarantees:
    - All calculations use Python ``Decimal`` with 8 decimal places
    - No LLM calls in the calculation path
    - Every intermediate value is deterministic and traceable
    - SHA-256 provenance hash for every result
    - Thread-safe singleton with ``threading.RLock``

Data Quality Indicator (DQI) Scoring:
    Spend-based calculations receive default DQI scores reflecting the
    inherent limitations of EEIO factors (temporal 4, geographical 3-4,
    technological 4, completeness 3, reliability 4) per GHG Protocol
    Scope 3 Standard Chapter 7.

Example:
    >>> from greenlang.agents.mrv.purchased_goods_services.spend_based_calculator import (
    ...     SpendBasedCalculatorEngine,
    ... )
    >>> from greenlang.agents.mrv.purchased_goods_services.models import (
    ...     ProcurementItem, EEIODatabase, CurrencyCode,
    ... )
    >>> from decimal import Decimal
    >>> engine = SpendBasedCalculatorEngine()
    >>> item = ProcurementItem(
    ...     description="Office IT Equipment",
    ...     spend_amount=Decimal("50000.00"),
    ...     currency=CurrencyCode.USD,
    ...     naics_code="334111",
    ...     procurement_type="goods",
    ... )
    >>> result = engine.calculate_single(item)
    >>> assert result.emissions_kgco2e > Decimal("0")
    >>> assert result.eeio_sector_code == "334111"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
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
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.mrv.purchased_goods_services.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    ZERO,
    ONE,
    ONE_HUNDRED,
    ONE_THOUSAND,
    DECIMAL_PLACES,
    CalculationMethod,
    SpendClassificationSystem,
    EEIODatabase,
    CurrencyCode,
    DQIDimension,
    DQIScore,
    EmissionGas,
    ProcurementType,
    EEIO_EMISSION_FACTORS,
    CURRENCY_EXCHANGE_RATES,
    INDUSTRY_MARGIN_PERCENTAGES,
    DQI_SCORE_VALUES,
    UNCERTAINTY_RANGES,
    PEDIGREE_UNCERTAINTY_FACTORS,
    ProcurementItem,
    SpendRecord,
    SpendBasedResult,
    EEIOFactor,
    DQIAssessment,
)
from greenlang.agents.mrv.purchased_goods_services.config import PurchasedGoodsServicesConfig
from greenlang.agents.mrv.purchased_goods_services.metrics import PurchasedGoodsServicesMetrics
from greenlang.schemas import utcnow
from greenlang.agents.mrv.purchased_goods_services.provenance import (
    PurchasedGoodsProvenanceTracker,
    ProvenanceStage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "SpendBasedCalculatorEngine",
]

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal(10) ** -DECIMAL_PLACES  # 8 decimal places

# ---------------------------------------------------------------------------
# Default DQI scores for spend-based method
# Per GHG Protocol Scope 3 Standard Chapter 7 guidance:
#   - Spend-based EEIO factors are the lowest-accuracy method
#   - Default scores reflect inherent EEIO limitations
# ---------------------------------------------------------------------------

_DEFAULT_DQI_TEMPORAL_SCORE = Decimal("4.0")
_DEFAULT_DQI_GEOGRAPHICAL_SCORE_DOMESTIC = Decimal("3.0")
_DEFAULT_DQI_GEOGRAPHICAL_SCORE_FOREIGN = Decimal("4.0")
_DEFAULT_DQI_TECHNOLOGICAL_SCORE = Decimal("4.0")
_DEFAULT_DQI_COMPLETENESS_SCORE = Decimal("3.0")
_DEFAULT_DQI_RELIABILITY_SCORE = Decimal("4.0")

# ---------------------------------------------------------------------------
# EF hierarchy level for spend-based (level 7 = national EEIO)
# ---------------------------------------------------------------------------

_SPEND_BASED_EF_HIERARCHY_LEVEL = 7

# ---------------------------------------------------------------------------
# Default margin rate when NAICS sector is not in the margin table
# ---------------------------------------------------------------------------

_DEFAULT_MARGIN_RATE = Decimal("20.0")

# ---------------------------------------------------------------------------
# NAICS code to sector name mapping for aggregation display
# ---------------------------------------------------------------------------

_NAICS_SECTOR_NAMES: Dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing",
    "21": "Mining, Quarrying, Oil/Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing - Food/Textile/Apparel",
    "32": "Manufacturing - Chemical/Plastics/Paper",
    "33": "Manufacturing - Metals/Machinery/Electronics",
    "42": "Wholesale Trade",
    "44": "Retail Trade - Store",
    "45": "Retail Trade - Non-store",
    "48": "Transportation",
    "49": "Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management of Companies",
    "56": "Administrative/Waste Services",
    "61": "Educational Services",
    "62": "Health Care",
    "71": "Arts, Entertainment",
    "72": "Accommodation and Food Services",
    "81": "Other Services",
    "92": "Public Administration",
}

# ---------------------------------------------------------------------------
# Maximum batch size per calculation invocation
# ---------------------------------------------------------------------------

_MAX_BATCH_SIZE = 100_000

# ---------------------------------------------------------------------------
# Database source labels for provenance metadata
# ---------------------------------------------------------------------------

_EEIO_DATABASE_LABELS: Dict[EEIODatabase, str] = {
    EEIODatabase.EPA_USEEIO: "EPA USEEIO v1.2 (2019 base year, 2021 USD)",
    EEIODatabase.EXIOBASE: "EXIOBASE 3.8 (multi-regional, EUR)",
    EEIODatabase.WIOD: "WIOD 2016 (43 countries, 56 sectors)",
    EEIODatabase.GTAP: "GTAP 11 (141 regions, 65 sectors)",
    EEIODatabase.DEFRA_EEIO: "DEFRA/DESNZ EEIO (UK sector-level, GBP)",
}

# ===========================================================================
# Helper utilities
# ===========================================================================

def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the configured precision.

    Args:
        value: Raw Decimal value.

    Returns:
        Decimal quantized to DECIMAL_PLACES places using ROUND_HALF_UP.
    """
    try:
        return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        logger.warning(
            "Quantize failed for value=%s, returning ZERO", value
        )
        return ZERO

def _hash_data(data: Any) -> str:
    """Compute SHA-256 hex digest for arbitrary data.

    Args:
        data: Value to hash. Dicts and Decimals are serialized
            deterministically.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    if isinstance(data, dict):
        serialized = json.dumps(
            data, sort_keys=True, default=str
        ).encode("utf-8")
    elif isinstance(data, Decimal):
        serialized = str(data).encode("utf-8")
    elif isinstance(data, str):
        serialized = data.encode("utf-8")
    else:
        serialized = json.dumps(data, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()

def _compute_provenance_hash(
    item_id: str,
    spend_original: Decimal,
    spend_usd: Decimal,
    spend_deflated: Decimal,
    spend_producer: Decimal,
    eeio_factor: Decimal,
    emissions_kgco2e: Decimal,
    eeio_database: str,
    sector_code: str,
    fx_rate: Decimal,
    cpi_ratio: Decimal,
    margin_rate: Decimal,
) -> str:
    """Compute deterministic SHA-256 provenance hash for one spend-based calculation.

    Captures every intermediate value in the spend-based pipeline to
    enable full audit trail reconstruction.

    Args:
        item_id: Procurement item identifier.
        spend_original: Original spend in source currency.
        spend_usd: Spend converted to USD.
        spend_deflated: Spend after inflation deflation.
        spend_producer: Spend after margin removal.
        eeio_factor: EEIO emission factor applied.
        emissions_kgco2e: Calculated emissions in kgCO2e.
        eeio_database: EEIO database identifier.
        sector_code: NAICS sector code matched.
        fx_rate: Exchange rate used.
        cpi_ratio: CPI ratio for deflation.
        margin_rate: Margin rate applied.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    payload = (
        f"spend_based|{item_id}|{spend_original}|{spend_usd}|"
        f"{spend_deflated}|{spend_producer}|{eeio_factor}|"
        f"{emissions_kgco2e}|{eeio_database}|{sector_code}|"
        f"{fx_rate}|{cpi_ratio}|{margin_rate}|{AGENT_ID}|{VERSION}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _naics_to_2digit(naics_code: str) -> str:
    """Extract 2-digit NAICS sector from a NAICS code.

    Args:
        naics_code: Full NAICS code (2-6 digits).

    Returns:
        First two characters of the NAICS code.
    """
    return naics_code[:2] if naics_code and len(naics_code) >= 2 else ""

def _sector_name_for_code(naics_code: str) -> str:
    """Get human-readable sector name from a NAICS code.

    Args:
        naics_code: Full NAICS code (2-6 digits).

    Returns:
        Sector name string, or 'Unknown Sector' if not mapped.
    """
    sector_2 = _naics_to_2digit(naics_code)
    return _NAICS_SECTOR_NAMES.get(sector_2, "Unknown Sector")

def _dqi_score_label(score: Decimal) -> str:
    """Map a numeric DQI score to its qualitative label.

    Args:
        score: Composite DQI score (1.0-5.0).

    Returns:
        Quality tier label string.
    """
    if score < Decimal("1.6"):
        return "Very Good"
    if score < Decimal("2.6"):
        return "Good"
    if score < Decimal("3.6"):
        return "Fair"
    if score < Decimal("4.6"):
        return "Poor"
    return "Very Poor"

def _pedigree_factor_for_score(score: Decimal) -> Decimal:
    """Get pedigree uncertainty factor for a DQI score.

    Maps composite score to nearest DQI label and returns the
    corresponding pedigree uncertainty multiplier from the
    ecoinvent methodology.

    Args:
        score: Composite DQI score (1.0-5.0).

    Returns:
        Pedigree uncertainty factor (1.00-1.50).
    """
    if score < Decimal("1.6"):
        return PEDIGREE_UNCERTAINTY_FACTORS[DQIScore.VERY_GOOD]
    if score < Decimal("2.6"):
        return PEDIGREE_UNCERTAINTY_FACTORS[DQIScore.GOOD]
    if score < Decimal("3.6"):
        return PEDIGREE_UNCERTAINTY_FACTORS[DQIScore.FAIR]
    if score < Decimal("4.6"):
        return PEDIGREE_UNCERTAINTY_FACTORS[DQIScore.POOR]
    return PEDIGREE_UNCERTAINTY_FACTORS[DQIScore.VERY_POOR]

# ===========================================================================
# SpendBasedCalculatorEngine
# ===========================================================================

class SpendBasedCalculatorEngine:
    """Thread-safe singleton engine for spend-based EEIO emission calculations.

    Implements the GHG Protocol Scope 3 Category 1 spend-based
    calculation method. Converts procurement spend into greenhouse gas
    emissions using EEIO factors through a deterministic pipeline of
    currency conversion, inflation deflation, margin removal, sector
    resolution, and factor application.

    The engine is a thread-safe singleton using ``threading.RLock``
    and double-checked locking. All arithmetic uses Python ``Decimal``
    with 8 decimal places (``ROUND_HALF_UP``) to guarantee
    zero-hallucination deterministic results.

    Core Pipeline (per item):
        1. **Currency Conversion**: Spend_local / FX_rate -> Spend_USD
        2. **Inflation Deflation**: Spend_USD * CPI_ratio -> Spend_deflated
        3. **Margin Removal**: Spend_deflated * (1 - margin/100) -> Spend_producer
        4. **NAICS Resolution**: Resolve item to NAICS-6 sector code
        5. **EEIO Factor Lookup**: NAICS-6 -> kgCO2e/USD factor
        6. **Emission Calculation**: Spend_producer * EEIO_factor -> kgCO2e
        7. **Unit Conversion**: kgCO2e / 1000 -> tCO2e
        8. **DQI Scoring**: Apply default spend-based DQI scores
        9. **Provenance**: SHA-256 hash of all intermediates

    Attributes:
        _config: Singleton configuration reference.
        _metrics: Singleton metrics collector reference.
        _provenance_tracker: Provenance chain tracker reference.
        _initialized: Whether the singleton has been fully initialized.
        _lock: Class-level reentrant lock for thread safety.
        _instance: Class-level singleton reference.

    Example:
        >>> engine = SpendBasedCalculatorEngine()
        >>> item = ProcurementItem(
        ...     description="Steel rebar",
        ...     spend_amount=Decimal("100000"),
        ...     currency=CurrencyCode.USD,
        ...     naics_code="331110",
        ... )
        >>> result = engine.calculate_single(item)
        >>> assert result.emissions_kgco2e > Decimal("0")
        >>> assert result.eeio_sector_code == "331110"
        >>> assert result.provenance_hash != ""

        >>> # Batch calculation
        >>> results = engine.calculate_batch([item])
        >>> assert len(results) == 1

        >>> # Singleton behavior
        >>> engine2 = SpendBasedCalculatorEngine()
        >>> assert engine is engine2
    """

    _instance: Optional[SpendBasedCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    def __new__(cls) -> SpendBasedCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with ``threading.RLock`` to ensure
        thread-safe initialization. Only one instance is created for
        the lifetime of the process.

        Returns:
            The singleton SpendBasedCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine with config, metrics, and provenance.

        Guarded by the ``_initialized`` class flag so repeated calls
        to ``__init__`` (from repeated instantiation) do not reset
        internal state.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._config = PurchasedGoodsServicesConfig()
            self._metrics = PurchasedGoodsServicesMetrics()
            self._provenance_tracker = (
                PurchasedGoodsProvenanceTracker.get_instance()
            )
            self._calculation_count: int = 0
            self._batch_count: int = 0
            self._total_emissions_kgco2e: Decimal = ZERO
            self._total_spend_processed_usd: Decimal = ZERO
            self._error_count: int = 0
            self._last_calculation_time: Optional[datetime] = None
            self.__class__._initialized = True
            logger.info(
                "SpendBasedCalculatorEngine initialized "
                "(agent=%s, version=%s, precision=%d, "
                "default_eeio=%s, margin_removal=%s, "
                "inflation_adjustment=%s)",
                AGENT_ID,
                VERSION,
                DECIMAL_PLACES,
                self._config.default_eeio_database,
                self._config.enable_margin_removal,
                self._config.enable_inflation_adjustment,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for testing purposes.

        Clears the singleton instance and the initialized flag so
        that the next instantiation creates a fresh engine. This
        method is intended for use in test fixtures only and must
        not be called in production code.

        Example:
            >>> SpendBasedCalculatorEngine.reset()
            >>> engine = SpendBasedCalculatorEngine()
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.info("SpendBasedCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Public API: Single Item Calculation
    # ------------------------------------------------------------------

    def calculate_single(
        self,
        item: ProcurementItem,
        database: EEIODatabase = EEIODatabase.EPA_USEEIO,
        cpi_ratio: Decimal = ONE,
        custom_fx_rate: Optional[Decimal] = None,
    ) -> SpendBasedResult:
        """Calculate spend-based emissions for a single procurement item.

        Executes the full spend-based pipeline: currency conversion,
        inflation deflation, margin removal, NAICS sector resolution,
        EEIO factor lookup, emission calculation, DQI scoring, and
        provenance hashing.

        Args:
            item: Procurement item with spend amount, currency, and
                optional NAICS code.
            database: EEIO database to use for factor lookup. Defaults
                to EPA USEEIO.
            cpi_ratio: CPI ratio for inflation deflation. Ratio of
                EEIO base year CPI to the current year CPI. A value
                of 1.0 means no deflation. Defaults to 1.0.
            custom_fx_rate: Optional custom exchange rate. If provided,
                overrides the built-in rate table. Must be positive.

        Returns:
            SpendBasedResult with calculated emissions, intermediate
            values, DQI scores, and provenance hash.

        Raises:
            ValueError: If the item has zero or negative spend, or if
                the NAICS code cannot be resolved, or if the EEIO
                factor lookup fails.
            TypeError: If item is not a ProcurementItem instance.

        Example:
            >>> result = engine.calculate_single(item)
            >>> print(f"Emissions: {result.emissions_tco2e} tCO2e")
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())
        status = "success"

        try:
            # -------------------------------------------------------
            # Step 0: Input validation
            # -------------------------------------------------------
            self._validate_single_input(item, cpi_ratio, custom_fx_rate)

            # -------------------------------------------------------
            # Step 1: Resolve NAICS sector code
            # -------------------------------------------------------
            naics_code = self.resolve_naics_code(item)
            if naics_code is None:
                raise ValueError(
                    f"Cannot resolve NAICS code for item "
                    f"'{item.item_id}' (description='{item.description}'). "
                    f"Provide a valid naics_code on the ProcurementItem."
                )

            # -------------------------------------------------------
            # Step 2: Resolve EEIO emission factor
            # -------------------------------------------------------
            eeio_factor = self.resolve_eeio_factor(naics_code, database)
            if eeio_factor is None:
                raise ValueError(
                    f"No EEIO factor found for NAICS code "
                    f"'{naics_code}' in database "
                    f"'{database.value}'. Ensure the NAICS-6 code "
                    f"exists in the factor table."
                )

            # -------------------------------------------------------
            # Step 3: Currency conversion to USD
            # -------------------------------------------------------
            spend_usd, fx_rate_used = self.convert_to_usd(
                item.spend_amount, item.currency, custom_fx_rate
            )

            # -------------------------------------------------------
            # Step 4: Inflation deflation (CPI adjustment)
            # -------------------------------------------------------
            if self._config.enable_inflation_adjustment:
                spend_deflated = self.deflate_to_base_year(
                    spend_usd, cpi_ratio
                )
            else:
                spend_deflated = spend_usd

            # -------------------------------------------------------
            # Step 5: Margin removal (purchaser -> producer price)
            # -------------------------------------------------------
            if self._config.enable_margin_removal:
                spend_producer, margin_rate = self.remove_margin(
                    spend_deflated, naics_code
                )
            else:
                spend_producer = spend_deflated
                margin_rate = ZERO

            # -------------------------------------------------------
            # Step 6: Emission calculation (ZERO HALLUCINATION)
            # -------------------------------------------------------
            emissions_kgco2e = _quantize(spend_producer * eeio_factor)
            emissions_tco2e = _quantize(emissions_kgco2e / ONE_THOUSAND)

            # -------------------------------------------------------
            # Step 7: Compute provenance hash
            # -------------------------------------------------------
            provenance_hash = _compute_provenance_hash(
                item_id=item.item_id,
                spend_original=item.spend_amount,
                spend_usd=spend_usd,
                spend_deflated=spend_deflated,
                spend_producer=spend_producer,
                eeio_factor=eeio_factor,
                emissions_kgco2e=emissions_kgco2e,
                eeio_database=database.value,
                sector_code=naics_code,
                fx_rate=fx_rate_used,
                cpi_ratio=cpi_ratio,
                margin_rate=margin_rate,
            )

            # -------------------------------------------------------
            # Step 8: Record provenance stage
            # -------------------------------------------------------
            self._record_provenance_stage(
                calculation_id=calculation_id,
                item_id=item.item_id,
                stage=ProvenanceStage.SPEND_BASED_CALCULATION,
                metadata={
                    "naics_code": naics_code,
                    "eeio_database": database.value,
                    "eeio_factor_kgco2e_per_usd": str(eeio_factor),
                    "spend_original": str(item.spend_amount),
                    "currency": item.currency.value,
                    "spend_usd": str(spend_usd),
                    "spend_deflated_usd": str(spend_deflated),
                    "spend_producer_usd": str(spend_producer),
                    "fx_rate": str(fx_rate_used),
                    "cpi_ratio": str(cpi_ratio),
                    "margin_rate": str(margin_rate),
                    "emissions_kgco2e": str(emissions_kgco2e),
                    "emissions_tco2e": str(emissions_tco2e),
                },
                output_data=provenance_hash,
            )

            # -------------------------------------------------------
            # Step 9: Build result
            # -------------------------------------------------------
            result = SpendBasedResult(
                item_id=item.item_id,
                emissions_kgco2e=emissions_kgco2e,
                emissions_tco2e=emissions_tco2e,
                spend_original=item.spend_amount,
                spend_usd=spend_usd,
                spend_deflated_usd=spend_deflated,
                spend_producer_usd=spend_producer,
                eeio_factor_kgco2e_per_usd=eeio_factor,
                eeio_database=database,
                eeio_sector_code=naics_code,
                currency=item.currency,
                fx_rate=fx_rate_used,
                cpi_ratio=cpi_ratio,
                margin_rate=margin_rate,
                provenance_hash=provenance_hash,
            )

            # -------------------------------------------------------
            # Step 10: Update internal counters
            # -------------------------------------------------------
            with self._lock:
                self._calculation_count += 1
                self._total_emissions_kgco2e += emissions_kgco2e
                self._total_spend_processed_usd += spend_usd
                self._last_calculation_time = utcnow()

            logger.info(
                "Spend-based calculation completed: item=%s, "
                "naics=%s, spend_usd=%s, emissions_kgco2e=%s, "
                "emissions_tco2e=%s, database=%s",
                item.item_id,
                naics_code,
                spend_usd,
                emissions_kgco2e,
                emissions_tco2e,
                database.value,
            )

            return result

        except Exception as exc:
            status = "failed"
            with self._lock:
                self._error_count += 1
            logger.error(
                "Spend-based calculation failed: item=%s, error=%s",
                item.item_id if item else "None",
                str(exc),
                exc_info=True,
            )
            raise

        finally:
            duration_s = time.monotonic() - start_time
            self._record_metrics(
                method="spend_based",
                status=status,
                duration_s=duration_s,
                emissions_kgco2e=float(
                    emissions_kgco2e if status == "success" else 0
                ),
                spend_usd=float(
                    spend_usd if status == "success" else 0
                ),
            )

    # ------------------------------------------------------------------
    # Public API: Batch Calculation
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        items: List[ProcurementItem],
        database: EEIODatabase = EEIODatabase.EPA_USEEIO,
        cpi_ratio: Decimal = ONE,
    ) -> List[SpendBasedResult]:
        """Calculate spend-based emissions for a batch of procurement items.

        Iterates through the item list and calculates each using
        ``calculate_single``. Items that fail validation or factor
        resolution are logged at WARNING level and skipped, returning
        only successful results.

        Args:
            items: List of procurement items. Maximum batch size is
                100,000 items.
            database: EEIO database for factor lookup.
            cpi_ratio: CPI ratio for inflation deflation.

        Returns:
            List of SpendBasedResult for successfully calculated items.
            Failed items are excluded from the result list but logged.

        Raises:
            ValueError: If items list is empty or exceeds maximum batch
                size.

        Example:
            >>> results = engine.calculate_batch(items_list)
            >>> print(f"Calculated {len(results)} of {len(items_list)}")
        """
        start_time = time.monotonic()

        if not items:
            raise ValueError("Items list cannot be empty for batch calculation")
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum "
                f"allowed {_MAX_BATCH_SIZE}"
            )

        logger.info(
            "Starting spend-based batch calculation: "
            "item_count=%d, database=%s, cpi_ratio=%s",
            len(items),
            database.value,
            cpi_ratio,
        )

        results: List[SpendBasedResult] = []
        success_count = 0
        fail_count = 0

        for idx, item in enumerate(items):
            try:
                result = self.calculate_single(
                    item=item,
                    database=database,
                    cpi_ratio=cpi_ratio,
                )
                results.append(result)
                success_count += 1
            except (ValueError, TypeError) as exc:
                fail_count += 1
                logger.warning(
                    "Batch item %d/%d failed (item_id=%s): %s",
                    idx + 1,
                    len(items),
                    item.item_id,
                    str(exc),
                )
            except Exception as exc:
                fail_count += 1
                logger.error(
                    "Batch item %d/%d unexpected error (item_id=%s): %s",
                    idx + 1,
                    len(items),
                    item.item_id,
                    str(exc),
                    exc_info=True,
                )

        duration_s = time.monotonic() - start_time

        with self._lock:
            self._batch_count += 1

        logger.info(
            "Spend-based batch calculation completed: "
            "total=%d, success=%d, failed=%d, duration=%.3fs",
            len(items),
            success_count,
            fail_count,
            duration_s,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: Currency Conversion
    # ------------------------------------------------------------------

    def convert_to_usd(
        self,
        amount: Decimal,
        currency: CurrencyCode,
        custom_fx_rate: Optional[Decimal] = None,
    ) -> Tuple[Decimal, Decimal]:
        """Convert a spend amount from local currency to USD.

        Uses the built-in exchange rate table or a custom rate.
        Exchange rates are expressed as units of foreign currency
        per 1 USD. Therefore: Spend_USD = Spend_local / FX_rate.

        For USD amounts, the rate is 1.0 and no conversion occurs.

        Args:
            amount: Spend amount in local currency.
            currency: ISO 4217 currency code.
            custom_fx_rate: Optional custom exchange rate. If provided,
                must be positive. Overrides the built-in table.

        Returns:
            Tuple of (spend_usd, fx_rate_used). Both are quantized
            to DECIMAL_PLACES precision.

        Raises:
            ValueError: If currency is not supported, or if
                custom_fx_rate is zero or negative, or if amount
                is negative.

        Example:
            >>> spend_usd, rate = engine.convert_to_usd(
            ...     Decimal("92410"), CurrencyCode.EUR
            ... )
            >>> print(f"${spend_usd} at rate {rate}")
        """
        if amount < ZERO:
            raise ValueError(
                f"Spend amount cannot be negative: {amount}"
            )

        # Determine FX rate
        if custom_fx_rate is not None:
            if custom_fx_rate <= ZERO:
                raise ValueError(
                    f"Custom FX rate must be positive: {custom_fx_rate}"
                )
            fx_rate = custom_fx_rate
        else:
            fx_rate = CURRENCY_EXCHANGE_RATES.get(currency)
            if fx_rate is None:
                raise ValueError(
                    f"Unsupported currency: {currency.value}. "
                    f"Supported: {sorted(c.value for c in CURRENCY_EXCHANGE_RATES)}"
                )

        # Handle zero spend (no conversion needed)
        if amount == ZERO:
            return ZERO, _quantize(fx_rate)

        # Convert: Spend_USD = Spend_local / FX_rate
        spend_usd = _quantize(amount / fx_rate)
        fx_rate_quantized = _quantize(fx_rate)

        logger.debug(
            "Currency conversion: %s %s -> %s USD (rate=%s)",
            amount,
            currency.value,
            spend_usd,
            fx_rate_quantized,
        )

        return spend_usd, fx_rate_quantized

    # ------------------------------------------------------------------
    # Public API: Inflation Deflation
    # ------------------------------------------------------------------

    def deflate_to_base_year(
        self,
        amount_usd: Decimal,
        cpi_ratio: Decimal,
    ) -> Decimal:
        """Deflate a USD amount to the EEIO base year using CPI ratio.

        Formula: Spend_deflated = Spend_USD * CPI_ratio
        Where CPI_ratio = CPI_base_year / CPI_current_year

        A CPI ratio > 1.0 indicates the base year had higher prices
        (deflation increases the effective spend). A ratio < 1.0
        indicates the current year has higher prices (deflation
        decreases the effective spend).

        Args:
            amount_usd: Spend amount in current-year USD.
            cpi_ratio: Ratio of EEIO base year CPI to current year
                CPI. Must be positive. A value of 1.0 means no
                adjustment.

        Returns:
            Deflated spend amount quantized to DECIMAL_PLACES.

        Raises:
            ValueError: If cpi_ratio is zero or negative, or if
                amount_usd is negative.

        Example:
            >>> deflated = engine.deflate_to_base_year(
            ...     Decimal("100000"), Decimal("0.95")
            ... )
            >>> print(f"Deflated: ${deflated}")
        """
        if cpi_ratio <= ZERO:
            raise ValueError(
                f"CPI ratio must be positive: {cpi_ratio}"
            )
        if amount_usd < ZERO:
            raise ValueError(
                f"Amount cannot be negative: {amount_usd}"
            )

        if amount_usd == ZERO:
            return ZERO

        deflated = _quantize(amount_usd * cpi_ratio)

        logger.debug(
            "Inflation deflation: $%s * %s = $%s",
            amount_usd,
            cpi_ratio,
            deflated,
        )

        return deflated

    # ------------------------------------------------------------------
    # Public API: Margin Removal
    # ------------------------------------------------------------------

    def remove_margin(
        self,
        amount: Decimal,
        naics_code: str,
    ) -> Tuple[Decimal, Decimal]:
        """Remove trade margins to convert from purchaser to producer price.

        EEIO factors are typically based on producer (basic) prices.
        Procurement spend is typically at purchaser prices which include
        wholesale, retail, and transport margins. This method removes
        the margin to align the spend with the EEIO factor price basis.

        Formula: Spend_producer = Spend * (1 - margin_rate / 100)

        The margin rate is looked up from the INDUSTRY_MARGIN_PERCENTAGES
        table using the 2-digit NAICS sector code. If the sector is not
        in the table, a default rate of 20% is applied.

        Args:
            amount: Spend amount after currency conversion and
                inflation deflation.
            naics_code: Full NAICS code (at least 2 digits).

        Returns:
            Tuple of (spend_producer, margin_rate). Both are quantized
            to DECIMAL_PLACES precision. margin_rate is the percentage
            value (e.g. 25.0 for 25%).

        Raises:
            ValueError: If amount is negative, or if naics_code is
                empty or too short.

        Example:
            >>> producer, margin = engine.remove_margin(
            ...     Decimal("100000"), "331110"
            ... )
            >>> print(f"Producer price: ${producer} (margin: {margin}%)")
        """
        if amount < ZERO:
            raise ValueError(
                f"Amount cannot be negative for margin removal: {amount}"
            )
        if not naics_code or len(naics_code) < 2:
            raise ValueError(
                f"NAICS code must be at least 2 digits: '{naics_code}'"
            )

        if amount == ZERO:
            return ZERO, ZERO

        # Lookup margin by 2-digit NAICS sector
        sector_2 = _naics_to_2digit(naics_code)
        margin_rate = INDUSTRY_MARGIN_PERCENTAGES.get(
            sector_2, _DEFAULT_MARGIN_RATE
        )

        # Calculate: Spend_producer = Spend * (1 - margin / 100)
        margin_fraction = ONE - _quantize(margin_rate / ONE_HUNDRED)
        spend_producer = _quantize(amount * margin_fraction)

        logger.debug(
            "Margin removal: $%s * (1 - %s/100) = $%s (sector=%s)",
            amount,
            margin_rate,
            spend_producer,
            sector_2,
        )

        return spend_producer, _quantize(margin_rate)

    # ------------------------------------------------------------------
    # Public API: NAICS Code Resolution
    # ------------------------------------------------------------------

    def resolve_naics_code(
        self,
        item: ProcurementItem,
    ) -> Optional[str]:
        """Resolve the best available NAICS-6 sector code for an item.

        Attempts to resolve a NAICS code using the following priority:
        1. Explicit ``naics_code`` field on the item (highest priority)
        2. Cross-mapping from ``nace_code`` (future extension)
        3. Cross-mapping from ``isic_code`` (future extension)
        4. Cross-mapping from ``unspsc_code`` (future extension)
        5. Material category fallback mapping

        Currently only the explicit NAICS code and material category
        fallback are implemented. Cross-mapping from NACE, ISIC, and
        UNSPSC will be added in future versions via the Classification
        Engine.

        Args:
            item: Procurement item with classification codes.

        Returns:
            Resolved NAICS-6 code string, or None if no code can be
            determined.

        Example:
            >>> code = engine.resolve_naics_code(item)
            >>> if code:
            ...     print(f"Resolved NAICS: {code}")
        """
        # Priority 1: Explicit NAICS code
        if item.naics_code and len(item.naics_code) >= 2:
            naics = item.naics_code.strip()
            if naics:
                logger.debug(
                    "NAICS resolved from item field: %s (item=%s)",
                    naics,
                    item.item_id,
                )
                return naics

        # Priority 2-4: Cross-mapping placeholders
        # These will be implemented when the ClassificationEngine
        # (Engine 3) is built. For now, log and fall through.
        if item.nace_code:
            logger.debug(
                "NACE code present but cross-mapping not yet "
                "implemented: nace=%s (item=%s)",
                item.nace_code,
                item.item_id,
            )

        if item.isic_code:
            logger.debug(
                "ISIC code present but cross-mapping not yet "
                "implemented: isic=%s (item=%s)",
                item.isic_code,
                item.item_id,
            )

        if item.unspsc_code:
            logger.debug(
                "UNSPSC code present but cross-mapping not yet "
                "implemented: unspsc=%s (item=%s)",
                item.unspsc_code,
                item.item_id,
            )

        # Priority 5: Material category fallback
        naics_from_category = self._material_category_to_naics(
            item.material_category
        )
        if naics_from_category:
            logger.debug(
                "NAICS resolved from material category: %s -> %s "
                "(item=%s)",
                item.material_category,
                naics_from_category,
                item.item_id,
            )
            return naics_from_category

        logger.warning(
            "Cannot resolve NAICS code for item: id=%s, "
            "description='%s'",
            item.item_id,
            item.description[:100],
        )
        return None

    # ------------------------------------------------------------------
    # Public API: EEIO Factor Resolution
    # ------------------------------------------------------------------

    def resolve_eeio_factor(
        self,
        naics_code: str,
        database: EEIODatabase,
    ) -> Optional[Decimal]:
        """Resolve the EEIO emission factor for a NAICS sector code.

        Looks up the emission factor in the EEIO factor table using
        the exact NAICS code first, then progressively shorter
        prefixes (5-digit, 4-digit, 3-digit, 2-digit) to find the
        best available match.

        For the EPA USEEIO database, factors are in kgCO2e per USD
        (purchaser price, 2021 USD). Other databases may use
        different units and are normalized during lookup.

        Args:
            naics_code: NAICS sector code (2-6 digits).
            database: EEIO database to search.

        Returns:
            Emission factor as Decimal (kgCO2e per USD), or None
            if no factor is found for any prefix of the NAICS code.

        Example:
            >>> factor = engine.resolve_eeio_factor("331110", EEIODatabase.EPA_USEEIO)
            >>> if factor:
            ...     print(f"Factor: {factor} kgCO2e/USD")
        """
        if not naics_code:
            return None

        # For EPA USEEIO, look up from built-in table with progressive
        # prefix matching (6-digit, 5-digit, ..., 2-digit)
        if database == EEIODatabase.EPA_USEEIO:
            return self._lookup_useeio_factor(naics_code)

        # For other databases, attempt the same built-in lookup as a
        # fallback. Full multi-database support will be implemented
        # when the EEIO Database Engine (Engine 7) is built.
        logger.debug(
            "Database %s not natively supported yet, "
            "falling back to EPA USEEIO factors for NAICS=%s",
            database.value,
            naics_code,
        )
        return self._lookup_useeio_factor(naics_code)

    # ------------------------------------------------------------------
    # Public API: DQI Scoring
    # ------------------------------------------------------------------

    def score_dqi_spend_based(
        self,
        item: ProcurementItem,
        database: EEIODatabase,
        result: SpendBasedResult,
    ) -> DQIAssessment:
        """Score data quality for a spend-based calculation result.

        Applies the GHG Protocol Scope 3 Standard Chapter 7 DQI
        framework with default scores appropriate for the spend-based
        EEIO method:
        - Temporal: Score 4 (EEIO factors are typically 3-5 years old)
        - Geographical: Score 3 (domestic) or 4 (foreign/global)
        - Technological: Score 4 (sector-average, not product-specific)
        - Completeness: Score 3 (cradle-to-gate included, sector level)
        - Reliability: Score 4 (model-derived, not measured)

        The composite score is the arithmetic mean of all five
        dimensions. The quality tier is derived from the composite.

        Args:
            item: Source procurement item.
            database: EEIO database used.
            result: SpendBasedResult to score.

        Returns:
            DQIAssessment with all five dimension scores, composite
            score, quality tier, uncertainty factor, and findings.

        Example:
            >>> dqi = engine.score_dqi_spend_based(item, database, result)
            >>> print(f"Composite DQI: {dqi.composite_score}")
        """
        # Determine geographical score based on currency
        is_domestic_us = (item.currency == CurrencyCode.USD)
        geo_score = (
            _DEFAULT_DQI_GEOGRAPHICAL_SCORE_DOMESTIC
            if is_domestic_us and database == EEIODatabase.EPA_USEEIO
            else _DEFAULT_DQI_GEOGRAPHICAL_SCORE_FOREIGN
        )

        temporal_score = _DEFAULT_DQI_TEMPORAL_SCORE
        technological_score = _DEFAULT_DQI_TECHNOLOGICAL_SCORE
        completeness_score = _DEFAULT_DQI_COMPLETENESS_SCORE
        reliability_score = _DEFAULT_DQI_RELIABILITY_SCORE

        # Adjust temporal score if EEIO base year is recent
        eeio_base_year = self._config.eeio_base_year
        current_year = utcnow().year
        year_gap = current_year - eeio_base_year
        if year_gap <= 2:
            temporal_score = Decimal("3.0")
        elif year_gap <= 5:
            temporal_score = Decimal("4.0")
        else:
            temporal_score = Decimal("5.0")

        # Adjust completeness if NAICS code is granular (6-digit)
        if (
            result.eeio_sector_code
            and len(result.eeio_sector_code) >= 6
        ):
            completeness_score = Decimal("3.0")
        else:
            completeness_score = Decimal("4.0")

        # Compute composite score (arithmetic mean of 5 dimensions)
        five = Decimal("5.0")
        composite_raw = (
            temporal_score
            + geo_score
            + technological_score
            + completeness_score
            + reliability_score
        ) / five
        composite_score = _quantize(composite_raw)

        # Determine quality tier
        quality_tier = _dqi_score_label(composite_score)

        # Compute pedigree uncertainty factor
        uncertainty_factor = _pedigree_factor_for_score(composite_score)

        # Generate findings
        findings = self._generate_dqi_findings(
            temporal_score=temporal_score,
            geo_score=geo_score,
            technological_score=technological_score,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            database=database,
            item=item,
            result=result,
        )

        return DQIAssessment(
            item_id=item.item_id,
            calculation_method=CalculationMethod.SPEND_BASED,
            temporal_score=temporal_score,
            geographical_score=geo_score,
            technological_score=technological_score,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            composite_score=composite_score,
            quality_tier=quality_tier,
            uncertainty_factor=uncertainty_factor,
            findings=findings,
            ef_hierarchy_level=_SPEND_BASED_EF_HIERARCHY_LEVEL,
        )

    # ------------------------------------------------------------------
    # Public API: Sector Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_sector(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Aggregate spend-based results by NAICS 2-digit sector.

        Groups results by the first two digits of the EEIO sector
        code and computes subtotals for emissions, spend, and item
        count per sector. Useful for hot-spot identification and
        sector-level reporting.

        Args:
            results: List of SpendBasedResult from batch calculation.

        Returns:
            Dictionary mapping 2-digit NAICS code to aggregation dict
            with keys:
            - ``sector_name``: Human-readable sector name
            - ``emissions_kgco2e``: Total kgCO2e for the sector
            - ``emissions_tco2e``: Total tCO2e for the sector
            - ``spend_usd``: Total USD spend for the sector
            - ``spend_producer_usd``: Total producer-price spend
            - ``item_count``: Number of items in the sector
            - ``avg_eeio_factor``: Weighted average EEIO factor
            - ``emissions_pct``: Percentage of total emissions

        Example:
            >>> sectors = engine.aggregate_by_sector(results)
            >>> for code, data in sorted(sectors.items()):
            ...     print(f"{data['sector_name']}: {data['emissions_tco2e']} tCO2e")
        """
        if not results:
            return {}

        sector_data: Dict[str, Dict[str, Decimal]] = {}
        total_emissions = ZERO

        # First pass: accumulate per sector
        for r in results:
            sector_2 = _naics_to_2digit(r.eeio_sector_code)
            if not sector_2:
                sector_2 = "XX"

            if sector_2 not in sector_data:
                sector_data[sector_2] = {
                    "sector_name": _sector_name_for_code(
                        r.eeio_sector_code
                    ),
                    "emissions_kgco2e": ZERO,
                    "emissions_tco2e": ZERO,
                    "spend_usd": ZERO,
                    "spend_producer_usd": ZERO,
                    "item_count": ZERO,
                    "weighted_factor_sum": ZERO,
                    "avg_eeio_factor": ZERO,
                    "emissions_pct": ZERO,
                }

            data = sector_data[sector_2]
            data["emissions_kgco2e"] += r.emissions_kgco2e
            data["emissions_tco2e"] += r.emissions_tco2e
            data["spend_usd"] += r.spend_usd
            data["spend_producer_usd"] += r.spend_producer_usd
            data["item_count"] += ONE
            data["weighted_factor_sum"] += (
                r.eeio_factor_kgco2e_per_usd * r.spend_producer_usd
            )
            total_emissions += r.emissions_kgco2e

        # Second pass: compute averages and percentages
        for sector_2, data in sector_data.items():
            if data["spend_producer_usd"] > ZERO:
                data["avg_eeio_factor"] = _quantize(
                    data["weighted_factor_sum"]
                    / data["spend_producer_usd"]
                )
            if total_emissions > ZERO:
                data["emissions_pct"] = _quantize(
                    (data["emissions_kgco2e"] / total_emissions)
                    * ONE_HUNDRED
                )
            # Quantize accumulated values
            data["emissions_kgco2e"] = _quantize(
                data["emissions_kgco2e"]
            )
            data["emissions_tco2e"] = _quantize(
                data["emissions_tco2e"]
            )
            data["spend_usd"] = _quantize(data["spend_usd"])
            data["spend_producer_usd"] = _quantize(
                data["spend_producer_usd"]
            )
            # Remove working field
            del data["weighted_factor_sum"]

        logger.info(
            "Sector aggregation completed: %d sectors from %d results",
            len(sector_data),
            len(results),
        )

        return sector_data

    # ------------------------------------------------------------------
    # Public API: Supplier Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_supplier(
        self,
        results: List[SpendBasedResult],
        items: List[ProcurementItem],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Aggregate spend-based results by supplier.

        Groups results by supplier identifier and computes subtotals
        for emissions, spend, and item count per supplier. Requires
        the original items list to resolve supplier IDs.

        Args:
            results: List of SpendBasedResult from batch calculation.
            items: Original procurement items (for supplier_id lookup).

        Returns:
            Dictionary mapping supplier_id to aggregation dict with
            keys:
            - ``supplier_name``: Supplier name (if available)
            - ``emissions_kgco2e``: Total kgCO2e for the supplier
            - ``emissions_tco2e``: Total tCO2e for the supplier
            - ``spend_usd``: Total USD spend for the supplier
            - ``item_count``: Number of items for the supplier
            - ``emissions_pct``: Percentage of total emissions

        Example:
            >>> suppliers = engine.aggregate_by_supplier(results, items)
            >>> for sid, data in suppliers.items():
            ...     print(f"{data['supplier_name']}: {data['emissions_tco2e']} tCO2e")
        """
        if not results:
            return {}

        # Build item_id -> item mapping for supplier resolution
        item_map: Dict[str, ProcurementItem] = {
            item.item_id: item for item in items
        }

        supplier_data: Dict[str, Dict[str, Any]] = {}
        total_emissions = ZERO

        for r in results:
            item = item_map.get(r.item_id)
            supplier_id = (
                item.supplier_id if item and item.supplier_id else "UNKNOWN"
            )
            supplier_name = (
                item.supplier_name
                if item and item.supplier_name
                else "Unknown Supplier"
            )

            if supplier_id not in supplier_data:
                supplier_data[supplier_id] = {
                    "supplier_name": supplier_name,
                    "emissions_kgco2e": ZERO,
                    "emissions_tco2e": ZERO,
                    "spend_usd": ZERO,
                    "item_count": ZERO,
                    "emissions_pct": ZERO,
                }

            data = supplier_data[supplier_id]
            data["emissions_kgco2e"] += r.emissions_kgco2e
            data["emissions_tco2e"] += r.emissions_tco2e
            data["spend_usd"] += r.spend_usd
            data["item_count"] += ONE
            total_emissions += r.emissions_kgco2e

        # Compute percentages and quantize
        for supplier_id, data in supplier_data.items():
            if total_emissions > ZERO:
                data["emissions_pct"] = _quantize(
                    (data["emissions_kgco2e"] / total_emissions)
                    * ONE_HUNDRED
                )
            data["emissions_kgco2e"] = _quantize(
                data["emissions_kgco2e"]
            )
            data["emissions_tco2e"] = _quantize(
                data["emissions_tco2e"]
            )
            data["spend_usd"] = _quantize(data["spend_usd"])

        logger.info(
            "Supplier aggregation completed: %d suppliers from %d results",
            len(supplier_data),
            len(results),
        )

        return supplier_data

    # ------------------------------------------------------------------
    # Public API: Coverage Analysis
    # ------------------------------------------------------------------

    def compute_coverage(
        self,
        results: List[SpendBasedResult],
        total_spend_usd: Decimal,
    ) -> Dict[str, Decimal]:
        """Compute spend coverage metrics for a batch of results.

        Calculates what percentage of total procurement spend has
        been covered by the spend-based method, along with emission
        intensity metrics. Used for completeness reporting and
        credibility assessment per GHG Protocol.

        Args:
            results: List of SpendBasedResult from batch calculation.
            total_spend_usd: Total procurement spend in USD across
                all categories (including those not calculated).
                Must be positive.

        Returns:
            Dictionary with coverage metrics:
            - ``covered_spend_usd``: Total USD spend covered by results
            - ``total_spend_usd``: Input total spend
            - ``coverage_pct``: Percentage of total spend covered
            - ``total_emissions_kgco2e``: Total emissions from results
            - ``total_emissions_tco2e``: Total emissions in tCO2e
            - ``emission_intensity_kgco2e_per_usd``: Average intensity
            - ``item_count``: Number of items in results
            - ``avg_emissions_per_item_kgco2e``: Average per-item

        Raises:
            ValueError: If total_spend_usd is zero or negative.

        Example:
            >>> coverage = engine.compute_coverage(results, Decimal("10000000"))
            >>> print(f"Coverage: {coverage['coverage_pct']}%")
        """
        if total_spend_usd <= ZERO:
            raise ValueError(
                f"Total spend must be positive: {total_spend_usd}"
            )

        covered_spend = ZERO
        total_emissions_kg = ZERO
        item_count = ZERO

        for r in results:
            covered_spend += r.spend_usd
            total_emissions_kg += r.emissions_kgco2e
            item_count += ONE

        total_emissions_t = _quantize(total_emissions_kg / ONE_THOUSAND)

        coverage_pct = _quantize(
            (covered_spend / total_spend_usd) * ONE_HUNDRED
        )

        # Cap at 100%
        if coverage_pct > ONE_HUNDRED:
            coverage_pct = ONE_HUNDRED

        intensity = ZERO
        if covered_spend > ZERO:
            intensity = _quantize(total_emissions_kg / covered_spend)

        avg_per_item = ZERO
        if item_count > ZERO:
            avg_per_item = _quantize(total_emissions_kg / item_count)

        coverage = {
            "covered_spend_usd": _quantize(covered_spend),
            "total_spend_usd": _quantize(total_spend_usd),
            "coverage_pct": coverage_pct,
            "total_emissions_kgco2e": _quantize(total_emissions_kg),
            "total_emissions_tco2e": total_emissions_t,
            "emission_intensity_kgco2e_per_usd": intensity,
            "item_count": item_count,
            "avg_emissions_per_item_kgco2e": avg_per_item,
        }

        logger.info(
            "Coverage analysis: covered=%s/%s USD (%s%%), "
            "emissions=%s tCO2e, items=%s",
            coverage["covered_spend_usd"],
            coverage["total_spend_usd"],
            coverage["coverage_pct"],
            coverage["total_emissions_tco2e"],
            coverage["item_count"],
        )

        return coverage

    # ------------------------------------------------------------------
    # Public API: Uncertainty Estimation
    # ------------------------------------------------------------------

    def estimate_uncertainty(
        self,
        result: SpendBasedResult,
    ) -> Dict[str, Decimal]:
        """Estimate uncertainty range for a spend-based calculation result.

        Uses the GHG Protocol uncertainty guidance for the spend-based
        method, which has a typical uncertainty of +/- 50-100%. The
        actual range depends on data quality (DQI scores) and the
        specificity of the EEIO factor.

        The method uses a pedigree matrix approach to refine the
        base uncertainty range using the composite DQI score.

        Args:
            result: SpendBasedResult to estimate uncertainty for.

        Returns:
            Dictionary with uncertainty metrics:
            - ``base_uncertainty_min_pct``: Base minimum uncertainty (50%)
            - ``base_uncertainty_max_pct``: Base maximum uncertainty (100%)
            - ``adjusted_uncertainty_pct``: Adjusted uncertainty based
              on DQI and pedigree factor
            - ``lower_bound_kgco2e``: Lower bound emission estimate
            - ``upper_bound_kgco2e``: Upper bound emission estimate
            - ``lower_bound_tco2e``: Lower bound in tCO2e
            - ``upper_bound_tco2e``: Upper bound in tCO2e
            - ``confidence_level_pct``: Confidence level (95%)
            - ``method``: Uncertainty method used

        Example:
            >>> unc = engine.estimate_uncertainty(result)
            >>> print(f"Range: {unc['lower_bound_tco2e']} - "
            ...       f"{unc['upper_bound_tco2e']} tCO2e")
        """
        base_min, base_max = UNCERTAINTY_RANGES[
            CalculationMethod.SPEND_BASED
        ]

        # Determine adjustment factor from NAICS code specificity
        code_len = len(result.eeio_sector_code) if result.eeio_sector_code else 0
        specificity_factor = ONE
        if code_len >= 6:
            specificity_factor = Decimal("0.85")
        elif code_len >= 4:
            specificity_factor = Decimal("0.95")
        else:
            specificity_factor = Decimal("1.10")

        # Adjust using composite DQI if available
        dqi_factor = ONE
        if result.dqi_scores:
            composite = result.dqi_scores.get("composite", Decimal("4.0"))
            dqi_factor = _pedigree_factor_for_score(composite)

        # Calculate adjusted uncertainty
        adjusted_uncertainty = _quantize(
            ((base_min + base_max) / Decimal("2"))
            * specificity_factor
            * dqi_factor
        )

        # Calculate bounds
        uncertainty_fraction = adjusted_uncertainty / ONE_HUNDRED
        lower_bound_kg = _quantize(
            result.emissions_kgco2e * (ONE - uncertainty_fraction)
        )
        upper_bound_kg = _quantize(
            result.emissions_kgco2e * (ONE + uncertainty_fraction)
        )

        # Ensure lower bound is non-negative
        if lower_bound_kg < ZERO:
            lower_bound_kg = ZERO

        lower_bound_t = _quantize(lower_bound_kg / ONE_THOUSAND)
        upper_bound_t = _quantize(upper_bound_kg / ONE_THOUSAND)

        return {
            "base_uncertainty_min_pct": base_min,
            "base_uncertainty_max_pct": base_max,
            "adjusted_uncertainty_pct": adjusted_uncertainty,
            "specificity_factor": specificity_factor,
            "dqi_factor": dqi_factor,
            "lower_bound_kgco2e": lower_bound_kg,
            "upper_bound_kgco2e": upper_bound_kg,
            "lower_bound_tco2e": lower_bound_t,
            "upper_bound_tco2e": upper_bound_t,
            "confidence_level_pct": Decimal("95.0"),
            "method": "pedigree_matrix",
        }

    # ------------------------------------------------------------------
    # Public API: Build Spend Record
    # ------------------------------------------------------------------

    def build_spend_record(
        self,
        item: ProcurementItem,
        database: EEIODatabase = EEIODatabase.EPA_USEEIO,
        cpi_ratio: Decimal = ONE,
        custom_fx_rate: Optional[Decimal] = None,
    ) -> SpendRecord:
        """Build a SpendRecord with all intermediate spend-pipeline values.

        Executes currency conversion, inflation deflation, and margin
        removal without performing the final emission calculation.
        Useful for inspection, debugging, and as input to manual or
        custom emission calculations.

        Args:
            item: Procurement item with spend amount and currency.
            database: EEIO database identifier.
            cpi_ratio: CPI ratio for inflation deflation.
            custom_fx_rate: Optional custom exchange rate.

        Returns:
            SpendRecord with all intermediate values populated.

        Raises:
            ValueError: If currency is not supported or inputs are
                invalid.

        Example:
            >>> record = engine.build_spend_record(item)
            >>> print(f"Producer spend: ${record.spend_producer_usd}")
        """
        # Currency conversion
        spend_usd, fx_rate = self.convert_to_usd(
            item.spend_amount, item.currency, custom_fx_rate
        )

        # Inflation deflation
        if self._config.enable_inflation_adjustment:
            spend_deflated = self.deflate_to_base_year(spend_usd, cpi_ratio)
        else:
            spend_deflated = spend_usd

        # NAICS resolution
        naics_code = self.resolve_naics_code(item)

        # Margin removal
        margin_rate = ZERO
        if self._config.enable_margin_removal and naics_code:
            spend_producer, margin_rate = self.remove_margin(
                spend_deflated, naics_code
            )
        else:
            spend_producer = spend_deflated

        return SpendRecord(
            item=item,
            spend_usd=spend_usd,
            spend_deflated_usd=spend_deflated,
            spend_producer_usd=spend_producer,
            eeio_database=database,
            eeio_sector_code=naics_code,
            margin_rate=margin_rate,
            cpi_ratio=cpi_ratio,
            fx_rate=fx_rate,
        )

    # ------------------------------------------------------------------
    # Public API: Batch DQI Scoring
    # ------------------------------------------------------------------

    def score_dqi_batch(
        self,
        items: List[ProcurementItem],
        results: List[SpendBasedResult],
        database: EEIODatabase = EEIODatabase.EPA_USEEIO,
    ) -> List[DQIAssessment]:
        """Score data quality for a batch of spend-based results.

        Applies DQI scoring to each result in the batch using
        ``score_dqi_spend_based``. Items and results must be aligned
        by index (items[i] corresponds to results[i]).

        Args:
            items: List of procurement items.
            results: List of SpendBasedResult (same order as items).
            database: EEIO database used.

        Returns:
            List of DQIAssessment, one per result.

        Example:
            >>> dqi_list = engine.score_dqi_batch(items, results)
            >>> avg_score = sum(d.composite_score for d in dqi_list) / len(dqi_list)
        """
        assessments: List[DQIAssessment] = []

        # Build item_id -> item mapping
        item_map: Dict[str, ProcurementItem] = {
            item.item_id: item for item in items
        }

        for result in results:
            item = item_map.get(result.item_id)
            if item is None:
                logger.warning(
                    "No matching item for result item_id=%s, "
                    "using default DQI scores",
                    result.item_id,
                )
                # Create a minimal item for scoring
                item = ProcurementItem(
                    item_id=result.item_id,
                    description="Unknown item",
                    spend_amount=result.spend_original,
                    currency=result.currency,
                )

            assessment = self.score_dqi_spend_based(
                item=item,
                database=database,
                result=result,
            )
            assessments.append(assessment)

        logger.info(
            "Batch DQI scoring completed: %d assessments",
            len(assessments),
        )

        return assessments

    # ------------------------------------------------------------------
    # Public API: Emission Intensity Analysis
    # ------------------------------------------------------------------

    def compute_emission_intensities(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Compute emission intensity metrics from spend-based results.

        Calculates per-sector and overall emission intensities
        (kgCO2e per USD) for benchmarking and trend analysis.

        Args:
            results: List of SpendBasedResult.

        Returns:
            Dictionary with keys:
            - ``overall``: Overall intensity metrics
            - ``by_sector``: Per-sector intensity metrics

        Example:
            >>> intensities = engine.compute_emission_intensities(results)
            >>> print(f"Overall: {intensities['overall']['kgco2e_per_usd']}")
        """
        if not results:
            return {
                "overall": {
                    "kgco2e_per_usd": ZERO,
                    "tco2e_per_million_usd": ZERO,
                    "total_emissions_kgco2e": ZERO,
                    "total_spend_usd": ZERO,
                },
                "by_sector": {},
            }

        total_emissions_kg = ZERO
        total_spend = ZERO
        sector_emissions: Dict[str, Decimal] = {}
        sector_spend: Dict[str, Decimal] = {}

        for r in results:
            total_emissions_kg += r.emissions_kgco2e
            total_spend += r.spend_usd
            sector_2 = _naics_to_2digit(r.eeio_sector_code)
            if not sector_2:
                sector_2 = "XX"
            sector_emissions[sector_2] = (
                sector_emissions.get(sector_2, ZERO) + r.emissions_kgco2e
            )
            sector_spend[sector_2] = (
                sector_spend.get(sector_2, ZERO) + r.spend_usd
            )

        # Overall intensity
        overall_intensity = ZERO
        overall_tco2e_per_million = ZERO
        if total_spend > ZERO:
            overall_intensity = _quantize(total_emissions_kg / total_spend)
            million = Decimal("1000000")
            overall_tco2e_per_million = _quantize(
                (total_emissions_kg / ONE_THOUSAND) / (total_spend / million)
            )

        # Per-sector intensities
        by_sector: Dict[str, Dict[str, Decimal]] = {}
        for sector_2 in sector_emissions:
            s_emissions = sector_emissions[sector_2]
            s_spend = sector_spend[sector_2]
            s_intensity = ZERO
            if s_spend > ZERO:
                s_intensity = _quantize(s_emissions / s_spend)
            by_sector[sector_2] = {
                "sector_name": _NAICS_SECTOR_NAMES.get(
                    sector_2, "Unknown"
                ),
                "kgco2e_per_usd": s_intensity,
                "total_emissions_kgco2e": _quantize(s_emissions),
                "total_spend_usd": _quantize(s_spend),
            }

        return {
            "overall": {
                "kgco2e_per_usd": overall_intensity,
                "tco2e_per_million_usd": overall_tco2e_per_million,
                "total_emissions_kgco2e": _quantize(total_emissions_kg),
                "total_spend_usd": _quantize(total_spend),
            },
            "by_sector": by_sector,
        }

    # ------------------------------------------------------------------
    # Public API: Top Emitters Analysis
    # ------------------------------------------------------------------

    def identify_top_emitters(
        self,
        results: List[SpendBasedResult],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Identify the top N emission-contributing items.

        Sorts results by emissions_kgco2e descending and returns
        the top N items with their emission share and cumulative
        percentage (Pareto analysis).

        Args:
            results: List of SpendBasedResult.
            top_n: Number of top items to return. Defaults to 10.

        Returns:
            List of dictionaries with keys:
            - ``rank``: 1-based rank
            - ``item_id``: Procurement item identifier
            - ``eeio_sector_code``: NAICS sector code
            - ``sector_name``: Human-readable sector name
            - ``emissions_kgco2e``: Item emissions
            - ``emissions_tco2e``: Item emissions in tCO2e
            - ``spend_usd``: Item spend
            - ``emissions_pct``: Share of total emissions
            - ``cumulative_pct``: Cumulative share

        Example:
            >>> top = engine.identify_top_emitters(results, top_n=5)
            >>> for entry in top:
            ...     print(f"#{entry['rank']} {entry['item_id']}: "
            ...           f"{entry['emissions_pct']}%")
        """
        if not results:
            return []

        # Sort by emissions descending
        sorted_results = sorted(
            results,
            key=lambda r: r.emissions_kgco2e,
            reverse=True,
        )

        total_emissions = sum(
            r.emissions_kgco2e for r in sorted_results
        )

        top_items: List[Dict[str, Any]] = []
        cumulative_pct = ZERO

        for i, r in enumerate(sorted_results[:top_n]):
            emissions_pct = ZERO
            if total_emissions > ZERO:
                emissions_pct = _quantize(
                    (r.emissions_kgco2e / total_emissions) * ONE_HUNDRED
                )
            cumulative_pct += emissions_pct

            top_items.append({
                "rank": i + 1,
                "item_id": r.item_id,
                "eeio_sector_code": r.eeio_sector_code,
                "sector_name": _sector_name_for_code(r.eeio_sector_code),
                "emissions_kgco2e": r.emissions_kgco2e,
                "emissions_tco2e": r.emissions_tco2e,
                "spend_usd": r.spend_usd,
                "emissions_pct": emissions_pct,
                "cumulative_pct": _quantize(cumulative_pct),
            })

        return top_items

    # ------------------------------------------------------------------
    # Public API: Summary Statistics
    # ------------------------------------------------------------------

    def compute_summary(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[str, Any]:
        """Compute summary statistics for a batch of spend-based results.

        Provides aggregate metrics including total emissions, total
        spend, average intensity, sector count, and distribution
        statistics for reporting and dashboard display.

        Args:
            results: List of SpendBasedResult.

        Returns:
            Dictionary with summary metrics:
            - ``total_emissions_kgco2e``: Sum of all emissions
            - ``total_emissions_tco2e``: Sum in tCO2e
            - ``total_spend_usd``: Sum of all USD spend
            - ``total_spend_producer_usd``: Sum of producer spend
            - ``item_count``: Number of results
            - ``sector_count``: Number of unique NAICS 2-digit sectors
            - ``avg_intensity_kgco2e_per_usd``: Average intensity
            - ``max_emission_kgco2e``: Maximum single-item emission
            - ``min_emission_kgco2e``: Minimum single-item emission
            - ``median_emission_kgco2e``: Median emission value
            - ``currency_count``: Number of unique currencies
            - ``database_used``: EEIO database(s) used

        Example:
            >>> summary = engine.compute_summary(results)
            >>> print(f"Total: {summary['total_emissions_tco2e']} tCO2e")
        """
        if not results:
            return {
                "total_emissions_kgco2e": ZERO,
                "total_emissions_tco2e": ZERO,
                "total_spend_usd": ZERO,
                "total_spend_producer_usd": ZERO,
                "item_count": 0,
                "sector_count": 0,
                "avg_intensity_kgco2e_per_usd": ZERO,
                "max_emission_kgco2e": ZERO,
                "min_emission_kgco2e": ZERO,
                "median_emission_kgco2e": ZERO,
                "currency_count": 0,
                "databases_used": [],
            }

        emissions_list = [r.emissions_kgco2e for r in results]
        total_emissions_kg = sum(emissions_list)
        total_spend = sum(r.spend_usd for r in results)
        total_spend_producer = sum(r.spend_producer_usd for r in results)

        sectors = set()
        currencies = set()
        databases = set()
        for r in results:
            sectors.add(_naics_to_2digit(r.eeio_sector_code))
            currencies.add(r.currency.value)
            databases.add(r.eeio_database.value)

        # Sort for median calculation
        sorted_emissions = sorted(emissions_list)
        n = len(sorted_emissions)
        if n % 2 == 0:
            median_val = (sorted_emissions[n // 2 - 1] + sorted_emissions[n // 2]) / Decimal("2")
        else:
            median_val = sorted_emissions[n // 2]

        avg_intensity = ZERO
        if total_spend > ZERO:
            avg_intensity = _quantize(total_emissions_kg / total_spend)

        return {
            "total_emissions_kgco2e": _quantize(total_emissions_kg),
            "total_emissions_tco2e": _quantize(
                total_emissions_kg / ONE_THOUSAND
            ),
            "total_spend_usd": _quantize(total_spend),
            "total_spend_producer_usd": _quantize(total_spend_producer),
            "item_count": len(results),
            "sector_count": len(sectors),
            "avg_intensity_kgco2e_per_usd": avg_intensity,
            "max_emission_kgco2e": _quantize(max(emissions_list)),
            "min_emission_kgco2e": _quantize(min(emissions_list)),
            "median_emission_kgco2e": _quantize(median_val),
            "currency_count": len(currencies),
            "databases_used": sorted(databases),
        }

    # ------------------------------------------------------------------
    # Public API: Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return engine health status and operational metrics.

        Provides a snapshot of the engine's internal state for
        monitoring and alerting. Used by the health check endpoint
        and Prometheus metrics scraping.

        Returns:
            Dictionary with health metrics:
            - ``status``: Engine status ("healthy" or "degraded")
            - ``agent_id``: Agent identifier
            - ``version``: Agent version
            - ``engine``: Engine name
            - ``singleton_initialized``: Whether singleton is init'd
            - ``calculation_count``: Total calculations performed
            - ``batch_count``: Total batches processed
            - ``error_count``: Total errors encountered
            - ``total_emissions_kgco2e``: Cumulative emissions
            - ``total_spend_processed_usd``: Cumulative spend
            - ``last_calculation_time``: ISO timestamp of last calc
            - ``eeio_factor_count``: Number of built-in EEIO factors
            - ``currency_count``: Number of supported currencies
            - ``margin_sector_count``: Number of margin table entries
            - ``config_margin_removal``: Margin removal enabled
            - ``config_inflation_adjustment``: Inflation adj enabled
            - ``config_default_eeio_database``: Default EEIO database

        Example:
            >>> health = engine.health_check()
            >>> assert health["status"] == "healthy"
        """
        with self._lock:
            calc_count = self._calculation_count
            batch_count = self._batch_count
            error_count = self._error_count
            total_emissions = self._total_emissions_kgco2e
            total_spend = self._total_spend_processed_usd
            last_calc = self._last_calculation_time

        status = "healthy"
        if error_count > 0 and calc_count > 0:
            error_rate = error_count / (calc_count + error_count)
            if error_rate > 0.1:
                status = "degraded"

        return {
            "status": status,
            "agent_id": AGENT_ID,
            "version": VERSION,
            "engine": "SpendBasedCalculatorEngine",
            "singleton_initialized": self.__class__._initialized,
            "calculation_count": calc_count,
            "batch_count": batch_count,
            "error_count": error_count,
            "total_emissions_kgco2e": str(total_emissions),
            "total_spend_processed_usd": str(total_spend),
            "last_calculation_time": (
                last_calc.isoformat() if last_calc else None
            ),
            "eeio_factor_count": len(EEIO_EMISSION_FACTORS),
            "currency_count": len(CURRENCY_EXCHANGE_RATES),
            "margin_sector_count": len(INDUSTRY_MARGIN_PERCENTAGES),
            "config_margin_removal": self._config.enable_margin_removal,
            "config_inflation_adjustment": (
                self._config.enable_inflation_adjustment
            ),
            "config_default_eeio_database": (
                self._config.default_eeio_database
            ),
        }

    # ------------------------------------------------------------------
    # Public API: Factor Inspection
    # ------------------------------------------------------------------

    def get_eeio_factor_info(
        self,
        naics_code: str,
        database: EEIODatabase = EEIODatabase.EPA_USEEIO,
    ) -> Optional[EEIOFactor]:
        """Get detailed EEIO factor information for a NAICS code.

        Returns a structured EEIOFactor object with the emission
        factor value and metadata. Uses the same progressive prefix
        matching as ``resolve_eeio_factor``.

        Args:
            naics_code: NAICS sector code (2-6 digits).
            database: EEIO database to query.

        Returns:
            EEIOFactor with factor value and metadata, or None if
            no factor is found.

        Example:
            >>> info = engine.get_eeio_factor_info("331110")
            >>> if info:
            ...     print(f"{info.sector_name}: {info.factor_kgco2e_per_unit}")
        """
        factor_value = self.resolve_eeio_factor(naics_code, database)
        if factor_value is None:
            return None

        # Determine which code actually matched
        matched_code = self._find_matching_code(naics_code)
        if matched_code is None:
            matched_code = naics_code

        sector_2 = _naics_to_2digit(matched_code)
        sector_name = _sector_name_for_code(matched_code)

        return EEIOFactor(
            sector_code=matched_code,
            sector_name=f"{sector_name} (NAICS {matched_code})",
            factor_kgco2e_per_unit=factor_value,
            database=database,
            database_version="v1.2" if database == EEIODatabase.EPA_USEEIO else "latest",
            base_year=self._config.eeio_base_year,
            base_currency=CurrencyCode.USD,
            region="US" if database == EEIODatabase.EPA_USEEIO else "GLOBAL",
            margin_type="purchaser",
            classification_system=SpendClassificationSystem.NAICS,
        )

    # ------------------------------------------------------------------
    # Public API: Currency Info
    # ------------------------------------------------------------------

    def get_supported_currencies(self) -> Dict[str, Decimal]:
        """Return the supported currencies and their USD exchange rates.

        Returns:
            Dictionary mapping currency code string to exchange rate
            (units of foreign currency per 1 USD).

        Example:
            >>> currencies = engine.get_supported_currencies()
            >>> print(f"EUR rate: {currencies['EUR']}")
        """
        return {
            code.value: rate
            for code, rate in CURRENCY_EXCHANGE_RATES.items()
        }

    # ------------------------------------------------------------------
    # Public API: Margin Info
    # ------------------------------------------------------------------

    def get_margin_rate(self, naics_code: str) -> Decimal:
        """Get the margin rate for a NAICS sector.

        Returns the margin percentage used to convert purchaser price
        to producer (basic) price for the given NAICS code.

        Args:
            naics_code: Full NAICS code (at least 2 digits).

        Returns:
            Margin rate as a percentage Decimal. Returns the default
            rate (20%) if the sector is not in the margin table.

        Example:
            >>> margin = engine.get_margin_rate("331110")
            >>> print(f"Margin: {margin}%")
        """
        if not naics_code or len(naics_code) < 2:
            return _DEFAULT_MARGIN_RATE

        sector_2 = _naics_to_2digit(naics_code)
        return INDUSTRY_MARGIN_PERCENTAGES.get(
            sector_2, _DEFAULT_MARGIN_RATE
        )

    # ==================================================================
    # Private Methods
    # ==================================================================

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    def _validate_single_input(
        self,
        item: ProcurementItem,
        cpi_ratio: Decimal,
        custom_fx_rate: Optional[Decimal],
    ) -> None:
        """Validate inputs for a single spend-based calculation.

        Args:
            item: Procurement item to validate.
            cpi_ratio: CPI ratio value.
            custom_fx_rate: Optional custom FX rate.

        Raises:
            TypeError: If item is not a ProcurementItem.
            ValueError: If item has invalid spend or boundary flags.
        """
        if not isinstance(item, ProcurementItem):
            raise TypeError(
                f"Expected ProcurementItem, got {type(item).__name__}"
            )

        if item.spend_amount <= ZERO:
            raise ValueError(
                f"Spend amount must be positive for spend-based "
                f"calculation: {item.spend_amount} (item={item.item_id})"
            )

        if item.is_credit_return:
            raise ValueError(
                f"Credit/return items cannot be calculated via "
                f"spend-based method (item={item.item_id})"
            )

        if item.is_capital_good:
            logger.warning(
                "Item %s is flagged as capital good (Category 2). "
                "Including in Category 1 spend-based calculation. "
                "Consider using Category 2 for capital goods with "
                "spend > $%s.",
                item.item_id,
                self._config.capital_threshold,
            )

        if item.is_fuel_energy:
            logger.warning(
                "Item %s is flagged as fuel/energy for own use "
                "(Category 3). Including in Category 1 spend-based "
                "calculation. Consider excluding if this is for "
                "own combustion.",
                item.item_id,
            )

        if item.is_transport:
            logger.warning(
                "Item %s is flagged as upstream transport "
                "(Category 4). Including in Category 1 spend-based "
                "calculation. Consider using Category 4 for transport "
                "services.",
                item.item_id,
            )

        if item.is_intercompany:
            logger.warning(
                "Item %s is flagged as intercompany. Including in "
                "Category 1 spend-based calculation. Consider "
                "excluding intercompany transactions to avoid "
                "double-counting.",
                item.item_id,
            )

        if cpi_ratio <= ZERO:
            raise ValueError(
                f"CPI ratio must be positive: {cpi_ratio}"
            )

        if custom_fx_rate is not None and custom_fx_rate <= ZERO:
            raise ValueError(
                f"Custom FX rate must be positive: {custom_fx_rate}"
            )

    # ------------------------------------------------------------------
    # USEEIO Factor Lookup (Progressive Prefix Matching)
    # ------------------------------------------------------------------

    def _lookup_useeio_factor(
        self,
        naics_code: str,
    ) -> Optional[Decimal]:
        """Look up EEIO factor with progressive prefix matching.

        Tries the full code first, then progressively shorter prefixes
        (5-digit, 4-digit, 3-digit, 2-digit) to find the best
        available match in the EEIO_EMISSION_FACTORS table.

        Args:
            naics_code: NAICS sector code.

        Returns:
            Emission factor Decimal or None if not found.
        """
        code = naics_code.strip()

        # Exact match first
        if code in EEIO_EMISSION_FACTORS:
            logger.debug(
                "EEIO factor exact match: NAICS=%s, factor=%s",
                code,
                EEIO_EMISSION_FACTORS[code],
            )
            return EEIO_EMISSION_FACTORS[code]

        # Progressive prefix matching (shorter codes)
        for prefix_len in range(len(code) - 1, 1, -1):
            prefix = code[:prefix_len]
            if prefix in EEIO_EMISSION_FACTORS:
                logger.debug(
                    "EEIO factor prefix match: NAICS=%s -> %s, factor=%s",
                    code,
                    prefix,
                    EEIO_EMISSION_FACTORS[prefix],
                )
                return EEIO_EMISSION_FACTORS[prefix]

        # Try matching any code that starts with the same prefix
        # (find the first factor whose key starts with the same
        # 2-digit sector)
        sector_2 = _naics_to_2digit(code)
        if sector_2:
            for key, value in EEIO_EMISSION_FACTORS.items():
                if key.startswith(sector_2):
                    logger.debug(
                        "EEIO factor sector fallback: NAICS=%s -> %s, "
                        "factor=%s",
                        code,
                        key,
                        value,
                    )
                    return value

        logger.warning(
            "No EEIO factor found for NAICS=%s (tried exact, "
            "prefix, and sector fallback)",
            code,
        )
        return None

    # ------------------------------------------------------------------
    # Find Matching Code (for factor info)
    # ------------------------------------------------------------------

    def _find_matching_code(self, naics_code: str) -> Optional[str]:
        """Find the actual NAICS code that matches in the factor table.

        Args:
            naics_code: Input NAICS code.

        Returns:
            Matched code string, or None if no match.
        """
        code = naics_code.strip()

        if code in EEIO_EMISSION_FACTORS:
            return code

        for prefix_len in range(len(code) - 1, 1, -1):
            prefix = code[:prefix_len]
            if prefix in EEIO_EMISSION_FACTORS:
                return prefix

        sector_2 = _naics_to_2digit(code)
        if sector_2:
            for key in EEIO_EMISSION_FACTORS:
                if key.startswith(sector_2):
                    return key

        return None

    # ------------------------------------------------------------------
    # Material Category -> NAICS Fallback Mapping
    # ------------------------------------------------------------------

    def _material_category_to_naics(
        self,
        category: Optional[str],
    ) -> Optional[str]:
        """Map a material category to a representative NAICS code.

        Used as a last-resort fallback when no explicit classification
        code is available on the procurement item. Maps the 20
        MaterialCategory values to representative NAICS-6 codes.

        Args:
            category: MaterialCategory enum value or string.

        Returns:
            Representative NAICS-6 code, or None if category is None
            or not recognized.
        """
        if category is None:
            return None

        # Convert enum to string if needed
        cat_str = category.value if hasattr(category, "value") else str(category)

        mapping: Dict[str, str] = {
            "raw_metals": "331110",        # Iron and Steel Mills
            "plastics": "325211",          # Plastics Material and Resin Mfg
            "chemicals": "325110",         # Petrochemical Manufacturing
            "paper": "322121",             # Paper Mills
            "textiles": "325211",          # Textiles (proxy via chemicals)
            "electronics": "334111",       # Electronic Computer Mfg
            "food": "311210",              # Flour Milling (general food)
            "packaging": "322211",         # Corrugated Container Mfg
            "construction": "236220",      # Commercial Building Construction
            "machinery": "333111",         # Farm Machinery Manufacturing
            "fuels": "211120",             # Crude Petroleum Extraction
            "minerals": "212310",          # Stone Mining and Quarrying
            "glass": "327211",             # Flat Glass Mfg (proxy)
            "rubber": "325211",            # Rubber (proxy via plastics)
            "wood": "113110",              # Timber Tract Operations
            "agriculture": "111110",       # Soybean Farming (general ag)
            "services_it": "541511",       # Custom Computer Programming
            "services_professional": "541611",  # Management Consulting
            "services_financial": "522110",     # Commercial Banking
            "other": "541611",             # Management Consulting (proxy)
        }

        naics = mapping.get(cat_str)
        if naics:
            logger.debug(
                "Material category fallback: %s -> NAICS %s",
                cat_str,
                naics,
            )
        return naics

    # ------------------------------------------------------------------
    # DQI Findings Generator
    # ------------------------------------------------------------------

    def _generate_dqi_findings(
        self,
        temporal_score: Decimal,
        geo_score: Decimal,
        technological_score: Decimal,
        completeness_score: Decimal,
        reliability_score: Decimal,
        database: EEIODatabase,
        item: ProcurementItem,
        result: SpendBasedResult,
    ) -> List[str]:
        """Generate DQI findings and recommendations.

        Produces human-readable findings based on individual
        dimension scores to guide data improvement efforts.

        Args:
            temporal_score: Temporal dimension score.
            geo_score: Geographical dimension score.
            technological_score: Technological dimension score.
            completeness_score: Completeness dimension score.
            reliability_score: Reliability dimension score.
            database: EEIO database used.
            item: Source procurement item.
            result: Spend-based calculation result.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        findings.append(
            f"Spend-based method (EEIO) used with "
            f"{_EEIO_DATABASE_LABELS.get(database, database.value)}."
        )

        if temporal_score >= Decimal("4.0"):
            findings.append(
                "Temporal: EEIO factors are based on multi-year "
                "economic models; consider checking if more recent "
                "factors are available."
            )

        if geo_score >= Decimal("4.0"):
            findings.append(
                "Geographical: Non-domestic currency detected "
                f"({item.currency.value}); EEIO factors may not "
                "reflect the supplier's actual production region. "
                "Consider using regional EEIO databases (e.g., "
                "EXIOBASE for EU suppliers)."
            )
        elif geo_score >= Decimal("3.0"):
            findings.append(
                "Geographical: Domestic US EEIO factors applied; "
                "representativeness is moderate for US-based "
                "suppliers."
            )

        if technological_score >= Decimal("4.0"):
            findings.append(
                "Technological: EEIO factors represent sector-average "
                "technology. Product-specific emission factors (EPD, "
                "ecoinvent) would improve accuracy. Consider the "
                "average-data or supplier-specific method."
            )

        if completeness_score >= Decimal("4.0"):
            findings.append(
                "Completeness: Sector-level factors may not capture "
                "all emission sources for the specific product. "
                "A more granular NAICS code (6-digit) would improve "
                "completeness."
            )

        if reliability_score >= Decimal("4.0"):
            findings.append(
                "Reliability: EEIO factors are model-derived, not "
                "based on direct measurement. Third-party verified "
                "supplier data (EPD, CDP) would improve reliability."
            )

        # Recommend method upgrade if spend is significant
        if result.spend_usd > Decimal("100000"):
            findings.append(
                f"High-spend item (${result.spend_usd} USD). "
                "Consider upgrading to average-data or "
                "supplier-specific method for better accuracy."
            )

        return findings

    # ------------------------------------------------------------------
    # Provenance Recording
    # ------------------------------------------------------------------

    def _record_provenance_stage(
        self,
        calculation_id: str,
        item_id: str,
        stage: ProvenanceStage,
        metadata: Dict[str, Any],
        output_data: Any,
    ) -> None:
        """Record a provenance stage for audit trail.

        Safely attempts to record provenance. Logs warnings on
        failure but does not raise exceptions, as provenance tracking
        should not block the calculation pipeline.

        Args:
            calculation_id: Unique calculation identifier.
            item_id: Procurement item identifier.
            stage: Provenance stage to record.
            metadata: Stage metadata.
            output_data: Output data to hash.
        """
        try:
            if not self._config.enable_provenance:
                return

            chain_id = f"spend-{calculation_id}-{item_id}"

            # Create chain if it does not exist
            try:
                self._provenance_tracker.create_chain(
                    calculation_id=chain_id,
                    organization_id=self._config.default_tenant,
                    reporting_period=str(utcnow().year),
                )
            except ValueError:
                # Chain already exists (ok for multi-stage)
                pass

            self._provenance_tracker.add_stage(
                chain_id=chain_id,
                stage=stage,
                metadata=metadata,
                output_data=output_data,
            )

        except Exception as exc:
            logger.warning(
                "Failed to record provenance stage %s for item %s: %s",
                stage.value,
                item_id,
                str(exc),
            )

    # ------------------------------------------------------------------
    # Metrics Recording
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        method: str,
        status: str,
        duration_s: float,
        emissions_kgco2e: float = 0.0,
        spend_usd: float = 0.0,
    ) -> None:
        """Record calculation metrics to Prometheus.

        Safely records metrics. Logs warnings on failure but does
        not raise exceptions.

        Args:
            method: Calculation method name.
            status: Calculation status.
            duration_s: Duration in seconds.
            emissions_kgco2e: Emissions in kgCO2e.
            spend_usd: Spend in USD.
        """
        try:
            self._metrics.record_calculation(
                tenant_id=self._config.default_tenant,
                method=method,
                status=status,
                duration_s=duration_s,
                emissions_kgco2e=emissions_kgco2e,
                spend_usd=spend_usd,
            )
        except Exception as exc:
            logger.warning(
                "Failed to record metrics: %s",
                str(exc),
            )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return string representation of the engine.

        Returns:
            String with engine name, agent ID, and version.
        """
        return (
            f"SpendBasedCalculatorEngine("
            f"agent={AGENT_ID}, "
            f"version={VERSION}, "
            f"initialized={self.__class__._initialized})"
        )

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns:
            String describing the engine.
        """
        return (
            f"SpendBasedCalculatorEngine [{AGENT_ID} v{VERSION}] "
            f"(calcs={self._calculation_count}, "
            f"errors={self._error_count})"
        )
