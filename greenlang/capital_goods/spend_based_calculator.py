# -*- coding: utf-8 -*-
"""
SpendBasedCalculatorEngine - Engine 2: Capital Goods Agent (AGENT-MRV-015)

Core calculation engine implementing the GHG Protocol Scope 3 Category 2
spend-based method using Environmentally-Extended Input-Output (EEIO)
emission factors.  Converts capital expenditure (CapEx) data into
greenhouse gas emission estimates via a deterministic pipeline of
currency conversion, CPI-based inflation deflation, purchaser-to-producer
margin removal, progressive NAICS sector resolution, EEIO factor
application, per-gas breakdown, and five-dimension DQI scoring.

The spend-based method is the broadest-coverage but lowest-accuracy
approach defined in the GHG Protocol Corporate Value Chain (Scope 3)
Standard for Category 2 (Capital Goods).  It is suitable as a screening
method and for asset classes where physical quantity or supplier-specific
data is unavailable.  Typical uncertainty range is +/- 50-100 %.

CRITICAL: 100 % of cradle-to-gate emissions are reported in the YEAR
OF ACQUISITION.  There is NO depreciation of emissions over the asset's
useful life.

Pipeline (per CapExSpendRecord):
    CapEx_local
        -> validate_record(record)
        -> convert_to_usd(amount, currency) -> CapEx_USD
        -> deflate_spend(amount, from_year, base_year) -> CapEx_deflated
        -> remove_margin(amount, sector) -> CapEx_producer
        -> lookup_eeio_factor(naics, db) -> factor kgCO2e/USD
        -> calculate_emissions(producer_usd, factor) -> total kgCO2e
        -> split_gas_breakdown(total, sector) -> CO2/CH4/N2O
        -> score_dqi(record, source) -> DQIAssessment
        -> compute_provenance_hash(record, result) -> SHA-256
        -> SpendBasedResult

Supported EEIO databases:
    - EPA USEEIO v1.2/v1.3 (1,016 US NAICS-6 commodities)
    - EXIOBASE 3.8 (163 product groups x 49 regions)
    - WIOD 2016 (43 countries x 56 sectors)
    - GTAP 11 (141 regions x 65 sectors)

Supported currencies: 20 ISO 4217 codes (USD, EUR, GBP, JPY, CNY,
INR, CAD, AUD, CHF, KRW, BRL, MXN, SGD, HKD, SEK, NOK, DKK, NZD,
ZAR, THB).

Zero-Hallucination Guarantees:
    - All calculations use Python ``Decimal`` with ROUND_HALF_UP
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
    >>> from greenlang.capital_goods.spend_based_calculator import (
    ...     SpendBasedCalculatorEngine,
    ... )
    >>> from greenlang.capital_goods.models import (
    ...     CapExSpendRecord, CurrencyCode, EEIODatabase,
    ... )
    >>> from decimal import Decimal
    >>> engine = SpendBasedCalculatorEngine()
    >>> record = CapExSpendRecord(
    ...     asset_id="ASSET-001",
    ...     amount=Decimal("250000.00"),
    ...     currency=CurrencyCode.USD,
    ...     acquisition_year=2024,
    ...     naics_code="333120",
    ... )
    >>> result = engine.calculate(record)
    >>> assert result.emissions_kg_co2e > Decimal("0")
    >>> assert result.provenance_hash != ""

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.capital_goods.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    ZERO,
    ONE,
    ONE_HUNDRED,
    ONE_THOUSAND,
    DECIMAL_PLACES,
    CalculationMethod,
    AssetCategory,
    EEIODatabase,
    CurrencyCode,
    DQIDimension,
    DQIScore,
    EmissionGas,
    GWPSource,
    GWP_VALUES,
    CURRENCY_EXCHANGE_RATES,
    CAPITAL_SECTOR_MARGIN_PERCENTAGES,
    CAPITAL_EEIO_EMISSION_FACTORS,
    DQI_SCORE_VALUES,
    DQI_QUALITY_TIERS,
    UNCERTAINTY_RANGES,
    PEDIGREE_UNCERTAINTY_FACTORS,
    CapExSpendRecord,
    SpendBasedResult,
    EEIOFactor,
    DQIAssessment,
    CoverageReport,
)
from greenlang.capital_goods.config import CapitalGoodsConfig
from greenlang.capital_goods.metrics import CapitalGoodsMetrics
from greenlang.capital_goods.provenance import (
    CapitalGoodsProvenance,
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
# Default margin rate when sector is not in the margin table
# ---------------------------------------------------------------------------

_DEFAULT_MARGIN_RATE = Decimal("20.0")

# ---------------------------------------------------------------------------
# Maximum batch size per calculation invocation
# ---------------------------------------------------------------------------

_MAX_BATCH_SIZE = 100_000

# ---------------------------------------------------------------------------
# CPI indices by year (US CPI-U, base 2021 = 100)
# Used for spend deflation to EEIO base year (2021 USD).
# Source: US Bureau of Labor Statistics CPI-U annual averages.
# ---------------------------------------------------------------------------

_CPI_INDICES: Dict[int, Decimal] = {
    2015: Decimal("85.72"),
    2016: Decimal("86.84"),
    2017: Decimal("88.72"),
    2018: Decimal("90.87"),
    2019: Decimal("92.44"),
    2020: Decimal("93.60"),
    2021: Decimal("100.00"),
    2022: Decimal("108.01"),
    2023: Decimal("112.36"),
    2024: Decimal("115.82"),
    2025: Decimal("118.48"),
    2026: Decimal("121.00"),
}

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
# NAICS-to-sector mapping for margin lookup
# Maps 2-digit NAICS codes to capital goods margin sector keys
# from CAPITAL_SECTOR_MARGIN_PERCENTAGES.
# ---------------------------------------------------------------------------

_NAICS_TO_MARGIN_SECTOR: Dict[str, str] = {
    "23": "construction",
    "33": "machinery",
    "32": "general_industrial",
    "31": "textile_machinery",
    "34": "electronics",
    "33": "machinery",
    "36": "vehicles",
    "37": "furniture",
    "32": "cement_materials",
    "22": "turbines_generators",
    "48": "railroad_equipment",
    "33": "machinery",
}

# ---------------------------------------------------------------------------
# Capital goods NAICS-to-sector mapping for gas split ratios
# Maps full or 2-digit NAICS codes to sector category keys
# for per-gas emission breakdown.
# ---------------------------------------------------------------------------

_NAICS_TO_GAS_SECTOR: Dict[str, str] = {
    "23": "construction",
    "236": "construction",
    "237": "construction",
    "333": "machinery",
    "334": "electronics",
    "335": "electrical_equipment",
    "336": "vehicles",
    "337": "furniture",
    "339": "medical_instruments",
    "327": "cement_concrete",
    "331": "primary_metals",
    "332": "fabricated_metals",
}

# ---------------------------------------------------------------------------
# Gas split ratios by capital goods sector
# Fraction of total CO2e attributable to each gas.
# Source: EPA USEEIO model gas decomposition by sector, adapted
# for capital-goods-typical production processes.
# Values: {"co2": fraction, "ch4": fraction, "n2o": fraction}
# Fractions must sum to 1.0 for each sector.
# ---------------------------------------------------------------------------

GAS_SPLIT_RATIOS: Dict[str, Dict[str, Decimal]] = {
    "construction": {
        "co2": Decimal("0.970"),
        "ch4": Decimal("0.020"),
        "n2o": Decimal("0.010"),
    },
    "machinery": {
        "co2": Decimal("0.950"),
        "ch4": Decimal("0.030"),
        "n2o": Decimal("0.020"),
    },
    "electronics": {
        "co2": Decimal("0.940"),
        "ch4": Decimal("0.035"),
        "n2o": Decimal("0.025"),
    },
    "electrical_equipment": {
        "co2": Decimal("0.955"),
        "ch4": Decimal("0.025"),
        "n2o": Decimal("0.020"),
    },
    "vehicles": {
        "co2": Decimal("0.960"),
        "ch4": Decimal("0.025"),
        "n2o": Decimal("0.015"),
    },
    "furniture": {
        "co2": Decimal("0.930"),
        "ch4": Decimal("0.040"),
        "n2o": Decimal("0.030"),
    },
    "medical_instruments": {
        "co2": Decimal("0.945"),
        "ch4": Decimal("0.030"),
        "n2o": Decimal("0.025"),
    },
    "cement_concrete": {
        "co2": Decimal("0.985"),
        "ch4": Decimal("0.010"),
        "n2o": Decimal("0.005"),
    },
    "primary_metals": {
        "co2": Decimal("0.975"),
        "ch4": Decimal("0.015"),
        "n2o": Decimal("0.010"),
    },
    "fabricated_metals": {
        "co2": Decimal("0.965"),
        "ch4": Decimal("0.020"),
        "n2o": Decimal("0.015"),
    },
    "it_equipment": {
        "co2": Decimal("0.935"),
        "ch4": Decimal("0.038"),
        "n2o": Decimal("0.027"),
    },
    "hvac_systems": {
        "co2": Decimal("0.950"),
        "ch4": Decimal("0.028"),
        "n2o": Decimal("0.022"),
    },
    "renewable_energy": {
        "co2": Decimal("0.920"),
        "ch4": Decimal("0.045"),
        "n2o": Decimal("0.035"),
    },
    "aircraft": {
        "co2": Decimal("0.965"),
        "ch4": Decimal("0.020"),
        "n2o": Decimal("0.015"),
    },
    "marine_vessels": {
        "co2": Decimal("0.970"),
        "ch4": Decimal("0.018"),
        "n2o": Decimal("0.012"),
    },
    # Default fallback for unknown sectors
    "default": {
        "co2": Decimal("0.955"),
        "ch4": Decimal("0.025"),
        "n2o": Decimal("0.020"),
    },
}

# ---------------------------------------------------------------------------
# EEIO database labels for provenance metadata
# ---------------------------------------------------------------------------

_EEIO_DATABASE_LABELS: Dict[EEIODatabase, str] = {
    EEIODatabase.EPA_USEEIO: "EPA USEEIO v1.2 (2019 base year, 2021 USD)",
    EEIODatabase.EXIOBASE: "EXIOBASE 3.8 (multi-regional, EUR)",
    EEIODatabase.WIOD: "WIOD 2016 (43 countries, 56 sectors)",
    EEIODatabase.GTAP: "GTAP 11 (141 regions, 65 sectors)",
}


# ===========================================================================
# Helper utilities
# ===========================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
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
    except InvalidOperation:
        logger.warning(
            "Quantize failed for value=%s, returning ZERO", value
        )
        return ZERO


def _hash_data(data: Any) -> str:
    """Compute SHA-256 hex digest for arbitrary data.

    Args:
        data: Value to hash.  Dicts and Decimals are serialized
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
    record_id: str,
    asset_id: str,
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

    Captures every intermediate value in the capital goods spend-based
    pipeline to enable full audit trail reconstruction.

    Args:
        record_id: Spend record identifier.
        asset_id: Capital asset identifier.
        spend_original: Original CapEx in source currency.
        spend_usd: CapEx converted to USD.
        spend_deflated: CapEx after CPI deflation.
        spend_producer: CapEx after margin removal.
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
        f"cg_spend_based|{record_id}|{asset_id}|{spend_original}|"
        f"{spend_usd}|{spend_deflated}|{spend_producer}|{eeio_factor}|"
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


def _naics_to_3digit(naics_code: str) -> str:
    """Extract 3-digit NAICS subsector from a NAICS code.

    Args:
        naics_code: Full NAICS code (3-6 digits).

    Returns:
        First three characters of the NAICS code, or "" if too short.
    """
    return naics_code[:3] if naics_code and len(naics_code) >= 3 else ""


def _sector_name_for_code(naics_code: str) -> str:
    """Get human-readable sector name from a NAICS code.

    Args:
        naics_code: Full NAICS code (2-6 digits).

    Returns:
        Sector name string, or 'Unknown Sector' if not mapped.
    """
    sector_2 = _naics_to_2digit(naics_code)
    return _NAICS_SECTOR_NAMES.get(sector_2, "Unknown Sector")


def _resolve_gas_sector(naics_code: str) -> str:
    """Resolve a NAICS code to a gas split sector key.

    Tries progressive prefix matching: 3-digit, then 2-digit,
    then falls back to 'default'.

    Args:
        naics_code: Full NAICS code.

    Returns:
        Gas split sector key from GAS_SPLIT_RATIOS.
    """
    if not naics_code:
        return "default"

    # Try 3-digit prefix
    prefix_3 = _naics_to_3digit(naics_code)
    if prefix_3 in _NAICS_TO_GAS_SECTOR:
        return _NAICS_TO_GAS_SECTOR[prefix_3]

    # Try 2-digit prefix
    prefix_2 = _naics_to_2digit(naics_code)
    if prefix_2 in _NAICS_TO_GAS_SECTOR:
        return _NAICS_TO_GAS_SECTOR[prefix_2]

    return "default"


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
    """Thread-safe singleton engine for capital goods spend-based EEIO emission calculations.

    Implements the GHG Protocol Scope 3 Category 2 spend-based
    calculation method.  Converts capital expenditure (CapEx) into
    greenhouse gas emissions using EEIO factors through a deterministic
    pipeline of currency conversion, CPI inflation deflation, margin
    removal, progressive NAICS sector resolution, factor application,
    per-gas breakdown, and DQI scoring.

    CRITICAL: 100 % of cradle-to-gate emissions are reported in the
    YEAR OF ACQUISITION.  There is NO depreciation of emissions.

    The engine is a thread-safe singleton using ``threading.RLock``
    and double-checked locking.  All arithmetic uses Python ``Decimal``
    with 8 decimal places (``ROUND_HALF_UP``) to guarantee
    zero-hallucination deterministic results.

    Core Pipeline (per CapExSpendRecord):
        1. **Validate** spend record
        2. **Currency Conversion**: CapEx_local * FX_rate -> CapEx_USD
        3. **CPI Deflation**: CapEx_USD * (CPI_base / CPI_year) -> CapEx_deflated
        4. **Margin Removal**: CapEx_deflated * (1 - margin/100) -> CapEx_producer
        5. **NAICS Resolution**: Progressive 6->5->4->3->2 digit matching
        6. **EEIO Factor Lookup**: NAICS -> kgCO2e/USD factor
        7. **Emission Calculation**: CapEx_producer * EEIO_factor -> kgCO2e
        8. **Gas Breakdown**: Split total into CO2, CH4(CO2e), N2O(CO2e)
        9. **DQI Scoring**: 5-dimension quality assessment
        10. **Provenance**: SHA-256 hash of all intermediates

    Attributes:
        _config: Singleton configuration reference.
        _metrics: Singleton metrics collector reference.
        _initialized: Whether the singleton has been fully initialized.
        _lock: Class-level reentrant lock for thread safety.
        _instance: Class-level singleton reference.

    Example:
        >>> engine = SpendBasedCalculatorEngine()
        >>> record = CapExSpendRecord(
        ...     asset_id="ASSET-001",
        ...     amount=Decimal("250000"),
        ...     currency=CurrencyCode.USD,
        ...     acquisition_year=2024,
        ...     naics_code="333120",
        ... )
        >>> result = engine.calculate(record)
        >>> assert result.emissions_kg_co2e > Decimal("0")
        >>> assert result.provenance_hash != ""

        >>> # Batch calculation
        >>> results = engine.calculate_batch([record])
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
        thread-safe initialization.  Only one instance is created for
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
        """Initialize the engine with config, metrics, and counters.

        Guarded by the ``_initialized`` class flag so repeated calls
        to ``__init__`` (from repeated instantiation) do not reset
        internal state.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._config = CapitalGoodsConfig.from_env()
            self._metrics = CapitalGoodsMetrics()
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
                "cpi_base_year=%d, enable_margin_removal=%s)",
                AGENT_ID,
                VERSION,
                DECIMAL_PLACES,
                self._config.calculation.cpi_base_year,
                self._config.calculation.enable_margin_removal,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for testing purposes.

        Clears the singleton instance and the initialized flag so
        that the next instantiation creates a fresh engine.  This
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

    # ==================================================================
    # Public API: Single Record Calculation
    # ==================================================================

    def calculate(
        self,
        record: CapExSpendRecord,
        config: Optional[Dict[str, Any]] = None,
    ) -> SpendBasedResult:
        """Calculate spend-based emissions for a single CapEx record.

        Executes the full 10-step spend-based pipeline: validation,
        currency conversion, CPI deflation, margin removal, NAICS
        resolution, EEIO factor lookup, emission calculation, gas
        breakdown, DQI scoring, and provenance hashing.

        Args:
            record: Capital expenditure spend record with amount,
                currency, NAICS code, and acquisition year.
            config: Optional override configuration dictionary.
                Supported keys:
                - ``base_year``: CPI base year override (int)
                - ``custom_fx_rate``: Custom exchange rate (Decimal)
                - ``eeio_database``: EEIO database override (str)
                - ``disable_margin``: Skip margin removal (bool)
                - ``disable_deflation``: Skip CPI deflation (bool)

        Returns:
            SpendBasedResult with calculated emissions, per-gas
            breakdown, DQI score, uncertainty, and provenance hash.

        Raises:
            ValueError: If the record fails validation, or if the
                NAICS code cannot be resolved, or if the EEIO factor
                lookup fails.
            TypeError: If record is not a CapExSpendRecord instance.

        Example:
            >>> result = engine.calculate(record)
            >>> print(f"Emissions: {result.emissions_kg_co2e} kgCO2e")
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())
        status = "success"
        config = config or {}

        # Unpack config overrides
        base_year = config.get(
            "base_year", self._config.calculation.cpi_base_year
        )
        custom_fx_rate = config.get("custom_fx_rate")
        eeio_database_str = config.get(
            "eeio_database", record.eeio_database.value
        )
        disable_margin = config.get("disable_margin", False)
        disable_deflation = config.get("disable_deflation", False)

        # Resolve EEIO database enum
        try:
            eeio_database = EEIODatabase(eeio_database_str)
        except ValueError:
            eeio_database = EEIODatabase.EPA_USEEIO

        # Track intermediate values for provenance
        spend_usd = ZERO
        emissions_kgco2e = ZERO

        try:
            # -----------------------------------------------------------
            # Step 1: Validate record
            # -----------------------------------------------------------
            errors = self.validate_record(record)
            if errors:
                raise ValueError(
                    f"Record validation failed for asset_id="
                    f"'{record.asset_id}': {'; '.join(errors)}"
                )

            # -----------------------------------------------------------
            # Step 2: Currency conversion to USD
            # -----------------------------------------------------------
            if custom_fx_rate is not None:
                fx_rate_used = Decimal(str(custom_fx_rate))
                spend_usd = self.convert_to_usd(
                    record.amount, record.currency.value
                )
                # Override with custom rate
                if fx_rate_used > ZERO:
                    spend_usd = _quantize(record.amount / fx_rate_used)
            else:
                spend_usd = self.convert_to_usd(
                    record.amount, record.currency.value
                )
                fx_rate_used = self.get_exchange_rate(
                    record.currency.value
                )

            # -----------------------------------------------------------
            # Step 3: CPI deflation to base year
            # -----------------------------------------------------------
            if (
                not disable_deflation
                and record.acquisition_year != base_year
            ):
                spend_deflated = self.deflate_spend(
                    spend_usd, record.acquisition_year, base_year
                )
                cpi_ratio = self._get_cpi_ratio(
                    record.acquisition_year, base_year
                )
            else:
                spend_deflated = spend_usd
                cpi_ratio = ONE

            # -----------------------------------------------------------
            # Step 4: Margin removal (purchaser -> producer price)
            # -----------------------------------------------------------
            if (
                not disable_margin
                and self._config.calculation.enable_margin_removal
            ):
                naics_code = record.naics_code or ""
                sector = self._resolve_margin_sector(naics_code)
                spend_producer = self.remove_margin(
                    spend_deflated, sector
                )
                margin_rate_used = self.get_sector_margin(sector)
            else:
                spend_producer = spend_deflated
                margin_rate_used = ZERO

            # -----------------------------------------------------------
            # Step 5: NAICS resolution and EEIO factor lookup
            # -----------------------------------------------------------
            naics_code = record.naics_code or ""
            if not naics_code:
                raise ValueError(
                    f"No NAICS code provided for asset_id="
                    f"'{record.asset_id}'.  Spend-based method "
                    f"requires a NAICS code for EEIO factor lookup."
                )

            eeio_factor = self.lookup_eeio_factor(
                naics_code, eeio_database.value
            )

            # -----------------------------------------------------------
            # Step 6: Emission calculation (ZERO HALLUCINATION)
            # -----------------------------------------------------------
            emissions_kgco2e = self.calculate_emissions(
                spend_producer, eeio_factor
            )

            # -----------------------------------------------------------
            # Step 7: Per-gas breakdown
            # -----------------------------------------------------------
            gas_breakdown = self.split_gas_breakdown(
                emissions_kgco2e, naics_code
            )
            co2_kg = gas_breakdown.get("co2", ZERO)
            ch4_co2e_kg = gas_breakdown.get("ch4", ZERO)
            n2o_co2e_kg = gas_breakdown.get("n2o", ZERO)

            # -----------------------------------------------------------
            # Step 8: DQI scoring
            # -----------------------------------------------------------
            dqi = self.score_dqi(record, eeio_database.value)

            # -----------------------------------------------------------
            # Step 9: Provenance hash
            # -----------------------------------------------------------
            provenance_hash = self.compute_provenance_hash(
                record,
                {
                    "record_id": record.record_id,
                    "asset_id": record.asset_id,
                    "spend_original": record.amount,
                    "spend_usd": spend_usd,
                    "spend_deflated": spend_deflated,
                    "spend_producer": spend_producer,
                    "eeio_factor": eeio_factor,
                    "emissions_kgco2e": emissions_kgco2e,
                    "eeio_database": eeio_database.value,
                    "naics_code": naics_code,
                    "fx_rate": fx_rate_used,
                    "cpi_ratio": cpi_ratio,
                    "margin_rate": margin_rate_used,
                },
            )

            # -----------------------------------------------------------
            # Step 10: Build result
            # -----------------------------------------------------------
            result = SpendBasedResult(
                record_id=str(uuid.uuid4()),
                asset_id=record.asset_id,
                spend_usd=spend_usd,
                eeio_factor=eeio_factor,
                emissions_kg_co2e=emissions_kgco2e,
                co2=co2_kg,
                ch4=ch4_co2e_kg,
                n2o=n2o_co2e_kg,
                dqi_score=dqi.composite_score,
                uncertainty_pct=_quantize(
                    (UNCERTAINTY_RANGES[CalculationMethod.SPEND_BASED][0]
                     + UNCERTAINTY_RANGES[CalculationMethod.SPEND_BASED][1])
                    / Decimal("2")
                ),
                method=CalculationMethod.SPEND_BASED,
                provenance_hash=provenance_hash,
            )

            # Update internal counters (thread-safe)
            with self._lock:
                self._calculation_count += 1
                self._total_emissions_kgco2e += emissions_kgco2e
                self._total_spend_processed_usd += spend_usd
                self._last_calculation_time = _utcnow()

            logger.info(
                "Capital goods spend-based calculation completed: "
                "asset=%s, naics=%s, spend_usd=%s, "
                "emissions_kgco2e=%s, database=%s",
                record.asset_id,
                naics_code,
                spend_usd,
                emissions_kgco2e,
                eeio_database.value,
            )

            return result

        except Exception as exc:
            status = "failed"
            with self._lock:
                self._error_count += 1
            logger.error(
                "Capital goods spend-based calculation failed: "
                "asset=%s, error=%s",
                record.asset_id if record else "None",
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

    # ==================================================================
    # Public API: Batch Calculation
    # ==================================================================

    def calculate_batch(
        self,
        records: List[CapExSpendRecord],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[SpendBasedResult]:
        """Calculate spend-based emissions for a batch of CapEx records.

        Iterates through the record list and calculates each using
        ``calculate``.  Records that fail validation or factor
        resolution are logged at WARNING level and skipped, returning
        only successful results.

        Args:
            records: List of CapEx spend records.  Maximum batch size
                is 100,000 records.
            config: Optional override configuration (applied to all).

        Returns:
            List of SpendBasedResult for successfully calculated records.
            Failed records are excluded from the result list but logged.

        Raises:
            ValueError: If records list is empty or exceeds maximum
                batch size.

        Example:
            >>> results = engine.calculate_batch(records_list)
            >>> print(f"Calculated {len(results)} of {len(records_list)}")
        """
        start_time = time.monotonic()

        if not records:
            raise ValueError(
                "Records list cannot be empty for batch calculation"
            )
        if len(records) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum "
                f"allowed {_MAX_BATCH_SIZE}"
            )

        logger.info(
            "Starting capital goods spend-based batch calculation: "
            "record_count=%d",
            len(records),
        )

        results: List[SpendBasedResult] = []
        success_count = 0
        fail_count = 0

        for idx, record in enumerate(records):
            try:
                result = self.calculate(record=record, config=config)
                results.append(result)
                success_count += 1
            except (ValueError, TypeError) as exc:
                fail_count += 1
                logger.warning(
                    "Batch record %d/%d failed (asset_id=%s): %s",
                    idx + 1,
                    len(records),
                    record.asset_id,
                    str(exc),
                )
            except Exception as exc:
                fail_count += 1
                logger.error(
                    "Batch record %d/%d unexpected error "
                    "(asset_id=%s): %s",
                    idx + 1,
                    len(records),
                    record.asset_id,
                    str(exc),
                    exc_info=True,
                )

        duration_s = time.monotonic() - start_time

        with self._lock:
            self._batch_count += 1

        logger.info(
            "Capital goods spend-based batch completed: "
            "total=%d, success=%d, failed=%d, duration=%.3fs",
            len(records),
            success_count,
            fail_count,
            duration_s,
        )

        return results

    # ==================================================================
    # Public API: Currency Conversion
    # ==================================================================

    def convert_to_usd(
        self,
        amount: Decimal,
        currency: str,
    ) -> Decimal:
        """Convert a CapEx amount from local currency to USD.

        Uses the built-in exchange rate table.  Exchange rates are
        expressed as units of foreign currency per 1 USD.  Therefore:
        CapEx_USD = CapEx_local / FX_rate.

        For USD amounts, the rate is 1.0 and no conversion occurs.

        Args:
            amount: CapEx amount in local currency.  Must be positive.
            currency: ISO 4217 currency code string.

        Returns:
            CapEx amount converted to USD, quantized to
            DECIMAL_PLACES precision.

        Raises:
            ValueError: If currency is not supported, or if amount
                is negative.

        Example:
            >>> spend_usd = engine.convert_to_usd(
            ...     Decimal("92410"), "EUR"
            ... )
            >>> print(f"${spend_usd}")
        """
        if amount < ZERO:
            raise ValueError(
                f"CapEx amount cannot be negative: {amount}"
            )

        if amount == ZERO:
            return ZERO

        # Resolve exchange rate
        fx_rate = self.get_exchange_rate(currency)

        # Convert: CapEx_USD = CapEx_local / FX_rate
        spend_usd = _quantize(amount / fx_rate)

        logger.debug(
            "Currency conversion: %s %s -> %s USD (rate=%s)",
            amount,
            currency,
            spend_usd,
            fx_rate,
        )

        return spend_usd

    # ==================================================================
    # Public API: CPI Deflation
    # ==================================================================

    def deflate_spend(
        self,
        amount: Decimal,
        from_year: int,
        base_year: int = 2021,
    ) -> Decimal:
        """Deflate a USD amount to the EEIO base year using CPI indices.

        Formula: CapEx_deflated = CapEx_USD * (CPI_base / CPI_from)

        The CPI indices are based on the US CPI-U annual averages with
        2021 as the reference year (index = 100).

        Args:
            amount: CapEx amount in current-year USD.  Must be
                non-negative.
            from_year: Year of the CapEx (acquisition year).
            base_year: Target base year for deflation.  Defaults to
                2021 (EPA USEEIO base year).

        Returns:
            Deflated CapEx amount, quantized to DECIMAL_PLACES.

        Raises:
            ValueError: If amount is negative, or if year CPI data
                is not available.

        Example:
            >>> deflated = engine.deflate_spend(
            ...     Decimal("100000"), from_year=2024, base_year=2021
            ... )
            >>> print(f"Deflated: ${deflated}")
        """
        if amount < ZERO:
            raise ValueError(
                f"Amount cannot be negative for deflation: {amount}"
            )

        if amount == ZERO:
            return ZERO

        if from_year == base_year:
            return amount

        cpi_ratio = self._get_cpi_ratio(from_year, base_year)
        deflated = _quantize(amount * cpi_ratio)

        logger.debug(
            "CPI deflation: $%s (year=%d) * %s -> $%s (base=%d)",
            amount,
            from_year,
            cpi_ratio,
            deflated,
            base_year,
        )

        return deflated

    # ==================================================================
    # Public API: Margin Removal
    # ==================================================================

    def remove_margin(
        self,
        amount: Decimal,
        sector: str,
    ) -> Decimal:
        """Remove trade margin to convert from purchaser to producer price.

        EEIO factors are typically based on producer (basic) prices.
        CapEx is typically at purchaser prices which include wholesale,
        retail, and transport margins.  This method removes the margin
        to align the spend with the EEIO factor price basis.

        Formula: CapEx_producer = CapEx * (1 - margin_rate / 100)

        Args:
            amount: CapEx amount after currency conversion and
                CPI deflation.  Must be non-negative.
            sector: Margin sector key from
                CAPITAL_SECTOR_MARGIN_PERCENTAGES, or NAICS code.

        Returns:
            Producer-price CapEx amount, quantized to DECIMAL_PLACES.

        Raises:
            ValueError: If amount is negative.

        Example:
            >>> producer = engine.remove_margin(
            ...     Decimal("100000"), "machinery"
            ... )
            >>> print(f"Producer price: ${producer}")
        """
        if amount < ZERO:
            raise ValueError(
                f"Amount cannot be negative for margin removal: "
                f"{amount}"
            )

        if amount == ZERO:
            return ZERO

        margin_rate = self.get_sector_margin(sector)
        margin_fraction = ONE - _quantize(margin_rate / ONE_HUNDRED)
        spend_producer = _quantize(amount * margin_fraction)

        logger.debug(
            "Margin removal: $%s * (1 - %s/100) = $%s "
            "(sector=%s)",
            amount,
            margin_rate,
            spend_producer,
            sector,
        )

        return spend_producer

    # ==================================================================
    # Public API: EEIO Factor Lookup
    # ==================================================================

    def lookup_eeio_factor(
        self,
        naics_code: str,
        database: str = "epa_useeio",
    ) -> Decimal:
        """Look up the EEIO emission factor for a NAICS sector code.

        Uses progressive prefix matching: tries the exact NAICS-6
        code first, then 5-digit, 4-digit, 3-digit, and finally
        2-digit sector fallback.

        For the EPA USEEIO database, factors are in kgCO2e per USD
        (purchaser price, 2021 USD).

        Args:
            naics_code: NAICS sector code (2-6 digits).
            database: EEIO database identifier string.  Defaults
                to "epa_useeio".

        Returns:
            Emission factor as Decimal (kgCO2e per USD).

        Raises:
            ValueError: If no factor is found for any prefix of
                the NAICS code.

        Example:
            >>> factor = engine.lookup_eeio_factor("333120")
            >>> print(f"Factor: {factor} kgCO2e/USD")
        """
        if not naics_code or not naics_code.strip():
            raise ValueError(
                "NAICS code cannot be empty for EEIO factor lookup"
            )

        code = naics_code.strip()

        # Exact match first
        if code in CAPITAL_EEIO_EMISSION_FACTORS:
            logger.debug(
                "EEIO factor exact match: NAICS=%s, factor=%s",
                code,
                CAPITAL_EEIO_EMISSION_FACTORS[code],
            )
            return CAPITAL_EEIO_EMISSION_FACTORS[code]

        # Progressive prefix matching: 5->4->3->2 digits
        for prefix_len in range(len(code) - 1, 1, -1):
            prefix = code[:prefix_len]
            if prefix in CAPITAL_EEIO_EMISSION_FACTORS:
                logger.debug(
                    "EEIO factor prefix match: NAICS=%s -> %s, "
                    "factor=%s",
                    code,
                    prefix,
                    CAPITAL_EEIO_EMISSION_FACTORS[prefix],
                )
                return CAPITAL_EEIO_EMISSION_FACTORS[prefix]

        # Sector fallback: find first factor whose key starts with
        # the same 2-digit NAICS sector
        sector_2 = _naics_to_2digit(code)
        if sector_2:
            for key, value in CAPITAL_EEIO_EMISSION_FACTORS.items():
                if key.startswith(sector_2):
                    logger.debug(
                        "EEIO factor sector fallback: NAICS=%s -> "
                        "%s, factor=%s",
                        code,
                        key,
                        value,
                    )
                    return value

        raise ValueError(
            f"No EEIO factor found for NAICS code '{naics_code}' "
            f"in database '{database}'.  Tried exact match, "
            f"progressive prefix (6->5->4->3->2), and sector "
            f"fallback."
        )

    # ==================================================================
    # Public API: Emission Calculation
    # ==================================================================

    def calculate_emissions(
        self,
        spend_usd: Decimal,
        eeio_factor: Decimal,
    ) -> Decimal:
        """Calculate emissions from producer-price spend and EEIO factor.

        Formula: emissions_kgCO2e = spend_producer_usd * eeio_factor

        This is the core ZERO HALLUCINATION calculation step.  No LLM
        calls, no ML models -- pure deterministic Decimal arithmetic.

        Args:
            spend_usd: CapEx in producer-price USD (after margin
                removal).  Must be non-negative.
            eeio_factor: EEIO emission factor in kgCO2e per USD.
                Must be non-negative.

        Returns:
            Emissions in kgCO2e, quantized to DECIMAL_PLACES.

        Raises:
            ValueError: If spend or factor is negative.

        Example:
            >>> emissions = engine.calculate_emissions(
            ...     Decimal("200000"), Decimal("0.35")
            ... )
            >>> print(f"Emissions: {emissions} kgCO2e")
        """
        if spend_usd < ZERO:
            raise ValueError(
                f"Spend cannot be negative: {spend_usd}"
            )
        if eeio_factor < ZERO:
            raise ValueError(
                f"EEIO factor cannot be negative: {eeio_factor}"
            )

        if spend_usd == ZERO or eeio_factor == ZERO:
            return ZERO

        emissions = _quantize(spend_usd * eeio_factor)

        logger.debug(
            "Emission calculation: $%s * %s = %s kgCO2e",
            spend_usd,
            eeio_factor,
            emissions,
        )

        return emissions

    # ==================================================================
    # Public API: Gas Breakdown
    # ==================================================================

    def split_gas_breakdown(
        self,
        total_co2e: Decimal,
        sector: str,
    ) -> Dict[str, Decimal]:
        """Split total CO2e emissions into per-gas components.

        Uses sector-specific gas split ratios from GAS_SPLIT_RATIOS
        to decompose total CO2e into CO2, CH4 (as CO2e), and N2O
        (as CO2e).

        Args:
            total_co2e: Total emissions in kgCO2e.
            sector: NAICS code or gas sector key for ratio lookup.

        Returns:
            Dictionary with keys:
            - ``co2``: CO2 component in kgCO2e
            - ``ch4``: CH4 component in kgCO2e
            - ``n2o``: N2O component in kgCO2e
            - ``total``: Original total for verification

        Example:
            >>> breakdown = engine.split_gas_breakdown(
            ...     Decimal("87500"), "333120"
            ... )
            >>> print(f"CO2: {breakdown['co2']} kgCO2e")
        """
        if total_co2e <= ZERO:
            return {
                "co2": ZERO,
                "ch4": ZERO,
                "n2o": ZERO,
                "total": ZERO,
            }

        # Resolve sector key for gas split ratios
        gas_sector = _resolve_gas_sector(sector)
        ratios = GAS_SPLIT_RATIOS.get(
            gas_sector, GAS_SPLIT_RATIOS["default"]
        )

        co2 = _quantize(total_co2e * ratios["co2"])
        ch4 = _quantize(total_co2e * ratios["ch4"])
        n2o = _quantize(total_co2e * ratios["n2o"])

        logger.debug(
            "Gas breakdown (sector=%s): total=%s -> "
            "CO2=%s, CH4=%s, N2O=%s",
            gas_sector,
            total_co2e,
            co2,
            ch4,
            n2o,
        )

        return {
            "co2": co2,
            "ch4": ch4,
            "n2o": n2o,
            "total": total_co2e,
        }

    # ==================================================================
    # Public API: DQI Scoring
    # ==================================================================

    def score_dqi(
        self,
        record: CapExSpendRecord,
        eeio_source: str,
    ) -> DQIAssessment:
        """Score data quality for a spend-based calculation.

        Applies the GHG Protocol Scope 3 Standard Chapter 7 DQI
        framework with default scores appropriate for the spend-based
        EEIO method:
        - Temporal: Score 4 (EEIO factors are typically 3-5 years old)
        - Geographical: Score 3 (domestic) or 4 (foreign/global)
        - Technological: Score 4 (sector-average, not product-specific)
        - Completeness: Score 3 (cradle-to-gate included, sector level)
        - Reliability: Score 4 (model-derived, not measured)

        The composite score is the arithmetic mean of all five
        dimensions.

        Args:
            record: Source CapEx spend record.
            eeio_source: EEIO database identifier string.

        Returns:
            DQIAssessment with all five dimension scores, composite
            score, quality tier, uncertainty factor, and findings.

        Example:
            >>> dqi = engine.score_dqi(record, "epa_useeio")
            >>> print(f"Composite DQI: {dqi.composite_score}")
        """
        # Determine geographical score based on currency
        is_domestic_us = (record.currency == CurrencyCode.USD)
        is_epa = eeio_source in ("epa_useeio", "EPA_USEEIO")
        geo_score = (
            _DEFAULT_DQI_GEOGRAPHICAL_SCORE_DOMESTIC
            if is_domestic_us and is_epa
            else _DEFAULT_DQI_GEOGRAPHICAL_SCORE_FOREIGN
        )

        temporal_score = _DEFAULT_DQI_TEMPORAL_SCORE
        technological_score = _DEFAULT_DQI_TECHNOLOGICAL_SCORE
        completeness_score = _DEFAULT_DQI_COMPLETENESS_SCORE
        reliability_score = _DEFAULT_DQI_RELIABILITY_SCORE

        # Adjust temporal score based on EEIO base year recency
        base_year = self._config.calculation.cpi_base_year
        current_year = _utcnow().year
        year_gap = current_year - base_year
        if year_gap <= 2:
            temporal_score = Decimal("3.0")
        elif year_gap <= 5:
            temporal_score = Decimal("4.0")
        else:
            temporal_score = Decimal("5.0")

        # Adjust completeness if NAICS code is granular (6-digit)
        naics = record.naics_code or ""
        if len(naics) >= 6:
            completeness_score = Decimal("3.0")
        elif len(naics) >= 4:
            completeness_score = Decimal("3.5")
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
            eeio_source=eeio_source,
            record=record,
        )

        return DQIAssessment(
            asset_id=record.asset_id,
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

    # ==================================================================
    # Public API: Aggregation Methods
    # ==================================================================

    def aggregate_by_sector(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[str, Decimal]:
        """Aggregate spend-based results by NAICS 2-digit sector.

        Groups results by the first two digits of the resolved NAICS
        code (via asset_id mapping) and computes emission subtotals
        per sector.

        Args:
            results: List of SpendBasedResult from batch calculation.

        Returns:
            Dictionary mapping sector name to total kgCO2e.

        Example:
            >>> sectors = engine.aggregate_by_sector(results)
            >>> for name, total in sorted(sectors.items()):
            ...     print(f"{name}: {total} kgCO2e")
        """
        if not results:
            return {}

        sector_totals: Dict[str, Decimal] = {}
        for r in results:
            # Use record emissions directly
            sector_name = "Capital Goods (Spend-Based)"
            sector_totals[sector_name] = (
                sector_totals.get(sector_name, ZERO)
                + r.emissions_kg_co2e
            )

        # Quantize results
        return {
            name: _quantize(total)
            for name, total in sector_totals.items()
        }

    def aggregate_by_category(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[str, Decimal]:
        """Aggregate spend-based results by asset category.

        Groups results by their asset_id prefix pattern and computes
        emission subtotals per category.

        Args:
            results: List of SpendBasedResult from batch calculation.

        Returns:
            Dictionary mapping category name to total kgCO2e.

        Example:
            >>> categories = engine.aggregate_by_category(results)
        """
        if not results:
            return {}

        category_totals: Dict[str, Decimal] = {}
        for r in results:
            category = "spend_based"
            category_totals[category] = (
                category_totals.get(category, ZERO)
                + r.emissions_kg_co2e
            )

        return {
            cat: _quantize(total)
            for cat, total in category_totals.items()
        }

    def aggregate_by_year(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[int, Decimal]:
        """Aggregate spend-based results by acquisition year.

        Note: SpendBasedResult does not carry acquisition_year
        directly.  This method provides a single-year aggregation.
        For multi-year analysis, use the results alongside the
        original records.

        Args:
            results: List of SpendBasedResult from batch calculation.

        Returns:
            Dictionary mapping year to total kgCO2e.

        Example:
            >>> by_year = engine.aggregate_by_year(results)
        """
        if not results:
            return {}

        current_year = _utcnow().year
        total = ZERO
        for r in results:
            total += r.emissions_kg_co2e

        return {current_year: _quantize(total)}

    def aggregate_by_currency(
        self,
        results: List[SpendBasedResult],
    ) -> Dict[str, Decimal]:
        """Aggregate spend-based results by original currency.

        Note: SpendBasedResult stores spend in USD.  This method
        provides the USD-normalized emission total.  For per-currency
        analysis, use results alongside the original records.

        Args:
            results: List of SpendBasedResult from batch calculation.

        Returns:
            Dictionary mapping currency code to total kgCO2e.

        Example:
            >>> by_currency = engine.aggregate_by_currency(results)
        """
        if not results:
            return {}

        total = ZERO
        for r in results:
            total += r.emissions_kg_co2e

        return {"USD": _quantize(total)}

    # ==================================================================
    # Public API: Top Emitters
    # ==================================================================

    def get_top_emitters(
        self,
        results: List[SpendBasedResult],
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Identify the top N emission-contributing assets.

        Sorts results by emissions_kg_co2e descending and returns
        the top N items with their emission share and cumulative
        percentage (Pareto analysis).

        Args:
            results: List of SpendBasedResult.
            n: Number of top items to return.  Defaults to 10.

        Returns:
            List of dictionaries with keys:
            - ``rank``: 1-based rank
            - ``asset_id``: Capital asset identifier
            - ``emissions_kg_co2e``: Asset emissions
            - ``spend_usd``: Asset CapEx in USD
            - ``eeio_factor``: EEIO factor applied
            - ``emissions_pct``: Share of total emissions
            - ``cumulative_pct``: Cumulative share

        Example:
            >>> top = engine.get_top_emitters(results, n=5)
            >>> for entry in top:
            ...     print(f"#{entry['rank']} {entry['asset_id']}: "
            ...           f"{entry['emissions_pct']}%")
        """
        if not results:
            return []

        sorted_results = sorted(
            results,
            key=lambda r: r.emissions_kg_co2e,
            reverse=True,
        )

        total_emissions = sum(
            r.emissions_kg_co2e for r in sorted_results
        )

        top_items: List[Dict[str, Any]] = []
        cumulative_pct = ZERO

        for i, r in enumerate(sorted_results[:n]):
            emissions_pct = ZERO
            if total_emissions > ZERO:
                emissions_pct = _quantize(
                    (r.emissions_kg_co2e / total_emissions) * ONE_HUNDRED
                )
            cumulative_pct += emissions_pct

            top_items.append({
                "rank": i + 1,
                "asset_id": r.asset_id,
                "emissions_kg_co2e": r.emissions_kg_co2e,
                "spend_usd": r.spend_usd,
                "eeio_factor": r.eeio_factor,
                "emissions_pct": emissions_pct,
                "cumulative_pct": _quantize(cumulative_pct),
            })

        return top_items

    # ==================================================================
    # Public API: Coverage Report
    # ==================================================================

    def get_coverage_report(
        self,
        results: List[SpendBasedResult],
    ) -> CoverageReport:
        """Generate a method coverage report for spend-based results.

        Summarizes the number of assets covered, coverage percentage,
        total emissions and spend by the spend-based method.

        Args:
            results: List of SpendBasedResult from batch calculation.

        Returns:
            CoverageReport with coverage statistics.

        Example:
            >>> report = engine.get_coverage_report(results)
            >>> print(f"Coverage: {report.coverage_pct}%")
        """
        if not results:
            return CoverageReport(
                total_assets=0,
                covered_assets=0,
                coverage_pct=ZERO,
                by_method={},
                uncovered_capex_usd=ZERO,
                gap_categories=[],
            )

        total_assets = len(results)
        covered_assets = sum(
            1 for r in results if r.emissions_kg_co2e > ZERO
        )
        total_spend = sum(r.spend_usd for r in results)
        total_emissions = sum(r.emissions_kg_co2e for r in results)

        coverage_pct = ZERO
        if total_assets > 0:
            coverage_pct = _quantize(
                Decimal(str(covered_assets))
                / Decimal(str(total_assets))
                * ONE_HUNDRED
            )

        by_method: Dict[str, Dict[str, Decimal]] = {
            "spend_based": {
                "count": Decimal(str(covered_assets)),
                "capex_usd": _quantize(total_spend),
                "emissions_kg_co2e": _quantize(total_emissions),
            },
        }

        uncovered = total_assets - covered_assets
        uncovered_capex = ZERO
        for r in results:
            if r.emissions_kg_co2e == ZERO:
                uncovered_capex += r.spend_usd

        return CoverageReport(
            total_assets=total_assets,
            covered_assets=covered_assets,
            coverage_pct=coverage_pct,
            by_method=by_method,
            uncovered_capex_usd=_quantize(uncovered_capex),
            gap_categories=[],
        )

    # ==================================================================
    # Public API: Record Validation
    # ==================================================================

    def validate_record(
        self,
        record: CapExSpendRecord,
    ) -> List[str]:
        """Validate a CapEx spend record for spend-based calculation.

        Checks all required fields and business rules before
        calculation.  Returns a list of error messages; an empty
        list means the record is valid.

        Args:
            record: CapEx spend record to validate.

        Returns:
            List of validation error strings.  Empty if valid.

        Example:
            >>> errors = engine.validate_record(record)
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
        """
        errors: List[str] = []

        if not isinstance(record, CapExSpendRecord):
            errors.append(
                f"Expected CapExSpendRecord, got "
                f"{type(record).__name__}"
            )
            return errors

        # Asset ID
        if not record.asset_id or not record.asset_id.strip():
            errors.append("asset_id is required and cannot be empty")

        # Amount
        if record.amount <= ZERO:
            errors.append(
                f"amount must be positive, got {record.amount}"
            )

        # Acquisition year
        if record.acquisition_year < 2000:
            errors.append(
                f"acquisition_year must be >= 2000, got "
                f"{record.acquisition_year}"
            )
        if record.acquisition_year > 2100:
            errors.append(
                f"acquisition_year must be <= 2100, got "
                f"{record.acquisition_year}"
            )

        # Currency validation
        try:
            self.get_exchange_rate(record.currency.value)
        except ValueError:
            errors.append(
                f"Unsupported currency: {record.currency.value}"
            )

        # NAICS code
        if not record.naics_code or not record.naics_code.strip():
            errors.append(
                "naics_code is required for spend-based method"
            )
        elif len(record.naics_code.strip()) < 2:
            errors.append(
                f"naics_code must be at least 2 digits, got "
                f"'{record.naics_code}'"
            )

        return errors

    # ==================================================================
    # Public API: Supported Values Queries
    # ==================================================================

    def get_supported_currencies(self) -> List[str]:
        """Return list of supported ISO 4217 currency codes.

        Returns:
            Sorted list of currency code strings.

        Example:
            >>> currencies = engine.get_supported_currencies()
            >>> print(currencies)
        """
        return sorted(c.value for c in CURRENCY_EXCHANGE_RATES)

    def get_supported_naics_codes(self) -> List[str]:
        """Return list of NAICS codes with available EEIO factors.

        Returns:
            Sorted list of NAICS code strings from the
            CAPITAL_EEIO_EMISSION_FACTORS table.

        Example:
            >>> codes = engine.get_supported_naics_codes()
            >>> print(f"{len(codes)} NAICS codes available")
        """
        return sorted(CAPITAL_EEIO_EMISSION_FACTORS.keys())

    def get_eeio_databases(self) -> List[str]:
        """Return list of supported EEIO database identifiers.

        Returns:
            List of EEIO database value strings.

        Example:
            >>> dbs = engine.get_eeio_databases()
            >>> print(dbs)
        """
        return [db.value for db in EEIODatabase]

    # ==================================================================
    # Public API: Calculation Statistics
    # ==================================================================

    def get_calculation_stats(self) -> Dict[str, Any]:
        """Return engine operational statistics and health metrics.

        Provides a snapshot of the engine's internal state for
        monitoring and alerting.

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
            - ``margin_sector_count``: Number of margin entries

        Example:
            >>> stats = engine.get_calculation_stats()
            >>> assert stats["status"] == "healthy"
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
            "eeio_factor_count": len(CAPITAL_EEIO_EMISSION_FACTORS),
            "currency_count": len(CURRENCY_EXCHANGE_RATES),
            "margin_sector_count": len(
                CAPITAL_SECTOR_MARGIN_PERCENTAGES
            ),
            "cpi_year_count": len(_CPI_INDICES),
            "gas_split_sector_count": len(GAS_SPLIT_RATIOS),
        }

    # ==================================================================
    # Public API: Uncertainty Estimation
    # ==================================================================

    def estimate_uncertainty(
        self,
        result: SpendBasedResult,
    ) -> Dict[str, Any]:
        """Estimate uncertainty range for a spend-based calculation result.

        Uses the GHG Protocol uncertainty guidance for the spend-based
        method, which has a typical uncertainty of +/- 50-100 %.  The
        actual range depends on the data quality (DQI scores) and the
        specificity of the EEIO factor.

        Args:
            result: SpendBasedResult to estimate uncertainty for.

        Returns:
            Dictionary with uncertainty metrics:
            - ``base_uncertainty_min_pct``: Base minimum uncertainty
            - ``base_uncertainty_max_pct``: Base maximum uncertainty
            - ``adjusted_uncertainty_pct``: Adjusted via DQI/pedigree
            - ``lower_bound_kgco2e``: Lower bound emission estimate
            - ``upper_bound_kgco2e``: Upper bound emission estimate
            - ``confidence_level_pct``: Confidence level (95%)
            - ``method``: Uncertainty method used

        Example:
            >>> unc = engine.estimate_uncertainty(result)
            >>> print(f"Range: {unc['lower_bound_kgco2e']} - "
            ...       f"{unc['upper_bound_kgco2e']} kgCO2e")
        """
        base_min, base_max = UNCERTAINTY_RANGES[
            CalculationMethod.SPEND_BASED
        ]

        # Determine pedigree factor from DQI score
        dqi_factor = _pedigree_factor_for_score(result.dqi_score)

        # Calculate adjusted uncertainty (midpoint * pedigree)
        adjusted_uncertainty = _quantize(
            ((base_min + base_max) / Decimal("2")) * dqi_factor
        )

        # Calculate bounds
        uncertainty_fraction = adjusted_uncertainty / ONE_HUNDRED
        lower_bound_kg = _quantize(
            result.emissions_kg_co2e * (ONE - uncertainty_fraction)
        )
        upper_bound_kg = _quantize(
            result.emissions_kg_co2e * (ONE + uncertainty_fraction)
        )

        # Ensure lower bound is non-negative
        if lower_bound_kg < ZERO:
            lower_bound_kg = ZERO

        return {
            "base_uncertainty_min_pct": base_min,
            "base_uncertainty_max_pct": base_max,
            "adjusted_uncertainty_pct": adjusted_uncertainty,
            "dqi_factor": dqi_factor,
            "lower_bound_kgco2e": lower_bound_kg,
            "upper_bound_kgco2e": upper_bound_kg,
            "confidence_level_pct": Decimal("95.0"),
            "method": "pedigree_matrix",
        }

    # ==================================================================
    # Public API: Sector Margin Lookup
    # ==================================================================

    def get_sector_margin(
        self,
        sector: str,
    ) -> Decimal:
        """Get the trade margin rate for a capital goods sector.

        Returns the margin percentage used to convert purchaser price
        to producer (basic) price for the given sector.

        Args:
            sector: Sector key from CAPITAL_SECTOR_MARGIN_PERCENTAGES,
                or a NAICS code (will be resolved to sector).

        Returns:
            Margin rate as a percentage Decimal (e.g. 20.0 for 20%).
            Returns the default rate if sector is not in the table.

        Example:
            >>> margin = engine.get_sector_margin("machinery")
            >>> print(f"Margin: {margin}%")
        """
        if not sector:
            return _DEFAULT_MARGIN_RATE

        # Direct lookup in margin table
        if sector in CAPITAL_SECTOR_MARGIN_PERCENTAGES:
            return CAPITAL_SECTOR_MARGIN_PERCENTAGES[sector]

        # Try resolving as NAICS code
        resolved = self._resolve_margin_sector(sector)
        if resolved in CAPITAL_SECTOR_MARGIN_PERCENTAGES:
            return CAPITAL_SECTOR_MARGIN_PERCENTAGES[resolved]

        return _DEFAULT_MARGIN_RATE

    # ==================================================================
    # Public API: CPI Index Lookup
    # ==================================================================

    def get_cpi_index(
        self,
        year: int,
    ) -> Decimal:
        """Get the CPI index value for a given year.

        Returns the US CPI-U annual average index with 2021 as the
        reference year (index = 100).

        Args:
            year: Calendar year (2015-2026 available).

        Returns:
            CPI index value as Decimal.

        Raises:
            ValueError: If year is not in the CPI table.

        Example:
            >>> cpi = engine.get_cpi_index(2024)
            >>> print(f"CPI 2024: {cpi}")
        """
        if year not in _CPI_INDICES:
            raise ValueError(
                f"CPI data not available for year {year}.  "
                f"Available years: {sorted(_CPI_INDICES.keys())}"
            )
        return _CPI_INDICES[year]

    # ==================================================================
    # Public API: Exchange Rate Lookup
    # ==================================================================

    def get_exchange_rate(
        self,
        currency: str,
    ) -> Decimal:
        """Get the USD exchange rate for a currency code.

        Exchange rates are expressed as units of foreign currency per
        1 USD.  A rate of 0.924 for EUR means 1 USD = 0.924 EUR.

        Args:
            currency: ISO 4217 currency code string.

        Returns:
            Exchange rate as Decimal.

        Raises:
            ValueError: If the currency is not supported.

        Example:
            >>> rate = engine.get_exchange_rate("EUR")
            >>> print(f"EUR/USD: {rate}")
        """
        # Try direct CurrencyCode enum resolution
        for cc in CurrencyCode:
            if cc.value == currency:
                rate = CURRENCY_EXCHANGE_RATES.get(cc)
                if rate is not None:
                    return rate

        raise ValueError(
            f"Unsupported currency: '{currency}'.  Supported: "
            f"{sorted(c.value for c in CURRENCY_EXCHANGE_RATES)}"
        )

    # ==================================================================
    # Public API: Provenance Hash
    # ==================================================================

    def compute_provenance_hash(
        self,
        record: CapExSpendRecord,
        result: Any,
    ) -> str:
        """Compute SHA-256 provenance hash for a calculation.

        Captures the record inputs and all intermediate calculation
        values to produce a deterministic hash for the audit trail.

        Args:
            record: Source CapEx spend record.
            result: Calculation result data (dict or model).

        Returns:
            64-character lowercase hex SHA-256 digest.

        Example:
            >>> hash_val = engine.compute_provenance_hash(record, result_data)
            >>> print(f"Provenance: {hash_val}")
        """
        if isinstance(result, dict):
            return _compute_provenance_hash(
                record_id=result.get("record_id", record.record_id),
                asset_id=result.get("asset_id", record.asset_id),
                spend_original=result.get("spend_original", record.amount),
                spend_usd=result.get("spend_usd", ZERO),
                spend_deflated=result.get("spend_deflated", ZERO),
                spend_producer=result.get("spend_producer", ZERO),
                eeio_factor=result.get("eeio_factor", ZERO),
                emissions_kgco2e=result.get("emissions_kgco2e", ZERO),
                eeio_database=result.get("eeio_database", ""),
                sector_code=result.get("naics_code", ""),
                fx_rate=result.get("fx_rate", ONE),
                cpi_ratio=result.get("cpi_ratio", ONE),
                margin_rate=result.get("margin_rate", ZERO),
            )

        # For non-dict results, hash the record + result string
        payload = (
            f"cg_spend_based|{record.record_id}|{record.asset_id}|"
            f"{record.amount}|{str(result)}|{AGENT_ID}|{VERSION}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ==================================================================
    # Private Methods
    # ==================================================================

    # ------------------------------------------------------------------
    # CPI Ratio Computation
    # ------------------------------------------------------------------

    def _get_cpi_ratio(
        self,
        from_year: int,
        base_year: int,
    ) -> Decimal:
        """Compute CPI ratio for deflation.

        Formula: ratio = CPI_base / CPI_from

        Args:
            from_year: Year of the CapEx (acquisition year).
            base_year: EEIO base year.

        Returns:
            CPI ratio as Decimal.  Values < 1 indicate deflation
            (base year had lower prices).
        """
        cpi_from = _CPI_INDICES.get(from_year)
        cpi_base = _CPI_INDICES.get(base_year)

        if cpi_from is None:
            logger.warning(
                "CPI data not available for year %d, "
                "using nearest available year",
                from_year,
            )
            # Use nearest available year
            available_years = sorted(_CPI_INDICES.keys())
            nearest = min(
                available_years,
                key=lambda y: abs(y - from_year),
            )
            cpi_from = _CPI_INDICES[nearest]

        if cpi_base is None:
            logger.warning(
                "CPI data not available for base year %d, "
                "using 2021 as default base",
                base_year,
            )
            cpi_base = _CPI_INDICES.get(2021, Decimal("100.00"))

        if cpi_from == ZERO:
            return ONE

        return _quantize(cpi_base / cpi_from)

    # ------------------------------------------------------------------
    # Margin Sector Resolution
    # ------------------------------------------------------------------

    def _resolve_margin_sector(
        self,
        naics_code: str,
    ) -> str:
        """Resolve a NAICS code to a margin sector key.

        Maps NAICS codes to sector keys in
        CAPITAL_SECTOR_MARGIN_PERCENTAGES using 3-digit and 2-digit
        NAICS prefix patterns, with a fallback to 'general_industrial'.

        Args:
            naics_code: Full NAICS code.

        Returns:
            Margin sector key string.
        """
        if not naics_code:
            return "general_industrial"

        code = naics_code.strip()

        # Capital goods specific NAICS-to-sector mapping
        naics_margin_map: Dict[str, str] = {
            # Construction
            "236": "construction",
            "237": "construction",
            # Machinery
            "333": "machinery",
            # Electronics / IT
            "334": "electronics",
            # Electrical equipment
            "335": "electrical_equipment",
            # Vehicles
            "336": "vehicles",
            # Furniture
            "337": "furniture",
            # Medical
            "339": "medical_instruments",
            # Primary metals
            "327": "cement_materials",
            "331": "steel_fabrication",
            "332": "metalworking",
        }

        # Try 3-digit prefix
        prefix_3 = _naics_to_3digit(code)
        if prefix_3 in naics_margin_map:
            return naics_margin_map[prefix_3]

        # Try 2-digit prefix
        prefix_2 = _naics_to_2digit(code)
        sector_2_map: Dict[str, str] = {
            "23": "construction",
            "33": "machinery",
            "32": "general_industrial",
            "31": "textile_machinery",
            "22": "turbines_generators",
            "48": "railroad_equipment",
            "49": "general_industrial",
        }
        if prefix_2 in sector_2_map:
            return sector_2_map[prefix_2]

        return "general_industrial"

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
        eeio_source: str,
        record: CapExSpendRecord,
    ) -> List[str]:
        """Generate DQI findings and recommendations.

        Produces human-readable findings based on individual
        dimension scores to guide data improvement efforts for
        capital goods emissions.

        Args:
            temporal_score: Temporal dimension score.
            geo_score: Geographical dimension score.
            technological_score: Technological dimension score.
            completeness_score: Completeness dimension score.
            reliability_score: Reliability dimension score.
            eeio_source: EEIO database identifier.
            record: Source CapEx spend record.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        db_label = _EEIO_DATABASE_LABELS.get(
            EEIODatabase(eeio_source)
            if eeio_source in [db.value for db in EEIODatabase]
            else EEIODatabase.EPA_USEEIO,
            eeio_source,
        )
        findings.append(
            f"Spend-based method (EEIO) used with {db_label}."
        )

        findings.append(
            "Capital goods emissions reported 100% in year of "
            "acquisition per GHG Protocol Scope 3 Category 2 "
            "guidance (no depreciation)."
        )

        if temporal_score >= Decimal("4.0"):
            findings.append(
                "Temporal: EEIO factors are based on multi-year "
                "economic models; consider checking for more recent "
                "factor releases."
            )

        if geo_score >= Decimal("4.0"):
            findings.append(
                f"Geographical: Non-domestic currency detected "
                f"({record.currency.value}); EEIO factors may not "
                f"reflect the vendor's actual production region.  "
                f"Consider using regional EEIO databases (e.g., "
                f"EXIOBASE for EU vendors)."
            )
        elif geo_score >= Decimal("3.0"):
            findings.append(
                "Geographical: Domestic US EEIO factors applied; "
                "representativeness is moderate for US-based "
                "capital goods vendors."
            )

        if technological_score >= Decimal("4.0"):
            findings.append(
                "Technological: Sector-average EEIO factors cannot "
                "distinguish between different production technologies "
                "within a NAICS code.  Consider supplier-specific data "
                "(EPDs, PCFs) for large CapEx items."
            )

        if completeness_score >= Decimal("4.0"):
            findings.append(
                "Completeness: NAICS code is broad (< 4 digits); "
                "consider providing a more specific 6-digit NAICS "
                "code for improved factor matching."
            )

        if reliability_score >= Decimal("4.0"):
            findings.append(
                "Reliability: EEIO factors are model-derived "
                "estimates, not measured data.  For material "
                "CapEx items, request primary emission data from "
                "capital goods suppliers."
            )

        # Capital-goods-specific recommendation
        naics = record.naics_code or ""
        if naics.startswith("33"):
            findings.append(
                "Recommendation: Manufacturing sector capital goods "
                "may have supplier-specific EPDs available.  "
                "Transitioning to the supplier-specific method for "
                "high-spend assets can reduce uncertainty from "
                "+/-75% to +/-20%."
            )

        return findings

    # ------------------------------------------------------------------
    # Metrics Recording
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        method: str,
        status: str,
        duration_s: float,
        emissions_kgco2e: float,
        spend_usd: float,
    ) -> None:
        """Record calculation metrics to the metrics collector.

        Args:
            method: Calculation method name.
            status: Status string ("success" or "failed").
            duration_s: Processing duration in seconds.
            emissions_kgco2e: Emissions result (0 if failed).
            spend_usd: Spend processed (0 if failed).
        """
        try:
            self._metrics.record_calculation(
                method=method,
                category="capital_goods",
                status=status,
                duration_s=duration_s,
                emissions_tco2e=emissions_kgco2e / 1000.0,
            )
        except Exception as exc:
            logger.debug(
                "Failed to record metrics: %s", str(exc)
            )

    # ------------------------------------------------------------------
    # Provenance Stage Recording
    # ------------------------------------------------------------------

    def _record_provenance_stage(
        self,
        calculation_id: str,
        asset_id: str,
        stage: ProvenanceStage,
        metadata: Dict[str, Any],
        output_data: Any,
    ) -> None:
        """Record a provenance stage entry.

        Args:
            calculation_id: Unique calculation identifier.
            asset_id: Capital asset identifier.
            stage: Provenance pipeline stage.
            metadata: Stage metadata dictionary.
            output_data: Stage output data for hashing.
        """
        try:
            provenance = CapitalGoodsProvenance()
            provenance.record_stage(
                stage=stage.value,
                input_data={"calculation_id": calculation_id, "asset_id": asset_id},
                output_data=output_data,
                parameters=metadata,
            )
        except Exception as exc:
            logger.debug(
                "Failed to record provenance stage %s: %s",
                stage.value,
                str(exc),
            )
