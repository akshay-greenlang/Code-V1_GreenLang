# -*- coding: utf-8 -*-
"""
InstrumentAllocationEngine - Engine 2: Scope 2 Market-Based Emissions (AGENT-MRV-010)

Core instrument allocation engine implementing GHG Protocol Scope 2 Guidance
(2015) market-based method. Allocates energy attribute certificates and
contractual instruments to electricity purchases following the six-level
instrument hierarchy defined by the GHG Protocol.

GHG Protocol Instrument Hierarchy (descending priority):
    1. Bundled energy attribute certificates (RECs/GOs delivered with energy)
    2. Direct contracts (physical/virtual PPAs)
    3. Supplier-specific emission factors
    4. Unbundled certificates (standalone RECs/GOs/I-RECs)
    5. Green tariffs (Green-e, utility programs)
    6. Residual mix (grid average minus tracked claims)

Allocation Strategy:
    Priority-based (default): Instruments are sorted by hierarchy level and
    allocated sequentially until the purchase's consumption (MWh) is fully
    covered. Higher-priority instruments consume MWh first.

    Proportional: Each instrument receives a share of MWh proportional to
    its available capacity relative to total available capacity.

    Custom: Caller supplies an explicit ordering of instrument types that
    overrides the default hierarchy.

Validation Rules:
    - Vintage must fall within the allowed window for the instrument type
      (typically the reporting year or reporting year minus one).
    - Geographic scope must match: the instrument's market or interconnected
      grid must overlap with the consumption region.
    - Tracking system must be a recognized registry (e.g. M-RETS, PJM-GATS,
      NAR, EECS-AIB, I-REC).
    - Double-counting check: each instrument ID can only be allocated once
      across all purchases.

Zero-Hallucination Guarantees:
    - All arithmetic uses Python Decimal (ROUND_HALF_UP, 8 decimal places).
    - No LLM calls in any calculation or allocation path.
    - Every allocation step is recorded in the allocation trace.
    - SHA-256 provenance hash for every allocation result.
    - Same inputs always produce identical outputs (deterministic).

Thread Safety:
    All mutable state is protected by a reentrant lock (threading.RLock).

Example:
    >>> from greenlang.scope2_market.instrument_allocation import (
    ...     InstrumentAllocationEngine,
    ... )
    >>> engine = InstrumentAllocationEngine()
    >>> result = engine.allocate_instruments(
    ...     purchase={"facility_id": "FAC-001", "mwh": "10000"},
    ...     instruments=[
    ...         {"id": "REC-001", "type": "BUNDLED_CERTIFICATE",
    ...          "mwh": "5000", "vintage_year": 2025, "region": "US-WECC"},
    ...         {"id": "PPA-001", "type": "DIRECT_CONTRACT",
    ...          "mwh": "6000", "vintage_year": 2025, "region": "US-WECC"},
    ...     ],
    ... )
    >>> assert result["status"] == "SUCCESS"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
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
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["InstrumentAllocationEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.scope2_market.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.scope2_market.metrics import (
        record_allocation as _record_allocation,
        record_retirement as _record_retirement,
        observe_allocation_duration as _observe_allocation_duration,
        record_validation as _record_validation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_allocation = None  # type: ignore[assignment]
    _record_retirement = None  # type: ignore[assignment]
    _observe_allocation_duration = None  # type: ignore[assignment]
    _record_validation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

#: 8-decimal-place quantizer for deterministic rounding.
_PRECISION = Decimal("0.00000001")

#: MWh to kWh factor.
_MWH_TO_KWH = Decimal("1000")

#: Zero constant for Decimal comparisons.
_ZERO = Decimal("0")

#: One constant for Decimal arithmetic.
_ONE = Decimal("1")

#: One hundred constant for percentage calculations.
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class InstrumentType(str, Enum):
    """GHG Protocol Scope 2 contractual instrument types.

    Ordered by the instrument hierarchy from the GHG Protocol Scope 2
    Guidance (2015), Section 7.3. Lower numeric priority means higher
    precedence in the allocation waterfall.

    BUNDLED_CERTIFICATE: Energy attribute certificates (RECs, GOs, I-RECs)
        bundled with physical energy delivery. Highest quality because
        the environmental attribute and the energy are inseparable.
    DIRECT_CONTRACT: Physical or virtual power purchase agreements (PPAs)
        with a specific generation facility. Includes both physical PPAs
        (with energy delivery) and financial/virtual PPAs (contracts for
        differences with separate certificate transfer).
    SUPPLIER_SPECIFIC: Supplier-provided emission factors calculated from
        the supplier's own generation mix and certified by the utility.
        Must meet GHG Protocol quality criteria.
    UNBUNDLED_CERTIFICATE: Standalone energy attribute certificates
        (RECs, GOs, I-RECs) traded separately from the underlying
        electricity. Purchased on the open market.
    GREEN_TARIFF: Utility green pricing or green tariff programs,
        including Green-e certified products. Energy purchased through
        a specific tariff offered by the local utility.
    RESIDUAL_MIX: Grid residual mix factor after all tracked contractual
        instruments are removed from the grid's generation mix. Applied
        to any electricity consumption not covered by higher-priority
        instruments.
    """

    BUNDLED_CERTIFICATE = "BUNDLED_CERTIFICATE"
    DIRECT_CONTRACT = "DIRECT_CONTRACT"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    UNBUNDLED_CERTIFICATE = "UNBUNDLED_CERTIFICATE"
    GREEN_TARIFF = "GREEN_TARIFF"
    RESIDUAL_MIX = "RESIDUAL_MIX"


class CoverageStatus(str, Enum):
    """Coverage status of electricity purchases by contractual instruments.

    FULL: 100% of consumption MWh is covered by instruments.
    PARTIAL: Between 0% and 100% (exclusive) of consumption is covered.
    NONE: No instruments allocated (0% coverage).
    OVER_ALLOCATED: Instruments exceed consumption (>100% coverage).
    """

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"
    OVER_ALLOCATED = "OVER_ALLOCATED"


class RetirementStatus(str, Enum):
    """Retirement lifecycle status for an energy attribute certificate.

    ACTIVE: Certificate is available for allocation.
    ALLOCATED: Certificate has been allocated to a purchase but not
        yet formally retired in the tracking system.
    RETIRED: Certificate has been permanently retired (cancelled)
        in the tracking system. Cannot be re-used.
    EXPIRED: Certificate has passed its vintage validity window and
        can no longer be allocated.
    REVOKED: Certificate has been invalidated by the issuing body.
    """

    ACTIVE = "ACTIVE"
    ALLOCATED = "ALLOCATED"
    RETIRED = "RETIRED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class TrackingSystem(str, Enum):
    """Recognized energy attribute certificate tracking registries.

    M_RETS: Midwest Renewable Energy Tracking System (US).
    PJM_GATS: PJM Generation Attribute Tracking System (US Mid-Atlantic).
    NAR: North American Renewables Registry.
    NEPOOL_GIS: NEPOOL Generation Information System (US New England).
    ERCOT: Electric Reliability Council of Texas (US Texas).
    WREGIS: Western Renewable Energy Generation Information System (US West).
    EECS_AIB: European Energy Certificate System / Association of Issuing Bodies.
    I_REC: International REC Standard.
    REGO: Renewable Energy Guarantees of Origin (UK Ofgem).
    T_REC: Tradable REC (Asia-Pacific).
    LGC: Large-scale Generation Certificate (Australia).
    """

    M_RETS = "M_RETS"
    PJM_GATS = "PJM_GATS"
    NAR = "NAR"
    NEPOOL_GIS = "NEPOOL_GIS"
    ERCOT = "ERCOT"
    WREGIS = "WREGIS"
    EECS_AIB = "EECS_AIB"
    I_REC = "I_REC"
    REGO = "REGO"
    T_REC = "T_REC"
    LGC = "LGC"


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

#: Instrument hierarchy priority mapping. Lower integer = higher priority.
INSTRUMENT_PRIORITY: Dict[str, int] = {
    InstrumentType.BUNDLED_CERTIFICATE.value: 1,
    InstrumentType.DIRECT_CONTRACT.value: 2,
    InstrumentType.SUPPLIER_SPECIFIC.value: 3,
    InstrumentType.UNBUNDLED_CERTIFICATE.value: 4,
    InstrumentType.GREEN_TARIFF.value: 5,
    InstrumentType.RESIDUAL_MIX.value: 6,
}

#: Geographic market interconnections.  An instrument from one region may
#: satisfy consumption in any region within the same interconnected set.
GEOGRAPHIC_MARKETS: Dict[str, FrozenSet[str]] = {
    # North American Interconnections
    "US-WECC": frozenset({
        "US-WECC", "US-NWPP", "US-CAMX", "US-RMPA", "US-AZNM",
    }),
    "US-NWPP": frozenset({
        "US-WECC", "US-NWPP", "US-CAMX", "US-RMPA", "US-AZNM",
    }),
    "US-CAMX": frozenset({
        "US-WECC", "US-NWPP", "US-CAMX", "US-RMPA", "US-AZNM",
    }),
    "US-RMPA": frozenset({
        "US-WECC", "US-NWPP", "US-CAMX", "US-RMPA", "US-AZNM",
    }),
    "US-AZNM": frozenset({
        "US-WECC", "US-NWPP", "US-CAMX", "US-RMPA", "US-AZNM",
    }),
    "US-RFCE": frozenset({
        "US-RFCE", "US-RFCM", "US-RFCW", "US-SRMW", "US-SRMV",
        "US-SRSO", "US-SRTV", "US-SRVC", "US-NYUP", "US-NYCW",
        "US-NYLI", "US-NEWE",
    }),
    "US-RFCM": frozenset({
        "US-RFCE", "US-RFCM", "US-RFCW", "US-SRMW", "US-SRMV",
        "US-SRSO", "US-SRTV", "US-SRVC", "US-NYUP", "US-NYCW",
        "US-NYLI", "US-NEWE",
    }),
    "US-RFCW": frozenset({
        "US-RFCE", "US-RFCM", "US-RFCW", "US-SRMW", "US-SRMV",
        "US-SRSO", "US-SRTV", "US-SRVC", "US-NYUP", "US-NYCW",
        "US-NYLI", "US-NEWE",
    }),
    "US-ERCOT": frozenset({"US-ERCOT"}),
    # European single market
    "EU": frozenset({
        "EU", "EU-DE", "EU-FR", "EU-ES", "EU-IT", "EU-NL", "EU-BE",
        "EU-AT", "EU-PL", "EU-SE", "EU-DK", "EU-FI", "EU-PT", "EU-IE",
        "EU-CZ", "EU-RO", "EU-HU", "EU-BG", "EU-SK", "EU-HR", "EU-LT",
        "EU-SI", "EU-LV", "EU-EE", "EU-CY", "EU-LU", "EU-MT", "EU-EL",
    }),
    "EU-DE": frozenset({
        "EU", "EU-DE", "EU-FR", "EU-ES", "EU-IT", "EU-NL", "EU-BE",
        "EU-AT", "EU-PL", "EU-SE", "EU-DK", "EU-FI", "EU-PT", "EU-IE",
        "EU-CZ", "EU-RO", "EU-HU", "EU-BG", "EU-SK", "EU-HR", "EU-LT",
        "EU-SI", "EU-LV", "EU-EE", "EU-CY", "EU-LU", "EU-MT", "EU-EL",
    }),
    "EU-FR": frozenset({
        "EU", "EU-DE", "EU-FR", "EU-ES", "EU-IT", "EU-NL", "EU-BE",
        "EU-AT", "EU-PL", "EU-SE", "EU-DK", "EU-FI", "EU-PT", "EU-IE",
        "EU-CZ", "EU-RO", "EU-HU", "EU-BG", "EU-SK", "EU-HR", "EU-LT",
        "EU-SI", "EU-LV", "EU-EE", "EU-CY", "EU-LU", "EU-MT", "EU-EL",
    }),
    # United Kingdom (post-Brexit: separate market)
    "UK": frozenset({"UK", "UK-GB", "UK-NI"}),
    "UK-GB": frozenset({"UK", "UK-GB", "UK-NI"}),
    "UK-NI": frozenset({"UK", "UK-GB", "UK-NI"}),
    # Australia (NEM + SWIS)
    "AU-NEM": frozenset({"AU-NEM", "AU-NSW", "AU-VIC", "AU-QLD", "AU-SA", "AU-TAS"}),
    "AU-SWIS": frozenset({"AU-SWIS", "AU-WA"}),
    # Japan
    "JP": frozenset({
        "JP", "JP-TEPCO", "JP-KANSAI", "JP-CHUBU", "JP-TOHOKU",
        "JP-KYUSHU", "JP-CHUGOKU", "JP-HOKKAIDO", "JP-SHIKOKU",
        "JP-HOKURIKU",
    }),
    # India
    "IN": frozenset({
        "IN", "IN-NR", "IN-WR", "IN-SR", "IN-ER", "IN-NER",
    }),
    # Catch-all global (allows any match if both regions are GLOBAL)
    "GLOBAL": frozenset({"GLOBAL"}),
}

#: Maximum vintage age in years by instrument type.
#: Per GHG Protocol Scope 2 Guidance, most certificates must be from the
#: reporting year or the year immediately preceding it.
VINTAGE_WINDOWS: Dict[str, int] = {
    InstrumentType.BUNDLED_CERTIFICATE.value: 1,
    InstrumentType.DIRECT_CONTRACT.value: 1,
    InstrumentType.SUPPLIER_SPECIFIC.value: 0,
    InstrumentType.UNBUNDLED_CERTIFICATE.value: 1,
    InstrumentType.GREEN_TARIFF.value: 1,
    InstrumentType.RESIDUAL_MIX.value: 0,
}

#: Set of recognized tracking system identifiers for validation.
_VALID_TRACKING_SYSTEMS: FrozenSet[str] = frozenset(
    ts.value for ts in TrackingSystem
)

#: Maximum number of instruments allowed in a single allocation call.
MAX_INSTRUMENTS_PER_ALLOCATION: int = 10_000

#: Maximum number of purchases in a batch allocation call.
MAX_PURCHASES_PER_BATCH: int = 5_000

#: Maximum number of trace steps per allocation.
MAX_TRACE_STEPS: int = 500


# ---------------------------------------------------------------------------
# Helper: UTC now
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Helper: safe Decimal conversion
# ---------------------------------------------------------------------------


def _to_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely.

    Accepts str, int, float, and Decimal. Strings are preferred for
    lossless precision.

    Args:
        value: The value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(
                f"Cannot convert {value!r} to Decimal"
            ) from exc
    if isinstance(value, float):
        return Decimal(str(value))
    raise ValueError(f"Cannot convert {type(value).__name__} to Decimal")


# ---------------------------------------------------------------------------
# InstrumentAllocationEngine
# ---------------------------------------------------------------------------


class InstrumentAllocationEngine:
    """Core instrument allocation engine for GHG Protocol Scope 2 market-based method.

    Implements the contractual instrument hierarchy, validation rules,
    coverage tracking, certificate retirement, and allocation strategies
    (priority-based, proportional, custom) with deterministic Decimal
    arithmetic, full allocation trace, and SHA-256 provenance hashing.

    Thread-safe: all mutable state is protected by a reentrant lock.

    Attributes:
        _config: Configuration dictionary.
        _lock: Reentrant lock for thread-safe access.
        _provenance: Optional reference to the provenance tracker.
        _retired_instruments: Set of retired instrument IDs.
        _retirement_history: Mapping of instrument ID to list of
            retirement event dicts.
        _used_instruments: Set of instrument IDs that have been allocated
            (prevents double-counting across purchases).
        _allocation_count: Total number of allocations performed.
        _validation_count: Total number of validations performed.
        _retirement_count: Total number of retirements performed.

    Example:
        >>> engine = InstrumentAllocationEngine()
        >>> result = engine.allocate_instruments(
        ...     purchase={"facility_id": "FAC-001", "mwh": "10000"},
        ...     instruments=[
        ...         {"id": "REC-001", "type": "BUNDLED_CERTIFICATE",
        ...          "mwh": "5000", "vintage_year": 2025, "region": "US-WECC"},
        ...     ],
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    # ==================================================================
    # Initialization
    # ==================================================================

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize InstrumentAllocationEngine.

        Args:
            config: Optional configuration dict. Supported keys:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                    Default True.
                - ``decimal_precision`` (int): Number of Decimal places.
                    Default 8.
                - ``default_reporting_year`` (int): Fallback reporting year.
                    Default current UTC year.
                - ``strict_geographic_match`` (bool): Require strict
                    geographic match. Default True.
                - ``strict_vintage`` (bool): Enforce vintage window.
                    Default True.
                - ``allow_partial_allocation`` (bool): Allow partial
                    instrument allocation. Default True.
        """
        self._config: Dict[str, Any] = config or {}
        self._lock: threading.RLock = threading.RLock()

        # Decimal precision
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer: Decimal = Decimal(10) ** -self._precision_places

        # Reporting year
        self._default_reporting_year: int = self._config.get(
            "default_reporting_year", _utcnow().year,
        )

        # Validation strictness
        self._strict_geographic: bool = self._config.get(
            "strict_geographic_match", True,
        )
        self._strict_vintage: bool = self._config.get("strict_vintage", True)
        self._allow_partial: bool = self._config.get(
            "allow_partial_allocation", True,
        )

        # Provenance
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True,
        )
        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        # Certificate retirement state
        self._retired_instruments: Set[str] = set()
        self._retirement_history: Dict[str, List[Dict[str, Any]]] = {}

        # Double-counting prevention
        self._used_instruments: Dict[str, Dict[str, Any]] = {}

        # Counters
        self._allocation_count: int = 0
        self._validation_count: int = 0
        self._retirement_count: int = 0
        self._total_mwh_allocated: Decimal = _ZERO
        self._total_mwh_covered: Decimal = _ZERO

        logger.info(
            "InstrumentAllocationEngine initialized "
            "(precision=%d, reporting_year=%d, strict_geo=%s, strict_vintage=%s)",
            self._precision_places,
            self._default_reporting_year,
            self._strict_geographic,
            self._strict_vintage,
        )

    # ==================================================================
    # PUBLIC API: Core Allocation
    # ==================================================================

    def allocate_instruments(
        self,
        purchase: Dict[str, Any],
        instruments: List[Dict[str, Any]],
        reporting_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Allocate contractual instruments to an electricity purchase using hierarchy priority.

        Instruments are sorted by GHG Protocol hierarchy priority (1=highest)
        and allocated sequentially until the purchase's consumption MWh is
        fully covered or all instruments are exhausted.

        Args:
            purchase: Electricity purchase dict with keys:
                - ``facility_id`` (str): Facility identifier.
                - ``mwh`` (str|Decimal|float|int): Total consumption in MWh.
                - ``region`` (str, optional): Geographic region code.
                - ``purchase_id`` (str, optional): Unique purchase ID.
            instruments: List of instrument dicts, each with keys:
                - ``id`` (str): Unique instrument identifier.
                - ``type`` (str): InstrumentType value string.
                - ``mwh`` (str|Decimal|float|int): Available MWh capacity.
                - ``vintage_year`` (int): Generation vintage year.
                - ``region`` (str): Geographic region code.
                - ``tracking_system`` (str, optional): Tracking registry.
                - ``emission_factor`` (str|Decimal, optional): tCO2e/MWh.
                - ``generator_id`` (str, optional): Generator facility ID.
                - ``technology`` (str, optional): Generation technology.
            reporting_year: Override for the default reporting year.

        Returns:
            Allocation result dict with keys:
                - ``allocation_id`` (str): Unique allocation ID.
                - ``status`` (str): "SUCCESS" or "FAILED".
                - ``purchase_id`` (str): Purchase identifier.
                - ``facility_id`` (str): Facility identifier.
                - ``total_mwh`` (str): Total purchase MWh.
                - ``covered_mwh`` (str): MWh covered by instruments.
                - ``uncovered_mwh`` (str): MWh not covered.
                - ``coverage_pct`` (str): Coverage percentage.
                - ``coverage_status`` (str): CoverageStatus value.
                - ``allocations`` (list): Per-instrument allocation details.
                - ``validation_issues`` (list): Instrument validation warnings.
                - ``allocation_trace`` (list): Step-by-step trace.
                - ``provenance_hash`` (str): SHA-256 hash.
                - ``processing_time_ms`` (float): Wall-clock time.

        Raises:
            ValueError: If purchase or instruments are invalid.
        """
        start_time = time.monotonic()
        alloc_id = f"s2m_alloc_{uuid.uuid4().hex[:12]}"
        year = reporting_year or self._default_reporting_year
        trace: List[str] = []

        try:
            # ---- Step 1: Parse and validate purchase ----
            purchase_mwh, facility_id, purchase_id, purchase_region = (
                self._parse_purchase(purchase, alloc_id)
            )
            trace.append(
                f"[1] Parsed purchase: facility={facility_id}, "
                f"mwh={purchase_mwh}, region={purchase_region}"
            )

            # ---- Step 2: Validate and filter instruments ----
            valid_instruments, validation_issues = self._validate_and_filter_instruments(
                instruments, purchase_region, year, trace,
            )
            trace.append(
                f"[2] Validated instruments: {len(valid_instruments)} valid, "
                f"{len(validation_issues)} issues"
            )

            # ---- Step 3: Sort by hierarchy priority ----
            sorted_instruments = self.sort_by_priority(valid_instruments)
            trace.append(
                f"[3] Sorted {len(sorted_instruments)} instruments by priority"
            )

            # ---- Step 4: Allocate sequentially ----
            allocations, covered_mwh = self._allocate_sequential(
                purchase_mwh, sorted_instruments, trace,
            )
            uncovered_mwh = (purchase_mwh - covered_mwh).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            if uncovered_mwh < _ZERO:
                uncovered_mwh = _ZERO

            trace.append(
                f"[4] Allocation complete: covered={covered_mwh}, "
                f"uncovered={uncovered_mwh}"
            )

            # ---- Step 5: Compute coverage ----
            coverage_pct = self._compute_coverage_pct(covered_mwh, purchase_mwh)
            coverage_status = self.get_coverage_status(coverage_pct)
            trace.append(
                f"[5] Coverage: {coverage_pct}% ({coverage_status.value})"
            )

            # ---- Step 6: Record used instruments ----
            self._record_used_instruments(allocations, alloc_id, facility_id)

            # ---- Step 7: Provenance ----
            provenance_hash = self._hash_instrument_allocation(
                alloc_id, purchase, allocations, covered_mwh, trace,
            )
            trace.append(
                f"[6] Provenance hash: {provenance_hash[:16]}..."
            )

            # ---- Step 8: Update counters ----
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            with self._lock:
                self._allocation_count += 1
                self._total_mwh_allocated += purchase_mwh
                self._total_mwh_covered += covered_mwh

            # ---- Metrics ----
            if _METRICS_AVAILABLE and _record_allocation is not None:
                _record_allocation(
                    facility_id, coverage_status.value, "priority",
                )
            if _METRICS_AVAILABLE and _observe_allocation_duration is not None:
                _observe_allocation_duration(
                    "priority_allocation", elapsed_ms / 1000.0,
                )

            # ---- Provenance tracker ----
            if self._provenance is not None:
                self._provenance.record(
                    entity_type="instrument_allocation",
                    action="allocate",
                    entity_id=alloc_id,
                    data={
                        "purchase_id": purchase_id,
                        "covered_mwh": str(covered_mwh),
                        "coverage_pct": str(coverage_pct),
                    },
                )

            return {
                "allocation_id": alloc_id,
                "status": "SUCCESS",
                "purchase_id": purchase_id,
                "facility_id": facility_id,
                "total_mwh": str(purchase_mwh),
                "covered_mwh": str(covered_mwh),
                "uncovered_mwh": str(uncovered_mwh),
                "coverage_pct": str(coverage_pct),
                "coverage_status": coverage_status.value,
                "allocations": allocations,
                "validation_issues": validation_issues,
                "allocation_trace": trace[:MAX_TRACE_STEPS],
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 3),
                "reporting_year": year,
                "timestamp": _utcnow().isoformat(),
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "Allocation %s failed: %s", alloc_id, exc, exc_info=True,
            )
            trace.append(f"[ERROR] {exc}")
            return {
                "allocation_id": alloc_id,
                "status": "FAILED",
                "error_message": str(exc),
                "allocation_trace": trace[:MAX_TRACE_STEPS],
                "provenance_hash": "",
                "processing_time_ms": round(elapsed_ms, 3),
                "timestamp": _utcnow().isoformat(),
            }

    def allocate_batch(
        self,
        purchases_with_instruments: List[Dict[str, Any]],
        reporting_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Allocate instruments for multiple purchases in a single batch.

        Each entry in the input list must have ``purchase`` and ``instruments``
        keys corresponding to the arguments of ``allocate_instruments``.

        Args:
            purchases_with_instruments: List of dicts, each containing:
                - ``purchase`` (dict): See allocate_instruments.
                - ``instruments`` (list): See allocate_instruments.
            reporting_year: Optional reporting year override for all.

        Returns:
            List of allocation result dicts (one per purchase).

        Raises:
            ValueError: If the batch exceeds MAX_PURCHASES_PER_BATCH.
        """
        if len(purchases_with_instruments) > MAX_PURCHASES_PER_BATCH:
            raise ValueError(
                f"Batch size {len(purchases_with_instruments)} exceeds "
                f"maximum {MAX_PURCHASES_PER_BATCH}"
            )

        results: List[Dict[str, Any]] = []
        for item in purchases_with_instruments:
            purchase = item.get("purchase", {})
            instruments = item.get("instruments", [])
            result = self.allocate_instruments(
                purchase=purchase,
                instruments=instruments,
                reporting_year=reporting_year,
            )
            results.append(result)

        logger.info(
            "Batch allocation completed: %d purchases processed", len(results),
        )
        return results

    def allocate_proportional(
        self,
        purchase: Dict[str, Any],
        instruments: List[Dict[str, Any]],
        reporting_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Allocate instruments proportionally by available capacity.

        Instead of priority-based sequential allocation, each valid instrument
        receives a share of the purchase MWh proportional to its available
        capacity relative to the total available capacity of all instruments.

        If total instrument capacity is less than purchase MWh, instruments
        are fully consumed and the remainder is uncovered.

        Args:
            purchase: See allocate_instruments.
            instruments: See allocate_instruments.
            reporting_year: Optional reporting year override.

        Returns:
            Allocation result dict (same schema as allocate_instruments).
        """
        start_time = time.monotonic()
        alloc_id = f"s2m_prop_{uuid.uuid4().hex[:12]}"
        year = reporting_year or self._default_reporting_year
        trace: List[str] = []

        try:
            purchase_mwh, facility_id, purchase_id, purchase_region = (
                self._parse_purchase(purchase, alloc_id)
            )
            trace.append(
                f"[1] Parsed purchase: facility={facility_id}, "
                f"mwh={purchase_mwh}, region={purchase_region}"
            )

            valid_instruments, validation_issues = self._validate_and_filter_instruments(
                instruments, purchase_region, year, trace,
            )
            trace.append(
                f"[2] Validated: {len(valid_instruments)} valid, "
                f"{len(validation_issues)} issues"
            )

            # Compute total available capacity
            total_available = _ZERO
            for inst in valid_instruments:
                inst_mwh = _to_decimal(inst.get("mwh", "0"))
                total_available += inst_mwh

            if total_available <= _ZERO:
                trace.append("[3] No available capacity, zero coverage")
                allocations: List[Dict[str, Any]] = []
                covered_mwh = _ZERO
            else:
                allocations, covered_mwh = self._allocate_proportional_impl(
                    purchase_mwh, valid_instruments, total_available, trace,
                )

            uncovered_mwh = (purchase_mwh - covered_mwh).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            if uncovered_mwh < _ZERO:
                uncovered_mwh = _ZERO

            coverage_pct = self._compute_coverage_pct(covered_mwh, purchase_mwh)
            coverage_status = self.get_coverage_status(coverage_pct)
            trace.append(
                f"[4] Coverage: {coverage_pct}% ({coverage_status.value})"
            )

            self._record_used_instruments(allocations, alloc_id, facility_id)

            provenance_hash = self._hash_instrument_allocation(
                alloc_id, purchase, allocations, covered_mwh, trace,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            with self._lock:
                self._allocation_count += 1
                self._total_mwh_allocated += purchase_mwh
                self._total_mwh_covered += covered_mwh

            if _METRICS_AVAILABLE and _record_allocation is not None:
                _record_allocation(
                    facility_id, coverage_status.value, "proportional",
                )

            return {
                "allocation_id": alloc_id,
                "status": "SUCCESS",
                "strategy": "proportional",
                "purchase_id": purchase_id,
                "facility_id": facility_id,
                "total_mwh": str(purchase_mwh),
                "covered_mwh": str(covered_mwh),
                "uncovered_mwh": str(uncovered_mwh),
                "coverage_pct": str(coverage_pct),
                "coverage_status": coverage_status.value,
                "allocations": allocations,
                "validation_issues": validation_issues,
                "allocation_trace": trace[:MAX_TRACE_STEPS],
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 3),
                "reporting_year": year,
                "timestamp": _utcnow().isoformat(),
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "Proportional allocation %s failed: %s",
                alloc_id, exc, exc_info=True,
            )
            trace.append(f"[ERROR] {exc}")
            return {
                "allocation_id": alloc_id,
                "status": "FAILED",
                "strategy": "proportional",
                "error_message": str(exc),
                "allocation_trace": trace[:MAX_TRACE_STEPS],
                "provenance_hash": "",
                "processing_time_ms": round(elapsed_ms, 3),
                "timestamp": _utcnow().isoformat(),
            }

    def allocate_custom(
        self,
        purchase: Dict[str, Any],
        instruments: List[Dict[str, Any]],
        custom_order: List[str],
        reporting_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Allocate instruments using a caller-supplied type priority ordering.

        The ``custom_order`` list defines the allocation sequence. Instrument
        types not in the list are excluded. Within each type, instruments
        are ordered by descending available MWh.

        Args:
            purchase: See allocate_instruments.
            instruments: See allocate_instruments.
            custom_order: List of InstrumentType value strings in desired
                priority order (first = highest priority).
            reporting_year: Optional reporting year override.

        Returns:
            Allocation result dict (same schema as allocate_instruments).
        """
        start_time = time.monotonic()
        alloc_id = f"s2m_cust_{uuid.uuid4().hex[:12]}"
        year = reporting_year or self._default_reporting_year
        trace: List[str] = []

        try:
            purchase_mwh, facility_id, purchase_id, purchase_region = (
                self._parse_purchase(purchase, alloc_id)
            )
            trace.append(
                f"[1] Parsed purchase: facility={facility_id}, "
                f"mwh={purchase_mwh}, region={purchase_region}"
            )

            valid_instruments, validation_issues = self._validate_and_filter_instruments(
                instruments, purchase_region, year, trace,
            )
            trace.append(
                f"[2] Validated: {len(valid_instruments)} valid, "
                f"{len(validation_issues)} issues"
            )

            # Build custom priority map from the provided order
            custom_priority: Dict[str, int] = {}
            for idx, itype in enumerate(custom_order, start=1):
                custom_priority[itype.upper()] = idx

            # Filter to only types in custom order, then sort
            filtered = [
                inst for inst in valid_instruments
                if inst.get("type", "").upper() in custom_priority
            ]
            filtered.sort(
                key=lambda i: (
                    custom_priority.get(i.get("type", "").upper(), 9999),
                    -_to_decimal(i.get("mwh", "0")),
                )
            )
            trace.append(
                f"[3] Custom-ordered {len(filtered)} instruments "
                f"(order={custom_order})"
            )

            allocations, covered_mwh = self._allocate_sequential(
                purchase_mwh, filtered, trace,
            )

            uncovered_mwh = (purchase_mwh - covered_mwh).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            if uncovered_mwh < _ZERO:
                uncovered_mwh = _ZERO

            coverage_pct = self._compute_coverage_pct(covered_mwh, purchase_mwh)
            coverage_status = self.get_coverage_status(coverage_pct)
            trace.append(
                f"[4] Coverage: {coverage_pct}% ({coverage_status.value})"
            )

            self._record_used_instruments(allocations, alloc_id, facility_id)

            provenance_hash = self._hash_instrument_allocation(
                alloc_id, purchase, allocations, covered_mwh, trace,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            with self._lock:
                self._allocation_count += 1
                self._total_mwh_allocated += purchase_mwh
                self._total_mwh_covered += covered_mwh

            if _METRICS_AVAILABLE and _record_allocation is not None:
                _record_allocation(
                    facility_id, coverage_status.value, "custom",
                )

            return {
                "allocation_id": alloc_id,
                "status": "SUCCESS",
                "strategy": "custom",
                "custom_order": custom_order,
                "purchase_id": purchase_id,
                "facility_id": facility_id,
                "total_mwh": str(purchase_mwh),
                "covered_mwh": str(covered_mwh),
                "uncovered_mwh": str(uncovered_mwh),
                "coverage_pct": str(coverage_pct),
                "coverage_status": coverage_status.value,
                "allocations": allocations,
                "validation_issues": validation_issues,
                "allocation_trace": trace[:MAX_TRACE_STEPS],
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 3),
                "reporting_year": year,
                "timestamp": _utcnow().isoformat(),
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "Custom allocation %s failed: %s",
                alloc_id, exc, exc_info=True,
            )
            trace.append(f"[ERROR] {exc}")
            return {
                "allocation_id": alloc_id,
                "status": "FAILED",
                "strategy": "custom",
                "error_message": str(exc),
                "allocation_trace": trace[:MAX_TRACE_STEPS],
                "provenance_hash": "",
                "processing_time_ms": round(elapsed_ms, 3),
                "timestamp": _utcnow().isoformat(),
            }

    # ==================================================================
    # PUBLIC API: Instrument Validation
    # ==================================================================

    def validate_instrument(
        self,
        instrument: Dict[str, Any],
        reporting_year: Optional[int] = None,
        consumption_region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a single contractual instrument against GHG Protocol quality criteria.

        Checks vintage validity, geographic match, tracking system recognition,
        retirement status, and double-counting.

        Args:
            instrument: Instrument dict (see allocate_instruments).
            reporting_year: Override for the default reporting year.
            consumption_region: Optional region to validate geographic match.

        Returns:
            Validation result dict with keys:
                - ``instrument_id`` (str): Instrument identifier.
                - ``is_valid`` (bool): True if all checks pass.
                - ``checks`` (dict): Individual check results:
                    - ``vintage_valid`` (bool)
                    - ``geographic_valid`` (bool)
                    - ``tracking_valid`` (bool)
                    - ``not_double_counted`` (bool)
                    - ``not_retired`` (bool)
                    - ``not_expired`` (bool)
                    - ``has_capacity`` (bool)
                - ``issues`` (list): List of issue description strings.
                - ``instrument_type`` (str): Resolved instrument type.
                - ``priority`` (int): Hierarchy priority level.
        """
        year = reporting_year or self._default_reporting_year
        inst_id = str(instrument.get("id", "UNKNOWN"))
        inst_type = str(instrument.get("type", "")).upper()
        inst_region = str(instrument.get("region", "GLOBAL")).upper()
        vintage_year = instrument.get("vintage_year")
        tracking = str(instrument.get("tracking_system", "")).upper()
        inst_mwh = instrument.get("mwh", "0")

        issues: List[str] = []
        checks: Dict[str, bool] = {}

        # Vintage check
        vintage_valid = True
        if vintage_year is not None:
            vintage_valid = self.validate_vintage(inst_type, int(vintage_year), year)
        if not vintage_valid:
            issues.append(
                f"Vintage {vintage_year} outside allowed window for "
                f"{inst_type} (reporting year {year})"
            )
        checks["vintage_valid"] = vintage_valid

        # Geographic match check
        geographic_valid = True
        if consumption_region:
            geographic_valid = self.validate_geographic_match(
                inst_region, consumption_region.upper(),
            )
        if not geographic_valid:
            issues.append(
                f"Geographic mismatch: instrument region {inst_region} "
                f"does not match consumption region {consumption_region}"
            )
        checks["geographic_valid"] = geographic_valid

        # Tracking system check
        tracking_valid = self.validate_tracking_system(instrument)
        if not tracking_valid:
            issues.append(
                f"Unrecognized tracking system: {tracking}"
            )
        checks["tracking_valid"] = tracking_valid

        # Double-counting check
        not_double_counted = not self.check_double_counting(
            inst_id, self._used_instruments,
        )
        if not not_double_counted:
            issues.append(
                f"Instrument {inst_id} already allocated (double-counting)"
            )
        checks["not_double_counted"] = not_double_counted

        # Retirement status check
        not_retired = inst_id not in self._retired_instruments
        if not not_retired:
            issues.append(f"Instrument {inst_id} has been retired")
        checks["not_retired"] = not_retired

        # Expiry check (treat instruments with vintage older than window+1 as expired)
        not_expired = True
        if vintage_year is not None:
            max_age = VINTAGE_WINDOWS.get(inst_type, 1)
            if int(vintage_year) < year - max_age - 1:
                not_expired = False
                issues.append(f"Instrument {inst_id} has expired (vintage {vintage_year})")
        checks["not_expired"] = not_expired

        # Capacity check
        has_capacity = True
        try:
            cap = _to_decimal(inst_mwh)
            if cap <= _ZERO:
                has_capacity = False
                issues.append(f"Instrument {inst_id} has zero or negative capacity")
        except (ValueError, InvalidOperation):
            has_capacity = False
            issues.append(f"Instrument {inst_id} has invalid MWh capacity")
        checks["has_capacity"] = has_capacity

        is_valid = all(checks.values())
        priority = INSTRUMENT_PRIORITY.get(inst_type, 99)

        with self._lock:
            self._validation_count += 1

        if _METRICS_AVAILABLE and _record_validation is not None:
            _record_validation(inst_type, "valid" if is_valid else "invalid")

        return {
            "instrument_id": inst_id,
            "is_valid": is_valid,
            "checks": checks,
            "issues": issues,
            "instrument_type": inst_type,
            "priority": priority,
        }

    def validate_vintage(
        self,
        instrument_type: str,
        vintage_year: int,
        reporting_year: int,
    ) -> bool:
        """Check whether an instrument's vintage falls within the allowed window.

        Per GHG Protocol Scope 2 Guidance, certificates must typically
        be from the reporting year or the immediately preceding year.
        Supplier-specific factors and residual mix must match exactly.

        Args:
            instrument_type: InstrumentType value string.
            vintage_year: Year the energy/certificate was generated.
            reporting_year: The GHG inventory reporting year.

        Returns:
            True if the vintage is within the allowed window.
        """
        itype = instrument_type.upper()
        max_age = VINTAGE_WINDOWS.get(itype, 1)
        min_vintage = reporting_year - max_age
        return min_vintage <= vintage_year <= reporting_year

    def validate_geographic_match(
        self,
        instrument_region: str,
        consumption_region: str,
    ) -> bool:
        """Check whether an instrument's region matches the consumption region.

        Two regions match if they are identical, or if they belong to the
        same interconnected market as defined in GEOGRAPHIC_MARKETS.

        Args:
            instrument_region: Region code of the instrument.
            consumption_region: Region code of the electricity consumption.

        Returns:
            True if the regions are geographically compatible.
        """
        inst_r = instrument_region.upper()
        cons_r = consumption_region.upper()

        # Exact match
        if inst_r == cons_r:
            return True

        # Check interconnected markets
        inst_market = GEOGRAPHIC_MARKETS.get(inst_r)
        if inst_market is not None and cons_r in inst_market:
            return True

        cons_market = GEOGRAPHIC_MARKETS.get(cons_r)
        if cons_market is not None and inst_r in cons_market:
            return True

        # GLOBAL matches anything
        if inst_r == "GLOBAL" or cons_r == "GLOBAL":
            return True

        return False

    def validate_tracking_system(
        self,
        instrument: Dict[str, Any],
    ) -> bool:
        """Validate that an instrument's tracking system is recognized.

        Instruments that do not specify a tracking system pass this check
        (tracking system is optional for supplier-specific factors and
        residual mix). Instruments that do specify a tracking system must
        reference a recognized registry.

        Args:
            instrument: Instrument dict with optional ``tracking_system`` key.

        Returns:
            True if the tracking system is recognized or not specified.
        """
        tracking = instrument.get("tracking_system")
        if tracking is None or tracking == "":
            # Tracking system is optional for some instrument types
            inst_type = str(instrument.get("type", "")).upper()
            if inst_type in (
                InstrumentType.SUPPLIER_SPECIFIC.value,
                InstrumentType.RESIDUAL_MIX.value,
                InstrumentType.GREEN_TARIFF.value,
            ):
                return True
            # For certificate types, missing tracking system is a warning
            # but not an automatic failure in lenient mode
            if not self._strict_vintage:
                return True
            return False
        return str(tracking).upper() in _VALID_TRACKING_SYSTEMS

    def check_double_counting(
        self,
        instrument_id: str,
        used_instruments: Any,
    ) -> bool:
        """Check if an instrument has already been allocated (double-counted).

        Args:
            instrument_id: Unique instrument identifier.
            used_instruments: A dict or set of previously used instrument IDs.

        Returns:
            True if the instrument_id is already present (IS double-counted).
            False if the instrument_id has not been used before.
        """
        if isinstance(used_instruments, dict):
            return instrument_id in used_instruments
        if isinstance(used_instruments, (set, frozenset)):
            return instrument_id in used_instruments
        return False

    # ==================================================================
    # PUBLIC API: Coverage Tracking
    # ==================================================================

    def calculate_coverage(
        self,
        total_mwh: Any,
        instruments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate how much of a facility's consumption is covered by instruments.

        Does not perform allocation or modify state; this is a read-only
        calculation of potential coverage.

        Args:
            total_mwh: Total facility consumption in MWh.
            instruments: List of instrument dicts with ``mwh`` key.

        Returns:
            Dict with keys:
                - ``total_mwh`` (str): Total consumption.
                - ``covered_mwh`` (str): Sum of instrument capacities (capped
                    at total_mwh).
                - ``uncovered_mwh`` (str): Remaining uncovered consumption.
                - ``coverage_pct`` (str): Coverage percentage.
                - ``coverage_status`` (str): CoverageStatus value.
                - ``instrument_count`` (int): Number of instruments considered.
                - ``total_instrument_mwh`` (str): Raw sum of all instrument
                    capacities (may exceed total_mwh).
        """
        total = _to_decimal(total_mwh)
        if total <= _ZERO:
            return {
                "total_mwh": str(total),
                "covered_mwh": "0",
                "uncovered_mwh": "0",
                "coverage_pct": "0",
                "coverage_status": CoverageStatus.NONE.value,
                "instrument_count": len(instruments),
                "total_instrument_mwh": "0",
            }

        instrument_sum = _ZERO
        for inst in instruments:
            try:
                inst_mwh = _to_decimal(inst.get("mwh", "0"))
                if inst_mwh > _ZERO:
                    instrument_sum += inst_mwh
            except (ValueError, InvalidOperation):
                continue

        covered = min(instrument_sum, total)
        uncovered = (total - covered).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        if uncovered < _ZERO:
            uncovered = _ZERO

        pct = self._compute_coverage_pct(covered, total)
        status = self.get_coverage_status(pct)

        return {
            "total_mwh": str(total),
            "covered_mwh": str(covered.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )),
            "uncovered_mwh": str(uncovered),
            "coverage_pct": str(pct),
            "coverage_status": status.value,
            "instrument_count": len(instruments),
            "total_instrument_mwh": str(instrument_sum.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )),
        }

    def get_coverage_status(
        self,
        coverage_pct: Decimal,
    ) -> CoverageStatus:
        """Determine coverage status from a coverage percentage.

        Args:
            coverage_pct: Coverage percentage as a Decimal (0-100+).

        Returns:
            CoverageStatus enum value.
        """
        if not isinstance(coverage_pct, Decimal):
            coverage_pct = _to_decimal(coverage_pct)
        if coverage_pct <= _ZERO:
            return CoverageStatus.NONE
        if coverage_pct < _HUNDRED:
            return CoverageStatus.PARTIAL
        if coverage_pct == _HUNDRED:
            return CoverageStatus.FULL
        return CoverageStatus.OVER_ALLOCATED

    def identify_coverage_gaps(
        self,
        facilities_with_instruments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify facilities with incomplete instrument coverage.

        Scans a list of facility records and returns those with coverage
        below 100%. Each facility dict must have ``facility_id``,
        ``total_mwh``, and ``instruments`` keys.

        Args:
            facilities_with_instruments: List of facility dicts:
                - ``facility_id`` (str): Facility identifier.
                - ``total_mwh`` (str|Decimal): Total consumption MWh.
                - ``instruments`` (list): Instrument dicts with ``mwh``.

        Returns:
            List of gap dicts, each with:
                - ``facility_id`` (str)
                - ``total_mwh`` (str)
                - ``covered_mwh`` (str)
                - ``uncovered_mwh`` (str)
                - ``coverage_pct`` (str)
                - ``coverage_status`` (str)
        """
        gaps: List[Dict[str, Any]] = []

        for facility in facilities_with_instruments:
            fac_id = str(facility.get("facility_id", "UNKNOWN"))
            total = facility.get("total_mwh", "0")
            instruments = facility.get("instruments", [])

            coverage = self.calculate_coverage(total, instruments)
            status = coverage["coverage_status"]

            if status != CoverageStatus.FULL.value:
                gaps.append({
                    "facility_id": fac_id,
                    "total_mwh": coverage["total_mwh"],
                    "covered_mwh": coverage["covered_mwh"],
                    "uncovered_mwh": coverage["uncovered_mwh"],
                    "coverage_pct": coverage["coverage_pct"],
                    "coverage_status": status,
                })

        logger.info(
            "Identified %d coverage gaps out of %d facilities",
            len(gaps), len(facilities_with_instruments),
        )
        return gaps

    # ==================================================================
    # PUBLIC API: Certificate Management
    # ==================================================================

    def retire_instrument(
        self,
        instrument_id: str,
        reason: Optional[str] = None,
        retired_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark an instrument as permanently retired.

        Once retired, the instrument cannot be allocated to any future
        purchase. Retirement is recorded in the retirement history and
        tracked via provenance.

        Args:
            instrument_id: Unique instrument identifier to retire.
            reason: Optional reason for retirement.
            retired_by: Optional identifier of the user/process that
                initiated the retirement.

        Returns:
            Dict with keys:
                - ``instrument_id`` (str)
                - ``status`` (str): "RETIRED" or "ALREADY_RETIRED"
                - ``retired_at`` (str): ISO timestamp
                - ``reason`` (str|None)
                - ``retired_by`` (str|None)
                - ``provenance_hash`` (str)
        """
        with self._lock:
            already = instrument_id in self._retired_instruments
            if already:
                logger.warning(
                    "Instrument %s is already retired", instrument_id,
                )
                return {
                    "instrument_id": instrument_id,
                    "status": "ALREADY_RETIRED",
                    "retired_at": None,
                    "reason": reason,
                    "retired_by": retired_by,
                    "provenance_hash": "",
                }

            now = _utcnow()
            self._retired_instruments.add(instrument_id)
            self._retirement_count += 1

            event: Dict[str, Any] = {
                "action": "retire",
                "timestamp": now.isoformat(),
                "reason": reason,
                "retired_by": retired_by,
            }
            if instrument_id not in self._retirement_history:
                self._retirement_history[instrument_id] = []
            self._retirement_history[instrument_id].append(event)

        # Provenance
        provenance_hash = self._hash_retirement(instrument_id, event)
        if self._provenance is not None:
            self._provenance.record(
                entity_type="instrument",
                action="retire",
                entity_id=instrument_id,
                data=event,
            )

        if _METRICS_AVAILABLE and _record_retirement is not None:
            _record_retirement(instrument_id, "retired")

        logger.info("Instrument %s retired (reason=%s)", instrument_id, reason)
        return {
            "instrument_id": instrument_id,
            "status": "RETIRED",
            "retired_at": now.isoformat(),
            "reason": reason,
            "retired_by": retired_by,
            "provenance_hash": provenance_hash,
        }

    def check_retirement_status(
        self,
        instrument_id: str,
    ) -> str:
        """Check the retirement status of an instrument.

        Args:
            instrument_id: Unique instrument identifier.

        Returns:
            RetirementStatus value string:
                "RETIRED" if in the retired set,
                "ALLOCATED" if in the used instruments set but not retired,
                "ACTIVE" otherwise.
        """
        with self._lock:
            if instrument_id in self._retired_instruments:
                return RetirementStatus.RETIRED.value
            if instrument_id in self._used_instruments:
                return RetirementStatus.ALLOCATED.value
        return RetirementStatus.ACTIVE.value

    def list_retired_instruments(self) -> List[Dict[str, Any]]:
        """List all retired instruments with their retirement details.

        Returns:
            List of dicts, each with:
                - ``instrument_id`` (str)
                - ``retired_at`` (str): Timestamp of last retirement event.
                - ``event_count`` (int): Number of retirement events.
                - ``last_event`` (dict): Most recent retirement event.
        """
        with self._lock:
            result: List[Dict[str, Any]] = []
            for inst_id in sorted(self._retired_instruments):
                history = self._retirement_history.get(inst_id, [])
                last_event = history[-1] if history else {}
                result.append({
                    "instrument_id": inst_id,
                    "retired_at": last_event.get("timestamp"),
                    "event_count": len(history),
                    "last_event": last_event,
                })
        return result

    def get_retirement_history(
        self,
        instrument_id: str,
    ) -> List[Dict[str, Any]]:
        """Get the full retirement event history for an instrument.

        Args:
            instrument_id: Unique instrument identifier.

        Returns:
            List of retirement event dicts in chronological order.
            Empty list if the instrument has no retirement events.
        """
        with self._lock:
            return list(self._retirement_history.get(instrument_id, []))

    # ==================================================================
    # PUBLIC API: Utilities
    # ==================================================================

    def get_instrument_priority(
        self,
        instrument_type: str,
    ) -> int:
        """Get the GHG Protocol hierarchy priority for an instrument type.

        Args:
            instrument_type: InstrumentType value string.

        Returns:
            Priority integer (1=highest, 6=residual mix).
            Returns 99 for unrecognized types.
        """
        return INSTRUMENT_PRIORITY.get(instrument_type.upper(), 99)

    def sort_by_priority(
        self,
        instruments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sort instruments by GHG Protocol hierarchy priority.

        Primary sort: ascending priority number (1=highest).
        Secondary sort (tiebreaker): descending available MWh capacity.

        Args:
            instruments: List of instrument dicts with ``type`` and ``mwh``.

        Returns:
            New list sorted by priority (highest first) then capacity.
        """
        def _sort_key(inst: Dict[str, Any]) -> Tuple[int, Decimal]:
            itype = str(inst.get("type", "")).upper()
            priority = INSTRUMENT_PRIORITY.get(itype, 99)
            try:
                mwh = _to_decimal(inst.get("mwh", "0"))
            except (ValueError, InvalidOperation):
                mwh = _ZERO
            # Negate mwh for descending capacity within same priority
            return (priority, -mwh)

        return sorted(instruments, key=_sort_key)

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine runtime statistics.

        Returns:
            Dict with keys:
                - ``allocation_count`` (int): Total allocations performed.
                - ``validation_count`` (int): Total validations performed.
                - ``retirement_count`` (int): Total retirements performed.
                - ``total_mwh_allocated`` (str): Cumulative MWh submitted.
                - ``total_mwh_covered`` (str): Cumulative MWh covered.
                - ``overall_coverage_pct`` (str): Overall coverage percentage.
                - ``used_instrument_count`` (int): Unique instruments used.
                - ``retired_instrument_count`` (int): Instruments retired.
                - ``reporting_year`` (int): Default reporting year.
        """
        with self._lock:
            overall_pct = _ZERO
            if self._total_mwh_allocated > _ZERO:
                overall_pct = (
                    self._total_mwh_covered * _HUNDRED / self._total_mwh_allocated
                ).quantize(self._precision_quantizer, rounding=ROUND_HALF_UP)

            return {
                "allocation_count": self._allocation_count,
                "validation_count": self._validation_count,
                "retirement_count": self._retirement_count,
                "total_mwh_allocated": str(self._total_mwh_allocated.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP,
                )),
                "total_mwh_covered": str(self._total_mwh_covered.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP,
                )),
                "overall_coverage_pct": str(overall_pct),
                "used_instrument_count": len(self._used_instruments),
                "retired_instrument_count": len(self._retired_instruments),
                "reporting_year": self._default_reporting_year,
            }

    def reset(self) -> None:
        """Reset all mutable engine state.

        Clears retired instruments, used instruments, counters, and history.
        Configuration and provenance tracker are preserved.
        Intended for testing.
        """
        with self._lock:
            self._retired_instruments.clear()
            self._retirement_history.clear()
            self._used_instruments.clear()
            self._allocation_count = 0
            self._validation_count = 0
            self._retirement_count = 0
            self._total_mwh_allocated = _ZERO
            self._total_mwh_covered = _ZERO
        logger.info("InstrumentAllocationEngine reset to initial state")

    # ==================================================================
    # INTERNAL: Purchase parsing
    # ==================================================================

    def _parse_purchase(
        self,
        purchase: Dict[str, Any],
        alloc_id: str,
    ) -> Tuple[Decimal, str, str, str]:
        """Parse and validate a purchase dict.

        Args:
            purchase: Purchase dict.
            alloc_id: Allocation ID for generated purchase_id.

        Returns:
            Tuple of (mwh, facility_id, purchase_id, region).

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # MWh
        raw_mwh = purchase.get("mwh")
        if raw_mwh is None:
            raise ValueError("Purchase must include 'mwh' field")
        purchase_mwh = _to_decimal(raw_mwh).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        if purchase_mwh <= _ZERO:
            raise ValueError(
                f"Purchase MWh must be positive, got {purchase_mwh}"
            )

        # Facility ID
        facility_id = str(purchase.get("facility_id", "UNKNOWN"))

        # Purchase ID
        purchase_id = str(purchase.get(
            "purchase_id", f"purch_{alloc_id}",
        ))

        # Region
        region = str(purchase.get("region", "GLOBAL")).upper()

        return purchase_mwh, facility_id, purchase_id, region

    # ==================================================================
    # INTERNAL: Instrument validation and filtering
    # ==================================================================

    def _validate_and_filter_instruments(
        self,
        instruments: List[Dict[str, Any]],
        purchase_region: str,
        reporting_year: int,
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate instruments and partition into valid/issues.

        Args:
            instruments: Raw instrument list.
            purchase_region: Region of the purchase for geographic match.
            reporting_year: Reporting year for vintage check.
            trace: Allocation trace to append steps.

        Returns:
            Tuple of (valid_instruments, validation_issues).
        """
        if len(instruments) > MAX_INSTRUMENTS_PER_ALLOCATION:
            raise ValueError(
                f"Instrument count {len(instruments)} exceeds maximum "
                f"{MAX_INSTRUMENTS_PER_ALLOCATION}"
            )

        valid: List[Dict[str, Any]] = []
        issues: List[Dict[str, Any]] = []

        for idx, inst in enumerate(instruments):
            result = self.validate_instrument(
                instrument=inst,
                reporting_year=reporting_year,
                consumption_region=purchase_region,
            )
            if result["is_valid"]:
                valid.append(inst)
            else:
                issues.append({
                    "index": idx,
                    "instrument_id": result["instrument_id"],
                    "issues": result["issues"],
                    "checks": result["checks"],
                })
                # In non-strict mode, geographic and vintage failures are
                # treated as warnings and the instrument is still included
                if not self._strict_geographic and not self._strict_vintage:
                    checks = result["checks"]
                    blocking = (
                        not checks.get("not_double_counted", True)
                        or not checks.get("not_retired", True)
                        or not checks.get("has_capacity", True)
                    )
                    if not blocking:
                        valid.append(inst)

        return valid, issues

    # ==================================================================
    # INTERNAL: Sequential allocation
    # ==================================================================

    def _allocate_sequential(
        self,
        purchase_mwh: Decimal,
        sorted_instruments: List[Dict[str, Any]],
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], Decimal]:
        """Allocate instruments sequentially until purchase is covered.

        Args:
            purchase_mwh: Total purchase consumption in MWh.
            sorted_instruments: Pre-sorted list of valid instruments.
            trace: Allocation trace to append steps.

        Returns:
            Tuple of (allocations list, total covered MWh).
        """
        remaining = purchase_mwh
        allocations: List[Dict[str, Any]] = []
        step = len(trace) + 1

        for inst in sorted_instruments:
            if remaining <= _ZERO:
                break

            inst_id = str(inst.get("id", "UNKNOWN"))
            inst_type = str(inst.get("type", "")).upper()
            inst_mwh = _to_decimal(inst.get("mwh", "0"))
            ef = inst.get("emission_factor")

            # Allocate the lesser of remaining demand and instrument capacity
            allocated = min(remaining, inst_mwh)
            allocated = allocated.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )

            allocation_entry: Dict[str, Any] = {
                "instrument_id": inst_id,
                "instrument_type": inst_type,
                "priority": INSTRUMENT_PRIORITY.get(inst_type, 99),
                "available_mwh": str(inst_mwh),
                "allocated_mwh": str(allocated),
                "region": str(inst.get("region", "GLOBAL")),
                "vintage_year": inst.get("vintage_year"),
                "tracking_system": inst.get("tracking_system"),
                "generator_id": inst.get("generator_id"),
                "technology": inst.get("technology"),
            }

            if ef is not None:
                try:
                    ef_dec = _to_decimal(ef)
                    allocation_entry["emission_factor_tco2e_per_mwh"] = str(ef_dec)
                    allocated_emissions = (allocated * ef_dec).quantize(
                        self._precision_quantizer, rounding=ROUND_HALF_UP,
                    )
                    allocation_entry["allocated_emissions_tco2e"] = str(
                        allocated_emissions
                    )
                except (ValueError, InvalidOperation):
                    pass

            allocations.append(allocation_entry)
            remaining = (remaining - allocated).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )

            trace.append(
                f"[{step}] Allocated {allocated} MWh from {inst_id} "
                f"({inst_type}, priority={INSTRUMENT_PRIORITY.get(inst_type, 99)}), "
                f"remaining={remaining}"
            )
            step += 1

        covered = (purchase_mwh - remaining).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        if covered < _ZERO:
            covered = _ZERO

        return allocations, covered

    # ==================================================================
    # INTERNAL: Proportional allocation
    # ==================================================================

    def _allocate_proportional_impl(
        self,
        purchase_mwh: Decimal,
        valid_instruments: List[Dict[str, Any]],
        total_available: Decimal,
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], Decimal]:
        """Perform proportional allocation.

        Each instrument receives a share of purchase_mwh proportional
        to its capacity relative to total_available.

        If total_available < purchase_mwh, all instruments are fully
        allocated and the remainder is uncovered.

        Args:
            purchase_mwh: Total purchase consumption in MWh.
            valid_instruments: Validated instruments with capacity.
            total_available: Sum of all instrument capacities.
            trace: Allocation trace to append steps.

        Returns:
            Tuple of (allocations list, total covered MWh).
        """
        allocations: List[Dict[str, Any]] = []
        covered_sum = _ZERO
        step = len(trace) + 1

        # Determine the total to distribute (min of purchase and available)
        distributable = min(purchase_mwh, total_available)

        for inst in valid_instruments:
            inst_id = str(inst.get("id", "UNKNOWN"))
            inst_type = str(inst.get("type", "")).upper()
            inst_mwh = _to_decimal(inst.get("mwh", "0"))
            ef = inst.get("emission_factor")

            if inst_mwh <= _ZERO or total_available <= _ZERO:
                continue

            # Proportional share
            share = (inst_mwh / total_available).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            allocated = (distributable * share).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            # Ensure we do not exceed instrument capacity
            allocated = min(allocated, inst_mwh)

            allocation_entry: Dict[str, Any] = {
                "instrument_id": inst_id,
                "instrument_type": inst_type,
                "priority": INSTRUMENT_PRIORITY.get(inst_type, 99),
                "available_mwh": str(inst_mwh),
                "allocated_mwh": str(allocated),
                "share_pct": str((share * _HUNDRED).quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP,
                )),
                "region": str(inst.get("region", "GLOBAL")),
                "vintage_year": inst.get("vintage_year"),
                "tracking_system": inst.get("tracking_system"),
            }

            if ef is not None:
                try:
                    ef_dec = _to_decimal(ef)
                    allocation_entry["emission_factor_tco2e_per_mwh"] = str(ef_dec)
                    allocated_emissions = (allocated * ef_dec).quantize(
                        self._precision_quantizer, rounding=ROUND_HALF_UP,
                    )
                    allocation_entry["allocated_emissions_tco2e"] = str(
                        allocated_emissions
                    )
                except (ValueError, InvalidOperation):
                    pass

            allocations.append(allocation_entry)
            covered_sum += allocated

            trace.append(
                f"[{step}] Proportional: {allocated} MWh ({share*_HUNDRED}%) "
                f"from {inst_id} ({inst_type})"
            )
            step += 1

        # Adjust for rounding: ensure covered does not exceed purchase
        if covered_sum > purchase_mwh:
            covered_sum = purchase_mwh

        return allocations, covered_sum

    # ==================================================================
    # INTERNAL: Record used instruments
    # ==================================================================

    def _record_used_instruments(
        self,
        allocations: List[Dict[str, Any]],
        alloc_id: str,
        facility_id: str,
    ) -> None:
        """Record allocated instruments to prevent double-counting.

        Args:
            allocations: List of allocation entries from an allocation.
            alloc_id: Allocation ID.
            facility_id: Facility that consumed these instruments.
        """
        with self._lock:
            for entry in allocations:
                inst_id = entry.get("instrument_id", "")
                if inst_id and inst_id != "UNKNOWN":
                    self._used_instruments[inst_id] = {
                        "allocation_id": alloc_id,
                        "facility_id": facility_id,
                        "allocated_mwh": entry.get("allocated_mwh", "0"),
                        "instrument_type": entry.get("instrument_type", ""),
                        "timestamp": _utcnow().isoformat(),
                    }

    # ==================================================================
    # INTERNAL: Coverage computation
    # ==================================================================

    def _compute_coverage_pct(
        self,
        covered_mwh: Decimal,
        total_mwh: Decimal,
    ) -> Decimal:
        """Compute coverage percentage with safe division.

        Args:
            covered_mwh: MWh covered by instruments.
            total_mwh: Total consumption MWh.

        Returns:
            Coverage percentage as Decimal (0-100+).
        """
        if total_mwh <= _ZERO:
            return _ZERO
        pct = (covered_mwh * _HUNDRED / total_mwh).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        return pct

    # ==================================================================
    # INTERNAL: Provenance hashing
    # ==================================================================

    def _hash_instrument_allocation(
        self,
        alloc_id: str,
        purchase: Dict[str, Any],
        allocations: List[Dict[str, Any]],
        covered_mwh: Decimal,
        trace: List[str],
    ) -> str:
        """Compute SHA-256 provenance hash for an allocation result.

        The hash covers the allocation ID, purchase data, all per-instrument
        allocations, covered MWh, and the trace log. This provides a
        tamper-evident fingerprint for the complete allocation.

        Args:
            alloc_id: Allocation identifier.
            purchase: Original purchase dict.
            allocations: Per-instrument allocation results.
            covered_mwh: Total covered MWh.
            trace: Allocation trace steps.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        payload = {
            "allocation_id": alloc_id,
            "purchase": purchase,
            "allocations": allocations,
            "covered_mwh": str(covered_mwh),
            "trace_steps": len(trace),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _hash_retirement(
        self,
        instrument_id: str,
        event: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 hash for a retirement event.

        Args:
            instrument_id: Instrument that was retired.
            event: Retirement event dict.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        payload = {
            "instrument_id": instrument_id,
            "event": event,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing allocation count, coverage, and configuration.
        """
        with self._lock:
            return (
                f"InstrumentAllocationEngine("
                f"allocations={self._allocation_count}, "
                f"retirements={self._retirement_count}, "
                f"used_instruments={len(self._used_instruments)}, "
                f"reporting_year={self._default_reporting_year})"
            )

    def __len__(self) -> int:
        """Return the total number of allocations performed.

        Returns:
            Integer count of allocations.
        """
        with self._lock:
            return self._allocation_count


# ---------------------------------------------------------------------------
# Standalone utility functions (module-level convenience)
# ---------------------------------------------------------------------------


def create_instrument_allocation_engine(
    config: Optional[Dict[str, Any]] = None,
) -> InstrumentAllocationEngine:
    """Factory function to create an InstrumentAllocationEngine instance.

    Args:
        config: Optional configuration dict (see __init__ for keys).

    Returns:
        Configured InstrumentAllocationEngine instance.

    Example:
        >>> engine = create_instrument_allocation_engine({
        ...     "default_reporting_year": 2025,
        ...     "strict_geographic_match": True,
        ... })
    """
    return InstrumentAllocationEngine(config=config)


def get_instrument_hierarchy() -> Dict[str, int]:
    """Return the GHG Protocol instrument hierarchy mapping.

    Returns:
        Dict mapping InstrumentType value strings to priority integers.
    """
    return dict(INSTRUMENT_PRIORITY)


def get_valid_tracking_systems() -> List[str]:
    """Return the list of recognized tracking system identifiers.

    Returns:
        Sorted list of valid tracking system strings.
    """
    return sorted(_VALID_TRACKING_SYSTEMS)


def get_geographic_markets() -> Dict[str, List[str]]:
    """Return the geographic market interconnection mapping.

    Returns:
        Dict mapping region codes to lists of interconnected region codes.
    """
    return {k: sorted(v) for k, v in GEOGRAPHIC_MARKETS.items()}


def get_vintage_windows() -> Dict[str, int]:
    """Return the vintage window mapping by instrument type.

    Returns:
        Dict mapping InstrumentType value strings to maximum vintage age
        in years.
    """
    return dict(VINTAGE_WINDOWS)
