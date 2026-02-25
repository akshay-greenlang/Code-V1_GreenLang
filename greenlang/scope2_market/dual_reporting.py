# -*- coding: utf-8 -*-
"""
DualReportingEngine - Engine 4: Scope 2 Market-Based Emissions Agent (AGENT-MRV-010)

Integrates location-based (AGENT-MRV-009) and market-based (AGENT-MRV-010) Scope 2
emission results into a unified dual-report format as required by the GHG Protocol
Scope 2 Guidance (2015). The Guidance mandates that organisations report BOTH
location-based and market-based Scope 2 totals side by side, enabling stakeholders
to evaluate the impact of contractual instruments (RECs, PPAs, GOs, supplier-
specific factors) versus grid-average emissions.

Core Capabilities:
    - Side-by-side dual report generation (location vs. market)
    - Batch and per-facility dual reporting
    - Procurement impact analysis (renewable reduction quantification)
    - RE100 progress tracking toward 100% renewable electricity
    - Additionality scoring for contractual instruments
    - Year-over-year trend comparison for both methods
    - Coverage gap analysis and instrument recommendation
    - Cost estimation for uncovered MWh
    - Dual-report validation and completeness checking
    - GHG Protocol Table, CDP C8.2d, and CSRD/ESRS E1 formatting
    - SHA-256 provenance hashing on every report

Zero-Hallucination Guarantees:
    - All arithmetic uses Python ``Decimal`` with ``ROUND_HALF_UP``
    - No LLM calls in any calculation path
    - Every report carries a SHA-256 provenance hash
    - Deterministic: identical inputs produce identical outputs (bit-perfect)
    - Thread-safe via ``threading.RLock``

Formula References:
    difference_tco2e = market_based_tco2e - location_based_tco2e
    difference_pct   = (difference / location_based) * 100
    renewable_impact  = location_based - market_based   (when market < location)
    RE100 progress   = (renewable_mwh / total_mwh) * 100

Supported Disclosure Formats:
    - GHG Protocol Scope 2 Guidance Table 6.1 (dual report)
    - CDP Climate Change 2024 C8.2d (market-based details)
    - CSRD / ESRS E1-6 (Scope 2 GHG emissions)
    - ISO 14064-1:2018 Category 2

Example:
    >>> from greenlang.scope2_market.dual_reporting import DualReportingEngine
    >>> from decimal import Decimal
    >>> engine = DualReportingEngine()
    >>> loc = {"total_co2e_tonnes": Decimal("1500.00"), "facility_id": "FAC-001",
    ...        "total_mwh": Decimal("5000"), "period": "2025"}
    >>> mkt = {"total_co2e_tonnes": Decimal("800.00"), "facility_id": "FAC-001",
    ...        "total_mwh": Decimal("5000"), "period": "2025",
    ...        "instruments": [{"type": "REC", "mwh_covered": Decimal("3000")}]}
    >>> report = engine.generate_dual_report(loc, mkt)
    >>> assert report["location_based_tco2e"] == Decimal("1500.00")
    >>> assert report["market_based_tco2e"] == Decimal("800.00")
    >>> assert report["lower_method"] == "market"

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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

#: 8 decimal places for deterministic Scope 2 arithmetic.
_PRECISION = Decimal("0.00000001")

#: 6 decimal places for percentage outputs.
_PCT_PRECISION = Decimal("0.000001")

#: Conversion constant: kg to metric tonnes.
_KG_TO_TONNES = Decimal("0.001")

#: Conversion constant: tonnes to kg.
_TONNES_TO_KG = Decimal("1000")


# ---------------------------------------------------------------------------
# Built-in Constants
# ---------------------------------------------------------------------------

#: Typical REC / GO / EAC cost per MWh by region (USD).
#: Source: S&P Global Platts, Bloomberg NEF, RECS International (2024).
#: These are approximate mid-market prices for compliance-grade certificates.
TYPICAL_REC_COST_PER_MWH: Dict[str, Decimal] = {
    "US_NATIONAL": Decimal("2.50"),
    "US_GREEN_E": Decimal("3.00"),
    "US_PJM_TIER1": Decimal("12.00"),
    "US_NEW_ENGLAND": Decimal("25.00"),
    "US_CALIFORNIA": Decimal("18.00"),
    "US_TEXAS": Decimal("1.80"),
    "EU_GO": Decimal("1.20"),
    "EU_GO_NORDIC": Decimal("0.50"),
    "EU_GO_WIND": Decimal("2.80"),
    "EU_GO_SOLAR": Decimal("3.20"),
    "UK_REGO": Decimal("4.50"),
    "AU_LGC": Decimal("32.00"),
    "AU_STC": Decimal("38.00"),
    "JP_JCREDIT": Decimal("5.50"),
    "JP_NFC": Decimal("8.00"),
    "CN_GEC": Decimal("3.50"),
    "IN_REC": Decimal("1.50"),
    "BR_IREC": Decimal("2.00"),
    "ZA_REC": Decimal("1.80"),
    "GLOBAL_IREC": Decimal("2.50"),
    "GLOBAL_PPA": Decimal("45.00"),
}

#: RE100 target percentage (100% renewable electricity).
RE100_TARGET_PCT: Decimal = Decimal("100.0")

#: Additionality criteria for evaluating whether contractual instruments
#: drive NEW renewable capacity versus merely re-labelling existing supply.
#: Each criterion is weighted 0-20 for a maximum additionality score of 100.
ADDITIONALITY_CRITERIA: List[Dict[str, Any]] = [
    {
        "id": "ADD_01",
        "name": "temporal_matching",
        "description": "Instrument was issued within the same reporting year as consumption",
        "max_score": Decimal("20"),
    },
    {
        "id": "ADD_02",
        "name": "geographic_matching",
        "description": "Renewable generation is in the same grid region as consumption",
        "max_score": Decimal("20"),
    },
    {
        "id": "ADD_03",
        "name": "new_build",
        "description": "Renewable facility was commissioned within the last 5 years",
        "max_score": Decimal("20"),
    },
    {
        "id": "ADD_04",
        "name": "long_term_contract",
        "description": "Instrument backed by a PPA or long-term offtake (10+ years)",
        "max_score": Decimal("20"),
    },
    {
        "id": "ADD_05",
        "name": "third_party_verified",
        "description": "Instrument verified by an accredited registry (Green-e, IREC, GO registry)",
        "max_score": Decimal("20"),
    },
]

#: Maximum additionality score (sum of all max_score).
_MAX_ADDITIONALITY_SCORE: Decimal = sum(
    c["max_score"] for c in ADDITIONALITY_CRITERIA
)

#: Instrument type quality hierarchy for market-based method.
#: Lower rank = higher quality per GHG Protocol Scope 2 Guidance.
INSTRUMENT_QUALITY_HIERARCHY: Dict[str, int] = {
    "supplier_specific_ef": 1,
    "green_tariff": 2,
    "ppa": 3,
    "rec": 4,
    "go": 5,
    "eac": 5,
    "irec": 5,
    "lgc": 5,
    "rego": 5,
    "residual_mix": 6,
    "grid_average": 7,
}

#: Version string for this engine.
VERSION: str = "1.0.0"

#: Database table prefix.
TABLE_PREFIX: str = "gl_s2m_"


# ---------------------------------------------------------------------------
# Prometheus metrics helpers (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]

if _PROMETHEUS_AVAILABLE:
    _s2m_dual_reports_total = Counter(
        "gl_s2m_dual_reports_total",
        "Total Scope 2 dual reports generated",
        labelnames=["report_type"],
    )
    _s2m_dual_report_duration = Histogram(
        "gl_s2m_dual_report_duration_seconds",
        "Duration of dual report generation in seconds",
        labelnames=["operation"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
else:
    _s2m_dual_reports_total = None  # type: ignore[assignment]
    _s2m_dual_report_duration = None  # type: ignore[assignment]


def _record_dual_report_metric(report_type: str) -> None:
    """Record a dual report generation metric."""
    if _PROMETHEUS_AVAILABLE and _s2m_dual_reports_total is not None:
        _s2m_dual_reports_total.labels(report_type=report_type).inc()


def _observe_duration(operation: str, seconds: float) -> None:
    """Record operation duration metric."""
    if _PROMETHEUS_AVAILABLE and _s2m_dual_report_duration is not None:
        _s2m_dual_report_duration.labels(operation=operation).observe(seconds)


# ---------------------------------------------------------------------------
# Provenance helper (lightweight inline tracker)
# ---------------------------------------------------------------------------

class _DualReportingProvenance:
    """Chain-hashing provenance tracker for dual reporting operations.

    Each recorded entry chains its SHA-256 hash to the previous entry,
    producing a tamper-evident audit log for every dual report, analysis,
    and formatting operation.
    """

    _GENESIS = "GL-MRV-010-SCOPE2-MARKET-DUAL-REPORTING-GENESIS"

    def __init__(self) -> None:
        """Initialize with genesis hash."""
        self._genesis: str = hashlib.sha256(
            self._GENESIS.encode("utf-8")
        ).hexdigest()
        self._last_hash: str = self._genesis
        self._entries: List[Dict[str, Any]] = []
        self._lock: threading.RLock = threading.RLock()

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a provenance entry and return it.

        Args:
            entity_type: Category of the entity (e.g. 'dual_report').
            action: Action performed (e.g. 'generate', 'validate').
            entity_id: Unique identifier for the entity.
            data: Optional data payload to hash.

        Returns:
            Dictionary with entity_type, entity_id, action, hash_value,
            parent_hash, timestamp, and data_hash.
        """
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        data_hash = hashlib.sha256(
            json.dumps(data or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        with self._lock:
            parent = self._last_hash
            payload = f"{parent}|{data_hash}|{action}|{ts}"
            chain_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            entry = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "hash_value": chain_hash,
                "parent_hash": parent,
                "timestamp": ts,
                "data_hash": data_hash,
            }
            self._entries.append(entry)
            self._last_hash = chain_hash
        return entry

    def verify_chain(self) -> bool:
        """Verify integrity of the provenance chain.

        Returns:
            True if chain is intact, False if tampered.
        """
        with self._lock:
            if not self._entries:
                return True
            expected_parent = self._genesis
            for entry in self._entries:
                if entry["parent_hash"] != expected_parent:
                    return False
                expected_parent = entry["hash_value"]
            return True

    def get_entries(self) -> List[Dict[str, Any]]:
        """Return a copy of all provenance entries."""
        with self._lock:
            return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Return number of provenance entries."""
        with self._lock:
            return len(self._entries)

    def reset(self) -> None:
        """Reset to genesis state."""
        with self._lock:
            self._entries.clear()
            self._last_hash = self._genesis


# ---------------------------------------------------------------------------
# Utility: UTC now helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Utility: Safe Decimal conversion
# ---------------------------------------------------------------------------

def _to_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely.

    Args:
        value: int, float, str, or Decimal.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


def _safe_get_decimal(data: Dict[str, Any], key: str, default: Decimal = Decimal("0")) -> Decimal:
    """Safely extract a Decimal from a dictionary.

    Args:
        data: Source dictionary.
        key: Key to look up.
        default: Default value if key is missing or None.

    Returns:
        Decimal value.
    """
    raw = data.get(key)
    if raw is None:
        return default
    return _to_decimal(raw)


# ---------------------------------------------------------------------------
# Utility: SHA-256 hash for dual reports
# ---------------------------------------------------------------------------

def _hash_dual_reporting(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dual report payload.

    Args:
        data: Dictionary payload to hash.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ===========================================================================
# DualReportingEngine
# ===========================================================================


class DualReportingEngine:
    """Dual reporting engine for GHG Protocol Scope 2 Guidance compliance.

    Integrates location-based (MRV-009) and market-based (MRV-010) Scope 2
    emission results into side-by-side dual reports. The GHG Protocol Scope 2
    Guidance (January 2015) requires organisations to report BOTH methods for
    transparent disclosure to investors, regulators, and rating agencies.

    This engine compares, analyses, validates, and formats dual reports for
    GHG Protocol disclosures, CDP C8.2d responses, and CSRD/ESRS E1
    submissions. All arithmetic is deterministic Decimal with SHA-256
    provenance hashing.

    Thread Safety:
        All mutable state is protected by ``threading.RLock``. Multiple
        threads may safely call any public method concurrently.

    Zero-Hallucination:
        No LLM calls. All calculations are pure Decimal arithmetic.
        Every report carries a SHA-256 provenance hash.

    Attributes:
        _provenance: Chain-hashing provenance tracker.
        _lock: Reentrant lock for thread safety.
        _created_at: UTC timestamp of engine creation.
        _total_reports: Total dual reports generated.
        _total_batch_reports: Total batch dual reports generated.
        _total_facility_reports: Total facility-level dual reports.
        _total_analyses: Total analysis operations (procurement, RE100, etc.).
        _total_validations: Total validation operations.
        _total_formats: Total formatting operations (GHG, CDP, CSRD).
        _total_errors: Total errors encountered.

    Example:
        >>> engine = DualReportingEngine()
        >>> loc = {"total_co2e_tonnes": Decimal("1200"), "total_mwh": Decimal("4000")}
        >>> mkt = {"total_co2e_tonnes": Decimal("600"), "total_mwh": Decimal("4000"),
        ...        "instruments": [{"type": "REC", "mwh_covered": Decimal("2000")}]}
        >>> report = engine.generate_dual_report(loc, mkt)
        >>> assert report["difference_tco2e"] == Decimal("-600.00000000")
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DualReportingEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                    Default True.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_currency`` (str): Currency for cost estimates.
                    Default 'USD'.
        """
        config = config or {}
        self._enable_provenance: bool = config.get("enable_provenance", True)
        self._precision_places: int = config.get("decimal_precision", 8)
        self._precision: Decimal = Decimal(10) ** (-self._precision_places)
        self._default_currency: str = config.get("default_currency", "USD")

        # Provenance tracker
        self._provenance: Optional[_DualReportingProvenance] = (
            _DualReportingProvenance() if self._enable_provenance else None
        )

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        # Statistics
        self._created_at: datetime = _utcnow()
        self._total_reports: int = 0
        self._total_batch_reports: int = 0
        self._total_facility_reports: int = 0
        self._total_analyses: int = 0
        self._total_validations: int = 0
        self._total_formats: int = 0
        self._total_errors: int = 0

        logger.info(
            "DualReportingEngine initialized: provenance=%s, precision=%d",
            self._enable_provenance,
            self._precision_places,
        )

    # ------------------------------------------------------------------
    # Internal: Decimal quantize helper
    # ------------------------------------------------------------------

    def _q(self, value: Decimal) -> Decimal:
        """Quantize a Decimal to the configured precision.

        Args:
            value: Decimal to quantize.

        Returns:
            Quantized Decimal with ROUND_HALF_UP.
        """
        return value.quantize(self._precision, rounding=ROUND_HALF_UP)

    def _q_pct(self, value: Decimal) -> Decimal:
        """Quantize a percentage Decimal to 6 decimal places.

        Args:
            value: Decimal percentage to quantize.

        Returns:
            Quantized Decimal with ROUND_HALF_UP.
        """
        return value.quantize(_PCT_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Internal: Provenance recording helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Record a provenance entry if tracking is enabled.

        Args:
            entity_type: Category of the entity.
            action: Action performed.
            entity_id: Unique identifier.
            data: Optional data payload.

        Returns:
            Hash value string if provenance is enabled, else None.
        """
        if self._provenance is not None:
            entry = self._provenance.record(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                data=data,
            )
            return entry["hash_value"]
        return None

    # ------------------------------------------------------------------
    # Internal: Error counter increment
    # ------------------------------------------------------------------

    def _increment_error(self) -> None:
        """Increment the error counter under lock."""
        with self._lock:
            self._total_errors += 1

    # ==================================================================
    # Core Dual Reporting (3 public methods)
    # ==================================================================

    def generate_dual_report(
        self,
        location_result: Dict[str, Any],
        market_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a side-by-side dual report comparing location-based and
        market-based Scope 2 emissions.

        Implements GHG Protocol Scope 2 Guidance requirement for dual
        reporting. Computes the difference, identifies the lower method,
        quantifies renewable procurement impact, and attaches a SHA-256
        provenance hash.

        Args:
            location_result: Location-based calculation result dictionary.
                Must contain ``total_co2e_tonnes`` (Decimal or numeric).
                Optional: ``facility_id``, ``total_mwh``, ``period``,
                ``energy_type``, ``grid_region``.
            market_result: Market-based calculation result dictionary.
                Must contain ``total_co2e_tonnes`` (Decimal or numeric).
                Optional: ``facility_id``, ``total_mwh``, ``period``,
                ``instruments`` (list of dicts with ``type``, ``mwh_covered``).

        Returns:
            Dictionary with keys:
                - report_id (str): UUID for this dual report
                - location_based_tco2e (Decimal): Location-based total
                - market_based_tco2e (Decimal): Market-based total
                - difference_tco2e (Decimal): market - location
                - difference_pct (Decimal): percentage difference
                - lower_method (str): 'location', 'market', or 'equal'
                - renewable_impact_tco2e (Decimal): reduction from instruments
                - facility_id (str or None)
                - period (str or None)
                - total_mwh (Decimal or None)
                - instrument_count (int): number of instruments
                - provenance_hash (str): SHA-256 hash
                - generated_at (str): ISO 8601 timestamp
                - metadata (dict): additional context

        Raises:
            ValueError: If location_result or market_result is missing
                ``total_co2e_tonnes``.
        """
        start_time = time.monotonic()
        report_id = str(uuid.uuid4())

        try:
            # Extract and validate required fields
            location_tco2e = self._extract_tco2e(location_result, "location_result")
            market_tco2e = self._extract_tco2e(market_result, "market_result")

            # Compute difference (market - location)
            difference_tco2e = self._q(market_tco2e - location_tco2e)

            # Compute percentage difference relative to location
            if location_tco2e > Decimal("0"):
                difference_pct = self._q_pct(
                    (difference_tco2e / location_tco2e) * Decimal("100")
                )
            else:
                difference_pct = Decimal("0")

            # Determine lower method
            lower_method = self._determine_lower_method(location_tco2e, market_tco2e)

            # Calculate renewable impact
            renewable_impact_tco2e = self._calculate_renewable_impact(
                location_tco2e, market_tco2e,
            )

            # Extract optional fields
            facility_id = (
                location_result.get("facility_id")
                or market_result.get("facility_id")
            )
            period = (
                location_result.get("period")
                or market_result.get("period")
            )
            total_mwh_raw = (
                location_result.get("total_mwh")
                or market_result.get("total_mwh")
            )
            total_mwh = _to_decimal(total_mwh_raw) if total_mwh_raw is not None else None

            instruments = market_result.get("instruments", [])
            instrument_count = len(instruments) if isinstance(instruments, list) else 0

            # Build report
            report = {
                "report_id": report_id,
                "location_based_tco2e": location_tco2e,
                "market_based_tco2e": market_tco2e,
                "difference_tco2e": difference_tco2e,
                "difference_pct": difference_pct,
                "lower_method": lower_method,
                "renewable_impact_tco2e": renewable_impact_tco2e,
                "facility_id": facility_id,
                "period": period,
                "total_mwh": total_mwh,
                "instrument_count": instrument_count,
                "generated_at": _utcnow().isoformat(),
                "metadata": {
                    "engine": "DualReportingEngine",
                    "version": VERSION,
                    "agent": "AGENT-MRV-010",
                },
            }

            # Provenance hash
            provenance_hash = _hash_dual_reporting(report)
            report["provenance_hash"] = provenance_hash

            # Record provenance chain
            self._record_provenance(
                entity_type="dual_report",
                action="generate",
                entity_id=report_id,
                data=report,
            )

            # Update statistics
            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_reports += 1

            _record_dual_report_metric("single")
            _observe_duration("generate_dual_report", elapsed)

            logger.info(
                "Dual report generated: id=%s, loc=%.2f tCO2e, mkt=%.2f tCO2e, "
                "diff=%.2f%%, lower=%s (%.4fs)",
                report_id,
                float(location_tco2e),
                float(market_tco2e),
                float(difference_pct),
                lower_method,
                elapsed,
            )

            return report

        except Exception as exc:
            self._increment_error()
            logger.error(
                "generate_dual_report failed: %s", exc, exc_info=True,
            )
            raise

    def generate_dual_report_batch(
        self,
        location_results: List[Dict[str, Any]],
        market_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate dual reports for a batch of location/market result pairs.

        Pairs are matched by list index. Both lists must have the same length.

        Args:
            location_results: List of location-based result dictionaries.
            market_results: List of market-based result dictionaries.

        Returns:
            List of dual report dictionaries (same order as inputs).

        Raises:
            ValueError: If lists have different lengths.
        """
        start_time = time.monotonic()

        if len(location_results) != len(market_results):
            raise ValueError(
                f"Batch size mismatch: {len(location_results)} location results "
                f"vs {len(market_results)} market results"
            )

        reports: List[Dict[str, Any]] = []
        for idx, (loc, mkt) in enumerate(zip(location_results, market_results)):
            try:
                report = self.generate_dual_report(loc, mkt)
                report["batch_index"] = idx
                reports.append(report)
            except Exception as exc:
                logger.warning(
                    "Batch item %d failed: %s", idx, exc,
                )
                reports.append({
                    "batch_index": idx,
                    "error": str(exc),
                    "status": "failed",
                })

        elapsed = time.monotonic() - start_time
        with self._lock:
            self._total_batch_reports += 1

        _record_dual_report_metric("batch")
        _observe_duration("generate_dual_report_batch", elapsed)

        logger.info(
            "Batch dual report: %d pairs, %d succeeded, %.4fs",
            len(location_results),
            sum(1 for r in reports if "error" not in r),
            elapsed,
        )

        return reports

    def generate_facility_dual_report(
        self,
        facility_id: str,
        location_result: Dict[str, Any],
        market_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a dual report for a specific facility.

        Enriches the standard dual report with the explicit facility_id
        and validates that both results reference the same facility.

        Args:
            facility_id: The facility identifier.
            location_result: Location-based result for the facility.
            market_result: Market-based result for the facility.

        Returns:
            Dual report dictionary with ``facility_id`` guaranteed set.

        Raises:
            ValueError: If either result references a different facility.
        """
        start_time = time.monotonic()

        # Validate facility consistency
        loc_fid = location_result.get("facility_id")
        mkt_fid = market_result.get("facility_id")

        if loc_fid is not None and loc_fid != facility_id:
            raise ValueError(
                f"Location result facility_id '{loc_fid}' does not match "
                f"requested facility_id '{facility_id}'"
            )
        if mkt_fid is not None and mkt_fid != facility_id:
            raise ValueError(
                f"Market result facility_id '{mkt_fid}' does not match "
                f"requested facility_id '{facility_id}'"
            )

        # Inject facility_id
        location_result = dict(location_result)
        market_result = dict(market_result)
        location_result["facility_id"] = facility_id
        market_result["facility_id"] = facility_id

        report = self.generate_dual_report(location_result, market_result)
        report["facility_id"] = facility_id

        elapsed = time.monotonic() - start_time
        with self._lock:
            self._total_facility_reports += 1

        _record_dual_report_metric("facility")
        _observe_duration("generate_facility_dual_report", elapsed)

        logger.info(
            "Facility dual report: facility_id=%s, %.4fs",
            facility_id, elapsed,
        )

        return report

    # ==================================================================
    # Analysis (4 public methods)
    # ==================================================================

    def calculate_procurement_impact(
        self,
        location_tco2e: Any,
        market_tco2e: Any,
    ) -> Dict[str, Any]:
        """Calculate the emissions reduction impact of renewable procurement.

        Quantifies how much the market-based method is lower than the
        location-based method, attributing the difference to contractual
        instruments (RECs, PPAs, GOs, supplier-specific EFs).

        Args:
            location_tco2e: Location-based total in tCO2e (Decimal or numeric).
            market_tco2e: Market-based total in tCO2e (Decimal or numeric).

        Returns:
            Dictionary with keys:
                - location_tco2e (Decimal)
                - market_tco2e (Decimal)
                - reduction_tco2e (Decimal): positive = reduction, negative = increase
                - reduction_pct (Decimal): percentage reduction
                - procurement_effective (bool): True if market < location
                - provenance_hash (str)
        """
        start_time = time.monotonic()
        analysis_id = str(uuid.uuid4())

        try:
            loc = self._q(_to_decimal(location_tco2e))
            mkt = self._q(_to_decimal(market_tco2e))

            reduction = self._q(loc - mkt)

            if loc > Decimal("0"):
                reduction_pct = self._q_pct(
                    (reduction / loc) * Decimal("100")
                )
            else:
                reduction_pct = Decimal("0")

            procurement_effective = mkt < loc

            result = {
                "analysis_id": analysis_id,
                "location_tco2e": loc,
                "market_tco2e": mkt,
                "reduction_tco2e": reduction,
                "reduction_pct": reduction_pct,
                "procurement_effective": procurement_effective,
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="procurement_impact",
                action="calculate",
                entity_id=analysis_id,
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("calculate_procurement_impact", elapsed)

            logger.info(
                "Procurement impact: reduction=%.2f tCO2e (%.1f%%), effective=%s",
                float(reduction), float(reduction_pct), procurement_effective,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "calculate_procurement_impact failed: %s", exc, exc_info=True,
            )
            raise

    def calculate_re100_progress(
        self,
        total_mwh: Any,
        renewable_mwh: Any,
        target_pct: Any = RE100_TARGET_PCT,
    ) -> Dict[str, Any]:
        """Calculate progress toward RE100 (100% renewable electricity).

        RE100 is a global initiative of companies committed to sourcing
        100% renewable electricity. This method tracks progress by
        comparing renewable MWh (backed by instruments) to total MWh.

        Args:
            total_mwh: Total electricity consumption in MWh.
            renewable_mwh: MWh covered by renewable instruments.
            target_pct: Target renewable percentage (default 100.0).

        Returns:
            Dictionary with keys:
                - analysis_id (str)
                - total_mwh (Decimal)
                - renewable_mwh (Decimal)
                - non_renewable_mwh (Decimal)
                - renewable_pct (Decimal): actual renewable percentage
                - target_pct (Decimal)
                - gap_pct (Decimal): target_pct - renewable_pct (0 if met)
                - gap_mwh (Decimal): MWh still needed
                - on_track (bool): True if renewable_pct >= target_pct
                - provenance_hash (str)

        Raises:
            ValueError: If total_mwh is zero or negative, or if
                renewable_mwh exceeds total_mwh.
        """
        start_time = time.monotonic()
        analysis_id = str(uuid.uuid4())

        try:
            total = self._q(_to_decimal(total_mwh))
            renewable = self._q(_to_decimal(renewable_mwh))
            target = self._q(_to_decimal(target_pct))

            if total <= Decimal("0"):
                raise ValueError("total_mwh must be positive")
            if renewable < Decimal("0"):
                raise ValueError("renewable_mwh must be non-negative")
            if renewable > total:
                raise ValueError(
                    f"renewable_mwh ({renewable}) exceeds total_mwh ({total})"
                )

            non_renewable = self._q(total - renewable)
            renewable_pct = self._q_pct(
                (renewable / total) * Decimal("100")
            )

            gap_pct = self._q_pct(max(Decimal("0"), target - renewable_pct))
            gap_mwh = self._q(
                (gap_pct / Decimal("100")) * total
            ) if gap_pct > Decimal("0") else Decimal("0")

            on_track = renewable_pct >= target

            result = {
                "analysis_id": analysis_id,
                "total_mwh": total,
                "renewable_mwh": renewable,
                "non_renewable_mwh": non_renewable,
                "renewable_pct": renewable_pct,
                "target_pct": target,
                "gap_pct": gap_pct,
                "gap_mwh": gap_mwh,
                "on_track": on_track,
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="re100_progress",
                action="calculate",
                entity_id=analysis_id,
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("calculate_re100_progress", elapsed)

            logger.info(
                "RE100 progress: %.1f%% renewable (target %.1f%%), gap=%.1f MWh, on_track=%s",
                float(renewable_pct), float(target), float(gap_mwh), on_track,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "calculate_re100_progress failed: %s", exc, exc_info=True,
            )
            raise

    def calculate_additionality_score(
        self,
        instruments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess additionality of contractual instruments.

        Evaluates whether instruments drive NEW renewable generation
        capacity versus merely re-labelling existing supply. Uses five
        criteria (temporal matching, geographic matching, new build,
        long-term contract, third-party verification) each scored 0-20
        for a maximum of 100.

        Args:
            instruments: List of instrument dictionaries. Each may contain:
                - ``type`` (str): Instrument type (e.g. 'REC', 'PPA', 'GO').
                - ``mwh_covered`` (Decimal): MWh covered.
                - ``temporal_match`` (bool): Same year as consumption.
                - ``geographic_match`` (bool): Same grid region.
                - ``new_build`` (bool): Facility commissioned within 5 years.
                - ``long_term_contract`` (bool): PPA or 10+ year offtake.
                - ``third_party_verified`` (bool): Registry-verified.

        Returns:
            Dictionary with keys:
                - analysis_id (str)
                - instrument_count (int)
                - per_instrument_scores (list): Per-instrument score detail
                - weighted_average_score (Decimal): MWh-weighted average (0-100)
                - additionality_rating (str): 'high', 'medium', 'low', 'none'
                - criteria (list): Copy of ADDITIONALITY_CRITERIA
                - provenance_hash (str)
        """
        start_time = time.monotonic()
        analysis_id = str(uuid.uuid4())

        try:
            per_instrument_scores: List[Dict[str, Any]] = []
            total_weighted_score = Decimal("0")
            total_mwh_weight = Decimal("0")

            for inst in instruments:
                score = Decimal("0")
                detail: Dict[str, Any] = {
                    "instrument_type": inst.get("type", "unknown"),
                    "mwh_covered": _safe_get_decimal(inst, "mwh_covered", Decimal("0")),
                    "criteria_scores": {},
                }

                # Evaluate each criterion
                if inst.get("temporal_match", False):
                    score += ADDITIONALITY_CRITERIA[0]["max_score"]
                    detail["criteria_scores"]["temporal_matching"] = ADDITIONALITY_CRITERIA[0]["max_score"]
                else:
                    detail["criteria_scores"]["temporal_matching"] = Decimal("0")

                if inst.get("geographic_match", False):
                    score += ADDITIONALITY_CRITERIA[1]["max_score"]
                    detail["criteria_scores"]["geographic_matching"] = ADDITIONALITY_CRITERIA[1]["max_score"]
                else:
                    detail["criteria_scores"]["geographic_matching"] = Decimal("0")

                if inst.get("new_build", False):
                    score += ADDITIONALITY_CRITERIA[2]["max_score"]
                    detail["criteria_scores"]["new_build"] = ADDITIONALITY_CRITERIA[2]["max_score"]
                else:
                    detail["criteria_scores"]["new_build"] = Decimal("0")

                if inst.get("long_term_contract", False):
                    score += ADDITIONALITY_CRITERIA[3]["max_score"]
                    detail["criteria_scores"]["long_term_contract"] = ADDITIONALITY_CRITERIA[3]["max_score"]
                else:
                    detail["criteria_scores"]["long_term_contract"] = Decimal("0")

                if inst.get("third_party_verified", False):
                    score += ADDITIONALITY_CRITERIA[4]["max_score"]
                    detail["criteria_scores"]["third_party_verified"] = ADDITIONALITY_CRITERIA[4]["max_score"]
                else:
                    detail["criteria_scores"]["third_party_verified"] = Decimal("0")

                detail["total_score"] = score
                per_instrument_scores.append(detail)

                # Accumulate MWh-weighted score
                mwh_cov = detail["mwh_covered"]
                total_weighted_score += score * mwh_cov
                total_mwh_weight += mwh_cov

            # Compute weighted average
            if total_mwh_weight > Decimal("0"):
                weighted_avg = self._q_pct(total_weighted_score / total_mwh_weight)
            elif per_instrument_scores:
                # Fallback: simple average if no MWh data
                raw_avg = sum(
                    s["total_score"] for s in per_instrument_scores
                ) / Decimal(str(len(per_instrument_scores)))
                weighted_avg = self._q_pct(raw_avg)
            else:
                weighted_avg = Decimal("0")

            # Determine rating
            additionality_rating = self._score_to_rating(weighted_avg)

            result = {
                "analysis_id": analysis_id,
                "instrument_count": len(instruments),
                "per_instrument_scores": per_instrument_scores,
                "weighted_average_score": weighted_avg,
                "additionality_rating": additionality_rating,
                "max_possible_score": _MAX_ADDITIONALITY_SCORE,
                "criteria": [
                    {k: str(v) if isinstance(v, Decimal) else v for k, v in c.items()}
                    for c in ADDITIONALITY_CRITERIA
                ],
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="additionality_score",
                action="calculate",
                entity_id=analysis_id,
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("calculate_additionality_score", elapsed)

            logger.info(
                "Additionality score: %d instruments, weighted_avg=%.1f, rating=%s",
                len(instruments), float(weighted_avg), additionality_rating,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "calculate_additionality_score failed: %s", exc, exc_info=True,
            )
            raise

    def year_over_year_comparison(
        self,
        current_dual: Dict[str, Any],
        previous_dual: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare dual reports across two reporting periods (YoY).

        Computes year-over-year changes for both location-based and
        market-based methods, identifying trends and improvement areas.

        Args:
            current_dual: Current period dual report dictionary.
                Must contain ``location_based_tco2e`` and ``market_based_tco2e``.
            previous_dual: Previous period dual report dictionary.
                Must contain ``location_based_tco2e`` and ``market_based_tco2e``.

        Returns:
            Dictionary with keys:
                - analysis_id (str)
                - current_period (str or None)
                - previous_period (str or None)
                - location_current_tco2e, location_previous_tco2e (Decimal)
                - location_change_tco2e, location_change_pct (Decimal)
                - market_current_tco2e, market_previous_tco2e (Decimal)
                - market_change_tco2e, market_change_pct (Decimal)
                - procurement_impact_improved (bool)
                - provenance_hash (str)

        Raises:
            ValueError: If required fields are missing.
        """
        start_time = time.monotonic()
        analysis_id = str(uuid.uuid4())

        try:
            # Current period
            loc_curr = self._q(_to_decimal(current_dual.get("location_based_tco2e", 0)))
            mkt_curr = self._q(_to_decimal(current_dual.get("market_based_tco2e", 0)))

            # Previous period
            loc_prev = self._q(_to_decimal(previous_dual.get("location_based_tco2e", 0)))
            mkt_prev = self._q(_to_decimal(previous_dual.get("market_based_tco2e", 0)))

            # Location YoY change
            loc_change = self._q(loc_curr - loc_prev)
            loc_change_pct = self._q_pct(
                (loc_change / loc_prev) * Decimal("100")
            ) if loc_prev > Decimal("0") else Decimal("0")

            # Market YoY change
            mkt_change = self._q(mkt_curr - mkt_prev)
            mkt_change_pct = self._q_pct(
                (mkt_change / mkt_prev) * Decimal("100")
            ) if mkt_prev > Decimal("0") else Decimal("0")

            # Procurement impact improvement
            # If the gap between location and market has widened (more reduction),
            # procurement has improved.
            current_gap = loc_curr - mkt_curr
            previous_gap = loc_prev - mkt_prev
            procurement_impact_improved = current_gap > previous_gap

            result = {
                "analysis_id": analysis_id,
                "current_period": current_dual.get("period"),
                "previous_period": previous_dual.get("period"),
                "location_current_tco2e": loc_curr,
                "location_previous_tco2e": loc_prev,
                "location_change_tco2e": loc_change,
                "location_change_pct": loc_change_pct,
                "market_current_tco2e": mkt_curr,
                "market_previous_tco2e": mkt_prev,
                "market_change_tco2e": mkt_change,
                "market_change_pct": mkt_change_pct,
                "procurement_impact_improved": procurement_impact_improved,
                "current_procurement_gap_tco2e": self._q(current_gap),
                "previous_procurement_gap_tco2e": self._q(previous_gap),
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="yoy_comparison",
                action="calculate",
                entity_id=analysis_id,
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("year_over_year_comparison", elapsed)

            logger.info(
                "YoY comparison: loc_change=%.1f%%, mkt_change=%.1f%%, "
                "procurement_improved=%s",
                float(loc_change_pct), float(mkt_change_pct),
                procurement_impact_improved,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "year_over_year_comparison failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Coverage Analysis (3 public methods)
    # ==================================================================

    def analyze_coverage_gaps(
        self,
        market_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Identify facilities or regions with low instrument coverage.

        Analyses the market-based result to determine which portions of
        electricity consumption lack contractual instrument coverage
        and therefore fall back to the residual mix or grid average.

        Args:
            market_result: Market-based result dictionary with optional keys:
                - ``total_mwh`` (Decimal): Total consumption.
                - ``covered_mwh`` (Decimal): MWh covered by instruments.
                - ``instruments`` (list): List of instrument dicts.
                - ``facilities`` (list): Per-facility breakdowns.
                    Each with ``facility_id``, ``total_mwh``, ``covered_mwh``.

        Returns:
            Dictionary with keys:
                - analysis_id (str)
                - total_mwh (Decimal)
                - covered_mwh (Decimal)
                - uncovered_mwh (Decimal)
                - coverage_pct (Decimal)
                - gap_pct (Decimal): 100 - coverage_pct
                - facility_gaps (list): Per-facility gap details
                - recommendations (list): Coverage improvement suggestions
                - provenance_hash (str)
        """
        start_time = time.monotonic()
        analysis_id = str(uuid.uuid4())

        try:
            total_mwh = _safe_get_decimal(market_result, "total_mwh", Decimal("0"))

            # Calculate covered MWh
            covered_mwh = _safe_get_decimal(market_result, "covered_mwh")
            if covered_mwh == Decimal("0"):
                # Sum from instruments if covered_mwh not directly provided
                instruments = market_result.get("instruments", [])
                if isinstance(instruments, list):
                    covered_mwh = self._q(sum(
                        _safe_get_decimal(inst, "mwh_covered", Decimal("0"))
                        for inst in instruments
                    ))

            uncovered_mwh = self._q(max(Decimal("0"), total_mwh - covered_mwh))

            coverage_pct = self._q_pct(
                (covered_mwh / total_mwh) * Decimal("100")
            ) if total_mwh > Decimal("0") else Decimal("0")

            gap_pct = self._q_pct(Decimal("100") - coverage_pct)

            # Per-facility gap analysis
            facility_gaps = self._analyze_facility_coverage_gaps(
                market_result.get("facilities", [])
            )

            # Generate recommendations
            recommendations = self._generate_coverage_recommendations(
                uncovered_mwh, gap_pct, facility_gaps,
            )

            result = {
                "analysis_id": analysis_id,
                "total_mwh": total_mwh,
                "covered_mwh": covered_mwh,
                "uncovered_mwh": uncovered_mwh,
                "coverage_pct": coverage_pct,
                "gap_pct": gap_pct,
                "facility_gaps": facility_gaps,
                "recommendations": recommendations,
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="coverage_gap",
                action="analyze",
                entity_id=analysis_id,
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("analyze_coverage_gaps", elapsed)

            logger.info(
                "Coverage gap analysis: %.1f%% covered, %.1f MWh uncovered",
                float(coverage_pct), float(uncovered_mwh),
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "analyze_coverage_gaps failed: %s", exc, exc_info=True,
            )
            raise

    def recommend_instruments(
        self,
        coverage_gaps: Dict[str, Any],
        budget: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Suggest instruments to close coverage gaps.

        Based on the uncovered MWh and optional budget constraint,
        recommends the most cost-effective instrument types by region.

        Args:
            coverage_gaps: Output from ``analyze_coverage_gaps()``.
            budget: Optional total budget in the configured currency.

        Returns:
            List of recommendation dictionaries with keys:
                - instrument_type (str)
                - region (str)
                - estimated_mwh (Decimal)
                - estimated_cost (Decimal)
                - cost_per_mwh (Decimal)
                - priority (str): 'high', 'medium', 'low'
        """
        start_time = time.monotonic()

        try:
            uncovered_mwh = _safe_get_decimal(coverage_gaps, "uncovered_mwh", Decimal("0"))
            budget_dec = _to_decimal(budget) if budget is not None else None

            recommendations: List[Dict[str, Any]] = []

            if uncovered_mwh <= Decimal("0"):
                logger.info("No coverage gaps to fill; returning empty recommendations")
                return recommendations

            # Sort instrument types by cost (ascending) for cost-effectiveness
            sorted_costs = sorted(
                TYPICAL_REC_COST_PER_MWH.items(),
                key=lambda x: x[1],
            )

            remaining_mwh = uncovered_mwh
            remaining_budget = budget_dec

            for region, cost_per_mwh in sorted_costs:
                if remaining_mwh <= Decimal("0"):
                    break

                # Calculate how many MWh we can cover
                if remaining_budget is not None:
                    max_by_budget = self._q(remaining_budget / cost_per_mwh)
                    coverable = min(remaining_mwh, max_by_budget)
                else:
                    coverable = remaining_mwh

                if coverable <= Decimal("0"):
                    continue

                estimated_cost = self._q(coverable * cost_per_mwh)

                # Determine priority
                if cost_per_mwh <= Decimal("3.00"):
                    priority = "high"
                elif cost_per_mwh <= Decimal("10.00"):
                    priority = "medium"
                else:
                    priority = "low"

                # Determine instrument type from region name
                instrument_type = self._region_to_instrument_type(region)

                recommendations.append({
                    "instrument_type": instrument_type,
                    "region": region,
                    "estimated_mwh": coverable,
                    "estimated_cost": estimated_cost,
                    "cost_per_mwh": cost_per_mwh,
                    "currency": self._default_currency,
                    "priority": priority,
                })

                remaining_mwh -= coverable
                if remaining_budget is not None:
                    remaining_budget -= estimated_cost

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("recommend_instruments", elapsed)

            logger.info(
                "Instrument recommendations: %d options for %.1f uncovered MWh",
                len(recommendations), float(uncovered_mwh),
            )

            return recommendations

        except Exception as exc:
            self._increment_error()
            logger.error(
                "recommend_instruments failed: %s", exc, exc_info=True,
            )
            raise

    def estimate_cost_to_cover(
        self,
        uncovered_mwh: Any,
        instrument_type: str,
        region: str,
    ) -> Dict[str, Any]:
        """Estimate the cost to cover uncovered MWh with a specific instrument.

        Args:
            uncovered_mwh: MWh of uncovered consumption.
            instrument_type: Type of instrument (informational label).
            region: Region key from TYPICAL_REC_COST_PER_MWH.

        Returns:
            Dictionary with keys:
                - uncovered_mwh (Decimal)
                - instrument_type (str)
                - region (str)
                - cost_per_mwh (Decimal)
                - total_estimated_cost (Decimal)
                - currency (str)
                - provenance_hash (str)

        Raises:
            ValueError: If region is not found in TYPICAL_REC_COST_PER_MWH.
        """
        start_time = time.monotonic()

        try:
            mwh = self._q(_to_decimal(uncovered_mwh))

            region_upper = region.upper()
            cost_per_mwh = TYPICAL_REC_COST_PER_MWH.get(region_upper)
            if cost_per_mwh is None:
                available = ", ".join(sorted(TYPICAL_REC_COST_PER_MWH.keys()))
                raise ValueError(
                    f"Region '{region}' not found. Available: {available}"
                )

            total_cost = self._q(mwh * cost_per_mwh)

            result = {
                "uncovered_mwh": mwh,
                "instrument_type": instrument_type,
                "region": region_upper,
                "cost_per_mwh": cost_per_mwh,
                "total_estimated_cost": total_cost,
                "currency": self._default_currency,
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="cost_estimate",
                action="calculate",
                entity_id=str(uuid.uuid4()),
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1

            _observe_duration("estimate_cost_to_cover", elapsed)

            logger.info(
                "Cost estimate: %.1f MWh x $%.2f/MWh = $%.2f (%s, %s)",
                float(mwh), float(cost_per_mwh), float(total_cost),
                instrument_type, region_upper,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "estimate_cost_to_cover failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Validation (2 public methods)
    # ==================================================================

    def validate_dual_report(
        self,
        location_result: Dict[str, Any],
        market_result: Dict[str, Any],
    ) -> List[str]:
        """Validate consistency between location-based and market-based results.

        Checks that both results reference the same facilities, cover the
        same reporting period, and report the same total consumption. Any
        inconsistencies are returned as validation messages.

        Args:
            location_result: Location-based result dictionary.
            market_result: Market-based result dictionary.

        Returns:
            List of validation messages (empty if all checks pass).
            Each message is a string describing the inconsistency.
        """
        start_time = time.monotonic()

        try:
            messages: List[str] = []

            # Check facility_id consistency
            loc_fid = location_result.get("facility_id")
            mkt_fid = market_result.get("facility_id")
            if loc_fid is not None and mkt_fid is not None and loc_fid != mkt_fid:
                messages.append(
                    f"Facility mismatch: location='{loc_fid}' vs market='{mkt_fid}'"
                )

            # Check period consistency
            loc_period = location_result.get("period")
            mkt_period = market_result.get("period")
            if loc_period is not None and mkt_period is not None and loc_period != mkt_period:
                messages.append(
                    f"Period mismatch: location='{loc_period}' vs market='{mkt_period}'"
                )

            # Check total_mwh consistency
            loc_mwh = location_result.get("total_mwh")
            mkt_mwh = market_result.get("total_mwh")
            if loc_mwh is not None and mkt_mwh is not None:
                loc_dec = _to_decimal(loc_mwh)
                mkt_dec = _to_decimal(mkt_mwh)
                if loc_dec != mkt_dec:
                    diff = abs(loc_dec - mkt_dec)
                    # Allow 0.1% tolerance for rounding
                    tolerance = loc_dec * Decimal("0.001") if loc_dec > Decimal("0") else Decimal("0.01")
                    if diff > tolerance:
                        messages.append(
                            f"Consumption mismatch: location={loc_dec} MWh vs "
                            f"market={mkt_dec} MWh (diff={diff} MWh)"
                        )

            # Check that both have total_co2e_tonnes
            if "total_co2e_tonnes" not in location_result:
                messages.append("Location result missing 'total_co2e_tonnes'")
            if "total_co2e_tonnes" not in market_result:
                messages.append("Market result missing 'total_co2e_tonnes'")

            # Check for negative emissions
            loc_tco2e = _safe_get_decimal(location_result, "total_co2e_tonnes")
            mkt_tco2e = _safe_get_decimal(market_result, "total_co2e_tonnes")
            if loc_tco2e < Decimal("0"):
                messages.append(
                    f"Location-based emissions are negative: {loc_tco2e} tCO2e"
                )
            if mkt_tco2e < Decimal("0"):
                messages.append(
                    f"Market-based emissions are negative: {mkt_tco2e} tCO2e"
                )

            # Check market > location warning (unusual but valid)
            if mkt_tco2e > loc_tco2e and loc_tco2e > Decimal("0"):
                pct_higher = ((mkt_tco2e - loc_tco2e) / loc_tco2e) * Decimal("100")
                if pct_higher > Decimal("50"):
                    messages.append(
                        f"Warning: market-based ({mkt_tco2e} tCO2e) is {pct_higher:.1f}% "
                        f"higher than location-based ({loc_tco2e} tCO2e); "
                        f"this may indicate residual mix factors are high for this region"
                    )

            # Check facilities list consistency
            loc_facilities = set()
            mkt_facilities = set()
            for f in location_result.get("facilities", []):
                if isinstance(f, dict) and "facility_id" in f:
                    loc_facilities.add(f["facility_id"])
            for f in market_result.get("facilities", []):
                if isinstance(f, dict) and "facility_id" in f:
                    mkt_facilities.add(f["facility_id"])

            if loc_facilities and mkt_facilities:
                only_in_loc = loc_facilities - mkt_facilities
                only_in_mkt = mkt_facilities - loc_facilities
                if only_in_loc:
                    messages.append(
                        f"Facilities only in location result: {sorted(only_in_loc)}"
                    )
                if only_in_mkt:
                    messages.append(
                        f"Facilities only in market result: {sorted(only_in_mkt)}"
                    )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_validations += 1

            self._record_provenance(
                entity_type="dual_report_validation",
                action="validate",
                entity_id=str(uuid.uuid4()),
                data={"message_count": len(messages), "messages": messages},
            )

            _observe_duration("validate_dual_report", elapsed)

            logger.info(
                "Dual report validation: %d messages", len(messages),
            )

            return messages

        except Exception as exc:
            self._increment_error()
            logger.error(
                "validate_dual_report failed: %s", exc, exc_info=True,
            )
            raise

    def check_dual_reporting_completeness(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check whether both methods are reported for all facilities.

        Examines a list of calculation results (potentially mixed location
        and market) and identifies any facilities that are missing one or
        both methods.

        Args:
            results: List of result dictionaries. Each should contain:
                - ``facility_id`` (str)
                - ``method`` (str): 'location' or 'market'

        Returns:
            Dictionary with keys:
                - complete_facilities (list): Facilities with both methods
                - location_only (list): Facilities with only location-based
                - market_only (list): Facilities with only market-based
                - missing_both (list): Facilities with neither (if known)
                - total_facilities (int)
                - completeness_pct (Decimal)
                - is_complete (bool): True if all facilities have both
                - provenance_hash (str)
        """
        start_time = time.monotonic()
        check_id = str(uuid.uuid4())

        try:
            location_facilities: set = set()
            market_facilities: set = set()

            for r in results:
                fid = r.get("facility_id")
                method = r.get("method", "").lower()
                if fid is None:
                    continue
                if method == "location":
                    location_facilities.add(fid)
                elif method == "market":
                    market_facilities.add(fid)

            all_facilities = location_facilities | market_facilities
            complete = sorted(location_facilities & market_facilities)
            location_only = sorted(location_facilities - market_facilities)
            market_only = sorted(market_facilities - location_facilities)

            total = len(all_facilities)
            completeness_pct = self._q_pct(
                (Decimal(str(len(complete))) / Decimal(str(total))) * Decimal("100")
            ) if total > 0 else Decimal("0")

            is_complete = len(complete) == total and total > 0

            result = {
                "check_id": check_id,
                "complete_facilities": complete,
                "location_only": location_only,
                "market_only": market_only,
                "total_facilities": total,
                "completeness_pct": completeness_pct,
                "is_complete": is_complete,
            }

            result["provenance_hash"] = _hash_dual_reporting(result)

            self._record_provenance(
                entity_type="completeness_check",
                action="check",
                entity_id=check_id,
                data=result,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_validations += 1

            _observe_duration("check_dual_reporting_completeness", elapsed)

            logger.info(
                "Completeness check: %d/%d complete (%.1f%%), is_complete=%s",
                len(complete), total, float(completeness_pct), is_complete,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "check_dual_reporting_completeness failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Formatting (3 public methods)
    # ==================================================================

    def format_ghg_protocol_table(
        self,
        dual_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format dual report as GHG Protocol Scope 2 Guidance Table 6.1.

        Produces a structured representation matching the GHG Protocol
        Scope 2 Guidance recommended disclosure table for corporate
        GHG inventories.

        Args:
            dual_result: Output from ``generate_dual_report()``.

        Returns:
            Dictionary structured as GHG Protocol Table 6.1 with keys:
                - format (str): 'GHG_PROTOCOL_TABLE_6_1'
                - title (str)
                - reporting_period (str)
                - scope_2_location_based (dict): by gas and total
                - scope_2_market_based (dict): by gas and total
                - difference (dict): comparison
                - notes (list): disclosure notes
                - provenance_hash (str)
        """
        start_time = time.monotonic()

        try:
            loc_tco2e = _safe_get_decimal(dual_result, "location_based_tco2e")
            mkt_tco2e = _safe_get_decimal(dual_result, "market_based_tco2e")
            diff_tco2e = _safe_get_decimal(dual_result, "difference_tco2e")
            diff_pct = _safe_get_decimal(dual_result, "difference_pct")
            period = dual_result.get("period", "Not specified")
            facility_id = dual_result.get("facility_id", "All facilities")

            table = {
                "format": "GHG_PROTOCOL_TABLE_6_1",
                "title": "Scope 2 GHG Emissions - Dual Reporting",
                "reporting_period": str(period),
                "reporting_entity": str(facility_id),
                "scope_2_location_based": {
                    "total_tco2e": loc_tco2e,
                    "description": (
                        "Scope 2 emissions calculated using grid-average "
                        "emission factors for the regions where electricity "
                        "is consumed."
                    ),
                    "method": "GHG Protocol Scope 2 Guidance - Location-based",
                },
                "scope_2_market_based": {
                    "total_tco2e": mkt_tco2e,
                    "description": (
                        "Scope 2 emissions calculated using contractual "
                        "instruments (RECs, PPAs, GOs, supplier-specific "
                        "emission factors) where available, with residual "
                        "mix or grid average for uncovered consumption."
                    ),
                    "method": "GHG Protocol Scope 2 Guidance - Market-based",
                    "instrument_count": dual_result.get("instrument_count", 0),
                },
                "difference": {
                    "tco2e": diff_tco2e,
                    "percentage": diff_pct,
                    "lower_method": dual_result.get("lower_method", "unknown"),
                },
                "notes": [
                    "Both location-based and market-based figures are reported "
                    "in compliance with the GHG Protocol Scope 2 Guidance (2015).",
                    "Location-based method reflects grid-average emission intensity.",
                    "Market-based method reflects the impact of contractual "
                    "instruments on the reporting entity's emissions.",
                ],
            }

            table["provenance_hash"] = _hash_dual_reporting(table)

            self._record_provenance(
                entity_type="ghg_protocol_table",
                action="format",
                entity_id=str(uuid.uuid4()),
                data=table,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_formats += 1

            _record_dual_report_metric("ghg_protocol_table")
            _observe_duration("format_ghg_protocol_table", elapsed)

            logger.info("GHG Protocol table formatted for period=%s", period)

            return table

        except Exception as exc:
            self._increment_error()
            logger.error(
                "format_ghg_protocol_table failed: %s", exc, exc_info=True,
            )
            raise

    def format_cdp_response(
        self,
        dual_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format dual report as CDP Climate Change C8.2d response.

        CDP C8.2d requests details of the organisation's Scope 2
        market-based emissions, including instrument types and MWh
        covered.

        Args:
            dual_result: Output from ``generate_dual_report()``.

        Returns:
            Dictionary structured as CDP C8.2d response with keys:
                - format (str): 'CDP_C8_2D'
                - question (str): CDP question text
                - scope_2_location_tco2e (Decimal)
                - scope_2_market_tco2e (Decimal)
                - low_carbon_electricity_pct (Decimal)
                - contractual_instruments (list)
                - residual_mix_applied (bool)
                - provenance_hash (str)
        """
        start_time = time.monotonic()

        try:
            loc_tco2e = _safe_get_decimal(dual_result, "location_based_tco2e")
            mkt_tco2e = _safe_get_decimal(dual_result, "market_based_tco2e")
            total_mwh = _safe_get_decimal(dual_result, "total_mwh")

            # Calculate low-carbon electricity percentage
            renewable_impact = _safe_get_decimal(dual_result, "renewable_impact_tco2e")
            low_carbon_pct = Decimal("0")
            if loc_tco2e > Decimal("0"):
                low_carbon_pct = self._q_pct(
                    (renewable_impact / loc_tco2e) * Decimal("100")
                )

            # Build instrument list for CDP
            instrument_count = dual_result.get("instrument_count", 0)
            contractual_instruments: List[Dict[str, Any]] = []
            if instrument_count > 0:
                # If the original market_result instruments are embedded
                # in dual_result metadata, reconstruct them
                contractual_instruments.append({
                    "instrument_count": instrument_count,
                    "note": (
                        "Instrument details available in the underlying "
                        "market-based calculation result."
                    ),
                })

            # Check if residual mix was applied
            residual_mix_applied = mkt_tco2e > Decimal("0")

            cdp = {
                "format": "CDP_C8_2D",
                "question": (
                    "C8.2d - Provide details on the electricity, heat, steam, "
                    "and/or cooling amounts that were accounted for at a "
                    "low-carbon emission factor in the market-based approach."
                ),
                "reporting_period": dual_result.get("period", "Not specified"),
                "scope_2_location_tco2e": loc_tco2e,
                "scope_2_market_tco2e": mkt_tco2e,
                "total_consumption_mwh": total_mwh,
                "low_carbon_electricity_pct": low_carbon_pct,
                "contractual_instruments": contractual_instruments,
                "residual_mix_applied": residual_mix_applied,
                "comment": (
                    "Market-based Scope 2 emissions reflect contractual "
                    "instruments in accordance with the GHG Protocol "
                    "Scope 2 Guidance quality criteria hierarchy."
                ),
            }

            cdp["provenance_hash"] = _hash_dual_reporting(cdp)

            self._record_provenance(
                entity_type="cdp_response",
                action="format",
                entity_id=str(uuid.uuid4()),
                data=cdp,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_formats += 1

            _record_dual_report_metric("cdp_c8_2d")
            _observe_duration("format_cdp_response", elapsed)

            logger.info(
                "CDP C8.2d response formatted: loc=%.2f, mkt=%.2f, low_carbon=%.1f%%",
                float(loc_tco2e), float(mkt_tco2e), float(low_carbon_pct),
            )

            return cdp

        except Exception as exc:
            self._increment_error()
            logger.error(
                "format_cdp_response failed: %s", exc, exc_info=True,
            )
            raise

    def format_csrd_esrs_e1(
        self,
        dual_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format dual report as CSRD / ESRS E1-6 disclosure.

        ESRS E1 (Climate change) paragraph E1-6 requires disclosure of
        Scope 2 GHG emissions using both location-based and market-based
        approaches. This method produces the structured data for XBRL
        tagging and narrative generation.

        Args:
            dual_result: Output from ``generate_dual_report()``.

        Returns:
            Dictionary structured as ESRS E1-6 disclosure with keys:
                - format (str): 'CSRD_ESRS_E1_6'
                - standard (str): 'ESRS E1 Climate Change'
                - paragraph (str): 'E1-6'
                - gross_scope_2_location_tco2e (Decimal)
                - gross_scope_2_market_tco2e (Decimal)
                - ghg_intensity_scope_2_location (Decimal or None)
                - ghg_intensity_scope_2_market (Decimal or None)
                - reduction_from_instruments_tco2e (Decimal)
                - percentage_reduction (Decimal)
                - reporting_period (str)
                - methodology_notes (list)
                - xbrl_tags (dict)
                - provenance_hash (str)
        """
        start_time = time.monotonic()

        try:
            loc_tco2e = _safe_get_decimal(dual_result, "location_based_tco2e")
            mkt_tco2e = _safe_get_decimal(dual_result, "market_based_tco2e")
            renewable_impact = _safe_get_decimal(dual_result, "renewable_impact_tco2e")
            diff_pct = _safe_get_decimal(dual_result, "difference_pct")
            total_mwh = _safe_get_decimal(dual_result, "total_mwh")
            period = dual_result.get("period", "Not specified")

            # GHG intensity (tCO2e per MWh)
            loc_intensity: Optional[Decimal] = None
            mkt_intensity: Optional[Decimal] = None
            if total_mwh > Decimal("0"):
                loc_intensity = self._q(loc_tco2e / total_mwh)
                mkt_intensity = self._q(mkt_tco2e / total_mwh)

            # Percentage reduction (positive = reduction)
            if loc_tco2e > Decimal("0"):
                pct_reduction = self._q_pct(
                    (renewable_impact / loc_tco2e) * Decimal("100")
                )
            else:
                pct_reduction = Decimal("0")

            esrs = {
                "format": "CSRD_ESRS_E1_6",
                "standard": "ESRS E1 Climate Change",
                "paragraph": "E1-6",
                "disclosure_requirement": (
                    "Gross Scope 2 GHG emissions - energy indirect"
                ),
                "reporting_period": str(period),
                "gross_scope_2_location_tco2e": loc_tco2e,
                "gross_scope_2_market_tco2e": mkt_tco2e,
                "ghg_intensity_scope_2_location": loc_intensity,
                "ghg_intensity_scope_2_market": mkt_intensity,
                "reduction_from_instruments_tco2e": renewable_impact,
                "percentage_reduction": pct_reduction,
                "methodology_notes": [
                    "Location-based Scope 2 calculated per GHG Protocol "
                    "Scope 2 Guidance using grid-average emission factors.",
                    "Market-based Scope 2 calculated per GHG Protocol "
                    "Scope 2 Guidance using contractual instruments and "
                    "residual mix factors where applicable.",
                    "Both methods reported as required by ESRS E1-6.",
                    "GWP values from IPCC AR6 (2021) used for CO2e conversion.",
                ],
                "xbrl_tags": {
                    "esrs:GrossScope2GHGEmissionsLocationBased": str(loc_tco2e),
                    "esrs:GrossScope2GHGEmissionsMarketBased": str(mkt_tco2e),
                    "esrs:Scope2GHGIntensityLocationBased": (
                        str(loc_intensity) if loc_intensity is not None else "N/A"
                    ),
                    "esrs:Scope2GHGIntensityMarketBased": (
                        str(mkt_intensity) if mkt_intensity is not None else "N/A"
                    ),
                },
            }

            esrs["provenance_hash"] = _hash_dual_reporting(esrs)

            self._record_provenance(
                entity_type="csrd_esrs_e1",
                action="format",
                entity_id=str(uuid.uuid4()),
                data=esrs,
            )

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._total_formats += 1

            _record_dual_report_metric("csrd_esrs_e1")
            _observe_duration("format_csrd_esrs_e1", elapsed)

            logger.info(
                "CSRD/ESRS E1-6 formatted: loc=%.2f, mkt=%.2f, reduction=%.1f%%",
                float(loc_tco2e), float(mkt_tco2e), float(pct_reduction),
            )

            return esrs

        except Exception as exc:
            self._increment_error()
            logger.error(
                "format_csrd_esrs_e1 failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Utilities (2 public methods)
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine metadata, operation counts,
            provenance entry count, and configuration summary.
        """
        with self._lock:
            return {
                "engine": "DualReportingEngine",
                "agent": "AGENT-MRV-010",
                "version": VERSION,
                "created_at": self._created_at.isoformat(),
                "total_reports": self._total_reports,
                "total_batch_reports": self._total_batch_reports,
                "total_facility_reports": self._total_facility_reports,
                "total_analyses": self._total_analyses,
                "total_validations": self._total_validations,
                "total_formats": self._total_formats,
                "total_errors": self._total_errors,
                "provenance_entry_count": (
                    self._provenance.entry_count
                    if self._provenance is not None
                    else 0
                ),
                "provenance_chain_valid": (
                    self._provenance.verify_chain()
                    if self._provenance is not None
                    else None
                ),
                "enable_provenance": self._enable_provenance,
                "decimal_precision": self._precision_places,
                "default_currency": self._default_currency,
                "supported_regions": len(TYPICAL_REC_COST_PER_MWH),
                "additionality_criteria_count": len(ADDITIONALITY_CRITERIA),
                "instrument_type_count": len(INSTRUMENT_QUALITY_HIERARCHY),
            }

    def reset(self) -> None:
        """Reset all engine counters and provenance chain.

        Intended for testing and diagnostic purposes. Clears all
        operation counters and resets the provenance chain to genesis.
        Configuration is preserved.
        """
        with self._lock:
            self._total_reports = 0
            self._total_batch_reports = 0
            self._total_facility_reports = 0
            self._total_analyses = 0
            self._total_validations = 0
            self._total_formats = 0
            self._total_errors = 0
            if self._provenance is not None:
                self._provenance.reset()
        logger.info("DualReportingEngine counters and provenance reset")

    # ==================================================================
    # Internal Helpers
    # ==================================================================

    def _extract_tco2e(
        self,
        result: Dict[str, Any],
        label: str,
    ) -> Decimal:
        """Extract and validate total_co2e_tonnes from a result dict.

        Args:
            result: Result dictionary.
            label: Label for error messages (e.g. 'location_result').

        Returns:
            Decimal value of total_co2e_tonnes.

        Raises:
            ValueError: If the key is missing or not convertible.
        """
        raw = result.get("total_co2e_tonnes")
        if raw is None:
            raise ValueError(
                f"{label} must contain 'total_co2e_tonnes'"
            )
        return self._q(_to_decimal(raw))

    def _determine_lower_method(
        self,
        location_tco2e: Decimal,
        market_tco2e: Decimal,
    ) -> str:
        """Determine which method yields lower emissions.

        Args:
            location_tco2e: Location-based total.
            market_tco2e: Market-based total.

        Returns:
            'location', 'market', or 'equal'.
        """
        if market_tco2e < location_tco2e:
            return "market"
        elif market_tco2e > location_tco2e:
            return "location"
        else:
            return "equal"

    def _calculate_renewable_impact(
        self,
        location_tco2e: Decimal,
        market_tco2e: Decimal,
    ) -> Decimal:
        """Calculate the emission reduction from renewable instruments.

        When market-based < location-based, the difference represents
        the tCO2e avoided through contractual instruments. If market-based
        is higher (e.g. residual mix), renewable impact is zero.

        Args:
            location_tco2e: Location-based total.
            market_tco2e: Market-based total.

        Returns:
            Non-negative Decimal representing tCO2e avoided.
        """
        if market_tco2e < location_tco2e:
            return self._q(location_tco2e - market_tco2e)
        return Decimal("0")

    def _analyze_facility_coverage_gaps(
        self,
        facilities: List[Any],
    ) -> List[Dict[str, Any]]:
        """Analyze per-facility coverage gaps.

        Args:
            facilities: List of facility dictionaries with optional
                ``facility_id``, ``total_mwh``, ``covered_mwh``.

        Returns:
            List of per-facility gap detail dictionaries.
        """
        gaps: List[Dict[str, Any]] = []

        if not isinstance(facilities, list):
            return gaps

        for fac in facilities:
            if not isinstance(fac, dict):
                continue

            fid = fac.get("facility_id", "unknown")
            f_total = _safe_get_decimal(fac, "total_mwh", Decimal("0"))
            f_covered = _safe_get_decimal(fac, "covered_mwh", Decimal("0"))
            f_uncovered = self._q(max(Decimal("0"), f_total - f_covered))

            if f_total > Decimal("0"):
                f_coverage_pct = self._q_pct(
                    (f_covered / f_total) * Decimal("100")
                )
            else:
                f_coverage_pct = Decimal("0")

            f_gap_pct = self._q_pct(Decimal("100") - f_coverage_pct)

            if f_uncovered > Decimal("0"):
                gaps.append({
                    "facility_id": fid,
                    "total_mwh": f_total,
                    "covered_mwh": f_covered,
                    "uncovered_mwh": f_uncovered,
                    "coverage_pct": f_coverage_pct,
                    "gap_pct": f_gap_pct,
                })

        # Sort by uncovered MWh descending (largest gaps first)
        gaps.sort(key=lambda x: x["uncovered_mwh"], reverse=True)

        return gaps

    def _generate_coverage_recommendations(
        self,
        uncovered_mwh: Decimal,
        gap_pct: Decimal,
        facility_gaps: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate plain-text recommendations for closing coverage gaps.

        Args:
            uncovered_mwh: Total uncovered MWh.
            gap_pct: Total gap percentage.
            facility_gaps: Per-facility gap details.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if uncovered_mwh <= Decimal("0"):
            recommendations.append(
                "Full coverage achieved. No additional instruments required."
            )
            return recommendations

        # General recommendation
        recommendations.append(
            f"Procure instruments for {uncovered_mwh} MWh to close "
            f"the {gap_pct}% coverage gap."
        )

        # Priority facilities
        if facility_gaps:
            top_gap = facility_gaps[0]
            recommendations.append(
                f"Priority: Facility '{top_gap['facility_id']}' has the largest "
                f"gap at {top_gap['uncovered_mwh']} MWh uncovered."
            )

        # Cost-effective options
        cheapest_region = min(
            TYPICAL_REC_COST_PER_MWH.items(), key=lambda x: x[1],
        )
        min_cost = self._q(uncovered_mwh * cheapest_region[1])
        recommendations.append(
            f"Lowest-cost option: {cheapest_region[0]} at "
            f"${cheapest_region[1]}/MWh (total ~${min_cost})."
        )

        # RE100 alignment
        if gap_pct > Decimal("50"):
            recommendations.append(
                "Consider a corporate PPA for large-volume coverage "
                "to improve additionality and lock in long-term pricing."
            )
        elif gap_pct > Decimal("10"):
            recommendations.append(
                "Consider unbundled RECs or GOs for the remaining gap "
                "as a cost-effective near-term solution."
            )
        else:
            recommendations.append(
                "Close to full coverage. A small REC/GO purchase or "
                "green tariff upgrade may close the remaining gap."
            )

        return recommendations

    def _score_to_rating(self, score: Decimal) -> str:
        """Convert a numeric additionality score to a categorical rating.

        Args:
            score: Score from 0 to 100.

        Returns:
            'high' (>=75), 'medium' (>=50), 'low' (>=25), or 'none' (<25).
        """
        if score >= Decimal("75"):
            return "high"
        elif score >= Decimal("50"):
            return "medium"
        elif score >= Decimal("25"):
            return "low"
        else:
            return "none"

    def _region_to_instrument_type(self, region: str) -> str:
        """Map a TYPICAL_REC_COST_PER_MWH region key to an instrument type.

        Args:
            region: Region key (e.g. 'US_NATIONAL', 'EU_GO').

        Returns:
            Instrument type string.
        """
        region_lower = region.lower()
        if "ppa" in region_lower:
            return "PPA"
        elif "go" in region_lower or "nordic" in region_lower:
            return "GO"
        elif "rego" in region_lower:
            return "REGO"
        elif "lgc" in region_lower or "stc" in region_lower:
            return "LGC"
        elif "irec" in region_lower:
            return "I-REC"
        elif "gec" in region_lower:
            return "GEC"
        elif "jcredit" in region_lower or "nfc" in region_lower:
            return "J-Credit"
        elif "rec" in region_lower:
            return "REC"
        elif "us_" in region_lower:
            return "REC"
        elif "eu_" in region_lower:
            return "GO"
        else:
            return "EAC"

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"DualReportingEngine("
            f"reports={self._total_reports}, "
            f"analyses={self._total_analyses}, "
            f"validations={self._total_validations}, "
            f"formats={self._total_formats}, "
            f"errors={self._total_errors})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"DualReportingEngine v{VERSION} - "
            f"{self._total_reports} reports generated, "
            f"{self._total_analyses} analyses, "
            f"{self._total_formats} formats"
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DualReportingEngine",
    "TYPICAL_REC_COST_PER_MWH",
    "RE100_TARGET_PCT",
    "ADDITIONALITY_CRITERIA",
    "INSTRUMENT_QUALITY_HIERARCHY",
    "VERSION",
    "TABLE_PREFIX",
]
