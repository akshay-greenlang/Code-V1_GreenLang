# -*- coding: utf-8 -*-
"""
DiscrepancyAnalyzerEngine - Discrepancy Analysis & Waterfall Decomposition (Engine 2 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Core analysis engine that calculates, classifies, and decomposes discrepancies
between location-based and market-based Scope 2 emission totals. Performs
waterfall decomposition to attribute every tonne of discrepancy to exactly
one of eight drivers.

Discrepancy Types (8):
    1. REC_GO_IMPACT: RECs/GOs zero-out market-based for covered MWh
    2. RESIDUAL_MIX_UPLIFT: Residual mix EFs exceed grid average EFs
    3. SUPPLIER_EF_DELTA: Supplier-specific EF differs from grid average
    4. GEOGRAPHIC_MISMATCH: Location region differs from instrument region
    5. TEMPORAL_MISMATCH: Instrument vintage misaligns with consumption period
    6. PARTIAL_COVERAGE: Instruments cover only a portion of consumption
    7. STEAM_HEAT_METHOD: Methodological divergence for steam/heating
    8. GRID_UPDATE_TIMING: Different grid EF vintages between methods

Analysis Levels (4):
    - Total: Aggregate discrepancy across all energy types and facilities
    - Energy-type: Per energy type (electricity, steam, heating, cooling)
    - Facility: Per facility across all energy types
    - Instrument: Per contractual instrument type

Key Metrics:
    - PIF (Procurement Impact Factor): (Location - Market) / Location x 100
    - Discrepancy Percentage: |Location - Market| / max(Location, Market) x 100
    - Waterfall Balance: Total_Discrepancy = Sum(all 8 driver components)

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal arithmetic.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Thread-safe singleton using threading.RLock with _instance/_initialized
    pattern. Per-calculation state is created fresh for each method call.
    Shared counters protected by reentrant lock.

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.discrepancy_analyzer import (
    ...     DiscrepancyAnalyzerEngine,
    ... )
    >>> engine = DiscrepancyAnalyzerEngine()
    >>> report = engine.analyze_discrepancies(workspace)
    >>> print(report.waterfall.total_discrepancy_tco2e)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["DiscrepancyAnalyzerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful degradation if sibling modules unavailable
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
        EnergyType,
        Scope2Method,
        UpstreamAgent,
        DiscrepancyType,
        DiscrepancyDirection,
        MaterialityLevel,
        FlagType,
        FlagSeverity,
        ReconciliationStatus,
        EFHierarchyPriority,
        MATERIALITY_THRESHOLDS,
        RESIDUAL_MIX_FACTORS,
        EF_HIERARCHY_QUALITY_SCORES,
        UpstreamResult,
        ReconciliationWorkspace,
        Discrepancy,
        WaterfallItem,
        WaterfallDecomposition,
        DiscrepancyReport,
        Flag,
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        DECIMAL_INF,
        MAX_DISCREPANCIES,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning(
        "greenlang.agents.mrv.dual_reporting_reconciliation.models not available; "
        "DiscrepancyAnalyzerEngine will operate in degraded mode"
    )

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        DualReportingReconciliationConfig,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    DualReportingReconciliationConfig = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        DualReportingReconciliationMetrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    DualReportingReconciliationMetrics = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        DualReportingProvenanceTracker,
        ProvenanceStage,
        hash_discrepancy as _hash_discrepancy,
        hash_waterfall as _hash_waterfall,
        hash_waterfall_item as _hash_waterfall_item,
        hash_materiality_classification as _hash_materiality_classification,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    DualReportingProvenanceTracker = None  # type: ignore[misc,assignment]
    ProvenanceStage = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

_ENGINE_ID: str = "discrepancy-analyzer-engine"
_ENGINE_VERSION: str = "1.0.0"
_ENGINE_COMPONENT: str = "engine-2-discrepancy-analyzer"

# Decimal constants for zero-hallucination arithmetic
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ROUNDING_TOLERANCE = Decimal("0.01")  # tCO2e equality tolerance
_WATERFALL_BALANCE_TOLERANCE = Decimal("0.1")  # tCO2e balance tolerance

# Flag code constants
_FLAG_PREFIX = "DRR-DA"
_FLAG_CODE_MATERIAL_DISCREPANCY = f"{_FLAG_PREFIX}-W-001"
_FLAG_CODE_SIGNIFICANT_DISCREPANCY = f"{_FLAG_PREFIX}-W-002"
_FLAG_CODE_EXTREME_DISCREPANCY = f"{_FLAG_PREFIX}-E-001"
_FLAG_CODE_WATERFALL_IMBALANCE = f"{_FLAG_PREFIX}-E-002"
_FLAG_CODE_NO_DISCREPANCY = f"{_FLAG_PREFIX}-I-001"
_FLAG_CODE_MARKET_HIGHER = f"{_FLAG_PREFIX}-I-002"
_FLAG_CODE_PARTIAL_COVERAGE = f"{_FLAG_PREFIX}-W-003"
_FLAG_CODE_GEOGRAPHIC_MISMATCH = f"{_FLAG_PREFIX}-W-004"
_FLAG_CODE_TEMPORAL_MISMATCH = f"{_FLAG_PREFIX}-W-005"
_FLAG_CODE_MISSING_DATA = f"{_FLAG_PREFIX}-E-003"
_FLAG_CODE_HIGH_PIF = f"{_FLAG_PREFIX}-I-003"
_FLAG_CODE_NEGATIVE_PIF = f"{_FLAG_PREFIX}-W-006"
_FLAG_CODE_RESIDUAL_UPLIFT = f"{_FLAG_PREFIX}-I-004"
_FLAG_CODE_STEAM_HEAT_DIVERGENCE = f"{_FLAG_PREFIX}-W-007"
_FLAG_CODE_GRID_TIMING = f"{_FLAG_PREFIX}-W-008"


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert.
        default: Default Decimal to return on failure.

    Returns:
        Converted Decimal or the default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (supports Pydantic models, dicts, primitives).

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _round_decimal(value: Decimal, places: int = 8) -> Decimal:
    """Round a Decimal to the specified number of places.

    Args:
        value: Decimal value to round.
        places: Number of decimal places.

    Returns:
        Rounded Decimal.
    """
    quantize_str = "0." + "0" * places
    try:
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return value


# ===========================================================================
# Discrepancy description templates
# ===========================================================================

_DISCREPANCY_DESCRIPTIONS: Dict[str, str] = {
    "REC_GO_IMPACT": (
        "Renewable Energy Certificates (RECs) or Guarantees of Origin (GOs) "
        "reduce market-based emissions to zero for {covered_mwh:.2f} MWh of "
        "covered consumption, while location-based retains the grid average "
        "emission factor for the same quantity. Net impact: {impact_tco2e:.4f} "
        "tCO2e reduction in market-based total."
    ),
    "RESIDUAL_MIX_UPLIFT": (
        "Residual mix emission factors ({residual_ef:.6f} tCO2e/MWh) exceed "
        "grid average factors ({grid_ef:.6f} tCO2e/MWh) for region {region} "
        "by a ratio of {ratio:.3f}x, increasing market-based emissions for "
        "uncovered consumption. Net impact: {impact_tco2e:.4f} tCO2e uplift."
    ),
    "SUPPLIER_EF_DELTA": (
        "Supplier-specific emission factor ({supplier_ef:.6f} tCO2e/MWh) "
        "differs from grid average ({grid_ef:.6f} tCO2e/MWh) by "
        "{delta_pct:.2f}%. Net impact: {impact_tco2e:.4f} tCO2e."
    ),
    "GEOGRAPHIC_MISMATCH": (
        "Facility {facility_id} consumes energy in region {consumption_region} "
        "(grid EF {consumption_ef:.6f}) but holds contractual instruments from "
        "region {instrument_region} (EF {instrument_ef:.6f}). Geographic "
        "mismatch contributes {impact_tco2e:.4f} tCO2e to the discrepancy."
    ),
    "TEMPORAL_MISMATCH": (
        "Contractual instruments have vintage year {instrument_vintage} while "
        "consumption occurs in period {consumption_period}. Temporal mismatch "
        "between emission factor vintages contributes {impact_tco2e:.4f} tCO2e."
    ),
    "PARTIAL_COVERAGE": (
        "Contractual instruments cover {coverage_pct:.1f}% of total energy "
        "consumption ({covered_mwh:.2f} of {total_mwh:.2f} MWh). The uncovered "
        "portion ({uncovered_mwh:.2f} MWh) uses residual mix factors instead of "
        "grid average, contributing {impact_tco2e:.4f} tCO2e to the discrepancy."
    ),
    "STEAM_HEAT_METHOD": (
        "Steam and heating emissions use different methodological approaches "
        "for location-based ({location_method}) versus market-based "
        "({market_method}). Methodological divergence contributes "
        "{impact_tco2e:.4f} tCO2e to the discrepancy."
    ),
    "GRID_UPDATE_TIMING": (
        "Grid emission factors were updated during the reporting period. "
        "Location-based uses {location_ef_vintage} vintage while market-based "
        "uses {market_ef_vintage} vintage. Timing difference contributes "
        "{impact_tco2e:.4f} tCO2e to the discrepancy."
    ),
}

_DISCREPANCY_RECOMMENDATIONS: Dict[str, str] = {
    "REC_GO_IMPACT": (
        "No action required. RECs/GOs are functioning as intended, reducing "
        "market-based emissions. Ensure all instruments are properly retired "
        "and documented for audit trail completeness."
    ),
    "RESIDUAL_MIX_UPLIFT": (
        "Consider procuring additional RECs/GOs to cover consumption currently "
        "subject to residual mix factors. Alternatively, negotiate supplier-"
        "specific emission rates for uncovered portions."
    ),
    "SUPPLIER_EF_DELTA": (
        "Review supplier-specific emission factor documentation. Ensure the "
        "supplier certificate is current and properly verified. Consider "
        "requesting third-party certification to improve quality score."
    ),
    "GEOGRAPHIC_MISMATCH": (
        "Where possible, procure contractual instruments from the same grid "
        "region as energy consumption. Review GHG Protocol Scope 2 Guidance "
        "Chapter 7 on geographic matching requirements."
    ),
    "TEMPORAL_MISMATCH": (
        "Align contractual instrument vintages with consumption reporting "
        "periods. Review GHG Protocol Scope 2 Guidance Chapter 7 on temporal "
        "matching requirements. Consider monthly or quarterly instrument "
        "procurement to reduce vintage mismatch."
    ),
    "PARTIAL_COVERAGE": (
        "Increase contractual instrument coverage toward 100% of consumption. "
        "Prioritize high-emission facilities for additional procurement. "
        "Consider power purchase agreements (PPAs) for baseload coverage."
    ),
    "STEAM_HEAT_METHOD": (
        "Review methodological consistency between location-based and market-"
        "based approaches for steam and heating. Ensure CHP allocation "
        "methodology is applied consistently across both methods."
    ),
    "GRID_UPDATE_TIMING": (
        "Use consistent grid emission factor vintages for both methods within "
        "a reporting period. Document any mid-period updates and their impact "
        "on reported totals."
    ),
}


# ===========================================================================
# DiscrepancyAnalyzerEngine
# ===========================================================================


class DiscrepancyAnalyzerEngine:
    """Engine 2: Analyzes discrepancies between location/market Scope 2 totals.

    Thread-safe singleton. Calculates discrepancies at total, energy-type,
    facility, and instrument levels. Classifies 8 discrepancy types and
    performs waterfall decomposition to attribute every tonne of difference
    to exactly one driver.

    The engine operates purely on deterministic Decimal arithmetic with no
    LLM calls in any calculation path. All results include SHA-256 provenance
    hashes for complete audit trail integrity.

    Attributes:
        _config: Agent configuration singleton.
        _metrics: Prometheus metrics singleton.
        _lock: Reentrant lock for thread-safe counter updates.
        _total_analyses: Counter of total analyses performed.
        _total_discrepancies_found: Counter of total discrepancies identified.
        _decimal_places: Number of decimal places for calculations.

    Example:
        >>> engine = DiscrepancyAnalyzerEngine()
        >>> report = engine.analyze_discrepancies(workspace)
        >>> assert report.waterfall is not None
        >>> for d in report.discrepancies:
        ...     print(d.discrepancy_type, d.absolute_tco2e)
    """

    _instance: Optional[DiscrepancyAnalyzerEngine] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> DiscrepancyAnalyzerEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with threading.RLock for thread safety.

        Returns:
            The singleton DiscrepancyAnalyzerEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the DiscrepancyAnalyzerEngine.

        Guarded by the _initialized flag so repeated calls do not
        re-initialize state. Loads configuration, sets up metrics,
        and initializes counters.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._init_engine()
            self.__class__._initialized = True
            logger.info(
                "DiscrepancyAnalyzerEngine initialized: engine_id=%s, "
                "version=%s, decimal_places=%d",
                _ENGINE_ID,
                _ENGINE_VERSION,
                self._decimal_places,
            )

    def _init_engine(self) -> None:
        """Perform actual initialization of engine components."""
        # Configuration
        self._config: Optional[Any] = None
        if _CONFIG_AVAILABLE and DualReportingReconciliationConfig is not None:
            try:
                self._config = DualReportingReconciliationConfig()
            except Exception as exc:
                logger.warning(
                    "Failed to load DualReportingReconciliationConfig: %s", exc,
                )

        # Metrics
        self._metrics: Optional[Any] = None
        if _METRICS_AVAILABLE and DualReportingReconciliationMetrics is not None:
            try:
                self._metrics = DualReportingReconciliationMetrics()
            except Exception as exc:
                logger.warning(
                    "Failed to load DualReportingReconciliationMetrics: %s", exc,
                )

        # Decimal places from config or default
        self._decimal_places: int = 8
        if self._config is not None and hasattr(self._config, "decimal_places"):
            self._decimal_places = int(self._config.decimal_places)

        # Materiality thresholds from config or defaults
        self._immaterial_threshold = Decimal("5")
        self._minor_threshold = Decimal("15")
        self._material_threshold = Decimal("50")
        self._significant_threshold = Decimal("100")
        if self._config is not None:
            if hasattr(self._config, "immaterial_threshold"):
                self._immaterial_threshold = _safe_decimal(
                    self._config.immaterial_threshold, Decimal("5"),
                )
            if hasattr(self._config, "minor_threshold"):
                self._minor_threshold = _safe_decimal(
                    self._config.minor_threshold, Decimal("15"),
                )
            if hasattr(self._config, "material_threshold"):
                self._material_threshold = _safe_decimal(
                    self._config.material_threshold, Decimal("50"),
                )
            if hasattr(self._config, "significant_threshold"):
                self._significant_threshold = _safe_decimal(
                    self._config.significant_threshold, Decimal("100"),
                )

        # Operation counters (protected by _lock)
        self._total_analyses: int = 0
        self._total_discrepancies_found: int = 0
        self._total_waterfall_builds: int = 0
        self._total_flags_generated: int = 0
        self._created_at: datetime = _utcnow()

    # ==================================================================
    # Public API: Primary analysis method
    # ==================================================================

    def analyze_discrepancies(
        self,
        workspace: ReconciliationWorkspace,
    ) -> DiscrepancyReport:
        """Perform complete discrepancy analysis on a reconciliation workspace.

        This is the main entry point for Engine 2. It performs:
        1. Total-level discrepancy calculation
        2. Energy-type level discrepancy calculation
        3. Facility-level discrepancy calculation
        4. Instrument-level discrepancy calculation
        5. Discrepancy classification and materiality assessment
        6. Waterfall decomposition
        7. Flag generation

        Args:
            workspace: ReconciliationWorkspace populated with upstream results
                and aggregate totals from Engine 1 (ResultCollectorEngine).

        Returns:
            DiscrepancyReport containing all identified discrepancies,
            waterfall decomposition, materiality summary, and flags.

        Raises:
            ValueError: If workspace is None or lacks required data.
            RuntimeError: If analysis fails due to internal error.
        """
        start_time = time.monotonic()
        reconciliation_id = workspace.reconciliation_id

        logger.info(
            "Starting discrepancy analysis for reconciliation %s: "
            "location=%.4f tCO2e, market=%.4f tCO2e",
            reconciliation_id,
            float(workspace.total_location_tco2e),
            float(workspace.total_market_tco2e),
        )

        try:
            self._validate_workspace(workspace)

            # Step 1: Calculate total discrepancy
            total_discrepancy = self.calculate_total_discrepancy(workspace)

            # Step 2: Calculate energy-type discrepancies
            energy_type_discrepancies = (
                self.calculate_energy_type_discrepancies(workspace)
            )

            # Step 3: Calculate facility discrepancies
            facility_discrepancies = (
                self.calculate_facility_discrepancies(workspace)
            )

            # Step 4: Calculate instrument discrepancies
            instrument_discrepancies = (
                self.calculate_instrument_discrepancies(workspace)
            )

            # Step 5: Combine all discrepancies
            all_discrepancies: List[Discrepancy] = []
            if total_discrepancy is not None:
                all_discrepancies.append(total_discrepancy)
            all_discrepancies.extend(energy_type_discrepancies)
            all_discrepancies.extend(facility_discrepancies)
            all_discrepancies.extend(instrument_discrepancies)

            # Enforce maximum
            if len(all_discrepancies) > MAX_DISCREPANCIES:
                logger.warning(
                    "Discrepancy count %d exceeds maximum %d; truncating",
                    len(all_discrepancies),
                    MAX_DISCREPANCIES,
                )
                all_discrepancies = all_discrepancies[:MAX_DISCREPANCIES]

            # Step 6: Build waterfall decomposition
            waterfall = self.build_waterfall(workspace, all_discrepancies)

            # Step 7: Generate flags
            flags = self.generate_discrepancy_flags(all_discrepancies)

            # Step 8: Build materiality summary
            materiality_summary = self._build_materiality_summary(
                all_discrepancies,
            )

            # Step 9: Assemble report
            report = DiscrepancyReport(
                reconciliation_id=reconciliation_id,
                discrepancies=all_discrepancies,
                materiality_summary=materiality_summary,
                waterfall=waterfall,
                flags=flags,
            )

            # Update counters
            elapsed_s = time.monotonic() - start_time
            with self._lock:
                self._total_analyses += 1
                self._total_discrepancies_found += len(all_discrepancies)

            # Record metrics
            self._record_analysis_metrics(workspace, report, elapsed_s)

            # Record provenance
            self._record_provenance(workspace, report)

            logger.info(
                "Discrepancy analysis complete for %s: %d discrepancies, "
                "waterfall_items=%d, flags=%d, elapsed=%.3fs",
                reconciliation_id,
                len(all_discrepancies),
                len(waterfall.items) if waterfall else 0,
                len(flags),
                elapsed_s,
            )

            return report

        except ValueError:
            raise
        except Exception as exc:
            elapsed_s = time.monotonic() - start_time
            logger.error(
                "Discrepancy analysis failed for %s after %.3fs: %s",
                reconciliation_id,
                elapsed_s,
                exc,
                exc_info=True,
            )
            if self._metrics is not None:
                try:
                    self._metrics.record_error(
                        "calculation_error", "analyze_discrepancies",
                    )
                except Exception:
                    pass
            raise RuntimeError(
                f"Discrepancy analysis failed for "
                f"{reconciliation_id}: {exc}"
            ) from exc

    # ==================================================================
    # Public API: Total discrepancy
    # ==================================================================

    def calculate_total_discrepancy(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[Discrepancy]:
        """Calculate the overall discrepancy between location and market totals.

        Computes the absolute difference, percentage, direction, materiality,
        and PIF for the aggregate totals.

        Args:
            workspace: ReconciliationWorkspace with populated totals.

        Returns:
            Discrepancy object for the total-level difference, or None if
            location and market totals are both zero.
        """
        loc_total = workspace.total_location_tco2e
        mkt_total = workspace.total_market_tco2e

        if loc_total == _ZERO and mkt_total == _ZERO:
            logger.debug(
                "Both location and market totals are zero; "
                "no total discrepancy",
            )
            return None

        abs_diff = abs(loc_total - mkt_total)
        disc_pct = self.calculate_discrepancy_percentage(loc_total, mkt_total)
        direction = self.determine_direction(loc_total, mkt_total)
        materiality = self.determine_materiality(disc_pct)
        disc_type = self._classify_total_discrepancy_type(workspace)
        pif = self.calculate_pif(loc_total, mkt_total)

        description = (
            f"Total Scope 2 discrepancy: location-based "
            f"{loc_total:.4f} tCO2e vs market-based "
            f"{mkt_total:.4f} tCO2e. "
            f"Absolute difference: {abs_diff:.4f} tCO2e "
            f"({disc_pct:.2f}%). "
            f"Direction: {direction.value}. "
            f"PIF: {pif:.2f}%."
        )

        recommendation = self._get_total_recommendation(
            direction, materiality,
        )

        return Discrepancy(
            discrepancy_type=disc_type,
            direction=direction,
            materiality=materiality,
            absolute_tco2e=abs_diff,
            percentage=disc_pct,
            energy_type=None,
            facility_id=None,
            region=None,
            description=description,
            recommendation=recommendation,
        )

    # ==================================================================
    # Public API: Energy-type discrepancies
    # ==================================================================

    def calculate_energy_type_discrepancies(
        self,
        workspace: ReconciliationWorkspace,
    ) -> List[Discrepancy]:
        """Calculate discrepancies for each energy type.

        Groups upstream results by energy type and computes location vs market
        differences for electricity, steam, district heating, and district
        cooling separately.

        Args:
            workspace: ReconciliationWorkspace with populated upstream results.

        Returns:
            List of Discrepancy objects, one per energy type with non-zero
            emissions in at least one method.
        """
        discrepancies: List[Discrepancy] = []

        loc_by_type = self._aggregate_by_energy_type(
            workspace.location_results,
        )
        mkt_by_type = self._aggregate_by_energy_type(
            workspace.market_results,
        )

        all_types = set(loc_by_type.keys()) | set(mkt_by_type.keys())

        for energy_type in sorted(all_types, key=lambda et: et.value):
            loc_val = loc_by_type.get(energy_type, _ZERO)
            mkt_val = mkt_by_type.get(energy_type, _ZERO)

            if loc_val == _ZERO and mkt_val == _ZERO:
                continue

            abs_diff = abs(loc_val - mkt_val)
            if abs_diff < _ROUNDING_TOLERANCE:
                continue

            disc_pct = self.calculate_discrepancy_percentage(
                loc_val, mkt_val,
            )
            direction = self.determine_direction(loc_val, mkt_val)
            materiality = self.determine_materiality(disc_pct)
            disc_type = self._classify_energy_type_discrepancy(
                workspace, energy_type,
            )

            description = (
                f"{energy_type.value} discrepancy: location-based "
                f"{loc_val:.4f} tCO2e vs market-based "
                f"{mkt_val:.4f} tCO2e. "
                f"Absolute difference: {abs_diff:.4f} tCO2e "
                f"({disc_pct:.2f}%)."
            )

            recommendation = _DISCREPANCY_RECOMMENDATIONS.get(
                disc_type.value.upper(),
                "Review the methodology and data sources for this "
                "energy type.",
            )

            discrepancies.append(
                Discrepancy(
                    discrepancy_type=disc_type,
                    direction=direction,
                    materiality=materiality,
                    absolute_tco2e=abs_diff,
                    percentage=disc_pct,
                    energy_type=energy_type,
                    facility_id=None,
                    region=None,
                    description=description,
                    recommendation=recommendation,
                )
            )

        logger.debug(
            "Energy-type discrepancies: %d types with non-trivial "
            "differences",
            len(discrepancies),
        )
        return discrepancies

    # ==================================================================
    # Public API: Facility discrepancies
    # ==================================================================

    def calculate_facility_discrepancies(
        self,
        workspace: ReconciliationWorkspace,
    ) -> List[Discrepancy]:
        """Calculate discrepancies for each facility.

        Groups upstream results by facility_id and computes location vs
        market differences per facility.

        Args:
            workspace: ReconciliationWorkspace with populated upstream results.

        Returns:
            List of Discrepancy objects, one per facility with material
            differences.
        """
        discrepancies: List[Discrepancy] = []

        loc_by_facility = self._aggregate_by_facility(
            workspace.location_results,
        )
        mkt_by_facility = self._aggregate_by_facility(
            workspace.market_results,
        )

        all_facilities = (
            set(loc_by_facility.keys()) | set(mkt_by_facility.keys())
        )

        for facility_id in sorted(all_facilities):
            loc_val = loc_by_facility.get(facility_id, _ZERO)
            mkt_val = mkt_by_facility.get(facility_id, _ZERO)

            if loc_val == _ZERO and mkt_val == _ZERO:
                continue

            abs_diff = abs(loc_val - mkt_val)
            if abs_diff < _ROUNDING_TOLERANCE:
                continue

            disc_pct = self.calculate_discrepancy_percentage(
                loc_val, mkt_val,
            )
            direction = self.determine_direction(loc_val, mkt_val)
            materiality = self.determine_materiality(disc_pct)
            disc_type = self._classify_facility_discrepancy(
                workspace, facility_id,
            )

            region = self._get_facility_region(workspace, facility_id)

            description = (
                f"Facility {facility_id} discrepancy: location-based "
                f"{loc_val:.4f} tCO2e vs market-based "
                f"{mkt_val:.4f} tCO2e. "
                f"Absolute difference: {abs_diff:.4f} tCO2e "
                f"({disc_pct:.2f}%)."
            )

            recommendation = _DISCREPANCY_RECOMMENDATIONS.get(
                disc_type.value.upper(),
                "Review facility-level data sources and instrument "
                "coverage.",
            )

            discrepancies.append(
                Discrepancy(
                    discrepancy_type=disc_type,
                    direction=direction,
                    materiality=materiality,
                    absolute_tco2e=abs_diff,
                    percentage=disc_pct,
                    energy_type=None,
                    facility_id=facility_id,
                    region=region,
                    description=description,
                    recommendation=recommendation,
                )
            )

        logger.debug(
            "Facility discrepancies: %d facilities with non-trivial "
            "differences",
            len(discrepancies),
        )
        return discrepancies

    # ==================================================================
    # Public API: Instrument discrepancies
    # ==================================================================

    def calculate_instrument_discrepancies(
        self,
        workspace: ReconciliationWorkspace,
    ) -> List[Discrepancy]:
        """Calculate discrepancies by contractual instrument type.

        Analyzes how different EF hierarchy tiers (supplier-specific, bundled
        cert, unbundled cert, residual mix, grid average) contribute to the
        overall discrepancy.

        Args:
            workspace: ReconciliationWorkspace with populated upstream results.

        Returns:
            List of Discrepancy objects, one per instrument type that
            contributes to the discrepancy.
        """
        discrepancies: List[Discrepancy] = []

        mkt_by_hierarchy: Dict[
            Optional[EFHierarchyPriority], Decimal
        ] = defaultdict(lambda: _ZERO)
        mkt_mwh_by_hierarchy: Dict[
            Optional[EFHierarchyPriority], Decimal
        ] = defaultdict(lambda: _ZERO)

        for result in workspace.market_results:
            hierarchy = result.ef_hierarchy
            mkt_by_hierarchy[hierarchy] = (
                mkt_by_hierarchy[hierarchy] + result.emissions_tco2e
            )
            mkt_mwh_by_hierarchy[hierarchy] = (
                mkt_mwh_by_hierarchy[hierarchy]
                + result.energy_quantity_mwh
            )

        default_loc_ef = self._calculate_overall_average_ef(
            workspace.location_results,
        )

        for hierarchy, mkt_emissions in sorted(
            mkt_by_hierarchy.items(),
            key=lambda x: (x[0].value if x[0] else "zzz"),
        ):
            if hierarchy is None:
                continue

            mkt_mwh = mkt_mwh_by_hierarchy[hierarchy]
            if mkt_mwh == _ZERO:
                continue

            hypothetical_loc = mkt_mwh * default_loc_ef
            abs_diff = abs(hypothetical_loc - mkt_emissions)

            if abs_diff < _ROUNDING_TOLERANCE:
                continue

            disc_pct = self.calculate_discrepancy_percentage(
                hypothetical_loc, mkt_emissions,
            )
            direction = self.determine_direction(
                hypothetical_loc, mkt_emissions,
            )
            materiality = self.determine_materiality(disc_pct)
            disc_type = self._hierarchy_to_discrepancy_type(hierarchy)

            description = (
                f"Instrument type {hierarchy.value}: "
                f"{mkt_mwh:.2f} MWh covered. "
                f"Market-based: {mkt_emissions:.4f} tCO2e vs "
                f"hypothetical location-based: "
                f"{hypothetical_loc:.4f} tCO2e. "
                f"Instrument impact: {abs_diff:.4f} tCO2e "
                f"({disc_pct:.2f}%)."
            )

            recommendation = _DISCREPANCY_RECOMMENDATIONS.get(
                disc_type.value.upper(),
                "Review instrument documentation and emission "
                "factor sources.",
            )

            discrepancies.append(
                Discrepancy(
                    discrepancy_type=disc_type,
                    direction=direction,
                    materiality=materiality,
                    absolute_tco2e=abs_diff,
                    percentage=disc_pct,
                    energy_type=None,
                    facility_id=None,
                    region=None,
                    description=description,
                    recommendation=recommendation,
                )
            )

        logger.debug(
            "Instrument discrepancies: %d instrument types with "
            "contributions",
            len(discrepancies),
        )
        return discrepancies

    # ==================================================================
    # Public API: Classification methods
    # ==================================================================

    def classify_discrepancy_type(
        self,
        location_result: UpstreamResult,
        market_result: UpstreamResult,
    ) -> DiscrepancyType:
        """Classify the discrepancy type between a location/market result pair.

        Examines the characteristics of both results to determine which of
        the 8 discrepancy types best explains the difference.

        Args:
            location_result: Location-based upstream result.
            market_result: Market-based upstream result.

        Returns:
            The DiscrepancyType that best explains the difference.
        """
        # Check for REC/GO (market EF is zero with cert hierarchy)
        if (
            market_result.ef_hierarchy
            in (
                EFHierarchyPriority.BUNDLED_CERT,
                EFHierarchyPriority.UNBUNDLED_CERT,
            )
            and market_result.ef_used <= Decimal("0.001")
        ):
            return DiscrepancyType.REC_GO_IMPACT

        # Check for supplier EF delta
        if market_result.ef_hierarchy in (
            EFHierarchyPriority.SUPPLIER_WITH_CERT,
            EFHierarchyPriority.SUPPLIER_NO_CERT,
        ):
            return DiscrepancyType.SUPPLIER_EF_DELTA

        # Check for geographic mismatch
        if (
            location_result.region is not None
            and market_result.region is not None
            and location_result.region != market_result.region
        ):
            return DiscrepancyType.GEOGRAPHIC_MISMATCH

        # Check for temporal mismatch
        if (
            location_result.period_start != market_result.period_start
            or location_result.period_end != market_result.period_end
        ):
            return DiscrepancyType.TEMPORAL_MISMATCH

        # Check for steam/heat method divergence
        if location_result.energy_type in (
            EnergyType.STEAM,
            EnergyType.DISTRICT_HEATING,
        ):
            if location_result.ef_source != market_result.ef_source:
                return DiscrepancyType.STEAM_HEAT_METHOD

        # Check for residual mix
        if market_result.ef_hierarchy == EFHierarchyPriority.RESIDUAL_MIX:
            return DiscrepancyType.RESIDUAL_MIX_UPLIFT

        # Check for grid update timing
        if (
            location_result.ef_source != market_result.ef_source
            and market_result.ef_hierarchy
            == EFHierarchyPriority.GRID_AVERAGE
        ):
            return DiscrepancyType.GRID_UPDATE_TIMING

        # Default to residual mix uplift for unclassified differences
        return DiscrepancyType.RESIDUAL_MIX_UPLIFT

    def determine_materiality(
        self,
        discrepancy_pct: Decimal,
    ) -> MaterialityLevel:
        """Determine the materiality level for a discrepancy percentage.

        Uses MATERIALITY_THRESHOLDS constant to classify the percentage
        into one of five materiality levels.

        Args:
            discrepancy_pct: Discrepancy percentage (0-100+).

        Returns:
            MaterialityLevel classification.
        """
        abs_pct = abs(discrepancy_pct)

        if _MODELS_AVAILABLE:
            for level, (low, high) in MATERIALITY_THRESHOLDS.items():
                if low <= abs_pct < high:
                    return level

        # Fallback using config thresholds
        if abs_pct < self._immaterial_threshold:
            return MaterialityLevel.IMMATERIAL
        elif abs_pct < self._minor_threshold:
            return MaterialityLevel.MINOR
        elif abs_pct < self._material_threshold:
            return MaterialityLevel.MATERIAL
        elif abs_pct < self._significant_threshold:
            return MaterialityLevel.SIGNIFICANT
        else:
            return MaterialityLevel.EXTREME

    def determine_direction(
        self,
        location_total: Decimal,
        market_total: Decimal,
    ) -> DiscrepancyDirection:
        """Determine the direction of the discrepancy.

        Compares location-based and market-based totals to determine
        whether market is lower, higher, or equal (within tolerance).

        Args:
            location_total: Location-based total in tCO2e.
            market_total: Market-based total in tCO2e.

        Returns:
            DiscrepancyDirection indicating relative position.
        """
        diff = location_total - market_total

        if abs(diff) < _ROUNDING_TOLERANCE:
            return DiscrepancyDirection.EQUAL
        elif diff > _ZERO:
            return DiscrepancyDirection.MARKET_LOWER
        else:
            return DiscrepancyDirection.MARKET_HIGHER

    # ==================================================================
    # Public API: Percentage and PIF calculations
    # ==================================================================

    def calculate_discrepancy_percentage(
        self,
        location: Decimal,
        market: Decimal,
    ) -> Decimal:
        """Calculate the discrepancy percentage.

        Formula: |Location - Market| / max(Location, Market) x 100

        Args:
            location: Location-based total in tCO2e.
            market: Market-based total in tCO2e.

        Returns:
            Discrepancy percentage. Returns Decimal("0") if both are zero.
        """
        denominator = max(location, market)
        if denominator == _ZERO:
            return _ZERO

        result = (abs(location - market) / denominator) * _HUNDRED
        return _round_decimal(result, self._decimal_places)

    def calculate_pif(
        self,
        location: Decimal,
        market: Decimal,
    ) -> Decimal:
        """Calculate the Procurement Impact Factor (PIF).

        Formula: (Location_tCO2e - Market_tCO2e) / Location_tCO2e x 100

        A positive PIF indicates market-based emissions are lower (typical
        when renewable energy is procured). A negative PIF indicates
        market-based emissions are higher.

        Args:
            location: Location-based total in tCO2e.
            market: Market-based total in tCO2e.

        Returns:
            PIF percentage. Returns Decimal("0") if location is zero.
        """
        if location == _ZERO:
            return _ZERO

        result = ((location - market) / location) * _HUNDRED
        return _round_decimal(result, self._decimal_places)

    # ==================================================================
    # Public API: Waterfall decomposition
    # ==================================================================

    def build_waterfall(
        self,
        workspace: ReconciliationWorkspace,
        discrepancies: List[Discrepancy],
    ) -> WaterfallDecomposition:
        """Build waterfall decomposition from location to market total.

        Decomposes the total discrepancy into 8 driver components. Each
        driver receives a signed contribution (negative = reduces toward
        market). The sum of all driver contributions equals the total
        discrepancy within tolerance.

        Args:
            workspace: ReconciliationWorkspace with populated totals.
            discrepancies: List of classified discrepancies.

        Returns:
            WaterfallDecomposition with ordered items.
        """
        start_time = time.monotonic()
        total_disc = (
            workspace.total_location_tco2e
            - workspace.total_market_tco2e
        )

        items: List[WaterfallItem] = []

        rec_go = self.decompose_rec_go_impact(workspace)
        if rec_go is not None:
            items.append(rec_go)

        residual = self.decompose_residual_mix_uplift(workspace)
        if residual is not None:
            items.append(residual)

        supplier = self.decompose_supplier_ef_delta(workspace)
        if supplier is not None:
            items.append(supplier)

        geographic = self.decompose_geographic_mismatch(workspace)
        if geographic is not None:
            items.append(geographic)

        temporal = self.decompose_temporal_mismatch(workspace)
        if temporal is not None:
            items.append(temporal)

        partial = self.decompose_partial_coverage(workspace)
        if partial is not None:
            items.append(partial)

        steam = self.decompose_steam_heat_method(workspace)
        if steam is not None:
            items.append(steam)

        grid = self.decompose_grid_update_timing(workspace)
        if grid is not None:
            items.append(grid)

        # Balance check: assign residual to an adjustment item
        items = self._balance_waterfall(items, total_disc)

        waterfall = WaterfallDecomposition(
            total_discrepancy_tco2e=total_disc,
            items=items,
        )

        with self._lock:
            self._total_waterfall_builds += 1

        elapsed = time.monotonic() - start_time
        logger.debug(
            "Waterfall built: %d items, total_disc=%.4f tCO2e, "
            "elapsed=%.4fs",
            len(items),
            float(total_disc),
            elapsed,
        )

        return waterfall

    # ==================================================================
    # Public API: Individual waterfall driver decomposition
    # ==================================================================

    def decompose_rec_go_impact(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the REC/GO impact driver.

        RECs and GOs reduce market-based emissions to zero for covered MWh.
        The impact equals the location-based emissions for those MWh.

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for REC/GO impact, or None if no RECs/GOs.
        """
        covered_emissions = _ZERO
        covered_mwh = _ZERO

        for result in workspace.market_results:
            if result.ef_hierarchy in (
                EFHierarchyPriority.BUNDLED_CERT,
                EFHierarchyPriority.UNBUNDLED_CERT,
            ) and result.ef_used <= Decimal("0.001"):
                loc_emissions = self._find_matching_location_emissions(
                    workspace, result.facility_id, result.energy_type,
                )
                loc_total_mwh = self._get_facility_type_mwh(
                    workspace.location_results,
                    result.facility_id,
                    result.energy_type,
                )
                if (
                    loc_total_mwh > _ZERO
                    and loc_emissions > _ZERO
                ):
                    proportion = (
                        result.energy_quantity_mwh / loc_total_mwh
                    )
                    covered_emissions += loc_emissions * proportion
                covered_mwh += result.energy_quantity_mwh

        if covered_emissions == _ZERO:
            return None

        contribution = covered_emissions
        loc_total = workspace.total_location_tco2e
        pct = (
            (contribution / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = _DISCREPANCY_DESCRIPTIONS["REC_GO_IMPACT"].format(
            covered_mwh=float(covered_mwh),
            impact_tco2e=float(contribution),
        )

        return WaterfallItem(
            driver=DiscrepancyType.REC_GO_IMPACT.value,
            contribution_tco2e=_round_decimal(
                contribution, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_residual_mix_uplift(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the residual mix uplift driver.

        Residual mix EFs are typically higher than grid average because
        tracked renewables are removed. The uplift equals:
        MWh_uncovered x (residual_mix_ef - grid_average_ef).

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for residual mix uplift, or None if none present.
        """
        uplift = _ZERO
        total_uncovered_mwh = _ZERO
        avg_residual_ef = _ZERO
        avg_grid_ef = _ZERO
        region_used = "unknown"
        count = 0

        for result in workspace.market_results:
            if result.ef_hierarchy == EFHierarchyPriority.RESIDUAL_MIX:
                loc_ef = self._find_matching_location_ef(
                    workspace, result.facility_id, result.energy_type,
                )
                if loc_ef is not None and loc_ef > _ZERO:
                    ef_diff = result.ef_used - loc_ef
                    uplift += ef_diff * result.energy_quantity_mwh
                    total_uncovered_mwh += result.energy_quantity_mwh
                    avg_residual_ef += result.ef_used
                    avg_grid_ef += loc_ef
                    count += 1
                    if result.region:
                        region_used = result.region

        if count == 0 or total_uncovered_mwh == _ZERO:
            return None

        avg_residual_ef = avg_residual_ef / _D(count)
        avg_grid_ef = avg_grid_ef / _D(count)
        ratio = (
            avg_residual_ef / avg_grid_ef
            if avg_grid_ef > _ZERO
            else _ONE
        )

        # Negative = increases market relative to location
        contribution = -uplift

        loc_total = workspace.total_location_tco2e
        pct = (
            (contribution / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = _DISCREPANCY_DESCRIPTIONS[
            "RESIDUAL_MIX_UPLIFT"
        ].format(
            residual_ef=float(avg_residual_ef),
            grid_ef=float(avg_grid_ef),
            region=region_used,
            ratio=float(ratio),
            impact_tco2e=float(abs(contribution)),
        )

        return WaterfallItem(
            driver=DiscrepancyType.RESIDUAL_MIX_UPLIFT.value,
            contribution_tco2e=_round_decimal(
                contribution, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_supplier_ef_delta(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the supplier-specific EF delta driver.

        Supplier EFs may be higher or lower than grid average. Impact:
        MWh x (grid_ef - supplier_ef).

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for supplier EF delta, or None if no supplier EFs.
        """
        total_impact = _ZERO
        total_mwh = _ZERO
        avg_supplier_ef = _ZERO
        avg_grid_ef = _ZERO
        count = 0

        for result in workspace.market_results:
            if result.ef_hierarchy in (
                EFHierarchyPriority.SUPPLIER_WITH_CERT,
                EFHierarchyPriority.SUPPLIER_NO_CERT,
            ):
                loc_ef = self._find_matching_location_ef(
                    workspace,
                    result.facility_id,
                    result.energy_type,
                )
                if loc_ef is not None and loc_ef > _ZERO:
                    ef_diff = loc_ef - result.ef_used
                    total_impact += ef_diff * result.energy_quantity_mwh
                    total_mwh += result.energy_quantity_mwh
                    avg_supplier_ef += result.ef_used
                    avg_grid_ef += loc_ef
                    count += 1

        if count == 0 or total_mwh == _ZERO:
            return None

        avg_supplier_ef = avg_supplier_ef / _D(count)
        avg_grid_ef = avg_grid_ef / _D(count)
        delta_pct = (
            (
                (avg_grid_ef - avg_supplier_ef)
                / avg_grid_ef
                * _HUNDRED
            )
            if avg_grid_ef > _ZERO
            else _ZERO
        )

        loc_total = workspace.total_location_tco2e
        pct = (
            (total_impact / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = _DISCREPANCY_DESCRIPTIONS[
            "SUPPLIER_EF_DELTA"
        ].format(
            supplier_ef=float(avg_supplier_ef),
            grid_ef=float(avg_grid_ef),
            delta_pct=float(delta_pct),
            impact_tco2e=float(abs(total_impact)),
        )

        return WaterfallItem(
            driver=DiscrepancyType.SUPPLIER_EF_DELTA.value,
            contribution_tco2e=_round_decimal(
                total_impact, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_geographic_mismatch(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the geographic mismatch driver.

        Calculates the impact of consuming energy in one region but holding
        contractual instruments from a different region.

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for geographic mismatch, or None if none found.
        """
        total_impact = _ZERO
        mismatch_count = 0

        loc_by_key = self._build_facility_type_lookup(
            workspace.location_results,
        )

        for result in workspace.market_results:
            key = (result.facility_id, result.energy_type)
            loc_result = loc_by_key.get(key)
            if loc_result is None:
                continue

            if (
                loc_result.region is not None
                and result.region is not None
                and loc_result.region != result.region
            ):
                ef_diff = loc_result.ef_used - result.ef_used
                impact = ef_diff * result.energy_quantity_mwh
                total_impact += impact
                mismatch_count += 1

        if mismatch_count == 0:
            return None

        loc_total = workspace.total_location_tco2e
        pct = (
            (total_impact / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = (
            f"Geographic mismatch across {mismatch_count} "
            f"facility/instrument pairs contributes "
            f"{abs(float(total_impact)):.4f} tCO2e to the discrepancy "
            f"due to differing regional emission factors."
        )

        return WaterfallItem(
            driver=DiscrepancyType.GEOGRAPHIC_MISMATCH.value,
            contribution_tco2e=_round_decimal(
                total_impact, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_temporal_mismatch(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the temporal mismatch driver.

        Calculates the impact of instrument vintages not aligning with
        the consumption reporting period.

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for temporal mismatch, or None if none found.
        """
        total_impact = _ZERO
        mismatch_count = 0

        loc_by_key = self._build_facility_type_lookup(
            workspace.location_results,
        )

        for result in workspace.market_results:
            key = (result.facility_id, result.energy_type)
            loc_result = loc_by_key.get(key)
            if loc_result is None:
                continue

            if (
                loc_result.period_start != result.period_start
                or loc_result.period_end != result.period_end
            ):
                ef_diff = loc_result.ef_used - result.ef_used
                impact = ef_diff * result.energy_quantity_mwh
                total_impact += impact
                mismatch_count += 1

        if mismatch_count == 0:
            return None

        loc_total = workspace.total_location_tco2e
        pct = (
            (total_impact / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = (
            f"Temporal mismatch across {mismatch_count} "
            f"facility/instrument pairs contributes "
            f"{abs(float(total_impact)):.4f} tCO2e due to instrument "
            f"vintage not aligning with consumption period."
        )

        return WaterfallItem(
            driver=DiscrepancyType.TEMPORAL_MISMATCH.value,
            contribution_tco2e=_round_decimal(
                total_impact, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_partial_coverage(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the partial coverage driver.

        Calculates the impact when instruments cover only a portion of
        total consumption, leaving uncovered MWh subject to residual mix
        instead of grid average.

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for partial coverage, or None if fully covered.
        """
        # Total MWh from location (represents total consumption)
        total_loc_mwh = _ZERO
        for result in workspace.location_results:
            total_loc_mwh += result.energy_quantity_mwh

        # Covered MWh (non-residual, non-grid instruments)
        covered_mwh = _ZERO
        for result in workspace.market_results:
            if result.ef_hierarchy in (
                EFHierarchyPriority.SUPPLIER_WITH_CERT,
                EFHierarchyPriority.SUPPLIER_NO_CERT,
                EFHierarchyPriority.BUNDLED_CERT,
                EFHierarchyPriority.UNBUNDLED_CERT,
            ):
                covered_mwh += result.energy_quantity_mwh

        if total_loc_mwh == _ZERO:
            return None

        uncovered_mwh = total_loc_mwh - covered_mwh
        if uncovered_mwh <= _ZERO:
            return None

        coverage_pct = (covered_mwh / total_loc_mwh) * _HUNDRED
        if coverage_pct >= Decimal("99.99"):
            return None

        avg_loc_ef = self._calculate_overall_average_ef(
            workspace.location_results,
        )
        avg_residual_ef = _ZERO
        residual_count = 0
        for result in workspace.market_results:
            if result.ef_hierarchy == EFHierarchyPriority.RESIDUAL_MIX:
                avg_residual_ef += result.ef_used
                residual_count += 1

        if residual_count > 0:
            avg_residual_ef = avg_residual_ef / _D(residual_count)
        else:
            # Estimate residual as 1.3x grid average if no data
            avg_residual_ef = avg_loc_ef * Decimal("1.3")

        impact = uncovered_mwh * (avg_residual_ef - avg_loc_ef)
        contribution = -impact  # negative = increases market

        loc_total = workspace.total_location_tco2e
        pct = (
            (contribution / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = _DISCREPANCY_DESCRIPTIONS[
            "PARTIAL_COVERAGE"
        ].format(
            coverage_pct=float(coverage_pct),
            covered_mwh=float(covered_mwh),
            total_mwh=float(total_loc_mwh),
            uncovered_mwh=float(uncovered_mwh),
            impact_tco2e=float(abs(contribution)),
        )

        return WaterfallItem(
            driver=DiscrepancyType.PARTIAL_COVERAGE.value,
            contribution_tco2e=_round_decimal(
                contribution, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_steam_heat_method(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the steam/heat method divergence driver.

        Calculates the impact of methodological differences between location
        and market for steam and heating energy types.

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for steam/heat method, or None if no divergence.
        """
        total_impact = _ZERO
        loc_method_desc = "grid_average"
        mkt_method_desc = "contractual"
        affected_count = 0

        steam_heat_types = {
            EnergyType.STEAM,
            EnergyType.DISTRICT_HEATING,
        }

        loc_by_key = self._build_facility_type_lookup(
            workspace.location_results,
        )

        for result in workspace.market_results:
            if result.energy_type not in steam_heat_types:
                continue

            key = (result.facility_id, result.energy_type)
            loc_result = loc_by_key.get(key)
            if loc_result is None:
                continue

            if loc_result.ef_source != result.ef_source:
                ef_diff = loc_result.ef_used - result.ef_used
                impact = ef_diff * result.energy_quantity_mwh
                total_impact += impact
                affected_count += 1
                loc_method_desc = loc_result.ef_source
                mkt_method_desc = result.ef_source

        if affected_count == 0:
            return None

        loc_total = workspace.total_location_tco2e
        pct = (
            (total_impact / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = _DISCREPANCY_DESCRIPTIONS[
            "STEAM_HEAT_METHOD"
        ].format(
            location_method=loc_method_desc,
            market_method=mkt_method_desc,
            impact_tco2e=float(abs(total_impact)),
        )

        return WaterfallItem(
            driver=DiscrepancyType.STEAM_HEAT_METHOD.value,
            contribution_tco2e=_round_decimal(
                total_impact, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    def decompose_grid_update_timing(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Optional[WaterfallItem]:
        """Decompose the grid update timing driver.

        Calculates the impact of using different grid EF vintages between
        location-based and market-based calculations.

        Args:
            workspace: ReconciliationWorkspace with upstream results.

        Returns:
            WaterfallItem for grid timing, or None if no timing differences.
        """
        total_impact = _ZERO
        loc_vintage = "unknown"
        mkt_vintage = "unknown"
        affected_count = 0

        loc_by_key = self._build_facility_type_lookup(
            workspace.location_results,
        )

        for result in workspace.market_results:
            if result.ef_hierarchy != EFHierarchyPriority.GRID_AVERAGE:
                continue

            key = (result.facility_id, result.energy_type)
            loc_result = loc_by_key.get(key)
            if loc_result is None:
                continue

            if loc_result.ef_source != result.ef_source:
                ef_diff = loc_result.ef_used - result.ef_used
                impact = ef_diff * result.energy_quantity_mwh
                total_impact += impact
                affected_count += 1
                loc_vintage = loc_result.ef_source
                mkt_vintage = result.ef_source

        if affected_count == 0:
            return None

        loc_total = workspace.total_location_tco2e
        pct = (
            (total_impact / loc_total * _HUNDRED)
            if loc_total > _ZERO
            else _ZERO
        )

        description = _DISCREPANCY_DESCRIPTIONS[
            "GRID_UPDATE_TIMING"
        ].format(
            location_ef_vintage=loc_vintage,
            market_ef_vintage=mkt_vintage,
            impact_tco2e=float(abs(total_impact)),
        )

        return WaterfallItem(
            driver=DiscrepancyType.GRID_UPDATE_TIMING.value,
            contribution_tco2e=_round_decimal(
                total_impact, self._decimal_places,
            ),
            contribution_pct=_round_decimal(pct, self._decimal_places),
            description=description,
        )

    # ==================================================================
    # Public API: Flag generation
    # ==================================================================

    def generate_discrepancy_flags(
        self,
        discrepancies: List[Discrepancy],
    ) -> List[Flag]:
        """Generate flags for material discrepancies and notable conditions.

        Examines each discrepancy and generates appropriate warning, error,
        or informational flags based on materiality, direction, and type.

        Args:
            discrepancies: List of Discrepancy objects to evaluate.

        Returns:
            List of Flag objects.
        """
        flags: List[Flag] = []

        if not discrepancies:
            flags.append(
                Flag(
                    flag_type=FlagType.INFO,
                    severity=FlagSeverity.LOW,
                    code=_FLAG_CODE_NO_DISCREPANCY,
                    message=(
                        "No material discrepancies found between "
                        "location-based and market-based Scope 2 totals."
                    ),
                    recommendation="",
                )
            )
            with self._lock:
                self._total_flags_generated += len(flags)
            return flags

        for disc in discrepancies:
            disc_flags = self._generate_single_discrepancy_flags(disc)
            flags.extend(disc_flags)

        with self._lock:
            self._total_flags_generated += len(flags)

        logger.debug(
            "Generated %d flags from %d discrepancies",
            len(flags),
            len(discrepancies),
        )
        return flags

    # ==================================================================
    # Public API: Report query methods
    # ==================================================================

    def get_material_discrepancies(
        self,
        report: DiscrepancyReport,
    ) -> List[Discrepancy]:
        """Get only material, significant, and extreme discrepancies.

        Filters the report to return those at or above MATERIAL (>= 15%).

        Args:
            report: DiscrepancyReport to filter.

        Returns:
            List of Discrepancy objects with materiality >= MATERIAL.
        """
        material_levels = {
            MaterialityLevel.MATERIAL,
            MaterialityLevel.SIGNIFICANT,
            MaterialityLevel.EXTREME,
        }
        return [
            d for d in report.discrepancies
            if d.materiality in material_levels
        ]

    def get_discrepancies_by_type(
        self,
        report: DiscrepancyReport,
        disc_type: DiscrepancyType,
    ) -> List[Discrepancy]:
        """Filter discrepancies by type.

        Args:
            report: DiscrepancyReport to filter.
            disc_type: DiscrepancyType to match.

        Returns:
            List of matching Discrepancy objects.
        """
        return [
            d for d in report.discrepancies
            if d.discrepancy_type == disc_type
        ]

    def get_discrepancies_by_energy_type(
        self,
        report: DiscrepancyReport,
        energy_type: EnergyType,
    ) -> List[Discrepancy]:
        """Filter discrepancies by energy type.

        Args:
            report: DiscrepancyReport to filter.
            energy_type: EnergyType to match.

        Returns:
            List of matching Discrepancy objects.
        """
        return [
            d for d in report.discrepancies
            if d.energy_type == energy_type
        ]

    # ==================================================================
    # Public API: Summary and aggregation
    # ==================================================================

    def summarize_discrepancy_report(
        self,
        report: DiscrepancyReport,
    ) -> Dict[str, Any]:
        """Generate a summary dictionary of the discrepancy report.

        Produces a machine-readable summary suitable for API responses
        and dashboard display.

        Args:
            report: DiscrepancyReport to summarize.

        Returns:
            Dictionary with summary statistics.
        """
        total_disc = _ZERO
        if report.waterfall is not None:
            total_disc = report.waterfall.total_discrepancy_tco2e

        by_type: Dict[str, int] = defaultdict(int)
        for d in report.discrepancies:
            by_type[d.discrepancy_type.value] += 1

        by_direction: Dict[str, int] = defaultdict(int)
        for d in report.discrepancies:
            by_direction[d.direction.value] += 1

        largest = max(
            report.discrepancies,
            key=lambda d: d.absolute_tco2e,
            default=None,
        )

        driver_ranking: List[Dict[str, Any]] = []
        if report.waterfall is not None:
            sorted_items = sorted(
                report.waterfall.items,
                key=lambda i: abs(i.contribution_tco2e),
                reverse=True,
            )
            for item in sorted_items:
                driver_ranking.append({
                    "driver": item.driver,
                    "contribution_tco2e": str(item.contribution_tco2e),
                    "contribution_pct": str(item.contribution_pct),
                })

        return {
            "reconciliation_id": report.reconciliation_id,
            "total_discrepancy_tco2e": str(total_disc),
            "discrepancy_count": len(report.discrepancies),
            "materiality_summary": report.materiality_summary,
            "by_type": dict(by_type),
            "by_direction": dict(by_direction),
            "largest_discrepancy": {
                "type": (
                    largest.discrepancy_type.value
                    if largest
                    else None
                ),
                "absolute_tco2e": (
                    str(largest.absolute_tco2e)
                    if largest
                    else "0"
                ),
                "percentage": (
                    str(largest.percentage) if largest else "0"
                ),
            },
            "waterfall_items_count": (
                len(report.waterfall.items)
                if report.waterfall
                else 0
            ),
            "driver_ranking": driver_ranking,
            "flag_count": len(report.flags),
            "provenance_hash": _compute_hash(report),
        }

    def calculate_weighted_discrepancy(
        self,
        discrepancies: List[Discrepancy],
    ) -> Decimal:
        """Calculate the weighted average discrepancy percentage.

        Weights each discrepancy by its absolute tCO2e contribution.

        Args:
            discrepancies: List of Discrepancy objects.

        Returns:
            Weighted average discrepancy percentage.
        """
        if not discrepancies:
            return _ZERO

        total_weight = _ZERO
        weighted_sum = _ZERO

        for d in discrepancies:
            weight = d.absolute_tco2e
            total_weight += weight
            weighted_sum += d.percentage * weight

        if total_weight == _ZERO:
            return _ZERO

        result = weighted_sum / total_weight
        return _round_decimal(result, self._decimal_places)

    # ==================================================================
    # Public API: Waterfall validation and queries
    # ==================================================================

    def validate_waterfall_balance(
        self,
        waterfall: WaterfallDecomposition,
        total_discrepancy: Decimal,
    ) -> bool:
        """Validate that waterfall items sum to the total discrepancy.

        Checks that sum of contributions equals total discrepancy within
        a tolerance of 0.1 tCO2e.

        Args:
            waterfall: WaterfallDecomposition to validate.
            total_discrepancy: Expected total (location - market).

        Returns:
            True if balanced within tolerance, False otherwise.
        """
        items_sum = _ZERO
        for item in waterfall.items:
            items_sum += item.contribution_tco2e

        diff = abs(items_sum - total_discrepancy)
        is_balanced = diff <= _WATERFALL_BALANCE_TOLERANCE

        if not is_balanced:
            logger.warning(
                "Waterfall imbalance: items_sum=%.4f, expected=%.4f, "
                "diff=%.4f",
                float(items_sum),
                float(total_discrepancy),
                float(diff),
            )

        return is_balanced

    def get_largest_drivers(
        self,
        waterfall: WaterfallDecomposition,
        top_n: int = 3,
    ) -> List[WaterfallItem]:
        """Get the top-N largest waterfall drivers by absolute contribution.

        Args:
            waterfall: WaterfallDecomposition to query.
            top_n: Number of top drivers to return (default 3).

        Returns:
            List of WaterfallItem sorted by |contribution|, largest first.
        """
        sorted_items = sorted(
            waterfall.items,
            key=lambda item: abs(item.contribution_tco2e),
            reverse=True,
        )
        return sorted_items[:top_n]

    # ==================================================================
    # Public API: Health and lifecycle
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Return engine health status and statistics.

        Returns:
            Dictionary with engine health information including uptime,
            operation counters, and configuration state.
        """
        with self._lock:
            return {
                "engine_id": _ENGINE_ID,
                "engine_version": _ENGINE_VERSION,
                "component": _ENGINE_COMPONENT,
                "status": "healthy",
                "initialized": self.__class__._initialized,
                "created_at": self._created_at.isoformat(),
                "uptime_seconds": (
                    _utcnow() - self._created_at
                ).total_seconds(),
                "total_analyses": self._total_analyses,
                "total_discrepancies_found": (
                    self._total_discrepancies_found
                ),
                "total_waterfall_builds": self._total_waterfall_builds,
                "total_flags_generated": self._total_flags_generated,
                "decimal_places": self._decimal_places,
                "config_available": self._config is not None,
                "metrics_available": self._metrics is not None,
                "models_available": _MODELS_AVAILABLE,
                "provenance_available": _PROVENANCE_AVAILABLE,
                "materiality_thresholds": {
                    "immaterial": str(self._immaterial_threshold),
                    "minor": str(self._minor_threshold),
                    "material": str(self._material_threshold),
                    "significant": str(self._significant_threshold),
                },
            }

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        Clears the singleton so the next instantiation creates a fresh
        engine. Intended for test teardown only.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.info("DiscrepancyAnalyzerEngine singleton reset")

    def get_engine_id(self) -> str:
        """Return the engine identifier.

        Returns:
            Engine ID string.
        """
        return _ENGINE_ID

    def get_engine_version(self) -> str:
        """Return the engine version.

        Returns:
            Engine version string.
        """
        return _ENGINE_VERSION

    # ==================================================================
    # Private helpers: Workspace validation
    # ==================================================================

    def _validate_workspace(
        self,
        workspace: ReconciliationWorkspace,
    ) -> None:
        """Validate that the workspace has minimum required data.

        Args:
            workspace: Workspace to validate.

        Raises:
            ValueError: If workspace is missing required fields.
        """
        if workspace is None:
            raise ValueError("Workspace cannot be None")

        if not workspace.reconciliation_id:
            raise ValueError("Workspace must have a reconciliation_id")

        if not workspace.tenant_id:
            raise ValueError("Workspace must have a tenant_id")

        if (
            not workspace.location_results
            and not workspace.market_results
        ):
            raise ValueError(
                "Workspace must have at least one location or "
                "market result"
            )

    # ==================================================================
    # Private helpers: Aggregation
    # ==================================================================

    def _aggregate_by_energy_type(
        self,
        results: List[UpstreamResult],
    ) -> Dict[EnergyType, Decimal]:
        """Aggregate emissions by energy type.

        Args:
            results: List of upstream results to aggregate.

        Returns:
            Dictionary mapping EnergyType to total tCO2e.
        """
        agg: Dict[EnergyType, Decimal] = defaultdict(lambda: _ZERO)
        for r in results:
            agg[r.energy_type] = agg[r.energy_type] + r.emissions_tco2e
        return dict(agg)

    def _aggregate_by_facility(
        self,
        results: List[UpstreamResult],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by facility_id.

        Args:
            results: List of upstream results to aggregate.

        Returns:
            Dictionary mapping facility_id to total tCO2e.
        """
        agg: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        for r in results:
            agg[r.facility_id] = agg[r.facility_id] + r.emissions_tco2e
        return dict(agg)

    def _build_facility_type_lookup(
        self,
        results: List[UpstreamResult],
    ) -> Dict[Tuple[str, EnergyType], UpstreamResult]:
        """Build a lookup dict from (facility_id, energy_type) to result.

        For facilities with multiple results per type, uses the first.

        Args:
            results: List of upstream results.

        Returns:
            Dictionary mapping (facility_id, energy_type) to result.
        """
        lookup: Dict[
            Tuple[str, EnergyType], UpstreamResult
        ] = {}
        for r in results:
            key = (r.facility_id, r.energy_type)
            if key not in lookup:
                lookup[key] = r
        return lookup

    def _find_matching_location_emissions(
        self,
        workspace: ReconciliationWorkspace,
        facility_id: str,
        energy_type: EnergyType,
    ) -> Decimal:
        """Find location-based emissions for a facility/energy type pair.

        Args:
            workspace: Workspace with location results.
            facility_id: Facility to match.
            energy_type: Energy type to match.

        Returns:
            Total location-based emissions for the match, or zero.
        """
        total = _ZERO
        for r in workspace.location_results:
            if (
                r.facility_id == facility_id
                and r.energy_type == energy_type
            ):
                total += r.emissions_tco2e
        return total

    def _find_matching_location_ef(
        self,
        workspace: ReconciliationWorkspace,
        facility_id: str,
        energy_type: EnergyType,
    ) -> Optional[Decimal]:
        """Find location-based emission factor for a facility/type pair.

        Args:
            workspace: Workspace with location results.
            facility_id: Facility to match.
            energy_type: Energy type to match.

        Returns:
            Emission factor from matching location result, or None.
        """
        for r in workspace.location_results:
            if (
                r.facility_id == facility_id
                and r.energy_type == energy_type
            ):
                return r.ef_used
        return None

    def _get_facility_type_mwh(
        self,
        results: List[UpstreamResult],
        facility_id: str,
        energy_type: EnergyType,
    ) -> Decimal:
        """Get total MWh for a facility/type pair.

        Args:
            results: List of upstream results.
            facility_id: Facility to match.
            energy_type: Energy type to match.

        Returns:
            Total MWh for the match.
        """
        total = _ZERO
        for r in results:
            if (
                r.facility_id == facility_id
                and r.energy_type == energy_type
            ):
                total += r.energy_quantity_mwh
        return total

    def _get_facility_region(
        self,
        workspace: ReconciliationWorkspace,
        facility_id: str,
    ) -> Optional[str]:
        """Get the region for a facility from upstream results.

        Args:
            workspace: Workspace with upstream results.
            facility_id: Facility to look up.

        Returns:
            Region string if found, None otherwise.
        """
        for r in workspace.location_results:
            if r.facility_id == facility_id and r.region:
                return r.region
        for r in workspace.market_results:
            if r.facility_id == facility_id and r.region:
                return r.region
        return None

    def _calculate_average_location_ef(
        self,
        location_results: List[UpstreamResult],
    ) -> Dict[Optional[str], Decimal]:
        """Calculate average location EF by region.

        Args:
            location_results: Location-based upstream results.

        Returns:
            Dictionary mapping region to average EF.
        """
        ef_sum: Dict[Optional[str], Decimal] = defaultdict(
            lambda: _ZERO,
        )
        ef_count: Dict[Optional[str], int] = defaultdict(int)

        for r in location_results:
            ef_sum[r.region] = ef_sum[r.region] + r.ef_used
            ef_count[r.region] += 1

        result: Dict[Optional[str], Decimal] = {}
        for region, total_ef in ef_sum.items():
            count = ef_count[region]
            if count > 0:
                result[region] = total_ef / _D(count)
        return result

    def _calculate_overall_average_ef(
        self,
        results: List[UpstreamResult],
    ) -> Decimal:
        """Calculate overall MWh-weighted average emission factor.

        Args:
            results: List of upstream results.

        Returns:
            MWh-weighted average emission factor.
        """
        total_emissions = _ZERO
        total_mwh = _ZERO

        for r in results:
            total_emissions += r.emissions_tco2e
            total_mwh += r.energy_quantity_mwh

        if total_mwh == _ZERO:
            return _ZERO

        return total_emissions / total_mwh

    # ==================================================================
    # Private helpers: Classification
    # ==================================================================

    def _classify_total_discrepancy_type(
        self,
        workspace: ReconciliationWorkspace,
    ) -> DiscrepancyType:
        """Classify the primary driver for the total-level discrepancy.

        Analyzes market-based results to determine which driver type
        contributes most to the overall discrepancy.

        Args:
            workspace: Workspace with upstream results.

        Returns:
            The dominant DiscrepancyType for the total discrepancy.
        """
        rec_go_mwh = _ZERO
        residual_mwh = _ZERO
        supplier_mwh = _ZERO
        grid_mwh = _ZERO

        for result in workspace.market_results:
            mwh = result.energy_quantity_mwh
            if (
                result.ef_hierarchy
                in (
                    EFHierarchyPriority.BUNDLED_CERT,
                    EFHierarchyPriority.UNBUNDLED_CERT,
                )
                and result.ef_used <= Decimal("0.001")
            ):
                rec_go_mwh += mwh
            elif (
                result.ef_hierarchy
                == EFHierarchyPriority.RESIDUAL_MIX
            ):
                residual_mwh += mwh
            elif result.ef_hierarchy in (
                EFHierarchyPriority.SUPPLIER_WITH_CERT,
                EFHierarchyPriority.SUPPLIER_NO_CERT,
            ):
                supplier_mwh += mwh
            else:
                grid_mwh += mwh

        categories = {
            DiscrepancyType.REC_GO_IMPACT: rec_go_mwh,
            DiscrepancyType.RESIDUAL_MIX_UPLIFT: residual_mwh,
            DiscrepancyType.SUPPLIER_EF_DELTA: supplier_mwh,
            DiscrepancyType.GRID_UPDATE_TIMING: grid_mwh,
        }

        dominant = max(categories, key=lambda k: categories[k])

        if all(v == _ZERO for v in categories.values()):
            return DiscrepancyType.RESIDUAL_MIX_UPLIFT

        return dominant

    def _classify_energy_type_discrepancy(
        self,
        workspace: ReconciliationWorkspace,
        energy_type: EnergyType,
    ) -> DiscrepancyType:
        """Classify discrepancy type for a specific energy type.

        Args:
            workspace: Workspace with upstream results.
            energy_type: Energy type to classify.

        Returns:
            DiscrepancyType for this energy type.
        """
        if energy_type in (
            EnergyType.STEAM,
            EnergyType.DISTRICT_HEATING,
        ):
            loc_sources = set()
            mkt_sources = set()
            for r in workspace.location_results:
                if r.energy_type == energy_type:
                    loc_sources.add(r.ef_source)
            for r in workspace.market_results:
                if r.energy_type == energy_type:
                    mkt_sources.add(r.ef_source)
            if (
                loc_sources
                and mkt_sources
                and loc_sources != mkt_sources
            ):
                return DiscrepancyType.STEAM_HEAT_METHOD

        rec_go_mwh = _ZERO
        total_mkt_mwh = _ZERO
        for r in workspace.market_results:
            if r.energy_type == energy_type:
                total_mkt_mwh += r.energy_quantity_mwh
                if (
                    r.ef_hierarchy
                    in (
                        EFHierarchyPriority.BUNDLED_CERT,
                        EFHierarchyPriority.UNBUNDLED_CERT,
                    )
                    and r.ef_used <= Decimal("0.001")
                ):
                    rec_go_mwh += r.energy_quantity_mwh

        if (
            total_mkt_mwh > _ZERO
            and rec_go_mwh / total_mkt_mwh > Decimal("0.5")
        ):
            return DiscrepancyType.REC_GO_IMPACT

        return DiscrepancyType.RESIDUAL_MIX_UPLIFT

    def _classify_facility_discrepancy(
        self,
        workspace: ReconciliationWorkspace,
        facility_id: str,
    ) -> DiscrepancyType:
        """Classify discrepancy type for a specific facility.

        Args:
            workspace: Workspace with upstream results.
            facility_id: Facility to classify.

        Returns:
            DiscrepancyType for this facility.
        """
        facility_mkt = [
            r
            for r in workspace.market_results
            if r.facility_id == facility_id
        ]
        facility_loc = [
            r
            for r in workspace.location_results
            if r.facility_id == facility_id
        ]

        if not facility_mkt:
            return DiscrepancyType.PARTIAL_COVERAGE

        loc_regions = {r.region for r in facility_loc if r.region}
        mkt_regions = {r.region for r in facility_mkt if r.region}
        if (
            loc_regions
            and mkt_regions
            and loc_regions != mkt_regions
        ):
            return DiscrepancyType.GEOGRAPHIC_MISMATCH

        for r in facility_mkt:
            if (
                r.ef_hierarchy
                in (
                    EFHierarchyPriority.BUNDLED_CERT,
                    EFHierarchyPriority.UNBUNDLED_CERT,
                )
                and r.ef_used <= Decimal("0.001")
            ):
                return DiscrepancyType.REC_GO_IMPACT

        for r in facility_mkt:
            if r.ef_hierarchy in (
                EFHierarchyPriority.SUPPLIER_WITH_CERT,
                EFHierarchyPriority.SUPPLIER_NO_CERT,
            ):
                return DiscrepancyType.SUPPLIER_EF_DELTA

        return DiscrepancyType.RESIDUAL_MIX_UPLIFT

    def _hierarchy_to_discrepancy_type(
        self,
        hierarchy: EFHierarchyPriority,
    ) -> DiscrepancyType:
        """Map an EF hierarchy tier to the corresponding discrepancy type.

        Args:
            hierarchy: EF hierarchy priority.

        Returns:
            Corresponding DiscrepancyType.
        """
        mapping = {
            EFHierarchyPriority.SUPPLIER_WITH_CERT: (
                DiscrepancyType.SUPPLIER_EF_DELTA
            ),
            EFHierarchyPriority.SUPPLIER_NO_CERT: (
                DiscrepancyType.SUPPLIER_EF_DELTA
            ),
            EFHierarchyPriority.BUNDLED_CERT: (
                DiscrepancyType.REC_GO_IMPACT
            ),
            EFHierarchyPriority.UNBUNDLED_CERT: (
                DiscrepancyType.REC_GO_IMPACT
            ),
            EFHierarchyPriority.RESIDUAL_MIX: (
                DiscrepancyType.RESIDUAL_MIX_UPLIFT
            ),
            EFHierarchyPriority.GRID_AVERAGE: (
                DiscrepancyType.GRID_UPDATE_TIMING
            ),
        }
        return mapping.get(
            hierarchy, DiscrepancyType.RESIDUAL_MIX_UPLIFT,
        )

    # ==================================================================
    # Private helpers: Waterfall balancing
    # ==================================================================

    def _balance_waterfall(
        self,
        items: List[WaterfallItem],
        total_discrepancy: Decimal,
    ) -> List[WaterfallItem]:
        """Balance waterfall items to match the total discrepancy.

        If the sum does not match, adds a rounding adjustment item to
        maintain the invariant that all items sum to the total.

        Args:
            items: Current waterfall items.
            total_discrepancy: Expected total (location - market).

        Returns:
            Balanced list of waterfall items.
        """
        items_sum = _ZERO
        for item in items:
            items_sum += item.contribution_tco2e

        residual = total_discrepancy - items_sum

        if abs(residual) <= _ROUNDING_TOLERANCE:
            return items

        pct = _ZERO
        if total_discrepancy != _ZERO:
            pct = (
                (residual / abs(total_discrepancy)) * _HUNDRED
            )

        adjustment = WaterfallItem(
            driver="rounding_adjustment",
            contribution_tco2e=_round_decimal(
                residual, self._decimal_places,
            ),
            contribution_pct=_round_decimal(
                pct, self._decimal_places,
            ),
            description=(
                f"Rounding adjustment of {float(residual):.4f} tCO2e "
                f"to balance waterfall decomposition. This residual "
                f"arises from interaction effects between multiple "
                f"drivers that cannot be cleanly attributed to a "
                f"single category."
            ),
        )

        result = list(items)
        result.append(adjustment)
        return result

    # ==================================================================
    # Private helpers: Recommendation generation
    # ==================================================================

    def _get_total_recommendation(
        self,
        direction: DiscrepancyDirection,
        materiality: MaterialityLevel,
    ) -> str:
        """Generate a recommendation for the total-level discrepancy.

        Args:
            direction: Discrepancy direction.
            materiality: Materiality level.

        Returns:
            Recommendation string.
        """
        if materiality == MaterialityLevel.IMMATERIAL:
            return (
                "No action required. The discrepancy between "
                "location-based and market-based totals is immaterial "
                "(< 5%)."
            )

        if materiality == MaterialityLevel.MINOR:
            return (
                "Document the discrepancy in reporting footnotes. "
                "No remediation required for minor discrepancies "
                "(5-15%)."
            )

        if materiality == MaterialityLevel.MATERIAL:
            return (
                "Investigate root causes of the material discrepancy "
                "(15-50%). Review waterfall decomposition to identify "
                "primary drivers. Disclose the discrepancy and root "
                "causes in reporting footnotes."
            )

        if materiality == MaterialityLevel.SIGNIFICANT:
            return (
                "Full investigation required for significant "
                "discrepancy (50-100%). Review all contractual "
                "instruments and emission factors. Obtain management "
                "sign-off before reporting. Engage external assurance "
                "provider for verification."
            )

        # EXTREME
        return (
            "Immediate escalation required for extreme discrepancy "
            "(> 100%). This likely indicates a data quality issue, "
            "missing contractual instruments, or fundamental "
            "methodological error. Halt reporting until root cause "
            "is identified and resolved."
        )

    # ==================================================================
    # Private helpers: Materiality summary
    # ==================================================================

    def _build_materiality_summary(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, int]:
        """Build a count of discrepancies by materiality level.

        Args:
            discrepancies: List of discrepancies to summarize.

        Returns:
            Dictionary mapping materiality level name to count.
        """
        summary: Dict[str, int] = {}
        for level in MaterialityLevel:
            count = sum(
                1
                for d in discrepancies
                if d.materiality == level
            )
            if count > 0:
                summary[level.value] = count
        return summary

    # ==================================================================
    # Private helpers: Single-discrepancy flag generation
    # ==================================================================

    def _generate_single_discrepancy_flags(
        self,
        disc: Discrepancy,
    ) -> List[Flag]:
        """Generate flags for a single discrepancy.

        Args:
            disc: Discrepancy to evaluate.

        Returns:
            List of Flag objects for this discrepancy.
        """
        flags: List[Flag] = []

        # Materiality-based flags
        if disc.materiality == MaterialityLevel.EXTREME:
            flags.append(
                Flag(
                    flag_type=FlagType.ERROR,
                    severity=FlagSeverity.CRITICAL,
                    code=_FLAG_CODE_EXTREME_DISCREPANCY,
                    message=(
                        f"Extreme discrepancy detected "
                        f"({disc.percentage:.2f}%): "
                        f"{disc.description}"
                    ),
                    recommendation=(
                        "Immediate investigation required. This "
                        "discrepancy exceeds 100% and likely "
                        "indicates a data quality issue."
                    ),
                )
            )

        elif disc.materiality == MaterialityLevel.SIGNIFICANT:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.HIGH,
                    code=_FLAG_CODE_SIGNIFICANT_DISCREPANCY,
                    message=(
                        f"Significant discrepancy detected "
                        f"({disc.percentage:.2f}%): "
                        f"{disc.discrepancy_type.value} affecting "
                        f"{disc.absolute_tco2e:.4f} tCO2e."
                    ),
                    recommendation=(
                        "Full investigation and management "
                        "sign-off required before reporting."
                    ),
                )
            )

        elif disc.materiality == MaterialityLevel.MATERIAL:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.MEDIUM,
                    code=_FLAG_CODE_MATERIAL_DISCREPANCY,
                    message=(
                        f"Material discrepancy detected "
                        f"({disc.percentage:.2f}%): "
                        f"{disc.discrepancy_type.value} affecting "
                        f"{disc.absolute_tco2e:.4f} tCO2e."
                    ),
                    recommendation=disc.recommendation,
                )
            )

        # Type-specific flags
        if disc.discrepancy_type == DiscrepancyType.PARTIAL_COVERAGE:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.MEDIUM,
                    code=_FLAG_CODE_PARTIAL_COVERAGE,
                    message=(
                        "Partial contractual instrument coverage "
                        "detected. Not all energy consumption is "
                        "covered by instruments."
                    ),
                    recommendation=_DISCREPANCY_RECOMMENDATIONS.get(
                        "PARTIAL_COVERAGE", "",
                    ),
                )
            )

        if disc.discrepancy_type == DiscrepancyType.GEOGRAPHIC_MISMATCH:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.MEDIUM,
                    code=_FLAG_CODE_GEOGRAPHIC_MISMATCH,
                    message=(
                        f"Geographic mismatch: consumption region "
                        f"differs from instrument region for "
                        f"facility "
                        f"{disc.facility_id or 'unknown'}."
                    ),
                    recommendation=_DISCREPANCY_RECOMMENDATIONS.get(
                        "GEOGRAPHIC_MISMATCH", "",
                    ),
                )
            )

        if disc.discrepancy_type == DiscrepancyType.TEMPORAL_MISMATCH:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.LOW,
                    code=_FLAG_CODE_TEMPORAL_MISMATCH,
                    message=(
                        "Temporal mismatch: instrument vintage does "
                        "not align with energy consumption period."
                    ),
                    recommendation=_DISCREPANCY_RECOMMENDATIONS.get(
                        "TEMPORAL_MISMATCH", "",
                    ),
                )
            )

        if disc.discrepancy_type == DiscrepancyType.STEAM_HEAT_METHOD:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.MEDIUM,
                    code=_FLAG_CODE_STEAM_HEAT_DIVERGENCE,
                    message=(
                        "Steam/heating methodological divergence "
                        "detected between location-based and "
                        "market-based approaches."
                    ),
                    recommendation=_DISCREPANCY_RECOMMENDATIONS.get(
                        "STEAM_HEAT_METHOD", "",
                    ),
                )
            )

        if disc.discrepancy_type == DiscrepancyType.GRID_UPDATE_TIMING:
            flags.append(
                Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.LOW,
                    code=_FLAG_CODE_GRID_TIMING,
                    message=(
                        "Grid emission factor vintage mismatch "
                        "between location-based and market-based "
                        "calculations."
                    ),
                    recommendation=_DISCREPANCY_RECOMMENDATIONS.get(
                        "GRID_UPDATE_TIMING", "",
                    ),
                )
            )

        # Direction-based flags
        if disc.direction == DiscrepancyDirection.MARKET_HIGHER:
            flags.append(
                Flag(
                    flag_type=FlagType.INFO,
                    severity=FlagSeverity.MEDIUM,
                    code=_FLAG_CODE_MARKET_HIGHER,
                    message=(
                        "Market-based emissions exceed location-"
                        "based. This is atypical and may indicate "
                        "residual mix uplift or supplier EF issues."
                    ),
                    recommendation=(
                        "Review market-based emission factor "
                        "sources. Consider switching to supplier-"
                        "specific or contractual EFs."
                    ),
                )
            )

        return flags

    # ==================================================================
    # Private helpers: Metrics recording
    # ==================================================================

    def _record_analysis_metrics(
        self,
        workspace: ReconciliationWorkspace,
        report: DiscrepancyReport,
        elapsed_s: float,
    ) -> None:
        """Record Prometheus metrics for the analysis.

        Args:
            workspace: Workspace that was analyzed.
            report: Resulting discrepancy report.
            elapsed_s: Elapsed time in seconds.
        """
        if self._metrics is None:
            return

        try:
            disc_pct = float(_ZERO)
            pif = float(_ZERO)
            if report.waterfall is not None:
                loc = workspace.total_location_tco2e
                mkt = workspace.total_market_tco2e
                if max(loc, mkt) > _ZERO:
                    disc_pct = float(
                        self.calculate_discrepancy_percentage(
                            loc, mkt,
                        )
                    )
                if loc > _ZERO:
                    pif = float(self.calculate_pif(loc, mkt))

            self._metrics.record_reconciliation(
                tenant_id=workspace.tenant_id,
                status="success",
                energy_type="all",
                duration_s=elapsed_s,
                discrepancy_pct=disc_pct,
                pif=pif,
            )

            for disc in report.discrepancies:
                self._metrics.record_discrepancy(
                    discrepancy_type=disc.discrepancy_type.value,
                    materiality=disc.materiality.value,
                    direction=disc.direction.value,
                )

        except Exception as exc:
            logger.warning(
                "Failed to record analysis metrics: %s", exc,
            )

    # ==================================================================
    # Private helpers: Provenance recording
    # ==================================================================

    def _record_provenance(
        self,
        workspace: ReconciliationWorkspace,
        report: DiscrepancyReport,
    ) -> None:
        """Record provenance for the discrepancy analysis stage.

        Args:
            workspace: Workspace that was analyzed.
            report: Resulting discrepancy report.
        """
        if (
            not _PROVENANCE_AVAILABLE
            or DualReportingProvenanceTracker is None
        ):
            return

        try:
            tracker = DualReportingProvenanceTracker.get_instance()

            metadata = {
                "engine_id": _ENGINE_ID,
                "engine_version": _ENGINE_VERSION,
                "reconciliation_id": workspace.reconciliation_id,
                "discrepancy_count": len(report.discrepancies),
                "waterfall_items": (
                    len(report.waterfall.items)
                    if report.waterfall
                    else 0
                ),
                "flag_count": len(report.flags),
                "total_discrepancy_tco2e": str(
                    report.waterfall.total_discrepancy_tco2e
                    if report.waterfall
                    else _ZERO
                ),
            }

            output_hash = _compute_hash({
                "discrepancies": [
                    {
                        "type": d.discrepancy_type.value,
                        "materiality": d.materiality.value,
                        "absolute_tco2e": str(d.absolute_tco2e),
                    }
                    for d in report.discrepancies
                ],
                "waterfall_total": str(
                    report.waterfall.total_discrepancy_tco2e
                    if report.waterfall
                    else _ZERO
                ),
            })

            tracker.add_stage(
                chain_id=workspace.reconciliation_id,
                stage=ProvenanceStage.ANALYZE_DISCREPANCIES,
                metadata=metadata,
                output_data=output_hash,
            )

        except Exception as exc:
            logger.warning(
                "Failed to record provenance: %s", exc,
            )
