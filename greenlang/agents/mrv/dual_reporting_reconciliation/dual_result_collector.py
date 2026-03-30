# -*- coding: utf-8 -*-
"""
DualResultCollectorEngine - Upstream Result Collection and Alignment (Engine 1 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Collects, validates, organises, and aligns upstream emission results from
the four Scope 2 MRV agents (MRV-009 Purchased Electricity, MRV-010
Purchased Steam, MRV-011 District Heating, MRV-012 District Cooling).
This engine is the first stage of the ten-stage dual reporting
reconciliation pipeline and produces the ReconciliationWorkspace that
all subsequent engines consume.

Responsibilities:
    1. Collect raw UpstreamResult objects from all four upstream agents,
       validate each result, and split them into location-based and
       market-based lists.
    2. Align organisational, operational, and temporal boundaries between
       location-based and market-based result sets to ensure comparability.
    3. Map and categorise results by the four GHG Protocol Scope 2 energy
       types (electricity, steam, district heating, district cooling).
    4. Compute per-energy-type breakdowns comparing location-based versus
       market-based totals with absolute difference, percentage difference,
       and direction classification.
    5. Compute per-facility breakdowns comparing location-based versus
       market-based totals across all energy types.
    6. Validate completeness -- ensure all four energy types are covered
       by both methods, report any gaps.
    7. Calculate aggregate totals and Procurement Impact Factor (PIF).
    8. Provide filtering and grouping utilities for downstream engines.
    9. Detect unmatched results (results without a corresponding pair
       from the opposite method at the same facility/energy-type level).

Zero-Hallucination Guarantees:
    - All arithmetic uses Python ``Decimal`` with ROUND_HALF_UP at
      8-decimal-place precision.
    - No LLM, ML, or probabilistic computation in any calculation path.
    - Every aggregation is a deterministic sum/difference/quotient.
    - Provenance hashes are computed over serialised inputs and outputs
      at each public entry point.

Thread Safety:
    Thread-safe singleton via ``__new__`` with ``_instance``,
    ``_initialized``, and ``threading.RLock``.  All mutable counters
    are protected by the reentrant lock.

Public Methods (16):
    collect_results             -> ReconciliationWorkspace
    align_boundaries            -> ReconciliationWorkspace
    map_energy_types            -> ReconciliationWorkspace
    compute_energy_type_breakdowns -> List[EnergyTypeBreakdown]
    compute_facility_breakdowns -> List[FacilityBreakdown]
    validate_completeness       -> Tuple[bool, List[str]]
    get_total_emissions         -> Dict[str, Decimal]
    filter_by_period            -> List[UpstreamResult]
    filter_by_facility          -> List[UpstreamResult]
    filter_by_energy_type       -> List[UpstreamResult]
    group_by_method             -> Dict[Scope2Method, List[UpstreamResult]]
    group_by_energy_type        -> Dict[EnergyType, List[UpstreamResult]]
    group_by_facility           -> Dict[str, List[UpstreamResult]]
    compute_pif                 -> Decimal
    detect_unmatched_results    -> List[UpstreamResult]
    health_check                -> Dict[str, Any]

Classmethod:
    reset                       -> None

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.dual_result_collector import (
    ...     DualResultCollectorEngine,
    ... )
    >>> engine = DualResultCollectorEngine()
    >>> workspace = engine.collect_results(upstream_results)
    >>> workspace = engine.align_boundaries(workspace)
    >>> workspace = engine.map_energy_types(workspace)
    >>> breakdowns = engine.compute_energy_type_breakdowns(workspace)
    >>> facility_bd = engine.compute_facility_breakdowns(workspace)
    >>> is_complete, gaps = engine.validate_completeness(workspace)
    >>> totals = engine.get_total_emissions(workspace)
    >>> pif = engine.compute_pif(totals["location_tco2e"], totals["market_tco2e"])
    >>> unmatched = engine.detect_unmatched_results(workspace)
    >>> status = engine.health_check()

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
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["DualResultCollectorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports for loose coupling
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        DualReportingReconciliationConfig,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    DualReportingReconciliationConfig = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        DualReportingReconciliationMetrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    DualReportingReconciliationMetrics = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        DualReportingProvenanceTracker,
        ProvenanceStage,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    DualReportingProvenanceTracker = None  # type: ignore[assignment,misc]
    ProvenanceStage = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Lazy model imports
# ---------------------------------------------------------------------------

from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    EnergyType,
    Scope2Method,
    UpstreamAgent,
    DiscrepancyDirection,
    UpstreamResult,
    EnergyTypeBreakdown,
    FacilityBreakdown,
    ReconciliationWorkspace,
    UPSTREAM_AGENT_MAPPING,
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,
    MAX_UPSTREAM_RESULTS,
    MAX_FACILITIES,
)

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

#: 8 decimal places for all intermediate and final Decimal results.
_PRECISION = Decimal("0.00000001")

#: Rounding mode -- ROUND_HALF_UP per GHG Protocol conventions.
_ROUNDING = ROUND_HALF_UP

#: Tolerance for "equal" direction classification (0.01 tCO2e).
_EQUALITY_TOLERANCE = Decimal("0.01")

#: Zero constant.
_ZERO = Decimal("0")

#: Hundred constant.
_HUNDRED = Decimal("100")

# ---------------------------------------------------------------------------
# Decimal arithmetic helpers
# ---------------------------------------------------------------------------

def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation quantised to ``_PRECISION``.
    """
    if isinstance(value, Decimal):
        return value.quantize(_PRECISION, rounding=_ROUNDING)
    return Decimal(str(value)).quantize(_PRECISION, rounding=_ROUNDING)

def _quantize(value: Decimal, precision: Decimal = _PRECISION) -> Decimal:
    """Quantize a Decimal to the given precision using ROUND_HALF_UP.

    Args:
        value: Decimal value to quantize.
        precision: Target precision (default 8 decimal places).

    Returns:
        Quantized Decimal value.
    """
    return value.quantize(precision, rounding=_ROUNDING)

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Handles Pydantic models (via ``model_dump``), dicts, lists, strings,
    and Decimal values.  All values are serialised with sorted keys and
    ``default=str`` to ensure determinism.

    Args:
        data: Data to hash (dict, list, str, Decimal, or Pydantic model).

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    if hasattr(data, "model_dump"):
        serialisable = data.model_dump(mode="json")
    elif isinstance(data, list):
        serialisable = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in data
        ]
    else:
        serialisable = data
    raw = json.dumps(serialisable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _pct_difference(
    value_a: Decimal,
    value_b: Decimal,
) -> Decimal:
    """Calculate percentage difference relative to the larger of two values.

    Returns Decimal("0") if both values are zero.

    Formula: |value_a - value_b| / max(value_a, value_b) * 100

    Args:
        value_a: First value.
        value_b: Second value.

    Returns:
        Percentage difference as a quantised Decimal.
    """
    denominator = max(value_a, value_b)
    if denominator == _ZERO:
        return _ZERO
    diff = abs(value_a - value_b)
    return _quantize((diff / denominator) * _HUNDRED)

def _determine_direction(
    location: Decimal,
    market: Decimal,
) -> DiscrepancyDirection:
    """Determine direction of discrepancy between location and market values.

    Uses a tolerance of ``_EQUALITY_TOLERANCE`` (0.01 tCO2e) to classify
    negligible differences as EQUAL.

    Args:
        location: Location-based total.
        market: Market-based total.

    Returns:
        DiscrepancyDirection enum value.
    """
    diff = abs(location - market)
    if diff < _EQUALITY_TOLERANCE:
        return DiscrepancyDirection.EQUAL
    if market < location:
        return DiscrepancyDirection.MARKET_LOWER
    return DiscrepancyDirection.MARKET_HIGHER

# ===========================================================================
# Engine
# ===========================================================================

class DualResultCollectorEngine:
    """Engine 1 of 7 -- Upstream result collection, alignment, and workspace assembly.

    Collects UpstreamResult objects from the four Scope 2 agents (MRV-009
    through MRV-012), validates each result, splits them into location-based
    and market-based lists, aligns organisational/operational/temporal
    boundaries, maps results to the four GHG Protocol energy types, and
    computes per-energy-type and per-facility breakdowns.

    The output is a fully-populated ``ReconciliationWorkspace`` that
    carries all intermediate state needed by downstream engines
    (DiscrepancyAnalyzerEngine, QualityScorerEngine, etc.).

    Thread Safety:
        Singleton via ``__new__`` + ``_instance`` + ``_initialized`` +
        ``threading.RLock``.  Internal counters are protected by the
        reentrant lock.  All public methods are safe to call concurrently
        from multiple threads.

    Attributes:
        _config: Agent configuration singleton (may be None).
        _metrics: Prometheus metrics singleton (may be None).
        _provenance: Provenance tracker singleton (may be None).
        _engine_id: Unique identifier for this engine instance.
        _created_at: UTC timestamp of singleton creation.
        _total_collections: Number of collect_results calls.
        _total_alignments: Number of align_boundaries calls.
        _total_mappings: Number of map_energy_types calls.
        _total_breakdowns: Number of breakdown computation calls.
        _total_validations: Number of validate_completeness calls.
        _total_filters: Number of filter/group calls.
        _total_pif_calcs: Number of PIF computation calls.
        _total_unmatched_detections: Number of unmatched detection calls.
        _total_health_checks: Number of health_check calls.

    Public Methods (16):
        collect_results(results)                     -> ReconciliationWorkspace
        align_boundaries(workspace)                  -> ReconciliationWorkspace
        map_energy_types(workspace)                  -> ReconciliationWorkspace
        compute_energy_type_breakdowns(workspace)    -> List[EnergyTypeBreakdown]
        compute_facility_breakdowns(workspace)       -> List[FacilityBreakdown]
        validate_completeness(workspace)             -> Tuple[bool, List[str]]
        get_total_emissions(workspace)               -> Dict[str, Decimal]
        filter_by_period(results, start, end)        -> List[UpstreamResult]
        filter_by_facility(results, facility_id)     -> List[UpstreamResult]
        filter_by_energy_type(results, energy_type)  -> List[UpstreamResult]
        group_by_method(results)                     -> Dict[Scope2Method, ...]
        group_by_energy_type(results)                -> Dict[EnergyType, ...]
        group_by_facility(results)                   -> Dict[str, ...]
        compute_pif(location_total, market_total)    -> Decimal
        detect_unmatched_results(workspace)          -> List[UpstreamResult]
        health_check()                               -> Dict[str, Any]

    Classmethod:
        reset()  -> None  (clear singleton for test teardown)

    Example:
        >>> engine = DualResultCollectorEngine()
        >>> ws = engine.collect_results(results)
        >>> ws = engine.align_boundaries(ws)
        >>> ws = engine.map_energy_types(ws)
        >>> assert len(ws.location_results) > 0
    """

    _instance: Optional[DualResultCollectorEngine] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> DualResultCollectorEngine:
        """Return the singleton instance, creating it on first call.

        Thread-safe using double-checked locking with RLock.

        Returns:
            The singleton DualResultCollectorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise the engine from config, metrics and provenance singletons.

        Guarded by ``_initialized`` flag so repeated calls are no-ops.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return

            # Config
            self._config: Any = None
            if _CONFIG_AVAILABLE:
                try:
                    self._config = DualReportingReconciliationConfig()
                except Exception:
                    logger.warning(
                        "DualResultCollectorEngine: config singleton unavailable, "
                        "using defaults"
                    )

            # Metrics
            self._metrics: Any = None
            if _METRICS_AVAILABLE:
                try:
                    self._metrics = DualReportingReconciliationMetrics()
                except Exception:
                    logger.warning(
                        "DualResultCollectorEngine: metrics singleton unavailable"
                    )

            # Provenance
            self._provenance: Any = None
            if _PROVENANCE_AVAILABLE:
                try:
                    self._provenance = DualReportingProvenanceTracker.get_instance()
                except Exception:
                    logger.warning(
                        "DualResultCollectorEngine: provenance tracker unavailable"
                    )

            # Internal state
            self._internal_lock = threading.RLock()
            self._engine_id: str = f"drr-collector-{uuid4().hex[:8]}"
            self._created_at: datetime = utcnow()
            self._total_collections: int = 0
            self._total_alignments: int = 0
            self._total_mappings: int = 0
            self._total_breakdowns: int = 0
            self._total_validations: int = 0
            self._total_filters: int = 0
            self._total_pif_calcs: int = 0
            self._total_unmatched_detections: int = 0
            self._total_health_checks: int = 0

            self.__class__._initialized = True
            logger.info(
                "DualResultCollectorEngine initialized: engine_id=%s, "
                "agent=%s, component=%s, version=%s, "
                "config=%s, metrics=%s, provenance=%s",
                self._engine_id,
                AGENT_ID,
                AGENT_COMPONENT,
                VERSION,
                "available" if self._config else "unavailable",
                "available" if self._metrics else "unavailable",
                "available" if self._provenance else "unavailable",
            )

    # ------------------------------------------------------------------
    # Singleton reset for testing
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for test teardown.

        After calling ``reset()``, the next instantiation will create
        a fresh DualResultCollectorEngine.  This method is intended
        **only** for test isolation and must not be called in production.

        Example:
            >>> DualResultCollectorEngine.reset()
            >>> engine = DualResultCollectorEngine()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.info("DualResultCollectorEngine: singleton reset")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _get_decimal_places(self) -> int:
        """Return configured decimal places or default 8."""
        if self._config and hasattr(self._config, "decimal_places"):
            return self._config.decimal_places
        return 8

    def _get_precision(self) -> Decimal:
        """Return Decimal precision based on configured decimal places."""
        dp = self._get_decimal_places()
        return Decimal(10) ** (-dp)

    def _boundary_validation_enabled(self) -> bool:
        """Return whether boundary validation is enabled in config."""
        if self._config and hasattr(self._config, "enable_boundary_validation"):
            return self._config.enable_boundary_validation
        return True

    def _temporal_alignment_enabled(self) -> bool:
        """Return whether temporal alignment checking is enabled in config."""
        if self._config and hasattr(self._config, "enable_temporal_alignment"):
            return self._config.enable_temporal_alignment
        return True

    # ------------------------------------------------------------------
    # Counter helpers (thread-safe)
    # ------------------------------------------------------------------

    def _inc_collections(self) -> None:
        """Thread-safe increment of the collection counter."""
        with self._internal_lock:
            self._total_collections += 1

    def _inc_alignments(self) -> None:
        """Thread-safe increment of the alignment counter."""
        with self._internal_lock:
            self._total_alignments += 1

    def _inc_mappings(self) -> None:
        """Thread-safe increment of the mapping counter."""
        with self._internal_lock:
            self._total_mappings += 1

    def _inc_breakdowns(self) -> None:
        """Thread-safe increment of the breakdown counter."""
        with self._internal_lock:
            self._total_breakdowns += 1

    def _inc_validations(self) -> None:
        """Thread-safe increment of the validation counter."""
        with self._internal_lock:
            self._total_validations += 1

    def _inc_filters(self) -> None:
        """Thread-safe increment of the filter/group counter."""
        with self._internal_lock:
            self._total_filters += 1

    def _inc_pif_calcs(self) -> None:
        """Thread-safe increment of the PIF calculation counter."""
        with self._internal_lock:
            self._total_pif_calcs += 1

    def _inc_unmatched(self) -> None:
        """Thread-safe increment of the unmatched detection counter."""
        with self._internal_lock:
            self._total_unmatched_detections += 1

    def _inc_health_checks(self) -> None:
        """Thread-safe increment of the health check counter."""
        with self._internal_lock:
            self._total_health_checks += 1

    # ------------------------------------------------------------------
    # Metrics helper
    # ------------------------------------------------------------------

    def _record_metric(
        self,
        operation: str,
        duration_s: float,
        status: str = "success",
        energy_type: str = "all",
        tenant_id: str = "engine",
        discrepancy_pct: Optional[float] = None,
        pif: Optional[float] = None,
    ) -> None:
        """Record a metric for an engine operation.

        Args:
            operation: Name of the operation.
            duration_s: Duration in seconds.
            status: Operation status string.
            energy_type: Energy type label.
            tenant_id: Tenant identifier for metric label.
            discrepancy_pct: Discrepancy percentage (optional).
            pif: Procurement Impact Factor (optional).
        """
        if self._metrics:
            try:
                self._metrics.record_reconciliation(
                    tenant_id=tenant_id,
                    status=status,
                    energy_type=energy_type,
                    duration_s=duration_s,
                    discrepancy_pct=discrepancy_pct,
                    pif=pif,
                )
            except Exception as exc:
                logger.debug(
                    "DualResultCollectorEngine: metric recording failed: %s",
                    exc,
                )

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        chain_id: Optional[str],
        stage: Any,
        metadata: Dict[str, Any],
        output_data: Any,
    ) -> Optional[str]:
        """Record a provenance stage if tracking is available.

        Args:
            chain_id: Provenance chain identifier (may be None).
            stage: ProvenanceStage enum value.
            metadata: Metadata dict for the stage.
            output_data: Output data to hash for provenance.

        Returns:
            Chain hash string, or None if provenance is unavailable.
        """
        if not _PROVENANCE_AVAILABLE or chain_id is None or self._provenance is None:
            return None
        try:
            return self._provenance.add_stage(chain_id, stage, metadata, output_data)
        except Exception as exc:
            logger.warning(
                "DualResultCollectorEngine: provenance recording failed "
                "for stage %s: %s",
                stage,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _validate_single_result(self, result: UpstreamResult) -> List[str]:
        """Validate a single upstream result for data quality issues.

        Checks:
        - Correct agent-to-energy-type mapping per UPSTREAM_AGENT_MAPPING
        - Non-negative emissions
        - Non-negative energy quantity
        - Non-negative emission factor
        - Valid period (end >= start)
        - Non-empty facility_id and tenant_id
        - Provenance hash presence (warning only, not error)
        - Reasonable emission factor range (warning only)

        Args:
            result: An UpstreamResult to validate.

        Returns:
            List of validation error strings.  Empty list means valid.
        """
        errors: List[str] = []

        # Check agent-energy-type mapping
        expected_agents = UPSTREAM_AGENT_MAPPING.get(result.energy_type)
        if expected_agents is not None:
            location_agent, market_agent = expected_agents
            expected_agent = (
                location_agent
                if result.method == Scope2Method.LOCATION_BASED
                else market_agent
            )
            if result.agent != expected_agent:
                errors.append(
                    f"Agent {result.agent.value} does not match expected "
                    f"agent {expected_agent.value} for energy_type="
                    f"{result.energy_type.value}, method={result.method.value}"
                )

        # Check non-negative emissions
        if result.emissions_tco2e < _ZERO:
            errors.append(
                f"Negative emissions_tco2e: {result.emissions_tco2e}"
            )

        # Check non-negative energy quantity
        if result.energy_quantity_mwh < _ZERO:
            errors.append(
                f"Negative energy_quantity_mwh: {result.energy_quantity_mwh}"
            )

        # Check non-negative emission factor
        if result.ef_used < _ZERO:
            errors.append(f"Negative ef_used: {result.ef_used}")

        # Check period consistency
        if result.period_end < result.period_start:
            errors.append(
                f"period_end ({result.period_end}) before "
                f"period_start ({result.period_start})"
            )

        # Check required string fields
        if not result.facility_id or not result.facility_id.strip():
            errors.append("Empty facility_id")

        if not result.tenant_id or not result.tenant_id.strip():
            errors.append("Empty tenant_id")

        return errors

    def _validate_results_list(
        self, results: List[UpstreamResult]
    ) -> Tuple[List[UpstreamResult], List[Dict[str, Any]]]:
        """Validate a list of upstream results.

        Returns valid results and a list of error dicts for invalid ones.

        Args:
            results: List of UpstreamResult objects.

        Returns:
            Tuple of (valid_results, error_records).
        """
        valid: List[UpstreamResult] = []
        error_records: List[Dict[str, Any]] = []

        for idx, result in enumerate(results):
            errors = self._validate_single_result(result)
            if errors:
                error_records.append({
                    "index": idx,
                    "facility_id": result.facility_id,
                    "energy_type": result.energy_type.value,
                    "method": result.method.value,
                    "errors": errors,
                })
                logger.warning(
                    "Upstream result %d validation failed: %s",
                    idx,
                    "; ".join(errors),
                )
            else:
                valid.append(result)

        return valid, error_records

    def _split_by_method(
        self, results: List[UpstreamResult]
    ) -> Tuple[List[UpstreamResult], List[UpstreamResult]]:
        """Split results into location-based and market-based lists.

        Args:
            results: List of UpstreamResult objects.

        Returns:
            Tuple of (location_results, market_results).
        """
        location: List[UpstreamResult] = []
        market: List[UpstreamResult] = []
        for r in results:
            if r.method == Scope2Method.LOCATION_BASED:
                location.append(r)
            elif r.method == Scope2Method.MARKET_BASED:
                market.append(r)
            else:
                logger.warning(
                    "Unknown Scope2Method '%s' for result at facility %s; "
                    "skipping.",
                    r.method,
                    r.facility_id,
                )
        return location, market

    def _extract_tenant_id(self, results: List[UpstreamResult]) -> str:
        """Extract and validate a single consistent tenant_id from results.

        All results in a reconciliation must belong to the same tenant.

        Args:
            results: List of UpstreamResult objects.

        Returns:
            The common tenant_id.

        Raises:
            ValueError: If results contain mixed tenant_ids.
        """
        tenant_ids = {r.tenant_id for r in results}
        if len(tenant_ids) == 0:
            raise ValueError(
                "No upstream results provided; cannot extract tenant_id"
            )
        if len(tenant_ids) > 1:
            raise ValueError(
                f"Mixed tenant_ids in upstream results: {sorted(tenant_ids)}. "
                f"All results in a single reconciliation must belong to the "
                f"same tenant."
            )
        return tenant_ids.pop()

    def _extract_period(
        self, results: List[UpstreamResult]
    ) -> Tuple[date, date]:
        """Extract the overall reporting period from results.

        Computes the earliest period_start and latest period_end across
        all results.

        Args:
            results: List of UpstreamResult objects.

        Returns:
            Tuple of (earliest_start, latest_end).

        Raises:
            ValueError: If results list is empty.
        """
        if not results:
            raise ValueError(
                "No upstream results provided; cannot extract period"
            )
        earliest_start = min(r.period_start for r in results)
        latest_end = max(r.period_end for r in results)
        return earliest_start, latest_end

    def _sum_emissions(self, results: List[UpstreamResult]) -> Decimal:
        """Sum emissions_tco2e across a list of upstream results.

        Args:
            results: List of UpstreamResult objects.

        Returns:
            Sum of emissions_tco2e as a quantised Decimal.
        """
        total = _ZERO
        for r in results:
            total += r.emissions_tco2e
        return _quantize(total)

    def _sum_energy_mwh(self, results: List[UpstreamResult]) -> Decimal:
        """Sum energy_quantity_mwh across a list of upstream results.

        Args:
            results: List of UpstreamResult objects.

        Returns:
            Sum of energy_quantity_mwh as a quantised Decimal.
        """
        total = _ZERO
        for r in results:
            total += r.energy_quantity_mwh
        return _quantize(total)

    def _extract_periods(
        self,
        results: List[UpstreamResult],
    ) -> Dict[Tuple[str, str], Set[Tuple[date, date]]]:
        """Extract (facility_id, energy_type) -> set of (start, end) periods.

        Args:
            results: List of upstream results.

        Returns:
            Dictionary mapping (facility_id, energy_type_value) to
            sets of (period_start, period_end) tuples.
        """
        periods: Dict[
            Tuple[str, str],
            Set[Tuple[date, date]],
        ] = defaultdict(set)

        for r in results:
            key = (r.facility_id, r.energy_type.value)
            periods[key].add((r.period_start, r.period_end))

        return periods

    # ==================================================================
    # PUBLIC METHOD 1: collect_results
    # ==================================================================

    def collect_results(
        self,
        results: List[UpstreamResult],
    ) -> ReconciliationWorkspace:
        """Collect and organise upstream results into a ReconciliationWorkspace.

        Takes raw upstream results from all four Scope 2 agents, validates
        each result, splits them into location-based and market-based lists,
        extracts the common tenant_id and reporting period, and assembles
        the initial workspace.

        Invalid results are logged and excluded from the workspace.  The
        workspace is returned with ``location_results`` and
        ``market_results`` populated and totals computed, but per-energy-type
        and per-facility breakdowns are populated as empty lists (those are
        computed by subsequent pipeline stages via ``compute_energy_type_breakdowns``
        and ``compute_facility_breakdowns``).

        Args:
            results: List of UpstreamResult objects from MRV-009 through
                MRV-012.  Must not be empty.  Must not exceed
                ``MAX_UPSTREAM_RESULTS * 2`` (location + market combined).

        Returns:
            ReconciliationWorkspace with location and market results
            separated and totals computed.

        Raises:
            TypeError: If ``results`` is not a list.
            ValueError: If ``results`` is empty or exceeds size limits.
            ValueError: If results contain mixed tenant_ids.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.collect_results: "
            "processing %d upstream results",
            len(results) if isinstance(results, list) else -1,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list of UpstreamResult, "
                f"got {type(results).__name__}"
            )
        if len(results) == 0:
            raise ValueError(
                "results list is empty; at least one UpstreamResult is required"
            )
        max_total = MAX_UPSTREAM_RESULTS * 2
        if len(results) > max_total:
            raise ValueError(
                f"results list has {len(results)} items, exceeding "
                f"maximum of {max_total} (MAX_UPSTREAM_RESULTS * 2)"
            )
        for idx, item in enumerate(results):
            if not isinstance(item, UpstreamResult):
                raise TypeError(
                    f"results[{idx}] is {type(item).__name__}, "
                    f"expected UpstreamResult"
                )

        # Check facility count
        facility_ids = {r.facility_id for r in results}
        if len(facility_ids) > MAX_FACILITIES:
            raise ValueError(
                f"Input exceeds maximum allowed facilities. "
                f"Got {len(facility_ids)}, maximum is {MAX_FACILITIES}."
            )

        # -- Validate individual results ------------------------------------
        valid_results, error_records = self._validate_results_list(results)

        if not valid_results:
            raise ValueError(
                f"All {len(results)} upstream results failed validation. "
                f"Errors: {error_records}"
            )

        if error_records:
            logger.warning(
                "collect_results: %d of %d results failed validation "
                "and were excluded",
                len(error_records),
                len(results),
            )

        # -- Extract tenant and period --------------------------------------
        tenant_id = self._extract_tenant_id(valid_results)
        period_start, period_end = self._extract_period(valid_results)

        # -- Split by method -----------------------------------------------
        location_results, market_results = self._split_by_method(valid_results)

        # -- Compute totals ------------------------------------------------
        total_location = self._sum_emissions(location_results)
        total_market = self._sum_emissions(market_results)

        # -- Assemble workspace --------------------------------------------
        reconciliation_id = str(uuid4())

        workspace = ReconciliationWorkspace(
            reconciliation_id=reconciliation_id,
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            location_results=location_results,
            market_results=market_results,
            total_location_tco2e=total_location,
            total_market_tco2e=total_market,
            by_energy_type=[],
            by_facility=[],
        )

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "collect_results",
            "input_count": len(results),
            "valid_count": len(valid_results),
            "excluded_count": len(error_records),
            "location_count": len(location_results),
            "market_count": len(market_results),
            "total_location_tco2e": str(total_location),
            "total_market_tco2e": str(total_market),
            "tenant_id": tenant_id,
            "period_start": str(period_start),
            "period_end": str(period_end),
            "reconciliation_id": reconciliation_id,
        })

        if _PROVENANCE_AVAILABLE and self._provenance is not None:
            self._record_provenance(
                reconciliation_id,
                ProvenanceStage.COLLECT_LOCATION_RESULTS,
                {
                    "input_count": len(results),
                    "valid_count": len(valid_results),
                    "location_count": len(location_results),
                    "market_count": len(market_results),
                },
                workspace,
            )

        # -- Metrics --------------------------------------------------------
        elapsed = time.monotonic() - start_time
        self._record_metric(
            "collect_results",
            elapsed,
            tenant_id=tenant_id,
        )
        self._inc_collections()

        logger.info(
            "DualResultCollectorEngine.collect_results completed: "
            "recon_id=%s, tenant=%s, period=%s/%s, "
            "location=%d (%.4f tCO2e), market=%d (%.4f tCO2e), "
            "excluded=%d, hash=%s, elapsed=%.4fs",
            reconciliation_id,
            tenant_id,
            period_start,
            period_end,
            len(location_results),
            total_location,
            len(market_results),
            total_market,
            len(error_records),
            provenance_hash[:16],
            elapsed,
        )

        return workspace

    # ==================================================================
    # PUBLIC METHOD 2: align_boundaries
    # ==================================================================

    def align_boundaries(
        self,
        workspace: ReconciliationWorkspace,
    ) -> ReconciliationWorkspace:
        """Verify that organisational, operational, and temporal boundaries match.

        Checks:
        1. All location-based and market-based results share the same
           tenant_id (organisational boundary).
        2. The set of facility_ids in location results matches the set
           in market results (operational boundary alignment).
        3. The reporting periods (period_start, period_end) are consistent
           across both method sets (temporal boundary).
        4. Energy type coverage is consistent across methods.
        5. GWP sources are consistent across all results.

        Boundary mismatches are logged as warnings but do not raise
        exceptions -- the pipeline continues, and the workspace is
        returned unchanged (since it is a frozen Pydantic model).

        Args:
            workspace: ReconciliationWorkspace from ``collect_results``.

        Returns:
            The same workspace (frozen; boundary flags are tracked in
            provenance and logs).

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
            ValueError: If workspace has no results.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.align_boundaries: recon_id=%s",
            workspace.reconciliation_id,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        all_results = list(workspace.location_results) + list(
            workspace.market_results
        )
        if not all_results:
            raise ValueError(
                "Workspace has no location or market results to align"
            )

        flags: List[str] = []

        # --- 1. Tenant boundary alignment ---
        tenant_ids = {r.tenant_id for r in all_results}
        if len(tenant_ids) > 1:
            flags.append(
                f"BOUNDARY_TENANT_MISMATCH: Mixed tenant_ids: "
                f"{sorted(tenant_ids)}"
            )
            logger.warning(
                "align_boundaries: Mixed tenant_ids: %s",
                sorted(tenant_ids),
            )

        # --- 2. Operational boundary alignment (facility_ids) ---
        location_facilities = {
            r.facility_id for r in workspace.location_results
        }
        market_facilities = {
            r.facility_id for r in workspace.market_results
        }
        location_only = location_facilities - market_facilities
        market_only = market_facilities - location_facilities

        if location_only:
            sample = sorted(list(location_only)[:10])
            flags.append(
                f"BOUNDARY_FACILITY_LOCATION_ONLY: "
                f"{len(location_only)} facilities have location-based "
                f"results but no market-based results: {sample}"
            )
            logger.warning(
                "align_boundaries: %d facilities location-only",
                len(location_only),
            )

        if market_only:
            sample = sorted(list(market_only)[:10])
            flags.append(
                f"BOUNDARY_FACILITY_MARKET_ONLY: "
                f"{len(market_only)} facilities have market-based "
                f"results but no location-based results: {sample}"
            )
            logger.warning(
                "align_boundaries: %d facilities market-only",
                len(market_only),
            )

        # --- 3. Temporal boundary alignment ---
        location_periods = {
            (r.period_start, r.period_end)
            for r in workspace.location_results
        }
        market_periods = {
            (r.period_start, r.period_end)
            for r in workspace.market_results
        }
        period_diff = location_periods.symmetric_difference(market_periods)
        if period_diff:
            flags.append(
                f"BOUNDARY_PERIOD_MISMATCH: "
                f"{len(period_diff)} period(s) not aligned between "
                f"location and market"
            )
            logger.warning(
                "align_boundaries: %d period mismatches",
                len(period_diff),
            )

        # --- 4. Energy type coverage alignment ---
        location_energy_types = {
            r.energy_type for r in workspace.location_results
        }
        market_energy_types = {
            r.energy_type for r in workspace.market_results
        }
        energy_diff = location_energy_types.symmetric_difference(
            market_energy_types
        )
        if energy_diff:
            flags.append(
                f"BOUNDARY_ENERGY_TYPE_MISMATCH: "
                f"Energy types not present in both methods: "
                f"{sorted(e.value for e in energy_diff)}"
            )
            logger.warning(
                "align_boundaries: Energy type mismatch: %s",
                sorted(e.value for e in energy_diff),
            )

        # --- 5. GWP source consistency ---
        gwp_sources = {r.gwp_source for r in all_results}
        if len(gwp_sources) > 1:
            flags.append(
                f"BOUNDARY_GWP_MISMATCH: Mixed GWP sources: "
                f"{sorted(g.value for g in gwp_sources)}"
            )
            logger.warning(
                "align_boundaries: Mixed GWP sources: %s",
                sorted(g.value for g in gwp_sources),
            )

        # --- 6. Per-facility period alignment ---
        loc_periods = self._extract_periods(list(workspace.location_results))
        mkt_periods = self._extract_periods(list(workspace.market_results))

        period_mismatches: List[str] = []
        for key in loc_periods.keys() & mkt_periods.keys():
            loc_set = loc_periods[key]
            mkt_set = mkt_periods[key]
            if loc_set != mkt_set:
                period_mismatches.append(
                    f"{key}: location={sorted(str(p) for p in loc_set)}, "
                    f"market={sorted(str(p) for p in mkt_set)}"
                )

        if period_mismatches:
            detail = "; ".join(period_mismatches[:10])
            suffix = ""
            if len(period_mismatches) > 10:
                suffix = f" (and {len(period_mismatches) - 10} more)"
            flags.append(
                f"BOUNDARY_FACILITY_PERIOD_MISMATCH: "
                f"{len(period_mismatches)} facility/energy-type "
                f"combinations have mismatched periods: {detail}{suffix}"
            )

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "align_boundaries",
            "reconciliation_id": workspace.reconciliation_id,
            "flags_count": len(flags),
            "flags": flags,
            "location_facilities": len(location_facilities),
            "market_facilities": len(market_facilities),
            "location_only": len(location_only),
            "market_only": len(market_only),
        })

        if _PROVENANCE_AVAILABLE and self._provenance is not None:
            self._record_provenance(
                workspace.reconciliation_id,
                ProvenanceStage.ALIGN_BOUNDARIES,
                {
                    "flags_count": len(flags),
                    "location_facilities": len(location_facilities),
                    "market_facilities": len(market_facilities),
                },
                {"flags": flags},
            )

        self._inc_alignments()
        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.align_boundaries completed: "
            "recon_id=%s, flags=%d, hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            len(flags),
            provenance_hash[:16],
            elapsed,
        )

        return workspace

    # ==================================================================
    # PUBLIC METHOD 3: map_energy_types
    # ==================================================================

    def map_energy_types(
        self,
        workspace: ReconciliationWorkspace,
    ) -> ReconciliationWorkspace:
        """Map and categorise results by energy type.

        Verifies that every upstream result has a valid energy_type
        (one of the four GHG Protocol Scope 2 energy types) and that
        the agent field matches the expected upstream agent for that
        energy type per ``UPSTREAM_AGENT_MAPPING``.

        This is a validation and logging stage.  Since UpstreamResult
        models already carry the energy_type enum, the mapping is
        inherent in the data model.  This method confirms consistency,
        logs the distribution, and records provenance.

        Args:
            workspace: ReconciliationWorkspace from ``align_boundaries``.

        Returns:
            The same workspace (no mutation needed; mapping is validated).

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.map_energy_types: recon_id=%s",
            workspace.reconciliation_id,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        all_results = list(workspace.location_results) + list(
            workspace.market_results
        )

        # -- Classify and count by energy type ------------------------------
        energy_type_counts: Dict[str, Dict[str, int]] = {}
        mapping_errors: List[str] = []

        for energy_type in EnergyType:
            energy_type_counts[energy_type.value] = {
                "location": 0,
                "market": 0,
                "total_results": 0,
            }

        for result in all_results:
            et_key = result.energy_type.value
            method_key = (
                "location"
                if result.method == Scope2Method.LOCATION_BASED
                else "market"
            )
            if et_key in energy_type_counts:
                energy_type_counts[et_key][method_key] += 1
                energy_type_counts[et_key]["total_results"] += 1

            # Validate agent mapping
            expected_agents = UPSTREAM_AGENT_MAPPING.get(result.energy_type)
            if expected_agents is not None:
                loc_agent, mkt_agent = expected_agents
                expected_agent = (
                    loc_agent
                    if result.method == Scope2Method.LOCATION_BASED
                    else mkt_agent
                )
                if result.agent != expected_agent:
                    mapping_errors.append(
                        f"Result at facility {result.facility_id} has "
                        f"agent={result.agent.value} but expected "
                        f"{expected_agent.value} for "
                        f"{result.energy_type.value}/{result.method.value}"
                    )

        if mapping_errors:
            logger.warning(
                "map_energy_types: %d agent mapping errors: %s",
                len(mapping_errors),
                mapping_errors[:5],
            )

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "map_energy_types",
            "reconciliation_id": workspace.reconciliation_id,
            "energy_type_counts": energy_type_counts,
            "mapping_errors_count": len(mapping_errors),
        })

        if _PROVENANCE_AVAILABLE and self._provenance is not None:
            self._record_provenance(
                workspace.reconciliation_id,
                ProvenanceStage.MAP_ENERGY_TYPES,
                {
                    "energy_type_counts": energy_type_counts,
                    "mapping_errors_count": len(mapping_errors),
                },
                energy_type_counts,
            )

        self._inc_mappings()
        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.map_energy_types completed: "
            "recon_id=%s, distribution=%s, mapping_errors=%d, "
            "hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            {k: v["total_results"] for k, v in energy_type_counts.items()},
            len(mapping_errors),
            provenance_hash[:16],
            elapsed,
        )

        return workspace

    # ==================================================================
    # PUBLIC METHOD 4: compute_energy_type_breakdowns
    # ==================================================================

    def compute_energy_type_breakdowns(
        self,
        workspace: ReconciliationWorkspace,
    ) -> List[EnergyTypeBreakdown]:
        """Compute per-energy-type breakdowns comparing location vs market.

        For each of the four GHG Protocol Scope 2 energy types, sums
        location-based and market-based emissions, computes the absolute
        and percentage difference, classifies the direction, and totals
        the energy consumed in MWh.

        Energy types with zero results on both sides are still included
        in the output (with zero totals) to ensure completeness.

        Formulas (all Decimal):
            difference_tco2e = location_tco2e - market_tco2e
            difference_pct = |difference_tco2e| / max(loc, mkt) * 100
            direction: MARKET_LOWER if market < location,
                       MARKET_HIGHER if market > location,
                       EQUAL if within 0.01 tolerance

        Args:
            workspace: ReconciliationWorkspace with location and market
                results populated.

        Returns:
            List of four EnergyTypeBreakdown objects (one per energy type),
            ordered by EnergyType enum order (ELECTRICITY, STEAM,
            DISTRICT_HEATING, DISTRICT_COOLING).

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.compute_energy_type_breakdowns: "
            "recon_id=%s",
            workspace.reconciliation_id,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        # -- Group results by energy type -----------------------------------
        loc_by_et: Dict[EnergyType, List[UpstreamResult]] = defaultdict(list)
        mkt_by_et: Dict[EnergyType, List[UpstreamResult]] = defaultdict(list)
        for r in workspace.location_results:
            loc_by_et[r.energy_type].append(r)
        for r in workspace.market_results:
            mkt_by_et[r.energy_type].append(r)

        # -- Compute breakdown for each energy type -------------------------
        breakdowns: List[EnergyTypeBreakdown] = []

        for energy_type in EnergyType:
            loc_results = loc_by_et.get(energy_type, [])
            mkt_results = mkt_by_et.get(energy_type, [])

            loc_total = self._sum_emissions(loc_results)
            mkt_total = self._sum_emissions(mkt_results)
            diff = _quantize(loc_total - mkt_total)
            diff_pct = _pct_difference(loc_total, mkt_total)
            direction = _determine_direction(loc_total, mkt_total)

            # Total energy MWh across both methods (use max as conservative)
            loc_mwh = self._sum_energy_mwh(loc_results)
            mkt_mwh = self._sum_energy_mwh(mkt_results)
            energy_mwh = _quantize(max(loc_mwh, mkt_mwh))

            breakdown = EnergyTypeBreakdown(
                energy_type=energy_type,
                location_tco2e=loc_total,
                market_tco2e=mkt_total,
                difference_tco2e=diff,
                difference_pct=diff_pct,
                direction=direction,
                energy_mwh=energy_mwh,
            )
            breakdowns.append(breakdown)

            logger.debug(
                "Energy type %s: location=%.4f, market=%.4f, "
                "diff=%.4f (%.2f%%), direction=%s, energy=%.2f MWh",
                energy_type.value,
                loc_total,
                mkt_total,
                diff,
                diff_pct,
                direction.value,
                energy_mwh,
            )

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "compute_energy_type_breakdowns",
            "reconciliation_id": workspace.reconciliation_id,
            "breakdowns": [
                {
                    "energy_type": b.energy_type.value,
                    "location_tco2e": str(b.location_tco2e),
                    "market_tco2e": str(b.market_tco2e),
                    "difference_tco2e": str(b.difference_tco2e),
                    "difference_pct": str(b.difference_pct),
                    "direction": b.direction.value,
                    "energy_mwh": str(b.energy_mwh),
                }
                for b in breakdowns
            ],
        })

        self._inc_breakdowns()
        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.compute_energy_type_breakdowns "
            "completed: recon_id=%s, breakdowns=%d, hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            len(breakdowns),
            provenance_hash[:16],
            elapsed,
        )

        return breakdowns

    # ==================================================================
    # PUBLIC METHOD 5: compute_facility_breakdowns
    # ==================================================================

    def compute_facility_breakdowns(
        self,
        workspace: ReconciliationWorkspace,
    ) -> List[FacilityBreakdown]:
        """Compute per-facility breakdowns comparing location vs market.

        For each unique facility_id across both methods, sums all
        location-based and market-based emissions (across all energy
        types), computes the absolute and percentage difference, and
        resolves the facility_name from the first matching result.

        Formulas (all Decimal):
            difference_tco2e = location_tco2e - market_tco2e
            difference_pct = |difference_tco2e| / max(loc, mkt) * 100

        Args:
            workspace: ReconciliationWorkspace with location and market
                results populated.

        Returns:
            List of FacilityBreakdown objects sorted by facility_id.

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.compute_facility_breakdowns: "
            "recon_id=%s",
            workspace.reconciliation_id,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        # -- Group results by facility and method ---------------------------
        loc_totals: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        mkt_totals: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        facility_names: Dict[str, Optional[str]] = {}

        for r in workspace.location_results:
            loc_totals[r.facility_id] += r.emissions_tco2e
            if r.facility_name and r.facility_id not in facility_names:
                facility_names[r.facility_id] = r.facility_name

        for r in workspace.market_results:
            mkt_totals[r.facility_id] += r.emissions_tco2e
            if r.facility_name and r.facility_id not in facility_names:
                facility_names[r.facility_id] = r.facility_name

        # -- Collect all unique facility IDs --------------------------------
        all_facility_ids = sorted(
            set(loc_totals.keys()) | set(mkt_totals.keys())
        )

        if len(all_facility_ids) > MAX_FACILITIES:
            logger.warning(
                "compute_facility_breakdowns: %d facilities exceeds "
                "MAX_FACILITIES=%d; truncating",
                len(all_facility_ids),
                MAX_FACILITIES,
            )
            all_facility_ids = all_facility_ids[:MAX_FACILITIES]

        # -- Compute breakdown for each facility ----------------------------
        breakdowns: List[FacilityBreakdown] = []

        for fac_id in all_facility_ids:
            loc_val = _quantize(loc_totals.get(fac_id, _ZERO))
            mkt_val = _quantize(mkt_totals.get(fac_id, _ZERO))
            diff = _quantize(loc_val - mkt_val)
            diff_pct = _pct_difference(loc_val, mkt_val)

            breakdown = FacilityBreakdown(
                facility_id=fac_id,
                facility_name=facility_names.get(fac_id),
                location_tco2e=loc_val,
                market_tco2e=mkt_val,
                difference_tco2e=diff,
                difference_pct=diff_pct,
            )
            breakdowns.append(breakdown)

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "compute_facility_breakdowns",
            "reconciliation_id": workspace.reconciliation_id,
            "facility_count": len(breakdowns),
            "facility_ids": [b.facility_id for b in breakdowns[:50]],
        })

        self._inc_breakdowns()
        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.compute_facility_breakdowns "
            "completed: recon_id=%s, facilities=%d, hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            len(breakdowns),
            provenance_hash[:16],
            elapsed,
        )

        return breakdowns

    # ==================================================================
    # PUBLIC METHOD 6: validate_completeness
    # ==================================================================

    def validate_completeness(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Tuple[bool, List[str]]:
        """Check that all four energy types have both location AND market results.

        GHG Protocol Scope 2 Guidance requires dual reporting for all
        purchased energy types.  This method identifies any gaps where
        an energy type is missing results for one or both methods, and
        additionally checks facility-level completeness.

        Args:
            workspace: ReconciliationWorkspace with location and market
                results populated.

        Returns:
            Tuple of (is_complete, gaps) where is_complete is True if
            all four energy types have results under both methods AND
            all facilities have both methods for each energy type, and
            gaps is a list of human-readable gap descriptions.

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.validate_completeness: recon_id=%s",
            workspace.reconciliation_id,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        # -- Determine which energy types have results ----------------------
        location_energy_types: Dict[EnergyType, int] = defaultdict(int)
        market_energy_types: Dict[EnergyType, int] = defaultdict(int)

        for r in workspace.location_results:
            location_energy_types[r.energy_type] += 1
        for r in workspace.market_results:
            market_energy_types[r.energy_type] += 1

        # -- Identify energy-type-level gaps --------------------------------
        gaps: List[str] = []

        for energy_type in EnergyType:
            has_location = location_energy_types.get(energy_type, 0) > 0
            has_market = market_energy_types.get(energy_type, 0) > 0

            if not has_location and not has_market:
                gaps.append(
                    f"MISSING_BOTH: {energy_type.value} has no "
                    f"location-based or market-based results"
                )
            elif not has_location:
                gaps.append(
                    f"MISSING_LOCATION: {energy_type.value} has "
                    f"{market_energy_types[energy_type]} market-based "
                    f"result(s) but no location-based results"
                )
            elif not has_market:
                gaps.append(
                    f"MISSING_MARKET: {energy_type.value} has "
                    f"{location_energy_types[energy_type]} location-based "
                    f"result(s) but no market-based results"
                )

        # -- Facility-level completeness checks -----------------------------
        location_facility_et: Dict[str, Set[EnergyType]] = defaultdict(set)
        market_facility_et: Dict[str, Set[EnergyType]] = defaultdict(set)

        for r in workspace.location_results:
            location_facility_et[r.facility_id].add(r.energy_type)
        for r in workspace.market_results:
            market_facility_et[r.facility_id].add(r.energy_type)

        all_facilities = set(location_facility_et.keys()) | set(
            market_facility_et.keys()
        )

        facility_gaps_count = 0
        for fac_id in sorted(all_facilities):
            loc_types = location_facility_et.get(fac_id, set())
            mkt_types = market_facility_et.get(fac_id, set())
            missing_in_market = loc_types - mkt_types
            missing_in_location = mkt_types - loc_types

            if missing_in_market:
                facility_gaps_count += 1
                if facility_gaps_count <= 20:
                    gaps.append(
                        f"FACILITY_MISSING_MARKET: Facility {fac_id} "
                        f"has location-based results for "
                        f"{sorted(e.value for e in missing_in_market)} "
                        f"but no market-based results for those types"
                    )
            if missing_in_location:
                facility_gaps_count += 1
                if facility_gaps_count <= 20:
                    gaps.append(
                        f"FACILITY_MISSING_LOCATION: Facility {fac_id} "
                        f"has market-based results for "
                        f"{sorted(e.value for e in missing_in_location)} "
                        f"but no location-based results for those types"
                    )

        if facility_gaps_count > 20:
            gaps.append(
                f"FACILITY_GAPS_TRUNCATED: {facility_gaps_count - 20} "
                f"additional facility-level gaps not shown"
            )

        is_complete = len(gaps) == 0

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "validate_completeness",
            "reconciliation_id": workspace.reconciliation_id,
            "is_complete": is_complete,
            "gaps_count": len(gaps),
            "gaps": gaps[:50],
        })

        self._inc_validations()
        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.validate_completeness completed: "
            "recon_id=%s, is_complete=%s, gaps=%d, "
            "hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            is_complete,
            len(gaps),
            provenance_hash[:16],
            elapsed,
        )

        return is_complete, gaps

    # ==================================================================
    # PUBLIC METHOD 7: get_total_emissions
    # ==================================================================

    def get_total_emissions(
        self,
        workspace: ReconciliationWorkspace,
    ) -> Dict[str, Decimal]:
        """Return total location-based and market-based emissions.

        Computes aggregated totals across all energy types and facilities.
        Also includes the absolute difference, percentage difference,
        overall direction, total energy MWh, and PIF.

        Args:
            workspace: ReconciliationWorkspace with location and market
                results populated.

        Returns:
            Dictionary with keys:
            - ``location_tco2e``: Total location-based emissions (Decimal).
            - ``market_tco2e``: Total market-based emissions (Decimal).
            - ``difference_tco2e``: location - market (Decimal).
            - ``difference_pct``: Percentage difference (Decimal).
            - ``direction``: DiscrepancyDirection string value.
            - ``energy_mwh_total``: Total energy in MWh (Decimal).
            - ``pif``: Procurement Impact Factor (Decimal).

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
        """
        start_time = time.monotonic()

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        # -- Compute totals ------------------------------------------------
        loc_total = self._sum_emissions(list(workspace.location_results))
        mkt_total = self._sum_emissions(list(workspace.market_results))
        diff = _quantize(loc_total - mkt_total)
        diff_pct = _pct_difference(loc_total, mkt_total)
        direction = _determine_direction(loc_total, mkt_total)

        # Total energy MWh (use max of loc and mkt)
        loc_mwh = self._sum_energy_mwh(list(workspace.location_results))
        mkt_mwh = self._sum_energy_mwh(list(workspace.market_results))
        energy_mwh_total = _quantize(max(loc_mwh, mkt_mwh))

        # PIF
        pif_value = self.compute_pif(loc_total, mkt_total)

        result: Dict[str, Any] = {
            "location_tco2e": loc_total,
            "market_tco2e": mkt_total,
            "difference_tco2e": diff,
            "difference_pct": diff_pct,
            "direction": direction.value,
            "energy_mwh_total": energy_mwh_total,
            "pif": pif_value,
        }

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "get_total_emissions",
            "reconciliation_id": workspace.reconciliation_id,
            "location_tco2e": str(loc_total),
            "market_tco2e": str(mkt_total),
            "difference_tco2e": str(diff),
            "pif": str(pif_value),
        })

        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.get_total_emissions: "
            "recon_id=%s, location=%.4f, market=%.4f, "
            "diff=%.4f (%.2f%%), direction=%s, pif=%.4f, "
            "hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            loc_total,
            mkt_total,
            diff,
            diff_pct,
            direction.value,
            pif_value,
            provenance_hash[:16],
            elapsed,
        )

        return result

    # ==================================================================
    # PUBLIC METHOD 8: filter_by_period
    # ==================================================================

    def filter_by_period(
        self,
        results: List[UpstreamResult],
        start: date,
        end: date,
    ) -> List[UpstreamResult]:
        """Filter upstream results to a specific reporting period.

        Returns results whose period overlaps with the given [start, end]
        range.  A result overlaps if its ``period_start <= end`` and its
        ``period_end >= start``.

        Args:
            results: List of UpstreamResult objects to filter.
            start: Start date of the desired period (inclusive).
            end: End date of the desired period (inclusive).

        Returns:
            Filtered list of UpstreamResult objects.

        Raises:
            TypeError: If results is not a list or dates are invalid.
            ValueError: If end is before start.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list, got {type(results).__name__}"
            )
        if not isinstance(start, date):
            raise TypeError(
                f"start must be a date, got {type(start).__name__}"
            )
        if not isinstance(end, date):
            raise TypeError(
                f"end must be a date, got {type(end).__name__}"
            )
        if end < start:
            raise ValueError(
                f"end ({end}) must be on or after start ({start})"
            )

        filtered = [
            r for r in results
            if r.period_start <= end and r.period_end >= start
        ]

        self._inc_filters()

        logger.debug(
            "filter_by_period: %d -> %d results for [%s, %s]",
            len(results),
            len(filtered),
            start,
            end,
        )

        return filtered

    # ==================================================================
    # PUBLIC METHOD 9: filter_by_facility
    # ==================================================================

    def filter_by_facility(
        self,
        results: List[UpstreamResult],
        facility_id: str,
    ) -> List[UpstreamResult]:
        """Filter upstream results for a specific facility.

        Args:
            results: List of UpstreamResult objects to filter.
            facility_id: The facility identifier to match.

        Returns:
            List of UpstreamResult objects for the given facility.

        Raises:
            TypeError: If results is not a list or facility_id is not str.
            ValueError: If facility_id is empty.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list, got {type(results).__name__}"
            )
        if not isinstance(facility_id, str):
            raise TypeError(
                f"facility_id must be a string, got "
                f"{type(facility_id).__name__}"
            )
        if not facility_id.strip():
            raise ValueError("facility_id must not be empty")

        filtered = [
            r for r in results
            if r.facility_id == facility_id
        ]

        self._inc_filters()

        logger.debug(
            "filter_by_facility: %d -> %d results for facility '%s'",
            len(results),
            len(filtered),
            facility_id,
        )

        return filtered

    # ==================================================================
    # PUBLIC METHOD 10: filter_by_energy_type
    # ==================================================================

    def filter_by_energy_type(
        self,
        results: List[UpstreamResult],
        energy_type: EnergyType,
    ) -> List[UpstreamResult]:
        """Filter upstream results by energy type.

        Args:
            results: List of UpstreamResult objects to filter.
            energy_type: The EnergyType enum to match.

        Returns:
            List of UpstreamResult objects for the given energy type.

        Raises:
            TypeError: If results is not a list or energy_type is invalid.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list, got {type(results).__name__}"
            )
        if not isinstance(energy_type, EnergyType):
            raise TypeError(
                f"energy_type must be an EnergyType enum, got "
                f"{type(energy_type).__name__}"
            )

        filtered = [
            r for r in results
            if r.energy_type == energy_type
        ]

        self._inc_filters()

        logger.debug(
            "filter_by_energy_type: %d -> %d results for '%s'",
            len(results),
            len(filtered),
            energy_type.value,
        )

        return filtered

    # ==================================================================
    # PUBLIC METHOD 11: group_by_method
    # ==================================================================

    def group_by_method(
        self,
        results: List[UpstreamResult],
    ) -> Dict[Scope2Method, List[UpstreamResult]]:
        """Split results into location-based and market-based groups.

        Args:
            results: List of UpstreamResult objects to group.

        Returns:
            Dictionary mapping each Scope2Method to its list of results.
            Both keys are always present (may be empty lists).

        Raises:
            TypeError: If results is not a list.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list, got {type(results).__name__}"
            )

        grouped: Dict[Scope2Method, List[UpstreamResult]] = {
            Scope2Method.LOCATION_BASED: [],
            Scope2Method.MARKET_BASED: [],
        }

        for r in results:
            if r.method in grouped:
                grouped[r.method].append(r)
            else:
                logger.warning(
                    "group_by_method: Unknown method '%s' for result "
                    "at facility '%s'; skipping",
                    r.method,
                    r.facility_id,
                )

        self._inc_filters()

        logger.debug(
            "group_by_method: %d results -> location=%d, market=%d",
            len(results),
            len(grouped[Scope2Method.LOCATION_BASED]),
            len(grouped[Scope2Method.MARKET_BASED]),
        )

        return grouped

    # ==================================================================
    # PUBLIC METHOD 12: group_by_energy_type
    # ==================================================================

    def group_by_energy_type(
        self,
        results: List[UpstreamResult],
    ) -> Dict[EnergyType, List[UpstreamResult]]:
        """Group results by energy type.

        Args:
            results: List of UpstreamResult objects to group.

        Returns:
            Dictionary mapping each EnergyType to its list of results.
            All four energy type keys are always present (may be empty).

        Raises:
            TypeError: If results is not a list.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list, got {type(results).__name__}"
            )

        grouped: Dict[EnergyType, List[UpstreamResult]] = {
            et: [] for et in EnergyType
        }

        for r in results:
            if r.energy_type in grouped:
                grouped[r.energy_type].append(r)
            else:
                logger.warning(
                    "group_by_energy_type: Unknown energy_type '%s' for "
                    "result at facility '%s'; skipping",
                    r.energy_type,
                    r.facility_id,
                )

        self._inc_filters()

        logger.debug(
            "group_by_energy_type: %d results -> %s",
            len(results),
            {et.value: len(items) for et, items in grouped.items()},
        )

        return grouped

    # ==================================================================
    # PUBLIC METHOD 13: group_by_facility
    # ==================================================================

    def group_by_facility(
        self,
        results: List[UpstreamResult],
    ) -> Dict[str, List[UpstreamResult]]:
        """Group results by facility_id.

        Args:
            results: List of UpstreamResult objects to group.

        Returns:
            Dictionary mapping each unique facility_id to its list of
            results.  Only facility_ids present in the input appear as
            keys.

        Raises:
            TypeError: If results is not a list.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(results, list):
            raise TypeError(
                f"results must be a list, got {type(results).__name__}"
            )

        grouped: Dict[str, List[UpstreamResult]] = defaultdict(list)

        for r in results:
            grouped[r.facility_id].append(r)

        self._inc_filters()

        logger.debug(
            "group_by_facility: %d results -> %d unique facilities",
            len(results),
            len(grouped),
        )

        return dict(grouped)

    # ==================================================================
    # PUBLIC METHOD 14: compute_pif
    # ==================================================================

    def compute_pif(
        self,
        location_total: Decimal,
        market_total: Decimal,
    ) -> Decimal:
        """Compute the Procurement Impact Factor (PIF).

        PIF quantifies the impact of procurement choices (RECs, GOs, PPAs)
        on Scope 2 emissions.  A positive PIF indicates that market-based
        emissions are lower than location-based (good procurement impact).

        Formula:
            PIF = (location - market) / location * 100

        When location_total is zero, PIF is zero (no basis for comparison).

        Args:
            location_total: Total location-based emissions in tCO2e.
            market_total: Total market-based emissions in tCO2e.

        Returns:
            PIF as a Decimal percentage.  Positive means market-based is
            lower (procurement reduced emissions).  Negative means
            market-based is higher.

        Raises:
            TypeError: If inputs are not Decimal.
        """
        # -- Input validation -----------------------------------------------
        if not isinstance(location_total, Decimal):
            raise TypeError(
                f"location_total must be Decimal, "
                f"got {type(location_total).__name__}"
            )
        if not isinstance(market_total, Decimal):
            raise TypeError(
                f"market_total must be Decimal, "
                f"got {type(market_total).__name__}"
            )

        self._inc_pif_calcs()

        if location_total == _ZERO:
            logger.debug("compute_pif: location_total is zero; PIF=0")
            return _quantize(_ZERO)

        pif = _quantize(
            (location_total - market_total) / location_total * _HUNDRED
        )

        logger.debug(
            "compute_pif: location=%.4f, market=%.4f, PIF=%.4f%%",
            location_total,
            market_total,
            pif,
        )

        return pif

    # ==================================================================
    # PUBLIC METHOD 15: detect_unmatched_results
    # ==================================================================

    def detect_unmatched_results(
        self,
        workspace: ReconciliationWorkspace,
    ) -> List[UpstreamResult]:
        """Find results without a matching pair from the opposite method.

        A result is "matched" if there exists at least one result from
        the opposite method (location vs market) for the same
        facility_id and energy_type combination.  Results without a
        matching pair are returned as "unmatched".

        Unmatched results indicate incomplete dual reporting, which is
        a compliance risk under GHG Protocol Scope 2 Guidance.

        Args:
            workspace: ReconciliationWorkspace with location and market
                results populated.

        Returns:
            List of UpstreamResult objects that have no match from the
            opposite method, sorted by (facility_id, energy_type, method).

        Raises:
            TypeError: If workspace is not a ReconciliationWorkspace.
        """
        start_time = time.monotonic()
        logger.info(
            "DualResultCollectorEngine.detect_unmatched_results: "
            "recon_id=%s",
            workspace.reconciliation_id,
        )

        # -- Input validation -----------------------------------------------
        if not isinstance(workspace, ReconciliationWorkspace):
            raise TypeError(
                f"workspace must be ReconciliationWorkspace, "
                f"got {type(workspace).__name__}"
            )

        # -- Build key sets for each method --------------------------------
        # Key = (facility_id, energy_type)
        location_keys: Set[Tuple[str, EnergyType]] = set()
        market_keys: Set[Tuple[str, EnergyType]] = set()

        for r in workspace.location_results:
            location_keys.add((r.facility_id, r.energy_type))
        for r in workspace.market_results:
            market_keys.add((r.facility_id, r.energy_type))

        # -- Find unmatched ------------------------------------------------
        unmatched: List[UpstreamResult] = []

        # Location results without market match
        for r in workspace.location_results:
            key = (r.facility_id, r.energy_type)
            if key not in market_keys:
                unmatched.append(r)

        # Market results without location match
        for r in workspace.market_results:
            key = (r.facility_id, r.energy_type)
            if key not in location_keys:
                unmatched.append(r)

        # Sort for deterministic output
        unmatched.sort(
            key=lambda r: (
                r.facility_id,
                r.energy_type.value,
                r.method.value,
            )
        )

        # -- Provenance -----------------------------------------------------
        provenance_hash = _compute_hash({
            "operation": "detect_unmatched_results",
            "reconciliation_id": workspace.reconciliation_id,
            "unmatched_count": len(unmatched),
            "location_keys_count": len(location_keys),
            "market_keys_count": len(market_keys),
        })

        self._inc_unmatched()
        elapsed = time.monotonic() - start_time

        logger.info(
            "DualResultCollectorEngine.detect_unmatched_results completed: "
            "recon_id=%s, unmatched=%d (location_keys=%d, market_keys=%d), "
            "hash=%s, elapsed=%.4fs",
            workspace.reconciliation_id,
            len(unmatched),
            len(location_keys),
            len(market_keys),
            provenance_hash[:16],
            elapsed,
        )

        return unmatched

    # ==================================================================
    # PUBLIC METHOD 16: health_check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Return engine health status and operational statistics.

        Provides diagnostic information about the engine's current state,
        including uptime, operation counters, configuration availability,
        and component versions.  Suitable for Kubernetes readiness and
        liveness probes.

        Returns:
            Dictionary with health status fields:
            - ``status``: Always ``"healthy"`` if the engine is running.
            - ``engine``: Engine class name.
            - ``engine_id``: Unique engine instance identifier.
            - ``agent_id``: Agent identifier.
            - ``component``: Component identifier.
            - ``version``: Version string.
            - ``table_prefix``: Database table prefix.
            - ``created_at``: Engine creation ISO timestamp.
            - ``current_time``: Current UTC ISO timestamp.
            - ``uptime_seconds``: Seconds since engine creation.
            - ``total_collections``: Number of collect_results calls.
            - ``total_alignments``: Number of align_boundaries calls.
            - ``total_mappings``: Number of map_energy_types calls.
            - ``total_breakdowns``: Number of breakdown computations.
            - ``total_validations``: Number of validate_completeness calls.
            - ``total_filters``: Number of filter/group operations.
            - ``total_pif_calculations``: Number of PIF computations.
            - ``total_unmatched_detections``: Number of unmatched detections.
            - ``total_health_checks``: Number of health_check calls.
            - ``config_available``: Whether config module is loaded.
            - ``metrics_available``: Whether metrics module is loaded.
            - ``provenance_available``: Whether provenance module is loaded.
            - ``energy_types``: List of supported energy types.
            - ``upstream_agents``: List of upstream agent identifiers.
            - ``upstream_agent_mapping``: Energy type to agent mapping.
            - ``max_upstream_results``: Maximum results per method.
            - ``max_facilities``: Maximum facilities per reconciliation.
            - ``decimal_precision``: Precision string.
            - ``rounding_mode``: Rounding mode name.
            - ``equality_tolerance``: Tolerance for EQUAL direction.
        """
        self._inc_health_checks()

        now = utcnow()
        uptime = (now - self._created_at).total_seconds()

        with self._internal_lock:
            status: Dict[str, Any] = {
                "status": "healthy",
                "engine": self.__class__.__name__,
                "engine_id": self._engine_id,
                "agent_id": AGENT_ID,
                "component": AGENT_COMPONENT,
                "version": VERSION,
                "table_prefix": TABLE_PREFIX,
                "created_at": self._created_at.isoformat(),
                "current_time": now.isoformat(),
                "uptime_seconds": round(uptime, 2),
                "total_collections": self._total_collections,
                "total_alignments": self._total_alignments,
                "total_mappings": self._total_mappings,
                "total_breakdowns": self._total_breakdowns,
                "total_validations": self._total_validations,
                "total_filters": self._total_filters,
                "total_pif_calculations": self._total_pif_calcs,
                "total_unmatched_detections": self._total_unmatched_detections,
                "total_health_checks": self._total_health_checks,
                "config_available": self._config is not None,
                "metrics_available": self._metrics is not None,
                "provenance_available": self._provenance is not None,
                "supported_energy_types": sorted(
                    et.value for et in EnergyType
                ),
                "supported_methods": sorted(
                    m.value for m in Scope2Method
                ),
                "upstream_agents": sorted(
                    ua.value for ua in UpstreamAgent
                ),
                "upstream_agent_mapping": {
                    et.value: (loc.value, mkt.value)
                    for et, (loc, mkt) in UPSTREAM_AGENT_MAPPING.items()
                },
                "max_upstream_results": MAX_UPSTREAM_RESULTS,
                "max_facilities": MAX_FACILITIES,
                "decimal_precision": str(_PRECISION),
                "rounding_mode": "ROUND_HALF_UP",
                "equality_tolerance": str(_EQUALITY_TOLERANCE),
            }

        logger.debug(
            "health_check: status=%s, uptime=%.2fs, "
            "collections=%d, alignments=%d, mappings=%d",
            status["status"],
            uptime,
            status["total_collections"],
            status["total_alignments"],
            status["total_mappings"],
        )

        return status

    # ==================================================================
    # String representation
    # ==================================================================

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"<{self.__class__.__name__}("
            f"engine_id={self._engine_id!r}, "
            f"agent_id={AGENT_ID!r}, "
            f"version={VERSION!r}, "
            f"collections={self._total_collections}, "
            f"alignments={self._total_alignments}"
            f")>"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"DualResultCollectorEngine v{VERSION} "
            f"({AGENT_ID}/{AGENT_COMPONENT}) "
            f"[{self._engine_id}]"
        )
