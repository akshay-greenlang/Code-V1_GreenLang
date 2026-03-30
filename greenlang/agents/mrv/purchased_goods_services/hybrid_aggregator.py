# -*- coding: utf-8 -*-
"""
HybridAggregatorEngine - Multi-Method Aggregation & Hot-Spot Analysis (Engine 5 of 7)

AGENT-MRV-014: Purchased Goods & Services Agent (GL-MRV-S3-001)

This engine combines results from the three Category 1 calculation methods
(spend-based EEIO, average-data physical EFs, supplier-specific EPD/PCF)
into a unified hybrid inventory. It implements the GHG Protocol's guidance
on combining methods by using the highest-quality data available for each
procurement item while preventing double-counting.

Core Capabilities:
    1. Method Prioritization -- For each procurement item, select the
       best available method: supplier > average_data > spend_based.
    2. Coverage Analysis -- Track percentage of total spend covered
       by each method and classify coverage level (FULL/HIGH/MEDIUM/
       LOW/MINIMAL) per COVERAGE_THRESHOLDS.
    3. Hot-Spot Analysis -- Pareto 80/20 ranking of procurement
       categories by emission contribution and materiality quadrant
       classification (Q1-Q4) using spend and EF intensity thresholds.
    4. Double-Counting Prevention -- Category boundary enforcement
       against Categories 2-8, overlap detection across methods,
       and intercompany/credit filtering.
    5. Gap Filling -- Fallback to lower-tier methods for uncovered
       items to maximize spend coverage.
    6. Total Aggregation -- Combine all method results into a single
       HybridResult with emission-weighted DQI score.
    7. YoY Decomposition -- Decompose year-over-year emission changes
       into activity, emission factor, method mix, and scope effects.
    8. Intensity Metrics -- Revenue, FTE, and spend intensity ratios
       (tCO2e per $M).

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal arithmetic.
    - No LLM calls in any calculation or aggregation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every aggregation result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Thread-safe singleton with ``threading.RLock()``. Mutable counters
    and aggregation state are protected by the reentrant lock. Each
    aggregation call is stateless with respect to previous calls.

Example:
    >>> from greenlang.agents.mrv.purchased_goods_services.hybrid_aggregator import (
    ...     HybridAggregatorEngine,
    ... )
    >>> engine = HybridAggregatorEngine()
    >>> result = engine.aggregate(
    ...     spend_results=spend_results,
    ...     avgdata_results=avgdata_results,
    ...     supplier_results=supplier_results,
    ...     items=items,
    ...     total_spend_usd=Decimal("10000000"),
    ... )
    >>> print(result.total_emissions_tco2e)
    >>> print(result.coverage_level)

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
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["HybridAggregatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- models
# ---------------------------------------------------------------------------

try:
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
        CoverageLevel,
        ProcurementType,
        DQIDimension,
        DQIScore,
        EmissionGas,
        BatchStatus,
        DQI_SCORE_VALUES,
        COVERAGE_THRESHOLDS,
        UNCERTAINTY_RANGES,
        ProcurementItem,
        SpendBasedResult,
        AverageDataResult,
        SupplierSpecificResult,
        HybridResult,
        DQIAssessment,
        MaterialityItem,
        CoverageReport,
        HotSpotAnalysis,
        CategoryBoundaryCheck,
        AggregationResult,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.agents.mrv.purchased_goods_services.models not available; "
        "using fallback stubs"
    )
    _MODELS_AVAILABLE = False

    # Fallback constants
    AGENT_ID = "GL-MRV-S3-001"
    VERSION = "1.0.0"
    TABLE_PREFIX = "gl_pgs_"
    ZERO = Decimal("0")
    ONE = Decimal("1")
    ONE_HUNDRED = Decimal("100")
    ONE_THOUSAND = Decimal("1000")
    DECIMAL_PLACES = 8

# ---------------------------------------------------------------------------
# Conditional imports -- config
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.purchased_goods_services.config import (
        PurchasedGoodsServicesConfig,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.agents.mrv.purchased_goods_services.config not available; "
        "using defaults"
    )
    _CONFIG_AVAILABLE = False
    PurchasedGoodsServicesConfig = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Conditional imports -- metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.purchased_goods_services.metrics import (
        PurchasedGoodsServicesMetrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.agents.mrv.purchased_goods_services.metrics not available; "
        "metrics will be no-ops"
    )
    _METRICS_AVAILABLE = False
    PurchasedGoodsServicesMetrics = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Conditional imports -- provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.purchased_goods_services.provenance import (
        PurchasedGoodsProvenanceTracker,
        ProvenanceStage,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.agents.mrv.purchased_goods_services.provenance not available; "
        "provenance tracking disabled"
    )
    _PROVENANCE_AVAILABLE = False
    PurchasedGoodsProvenanceTracker = None  # type: ignore[assignment,misc]
    ProvenanceStage = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Engine-local constants
# ---------------------------------------------------------------------------

#: Quantizer for Decimal arithmetic.
_QUANTIZER = Decimal(10) ** -DECIMAL_PLACES

#: Spend percentage threshold for "high spend" in materiality quadrant.
_HIGH_SPEND_PCT_THRESHOLD = Decimal("5.0")

#: EF intensity threshold for "high EF" in materiality quadrant (kgCO2e/$).
_HIGH_EF_INTENSITY_THRESHOLD = Decimal("0.5")

#: Pareto cumulative percentage for 80/20 rule.
_PARETO_THRESHOLD_PCT = Decimal("80.0")

#: Default top-N items for hot-spot analysis.
_DEFAULT_TOP_N = 20

#: Million constant for intensity denominators.
_MILLION = Decimal("1000000")

#: Minimum DQI score.
_MIN_DQI = Decimal("1.0")

#: Maximum DQI score.
_MAX_DQI = Decimal("5.0")

#: Default coverage thresholds when models unavailable.
_FALLBACK_COVERAGE_THRESHOLDS: Dict[str, Decimal] = {
    "full": Decimal("100.0"),
    "high": Decimal("95.0"),
    "medium": Decimal("90.0"),
    "low": Decimal("80.0"),
    "minimal": Decimal("0.0"),
}

#: Default uncertainty ranges by method when models unavailable.
_FALLBACK_UNCERTAINTY_RANGES: Dict[str, Tuple[Decimal, Decimal]] = {
    "supplier_specific": (Decimal("10"), Decimal("30")),
    "hybrid": (Decimal("20"), Decimal("50")),
    "average_data": (Decimal("30"), Decimal("60")),
    "spend_based": (Decimal("50"), Decimal("100")),
}

#: Category boundary exclusion flags mapping to Scope 3 categories.
_BOUNDARY_FLAG_MAP: Dict[str, str] = {
    "is_capital_good": "Category 2 - Capital Goods",
    "is_fuel_energy": "Category 3 - Fuel and Energy Related Activities",
    "is_transport": "Category 4 - Upstream Transportation and Distribution",
    "is_business_travel": "Category 6 - Business Travel",
    "is_intercompany": "Intercompany elimination",
    "is_credit_return": "Credit memo / return",
}

# ===========================================================================
# Helper functions
# ===========================================================================

def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal value to the configured decimal places.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal with DECIMAL_PLACES precision.
    """
    try:
        return value.quantize(_QUANTIZER, rounding=ROUND_HALF_UP)
    except (InvalidOperation, OverflowError):
        return ZERO

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values, returning default on zero division.

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

    Serialises the data to a canonical JSON string, encodes as UTF-8,
    and returns the hexadecimal SHA-256 digest.

    Args:
        data: Any JSON-serialisable object.

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

def _decimal_to_str(value: Decimal) -> str:
    """Convert Decimal to string for JSON serialisation.

    Args:
        value: Decimal value.

    Returns:
        String representation of the quantized Decimal.
    """
    return str(_quantize(value))

def _pct(part: Decimal, whole: Decimal) -> Decimal:
    """Calculate percentage of part relative to whole.

    Args:
        part: Numerator.
        whole: Denominator (total).

    Returns:
        Percentage as Decimal in range [0, 100], quantized.
    """
    if whole == ZERO:
        return ZERO
    raw = (part / whole) * ONE_HUNDRED
    result = _quantize(raw)
    # Clamp to [0, 100]
    if result < ZERO:
        return ZERO
    if result > ONE_HUNDRED:
        return ONE_HUNDRED
    return result

# ===========================================================================
# HybridAggregatorEngine
# ===========================================================================

class HybridAggregatorEngine:
    """Engine 5 of 7 -- Multi-method aggregation and hot-spot analysis.

    Combines results from spend-based, average-data, and supplier-specific
    calculation engines into a unified Category 1 inventory using method
    prioritisation, coverage analysis, hot-spot identification, and
    double-counting prevention.

    This engine follows the GHG Protocol's recommended hybrid approach:
    for each procurement item, the highest-quality available method is
    selected (supplier > average_data > spend_based). Items excluded by
    category boundary checks (Categories 2-8, intercompany, credits) are
    removed before aggregation. Gap filling applies lower-tier methods to
    uncovered items.

    Thread Safety:
        Thread-safe singleton via ``__new__`` with ``threading.RLock()``.
        All mutable state is protected by the lock. Aggregation calls are
        stateless with respect to previous calls.

    Attributes:
        _config: Agent configuration singleton.
        _metrics: Prometheus metrics collector.
        _provenance: SHA-256 provenance chain tracker.
        _aggregation_count: Total number of aggregations performed.
        _total_items_aggregated: Total items processed across all calls.
        _total_emissions_aggregated_tco2e: Cumulative emissions aggregated.

    Example:
        >>> engine = HybridAggregatorEngine()
        >>> result = engine.aggregate(
        ...     spend_results=[...],
        ...     avgdata_results=[...],
        ...     supplier_results=[...],
        ...     items=[...],
        ...     total_spend_usd=Decimal("10000000"),
        ... )
        >>> assert result.total_coverage_pct > Decimal("80")
    """

    _instance: Optional[HybridAggregatorEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> HybridAggregatorEngine:
        """Thread-safe singleton instantiation.

        Returns:
            The single HybridAggregatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialise the engine (runs only once due to singleton guard)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Configuration
            self._config: Any = None
            if _CONFIG_AVAILABLE:
                try:
                    self._config = PurchasedGoodsServicesConfig()
                except Exception as exc:
                    logger.warning(
                        "Failed to load PurchasedGoodsServicesConfig: %s", exc
                    )

            # Metrics
            self._metrics: Any = None
            if _METRICS_AVAILABLE:
                try:
                    self._metrics = PurchasedGoodsServicesMetrics()
                except Exception as exc:
                    logger.warning(
                        "Failed to load PurchasedGoodsServicesMetrics: %s", exc
                    )

            # Provenance
            self._provenance: Any = None
            if _PROVENANCE_AVAILABLE:
                try:
                    self._provenance = (
                        PurchasedGoodsProvenanceTracker.get_instance()
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load PurchasedGoodsProvenanceTracker: %s",
                        exc,
                    )

            # Internal counters (protected by _lock)
            self._aggregation_count: int = 0
            self._total_items_aggregated: int = 0
            self._total_emissions_aggregated_tco2e: Decimal = ZERO

            self._initialized = True

            logger.info(
                "HybridAggregatorEngine initialised "
                "(config=%s, metrics=%s, provenance=%s)",
                self._config is not None,
                self._metrics is not None,
                self._provenance is not None,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        Clears the singleton so the next instantiation creates a fresh
        engine. This method is intended **only** for unit tests.
        """
        with cls._lock:
            cls._instance = None
        logger.info("HybridAggregatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Public API -- aggregate
    # ------------------------------------------------------------------

    def aggregate(
        self,
        spend_results: List[Any],
        avgdata_results: List[Any],
        supplier_results: List[Any],
        items: List[Any],
        total_spend_usd: Decimal,
        calculation_id: Optional[str] = None,
    ) -> Any:
        """Aggregate results from all three calculation methods.

        This is the primary entry point for the hybrid aggregation engine.
        It performs the following steps in order:

        1. Validate inputs and generate calculation_id.
        2. Build item lookup map (item_id -> ProcurementItem).
        3. Check category boundaries to identify excluded items.
        4. Filter excluded items from the result sets.
        5. Detect cross-method overlaps and resolve by priority.
        6. Build item-to-method map (best method per item).
        7. Fill gaps for uncovered items using fallback spend-based.
        8. Compute method-level emissions and counts.
        9. Compute total emissions and coverage percentages.
        10. Compute emission-weighted DQI.
        11. Generate provenance hash.
        12. Assemble and return HybridResult.

        Args:
            spend_results: List of SpendBasedResult objects.
            avgdata_results: List of AverageDataResult objects.
            supplier_results: List of SupplierSpecificResult objects.
            items: List of ProcurementItem objects.
            total_spend_usd: Total procurement spend in USD.
            calculation_id: Optional calculation identifier (auto-generated
                if not provided).

        Returns:
            HybridResult with aggregated emissions, coverage, and DQI.

        Raises:
            ValueError: If total_spend_usd is negative.
            TypeError: If result lists contain invalid types.
        """
        start_time = time.monotonic()
        calc_id = calculation_id or str(uuid4())

        logger.info(
            "HybridAggregatorEngine.aggregate started "
            "(calc_id=%s, items=%d, spend=%d, avg=%d, supplier=%d, "
            "total_spend_usd=%s)",
            calc_id,
            len(items),
            len(spend_results),
            len(avgdata_results),
            len(supplier_results),
            str(total_spend_usd),
        )

        # -- Step 1: Validate inputs
        self._validate_aggregate_inputs(
            spend_results, avgdata_results, supplier_results,
            items, total_spend_usd,
        )

        # -- Step 2: Build item lookup
        item_map = self._build_item_map(items)

        # -- Step 3: Check category boundaries
        boundary_checks = self.check_category_boundaries(items)
        excluded_ids = self._collect_excluded_ids(boundary_checks)

        # -- Step 4: Filter excluded items
        eligible_items, excluded_items = self.filter_excluded_items(
            items, excluded_ids,
        )

        # -- Step 5: Detect overlaps
        overlap_ids = self.detect_overlap(
            spend_results, avgdata_results, supplier_results,
        )
        if overlap_ids:
            logger.info(
                "Detected %d cross-method overlaps; resolving by priority",
                len(overlap_ids),
            )

        # -- Step 6: Build item-method map with deduplication
        spend_results_clean = self._filter_results_by_exclusion(
            spend_results, excluded_ids,
        )
        avgdata_results_clean = self._filter_results_by_exclusion(
            avgdata_results, excluded_ids,
        )
        supplier_results_clean = self._filter_results_by_exclusion(
            supplier_results, excluded_ids,
        )

        item_method_map = self.build_item_method_map(
            eligible_items,
            supplier_results_clean,
            avgdata_results_clean,
            spend_results_clean,
        )

        # -- Step 7: Gap filling
        covered_ids = set(item_method_map.keys())
        gap_results = self.fill_gaps(
            eligible_items, covered_ids, spend_results_clean,
        )
        for gr in gap_results:
            gid = self._get_result_item_id(gr)
            if gid and gid not in item_method_map:
                item_method_map[gid] = self._get_calc_method("spend_based")

        # Merge gap results into spend_results_clean
        all_spend_results = list(spend_results_clean) + list(gap_results)

        # -- Step 8: Compute per-method emissions and counts
        supplier_ids = self._ids_by_method(
            item_method_map, "supplier_specific",
        )
        avgdata_ids = self._ids_by_method(
            item_method_map, "average_data",
        )
        spend_ids = self._ids_by_method(
            item_method_map, "spend_based",
        )

        supplier_emissions_tco2e = self._sum_emissions_tco2e_for_ids(
            supplier_results_clean, supplier_ids,
        )
        avgdata_emissions_tco2e = self._sum_emissions_tco2e_for_ids(
            avgdata_results_clean, avgdata_ids,
        )
        spend_emissions_tco2e = self._sum_emissions_tco2e_for_ids(
            all_spend_results, spend_ids,
        )

        # -- Step 9: Total emissions and coverage
        total_emissions_tco2e = _quantize(
            supplier_emissions_tco2e
            + avgdata_emissions_tco2e
            + spend_emissions_tco2e
        )
        total_emissions_kgco2e = _quantize(
            total_emissions_tco2e * ONE_THOUSAND
        )

        # Coverage by spend
        supplier_spend = self._sum_spend_for_ids(
            eligible_items, item_map, supplier_ids,
        )
        avgdata_spend = self._sum_spend_for_ids(
            eligible_items, item_map, avgdata_ids,
        )
        spend_spend = self._sum_spend_for_ids(
            eligible_items, item_map, spend_ids,
        )

        effective_total_spend = (
            total_spend_usd if total_spend_usd > ZERO
            else _quantize(supplier_spend + avgdata_spend + spend_spend)
        )

        supplier_coverage_pct = _pct(supplier_spend, effective_total_spend)
        avgdata_coverage_pct = _pct(avgdata_spend, effective_total_spend)
        spend_coverage_pct = _pct(spend_spend, effective_total_spend)
        total_coverage_pct = _pct(
            supplier_spend + avgdata_spend + spend_spend,
            effective_total_spend,
        )

        coverage_level = self.classify_coverage_level(total_coverage_pct)

        # -- Step 10: Weighted DQI
        all_results_with_dqi = (
            list(supplier_results_clean)
            + list(avgdata_results_clean)
            + list(all_spend_results)
        )
        weighted_dqi = self.compute_weighted_dqi(all_results_with_dqi)

        # -- Step 11: Provenance hash
        provenance_input = {
            "calculation_id": calc_id,
            "total_emissions_tco2e": _decimal_to_str(total_emissions_tco2e),
            "supplier_emissions_tco2e": _decimal_to_str(
                supplier_emissions_tco2e
            ),
            "avgdata_emissions_tco2e": _decimal_to_str(
                avgdata_emissions_tco2e
            ),
            "spend_emissions_tco2e": _decimal_to_str(spend_emissions_tco2e),
            "total_coverage_pct": _decimal_to_str(total_coverage_pct),
            "coverage_level": (
                coverage_level.value
                if hasattr(coverage_level, "value")
                else str(coverage_level)
            ),
            "weighted_dqi": _decimal_to_str(weighted_dqi),
            "item_count": len(eligible_items),
            "excluded_count": len(excluded_items),
        }
        provenance_hash = _compute_sha256(provenance_input)

        # Record provenance stage
        self._record_provenance_stage(
            calc_id, provenance_input, provenance_hash,
        )

        # -- Step 12: Processing time
        elapsed_ms = Decimal(str(
            (time.monotonic() - start_time) * 1000
        ))
        processing_time_ms = _quantize(elapsed_ms)

        # -- Step 13: Assemble HybridResult
        hybrid_result = self._build_hybrid_result(
            calculation_id=calc_id,
            total_emissions_kgco2e=total_emissions_kgco2e,
            total_emissions_tco2e=total_emissions_tco2e,
            spend_based_emissions_tco2e=spend_emissions_tco2e,
            average_data_emissions_tco2e=avgdata_emissions_tco2e,
            supplier_specific_emissions_tco2e=supplier_emissions_tco2e,
            spend_based_coverage_pct=spend_coverage_pct,
            average_data_coverage_pct=avgdata_coverage_pct,
            supplier_specific_coverage_pct=supplier_coverage_pct,
            total_coverage_pct=total_coverage_pct,
            coverage_level=coverage_level,
            total_spend_usd=effective_total_spend,
            item_count=len(eligible_items),
            spend_based_count=len(spend_ids),
            average_data_count=len(avgdata_ids),
            supplier_specific_count=len(supplier_ids),
            excluded_count=len(excluded_items),
            weighted_dqi=weighted_dqi,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

        # Update internal counters
        with self._lock:
            self._aggregation_count += 1
            self._total_items_aggregated += len(eligible_items)
            self._total_emissions_aggregated_tco2e += total_emissions_tco2e

        # Record metrics
        self._record_aggregation_metrics(
            calc_id=calc_id,
            total_emissions_tco2e=total_emissions_tco2e,
            total_spend_usd=effective_total_spend,
            item_count=len(eligible_items),
            processing_time_s=float(processing_time_ms) / 1000.0,
            coverage_level=coverage_level,
        )

        logger.info(
            "HybridAggregatorEngine.aggregate completed "
            "(calc_id=%s, total_tco2e=%s, coverage=%s%%, level=%s, "
            "items=%d, excluded=%d, elapsed_ms=%s)",
            calc_id,
            _decimal_to_str(total_emissions_tco2e),
            _decimal_to_str(total_coverage_pct),
            coverage_level.value if hasattr(coverage_level, "value")
            else str(coverage_level),
            len(eligible_items),
            len(excluded_items),
            _decimal_to_str(processing_time_ms),
        )

        return hybrid_result

    # ------------------------------------------------------------------
    # Public API -- select_method_for_item
    # ------------------------------------------------------------------

    def select_method_for_item(
        self,
        item_id: str,
        has_supplier: bool,
        has_avgdata: bool,
        has_spend: bool,
    ) -> Any:
        """Select the best calculation method for a procurement item.

        Applies the GHG Protocol method hierarchy:
        supplier_specific > average_data > spend_based.

        If no method is available, returns spend_based as fallback
        (the item may need gap filling or manual data collection).

        Args:
            item_id: Procurement item identifier.
            has_supplier: Whether supplier-specific result exists.
            has_avgdata: Whether average-data result exists.
            has_spend: Whether spend-based result exists.

        Returns:
            CalculationMethod enum value representing the best method.
        """
        if has_supplier:
            method = self._get_calc_method("supplier_specific")
            logger.debug(
                "Item %s: selected supplier_specific", item_id,
            )
            return method

        if has_avgdata:
            method = self._get_calc_method("average_data")
            logger.debug(
                "Item %s: selected average_data", item_id,
            )
            return method

        if has_spend:
            method = self._get_calc_method("spend_based")
            logger.debug(
                "Item %s: selected spend_based", item_id,
            )
            return method

        # No data available -- default to spend_based for gap filling
        logger.warning(
            "Item %s: no results available; defaulting to spend_based "
            "for gap filling",
            item_id,
        )
        return self._get_calc_method("spend_based")

    # ------------------------------------------------------------------
    # Public API -- build_item_method_map
    # ------------------------------------------------------------------

    def build_item_method_map(
        self,
        items: List[Any],
        supplier_results: List[Any],
        avgdata_results: List[Any],
        spend_results: List[Any],
    ) -> Dict[str, Any]:
        """Build a mapping of item_id to best calculation method.

        For each procurement item, checks which method results are
        available and selects the highest-priority method. This map
        drives the aggregation: each item's emissions are taken from
        exactly one method to prevent double-counting.

        Args:
            items: List of ProcurementItem objects.
            supplier_results: List of SupplierSpecificResult objects.
            avgdata_results: List of AverageDataResult objects.
            spend_results: List of SpendBasedResult objects.

        Returns:
            Dict mapping item_id to CalculationMethod.
        """
        supplier_ids = self._result_ids_set(supplier_results)
        avgdata_ids = self._result_ids_set(avgdata_results)
        spend_ids = self._result_ids_set(spend_results)

        method_map: Dict[str, Any] = {}
        for item in items:
            item_id = self._get_item_id(item)
            if not item_id:
                continue

            has_supplier = item_id in supplier_ids
            has_avgdata = item_id in avgdata_ids
            has_spend = item_id in spend_ids

            if has_supplier or has_avgdata or has_spend:
                method_map[item_id] = self.select_method_for_item(
                    item_id, has_supplier, has_avgdata, has_spend,
                )

        logger.info(
            "build_item_method_map: %d items mapped "
            "(supplier=%d, avgdata=%d, spend=%d)",
            len(method_map),
            sum(
                1 for m in method_map.values()
                if self._method_name(m) == "supplier_specific"
            ),
            sum(
                1 for m in method_map.values()
                if self._method_name(m) == "average_data"
            ),
            sum(
                1 for m in method_map.values()
                if self._method_name(m) == "spend_based"
            ),
        )
        return method_map

    # ------------------------------------------------------------------
    # Public API -- compute_coverage
    # ------------------------------------------------------------------

    def compute_coverage(
        self,
        spend_results: List[Any],
        avgdata_results: List[Any],
        supplier_results: List[Any],
        total_spend_usd: Decimal,
        items: Optional[List[Any]] = None,
    ) -> Any:
        """Compute method coverage analysis for the Category 1 inventory.

        Calculates the percentage of total procurement spend covered by
        each calculation method and classifies the overall coverage level.

        Args:
            spend_results: List of SpendBasedResult objects.
            avgdata_results: List of AverageDataResult objects.
            supplier_results: List of SupplierSpecificResult objects.
            total_spend_usd: Total procurement spend in USD.
            items: Optional list of ProcurementItem objects for
                per-category breakdown.

        Returns:
            CoverageReport with method-level and total coverage.
        """
        logger.info(
            "compute_coverage: spend=%d, avg=%d, supplier=%d, "
            "total_spend_usd=%s",
            len(spend_results),
            len(avgdata_results),
            len(supplier_results),
            str(total_spend_usd),
        )

        # Sum spend by method
        supplier_spend = self._sum_result_spend_usd(supplier_results)
        avgdata_spend = self._sum_result_spend_usd(avgdata_results)
        spend_spend = self._sum_result_spend_usd(spend_results)

        covered_total = _quantize(
            supplier_spend + avgdata_spend + spend_spend
        )
        uncovered = _quantize(
            max(total_spend_usd - covered_total, ZERO)
        )

        effective_total = (
            total_spend_usd if total_spend_usd > ZERO
            else covered_total
        )

        supplier_pct = _pct(supplier_spend, effective_total)
        avgdata_pct = _pct(avgdata_spend, effective_total)
        spend_pct = _pct(spend_spend, effective_total)
        total_pct = _pct(covered_total, effective_total)
        coverage_level = self.classify_coverage_level(total_pct)

        # Per-category coverage breakdown
        coverage_by_category: Dict[str, Decimal] = {}
        gap_categories: List[str] = []
        if items:
            category_spend = self._group_spend_by_category(items)
            covered_ids = (
                self._result_ids_set(supplier_results)
                | self._result_ids_set(avgdata_results)
                | self._result_ids_set(spend_results)
            )
            for cat, cat_spend in category_spend.items():
                cat_items_covered = ZERO
                for item in items:
                    iid = self._get_item_id(item)
                    icat = self._get_item_category(item)
                    if icat == cat and iid in covered_ids:
                        cat_items_covered += self._get_item_spend(item)
                cat_pct = _pct(cat_items_covered, cat_spend)
                coverage_by_category[cat] = cat_pct
                if cat_pct == ZERO:
                    gap_categories.append(cat)

        report = self._build_coverage_report(
            total_spend_usd=effective_total,
            supplier_specific_spend_usd=supplier_spend,
            average_data_spend_usd=avgdata_spend,
            spend_based_spend_usd=spend_spend,
            uncovered_spend_usd=uncovered,
            supplier_specific_pct=supplier_pct,
            average_data_pct=avgdata_pct,
            spend_based_pct=spend_pct,
            total_coverage_pct=total_pct,
            coverage_level=coverage_level,
            gap_categories=gap_categories,
            coverage_by_category=coverage_by_category,
        )

        logger.info(
            "compute_coverage completed: total=%s%%, level=%s, gaps=%d",
            _decimal_to_str(total_pct),
            coverage_level.value if hasattr(coverage_level, "value")
            else str(coverage_level),
            len(gap_categories),
        )
        return report

    # ------------------------------------------------------------------
    # Public API -- classify_coverage_level
    # ------------------------------------------------------------------

    def classify_coverage_level(
        self,
        coverage_pct: Decimal,
    ) -> Any:
        """Classify a coverage percentage into a CoverageLevel.

        Uses the COVERAGE_THRESHOLDS constant from models:
        - FULL: 100%
        - HIGH: >= 95%
        - MEDIUM: >= 90%
        - LOW: >= 80%
        - MINIMAL: < 80%

        Args:
            coverage_pct: Coverage percentage (0-100).

        Returns:
            CoverageLevel enum value.
        """
        if _MODELS_AVAILABLE:
            thresholds = COVERAGE_THRESHOLDS
            if coverage_pct >= thresholds.get(
                CoverageLevel.FULL, Decimal("100")
            ):
                return CoverageLevel.FULL
            if coverage_pct >= thresholds.get(
                CoverageLevel.HIGH, Decimal("95")
            ):
                return CoverageLevel.HIGH
            if coverage_pct >= thresholds.get(
                CoverageLevel.MEDIUM, Decimal("90")
            ):
                return CoverageLevel.MEDIUM
            if coverage_pct >= thresholds.get(
                CoverageLevel.LOW, Decimal("80")
            ):
                return CoverageLevel.LOW
            return CoverageLevel.MINIMAL

        # Fallback when models unavailable
        if coverage_pct >= Decimal("100"):
            return "full"
        if coverage_pct >= Decimal("95"):
            return "high"
        if coverage_pct >= Decimal("90"):
            return "medium"
        if coverage_pct >= Decimal("80"):
            return "low"
        return "minimal"

    # ------------------------------------------------------------------
    # Public API -- perform_hotspot_analysis
    # ------------------------------------------------------------------

    def perform_hotspot_analysis(
        self,
        results: List[Any],
        items: List[Any],
        top_n: int = _DEFAULT_TOP_N,
        total_spend_usd: Optional[Decimal] = None,
    ) -> Any:
        """Perform Pareto 80/20 hot-spot analysis on aggregated results.

        Groups emissions by procurement category, ranks them from
        highest to lowest, computes cumulative percentages for Pareto
        analysis, and classifies each category into a materiality
        quadrant based on spend percentage and EF intensity.

        Materiality Quadrants:
            Q1 (Prioritize): High spend (>5%) AND high EF (>0.5 kgCO2e/$)
            Q2 (Monitor): Low spend AND high EF
            Q3 (Improve Data): High spend AND low EF
            Q4 (Low Priority): Low spend AND low EF

        Args:
            results: Combined list of all calculation results.
            items: List of ProcurementItem objects.
            top_n: Number of top categories to include (default 20).
            total_spend_usd: Total spend for percentage calculations.

        Returns:
            HotSpotAnalysis with ranked items and quadrant summary.
        """
        logger.info(
            "perform_hotspot_analysis: results=%d, items=%d, top_n=%d",
            len(results), len(items), top_n,
        )

        # Build category-level aggregations
        cat_emissions: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        cat_spend: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        cat_names: Dict[str, str] = {}

        result_map: Dict[str, Any] = {}
        for r in results:
            rid = self._get_result_item_id(r)
            if rid:
                result_map[rid] = r

        for item in items:
            iid = self._get_item_id(item)
            cat = self._get_item_category(item) or "uncategorized"
            cat_name = self._get_item_category_name(item) or cat
            cat_names[cat] = cat_name
            item_spend = self._get_item_spend(item)
            cat_spend[cat] = _quantize(cat_spend[cat] + item_spend)

            if iid in result_map:
                em = self._get_result_emissions_tco2e(result_map[iid])
                cat_emissions[cat] = _quantize(cat_emissions[cat] + em)

        # Total emissions and spend
        total_em = _quantize(sum(cat_emissions.values(), ZERO))
        eff_spend = (
            total_spend_usd
            if total_spend_usd and total_spend_usd > ZERO
            else _quantize(sum(cat_spend.values(), ZERO))
        )

        # Pareto ranking
        pareto_list = self.compute_pareto_ranking(cat_emissions)

        # Build MaterialityItem list
        materiality_items: List[Any] = []
        cumulative = ZERO
        top_80_count = 0
        quadrant_summary: Dict[str, int] = {
            "prioritize": 0,
            "monitor": 0,
            "improve_data": 0,
            "low_priority": 0,
        }

        for rank, (cat, em) in enumerate(pareto_list[:top_n], start=1):
            em_pct = _pct(em, total_em)
            cumulative = _quantize(cumulative + em_pct)
            sp = cat_spend.get(cat, ZERO)
            sp_pct = _pct(sp, eff_spend)
            ef_intensity = _safe_divide(
                em * ONE_THOUSAND, sp,
            )  # kgCO2e / USD
            quadrant = self.classify_materiality_quadrant(
                sp_pct, ef_intensity,
            )
            recommended = self._recommend_method_for_quadrant(quadrant)
            cat_name = cat_names.get(cat, cat)

            if cumulative <= _PARETO_THRESHOLD_PCT:
                top_80_count = rank

            item = self._build_materiality_item(
                category=cat,
                category_name=cat_name,
                emissions_tco2e=em,
                emissions_pct=em_pct,
                cumulative_pct=cumulative,
                spend_usd=sp,
                spend_pct=sp_pct,
                ef_intensity_kgco2e_per_usd=ef_intensity,
                quadrant=quadrant,
                recommended_method=recommended,
                rank=rank,
            )
            materiality_items.append(item)
            quadrant_summary[quadrant] = (
                quadrant_summary.get(quadrant, 0) + 1
            )

        # If all items are within 80%, set top_80_count to total
        if top_80_count == 0 and materiality_items:
            top_80_count = len(materiality_items)

        # Recommendations
        recommendations = self._generate_hotspot_recommendations(
            materiality_items, quadrant_summary,
        )

        hotspot = self._build_hotspot_analysis(
            calculation_id=str(uuid4()),
            total_emissions_tco2e=total_em,
            total_categories=len(pareto_list),
            top_80_pct_count=top_80_count,
            items=materiality_items,
            quadrant_summary=quadrant_summary,
            recommendations=recommendations,
        )

        logger.info(
            "perform_hotspot_analysis completed: categories=%d, "
            "top_80=%d, Q1=%d, Q2=%d, Q3=%d, Q4=%d",
            len(pareto_list),
            top_80_count,
            quadrant_summary.get("prioritize", 0),
            quadrant_summary.get("monitor", 0),
            quadrant_summary.get("improve_data", 0),
            quadrant_summary.get("low_priority", 0),
        )
        return hotspot

    # ------------------------------------------------------------------
    # Public API -- classify_materiality_quadrant
    # ------------------------------------------------------------------

    def classify_materiality_quadrant(
        self,
        spend_pct: Decimal,
        ef_intensity: Decimal,
    ) -> str:
        """Classify a category into a materiality quadrant.

        Uses two thresholds:
        - Spend: >5% of total = High
        - EF Intensity: >0.5 kgCO2e/$ = High

        Quadrants:
            Q1 (prioritize): High spend + High EF
            Q2 (monitor): Low spend + High EF
            Q3 (improve_data): High spend + Low EF
            Q4 (low_priority): Low spend + Low EF

        Args:
            spend_pct: Spend percentage of total (0-100).
            ef_intensity: Emission factor intensity (kgCO2e/$).

        Returns:
            Quadrant string: 'prioritize', 'monitor', 'improve_data',
            or 'low_priority'.
        """
        high_spend = spend_pct > _HIGH_SPEND_PCT_THRESHOLD
        high_ef = ef_intensity > _HIGH_EF_INTENSITY_THRESHOLD

        if high_spend and high_ef:
            return "prioritize"
        if not high_spend and high_ef:
            return "monitor"
        if high_spend and not high_ef:
            return "improve_data"
        return "low_priority"

    # ------------------------------------------------------------------
    # Public API -- compute_pareto_ranking
    # ------------------------------------------------------------------

    def compute_pareto_ranking(
        self,
        category_emissions: Dict[str, Decimal],
    ) -> List[Tuple[str, Decimal]]:
        """Rank categories by emissions from highest to lowest.

        Returns a sorted list of (category, emissions) tuples for
        Pareto analysis. Categories with zero emissions are included
        at the end.

        Args:
            category_emissions: Dict mapping category to tCO2e.

        Returns:
            List of (category, emissions_tco2e) sorted descending.
        """
        sorted_cats = sorted(
            category_emissions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.debug(
            "compute_pareto_ranking: %d categories ranked",
            len(sorted_cats),
        )
        return sorted_cats

    # ------------------------------------------------------------------
    # Public API -- check_category_boundaries
    # ------------------------------------------------------------------

    def check_category_boundaries(
        self,
        items: List[Any],
    ) -> List[Any]:
        """Check category boundaries for double-counting prevention.

        Examines each procurement item's boolean flags to determine if
        it belongs to a different Scope 3 category:
        - is_capital_good -> Category 2
        - is_fuel_energy -> Category 3
        - is_transport -> Category 4
        - is_business_travel -> Category 6
        - is_intercompany -> Intercompany elimination
        - is_credit_return -> Credit memo / return

        Items matching any exclusion flag are marked as excluded from
        Category 1 with the reason and target category documented.

        Args:
            items: List of ProcurementItem objects to check.

        Returns:
            List of CategoryBoundaryCheck results.
        """
        checks: List[Any] = []
        excluded_count = 0

        for item in items:
            item_id = self._get_item_id(item)
            if not item_id:
                continue

            excluded = False
            reason = ""
            target = None

            for flag, category_desc in _BOUNDARY_FLAG_MAP.items():
                flag_value = getattr(item, flag, False)
                if flag_value:
                    excluded = True
                    reason = (
                        f"Excluded: item belongs to {category_desc}"
                    )
                    target = category_desc
                    break

            check = self._build_boundary_check(
                item_id=item_id,
                excluded=excluded,
                exclusion_reason=reason,
                target_category=target,
                confidence=ONE_HUNDRED if excluded else ONE_HUNDRED,
                description=(
                    reason if excluded
                    else "Item is within Category 1 boundary"
                ),
            )
            checks.append(check)
            if excluded:
                excluded_count += 1

        logger.info(
            "check_category_boundaries: %d items checked, %d excluded",
            len(items), excluded_count,
        )
        return checks

    # ------------------------------------------------------------------
    # Public API -- filter_excluded_items
    # ------------------------------------------------------------------

    def filter_excluded_items(
        self,
        items: List[Any],
        excluded_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Partition items into eligible and excluded lists.

        If excluded_ids is not provided, runs boundary checks first to
        determine which items to exclude.

        Args:
            items: List of ProcurementItem objects.
            excluded_ids: Optional pre-computed set of excluded item IDs.

        Returns:
            Tuple of (eligible_items, excluded_items).
        """
        if excluded_ids is None:
            boundary_checks = self.check_category_boundaries(items)
            excluded_ids = self._collect_excluded_ids(boundary_checks)

        eligible: List[Any] = []
        excluded: List[Any] = []

        for item in items:
            item_id = self._get_item_id(item)
            if item_id and item_id in excluded_ids:
                excluded.append(item)
            else:
                eligible.append(item)

        logger.debug(
            "filter_excluded_items: %d eligible, %d excluded",
            len(eligible), len(excluded),
        )
        return eligible, excluded

    # ------------------------------------------------------------------
    # Public API -- detect_overlap
    # ------------------------------------------------------------------

    def detect_overlap(
        self,
        spend_results: List[Any],
        avgdata_results: List[Any],
        supplier_results: List[Any],
    ) -> List[str]:
        """Detect items covered by multiple calculation methods.

        Identifies item_ids that appear in more than one result set.
        These overlaps must be resolved by method prioritisation to
        prevent double-counting within Category 1.

        Args:
            spend_results: List of SpendBasedResult objects.
            avgdata_results: List of AverageDataResult objects.
            supplier_results: List of SupplierSpecificResult objects.

        Returns:
            List of item_ids that appear in multiple result sets.
        """
        spend_ids = self._result_ids_set(spend_results)
        avgdata_ids = self._result_ids_set(avgdata_results)
        supplier_ids = self._result_ids_set(supplier_results)

        overlaps: Set[str] = set()
        overlaps |= spend_ids & avgdata_ids
        overlaps |= spend_ids & supplier_ids
        overlaps |= avgdata_ids & supplier_ids

        if overlaps:
            logger.info(
                "detect_overlap: %d items in multiple result sets",
                len(overlaps),
            )
        return sorted(overlaps)

    # ------------------------------------------------------------------
    # Public API -- fill_gaps
    # ------------------------------------------------------------------

    def fill_gaps(
        self,
        items: List[Any],
        covered_ids: Set[str],
        spend_results: List[Any],
    ) -> List[Any]:
        """Fill gaps for uncovered items using fallback spend-based results.

        For items that have no result in any method, attempts to find a
        spend-based result. If a spend result exists but was not selected
        (because a higher-tier method was preferred for a different
        overlap), it is included here. Items with no spend result at all
        are logged as gaps requiring manual intervention.

        Args:
            items: List of eligible ProcurementItem objects.
            covered_ids: Set of item_ids already covered by a method.
            spend_results: List of SpendBasedResult objects available
                for gap filling.

        Returns:
            List of SpendBasedResult objects used for gap filling.
        """
        spend_result_map: Dict[str, Any] = {}
        for r in spend_results:
            rid = self._get_result_item_id(r)
            if rid:
                spend_result_map[rid] = r

        gap_fills: List[Any] = []
        unfilled_count = 0

        for item in items:
            item_id = self._get_item_id(item)
            if not item_id or item_id in covered_ids:
                continue

            if item_id in spend_result_map:
                gap_fills.append(spend_result_map[item_id])
                logger.debug(
                    "Gap filled item %s with spend-based result", item_id,
                )
            else:
                unfilled_count += 1
                logger.warning(
                    "Item %s has no result in any method; "
                    "manual data collection needed",
                    item_id,
                )

        if gap_fills or unfilled_count:
            logger.info(
                "fill_gaps: %d items gap-filled, %d remain unfilled",
                len(gap_fills), unfilled_count,
            )
        return gap_fills

    # ------------------------------------------------------------------
    # Public API -- compute_weighted_dqi
    # ------------------------------------------------------------------

    def compute_weighted_dqi(
        self,
        results_with_dqi: List[Any],
    ) -> Decimal:
        """Compute emission-weighted composite DQI score.

        Weights each result's DQI score by its emission contribution to
        produce a single composite score. Higher emissions have more
        influence on the overall quality assessment.

        Formula:
            weighted_dqi = SUM(dqi_i * emissions_i) / SUM(emissions_i)

        If no DQI scores are available, returns the default worst-case
        score of 5.0.

        Args:
            results_with_dqi: List of result objects with dqi_scores
                and emissions_tco2e attributes.

        Returns:
            Emission-weighted composite DQI score (1.0 to 5.0).
        """
        weighted_sum = ZERO
        total_emissions = ZERO

        for result in results_with_dqi:
            dqi = self._extract_composite_dqi(result)
            emissions = self._get_result_emissions_tco2e(result)

            if dqi is not None and emissions > ZERO:
                weighted_sum = _quantize(
                    weighted_sum + (dqi * emissions)
                )
                total_emissions = _quantize(
                    total_emissions + emissions
                )

        if total_emissions == ZERO:
            logger.debug(
                "compute_weighted_dqi: no emissions with DQI; "
                "returning default 5.0"
            )
            return _MAX_DQI

        raw_dqi = _safe_divide(weighted_sum, total_emissions, _MAX_DQI)

        # Clamp to [1.0, 5.0]
        if raw_dqi < _MIN_DQI:
            raw_dqi = _MIN_DQI
        if raw_dqi > _MAX_DQI:
            raw_dqi = _MAX_DQI

        logger.debug(
            "compute_weighted_dqi: %s (from %d results with DQI)",
            _decimal_to_str(raw_dqi),
            sum(
                1 for r in results_with_dqi
                if self._extract_composite_dqi(r) is not None
            ),
        )
        return raw_dqi

    # ------------------------------------------------------------------
    # Public API -- decompose_yoy_change
    # ------------------------------------------------------------------

    def decompose_yoy_change(
        self,
        current: Any,
        prior: Any,
        current_spend: Decimal,
        prior_spend: Decimal,
    ) -> Dict[str, Any]:
        """Decompose year-over-year emission changes into drivers.

        Breaks down the total change in Category 1 emissions between
        two periods into four components:

        1. Activity Effect -- Change due to spend volume changes
           (holding EF and method mix constant).
        2. EF Effect -- Change due to emission factor changes
           (holding activity and method mix constant).
        3. Method Effect -- Change due to method mix changes
           (e.g. upgrading from spend-based to supplier-specific).
        4. Scope Effect -- Residual change due to boundary changes,
           new categories, or discontinued procurement.

        Formula (additive decomposition):
            Total Delta = Activity + EF + Method + Scope

            Activity = prior_tco2e * (current_spend / prior_spend - 1)
            EF = current_spend * (current_intensity - prior_intensity)
                 / 1,000,000
            Method = method_mix_effect (approximated)
            Scope = total_delta - activity - ef - method

        Args:
            current: HybridResult for the current period.
            prior: HybridResult for the prior period.
            current_spend: Total spend in USD for the current period.
            prior_spend: Total spend in USD for the prior period.

        Returns:
            Dict with keys: total_delta_tco2e, activity_effect_tco2e,
            ef_effect_tco2e, method_effect_tco2e, scope_effect_tco2e,
            total_delta_pct, and per-component percentages.
        """
        logger.info("decompose_yoy_change started")

        current_tco2e = self._get_hybrid_total_tco2e(current)
        prior_tco2e = self._get_hybrid_total_tco2e(prior)

        total_delta = _quantize(current_tco2e - prior_tco2e)
        total_delta_pct = (
            _pct(total_delta, prior_tco2e)
            if prior_tco2e > ZERO
            else ZERO
        )

        # Activity effect: change in spend volume
        if prior_spend > ZERO:
            spend_ratio = _safe_divide(current_spend, prior_spend, ONE)
            activity_effect = _quantize(
                prior_tco2e * (spend_ratio - ONE)
            )
        else:
            activity_effect = ZERO

        # EF effect: change in emission intensity
        current_intensity = _safe_divide(
            current_tco2e * _MILLION, current_spend, ZERO,
        )
        prior_intensity = _safe_divide(
            prior_tco2e * _MILLION, prior_spend, ZERO,
        )
        intensity_delta = _quantize(current_intensity - prior_intensity)
        ef_effect = _quantize(
            current_spend * intensity_delta / _MILLION
        )

        # Method effect: change in method mix
        method_effect = self._compute_method_mix_effect(
            current, prior, current_tco2e, prior_tco2e,
        )

        # Scope effect: residual
        scope_effect = _quantize(
            total_delta - activity_effect - ef_effect - method_effect
        )

        # Per-component percentages
        activity_pct = (
            _pct(activity_effect, prior_tco2e)
            if prior_tco2e > ZERO else ZERO
        )
        ef_pct = (
            _pct(ef_effect, prior_tco2e)
            if prior_tco2e > ZERO else ZERO
        )
        method_pct = (
            _pct(method_effect, prior_tco2e)
            if prior_tco2e > ZERO else ZERO
        )
        scope_pct = (
            _pct(scope_effect, prior_tco2e)
            if prior_tco2e > ZERO else ZERO
        )

        result = {
            "total_delta_tco2e": total_delta,
            "total_delta_pct": total_delta_pct,
            "activity_effect_tco2e": activity_effect,
            "activity_effect_pct": activity_pct,
            "ef_effect_tco2e": ef_effect,
            "ef_effect_pct": ef_pct,
            "method_effect_tco2e": method_effect,
            "method_effect_pct": method_pct,
            "scope_effect_tco2e": scope_effect,
            "scope_effect_pct": scope_pct,
            "current_tco2e": current_tco2e,
            "prior_tco2e": prior_tco2e,
            "current_spend_usd": current_spend,
            "prior_spend_usd": prior_spend,
            "current_intensity_tco2e_per_musd": current_intensity,
            "prior_intensity_tco2e_per_musd": prior_intensity,
        }

        logger.info(
            "decompose_yoy_change completed: delta=%s tCO2e (%s%%), "
            "activity=%s, ef=%s, method=%s, scope=%s",
            _decimal_to_str(total_delta),
            _decimal_to_str(total_delta_pct),
            _decimal_to_str(activity_effect),
            _decimal_to_str(ef_effect),
            _decimal_to_str(method_effect),
            _decimal_to_str(scope_effect),
        )
        return result

    # ------------------------------------------------------------------
    # Public API -- compute_intensity_metrics
    # ------------------------------------------------------------------

    def compute_intensity_metrics(
        self,
        total_tco2e: Decimal,
        revenue_usd: Optional[Decimal] = None,
        fte_count: Optional[int] = None,
        total_spend_usd: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Compute emission intensity metrics.

        Calculates standard intensity ratios used for benchmarking and
        target setting:
        - Revenue intensity: tCO2e per $M revenue
        - FTE intensity: tCO2e per full-time equivalent
        - Spend intensity: tCO2e per $M procurement spend

        Args:
            total_tco2e: Total Category 1 emissions in tonnes CO2e.
            revenue_usd: Total company revenue in USD (optional).
            fte_count: Total full-time equivalents (optional).
            total_spend_usd: Total procurement spend in USD (optional).

        Returns:
            Dict with intensity metrics (None for metrics where
            denominator data is unavailable).
        """
        logger.info(
            "compute_intensity_metrics: tco2e=%s, revenue=%s, "
            "fte=%s, spend=%s",
            _decimal_to_str(total_tco2e),
            str(revenue_usd) if revenue_usd else "N/A",
            str(fte_count) if fte_count else "N/A",
            str(total_spend_usd) if total_spend_usd else "N/A",
        )

        result: Dict[str, Any] = {
            "total_emissions_tco2e": total_tco2e,
            "intensity_per_revenue_tco2e_per_musd": None,
            "intensity_per_fte_tco2e_per_fte": None,
            "intensity_per_spend_tco2e_per_musd": None,
        }

        # Revenue intensity
        if revenue_usd is not None and revenue_usd > ZERO:
            revenue_m = _safe_divide(revenue_usd, _MILLION, ZERO)
            if revenue_m > ZERO:
                result["intensity_per_revenue_tco2e_per_musd"] = (
                    _safe_divide(total_tco2e, revenue_m, ZERO)
                )

        # FTE intensity
        if fte_count is not None and fte_count > 0:
            fte_dec = Decimal(str(fte_count))
            result["intensity_per_fte_tco2e_per_fte"] = (
                _safe_divide(total_tco2e, fte_dec, ZERO)
            )

        # Spend intensity
        if total_spend_usd is not None and total_spend_usd > ZERO:
            spend_m = _safe_divide(total_spend_usd, _MILLION, ZERO)
            if spend_m > ZERO:
                result["intensity_per_spend_tco2e_per_musd"] = (
                    _safe_divide(total_tco2e, spend_m, ZERO)
                )

        logger.info(
            "compute_intensity_metrics completed: "
            "rev=%s, fte=%s, spend=%s tCO2e/$M",
            (
                _decimal_to_str(
                    result["intensity_per_revenue_tco2e_per_musd"]
                )
                if result["intensity_per_revenue_tco2e_per_musd"]
                is not None
                else "N/A"
            ),
            (
                _decimal_to_str(
                    result["intensity_per_fte_tco2e_per_fte"]
                )
                if result["intensity_per_fte_tco2e_per_fte"]
                is not None
                else "N/A"
            ),
            (
                _decimal_to_str(
                    result["intensity_per_spend_tco2e_per_musd"]
                )
                if result["intensity_per_spend_tco2e_per_musd"]
                is not None
                else "N/A"
            ),
        )
        return result

    # ------------------------------------------------------------------
    # Public API -- estimate_combined_uncertainty
    # ------------------------------------------------------------------

    def estimate_combined_uncertainty(
        self,
        hybrid_result: Any,
    ) -> Dict[str, Any]:
        """Estimate combined uncertainty for the hybrid result.

        Uses the analytical error propagation method (root-sum-of-squares)
        to combine uncertainties from different calculation methods. Each
        method has a characteristic uncertainty range per UNCERTAINTY_RANGES.

        The combined uncertainty for independent sources is:
            U_combined = sqrt(SUM((U_i * E_i)^2)) / E_total

        Where U_i is the uncertainty factor for method i and E_i is the
        emissions from method i.

        Args:
            hybrid_result: HybridResult containing per-method emissions.

        Returns:
            Dict with combined_uncertainty_pct, lower_bound_tco2e,
            upper_bound_tco2e, and per-method uncertainty contributions.
        """
        logger.info("estimate_combined_uncertainty started")

        total_tco2e = self._get_hybrid_total_tco2e(hybrid_result)
        supplier_tco2e = self._get_hybrid_supplier_tco2e(hybrid_result)
        avgdata_tco2e = self._get_hybrid_avgdata_tco2e(hybrid_result)
        spend_tco2e = self._get_hybrid_spend_tco2e(hybrid_result)

        # Get midpoint uncertainty for each method
        supplier_unc = self._get_method_uncertainty_midpoint(
            "supplier_specific"
        )
        avgdata_unc = self._get_method_uncertainty_midpoint(
            "average_data"
        )
        spend_unc = self._get_method_uncertainty_midpoint(
            "spend_based"
        )

        # Root-sum-of-squares propagation
        sum_sq = ZERO
        method_contributions: Dict[str, Any] = {}

        for name, em, unc in [
            ("supplier_specific", supplier_tco2e, supplier_unc),
            ("average_data", avgdata_tco2e, avgdata_unc),
            ("spend_based", spend_tco2e, spend_unc),
        ]:
            contrib = _quantize(
                (unc / ONE_HUNDRED) * em
            )
            sq = _quantize(contrib * contrib)
            sum_sq = _quantize(sum_sq + sq)
            method_contributions[name] = {
                "emissions_tco2e": em,
                "uncertainty_pct": unc,
                "absolute_uncertainty_tco2e": contrib,
            }

        # Combined absolute uncertainty
        combined_abs = self._decimal_sqrt(sum_sq)
        combined_pct = (
            _pct(combined_abs, total_tco2e)
            if total_tco2e > ZERO
            else ZERO
        )

        lower_bound = _quantize(total_tco2e - combined_abs)
        upper_bound = _quantize(total_tco2e + combined_abs)
        if lower_bound < ZERO:
            lower_bound = ZERO

        result = {
            "total_emissions_tco2e": total_tco2e,
            "combined_uncertainty_pct": combined_pct,
            "combined_absolute_tco2e": combined_abs,
            "lower_bound_tco2e": lower_bound,
            "upper_bound_tco2e": upper_bound,
            "confidence_level_pct": Decimal("95"),
            "method": "analytical_propagation",
            "method_contributions": method_contributions,
        }

        logger.info(
            "estimate_combined_uncertainty completed: +/-%s%% "
            "(%s to %s tCO2e)",
            _decimal_to_str(combined_pct),
            _decimal_to_str(lower_bound),
            _decimal_to_str(upper_bound),
        )
        return result

    # ------------------------------------------------------------------
    # Public API -- health_check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return engine health status and internal counters.

        Provides operational status, configuration availability, and
        cumulative processing statistics for monitoring dashboards.

        Returns:
            Dict with health status, version, counters, and
            component availability flags.
        """
        with self._lock:
            return {
                "engine": "HybridAggregatorEngine",
                "agent_id": AGENT_ID,
                "version": VERSION,
                "status": "healthy",
                "initialized": self._initialized,
                "aggregation_count": self._aggregation_count,
                "total_items_aggregated": self._total_items_aggregated,
                "total_emissions_aggregated_tco2e": (
                    _decimal_to_str(
                        self._total_emissions_aggregated_tco2e
                    )
                ),
                "components": {
                    "config_available": self._config is not None,
                    "metrics_available": self._metrics is not None,
                    "provenance_available": (
                        self._provenance is not None
                    ),
                    "models_available": _MODELS_AVAILABLE,
                },
                "constants": {
                    "high_spend_pct_threshold": str(
                        _HIGH_SPEND_PCT_THRESHOLD
                    ),
                    "high_ef_intensity_threshold": str(
                        _HIGH_EF_INTENSITY_THRESHOLD
                    ),
                    "pareto_threshold_pct": str(
                        _PARETO_THRESHOLD_PCT
                    ),
                    "decimal_places": DECIMAL_PLACES,
                },
                "timestamp": utcnow().isoformat(),
            }

    # ==================================================================
    # Private methods -- validation
    # ==================================================================

    def _validate_aggregate_inputs(
        self,
        spend_results: List[Any],
        avgdata_results: List[Any],
        supplier_results: List[Any],
        items: List[Any],
        total_spend_usd: Decimal,
    ) -> None:
        """Validate inputs for the aggregate method.

        Args:
            spend_results: Spend-based results.
            avgdata_results: Average-data results.
            supplier_results: Supplier-specific results.
            items: Procurement items.
            total_spend_usd: Total spend in USD.

        Raises:
            ValueError: If total_spend_usd is negative.
            TypeError: If inputs are not lists.
        """
        if not isinstance(spend_results, list):
            raise TypeError("spend_results must be a list")
        if not isinstance(avgdata_results, list):
            raise TypeError("avgdata_results must be a list")
        if not isinstance(supplier_results, list):
            raise TypeError("supplier_results must be a list")
        if not isinstance(items, list):
            raise TypeError("items must be a list")
        if total_spend_usd < ZERO:
            raise ValueError(
                f"total_spend_usd must be >= 0, got {total_spend_usd}"
            )

    # ==================================================================
    # Private methods -- item and result accessors
    # ==================================================================

    def _get_item_id(self, item: Any) -> Optional[str]:
        """Extract item_id from a ProcurementItem."""
        return getattr(item, "item_id", None)

    def _get_item_spend(self, item: Any) -> Decimal:
        """Extract spend_amount from a ProcurementItem as Decimal."""
        val = getattr(item, "spend_amount", ZERO)
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_item_category(self, item: Any) -> Optional[str]:
        """Extract material_category from a ProcurementItem."""
        cat = getattr(item, "material_category", None)
        if cat is None:
            return getattr(item, "naics_code", None)
        if hasattr(cat, "value"):
            return cat.value
        return str(cat) if cat else None

    def _get_item_category_name(self, item: Any) -> Optional[str]:
        """Extract a human-readable category name."""
        cat = getattr(item, "material_category", None)
        if cat is not None:
            if hasattr(cat, "value"):
                return cat.value.replace("_", " ").title()
            return str(cat)
        desc = getattr(item, "description", None)
        if desc:
            return str(desc)[:80]
        return None

    def _get_result_item_id(self, result: Any) -> Optional[str]:
        """Extract item_id from a calculation result."""
        return getattr(result, "item_id", None)

    def _get_result_emissions_tco2e(self, result: Any) -> Decimal:
        """Extract emissions_tco2e from a calculation result."""
        val = getattr(result, "emissions_tco2e", ZERO)
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_result_emissions_kgco2e(self, result: Any) -> Decimal:
        """Extract emissions_kgco2e from a calculation result."""
        val = getattr(result, "emissions_kgco2e", ZERO)
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_result_spend_usd(self, result: Any) -> Decimal:
        """Extract spend_usd from a calculation result."""
        val = getattr(result, "spend_usd", None)
        if val is not None:
            return (
                Decimal(str(val))
                if not isinstance(val, Decimal) else val
            )
        # Try spend_original for spend-based results
        val = getattr(result, "spend_original", ZERO)
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_total_tco2e(self, hybrid: Any) -> Decimal:
        """Extract total_emissions_tco2e from a HybridResult."""
        val = getattr(hybrid, "total_emissions_tco2e", ZERO)
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_supplier_tco2e(self, hybrid: Any) -> Decimal:
        """Extract supplier_specific_emissions_tco2e from HybridResult."""
        val = getattr(
            hybrid, "supplier_specific_emissions_tco2e", ZERO,
        )
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_avgdata_tco2e(self, hybrid: Any) -> Decimal:
        """Extract average_data_emissions_tco2e from HybridResult."""
        val = getattr(
            hybrid, "average_data_emissions_tco2e", ZERO,
        )
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_spend_tco2e(self, hybrid: Any) -> Decimal:
        """Extract spend_based_emissions_tco2e from HybridResult."""
        val = getattr(
            hybrid, "spend_based_emissions_tco2e", ZERO,
        )
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_supplier_pct(self, hybrid: Any) -> Decimal:
        """Extract supplier_specific_coverage_pct from HybridResult."""
        val = getattr(
            hybrid, "supplier_specific_coverage_pct", ZERO,
        )
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_avgdata_pct(self, hybrid: Any) -> Decimal:
        """Extract average_data_coverage_pct from HybridResult."""
        val = getattr(
            hybrid, "average_data_coverage_pct", ZERO,
        )
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    def _get_hybrid_spend_pct(self, hybrid: Any) -> Decimal:
        """Extract spend_based_coverage_pct from HybridResult."""
        val = getattr(
            hybrid, "spend_based_coverage_pct", ZERO,
        )
        if val is None:
            return ZERO
        return Decimal(str(val)) if not isinstance(val, Decimal) else val

    # ==================================================================
    # Private methods -- set operations on results
    # ==================================================================

    def _result_ids_set(self, results: List[Any]) -> Set[str]:
        """Build a set of item_ids from a list of results."""
        ids: Set[str] = set()
        for r in results:
            rid = self._get_result_item_id(r)
            if rid:
                ids.add(rid)
        return ids

    def _ids_by_method(
        self,
        item_method_map: Dict[str, Any],
        method_name: str,
    ) -> Set[str]:
        """Get set of item_ids assigned to a specific method."""
        return {
            iid for iid, m in item_method_map.items()
            if self._method_name(m) == method_name
        }

    def _method_name(self, method: Any) -> str:
        """Extract string name from a CalculationMethod or string."""
        if hasattr(method, "value"):
            return method.value
        return str(method)

    def _get_calc_method(self, name: str) -> Any:
        """Get CalculationMethod enum value by name."""
        if _MODELS_AVAILABLE:
            method_map = {
                "supplier_specific": CalculationMethod.SUPPLIER_SPECIFIC,
                "average_data": CalculationMethod.AVERAGE_DATA,
                "spend_based": CalculationMethod.SPEND_BASED,
                "hybrid": CalculationMethod.HYBRID,
            }
            return method_map.get(name, CalculationMethod.SPEND_BASED)
        return name

    # ==================================================================
    # Private methods -- filtering
    # ==================================================================

    def _build_item_map(
        self,
        items: List[Any],
    ) -> Dict[str, Any]:
        """Build item_id -> ProcurementItem lookup map."""
        result: Dict[str, Any] = {}
        for item in items:
            iid = self._get_item_id(item)
            if iid:
                result[iid] = item
        return result

    def _collect_excluded_ids(
        self,
        boundary_checks: List[Any],
    ) -> Set[str]:
        """Collect excluded item_ids from boundary check results."""
        excluded: Set[str] = set()
        for check in boundary_checks:
            if getattr(check, "excluded", False):
                iid = getattr(check, "item_id", None)
                if iid:
                    excluded.add(iid)
        return excluded

    def _filter_results_by_exclusion(
        self,
        results: List[Any],
        excluded_ids: Set[str],
    ) -> List[Any]:
        """Filter results to remove excluded items."""
        return [
            r for r in results
            if self._get_result_item_id(r) not in excluded_ids
        ]

    # ==================================================================
    # Private methods -- aggregation helpers
    # ==================================================================

    def _sum_emissions_tco2e_for_ids(
        self,
        results: List[Any],
        target_ids: Set[str],
    ) -> Decimal:
        """Sum emissions_tco2e for results matching target item_ids."""
        total = ZERO
        for r in results:
            rid = self._get_result_item_id(r)
            if rid and rid in target_ids:
                em = self._get_result_emissions_tco2e(r)
                total = _quantize(total + em)
        return total

    def _sum_spend_for_ids(
        self,
        items: List[Any],
        item_map: Dict[str, Any],
        target_ids: Set[str],
    ) -> Decimal:
        """Sum spend_amount for items matching target item_ids."""
        total = ZERO
        for iid in target_ids:
            if iid in item_map:
                sp = self._get_item_spend(item_map[iid])
                total = _quantize(total + sp)
        return total

    def _sum_result_spend_usd(self, results: List[Any]) -> Decimal:
        """Sum spend_usd across all results."""
        total = ZERO
        for r in results:
            sp = self._get_result_spend_usd(r)
            total = _quantize(total + sp)
        return total

    def _group_spend_by_category(
        self,
        items: List[Any],
    ) -> Dict[str, Decimal]:
        """Group item spend by material category."""
        cat_spend: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for item in items:
            cat = self._get_item_category(item) or "uncategorized"
            sp = self._get_item_spend(item)
            cat_spend[cat] = _quantize(cat_spend[cat] + sp)
        return dict(cat_spend)

    # ==================================================================
    # Private methods -- DQI extraction
    # ==================================================================

    def _extract_composite_dqi(self, result: Any) -> Optional[Decimal]:
        """Extract composite DQI score from a result's dqi_scores dict.

        Computes the arithmetic mean of the five GHG Protocol DQI
        dimensions if present. Returns None if no DQI data available.

        Args:
            result: Calculation result with optional dqi_scores attribute.

        Returns:
            Composite DQI score as Decimal, or None.
        """
        dqi_scores = getattr(result, "dqi_scores", None)
        if not dqi_scores or not isinstance(dqi_scores, dict):
            return None

        dimensions = [
            "temporal", "geographical", "technological",
            "completeness", "reliability",
        ]
        scores: List[Decimal] = []
        for dim in dimensions:
            val = dqi_scores.get(dim)
            if val is not None:
                scores.append(
                    Decimal(str(val))
                    if not isinstance(val, Decimal) else val
                )

        if not scores:
            return None

        total = sum(scores, ZERO)
        count = Decimal(str(len(scores)))
        return _quantize(total / count)

    # ==================================================================
    # Private methods -- YoY decomposition
    # ==================================================================

    def _compute_method_mix_effect(
        self,
        current: Any,
        prior: Any,
        current_tco2e: Decimal,
        prior_tco2e: Decimal,
    ) -> Decimal:
        """Compute the method mix component of YoY decomposition.

        Estimates the emission impact of changes in method mix (e.g.,
        upgrading from spend-based to supplier-specific). Uses the
        difference in method coverage percentages weighted by average
        uncertainty improvement.

        The logic: higher-quality methods typically yield different
        (often lower) emission estimates. The method mix effect
        captures this by comparing the weighted-average method quality
        shift between periods.

        Args:
            current: Current period HybridResult.
            prior: Prior period HybridResult.
            current_tco2e: Current period total emissions.
            prior_tco2e: Prior period total emissions.

        Returns:
            Method mix effect in tCO2e.
        """
        # Compute method-weighted quality index
        # supplier=1.0, avgdata=2.0, spend=3.0 (lower is better)
        current_qi = self._compute_quality_index(current)
        prior_qi = self._compute_quality_index(prior)

        qi_delta = _quantize(current_qi - prior_qi)

        if qi_delta == ZERO:
            return ZERO

        # Approximate method effect as proportional to quality change
        # A move from quality 3.0 to 2.0 (improvement of 1.0) reduces
        # emissions by approximately 10-15% per GHG Protocol guidance
        quality_emission_factor = Decimal("0.10")
        avg_tco2e = _quantize((current_tco2e + prior_tco2e) / Decimal("2"))
        method_effect = _quantize(
            qi_delta * quality_emission_factor * avg_tco2e * Decimal("-1")
        )

        return method_effect

    def _compute_quality_index(self, hybrid: Any) -> Decimal:
        """Compute a method-weighted quality index for a HybridResult.

        Quality weights: supplier=1.0, avgdata=2.0, spend=3.0
        Index = weighted average using coverage percentages.

        Args:
            hybrid: HybridResult.

        Returns:
            Quality index (1.0=best, 3.0=worst).
        """
        supplier_pct = self._get_hybrid_supplier_pct(hybrid)
        avgdata_pct = self._get_hybrid_avgdata_pct(hybrid)
        spend_pct = self._get_hybrid_spend_pct(hybrid)

        total_pct = _quantize(supplier_pct + avgdata_pct + spend_pct)
        if total_pct == ZERO:
            return Decimal("3.0")  # Worst case

        weighted_sum = _quantize(
            supplier_pct * ONE
            + avgdata_pct * Decimal("2")
            + spend_pct * Decimal("3")
        )
        return _safe_divide(weighted_sum, total_pct, Decimal("3.0"))

    # ==================================================================
    # Private methods -- uncertainty helpers
    # ==================================================================

    def _get_method_uncertainty_midpoint(
        self,
        method_name: str,
    ) -> Decimal:
        """Get the midpoint uncertainty percentage for a method.

        Args:
            method_name: Calculation method name string.

        Returns:
            Midpoint uncertainty as Decimal percentage.
        """
        if _MODELS_AVAILABLE:
            method_enum = self._get_calc_method(method_name)
            if method_enum in UNCERTAINTY_RANGES:
                low, high = UNCERTAINTY_RANGES[method_enum]
                return _quantize((low + high) / Decimal("2"))

        # Fallback
        if method_name in _FALLBACK_UNCERTAINTY_RANGES:
            low, high = _FALLBACK_UNCERTAINTY_RANGES[method_name]
            return _quantize((low + high) / Decimal("2"))

        return Decimal("75")  # Conservative default

    def _decimal_sqrt(self, value: Decimal) -> Decimal:
        """Compute the square root of a Decimal using Newton's method.

        Uses iterative approximation to avoid floating-point conversion.

        Args:
            value: Non-negative Decimal value.

        Returns:
            Square root as quantized Decimal.
        """
        if value <= ZERO:
            return ZERO
        if value == ONE:
            return ONE

        # Initial guess using float for convergence speed
        try:
            guess = Decimal(str(math.sqrt(float(value))))
        except (OverflowError, ValueError):
            guess = value / Decimal("2")

        # Newton's method iterations
        for _ in range(50):
            new_guess = _quantize((guess + value / guess) / Decimal("2"))
            if abs(new_guess - guess) <= _QUANTIZER:
                return new_guess
            guess = new_guess

        return _quantize(guess)

    # ==================================================================
    # Private methods -- hot-spot helpers
    # ==================================================================

    def _recommend_method_for_quadrant(self, quadrant: str) -> Any:
        """Recommend a calculation method based on materiality quadrant.

        Q1 (prioritize) -> supplier_specific
        Q2 (monitor) -> average_data
        Q3 (improve_data) -> average_data
        Q4 (low_priority) -> spend_based

        Args:
            quadrant: Materiality quadrant string.

        Returns:
            CalculationMethod enum value.
        """
        recommendations = {
            "prioritize": "supplier_specific",
            "monitor": "average_data",
            "improve_data": "average_data",
            "low_priority": "spend_based",
        }
        method_name = recommendations.get(quadrant, "spend_based")
        return self._get_calc_method(method_name)

    def _generate_hotspot_recommendations(
        self,
        items: List[Any],
        quadrant_summary: Dict[str, int],
    ) -> List[str]:
        """Generate actionable recommendations from hot-spot analysis.

        Args:
            items: List of MaterialityItem objects.
            quadrant_summary: Count of items by quadrant.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        q1_count = quadrant_summary.get("prioritize", 0)
        q2_count = quadrant_summary.get("monitor", 0)
        q3_count = quadrant_summary.get("improve_data", 0)

        if q1_count > 0:
            recommendations.append(
                f"PRIORITY: {q1_count} categories have high spend AND "
                f"high emission intensity. Engage these suppliers for "
                f"primary data (EPDs, PCFs, CDP Supply Chain) to "
                f"improve both accuracy and enable targeted reductions."
            )

        if q2_count > 0:
            recommendations.append(
                f"MONITOR: {q2_count} categories have high emission "
                f"intensity but low spend. Monitor for spend increases "
                f"and consider switching to lower-carbon suppliers."
            )

        if q3_count > 0:
            recommendations.append(
                f"IMPROVE DATA: {q3_count} categories have high spend "
                f"but low EF intensity. Upgrade from spend-based to "
                f"average-data or supplier-specific methods for more "
                f"accurate measurement."
            )

        # Pareto recommendation
        if items:
            top_80 = [
                i for i in items
                if (
                    getattr(i, "cumulative_pct", ONE_HUNDRED)
                    <= _PARETO_THRESHOLD_PCT
                )
            ]
            if top_80:
                recommendations.append(
                    f"PARETO: The top {len(top_80)} categories account "
                    f"for approximately 80% of Category 1 emissions. "
                    f"Focus engagement efforts on these categories for "
                    f"maximum impact."
                )

        if not recommendations:
            recommendations.append(
                "No significant hot-spots identified. Maintain current "
                "data collection and supplier engagement strategy."
            )

        return recommendations

    # ==================================================================
    # Private methods -- model builders
    # ==================================================================

    def _build_hybrid_result(
        self,
        calculation_id: str,
        total_emissions_kgco2e: Decimal,
        total_emissions_tco2e: Decimal,
        spend_based_emissions_tco2e: Decimal,
        average_data_emissions_tco2e: Decimal,
        supplier_specific_emissions_tco2e: Decimal,
        spend_based_coverage_pct: Decimal,
        average_data_coverage_pct: Decimal,
        supplier_specific_coverage_pct: Decimal,
        total_coverage_pct: Decimal,
        coverage_level: Any,
        total_spend_usd: Decimal,
        item_count: int,
        spend_based_count: int,
        average_data_count: int,
        supplier_specific_count: int,
        excluded_count: int,
        weighted_dqi: Decimal,
        provenance_hash: str,
        processing_time_ms: Decimal,
    ) -> Any:
        """Build a HybridResult Pydantic model instance.

        Args:
            All fields required by the HybridResult model.

        Returns:
            HybridResult instance.
        """
        if _MODELS_AVAILABLE:
            return HybridResult(
                calculation_id=calculation_id,
                total_emissions_kgco2e=total_emissions_kgco2e,
                total_emissions_tco2e=total_emissions_tco2e,
                spend_based_emissions_tco2e=spend_based_emissions_tco2e,
                average_data_emissions_tco2e=(
                    average_data_emissions_tco2e
                ),
                supplier_specific_emissions_tco2e=(
                    supplier_specific_emissions_tco2e
                ),
                spend_based_coverage_pct=spend_based_coverage_pct,
                average_data_coverage_pct=average_data_coverage_pct,
                supplier_specific_coverage_pct=(
                    supplier_specific_coverage_pct
                ),
                total_coverage_pct=total_coverage_pct,
                coverage_level=coverage_level,
                total_spend_usd=total_spend_usd,
                item_count=item_count,
                spend_based_count=spend_based_count,
                average_data_count=average_data_count,
                supplier_specific_count=supplier_specific_count,
                excluded_count=excluded_count,
                weighted_dqi=weighted_dqi,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

        # Fallback: return a simple namespace-like dict wrapper
        return _FallbackResult(
            calculation_id=calculation_id,
            total_emissions_kgco2e=total_emissions_kgco2e,
            total_emissions_tco2e=total_emissions_tco2e,
            spend_based_emissions_tco2e=spend_based_emissions_tco2e,
            average_data_emissions_tco2e=average_data_emissions_tco2e,
            supplier_specific_emissions_tco2e=(
                supplier_specific_emissions_tco2e
            ),
            spend_based_coverage_pct=spend_based_coverage_pct,
            average_data_coverage_pct=average_data_coverage_pct,
            supplier_specific_coverage_pct=supplier_specific_coverage_pct,
            total_coverage_pct=total_coverage_pct,
            coverage_level=coverage_level,
            total_spend_usd=total_spend_usd,
            item_count=item_count,
            spend_based_count=spend_based_count,
            average_data_count=average_data_count,
            supplier_specific_count=supplier_specific_count,
            excluded_count=excluded_count,
            weighted_dqi=weighted_dqi,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def _build_coverage_report(
        self,
        total_spend_usd: Decimal,
        supplier_specific_spend_usd: Decimal,
        average_data_spend_usd: Decimal,
        spend_based_spend_usd: Decimal,
        uncovered_spend_usd: Decimal,
        supplier_specific_pct: Decimal,
        average_data_pct: Decimal,
        spend_based_pct: Decimal,
        total_coverage_pct: Decimal,
        coverage_level: Any,
        gap_categories: List[str],
        coverage_by_category: Dict[str, Decimal],
    ) -> Any:
        """Build a CoverageReport Pydantic model instance."""
        if _MODELS_AVAILABLE:
            return CoverageReport(
                total_spend_usd=total_spend_usd,
                supplier_specific_spend_usd=supplier_specific_spend_usd,
                average_data_spend_usd=average_data_spend_usd,
                spend_based_spend_usd=spend_based_spend_usd,
                uncovered_spend_usd=uncovered_spend_usd,
                supplier_specific_pct=supplier_specific_pct,
                average_data_pct=average_data_pct,
                spend_based_pct=spend_based_pct,
                total_coverage_pct=total_coverage_pct,
                coverage_level=coverage_level,
                gap_categories=gap_categories,
                coverage_by_category=coverage_by_category,
            )

        return _FallbackResult(
            total_spend_usd=total_spend_usd,
            supplier_specific_spend_usd=supplier_specific_spend_usd,
            average_data_spend_usd=average_data_spend_usd,
            spend_based_spend_usd=spend_based_spend_usd,
            uncovered_spend_usd=uncovered_spend_usd,
            supplier_specific_pct=supplier_specific_pct,
            average_data_pct=average_data_pct,
            spend_based_pct=spend_based_pct,
            total_coverage_pct=total_coverage_pct,
            coverage_level=coverage_level,
            gap_categories=gap_categories,
            coverage_by_category=coverage_by_category,
        )

    def _build_materiality_item(
        self,
        category: str,
        category_name: str,
        emissions_tco2e: Decimal,
        emissions_pct: Decimal,
        cumulative_pct: Decimal,
        spend_usd: Decimal,
        spend_pct: Decimal,
        ef_intensity_kgco2e_per_usd: Decimal,
        quadrant: str,
        recommended_method: Any,
        rank: int,
    ) -> Any:
        """Build a MaterialityItem Pydantic model instance."""
        if _MODELS_AVAILABLE:
            return MaterialityItem(
                category=category,
                category_name=category_name,
                emissions_tco2e=emissions_tco2e,
                emissions_pct=emissions_pct,
                cumulative_pct=cumulative_pct,
                spend_usd=spend_usd,
                spend_pct=spend_pct,
                ef_intensity_kgco2e_per_usd=ef_intensity_kgco2e_per_usd,
                quadrant=quadrant,
                recommended_method=recommended_method,
                rank=rank,
            )

        return _FallbackResult(
            category=category,
            category_name=category_name,
            emissions_tco2e=emissions_tco2e,
            emissions_pct=emissions_pct,
            cumulative_pct=cumulative_pct,
            spend_usd=spend_usd,
            spend_pct=spend_pct,
            ef_intensity_kgco2e_per_usd=ef_intensity_kgco2e_per_usd,
            quadrant=quadrant,
            recommended_method=recommended_method,
            rank=rank,
        )

    def _build_hotspot_analysis(
        self,
        calculation_id: str,
        total_emissions_tco2e: Decimal,
        total_categories: int,
        top_80_pct_count: int,
        items: List[Any],
        quadrant_summary: Dict[str, int],
        recommendations: List[str],
    ) -> Any:
        """Build a HotSpotAnalysis Pydantic model instance."""
        if _MODELS_AVAILABLE:
            return HotSpotAnalysis(
                calculation_id=calculation_id,
                total_emissions_tco2e=total_emissions_tco2e,
                total_categories=total_categories,
                top_80_pct_count=top_80_pct_count,
                items=items,
                quadrant_summary=quadrant_summary,
                recommendations=recommendations,
            )

        return _FallbackResult(
            calculation_id=calculation_id,
            total_emissions_tco2e=total_emissions_tco2e,
            total_categories=total_categories,
            top_80_pct_count=top_80_pct_count,
            items=items,
            quadrant_summary=quadrant_summary,
            recommendations=recommendations,
        )

    def _build_boundary_check(
        self,
        item_id: str,
        excluded: bool,
        exclusion_reason: str,
        target_category: Optional[str],
        confidence: Decimal,
        description: str,
    ) -> Any:
        """Build a CategoryBoundaryCheck Pydantic model instance."""
        if _MODELS_AVAILABLE:
            return CategoryBoundaryCheck(
                item_id=item_id,
                excluded=excluded,
                exclusion_reason=exclusion_reason,
                target_category=target_category,
                confidence=confidence,
                description=description,
            )

        return _FallbackResult(
            item_id=item_id,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            target_category=target_category,
            confidence=confidence,
            description=description,
        )

    # ==================================================================
    # Private methods -- provenance
    # ==================================================================

    def _record_provenance_stage(
        self,
        calc_id: str,
        provenance_input: Dict[str, Any],
        provenance_hash: str,
    ) -> None:
        """Record the hybrid aggregation stage in the provenance chain.

        Attempts to add the HYBRID_AGGREGATION stage to the provenance
        chain. Silently logs a warning if the chain does not exist or
        provenance tracking is unavailable.

        Args:
            calc_id: Calculation identifier (chain ID).
            provenance_input: Input data for provenance hashing.
            provenance_hash: Pre-computed SHA-256 hash.
        """
        if not self._provenance:
            return
        if not _PROVENANCE_AVAILABLE or ProvenanceStage is None:
            return

        try:
            self._provenance.add_stage(
                chain_id=calc_id,
                stage=ProvenanceStage.HYBRID_AGGREGATION,
                metadata={
                    "engine": "HybridAggregatorEngine",
                    "version": VERSION,
                    "provenance_hash": provenance_hash,
                },
                output_data=provenance_input,
            )
        except (ValueError, KeyError) as exc:
            logger.debug(
                "Provenance stage recording skipped for %s: %s",
                calc_id, exc,
            )
        except Exception as exc:
            logger.warning(
                "Unexpected error recording provenance for %s: %s",
                calc_id, exc,
            )

    # ==================================================================
    # Private methods -- metrics
    # ==================================================================

    def _record_aggregation_metrics(
        self,
        calc_id: str,
        total_emissions_tco2e: Decimal,
        total_spend_usd: Decimal,
        item_count: int,
        processing_time_s: float,
        coverage_level: Any,
    ) -> None:
        """Record Prometheus metrics for the aggregation.

        Args:
            calc_id: Calculation identifier.
            total_emissions_tco2e: Total aggregated emissions.
            total_spend_usd: Total spend processed.
            item_count: Number of items aggregated.
            processing_time_s: Processing duration in seconds.
            coverage_level: Coverage level classification.
        """
        if not self._metrics:
            return

        try:
            self._metrics.record_calculation(
                tenant_id="system",
                method="hybrid",
                status="success",
                duration_s=processing_time_s,
                emissions_kgco2e=float(
                    total_emissions_tco2e * ONE_THOUSAND
                ),
                spend_usd=float(total_spend_usd),
            )
        except Exception as exc:
            logger.debug(
                "Metrics recording failed for %s: %s", calc_id, exc,
            )

# ===========================================================================
# Fallback result container
# ===========================================================================

class _FallbackResult:
    """Simple attribute container used when Pydantic models unavailable.

    Provides attribute-style access to keyword arguments passed at
    construction. Used only when the models module cannot be imported.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Store all keyword arguments as instance attributes."""
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        """Return string representation."""
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        return f"_FallbackResult({attrs})"
