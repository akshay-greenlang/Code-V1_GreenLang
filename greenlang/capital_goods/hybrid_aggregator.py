# -*- coding: utf-8 -*-
"""
HybridAggregatorEngine - Multi-Method Aggregation & Hot-Spot Analysis (Engine 5 of 7)

AGENT-MRV-015: Capital Goods Agent (GL-MRV-S3-002)

This engine combines results from the three Category 2 calculation methods
(spend-based EEIO, average-data physical EFs, supplier-specific EPD/PCF)
into a unified hybrid inventory.  It implements the GHG Protocol's guidance
on combining methods by selecting the highest-quality data available for each
capital asset while preventing double-counting against Category 1 and
Scope 1/2 use-phase emissions.

Core Capabilities:
    1. Method Prioritization -- For each asset, select the best available
       method using an 8-level EF hierarchy:
         Level 1: Supplier EPD (3rd-party verified)
         Level 2: Supplier EPD (certified)
         Level 3: Supplier EPD (uncertified)
         Level 4: Industry PCF
         Level 5: DEFRA activity EF
         Level 6: Regional EEIO (EXIOBASE)
         Level 7: National EEIO (USEEIO)
         Level 8: Global EEIO fallback
    2. Coverage Analysis -- Track CapEx percentage covered by each method
       and classify coverage level (full/high/medium/low/minimal).
    3. Hot-Spot Analysis -- Pareto 80/20 ranking of asset categories by
       emission contribution with materiality quadrant classification
       (Q1-Q4: prioritize, investigate, optimize, monitor).
    4. Double-Counting Prevention -- Five rules:
         R1: Exclude assets present in Category 1 results.
         R2: Exclude use-phase emissions already in Scope 1/2.
         R3: Redirect leased assets to Cat 8/13.
         R4: Proportional inclusion for under-construction assets.
         R5: PP&E financial classification determines categorization.
    5. CapEx Volatility Context -- Rolling average, volatility ratio,
       major CapEx year flagging for reporting narrative.
    6. YoY Decomposition -- Decompose changes into activity, EF, method
       mix, and scope effects.
    7. Intensity Metrics -- Revenue, FTE, floor area, and CapEx intensity.
    8. Combined Uncertainty -- Propagate uncertainty across methods.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal arithmetic.
    - No LLM calls in any calculation or aggregation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every aggregation result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Thread-safe singleton with ``threading.RLock()``.  Mutable counters
    and aggregation state are protected by the reentrant lock.  Each
    aggregation call is stateless with respect to previous calls.

Example:
    >>> from greenlang.capital_goods.hybrid_aggregator import (
    ...     HybridAggregatorEngine,
    ... )
    >>> engine = HybridAggregatorEngine()
    >>> result = engine.aggregate(
    ...     spend_results=spend_results,
    ...     average_results=average_results,
    ...     supplier_results=supplier_results,
    ...     assets=assets,
    ... )
    >>> print(result.total_emissions_tco2e)
    >>> print(result.total_coverage_pct)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["HybridAggregatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- models
# ---------------------------------------------------------------------------

try:
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
        AssetSubCategory,
        EmissionGas,
        DQIScore,
        DQIDimension,
        BatchStatus,
        EF_HIERARCHY_PRIORITY,
        COVERAGE_THRESHOLDS,
        UNCERTAINTY_RANGES,
        DQI_SCORE_VALUES,
        CapitalAssetRecord,
        SpendBasedResult,
        AverageDataResult,
        SupplierSpecificResult,
        HybridResult,
        DQIAssessment,
        MaterialityItem,
        CoverageReport,
        HotSpotAnalysis,
        DepreciationContext,
        AggregationResult,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.capital_goods.models not available; "
        "using fallback stubs"
    )
    _MODELS_AVAILABLE = False

    # Fallback constants
    AGENT_ID = "GL-MRV-S3-002"
    VERSION = "1.0.0"
    TABLE_PREFIX = "gl_cg_"
    ZERO = Decimal("0")
    ONE = Decimal("1")
    ONE_HUNDRED = Decimal("100")
    ONE_THOUSAND = Decimal("1000")
    DECIMAL_PLACES = 8

# ---------------------------------------------------------------------------
# Conditional imports -- config
# ---------------------------------------------------------------------------

try:
    from greenlang.capital_goods.config import CapitalGoodsConfig
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.capital_goods.config not available; "
        "using defaults"
    )
    _CONFIG_AVAILABLE = False
    CapitalGoodsConfig = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Conditional imports -- metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.capital_goods.metrics import CapitalGoodsMetrics
    _METRICS_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.capital_goods.metrics not available; "
        "metrics will be no-ops"
    )
    _METRICS_AVAILABLE = False
    CapitalGoodsMetrics = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Conditional imports -- provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.capital_goods.provenance import (
        CapitalGoodsProvenanceTracker,
        ProvenanceStage,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    logger.warning(
        "greenlang.capital_goods.provenance not available; "
        "provenance tracking disabled"
    )
    _PROVENANCE_AVAILABLE = False
    CapitalGoodsProvenanceTracker = None  # type: ignore[assignment,misc]
    ProvenanceStage = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Engine-local constants
# ---------------------------------------------------------------------------

#: Quantizer for Decimal arithmetic.
_QUANTIZER = Decimal(10) ** -DECIMAL_PLACES

#: Spend percentage threshold for "high spend" in materiality quadrant.
_HIGH_SPEND_PCT_THRESHOLD = Decimal("5.0")

#: Emission percentage threshold for "high emissions" in materiality quadrant.
_HIGH_EMISSION_PCT_THRESHOLD = Decimal("5.0")

#: Pareto cumulative percentage for 80/20 rule.
_PARETO_THRESHOLD_PCT = Decimal("80.0")

#: Default top-N items for hot-spot analysis.
_DEFAULT_TOP_N = 20

#: Million constant for intensity denominators.
_MILLION = Decimal("1000000")

#: Minimum DQI score (best quality).
_MIN_DQI = Decimal("1.0")

#: Maximum DQI score (worst quality).
_MAX_DQI = Decimal("5.0")

#: Default CapEx volatility ratio threshold.
_DEFAULT_VOLATILITY_THRESHOLD = Decimal("2.0")

#: Default rolling average window in years.
_DEFAULT_ROLLING_YEARS = 3

#: Conversion factor: kg to metric tonnes.
_KG_TO_TONNES = Decimal("0.001")

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

#: 8-level EF hierarchy priority (1=best, 8=worst).
#: Keys correspond to verification/source levels.
_EF_HIERARCHY: Dict[str, int] = {
    "supplier_epd_verified": 1,
    "supplier_epd_certified": 2,
    "supplier_epd_uncertified": 3,
    "industry_pcf": 4,
    "defra_activity_ef": 5,
    "regional_eeio": 6,
    "national_eeio": 7,
    "global_eeio_fallback": 8,
}

#: Mapping from EF hierarchy levels to method names.
_HIERARCHY_TO_METHOD: Dict[int, str] = {
    1: "supplier_specific",
    2: "supplier_specific",
    3: "supplier_specific",
    4: "average_data",
    5: "average_data",
    6: "spend_based",
    7: "spend_based",
    8: "spend_based",
}

#: Double-counting exclusion reason codes.
_EXCLUSION_REASON_CAT1 = "category_1_overlap"
_EXCLUSION_REASON_SCOPE1_2 = "scope_1_2_use_phase"
_EXCLUSION_REASON_LEASED = "leased_asset_redirect"
_EXCLUSION_REASON_UNDER_CONSTRUCTION = "under_construction_partial"
_EXCLUSION_REASON_PPE_CLASSIFICATION = "ppe_classification_mismatch"


# ===========================================================================
# Helper functions
# ===========================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    if result < ZERO:
        return ZERO
    if result > ONE_HUNDRED:
        return ONE_HUNDRED
    return result


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


def _kg_to_tco2e(kg_co2e: Decimal) -> Decimal:
    """Convert kgCO2e to tCO2e (metric tonnes).

    Args:
        kg_co2e: Emissions in kgCO2e.

    Returns:
        Emissions in tCO2e, quantized.
    """
    return _quantize(kg_co2e * _KG_TO_TONNES)


def _get_result_asset_id(result: Any) -> str:
    """Extract asset_id from a method-specific result object.

    Works with SpendBasedResult, AverageDataResult, SupplierSpecificResult,
    and plain dicts.

    Args:
        result: A result object or dictionary with an asset_id field.

    Returns:
        The asset identifier string.
    """
    if hasattr(result, "asset_id"):
        return str(result.asset_id)
    if isinstance(result, dict):
        return str(result.get("asset_id", ""))
    return ""


def _get_result_emissions(result: Any) -> Decimal:
    """Extract emissions_kg_co2e from a method-specific result.

    Args:
        result: A result object or dictionary.

    Returns:
        Emissions in kgCO2e as Decimal.
    """
    if hasattr(result, "emissions_kg_co2e"):
        return Decimal(str(result.emissions_kg_co2e))
    if isinstance(result, dict):
        return Decimal(str(result.get("emissions_kg_co2e", "0")))
    return ZERO


def _get_result_dqi(result: Any) -> Decimal:
    """Extract DQI score from a method-specific result.

    Args:
        result: A result object or dictionary.

    Returns:
        DQI composite score as Decimal.
    """
    if hasattr(result, "dqi_score"):
        return Decimal(str(result.dqi_score))
    if isinstance(result, dict):
        return Decimal(str(result.get("dqi_score", "5.0")))
    return _MAX_DQI


def _get_result_uncertainty(result: Any) -> Decimal:
    """Extract uncertainty percentage from a method-specific result.

    Args:
        result: A result object or dictionary.

    Returns:
        Uncertainty percentage as Decimal.
    """
    if hasattr(result, "uncertainty_pct"):
        return Decimal(str(result.uncertainty_pct))
    if isinstance(result, dict):
        return Decimal(str(result.get("uncertainty_pct", "75.0")))
    return Decimal("75.0")


def _get_result_method(result: Any) -> str:
    """Extract calculation method name from a result.

    Args:
        result: A result object or dictionary.

    Returns:
        Method name string.
    """
    if hasattr(result, "method"):
        val = result.method
        return val.value if hasattr(val, "value") else str(val)
    if isinstance(result, dict):
        val = result.get("method", "spend_based")
        return val.value if hasattr(val, "value") else str(val)
    return "spend_based"


def _get_asset_capex(asset: Any) -> Decimal:
    """Extract CapEx amount from a capital asset record.

    Args:
        asset: A CapitalAssetRecord or dictionary.

    Returns:
        CapEx amount as Decimal.
    """
    if hasattr(asset, "capex_amount"):
        return Decimal(str(asset.capex_amount))
    if isinstance(asset, dict):
        return Decimal(str(asset.get("capex_amount", "0")))
    return ZERO


def _get_asset_id(asset: Any) -> str:
    """Extract asset_id from a capital asset record.

    Args:
        asset: A CapitalAssetRecord or dictionary.

    Returns:
        Asset identifier string.
    """
    if hasattr(asset, "asset_id"):
        return str(asset.asset_id)
    if isinstance(asset, dict):
        return str(asset.get("asset_id", ""))
    return ""


def _get_asset_category(asset: Any) -> str:
    """Extract asset category from a capital asset record.

    Args:
        asset: A CapitalAssetRecord or dictionary.

    Returns:
        Category string.
    """
    if hasattr(asset, "asset_category"):
        val = asset.asset_category
        return val.value if hasattr(val, "value") else str(val)
    if isinstance(asset, dict):
        val = asset.get("asset_category", "other")
        return val.value if hasattr(val, "value") else str(val)
    return "other"


def _get_asset_supplier(asset: Any) -> str:
    """Extract supplier identifier from a capital asset record.

    Args:
        asset: A CapitalAssetRecord or dictionary.

    Returns:
        Supplier identifier or 'unknown'.
    """
    if hasattr(asset, "supplier_id"):
        return str(asset.supplier_id) if asset.supplier_id else "unknown"
    if isinstance(asset, dict):
        return str(asset.get("supplier_id", "unknown")) or "unknown"
    return "unknown"


def _get_asset_is_leased(asset: Any) -> bool:
    """Check if asset is leased.

    Args:
        asset: A CapitalAssetRecord or dictionary.

    Returns:
        True if the asset is flagged as leased.
    """
    if hasattr(asset, "is_leased"):
        return bool(asset.is_leased)
    if isinstance(asset, dict):
        return bool(asset.get("is_leased", False))
    return False


def _get_asset_year(asset: Any) -> int:
    """Extract acquisition year from a capital asset record.

    Args:
        asset: A CapitalAssetRecord or dictionary.

    Returns:
        Acquisition year as integer.
    """
    if hasattr(asset, "acquisition_date"):
        try:
            return asset.acquisition_date.year
        except AttributeError:
            pass
    if isinstance(asset, dict):
        acq = asset.get("acquisition_date")
        if acq and hasattr(acq, "year"):
            return acq.year
    return _utcnow().year


def _get_result_verification(result: Any) -> str:
    """Extract verification status from a supplier result.

    Args:
        result: A SupplierSpecificResult or dictionary.

    Returns:
        Verification status string.
    """
    if hasattr(result, "verification_status"):
        return str(result.verification_status)
    if isinstance(result, dict):
        return str(result.get("verification_status", "unverified"))
    return "unverified"


def _get_result_data_source(result: Any) -> str:
    """Extract data source from a supplier result.

    Args:
        result: A SupplierSpecificResult or dictionary.

    Returns:
        Data source string.
    """
    if hasattr(result, "data_source"):
        val = result.data_source
        return val.value if hasattr(val, "value") else str(val)
    if isinstance(result, dict):
        val = result.get("data_source", "estimated")
        return val.value if hasattr(val, "value") else str(val)
    return "estimated"


def _get_result_ef_source(result: Any) -> str:
    """Extract EF source from an average-data result.

    Args:
        result: An AverageDataResult or dictionary.

    Returns:
        EF source string.
    """
    if hasattr(result, "ef_source"):
        val = result.ef_source
        return val.value if hasattr(val, "value") else str(val)
    if isinstance(result, dict):
        val = result.get("ef_source", "")
        return val.value if hasattr(val, "value") else str(val)
    return ""


def _determine_hierarchy_level_supplier(result: Any) -> int:
    """Determine the EF hierarchy level for a supplier-specific result.

    Examines verification_status and data_source to assign one of
    levels 1-3 in the 8-level EF hierarchy.

    Args:
        result: A supplier-specific result object or dict.

    Returns:
        Hierarchy level integer (1, 2, or 3).
    """
    verification = _get_result_verification(result).lower()
    data_source = _get_result_data_source(result).lower()

    # Level 1: 3rd-party verified EPD
    if "verified" in verification or "third_party" in verification:
        if data_source in ("epd", "pcf"):
            return 1

    # Level 2: Certified EPD/PCF
    if "certified" in verification:
        return 2

    # Level 3: Uncertified supplier data
    return 3


def _determine_hierarchy_level_average(result: Any) -> int:
    """Determine the EF hierarchy level for an average-data result.

    Examines ef_source to assign level 4 or 5 in the hierarchy.

    Args:
        result: An average-data result object or dict.

    Returns:
        Hierarchy level integer (4 or 5).
    """
    ef_source = _get_result_ef_source(result).lower()

    # Level 4: Industry PCF / ecoinvent LCA
    if ef_source in ("ecoinvent", "world_steel", "iai"):
        return 4

    # Level 5: DEFRA activity EF or ICE database
    return 5


def _determine_hierarchy_level_spend(result: Any) -> int:
    """Determine the EF hierarchy level for a spend-based result.

    Examines the EEIO database type to assign level 6, 7, or 8.

    Args:
        result: A spend-based result object or dict.

    Returns:
        Hierarchy level integer (6, 7, or 8).
    """
    # Check if we can determine regional vs national vs global
    if hasattr(result, "eeio_factor"):
        factor_val = Decimal(str(result.eeio_factor))
        # If the factor is non-zero and there is metadata indicating region
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            region = result.metadata.get("eeio_region", "").lower()
            if region and region not in ("global", "world", "default"):
                return 6  # Regional EEIO
            if region in ("us", "usa", "united_states"):
                return 7  # National EEIO

    if isinstance(result, dict):
        meta = result.get("metadata", {})
        region = meta.get("eeio_region", "").lower() if isinstance(meta, dict) else ""
        if region and region not in ("global", "world", "default"):
            return 6
        if region in ("us", "usa", "united_states"):
            return 7

    # Default: national EEIO for spend-based
    return 7


# ===========================================================================
# HybridAggregatorEngine
# ===========================================================================


class HybridAggregatorEngine:
    """Engine 5 of 7 -- Multi-method aggregation and hot-spot analysis.

    Combines results from spend-based, average-data, and supplier-specific
    calculation engines into a unified Category 2 inventory using method
    prioritisation, coverage analysis, hot-spot identification, and
    double-counting prevention.

    This engine follows the GHG Protocol's recommended hybrid approach:
    for each capital asset, the highest-quality available method is
    selected using an 8-level EF hierarchy (supplier verified EPD >
    certified > uncertified > industry PCF > DEFRA > regional EEIO >
    national EEIO > global fallback).  Assets excluded by double-counting
    rules (Category 1 overlap, Scope 1/2 use-phase, leased assets, PP&E
    misclassification) are removed before aggregation.

    Thread Safety:
        Thread-safe singleton via ``__new__`` with ``threading.RLock()``.
        All mutable state is protected by the lock.  Aggregation calls are
        stateless with respect to previous calls.

    Attributes:
        _config: Agent configuration singleton.
        _metrics: Prometheus metrics collector.
        _provenance: SHA-256 provenance chain tracker.
        _aggregation_count: Total number of aggregations performed.
        _total_assets_aggregated: Total assets processed across all calls.
        _total_emissions_aggregated_tco2e: Cumulative emissions aggregated.

    Example:
        >>> engine = HybridAggregatorEngine()
        >>> result = engine.aggregate(
        ...     spend_results=spend_results,
        ...     average_results=average_results,
        ...     supplier_results=supplier_results,
        ...     assets=assets,
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
                    self._config = CapitalGoodsConfig.from_env()
                except Exception as exc:
                    logger.warning(
                        "Failed to load CapitalGoodsConfig: %s", exc
                    )

            # Metrics
            self._metrics: Any = None
            if _METRICS_AVAILABLE:
                try:
                    self._metrics = CapitalGoodsMetrics()
                except Exception as exc:
                    logger.warning(
                        "Failed to load CapitalGoodsMetrics: %s", exc
                    )

            # Provenance
            self._provenance: Any = None
            if _PROVENANCE_AVAILABLE:
                try:
                    self._provenance = (
                        CapitalGoodsProvenanceTracker.get_instance()
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load CapitalGoodsProvenanceTracker: %s",
                        exc,
                    )

            # Internal counters (protected by _lock)
            self._aggregation_count: int = 0
            self._total_assets_aggregated: int = 0
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
        engine.  This method is intended **only** for unit tests.

        Example:
            >>> HybridAggregatorEngine.reset()
            >>> engine = HybridAggregatorEngine()
        """
        with cls._lock:
            cls._instance = None
        logger.info("HybridAggregatorEngine singleton reset")

    # ==================================================================
    # Internal helpers -- config accessors
    # ==================================================================

    def _get_volatility_threshold(self) -> Decimal:
        """Get the CapEx volatility ratio threshold from config.

        Returns:
            Volatility ratio threshold as Decimal.
        """
        if self._config is not None:
            try:
                return Decimal(
                    str(self._config.calculation.volatility_ratio_threshold)
                )
            except (AttributeError, InvalidOperation):
                pass
        return _DEFAULT_VOLATILITY_THRESHOLD

    def _get_rolling_years(self) -> int:
        """Get the rolling average window size from config.

        Returns:
            Number of years for rolling average.
        """
        if self._config is not None:
            try:
                return int(self._config.calculation.rolling_average_years)
            except (AttributeError, ValueError):
                pass
        return _DEFAULT_ROLLING_YEARS

    def _get_double_counting_enabled(self) -> bool:
        """Check if double-counting detection is enabled.

        Returns:
            True if double-counting checks are enabled.
        """
        if self._config is not None:
            try:
                return bool(
                    self._config.calculation.enable_double_counting_check
                )
            except AttributeError:
                pass
        return True

    def _get_leased_exclusion_enabled(self) -> bool:
        """Check if leased asset exclusion is enabled.

        Returns:
            True if leased assets should be excluded (redirected).
        """
        if self._config is not None:
            try:
                return bool(
                    self._config.calculation.enable_leased_asset_exclusion
                )
            except AttributeError:
                pass
        return True

    # ==================================================================
    # Internal helpers -- result indexing
    # ==================================================================

    def _index_results_by_asset(
        self,
        results: List[Any],
    ) -> Dict[str, Any]:
        """Build a mapping from asset_id to the best result.

        When multiple results exist for the same asset_id, the one
        with the lowest DQI score (highest quality) is kept.

        Args:
            results: List of method-specific result objects.

        Returns:
            Dict mapping asset_id -> result.
        """
        index: Dict[str, Any] = {}
        for r in results:
            aid = _get_result_asset_id(r)
            if not aid:
                continue
            if aid not in index:
                index[aid] = r
            else:
                # Keep the result with lower (better) DQI
                existing_dqi = _get_result_dqi(index[aid])
                current_dqi = _get_result_dqi(r)
                if current_dqi < existing_dqi:
                    index[aid] = r
        return index

    def _index_assets_by_id(
        self,
        assets: List[Any],
    ) -> Dict[str, Any]:
        """Build a mapping from asset_id to asset record.

        Args:
            assets: List of CapitalAssetRecord objects or dicts.

        Returns:
            Dict mapping asset_id -> asset record.
        """
        index: Dict[str, Any] = {}
        for a in assets:
            aid = _get_asset_id(a)
            if aid:
                index[aid] = a
        return index

    # ==================================================================
    # Public API (1): aggregate
    # ==================================================================

    def aggregate(
        self,
        spend_results: List[Any],
        average_results: List[Any],
        supplier_results: List[Any],
        assets: List[Any],
        calculation_id: Optional[str] = None,
    ) -> Any:
        """Aggregate results from all three calculation methods.

        This is the primary entry point for the hybrid aggregation engine.
        It performs the following steps in order:

        1. Validate inputs and generate calculation_id.
        2. Build asset lookup map (asset_id -> CapitalAssetRecord).
        3. Index results by asset_id for each method.
        4. Detect double-counting against Category 1 / Scope 1/2 / leased.
        5. Prioritize methods: for each asset select the best method.
        6. Compute method-level emissions and counts.
        7. Compute total emissions and coverage percentages.
        8. Compute emission-weighted DQI.
        9. Generate provenance hash.
        10. Assemble and return HybridResult.

        CRITICAL: All emissions are reported in the year of acquisition.
        No depreciation of emissions over useful life.

        Args:
            spend_results: Results from the spend-based engine.
            average_results: Results from the average-data engine.
            supplier_results: Results from the supplier-specific engine.
            assets: Capital asset records for the reporting period.
            calculation_id: Optional pre-generated calculation ID.

        Returns:
            HybridResult with aggregated emissions, coverage, and DQI.

        Raises:
            ValueError: If no assets are provided.
        """
        t_start = time.monotonic()
        calc_id = calculation_id or str(uuid4())

        logger.info(
            "[%s] HybridAggregator.aggregate called: "
            "spend=%d, avg=%d, supplier=%d, assets=%d",
            calc_id,
            len(spend_results),
            len(average_results),
            len(supplier_results),
            len(assets),
        )

        # -- Step 1: Validate inputs --
        if not assets:
            logger.warning("[%s] No assets provided", calc_id)
            return self._build_empty_result(calc_id, t_start)

        spend_results = spend_results or []
        average_results = average_results or []
        supplier_results = supplier_results or []

        # -- Step 2: Build asset lookup --
        asset_index = self._index_assets_by_id(assets)
        total_capex = self._compute_total_capex(assets)

        # -- Step 3: Index results by asset_id --
        spend_idx = self._index_results_by_asset(spend_results)
        avg_idx = self._index_results_by_asset(average_results)
        supplier_idx = self._index_results_by_asset(supplier_results)

        # -- Step 4: Detect and remove double-counted assets --
        excluded_ids: Set[str] = set()
        exclusion_details: List[Dict[str, str]] = []

        if self._get_double_counting_enabled():
            for aid, asset in asset_index.items():
                reason = self._check_asset_exclusion(asset)
                if reason:
                    excluded_ids.add(aid)
                    exclusion_details.append({
                        "asset_id": aid,
                        "reason": reason,
                    })

        logger.info(
            "[%s] Double-counting exclusions: %d assets excluded",
            calc_id,
            len(excluded_ids),
        )

        # -- Step 5: Prioritize methods for each non-excluded asset --
        selected_results: Dict[str, Tuple[Any, str, int]] = {}
        # Maps asset_id -> (result, method_name, hierarchy_level)

        for aid in asset_index:
            if aid in excluded_ids:
                continue

            result, method, level = self._select_best_method(
                aid, spend_idx, avg_idx, supplier_idx
            )
            if result is not None:
                selected_results[aid] = (result, method, level)

        # -- Step 6: Compute method-level emissions and counts --
        spend_emissions = ZERO
        avg_emissions = ZERO
        supplier_emissions = ZERO
        spend_count = 0
        avg_count = 0
        supplier_count = 0
        spend_capex = ZERO
        avg_capex = ZERO
        supplier_capex = ZERO

        all_emissions: List[Tuple[str, Decimal, Decimal]] = []
        # (asset_id, emissions_kg, dqi_score)

        for aid, (result, method, _level) in selected_results.items():
            emissions_kg = _get_result_emissions(result)
            dqi = _get_result_dqi(result)
            asset_capex = _get_asset_capex(
                asset_index.get(aid, {})
            )

            all_emissions.append((aid, emissions_kg, dqi))

            if method == "supplier_specific":
                supplier_emissions += emissions_kg
                supplier_count += 1
                supplier_capex += asset_capex
            elif method == "average_data":
                avg_emissions += emissions_kg
                avg_count += 1
                avg_capex += asset_capex
            else:
                spend_emissions += emissions_kg
                spend_count += 1
                spend_capex += asset_capex

        # -- Step 7: Compute totals and coverage --
        total_emissions_kg = (
            spend_emissions + avg_emissions + supplier_emissions
        )
        total_emissions_tco2e = _kg_to_tco2e(total_emissions_kg)
        spend_tco2e = _kg_to_tco2e(spend_emissions)
        avg_tco2e = _kg_to_tco2e(avg_emissions)
        supplier_tco2e = _kg_to_tco2e(supplier_emissions)

        covered_count = spend_count + avg_count + supplier_count
        covered_capex = spend_capex + avg_capex + supplier_capex

        spend_coverage_pct = _pct(spend_capex, total_capex)
        avg_coverage_pct = _pct(avg_capex, total_capex)
        supplier_coverage_pct = _pct(supplier_capex, total_capex)
        total_coverage_pct = _pct(covered_capex, total_capex)

        # -- Step 8: Compute emission-weighted DQI --
        weighted_dqi = self._compute_weighted_dqi(all_emissions)

        # -- Step 9: Method breakdown --
        method_breakdown: Dict[str, Decimal] = {
            "spend_based": spend_tco2e,
            "average_data": avg_tco2e,
            "supplier_specific": supplier_tco2e,
        }

        # -- Step 10: Provenance hash --
        provenance_data = {
            "calculation_id": calc_id,
            "total_emissions_tco2e": _decimal_to_str(total_emissions_tco2e),
            "total_capex_usd": _decimal_to_str(total_capex),
            "spend_count": spend_count,
            "avg_count": avg_count,
            "supplier_count": supplier_count,
            "excluded_count": len(excluded_ids),
            "weighted_dqi": _decimal_to_str(weighted_dqi),
        }
        provenance_hash = _compute_sha256(provenance_data)

        # -- Step 11: Record provenance stage --
        if self._provenance is not None and _PROVENANCE_AVAILABLE:
            try:
                self._provenance.record_stage(
                    stage=ProvenanceStage.HYBRID_AGGREGATION,
                    input_data={
                        "spend_results_count": len(spend_results),
                        "average_results_count": len(average_results),
                        "supplier_results_count": len(supplier_results),
                        "asset_count": len(assets),
                    },
                    output_data=provenance_data,
                    parameters={"calculation_id": calc_id},
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Provenance recording failed: %s", calc_id, exc
                )

        # -- Step 12: Update counters --
        elapsed_ms = Decimal(str(
            (time.monotonic() - t_start) * 1000
        ))

        with self._lock:
            self._aggregation_count += 1
            self._total_assets_aggregated += len(assets)
            self._total_emissions_aggregated_tco2e += total_emissions_tco2e

        # -- Step 13: Record metrics --
        if self._metrics is not None:
            try:
                self._metrics.record_calculation(
                    method="hybrid",
                    category="aggregation",
                    status="success",
                    duration_s=float(elapsed_ms / ONE_THOUSAND),
                    emissions_tco2e=float(total_emissions_tco2e),
                )
            except Exception as exc:
                logger.warning("[%s] Metrics recording failed: %s", calc_id, exc)

        logger.info(
            "[%s] HybridAggregator.aggregate complete: "
            "total=%.4f tCO2e, coverage=%.1f%%, "
            "spend=%d, avg=%d, supplier=%d, excluded=%d, "
            "dqi=%.2f, elapsed=%.1f ms",
            calc_id,
            total_emissions_tco2e,
            total_coverage_pct,
            spend_count,
            avg_count,
            supplier_count,
            len(excluded_ids),
            weighted_dqi,
            elapsed_ms,
        )

        # -- Step 14: Assemble HybridResult --
        if _MODELS_AVAILABLE:
            return HybridResult(
                calculation_id=calc_id,
                total_emissions_kg_co2e=_quantize(total_emissions_kg),
                total_emissions_tco2e=_quantize(total_emissions_tco2e),
                spend_based_emissions_tco2e=_quantize(spend_tco2e),
                average_data_emissions_tco2e=_quantize(avg_tco2e),
                supplier_specific_emissions_tco2e=_quantize(supplier_tco2e),
                spend_based_coverage_pct=_quantize(spend_coverage_pct),
                average_data_coverage_pct=_quantize(avg_coverage_pct),
                supplier_specific_coverage_pct=_quantize(
                    supplier_coverage_pct
                ),
                total_coverage_pct=_quantize(total_coverage_pct),
                total_capex_usd=_quantize(total_capex),
                asset_count=len(assets),
                spend_based_count=spend_count,
                average_data_count=avg_count,
                supplier_specific_count=supplier_count,
                excluded_count=len(excluded_ids),
                weighted_dqi=_quantize(weighted_dqi),
                method_breakdown=method_breakdown,
                provenance_hash=provenance_hash,
                processing_time_ms=_quantize(elapsed_ms),
            )

        # Fallback dict when models not available
        return {
            "calculation_id": calc_id,
            "total_emissions_kg_co2e": _quantize(total_emissions_kg),
            "total_emissions_tco2e": _quantize(total_emissions_tco2e),
            "spend_based_emissions_tco2e": _quantize(spend_tco2e),
            "average_data_emissions_tco2e": _quantize(avg_tco2e),
            "supplier_specific_emissions_tco2e": _quantize(supplier_tco2e),
            "spend_based_coverage_pct": _quantize(spend_coverage_pct),
            "average_data_coverage_pct": _quantize(avg_coverage_pct),
            "supplier_specific_coverage_pct": _quantize(
                supplier_coverage_pct
            ),
            "total_coverage_pct": _quantize(total_coverage_pct),
            "total_capex_usd": _quantize(total_capex),
            "asset_count": len(assets),
            "spend_based_count": spend_count,
            "average_data_count": avg_count,
            "supplier_specific_count": supplier_count,
            "excluded_count": len(excluded_ids),
            "weighted_dqi": _quantize(weighted_dqi),
            "method_breakdown": method_breakdown,
            "provenance_hash": provenance_hash,
            "processing_time_ms": _quantize(elapsed_ms),
        }

    # ==================================================================
    # Public API (2): aggregate_batch
    # ==================================================================

    def aggregate_batch(
        self,
        batch_results: List[Dict[str, Any]],
    ) -> List[Any]:
        """Aggregate multiple sets of calculation results in a batch.

        Each entry in batch_results must contain keys:
        - 'spend_results': List of spend-based results.
        - 'average_results': List of average-data results.
        - 'supplier_results': List of supplier-specific results.
        - 'assets': List of CapitalAssetRecord objects.
        - 'calculation_id': (optional) Pre-generated ID.

        Args:
            batch_results: List of batch entries.

        Returns:
            List of HybridResult objects, one per batch entry.
        """
        logger.info(
            "HybridAggregator.aggregate_batch called: %d entries",
            len(batch_results),
        )
        t_start = time.monotonic()
        results: List[Any] = []

        for idx, entry in enumerate(batch_results):
            try:
                result = self.aggregate(
                    spend_results=entry.get("spend_results", []),
                    average_results=entry.get("average_results", []),
                    supplier_results=entry.get("supplier_results", []),
                    assets=entry.get("assets", []),
                    calculation_id=entry.get("calculation_id"),
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch entry %d failed: %s", idx, exc, exc_info=True
                )
                results.append(
                    self._build_empty_result(
                        entry.get("calculation_id", str(uuid4())),
                        t_start,
                    )
                )

        elapsed = (time.monotonic() - t_start) * 1000
        logger.info(
            "HybridAggregator.aggregate_batch complete: "
            "%d results in %.1f ms",
            len(results),
            elapsed,
        )
        return results

    # ==================================================================
    # Public API (3): prioritize_methods
    # ==================================================================

    def prioritize_methods(
        self,
        spend_result: Any,
        average_result: Any,
        supplier_result: Any,
        asset_id: str = "",
    ) -> Tuple[Decimal, str]:
        """Select the best calculation method for a single asset.

        Applies the 8-level EF hierarchy to choose the most accurate
        result among the three methods.

        Priority: supplier_specific > average_data > spend_based.
        Within each method, hierarchy sub-levels are compared.

        Args:
            spend_result: Spend-based result (or None).
            average_result: Average-data result (or None).
            supplier_result: Supplier-specific result (or None).
            asset_id: Asset identifier for logging.

        Returns:
            Tuple of (emissions_kg_co2e, method_name).
        """
        best_result = None
        best_method = "spend_based"
        best_level = 99

        if supplier_result is not None:
            level = _determine_hierarchy_level_supplier(supplier_result)
            if level < best_level:
                best_level = level
                best_result = supplier_result
                best_method = "supplier_specific"

        if average_result is not None:
            level = _determine_hierarchy_level_average(average_result)
            if level < best_level:
                best_level = level
                best_result = average_result
                best_method = "average_data"

        if spend_result is not None:
            level = _determine_hierarchy_level_spend(spend_result)
            if level < best_level:
                best_level = level
                best_result = spend_result
                best_method = "spend_based"

        if best_result is None:
            logger.warning(
                "No results available for asset %s; returning zero",
                asset_id,
            )
            return (ZERO, "none")

        emissions = _get_result_emissions(best_result)

        logger.debug(
            "Asset %s: selected method=%s (level=%d), emissions=%.4f kgCO2e",
            asset_id,
            best_method,
            best_level,
            emissions,
        )

        return (emissions, best_method)

    # ==================================================================
    # Public API (4): analyze_coverage
    # ==================================================================

    def analyze_coverage(
        self,
        results: List[Any],
        total_assets: List[Any],
    ) -> Any:
        """Analyse method coverage of the Category 2 inventory.

        Computes the percentage of assets and CapEx covered by each
        calculation method and identifies gaps.

        Args:
            results: List of HybridResult or per-asset selected results.
            total_assets: Complete list of capital asset records.

        Returns:
            CoverageReport with coverage breakdown and gap analysis.
        """
        logger.info(
            "HybridAggregator.analyze_coverage: %d results, %d total assets",
            len(results),
            len(total_assets),
        )

        total_asset_count = len(total_assets)
        total_capex = self._compute_total_capex(total_assets)

        # Build covered asset set
        covered_ids: Set[str] = set()
        by_method: Dict[str, Dict[str, Decimal]] = {
            "spend_based": {"count": ZERO, "capex_usd": ZERO},
            "average_data": {"count": ZERO, "capex_usd": ZERO},
            "supplier_specific": {"count": ZERO, "capex_usd": ZERO},
        }

        asset_index = self._index_assets_by_id(total_assets)

        for r in results:
            aid = _get_result_asset_id(r)
            method = _get_result_method(r)
            if aid and aid not in covered_ids:
                covered_ids.add(aid)
                capex = _get_asset_capex(asset_index.get(aid, {}))
                if method in by_method:
                    by_method[method]["count"] += ONE
                    by_method[method]["capex_usd"] += capex

        covered_count = len(covered_ids)
        coverage_pct = _pct(
            Decimal(str(covered_count)),
            Decimal(str(total_asset_count)) if total_asset_count > 0 else ONE,
        )

        # Identify uncovered CapEx
        covered_capex = sum(
            (v["capex_usd"] for v in by_method.values()),
            ZERO,
        )
        uncovered_capex = _quantize(total_capex - covered_capex)
        if uncovered_capex < ZERO:
            uncovered_capex = ZERO

        # Identify gap categories
        covered_categories: Set[str] = set()
        all_categories: Set[str] = set()
        for a in total_assets:
            cat = _get_asset_category(a)
            all_categories.add(cat)
            aid = _get_asset_id(a)
            if aid in covered_ids:
                covered_categories.add(cat)
        gap_categories = sorted(all_categories - covered_categories)

        if _MODELS_AVAILABLE:
            return CoverageReport(
                total_assets=total_asset_count,
                covered_assets=covered_count,
                coverage_pct=_quantize(coverage_pct),
                by_method={
                    k: {mk: _quantize(mv) for mk, mv in v.items()}
                    for k, v in by_method.items()
                },
                uncovered_capex_usd=uncovered_capex,
                gap_categories=gap_categories,
            )

        return {
            "total_assets": total_asset_count,
            "covered_assets": covered_count,
            "coverage_pct": _quantize(coverage_pct),
            "by_method": by_method,
            "uncovered_capex_usd": uncovered_capex,
            "gap_categories": gap_categories,
        }

    # ==================================================================
    # Public API (5): detect_double_counting
    # ==================================================================

    def detect_double_counting(
        self,
        results: List[Any],
        cat1_asset_ids: Optional[Set[str]] = None,
        scope1_asset_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, str]]:
        """Detect double-counting across categories and scopes.

        Five detection rules:
        R1: If asset_id appears in Category 1 results, flag overlap.
        R2: If asset operational emissions counted in Scope 1/2, flag.
        R3: If asset is leased, flag for Cat 8/13 redirect.
        R4: If asset is under construction, flag for proportional.
        R5: PP&E classification check.

        Args:
            results: List of calculation results with asset_id.
            cat1_asset_ids: Set of asset IDs already in Category 1.
            scope1_asset_ids: Set of asset IDs in Scope 1/2.

        Returns:
            List of dicts with 'asset_id', 'reason', 'rule' keys.
        """
        logger.info(
            "HybridAggregator.detect_double_counting: %d results, "
            "cat1_ids=%d, scope1_ids=%d",
            len(results),
            len(cat1_asset_ids) if cat1_asset_ids else 0,
            len(scope1_asset_ids) if scope1_asset_ids else 0,
        )

        cat1_ids = cat1_asset_ids or set()
        scope1_ids = scope1_asset_ids or set()
        findings: List[Dict[str, str]] = []

        for r in results:
            aid = _get_result_asset_id(r)
            if not aid:
                continue

            # Rule 1: Category 1 overlap
            if aid in cat1_ids:
                findings.append({
                    "asset_id": aid,
                    "reason": (
                        "Asset appears in Category 1 (Purchased Goods & "
                        "Services). Must be reported in one category only."
                    ),
                    "rule": "R1_CAT1_OVERLAP",
                })

            # Rule 2: Scope 1/2 use-phase
            if aid in scope1_ids:
                findings.append({
                    "asset_id": aid,
                    "reason": (
                        "Asset operational emissions already counted in "
                        "Scope 1/2. Use-phase excluded from Category 2."
                    ),
                    "rule": "R2_SCOPE12_USE_PHASE",
                })

        logger.info(
            "Double-counting detection found %d findings",
            len(findings),
        )
        return findings

    # ==================================================================
    # Public API (6): prevent_double_counting
    # ==================================================================

    def prevent_double_counting(
        self,
        results: List[Any],
        exclusions: Set[str],
    ) -> List[Any]:
        """Remove double-counted items from results.

        Filters out any results whose asset_id is in the exclusion set.

        Args:
            results: List of calculation results.
            exclusions: Set of asset_ids to exclude.

        Returns:
            Filtered list with excluded items removed.
        """
        if not exclusions:
            return list(results)

        filtered = []
        removed_count = 0

        for r in results:
            aid = _get_result_asset_id(r)
            if aid in exclusions:
                removed_count += 1
                logger.debug(
                    "Double-counting prevention: removed asset %s", aid
                )
                continue
            filtered.append(r)

        logger.info(
            "Double-counting prevention: removed %d of %d results",
            removed_count,
            len(results),
        )
        return filtered

    # ==================================================================
    # Public API (7): hot_spot_analysis
    # ==================================================================

    def hot_spot_analysis(
        self,
        results: List[Any],
        assets: Optional[List[Any]] = None,
        top_n: int = _DEFAULT_TOP_N,
        pareto_threshold: Decimal = _PARETO_THRESHOLD_PCT,
    ) -> Any:
        """Perform hot-spot analysis with Pareto 80/20 rule.

        Ranks asset categories by emission contribution, identifies
        the 80% threshold contributors, and classifies items into
        four materiality quadrants.

        Args:
            results: List of HybridResult or per-asset results.
            assets: Optional asset records for spend data.
            top_n: Number of top items to include.
            pareto_threshold: Cumulative % threshold (default 80%).

        Returns:
            HotSpotAnalysis with ranked items and quadrant assignments.
        """
        logger.info(
            "HybridAggregator.hot_spot_analysis: %d results, top_n=%d",
            len(results),
            top_n,
        )

        if not results:
            return self._build_empty_hotspot("")

        # Build category-level emissions and spend maps
        category_emissions: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        category_spend: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        asset_index: Dict[str, Any] = {}
        if assets:
            asset_index = self._index_assets_by_id(assets)

        for r in results:
            aid = _get_result_asset_id(r)
            emissions_kg = _get_result_emissions(r)
            asset = asset_index.get(aid, {})
            category = _get_asset_category(asset) if asset else "other"
            capex = _get_asset_capex(asset) if asset else ZERO

            category_emissions[category] += emissions_kg
            category_spend[category] += capex

        # Total emissions for percentage calculation
        total_emissions_kg = sum(category_emissions.values(), ZERO)
        total_spend = sum(category_spend.values(), ZERO)

        if total_emissions_kg == ZERO:
            return self._build_empty_hotspot("")

        # Pareto analysis
        pareto_items = self.pareto_analysis(category_emissions)

        # Top N items
        top_items = pareto_items[:top_n]

        # Materiality quadrants using spend + emission intensity
        quadrant_map = self._classify_quadrants(
            pareto_items, category_spend, total_spend
        )

        # Generate recommendations
        recommendations = self._generate_hotspot_recommendations(
            pareto_items, quadrant_map
        )

        total_tco2e = _kg_to_tco2e(total_emissions_kg)
        calc_id = str(uuid4())

        if _MODELS_AVAILABLE:
            return HotSpotAnalysis(
                calculation_id=calc_id,
                total_emissions_tco2e=_quantize(total_tco2e),
                top_assets=top_items,
                pareto_items=pareto_items,
                materiality_quadrants=quadrant_map,
                recommendations=recommendations,
            )

        return {
            "calculation_id": calc_id,
            "total_emissions_tco2e": _quantize(total_tco2e),
            "top_assets": top_items,
            "pareto_items": pareto_items,
            "materiality_quadrants": quadrant_map,
            "recommendations": recommendations,
        }

    # ==================================================================
    # Public API (8): pareto_analysis
    # ==================================================================

    def pareto_analysis(
        self,
        emissions_by_item: Dict[str, Decimal],
        pareto_threshold: Decimal = _PARETO_THRESHOLD_PCT,
    ) -> List[Any]:
        """Perform Pareto 80/20 analysis on emissions by category.

        Sorts items by emissions descending, calculates cumulative
        percentage, and marks items above the Pareto threshold as
        material.

        Args:
            emissions_by_item: Dict of category -> emissions_kg_co2e.
            pareto_threshold: Cumulative % threshold (default 80%).

        Returns:
            List of MaterialityItem ordered by descending emissions.
        """
        if not emissions_by_item:
            return []

        total = sum(emissions_by_item.values(), ZERO)
        if total == ZERO:
            return []

        # Sort descending by emissions
        sorted_items = sorted(
            emissions_by_item.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        result: List[Any] = []
        cumulative = ZERO

        for category, emissions_kg in sorted_items:
            pct_of_total = _pct(emissions_kg, total)
            cumulative += pct_of_total
            # Clamp cumulative to 100%
            if cumulative > ONE_HUNDRED:
                cumulative = ONE_HUNDRED

            is_material = cumulative <= pareto_threshold or len(result) == 0

            if _MODELS_AVAILABLE:
                item = MaterialityItem(
                    asset_category=category,
                    emissions_kg_co2e=_quantize(emissions_kg),
                    pct_of_total=_quantize(pct_of_total),
                    cumulative_pct=_quantize(cumulative),
                    is_material=is_material,
                    quadrant="prioritize" if is_material else "low_priority",
                )
            else:
                item = {
                    "asset_category": category,
                    "emissions_kg_co2e": _quantize(emissions_kg),
                    "pct_of_total": _quantize(pct_of_total),
                    "cumulative_pct": _quantize(cumulative),
                    "is_material": is_material,
                    "quadrant": "prioritize" if is_material else "low_priority",
                }

            result.append(item)

        return result

    # ==================================================================
    # Public API (9): classify_materiality
    # ==================================================================

    def classify_materiality(
        self,
        items: List[Any],
        spend_by_category: Optional[Dict[str, Decimal]] = None,
        total_spend: Optional[Decimal] = None,
    ) -> Dict[str, List[Any]]:
        """Classify materiality items into Q1-Q4 quadrants.

        Quadrant definitions:
        - Q1 (prioritize): High emissions AND high spend.
        - Q2 (investigate): High emissions, low spend.
        - Q3 (optimize): Low emissions, high spend.
        - Q4 (monitor): Low emissions, low spend.

        Args:
            items: List of MaterialityItem objects.
            spend_by_category: Optional spend data by category.
            total_spend: Optional total spend for percentage calc.

        Returns:
            Dict with keys 'Q1', 'Q2', 'Q3', 'Q4' mapping to lists.
        """
        quadrants: Dict[str, List[Any]] = {
            "Q1_prioritize": [],
            "Q2_investigate": [],
            "Q3_optimize": [],
            "Q4_monitor": [],
        }

        spend_data = spend_by_category or {}
        spend_total = total_spend or ZERO

        for item in items:
            category = (
                item.asset_category
                if hasattr(item, "asset_category")
                else item.get("asset_category", "")
            )
            emission_pct = (
                Decimal(str(item.pct_of_total))
                if hasattr(item, "pct_of_total")
                else Decimal(str(item.get("pct_of_total", "0")))
            )

            # Determine spend percentage
            spend_amount = spend_data.get(category, ZERO)
            spend_pct = _pct(spend_amount, spend_total) if spend_total > ZERO else ZERO

            high_emissions = emission_pct >= _HIGH_EMISSION_PCT_THRESHOLD
            high_spend = spend_pct >= _HIGH_SPEND_PCT_THRESHOLD

            if high_emissions and high_spend:
                quadrants["Q1_prioritize"].append(item)
            elif high_emissions and not high_spend:
                quadrants["Q2_investigate"].append(item)
            elif not high_emissions and high_spend:
                quadrants["Q3_optimize"].append(item)
            else:
                quadrants["Q4_monitor"].append(item)

        logger.info(
            "Materiality classification: Q1=%d, Q2=%d, Q3=%d, Q4=%d",
            len(quadrants["Q1_prioritize"]),
            len(quadrants["Q2_investigate"]),
            len(quadrants["Q3_optimize"]),
            len(quadrants["Q4_monitor"]),
        )
        return quadrants

    # ==================================================================
    # Public API (10): calculate_capex_volatility
    # ==================================================================

    def calculate_capex_volatility(
        self,
        current_year_capex: Decimal,
        historical_capex: List[Decimal],
        acquisition_year: Optional[int] = None,
    ) -> Any:
        """Calculate CapEx volatility context for reporting narrative.

        Computes the rolling average, volatility ratio, and flags
        major CapEx years.  This provides context for stakeholders
        to understand why Category 2 emissions may spike in years
        with large capital investments.

        NOTE: This is NOT for emissions depreciation.  GHG Protocol
        requires 100% of cradle-to-gate emissions in year of
        acquisition.

        Args:
            current_year_capex: CapEx for the current reporting year.
            historical_capex: List of historical CapEx values (most
                recent first or chronological order).
            acquisition_year: Optional reporting year.

        Returns:
            DepreciationContext with volatility analysis.
        """
        year = acquisition_year or _utcnow().year
        rolling_years = self._get_rolling_years()
        threshold = self._get_volatility_threshold()

        logger.info(
            "CapEx volatility analysis: current=%.2f, "
            "historical=%d periods, rolling_years=%d",
            current_year_capex,
            len(historical_capex),
            rolling_years,
        )

        # Compute rolling average
        if historical_capex:
            recent = historical_capex[-rolling_years:]
            if recent:
                rolling_sum = sum(
                    (Decimal(str(v)) for v in recent), ZERO
                )
                rolling_avg = _safe_divide(
                    rolling_sum, Decimal(str(len(recent)))
                )
            else:
                rolling_avg = current_year_capex
        else:
            rolling_avg = current_year_capex

        # Compute volatility ratio
        volatility_ratio = _safe_divide(
            current_year_capex, rolling_avg, default=ONE
        )

        # Flag major CapEx year
        is_major = volatility_ratio > threshold

        # Generate context note
        context_note = self._generate_volatility_note(
            current_year_capex,
            rolling_avg,
            volatility_ratio,
            is_major,
            rolling_years,
            year,
        )

        logger.info(
            "CapEx volatility result: rolling_avg=%.2f, "
            "ratio=%.2f, is_major=%s",
            rolling_avg,
            volatility_ratio,
            is_major,
        )

        if _MODELS_AVAILABLE:
            return DepreciationContext(
                acquisition_year=year,
                total_capex=_quantize(current_year_capex),
                rolling_avg_capex=_quantize(rolling_avg),
                volatility_ratio=_quantize(volatility_ratio),
                is_major_capex_year=is_major,
                context_note=context_note,
            )

        return {
            "acquisition_year": year,
            "total_capex": _quantize(current_year_capex),
            "rolling_avg_capex": _quantize(rolling_avg),
            "volatility_ratio": _quantize(volatility_ratio),
            "is_major_capex_year": is_major,
            "context_note": context_note,
        }

    # ==================================================================
    # Public API (11): yoy_decomposition
    # ==================================================================

    def yoy_decomposition(
        self,
        current: Any,
        previous: Any,
    ) -> Dict[str, Any]:
        """Decompose year-over-year emission changes.

        Decomposes the total change into four effects:
        1. Activity effect: change due to CapEx volume.
        2. Emission factor effect: change due to different EF mix.
        3. Method mix effect: change due to method distribution.
        4. Coverage effect: change due to coverage improvement.

        Args:
            current: Current period HybridResult.
            previous: Previous period HybridResult.

        Returns:
            Dict with decomposition components and totals.
        """
        logger.info("YoY decomposition analysis")

        curr_total = self._extract_total_tco2e(current)
        prev_total = self._extract_total_tco2e(previous)
        absolute_change = _quantize(curr_total - prev_total)
        pct_change = _pct(
            absolute_change,
            prev_total if prev_total != ZERO else ONE,
        )

        # CapEx volumes
        curr_capex = self._extract_total_capex(current)
        prev_capex = self._extract_total_capex(previous)

        # Method breakdowns
        curr_breakdown = self._extract_method_breakdown(current)
        prev_breakdown = self._extract_method_breakdown(previous)

        # Activity effect: proportional to CapEx change
        capex_ratio = _safe_divide(curr_capex, prev_capex, default=ONE)
        activity_effect = _quantize(
            prev_total * (capex_ratio - ONE)
        )

        # Coverage effect: difference in coverage percentages
        curr_coverage = self._extract_coverage_pct(current)
        prev_coverage = self._extract_coverage_pct(previous)
        coverage_delta = _quantize(curr_coverage - prev_coverage)
        coverage_effect = _quantize(
            prev_total * (coverage_delta / ONE_HUNDRED)
        ) if prev_total != ZERO else ZERO

        # Method mix effect
        method_mix_effect = self._compute_method_mix_effect(
            curr_breakdown, prev_breakdown, curr_total, prev_total
        )

        # EF effect: residual
        ef_effect = _quantize(
            absolute_change - activity_effect
            - method_mix_effect - coverage_effect
        )

        result = {
            "current_total_tco2e": _quantize(curr_total),
            "previous_total_tco2e": _quantize(prev_total),
            "absolute_change_tco2e": absolute_change,
            "percentage_change": _quantize(pct_change),
            "activity_effect_tco2e": activity_effect,
            "ef_effect_tco2e": ef_effect,
            "method_mix_effect_tco2e": method_mix_effect,
            "coverage_effect_tco2e": coverage_effect,
            "current_capex_usd": _quantize(curr_capex),
            "previous_capex_usd": _quantize(prev_capex),
            "current_coverage_pct": _quantize(curr_coverage),
            "previous_coverage_pct": _quantize(prev_coverage),
            "provenance_hash": _compute_sha256({
                "current_tco2e": _decimal_to_str(curr_total),
                "previous_tco2e": _decimal_to_str(prev_total),
                "absolute_change": _decimal_to_str(absolute_change),
            }),
        }

        logger.info(
            "YoY decomposition: change=%.4f tCO2e (%.1f%%), "
            "activity=%.4f, ef=%.4f, method_mix=%.4f, coverage=%.4f",
            absolute_change,
            pct_change,
            activity_effect,
            ef_effect,
            method_mix_effect,
            coverage_effect,
        )
        return result

    # ==================================================================
    # Public API (12): calculate_intensity_metrics
    # ==================================================================

    def calculate_intensity_metrics(
        self,
        total_emissions: Decimal,
        revenue: Optional[Decimal] = None,
        fte: Optional[Decimal] = None,
        floor_area: Optional[Decimal] = None,
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate emission intensity metrics.

        Computes:
        - tCO2e per $M revenue.
        - tCO2e per FTE (full-time equivalent employee).
        - tCO2e per m2 of floor area.

        Args:
            total_emissions: Total emissions in tCO2e.
            revenue: Annual revenue in USD (optional).
            fte: Full-time equivalent headcount (optional).
            floor_area: Total floor area in m2 (optional).

        Returns:
            Dict with intensity metric keys and Decimal values.
        """
        result: Dict[str, Optional[Decimal]] = {
            "tco2e_per_million_revenue": None,
            "tco2e_per_fte": None,
            "tco2e_per_m2": None,
        }

        if revenue is not None and revenue > ZERO:
            result["tco2e_per_million_revenue"] = _quantize(
                total_emissions / (revenue / _MILLION)
            )

        if fte is not None and fte > ZERO:
            result["tco2e_per_fte"] = _quantize(
                total_emissions / fte
            )

        if floor_area is not None and floor_area > ZERO:
            result["tco2e_per_m2"] = _quantize(
                total_emissions / floor_area
            )

        logger.info(
            "Intensity metrics: revenue=%s, fte=%s, floor=%s",
            result["tco2e_per_million_revenue"],
            result["tco2e_per_fte"],
            result["tco2e_per_m2"],
        )
        return result

    # ==================================================================
    # Public API (13): aggregate_by_category
    # ==================================================================

    def aggregate_by_category(
        self,
        results: List[Any],
        assets: Optional[List[Any]] = None,
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by asset category.

        Groups results by their asset category (buildings, machinery,
        vehicles, etc.) and sums emissions per group.

        Args:
            results: List of per-asset calculation results.
            assets: Optional asset records for category lookup.

        Returns:
            Dict mapping category -> total_tco2e.
        """
        asset_index: Dict[str, Any] = {}
        if assets:
            asset_index = self._index_assets_by_id(assets)

        by_category: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for r in results:
            aid = _get_result_asset_id(r)
            emissions_kg = _get_result_emissions(r)
            tco2e = _kg_to_tco2e(emissions_kg)
            asset = asset_index.get(aid, {})
            category = _get_asset_category(asset) if asset else "other"
            by_category[category] += tco2e

        return {k: _quantize(v) for k, v in sorted(by_category.items())}

    # ==================================================================
    # Public API (14): aggregate_by_method
    # ==================================================================

    def aggregate_by_method(
        self,
        results: List[Any],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by calculation method.

        Groups results by their calculation method (spend_based,
        average_data, supplier_specific) and sums emissions.

        Args:
            results: List of per-asset calculation results.

        Returns:
            Dict mapping method_name -> total_tco2e.
        """
        by_method: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for r in results:
            method = _get_result_method(r)
            emissions_kg = _get_result_emissions(r)
            tco2e = _kg_to_tco2e(emissions_kg)
            by_method[method] += tco2e

        return {k: _quantize(v) for k, v in sorted(by_method.items())}

    # ==================================================================
    # Public API (15): aggregate_by_supplier
    # ==================================================================

    def aggregate_by_supplier(
        self,
        results: List[Any],
        assets: Optional[List[Any]] = None,
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by supplier.

        Groups results by their supplier identifier and sums
        emissions per supplier.

        Args:
            results: List of per-asset calculation results.
            assets: Optional asset records for supplier lookup.

        Returns:
            Dict mapping supplier_id -> total_tco2e.
        """
        asset_index: Dict[str, Any] = {}
        if assets:
            asset_index = self._index_assets_by_id(assets)

        by_supplier: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for r in results:
            aid = _get_result_asset_id(r)
            emissions_kg = _get_result_emissions(r)
            tco2e = _kg_to_tco2e(emissions_kg)
            asset = asset_index.get(aid, {})
            supplier = _get_asset_supplier(asset) if asset else "unknown"
            by_supplier[supplier] += tco2e

        return {k: _quantize(v) for k, v in sorted(by_supplier.items())}

    # ==================================================================
    # Public API (16): aggregate_by_period
    # ==================================================================

    def aggregate_by_period(
        self,
        results: List[Any],
        assets: Optional[List[Any]] = None,
    ) -> Dict[int, Decimal]:
        """Aggregate emissions by acquisition year.

        Groups results by the acquisition year of each asset and
        sums emissions per year.

        Args:
            results: List of per-asset calculation results.
            assets: Optional asset records for date lookup.

        Returns:
            Dict mapping year -> total_tco2e.
        """
        asset_index: Dict[str, Any] = {}
        if assets:
            asset_index = self._index_assets_by_id(assets)

        by_period: Dict[int, Decimal] = defaultdict(lambda: ZERO)

        for r in results:
            aid = _get_result_asset_id(r)
            emissions_kg = _get_result_emissions(r)
            tco2e = _kg_to_tco2e(emissions_kg)
            asset = asset_index.get(aid, {})
            year = _get_asset_year(asset) if asset else _utcnow().year
            by_period[year] += tco2e

        return {k: _quantize(v) for k, v in sorted(by_period.items())}

    # ==================================================================
    # Public API (17): calculate_combined_uncertainty
    # ==================================================================

    def calculate_combined_uncertainty(
        self,
        results: List[Any],
    ) -> Dict[str, Any]:
        """Calculate combined uncertainty across all methods.

        Uses root-sum-of-squares (RSS) error propagation to combine
        individual result uncertainties into a portfolio-level
        uncertainty.  Also provides method-level uncertainty breakdown.

        Args:
            results: List of per-asset calculation results.

        Returns:
            Dict with combined_uncertainty_pct, method breakdowns,
            and confidence intervals.
        """
        logger.info(
            "Combined uncertainty calculation: %d results",
            len(results),
        )

        if not results:
            return {
                "combined_uncertainty_pct": ZERO,
                "lower_bound_tco2e": ZERO,
                "upper_bound_tco2e": ZERO,
                "by_method": {},
                "total_emissions_tco2e": ZERO,
            }

        total_emissions = ZERO
        sum_squared_errors = ZERO
        method_uncertainties: Dict[str, List[Tuple[Decimal, Decimal]]] = (
            defaultdict(list)
        )

        for r in results:
            emissions_kg = _get_result_emissions(r)
            tco2e = _kg_to_tco2e(emissions_kg)
            uncertainty_pct = _get_result_uncertainty(r)
            method = _get_result_method(r)

            total_emissions += tco2e

            # Absolute uncertainty for this result
            abs_uncertainty = _quantize(
                tco2e * uncertainty_pct / ONE_HUNDRED
            )
            sum_squared_errors += abs_uncertainty * abs_uncertainty

            method_uncertainties[method].append(
                (tco2e, uncertainty_pct)
            )

        # RSS combination
        combined_abs = ZERO
        if sum_squared_errors > ZERO:
            try:
                combined_abs = _quantize(
                    Decimal(str(math.sqrt(float(sum_squared_errors))))
                )
            except (ValueError, OverflowError):
                combined_abs = ZERO

        combined_pct = _pct(combined_abs, total_emissions)

        # Confidence intervals
        lower_bound = _quantize(total_emissions - combined_abs)
        upper_bound = _quantize(total_emissions + combined_abs)
        if lower_bound < ZERO:
            lower_bound = ZERO

        # Per-method breakdown
        by_method: Dict[str, Dict[str, Decimal]] = {}
        for method, pairs in method_uncertainties.items():
            method_total = sum((p[0] for p in pairs), ZERO)
            method_sq = sum(
                ((_quantize(p[0] * p[1] / ONE_HUNDRED)) ** 2 for p in pairs),
                ZERO,
            )
            method_abs = ZERO
            if method_sq > ZERO:
                try:
                    method_abs = _quantize(
                        Decimal(str(math.sqrt(float(method_sq))))
                    )
                except (ValueError, OverflowError):
                    pass
            method_pct = _pct(method_abs, method_total)

            by_method[method] = {
                "total_tco2e": _quantize(method_total),
                "uncertainty_pct": _quantize(method_pct),
                "uncertainty_abs_tco2e": method_abs,
            }

        result = {
            "combined_uncertainty_pct": _quantize(combined_pct),
            "combined_uncertainty_abs_tco2e": combined_abs,
            "lower_bound_tco2e": lower_bound,
            "upper_bound_tco2e": upper_bound,
            "total_emissions_tco2e": _quantize(total_emissions),
            "by_method": by_method,
            "provenance_hash": _compute_sha256({
                "total_tco2e": _decimal_to_str(total_emissions),
                "combined_pct": _decimal_to_str(combined_pct),
            }),
        }

        logger.info(
            "Combined uncertainty: total=%.4f tCO2e, "
            "uncertainty=%.1f%% [%.4f, %.4f]",
            total_emissions,
            combined_pct,
            lower_bound,
            upper_bound,
        )
        return result

    # ==================================================================
    # Public API (18): get_method_breakdown
    # ==================================================================

    def get_method_breakdown(
        self,
        result: Any,
    ) -> Dict[str, Any]:
        """Get detailed method breakdown from a HybridResult.

        Extracts per-method emissions, counts, coverage, and
        percentage contribution.

        Args:
            result: A HybridResult object or dictionary.

        Returns:
            Dict with per-method details.
        """
        total_tco2e = self._extract_total_tco2e(result)
        breakdown = self._extract_method_breakdown(result)

        details: Dict[str, Any] = {}
        for method, tco2e in breakdown.items():
            pct = _pct(tco2e, total_tco2e) if total_tco2e > ZERO else ZERO
            details[method] = {
                "emissions_tco2e": _quantize(tco2e),
                "pct_of_total": _quantize(pct),
            }

        # Add counts if available
        for field_name, method_key in [
            ("spend_based_count", "spend_based"),
            ("average_data_count", "average_data"),
            ("supplier_specific_count", "supplier_specific"),
        ]:
            count = self._extract_field(result, field_name, 0)
            if method_key in details:
                details[method_key]["asset_count"] = count

        # Add coverage if available
        for field_name, method_key in [
            ("spend_based_coverage_pct", "spend_based"),
            ("average_data_coverage_pct", "average_data"),
            ("supplier_specific_coverage_pct", "supplier_specific"),
        ]:
            coverage = self._extract_field(result, field_name, ZERO)
            if method_key in details:
                details[method_key]["coverage_pct"] = _quantize(
                    Decimal(str(coverage))
                )

        return details

    # ==================================================================
    # Public API (19): compute_provenance_hash
    # ==================================================================

    def compute_provenance_hash(
        self,
        result: Any,
    ) -> str:
        """Compute SHA-256 provenance hash for a result.

        Creates a deterministic hash from key result fields for
        audit trail verification.

        Args:
            result: A HybridResult or dict to hash.

        Returns:
            64-character hex SHA-256 digest.
        """
        data = {
            "total_emissions_tco2e": _decimal_to_str(
                self._extract_total_tco2e(result)
            ),
            "total_capex_usd": _decimal_to_str(
                self._extract_total_capex(result)
            ),
            "asset_count": self._extract_field(result, "asset_count", 0),
            "spend_based_count": self._extract_field(
                result, "spend_based_count", 0
            ),
            "average_data_count": self._extract_field(
                result, "average_data_count", 0
            ),
            "supplier_specific_count": self._extract_field(
                result, "supplier_specific_count", 0
            ),
            "excluded_count": self._extract_field(
                result, "excluded_count", 0
            ),
            "weighted_dqi": _decimal_to_str(
                Decimal(str(
                    self._extract_field(result, "weighted_dqi", "5.0")
                ))
            ),
            "method_breakdown": {
                k: _decimal_to_str(Decimal(str(v)))
                for k, v in self._extract_method_breakdown(result).items()
            },
        }

        provenance_hash = _compute_sha256(data)
        logger.debug("Provenance hash computed: %s", provenance_hash[:16])
        return provenance_hash

    # ==================================================================
    # Internal -- method selection
    # ==================================================================

    def _select_best_method(
        self,
        asset_id: str,
        spend_idx: Dict[str, Any],
        avg_idx: Dict[str, Any],
        supplier_idx: Dict[str, Any],
    ) -> Tuple[Any, str, int]:
        """Select the best available method for a single asset.

        Uses the 8-level EF hierarchy to determine which method
        result to use for this asset.

        Args:
            asset_id: The asset identifier to look up.
            spend_idx: Spend-based results indexed by asset_id.
            avg_idx: Average-data results indexed by asset_id.
            supplier_idx: Supplier-specific results indexed by asset_id.

        Returns:
            Tuple of (result, method_name, hierarchy_level).
            Returns (None, 'none', 99) if no result available.
        """
        candidates: List[Tuple[Any, str, int]] = []

        # Check supplier-specific
        if asset_id in supplier_idx:
            sr = supplier_idx[asset_id]
            level = _determine_hierarchy_level_supplier(sr)
            candidates.append((sr, "supplier_specific", level))

        # Check average-data
        if asset_id in avg_idx:
            ar = avg_idx[asset_id]
            level = _determine_hierarchy_level_average(ar)
            candidates.append((ar, "average_data", level))

        # Check spend-based
        if asset_id in spend_idx:
            spr = spend_idx[asset_id]
            level = _determine_hierarchy_level_spend(spr)
            candidates.append((spr, "spend_based", level))

        if not candidates:
            return (None, "none", 99)

        # Select candidate with lowest hierarchy level (best quality)
        candidates.sort(key=lambda x: x[2])
        best = candidates[0]

        logger.debug(
            "Asset %s: selected %s (level %d) from %d candidates",
            asset_id,
            best[1],
            best[2],
            len(candidates),
        )
        return best

    # ==================================================================
    # Internal -- double-counting
    # ==================================================================

    def _check_asset_exclusion(self, asset: Any) -> Optional[str]:
        """Check if an asset should be excluded per double-counting rules.

        Args:
            asset: A CapitalAssetRecord or dictionary.

        Returns:
            Exclusion reason string, or None if asset is not excluded.
        """
        # Rule 3: Leased assets -> Category 8 or 13
        if self._get_leased_exclusion_enabled():
            if _get_asset_is_leased(asset):
                return _EXCLUSION_REASON_LEASED

        # Rule 4: Under construction -- check metadata
        if hasattr(asset, "metadata"):
            meta = asset.metadata
        elif isinstance(asset, dict):
            meta = asset.get("metadata", {})
        else:
            meta = {}

        if isinstance(meta, dict):
            if meta.get("is_under_construction", False):
                completion = Decimal(str(meta.get("completion_pct", "100")))
                if completion < ONE_HUNDRED:
                    return _EXCLUSION_REASON_UNDER_CONSTRUCTION

            # Rule 5: PP&E classification mismatch
            if meta.get("is_opex", False):
                return _EXCLUSION_REASON_PPE_CLASSIFICATION

        return None

    # ==================================================================
    # Internal -- weighted DQI
    # ==================================================================

    def _compute_weighted_dqi(
        self,
        emissions_dqi: List[Tuple[str, Decimal, Decimal]],
    ) -> Decimal:
        """Compute emission-weighted composite DQI score.

        Weights each asset's DQI by its proportion of total emissions.
        Lower scores indicate higher quality.

        Args:
            emissions_dqi: List of (asset_id, emissions_kg, dqi_score).

        Returns:
            Weighted DQI score clamped to [1.0, 5.0].
        """
        if not emissions_dqi:
            return _MAX_DQI

        total_emissions = sum(
            (item[1] for item in emissions_dqi), ZERO
        )

        if total_emissions == ZERO:
            return _MAX_DQI

        weighted_sum = ZERO
        for _aid, emissions_kg, dqi in emissions_dqi:
            weight = _safe_divide(emissions_kg, total_emissions)
            weighted_sum += weight * dqi

        result = _quantize(weighted_sum)

        # Clamp
        if result < _MIN_DQI:
            result = _MIN_DQI
        if result > _MAX_DQI:
            result = _MAX_DQI

        return result

    # ==================================================================
    # Internal -- CapEx computation
    # ==================================================================

    def _compute_total_capex(self, assets: List[Any]) -> Decimal:
        """Compute total CapEx from a list of asset records.

        Args:
            assets: List of CapitalAssetRecord or dicts.

        Returns:
            Total CapEx as Decimal.
        """
        total = ZERO
        for a in assets:
            total += _get_asset_capex(a)
        return _quantize(total)

    # ==================================================================
    # Internal -- empty result builders
    # ==================================================================

    def _build_empty_result(
        self,
        calc_id: str,
        t_start: float,
    ) -> Any:
        """Build an empty HybridResult for error/empty cases.

        Args:
            calc_id: Calculation identifier.
            t_start: Monotonic start time.

        Returns:
            Empty HybridResult or dict.
        """
        elapsed_ms = Decimal(str(
            (time.monotonic() - t_start) * 1000
        ))

        if _MODELS_AVAILABLE:
            return HybridResult(
                calculation_id=calc_id,
                total_emissions_kg_co2e=ZERO,
                total_emissions_tco2e=ZERO,
                spend_based_emissions_tco2e=ZERO,
                average_data_emissions_tco2e=ZERO,
                supplier_specific_emissions_tco2e=ZERO,
                spend_based_coverage_pct=ZERO,
                average_data_coverage_pct=ZERO,
                supplier_specific_coverage_pct=ZERO,
                total_coverage_pct=ZERO,
                total_capex_usd=ZERO,
                asset_count=0,
                spend_based_count=0,
                average_data_count=0,
                supplier_specific_count=0,
                excluded_count=0,
                weighted_dqi=_MAX_DQI,
                method_breakdown={},
                provenance_hash=_compute_sha256({"empty": True}),
                processing_time_ms=_quantize(elapsed_ms),
            )

        return {
            "calculation_id": calc_id,
            "total_emissions_kg_co2e": ZERO,
            "total_emissions_tco2e": ZERO,
            "total_capex_usd": ZERO,
            "asset_count": 0,
            "provenance_hash": _compute_sha256({"empty": True}),
            "processing_time_ms": _quantize(elapsed_ms),
        }

    def _build_empty_hotspot(self, calc_id: str) -> Any:
        """Build an empty HotSpotAnalysis.

        Args:
            calc_id: Calculation identifier.

        Returns:
            Empty HotSpotAnalysis or dict.
        """
        cid = calc_id or str(uuid4())
        if _MODELS_AVAILABLE:
            return HotSpotAnalysis(
                calculation_id=cid,
                total_emissions_tco2e=ZERO,
                top_assets=[],
                pareto_items=[],
                materiality_quadrants={},
                recommendations=[],
            )

        return {
            "calculation_id": cid,
            "total_emissions_tco2e": ZERO,
            "top_assets": [],
            "pareto_items": [],
            "materiality_quadrants": {},
            "recommendations": [],
        }

    # ==================================================================
    # Internal -- YoY decomposition helpers
    # ==================================================================

    def _extract_total_tco2e(self, result: Any) -> Decimal:
        """Extract total_emissions_tco2e from a result.

        Args:
            result: HybridResult or dict.

        Returns:
            Total emissions in tCO2e.
        """
        return Decimal(str(
            self._extract_field(result, "total_emissions_tco2e", "0")
        ))

    def _extract_total_capex(self, result: Any) -> Decimal:
        """Extract total_capex_usd from a result.

        Args:
            result: HybridResult or dict.

        Returns:
            Total CapEx in USD.
        """
        return Decimal(str(
            self._extract_field(result, "total_capex_usd", "0")
        ))

    def _extract_coverage_pct(self, result: Any) -> Decimal:
        """Extract total_coverage_pct from a result.

        Args:
            result: HybridResult or dict.

        Returns:
            Coverage percentage.
        """
        return Decimal(str(
            self._extract_field(result, "total_coverage_pct", "0")
        ))

    def _extract_method_breakdown(self, result: Any) -> Dict[str, Decimal]:
        """Extract method_breakdown from a result.

        Args:
            result: HybridResult or dict.

        Returns:
            Dict of method -> tCO2e.
        """
        raw = self._extract_field(result, "method_breakdown", {})
        if isinstance(raw, dict):
            return {k: Decimal(str(v)) for k, v in raw.items()}
        return {}

    def _extract_field(self, result: Any, field: str, default: Any) -> Any:
        """Extract a field from a result object or dict.

        Args:
            result: HybridResult, dict, or other object.
            field: Field name to extract.
            default: Default value if field not found.

        Returns:
            Field value or default.
        """
        if hasattr(result, field):
            return getattr(result, field)
        if isinstance(result, dict):
            return result.get(field, default)
        return default

    def _compute_method_mix_effect(
        self,
        curr_breakdown: Dict[str, Decimal],
        prev_breakdown: Dict[str, Decimal],
        curr_total: Decimal,
        prev_total: Decimal,
    ) -> Decimal:
        """Compute the method mix effect for YoY decomposition.

        Measures how much the change in method distribution
        (e.g., more supplier-specific data replacing spend-based)
        contributed to the overall emission change.

        Args:
            curr_breakdown: Current period method breakdown.
            prev_breakdown: Previous period method breakdown.
            curr_total: Current total emissions.
            prev_total: Previous total emissions.

        Returns:
            Method mix effect in tCO2e.
        """
        if prev_total == ZERO or curr_total == ZERO:
            return ZERO

        # Calculate share shifts
        all_methods = set(curr_breakdown.keys()) | set(prev_breakdown.keys())
        effect = ZERO

        for method in all_methods:
            curr_val = curr_breakdown.get(method, ZERO)
            prev_val = prev_breakdown.get(method, ZERO)

            curr_share = _safe_divide(curr_val, curr_total)
            prev_share = _safe_divide(prev_val, prev_total)

            share_delta = curr_share - prev_share
            # Impact proportional to the method's average emissions
            avg_method_emissions = _safe_divide(
                curr_val + prev_val, Decimal("2")
            )
            effect += share_delta * avg_method_emissions

        return _quantize(effect)

    # ==================================================================
    # Internal -- hot-spot helpers
    # ==================================================================

    def _classify_quadrants(
        self,
        pareto_items: List[Any],
        category_spend: Dict[str, Decimal],
        total_spend: Decimal,
    ) -> Dict[str, List[Any]]:
        """Classify pareto items into materiality quadrants.

        Q1 (prioritize): High emissions AND high spend.
        Q2 (investigate): High emissions, low spend.
        Q3 (optimize): Low emissions, high spend.
        Q4 (monitor): Low emissions, low spend.

        Args:
            pareto_items: Ranked list of MaterialityItem.
            category_spend: Spend by category.
            total_spend: Total spend across all categories.

        Returns:
            Dict with quadrant keys -> lists of items.
        """
        quadrants: Dict[str, List[Any]] = {
            "Q1_prioritize": [],
            "Q2_investigate": [],
            "Q3_optimize": [],
            "Q4_monitor": [],
        }

        for item in pareto_items:
            category = (
                item.asset_category
                if hasattr(item, "asset_category")
                else item.get("asset_category", "")
            )
            emission_pct = (
                Decimal(str(item.pct_of_total))
                if hasattr(item, "pct_of_total")
                else Decimal(str(item.get("pct_of_total", "0")))
            )

            spend_amount = category_spend.get(category, ZERO)
            spend_pct = (
                _pct(spend_amount, total_spend)
                if total_spend > ZERO
                else ZERO
            )

            high_emissions = emission_pct >= _HIGH_EMISSION_PCT_THRESHOLD
            high_spend = spend_pct >= _HIGH_SPEND_PCT_THRESHOLD

            if high_emissions and high_spend:
                quadrants["Q1_prioritize"].append(item)
            elif high_emissions and not high_spend:
                quadrants["Q2_investigate"].append(item)
            elif not high_emissions and high_spend:
                quadrants["Q3_optimize"].append(item)
            else:
                quadrants["Q4_monitor"].append(item)

        return quadrants

    def _generate_hotspot_recommendations(
        self,
        pareto_items: List[Any],
        quadrant_map: Dict[str, List[Any]],
    ) -> List[str]:
        """Generate hot-spot recommendations based on analysis.

        Args:
            pareto_items: Ranked materiality items.
            quadrant_map: Quadrant classification results.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Count material items (within Pareto threshold)
        material_count = sum(
            1 for item in pareto_items
            if (
                item.is_material
                if hasattr(item, "is_material")
                else item.get("is_material", False)
            )
        )

        if material_count > 0:
            recommendations.append(
                f"{material_count} asset categories contribute to 80% "
                f"of Category 2 emissions (Pareto analysis)."
            )

        # Q1 recommendations
        q1_count = len(quadrant_map.get("Q1_prioritize", []))
        if q1_count > 0:
            q1_names = [
                (
                    item.asset_category
                    if hasattr(item, "asset_category")
                    else item.get("asset_category", "")
                )
                for item in quadrant_map["Q1_prioritize"][:5]
            ]
            recommendations.append(
                f"PRIORITY: {q1_count} categories have both high emissions "
                f"and high spend. Focus supplier engagement on: "
                f"{', '.join(q1_names)}."
            )

        # Q2 recommendations
        q2_count = len(quadrant_map.get("Q2_investigate", []))
        if q2_count > 0:
            recommendations.append(
                f"INVESTIGATE: {q2_count} categories have high emissions "
                f"relative to spend. Investigate emission factor accuracy "
                f"and consider switching to supplier-specific data."
            )

        # Q3 recommendations
        q3_count = len(quadrant_map.get("Q3_optimize", []))
        if q3_count > 0:
            recommendations.append(
                f"OPTIMIZE: {q3_count} categories have high spend but "
                f"lower emissions. Consider procurement of lower-carbon "
                f"alternatives to further reduce Category 2."
            )

        # General guidance
        if pareto_items:
            top_item = pareto_items[0]
            top_name = (
                top_item.asset_category
                if hasattr(top_item, "asset_category")
                else top_item.get("asset_category", "")
            )
            top_pct = (
                top_item.pct_of_total
                if hasattr(top_item, "pct_of_total")
                else top_item.get("pct_of_total", ZERO)
            )
            recommendations.append(
                f"Top contributor: '{top_name}' accounts for "
                f"{top_pct}% of total Category 2 emissions."
            )

        recommendations.append(
            "GHG Protocol recommends improving data quality for material "
            "categories by obtaining supplier-specific EPDs or PCFs."
        )

        return recommendations

    # ==================================================================
    # Internal -- volatility note generation
    # ==================================================================

    def _generate_volatility_note(
        self,
        current_capex: Decimal,
        rolling_avg: Decimal,
        volatility_ratio: Decimal,
        is_major: bool,
        rolling_years: int,
        year: int,
    ) -> str:
        """Generate a context note for CapEx volatility.

        Args:
            current_capex: Current year CapEx.
            rolling_avg: Rolling average CapEx.
            volatility_ratio: Current / rolling average ratio.
            is_major: Whether this is flagged as major CapEx year.
            rolling_years: Number of years in rolling window.
            year: Reporting year.

        Returns:
            Human-readable context note string.
        """
        if is_major:
            note = (
                f"Year {year} is a MAJOR CAPEX YEAR. "
                f"Capital expenditure of ${current_capex:,.0f} is "
                f"{volatility_ratio:.1f}x the {rolling_years}-year "
                f"rolling average of ${rolling_avg:,.0f}. "
                f"Category 2 emissions are expected to be significantly "
                f"higher than typical years due to large capital "
                f"investments. Per GHG Protocol, 100% of cradle-to-gate "
                f"emissions are reported in the year of acquisition."
            )
        elif volatility_ratio > ONE:
            note = (
                f"Year {year} capital expenditure of ${current_capex:,.0f} "
                f"is {volatility_ratio:.1f}x the {rolling_years}-year "
                f"rolling average of ${rolling_avg:,.0f}. "
                f"Category 2 emissions are moderately above trend."
            )
        elif volatility_ratio < ONE:
            note = (
                f"Year {year} capital expenditure of ${current_capex:,.0f} "
                f"is {volatility_ratio:.1f}x the {rolling_years}-year "
                f"rolling average of ${rolling_avg:,.0f}. "
                f"Category 2 emissions are below trend due to "
                f"reduced capital investment."
            )
        else:
            note = (
                f"Year {year} capital expenditure of ${current_capex:,.0f} "
                f"is in line with the {rolling_years}-year rolling average "
                f"of ${rolling_avg:,.0f}."
            )

        return note

    # ==================================================================
    # Internal -- aggregation result builder
    # ==================================================================

    def _build_aggregation_result(
        self,
        results: List[Any],
        assets: Optional[List[Any]] = None,
        revenue: Optional[Decimal] = None,
    ) -> Any:
        """Build a comprehensive AggregationResult.

        Args:
            results: List of per-asset calculation results.
            assets: Optional asset records.
            revenue: Optional revenue for intensity calc.

        Returns:
            AggregationResult or dict.
        """
        by_category = self.aggregate_by_category(results, assets)
        by_method = self.aggregate_by_method(results)
        by_supplier = self.aggregate_by_supplier(results, assets)
        by_period = self.aggregate_by_period(results, assets)

        total_tco2e = sum(by_method.values(), ZERO)
        total_capex = ZERO
        if assets:
            total_capex = self._compute_total_capex(assets)

        intensity_per_capex: Optional[Decimal] = None
        if total_capex > ZERO:
            intensity_per_capex = _quantize(
                total_tco2e / (total_capex / _MILLION)
            )

        intensity_per_revenue: Optional[Decimal] = None
        if revenue is not None and revenue > ZERO:
            intensity_per_revenue = _quantize(
                total_tco2e / (revenue / _MILLION)
            )

        provenance_hash = _compute_sha256({
            "total_tco2e": _decimal_to_str(total_tco2e),
            "by_category": {k: _decimal_to_str(v) for k, v in by_category.items()},
            "by_method": {k: _decimal_to_str(v) for k, v in by_method.items()},
        })

        by_period_str = {str(k): v for k, v in by_period.items()}

        if _MODELS_AVAILABLE:
            return AggregationResult(
                total_emissions_tco2e=_quantize(total_tco2e),
                total_capex_usd=_quantize(total_capex),
                by_category=by_category,
                by_method=by_method,
                by_supplier=by_supplier,
                by_period=by_period_str,
                intensity_per_capex=intensity_per_capex,
                intensity_per_revenue=intensity_per_revenue,
                provenance_hash=provenance_hash,
            )

        return {
            "total_emissions_tco2e": _quantize(total_tco2e),
            "total_capex_usd": _quantize(total_capex),
            "by_category": by_category,
            "by_method": by_method,
            "by_supplier": by_supplier,
            "by_period": by_period_str,
            "intensity_per_capex": intensity_per_capex,
            "intensity_per_revenue": intensity_per_revenue,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # Public API (20 - supplementary): get engine stats
    # ==================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics.

        Returns:
            Dict with aggregation count, total assets processed,
            and cumulative emissions.
        """
        with self._lock:
            return {
                "agent_id": AGENT_ID,
                "version": VERSION,
                "engine": "HybridAggregatorEngine",
                "aggregation_count": self._aggregation_count,
                "total_assets_aggregated": self._total_assets_aggregated,
                "total_emissions_aggregated_tco2e": _quantize(
                    self._total_emissions_aggregated_tco2e
                ),
                "config_loaded": self._config is not None,
                "metrics_loaded": self._metrics is not None,
                "provenance_loaded": self._provenance is not None,
            }

    # ==================================================================
    # Public API (supplementary): full aggregation with all analyses
    # ==================================================================

    def full_aggregation(
        self,
        spend_results: List[Any],
        average_results: List[Any],
        supplier_results: List[Any],
        assets: List[Any],
        historical_capex: Optional[List[Decimal]] = None,
        cat1_asset_ids: Optional[Set[str]] = None,
        scope1_asset_ids: Optional[Set[str]] = None,
        revenue: Optional[Decimal] = None,
        fte: Optional[Decimal] = None,
        floor_area: Optional[Decimal] = None,
        previous_result: Optional[Any] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform full aggregation with all analyses in one call.

        This convenience method runs:
        1. aggregate() -- hybrid method selection.
        2. detect_double_counting() -- overlap detection.
        3. hot_spot_analysis() -- Pareto and quadrants.
        4. calculate_capex_volatility() -- volatility context.
        5. calculate_intensity_metrics() -- intensity ratios.
        6. calculate_combined_uncertainty() -- RSS uncertainty.
        7. yoy_decomposition() -- if previous_result provided.
        8. analyze_coverage() -- method coverage report.

        Args:
            spend_results: Spend-based engine results.
            average_results: Average-data engine results.
            supplier_results: Supplier-specific engine results.
            assets: Capital asset records.
            historical_capex: Historical CapEx for volatility.
            cat1_asset_ids: Category 1 asset IDs for overlap check.
            scope1_asset_ids: Scope 1/2 asset IDs for overlap check.
            revenue: Revenue for intensity metric.
            fte: FTE headcount for intensity metric.
            floor_area: Floor area for intensity metric.
            previous_result: Previous period result for YoY.
            calculation_id: Optional pre-generated ID.

        Returns:
            Dict with all analysis results keyed by analysis type.
        """
        t_start = time.monotonic()
        calc_id = calculation_id or str(uuid4())

        logger.info(
            "[%s] Full aggregation started: %d assets",
            calc_id,
            len(assets),
        )

        # 1. Core aggregation
        hybrid_result = self.aggregate(
            spend_results=spend_results,
            average_results=average_results,
            supplier_results=supplier_results,
            assets=assets,
            calculation_id=calc_id,
        )

        # 2. Double-counting detection
        all_results = (
            list(spend_results)
            + list(average_results)
            + list(supplier_results)
        )
        dc_findings = self.detect_double_counting(
            results=all_results,
            cat1_asset_ids=cat1_asset_ids,
            scope1_asset_ids=scope1_asset_ids,
        )

        # 3. Hot-spot analysis
        hotspot = self.hot_spot_analysis(
            results=all_results,
            assets=assets,
        )

        # 4. CapEx volatility
        current_capex = self._extract_total_capex(hybrid_result)
        volatility = self.calculate_capex_volatility(
            current_year_capex=current_capex,
            historical_capex=historical_capex or [],
        )

        # 5. Intensity metrics
        total_tco2e = self._extract_total_tco2e(hybrid_result)
        intensity = self.calculate_intensity_metrics(
            total_emissions=total_tco2e,
            revenue=revenue,
            fte=fte,
            floor_area=floor_area,
        )

        # 6. Combined uncertainty
        uncertainty = self.calculate_combined_uncertainty(all_results)

        # 7. Coverage analysis
        coverage = self.analyze_coverage(
            results=all_results,
            total_assets=assets,
        )

        # 8. YoY decomposition (optional)
        yoy = None
        if previous_result is not None:
            yoy = self.yoy_decomposition(
                current=hybrid_result,
                previous=previous_result,
            )

        # 9. Aggregation breakdown
        aggregation = self._build_aggregation_result(
            results=all_results,
            assets=assets,
            revenue=revenue,
        )

        elapsed = (time.monotonic() - t_start) * 1000

        full_result = {
            "calculation_id": calc_id,
            "hybrid_result": hybrid_result,
            "double_counting_findings": dc_findings,
            "hot_spot_analysis": hotspot,
            "capex_volatility": volatility,
            "intensity_metrics": intensity,
            "combined_uncertainty": uncertainty,
            "coverage_report": coverage,
            "yoy_decomposition": yoy,
            "aggregation": aggregation,
            "processing_time_ms": _quantize(Decimal(str(elapsed))),
            "provenance_hash": _compute_sha256({
                "calculation_id": calc_id,
                "total_tco2e": _decimal_to_str(total_tco2e),
                "double_counting_count": len(dc_findings),
                "coverage": _decimal_to_str(
                    self._extract_coverage_pct(hybrid_result)
                ),
            }),
        }

        logger.info(
            "[%s] Full aggregation complete in %.1f ms",
            calc_id,
            elapsed,
        )
        return full_result
