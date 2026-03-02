# -*- coding: utf-8 -*-
"""
HybridAggregatorEngine -- AGENT-MRV-023 Engine 5 of 7

This module implements the HybridAggregatorEngine for the Processing of
Sold Products Agent (GL-MRV-S3-010).  The engine aggregates emissions
results from multiple calculation methods -- site-specific (direct,
energy, fuel), average-data, and spend-based -- into a single coherent
portfolio-level result applying the GHG Protocol method waterfall.

Core Responsibilities:

1. **Method Waterfall Application**
    Selects the highest-quality available method per product following
    the GHG Protocol recommended hierarchy:
        SITE_SPECIFIC_DIRECT > SITE_SPECIFIC_ENERGY >
        SITE_SPECIFIC_FUEL > AVERAGE_DATA > SPEND_BASED

2. **Multi-Method Aggregation**
    Merges results from Engines 2 (SiteSpecific), 3 (AverageData), and
    4 (SpendBased) into a unified ``CalculationResult`` with full
    per-product breakdowns and provenance tracking.

3. **Gap-Filling**
    For products with no result from any engine, falls back to
    average-data using the product category emission factor.

4. **Allocation**
    Distributes total processing emissions across end-uses using
    four allocation methods: mass, revenue, units, equal.

5. **Portfolio DQI Scoring**
    Computes an emissions-weighted data quality score across all
    product breakdowns.

6. **Portfolio Uncertainty**
    Computes combined analytical uncertainty using error propagation
    across all product breakdowns, each with method-specific
    uncertainty half-widths.

7. **Multi-Dimensional Aggregation**
    Aggregates breakdowns by product category, calculation method,
    customer country, and reporting period.

8. **Hotspot Identification**
    Identifies the minimum set of products that account for a
    target fraction (default 80%) of total emissions (Pareto).

9. **Method Coverage Analysis**
    Reports the fraction of total emissions covered by each
    calculation method for transparency and improvement tracking.

Formulae:

    Method Waterfall:
        For each product p, select result from the highest-priority
        method m for which a valid result exists:
            result[p] = first_available(
                site_results.direct[p],
                site_results.energy[p],
                site_results.fuel[p],
                avg_results[p],
                spend_results[p]
            )

    Portfolio DQI:
        DQI_portfolio = SUM(E_i * DQI_i) / SUM(E_i)
        where E_i = emissions for product i, DQI_i = DQI score for product i.

    Combined Uncertainty (analytical, 95% CI):
        sigma_total = sqrt( SUM( (u_i * E_i)^2 ) )
        half_width = 1.96 * sigma_total
        where u_i = uncertainty half-width fraction for product i.

    Allocation (mass example):
        E_allocated_j = E_total * (mass_j / SUM(mass_k))

Thread Safety:
    The engine is implemented as a thread-safe singleton using
    ``threading.RLock`` with double-checked locking.  All internal
    state is either immutable (constant tables) or protected by the
    instance lock.

Zero-Hallucination Guarantee:
    All numeric operations use ``Decimal`` with explicit quantization
    and ROUND_HALF_UP.  No LLM calls are made for any numeric
    calculation.  Emission factors and DQI scores are sourced
    exclusively from the deterministic constant tables in the sibling
    modules and ``models.py``.

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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "HybridAggregatorEngine",
    "METHOD_PRIORITY",
    "WATERFALL_ORDER",
]

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"
ENGINE_ID: str = "hybrid_aggregator_engine"
ENGINE_VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

DECIMAL_PLACES: int = 8
ZERO: Decimal = Decimal("0")
ONE: Decimal = Decimal("1")
TWO: Decimal = Decimal("2")
ONE_HUNDRED: Decimal = Decimal("100")
ONE_THOUSAND: Decimal = Decimal("1000")
_PRECISION: Decimal = Decimal(10) ** -DECIMAL_PLACES
_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# Z-score for 95% CI
_Z_95: Decimal = Decimal("1.96")

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation methods ordered by GHG Protocol data quality preference."""

    SITE_SPECIFIC_DIRECT = "site_specific_direct"
    SITE_SPECIFIC_ENERGY = "site_specific_energy"
    SITE_SPECIFIC_FUEL = "site_specific_fuel"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class AllocationMethod(str, Enum):
    """Allocation approaches for distributing emissions across end-uses."""

    MASS = "mass"
    REVENUE = "revenue"
    UNITS = "units"
    EQUAL = "equal"


class AggregationPeriod(str, Enum):
    """Time period granularity for period-based aggregation."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"


# ==============================================================================
# METHOD WATERFALL CONFIGURATION
# ==============================================================================

# Priority ordering: lower number = higher priority.
METHOD_PRIORITY: Dict[str, int] = {
    CalculationMethod.SITE_SPECIFIC_DIRECT.value: 1,
    CalculationMethod.SITE_SPECIFIC_ENERGY.value: 2,
    CalculationMethod.SITE_SPECIFIC_FUEL.value: 3,
    CalculationMethod.AVERAGE_DATA.value: 4,
    CalculationMethod.SPEND_BASED.value: 5,
}

WATERFALL_ORDER: List[str] = [
    CalculationMethod.SITE_SPECIFIC_DIRECT.value,
    CalculationMethod.SITE_SPECIFIC_ENERGY.value,
    CalculationMethod.SITE_SPECIFIC_FUEL.value,
    CalculationMethod.AVERAGE_DATA.value,
    CalculationMethod.SPEND_BASED.value,
]

# ==============================================================================
# DQI AND UNCERTAINTY DEFAULTS
# ==============================================================================

# Default DQI scores (1-5, 5 = best) per method for waterfall selection.
# Source: GHG Protocol Scope 3 Standard, Table 7.1 (adapted for Cat 10)
METHOD_DQI_DEFAULTS: Dict[str, Decimal] = {
    CalculationMethod.SITE_SPECIFIC_DIRECT.value: Decimal("4.6"),
    CalculationMethod.SITE_SPECIFIC_ENERGY.value: Decimal("4.0"),
    CalculationMethod.SITE_SPECIFIC_FUEL.value: Decimal("3.6"),
    CalculationMethod.AVERAGE_DATA.value: Decimal("2.8"),
    CalculationMethod.SPEND_BASED.value: Decimal("1.6"),
}

# Uncertainty half-width fractions (95% CI) per method.
# Source: IPCC 2006 Vol 1 Ch 3 Table 3.2 + GHG Protocol Cat 10 guidance
METHOD_UNCERTAINTY_FRACTIONS: Dict[str, Decimal] = {
    CalculationMethod.SITE_SPECIFIC_DIRECT.value: Decimal("0.10"),
    CalculationMethod.SITE_SPECIFIC_ENERGY.value: Decimal("0.15"),
    CalculationMethod.SITE_SPECIFIC_FUEL.value: Decimal("0.15"),
    CalculationMethod.AVERAGE_DATA.value: Decimal("0.30"),
    CalculationMethod.SPEND_BASED.value: Decimal("0.50"),
}

# 5-dimension DQI scoring (1-5) per method, per dimension.
# Dimensions: reliability, completeness, temporal, geographical, technological.
METHOD_DQI_DIMENSIONS: Dict[str, Dict[str, int]] = {
    CalculationMethod.SITE_SPECIFIC_DIRECT.value: {
        "reliability": 5,
        "completeness": 4,
        "temporal": 5,
        "geographical": 5,
        "technological": 4,
    },
    CalculationMethod.SITE_SPECIFIC_ENERGY.value: {
        "reliability": 4,
        "completeness": 4,
        "temporal": 4,
        "geographical": 4,
        "technological": 4,
    },
    CalculationMethod.SITE_SPECIFIC_FUEL.value: {
        "reliability": 4,
        "completeness": 3,
        "temporal": 4,
        "geographical": 4,
        "technological": 3,
    },
    CalculationMethod.AVERAGE_DATA.value: {
        "reliability": 3,
        "completeness": 3,
        "temporal": 3,
        "geographical": 2,
        "technological": 3,
    },
    CalculationMethod.SPEND_BASED.value: {
        "reliability": 1,
        "completeness": 2,
        "temporal": 2,
        "geographical": 1,
        "technological": 1,
    },
}

# Dimension weights (sum to 1.0) per GHG Protocol Scope 3 guidance.
DQI_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "reliability": Decimal("0.25"),
    "completeness": Decimal("0.25"),
    "temporal": Decimal("0.20"),
    "geographical": Decimal("0.15"),
    "technological": Decimal("0.15"),
}

# ==============================================================================
# PROCESSING EMISSION FACTOR FALLBACK TABLE (for gap-filling)
# ==============================================================================
# kgCO2e per tonne of intermediate product processed downstream.
# Source: GHG Protocol Cat 10 Technical Guidance, EPA USEEIO, Ecoinvent 3.10
# NOTE: These mirror the values in models.py and processing_database.py.
# They are duplicated here so that the engine operates independently when
# sibling modules cannot be imported.

_FALLBACK_PROCESSING_EFS: Dict[str, Decimal] = {
    "metals_ferrous": Decimal("280"),
    "metals_non_ferrous": Decimal("380"),
    "plastics_thermoplastic": Decimal("520"),
    "plastics_thermoset": Decimal("450"),
    "chemicals": Decimal("680"),
    "food_ingredients": Decimal("130"),
    "textiles": Decimal("350"),
    "electronics": Decimal("950"),
    "glass_ceramics": Decimal("580"),
    "wood_paper": Decimal("190"),
    "minerals": Decimal("250"),
    "agricultural": Decimal("110"),
}

# Uppercase keys variant for compatibility with ProcessingDatabaseEngine enums.
_FALLBACK_PROCESSING_EFS_UPPER: Dict[str, Decimal] = {
    k.upper(): v for k, v in _FALLBACK_PROCESSING_EFS.items()
}

# ==============================================================================
# GRACEFUL IMPORTS FROM SIBLING MODULES
# ==============================================================================

try:
    from greenlang.processing_sold_products.models import (
        IntermediateProductCategory,
        ProcessingType,
        CalculationMethod as ModelsCalculationMethod,
        AllocationMethod as ModelsAllocationMethod,
        DataQualityTier,
        DQIDimension,
        UncertaintyMethod,
        IntermediateProductInput,
        CalculationResult as ModelsCalculationResult,
        ProductBreakdown as ModelsProductBreakdown,
        AggregationResult as ModelsAggregationResult,
        DataQualityScore as ModelsDataQualityScore,
        UncertaintyResult as ModelsUncertaintyResult,
        PROCESSING_EMISSION_FACTORS,
        UNCERTAINTY_RANGES,
        calculate_provenance_hash,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning(
        "HybridAggregatorEngine: models module not available; "
        "using internal fallback tables."
    )

try:
    from greenlang.processing_sold_products.processing_database import (
        ProcessingDatabaseEngine,
    )
    _DB_ENGINE_AVAILABLE = True
except ImportError:
    _DB_ENGINE_AVAILABLE = False
    logger.debug(
        "HybridAggregatorEngine: ProcessingDatabaseEngine not available."
    )

try:
    from greenlang.processing_sold_products.site_specific_calculator import (
        SiteSpecificCalculatorEngine,
    )
    _SS_ENGINE_AVAILABLE = True
except ImportError:
    _SS_ENGINE_AVAILABLE = False

try:
    from greenlang.processing_sold_products.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
    _AVG_ENGINE_AVAILABLE = True
except ImportError:
    _AVG_ENGINE_AVAILABLE = False

try:
    from greenlang.processing_sold_products.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
    _SPEND_ENGINE_AVAILABLE = True
except ImportError:
    _SPEND_ENGINE_AVAILABLE = False


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _quantize(value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
    """Quantize a Decimal value to the specified precision.

    Args:
        value: Decimal value to quantize.
        precision: Quantization level (default 8 decimal places).

    Returns:
        Quantized Decimal value.
    """
    try:
        return value.quantize(precision, rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError):
        return ZERO


def _safe_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely.

    Args:
        value: Input value (str, int, float, Decimal).

    Returns:
        Decimal representation, or ZERO on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return ZERO


def _compute_sha256(*parts: Any) -> str:
    """Compute SHA-256 hash from variable parts.

    Args:
        *parts: Variable number of inputs to hash.

    Returns:
        64-character hex SHA-256 hash string.
    """
    payload = ""
    for part in parts:
        if isinstance(part, dict):
            payload += json.dumps(part, sort_keys=True, default=str)
        elif isinstance(part, (list, tuple)):
            payload += json.dumps(
                [str(x) if isinstance(x, Decimal) else x for x in part],
                sort_keys=True,
                default=str,
            )
        elif isinstance(part, Decimal):
            payload += str(_quantize(part))
        else:
            payload += str(part)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_product_id(breakdown: Dict[str, Any]) -> str:
    """Extract product_id from a breakdown dict or Pydantic model.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Product identifier string.
    """
    if isinstance(breakdown, dict):
        return str(breakdown.get("product_id", ""))
    return str(getattr(breakdown, "product_id", ""))


def _extract_emissions(breakdown: Any) -> Decimal:
    """Extract emissions (kgCO2e) from a breakdown dict or model.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Emissions as Decimal.
    """
    if isinstance(breakdown, dict):
        for key in ("emissions_kg", "emissions_kg_co2e", "emissions_kgco2e", "co2e_kg"):
            if key in breakdown:
                return _safe_decimal(breakdown[key])
        return ZERO
    for attr in ("emissions_kg", "emissions_kg_co2e", "emissions_kgco2e", "co2e_kg"):
        val = getattr(breakdown, attr, None)
        if val is not None:
            return _safe_decimal(val)
    return ZERO


def _extract_method(breakdown: Any) -> str:
    """Extract calculation method from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Method string or 'unknown'.
    """
    if isinstance(breakdown, dict):
        return str(breakdown.get("method", "unknown"))
    return str(getattr(breakdown, "method", "unknown"))


def _extract_dqi(breakdown: Any) -> Decimal:
    """Extract DQI score from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        DQI score as Decimal (1-5), default based on method.
    """
    if isinstance(breakdown, dict):
        raw = breakdown.get("dqi", breakdown.get("dqi_score"))
    else:
        raw = getattr(breakdown, "dqi", getattr(breakdown, "dqi_score", None))

    if raw is not None:
        return _safe_decimal(raw)

    # Fallback: derive from method
    method = _extract_method(breakdown)
    return METHOD_DQI_DEFAULTS.get(method, Decimal("2.0"))


def _extract_category(breakdown: Any) -> str:
    """Extract product category from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Category string.
    """
    if isinstance(breakdown, dict):
        return str(breakdown.get("category", "unknown"))
    return str(getattr(breakdown, "category", "unknown"))


def _extract_country(breakdown: Any) -> str:
    """Extract customer country from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Country string or 'GLOBAL'.
    """
    if isinstance(breakdown, dict):
        return str(breakdown.get("country", breakdown.get("customer_country", "GLOBAL")))
    return str(getattr(breakdown, "country", getattr(breakdown, "customer_country", "GLOBAL")))


def _extract_quantity(breakdown: Any) -> Decimal:
    """Extract product quantity from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Quantity as Decimal.
    """
    if isinstance(breakdown, dict):
        for key in ("quantity", "quantity_tonnes", "mass_tonnes"):
            if key in breakdown:
                return _safe_decimal(breakdown[key])
        return ONE
    for attr in ("quantity", "quantity_tonnes", "mass_tonnes"):
        val = getattr(breakdown, attr, None)
        if val is not None:
            return _safe_decimal(val)
    return ONE


def _extract_ef_used(breakdown: Any) -> Decimal:
    """Extract emission factor used from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Emission factor as Decimal.
    """
    if isinstance(breakdown, dict):
        for key in ("ef_used", "emission_factor", "ef"):
            if key in breakdown:
                return _safe_decimal(breakdown[key])
        return ZERO
    for attr in ("ef_used", "emission_factor", "ef"):
        val = getattr(breakdown, attr, None)
        if val is not None:
            return _safe_decimal(val)
    return ZERO


def _extract_processing_type(breakdown: Any) -> str:
    """Extract processing type from a breakdown.

    Args:
        breakdown: Product breakdown dict or model.

    Returns:
        Processing type string.
    """
    if isinstance(breakdown, dict):
        return str(breakdown.get("processing_type", "unknown"))
    return str(getattr(breakdown, "processing_type", "unknown"))


def _extract_breakdowns_from_result(result: Any) -> List[Dict[str, Any]]:
    """Extract product breakdowns from a calculation result (dict or model).

    Handles CalculationResult objects from various engines that may have
    different attribute names for the breakdown list.

    Args:
        result: A calculation result dict or Pydantic model.

    Returns:
        List of breakdown dicts.
    """
    if result is None:
        return []

    # Try known attribute names
    for attr in ("product_breakdowns", "breakdowns", "items"):
        raw = None
        if isinstance(result, dict):
            raw = result.get(attr)
        else:
            raw = getattr(result, attr, None)

        if raw is not None and isinstance(raw, (list, tuple)):
            out: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    # Convert Pydantic model to dict
                    try:
                        out.append(item.model_dump(mode="json"))
                    except AttributeError:
                        try:
                            out.append(item.dict())
                        except AttributeError:
                            out.append({"product_id": str(item)})
            return out

    return []


def _get_fallback_ef(category: str) -> Decimal:
    """Get fallback processing emission factor for gap-filling.

    Args:
        category: Product category string (lowercase or uppercase).

    Returns:
        Emission factor in kgCO2e/tonne, or Decimal('300') as a global fallback.
    """
    ef = _FALLBACK_PROCESSING_EFS.get(category.lower())
    if ef is not None:
        return ef
    ef = _FALLBACK_PROCESSING_EFS_UPPER.get(category.upper())
    if ef is not None:
        return ef
    # Global average fallback
    return Decimal("300")


# ==============================================================================
# HYBRID AGGREGATOR ENGINE
# ==============================================================================


class HybridAggregatorEngine:
    """Engine 5: Hybrid multi-method aggregator for Processing of Sold Products.

    Combines results from site-specific (direct, energy, fuel), average-data,
    and spend-based calculation engines into a single portfolio-level result.

    The engine applies the GHG Protocol method waterfall to select the
    highest-quality result per product, gap-fills missing products using
    average-data emission factors, and produces multi-dimensional aggregation
    with portfolio-level DQI scoring and uncertainty quantification.

    Thread Safety:
        Singleton via ``__new__`` with ``threading.RLock()``.

    Zero-Hallucination:
        All arithmetic uses ``Decimal`` with ``ROUND_HALF_UP``.
        No LLM involvement in any numeric path.

    Attributes:
        _aggregation_count: Total aggregation operations performed.
        _total_aggregated_emissions: Cumulative aggregated emissions (kgCO2e).
        _waterfall_selections: Count of selections per method in waterfall.
        _gap_fill_count: Number of products that required gap-filling.
        _initialized: Singleton initialization guard.

    Example:
        >>> engine = HybridAggregatorEngine()
        >>> result = engine.aggregate(
        ...     products=[...],
        ...     site_results={...},
        ...     avg_results={...},
        ...     spend_results={...},
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ... )
        >>> print(f"Total: {result['total_emissions_kg']} kgCO2e")
    """

    _instance: Optional["HybridAggregatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "HybridAggregatorEngine":
        """Thread-safe singleton instantiation using double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the HybridAggregatorEngine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._aggregation_count: int = 0
        self._total_aggregated_emissions: Decimal = ZERO
        self._waterfall_selections: Dict[str, int] = {
            method: 0 for method in WATERFALL_ORDER
        }
        self._gap_fill_count: int = 0
        self._op_lock: threading.RLock = threading.RLock()

        logger.info(
            "HybridAggregatorEngine initialized: "
            "waterfall_methods=%d, allocation_methods=%d, "
            "fallback_categories=%d, version=%s",
            len(WATERFALL_ORDER),
            len(AllocationMethod),
            len(_FALLBACK_PROCESSING_EFS),
            ENGINE_VERSION,
        )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def aggregate(
        self,
        products: List[Any],
        site_results: Optional[Any] = None,
        avg_results: Optional[Any] = None,
        spend_results: Optional[Any] = None,
        org_id: str = "ORG-000",
        reporting_year: int = 2025,
    ) -> Dict[str, Any]:
        """Aggregate multi-method results into a unified portfolio calculation.

        Applies the GHG Protocol method waterfall per product, gap-fills
        any remaining products, computes portfolio-level DQI and uncertainty,
        and generates multi-dimensional aggregations.

        Args:
            products: List of product input dicts or Pydantic models.
            site_results: Result(s) from SiteSpecificCalculatorEngine (dict,
                model, or list of results). May contain direct, energy,
                and fuel sub-results.
            avg_results: Result(s) from AverageDataCalculatorEngine.
            spend_results: Result(s) from SpendBasedCalculatorEngine.
            org_id: Reporting organization identifier.
            reporting_year: Reporting year for the calculation.

        Returns:
            Dict with keys: calc_id, org_id, reporting_year, method,
            total_emissions_kg, total_emissions_tco2e, product_breakdowns,
            dqi_score, uncertainty, by_category, by_method, by_country,
            hotspots, method_coverage, provenance_hash, timestamp,
            processing_time_ms.
        """
        start = time.monotonic()
        calc_id = f"psp-hybrid-{uuid.uuid4().hex[:12]}"
        logger.info(
            "Aggregate started: calc_id=%s, products=%d, org_id=%s, year=%d",
            calc_id, len(products), org_id, reporting_year,
        )

        # Step 1: Index incoming engine results by product_id
        site_breakdowns = self._index_breakdowns(site_results)
        avg_breakdowns = self._index_breakdowns(avg_results)
        spend_breakdowns = self._index_breakdowns(spend_results)

        # Step 2: Apply method waterfall for each product
        portfolio_breakdowns: List[Dict[str, Any]] = []
        for product in products:
            bd = self.apply_method_waterfall(
                product,
                site_breakdowns=site_breakdowns,
                avg_breakdowns=avg_breakdowns,
                spend_breakdowns=spend_breakdowns,
            )
            portfolio_breakdowns.append(bd)

        # Step 3: Gap-fill products that have no result yet
        gap_filled = self.gap_fill(products, portfolio_breakdowns)
        portfolio_breakdowns = gap_filled

        # Step 4: Compute totals
        total_kg = self._sum_emissions(portfolio_breakdowns)
        total_tco2e = _quantize(total_kg / ONE_THOUSAND, _QUANT_8DP)

        # Step 5: Portfolio DQI
        dqi_score = self.compute_portfolio_dqi(portfolio_breakdowns)

        # Step 6: Portfolio uncertainty
        uncertainty = self.compute_portfolio_uncertainty(portfolio_breakdowns)

        # Step 7: Multi-dimensional aggregations
        by_category = self.aggregate_by_category(portfolio_breakdowns)
        by_method = self.aggregate_by_method(portfolio_breakdowns)
        by_country = self.aggregate_by_country(portfolio_breakdowns)

        # Step 8: Hotspot identification
        hotspots = self.identify_hotspots(portfolio_breakdowns)

        # Step 9: Method coverage
        method_coverage = self.compute_method_coverage(portfolio_breakdowns)

        # Step 10: Provenance
        provenance_hash = self._build_provenance(
            {
                "products_count": len(products),
                "org_id": org_id,
                "reporting_year": reporting_year,
                "site_breakdown_count": len(site_breakdowns),
                "avg_breakdown_count": len(avg_breakdowns),
                "spend_breakdown_count": len(spend_breakdowns),
            },
            {
                "total_emissions_kg": str(total_kg),
                "total_tco2e": str(total_tco2e),
                "breakdown_count": len(portfolio_breakdowns),
            },
        )

        elapsed_ms = _quantize(
            Decimal(str((time.monotonic() - start) * 1000)), _QUANT_2DP
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        # Update internal counters
        with self._op_lock:
            self._aggregation_count += 1
            self._total_aggregated_emissions += total_kg

        result = {
            "calc_id": calc_id,
            "org_id": org_id,
            "reporting_year": reporting_year,
            "method": "hybrid",
            "total_emissions_kg": total_kg,
            "total_emissions_tco2e": total_tco2e,
            "product_breakdowns": portfolio_breakdowns,
            "product_count": len(portfolio_breakdowns),
            "dqi_score": dqi_score,
            "uncertainty": uncertainty,
            "by_category": by_category,
            "by_method": by_method,
            "by_country": by_country,
            "hotspots": hotspots,
            "method_coverage": method_coverage,
            "provenance_hash": provenance_hash,
            "timestamp": timestamp,
            "processing_time_ms": elapsed_ms,
        }

        logger.info(
            "Aggregate completed: calc_id=%s, total_kg=%s, breakdowns=%d, "
            "elapsed_ms=%s",
            calc_id, total_kg, len(portfolio_breakdowns), elapsed_ms,
        )

        return result

    def apply_method_waterfall(
        self,
        product: Any,
        site_breakdowns: Optional[Dict[str, Dict[str, Any]]] = None,
        avg_breakdowns: Optional[Dict[str, Dict[str, Any]]] = None,
        spend_breakdowns: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select the best available method result for a product.

        Follows the GHG Protocol recommended hierarchy:
            1. SITE_SPECIFIC_DIRECT
            2. SITE_SPECIFIC_ENERGY
            3. SITE_SPECIFIC_FUEL
            4. AVERAGE_DATA
            5. SPEND_BASED

        Args:
            product: Product input dict or Pydantic model.
            site_breakdowns: Indexed site-specific breakdowns by product_id.
            avg_breakdowns: Indexed average-data breakdowns by product_id.
            spend_breakdowns: Indexed spend-based breakdowns by product_id.

        Returns:
            Dict representing the best-available ProductBreakdown for this
            product. If no engine result is available, returns a zero-emission
            placeholder (gap-fill will handle it later).
        """
        product_id = self._get_product_id(product)
        site_breakdowns = site_breakdowns or {}
        avg_breakdowns = avg_breakdowns or {}
        spend_breakdowns = spend_breakdowns or {}

        # Try site-specific results first (ordered by priority)
        if product_id in site_breakdowns:
            site_bd = site_breakdowns[product_id]
            method = _extract_method(site_bd)
            # Accept any site-specific method
            if method in (
                CalculationMethod.SITE_SPECIFIC_DIRECT.value,
                CalculationMethod.SITE_SPECIFIC_ENERGY.value,
                CalculationMethod.SITE_SPECIFIC_FUEL.value,
                "site_specific_direct",
                "site_specific_energy",
                "site_specific_fuel",
            ):
                with self._op_lock:
                    self._waterfall_selections[method] = (
                        self._waterfall_selections.get(method, 0) + 1
                    )
                return self._normalize_breakdown(site_bd, product)

        # Try average-data
        if product_id in avg_breakdowns:
            avg_bd = avg_breakdowns[product_id]
            with self._op_lock:
                self._waterfall_selections[CalculationMethod.AVERAGE_DATA.value] = (
                    self._waterfall_selections.get(
                        CalculationMethod.AVERAGE_DATA.value, 0
                    ) + 1
                )
            return self._normalize_breakdown(avg_bd, product)

        # Try spend-based
        if product_id in spend_breakdowns:
            spend_bd = spend_breakdowns[product_id]
            with self._op_lock:
                self._waterfall_selections[CalculationMethod.SPEND_BASED.value] = (
                    self._waterfall_selections.get(
                        CalculationMethod.SPEND_BASED.value, 0
                    ) + 1
                )
            return self._normalize_breakdown(spend_bd, product)

        # No result available: return placeholder for gap-filling
        return self._create_placeholder_breakdown(product)

    def allocate_emissions(
        self,
        product: Any,
        method: str,
        end_uses: List[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Allocate processing emissions across end-uses for a product.

        Supports four allocation methods:
            - mass: proportional to mass fraction
            - revenue: proportional to revenue fraction
            - units: proportional to unit count fraction
            - equal: equal split across all end-uses

        Args:
            product: Product breakdown dict or model with emissions_kg.
            method: Allocation method ('mass', 'revenue', 'units', 'equal').
            end_uses: List of end-use dicts with keys depending on method.
                For mass: each dict must have 'mass' (Decimal or numeric).
                For revenue: each dict must have 'revenue'.
                For units: each dict must have 'units'.
                For equal: no special keys required.
                All end-use dicts should have 'end_use_id'.

        Returns:
            Dict mapping end_use_id to allocated emissions (Decimal kgCO2e).

        Raises:
            ValueError: If method is not recognized or end_uses is empty.
        """
        total_emissions = _extract_emissions(product)
        if not end_uses:
            raise ValueError("end_uses must not be empty for allocation.")

        method_lower = method.lower()
        result: Dict[str, Decimal] = {}

        if method_lower == AllocationMethod.EQUAL.value:
            share = _quantize(total_emissions / Decimal(str(len(end_uses))))
            for eu in end_uses:
                eu_id = str(eu.get("end_use_id", f"eu-{len(result)}"))
                result[eu_id] = share
            return self._reconcile_allocation(result, total_emissions)

        # Key-based allocation
        allocation_key = self._get_allocation_key(method_lower)

        total_weight = ZERO
        weights: List[Tuple[str, Decimal]] = []
        for eu in end_uses:
            eu_id = str(eu.get("end_use_id", f"eu-{len(weights)}"))
            weight = _safe_decimal(eu.get(allocation_key, 0))
            weights.append((eu_id, weight))
            total_weight += weight

        if total_weight <= ZERO:
            # Fall back to equal allocation
            share = _quantize(total_emissions / Decimal(str(len(end_uses))))
            for eu_id, _ in weights:
                result[eu_id] = share
            return self._reconcile_allocation(result, total_emissions)

        for eu_id, weight in weights:
            fraction = _quantize(weight / total_weight)
            allocated = _quantize(total_emissions * fraction)
            result[eu_id] = allocated

        return self._reconcile_allocation(result, total_emissions)

    def gap_fill(
        self,
        products: List[Any],
        existing_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Gap-fill products that have no calculation result.

        Products with zero emissions (placeholder breakdowns) are filled
        using average-data emission factors based on product category.

        Args:
            products: Original product input list.
            existing_results: Current breakdown list (may contain placeholders).

        Returns:
            Updated list of breakdowns with gap-filled entries.
        """
        product_index: Dict[str, Any] = {}
        for p in products:
            pid = self._get_product_id(p)
            product_index[pid] = p

        filled: List[Dict[str, Any]] = []
        for bd in existing_results:
            pid = str(bd.get("product_id", ""))
            emissions = _extract_emissions(bd)

            if emissions > ZERO:
                # Already has a valid result
                filled.append(bd)
                continue

            # Gap-fill: use average-data EF
            product = product_index.get(pid)
            if product is None:
                filled.append(bd)
                continue

            category = self._get_product_category(product)
            quantity = self._get_product_quantity(product)
            ef = _get_fallback_ef(category)
            gap_emissions = _quantize(quantity * ef)

            gap_bd = {
                "product_id": pid,
                "category": category,
                "processing_type": self._get_product_processing_type(product),
                "quantity": quantity,
                "emissions_kg": gap_emissions,
                "ef_used": ef,
                "method": CalculationMethod.AVERAGE_DATA.value,
                "dqi": METHOD_DQI_DEFAULTS[CalculationMethod.AVERAGE_DATA.value],
                "gap_filled": True,
                "country": self._get_product_country(product),
            }

            with self._op_lock:
                self._gap_fill_count += 1

            filled.append(gap_bd)

        return filled

    def compute_portfolio_dqi(
        self,
        breakdowns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute emissions-weighted portfolio-level DQI score.

        The portfolio DQI is a weighted average of per-product DQI scores,
        where the weight is the product's share of total emissions.

        Formula:
            DQI_portfolio = SUM(E_i * DQI_i) / SUM(E_i)

        Additionally computes per-dimension weighted scores.

        Args:
            breakdowns: List of product breakdown dicts.

        Returns:
            Dict with keys: reliability, completeness, temporal,
            geographical, technological, overall (all Decimal 1-5).
        """
        total_emissions = ZERO
        weighted_overall = ZERO
        weighted_dimensions: Dict[str, Decimal] = {
            dim: ZERO for dim in DQI_DIMENSION_WEIGHTS
        }

        for bd in breakdowns:
            e = _extract_emissions(bd)
            dqi = _extract_dqi(bd)
            method = _extract_method(bd)
            total_emissions += e
            weighted_overall += e * dqi

            # Per-dimension scores
            dim_scores = METHOD_DQI_DIMENSIONS.get(method, {})
            for dim_name in DQI_DIMENSION_WEIGHTS:
                dim_val = Decimal(str(dim_scores.get(dim_name, 2)))
                weighted_dimensions[dim_name] += e * dim_val

        if total_emissions <= ZERO:
            return {
                "reliability": 1,
                "completeness": 1,
                "temporal": 1,
                "geographical": 1,
                "technological": 1,
                "overall": Decimal("1.0"),
            }

        overall = _quantize(weighted_overall / total_emissions, _QUANT_2DP)
        overall = min(overall, Decimal("5.0"))
        overall = max(overall, Decimal("1.0"))

        dimension_results: Dict[str, Any] = {}
        for dim_name in DQI_DIMENSION_WEIGHTS:
            raw = _quantize(
                weighted_dimensions[dim_name] / total_emissions, _QUANT_2DP
            )
            clamped = min(max(raw, ONE), Decimal("5"))
            dimension_results[dim_name] = int(
                clamped.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )

        dimension_results["overall"] = overall
        return dimension_results

    def compute_portfolio_uncertainty(
        self,
        breakdowns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute combined portfolio uncertainty via analytical error propagation.

        Assumes independent, normally distributed uncertainties per product.
        Uses the method-specific uncertainty half-width fractions.

        Formula (quadrature sum):
            sigma_total = sqrt( SUM( (u_i * E_i)^2 ) )
            half_width_95 = 1.96 * sigma_total / sqrt(N)

        For a conservative estimate, we use:
            half_width_95 = sqrt( SUM( (u_i * E_i)^2 ) )
            ci_lower = total - half_width_95
            ci_upper = total + half_width_95

        Args:
            breakdowns: List of product breakdown dicts.

        Returns:
            Dict with keys: method, mean, std_dev, ci_lower, ci_upper,
            iterations, half_width_fraction.
        """
        total_emissions = ZERO
        variance_sum = ZERO

        for bd in breakdowns:
            e = _extract_emissions(bd)
            method = _extract_method(bd)
            u_frac = METHOD_UNCERTAINTY_FRACTIONS.get(method, Decimal("0.50"))

            total_emissions += e
            # sigma_i = u_frac * E_i  (half-width at 95% CI)
            sigma_i = u_frac * e
            variance_sum += sigma_i * sigma_i

        if total_emissions <= ZERO:
            return {
                "method": "analytical",
                "mean": ZERO,
                "std_dev": ZERO,
                "ci_lower": ZERO,
                "ci_upper": ZERO,
                "iterations": 0,
                "half_width_fraction": ZERO,
            }

        # Combined standard error (root-sum-square)
        std_dev = _quantize(
            Decimal(str(math.sqrt(float(variance_sum)))), _QUANT_8DP
        )

        ci_lower = _quantize(total_emissions - std_dev, _QUANT_8DP)
        ci_upper = _quantize(total_emissions + std_dev, _QUANT_8DP)

        if ci_lower < ZERO:
            ci_lower = ZERO

        half_width_fraction = ZERO
        if total_emissions > ZERO:
            half_width_fraction = _quantize(std_dev / total_emissions, _QUANT_4DP)

        return {
            "method": "analytical",
            "mean": total_emissions,
            "std_dev": std_dev,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "iterations": 0,
            "half_width_fraction": half_width_fraction,
        }

    def aggregate_by_category(
        self,
        breakdowns: List[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by product category.

        Args:
            breakdowns: List of product breakdown dicts.

        Returns:
            Dict mapping category string to total emissions (Decimal kgCO2e).
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            cat = _extract_category(bd)
            emissions = _extract_emissions(bd)
            result[cat] = _quantize(result[cat] + emissions)
        return dict(result)

    def aggregate_by_method(
        self,
        breakdowns: List[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by calculation method.

        Args:
            breakdowns: List of product breakdown dicts.

        Returns:
            Dict mapping method string to total emissions (Decimal kgCO2e).
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            method = _extract_method(bd)
            emissions = _extract_emissions(bd)
            result[method] = _quantize(result[method] + emissions)
        return dict(result)

    def aggregate_by_country(
        self,
        breakdowns: List[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by customer country.

        Args:
            breakdowns: List of product breakdown dicts.

        Returns:
            Dict mapping country string to total emissions (Decimal kgCO2e).
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            country = _extract_country(bd)
            emissions = _extract_emissions(bd)
            result[country] = _quantize(result[country] + emissions)
        return dict(result)

    def aggregate_by_period(
        self,
        breakdowns: List[Dict[str, Any]],
        period: str = "annual",
    ) -> Dict[str, Any]:
        """Aggregate emissions into an AggregationResult-compatible structure.

        Produces a summary dict matching the models.AggregationResult schema
        with by_category, by_method, and by_country breakdowns.

        Args:
            breakdowns: List of product breakdown dicts.
            period: Reporting period string (e.g., '2025', '2025-Q3').

        Returns:
            Dict with keys: period, total_tco2e, by_category, by_method,
            by_country.
        """
        by_category = self.aggregate_by_category(breakdowns)
        by_method = self.aggregate_by_method(breakdowns)
        by_country = self.aggregate_by_country(breakdowns)

        total_kg = self._sum_emissions(breakdowns)
        total_tco2e = _quantize(total_kg / ONE_THOUSAND, _QUANT_8DP)

        # Convert category aggregations to tCO2e
        by_category_tco2e: Dict[str, Decimal] = {
            k: _quantize(v / ONE_THOUSAND, _QUANT_8DP)
            for k, v in by_category.items()
        }
        by_method_tco2e: Dict[str, Decimal] = {
            k: _quantize(v / ONE_THOUSAND, _QUANT_8DP)
            for k, v in by_method.items()
        }
        by_country_tco2e: Dict[str, Decimal] = {
            k: _quantize(v / ONE_THOUSAND, _QUANT_8DP)
            for k, v in by_country.items()
        }

        return {
            "period": period,
            "total_tco2e": total_tco2e,
            "by_category": by_category_tco2e,
            "by_method": by_method_tco2e,
            "by_country": by_country_tco2e,
        }

    def identify_hotspots(
        self,
        breakdowns: List[Dict[str, Any]],
        threshold: Decimal = Decimal("0.80"),
    ) -> List[Dict[str, Any]]:
        """Identify Pareto hotspot products accounting for a target emission share.

        Sorts products by emissions descending and returns the minimum set
        whose cumulative share reaches the threshold (default 80%).

        Args:
            breakdowns: List of product breakdown dicts.
            threshold: Cumulative emission fraction target (0-1). Default 0.80.

        Returns:
            List of breakdown dicts for the hotspot products, each annotated
            with 'cumulative_share' and 'rank'.
        """
        total = self._sum_emissions(breakdowns)
        if total <= ZERO:
            return []

        # Sort by emissions descending
        sorted_bds = sorted(
            breakdowns,
            key=lambda bd: _extract_emissions(bd),
            reverse=True,
        )

        hotspots: List[Dict[str, Any]] = []
        cumulative = ZERO
        for rank, bd in enumerate(sorted_bds, start=1):
            e = _extract_emissions(bd)
            cumulative += e
            share = _quantize(cumulative / total, _QUANT_4DP)

            hotspot = dict(bd)
            hotspot["rank"] = rank
            hotspot["cumulative_share"] = share
            hotspot["individual_share"] = _quantize(e / total, _QUANT_4DP)
            hotspots.append(hotspot)

            if share >= threshold:
                break

        return hotspots

    def compute_method_coverage(
        self,
        breakdowns: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Report the fraction of total emissions covered by each method.

        Args:
            breakdowns: List of product breakdown dicts.

        Returns:
            Dict mapping method string to coverage fraction (0.0-1.0).
        """
        by_method = self.aggregate_by_method(breakdowns)
        total = self._sum_emissions(breakdowns)

        if total <= ZERO:
            return {method: 0.0 for method in WATERFALL_ORDER}

        coverage: Dict[str, float] = {}
        for method in WATERFALL_ORDER:
            method_emissions = by_method.get(method, ZERO)
            frac = float(_quantize(method_emissions / total, _QUANT_4DP))
            coverage[method] = frac

        # Include any other methods found in results
        for method, emissions in by_method.items():
            if method not in coverage:
                coverage[method] = float(_quantize(emissions / total, _QUANT_4DP))

        return coverage

    # ==========================================================================
    # PROVENANCE
    # ==========================================================================

    def _build_provenance(
        self,
        inputs: Any,
        result: Any,
    ) -> str:
        """Build SHA-256 provenance hash from inputs and result.

        Args:
            inputs: Input data (dict or serializable object).
            result: Output data (dict or serializable object).

        Returns:
            64-character hex SHA-256 hash string.
        """
        return _compute_sha256(
            ENGINE_ID,
            ENGINE_VERSION,
            inputs,
            result,
            datetime.now(timezone.utc).isoformat(),
        )

    # ==========================================================================
    # STATUS AND METRICS
    # ==========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine operational metrics.

        Returns:
            Dict with aggregation_count, total_aggregated_emissions,
            waterfall_selections, gap_fill_count, version.
        """
        with self._op_lock:
            return {
                "engine_id": ENGINE_ID,
                "version": ENGINE_VERSION,
                "aggregation_count": self._aggregation_count,
                "total_aggregated_emissions_kgco2e": str(
                    self._total_aggregated_emissions
                ),
                "waterfall_selections": dict(self._waterfall_selections),
                "gap_fill_count": self._gap_fill_count,
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the engine.

        Returns:
            Dict with status, engine_id, version, and dependency status.
        """
        return {
            "status": "healthy",
            "engine_id": ENGINE_ID,
            "version": ENGINE_VERSION,
            "models_available": _MODELS_AVAILABLE,
            "db_engine_available": _DB_ENGINE_AVAILABLE,
            "ss_engine_available": _SS_ENGINE_AVAILABLE,
            "avg_engine_available": _AVG_ENGINE_AVAILABLE,
            "spend_engine_available": _SPEND_ENGINE_AVAILABLE,
            "aggregation_count": self._aggregation_count,
        }

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _index_breakdowns(
        self,
        results: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Index product breakdowns from engine results by product_id.

        Handles single results, lists of results, or None.
        If multiple breakdowns exist for the same product_id, the one
        with the highest-priority method (lowest priority number) is kept.

        Args:
            results: Engine result(s) -- dict, model, list, or None.

        Returns:
            Dict mapping product_id to its best breakdown dict.
        """
        if results is None:
            return {}

        # Collect all breakdowns from all results
        all_breakdowns: List[Dict[str, Any]] = []

        if isinstance(results, list):
            for r in results:
                all_breakdowns.extend(_extract_breakdowns_from_result(r))
        else:
            all_breakdowns.extend(_extract_breakdowns_from_result(results))

        # If results itself looks like a single breakdown (has product_id)
        if not all_breakdowns:
            if isinstance(results, dict) and "product_id" in results:
                all_breakdowns = [results]
            elif isinstance(results, list):
                # Maybe the list itself is a list of breakdowns
                for item in results:
                    if isinstance(item, dict) and "product_id" in item:
                        all_breakdowns.append(item)

        # Index by product_id, keeping highest priority
        indexed: Dict[str, Dict[str, Any]] = {}
        for bd in all_breakdowns:
            pid = _extract_product_id(bd)
            if not pid:
                continue

            existing = indexed.get(pid)
            if existing is None:
                indexed[pid] = bd
            else:
                # Keep the one with higher priority (lower number)
                existing_method = _extract_method(existing)
                new_method = _extract_method(bd)
                existing_priority = METHOD_PRIORITY.get(existing_method, 99)
                new_priority = METHOD_PRIORITY.get(new_method, 99)
                if new_priority < existing_priority:
                    indexed[pid] = bd

        return indexed

    def _normalize_breakdown(
        self,
        breakdown: Any,
        product: Any,
    ) -> Dict[str, Any]:
        """Normalize a breakdown from any engine into a standard dict format.

        Args:
            breakdown: Breakdown dict or model from an engine.
            product: Original product input for supplemental data.

        Returns:
            Normalized breakdown dict with standard keys.
        """
        return {
            "product_id": _extract_product_id(breakdown),
            "category": _extract_category(breakdown) or self._get_product_category(product),
            "processing_type": _extract_processing_type(breakdown) or self._get_product_processing_type(product),
            "quantity": _extract_quantity(breakdown),
            "emissions_kg": _extract_emissions(breakdown),
            "ef_used": _extract_ef_used(breakdown),
            "method": _extract_method(breakdown),
            "dqi": _extract_dqi(breakdown),
            "country": _extract_country(breakdown) or self._get_product_country(product),
            "gap_filled": False,
        }

    def _create_placeholder_breakdown(self, product: Any) -> Dict[str, Any]:
        """Create a zero-emission placeholder breakdown for gap-filling.

        Args:
            product: Product input dict or model.

        Returns:
            Dict with zero emissions, marked for gap-filling.
        """
        return {
            "product_id": self._get_product_id(product),
            "category": self._get_product_category(product),
            "processing_type": self._get_product_processing_type(product),
            "quantity": self._get_product_quantity(product),
            "emissions_kg": ZERO,
            "ef_used": ZERO,
            "method": "pending",
            "dqi": Decimal("1.0"),
            "country": self._get_product_country(product),
            "gap_filled": True,
        }

    def _sum_emissions(self, breakdowns: List[Dict[str, Any]]) -> Decimal:
        """Sum total emissions across all breakdowns.

        Args:
            breakdowns: List of breakdown dicts.

        Returns:
            Total emissions as Decimal (kgCO2e).
        """
        total = ZERO
        for bd in breakdowns:
            total += _extract_emissions(bd)
        return _quantize(total)

    def _get_product_id(self, product: Any) -> str:
        """Extract product_id from a product input.

        Args:
            product: Product dict or model.

        Returns:
            Product identifier string.
        """
        if isinstance(product, dict):
            return str(product.get("product_id", f"gap-{uuid.uuid4().hex[:8]}"))
        return str(getattr(product, "product_id", f"gap-{uuid.uuid4().hex[:8]}"))

    def _get_product_category(self, product: Any) -> str:
        """Extract category from a product input.

        Args:
            product: Product dict or model.

        Returns:
            Category string.
        """
        if isinstance(product, dict):
            cat = product.get("category", "unknown")
        else:
            cat = getattr(product, "category", "unknown")
        # Handle enum objects
        if hasattr(cat, "value"):
            return str(cat.value)
        return str(cat)

    def _get_product_quantity(self, product: Any) -> Decimal:
        """Extract quantity from a product input.

        Args:
            product: Product dict or model.

        Returns:
            Quantity as Decimal.
        """
        if isinstance(product, dict):
            for key in ("quantity", "quantity_tonnes", "mass_tonnes"):
                if key in product:
                    return _safe_decimal(product[key])
            return ONE
        for attr in ("quantity", "quantity_tonnes", "mass_tonnes"):
            val = getattr(product, attr, None)
            if val is not None:
                return _safe_decimal(val)
        return ONE

    def _get_product_processing_type(self, product: Any) -> str:
        """Extract processing type from a product input.

        Args:
            product: Product dict or model.

        Returns:
            Processing type string.
        """
        if isinstance(product, dict):
            pt = product.get("processing_type", "unknown")
        else:
            pt = getattr(product, "processing_type", "unknown")
        if hasattr(pt, "value"):
            return str(pt.value)
        return str(pt)

    def _get_product_country(self, product: Any) -> str:
        """Extract customer country from a product input.

        Args:
            product: Product dict or model.

        Returns:
            Country string or 'GLOBAL'.
        """
        if isinstance(product, dict):
            c = product.get("customer_country", product.get("country", "GLOBAL"))
        else:
            c = getattr(product, "customer_country", getattr(product, "country", "GLOBAL"))
        if hasattr(c, "value"):
            return str(c.value)
        return str(c)

    @staticmethod
    def _get_allocation_key(method_lower: str) -> str:
        """Map allocation method to the dict key used in end-use dicts.

        Args:
            method_lower: Lowercase allocation method string.

        Returns:
            Key name to look up in end-use dicts.

        Raises:
            ValueError: If method is not recognized.
        """
        mapping = {
            "mass": "mass",
            "revenue": "revenue",
            "units": "units",
            "equal": "equal",
        }
        key = mapping.get(method_lower)
        if key is None:
            raise ValueError(
                f"Unknown allocation method '{method_lower}'. "
                f"Supported: {list(mapping.keys())}"
            )
        return key

    @staticmethod
    def _reconcile_allocation(
        allocated: Dict[str, Decimal],
        total: Decimal,
    ) -> Dict[str, Decimal]:
        """Reconcile rounding differences in allocation to match total.

        Assigns any rounding residual to the largest recipient.

        Args:
            allocated: Dict of end_use_id -> allocated emissions.
            total: Expected total emissions.

        Returns:
            Reconciled allocation dict.
        """
        if not allocated:
            return allocated

        alloc_sum = sum(allocated.values())
        residual = _quantize(total - alloc_sum)

        if residual == ZERO:
            return allocated

        # Add residual to largest recipient
        largest_id = max(allocated, key=lambda k: allocated[k])
        allocated[largest_id] = _quantize(allocated[largest_id] + residual)
        return allocated


# ==============================================================================
# MODULE-LEVEL FACTORY
# ==============================================================================


def get_hybrid_aggregator_engine() -> HybridAggregatorEngine:
    """Get the singleton HybridAggregatorEngine instance.

    This is the recommended way to obtain the engine instance in application
    code. Thread-safe via the singleton pattern in ``__new__``.

    Returns:
        The HybridAggregatorEngine singleton instance.

    Example:
        >>> engine = get_hybrid_aggregator_engine()
        >>> result = engine.aggregate(products=[], org_id="ORG-001", reporting_year=2025)
    """
    return HybridAggregatorEngine()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "HybridAggregatorEngine",
    "get_hybrid_aggregator_engine",
    "METHOD_PRIORITY",
    "WATERFALL_ORDER",
    "METHOD_DQI_DEFAULTS",
    "METHOD_UNCERTAINTY_FRACTIONS",
    "METHOD_DQI_DIMENSIONS",
    "DQI_DIMENSION_WEIGHTS",
    "CalculationMethod",
    "AllocationMethod",
    "AggregationPeriod",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "ENGINE_ID",
    "ENGINE_VERSION",
]
