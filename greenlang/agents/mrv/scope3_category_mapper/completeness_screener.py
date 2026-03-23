# -*- coding: utf-8 -*-
"""
CompletenessScreenerEngine - AGENT-MRV-029 Engine 5

This module implements Scope 3 category completeness screening, gap analysis,
and materiality assessment across all 15 GHG Protocol Scope 3 categories.

The engine evaluates which categories are relevant based on company type,
identifies data gaps, estimates materiality of missing categories using
industry benchmarks, and generates prioritized data collection recommendations.

Category Relevance Matrix:
    - Hardcoded relevance (MATERIAL / RELEVANT / NOT_RELEVANT) for 8 company
      types across all 15 Scope 3 categories.
    - Based on GHG Protocol Scope 3 Guidance, CDP sector guidance, and
      SBTi sector pathways.

Industry Benchmarks:
    - Typical percentage distribution of Scope 3 emissions by category for
      each company type, sourced from CDP aggregate data and GHG Protocol
      technical guidance.

Zero-Hallucination Guarantee:
    - All relevance mappings are deterministic lookup tables.
    - All benchmark values are hardcoded from authoritative sources.
    - No LLM or ML models are used for screening or scoring.

Example:
    >>> engine = CompletenessScreenerEngine.get_instance()
    >>> report = engine.screen_completeness(
    ...     company_type=CompanyType.MANUFACTURER,
    ...     categories_reported=[
    ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
    ...         Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
    ...     ],
    ...     data_by_category={
    ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: {
    ...             "emissions_kg": 500000
    ...         },
    ...     },
    ... )
    >>> print(f"Completeness: {report.overall_score}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from greenlang.agents.mrv.scope3_category_mapper.models import (
    ALL_SCOPE3_CATEGORIES,
    SCOPE3_CATEGORY_METADATA,
    SCOPE3_CATEGORY_NAMES,
    SCOPE3_CATEGORY_NUMBERS,
    BenchmarkComparison,
    CategoryCompletenessEntry,
    CategoryRelevance,
    CompanyType,
    CompletenessReport,
    DataQualityTier,
    Scope3Category,
    ScreeningResult,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "gl_scm_completeness_screener_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-X-040"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
ROUNDING: str = ROUND_HALF_UP

# Tolerance for benchmark comparison: deviations within +/- 10% are acceptable
BENCHMARK_TOLERANCE_PCT: Decimal = Decimal("10.00")

# Materiality threshold: categories estimated at >= 1% of total Scope 3
MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("1.00")


# ==============================================================================
# CATEGORY RELEVANCE MATRIX
# ==============================================================================

# Maps (CompanyType) -> {Scope3Category -> CategoryRelevance}
# Based on GHG Protocol Scope 3 Guidance, CDP sector guidance, SBTi pathways.
# Abbreviations: M=MATERIAL, R=RELEVANT, NR=NOT_RELEVANT

_M = CategoryRelevance.MATERIAL
_R = CategoryRelevance.RELEVANT
_NR = CategoryRelevance.NOT_RELEVANT

# Short aliases for enum members to keep the matrix readable
_C1 = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
_C2 = Scope3Category.CAT_2_CAPITAL_GOODS
_C3 = Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES
_C4 = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION
_C5 = Scope3Category.CAT_5_WASTE_GENERATED
_C6 = Scope3Category.CAT_6_BUSINESS_TRAVEL
_C7 = Scope3Category.CAT_7_EMPLOYEE_COMMUTING
_C8 = Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS
_C9 = Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION
_C10 = Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS
_C11 = Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS
_C12 = Scope3Category.CAT_12_END_OF_LIFE_TREATMENT
_C13 = Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS
_C14 = Scope3Category.CAT_14_FRANCHISES
_C15 = Scope3Category.CAT_15_INVESTMENTS


COMPANY_TYPE_RELEVANCE: Dict[CompanyType, Dict[Scope3Category, CategoryRelevance]] = {
    CompanyType.MANUFACTURER: {
        _C1: _M, _C2: _M, _C3: _R, _C4: _M, _C5: _R, _C6: _R, _C7: _R,
        _C8: _R, _C9: _M, _C10: _M, _C11: _M, _C12: _M, _C13: _NR,
        _C14: _NR, _C15: _NR,
    },
    CompanyType.SERVICES: {
        _C1: _M, _C2: _R, _C3: _R, _C4: _NR, _C5: _R, _C6: _M, _C7: _M,
        _C8: _M, _C9: _NR, _C10: _NR, _C11: _NR, _C12: _NR, _C13: _R,
        _C14: _NR, _C15: _R,
    },
    CompanyType.FINANCIAL: {
        _C1: _R, _C2: _R, _C3: _R, _C4: _NR, _C5: _NR, _C6: _M, _C7: _M,
        _C8: _M, _C9: _NR, _C10: _NR, _C11: _NR, _C12: _NR, _C13: _R,
        _C14: _NR, _C15: _M,
    },
    CompanyType.RETAILER: {
        _C1: _M, _C2: _R, _C3: _R, _C4: _M, _C5: _R, _C6: _R, _C7: _R,
        _C8: _M, _C9: _M, _C10: _NR, _C11: _R, _C12: _R, _C13: _R,
        _C14: _R, _C15: _NR,
    },
    CompanyType.ENERGY: {
        _C1: _M, _C2: _M, _C3: _M, _C4: _M, _C5: _R, _C6: _R, _C7: _R,
        _C8: _R, _C9: _M, _C10: _R, _C11: _M, _C12: _R, _C13: _R,
        _C14: _NR, _C15: _R,
    },
    CompanyType.MINING: {
        _C1: _M, _C2: _M, _C3: _M, _C4: _M, _C5: _M, _C6: _R, _C7: _R,
        _C8: _R, _C9: _M, _C10: _M, _C11: _NR, _C12: _NR, _C13: _NR,
        _C14: _NR, _C15: _NR,
    },
    CompanyType.AGRICULTURE: {
        _C1: _M, _C2: _R, _C3: _M, _C4: _M, _C5: _R, _C6: _NR, _C7: _R,
        _C8: _NR, _C9: _M, _C10: _M, _C11: _R, _C12: _R, _C13: _NR,
        _C14: _NR, _C15: _NR,
    },
    CompanyType.TRANSPORT: {
        _C1: _R, _C2: _M, _C3: _M, _C4: _NR, _C5: _R, _C6: _R, _C7: _R,
        _C8: _M, _C9: _M, _C10: _NR, _C11: _M, _C12: _NR, _C13: _M,
        _C14: _NR, _C15: _NR,
    },
}


# ==============================================================================
# INDUSTRY BENCHMARKS
# ==============================================================================

# Expected % distribution of Scope 3 emissions by category for each company
# type. Sourced from CDP aggregate sector data and GHG Protocol technical
# guidance. Values represent the typical proportion of total Scope 3 that
# falls into each category. Sums to 100% per company type.

INDUSTRY_BENCHMARKS: Dict[CompanyType, Dict[Scope3Category, Decimal]] = {
    CompanyType.MANUFACTURER: {
        _C1: Decimal("60.00"), _C2: Decimal("3.00"), _C3: Decimal("2.00"),
        _C4: Decimal("5.00"), _C5: Decimal("1.00"), _C6: Decimal("0.50"),
        _C7: Decimal("0.50"), _C8: Decimal("0.50"), _C9: Decimal("3.00"),
        _C10: Decimal("3.00"), _C11: Decimal("15.00"), _C12: Decimal("5.00"),
        _C13: Decimal("0.50"), _C14: Decimal("0.50"), _C15: Decimal("0.50"),
    },
    CompanyType.SERVICES: {
        _C1: Decimal("40.00"), _C2: Decimal("5.00"), _C3: Decimal("3.00"),
        _C4: Decimal("1.00"), _C5: Decimal("2.00"), _C6: Decimal("15.00"),
        _C7: Decimal("12.00"), _C8: Decimal("10.00"), _C9: Decimal("0.50"),
        _C10: Decimal("0.50"), _C11: Decimal("0.50"), _C12: Decimal("0.50"),
        _C13: Decimal("5.00"), _C14: Decimal("0.50"), _C15: Decimal("4.50"),
    },
    CompanyType.FINANCIAL: {
        _C1: Decimal("5.00"), _C2: Decimal("2.00"), _C3: Decimal("1.00"),
        _C4: Decimal("0.50"), _C5: Decimal("0.50"), _C6: Decimal("3.00"),
        _C7: Decimal("2.00"), _C8: Decimal("4.00"), _C9: Decimal("0.00"),
        _C10: Decimal("0.00"), _C11: Decimal("0.00"), _C12: Decimal("0.00"),
        _C13: Decimal("2.00"), _C14: Decimal("0.00"), _C15: Decimal("80.00"),
    },
    CompanyType.RETAILER: {
        _C1: Decimal("70.00"), _C2: Decimal("2.00"), _C3: Decimal("2.00"),
        _C4: Decimal("8.00"), _C5: Decimal("1.50"), _C6: Decimal("0.50"),
        _C7: Decimal("1.00"), _C8: Decimal("3.00"), _C9: Decimal("5.00"),
        _C10: Decimal("0.00"), _C11: Decimal("2.00"), _C12: Decimal("2.00"),
        _C13: Decimal("1.00"), _C14: Decimal("1.50"), _C15: Decimal("0.50"),
    },
    CompanyType.ENERGY: {
        _C1: Decimal("8.00"), _C2: Decimal("5.00"), _C3: Decimal("10.00"),
        _C4: Decimal("5.00"), _C5: Decimal("1.00"), _C6: Decimal("0.50"),
        _C7: Decimal("0.50"), _C8: Decimal("1.00"), _C9: Decimal("4.00"),
        _C10: Decimal("2.00"), _C11: Decimal("55.00"), _C12: Decimal("2.00"),
        _C13: Decimal("1.00"), _C14: Decimal("0.00"), _C15: Decimal("5.00"),
    },
    CompanyType.MINING: {
        _C1: Decimal("15.00"), _C2: Decimal("8.00"), _C3: Decimal("12.00"),
        _C4: Decimal("10.00"), _C5: Decimal("5.00"), _C6: Decimal("0.50"),
        _C7: Decimal("0.50"), _C8: Decimal("2.00"), _C9: Decimal("12.00"),
        _C10: Decimal("30.00"), _C11: Decimal("2.00"), _C12: Decimal("1.00"),
        _C13: Decimal("0.50"), _C14: Decimal("0.50"), _C15: Decimal("1.00"),
    },
    CompanyType.AGRICULTURE: {
        _C1: Decimal("35.00"), _C2: Decimal("3.00"), _C3: Decimal("8.00"),
        _C4: Decimal("10.00"), _C5: Decimal("2.00"), _C6: Decimal("0.50"),
        _C7: Decimal("1.00"), _C8: Decimal("0.50"), _C9: Decimal("12.00"),
        _C10: Decimal("15.00"), _C11: Decimal("5.00"), _C12: Decimal("5.00"),
        _C13: Decimal("0.50"), _C14: Decimal("0.50"), _C15: Decimal("1.50"),
    },
    CompanyType.TRANSPORT: {
        _C1: Decimal("5.00"), _C2: Decimal("20.00"), _C3: Decimal("30.00"),
        _C4: Decimal("1.00"), _C5: Decimal("1.00"), _C6: Decimal("1.00"),
        _C7: Decimal("2.00"), _C8: Decimal("10.00"), _C9: Decimal("5.00"),
        _C10: Decimal("0.00"), _C11: Decimal("15.00"), _C12: Decimal("0.00"),
        _C13: Decimal("8.00"), _C14: Decimal("0.00"), _C15: Decimal("2.00"),
    },
}


# ==============================================================================
# SERIALIZATION UTILITIES
# ==============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize an object to a deterministic JSON string for hashing.

    Handles Decimal, datetime, Enum, and Pydantic model types.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def _default_handler(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash.

    Returns:
        Lowercase hex SHA-256 hash string.
    """
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ==============================================================================
# CompletenessScreenerEngine
# ==============================================================================


class CompletenessScreenerEngine:
    """
    CompletenessScreenerEngine - screens Scope 3 category completeness.

    This engine evaluates which GHG Protocol Scope 3 categories are relevant
    to a given company type, identifies gaps in reported categories, estimates
    materiality of missing categories using industry benchmarks, and generates
    prioritized data collection recommendations.

    All relevance mappings and benchmarks are deterministic lookup tables.
    No LLM or ML models are used (zero-hallucination guarantee).

    Thread-Safe: Singleton pattern with lock for concurrent access.

    Attributes:
        _instance: Singleton instance.
        _lock: Thread lock for singleton creation.

    Example:
        >>> engine = CompletenessScreenerEngine.get_instance()
        >>> report = engine.screen_completeness(
        ...     company_type=CompanyType.MANUFACTURER,
        ...     categories_reported=[
        ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     ],
        ...     data_by_category={},
        ... )
        >>> assert report.overall_score >= Decimal("0")
    """

    _instance: Optional["CompletenessScreenerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize CompletenessScreenerEngine."""
        logger.info(
            "CompletenessScreenerEngine initialized (version=%s)",
            ENGINE_VERSION,
        )

    @classmethod
    def get_instance(cls) -> "CompletenessScreenerEngine":
        """
        Get singleton instance of CompletenessScreenerEngine (thread-safe).

        Returns:
            Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def screen_completeness(
        self,
        company_type: CompanyType,
        categories_reported: List[Scope3Category],
        data_by_category: Optional[Dict[Scope3Category, Dict[str, Any]]] = None,
    ) -> CompletenessReport:
        """
        Screen Scope 3 category completeness for a company.

        Evaluates all 15 categories against the company type relevance matrix,
        identifies gaps where material or relevant categories are not reported,
        and generates a scored report with recommendations.

        Args:
            company_type: Company type for relevance lookup.
            categories_reported: List of categories with reported data.
            data_by_category: Optional dict of category -> data payload
                for assessing data quality of reported categories.

        Returns:
            CompletenessReport with per-category details, gaps, score,
            and recommendations.

        Raises:
            ValueError: If company_type is not recognized.
        """
        start_time = time.monotonic()
        logger.info(
            "Screening completeness: company_type=%s, reported=%d categories",
            company_type.value,
            len(categories_reported),
        )

        if data_by_category is None:
            data_by_category = {}

        if company_type not in COMPANY_TYPE_RELEVANCE:
            raise ValueError(
                f"Unknown company type: {company_type.value}. "
                f"Supported: {[ct.value for ct in COMPANY_TYPE_RELEVANCE]}"
            )

        reported_set = set(categories_reported)
        relevance_map = COMPANY_TYPE_RELEVANCE[company_type]
        benchmarks = INDUSTRY_BENCHMARKS.get(company_type, {})

        # Build per-category entries
        entries: List[CategoryCompletenessEntry] = []
        gaps: List[str] = []
        material_count = 0
        material_reported_count = 0

        for cat in ALL_SCOPE3_CATEGORIES:
            relevance = relevance_map.get(cat, CategoryRelevance.UNKNOWN)
            is_reported = cat in reported_set
            benchmark_pct = benchmarks.get(cat, Decimal("0.00"))
            materiality_pct = self.estimate_materiality(company_type, cat)
            cat_name = SCOPE3_CATEGORY_NAMES.get(cat, cat.value)
            cat_num = SCOPE3_CATEGORY_NUMBERS.get(cat, 0)

            # Count material categories
            if relevance == CategoryRelevance.MATERIAL:
                material_count += 1
                if is_reported:
                    material_reported_count += 1

            # Determine screening result
            screening_result = self._determine_screening_result(
                relevance, is_reported, data_by_category.get(cat)
            )

            # Assess data quality tier
            data_quality_tier = self._assess_data_quality_tier(
                data_by_category.get(cat)
            )

            # Generate recommended action
            recommended_action: Optional[str] = None
            if screening_result == ScreeningResult.MISSING:
                gap_desc = (
                    f"Category {cat_num} ({cat_name}) is {relevance.value} "
                    f"but has no reported data. "
                    f"Estimated {materiality_pct}% of total Scope 3."
                )
                gaps.append(gap_desc)
                recommended_action = (
                    f"Collect data for Category {cat_num} ({cat_name}). "
                    f"Engage {self._get_data_source_hint(cat)}."
                )
            elif screening_result == ScreeningResult.PARTIAL:
                gap_desc = (
                    f"Category {cat_num} ({cat_name}) has partial data "
                    f"coverage. Data quality improvement recommended."
                )
                gaps.append(gap_desc)
                recommended_action = (
                    f"Improve data coverage for Category {cat_num} "
                    f"({cat_name}). Target supplier-specific data."
                )

            entry = CategoryCompletenessEntry(
                category=cat,
                relevance=relevance,
                data_available=is_reported,
                data_quality_tier=data_quality_tier,
                estimated_materiality_pct=materiality_pct,
                screening_result=screening_result,
                recommended_action=recommended_action,
            )
            entries.append(entry)

        # Calculate completeness score using entries
        overall_score = self._calculate_score_from_entries(entries)

        # Compute provenance
        provenance_hash = self._compute_provenance_hash({
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "company_type": company_type.value,
            "categories_reported": [c.value for c in categories_reported],
            "overall_score": str(overall_score),
            "material_count": material_count,
            "material_reported_count": material_reported_count,
        })

        processing_time = (time.monotonic() - start_time) * 1000

        report = CompletenessReport(
            company_type=company_type,
            entries=entries,
            overall_score=overall_score,
            categories_reported=len(reported_set),
            categories_material=material_count,
            gaps=gaps,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Completeness screening done: score=%s, gaps=%d, time=%.1fms",
            overall_score,
            len(gaps),
            processing_time,
        )
        return report

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def assess_category_relevance(
        self,
        company_type: CompanyType,
        category: Scope3Category,
    ) -> CategoryRelevance:
        """
        Assess the relevance of a single category for a company type.

        Args:
            company_type: Company type.
            category: Scope 3 category to assess.

        Returns:
            CategoryRelevance (MATERIAL, RELEVANT, NOT_RELEVANT, or UNKNOWN).

        Raises:
            ValueError: If company_type is not in the relevance matrix.
        """
        if company_type not in COMPANY_TYPE_RELEVANCE:
            raise ValueError(
                f"Unknown company type: {company_type.value}. "
                f"Supported: {[ct.value for ct in COMPANY_TYPE_RELEVANCE]}"
            )

        relevance_map = COMPANY_TYPE_RELEVANCE[company_type]
        return relevance_map.get(category, CategoryRelevance.UNKNOWN)

    def identify_gaps(
        self,
        company_type: CompanyType,
        categories_reported: List[Scope3Category],
    ) -> List[str]:
        """
        Identify gaps where material or relevant categories are not reported.

        Args:
            company_type: Company type for relevance lookup.
            categories_reported: List of categories with data.

        Returns:
            List of gap description strings, sorted by materiality
            (highest first).

        Raises:
            ValueError: If company_type is not recognized.
        """
        if company_type not in COMPANY_TYPE_RELEVANCE:
            raise ValueError(
                f"Unknown company type: {company_type.value}"
            )

        reported_set = set(categories_reported)
        relevance_map = COMPANY_TYPE_RELEVANCE[company_type]
        gap_entries: List[Dict[str, Any]] = []

        for cat in ALL_SCOPE3_CATEGORIES:
            relevance = relevance_map.get(cat, CategoryRelevance.UNKNOWN)
            if cat not in reported_set and relevance in (
                CategoryRelevance.MATERIAL,
                CategoryRelevance.RELEVANT,
            ):
                materiality = self.estimate_materiality(company_type, cat)
                cat_name = SCOPE3_CATEGORY_NAMES.get(cat, cat.value)
                cat_num = SCOPE3_CATEGORY_NUMBERS.get(cat, 0)
                gap_entries.append({
                    "materiality": materiality,
                    "description": (
                        f"Category {cat_num} ({cat_name}): "
                        f"{relevance.value} category not reported. "
                        f"Estimated {materiality}% of total Scope 3."
                    ),
                })

        # Sort by materiality descending (highest gaps first)
        gap_entries.sort(key=lambda g: g["materiality"], reverse=True)
        return [g["description"] for g in gap_entries]

    def estimate_materiality(
        self,
        company_type: CompanyType,
        category: Scope3Category,
    ) -> Decimal:
        """
        Estimate the materiality of a category as % of total Scope 3.

        Uses industry benchmark data as the best estimate for expected
        category contribution.

        Args:
            company_type: Company type.
            category: Scope 3 category.

        Returns:
            Estimated percentage (0-100) of total Scope 3 emissions.
        """
        benchmarks = INDUSTRY_BENCHMARKS.get(company_type, {})
        return benchmarks.get(category, Decimal("0.00"))

    def get_industry_benchmark(
        self,
        company_type: CompanyType,
    ) -> Dict[Scope3Category, Decimal]:
        """
        Get the industry benchmark distribution for a company type.

        Returns the expected percentage distribution of Scope 3 emissions
        across all 15 categories for the given company type.

        Args:
            company_type: Company type.

        Returns:
            Dictionary mapping each category to its benchmark percentage.

        Raises:
            ValueError: If company_type has no benchmark data.
        """
        if company_type not in INDUSTRY_BENCHMARKS:
            raise ValueError(
                f"No industry benchmarks for company type: "
                f"{company_type.value}"
            )
        return dict(INDUSTRY_BENCHMARKS[company_type])

    def calculate_completeness_score(
        self,
        report: CompletenessReport,
    ) -> Decimal:
        """
        Calculate the completeness score (0-100) for a report.

        Delegates to _calculate_score_from_entries using the report's
        entry list.

        Args:
            report: CompletenessReport to score.

        Returns:
            Decimal score in range [0, 100].
        """
        return self._calculate_score_from_entries(report.entries)

    def recommend_actions(
        self,
        report: CompletenessReport,
    ) -> List[str]:
        """
        Generate priority-ordered data collection recommendations.

        Actions are prioritized by:
        1. Missing material categories (highest priority)
        2. Partial material categories (data quality improvement)
        3. Missing relevant categories
        4. Partial relevant categories
        5. General improvement actions

        Args:
            report: CompletenessReport to generate actions for.

        Returns:
            List of recommendation strings, priority-ordered.
        """
        actions: List[str] = []
        missing_material: List[CategoryCompletenessEntry] = []
        partial_material: List[CategoryCompletenessEntry] = []
        missing_relevant: List[CategoryCompletenessEntry] = []
        partial_relevant: List[CategoryCompletenessEntry] = []

        for entry in report.entries:
            if entry.relevance == CategoryRelevance.MATERIAL:
                if entry.screening_result == ScreeningResult.MISSING:
                    missing_material.append(entry)
                elif entry.screening_result == ScreeningResult.PARTIAL:
                    partial_material.append(entry)
            elif entry.relevance == CategoryRelevance.RELEVANT:
                if entry.screening_result == ScreeningResult.MISSING:
                    missing_relevant.append(entry)
                elif entry.screening_result == ScreeningResult.PARTIAL:
                    partial_relevant.append(entry)

        # Sort each group by estimated materiality descending
        def _mat_key(e: CategoryCompletenessEntry) -> Decimal:
            return e.estimated_materiality_pct or Decimal("0")

        missing_material.sort(key=_mat_key, reverse=True)
        partial_material.sort(key=_mat_key, reverse=True)
        missing_relevant.sort(key=_mat_key, reverse=True)
        partial_relevant.sort(key=_mat_key, reverse=True)

        for entry in missing_material:
            cat_name = SCOPE3_CATEGORY_NAMES.get(entry.category, "")
            cat_num = SCOPE3_CATEGORY_NUMBERS.get(entry.category, 0)
            mat_pct = entry.estimated_materiality_pct or Decimal("0")
            actions.append(
                f"[CRITICAL] Collect data for Category {cat_num} "
                f"({cat_name}). This is a MATERIAL category estimated at "
                f"{mat_pct}% of total Scope 3. Engage "
                f"{self._get_data_source_hint(entry.category)}."
            )

        for entry in partial_material:
            cat_name = SCOPE3_CATEGORY_NAMES.get(entry.category, "")
            cat_num = SCOPE3_CATEGORY_NUMBERS.get(entry.category, 0)
            actions.append(
                f"[HIGH] Improve data coverage for Category {cat_num} "
                f"({cat_name}). Current data is incomplete. Target "
                f"supplier-specific data to improve quality."
            )

        for entry in missing_relevant:
            cat_name = SCOPE3_CATEGORY_NAMES.get(entry.category, "")
            cat_num = SCOPE3_CATEGORY_NUMBERS.get(entry.category, 0)
            mat_pct = entry.estimated_materiality_pct or Decimal("0")
            actions.append(
                f"[MEDIUM] Collect data for Category {cat_num} "
                f"({cat_name}). This is a RELEVANT category estimated at "
                f"{mat_pct}% of total Scope 3."
            )

        for entry in partial_relevant:
            cat_name = SCOPE3_CATEGORY_NAMES.get(entry.category, "")
            cat_num = SCOPE3_CATEGORY_NUMBERS.get(entry.category, 0)
            actions.append(
                f"[LOW] Improve data quality for Category {cat_num} "
                f"({cat_name}). Consider upgrading from spend-based to "
                f"activity-based data."
            )

        # General action if material categories are missing
        material_reported = sum(
            1 for e in report.entries
            if e.relevance == CategoryRelevance.MATERIAL and e.data_available
        )
        if material_reported < report.categories_material:
            material_gap = report.categories_material - material_reported
            actions.append(
                f"[GENERAL] {material_gap} material category(ies) are "
                f"missing. Address these before next reporting cycle to "
                f"meet GHG Protocol and SBTi completeness requirements."
            )

        return actions

    def compare_to_benchmark(
        self,
        company_type: CompanyType,
        actual_distribution: Dict[Scope3Category, Decimal],
    ) -> Dict[Scope3Category, BenchmarkComparison]:
        """
        Compare actual emissions distribution against industry benchmarks.

        Identifies categories where the actual percentage deviates
        significantly from the expected benchmark, which may indicate
        data quality issues or misclassification.

        Args:
            company_type: Company type for benchmark lookup.
            actual_distribution: Actual percentage distribution by category.

        Returns:
            Dictionary mapping each category to its BenchmarkComparison.

        Raises:
            ValueError: If company_type has no benchmark data.
        """
        if company_type not in INDUSTRY_BENCHMARKS:
            raise ValueError(
                f"No industry benchmarks for company type: "
                f"{company_type.value}"
            )

        benchmarks = INDUSTRY_BENCHMARKS[company_type]
        comparisons: Dict[Scope3Category, BenchmarkComparison] = {}

        for cat in ALL_SCOPE3_CATEGORIES:
            benchmark_pct = benchmarks.get(cat, Decimal("0.00"))
            actual_pct = actual_distribution.get(cat, Decimal("0.00"))
            deviation = (actual_pct - benchmark_pct).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            abs_deviation = abs(deviation)
            within_tolerance = abs_deviation <= BENCHMARK_TOLERANCE_PCT

            flag: Optional[str] = None
            if not within_tolerance:
                cat_name = SCOPE3_CATEGORY_NAMES.get(cat, cat.value)
                cat_num = SCOPE3_CATEGORY_NUMBERS.get(cat, 0)
                if deviation > Decimal("0"):
                    flag = (
                        f"Category {cat_num} ({cat_name}) is "
                        f"{abs_deviation}% ABOVE benchmark. Verify data "
                        f"source and classification."
                    )
                else:
                    flag = (
                        f"Category {cat_num} ({cat_name}) is "
                        f"{abs_deviation}% BELOW benchmark. Possible "
                        f"data gap or under-reporting."
                    )

            comparisons[cat] = BenchmarkComparison(
                category=cat,
                benchmark_pct=benchmark_pct,
                actual_pct=actual_pct,
                deviation_pct=deviation,
                within_tolerance=within_tolerance,
                flag=flag,
            )

        logger.info(
            "Benchmark comparison: company_type=%s, flags=%d",
            company_type.value,
            sum(1 for c in comparisons.values() if c.flag is not None),
        )
        return comparisons

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _compute_provenance_hash(self, data: Any) -> str:
        """
        Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            Lowercase hex SHA-256 hash string.
        """
        return _compute_hash(data)

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _calculate_score_from_entries(
        self,
        entries: List[CategoryCompletenessEntry],
    ) -> Decimal:
        """
        Calculate completeness score (0-100) from category entries.

        Scoring algorithm:
        - Material categories reported: 60% weight
        - Relevant categories reported: 30% weight
        - Not-relevant categories (bonus for reporting): 10% weight

        Args:
            entries: List of per-category completeness entries.

        Returns:
            Decimal score in range [0, 100].
        """
        material_total = Decimal("0")
        material_reported = Decimal("0")
        relevant_total = Decimal("0")
        relevant_reported = Decimal("0")
        nr_total = Decimal("0")
        nr_reported = Decimal("0")

        for entry in entries:
            if entry.relevance == CategoryRelevance.MATERIAL:
                material_total += Decimal("1")
                if entry.data_available:
                    material_reported += Decimal("1")
            elif entry.relevance == CategoryRelevance.RELEVANT:
                relevant_total += Decimal("1")
                if entry.data_available:
                    relevant_reported += Decimal("1")
            elif entry.relevance == CategoryRelevance.NOT_RELEVANT:
                nr_total += Decimal("1")
                if entry.data_available:
                    nr_reported += Decimal("1")

        # Component scores (each 0.0 to 1.0)
        m_score = (
            material_reported / material_total
            if material_total > 0
            else Decimal("1")
        )
        r_score = (
            relevant_reported / relevant_total
            if relevant_total > 0
            else Decimal("1")
        )
        nr_score = (
            nr_reported / nr_total
            if nr_total > 0
            else Decimal("1")
        )

        # Weighted total: material=60%, relevant=30%, not_relevant=10%
        weighted = (
            m_score * Decimal("60")
            + r_score * Decimal("30")
            + nr_score * Decimal("10")
        )

        # Clamp
        if weighted < Decimal("0"):
            weighted = Decimal("0")
        if weighted > Decimal("100"):
            weighted = Decimal("100")

        return weighted.quantize(_QUANT_2DP, rounding=ROUNDING)

    def _determine_screening_result(
        self,
        relevance: CategoryRelevance,
        is_reported: bool,
        category_data: Optional[Dict[str, Any]],
    ) -> ScreeningResult:
        """
        Determine screening result for a single category.

        Args:
            relevance: Category relevance level.
            is_reported: Whether the category has reported data.
            category_data: Optional data payload for quality assessment.

        Returns:
            ScreeningResult (COMPLETE, PARTIAL, or MISSING).
        """
        if relevance == CategoryRelevance.NOT_RELEVANT:
            return ScreeningResult.COMPLETE

        if not is_reported:
            return ScreeningResult.MISSING

        if category_data is not None:
            has_emissions = (
                "emissions_kg" in category_data
                or "emissions_tco2e" in category_data
                or "total_emissions" in category_data
            )
            has_method = "calculation_method" in category_data
            if has_emissions and has_method:
                return ScreeningResult.COMPLETE
            return ScreeningResult.PARTIAL

        return ScreeningResult.COMPLETE

    def _assess_data_quality_tier(
        self,
        category_data: Optional[Dict[str, Any]],
    ) -> Optional[DataQualityTier]:
        """
        Assess data quality tier from category data payload.

        Args:
            category_data: Data payload with optional quality indicators.

        Returns:
            DataQualityTier if assessment is possible, None otherwise.
        """
        if category_data is None:
            return None

        explicit_tier = category_data.get("data_quality_tier")
        if explicit_tier is not None:
            try:
                return DataQualityTier(explicit_tier)
            except ValueError:
                pass

        method = category_data.get("calculation_method", "")
        tier_map: Dict[str, DataQualityTier] = {
            "supplier_specific": DataQualityTier.TIER_1,
            "reported_verified": DataQualityTier.TIER_1,
            "activity_based": DataQualityTier.TIER_2,
            "physical_activity": DataQualityTier.TIER_2,
            "average_data": DataQualityTier.TIER_3,
            "hybrid": DataQualityTier.TIER_3,
            "spend_based": DataQualityTier.TIER_4,
            "revenue_eeio": DataQualityTier.TIER_4,
            "estimated": DataQualityTier.TIER_5,
            "proxy": DataQualityTier.TIER_5,
            "sector_average": DataQualityTier.TIER_5,
        }
        return tier_map.get(method)

    def _get_data_source_hint(self, category: Scope3Category) -> str:
        """
        Get a hint about likely data sources for a category.

        Args:
            category: Scope 3 category.

        Returns:
            Data source suggestion string.
        """
        hints: Dict[Scope3Category, str] = {
            _C1: (
                "procurement/AP systems for spend data; "
                "suppliers for product-level emission factors"
            ),
            _C2: "fixed asset register and capex records",
            _C3: (
                "energy invoices and utility bills "
                "(WTT factors from DEFRA/IEA)"
            ),
            _C4: (
                "freight invoices, logistics providers, "
                "shipping records (tkm data preferred)"
            ),
            _C5: (
                "waste manifests, disposal records, "
                "waste management contractors"
            ),
            _C6: (
                "travel management company data, "
                "expense reports, booking systems"
            ),
            _C7: (
                "employee commuting surveys, "
                "HR/payroll records for headcount"
            ),
            _C8: (
                "lease agreements, landlord utility data, "
                "building energy benchmarks"
            ),
            _C9: (
                "distribution/logistics records, "
                "3PL provider data"
            ),
            _C10: (
                "customer processing data, "
                "industry processing benchmarks"
            ),
            _C11: (
                "product energy ratings, lifetime use profiles, "
                "product specification sheets"
            ),
            _C12: (
                "product material composition, "
                "regional waste treatment statistics"
            ),
            _C13: (
                "tenant utility data, "
                "lease agreements, building benchmarks"
            ),
            _C14: (
                "franchise unit energy data, "
                "franchise agreements"
            ),
            _C15: (
                "investee company emissions data, "
                "PCAF-aligned financial data (EVIC, AUM)"
            ),
        }
        return hints.get(category, "relevant internal data systems")
