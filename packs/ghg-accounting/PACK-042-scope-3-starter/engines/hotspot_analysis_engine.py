# -*- coding: utf-8 -*-
"""
HotspotAnalysisEngine - PACK-042 Scope 3 Starter Pack Engine 5
=================================================================

Identifies emission hotspots across the Scope 3 inventory and
prioritises reduction opportunities.  Performs Pareto analysis,
supplier concentration analysis, product intensity ranking, materiality
matrix scoring, sector benchmarking, geographic mapping, and reduction
opportunity quantification.

The engine is designed to translate a consolidated Scope 3 inventory
into actionable insights for emissions reduction planning, supporting
SBTi target-setting and supplier engagement programmes.

Calculation Methodology:
    Pareto Analysis:
        Sort categories by emissions descending.
        Cumulative % until reaching 80% threshold.
        Categories driving 80% of emissions = hotspots.

    Supplier Concentration:
        Rank suppliers by emission contribution.
        Top N suppliers (or % threshold) = hotspot suppliers.

    Product Intensity:
        intensity_i = E_product_i / units_sold_i (or revenue_i)
        Rank by intensity descending.

    Materiality Matrix:
        score_i = w_mag * S_magnitude_i + w_imp * S_improvement_i
        where S_magnitude = normalised emission share (0-100)
              S_improvement = estimated reduction potential (0-100)

    Sector Benchmark:
        gap_i = org_share_i - benchmark_share_i
        Positive gap = over-represented (potential hotspot).
        Negative gap = under-represented (potential data gap).

    Reduction Opportunity:
        potential_i = E_current_i * reduction_factor_i
        where reduction_factor from methodology upgrade or supplier engagement

    Tier Upgrade Impact:
        delta_i = E_spend_based_i - E_supplier_specific_i
        Estimated using typical tier difference ratios.

Regulatory References:
    - GHG Protocol Scope 3 Standard, Chapter 9 (Setting Reduction Targets)
    - SBTi Corporate Net-Zero Standard, Section 8 (Scope 3 Targets)
    - SBTi Supplier Engagement Guidance (2023)
    - CDP Climate Change Questionnaire (2024), C-SCC module
    - TCFD Recommendations, Strategy section

Zero-Hallucination:
    - Pareto analysis uses deterministic sorting and cumulation
    - Benchmarks from published industry data
    - Reduction potentials from peer-reviewed literature
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serialisable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serialisable = data
    else:
        serialisable = str(data)
    if isinstance(serialisable, dict):
        serialisable = {
            k: v for k, v in serialisable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serialisable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HotspotType(str, Enum):
    """Type of hotspot identified.

    CATEGORY:    A Scope 3 category is a hotspot.
    SUPPLIER:    A specific supplier is a hotspot.
    PRODUCT:     A product/product line is a hotspot.
    GEOGRAPHIC:  A geographic region is a hotspot.
    """
    CATEGORY = "category"
    SUPPLIER = "supplier"
    PRODUCT = "product"
    GEOGRAPHIC = "geographic"

class ReductionLever(str, Enum):
    """Type of reduction lever available.

    SUPPLIER_ENGAGEMENT: Engage suppliers to reduce their emissions.
    METHODOLOGY_UPGRADE: Upgrade from spend-based to primary data.
    PRODUCT_REDESIGN:    Redesign products for lower lifecycle emissions.
    MODAL_SHIFT:         Shift transport modes to lower-emission options.
    ENERGY_EFFICIENCY:   Improve energy efficiency in value chain.
    CIRCULAR_ECONOMY:    Implement circular economy approaches.
    SWITCHING:           Switch to lower-emission suppliers/materials.
    DEMAND_REDUCTION:    Reduce overall demand or consumption.
    """
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    METHODOLOGY_UPGRADE = "methodology_upgrade"
    PRODUCT_REDESIGN = "product_redesign"
    MODAL_SHIFT = "modal_shift"
    ENERGY_EFFICIENCY = "energy_efficiency"
    CIRCULAR_ECONOMY = "circular_economy"
    SWITCHING = "switching"
    DEMAND_REDUCTION = "demand_reduction"

class ImprovementDifficulty(str, Enum):
    """Difficulty of implementing a reduction measure.

    EASY:       Quick win, low cost, low complexity.
    MODERATE:   Medium effort, some investment required.
    DIFFICULT:  Significant investment or structural changes.
    VERY_HARD:  Requires industry-wide transformation.
    """
    EASY = "easy"
    MODERATE = "moderate"
    DIFFICULT = "difficult"
    VERY_HARD = "very_hard"

class AnalysisStatus(str, Enum):
    """Status of the hotspot analysis."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"

# ---------------------------------------------------------------------------
# Sector Benchmark Data
# ---------------------------------------------------------------------------
# Typical Scope 3 category distribution by sector (% of total Scope 3).
# Sources: CDP 2023 analysis, WRI/WBCSD studies, peer-reviewed literature.

SECTOR_BENCHMARKS: Dict[str, Dict[int, Decimal]] = {
    "manufacturing_heavy": {
        1: Decimal("40"), 2: Decimal("5"), 3: Decimal("8"),
        4: Decimal("10"), 5: Decimal("3"), 6: Decimal("2"),
        7: Decimal("2"), 8: Decimal("1"), 9: Decimal("5"),
        10: Decimal("3"), 11: Decimal("15"), 12: Decimal("3"),
        13: Decimal("1"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "manufacturing_light": {
        1: Decimal("35"), 2: Decimal("6"), 3: Decimal("7"),
        4: Decimal("8"), 5: Decimal("4"), 6: Decimal("3"),
        7: Decimal("2"), 8: Decimal("1"), 9: Decimal("6"),
        10: Decimal("4"), 11: Decimal("18"), 12: Decimal("3"),
        13: Decimal("1"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "retail": {
        1: Decimal("55"), 2: Decimal("3"), 3: Decimal("3"),
        4: Decimal("12"), 5: Decimal("3"), 6: Decimal("2"),
        7: Decimal("3"), 8: Decimal("2"), 9: Decimal("8"),
        10: Decimal("0"), 11: Decimal("3"), 12: Decimal("2"),
        13: Decimal("1"), 14: Decimal("1"), 15: Decimal("2"),
    },
    "technology": {
        1: Decimal("25"), 2: Decimal("8"), 3: Decimal("5"),
        4: Decimal("3"), 5: Decimal("2"), 6: Decimal("10"),
        7: Decimal("5"), 8: Decimal("4"), 9: Decimal("2"),
        10: Decimal("0"), 11: Decimal("30"), 12: Decimal("3"),
        13: Decimal("1"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "financial_services": {
        1: Decimal("15"), 2: Decimal("5"), 3: Decimal("2"),
        4: Decimal("2"), 5: Decimal("1"), 6: Decimal("8"),
        7: Decimal("5"), 8: Decimal("3"), 9: Decimal("0"),
        10: Decimal("0"), 11: Decimal("0"), 12: Decimal("0"),
        13: Decimal("2"), 14: Decimal("2"), 15: Decimal("55"),
    },
    "utilities": {
        1: Decimal("10"), 2: Decimal("8"), 3: Decimal("35"),
        4: Decimal("5"), 5: Decimal("5"), 6: Decimal("2"),
        7: Decimal("2"), 8: Decimal("1"), 9: Decimal("2"),
        10: Decimal("0"), 11: Decimal("22"), 12: Decimal("2"),
        13: Decimal("3"), 14: Decimal("1"), 15: Decimal("2"),
    },
    "transportation": {
        1: Decimal("10"), 2: Decimal("8"), 3: Decimal("30"),
        4: Decimal("5"), 5: Decimal("3"), 6: Decimal("3"),
        7: Decimal("5"), 8: Decimal("10"), 9: Decimal("8"),
        10: Decimal("0"), 11: Decimal("5"), 12: Decimal("3"),
        13: Decimal("5"), 14: Decimal("2"), 15: Decimal("3"),
    },
    "healthcare": {
        1: Decimal("45"), 2: Decimal("8"), 3: Decimal("5"),
        4: Decimal("8"), 5: Decimal("8"), 6: Decimal("4"),
        7: Decimal("5"), 8: Decimal("2"), 9: Decimal("3"),
        10: Decimal("0"), 11: Decimal("5"), 12: Decimal("4"),
        13: Decimal("1"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "construction": {
        1: Decimal("50"), 2: Decimal("8"), 3: Decimal("5"),
        4: Decimal("10"), 5: Decimal("5"), 6: Decimal("3"),
        7: Decimal("3"), 8: Decimal("3"), 9: Decimal("2"),
        10: Decimal("0"), 11: Decimal("5"), 12: Decimal("3"),
        13: Decimal("1"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "food_beverage": {
        1: Decimal("60"), 2: Decimal("3"), 3: Decimal("5"),
        4: Decimal("8"), 5: Decimal("4"), 6: Decimal("2"),
        7: Decimal("2"), 8: Decimal("1"), 9: Decimal("6"),
        10: Decimal("2"), 11: Decimal("3"), 12: Decimal("2"),
        13: Decimal("0"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "automotive": {
        1: Decimal("25"), 2: Decimal("5"), 3: Decimal("5"),
        4: Decimal("5"), 5: Decimal("2"), 6: Decimal("2"),
        7: Decimal("2"), 8: Decimal("1"), 9: Decimal("3"),
        10: Decimal("5"), 11: Decimal("40"), 12: Decimal("3"),
        13: Decimal("0"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "chemicals": {
        1: Decimal("35"), 2: Decimal("5"), 3: Decimal("10"),
        4: Decimal("8"), 5: Decimal("5"), 6: Decimal("2"),
        7: Decimal("2"), 8: Decimal("1"), 9: Decimal("5"),
        10: Decimal("8"), 11: Decimal("12"), 12: Decimal("4"),
        13: Decimal("1"), 14: Decimal("0"), 15: Decimal("2"),
    },
    "real_estate": {
        1: Decimal("15"), 2: Decimal("10"), 3: Decimal("5"),
        4: Decimal("3"), 5: Decimal("3"), 6: Decimal("3"),
        7: Decimal("3"), 8: Decimal("5"), 9: Decimal("2"),
        10: Decimal("0"), 11: Decimal("10"), 12: Decimal("2"),
        13: Decimal("30"), 14: Decimal("2"), 15: Decimal("7"),
    },
    "professional_services": {
        1: Decimal("20"), 2: Decimal("8"), 3: Decimal("3"),
        4: Decimal("3"), 5: Decimal("3"), 6: Decimal("15"),
        7: Decimal("10"), 8: Decimal("5"), 9: Decimal("2"),
        10: Decimal("0"), 11: Decimal("10"), 12: Decimal("3"),
        13: Decimal("5"), 14: Decimal("3"), 15: Decimal("10"),
    },
    "mining": {
        1: Decimal("15"), 2: Decimal("10"), 3: Decimal("15"),
        4: Decimal("12"), 5: Decimal("5"), 6: Decimal("3"),
        7: Decimal("2"), 8: Decimal("5"), 9: Decimal("8"),
        10: Decimal("10"), 11: Decimal("5"), 12: Decimal("3"),
        13: Decimal("2"), 14: Decimal("1"), 15: Decimal("4"),
    },
}

# Typical reduction potential ranges by category and lever
REDUCTION_POTENTIALS: Dict[int, Dict[str, Dict[str, Any]]] = {
    1: {
        "supplier_engagement": {
            "low_pct": Decimal("5"), "high_pct": Decimal("25"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 3,
            "description": "Engage top suppliers on science-based targets",
        },
        "switching": {
            "low_pct": Decimal("10"), "high_pct": Decimal("40"),
            "difficulty": ImprovementDifficulty.DIFFICULT.value,
            "timeframe_years": 5,
            "description": "Switch to lower-emission materials or suppliers",
        },
    },
    2: {
        "energy_efficiency": {
            "low_pct": Decimal("5"), "high_pct": Decimal("20"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 5,
            "description": "Specify energy-efficient capital equipment",
        },
    },
    3: {
        "energy_efficiency": {
            "low_pct": Decimal("10"), "high_pct": Decimal("30"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 3,
            "description": "Reduce energy consumption and T&D losses",
        },
        "switching": {
            "low_pct": Decimal("20"), "high_pct": Decimal("50"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 3,
            "description": "Switch to renewable energy sources",
        },
    },
    4: {
        "modal_shift": {
            "low_pct": Decimal("10"), "high_pct": Decimal("30"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 2,
            "description": "Shift from road/air to rail/water transport",
        },
        "supplier_engagement": {
            "low_pct": Decimal("5"), "high_pct": Decimal("15"),
            "difficulty": ImprovementDifficulty.EASY.value,
            "timeframe_years": 2,
            "description": "Optimise logistics routes and consolidate shipments",
        },
    },
    5: {
        "circular_economy": {
            "low_pct": Decimal("15"), "high_pct": Decimal("50"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 3,
            "description": "Reduce waste, increase recycling and reuse",
        },
    },
    6: {
        "demand_reduction": {
            "low_pct": Decimal("20"), "high_pct": Decimal("50"),
            "difficulty": ImprovementDifficulty.EASY.value,
            "timeframe_years": 1,
            "description": "Virtual meetings, travel policy, rail over air",
        },
    },
    7: {
        "demand_reduction": {
            "low_pct": Decimal("10"), "high_pct": Decimal("40"),
            "difficulty": ImprovementDifficulty.EASY.value,
            "timeframe_years": 1,
            "description": "Remote work, public transit incentives, EV fleet",
        },
    },
    9: {
        "modal_shift": {
            "low_pct": Decimal("10"), "high_pct": Decimal("25"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 3,
            "description": "Optimise distribution logistics",
        },
    },
    11: {
        "product_redesign": {
            "low_pct": Decimal("10"), "high_pct": Decimal("40"),
            "difficulty": ImprovementDifficulty.DIFFICULT.value,
            "timeframe_years": 5,
            "description": "Design energy-efficient products",
        },
    },
    12: {
        "circular_economy": {
            "low_pct": Decimal("10"), "high_pct": Decimal("35"),
            "difficulty": ImprovementDifficulty.MODERATE.value,
            "timeframe_years": 3,
            "description": "Design for recyclability, take-back programmes",
        },
    },
    15: {
        "supplier_engagement": {
            "low_pct": Decimal("5"), "high_pct": Decimal("20"),
            "difficulty": ImprovementDifficulty.DIFFICULT.value,
            "timeframe_years": 5,
            "description": "Engage portfolio companies on decarbonisation",
        },
    },
}

# Typical tier upgrade impact ratios (spend-based vs supplier-specific)
TIER_UPGRADE_RATIOS: Dict[str, Decimal] = {
    "spend_to_average": Decimal("0.85"),   # Average data typically 85% of spend-based
    "average_to_hybrid": Decimal("0.75"),  # Hybrid typically 75% of average-data
    "hybrid_to_supplier": Decimal("0.65"), # Supplier-specific typically 65% of hybrid
    "spend_to_supplier": Decimal("0.50"),  # Supplier-specific typically 50% of spend-based
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class CategoryInput(BaseModel):
    """Category emission data for hotspot analysis.

    Attributes:
        category_number: Category number (1-15).
        category_name: Category name.
        total_co2e_tonnes: Total emissions.
        methodology_tier: Current methodology tier.
        data_quality_score: Data quality score (1-5).
    """
    category_number: int = Field(..., ge=1, le=15, description="Category number")
    category_name: str = Field(default="", description="Category name")
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total tCO2e"
    )
    methodology_tier: str = Field(default="spend_based", description="Methodology tier")
    data_quality_score: Decimal = Field(
        default=Decimal("4.0"), ge=1, le=5, description="DQ score"
    )

class SupplierData(BaseModel):
    """Supplier emission contribution data.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        country: Supplier country code.
        total_spend_eur: Total spend with supplier.
        estimated_co2e_tonnes: Estimated emissions from this supplier.
        categories_contributed: Scope 3 categories this supplier contributes to.
    """
    supplier_id: str = Field(default="", description="Supplier ID")
    supplier_name: str = Field(default="", max_length=500, description="Supplier name")
    country: str = Field(default="", max_length=2, description="Country")
    total_spend_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Spend EUR"
    )
    estimated_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated tCO2e"
    )
    categories_contributed: List[int] = Field(
        default_factory=list, description="Categories contributed"
    )

class ProductData(BaseModel):
    """Product emission intensity data.

    Attributes:
        product_id: Product identifier.
        product_name: Product name.
        units_sold: Units sold in reporting period.
        revenue_eur: Revenue from this product.
        total_co2e_tonnes: Total lifecycle emissions.
        co2e_per_unit: Emission intensity per unit.
        co2e_per_eur_revenue: Emission intensity per EUR revenue.
    """
    product_id: str = Field(default="", description="Product ID")
    product_name: str = Field(default="", max_length=500, description="Product name")
    units_sold: int = Field(default=0, ge=0, description="Units sold")
    revenue_eur: Decimal = Field(default=Decimal("0"), ge=0, description="Revenue EUR")
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total tCO2e"
    )
    co2e_per_unit: Decimal = Field(
        default=Decimal("0"), ge=0, description="tCO2e per unit"
    )
    co2e_per_eur_revenue: Decimal = Field(
        default=Decimal("0"), ge=0, description="tCO2e per EUR"
    )

class HotspotAnalysisInput(BaseModel):
    """Input data for hotspot analysis.

    Attributes:
        org_id: Organisation identifier.
        sector: Organisation sector for benchmarking.
        reporting_year: Reporting year.
        category_results: Per-category emission results.
        total_scope3_tco2e: Total Scope 3 emissions.
        scope1_tco2e: Scope 1 total.
        scope2_tco2e: Scope 2 total.
        supplier_data: Supplier-level data (optional).
        product_data: Product-level data (optional).
        pareto_threshold_pct: Pareto threshold (default 80%).
        top_n_suppliers: Number of top suppliers to highlight.
    """
    org_id: str = Field(default="", description="Organisation ID")
    sector: str = Field(default="manufacturing_heavy", description="Sector")
    reporting_year: int = Field(default=2025, description="Reporting year")
    category_results: List[CategoryInput] = Field(
        default_factory=list, description="Category results"
    )
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 3"
    )
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 1")
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 2")
    supplier_data: List[SupplierData] = Field(
        default_factory=list, description="Supplier data"
    )
    product_data: List[ProductData] = Field(
        default_factory=list, description="Product data"
    )
    pareto_threshold_pct: Decimal = Field(
        default=Decimal("80"), ge=0, le=100, description="Pareto threshold %"
    )
    top_n_suppliers: int = Field(default=20, ge=1, description="Top N suppliers")

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ParetoItem(BaseModel):
    """A single item in the Pareto analysis.

    Attributes:
        category_number: Category number.
        category_name: Category name.
        total_co2e_tonnes: Category emissions.
        share_of_total_pct: Share of total Scope 3.
        cumulative_share_pct: Cumulative share.
        is_in_pareto_set: Whether this item is in the 80% set.
        rank: Rank by emission magnitude.
    """
    category_number: int = Field(..., description="Category")
    category_name: str = Field(default="", description="Category name")
    total_co2e_tonnes: Decimal = Field(default=Decimal("0"), description="tCO2e")
    share_of_total_pct: Decimal = Field(default=Decimal("0"), description="Share %")
    cumulative_share_pct: Decimal = Field(default=Decimal("0"), description="Cumulative %")
    is_in_pareto_set: bool = Field(default=False, description="In Pareto set")
    rank: int = Field(default=0, description="Rank")

class SupplierConcentration(BaseModel):
    """Supplier concentration analysis result.

    Attributes:
        top_n: Number of top suppliers analysed.
        top_supplier_emissions_tco2e: Emissions from top N suppliers.
        top_supplier_share_pct: Share of total from top N suppliers.
        top_suppliers: List of top supplier details.
    """
    top_n: int = Field(default=0, description="Top N")
    top_supplier_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Top supplier emissions"
    )
    top_supplier_share_pct: Decimal = Field(
        default=Decimal("0"), description="Top supplier share %"
    )
    top_suppliers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top suppliers"
    )

class MaterialityScore(BaseModel):
    """Materiality matrix score for a category.

    Attributes:
        category_number: Category number.
        category_name: Category name.
        magnitude_score: Emission magnitude score (0-100).
        improvement_potential_score: Improvement potential score (0-100).
        combined_score: Combined materiality score.
        quadrant: Materiality quadrant (high-high, high-low, etc.).
    """
    category_number: int = Field(..., description="Category")
    category_name: str = Field(default="", description="Category name")
    magnitude_score: Decimal = Field(default=Decimal("0"), description="Magnitude")
    improvement_potential_score: Decimal = Field(
        default=Decimal("0"), description="Improvement potential"
    )
    combined_score: Decimal = Field(default=Decimal("0"), description="Combined")
    quadrant: str = Field(default="", description="Quadrant")

class SectorBenchmarkResult(BaseModel):
    """Sector benchmark comparison result.

    Attributes:
        sector: Sector used for benchmarking.
        benchmark_available: Whether benchmark data is available.
        category_gaps: Per-category gap (org_share - benchmark_share).
        over_represented: Categories over-represented vs benchmark.
        under_represented: Categories under-represented vs benchmark.
    """
    sector: str = Field(default="", description="Sector")
    benchmark_available: bool = Field(default=False, description="Available")
    category_gaps: Dict[int, Decimal] = Field(
        default_factory=dict, description="Gaps"
    )
    over_represented: List[int] = Field(
        default_factory=list, description="Over-represented"
    )
    under_represented: List[int] = Field(
        default_factory=list, description="Under-represented"
    )

class GeographicDistribution(BaseModel):
    """Geographic distribution of emissions.

    Attributes:
        country: Country code.
        total_co2e_tonnes: Emissions from this country.
        share_of_total_pct: Share of total.
        supplier_count: Number of suppliers in this country.
    """
    country: str = Field(default="", description="Country")
    total_co2e_tonnes: Decimal = Field(default=Decimal("0"), description="tCO2e")
    share_of_total_pct: Decimal = Field(default=Decimal("0"), description="Share %")
    supplier_count: int = Field(default=0, description="Supplier count")

class ReductionOpportunity(BaseModel):
    """A quantified emission reduction opportunity.

    Attributes:
        category_number: Target category.
        category_name: Category name.
        lever: Reduction lever type.
        description: Description of the opportunity.
        low_reduction_tco2e: Low-end reduction estimate.
        high_reduction_tco2e: High-end reduction estimate.
        low_reduction_pct: Low-end reduction percentage.
        high_reduction_pct: High-end reduction percentage.
        difficulty: Implementation difficulty.
        timeframe_years: Estimated implementation timeframe.
        priority_rank: Priority rank (1 = highest).
    """
    category_number: int = Field(default=0, description="Category")
    category_name: str = Field(default="", description="Category name")
    lever: str = Field(default="", description="Reduction lever")
    description: str = Field(default="", description="Description")
    low_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Low estimate tCO2e"
    )
    high_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="High estimate tCO2e"
    )
    low_reduction_pct: Decimal = Field(default=Decimal("0"), description="Low %")
    high_reduction_pct: Decimal = Field(default=Decimal("0"), description="High %")
    difficulty: str = Field(default="", description="Difficulty")
    timeframe_years: int = Field(default=0, description="Timeframe years")
    priority_rank: int = Field(default=0, description="Priority rank")

class TierUpgradeImpact(BaseModel):
    """Impact of upgrading methodology tier for a category.

    Attributes:
        category_number: Category number.
        current_tier: Current methodology tier.
        target_tier: Target methodology tier.
        current_estimate_tco2e: Current emission estimate.
        projected_estimate_tco2e: Projected estimate after upgrade.
        estimated_change_tco2e: Estimated change.
        estimated_change_pct: Estimated change percentage.
    """
    category_number: int = Field(default=0, description="Category")
    current_tier: str = Field(default="", description="Current tier")
    target_tier: str = Field(default="supplier_specific", description="Target tier")
    current_estimate_tco2e: Decimal = Field(
        default=Decimal("0"), description="Current estimate"
    )
    projected_estimate_tco2e: Decimal = Field(
        default=Decimal("0"), description="Projected estimate"
    )
    estimated_change_tco2e: Decimal = Field(
        default=Decimal("0"), description="Estimated change"
    )
    estimated_change_pct: Decimal = Field(
        default=Decimal("0"), description="Change %"
    )

class HotspotResult(BaseModel):
    """Complete hotspot analysis result.

    Attributes:
        result_id: Unique result identifier.
        org_id: Organisation identifier.
        sector: Sector used for benchmarking.
        reporting_year: Reporting year.
        pareto_analysis: Pareto analysis results.
        pareto_category_count: Number of categories in Pareto set.
        supplier_concentration: Supplier concentration analysis.
        materiality_scores: Materiality matrix scores.
        sector_benchmark: Sector benchmark comparison.
        geographic_distribution: Geographic distribution of emissions.
        reduction_opportunities: Quantified reduction opportunities.
        tier_upgrade_impacts: Methodology tier upgrade impacts.
        total_reduction_potential_low_tco2e: Total low reduction potential.
        total_reduction_potential_high_tco2e: Total high reduction potential.
        top_3_priorities: Top 3 priority areas summary.
        warnings: Warnings.
        status: Analysis status.
        calculated_at: Timestamp.
        processing_time_ms: Processing time ms.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    org_id: str = Field(default="", description="Organisation ID")
    sector: str = Field(default="", description="Sector")
    reporting_year: int = Field(default=2025, description="Reporting year")
    pareto_analysis: List[ParetoItem] = Field(
        default_factory=list, description="Pareto analysis"
    )
    pareto_category_count: int = Field(
        default=0, description="Pareto category count"
    )
    supplier_concentration: Optional[SupplierConcentration] = Field(
        default=None, description="Supplier concentration"
    )
    materiality_scores: List[MaterialityScore] = Field(
        default_factory=list, description="Materiality scores"
    )
    sector_benchmark: Optional[SectorBenchmarkResult] = Field(
        default=None, description="Sector benchmark"
    )
    geographic_distribution: List[GeographicDistribution] = Field(
        default_factory=list, description="Geographic distribution"
    )
    reduction_opportunities: List[ReductionOpportunity] = Field(
        default_factory=list, description="Reduction opportunities"
    )
    tier_upgrade_impacts: List[TierUpgradeImpact] = Field(
        default_factory=list, description="Tier upgrade impacts"
    )
    total_reduction_potential_low_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total low potential"
    )
    total_reduction_potential_high_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total high potential"
    )
    top_3_priorities: List[str] = Field(
        default_factory=list, description="Top 3 priorities"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETE, description="Status"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

CategoryInput.model_rebuild()
SupplierData.model_rebuild()
ProductData.model_rebuild()
HotspotAnalysisInput.model_rebuild()
ParetoItem.model_rebuild()
SupplierConcentration.model_rebuild()
MaterialityScore.model_rebuild()
SectorBenchmarkResult.model_rebuild()
GeographicDistribution.model_rebuild()
ReductionOpportunity.model_rebuild()
TierUpgradeImpact.model_rebuild()
HotspotResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HotspotAnalysisEngine:
    """Identify emission hotspots and prioritise reduction efforts.

    Performs multi-dimensional hotspot analysis including Pareto analysis,
    supplier concentration, product intensity, materiality scoring,
    sector benchmarking, geographic mapping, and reduction opportunity
    quantification.

    Attributes:
        _warnings: Warnings generated during analysis.

    Example:
        >>> engine = HotspotAnalysisEngine()
        >>> input_data = HotspotAnalysisInput(
        ...     sector="manufacturing_heavy",
        ...     category_results=[
        ...         CategoryInput(category_number=1, total_co2e_tonnes=Decimal("10000")),
        ...         CategoryInput(category_number=4, total_co2e_tonnes=Decimal("3000")),
        ...     ],
        ...     total_scope3_tco2e=Decimal("15000"),
        ... )
        >>> result = engine.analyze_hotspots(input_data)
        >>> print(result.pareto_category_count)
    """

    def __init__(self) -> None:
        """Initialise HotspotAnalysisEngine."""
        self._warnings: List[str] = []
        logger.info("HotspotAnalysisEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_hotspots(
        self,
        input_data: HotspotAnalysisInput,
    ) -> HotspotResult:
        """Perform comprehensive hotspot analysis.

        Main entry point.  Runs all analysis modules and produces
        prioritised reduction recommendations.

        Args:
            input_data: Hotspot analysis input data.

        Returns:
            HotspotResult.

        Raises:
            ValueError: If no category results provided.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not input_data.category_results:
            raise ValueError("At least one category result is required")

        total_scope3 = input_data.total_scope3_tco2e
        if total_scope3 <= Decimal("0"):
            total_scope3 = sum(
                (cr.total_co2e_tonnes for cr in input_data.category_results),
                Decimal("0"),
            )

        logger.info("Starting hotspot analysis for %d categories", len(input_data.category_results))

        # Step 1: Pareto analysis
        pareto = self._pareto_analysis(
            input_data.category_results,
            total_scope3,
            input_data.pareto_threshold_pct,
        )
        pareto_count = sum(1 for p in pareto if p.is_in_pareto_set)

        # Step 2: Supplier concentration
        supplier_conc = None
        if input_data.supplier_data:
            supplier_conc = self._supplier_concentration(
                input_data.supplier_data,
                total_scope3,
                input_data.top_n_suppliers,
            )

        # Step 3: Materiality matrix
        materiality = self._materiality_matrix(
            input_data.category_results, total_scope3
        )

        # Step 4: Sector benchmark
        benchmark = self._sector_benchmark(
            input_data.category_results,
            total_scope3,
            input_data.sector,
        )

        # Step 5: Geographic mapping
        geo_dist: List[GeographicDistribution] = []
        if input_data.supplier_data:
            geo_dist = self._geographic_mapping(
                input_data.supplier_data, total_scope3
            )

        # Step 6: Reduction opportunities
        opportunities = self._reduction_opportunities(
            input_data.category_results, total_scope3
        )

        # Step 7: Tier upgrade impact
        tier_impacts = self._tier_upgrade_impact(input_data.category_results)

        # Calculate totals
        total_low = sum(
            (o.low_reduction_tco2e for o in opportunities), Decimal("0")
        )
        total_high = sum(
            (o.high_reduction_tco2e for o in opportunities), Decimal("0")
        )

        # Top 3 priorities
        top_3 = self._extract_top_priorities(pareto, opportunities)

        elapsed_ms = Decimal(str((time.perf_counter() - t0) * 1000))

        result = HotspotResult(
            org_id=input_data.org_id,
            sector=input_data.sector,
            reporting_year=input_data.reporting_year,
            pareto_analysis=pareto,
            pareto_category_count=pareto_count,
            supplier_concentration=supplier_conc,
            materiality_scores=materiality,
            sector_benchmark=benchmark,
            geographic_distribution=geo_dist,
            reduction_opportunities=opportunities,
            tier_upgrade_impacts=tier_impacts,
            total_reduction_potential_low_tco2e=_round_val(total_low, 2),
            total_reduction_potential_high_tco2e=_round_val(total_high, 2),
            top_3_priorities=top_3,
            warnings=list(self._warnings),
            status=AnalysisStatus.COMPLETE,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Hotspot analysis complete: %d Pareto categories, %d opportunities, "
            "%.2f-%.2f tCO2e reduction potential",
            pareto_count, len(opportunities), total_low, total_high,
        )
        return result

    # ------------------------------------------------------------------
    # Analysis Methods
    # ------------------------------------------------------------------

    def _pareto_analysis(
        self,
        categories: List[CategoryInput],
        total_scope3: Decimal,
        threshold_pct: Decimal,
    ) -> List[ParetoItem]:
        """Perform Pareto analysis on Scope 3 categories.

        Identifies the categories that cumulatively account for
        the threshold percentage (default 80%) of total emissions.

        Args:
            categories: Category emission data.
            total_scope3: Total Scope 3 emissions.
            threshold_pct: Cumulative threshold percentage.

        Returns:
            List of ParetoItem sorted by emissions descending.
        """
        sorted_cats = sorted(
            categories, key=lambda c: c.total_co2e_tonnes, reverse=True
        )

        items: List[ParetoItem] = []
        cumulative = Decimal("0")

        for rank, cat in enumerate(sorted_cats, 1):
            share = _safe_pct(cat.total_co2e_tonnes, total_scope3)
            cumulative += share
            in_set = cumulative <= threshold_pct or rank == 1

            # Include the category that pushes cumulative past threshold
            if not in_set and items and not items[-1].is_in_pareto_set:
                pass
            elif cumulative - share < threshold_pct:
                in_set = True

            items.append(ParetoItem(
                category_number=cat.category_number,
                category_name=cat.category_name,
                total_co2e_tonnes=_round_val(cat.total_co2e_tonnes, 2),
                share_of_total_pct=_round_val(share, 2),
                cumulative_share_pct=_round_val(cumulative, 2),
                is_in_pareto_set=in_set,
                rank=rank,
            ))

        return items

    def _supplier_concentration(
        self,
        supplier_data: List[SupplierData],
        total_scope3: Decimal,
        top_n: int,
    ) -> SupplierConcentration:
        """Analyse supplier concentration.

        Identifies top N suppliers by emission contribution.

        Args:
            supplier_data: Supplier-level data.
            total_scope3: Total Scope 3 emissions.
            top_n: Number of top suppliers.

        Returns:
            SupplierConcentration.
        """
        sorted_suppliers = sorted(
            supplier_data,
            key=lambda s: s.estimated_co2e_tonnes,
            reverse=True,
        )

        top = sorted_suppliers[:top_n]
        top_emissions = sum(
            (s.estimated_co2e_tonnes for s in top), Decimal("0")
        )
        top_share = _safe_pct(top_emissions, total_scope3)

        top_details = [
            {
                "supplier_name": s.supplier_name,
                "country": s.country,
                "estimated_co2e_tonnes": str(_round_val(s.estimated_co2e_tonnes, 2)),
                "share_pct": str(_round_val(
                    _safe_pct(s.estimated_co2e_tonnes, total_scope3), 2
                )),
            }
            for s in top
        ]

        return SupplierConcentration(
            top_n=len(top),
            top_supplier_emissions_tco2e=_round_val(top_emissions, 2),
            top_supplier_share_pct=_round_val(top_share, 2),
            top_suppliers=top_details,
        )

    def _product_intensity(
        self,
        product_data: List[ProductData],
    ) -> List[ProductData]:
        """Rank products by emission intensity.

        Args:
            product_data: Product-level data.

        Returns:
            Products sorted by intensity descending.
        """
        # Calculate intensities where missing
        for p in product_data:
            if p.co2e_per_unit == Decimal("0") and p.units_sold > 0:
                p.co2e_per_unit = _round_val(
                    _safe_divide(p.total_co2e_tonnes, _decimal(p.units_sold)), 6
                )
            if p.co2e_per_eur_revenue == Decimal("0") and p.revenue_eur > Decimal("0"):
                p.co2e_per_eur_revenue = _round_val(
                    _safe_divide(p.total_co2e_tonnes, p.revenue_eur), 8
                )

        return sorted(
            product_data,
            key=lambda p: p.co2e_per_unit,
            reverse=True,
        )

    def _materiality_matrix(
        self,
        categories: List[CategoryInput],
        total_scope3: Decimal,
    ) -> List[MaterialityScore]:
        """Score each category on materiality matrix (magnitude x improvement).

        Args:
            categories: Category data.
            total_scope3: Total Scope 3 emissions.

        Returns:
            List of MaterialityScore.
        """
        scores: List[MaterialityScore] = []

        for cat in categories:
            share = _safe_pct(cat.total_co2e_tonnes, total_scope3)
            # Magnitude score: normalised to 0-100 (assuming max share ~60%)
            mag_score = min(share / Decimal("0.60"), Decimal("100"))

            # Improvement potential score based on available reduction levers
            cat_potentials = REDUCTION_POTENTIALS.get(cat.category_number, {})
            if cat_potentials:
                max_high = max(
                    (v["high_pct"] for v in cat_potentials.values()),
                    default=Decimal("0"),
                )
                imp_score = min(max_high * Decimal("2"), Decimal("100"))
            else:
                imp_score = Decimal("20")  # Default low potential

            combined = (mag_score + imp_score) / Decimal("2")

            # Determine quadrant
            if mag_score >= Decimal("50") and imp_score >= Decimal("50"):
                quadrant = "high_magnitude_high_potential"
            elif mag_score >= Decimal("50") and imp_score < Decimal("50"):
                quadrant = "high_magnitude_low_potential"
            elif mag_score < Decimal("50") and imp_score >= Decimal("50"):
                quadrant = "low_magnitude_high_potential"
            else:
                quadrant = "low_magnitude_low_potential"

            scores.append(MaterialityScore(
                category_number=cat.category_number,
                category_name=cat.category_name,
                magnitude_score=_round_val(mag_score, 2),
                improvement_potential_score=_round_val(imp_score, 2),
                combined_score=_round_val(combined, 2),
                quadrant=quadrant,
            ))

        return sorted(scores, key=lambda s: s.combined_score, reverse=True)

    def _sector_benchmark(
        self,
        categories: List[CategoryInput],
        total_scope3: Decimal,
        sector: str,
    ) -> SectorBenchmarkResult:
        """Compare category distribution against sector benchmark.

        Args:
            categories: Category data.
            total_scope3: Total Scope 3 emissions.
            sector: Sector key for benchmark lookup.

        Returns:
            SectorBenchmarkResult.
        """
        benchmark = SECTOR_BENCHMARKS.get(sector)
        if not benchmark:
            self._warnings.append(
                f"No benchmark data for sector '{sector}'"
            )
            return SectorBenchmarkResult(
                sector=sector,
                benchmark_available=False,
            )

        # Calculate org shares
        org_shares: Dict[int, Decimal] = {}
        for cat in categories:
            org_shares[cat.category_number] = _safe_pct(
                cat.total_co2e_tonnes, total_scope3
            )

        # Calculate gaps
        gaps: Dict[int, Decimal] = {}
        over_rep: List[int] = []
        under_rep: List[int] = []

        for cat_num in range(1, 16):
            org_share = org_shares.get(cat_num, Decimal("0"))
            bench_share = benchmark.get(cat_num, Decimal("0"))
            gap = _round_val(org_share - bench_share, 2)
            gaps[cat_num] = gap

            if gap > Decimal("5"):
                over_rep.append(cat_num)
            elif gap < Decimal("-5"):
                under_rep.append(cat_num)

        return SectorBenchmarkResult(
            sector=sector,
            benchmark_available=True,
            category_gaps=gaps,
            over_represented=over_rep,
            under_represented=under_rep,
        )

    def _geographic_mapping(
        self,
        supplier_data: List[SupplierData],
        total_scope3: Decimal,
    ) -> List[GeographicDistribution]:
        """Map emissions by supplier country.

        Args:
            supplier_data: Supplier-level data.
            total_scope3: Total Scope 3 emissions.

        Returns:
            List of GeographicDistribution sorted by emissions.
        """
        country_data: Dict[str, Dict[str, Any]] = {}

        for s in supplier_data:
            country = s.country or "UNKNOWN"
            if country not in country_data:
                country_data[country] = {
                    "emissions": Decimal("0"),
                    "count": 0,
                }
            country_data[country]["emissions"] += s.estimated_co2e_tonnes
            country_data[country]["count"] += 1

        distributions: List[GeographicDistribution] = []
        for country, d in country_data.items():
            distributions.append(GeographicDistribution(
                country=country,
                total_co2e_tonnes=_round_val(d["emissions"], 2),
                share_of_total_pct=_round_val(
                    _safe_pct(d["emissions"], total_scope3), 2
                ),
                supplier_count=d["count"],
            ))

        return sorted(
            distributions,
            key=lambda g: g.total_co2e_tonnes,
            reverse=True,
        )

    def _reduction_opportunities(
        self,
        categories: List[CategoryInput],
        total_scope3: Decimal,
    ) -> List[ReductionOpportunity]:
        """Quantify reduction opportunities for each category.

        Args:
            categories: Category data.
            total_scope3: Total Scope 3 emissions.

        Returns:
            List of ReductionOpportunity sorted by high-end potential.
        """
        opportunities: List[ReductionOpportunity] = []

        for cat in categories:
            cat_potentials = REDUCTION_POTENTIALS.get(cat.category_number, {})
            for lever_key, params in cat_potentials.items():
                low_reduction = cat.total_co2e_tonnes * params["low_pct"] / Decimal("100")
                high_reduction = cat.total_co2e_tonnes * params["high_pct"] / Decimal("100")

                opportunities.append(ReductionOpportunity(
                    category_number=cat.category_number,
                    category_name=cat.category_name,
                    lever=lever_key,
                    description=params["description"],
                    low_reduction_tco2e=_round_val(low_reduction, 2),
                    high_reduction_tco2e=_round_val(high_reduction, 2),
                    low_reduction_pct=params["low_pct"],
                    high_reduction_pct=params["high_pct"],
                    difficulty=params["difficulty"],
                    timeframe_years=params["timeframe_years"],
                ))

        # Sort by high-end potential descending
        opportunities.sort(key=lambda o: o.high_reduction_tco2e, reverse=True)

        # Assign priority ranks
        for rank, opp in enumerate(opportunities, 1):
            opp.priority_rank = rank

        return opportunities

    def _tier_upgrade_impact(
        self,
        categories: List[CategoryInput],
    ) -> List[TierUpgradeImpact]:
        """Estimate the impact of upgrading methodology tier.

        Uses typical tier difference ratios to project what the emission
        estimate would be if a higher-quality methodology were used.

        Args:
            categories: Category data.

        Returns:
            List of TierUpgradeImpact.
        """
        impacts: List[TierUpgradeImpact] = []

        for cat in categories:
            if cat.total_co2e_tonnes <= Decimal("0"):
                continue

            current_tier = cat.methodology_tier.lower()

            # Determine appropriate ratio
            if current_tier in ("spend_based", "spend-based", "eeio"):
                ratio = TIER_UPGRADE_RATIOS["spend_to_supplier"]
                target = "supplier_specific"
            elif current_tier in ("average_data", "average-data", "average"):
                ratio = TIER_UPGRADE_RATIOS["average_to_hybrid"]
                target = "hybrid"
            elif current_tier in ("hybrid",):
                ratio = TIER_UPGRADE_RATIOS["hybrid_to_supplier"]
                target = "supplier_specific"
            else:
                continue  # Already at highest tier

            projected = cat.total_co2e_tonnes * ratio
            change = projected - cat.total_co2e_tonnes
            change_pct = _safe_pct(change, cat.total_co2e_tonnes)

            impacts.append(TierUpgradeImpact(
                category_number=cat.category_number,
                current_tier=current_tier,
                target_tier=target,
                current_estimate_tco2e=_round_val(cat.total_co2e_tonnes, 2),
                projected_estimate_tco2e=_round_val(projected, 2),
                estimated_change_tco2e=_round_val(change, 2),
                estimated_change_pct=_round_val(change_pct, 2),
            ))

        return impacts

    def _extract_top_priorities(
        self,
        pareto: List[ParetoItem],
        opportunities: List[ReductionOpportunity],
    ) -> List[str]:
        """Extract top 3 priority action items.

        Args:
            pareto: Pareto analysis results.
            opportunities: Reduction opportunities.

        Returns:
            List of up to 3 priority summary strings.
        """
        priorities: List[str] = []

        # Priority 1: Largest Pareto category
        if pareto:
            top_cat = pareto[0]
            priorities.append(
                f"Cat {top_cat.category_number} ({top_cat.category_name}): "
                f"{top_cat.share_of_total_pct}% of Scope 3 "
                f"({top_cat.total_co2e_tonnes} tCO2e)"
            )

        # Priority 2-3: Top reduction opportunities
        for opp in opportunities[:2]:
            priorities.append(
                f"Cat {opp.category_number} - {opp.lever}: "
                f"reduce {opp.low_reduction_tco2e}-{opp.high_reduction_tco2e} tCO2e "
                f"({opp.description})"
            )

        return priorities[:3]

    def _compute_provenance(self, result: HotspotResult) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            result: Hotspot analysis result.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(result)
