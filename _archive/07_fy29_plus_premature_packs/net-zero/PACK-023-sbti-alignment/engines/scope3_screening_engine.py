# -*- coding: utf-8 -*-
"""
Scope3ScreeningEngine - PACK-023 SBTi Alignment Engine 3
==========================================================

15-category Scope 3 materiality screening with 40% trigger assessment,
67%/90% coverage tracking for near-term/long-term targets, supplier
engagement target validation, category prioritisation, and data quality
scoring per category (primary/secondary/proxy/spend-based).

This engine systematically evaluates every GHG Protocol Scope 3 category
to determine materiality, assess data quality, calculate coverage against
SBTi thresholds, validate supplier engagement targets, and produce a
prioritised action plan for Scope 3 target setting.

Calculation Methodology:
    Scope 3 Trigger Assessment:
        scope3_fraction = S3_total / (S1 + S2_market + S3_total)
        scope3_required = scope3_fraction >= 0.40

    Category Materiality:
        cat_fraction = cat_emissions / S3_total
        HIGH       if cat_fraction >= 0.10 (>= 10% of S3)
        MEDIUM     if cat_fraction >= 0.03 (>= 3% of S3)
        LOW        if cat_fraction >= 0.01 (>= 1% of S3)
        NEGLIGIBLE if cat_fraction <  0.01 (<  1% of S3)

    Coverage Tracking:
        near_term_coverage = sum(targeted_categories) / S3_total >= 0.67
        long_term_coverage = sum(targeted_categories) / S3_total >= 0.90

    Data Quality Scoring (per category):
        PRIMARY    = 1.00 (measured, supplier-specific)
        SECONDARY  = 0.75 (industry-average, activity-based)
        PROXY      = 0.50 (proxy data, modelled)
        SPEND      = 0.25 (spend-based EEIO)
        NONE       = 0.00 (no data available)

        weighted_quality = sum(cat_quality * cat_fraction) / sum(cat_fraction)

    Supplier Engagement Target:
        SBTi requires engagement targets for purchased goods (Cat 1)
        and upstream transportation (Cat 4) when material.
        engagement_target_met = pct_suppliers_with_sbti >= required_pct

    Category Prioritisation Score:
        priority = (materiality_weight * 0.40
                  + data_quality_inverse * 0.25
                  + reduction_potential * 0.20
                  + influence_score * 0.15)

Regulatory References:
    - SBTi Corporate Manual V5.3 (2024) - Scope 3 criteria C17-C20
    - SBTi Corporate Net-Zero Standard V1.3 (2024) - NZ-C6, NZ-C7
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Technical Guidance for Calculating Scope 3 (2013)
    - SBTi Scope 3 Target-Setting Guidance (2024)
    - SBTi Supplier Engagement Guidance (2024)
    - IPCC AR6 WG3 (2022) - Sectoral mitigation potentials
    - ISO 14064-1:2018 - GHG quantification

Zero-Hallucination:
    - All thresholds from published SBTi Corporate Manual V5.3
    - Materiality bands from GHG Protocol Scope 3 Guidance
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  3 of 10
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
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
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

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15).

    Each value corresponds to the standard category number and short name
    as defined in the GHG Protocol Corporate Value Chain Standard (2011).
    """
    CAT_1 = "cat_1_purchased_goods_services"
    CAT_2 = "cat_2_capital_goods"
    CAT_3 = "cat_3_fuel_energy_activities"
    CAT_4 = "cat_4_upstream_transportation"
    CAT_5 = "cat_5_waste_generated"
    CAT_6 = "cat_6_business_travel"
    CAT_7 = "cat_7_employee_commuting"
    CAT_8 = "cat_8_upstream_leased_assets"
    CAT_9 = "cat_9_downstream_transportation"
    CAT_10 = "cat_10_processing_sold_products"
    CAT_11 = "cat_11_use_of_sold_products"
    CAT_12 = "cat_12_end_of_life_treatment"
    CAT_13 = "cat_13_downstream_leased_assets"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"

class MaterialityLevel(str, Enum):
    """Materiality classification for a Scope 3 category.

    HIGH:       >= 10% of total Scope 3 emissions.
    MEDIUM:     >= 3% and < 10% of total Scope 3 emissions.
    LOW:        >= 1% and < 3% of total Scope 3 emissions.
    NEGLIGIBLE: < 1% of total Scope 3 emissions.
    NOT_ASSESSED: Category has not been screened.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"
    NOT_ASSESSED = "not_assessed"

class DataQualityTier(str, Enum):
    """Data quality classification per SBTi and GHG Protocol guidance.

    PRIMARY:   Supplier-specific, measured data (highest quality).
    SECONDARY: Industry-average, activity-based data.
    PROXY:     Proxy or modelled data.
    SPEND:     Spend-based EEIO estimates (lowest quality with data).
    NONE:      No data available.
    """
    PRIMARY = "primary"
    SECONDARY = "secondary"
    PROXY = "proxy"
    SPEND = "spend"
    NONE = "none"

class TargetApproach(str, Enum):
    """Scope 3 target-setting approach per SBTi guidance.

    ABSOLUTE: Absolute reduction target for the category.
    INTENSITY: Intensity-based reduction target.
    SUPPLIER_ENGAGEMENT: Supplier engagement target (% with SBTi targets).
    COMBINED: Combined absolute + engagement approach.
    NOT_SET: No target set for this category.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    COMBINED = "combined"
    NOT_SET = "not_set"

class ScreeningStatus(str, Enum):
    """Status of category screening.

    COMPLETE: Category fully screened with emissions estimate.
    PARTIAL: Category partially screened (some data gaps).
    NOT_STARTED: Category not yet screened.
    NOT_RELEVANT: Category confirmed not relevant to the entity.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not_started"
    NOT_RELEVANT = "not_relevant"

class ReductionPotential(str, Enum):
    """Estimated reduction potential for a category.

    HIGH: Significant reduction achievable (> 30%).
    MEDIUM: Moderate reduction achievable (10-30%).
    LOW: Limited reduction potential (< 10%).
    UNKNOWN: Reduction potential not yet assessed.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class InfluenceLevel(str, Enum):
    """Degree of organisational influence over the category.

    DIRECT: Organisation has direct control (procurement, design).
    SIGNIFICANT: Meaningful influence through contracts or design.
    LIMITED: Some influence through engagement.
    MINIMAL: Little ability to influence emissions.
    """
    DIRECT = "direct"
    SIGNIFICANT = "significant"
    LIMITED = "limited"
    MINIMAL = "minimal"

# ---------------------------------------------------------------------------
# Constants -- SBTi Scope 3 Thresholds (Corporate Manual V5.3)
# ---------------------------------------------------------------------------

# Scope 3 materiality trigger: S3 >= 40% of (S1 + S2 + S3).
# Source: SBTi Corporate Manual V5.3, Section 6.1.
SCOPE3_TRIGGER_THRESHOLD: Decimal = Decimal("0.40")

# Scope 3 near-term coverage minimum: 67% of total S3.
# Source: SBTi Corporate Manual V5.3, Section 6.3.
SCOPE3_NT_COVERAGE_MIN: Decimal = Decimal("0.67")

# Scope 3 long-term coverage minimum: 90% of total S3.
# Source: SBTi Net-Zero Standard V1.3, Section 5.2.
SCOPE3_LT_COVERAGE_MIN: Decimal = Decimal("0.90")

# Category materiality thresholds (fraction of S3 total).
# Source: GHG Protocol Scope 3 Guidance, Chapter 7.
MATERIALITY_HIGH_THRESHOLD: Decimal = Decimal("0.10")
MATERIALITY_MEDIUM_THRESHOLD: Decimal = Decimal("0.03")
MATERIALITY_LOW_THRESHOLD: Decimal = Decimal("0.01")

# Supplier engagement: minimum % of suppliers with SBTi targets.
# Source: SBTi Supplier Engagement Guidance (2024), Table 2.
SUPPLIER_ENGAGEMENT_MIN_PCT: Decimal = Decimal("67.0")

# Supplier engagement timeline: years to achieve engagement target.
SUPPLIER_ENGAGEMENT_TIMELINE_YEARS: int = 5

# Data quality scores (numeric) for weighted quality calculation.
DATA_QUALITY_SCORES: Dict[str, Decimal] = {
    DataQualityTier.PRIMARY.value: Decimal("1.00"),
    DataQualityTier.SECONDARY.value: Decimal("0.75"),
    DataQualityTier.PROXY.value: Decimal("0.50"),
    DataQualityTier.SPEND.value: Decimal("0.25"),
    DataQualityTier.NONE.value: Decimal("0.00"),
}

# Minimum data quality score for reliable target-setting.
MIN_DATA_QUALITY_FOR_TARGET: Decimal = Decimal("0.50")

# Prioritisation weights.
PRIORITY_WEIGHT_MATERIALITY: Decimal = Decimal("0.40")
PRIORITY_WEIGHT_DATA_QUALITY: Decimal = Decimal("0.25")
PRIORITY_WEIGHT_REDUCTION: Decimal = Decimal("0.20")
PRIORITY_WEIGHT_INFLUENCE: Decimal = Decimal("0.15")

# Number of Scope 3 categories per GHG Protocol.
TOTAL_SCOPE3_CATEGORIES: int = 15

# Minimum annual reduction rate for Scope 3 near-term targets.
# Source: SBTi Corporate Manual V5.3, Table 6.
SCOPE3_MIN_ANNUAL_RATE: Decimal = Decimal("0.025")

# Minimum annual reduction rate for 1.5C-aligned Scope 3.
SCOPE3_15C_ANNUAL_RATE: Decimal = Decimal("0.042")

# ---------------------------------------------------------------------------
# Scope 3 Category Reference Data
# ---------------------------------------------------------------------------

CATEGORY_DEFINITIONS: Dict[str, Dict[str, str]] = {
    Scope3Category.CAT_1.value: {
        "number": "1",
        "name": "Purchased Goods and Services",
        "description": "Extraction, production, and transportation of goods and services purchased by the reporting company",
        "typical_methods": "Spend-based, average-data, supplier-specific, hybrid",
        "typical_materiality": "high",
        "engagement_eligible": "yes",
    },
    Scope3Category.CAT_2.value: {
        "number": "2",
        "name": "Capital Goods",
        "description": "Extraction, production, and transportation of capital goods purchased by the reporting company",
        "typical_methods": "Spend-based, average-data, supplier-specific",
        "typical_materiality": "medium",
        "engagement_eligible": "yes",
    },
    Scope3Category.CAT_3.value: {
        "number": "3",
        "name": "Fuel- and Energy-Related Activities",
        "description": "Upstream emissions from purchased fuels and electricity not included in Scope 1 or 2",
        "typical_methods": "Average-data, supplier-specific",
        "typical_materiality": "medium",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_4.value: {
        "number": "4",
        "name": "Upstream Transportation and Distribution",
        "description": "Transportation and distribution of products purchased in vehicles not owned or controlled",
        "typical_methods": "Distance-based, spend-based, supplier-specific",
        "typical_materiality": "medium",
        "engagement_eligible": "yes",
    },
    Scope3Category.CAT_5.value: {
        "number": "5",
        "name": "Waste Generated in Operations",
        "description": "Disposal and treatment of waste generated in operations",
        "typical_methods": "Waste-type-specific, average-data",
        "typical_materiality": "low",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_6.value: {
        "number": "6",
        "name": "Business Travel",
        "description": "Transportation of employees for business-related activities",
        "typical_methods": "Distance-based, spend-based, fuel-based",
        "typical_materiality": "low",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_7.value: {
        "number": "7",
        "name": "Employee Commuting",
        "description": "Transportation of employees between homes and worksites",
        "typical_methods": "Distance-based, average-data",
        "typical_materiality": "low",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_8.value: {
        "number": "8",
        "name": "Upstream Leased Assets",
        "description": "Operation of assets leased by the reporting company",
        "typical_methods": "Asset-specific, average-data",
        "typical_materiality": "low",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_9.value: {
        "number": "9",
        "name": "Downstream Transportation and Distribution",
        "description": "Transportation and distribution of products sold between reporting company and end consumer",
        "typical_methods": "Distance-based, average-data",
        "typical_materiality": "medium",
        "engagement_eligible": "yes",
    },
    Scope3Category.CAT_10.value: {
        "number": "10",
        "name": "Processing of Sold Products",
        "description": "Processing of intermediate products sold by the reporting company",
        "typical_methods": "Average-data, site-specific",
        "typical_materiality": "medium",
        "engagement_eligible": "yes",
    },
    Scope3Category.CAT_11.value: {
        "number": "11",
        "name": "Use of Sold Products",
        "description": "End use of goods and services sold by the reporting company",
        "typical_methods": "Product-specific, average-data",
        "typical_materiality": "high",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_12.value: {
        "number": "12",
        "name": "End-of-Life Treatment of Sold Products",
        "description": "Waste disposal and treatment of products sold by the reporting company",
        "typical_methods": "Waste-type-specific, average-data",
        "typical_materiality": "low",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_13.value: {
        "number": "13",
        "name": "Downstream Leased Assets",
        "description": "Operation of assets owned by the reporting company and leased to others",
        "typical_methods": "Asset-specific, average-data",
        "typical_materiality": "low",
        "engagement_eligible": "no",
    },
    Scope3Category.CAT_14.value: {
        "number": "14",
        "name": "Franchises",
        "description": "Operation of franchises not included in Scope 1 and 2",
        "typical_methods": "Franchise-specific, average-data",
        "typical_materiality": "medium",
        "engagement_eligible": "yes",
    },
    Scope3Category.CAT_15.value: {
        "number": "15",
        "name": "Investments",
        "description": "Operation of investments not included in Scope 1 and 2",
        "typical_methods": "Investment-specific, PCAF-based, average-data",
        "typical_materiality": "high",
        "engagement_eligible": "yes",
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class CategoryInput(BaseModel):
    """Input data for a single Scope 3 category.

    Attributes:
        category: GHG Protocol Scope 3 category identifier.
        emissions_tco2e: Estimated emissions for the category (tCO2e).
        data_quality: Data quality tier for the estimate.
        screening_status: Current screening status.
        is_relevant: Whether the category is relevant to the entity.
        is_targeted: Whether the category is included in targets.
        target_approach: Target-setting approach if targeted.
        reduction_potential: Estimated reduction potential.
        influence_level: Degree of organisational influence.
        supplier_count: Number of suppliers in this category.
        suppliers_with_sbti: Number of suppliers with SBTi targets.
        notes: Free-text notes on the category.
    """
    category: str = Field(
        ..., description="Scope 3 category identifier"
    )
    emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Category emissions in tCO2e"
    )
    data_quality: str = Field(
        default=DataQualityTier.NONE.value,
        description="Data quality tier"
    )
    screening_status: str = Field(
        default=ScreeningStatus.NOT_STARTED.value,
        description="Screening completion status"
    )
    is_relevant: bool = Field(
        default=True,
        description="Whether category is relevant to the entity"
    )
    is_targeted: bool = Field(
        default=False,
        description="Whether category is included in S3 targets"
    )
    target_approach: str = Field(
        default=TargetApproach.NOT_SET.value,
        description="Target-setting approach for this category"
    )
    reduction_potential: str = Field(
        default=ReductionPotential.UNKNOWN.value,
        description="Estimated reduction potential"
    )
    influence_level: str = Field(
        default=InfluenceLevel.LIMITED.value,
        description="Degree of organisational influence"
    )
    supplier_count: int = Field(
        default=0, ge=0,
        description="Number of suppliers in this category"
    )
    suppliers_with_sbti: int = Field(
        default=0, ge=0,
        description="Number of suppliers with validated SBTi targets"
    )
    notes: str = Field(
        default="",
        description="Additional notes on the category"
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category is a known Scope 3 category."""
        valid = {c.value for c in Scope3Category}
        if v not in valid:
            raise ValueError(
                f"Unknown Scope 3 category '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("data_quality")
    @classmethod
    def validate_data_quality(cls, v: str) -> str:
        """Validate data quality tier."""
        valid = {t.value for t in DataQualityTier}
        if v not in valid:
            raise ValueError(
                f"Unknown data quality tier '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

class SupplierEngagementInput(BaseModel):
    """Input data for supplier engagement target assessment.

    Attributes:
        total_suppliers: Total number of tier-1 suppliers.
        suppliers_with_sbti: Number of suppliers with validated SBTi targets.
        suppliers_committed: Number of suppliers with SBTi commitments.
        target_pct: Target percentage of suppliers with SBTi targets.
        target_year: Year by which engagement target should be met.
        engagement_strategy: Description of engagement approach.
        covers_purchased_goods: Whether engagement covers Cat 1.
        covers_upstream_transport: Whether engagement covers Cat 4.
        covers_capital_goods: Whether engagement covers Cat 2.
        annual_review: Whether annual progress review is conducted.
    """
    total_suppliers: int = Field(
        default=0, ge=0,
        description="Total tier-1 suppliers"
    )
    suppliers_with_sbti: int = Field(
        default=0, ge=0,
        description="Suppliers with validated SBTi targets"
    )
    suppliers_committed: int = Field(
        default=0, ge=0,
        description="Suppliers with SBTi commitments (not yet validated)"
    )
    target_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Target % of suppliers with SBTi targets"
    )
    target_year: int = Field(
        default=0, ge=0, le=2060,
        description="Target year for engagement goal"
    )
    engagement_strategy: str = Field(
        default="",
        description="Description of supplier engagement approach"
    )
    covers_purchased_goods: bool = Field(
        default=False,
        description="Whether engagement covers Category 1"
    )
    covers_upstream_transport: bool = Field(
        default=False,
        description="Whether engagement covers Category 4"
    )
    covers_capital_goods: bool = Field(
        default=False,
        description="Whether engagement covers Category 2"
    )
    annual_review: bool = Field(
        default=False,
        description="Whether annual progress review is conducted"
    )

class Scope3ScreeningInput(BaseModel):
    """Input data for complete Scope 3 screening.

    Attributes:
        entity_name: Reporting entity name.
        base_year: Emissions base year.
        scope1_tco2e: Scope 1 direct emissions (tCO2e).
        scope2_market_tco2e: Scope 2 market-based emissions (tCO2e).
        scope2_location_tco2e: Scope 2 location-based emissions (tCO2e).
        categories: Per-category screening data (up to 15).
        supplier_engagement: Supplier engagement data.
        sector: Industry sector for context.
        target_year_near_term: Near-term target year.
        target_year_long_term: Long-term target year.
        include_prioritisation: Whether to generate prioritisation.
        include_recommendations: Whether to generate recommendations.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Reporting entity name"
    )
    base_year: int = Field(
        ..., ge=2015, le=2030,
        description="Emissions base year"
    )
    scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 1 direct emissions (tCO2e)"
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 market-based emissions (tCO2e)"
    )
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 location-based emissions (tCO2e)"
    )
    categories: List[CategoryInput] = Field(
        default_factory=list,
        description="Per-category screening data"
    )
    supplier_engagement: Optional[SupplierEngagementInput] = Field(
        default=None,
        description="Supplier engagement data"
    )
    sector: str = Field(
        default="general", max_length=100,
        description="Industry sector"
    )
    target_year_near_term: int = Field(
        default=0, ge=0, le=2040,
        description="Near-term target year (0 = auto)"
    )
    target_year_long_term: int = Field(
        default=2050, ge=2035, le=2060,
        description="Long-term target year"
    )
    include_prioritisation: bool = Field(
        default=True,
        description="Whether to generate category prioritisation"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to generate action recommendations"
    )

    @field_validator("target_year_near_term")
    @classmethod
    def validate_near_term(cls, v: int, info: Any) -> int:
        """Auto-calculate near-term year if zero."""
        if v == 0:
            base = info.data.get("base_year", 2023)
            return base + 7
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class CategoryScreeningResult(BaseModel):
    """Screening result for a single Scope 3 category.

    Attributes:
        category: Category identifier.
        category_number: GHG Protocol category number (1-15).
        category_name: Human-readable category name.
        emissions_tco2e: Estimated emissions (tCO2e).
        pct_of_scope3: Percentage of total Scope 3.
        pct_of_total: Percentage of total (S1+S2+S3).
        materiality_level: HIGH/MEDIUM/LOW/NEGLIGIBLE.
        data_quality: Data quality tier.
        data_quality_score: Numeric quality score (0.00-1.00).
        screening_status: Screening completion status.
        is_relevant: Whether category is relevant.
        is_targeted: Whether category is in target boundary.
        target_approach: Target-setting approach used.
        reduction_potential: Estimated reduction potential.
        influence_level: Organisational influence level.
        supplier_engagement_pct: % of suppliers with SBTi targets.
        meets_engagement_target: Whether engagement target is met.
        priority_score: Prioritisation score (0-100).
        priority_rank: Rank among all categories (1 = highest).
        data_improvement_needed: Whether data quality upgrade is needed.
        recommendations: Category-specific recommendations.
    """
    category: str = Field(default="")
    category_number: int = Field(default=0)
    category_name: str = Field(default="")
    emissions_tco2e: Decimal = Field(default=Decimal("0"))
    pct_of_scope3: Decimal = Field(default=Decimal("0"))
    pct_of_total: Decimal = Field(default=Decimal("0"))
    materiality_level: str = Field(default=MaterialityLevel.NOT_ASSESSED.value)
    data_quality: str = Field(default=DataQualityTier.NONE.value)
    data_quality_score: Decimal = Field(default=Decimal("0"))
    screening_status: str = Field(default=ScreeningStatus.NOT_STARTED.value)
    is_relevant: bool = Field(default=True)
    is_targeted: bool = Field(default=False)
    target_approach: str = Field(default=TargetApproach.NOT_SET.value)
    reduction_potential: str = Field(default=ReductionPotential.UNKNOWN.value)
    influence_level: str = Field(default=InfluenceLevel.LIMITED.value)
    supplier_engagement_pct: Decimal = Field(default=Decimal("0"))
    meets_engagement_target: bool = Field(default=False)
    priority_score: Decimal = Field(default=Decimal("0"))
    priority_rank: int = Field(default=0)
    data_improvement_needed: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)

class TriggerAssessment(BaseModel):
    """Scope 3 materiality trigger assessment result.

    Attributes:
        scope1_tco2e: Scope 1 emissions used.
        scope2_market_tco2e: Scope 2 market-based used.
        scope3_total_tco2e: Total Scope 3 emissions.
        total_s1s2s3_tco2e: Sum of S1+S2+S3.
        scope3_fraction: S3 / (S1+S2+S3) as fraction.
        scope3_pct: S3 as percentage of total.
        trigger_threshold_pct: SBTi trigger threshold (40%).
        scope3_target_required: Whether S3 targets are required.
        margin_to_trigger_pct: Distance from trigger (negative = above).
        message: Human-readable trigger assessment.
    """
    scope1_tco2e: Decimal = Field(default=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"))
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"))
    total_s1s2s3_tco2e: Decimal = Field(default=Decimal("0"))
    scope3_fraction: Decimal = Field(default=Decimal("0"))
    scope3_pct: Decimal = Field(default=Decimal("0"))
    trigger_threshold_pct: Decimal = Field(default=Decimal("40.0"))
    scope3_target_required: bool = Field(default=False)
    margin_to_trigger_pct: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="")

class CoverageAssessment(BaseModel):
    """Coverage assessment against SBTi thresholds.

    Attributes:
        targeted_emissions_tco2e: Sum of emissions in targeted categories.
        total_scope3_tco2e: Total Scope 3 emissions.
        coverage_fraction: Targeted / Total as fraction.
        coverage_pct: Coverage as percentage.
        near_term_required_pct: Required coverage for near-term (67%).
        long_term_required_pct: Required coverage for long-term (90%).
        meets_near_term: Whether 67% near-term coverage is met.
        meets_long_term: Whether 90% long-term coverage is met.
        gap_to_near_term_pct: Gap to 67% threshold.
        gap_to_long_term_pct: Gap to 90% threshold.
        categories_targeted: Number of categories in target boundary.
        categories_needed_for_near_term: Additional cats for 67%.
        categories_needed_for_long_term: Additional cats for 90%.
        message: Human-readable coverage assessment.
    """
    targeted_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_scope3_tco2e: Decimal = Field(default=Decimal("0"))
    coverage_fraction: Decimal = Field(default=Decimal("0"))
    coverage_pct: Decimal = Field(default=Decimal("0"))
    near_term_required_pct: Decimal = Field(default=Decimal("67.0"))
    long_term_required_pct: Decimal = Field(default=Decimal("90.0"))
    meets_near_term: bool = Field(default=False)
    meets_long_term: bool = Field(default=False)
    gap_to_near_term_pct: Decimal = Field(default=Decimal("0"))
    gap_to_long_term_pct: Decimal = Field(default=Decimal("0"))
    categories_targeted: int = Field(default=0)
    categories_needed_for_near_term: int = Field(default=0)
    categories_needed_for_long_term: int = Field(default=0)
    message: str = Field(default="")

class SupplierEngagementAssessment(BaseModel):
    """Supplier engagement target assessment result.

    Attributes:
        total_suppliers: Total tier-1 suppliers.
        suppliers_with_sbti: Suppliers with validated SBTi targets.
        suppliers_committed: Suppliers with SBTi commitments.
        current_engagement_pct: Current % with SBTi targets.
        including_committed_pct: % including committed suppliers.
        target_pct: Target engagement percentage.
        target_year: Target year for engagement.
        meets_sbti_minimum: Whether SBTi minimum (67%) is met.
        gap_to_target_pct: Gap to target percentage.
        gap_to_sbti_min_pct: Gap to SBTi 67% minimum.
        covers_material_categories: Whether engagement covers material cats.
        has_annual_review: Whether annual review process exists.
        engagement_score: Overall engagement assessment (0-100).
        recommendations: Engagement-specific recommendations.
    """
    total_suppliers: int = Field(default=0)
    suppliers_with_sbti: int = Field(default=0)
    suppliers_committed: int = Field(default=0)
    current_engagement_pct: Decimal = Field(default=Decimal("0"))
    including_committed_pct: Decimal = Field(default=Decimal("0"))
    target_pct: Decimal = Field(default=Decimal("0"))
    target_year: int = Field(default=0)
    meets_sbti_minimum: bool = Field(default=False)
    gap_to_target_pct: Decimal = Field(default=Decimal("0"))
    gap_to_sbti_min_pct: Decimal = Field(default=Decimal("0"))
    covers_material_categories: bool = Field(default=False)
    has_annual_review: bool = Field(default=False)
    engagement_score: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)

class DataQualityAssessment(BaseModel):
    """Overall data quality assessment across all categories.

    Attributes:
        categories_assessed: Number of categories with data.
        categories_primary: Number with primary data.
        categories_secondary: Number with secondary data.
        categories_proxy: Number with proxy data.
        categories_spend: Number with spend-based data.
        categories_no_data: Number with no data.
        weighted_quality_score: Emissions-weighted quality score.
        simple_average_score: Simple average quality score.
        meets_minimum_for_target: Whether quality supports target-setting.
        lowest_quality_material_cats: Material cats with lowest quality.
        improvement_priorities: Priority categories for data improvement.
        message: Human-readable quality assessment.
    """
    categories_assessed: int = Field(default=0)
    categories_primary: int = Field(default=0)
    categories_secondary: int = Field(default=0)
    categories_proxy: int = Field(default=0)
    categories_spend: int = Field(default=0)
    categories_no_data: int = Field(default=0)
    weighted_quality_score: Decimal = Field(default=Decimal("0"))
    simple_average_score: Decimal = Field(default=Decimal("0"))
    meets_minimum_for_target: bool = Field(default=False)
    lowest_quality_material_cats: List[str] = Field(default_factory=list)
    improvement_priorities: List[str] = Field(default_factory=list)
    message: str = Field(default="")

class MaterialitySummary(BaseModel):
    """Summary of materiality distribution across categories.

    Attributes:
        high_count: Number of HIGH materiality categories.
        medium_count: Number of MEDIUM materiality categories.
        low_count: Number of LOW materiality categories.
        negligible_count: Number of NEGLIGIBLE categories.
        not_assessed_count: Number of NOT_ASSESSED categories.
        high_emissions_tco2e: Total emissions from HIGH categories.
        medium_emissions_tco2e: Total emissions from MEDIUM categories.
        low_emissions_tco2e: Total emissions from LOW categories.
        negligible_emissions_tco2e: Total from NEGLIGIBLE categories.
        high_pct_of_scope3: HIGH categories as % of S3.
        top_3_categories: Top 3 categories by emissions.
        concentration_index: How concentrated S3 is (HHI-style).
    """
    high_count: int = Field(default=0)
    medium_count: int = Field(default=0)
    low_count: int = Field(default=0)
    negligible_count: int = Field(default=0)
    not_assessed_count: int = Field(default=0)
    high_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    medium_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    low_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    negligible_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    high_pct_of_scope3: Decimal = Field(default=Decimal("0"))
    top_3_categories: List[str] = Field(default_factory=list)
    concentration_index: Decimal = Field(default=Decimal("0"))

class PrioritisationResult(BaseModel):
    """Category prioritisation result.

    Attributes:
        ranked_categories: Categories sorted by priority score (highest first).
        quick_wins: Categories with high influence + high reduction potential.
        strategic_priorities: High materiality + improvement needed.
        data_gaps: Categories needing data quality improvement.
        engagement_priorities: Categories where supplier engagement helps.
    """
    ranked_categories: List[str] = Field(default_factory=list)
    quick_wins: List[str] = Field(default_factory=list)
    strategic_priorities: List[str] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)
    engagement_priorities: List[str] = Field(default_factory=list)

class ActionRecommendation(BaseModel):
    """A single action recommendation.

    Attributes:
        action_id: Unique action identifier.
        category: Related Scope 3 category (or 'all').
        priority: Priority level (immediate/short/medium/long).
        action: Description of recommended action.
        rationale: Why this action is recommended.
        estimated_impact: Expected emissions impact description.
        estimated_effort: Level of effort required.
        timeline_months: Estimated implementation time.
    """
    action_id: str = Field(default_factory=_new_uuid)
    category: str = Field(default="all")
    priority: str = Field(default="medium_term")
    action: str = Field(default="")
    rationale: str = Field(default="")
    estimated_impact: str = Field(default="")
    estimated_effort: str = Field(default="medium")
    timeline_months: int = Field(default=12)

class Scope3ScreeningResult(BaseModel):
    """Complete Scope 3 screening result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        base_year: Emissions base year.
        category_results: Per-category screening results.
        trigger_assessment: 40% trigger assessment.
        coverage_assessment: Coverage against 67%/90% thresholds.
        supplier_engagement: Supplier engagement assessment.
        data_quality_assessment: Overall data quality assessment.
        materiality_summary: Materiality distribution summary.
        prioritisation: Category prioritisation.
        recommendations: Action recommendations.
        categories_screened: Number of categories screened.
        categories_relevant: Number of relevant categories.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    base_year: int = Field(default=0)
    category_results: List[CategoryScreeningResult] = Field(
        default_factory=list
    )
    trigger_assessment: Optional[TriggerAssessment] = Field(None)
    coverage_assessment: Optional[CoverageAssessment] = Field(None)
    supplier_engagement: Optional[SupplierEngagementAssessment] = Field(None)
    data_quality_assessment: Optional[DataQualityAssessment] = Field(None)
    materiality_summary: Optional[MaterialitySummary] = Field(None)
    prioritisation: Optional[PrioritisationResult] = Field(None)
    recommendations: List[ActionRecommendation] = Field(default_factory=list)
    categories_screened: int = Field(default=0)
    categories_relevant: int = Field(default=0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Scope3ScreeningEngine:
    """SBTi Scope 3 materiality screening engine.

    Performs systematic 15-category Scope 3 screening including:
      - 40% materiality trigger assessment (S3 / total)
      - Per-category materiality classification (HIGH/MEDIUM/LOW/NEGLIGIBLE)
      - 67%/90% coverage tracking for near-term/long-term targets
      - Supplier engagement target validation
      - Data quality scoring and improvement prioritisation
      - Category prioritisation for target-setting focus

    All calculations use deterministic Decimal arithmetic with SHA-256
    provenance hashing.  No LLM involvement in any calculation path.

    Usage::

        engine = Scope3ScreeningEngine()
        result = engine.screen_all(input_data)
        print(f"S3 trigger: {result.trigger_assessment.scope3_target_required}")
        for cat in result.category_results:
            print(f"  {cat.category_name}: {cat.materiality_level}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise Scope3ScreeningEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - materiality_high_threshold (Decimal)
                - materiality_medium_threshold (Decimal)
                - materiality_low_threshold (Decimal)
                - supplier_engagement_min_pct (Decimal)
                - include_not_relevant (bool)
        """
        self.config = config or {}
        self._mat_high = _decimal(
            self.config.get(
                "materiality_high_threshold", MATERIALITY_HIGH_THRESHOLD
            )
        )
        self._mat_medium = _decimal(
            self.config.get(
                "materiality_medium_threshold", MATERIALITY_MEDIUM_THRESHOLD
            )
        )
        self._mat_low = _decimal(
            self.config.get(
                "materiality_low_threshold", MATERIALITY_LOW_THRESHOLD
            )
        )
        self._engagement_min = _decimal(
            self.config.get(
                "supplier_engagement_min_pct", SUPPLIER_ENGAGEMENT_MIN_PCT
            )
        )
        self._include_not_relevant = bool(
            self.config.get("include_not_relevant", True)
        )
        logger.info(
            "Scope3ScreeningEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def screen_all(
        self, data: Scope3ScreeningInput,
    ) -> Scope3ScreeningResult:
        """Perform complete 15-category Scope 3 screening.

        Orchestrates the full screening pipeline: assesses the 40% trigger,
        classifies each category by materiality, checks coverage against
        67%/90% thresholds, evaluates supplier engagement, scores data
        quality, and produces prioritised recommendations.

        Args:
            data: Validated screening input.

        Returns:
            Scope3ScreeningResult with all assessments and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Scope 3 screening: entity=%s, base=%d, categories=%d",
            data.entity_name, data.base_year, len(data.categories),
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Calculate Scope 3 total from categories
        scope3_total = self._calculate_scope3_total(data.categories)

        # Validate minimum data
        if scope3_total <= Decimal("0") and len(data.categories) > 0:
            warnings.append(
                "Total Scope 3 emissions are zero despite categories "
                "being provided. Check emissions data."
            )
        if len(data.categories) == 0:
            warnings.append(
                "No Scope 3 categories provided. Screening will be limited."
            )

        # Step 2: Trigger assessment
        trigger = self.assess_trigger(
            data.scope1_tco2e,
            data.scope2_market_tco2e,
            scope3_total,
        )

        # Step 3: Screen each category
        category_results = self._screen_categories(
            data.categories, scope3_total,
            data.scope1_tco2e + data.scope2_market_tco2e + scope3_total,
        )

        # Step 4: Coverage assessment
        coverage = self.assess_coverage(
            category_results, scope3_total
        )

        # Step 5: Supplier engagement assessment
        engagement: Optional[SupplierEngagementAssessment] = None
        if data.supplier_engagement is not None:
            engagement = self.assess_supplier_engagement(
                data.supplier_engagement, category_results
            )

        # Step 6: Data quality assessment
        quality = self.assess_data_quality(category_results, scope3_total)

        # Step 7: Materiality summary
        mat_summary = self._build_materiality_summary(
            category_results, scope3_total
        )

        # Step 8: Prioritisation
        prioritisation: Optional[PrioritisationResult] = None
        if data.include_prioritisation:
            category_results = self._calculate_priority_scores(
                category_results
            )
            prioritisation = self._build_prioritisation(category_results)

        # Step 9: Recommendations
        recommendations: List[ActionRecommendation] = []
        if data.include_recommendations:
            recommendations = self._generate_recommendations(
                trigger, coverage, engagement, quality,
                category_results, mat_summary,
            )

        # Step 10: Add warnings for coverage gaps
        if trigger.scope3_target_required and not coverage.meets_near_term:
            warnings.append(
                f"Scope 3 near-term coverage is {coverage.coverage_pct}%, "
                f"below the required 67%. Gap: {coverage.gap_to_near_term_pct}%."
            )
        if trigger.scope3_target_required and not coverage.meets_long_term:
            warnings.append(
                f"Scope 3 long-term coverage is {coverage.coverage_pct}%, "
                f"below the required 90%. Gap: {coverage.gap_to_long_term_pct}%."
            )

        # Screening stats
        cats_screened = sum(
            1 for cr in category_results
            if cr.screening_status in (
                ScreeningStatus.COMPLETE.value,
                ScreeningStatus.PARTIAL.value,
            )
        )
        cats_relevant = sum(
            1 for cr in category_results if cr.is_relevant
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = Scope3ScreeningResult(
            entity_name=data.entity_name,
            base_year=data.base_year,
            category_results=category_results,
            trigger_assessment=trigger,
            coverage_assessment=coverage,
            supplier_engagement=engagement,
            data_quality_assessment=quality,
            materiality_summary=mat_summary,
            prioritisation=prioritisation,
            recommendations=recommendations,
            categories_screened=cats_screened,
            categories_relevant=cats_relevant,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Scope 3 screening complete: %d categories, trigger=%s, "
            "coverage=%.1f%%, quality=%.2f, hash=%s",
            len(category_results),
            trigger.scope3_target_required,
            float(coverage.coverage_pct),
            float(quality.weighted_quality_score),
            result.provenance_hash[:16],
        )
        return result

    def assess_trigger(
        self,
        scope1_tco2e: Decimal,
        scope2_market_tco2e: Decimal,
        scope3_total_tco2e: Decimal,
    ) -> TriggerAssessment:
        """Assess whether Scope 3 targets are required (40% trigger).

        Per SBTi Corporate Manual V5.3, Section 6.1: companies must set
        Scope 3 targets if Scope 3 emissions are >= 40% of total
        (S1 + S2_market + S3).

        Args:
            scope1_tco2e: Scope 1 emissions.
            scope2_market_tco2e: Scope 2 market-based emissions.
            scope3_total_tco2e: Total Scope 3 emissions.

        Returns:
            TriggerAssessment with determination and supporting data.
        """
        total = scope1_tco2e + scope2_market_tco2e + scope3_total_tco2e
        fraction = _safe_divide(scope3_total_tco2e, total)
        pct = _round_val(fraction * Decimal("100"), 2)
        threshold_pct = SCOPE3_TRIGGER_THRESHOLD * Decimal("100")
        required = fraction >= SCOPE3_TRIGGER_THRESHOLD
        margin = _round_val(pct - threshold_pct, 2)

        if required:
            msg = (
                f"Scope 3 emissions represent {pct}% of total emissions, "
                f"exceeding the 40% trigger threshold. "
                f"Scope 3 targets are REQUIRED."
            )
        else:
            msg = (
                f"Scope 3 emissions represent {pct}% of total emissions, "
                f"below the 40% trigger threshold. "
                f"Scope 3 targets are VOLUNTARY but recommended."
            )

        return TriggerAssessment(
            scope1_tco2e=_round_val(scope1_tco2e),
            scope2_market_tco2e=_round_val(scope2_market_tco2e),
            scope3_total_tco2e=_round_val(scope3_total_tco2e),
            total_s1s2s3_tco2e=_round_val(total),
            scope3_fraction=_round_val(fraction, 4),
            scope3_pct=pct,
            trigger_threshold_pct=threshold_pct,
            scope3_target_required=required,
            margin_to_trigger_pct=margin,
            message=msg,
        )

    def assess_coverage(
        self,
        category_results: List[CategoryScreeningResult],
        scope3_total: Decimal,
    ) -> CoverageAssessment:
        """Assess coverage against SBTi 67%/90% thresholds.

        Near-term coverage (67%): SBTi Corporate Manual V5.3, Section 6.3.
        Long-term coverage (90%): SBTi Net-Zero Standard V1.3, Section 5.2.

        Coverage = sum of emissions from targeted categories / total S3.

        Args:
            category_results: Per-category screening results.
            scope3_total: Total Scope 3 emissions.

        Returns:
            CoverageAssessment with gap analysis.
        """
        targeted_total = sum(
            (cr.emissions_tco2e for cr in category_results if cr.is_targeted),
            Decimal("0"),
        )
        fraction = _safe_divide(targeted_total, scope3_total)
        coverage_pct = _round_val(fraction * Decimal("100"), 2)

        nt_pct = SCOPE3_NT_COVERAGE_MIN * Decimal("100")
        lt_pct = SCOPE3_LT_COVERAGE_MIN * Decimal("100")

        meets_nt = coverage_pct >= nt_pct
        meets_lt = coverage_pct >= lt_pct

        gap_nt = _round_val(
            max(Decimal("0"), nt_pct - coverage_pct), 2
        )
        gap_lt = _round_val(
            max(Decimal("0"), lt_pct - coverage_pct), 2
        )

        cats_targeted = sum(
            1 for cr in category_results if cr.is_targeted
        )

        # Calculate how many additional categories are needed
        cats_needed_nt = self._categories_needed_for_coverage(
            category_results, scope3_total, SCOPE3_NT_COVERAGE_MIN
        )
        cats_needed_lt = self._categories_needed_for_coverage(
            category_results, scope3_total, SCOPE3_LT_COVERAGE_MIN
        )

        if meets_lt:
            msg = (
                f"Coverage of {coverage_pct}% meets both near-term (67%) "
                f"and long-term (90%) requirements."
            )
        elif meets_nt:
            msg = (
                f"Coverage of {coverage_pct}% meets near-term (67%) "
                f"requirement but needs {gap_lt}% more for long-term (90%)."
            )
        else:
            msg = (
                f"Coverage of {coverage_pct}% is below near-term (67%) "
                f"requirement. Gap: {gap_nt}%."
            )

        return CoverageAssessment(
            targeted_emissions_tco2e=_round_val(targeted_total),
            total_scope3_tco2e=_round_val(scope3_total),
            coverage_fraction=_round_val(fraction, 4),
            coverage_pct=coverage_pct,
            near_term_required_pct=nt_pct,
            long_term_required_pct=lt_pct,
            meets_near_term=meets_nt,
            meets_long_term=meets_lt,
            gap_to_near_term_pct=gap_nt,
            gap_to_long_term_pct=gap_lt,
            categories_targeted=cats_targeted,
            categories_needed_for_near_term=cats_needed_nt,
            categories_needed_for_long_term=cats_needed_lt,
            message=msg,
        )

    def assess_supplier_engagement(
        self,
        engagement: SupplierEngagementInput,
        category_results: List[CategoryScreeningResult],
    ) -> SupplierEngagementAssessment:
        """Assess supplier engagement target against SBTi requirements.

        Per SBTi Supplier Engagement Guidance (2024): companies may set
        a supplier engagement target requiring a specified percentage of
        suppliers (by emissions) to have SBTi targets within 5 years.
        The minimum threshold is 67% of suppliers.

        Args:
            engagement: Supplier engagement input data.
            category_results: Category results for context.

        Returns:
            SupplierEngagementAssessment with gap analysis.
        """
        total = engagement.total_suppliers
        with_sbti = engagement.suppliers_with_sbti
        committed = engagement.suppliers_committed

        current_pct = Decimal("0")
        incl_committed_pct = Decimal("0")
        if total > 0:
            current_pct = _round_val(
                _decimal(with_sbti) / _decimal(total) * Decimal("100"), 2
            )
            incl_committed_pct = _round_val(
                _decimal(with_sbti + committed)
                / _decimal(total) * Decimal("100"), 2
            )

        target_pct = engagement.target_pct
        if target_pct <= Decimal("0"):
            target_pct = self._engagement_min

        meets_min = current_pct >= self._engagement_min

        gap_target = _round_val(
            max(Decimal("0"), target_pct - current_pct), 2
        )
        gap_min = _round_val(
            max(Decimal("0"), self._engagement_min - current_pct), 2
        )

        # Check if engagement covers material upstream categories
        material_cats_covered = (
            engagement.covers_purchased_goods
            or engagement.covers_upstream_transport
        )

        # Calculate engagement score (0-100)
        score_components: List[Tuple[Decimal, Decimal]] = [
            # Weight, value
            (Decimal("0.40"), min(current_pct, Decimal("100"))),
            (
                Decimal("0.20"),
                Decimal("100") if material_cats_covered else Decimal("0"),
            ),
            (
                Decimal("0.15"),
                Decimal("100") if engagement.annual_review else Decimal("0"),
            ),
            (
                Decimal("0.15"),
                min(incl_committed_pct, Decimal("100")),
            ),
            (
                Decimal("0.10"),
                Decimal("100")
                if engagement.engagement_strategy else Decimal("0"),
            ),
        ]
        engagement_score = sum(
            w * v for w, v in score_components
        )
        engagement_score = _round_val(
            min(engagement_score, Decimal("100")), 1
        )

        recs: List[str] = []
        if not meets_min:
            recs.append(
                f"Increase supplier engagement to at least "
                f"{self._engagement_min}% (currently {current_pct}%)."
            )
        if not material_cats_covered:
            recs.append(
                "Extend engagement programme to cover Category 1 "
                "(Purchased Goods) and/or Category 4 (Upstream Transport)."
            )
        if not engagement.annual_review:
            recs.append(
                "Establish an annual review process for supplier "
                "engagement progress."
            )
        if not engagement.engagement_strategy:
            recs.append(
                "Document a formal supplier engagement strategy "
                "aligned with SBTi guidance."
            )
        if gap_target > Decimal("0"):
            suppliers_needed = 0
            if total > 0:
                pct_gap_frac = gap_target / Decimal("100")
                suppliers_needed = int(
                    (pct_gap_frac * _decimal(total)).to_integral_value()
                )
            recs.append(
                f"Engage approximately {suppliers_needed} additional "
                f"suppliers to reach {target_pct}% target."
            )

        return SupplierEngagementAssessment(
            total_suppliers=total,
            suppliers_with_sbti=with_sbti,
            suppliers_committed=committed,
            current_engagement_pct=current_pct,
            including_committed_pct=incl_committed_pct,
            target_pct=target_pct,
            target_year=engagement.target_year,
            meets_sbti_minimum=meets_min,
            gap_to_target_pct=gap_target,
            gap_to_sbti_min_pct=gap_min,
            covers_material_categories=material_cats_covered,
            has_annual_review=engagement.annual_review,
            engagement_score=engagement_score,
            recommendations=recs,
        )

    def assess_data_quality(
        self,
        category_results: List[CategoryScreeningResult],
        scope3_total: Decimal,
    ) -> DataQualityAssessment:
        """Assess overall data quality across all Scope 3 categories.

        Calculates both a simple average and an emissions-weighted quality
        score.  Identifies material categories with low data quality as
        improvement priorities.

        Args:
            category_results: Per-category screening results.
            scope3_total: Total Scope 3 emissions.

        Returns:
            DataQualityAssessment with scores and improvement priorities.
        """
        assessed = 0
        primary = 0
        secondary = 0
        proxy = 0
        spend = 0
        no_data = 0

        score_sum = Decimal("0")
        weighted_num = Decimal("0")
        weighted_den = Decimal("0")

        lowest_quality_material: List[Tuple[Decimal, str]] = []

        for cr in category_results:
            if not cr.is_relevant:
                continue

            dq_score = cr.data_quality_score
            assessed += 1
            score_sum += dq_score

            if cr.data_quality == DataQualityTier.PRIMARY.value:
                primary += 1
            elif cr.data_quality == DataQualityTier.SECONDARY.value:
                secondary += 1
            elif cr.data_quality == DataQualityTier.PROXY.value:
                proxy += 1
            elif cr.data_quality == DataQualityTier.SPEND.value:
                spend += 1
            else:
                no_data += 1

            # Weighted by emissions
            if cr.emissions_tco2e > Decimal("0"):
                weighted_num += dq_score * cr.emissions_tco2e
                weighted_den += cr.emissions_tco2e

            # Track low-quality material categories
            if (
                cr.materiality_level
                in (MaterialityLevel.HIGH.value, MaterialityLevel.MEDIUM.value)
                and dq_score < MIN_DATA_QUALITY_FOR_TARGET
            ):
                lowest_quality_material.append(
                    (dq_score, cr.category_name)
                )

        simple_avg = _safe_divide(
            score_sum, _decimal(assessed)
        ) if assessed > 0 else Decimal("0")

        weighted_avg = _safe_divide(weighted_num, weighted_den)

        meets_min = weighted_avg >= MIN_DATA_QUALITY_FOR_TARGET

        # Sort by quality score ascending for priorities
        lowest_quality_material.sort(key=lambda x: x[0])
        lqm_names = [name for _, name in lowest_quality_material]

        # Improvement priorities: material categories with quality < 0.50
        improvement = []
        for cr in category_results:
            if (
                cr.is_relevant
                and cr.materiality_level
                in (
                    MaterialityLevel.HIGH.value,
                    MaterialityLevel.MEDIUM.value,
                )
                and cr.data_quality_score < MIN_DATA_QUALITY_FOR_TARGET
            ):
                improvement.append(cr.category_name)

        if meets_min:
            msg = (
                f"Weighted data quality score of "
                f"{_round_val(weighted_avg, 2)} meets the minimum "
                f"threshold of {MIN_DATA_QUALITY_FOR_TARGET} for "
                f"target-setting."
            )
        else:
            msg = (
                f"Weighted data quality score of "
                f"{_round_val(weighted_avg, 2)} is below the minimum "
                f"threshold of {MIN_DATA_QUALITY_FOR_TARGET}. "
                f"Data improvement needed before target-setting."
            )

        return DataQualityAssessment(
            categories_assessed=assessed,
            categories_primary=primary,
            categories_secondary=secondary,
            categories_proxy=proxy,
            categories_spend=spend,
            categories_no_data=no_data,
            weighted_quality_score=_round_val(weighted_avg, 4),
            simple_average_score=_round_val(simple_avg, 4),
            meets_minimum_for_target=meets_min,
            lowest_quality_material_cats=lqm_names,
            improvement_priorities=improvement,
            message=msg,
        )

    def screen_single_category(
        self,
        cat_input: CategoryInput,
        scope3_total: Decimal,
        total_emissions: Decimal,
    ) -> CategoryScreeningResult:
        """Screen a single Scope 3 category.

        Calculates materiality, data quality score, supplier engagement
        percentage, and whether data improvement is needed.

        Args:
            cat_input: Category input data.
            scope3_total: Total Scope 3 emissions.
            total_emissions: Total S1+S2+S3 emissions.

        Returns:
            CategoryScreeningResult with full assessment.
        """
        defn = CATEGORY_DEFINITIONS.get(cat_input.category, {})
        cat_number = int(defn.get("number", "0"))
        cat_name = defn.get("name", cat_input.category)

        emissions = cat_input.emissions_tco2e
        pct_s3 = _safe_pct(emissions, scope3_total)
        pct_total = _safe_pct(emissions, total_emissions)
        fraction_s3 = _safe_divide(emissions, scope3_total)

        # Materiality classification
        materiality = self._classify_materiality(fraction_s3)
        if not cat_input.is_relevant:
            materiality = MaterialityLevel.NOT_ASSESSED.value

        # Data quality score
        dq_score = DATA_QUALITY_SCORES.get(
            cat_input.data_quality, Decimal("0")
        )

        # Supplier engagement percentage
        supplier_pct = Decimal("0")
        meets_engagement = False
        if cat_input.supplier_count > 0:
            supplier_pct = _round_val(
                _decimal(cat_input.suppliers_with_sbti)
                / _decimal(cat_input.supplier_count)
                * Decimal("100"),
                2,
            )
            meets_engagement = supplier_pct >= self._engagement_min

        # Data improvement needed
        data_improvement = (
            cat_input.is_relevant
            and materiality
            in (MaterialityLevel.HIGH.value, MaterialityLevel.MEDIUM.value)
            and dq_score < MIN_DATA_QUALITY_FOR_TARGET
        )

        # Generate per-category recommendations
        recs: List[str] = []
        if data_improvement:
            recs.append(
                f"Upgrade data quality from {cat_input.data_quality} to "
                f"at least secondary (activity-based) for {cat_name}."
            )
        if (
            materiality
            in (MaterialityLevel.HIGH.value, MaterialityLevel.MEDIUM.value)
            and not cat_input.is_targeted
        ):
            recs.append(
                f"Include {cat_name} in Scope 3 target boundary "
                f"(materiality: {materiality})."
            )
        engagement_eligible = defn.get("engagement_eligible", "no") == "yes"
        if (
            engagement_eligible
            and materiality == MaterialityLevel.HIGH.value
            and not meets_engagement
            and cat_input.supplier_count > 0
        ):
            recs.append(
                f"Set supplier engagement target for {cat_name} -- "
                f"currently {supplier_pct}% of suppliers have SBTi targets."
            )

        return CategoryScreeningResult(
            category=cat_input.category,
            category_number=cat_number,
            category_name=cat_name,
            emissions_tco2e=_round_val(emissions),
            pct_of_scope3=_round_val(pct_s3, 2),
            pct_of_total=_round_val(pct_total, 2),
            materiality_level=materiality,
            data_quality=cat_input.data_quality,
            data_quality_score=_round_val(dq_score, 2),
            screening_status=cat_input.screening_status,
            is_relevant=cat_input.is_relevant,
            is_targeted=cat_input.is_targeted,
            target_approach=cat_input.target_approach,
            reduction_potential=cat_input.reduction_potential,
            influence_level=cat_input.influence_level,
            supplier_engagement_pct=supplier_pct,
            meets_engagement_target=meets_engagement,
            priority_score=Decimal("0"),  # Calculated later
            priority_rank=0,
            data_improvement_needed=data_improvement,
            recommendations=recs,
        )

    # ------------------------------------------------------------------ #
    # Category Screening                                                  #
    # ------------------------------------------------------------------ #

    def _screen_categories(
        self,
        categories: List[CategoryInput],
        scope3_total: Decimal,
        total_emissions: Decimal,
    ) -> List[CategoryScreeningResult]:
        """Screen all provided categories.

        Args:
            categories: Input category data.
            scope3_total: Total Scope 3 emissions.
            total_emissions: Total S1+S2+S3 emissions.

        Returns:
            List of CategoryScreeningResult sorted by emissions descending.
        """
        results: List[CategoryScreeningResult] = []
        seen_categories: set = set()

        for cat_input in categories:
            seen_categories.add(cat_input.category)
            result = self.screen_single_category(
                cat_input, scope3_total, total_emissions
            )
            results.append(result)

        # Add placeholder results for any missing categories
        if self._include_not_relevant:
            for cat_enum in Scope3Category:
                if cat_enum.value not in seen_categories:
                    defn = CATEGORY_DEFINITIONS.get(cat_enum.value, {})
                    results.append(CategoryScreeningResult(
                        category=cat_enum.value,
                        category_number=int(defn.get("number", "0")),
                        category_name=defn.get("name", cat_enum.value),
                        emissions_tco2e=Decimal("0"),
                        pct_of_scope3=Decimal("0"),
                        pct_of_total=Decimal("0"),
                        materiality_level=MaterialityLevel.NOT_ASSESSED.value,
                        data_quality=DataQualityTier.NONE.value,
                        data_quality_score=Decimal("0"),
                        screening_status=ScreeningStatus.NOT_STARTED.value,
                        is_relevant=True,
                        is_targeted=False,
                        target_approach=TargetApproach.NOT_SET.value,
                        reduction_potential=ReductionPotential.UNKNOWN.value,
                        influence_level=InfluenceLevel.LIMITED.value,
                        supplier_engagement_pct=Decimal("0"),
                        meets_engagement_target=False,
                        priority_score=Decimal("0"),
                        priority_rank=0,
                        data_improvement_needed=False,
                        recommendations=[
                            f"Screen {defn.get('name', cat_enum.value)} "
                            f"to complete Scope 3 assessment."
                        ],
                    ))

        # Sort by emissions descending
        results.sort(
            key=lambda r: r.emissions_tco2e, reverse=True
        )

        return results

    def _calculate_scope3_total(
        self, categories: List[CategoryInput],
    ) -> Decimal:
        """Sum emissions across all categories.

        Args:
            categories: Input category data.

        Returns:
            Total Scope 3 emissions (Decimal).
        """
        return sum(
            (c.emissions_tco2e for c in categories if c.is_relevant),
            Decimal("0"),
        )

    # ------------------------------------------------------------------ #
    # Materiality Classification                                          #
    # ------------------------------------------------------------------ #

    def _classify_materiality(self, fraction: Decimal) -> str:
        """Classify materiality based on fraction of total Scope 3.

        Thresholds per GHG Protocol Scope 3 Guidance:
            HIGH:       >= 10%
            MEDIUM:     >= 3%  and < 10%
            LOW:        >= 1%  and < 3%
            NEGLIGIBLE: < 1%

        Args:
            fraction: Category emissions / total Scope 3 (as fraction).

        Returns:
            MaterialityLevel value string.
        """
        if fraction >= self._mat_high:
            return MaterialityLevel.HIGH.value
        if fraction >= self._mat_medium:
            return MaterialityLevel.MEDIUM.value
        if fraction >= self._mat_low:
            return MaterialityLevel.LOW.value
        return MaterialityLevel.NEGLIGIBLE.value

    # ------------------------------------------------------------------ #
    # Coverage Helpers                                                    #
    # ------------------------------------------------------------------ #

    def _categories_needed_for_coverage(
        self,
        category_results: List[CategoryScreeningResult],
        scope3_total: Decimal,
        target_fraction: Decimal,
    ) -> int:
        """Calculate additional categories needed to meet coverage target.

        Greedily adds the largest untargeted categories until coverage
        threshold is met.

        Args:
            category_results: Current screening results.
            scope3_total: Total Scope 3 emissions.
            target_fraction: Required coverage fraction (e.g. 0.67).

        Returns:
            Number of additional categories needed (0 if already met).
        """
        if scope3_total <= Decimal("0"):
            return 0

        current_targeted = sum(
            (cr.emissions_tco2e for cr in category_results if cr.is_targeted),
            Decimal("0"),
        )

        current_coverage = _safe_divide(current_targeted, scope3_total)
        if current_coverage >= target_fraction:
            return 0

        # Sort untargeted categories by emissions descending
        untargeted = sorted(
            [cr for cr in category_results if not cr.is_targeted and cr.is_relevant],
            key=lambda cr: cr.emissions_tco2e,
            reverse=True,
        )

        additional = 0
        running_total = current_targeted
        for cr in untargeted:
            running_total += cr.emissions_tco2e
            additional += 1
            if _safe_divide(running_total, scope3_total) >= target_fraction:
                break

        return additional

    # ------------------------------------------------------------------ #
    # Materiality Summary                                                 #
    # ------------------------------------------------------------------ #

    def _build_materiality_summary(
        self,
        category_results: List[CategoryScreeningResult],
        scope3_total: Decimal,
    ) -> MaterialitySummary:
        """Build materiality distribution summary.

        Args:
            category_results: Screening results.
            scope3_total: Total Scope 3 emissions.

        Returns:
            MaterialitySummary with counts and totals.
        """
        high_count = 0
        medium_count = 0
        low_count = 0
        negligible_count = 0
        not_assessed_count = 0

        high_emissions = Decimal("0")
        medium_emissions = Decimal("0")
        low_emissions = Decimal("0")
        negligible_emissions = Decimal("0")

        for cr in category_results:
            ml = cr.materiality_level
            if ml == MaterialityLevel.HIGH.value:
                high_count += 1
                high_emissions += cr.emissions_tco2e
            elif ml == MaterialityLevel.MEDIUM.value:
                medium_count += 1
                medium_emissions += cr.emissions_tco2e
            elif ml == MaterialityLevel.LOW.value:
                low_count += 1
                low_emissions += cr.emissions_tco2e
            elif ml == MaterialityLevel.NEGLIGIBLE.value:
                negligible_count += 1
                negligible_emissions += cr.emissions_tco2e
            else:
                not_assessed_count += 1

        high_pct = _safe_pct(high_emissions, scope3_total)

        # Top 3 categories by emissions
        sorted_cats = sorted(
            category_results,
            key=lambda cr: cr.emissions_tco2e,
            reverse=True,
        )
        top_3 = [
            cr.category_name for cr in sorted_cats[:3]
            if cr.emissions_tco2e > Decimal("0")
        ]

        # Concentration index (HHI-style using category shares)
        hhi = Decimal("0")
        for cr in category_results:
            if scope3_total > Decimal("0"):
                share = cr.emissions_tco2e / scope3_total
                hhi += share * share
        concentration = _round_val(hhi * Decimal("10000"), 0)

        return MaterialitySummary(
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            negligible_count=negligible_count,
            not_assessed_count=not_assessed_count,
            high_emissions_tco2e=_round_val(high_emissions),
            medium_emissions_tco2e=_round_val(medium_emissions),
            low_emissions_tco2e=_round_val(low_emissions),
            negligible_emissions_tco2e=_round_val(negligible_emissions),
            high_pct_of_scope3=_round_val(high_pct, 2),
            top_3_categories=top_3,
            concentration_index=concentration,
        )

    # ------------------------------------------------------------------ #
    # Prioritisation                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_priority_scores(
        self,
        category_results: List[CategoryScreeningResult],
    ) -> List[CategoryScreeningResult]:
        """Calculate priority scores for all categories and assign ranks.

        Priority formula:
            score = (materiality_weight * 0.40
                   + data_quality_inverse * 0.25
                   + reduction_potential * 0.20
                   + influence_score * 0.15) * 100

        Args:
            category_results: Category screening results.

        Returns:
            Updated list with priority_score and priority_rank set.
        """
        materiality_weights: Dict[str, Decimal] = {
            MaterialityLevel.HIGH.value: Decimal("1.0"),
            MaterialityLevel.MEDIUM.value: Decimal("0.6"),
            MaterialityLevel.LOW.value: Decimal("0.3"),
            MaterialityLevel.NEGLIGIBLE.value: Decimal("0.1"),
            MaterialityLevel.NOT_ASSESSED.value: Decimal("0.0"),
        }

        reduction_weights: Dict[str, Decimal] = {
            ReductionPotential.HIGH.value: Decimal("1.0"),
            ReductionPotential.MEDIUM.value: Decimal("0.6"),
            ReductionPotential.LOW.value: Decimal("0.3"),
            ReductionPotential.UNKNOWN.value: Decimal("0.5"),
        }

        influence_weights: Dict[str, Decimal] = {
            InfluenceLevel.DIRECT.value: Decimal("1.0"),
            InfluenceLevel.SIGNIFICANT.value: Decimal("0.7"),
            InfluenceLevel.LIMITED.value: Decimal("0.4"),
            InfluenceLevel.MINIMAL.value: Decimal("0.1"),
        }

        for cr in category_results:
            mat_w = materiality_weights.get(
                cr.materiality_level, Decimal("0")
            )
            # Data quality inverse: lower quality = higher priority
            dq_inv = Decimal("1") - cr.data_quality_score
            red_w = reduction_weights.get(
                cr.reduction_potential, Decimal("0.5")
            )
            inf_w = influence_weights.get(
                cr.influence_level, Decimal("0.4")
            )

            score = (
                mat_w * PRIORITY_WEIGHT_MATERIALITY
                + dq_inv * PRIORITY_WEIGHT_DATA_QUALITY
                + red_w * PRIORITY_WEIGHT_REDUCTION
                + inf_w * PRIORITY_WEIGHT_INFLUENCE
            ) * Decimal("100")

            cr.priority_score = _round_val(score, 2)

        # Assign ranks (highest score = rank 1)
        sorted_by_score = sorted(
            category_results,
            key=lambda cr: cr.priority_score,
            reverse=True,
        )
        for rank, cr in enumerate(sorted_by_score, start=1):
            cr.priority_rank = rank

        return category_results

    def _build_prioritisation(
        self,
        category_results: List[CategoryScreeningResult],
    ) -> PrioritisationResult:
        """Build prioritisation result from scored categories.

        Identifies:
          - Ranked list of all categories
          - Quick wins: high influence + high reduction potential
          - Strategic priorities: high materiality + data improvement needed
          - Data gaps: categories needing quality improvement
          - Engagement priorities: engagement-eligible + high materiality

        Args:
            category_results: Scored category results.

        Returns:
            PrioritisationResult with categorised lists.
        """
        sorted_cats = sorted(
            category_results,
            key=lambda cr: cr.priority_score,
            reverse=True,
        )

        ranked = [cr.category_name for cr in sorted_cats if cr.is_relevant]

        quick_wins = [
            cr.category_name for cr in sorted_cats
            if (
                cr.is_relevant
                and cr.influence_level
                in (InfluenceLevel.DIRECT.value, InfluenceLevel.SIGNIFICANT.value)
                and cr.reduction_potential == ReductionPotential.HIGH.value
            )
        ]

        strategic = [
            cr.category_name for cr in sorted_cats
            if (
                cr.is_relevant
                and cr.materiality_level == MaterialityLevel.HIGH.value
                and cr.data_improvement_needed
            )
        ]

        data_gaps = [
            cr.category_name for cr in sorted_cats
            if cr.is_relevant and cr.data_improvement_needed
        ]

        engagement_eligible_cats = {
            cat_val
            for cat_val, defn in CATEGORY_DEFINITIONS.items()
            if defn.get("engagement_eligible") == "yes"
        }
        engagement_priorities = [
            cr.category_name for cr in sorted_cats
            if (
                cr.is_relevant
                and cr.category in engagement_eligible_cats
                and cr.materiality_level
                in (MaterialityLevel.HIGH.value, MaterialityLevel.MEDIUM.value)
                and not cr.meets_engagement_target
            )
        ]

        return PrioritisationResult(
            ranked_categories=ranked,
            quick_wins=quick_wins,
            strategic_priorities=strategic,
            data_gaps=data_gaps,
            engagement_priorities=engagement_priorities,
        )

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        trigger: TriggerAssessment,
        coverage: CoverageAssessment,
        engagement: Optional[SupplierEngagementAssessment],
        quality: DataQualityAssessment,
        category_results: List[CategoryScreeningResult],
        mat_summary: MaterialitySummary,
    ) -> List[ActionRecommendation]:
        """Generate prioritised action recommendations.

        Produces recommendations based on trigger assessment, coverage gaps,
        engagement gaps, data quality issues, and category-level findings.

        Args:
            trigger: Trigger assessment result.
            coverage: Coverage assessment result.
            engagement: Supplier engagement assessment (optional).
            quality: Data quality assessment result.
            category_results: Per-category results.
            mat_summary: Materiality summary.

        Returns:
            List of ActionRecommendation sorted by priority.
        """
        recs: List[ActionRecommendation] = []

        # R1: Complete screening if not all 15 categories are screened
        not_started = [
            cr for cr in category_results
            if cr.screening_status == ScreeningStatus.NOT_STARTED.value
            and cr.is_relevant
        ]
        if not_started:
            recs.append(ActionRecommendation(
                category="all",
                priority="immediate",
                action=(
                    f"Complete Scope 3 screening for {len(not_started)} "
                    f"unscreened categories."
                ),
                rationale=(
                    "SBTi requires screening of all 15 Scope 3 categories "
                    "to determine materiality (C8)."
                ),
                estimated_impact="Enables accurate materiality assessment",
                estimated_effort="medium",
                timeline_months=3,
            ))

        # R2: Address Scope 3 trigger
        if trigger.scope3_target_required:
            recs.append(ActionRecommendation(
                category="all",
                priority="immediate",
                action=(
                    "Set Scope 3 reduction targets -- S3 represents "
                    f"{trigger.scope3_pct}% of total emissions "
                    "(above 40% threshold)."
                ),
                rationale=(
                    "SBTi Corporate Manual V5.3 requires Scope 3 targets "
                    "when S3 >= 40% of total emissions."
                ),
                estimated_impact="SBTi submission requirement",
                estimated_effort="high",
                timeline_months=6,
            ))

        # R3: Address near-term coverage gap
        if not coverage.meets_near_term and trigger.scope3_target_required:
            recs.append(ActionRecommendation(
                category="all",
                priority="immediate",
                action=(
                    f"Expand Scope 3 target boundary to cover at least "
                    f"67% of S3 emissions. Current: {coverage.coverage_pct}%. "
                    f"Add {coverage.categories_needed_for_near_term} "
                    f"more categories."
                ),
                rationale=(
                    "SBTi requires near-term S3 targets to cover >= 67% "
                    "of total Scope 3 emissions."
                ),
                estimated_impact=(
                    f"Close {coverage.gap_to_near_term_pct}% coverage gap"
                ),
                estimated_effort="medium",
                timeline_months=3,
            ))

        # R4: Address long-term coverage gap
        if (
            coverage.meets_near_term
            and not coverage.meets_long_term
            and trigger.scope3_target_required
        ):
            recs.append(ActionRecommendation(
                category="all",
                priority="short_term",
                action=(
                    f"Plan long-term S3 coverage expansion to 90%. "
                    f"Current: {coverage.coverage_pct}%. "
                    f"Add {coverage.categories_needed_for_long_term} "
                    f"more categories."
                ),
                rationale=(
                    "SBTi Net-Zero Standard requires long-term S3 targets "
                    "to cover >= 90% of total Scope 3."
                ),
                estimated_impact=(
                    f"Close {coverage.gap_to_long_term_pct}% long-term "
                    f"coverage gap"
                ),
                estimated_effort="high",
                timeline_months=12,
            ))

        # R5: Data quality improvements
        if not quality.meets_minimum_for_target:
            recs.append(ActionRecommendation(
                category="all",
                priority="short_term",
                action=(
                    f"Improve data quality for material categories. "
                    f"Current weighted score: "
                    f"{quality.weighted_quality_score} "
                    f"(minimum: {MIN_DATA_QUALITY_FOR_TARGET})."
                ),
                rationale=(
                    "Higher data quality enables more accurate targets "
                    "and reduces uncertainty in SBTi submissions."
                ),
                estimated_impact="Enables reliable target-setting",
                estimated_effort="high",
                timeline_months=6,
            ))

        # R6: Specific data improvement priorities
        for cat_name in quality.improvement_priorities[:5]:
            recs.append(ActionRecommendation(
                category=cat_name,
                priority="short_term",
                action=(
                    f"Upgrade data quality for {cat_name} from "
                    f"spend-based/proxy to activity-based or "
                    f"supplier-specific data."
                ),
                rationale=(
                    f"{cat_name} is a material category with low "
                    f"data quality, limiting target accuracy."
                ),
                estimated_impact="Improved target accuracy for category",
                estimated_effort="medium",
                timeline_months=6,
            ))

        # R7: Supplier engagement
        if engagement and not engagement.meets_sbti_minimum:
            recs.append(ActionRecommendation(
                category="all",
                priority="short_term",
                action=(
                    f"Establish supplier engagement programme to reach "
                    f"{self._engagement_min}% of suppliers with SBTi "
                    f"targets. Current: {engagement.current_engagement_pct}%."
                ),
                rationale=(
                    "SBTi Supplier Engagement Guidance requires "
                    f"{self._engagement_min}% of suppliers (by emissions) "
                    f"to have SBTi targets within "
                    f"{SUPPLIER_ENGAGEMENT_TIMELINE_YEARS} years."
                ),
                estimated_impact="Alternative to absolute S3 reduction",
                estimated_effort="high",
                timeline_months=18,
            ))

        # R8: Include untargeted HIGH categories
        high_untargeted = [
            cr for cr in category_results
            if (
                cr.materiality_level == MaterialityLevel.HIGH.value
                and not cr.is_targeted
            )
        ]
        for cr in high_untargeted[:5]:
            recs.append(ActionRecommendation(
                category=cr.category_name,
                priority="immediate",
                action=(
                    f"Include {cr.category_name} in Scope 3 target "
                    f"boundary ({cr.pct_of_scope3}% of S3, "
                    f"materiality: HIGH)."
                ),
                rationale=(
                    "High-materiality categories should be prioritised "
                    "in the target boundary for SBTi compliance."
                ),
                estimated_impact=f"{cr.pct_of_scope3}% of S3 covered",
                estimated_effort="medium",
                timeline_months=3,
            ))

        # R9: Include untargeted MEDIUM categories if needed for coverage
        if not coverage.meets_near_term:
            medium_untargeted = [
                cr for cr in category_results
                if (
                    cr.materiality_level == MaterialityLevel.MEDIUM.value
                    and not cr.is_targeted
                )
            ]
            for cr in medium_untargeted[:3]:
                recs.append(ActionRecommendation(
                    category=cr.category_name,
                    priority="short_term",
                    action=(
                        f"Consider including {cr.category_name} in "
                        f"target boundary to reach 67% coverage "
                        f"({cr.pct_of_scope3}% of S3)."
                    ),
                    rationale=(
                        "Medium-materiality categories can help close "
                        "the coverage gap to 67% near-term threshold."
                    ),
                    estimated_impact=f"{cr.pct_of_scope3}% additional S3",
                    estimated_effort="medium",
                    timeline_months=6,
                ))

        # R10: Annual review recommendation
        recs.append(ActionRecommendation(
            category="all",
            priority="medium_term",
            action=(
                "Establish annual Scope 3 screening review process "
                "with data quality improvement plan."
            ),
            rationale=(
                "SBTi requires regular review of Scope 3 screening "
                "and progress towards coverage targets."
            ),
            estimated_impact="Ongoing compliance and data improvement",
            estimated_effort="low",
            timeline_months=3,
        ))

        # Sort by priority
        priority_order = {
            "immediate": 0,
            "short_term": 1,
            "medium_term": 2,
            "long_term": 3,
        }
        recs.sort(key=lambda r: priority_order.get(r.priority, 4))

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_summary(
        self, result: Scope3ScreeningResult,
    ) -> Dict[str, Any]:
        """Generate concise summary from screening result.

        Args:
            result: Scope 3 screening result to summarise.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "base_year": result.base_year,
            "categories_screened": result.categories_screened,
            "categories_relevant": result.categories_relevant,
        }

        if result.trigger_assessment:
            summary["scope3_pct_of_total"] = str(
                result.trigger_assessment.scope3_pct
            )
            summary["scope3_target_required"] = (
                result.trigger_assessment.scope3_target_required
            )

        if result.coverage_assessment:
            summary["coverage_pct"] = str(
                result.coverage_assessment.coverage_pct
            )
            summary["meets_near_term_coverage"] = (
                result.coverage_assessment.meets_near_term
            )
            summary["meets_long_term_coverage"] = (
                result.coverage_assessment.meets_long_term
            )

        if result.data_quality_assessment:
            summary["weighted_quality_score"] = str(
                result.data_quality_assessment.weighted_quality_score
            )

        if result.materiality_summary:
            summary["high_materiality_categories"] = (
                result.materiality_summary.high_count
            )
            summary["top_3_categories"] = (
                result.materiality_summary.top_3_categories
            )

        if result.supplier_engagement:
            summary["supplier_engagement_pct"] = str(
                result.supplier_engagement.current_engagement_pct
            )

        summary["recommendations_count"] = len(result.recommendations)
        summary["warnings_count"] = len(result.warnings)
        summary["provenance_hash"] = _compute_hash(summary)
        return summary

    def get_category_definitions(self) -> List[Dict[str, str]]:
        """Return the full list of 15 Scope 3 category definitions.

        Returns:
            List of dicts with category value, number, name, description.
        """
        return [
            {
                "category": cat_val,
                "number": defn["number"],
                "name": defn["name"],
                "description": defn["description"],
                "typical_methods": defn["typical_methods"],
                "typical_materiality": defn["typical_materiality"],
                "engagement_eligible": defn["engagement_eligible"],
            }
            for cat_val, defn in CATEGORY_DEFINITIONS.items()
        ]

    def get_materiality_thresholds(self) -> Dict[str, str]:
        """Return materiality threshold reference.

        Returns:
            Dict mapping level to threshold description.
        """
        return {
            MaterialityLevel.HIGH.value: (
                f">= {self._mat_high * 100}% of total Scope 3"
            ),
            MaterialityLevel.MEDIUM.value: (
                f">= {self._mat_medium * 100}% and "
                f"< {self._mat_high * 100}% of total Scope 3"
            ),
            MaterialityLevel.LOW.value: (
                f">= {self._mat_low * 100}% and "
                f"< {self._mat_medium * 100}% of total Scope 3"
            ),
            MaterialityLevel.NEGLIGIBLE.value: (
                f"< {self._mat_low * 100}% of total Scope 3"
            ),
        }

    def get_coverage_requirements(self) -> Dict[str, str]:
        """Return SBTi coverage requirements.

        Returns:
            Dict mapping target type to coverage requirement.
        """
        return {
            "near_term": (
                f">= {SCOPE3_NT_COVERAGE_MIN * 100}% of total S3 "
                f"(SBTi Corporate Manual V5.3, Section 6.3)"
            ),
            "long_term": (
                f">= {SCOPE3_LT_COVERAGE_MIN * 100}% of total S3 "
                f"(SBTi Net-Zero Standard V1.3, Section 5.2)"
            ),
        }

    def get_data_quality_tiers(self) -> Dict[str, Dict[str, Any]]:
        """Return data quality tier definitions with scores.

        Returns:
            Dict mapping tier to score and description.
        """
        descriptions = {
            DataQualityTier.PRIMARY.value: (
                "Supplier-specific, measured data"
            ),
            DataQualityTier.SECONDARY.value: (
                "Industry-average, activity-based data"
            ),
            DataQualityTier.PROXY.value: "Proxy or modelled data",
            DataQualityTier.SPEND.value: "Spend-based EEIO estimates",
            DataQualityTier.NONE.value: "No data available",
        }
        return {
            tier: {
                "score": str(score),
                "description": descriptions.get(tier, ""),
            }
            for tier, score in DATA_QUALITY_SCORES.items()
        }

    def calculate_minimum_categories_for_coverage(
        self,
        categories: List[CategoryInput],
        target_coverage: Decimal = SCOPE3_NT_COVERAGE_MIN,
    ) -> List[str]:
        """Determine minimum set of categories to meet coverage target.

        Uses a greedy algorithm: selects categories by descending emissions
        until the coverage threshold is met.

        Args:
            categories: Category input data.
            target_coverage: Required coverage fraction (default 0.67).

        Returns:
            List of category identifiers in selection order.
        """
        scope3_total = self._calculate_scope3_total(categories)
        if scope3_total <= Decimal("0"):
            return []

        # Sort by emissions descending
        sorted_cats = sorted(
            [c for c in categories if c.is_relevant],
            key=lambda c: c.emissions_tco2e,
            reverse=True,
        )

        selected: List[str] = []
        running_total = Decimal("0")
        for cat in sorted_cats:
            selected.append(cat.category)
            running_total += cat.emissions_tco2e
            if _safe_divide(running_total, scope3_total) >= target_coverage:
                break

        return selected

    def estimate_coverage_with_additions(
        self,
        current_results: List[CategoryScreeningResult],
        additional_categories: List[str],
        scope3_total: Decimal,
    ) -> Decimal:
        """Estimate coverage if additional categories are targeted.

        Args:
            current_results: Current screening results.
            additional_categories: Category identifiers to add.
            scope3_total: Total Scope 3 emissions.

        Returns:
            Projected coverage percentage.
        """
        if scope3_total <= Decimal("0"):
            return Decimal("0")

        current = sum(
            (cr.emissions_tco2e for cr in current_results if cr.is_targeted),
            Decimal("0"),
        )

        additional = sum(
            (
                cr.emissions_tco2e
                for cr in current_results
                if cr.category in additional_categories and not cr.is_targeted
            ),
            Decimal("0"),
        )

        projected = _safe_pct(current + additional, scope3_total)
        return _round_val(projected, 2)

    def validate_target_approach(
        self,
        category: str,
        approach: str,
        materiality: str,
    ) -> Tuple[bool, str]:
        """Validate whether a target approach is appropriate for a category.

        Args:
            category: Scope 3 category identifier.
            approach: Proposed target approach.
            materiality: Category materiality level.

        Returns:
            Tuple of (is_valid, message).
        """
        defn = CATEGORY_DEFINITIONS.get(category, {})
        engagement_eligible = defn.get("engagement_eligible", "no") == "yes"
        cat_name = defn.get("name", category)

        if approach == TargetApproach.NOT_SET.value:
            if materiality in (
                MaterialityLevel.HIGH.value,
                MaterialityLevel.MEDIUM.value,
            ):
                return False, (
                    f"{cat_name} has {materiality} materiality but no "
                    f"target is set. A target approach is required."
                )
            return True, f"{cat_name} has {materiality} materiality."

        if (
            approach == TargetApproach.SUPPLIER_ENGAGEMENT.value
            and not engagement_eligible
        ):
            return False, (
                f"Supplier engagement targets are not applicable to "
                f"{cat_name}. Use absolute or intensity approach."
            )

        if approach in (
            TargetApproach.ABSOLUTE.value,
            TargetApproach.INTENSITY.value,
            TargetApproach.COMBINED.value,
        ):
            return True, (
                f"{approach} target approach is valid for {cat_name}."
            )

        if (
            approach == TargetApproach.SUPPLIER_ENGAGEMENT.value
            and engagement_eligible
        ):
            return True, (
                f"Supplier engagement target is valid for {cat_name}."
            )

        return False, f"Unknown target approach: {approach}."

    def get_engagement_eligible_categories(self) -> List[Dict[str, str]]:
        """Return categories eligible for supplier engagement targets.

        Returns:
            List of dicts with category and name for eligible categories.
        """
        return [
            {
                "category": cat_val,
                "number": defn["number"],
                "name": defn["name"],
            }
            for cat_val, defn in CATEGORY_DEFINITIONS.items()
            if defn.get("engagement_eligible") == "yes"
        ]

    def calculate_weighted_reduction_needed(
        self,
        category_results: List[CategoryScreeningResult],
        scope3_total: Decimal,
        target_reduction_pct: Decimal,
    ) -> Dict[str, Decimal]:
        """Calculate per-category reduction needed to meet overall target.

        Distributes the overall Scope 3 reduction target across targeted
        categories proportionally to their emissions share.

        Args:
            category_results: Category screening results.
            scope3_total: Total Scope 3 emissions.
            target_reduction_pct: Overall S3 reduction target (%).

        Returns:
            Dict mapping category to required reduction in tCO2e.
        """
        if scope3_total <= Decimal("0"):
            return {}

        total_reduction_tco2e = (
            scope3_total * target_reduction_pct / Decimal("100")
        )

        targeted_total = sum(
            (cr.emissions_tco2e for cr in category_results if cr.is_targeted),
            Decimal("0"),
        )
        if targeted_total <= Decimal("0"):
            return {}

        result: Dict[str, Decimal] = {}
        for cr in category_results:
            if cr.is_targeted and cr.emissions_tco2e > Decimal("0"):
                share = cr.emissions_tco2e / targeted_total
                reduction = _round_val(
                    total_reduction_tco2e * share
                )
                result[cr.category] = reduction

        return result
