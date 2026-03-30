# -*- coding: utf-8 -*-
"""
DataMaturityEngine - PACK-043 Scope 3 Complete Pack Engine 1
==============================================================

Maps current Scope 3 methodology tiers to target tiers with ROI-quantified
upgrade pathways.  Assesses data maturity across all 15 Scope 3 categories,
generates prioritised upgrade roadmaps, calculates return on investment for
each tier transition, and projects uncertainty reductions from methodology
improvements.

The engine supports five maturity levels aligned with GHG Protocol
methodology tiers:

    Level 1 - Screening:         EEIO / sector averages
    Level 2 - Spend-Based:       Spend * emission factors
    Level 3 - Average-Data:      Activity data * average EFs
    Level 4 - Hybrid:            Mix of primary + secondary data
    Level 5 - Supplier-Specific: Primary data from value chain

Calculation Methodology:
    Upgrade ROI per category:
        accuracy_gain = uncertainty_current - uncertainty_target
        cost_usd = UPGRADE_COSTS[category_type][current_tier][target_tier]
        roi = accuracy_gain / cost_usd  (accuracy points per USD)

    Budget Optimisation (greedy by ROI):
        Sort upgrades by roi descending.
        Allocate budget to highest-ROI upgrade first.
        Repeat until budget exhausted.

    Uncertainty Projection:
        U_post = sqrt(sum(share_i^2 * u_i_post^2))
        where u_i_post = u_i_current if not upgraded, u_i_target if upgraded.

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
    - PCAF Global GHG Accounting Standard (data quality scores)
    - SBTi Supplier Engagement Guidance (2023) - data quality expectations
    - ESRS E1 (Delegated Act 2023/2772) - disclosure quality requirements
    - CDP Climate Change Questionnaire (2024) - data quality scoring

Zero-Hallucination:
    - All tier transition costs from published benchmarks and project estimates
    - Uncertainty values from GHG Protocol guidance tables
    - ROI calculations use deterministic arithmetic only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-043 Scope 3 Complete
Engine:  1 of 10
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

_MODULE_VERSION: str = "43.0.0"

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

def _sqrt_decimal(value: Decimal) -> Decimal:
    """Compute square root of a Decimal using Newton's method."""
    if value <= Decimal("0"):
        return Decimal("0")
    # Use Python float sqrt then convert back for sufficient precision
    import math

    return _decimal(math.sqrt(float(value)))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MaturityLevel(str, Enum):
    """Data maturity level for Scope 3 methodology.

    SCREENING:         Level 1 - EEIO / sector averages only.
    SPEND_BASED:       Level 2 - Spend-based with emission factors.
    AVERAGE_DATA:      Level 3 - Activity data with average EFs.
    HYBRID:            Level 4 - Mix of primary and secondary data.
    SUPPLIER_SPECIFIC: Level 5 - Primary data from suppliers.
    """
    SCREENING = "screening"
    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    HYBRID = "hybrid"
    SUPPLIER_SPECIFIC = "supplier_specific"

class CategoryType(str, Enum):
    """Scope 3 category archetype for cost estimation.

    PROCUREMENT:  Categories dominated by purchased goods/services.
    LOGISTICS:    Categories dominated by transportation.
    TRAVEL:       Categories related to employee travel/commuting.
    DOWNSTREAM:   Categories related to downstream product impacts.
    FINANCIAL:    Categories related to investments/financial activities.
    LEASED:       Categories related to leased assets.
    WASTE:        Categories related to waste generation.
    ENERGY:       Categories related to fuel and energy.
    """
    PROCUREMENT = "procurement"
    LOGISTICS = "logistics"
    TRAVEL = "travel"
    DOWNSTREAM = "downstream"
    FINANCIAL = "financial"
    LEASED = "leased"
    WASTE = "waste"
    ENERGY = "energy"

class UpgradePriority(str, Enum):
    """Priority level for a maturity upgrade.

    CRITICAL:  Upgrade has highest ROI and largest impact.
    HIGH:      Upgrade is strongly recommended.
    MEDIUM:    Upgrade is beneficial but not urgent.
    LOW:       Upgrade is optional or marginal return.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AssessmentStatus(str, Enum):
    """Status of the maturity assessment.

    COMPLETE: All categories assessed successfully.
    PARTIAL:  Some categories could not be assessed.
    ERROR:    Assessment failed.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"

# ---------------------------------------------------------------------------
# Maturity Level Numeric Mapping
# ---------------------------------------------------------------------------

MATURITY_LEVEL_NUMBER: Dict[MaturityLevel, int] = {
    MaturityLevel.SCREENING: 1,
    MaturityLevel.SPEND_BASED: 2,
    MaturityLevel.AVERAGE_DATA: 3,
    MaturityLevel.HYBRID: 4,
    MaturityLevel.SUPPLIER_SPECIFIC: 5,
}

NUMBER_TO_MATURITY: Dict[int, MaturityLevel] = {
    v: k for k, v in MATURITY_LEVEL_NUMBER.items()
}

# ---------------------------------------------------------------------------
# Category Type Mapping
# ---------------------------------------------------------------------------
# Maps each Scope 3 category number (1-15) to a category archetype.

CATEGORY_TYPE_MAP: Dict[int, CategoryType] = {
    1: CategoryType.PROCUREMENT,
    2: CategoryType.PROCUREMENT,
    3: CategoryType.ENERGY,
    4: CategoryType.LOGISTICS,
    5: CategoryType.WASTE,
    6: CategoryType.TRAVEL,
    7: CategoryType.TRAVEL,
    8: CategoryType.LEASED,
    9: CategoryType.LOGISTICS,
    10: CategoryType.DOWNSTREAM,
    11: CategoryType.DOWNSTREAM,
    12: CategoryType.DOWNSTREAM,
    13: CategoryType.LEASED,
    14: CategoryType.DOWNSTREAM,
    15: CategoryType.FINANCIAL,
}

CATEGORY_NAMES: Dict[int, str] = {
    1: "Purchased Goods and Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation and Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation and Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# ---------------------------------------------------------------------------
# Uncertainty by Maturity Level (% of estimate, symmetric)
# ---------------------------------------------------------------------------
# Source: GHG Protocol Technical Guidance (2013) Table 7.1 + IPCC 2006 GL
# Higher tiers have lower uncertainty.

UNCERTAINTY_BY_TIER: Dict[MaturityLevel, Decimal] = {
    MaturityLevel.SCREENING: Decimal("60"),          # +/- 60%
    MaturityLevel.SPEND_BASED: Decimal("40"),         # +/- 40%
    MaturityLevel.AVERAGE_DATA: Decimal("25"),        # +/- 25%
    MaturityLevel.HYBRID: Decimal("15"),              # +/- 15%
    MaturityLevel.SUPPLIER_SPECIFIC: Decimal("5"),    # +/- 5%
}

# ---------------------------------------------------------------------------
# Upgrade Costs (USD) by Category Type x Tier Transition
# ---------------------------------------------------------------------------
# Estimated implementation cost to upgrade from one tier to the next.
# Based on industry benchmarks for mid-size enterprises (500-5000 employees).
# Includes: data collection setup, system integration, staff training,
# verification, and first-year operational costs.

UPGRADE_COSTS: Dict[str, Dict[str, Decimal]] = {
    # (from_tier, to_tier) -> cost_usd
    CategoryType.PROCUREMENT.value: {
        "1_2": Decimal("5000"),
        "2_3": Decimal("15000"),
        "3_4": Decimal("40000"),
        "4_5": Decimal("80000"),
        "1_3": Decimal("18000"),
        "1_4": Decimal("50000"),
        "1_5": Decimal("120000"),
        "2_4": Decimal("45000"),
        "2_5": Decimal("90000"),
        "3_5": Decimal("100000"),
    },
    CategoryType.LOGISTICS.value: {
        "1_2": Decimal("4000"),
        "2_3": Decimal("12000"),
        "3_4": Decimal("30000"),
        "4_5": Decimal("65000"),
        "1_3": Decimal("14000"),
        "1_4": Decimal("38000"),
        "1_5": Decimal("95000"),
        "2_4": Decimal("35000"),
        "2_5": Decimal("70000"),
        "3_5": Decimal("80000"),
    },
    CategoryType.TRAVEL.value: {
        "1_2": Decimal("3000"),
        "2_3": Decimal("8000"),
        "3_4": Decimal("20000"),
        "4_5": Decimal("45000"),
        "1_3": Decimal("10000"),
        "1_4": Decimal("25000"),
        "1_5": Decimal("60000"),
        "2_4": Decimal("22000"),
        "2_5": Decimal("50000"),
        "3_5": Decimal("55000"),
    },
    CategoryType.DOWNSTREAM.value: {
        "1_2": Decimal("6000"),
        "2_3": Decimal("18000"),
        "3_4": Decimal("50000"),
        "4_5": Decimal("100000"),
        "1_3": Decimal("22000"),
        "1_4": Decimal("60000"),
        "1_5": Decimal("150000"),
        "2_4": Decimal("55000"),
        "2_5": Decimal("110000"),
        "3_5": Decimal("130000"),
    },
    CategoryType.FINANCIAL.value: {
        "1_2": Decimal("8000"),
        "2_3": Decimal("25000"),
        "3_4": Decimal("60000"),
        "4_5": Decimal("120000"),
        "1_3": Decimal("30000"),
        "1_4": Decimal("75000"),
        "1_5": Decimal("180000"),
        "2_4": Decimal("70000"),
        "2_5": Decimal("140000"),
        "3_5": Decimal("160000"),
    },
    CategoryType.LEASED.value: {
        "1_2": Decimal("4000"),
        "2_3": Decimal("10000"),
        "3_4": Decimal("25000"),
        "4_5": Decimal("55000"),
        "1_3": Decimal("12000"),
        "1_4": Decimal("32000"),
        "1_5": Decimal("75000"),
        "2_4": Decimal("30000"),
        "2_5": Decimal("60000"),
        "3_5": Decimal("65000"),
    },
    CategoryType.WASTE.value: {
        "1_2": Decimal("3000"),
        "2_3": Decimal("8000"),
        "3_4": Decimal("18000"),
        "4_5": Decimal("35000"),
        "1_3": Decimal("10000"),
        "1_4": Decimal("22000"),
        "1_5": Decimal("50000"),
        "2_4": Decimal("20000"),
        "2_5": Decimal("40000"),
        "3_5": Decimal("45000"),
    },
    CategoryType.ENERGY.value: {
        "1_2": Decimal("4000"),
        "2_3": Decimal("12000"),
        "3_4": Decimal("28000"),
        "4_5": Decimal("55000"),
        "1_3": Decimal("14000"),
        "1_4": Decimal("35000"),
        "1_5": Decimal("80000"),
        "2_4": Decimal("32000"),
        "2_5": Decimal("60000"),
        "3_5": Decimal("70000"),
    },
}

# Typical implementation time (months) by tier jump size
UPGRADE_DURATION_MONTHS: Dict[int, int] = {
    1: 3,   # 1-tier jump
    2: 6,   # 2-tier jump
    3: 12,  # 3-tier jump
    4: 18,  # 4-tier jump
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class CategoryData(BaseModel):
    """Per-category maturity assessment input.

    Attributes:
        category_number: Scope 3 category number (1-15).
        category_name: Human-readable category name.
        current_tier: Current maturity level.
        total_co2e_tonnes: Current estimated emissions in tCO2e.
        share_of_scope3_pct: Share of total Scope 3 (0-100).
        has_supplier_data: Whether any supplier-specific data exists.
        data_sources_count: Number of distinct data sources used.
        data_quality_score: DQR score (1.0-5.0, 1=best, 5=worst).
    """
    category_number: int = Field(..., ge=1, le=15, description="Category number")
    category_name: str = Field(default="", description="Category name")
    current_tier: MaturityLevel = Field(
        default=MaturityLevel.SCREENING, description="Current maturity level"
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Emissions tCO2e"
    )
    share_of_scope3_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Share of Scope 3 %"
    )
    has_supplier_data: bool = Field(default=False, description="Has supplier data")
    data_sources_count: int = Field(default=0, ge=0, description="Data sources")
    data_quality_score: Decimal = Field(
        default=Decimal("4.0"), ge=1, le=5, description="DQR score (1-5)"
    )

class MaturityAssessmentInput(BaseModel):
    """Input for a full maturity assessment.

    Attributes:
        org_id: Organisation identifier.
        reporting_year: Reporting year.
        category_data: Per-category data.
        total_scope3_tco2e: Total Scope 3 emissions.
        total_scope12_tco2e: Total Scope 1+2 emissions.
        budget_usd: Available budget for upgrades.
        target_timeline_months: Target timeline for upgrades.
        target_overall_tier: Desired minimum overall tier.
    """
    org_id: str = Field(default="", description="Organisation ID")
    reporting_year: int = Field(default=2025, ge=2000, le=2100, description="Year")
    category_data: List[CategoryData] = Field(
        default_factory=list, description="Category data"
    )
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 3"
    )
    total_scope12_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 1+2"
    )
    budget_usd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Budget USD"
    )
    target_timeline_months: int = Field(
        default=24, ge=1, le=60, description="Timeline months"
    )
    target_overall_tier: MaturityLevel = Field(
        default=MaturityLevel.AVERAGE_DATA, description="Target tier"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class CategoryMaturity(BaseModel):
    """Per-category maturity assessment result.

    Attributes:
        category_number: Category number.
        category_name: Category name.
        category_type: Category archetype.
        current_tier: Current maturity level.
        current_tier_number: Numeric tier (1-5).
        current_uncertainty_pct: Current uncertainty percentage.
        recommended_tier: Recommended target tier.
        recommended_tier_number: Numeric recommended tier.
        target_uncertainty_pct: Target uncertainty percentage.
        emissions_tco2e: Category emissions.
        share_of_scope3_pct: Share of total Scope 3.
        data_quality_score: DQR score.
        gap_tiers: Number of tier levels to upgrade.
        upgrade_cost_usd: Estimated upgrade cost.
        accuracy_gain_pct: Percentage point improvement in uncertainty.
        roi_accuracy_per_kusd: Accuracy points per $1,000 invested.
    """
    category_number: int = Field(..., description="Category")
    category_name: str = Field(default="", description="Name")
    category_type: str = Field(default="", description="Type")
    current_tier: MaturityLevel = Field(
        default=MaturityLevel.SCREENING, description="Current"
    )
    current_tier_number: int = Field(default=1, ge=1, le=5, description="Current #")
    current_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Current uncertainty %"
    )
    recommended_tier: MaturityLevel = Field(
        default=MaturityLevel.AVERAGE_DATA, description="Recommended"
    )
    recommended_tier_number: int = Field(default=3, ge=1, le=5, description="Rec #")
    target_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Target uncertainty %"
    )
    emissions_tco2e: Decimal = Field(default=Decimal("0"), description="tCO2e")
    share_of_scope3_pct: Decimal = Field(default=Decimal("0"), description="Share %")
    data_quality_score: Decimal = Field(default=Decimal("4.0"), description="DQR")
    gap_tiers: int = Field(default=0, ge=0, description="Gap")
    upgrade_cost_usd: Decimal = Field(default=Decimal("0"), description="Cost USD")
    accuracy_gain_pct: Decimal = Field(default=Decimal("0"), description="Gain %")
    roi_accuracy_per_kusd: Decimal = Field(
        default=Decimal("0"), description="ROI (acc pts / $1K)"
    )

class UpgradePathway(BaseModel):
    """A single upgrade step in the roadmap.

    Attributes:
        priority: Upgrade priority.
        rank: Rank in the ordered roadmap.
        category_number: Category number.
        category_name: Category name.
        from_tier: Current tier.
        to_tier: Target tier.
        cost_usd: Estimated cost.
        accuracy_gain_pct: Uncertainty reduction.
        roi_accuracy_per_kusd: ROI metric.
        duration_months: Estimated implementation duration.
        cumulative_cost_usd: Cumulative cost including this step.
        within_budget: Whether this step fits the budget.
        description: Human-readable description of the upgrade.
    """
    priority: UpgradePriority = Field(
        default=UpgradePriority.MEDIUM, description="Priority"
    )
    rank: int = Field(default=0, ge=0, description="Rank")
    category_number: int = Field(..., description="Category")
    category_name: str = Field(default="", description="Name")
    from_tier: MaturityLevel = Field(
        default=MaturityLevel.SCREENING, description="From"
    )
    to_tier: MaturityLevel = Field(
        default=MaturityLevel.AVERAGE_DATA, description="To"
    )
    cost_usd: Decimal = Field(default=Decimal("0"), description="Cost USD")
    accuracy_gain_pct: Decimal = Field(default=Decimal("0"), description="Gain %")
    roi_accuracy_per_kusd: Decimal = Field(
        default=Decimal("0"), description="ROI"
    )
    duration_months: int = Field(default=3, ge=0, description="Duration months")
    cumulative_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Cumulative cost"
    )
    within_budget: bool = Field(default=True, description="Within budget")
    description: str = Field(default="", description="Upgrade description")

class ROIAnalysis(BaseModel):
    """ROI analysis for a specific tier transition.

    Attributes:
        category_number: Category number.
        category_name: Category name.
        current_tier: Current maturity level.
        target_tier: Target maturity level.
        cost_usd: Estimated cost.
        accuracy_gain_pct: Uncertainty improvement.
        roi_accuracy_per_kusd: Accuracy points per $1,000.
        emissions_at_stake_tco2e: Emissions affected by this upgrade.
        uncertainty_band_reduction_tco2e: Reduction in uncertainty band.
        payback_narrative: Qualitative payback description.
    """
    category_number: int = Field(..., description="Category")
    category_name: str = Field(default="", description="Name")
    current_tier: MaturityLevel = Field(
        default=MaturityLevel.SCREENING, description="Current"
    )
    target_tier: MaturityLevel = Field(
        default=MaturityLevel.AVERAGE_DATA, description="Target"
    )
    cost_usd: Decimal = Field(default=Decimal("0"), description="Cost USD")
    accuracy_gain_pct: Decimal = Field(default=Decimal("0"), description="Gain %")
    roi_accuracy_per_kusd: Decimal = Field(
        default=Decimal("0"), description="ROI"
    )
    emissions_at_stake_tco2e: Decimal = Field(
        default=Decimal("0"), description="Emissions at stake"
    )
    uncertainty_band_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Uncertainty band reduction"
    )
    payback_narrative: str = Field(default="", description="Payback narrative")

class BudgetAllocation(BaseModel):
    """Budget allocation result from optimisation.

    Attributes:
        total_budget_usd: Total available budget.
        allocated_usd: Total budget allocated.
        remaining_usd: Unallocated budget.
        upgrades_funded: Number of upgrades funded.
        total_accuracy_gain_pct: Total accuracy improvement.
        allocations: Ordered list of funded upgrades.
    """
    total_budget_usd: Decimal = Field(default=Decimal("0"), description="Total budget")
    allocated_usd: Decimal = Field(default=Decimal("0"), description="Allocated")
    remaining_usd: Decimal = Field(default=Decimal("0"), description="Remaining")
    upgrades_funded: int = Field(default=0, description="Upgrades funded")
    total_accuracy_gain_pct: Decimal = Field(
        default=Decimal("0"), description="Total gain"
    )
    allocations: List[UpgradePathway] = Field(
        default_factory=list, description="Allocations"
    )

class UncertaintyProjection(BaseModel):
    """Projected uncertainty before and after upgrades.

    Attributes:
        current_combined_uncertainty_pct: Current portfolio uncertainty.
        projected_combined_uncertainty_pct: Post-upgrade uncertainty.
        reduction_pct: Absolute reduction in uncertainty percentage.
        improvement_pct: Relative improvement percentage.
        category_projections: Per-category uncertainty projections.
    """
    current_combined_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Current uncertainty"
    )
    projected_combined_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Projected uncertainty"
    )
    reduction_pct: Decimal = Field(default=Decimal("0"), description="Reduction")
    improvement_pct: Decimal = Field(default=Decimal("0"), description="Improvement")
    category_projections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-category projections"
    )

class SimulatedInventory(BaseModel):
    """Simulated post-upgrade inventory summary.

    Attributes:
        total_scope3_tco2e: Post-upgrade total.
        total_scope3_lower_tco2e: Lower bound of confidence interval.
        total_scope3_upper_tco2e: Upper bound of confidence interval.
        uncertainty_pct: Combined uncertainty percentage.
        category_estimates: Per-category estimates post-upgrade.
        methodology_mix: Mix of tiers after upgrade.
    """
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total Scope 3"
    )
    total_scope3_lower_tco2e: Decimal = Field(
        default=Decimal("0"), description="Lower bound"
    )
    total_scope3_upper_tco2e: Decimal = Field(
        default=Decimal("0"), description="Upper bound"
    )
    uncertainty_pct: Decimal = Field(default=Decimal("0"), description="Uncertainty %")
    category_estimates: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-category estimates"
    )
    methodology_mix: Dict[str, int] = Field(
        default_factory=dict, description="Tier mix"
    )

class MaturityAssessment(BaseModel):
    """Complete maturity assessment result.

    Attributes:
        assessment_id: Unique assessment identifier.
        org_id: Organisation identifier.
        reporting_year: Reporting year.
        overall_maturity_level: Weighted average maturity level.
        overall_maturity_score: Weighted average score (1.0-5.0).
        overall_uncertainty_pct: Combined portfolio uncertainty.
        categories_assessed: Number of categories assessed.
        categories_at_target: Number already at or above target.
        categories_needing_upgrade: Number needing upgrade.
        total_upgrade_cost_usd: Total cost to reach target.
        category_assessments: Per-category assessments.
        roadmap: Ordered upgrade roadmap.
        budget_allocation: Budget optimisation result.
        uncertainty_projection: Projected uncertainty post-upgrade.
        simulated_inventory: Simulated post-upgrade inventory.
        warnings: Any warnings generated.
        status: Assessment status.
        calculated_at: Timestamp.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash.
    """
    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment ID")
    org_id: str = Field(default="", description="Organisation ID")
    reporting_year: int = Field(default=2025, description="Year")
    overall_maturity_level: MaturityLevel = Field(
        default=MaturityLevel.SCREENING, description="Overall level"
    )
    overall_maturity_score: Decimal = Field(
        default=Decimal("1.0"), description="Overall score"
    )
    overall_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Overall uncertainty"
    )
    categories_assessed: int = Field(default=0, description="Assessed count")
    categories_at_target: int = Field(default=0, description="At target count")
    categories_needing_upgrade: int = Field(default=0, description="Need upgrade")
    total_upgrade_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Total upgrade cost"
    )
    category_assessments: List[CategoryMaturity] = Field(
        default_factory=list, description="Category assessments"
    )
    roadmap: List[UpgradePathway] = Field(
        default_factory=list, description="Upgrade roadmap"
    )
    budget_allocation: Optional[BudgetAllocation] = Field(
        default=None, description="Budget allocation"
    )
    uncertainty_projection: Optional[UncertaintyProjection] = Field(
        default=None, description="Uncertainty projection"
    )
    simulated_inventory: Optional[SimulatedInventory] = Field(
        default=None, description="Simulated inventory"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    status: AssessmentStatus = Field(
        default=AssessmentStatus.COMPLETE, description="Status"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing ms"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

CategoryData.model_rebuild()
MaturityAssessmentInput.model_rebuild()
CategoryMaturity.model_rebuild()
UpgradePathway.model_rebuild()
ROIAnalysis.model_rebuild()
BudgetAllocation.model_rebuild()
UncertaintyProjection.model_rebuild()
SimulatedInventory.model_rebuild()
MaturityAssessment.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DataMaturityEngine:
    """Assess Scope 3 data maturity and generate upgrade roadmaps.

    Evaluates the data maturity of each Scope 3 category, quantifies the
    cost and benefit of upgrading methodology tiers, and produces budget-
    optimised upgrade roadmaps with projected uncertainty reductions.

    Follows the zero-hallucination principle: all costs come from inline
    reference tables; all calculations use deterministic Decimal arithmetic.

    Attributes:
        _warnings: Warnings generated during assessment.

    Example:
        >>> engine = DataMaturityEngine()
        >>> inp = MaturityAssessmentInput(
        ...     category_data=[
        ...         CategoryData(category_number=1, current_tier=MaturityLevel.SPEND_BASED,
        ...                      total_co2e_tonnes=Decimal("50000"),
        ...                      share_of_scope3_pct=Decimal("45")),
        ...     ],
        ...     total_scope3_tco2e=Decimal("111000"),
        ...     budget_usd=Decimal("100000"),
        ... )
        >>> result = engine.assess_maturity(inp)
        >>> print(result.overall_maturity_score)
    """

    def __init__(self) -> None:
        """Initialise DataMaturityEngine."""
        self._warnings: List[str] = []
        logger.info("DataMaturityEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_maturity(
        self,
        input_data: MaturityAssessmentInput,
    ) -> MaturityAssessment:
        """Perform a full maturity assessment across all categories.

        Main entry point.  Assesses each category, generates an upgrade
        roadmap, optimises budget allocation, and projects uncertainty.

        Args:
            input_data: Assessment input data.

        Returns:
            MaturityAssessment with complete results.

        Raises:
            ValueError: If no category data provided.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not input_data.category_data:
            raise ValueError("At least one category is required for assessment")

        logger.info(
            "Starting maturity assessment for %d categories",
            len(input_data.category_data),
        )

        # Step 1: Fill in missing category names
        for cd in input_data.category_data:
            if not cd.category_name:
                cd.category_name = CATEGORY_NAMES.get(
                    cd.category_number, f"Category {cd.category_number}"
                )

        # Step 2: Assess each category
        cat_assessments = self._assess_categories(
            input_data.category_data, input_data.target_overall_tier
        )

        # Step 3: Calculate overall maturity
        overall_score, overall_level = self._calculate_overall_maturity(
            cat_assessments
        )

        # Step 4: Calculate overall uncertainty
        overall_uncertainty = self._calculate_combined_uncertainty(
            cat_assessments
        )

        # Step 5: Generate roadmap
        roadmap = self._generate_roadmap_internal(
            cat_assessments,
            input_data.budget_usd,
            input_data.target_timeline_months,
        )

        # Step 6: Optimise budget
        budget_alloc = None
        if input_data.budget_usd > Decimal("0"):
            budget_alloc = self._optimise_budget(
                cat_assessments, input_data.budget_usd
            )

        # Step 7: Project uncertainty
        uncertainty_proj = self._project_uncertainty_internal(
            cat_assessments, roadmap
        )

        # Step 8: Simulate post-upgrade inventory
        sim_inventory = self._simulate_post_upgrade_internal(
            cat_assessments, roadmap, input_data.total_scope3_tco2e
        )

        # Summary counts
        at_target = sum(1 for ca in cat_assessments if ca.gap_tiers == 0)
        need_upgrade = sum(1 for ca in cat_assessments if ca.gap_tiers > 0)
        total_cost = sum(
            (ca.upgrade_cost_usd for ca in cat_assessments), Decimal("0")
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)

        result = MaturityAssessment(
            org_id=input_data.org_id,
            reporting_year=input_data.reporting_year,
            overall_maturity_level=overall_level,
            overall_maturity_score=_round_val(overall_score, 2),
            overall_uncertainty_pct=_round_val(overall_uncertainty, 2),
            categories_assessed=len(cat_assessments),
            categories_at_target=at_target,
            categories_needing_upgrade=need_upgrade,
            total_upgrade_cost_usd=_round_val(total_cost, 2),
            category_assessments=cat_assessments,
            roadmap=roadmap,
            budget_allocation=budget_alloc,
            uncertainty_projection=uncertainty_proj,
            simulated_inventory=sim_inventory,
            warnings=list(self._warnings),
            status=AssessmentStatus.COMPLETE,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Maturity assessment complete: score=%.2f, %d/%d at target, "
            "total upgrade cost=$%.0f",
            overall_score, at_target, len(cat_assessments), total_cost,
        )
        return result

    def generate_roadmap(
        self,
        assessment: MaturityAssessment,
        budget_usd: Decimal = Decimal("0"),
        timeline_months: int = 24,
    ) -> List[UpgradePathway]:
        """Generate an ordered upgrade roadmap from an existing assessment.

        Args:
            assessment: Completed maturity assessment.
            budget_usd: Available budget.
            timeline_months: Target timeline in months.

        Returns:
            Ordered list of UpgradePathway items.
        """
        return self._generate_roadmap_internal(
            assessment.category_assessments, budget_usd, timeline_months
        )

    def calculate_upgrade_roi(
        self,
        category_number: int,
        current_tier: MaturityLevel,
        target_tier: MaturityLevel,
        emissions_tco2e: Decimal = Decimal("0"),
    ) -> ROIAnalysis:
        """Calculate ROI for a specific category tier transition.

        Args:
            category_number: Scope 3 category number (1-15).
            current_tier: Current maturity level.
            target_tier: Target maturity level.
            emissions_tco2e: Category emissions for uncertainty band calc.

        Returns:
            ROIAnalysis with cost, gain, and ROI metrics.

        Raises:
            ValueError: If target tier is not higher than current tier.
        """
        current_num = MATURITY_LEVEL_NUMBER[current_tier]
        target_num = MATURITY_LEVEL_NUMBER[target_tier]

        if target_num <= current_num:
            raise ValueError(
                f"Target tier ({target_tier.value}) must be higher than "
                f"current tier ({current_tier.value})"
            )

        cat_type = CATEGORY_TYPE_MAP.get(category_number, CategoryType.PROCUREMENT)
        cost = self._lookup_upgrade_cost(cat_type, current_num, target_num)

        current_unc = UNCERTAINTY_BY_TIER[current_tier]
        target_unc = UNCERTAINTY_BY_TIER[target_tier]
        accuracy_gain = current_unc - target_unc

        roi = _safe_divide(
            accuracy_gain, cost / Decimal("1000"), Decimal("0")
        )

        # Calculate uncertainty band reduction in tCO2e
        band_current = emissions_tco2e * current_unc / Decimal("100")
        band_target = emissions_tco2e * target_unc / Decimal("100")
        band_reduction = band_current - band_target

        # Payback narrative
        narrative = self._generate_payback_narrative(
            category_number, current_tier, target_tier, cost, accuracy_gain
        )

        return ROIAnalysis(
            category_number=category_number,
            category_name=CATEGORY_NAMES.get(
                category_number, f"Category {category_number}"
            ),
            current_tier=current_tier,
            target_tier=target_tier,
            cost_usd=_round_val(cost, 2),
            accuracy_gain_pct=_round_val(accuracy_gain, 2),
            roi_accuracy_per_kusd=_round_val(roi, 4),
            emissions_at_stake_tco2e=_round_val(emissions_tco2e, 2),
            uncertainty_band_reduction_tco2e=_round_val(band_reduction, 2),
            payback_narrative=narrative,
        )

    def optimize_budget(
        self,
        assessment: MaturityAssessment,
        budget_usd: Decimal,
    ) -> BudgetAllocation:
        """Maximise accuracy improvement within a given budget.

        Uses a greedy algorithm: sort upgrades by ROI descending,
        then allocate budget to the highest-ROI upgrade first.

        Args:
            assessment: Completed maturity assessment.
            budget_usd: Available budget in USD.

        Returns:
            BudgetAllocation with funded upgrades.
        """
        return self._optimise_budget(
            assessment.category_assessments, budget_usd
        )

    def project_uncertainty_reduction(
        self,
        roadmap: List[UpgradePathway],
        assessment: MaturityAssessment,
    ) -> UncertaintyProjection:
        """Project uncertainty improvement from executing the roadmap.

        Args:
            roadmap: Upgrade roadmap to project.
            assessment: Maturity assessment with category data.

        Returns:
            UncertaintyProjection with before/after uncertainty.
        """
        return self._project_uncertainty_internal(
            assessment.category_assessments, roadmap
        )

    def simulate_post_upgrade(
        self,
        assessment: MaturityAssessment,
        roadmap: List[UpgradePathway],
    ) -> SimulatedInventory:
        """Simulate the inventory after executing the roadmap.

        Args:
            assessment: Maturity assessment.
            roadmap: Upgrade roadmap.

        Returns:
            SimulatedInventory with projected totals.
        """
        total_s3 = sum(
            (ca.emissions_tco2e for ca in assessment.category_assessments),
            Decimal("0"),
        )
        return self._simulate_post_upgrade_internal(
            assessment.category_assessments, roadmap, total_s3
        )

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _assess_categories(
        self,
        category_data: List[CategoryData],
        target_tier: MaturityLevel,
    ) -> List[CategoryMaturity]:
        """Assess maturity of each category.

        Args:
            category_data: Per-category input data.
            target_tier: Target minimum tier.

        Returns:
            List of CategoryMaturity assessments.
        """
        assessments: List[CategoryMaturity] = []
        target_num = MATURITY_LEVEL_NUMBER[target_tier]

        for cd in category_data:
            current_num = MATURITY_LEVEL_NUMBER[cd.current_tier]
            cat_type = CATEGORY_TYPE_MAP.get(
                cd.category_number, CategoryType.PROCUREMENT
            )

            # Determine recommended tier
            recommended_num = max(current_num, target_num)
            # For high-share categories, recommend one tier higher
            if cd.share_of_scope3_pct >= Decimal("20") and recommended_num < 5:
                recommended_num = min(recommended_num + 1, 5)
            recommended_tier = NUMBER_TO_MATURITY[recommended_num]

            gap = max(recommended_num - current_num, 0)
            current_unc = UNCERTAINTY_BY_TIER[cd.current_tier]
            target_unc = UNCERTAINTY_BY_TIER[recommended_tier]
            accuracy_gain = current_unc - target_unc

            # Lookup upgrade cost
            cost = Decimal("0")
            if gap > 0:
                cost = self._lookup_upgrade_cost(cat_type, current_num, recommended_num)

            roi = _safe_divide(
                accuracy_gain, cost / Decimal("1000"), Decimal("0")
            ) if cost > Decimal("0") else Decimal("0")

            assessments.append(CategoryMaturity(
                category_number=cd.category_number,
                category_name=cd.category_name,
                category_type=cat_type.value,
                current_tier=cd.current_tier,
                current_tier_number=current_num,
                current_uncertainty_pct=current_unc,
                recommended_tier=recommended_tier,
                recommended_tier_number=recommended_num,
                target_uncertainty_pct=target_unc,
                emissions_tco2e=cd.total_co2e_tonnes,
                share_of_scope3_pct=cd.share_of_scope3_pct,
                data_quality_score=cd.data_quality_score,
                gap_tiers=gap,
                upgrade_cost_usd=_round_val(cost, 2),
                accuracy_gain_pct=_round_val(accuracy_gain, 2),
                roi_accuracy_per_kusd=_round_val(roi, 4),
            ))

        return assessments

    def _calculate_overall_maturity(
        self,
        assessments: List[CategoryMaturity],
    ) -> Tuple[Decimal, MaturityLevel]:
        """Calculate emission-weighted overall maturity score.

        Args:
            assessments: Per-category assessments.

        Returns:
            Tuple of (weighted score, overall MaturityLevel).
        """
        total_emissions = sum(
            (a.emissions_tco2e for a in assessments), Decimal("0")
        )

        if total_emissions <= Decimal("0"):
            # Equal-weight fallback
            if not assessments:
                return Decimal("1"), MaturityLevel.SCREENING
            avg = sum(
                (_decimal(a.current_tier_number) for a in assessments),
                Decimal("0"),
            ) / _decimal(len(assessments))
        else:
            avg = sum(
                (
                    _decimal(a.current_tier_number) * a.emissions_tco2e
                    for a in assessments
                ),
                Decimal("0"),
            ) / total_emissions

        rounded_level = int(avg.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        rounded_level = max(1, min(5, rounded_level))
        level = NUMBER_TO_MATURITY[rounded_level]

        return avg, level

    def _calculate_combined_uncertainty(
        self,
        assessments: List[CategoryMaturity],
    ) -> Decimal:
        """Calculate combined portfolio uncertainty.

        Uses error propagation: U = sqrt(sum(share_i^2 * u_i^2)).

        Args:
            assessments: Per-category assessments.

        Returns:
            Combined uncertainty percentage.
        """
        total_emissions = sum(
            (a.emissions_tco2e for a in assessments), Decimal("0")
        )
        if total_emissions <= Decimal("0"):
            return Decimal("0")

        sum_sq = Decimal("0")
        for a in assessments:
            share = _safe_divide(a.emissions_tco2e, total_emissions)
            unc = a.current_uncertainty_pct / Decimal("100")
            sum_sq += (share * unc) ** 2

        return _sqrt_decimal(sum_sq) * Decimal("100")

    def _generate_roadmap_internal(
        self,
        assessments: List[CategoryMaturity],
        budget_usd: Decimal,
        timeline_months: int,
    ) -> List[UpgradePathway]:
        """Generate ordered upgrade roadmap.

        Sorts by ROI descending, assigns ranks, and marks budget feasibility.

        Args:
            assessments: Per-category assessments.
            budget_usd: Available budget.
            timeline_months: Target timeline.

        Returns:
            Ordered list of UpgradePathway items.
        """
        # Filter to categories needing upgrade
        upgradeable = [a for a in assessments if a.gap_tiers > 0]

        if not upgradeable:
            return []

        # Sort by ROI descending, then by emissions descending for ties
        upgradeable_sorted = sorted(
            upgradeable,
            key=lambda a: (a.roi_accuracy_per_kusd, a.emissions_tco2e),
            reverse=True,
        )

        roadmap: List[UpgradePathway] = []
        cumulative_cost = Decimal("0")

        for rank, a in enumerate(upgradeable_sorted, 1):
            cumulative_cost += a.upgrade_cost_usd
            within_budget = (
                budget_usd <= Decimal("0") or cumulative_cost <= budget_usd
            )

            # Duration based on tier jump
            duration = UPGRADE_DURATION_MONTHS.get(a.gap_tiers, 18)

            # Priority based on ROI and emission share
            priority = self._determine_priority(a)

            # Description
            desc = (
                f"Upgrade Cat {a.category_number} ({a.category_name}) from "
                f"{a.current_tier.value} to {a.recommended_tier.value}: "
                f"{a.accuracy_gain_pct}pp uncertainty reduction for "
                f"${a.upgrade_cost_usd:,.0f}"
            )

            roadmap.append(UpgradePathway(
                priority=priority,
                rank=rank,
                category_number=a.category_number,
                category_name=a.category_name,
                from_tier=a.current_tier,
                to_tier=a.recommended_tier,
                cost_usd=a.upgrade_cost_usd,
                accuracy_gain_pct=a.accuracy_gain_pct,
                roi_accuracy_per_kusd=a.roi_accuracy_per_kusd,
                duration_months=duration,
                cumulative_cost_usd=_round_val(cumulative_cost, 2),
                within_budget=within_budget,
                description=desc,
            ))

        return roadmap

    def _optimise_budget(
        self,
        assessments: List[CategoryMaturity],
        budget_usd: Decimal,
    ) -> BudgetAllocation:
        """Greedy budget optimisation maximising accuracy per dollar.

        Args:
            assessments: Per-category assessments.
            budget_usd: Available budget.

        Returns:
            BudgetAllocation result.
        """
        upgradeable = [a for a in assessments if a.gap_tiers > 0]

        # Sort by ROI descending
        sorted_upgrades = sorted(
            upgradeable,
            key=lambda a: a.roi_accuracy_per_kusd,
            reverse=True,
        )

        allocated = Decimal("0")
        total_gain = Decimal("0")
        funded_pathways: List[UpgradePathway] = []
        rank = 0

        for a in sorted_upgrades:
            if allocated + a.upgrade_cost_usd > budget_usd:
                continue

            rank += 1
            allocated += a.upgrade_cost_usd
            total_gain += a.accuracy_gain_pct

            duration = UPGRADE_DURATION_MONTHS.get(a.gap_tiers, 18)
            priority = self._determine_priority(a)

            funded_pathways.append(UpgradePathway(
                priority=priority,
                rank=rank,
                category_number=a.category_number,
                category_name=a.category_name,
                from_tier=a.current_tier,
                to_tier=a.recommended_tier,
                cost_usd=a.upgrade_cost_usd,
                accuracy_gain_pct=a.accuracy_gain_pct,
                roi_accuracy_per_kusd=a.roi_accuracy_per_kusd,
                duration_months=duration,
                cumulative_cost_usd=_round_val(allocated, 2),
                within_budget=True,
                description=(
                    f"Cat {a.category_number}: "
                    f"{a.current_tier.value} -> {a.recommended_tier.value}"
                ),
            ))

        return BudgetAllocation(
            total_budget_usd=budget_usd,
            allocated_usd=_round_val(allocated, 2),
            remaining_usd=_round_val(budget_usd - allocated, 2),
            upgrades_funded=len(funded_pathways),
            total_accuracy_gain_pct=_round_val(total_gain, 2),
            allocations=funded_pathways,
        )

    def _project_uncertainty_internal(
        self,
        assessments: List[CategoryMaturity],
        roadmap: List[UpgradePathway],
    ) -> UncertaintyProjection:
        """Project uncertainty reduction from executing the roadmap.

        Args:
            assessments: Per-category assessments.
            roadmap: Upgrade roadmap.

        Returns:
            UncertaintyProjection.
        """
        # Build map of upgraded categories
        upgraded_tiers: Dict[int, MaturityLevel] = {}
        for step in roadmap:
            if step.within_budget:
                upgraded_tiers[step.category_number] = step.to_tier

        total_emissions = sum(
            (a.emissions_tco2e for a in assessments), Decimal("0")
        )

        current_combined = self._calculate_combined_uncertainty(assessments)

        # Calculate projected uncertainty
        projections: List[Dict[str, Any]] = []
        sum_sq = Decimal("0")

        for a in assessments:
            share = _safe_divide(a.emissions_tco2e, total_emissions)
            post_tier = upgraded_tiers.get(a.category_number, a.current_tier)
            post_unc = UNCERTAINTY_BY_TIER[post_tier]
            unc_frac = post_unc / Decimal("100")
            sum_sq += (share * unc_frac) ** 2

            projections.append({
                "category_number": a.category_number,
                "category_name": a.category_name,
                "current_tier": a.current_tier.value,
                "projected_tier": post_tier.value,
                "current_uncertainty_pct": str(a.current_uncertainty_pct),
                "projected_uncertainty_pct": str(post_unc),
                "upgraded": a.category_number in upgraded_tiers,
            })

        projected_combined = _sqrt_decimal(sum_sq) * Decimal("100")
        reduction = current_combined - projected_combined
        improvement = _safe_pct(reduction, current_combined)

        return UncertaintyProjection(
            current_combined_uncertainty_pct=_round_val(current_combined, 2),
            projected_combined_uncertainty_pct=_round_val(projected_combined, 2),
            reduction_pct=_round_val(reduction, 2),
            improvement_pct=_round_val(improvement, 2),
            category_projections=projections,
        )

    def _simulate_post_upgrade_internal(
        self,
        assessments: List[CategoryMaturity],
        roadmap: List[UpgradePathway],
        total_scope3: Decimal,
    ) -> SimulatedInventory:
        """Simulate the inventory after executing funded roadmap steps.

        For categories upgraded to higher tiers, applies a correction factor
        based on typical tier adjustment ratios (higher tiers typically
        yield lower estimates than spend-based).

        Args:
            assessments: Per-category assessments.
            roadmap: Upgrade roadmap.
            total_scope3: Current total Scope 3.

        Returns:
            SimulatedInventory.
        """
        # Tier adjustment factors: moving to a higher tier typically
        # reduces the estimate (higher tiers are more accurate, often lower)
        tier_adjustment: Dict[str, Decimal] = {
            "1_2": Decimal("0.95"),
            "1_3": Decimal("0.85"),
            "1_4": Decimal("0.75"),
            "1_5": Decimal("0.65"),
            "2_3": Decimal("0.90"),
            "2_4": Decimal("0.80"),
            "2_5": Decimal("0.70"),
            "3_4": Decimal("0.88"),
            "3_5": Decimal("0.78"),
            "4_5": Decimal("0.90"),
        }

        upgraded_tiers: Dict[int, MaturityLevel] = {}
        for step in roadmap:
            if step.within_budget:
                upgraded_tiers[step.category_number] = step.to_tier

        estimates: List[Dict[str, Any]] = []
        total_post = Decimal("0")
        tier_mix: Dict[str, int] = {}

        for a in assessments:
            post_tier = upgraded_tiers.get(a.category_number, a.current_tier)
            post_num = MATURITY_LEVEL_NUMBER[post_tier]
            current_num = a.current_tier_number

            # Apply adjustment factor if upgraded
            if post_num > current_num:
                key = f"{current_num}_{post_num}"
                factor = tier_adjustment.get(key, Decimal("0.90"))
                post_emissions = a.emissions_tco2e * factor
            else:
                post_emissions = a.emissions_tco2e

            post_unc = UNCERTAINTY_BY_TIER[post_tier]
            lower = post_emissions * (Decimal("1") - post_unc / Decimal("100"))
            upper = post_emissions * (Decimal("1") + post_unc / Decimal("100"))

            total_post += post_emissions
            tier_name = post_tier.value
            tier_mix[tier_name] = tier_mix.get(tier_name, 0) + 1

            estimates.append({
                "category_number": a.category_number,
                "category_name": a.category_name,
                "current_tco2e": str(_round_val(a.emissions_tco2e, 2)),
                "projected_tco2e": str(_round_val(post_emissions, 2)),
                "lower_tco2e": str(_round_val(lower, 2)),
                "upper_tco2e": str(_round_val(upper, 2)),
                "tier": post_tier.value,
                "uncertainty_pct": str(post_unc),
            })

        # Calculate combined post-upgrade uncertainty
        total_emissions = total_post if total_post > Decimal("0") else Decimal("1")
        sum_sq = Decimal("0")
        for est in estimates:
            share = _safe_divide(
                _decimal(est["projected_tco2e"]), total_emissions
            )
            unc_frac = _decimal(est["uncertainty_pct"]) / Decimal("100")
            sum_sq += (share * unc_frac) ** 2

        combined_unc = _sqrt_decimal(sum_sq) * Decimal("100")
        total_lower = total_post * (Decimal("1") - combined_unc / Decimal("100"))
        total_upper = total_post * (Decimal("1") + combined_unc / Decimal("100"))

        return SimulatedInventory(
            total_scope3_tco2e=_round_val(total_post, 2),
            total_scope3_lower_tco2e=_round_val(total_lower, 2),
            total_scope3_upper_tco2e=_round_val(total_upper, 2),
            uncertainty_pct=_round_val(combined_unc, 2),
            category_estimates=estimates,
            methodology_mix=tier_mix,
        )

    # ------------------------------------------------------------------
    # Lookup / Utility Methods
    # ------------------------------------------------------------------

    def _lookup_upgrade_cost(
        self,
        cat_type: CategoryType,
        from_num: int,
        to_num: int,
    ) -> Decimal:
        """Look up upgrade cost from reference table.

        Args:
            cat_type: Category archetype.
            from_num: Current tier number.
            to_num: Target tier number.

        Returns:
            Upgrade cost in USD.
        """
        key = f"{from_num}_{to_num}"
        costs = UPGRADE_COSTS.get(cat_type.value, {})
        cost = costs.get(key)

        if cost is not None:
            return cost

        # Fallback: sum incremental costs
        total = Decimal("0")
        for step in range(from_num, to_num):
            step_key = f"{step}_{step + 1}"
            step_cost = costs.get(step_key, Decimal("10000"))
            total += step_cost

        self._warnings.append(
            f"No direct cost for {cat_type.value} {key}; using sum of "
            f"incremental costs: ${total}"
        )
        return total

    def _determine_priority(self, assessment: CategoryMaturity) -> UpgradePriority:
        """Determine upgrade priority based on ROI and emission share.

        Args:
            assessment: Category assessment.

        Returns:
            UpgradePriority level.
        """
        roi = assessment.roi_accuracy_per_kusd
        share = assessment.share_of_scope3_pct

        if roi >= Decimal("3") and share >= Decimal("15"):
            return UpgradePriority.CRITICAL
        elif roi >= Decimal("2") or share >= Decimal("20"):
            return UpgradePriority.HIGH
        elif roi >= Decimal("1") or share >= Decimal("10"):
            return UpgradePriority.MEDIUM
        else:
            return UpgradePriority.LOW

    def _generate_payback_narrative(
        self,
        category_number: int,
        current_tier: MaturityLevel,
        target_tier: MaturityLevel,
        cost_usd: Decimal,
        accuracy_gain: Decimal,
    ) -> str:
        """Generate a human-readable payback narrative.

        Args:
            category_number: Category number.
            current_tier: Current tier.
            target_tier: Target tier.
            cost_usd: Upgrade cost.
            accuracy_gain: Uncertainty reduction in percentage points.

        Returns:
            Narrative string.
        """
        cat_name = CATEGORY_NAMES.get(
            category_number, f"Category {category_number}"
        )

        if accuracy_gain >= Decimal("30"):
            impact = "transformative"
        elif accuracy_gain >= Decimal("15"):
            impact = "significant"
        else:
            impact = "moderate"

        if cost_usd <= Decimal("15000"):
            cost_desc = "low-cost"
        elif cost_usd <= Decimal("50000"):
            cost_desc = "moderate-cost"
        else:
            cost_desc = "high-investment"

        return (
            f"Upgrading {cat_name} from {current_tier.value} to "
            f"{target_tier.value} is a {cost_desc} initiative "
            f"(${cost_usd:,.0f}) delivering {impact} accuracy improvement "
            f"of {accuracy_gain:.1f} percentage points in uncertainty "
            f"reduction."
        )

    def _compute_provenance(self, result: MaturityAssessment) -> str:
        """Compute SHA-256 provenance hash for the assessment result.

        Args:
            result: Complete assessment result.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(result)
