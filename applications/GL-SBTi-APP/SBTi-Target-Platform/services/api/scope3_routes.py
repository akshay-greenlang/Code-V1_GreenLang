"""
GL-SBTi-APP Scope 3 Screening API

Implements Scope 3 screening and assessment per SBTi criteria C13-C15.
Evaluates the 40% trigger threshold, provides 15-category breakdowns,
identifies emission hotspot categories, calculates target coverage
percentages, recommends target categories, and assesses data quality
by category.

SBTi Scope 3 Rules:
    - If Scope 3 >= 40% of total S1+S2+S3 -> Scope 3 target required (C13)
    - Scope 3 target must cover >= 67% of Scope 3 emissions (C14)
    - All 15 categories must be screened (C15)
    - Supplier engagement approach available for Cat 1 and Cat 4

GHG Protocol Scope 3 Categories (1-15):
    1. Purchased Goods & Services
    2. Capital Goods
    3. Fuel & Energy Activities
    4. Upstream Transportation
    5. Waste Generated in Operations
    6. Business Travel
    7. Employee Commuting
    8. Upstream Leased Assets
    9. Downstream Transportation
    10. Processing of Sold Products
    11. Use of Sold Products
    12. End-of-Life Treatment
    13. Downstream Leased Assets
    14. Franchises
    15. Investments
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/scope3", tags=["Scope 3 Screening"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class TriggerAssessmentRequest(BaseModel):
    """Request to assess the 40% Scope 3 trigger."""
    org_id: str = Field(..., description="Organization ID")
    scope1_tco2e: float = Field(..., ge=0, description="Scope 1 emissions")
    scope2_tco2e: float = Field(..., ge=0, description="Scope 2 emissions")
    scope3_tco2e: float = Field(..., ge=0, description="Total Scope 3 emissions")
    scope3_category_data: Optional[Dict[str, float]] = Field(
        None, description="Emissions by category (cat_1 through cat_15)",
    )


class HotspotRequest(BaseModel):
    """Request to identify Scope 3 hotspot categories."""
    org_id: str = Field(..., description="Organization ID")
    category_emissions: Dict[str, float] = Field(
        ..., description="Emissions by category name",
    )
    threshold_pct: float = Field(
        5.0, ge=0, le=100,
        description="Minimum percentage to qualify as hotspot",
    )


class CoverageCalculatorRequest(BaseModel):
    """Request to calculate target coverage percentage."""
    org_id: str = Field(..., description="Organization ID")
    total_scope3_tco2e: float = Field(..., gt=0)
    categories_in_target: List[str] = Field(
        ..., description="Category names included in target",
    )
    category_emissions: Dict[str, float] = Field(
        ..., description="Emissions by category name",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TriggerAssessmentResponse(BaseModel):
    """Scope 3 trigger assessment result."""
    org_id: str
    scope1_tco2e: float
    scope2_tco2e: float
    scope3_tco2e: float
    total_emissions_tco2e: float
    scope3_pct_of_total: float
    trigger_threshold_pct: float
    scope3_target_required: bool
    minimum_target_coverage_pct: float
    category_breakdown: Optional[Dict[str, float]]
    recommendation: str
    generated_at: datetime


class CategoryBreakdownResponse(BaseModel):
    """15-category Scope 3 breakdown."""
    org_id: str
    categories: List[Dict[str, Any]]
    total_scope3_tco2e: float
    top_3_categories: List[str]
    categories_above_5pct: int
    data_completeness_pct: float
    generated_at: datetime


class HotspotResponse(BaseModel):
    """Scope 3 hotspot analysis result."""
    org_id: str
    hotspot_categories: List[Dict[str, Any]]
    total_hotspot_emissions_tco2e: float
    hotspot_pct_of_scope3: float
    threshold_pct: float
    recommendation: str
    generated_at: datetime


class CoverageResponse(BaseModel):
    """Target coverage calculation result."""
    org_id: str
    total_scope3_tco2e: float
    covered_emissions_tco2e: float
    coverage_pct: float
    minimum_required_pct: float
    meets_minimum: bool
    categories_included: List[str]
    categories_excluded: List[str]
    gap_tco2e: float
    recommendation: str
    generated_at: datetime


class CategoryRecommendationResponse(BaseModel):
    """Recommended Scope 3 target categories."""
    org_id: str
    recommended_categories: List[Dict[str, Any]]
    total_recommended_coverage_pct: float
    minimum_required_pct: float
    strategy: str
    generated_at: datetime


class CategoryDefinition(BaseModel):
    """Scope 3 category definition."""
    category_number: int
    name: str
    description: str
    typical_share_pct: str
    data_sources: List[str]
    sbti_methods: List[str]


class DataQualityResponse(BaseModel):
    """Data quality assessment by Scope 3 category."""
    org_id: str
    categories: List[Dict[str, Any]]
    overall_quality_score: float
    quality_level: str
    improvement_priorities: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

SCOPE3_CATEGORIES = [
    {"category_number": 1, "name": "Purchased Goods & Services",
     "description": "Cradle-to-gate emissions of purchased goods and services.",
     "typical_share_pct": "30-60%", "data_sources": ["supplier data", "spend data", "EEIO"],
     "sbti_methods": ["absolute", "intensity", "supplier_engagement"]},
    {"category_number": 2, "name": "Capital Goods",
     "description": "Cradle-to-gate emissions of purchased capital equipment.",
     "typical_share_pct": "5-15%", "data_sources": ["supplier data", "spend data"],
     "sbti_methods": ["absolute", "intensity"]},
    {"category_number": 3, "name": "Fuel & Energy Activities",
     "description": "Upstream emissions from purchased fuels and electricity (not in S1/S2).",
     "typical_share_pct": "3-8%", "data_sources": ["utility bills", "fuel records"],
     "sbti_methods": ["absolute"]},
    {"category_number": 4, "name": "Upstream Transportation",
     "description": "Transportation of purchased goods from suppliers.",
     "typical_share_pct": "3-10%", "data_sources": ["logistics data", "spend data"],
     "sbti_methods": ["absolute", "intensity", "supplier_engagement"]},
    {"category_number": 5, "name": "Waste Generated in Operations",
     "description": "Disposal and treatment of waste generated in operations.",
     "typical_share_pct": "1-3%", "data_sources": ["waste records", "hauler data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 6, "name": "Business Travel",
     "description": "Transportation for business-related activities.",
     "typical_share_pct": "1-5%", "data_sources": ["travel records", "expense data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 7, "name": "Employee Commuting",
     "description": "Transportation of employees between home and work.",
     "typical_share_pct": "1-5%", "data_sources": ["surveys", "HR data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 8, "name": "Upstream Leased Assets",
     "description": "Emissions from operation of leased assets (lessee).",
     "typical_share_pct": "0-5%", "data_sources": ["lease data", "energy data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 9, "name": "Downstream Transportation",
     "description": "Transportation of sold products to end customers.",
     "typical_share_pct": "2-8%", "data_sources": ["logistics data", "distribution records"],
     "sbti_methods": ["absolute", "intensity"]},
    {"category_number": 10, "name": "Processing of Sold Products",
     "description": "Processing of intermediate products sold to other companies.",
     "typical_share_pct": "0-10%", "data_sources": ["product specs", "industry data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 11, "name": "Use of Sold Products",
     "description": "End-use emissions from sold products (direct and indirect).",
     "typical_share_pct": "5-40%", "data_sources": ["product specs", "usage patterns"],
     "sbti_methods": ["absolute", "intensity"]},
    {"category_number": 12, "name": "End-of-Life Treatment",
     "description": "Disposal of sold products at end of life.",
     "typical_share_pct": "1-5%", "data_sources": ["product composition", "waste data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 13, "name": "Downstream Leased Assets",
     "description": "Emissions from operation of assets leased to others.",
     "typical_share_pct": "0-5%", "data_sources": ["lease data", "tenant data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 14, "name": "Franchises",
     "description": "Emissions from operation of franchises.",
     "typical_share_pct": "0-10%", "data_sources": ["franchise data", "energy data"],
     "sbti_methods": ["absolute"]},
    {"category_number": 15, "name": "Investments",
     "description": "Emissions from equity and debt investments.",
     "typical_share_pct": "0-30%", "data_sources": ["portfolio data", "PCAF"],
     "sbti_methods": ["absolute", "intensity"]},
]


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/trigger-assessment",
    response_model=TriggerAssessmentResponse,
    summary="Assess 40% trigger",
    description=(
        "Assess whether Scope 3 emissions meet the SBTi 40% trigger threshold "
        "(criterion C13). If Scope 3 >= 40% of total S1+S2+S3, a Scope 3 "
        "target covering at least 67% of Scope 3 is required."
    ),
)
async def assess_scope3_trigger(
    request: TriggerAssessmentRequest,
) -> TriggerAssessmentResponse:
    """Assess Scope 3 trigger threshold."""
    total = request.scope1_tco2e + request.scope2_tco2e + request.scope3_tco2e
    pct = round((request.scope3_tco2e / total) * 100, 1) if total > 0 else 0.0
    required = pct >= 40.0

    if required:
        recommendation = (
            f"Scope 3 is {pct}% of total emissions (>= 40% threshold). "
            f"A Scope 3 target covering at least 67% of Scope 3 emissions is required. "
            f"Focus on the largest categories first."
        )
    else:
        recommendation = (
            f"Scope 3 is {pct}% of total emissions (< 40% threshold). "
            f"A Scope 3 target is encouraged but not mandatory. Consider voluntary "
            f"Scope 3 targets for credibility and ambition."
        )

    return TriggerAssessmentResponse(
        org_id=request.org_id,
        scope1_tco2e=request.scope1_tco2e,
        scope2_tco2e=request.scope2_tco2e,
        scope3_tco2e=request.scope3_tco2e,
        total_emissions_tco2e=round(total, 1),
        scope3_pct_of_total=pct,
        trigger_threshold_pct=40.0,
        scope3_target_required=required,
        minimum_target_coverage_pct=67.0 if required else 0.0,
        category_breakdown=request.scope3_category_data,
        recommendation=recommendation,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/category-breakdown",
    response_model=CategoryBreakdownResponse,
    summary="15-category breakdown",
    description=(
        "Retrieve Scope 3 emissions breakdown across all 15 GHG Protocol "
        "categories for an organization. Returns per-category emissions, "
        "percentage share, and data quality indicators."
    ),
)
async def get_category_breakdown(org_id: str) -> CategoryBreakdownResponse:
    """Get 15-category Scope 3 breakdown."""
    # Simulated category data
    category_data = [
        {"category": 1, "name": "Purchased Goods & Services", "emissions_tco2e": 48000, "pct_of_scope3": 40.0, "data_quality": "medium"},
        {"category": 2, "name": "Capital Goods", "emissions_tco2e": 12000, "pct_of_scope3": 10.0, "data_quality": "low"},
        {"category": 3, "name": "Fuel & Energy Activities", "emissions_tco2e": 6000, "pct_of_scope3": 5.0, "data_quality": "high"},
        {"category": 4, "name": "Upstream Transportation", "emissions_tco2e": 9600, "pct_of_scope3": 8.0, "data_quality": "medium"},
        {"category": 5, "name": "Waste Generated", "emissions_tco2e": 2400, "pct_of_scope3": 2.0, "data_quality": "high"},
        {"category": 6, "name": "Business Travel", "emissions_tco2e": 3600, "pct_of_scope3": 3.0, "data_quality": "high"},
        {"category": 7, "name": "Employee Commuting", "emissions_tco2e": 3600, "pct_of_scope3": 3.0, "data_quality": "low"},
        {"category": 8, "name": "Upstream Leased Assets", "emissions_tco2e": 1200, "pct_of_scope3": 1.0, "data_quality": "medium"},
        {"category": 9, "name": "Downstream Transportation", "emissions_tco2e": 6000, "pct_of_scope3": 5.0, "data_quality": "medium"},
        {"category": 10, "name": "Processing of Sold Products", "emissions_tco2e": 2400, "pct_of_scope3": 2.0, "data_quality": "low"},
        {"category": 11, "name": "Use of Sold Products", "emissions_tco2e": 15600, "pct_of_scope3": 13.0, "data_quality": "low"},
        {"category": 12, "name": "End-of-Life Treatment", "emissions_tco2e": 3600, "pct_of_scope3": 3.0, "data_quality": "low"},
        {"category": 13, "name": "Downstream Leased Assets", "emissions_tco2e": 1200, "pct_of_scope3": 1.0, "data_quality": "medium"},
        {"category": 14, "name": "Franchises", "emissions_tco2e": 2400, "pct_of_scope3": 2.0, "data_quality": "low"},
        {"category": 15, "name": "Investments", "emissions_tco2e": 2400, "pct_of_scope3": 2.0, "data_quality": "low"},
    ]
    total = sum(c["emissions_tco2e"] for c in category_data)
    sorted_cats = sorted(category_data, key=lambda c: c["emissions_tco2e"], reverse=True)
    top3 = [f"Cat {c['category']}: {c['name']}" for c in sorted_cats[:3]]
    above_5 = sum(1 for c in category_data if c["pct_of_scope3"] >= 5.0)

    high_quality = sum(1 for c in category_data if c["data_quality"] == "high")
    completeness = round(high_quality / len(category_data) * 100, 1)

    return CategoryBreakdownResponse(
        org_id=org_id,
        categories=category_data,
        total_scope3_tco2e=total,
        top_3_categories=top3,
        categories_above_5pct=above_5,
        data_completeness_pct=completeness,
        generated_at=_now(),
    )


@router.post(
    "/hotspot-analysis",
    response_model=HotspotResponse,
    summary="Identify hotspot categories",
    description=(
        "Identify Scope 3 hotspot categories that exceed the specified "
        "threshold percentage. Hotspots are priority categories for "
        "target-setting and data quality improvement."
    ),
)
async def hotspot_analysis(request: HotspotRequest) -> HotspotResponse:
    """Identify Scope 3 hotspot categories."""
    total = sum(request.category_emissions.values())
    if total <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Total category emissions must be greater than zero.",
        )

    hotspots = []
    for cat_name, emissions in request.category_emissions.items():
        pct = round((emissions / total) * 100, 1)
        if pct >= request.threshold_pct:
            hotspots.append({
                "category": cat_name,
                "emissions_tco2e": emissions,
                "pct_of_scope3": pct,
            })

    hotspots.sort(key=lambda h: h["emissions_tco2e"], reverse=True)
    hotspot_total = sum(h["emissions_tco2e"] for h in hotspots)
    hotspot_pct = round((hotspot_total / total) * 100, 1)

    return HotspotResponse(
        org_id=request.org_id,
        hotspot_categories=hotspots,
        total_hotspot_emissions_tco2e=hotspot_total,
        hotspot_pct_of_scope3=hotspot_pct,
        threshold_pct=request.threshold_pct,
        recommendation=(
            f"Found {len(hotspots)} hotspot categories representing {hotspot_pct}% "
            f"of Scope 3 emissions. Prioritize these categories for target-setting "
            f"and data quality improvement."
        ),
        generated_at=_now(),
    )


@router.post(
    "/coverage-calculator",
    response_model=CoverageResponse,
    summary="Calculate coverage percentage",
    description=(
        "Calculate Scope 3 target coverage percentage based on categories "
        "included in the target. Validates against the SBTi 67% minimum "
        "coverage requirement (criterion C14)."
    ),
)
async def calculate_coverage(request: CoverageCalculatorRequest) -> CoverageResponse:
    """Calculate Scope 3 target coverage."""
    covered = sum(
        em for cat, em in request.category_emissions.items()
        if cat in request.categories_in_target
    )
    coverage_pct = round((covered / request.total_scope3_tco2e) * 100, 1) if request.total_scope3_tco2e > 0 else 0.0
    minimum = 67.0
    meets = coverage_pct >= minimum
    excluded = [c for c in request.category_emissions if c not in request.categories_in_target]
    gap = max(round(request.total_scope3_tco2e * minimum / 100 - covered, 1), 0)

    if meets:
        recommendation = (
            f"Coverage of {coverage_pct}% meets the SBTi 67% minimum. "
            f"Consider including additional categories for stronger ambition."
        )
    else:
        recommendation = (
            f"Coverage of {coverage_pct}% is below the SBTi 67% minimum. "
            f"Include additional categories to close the {gap:,.0f} tCO2e gap."
        )

    return CoverageResponse(
        org_id=request.org_id,
        total_scope3_tco2e=request.total_scope3_tco2e,
        covered_emissions_tco2e=round(covered, 1),
        coverage_pct=coverage_pct,
        minimum_required_pct=minimum,
        meets_minimum=meets,
        categories_included=request.categories_in_target,
        categories_excluded=excluded,
        gap_tco2e=gap,
        recommendation=recommendation,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/recommendations",
    response_model=CategoryRecommendationResponse,
    summary="Recommended target categories",
    description=(
        "Recommend which Scope 3 categories to include in the target to "
        "achieve the 67% minimum coverage with the fewest categories. "
        "Prioritizes by emission size and data availability."
    ),
)
async def get_recommendations(org_id: str) -> CategoryRecommendationResponse:
    """Get recommended Scope 3 target categories."""
    recommended = [
        {"category": 1, "name": "Purchased Goods & Services", "emissions_tco2e": 48000, "pct_of_scope3": 40.0, "priority": "critical", "method": "supplier_engagement"},
        {"category": 11, "name": "Use of Sold Products", "emissions_tco2e": 15600, "pct_of_scope3": 13.0, "priority": "high", "method": "absolute"},
        {"category": 2, "name": "Capital Goods", "emissions_tco2e": 12000, "pct_of_scope3": 10.0, "priority": "high", "method": "absolute"},
        {"category": 4, "name": "Upstream Transportation", "emissions_tco2e": 9600, "pct_of_scope3": 8.0, "priority": "medium", "method": "supplier_engagement"},
    ]

    total_pct = sum(r["pct_of_scope3"] for r in recommended)

    return CategoryRecommendationResponse(
        org_id=org_id,
        recommended_categories=recommended,
        total_recommended_coverage_pct=total_pct,
        minimum_required_pct=67.0,
        strategy=(
            f"Including {len(recommended)} categories achieves {total_pct}% coverage, "
            f"exceeding the 67% minimum. Use supplier engagement for Cat 1 and Cat 4 "
            f"and absolute reduction for the remaining categories."
        ),
        generated_at=_now(),
    )


@router.get(
    "/categories",
    response_model=List[CategoryDefinition],
    summary="List all 15 Scope 3 categories",
    description="List all 15 GHG Protocol Scope 3 categories with definitions and SBTi methods.",
)
async def list_categories() -> List[CategoryDefinition]:
    """List all 15 Scope 3 categories."""
    return [CategoryDefinition(**c) for c in SCOPE3_CATEGORIES]


@router.get(
    "/org/{org_id}/data-quality",
    response_model=DataQualityResponse,
    summary="Data quality by category",
    description=(
        "Assess data quality across all 15 Scope 3 categories. Returns "
        "quality scores, data source types, and improvement priorities."
    ),
)
async def get_data_quality(org_id: str) -> DataQualityResponse:
    """Assess Scope 3 data quality by category."""
    categories = [
        {"category": 1, "name": "Purchased Goods & Services", "quality_score": 3.0, "data_source": "spend_based", "completeness_pct": 85},
        {"category": 2, "name": "Capital Goods", "quality_score": 2.0, "data_source": "spend_based", "completeness_pct": 70},
        {"category": 3, "name": "Fuel & Energy Activities", "quality_score": 4.5, "data_source": "activity_based", "completeness_pct": 95},
        {"category": 4, "name": "Upstream Transportation", "quality_score": 3.0, "data_source": "distance_based", "completeness_pct": 80},
        {"category": 5, "name": "Waste Generated", "quality_score": 4.0, "data_source": "waste_type_specific", "completeness_pct": 90},
        {"category": 6, "name": "Business Travel", "quality_score": 4.5, "data_source": "activity_based", "completeness_pct": 95},
        {"category": 7, "name": "Employee Commuting", "quality_score": 2.0, "data_source": "survey_based", "completeness_pct": 60},
        {"category": 8, "name": "Upstream Leased Assets", "quality_score": 3.0, "data_source": "energy_based", "completeness_pct": 75},
        {"category": 9, "name": "Downstream Transportation", "quality_score": 2.5, "data_source": "average_data", "completeness_pct": 65},
        {"category": 10, "name": "Processing of Sold Products", "quality_score": 1.5, "data_source": "screening", "completeness_pct": 40},
        {"category": 11, "name": "Use of Sold Products", "quality_score": 2.0, "data_source": "product_spec", "completeness_pct": 55},
        {"category": 12, "name": "End-of-Life Treatment", "quality_score": 2.0, "data_source": "average_data", "completeness_pct": 50},
        {"category": 13, "name": "Downstream Leased Assets", "quality_score": 3.0, "data_source": "energy_based", "completeness_pct": 70},
        {"category": 14, "name": "Franchises", "quality_score": 2.0, "data_source": "average_data", "completeness_pct": 55},
        {"category": 15, "name": "Investments", "quality_score": 2.0, "data_source": "pcaf", "completeness_pct": 50},
    ]

    avg_quality = round(sum(c["quality_score"] for c in categories) / len(categories), 1)
    if avg_quality >= 4.0:
        level = "high"
    elif avg_quality >= 3.0:
        level = "medium"
    elif avg_quality >= 2.0:
        level = "low"
    else:
        level = "very_low"

    low_quality = [c for c in categories if c["quality_score"] < 3.0]
    priorities = [
        f"Improve data for Cat {c['category']} ({c['name']}): current score {c['quality_score']}/5"
        for c in sorted(low_quality, key=lambda x: x["quality_score"])[:5]
    ]

    return DataQualityResponse(
        org_id=org_id,
        categories=categories,
        overall_quality_score=avg_quality,
        quality_level=level,
        improvement_priorities=priorities,
        generated_at=_now(),
    )
