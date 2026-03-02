"""
GL-GHG-APP Scope 3 Value Chain Emissions API

Manages all 15 Scope 3 categories per GHG Protocol Corporate Value Chain
(Scope 3) Accounting and Reporting Standard:

Upstream (Cat 1-8):
    1. Purchased Goods & Services
    2. Capital Goods
    3. Fuel- and Energy-Related Activities
    4. Upstream Transportation & Distribution
    5. Waste Generated in Operations
    6. Business Travel
    7. Employee Commuting
    8. Upstream Leased Assets

Downstream (Cat 9-15):
    9. Downstream Transportation & Distribution
   10. Processing of Sold Products
   11. Use of Sold Products
   12. End-of-Life Treatment of Sold Products
   13. Downstream Leased Assets
   14. Franchises
   15. Investments

Includes materiality screening (>1% threshold), data quality tiers,
and calculation method tracking per category.
"""

from fastapi import APIRouter, HTTPException, Query, Path, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/scope3", tags=["Scope 3 Emissions"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CalculationMethod(str, Enum):
    """Scope 3 calculation methodologies per GHG Protocol."""
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    DISTANCE_BASED = "distance_based"
    FUEL_BASED = "fuel_based"
    WASTE_TYPE_SPECIFIC = "waste_type_specific"
    ASSET_SPECIFIC = "asset_specific"
    INVESTMENT_SPECIFIC = "investment_specific"


class DataQualityTier(str, Enum):
    """Data quality tiers from highest to lowest accuracy."""
    PRIMARY_SUPPLIER = "primary_supplier"      # Supplier-specific primary data
    VERIFIED_SECONDARY = "verified_secondary"  # Verified industry averages
    UNVERIFIED_SECONDARY = "unverified_secondary"  # Unverified averages
    ESTIMATED_PROXY = "estimated_proxy"        # Proxy data or estimates
    SPEND_BASED_PROXY = "spend_based_proxy"    # EEIO spend-based estimates


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class Scope3DataSubmission(BaseModel):
    """Request to submit Scope 3 category data."""
    category_number: int = Field(..., ge=1, le=15, description="Scope 3 category number (1-15)")
    subcategory: Optional[str] = Field(None, description="Subcategory label")
    description: str = Field(..., min_length=1, max_length=500, description="Activity description")
    calculation_method: CalculationMethod = Field(..., description="Calculation method used")
    activity_data: float = Field(..., gt=0, description="Activity data value")
    activity_unit: str = Field(..., description="Unit (kg, USD, km, kWh, etc.)")
    emission_factor: Optional[float] = Field(None, ge=0, description="Custom EF (kg CO2e/unit)")
    emission_factor_source: Optional[str] = Field(None, description="EF database source")
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.UNVERIFIED_SECONDARY, description="Data quality tier"
    )
    supplier_name: Optional[str] = Field(None, description="Supplier name (for Cat 1-2)")
    origin_country: Optional[str] = Field(None, description="Origin country code")
    period_start: Optional[str] = Field(None, description="Period start (YYYY-MM-DD)")
    period_end: Optional[str] = Field(None, description="Period end (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "category_number": 1,
                "subcategory": "Raw Materials",
                "description": "Steel purchases from primary supplier",
                "calculation_method": "supplier_specific",
                "activity_data": 5000,
                "activity_unit": "tonnes",
                "emission_factor": 1.85,
                "emission_factor_source": "Supplier EPD",
                "data_quality_tier": "primary_supplier",
                "supplier_name": "ArcelorMittal",
                "origin_country": "DE",
                "period_start": "2025-01-01",
                "period_end": "2025-12-31"
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation & Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation & Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}


class Scope3CategorySummary(BaseModel):
    """Summary of a single Scope 3 category."""
    category_number: int
    category_name: str
    direction: str  # upstream or downstream
    total_tco2e: float
    percentage_of_scope3: float
    percentage_of_total: float
    is_material: bool
    calculation_method: str
    data_quality_tier: str
    data_quality_score: float
    data_completeness_pct: float
    supplier_count: Optional[int] = None
    activity_data_summary: Optional[str] = None


class Scope3Summary(BaseModel):
    """Scope 3 summary across all 15 categories."""
    inventory_id: str
    total_tco2e: float
    upstream_tco2e: float
    downstream_tco2e: float
    categories_reported: int
    categories_material: int
    top_category: str
    top_category_pct: float
    data_quality_score: float
    completeness_pct: float
    year_over_year_change_pct: Optional[float]


class MaterialityResult(BaseModel):
    """Materiality screening result for a category."""
    category_number: int
    category_name: str
    total_tco2e: float
    percentage_of_total: float
    is_material: bool
    threshold_pct: float
    screening_method: str
    justification: str
    recommended_action: str


class MethodDetail(BaseModel):
    """Calculation method details for a category."""
    category_number: int
    category_name: str
    primary_method: str
    method_description: str
    data_sources: List[str]
    emission_factor_databases: List[str]
    uncertainty_range_pct: float
    improvement_opportunities: List[str]


class CategoryDataQuality(BaseModel):
    """Data quality assessment for a category."""
    category_number: int
    category_name: str
    overall_score: float
    tier: str
    completeness_pct: float
    accuracy_score: float
    temporal_representativeness: float
    geographical_representativeness: float
    technological_representativeness: float
    primary_data_pct: float
    improvement_priority: str


class Scope3AggregationResponse(BaseModel):
    """Response for Scope 3 aggregation."""
    inventory_id: str
    status: str
    total_tco2e: float
    categories_aggregated: int
    categories_material: int
    data_records_processed: int
    aggregated_at: datetime


class Scope3DataResponse(BaseModel):
    """Response for Scope 3 data submission."""
    record_id: str
    inventory_id: str
    category_number: int
    category_name: str
    calculated_tco2e: float
    calculation_method: str
    data_quality_tier: str
    status: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Simulated Data
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _simulated_scope3_categories(inventory_id: str) -> List[Dict[str, Any]]:
    """Generate simulated data for all 15 Scope 3 categories."""
    total_scope3 = 45230.2
    total_inventory = 66001.5  # scope1 + scope2 + scope3

    raw_categories = [
        # Upstream
        (1, 18500.0, "spend_based", "spend_based_proxy", 68.0, 78, "USD 125M in purchases"),
        (2, 3200.0, "spend_based", "estimated_proxy", 55.0, None, "USD 22M capital expenditure"),
        (3, 2100.0, "average_data", "verified_secondary", 82.0, None, "WTT + T&D losses"),
        (4, 4800.0, "distance_based", "unverified_secondary", 72.0, 12, "850K tonne-km transported"),
        (5, 1950.0, "waste_type_specific", "verified_secondary", 85.0, None, "3,200 tonnes waste"),
        (6, 2800.0, "distance_based", "unverified_secondary", 78.0, None, "4.2M passenger-km"),
        (7, 1600.0, "average_data", "estimated_proxy", 50.0, None, "1,850 employees commuting"),
        (8, 980.0, "asset_specific", "unverified_secondary", 65.0, None, "3 leased warehouses"),
        # Downstream
        (9, 2400.0, "distance_based", "unverified_secondary", 70.0, None, "Product distribution"),
        (10, 1200.0, "average_data", "estimated_proxy", 45.0, None, "Intermediate products"),
        (11, 3500.0, "average_data", "estimated_proxy", 40.0, None, "Product energy use in lifetime"),
        (12, 850.0, "waste_type_specific", "estimated_proxy", 48.0, None, "End-of-life treatment"),
        (13, 650.0, "asset_specific", "unverified_secondary", 62.0, None, "2 leased retail spaces"),
        (14, 0.0, "average_data", "estimated_proxy", 0.0, None, "No franchises"),
        (15, 700.0, "investment_specific", "estimated_proxy", 35.0, None, "USD 15M equity investments"),
    ]

    results = []
    for cat_num, tco2e, method, tier, completeness, suppliers, summary in raw_categories:
        pct_scope3 = round(tco2e / total_scope3 * 100, 2) if total_scope3 > 0 else 0.0
        pct_total = round(tco2e / total_inventory * 100, 2) if total_inventory > 0 else 0.0
        is_material = pct_total >= 1.0  # 1% materiality threshold
        direction = "upstream" if cat_num <= 8 else "downstream"

        # Data quality score based on tier
        tier_scores = {
            "primary_supplier": 95.0,
            "verified_secondary": 80.0,
            "unverified_secondary": 65.0,
            "estimated_proxy": 45.0,
            "spend_based_proxy": 35.0,
        }
        dq_score = tier_scores.get(tier, 50.0)

        results.append({
            "category_number": cat_num,
            "category_name": CATEGORY_NAMES[cat_num],
            "direction": direction,
            "total_tco2e": tco2e,
            "percentage_of_scope3": pct_scope3,
            "percentage_of_total": pct_total,
            "is_material": is_material,
            "calculation_method": method,
            "data_quality_tier": tier,
            "data_quality_score": dq_score,
            "data_completeness_pct": completeness,
            "supplier_count": suppliers,
            "activity_data_summary": summary,
        })
    return results


# Default Scope 3 emission factors (kg CO2e per unit) for demo calculations
SCOPE3_DEFAULT_EFS: Dict[int, Dict[str, float]] = {
    1: {"USD": 0.39, "kg_steel": 1.85, "kg_plastic": 3.1, "kg_paper": 0.92},
    2: {"USD": 0.47},
    3: {"kWh": 0.05, "litre_diesel_wtt": 0.62},
    4: {"tonne_km_road": 0.107, "tonne_km_rail": 0.028, "tonne_km_sea": 0.016, "tonne_km_air": 0.602},
    5: {"tonne_landfill": 0.587, "tonne_recycled": 0.021, "tonne_incineration": 0.395},
    6: {"passenger_km_air_short": 0.255, "passenger_km_air_long": 0.195, "hotel_night": 20.6},
    7: {"passenger_km_car": 0.171, "passenger_km_bus": 0.089, "passenger_km_rail": 0.041},
    8: {"sqm_warehouse": 35.0, "sqm_office": 120.0},
    9: {"tonne_km_road": 0.107},
    10: {"tonne_product": 0.5},
    11: {"kWh": 0.417},
    12: {"tonne_eol": 0.45},
    13: {"sqm_retail": 150.0},
    14: {"revenue_USD": 0.0},
    15: {"USD_invested": 0.12},
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/aggregate/{inventory_id}",
    response_model=Scope3AggregationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aggregate all 15 Scope 3 categories",
    description=(
        "Trigger aggregation of all Scope 3 value chain categories. "
        "Runs materiality screening, applies calculation methods, and "
        "computes totals for upstream (Cat 1-8) and downstream (Cat 9-15)."
    ),
)
async def aggregate_scope3(inventory_id: str) -> Scope3AggregationResponse:
    cats = _simulated_scope3_categories(inventory_id)
    material_count = sum(1 for c in cats if c["is_material"])
    return Scope3AggregationResponse(
        inventory_id=inventory_id,
        status="completed",
        total_tco2e=45230.2,
        categories_aggregated=15,
        categories_material=material_count,
        data_records_processed=215,
        aggregated_at=_now(),
    )


@router.get(
    "/{inventory_id}/summary",
    response_model=Scope3Summary,
    summary="Scope 3 summary",
    description="High-level Scope 3 summary with upstream/downstream split.",
)
async def get_scope3_summary(inventory_id: str) -> Scope3Summary:
    cats = _simulated_scope3_categories(inventory_id)
    upstream = sum(c["total_tco2e"] for c in cats if c["direction"] == "upstream")
    downstream = sum(c["total_tco2e"] for c in cats if c["direction"] == "downstream")
    reported = sum(1 for c in cats if c["total_tco2e"] > 0)
    material = sum(1 for c in cats if c["is_material"])
    top = max(cats, key=lambda c: c["total_tco2e"])
    avg_dq = sum(c["data_quality_score"] for c in cats if c["total_tco2e"] > 0) / max(reported, 1)
    avg_comp = sum(c["data_completeness_pct"] for c in cats if c["total_tco2e"] > 0) / max(reported, 1)

    return Scope3Summary(
        inventory_id=inventory_id,
        total_tco2e=round(upstream + downstream, 2),
        upstream_tco2e=round(upstream, 2),
        downstream_tco2e=round(downstream, 2),
        categories_reported=reported,
        categories_material=material,
        top_category=top["category_name"],
        top_category_pct=top["percentage_of_scope3"],
        data_quality_score=round(avg_dq, 1),
        completeness_pct=round(avg_comp, 1),
        year_over_year_change_pct=2.1,
    )


@router.get(
    "/{inventory_id}/categories",
    response_model=List[Scope3CategorySummary],
    summary="All 15 Scope 3 categories",
    description=(
        "Detailed breakdown of all 15 Scope 3 categories with materiality "
        "flags, calculation methods, data quality tiers, and completeness."
    ),
)
async def get_scope3_categories(
    inventory_id: str,
    material_only: bool = Query(False, description="Return only material categories"),
    direction: Optional[str] = Query(None, description="Filter: upstream or downstream"),
) -> List[Scope3CategorySummary]:
    cats = _simulated_scope3_categories(inventory_id)
    if material_only:
        cats = [c for c in cats if c["is_material"]]
    if direction:
        cats = [c for c in cats if c["direction"] == direction]
    return [Scope3CategorySummary(**c) for c in cats]


@router.get(
    "/{inventory_id}/category/{cat_number}",
    response_model=Scope3CategorySummary,
    summary="Single Scope 3 category detail",
    description="Retrieve details for a specific Scope 3 category (1-15).",
)
async def get_scope3_category(
    inventory_id: str,
    cat_number: int = Path(..., ge=1, le=15, description="Category number 1-15"),
) -> Scope3CategorySummary:
    cats = _simulated_scope3_categories(inventory_id)
    cat = next((c for c in cats if c["category_number"] == cat_number), None)
    if not cat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {cat_number} not found",
        )
    return Scope3CategorySummary(**cat)


@router.get(
    "/{inventory_id}/materiality",
    response_model=List[MaterialityResult],
    summary="Materiality screening results",
    description=(
        "Materiality screening for all 15 categories using the 1% threshold "
        "rule: categories contributing >= 1% of total inventory emissions "
        "are considered material and must be reported."
    ),
)
async def get_scope3_materiality(
    inventory_id: str,
    threshold_pct: float = Query(1.0, ge=0, le=100, description="Materiality threshold percentage"),
) -> List[MaterialityResult]:
    cats = _simulated_scope3_categories(inventory_id)
    total_inventory = 66001.5
    results = []
    for c in cats:
        pct = c["percentage_of_total"]
        is_material = pct >= threshold_pct
        if is_material:
            justification = (
                f"Category contributes {pct}% of total emissions, "
                f"exceeding the {threshold_pct}% materiality threshold."
            )
            action = "Report and set reduction targets"
        elif c["total_tco2e"] > 0:
            justification = (
                f"Category contributes {pct}% of total emissions, "
                f"below the {threshold_pct}% threshold. Monitoring recommended."
            )
            action = "Monitor and reassess annually"
        else:
            justification = "No emissions identified for this category."
            action = "Document exclusion rationale"

        results.append(MaterialityResult(
            category_number=c["category_number"],
            category_name=c["category_name"],
            total_tco2e=c["total_tco2e"],
            percentage_of_total=pct,
            is_material=is_material,
            threshold_pct=threshold_pct,
            screening_method="quantitative_threshold",
            justification=justification,
            recommended_action=action,
        ))
    return results


@router.get(
    "/{inventory_id}/methods",
    response_model=List[MethodDetail],
    summary="Calculation methods per category",
    description=(
        "Show the calculation methodology used for each Scope 3 category, "
        "including data sources, emission factor databases, uncertainty, "
        "and improvement opportunities."
    ),
)
async def get_scope3_methods(inventory_id: str) -> List[MethodDetail]:
    method_descriptions = {
        "supplier_specific": "Primary data from suppliers (EPDs, direct measurement)",
        "hybrid": "Combination of supplier-specific and secondary data",
        "average_data": "Industry-average emission factors applied to activity data",
        "spend_based": "EEIO factors applied to spend data (USD)",
        "distance_based": "Distance-based factors applied to transport data",
        "fuel_based": "Fuel-based factors applied to fuel consumption",
        "waste_type_specific": "Waste-type-specific factors by treatment method",
        "asset_specific": "Asset-level data for leased/owned properties",
        "investment_specific": "Investment-specific data from portfolio companies",
    }

    cats = _simulated_scope3_categories(inventory_id)
    results = []
    for c in cats:
        method = c["calculation_method"]
        results.append(MethodDetail(
            category_number=c["category_number"],
            category_name=c["category_name"],
            primary_method=method,
            method_description=method_descriptions.get(method, "Standard calculation"),
            data_sources=_get_data_sources(c["category_number"], method),
            emission_factor_databases=_get_ef_databases(c["category_number"], method),
            uncertainty_range_pct=_get_uncertainty(c["data_quality_tier"]),
            improvement_opportunities=_get_improvements(c["category_number"], method),
        ))
    return results


@router.get(
    "/{inventory_id}/data-quality",
    response_model=List[CategoryDataQuality],
    summary="Data quality per category",
    description=(
        "Data quality assessment for each Scope 3 category using five "
        "representativeness dimensions: temporal, geographical, technological, "
        "completeness, and accuracy."
    ),
)
async def get_scope3_data_quality(inventory_id: str) -> List[CategoryDataQuality]:
    cats = _simulated_scope3_categories(inventory_id)
    results = []
    for c in cats:
        if c["total_tco2e"] == 0:
            priority = "not_applicable"
        elif c["data_quality_score"] < 50:
            priority = "high"
        elif c["data_quality_score"] < 70:
            priority = "medium"
        else:
            priority = "low"

        base_score = c["data_quality_score"]
        results.append(CategoryDataQuality(
            category_number=c["category_number"],
            category_name=c["category_name"],
            overall_score=base_score,
            tier=c["data_quality_tier"],
            completeness_pct=c["data_completeness_pct"],
            accuracy_score=base_score * 0.95,
            temporal_representativeness=min(base_score * 1.05, 100.0),
            geographical_representativeness=base_score * 0.9,
            technological_representativeness=base_score * 0.85,
            primary_data_pct=_tier_to_primary_pct(c["data_quality_tier"]),
            improvement_priority=priority,
        ))
    return results


@router.post(
    "/{inventory_id}/data",
    response_model=Scope3DataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Scope 3 category data",
    description=(
        "Submit activity data for a Scope 3 category. Provide category number, "
        "calculation method, activity data, and optionally a custom emission factor."
    ),
)
async def submit_scope3_data(
    inventory_id: str,
    data: Scope3DataSubmission,
) -> Scope3DataResponse:
    if data.category_number not in CATEGORY_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category number {data.category_number}. Must be 1-15.",
        )

    # Determine emission factor
    ef = data.emission_factor
    if ef is None:
        cat_efs = SCOPE3_DEFAULT_EFS.get(data.category_number, {})
        ef = cat_efs.get(data.activity_unit, 0.0)

    calculated_tco2e = round(data.activity_data * ef / 1000.0, 4)

    record_id = _generate_id("s3d")
    return Scope3DataResponse(
        record_id=record_id,
        inventory_id=inventory_id,
        category_number=data.category_number,
        category_name=CATEGORY_NAMES[data.category_number],
        calculated_tco2e=calculated_tco2e,
        calculation_method=data.calculation_method.value,
        data_quality_tier=data.data_quality_tier.value,
        status="accepted",
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_data_sources(cat_number: int, method: str) -> List[str]:
    """Return plausible data sources for a category and method."""
    sources_map = {
        1: ["Supplier invoices", "Procurement system (SAP)", "EPDs from suppliers"],
        2: ["CapEx ledger", "Fixed asset register", "Supplier EPDs"],
        3: ["Utility bills", "Fuel purchase records", "Grid operator data"],
        4: ["Shipping manifests", "3PL reports", "Freight invoices"],
        5: ["Waste hauler reports", "Waste tracking manifests", "On-site waste logs"],
        6: ["Travel booking system", "Expense reports", "Travel management company data"],
        7: ["Employee survey", "HR commuting data", "Parking records"],
        8: ["Lease agreements", "Landlord utility data", "Energy audits"],
        9: ["Distribution records", "3PL reports"],
        10: ["Customer processing data", "Industry averages"],
        11: ["Product specifications", "Usage surveys"],
        12: ["Product lifecycle data", "Waste statistics"],
        13: ["Lease agreements", "Tenant utility data"],
        14: ["Franchise reporting"],
        15: ["Investment portfolio data", "Company annual reports"],
    }
    return sources_map.get(cat_number, ["Activity data records"])


def _get_ef_databases(cat_number: int, method: str) -> List[str]:
    """Return relevant emission factor databases."""
    if method == "spend_based":
        return ["USEEIO v2.0", "EXIOBASE 3", "DEFRA Conversion Factors"]
    if method == "distance_based":
        return ["DEFRA Conversion Factors", "EPA SmartWay", "GLEC Framework"]
    if method == "waste_type_specific":
        return ["EPA WARM v16", "DEFRA Conversion Factors", "IPCC Guidelines"]
    return ["EPA GHG Emission Factors Hub", "DEFRA Conversion Factors", "ecoinvent 3.10"]


def _get_uncertainty(tier: str) -> float:
    """Return estimated uncertainty range based on data quality tier."""
    uncertainty_map = {
        "primary_supplier": 5.0,
        "verified_secondary": 15.0,
        "unverified_secondary": 30.0,
        "estimated_proxy": 50.0,
        "spend_based_proxy": 60.0,
    }
    return uncertainty_map.get(tier, 40.0)


def _get_improvements(cat_number: int, method: str) -> List[str]:
    """Return improvement opportunities for a category."""
    if method == "spend_based":
        return [
            "Transition to supplier-specific data for top 20 suppliers",
            "Collect product-level EPDs from key suppliers",
            "Improve spend categorization granularity",
        ]
    if method == "average_data":
        return [
            "Engage suppliers for primary activity data",
            "Use more region-specific emission factors",
            "Improve data collection frequency to quarterly",
        ]
    if method == "distance_based":
        return [
            "Collect actual fuel consumption from carriers",
            "Obtain mode-specific transport data",
            "Implement real-time tracking for logistics",
        ]
    return ["Collect primary data where possible", "Improve temporal coverage"]


def _tier_to_primary_pct(tier: str) -> float:
    """Convert data quality tier to approximate primary data percentage."""
    mapping = {
        "primary_supplier": 90.0,
        "verified_secondary": 40.0,
        "unverified_secondary": 15.0,
        "estimated_proxy": 5.0,
        "spend_based_proxy": 0.0,
    }
    return mapping.get(tier, 10.0)
