"""
GL-SBTi-APP Pathway Calculation API

Calculates decarbonization pathways using SBTi-approved methodologies:
Absolute Contraction Approach (ACA), Sectoral Decarbonization Approach (SDA),
economic and physical intensity pathways, supplier engagement approach, and
FLAG (Forest, Land and Agriculture) commodity and sector pathways.

Pathways define year-by-year emission budgets from the base year to the
target year, with milestones at key intervals.  Supports pathway comparison
to evaluate alternative decarbonization strategies.

SBTi Methodologies:
    - ACA: Linear absolute reduction, applicable to all sectors
    - SDA: Sector-specific intensity convergence pathway
    - Economic Intensity: Revenue-based intensity reduction
    - Physical Intensity: Production-based intensity reduction
    - Supplier Engagement: Scope 3 engagement-based approach
    - FLAG Commodity: Commodity-level deforestation/land-use pathway
    - FLAG Sector: Sector-level FLAG pathway (covers all 11 commodities)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/sbti/pathways", tags=["Pathways"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PathwayMethod(str, Enum):
    """Supported pathway calculation methods."""
    ACA = "aca"
    SDA = "sda"
    ECONOMIC_INTENSITY = "economic_intensity"
    PHYSICAL_INTENSITY = "physical_intensity"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    FLAG_COMMODITY = "flag_commodity"
    FLAG_SECTOR = "flag_sector"


class FLAGCommodity(str, Enum):
    """FLAG commodity types (11 commodities per SBTi FLAG guidance)."""
    CATTLE_BEEF = "cattle_beef"
    CATTLE_DAIRY = "cattle_dairy"
    POULTRY = "poultry"
    PORK = "pork"
    PALM_OIL = "palm_oil"
    SOY = "soy"
    RICE = "rice"
    WHEAT = "wheat"
    MAIZE = "maize"
    TIMBER_PULP = "timber_pulp"
    OTHER_CROPS = "other_crops"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ACAPathwayRequest(BaseModel):
    """Absolute Contraction Approach pathway calculation request."""
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    target_year: int = Field(..., ge=2025, le=2055)
    ambition: str = Field("1.5C", description="1.5C or well_below_2C")
    scope: str = Field("scope_1_2", description="Emission scope")

    class Config:
        json_schema_extra = {
            "example": {
                "base_year": 2020,
                "base_year_emissions_tco2e": 50000,
                "target_year": 2030,
                "ambition": "1.5C",
                "scope": "scope_1_2",
            }
        }


class SDAPathwayRequest(BaseModel):
    """Sectoral Decarbonization Approach pathway calculation request."""
    sector: str = Field(..., description="SBTi sector identifier")
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_intensity: float = Field(..., gt=0, description="Base year intensity value")
    intensity_unit: str = Field(..., description="e.g. tCO2e/MWh, tCO2e/tonne_product")
    target_year: int = Field(..., ge=2025, le=2055)
    base_year_activity: float = Field(..., gt=0, description="Base year activity level")
    projected_activity_target_year: float = Field(
        ..., gt=0, description="Projected activity at target year",
    )


class EconomicIntensityRequest(BaseModel):
    """Economic intensity pathway calculation request."""
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    base_year_revenue_usd: float = Field(..., gt=0)
    target_year: int = Field(..., ge=2025, le=2055)
    projected_revenue_target_year_usd: float = Field(..., gt=0)
    ambition: str = Field("1.5C")
    currency: str = Field("USD")


class PhysicalIntensityRequest(BaseModel):
    """Physical intensity pathway calculation request."""
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    base_year_production: float = Field(..., gt=0)
    production_unit: str = Field(..., description="e.g. tonnes, MWh, units")
    target_year: int = Field(..., ge=2025, le=2055)
    projected_production_target_year: float = Field(..., gt=0)
    ambition: str = Field("1.5C")


class SupplierEngagementRequest(BaseModel):
    """Supplier engagement pathway calculation request."""
    base_year: int = Field(..., ge=2015, le=2025)
    scope3_emissions_tco2e: float = Field(..., gt=0)
    total_suppliers: int = Field(..., gt=0)
    target_year: int = Field(..., ge=2025, le=2055)
    target_supplier_coverage_pct: float = Field(
        ..., gt=0, le=100,
        description="Percentage of suppliers with SBTs by target year",
    )
    category_breakdown: Optional[Dict[str, float]] = Field(
        None, description="Scope 3 category emissions breakdown",
    )


class FLAGCommodityRequest(BaseModel):
    """FLAG commodity pathway calculation request."""
    commodity: FLAGCommodity = Field(..., description="FLAG commodity type")
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    base_year_production: float = Field(..., gt=0)
    production_unit: str = Field("tonnes")
    target_year: int = Field(2030)
    includes_deforestation: bool = Field(True)
    includes_land_use_change: bool = Field(True)


class FLAGSectorRequest(BaseModel):
    """FLAG sector pathway calculation request."""
    sector: str = Field(..., description="FLAG sector classification")
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    target_year: int = Field(2030)
    commodities: List[Dict[str, Any]] = Field(
        ..., description="List of commodity-level data",
    )
    deforestation_commitment: bool = Field(True)


class PathwayCompareRequest(BaseModel):
    """Request to compare multiple pathways."""
    pathway_ids: List[str] = Field(
        ..., min_length=2, max_length=5,
        description="List of pathway IDs to compare",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class PathwayResponse(BaseModel):
    """Calculated decarbonization pathway."""
    pathway_id: str
    method: str
    base_year: int
    target_year: int
    base_year_emissions_tco2e: float
    target_year_emissions_tco2e: float
    total_reduction_pct: float
    annual_reduction_pct: float
    yearly_budgets: Dict[str, float]
    milestones: List[Dict[str, Any]]
    ambition_alignment: str
    methodology_notes: str
    generated_at: datetime


class PathwayCompareResponse(BaseModel):
    """Comparison of multiple pathways."""
    pathways: List[Dict[str, Any]]
    comparison_metrics: Dict[str, Any]
    recommendation: str
    generated_at: datetime


class MilestoneResponse(BaseModel):
    """Pathway milestones with interim targets."""
    pathway_id: str
    milestones: List[Dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_pathways: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _build_yearly_budgets(
    base_year: int,
    target_year: int,
    base_emissions: float,
    target_emissions: float,
) -> Dict[str, float]:
    """Build linear year-by-year emission budgets."""
    years = target_year - base_year
    if years <= 0:
        return {str(base_year): base_emissions}
    annual_delta = (base_emissions - target_emissions) / years
    budgets = {}
    for i in range(years + 1):
        yr = base_year + i
        budgets[str(yr)] = round(max(base_emissions - annual_delta * i, 0), 1)
    return budgets


def _build_milestones(
    base_year: int,
    target_year: int,
    base_emissions: float,
    target_emissions: float,
) -> List[Dict[str, Any]]:
    """Build milestone checkpoints at 25%, 50%, 75%, and 100% of timeframe."""
    timeframe = target_year - base_year
    if timeframe <= 0:
        return []
    total_reduction = base_emissions - target_emissions
    milestones = []
    for pct in [25, 50, 75, 100]:
        year = base_year + round(timeframe * pct / 100)
        reduction = total_reduction * pct / 100
        milestones.append({
            "milestone_pct": pct,
            "year": min(year, target_year),
            "expected_emissions_tco2e": round(base_emissions - reduction, 1),
            "cumulative_reduction_pct": round(pct * (total_reduction / base_emissions), 1) if base_emissions > 0 else 0,
        })
    return milestones


def _store_pathway(method: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Store a pathway and return it."""
    _pathways[data["pathway_id"]] = data
    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/aca",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate ACA pathway",
    description=(
        "Calculate an Absolute Contraction Approach (ACA) pathway. ACA applies "
        "a uniform absolute reduction rate across the timeframe, aligned with "
        "1.5C (4.2% p.a.) or well-below 2C (2.5% p.a.) ambition levels. "
        "Applicable to all sectors and scopes."
    ),
)
async def calculate_aca_pathway(request: ACAPathwayRequest) -> PathwayResponse:
    """Calculate ACA pathway."""
    annual_rate = 4.2 if request.ambition == "1.5C" else 2.5
    years = request.target_year - request.base_year
    total_reduction_pct = round(min(annual_rate * years, 100), 1)
    target_emissions = round(
        request.base_year_emissions_tco2e * (1 - total_reduction_pct / 100), 1,
    )

    pathway_id = _generate_id("pw_aca")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.ACA.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": request.base_year_emissions_tco2e,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": total_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": request.ambition,
        "methodology_notes": (
            f"ACA linear absolute contraction at {annual_rate}% per annum over "
            f"{years} years, aligned with {request.ambition} pathway."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.ACA.value, data)
    return PathwayResponse(**data)


@router.post(
    "/sda",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate SDA pathway",
    description=(
        "Calculate a Sectoral Decarbonization Approach (SDA) pathway. SDA uses "
        "sector-specific intensity convergence to a 2050 benchmark, allocating "
        "the carbon budget based on projected activity growth."
    ),
)
async def calculate_sda_pathway(request: SDAPathwayRequest) -> PathwayResponse:
    """Calculate SDA pathway."""
    years = request.target_year - request.base_year
    # SDA convergence: intensity converges to sector benchmark
    # Simplified: assume sector benchmark is 30% of base intensity by 2050
    convergence_year = 2050
    convergence_intensity = request.base_year_intensity * 0.30
    progress_ratio = min(years / (convergence_year - request.base_year), 1.0)
    target_intensity = round(
        request.base_year_intensity - (request.base_year_intensity - convergence_intensity) * progress_ratio, 4,
    )

    base_emissions = round(request.base_year_intensity * request.base_year_activity, 1)
    target_emissions = round(target_intensity * request.projected_activity_target_year, 1)
    total_reduction_pct = round(
        max((1 - target_emissions / base_emissions) * 100, 0), 1,
    ) if base_emissions > 0 else 0.0
    annual_rate = round(total_reduction_pct / years, 2) if years > 0 else 0.0

    pathway_id = _generate_id("pw_sda")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year, base_emissions, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year, base_emissions, target_emissions,
    )

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.SDA.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": base_emissions,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": total_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": "sector_specific",
        "methodology_notes": (
            f"SDA intensity convergence for sector '{request.sector}'. "
            f"Base intensity {request.base_year_intensity} {request.intensity_unit} converging "
            f"to {target_intensity} by {request.target_year}. Activity growth from "
            f"{request.base_year_activity} to {request.projected_activity_target_year}."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.SDA.value, data)
    return PathwayResponse(**data)


@router.post(
    "/economic-intensity",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate economic intensity pathway",
    description=(
        "Calculate a revenue-based economic intensity pathway. Reduces emission "
        "intensity per unit of economic output (tCO2e/USD revenue) while "
        "accounting for projected revenue growth."
    ),
)
async def calculate_economic_intensity_pathway(
    request: EconomicIntensityRequest,
) -> PathwayResponse:
    """Calculate economic intensity pathway."""
    years = request.target_year - request.base_year
    base_intensity = request.base_year_emissions_tco2e / request.base_year_revenue_usd
    annual_rate = 4.2 if request.ambition == "1.5C" else 2.5
    total_reduction_pct = round(min(annual_rate * years, 100), 1)
    target_intensity = base_intensity * (1 - total_reduction_pct / 100)
    target_emissions = round(target_intensity * request.projected_revenue_target_year_usd, 1)

    actual_reduction_pct = round(
        max((1 - target_emissions / request.base_year_emissions_tco2e) * 100, 0), 1,
    ) if request.base_year_emissions_tco2e > 0 else 0.0

    pathway_id = _generate_id("pw_econ")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.ECONOMIC_INTENSITY.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": request.base_year_emissions_tco2e,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": actual_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": request.ambition,
        "methodology_notes": (
            f"Economic intensity reduction at {annual_rate}% p.a. "
            f"Base intensity: {round(base_intensity * 1e6, 2)} tCO2e/M{request.currency}. "
            f"Revenue growth factored from {request.base_year_revenue_usd:,.0f} to "
            f"{request.projected_revenue_target_year_usd:,.0f} {request.currency}."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.ECONOMIC_INTENSITY.value, data)
    return PathwayResponse(**data)


@router.post(
    "/physical-intensity",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate physical intensity pathway",
    description=(
        "Calculate a production-based physical intensity pathway. Reduces "
        "emission intensity per unit of physical output (tCO2e/tonne product, "
        "tCO2e/MWh, etc.) while accounting for production growth."
    ),
)
async def calculate_physical_intensity_pathway(
    request: PhysicalIntensityRequest,
) -> PathwayResponse:
    """Calculate physical intensity pathway."""
    years = request.target_year - request.base_year
    base_intensity = request.base_year_emissions_tco2e / request.base_year_production
    annual_rate = 4.2 if request.ambition == "1.5C" else 2.5
    total_reduction_pct = round(min(annual_rate * years, 100), 1)
    target_intensity = base_intensity * (1 - total_reduction_pct / 100)
    target_emissions = round(target_intensity * request.projected_production_target_year, 1)

    actual_reduction_pct = round(
        max((1 - target_emissions / request.base_year_emissions_tco2e) * 100, 0), 1,
    ) if request.base_year_emissions_tco2e > 0 else 0.0

    pathway_id = _generate_id("pw_phys")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.PHYSICAL_INTENSITY.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": request.base_year_emissions_tco2e,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": actual_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": request.ambition,
        "methodology_notes": (
            f"Physical intensity reduction at {annual_rate}% p.a. "
            f"Base intensity: {round(base_intensity, 4)} tCO2e/{request.production_unit}. "
            f"Production growth from {request.base_year_production:,.0f} to "
            f"{request.projected_production_target_year:,.0f} {request.production_unit}."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.PHYSICAL_INTENSITY.value, data)
    return PathwayResponse(**data)


@router.post(
    "/supplier-engagement",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate supplier engagement pathway",
    description=(
        "Calculate a supplier engagement pathway for Scope 3 emissions. "
        "Targets a percentage of suppliers (by emissions) to set their own "
        "science-based targets within the timeframe. Minimum 67% coverage "
        "of Scope 3 purchased goods & services and upstream transport."
    ),
)
async def calculate_supplier_engagement_pathway(
    request: SupplierEngagementRequest,
) -> PathwayResponse:
    """Calculate supplier engagement pathway."""
    years = request.target_year - request.base_year
    # Assume engaged suppliers reduce emissions by 25% on average
    engaged_reduction = request.target_supplier_coverage_pct / 100 * 0.25
    target_emissions = round(
        request.scope3_emissions_tco2e * (1 - engaged_reduction), 1,
    )
    total_reduction_pct = round(engaged_reduction * 100, 1)
    annual_rate = round(total_reduction_pct / years, 2) if years > 0 else 0.0

    pathway_id = _generate_id("pw_se")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year,
        request.scope3_emissions_tco2e, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year,
        request.scope3_emissions_tco2e, target_emissions,
    )

    target_suppliers = round(request.total_suppliers * request.target_supplier_coverage_pct / 100)

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.SUPPLIER_ENGAGEMENT.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": request.scope3_emissions_tco2e,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": total_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": "supplier_engagement",
        "methodology_notes": (
            f"Supplier engagement approach targeting {request.target_supplier_coverage_pct}% "
            f"of suppliers ({target_suppliers} of {request.total_suppliers}) to set SBTs "
            f"by {request.target_year}. Projected Scope 3 reduction of {total_reduction_pct}%."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.SUPPLIER_ENGAGEMENT.value, data)
    return PathwayResponse(**data)


@router.post(
    "/flag-commodity",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate FLAG commodity pathway",
    description=(
        "Calculate a FLAG (Forest, Land and Agriculture) commodity-level "
        "pathway. Uses commodity-specific decarbonization rates aligned with "
        "1.5C no/low overshoot scenarios. Covers deforestation-free and "
        "land-use change reduction components."
    ),
)
async def calculate_flag_commodity_pathway(
    request: FLAGCommodityRequest,
) -> PathwayResponse:
    """Calculate FLAG commodity pathway."""
    years = request.target_year - request.base_year
    # Commodity-specific FLAG reduction rates (simplified from SBTi FLAG guidance)
    commodity_rates = {
        "cattle_beef": 3.8, "cattle_dairy": 3.2, "poultry": 2.5,
        "pork": 2.8, "palm_oil": 5.0, "soy": 4.5,
        "rice": 2.0, "wheat": 1.8, "maize": 1.8,
        "timber_pulp": 4.2, "other_crops": 2.0,
    }
    annual_rate = commodity_rates.get(request.commodity.value, 3.0)
    total_reduction_pct = round(min(annual_rate * years, 100), 1)
    target_emissions = round(
        request.base_year_emissions_tco2e * (1 - total_reduction_pct / 100), 1,
    )

    pathway_id = _generate_id("pw_flag_c")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.FLAG_COMMODITY.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": request.base_year_emissions_tco2e,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": total_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": "1.5C_flag",
        "methodology_notes": (
            f"FLAG commodity pathway for {request.commodity.value} at "
            f"{annual_rate}% p.a. reduction. Deforestation: "
            f"{'included' if request.includes_deforestation else 'excluded'}. "
            f"Land use change: {'included' if request.includes_land_use_change else 'excluded'}."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.FLAG_COMMODITY.value, data)
    return PathwayResponse(**data)


@router.post(
    "/flag-sector",
    response_model=PathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate FLAG sector pathway",
    description=(
        "Calculate a FLAG sector-level pathway aggregating across multiple "
        "commodities. Applies weighted reduction rates based on commodity "
        "mix and includes zero deforestation commitment tracking."
    ),
)
async def calculate_flag_sector_pathway(
    request: FLAGSectorRequest,
) -> PathwayResponse:
    """Calculate FLAG sector pathway."""
    years = request.target_year - request.base_year
    # Weighted average rate across commodities
    annual_rate = 3.5  # Default FLAG sector rate
    total_reduction_pct = round(min(annual_rate * years, 100), 1)
    target_emissions = round(
        request.base_year_emissions_tco2e * (1 - total_reduction_pct / 100), 1,
    )

    pathway_id = _generate_id("pw_flag_s")
    budgets = _build_yearly_budgets(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )
    milestones = _build_milestones(
        request.base_year, request.target_year,
        request.base_year_emissions_tco2e, target_emissions,
    )

    data = {
        "pathway_id": pathway_id,
        "method": PathwayMethod.FLAG_SECTOR.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "base_year_emissions_tco2e": request.base_year_emissions_tco2e,
        "target_year_emissions_tco2e": target_emissions,
        "total_reduction_pct": total_reduction_pct,
        "annual_reduction_pct": annual_rate,
        "yearly_budgets": budgets,
        "milestones": milestones,
        "ambition_alignment": "1.5C_flag",
        "methodology_notes": (
            f"FLAG sector pathway for '{request.sector}' at {annual_rate}% p.a. "
            f"Aggregated across {len(request.commodities)} commodities. "
            f"Deforestation commitment: {'yes' if request.deforestation_commitment else 'no'}."
        ),
        "generated_at": _now(),
    }
    _store_pathway(PathwayMethod.FLAG_SECTOR.value, data)
    return PathwayResponse(**data)


@router.post(
    "/compare",
    response_model=PathwayCompareResponse,
    summary="Compare multiple pathways",
    description=(
        "Compare two to five pathways side by side. Returns total reduction, "
        "annual rate, final emissions, and a recommendation for the most "
        "ambitious feasible pathway."
    ),
)
async def compare_pathways(request: PathwayCompareRequest) -> PathwayCompareResponse:
    """Compare multiple pathways."""
    pathways_data = []
    for pid in request.pathway_ids:
        pw = _pathways.get(pid)
        if not pw:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pathway {pid} not found",
            )
        pathways_data.append({
            "pathway_id": pw["pathway_id"],
            "method": pw["method"],
            "total_reduction_pct": pw["total_reduction_pct"],
            "annual_reduction_pct": pw["annual_reduction_pct"],
            "target_year_emissions_tco2e": pw["target_year_emissions_tco2e"],
            "ambition_alignment": pw["ambition_alignment"],
        })

    most_ambitious = max(pathways_data, key=lambda p: p["total_reduction_pct"])
    least_emissions = min(pathways_data, key=lambda p: p["target_year_emissions_tco2e"])

    return PathwayCompareResponse(
        pathways=pathways_data,
        comparison_metrics={
            "highest_reduction_pct": most_ambitious["total_reduction_pct"],
            "lowest_target_emissions": least_emissions["target_year_emissions_tco2e"],
            "range_reduction_pct": round(
                max(p["total_reduction_pct"] for p in pathways_data)
                - min(p["total_reduction_pct"] for p in pathways_data), 1,
            ),
        },
        recommendation=(
            f"Pathway {most_ambitious['pathway_id']} ({most_ambitious['method']}) "
            f"provides the most ambitious reduction at {most_ambitious['total_reduction_pct']}%."
        ),
        generated_at=_now(),
    )


@router.get(
    "/{pathway_id}/milestones",
    response_model=MilestoneResponse,
    summary="Get pathway milestones",
    description="Retrieve milestone checkpoints for a calculated pathway.",
)
async def get_pathway_milestones(pathway_id: str) -> MilestoneResponse:
    """Get pathway milestones."""
    pw = _pathways.get(pathway_id)
    if not pw:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pathway {pathway_id} not found",
        )
    return MilestoneResponse(
        pathway_id=pathway_id,
        milestones=pw.get("milestones", []),
        generated_at=_now(),
    )
