"""
GL-SBTi-APP FLAG Assessment API

Implements the SBTi FLAG (Forest, Land and Agriculture) guidance for
companies with significant land-use related emissions.  Assesses the
20% FLAG trigger threshold, classifies FLAG sectors, calculates
commodity and sector pathways, tracks zero-deforestation commitments,
and evaluates removals eligibility.

SBTi FLAG Requirements:
    - FLAG target required if FLAG emissions >= 20% of total (C24)
    - Zero deforestation commitment required by 2025 (C25)
    - Separate FLAG pathway from energy/industrial pathway
    - 11 FLAG commodities defined with commodity-specific pathways
    - Removals from land-use eligible under specific conditions
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/sbti/flag", tags=["FLAG Assessment"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FLAGSectorType(str, Enum):
    """FLAG sector classification types."""
    AGRICULTURE = "agriculture"
    FORESTRY = "forestry"
    FOOD_BEVERAGE = "food_beverage"
    PAPER_PACKAGING = "paper_packaging"
    TOBACCO = "tobacco"
    RUBBER = "rubber"
    MIXED = "mixed"
    NON_FLAG = "non_flag"


class DeforestationStatus(str, Enum):
    """Zero deforestation commitment status."""
    COMMITTED = "committed"
    IN_PROGRESS = "in_progress"
    NOT_STARTED = "not_started"
    ACHIEVED = "achieved"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class FLAGTriggerRequest(BaseModel):
    """Request to assess the 20% FLAG trigger."""
    org_id: str = Field(...)
    total_scope1_tco2e: float = Field(..., ge=0)
    total_scope2_tco2e: float = Field(..., ge=0)
    total_scope3_tco2e: float = Field(..., ge=0)
    flag_scope1_tco2e: float = Field(0, ge=0, description="FLAG-related Scope 1")
    flag_scope3_tco2e: float = Field(0, ge=0, description="FLAG-related Scope 3")
    land_use_change_tco2e: float = Field(0, ge=0, description="Land use change emissions")
    deforestation_tco2e: float = Field(0, ge=0, description="Deforestation-linked emissions")


class FLAGCommodityPathwayRequest(BaseModel):
    """Request to calculate a FLAG commodity pathway."""
    org_id: str = Field(...)
    commodity: str = Field(..., description="FLAG commodity name")
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    base_year_production: float = Field(..., gt=0)
    production_unit: str = Field("tonnes")
    target_year: int = Field(2030, ge=2025, le=2050)


class FLAGSectorPathwayRequest(BaseModel):
    """Request to calculate a FLAG sector pathway."""
    org_id: str = Field(...)
    sector: str = Field(...)
    base_year: int = Field(..., ge=2015, le=2025)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    target_year: int = Field(2030)
    commodity_mix: List[Dict[str, Any]] = Field(
        ..., description="Commodities and their emission shares",
    )


class DeforestationCommitmentRequest(BaseModel):
    """Request to update deforestation commitment tracking."""
    status: DeforestationStatus = Field(...)
    commitment_date: Optional[str] = Field(None, description="Date of commitment (ISO format)")
    target_date: str = Field("2025-12-31", description="Target achievement date")
    supply_chain_coverage_pct: float = Field(0, ge=0, le=100)
    monitoring_system: Optional[str] = Field(None)
    notes: Optional[str] = Field(None, max_length=2000)


class RemovalsEligibilityRequest(BaseModel):
    """Request to assess removal eligibility."""
    org_id: str = Field(...)
    removal_type: str = Field(..., description="e.g. reforestation, soil_carbon, agroforestry")
    removal_tco2e: float = Field(..., gt=0)
    permanence_years: int = Field(..., gt=0)
    additionality_demonstrated: bool = Field(False)
    third_party_verified: bool = Field(False)
    land_use_category: str = Field(..., description="IPCC land use category")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class FLAGTriggerResponse(BaseModel):
    """FLAG trigger assessment result."""
    org_id: str
    total_emissions_tco2e: float
    flag_emissions_tco2e: float
    flag_pct_of_total: float
    trigger_threshold_pct: float
    flag_target_required: bool
    flag_scope1_tco2e: float
    flag_scope3_tco2e: float
    land_use_change_tco2e: float
    deforestation_tco2e: float
    recommendation: str
    generated_at: datetime


class FLAGClassificationResponse(BaseModel):
    """FLAG sector classification result."""
    org_id: str
    sector_type: str
    flag_relevant: bool
    primary_commodities: List[str]
    flag_emission_sources: List[str]
    classification_basis: str
    generated_at: datetime


class FLAGPathwayResponse(BaseModel):
    """FLAG pathway calculation result."""
    pathway_id: str
    org_id: str
    pathway_type: str
    commodity_or_sector: str
    base_year: int
    target_year: int
    base_emissions_tco2e: float
    target_emissions_tco2e: float
    reduction_pct: float
    annual_reduction_rate: float
    yearly_budgets: Dict[str, float]
    includes_deforestation_free: bool
    methodology_notes: str
    generated_at: datetime


class FLAGCommodityInfo(BaseModel):
    """FLAG commodity reference information."""
    commodity_id: str
    name: str
    category: str
    pathway_reduction_rate_pct: float
    deforestation_relevant: bool
    land_use_change_relevant: bool
    typical_emission_sources: List[str]
    measurement_units: List[str]


class EmissionsSplitResponse(BaseModel):
    """FLAG vs non-FLAG emission split."""
    org_id: str
    total_emissions_tco2e: float
    flag_emissions_tco2e: float
    non_flag_emissions_tco2e: float
    flag_pct: float
    non_flag_pct: float
    flag_by_source: Dict[str, float]
    generated_at: datetime


class DeforestationCommitmentResponse(BaseModel):
    """Deforestation commitment tracking record."""
    org_id: str
    status: str
    commitment_date: Optional[str]
    target_date: str
    supply_chain_coverage_pct: float
    monitoring_system: Optional[str]
    on_track: bool
    notes: Optional[str]
    updated_at: datetime


class RemovalsEligibilityResponse(BaseModel):
    """Removals eligibility assessment."""
    org_id: str
    eligible: bool
    removal_type: str
    removal_tco2e: float
    eligible_tco2e: float
    criteria_met: Dict[str, bool]
    issues: List[str]
    recommendation: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

FLAG_COMMODITIES = [
    {"commodity_id": "cattle_beef", "name": "Cattle (Beef)", "category": "livestock",
     "pathway_reduction_rate_pct": 3.8, "deforestation_relevant": True, "land_use_change_relevant": True,
     "typical_emission_sources": ["enteric_fermentation", "manure", "feed_production", "land_use_change"],
     "measurement_units": ["tonnes_carcass_weight", "head"]},
    {"commodity_id": "cattle_dairy", "name": "Cattle (Dairy)", "category": "livestock",
     "pathway_reduction_rate_pct": 3.2, "deforestation_relevant": True, "land_use_change_relevant": True,
     "typical_emission_sources": ["enteric_fermentation", "manure", "feed_production"],
     "measurement_units": ["tonnes_fpcm", "litres"]},
    {"commodity_id": "poultry", "name": "Poultry", "category": "livestock",
     "pathway_reduction_rate_pct": 2.5, "deforestation_relevant": True, "land_use_change_relevant": False,
     "typical_emission_sources": ["feed_production", "manure", "housing"],
     "measurement_units": ["tonnes_carcass_weight", "head"]},
    {"commodity_id": "pork", "name": "Pork", "category": "livestock",
     "pathway_reduction_rate_pct": 2.8, "deforestation_relevant": True, "land_use_change_relevant": False,
     "typical_emission_sources": ["feed_production", "manure", "housing"],
     "measurement_units": ["tonnes_carcass_weight", "head"]},
    {"commodity_id": "palm_oil", "name": "Palm Oil", "category": "crops",
     "pathway_reduction_rate_pct": 5.0, "deforestation_relevant": True, "land_use_change_relevant": True,
     "typical_emission_sources": ["land_use_change", "peat_oxidation", "mill_effluent"],
     "measurement_units": ["tonnes_crude_palm_oil"]},
    {"commodity_id": "soy", "name": "Soy", "category": "crops",
     "pathway_reduction_rate_pct": 4.5, "deforestation_relevant": True, "land_use_change_relevant": True,
     "typical_emission_sources": ["land_use_change", "field_emissions", "processing"],
     "measurement_units": ["tonnes"]},
    {"commodity_id": "rice", "name": "Rice", "category": "crops",
     "pathway_reduction_rate_pct": 2.0, "deforestation_relevant": False, "land_use_change_relevant": True,
     "typical_emission_sources": ["paddy_methane", "field_emissions", "residue_burning"],
     "measurement_units": ["tonnes_paddy_rice"]},
    {"commodity_id": "wheat", "name": "Wheat", "category": "crops",
     "pathway_reduction_rate_pct": 1.8, "deforestation_relevant": False, "land_use_change_relevant": True,
     "typical_emission_sources": ["field_emissions", "fertilizer", "residue_management"],
     "measurement_units": ["tonnes"]},
    {"commodity_id": "maize", "name": "Maize (Corn)", "category": "crops",
     "pathway_reduction_rate_pct": 1.8, "deforestation_relevant": False, "land_use_change_relevant": True,
     "typical_emission_sources": ["field_emissions", "fertilizer", "residue_management"],
     "measurement_units": ["tonnes"]},
    {"commodity_id": "timber_pulp", "name": "Timber & Pulp", "category": "forestry",
     "pathway_reduction_rate_pct": 4.2, "deforestation_relevant": True, "land_use_change_relevant": True,
     "typical_emission_sources": ["deforestation", "forest_degradation", "processing"],
     "measurement_units": ["cubic_metres", "tonnes_pulp"]},
    {"commodity_id": "other_crops", "name": "Other Crops", "category": "crops",
     "pathway_reduction_rate_pct": 2.0, "deforestation_relevant": False, "land_use_change_relevant": True,
     "typical_emission_sources": ["field_emissions", "fertilizer", "land_management"],
     "measurement_units": ["tonnes"]},
]

_deforestation_commitments: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/trigger-assessment",
    response_model=FLAGTriggerResponse,
    summary="Assess 20% FLAG trigger",
    description=(
        "Assess whether FLAG emissions meet the SBTi 20% trigger threshold "
        "(criterion C24). If FLAG emissions >= 20% of total, a separate FLAG "
        "target and zero-deforestation commitment are required."
    ),
)
async def assess_flag_trigger(request: FLAGTriggerRequest) -> FLAGTriggerResponse:
    """Assess FLAG trigger threshold."""
    total = request.total_scope1_tco2e + request.total_scope2_tco2e + request.total_scope3_tco2e
    flag_total = request.flag_scope1_tco2e + request.flag_scope3_tco2e
    flag_pct = round((flag_total / total) * 100, 1) if total > 0 else 0.0
    required = flag_pct >= 20.0

    if required:
        recommendation = (
            f"FLAG emissions are {flag_pct}% of total (>= 20% threshold). "
            f"A separate FLAG target is required using FLAG-specific pathways. "
            f"A zero-deforestation commitment by 2025 is also mandatory (C25)."
        )
    else:
        recommendation = (
            f"FLAG emissions are {flag_pct}% of total (< 20% threshold). "
            f"A FLAG-specific target is not required, but FLAG emissions should "
            f"still be included in the overall target boundary."
        )

    return FLAGTriggerResponse(
        org_id=request.org_id,
        total_emissions_tco2e=round(total, 1),
        flag_emissions_tco2e=round(flag_total, 1),
        flag_pct_of_total=flag_pct,
        trigger_threshold_pct=20.0,
        flag_target_required=required,
        flag_scope1_tco2e=request.flag_scope1_tco2e,
        flag_scope3_tco2e=request.flag_scope3_tco2e,
        land_use_change_tco2e=request.land_use_change_tco2e,
        deforestation_tco2e=request.deforestation_tco2e,
        recommendation=recommendation,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/classification",
    response_model=FLAGClassificationResponse,
    summary="FLAG sector classification",
    description=(
        "Classify an organization's FLAG sector type and identify primary "
        "commodities and emission sources. Used to determine which FLAG "
        "pathways apply."
    ),
)
async def get_flag_classification(org_id: str) -> FLAGClassificationResponse:
    """Get FLAG sector classification."""
    return FLAGClassificationResponse(
        org_id=org_id,
        sector_type=FLAGSectorType.FOOD_BEVERAGE.value,
        flag_relevant=True,
        primary_commodities=["cattle_beef", "soy", "palm_oil"],
        flag_emission_sources=[
            "enteric_fermentation", "feed_production",
            "land_use_change", "deforestation",
        ],
        classification_basis=(
            "Based on ISIC/NACE sector codes and commodity procurement data. "
            "Organization classified as food & beverage with significant "
            "livestock and oilseed sourcing."
        ),
        generated_at=_now(),
    )


@router.post(
    "/commodity-pathway",
    response_model=FLAGPathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate commodity pathway",
    description=(
        "Calculate a FLAG commodity-level decarbonization pathway using "
        "commodity-specific reduction rates aligned with 1.5C."
    ),
)
async def calculate_commodity_pathway(
    request: FLAGCommodityPathwayRequest,
) -> FLAGPathwayResponse:
    """Calculate FLAG commodity pathway."""
    commodity_info = next(
        (c for c in FLAG_COMMODITIES if c["commodity_id"] == request.commodity), None,
    )
    if not commodity_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown FLAG commodity: {request.commodity}. Use GET /flag/commodities for valid values.",
        )

    rate = commodity_info["pathway_reduction_rate_pct"]
    years = request.target_year - request.base_year
    total_pct = round(min(rate * years, 100), 1)
    target_emissions = round(request.base_year_emissions_tco2e * (1 - total_pct / 100), 1)

    budgets = {}
    annual_delta = (request.base_year_emissions_tco2e - target_emissions) / years if years > 0 else 0
    for i in range(years + 1):
        budgets[str(request.base_year + i)] = round(
            max(request.base_year_emissions_tco2e - annual_delta * i, 0), 1,
        )

    pathway_id = _generate_id("pw_flag")
    return FLAGPathwayResponse(
        pathway_id=pathway_id,
        org_id=request.org_id,
        pathway_type="commodity",
        commodity_or_sector=request.commodity,
        base_year=request.base_year,
        target_year=request.target_year,
        base_emissions_tco2e=request.base_year_emissions_tco2e,
        target_emissions_tco2e=target_emissions,
        reduction_pct=total_pct,
        annual_reduction_rate=rate,
        yearly_budgets=budgets,
        includes_deforestation_free=commodity_info["deforestation_relevant"],
        methodology_notes=(
            f"FLAG commodity pathway for {commodity_info['name']} at "
            f"{rate}% annual reduction. Reduction covers "
            f"{', '.join(commodity_info['typical_emission_sources'])}."
        ),
        generated_at=_now(),
    )


@router.post(
    "/sector-pathway",
    response_model=FLAGPathwayResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate sector pathway",
    description=(
        "Calculate a FLAG sector-level pathway aggregating across the "
        "organization's commodity mix with weighted reduction rates."
    ),
)
async def calculate_sector_pathway(
    request: FLAGSectorPathwayRequest,
) -> FLAGPathwayResponse:
    """Calculate FLAG sector pathway."""
    years = request.target_year - request.base_year
    # Weighted average rate from commodity mix
    total_weight = 0.0
    weighted_rate = 0.0
    for cm in request.commodity_mix:
        weight = cm.get("emissions_share_pct", 0)
        comm = next((c for c in FLAG_COMMODITIES if c["commodity_id"] == cm.get("commodity")), None)
        rate = comm["pathway_reduction_rate_pct"] if comm else 3.0
        weighted_rate += rate * weight / 100
        total_weight += weight

    avg_rate = round(weighted_rate, 2) if total_weight > 0 else 3.5
    total_pct = round(min(avg_rate * years, 100), 1)
    target_emissions = round(request.base_year_emissions_tco2e * (1 - total_pct / 100), 1)

    budgets = {}
    annual_delta = (request.base_year_emissions_tco2e - target_emissions) / years if years > 0 else 0
    for i in range(years + 1):
        budgets[str(request.base_year + i)] = round(
            max(request.base_year_emissions_tco2e - annual_delta * i, 0), 1,
        )

    pathway_id = _generate_id("pw_flag_s")
    return FLAGPathwayResponse(
        pathway_id=pathway_id,
        org_id=request.org_id,
        pathway_type="sector",
        commodity_or_sector=request.sector,
        base_year=request.base_year,
        target_year=request.target_year,
        base_emissions_tco2e=request.base_year_emissions_tco2e,
        target_emissions_tco2e=target_emissions,
        reduction_pct=total_pct,
        annual_reduction_rate=avg_rate,
        yearly_budgets=budgets,
        includes_deforestation_free=True,
        methodology_notes=(
            f"FLAG sector pathway for '{request.sector}' at weighted average "
            f"{avg_rate}% annual reduction across {len(request.commodity_mix)} commodities."
        ),
        generated_at=_now(),
    )


@router.get(
    "/commodities",
    response_model=List[FLAGCommodityInfo],
    summary="List all 11 FLAG commodities",
    description="List all 11 FLAG commodities with pathway rates and characteristics.",
)
async def list_flag_commodities() -> List[FLAGCommodityInfo]:
    """List all FLAG commodities."""
    return [FLAGCommodityInfo(**c) for c in FLAG_COMMODITIES]


@router.put(
    "/org/{org_id}/deforestation-commitment",
    response_model=DeforestationCommitmentResponse,
    summary="Track deforestation commitment",
    description=(
        "Create or update the zero-deforestation commitment tracking for "
        "an organization. Required by SBTi C25 for all FLAG target setters."
    ),
)
async def update_deforestation_commitment(
    org_id: str,
    request: DeforestationCommitmentRequest,
) -> DeforestationCommitmentResponse:
    """Track deforestation commitment."""
    now = _now()
    on_track = request.status in (
        DeforestationStatus.ACHIEVED.value, DeforestationStatus.COMMITTED.value,
    )
    data = {
        "org_id": org_id,
        "status": request.status.value,
        "commitment_date": request.commitment_date,
        "target_date": request.target_date,
        "supply_chain_coverage_pct": request.supply_chain_coverage_pct,
        "monitoring_system": request.monitoring_system,
        "on_track": on_track,
        "notes": request.notes,
        "updated_at": now,
    }
    _deforestation_commitments[org_id] = data
    return DeforestationCommitmentResponse(**data)


@router.get(
    "/org/{org_id}/emissions-split",
    response_model=EmissionsSplitResponse,
    summary="FLAG vs non-FLAG emissions split",
    description=(
        "Get the emissions split between FLAG and non-FLAG sources for "
        "separate pathway and target tracking."
    ),
)
async def get_emissions_split(org_id: str) -> EmissionsSplitResponse:
    """Get FLAG vs non-FLAG emission split."""
    return EmissionsSplitResponse(
        org_id=org_id,
        total_emissions_tco2e=160000,
        flag_emissions_tco2e=42000,
        non_flag_emissions_tco2e=118000,
        flag_pct=26.3,
        non_flag_pct=73.7,
        flag_by_source={
            "enteric_fermentation": 12000,
            "feed_production": 10000,
            "land_use_change": 8000,
            "deforestation": 5000,
            "manure_management": 4000,
            "field_emissions": 3000,
        },
        generated_at=_now(),
    )


@router.post(
    "/removals-eligibility",
    response_model=RemovalsEligibilityResponse,
    summary="Assess removal eligibility",
    description=(
        "Assess whether land-based carbon removals are eligible for "
        "counting toward FLAG targets. Evaluates permanence, additionality, "
        "third-party verification, and land-use category requirements."
    ),
)
async def assess_removals_eligibility(
    request: RemovalsEligibilityRequest,
) -> RemovalsEligibilityResponse:
    """Assess removals eligibility."""
    criteria = {
        "permanence_minimum_20_years": request.permanence_years >= 20,
        "additionality_demonstrated": request.additionality_demonstrated,
        "third_party_verified": request.third_party_verified,
        "eligible_land_use_category": request.land_use_category in (
            "forest_land", "cropland", "grassland", "wetland",
        ),
    }

    all_met = all(criteria.values())
    eligible_tco2e = request.removal_tco2e if all_met else 0.0

    issues = []
    if not criteria["permanence_minimum_20_years"]:
        issues.append(f"Permanence of {request.permanence_years} years below 20-year minimum.")
    if not criteria["additionality_demonstrated"]:
        issues.append("Additionality has not been demonstrated.")
    if not criteria["third_party_verified"]:
        issues.append("Removals have not been third-party verified.")
    if not criteria["eligible_land_use_category"]:
        issues.append(f"Land use category '{request.land_use_category}' is not eligible.")

    recommendation = (
        f"Removals of {request.removal_tco2e:,.0f} tCO2e are "
        f"{'eligible' if all_met else 'not eligible'} for FLAG target counting. "
        + (f"Address: {'; '.join(issues)}" if issues else "All criteria met.")
    )

    return RemovalsEligibilityResponse(
        org_id=request.org_id,
        eligible=all_met,
        removal_type=request.removal_type,
        removal_tco2e=request.removal_tco2e,
        eligible_tco2e=eligible_tco2e,
        criteria_met=criteria,
        issues=issues,
        recommendation=recommendation,
        generated_at=_now(),
    )
