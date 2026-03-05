"""
GL-TCFD-APP Physical Risk API

Manages physical risk assessment including asset registration with geo-location,
acute and chronic hazard exposure, portfolio-level risk aggregation, risk
mapping (GeoJSON), hazard projections, insurance impact, and supply chain
physical risk assessment.

Physical Risk Categories:
    Acute:   Cyclones/hurricanes, floods, wildfires, extreme heat events,
             droughts, storms
    Chronic: Sea level rise, temperature increase, water stress, permafrost
             thaw, precipitation changes

ISSB/IFRS S2 references: paragraphs 10-12 (physical risks).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/physical-risk", tags=["Physical Risk"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    MANUFACTURING_PLANT = "manufacturing_plant"
    WAREHOUSE = "warehouse"
    OFFICE = "office"
    DATA_CENTER = "data_center"
    RETAIL_STORE = "retail_store"
    AGRICULTURAL_LAND = "agricultural_land"
    INFRASTRUCTURE = "infrastructure"
    SUPPLY_CHAIN_NODE = "supply_chain_node"


class HazardType(str, Enum):
    # Acute
    CYCLONE = "cyclone"
    FLOOD_RIVERINE = "flood_riverine"
    FLOOD_COASTAL = "flood_coastal"
    WILDFIRE = "wildfire"
    EXTREME_HEAT = "extreme_heat"
    DROUGHT = "drought"
    STORM = "storm"
    # Chronic
    SEA_LEVEL_RISE = "sea_level_rise"
    TEMPERATURE_INCREASE = "temperature_increase"
    WATER_STRESS = "water_stress"
    PERMAFROST_THAW = "permafrost_thaw"
    PRECIPITATION_CHANGE = "precipitation_change"


class RiskRating(str, Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RegisterAssetRequest(BaseModel):
    """Request to register an asset for physical risk assessment."""
    asset_name: str = Field(..., min_length=1, max_length=300, description="Asset name")
    asset_type: AssetType = Field(..., description="Asset type")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code")
    city: Optional[str] = Field(None, max_length=200, description="City or region")
    asset_value_usd: float = Field(..., ge=0, description="Asset replacement value (USD)")
    annual_revenue_usd: Optional[float] = Field(None, ge=0, description="Annual revenue generated")
    employees: Optional[int] = Field(None, ge=0, description="Number of employees")
    criticality: str = Field("medium", description="Business criticality: low, medium, high, critical")

    class Config:
        json_schema_extra = {
            "example": {
                "asset_name": "Houston Manufacturing Complex",
                "asset_type": "manufacturing_plant",
                "latitude": 29.7604,
                "longitude": -95.3698,
                "country": "US",
                "city": "Houston, TX",
                "asset_value_usd": 150000000,
                "annual_revenue_usd": 80000000,
                "employees": 450,
                "criticality": "critical",
            }
        }


class UpdateAssetRequest(BaseModel):
    """Request to update an asset."""
    asset_name: Optional[str] = Field(None, max_length=300)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    asset_value_usd: Optional[float] = Field(None, ge=0)
    annual_revenue_usd: Optional[float] = Field(None, ge=0)
    employees: Optional[int] = Field(None, ge=0)
    criticality: Optional[str] = None


class AssessPhysicalRiskRequest(BaseModel):
    """Request to assess physical risk for an asset."""
    scenario: str = Field("rcp85", description="Climate scenario: rcp26, rcp45, rcp85")
    time_horizon: str = Field("2050", description="Assessment year: 2030, 2050, 2100")

    class Config:
        json_schema_extra = {
            "example": {"scenario": "rcp85", "time_horizon": "2050"}
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class AssetResponse(BaseModel):
    """A registered asset."""
    asset_id: str
    org_id: str
    asset_name: str
    asset_type: str
    latitude: float
    longitude: float
    country: str
    city: Optional[str]
    asset_value_usd: float
    annual_revenue_usd: Optional[float]
    employees: Optional[int]
    criticality: str
    created_at: datetime
    updated_at: datetime


class HazardExposureResponse(BaseModel):
    """Hazard exposure for a single hazard type."""
    hazard_type: str
    hazard_category: str
    exposure_rating: str
    exposure_score: float
    annualized_loss_usd: float
    description: str


class PhysicalRiskAssessmentResponse(BaseModel):
    """Physical risk assessment result."""
    assessment_id: str
    asset_id: str
    org_id: str
    asset_name: str
    scenario: str
    time_horizon: str
    overall_risk_rating: str
    overall_risk_score: float
    acute_risk_score: float
    chronic_risk_score: float
    hazard_exposures: List[HazardExposureResponse]
    total_annualized_loss_usd: float
    asset_value_at_risk_pct: float
    adaptation_recommendations: List[str]
    assessed_at: datetime


class PortfolioRiskResponse(BaseModel):
    """Portfolio-level physical risk assessment."""
    org_id: str
    total_assets: int
    total_portfolio_value_usd: float
    total_annualized_loss_usd: float
    portfolio_risk_score: float
    risk_distribution: Dict[str, int]
    top_risk_assets: List[Dict[str, Any]]
    by_hazard_type: Dict[str, float]
    by_geography: Dict[str, float]
    assessed_at: datetime


class RiskMapResponse(BaseModel):
    """GeoJSON risk map data."""
    org_id: str
    type: str
    features: List[Dict[str, Any]]
    generated_at: datetime


class HazardProjectionResponse(BaseModel):
    """Hazard projections for an asset."""
    asset_id: str
    asset_name: str
    projections: List[Dict[str, Any]]
    generated_at: datetime


class InsuranceImpactResponse(BaseModel):
    """Insurance cost impact assessment."""
    org_id: str
    current_premium_estimate_usd: float
    projected_premium_2030_usd: float
    projected_premium_2050_usd: float
    premium_increase_pct: float
    uninsurable_risk_usd: float
    high_risk_assets: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime


class SupplyChainPhysicalRiskResponse(BaseModel):
    """Supply chain physical risk assessment."""
    org_id: str
    supplier_locations_assessed: int
    high_risk_locations: int
    critical_supply_chain_risks: List[Dict[str, Any]]
    supply_chain_risk_score: float
    most_exposed_regions: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_assets: Dict[str, Dict[str, Any]] = {}
_assessments: Dict[str, Dict[str, Any]] = {}

HAZARD_BASE_SCORES = {
    "cyclone": 0.6, "flood_riverine": 0.5, "flood_coastal": 0.55,
    "wildfire": 0.4, "extreme_heat": 0.5, "drought": 0.35, "storm": 0.45,
    "sea_level_rise": 0.4, "temperature_increase": 0.5, "water_stress": 0.45,
    "permafrost_thaw": 0.2, "precipitation_change": 0.3,
}

SCENARIO_MULTIPLIER = {"rcp26": 0.6, "rcp45": 0.8, "rcp85": 1.2}
TIME_MULTIPLIER = {"2030": 0.7, "2050": 1.0, "2100": 1.5}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _rating_from_score(score: float) -> str:
    if score >= 0.8:
        return RiskRating.VERY_HIGH.value
    if score >= 0.6:
        return RiskRating.HIGH.value
    if score >= 0.4:
        return RiskRating.MEDIUM.value
    if score >= 0.2:
        return RiskRating.LOW.value
    return RiskRating.NEGLIGIBLE.value


def _assess_asset(asset: Dict[str, Any], scenario: str, time_horizon: str) -> Dict[str, Any]:
    """Run physical risk assessment for a single asset."""
    assessment_id = _generate_id("pra")
    scn_mult = SCENARIO_MULTIPLIER.get(scenario, 1.0)
    time_mult = TIME_MULTIPLIER.get(time_horizon, 1.0)

    # Latitude-based adjustment (higher risk in coastal/tropical zones)
    lat = abs(asset["latitude"])
    lat_factor = 1.0
    if lat < 25:
        lat_factor = 1.3  # Tropical
    elif lat > 55:
        lat_factor = 0.8  # High latitude

    hazard_exposures = []
    total_loss = 0.0
    acute_total = 0.0
    chronic_total = 0.0

    for hazard, base in HAZARD_BASE_SCORES.items():
        score = round(min(base * scn_mult * time_mult * lat_factor, 1.0), 3)
        is_acute = hazard in ("cyclone", "flood_riverine", "flood_coastal", "wildfire", "extreme_heat", "drought", "storm")
        annualized_loss = round(asset["asset_value_usd"] * score * 0.002, 2)
        total_loss += annualized_loss

        if is_acute:
            acute_total += score
        else:
            chronic_total += score

        hazard_exposures.append({
            "hazard_type": hazard,
            "hazard_category": "acute" if is_acute else "chronic",
            "exposure_rating": _rating_from_score(score),
            "exposure_score": score,
            "annualized_loss_usd": annualized_loss,
            "description": f"{'Acute' if is_acute else 'Chronic'} {hazard.replace('_', ' ')} exposure under {scenario} at {time_horizon}",
        })

    acute_count = sum(1 for h in HAZARD_BASE_SCORES if h in ("cyclone", "flood_riverine", "flood_coastal", "wildfire", "extreme_heat", "drought", "storm"))
    chronic_count = len(HAZARD_BASE_SCORES) - acute_count
    acute_avg = round(acute_total / acute_count, 3) if acute_count else 0
    chronic_avg = round(chronic_total / chronic_count, 3) if chronic_count else 0
    overall_score = round(acute_avg * 0.6 + chronic_avg * 0.4, 3)
    overall_rating = _rating_from_score(overall_score)
    val_at_risk = round(total_loss / asset["asset_value_usd"] * 100, 2) if asset["asset_value_usd"] > 0 else 0

    recommendations = []
    if overall_score >= 0.6:
        recommendations.append("Develop asset-specific adaptation plan")
        recommendations.append("Review insurance coverage adequacy")
    if acute_avg > 0.5:
        recommendations.append("Implement business continuity measures for acute hazards")
    if chronic_avg > 0.4:
        recommendations.append("Assess long-term asset viability under chronic climate shifts")
    recommendations.append("Monitor climate projections annually")

    result = {
        "assessment_id": assessment_id,
        "asset_id": asset["asset_id"],
        "org_id": asset["org_id"],
        "asset_name": asset["asset_name"],
        "scenario": scenario,
        "time_horizon": time_horizon,
        "overall_risk_rating": overall_rating,
        "overall_risk_score": overall_score,
        "acute_risk_score": acute_avg,
        "chronic_risk_score": chronic_avg,
        "hazard_exposures": hazard_exposures,
        "total_annualized_loss_usd": round(total_loss, 2),
        "asset_value_at_risk_pct": val_at_risk,
        "adaptation_recommendations": recommendations,
        "assessed_at": _now(),
    }
    _assessments[assessment_id] = result
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assets",
    response_model=AssetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register asset with location",
    description=(
        "Register a physical asset with geo-location for climate physical "
        "risk assessment.  Provide coordinates, asset type, value, and "
        "business criticality."
    ),
)
async def register_asset(
    org_id: str = Query(..., description="Organization ID"),
    request: RegisterAssetRequest = ...,
) -> AssetResponse:
    """Register a physical asset."""
    asset_id = _generate_id("ast")
    now = _now()
    asset = {
        "asset_id": asset_id,
        "org_id": org_id,
        "asset_name": request.asset_name,
        "asset_type": request.asset_type.value,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "country": request.country,
        "city": request.city,
        "asset_value_usd": request.asset_value_usd,
        "annual_revenue_usd": request.annual_revenue_usd,
        "employees": request.employees,
        "criticality": request.criticality,
        "created_at": now,
        "updated_at": now,
    }
    _assets[asset_id] = asset
    return AssetResponse(**asset)


@router.get(
    "/assets/{org_id}",
    response_model=List[AssetResponse],
    summary="List assets",
    description="Retrieve all registered assets for an organization.",
)
async def list_assets(
    org_id: str,
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    country: Optional[str] = Query(None, description="Filter by country"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[AssetResponse]:
    """List registered assets."""
    results = [a for a in _assets.values() if a["org_id"] == org_id]
    if asset_type:
        results = [a for a in results if a["asset_type"] == asset_type]
    if country:
        results = [a for a in results if a["country"] == country]
    results.sort(key=lambda a: a["asset_value_usd"], reverse=True)
    return [AssetResponse(**a) for a in results[:limit]]


@router.put(
    "/assets/{asset_id}",
    response_model=AssetResponse,
    summary="Update asset",
    description="Update a registered asset.",
)
async def update_asset(asset_id: str, request: UpdateAssetRequest) -> AssetResponse:
    """Update an asset."""
    asset = _assets.get(asset_id)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset {asset_id} not found")
    updates = request.model_dump(exclude_unset=True)
    asset.update(updates)
    asset["updated_at"] = _now()
    return AssetResponse(**asset)


@router.delete(
    "/assets/{asset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete asset",
    description="Remove a registered asset.",
)
async def delete_asset(asset_id: str) -> None:
    """Delete an asset."""
    if asset_id not in _assets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset {asset_id} not found")
    del _assets[asset_id]
    return None


@router.post(
    "/assess/{asset_id}",
    response_model=PhysicalRiskAssessmentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Assess physical risk for asset",
    description=(
        "Run physical risk assessment for a single asset under a specified "
        "climate scenario and time horizon.  Evaluates acute and chronic "
        "hazard exposures and calculates annualized loss estimates."
    ),
)
async def assess_physical_risk(
    asset_id: str,
    request: AssessPhysicalRiskRequest,
) -> PhysicalRiskAssessmentResponse:
    """Assess physical risk for a single asset."""
    asset = _assets.get(asset_id)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset {asset_id} not found")
    result = _assess_asset(asset, request.scenario, request.time_horizon)
    return PhysicalRiskAssessmentResponse(**result)


@router.post(
    "/assess-portfolio/{org_id}",
    response_model=PortfolioRiskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch assess all assets",
    description="Run physical risk assessment for all assets in an organization portfolio.",
)
async def assess_portfolio(
    org_id: str,
    scenario: str = Query("rcp85", description="Climate scenario"),
    time_horizon: str = Query("2050", description="Assessment year"),
) -> PortfolioRiskResponse:
    """Batch assess physical risk for all org assets."""
    org_assets = [a for a in _assets.values() if a["org_id"] == org_id]
    if not org_assets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No assets registered for org {org_id}")

    total_value = sum(a["asset_value_usd"] for a in org_assets)
    total_loss = 0.0
    risk_dist: Dict[str, int] = {"negligible": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0}
    by_hazard: Dict[str, float] = {}
    by_geo: Dict[str, float] = {}
    asset_results = []

    for asset in org_assets:
        result = _assess_asset(asset, scenario, time_horizon)
        total_loss += result["total_annualized_loss_usd"]
        risk_dist[result["overall_risk_rating"]] = risk_dist.get(result["overall_risk_rating"], 0) + 1
        for he in result["hazard_exposures"]:
            ht = he["hazard_type"]
            by_hazard[ht] = round(by_hazard.get(ht, 0) + he["annualized_loss_usd"], 2)
        country = asset.get("country", "Unknown")
        by_geo[country] = round(by_geo.get(country, 0) + result["total_annualized_loss_usd"], 2)
        asset_results.append({
            "asset_id": asset["asset_id"],
            "asset_name": asset["asset_name"],
            "risk_rating": result["overall_risk_rating"],
            "risk_score": result["overall_risk_score"],
            "annualized_loss_usd": result["total_annualized_loss_usd"],
        })

    asset_results.sort(key=lambda a: a["risk_score"], reverse=True)
    portfolio_score = round(total_loss / total_value * 100, 2) if total_value > 0 else 0

    return PortfolioRiskResponse(
        org_id=org_id,
        total_assets=len(org_assets),
        total_portfolio_value_usd=round(total_value, 2),
        total_annualized_loss_usd=round(total_loss, 2),
        portfolio_risk_score=portfolio_score,
        risk_distribution=risk_dist,
        top_risk_assets=asset_results[:10],
        by_hazard_type=by_hazard,
        by_geography=by_geo,
        assessed_at=_now(),
    )


@router.get(
    "/results/{org_id}",
    response_model=List[PhysicalRiskAssessmentResponse],
    summary="List physical risk results",
    description="List all physical risk assessment results for an organization.",
)
async def list_physical_risk_results(
    org_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[PhysicalRiskAssessmentResponse]:
    """List physical risk results."""
    results = [r for r in _assessments.values() if r["org_id"] == org_id]
    results.sort(key=lambda r: r["assessed_at"], reverse=True)
    return [PhysicalRiskAssessmentResponse(**r) for r in results[:limit]]


@router.get(
    "/results/{org_id}/{assessment_id}",
    response_model=PhysicalRiskAssessmentResponse,
    summary="Get assessment detail",
    description="Retrieve a specific physical risk assessment result.",
)
async def get_physical_risk_result(org_id: str, assessment_id: str) -> PhysicalRiskAssessmentResponse:
    """Get a physical risk assessment result."""
    result = _assessments.get(assessment_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Assessment {assessment_id} not found")
    if result["org_id"] != org_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Assessment does not belong to org {org_id}")
    return PhysicalRiskAssessmentResponse(**result)


@router.get(
    "/map/{org_id}",
    response_model=RiskMapResponse,
    summary="Get risk map GeoJSON data",
    description="Generate GeoJSON feature collection for mapping physical risk of all assets.",
)
async def get_risk_map(org_id: str) -> RiskMapResponse:
    """Generate GeoJSON risk map data."""
    org_assets = [a for a in _assets.values() if a["org_id"] == org_id]
    features = []
    for asset in org_assets:
        # Get latest assessment for this asset
        asset_assessments = [r for r in _assessments.values() if r["asset_id"] == asset["asset_id"]]
        latest = max(asset_assessments, key=lambda r: r["assessed_at"]) if asset_assessments else None

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [asset["longitude"], asset["latitude"]],
            },
            "properties": {
                "asset_id": asset["asset_id"],
                "asset_name": asset["asset_name"],
                "asset_type": asset["asset_type"],
                "country": asset["country"],
                "city": asset.get("city"),
                "asset_value_usd": asset["asset_value_usd"],
                "risk_rating": latest["overall_risk_rating"] if latest else "unassessed",
                "risk_score": latest["overall_risk_score"] if latest else 0,
                "annualized_loss_usd": latest["total_annualized_loss_usd"] if latest else 0,
            },
        }
        features.append(feature)

    return RiskMapResponse(
        org_id=org_id,
        type="FeatureCollection",
        features=features,
        generated_at=_now(),
    )


@router.get(
    "/hazards/{asset_id}",
    response_model=HazardProjectionResponse,
    summary="Get hazard projections",
    description="Get hazard projections for an asset across multiple time horizons and scenarios.",
)
async def get_hazard_projections(asset_id: str) -> HazardProjectionResponse:
    """Get hazard projections for an asset."""
    asset = _assets.get(asset_id)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset {asset_id} not found")

    projections = []
    for hazard in ["flood_riverine", "extreme_heat", "drought", "sea_level_rise", "wildfire"]:
        base_score = HAZARD_BASE_SCORES.get(hazard, 0.3)
        projection = {
            "hazard_type": hazard,
            "baseline_score": round(base_score * 0.5, 3),
            "rcp26_2030": round(base_score * 0.6 * 0.7, 3),
            "rcp26_2050": round(base_score * 0.6, 3),
            "rcp45_2030": round(base_score * 0.8 * 0.7, 3),
            "rcp45_2050": round(base_score * 0.8, 3),
            "rcp85_2030": round(base_score * 1.2 * 0.7, 3),
            "rcp85_2050": round(base_score * 1.2, 3),
            "rcp85_2100": round(min(base_score * 1.2 * 1.5, 1.0), 3),
        }
        projections.append(projection)

    return HazardProjectionResponse(
        asset_id=asset_id,
        asset_name=asset["asset_name"],
        projections=projections,
        generated_at=_now(),
    )


@router.get(
    "/insurance/{org_id}",
    response_model=InsuranceImpactResponse,
    summary="Insurance cost impact",
    description="Estimate the insurance cost impact of physical climate risk on the portfolio.",
)
async def get_insurance_impact(org_id: str) -> InsuranceImpactResponse:
    """Estimate insurance cost impact."""
    org_assets = [a for a in _assets.values() if a["org_id"] == org_id]
    total_value = sum(a["asset_value_usd"] for a in org_assets)

    # Simulated insurance modeling
    current_premium = round(total_value * 0.005, 2)
    projected_2030 = round(current_premium * 1.25, 2)
    projected_2050 = round(current_premium * 1.65, 2)
    increase_pct = round((projected_2050 / current_premium - 1) * 100, 1) if current_premium > 0 else 0

    high_risk = []
    for asset in org_assets:
        assessments = [r for r in _assessments.values() if r["asset_id"] == asset["asset_id"]]
        if assessments:
            latest = max(assessments, key=lambda r: r["assessed_at"])
            if latest["overall_risk_score"] >= 0.6:
                high_risk.append({
                    "asset_id": asset["asset_id"],
                    "asset_name": asset["asset_name"],
                    "risk_score": latest["overall_risk_score"],
                    "premium_surcharge_pct": round(latest["overall_risk_score"] * 50, 1),
                })

    uninsurable = round(total_value * 0.02, 2)  # Simplified

    return InsuranceImpactResponse(
        org_id=org_id,
        current_premium_estimate_usd=current_premium,
        projected_premium_2030_usd=projected_2030,
        projected_premium_2050_usd=projected_2050,
        premium_increase_pct=increase_pct,
        uninsurable_risk_usd=uninsurable,
        high_risk_assets=high_risk,
        recommendations=[
            "Review insurance coverage for high-risk assets annually",
            "Consider parametric insurance for acute weather events",
            "Implement adaptation measures to reduce premiums",
            "Explore captive insurance for concentrated portfolio risk",
        ],
        generated_at=_now(),
    )


@router.get(
    "/supply-chain/{org_id}",
    response_model=SupplyChainPhysicalRiskResponse,
    summary="Supply chain physical risk",
    description="Assess physical climate risk across the supply chain network.",
)
async def get_supply_chain_physical_risk(org_id: str) -> SupplyChainPhysicalRiskResponse:
    """Assess supply chain physical risk."""
    # Simulated supply chain risk data
    supply_chain_risks = [
        {
            "supplier": "Raw Material Supplier A",
            "location": "Southeast Asia",
            "hazard": "flood_riverine",
            "risk_rating": "high",
            "impact": "Potential 3-week supply disruption during monsoon season",
        },
        {
            "supplier": "Component Manufacturer B",
            "location": "Gulf Coast, US",
            "hazard": "cyclone",
            "risk_rating": "high",
            "impact": "Hurricane exposure threatens Q3 production capacity",
        },
        {
            "supplier": "Logistics Hub C",
            "location": "Netherlands",
            "hazard": "flood_coastal",
            "risk_rating": "medium",
            "impact": "Sea level rise may affect port operations by 2040",
        },
    ]

    exposed_regions = [
        {"region": "Southeast Asia", "risk_score": 0.72, "supplier_count": 12, "primary_hazards": ["flood_riverine", "cyclone"]},
        {"region": "Gulf Coast US", "risk_score": 0.65, "supplier_count": 5, "primary_hazards": ["cyclone", "flood_coastal"]},
        {"region": "Mediterranean Europe", "risk_score": 0.55, "supplier_count": 8, "primary_hazards": ["extreme_heat", "drought"]},
    ]

    return SupplyChainPhysicalRiskResponse(
        org_id=org_id,
        supplier_locations_assessed=45,
        high_risk_locations=8,
        critical_supply_chain_risks=supply_chain_risks,
        supply_chain_risk_score=0.58,
        most_exposed_regions=exposed_regions,
        recommendations=[
            "Diversify supplier base in high-risk regions",
            "Develop dual-sourcing strategy for critical components",
            "Require climate risk disclosure from Tier 1 suppliers",
            "Build strategic inventory buffers for high-risk materials",
            "Conduct annual supply chain physical risk reassessment",
        ],
        generated_at=_now(),
    )
