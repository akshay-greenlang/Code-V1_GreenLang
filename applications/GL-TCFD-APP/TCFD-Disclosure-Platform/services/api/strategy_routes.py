"""
GL-TCFD-APP Strategy API

TCFD Pillar 2 -- Strategy.  Manages climate-related risks and opportunities
identification, business model impact analysis, value chain mapping, and
strategy disclosure text generation.

TCFD Recommended Disclosures (Strategy):
    a) Climate-related risks and opportunities identified over short, medium,
       and long term
    b) Impact on business, strategy, and financial planning
    c) Resilience of strategy under different climate scenarios

ISSB/IFRS S2 references: paragraphs 8-22 (Strategy).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/strategy", tags=["Strategy"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskType(str, Enum):
    """TCFD climate risk types."""
    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"


class TimeHorizon(str, Enum):
    """Time horizons for climate risk/opportunity assessment."""
    SHORT_TERM = "short_term"       # 0-3 years
    MEDIUM_TERM = "medium_term"     # 3-10 years
    LONG_TERM = "long_term"         # 10-30+ years


class ImpactLevel(str, Enum):
    """Impact severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LikelihoodLevel(str, Enum):
    """Probability levels."""
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    LIKELY = "likely"
    VERY_LIKELY = "very_likely"
    ALMOST_CERTAIN = "almost_certain"


class OpportunityType(str, Enum):
    """TCFD opportunity categories."""
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateClimateRiskRequest(BaseModel):
    """Request to create a climate risk."""
    risk_name: str = Field(..., min_length=1, max_length=500, description="Risk name")
    risk_type: RiskType = Field(..., description="Type of climate risk")
    description: str = Field(..., min_length=1, max_length=5000, description="Detailed risk description")
    time_horizon: TimeHorizon = Field(..., description="Assessed time horizon")
    impact_level: ImpactLevel = Field(..., description="Impact severity")
    likelihood: LikelihoodLevel = Field(..., description="Probability of occurrence")
    financial_impact_usd: Optional[float] = Field(None, ge=0, description="Estimated financial impact in USD")
    affected_assets: Optional[List[str]] = Field(None, description="List of affected asset types or names")
    affected_geographies: Optional[List[str]] = Field(None, description="Affected geographies/regions")
    mitigation_strategy: Optional[str] = Field(None, max_length=3000, description="Current mitigation approach")
    sector: Optional[str] = Field(None, max_length=100, description="Industry sector")

    class Config:
        json_schema_extra = {
            "example": {
                "risk_name": "Carbon pricing regulation",
                "risk_type": "transition_policy",
                "description": "Introduction of carbon border adjustment mechanism increasing input costs by 8-15%",
                "time_horizon": "medium_term",
                "impact_level": "high",
                "likelihood": "very_likely",
                "financial_impact_usd": 12500000.0,
                "affected_assets": ["Manufacturing plants", "Supply chain"],
                "affected_geographies": ["EU", "UK"],
                "mitigation_strategy": "Accelerate scope 1 reduction program and evaluate low-carbon alternatives",
            }
        }


class UpdateClimateRiskRequest(BaseModel):
    """Request to update a climate risk."""
    risk_name: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    time_horizon: Optional[TimeHorizon] = None
    impact_level: Optional[ImpactLevel] = None
    likelihood: Optional[LikelihoodLevel] = None
    financial_impact_usd: Optional[float] = Field(None, ge=0)
    affected_assets: Optional[List[str]] = None
    affected_geographies: Optional[List[str]] = None
    mitigation_strategy: Optional[str] = Field(None, max_length=3000)


class CreateOpportunityRequest(BaseModel):
    """Request to create a climate opportunity."""
    opportunity_name: str = Field(..., min_length=1, max_length=500, description="Opportunity name")
    opportunity_type: OpportunityType = Field(..., description="TCFD opportunity category")
    description: str = Field(..., min_length=1, max_length=5000, description="Detailed description")
    time_horizon: TimeHorizon = Field(..., description="Realization time horizon")
    revenue_potential_usd: Optional[float] = Field(None, ge=0, description="Revenue potential (USD)")
    cost_savings_usd: Optional[float] = Field(None, ge=0, description="Cost savings potential (USD)")
    investment_required_usd: Optional[float] = Field(None, ge=0, description="Required investment (USD)")
    strategic_fit: Optional[str] = Field(None, max_length=2000, description="Alignment with business strategy")

    class Config:
        json_schema_extra = {
            "example": {
                "opportunity_name": "Green product line expansion",
                "opportunity_type": "products_services",
                "description": "Develop low-carbon product variants to capture growing sustainable procurement demand",
                "time_horizon": "medium_term",
                "revenue_potential_usd": 25000000.0,
                "cost_savings_usd": 3000000.0,
                "investment_required_usd": 8000000.0,
                "strategic_fit": "Aligned with 2030 sustainability strategy and R&D roadmap",
            }
        }


class UpdateOpportunityRequest(BaseModel):
    """Request to update a climate opportunity."""
    opportunity_name: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    time_horizon: Optional[TimeHorizon] = None
    revenue_potential_usd: Optional[float] = Field(None, ge=0)
    cost_savings_usd: Optional[float] = Field(None, ge=0)
    investment_required_usd: Optional[float] = Field(None, ge=0)
    strategic_fit: Optional[str] = Field(None, max_length=2000)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ClimateRiskResponse(BaseModel):
    """A climate risk entry."""
    risk_id: str
    org_id: str
    risk_name: str
    risk_type: str
    description: str
    time_horizon: str
    impact_level: str
    likelihood: str
    risk_score: float
    financial_impact_usd: Optional[float]
    affected_assets: Optional[List[str]]
    affected_geographies: Optional[List[str]]
    mitigation_strategy: Optional[str]
    sector: Optional[str]
    created_at: datetime
    updated_at: datetime


class OpportunityResponse(BaseModel):
    """A climate opportunity entry."""
    opportunity_id: str
    org_id: str
    opportunity_name: str
    opportunity_type: str
    description: str
    time_horizon: str
    revenue_potential_usd: Optional[float]
    cost_savings_usd: Optional[float]
    investment_required_usd: Optional[float]
    net_value_usd: Optional[float]
    strategic_fit: Optional[str]
    created_at: datetime
    updated_at: datetime


class BusinessImpactResponse(BaseModel):
    """Business model impact analysis."""
    org_id: str
    total_risk_exposure_usd: float
    total_opportunity_value_usd: float
    net_climate_impact_usd: float
    risk_count: int
    opportunity_count: int
    by_time_horizon: Dict[str, Dict[str, float]]
    by_risk_type: Dict[str, float]
    by_opportunity_type: Dict[str, float]
    top_risks: List[Dict[str, Any]]
    top_opportunities: List[Dict[str, Any]]
    strategic_implications: List[str]
    generated_at: datetime


class ValueChainImpactResponse(BaseModel):
    """Value chain impact mapping."""
    org_id: str
    upstream_risks: List[Dict[str, Any]]
    direct_operations_risks: List[Dict[str, Any]]
    downstream_risks: List[Dict[str, Any]]
    upstream_opportunities: List[Dict[str, Any]]
    direct_operations_opportunities: List[Dict[str, Any]]
    downstream_opportunities: List[Dict[str, Any]]
    value_chain_exposure_usd: float
    most_exposed_segment: str
    generated_at: datetime


class StrategyDisclosureResponse(BaseModel):
    """Generated strategy disclosure text."""
    org_id: str
    reporting_year: int
    pillar: str
    disclosure_a: str
    disclosure_b: str
    disclosure_c: str
    word_count: int
    compliance_score: float
    issb_references: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_risks: Dict[str, Dict[str, Any]] = {}
_opportunities: Dict[str, Dict[str, Any]] = {}

IMPACT_SCORES = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
LIKELIHOOD_SCORES = {"unlikely": 1, "possible": 2, "likely": 3, "very_likely": 4, "almost_certain": 5}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _calc_risk_score(impact: str, likelihood: str) -> float:
    """Calculate risk score as impact * likelihood, normalized 0-100."""
    i = IMPACT_SCORES.get(impact, 1)
    l = LIKELIHOOD_SCORES.get(likelihood, 1)
    return round(i * l / 20.0 * 100.0, 1)


# ---------------------------------------------------------------------------
# Endpoints -- Climate Risks
# ---------------------------------------------------------------------------

@router.post(
    "/risks",
    response_model=ClimateRiskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create climate risk",
    description=(
        "Identify and register a climate-related risk per TCFD Strategy "
        "Recommended Disclosure (a).  Specify risk type (physical/transition), "
        "time horizon, impact severity, and likelihood."
    ),
)
async def create_climate_risk(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateClimateRiskRequest = ...,
) -> ClimateRiskResponse:
    """Create a new climate risk."""
    risk_id = _generate_id("crisk")
    now = _now()
    risk = {
        "risk_id": risk_id,
        "org_id": org_id,
        "risk_name": request.risk_name,
        "risk_type": request.risk_type.value,
        "description": request.description,
        "time_horizon": request.time_horizon.value,
        "impact_level": request.impact_level.value,
        "likelihood": request.likelihood.value,
        "risk_score": _calc_risk_score(request.impact_level.value, request.likelihood.value),
        "financial_impact_usd": request.financial_impact_usd,
        "affected_assets": request.affected_assets,
        "affected_geographies": request.affected_geographies,
        "mitigation_strategy": request.mitigation_strategy,
        "sector": request.sector,
        "created_at": now,
        "updated_at": now,
    }
    _risks[risk_id] = risk
    return ClimateRiskResponse(**risk)


@router.get(
    "/risks/{org_id}",
    response_model=List[ClimateRiskResponse],
    summary="List climate risks",
    description="Retrieve all climate risks for an organization, with optional filters.",
)
async def list_climate_risks(
    org_id: str,
    risk_type: Optional[str] = Query(None, description="Filter by risk type"),
    time_horizon: Optional[str] = Query(None, description="Filter by time horizon"),
    impact_level: Optional[str] = Query(None, description="Filter by impact level"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[ClimateRiskResponse]:
    """List climate risks with filters."""
    results = [r for r in _risks.values() if r["org_id"] == org_id]
    if risk_type:
        results = [r for r in results if r["risk_type"] == risk_type]
    if time_horizon:
        results = [r for r in results if r["time_horizon"] == time_horizon]
    if impact_level:
        results = [r for r in results if r["impact_level"] == impact_level]
    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return [ClimateRiskResponse(**r) for r in results[:limit]]


@router.get(
    "/risks/{org_id}/{risk_id}",
    response_model=ClimateRiskResponse,
    summary="Get climate risk detail",
    description="Retrieve a single climate risk by ID.",
)
async def get_climate_risk(org_id: str, risk_id: str) -> ClimateRiskResponse:
    """Retrieve a climate risk by ID."""
    risk = _risks.get(risk_id)
    if not risk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Risk {risk_id} not found")
    if risk["org_id"] != org_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Risk {risk_id} does not belong to org {org_id}")
    return ClimateRiskResponse(**risk)


@router.put(
    "/risks/{risk_id}",
    response_model=ClimateRiskResponse,
    summary="Update climate risk",
    description="Update an existing climate risk and recalculate risk score.",
)
async def update_climate_risk(risk_id: str, request: UpdateClimateRiskRequest) -> ClimateRiskResponse:
    """Update an existing climate risk."""
    risk = _risks.get(risk_id)
    if not risk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Risk {risk_id} not found")
    updates = request.model_dump(exclude_unset=True)
    for key in ("time_horizon", "impact_level", "likelihood"):
        if key in updates and hasattr(updates[key], "value"):
            updates[key] = updates[key].value
    risk.update(updates)
    risk["risk_score"] = _calc_risk_score(risk["impact_level"], risk["likelihood"])
    risk["updated_at"] = _now()
    return ClimateRiskResponse(**risk)


@router.delete(
    "/risks/{risk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete climate risk",
    description="Remove a climate risk from the register.",
)
async def delete_climate_risk(risk_id: str) -> None:
    """Delete a climate risk."""
    if risk_id not in _risks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Risk {risk_id} not found")
    del _risks[risk_id]
    return None


# ---------------------------------------------------------------------------
# Endpoints -- Climate Opportunities
# ---------------------------------------------------------------------------

@router.post(
    "/opportunities",
    response_model=OpportunityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create climate opportunity",
    description=(
        "Register a climate-related opportunity per TCFD Strategy Recommended "
        "Disclosure (a).  Specify opportunity type, time horizon, revenue "
        "potential, cost savings, and investment required."
    ),
)
async def create_opportunity(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateOpportunityRequest = ...,
) -> OpportunityResponse:
    """Create a new climate opportunity."""
    opp_id = _generate_id("copp")
    now = _now()
    revenue = request.revenue_potential_usd or 0.0
    savings = request.cost_savings_usd or 0.0
    investment = request.investment_required_usd or 0.0
    net_value = round(revenue + savings - investment, 2)
    opp = {
        "opportunity_id": opp_id,
        "org_id": org_id,
        "opportunity_name": request.opportunity_name,
        "opportunity_type": request.opportunity_type.value,
        "description": request.description,
        "time_horizon": request.time_horizon.value,
        "revenue_potential_usd": request.revenue_potential_usd,
        "cost_savings_usd": request.cost_savings_usd,
        "investment_required_usd": request.investment_required_usd,
        "net_value_usd": net_value,
        "strategic_fit": request.strategic_fit,
        "created_at": now,
        "updated_at": now,
    }
    _opportunities[opp_id] = opp
    return OpportunityResponse(**opp)


@router.get(
    "/opportunities/{org_id}",
    response_model=List[OpportunityResponse],
    summary="List climate opportunities",
    description="Retrieve all climate opportunities for an organization with optional filters.",
)
async def list_opportunities(
    org_id: str,
    opportunity_type: Optional[str] = Query(None, description="Filter by opportunity type"),
    time_horizon: Optional[str] = Query(None, description="Filter by time horizon"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[OpportunityResponse]:
    """List climate opportunities."""
    results = [o for o in _opportunities.values() if o["org_id"] == org_id]
    if opportunity_type:
        results = [o for o in results if o["opportunity_type"] == opportunity_type]
    if time_horizon:
        results = [o for o in results if o["time_horizon"] == time_horizon]
    results.sort(key=lambda o: o.get("net_value_usd", 0) or 0, reverse=True)
    return [OpportunityResponse(**o) for o in results[:limit]]


@router.get(
    "/opportunities/{org_id}/{opp_id}",
    response_model=OpportunityResponse,
    summary="Get opportunity detail",
    description="Retrieve a single climate opportunity by ID.",
)
async def get_opportunity(org_id: str, opp_id: str) -> OpportunityResponse:
    """Retrieve a climate opportunity by ID."""
    opp = _opportunities.get(opp_id)
    if not opp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")
    if opp["org_id"] != org_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Opportunity {opp_id} does not belong to org {org_id}")
    return OpportunityResponse(**opp)


@router.put(
    "/opportunities/{opp_id}",
    response_model=OpportunityResponse,
    summary="Update climate opportunity",
    description="Update an existing climate opportunity.",
)
async def update_opportunity(opp_id: str, request: UpdateOpportunityRequest) -> OpportunityResponse:
    """Update a climate opportunity."""
    opp = _opportunities.get(opp_id)
    if not opp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")
    updates = request.model_dump(exclude_unset=True)
    if "time_horizon" in updates and hasattr(updates["time_horizon"], "value"):
        updates["time_horizon"] = updates["time_horizon"].value
    opp.update(updates)
    revenue = opp.get("revenue_potential_usd") or 0.0
    savings = opp.get("cost_savings_usd") or 0.0
    investment = opp.get("investment_required_usd") or 0.0
    opp["net_value_usd"] = round(revenue + savings - investment, 2)
    opp["updated_at"] = _now()
    return OpportunityResponse(**opp)


@router.delete(
    "/opportunities/{opp_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete climate opportunity",
    description="Remove a climate opportunity.",
)
async def delete_opportunity(opp_id: str) -> None:
    """Delete a climate opportunity."""
    if opp_id not in _opportunities:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")
    del _opportunities[opp_id]
    return None


# ---------------------------------------------------------------------------
# Endpoints -- Business Impact & Value Chain
# ---------------------------------------------------------------------------

@router.get(
    "/business-impact/{org_id}",
    response_model=BusinessImpactResponse,
    summary="Business model impact analysis",
    description=(
        "Analyze the aggregate impact of climate risks and opportunities on "
        "the organization's business model, strategy, and financial planning. "
        "Corresponds to TCFD Strategy Recommended Disclosure (b)."
    ),
)
async def get_business_impact(org_id: str) -> BusinessImpactResponse:
    """Analyze business model impact from climate risks and opportunities."""
    org_risks = [r for r in _risks.values() if r["org_id"] == org_id]
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]

    total_risk = sum(r.get("financial_impact_usd") or 0 for r in org_risks)
    total_opp = sum((o.get("revenue_potential_usd") or 0) + (o.get("cost_savings_usd") or 0) for o in org_opps)
    net = round(total_opp - total_risk, 2)

    by_time = {}
    for th in ["short_term", "medium_term", "long_term"]:
        th_risks = sum(r.get("financial_impact_usd") or 0 for r in org_risks if r["time_horizon"] == th)
        th_opps = sum((o.get("revenue_potential_usd") or 0) + (o.get("cost_savings_usd") or 0) for o in org_opps if o["time_horizon"] == th)
        by_time[th] = {"risk_exposure_usd": round(th_risks, 2), "opportunity_value_usd": round(th_opps, 2)}

    by_risk_type: Dict[str, float] = {}
    for r in org_risks:
        rt = r["risk_type"]
        by_risk_type[rt] = round(by_risk_type.get(rt, 0) + (r.get("financial_impact_usd") or 0), 2)

    by_opp_type: Dict[str, float] = {}
    for o in org_opps:
        ot = o["opportunity_type"]
        val = (o.get("revenue_potential_usd") or 0) + (o.get("cost_savings_usd") or 0)
        by_opp_type[ot] = round(by_opp_type.get(ot, 0) + val, 2)

    top_risks = sorted(org_risks, key=lambda r: r.get("risk_score", 0), reverse=True)[:5]
    top_opps = sorted(org_opps, key=lambda o: o.get("net_value_usd", 0) or 0, reverse=True)[:5]

    implications = []
    if total_risk > total_opp:
        implications.append("Net climate impact is negative; risk mitigation should be prioritized")
    else:
        implications.append("Net climate impact is positive; opportunity capture is strategically advantageous")
    if any(r["time_horizon"] == "short_term" and r["impact_level"] in ("high", "very_high") for r in org_risks):
        implications.append("High-impact short-term risks require immediate management attention")
    if by_risk_type.get("transition_policy", 0) > total_risk * 0.4:
        implications.append("Regulatory/policy risk dominates exposure; regulatory engagement recommended")

    return BusinessImpactResponse(
        org_id=org_id,
        total_risk_exposure_usd=round(total_risk, 2),
        total_opportunity_value_usd=round(total_opp, 2),
        net_climate_impact_usd=net,
        risk_count=len(org_risks),
        opportunity_count=len(org_opps),
        by_time_horizon=by_time,
        by_risk_type=by_risk_type,
        by_opportunity_type=by_opp_type,
        top_risks=[{"risk_id": r["risk_id"], "name": r["risk_name"], "score": r["risk_score"]} for r in top_risks],
        top_opportunities=[{"opp_id": o["opportunity_id"], "name": o["opportunity_name"], "net_value": o.get("net_value_usd")} for o in top_opps],
        strategic_implications=implications,
        generated_at=_now(),
    )


@router.get(
    "/value-chain/{org_id}",
    response_model=ValueChainImpactResponse,
    summary="Value chain impact mapping",
    description=(
        "Map climate-related risks and opportunities across the value chain: "
        "upstream (supply chain), direct operations, and downstream (customers/markets)."
    ),
)
async def get_value_chain_impact(org_id: str) -> ValueChainImpactResponse:
    """Map climate impacts across the value chain."""
    org_risks = [r for r in _risks.values() if r["org_id"] == org_id]
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]

    # Classify risks by value chain segment based on type
    upstream_risk_types = {"transition_market"}
    downstream_risk_types = {"transition_reputation"}
    direct_risk_types = {"physical_acute", "physical_chronic", "transition_policy", "transition_technology"}

    upstream_risks = [
        {"risk_id": r["risk_id"], "name": r["risk_name"], "type": r["risk_type"],
         "impact": r.get("financial_impact_usd")}
        for r in org_risks if r["risk_type"] in upstream_risk_types
    ]
    direct_risks = [
        {"risk_id": r["risk_id"], "name": r["risk_name"], "type": r["risk_type"],
         "impact": r.get("financial_impact_usd")}
        for r in org_risks if r["risk_type"] in direct_risk_types
    ]
    downstream_risks = [
        {"risk_id": r["risk_id"], "name": r["risk_name"], "type": r["risk_type"],
         "impact": r.get("financial_impact_usd")}
        for r in org_risks if r["risk_type"] in downstream_risk_types
    ]

    # Classify opportunities
    upstream_opp_types = {"resource_efficiency"}
    downstream_opp_types = {"products_services", "markets"}
    direct_opp_types = {"energy_source", "resilience"}

    upstream_opps = [
        {"opp_id": o["opportunity_id"], "name": o["opportunity_name"], "type": o["opportunity_type"],
         "value": o.get("net_value_usd")}
        for o in org_opps if o["opportunity_type"] in upstream_opp_types
    ]
    direct_opps = [
        {"opp_id": o["opportunity_id"], "name": o["opportunity_name"], "type": o["opportunity_type"],
         "value": o.get("net_value_usd")}
        for o in org_opps if o["opportunity_type"] in direct_opp_types
    ]
    downstream_opps = [
        {"opp_id": o["opportunity_id"], "name": o["opportunity_name"], "type": o["opportunity_type"],
         "value": o.get("net_value_usd")}
        for o in org_opps if o["opportunity_type"] in downstream_opp_types
    ]

    total_exposure = sum(r.get("financial_impact_usd") or 0 for r in org_risks)
    segments = {
        "upstream": sum(r.get("impact") or 0 for r in upstream_risks),
        "direct_operations": sum(r.get("impact") or 0 for r in direct_risks),
        "downstream": sum(r.get("impact") or 0 for r in downstream_risks),
    }
    most_exposed = max(segments, key=segments.get) if segments else "direct_operations"

    return ValueChainImpactResponse(
        org_id=org_id,
        upstream_risks=upstream_risks,
        direct_operations_risks=direct_risks,
        downstream_risks=downstream_risks,
        upstream_opportunities=upstream_opps,
        direct_operations_opportunities=direct_opps,
        downstream_opportunities=downstream_opps,
        value_chain_exposure_usd=round(total_exposure, 2),
        most_exposed_segment=most_exposed,
        generated_at=_now(),
    )


@router.get(
    "/disclosure/{org_id}/{year}",
    response_model=StrategyDisclosureResponse,
    summary="Generate strategy disclosure text",
    description=(
        "Generate TCFD-aligned strategy disclosure text covering all three "
        "recommended disclosures: (a) risks and opportunities, (b) business "
        "impact, and (c) strategy resilience."
    ),
)
async def generate_strategy_disclosure(org_id: str, year: int) -> StrategyDisclosureResponse:
    """Generate strategy disclosure text."""
    org_risks = [r for r in _risks.values() if r["org_id"] == org_id]
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]

    risk_count = len(org_risks)
    opp_count = len(org_opps)
    phys_count = sum(1 for r in org_risks if r["risk_type"].startswith("physical"))
    trans_count = sum(1 for r in org_risks if r["risk_type"].startswith("transition"))

    disclosure_a = (
        f"The organization has identified {risk_count} climate-related risks "
        f"({phys_count} physical, {trans_count} transition) and {opp_count} "
        f"climate-related opportunities across short, medium, and long-term "
        f"time horizons. Physical risks include acute events (extreme weather, "
        f"flooding) and chronic shifts (temperature rise, water stress). "
        f"Transition risks span policy/regulatory changes, technology shifts, "
        f"market dynamics, and reputational factors."
    )

    total_risk = sum(r.get("financial_impact_usd") or 0 for r in org_risks)
    total_opp = sum((o.get("revenue_potential_usd") or 0) + (o.get("cost_savings_usd") or 0) for o in org_opps)

    disclosure_b = (
        f"The total estimated financial exposure to climate-related risks is "
        f"${total_risk:,.0f}, while the identified opportunity value is "
        f"${total_opp:,.0f}. Climate considerations have been integrated into "
        f"the organization's strategic planning process, capital allocation "
        f"decisions, and product development roadmap. The organization is "
        f"pursuing a portfolio approach to manage transition risk while "
        f"capturing growth from low-carbon products and services."
    )

    disclosure_c = (
        f"The organization has assessed the resilience of its strategy under "
        f"multiple climate scenarios including a 1.5C pathway (IEA NZE 2050), "
        f"a 2C pathway (IEA APS), and a >3C pathway (NGFS Current Policies). "
        f"Under all scenarios, the organization maintains business viability "
        f"through adaptation measures and strategic pivots. The scenario "
        f"analysis informs capital expenditure planning and R&D priorities."
    )

    word_count = sum(len(d.split()) for d in [disclosure_a, disclosure_b, disclosure_c])
    risk_completeness = min(risk_count * 10, 40)
    opp_completeness = min(opp_count * 10, 30)
    compliance_score = min(risk_completeness + opp_completeness + 30.0, 100.0)

    return StrategyDisclosureResponse(
        org_id=org_id,
        reporting_year=year,
        pillar="strategy",
        disclosure_a=disclosure_a,
        disclosure_b=disclosure_b,
        disclosure_c=disclosure_c,
        word_count=word_count,
        compliance_score=round(compliance_score, 1),
        issb_references=[
            "IFRS S2 para 8-9", "IFRS S2 para 10-12", "IFRS S2 para 13-15",
            "IFRS S2 para 16-21", "IFRS S2 para 22",
        ],
        generated_at=_now(),
    )
