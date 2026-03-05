"""
GL-TCFD-APP Opportunity API

Manages climate-related opportunity pipeline including identification, sizing,
ROI analysis, cost savings estimation, priority matrix generation, and green
financing assessment.

TCFD Opportunity Categories:
    - Resource Efficiency: Energy, water, waste, materials optimization
    - Energy Source: Renewable energy, distributed generation, fuel switching
    - Products & Services: Low-carbon products, climate solutions, labeling
    - Markets: New market access, green procurement, carbon credits
    - Resilience: Adaptation solutions, supply chain diversification, insurance

ISSB/IFRS S2 references: paragraphs 8-9 (climate opportunities).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/opportunities", tags=["Opportunities"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OpportunityCategory(str, Enum):
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"


class OpportunityStatus(str, Enum):
    IDENTIFIED = "identified"
    UNDER_EVALUATION = "under_evaluation"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    REALIZED = "realized"
    DEFERRED = "deferred"


class TimeHorizon(str, Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateOpportunityRequest(BaseModel):
    """Request to create a climate opportunity."""
    name: str = Field(..., min_length=1, max_length=500, description="Opportunity name")
    category: OpportunityCategory = Field(..., description="Opportunity category")
    description: str = Field(..., min_length=1, max_length=5000, description="Detailed description")
    time_horizon: TimeHorizon = Field(..., description="Realization time horizon")
    status: OpportunityStatus = Field(OpportunityStatus.IDENTIFIED, description="Current status")
    revenue_potential_usd: Optional[float] = Field(None, ge=0, description="Revenue potential")
    cost_savings_usd: Optional[float] = Field(None, ge=0, description="Annual cost savings")
    investment_required_usd: Optional[float] = Field(None, ge=0, description="Investment required")
    abatement_potential_tco2e: Optional[float] = Field(None, ge=0, description="Emission reduction potential")
    probability_pct: float = Field(50.0, ge=0, le=100, description="Probability of realization")
    strategic_fit_score: float = Field(50.0, ge=0, le=100, description="Alignment with strategy (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "On-site solar PV installation",
                "category": "energy_source",
                "description": "Install 5 MW rooftop solar across 3 manufacturing facilities",
                "time_horizon": "short_term",
                "status": "approved",
                "revenue_potential_usd": 0,
                "cost_savings_usd": 1200000,
                "investment_required_usd": 8000000,
                "abatement_potential_tco2e": 3500,
                "probability_pct": 85,
                "strategic_fit_score": 90,
            }
        }


class UpdateOpportunityRequest(BaseModel):
    """Request to update a climate opportunity."""
    name: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    time_horizon: Optional[TimeHorizon] = None
    status: Optional[OpportunityStatus] = None
    revenue_potential_usd: Optional[float] = Field(None, ge=0)
    cost_savings_usd: Optional[float] = Field(None, ge=0)
    investment_required_usd: Optional[float] = Field(None, ge=0)
    abatement_potential_tco2e: Optional[float] = Field(None, ge=0)
    probability_pct: Optional[float] = Field(None, ge=0, le=100)
    strategic_fit_score: Optional[float] = Field(None, ge=0, le=100)


class SizeOpportunityRequest(BaseModel):
    """Request to size a revenue/savings opportunity."""
    market_size_usd: float = Field(0.0, ge=0, description="Addressable market size")
    market_share_pct: float = Field(5.0, ge=0, le=100, description="Target market share")
    growth_rate_pct: float = Field(10.0, ge=-100, le=500, description="Annual growth rate")
    years: int = Field(10, ge=1, le=30, description="Projection years")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class OpportunityResponse(BaseModel):
    """A climate opportunity."""
    opportunity_id: str
    org_id: str
    name: str
    category: str
    description: str
    time_horizon: str
    status: str
    revenue_potential_usd: Optional[float]
    cost_savings_usd: Optional[float]
    investment_required_usd: Optional[float]
    net_value_usd: float
    abatement_potential_tco2e: Optional[float]
    probability_pct: float
    strategic_fit_score: float
    expected_value_usd: float
    created_at: datetime
    updated_at: datetime


class PipelineResponse(BaseModel):
    """Opportunity pipeline summary."""
    org_id: str
    total_opportunities: int
    total_potential_value_usd: float
    total_expected_value_usd: float
    total_investment_required_usd: float
    total_abatement_tco2e: float
    by_category: Dict[str, Dict[str, Any]]
    by_status: Dict[str, int]
    by_time_horizon: Dict[str, Dict[str, Any]]
    generated_at: datetime


class SizingResponse(BaseModel):
    """Opportunity sizing result."""
    opportunity_id: str
    revenue_projection: Dict[str, float]
    total_revenue_usd: float
    npv_usd: float
    payback_years: Optional[float]
    generated_at: datetime


class SavingsEstimateResponse(BaseModel):
    """Cost savings estimate."""
    org_id: str
    total_annual_savings_usd: float
    total_5yr_savings_usd: float
    by_category: Dict[str, float]
    by_opportunity: List[Dict[str, Any]]
    generated_at: datetime


class ROIResponse(BaseModel):
    """ROI analysis for an opportunity."""
    opportunity_id: str
    investment_usd: float
    annual_return_usd: float
    roi_pct: float
    payback_years: float
    npv_usd: float
    irr_pct: float
    generated_at: datetime


class PriorityMatrixResponse(BaseModel):
    """Priority matrix (impact vs feasibility)."""
    org_id: str
    quadrants: Dict[str, List[Dict[str, Any]]]
    recommended_sequence: List[Dict[str, Any]]
    generated_at: datetime


class GreenFinancingResponse(BaseModel):
    """Green financing assessment."""
    org_id: str
    eligible_investment_usd: float
    green_bond_eligible: bool
    sustainability_linked_loan_eligible: bool
    eligible_instruments: List[Dict[str, Any]]
    estimated_savings_bps: int
    taxonomy_alignment_pct: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_opportunities: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _calc_net_value(opp: Dict[str, Any]) -> float:
    rev = opp.get("revenue_potential_usd") or 0
    sav = opp.get("cost_savings_usd") or 0
    inv = opp.get("investment_required_usd") or 0
    return round(rev + sav - inv, 2)


def _calc_expected_value(opp: Dict[str, Any]) -> float:
    net = _calc_net_value(opp)
    prob = opp.get("probability_pct", 50) / 100
    return round(net * prob, 2)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=OpportunityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create opportunity",
    description="Register a climate-related opportunity with financial and abatement estimates.",
)
async def create_opportunity(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateOpportunityRequest = ...,
) -> OpportunityResponse:
    """Create a climate opportunity."""
    opp_id = _generate_id("opp")
    now = _now()
    opp = {
        "opportunity_id": opp_id,
        "org_id": org_id,
        "name": request.name,
        "category": request.category.value,
        "description": request.description,
        "time_horizon": request.time_horizon.value,
        "status": request.status.value,
        "revenue_potential_usd": request.revenue_potential_usd,
        "cost_savings_usd": request.cost_savings_usd,
        "investment_required_usd": request.investment_required_usd,
        "abatement_potential_tco2e": request.abatement_potential_tco2e,
        "probability_pct": request.probability_pct,
        "strategic_fit_score": request.strategic_fit_score,
        "created_at": now,
        "updated_at": now,
    }
    opp["net_value_usd"] = _calc_net_value(opp)
    opp["expected_value_usd"] = _calc_expected_value(opp)
    _opportunities[opp_id] = opp
    return OpportunityResponse(**opp)


@router.get(
    "/{org_id}",
    response_model=List[OpportunityResponse],
    summary="List opportunities",
    description="Retrieve all climate opportunities for an organization.",
)
async def list_opportunities(
    org_id: str,
    category: Optional[str] = Query(None, description="Filter by category"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[OpportunityResponse]:
    """List climate opportunities."""
    results = [o for o in _opportunities.values() if o["org_id"] == org_id]
    if category:
        results = [o for o in results if o["category"] == category]
    if status_filter:
        results = [o for o in results if o["status"] == status_filter]
    results.sort(key=lambda o: o["expected_value_usd"], reverse=True)
    return [OpportunityResponse(**o) for o in results[:limit]]


@router.get(
    "/{org_id}/{opp_id}",
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
    "/{opp_id}",
    response_model=OpportunityResponse,
    summary="Update opportunity",
    description="Update an existing climate opportunity.",
)
async def update_opportunity(opp_id: str, request: UpdateOpportunityRequest) -> OpportunityResponse:
    """Update a climate opportunity."""
    opp = _opportunities.get(opp_id)
    if not opp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")
    updates = request.model_dump(exclude_unset=True)
    for key in ("time_horizon", "status"):
        if key in updates and hasattr(updates[key], "value"):
            updates[key] = updates[key].value
    opp.update(updates)
    opp["net_value_usd"] = _calc_net_value(opp)
    opp["expected_value_usd"] = _calc_expected_value(opp)
    opp["updated_at"] = _now()
    return OpportunityResponse(**opp)


@router.delete(
    "/{opp_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete opportunity",
    description="Remove a climate opportunity.",
)
async def delete_opportunity(opp_id: str) -> None:
    """Delete a climate opportunity."""
    if opp_id not in _opportunities:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")
    del _opportunities[opp_id]
    return None


@router.get(
    "/pipeline/{org_id}",
    response_model=PipelineResponse,
    summary="Pipeline summary",
    description="Get a summary of the opportunity pipeline including totals by category, status, and time horizon.",
)
async def get_pipeline(org_id: str) -> PipelineResponse:
    """Get opportunity pipeline summary."""
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]
    total_value = sum(o["net_value_usd"] for o in org_opps)
    total_ev = sum(o["expected_value_usd"] for o in org_opps)
    total_inv = sum(o.get("investment_required_usd") or 0 for o in org_opps)
    total_abatement = sum(o.get("abatement_potential_tco2e") or 0 for o in org_opps)

    by_cat: Dict[str, Dict[str, Any]] = {}
    for opp in org_opps:
        cat = opp["category"]
        if cat not in by_cat:
            by_cat[cat] = {"count": 0, "value_usd": 0, "investment_usd": 0, "abatement_tco2e": 0}
        by_cat[cat]["count"] += 1
        by_cat[cat]["value_usd"] = round(by_cat[cat]["value_usd"] + opp["net_value_usd"], 2)
        by_cat[cat]["investment_usd"] = round(by_cat[cat]["investment_usd"] + (opp.get("investment_required_usd") or 0), 2)
        by_cat[cat]["abatement_tco2e"] = round(by_cat[cat]["abatement_tco2e"] + (opp.get("abatement_potential_tco2e") or 0), 2)

    by_status: Dict[str, int] = {}
    for opp in org_opps:
        s = opp["status"]
        by_status[s] = by_status.get(s, 0) + 1

    by_th: Dict[str, Dict[str, Any]] = {}
    for opp in org_opps:
        th = opp["time_horizon"]
        if th not in by_th:
            by_th[th] = {"count": 0, "value_usd": 0}
        by_th[th]["count"] += 1
        by_th[th]["value_usd"] = round(by_th[th]["value_usd"] + opp["net_value_usd"], 2)

    return PipelineResponse(
        org_id=org_id,
        total_opportunities=len(org_opps),
        total_potential_value_usd=round(total_value, 2),
        total_expected_value_usd=round(total_ev, 2),
        total_investment_required_usd=round(total_inv, 2),
        total_abatement_tco2e=round(total_abatement, 2),
        by_category=by_cat,
        by_status=by_status,
        by_time_horizon=by_th,
        generated_at=_now(),
    )


@router.post(
    "/size/{opp_id}",
    response_model=SizingResponse,
    summary="Size revenue opportunity",
    description="Project the revenue potential of a climate opportunity over time.",
)
async def size_opportunity(opp_id: str, request: SizeOpportunityRequest) -> SizingResponse:
    """Size a revenue opportunity."""
    opp = _opportunities.get(opp_id)
    if not opp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")

    projection = {}
    total = 0.0
    base_revenue = request.market_size_usd * request.market_share_pct / 100
    for yr in range(1, request.years + 1):
        annual = round(base_revenue * (1 + request.growth_rate_pct / 100) ** yr, 2)
        projection[str(2025 + yr)] = annual
        total += annual

    discount_rate = 0.08
    npv = round(sum(v / (1 + discount_rate) ** i for i, v in enumerate(projection.values(), 1)), 2)
    investment = opp.get("investment_required_usd") or 0
    payback = None
    if investment > 0:
        cumulative = 0.0
        for i, v in enumerate(projection.values(), 1):
            cumulative += v
            if cumulative >= investment:
                payback = round(i - (cumulative - investment) / v, 1)
                break

    return SizingResponse(
        opportunity_id=opp_id,
        revenue_projection=projection,
        total_revenue_usd=round(total, 2),
        npv_usd=npv,
        payback_years=payback,
        generated_at=_now(),
    )


@router.get(
    "/savings/{org_id}",
    response_model=SavingsEstimateResponse,
    summary="Cost savings estimate",
    description="Estimate total cost savings from all identified opportunities.",
)
async def get_savings_estimate(org_id: str) -> SavingsEstimateResponse:
    """Estimate cost savings."""
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]
    by_cat: Dict[str, float] = {}
    by_opp = []
    total = 0.0

    for opp in org_opps:
        savings = opp.get("cost_savings_usd") or 0
        if savings > 0:
            total += savings
            cat = opp["category"]
            by_cat[cat] = round(by_cat.get(cat, 0) + savings, 2)
            by_opp.append({
                "opportunity_id": opp["opportunity_id"],
                "name": opp["name"],
                "annual_savings_usd": savings,
                "probability_pct": opp["probability_pct"],
            })

    return SavingsEstimateResponse(
        org_id=org_id,
        total_annual_savings_usd=round(total, 2),
        total_5yr_savings_usd=round(total * 5 * 0.9, 2),
        by_category=by_cat,
        by_opportunity=sorted(by_opp, key=lambda x: x["annual_savings_usd"], reverse=True),
        generated_at=_now(),
    )


@router.get(
    "/roi/{opp_id}",
    response_model=ROIResponse,
    summary="ROI analysis",
    description="Calculate ROI, payback period, NPV, and IRR for an opportunity.",
)
async def get_roi_analysis(opp_id: str) -> ROIResponse:
    """Calculate ROI analysis."""
    opp = _opportunities.get(opp_id)
    if not opp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Opportunity {opp_id} not found")

    investment = opp.get("investment_required_usd") or 1  # avoid div zero
    annual_return = (opp.get("revenue_potential_usd") or 0) + (opp.get("cost_savings_usd") or 0)
    roi_pct = round((annual_return / investment) * 100, 2) if investment > 0 else 0
    payback = round(investment / annual_return, 2) if annual_return > 0 else 99.0

    # NPV at 8% discount rate, 10 year horizon
    discount_rate = 0.08
    npv = round(-investment + sum(annual_return / (1 + discount_rate) ** yr for yr in range(1, 11)), 2)

    # Approximate IRR
    irr = round((annual_return / investment - 1) * 100 * 0.5 + 8.0, 2) if investment > 0 else 0

    return ROIResponse(
        opportunity_id=opp_id,
        investment_usd=investment,
        annual_return_usd=annual_return,
        roi_pct=roi_pct,
        payback_years=payback,
        npv_usd=npv,
        irr_pct=max(irr, 0),
        generated_at=_now(),
    )


@router.get(
    "/prioritize/{org_id}",
    response_model=PriorityMatrixResponse,
    summary="Priority matrix",
    description="Generate an impact vs feasibility priority matrix for all opportunities.",
)
async def get_priority_matrix(org_id: str) -> PriorityMatrixResponse:
    """Generate priority matrix."""
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]
    quadrants: Dict[str, List[Dict[str, Any]]] = {
        "quick_wins": [],       # High impact, high feasibility
        "strategic_bets": [],   # High impact, low feasibility
        "easy_gains": [],       # Low impact, high feasibility
        "lower_priority": [],   # Low impact, low feasibility
    }

    for opp in org_opps:
        impact_score = min((opp.get("net_value_usd", 0) / 1000000) * 10, 100)
        feasibility = opp.get("probability_pct", 50)
        entry = {
            "opportunity_id": opp["opportunity_id"],
            "name": opp["name"],
            "impact_score": round(impact_score, 1),
            "feasibility_score": feasibility,
            "net_value_usd": opp["net_value_usd"],
        }
        if impact_score >= 50 and feasibility >= 50:
            quadrants["quick_wins"].append(entry)
        elif impact_score >= 50:
            quadrants["strategic_bets"].append(entry)
        elif feasibility >= 50:
            quadrants["easy_gains"].append(entry)
        else:
            quadrants["lower_priority"].append(entry)

    # Recommended sequence: quick wins first, then easy gains, then strategic bets
    sequence = []
    for q in ["quick_wins", "easy_gains", "strategic_bets", "lower_priority"]:
        sorted_q = sorted(quadrants[q], key=lambda x: x["net_value_usd"], reverse=True)
        for item in sorted_q:
            sequence.append({**item, "quadrant": q})

    return PriorityMatrixResponse(
        org_id=org_id,
        quadrants=quadrants,
        recommended_sequence=sequence,
        generated_at=_now(),
    )


@router.get(
    "/financing/{org_id}",
    response_model=GreenFinancingResponse,
    summary="Green financing assessment",
    description="Assess eligibility for green bonds, sustainability-linked loans, and other green financing instruments.",
)
async def get_green_financing(org_id: str) -> GreenFinancingResponse:
    """Assess green financing eligibility."""
    org_opps = [o for o in _opportunities.values() if o["org_id"] == org_id]
    total_investment = sum(o.get("investment_required_usd") or 0 for o in org_opps)
    green_eligible_cats = {"energy_source", "resource_efficiency", "resilience"}
    eligible = sum(
        o.get("investment_required_usd") or 0
        for o in org_opps if o["category"] in green_eligible_cats
    )

    instruments = [
        {
            "instrument": "Green Bond",
            "eligible": eligible > 50000000,
            "min_size_usd": 50000000,
            "rate_savings_bps": 15,
            "requirements": ["ICMA Green Bond Principles", "Use of proceeds verification", "Annual impact reporting"],
        },
        {
            "instrument": "Sustainability-Linked Loan",
            "eligible": True,
            "min_size_usd": 10000000,
            "rate_savings_bps": 25,
            "requirements": ["KPI targets (e.g. Scope 1+2 reduction)", "Annual verification", "LMA/APLMA principles"],
        },
        {
            "instrument": "Green Loan",
            "eligible": eligible > 10000000,
            "min_size_usd": 10000000,
            "rate_savings_bps": 10,
            "requirements": ["LMA Green Loan Principles", "Project eligibility criteria", "Reporting"],
        },
        {
            "instrument": "Transition Bond",
            "eligible": True,
            "min_size_usd": 25000000,
            "rate_savings_bps": 20,
            "requirements": ["ICMA Climate Transition Finance Handbook", "Science-based pathway", "Governance"],
        },
    ]

    taxonomy_pct = round(eligible / total_investment * 100, 1) if total_investment > 0 else 0

    return GreenFinancingResponse(
        org_id=org_id,
        eligible_investment_usd=round(eligible, 2),
        green_bond_eligible=eligible > 50000000,
        sustainability_linked_loan_eligible=True,
        eligible_instruments=instruments,
        estimated_savings_bps=20,
        taxonomy_alignment_pct=taxonomy_pct,
        generated_at=_now(),
    )
