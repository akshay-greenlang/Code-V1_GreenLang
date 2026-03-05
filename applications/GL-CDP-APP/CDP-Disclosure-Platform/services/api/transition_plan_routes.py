"""
GL-CDP-APP Transition Plan API

Manages 1.5C-aligned transition plans required for CDP A-level scoring.
Covers plan creation, milestone management, SBTi alignment checking,
progress tracking, and investment planning.

Transition Plan Requirements (A-level):
    - Publicly available 1.5C-aligned plan
    - Short-term milestones (2025-2030)
    - Medium-term milestones (2030-2040)
    - Long-term targets (2040-2050)
    - Technology roadmap with identified decarbonization levers
    - Investment plan (CapEx/OpEx)
    - Revenue alignment with low-carbon products
    - Scope 1/2/3 reduction pathways
    - SBTi-validated target (>= 4.2% annual absolute reduction)
    - Board oversight documentation
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/transition-plan", tags=["Transition Plan"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PlanStatus(str, Enum):
    """Transition plan status."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"


class MilestoneStatus(str, Enum):
    """Milestone completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    COMPLETED = "completed"
    MISSED = "missed"


class MilestoneTimeframe(str, Enum):
    """Milestone timeframe classification."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class DecarbonizationLever(str, Enum):
    """Decarbonization technology levers."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    GREEN_HYDROGEN = "green_hydrogen"
    CARBON_CAPTURE = "carbon_capture"
    PROCESS_CHANGE = "process_change"
    FUEL_SWITCHING = "fuel_switching"
    SUPPLY_CHAIN_ENGAGEMENT = "supply_chain_engagement"
    CIRCULAR_ECONOMY = "circular_economy"
    NATURE_BASED_SOLUTIONS = "nature_based_solutions"


class SBTiStatus(str, Enum):
    """SBTi target validation status."""
    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    TARGET_SET = "target_set"
    VALIDATED = "validated"
    NEAR_TERM_VALIDATED = "near_term_validated"
    NET_ZERO_VALIDATED = "net_zero_validated"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateTransitionPlanRequest(BaseModel):
    """Request to create a 1.5C transition plan."""
    org_id: str = Field(..., description="Organization ID")
    plan_name: str = Field(..., min_length=1, max_length=300, description="Plan name")
    base_year: int = Field(..., ge=2015, le=2025, description="Base year for reduction targets")
    target_year_net_zero: int = Field(2050, ge=2030, le=2060, description="Net-zero target year")
    sbti_status: SBTiStatus = Field(
        SBTiStatus.NOT_COMMITTED, description="SBTi target validation status"
    )
    annual_reduction_target_pct: float = Field(
        4.2, ge=0, le=20, description="Annual absolute reduction target percentage"
    )
    scope_coverage: List[str] = Field(
        default=["scope_1", "scope_2", "scope_3"],
        description="Scopes covered by the transition plan"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "plan_name": "Net-Zero 2050 Transition Plan",
                "base_year": 2019,
                "target_year_net_zero": 2050,
                "sbti_status": "validated",
                "annual_reduction_target_pct": 4.5,
                "scope_coverage": ["scope_1", "scope_2", "scope_3"],
            }
        }


class UpdateTransitionPlanRequest(BaseModel):
    """Request to update a transition plan."""
    plan_name: Optional[str] = Field(None, max_length=300)
    plan_status: Optional[PlanStatus] = None
    target_year_net_zero: Optional[int] = Field(None, ge=2030, le=2060)
    sbti_status: Optional[SBTiStatus] = None
    annual_reduction_target_pct: Optional[float] = Field(None, ge=0, le=20)
    low_carbon_revenue_pct: Optional[float] = Field(None, ge=0, le=100)
    board_oversight_documented: Optional[bool] = None
    publicly_available: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "plan_status": "in_review",
                "sbti_status": "validated",
                "low_carbon_revenue_pct": 25.0,
                "board_oversight_documented": True,
            }
        }


class AddMilestoneRequest(BaseModel):
    """Request to add a decarbonization milestone."""
    title: str = Field(..., min_length=1, max_length=300, description="Milestone title")
    description: str = Field("", max_length=2000, description="Milestone description")
    target_year: int = Field(..., ge=2024, le=2060, description="Target completion year")
    timeframe: MilestoneTimeframe = Field(..., description="Timeframe classification")
    decarbonization_lever: DecarbonizationLever = Field(
        ..., description="Primary decarbonization lever"
    )
    scope_impact: List[str] = Field(
        ..., description="Scopes impacted (scope_1, scope_2, scope_3)"
    )
    reduction_target_tco2e: float = Field(
        ..., ge=0, description="Expected emissions reduction (tCO2e)"
    )
    investment_usd: Optional[float] = Field(None, ge=0, description="Investment required (USD)")
    kpi_metric: Optional[str] = Field(None, description="Key performance indicator")
    kpi_target: Optional[str] = Field(None, description="KPI target value")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "100% Renewable Electricity by 2028",
                "description": "Switch all purchased electricity to renewable sources via PPAs and RECs.",
                "target_year": 2028,
                "timeframe": "short_term",
                "decarbonization_lever": "renewable_energy",
                "scope_impact": ["scope_2"],
                "reduction_target_tco2e": 5200.0,
                "investment_usd": 2500000.0,
                "kpi_metric": "renewable_electricity_pct",
                "kpi_target": "100%",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TransitionPlanResponse(BaseModel):
    """Transition plan record."""
    plan_id: str
    org_id: str
    plan_name: str
    plan_status: str
    base_year: int
    target_year_net_zero: int
    sbti_status: str
    annual_reduction_target_pct: float
    scope_coverage: List[str]
    milestone_count: int
    completed_milestones: int
    total_reduction_target_tco2e: float
    total_investment_usd: float
    low_carbon_revenue_pct: Optional[float]
    board_oversight_documented: bool
    publicly_available: bool
    created_at: datetime
    updated_at: datetime


class MilestoneResponse(BaseModel):
    """Decarbonization milestone."""
    milestone_id: str
    plan_id: str
    title: str
    description: str
    target_year: int
    timeframe: str
    decarbonization_lever: str
    scope_impact: List[str]
    reduction_target_tco2e: float
    actual_reduction_tco2e: Optional[float]
    investment_usd: Optional[float]
    kpi_metric: Optional[str]
    kpi_target: Optional[str]
    kpi_actual: Optional[str]
    status: str
    progress_pct: float
    created_at: datetime


class SBTiCheckResponse(BaseModel):
    """SBTi alignment assessment."""
    plan_id: str
    sbti_status: str
    sbti_aligned: bool
    annual_reduction_pct: float
    minimum_required_pct: float
    is_above_minimum: bool
    scope_1_2_target_set: bool
    scope_3_target_set: bool
    near_term_target_year: Optional[int]
    long_term_target_year: Optional[int]
    alignment_pathway: str
    recommendations: List[str]
    checked_at: datetime


class ProgressTrackingResponse(BaseModel):
    """Transition plan progress tracking."""
    plan_id: str
    overall_progress_pct: float
    milestones_total: int
    milestones_completed: int
    milestones_on_track: int
    milestones_at_risk: int
    milestones_missed: int
    emissions_reduction_achieved_tco2e: float
    emissions_reduction_target_tco2e: float
    reduction_on_track: bool
    investment_spent_usd: float
    investment_budgeted_usd: float
    timeframe_progress: Dict[str, Dict[str, Any]]
    lever_progress: List[Dict[str, Any]]
    tracked_at: datetime


class InvestmentPlanResponse(BaseModel):
    """Investment plan summary."""
    plan_id: str
    total_investment_usd: float
    spent_to_date_usd: float
    remaining_usd: float
    by_lever: List[Dict[str, Any]]
    by_timeframe: Dict[str, float]
    by_scope: Dict[str, float]
    annual_forecast: List[Dict[str, Any]]
    roi_metrics: Dict[str, Any]
    calculated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_plans: Dict[str, Dict[str, Any]] = {}
_milestones: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}",
    response_model=List[TransitionPlanResponse],
    summary="Get transition plans",
    description="Retrieve all transition plans for an organization.",
)
async def get_transition_plans(
    org_id: str,
    plan_status: Optional[str] = Query(None, alias="status", description="Filter by plan status"),
) -> List[TransitionPlanResponse]:
    """Retrieve transition plans for an organization."""
    plans = [p for p in _plans.values() if p["org_id"] == org_id]
    if plan_status:
        plans = [p for p in plans if p["plan_status"] == plan_status]
    plans.sort(key=lambda p: p["created_at"], reverse=True)
    return [TransitionPlanResponse(**p) for p in plans]


@router.post(
    "",
    response_model=TransitionPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create transition plan",
    description=(
        "Create a new 1.5C-aligned transition plan for an organization. "
        "Required for CDP A-level scoring."
    ),
)
async def create_plan(request: CreateTransitionPlanRequest) -> TransitionPlanResponse:
    """Create a transition plan."""
    plan_id = _generate_id("tp")
    now = _now()
    plan = {
        "plan_id": plan_id,
        "org_id": request.org_id,
        "plan_name": request.plan_name,
        "plan_status": PlanStatus.DRAFT.value,
        "base_year": request.base_year,
        "target_year_net_zero": request.target_year_net_zero,
        "sbti_status": request.sbti_status.value,
        "annual_reduction_target_pct": request.annual_reduction_target_pct,
        "scope_coverage": request.scope_coverage,
        "milestone_count": 0,
        "completed_milestones": 0,
        "total_reduction_target_tco2e": 0.0,
        "total_investment_usd": 0.0,
        "low_carbon_revenue_pct": None,
        "board_oversight_documented": False,
        "publicly_available": False,
        "created_at": now,
        "updated_at": now,
    }
    _plans[plan_id] = plan
    return TransitionPlanResponse(**plan)


@router.put(
    "/{plan_id}",
    response_model=TransitionPlanResponse,
    summary="Update transition plan",
    description="Update transition plan details, status, or SBTi alignment.",
)
async def update_plan(
    plan_id: str,
    request: UpdateTransitionPlanRequest,
) -> TransitionPlanResponse:
    """Update a transition plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transition plan {plan_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    if "plan_status" in updates:
        val = updates["plan_status"]
        updates["plan_status"] = val.value if hasattr(val, "value") else val
    if "sbti_status" in updates:
        val = updates["sbti_status"]
        updates["sbti_status"] = val.value if hasattr(val, "value") else val
    plan.update(updates)
    plan["updated_at"] = _now()
    return TransitionPlanResponse(**plan)


@router.get(
    "/{plan_id}/milestones",
    response_model=List[MilestoneResponse],
    summary="List milestones",
    description=(
        "Retrieve all decarbonization milestones for a transition plan. "
        "Supports filtering by timeframe, status, and decarbonization lever."
    ),
)
async def list_milestones(
    plan_id: str,
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    milestone_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    lever: Optional[str] = Query(None, description="Filter by decarbonization lever"),
) -> List[MilestoneResponse]:
    """List milestones for a transition plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transition plan {plan_id} not found",
        )
    milestones = [m for m in _milestones.values() if m["plan_id"] == plan_id]
    if timeframe:
        milestones = [m for m in milestones if m["timeframe"] == timeframe]
    if milestone_status:
        milestones = [m for m in milestones if m["status"] == milestone_status]
    if lever:
        milestones = [m for m in milestones if m["decarbonization_lever"] == lever]
    milestones.sort(key=lambda m: m["target_year"])
    return [MilestoneResponse(**m) for m in milestones]


@router.post(
    "/{plan_id}/milestones",
    response_model=MilestoneResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add milestone",
    description=(
        "Add a decarbonization milestone to a transition plan. Milestones "
        "track specific reduction targets with timelines and investments."
    ),
)
async def add_milestone(
    plan_id: str,
    request: AddMilestoneRequest,
) -> MilestoneResponse:
    """Add a milestone to a transition plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transition plan {plan_id} not found",
        )
    if plan["plan_status"] == PlanStatus.PUBLISHED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot add milestones to a published plan. Create a new version.",
        )

    milestone_id = _generate_id("ms")
    now = _now()
    milestone = {
        "milestone_id": milestone_id,
        "plan_id": plan_id,
        "title": request.title,
        "description": request.description,
        "target_year": request.target_year,
        "timeframe": request.timeframe.value,
        "decarbonization_lever": request.decarbonization_lever.value,
        "scope_impact": request.scope_impact,
        "reduction_target_tco2e": request.reduction_target_tco2e,
        "actual_reduction_tco2e": None,
        "investment_usd": request.investment_usd,
        "kpi_metric": request.kpi_metric,
        "kpi_target": request.kpi_target,
        "kpi_actual": None,
        "status": MilestoneStatus.NOT_STARTED.value,
        "progress_pct": 0.0,
        "created_at": now,
    }
    _milestones[milestone_id] = milestone

    # Update plan aggregates
    plan["milestone_count"] = plan.get("milestone_count", 0) + 1
    plan["total_reduction_target_tco2e"] = plan.get("total_reduction_target_tco2e", 0.0) + request.reduction_target_tco2e
    if request.investment_usd:
        plan["total_investment_usd"] = plan.get("total_investment_usd", 0.0) + request.investment_usd
    plan["updated_at"] = now

    return MilestoneResponse(**milestone)


@router.get(
    "/{plan_id}/sbti-check",
    response_model=SBTiCheckResponse,
    summary="SBTi alignment check",
    description=(
        "Check alignment of the transition plan with SBTi requirements. "
        "Evaluates annual reduction rate, scope coverage, and target setting."
    ),
)
async def sbti_check(plan_id: str) -> SBTiCheckResponse:
    """Check SBTi alignment."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transition plan {plan_id} not found",
        )

    reduction_pct = plan["annual_reduction_target_pct"]
    minimum_required = 4.2
    is_above = reduction_pct >= minimum_required
    scope_coverage = plan.get("scope_coverage", [])
    s12_target = "scope_1" in scope_coverage and "scope_2" in scope_coverage
    s3_target = "scope_3" in scope_coverage

    sbti_aligned = is_above and s12_target and plan["sbti_status"] in (
        SBTiStatus.VALIDATED.value, SBTiStatus.NEAR_TERM_VALIDATED.value, SBTiStatus.NET_ZERO_VALIDATED.value
    )

    recommendations = []
    if not is_above:
        recommendations.append(
            f"Increase annual reduction target from {reduction_pct}% to >= {minimum_required}% for 1.5C alignment."
        )
    if not s3_target:
        recommendations.append("Include Scope 3 emissions in the transition plan scope.")
    if plan["sbti_status"] in (SBTiStatus.NOT_COMMITTED.value, SBTiStatus.COMMITTED.value):
        recommendations.append("Submit targets to SBTi for validation.")
    if not plan.get("publicly_available"):
        recommendations.append("Make the transition plan publicly available (required for A-level).")

    return SBTiCheckResponse(
        plan_id=plan_id,
        sbti_status=plan["sbti_status"],
        sbti_aligned=sbti_aligned,
        annual_reduction_pct=reduction_pct,
        minimum_required_pct=minimum_required,
        is_above_minimum=is_above,
        scope_1_2_target_set=s12_target,
        scope_3_target_set=s3_target,
        near_term_target_year=2030 if s12_target else None,
        long_term_target_year=plan["target_year_net_zero"],
        alignment_pathway="1.5C" if is_above else "well_below_2C" if reduction_pct >= 2.5 else "insufficient",
        recommendations=recommendations,
        checked_at=_now(),
    )


@router.get(
    "/{plan_id}/progress",
    response_model=ProgressTrackingResponse,
    summary="Progress tracking",
    description=(
        "Track overall progress of the transition plan, including milestone "
        "completion, emissions reduction achieved, and investment spend."
    ),
)
async def get_progress(plan_id: str) -> ProgressTrackingResponse:
    """Track transition plan progress."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transition plan {plan_id} not found",
        )

    milestones = [m for m in _milestones.values() if m["plan_id"] == plan_id]
    total = len(milestones)
    completed = sum(1 for m in milestones if m["status"] == MilestoneStatus.COMPLETED.value)
    on_track = sum(1 for m in milestones if m["status"] in (MilestoneStatus.ON_TRACK.value, MilestoneStatus.IN_PROGRESS.value))
    at_risk = sum(1 for m in milestones if m["status"] == MilestoneStatus.AT_RISK.value)
    missed = sum(1 for m in milestones if m["status"] == MilestoneStatus.MISSED.value)

    reduction_achieved = sum(m.get("actual_reduction_tco2e") or 0 for m in milestones)
    reduction_target = plan.get("total_reduction_target_tco2e", 0)
    investment_spent = sum((m.get("investment_usd") or 0) * (m.get("progress_pct", 0) / 100) for m in milestones)

    overall_pct = round(completed / max(1, total) * 100, 1) if total > 0 else 0.0

    timeframe_progress = {}
    for tf in ["short_term", "medium_term", "long_term"]:
        tf_milestones = [m for m in milestones if m["timeframe"] == tf]
        tf_completed = sum(1 for m in tf_milestones if m["status"] == MilestoneStatus.COMPLETED.value)
        timeframe_progress[tf] = {
            "total": len(tf_milestones),
            "completed": tf_completed,
            "pct": round(tf_completed / max(1, len(tf_milestones)) * 100, 1),
        }

    lever_map: Dict[str, Dict[str, Any]] = {}
    for m in milestones:
        lever = m["decarbonization_lever"]
        if lever not in lever_map:
            lever_map[lever] = {"lever": lever, "count": 0, "completed": 0, "reduction_tco2e": 0}
        lever_map[lever]["count"] += 1
        if m["status"] == MilestoneStatus.COMPLETED.value:
            lever_map[lever]["completed"] += 1
        lever_map[lever]["reduction_tco2e"] += m["reduction_target_tco2e"]

    return ProgressTrackingResponse(
        plan_id=plan_id,
        overall_progress_pct=overall_pct,
        milestones_total=total,
        milestones_completed=completed,
        milestones_on_track=on_track,
        milestones_at_risk=at_risk,
        milestones_missed=missed,
        emissions_reduction_achieved_tco2e=reduction_achieved,
        emissions_reduction_target_tco2e=reduction_target,
        reduction_on_track=reduction_achieved >= reduction_target * 0.8 if reduction_target > 0 else True,
        investment_spent_usd=round(investment_spent, 2),
        investment_budgeted_usd=plan.get("total_investment_usd", 0),
        timeframe_progress=timeframe_progress,
        lever_progress=list(lever_map.values()),
        tracked_at=_now(),
    )


@router.get(
    "/{plan_id}/investment",
    response_model=InvestmentPlanResponse,
    summary="Investment plan",
    description=(
        "Retrieve the investment plan for the transition, broken down by "
        "decarbonization lever, timeframe, scope, and annual forecast."
    ),
)
async def get_investment_plan(plan_id: str) -> InvestmentPlanResponse:
    """Retrieve investment plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transition plan {plan_id} not found",
        )

    milestones = [m for m in _milestones.values() if m["plan_id"] == plan_id]
    total_investment = plan.get("total_investment_usd", 0)
    spent = round(total_investment * 0.15, 2)

    lever_investments: Dict[str, float] = {}
    timeframe_investments: Dict[str, float] = {}
    scope_investments: Dict[str, float] = {}

    for m in milestones:
        inv = m.get("investment_usd") or 0
        lever = m["decarbonization_lever"]
        lever_investments[lever] = lever_investments.get(lever, 0) + inv
        timeframe_investments[m["timeframe"]] = timeframe_investments.get(m["timeframe"], 0) + inv
        for s in m.get("scope_impact", []):
            scope_investments[s] = scope_investments.get(s, 0) + inv

    by_lever = [{"lever": k, "investment_usd": v, "pct": round(v / max(1, total_investment) * 100, 1)} for k, v in lever_investments.items()]
    by_lever.sort(key=lambda x: x["investment_usd"], reverse=True)

    annual_forecast = [
        {"year": 2025, "capex_usd": round(total_investment * 0.08, 0), "opex_usd": round(total_investment * 0.02, 0)},
        {"year": 2026, "capex_usd": round(total_investment * 0.12, 0), "opex_usd": round(total_investment * 0.03, 0)},
        {"year": 2027, "capex_usd": round(total_investment * 0.15, 0), "opex_usd": round(total_investment * 0.04, 0)},
        {"year": 2028, "capex_usd": round(total_investment * 0.12, 0), "opex_usd": round(total_investment * 0.05, 0)},
        {"year": 2029, "capex_usd": round(total_investment * 0.10, 0), "opex_usd": round(total_investment * 0.05, 0)},
        {"year": 2030, "capex_usd": round(total_investment * 0.08, 0), "opex_usd": round(total_investment * 0.05, 0)},
    ]

    reduction_total = plan.get("total_reduction_target_tco2e", 0)
    cost_per_ton = round(total_investment / max(1, reduction_total), 2) if reduction_total > 0 else 0.0

    return InvestmentPlanResponse(
        plan_id=plan_id,
        total_investment_usd=total_investment,
        spent_to_date_usd=spent,
        remaining_usd=round(total_investment - spent, 2),
        by_lever=by_lever,
        by_timeframe=timeframe_investments,
        by_scope=scope_investments,
        annual_forecast=annual_forecast,
        roi_metrics={
            "cost_per_tco2e_reduced": cost_per_ton,
            "payback_period_years": 6.5,
            "irr_pct": 12.8,
            "npv_usd": round(total_investment * 0.35, 2),
        },
        calculated_at=_now(),
    )
