"""
GL-ISO14064-APP Management Plan & Actions API

Manages GHG management plans and improvement actions per ISO 14064-1:2018
Clause 9.  A management plan groups actions for an organization-year,
tracking planned reductions, investment, and review cycles.

Action categories:
    - emission_reduction: Reduce GHG emissions
    - removal_enhancement: Increase GHG removals
    - data_improvement: Improve data quality and completeness
    - process_improvement: Improve GHG management processes

Action statuses: planned -> in_progress -> completed | deferred | cancelled
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/management", tags=["Management Plans"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionStatus(str, Enum):
    """Status of improvement actions."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class ActionCategory(str, Enum):
    """Categories for management actions."""
    EMISSION_REDUCTION = "emission_reduction"
    REMOVAL_ENHANCEMENT = "removal_enhancement"
    DATA_IMPROVEMENT = "data_improvement"
    PROCESS_IMPROVEMENT = "process_improvement"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateManagementPlanRequest(BaseModel):
    """Request to create a management plan."""
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(2025, ge=1990, le=2100, description="Reporting year")
    objectives: List[str] = Field(
        default_factory=list, description="Plan objectives"
    )
    review_cycle: str = Field("annual", description="Review cycle (annual, semi-annual, quarterly)")

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "reporting_year": 2025,
                "objectives": [
                    "Reduce Category 1 emissions by 10% through energy efficiency",
                    "Improve data quality for Category 3 from Tier 1 to Tier 2",
                    "Establish monitoring plan for forestry removals",
                ],
                "review_cycle": "annual",
            }
        }


class UpdateManagementPlanRequest(BaseModel):
    """Request to update a management plan."""
    objectives: Optional[List[str]] = None
    review_cycle: Optional[str] = None


class CreateActionRequest(BaseModel):
    """Request to create a management action."""
    title: str = Field(..., min_length=1, max_length=500, description="Action title")
    description: str = Field("", max_length=2000, description="Action description")
    action_category: ActionCategory = Field(
        ActionCategory.EMISSION_REDUCTION, description="Action category"
    )
    target_category: Optional[str] = Field(None, description="Target ISO category")
    priority: str = Field("medium", description="Priority: low, medium, high, critical")
    estimated_reduction_tco2e: Optional[float] = Field(None, ge=0, description="Estimated reduction")
    estimated_cost_usd: Optional[float] = Field(None, ge=0, description="Estimated cost (USD)")
    responsible_person: str = Field("", description="Person responsible")
    target_date: Optional[str] = Field(None, description="Target completion date (YYYY-MM-DD)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Install variable frequency drives on HVAC systems",
                "description": "Retrofit 12 AHUs with VFDs to reduce electricity consumption by 15%",
                "action_category": "emission_reduction",
                "target_category": "category_2_energy",
                "priority": "high",
                "estimated_reduction_tco2e": 320.0,
                "estimated_cost_usd": 180000.0,
                "responsible_person": "John Smith, Facilities Manager",
                "target_date": "2025-12-31",
            }
        }


class UpdateActionRequest(BaseModel):
    """Request to update a management action."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[ActionStatus] = None
    priority: Optional[str] = None
    estimated_reduction_tco2e: Optional[float] = Field(None, ge=0)
    estimated_cost_usd: Optional[float] = Field(None, ge=0)
    responsible_person: Optional[str] = None
    target_date: Optional[str] = None
    progress_note: Optional[str] = Field(None, max_length=1000, description="Progress update note")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ActionResponse(BaseModel):
    """A management action."""
    action_id: str
    plan_id: str
    org_id: str
    title: str
    description: str
    action_category: str
    target_category: Optional[str]
    status: str
    priority: str
    estimated_reduction_tco2e: Optional[float]
    estimated_cost_usd: Optional[float]
    responsible_person: str
    target_date: Optional[str]
    progress_notes: List[str]
    created_at: datetime
    updated_at: datetime


class ManagementPlanResponse(BaseModel):
    """A GHG management plan."""
    plan_id: str
    org_id: str
    reporting_year: int
    objectives: List[str]
    review_cycle: str
    action_count: int
    total_planned_reduction_tco2e: float
    total_planned_investment_usd: float
    actions: Optional[List[ActionResponse]] = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_plans: Dict[str, Dict[str, Any]] = {}
_actions: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _compute_plan_totals(plan_id: str) -> tuple:
    """Compute total planned reduction and investment for a plan."""
    plan_actions = [a for a in _actions.values() if a["plan_id"] == plan_id]
    total_reduction = sum(
        a.get("estimated_reduction_tco2e", 0) or 0 for a in plan_actions
    )
    total_investment = sum(
        a.get("estimated_cost_usd", 0) or 0 for a in plan_actions
    )
    return round(total_reduction, 2), round(total_investment, 2), len(plan_actions)


# ---------------------------------------------------------------------------
# Endpoints -- Plans
# ---------------------------------------------------------------------------

@router.post(
    "/plans",
    response_model=ManagementPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create management plan",
    description=(
        "Create a GHG management plan per ISO 14064-1 Clause 9. "
        "Define objectives and review cycle.  Add actions separately."
    ),
)
async def create_plan(request: CreateManagementPlanRequest) -> ManagementPlanResponse:
    """Create a GHG management plan."""
    plan_id = _generate_id("plan")
    now = _now()
    plan = {
        "plan_id": plan_id,
        "org_id": request.org_id,
        "reporting_year": request.reporting_year,
        "objectives": request.objectives,
        "review_cycle": request.review_cycle,
        "action_count": 0,
        "total_planned_reduction_tco2e": 0.0,
        "total_planned_investment_usd": 0.0,
        "created_at": now,
        "updated_at": now,
    }
    _plans[plan_id] = plan
    return ManagementPlanResponse(**plan, actions=[])


@router.get(
    "/plans/{plan_id}",
    response_model=ManagementPlanResponse,
    summary="Get management plan",
    description="Retrieve a management plan with all its actions.",
)
async def get_plan(
    plan_id: str,
    include_actions: bool = Query(True, description="Include actions in response"),
) -> ManagementPlanResponse:
    """Retrieve a management plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Management plan {plan_id} not found",
        )
    total_red, total_inv, count = _compute_plan_totals(plan_id)
    plan["total_planned_reduction_tco2e"] = total_red
    plan["total_planned_investment_usd"] = total_inv
    plan["action_count"] = count
    actions = None
    if include_actions:
        plan_actions = [a for a in _actions.values() if a["plan_id"] == plan_id]
        plan_actions.sort(key=lambda a: a["created_at"], reverse=True)
        actions = [ActionResponse(**a) for a in plan_actions]
    return ManagementPlanResponse(**plan, actions=actions)


@router.get(
    "/plans",
    response_model=List[ManagementPlanResponse],
    summary="List management plans",
    description="List all management plans for an organization.",
)
async def list_plans(
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[ManagementPlanResponse]:
    """List management plans for an organization."""
    plans = [p for p in _plans.values() if p["org_id"] == org_id]
    plans.sort(key=lambda p: p["reporting_year"], reverse=True)
    results = []
    for plan in plans[:limit]:
        total_red, total_inv, count = _compute_plan_totals(plan["plan_id"])
        plan["total_planned_reduction_tco2e"] = total_red
        plan["total_planned_investment_usd"] = total_inv
        plan["action_count"] = count
        results.append(ManagementPlanResponse(**plan, actions=None))
    return results


@router.put(
    "/plans/{plan_id}",
    response_model=ManagementPlanResponse,
    summary="Update management plan",
    description="Update management plan objectives or review cycle.",
)
async def update_plan(
    plan_id: str,
    request: UpdateManagementPlanRequest,
) -> ManagementPlanResponse:
    """Update a management plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Management plan {plan_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    plan.update(updates)
    plan["updated_at"] = _now()
    total_red, total_inv, count = _compute_plan_totals(plan_id)
    plan["total_planned_reduction_tco2e"] = total_red
    plan["total_planned_investment_usd"] = total_inv
    plan["action_count"] = count
    return ManagementPlanResponse(**plan, actions=None)


# ---------------------------------------------------------------------------
# Endpoints -- Actions
# ---------------------------------------------------------------------------

@router.post(
    "/plans/{plan_id}/actions",
    response_model=ActionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create management action",
    description="Add an improvement action to a management plan.",
)
async def create_action(
    plan_id: str,
    request: CreateActionRequest,
) -> ActionResponse:
    """Create a management action under a plan."""
    plan = _plans.get(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Management plan {plan_id} not found",
        )
    action_id = _generate_id("act")
    now = _now()
    action = {
        "action_id": action_id,
        "plan_id": plan_id,
        "org_id": plan["org_id"],
        "title": request.title,
        "description": request.description,
        "action_category": request.action_category.value,
        "target_category": request.target_category,
        "status": ActionStatus.PLANNED.value,
        "priority": request.priority,
        "estimated_reduction_tco2e": request.estimated_reduction_tco2e,
        "estimated_cost_usd": request.estimated_cost_usd,
        "responsible_person": request.responsible_person,
        "target_date": request.target_date,
        "progress_notes": [],
        "created_at": now,
        "updated_at": now,
    }
    _actions[action_id] = action
    return ActionResponse(**action)


@router.get(
    "/plans/{plan_id}/actions",
    response_model=List[ActionResponse],
    summary="List actions for a plan",
    description="Retrieve all actions for a management plan.",
)
async def list_actions(
    plan_id: str,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
) -> List[ActionResponse]:
    """List actions for a management plan."""
    if plan_id not in _plans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Management plan {plan_id} not found",
        )
    actions = [a for a in _actions.values() if a["plan_id"] == plan_id]
    if status_filter:
        actions = [a for a in actions if a["status"] == status_filter]
    actions.sort(key=lambda a: a["created_at"], reverse=True)
    return [ActionResponse(**a) for a in actions]


@router.put(
    "/actions/{action_id}",
    response_model=ActionResponse,
    summary="Update management action",
    description="Update action details, status, or add a progress note.",
)
async def update_action(
    action_id: str,
    request: UpdateActionRequest,
) -> ActionResponse:
    """Update a management action."""
    action = _actions.get(action_id)
    if not action:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Action {action_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    progress_note = updates.pop("progress_note", None)
    if "status" in updates:
        updates["status"] = updates["status"].value if hasattr(updates["status"], "value") else updates["status"]
    action.update(updates)
    if progress_note:
        action["progress_notes"].append(progress_note)
    action["updated_at"] = _now()
    return ActionResponse(**action)
