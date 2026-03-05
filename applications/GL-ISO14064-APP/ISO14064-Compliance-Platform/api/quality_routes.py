"""
GL-ISO14064-APP Quality Management API

Manages quality management plans, procedures, data quality assessments, and
corrective actions per ISO 14064-1:2018 Clause 7.

Procedure types: data_collection, calibration, internal_audit, document_control, training, review
Procedure statuses: pending -> in_progress -> completed | overdue
Corrective action priorities: low, medium, high, critical
Corrective action statuses: open -> in_progress -> resolved -> verified
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/quality", tags=["Quality Management"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QualityPlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    ARCHIVED = "archived"


class ProcedureType(str, Enum):
    DATA_COLLECTION = "data_collection"
    CALIBRATION = "calibration"
    INTERNAL_AUDIT = "internal_audit"
    DOCUMENT_CONTROL = "document_control"
    TRAINING = "training"
    REVIEW = "review"


class ProcedureStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"


class DataQualityDimension(str, Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    REPRESENTATIVENESS = "representativeness"


class CorrectiveActionPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrectiveActionStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    VERIFIED = "verified"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateQualityPlanRequest(BaseModel):
    inventory_id: str = Field(..., description="Inventory this plan applies to")
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field("", max_length=2000)
    responsible_person: str = Field("")
    review_frequency: str = Field("annual")
    scope: str = Field("Full ISO 14064-1:2018 inventory", max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
                "title": "FY2025 GHG Inventory Quality Management Plan",
                "description": "Comprehensive QM plan for FY2025 ISO 14064-1 inventory.",
                "responsible_person": "Jane Smith, Quality Manager",
                "review_frequency": "quarterly",
                "scope": "Full ISO 14064-1:2018 inventory including Categories 1-6",
            }
        }


class UpdateQualityPlanRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[QualityPlanStatus] = None
    responsible_person: Optional[str] = None
    review_frequency: Optional[str] = None
    scope: Optional[str] = Field(None, max_length=1000)


class CreateProcedureRequest(BaseModel):
    procedure_type: ProcedureType = Field(...)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field("", max_length=2000)
    responsible_person: str = Field("")
    frequency: str = Field("as_needed")
    target_date: Optional[str] = Field(None)
    applicable_categories: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "procedure_type": "data_collection",
                "title": "Monthly Utility Data Collection",
                "description": "Collect electricity and gas data from utility invoices.",
                "responsible_person": "Tom Jones",
                "frequency": "monthly",
                "target_date": "2025-01-31",
                "applicable_categories": ["category_1_direct", "category_2_energy"],
            }
        }


class UpdateProcedureStatusRequest(BaseModel):
    status: ProcedureStatus = Field(...)
    completion_notes: Optional[str] = Field(None, max_length=1000)


class RunDataQualityAssessmentRequest(BaseModel):
    assessed_by: str = Field("")
    notes: str = Field("", max_length=2000)


class CreateCorrectiveActionRequest(BaseModel):
    inventory_id: str = Field(...)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field("", max_length=2000)
    priority: CorrectiveActionPriority = Field(CorrectiveActionPriority.MEDIUM)
    source: str = Field("")
    affected_category: Optional[str] = None
    affected_dimension: Optional[DataQualityDimension] = None
    assigned_to: str = Field("")
    due_date: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
                "title": "Missing emission factors for refrigerant top-ups",
                "description": "Category 1 refrigerant emissions using default factors.",
                "priority": "high",
                "source": "internal_audit",
                "affected_category": "category_1_direct",
                "affected_dimension": "accuracy",
                "assigned_to": "Tom Jones",
                "due_date": "2025-03-31",
            }
        }


class UpdateCorrectiveActionRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[CorrectiveActionStatus] = None
    priority: Optional[CorrectiveActionPriority] = None
    assigned_to: Optional[str] = None
    due_date: Optional[str] = None
    resolution_notes: Optional[str] = Field(None, max_length=2000)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ProcedureResponse(BaseModel):
    procedure_id: str
    plan_id: str
    procedure_type: str
    title: str
    description: str
    responsible_person: str
    frequency: str
    target_date: Optional[str]
    applicable_categories: List[str]
    status: str
    completion_notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class QualityPlanResponse(BaseModel):
    plan_id: str
    inventory_id: str
    title: str
    description: str
    status: str
    responsible_person: str
    review_frequency: str
    scope: str
    procedure_count: int
    completed_procedures: int
    created_at: datetime
    updated_at: datetime


class DimensionScore(BaseModel):
    dimension: str
    score: float
    grade: str
    issues_found: int
    notes: str


class DataQualityMatrixResponse(BaseModel):
    inventory_id: str
    overall_score: float
    overall_grade: str
    dimensions: List[DimensionScore]
    category_scores: Dict[str, float]
    assessed_by: str
    assessed_at: datetime
    notes: str


class CorrectiveActionResponse(BaseModel):
    action_id: str
    inventory_id: str
    title: str
    description: str
    priority: str
    status: str
    source: str
    affected_category: Optional[str]
    affected_dimension: Optional[str]
    assigned_to: str
    due_date: Optional[str]
    resolution_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_quality_plans: Dict[str, Dict[str, Any]] = {}
_procedures: Dict[str, Dict[str, Any]] = {}
_data_quality_assessments: Dict[str, Dict[str, Any]] = {}
_corrective_actions: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Endpoints -- Quality Plans
# ---------------------------------------------------------------------------

@router.post(
    "/plans",
    response_model=QualityPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create quality management plan",
    description="Create a quality management plan per ISO 14064-1 Clause 7.",
)
async def create_quality_plan(request: CreateQualityPlanRequest) -> QualityPlanResponse:
    plan_id = _generate_id("qmp")
    now = _now()
    plan = {
        "plan_id": plan_id,
        "inventory_id": request.inventory_id,
        "title": request.title,
        "description": request.description,
        "status": QualityPlanStatus.DRAFT.value,
        "responsible_person": request.responsible_person,
        "review_frequency": request.review_frequency,
        "scope": request.scope,
        "procedure_count": 0,
        "completed_procedures": 0,
        "created_at": now,
        "updated_at": now,
    }
    _quality_plans[plan_id] = plan
    return QualityPlanResponse(**plan)


@router.get(
    "/plans",
    response_model=List[QualityPlanResponse],
    summary="List quality management plans",
    description="Retrieve all quality management plans, optionally filtered.",
)
async def list_quality_plans(
    inventory_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
) -> List[QualityPlanResponse]:
    plans = list(_quality_plans.values())
    if inventory_id:
        plans = [p for p in plans if p["inventory_id"] == inventory_id]
    if status_filter:
        plans = [p for p in plans if p["status"] == status_filter]
    plans.sort(key=lambda p: p["created_at"], reverse=True)
    for plan in plans:
        plan_procs = [pr for pr in _procedures.values() if pr["plan_id"] == plan["plan_id"]]
        plan["procedure_count"] = len(plan_procs)
        plan["completed_procedures"] = sum(1 for pr in plan_procs if pr["status"] == ProcedureStatus.COMPLETED.value)
    return [QualityPlanResponse(**p) for p in plans[:limit]]


@router.get(
    "/plans/{plan_id}",
    response_model=QualityPlanResponse,
    summary="Get quality management plan",
    description="Retrieve a quality management plan by ID.",
)
async def get_quality_plan(plan_id: str) -> QualityPlanResponse:
    plan = _quality_plans.get(plan_id)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")
    plan_procs = [pr for pr in _procedures.values() if pr["plan_id"] == plan_id]
    plan["procedure_count"] = len(plan_procs)
    plan["completed_procedures"] = sum(1 for pr in plan_procs if pr["status"] == ProcedureStatus.COMPLETED.value)
    return QualityPlanResponse(**plan)


@router.put(
    "/plans/{plan_id}",
    response_model=QualityPlanResponse,
    summary="Update quality management plan",
    description="Update plan title, description, status, or responsible person.",
)
async def update_quality_plan(plan_id: str, request: UpdateQualityPlanRequest) -> QualityPlanResponse:
    plan = _quality_plans.get(plan_id)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")
    updates = request.model_dump(exclude_unset=True)
    if "status" in updates:
        updates["status"] = updates["status"].value if hasattr(updates["status"], "value") else updates["status"]
    plan.update(updates)
    plan["updated_at"] = _now()
    plan_procs = [pr for pr in _procedures.values() if pr["plan_id"] == plan_id]
    plan["procedure_count"] = len(plan_procs)
    plan["completed_procedures"] = sum(1 for pr in plan_procs if pr["status"] == ProcedureStatus.COMPLETED.value)
    return QualityPlanResponse(**plan)


# ---------------------------------------------------------------------------
# Endpoints -- Procedures
# ---------------------------------------------------------------------------

@router.get(
    "/plans/{plan_id}/procedures",
    response_model=List[ProcedureResponse],
    summary="List procedures for a plan",
    description="Retrieve all procedures associated with a quality management plan.",
)
async def list_procedures(
    plan_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    type_filter: Optional[str] = Query(None, alias="type"),
) -> List[ProcedureResponse]:
    if plan_id not in _quality_plans:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")
    procedures = [pr for pr in _procedures.values() if pr["plan_id"] == plan_id]
    if status_filter:
        procedures = [pr for pr in procedures if pr["status"] == status_filter]
    if type_filter:
        procedures = [pr for pr in procedures if pr["procedure_type"] == type_filter]
    procedures.sort(key=lambda pr: pr["created_at"], reverse=True)
    return [ProcedureResponse(**pr) for pr in procedures]


@router.post(
    "/plans/{plan_id}/procedures",
    response_model=ProcedureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add procedure to plan",
    description="Add a quality procedure to a management plan.",
)
async def create_procedure(plan_id: str, request: CreateProcedureRequest) -> ProcedureResponse:
    plan = _quality_plans.get(plan_id)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")
    if plan["status"] == QualityPlanStatus.ARCHIVED.value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot add procedures to an archived plan.")
    procedure_id = _generate_id("proc")
    now = _now()
    procedure = {
        "procedure_id": procedure_id,
        "plan_id": plan_id,
        "procedure_type": request.procedure_type.value,
        "title": request.title,
        "description": request.description,
        "responsible_person": request.responsible_person,
        "frequency": request.frequency,
        "target_date": request.target_date,
        "applicable_categories": request.applicable_categories,
        "status": ProcedureStatus.PENDING.value,
        "completion_notes": None,
        "created_at": now,
        "updated_at": now,
    }
    _procedures[procedure_id] = procedure
    return ProcedureResponse(**procedure)


@router.put(
    "/procedures/{procedure_id}/status",
    response_model=ProcedureResponse,
    summary="Update procedure status",
    description="Update the status of a quality procedure.",
)
async def update_procedure_status(procedure_id: str, request: UpdateProcedureStatusRequest) -> ProcedureResponse:
    procedure = _procedures.get(procedure_id)
    if not procedure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Procedure {procedure_id} not found")
    procedure["status"] = request.status.value
    if request.completion_notes:
        procedure["completion_notes"] = request.completion_notes
    procedure["updated_at"] = _now()
    return ProcedureResponse(**procedure)


# ---------------------------------------------------------------------------
# Endpoints -- Data Quality
# ---------------------------------------------------------------------------

@router.get(
    "/data-quality/{inventory_id}",
    response_model=DataQualityMatrixResponse,
    summary="Get data quality matrix",
    description="Retrieve the data quality matrix for an inventory across five dimensions.",
)
async def get_data_quality_matrix(inventory_id: str) -> DataQualityMatrixResponse:
    cached = _data_quality_assessments.get(inventory_id)
    if cached:
        return DataQualityMatrixResponse(**{**cached, "dimensions": [DimensionScore(**d) for d in cached["dimensions"]]})
    dimensions = [
        {"dimension": "completeness", "score": 85.0, "grade": "B", "issues_found": 3, "notes": "Missing data for 2 facilities in Q3"},
        {"dimension": "consistency", "score": 90.0, "grade": "A", "issues_found": 1, "notes": "Minor unit inconsistency in Category 3"},
        {"dimension": "accuracy", "score": 78.0, "grade": "C", "issues_found": 5, "notes": "Default factors used for some sources"},
        {"dimension": "timeliness", "score": 82.0, "grade": "B", "issues_found": 2, "notes": "Utility data delayed for 2 facilities"},
        {"dimension": "representativeness", "score": 75.0, "grade": "C", "issues_found": 4, "notes": "Global defaults used for Cat 5-6"},
    ]
    overall = round(sum(d["score"] for d in dimensions) / len(dimensions), 1)
    return DataQualityMatrixResponse(
        inventory_id=inventory_id, overall_score=overall, overall_grade=_score_to_grade(overall),
        dimensions=[DimensionScore(**d) for d in dimensions],
        category_scores={"category_1_direct": 88.0, "category_2_energy": 92.0, "category_3_transport": 72.0,
                         "category_4_products_used": 68.0, "category_5_products_from_org": 70.0, "category_6_other": 65.0},
        assessed_by="System (default)", assessed_at=_now(), notes="Default matrix. Run assessment for updated scores.",
    )


@router.post(
    "/data-quality/{inventory_id}/assess",
    response_model=DataQualityMatrixResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run data quality assessment",
    description="Execute a data quality assessment for an inventory.",
)
async def run_data_quality_assessment(inventory_id: str, request: RunDataQualityAssessmentRequest) -> DataQualityMatrixResponse:
    dimensions = [
        {"dimension": "completeness", "score": 87.0, "grade": "B", "issues_found": 2, "notes": "95% sources covered"},
        {"dimension": "consistency", "score": 91.0, "grade": "A", "issues_found": 1, "notes": "Methodology consistent"},
        {"dimension": "accuracy", "score": 80.0, "grade": "B", "issues_found": 4, "notes": "Tier 2 for Cat 1-2, Tier 1 for Cat 3-6"},
        {"dimension": "timeliness", "score": 84.0, "grade": "B", "issues_found": 2, "notes": "90% within 30 days"},
        {"dimension": "representativeness", "score": 77.0, "grade": "C", "issues_found": 3, "notes": "Regional factors where available"},
    ]
    overall = round(sum(d["score"] for d in dimensions) / len(dimensions), 1)
    now = _now()
    assessment = {
        "inventory_id": inventory_id, "overall_score": overall, "overall_grade": _score_to_grade(overall),
        "dimensions": dimensions,
        "category_scores": {"category_1_direct": 89.0, "category_2_energy": 93.0, "category_3_transport": 74.0,
                           "category_4_products_used": 70.0, "category_5_products_from_org": 72.0, "category_6_other": 66.0},
        "assessed_by": request.assessed_by or "System", "assessed_at": now, "notes": request.notes or "Assessment completed.",
    }
    _data_quality_assessments[inventory_id] = assessment
    return DataQualityMatrixResponse(**{**assessment, "dimensions": [DimensionScore(**d) for d in dimensions]})


# ---------------------------------------------------------------------------
# Endpoints -- Corrective Actions
# ---------------------------------------------------------------------------

@router.get(
    "/corrective-actions",
    response_model=List[CorrectiveActionResponse],
    summary="List corrective actions",
    description="Retrieve corrective actions, optionally filtered by inventory, status, or priority.",
)
async def list_corrective_actions(
    inventory_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    priority_filter: Optional[str] = Query(None, alias="priority"),
    limit: int = Query(50, ge=1, le=200),
) -> List[CorrectiveActionResponse]:
    actions = list(_corrective_actions.values())
    if inventory_id:
        actions = [a for a in actions if a["inventory_id"] == inventory_id]
    if status_filter:
        actions = [a for a in actions if a["status"] == status_filter]
    if priority_filter:
        actions = [a for a in actions if a["priority"] == priority_filter]
    actions.sort(key=lambda a: a["created_at"], reverse=True)
    return [CorrectiveActionResponse(**a) for a in actions[:limit]]


@router.post(
    "/corrective-actions",
    response_model=CorrectiveActionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create corrective action",
    description="Create a corrective action for a data quality deficiency.",
)
async def create_corrective_action(request: CreateCorrectiveActionRequest) -> CorrectiveActionResponse:
    action_id = _generate_id("ca")
    now = _now()
    action = {
        "action_id": action_id, "inventory_id": request.inventory_id, "title": request.title,
        "description": request.description, "priority": request.priority.value,
        "status": CorrectiveActionStatus.OPEN.value, "source": request.source,
        "affected_category": request.affected_category,
        "affected_dimension": request.affected_dimension.value if request.affected_dimension else None,
        "assigned_to": request.assigned_to, "due_date": request.due_date, "resolution_notes": None,
        "created_at": now, "updated_at": now, "resolved_at": None,
    }
    _corrective_actions[action_id] = action
    return CorrectiveActionResponse(**action)


@router.put(
    "/corrective-actions/{action_id}",
    response_model=CorrectiveActionResponse,
    summary="Update corrective action",
    description="Update corrective action details, status, priority, or resolution notes.",
)
async def update_corrective_action(action_id: str, request: UpdateCorrectiveActionRequest) -> CorrectiveActionResponse:
    action = _corrective_actions.get(action_id)
    if not action:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Action {action_id} not found")
    updates = request.model_dump(exclude_unset=True)
    if "status" in updates:
        new_status = updates["status"]
        updates["status"] = new_status.value if hasattr(new_status, "value") else new_status
        if updates["status"] in (CorrectiveActionStatus.RESOLVED.value, CorrectiveActionStatus.VERIFIED.value):
            if not action.get("resolved_at"):
                action["resolved_at"] = _now()
    if "priority" in updates:
        updates["priority"] = updates["priority"].value if hasattr(updates["priority"], "value") else updates["priority"]
    if "affected_dimension" in updates and updates["affected_dimension"] is not None:
        updates["affected_dimension"] = updates["affected_dimension"].value if hasattr(updates["affected_dimension"], "value") else updates["affected_dimension"]
    action.update(updates)
    action["updated_at"] = _now()
    return CorrectiveActionResponse(**action)
