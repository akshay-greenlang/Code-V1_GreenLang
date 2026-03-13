# -*- coding: utf-8 -*-
"""
Remediation Plan Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for remediation plan lifecycle management including creation,
listing, detail retrieval, updates, status transitions, plan cloning,
Gantt chart generation, milestone management, and evidence uploads.

Endpoints (10):
    POST /plans                                          - Create remediation plan
    GET  /plans                                          - List plans with filters
    GET  /plans/{plan_id}                                - Get plan detail
    PUT  /plans/{plan_id}                                - Update plan details
    PUT  /plans/{plan_id}/status                         - Update plan status
    POST /plans/{plan_id}/clone                          - Clone plan for another supplier
    GET  /plans/{plan_id}/gantt                          - Get Gantt chart data
    POST /plans/{plan_id}/milestones                     - Add milestone to plan
    PUT  /plans/{plan_id}/milestones/{milestone_id}      - Update milestone
    POST /plans/{plan_id}/milestones/{milestone_id}/evidence - Upload evidence

RBAC Permissions:
    eudr-rma:plans:read     - View plans and milestones
    eudr-rma:plans:write    - Create, update, clone plans, manage milestones
    eudr-rma:plans:approve  - Approve plan activation and completion

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 2: Remediation Plan Design
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_rma_service,
    rate_limit_standard,
    rate_limit_upload,
    rate_limit_write,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    CreatePlanRequest,
    ErrorResponse,
    EvidenceUploadRequest,
    GanttChartResponse,
    MilestoneCreateRequest,
    MilestoneUpdateRequest,
    PaginatedMeta,
    PlanCloneRequest,
    PlanDetailResponse,
    PlanEntry,
    PlanListResponse,
    PlanStatusUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plans", tags=["Remediation Plans"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _plan_dict_to_entry(p: Dict[str, Any]) -> PlanEntry:
    """Convert a plan dictionary to PlanEntry schema."""
    return PlanEntry(
        plan_id=p.get("plan_id", ""),
        operator_id=p.get("operator_id", ""),
        supplier_id=p.get("supplier_id"),
        plan_name=p.get("plan_name", ""),
        status=p.get("status", "draft"),
        plan_template=p.get("plan_template"),
        budget_allocated=Decimal(str(p.get("budget_allocated", 0))),
        budget_spent=Decimal(str(p.get("budget_spent", 0))),
        start_date=p.get("start_date"),
        target_end_date=p.get("target_end_date"),
        milestone_count=p.get("milestone_count", 0),
        milestones_completed=p.get("milestones_completed", 0),
        progress_pct=Decimal(str(p.get("progress_pct", 0))),
        version=p.get("version", 1),
        created_at=p.get("created_at"),
        updated_at=p.get("updated_at"),
    )


# ---------------------------------------------------------------------------
# POST /plans
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=PlanDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new remediation plan",
    description=(
        "Generate a structured, multi-phase remediation plan based on selected "
        "strategies. Supports 8 plan templates: supplier capacity building, "
        "emergency deforestation response, certification enrollment, enhanced "
        "monitoring deployment, FPIC remediation, legal gap closure, anti-corruption "
        "measures, and buffer zone restoration. Each plan includes SMART milestones, "
        "KPIs, responsible parties, and escalation triggers per ISO 31000."
    ),
    responses={
        201: {"description": "Plan created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid plan parameters"},
    },
)
async def create_plan(
    request: Request,
    body: CreatePlanRequest,
    user: AuthUser = Depends(
        require_permission("eudr-rma:plans:write")
    ),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> PlanDetailResponse:
    """Create a new remediation plan from selected strategies.

    Args:
        body: Plan creation request with strategies, template, and budget.
        user: Authenticated user.
        service: RMA service singleton.

    Returns:
        PlanDetailResponse with full plan structure.
    """
    start = time.monotonic()

    try:
        from greenlang.agents.eudr.risk_mitigation_advisor.models import (
            CreatePlanRequest as EngineRequest,
        )

        engine_request = EngineRequest(
            operator_id=body.operator_id,
            supplier_id=body.supplier_id,
            plan_name=body.plan_name,
            strategy_ids=body.strategy_ids,
            risk_finding_ids=body.risk_finding_ids,
            template=body.template,
            budget_allocated=body.budget_allocated,
            start_date=body.start_date,
            responsible_parties=body.responsible_parties,
            commodity=body.commodity,
        )

        result = await service.create_plan(engine_request)

        plan_data = result if isinstance(result, dict) else {}
        plan_entry = _plan_dict_to_entry(plan_data)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Plan created: plan_id=%s user=%s template=%s elapsed_ms=%d",
            plan_entry.plan_id, user.user_id, body.template, elapsed_ms,
        )

        return PlanDetailResponse(
            plan=plan_entry,
            phases=plan_data.get("phases", []),
            milestones=plan_data.get("milestones", []),
            kpis=plan_data.get("kpis", []),
            responsible_parties=plan_data.get("responsible_parties", []),
            escalation_triggers=plan_data.get("escalation_triggers", []),
            strategy_ids=plan_data.get("strategy_ids", body.strategy_ids),
            risk_finding_ids=plan_data.get("risk_finding_ids", body.risk_finding_ids),
            provenance_hash=plan_data.get("provenance_hash", ""),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Plan creation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Plan creation processing error",
        )


# ---------------------------------------------------------------------------
# GET /plans
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=PlanListResponse,
    summary="List remediation plans",
    description=(
        "Retrieve a paginated list of remediation plans with optional "
        "filters for status, supplier, commodity, and template."
    ),
    responses={
        200: {"description": "Plans listed successfully"},
    },
)
async def list_plans(
    request: Request,
    plan_status: Optional[str] = Query(None, alias="status", description="Filter by plan status"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    commodity: Optional[str] = Query(None, description="Filter by EUDR commodity"),
    template: Optional[str] = Query(None, description="Filter by plan template"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-rma:plans:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> PlanListResponse:
    """List remediation plans with filters and pagination."""
    try:
        result = await service.list_plans(
            operator_id=user.operator_id,
            status=plan_status,
            supplier_id=supplier_id,
            commodity=commodity,
            template=template,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        plans_raw = result.get("plans", []) if isinstance(result, dict) else []
        total = result.get("total", 0) if isinstance(result, dict) else 0
        plans = [_plan_dict_to_entry(p) for p in plans_raw]

        return PlanListResponse(
            plans=plans,
            meta=PaginatedMeta(
                total=total, limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as e:
        logger.error("Plan list failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve plans")


# ---------------------------------------------------------------------------
# GET /plans/{plan_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plan_id}",
    response_model=PlanDetailResponse,
    summary="Get plan detail with milestones and progress",
    description="Retrieve full details of a remediation plan including phases, milestones, KPIs, and progress metrics.",
    responses={
        200: {"description": "Plan details retrieved"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def get_plan_detail(
    request: Request,
    plan_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> PlanDetailResponse:
    """Get full plan details with milestones and progress."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.get_plan(plan_id, operator_id=user.operator_id)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")

        plan_data = result if isinstance(result, dict) else {}
        return PlanDetailResponse(
            plan=_plan_dict_to_entry(plan_data),
            phases=plan_data.get("phases", []),
            milestones=plan_data.get("milestones", []),
            kpis=plan_data.get("kpis", []),
            responsible_parties=plan_data.get("responsible_parties", []),
            escalation_triggers=plan_data.get("escalation_triggers", []),
            strategy_ids=plan_data.get("strategy_ids", []),
            risk_finding_ids=plan_data.get("risk_finding_ids", []),
            provenance_hash=plan_data.get("provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plan detail failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve plan")


# ---------------------------------------------------------------------------
# PUT /plans/{plan_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{plan_id}",
    response_model=PlanDetailResponse,
    summary="Update plan details",
    description="Update remediation plan details including budget, dates, responsible parties, and escalation triggers.",
    responses={
        200: {"description": "Plan updated successfully"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def update_plan(
    request: Request,
    plan_id: str,
    body: CreatePlanRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:write")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> PlanDetailResponse:
    """Update an existing remediation plan."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.update_plan(
            plan_id=plan_id,
            updates=body.model_dump(exclude_none=True),
            operator_id=user.operator_id,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")

        plan_data = result if isinstance(result, dict) else {}
        logger.info("Plan updated: plan_id=%s user=%s", plan_id, user.user_id)

        return PlanDetailResponse(
            plan=_plan_dict_to_entry(plan_data),
            phases=plan_data.get("phases", []),
            milestones=plan_data.get("milestones", []),
            kpis=plan_data.get("kpis", []),
            responsible_parties=plan_data.get("responsible_parties", []),
            escalation_triggers=plan_data.get("escalation_triggers", []),
            strategy_ids=plan_data.get("strategy_ids", []),
            risk_finding_ids=plan_data.get("risk_finding_ids", []),
            provenance_hash=plan_data.get("provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plan update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update plan")


# ---------------------------------------------------------------------------
# PUT /plans/{plan_id}/status
# ---------------------------------------------------------------------------


@router.put(
    "/{plan_id}/status",
    response_model=PlanEntry,
    summary="Update plan status",
    description=(
        "Transition plan status: draft -> active (requires approval), "
        "active -> on_track/at_risk/delayed/completed/suspended/abandoned. "
        "Activation and completion require eudr-rma:plans:approve permission."
    ),
    responses={
        200: {"description": "Status updated"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
        409: {"model": ErrorResponse, "description": "Invalid status transition"},
    },
)
async def update_plan_status(
    request: Request,
    plan_id: str,
    body: PlanStatusUpdateRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:approve")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> PlanEntry:
    """Update plan lifecycle status."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.update_plan_status(
            plan_id=plan_id,
            new_status=body.status,
            reason=body.reason,
            approved_by=body.approved_by or user.user_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")

        plan_data = result if isinstance(result, dict) else {}
        logger.info("Plan status updated: plan_id=%s status=%s user=%s", plan_id, body.status, user.user_id)
        return _plan_dict_to_entry(plan_data)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error("Plan status update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update plan status")


# ---------------------------------------------------------------------------
# POST /plans/{plan_id}/clone
# ---------------------------------------------------------------------------


@router.post(
    "/{plan_id}/clone",
    response_model=PlanDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Clone plan for another supplier",
    description="Clone a successful remediation plan as a template for another supplier, preserving structure and milestones.",
    responses={
        201: {"description": "Plan cloned successfully"},
        404: {"model": ErrorResponse, "description": "Source plan not found"},
    },
)
async def clone_plan(
    request: Request,
    plan_id: str,
    body: PlanCloneRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:write")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> PlanDetailResponse:
    """Clone plan for another supplier."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.clone_plan(
            source_plan_id=plan_id,
            target_supplier_id=body.target_supplier_id,
            new_plan_name=body.new_plan_name,
            start_date=body.start_date,
            budget_allocated=body.budget_allocated,
            operator_id=user.operator_id,
            cloned_by=user.user_id,
        )

        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source plan {plan_id} not found")

        plan_data = result if isinstance(result, dict) else {}
        logger.info("Plan cloned: source=%s target_supplier=%s user=%s", plan_id, body.target_supplier_id, user.user_id)

        return PlanDetailResponse(
            plan=_plan_dict_to_entry(plan_data),
            phases=plan_data.get("phases", []),
            milestones=plan_data.get("milestones", []),
            kpis=plan_data.get("kpis", []),
            responsible_parties=plan_data.get("responsible_parties", []),
            escalation_triggers=plan_data.get("escalation_triggers", []),
            strategy_ids=plan_data.get("strategy_ids", []),
            risk_finding_ids=plan_data.get("risk_finding_ids", []),
            provenance_hash=plan_data.get("provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plan clone failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clone plan")


# ---------------------------------------------------------------------------
# GET /plans/{plan_id}/gantt
# ---------------------------------------------------------------------------


@router.get(
    "/{plan_id}/gantt",
    response_model=GanttChartResponse,
    summary="Get Gantt chart data with dependencies",
    description="Retrieve Gantt chart timeline data with phase durations, milestone dates, dependency graph, and critical path analysis.",
    responses={
        200: {"description": "Gantt chart data retrieved"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def get_gantt_chart(
    request: Request,
    plan_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> GanttChartResponse:
    """Get Gantt chart data for a plan."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.get_gantt_data(plan_id, operator_id=user.operator_id)

        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")

        data = result if isinstance(result, dict) else {}
        return GanttChartResponse(
            plan_id=plan_id,
            plan_name=data.get("plan_name", ""),
            phases=data.get("phases", []),
            milestones=data.get("milestones", []),
            dependencies=data.get("dependencies", []),
            critical_path=data.get("critical_path", []),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Gantt chart retrieval failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve Gantt data")


# ---------------------------------------------------------------------------
# POST /plans/{plan_id}/milestones
# ---------------------------------------------------------------------------


@router.post(
    "/{plan_id}/milestones",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Add milestone to plan",
    description="Add a new SMART milestone to a remediation plan with due date, KPI target, and evidence requirements.",
    responses={
        201: {"description": "Milestone created"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def create_milestone(
    request: Request,
    plan_id: str,
    body: MilestoneCreateRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:write")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> Dict[str, Any]:
    """Add a milestone to an existing plan."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.add_milestone(
            plan_id=plan_id,
            milestone_data=body.model_dump(),
            operator_id=user.operator_id,
            created_by=user.user_id,
        )

        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan {plan_id} not found")

        logger.info("Milestone created: plan_id=%s milestone=%s user=%s", plan_id, body.name, user.user_id)
        return result if isinstance(result, dict) else {"milestone_id": "", "status": "created"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Milestone creation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create milestone")


# ---------------------------------------------------------------------------
# PUT /plans/{plan_id}/milestones/{milestone_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{plan_id}/milestones/{milestone_id}",
    response_model=Dict[str, Any],
    summary="Update milestone status and evidence",
    description="Update milestone status, completion date, actual KPI value, and notes.",
    responses={
        200: {"description": "Milestone updated"},
        404: {"model": ErrorResponse, "description": "Plan or milestone not found"},
    },
)
async def update_milestone(
    request: Request,
    plan_id: str,
    milestone_id: str,
    body: MilestoneUpdateRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:write")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> Dict[str, Any]:
    """Update a milestone within a plan."""
    validate_uuid(plan_id, "plan_id")
    validate_uuid(milestone_id, "milestone_id")

    try:
        result = await service.update_milestone(
            plan_id=plan_id,
            milestone_id=milestone_id,
            updates=body.model_dump(exclude_none=True),
            operator_id=user.operator_id,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Milestone {milestone_id} in plan {plan_id} not found",
            )

        logger.info("Milestone updated: plan_id=%s milestone_id=%s user=%s", plan_id, milestone_id, user.user_id)
        return result if isinstance(result, dict) else {"milestone_id": milestone_id, "status": "updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Milestone update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update milestone")


# ---------------------------------------------------------------------------
# POST /plans/{plan_id}/milestones/{milestone_id}/evidence
# ---------------------------------------------------------------------------


@router.post(
    "/{plan_id}/milestones/{milestone_id}/evidence",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Upload evidence for milestone",
    description=(
        "Register evidence document for a milestone. Evidence is stored in S3 "
        "with SHA-256 integrity hash. Supports documents, photos, certificates, "
        "audit reports, satellite imagery, and GPS data per EUDR Article 31."
    ),
    responses={
        201: {"description": "Evidence registered"},
        404: {"model": ErrorResponse, "description": "Plan or milestone not found"},
    },
)
async def upload_evidence(
    request: Request,
    plan_id: str,
    milestone_id: str,
    body: EvidenceUploadRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:plans:write")),
    _rate: None = Depends(rate_limit_upload),
    service: Any = Depends(get_rma_service),
) -> Dict[str, Any]:
    """Register evidence document for a plan milestone."""
    validate_uuid(plan_id, "plan_id")
    validate_uuid(milestone_id, "milestone_id")

    try:
        result = await service.upload_evidence(
            plan_id=plan_id,
            milestone_id=milestone_id,
            evidence_data=body.model_dump(),
            operator_id=user.operator_id,
            uploaded_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Milestone {milestone_id} in plan {plan_id} not found",
            )

        logger.info(
            "Evidence uploaded: plan_id=%s milestone_id=%s file=%s user=%s",
            plan_id, milestone_id, body.file_name, user.user_id,
        )
        return result if isinstance(result, dict) else {"evidence_id": "", "status": "uploaded"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Evidence upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload evidence")
