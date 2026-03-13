# -*- coding: utf-8 -*-
"""
Capacity Building Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for supplier capacity building program management including
enrollment, progress tracking, competency assessment, tier advancement,
and supplier scorecard generation.

Endpoints (5):
    POST /capacity-building/enroll                                - Enroll supplier
    GET  /capacity-building/enrollments                           - List enrollments
    GET  /capacity-building/enrollments/{enrollment_id}           - Get enrollment detail
    PUT  /capacity-building/enrollments/{enrollment_id}/progress  - Update progress
    GET  /capacity-building/scorecard/{supplier_id}               - Get supplier scorecard

RBAC Permissions:
    eudr-rma:capacity:read   - View enrollments and scorecards
    eudr-rma:capacity:manage - Enroll suppliers, update progress

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 3: Capacity Building Manager
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_rma_service,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    EnrollmentEntry,
    EnrollmentListResponse,
    EnrollSupplierRequest,
    ErrorResponse,
    PaginatedMeta,
    ProgressUpdateRequest,
    ScorecardResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/capacity-building", tags=["Capacity Building"])


def _enrollment_dict_to_entry(e: Dict[str, Any]) -> EnrollmentEntry:
    """Convert enrollment dictionary to EnrollmentEntry schema."""
    modules_completed = e.get("modules_completed", 0)
    modules_total = e.get("modules_total", 22)
    completion_pct = (
        Decimal(str(modules_completed)) / Decimal(str(max(modules_total, 1))) * Decimal("100")
    ).quantize(Decimal("0.01")) if modules_total > 0 else Decimal("0")

    risk_at = e.get("risk_score_at_enrollment")
    risk_now = e.get("current_risk_score")
    risk_reduction = None
    if risk_at is not None and risk_now is not None:
        try:
            risk_reduction = Decimal(str(risk_at)) - Decimal(str(risk_now))
        except Exception:
            risk_reduction = None

    return EnrollmentEntry(
        enrollment_id=e.get("enrollment_id", ""),
        supplier_id=e.get("supplier_id", ""),
        program_id=e.get("program_id", ""),
        commodity=e.get("commodity", ""),
        current_tier=e.get("current_tier", 1),
        modules_completed=modules_completed,
        modules_total=modules_total,
        completion_pct=completion_pct,
        competency_scores=e.get("competency_scores", {}),
        enrolled_date=e.get("enrolled_date"),
        target_completion_date=e.get("target_completion_date"),
        status=e.get("status", "active"),
        risk_score_at_enrollment=Decimal(str(risk_at)) if risk_at is not None else None,
        current_risk_score=Decimal(str(risk_now)) if risk_now is not None else None,
        risk_reduction_pct=risk_reduction,
    )


# ---------------------------------------------------------------------------
# POST /capacity-building/enroll
# ---------------------------------------------------------------------------


@router.post(
    "/enroll",
    response_model=EnrollmentEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll supplier in capacity building program",
    description=(
        "Enroll a supplier in a structured capacity building program with "
        "4 tiers: Tier 1 (Awareness), Tier 2 (Basic Compliance), Tier 3 "
        "(Advanced Practices), Tier 4 (Leadership). Each tier has commodity-specific "
        "training modules (22 per commodity for 7 EUDR commodities)."
    ),
    responses={
        201: {"description": "Supplier enrolled successfully"},
        400: {"model": ErrorResponse, "description": "Invalid enrollment parameters"},
        409: {"model": ErrorResponse, "description": "Supplier already enrolled in this program"},
    },
)
async def enroll_supplier(
    request: Request,
    body: EnrollSupplierRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:capacity:manage")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> EnrollmentEntry:
    """Enroll a supplier in a capacity building program."""
    try:
        from greenlang.agents.eudr.risk_mitigation_advisor.models import (
            EnrollSupplierRequest as EngineRequest,
        )

        engine_request = EngineRequest(
            supplier_id=body.supplier_id,
            commodity=body.commodity,
            program_id=body.program_id,
            initial_tier=body.initial_tier,
            target_completion_date=body.target_completion_date,
            risk_score_at_enrollment=body.risk_score_at_enrollment,
        )

        result = await service.enroll_supplier(engine_request)
        enrollment_data = result if isinstance(result, dict) else {}

        logger.info(
            "Supplier enrolled: supplier_id=%s commodity=%s tier=%d user=%s",
            body.supplier_id, body.commodity, body.initial_tier, user.user_id,
        )
        return _enrollment_dict_to_entry(enrollment_data)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Supplier enrollment failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Enrollment processing error")


# ---------------------------------------------------------------------------
# GET /capacity-building/enrollments
# ---------------------------------------------------------------------------


@router.get(
    "/enrollments",
    response_model=EnrollmentListResponse,
    summary="List capacity building enrollments",
    description="Retrieve a paginated list of capacity building enrollments with optional filters.",
    responses={200: {"description": "Enrollments listed"}},
)
async def list_enrollments(
    request: Request,
    enrollment_status: Optional[str] = Query(None, alias="status", description="Filter by status: active, paused, completed, withdrawn"),
    commodity: Optional[str] = Query(None, description="Filter by EUDR commodity"),
    tier: Optional[int] = Query(None, ge=1, le=4, description="Filter by current tier"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-rma:capacity:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> EnrollmentListResponse:
    """List capacity building enrollments."""
    try:
        result = await service.list_enrollments(
            operator_id=user.operator_id,
            status=enrollment_status,
            commodity=commodity,
            tier=tier,
            supplier_id=supplier_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        enrollments_raw = result.get("enrollments", []) if isinstance(result, dict) else []
        total = result.get("total", 0) if isinstance(result, dict) else 0
        enrollments = [_enrollment_dict_to_entry(e) for e in enrollments_raw]

        return EnrollmentListResponse(
            enrollments=enrollments,
            meta=PaginatedMeta(
                total=total, limit=pagination.limit, offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as e:
        logger.error("Enrollment list failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve enrollments")


# ---------------------------------------------------------------------------
# GET /capacity-building/enrollments/{enrollment_id}
# ---------------------------------------------------------------------------


@router.get(
    "/enrollments/{enrollment_id}",
    response_model=EnrollmentEntry,
    summary="Get enrollment progress detail",
    description="Retrieve detailed progress for a capacity building enrollment including competency scores and tier advancement eligibility.",
    responses={
        200: {"description": "Enrollment detail retrieved"},
        404: {"model": ErrorResponse, "description": "Enrollment not found"},
    },
)
async def get_enrollment_detail(
    request: Request,
    enrollment_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:capacity:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> EnrollmentEntry:
    """Get enrollment detail."""
    validate_uuid(enrollment_id, "enrollment_id")

    try:
        result = await service.get_enrollment(enrollment_id, operator_id=user.operator_id)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Enrollment {enrollment_id} not found")

        return _enrollment_dict_to_entry(result if isinstance(result, dict) else {})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Enrollment detail failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve enrollment")


# ---------------------------------------------------------------------------
# PUT /capacity-building/enrollments/{enrollment_id}/progress
# ---------------------------------------------------------------------------


@router.put(
    "/enrollments/{enrollment_id}/progress",
    response_model=EnrollmentEntry,
    summary="Update module completion and competency",
    description="Update supplier progress through the capacity building program including modules completed, competency scores, and current risk score.",
    responses={
        200: {"description": "Progress updated"},
        404: {"model": ErrorResponse, "description": "Enrollment not found"},
    },
)
async def update_progress(
    request: Request,
    enrollment_id: str,
    body: ProgressUpdateRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:capacity:manage")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> EnrollmentEntry:
    """Update enrollment progress."""
    validate_uuid(enrollment_id, "enrollment_id")

    try:
        result = await service.update_enrollment_progress(
            enrollment_id=enrollment_id,
            updates=body.model_dump(exclude_none=True),
            operator_id=user.operator_id,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Enrollment {enrollment_id} not found")

        logger.info("Enrollment progress updated: enrollment_id=%s user=%s", enrollment_id, user.user_id)
        return _enrollment_dict_to_entry(result if isinstance(result, dict) else {})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Progress update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update progress")


# ---------------------------------------------------------------------------
# GET /capacity-building/scorecard/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/scorecard/{supplier_id}",
    response_model=ScorecardResponse,
    summary="Get supplier capacity building scorecard",
    description=(
        "Retrieve a comprehensive capacity building scorecard for a supplier "
        "showing all enrollments, tier progress, competency scores, risk "
        "improvement, and tier advancement eligibility."
    ),
    responses={
        200: {"description": "Scorecard generated"},
        404: {"model": ErrorResponse, "description": "Supplier not found or no enrollments"},
    },
)
async def get_scorecard(
    request: Request,
    supplier_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:capacity:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> ScorecardResponse:
    """Get supplier capacity building scorecard."""
    try:
        result = await service.get_supplier_scorecard(
            supplier_id=supplier_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No capacity building data found for supplier {supplier_id}",
            )

        data = result if isinstance(result, dict) else {}
        enrollments = [_enrollment_dict_to_entry(e) for e in data.get("enrollments", [])]

        return ScorecardResponse(
            supplier_id=supplier_id,
            enrollments=enrollments,
            overall_tier=data.get("overall_tier", 1),
            overall_completion_pct=Decimal(str(data.get("overall_completion_pct", 0))),
            risk_score_improvement=Decimal(str(data["risk_score_improvement"])) if data.get("risk_score_improvement") is not None else None,
            tier_advancement_eligible=data.get("tier_advancement_eligible", False),
            recommendations=data.get("recommendations", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Scorecard generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate scorecard")
