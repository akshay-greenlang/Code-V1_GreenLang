# -*- coding: utf-8 -*-
"""
Audit Planning Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for risk-based audit schedule generation, audit calendar management,
and audit CRUD operations per EUDR Articles 10-11, 14-16.

Endpoints (9):
    POST /audits/schedule/generate           - Generate risk-based audit schedule
    GET  /audits/schedule                    - Get audit calendar (paginated)
    PUT  /audits/schedule/{schedule_id}      - Update scheduled audit
    POST /audits/schedule/trigger            - Trigger unscheduled audit
    POST /audits                             - Create a new audit
    GET  /audits                             - List audits (paginated)
    GET  /audits/{audit_id}                  - Get audit details
    PUT  /audits/{audit_id}                  - Update audit
    DELETE /audits/{audit_id}                - Cancel audit

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024, AuditPlanningSchedulingEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_planning_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_date_range,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    AuditCreateRequest,
    AuditCreateResponse,
    AuditDetailResponse,
    AuditEntry,
    AuditListResponse,
    AuditStatusEnum,
    AuditTypeEnum,
    AuditUpdateRequest,
    CertSchemeEnum,
    ErrorResponse,
    EUDRCommodityEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    ScheduleEntry,
    ScheduleGenerateRequest,
    ScheduleGenerateResponse,
    ScheduleListResponse,
    ScheduleStatusEnum,
    ScheduleUpdateRequest,
    TriggerAuditRequest,
    TriggerAuditResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Audit Planning"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /audits/schedule/generate
# ---------------------------------------------------------------------------


@router.post(
    "/audits/schedule/generate",
    response_model=ScheduleGenerateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate risk-based audit schedule",
    description=(
        "Generate a risk-based audit schedule for an operator using composite "
        "priority scoring from country risk (EUDR-016), supplier risk (EUDR-017), "
        "NC history, certification gaps, and deforestation alerts (EUDR-020)."
    ),
    responses={
        201: {"description": "Schedule generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_schedule(
    request: Request,
    body: ScheduleGenerateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:schedule")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> ScheduleGenerateResponse:
    """Generate risk-based audit schedule for an operator.

    Args:
        body: Schedule generation request with operator_id and options.
        user: Authenticated user with audits:schedule permission.

    Returns:
        ScheduleGenerateResponse with scheduled audit entries.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.generate_schedule(
            operator_id=body.operator_id,
            planning_quarter=body.planning_quarter,
            force_recalculate=body.force_recalculate,
            supplier_ids=body.supplier_ids,
        )

        entries = []
        for entry_data in result.get("entries", []):
            entries.append(ScheduleEntry(
                schedule_id=entry_data.get("schedule_id", ""),
                operator_id=body.operator_id,
                supplier_id=entry_data.get("supplier_id", ""),
                planned_quarter=entry_data.get("planned_quarter", ""),
                audit_type=AuditTypeEnum(entry_data.get("audit_type", "full")),
                priority_score=Decimal(str(entry_data.get("priority_score", 0))),
                risk_factors=entry_data.get("risk_factors", {}),
            ))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"schedule_generate:{body.operator_id}",
            f"entries:{len(entries)}",
        )

        logger.info(
            "Schedule generated: operator=%s entries=%d user=%s",
            body.operator_id,
            len(entries),
            user.user_id,
        )

        return ScheduleGenerateResponse(
            schedule_entries=entries,
            total_scheduled=len(entries),
            high_priority_count=result.get("high_priority_count", 0),
            standard_priority_count=result.get("standard_priority_count", 0),
            low_priority_count=result.get("low_priority_count", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Schedule generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Schedule generation failed",
        )


# ---------------------------------------------------------------------------
# GET /audits/schedule
# ---------------------------------------------------------------------------


@router.get(
    "/audits/schedule",
    response_model=ScheduleListResponse,
    summary="Get audit calendar",
    description=(
        "Retrieve a paginated list of scheduled audits with optional "
        "filtering by date range, supplier, and certification scheme."
    ),
    responses={
        200: {"description": "Schedule retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_schedule(
    request: Request,
    supplier_id: Optional[str] = Query(None, description="Filter by supplier"),
    scheme: Optional[str] = Query(None, description="Filter by scheme"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    date_range: Dict = Depends(validate_date_range),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ScheduleListResponse:
    """Retrieve paginated audit schedule.

    Args:
        supplier_id: Optional supplier filter.
        scheme: Optional certification scheme filter.
        status_filter: Optional status filter.
        date_range: Optional date range filter.
        pagination: Pagination parameters.
        user: Authenticated user with audits:read permission.

    Returns:
        ScheduleListResponse with schedule entries and pagination.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.list_schedule(
            supplier_id=supplier_id,
            scheme=scheme,
            status=status_filter,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
            limit=pagination.limit,
            offset=pagination.offset,
        )

        entries = []
        for entry_data in result.get("schedules", []):
            entries.append(ScheduleEntry(
                schedule_id=entry_data.get("schedule_id", ""),
                operator_id=entry_data.get("operator_id", ""),
                supplier_id=entry_data.get("supplier_id", ""),
                planned_quarter=entry_data.get("planned_quarter", ""),
                audit_type=AuditTypeEnum(entry_data.get("audit_type", "full")),
                priority_score=Decimal(str(entry_data.get("priority_score", 0))),
            ))

        total = result.get("total", len(entries))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return ScheduleListResponse(
            schedules=entries,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("schedule_list", f"total:{total}"),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Schedule list failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit schedule",
        )


# ---------------------------------------------------------------------------
# PUT /audits/schedule/{schedule_id}
# ---------------------------------------------------------------------------


@router.put(
    "/audits/schedule/{schedule_id}",
    response_model=ScheduleGenerateResponse,
    summary="Update scheduled audit",
    description="Update a scheduled audit entry (reschedule, reassign, cancel).",
    responses={
        200: {"description": "Schedule updated"},
        404: {"model": ErrorResponse, "description": "Schedule entry not found"},
    },
)
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:schedule")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ScheduleGenerateResponse:
    """Update an existing schedule entry.

    Args:
        schedule_id: UUID of the schedule entry.
        body: Update request with optional fields.
        user: Authenticated user with audits:schedule permission.

    Returns:
        Updated schedule response.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.update_schedule(
            schedule_id=schedule_id,
            updates=body.model_dump(exclude_none=True),
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schedule entry not found: {schedule_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"schedule_update:{schedule_id}",
            f"user:{user.user_id}",
        )

        logger.info(
            "Schedule updated: id=%s user=%s", schedule_id, user.user_id
        )

        return ScheduleGenerateResponse(
            schedule_entries=[],
            total_scheduled=0,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Schedule update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Schedule update failed",
        )


# ---------------------------------------------------------------------------
# POST /audits/schedule/trigger
# ---------------------------------------------------------------------------


@router.post(
    "/audits/schedule/trigger",
    response_model=TriggerAuditResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger unscheduled audit",
    description=(
        "Trigger an unscheduled audit based on events such as deforestation "
        "alerts, certification suspensions, or competent authority requests."
    ),
    responses={
        201: {"description": "Unscheduled audit triggered"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def trigger_audit(
    request: Request,
    body: TriggerAuditRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:schedule")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TriggerAuditResponse:
    """Trigger an unscheduled audit based on risk events.

    Args:
        body: Trigger request with supplier, reason, and scope.
        user: Authenticated user with audits:schedule permission.

    Returns:
        TriggerAuditResponse with new audit and schedule IDs.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.trigger_unscheduled_audit(
            operator_id=body.operator_id,
            supplier_id=body.supplier_id,
            trigger_reason=body.trigger_reason,
            audit_type=body.audit_type.value,
            modality=body.modality.value,
            scope=body.scope.value,
            country_code=body.country_code.upper(),
            commodity=body.commodity.value,
            triggered_by=user.user_id,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"trigger:{body.supplier_id}:{body.trigger_reason}",
            result.get("audit_id", ""),
        )

        logger.info(
            "Unscheduled audit triggered: supplier=%s reason=%s user=%s",
            body.supplier_id,
            body.trigger_reason,
            user.user_id,
        )

        return TriggerAuditResponse(
            audit_id=result.get("audit_id", ""),
            schedule_id=result.get("schedule_id"),
            status=result.get("status", "scheduled"),
            priority_score=Decimal(str(result.get("priority_score", 0))),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit trigger failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit trigger failed",
        )


# ---------------------------------------------------------------------------
# POST /audits
# ---------------------------------------------------------------------------


@router.post(
    "/audits",
    response_model=AuditCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new audit",
    description="Create a new audit record for an operator and supplier.",
    responses={
        201: {"description": "Audit created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def create_audit(
    request: Request,
    body: AuditCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AuditCreateResponse:
    """Create a new audit.

    Args:
        body: Audit creation request.
        user: Authenticated user with audits:write permission.

    Returns:
        AuditCreateResponse with the created audit record.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.create_audit(
            operator_id=body.operator_id,
            supplier_id=body.supplier_id,
            audit_type=body.audit_type.value,
            modality=body.modality.value,
            certification_scheme=body.certification_scheme.value if body.certification_scheme else None,
            planned_date=str(body.planned_date),
            country_code=body.country_code.upper(),
            commodity=body.commodity.value,
            eudr_articles=body.eudr_articles,
            site_ids=body.site_ids,
            trigger_reason=body.trigger_reason,
            created_by=user.user_id,
        )

        audit = AuditEntry(
            audit_id=result.get("audit_id", ""),
            operator_id=body.operator_id,
            supplier_id=body.supplier_id,
            audit_type=body.audit_type,
            modality=body.modality,
            certification_scheme=body.certification_scheme.value if body.certification_scheme else None,
            planned_date=body.planned_date,
            country_code=body.country_code.upper(),
            commodity=body.commodity.value,
            provenance_hash=result.get("provenance_hash", ""),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"audit_create:{body.operator_id}:{body.supplier_id}",
            audit.audit_id,
        )

        logger.info(
            "Audit created: id=%s operator=%s supplier=%s user=%s",
            audit.audit_id,
            body.operator_id,
            body.supplier_id,
            user.user_id,
        )

        return AuditCreateResponse(
            audit=audit,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit creation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit creation failed",
        )


# ---------------------------------------------------------------------------
# GET /audits
# ---------------------------------------------------------------------------


@router.get(
    "/audits",
    response_model=AuditListResponse,
    summary="List audits",
    description=(
        "Retrieve a paginated list of audits with optional filtering "
        "by status, supplier, scheme, and date range."
    ),
    responses={
        200: {"description": "Audits retrieved"},
    },
)
async def list_audits(
    request: Request,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier"),
    scheme: Optional[str] = Query(None, description="Filter by certification scheme"),
    country_code: Optional[str] = Query(None, description="Filter by country"),
    date_range: Dict = Depends(validate_date_range),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AuditListResponse:
    """Retrieve paginated audit list.

    Args:
        status_filter: Optional status filter.
        supplier_id: Optional supplier filter.
        scheme: Optional scheme filter.
        country_code: Optional country filter.
        date_range: Optional date range filter.
        pagination: Pagination parameters.
        user: Authenticated user with audits:read permission.

    Returns:
        AuditListResponse with audit entries and pagination.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.list_audits(
            status=status_filter,
            supplier_id=supplier_id,
            scheme=scheme,
            country_code=country_code.upper() if country_code else None,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
            limit=pagination.limit,
            offset=pagination.offset,
        )

        audits = []
        for audit_data in result.get("audits", []):
            audits.append(AuditEntry(
                audit_id=audit_data.get("audit_id", ""),
                operator_id=audit_data.get("operator_id", ""),
                supplier_id=audit_data.get("supplier_id", ""),
                audit_type=AuditTypeEnum(audit_data.get("audit_type", "full")),
                planned_date=audit_data.get("planned_date", "2026-01-01"),
                status=AuditStatusEnum(audit_data.get("status", "planned")),
                country_code=audit_data.get("country_code", ""),
                commodity=audit_data.get("commodity", ""),
                priority_score=Decimal(str(audit_data.get("priority_score", 0))),
                provenance_hash=audit_data.get("provenance_hash", ""),
            ))

        total = result.get("total", len(audits))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return AuditListResponse(
            audits=audits,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("audit_list", f"total:{total}"),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit list failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audits",
        )


# ---------------------------------------------------------------------------
# GET /audits/{audit_id}
# ---------------------------------------------------------------------------


@router.get(
    "/audits/{audit_id}",
    response_model=AuditDetailResponse,
    summary="Get audit details",
    description="Retrieve full details of a specific audit including team and checklists.",
    responses={
        200: {"description": "Audit details retrieved"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def get_audit(
    audit_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AuditDetailResponse:
    """Retrieve detailed audit information.

    Args:
        audit_id: UUID of the audit.
        user: Authenticated user with audits:read permission.

    Returns:
        AuditDetailResponse with full audit details.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.get_audit(audit_id=audit_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        audit = AuditEntry(
            audit_id=result.get("audit_id", audit_id),
            operator_id=result.get("operator_id", ""),
            supplier_id=result.get("supplier_id", ""),
            audit_type=AuditTypeEnum(result.get("audit_type", "full")),
            planned_date=result.get("planned_date", "2026-01-01"),
            status=AuditStatusEnum(result.get("status", "planned")),
            country_code=result.get("country_code", ""),
            commodity=result.get("commodity", ""),
            provenance_hash=result.get("provenance_hash", ""),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return AuditDetailResponse(
            audit=audit,
            team_assignments=result.get("team_assignments", []),
            checklists=result.get("checklists", []),
            recent_ncs=result.get("recent_ncs", []),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(f"audit_detail:{audit_id}", audit.status.value),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit detail failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit details",
        )


# ---------------------------------------------------------------------------
# PUT /audits/{audit_id}
# ---------------------------------------------------------------------------


@router.put(
    "/audits/{audit_id}",
    response_model=AuditCreateResponse,
    summary="Update audit",
    description="Update audit status, team, dates, or modality.",
    responses={
        200: {"description": "Audit updated"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def update_audit(
    audit_id: str,
    body: AuditUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AuditCreateResponse:
    """Update an existing audit.

    Args:
        audit_id: UUID of the audit.
        body: Update request with optional fields.
        user: Authenticated user with audits:write permission.

    Returns:
        Updated audit response.
    """
    start = time.monotonic()

    try:
        engine = get_planning_engine()
        result = engine.update_audit(
            audit_id=audit_id,
            updates=body.model_dump(exclude_none=True),
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        audit = AuditEntry(
            audit_id=result.get("audit_id", audit_id),
            operator_id=result.get("operator_id", ""),
            supplier_id=result.get("supplier_id", ""),
            audit_type=AuditTypeEnum(result.get("audit_type", "full")),
            planned_date=result.get("planned_date", "2026-01-01"),
            status=AuditStatusEnum(result.get("status", "planned")),
            country_code=result.get("country_code", ""),
            commodity=result.get("commodity", ""),
            provenance_hash=result.get("provenance_hash", ""),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"audit_update:{audit_id}", f"user:{user.user_id}"
        )

        logger.info("Audit updated: id=%s user=%s", audit_id, user.user_id)

        return AuditCreateResponse(
            audit=audit,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit update failed",
        )


# ---------------------------------------------------------------------------
# DELETE /audits/{audit_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/audits/{audit_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel audit",
    description="Cancel a planned or in-progress audit.",
    responses={
        204: {"description": "Audit cancelled"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def cancel_audit(
    audit_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-tam:audits:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> None:
    """Cancel an audit.

    Args:
        audit_id: UUID of the audit.
        user: Authenticated user with audits:write permission.
    """
    try:
        engine = get_planning_engine()
        result = engine.cancel_audit(
            audit_id=audit_id,
            cancelled_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        logger.info("Audit cancelled: id=%s user=%s", audit_id, user.user_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit cancel failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit cancellation failed",
        )
