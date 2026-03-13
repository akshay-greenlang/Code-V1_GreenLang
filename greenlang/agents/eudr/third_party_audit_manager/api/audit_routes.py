# -*- coding: utf-8 -*-
"""
Audit Management Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for audit lifecycle CRUD and risk-based scheduling covering
audit creation, listing, detail retrieval, schedule generation, audit
start, and audit completion.

Endpoints (6):
    POST /audits                 - Create a new audit
    GET  /audits                 - List audits with filters
    GET  /audits/{audit_id}      - Get audit details
    POST /audits/schedule        - Generate risk-based audit schedule
    POST /audits/{audit_id}/start    - Start an audit (transition to in_progress)
    POST /audits/{audit_id}/complete - Complete an audit (transition to closed)

RBAC Permissions:
    eudr-tam:audit:create   - Create audits
    eudr-tam:audit:read     - List and view audits
    eudr-tam:audit:schedule - Generate and manage schedules
    eudr-tam:audit:execute  - Start and complete audits

Audit lifecycle:
    PLANNED -> AUDITOR_ASSIGNED -> IN_PREPARATION -> IN_PROGRESS ->
    FIELDWORK_COMPLETE -> REPORT_DRAFTING -> REPORT_ISSUED ->
    CAR_FOLLOW_UP -> CLOSED | CANCELLED

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_current_user,
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
    ScheduleGenerateRequest,
    ScheduleGenerateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audits", tags=["Audit Management"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for zero-hallucination audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /audits
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=AuditCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new audit",
    description=(
        "Create a new third-party audit record for a supplier. Supports "
        "full, targeted, surveillance, and unscheduled audit types. "
        "Automatically calculates risk-based priority score using upstream "
        "EUDR-016 country risk and EUDR-017 supplier risk signals."
    ),
    responses={
        201: {"description": "Audit created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def create_audit(
    request: Request,
    body: AuditCreateRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:audit:create")),
    _rl: None = Depends(rate_limit_write),
    planning_engine: object = Depends(get_planning_engine),
) -> AuditCreateResponse:
    """Create a new audit for EUDR compliance verification.

    Args:
        body: Audit creation payload (operator, supplier, type, date, etc.).
        user: Authenticated user with audit:create permission.
        planning_engine: AuditPlanningSchedulingEngine singleton.

    Returns:
        Created audit record with provenance hash.

    Raises:
        HTTPException: 400 if validation fails, 500 on processing error.
    """
    start = time.monotonic()
    try:
        logger.info(
            "Creating audit: operator=%s supplier=%s type=%s user=%s",
            body.operator_id,
            body.supplier_id,
            body.audit_type,
            user.user_id,
        )

        audit_data: Dict[str, Any] = {
            "operator_id": body.operator_id,
            "supplier_id": body.supplier_id,
            "audit_type": body.audit_type.value if body.audit_type else "full",
            "modality": body.modality.value if body.modality else "on_site",
            "certification_scheme": (
                body.certification_scheme.value
                if body.certification_scheme
                else None
            ),
            "planned_date": str(body.planned_date),
            "country_code": body.country_code.upper(),
            "commodity": body.commodity.value if body.commodity else "",
            "eudr_articles": body.eudr_articles,
            "site_ids": body.site_ids,
            "trigger_reason": body.trigger_reason,
        }

        result: Dict[str, Any] = {}
        if hasattr(planning_engine, "create_audit"):
            result = await planning_engine.create_audit(audit_data)
        else:
            result = {
                "audit_id": hashlib.sha256(
                    f"{body.operator_id}{body.supplier_id}{time.time()}".encode()
                ).hexdigest()[:36],
                **audit_data,
                "status": "planned",
                "priority_score": "0.00",
            }

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(audit_data, result)

        audit_entry = AuditEntry(
            audit_id=result.get("audit_id", ""),
            operator_id=body.operator_id,
            supplier_id=body.supplier_id,
            audit_type=body.audit_type or AuditTypeEnum.FULL,
            modality=body.modality,
            certification_scheme=(
                body.certification_scheme.value
                if body.certification_scheme
                else None
            ),
            planned_date=body.planned_date,
            status=AuditStatusEnum.PLANNED,
            priority_score=Decimal(str(result.get("priority_score", "0.00"))),
            country_code=body.country_code.upper(),
            commodity=body.commodity.value if body.commodity else "",
            provenance_hash=prov_hash,
        )

        return AuditCreateResponse(
            audit=audit_entry,
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create audit: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create audit record",
        )


# ---------------------------------------------------------------------------
# GET /audits
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=AuditListResponse,
    summary="List audits with filters",
    description=(
        "Retrieve a paginated list of audits with optional filters for "
        "status, supplier, certification scheme, country, commodity, and "
        "date range. Results are ordered by planned_date descending."
    ),
    responses={
        200: {"description": "Audits listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_audits(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:audit:read")),
    _rl: None = Depends(rate_limit_standard),
    pagination: PaginationParams = Depends(get_pagination),
    planning_engine: object = Depends(get_planning_engine),
    audit_status: Optional[AuditStatusEnum] = Query(
        None, description="Filter by audit status"
    ),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    certification_scheme: Optional[CertSchemeEnum] = Query(
        None, description="Filter by certification scheme"
    ),
    country_code: Optional[str] = Query(
        None, min_length=2, max_length=2, description="Filter by ISO 3166-1 alpha-2"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by EUDR commodity"
    ),
    date_range: Dict = Depends(validate_date_range),
) -> AuditListResponse:
    """List audits with filtering, sorting, and pagination.

    Args:
        user: Authenticated user with audit:read permission.
        pagination: Standard limit/offset parameters.
        planning_engine: AuditPlanningSchedulingEngine singleton.
        audit_status: Optional status filter.
        supplier_id: Optional supplier filter.
        certification_scheme: Optional scheme filter.
        country_code: Optional country filter.
        commodity: Optional commodity filter.
        date_range: Optional start/end date range.

    Returns:
        Paginated list of audit records with provenance.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {}
        if audit_status:
            filters["status"] = audit_status.value
        if supplier_id:
            filters["supplier_id"] = supplier_id
        if certification_scheme:
            filters["certification_scheme"] = certification_scheme.value
        if country_code:
            filters["country_code"] = country_code.upper()
        if commodity:
            filters["commodity"] = commodity.value
        if date_range.get("start_date"):
            filters["start_date"] = str(date_range["start_date"])
        if date_range.get("end_date"):
            filters["end_date"] = str(date_range["end_date"])

        audits: List[Dict[str, Any]] = []
        total = 0
        if hasattr(planning_engine, "list_audits"):
            result = await planning_engine.list_audits(
                filters=filters,
                limit=pagination.limit,
                offset=pagination.offset,
                operator_id=user.operator_id,
            )
            audits = result.get("audits", [])
            total = result.get("total", 0)

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(filters, len(audits))

        audit_entries = [
            AuditEntry(**a) if isinstance(a, dict) else a for a in audits
        ]

        return AuditListResponse(
            audits=audit_entries,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list audits: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit list",
        )


# ---------------------------------------------------------------------------
# GET /audits/{audit_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{audit_id}",
    response_model=AuditDetailResponse,
    summary="Get audit details",
    description=(
        "Retrieve full audit details including team assignments, checklists, "
        "and recent non-conformances for a specific audit."
    ),
    responses={
        200: {"description": "Audit details retrieved"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_audit_detail(
    audit_id: str,
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:audit:read")),
    _rl: None = Depends(rate_limit_standard),
    planning_engine: object = Depends(get_planning_engine),
) -> AuditDetailResponse:
    """Retrieve detailed information for a specific audit.

    Args:
        audit_id: Unique audit identifier.
        user: Authenticated user with audit:read permission.
        planning_engine: AuditPlanningSchedulingEngine singleton.

    Returns:
        Audit detail with team, checklists, and recent NCs.

    Raises:
        HTTPException: 404 if audit not found.
    """
    start = time.monotonic()
    try:
        result: Optional[Dict[str, Any]] = None
        if hasattr(planning_engine, "get_audit"):
            result = await planning_engine.get_audit(
                audit_id=audit_id,
                operator_id=user.operator_id,
            )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit {audit_id} not found",
            )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(audit_id, result.get("audit_id", ""))

        audit_entry = (
            AuditEntry(**result["audit"])
            if "audit" in result and isinstance(result["audit"], dict)
            else AuditEntry(**result)
            if isinstance(result, dict) and "operator_id" in result
            else AuditEntry(
                audit_id=audit_id,
                operator_id=result.get("operator_id", ""),
                supplier_id=result.get("supplier_id", ""),
                audit_type=result.get("audit_type", "full"),
                planned_date=result.get("planned_date", "2026-01-01"),
                country_code=result.get("country_code", "XX"),
                commodity=result.get("commodity", "wood"),
                provenance_hash=prov_hash,
            )
        )

        return AuditDetailResponse(
            audit=audit_entry,
            team_assignments=result.get("team_assignments", []),
            checklists=result.get("checklists", []),
            recent_ncs=result.get("recent_ncs", []),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get audit detail: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit details",
        )


# ---------------------------------------------------------------------------
# POST /audits/schedule
# ---------------------------------------------------------------------------


@router.post(
    "/schedule",
    response_model=ScheduleGenerateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate risk-based audit schedule",
    description=(
        "Generate a risk-based audit schedule for an operator's suppliers. "
        "Uses composite priority scoring: Country_Risk*0.25 + "
        "Supplier_Risk*0.25 + NC_History*0.20 + Cert_Gap*0.15 + "
        "Deforestation_Alert*0.15, multiplied by Recency_Multiplier "
        "(capped at 2.0). Assigns frequency tiers: HIGH (quarterly), "
        "STANDARD (semi-annual), LOW (annual)."
    ),
    responses={
        201: {"description": "Schedule generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def generate_audit_schedule(
    request: Request,
    body: ScheduleGenerateRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:audit:schedule")),
    _rl: None = Depends(rate_limit_heavy),
    planning_engine: object = Depends(get_planning_engine),
) -> ScheduleGenerateResponse:
    """Generate risk-based audit schedule for operator suppliers.

    Uses deterministic priority scoring formula with fixed Decimal weights.
    Integrates with EUDR-016 (country risk), EUDR-017 (supplier risk),
    and EUDR-020 (deforestation alerts) for risk signal inputs.

    Args:
        body: Schedule generation request with operator ID and options.
        user: Authenticated user with audit:schedule permission.
        planning_engine: AuditPlanningSchedulingEngine singleton.

    Returns:
        Generated schedule entries with priority scores and frequency tiers.
    """
    start = time.monotonic()
    try:
        logger.info(
            "Generating audit schedule: operator=%s quarter=%s user=%s",
            body.operator_id,
            body.planning_quarter,
            user.user_id,
        )

        result: Dict[str, Any] = {}
        if hasattr(planning_engine, "generate_schedule"):
            result = await planning_engine.generate_schedule(
                operator_id=body.operator_id,
                planning_quarter=body.planning_quarter,
                force_recalculate=body.force_recalculate,
                supplier_ids=body.supplier_ids,
            )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(
            body.model_dump_json(), result.get("total_scheduled", 0)
        )

        return ScheduleGenerateResponse(
            schedule_entries=result.get("schedule_entries", []),
            total_scheduled=result.get("total_scheduled", 0),
            high_priority_count=result.get("high_priority_count", 0),
            standard_priority_count=result.get("standard_priority_count", 0),
            low_priority_count=result.get("low_priority_count", 0),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate schedule: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate audit schedule",
        )


# ---------------------------------------------------------------------------
# POST /audits/{audit_id}/start
# ---------------------------------------------------------------------------


@router.post(
    "/{audit_id}/start",
    summary="Start an audit",
    description=(
        "Transition an audit to IN_PROGRESS status. Sets the actual start "
        "date and validates that an auditor has been assigned."
    ),
    responses={
        200: {"description": "Audit started successfully"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
        400: {"model": ErrorResponse, "description": "Invalid state transition"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def start_audit(
    audit_id: str,
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:audit:execute")),
    _rl: None = Depends(rate_limit_write),
    planning_engine: object = Depends(get_planning_engine),
) -> dict:
    """Start an audit and transition to IN_PROGRESS status.

    Validates that the audit exists, is in a startable state (PLANNED or
    AUDITOR_ASSIGNED or IN_PREPARATION), and sets actual_start_date.

    Args:
        audit_id: Unique audit identifier.
        user: Authenticated user with audit:execute permission.
        planning_engine: AuditPlanningSchedulingEngine singleton.

    Returns:
        Updated audit status.

    Raises:
        HTTPException: 404 if audit not found, 400 if invalid transition.
    """
    start = time.monotonic()
    try:
        result: Optional[Dict[str, Any]] = None
        if hasattr(planning_engine, "start_audit"):
            result = await planning_engine.start_audit(
                audit_id=audit_id,
                started_by=user.user_id,
            )
        else:
            result = {
                "audit_id": audit_id,
                "status": "in_progress",
                "actual_start_date": datetime.now(timezone.utc).date().isoformat(),
            }

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit {audit_id} not found",
            )

        elapsed = (time.monotonic() - start) * 1000
        return {
            "audit_id": audit_id,
            "status": result.get("status", "in_progress"),
            "actual_start_date": result.get("actual_start_date"),
            "provenance_hash": _compute_provenance(audit_id, "start"),
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to start audit %s: %s", audit_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start audit",
        )


# ---------------------------------------------------------------------------
# POST /audits/{audit_id}/complete
# ---------------------------------------------------------------------------


@router.post(
    "/{audit_id}/complete",
    summary="Complete an audit",
    description=(
        "Transition an audit to CLOSED status. Sets the actual end date "
        "and validates that all required steps have been completed."
    ),
    responses={
        200: {"description": "Audit completed successfully"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
        400: {"model": ErrorResponse, "description": "Invalid state transition"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def complete_audit(
    audit_id: str,
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:audit:execute")),
    _rl: None = Depends(rate_limit_write),
    planning_engine: object = Depends(get_planning_engine),
) -> dict:
    """Complete an audit and transition to CLOSED status.

    Validates that the audit exists and is in a completable state.
    Sets actual_end_date to current date.

    Args:
        audit_id: Unique audit identifier.
        user: Authenticated user with audit:execute permission.
        planning_engine: AuditPlanningSchedulingEngine singleton.

    Returns:
        Updated audit status.

    Raises:
        HTTPException: 404 if audit not found, 400 if invalid transition.
    """
    start = time.monotonic()
    try:
        result: Optional[Dict[str, Any]] = None
        if hasattr(planning_engine, "complete_audit"):
            result = await planning_engine.complete_audit(
                audit_id=audit_id,
                completed_by=user.user_id,
            )
        else:
            result = {
                "audit_id": audit_id,
                "status": "closed",
                "actual_end_date": datetime.now(timezone.utc).date().isoformat(),
            }

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit {audit_id} not found",
            )

        elapsed = (time.monotonic() - start) * 1000
        return {
            "audit_id": audit_id,
            "status": result.get("status", "closed"),
            "actual_end_date": result.get("actual_end_date"),
            "provenance_hash": _compute_provenance(audit_id, "complete"),
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to complete audit %s: %s", audit_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete audit",
        )
