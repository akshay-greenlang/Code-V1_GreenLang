# -*- coding: utf-8 -*-
"""
Community Consultation Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for recording and tracking community consultation activities
including public hearings, community meetings, field visits, and
traditional assemblies. Consultations are a key component of FPIC
compliance and due diligence under EUDR.

Endpoints:
    POST /consultations                        - Record consultation activity
    GET  /consultations                        - List consultations with filters
    GET  /consultations/{consultation_id}      - Get consultation details

Consultation lifecycle: SCHEDULED -> IN_PROGRESS -> COMPLETED
Follow-up tracking and outcome documentation supported.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, ConsultationTracker Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.indigenous_rights_checker.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_consultation_tracker,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    ConsultationCreateRequest,
    ConsultationEntry,
    ConsultationListResponse,
    ConsultationResponse,
    ConsultationStatusEnum,
    ConsultationTypeEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    SortOrderEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consultations", tags=["Community Consultations"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _build_consultation_entry(entry: dict) -> ConsultationEntry:
    """Build a ConsultationEntry from engine result dictionary."""
    return ConsultationEntry(
        consultation_id=entry.get("consultation_id", ""),
        territory_id=entry.get("territory_id", ""),
        community_id=entry.get("community_id", ""),
        consultation_type=ConsultationTypeEnum(
            entry.get("consultation_type", "community_meeting")
        ),
        title=entry.get("title", ""),
        status=ConsultationStatusEnum(entry.get("status", "scheduled")),
        description=entry.get("description"),
        scheduled_date=entry.get("scheduled_date"),
        completed_date=entry.get("completed_date"),
        location=entry.get("location"),
        attendees=entry.get("attendees"),
        attendee_count=entry.get("attendee_count"),
        language=entry.get("language"),
        outcomes=entry.get("outcomes"),
        follow_up_actions=entry.get("follow_up_actions"),
        documents=entry.get("documents"),
        created_at=entry.get("created_at"),
        updated_at=entry.get("updated_at"),
    )


# ---------------------------------------------------------------------------
# POST /consultations
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ConsultationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record consultation activity",
    description=(
        "Record a community consultation activity such as a public hearing, "
        "community meeting, field visit, or traditional assembly. Links the "
        "consultation to a territory and community for FPIC tracking. "
        "Supports outcome and follow-up action documentation."
    ),
    responses={
        201: {"description": "Consultation recorded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_consultation(
    request: Request,
    body: ConsultationCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:consultations:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ConsultationResponse:
    """Record a new community consultation activity.

    Args:
        body: Consultation creation request.
        user: Authenticated user with consultations:create permission.

    Returns:
        ConsultationResponse with the created consultation record.
    """
    start = time.monotonic()

    try:
        engine = get_consultation_tracker()
        result = engine.record_consultation(
            territory_id=body.territory_id,
            community_id=body.community_id,
            consultation_type=body.consultation_type.value,
            title=body.title,
            description=body.description,
            scheduled_date=str(body.scheduled_date) if body.scheduled_date else None,
            completed_date=str(body.completed_date) if body.completed_date else None,
            location=body.location,
            attendees=body.attendees,
            language=body.language,
            outcomes=body.outcomes,
            follow_up_actions=body.follow_up_actions,
            documents=body.documents,
            created_by=user.user_id,
        )

        consultation_data = result.get("consultation", {})
        consultation_entry = _build_consultation_entry(consultation_data)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"record_consultation:{body.territory_id}:{body.community_id}:{body.consultation_type.value}",
            consultation_entry.consultation_id,
        )

        logger.info(
            "Consultation recorded: id=%s territory=%s community=%s type=%s operator=%s",
            consultation_entry.consultation_id,
            body.territory_id,
            body.community_id,
            body.consultation_type.value,
            user.operator_id or user.user_id,
        )

        return ConsultationResponse(
            consultation=consultation_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "ConsultationTracker"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Consultation recording failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consultation recording failed",
        )


# ---------------------------------------------------------------------------
# GET /consultations
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ConsultationListResponse,
    summary="List consultations",
    description=(
        "Retrieve a paginated list of community consultations with optional "
        "filters for territory, community, consultation type, status, and "
        "date range. Results ordered by scheduled date descending."
    ),
    responses={
        200: {"description": "Consultations listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_consultations(
    request: Request,
    territory_id: Optional[str] = Query(
        None, description="Filter by territory ID"
    ),
    community_id: Optional[str] = Query(
        None, description="Filter by community ID"
    ),
    consultation_type: Optional[ConsultationTypeEnum] = Query(
        None, description="Filter by consultation type"
    ),
    consultation_status: Optional[ConsultationStatusEnum] = Query(
        None, alias="status", description="Filter by consultation status"
    ),
    search: Optional[str] = Query(
        None, max_length=200, description="Search by title"
    ),
    sort_by: Optional[str] = Query(
        "scheduled_date",
        description="Sort field (scheduled_date, created_at, title)",
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.DESC, description="Sort order"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:consultations:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ConsultationListResponse:
    """List consultations with optional filters and pagination.

    Args:
        territory_id: Optional territory filter.
        community_id: Optional community filter.
        consultation_type: Optional type filter.
        consultation_status: Optional status filter.
        search: Optional title search.
        sort_by: Sort field.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        ConsultationListResponse with paginated list.
    """
    start = time.monotonic()

    try:
        engine = get_consultation_tracker()
        result = engine.list_consultations(
            territory_id=territory_id,
            community_id=community_id,
            consultation_type=consultation_type.value if consultation_type else None,
            status=consultation_status.value if consultation_status else None,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order.value if sort_order else "desc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        consultations = [
            _build_consultation_entry(entry)
            for entry in result.get("consultations", [])
        ]
        total = result.get("total", len(consultations))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_consultations:{territory_id}:{community_id}:{consultation_type}:{consultation_status}",
            str(total),
        )

        logger.info(
            "Consultations listed: total=%d returned=%d operator=%s",
            total,
            len(consultations),
            user.operator_id or user.user_id,
        )

        return ConsultationListResponse(
            consultations=consultations,
            total_consultations=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "ConsultationTracker"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Consultation listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consultation listing failed",
        )


# ---------------------------------------------------------------------------
# GET /consultations/{consultation_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{consultation_id}",
    response_model=ConsultationResponse,
    summary="Get consultation details",
    description=(
        "Retrieve detailed information for a specific consultation activity "
        "including attendees, outcomes, follow-up actions, and associated "
        "documents."
    ),
    responses={
        200: {"description": "Consultation details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Consultation not found"},
    },
)
async def get_consultation(
    consultation_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:consultations:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ConsultationResponse:
    """Get detailed information for a specific consultation.

    Args:
        consultation_id: Unique consultation identifier.
        user: Authenticated user.

    Returns:
        ConsultationResponse with full consultation details.
    """
    start = time.monotonic()

    try:
        engine = get_consultation_tracker()
        result = engine.get_consultation(consultation_id=consultation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Consultation not found: {consultation_id}",
            )

        consultation_data = result.get("consultation", {})
        consultation_entry = _build_consultation_entry(consultation_data)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"get_consultation:{consultation_id}",
            consultation_entry.title,
        )

        logger.info(
            "Consultation retrieved: id=%s territory=%s operator=%s",
            consultation_id,
            consultation_entry.territory_id,
            user.operator_id or user.user_id,
        )

        return ConsultationResponse(
            consultation=consultation_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "ConsultationTracker"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Consultation retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consultation retrieval failed",
        )
