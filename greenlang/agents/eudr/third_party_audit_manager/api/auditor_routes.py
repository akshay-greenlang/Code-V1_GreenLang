# -*- coding: utf-8 -*-
"""
Auditor Registry Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for auditor profile management and competence-based matching
per ISO/IEC 17065:2012 and ISO/IEC 17021-1:2015 requirements.

Endpoints (5):
    POST /auditors                      - Register a new auditor
    GET  /auditors                      - List auditors with filters
    GET  /auditors/{auditor_id}         - Get auditor profile
    POST /auditors/match                - Match auditors to audit requirements
    POST /auditors/{auditor_id}/qualification - Update auditor qualification

RBAC Permissions:
    eudr-tam:auditor:create  - Register new auditors
    eudr-tam:auditor:read    - View auditor profiles
    eudr-tam:auditor:match   - Execute auditor matching queries
    eudr-tam:auditor:update  - Update auditor qualifications

Auditor matching uses deterministic weighted scoring:
    Commodity competence (30) + Scheme qualification (25) +
    Country expertise (20) + Language match (15) +
    Performance rating (10) = 100 max

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_auditor_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    AccreditationStatusEnum,
    AuditorDetailResponse,
    AuditorListResponse,
    AuditorMatchRequest,
    AuditorMatchResponse,
    AuditorRegisterRequest,
    AuditorRegisterResponse,
    CertSchemeEnum,
    ErrorResponse,
    EUDRCommodityEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auditors", tags=["Auditor Registry"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /auditors
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=AuditorRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new auditor",
    description=(
        "Register a new auditor in the qualification registry. Tracks "
        "accreditation status, commodity competencies, scheme qualifications, "
        "country expertise, language proficiency, and conflict-of-interest "
        "declarations per ISO/IEC 17065 and ISO/IEC 17021-1."
    ),
    responses={
        201: {"description": "Auditor registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def register_auditor(
    request: Request,
    body: AuditorRegisterRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:auditor:create")),
    _rl: None = Depends(rate_limit_write),
    auditor_engine: object = Depends(get_auditor_engine),
) -> AuditorRegisterResponse:
    """Register a new auditor in the qualification registry.

    Args:
        body: Auditor registration payload (name, org, qualifications).
        user: Authenticated user with auditor:create permission.
        auditor_engine: AuditorRegistryQualificationEngine singleton.

    Returns:
        Registered auditor profile with ID and provenance.
    """
    start = time.monotonic()
    try:
        logger.info(
            "Registering auditor: name=%s org=%s user=%s",
            body.full_name,
            body.organization,
            user.user_id,
        )

        auditor_data = body.model_dump()
        result: Dict[str, Any] = {}

        if hasattr(auditor_engine, "register_auditor"):
            result = await auditor_engine.register_auditor(auditor_data)
        else:
            result = {
                "auditor_id": hashlib.sha256(
                    f"{body.full_name}{body.organization}{time.time()}".encode()
                ).hexdigest()[:36],
                **auditor_data,
            }

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(body.full_name, result.get("auditor_id", ""))

        return AuditorRegisterResponse(
            auditor=result,
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to register auditor: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register auditor",
        )


# ---------------------------------------------------------------------------
# GET /auditors
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=AuditorListResponse,
    summary="List auditors with filters",
    description=(
        "Retrieve a paginated list of auditors with optional filters for "
        "accreditation status, certification scheme qualification, commodity "
        "competence, country expertise, and organization."
    ),
    responses={
        200: {"description": "Auditors listed successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_auditors(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:auditor:read")),
    _rl: None = Depends(rate_limit_standard),
    pagination: PaginationParams = Depends(get_pagination),
    auditor_engine: object = Depends(get_auditor_engine),
    accreditation_status: Optional[AccreditationStatusEnum] = Query(
        None, description="Filter by accreditation status"
    ),
    scheme: Optional[CertSchemeEnum] = Query(
        None, description="Filter by scheme qualification"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by commodity competence"
    ),
    country_code: Optional[str] = Query(
        None, min_length=2, max_length=2, description="Filter by country expertise"
    ),
    organization: Optional[str] = Query(
        None, description="Filter by organization name (partial match)"
    ),
) -> AuditorListResponse:
    """List auditors with optional filters and pagination.

    Args:
        user: Authenticated user with auditor:read permission.
        pagination: Standard limit/offset parameters.
        auditor_engine: AuditorRegistryQualificationEngine singleton.

    Returns:
        Paginated list of auditor profiles.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {}
        if accreditation_status:
            filters["accreditation_status"] = accreditation_status.value
        if scheme:
            filters["scheme"] = scheme.value
        if commodity:
            filters["commodity"] = commodity.value
        if country_code:
            filters["country_code"] = country_code.upper()
        if organization:
            filters["organization"] = organization

        auditors: List[Dict[str, Any]] = []
        total = 0
        if hasattr(auditor_engine, "list_auditors"):
            result = await auditor_engine.list_auditors(
                filters=filters,
                limit=pagination.limit,
                offset=pagination.offset,
            )
            auditors = result.get("auditors", [])
            total = result.get("total", 0)

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(filters, len(auditors))

        return AuditorListResponse(
            auditors=auditors,
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
        logger.exception("Failed to list auditors: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve auditor list",
        )


# ---------------------------------------------------------------------------
# GET /auditors/{auditor_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{auditor_id}",
    response_model=AuditorDetailResponse,
    summary="Get auditor profile",
    description=(
        "Retrieve detailed auditor profile including qualifications, "
        "performance metrics, audit history, and CPD compliance status."
    ),
    responses={
        200: {"description": "Auditor profile retrieved"},
        404: {"model": ErrorResponse, "description": "Auditor not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_auditor_detail(
    auditor_id: str,
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:auditor:read")),
    _rl: None = Depends(rate_limit_standard),
    auditor_engine: object = Depends(get_auditor_engine),
) -> AuditorDetailResponse:
    """Retrieve detailed auditor profile.

    Args:
        auditor_id: Unique auditor identifier.
        user: Authenticated user with auditor:read permission.
        auditor_engine: AuditorRegistryQualificationEngine singleton.

    Returns:
        Auditor profile with qualifications and performance data.

    Raises:
        HTTPException: 404 if auditor not found.
    """
    start = time.monotonic()
    try:
        result: Optional[Dict[str, Any]] = None
        if hasattr(auditor_engine, "get_auditor"):
            result = await auditor_engine.get_auditor(auditor_id=auditor_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Auditor {auditor_id} not found",
            )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(auditor_id, result.get("auditor_id", ""))

        return AuditorDetailResponse(
            auditor=result,
            audit_history=result.get("audit_history", []),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get auditor %s: %s", auditor_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve auditor profile",
        )


# ---------------------------------------------------------------------------
# POST /auditors/match
# ---------------------------------------------------------------------------


@router.post(
    "/match",
    response_model=AuditorMatchResponse,
    summary="Match auditors to audit requirements",
    description=(
        "Find the best-matching auditors for a specific audit based on "
        "deterministic weighted scoring: commodity competence (30), scheme "
        "qualification (25), country expertise (20), language match (15), "
        "performance rating (10). Disqualifies auditors with expired "
        "accreditation, CPD non-compliance, or conflict of interest."
    ),
    responses={
        200: {"description": "Auditor matches computed"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def match_auditors(
    request: Request,
    body: AuditorMatchRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:auditor:match")),
    _rl: None = Depends(rate_limit_standard),
    auditor_engine: object = Depends(get_auditor_engine),
) -> AuditorMatchResponse:
    """Match auditors to audit requirements using deterministic scoring.

    Args:
        body: Audit requirements (commodity, scheme, country, language, date).
        user: Authenticated user with auditor:match permission.
        auditor_engine: AuditorRegistryQualificationEngine singleton.

    Returns:
        Ranked list of matching auditors with match scores.
    """
    start = time.monotonic()
    try:
        logger.info(
            "Matching auditors: commodity=%s scheme=%s country=%s user=%s",
            body.commodity,
            body.scheme,
            body.country_code,
            user.user_id,
        )

        matches: List[Dict[str, Any]] = []
        if hasattr(auditor_engine, "match_auditors"):
            matches = await auditor_engine.match_auditors(
                requirements=body.model_dump()
            )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(body.model_dump_json(), len(matches))

        return AuditorMatchResponse(
            matches=matches,
            total_matches=len(matches),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to match auditors: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to match auditors",
        )


# ---------------------------------------------------------------------------
# POST /auditors/{auditor_id}/qualification
# ---------------------------------------------------------------------------


@router.post(
    "/{auditor_id}/qualification",
    summary="Update auditor qualification",
    description=(
        "Update an auditor's qualification record including accreditation "
        "renewal, new scheme qualifications, commodity competence additions, "
        "CPD hours, and performance rating updates."
    ),
    responses={
        200: {"description": "Qualification updated successfully"},
        404: {"model": ErrorResponse, "description": "Auditor not found"},
        400: {"model": ErrorResponse, "description": "Invalid qualification data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def update_auditor_qualification(
    auditor_id: str,
    request: Request,
    body: Dict[str, Any] = {},
    user: AuthUser = Depends(require_permission("eudr-tam:auditor:update")),
    _rl: None = Depends(rate_limit_write),
    auditor_engine: object = Depends(get_auditor_engine),
) -> dict:
    """Update auditor qualification records.

    Supports updating accreditation status/expiry, adding new scheme
    qualifications or commodity competencies, updating CPD hours,
    and adjusting performance ratings.

    Args:
        auditor_id: Unique auditor identifier.
        body: Qualification update payload.
        user: Authenticated user with auditor:update permission.
        auditor_engine: AuditorRegistryQualificationEngine singleton.

    Returns:
        Updated auditor qualification summary.

    Raises:
        HTTPException: 404 if auditor not found.
    """
    start = time.monotonic()
    try:
        result: Optional[Dict[str, Any]] = None
        if hasattr(auditor_engine, "update_qualification"):
            result = await auditor_engine.update_qualification(
                auditor_id=auditor_id,
                updates=body,
                updated_by=user.user_id,
            )
        else:
            result = {"auditor_id": auditor_id, "updated": True}

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Auditor {auditor_id} not found",
            )

        elapsed = (time.monotonic() - start) * 1000
        return {
            "auditor_id": auditor_id,
            "updated": True,
            "qualification_summary": result,
            "provenance_hash": _compute_provenance(auditor_id, str(body)),
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to update qualification for %s: %s", auditor_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update auditor qualification",
        )
