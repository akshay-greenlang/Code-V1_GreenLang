# -*- coding: utf-8 -*-
"""
Red Flag Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for red flag detection, listing, detail retrieval, and
suppression of false positives across operators, suppliers, documents,
and certifications per EUDR Articles 9, 10, 11.

Endpoints:
    POST /red-flags/detect              - Detect red flags for an entity
    GET  /red-flags                     - List red flags (paginated)
    GET  /red-flags/{flag_id}           - Get red flag details
    PUT  /red-flags/{flag_id}/suppress  - Suppress false positive

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, RedFlagDetectionEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_red_flag_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    EUDRCommodityEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RedFlagCategoryEnum,
    RedFlagDetectRequest,
    RedFlagDetectResponse,
    RedFlagDetailResponse,
    RedFlagEntry,
    RedFlagListResponse,
    RedFlagSeverityEnum,
    RedFlagStatusEnum,
    RedFlagSuppressRequest,
    RedFlagSuppressResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/red-flags", tags=["Red Flags"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /red-flags/detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=RedFlagDetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect red flags for an entity",
    description=(
        "Screen an operator, supplier, or set of documents/certifications "
        "for compliance red flags including document forgery, expired "
        "certifications, sanctioned entities, high-risk country origin, "
        "deforestation links, and regulatory violations."
    ),
    responses={
        200: {"description": "Red flag detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_red_flags(
    request: Request,
    body: RedFlagDetectRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:red-flag:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> RedFlagDetectResponse:
    """Detect red flags for an entity or set of documents.

    Args:
        body: Red flag detection request.
        user: Authenticated user with red-flag:create permission.

    Returns:
        RedFlagDetectResponse with detected red flags.
    """
    start = time.monotonic()

    try:
        engine = get_red_flag_engine()
        result = engine.detect(
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            document_ids=body.document_ids,
            certification_ids=body.certification_ids,
            country_code=body.country_code,
            commodity=body.commodity.value if body.commodity else None,
            categories=[c.value for c in body.categories] if body.categories else None,
            include_cross_references=body.include_cross_references,
            detected_by=user.user_id,
        )

        red_flags = []
        for f in result.get("red_flags", []):
            red_flags.append(
                RedFlagEntry(
                    flag_id=f.get("flag_id", ""),
                    category=RedFlagCategoryEnum(f.get("category", "other")),
                    severity=RedFlagSeverityEnum(f.get("severity", "medium")),
                    status=RedFlagStatusEnum(f.get("status", "active")),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    entity_type=f.get("entity_type", ""),
                    entity_id=f.get("entity_id", ""),
                    country_code=f.get("country_code"),
                    commodity=EUDRCommodityEnum(f["commodity"]) if f.get("commodity") else None,
                    regulatory_reference=f.get("regulatory_reference"),
                )
            )

        critical_count = sum(
            1 for f in red_flags if f.severity == RedFlagSeverityEnum.CRITICAL
        )
        high_count = sum(
            1 for f in red_flags if f.severity == RedFlagSeverityEnum.HIGH
        )
        requires_action = critical_count > 0

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"red_flag_detect:{body.operator_id}:{body.supplier_id}",
            str(len(red_flags)),
        )

        logger.info(
            "Red flags detected: total=%d critical=%d high=%d "
            "operator=%s supplier=%s user=%s",
            len(red_flags),
            critical_count,
            high_count,
            body.operator_id,
            body.supplier_id,
            user.user_id,
        )

        return RedFlagDetectResponse(
            red_flags=red_flags,
            total_flags=len(red_flags),
            critical_count=critical_count,
            high_count=high_count,
            requires_immediate_action=requires_action,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "RedFlagDetectionEngine",
                    "Sanctions Database",
                    "Country Risk Database",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Red flag detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Red flag detection failed",
        )


# ---------------------------------------------------------------------------
# GET /red-flags
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=RedFlagListResponse,
    summary="List red flags",
    description=(
        "Retrieve a paginated list of red flags with optional filtering "
        "by category, severity, status, entity type, and commodity."
    ),
    responses={
        200: {"description": "Red flags retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_red_flags(
    request: Request,
    category: Optional[RedFlagCategoryEnum] = Query(
        None, description="Filter by red flag category"
    ),
    severity: Optional[RedFlagSeverityEnum] = Query(
        None, description="Filter by severity"
    ),
    flag_status: Optional[RedFlagStatusEnum] = Query(
        None, alias="status", description="Filter by status"
    ),
    entity_type: Optional[str] = Query(
        None, description="Filter by entity type (operator, supplier, document)"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by commodity"
    ),
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    supplier_id: Optional[str] = Query(
        None, description="Filter by supplier ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:red-flag:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RedFlagListResponse:
    """List red flags with pagination and filtering.

    Args:
        category: Optional category filter.
        severity: Optional severity filter.
        flag_status: Optional status filter.
        entity_type: Optional entity type filter.
        commodity: Optional commodity filter.
        operator_id: Optional operator filter.
        supplier_id: Optional supplier filter.
        pagination: Pagination parameters.
        user: Authenticated user with red-flag:read permission.

    Returns:
        RedFlagListResponse with paginated red flags.
    """
    start = time.monotonic()

    try:
        engine = get_red_flag_engine()
        result = engine.list_flags(
            category=category.value if category else None,
            severity=severity.value if severity else None,
            status=flag_status.value if flag_status else None,
            entity_type=entity_type,
            commodity=commodity.value if commodity else None,
            operator_id=operator_id,
            supplier_id=supplier_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        red_flags = []
        for f in result.get("red_flags", []):
            red_flags.append(
                RedFlagEntry(
                    flag_id=f.get("flag_id", ""),
                    category=RedFlagCategoryEnum(f.get("category", "other")),
                    severity=RedFlagSeverityEnum(f.get("severity", "medium")),
                    status=RedFlagStatusEnum(f.get("status", "active")),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    entity_type=f.get("entity_type", ""),
                    entity_id=f.get("entity_id", ""),
                    country_code=f.get("country_code"),
                    commodity=EUDRCommodityEnum(f["commodity"]) if f.get("commodity") else None,
                    regulatory_reference=f.get("regulatory_reference"),
                    suppressed=f.get("suppressed", False),
                )
            )

        total = result.get("total", len(red_flags))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"red_flag_list:{category}:{severity}:{flag_status}",
            str(total),
        )

        logger.info(
            "Red flags listed: total=%d user=%s",
            total,
            user.user_id,
        )

        return RedFlagListResponse(
            red_flags=red_flags,
            total_flags=total,
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
                data_sources=["RedFlagDetectionEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Red flag listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Red flag listing failed",
        )


# ---------------------------------------------------------------------------
# GET /red-flags/{flag_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{flag_id}",
    response_model=RedFlagDetailResponse,
    summary="Get red flag details",
    description=(
        "Retrieve full details of a red flag including evidence, "
        "related flags, recommended actions, and audit trail."
    ),
    responses={
        200: {"description": "Red flag details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Red flag not found"},
    },
)
async def get_red_flag_detail(
    flag_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:red-flag:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RedFlagDetailResponse:
    """Get detailed information about a red flag.

    Args:
        flag_id: Unique red flag identifier.
        user: Authenticated user with red-flag:read permission.

    Returns:
        RedFlagDetailResponse with full red flag details.
    """
    start = time.monotonic()

    try:
        engine = get_red_flag_engine()
        result = engine.get_detail(flag_id=flag_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Red flag not found: {flag_id}",
            )

        red_flag = RedFlagEntry(
            flag_id=result.get("flag_id", flag_id),
            category=RedFlagCategoryEnum(result.get("category", "other")),
            severity=RedFlagSeverityEnum(result.get("severity", "medium")),
            status=RedFlagStatusEnum(result.get("status", "active")),
            title=result.get("title", ""),
            description=result.get("description", ""),
            entity_type=result.get("entity_type", ""),
            entity_id=result.get("entity_id", ""),
            country_code=result.get("country_code"),
            commodity=EUDRCommodityEnum(result["commodity"]) if result.get("commodity") else None,
            regulatory_reference=result.get("regulatory_reference"),
            suppressed=result.get("suppressed", False),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"red_flag_detail:{flag_id}",
            red_flag.title,
        )

        logger.info(
            "Red flag detail retrieved: id=%s user=%s",
            flag_id,
            user.user_id,
        )

        return RedFlagDetailResponse(
            red_flag=red_flag,
            evidence=result.get("evidence", []),
            related_flags=result.get("related_flags", []),
            recommended_actions=result.get("recommended_actions", []),
            audit_trail=result.get("audit_trail", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["RedFlagDetectionEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Red flag detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Red flag detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# PUT /red-flags/{flag_id}/suppress
# ---------------------------------------------------------------------------


@router.put(
    "/{flag_id}/suppress",
    response_model=RedFlagSuppressResponse,
    summary="Suppress red flag as false positive",
    description=(
        "Suppress a red flag identified as a false positive. Requires a "
        "justification reason (minimum 10 characters) and optionally "
        "evidence URLs. Creates an audit trail entry for compliance records."
    ),
    responses={
        200: {"description": "Red flag suppressed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Red flag not found"},
    },
)
async def suppress_red_flag(
    flag_id: str,
    request: Request,
    body: RedFlagSuppressRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:red-flag:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> RedFlagSuppressResponse:
    """Suppress a red flag as false positive.

    Args:
        flag_id: Red flag to suppress.
        body: Suppression request with reason and evidence.
        user: Authenticated user with red-flag:update permission.

    Returns:
        RedFlagSuppressResponse with suppression details.
    """
    start = time.monotonic()

    try:
        engine = get_red_flag_engine()
        result = engine.suppress(
            flag_id=flag_id,
            reason=body.reason,
            evidence_urls=body.evidence_urls,
            reviewed_by=body.reviewed_by or user.user_id,
            suppressed_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Red flag not found: {flag_id}",
            )

        previous_status = RedFlagStatusEnum(
            result.get("previous_status", "active")
        )
        new_status = RedFlagStatusEnum(
            result.get("new_status", "suppressed")
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"red_flag_suppress:{flag_id}",
            new_status.value,
        )

        logger.info(
            "Red flag suppressed: id=%s previous=%s new=%s reason=%s user=%s",
            flag_id,
            previous_status.value,
            new_status.value,
            body.reason[:50],
            user.user_id,
        )

        return RedFlagSuppressResponse(
            flag_id=flag_id,
            previous_status=previous_status,
            new_status=new_status,
            suppressed_by=user.user_id,
            reason=body.reason,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["RedFlagDetectionEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Red flag suppression failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Red flag suppression failed",
        )
