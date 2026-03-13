# -*- coding: utf-8 -*-
"""
Rights Violation Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for detecting, listing, viewing, and resolving indigenous
rights violations including unauthorized access, missing FPIC, expired
consent, land encroachment, cultural site damage, and other violation
types tracked for EUDR compliance and human rights due diligence.

Endpoints:
    POST /violations/detect                  - Detect rights violations
    GET  /violations                         - List violations with filters
    GET  /violations/{violation_id}          - Get violation details
    PUT  /violations/{violation_id}/resolve  - Mark violation resolved

Violation lifecycle: DETECTED -> CONFIRMED -> UNDER_INVESTIGATION ->
    REMEDIATION_PLANNED -> RESOLVED (or ESCALATED/DISMISSED)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, ViolationDetector Engine
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
    get_pagination,
    get_violation_detector,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    ComplianceStatusEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    SortOrderEnum,
    ViolationDetectRequest,
    ViolationDetectResponse,
    ViolationEntry,
    ViolationListResponse,
    ViolationResolveRequest,
    ViolationResolveResponse,
    ViolationResponse,
    ViolationSeverityEnum,
    ViolationStatusEnum,
    ViolationTypeEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/violations", tags=["Rights Violations"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _build_violation_entry(entry: dict) -> ViolationEntry:
    """Build a ViolationEntry from engine result dictionary."""
    return ViolationEntry(
        violation_id=entry.get("violation_id", ""),
        violation_type=ViolationTypeEnum(
            entry.get("violation_type", "missing_fpic")
        ),
        severity=ViolationSeverityEnum(entry.get("severity", "medium")),
        status=ViolationStatusEnum(entry.get("status", "detected")),
        plot_id=entry.get("plot_id"),
        territory_id=entry.get("territory_id"),
        community_id=entry.get("community_id"),
        supplier_id=entry.get("supplier_id"),
        description=entry.get("description", ""),
        evidence=entry.get("evidence"),
        remediation_actions=entry.get("remediation_actions"),
        detected_at=entry.get("detected_at"),
        resolved_at=entry.get("resolved_at"),
        resolution_notes=entry.get("resolution_notes"),
    )


# ---------------------------------------------------------------------------
# POST /violations/detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=ViolationDetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect rights violations",
    description=(
        "Detect indigenous rights violations for a plot, territory, or "
        "supplier. Checks for missing FPIC, expired consent, land "
        "encroachment, unauthorized access, cultural site impacts, and "
        "other violation types. Returns detected violations with severity, "
        "evidence references, and compliance impact assessment."
    ),
    responses={
        200: {"description": "Violation detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_violations(
    request: Request,
    body: ViolationDetectRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:violations:detect")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ViolationDetectResponse:
    """Detect rights violations for a plot, territory, or supplier.

    Args:
        body: Violation detection request.
        user: Authenticated user with violations:detect permission.

    Returns:
        ViolationDetectResponse with detected violations.
    """
    start = time.monotonic()

    try:
        # Validate at least one target is provided
        if not body.plot_id and not body.territory_id and not body.supplier_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of plot_id, territory_id, or supplier_id must be provided",
            )

        engine = get_violation_detector()
        result = engine.detect_violations(
            plot_id=body.plot_id,
            territory_id=body.territory_id,
            supplier_id=body.supplier_id,
            check_types=[ct.value for ct in body.check_types] if body.check_types else None,
            commodity=body.commodity.value if body.commodity else None,
            include_resolved=body.include_resolved,
            detected_by=user.user_id,
        )

        violations = [
            _build_violation_entry(entry)
            for entry in result.get("violations", [])
        ]
        critical_count = sum(
            1 for v in violations if v.severity == ViolationSeverityEnum.CRITICAL
        )
        high_count = sum(
            1 for v in violations if v.severity == ViolationSeverityEnum.HIGH
        )

        # Determine compliance impact
        compliance_impact = ComplianceStatusEnum.COMPLIANT
        if critical_count > 0:
            compliance_impact = ComplianceStatusEnum.NON_COMPLIANT
        elif high_count > 0:
            compliance_impact = ComplianceStatusEnum.AT_RISK
        elif violations:
            compliance_impact = ComplianceStatusEnum.REQUIRES_ASSESSMENT

        # Override with engine result if available
        if result.get("compliance_impact"):
            compliance_impact = ComplianceStatusEnum(result["compliance_impact"])

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"detect_violations:{body.plot_id}:{body.territory_id}:{body.supplier_id}",
            f"{len(violations)}:{critical_count}:{high_count}",
        )

        logger.info(
            "Violations detected: total=%d critical=%d high=%d impact=%s operator=%s",
            len(violations),
            critical_count,
            high_count,
            compliance_impact.value,
            user.operator_id or user.user_id,
        )

        return ViolationDetectResponse(
            violations=violations,
            total_violations=len(violations),
            critical_count=critical_count,
            high_count=high_count,
            compliance_impact=compliance_impact,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "ViolationDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Violation detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Violation detection failed",
        )


# ---------------------------------------------------------------------------
# GET /violations
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ViolationListResponse,
    summary="List violations",
    description=(
        "Retrieve a paginated list of rights violations with optional filters "
        "for violation type, severity, status, plot, territory, supplier, and "
        "community. Results ordered by detection date descending."
    ),
    responses={
        200: {"description": "Violations listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_violations(
    request: Request,
    violation_type: Optional[ViolationTypeEnum] = Query(
        None, description="Filter by violation type"
    ),
    severity: Optional[ViolationSeverityEnum] = Query(
        None, description="Filter by severity"
    ),
    violation_status: Optional[ViolationStatusEnum] = Query(
        None, alias="status", description="Filter by violation status"
    ),
    plot_id: Optional[str] = Query(
        None, description="Filter by plot ID"
    ),
    territory_id: Optional[str] = Query(
        None, description="Filter by territory ID"
    ),
    supplier_id: Optional[str] = Query(
        None, description="Filter by supplier ID"
    ),
    community_id: Optional[str] = Query(
        None, description="Filter by community ID"
    ),
    sort_by: Optional[str] = Query(
        "detected_at",
        description="Sort field (detected_at, severity, violation_type)",
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.DESC, description="Sort order"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:violations:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ViolationListResponse:
    """List violations with optional filters and pagination.

    Args:
        violation_type: Optional violation type filter.
        severity: Optional severity filter.
        violation_status: Optional status filter.
        plot_id: Optional plot filter.
        territory_id: Optional territory filter.
        supplier_id: Optional supplier filter.
        community_id: Optional community filter.
        sort_by: Sort field.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        ViolationListResponse with paginated violation list.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.list_violations(
            violation_type=violation_type.value if violation_type else None,
            severity=severity.value if severity else None,
            status=violation_status.value if violation_status else None,
            plot_id=plot_id,
            territory_id=territory_id,
            supplier_id=supplier_id,
            community_id=community_id,
            sort_by=sort_by,
            sort_order=sort_order.value if sort_order else "desc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        violations = [
            _build_violation_entry(entry)
            for entry in result.get("violations", [])
        ]
        total = result.get("total", len(violations))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_violations:{violation_type}:{severity}:{violation_status}",
            str(total),
        )

        logger.info(
            "Violations listed: total=%d returned=%d operator=%s",
            total,
            len(violations),
            user.operator_id or user.user_id,
        )

        return ViolationListResponse(
            violations=violations,
            total_violations=total,
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
                data_sources=["IndigenousRightsChecker", "ViolationDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Violation listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Violation listing failed",
        )


# ---------------------------------------------------------------------------
# GET /violations/{violation_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{violation_id}",
    response_model=ViolationResponse,
    summary="Get violation details",
    description=(
        "Retrieve detailed information for a specific rights violation "
        "including evidence references, remediation actions, resolution "
        "status, and associated plot/territory/community information."
    ),
    responses={
        200: {"description": "Violation details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Violation not found"},
    },
)
async def get_violation(
    violation_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:violations:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ViolationResponse:
    """Get detailed information for a specific violation.

    Args:
        violation_id: Unique violation identifier.
        user: Authenticated user.

    Returns:
        ViolationResponse with full violation details.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.get_violation(violation_id=violation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Violation not found: {violation_id}",
            )

        violation_data = result.get("violation", {})
        violation_entry = _build_violation_entry(violation_data)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"get_violation:{violation_id}",
            violation_entry.violation_type.value,
        )

        logger.info(
            "Violation retrieved: id=%s type=%s severity=%s operator=%s",
            violation_id,
            violation_entry.violation_type.value,
            violation_entry.severity.value,
            user.operator_id or user.user_id,
        )

        return ViolationResponse(
            violation=violation_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "ViolationDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Violation retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Violation retrieval failed",
        )


# ---------------------------------------------------------------------------
# PUT /violations/{violation_id}/resolve
# ---------------------------------------------------------------------------


@router.put(
    "/{violation_id}/resolve",
    response_model=ViolationResolveResponse,
    summary="Resolve a violation",
    description=(
        "Mark a violation as resolved with resolution notes, remediation "
        "actions taken, and supporting evidence. The violation status is "
        "updated to 'resolved' and a resolution timestamp is recorded. "
        "If resolution criteria are not fully met, the response indicates "
        "remaining actions required."
    ),
    responses={
        200: {"description": "Violation resolved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid resolution data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Violation not found"},
    },
)
async def resolve_violation(
    violation_id: str,
    request: Request,
    body: ViolationResolveRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:violations:resolve")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ViolationResolveResponse:
    """Mark a violation as resolved.

    Args:
        violation_id: Unique violation identifier.
        body: Resolution details including notes and evidence.
        user: Authenticated user with violations:resolve permission.

    Returns:
        ViolationResolveResponse with updated violation and resolution status.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.resolve_violation(
            violation_id=violation_id,
            resolution_notes=body.resolution_notes,
            remediation_actions_taken=body.remediation_actions_taken,
            evidence_ids=body.evidence_ids,
            resolved_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Violation not found: {violation_id}",
            )

        violation_data = result.get("violation", {})
        violation_entry = _build_violation_entry(violation_data)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"resolve_violation:{violation_id}",
            str(result.get("resolution_accepted", False)),
        )

        logger.info(
            "Violation resolved: id=%s accepted=%s operator=%s",
            violation_id,
            result.get("resolution_accepted", False),
            user.operator_id or user.user_id,
        )

        return ViolationResolveResponse(
            violation=violation_entry,
            resolution_accepted=result.get("resolution_accepted", True),
            remaining_actions=result.get("remaining_actions"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "ViolationDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Violation resolution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Violation resolution failed",
        )
