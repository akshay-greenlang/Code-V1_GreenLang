# -*- coding: utf-8 -*-
"""
Violation Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for detecting, listing, resolving, and escalating protected area
violations including encroachment, buffer breaches, illegal clearing, and
designation non-compliance.

Endpoints:
    POST /violations/detect                     - Detect violations for a plot
    GET  /violations                            - List violations with filters
    PUT  /violations/{violation_id}/resolve      - Resolve a violation
    PUT  /violations/{violation_id}/escalate     - Escalate a violation

Auth: eudr-pav:violation:{create|read|update}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, ViolationDetector Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_violation_detector,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    EscalationLevelEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RiskLevelEnum,
    ViolationDetectRequest,
    ViolationDetectResponse,
    ViolationEntry,
    ViolationEscalateRequest,
    ViolationEscalateResponse,
    ViolationListResponse,
    ViolationResolveRequest,
    ViolationResolveResponse,
    ViolationStatusEnum,
    ViolationTypeEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/violations", tags=["Violations"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _parse_violation_entry(v: dict) -> ViolationEntry:
    """Parse a raw violation result into a ViolationEntry schema."""
    return ViolationEntry(
        violation_id=v.get("violation_id", ""),
        plot_id=v.get("plot_id", ""),
        area_id=v.get("area_id", ""),
        area_name=v.get("area_name", ""),
        violation_type=ViolationTypeEnum(v.get("violation_type", "other")),
        status=ViolationStatusEnum(v.get("status", "detected")),
        risk_level=RiskLevelEnum(v.get("risk_level", "medium")),
        overlap_area_km2=Decimal(str(v.get("overlap_area_km2", 0)))
        if v.get("overlap_area_km2") is not None else None,
        distance_km=Decimal(str(v.get("distance_km", 0)))
        if v.get("distance_km") is not None else None,
        description=v.get("description", ""),
        regulatory_reference=v.get("regulatory_reference"),
        resolved_at=v.get("resolved_at"),
    )


# ---------------------------------------------------------------------------
# POST /violations/detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=ViolationDetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect protected area violations for a plot",
    description=(
        "Detect all protected area violations for a supply chain plot "
        "including encroachment into protected areas, buffer zone breaches, "
        "boundary violations, and designation non-compliance."
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
        require_permission("eudr-pav:violation:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ViolationDetectResponse:
    """Detect protected area violations for a plot.

    Args:
        body: Detection request with plot boundary and parameters.
        user: Authenticated user with violation:create permission.

    Returns:
        ViolationDetectResponse with detected violations.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.detect(
            plot_id=body.plot_id,
            plot_boundary=[
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.plot_boundary.coordinates
            ],
            plot_center=(
                {"latitude": float(body.plot_center.latitude), "longitude": float(body.plot_center.longitude)}
                if body.plot_center else None
            ),
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            include_buffer_violations=body.include_buffer_violations,
            include_designation_violations=body.include_designation_violations,
            detected_by=user.user_id,
        )

        violations = [_parse_violation_entry(v) for v in result.get("violations", [])]
        has_violations = len(violations) > 0

        risk_priority = {
            RiskLevelEnum.CRITICAL: 5, RiskLevelEnum.HIGH: 4,
            RiskLevelEnum.MEDIUM: 3, RiskLevelEnum.LOW: 2,
            RiskLevelEnum.NEGLIGIBLE: 1,
        }
        highest_risk = max(
            (v.risk_level for v in violations),
            key=lambda r: risk_priority.get(r, 0),
            default=RiskLevelEnum.NEGLIGIBLE,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"detect_violations:{body.plot_id}", str(len(violations)),
        )

        logger.info(
            "Violations detected: plot_id=%s violations=%d risk=%s operator=%s",
            body.plot_id,
            len(violations),
            highest_risk.value,
            user.operator_id or user.user_id,
        )

        return ViolationDetectResponse(
            plot_id=body.plot_id,
            violations=violations,
            total_violations=len(violations),
            has_violations=has_violations,
            highest_risk_level=highest_risk,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ViolationDetector", "WDPA", "OECM"],
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
    summary="List violations with filters",
    description=(
        "Retrieve a paginated list of protected area violations with "
        "optional filters for plot, area, type, status, and risk level."
    ),
    responses={
        200: {"description": "Violations listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_violations(
    request: Request,
    plot_id: Optional[str] = Query(None, description="Filter by plot ID"),
    area_id: Optional[str] = Query(None, description="Filter by area ID"),
    violation_type: Optional[ViolationTypeEnum] = Query(None, description="Filter by violation type"),
    violation_status: Optional[ViolationStatusEnum] = Query(None, description="Filter by status"),
    risk_level: Optional[RiskLevelEnum] = Query(None, description="Filter by risk level"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:violation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ViolationListResponse:
    """List violations with optional filters.

    Args:
        plot_id: Optional plot filter.
        area_id: Optional area filter.
        violation_type: Optional type filter.
        violation_status: Optional status filter.
        risk_level: Optional risk filter.
        pagination: Pagination parameters.
        user: Authenticated user with violation:read permission.

    Returns:
        ViolationListResponse with paginated violations.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.list_violations(
            plot_id=plot_id,
            area_id=area_id,
            violation_type=violation_type.value if violation_type else None,
            status=violation_status.value if violation_status else None,
            risk_level=risk_level.value if risk_level else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        violations = [_parse_violation_entry(v) for v in result.get("violations", [])]
        total = result.get("total", len(violations))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance("list_violations", str(total))

        logger.info(
            "Violations listed: total=%d operator=%s",
            total,
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
                data_sources=["ViolationDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Violations listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Violations listing failed",
        )


# ---------------------------------------------------------------------------
# PUT /violations/{violation_id}/resolve
# ---------------------------------------------------------------------------


@router.put(
    "/{violation_id}/resolve",
    response_model=ViolationResolveResponse,
    summary="Resolve a violation",
    description=(
        "Resolve a protected area violation by recording the resolution, "
        "root cause analysis, and corrective actions taken."
    ),
    responses={
        200: {"description": "Violation resolved"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
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
        require_permission("eudr-pav:violation:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ViolationResolveResponse:
    """Resolve a violation.

    Args:
        violation_id: Violation identifier.
        body: Resolution details.
        user: Authenticated user with violation:update permission.

    Returns:
        ViolationResolveResponse with resolution confirmation.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.resolve_violation(
            violation_id=violation_id,
            resolution=body.resolution,
            root_cause=body.root_cause,
            corrective_actions=body.corrective_actions,
            evidence_urls=body.evidence_urls,
            is_false_positive=body.is_false_positive,
            resolved_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Violation not found: {violation_id}",
            )

        new_status = (
            ViolationStatusEnum.FALSE_POSITIVE
            if body.is_false_positive
            else ViolationStatusEnum.RESOLVED
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"resolve_violation:{violation_id}", new_status.value,
        )

        logger.info(
            "Violation resolved: violation_id=%s status=%s by=%s operator=%s",
            violation_id,
            new_status.value,
            user.user_id,
            user.operator_id or user.user_id,
        )

        return ViolationResolveResponse(
            violation_id=violation_id,
            previous_status=ViolationStatusEnum(result.get("previous_status", "detected")),
            new_status=new_status,
            resolved_by=user.user_id,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ViolationDetector"],
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


# ---------------------------------------------------------------------------
# PUT /violations/{violation_id}/escalate
# ---------------------------------------------------------------------------


@router.put(
    "/{violation_id}/escalate",
    response_model=ViolationEscalateResponse,
    summary="Escalate a violation",
    description=(
        "Escalate a protected area violation to a higher authority level. "
        "Level 1: team lead, Level 2: compliance officer, Level 3: competent authority."
    ),
    responses={
        200: {"description": "Violation escalated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Violation not found"},
    },
)
async def escalate_violation(
    violation_id: str,
    request: Request,
    body: ViolationEscalateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:violation:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ViolationEscalateResponse:
    """Escalate a violation.

    Args:
        violation_id: Violation identifier.
        body: Escalation details.
        user: Authenticated user with violation:update permission.

    Returns:
        ViolationEscalateResponse with escalation confirmation.
    """
    start = time.monotonic()

    try:
        engine = get_violation_detector()
        result = engine.escalate_violation(
            violation_id=violation_id,
            escalation_level=body.escalation_level.value,
            reason=body.reason,
            escalate_to=body.escalate_to,
            requires_authority_notification=body.requires_authority_notification,
            escalated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Violation not found: {violation_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"escalate_violation:{violation_id}",
            body.escalation_level.value,
        )

        logger.info(
            "Violation escalated: violation_id=%s level=%s by=%s operator=%s",
            violation_id,
            body.escalation_level.value,
            user.user_id,
            user.operator_id or user.user_id,
        )

        return ViolationEscalateResponse(
            violation_id=violation_id,
            previous_status=ViolationStatusEnum(result.get("previous_status", "detected")),
            new_status=ViolationStatusEnum.ESCALATED,
            escalation_level=body.escalation_level,
            escalated_by=user.user_id,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ViolationDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Violation escalation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Violation escalation failed",
        )
