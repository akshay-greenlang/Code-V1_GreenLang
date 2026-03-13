# -*- coding: utf-8 -*-
"""
Compliance Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for comprehensive protected area compliance assessment,
compliance report generation, and audit trail retrieval for EUDR
due diligence on protected area impacts.

Endpoints:
    POST /compliance/assess                - Full compliance assessment
    GET  /compliance/report/{plot_id}      - Generate compliance report
    GET  /compliance/audit-trail/{plot_id}  - Get audit trail

Auth: eudr-pav:compliance:{create|read}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, ComplianceAssessor Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_compliance_assessor,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    AuditActionEnum,
    AuditTrailEntry,
    ComplianceAssessRequest,
    ComplianceAssessResponse,
    ComplianceAuditTrailResponse,
    ComplianceOutcomeEnum,
    ComplianceReportResponse,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RiskLevelEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /compliance/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=ComplianceAssessResponse,
    status_code=status.HTTP_200_OK,
    summary="Full protected area compliance assessment",
    description=(
        "Perform a comprehensive compliance assessment for a supply chain plot "
        "against all protected area requirements under EUDR Articles 2, 3, 9, "
        "10, and 29. Integrates overlap detection, buffer zone analysis, "
        "designation validation, PADDD event checking, and risk scoring."
    ),
    responses={
        200: {"description": "Compliance assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_compliance(
    request: Request,
    body: ComplianceAssessRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:compliance:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ComplianceAssessResponse:
    """Perform full protected area compliance assessment.

    Args:
        body: Assessment request with plot boundary and options.
        user: Authenticated user with compliance:create permission.

    Returns:
        ComplianceAssessResponse with assessment results.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.assess(
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
            operator_id=body.operator_id or user.operator_id,
            include_overlap_analysis=body.include_overlap_analysis,
            include_buffer_analysis=body.include_buffer_analysis,
            include_designation_check=body.include_designation_check,
            include_paddd_check=body.include_paddd_check,
            include_risk_score=body.include_risk_score,
            assessed_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Compliance assessment returned no result",
            )

        outcome = ComplianceOutcomeEnum(result.get("compliance_outcome", "requires_investigation"))
        risk_level = RiskLevelEnum(result.get("risk_level", "medium"))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_assess:{body.plot_id}", outcome.value,
        )

        logger.info(
            "Compliance assessed: plot_id=%s outcome=%s risk=%s overlaps=%d violations=%d operator=%s",
            body.plot_id,
            outcome.value,
            risk_level.value,
            result.get("total_overlaps", 0),
            result.get("total_violations", 0),
            user.operator_id or user.user_id,
        )

        return ComplianceAssessResponse(
            plot_id=body.plot_id,
            compliance_outcome=outcome,
            risk_level=risk_level,
            risk_score=Decimal(str(result.get("risk_score", 0))),
            total_overlaps=result.get("total_overlaps", 0),
            total_violations=result.get("total_violations", 0),
            buffer_violations=result.get("buffer_violations", 0),
            has_direct_overlap=result.get("has_direct_overlap", False),
            designation_issues=result.get("designation_issues", 0),
            paddd_events=result.get("paddd_events", 0),
            nearest_area_name=result.get("nearest_area_name"),
            nearest_area_distance_km=Decimal(str(result.get("nearest_area_distance_km", 0)))
            if result.get("nearest_area_distance_km") is not None else None,
            regulatory_articles=result.get(
                "regulatory_articles",
                ["Art. 2", "Art. 3", "Art. 9", "Art. 10", "Art. 29"],
            ),
            recommendations=result.get("recommendations", []),
            assessment_rationale=result.get("assessment_rationale", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "ComplianceAssessor",
                    "OverlapDetector",
                    "BufferZoneMonitor",
                    "DesignationValidator",
                    "RiskScorer",
                    "PADDDMonitor",
                    "WDPA",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance assessment failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/report/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/report/{plot_id}",
    response_model=ComplianceReportResponse,
    summary="Generate compliance report for a plot",
    description=(
        "Generate a comprehensive compliance report for a supply chain plot "
        "summarizing all protected area analyses including overlaps, violations, "
        "buffer zones, designations, risk scores, and PADDD events."
    ),
    responses={
        200: {"description": "Report generated"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot data not found"},
    },
)
async def generate_compliance_report(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceReportResponse:
    """Generate compliance report for a plot.

    Args:
        plot_id: Supply chain plot identifier.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceReportResponse with full report data.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.generate_report(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No compliance data found for plot: {plot_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_report:{plot_id}",
            result.get("compliance_outcome", "unknown"),
        )

        logger.info(
            "Compliance report generated: plot_id=%s outcome=%s operator=%s",
            plot_id,
            result.get("compliance_outcome", "unknown"),
            user.operator_id or user.user_id,
        )

        return ComplianceReportResponse(
            plot_id=plot_id,
            report_id=result.get("report_id", ""),
            compliance_outcome=ComplianceOutcomeEnum(
                result.get("compliance_outcome", "requires_investigation")
            ),
            risk_level=RiskLevelEnum(result.get("risk_level", "medium")),
            overlaps_summary=result.get("overlaps_summary", {}),
            violations_summary=result.get("violations_summary", {}),
            buffer_zone_summary=result.get("buffer_zone_summary", {}),
            designation_summary=result.get("designation_summary", {}),
            risk_score_summary=result.get("risk_score_summary", {}),
            paddd_summary=result.get("paddd_summary", {}),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceAssessor", "WDPA", "OECM"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance report generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance report generation failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/audit-trail/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/audit-trail/{plot_id}",
    response_model=ComplianceAuditTrailResponse,
    summary="Get compliance audit trail for a plot",
    description=(
        "Retrieve the full audit trail of protected area compliance actions "
        "for a plot including area registrations, overlap detections, "
        "violation events, resolutions, and compliance assessments."
    ),
    responses={
        200: {"description": "Audit trail retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No audit trail found"},
    },
)
async def get_audit_trail(
    plot_id: str,
    request: Request,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceAuditTrailResponse:
    """Get compliance audit trail for a plot.

    Args:
        plot_id: Supply chain plot identifier.
        pagination: Pagination parameters.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceAuditTrailResponse with audit entries.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.get_audit_trail(
            plot_id=plot_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No audit trail found for plot: {plot_id}",
            )

        entries = []
        for e in result.get("entries", []):
            entries.append(
                AuditTrailEntry(
                    entry_id=e.get("entry_id", ""),
                    action=AuditActionEnum(e.get("action", "compliance_assessed")),
                    entity_type=e.get("entity_type", "plot"),
                    entity_id=e.get("entity_id", plot_id),
                    performed_by=e.get("performed_by", "system"),
                    timestamp=e.get("timestamp"),
                    details=e.get("details"),
                    ip_address=e.get("ip_address"),
                )
            )

        total = result.get("total", len(entries))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"audit_trail:{plot_id}", str(total),
        )

        logger.info(
            "Audit trail retrieved: plot_id=%s entries=%d operator=%s",
            plot_id,
            total,
            user.operator_id or user.user_id,
        )

        return ComplianceAuditTrailResponse(
            plot_id=plot_id,
            entries=entries,
            total_entries=total,
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
                data_sources=["ComplianceAssessor", "AuditLog"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit trail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit trail retrieval failed",
        )
