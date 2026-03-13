# -*- coding: utf-8 -*-
"""
Report Routes - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Endpoints for generating EUDR Article 14 audit reports, tier depth
summary reports, gap analysis reports, and report retrieval/download.

Endpoints:
    POST /reports/audit             - Generate audit report
    POST /reports/tier-summary      - Tier depth summary report
    POST /reports/gaps              - Gap analysis report
    GET  /reports/{report_id}       - Get report metadata
    GET  /reports/{report_id}/download - Download report

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from greenlang.agents.eudr.multi_tier_supplier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_supplier_service,
    rate_limit_report,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    AuditReportRequestSchema,
    AuditReportSchema,
    GapReportRequestSchema,
    GapReportSchema,
    ReportDownloadSchema,
    ReportMetadataSchema,
    TierSummaryRequestSchema,
    TierSummarySchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reporting"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /reports/audit
# ---------------------------------------------------------------------------


@router.post(
    "/audit",
    response_model=AuditReportSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate EUDR audit report",
    description=(
        "Generate a comprehensive EUDR Article 14 audit report for "
        "a complete supplier chain. Includes supplier hierarchy, "
        "compliance status, risk scores, and optionally relationship "
        "history. Supports JSON, PDF, CSV, and EUDR XML formats."
    ),
    responses={
        200: {"description": "Audit report generated"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_audit_report(
    body: AuditReportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:reports:write")
    ),
    _rate: None = Depends(rate_limit_report),
) -> AuditReportSchema:
    """Generate an EUDR Article 14 audit report.

    Produces a full supply chain audit report within 30 seconds
    as required by EUDR Article 14 readiness targets.

    Args:
        body: Audit report request with root supplier and options.
        request: FastAPI request object.
        user: Authenticated user with reports:write permission.

    Returns:
        AuditReportSchema with report data or download URL.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Generate audit report: user=%s root=%s commodity=%s format=%s",
        user.user_id,
        body.root_supplier_id,
        body.commodity,
        body.report_format,
    )

    try:
        service = get_supplier_service()

        result = service.generate_audit_report(
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            report_format=body.report_format,
            include_risk_details=body.include_risk_details,
            include_compliance_details=body.include_compliance_details,
            include_relationship_history=body.include_relationship_history,
            date_from=body.date_from,
            date_to=body.date_to,
            generated_by=user.user_id,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"audit_report|{body.root_supplier_id}|{body.commodity}|"
            f"{body.report_format}|{result.get('total_suppliers', 0)}|{elapsed}"
        )

        logger.info(
            "Audit report generated: user=%s report_id=%s "
            "suppliers=%d depth=%d elapsed_ms=%.1f",
            user.user_id,
            result.get("report_id", ""),
            result.get("total_suppliers", 0),
            result.get("max_tier_depth", 0),
            elapsed * 1000,
        )

        return AuditReportSchema(
            report_id=result.get("report_id", str(uuid.uuid4())),
            report_type="audit",
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            report_format=body.report_format,
            total_suppliers=result.get("total_suppliers", 0),
            max_tier_depth=result.get("max_tier_depth", 0),
            compliance_summary=result.get("compliance_summary", {}),
            risk_summary=result.get("risk_summary", {}),
            data=result.get("data") if body.report_format == "json" else None,
            download_url=result.get("download_url"),
            expires_at=result.get("expires_at"),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Audit report validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audit report generation validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Audit report generation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit report generation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /reports/tier-summary
# ---------------------------------------------------------------------------


@router.post(
    "/tier-summary",
    response_model=TierSummarySchema,
    status_code=status.HTTP_200_OK,
    summary="Generate tier depth summary report",
    description=(
        "Generate a summary report of tier depth metrics across all "
        "supply chains. Includes per-commodity breakdowns, visibility "
        "scores, and optional industry benchmarks."
    ),
    responses={
        200: {"description": "Tier summary report generated"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_tier_summary(
    body: TierSummaryRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:reports:write")
    ),
    _rate: None = Depends(rate_limit_report),
) -> TierSummarySchema:
    """Generate a tier depth summary report.

    Args:
        body: Tier summary request with optional filters and format.
        request: FastAPI request object.
        user: Authenticated user with reports:write permission.

    Returns:
        TierSummarySchema with tier depth statistics.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Generate tier summary: user=%s commodity=%s country=%s format=%s",
        user.user_id,
        body.commodity,
        body.country_iso,
        body.report_format,
    )

    try:
        service = get_supplier_service()

        result = service.generate_tier_summary(
            commodity=body.commodity,
            country_iso=body.country_iso,
            include_benchmarks=body.include_benchmarks,
            report_format=body.report_format,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"tier_summary|{body.commodity}|{body.country_iso}|"
            f"{result.get('total_supply_chains', 0)}|{elapsed}"
        )

        logger.info(
            "Tier summary generated: user=%s chains=%d avg_depth=%.1f "
            "visibility=%.1f elapsed_ms=%.1f",
            user.user_id,
            result.get("total_supply_chains", 0),
            result.get("avg_tier_depth", 0),
            result.get("overall_visibility_score", 0),
            elapsed * 1000,
        )

        return TierSummarySchema(
            report_id=result.get("report_id", str(uuid.uuid4())),
            total_supply_chains=result.get("total_supply_chains", 0),
            avg_tier_depth=result.get("avg_tier_depth", 0.0),
            max_tier_depth=result.get("max_tier_depth", 0),
            total_suppliers=result.get("total_suppliers", 0),
            overall_visibility_score=result.get("overall_visibility_score", 0.0),
            commodity_breakdown=result.get("commodity_breakdown", []),
            benchmarks=result.get("benchmarks") if body.include_benchmarks else None,
            report_format=body.report_format,
            download_url=result.get("download_url"),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Tier summary validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tier summary report validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Tier summary generation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tier summary report generation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /reports/gaps
# ---------------------------------------------------------------------------


@router.post(
    "/gaps",
    response_model=GapReportSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate gap analysis report",
    description=(
        "Generate a data gap analysis report identifying missing GPS, "
        "missing certifications, missing legal entities, and other "
        "data quality gaps. Includes severity classification and "
        "remediation action plans."
    ),
    responses={
        200: {"description": "Gap analysis report generated"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_gap_report(
    body: GapReportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:reports:write")
    ),
    _rate: None = Depends(rate_limit_report),
) -> GapReportSchema:
    """Generate a gap analysis report.

    Identifies all data gaps across the supplier hierarchy,
    classifies by severity, and generates remediation plans.

    Args:
        body: Gap report request with optional filters.
        request: FastAPI request object.
        user: Authenticated user with reports:write permission.

    Returns:
        GapReportSchema with identified gaps and remediation.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Generate gap report: user=%s root=%s commodity=%s severity=%s",
        user.user_id,
        body.root_supplier_id,
        body.commodity,
        body.severity_filter,
    )

    try:
        service = get_supplier_service()

        result = service.generate_gap_report(
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            severity_filter=body.severity_filter,
            include_remediation_plans=body.include_remediation_plans,
            report_format=body.report_format,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        gaps = result.get("gaps", [])
        critical = sum(1 for g in gaps if g.get("severity") == "critical")
        major = sum(1 for g in gaps if g.get("severity") == "major")
        minor = sum(1 for g in gaps if g.get("severity") == "minor")
        dds_blocking = sum(1 for g in gaps if g.get("dds_blocking", False))

        provenance = _compute_provenance(
            f"gap_report|{body.root_supplier_id}|{body.commodity}|"
            f"{len(gaps)}|{critical}|{elapsed}"
        )

        logger.info(
            "Gap report generated: user=%s total=%d critical=%d "
            "dds_blocking=%d elapsed_ms=%.1f",
            user.user_id,
            len(gaps),
            critical,
            dds_blocking,
            elapsed * 1000,
        )

        return GapReportSchema(
            report_id=result.get("report_id", str(uuid.uuid4())),
            total_gaps=len(gaps),
            critical_gaps=critical,
            major_gaps=major,
            minor_gaps=minor,
            dds_blocking_gaps=dds_blocking,
            gaps=gaps,
            gap_trend=result.get("gap_trend", "stable"),
            remediation_summary=result.get("remediation_summary", {}),
            report_format=body.report_format,
            download_url=result.get("download_url"),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Gap report validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gap report generation validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Gap report generation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gap report generation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /reports/{report_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}",
    response_model=ReportMetadataSchema,
    status_code=status.HTTP_200_OK,
    summary="Get report metadata",
    description=(
        "Retrieve metadata for a previously generated report including "
        "status, format, file size, and expiry information."
    ),
    responses={
        200: {"description": "Report metadata retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def get_report(
    report_id: str = Path(
        ..., min_length=1, max_length=100, description="Report identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportMetadataSchema:
    """Retrieve report metadata by ID.

    Args:
        report_id: Report identifier.
        request: FastAPI request object.
        user: Authenticated user with reports:read permission.

    Returns:
        ReportMetadataSchema with report status and metadata.

    Raises:
        HTTPException: 404 if report not found.
    """
    start = time.monotonic()
    logger.info(
        "Get report: user=%s report_id=%s",
        user.user_id,
        report_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_report(report_id=report_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found: {report_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"report_get|{report_id}|{elapsed}"
        )

        logger.info(
            "Report retrieved: user=%s report_id=%s status=%s "
            "elapsed_ms=%.1f",
            user.user_id,
            report_id,
            result.get("status", "unknown"),
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        return ReportMetadataSchema(**result)

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get report failed: user=%s report_id=%s error=%s",
            user.user_id,
            report_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /reports/{report_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}/download",
    response_model=ReportDownloadSchema,
    status_code=status.HTTP_200_OK,
    summary="Download report",
    description=(
        "Get a pre-signed download URL for a previously generated report. "
        "The URL is valid for a limited time (typically 1 hour). "
        "Supports PDF, CSV, and EUDR XML formats."
    ),
    responses={
        200: {"description": "Download URL generated"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
        410: {"model": ErrorResponse, "description": "Report expired"},
    },
)
async def download_report(
    report_id: str = Path(
        ..., min_length=1, max_length=100, description="Report identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportDownloadSchema:
    """Get a download URL for a report.

    Args:
        report_id: Report identifier.
        request: FastAPI request object.
        user: Authenticated user with reports:read permission.

    Returns:
        ReportDownloadSchema with pre-signed download URL.

    Raises:
        HTTPException: 404 if report not found, 410 if expired.
    """
    start = time.monotonic()
    logger.info(
        "Download report: user=%s report_id=%s",
        user.user_id,
        report_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_report_download(report_id=report_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found: {report_id}",
            )

        # Check if report has expired
        expires_at = result.get("expires_at")
        if expires_at and isinstance(expires_at, datetime):
            now = datetime.now(timezone.utc)
            if expires_at < now:
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail=f"Report has expired: {report_id}",
                )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"report_download|{report_id}|{elapsed}"
        )

        logger.info(
            "Report download URL generated: user=%s report_id=%s "
            "format=%s elapsed_ms=%.1f",
            user.user_id,
            report_id,
            result.get("report_format", "unknown"),
            elapsed * 1000,
        )

        return ReportDownloadSchema(
            report_id=report_id,
            report_type=result.get("report_type", "audit"),
            report_format=result.get("report_format", "pdf"),
            download_url=result.get("download_url", ""),
            file_size_bytes=result.get("file_size_bytes"),
            expires_at=result.get(
                "expires_at",
                datetime.now(timezone.utc),
            ),
            generated_at=result.get(
                "generated_at",
                datetime.now(timezone.utc),
            ),
            provenance_hash=provenance,
        )

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Download report failed: user=%s report_id=%s error=%s",
            user.user_id,
            report_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report download failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
