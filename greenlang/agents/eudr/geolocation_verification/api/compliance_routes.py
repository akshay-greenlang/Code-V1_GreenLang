# -*- coding: utf-8 -*-
"""
Compliance Reporting Routes - AGENT-EUDR-002 Geolocation Verification API

Endpoints for generating Article 9 compliance reports, retrieving
generated reports, and accessing compliance dashboard summary data.

Endpoints:
    POST /compliance/report              - Generate Article 9 compliance report
    GET  /compliance/report/{report_id}  - Get generated report by ID
    GET  /compliance/summary             - Compliance summary dashboard data

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.geolocation_verification.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_article9_reporter,
    get_verification_service,
    rate_limit_export,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.geolocation_verification.api.schemas import (
    ComplianceReportRequest,
    ComplianceReportResponse,
    ComplianceSummaryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Compliance Reporting"])


# ---------------------------------------------------------------------------
# In-memory report store (replaced by database in production)
# ---------------------------------------------------------------------------

_report_store: Dict[str, Dict[str, Any]] = {}


def _get_report_store() -> Dict[str, Dict[str, Any]]:
    """Return the report store. Replaceable for testing."""
    return _report_store


# ---------------------------------------------------------------------------
# POST /compliance/report
# ---------------------------------------------------------------------------


@router.post(
    "/compliance/report",
    response_model=ComplianceReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate Article 9 compliance report",
    description=(
        "Generate a comprehensive EUDR Article 9 geolocation compliance "
        "report for a specified operator. The report aggregates all "
        "verification results, accuracy scores, and quality tier "
        "distributions. Supports JSON, PDF, and CSV output formats. "
        "For PDF/CSV formats, a download URL is provided."
    ),
    responses={
        202: {"description": "Report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_compliance_report(
    body: ComplianceReportRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:compliance:write")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ComplianceReportResponse:
    """Generate an Article 9 compliance report.

    Creates a compliance report aggregating all verification results
    for the specified operator. The report includes plot-level detail,
    quality tier distributions, and overall compliance status.

    Args:
        body: Report generation request with operator and format.
        user: Authenticated user with compliance:write permission.

    Returns:
        ComplianceReportResponse with report data or download URL.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    report_id = f"rpt-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Compliance report generation: user=%s operator=%s commodity=%s format=%s",
        user.user_id,
        body.operator_id,
        body.commodity,
        body.format,
    )

    # Authorization: ensure user can only report on their own operator
    operator_id = user.operator_id or user.user_id
    if body.operator_id != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to generate reports for this operator",
        )

    try:
        reporter = get_article9_reporter()

        result = reporter.generate_report(
            operator_id=body.operator_id,
            commodity=body.commodity,
            output_format=body.format,
        )

        elapsed = time.monotonic() - start

        # Build response from reporter result
        report_data = None
        download_url = None

        if body.format == "json":
            report_data = getattr(result, "report_data", {})
        else:
            download_url = getattr(result, "download_url", None)

        total_plots = getattr(result, "total_plots", 0)
        compliant_plots = getattr(result, "compliant_plots", 0)
        non_compliant_plots = getattr(result, "non_compliant_plots", 0)
        compliance_rate = (
            (compliant_plots / total_plots * 100.0) if total_plots > 0 else 0.0
        )

        response = ComplianceReportResponse(
            report_id=report_id,
            operator_id=body.operator_id,
            commodity=body.commodity,
            format=body.format,
            status="generated",
            total_plots=total_plots,
            compliant_plots=compliant_plots,
            non_compliant_plots=non_compliant_plots,
            compliance_rate=round(compliance_rate, 2),
            average_accuracy_score=getattr(result, "average_accuracy_score", 0.0),
            quality_distribution=getattr(
                result, "quality_distribution",
                {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
            ),
            issues_summary=getattr(result, "issues_summary", {}),
            report_data=report_data,
            download_url=download_url,
        )

        # Store report for retrieval
        store = _get_report_store()
        store[report_id] = {
            "report_id": report_id,
            "operator_id": body.operator_id,
            "commodity": body.commodity,
            "format": body.format,
            "status": "generated",
            "total_plots": total_plots,
            "compliant_plots": compliant_plots,
            "non_compliant_plots": non_compliant_plots,
            "compliance_rate": compliance_rate,
            "average_accuracy_score": getattr(result, "average_accuracy_score", 0.0),
            "quality_distribution": getattr(
                result, "quality_distribution",
                {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
            ),
            "issues_summary": getattr(result, "issues_summary", {}),
            "report_data": report_data,
            "download_url": download_url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": user.user_id,
        }

        logger.info(
            "Compliance report generated: report_id=%s operator=%s "
            "total_plots=%d compliance_rate=%.1f%% elapsed_ms=%.1f",
            report_id,
            body.operator_id,
            total_plots,
            compliance_rate,
            elapsed * 1000,
        )

        return response

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Compliance report error: user=%s operator=%s error=%s",
            user.user_id,
            body.operator_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Compliance report generation failed: user=%s operator=%s error=%s",
            user.user_id,
            body.operator_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance report generation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /compliance/report/{report_id}
# ---------------------------------------------------------------------------


@router.get(
    "/compliance/report/{report_id}",
    response_model=ComplianceReportResponse,
    summary="Get generated compliance report",
    description=(
        "Retrieve a previously generated Article 9 compliance report "
        "by its unique report ID."
    ),
    responses={
        200: {"description": "Compliance report"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def get_compliance_report(
    report_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceReportResponse:
    """Retrieve a previously generated compliance report.

    Args:
        report_id: Report identifier.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceReportResponse with the stored report data.

    Raises:
        HTTPException: 404 if report not found, 403 if unauthorized.
    """
    logger.info(
        "Compliance report retrieval: user=%s report_id=%s",
        user.user_id,
        report_id,
    )

    store = _get_report_store()
    report = store.get(report_id)

    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found",
        )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    report_operator = report.get("operator_id", "")
    if report_operator != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this report",
        )

    return ComplianceReportResponse(
        report_id=report["report_id"],
        operator_id=report["operator_id"],
        commodity=report.get("commodity"),
        format=report.get("format", "json"),
        status=report.get("status", "generated"),
        total_plots=report.get("total_plots", 0),
        compliant_plots=report.get("compliant_plots", 0),
        non_compliant_plots=report.get("non_compliant_plots", 0),
        compliance_rate=report.get("compliance_rate", 0.0),
        average_accuracy_score=report.get("average_accuracy_score", 0.0),
        quality_distribution=report.get(
            "quality_distribution",
            {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
        ),
        issues_summary=report.get("issues_summary", {}),
        report_data=report.get("report_data"),
        download_url=report.get("download_url"),
    )


# ---------------------------------------------------------------------------
# GET /compliance/summary
# ---------------------------------------------------------------------------


@router.get(
    "/compliance/summary",
    response_model=ComplianceSummaryResponse,
    summary="Get compliance summary dashboard data",
    description=(
        "Retrieve high-level compliance summary data for the dashboard. "
        "Includes overall compliance rate, quality tier distribution, "
        "top issues, and breakdowns by commodity and country."
    ),
    responses={
        200: {"description": "Compliance summary"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_compliance_summary(
    request: Request,
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID (admins only)"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceSummaryResponse:
    """Get compliance summary dashboard data.

    Returns aggregate compliance metrics including overall rate,
    quality distribution, top issues, and breakdowns by commodity
    and country.

    Args:
        operator_id: Optional operator filter (admin only).
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceSummaryResponse with dashboard data.
    """
    logger.info(
        "Compliance summary request: user=%s operator_filter=%s",
        user.user_id,
        operator_id,
    )

    # Non-admins can only view their own operator's data
    effective_operator = operator_id
    if operator_id and operator_id != (user.operator_id or user.user_id):
        if "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can view other operators' compliance data",
            )
    elif not operator_id:
        effective_operator = user.operator_id or user.user_id

    try:
        reporter = get_article9_reporter()
        service = get_verification_service()

        summary = reporter.get_compliance_summary(
            operator_id=effective_operator,
        )

        if summary is None:
            return ComplianceSummaryResponse(
                operator_id=effective_operator,
                total_operators=1 if effective_operator else 0,
                total_plots=0,
                verified_plots=0,
                compliant_plots=0,
                overall_compliance_rate=0.0,
                average_accuracy_score=0.0,
                quality_distribution={"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
                top_issues=[],
                by_commodity={},
                by_country={},
            )

        return ComplianceSummaryResponse(
            operator_id=effective_operator,
            total_operators=getattr(summary, "total_operators", 1),
            total_plots=getattr(summary, "total_plots", 0),
            verified_plots=getattr(summary, "verified_plots", 0),
            compliant_plots=getattr(summary, "compliant_plots", 0),
            overall_compliance_rate=getattr(summary, "overall_compliance_rate", 0.0),
            average_accuracy_score=getattr(summary, "average_accuracy_score", 0.0),
            quality_distribution=getattr(
                summary, "quality_distribution",
                {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
            ),
            top_issues=getattr(summary, "top_issues", []),
            by_commodity=getattr(summary, "by_commodity", {}),
            by_country=getattr(summary, "by_country", {}),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Compliance summary failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        # Return empty summary rather than 500
        return ComplianceSummaryResponse(
            operator_id=effective_operator,
            total_operators=0,
            total_plots=0,
            verified_plots=0,
            compliant_plots=0,
            overall_compliance_rate=0.0,
            average_accuracy_score=0.0,
        )
