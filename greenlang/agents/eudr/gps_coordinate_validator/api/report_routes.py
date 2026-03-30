# -*- coding: utf-8 -*-
"""
Compliance Reporting Routes - AGENT-EUDR-007 GPS Coordinate Validator API

Endpoints for generating compliance certificates, batch summaries,
remediation plans, and downloadable reports for coordinate quality
in EUDR due diligence statements.

Endpoints:
    POST /report/compliance       - Generate compliance certificate
    POST /report/summary          - Generate batch summary report
    POST /report/remediation      - Generate remediation plan
    GET  /report/{report_id}      - Retrieve stored report
    GET  /report/{report_id}/download - Download report (PDF/CSV)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from fastapi.responses import JSONResponse
from greenlang.schemas import utcnow

from greenlang.agents.eudr.gps_coordinate_validator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_gps_validator_service,
    rate_limit_report,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.schemas import (
    BatchSummaryResponseSchema,
    BatchValidateRequestSchema,
    CoordinatePairSchema,
    ComplianceCertRequestSchema,
    ComplianceCertResponseSchema,
    RemediationItemSchema,
    RemediationResponseSchema,
    ReportResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Compliance Reporting"])

# ---------------------------------------------------------------------------
# In-memory report store (replaced by database in production)
# ---------------------------------------------------------------------------

_report_store: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# POST /report/compliance
# ---------------------------------------------------------------------------

@router.post(
    "/report/compliance",
    response_model=ComplianceCertResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Generate compliance certificate",
    description=(
        "Generate a coordinate compliance certificate for EUDR due "
        "diligence. Runs validation, plausibility, and precision checks "
        "to determine compliance status. The certificate includes the "
        "accuracy score, check summary, and a provenance hash for "
        "audit trail verification."
    ),
    responses={
        201: {"description": "Compliance certificate generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_compliance_cert(
    body: ComplianceCertRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:report:write")
    ),
    _rate: None = Depends(rate_limit_report),
) -> ComplianceCertResponseSchema:
    """Generate a compliance certificate for a coordinate.

    Performs all validation, plausibility, and precision checks to
    issue a compliance certificate with one of three statuses:
    compliant, non_compliant, or conditional.

    Args:
        body: Compliance certificate request with coordinate and context.
        request: FastAPI request object.
        user: Authenticated user with report:write permission.

    Returns:
        ComplianceCertResponseSchema with certificate details.

    Raises:
        HTTPException: 400 if input invalid, 500 on processing error.
    """
    start = time.monotonic()
    cert_id = f"cert-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Compliance cert request: user=%s cert_id=%s lat=%.6f lon=%.6f "
        "commodity=%s country=%s",
        user.user_id,
        cert_id,
        body.latitude,
        body.longitude,
        body.commodity,
        body.country_iso,
    )

    try:
        service = get_gps_validator_service()

        result = service.generate_compliance_cert(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity,
            country_iso=body.country_iso,
            source_type=body.source_type,
        )

        now = utcnow()
        valid_until = now + timedelta(days=365)

        coordinate = CoordinatePairSchema(
            latitude=body.latitude,
            longitude=body.longitude,
            datum="WGS84",
            commodity=body.commodity,
            country_iso=body.country_iso,
            source_type=body.source_type,
        )

        provenance = _compute_provenance(
            f"cert|{cert_id}|{body.latitude}|{body.longitude}|"
            f"{result.get('status', 'unknown')}"
        )

        response = ComplianceCertResponseSchema(
            cert_id=cert_id,
            status=result.get("status", "non_compliant"),
            accuracy_score=result.get("accuracy_score", 0.0),
            coordinate=coordinate,
            issued_at=now,
            valid_until=valid_until,
            provenance_hash=provenance,
            checks_summary=result.get("checks_summary", {}),
        )

        # Store report
        _report_store[cert_id] = {
            "report_id": cert_id,
            "report_type": "compliance_cert",
            "status": "generated",
            "data": response.model_dump(mode="json"),
            "format": "json",
            "user_id": user.user_id,
            "generated_at": now.isoformat(),
            "provenance_hash": provenance,
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Compliance cert generated: cert_id=%s status=%s score=%.1f "
            "elapsed_ms=%.1f",
            cert_id,
            response.status,
            response.accuracy_score,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Compliance cert error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Compliance cert failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance certificate generation failed",
        )

# ---------------------------------------------------------------------------
# POST /report/summary
# ---------------------------------------------------------------------------

@router.post(
    "/report/summary",
    response_model=BatchSummaryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate batch summary report",
    description=(
        "Generate a summary report for a batch of coordinates. "
        "Validates all coordinates and produces aggregate statistics "
        "including error breakdown, precision distribution, quality "
        "tier distribution, and improvement recommendations."
    ),
    responses={
        200: {"description": "Batch summary report"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_batch_summary(
    body: BatchValidateRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:report:write")
    ),
    _rate: None = Depends(rate_limit_report),
) -> BatchSummaryResponseSchema:
    """Generate a batch summary report with aggregate statistics.

    Validates all coordinates and computes summary statistics. Stores
    the report for later retrieval.

    Args:
        body: Batch validate request with coordinates.
        request: FastAPI request object.
        user: Authenticated user with report:write permission.

    Returns:
        BatchSummaryResponseSchema with aggregate stats.

    Raises:
        HTTPException: 400 if request invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)
    report_id = f"summary-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Batch summary request: user=%s report_id=%s total=%d",
        user.user_id,
        report_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        result = service.generate_batch_summary(
            coordinates=[
                {
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "commodity": c.commodity,
                    "country_iso": c.country_iso,
                    "source_type": c.source_type,
                }
                for c in body.coordinates
            ]
        )

        response = BatchSummaryResponseSchema(
            total=total,
            valid=result.get("valid", 0),
            invalid=result.get("invalid", 0),
            warning_count=result.get("warning_count", 0),
            error_breakdown=result.get("error_breakdown", {}),
            precision_distribution=result.get("precision_distribution", {}),
            tier_distribution=result.get("tier_distribution", {
                "gold": 0, "silver": 0, "bronze": 0, "fail": 0,
            }),
            recommendations=result.get("recommendations", []),
        )

        # Store report
        _report_store[report_id] = {
            "report_id": report_id,
            "report_type": "batch_summary",
            "status": "generated",
            "data": response.model_dump(mode="json"),
            "format": "json",
            "user_id": user.user_id,
            "generated_at": utcnow().isoformat(),
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Batch summary generated: report_id=%s valid=%d invalid=%d "
            "elapsed_ms=%.1f",
            report_id,
            response.valid,
            response.invalid,
            elapsed * 1000,
        )

        return response

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch summary failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch summary generation failed",
        )

# ---------------------------------------------------------------------------
# POST /report/remediation
# ---------------------------------------------------------------------------

@router.post(
    "/report/remediation",
    response_model=RemediationResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate remediation plan",
    description=(
        "Generate a remediation plan for coordinates with validation "
        "errors. Identifies all errors, classifies them by severity, "
        "determines which can be auto-fixed, and provides suggested "
        "actions for manual review items."
    ),
    responses={
        200: {"description": "Remediation plan"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_remediation(
    body: BatchValidateRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:report:write")
    ),
    _rate: None = Depends(rate_limit_report),
) -> RemediationResponseSchema:
    """Generate a remediation plan for failed coordinates.

    Validates all coordinates and produces a remediation plan with
    per-error action items, auto-fix suggestions, and severity
    classification.

    Args:
        body: Batch validate request with coordinates.
        request: FastAPI request object.
        user: Authenticated user with report:write permission.

    Returns:
        RemediationResponseSchema with action items.

    Raises:
        HTTPException: 400 if request invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Remediation plan request: user=%s total=%d",
        user.user_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        result = service.generate_remediation_plan(
            coordinates=[
                {
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "commodity": c.commodity,
                    "country_iso": c.country_iso,
                    "source_type": c.source_type,
                }
                for c in body.coordinates
            ]
        )

        items = [
            RemediationItemSchema(**item)
            for item in result.get("remediation_items", [])
        ]

        auto_fixable = sum(1 for item in items if item.auto_fixable)
        manual_review = len(items) - auto_fixable

        error_type_counts: Dict[str, int] = {}
        for item in items:
            et = item.error_type
            error_type_counts[et] = error_type_counts.get(et, 0) + 1

        elapsed = time.monotonic() - start
        logger.info(
            "Remediation plan generated: user=%s total_errors=%d "
            "auto_fixable=%d manual=%d elapsed_ms=%.1f",
            user.user_id,
            len(items),
            auto_fixable,
            manual_review,
            elapsed * 1000,
        )

        return RemediationResponseSchema(
            total_errors=len(items),
            auto_fixable_count=auto_fixable,
            manual_review_count=manual_review,
            remediation_items=items,
            summary_by_error_type=error_type_counts,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Remediation plan failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Remediation plan generation failed",
        )

# ---------------------------------------------------------------------------
# GET /report/{report_id}
# ---------------------------------------------------------------------------

@router.get(
    "/report/{report_id}",
    response_model=ReportResponseSchema,
    summary="Retrieve stored report",
    description=(
        "Retrieve a previously generated report by its unique identifier. "
        "Returns the full report data including compliance certificates, "
        "batch summaries, and remediation plans."
    ),
    responses={
        200: {"description": "Report data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def get_report(
    report_id: str = Path(
        ...,
        description="Unique report identifier",
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:report:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportResponseSchema:
    """Retrieve a stored report by ID.

    Args:
        report_id: Unique report identifier.
        request: FastAPI request object.
        user: Authenticated user with report:read permission.

    Returns:
        ReportResponseSchema with report data.

    Raises:
        HTTPException: 404 if report not found.
    """
    logger.info(
        "Get report: user=%s report_id=%s",
        user.user_id,
        report_id,
    )

    stored = _report_store.get(report_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found",
        )

    return ReportResponseSchema(
        report_id=stored["report_id"],
        report_type=stored["report_type"],
        status=stored["status"],
        data=stored["data"],
        format=stored.get("format", "json"),
        download_url=stored.get("download_url"),
        provenance_hash=stored.get("provenance_hash", ""),
    )

# ---------------------------------------------------------------------------
# GET /report/{report_id}/download
# ---------------------------------------------------------------------------

@router.get(
    "/report/{report_id}/download",
    summary="Download report",
    description=(
        "Download a previously generated report in its specified format "
        "(JSON, PDF, or CSV). For JSON reports, returns the data inline. "
        "For PDF/CSV, returns a redirect or streaming response."
    ),
    responses={
        200: {"description": "Report download"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def download_report(
    report_id: str = Path(
        ...,
        description="Unique report identifier",
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:report:read")
    ),
    _rate: None = Depends(rate_limit_report),
) -> JSONResponse:
    """Download a report in its specified format.

    Currently supports JSON format inline. PDF and CSV formats
    return a redirect URL to the object storage download endpoint.

    Args:
        report_id: Unique report identifier.
        request: FastAPI request object.
        user: Authenticated user with report:read permission.

    Returns:
        JSONResponse with report data or download URL.

    Raises:
        HTTPException: 404 if report not found.
    """
    logger.info(
        "Download report: user=%s report_id=%s",
        user.user_id,
        report_id,
    )

    stored = _report_store.get(report_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found",
        )

    report_format = stored.get("format", "json")

    if report_format == "json":
        return JSONResponse(
            content={
                "report_id": stored["report_id"],
                "report_type": stored["report_type"],
                "format": "json",
                "data": stored["data"],
                "generated_at": stored.get("generated_at", ""),
            },
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{report_id}.json"'
                ),
            },
        )

    # For PDF/CSV, return a download URL (implemented with object storage)
    download_url = stored.get("download_url")
    if download_url:
        return JSONResponse(
            content={
                "report_id": stored["report_id"],
                "format": report_format,
                "download_url": download_url,
            }
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"Report {report_id} in {report_format} format is not "
            "available for download. Generate with format='json'."
        ),
    )
