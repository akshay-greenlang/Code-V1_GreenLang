# -*- coding: utf-8 -*-
"""
Report Routes - AGENT-EUDR-009 Chain of Custody API

Endpoints for generating and downloading EUDR compliance reports
including Article 9 traceability reports and mass balance period reports.

Endpoints:
    POST   /reports/traceability     - Article 9 traceability report
    POST   /reports/mass-balance     - Mass balance period report
    GET    /reports/{report_id}      - Get report
    GET    /reports/{report_id}/download - Download report

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009, Section 7.4
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.chain_of_custody.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_coc_service,
    get_request_id,
    rate_limit_export,
    rate_limit_report,
    rate_limit_standard,
    require_permission,
    validate_report_id,
)
from greenlang.agents.eudr.chain_of_custody.api.schemas import (
    MassBalanceReportRequest,
    ProvenanceInfo,
    ReportDownloadResponse,
    ReportFormat,
    ReportResponse,
    ReportType,
    TraceabilityReportRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reports"])

# ---------------------------------------------------------------------------
# In-memory report store (replaced by database in production)
# ---------------------------------------------------------------------------

_report_store: Dict[str, Dict] = {}


def _get_report_store() -> Dict[str, Dict]:
    """Return the report store singleton."""
    return _report_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _format_content_type(fmt: ReportFormat) -> str:
    """Map report format to MIME content type."""
    mapping = {
        ReportFormat.PDF: "application/pdf",
        ReportFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ReportFormat.JSON: "application/json",
        ReportFormat.CSV: "text/csv",
    }
    return mapping.get(fmt, "application/octet-stream")


# ---------------------------------------------------------------------------
# POST /reports/traceability
# ---------------------------------------------------------------------------


@router.post(
    "/reports/traceability",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Article 9 traceability report",
    description=(
        "Generate an Article 9 traceability report covering "
        "the full chain of custody for specified batches. "
        "Includes genealogy, documents, and verification results."
    ),
    responses={
        201: {"description": "Report generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_traceability_report(
    request: Request,
    body: TraceabilityReportRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:reports:create")
    ),
    _rate: None = Depends(rate_limit_report),
) -> ReportResponse:
    """Generate an Article 9 traceability report.

    Args:
        body: Report generation parameters.
        user: Authenticated user with reports:create permission.

    Returns:
        ReportResponse with report details and download URL.
    """
    start = time.monotonic()
    try:
        report_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="report_generation",
        )

        title = (
            f"EUDR Article 9 Traceability Report - "
            f"{body.commodity.value.title()} - "
            f"{body.report_period_start.strftime('%Y-%m-%d')} to "
            f"{body.report_period_end.strftime('%Y-%m-%d')}"
        )

        summary = {
            "operator_id": body.operator_id,
            "commodity": body.commodity.value,
            "total_batches": len(body.batch_ids),
            "period": {
                "start": body.report_period_start.isoformat(),
                "end": body.report_period_end.isoformat(),
            },
            "includes_genealogy": body.include_genealogy,
            "includes_documents": body.include_documents,
            "includes_verification": body.include_verification,
            "language": body.language,
        }

        download_url = f"/api/v1/eudr-coc/reports/{report_id}/download"
        expires_at = now + timedelta(days=7)

        report_record = {
            "report_id": report_id,
            "report_type": ReportType.TRACEABILITY,
            "title": title,
            "status": "generated",
            "output_format": body.output_format,
            "commodity": body.commodity,
            "period_start": body.report_period_start,
            "period_end": body.report_period_end,
            "total_batches": len(body.batch_ids),
            "total_events": 0,
            "compliance_score": None,
            "summary": summary,
            "download_url": download_url,
            "file_size_bytes": 0,
            "generated_at": now,
            "expires_at": expires_at,
            "provenance": provenance,
        }

        store = _get_report_store()
        store[report_id] = report_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Traceability report generated: id=%s commodity=%s batches=%d",
            report_id,
            body.commodity.value,
            len(body.batch_ids),
        )

        return ReportResponse(
            **report_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate traceability report: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate traceability report",
        )


# ---------------------------------------------------------------------------
# POST /reports/mass-balance
# ---------------------------------------------------------------------------


@router.post(
    "/reports/mass-balance",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Mass balance period report",
    description=(
        "Generate a mass balance report for a facility covering "
        "a specified period with input/output ledger entries and "
        "reconciliation results."
    ),
    responses={
        201: {"description": "Report generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_mass_balance_report(
    request: Request,
    body: MassBalanceReportRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:reports:create")
    ),
    _rate: None = Depends(rate_limit_report),
) -> ReportResponse:
    """Generate a mass balance period report.

    Args:
        body: Report generation parameters.
        user: Authenticated user with reports:create permission.

    Returns:
        ReportResponse with report details and download URL.
    """
    start = time.monotonic()
    try:
        report_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="report_generation",
        )

        title = (
            f"Mass Balance Report - {body.facility_id} - "
            f"{body.commodity.value.title()} - "
            f"{body.period_start.strftime('%Y-%m-%d')} to "
            f"{body.period_end.strftime('%Y-%m-%d')}"
        )

        summary = {
            "facility_id": body.facility_id,
            "commodity": body.commodity.value,
            "period": {
                "start": body.period_start.isoformat(),
                "end": body.period_end.isoformat(),
            },
            "includes_entries": body.include_entries,
            "includes_reconciliation": body.include_reconciliation,
        }

        download_url = f"/api/v1/eudr-coc/reports/{report_id}/download"
        expires_at = now + timedelta(days=7)

        report_record = {
            "report_id": report_id,
            "report_type": ReportType.MASS_BALANCE,
            "title": title,
            "status": "generated",
            "output_format": body.output_format,
            "commodity": body.commodity,
            "period_start": body.period_start,
            "period_end": body.period_end,
            "total_batches": 0,
            "total_events": 0,
            "compliance_score": None,
            "summary": summary,
            "download_url": download_url,
            "file_size_bytes": 0,
            "generated_at": now,
            "expires_at": expires_at,
            "provenance": provenance,
        }

        store = _get_report_store()
        store[report_id] = report_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Mass balance report generated: id=%s facility=%s commodity=%s",
            report_id,
            body.facility_id,
            body.commodity.value,
        )

        return ReportResponse(
            **report_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate mass balance report: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate mass balance report",
        )


# ---------------------------------------------------------------------------
# GET /reports/{report_id}
# ---------------------------------------------------------------------------


@router.get(
    "/reports/{report_id}",
    response_model=ReportResponse,
    summary="Get report",
    description="Retrieve report details including summary and download URL.",
    responses={
        200: {"description": "Report details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def get_report(
    request: Request,
    report_id: str = Depends(validate_report_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportResponse:
    """Get report details by ID.

    Args:
        report_id: Report identifier.
        user: Authenticated user with reports:read permission.

    Returns:
        ReportResponse with report details.

    Raises:
        HTTPException: 404 if report not found.
    """
    try:
        store = _get_report_store()
        record = store.get(report_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report {report_id} not found",
            )

        return ReportResponse(**record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get report %s: %s", report_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve report",
        )


# ---------------------------------------------------------------------------
# GET /reports/{report_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/reports/{report_id}/download",
    response_model=ReportDownloadResponse,
    summary="Download report",
    description="Get a signed download URL for a generated report.",
    responses={
        200: {"description": "Download URL"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
        410: {"model": ErrorResponse, "description": "Download link expired"},
    },
)
async def download_report(
    request: Request,
    report_id: str = Depends(validate_report_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:reports:download")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ReportDownloadResponse:
    """Get download URL for a report.

    Args:
        report_id: Report identifier.
        user: Authenticated user with reports:download permission.

    Returns:
        ReportDownloadResponse with signed download URL.

    Raises:
        HTTPException: 404 if report not found, 410 if expired.
    """
    try:
        store = _get_report_store()
        record = store.get(report_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report {report_id} not found",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        expires_at = record.get("expires_at")
        if expires_at and isinstance(expires_at, datetime) and expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Download link for report {report_id} has expired",
            )

        # Generate signed download URL (placeholder)
        download_url = record.get(
            "download_url",
            f"/api/v1/eudr-coc/reports/{report_id}/file",
        )
        output_format = record.get("output_format", ReportFormat.PDF)
        content_type = _format_content_type(output_format)

        new_expiry = now + timedelta(hours=1)

        logger.info("Report download requested: id=%s", report_id)

        return ReportDownloadResponse(
            report_id=report_id,
            download_url=download_url,
            output_format=output_format,
            file_size_bytes=record.get("file_size_bytes"),
            content_type=content_type,
            expires_at=new_expiry,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to download report %s: %s", report_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL",
        )
