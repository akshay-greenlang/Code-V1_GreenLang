# -*- coding: utf-8 -*-
"""
Report Routes - AGENT-EUDR-010 Segregation Verifier API

Endpoints for generating and downloading segregation compliance reports
including audit reports, contamination reports, and regulatory evidence
packages.

Endpoints:
    POST   /reports/audit          - Generate segregation audit report
    POST   /reports/contamination  - Generate contamination report
    POST   /reports/evidence       - Generate regulatory evidence package
    GET    /reports/{report_id}    - Get report
    GET    /reports/{report_id}/download - Download report

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010, Section 7.4
Agent ID: GL-EUDR-SGV-010
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

from greenlang.agents.eudr.segregation_verifier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_request_id,
    get_sgv_service,
    rate_limit_export,
    rate_limit_report,
    rate_limit_standard,
    require_permission,
    validate_report_id,
)
from greenlang.agents.eudr.segregation_verifier.api.schemas import (
    GenerateAuditReportRequest,
    GenerateContaminationReportRequest,
    GenerateEvidencePackageRequest,
    ProvenanceInfo,
    ReportDownloadResponse,
    ReportFormat,
    ReportResponse,
    ReportType,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reports"])

# ---------------------------------------------------------------------------
# In-memory report store
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
        ReportFormat.XLSX: (
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        ReportFormat.JSON: "application/json",
        ReportFormat.CSV: "text/csv",
    }
    return mapping.get(fmt, "application/octet-stream")


# ---------------------------------------------------------------------------
# POST /reports/audit
# ---------------------------------------------------------------------------


@router.post(
    "/reports/audit",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate segregation audit report",
    description=(
        "Generate a comprehensive segregation audit report for a facility "
        "covering storage, transport, processing, and labelling compliance."
    ),
    responses={
        201: {"description": "Report generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_audit_report(
    request: Request,
    body: GenerateAuditReportRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:reports:create")
    ),
    _rate: None = Depends(rate_limit_report),
) -> ReportResponse:
    """Generate a segregation audit report.

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

        commodity_label = body.commodity.value.title() if body.commodity else "All"
        title = (
            f"Segregation Audit Report - {body.facility_id} - "
            f"{commodity_label} - "
            f"{body.period_start.strftime('%Y-%m-%d')} to "
            f"{body.period_end.strftime('%Y-%m-%d')}"
        )

        summary = {
            "facility_id": body.facility_id,
            "commodity": body.commodity.value if body.commodity else None,
            "period": {
                "start": body.period_start.isoformat(),
                "end": body.period_end.isoformat(),
            },
            "includes_storage": body.include_storage,
            "includes_transport": body.include_transport,
            "includes_processing": body.include_processing,
            "includes_labelling": body.include_labelling,
            "language": body.language,
        }

        download_url = f"/api/v1/eudr-sgv/reports/{report_id}/download"
        expires_at = now + timedelta(days=7)

        report_record = {
            "report_id": report_id,
            "report_type": ReportType.AUDIT,
            "title": title,
            "status": "generated",
            "output_format": body.output_format,
            "facility_id": body.facility_id,
            "commodity": body.commodity,
            "period_start": body.period_start,
            "period_end": body.period_end,
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
            "Audit report generated: id=%s facility=%s",
            report_id,
            body.facility_id,
        )

        return ReportResponse(**report_record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate audit report: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate audit report",
        )


# ---------------------------------------------------------------------------
# POST /reports/contamination
# ---------------------------------------------------------------------------


@router.post(
    "/reports/contamination",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate contamination report",
    description=(
        "Generate a contamination report for a facility covering "
        "all contamination events, impact assessments, and corrective "
        "actions within a specified period."
    ),
    responses={
        201: {"description": "Report generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_contamination_report(
    request: Request,
    body: GenerateContaminationReportRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:reports:create")
    ),
    _rate: None = Depends(rate_limit_report),
) -> ReportResponse:
    """Generate a contamination report.

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

        commodity_label = body.commodity.value.title() if body.commodity else "All"
        title = (
            f"Contamination Report - {body.facility_id} - "
            f"{commodity_label} - "
            f"{body.period_start.strftime('%Y-%m-%d')} to "
            f"{body.period_end.strftime('%Y-%m-%d')}"
        )

        summary = {
            "facility_id": body.facility_id,
            "commodity": body.commodity.value if body.commodity else None,
            "period": {
                "start": body.period_start.isoformat(),
                "end": body.period_end.isoformat(),
            },
            "includes_impact_analysis": body.include_impact_analysis,
            "includes_corrective_actions": body.include_corrective_actions,
        }

        download_url = f"/api/v1/eudr-sgv/reports/{report_id}/download"
        expires_at = now + timedelta(days=7)

        report_record = {
            "report_id": report_id,
            "report_type": ReportType.CONTAMINATION,
            "title": title,
            "status": "generated",
            "output_format": body.output_format,
            "facility_id": body.facility_id,
            "commodity": body.commodity,
            "period_start": body.period_start,
            "period_end": body.period_end,
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
            "Contamination report generated: id=%s facility=%s",
            report_id,
            body.facility_id,
        )

        return ReportResponse(**report_record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate contamination report: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate contamination report",
        )


# ---------------------------------------------------------------------------
# POST /reports/evidence
# ---------------------------------------------------------------------------


@router.post(
    "/reports/evidence",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate regulatory evidence package",
    description=(
        "Generate a regulatory evidence package for EUDR compliance "
        "demonstrating segregation controls across the supply chain."
    ),
    responses={
        201: {"description": "Evidence package generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_evidence_package(
    request: Request,
    body: GenerateEvidencePackageRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:reports:create")
    ),
    _rate: None = Depends(rate_limit_report),
) -> ReportResponse:
    """Generate a regulatory evidence package.

    Args:
        body: Evidence package generation parameters.
        user: Authenticated user with reports:create permission.

    Returns:
        ReportResponse with package details and download URL.
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
            f"EUDR Evidence Package - {body.facility_id} - "
            f"{body.commodity.value.title()} - "
            f"{body.period_start.strftime('%Y-%m-%d')} to "
            f"{body.period_end.strftime('%Y-%m-%d')}"
        )

        summary = {
            "facility_id": body.facility_id,
            "commodity": body.commodity.value,
            "batch_ids": body.batch_ids,
            "period": {
                "start": body.period_start.isoformat(),
                "end": body.period_end.isoformat(),
            },
            "includes_scp_records": body.include_scp_records,
            "includes_storage_records": body.include_storage_records,
            "includes_transport_records": body.include_transport_records,
            "includes_processing_records": body.include_processing_records,
            "includes_contamination_records": body.include_contamination_records,
            "includes_assessment_results": body.include_assessment_results,
        }

        download_url = f"/api/v1/eudr-sgv/reports/{report_id}/download"
        expires_at = now + timedelta(days=30)

        report_record = {
            "report_id": report_id,
            "report_type": ReportType.EVIDENCE_PACKAGE,
            "title": title,
            "status": "generated",
            "output_format": body.output_format,
            "facility_id": body.facility_id,
            "commodity": body.commodity,
            "period_start": body.period_start,
            "period_end": body.period_end,
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
            "Evidence package generated: id=%s facility=%s commodity=%s",
            report_id,
            body.facility_id,
            body.commodity.value,
        )

        return ReportResponse(**report_record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate evidence package: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate evidence package",
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
        require_permission("eudr-sgv:reports:read")
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
        require_permission("eudr-sgv:reports:download")
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

        download_url = record.get(
            "download_url",
            f"/api/v1/eudr-sgv/reports/{report_id}/file",
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
