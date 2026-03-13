# -*- coding: utf-8 -*-
"""
Reporting Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for compliance report generation, listing, downloading, and
scheduling recurring reports per EUDR Articles 11, 12, 33.

Endpoints:
    POST /reports/generate             - Generate a compliance report
    GET  /reports                      - List reports (paginated)
    GET  /reports/{report_id}/download - Download a report
    POST /reports/schedule             - Schedule a recurring report

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, ReportGenerationEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_report_engine,
    rate_limit_export,
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
    ReportDownloadResponse,
    ReportEntry,
    ReportFormatEnum,
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportListResponse,
    ReportScheduleRequest,
    ReportScheduleResponse,
    ReportStatusEnum,
    ReportTypeEnum,
    ScheduleEntry,
    ScheduleFrequencyEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["Reporting"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /reports/generate
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=ReportGenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate a compliance report",
    description=(
        "Generate a compliance report in the specified format (PDF, XLSX, "
        "CSV, JSON, HTML). Reports can cover compliance summaries, due "
        "diligence reports, red flag reports, audit summaries, "
        "certification status, and regulatory filings."
    ),
    responses={
        202: {"description": "Report generation started"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_report(
    request: Request,
    body: ReportGenerateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:report:create")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ReportGenerateResponse:
    """Generate a compliance report.

    Args:
        body: Report generation request.
        user: Authenticated user with report:create permission.

    Returns:
        ReportGenerateResponse with report record (status: queued/generating).
    """
    start = time.monotonic()

    try:
        engine = get_report_engine()
        result = engine.generate(
            report_type=body.report_type.value,
            format=body.format.value,
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            commodity=body.commodity.value if body.commodity else None,
            country_code=body.country_code,
            date_from=body.date_from,
            date_to=body.date_to,
            include_details=body.include_details,
            title=body.title,
            requested_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report generation failed: insufficient data",
            )

        report = ReportEntry(
            report_id=result.get("report_id", ""),
            report_type=ReportTypeEnum(
                result.get("report_type", body.report_type.value)
            ),
            format=ReportFormatEnum(result.get("format", body.format.value)),
            status=ReportStatusEnum(result.get("status", "queued")),
            title=result.get("title", body.title),
            operator_id=result.get("operator_id", body.operator_id),
            supplier_id=result.get("supplier_id", body.supplier_id),
            file_size_bytes=result.get("file_size_bytes"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"report_generate:{body.report_type.value}:{body.format.value}",
            report.report_id,
        )

        logger.info(
            "Report generation started: id=%s type=%s format=%s user=%s",
            report.report_id,
            body.report_type.value,
            body.format.value,
            user.user_id,
        )

        return ReportGenerateResponse(
            report=report,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ReportGenerationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed",
        )


# ---------------------------------------------------------------------------
# GET /reports
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List reports",
    description=(
        "Retrieve a paginated list of generated reports with optional "
        "filtering by type, format, status, and operator."
    ),
    responses={
        200: {"description": "Reports retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_reports(
    request: Request,
    report_type: Optional[ReportTypeEnum] = Query(
        None, description="Filter by report type"
    ),
    report_format: Optional[ReportFormatEnum] = Query(
        None, alias="format", description="Filter by format"
    ),
    report_status: Optional[ReportStatusEnum] = Query(
        None, alias="status", description="Filter by status"
    ),
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:report:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportListResponse:
    """List generated reports with pagination.

    Args:
        report_type: Optional type filter.
        report_format: Optional format filter.
        report_status: Optional status filter.
        operator_id: Optional operator filter.
        pagination: Pagination parameters.
        user: Authenticated user with report:read permission.

    Returns:
        ReportListResponse with paginated reports.
    """
    start = time.monotonic()

    try:
        engine = get_report_engine()
        result = engine.list_reports(
            report_type=report_type.value if report_type else None,
            format=report_format.value if report_format else None,
            status=report_status.value if report_status else None,
            operator_id=operator_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        reports = []
        for r in result.get("reports", []):
            reports.append(
                ReportEntry(
                    report_id=r.get("report_id", ""),
                    report_type=ReportTypeEnum(r.get("report_type", "compliance_summary")),
                    format=ReportFormatEnum(r.get("format", "pdf")),
                    status=ReportStatusEnum(r.get("status", "queued")),
                    title=r.get("title"),
                    operator_id=r.get("operator_id"),
                    supplier_id=r.get("supplier_id"),
                    file_size_bytes=r.get("file_size_bytes"),
                    completed_at=r.get("completed_at"),
                )
            )

        total = result.get("total", len(reports))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"report_list:{report_type}:{report_status}",
            str(total),
        )

        logger.info(
            "Reports listed: total=%d user=%s",
            total,
            user.user_id,
        )

        return ReportListResponse(
            reports=reports,
            total_reports=total,
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
                data_sources=["ReportGenerationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report listing failed",
        )


# ---------------------------------------------------------------------------
# GET /reports/{report_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}/download",
    response_model=ReportDownloadResponse,
    summary="Download a report",
    description=(
        "Get a pre-signed download URL for a completed report. The URL "
        "expires after 1 hour. Returns 404 if report not found or not "
        "yet completed."
    ),
    responses={
        200: {"description": "Download URL generated"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found or not ready"},
    },
)
async def download_report(
    report_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:report:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportDownloadResponse:
    """Get download URL for a completed report.

    Args:
        report_id: Report to download.
        user: Authenticated user with report:read permission.

    Returns:
        ReportDownloadResponse with pre-signed download URL.
    """
    start = time.monotonic()

    try:
        engine = get_report_engine()
        result = engine.get_download(report_id=report_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found or not yet completed: {report_id}",
            )

        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"report_download:{report_id}",
            str(expires_at),
        )

        logger.info(
            "Report download URL generated: id=%s user=%s",
            report_id,
            user.user_id,
        )

        return ReportDownloadResponse(
            report_id=report_id,
            download_url=result.get("download_url", ""),
            expires_at=result.get("expires_at", expires_at),
            format=ReportFormatEnum(result.get("format", "pdf")),
            file_size_bytes=result.get("file_size_bytes"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ReportGenerationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report download failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report download failed",
        )


# ---------------------------------------------------------------------------
# POST /reports/schedule
# ---------------------------------------------------------------------------


@router.post(
    "/schedule",
    response_model=ReportScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule a recurring report",
    description=(
        "Schedule a report for automatic recurring generation (daily, weekly, "
        "biweekly, monthly, or quarterly). Reports are sent to the specified "
        "email recipients upon generation."
    ),
    responses={
        201: {"description": "Report schedule created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def schedule_report(
    request: Request,
    body: ReportScheduleRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:report:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ReportScheduleResponse:
    """Schedule a recurring report.

    Args:
        body: Report schedule request.
        user: Authenticated user with report:create permission.

    Returns:
        ReportScheduleResponse with created schedule.
    """
    start = time.monotonic()

    try:
        engine = get_report_engine()
        result = engine.schedule(
            report_type=body.report_type.value,
            format=body.format.value,
            frequency=body.frequency.value,
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            commodity=body.commodity.value if body.commodity else None,
            recipients=body.recipients,
            title=body.title,
            start_date=body.start_date,
            end_date=body.end_date,
            created_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report scheduling failed: invalid parameters",
            )

        schedule = ScheduleEntry(
            schedule_id=result.get("schedule_id", ""),
            report_type=ReportTypeEnum(
                result.get("report_type", body.report_type.value)
            ),
            frequency=ScheduleFrequencyEnum(
                result.get("frequency", body.frequency.value)
            ),
            format=ReportFormatEnum(result.get("format", body.format.value)),
            recipients=result.get("recipients", body.recipients),
            next_run=result.get("next_run"),
            active=result.get("active", True),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"report_schedule:{body.report_type.value}:{body.frequency.value}",
            schedule.schedule_id,
        )

        logger.info(
            "Report scheduled: id=%s type=%s frequency=%s recipients=%d user=%s",
            schedule.schedule_id,
            body.report_type.value,
            body.frequency.value,
            len(body.recipients),
            user.user_id,
        )

        return ReportScheduleResponse(
            schedule=schedule,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ReportGenerationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report scheduling failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report scheduling failed",
        )
