# -*- coding: utf-8 -*-
"""
Audit Compliance Report REST API Routes - SEC-005

FastAPI router for compliance report generation:

    POST /api/v1/audit/reports/soc2        - Generate SOC2 report
    POST /api/v1/audit/reports/iso27001    - Generate ISO 27001 report
    POST /api/v1/audit/reports/gdpr        - Generate GDPR report
    GET  /api/v1/audit/reports/{job_id}    - Get report status
    GET  /api/v1/audit/reports/{job_id}/download - Download report

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class ReportRequest(BaseModel):
        """Request to generate a compliance report."""

        model_config = ConfigDict(
            json_schema_extra={
                "examples": [
                    {
                        "period": "last_30_days",
                        "format": "pdf",
                    }
                ]
            }
        )

        period: str = Field(
            "last_30_days",
            description="Report period: last_7_days, last_30_days, last_90_days, last_year, custom",
        )
        format: str = Field(
            "pdf",
            description="Output format: pdf, json, html, csv",
        )
        period_start: Optional[datetime] = Field(
            None, description="Custom period start (required if period=custom)"
        )
        period_end: Optional[datetime] = Field(
            None, description="Custom period end (required if period=custom)"
        )
        organization_id: Optional[str] = Field(
            None, description="Filter by organization ID"
        )

    class ReportJobResponse(BaseModel):
        """Report job status response."""

        job_id: str = Field(..., description="Unique job identifier")
        report_type: str = Field(..., description="Report type: soc2, iso27001, gdpr")
        status: str = Field(..., description="Job status: pending, processing, completed, failed")
        progress_percent: float = Field(0.0, description="Progress percentage")
        format: str = Field(..., description="Output format")
        created_at: datetime = Field(..., description="Job creation timestamp")
        started_at: Optional[datetime] = Field(None, description="Processing start time")
        completed_at: Optional[datetime] = Field(None, description="Completion time")
        error_message: Optional[str] = Field(None, description="Error message if failed")
        download_url: Optional[str] = Field(None, description="Download URL if completed")
        file_size_bytes: Optional[int] = Field(None, description="File size in bytes")

    class ReportJobCreatedResponse(BaseModel):
        """Response when report job is created."""

        job_id: str = Field(..., description="Unique job identifier")
        status: str = Field("pending", description="Initial status")
        message: str = Field(..., description="Status message")
        check_status_url: str = Field(..., description="URL to check job status")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_report_service() -> Any:
    """FastAPI dependency that provides the ComplianceReportService.

    Returns:
        The ComplianceReportService singleton.

    Raises:
        HTTPException 503: If service is not available.
    """
    try:
        from greenlang.infrastructure.audit_service.reporting.report_service import (
            get_report_service,
        )
        return get_report_service()
    except (ImportError, RuntimeError) as exc:
        logger.error("Report service not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Report service is not available.",
        )


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user ID from request headers."""
    return request.headers.get("x-user-id")


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from greenlang.infrastructure.audit_service.reporting.report_service import (
        ReportFormat,
        ReportPeriod,
    )

    report_router = APIRouter(
        prefix="/api/v1/audit/reports",
        tags=["Audit Reports"],
        responses={
            400: {"description": "Bad Request"},
            404: {"description": "Report Not Found"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @report_router.post(
        "/soc2",
        response_model=ReportJobCreatedResponse,
        status_code=202,
        summary="Generate SOC2 report",
        description="Start SOC2 Type II compliance report generation.",
        operation_id="generate_soc2_report",
    )
    async def generate_soc2_report(
        request: Request,
        body: ReportRequest,
        service: Any = Depends(_get_report_service),
    ) -> ReportJobCreatedResponse:
        """Generate SOC2 Type II compliance report.

        Creates a background job to generate the report. Use the returned
        job_id to check status and download when complete.

        Args:
            request: HTTP request.
            body: Report generation request.
            service: Injected report service.

        Returns:
            Job creation response with job_id.
        """
        return await _create_report_job("soc2", body, request, service)

    @report_router.post(
        "/iso27001",
        response_model=ReportJobCreatedResponse,
        status_code=202,
        summary="Generate ISO 27001 report",
        description="Start ISO 27001 ISMS compliance report generation.",
        operation_id="generate_iso27001_report",
    )
    async def generate_iso27001_report(
        request: Request,
        body: ReportRequest,
        service: Any = Depends(_get_report_service),
    ) -> ReportJobCreatedResponse:
        """Generate ISO 27001 ISMS compliance report.

        Args:
            request: HTTP request.
            body: Report generation request.
            service: Injected report service.

        Returns:
            Job creation response with job_id.
        """
        return await _create_report_job("iso27001", body, request, service)

    @report_router.post(
        "/gdpr",
        response_model=ReportJobCreatedResponse,
        status_code=202,
        summary="Generate GDPR report",
        description="Start GDPR compliance report generation (Art. 30 ROPA).",
        operation_id="generate_gdpr_report",
    )
    async def generate_gdpr_report(
        request: Request,
        body: ReportRequest,
        service: Any = Depends(_get_report_service),
    ) -> ReportJobCreatedResponse:
        """Generate GDPR compliance report.

        Args:
            request: HTTP request.
            body: Report generation request.
            service: Injected report service.

        Returns:
            Job creation response with job_id.
        """
        return await _create_report_job("gdpr", body, request, service)

    @report_router.get(
        "/{job_id}",
        response_model=ReportJobResponse,
        summary="Get report job status",
        description="Check the status of a report generation job.",
        operation_id="get_report_job_status",
    )
    async def get_report_job_status(
        job_id: str,
        service: Any = Depends(_get_report_service),
    ) -> ReportJobResponse:
        """Get report job status.

        Args:
            job_id: Report job identifier.
            service: Injected report service.

        Returns:
            Job status details.

        Raises:
            HTTPException 404: If job not found.
        """
        job = await service.get_job_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report job '{job_id}' not found.",
            )

        # Get download URL if completed
        download_url = None
        if job.status.value == "completed":
            download_url = f"/api/v1/audit/reports/{job_id}/download"

        return ReportJobResponse(
            job_id=job.job_id,
            report_type=job.report_type,
            status=job.status.value,
            progress_percent=job.progress_percent,
            format=job.format.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            download_url=download_url,
            file_size_bytes=job.file_size_bytes,
        )

    @report_router.get(
        "/{job_id}/download",
        summary="Download report",
        description="Download a completed compliance report.",
        operation_id="download_report",
    )
    async def download_report(
        job_id: str,
        service: Any = Depends(_get_report_service),
    ):
        """Download a completed report.

        Args:
            job_id: Report job identifier.
            service: Injected report service.

        Returns:
            File download response.

        Raises:
            HTTPException 404: If job not found or not completed.
        """
        job = await service.get_job_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report job '{job_id}' not found.",
            )

        if job.status.value != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Report not ready. Current status: {job.status.value}",
            )

        # Get download URL or file path
        download_url = await service.get_download_url(job_id)

        if not download_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report file not found.",
            )

        # If it's an S3 URL, redirect
        if download_url.startswith("http"):
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=download_url)

        # Otherwise, serve the local file
        import os
        if not os.path.exists(download_url):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report file not found on server.",
            )

        # Determine content type
        content_type_map = {
            "pdf": "application/pdf",
            "json": "application/json",
            "html": "text/html",
            "csv": "text/csv",
        }
        content_type = content_type_map.get(job.format.value, "application/octet-stream")

        filename = os.path.basename(download_url)

        return FileResponse(
            path=download_url,
            media_type=content_type,
            filename=filename,
        )

    async def _create_report_job(
        report_type: str,
        body: ReportRequest,
        request: Request,
        service: Any,
    ) -> ReportJobCreatedResponse:
        """Create a report generation job.

        Args:
            report_type: Type of report (soc2, iso27001, gdpr).
            body: Report request.
            request: HTTP request.
            service: Report service.

        Returns:
            Job creation response.
        """
        # Parse period
        try:
            period = ReportPeriod(body.period.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid period '{body.period}'. "
                       f"Allowed: {[p.value for p in ReportPeriod]}",
            )

        # Parse format
        try:
            format_enum = ReportFormat(body.format.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid format '{body.format}'. "
                       f"Allowed: {[f.value for f in ReportFormat]}",
            )

        # Validate custom period
        if period == ReportPeriod.CUSTOM:
            if not body.period_start or not body.period_end:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Custom period requires period_start and period_end.",
                )

        try:
            user_id = _get_user_id(request)

            job_id = await service.generate_report(
                report_type=report_type,
                period=period,
                format=format_enum,
                period_start=body.period_start,
                period_end=body.period_end,
                organization_id=body.organization_id,
                requested_by=user_id,
            )

            return ReportJobCreatedResponse(
                job_id=job_id,
                status="pending",
                message=f"{report_type.upper()} report generation started.",
                check_status_url=f"/api/v1/audit/reports/{job_id}",
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        except Exception as exc:
            logger.exception("Failed to create report job")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create report job: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(report_router)
    except ImportError:
        pass  # auth_service not available

else:
    report_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - report_router is None")


__all__ = ["report_router"]
