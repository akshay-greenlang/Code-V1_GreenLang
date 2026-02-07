# -*- coding: utf-8 -*-
"""
Audit Export REST API Routes - SEC-005

FastAPI router for audit event export:

    POST /api/v1/audit/export              - Start export job
    GET  /api/v1/audit/export/{job_id}     - Get export status
    GET  /api/v1/audit/export/{job_id}/download - Download export

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional

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

    class ExportFiltersRequest(BaseModel):
        """Filters for export query."""

        since: Optional[datetime] = Field(
            None, description="Start of time range (ISO 8601)"
        )
        until: Optional[datetime] = Field(
            None, description="End of time range (ISO 8601)"
        )
        event_types: Optional[List[str]] = Field(
            None, description="Event types to include"
        )
        categories: Optional[List[str]] = Field(
            None, description="Event categories to include"
        )
        severities: Optional[List[str]] = Field(
            None, description="Severity levels to include"
        )
        organization_id: Optional[str] = Field(
            None, description="Filter by organization ID"
        )
        user_id: Optional[str] = Field(
            None, description="Filter by user ID"
        )

    class ExportRequest(BaseModel):
        """Request to start an export job."""

        model_config = ConfigDict(
            json_schema_extra={
                "examples": [
                    {
                        "format": "csv",
                        "compress": True,
                        "filters": {
                            "since": "2026-02-01T00:00:00Z",
                            "until": "2026-02-06T23:59:59Z",
                            "categories": ["authentication", "authorization"],
                        },
                    }
                ]
            }
        )

        format: str = Field(
            "csv",
            description="Export format: csv, json, parquet",
        )
        compress: bool = Field(
            True,
            description="Whether to compress output (gzip for CSV/JSON)",
        )
        filters: Optional[ExportFiltersRequest] = Field(
            None, description="Query filters"
        )

    class ExportJobResponse(BaseModel):
        """Export job status response."""

        job_id: str = Field(..., description="Unique job identifier")
        status: str = Field(..., description="Job status: pending, processing, completed, failed")
        export_format: str = Field(..., description="Export format")
        progress_percent: float = Field(0.0, description="Progress percentage")
        total_records: int = Field(0, description="Total records to export")
        exported_records: int = Field(0, description="Records exported so far")
        created_at: datetime = Field(..., description="Job creation timestamp")
        started_at: Optional[datetime] = Field(None, description="Processing start time")
        completed_at: Optional[datetime] = Field(None, description="Completion time")
        error_message: Optional[str] = Field(None, description="Error message if failed")
        download_url: Optional[str] = Field(None, description="Download URL if completed")
        file_size_bytes: Optional[int] = Field(None, description="File size in bytes")

    class ExportJobCreatedResponse(BaseModel):
        """Response when export job is created."""

        job_id: str = Field(..., description="Unique job identifier")
        status: str = Field("pending", description="Initial status")
        message: str = Field(..., description="Status message")
        check_status_url: str = Field(..., description="URL to check job status")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_export_service() -> Any:
    """FastAPI dependency that provides the AuditExportService.

    Returns:
        The AuditExportService singleton.

    Raises:
        HTTPException 503: If service is not available.
    """
    try:
        from greenlang.infrastructure.audit_service.export.export_service import (
            get_export_service,
        )
        return get_export_service()
    except (ImportError, RuntimeError) as exc:
        logger.error("Export service not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Export service is not available.",
        )


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user ID from request headers."""
    return request.headers.get("x-user-id")


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from greenlang.infrastructure.audit_service.export.export_service import (
        ExportFormat,
        ExportFilters,
    )

    export_router = APIRouter(
        prefix="/api/v1/audit/export",
        tags=["Audit Export"],
        responses={
            400: {"description": "Bad Request"},
            404: {"description": "Export Not Found"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @export_router.post(
        "",
        response_model=ExportJobCreatedResponse,
        status_code=202,
        summary="Start audit export",
        description="Start an async audit event export job.",
        operation_id="start_audit_export",
    )
    async def start_audit_export(
        request: Request,
        body: ExportRequest,
        service: Any = Depends(_get_export_service),
    ) -> ExportJobCreatedResponse:
        """Start an audit event export job.

        Creates a background job to export audit events to the specified
        format. Use the returned job_id to check status and download.

        Supported formats:
        - csv: Comma-separated values (gzip compressed by default)
        - json: JSON Lines (JSONL) format
        - parquet: Apache Parquet columnar format (requires PyArrow)

        Args:
            request: HTTP request.
            body: Export request.
            service: Injected export service.

        Returns:
            Job creation response with job_id.
        """
        # Parse format
        try:
            format_enum = ExportFormat(body.format.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid format '{body.format}'. "
                       f"Allowed: {[f.value for f in ExportFormat]}",
            )

        # Convert filters
        filters = None
        if body.filters:
            filters = ExportFilters(
                since=body.filters.since,
                until=body.filters.until,
                event_types=body.filters.event_types,
                categories=body.filters.categories,
                severities=body.filters.severities,
                organization_id=body.filters.organization_id,
                user_id=body.filters.user_id,
            )

        try:
            user_id = _get_user_id(request)

            job_id = await service.start_export(
                format=format_enum,
                filters=filters,
                compress=body.compress,
                requested_by=user_id,
            )

            return ExportJobCreatedResponse(
                job_id=job_id,
                status="pending",
                message=f"Export job started ({format_enum.value} format).",
                check_status_url=f"/api/v1/audit/export/{job_id}",
            )

        except Exception as exc:
            logger.exception("Failed to create export job")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create export job: {exc}",
            )

    @export_router.get(
        "/{job_id}",
        response_model=ExportJobResponse,
        summary="Get export job status",
        description="Check the status of an export job.",
        operation_id="get_export_job_status",
    )
    async def get_export_job_status(
        job_id: str,
        service: Any = Depends(_get_export_service),
    ) -> ExportJobResponse:
        """Get export job status.

        Args:
            job_id: Export job identifier.
            service: Injected export service.

        Returns:
            Job status details.

        Raises:
            HTTPException 404: If job not found.
        """
        job = await service.get_job_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Export job '{job_id}' not found.",
            )

        # Get download URL if completed
        download_url = None
        if job.status.value == "completed":
            download_url = f"/api/v1/audit/export/{job_id}/download"

        return ExportJobResponse(
            job_id=job.job_id,
            status=job.status.value,
            export_format=job.export_format.value,
            progress_percent=job.progress_percent,
            total_records=job.total_records,
            exported_records=job.exported_records,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            download_url=download_url,
            file_size_bytes=job.file_size_bytes,
        )

    @export_router.get(
        "/{job_id}/download",
        summary="Download export",
        description="Download a completed export file.",
        operation_id="download_export",
    )
    async def download_export(
        job_id: str,
        service: Any = Depends(_get_export_service),
    ):
        """Download a completed export.

        Args:
            job_id: Export job identifier.
            service: Injected export service.

        Returns:
            File download response.

        Raises:
            HTTPException 404: If job not found or not completed.
        """
        job = await service.get_job_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Export job '{job_id}' not found.",
            )

        if job.status.value != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Export not ready. Current status: {job.status.value}",
            )

        # Get download URL or file path
        download_url = await service.get_download_url(job_id)

        if not download_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export file not found.",
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
                detail="Export file not found on server.",
            )

        # Determine content type
        content_type_map = {
            "csv": "text/csv",
            "json": "application/x-ndjson",
            "parquet": "application/vnd.apache.parquet",
        }

        # Check for compression
        if download_url.endswith(".gz"):
            content_type = "application/gzip"
        else:
            content_type = content_type_map.get(
                job.export_format.value, "application/octet-stream"
            )

        filename = os.path.basename(download_url)

        return FileResponse(
            path=download_url,
            media_type=content_type,
            filename=filename,
        )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(export_router)
    except ImportError:
        pass  # auth_service not available

else:
    export_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - export_router is None")


__all__ = ["export_router"]
