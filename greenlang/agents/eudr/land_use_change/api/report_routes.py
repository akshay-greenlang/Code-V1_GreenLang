# -*- coding: utf-8 -*-
"""
Report Routes - AGENT-EUDR-005 Land Use Change Detector API

Endpoints for compliance report generation, retrieval, download in
multiple formats, and batch report generation.

Endpoints:
    POST /generate            - Generate a land use change report
    GET  /{report_id}         - Get a generated report
    GET  /{report_id}/download - Download report in specified format
    POST /batch               - Generate batch reports

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from greenlang.agents.eudr.land_use_change.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_land_use_service,
    get_request_id,
    rate_limit_export,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    ReportBatchRequest,
    ReportBatchResponse,
    ReportGenerateRequest,
    ReportResult,
    ReportSection,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Land Use Change Reports"])

# ---------------------------------------------------------------------------
# In-memory report store (replaced by database in production)
# ---------------------------------------------------------------------------

_report_store: Dict[str, Dict[str, Any]] = {}


def _get_report_store() -> Dict[str, Dict[str, Any]]:
    """Return the report store. Replaceable for testing."""
    return _report_store


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=ReportResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate a land use change report",
    description=(
        "Generate a comprehensive land use change report for specified "
        "plots. Supports report types: classification_summary, "
        "transition_analysis, trajectory_report, cutoff_verification, "
        "risk_assessment, urban_encroachment, comprehensive. Output "
        "formats: JSON, CSV, PDF, XLSX. Reports include analysis "
        "results, evidence, maps, and recommendations."
    ),
    responses={
        202: {"description": "Report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_report(
    body: ReportGenerateRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:reports:write")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ReportResult:
    """Generate a land use change report.

    Compiles analysis data for specified plots into a structured
    report with evidence, verdicts, and recommendations.

    Args:
        body: Report generation request with type, plots, and format.
        user: Authenticated user with reports:write permission.

    Returns:
        ReportResult with generated report data.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    report_id = f"rpt-luc-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Report generation: user=%s report_id=%s type=%s "
        "format=%s plots=%d",
        user.user_id,
        report_id,
        body.report_type.value,
        body.format.value,
        len(body.plot_ids),
    )

    try:
        service = get_land_use_service()

        # Extract options
        include_maps = True
        include_evidence = True
        operator_id = user.operator_id or user.user_id
        language = "en"

        if body.options:
            include_maps = body.options.include_maps
            include_evidence = body.options.include_evidence
            language = body.options.language
            if body.options.operator_id:
                operator_id = body.options.operator_id

        result = service.generate_report(
            report_type=body.report_type.value,
            plot_ids=body.plot_ids,
            output_format=body.format.value,
            include_maps=include_maps,
            include_evidence=include_evidence,
            operator_id=operator_id,
            title=body.title,
            date_range_start=body.date_range_start,
            date_range_end=body.date_range_end,
        )

        elapsed = time.monotonic() - start

        # Build sections
        sections = []
        raw_sections = getattr(result, "sections", [])
        for sec in raw_sections:
            sections.append(
                ReportSection(
                    section_id=getattr(sec, "section_id", ""),
                    title=getattr(sec, "title", ""),
                    content_type=getattr(
                        sec, "content_type", "text"
                    ),
                    content=getattr(sec, "content", None),
                    order=getattr(sec, "order", 0),
                )
            )

        # Build download URL for non-JSON formats
        download_url = None
        if body.format.value != "json":
            download_url = getattr(result, "download_url", None)

        response = ReportResult(
            request_id=get_request_id(),
            report_id=getattr(result, "report_id", report_id),
            report_type=body.report_type.value,
            format=body.format.value,
            status="generated",
            title=getattr(
                result, "title",
                body.title or f"Land Use Change Report - {body.report_type.value}",
            ),
            summary=getattr(result, "summary", ""),
            sections=sections,
            plot_count=len(body.plot_ids),
            download_url=download_url,
            file_size_bytes=getattr(result, "file_size_bytes", None),
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        effective_report_id = getattr(
            result, "report_id", report_id
        )
        store = _get_report_store()
        store[effective_report_id] = {
            "report_id": effective_report_id,
            "report_type": body.report_type.value,
            "format": body.format.value,
            "status": "generated",
            "plot_ids": body.plot_ids,
            "response_data": response.model_dump(mode="json"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": user.user_id,
        }

        logger.info(
            "Report generated: report_id=%s type=%s format=%s "
            "plots=%d sections=%d elapsed_ms=%.1f",
            effective_report_id,
            body.report_type.value,
            body.format.value,
            len(body.plot_ids),
            len(sections),
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Report generation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Report generation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /{report_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}",
    response_model=ReportResult,
    status_code=status.HTTP_200_OK,
    summary="Get a generated report",
    description="Retrieve a previously generated report by report ID.",
    responses={
        200: {"description": "Report data"},
        404: {"model": ErrorResponse, "description": "Report not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_report(
    report_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReportResult:
    """Retrieve a generated report by report ID.

    Args:
        report_id: Report identifier to look up.
        user: Authenticated user with reports:read permission.

    Returns:
        ReportResult with report data.

    Raises:
        HTTPException: 404 if report not found.
    """
    store = _get_report_store()

    if report_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found",
        )

    record = store[report_id]
    return ReportResult(**record["response_data"])


# ---------------------------------------------------------------------------
# GET /{report_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Download report",
    description=(
        "Download a generated report in its specified format. For JSON "
        "format, returns the report data directly. For PDF, CSV, and "
        "XLSX formats, returns a download URL or the file content."
    ),
    responses={
        200: {"description": "Report download"},
        404: {"model": ErrorResponse, "description": "Report not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def download_report(
    report_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> JSONResponse:
    """Download a report in its specified format.

    For JSON reports, returns the full report data. For other formats,
    returns the download URL or a redirect.

    Args:
        report_id: Report identifier to download.
        user: Authenticated user with reports:read permission.

    Returns:
        JSONResponse with report content or download URL.

    Raises:
        HTTPException: 404 if report not found.
    """
    store = _get_report_store()

    if report_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found",
        )

    record = store[report_id]
    report_format = record.get("format", "json")

    logger.info(
        "Report download: user=%s report_id=%s format=%s",
        user.user_id,
        report_id,
        report_format,
    )

    if report_format == "json":
        return JSONResponse(
            content=record["response_data"],
            headers={
                "Content-Disposition": (
                    f'attachment; filename="report_{report_id}.json"'
                ),
            },
        )

    # For non-JSON formats, return download URL info
    response_data = record.get("response_data", {})
    download_url = response_data.get("download_url")

    if download_url:
        return JSONResponse(
            content={
                "report_id": report_id,
                "format": report_format,
                "download_url": download_url,
                "message": (
                    f"Use the download_url to retrieve the "
                    f"{report_format.upper()} report"
                ),
            }
        )

    return JSONResponse(
        content={
            "report_id": report_id,
            "format": report_format,
            "message": (
                f"Report in {report_format.upper()} format is being "
                "prepared. Check back shortly."
            ),
            "status": record.get("status", "processing"),
        }
    )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=ReportBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch report generation",
    description=(
        "Generate multiple reports in a single request. Supports up "
        "to 50 report configurations per batch. Each configuration "
        "can specify different report types, plots, and formats."
    ),
    responses={
        202: {"description": "Batch report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_batch_reports(
    body: ReportBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:reports:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> ReportBatchResponse:
    """Generate multiple reports in batch.

    Args:
        body: Batch request with list of report configurations.
        user: Authenticated user with reports:write permission.

    Returns:
        ReportBatchResponse with results for each report.
    """
    start = time.monotonic()
    total = len(body.report_configs)

    logger.info(
        "Batch report generation: user=%s reports=%d",
        user.user_id,
        total,
    )

    results: List[ReportResult] = []
    successful = 0
    failed = 0

    try:
        service = get_land_use_service()
        store = _get_report_store()

        for config in body.report_configs:
            report_id = f"rpt-luc-{uuid.uuid4().hex[:12]}"

            try:
                # Extract options
                include_maps = True
                include_evidence = True
                operator_id = user.operator_id or user.user_id

                if config.options:
                    include_maps = config.options.include_maps
                    include_evidence = config.options.include_evidence
                    if config.options.operator_id:
                        operator_id = config.options.operator_id

                result = service.generate_report(
                    report_type=config.report_type.value,
                    plot_ids=config.plot_ids,
                    output_format=config.format.value,
                    include_maps=include_maps,
                    include_evidence=include_evidence,
                    operator_id=operator_id,
                    title=config.title,
                )

                # Build sections
                sections = []
                raw_sections = getattr(result, "sections", [])
                for sec in raw_sections:
                    sections.append(
                        ReportSection(
                            section_id=getattr(
                                sec, "section_id", ""
                            ),
                            title=getattr(sec, "title", ""),
                            content_type=getattr(
                                sec, "content_type", "text"
                            ),
                            content=getattr(sec, "content", None),
                            order=getattr(sec, "order", 0),
                        )
                    )

                download_url = None
                if config.format.value != "json":
                    download_url = getattr(
                        result, "download_url", None
                    )

                effective_report_id = getattr(
                    result, "report_id", report_id
                )

                report_result = ReportResult(
                    request_id=get_request_id(),
                    report_id=effective_report_id,
                    report_type=config.report_type.value,
                    format=config.format.value,
                    status="generated",
                    title=getattr(
                        result, "title",
                        config.title
                        or f"LUC Report - {config.report_type.value}",
                    ),
                    summary=getattr(result, "summary", ""),
                    sections=sections,
                    plot_count=len(config.plot_ids),
                    download_url=download_url,
                    file_size_bytes=getattr(
                        result, "file_size_bytes", None
                    ),
                    data_sources=getattr(
                        result, "data_sources", []
                    ),
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(report_result)
                successful += 1

                store[effective_report_id] = {
                    "report_id": effective_report_id,
                    "report_type": config.report_type.value,
                    "format": config.format.value,
                    "status": "generated",
                    "plot_ids": config.plot_ids,
                    "response_data": report_result.model_dump(
                        mode="json"
                    ),
                    "generated_at": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "generated_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch report failed for config %s: %s",
                    config.report_type.value,
                    exc,
                )
                failed += 1

        elapsed = time.monotonic() - start

        logger.info(
            "Batch reports completed: user=%s total=%d "
            "successful=%d failed=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed,
            elapsed * 1000,
        )

        return ReportBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch reports failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch report generation failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
