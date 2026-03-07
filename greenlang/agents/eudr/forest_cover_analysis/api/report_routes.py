# -*- coding: utf-8 -*-
"""
Report Routes - AGENT-EUDR-004 Forest Cover Analysis API

Endpoints for compliance report generation, retrieval, download in
multiple formats, and batch report generation.

Endpoints:
    POST /generate            - Generate a compliance report
    GET  /{report_id}         - Get a generated report
    GET  /{report_id}/download - Download report in specified format
    POST /batch               - Generate batch reports

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from greenlang.agents.eudr.forest_cover_analysis.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_forest_cover_service,
    get_request_id,
    rate_limit_export,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    BatchReportRequest,
    ComplianceReportResponse,
    GenerateReportRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Compliance Reports"])


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
    response_model=ComplianceReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate a compliance report",
    description=(
        "Generate a comprehensive EUDR compliance report for a production "
        "plot. Supports report types: eudr_compliance, due_diligence, "
        "risk_assessment, monitoring_summary. Output formats: JSON, CSV, "
        "PDF, XLSX. Reports include forest cover analysis results, "
        "deforestation-free verdicts, evidence, and recommendations."
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
    body: GenerateReportRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:reports:write")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ComplianceReportResponse:
    """Generate a compliance report for a production plot.

    Compiles forest cover analysis data into a structured compliance
    report with evidence, verdict, and recommendations.

    Args:
        body: Report generation request with plot, type, and format.
        user: Authenticated user with reports:write permission.

    Returns:
        ComplianceReportResponse with generated report data.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    report_id = f"rpt-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Report generation: user=%s plot_id=%s type=%s format=%s",
        user.user_id,
        body.plot_id,
        body.report_type,
        body.format.value,
    )

    try:
        service = get_forest_cover_service()

        result = service.generate_report(
            plot_id=body.plot_id,
            report_type=body.report_type,
            output_format=body.format.value,
            include_evidence=body.include_evidence,
            include_maps=body.include_maps,
            operator_id=body.operator_id or user.operator_id or user.user_id,
        )

        elapsed = time.monotonic() - start

        # Build download URL for non-JSON formats
        download_url = None
        if body.format.value != "json":
            download_url = getattr(result, "download_url", None)

        response = ComplianceReportResponse(
            request_id=get_request_id(),
            report_id=getattr(result, "report_id", report_id),
            plot_id=body.plot_id,
            report_type=body.report_type,
            format=body.format.value,
            status="generated",
            title=getattr(result, "title", ""),
            summary=getattr(result, "summary", ""),
            verdict=getattr(result, "verdict", None),
            risk_level=getattr(result, "risk_level", None),
            sections=getattr(result, "sections", []),
            evidence_items=getattr(result, "evidence_items", []),
            recommendations=getattr(result, "recommendations", []),
            operator_id=body.operator_id or user.operator_id or user.user_id,
            download_url=download_url,
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        effective_report_id = getattr(result, "report_id", report_id)
        store = _get_report_store()
        store[effective_report_id] = {
            "report_id": effective_report_id,
            "plot_id": body.plot_id,
            "report_type": body.report_type,
            "format": body.format.value,
            "status": "generated",
            "response_data": response.model_dump(mode="json"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": user.user_id,
        }

        logger.info(
            "Report generated: report_id=%s plot_id=%s type=%s "
            "elapsed_ms=%.1f",
            effective_report_id,
            body.plot_id,
            body.report_type,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Report generation error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Report generation failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{report_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}",
    response_model=ComplianceReportResponse,
    summary="Get a generated report",
    description="Retrieve a previously generated compliance report by its ID.",
    responses={
        200: {"description": "Generated report"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def get_report(
    report_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceReportResponse:
    """Retrieve a previously generated report.

    Args:
        report_id: Report identifier.
        user: Authenticated user with reports:read permission.

    Returns:
        ComplianceReportResponse with stored report data.

    Raises:
        HTTPException: 404 if report not found.
    """
    logger.info(
        "Report retrieval: user=%s report_id=%s",
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

    response_data = report.get("response_data", {})
    return ComplianceReportResponse(**response_data)


# ---------------------------------------------------------------------------
# GET /{report_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}/download",
    summary="Download report in specified format",
    description=(
        "Download the compliance report in the format specified during "
        "generation. For JSON, returns data directly. For CSV/PDF/XLSX, "
        "returns the formatted content or download metadata."
    ),
    responses={
        200: {"description": "Report download"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def download_report(
    report_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:reports:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> JSONResponse:
    """Download report in the requested format.

    Args:
        report_id: Report identifier.
        user: Authenticated user with reports:read permission.

    Returns:
        JSON response with report content or download metadata.

    Raises:
        HTTPException: 404 if report not found.
    """
    logger.info(
        "Report download: user=%s report_id=%s",
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

    output_format = report.get("format", "json")
    response_data = report.get("response_data", {})

    if output_format == "json":
        return JSONResponse(
            content=response_data,
            headers={
                "Content-Disposition": (
                    f"attachment; filename=report_{report_id}.json"
                ),
            },
        )
    elif output_format == "csv":
        csv_content = _generate_csv_content(response_data)
        return JSONResponse(
            content={
                "report_id": report_id,
                "format": "csv",
                "content": csv_content,
                "filename": f"report_{report_id}.csv",
            },
            headers={
                "Content-Disposition": (
                    f"attachment; filename=report_{report_id}.csv"
                ),
            },
        )
    elif output_format == "xlsx":
        return JSONResponse(
            content={
                "report_id": report_id,
                "format": "xlsx",
                "xlsx_data": response_data,
                "filename": f"report_{report_id}.xlsx",
                "note": "Use an XLSX rendering service to generate the file",
            },
            headers={
                "Content-Disposition": (
                    f"attachment; filename=report_{report_id}.xlsx"
                ),
            },
        )
    else:
        # PDF format
        return JSONResponse(
            content={
                "report_id": report_id,
                "format": "pdf",
                "pdf_data": response_data,
                "filename": f"report_{report_id}.pdf",
                "note": "Use a PDF rendering service to generate the file",
            },
            headers={
                "Content-Disposition": (
                    f"attachment; filename=report_{report_id}.pdf"
                ),
            },
        )


def _generate_csv_content(data: Dict[str, Any]) -> str:
    """Generate CSV string from report data.

    Args:
        data: Report response data dictionary.

    Returns:
        CSV-formatted string with key report fields.
    """
    lines = [
        "plot_id,report_type,verdict,risk_level,confidence,"
        "forest_cover_pct,processing_time_ms",
    ]

    plot_id = data.get("plot_id", "")
    report_type = data.get("report_type", "")
    verdict = data.get("verdict", "")
    risk_level = data.get("risk_level", "")
    # Extract confidence from sections or provenance
    sections = data.get("sections", [])
    confidence = ""
    for section in sections:
        if "confidence" in section:
            confidence = str(section["confidence"])
            break

    lines.append(
        f"{plot_id},{report_type},{verdict},{risk_level},"
        f"{confidence},,{data.get('processing_time_ms', '')}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=List[ComplianceReportResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate batch reports",
    description=(
        "Generate compliance reports for multiple plots in a single "
        "request. Supports up to 1,000 reports per batch."
    ),
    responses={
        202: {"description": "Batch report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_reports(
    body: BatchReportRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:reports:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> List[ComplianceReportResponse]:
    """Generate compliance reports for multiple plots.

    Args:
        body: Batch request with list of report generation requests.
        user: Authenticated user with reports:write permission.

    Returns:
        List of ComplianceReportResponse for each report.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Batch report generation: user=%s reports=%d",
        user.user_id,
        len(body.reports),
    )

    try:
        service = get_forest_cover_service()
        store = _get_report_store()
        results = []

        for report_req in body.reports:
            report_id = f"rpt-{uuid.uuid4().hex[:12]}"
            operator = (
                report_req.operator_id
                or user.operator_id
                or user.user_id
            )

            try:
                result = service.generate_report(
                    plot_id=report_req.plot_id,
                    report_type=report_req.report_type,
                    output_format=report_req.format.value,
                    include_evidence=report_req.include_evidence,
                    include_maps=report_req.include_maps,
                    operator_id=operator,
                )

                download_url = None
                if report_req.format.value != "json":
                    download_url = getattr(result, "download_url", None)

                response = ComplianceReportResponse(
                    request_id=get_request_id(),
                    report_id=getattr(result, "report_id", report_id),
                    plot_id=report_req.plot_id,
                    report_type=report_req.report_type,
                    format=report_req.format.value,
                    status="generated",
                    title=getattr(result, "title", ""),
                    summary=getattr(result, "summary", ""),
                    verdict=getattr(result, "verdict", None),
                    risk_level=getattr(result, "risk_level", None),
                    sections=getattr(result, "sections", []),
                    evidence_items=getattr(result, "evidence_items", []),
                    recommendations=getattr(result, "recommendations", []),
                    operator_id=operator,
                    download_url=download_url,
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(result, "provenance_hash", ""),
                )

                effective_id = getattr(result, "report_id", report_id)
                store[effective_id] = {
                    "report_id": effective_id,
                    "plot_id": report_req.plot_id,
                    "report_type": report_req.report_type,
                    "format": report_req.format.value,
                    "status": "generated",
                    "response_data": response.model_dump(mode="json"),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "generated_by": user.user_id,
                }

                results.append(response)

            except Exception as rpt_exc:
                logger.warning(
                    "Batch report: plot %s failed: %s",
                    report_req.plot_id,
                    rpt_exc,
                )
                results.append(ComplianceReportResponse(
                    request_id=get_request_id(),
                    report_id=report_id,
                    plot_id=report_req.plot_id,
                    report_type=report_req.report_type,
                    format=report_req.format.value,
                    status="failed",
                    summary=f"Report generation failed: {rpt_exc}",
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch report completed: user=%s reports=%d elapsed_ms=%.1f",
            user.user_id,
            len(results),
            elapsed * 1000,
        )

        return results

    except Exception as exc:
        logger.error(
            "Batch report generation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch report generation failed due to an internal error",
        )
