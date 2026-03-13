# -*- coding: utf-8 -*-
"""
Reporting Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for compliance report generation, listing, download, and
DDS (Due Diligence Statement) mitigation section data extraction.
Supports 7 report types across 5 output formats and 5 EU languages.

Endpoints (4):
    POST /reports/generate                  - Generate mitigation report
    GET  /reports                           - List generated reports
    GET  /reports/{report_id}/download      - Download report file
    GET  /reports/dds-section/{operator_id} - Get DDS mitigation section data

RBAC Permissions:
    eudr-rma:reports:generate - Generate new reports
    eudr-rma:reports:read     - List and download reports
    eudr-rma:reports:dds      - Access DDS section data

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 9: Report Generation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_rma_service,
    rate_limit_report,
    rate_limit_standard,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    DDSSectionResponse,
    ErrorResponse,
    GenerateReportRequest,
    PaginatedMeta,
    ReportDownloadResponse,
    ReportEntry,
    ReportListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["Reporting"])


def _report_dict_to_entry(r: Dict[str, Any]) -> ReportEntry:
    """Convert report dictionary to ReportEntry schema."""
    return ReportEntry(
        report_id=r.get("report_id", ""),
        operator_id=r.get("operator_id", ""),
        report_type=r.get("report_type", ""),
        format=r.get("format", "pdf"),
        language=r.get("language", "en"),
        report_scope=r.get("report_scope", r.get("scope", {})),
        s3_key=r.get("s3_key"),
        provenance_hash=r.get("provenance_hash", ""),
        generated_at=r.get("generated_at"),
    )


# ---------------------------------------------------------------------------
# POST /reports/generate
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=ReportEntry,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate mitigation report",
    description=(
        "Generate a compliance-ready mitigation report. Supported report types: "
        "dds_mitigation (Article 12(2)(d) DDS section), authority_package "
        "(competent authority submission), annual_review (Article 8(3) review), "
        "supplier_scorecard, portfolio_summary, risk_mapping, "
        "effectiveness_analysis. Output formats: PDF, JSON, HTML, XLSX, XML. "
        "Languages: en, fr, de, es, pt (all 5 EU official languages)."
    ),
    responses={
        202: {"description": "Report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid report parameters"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded (10/min)"},
    },
)
async def generate_report(
    request: Request,
    body: GenerateReportRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:reports:generate")),
    _rate: None = Depends(rate_limit_report),
    service: Any = Depends(get_rma_service),
) -> ReportEntry:
    """Generate a mitigation report."""
    valid_report_types = {
        "dds_mitigation",
        "authority_package",
        "annual_review",
        "supplier_scorecard",
        "portfolio_summary",
        "risk_mapping",
        "effectiveness_analysis",
    }
    if body.report_type not in valid_report_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid report_type: {body.report_type}. "
                f"Valid types: {', '.join(sorted(valid_report_types))}"
            ),
        )

    valid_formats = {"pdf", "json", "html", "xlsx", "xml"}
    if body.format not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format: {body.format}. Valid formats: {', '.join(sorted(valid_formats))}",
        )

    valid_languages = {"en", "fr", "de", "es", "pt"}
    if body.language not in valid_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language: {body.language}. Valid languages: {', '.join(sorted(valid_languages))}",
        )

    try:
        result = await service.generate_report(
            operator_id=body.operator_id,
            report_type=body.report_type,
            output_format=body.format,
            language=body.language,
            scope=body.scope,
            generated_by=user.user_id,
        )

        data = result if isinstance(result, dict) else {}

        logger.info(
            "Report generation initiated: type=%s format=%s operator=%s user=%s",
            body.report_type, body.format, body.operator_id, user.user_id,
        )

        return _report_dict_to_entry(data)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Report generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate report",
        )


# ---------------------------------------------------------------------------
# GET /reports
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List generated reports",
    description=(
        "Retrieve a paginated list of generated reports with optional filters "
        "by report type, format, language, and date range."
    ),
    responses={200: {"description": "Reports listed"}},
)
async def list_reports(
    request: Request,
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    format: Optional[str] = Query(None, alias="format", description="Filter by output format"),
    language: Optional[str] = Query(None, description="Filter by language"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-rma:reports:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> ReportListResponse:
    """List generated reports with optional filters."""
    try:
        result = await service.list_reports(
            operator_id=user.operator_id,
            report_type=report_type,
            output_format=format,
            language=language,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        reports_raw = result.get("reports", []) if isinstance(result, dict) else []
        total = result.get("total", 0) if isinstance(result, dict) else 0
        reports = [_report_dict_to_entry(r) for r in reports_raw]

        return ReportListResponse(
            reports=reports,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as e:
        logger.error("Report list failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve reports",
        )


# ---------------------------------------------------------------------------
# GET /reports/dds-section/{operator_id}  (before /{report_id} to avoid conflict)
# ---------------------------------------------------------------------------


@router.get(
    "/dds-section/{operator_id}",
    response_model=DDSSectionResponse,
    summary="Get DDS mitigation section data",
    description=(
        "Retrieve structured data for the Article 12(2)(d) mitigation measures "
        "section of the Due Diligence Statement. Returns risk findings count, "
        "active plans, deployed measures, average risk reduction, compliance "
        "status, and evidence completeness. Used by the DDS generation pipeline "
        "to populate the mitigation section."
    ),
    responses={
        200: {"description": "DDS section data generated"},
        404: {"model": ErrorResponse, "description": "No mitigation data for operator"},
    },
)
async def get_dds_section(
    request: Request,
    operator_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:reports:dds")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> DDSSectionResponse:
    """Get DDS mitigation section data for an operator."""
    try:
        result = await service.get_dds_mitigation_section(
            operator_id=operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No mitigation data found for operator {operator_id}",
            )

        data = result if isinstance(result, dict) else {}
        return DDSSectionResponse(
            operator_id=operator_id,
            article_12_2_d_data=data.get("article_12_2_d_data", {}),
            risk_findings_count=data.get("risk_findings_count", 0),
            active_plans_count=data.get("active_plans_count", 0),
            mitigation_measures_deployed=data.get("mitigation_measures_deployed", 0),
            average_risk_reduction_pct=Decimal(str(data.get("average_risk_reduction_pct", 0))),
            compliance_status=data.get("compliance_status", ""),
            evidence_completeness_pct=Decimal(str(data.get("evidence_completeness_pct", 0))),
            provenance_hash=data.get("provenance_hash", ""),
            generated_at=data.get("generated_at", datetime.now(timezone.utc)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("DDS section generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate DDS section data",
        )


# ---------------------------------------------------------------------------
# GET /reports/{report_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{report_id}/download",
    response_model=ReportDownloadResponse,
    summary="Download report file",
    description=(
        "Generate a pre-signed S3 download URL for a previously generated "
        "report file. The URL is valid for 1 hour. Returns the file format, "
        "size, and provenance hash for integrity verification."
    ),
    responses={
        200: {"description": "Download URL generated"},
        404: {"model": ErrorResponse, "description": "Report not found"},
    },
)
async def download_report(
    request: Request,
    report_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:reports:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> ReportDownloadResponse:
    """Generate a pre-signed download URL for a report."""
    validate_uuid(report_id, "report_id")

    try:
        result = await service.get_report_download(
            report_id=report_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report {report_id} not found",
            )

        data = result if isinstance(result, dict) else {}
        return ReportDownloadResponse(
            report_id=report_id,
            download_url=data.get("download_url", ""),
            format=data.get("format", "pdf"),
            file_size_bytes=data.get("file_size_bytes", 0),
            expires_at=data.get("expires_at"),
            provenance_hash=data.get("provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report download failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL",
        )
