# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Quarterly Engine API Routes v1.1

FastAPI router implementing the quarterly CBAM report management API.
Provides endpoints for report generation, retrieval, submission, amendment,
deadline monitoring, and notification configuration.

Per EU CBAM Regulation 2023/956 and Implementing Regulation 2023/1773:
  - Articles 6-8: Quarterly reporting obligation
  - Article 9: Amendment/correction provisions (T+60 day window)
  - Article 6(1): Submission deadline (T+30 after quarter end)

All calculation endpoints use deterministic arithmetic (ZERO HALLUCINATION).

Endpoints:
    GET  /calendar/{year}                              - Quarterly reporting calendar
    GET  /current                                      - Current quarter details
    POST /reports/generate                             - Trigger report generation
    GET  /reports                                      - List quarterly reports
    GET  /reports/{report_id}                          - Get report details
    GET  /reports/{report_id}/xml                      - Download XML output
    GET  /reports/{report_id}/summary                  - Download markdown summary
    PUT  /reports/{report_id}/submit                   - Submit report for review
    POST /reports/{report_id}/amend                    - Create amendment
    GET  /reports/{report_id}/amendments               - List amendments
    GET  /reports/{report_id}/amendments/{id}/diff     - Get amendment diff
    GET  /deadlines                                    - Upcoming deadlines
    GET  /deadlines/overdue                            - Overdue reports
    PUT  /deadlines/{alert_id}/acknowledge             - Acknowledge alert
    GET  /notifications                                - Notification history
    PUT  /notifications/configure                      - Configure recipients

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Path, Body
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..models import (
    AlertLevel,
    AmendmentReason,
    CBAMSector,
    CalculationMethod,
    DeadlineAlert,
    NotificationConfig,
    NotificationLogEntry,
    NotificationType,
    QuarterlyPeriod,
    QuarterlyReport,
    QuarterlyReportPeriod,
    ReportAmendment,
    ReportStatus,
    ShipmentAggregation,
    compute_sha256,
    quantize_decimal,
    validate_status_transition,
)
from ..quarterly_scheduler import QuarterlySchedulerEngine
from ..report_assembler import ReportAssemblerEngine
from ..amendment_manager import AmendmentManagerEngine
from ..deadline_tracker import DeadlineTrackerEngine
from ..notification_service import NotificationService

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER DEFINITION
# ============================================================================

router = APIRouter(
    prefix="/api/v1/cbam/quarterly",
    tags=["cbam-quarterly"],
    responses={
        404: {"description": "Resource not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class ReportGenerateRequest(BaseModel):
    """Request body for triggering quarterly report generation."""

    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer EORI number or internal identifier"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Reporting year"
    )
    quarter: QuarterlyPeriod = Field(
        ...,
        description="Reporting quarter (Q1-Q4)"
    )
    shipments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of shipment records to include in the report"
    )
    force_regenerate: bool = Field(
        default=False,
        description="Whether to regenerate even if a draft already exists"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "importer_id": "NL123456789012",
                "year": 2026,
                "quarter": "Q1",
                "shipments": [
                    {
                        "cn_code": "72031000",
                        "country_of_origin": "TR",
                        "quantity_mt": 500.0,
                        "direct_emissions_tCO2e": 925.0,
                        "indirect_emissions_tCO2e": 75.0,
                    }
                ],
                "force_regenerate": False,
            }
        }


class ReportSubmitRequest(BaseModel):
    """Request body for submitting a report for review."""

    submitted_by: str = Field(
        ...,
        min_length=1,
        description="User or system identifier submitting the report"
    )
    declaration: bool = Field(
        ...,
        description="Confirmation that data is accurate and complete"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "submitted_by": "compliance-officer@acme.eu",
                "declaration": True,
            }
        }


class AmendmentRequest(BaseModel):
    """Request body for creating a report amendment."""

    reason: AmendmentReason = Field(
        ...,
        description="Categorized reason for the amendment"
    )
    changes_summary: str = Field(
        ...,
        min_length=10,
        description="Human-readable description of what changed and why"
    )
    changes: Dict[str, Any] = Field(
        ...,
        description="Map of field names to new values"
    )
    amended_by: str = Field(
        ...,
        min_length=1,
        description="User or system identifier creating the amendment"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "reason": "new_supplier_data",
                "changes_summary": "Updated steel emissions for installation TR-001 with verified EPD data",
                "changes": {
                    "total_direct_emissions": "2800.000",
                },
                "amended_by": "compliance@acme.eu",
            }
        }


class DeadlineAcknowledgeRequest(BaseModel):
    """Request body for acknowledging a deadline alert."""

    acknowledged_by: str = Field(
        ...,
        min_length=1,
        description="User acknowledging the alert"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes about the acknowledgement"
    )


class NotificationConfigureRequest(BaseModel):
    """Request body for configuring notification recipients."""

    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer EORI or internal identifier"
    )
    email_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses for notifications"
    )
    webhook_urls: List[str] = Field(
        default_factory=list,
        description="Webhook endpoints for external integrations"
    )
    alert_levels_enabled: List[AlertLevel] = Field(
        default_factory=lambda: list(AlertLevel),
        description="Which alert levels trigger notifications"
    )
    notification_types_enabled: List[NotificationType] = Field(
        default_factory=lambda: list(NotificationType),
        description="Which notification types are active"
    )
    quiet_hours_start: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="Start of quiet hours (UTC hour, 0-23)"
    )
    quiet_hours_end: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="End of quiet hours (UTC hour, 0-23)"
    )


class ReportListItem(BaseModel):
    """Condensed report item for listing endpoints."""

    report_id: str
    period_label: str
    importer_id: str
    status: ReportStatus
    shipments_count: int
    total_embedded_emissions: str
    version: int
    created_at: datetime
    submitted_at: Optional[datetime] = None
    provenance_hash: str


class QuarterCalendarEntry(BaseModel):
    """A single quarter in the annual reporting calendar."""

    quarter: QuarterlyPeriod
    start_date: date
    end_date: date
    submission_deadline: date
    amendment_deadline: date
    is_transitional: bool
    status: str = Field(
        default="upcoming",
        description="upcoming, current, past"
    )


class QuarterCalendarResponse(BaseModel):
    """Full annual reporting calendar response."""

    year: int
    quarters: List[QuarterCalendarEntry]
    is_transitional_year: bool
    is_definitive_year: bool


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ============================================================================
# ENGINE INSTANCES (lazy singletons)
# ============================================================================

_scheduler: Optional[QuarterlySchedulerEngine] = None
_assembler: Optional[ReportAssemblerEngine] = None
_amendment_mgr: Optional[AmendmentManagerEngine] = None
_deadline_tracker: Optional[DeadlineTrackerEngine] = None
_notification_svc: Optional[NotificationService] = None


def _get_scheduler() -> QuarterlySchedulerEngine:
    """Get or create the quarterly scheduler engine singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = QuarterlySchedulerEngine()
    return _scheduler


def _get_assembler() -> ReportAssemblerEngine:
    """Get or create the report assembler engine singleton."""
    global _assembler
    if _assembler is None:
        _assembler = ReportAssemblerEngine()
    return _assembler


def _get_amendment_mgr() -> AmendmentManagerEngine:
    """Get or create the amendment manager engine."""
    global _amendment_mgr
    if _amendment_mgr is None:
        _amendment_mgr = AmendmentManagerEngine()
    return _amendment_mgr


def _get_deadline_tracker() -> DeadlineTrackerEngine:
    """Get or create the deadline tracker engine."""
    global _deadline_tracker
    if _deadline_tracker is None:
        _deadline_tracker = DeadlineTrackerEngine()
    return _deadline_tracker


def _get_notification_svc() -> NotificationService:
    """Get or create the notification service."""
    global _notification_svc
    if _notification_svc is None:
        _notification_svc = NotificationService()
    return _notification_svc


# ============================================================================
# IN-MEMORY STORES (production would use PostgreSQL)
# ============================================================================

# report_id -> QuarterlyReport
_reports_store: Dict[str, QuarterlyReport] = {}

# importer_id -> list of report_ids
_importer_reports: Dict[str, List[str]] = {}

# alert_id -> DeadlineAlert
_alerts_store: Dict[str, DeadlineAlert] = {}

# importer_id -> NotificationConfig
_notification_configs: Dict[str, NotificationConfig] = {}

# Notification log entries
_notification_log: List[NotificationLogEntry] = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _store_report(report: QuarterlyReport) -> None:
    """Store a report in the in-memory registry."""
    _reports_store[report.report_id] = report
    if report.importer_id not in _importer_reports:
        _importer_reports[report.importer_id] = []
    if report.report_id not in _importer_reports[report.importer_id]:
        _importer_reports[report.importer_id].append(report.report_id)


def _get_report(report_id: str) -> QuarterlyReport:
    """Retrieve a report or raise 404."""
    report = _reports_store.get(report_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {report_id}"
        )
    return report


def _report_to_list_item(report: QuarterlyReport) -> ReportListItem:
    """Convert a QuarterlyReport to a condensed list item."""
    return ReportListItem(
        report_id=report.report_id,
        period_label=report.period.period_label,
        importer_id=report.importer_id,
        status=report.status,
        shipments_count=report.shipments_count,
        total_embedded_emissions=str(report.total_embedded_emissions),
        version=report.version,
        created_at=report.created_at,
        submitted_at=report.submitted_at,
        provenance_hash=report.provenance_hash,
    )


# ============================================================================
# CALENDAR ENDPOINTS
# ============================================================================

@router.get(
    "/calendar/{year}",
    response_model=APIResponse,
    summary="Get quarterly reporting calendar",
    description=(
        "Returns the full quarterly reporting calendar for a given year, "
        "including submission and amendment deadlines for each quarter."
    ),
)
async def get_quarterly_calendar(
    year: int = Path(..., ge=2023, le=2099, description="Reporting year"),
) -> APIResponse:
    """Get the quarterly reporting calendar for a given year."""
    start_time = time.time()
    logger.info("Fetching quarterly calendar for year=%d", year)

    try:
        scheduler = _get_scheduler()
        quarters: List[QuarterCalendarEntry] = []
        today = date.today()

        for qp in QuarterlyPeriod:
            period = scheduler.get_quarter(year, qp)
            # Determine status relative to today
            if today < period.start_date:
                status = "upcoming"
            elif today > period.end_date:
                status = "past"
            else:
                status = "current"

            quarters.append(QuarterCalendarEntry(
                quarter=qp,
                start_date=period.start_date,
                end_date=period.end_date,
                submission_deadline=period.submission_deadline,
                amendment_deadline=period.amendment_deadline,
                is_transitional=period.is_transitional,
                status=status,
            ))

        calendar_response = QuarterCalendarResponse(
            year=year,
            quarters=quarters,
            is_transitional_year=year <= 2025,
            is_definitive_year=year >= 2026,
        )

        processing_ms = (time.time() - start_time) * 1000
        logger.info("Calendar retrieved for year=%d in %.1fms", year, processing_ms)

        return APIResponse(
            success=True,
            message=f"Quarterly calendar for {year}",
            data=calendar_response.model_dump(mode="json"),
            processing_time_ms=round(processing_ms, 2),
        )

    except Exception as exc:
        logger.error("Failed to get calendar for year=%d: %s", year, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/current",
    response_model=APIResponse,
    summary="Get current quarter details",
    description=(
        "Returns details of the current quarterly reporting period, "
        "including deadlines and phase (transitional vs definitive)."
    ),
)
async def get_current_quarter() -> APIResponse:
    """Get the current quarterly reporting period."""
    start_time = time.time()
    logger.info("Fetching current quarter details")

    try:
        scheduler = _get_scheduler()
        period = scheduler.get_current_quarter()
        days_to_deadline = scheduler.get_days_until_deadline(period)

        result = {
            "period": period.model_dump(mode="json"),
            "period_label": period.period_label,
            "days_until_submission_deadline": days_to_deadline,
            "is_transitional": period.is_transitional,
            "is_definitive": period.is_definitive,
        }

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Current quarter: %s, days to deadline: %d, in %.1fms",
            period.period_label,
            days_to_deadline,
            processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Current quarter: {period.period_label}",
            data=result,
            processing_time_ms=round(processing_ms, 2),
        )

    except Exception as exc:
        logger.error("Failed to get current quarter: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# REPORT GENERATION & RETRIEVAL ENDPOINTS
# ============================================================================

@router.post(
    "/reports/generate",
    response_model=APIResponse,
    status_code=201,
    summary="Generate quarterly report",
    description=(
        "Trigger quarterly report generation for a given period and importer. "
        "Assembles shipment data, calculates aggregated emissions, generates "
        "XML and Markdown outputs, and computes provenance hash."
    ),
)
async def generate_report(
    request: ReportGenerateRequest = Body(...),
) -> APIResponse:
    """Generate a quarterly CBAM report from shipment data."""
    start_time = time.time()
    logger.info(
        "Report generation requested: importer=%s, period=%d%s, shipments=%d",
        request.importer_id,
        request.year,
        request.quarter.value,
        len(request.shipments),
    )

    try:
        scheduler = _get_scheduler()
        assembler = _get_assembler()

        # Build the period
        period = scheduler.get_quarter(request.year, request.quarter)

        # Check for existing draft (unless force_regenerate)
        if not request.force_regenerate:
            for rid in _importer_reports.get(request.importer_id, []):
                existing = _reports_store.get(rid)
                if (
                    existing is not None
                    and existing.period.period_label == period.period_label
                    and existing.status == ReportStatus.DRAFT
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Draft report already exists for {period.period_label}: "
                            f"{existing.report_id}. Use force_regenerate=true to overwrite."
                        ),
                    )

        # Assemble the report
        report = assembler.assemble_quarterly_report(
            period=period,
            importer_id=request.importer_id,
            shipments=request.shipments,
        )

        # Store the report
        _store_report(report)

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Report generated: id=%s, emissions=%s tCO2e, in %.1fms",
            report.report_id,
            report.total_embedded_emissions,
            processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Report generated: {report.report_id}",
            data=report.model_dump(mode="json"),
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/reports",
    response_model=APIResponse,
    summary="List quarterly reports",
    description=(
        "List quarterly reports with optional filtering by period, status, "
        "and importer. Supports pagination via offset and limit."
    ),
)
async def list_reports(
    period: Optional[str] = Query(
        None,
        description="Filter by period label (e.g., '2026Q1')"
    ),
    status: Optional[ReportStatus] = Query(
        None,
        description="Filter by report status"
    ),
    importer_id: Optional[str] = Query(
        None,
        description="Filter by importer EORI"
    ),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
) -> APIResponse:
    """List quarterly reports with optional filters."""
    start_time = time.time()
    logger.info(
        "Listing reports: period=%s, status=%s, importer=%s",
        period, status, importer_id,
    )

    try:
        # Apply filters
        filtered: List[QuarterlyReport] = []
        for report in _reports_store.values():
            if period and report.period.period_label != period:
                continue
            if status and report.status != status:
                continue
            if importer_id and report.importer_id != importer_id:
                continue
            filtered.append(report)

        # Sort by created_at descending
        filtered.sort(key=lambda r: r.created_at, reverse=True)

        total = len(filtered)
        page = filtered[offset: offset + limit]
        items = [_report_to_list_item(r).model_dump(mode="json") for r in page]

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Listed %d/%d reports in %.1fms", len(items), total, processing_ms
        )

        return APIResponse(
            success=True,
            message=f"Found {total} report(s)",
            data={
                "items": items,
                "total": total,
                "offset": offset,
                "limit": limit,
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except Exception as exc:
        logger.error("Failed to list reports: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/reports/{report_id}",
    response_model=APIResponse,
    summary="Get report details",
    description="Retrieve full details of a quarterly report by ID.",
)
async def get_report_details(
    report_id: str = Path(..., description="Report identifier"),
) -> APIResponse:
    """Get full details of a quarterly report."""
    start_time = time.time()
    logger.info("Fetching report details: id=%s", report_id)

    try:
        report = _get_report(report_id)

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Report retrieved: id=%s, status=%s, in %.1fms",
            report_id, report.status.value, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Report {report_id}",
            data=report.model_dump(mode="json"),
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get report %s: %s", report_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/reports/{report_id}/xml",
    summary="Download report XML",
    description=(
        "Download the XML output of a quarterly report in EU CBAM Registry format."
    ),
)
async def download_report_xml(
    report_id: str = Path(..., description="Report identifier"),
) -> Response:
    """Download the XML output of a quarterly report."""
    logger.info("Downloading XML for report: id=%s", report_id)

    try:
        report = _get_report(report_id)

        if not report.report_xml:
            raise HTTPException(
                status_code=404,
                detail=f"No XML output available for report {report_id}. "
                       f"Report may not have been fully generated.",
            )

        filename = f"CBAM_{report.period.period_label}_{report.importer_id}.xml"
        return Response(
            content=report.report_xml,
            media_type="application/xml",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Report-Id": report_id,
                "X-Provenance-Hash": report.provenance_hash,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to download XML for %s: %s", report_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/reports/{report_id}/summary",
    summary="Download report summary",
    description="Download the human-readable Markdown summary of a quarterly report.",
)
async def download_report_summary(
    report_id: str = Path(..., description="Report identifier"),
) -> Response:
    """Download the Markdown summary of a quarterly report."""
    logger.info("Downloading summary for report: id=%s", report_id)

    try:
        report = _get_report(report_id)

        if not report.report_summary_md:
            raise HTTPException(
                status_code=404,
                detail=f"No summary available for report {report_id}. "
                       f"Report may not have been fully generated.",
            )

        filename = f"CBAM_{report.period.period_label}_{report.importer_id}_summary.md"
        return Response(
            content=report.report_summary_md,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Report-Id": report_id,
                "X-Provenance-Hash": report.provenance_hash,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to download summary for %s: %s", report_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# REPORT SUBMISSION ENDPOINTS
# ============================================================================

@router.put(
    "/reports/{report_id}/submit",
    response_model=APIResponse,
    summary="Submit report for review",
    description=(
        "Submit a quarterly report for review. The report must be in draft "
        "or amended status. The submitter must confirm the data declaration."
    ),
)
async def submit_report(
    report_id: str = Path(..., description="Report identifier"),
    request: ReportSubmitRequest = Body(...),
) -> APIResponse:
    """Submit a quarterly report for review."""
    start_time = time.time()
    logger.info(
        "Report submission requested: id=%s, by=%s",
        report_id, request.submitted_by,
    )

    try:
        report = _get_report(report_id)

        # Validate declaration
        if not request.declaration:
            raise HTTPException(
                status_code=400,
                detail="Data accuracy declaration must be confirmed (declaration=true)",
            )

        # Validate status transition: draft -> in_review -> submitted
        # For simplicity, allow draft -> submitted with declaration
        if report.status not in {ReportStatus.DRAFT, ReportStatus.AMENDED, ReportStatus.IN_REVIEW}:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Report cannot be submitted from status '{report.status.value}'. "
                    f"Only draft, amended, or in_review reports can be submitted."
                ),
            )

        # Validate report has content
        if report.shipments_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot submit a report with zero shipments.",
            )

        # Transition to submitted
        now = datetime.now(timezone.utc)
        updated_data = report.model_dump()
        updated_data["status"] = ReportStatus.SUBMITTED
        updated_data["submitted_at"] = now

        # Recompute provenance hash with submission timestamp
        updated_report = QuarterlyReport(**updated_data)
        updated_data["provenance_hash"] = updated_report.compute_provenance_hash()
        updated_report = QuarterlyReport(**updated_data)

        _reports_store[report_id] = updated_report

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Report submitted: id=%s, by=%s, in %.1fms",
            report_id, request.submitted_by, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Report {report_id} submitted successfully",
            data={
                "report_id": report_id,
                "status": ReportStatus.SUBMITTED.value,
                "submitted_at": now.isoformat(),
                "submitted_by": request.submitted_by,
                "provenance_hash": updated_report.provenance_hash,
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to submit report %s: %s", report_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# AMENDMENT ENDPOINTS
# ============================================================================

@router.post(
    "/reports/{report_id}/amend",
    response_model=APIResponse,
    status_code=201,
    summary="Create amendment",
    description=(
        "Create an amendment to a submitted or accepted quarterly report. "
        "The amendment must be within the 60-day amendment window."
    ),
)
async def create_amendment(
    report_id: str = Path(..., description="Report identifier"),
    request: AmendmentRequest = Body(...),
) -> APIResponse:
    """Create an amendment to a quarterly report."""
    start_time = time.time()
    logger.info(
        "Amendment requested: report=%s, reason=%s, by=%s",
        report_id, request.reason.value, request.amended_by,
    )

    try:
        report = _get_report(report_id)
        amendment_mgr = _get_amendment_mgr()

        # Validate report status allows amendment
        if not report.status.allows_amendment:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Report in status '{report.status.value}' cannot be amended. "
                    f"Only submitted, accepted, or rejected reports can be amended."
                ),
            )

        # Validate within amendment window
        today = date.today()
        if today > report.period.amendment_deadline:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Amendment window has closed. Deadline was "
                    f"{report.period.amendment_deadline.isoformat()}. "
                    f"Today is {today.isoformat()}."
                ),
            )

        # Register the report version if not already tracked
        amendment_mgr.register_report(report)

        # Create amendment through the manager
        amendment = amendment_mgr.create_amendment(
            report_id=report_id,
            changes=request.changes,
            reason=request.reason,
            changes_summary=request.changes_summary,
            amended_by=request.amended_by,
        )

        # Apply the amendment to get updated report
        updated_report = amendment_mgr.apply_amendment(amendment.amendment_id)

        # Store the updated report
        _reports_store[report_id] = updated_report

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Amendment created: id=%s, report=%s, version=%d, in %.1fms",
            amendment.amendment_id,
            report_id,
            amendment.version,
            processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Amendment {amendment.amendment_id} created for report {report_id}",
            data={
                "amendment": amendment.model_dump(mode="json"),
                "report_version": updated_report.version,
                "report_status": updated_report.status.value,
                "new_provenance_hash": updated_report.provenance_hash,
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Failed to create amendment for %s: %s", report_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/reports/{report_id}/amendments",
    response_model=APIResponse,
    summary="List amendments",
    description="List all amendments for a quarterly report.",
)
async def list_amendments(
    report_id: str = Path(..., description="Report identifier"),
) -> APIResponse:
    """List all amendments for a quarterly report."""
    start_time = time.time()
    logger.info("Listing amendments for report: %s", report_id)

    try:
        # Verify report exists
        _ = _get_report(report_id)
        amendment_mgr = _get_amendment_mgr()

        amendments = amendment_mgr.get_amendments(report_id)
        items = [a.model_dump(mode="json") for a in amendments]

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Listed %d amendments for report %s in %.1fms",
            len(items), report_id, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"{len(items)} amendment(s) for report {report_id}",
            data={
                "report_id": report_id,
                "amendments": items,
                "total": len(items),
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to list amendments for %s: %s", report_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/reports/{report_id}/amendments/{amendment_id}/diff",
    response_model=APIResponse,
    summary="Get amendment diff",
    description="Get the structured diff showing exactly what changed in an amendment.",
)
async def get_amendment_diff(
    report_id: str = Path(..., description="Report identifier"),
    amendment_id: str = Path(..., description="Amendment identifier"),
) -> APIResponse:
    """Get the structured diff for a specific amendment."""
    start_time = time.time()
    logger.info("Fetching diff: report=%s, amendment=%s", report_id, amendment_id)

    try:
        _ = _get_report(report_id)
        amendment_mgr = _get_amendment_mgr()

        amendment = amendment_mgr.get_amendment(amendment_id)
        if amendment is None:
            raise HTTPException(
                status_code=404,
                detail=f"Amendment not found: {amendment_id}",
            )

        if amendment.report_id != report_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Amendment {amendment_id} does not belong to report {report_id}"
                ),
            )

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Diff retrieved: amendment=%s, in %.1fms",
            amendment_id, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Diff for amendment {amendment_id}",
            data={
                "amendment_id": amendment_id,
                "report_id": report_id,
                "version": amendment.version,
                "reason": amendment.reason.value,
                "reason_description": amendment.reason.description,
                "changes_summary": amendment.changes_summary,
                "diff_data": amendment.diff_data,
                "previous_hash": amendment.previous_hash,
                "new_hash": amendment.new_hash,
                "amended_by": amendment.amended_by,
                "amended_at": amendment.amended_at.isoformat(),
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get diff for amendment %s: %s",
            amendment_id, exc, exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# DEADLINE ENDPOINTS
# ============================================================================

@router.get(
    "/deadlines",
    response_model=APIResponse,
    summary="Get upcoming deadlines",
    description=(
        "Get upcoming CBAM report submission deadlines for an importer. "
        "Returns alerts at various threshold levels."
    ),
)
async def get_upcoming_deadlines(
    importer_id: str = Query(..., description="Importer EORI or identifier"),
) -> APIResponse:
    """Get upcoming deadlines for an importer."""
    start_time = time.time()
    logger.info("Checking deadlines for importer: %s", importer_id)

    try:
        tracker = _get_deadline_tracker()
        alerts = tracker.check_upcoming_deadlines(importer_id)

        # Store alerts for later acknowledgement
        for alert in alerts:
            _alerts_store[alert.alert_id] = alert

        items = [a.model_dump(mode="json") for a in alerts]

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Found %d deadline alerts for importer %s in %.1fms",
            len(items), importer_id, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"{len(items)} deadline alert(s) for {importer_id}",
            data={
                "importer_id": importer_id,
                "alerts": items,
                "total": len(items),
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except Exception as exc:
        logger.error(
            "Failed to check deadlines for %s: %s", importer_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/deadlines/overdue",
    response_model=APIResponse,
    summary="Get overdue reports",
    description="Get reports that are past their submission deadline.",
)
async def get_overdue_deadlines(
    importer_id: str = Query(..., description="Importer EORI or identifier"),
) -> APIResponse:
    """Get overdue report deadlines for an importer."""
    start_time = time.time()
    logger.info("Checking overdue deadlines for importer: %s", importer_id)

    try:
        tracker = _get_deadline_tracker()
        overdue_alerts = tracker.check_overdue_reports(importer_id)

        for alert in overdue_alerts:
            _alerts_store[alert.alert_id] = alert

        items = [a.model_dump(mode="json") for a in overdue_alerts]

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Found %d overdue alerts for importer %s in %.1fms",
            len(items), importer_id, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"{len(items)} overdue alert(s) for {importer_id}",
            data={
                "importer_id": importer_id,
                "overdue_alerts": items,
                "total": len(items),
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except Exception as exc:
        logger.error(
            "Failed to check overdue for %s: %s", importer_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.put(
    "/deadlines/{alert_id}/acknowledge",
    response_model=APIResponse,
    summary="Acknowledge deadline alert",
    description="Acknowledge a deadline alert to mark it as reviewed.",
)
async def acknowledge_deadline(
    alert_id: str = Path(..., description="Alert identifier"),
    request: DeadlineAcknowledgeRequest = Body(...),
) -> APIResponse:
    """Acknowledge a deadline alert."""
    start_time = time.time()
    logger.info(
        "Acknowledging alert: id=%s, by=%s", alert_id, request.acknowledged_by
    )

    try:
        alert = _alerts_store.get(alert_id)
        if alert is None:
            raise HTTPException(
                status_code=404,
                detail=f"Alert not found: {alert_id}",
            )

        if alert.acknowledged:
            raise HTTPException(
                status_code=409,
                detail=f"Alert {alert_id} has already been acknowledged.",
            )

        # Update the alert
        updated_data = alert.model_dump()
        updated_data["acknowledged"] = True
        updated_alert = DeadlineAlert(**updated_data)
        _alerts_store[alert_id] = updated_alert

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Alert acknowledged: id=%s, by=%s, in %.1fms",
            alert_id, request.acknowledged_by, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Alert {alert_id} acknowledged",
            data={
                "alert_id": alert_id,
                "acknowledged": True,
                "acknowledged_by": request.acknowledged_by,
                "notes": request.notes,
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to acknowledge alert %s: %s", alert_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# NOTIFICATION ENDPOINTS
# ============================================================================

@router.get(
    "/notifications",
    response_model=APIResponse,
    summary="Get notification history",
    description="Retrieve notification delivery history for an importer.",
)
async def get_notification_history(
    importer_id: str = Query(..., description="Importer EORI or identifier"),
    notification_type: Optional[NotificationType] = Query(
        None,
        description="Filter by notification type"
    ),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
) -> APIResponse:
    """Get notification history for an importer."""
    start_time = time.time()
    logger.info(
        "Fetching notification history: importer=%s, type=%s",
        importer_id, notification_type,
    )

    try:
        notification_svc = _get_notification_svc()
        log_entries = notification_svc.get_notification_log(importer_id)

        # Apply type filter
        if notification_type is not None:
            log_entries = [
                e for e in log_entries
                if e.notification_type == notification_type
            ]

        total = len(log_entries)
        page = log_entries[offset: offset + limit]
        items = [e.model_dump(mode="json") for e in page]

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Retrieved %d/%d notification log entries in %.1fms",
            len(items), total, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"{total} notification(s) for {importer_id}",
            data={
                "importer_id": importer_id,
                "notifications": items,
                "total": total,
                "offset": offset,
                "limit": limit,
            },
            processing_time_ms=round(processing_ms, 2),
        )

    except Exception as exc:
        logger.error(
            "Failed to get notifications for %s: %s", importer_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.put(
    "/notifications/configure",
    response_model=APIResponse,
    summary="Configure notification recipients",
    description=(
        "Configure notification routing for an importer, including email "
        "recipients, webhook endpoints, alert levels, and quiet hours."
    ),
)
async def configure_notifications(
    request: NotificationConfigureRequest = Body(...),
) -> APIResponse:
    """Configure notification recipients and routing for an importer."""
    start_time = time.time()
    logger.info(
        "Configuring notifications for importer: %s, emails=%d, webhooks=%d",
        request.importer_id,
        len(request.email_recipients),
        len(request.webhook_urls),
    )

    try:
        # Validate at least one recipient channel
        if not request.email_recipients and not request.webhook_urls:
            raise HTTPException(
                status_code=400,
                detail="At least one email recipient or webhook URL must be configured.",
            )

        # Build the notification config
        config = NotificationConfig(
            importer_id=request.importer_id,
            email_recipients=request.email_recipients,
            webhook_urls=request.webhook_urls,
            alert_levels_enabled=request.alert_levels_enabled,
            notification_types_enabled=request.notification_types_enabled,
            quiet_hours_start=request.quiet_hours_start,
            quiet_hours_end=request.quiet_hours_end,
        )

        # Store the configuration
        _notification_configs[request.importer_id] = config

        # Register with the notification service
        notification_svc = _get_notification_svc()
        notification_svc.register_config(config)

        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            "Notifications configured for importer %s in %.1fms",
            request.importer_id, processing_ms,
        )

        return APIResponse(
            success=True,
            message=f"Notification configuration saved for {request.importer_id}",
            data=config.model_dump(mode="json"),
            processing_time_ms=round(processing_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to configure notifications for %s: %s",
            request.importer_id, exc, exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc))
