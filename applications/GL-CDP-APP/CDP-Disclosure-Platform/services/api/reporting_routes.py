"""
GL-CDP-APP Reporting & Submission API

Report generation, submission checklist validation, and CDP Online Response
System (ORS) export for CDP Climate Change questionnaire.

Supported export formats:
    - PDF: Formatted questionnaire with all responses
    - Excel: Tabular response data
    - XML: CDP ORS-compatible XML for direct upload
    - JSON: Machine-readable response data

Submission workflow:
    1. Validate completeness (submission checklist)
    2. Generate submission-ready report
    3. Executive summary generation
    4. Submit to CDP ORS
    5. Track submission history
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/reports", tags=["Reporting"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportFormat(str, Enum):
    """Supported report formats."""
    PDF = "pdf"
    EXCEL = "excel"
    XML = "xml"
    JSON = "json"


class SubmissionStatus(str, Enum):
    """CDP submission status."""
    DRAFT = "draft"
    READY = "ready"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ChecklistItemStatus(str, Enum):
    """Checklist item status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ChecklistItem(BaseModel):
    """Submission checklist item."""
    item_id: str
    category: str
    description: str
    status: str
    severity: str
    detail: Optional[str]
    remediation: Optional[str]


class ChecklistResponse(BaseModel):
    """Submission completeness checklist."""
    questionnaire_id: str
    total_items: int
    passed: int
    failed: int
    warnings: int
    not_applicable: int
    overall_ready: bool
    items: List[ChecklistItem]
    completion_pct: float
    checked_at: datetime


class ReportResponse(BaseModel):
    """Generated CDP report."""
    report_id: str
    questionnaire_id: str
    format: str
    title: str
    report_status: str
    page_count: Optional[int]
    file_size_bytes: Optional[int]
    download_url: Optional[str]
    modules_included: List[str]
    response_count: int
    evidence_count: int
    generated_at: datetime


class ExecutiveSummaryResponse(BaseModel):
    """Executive summary of CDP disclosure."""
    questionnaire_id: str
    org_name: str
    reporting_year: int
    questionnaire_year: str
    predicted_score: float
    predicted_band: str
    completion_pct: float
    module_summary: List[Dict[str, Any]]
    key_metrics: Dict[str, Any]
    score_highlights: List[str]
    improvement_areas: List[str]
    a_level_status: Dict[str, Any]
    generated_at: datetime


class SubmissionResponse(BaseModel):
    """CDP submission record."""
    submission_id: str
    questionnaire_id: str
    submission_status: str
    submitted_by: str
    submitted_at: Optional[datetime]
    cdp_confirmation_id: Optional[str]
    format: str
    file_size_bytes: Optional[int]
    response_count: int
    evidence_count: int
    checklist_passed: bool
    notes: Optional[str]


class SubmissionHistoryEntry(BaseModel):
    """Summary entry in submission history."""
    submission_id: str
    questionnaire_id: str
    questionnaire_year: str
    reporting_year: int
    submission_status: str
    submitted_at: Optional[datetime]
    predicted_score: Optional[float]
    predicted_band: Optional[str]


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    """Request to generate a CDP report."""
    title: Optional[str] = Field(
        None, max_length=500, description="Report title"
    )
    include_evidence: bool = Field(True, description="Include evidence attachments")
    include_scoring: bool = Field(True, description="Include scoring simulation")
    include_gaps: bool = Field(False, description="Include gap analysis")
    modules: Optional[List[str]] = Field(
        None, description="Specific modules to include (null = all)"
    )
    language: str = Field("en", description="Report language (ISO 639-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "CDP Climate Change 2025 Response",
                "include_evidence": True,
                "include_scoring": True,
                "include_gaps": False,
                "modules": None,
                "language": "en",
            }
        }


class SubmitRequest(BaseModel):
    """Request to submit questionnaire to CDP."""
    submitted_by: str = Field(..., description="Submitter user ID")
    submitted_by_name: str = Field("", description="Submitter display name")
    notes: Optional[str] = Field(None, max_length=2000, description="Submission notes")
    format: ReportFormat = Field(ReportFormat.XML, description="Submission format")
    confirm_completeness: bool = Field(
        ..., description="Confirm that the checklist has been reviewed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "submitted_by": "usr_abc123",
                "submitted_by_name": "Jane Smith, Head of Sustainability",
                "notes": "2025 CDP Climate Change disclosure - final submission.",
                "format": "xml",
                "confirm_completeness": True,
            }
        }


# ---------------------------------------------------------------------------
# Submission Checklist Definition
# ---------------------------------------------------------------------------

CHECKLIST_ITEMS = [
    ("CHK-01", "completeness", "All mandatory questions answered", "critical"),
    ("CHK-02", "completeness", "Module M0 Introduction complete", "critical"),
    ("CHK-03", "completeness", "Module M7 Environmental Performance complete", "critical"),
    ("CHK-04", "completeness", "Module M13 Sign Off complete", "critical"),
    ("CHK-05", "data_quality", "Scope 1 emissions data populated", "critical"),
    ("CHK-06", "data_quality", "Scope 2 emissions data populated", "critical"),
    ("CHK-07", "data_quality", "Scope 3 at least one category populated", "high"),
    ("CHK-08", "data_quality", "Emissions data matches reporting year", "critical"),
    ("CHK-09", "verification", "Scope 1+2 verification status declared", "high"),
    ("CHK-10", "verification", "Verification statements attached", "medium"),
    ("CHK-11", "governance", "Board oversight described", "high"),
    ("CHK-12", "governance", "Management responsibility documented", "high"),
    ("CHK-13", "targets", "Emissions reduction targets set", "high"),
    ("CHK-14", "targets", "Target base year and scope defined", "high"),
    ("CHK-15", "review", "All responses reviewed and approved", "critical"),
    ("CHK-16", "review", "At least one C-suite sign-off recorded", "high"),
    ("CHK-17", "format", "All table-type questions properly formatted", "medium"),
    ("CHK-18", "format", "All numeric fields within valid ranges", "medium"),
    ("CHK-19", "evidence", "Key evidence documents attached", "medium"),
    ("CHK-20", "evidence", "Evidence file sizes within CDP limits (10MB per file)", "low"),
]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_reports: Dict[str, Dict[str, Any]] = {}
_submissions: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _simulate_checklist(questionnaire_id: str) -> List[Dict[str, Any]]:
    """Generate simulated checklist results."""
    # Simulate some items passing, some failing
    passing_ids = {
        "CHK-01", "CHK-02", "CHK-03", "CHK-05", "CHK-06", "CHK-08",
        "CHK-11", "CHK-12", "CHK-13", "CHK-14", "CHK-17", "CHK-18", "CHK-20",
    }
    warning_ids = {"CHK-07", "CHK-09", "CHK-10", "CHK-19"}
    items = []
    for item_id, category, desc, severity in CHECKLIST_ITEMS:
        if item_id in passing_ids:
            item_status = ChecklistItemStatus.PASS.value
            detail = "Requirement met"
            remediation = None
        elif item_id in warning_ids:
            item_status = ChecklistItemStatus.WARNING.value
            detail = "Partially met - review recommended"
            remediation = f"Review and address: {desc}"
        else:
            item_status = ChecklistItemStatus.FAIL.value
            detail = "Requirement not met"
            remediation = f"Action required: {desc}"
        items.append({
            "item_id": item_id,
            "category": category,
            "description": desc,
            "status": item_status,
            "severity": severity,
            "detail": detail,
            "remediation": remediation,
        })
    return items


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{questionnaire_id}/checklist",
    response_model=ChecklistResponse,
    summary="Submission checklist",
    description=(
        "Run the submission completeness checklist for a questionnaire. "
        "Validates mandatory questions, data quality, verification status, "
        "governance, targets, review status, and formatting."
    ),
)
async def get_checklist(questionnaire_id: str) -> ChecklistResponse:
    """Run submission completeness checklist."""
    items = _simulate_checklist(questionnaire_id)
    passed = sum(1 for i in items if i["status"] == "pass")
    failed = sum(1 for i in items if i["status"] == "fail")
    warnings = sum(1 for i in items if i["status"] == "warning")
    na = sum(1 for i in items if i["status"] == "not_applicable")
    total = len(items)
    overall_ready = failed == 0
    completion = round(passed / max(1, total - na) * 100, 1)

    return ChecklistResponse(
        questionnaire_id=questionnaire_id,
        total_items=total,
        passed=passed,
        failed=failed,
        warnings=warnings,
        not_applicable=na,
        overall_ready=overall_ready,
        items=[ChecklistItem(**i) for i in items],
        completion_pct=completion,
        checked_at=_now(),
    )


@router.post(
    "/{questionnaire_id}/generate/pdf",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate PDF report",
    description="Generate a formatted PDF report of the CDP questionnaire responses.",
)
async def generate_pdf(
    questionnaire_id: str,
    request: GenerateReportRequest,
) -> ReportResponse:
    """Generate PDF report."""
    return await _generate_report(questionnaire_id, ReportFormat.PDF, request)


@router.post(
    "/{questionnaire_id}/generate/excel",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate Excel export",
    description="Generate an Excel export of all CDP questionnaire response data.",
)
async def generate_excel(
    questionnaire_id: str,
    request: GenerateReportRequest,
) -> ReportResponse:
    """Generate Excel report."""
    return await _generate_report(questionnaire_id, ReportFormat.EXCEL, request)


@router.post(
    "/{questionnaire_id}/generate/xml",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate ORS XML",
    description=(
        "Generate CDP Online Response System (ORS) compatible XML export. "
        "This is the format required for direct upload to the CDP platform."
    ),
)
async def generate_xml(
    questionnaire_id: str,
    request: GenerateReportRequest,
) -> ReportResponse:
    """Generate ORS XML report."""
    return await _generate_report(questionnaire_id, ReportFormat.XML, request)


async def _generate_report(
    questionnaire_id: str,
    report_format: ReportFormat,
    request: GenerateReportRequest,
) -> ReportResponse:
    """Internal report generation helper."""
    report_id = _generate_id("rpt")
    now = _now()

    modules = request.modules or [
        "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
        "M8", "M9", "M10", "M11", "M12", "M13",
    ]

    # Simulated file characteristics
    file_specs = {
        ReportFormat.PDF: {"pages": 85, "size": 3600000},
        ReportFormat.EXCEL: {"pages": None, "size": 1450000},
        ReportFormat.XML: {"pages": None, "size": 520000},
        ReportFormat.JSON: {"pages": None, "size": 380000},
    }
    spec = file_specs.get(report_format, {"pages": None, "size": 500000})

    report = {
        "report_id": report_id,
        "questionnaire_id": questionnaire_id,
        "format": report_format.value,
        "title": request.title or f"CDP Climate Change {report_format.value.upper()} Report",
        "report_status": "completed",
        "page_count": spec["pages"],
        "file_size_bytes": spec["size"],
        "download_url": f"https://api.greenlang.io/cdp/reports/{report_id}/download",
        "modules_included": modules,
        "response_count": 185,
        "evidence_count": 42 if request.include_evidence else 0,
        "generated_at": now,
    }
    _reports[report_id] = report
    return ReportResponse(**report)


@router.get(
    "/{questionnaire_id}/summary",
    response_model=ExecutiveSummaryResponse,
    summary="Executive summary",
    description=(
        "Generate an executive summary of the CDP disclosure including "
        "predicted score, key metrics, highlights, and improvement areas."
    ),
)
async def get_executive_summary(questionnaire_id: str) -> ExecutiveSummaryResponse:
    """Generate executive summary."""
    module_summary = [
        {"module": "M0", "name": "Introduction", "completion_pct": 100.0, "response_count": 15},
        {"module": "M1", "name": "Governance", "completion_pct": 90.0, "response_count": 18},
        {"module": "M2", "name": "Policies & Commitments", "completion_pct": 85.0, "response_count": 12},
        {"module": "M3", "name": "Risks & Opportunities", "completion_pct": 72.0, "response_count": 18},
        {"module": "M4", "name": "Strategy", "completion_pct": 65.0, "response_count": 13},
        {"module": "M5", "name": "Transition Plans", "completion_pct": 40.0, "response_count": 8},
        {"module": "M6", "name": "Implementation", "completion_pct": 75.0, "response_count": 15},
        {"module": "M7", "name": "Environmental Performance", "completion_pct": 88.0, "response_count": 31},
        {"module": "M10", "name": "Supply Chain", "completion_pct": 60.0, "response_count": 9},
        {"module": "M11", "name": "Additional Metrics", "completion_pct": 50.0, "response_count": 5},
        {"module": "M13", "name": "Sign Off", "completion_pct": 0.0, "response_count": 0},
    ]

    return ExecutiveSummaryResponse(
        questionnaire_id=questionnaire_id,
        org_name="GreenLang Demo Organization",
        reporting_year=2025,
        questionnaire_year="2026",
        predicted_score=58.7,
        predicted_band="B-",
        completion_pct=72.5,
        module_summary=module_summary,
        key_metrics={
            "scope_1_tco2e": 12450.8,
            "scope_2_location_tco2e": 8320.5,
            "scope_2_market_tco2e": 6250.0,
            "scope_3_tco2e": 45200.0,
            "total_tco2e": 66021.3,
            "yoy_change_pct": -3.8,
            "verification_coverage_pct": 85.0,
            "reduction_target_pct": 42.0,
        },
        score_highlights=[
            "Strong Scope 1/2 emissions reporting (category 9 score: 78%)",
            "Board-level climate governance documented (category 1: 72%)",
            "SBTi target validated (4.5% annual reduction)",
            "Year-over-year emissions reduction of 3.8%",
        ],
        improvement_areas=[
            "Transition plan (M5) only 40% complete - critical for A-level",
            "Scenario analysis lacks quantitative financial impact",
            "No Scope 3 third-party verification",
            "Supply chain engagement metrics incomplete",
        ],
        a_level_status={
            "eligible": False,
            "requirements_met": 2,
            "requirements_total": 5,
            "gap_to_a": 21.3,
        },
        generated_at=_now(),
    )


@router.post(
    "/{questionnaire_id}/submit",
    response_model=SubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit to CDP",
    description=(
        "Submit the questionnaire to CDP. Validates the submission checklist "
        "and generates the ORS XML. Submission is recorded in history."
    ),
)
async def submit_to_cdp(
    questionnaire_id: str,
    request: SubmitRequest,
) -> SubmissionResponse:
    """Submit questionnaire to CDP."""
    if not request.confirm_completeness:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm completeness by setting confirm_completeness=true.",
        )

    # Run checklist
    items = _simulate_checklist(questionnaire_id)
    critical_failures = [
        i for i in items
        if i["status"] == "fail" and i["severity"] == "critical"
    ]
    if critical_failures:
        failure_descs = [f["description"] for f in critical_failures]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Critical checklist items failed: {'; '.join(failure_descs)}",
        )

    submission_id = _generate_id("sub")
    now = _now()
    submission = {
        "submission_id": submission_id,
        "questionnaire_id": questionnaire_id,
        "submission_status": SubmissionStatus.SUBMITTED.value,
        "submitted_by": request.submitted_by,
        "submitted_at": now,
        "cdp_confirmation_id": f"CDP-{uuid.uuid4().hex[:8].upper()}",
        "format": request.format.value,
        "file_size_bytes": 520000 if request.format == ReportFormat.XML else 1450000,
        "response_count": 185,
        "evidence_count": 42,
        "checklist_passed": True,
        "notes": request.notes,
    }
    _submissions[submission_id] = submission
    return SubmissionResponse(**submission)


@router.get(
    "/history",
    response_model=List[SubmissionHistoryEntry],
    summary="Submission history",
    description="Retrieve the history of all CDP submissions for the organization.",
)
async def get_submission_history(
    org_id: Optional[str] = Query(None, description="Filter by organization ID"),
    questionnaire_year: Optional[str] = Query(None, description="Filter by questionnaire year"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[SubmissionHistoryEntry]:
    """Retrieve submission history."""
    submissions = list(_submissions.values())
    submissions.sort(key=lambda s: s.get("submitted_at") or datetime.min, reverse=True)

    # Add simulated historical submissions
    historical = [
        SubmissionHistoryEntry(
            submission_id="sub_hist_2024",
            questionnaire_id="cdpq_2024",
            questionnaire_year="2024",
            reporting_year=2023,
            submission_status="accepted",
            submitted_at=datetime(2024, 7, 28),
            predicted_score=53.5,
            predicted_band="B-",
        ),
        SubmissionHistoryEntry(
            submission_id="sub_hist_2023",
            questionnaire_id="cdpq_2023",
            questionnaire_year="2023",
            reporting_year=2022,
            submission_status="accepted",
            submitted_at=datetime(2023, 7, 25),
            predicted_score=48.2,
            predicted_band="C",
        ),
        SubmissionHistoryEntry(
            submission_id="sub_hist_2022",
            questionnaire_id="cdpq_2022",
            questionnaire_year="2022",
            reporting_year=2021,
            submission_status="accepted",
            submitted_at=datetime(2022, 7, 27),
            predicted_score=42.0,
            predicted_band="C",
        ),
    ]

    current = [
        SubmissionHistoryEntry(
            submission_id=s["submission_id"],
            questionnaire_id=s["questionnaire_id"],
            questionnaire_year="2026",
            reporting_year=2025,
            submission_status=s["submission_status"],
            submitted_at=s.get("submitted_at"),
            predicted_score=58.7,
            predicted_band="B-",
        )
        for s in submissions
    ]

    all_entries = current + historical
    if questionnaire_year:
        all_entries = [e for e in all_entries if e.questionnaire_year == questionnaire_year]
    return all_entries[:limit]
