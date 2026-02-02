"""
GreenLang Compliance REST API Routes

This module provides REST API endpoints for compliance report management,
including listing reports, generating new reports, and retrieving details.

Endpoints:
    GET  /api/v1/compliance/reports      - List compliance reports
    POST /api/v1/compliance/reports      - Generate new report
    GET  /api/v1/compliance/reports/{id} - Get report details

Features:
    - Multiple compliance frameworks (GHG Protocol, CDP, TCFD, etc.)
    - Report generation with configurable parameters
    - Export in multiple formats (PDF, Excel, JSON)
    - Audit trail and versioning
    - Regulatory deadline tracking

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.api.routes.compliance_routes import compliance_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(compliance_router, prefix="/api/v1")
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Depends = None
    HTTPException = Exception
    Query = None
    Request = None
    status = None
    JSONResponse = None

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GHG_PROTOCOL = "ghg_protocol"
    CDP = "cdp"
    TCFD = "tcfd"
    SBTi = "sbti"
    SEC_CLIMATE = "sec_climate"
    CSRD = "csrd"
    ISO_14064 = "iso_14064"
    EPA_MRR = "epa_mrr"
    EU_ETS = "eu_ets"
    UK_SECR = "uk_secr"


class ReportStatus(str, Enum):
    """Report generation status."""
    DRAFT = "draft"
    GENERATING = "generating"
    REVIEW = "review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ReportType(str, Enum):
    """Types of compliance reports."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    AD_HOC = "ad_hoc"
    VERIFICATION = "verification"
    DISCLOSURE = "disclosure"


class ReportFormat(str, Enum):
    """Report export formats."""
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"
    CSV = "csv"


class ScopeInclusion(str, Enum):
    """GHG scope inclusion options."""
    SCOPE_1_ONLY = "scope_1_only"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"
    ALL_SCOPES = "all_scopes"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class ReportGenerationRequest(BaseModel):
    """
    Request model for compliance report generation.

    Attributes:
        framework: Compliance framework for the report
        report_type: Type of report (annual, quarterly, etc.)
        reporting_period_start: Start of reporting period
        reporting_period_end: End of reporting period
        scope_inclusion: Which GHG scopes to include
        organizational_boundary: Organizational boundary definition
        facilities: List of facility IDs to include (empty = all)
        export_formats: Desired export formats
        include_verification: Include verification statement
        include_methodology: Include methodology appendix
        custom_parameters: Framework-specific parameters
    """
    framework: ComplianceFramework = Field(
        ...,
        description="Compliance framework for the report"
    )
    report_type: ReportType = Field(
        default=ReportType.ANNUAL,
        description="Type of report"
    )
    reporting_period_start: datetime = Field(
        ...,
        description="Start of reporting period"
    )
    reporting_period_end: datetime = Field(
        ...,
        description="End of reporting period"
    )
    scope_inclusion: ScopeInclusion = Field(
        default=ScopeInclusion.SCOPE_1_2_3,
        description="Which GHG scopes to include"
    )
    organizational_boundary: str = Field(
        default="operational_control",
        description="Organizational boundary (operational_control, equity_share, financial_control)"
    )
    facilities: List[str] = Field(
        default_factory=list,
        description="Facility IDs to include (empty = all)"
    )
    export_formats: List[ReportFormat] = Field(
        default=[ReportFormat.PDF, ReportFormat.EXCEL],
        description="Desired export formats"
    )
    include_verification: bool = Field(
        default=False,
        description="Include verification statement"
    )
    include_methodology: bool = Field(
        default=True,
        description="Include methodology appendix"
    )
    custom_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Framework-specific parameters"
    )

    class Config:
        schema_extra = {
            "example": {
                "framework": "ghg_protocol",
                "report_type": "annual",
                "reporting_period_start": "2025-01-01T00:00:00Z",
                "reporting_period_end": "2025-12-31T23:59:59Z",
                "scope_inclusion": "scope_1_2_3",
                "organizational_boundary": "operational_control",
                "facilities": [],
                "export_formats": ["pdf", "excel"],
                "include_verification": False,
                "include_methodology": True
            }
        }

    @validator("reporting_period_end")
    def validate_period(cls, v, values):
        """Ensure end date is after start date."""
        if "reporting_period_start" in values and v <= values["reporting_period_start"]:
            raise ValueError("reporting_period_end must be after reporting_period_start")
        return v


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class EmissionsSummary(BaseModel):
    """
    Summary of emissions for the report.

    Attributes:
        total_emissions_tco2e: Total emissions in tonnes CO2e
        scope_1_emissions_tco2e: Scope 1 emissions
        scope_2_location_tco2e: Scope 2 location-based emissions
        scope_2_market_tco2e: Scope 2 market-based emissions
        scope_3_emissions_tco2e: Scope 3 emissions (if included)
        biogenic_emissions_tco2e: Biogenic emissions (reported separately)
        year_over_year_change_percent: Change from previous period
    """
    total_emissions_tco2e: float = Field(..., description="Total emissions (tCO2e)")
    scope_1_emissions_tco2e: float = Field(..., description="Scope 1 emissions")
    scope_2_location_tco2e: float = Field(..., description="Scope 2 location-based")
    scope_2_market_tco2e: float = Field(..., description="Scope 2 market-based")
    scope_3_emissions_tco2e: Optional[float] = Field(default=None, description="Scope 3 emissions")
    biogenic_emissions_tco2e: float = Field(default=0.0, description="Biogenic emissions")
    year_over_year_change_percent: Optional[float] = Field(default=None, description="YoY change %")


class DataQualityMetrics(BaseModel):
    """
    Data quality metrics for the report.

    Attributes:
        overall_quality_score: Overall quality score (1-5)
        data_completeness_percent: Data completeness percentage
        primary_data_percent: Percentage of primary data
        uncertainty_percent: Overall uncertainty percentage
        methodology_adherence: Methodology adherence score
    """
    overall_quality_score: float = Field(..., ge=1, le=5, description="Quality score (1-5)")
    data_completeness_percent: float = Field(..., ge=0, le=100, description="Completeness %")
    primary_data_percent: float = Field(..., ge=0, le=100, description="Primary data %")
    uncertainty_percent: float = Field(..., ge=0, description="Uncertainty %")
    methodology_adherence: float = Field(..., ge=0, le=100, description="Methodology adherence %")


class ReportArtifact(BaseModel):
    """
    Report artifact (export file).

    Attributes:
        artifact_id: Unique artifact identifier
        format: File format
        file_name: File name
        file_size_bytes: File size in bytes
        download_url: URL to download the artifact
        generated_at: Generation timestamp
        expires_at: URL expiration timestamp
    """
    artifact_id: str = Field(..., description="Artifact ID")
    format: ReportFormat = Field(..., description="File format")
    file_name: str = Field(..., description="File name")
    file_size_bytes: int = Field(..., description="File size (bytes)")
    download_url: str = Field(..., description="Download URL")
    generated_at: datetime = Field(..., description="Generation timestamp")
    expires_at: datetime = Field(..., description="URL expiration")


class ComplianceCheckResult(BaseModel):
    """
    Individual compliance check result.

    Attributes:
        check_id: Check identifier
        check_name: Human-readable check name
        passed: Whether check passed
        severity: Check severity (info, warning, error)
        message: Result message
        recommendation: Recommendation if failed
    """
    check_id: str = Field(..., description="Check ID")
    check_name: str = Field(..., description="Check name")
    passed: bool = Field(..., description="Check passed")
    severity: str = Field(..., description="Severity level")
    message: str = Field(..., description="Result message")
    recommendation: Optional[str] = Field(default=None, description="Recommendation")


class ReportSummary(BaseModel):
    """
    Summary information about a compliance report.

    Attributes:
        report_id: Unique report identifier
        framework: Compliance framework
        report_type: Report type
        status: Current report status
        reporting_period_start: Start of reporting period
        reporting_period_end: End of reporting period
        total_emissions_tco2e: Total emissions (preview)
        created_at: Report creation timestamp
        updated_at: Last update timestamp
    """
    report_id: str = Field(..., description="Unique report ID")
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    report_type: ReportType = Field(..., description="Report type")
    status: ReportStatus = Field(..., description="Current status")
    reporting_period_start: datetime = Field(..., description="Period start")
    reporting_period_end: datetime = Field(..., description="Period end")
    total_emissions_tco2e: float = Field(..., description="Total emissions")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ReportDetail(BaseModel):
    """
    Detailed compliance report information.

    Attributes:
        report_id: Unique report identifier
        framework: Compliance framework
        report_type: Report type
        status: Current report status
        reporting_period_start: Start of reporting period
        reporting_period_end: End of reporting period
        scope_inclusion: Included scopes
        organizational_boundary: Organizational boundary used
        emissions_summary: Emissions summary
        data_quality: Data quality metrics
        compliance_checks: List of compliance check results
        artifacts: Generated report artifacts
        facilities_included: Number of facilities included
        created_by: User who created the report
        reviewed_by: User who reviewed (if applicable)
        submitted_at: Submission timestamp (if submitted)
        provenance_hash: SHA-256 hash for audit trail
        created_at: Report creation timestamp
        updated_at: Last update timestamp
    """
    report_id: str = Field(..., description="Unique report ID")
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    report_type: ReportType = Field(..., description="Report type")
    status: ReportStatus = Field(..., description="Current status")
    reporting_period_start: datetime = Field(..., description="Period start")
    reporting_period_end: datetime = Field(..., description="Period end")
    scope_inclusion: ScopeInclusion = Field(..., description="Included scopes")
    organizational_boundary: str = Field(..., description="Organizational boundary")
    emissions_summary: EmissionsSummary = Field(..., description="Emissions summary")
    data_quality: DataQualityMetrics = Field(..., description="Data quality metrics")
    compliance_checks: List[ComplianceCheckResult] = Field(..., description="Compliance checks")
    artifacts: List[ReportArtifact] = Field(default_factory=list, description="Report artifacts")
    facilities_included: int = Field(..., description="Facilities included")
    created_by: str = Field(..., description="Created by user")
    reviewed_by: Optional[str] = Field(default=None, description="Reviewed by user")
    submitted_at: Optional[datetime] = Field(default=None, description="Submission timestamp")
    provenance_hash: str = Field(..., description="Provenance hash")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "report_id": "rpt_abc123xyz",
                "framework": "ghg_protocol",
                "report_type": "annual",
                "status": "draft",
                "reporting_period_start": "2025-01-01T00:00:00Z",
                "reporting_period_end": "2025-12-31T23:59:59Z",
                "scope_inclusion": "scope_1_2_3",
                "organizational_boundary": "operational_control",
                "emissions_summary": {
                    "total_emissions_tco2e": 125430.5,
                    "scope_1_emissions_tco2e": 45230.2,
                    "scope_2_location_tco2e": 32100.8,
                    "scope_2_market_tco2e": 28500.5,
                    "scope_3_emissions_tco2e": 48099.5,
                    "biogenic_emissions_tco2e": 1250.0,
                    "year_over_year_change_percent": -5.2
                },
                "data_quality": {
                    "overall_quality_score": 4.2,
                    "data_completeness_percent": 98.5,
                    "primary_data_percent": 75.0,
                    "uncertainty_percent": 8.5,
                    "methodology_adherence": 95.0
                },
                "compliance_checks": [
                    {
                        "check_id": "chk_001",
                        "check_name": "Boundary Completeness",
                        "passed": True,
                        "severity": "error",
                        "message": "All facilities within boundary included"
                    }
                ],
                "facilities_included": 15,
                "created_by": "user@example.com",
                "provenance_hash": "sha256:abc123...",
                "created_at": "2025-12-07T10:00:00Z",
                "updated_at": "2025-12-07T10:00:00Z"
            }
        }


class ReportListResponse(BaseModel):
    """
    Paginated list of compliance reports.

    Attributes:
        items: List of report summaries
        total: Total number of reports
        page: Current page number
        page_size: Items per page
        total_pages: Total pages
        has_next: Has next page
        has_prev: Has previous page
    """
    items: List[ReportSummary] = Field(..., description="Report summaries")
    total: int = Field(..., description="Total reports")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# STORAGE (In-memory for demonstration)
# =============================================================================

# Sample reports for demonstration
_reports: Dict[str, ReportDetail] = {}


def _initialize_sample_reports():
    """Initialize sample reports for demonstration."""
    now = datetime.now(timezone.utc)

    sample_reports = [
        ReportDetail(
            report_id="rpt_001abc",
            framework=ComplianceFramework.GHG_PROTOCOL,
            report_type=ReportType.ANNUAL,
            status=ReportStatus.APPROVED,
            reporting_period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            reporting_period_end=datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            scope_inclusion=ScopeInclusion.SCOPE_1_2_3,
            organizational_boundary="operational_control",
            emissions_summary=EmissionsSummary(
                total_emissions_tco2e=125430.5,
                scope_1_emissions_tco2e=45230.2,
                scope_2_location_tco2e=32100.8,
                scope_2_market_tco2e=28500.5,
                scope_3_emissions_tco2e=48099.5,
                biogenic_emissions_tco2e=1250.0,
                year_over_year_change_percent=-5.2
            ),
            data_quality=DataQualityMetrics(
                overall_quality_score=4.2,
                data_completeness_percent=98.5,
                primary_data_percent=75.0,
                uncertainty_percent=8.5,
                methodology_adherence=95.0
            ),
            compliance_checks=[
                ComplianceCheckResult(
                    check_id="chk_001",
                    check_name="Boundary Completeness",
                    passed=True,
                    severity="error",
                    message="All facilities within boundary included"
                ),
                ComplianceCheckResult(
                    check_id="chk_002",
                    check_name="Scope 3 Categories",
                    passed=True,
                    severity="warning",
                    message="All relevant Scope 3 categories included"
                )
            ],
            artifacts=[
                ReportArtifact(
                    artifact_id="art_001",
                    format=ReportFormat.PDF,
                    file_name="ghg_report_2024_annual.pdf",
                    file_size_bytes=2456789,
                    download_url="https://api.greenlang.io/v1/artifacts/art_001/download",
                    generated_at=now - timedelta(days=30),
                    expires_at=now + timedelta(days=30)
                )
            ],
            facilities_included=15,
            created_by="sustainability@example.com",
            reviewed_by="compliance@example.com",
            submitted_at=now - timedelta(days=15),
            provenance_hash="sha256:abc123def456",
            created_at=now - timedelta(days=45),
            updated_at=now - timedelta(days=15)
        ),
        ReportDetail(
            report_id="rpt_002def",
            framework=ComplianceFramework.CDP,
            report_type=ReportType.ANNUAL,
            status=ReportStatus.DRAFT,
            reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            reporting_period_end=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            scope_inclusion=ScopeInclusion.SCOPE_1_2_3,
            organizational_boundary="operational_control",
            emissions_summary=EmissionsSummary(
                total_emissions_tco2e=118250.0,
                scope_1_emissions_tco2e=42100.0,
                scope_2_location_tco2e=30500.0,
                scope_2_market_tco2e=26800.0,
                scope_3_emissions_tco2e=45650.0,
                biogenic_emissions_tco2e=1100.0,
                year_over_year_change_percent=-5.7
            ),
            data_quality=DataQualityMetrics(
                overall_quality_score=4.0,
                data_completeness_percent=92.0,
                primary_data_percent=70.0,
                uncertainty_percent=10.0,
                methodology_adherence=92.0
            ),
            compliance_checks=[
                ComplianceCheckResult(
                    check_id="chk_003",
                    check_name="CDP Questions Completeness",
                    passed=False,
                    severity="warning",
                    message="3 required questions not yet answered",
                    recommendation="Complete questions C2.1a, C4.2, and C7.5"
                )
            ],
            facilities_included=15,
            created_by="sustainability@example.com",
            provenance_hash="sha256:xyz789abc123",
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(days=2)
        )
    ]

    for report in sample_reports:
        _reports[report.report_id] = report


# Initialize sample data
_initialize_sample_reports()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_report_id() -> str:
    """Generate a unique report ID."""
    return f"rpt_{uuid.uuid4().hex[:12]}"


def compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    import hashlib
    import json
    serialized = json.dumps(data, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"


def generate_report(request: ReportGenerationRequest) -> ReportDetail:
    """
    Generate a compliance report based on the request.

    In production, this would integrate with the calculation engine
    and data sources.
    """
    report_id = generate_report_id()
    now = datetime.now(timezone.utc)

    # Simulated emissions calculation
    emissions = EmissionsSummary(
        total_emissions_tco2e=round(100000 + (50000 * (now.month / 12)), 2),
        scope_1_emissions_tco2e=round(35000 + (15000 * (now.month / 12)), 2),
        scope_2_location_tco2e=28500.0,
        scope_2_market_tco2e=25000.0,
        scope_3_emissions_tco2e=45000.0 if request.scope_inclusion in [
            ScopeInclusion.SCOPE_1_2_3, ScopeInclusion.ALL_SCOPES
        ] else None,
        biogenic_emissions_tco2e=1200.0
    )

    # Simulated data quality
    data_quality = DataQualityMetrics(
        overall_quality_score=4.1,
        data_completeness_percent=95.0,
        primary_data_percent=72.0,
        uncertainty_percent=9.0,
        methodology_adherence=93.0
    )

    # Compliance checks
    checks = [
        ComplianceCheckResult(
            check_id="chk_auto_001",
            check_name="Reporting Period Validity",
            passed=True,
            severity="error",
            message="Reporting period is valid"
        ),
        ComplianceCheckResult(
            check_id="chk_auto_002",
            check_name="Boundary Definition",
            passed=True,
            severity="error",
            message="Organizational boundary properly defined"
        )
    ]

    provenance_data = {
        "framework": request.framework.value,
        "period_start": request.reporting_period_start.isoformat(),
        "period_end": request.reporting_period_end.isoformat(),
        "scope_inclusion": request.scope_inclusion.value,
        "generated_at": now.isoformat()
    }

    report = ReportDetail(
        report_id=report_id,
        framework=request.framework,
        report_type=request.report_type,
        status=ReportStatus.DRAFT,
        reporting_period_start=request.reporting_period_start,
        reporting_period_end=request.reporting_period_end,
        scope_inclusion=request.scope_inclusion,
        organizational_boundary=request.organizational_boundary,
        emissions_summary=emissions,
        data_quality=data_quality,
        compliance_checks=checks,
        artifacts=[],
        facilities_included=len(request.facilities) if request.facilities else 15,
        created_by="api_user",
        provenance_hash=compute_provenance_hash(provenance_data),
        created_at=now,
        updated_at=now
    )

    # Store report
    _reports[report_id] = report

    return report


# =============================================================================
# ROUTER DEFINITION
# =============================================================================

if FASTAPI_AVAILABLE:
    compliance_router = APIRouter(
        prefix="/compliance",
        tags=["Compliance"],
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            429: {"model": ErrorResponse, "description": "Rate Limited"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        }
    )


    @compliance_router.get(
        "/reports",
        response_model=ReportListResponse,
        summary="List compliance reports",
        description="""
        Retrieve a paginated list of compliance reports.

        Supports filtering by:
        - Framework (GHG Protocol, CDP, TCFD, etc.)
        - Status (draft, approved, submitted, etc.)
        - Report type (annual, quarterly, etc.)
        - Date range

        Results are sorted by creation date (most recent first).
        """,
        operation_id="list_compliance_reports"
    )
    async def list_compliance_reports(
        request: Request,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        framework: Optional[ComplianceFramework] = Query(None, description="Filter by framework"),
        status_filter: Optional[ReportStatus] = Query(None, alias="status", description="Filter by status"),
        report_type: Optional[ReportType] = Query(None, alias="type", description="Filter by report type"),
        from_date: Optional[datetime] = Query(None, description="Filter from date"),
        to_date: Optional[datetime] = Query(None, description="Filter to date"),
    ) -> ReportListResponse:
        """
        List compliance reports with pagination and filtering.

        Args:
            request: FastAPI request object
            page: Page number
            page_size: Items per page
            framework: Optional framework filter
            status_filter: Optional status filter
            report_type: Optional report type filter
            from_date: Optional start date filter
            to_date: Optional end date filter

        Returns:
            Paginated list of compliance reports
        """
        logger.info(f"Listing compliance reports: page={page}, page_size={page_size}")

        # Filter reports
        reports = list(_reports.values())

        if framework:
            reports = [r for r in reports if r.framework == framework]

        if status_filter:
            reports = [r for r in reports if r.status == status_filter]

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        if from_date:
            reports = [r for r in reports if r.created_at >= from_date]

        if to_date:
            reports = [r for r in reports if r.created_at <= to_date]

        # Sort by creation date (most recent first)
        reports.sort(key=lambda x: x.created_at, reverse=True)

        # Convert to summaries
        summaries = [
            ReportSummary(
                report_id=r.report_id,
                framework=r.framework,
                report_type=r.report_type,
                status=r.status,
                reporting_period_start=r.reporting_period_start,
                reporting_period_end=r.reporting_period_end,
                total_emissions_tco2e=r.emissions_summary.total_emissions_tco2e,
                created_at=r.created_at,
                updated_at=r.updated_at
            )
            for r in reports
        ]

        # Paginate
        total = len(summaries)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = summaries[start_idx:end_idx]

        return ReportListResponse(
            items=paginated,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


    @compliance_router.post(
        "/reports",
        response_model=ReportDetail,
        status_code=status.HTTP_201_CREATED,
        summary="Generate new compliance report",
        description="""
        Generate a new compliance report.

        The report is generated based on:
        - Selected compliance framework
        - Reporting period
        - Scope inclusion settings
        - Organizational boundary definition

        Report generation includes:
        - Emissions calculation for the period
        - Data quality assessment
        - Compliance checks against framework requirements
        - Optional artifact generation (PDF, Excel)

        Returns the generated report in draft status.
        """,
        operation_id="generate_compliance_report"
    )
    async def generate_compliance_report(
        request: Request,
        report_request: ReportGenerationRequest,
    ) -> ReportDetail:
        """
        Generate a new compliance report.

        Args:
            request: FastAPI request object
            report_request: Report generation parameters

        Returns:
            Generated compliance report

        Raises:
            HTTPException: If generation fails
        """
        logger.info(
            f"Generating compliance report: framework={report_request.framework.value}, "
            f"type={report_request.report_type.value}"
        )

        try:
            report = generate_report(report_request)

            logger.info(f"Report generated: {report.report_id}")

            return report

        except ValueError as e:
            logger.warning(f"Validation error in report generation: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "VALIDATION_ERROR",
                    "message": str(e)
                }
            )
        except Exception as e:
            logger.error(f"Report generation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "GENERATION_ERROR",
                    "message": "An error occurred during report generation"
                }
            )


    @compliance_router.get(
        "/reports/{report_id}",
        response_model=ReportDetail,
        summary="Get compliance report details",
        description="""
        Retrieve detailed information about a specific compliance report.

        Returns:
        - Report configuration and status
        - Emissions summary by scope
        - Data quality metrics
        - Compliance check results
        - Generated artifacts (if available)
        - Audit trail information
        """,
        operation_id="get_compliance_report"
    )
    async def get_compliance_report(
        request: Request,
        report_id: str,
    ) -> ReportDetail:
        """
        Get detailed compliance report information.

        Args:
            request: FastAPI request object
            report_id: Report identifier

        Returns:
            Detailed compliance report

        Raises:
            HTTPException: If report not found
        """
        logger.info(f"Getting compliance report: {report_id}")

        report = _reports.get(report_id)

        if not report:
            logger.warning(f"Report not found: {report_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "REPORT_NOT_FOUND",
                    "message": f"Compliance report '{report_id}' not found"
                }
            )

        return report

else:
    # Provide stub router when FastAPI is not available
    compliance_router = None
    logger.warning("FastAPI not available - compliance_router is None")
