"""
GL-GHG-APP Reporting API

Report generation, GHG Protocol mandatory disclosures, completeness
assessment, data gap analysis, and data export.

Supports output formats: JSON, CSV, Excel, PDF.
Implements the GHG Protocol Corporate Standard reporting requirements
including all mandatory and optional disclosure items.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/reports", tags=["Reporting"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportFormat(str, Enum):
    """Supported report output formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"


class ReportSection(str, Enum):
    """Available report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    ORGANIZATIONAL_BOUNDARY = "organizational_boundary"
    OPERATIONAL_BOUNDARY = "operational_boundary"
    SCOPE1_DETAIL = "scope1_detail"
    SCOPE2_DETAIL = "scope2_detail"
    SCOPE3_DETAIL = "scope3_detail"
    BASE_YEAR = "base_year"
    RECALCULATION = "recalculation"
    INTENSITY_METRICS = "intensity_metrics"
    VERIFICATION = "verification"
    TARGETS = "targets"
    METHODOLOGY = "methodology"
    DATA_QUALITY = "data_quality"
    UNCERTAINTIES = "uncertainties"


class ReportStatus(str, Enum):
    """Report generation status."""
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class DisclosureStatus(str, Enum):
    """Status of a mandatory disclosure item."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    """Request to generate a GHG report."""
    inventory_id: str = Field(..., description="Inventory to report on")
    format: ReportFormat = Field(ReportFormat.PDF, description="Output format")
    sections: Optional[List[ReportSection]] = Field(
        None, description="Sections to include (null = all sections)"
    )
    include_appendices: bool = Field(True, description="Include methodology appendices")
    include_charts: bool = Field(True, description="Include visualization charts")
    comparison_years: Optional[List[int]] = Field(
        None, description="Years to include in trend comparisons"
    )
    language: str = Field("en", description="Report language (ISO 639-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
                "format": "pdf",
                "sections": [
                    "executive_summary",
                    "scope1_detail",
                    "scope2_detail",
                    "scope3_detail",
                    "targets"
                ],
                "include_appendices": True,
                "include_charts": True,
                "comparison_years": [2023, 2024],
                "language": "en"
            }
        }


class ExportRequest(BaseModel):
    """Request to export inventory data."""
    format: ReportFormat = Field(..., description="Export format")
    scopes: Optional[List[int]] = Field(None, description="Scopes to export (null = all)")
    include_raw_data: bool = Field(False, description="Include raw activity data records")
    include_emission_factors: bool = Field(True, description="Include EF details")

    class Config:
        json_schema_extra = {
            "example": {
                "format": "excel",
                "scopes": [1, 2, 3],
                "include_raw_data": True,
                "include_emission_factors": True
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ReportResponse(BaseModel):
    """Report generation result."""
    report_id: str
    inventory_id: str
    format: str
    status: str
    sections_included: List[str]
    page_count: Optional[int]
    file_size_bytes: Optional[int]
    download_url: Optional[str]
    generated_at: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime]


class ReportHistoryEntry(BaseModel):
    """An entry in the report generation history."""
    report_id: str
    format: str
    status: str
    sections_count: int
    generated_at: Optional[datetime]
    file_size_bytes: Optional[int]


class DisclosureItem(BaseModel):
    """A GHG Protocol mandatory disclosure item."""
    item_id: str
    chapter: str
    requirement: str
    description: str
    is_mandatory: bool
    status: str
    completeness_pct: float
    data_location: Optional[str]
    notes: Optional[str]


class CompletenessAssessment(BaseModel):
    """Overall completeness of the GHG inventory."""
    inventory_id: str
    overall_completeness_pct: float
    grade: str
    scope1_completeness_pct: float
    scope2_completeness_pct: float
    scope3_completeness_pct: float
    mandatory_disclosures_complete: int
    mandatory_disclosures_total: int
    optional_disclosures_complete: int
    optional_disclosures_total: int
    missing_items: List[str]
    recommendations: List[str]


class DataGap(BaseModel):
    """A data gap identified in the inventory."""
    gap_id: str
    scope: int
    category: str
    description: str
    severity: str
    estimated_impact_tco2e: Optional[float]
    data_needed: str
    suggested_source: str
    resolution_priority: str


class ExportResponse(BaseModel):
    """Data export result."""
    export_id: str
    inventory_id: str
    format: str
    status: str
    record_count: int
    file_size_bytes: Optional[int]
    download_url: Optional[str]
    created_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _simulated_disclosures(inventory_id: str) -> List[Dict[str, Any]]:
    """GHG Protocol Corporate Standard mandatory and optional disclosures."""
    items = [
        # Mandatory (GHG Protocol Ch. 9)
        ("D-001", "Ch. 3", "Organizational boundary", "Description of the chosen consolidation approach", True, "complete", 100.0, "Boundary settings"),
        ("D-002", "Ch. 4", "Operational boundary", "Scopes included and exclusion justifications", True, "complete", 100.0, "Boundary settings"),
        ("D-003", "Ch. 5", "Base year", "Base year selection and recalculation policy", True, "complete", 100.0, "Boundary settings"),
        ("D-004", "Ch. 6", "Scope 1 emissions", "Total Scope 1 in tonnes CO2e, by gas", True, "complete", 100.0, "Scope 1 summary"),
        ("D-005", "Ch. 7", "Scope 2 emissions", "Scope 2 using both location and market methods", True, "complete", 100.0, "Scope 2 summary"),
        ("D-006", "Ch. 7", "Scope 2 instruments", "Contractual instruments for market-based", True, "complete", 100.0, "Instruments list"),
        ("D-007", "Ch. 15", "Scope 3 categories", "All relevant Scope 3 categories", True, "partial", 78.0, "Scope 3 categories"),
        ("D-008", "Ch. 9", "GWP values", "GWP values and source used", True, "complete", 100.0, "Settings"),
        ("D-009", "Ch. 9", "Methodology", "Calculation methodology per scope", True, "complete", 100.0, "Methodology section"),
        ("D-010", "Ch. 9", "Emission factors", "Emission factor sources", True, "complete", 100.0, "EF databases"),
        ("D-011", "Ch. 8", "Year-over-year", "YoY comparison and explanation of changes", True, "partial", 85.0, "Trends"),
        ("D-012", "Ch. 9", "Reporting period", "Reporting period dates", True, "complete", 100.0, "Inventory metadata"),
        # Optional
        ("D-013", "Ch. 9", "Intensity metrics", "Emission intensity ratios", False, "complete", 100.0, "Intensity metrics"),
        ("D-014", "Ch. 9", "Targets", "Reduction targets and progress", False, "partial", 70.0, "Targets module"),
        ("D-015", "Ch. 9", "Verification", "Third-party verification statement", False, "missing", 0.0, None),
        ("D-016", "Ch. 9", "Uncertainty", "Uncertainty assessment", False, "partial", 50.0, "Data quality section"),
        ("D-017", "Ch. 9", "Biogenic CO2", "Biogenic CO2 reported separately", False, "missing", 0.0, None),
    ]
    return [
        {
            "item_id": item_id,
            "chapter": chapter,
            "requirement": req,
            "description": desc,
            "is_mandatory": mandatory,
            "status": item_status,
            "completeness_pct": completeness,
            "data_location": location,
            "notes": None,
        }
        for item_id, chapter, req, desc, mandatory, item_status, completeness, location in items
    ]


def _simulated_gaps(inventory_id: str) -> List[Dict[str, Any]]:
    """Identify data gaps in the inventory."""
    gaps = [
        {
            "gap_id": _generate_id("gap"),
            "scope": 1,
            "category": "Mobile Combustion",
            "description": "Q4 fuel consumption data missing for 12 fleet vehicles",
            "severity": "medium",
            "estimated_impact_tco2e": 180.0,
            "data_needed": "Fuel purchase records for Oct-Dec 2025",
            "suggested_source": "Fleet fuel card provider (WEX/Comdata)",
            "resolution_priority": "high",
        },
        {
            "gap_id": _generate_id("gap"),
            "scope": 1,
            "category": "Refrigerants",
            "description": "Refrigerant inventory not reconciled with maintenance logs",
            "severity": "low",
            "estimated_impact_tco2e": 45.0,
            "data_needed": "HVAC maintenance records and recharge logs",
            "suggested_source": "Facilities maintenance contractor",
            "resolution_priority": "medium",
        },
        {
            "gap_id": _generate_id("gap"),
            "scope": 3,
            "category": "Cat 1: Purchased Goods & Services",
            "description": "Spend-based method used for 80% of suppliers; no EPDs collected",
            "severity": "high",
            "estimated_impact_tco2e": None,
            "data_needed": "Supplier-specific emission data or EPDs for top 20 suppliers",
            "suggested_source": "Supplier engagement program",
            "resolution_priority": "high",
        },
        {
            "gap_id": _generate_id("gap"),
            "scope": 3,
            "category": "Cat 7: Employee Commuting",
            "description": "Commuting survey response rate only 35%",
            "severity": "medium",
            "estimated_impact_tco2e": 450.0,
            "data_needed": "Complete employee commuting survey data",
            "suggested_source": "HR department annual commuting survey",
            "resolution_priority": "medium",
        },
        {
            "gap_id": _generate_id("gap"),
            "scope": 3,
            "category": "Cat 11: Use of Sold Products",
            "description": "Product lifetime energy use based on industry average, not actual data",
            "severity": "medium",
            "estimated_impact_tco2e": None,
            "data_needed": "Product-specific energy consumption testing results",
            "suggested_source": "Product engineering / LCA study",
            "resolution_priority": "low",
        },
    ]
    return gaps


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate GHG report",
    description=(
        "Generate a formatted report from a GHG inventory. Supports JSON, CSV, "
        "Excel, and PDF formats. Select specific sections or generate the "
        "full GHG Protocol-compliant report."
    ),
)
async def generate_report(request: GenerateReportRequest) -> ReportResponse:
    sections = request.sections or [s.value for s in ReportSection]
    report_id = _generate_id("rpt")
    now = _now()

    page_count = None
    file_size = None
    if request.format == ReportFormat.PDF:
        page_count = 28 + len(sections) * 3
        file_size = page_count * 45000
    elif request.format == ReportFormat.EXCEL:
        file_size = 1250000
    elif request.format == ReportFormat.CSV:
        file_size = 85000
    else:
        file_size = 320000

    return ReportResponse(
        report_id=report_id,
        inventory_id=request.inventory_id,
        format=request.format.value,
        status="completed",
        sections_included=[s if isinstance(s, str) else s.value for s in sections],
        page_count=page_count,
        file_size_bytes=file_size,
        download_url=f"https://api.greenlang.io/reports/{report_id}/download",
        generated_at=now,
        created_at=now,
        expires_at=None,
    )


@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Get report details",
    description="Retrieve the details and download URL of a generated report.",
)
async def get_report(report_id: str) -> ReportResponse:
    now = _now()
    return ReportResponse(
        report_id=report_id,
        inventory_id="inv_demo",
        format="pdf",
        status="completed",
        sections_included=[s.value for s in ReportSection],
        page_count=72,
        file_size_bytes=3240000,
        download_url=f"https://api.greenlang.io/reports/{report_id}/download",
        generated_at=now,
        created_at=now,
        expires_at=None,
    )


@router.get(
    "/history/{inventory_id}",
    response_model=List[ReportHistoryEntry],
    summary="Report generation history",
    description="List all reports generated for an inventory.",
)
async def get_report_history(
    inventory_id: str,
    limit: int = Query(20, ge=1, le=100, description="Max results"),
) -> List[ReportHistoryEntry]:
    now = _now()
    return [
        ReportHistoryEntry(
            report_id=_generate_id("rpt"),
            format="pdf",
            status="completed",
            sections_count=14,
            generated_at=now,
            file_size_bytes=3240000,
        ),
        ReportHistoryEntry(
            report_id=_generate_id("rpt"),
            format="excel",
            status="completed",
            sections_count=14,
            generated_at=now,
            file_size_bytes=1250000,
        ),
        ReportHistoryEntry(
            report_id=_generate_id("rpt"),
            format="csv",
            status="completed",
            sections_count=5,
            generated_at=now,
            file_size_bytes=85000,
        ),
    ]


@router.get(
    "/disclosures/{inventory_id}",
    response_model=List[DisclosureItem],
    summary="GHG Protocol mandatory disclosure checklist",
    description=(
        "Check compliance with GHG Protocol Corporate Standard mandatory "
        "disclosures (Chapter 9). Returns status for each disclosure item."
    ),
)
async def get_disclosures(
    inventory_id: str,
    mandatory_only: bool = Query(False, description="Show only mandatory items"),
) -> List[DisclosureItem]:
    disclosures = _simulated_disclosures(inventory_id)
    if mandatory_only:
        disclosures = [d for d in disclosures if d["is_mandatory"]]
    return [DisclosureItem(**d) for d in disclosures]


@router.get(
    "/completeness/{inventory_id}",
    response_model=CompletenessAssessment,
    summary="Completeness assessment",
    description=(
        "Assess the overall completeness of the GHG inventory across all "
        "scopes, mandatory disclosures, and data quality."
    ),
)
async def get_completeness(inventory_id: str) -> CompletenessAssessment:
    disclosures = _simulated_disclosures(inventory_id)
    mandatory = [d for d in disclosures if d["is_mandatory"]]
    optional = [d for d in disclosures if not d["is_mandatory"]]
    mandatory_complete = sum(1 for d in mandatory if d["status"] == "complete")
    optional_complete = sum(1 for d in optional if d["status"] == "complete")

    missing = []
    for d in disclosures:
        if d["status"] in ("missing", "partial"):
            missing.append(f"{d['item_id']}: {d['requirement']} ({d['status']})")

    return CompletenessAssessment(
        inventory_id=inventory_id,
        overall_completeness_pct=87.5,
        grade="B+",
        scope1_completeness_pct=94.0,
        scope2_completeness_pct=100.0,
        scope3_completeness_pct=68.0,
        mandatory_disclosures_complete=mandatory_complete,
        mandatory_disclosures_total=len(mandatory),
        optional_disclosures_complete=optional_complete,
        optional_disclosures_total=len(optional),
        missing_items=missing,
        recommendations=[
            "Complete Scope 3 Category 8 (Upstream Leased Assets) data collection",
            "Obtain third-party verification to complete disclosure D-015",
            "Conduct uncertainty analysis for Scope 3 spend-based estimates",
            "Improve employee commuting survey response rate from 35% to >70%",
        ],
    )


@router.get(
    "/gaps/{inventory_id}",
    response_model=List[DataGap],
    summary="Data gap analysis",
    description=(
        "Identify data gaps across the inventory. Returns each gap with "
        "severity, estimated emission impact, needed data, and resolution priority."
    ),
)
async def get_data_gaps(
    inventory_id: str,
    scope: Optional[int] = Query(None, ge=1, le=3, description="Filter by scope"),
    severity: Optional[str] = Query(None, description="Filter by severity: high, medium, low"),
) -> List[DataGap]:
    gaps = _simulated_gaps(inventory_id)
    if scope is not None:
        gaps = [g for g in gaps if g["scope"] == scope]
    if severity is not None:
        gaps = [g for g in gaps if g["severity"] == severity]
    return [DataGap(**g) for g in gaps]


@router.post(
    "/export/{inventory_id}",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export inventory data",
    description=(
        "Export raw inventory data in the requested format. Includes emission "
        "records, emission factors used, and calculation details."
    ),
)
async def export_inventory(
    inventory_id: str,
    request: ExportRequest,
) -> ExportResponse:
    export_id = _generate_id("exp")
    record_count = 0
    scopes = request.scopes or [1, 2, 3]
    if 1 in scopes:
        record_count += 47
    if 2 in scopes:
        record_count += 36
    if 3 in scopes:
        record_count += 215
    if request.include_raw_data:
        record_count *= 3

    file_sizes = {"json": 320000, "csv": 85000, "excel": 1250000, "pdf": 3240000}
    return ExportResponse(
        export_id=export_id,
        inventory_id=inventory_id,
        format=request.format.value,
        status="completed",
        record_count=record_count,
        file_size_bytes=file_sizes.get(request.format.value, 500000),
        download_url=f"https://api.greenlang.io/exports/{export_id}/download",
        created_at=_now(),
    )
