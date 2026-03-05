"""
GL-ISO14064-APP Reports API

Report generation, mandatory reporting element checklist, and data export
per ISO 14064-1:2018 Clause 9.

ISO 14064-1 mandates 14 reporting elements (MRE-01 through MRE-14) covering
organization description, boundary, emissions by category, methodology,
GWP values, biogenic CO2, base year, significance assessment, exclusions,
and uncertainty.

Supported export formats: JSON, CSV, Excel, PDF.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/reports", tags=["Reports"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportFormat(str, Enum):
    """Supported report output formats."""
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"


class ReportStatus(str, Enum):
    """Report generation status."""
    DRAFT = "draft"
    REVIEW = "review"
    FINAL = "final"
    PUBLISHED = "published"


# ---------------------------------------------------------------------------
# Mandatory Reporting Elements (ISO 14064-1 Clause 9)
# ---------------------------------------------------------------------------

MANDATORY_ELEMENTS = [
    ("MRE-01", "Reporting organization description"),
    ("MRE-02", "Responsible person"),
    ("MRE-03", "Reporting period"),
    ("MRE-04", "Organizational boundary and consolidation approach"),
    ("MRE-05", "Direct GHG emissions (Category 1)"),
    ("MRE-06", "Indirect GHG emissions from imported energy (Category 2)"),
    ("MRE-07", "Quantification methodology description"),
    ("MRE-08", "GHG emissions and removals by gas type"),
    ("MRE-09", "Emission factors and GWP values used"),
    ("MRE-10", "Biogenic CO2 emissions reported separately"),
    ("MRE-11", "Base year and recalculation policy"),
    ("MRE-12", "Significance assessment for indirect categories (3-6)"),
    ("MRE-13", "Exclusions with justification"),
    ("MRE-14", "Uncertainty assessment"),
]


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    """Request to generate an ISO 14064-1 report."""
    inventory_id: str = Field(..., description="Inventory to report on")
    format: ReportFormat = Field(ReportFormat.JSON, description="Output format")
    title: str = Field(
        "ISO 14064-1:2018 GHG Report", max_length=500, description="Report title"
    )
    include_appendices: bool = Field(True, description="Include methodology appendices")
    include_charts: bool = Field(True, description="Include visualization charts")
    language: str = Field("en", description="Report language (ISO 639-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
                "format": "pdf",
                "title": "2025 ISO 14064-1 GHG Inventory Report",
                "include_appendices": True,
                "include_charts": True,
                "language": "en",
            }
        }


class ExportDataRequest(BaseModel):
    """Request to export inventory data."""
    format: ReportFormat = Field(..., description="Export format")
    categories: Optional[List[str]] = Field(None, description="ISO categories to include (null = all)")
    include_removals: bool = Field(True, description="Include removal data")

    class Config:
        json_schema_extra = {
            "example": {
                "format": "excel",
                "categories": ["category_1_direct", "category_2_energy"],
                "include_removals": True,
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class MandatoryElementResponse(BaseModel):
    """Status of a mandatory reporting element."""
    element_id: str
    description: str
    complete: bool
    content: Optional[str]


class ReportResponse(BaseModel):
    """Generated ISO 14064-1 report."""
    report_id: str
    inventory_id: str
    title: str
    format: str
    status: str
    mandatory_elements: List[MandatoryElementResponse]
    mandatory_completeness_pct: float
    page_count: Optional[int]
    file_size_bytes: Optional[int]
    download_url: Optional[str]
    generated_at: datetime


class ReportListEntry(BaseModel):
    """Summary entry in a report list."""
    report_id: str
    inventory_id: str
    format: str
    status: str
    mandatory_completeness_pct: float
    generated_at: datetime


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
# In-Memory Store
# ---------------------------------------------------------------------------

_reports: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _simulated_elements() -> List[Dict[str, Any]]:
    """Generate simulated mandatory element completeness for demo."""
    complete_ids = {
        "MRE-01", "MRE-03", "MRE-04", "MRE-05", "MRE-06",
        "MRE-07", "MRE-08", "MRE-09", "MRE-11",
    }
    elements = []
    for eid, desc in MANDATORY_ELEMENTS:
        elements.append({
            "element_id": eid,
            "description": desc,
            "complete": eid in complete_ids,
            "content": f"Completed for reporting year" if eid in complete_ids else None,
        })
    return elements


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate ISO 14064-1 report",
    description=(
        "Generate an ISO 14064-1:2018 compliant GHG report from an inventory. "
        "Supports JSON, CSV, Excel, and PDF formats.  Includes mandatory "
        "reporting element completeness assessment."
    ),
)
async def generate_report(request: GenerateReportRequest) -> ReportResponse:
    """Generate an ISO 14064-1 report."""
    report_id = _generate_id("rpt")
    now = _now()
    elements = _simulated_elements()
    complete_count = sum(1 for e in elements if e["complete"])
    completeness_pct = round(complete_count / len(elements) * 100, 1) if elements else 0.0

    page_count = None
    file_size = None
    if request.format == ReportFormat.PDF:
        page_count = 35 + len(elements) * 2
        file_size = page_count * 42000
    elif request.format == ReportFormat.EXCEL:
        file_size = 980000
    elif request.format == ReportFormat.CSV:
        file_size = 65000
    else:
        file_size = 280000

    report = {
        "report_id": report_id,
        "inventory_id": request.inventory_id,
        "title": request.title,
        "format": request.format.value,
        "status": ReportStatus.DRAFT.value,
        "mandatory_elements": elements,
        "mandatory_completeness_pct": completeness_pct,
        "page_count": page_count,
        "file_size_bytes": file_size,
        "download_url": f"https://api.greenlang.io/iso14064/reports/{report_id}/download",
        "generated_at": now,
    }
    _reports[report_id] = report
    return ReportResponse(**report)


@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Get report details",
    description="Retrieve the details and download URL of a generated report.",
)
async def get_report(report_id: str) -> ReportResponse:
    """Retrieve a report by ID."""
    report = _reports.get(report_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found",
        )
    return ReportResponse(**report)


@router.get(
    "/history/{inventory_id}",
    response_model=List[ReportListEntry],
    summary="Report generation history",
    description="List all reports generated for an inventory.",
)
async def get_report_history(
    inventory_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[ReportListEntry]:
    """List report history for an inventory."""
    reports = [r for r in _reports.values() if r["inventory_id"] == inventory_id]
    reports.sort(key=lambda r: r["generated_at"], reverse=True)
    return [
        ReportListEntry(
            report_id=r["report_id"],
            inventory_id=r["inventory_id"],
            format=r["format"],
            status=r["status"],
            mandatory_completeness_pct=r["mandatory_completeness_pct"],
            generated_at=r["generated_at"],
        )
        for r in reports[:limit]
    ]


@router.post(
    "/export/{inventory_id}",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export inventory data",
    description=(
        "Export raw inventory data in the requested format. Includes emission "
        "sources, removal sources, and calculation details."
    ),
)
async def export_inventory(
    inventory_id: str,
    request: ExportDataRequest,
) -> ExportResponse:
    """Export inventory data."""
    export_id = _generate_id("exp")
    record_count = 0
    categories = request.categories or [
        "category_1_direct", "category_2_energy", "category_3_transport",
        "category_4_products_used", "category_5_products_from_org", "category_6_other",
    ]
    # Simulated record counts per category
    counts = {
        "category_1_direct": 45, "category_2_energy": 12,
        "category_3_transport": 28, "category_4_products_used": 35,
        "category_5_products_from_org": 18, "category_6_other": 8,
    }
    for cat in categories:
        record_count += counts.get(cat, 10)
    if request.include_removals:
        record_count += 15

    file_sizes = {"json": 280000, "csv": 65000, "excel": 980000, "pdf": 2800000}
    return ExportResponse(
        export_id=export_id,
        inventory_id=inventory_id,
        format=request.format.value,
        status="completed",
        record_count=record_count,
        file_size_bytes=file_sizes.get(request.format.value, 500000),
        download_url=f"https://api.greenlang.io/iso14064/exports/{export_id}/download",
        created_at=_now(),
    )
