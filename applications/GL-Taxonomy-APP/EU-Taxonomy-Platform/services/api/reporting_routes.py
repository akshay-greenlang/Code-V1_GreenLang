"""
GL-Taxonomy-APP Reporting API

Generates EU Taxonomy-specific reports including Article 8 disclosures
for non-financial undertakings, EBA Pillar III ESG disclosures for
financial institutions, XBRL tagging, and multi-format exports.

Report Types:
    - Article 8: Mandatory disclosure for NFCs under NFRD/CSRD
    - EBA Pillar III: Mandatory for large CRR institutions
    - XBRL: EU ESEF / ESRS digital tagging
    - Qualitative: Contextual narrative disclosures

Export Formats:
    - PDF: Board-ready presentation
    - Excel: Detailed data workbook with Article 8 templates
    - CSV: Raw data for downstream processing
    - XBRL/iXBRL: Digital reporting format
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/reports", tags=["Reporting"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class Article8ReportRequest(BaseModel):
    """Generate Article 8 disclosure report."""
    org_id: str = Field(...)
    reporting_year: int = Field(..., ge=2022, le=2030)
    include_kpi_tables: bool = Field(True)
    include_qualitative: bool = Field(True)
    include_voluntary: bool = Field(False, description="Include voluntary CCM+CCA breakdown")
    framework: str = Field("csrd", description="csrd or nfrd")


class EBAReportRequest(BaseModel):
    """Generate EBA Pillar III ESG report."""
    institution_id: str = Field(...)
    reporting_date: str = Field(...)
    include_template_0: bool = Field(True)
    include_template_4: bool = Field(True)
    include_template_5: bool = Field(True)
    include_qualitative_narrative: bool = Field(True)


class ExportRequest(BaseModel):
    """Export report in specified format."""
    report_id: str = Field(...)
    format: str = Field(..., description="pdf, excel, csv, xbrl")
    language: str = Field("en", description="Report language")


class XBRLRequest(BaseModel):
    """Generate XBRL taxonomy mapping."""
    org_id: str = Field(...)
    reporting_year: int = Field(...)
    taxonomy_version: str = Field("ESRS_2024", description="ESRS taxonomy version")
    include_inline: bool = Field(True, description="Generate iXBRL")


class QualitativeDisclosureRequest(BaseModel):
    """Add qualitative disclosure to report."""
    section: str = Field(..., description="accounting_policy, methodology, limitations, plans")
    content: str = Field(..., min_length=10, max_length=10000)
    author: Optional[str] = Field(None, max_length=200)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ReportResponse(BaseModel):
    """Generated report."""
    report_id: str
    org_id: str
    report_type: str
    title: str
    reporting_year: int
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    generated_at: datetime


class ReportHistoryEntry(BaseModel):
    """Report history entry."""
    report_id: str
    report_type: str
    title: str
    reporting_year: int
    generated_at: datetime
    format: str


class ReportHistoryResponse(BaseModel):
    """Report history."""
    org_id: str
    reports: List[ReportHistoryEntry]
    total_count: int
    generated_at: datetime


class ExportResponse(BaseModel):
    """Report export result."""
    report_id: str
    format: str
    file_name: str
    file_size_kb: float
    download_url: str
    expires_at: datetime
    generated_at: datetime


class XBRLResponse(BaseModel):
    """XBRL mapping result."""
    mapping_id: str
    org_id: str
    taxonomy_version: str
    reporting_year: int
    mapped_elements: int
    unmapped_elements: int
    coverage_pct: float
    xbrl_tags: List[Dict[str, Any]]
    inline_xbrl: bool
    generated_at: datetime


class CompareReportsResponse(BaseModel):
    """Report comparison."""
    org_id: str
    report_1: Dict[str, Any]
    report_2: Dict[str, Any]
    changes: Dict[str, Any]
    generated_at: datetime


class QualitativeDisclosureResponse(BaseModel):
    """Qualitative disclosure added."""
    disclosure_id: str
    report_id: str
    section: str
    content_preview: str
    word_count: int
    author: Optional[str]
    added_at: datetime


class DisclosureSummaryResponse(BaseModel):
    """Disclosure summary."""
    org_id: str
    mandatory_sections: Dict[str, str]
    voluntary_sections: Dict[str, str]
    completeness_pct: float
    missing_sections: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_reports: Dict[str, Dict[str, Any]] = {}
_disclosures: Dict[str, List[Dict[str, Any]]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/article-8",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Article 8 report",
    description=(
        "Generate Article 8 disclosure report per EU Taxonomy Regulation "
        "and Article 8 Delegated Act (2021/2178). Includes KPI tables, "
        "activity-level breakdown, and qualitative disclosures."
    ),
)
async def generate_article8(request: Article8ReportRequest) -> ReportResponse:
    """Generate Article 8 report."""
    report_id = _generate_id("rpt_a8")
    sections = [
        {"section": "Executive Summary", "content": f"EU Taxonomy Report for FY{request.reporting_year}. Non-financial undertaking reporting under {request.framework.upper()}."},
    ]

    if request.include_kpi_tables:
        sections.extend([
            {"section": "KPI Table 1: Turnover", "kpi": "turnover", "eligible_pct": 65.0, "aligned_pct": 42.0, "non_eligible_pct": 35.0,
             "by_objective": {"CCM": 38.0, "CCA": 3.0, "WTR": 0.5, "CE": 0.3, "PPC": 0.1, "BIO": 0.1}},
            {"section": "KPI Table 2: CapEx", "kpi": "capex", "eligible_pct": 72.0, "aligned_pct": 56.0, "non_eligible_pct": 28.0},
            {"section": "KPI Table 3: OpEx", "kpi": "opex", "eligible_pct": 66.7, "aligned_pct": 46.7, "non_eligible_pct": 33.3},
        ])

    sections.append({
        "section": "Activity-Level Disclosure",
        "activities": [
            {"code": "4.1", "name": "Solar PV", "sector": "Energy", "turnover_pct": 15.0, "aligned": True},
            {"code": "7.7", "name": "Building Acquisition", "sector": "Real Estate", "turnover_pct": 12.0, "aligned": True},
            {"code": "6.5", "name": "Vehicle Fleet", "sector": "Transport", "turnover_pct": 8.0, "aligned": True},
            {"code": "3.1", "name": "RE Tech Manufacturing", "sector": "Manufacturing", "turnover_pct": 7.0, "aligned": True},
        ],
    })

    if request.include_qualitative:
        sections.extend([
            {"section": "Accounting Policy", "content": "Taxonomy KPIs calculated per Article 8 DA. Denominators per IFRS."},
            {"section": "Contextual Information", "content": "Alignment improvements driven by CapEx plan for building renovations and EV fleet transition."},
            {"section": "Compliance Statement", "content": f"Report prepared in accordance with Article 8 of Regulation (EU) 2020/852 and Commission Delegated Regulation (EU) 2021/2178. Reporting framework: {request.framework.upper()}."},
        ])

    if request.include_voluntary:
        sections.append({"section": "Voluntary CCM/CCA Breakdown", "ccm_aligned_pct": 38.0, "cca_aligned_pct": 3.0, "env_da_aligned_pct": 1.0})

    data = {
        "report_id": report_id, "org_id": request.org_id,
        "report_type": "article_8", "title": f"EU Taxonomy Article 8 Report - FY{request.reporting_year}",
        "reporting_year": request.reporting_year, "sections": sections,
        "metadata": {"framework": request.framework, "delegated_act": "2021/2178", "version": "v3"},
        "generated_at": _now(),
    }
    _reports[report_id] = data
    return ReportResponse(**data)


@router.post(
    "/eba",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate EBA report",
    description="Generate EBA Pillar III ESG disclosure report for financial institutions.",
)
async def generate_eba(request: EBAReportRequest) -> ReportResponse:
    """Generate EBA Pillar III report."""
    report_id = _generate_id("rpt_eba")
    sections = [
        {"section": "Institution Overview", "institution_id": request.institution_id, "reporting_date": request.reporting_date},
    ]

    if request.include_template_0:
        sections.append({"section": "Template 0: Summary of KPIs", "gar_stock_pct": 16.2, "gar_flow_pct": 25.5, "btar_pct": 18.0})
    if request.include_template_4:
        sections.append({"section": "Template 4: GAR Stock", "total_covered_eur": 38000000000, "aligned_eur": 6156000000, "by_asset_class": {"loans_nfc": 45, "mortgages": 30, "securities": 15, "equity": 10}})
    if request.include_template_5:
        sections.append({"section": "Template 5: GAR Flow", "new_originations_eur": 8000000000, "aligned_new_eur": 2040000000})
    if request.include_qualitative_narrative:
        sections.append({"section": "Qualitative Narrative", "content": "The institution's green lending strategy targets 25% GAR by 2027 through increased renewable energy project finance and green mortgage origination."})

    data = {
        "report_id": report_id, "org_id": request.institution_id,
        "report_type": "eba_pillar3", "title": f"EBA Pillar III ESG Disclosure - {request.reporting_date}",
        "reporting_year": int(request.reporting_date[:4]),
        "sections": sections,
        "metadata": {"regulation": "ITS on ESG Risks", "eba_version": "v3"},
        "generated_at": _now(),
    }
    _reports[report_id] = data
    return ReportResponse(**data)


@router.post(
    "/export",
    response_model=ExportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Export report (PDF/Excel/CSV)",
    description="Export a generated report in the specified format.",
)
async def export_report(request: ExportRequest) -> ExportResponse:
    """Export report."""
    if request.format not in ("pdf", "excel", "csv", "xbrl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{request.format}'. Supported: pdf, excel, csv, xbrl",
        )

    report = _reports.get(request.report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report {request.report_id} not found.")

    ext_map = {"pdf": "pdf", "excel": "xlsx", "csv": "csv", "xbrl": "xbrl"}
    size_map = {"pdf": 385.5, "excel": 245.2, "csv": 42.8, "xbrl": 125.0}
    now = _now()

    return ExportResponse(
        report_id=request.report_id, format=request.format,
        file_name=f"taxonomy_{report['report_type']}_{report['reporting_year']}.{ext_map[request.format]}",
        file_size_kb=size_map.get(request.format, 100),
        download_url=f"/api/v1/taxonomy/reports/download/{request.report_id}.{ext_map[request.format]}",
        expires_at=datetime(now.year, now.month, now.day, 23, 59, 59),
        generated_at=now,
    )


@router.post(
    "/xbrl",
    response_model=XBRLResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate XBRL mapping",
    description="Generate XBRL / iXBRL taxonomy mapping for digital reporting.",
)
async def generate_xbrl(request: XBRLRequest) -> XBRLResponse:
    """Generate XBRL mapping."""
    mapping_id = _generate_id("xbrl")
    tags = [
        {"element": "esrs:TaxonomyAlignedTurnoverProportion", "value": 42.0, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:TaxonomyAlignedCapExProportion", "value": 56.0, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:TaxonomyAlignedOpExProportion", "value": 46.7, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:TaxonomyEligibleTurnoverProportion", "value": 65.0, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:TaxonomyEligibleCapExProportion", "value": 72.0, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:TaxonomyEligibleOpExProportion", "value": 66.7, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:TransitionalActivitiesTurnoverProportion", "value": 8.5, "unit": "percent", "period": str(request.reporting_year)},
        {"element": "esrs:EnablingActivitiesTurnoverProportion", "value": 12.0, "unit": "percent", "period": str(request.reporting_year)},
    ]

    return XBRLResponse(
        mapping_id=mapping_id, org_id=request.org_id,
        taxonomy_version=request.taxonomy_version,
        reporting_year=request.reporting_year,
        mapped_elements=len(tags), unmapped_elements=2,
        coverage_pct=round(len(tags) / (len(tags) + 2) * 100, 1),
        xbrl_tags=tags, inline_xbrl=request.include_inline,
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/history",
    response_model=ReportHistoryResponse,
    summary="Report history",
    description="Get the history of all generated reports for an organization.",
)
async def get_report_history(
    org_id: str,
    report_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> ReportHistoryResponse:
    """Get report history."""
    records = [r for r in _reports.values() if r["org_id"] == org_id]
    if report_type:
        records = [r for r in records if r["report_type"] == report_type]
    records.sort(key=lambda r: r["generated_at"], reverse=True)

    entries = [
        ReportHistoryEntry(
            report_id=r["report_id"], report_type=r["report_type"],
            title=r["title"], reporting_year=r["reporting_year"],
            generated_at=r["generated_at"], format="json",
        )
        for r in records[:limit]
    ]

    return ReportHistoryResponse(
        org_id=org_id, reports=entries, total_count=len(records), generated_at=_now(),
    )


@router.get(
    "/{org_id}/compare",
    response_model=CompareReportsResponse,
    summary="Compare reports",
    description="Compare two reports across reporting periods.",
)
async def compare_reports(
    org_id: str,
    year_1: int = Query(2024, ge=2022, le=2030),
    year_2: int = Query(2025, ge=2022, le=2030),
) -> CompareReportsResponse:
    """Compare reports across periods."""
    r1 = {"year": year_1, "turnover_alignment_pct": 35.5, "capex_alignment_pct": 45.0, "opex_alignment_pct": 38.0}
    r2 = {"year": year_2, "turnover_alignment_pct": 42.0, "capex_alignment_pct": 56.0, "opex_alignment_pct": 46.7}
    changes = {k: round(r2[k] - r1[k], 1) for k in r1 if k != "year"}

    return CompareReportsResponse(
        org_id=org_id, report_1=r1, report_2=r2, changes=changes, generated_at=_now(),
    )


@router.post(
    "/{report_id}/qualitative",
    response_model=QualitativeDisclosureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add qualitative disclosure",
    description="Add a qualitative disclosure section to a report.",
)
async def add_qualitative(
    report_id: str,
    request: QualitativeDisclosureRequest,
) -> QualitativeDisclosureResponse:
    """Add qualitative disclosure."""
    if report_id not in _reports:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report {report_id} not found.")

    disclosure_id = _generate_id("disc")
    words = len(request.content.split())
    preview = request.content[:200] + "..." if len(request.content) > 200 else request.content

    entry = {
        "disclosure_id": disclosure_id, "report_id": report_id,
        "section": request.section, "content_preview": preview,
        "word_count": words, "author": request.author, "added_at": _now(),
    }

    if report_id not in _disclosures:
        _disclosures[report_id] = []
    _disclosures[report_id].append(entry)

    return QualitativeDisclosureResponse(**entry)


@router.get(
    "/{org_id}/disclosure-summary",
    response_model=DisclosureSummaryResponse,
    summary="Disclosure summary",
    description="Get summary of mandatory and voluntary disclosure completeness.",
)
async def get_disclosure_summary(org_id: str) -> DisclosureSummaryResponse:
    """Get disclosure summary."""
    mandatory = {
        "kpi_turnover": "complete",
        "kpi_capex": "complete",
        "kpi_opex": "complete",
        "activity_breakdown": "complete",
        "accounting_policy": "complete",
        "compliance_statement": "complete",
    }
    voluntary = {
        "ccm_cca_breakdown": "partial",
        "contextual_narrative": "complete",
        "forward_looking_plans": "missing",
    }
    total = len(mandatory) + len(voluntary)
    complete = sum(1 for v in mandatory.values() if v == "complete") + sum(1 for v in voluntary.values() if v == "complete")
    missing = [k for k, v in {**mandatory, **voluntary}.items() if v == "missing"]

    return DisclosureSummaryResponse(
        org_id=org_id,
        mandatory_sections=mandatory, voluntary_sections=voluntary,
        completeness_pct=round(complete / total * 100, 1),
        missing_sections=missing, generated_at=_now(),
    )
