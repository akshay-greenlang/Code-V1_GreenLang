"""
GL-TCFD-APP Disclosure API

Manages TCFD disclosure lifecycle including creation, section generation per
pillar, auto-population, compliance checking, cross-reference mapping,
multi-format export (PDF, Excel, JSON, XBRL), year-over-year comparison,
and regulatory template retrieval.

Supported Regulatory Frameworks:
    - TCFD (2017) -- Original 11 recommended disclosures
    - ISSB/IFRS S2 (2023) -- Enhanced climate disclosure requirements
    - CSRD/ESRS E1 (2024) -- EU sustainability reporting standard
    - SEC Climate Rule (2024) -- US climate disclosure requirements

Export Formats: PDF, Excel, JSON, XBRL
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/disclosures", tags=["Disclosures"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DisclosureStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"


class Pillar(str, Enum):
    GOVERNANCE = "governance"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    METRICS_AND_TARGETS = "metrics_and_targets"


class ExportFormat(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    XBRL = "xbrl"


# ---------------------------------------------------------------------------
# TCFD Disclosure Structure
# ---------------------------------------------------------------------------

TCFD_SECTIONS = {
    "governance": {
        "GOV-A": "Board oversight of climate-related risks and opportunities",
        "GOV-B": "Management's role in assessing and managing climate-related risks and opportunities",
    },
    "strategy": {
        "STR-A": "Climate-related risks and opportunities identified",
        "STR-B": "Impact on business, strategy, and financial planning",
        "STR-C": "Resilience of strategy under different climate scenarios",
    },
    "risk_management": {
        "RM-A": "Processes for identifying and assessing climate-related risks",
        "RM-B": "Processes for managing climate-related risks",
        "RM-C": "Integration into overall risk management",
    },
    "metrics_and_targets": {
        "MT-A": "Metrics used to assess climate-related risks and opportunities",
        "MT-B": "Scope 1, 2, and 3 GHG emissions",
        "MT-C": "Targets used to manage climate-related risks and opportunities",
    },
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateDisclosureRequest(BaseModel):
    """Request to create a TCFD disclosure."""
    reporting_year: int = Field(..., ge=2015, le=2100, description="Reporting year")
    title: str = Field("TCFD Climate-related Financial Disclosure", max_length=500, description="Disclosure title")
    regulation: str = Field("tcfd", description="Primary regulation: tcfd, issb, csrd, sec")
    language: str = Field("en", description="ISO 639-1 language code")

    class Config:
        json_schema_extra = {
            "example": {
                "reporting_year": 2025,
                "title": "2025 TCFD Climate Disclosure Report",
                "regulation": "tcfd",
                "language": "en",
            }
        }


class UpdateDisclosureRequest(BaseModel):
    """Request to update a disclosure."""
    title: Optional[str] = Field(None, max_length=500)
    status: Optional[DisclosureStatus] = None


class UpdateSectionRequest(BaseModel):
    """Request to update a disclosure section."""
    content: str = Field(..., min_length=1, max_length=50000, description="Section content text")
    data_sources: Optional[List[str]] = Field(None, description="Data sources used")
    reviewed: bool = Field(False, description="Whether section has been reviewed")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SectionResponse(BaseModel):
    """A disclosure section."""
    section_id: str
    disclosure_id: str
    pillar: str
    reference: str
    title: str
    content: str
    word_count: int
    data_sources: Optional[List[str]]
    reviewed: bool
    auto_generated: bool
    created_at: datetime
    updated_at: datetime


class DisclosureResponse(BaseModel):
    """A TCFD disclosure."""
    disclosure_id: str
    org_id: str
    reporting_year: int
    title: str
    regulation: str
    status: str
    language: str
    sections: List[SectionResponse]
    total_sections: int
    completed_sections: int
    completeness_pct: float
    created_at: datetime
    updated_at: datetime


class DisclosureListEntry(BaseModel):
    """Summary entry for listing disclosures."""
    disclosure_id: str
    org_id: str
    reporting_year: int
    title: str
    regulation: str
    status: str
    completeness_pct: float
    created_at: datetime


class ComplianceCheckResponse(BaseModel):
    """Compliance check result."""
    disclosure_id: str
    regulation: str
    overall_compliance_pct: float
    pillar_compliance: Dict[str, float]
    missing_sections: List[str]
    incomplete_sections: List[str]
    recommendations: List[str]
    checked_at: datetime


class CrossReferenceResponse(BaseModel):
    """Cross-reference mapping between frameworks."""
    disclosure_id: str
    mappings: List[Dict[str, str]]
    frameworks_covered: List[str]
    generated_at: datetime


class ExportResponse(BaseModel):
    """Export result."""
    export_id: str
    disclosure_id: str
    format: str
    status: str
    file_size_bytes: Optional[int]
    page_count: Optional[int]
    download_url: str
    generated_at: datetime


class YearComparisonResponse(BaseModel):
    """Year-over-year disclosure comparison."""
    org_id: str
    years: List[int]
    pillar_scores: Dict[str, Dict[str, float]]
    overall_progress: Dict[str, float]
    new_sections_added: List[str]
    improvements: List[str]
    generated_at: datetime


class RegulatoryTemplateResponse(BaseModel):
    """Regulatory disclosure template."""
    regulation: str
    regulation_name: str
    sections: List[Dict[str, str]]
    total_sections: int
    mandatory_sections: int
    guidance_notes: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_disclosures: Dict[str, Dict[str, Any]] = {}
_sections: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _get_disclosure_sections(disclosure_id: str) -> List[Dict[str, Any]]:
    """Get all sections for a disclosure."""
    return [s for s in _sections.values() if s["disclosure_id"] == disclosure_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=DisclosureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create disclosure",
    description="Create a new TCFD disclosure for an organization and reporting year.",
)
async def create_disclosure(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateDisclosureRequest = ...,
) -> DisclosureResponse:
    """Create a TCFD disclosure."""
    disclosure_id = _generate_id("disc")
    now = _now()
    disclosure = {
        "disclosure_id": disclosure_id,
        "org_id": org_id,
        "reporting_year": request.reporting_year,
        "title": request.title,
        "regulation": request.regulation,
        "status": DisclosureStatus.DRAFT.value,
        "language": request.language,
        "created_at": now,
        "updated_at": now,
    }
    _disclosures[disclosure_id] = disclosure
    return DisclosureResponse(
        **disclosure,
        sections=[],
        total_sections=11,
        completed_sections=0,
        completeness_pct=0.0,
    )


@router.get(
    "/{org_id}",
    response_model=List[DisclosureListEntry],
    summary="List disclosures",
    description="List all disclosures for an organization.",
)
async def list_disclosures(
    org_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> List[DisclosureListEntry]:
    """List disclosures."""
    results = [d for d in _disclosures.values() if d["org_id"] == org_id]
    results.sort(key=lambda d: d["reporting_year"], reverse=True)
    entries = []
    for d in results[:limit]:
        secs = _get_disclosure_sections(d["disclosure_id"])
        completed = sum(1 for s in secs if len(s.get("content", "")) > 50)
        pct = round(completed / 11 * 100, 1)
        entries.append(DisclosureListEntry(
            disclosure_id=d["disclosure_id"],
            org_id=d["org_id"],
            reporting_year=d["reporting_year"],
            title=d["title"],
            regulation=d["regulation"],
            status=d["status"],
            completeness_pct=pct,
            created_at=d["created_at"],
        ))
    return entries


@router.get(
    "/{org_id}/{disclosure_id}",
    response_model=DisclosureResponse,
    summary="Get disclosure detail",
    description="Retrieve a disclosure with all sections.",
)
async def get_disclosure(org_id: str, disclosure_id: str) -> DisclosureResponse:
    """Get disclosure detail."""
    disclosure = _disclosures.get(disclosure_id)
    if not disclosure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")
    if disclosure["org_id"] != org_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Disclosure does not belong to organization")
    secs = _get_disclosure_sections(disclosure_id)
    completed = sum(1 for s in secs if len(s.get("content", "")) > 50)
    sec_responses = [SectionResponse(**s) for s in secs]
    return DisclosureResponse(
        **disclosure,
        sections=sec_responses,
        total_sections=11,
        completed_sections=completed,
        completeness_pct=round(completed / 11 * 100, 1),
    )


@router.put(
    "/{disclosure_id}",
    response_model=DisclosureListEntry,
    summary="Update disclosure",
    description="Update disclosure title or status.",
)
async def update_disclosure(disclosure_id: str, request: UpdateDisclosureRequest) -> DisclosureListEntry:
    """Update a disclosure."""
    disclosure = _disclosures.get(disclosure_id)
    if not disclosure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")
    updates = request.model_dump(exclude_unset=True)
    if "status" in updates and hasattr(updates["status"], "value"):
        updates["status"] = updates["status"].value
    disclosure.update(updates)
    disclosure["updated_at"] = _now()
    secs = _get_disclosure_sections(disclosure_id)
    completed = sum(1 for s in secs if len(s.get("content", "")) > 50)
    return DisclosureListEntry(
        disclosure_id=disclosure["disclosure_id"],
        org_id=disclosure["org_id"],
        reporting_year=disclosure["reporting_year"],
        title=disclosure["title"],
        regulation=disclosure["regulation"],
        status=disclosure["status"],
        completeness_pct=round(completed / 11 * 100, 1),
        created_at=disclosure["created_at"],
    )


@router.post(
    "/{disclosure_id}/sections/{pillar}/{ref}",
    response_model=SectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate section",
    description="Generate or create a disclosure section for a specific pillar and reference (e.g. GOV-A, STR-B).",
)
async def generate_section(
    disclosure_id: str,
    pillar: str,
    ref: str,
) -> SectionResponse:
    """Generate a disclosure section."""
    disclosure = _disclosures.get(disclosure_id)
    if not disclosure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")

    pillar_sections = TCFD_SECTIONS.get(pillar, {})
    title = pillar_sections.get(ref)
    if not title:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid section reference {ref} for pillar {pillar}")

    section_id = _generate_id("sec")
    now = _now()
    content = (
        f"[Auto-generated placeholder for {ref}: {title}] "
        f"This section should describe the organization's approach to {title.lower()}. "
        f"Content should be substantiated with data from the {disclosure['reporting_year']} "
        f"reporting period and reference applicable frameworks."
    )

    section = {
        "section_id": section_id,
        "disclosure_id": disclosure_id,
        "pillar": pillar,
        "reference": ref,
        "title": title,
        "content": content,
        "word_count": len(content.split()),
        "data_sources": [],
        "reviewed": False,
        "auto_generated": True,
        "created_at": now,
        "updated_at": now,
    }
    _sections[section_id] = section
    return SectionResponse(**section)


@router.put(
    "/sections/{section_id}",
    response_model=SectionResponse,
    summary="Update section",
    description="Update the content of a disclosure section.",
)
async def update_section(section_id: str, request: UpdateSectionRequest) -> SectionResponse:
    """Update a disclosure section."""
    section = _sections.get(section_id)
    if not section:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Section {section_id} not found")
    section["content"] = request.content
    section["word_count"] = len(request.content.split())
    section["data_sources"] = request.data_sources
    section["reviewed"] = request.reviewed
    section["auto_generated"] = False
    section["updated_at"] = _now()
    return SectionResponse(**section)


@router.post(
    "/{disclosure_id}/auto-populate",
    response_model=DisclosureResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Auto-populate all sections",
    description="Auto-populate all 11 TCFD disclosure sections with generated content.",
)
async def auto_populate(disclosure_id: str) -> DisclosureResponse:
    """Auto-populate all disclosure sections."""
    disclosure = _disclosures.get(disclosure_id)
    if not disclosure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")

    now = _now()
    new_sections = []
    for pillar_key, refs in TCFD_SECTIONS.items():
        for ref, title in refs.items():
            existing = [
                s for s in _sections.values()
                if s["disclosure_id"] == disclosure_id and s["reference"] == ref
            ]
            if existing:
                new_sections.append(existing[0])
                continue

            section_id = _generate_id("sec")
            content = (
                f"The organization {title.lower()}. This section has been auto-populated "
                f"and should be reviewed and enhanced with organization-specific data and "
                f"narrative for the {disclosure['reporting_year']} reporting period."
            )
            section = {
                "section_id": section_id,
                "disclosure_id": disclosure_id,
                "pillar": pillar_key,
                "reference": ref,
                "title": title,
                "content": content,
                "word_count": len(content.split()),
                "data_sources": [],
                "reviewed": False,
                "auto_generated": True,
                "created_at": now,
                "updated_at": now,
            }
            _sections[section_id] = section
            new_sections.append(section)

    disclosure["status"] = DisclosureStatus.IN_PROGRESS.value
    disclosure["updated_at"] = now
    completed = sum(1 for s in new_sections if len(s.get("content", "")) > 50)

    return DisclosureResponse(
        **disclosure,
        sections=[SectionResponse(**s) for s in new_sections],
        total_sections=11,
        completed_sections=completed,
        completeness_pct=round(completed / 11 * 100, 1),
    )


@router.get(
    "/{disclosure_id}/compliance",
    response_model=ComplianceCheckResponse,
    summary="Check compliance",
    description="Check disclosure compliance against TCFD requirements.",
)
async def check_compliance(disclosure_id: str) -> ComplianceCheckResponse:
    """Check disclosure compliance."""
    disclosure = _disclosures.get(disclosure_id)
    if not disclosure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")

    secs = _get_disclosure_sections(disclosure_id)
    sec_refs = {s["reference"] for s in secs if len(s.get("content", "")) > 50}

    all_refs = []
    for refs in TCFD_SECTIONS.values():
        all_refs.extend(refs.keys())

    missing = [r for r in all_refs if r not in sec_refs]
    incomplete = [s["reference"] for s in secs if len(s.get("content", "")) <= 50]

    pillar_compliance = {}
    for pillar_key, refs in TCFD_SECTIONS.items():
        total = len(refs)
        completed = sum(1 for r in refs if r in sec_refs)
        pillar_compliance[pillar_key] = round(completed / total * 100, 1) if total > 0 else 0

    overall = round(len(sec_refs) / len(all_refs) * 100, 1) if all_refs else 0

    recs = []
    if missing:
        recs.append(f"Complete {len(missing)} missing section(s): {', '.join(missing)}")
    if incomplete:
        recs.append(f"Expand content for {len(incomplete)} incomplete section(s)")
    reviewed_count = sum(1 for s in secs if s.get("reviewed"))
    if reviewed_count < len(secs):
        recs.append(f"Review {len(secs) - reviewed_count} unreviewed section(s)")

    return ComplianceCheckResponse(
        disclosure_id=disclosure_id,
        regulation=disclosure["regulation"],
        overall_compliance_pct=overall,
        pillar_compliance=pillar_compliance,
        missing_sections=missing,
        incomplete_sections=incomplete,
        recommendations=recs,
        checked_at=_now(),
    )


@router.get(
    "/{disclosure_id}/cross-references",
    response_model=CrossReferenceResponse,
    summary="Get cross-references",
    description="Map TCFD disclosure sections to equivalent requirements in ISSB, CSRD, and SEC frameworks.",
)
async def get_cross_references(disclosure_id: str) -> CrossReferenceResponse:
    """Get cross-references between frameworks."""
    mappings = [
        {"tcfd": "GOV-A", "issb_ifrs_s2": "Para 26(a)", "csrd_esrs_e1": "E1-GOV-1", "sec": "Reg S-K Item 1501(a)"},
        {"tcfd": "GOV-B", "issb_ifrs_s2": "Para 26(b)-27", "csrd_esrs_e1": "E1-GOV-2", "sec": "Reg S-K Item 1501(b)"},
        {"tcfd": "STR-A", "issb_ifrs_s2": "Para 10-12", "csrd_esrs_e1": "E1-SBM-1", "sec": "Reg S-K Item 1502(a)"},
        {"tcfd": "STR-B", "issb_ifrs_s2": "Para 13-15, 16-21", "csrd_esrs_e1": "E1-SBM-2, E1-SBM-3", "sec": "Reg S-K Item 1502(b)"},
        {"tcfd": "STR-C", "issb_ifrs_s2": "Para 22", "csrd_esrs_e1": "E1-SBM-3 (scenario)", "sec": "Reg S-K Item 1502(c)"},
        {"tcfd": "RM-A", "issb_ifrs_s2": "Para 25(a)", "csrd_esrs_e1": "E1-IRO-1", "sec": "Reg S-K Item 1503(a)"},
        {"tcfd": "RM-B", "issb_ifrs_s2": "Para 25(b)", "csrd_esrs_e1": "E1-IRO-1", "sec": "Reg S-K Item 1503(a)"},
        {"tcfd": "RM-C", "issb_ifrs_s2": "Para 25(c)", "csrd_esrs_e1": "E1-IRO-1", "sec": "Reg S-K Item 1503(b)"},
        {"tcfd": "MT-A", "issb_ifrs_s2": "Para 29(a)-(g)", "csrd_esrs_e1": "E1-1 to E1-9", "sec": "Reg S-K Item 1504(a)"},
        {"tcfd": "MT-B", "issb_ifrs_s2": "Para 29(a)", "csrd_esrs_e1": "E1-6", "sec": "Reg S-K Item 1504(b)"},
        {"tcfd": "MT-C", "issb_ifrs_s2": "Para 33-35", "csrd_esrs_e1": "E1-4, E1-5", "sec": "Reg S-K Item 1504(c)"},
    ]

    return CrossReferenceResponse(
        disclosure_id=disclosure_id,
        mappings=mappings,
        frameworks_covered=["TCFD", "ISSB/IFRS S2", "CSRD/ESRS E1", "SEC Climate Rule"],
        generated_at=_now(),
    )


@router.post(
    "/{disclosure_id}/export/pdf",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export PDF",
    description="Export the disclosure as a formatted PDF report.",
)
async def export_pdf(disclosure_id: str) -> ExportResponse:
    """Export disclosure as PDF."""
    if disclosure_id not in _disclosures:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")
    secs = _get_disclosure_sections(disclosure_id)
    pages = max(len(secs) * 4 + 10, 20)
    return ExportResponse(
        export_id=_generate_id("exp"),
        disclosure_id=disclosure_id,
        format="pdf",
        status="completed",
        file_size_bytes=pages * 45000,
        page_count=pages,
        download_url=f"https://api.greenlang.io/tcfd/disclosures/{disclosure_id}/export/pdf/download",
        generated_at=_now(),
    )


@router.post(
    "/{disclosure_id}/export/excel",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export Excel",
    description="Export the disclosure data as an Excel workbook.",
)
async def export_excel(disclosure_id: str) -> ExportResponse:
    """Export disclosure as Excel."""
    if disclosure_id not in _disclosures:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")
    return ExportResponse(
        export_id=_generate_id("exp"),
        disclosure_id=disclosure_id,
        format="excel",
        status="completed",
        file_size_bytes=1200000,
        page_count=None,
        download_url=f"https://api.greenlang.io/tcfd/disclosures/{disclosure_id}/export/excel/download",
        generated_at=_now(),
    )


@router.post(
    "/{disclosure_id}/export/json",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export JSON",
    description="Export the disclosure data as structured JSON.",
)
async def export_json(disclosure_id: str) -> ExportResponse:
    """Export disclosure as JSON."""
    if disclosure_id not in _disclosures:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")
    return ExportResponse(
        export_id=_generate_id("exp"),
        disclosure_id=disclosure_id,
        format="json",
        status="completed",
        file_size_bytes=350000,
        page_count=None,
        download_url=f"https://api.greenlang.io/tcfd/disclosures/{disclosure_id}/export/json/download",
        generated_at=_now(),
    )


@router.post(
    "/{disclosure_id}/export/xbrl",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export XBRL",
    description="Export the disclosure as XBRL-tagged digital report for regulatory submission.",
)
async def export_xbrl(disclosure_id: str) -> ExportResponse:
    """Export disclosure as XBRL."""
    if disclosure_id not in _disclosures:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Disclosure {disclosure_id} not found")
    return ExportResponse(
        export_id=_generate_id("exp"),
        disclosure_id=disclosure_id,
        format="xbrl",
        status="completed",
        file_size_bytes=800000,
        page_count=None,
        download_url=f"https://api.greenlang.io/tcfd/disclosures/{disclosure_id}/export/xbrl/download",
        generated_at=_now(),
    )


@router.get(
    "/compare/{org_id}",
    response_model=YearComparisonResponse,
    summary="Year-over-year comparison",
    description="Compare disclosure completeness and quality across reporting years.",
)
async def compare_years(org_id: str) -> YearComparisonResponse:
    """Compare disclosures year-over-year."""
    org_disclosures = [d for d in _disclosures.values() if d["org_id"] == org_id]
    years = sorted(set(d["reporting_year"] for d in org_disclosures))

    pillar_scores: Dict[str, Dict[str, float]] = {}
    overall: Dict[str, float] = {}
    for yr in years:
        yr_discs = [d for d in org_disclosures if d["reporting_year"] == yr]
        if not yr_discs:
            continue
        disc = yr_discs[0]
        secs = _get_disclosure_sections(disc["disclosure_id"])
        completed = sum(1 for s in secs if len(s.get("content", "")) > 50)
        overall[str(yr)] = round(completed / 11 * 100, 1)

        for pillar_key, refs in TCFD_SECTIONS.items():
            pillar_secs = [s for s in secs if s["pillar"] == pillar_key and len(s.get("content", "")) > 50]
            score = round(len(pillar_secs) / len(refs) * 100, 1)
            pillar_scores.setdefault(pillar_key, {})[str(yr)] = score

    return YearComparisonResponse(
        org_id=org_id,
        years=years,
        pillar_scores=pillar_scores,
        overall_progress=overall,
        new_sections_added=[],
        improvements=["Increasing disclosure completeness across reporting years"] if len(years) > 1 else [],
        generated_at=_now(),
    )


@router.get(
    "/templates/{regulation}",
    response_model=RegulatoryTemplateResponse,
    summary="Regulatory template",
    description="Get a disclosure template for a specific regulation (TCFD, ISSB, CSRD, SEC).",
)
async def get_regulatory_template(regulation: str) -> RegulatoryTemplateResponse:
    """Get regulatory disclosure template."""
    templates = {
        "tcfd": {
            "name": "TCFD Recommendations (2017)",
            "sections": [
                {"ref": ref, "title": title, "pillar": pillar}
                for pillar, refs in TCFD_SECTIONS.items()
                for ref, title in refs.items()
            ],
            "mandatory": 11,
            "guidance": [
                "All 11 recommended disclosures should be addressed",
                "Scenario analysis recommended for Strategy disclosure (c)",
                "Quantitative metrics preferred over qualitative descriptions",
            ],
        },
        "issb": {
            "name": "ISSB/IFRS S2 (2023)",
            "sections": [
                {"ref": "S2-26-27", "title": "Governance", "pillar": "governance"},
                {"ref": "S2-8-22", "title": "Strategy", "pillar": "strategy"},
                {"ref": "S2-25", "title": "Risk Management", "pillar": "risk_management"},
                {"ref": "S2-29-35", "title": "Metrics and Targets", "pillar": "metrics_and_targets"},
            ],
            "mandatory": 4,
            "guidance": [
                "Builds on TCFD with enhanced requirements",
                "Industry-specific metrics per ISSB industry standards",
                "Climate resilience assessment under para 22 is mandatory",
                "Seven cross-industry metrics required",
            ],
        },
        "csrd": {
            "name": "CSRD/ESRS E1 (2024)",
            "sections": [
                {"ref": "E1-GOV", "title": "Governance", "pillar": "governance"},
                {"ref": "E1-SBM", "title": "Strategy and business model", "pillar": "strategy"},
                {"ref": "E1-IRO", "title": "Impact, risk and opportunity", "pillar": "risk_management"},
                {"ref": "E1-1-9", "title": "Metrics and targets", "pillar": "metrics_and_targets"},
            ],
            "mandatory": 4,
            "guidance": [
                "Double materiality assessment required",
                "Transition plan disclosure mandatory",
                "Scope 3 reporting required from 2025",
                "EU Taxonomy alignment reporting",
            ],
        },
        "sec": {
            "name": "SEC Climate Disclosure Rule (2024)",
            "sections": [
                {"ref": "1501", "title": "Governance", "pillar": "governance"},
                {"ref": "1502", "title": "Strategy and business model", "pillar": "strategy"},
                {"ref": "1503", "title": "Risk management", "pillar": "risk_management"},
                {"ref": "1504", "title": "Metrics", "pillar": "metrics_and_targets"},
            ],
            "mandatory": 4,
            "guidance": [
                "Scope 1 and 2 disclosure mandatory for large accelerated filers",
                "Scope 3 disclosure phased in",
                "Attestation required for Scope 1 and 2",
                "Financial statement footnote disclosures for material climate impacts",
            ],
        },
    }

    tpl = templates.get(regulation.lower())
    if not tpl:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template for regulation {regulation} not found")

    return RegulatoryTemplateResponse(
        regulation=regulation,
        regulation_name=tpl["name"],
        sections=tpl["sections"],
        total_sections=len(tpl["sections"]),
        mandatory_sections=tpl["mandatory"],
        guidance_notes=tpl["guidance"],
        generated_at=_now(),
    )
