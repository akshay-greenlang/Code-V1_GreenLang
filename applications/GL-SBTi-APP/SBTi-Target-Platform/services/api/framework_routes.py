"""
GL-SBTi-APP Framework Alignment API

Provides cross-framework alignment between SBTi targets and other
sustainability reporting frameworks (CSRD/ESRS, CDP, TCFD, ISO 14064,
GHG Protocol, SEC Climate).  Maps SBTi criteria to framework-specific
requirements, identifies cross-framework gaps, and generates unified
reports that satisfy multiple frameworks simultaneously.

Supported Frameworks:
    - SBTi v2.1 (primary)
    - CSRD/ESRS E1 (Climate Change)
    - CDP Climate Change questionnaire
    - TCFD recommended disclosures
    - ISO 14064-1 (GHG inventory)
    - GHG Protocol Corporate Standard
    - SEC Climate Disclosure Rule
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/frameworks", tags=["Framework Alignment"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class AlignmentSummaryResponse(BaseModel):
    """Full cross-framework alignment summary."""
    org_id: str
    sbti_status: str
    frameworks: List[Dict[str, Any]]
    overall_alignment_pct: float
    strongest_alignment: str
    weakest_alignment: str
    synergies: List[str]
    generated_at: datetime


class FrameworkMappingResponse(BaseModel):
    """Specific framework mapping to SBTi."""
    org_id: str
    framework: str
    framework_version: str
    total_requirements: int
    mapped_to_sbti: int
    coverage_pct: float
    mappings: List[Dict[str, Any]]
    additional_requirements: List[str]
    generated_at: datetime


class CrossFrameworkGapsResponse(BaseModel):
    """Cross-framework gap identification."""
    org_id: str
    gaps: List[Dict[str, Any]]
    total_gaps: int
    critical_gaps: int
    frameworks_assessed: List[str]
    recommendations: List[str]
    generated_at: datetime


class UnifiedReportRequest(BaseModel):
    """Request to generate a unified cross-framework report."""
    frameworks: List[str] = Field(
        ..., description="Frameworks to include (sbti, csrd, cdp, tcfd, iso14064, ghg_protocol, sec)",
    )
    reporting_year: int = Field(..., ge=2020, le=2055)
    include_gap_analysis: bool = Field(True)


class UnifiedReportResponse(BaseModel):
    """Unified cross-framework report."""
    report_id: str
    org_id: str
    frameworks_included: List[str]
    reporting_year: int
    sections: List[Dict[str, Any]]
    cross_references: List[Dict[str, Any]]
    compliance_summary: Dict[str, float]
    gap_analysis: Optional[Dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

FRAMEWORK_MAPPINGS = {
    "csrd": {
        "name": "CSRD/ESRS E1",
        "version": "ESRS E1 v1.0",
        "mappings": [
            {"sbti_criterion": "C1-C2", "framework_req": "E1-6: GHG reduction targets", "overlap": "high"},
            {"sbti_criterion": "C3", "framework_req": "E1-5: Base year emissions", "overlap": "high"},
            {"sbti_criterion": "C6-C8", "framework_req": "E1-6: 1.5C alignment", "overlap": "high"},
            {"sbti_criterion": "C12", "framework_req": "E1-4: GHG emissions", "overlap": "high"},
            {"sbti_criterion": "C13-C15", "framework_req": "E1-4: Scope 3 categories", "overlap": "high"},
            {"sbti_criterion": "C16-C17", "framework_req": "E1-7: Progress disclosure", "overlap": "medium"},
        ],
        "additional": [
            "Double materiality assessment (ESRS 1)",
            "Transition plan disclosure (E1-1)",
            "Energy consumption and mix (E1-5)",
            "Carbon credits disclosure (E1-7)",
        ],
    },
    "cdp": {
        "name": "CDP Climate Change",
        "version": "CDP 2025",
        "mappings": [
            {"sbti_criterion": "C1-C5", "framework_req": "C4: Targets and performance", "overlap": "high"},
            {"sbti_criterion": "C6-C8", "framework_req": "C4.1: Ambition level", "overlap": "high"},
            {"sbti_criterion": "C12", "framework_req": "C6-C7: GHG emissions", "overlap": "high"},
            {"sbti_criterion": "C13-C15", "framework_req": "C6.5: Scope 3 categories", "overlap": "high"},
            {"sbti_criterion": "C16-C17", "framework_req": "C4.2: Progress tracking", "overlap": "high"},
            {"sbti_criterion": "C20-C23", "framework_req": "C4.2c: Net-zero targets", "overlap": "high"},
        ],
        "additional": [
            "Board-level governance (C1)",
            "Climate-related risks and opportunities (C2-C3)",
            "Business strategy (C3)",
            "Verification (C10)",
            "Carbon pricing (C11)",
        ],
    },
    "tcfd": {
        "name": "TCFD Recommended Disclosures",
        "version": "TCFD v4",
        "mappings": [
            {"sbti_criterion": "C6-C8", "framework_req": "Metrics & Targets: Climate targets", "overlap": "high"},
            {"sbti_criterion": "C12", "framework_req": "Metrics & Targets: GHG emissions", "overlap": "high"},
            {"sbti_criterion": "C16-C17", "framework_req": "Metrics & Targets: Progress", "overlap": "medium"},
        ],
        "additional": [
            "Governance (board oversight, management role)",
            "Strategy (risks, opportunities, scenario analysis)",
            "Risk Management (processes, integration)",
            "Metrics (cross-industry metrics)",
        ],
    },
    "iso14064": {
        "name": "ISO 14064-1",
        "version": "ISO 14064-1:2018",
        "mappings": [
            {"sbti_criterion": "C1-C2", "framework_req": "Clause 5: Organizational boundary", "overlap": "high"},
            {"sbti_criterion": "C3", "framework_req": "Clause 5.2: Base year", "overlap": "high"},
            {"sbti_criterion": "C12", "framework_req": "Clause 5.2-5.4: Quantification", "overlap": "high"},
            {"sbti_criterion": "C13-C15", "framework_req": "Clause 5.2.4: Indirect emissions", "overlap": "high"},
        ],
        "additional": [
            "Uncertainty assessment (Clause 5.3)",
            "GHG management improvements (Clause 5.5)",
            "Verification requirements (Clause 9)",
        ],
    },
    "ghg_protocol": {
        "name": "GHG Protocol Corporate Standard",
        "version": "Revised 2025",
        "mappings": [
            {"sbti_criterion": "C1-C2", "framework_req": "Organizational/operational boundary", "overlap": "high"},
            {"sbti_criterion": "C3", "framework_req": "Base year selection", "overlap": "high"},
            {"sbti_criterion": "C12", "framework_req": "Quantification methodology", "overlap": "high"},
            {"sbti_criterion": "C19", "framework_req": "Base year recalculation", "overlap": "high"},
        ],
        "additional": [
            "Scope 2 dual reporting (location + market)",
            "Reporting principles (relevance, completeness, consistency, transparency, accuracy)",
            "Quality management system",
        ],
    },
    "sec": {
        "name": "SEC Climate Disclosure Rule",
        "version": "SEC S7-10-22",
        "mappings": [
            {"sbti_criterion": "C6-C8", "framework_req": "Climate-related targets disclosure", "overlap": "medium"},
            {"sbti_criterion": "C12", "framework_req": "GHG emissions disclosure (Reg S-X)", "overlap": "high"},
            {"sbti_criterion": "C16-C17", "framework_req": "Annual reporting requirement", "overlap": "medium"},
        ],
        "additional": [
            "Risk factor disclosure",
            "Financial impact disclosure",
            "Governance disclosure",
            "Attestation requirements (Large Accelerated Filers)",
        ],
    },
}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}/alignment",
    response_model=AlignmentSummaryResponse,
    summary="Full alignment summary",
    description=(
        "Get a comprehensive cross-framework alignment summary showing how "
        "SBTi targets map to CSRD, CDP, TCFD, ISO 14064, GHG Protocol, "
        "and SEC requirements."
    ),
)
async def get_alignment_summary(org_id: str) -> AlignmentSummaryResponse:
    """Get cross-framework alignment summary."""
    frameworks = []
    for fid, fdata in FRAMEWORK_MAPPINGS.items():
        coverage = round(len(fdata["mappings"]) / (len(fdata["mappings"]) + len(fdata["additional"])) * 100, 1)
        frameworks.append({
            "framework_id": fid,
            "name": fdata["name"],
            "version": fdata["version"],
            "alignment_pct": coverage,
            "mapped_requirements": len(fdata["mappings"]),
            "additional_requirements": len(fdata["additional"]),
        })

    avg_alignment = round(sum(f["alignment_pct"] for f in frameworks) / len(frameworks), 1)
    strongest = max(frameworks, key=lambda f: f["alignment_pct"])
    weakest = min(frameworks, key=lambda f: f["alignment_pct"])

    return AlignmentSummaryResponse(
        org_id=org_id,
        sbti_status="validated",
        frameworks=frameworks,
        overall_alignment_pct=avg_alignment,
        strongest_alignment=strongest["name"],
        weakest_alignment=weakest["name"],
        synergies=[
            "GHG Protocol inventory directly supports SBTi base year and targets",
            "CDP questionnaire sections C4 and C6-C7 closely align with SBTi submission data",
            "CSRD E1-6 target disclosure mirrors SBTi near-term and long-term targets",
            "ISO 14064-1 quantification methodology satisfies SBTi C12",
        ],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/{framework}/mapping",
    response_model=FrameworkMappingResponse,
    summary="Specific framework mapping",
    description="Get detailed mapping between SBTi criteria and a specific framework.",
)
async def get_framework_mapping(
    org_id: str,
    framework: str,
) -> FrameworkMappingResponse:
    """Get specific framework mapping."""
    fdata = FRAMEWORK_MAPPINGS.get(framework)
    if not fdata:
        valid = list(FRAMEWORK_MAPPINGS.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Framework '{framework}' not found. Valid: {valid}",
        )

    total = len(fdata["mappings"]) + len(fdata["additional"])

    return FrameworkMappingResponse(
        org_id=org_id,
        framework=framework,
        framework_version=fdata["version"],
        total_requirements=total,
        mapped_to_sbti=len(fdata["mappings"]),
        coverage_pct=round(len(fdata["mappings"]) / total * 100, 1) if total > 0 else 0,
        mappings=fdata["mappings"],
        additional_requirements=fdata["additional"],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/gaps",
    response_model=CrossFrameworkGapsResponse,
    summary="Cross-framework gaps",
    description=(
        "Identify requirements that are in other frameworks but not covered "
        "by SBTi. Helps organizations understand what additional work is "
        "needed beyond SBTi target-setting."
    ),
)
async def get_cross_framework_gaps(org_id: str) -> CrossFrameworkGapsResponse:
    """Get cross-framework gaps."""
    gaps = []
    for fid, fdata in FRAMEWORK_MAPPINGS.items():
        for additional in fdata["additional"]:
            gaps.append({
                "framework": fdata["name"],
                "framework_id": fid,
                "requirement": additional,
                "severity": "high" if "governance" in additional.lower() or "risk" in additional.lower() else "medium",
                "sbti_coverage": "not_covered",
            })

    critical = sum(1 for g in gaps if g["severity"] == "high")

    return CrossFrameworkGapsResponse(
        org_id=org_id,
        gaps=gaps,
        total_gaps=len(gaps),
        critical_gaps=critical,
        frameworks_assessed=list(FRAMEWORK_MAPPINGS.keys()),
        recommendations=[
            "Develop TCFD-aligned governance disclosures (board oversight, management role)",
            "Conduct climate scenario analysis for TCFD Strategy and CSRD E1-1",
            "Implement CSRD double materiality assessment",
            "Prepare SEC-compliant financial impact disclosures",
            "Establish ISO 14064-1 compliant uncertainty assessment",
        ],
        generated_at=_now(),
    )


@router.post(
    "/org/{org_id}/unified-report",
    response_model=UnifiedReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Unified cross-framework report",
    description=(
        "Generate a unified report that maps SBTi target data across "
        "multiple frameworks. Reduces reporting burden by identifying "
        "shared data points and cross-references."
    ),
)
async def generate_unified_report(
    org_id: str,
    request: UnifiedReportRequest,
) -> UnifiedReportResponse:
    """Generate unified cross-framework report."""
    report_id = _generate_id("rpt_fw")

    sections = []
    for fw in request.frameworks:
        fdata = FRAMEWORK_MAPPINGS.get(fw)
        if fdata:
            sections.append({
                "framework": fw,
                "name": fdata["name"],
                "sections_populated": len(fdata["mappings"]),
                "sections_needing_additional_data": len(fdata["additional"]),
                "status": "partially_complete",
            })

    cross_refs = [
        {"data_point": "Base year emissions", "frameworks": ["sbti", "ghg_protocol", "iso14064", "cdp", "csrd"], "source": "GHG inventory"},
        {"data_point": "Target reduction percentage", "frameworks": ["sbti", "cdp", "csrd", "tcfd"], "source": "SBTi submission"},
        {"data_point": "Scope 3 categories", "frameworks": ["sbti", "ghg_protocol", "cdp", "csrd"], "source": "Scope 3 screening"},
        {"data_point": "Annual progress", "frameworks": ["sbti", "cdp", "csrd", "sec"], "source": "Progress tracking"},
    ]

    compliance = {}
    for fw in request.frameworks:
        fdata = FRAMEWORK_MAPPINGS.get(fw)
        if fdata:
            total = len(fdata["mappings"]) + len(fdata["additional"])
            compliance[fw] = round(len(fdata["mappings"]) / total * 100, 1) if total > 0 else 0

    gap_analysis = None
    if request.include_gap_analysis:
        gap_analysis = {
            "total_unique_requirements": 45,
            "met_by_sbti": 28,
            "requiring_additional_work": 17,
            "estimated_effort_weeks": 12,
        }

    return UnifiedReportResponse(
        report_id=report_id,
        org_id=org_id,
        frameworks_included=request.frameworks,
        reporting_year=request.reporting_year,
        sections=sections,
        cross_references=cross_refs,
        compliance_summary=compliance,
        gap_analysis=gap_analysis,
        generated_at=_now(),
    )
