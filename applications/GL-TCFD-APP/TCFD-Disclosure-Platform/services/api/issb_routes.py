"""
GL-TCFD-APP ISSB Cross-walk API

Manages the mapping between TCFD recommended disclosures and ISSB/IFRS S2
requirements, dual compliance scoring, gap identification for IFRS S2
additional requirements, industry-specific metrics guidance, and migration
pathway from TCFD to IFRS S2.

IFRS S2 Key Additions over TCFD:
    - Industry-based metrics (per ISSB industry standards)
    - Seven cross-industry metrics (mandatory)
    - Climate resilience assessment (para 22)
    - Current and anticipated financial effects (para 16-21)
    - Scope 3 transition plan specifics
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/v1/tcfd/issb", tags=["ISSB Cross-walk"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ISSBMappingEntry(BaseModel):
    """A single TCFD to IFRS S2 mapping entry."""
    tcfd_reference: str
    tcfd_description: str
    ifrs_s2_reference: str
    ifrs_s2_description: str
    alignment: str
    additional_requirements: Optional[str]


class ISSBMappingResponse(BaseModel):
    """Full TCFD to IFRS S2 mapping."""
    total_mappings: int
    fully_aligned: int
    partially_aligned: int
    additional_in_ifrs_s2: int
    mappings: List[ISSBMappingEntry]
    generated_at: datetime


class DualComplianceResponse(BaseModel):
    """Dual compliance scoring."""
    org_id: str
    tcfd_compliance_pct: float
    ifrs_s2_compliance_pct: float
    combined_score: float
    tcfd_gaps: List[str]
    ifrs_s2_gaps: List[str]
    shared_coverage: List[str]
    recommendations: List[str]
    generated_at: datetime


class ISSBGapsResponse(BaseModel):
    """ISSB/IFRS S2 gap identification."""
    org_id: str
    total_gaps: int
    critical_gaps: int
    gaps: List[Dict[str, Any]]
    effort_estimate_months: int
    generated_at: datetime


class AdditionalRequirementsResponse(BaseModel):
    """IFRS S2 additional requirements beyond TCFD."""
    org_id: str
    requirements: List[Dict[str, Any]]
    total_additional: int
    met: int
    not_met: int
    generated_at: datetime


class IndustryMetricsRequirementsResponse(BaseModel):
    """Industry-based metrics requirements from ISSB."""
    industry: str
    standard_reference: str
    metrics: List[Dict[str, Any]]
    total_metrics: int
    mandatory_metrics: int
    guidance_notes: List[str]
    generated_at: datetime


class MigrationPathwayResponse(BaseModel):
    """Migration pathway from TCFD to IFRS S2."""
    org_id: str
    current_tcfd_coverage_pct: float
    ifrs_s2_readiness_pct: float
    migration_steps: List[Dict[str, Any]]
    total_steps: int
    completed_steps: int
    estimated_months: int
    key_additions_needed: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Static Mapping Data
# ---------------------------------------------------------------------------

TCFD_TO_IFRS_S2_MAPPING = [
    {
        "tcfd_reference": "GOV-A",
        "tcfd_description": "Board oversight of climate-related risks and opportunities",
        "ifrs_s2_reference": "Para 26(a)",
        "ifrs_s2_description": "Governance body responsible for oversight of climate-related risks and opportunities",
        "alignment": "fully_aligned",
        "additional_requirements": None,
    },
    {
        "tcfd_reference": "GOV-B",
        "tcfd_description": "Management's role in assessing and managing climate-related risks and opportunities",
        "ifrs_s2_reference": "Para 26(b), 27",
        "ifrs_s2_description": "Management's role, including skills and competencies",
        "alignment": "partially_aligned",
        "additional_requirements": "IFRS S2 requires disclosure of management skills and competencies for climate",
    },
    {
        "tcfd_reference": "STR-A",
        "tcfd_description": "Climate-related risks and opportunities over short, medium, long term",
        "ifrs_s2_reference": "Para 10-12",
        "ifrs_s2_description": "Climate-related risks and opportunities with time horizon specification",
        "alignment": "fully_aligned",
        "additional_requirements": None,
    },
    {
        "tcfd_reference": "STR-B",
        "tcfd_description": "Impact on business, strategy, and financial planning",
        "ifrs_s2_reference": "Para 13-15, 16-21",
        "ifrs_s2_description": "Effects on business model, strategy, and current/anticipated financial effects",
        "alignment": "partially_aligned",
        "additional_requirements": "IFRS S2 requires quantified current and anticipated financial effects (para 16-21)",
    },
    {
        "tcfd_reference": "STR-C",
        "tcfd_description": "Resilience of strategy under different climate scenarios",
        "ifrs_s2_reference": "Para 22",
        "ifrs_s2_description": "Climate resilience assessment with scenario analysis",
        "alignment": "fully_aligned",
        "additional_requirements": "IFRS S2 makes scenario analysis mandatory and requires disclosure of analytical approach",
    },
    {
        "tcfd_reference": "RM-A",
        "tcfd_description": "Processes for identifying and assessing climate-related risks",
        "ifrs_s2_reference": "Para 25(a)",
        "ifrs_s2_description": "Processes for identifying, assessing, prioritizing, and monitoring climate risks",
        "alignment": "fully_aligned",
        "additional_requirements": None,
    },
    {
        "tcfd_reference": "RM-B",
        "tcfd_description": "Processes for managing climate-related risks",
        "ifrs_s2_reference": "Para 25(b)",
        "ifrs_s2_description": "Processes for managing climate risks including mitigation, transfer, acceptance",
        "alignment": "fully_aligned",
        "additional_requirements": None,
    },
    {
        "tcfd_reference": "RM-C",
        "tcfd_description": "Integration into overall risk management",
        "ifrs_s2_reference": "Para 25(c)",
        "ifrs_s2_description": "Integration of climate risk processes into overall risk management",
        "alignment": "fully_aligned",
        "additional_requirements": None,
    },
    {
        "tcfd_reference": "MT-A",
        "tcfd_description": "Metrics used to assess climate-related risks and opportunities",
        "ifrs_s2_reference": "Para 29(a)-(g)",
        "ifrs_s2_description": "Seven cross-industry climate-related metrics",
        "alignment": "partially_aligned",
        "additional_requirements": "IFRS S2 specifies 7 mandatory cross-industry metrics and industry-specific metrics",
    },
    {
        "tcfd_reference": "MT-B",
        "tcfd_description": "Scope 1, 2, 3 GHG emissions",
        "ifrs_s2_reference": "Para 29(a)",
        "ifrs_s2_description": "Absolute Scope 1, 2, 3 GHG emissions measured in accordance with GHG Protocol",
        "alignment": "fully_aligned",
        "additional_requirements": "IFRS S2 requires Scope 3 for all material categories (not optional)",
    },
    {
        "tcfd_reference": "MT-C",
        "tcfd_description": "Targets used to manage climate-related risks and opportunities",
        "ifrs_s2_reference": "Para 33-35",
        "ifrs_s2_description": "Climate-related targets including metrics, base period, milestones, and progress",
        "alignment": "fully_aligned",
        "additional_requirements": "IFRS S2 requires interim milestones and annual progress reporting",
    },
]


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/mapping",
    response_model=ISSBMappingResponse,
    summary="Full TCFD to IFRS S2 mapping",
    description=(
        "Get the complete mapping between TCFD 11 recommended disclosures and "
        "ISSB/IFRS S2 requirements showing alignment status and additional "
        "requirements in IFRS S2."
    ),
)
async def get_full_mapping() -> ISSBMappingResponse:
    """Get full TCFD to IFRS S2 mapping."""
    fully = sum(1 for m in TCFD_TO_IFRS_S2_MAPPING if m["alignment"] == "fully_aligned")
    partial = sum(1 for m in TCFD_TO_IFRS_S2_MAPPING if m["alignment"] == "partially_aligned")
    additional = sum(1 for m in TCFD_TO_IFRS_S2_MAPPING if m["additional_requirements"])

    return ISSBMappingResponse(
        total_mappings=len(TCFD_TO_IFRS_S2_MAPPING),
        fully_aligned=fully,
        partially_aligned=partial,
        additional_in_ifrs_s2=additional,
        mappings=[ISSBMappingEntry(**m) for m in TCFD_TO_IFRS_S2_MAPPING],
        generated_at=_now(),
    )


@router.get(
    "/compliance/{org_id}",
    response_model=DualComplianceResponse,
    summary="Dual compliance score",
    description="Calculate dual compliance scores for both TCFD and IFRS S2.",
)
async def get_dual_compliance(org_id: str) -> DualComplianceResponse:
    """Calculate dual compliance scores."""
    # Simulated compliance assessment
    tcfd_pct = 72.7
    ifrs_s2_pct = 58.0
    combined = round((tcfd_pct + ifrs_s2_pct) / 2, 1)

    tcfd_gaps = ["STR-C: Climate resilience (scenario analysis incomplete)", "MT-C: Target progress tracking"]
    ifrs_s2_gaps = [
        "Para 16-21: Quantified financial effects not reported",
        "Para 22: Formal climate resilience assessment missing",
        "Para 29(b): Transition risk amount/percentage not disclosed",
        "Para 29(c): Physical risk amount/percentage not disclosed",
        "Industry-specific metrics not reported",
    ]
    shared = [
        "GOV-A/Para 26(a): Board oversight - Covered",
        "MT-B/Para 29(a): GHG emissions Scope 1+2 - Covered",
        "RM-A/Para 25(a): Risk identification process - Covered",
    ]

    return DualComplianceResponse(
        org_id=org_id,
        tcfd_compliance_pct=tcfd_pct,
        ifrs_s2_compliance_pct=ifrs_s2_pct,
        combined_score=combined,
        tcfd_gaps=tcfd_gaps,
        ifrs_s2_gaps=ifrs_s2_gaps,
        shared_coverage=shared,
        recommendations=[
            "Quantify current and anticipated financial effects per IFRS S2 para 16-21",
            "Conduct formal scenario-based climate resilience assessment",
            "Report all 7 ISSB cross-industry metrics",
            "Add industry-specific metrics per ISSB industry guidance",
            "Ensure Scope 3 covers all material categories",
        ],
        generated_at=_now(),
    )


@router.get(
    "/gaps/{org_id}",
    response_model=ISSBGapsResponse,
    summary="ISSB gap identification",
    description="Identify specific gaps between current TCFD compliance and IFRS S2 requirements.",
)
async def get_issb_gaps(org_id: str) -> ISSBGapsResponse:
    """Identify ISSB/IFRS S2 gaps."""
    gaps = [
        {"reference": "Para 16-21", "requirement": "Current and anticipated financial effects", "status": "not_met", "severity": "critical",
         "remediation": "Build financial impact model linking climate risks to income statement, balance sheet, and cash flow"},
        {"reference": "Para 22", "requirement": "Climate resilience assessment via scenario analysis", "status": "partial", "severity": "critical",
         "remediation": "Complete scenario analysis under at least 2 scenarios including 1.5C pathway"},
        {"reference": "Para 27", "requirement": "Management skills and competencies for climate", "status": "not_met", "severity": "medium",
         "remediation": "Disclose management climate expertise and training programs"},
        {"reference": "Para 29(b)-(c)", "requirement": "Transition/physical risk amounts as % of assets", "status": "not_met", "severity": "critical",
         "remediation": "Quantify and disclose climate risk exposure as percentage of total assets"},
        {"reference": "Para 29(d)", "requirement": "Climate opportunity amounts as % of revenue", "status": "not_met", "severity": "medium",
         "remediation": "Quantify and disclose climate opportunity value as percentage of revenue"},
        {"reference": "Para 29(e)", "requirement": "Capital deployment to climate-related activities", "status": "partial", "severity": "medium",
         "remediation": "Track and disclose capex/opex allocated to climate transition"},
        {"reference": "Para 29(f)", "requirement": "Internal carbon price", "status": "not_met", "severity": "medium",
         "remediation": "Implement and disclose internal carbon pricing mechanism"},
        {"reference": "Para 29(g)", "requirement": "Remuneration linked to climate", "status": "partial", "severity": "medium",
         "remediation": "Disclose percentage of executive remuneration linked to climate KPIs"},
        {"reference": "Industry", "requirement": "Industry-specific metrics per ISSB standards", "status": "not_met", "severity": "critical",
         "remediation": "Identify and report applicable industry-specific metrics"},
    ]

    critical = sum(1 for g in gaps if g["severity"] == "critical")
    months = max(6, critical * 2 + len(gaps))

    return ISSBGapsResponse(
        org_id=org_id,
        total_gaps=len(gaps),
        critical_gaps=critical,
        gaps=gaps,
        effort_estimate_months=months,
        generated_at=_now(),
    )


@router.get(
    "/additional-requirements/{org_id}",
    response_model=AdditionalRequirementsResponse,
    summary="IFRS S2 additional requirements",
    description="List IFRS S2 requirements that go beyond TCFD recommendations.",
)
async def get_additional_requirements(org_id: str) -> AdditionalRequirementsResponse:
    """List additional IFRS S2 requirements."""
    requirements = [
        {"requirement": "Seven cross-industry metrics (mandatory)", "ifrs_s2_ref": "Para 29(a)-(g)", "met": False, "effort": "high"},
        {"requirement": "Industry-specific metrics", "ifrs_s2_ref": "Industry standards", "met": False, "effort": "high"},
        {"requirement": "Quantified current financial effects", "ifrs_s2_ref": "Para 16-17", "met": False, "effort": "high"},
        {"requirement": "Quantified anticipated financial effects", "ifrs_s2_ref": "Para 18-21", "met": False, "effort": "high"},
        {"requirement": "Climate resilience assessment (mandatory)", "ifrs_s2_ref": "Para 22", "met": True, "effort": "high"},
        {"requirement": "Scope 3 all material categories (not optional)", "ifrs_s2_ref": "Para 29(a)", "met": True, "effort": "medium"},
        {"requirement": "Management competencies disclosure", "ifrs_s2_ref": "Para 27", "met": False, "effort": "low"},
        {"requirement": "Interim target milestones", "ifrs_s2_ref": "Para 33-35", "met": False, "effort": "medium"},
        {"requirement": "Transition plan specifics", "ifrs_s2_ref": "Para 14", "met": False, "effort": "high"},
    ]

    met_count = sum(1 for r in requirements if r["met"])
    not_met = len(requirements) - met_count

    return AdditionalRequirementsResponse(
        org_id=org_id,
        requirements=requirements,
        total_additional=len(requirements),
        met=met_count,
        not_met=not_met,
        generated_at=_now(),
    )


@router.get(
    "/industry-metrics/{industry}",
    response_model=IndustryMetricsRequirementsResponse,
    summary="Industry-based metrics requirements",
    description="Get ISSB industry-based metrics requirements for a specific industry.",
)
async def get_industry_metrics_requirements(industry: str) -> IndustryMetricsRequirementsResponse:
    """Get industry-based metrics requirements."""
    industry_data = {
        "energy": {
            "standard": "IFRS S2 B37-B45 (Energy)",
            "metrics": [
                {"metric": "Gross Scope 1 CO2 emissions from power generation", "unit": "tCO2e", "mandatory": True},
                {"metric": "Methane emissions (Scope 1)", "unit": "tCO2e", "mandatory": True},
                {"metric": "Power generation from renewables (%)", "unit": "%", "mandatory": True},
                {"metric": "Installed renewable generation capacity", "unit": "MW", "mandatory": True},
                {"metric": "Reserves CO2 emissions if burned", "unit": "GtCO2", "mandatory": True},
            ],
        },
        "financial": {
            "standard": "IFRS S2 B46-B55 (Financial Institutions)",
            "metrics": [
                {"metric": "Financed emissions (PCAF)", "unit": "tCO2e", "mandatory": True},
                {"metric": "Weighted Average Carbon Intensity (WACI)", "unit": "tCO2e/$M", "mandatory": True},
                {"metric": "Green asset ratio", "unit": "%", "mandatory": False},
                {"metric": "Climate VaR", "unit": "$", "mandatory": False},
            ],
        },
        "manufacturing": {
            "standard": "IFRS S2 Industry Guidance (Manufacturing)",
            "metrics": [
                {"metric": "Process emissions", "unit": "tCO2e", "mandatory": True},
                {"metric": "Energy intensity per unit produced", "unit": "MWh/unit", "mandatory": True},
                {"metric": "Waste recycled (%)", "unit": "%", "mandatory": False},
                {"metric": "Water withdrawal in water-stressed areas", "unit": "ML", "mandatory": False},
            ],
        },
    }

    data = industry_data.get(industry.lower(), {
        "standard": f"IFRS S2 Industry Guidance ({industry.title()})",
        "metrics": [
            {"metric": "GHG emissions intensity", "unit": "tCO2e/unit", "mandatory": True},
            {"metric": "Energy consumption", "unit": "MWh", "mandatory": True},
        ],
    })

    metrics = data["metrics"]
    mandatory = sum(1 for m in metrics if m.get("mandatory"))

    return IndustryMetricsRequirementsResponse(
        industry=industry,
        standard_reference=data["standard"],
        metrics=metrics,
        total_metrics=len(metrics),
        mandatory_metrics=mandatory,
        guidance_notes=[
            "Industry-specific metrics supplement the 7 cross-industry metrics",
            "Mandatory metrics must be reported; recommended metrics encouraged",
            "Entities in multiple industries should report metrics for all applicable industries",
        ],
        generated_at=_now(),
    )


@router.get(
    "/migration/{org_id}",
    response_model=MigrationPathwayResponse,
    summary="Migration pathway",
    description="Generate a migration pathway from TCFD-based reporting to full IFRS S2 compliance.",
)
async def get_migration_pathway(org_id: str) -> MigrationPathwayResponse:
    """Generate TCFD to IFRS S2 migration pathway."""
    steps = [
        {"step": 1, "action": "Map existing TCFD disclosures to IFRS S2 requirements", "status": "completed", "months": 1},
        {"step": 2, "action": "Identify IFRS S2 additional requirements (gaps)", "status": "completed", "months": 1},
        {"step": 3, "action": "Build financial impact quantification model (para 16-21)", "status": "in_progress", "months": 4},
        {"step": 4, "action": "Enhance scenario analysis for climate resilience (para 22)", "status": "in_progress", "months": 3},
        {"step": 5, "action": "Implement 7 cross-industry metrics reporting", "status": "not_started", "months": 3},
        {"step": 6, "action": "Add industry-specific metrics", "status": "not_started", "months": 3},
        {"step": 7, "action": "Enhance Scope 3 to cover all material categories", "status": "not_started", "months": 4},
        {"step": 8, "action": "Implement transition plan disclosures (para 14)", "status": "not_started", "months": 3},
        {"step": 9, "action": "Set interim target milestones and track progress", "status": "not_started", "months": 2},
        {"step": 10, "action": "Integrate IFRS S2 into annual report", "status": "not_started", "months": 2},
    ]

    completed = sum(1 for s in steps if s["status"] == "completed")
    tcfd_coverage = 72.7
    ifrs_s2_readiness = round(completed / len(steps) * 100, 1)
    total_months = max(s["months"] for s in steps) + 6  # Sequential overlap

    key_additions = [
        "Quantified financial effects (income statement, balance sheet, cash flow)",
        "Mandatory scenario-based climate resilience assessment",
        "7 cross-industry metrics (transition risk %, physical risk %, etc.)",
        "Industry-specific metrics per ISSB standards",
        "Transition plan specifics with milestones",
    ]

    return MigrationPathwayResponse(
        org_id=org_id,
        current_tcfd_coverage_pct=tcfd_coverage,
        ifrs_s2_readiness_pct=ifrs_s2_readiness,
        migration_steps=steps,
        total_steps=len(steps),
        completed_steps=completed,
        estimated_months=total_months,
        key_additions_needed=key_additions,
        generated_at=_now(),
    )
