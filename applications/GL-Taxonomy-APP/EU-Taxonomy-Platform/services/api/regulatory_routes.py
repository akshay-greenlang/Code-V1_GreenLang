"""
GL-Taxonomy-APP Regulatory Tracking API

Tracks EU Taxonomy regulatory developments including Delegated Act
versions, Technical Screening Criteria (TSC) updates, Omnibus
simplification impacts, and provides version-aware criteria lookups.

Regulatory Instruments:
    - Climate Delegated Act (CDA) -- 2021/2139
    - Complementary CDA -- 2022/1214 (nuclear, gas)
    - Environmental Delegated Act (EDA) -- 2023/2486 (WTR, CE, PPC, BIO)
    - Article 8 Delegated Act -- 2021/2178 (disclosure)
    - Omnibus Proposal -- 2024 (simplification for SMEs, thresholds)
    - EBA ITS on ESG Risks (Pillar III GAR disclosure)

TSC Update Tracking:
    The EU Platform on Sustainable Finance reviews TSC and proposes
    amendments.  This router tracks version history and applicable
    dates so organizations always use the correct criteria version.
"""

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/v1/taxonomy/regulatory", tags=["Regulatory Tracking"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class DelegatedActResponse(BaseModel):
    """Delegated act information."""
    act_id: str
    name: str
    regulation_number: str
    adoption_date: str
    application_date: str
    scope: str
    objectives_covered: List[str]
    status: str
    amendments: List[Dict[str, str]]
    key_thresholds: List[str]


class TSCUpdateResponse(BaseModel):
    """TSC update timeline entry."""
    update_id: str
    title: str
    description: str
    affected_activities: List[str]
    effective_date: str
    published_date: str
    status: str
    impact_level: str
    source: str


class OmnibusImpactResponse(BaseModel):
    """Omnibus simplification impact assessment."""
    org_id: str
    applies_to_org: bool
    org_size: str
    employee_count_threshold: int
    turnover_threshold_eur: float
    impacts: List[Dict[str, Any]]
    simplified_kpis: bool
    reduced_activity_detail: bool
    voluntary_reporting_option: bool
    recommendations: List[str]
    generated_at: datetime


class ApplicableVersionResponse(BaseModel):
    """Applicable DA version for an activity."""
    activity_code: str
    activity_name: str
    applicable_da: str
    da_version: str
    effective_from: str
    next_review: Optional[str]
    tsc_version: str
    amendments_applied: List[str]
    generated_at: datetime


class TransitionPlanResponse(BaseModel):
    """Regulatory transition plan."""
    org_id: str
    current_framework: str
    upcoming_changes: List[Dict[str, Any]]
    action_items: List[Dict[str, Any]]
    next_reporting_deadline: str
    prepared_pct: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

DELEGATED_ACTS = [
    {
        "act_id": "cda_2021",
        "name": "Climate Delegated Act",
        "regulation_number": "(EU) 2021/2139",
        "adoption_date": "2021-06-04",
        "application_date": "2022-01-01",
        "scope": "Technical screening criteria for CCM and CCA",
        "objectives_covered": ["climate_change_mitigation", "climate_change_adaptation"],
        "status": "in_force",
        "amendments": [
            {"amendment": "Corrigendum C/2022/631", "date": "2022-03-15"},
            {"amendment": "Amendment (EU) 2023/2485", "date": "2023-11-21"},
        ],
        "key_thresholds": [
            "100 gCO2e/kWh for electricity generation",
            "270 gCO2e/kWh for transitional gas generation",
            "10% below NZEB for new buildings",
            "30% primary energy reduction for renovations",
        ],
    },
    {
        "act_id": "comp_cda_2022",
        "name": "Complementary Climate Delegated Act",
        "regulation_number": "(EU) 2022/1214",
        "adoption_date": "2022-03-09",
        "application_date": "2023-01-01",
        "scope": "Nuclear energy and fossil gas transitional activities",
        "objectives_covered": ["climate_change_mitigation"],
        "status": "in_force",
        "amendments": [],
        "key_thresholds": [
            "270 gCO2e/kWh for gas generation (direct emissions)",
            "Nuclear: operational waste management plan",
            "Gas: switch to renewable/low-carbon by 2035",
        ],
    },
    {
        "act_id": "eda_2023",
        "name": "Environmental Delegated Act",
        "regulation_number": "(EU) 2023/2486",
        "adoption_date": "2023-06-27",
        "application_date": "2024-01-01",
        "scope": "TSC for four non-climate environmental objectives",
        "objectives_covered": ["water", "circular_economy", "pollution_prevention", "biodiversity"],
        "status": "in_force",
        "amendments": [],
        "key_thresholds": [
            "Water: WFD compliance + water use efficiency plan",
            "CE: Waste hierarchy + recyclability standards",
            "PPC: IED BAT + REACH + RoHS compliance",
            "BIO: EIA + Natura 2000 + no deforestation",
        ],
    },
    {
        "act_id": "art8_da_2021",
        "name": "Article 8 Delegated Act",
        "regulation_number": "(EU) 2021/2178",
        "adoption_date": "2021-07-06",
        "application_date": "2022-01-01",
        "scope": "Disclosure obligations for Article 8",
        "objectives_covered": ["all"],
        "status": "in_force",
        "amendments": [
            {"amendment": "Amendment for EDA activities", "date": "2023-12-01"},
        ],
        "key_thresholds": [
            "Turnover KPI mandatory",
            "CapEx KPI mandatory",
            "OpEx KPI mandatory",
            "Activity-level breakdown required",
        ],
    },
    {
        "act_id": "eba_its_2022",
        "name": "EBA ITS on ESG Risk Disclosures",
        "regulation_number": "EBA/ITS/2022/01",
        "adoption_date": "2022-01-24",
        "application_date": "2022-06-28",
        "scope": "Pillar III ESG disclosure for credit institutions",
        "objectives_covered": ["all"],
        "status": "in_force",
        "amendments": [
            {"amendment": "Revision for GAR/BTAR", "date": "2024-01-01"},
        ],
        "key_thresholds": [
            "GAR stock and flow mandatory",
            "BTAR for extended scope",
            "Template 0-10 disclosure",
        ],
    },
]

TSC_UPDATES = [
    {"update_id": "upd_001", "title": "Environmental Delegated Act enters into force", "description": "TSC for WTR, CE, PPC, BIO objectives become applicable.", "affected_activities": ["All new EDA activities"], "effective_date": "2024-01-01", "published_date": "2023-06-27", "status": "in_force", "impact_level": "high", "source": "Official Journal EU"},
    {"update_id": "upd_002", "title": "Omnibus Proposal tabled", "description": "Simplification of reporting for SMEs and reduced activity detail.", "affected_activities": ["SME-related activities"], "effective_date": "2026-01-01", "published_date": "2024-02-26", "status": "proposed", "impact_level": "high", "source": "European Commission"},
    {"update_id": "upd_003", "title": "CDA Amendment for manufacturing thresholds", "description": "Updated GHG intensity thresholds for cement, steel, aluminium.", "affected_activities": ["3.9", "3.12", "3.14"], "effective_date": "2025-01-01", "published_date": "2024-06-15", "status": "in_force", "impact_level": "medium", "source": "Commission Delegated Regulation (EU) 2023/2485"},
    {"update_id": "upd_004", "title": "EBA GAR revision", "description": "Updated GAR templates and BTAR methodology.", "affected_activities": ["Financial institutions"], "effective_date": "2025-06-28", "published_date": "2024-12-01", "status": "in_force", "impact_level": "medium", "source": "EBA Final Report"},
    {"update_id": "upd_005", "title": "Platform on Sustainable Finance TSC review", "description": "Proposed revisions to transport and building TSC thresholds.", "affected_activities": ["6.5", "6.6", "7.1", "7.2", "7.7"], "effective_date": "2027-01-01", "published_date": "2025-03-01", "status": "under_review", "impact_level": "medium", "source": "Platform on Sustainable Finance"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/delegated-acts",
    response_model=List[DelegatedActResponse],
    summary="List delegated acts and versions",
    description="List all EU Taxonomy delegated acts with versions, dates, and key thresholds.",
)
async def list_delegated_acts(
    act_status: Optional[str] = Query(None, description="Filter by status (in_force, proposed)"),
) -> List[DelegatedActResponse]:
    """List delegated acts."""
    results = DELEGATED_ACTS
    if act_status:
        results = [a for a in results if a["status"] == act_status]
    return [DelegatedActResponse(**a) for a in results]


@router.get(
    "/updates",
    response_model=List[TSCUpdateResponse],
    summary="TSC updates timeline",
    description="Get timeline of TSC updates, amendments, and proposed changes.",
)
async def list_updates(
    impact_level: Optional[str] = Query(None, description="Filter by impact level"),
    update_status: Optional[str] = Query(None, description="Filter by status"),
) -> List[TSCUpdateResponse]:
    """Get TSC updates timeline."""
    results = TSC_UPDATES
    if impact_level:
        results = [u for u in results if u["impact_level"] == impact_level]
    if update_status:
        results = [u for u in results if u["status"] == update_status]
    return [TSCUpdateResponse(**u) for u in results]


@router.get(
    "/{org_id}/omnibus-impact",
    response_model=OmnibusImpactResponse,
    summary="Assess Omnibus impact",
    description=(
        "Assess the impact of the EU Omnibus simplification proposal on "
        "the organization's taxonomy reporting obligations."
    ),
)
async def assess_omnibus_impact(
    org_id: str,
    employee_count: int = Query(500, ge=1, description="Number of employees"),
    annual_turnover_eur: float = Query(100000000, ge=0, description="Annual turnover in EUR"),
) -> OmnibusImpactResponse:
    """Assess Omnibus impact."""
    # Omnibus thresholds (proposed)
    sme_employee_threshold = 250
    sme_turnover_threshold = 50000000
    large_employee_threshold = 1000
    large_turnover_threshold = 450000000

    if employee_count <= sme_employee_threshold and annual_turnover_eur <= sme_turnover_threshold:
        org_size = "sme"
        applies = True
        simplified = True
        voluntary = True
        impacts = [
            {"impact": "Voluntary taxonomy reporting (not mandatory)", "effect": "positive", "action_required": False},
            {"impact": "Simplified KPI templates available", "effect": "positive", "action_required": False},
            {"impact": "Reduced activity-level detail", "effect": "positive", "action_required": False},
        ]
        recommendations = [
            "Consider voluntary reporting for green financing access",
            "Use simplified templates if choosing to report",
            "Monitor Omnibus adoption timeline (expected 2026)",
        ]
    elif employee_count <= large_employee_threshold:
        org_size = "mid_cap"
        applies = True
        simplified = True
        voluntary = False
        impacts = [
            {"impact": "Simplified KPI tables available", "effect": "positive", "action_required": True},
            {"impact": "Phased-in activity detail (3-year transition)", "effect": "neutral", "action_required": True},
            {"impact": "DNSH assessment may be simplified", "effect": "positive", "action_required": False},
        ]
        recommendations = [
            "Prepare for simplified reporting templates",
            "Maintain full data collection during transition",
            "Monitor final Omnibus text for threshold confirmation",
        ]
    else:
        org_size = "large"
        applies = False
        simplified = False
        voluntary = False
        impacts = [
            {"impact": "Full Article 8 reporting continues", "effect": "neutral", "action_required": False},
            {"impact": "Enhanced GAR/BTAR reporting for FIs", "effect": "negative", "action_required": True},
            {"impact": "Expanded EDA activity coverage", "effect": "neutral", "action_required": True},
        ]
        recommendations = [
            "Continue full taxonomy alignment assessment",
            "Prepare for expanded EDA activity coverage",
            "Update systems for revised templates",
        ]

    return OmnibusImpactResponse(
        org_id=org_id, applies_to_org=applies, org_size=org_size,
        employee_count_threshold=sme_employee_threshold,
        turnover_threshold_eur=sme_turnover_threshold,
        impacts=impacts, simplified_kpis=simplified,
        reduced_activity_detail=simplified, voluntary_reporting_option=voluntary,
        recommendations=recommendations, generated_at=_now(),
    )


@router.get(
    "/{activity_code}/applicable-version",
    response_model=ApplicableVersionResponse,
    summary="Get applicable DA version",
    description="Get the applicable Delegated Act version and TSC for an activity.",
)
async def get_applicable_version(
    activity_code: str,
    reporting_date: str = Query("2025-12-31", description="Reporting date to determine applicable version"),
) -> ApplicableVersionResponse:
    """Get applicable DA version for an activity."""
    # Determine DA based on activity code prefix
    code_prefix = activity_code.split(".")[0]
    activity_names = {
        "1": "Forestry activity", "2": "Environmental protection activity",
        "3": "Manufacturing activity", "4": "Energy activity",
        "5": "Water/Waste activity", "6": "Transport activity",
        "7": "Construction/Real Estate activity", "8": "ICT activity",
        "9": "Professional/Scientific activity", "10": "Financial/Insurance activity",
    }
    activity_name = activity_names.get(code_prefix, f"Activity {activity_code}")

    # Determine applicable DA
    if activity_code == "4.29":
        da = "Complementary Climate Delegated Act"
        version = "(EU) 2022/1214"
        effective = "2023-01-01"
        tsc_version = "v1.0"
        amendments = []
    elif code_prefix in ("1", "3", "4", "5", "6", "7", "8", "9"):
        da = "Climate Delegated Act"
        version = "(EU) 2021/2139"
        effective = "2022-01-01"
        tsc_version = "v2.0 (amended 2023)"
        amendments = ["(EU) 2023/2485"]
    elif code_prefix == "2":
        da = "Environmental Delegated Act"
        version = "(EU) 2023/2486"
        effective = "2024-01-01"
        tsc_version = "v1.0"
        amendments = []
    elif code_prefix == "10":
        da = "Climate Delegated Act"
        version = "(EU) 2021/2139"
        effective = "2022-01-01"
        tsc_version = "v1.0"
        amendments = []
    else:
        da = "Climate Delegated Act"
        version = "(EU) 2021/2139"
        effective = "2022-01-01"
        tsc_version = "v1.0"
        amendments = []

    return ApplicableVersionResponse(
        activity_code=activity_code, activity_name=activity_name,
        applicable_da=da, da_version=version, effective_from=effective,
        next_review="2027-01-01", tsc_version=tsc_version,
        amendments_applied=amendments, generated_at=_now(),
    )


@router.get(
    "/transition-plan/{org_id}",
    response_model=TransitionPlanResponse,
    summary="Get transition plan",
    description="Get regulatory transition plan with upcoming changes and deadlines.",
)
async def get_transition_plan(org_id: str) -> TransitionPlanResponse:
    """Get transition plan."""
    return TransitionPlanResponse(
        org_id=org_id, current_framework="CSRD + Article 8 DA",
        upcoming_changes=[
            {"change": "EDA activities mandatory reporting", "effective": "2025-01-01", "status": "in_force", "impact": "medium"},
            {"change": "Updated manufacturing TSC thresholds", "effective": "2025-01-01", "status": "in_force", "impact": "medium"},
            {"change": "Omnibus simplification (if adopted)", "effective": "2026-01-01", "status": "proposed", "impact": "high"},
            {"change": "Platform TSC review (transport, buildings)", "effective": "2027-01-01", "status": "under_review", "impact": "medium"},
        ],
        action_items=[
            {"action": "Assess EDA activities for eligibility", "deadline": "2025-06-30", "priority": "high", "status": "in_progress"},
            {"action": "Update manufacturing TSC calculations", "deadline": "2025-03-31", "priority": "high", "status": "complete"},
            {"action": "Monitor Omnibus legislative process", "deadline": "2025-12-31", "priority": "medium", "status": "ongoing"},
            {"action": "Prepare for potential TSC threshold changes", "deadline": "2026-06-30", "priority": "low", "status": "not_started"},
        ],
        next_reporting_deadline="2026-04-30",
        prepared_pct=72.0,
        generated_at=_now(),
    )
