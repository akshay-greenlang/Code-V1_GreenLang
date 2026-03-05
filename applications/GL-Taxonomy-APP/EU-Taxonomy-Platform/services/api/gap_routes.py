"""
GL-Taxonomy-APP Gap Analysis API

Identifies gaps in EU Taxonomy alignment readiness across eligibility,
substantial contribution, DNSH, minimum safeguards, and data quality.
Generates prioritized action plans and provides a priority matrix for
remediation planning.

Gap Analysis Dimensions:
    - SC Gaps: Activities failing TSC thresholds or missing data
    - DNSH Gaps: Non-compliant DNSH objectives
    - Safeguard Gaps: Missing policies or adverse findings
    - Data Gaps: Insufficient evidence or low DQ scores
    - Reporting Gaps: Incomplete Article 8 / EBA disclosures
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/gap-analysis", tags=["Gap Analysis"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GapAnalysisRequest(BaseModel):
    """Run gap analysis."""
    sc_score: float = Query(75, ge=0, le=100, description="SC pass rate %")
    dnsh_score: float = Query(70, ge=0, le=100, description="DNSH compliance rate %")
    safeguard_score: float = Query(80, ge=0, le=100, description="Safeguard compliance %")
    data_quality_score: float = Query(65, ge=0, le=100, description="Data quality score %")
    reporting_score: float = Query(85, ge=0, le=100, description="Reporting completeness %")


class ActionPlanRequest(BaseModel):
    """Generate action plan."""
    budget_eur: Optional[float] = Field(None, ge=0)
    timeline_months: int = Field(12, ge=1, le=36)
    priority_focus: Optional[str] = Field(None, description="sc, dnsh, safeguards, data_quality, reporting")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class GapAnalysisResponse(BaseModel):
    """Full gap analysis result."""
    analysis_id: str
    org_id: str
    overall_readiness_pct: float
    readiness_level: str
    total_gaps: int
    critical_gaps: int
    gap_categories: Dict[str, Any]
    assessed_at: datetime


class GapResultsResponse(BaseModel):
    """Detailed gap results."""
    org_id: str
    gaps: List[Dict[str, Any]]
    total_gaps: int
    critical: int
    high: int
    medium: int
    low: int
    recommendations: List[str]
    generated_at: datetime


class DNSHGapsResponse(BaseModel):
    """DNSH-specific gaps."""
    org_id: str
    dnsh_gaps: List[Dict[str, Any]]
    total_gaps: int
    by_objective: Dict[str, int]
    most_common_gap: str
    recommendations: List[str]
    generated_at: datetime


class SafeguardGapsResponse(BaseModel):
    """Safeguard gaps."""
    org_id: str
    safeguard_gaps: List[Dict[str, Any]]
    total_gaps: int
    by_topic: Dict[str, int]
    procedural_gaps: int
    outcome_gaps: int
    recommendations: List[str]
    generated_at: datetime


class DataGapsResponse(BaseModel):
    """Data quality gaps."""
    org_id: str
    data_gaps: List[Dict[str, Any]]
    total_gaps: int
    by_dimension: Dict[str, int]
    activities_below_threshold: int
    recommendations: List[str]
    generated_at: datetime


class ActionPlanResponse(BaseModel):
    """Prioritized action plan."""
    plan_id: str
    org_id: str
    actions: List[Dict[str, Any]]
    total_actions: int
    critical_actions: int
    estimated_cost_eur: float
    estimated_weeks: int
    expected_improvement_pct: float
    priority_sequence: List[str]
    generated_at: datetime


class PriorityMatrixResponse(BaseModel):
    """Priority matrix (impact vs effort)."""
    org_id: str
    quadrants: Dict[str, List[Dict[str, Any]]]
    total_items: int
    quick_wins: int
    strategic: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_analyses: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{org_id}",
    response_model=GapAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run gap analysis",
    description=(
        "Run a comprehensive gap analysis evaluating SC pass rates, "
        "DNSH compliance, safeguard readiness, data quality, and "
        "reporting completeness."
    ),
)
async def run_gap_analysis(
    org_id: str,
    sc_score: float = Query(75, ge=0, le=100),
    dnsh_score: float = Query(70, ge=0, le=100),
    safeguard_score: float = Query(80, ge=0, le=100),
    data_quality_score: float = Query(65, ge=0, le=100),
    reporting_score: float = Query(85, ge=0, le=100),
) -> GapAnalysisResponse:
    """Run gap analysis."""
    analysis_id = _generate_id("gap")
    scores = {
        "substantial_contribution": sc_score,
        "dnsh_compliance": dnsh_score,
        "minimum_safeguards": safeguard_score,
        "data_quality": data_quality_score,
        "reporting_completeness": reporting_score,
    }
    overall = round(sum(scores.values()) / len(scores), 1)

    if overall >= 90:
        level = "ready"
    elif overall >= 75:
        level = "nearly_ready"
    elif overall >= 50:
        level = "partial"
    else:
        level = "not_ready"

    gaps = []
    critical = 0
    if sc_score < 80:
        gaps.append({"category": "sc", "description": "Activities not meeting TSC", "severity": "high" if sc_score < 60 else "medium"})
        if sc_score < 60:
            critical += 1
    if dnsh_score < 80:
        gaps.append({"category": "dnsh", "description": "DNSH non-compliance", "severity": "high" if dnsh_score < 60 else "medium"})
        if dnsh_score < 60:
            critical += 1
    if safeguard_score < 90:
        gaps.append({"category": "safeguards", "description": "Safeguard gaps", "severity": "high" if safeguard_score < 70 else "medium"})
        if safeguard_score < 70:
            critical += 1
    if data_quality_score < 70:
        gaps.append({"category": "data_quality", "description": "Data quality below threshold", "severity": "high"})
        critical += 1
    if reporting_score < 90:
        gaps.append({"category": "reporting", "description": "Reporting incomplete", "severity": "medium"})

    data = {
        "analysis_id": analysis_id, "org_id": org_id,
        "overall_readiness_pct": overall, "readiness_level": level,
        "total_gaps": len(gaps), "critical_gaps": critical,
        "gap_categories": scores, "assessed_at": _now(),
    }
    _analyses[analysis_id] = data
    return GapAnalysisResponse(**data)


@router.get(
    "/{org_id}/results",
    response_model=GapResultsResponse,
    summary="Get gap results",
    description="Get detailed gap analysis results.",
)
async def get_results(org_id: str) -> GapResultsResponse:
    """Get gap results."""
    gaps = [
        {"gap": "Activity 4.29 exceeds 270 gCO2e/kWh threshold", "category": "sc", "severity": "critical", "activity_code": "4.29", "remediation": "Upgrade to lower-emission gas turbine or switch to renewable"},
        {"gap": "DNSH water compliance missing for 3 activities", "category": "dnsh", "severity": "high", "objective": "water", "remediation": "Implement WFD compliance and water use efficiency plans"},
        {"gap": "Climate risk assessment not performed for 5 activities", "category": "dnsh", "severity": "high", "objective": "climate_change_adaptation", "remediation": "Conduct Appendix A climate risk assessment"},
        {"gap": "Anti-corruption policy not formally adopted", "category": "safeguards", "severity": "medium", "topic": "anti_corruption", "remediation": "Adopt and publish anti-corruption policy"},
        {"gap": "Third-party verification missing for 8 activities", "category": "data_quality", "severity": "high", "dimension": "verifiability", "remediation": "Engage external auditor for DQ verification"},
        {"gap": "Qualitative disclosure incomplete", "category": "reporting", "severity": "low", "section": "contextual_narrative", "remediation": "Complete narrative disclosure sections"},
    ]

    sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for g in gaps:
        sev_counts[g["severity"]] = sev_counts.get(g["severity"], 0) + 1

    return GapResultsResponse(
        org_id=org_id, gaps=gaps, total_gaps=len(gaps),
        critical=sev_counts["critical"], high=sev_counts["high"],
        medium=sev_counts["medium"], low=sev_counts["low"],
        recommendations=[
            "Priority 1: Address critical SC threshold failure for Activity 4.29",
            "Priority 2: Complete climate risk assessments for DNSH CCA",
            "Priority 3: Implement water compliance plans",
            "Priority 4: Obtain third-party verification",
            "Priority 5: Formalize anti-corruption policy",
        ],
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/dnsh-gaps",
    response_model=DNSHGapsResponse,
    summary="DNSH-specific gaps",
    description="Get DNSH-specific gaps across all five environmental objectives.",
)
async def get_dnsh_gaps(org_id: str) -> DNSHGapsResponse:
    """Get DNSH gaps."""
    dnsh_gaps = [
        {"activity_code": "4.1", "objective": "climate_change_adaptation", "gap": "No climate risk assessment", "severity": "high"},
        {"activity_code": "4.3", "objective": "climate_change_adaptation", "gap": "No climate risk assessment", "severity": "high"},
        {"activity_code": "7.1", "objective": "water", "gap": "WFD compliance not demonstrated", "severity": "high"},
        {"activity_code": "7.2", "objective": "water", "gap": "No water use efficiency plan", "severity": "medium"},
        {"activity_code": "3.9", "objective": "pollution_prevention", "gap": "IED BAT compliance pending", "severity": "medium"},
        {"activity_code": "6.5", "objective": "circular_economy", "gap": "Waste hierarchy not documented", "severity": "low"},
        {"activity_code": "7.7", "objective": "biodiversity", "gap": "EIA not completed", "severity": "medium"},
    ]

    by_objective = {}
    for g in dnsh_gaps:
        obj = g["objective"]
        by_objective[obj] = by_objective.get(obj, 0) + 1

    most_common = max(by_objective, key=by_objective.get) if by_objective else "none"

    return DNSHGapsResponse(
        org_id=org_id, dnsh_gaps=dnsh_gaps, total_gaps=len(dnsh_gaps),
        by_objective=by_objective, most_common_gap=most_common,
        recommendations=[
            "Conduct climate risk assessments for all CCM activities (DNSH CCA)",
            "Implement Water Framework Directive compliance for construction activities",
            "Document waste hierarchy application for transport activities",
            "Complete EIA for building acquisition activities",
        ],
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/safeguard-gaps",
    response_model=SafeguardGapsResponse,
    summary="Safeguard gaps",
    description="Get minimum safeguard gaps across all four topics.",
)
async def get_safeguard_gaps(org_id: str) -> SafeguardGapsResponse:
    """Get safeguard gaps."""
    safeguard_gaps = [
        {"topic": "human_rights", "gap_type": "procedural", "gap": "Grievance mechanism not operational", "severity": "high"},
        {"topic": "anti_corruption", "gap_type": "procedural", "gap": "Anti-corruption policy not formally adopted", "severity": "medium"},
        {"topic": "anti_corruption", "gap_type": "procedural", "gap": "Employee training coverage below 80%", "severity": "low"},
        {"topic": "taxation", "gap_type": "procedural", "gap": "CBCR not published", "severity": "medium"},
    ]

    by_topic = {}
    procedural = 0
    outcome = 0
    for g in safeguard_gaps:
        by_topic[g["topic"]] = by_topic.get(g["topic"], 0) + 1
        if g["gap_type"] == "procedural":
            procedural += 1
        else:
            outcome += 1

    return SafeguardGapsResponse(
        org_id=org_id, safeguard_gaps=safeguard_gaps,
        total_gaps=len(safeguard_gaps), by_topic=by_topic,
        procedural_gaps=procedural, outcome_gaps=outcome,
        recommendations=[
            "Establish and operationalize grievance mechanism per UNGP",
            "Formally adopt and publish anti-corruption policy",
            "Expand anti-corruption training to 100% of employees",
            "Publish country-by-country tax reporting per BEPS Action 13",
        ],
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/data-gaps",
    response_model=DataGapsResponse,
    summary="Data quality gaps",
    description="Get data quality gaps by dimension and activity.",
)
async def get_data_gaps(org_id: str) -> DataGapsResponse:
    """Get data gaps."""
    data_gaps = [
        {"activity_code": "4.29", "dimension": "accuracy", "gap": "Gas generation emissions data from estimates, not measurements", "dq_score": 2.0, "severity": "high"},
        {"activity_code": "7.2", "dimension": "completeness", "gap": "Primary energy demand data missing for 3 buildings", "dq_score": 2.5, "severity": "high"},
        {"activity_code": "3.9", "dimension": "verifiability", "gap": "No third-party verification of steel GHG intensity", "dq_score": 2.0, "severity": "high"},
        {"activity_code": "6.5", "dimension": "granularity", "gap": "Fleet data at aggregate level, not per-vehicle", "dq_score": 2.5, "severity": "medium"},
        {"activity_code": "7.7", "dimension": "timeliness", "gap": "EPC data from 2022, not current", "dq_score": 3.0, "severity": "medium"},
        {"activity_code": "8.1", "dimension": "consistency", "gap": "PUE values from different measurement methodologies", "dq_score": 3.0, "severity": "low"},
    ]

    by_dim = {}
    below_threshold = 0
    for g in data_gaps:
        dim = g["dimension"]
        by_dim[dim] = by_dim.get(dim, 0) + 1
        if g["dq_score"] < 3.0:
            below_threshold += 1

    return DataGapsResponse(
        org_id=org_id, data_gaps=data_gaps, total_gaps=len(data_gaps),
        by_dimension=by_dim, activities_below_threshold=below_threshold,
        recommendations=[
            "Install continuous emissions monitoring for gas generation (Activity 4.29)",
            "Collect primary energy demand certificates for all renovation buildings",
            "Engage external auditor for steel GHG intensity verification",
            "Upgrade fleet data to per-vehicle level for accurate SC assessment",
            "Obtain current EPC ratings for all building acquisitions",
        ],
        generated_at=_now(),
    )


@router.post(
    "/{org_id}/action-plan",
    response_model=ActionPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate action plan",
    description="Generate a prioritized action plan to close identified gaps.",
)
async def generate_action_plan(
    org_id: str,
    request: ActionPlanRequest,
) -> ActionPlanResponse:
    """Generate action plan."""
    plan_id = _generate_id("ap")

    actions = [
        {"action": "Upgrade gas generation emissions monitoring", "category": "data_quality", "priority": "critical", "effort": "high", "weeks": 8, "cost_eur": 75000, "impact_pct": 5.0},
        {"action": "Conduct Appendix A climate risk assessments (5 activities)", "category": "dnsh", "priority": "critical", "effort": "medium", "weeks": 6, "cost_eur": 40000, "impact_pct": 4.0},
        {"action": "Implement water compliance plans (3 activities)", "category": "dnsh", "priority": "high", "effort": "medium", "weeks": 8, "cost_eur": 30000, "impact_pct": 3.0},
        {"action": "Obtain third-party DQ verification (8 activities)", "category": "data_quality", "priority": "high", "effort": "high", "weeks": 10, "cost_eur": 60000, "impact_pct": 4.0},
        {"action": "Establish grievance mechanism", "category": "safeguards", "priority": "high", "effort": "medium", "weeks": 6, "cost_eur": 25000, "impact_pct": 2.0},
        {"action": "Adopt anti-corruption policy + training", "category": "safeguards", "priority": "medium", "effort": "low", "weeks": 4, "cost_eur": 15000, "impact_pct": 1.5},
        {"action": "Complete qualitative disclosures", "category": "reporting", "priority": "medium", "effort": "low", "weeks": 3, "cost_eur": 8000, "impact_pct": 1.0},
        {"action": "Publish CBCR report", "category": "safeguards", "priority": "medium", "effort": "medium", "weeks": 6, "cost_eur": 20000, "impact_pct": 1.0},
    ]

    if request.budget_eur is not None:
        cumulative = 0
        filtered = []
        for a in actions:
            cumulative += a["cost_eur"]
            if cumulative <= request.budget_eur:
                filtered.append(a)
        actions = filtered

    if request.priority_focus:
        focus_actions = [a for a in actions if a["category"] == request.priority_focus]
        other_actions = [a for a in actions if a["category"] != request.priority_focus]
        actions = focus_actions + other_actions

    total_cost = sum(a["cost_eur"] for a in actions)
    max_weeks = max((a["weeks"] for a in actions), default=0)
    total_impact = sum(a["impact_pct"] for a in actions)
    critical = sum(1 for a in actions if a["priority"] == "critical")

    return ActionPlanResponse(
        plan_id=plan_id, org_id=org_id,
        actions=actions, total_actions=len(actions),
        critical_actions=critical, estimated_cost_eur=total_cost,
        estimated_weeks=max_weeks, expected_improvement_pct=total_impact,
        priority_sequence=[
            "1. Upgrade emissions monitoring (critical, blocking)",
            "2. Climate risk assessments (critical, DNSH CCA)",
            "3. Water compliance plans (high, DNSH WTR)",
            "4. Third-party verification (high, data quality)",
            "5. Safeguard policies and mechanisms",
        ],
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/priority-matrix",
    response_model=PriorityMatrixResponse,
    summary="Priority matrix",
    description="Get gap remediation priority matrix (impact vs effort).",
)
async def get_priority_matrix(org_id: str) -> PriorityMatrixResponse:
    """Get priority matrix."""
    quadrants = {
        "quick_wins": [
            {"item": "Adopt anti-corruption policy", "impact": "medium", "effort": "low", "cost_eur": 15000, "weeks": 4},
            {"item": "Complete qualitative disclosures", "impact": "medium", "effort": "low", "cost_eur": 8000, "weeks": 3},
        ],
        "strategic": [
            {"item": "Climate risk assessments", "impact": "high", "effort": "medium", "cost_eur": 40000, "weeks": 6},
            {"item": "Water compliance plans", "impact": "high", "effort": "medium", "cost_eur": 30000, "weeks": 8},
            {"item": "Grievance mechanism", "impact": "high", "effort": "medium", "cost_eur": 25000, "weeks": 6},
        ],
        "major_projects": [
            {"item": "Emissions monitoring upgrade", "impact": "high", "effort": "high", "cost_eur": 75000, "weeks": 8},
            {"item": "Third-party verification", "impact": "high", "effort": "high", "cost_eur": 60000, "weeks": 10},
        ],
        "low_priority": [
            {"item": "Fleet data granularity upgrade", "impact": "low", "effort": "medium", "cost_eur": 12000, "weeks": 4},
            {"item": "PUE measurement standardization", "impact": "low", "effort": "low", "cost_eur": 5000, "weeks": 2},
        ],
    }

    total = sum(len(items) for items in quadrants.values())
    quick_wins = len(quadrants["quick_wins"])
    strategic = len(quadrants["strategic"])

    return PriorityMatrixResponse(
        org_id=org_id, quadrants=quadrants,
        total_items=total, quick_wins=quick_wins, strategic=strategic,
        generated_at=_now(),
    )
