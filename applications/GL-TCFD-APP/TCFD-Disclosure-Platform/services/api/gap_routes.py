"""
GL-TCFD-APP Gap Analysis API

Assesses organizational readiness against TCFD recommended disclosures and
ISSB/IFRS S2 requirements.  Provides gap assessment, pillar maturity scores,
action plan generation, peer benchmarking, compliance timeline estimation,
and improvement recommendations.

Assessment Dimensions:
    - Governance: Board oversight, management role, integration
    - Strategy: Risk/opportunity identification, scenario analysis, resilience
    - Risk Management: Process maturity, ERM integration, response actions
    - Metrics & Targets: GHG reporting, cross-industry metrics, science-based targets
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/tcfd/gap-analysis", tags=["Gap Analysis"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class GapAssessmentResponse(BaseModel):
    """Gap assessment result."""
    assessment_id: str
    org_id: str
    overall_readiness_pct: float
    overall_maturity: str
    pillar_scores: Dict[str, float]
    total_gaps: int
    critical_gaps: int
    gap_details: List[Dict[str, Any]]
    assessed_at: datetime


class AssessmentHistoryEntry(BaseModel):
    """Assessment history entry."""
    assessment_id: str
    org_id: str
    overall_readiness_pct: float
    overall_maturity: str
    assessed_at: datetime


class PillarScoresResponse(BaseModel):
    """Pillar maturity scores."""
    org_id: str
    governance: Dict[str, Any]
    strategy: Dict[str, Any]
    risk_management: Dict[str, Any]
    metrics_and_targets: Dict[str, Any]
    overall_score: float
    overall_maturity: str
    generated_at: datetime


class ActionPlanResponse(BaseModel):
    """Generated action plan."""
    org_id: str
    actions: List[Dict[str, Any]]
    total_actions: int
    estimated_completion_months: int
    estimated_cost_usd: float
    priority_actions: List[Dict[str, Any]]
    generated_at: datetime


class GapBenchmarkResponse(BaseModel):
    """Peer benchmarking for gap analysis."""
    org_id: str
    org_readiness_pct: float
    peer_average_pct: float
    peer_median_pct: float
    sector_average_pct: float
    percentile_rank: int
    peer_count: int
    by_pillar: Dict[str, Dict[str, float]]
    generated_at: datetime


class ComplianceTimelineResponse(BaseModel):
    """Compliance timeline estimate."""
    org_id: str
    current_readiness_pct: float
    milestones: List[Dict[str, Any]]
    estimated_full_compliance_date: str
    estimated_months_to_compliance: int
    resource_requirements: Dict[str, Any]
    generated_at: datetime


class RecommendationsResponse(BaseModel):
    """Improvement recommendations."""
    org_id: str
    recommendations: List[Dict[str, Any]]
    quick_wins: List[Dict[str, Any]]
    strategic_initiatives: List[Dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_assessments: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Gap Assessment Logic
# ---------------------------------------------------------------------------

GAP_CRITERIA = {
    "governance": [
        {"ref": "GOV-1", "criterion": "Board-level climate oversight established", "weight": 15, "category": "critical"},
        {"ref": "GOV-2", "criterion": "Dedicated sustainability committee", "weight": 10, "category": "important"},
        {"ref": "GOV-3", "criterion": "Management climate accountability defined", "weight": 12, "category": "critical"},
        {"ref": "GOV-4", "criterion": "Climate integrated into strategic planning", "weight": 10, "category": "important"},
        {"ref": "GOV-5", "criterion": "Climate KPIs in executive remuneration", "weight": 8, "category": "recommended"},
        {"ref": "GOV-6", "criterion": "Board climate competency assessment", "weight": 5, "category": "recommended"},
    ],
    "strategy": [
        {"ref": "STR-1", "criterion": "Climate risks identified (physical and transition)", "weight": 15, "category": "critical"},
        {"ref": "STR-2", "criterion": "Climate opportunities identified", "weight": 10, "category": "important"},
        {"ref": "STR-3", "criterion": "Scenario analysis performed (at least 2 scenarios)", "weight": 15, "category": "critical"},
        {"ref": "STR-4", "criterion": "Financial impact quantified", "weight": 12, "category": "critical"},
        {"ref": "STR-5", "criterion": "Strategy resilience assessed", "weight": 10, "category": "important"},
        {"ref": "STR-6", "criterion": "Value chain impact mapped", "weight": 8, "category": "recommended"},
    ],
    "risk_management": [
        {"ref": "RM-1", "criterion": "Climate risk identification process defined", "weight": 12, "category": "critical"},
        {"ref": "RM-2", "criterion": "Risk assessment methodology established", "weight": 12, "category": "critical"},
        {"ref": "RM-3", "criterion": "Risk response actions defined", "weight": 10, "category": "important"},
        {"ref": "RM-4", "criterion": "Climate risks integrated into ERM", "weight": 12, "category": "critical"},
        {"ref": "RM-5", "criterion": "Key risk indicators monitored", "weight": 8, "category": "important"},
        {"ref": "RM-6", "criterion": "Regular risk review schedule", "weight": 6, "category": "recommended"},
    ],
    "metrics_and_targets": [
        {"ref": "MT-1", "criterion": "Scope 1 GHG emissions reported", "weight": 15, "category": "critical"},
        {"ref": "MT-2", "criterion": "Scope 2 GHG emissions reported (location and market)", "weight": 12, "category": "critical"},
        {"ref": "MT-3", "criterion": "Scope 3 GHG emissions reported (material categories)", "weight": 12, "category": "critical"},
        {"ref": "MT-4", "criterion": "Emission intensity metrics calculated", "weight": 8, "category": "important"},
        {"ref": "MT-5", "criterion": "Science-based targets set", "weight": 10, "category": "critical"},
        {"ref": "MT-6", "criterion": "Target progress tracked annually", "weight": 8, "category": "important"},
        {"ref": "MT-7", "criterion": "Cross-industry metrics reported (ISSB 7)", "weight": 8, "category": "important"},
    ],
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assess/{org_id}",
    response_model=GapAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run gap assessment",
    description=(
        "Run a comprehensive TCFD gap assessment across all four pillars. "
        "Evaluates readiness against 25 criteria and generates gap details "
        "with prioritization."
    ),
)
async def run_gap_assessment(
    org_id: str,
    governance_score: float = Query(50, ge=0, le=100, description="Governance readiness (0-100)"),
    strategy_score: float = Query(40, ge=0, le=100, description="Strategy readiness (0-100)"),
    risk_mgmt_score: float = Query(35, ge=0, le=100, description="Risk management readiness (0-100)"),
    metrics_score: float = Query(45, ge=0, le=100, description="Metrics readiness (0-100)"),
) -> GapAssessmentResponse:
    """Run gap assessment."""
    assessment_id = _generate_id("gap")
    scores = {
        "governance": governance_score,
        "strategy": strategy_score,
        "risk_management": risk_mgmt_score,
        "metrics_and_targets": metrics_score,
    }

    weights = {"governance": 0.20, "strategy": 0.30, "risk_management": 0.20, "metrics_and_targets": 0.30}
    overall = round(sum(scores[p] * weights[p] for p in scores), 1)

    if overall >= 85:
        maturity = "leading"
    elif overall >= 70:
        maturity = "advanced"
    elif overall >= 50:
        maturity = "intermediate"
    elif overall >= 30:
        maturity = "developing"
    else:
        maturity = "initial"

    gaps = []
    critical_count = 0
    for pillar, criteria in GAP_CRITERIA.items():
        pillar_score = scores.get(pillar, 0)
        threshold = 60.0
        for c in criteria:
            # Simulate gap based on pillar score and criterion weight
            if pillar_score < threshold + c["weight"]:
                gap_item = {
                    "pillar": pillar,
                    "reference": c["ref"],
                    "criterion": c["criterion"],
                    "category": c["category"],
                    "current_status": "partial" if pillar_score > 30 else "not_started",
                    "gap_severity": "high" if c["category"] == "critical" else "medium" if c["category"] == "important" else "low",
                    "remediation_effort": "high" if c["weight"] >= 12 else "medium" if c["weight"] >= 8 else "low",
                }
                gaps.append(gap_item)
                if c["category"] == "critical":
                    critical_count += 1

    result = {
        "assessment_id": assessment_id,
        "org_id": org_id,
        "overall_readiness_pct": overall,
        "overall_maturity": maturity,
        "pillar_scores": scores,
        "total_gaps": len(gaps),
        "critical_gaps": critical_count,
        "gap_details": gaps,
        "assessed_at": _now(),
    }
    _assessments[assessment_id] = result
    return GapAssessmentResponse(**result)


@router.get(
    "/results/{org_id}",
    response_model=GapAssessmentResponse,
    summary="Get latest assessment",
    description="Get the most recent gap assessment for an organization.",
)
async def get_latest_assessment(org_id: str) -> GapAssessmentResponse:
    """Get the latest gap assessment."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]
    if not org_assessments:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No assessments found for org {org_id}")
    latest = max(org_assessments, key=lambda a: a["assessed_at"])
    return GapAssessmentResponse(**latest)


@router.get(
    "/history/{org_id}",
    response_model=List[AssessmentHistoryEntry],
    summary="Assessment history",
    description="Get assessment history for an organization.",
)
async def get_assessment_history(
    org_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[AssessmentHistoryEntry]:
    """Get assessment history."""
    results = [a for a in _assessments.values() if a["org_id"] == org_id]
    results.sort(key=lambda a: a["assessed_at"], reverse=True)
    return [
        AssessmentHistoryEntry(
            assessment_id=a["assessment_id"],
            org_id=a["org_id"],
            overall_readiness_pct=a["overall_readiness_pct"],
            overall_maturity=a["overall_maturity"],
            assessed_at=a["assessed_at"],
        )
        for a in results[:limit]
    ]


@router.get(
    "/pillar-scores/{org_id}",
    response_model=PillarScoresResponse,
    summary="Pillar maturity scores",
    description="Get detailed maturity scores for each TCFD pillar.",
)
async def get_pillar_scores(org_id: str) -> PillarScoresResponse:
    """Get pillar maturity scores."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]
    if org_assessments:
        latest = max(org_assessments, key=lambda a: a["assessed_at"])
        scores = latest["pillar_scores"]
    else:
        scores = {"governance": 0, "strategy": 0, "risk_management": 0, "metrics_and_targets": 0}

    def _pillar_detail(name: str, score: float) -> Dict[str, Any]:
        criteria = GAP_CRITERIA.get(name, [])
        met = sum(1 for c in criteria if score > 50 + c["weight"])
        return {
            "score": score,
            "maturity": "leading" if score >= 85 else "advanced" if score >= 70 else "intermediate" if score >= 50 else "developing",
            "criteria_total": len(criteria),
            "criteria_met": met,
        }

    overall = round(sum(scores.values()) / len(scores), 1) if scores else 0

    return PillarScoresResponse(
        org_id=org_id,
        governance=_pillar_detail("governance", scores.get("governance", 0)),
        strategy=_pillar_detail("strategy", scores.get("strategy", 0)),
        risk_management=_pillar_detail("risk_management", scores.get("risk_management", 0)),
        metrics_and_targets=_pillar_detail("metrics_and_targets", scores.get("metrics_and_targets", 0)),
        overall_score=overall,
        overall_maturity="leading" if overall >= 85 else "advanced" if overall >= 70 else "intermediate" if overall >= 50 else "developing",
        generated_at=_now(),
    )


@router.get(
    "/action-plan/{org_id}",
    response_model=ActionPlanResponse,
    summary="Generated action plan",
    description="Generate a prioritized action plan to close identified gaps.",
)
async def get_action_plan(org_id: str) -> ActionPlanResponse:
    """Generate action plan."""
    actions = [
        {"action": "Establish board climate oversight committee", "pillar": "governance", "priority": "critical", "effort": "medium", "timeline_months": 3, "cost_usd": 50000},
        {"action": "Conduct climate scenario analysis (2+ scenarios)", "pillar": "strategy", "priority": "critical", "effort": "high", "timeline_months": 6, "cost_usd": 150000},
        {"action": "Quantify financial impact of climate risks", "pillar": "strategy", "priority": "critical", "effort": "high", "timeline_months": 4, "cost_usd": 100000},
        {"action": "Integrate climate risks into ERM framework", "pillar": "risk_management", "priority": "critical", "effort": "medium", "timeline_months": 4, "cost_usd": 75000},
        {"action": "Report Scope 3 emissions (material categories)", "pillar": "metrics_and_targets", "priority": "critical", "effort": "high", "timeline_months": 6, "cost_usd": 120000},
        {"action": "Set science-based targets (SBTi)", "pillar": "metrics_and_targets", "priority": "critical", "effort": "medium", "timeline_months": 6, "cost_usd": 80000},
        {"action": "Link executive remuneration to climate KPIs", "pillar": "governance", "priority": "important", "effort": "medium", "timeline_months": 6, "cost_usd": 30000},
        {"action": "Develop physical risk assessment for key assets", "pillar": "strategy", "priority": "important", "effort": "medium", "timeline_months": 3, "cost_usd": 80000},
        {"action": "Implement climate KRI monitoring dashboard", "pillar": "risk_management", "priority": "important", "effort": "medium", "timeline_months": 4, "cost_usd": 60000},
        {"action": "Report ISSB cross-industry metrics", "pillar": "metrics_and_targets", "priority": "important", "effort": "medium", "timeline_months": 3, "cost_usd": 40000},
    ]

    total_cost = sum(a["cost_usd"] for a in actions)
    max_months = max(a["timeline_months"] for a in actions)
    priority_actions = [a for a in actions if a["priority"] == "critical"]

    return ActionPlanResponse(
        org_id=org_id,
        actions=actions,
        total_actions=len(actions),
        estimated_completion_months=max_months,
        estimated_cost_usd=total_cost,
        priority_actions=priority_actions,
        generated_at=_now(),
    )


@router.get(
    "/benchmark/{org_id}",
    response_model=GapBenchmarkResponse,
    summary="Peer benchmarking",
    description="Benchmark the organization's TCFD readiness against sector peers.",
)
async def get_gap_benchmark(org_id: str) -> GapBenchmarkResponse:
    """Benchmark gap analysis against peers."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]
    org_readiness = org_assessments[-1]["overall_readiness_pct"] if org_assessments else 0

    peer_scores = [35, 42, 48, 52, 58, 62, 68, 72, 78, 85]
    peer_avg = round(sum(peer_scores) / len(peer_scores), 1)
    sorted_peers = sorted(peer_scores)
    peer_median = round((sorted_peers[4] + sorted_peers[5]) / 2, 1)
    sector_avg = round(peer_avg * 0.95, 1)
    below = sum(1 for s in peer_scores if s <= org_readiness)
    percentile = round(below / len(peer_scores) * 100)

    by_pillar = {
        "governance": {"org": 50, "peer_avg": 55},
        "strategy": {"org": 40, "peer_avg": 48},
        "risk_management": {"org": 35, "peer_avg": 45},
        "metrics_and_targets": {"org": 45, "peer_avg": 52},
    }

    return GapBenchmarkResponse(
        org_id=org_id,
        org_readiness_pct=org_readiness,
        peer_average_pct=peer_avg,
        peer_median_pct=peer_median,
        sector_average_pct=sector_avg,
        percentile_rank=percentile,
        peer_count=len(peer_scores),
        by_pillar=by_pillar,
        generated_at=_now(),
    )


@router.get(
    "/timeline/{org_id}",
    response_model=ComplianceTimelineResponse,
    summary="Compliance timeline estimate",
    description="Estimate the timeline to achieve full TCFD/ISSB compliance.",
)
async def get_compliance_timeline(org_id: str) -> ComplianceTimelineResponse:
    """Estimate compliance timeline."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]
    current = org_assessments[-1]["overall_readiness_pct"] if org_assessments else 30

    gap = 100 - current
    months = max(round(gap / 5), 6)

    milestones = [
        {"milestone": "Governance structure in place", "target_month": 3, "readiness_at_completion": min(current + 15, 100)},
        {"milestone": "Scenario analysis completed", "target_month": 6, "readiness_at_completion": min(current + 30, 100)},
        {"milestone": "Risk management integrated", "target_month": 9, "readiness_at_completion": min(current + 45, 100)},
        {"milestone": "GHG reporting complete (Scope 1/2/3)", "target_month": 12, "readiness_at_completion": min(current + 55, 100)},
        {"milestone": "Science-based targets validated", "target_month": 15, "readiness_at_completion": min(current + 65, 100)},
        {"milestone": "Full TCFD disclosure published", "target_month": months, "readiness_at_completion": 100},
    ]

    return ComplianceTimelineResponse(
        org_id=org_id,
        current_readiness_pct=current,
        milestones=milestones,
        estimated_full_compliance_date=f"2027-{min(months % 12 + 1, 12):02d}-01",
        estimated_months_to_compliance=months,
        resource_requirements={
            "full_time_equivalents": 2.5,
            "external_consulting_usd": 200000,
            "technology_usd": 80000,
            "training_usd": 30000,
        },
        generated_at=_now(),
    )


@router.get(
    "/recommendations/{org_id}",
    response_model=RecommendationsResponse,
    summary="Improvement recommendations",
    description="Get prioritized improvement recommendations based on gap analysis.",
)
async def get_recommendations(org_id: str) -> RecommendationsResponse:
    """Get improvement recommendations."""
    quick_wins = [
        {"recommendation": "Publish climate governance statement on website", "pillar": "governance", "effort": "low", "impact": "medium", "timeline": "1 month"},
        {"recommendation": "Calculate and report Scope 1+2 emissions", "pillar": "metrics", "effort": "low", "impact": "high", "timeline": "2 months"},
        {"recommendation": "Define climate risk identification process", "pillar": "risk_management", "effort": "low", "impact": "medium", "timeline": "1 month"},
        {"recommendation": "Create climate risk register in existing ERM tool", "pillar": "risk_management", "effort": "low", "impact": "medium", "timeline": "1 month"},
    ]

    strategic = [
        {"recommendation": "Conduct multi-scenario climate analysis (IEA/NGFS)", "pillar": "strategy", "effort": "high", "impact": "very_high", "timeline": "6 months"},
        {"recommendation": "Develop and validate science-based targets", "pillar": "metrics", "effort": "high", "impact": "very_high", "timeline": "6 months"},
        {"recommendation": "Build financial impact model for climate risks", "pillar": "strategy", "effort": "high", "impact": "high", "timeline": "4 months"},
        {"recommendation": "Implement physical risk assessment platform", "pillar": "strategy", "effort": "high", "impact": "high", "timeline": "6 months"},
        {"recommendation": "Establish board-level sustainability committee", "pillar": "governance", "effort": "medium", "impact": "very_high", "timeline": "3 months"},
    ]

    all_recs = quick_wins + strategic

    return RecommendationsResponse(
        org_id=org_id,
        recommendations=all_recs,
        quick_wins=quick_wins,
        strategic_initiatives=strategic,
        generated_at=_now(),
    )
