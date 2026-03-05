"""
GL-SBTi-APP Gap Analysis API

Assesses organizational readiness for SBTi target-setting and validation
across data completeness, ambition alignment, methodology selection, and
submission preparedness.  Provides gap identification, action plans,
readiness scoring, and peer benchmarking.

Gap Analysis Dimensions:
    - Data Gaps: Missing emissions data, inventory incompleteness
    - Ambition Gaps: Target reduction vs. required pathway
    - Methodology Gaps: Appropriate method selection
    - Scope 3 Gaps: Category screening and coverage
    - FLAG Gaps: FLAG assessment completeness
    - Reporting Gaps: Disclosure and documentation
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/gap-analysis", tags=["Gap Analysis"])


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


class DataGapsResponse(BaseModel):
    """Data completeness gaps."""
    org_id: str
    overall_data_completeness_pct: float
    gaps: List[Dict[str, Any]]
    scope1_completeness: float
    scope2_completeness: float
    scope3_completeness: float
    base_year_complete: bool
    recommendations: List[str]
    generated_at: datetime


class AmbitionGapsResponse(BaseModel):
    """Ambition alignment gaps."""
    org_id: str
    scope1_2_ambition: Dict[str, Any]
    scope3_ambition: Dict[str, Any]
    overall_ambition_gap_pct: float
    on_minimum_pathway: bool
    recommendations: List[str]
    generated_at: datetime


class ActionPlanResponse(BaseModel):
    """Generated action plan to close gaps."""
    org_id: str
    actions: List[Dict[str, Any]]
    total_actions: int
    critical_actions: int
    estimated_weeks: int
    estimated_cost_usd: float
    priority_sequence: List[str]
    generated_at: datetime


class ReadinessScoreResponse(BaseModel):
    """Readiness score with category breakdown."""
    org_id: str
    overall_score: float
    level: str
    categories: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    next_steps: List[str]
    generated_at: datetime


class BenchmarkResponse(BaseModel):
    """Peer benchmark comparison."""
    org_id: str
    org_readiness_pct: float
    peer_average_pct: float
    peer_median_pct: float
    sector_average_pct: float
    percentile_rank: int
    peer_count: int
    by_category: Dict[str, Dict[str, float]]
    top_performers: List[Dict[str, Any]]
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
    "/org/{org_id}/run",
    response_model=GapAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run full gap analysis",
    description=(
        "Run a comprehensive gap analysis evaluating data completeness, "
        "ambition alignment, methodology selection, Scope 3 coverage, "
        "FLAG assessment, and reporting readiness."
    ),
)
async def run_gap_analysis(
    org_id: str,
    data_score: float = Query(80, ge=0, le=100, description="Data completeness score"),
    ambition_score: float = Query(75, ge=0, le=100, description="Ambition alignment score"),
    scope3_score: float = Query(55, ge=0, le=100, description="Scope 3 readiness score"),
    reporting_score: float = Query(60, ge=0, le=100, description="Reporting readiness score"),
) -> GapAnalysisResponse:
    """Run full gap analysis."""
    analysis_id = _generate_id("gap")
    scores = {
        "data_completeness": data_score,
        "ambition_alignment": ambition_score,
        "scope3_readiness": scope3_score,
        "reporting_readiness": reporting_score,
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
    if data_score < 90:
        gaps.append({"category": "data", "description": "Emissions inventory incomplete", "severity": "high" if data_score < 70 else "medium"})
        if data_score < 70:
            critical += 1
    if ambition_score < 80:
        gaps.append({"category": "ambition", "description": "Target ambition below minimum pathway", "severity": "high" if ambition_score < 60 else "medium"})
        if ambition_score < 60:
            critical += 1
    if scope3_score < 67:
        gaps.append({"category": "scope3", "description": "Scope 3 screening or coverage insufficient", "severity": "high"})
        critical += 1
    if reporting_score < 70:
        gaps.append({"category": "reporting", "description": "Reporting and disclosure gaps", "severity": "medium"})

    result = {
        "analysis_id": analysis_id,
        "org_id": org_id,
        "overall_readiness_pct": overall,
        "readiness_level": level,
        "total_gaps": len(gaps),
        "critical_gaps": critical,
        "gap_categories": scores,
        "assessed_at": _now(),
    }
    _analyses[analysis_id] = result
    return GapAnalysisResponse(**result)


@router.get(
    "/org/{org_id}/results",
    response_model=GapAnalysisResponse,
    summary="Get gap analysis results",
    description="Get the latest gap analysis results for an organization.",
)
async def get_results(org_id: str) -> GapAnalysisResponse:
    """Get latest gap analysis results."""
    org_analyses = [a for a in _analyses.values() if a["org_id"] == org_id]
    if not org_analyses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No gap analysis found for org {org_id}. Run POST /gap-analysis/org/{org_id}/run first.",
        )
    latest = max(org_analyses, key=lambda a: a["assessed_at"])
    return GapAnalysisResponse(**latest)


@router.get(
    "/org/{org_id}/data-gaps",
    response_model=DataGapsResponse,
    summary="Data gaps",
    description="Identify data completeness gaps across all emission scopes.",
)
async def get_data_gaps(org_id: str) -> DataGapsResponse:
    """Get data gaps."""
    gaps = [
        {"scope": "scope1", "category": "stationary_combustion", "gap": "Missing fuel records for 2 facilities", "severity": "high"},
        {"scope": "scope2", "category": "market_based", "gap": "Missing contractual instruments for 3 markets", "severity": "medium"},
        {"scope": "scope3", "category": "cat_2_capital_goods", "gap": "Only spend-based data; need supplier-specific", "severity": "medium"},
        {"scope": "scope3", "category": "cat_7_commuting", "gap": "No employee survey conducted", "severity": "low"},
        {"scope": "scope3", "category": "cat_10_processing", "gap": "No data collected; using screening estimates", "severity": "high"},
        {"scope": "scope3", "category": "cat_11_use_of_products", "gap": "Product use-phase data incomplete", "severity": "high"},
    ]

    return DataGapsResponse(
        org_id=org_id,
        overall_data_completeness_pct=78.0,
        gaps=gaps,
        scope1_completeness=92.0,
        scope2_completeness=88.0,
        scope3_completeness=62.0,
        base_year_complete=True,
        recommendations=[
            "Complete fuel records for remaining 2 facilities (Scope 1 gap)",
            "Obtain contractual instruments for market-based Scope 2",
            "Conduct employee commuting survey for Cat 7",
            "Engage downstream customers for Cat 10 and Cat 11 data",
            "Upgrade Cat 2 from spend-based to supplier-specific data",
        ],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/ambition-gaps",
    response_model=AmbitionGapsResponse,
    summary="Ambition gaps",
    description="Identify gaps between current target ambition and SBTi minimum pathways.",
)
async def get_ambition_gaps(org_id: str) -> AmbitionGapsResponse:
    """Get ambition gaps."""
    return AmbitionGapsResponse(
        org_id=org_id,
        scope1_2_ambition={
            "current_annual_reduction_pct": 4.2,
            "minimum_1_5C_pct": 4.2,
            "minimum_wb2C_pct": 2.5,
            "gap_pct": 0.0,
            "aligned_with": "1.5C",
        },
        scope3_ambition={
            "current_annual_reduction_pct": 2.0,
            "minimum_wb2C_pct": 2.5,
            "gap_pct": 0.5,
            "aligned_with": "below_minimum",
        },
        overall_ambition_gap_pct=0.25,
        on_minimum_pathway=False,
        recommendations=[
            "Increase Scope 3 annual reduction from 2.0% to at least 2.5% to meet well-below 2C minimum",
            "Consider supplier engagement approach for Cat 1 and Cat 4 to accelerate Scope 3 reduction",
            "Scope 1+2 ambition meets 1.5C alignment -- maintain current trajectory",
        ],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/action-plan",
    response_model=ActionPlanResponse,
    summary="Action plan",
    description="Generate a prioritized action plan to close identified gaps.",
)
async def get_action_plan(org_id: str) -> ActionPlanResponse:
    """Generate action plan."""
    actions = [
        {"action": "Complete Scope 3 screening (all 15 categories)", "category": "scope3", "priority": "critical", "effort": "high", "weeks": 6, "cost_usd": 80000},
        {"action": "Increase Scope 3 reduction ambition to 2.5% p.a.", "category": "ambition", "priority": "critical", "effort": "medium", "weeks": 4, "cost_usd": 30000},
        {"action": "Complete fuel records for remaining facilities", "category": "data", "priority": "high", "effort": "low", "weeks": 2, "cost_usd": 5000},
        {"action": "Obtain market-based Scope 2 instruments", "category": "data", "priority": "high", "effort": "medium", "weeks": 4, "cost_usd": 10000},
        {"action": "Launch supplier engagement program (top 50 suppliers)", "category": "scope3", "priority": "high", "effort": "high", "weeks": 12, "cost_usd": 120000},
        {"action": "Conduct employee commuting survey", "category": "data", "priority": "medium", "effort": "low", "weeks": 3, "cost_usd": 8000},
        {"action": "Establish public disclosure process", "category": "reporting", "priority": "high", "effort": "medium", "weeks": 4, "cost_usd": 15000},
        {"action": "Document recalculation policy", "category": "reporting", "priority": "medium", "effort": "low", "weeks": 2, "cost_usd": 5000},
    ]

    critical_count = sum(1 for a in actions if a["priority"] == "critical")
    total_cost = sum(a["cost_usd"] for a in actions)
    max_weeks = max(a["weeks"] for a in actions)

    return ActionPlanResponse(
        org_id=org_id,
        actions=actions,
        total_actions=len(actions),
        critical_actions=critical_count,
        estimated_weeks=max_weeks,
        estimated_cost_usd=total_cost,
        priority_sequence=[
            "1. Complete Scope 3 screening (blocking)",
            "2. Increase Scope 3 ambition",
            "3. Fix Scope 1 data gaps",
            "4. Launch supplier engagement",
            "5. Establish reporting processes",
        ],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/readiness-score",
    response_model=ReadinessScoreResponse,
    summary="Readiness score",
    description="Get the SBTi readiness score with category breakdown.",
)
async def get_readiness_score(org_id: str) -> ReadinessScoreResponse:
    """Get readiness score."""
    categories = {
        "emissions_inventory": 88.0,
        "scope3_coverage": 62.0,
        "target_ambition": 85.0,
        "methodology_selection": 90.0,
        "flag_assessment": 70.0,
        "reporting_disclosure": 65.0,
        "governance_approval": 80.0,
    }
    overall = round(sum(categories.values()) / len(categories), 1)

    if overall >= 90:
        level = "ready"
    elif overall >= 75:
        level = "nearly_ready"
    elif overall >= 50:
        level = "partial"
    else:
        level = "not_ready"

    strengths = [k for k, v in categories.items() if v >= 85]
    weaknesses = [k for k, v in categories.items() if v < 70]

    return ReadinessScoreResponse(
        org_id=org_id,
        overall_score=overall,
        level=level,
        categories=categories,
        strengths=strengths,
        weaknesses=weaknesses,
        next_steps=[
            "Expand Scope 3 coverage from 62% to 67% minimum",
            "Complete reporting disclosure framework",
            "Finalize FLAG commodity-level assessment",
        ],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/benchmarks",
    response_model=BenchmarkResponse,
    summary="Peer benchmarks",
    description="Benchmark SBTi readiness against sector peers.",
)
async def get_benchmarks(org_id: str) -> BenchmarkResponse:
    """Get peer benchmarks."""
    org_readiness = 77.1
    peer_scores = [55, 62, 68, 72, 78, 80, 85, 88, 92, 95]
    peer_avg = round(sum(peer_scores) / len(peer_scores), 1)
    sorted_peers = sorted(peer_scores)
    peer_median = round((sorted_peers[4] + sorted_peers[5]) / 2, 1)
    sector_avg = round(peer_avg * 0.92, 1)
    below = sum(1 for s in peer_scores if s <= org_readiness)
    percentile = round(below / len(peer_scores) * 100)

    return BenchmarkResponse(
        org_id=org_id,
        org_readiness_pct=org_readiness,
        peer_average_pct=peer_avg,
        peer_median_pct=peer_median,
        sector_average_pct=sector_avg,
        percentile_rank=percentile,
        peer_count=len(peer_scores),
        by_category={
            "emissions_inventory": {"org": 88, "peer_avg": 82},
            "scope3_coverage": {"org": 62, "peer_avg": 58},
            "target_ambition": {"org": 85, "peer_avg": 78},
            "methodology": {"org": 90, "peer_avg": 80},
            "reporting": {"org": 65, "peer_avg": 72},
        },
        top_performers=[
            {"company": "SustainaCo", "readiness_pct": 95, "sbti_status": "validated"},
            {"company": "GreenTarget Inc", "readiness_pct": 92, "sbti_status": "validated"},
            {"company": "NetZero Corp", "readiness_pct": 88, "sbti_status": "committed"},
        ],
        generated_at=_now(),
    )
