"""
GL-Taxonomy-APP Data Quality API

Assesses and tracks data quality for EU Taxonomy reporting across
multiple dimensions.  Provides DQ scoring, evidence tracking,
improvement plan generation, and trend monitoring.

Data Quality Dimensions:
    - Completeness: All required data fields populated
    - Accuracy: Data values are correct and verifiable
    - Timeliness: Data reflects the current reporting period
    - Consistency: Data aligns across sources and periods
    - Granularity: Activity-level detail available (not just entity-level)
    - Verifiability: Third-party verification / audit trail

DQ Scoring (1-5 scale):
    5: Verified primary data from activity operator
    4: Primary data with partial verification
    3: High-quality secondary/modeled data
    2: Estimated/extrapolated data
    1: Proxy/screening-level data
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/data-quality", tags=["Data Quality"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class DQAssessRequest(BaseModel):
    """Full data quality assessment."""
    org_id: str = Field(...)
    reporting_year: int = Field(2025, ge=2022, le=2030)
    completeness_pct: float = Field(80, ge=0, le=100)
    accuracy_score: float = Field(3.5, ge=1, le=5)
    timeliness_score: float = Field(4.0, ge=1, le=5)
    consistency_score: float = Field(3.0, ge=1, le=5)
    granularity_score: float = Field(3.5, ge=1, le=5)
    verifiability_score: float = Field(2.5, ge=1, le=5)


class EvidenceTrackRequest(BaseModel):
    """Track evidence for data quality."""
    activity_code: str = Field(...)
    evidence_type: str = Field(..., description="primary_data, secondary_data, estimate, verification, audit")
    description: str = Field(..., max_length=2000)
    source: str = Field(..., max_length=500)
    dq_score: float = Field(..., ge=1, le=5)
    document_ref: Optional[str] = Field(None, max_length=300)


class ImprovementPlanRequest(BaseModel):
    """Generate data quality improvement plan."""
    target_score: float = Field(4.0, ge=1, le=5, description="Target overall DQ score")
    budget_eur: Optional[float] = Field(None, ge=0, description="Available budget for improvements")
    timeline_months: int = Field(12, ge=1, le=36)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class DQAssessmentResponse(BaseModel):
    """Data quality assessment result."""
    assessment_id: str
    org_id: str
    reporting_year: int
    overall_score: float
    overall_level: str
    dimensions: Dict[str, Dict[str, Any]]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    assessed_at: datetime


class DQDashboardResponse(BaseModel):
    """DQ dashboard."""
    org_id: str
    overall_score: float
    overall_level: str
    dimension_scores: Dict[str, float]
    activities_with_high_dq: int
    activities_with_low_dq: int
    total_evidence_items: int
    verification_coverage_pct: float
    generated_at: datetime


class DQDimensionsResponse(BaseModel):
    """DQ dimension scores."""
    org_id: str
    dimensions: List[Dict[str, Any]]
    overall_score: float
    target_score: float
    gap: float
    generated_at: datetime


class EvidenceTrackResponse(BaseModel):
    """Evidence tracking result."""
    evidence_id: str
    org_id: str
    activity_code: str
    evidence_type: str
    description: str
    source: str
    dq_score: float
    document_ref: Optional[str]
    tracked_at: datetime


class ImprovementPlanResponse(BaseModel):
    """DQ improvement plan."""
    plan_id: str
    org_id: str
    current_score: float
    target_score: float
    gap: float
    actions: List[Dict[str, Any]]
    total_actions: int
    estimated_cost_eur: float
    timeline_months: int
    expected_score_after: float
    generated_at: datetime


class DQTrendsResponse(BaseModel):
    """DQ trends over time."""
    org_id: str
    periods: List[Dict[str, Any]]
    trend: str
    improvement_rate: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_dq_assessments: Dict[str, Dict[str, Any]] = {}
_dq_evidence: Dict[str, List[Dict[str, Any]]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _dq_level(score: float) -> str:
    if score >= 4.5:
        return "excellent"
    elif score >= 3.5:
        return "good"
    elif score >= 2.5:
        return "moderate"
    elif score >= 1.5:
        return "low"
    return "very_low"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assess",
    response_model=DQAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Full data quality assessment",
    description="Run a comprehensive data quality assessment across all dimensions.",
)
async def assess_dq(request: DQAssessRequest) -> DQAssessmentResponse:
    """Full DQ assessment."""
    assessment_id = _generate_id("dq")

    dimension_scores = {
        "completeness": request.completeness_pct / 20,  # Convert 0-100 to 1-5
        "accuracy": request.accuracy_score,
        "timeliness": request.timeliness_score,
        "consistency": request.consistency_score,
        "granularity": request.granularity_score,
        "verifiability": request.verifiability_score,
    }

    overall = round(sum(dimension_scores.values()) / len(dimension_scores), 2)
    level = _dq_level(overall)

    dimensions = {}
    for dim, score in dimension_scores.items():
        dimensions[dim] = {"score": round(score, 2), "level": _dq_level(score), "weight": round(1 / len(dimension_scores), 2)}

    strengths = [dim for dim, score in dimension_scores.items() if score >= 4.0]
    weaknesses = [dim for dim, score in dimension_scores.items() if score < 3.0]

    recommendations = []
    if request.completeness_pct < 90:
        recommendations.append(f"Increase data completeness from {request.completeness_pct}% to 95%+")
    if request.verifiability_score < 3.0:
        recommendations.append("Obtain third-party verification for key taxonomy data")
    if request.granularity_score < 3.5:
        recommendations.append("Collect activity-level data rather than entity-level aggregates")
    if request.consistency_score < 3.5:
        recommendations.append("Implement cross-source data reconciliation checks")
    if not recommendations:
        recommendations.append("Data quality meets target thresholds across all dimensions")

    data = {
        "assessment_id": assessment_id, "org_id": request.org_id,
        "reporting_year": request.reporting_year,
        "overall_score": overall, "overall_level": level,
        "dimensions": dimensions, "strengths": strengths,
        "weaknesses": weaknesses, "recommendations": recommendations,
        "assessed_at": _now(),
    }
    _dq_assessments[assessment_id] = data
    return DQAssessmentResponse(**data)


@router.get(
    "/{org_id}/dashboard",
    response_model=DQDashboardResponse,
    summary="DQ dashboard",
    description="Get data quality dashboard with key metrics.",
)
async def get_dq_dashboard(org_id: str) -> DQDashboardResponse:
    """Get DQ dashboard."""
    return DQDashboardResponse(
        org_id=org_id, overall_score=3.4, overall_level="moderate",
        dimension_scores={
            "completeness": 4.0, "accuracy": 3.5, "timeliness": 4.0,
            "consistency": 3.0, "granularity": 3.5, "verifiability": 2.5,
        },
        activities_with_high_dq=8, activities_with_low_dq=5,
        total_evidence_items=45, verification_coverage_pct=60.0,
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/dimensions",
    response_model=DQDimensionsResponse,
    summary="Dimension scores",
    description="Get detailed dimension-level DQ scores.",
)
async def get_dimensions(org_id: str) -> DQDimensionsResponse:
    """Get DQ dimension scores."""
    dimensions = [
        {"dimension": "Completeness", "score": 4.0, "target": 4.5, "gap": 0.5, "priority": "medium"},
        {"dimension": "Accuracy", "score": 3.5, "target": 4.0, "gap": 0.5, "priority": "high"},
        {"dimension": "Timeliness", "score": 4.0, "target": 4.0, "gap": 0.0, "priority": "low"},
        {"dimension": "Consistency", "score": 3.0, "target": 4.0, "gap": 1.0, "priority": "high"},
        {"dimension": "Granularity", "score": 3.5, "target": 4.0, "gap": 0.5, "priority": "medium"},
        {"dimension": "Verifiability", "score": 2.5, "target": 4.0, "gap": 1.5, "priority": "critical"},
    ]
    overall = round(sum(d["score"] for d in dimensions) / len(dimensions), 2)
    target = 4.0
    return DQDimensionsResponse(
        org_id=org_id, dimensions=dimensions,
        overall_score=overall, target_score=target, gap=round(target - overall, 2),
        generated_at=_now(),
    )


@router.post(
    "/{org_id}/evidence",
    response_model=EvidenceTrackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Track evidence",
    description="Track evidence documentation for a specific activity.",
)
async def track_evidence(
    org_id: str,
    request: EvidenceTrackRequest,
) -> EvidenceTrackResponse:
    """Track evidence."""
    evidence_id = _generate_id("dqe")
    entry = {
        "evidence_id": evidence_id, "org_id": org_id,
        "activity_code": request.activity_code,
        "evidence_type": request.evidence_type,
        "description": request.description, "source": request.source,
        "dq_score": request.dq_score, "document_ref": request.document_ref,
        "tracked_at": _now(),
    }
    if org_id not in _dq_evidence:
        _dq_evidence[org_id] = []
    _dq_evidence[org_id].append(entry)
    return EvidenceTrackResponse(**entry)


@router.post(
    "/{org_id}/improvement-plan",
    response_model=ImprovementPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate improvement plan",
    description="Generate a data quality improvement plan with prioritized actions.",
)
async def generate_improvement_plan(
    org_id: str,
    request: ImprovementPlanRequest,
) -> ImprovementPlanResponse:
    """Generate DQ improvement plan."""
    plan_id = _generate_id("dqp")
    current = 3.4
    target = request.target_score
    gap = round(target - current, 2)

    actions = [
        {"action": "Obtain third-party verification for top 10 aligned activities", "dimension": "verifiability", "priority": "critical", "effort": "high", "cost_eur": 50000, "expected_improvement": 1.0},
        {"action": "Implement cross-source data reconciliation framework", "dimension": "consistency", "priority": "high", "effort": "medium", "cost_eur": 25000, "expected_improvement": 0.7},
        {"action": "Collect activity-level SC metrics from operators", "dimension": "granularity", "priority": "high", "effort": "high", "cost_eur": 30000, "expected_improvement": 0.5},
        {"action": "Upgrade data collection to primary source data", "dimension": "accuracy", "priority": "medium", "effort": "medium", "cost_eur": 15000, "expected_improvement": 0.4},
        {"action": "Complete missing data fields for 5 remaining activities", "dimension": "completeness", "priority": "medium", "effort": "low", "cost_eur": 8000, "expected_improvement": 0.3},
    ]

    total_cost = sum(a["cost_eur"] for a in actions)
    if request.budget_eur is not None:
        actions = [a for a in actions if sum(x["cost_eur"] for x in actions[:actions.index(a) + 1]) <= request.budget_eur]
        total_cost = sum(a["cost_eur"] for a in actions)

    expected_improvement = sum(a["expected_improvement"] for a in actions)
    expected_score = min(round(current + expected_improvement, 2), 5.0)

    return ImprovementPlanResponse(
        plan_id=plan_id, org_id=org_id,
        current_score=current, target_score=target, gap=gap,
        actions=actions, total_actions=len(actions),
        estimated_cost_eur=total_cost, timeline_months=request.timeline_months,
        expected_score_after=expected_score, generated_at=_now(),
    )


@router.get(
    "/{org_id}/trends",
    response_model=DQTrendsResponse,
    summary="DQ trends",
    description="Get data quality trends over time.",
)
async def get_dq_trends(org_id: str) -> DQTrendsResponse:
    """Get DQ trends."""
    periods = [
        {"period": "2022", "overall_score": 2.2, "completeness": 2.5, "verifiability": 1.5},
        {"period": "2023", "overall_score": 2.8, "completeness": 3.2, "verifiability": 2.0},
        {"period": "2024", "overall_score": 3.1, "completeness": 3.8, "verifiability": 2.3},
        {"period": "2025", "overall_score": 3.4, "completeness": 4.0, "verifiability": 2.5},
    ]

    improvement = round(periods[-1]["overall_score"] - periods[0]["overall_score"], 2)
    rate = round(improvement / (len(periods) - 1), 2)

    return DQTrendsResponse(
        org_id=org_id, periods=periods,
        trend="improving" if improvement > 0 else "stable",
        improvement_rate=rate, generated_at=_now(),
    )
