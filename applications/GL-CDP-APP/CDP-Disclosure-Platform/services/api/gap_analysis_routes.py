"""
GL-CDP-APP Gap Analysis API

Identifies gaps in CDP questionnaire responses, provides actionable
recommendations for score improvement, priority-ranks gaps by impact,
predicts score uplift, and tracks gap closure progress.

Gap Severity Levels:
    - critical: Missing mandatory response (disclosure-level gap)
    - high: Weak response missing management-level detail
    - medium: Missing best-practice elements (leadership-level gap)
    - low: Minor formatting or completeness improvement

Effort Levels:
    - low: < 2 hours, data already available
    - medium: 2-8 hours, requires data gathering
    - high: > 8 hours, requires new processes or verification
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/gaps", tags=["Gap Analysis"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GapSeverity(str, Enum):
    """Severity classification for a gap."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapEffort(str, Enum):
    """Estimated effort to close a gap."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GapStatus(str, Enum):
    """Gap resolution status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DEFERRED = "deferred"


class GapType(str, Enum):
    """Type of gap identified."""
    MISSING_RESPONSE = "missing_response"
    INCOMPLETE_RESPONSE = "incomplete_response"
    WEAK_EVIDENCE = "weak_evidence"
    MISSING_VERIFICATION = "missing_verification"
    MISSING_TARGET = "missing_target"
    MISSING_TRANSITION_PLAN = "missing_transition_plan"
    DATA_QUALITY = "data_quality"
    FORMAT_COMPLIANCE = "format_compliance"


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class GapItem(BaseModel):
    """A single identified gap."""
    gap_id: str
    questionnaire_id: str
    question_id: Optional[str]
    module_code: str
    scoring_category: str
    scoring_category_id: int
    gap_type: str
    severity: str
    effort: str
    status: str
    title: str
    description: str
    current_state: str
    target_state: str
    recommendation: str
    example_response: Optional[str]
    score_uplift_pct: float
    priority_rank: int
    created_at: datetime
    resolved_at: Optional[datetime]


class GapSummaryResponse(BaseModel):
    """Aggregated gap summary by severity."""
    questionnaire_id: str
    total_gaps: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    open_count: int
    in_progress_count: int
    resolved_count: int
    deferred_count: int
    total_uplift_potential_pct: float
    top_priority_category: str
    modules_with_gaps: List[Dict[str, Any]]
    generated_at: datetime


class RecommendationResponse(BaseModel):
    """Improvement recommendation."""
    recommendation_id: str
    gap_id: str
    module_code: str
    scoring_category: str
    title: str
    description: str
    action_items: List[str]
    effort: str
    score_uplift_pct: float
    priority: int
    related_questions: List[str]


class PriorityGapResponse(BaseModel):
    """Priority-ranked gap with impact analysis."""
    gap_id: str
    title: str
    severity: str
    effort: str
    score_uplift_pct: float
    impact_score: float
    priority_rank: int
    module_code: str
    scoring_category: str
    recommendation_summary: str


class UpliftPrediction(BaseModel):
    """Score uplift prediction for closing gaps."""
    questionnaire_id: str
    current_score: float
    current_band: str
    predicted_score_all_gaps: float
    predicted_band_all_gaps: str
    uplift_all_gaps: float
    predicted_score_critical_only: float
    predicted_band_critical_only: str
    uplift_critical_only: float
    predicted_score_high_priority: float
    predicted_band_high_priority: str
    uplift_high_priority: float
    gap_closure_scenarios: List[Dict[str, Any]]
    calculated_at: datetime


class GapProgressResponse(BaseModel):
    """Gap closure progress tracking."""
    questionnaire_id: str
    total_gaps: int
    resolved_gaps: int
    resolution_pct: float
    score_at_start: float
    current_score: float
    score_improvement: float
    weekly_progress: List[Dict[str, Any]]
    estimated_completion_date: Optional[str]
    on_track: bool
    calculated_at: datetime


class AnalyzeResponse(BaseModel):
    """Result of running a gap analysis."""
    analysis_id: str
    questionnaire_id: str
    gaps_identified: int
    critical_gaps: int
    high_gaps: int
    medium_gaps: int
    low_gaps: int
    total_uplift_potential: float
    top_recommendations: List[Dict[str, Any]]
    analyzed_at: datetime


# ---------------------------------------------------------------------------
# Simulated Gap Data
# ---------------------------------------------------------------------------

def _generate_gaps(questionnaire_id: str) -> List[Dict[str, Any]]:
    """Generate simulated gap analysis results."""
    now = datetime.utcnow()
    gaps = [
        {
            "gap_id": "gap_001", "questionnaire_id": questionnaire_id,
            "question_id": "q_m5_001", "module_code": "M5",
            "scoring_category": "Transition plan", "scoring_category_id": 15,
            "gap_type": GapType.MISSING_TRANSITION_PLAN.value,
            "severity": GapSeverity.CRITICAL.value, "effort": GapEffort.HIGH.value,
            "status": GapStatus.OPEN.value, "title": "Missing 1.5C-aligned transition plan",
            "description": "No transition plan has been submitted. This is mandatory for A-level scoring.",
            "current_state": "No transition plan documented",
            "target_state": "Publicly available 1.5C-aligned transition plan with milestones",
            "recommendation": "Develop a comprehensive transition plan with short/medium/long-term milestones, technology roadmap, and investment plan.",
            "example_response": "Our 1.5C-aligned transition plan targets net-zero by 2050 with interim milestones...",
            "score_uplift_pct": 6.8, "priority_rank": 1, "created_at": now, "resolved_at": None,
        },
        {
            "gap_id": "gap_002", "questionnaire_id": questionnaire_id,
            "question_id": "q_m7_015", "module_code": "M7",
            "scoring_category": "Scope 3 emissions", "scoring_category_id": 10,
            "gap_type": GapType.MISSING_VERIFICATION.value,
            "severity": GapSeverity.CRITICAL.value, "effort": GapEffort.HIGH.value,
            "status": GapStatus.OPEN.value, "title": "No Scope 3 third-party verification",
            "description": "No Scope 3 category has third-party verification. >= 70% of at least one category required for A-level.",
            "current_state": "0% Scope 3 verification coverage",
            "target_state": ">= 70% verification of at least one Scope 3 category",
            "recommendation": "Engage a third-party verifier for the largest Scope 3 category (typically Cat 1 or Cat 4).",
            "example_response": None,
            "score_uplift_pct": 5.2, "priority_rank": 2, "created_at": now, "resolved_at": None,
        },
        {
            "gap_id": "gap_003", "questionnaire_id": questionnaire_id,
            "question_id": "q_m4_005", "module_code": "M4",
            "scoring_category": "Scenario analysis", "scoring_category_id": 6,
            "gap_type": GapType.INCOMPLETE_RESPONSE.value,
            "severity": GapSeverity.HIGH.value, "effort": GapEffort.MEDIUM.value,
            "status": GapStatus.OPEN.value, "title": "Incomplete scenario analysis",
            "description": "Scenario analysis response lacks quantitative financial impact assessment.",
            "current_state": "Qualitative scenario analysis only",
            "target_state": "Quantitative scenario analysis with financial impact (TCFD-aligned)",
            "recommendation": "Add quantitative financial impact estimates for at least two climate scenarios (e.g., 1.5C and 3C).",
            "example_response": "Under a 1.5C scenario, we estimate a USD 50M revenue impact from carbon pricing by 2030...",
            "score_uplift_pct": 3.5, "priority_rank": 3, "created_at": now, "resolved_at": None,
        },
        {
            "gap_id": "gap_004", "questionnaire_id": questionnaire_id,
            "question_id": "q_m1_008", "module_code": "M1",
            "scoring_category": "Governance", "scoring_category_id": 1,
            "gap_type": GapType.WEAK_EVIDENCE.value,
            "severity": GapSeverity.HIGH.value, "effort": GapEffort.LOW.value,
            "status": GapStatus.IN_PROGRESS.value, "title": "Board oversight evidence insufficient",
            "description": "Board climate oversight described but no evidence of frequency, agenda items, or decisions made.",
            "current_state": "Generic board oversight statement",
            "target_state": "Specific meeting frequency, agenda items, and documented climate decisions",
            "recommendation": "Add specific details: board reviews climate quarterly, approved SBTi target in Q2 2025.",
            "example_response": "The Board reviews climate strategy quarterly. In 2025, the Board approved...",
            "score_uplift_pct": 2.1, "priority_rank": 4, "created_at": now, "resolved_at": None,
        },
        {
            "gap_id": "gap_005", "questionnaire_id": questionnaire_id,
            "question_id": "q_m6_010", "module_code": "M6",
            "scoring_category": "Emissions reduction initiatives", "scoring_category_id": 8,
            "gap_type": GapType.INCOMPLETE_RESPONSE.value,
            "severity": GapSeverity.MEDIUM.value, "effort": GapEffort.MEDIUM.value,
            "status": GapStatus.OPEN.value, "title": "Reduction initiatives missing financial data",
            "description": "Emissions reduction initiatives described but missing investment amounts and payback periods.",
            "current_state": "Initiatives listed without financial details",
            "target_state": "Each initiative with CapEx/OpEx, payback period, and annual savings",
            "recommendation": "For each initiative, add investment amount, expected payback period, and annual cost savings.",
            "example_response": None,
            "score_uplift_pct": 2.8, "priority_rank": 5, "created_at": now, "resolved_at": None,
        },
        {
            "gap_id": "gap_006", "questionnaire_id": questionnaire_id,
            "question_id": "q_m12_001", "module_code": "M12",
            "scoring_category": "Carbon pricing", "scoring_category_id": 12,
            "gap_type": GapType.MISSING_RESPONSE.value,
            "severity": GapSeverity.MEDIUM.value, "effort": GapEffort.LOW.value,
            "status": GapStatus.OPEN.value, "title": "Carbon pricing response not provided",
            "description": "Internal carbon pricing question not answered.",
            "current_state": "No response",
            "target_state": "Description of internal carbon price, how it is used in investment decisions",
            "recommendation": "If using internal carbon pricing, describe the price, scope, and application in capital allocation.",
            "example_response": "We use an internal carbon price of USD 85/tCO2e for all capital expenditure decisions...",
            "score_uplift_pct": 1.5, "priority_rank": 6, "created_at": now, "resolved_at": None,
        },
        {
            "gap_id": "gap_007", "questionnaire_id": questionnaire_id,
            "question_id": "q_m10_003", "module_code": "M10",
            "scoring_category": "Value chain engagement", "scoring_category_id": 13,
            "gap_type": GapType.INCOMPLETE_RESPONSE.value,
            "severity": GapSeverity.LOW.value, "effort": GapEffort.MEDIUM.value,
            "status": GapStatus.OPEN.value, "title": "Supplier engagement metrics missing",
            "description": "Supplier engagement described qualitatively but missing quantitative metrics.",
            "current_state": "Qualitative supplier engagement description",
            "target_state": "% suppliers engaged, % by emissions, targets set, improvements tracked",
            "recommendation": "Add supplier engagement rate, % of Scope 3 emissions covered, and year-over-year improvement.",
            "example_response": None,
            "score_uplift_pct": 1.2, "priority_rank": 7, "created_at": now, "resolved_at": None,
        },
    ]
    return gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _score_to_band(score_pct: float) -> str:
    """Convert a percentage score to a CDP band."""
    if score_pct >= 80:
        return "A"
    elif score_pct >= 70:
        return "A-"
    elif score_pct >= 60:
        return "B"
    elif score_pct >= 50:
        return "B-"
    elif score_pct >= 40:
        return "C"
    elif score_pct >= 30:
        return "C-"
    elif score_pct >= 20:
        return "D"
    return "D-"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{questionnaire_id}",
    response_model=List[GapItem],
    summary="Get all gaps",
    description=(
        "Retrieve all identified gaps for a questionnaire. Gaps are classified "
        "by severity, type, and effort. Supports filtering by module, severity, "
        "effort, status, and gap type."
    ),
)
async def get_gaps(
    questionnaire_id: str,
    module_code: Optional[str] = Query(None, description="Filter by module code"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    effort: Optional[str] = Query(None, description="Filter by effort level"),
    gap_status: Optional[str] = Query(None, alias="status", description="Filter by gap status"),
    gap_type: Optional[str] = Query(None, description="Filter by gap type"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
) -> List[GapItem]:
    """Retrieve all gaps for a questionnaire."""
    gaps = _generate_gaps(questionnaire_id)

    if module_code is not None:
        gaps = [g for g in gaps if g["module_code"] == module_code]
    if severity is not None:
        gaps = [g for g in gaps if g["severity"] == severity]
    if effort is not None:
        gaps = [g for g in gaps if g["effort"] == effort]
    if gap_status is not None:
        gaps = [g for g in gaps if g["status"] == gap_status]
    if gap_type is not None:
        gaps = [g for g in gaps if g["gap_type"] == gap_type]

    gaps.sort(key=lambda g: g["priority_rank"])
    page = gaps[offset: offset + limit]
    return [GapItem(**g) for g in page]


@router.get(
    "/{questionnaire_id}/summary",
    response_model=GapSummaryResponse,
    summary="Gap summary by severity",
    description=(
        "Retrieve an aggregated gap summary showing counts by severity, "
        "status, total uplift potential, and per-module breakdown."
    ),
)
async def get_gap_summary(questionnaire_id: str) -> GapSummaryResponse:
    """Retrieve gap summary."""
    gaps = _generate_gaps(questionnaire_id)
    now = _now()

    critical = sum(1 for g in gaps if g["severity"] == "critical")
    high = sum(1 for g in gaps if g["severity"] == "high")
    medium = sum(1 for g in gaps if g["severity"] == "medium")
    low = sum(1 for g in gaps if g["severity"] == "low")
    open_c = sum(1 for g in gaps if g["status"] == "open")
    in_prog = sum(1 for g in gaps if g["status"] == "in_progress")
    resolved = sum(1 for g in gaps if g["status"] == "resolved")
    deferred = sum(1 for g in gaps if g["status"] == "deferred")
    total_uplift = round(sum(g["score_uplift_pct"] for g in gaps), 1)

    # Per-module breakdown
    module_gaps: Dict[str, Dict[str, Any]] = {}
    for g in gaps:
        mc = g["module_code"]
        if mc not in module_gaps:
            module_gaps[mc] = {"module_code": mc, "gap_count": 0, "critical_count": 0, "uplift_pct": 0.0}
        module_gaps[mc]["gap_count"] += 1
        if g["severity"] == "critical":
            module_gaps[mc]["critical_count"] += 1
        module_gaps[mc]["uplift_pct"] = round(module_gaps[mc]["uplift_pct"] + g["score_uplift_pct"], 1)

    # Top priority category
    top_cat = max(gaps, key=lambda g: g["score_uplift_pct"])["scoring_category"] if gaps else ""

    return GapSummaryResponse(
        questionnaire_id=questionnaire_id,
        total_gaps=len(gaps),
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        low_count=low,
        open_count=open_c,
        in_progress_count=in_prog,
        resolved_count=resolved,
        deferred_count=deferred,
        total_uplift_potential_pct=total_uplift,
        top_priority_category=top_cat,
        modules_with_gaps=list(module_gaps.values()),
        generated_at=now,
    )


@router.get(
    "/{questionnaire_id}/recommendations",
    response_model=List[RecommendationResponse],
    summary="Improvement recommendations",
    description=(
        "Retrieve actionable improvement recommendations for all identified "
        "gaps, sorted by priority. Each recommendation includes action items, "
        "effort estimate, and expected score uplift."
    ),
)
async def get_recommendations(
    questionnaire_id: str,
    severity: Optional[str] = Query(None, description="Filter by gap severity"),
    effort: Optional[str] = Query(None, description="Filter by effort level"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[RecommendationResponse]:
    """Retrieve improvement recommendations."""
    gaps = _generate_gaps(questionnaire_id)
    if severity:
        gaps = [g for g in gaps if g["severity"] == severity]
    if effort:
        gaps = [g for g in gaps if g["effort"] == effort]
    gaps.sort(key=lambda g: g["score_uplift_pct"], reverse=True)

    recommendations = []
    for idx, g in enumerate(gaps[:limit]):
        recommendations.append(RecommendationResponse(
            recommendation_id=_generate_id("rec"),
            gap_id=g["gap_id"],
            module_code=g["module_code"],
            scoring_category=g["scoring_category"],
            title=f"Improve: {g['title']}",
            description=g["recommendation"],
            action_items=[
                f"Review current {g['module_code']} response against CDP guidance",
                f"Address gap: {g['description']}",
                f"Target state: {g['target_state']}",
            ],
            effort=g["effort"],
            score_uplift_pct=g["score_uplift_pct"],
            priority=idx + 1,
            related_questions=[g["question_id"]] if g["question_id"] else [],
        ))

    return recommendations


@router.get(
    "/{questionnaire_id}/priority",
    response_model=List[PriorityGapResponse],
    summary="Priority-ranked gaps",
    description=(
        "Retrieve gaps ranked by priority using an impact-effort matrix. "
        "Impact score combines severity, scoring weight, and uplift potential. "
        "Higher impact + lower effort = higher priority."
    ),
)
async def get_priority_gaps(
    questionnaire_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[PriorityGapResponse]:
    """Retrieve priority-ranked gaps."""
    gaps = _generate_gaps(questionnaire_id)

    severity_weight = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
    effort_weight = {"low": 3.0, "medium": 2.0, "high": 1.0}

    for g in gaps:
        g["impact_score"] = round(
            severity_weight.get(g["severity"], 1) * g["score_uplift_pct"]
            * effort_weight.get(g["effort"], 1),
            2,
        )

    gaps.sort(key=lambda g: g["impact_score"], reverse=True)

    return [
        PriorityGapResponse(
            gap_id=g["gap_id"],
            title=g["title"],
            severity=g["severity"],
            effort=g["effort"],
            score_uplift_pct=g["score_uplift_pct"],
            impact_score=g["impact_score"],
            priority_rank=idx + 1,
            module_code=g["module_code"],
            scoring_category=g["scoring_category"],
            recommendation_summary=g["recommendation"][:200],
        )
        for idx, g in enumerate(gaps[:limit])
    ]


@router.get(
    "/{questionnaire_id}/uplift",
    response_model=UpliftPrediction,
    summary="Score uplift predictions",
    description=(
        "Predict score improvement from closing gaps at different levels: "
        "all gaps, critical-only, and high-priority subset."
    ),
)
async def get_uplift(questionnaire_id: str) -> UpliftPrediction:
    """Predict score uplift from gap closure."""
    gaps = _generate_gaps(questionnaire_id)
    current_score = 58.7
    current_band = _score_to_band(current_score)

    uplift_all = round(sum(g["score_uplift_pct"] for g in gaps), 1)
    uplift_critical = round(sum(g["score_uplift_pct"] for g in gaps if g["severity"] == "critical"), 1)
    uplift_high_priority = round(sum(g["score_uplift_pct"] for g in gaps if g["severity"] in ("critical", "high")), 1)

    score_all = min(100.0, round(current_score + uplift_all, 1))
    score_critical = min(100.0, round(current_score + uplift_critical, 1))
    score_high = min(100.0, round(current_score + uplift_high_priority, 1))

    scenarios = [
        {"scenario": "Close all gaps", "predicted_score": score_all, "predicted_band": _score_to_band(score_all), "uplift": uplift_all, "gap_count": len(gaps)},
        {"scenario": "Close critical gaps", "predicted_score": score_critical, "predicted_band": _score_to_band(score_critical), "uplift": uplift_critical, "gap_count": sum(1 for g in gaps if g["severity"] == "critical")},
        {"scenario": "Close critical + high gaps", "predicted_score": score_high, "predicted_band": _score_to_band(score_high), "uplift": uplift_high_priority, "gap_count": sum(1 for g in gaps if g["severity"] in ("critical", "high"))},
        {"scenario": "Top 3 by priority", "predicted_score": min(100.0, round(current_score + sum(g["score_uplift_pct"] for g in sorted(gaps, key=lambda g: g["priority_rank"])[:3]), 1)), "predicted_band": _score_to_band(min(100.0, current_score + sum(g["score_uplift_pct"] for g in sorted(gaps, key=lambda g: g["priority_rank"])[:3]))), "uplift": round(sum(g["score_uplift_pct"] for g in sorted(gaps, key=lambda g: g["priority_rank"])[:3]), 1), "gap_count": 3},
    ]

    return UpliftPrediction(
        questionnaire_id=questionnaire_id,
        current_score=current_score,
        current_band=current_band,
        predicted_score_all_gaps=score_all,
        predicted_band_all_gaps=_score_to_band(score_all),
        uplift_all_gaps=uplift_all,
        predicted_score_critical_only=score_critical,
        predicted_band_critical_only=_score_to_band(score_critical),
        uplift_critical_only=uplift_critical,
        predicted_score_high_priority=score_high,
        predicted_band_high_priority=_score_to_band(score_high),
        uplift_high_priority=uplift_high_priority,
        gap_closure_scenarios=scenarios,
        calculated_at=_now(),
    )


@router.post(
    "/{questionnaire_id}/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run gap analysis",
    description=(
        "Run a fresh gap analysis against the current questionnaire responses. "
        "Compares each response against CDP scoring criteria and identifies "
        "gaps at each scoring level."
    ),
)
async def run_analysis(questionnaire_id: str) -> AnalyzeResponse:
    """Run a gap analysis."""
    gaps = _generate_gaps(questionnaire_id)
    now = _now()

    critical = sum(1 for g in gaps if g["severity"] == "critical")
    high = sum(1 for g in gaps if g["severity"] == "high")
    medium = sum(1 for g in gaps if g["severity"] == "medium")
    low = sum(1 for g in gaps if g["severity"] == "low")
    total_uplift = round(sum(g["score_uplift_pct"] for g in gaps), 1)

    top_recs = [
        {"title": g["title"], "severity": g["severity"], "uplift": g["score_uplift_pct"]}
        for g in sorted(gaps, key=lambda g: g["score_uplift_pct"], reverse=True)[:3]
    ]

    return AnalyzeResponse(
        analysis_id=_generate_id("ana"),
        questionnaire_id=questionnaire_id,
        gaps_identified=len(gaps),
        critical_gaps=critical,
        high_gaps=high,
        medium_gaps=medium,
        low_gaps=low,
        total_uplift_potential=total_uplift,
        top_recommendations=top_recs,
        analyzed_at=now,
    )


@router.get(
    "/{questionnaire_id}/progress",
    response_model=GapProgressResponse,
    summary="Gap closure progress",
    description=(
        "Track gap closure progress over time. Shows resolution rate, "
        "score improvement, and weekly progress trend."
    ),
)
async def get_gap_progress(questionnaire_id: str) -> GapProgressResponse:
    """Track gap closure progress."""
    gaps = _generate_gaps(questionnaire_id)
    resolved = sum(1 for g in gaps if g["status"] == "resolved")
    total = len(gaps)
    resolution_pct = round(resolved / total * 100, 1) if total > 0 else 0.0

    weekly_progress = [
        {"week": "2025-W44", "gaps_resolved": 0, "cumulative_resolved": 0, "score": 55.2},
        {"week": "2025-W45", "gaps_resolved": 0, "cumulative_resolved": 0, "score": 55.8},
        {"week": "2025-W46", "gaps_resolved": 1, "cumulative_resolved": 1, "score": 57.1},
        {"week": "2025-W47", "gaps_resolved": 0, "cumulative_resolved": 1, "score": 57.1},
        {"week": "2025-W48", "gaps_resolved": 0, "cumulative_resolved": 1, "score": 58.7},
    ]

    return GapProgressResponse(
        questionnaire_id=questionnaire_id,
        total_gaps=total,
        resolved_gaps=resolved,
        resolution_pct=resolution_pct,
        score_at_start=55.2,
        current_score=58.7,
        score_improvement=3.5,
        weekly_progress=weekly_progress,
        estimated_completion_date="2026-04-15",
        on_track=True,
        calculated_at=_now(),
    )
