"""
GL-CDP-APP Dashboard API

Provides pre-aggregated metrics and KPIs for the CDP disclosure dashboard.
Returns score simulation, module progress, gap summary, timeline countdown,
recent activity, and readiness metrics -- all in efficient endpoints for
frontend rendering.

Dashboard widgets:
    - CDP score gauge (D- to A) with predicted band
    - Module completion progress bars (13 modules)
    - Gap count by severity
    - Submission deadline countdown
    - Readiness score (% answered, % reviewed, % approved)
    - Year-over-year score trend
    - Category radar chart data (17 categories)
    - A-level eligibility checklist
    - Recent activity feed
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

router = APIRouter(prefix="/api/v1/cdp/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ScoreOverview(BaseModel):
    """Current score overview for dashboard."""
    predicted_score: float
    predicted_band: str
    score_level: str
    previous_year_score: Optional[float]
    previous_year_band: Optional[str]
    score_delta: Optional[float]
    band_improved: Optional[bool]
    a_level_eligible: bool
    a_level_requirements_met: int
    a_level_requirements_total: int
    category_scores: List[Dict[str, Any]]
    updated_at: datetime


class ModuleProgressItem(BaseModel):
    """Module completion progress."""
    module_code: str
    module_name: str
    total_questions: int
    answered_questions: int
    completion_pct: float
    draft_count: int
    in_review_count: int
    approved_count: int
    is_applicable: bool


class ModuleProgressResponse(BaseModel):
    """Module completion progress overview."""
    org_id: str
    questionnaire_id: str
    overall_completion_pct: float
    total_questions: int
    answered_questions: int
    modules: List[ModuleProgressItem]
    updated_at: datetime


class TimelineResponse(BaseModel):
    """Submission timeline and deadlines."""
    org_id: str
    questionnaire_year: str
    submission_deadline: str
    days_until_deadline: int
    is_overdue: bool
    milestones: List[Dict[str, Any]]
    current_phase: str
    phase_progress_pct: float
    updated_at: datetime


class ActivityItem(BaseModel):
    """Recent activity feed item."""
    activity_id: str
    activity_type: str
    description: str
    actor: str
    module_code: Optional[str]
    question_number: Optional[str]
    timestamp: datetime


class ActivityFeedResponse(BaseModel):
    """Recent activity feed."""
    org_id: str
    activities: List[ActivityItem]
    total_count: int
    updated_at: datetime


class ReadinessResponse(BaseModel):
    """Submission readiness assessment."""
    org_id: str
    questionnaire_id: str
    overall_readiness_pct: float
    readiness_grade: str
    completion_pct: float
    review_pct: float
    approval_pct: float
    evidence_pct: float
    verification_pct: float
    checklist_pass_pct: float
    critical_blockers: List[str]
    readiness_by_module: List[Dict[str, Any]]
    estimated_score: float
    estimated_band: str
    updated_at: datetime


class DashboardResponse(BaseModel):
    """Complete dashboard data."""
    org_id: str
    questionnaire_id: str
    questionnaire_year: str
    reporting_year: int
    score: ScoreOverview
    progress: ModuleProgressResponse
    timeline: TimelineResponse
    readiness: ReadinessResponse
    gap_summary: Dict[str, Any]
    recent_activity: List[ActivityItem]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.utcnow()


def _score_to_band(score_pct: float) -> str:
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


# Simulated module progress data
MODULE_PROGRESS = [
    {"code": "M0", "name": "Introduction", "total": 15, "answered": 15, "draft": 0, "review": 0, "approved": 15, "applicable": True},
    {"code": "M1", "name": "Governance", "total": 20, "answered": 18, "draft": 2, "review": 4, "approved": 12, "applicable": True},
    {"code": "M2", "name": "Policies & Commitments", "total": 15, "answered": 12, "draft": 3, "review": 2, "approved": 7, "applicable": True},
    {"code": "M3", "name": "Risks & Opportunities", "total": 25, "answered": 18, "draft": 5, "review": 3, "approved": 10, "applicable": True},
    {"code": "M4", "name": "Strategy", "total": 20, "answered": 13, "draft": 4, "review": 2, "approved": 7, "applicable": True},
    {"code": "M5", "name": "Transition Plans", "total": 20, "answered": 8, "draft": 5, "review": 1, "approved": 2, "applicable": True},
    {"code": "M6", "name": "Implementation", "total": 20, "answered": 15, "draft": 3, "review": 2, "approved": 10, "applicable": True},
    {"code": "M7", "name": "Environmental Performance", "total": 35, "answered": 31, "draft": 2, "review": 5, "approved": 24, "applicable": True},
    {"code": "M8", "name": "Forests", "total": 15, "answered": 0, "draft": 0, "review": 0, "approved": 0, "applicable": False},
    {"code": "M9", "name": "Water Security", "total": 15, "answered": 0, "draft": 0, "review": 0, "approved": 0, "applicable": False},
    {"code": "M10", "name": "Supply Chain", "total": 15, "answered": 9, "draft": 3, "review": 2, "approved": 4, "applicable": True},
    {"code": "M11", "name": "Additional Metrics", "total": 10, "answered": 5, "draft": 2, "review": 1, "approved": 2, "applicable": True},
    {"code": "M12", "name": "Financial Services", "total": 20, "answered": 0, "draft": 0, "review": 0, "approved": 0, "applicable": False},
    {"code": "M13", "name": "Sign Off", "total": 5, "answered": 0, "draft": 0, "review": 0, "approved": 0, "applicable": True},
]

CATEGORY_SCORES = [
    {"id": 1, "name": "Governance", "score": 72.0},
    {"id": 2, "name": "Risk management processes", "score": 65.0},
    {"id": 3, "name": "Risk disclosure", "score": 58.0},
    {"id": 4, "name": "Opportunity disclosure", "score": 55.0},
    {"id": 5, "name": "Business strategy", "score": 60.0},
    {"id": 6, "name": "Scenario analysis", "score": 45.0},
    {"id": 7, "name": "Targets", "score": 68.0},
    {"id": 8, "name": "Emissions reduction initiatives", "score": 62.0},
    {"id": 9, "name": "Scope 1 & 2 emissions", "score": 78.0},
    {"id": 10, "name": "Scope 3 emissions", "score": 52.0},
    {"id": 11, "name": "Energy", "score": 70.0},
    {"id": 12, "name": "Carbon pricing", "score": 40.0},
    {"id": 13, "name": "Value chain engagement", "score": 55.0},
    {"id": 14, "name": "Public policy engagement", "score": 48.0},
    {"id": 15, "name": "Transition plan", "score": 42.0},
    {"id": 16, "name": "Portfolio climate performance", "score": 0.0},
    {"id": 17, "name": "Financial impact assessment", "score": 50.0},
]

RECENT_ACTIVITIES = [
    {"type": "response_approved", "desc": "M7.12 Scope 1 emissions response approved", "actor": "Dr. Alice Chen", "module": "M7", "qn": "M7.12"},
    {"type": "evidence_attached", "desc": "Verification statement attached to M7.15", "actor": "Jane Smith", "module": "M7", "qn": "M7.15"},
    {"type": "response_updated", "desc": "M1.5 Board oversight response updated", "actor": "John Doe", "module": "M1", "qn": "M1.5"},
    {"type": "review_submitted", "desc": "M3.8 Risk assessment review completed", "actor": "Dr. Alice Chen", "module": "M3", "qn": "M3.8"},
    {"type": "response_created", "desc": "M5.2 Transition plan milestones drafted", "actor": "Jane Smith", "module": "M5", "qn": "M5.2"},
    {"type": "scoring_simulated", "desc": "Scoring simulation run - predicted B-", "actor": "System", "module": None, "qn": None},
    {"type": "gap_resolved", "desc": "Gap: Board oversight evidence resolved", "actor": "John Doe", "module": "M1", "qn": "M1.8"},
    {"type": "supplier_responded", "desc": "Acme Materials Corp submitted CDP response", "actor": "Supply Chain", "module": "M10", "qn": None},
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}",
    response_model=DashboardResponse,
    summary="Full dashboard data",
    description=(
        "Retrieve all dashboard data for an organization in a single call. "
        "Includes score, progress, timeline, readiness, gap summary, and activity."
    ),
)
async def get_full_dashboard(
    org_id: str,
    questionnaire_id: Optional[str] = Query(None, description="Specific questionnaire ID"),
    reporting_year: int = Query(2025, ge=2020, le=2100, description="Reporting year"),
) -> DashboardResponse:
    """Retrieve complete dashboard data."""
    now = _now()
    qid = questionnaire_id or f"cdpq_{org_id}_2025"

    # Score
    score = _build_score_overview(now)

    # Progress
    progress = _build_module_progress(org_id, qid, now)

    # Timeline
    timeline = _build_timeline(org_id, now)

    # Readiness
    readiness = _build_readiness(org_id, qid, now)

    # Gap summary
    gap_summary = {
        "total_gaps": 7,
        "critical": 2,
        "high": 2,
        "medium": 2,
        "low": 1,
        "open": 5,
        "in_progress": 1,
        "resolved": 1,
        "total_uplift_pct": 23.1,
    }

    # Activity
    activities = _build_activity_feed(now)

    return DashboardResponse(
        org_id=org_id,
        questionnaire_id=qid,
        questionnaire_year="2026",
        reporting_year=reporting_year,
        score=score,
        progress=progress,
        timeline=timeline,
        readiness=readiness,
        gap_summary=gap_summary,
        recent_activity=activities[:5],
        generated_at=now,
    )


@router.get(
    "/{org_id}/score",
    response_model=ScoreOverview,
    summary="Current score overview",
    description="Retrieve the current CDP score simulation with band, category breakdown, and A-level status.",
)
async def get_score_overview(org_id: str) -> ScoreOverview:
    """Get score overview."""
    return _build_score_overview(_now())


@router.get(
    "/{org_id}/progress",
    response_model=ModuleProgressResponse,
    summary="Module completion progress",
    description="Retrieve per-module completion progress with question counts and status breakdown.",
)
async def get_progress(
    org_id: str,
    questionnaire_id: Optional[str] = Query(None, description="Specific questionnaire ID"),
) -> ModuleProgressResponse:
    """Get module completion progress."""
    qid = questionnaire_id or f"cdpq_{org_id}_2025"
    return _build_module_progress(org_id, qid, _now())


@router.get(
    "/{org_id}/timeline",
    response_model=TimelineResponse,
    summary="Submission timeline",
    description="Retrieve submission timeline with deadline countdown and phase progress.",
)
async def get_timeline(org_id: str) -> TimelineResponse:
    """Get submission timeline."""
    return _build_timeline(org_id, _now())


@router.get(
    "/{org_id}/activity",
    response_model=ActivityFeedResponse,
    summary="Recent activity feed",
    description="Retrieve the recent activity feed for the CDP disclosure.",
)
async def get_activity(
    org_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum activities"),
) -> ActivityFeedResponse:
    """Get recent activity feed."""
    now = _now()
    activities = _build_activity_feed(now)
    return ActivityFeedResponse(
        org_id=org_id,
        activities=activities[:limit],
        total_count=len(activities),
        updated_at=now,
    )


@router.get(
    "/{org_id}/readiness",
    response_model=ReadinessResponse,
    summary="Readiness score",
    description=(
        "Assess overall submission readiness including completion, review, "
        "approval, evidence, verification, and checklist pass rates."
    ),
)
async def get_readiness(
    org_id: str,
    questionnaire_id: Optional[str] = Query(None, description="Specific questionnaire ID"),
) -> ReadinessResponse:
    """Get readiness score."""
    qid = questionnaire_id or f"cdpq_{org_id}_2025"
    return _build_readiness(org_id, qid, _now())


# ---------------------------------------------------------------------------
# Internal Builders
# ---------------------------------------------------------------------------

def _build_score_overview(now: datetime) -> ScoreOverview:
    """Build score overview."""
    return ScoreOverview(
        predicted_score=58.7,
        predicted_band="B-",
        score_level="management",
        previous_year_score=53.5,
        previous_year_band="B-",
        score_delta=5.2,
        band_improved=False,
        a_level_eligible=False,
        a_level_requirements_met=2,
        a_level_requirements_total=5,
        category_scores=[
            {"category_id": c["id"], "category_name": c["name"], "score": c["score"]}
            for c in CATEGORY_SCORES
        ],
        updated_at=now,
    )


def _build_module_progress(org_id: str, qid: str, now: datetime) -> ModuleProgressResponse:
    """Build module progress."""
    modules = []
    total_q = 0
    total_a = 0
    for m in MODULE_PROGRESS:
        pct = round(m["answered"] / max(1, m["total"]) * 100, 1)
        modules.append(ModuleProgressItem(
            module_code=m["code"], module_name=m["name"],
            total_questions=m["total"], answered_questions=m["answered"],
            completion_pct=pct, draft_count=m["draft"],
            in_review_count=m["review"], approved_count=m["approved"],
            is_applicable=m["applicable"],
        ))
        if m["applicable"]:
            total_q += m["total"]
            total_a += m["answered"]

    overall_pct = round(total_a / max(1, total_q) * 100, 1)

    return ModuleProgressResponse(
        org_id=org_id,
        questionnaire_id=qid,
        overall_completion_pct=overall_pct,
        total_questions=total_q,
        answered_questions=total_a,
        modules=modules,
        updated_at=now,
    )


def _build_timeline(org_id: str, now: datetime) -> TimelineResponse:
    """Build timeline."""
    deadline = datetime(2026, 7, 31)
    days_left = max(0, (deadline - now).days)

    milestones = [
        {"label": "Questionnaire opened", "date": "2026-02-01", "status": "completed"},
        {"label": "Data collection start", "date": "2026-02-15", "status": "completed"},
        {"label": "Module M7 data freeze", "date": "2026-04-30", "status": "in_progress"},
        {"label": "Internal review deadline", "date": "2026-05-31", "status": "upcoming"},
        {"label": "C-suite sign-off", "date": "2026-06-30", "status": "upcoming"},
        {"label": "CDP submission deadline", "date": "2026-07-31", "status": "upcoming"},
    ]

    return TimelineResponse(
        org_id=org_id,
        questionnaire_year="2026",
        submission_deadline="2026-07-31",
        days_until_deadline=days_left,
        is_overdue=days_left <= 0,
        milestones=milestones,
        current_phase="data_collection",
        phase_progress_pct=65.0,
        updated_at=now,
    )


def _build_readiness(org_id: str, qid: str, now: datetime) -> ReadinessResponse:
    """Build readiness assessment."""
    completion = 72.5
    review = 45.0
    approval = 38.0
    evidence = 52.0
    verification = 65.0
    checklist_pass = 65.0

    overall = round(
        (completion * 0.25 + review * 0.20 + approval * 0.20 +
         evidence * 0.15 + verification * 0.10 + checklist_pass * 0.10),
        1
    )

    grade = "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 60 else "D" if overall >= 40 else "F"

    blockers = []
    if completion < 80:
        blockers.append(f"Questionnaire completion at {completion}% (target: 80%)")
    if approval < 50:
        blockers.append(f"Response approval at {approval}% (target: 50%)")
    if verification < 70:
        blockers.append(f"Verification coverage at {verification}% (target: 70%)")

    module_readiness = []
    for m in MODULE_PROGRESS:
        if not m["applicable"]:
            continue
        m_completion = round(m["answered"] / max(1, m["total"]) * 100, 1)
        m_approval = round(m["approved"] / max(1, m["answered"]) * 100, 1) if m["answered"] > 0 else 0.0
        module_readiness.append({
            "module_code": m["code"],
            "module_name": m["name"],
            "completion_pct": m_completion,
            "approval_pct": m_approval,
            "readiness_pct": round((m_completion * 0.6 + m_approval * 0.4), 1),
        })

    return ReadinessResponse(
        org_id=org_id,
        questionnaire_id=qid,
        overall_readiness_pct=overall,
        readiness_grade=grade,
        completion_pct=completion,
        review_pct=review,
        approval_pct=approval,
        evidence_pct=evidence,
        verification_pct=verification,
        checklist_pass_pct=checklist_pass,
        critical_blockers=blockers,
        readiness_by_module=module_readiness,
        estimated_score=58.7,
        estimated_band="B-",
        updated_at=now,
    )


def _build_activity_feed(now: datetime) -> List[ActivityItem]:
    """Build activity feed."""
    activities = []
    for i, act in enumerate(RECENT_ACTIVITIES):
        activities.append(ActivityItem(
            activity_id=f"act_{i + 1:03d}",
            activity_type=act["type"],
            description=act["desc"],
            actor=act["actor"],
            module_code=act["module"],
            question_number=act["qn"],
            timestamp=now - timedelta(hours=i * 3 + 1),
        ))
    return activities
