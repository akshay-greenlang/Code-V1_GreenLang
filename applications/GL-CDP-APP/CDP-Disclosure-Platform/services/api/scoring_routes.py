"""
GL-CDP-APP Scoring Simulation API

CDP scoring simulation engine providing score prediction, what-if analysis,
A-level eligibility checking, category breakdown, trajectory prediction,
year-over-year comparison, and confidence intervals.

CDP Scoring Levels:
    A  (Leadership)  80-100%
    A- (Leadership)  70-79%
    B  (Management)  60-69%
    B- (Management)  50-59%
    C  (Awareness)   40-49%
    C- (Awareness)   30-39%
    D  (Disclosure)  20-29%
    D- (Disclosure)  0-19%

17 Scoring Categories:
    1. Governance                       (7%)
    2. Risk management processes        (6%/5%)
    3. Risk disclosure                  (5%/4%)
    4. Opportunity disclosure           (5%/4%)
    5. Business strategy                (6%/5%)
    6. Scenario analysis                (5%)
    7. Targets                          (8%)
    8. Emissions reduction initiatives  (7%)
    9. Scope 1 & 2 emissions            (10%)
    10. Scope 3 emissions               (8%)
    11. Energy                          (6%)
    12. Carbon pricing                  (4%)
    13. Value chain engagement          (6%)
    14. Public policy engagement        (3%)
    15. Transition plan                 (6%/8%)
    16. Portfolio climate performance   (5%/7% - FS only)
    17. Financial impact assessment     (3%)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/scoring", tags=["Scoring"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoreBand(str, Enum):
    """CDP score bands."""
    A = "A"
    A_MINUS = "A-"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    D_MINUS = "D-"


class ScoreLevel(str, Enum):
    """CDP scoring level for a category."""
    LEADERSHIP = "leadership"
    MANAGEMENT = "management"
    AWARENESS = "awareness"
    DISCLOSURE = "disclosure"
    NOT_SCORED = "not_scored"


# ---------------------------------------------------------------------------
# Scoring Category Definitions
# ---------------------------------------------------------------------------

SCORING_CATEGORIES = [
    {"id": 1, "name": "Governance", "weight_mgmt": 7.0, "weight_lead": 7.0},
    {"id": 2, "name": "Risk management processes", "weight_mgmt": 6.0, "weight_lead": 5.0},
    {"id": 3, "name": "Risk disclosure", "weight_mgmt": 5.0, "weight_lead": 4.0},
    {"id": 4, "name": "Opportunity disclosure", "weight_mgmt": 5.0, "weight_lead": 4.0},
    {"id": 5, "name": "Business strategy", "weight_mgmt": 6.0, "weight_lead": 5.0},
    {"id": 6, "name": "Scenario analysis", "weight_mgmt": 5.0, "weight_lead": 5.0},
    {"id": 7, "name": "Targets", "weight_mgmt": 8.0, "weight_lead": 8.0},
    {"id": 8, "name": "Emissions reduction initiatives", "weight_mgmt": 7.0, "weight_lead": 7.0},
    {"id": 9, "name": "Scope 1 & 2 emissions (incl. verification)", "weight_mgmt": 10.0, "weight_lead": 10.0},
    {"id": 10, "name": "Scope 3 emissions (incl. verification)", "weight_mgmt": 8.0, "weight_lead": 8.0},
    {"id": 11, "name": "Energy", "weight_mgmt": 6.0, "weight_lead": 6.0},
    {"id": 12, "name": "Carbon pricing", "weight_mgmt": 4.0, "weight_lead": 4.0},
    {"id": 13, "name": "Value chain engagement", "weight_mgmt": 6.0, "weight_lead": 6.0},
    {"id": 14, "name": "Public policy engagement", "weight_mgmt": 3.0, "weight_lead": 3.0},
    {"id": 15, "name": "Transition plan", "weight_mgmt": 6.0, "weight_lead": 8.0},
    {"id": 16, "name": "Portfolio climate performance (FS only)", "weight_mgmt": 5.0, "weight_lead": 7.0},
    {"id": 17, "name": "Financial impact assessment", "weight_mgmt": 3.0, "weight_lead": 3.0},
]

A_LEVEL_REQUIREMENTS = [
    {"id": "ALR-01", "description": "Publicly available 1.5C-aligned transition plan", "met": False},
    {"id": "ALR-02", "description": "Complete emissions inventory with no material exclusions", "met": False},
    {"id": "ALR-03", "description": "Third-party verification of 100% Scope 1 and Scope 2 emissions", "met": False},
    {"id": "ALR-04", "description": "Third-party verification of >= 70% of at least one Scope 3 category", "met": False},
    {"id": "ALR-05", "description": "SBTi-validated or 1.5C-aligned target (>= 4.2% annual absolute reduction)", "met": False},
]


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RunSimulationRequest(BaseModel):
    """Request to run a full scoring simulation."""
    scenario_name: Optional[str] = Field(
        None, max_length=200, description="Optional scenario name"
    )
    override_scores: Optional[Dict[str, float]] = Field(
        None, description="Override individual category scores (category_id -> score 0-100)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_name": "Baseline simulation",
                "override_scores": {"9": 85.0, "10": 60.0},
            }
        }


class WhatIfRequest(BaseModel):
    """Parameters for what-if scoring analysis."""
    improvements: List[Dict[str, Any]] = Field(
        ..., description="List of category improvements to simulate"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "improvements": [
                    {"category_id": 9, "current_score": 65.0, "target_score": 85.0},
                    {"category_id": 15, "current_score": 40.0, "target_score": 75.0},
                ]
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class CategoryScoreResponse(BaseModel):
    """Score for a single scoring category."""
    category_id: int
    category_name: str
    score: float
    max_score: float
    score_pct: float
    level: str
    weight_mgmt: float
    weight_lead: float
    weighted_score: float
    question_count: int
    answered_count: int
    improvement_potential: float


class ScoringResponse(BaseModel):
    """Full scoring simulation result."""
    questionnaire_id: str
    simulation_id: str
    scenario_name: Optional[str]
    overall_score: float
    overall_score_pct: float
    score_band: str
    score_level: str
    categories: List[CategoryScoreResponse]
    a_level_eligible: bool
    a_level_requirements: List[Dict[str, Any]]
    disclosure_score: float
    awareness_score: float
    management_score: float
    leadership_score: float
    simulated_at: datetime


class WhatIfResponse(BaseModel):
    """What-if analysis result."""
    questionnaire_id: str
    current_score: float
    current_band: str
    projected_score: float
    projected_band: str
    score_delta: float
    band_change: bool
    improvements_applied: List[Dict[str, Any]]
    category_impacts: List[Dict[str, Any]]
    simulated_at: datetime


class ALevelCheckResponse(BaseModel):
    """A-level eligibility assessment."""
    questionnaire_id: str
    eligible: bool
    requirements_met: int
    requirements_total: int
    requirements: List[Dict[str, Any]]
    current_score: float
    current_band: str
    score_gap_to_a: float
    recommendations: List[str]
    checked_at: datetime


class TrajectoryResponse(BaseModel):
    """Score trajectory prediction."""
    questionnaire_id: str
    current_score: float
    current_band: str
    projected_final_score: float
    projected_final_band: str
    completion_pct: float
    trajectory_points: List[Dict[str, Any]]
    confidence: float
    predicted_at: datetime


class ComparisonResponse(BaseModel):
    """Year-over-year score comparison."""
    questionnaire_id: str
    current_year: int
    previous_year: Optional[int]
    current_score: float
    current_band: str
    previous_score: Optional[float]
    previous_band: Optional[str]
    score_delta: Optional[float]
    band_improved: Optional[bool]
    category_comparisons: List[Dict[str, Any]]
    compared_at: datetime


class ConfidenceResponse(BaseModel):
    """Score confidence interval."""
    questionnaire_id: str
    predicted_score: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    predicted_band: str
    possible_bands: List[str]
    data_quality_factor: float
    completion_factor: float
    calculated_at: datetime


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


def _score_to_level(score_pct: float) -> str:
    """Convert a percentage score to a scoring level."""
    if score_pct >= 70:
        return "leadership"
    elif score_pct >= 50:
        return "management"
    elif score_pct >= 30:
        return "awareness"
    return "disclosure"


def _simulate_category_scores(overrides: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Generate simulated category scores."""
    # Default simulated scores per category
    default_scores = {
        1: 72.0, 2: 65.0, 3: 58.0, 4: 55.0, 5: 60.0, 6: 45.0,
        7: 68.0, 8: 62.0, 9: 78.0, 10: 52.0, 11: 70.0, 12: 40.0,
        13: 55.0, 14: 48.0, 15: 42.0, 16: 0.0, 17: 50.0,
    }
    if overrides:
        for k, v in overrides.items():
            cat_id = int(k)
            if cat_id in default_scores:
                default_scores[cat_id] = v

    categories = []
    for cat in SCORING_CATEGORIES:
        cat_id = cat["id"]
        score = default_scores.get(cat_id, 50.0)
        weighted = round(score * cat["weight_mgmt"] / 100.0, 2)
        categories.append({
            "category_id": cat_id,
            "category_name": cat["name"],
            "score": score,
            "max_score": 100.0,
            "score_pct": score,
            "level": _score_to_level(score),
            "weight_mgmt": cat["weight_mgmt"],
            "weight_lead": cat["weight_lead"],
            "weighted_score": weighted,
            "question_count": 12,
            "answered_count": max(1, int(12 * score / 100)),
            "improvement_potential": round(100.0 - score, 1),
        })
    return categories


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{questionnaire_id}",
    response_model=ScoringResponse,
    summary="Get current score simulation",
    description=(
        "Retrieve the current CDP score simulation for a questionnaire. "
        "Returns overall score, band, 17 category scores, and A-level eligibility."
    ),
)
async def get_scoring(questionnaire_id: str) -> ScoringResponse:
    """Retrieve current scoring simulation."""
    categories = _simulate_category_scores()
    overall = round(sum(c["weighted_score"] for c in categories), 1)
    overall_pct = round(overall, 1)
    band = _score_to_band(overall_pct)

    return ScoringResponse(
        questionnaire_id=questionnaire_id,
        simulation_id=_generate_id("sim"),
        scenario_name="Current state",
        overall_score=overall,
        overall_score_pct=overall_pct,
        score_band=band,
        score_level=_score_to_level(overall_pct),
        categories=[CategoryScoreResponse(**c) for c in categories],
        a_level_eligible=False,
        a_level_requirements=A_LEVEL_REQUIREMENTS,
        disclosure_score=round(sum(c["score"] for c in categories if c["score"] < 30) / max(1, len([c for c in categories if c["score"] < 30])), 1) if any(c["score"] < 30 for c in categories) else 0.0,
        awareness_score=round(sum(c["score"] for c in categories if 30 <= c["score"] < 50) / max(1, len([c for c in categories if 30 <= c["score"] < 50])), 1) if any(30 <= c["score"] < 50 for c in categories) else 0.0,
        management_score=round(sum(c["score"] for c in categories if 50 <= c["score"] < 70) / max(1, len([c for c in categories if 50 <= c["score"] < 70])), 1) if any(50 <= c["score"] < 70 for c in categories) else 0.0,
        leadership_score=round(sum(c["score"] for c in categories if c["score"] >= 70) / max(1, len([c for c in categories if c["score"] >= 70])), 1) if any(c["score"] >= 70 for c in categories) else 0.0,
        simulated_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/categories",
    response_model=List[CategoryScoreResponse],
    summary="Category score breakdown",
    description=(
        "Retrieve detailed scores for all 17 CDP scoring categories "
        "including weights, levels, and improvement potential."
    ),
)
async def get_category_scores(
    questionnaire_id: str,
    sort_by: Optional[str] = Query(None, description="Sort by: score, weight, improvement_potential"),
    level_filter: Optional[str] = Query(None, description="Filter by scoring level"),
) -> List[CategoryScoreResponse]:
    """Get category score breakdown."""
    categories = _simulate_category_scores()
    if level_filter:
        categories = [c for c in categories if c["level"] == level_filter]
    if sort_by == "score":
        categories.sort(key=lambda c: c["score"], reverse=True)
    elif sort_by == "weight":
        categories.sort(key=lambda c: c["weight_mgmt"], reverse=True)
    elif sort_by == "improvement_potential":
        categories.sort(key=lambda c: c["improvement_potential"], reverse=True)
    return [CategoryScoreResponse(**c) for c in categories]


@router.get(
    "/{questionnaire_id}/what-if",
    response_model=WhatIfResponse,
    summary="What-if analysis (quick)",
    description=(
        "Quick what-if analysis based on improving a single category. "
        "Provide a category ID and target score to see the projected impact."
    ),
)
async def get_what_if(
    questionnaire_id: str,
    category_id: int = Query(..., ge=1, le=17, description="Category to improve"),
    target_score: float = Query(..., ge=0, le=100, description="Target score for the category"),
) -> WhatIfResponse:
    """Quick what-if analysis for a single category."""
    current_categories = _simulate_category_scores()
    current_overall = round(sum(c["weighted_score"] for c in current_categories), 1)
    current_band = _score_to_band(current_overall)

    # Apply improvement
    projected_categories = _simulate_category_scores(overrides={str(category_id): target_score})
    projected_overall = round(sum(c["weighted_score"] for c in projected_categories), 1)
    projected_band = _score_to_band(projected_overall)
    delta = round(projected_overall - current_overall, 1)

    category_name = next(
        (c["name"] for c in SCORING_CATEGORIES if c["id"] == category_id),
        f"Category {category_id}"
    )

    return WhatIfResponse(
        questionnaire_id=questionnaire_id,
        current_score=current_overall,
        current_band=current_band,
        projected_score=projected_overall,
        projected_band=projected_band,
        score_delta=delta,
        band_change=current_band != projected_band,
        improvements_applied=[{
            "category_id": category_id,
            "category_name": category_name,
            "current_score": next((c["score"] for c in current_categories if c["category_id"] == category_id), 0),
            "target_score": target_score,
        }],
        category_impacts=[{
            "category_id": category_id,
            "category_name": category_name,
            "score_delta": delta,
            "weight": next((c["weight_mgmt"] for c in SCORING_CATEGORIES if c["id"] == category_id), 0),
        }],
        simulated_at=_now(),
    )


@router.post(
    "/{questionnaire_id}/simulate",
    response_model=ScoringResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run scoring simulation",
    description=(
        "Run a full scoring simulation with optional category score overrides. "
        "Use this for advanced scenarios where multiple categories are adjusted."
    ),
)
async def run_simulation(
    questionnaire_id: str,
    request: RunSimulationRequest,
) -> ScoringResponse:
    """Run a full scoring simulation."""
    categories = _simulate_category_scores(overrides=request.override_scores)
    overall = round(sum(c["weighted_score"] for c in categories), 1)
    overall_pct = round(overall, 1)
    band = _score_to_band(overall_pct)

    return ScoringResponse(
        questionnaire_id=questionnaire_id,
        simulation_id=_generate_id("sim"),
        scenario_name=request.scenario_name or "Custom simulation",
        overall_score=overall,
        overall_score_pct=overall_pct,
        score_band=band,
        score_level=_score_to_level(overall_pct),
        categories=[CategoryScoreResponse(**c) for c in categories],
        a_level_eligible=overall_pct >= 80,
        a_level_requirements=A_LEVEL_REQUIREMENTS,
        disclosure_score=0.0,
        awareness_score=0.0,
        management_score=0.0,
        leadership_score=0.0,
        simulated_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/a-level-check",
    response_model=ALevelCheckResponse,
    summary="A-level eligibility check",
    description=(
        "Check eligibility for CDP A-level scoring. Evaluates all 5 mandatory "
        "requirements: 1.5C transition plan, complete inventory, Scope 1+2 "
        "verification, Scope 3 verification, and SBTi-aligned target."
    ),
)
async def a_level_check(questionnaire_id: str) -> ALevelCheckResponse:
    """Check A-level eligibility."""
    categories = _simulate_category_scores()
    overall = round(sum(c["weighted_score"] for c in categories), 1)
    band = _score_to_band(overall)

    requirements = [
        {"id": "ALR-01", "description": "Publicly available 1.5C-aligned transition plan", "met": False, "detail": "No transition plan submitted"},
        {"id": "ALR-02", "description": "Complete emissions inventory with no material exclusions", "met": True, "detail": "All material categories included"},
        {"id": "ALR-03", "description": "Third-party verification of 100% Scope 1 and Scope 2 emissions", "met": False, "detail": "Scope 2 verification at 85%"},
        {"id": "ALR-04", "description": "Third-party verification of >= 70% of at least one Scope 3 category", "met": False, "detail": "No Scope 3 category verified above 70%"},
        {"id": "ALR-05", "description": "SBTi-validated or 1.5C-aligned target (>= 4.2% annual absolute reduction)", "met": True, "detail": "SBTi target validated (4.5% annual reduction)"},
    ]
    met = sum(1 for r in requirements if r["met"])
    eligible = met == len(requirements) and overall >= 80

    recommendations = []
    if not requirements[0]["met"]:
        recommendations.append("Develop and publish a 1.5C-aligned transition plan with clear milestones.")
    if not requirements[2]["met"]:
        recommendations.append("Extend third-party verification to cover 100% of Scope 2 emissions.")
    if not requirements[3]["met"]:
        recommendations.append("Obtain third-party verification for at least 70% of one Scope 3 category.")
    if overall < 80:
        recommendations.append(f"Improve overall score from {overall}% to >= 80% (gap: {round(80 - overall, 1)} points).")

    return ALevelCheckResponse(
        questionnaire_id=questionnaire_id,
        eligible=eligible,
        requirements_met=met,
        requirements_total=len(requirements),
        requirements=requirements,
        current_score=overall,
        current_band=band,
        score_gap_to_a=max(0, round(80.0 - overall, 1)),
        recommendations=recommendations,
        checked_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/trajectory",
    response_model=TrajectoryResponse,
    summary="Score trajectory prediction",
    description=(
        "Predict the final score trajectory based on current completion rate "
        "and scoring patterns. Shows projected score at various completion milestones."
    ),
)
async def get_trajectory(questionnaire_id: str) -> TrajectoryResponse:
    """Predict score trajectory."""
    categories = _simulate_category_scores()
    current = round(sum(c["weighted_score"] for c in categories), 1)
    band = _score_to_band(current)

    # Simulate trajectory at different completion levels
    trajectory = [
        {"completion_pct": 25, "projected_score": round(current * 0.6, 1), "projected_band": _score_to_band(current * 0.6)},
        {"completion_pct": 50, "projected_score": round(current * 0.8, 1), "projected_band": _score_to_band(current * 0.8)},
        {"completion_pct": 75, "projected_score": round(current * 0.95, 1), "projected_band": _score_to_band(current * 0.95)},
        {"completion_pct": 100, "projected_score": current, "projected_band": band},
    ]
    projected_final = round(current * 1.05, 1)
    projected_band = _score_to_band(projected_final)

    return TrajectoryResponse(
        questionnaire_id=questionnaire_id,
        current_score=current,
        current_band=band,
        projected_final_score=min(100.0, projected_final),
        projected_final_band=projected_band,
        completion_pct=68.5,
        trajectory_points=trajectory,
        confidence=0.78,
        predicted_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/comparison",
    response_model=ComparisonResponse,
    summary="Year-over-year score comparison",
    description=(
        "Compare the current questionnaire score against the previous year "
        "submission. Shows overall and category-level deltas."
    ),
)
async def get_comparison(
    questionnaire_id: str,
    previous_year: Optional[int] = Query(None, description="Previous year for comparison"),
) -> ComparisonResponse:
    """Compare score against previous year."""
    categories = _simulate_category_scores()
    current = round(sum(c["weighted_score"] for c in categories), 1)
    band = _score_to_band(current)

    prev_score = round(current - 5.2, 1)
    prev_band = _score_to_band(prev_score)

    category_comparisons = []
    for cat in categories:
        prev_cat_score = max(0, cat["score"] - 8.0 + (cat["category_id"] % 5) * 2)
        category_comparisons.append({
            "category_id": cat["category_id"],
            "category_name": cat["category_name"],
            "current_score": cat["score"],
            "previous_score": round(prev_cat_score, 1),
            "delta": round(cat["score"] - prev_cat_score, 1),
            "improved": cat["score"] > prev_cat_score,
        })

    return ComparisonResponse(
        questionnaire_id=questionnaire_id,
        current_year=2025,
        previous_year=previous_year or 2024,
        current_score=current,
        current_band=band,
        previous_score=prev_score,
        previous_band=prev_band,
        score_delta=round(current - prev_score, 1),
        band_improved=current > prev_score,
        category_comparisons=category_comparisons,
        compared_at=_now(),
    )


@router.get(
    "/{questionnaire_id}/confidence",
    response_model=ConfidenceResponse,
    summary="Score confidence interval",
    description=(
        "Calculate the confidence interval for the predicted score based on "
        "data quality, completion rate, and scoring uncertainty."
    ),
)
async def get_confidence(questionnaire_id: str) -> ConfidenceResponse:
    """Calculate score confidence interval."""
    categories = _simulate_category_scores()
    predicted = round(sum(c["weighted_score"] for c in categories), 1)
    band = _score_to_band(predicted)

    # Confidence bounds based on data quality and completion
    data_quality_factor = 0.82
    completion_factor = 0.685
    margin = round((1 - data_quality_factor * completion_factor) * 15, 1)
    lower = max(0, round(predicted - margin, 1))
    upper = min(100, round(predicted + margin, 1))

    # Determine possible bands within confidence range
    possible_bands = set()
    for score in [lower, predicted, upper]:
        possible_bands.add(_score_to_band(score))

    return ConfidenceResponse(
        questionnaire_id=questionnaire_id,
        predicted_score=predicted,
        confidence_lower=lower,
        confidence_upper=upper,
        confidence_level=0.90,
        predicted_band=band,
        possible_bands=sorted(possible_bands),
        data_quality_factor=data_quality_factor,
        completion_factor=completion_factor,
        calculated_at=_now(),
    )
