"""
GL-SBTi-APP Five-Year Review API

Manages the SBTi mandatory five-year target review cycle.  After SBTi
validation, companies must re-assess and update their targets at least
every five years to ensure continued alignment with the latest climate
science and SBTi criteria.

Five-Year Review Requirements:
    - Targets must be reviewed within 5 years of validation date
    - Review may result in: reaffirm, strengthen, or withdraw
    - Must incorporate latest SBTi criteria version
    - Must reflect significant business changes
    - Failure to review within deadline may result in removal from SBTi
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/sbti/reviews", tags=["Five-Year Review"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReviewOutcome(str, Enum):
    """Possible review outcomes."""
    REAFFIRMED = "reaffirmed"
    STRENGTHENED = "strengthened"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class ReviewStatus(str, Enum):
    """Review lifecycle status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateReviewRequest(BaseModel):
    """Request to create a five-year review."""
    org_id: str = Field(...)
    target_id: str = Field(...)
    original_validation_date: str = Field(..., description="ISO date of original SBTi validation")
    review_due_date: str = Field(..., description="ISO date of review deadline")
    assigned_reviewer: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=5000)


class RecordOutcomeRequest(BaseModel):
    """Request to record review outcome."""
    outcome: ReviewOutcome = Field(...)
    new_target_reduction_pct: Optional[float] = Field(None, ge=0, le=100)
    new_target_year: Optional[int] = Field(None, ge=2025, le=2055)
    criteria_version: str = Field("v2.1", description="SBTi criteria version applied")
    justification: str = Field(..., max_length=2000)
    reviewer: str = Field(..., max_length=200)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ReviewResponse(BaseModel):
    """Five-year review record."""
    review_id: str
    org_id: str
    target_id: str
    original_validation_date: str
    review_due_date: str
    status: str
    outcome: Optional[str]
    assigned_reviewer: Optional[str]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class ReviewReadinessResponse(BaseModel):
    """Review readiness assessment."""
    review_id: str
    readiness_score: float
    readiness_level: str
    checklist: List[Dict[str, Any]]
    blocking_items: List[str]
    estimated_weeks_to_ready: int
    generated_at: datetime


class ReviewHistoryResponse(BaseModel):
    """Review history for an organization."""
    org_id: str
    reviews: List[Dict[str, Any]]
    total_count: int
    generated_at: datetime


class UpcomingReviewResponse(BaseModel):
    """Upcoming reviews for an organization."""
    org_id: str
    upcoming: List[Dict[str, Any]]
    overdue: List[Dict[str, Any]]
    total_upcoming: int
    total_overdue: int
    generated_at: datetime


class DeadlineAlertResponse(BaseModel):
    """Deadline alert summary."""
    alerts: List[Dict[str, Any]]
    total_alerts: int
    critical_count: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_reviews: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create five-year review",
    description=(
        "Create a five-year review record for a science-based target. "
        "Scheduled reviews track the mandatory review deadline and "
        "assigned reviewer."
    ),
)
async def create_review(request: CreateReviewRequest) -> ReviewResponse:
    """Create a five-year review."""
    review_id = _generate_id("rev")
    now = _now()
    data = {
        "review_id": review_id,
        "org_id": request.org_id,
        "target_id": request.target_id,
        "original_validation_date": request.original_validation_date,
        "review_due_date": request.review_due_date,
        "status": ReviewStatus.SCHEDULED.value,
        "outcome": None,
        "assigned_reviewer": request.assigned_reviewer,
        "notes": request.notes,
        "created_at": now,
        "updated_at": now,
    }
    _reviews[review_id] = data
    return ReviewResponse(**data)


@router.get(
    "/{review_id}",
    response_model=ReviewResponse,
    summary="Get review details",
    description="Retrieve a five-year review record by its ID.",
)
async def get_review(review_id: str) -> ReviewResponse:
    """Get review details."""
    review = _reviews.get(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found",
        )
    return ReviewResponse(**review)


@router.get(
    "/org/{org_id}/upcoming",
    response_model=UpcomingReviewResponse,
    summary="Upcoming reviews",
    description="Get upcoming and overdue five-year reviews for an organization.",
)
async def get_upcoming_reviews(org_id: str) -> UpcomingReviewResponse:
    """Get upcoming reviews."""
    now = _now()
    org_reviews = [r for r in _reviews.values() if r["org_id"] == org_id]

    upcoming = []
    overdue = []
    for r in org_reviews:
        if r["status"] in (ReviewStatus.COMPLETED.value,):
            continue
        try:
            due = datetime.fromisoformat(r["review_due_date"])
        except (ValueError, TypeError):
            due = now
        entry = {
            "review_id": r["review_id"],
            "target_id": r["target_id"],
            "due_date": r["review_due_date"],
            "days_until_due": (due - now).days,
            "status": r["status"],
        }
        if due < now:
            overdue.append(entry)
        else:
            upcoming.append(entry)

    upcoming.sort(key=lambda x: x.get("days_until_due", 0))

    return UpcomingReviewResponse(
        org_id=org_id,
        upcoming=upcoming,
        overdue=overdue,
        total_upcoming=len(upcoming),
        total_overdue=len(overdue),
        generated_at=now,
    )


@router.get(
    "/{review_id}/readiness",
    response_model=ReviewReadinessResponse,
    summary="Review readiness",
    description="Assess readiness for completing a five-year review.",
)
async def get_review_readiness(review_id: str) -> ReviewReadinessResponse:
    """Assess review readiness."""
    review = _reviews.get(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found",
        )

    checklist = [
        {"item": "Updated emissions inventory available", "complete": True, "required": True},
        {"item": "Latest SBTi criteria version reviewed", "complete": True, "required": True},
        {"item": "Progress data for all years compiled", "complete": True, "required": True},
        {"item": "Structural changes documented", "complete": False, "required": True},
        {"item": "Base year recalculation assessed", "complete": True, "required": True},
        {"item": "Scope 3 screening updated", "complete": False, "required": True},
        {"item": "Target ambition re-assessed", "complete": False, "required": True},
        {"item": "Management sign-off obtained", "complete": False, "required": True},
    ]

    completed = sum(1 for c in checklist if c["complete"])
    total_required = sum(1 for c in checklist if c["required"])
    completed_required = sum(1 for c in checklist if c["required"] and c["complete"])
    score = round(completed_required / total_required * 100, 1) if total_required > 0 else 0.0

    blockers = [c["item"] for c in checklist if c["required"] and not c["complete"]]
    weeks = max(round((total_required - completed_required) * 2), 2)

    if score >= 90:
        level = "ready"
    elif score >= 70:
        level = "nearly_ready"
    elif score >= 50:
        level = "partial"
    else:
        level = "not_ready"

    return ReviewReadinessResponse(
        review_id=review_id,
        readiness_score=score,
        readiness_level=level,
        checklist=checklist,
        blocking_items=blockers,
        estimated_weeks_to_ready=weeks,
        generated_at=_now(),
    )


@router.put(
    "/{review_id}/outcome",
    response_model=ReviewResponse,
    summary="Record review outcome",
    description=(
        "Record the outcome of a five-year review: reaffirmed (target "
        "remains), strengthened (target increased), or withdrawn."
    ),
)
async def record_outcome(
    review_id: str,
    request: RecordOutcomeRequest,
) -> ReviewResponse:
    """Record review outcome."""
    review = _reviews.get(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found",
        )

    review["outcome"] = request.outcome.value
    review["status"] = ReviewStatus.COMPLETED.value
    review["updated_at"] = _now()

    if request.new_target_reduction_pct is not None:
        review["new_target_reduction_pct"] = request.new_target_reduction_pct
    if request.new_target_year is not None:
        review["new_target_year"] = request.new_target_year
    review["criteria_version"] = request.criteria_version
    review["justification"] = request.justification
    review["reviewer"] = request.reviewer

    return ReviewResponse(**{k: v for k, v in review.items() if k in ReviewResponse.model_fields})


@router.get(
    "/org/{org_id}/history",
    response_model=ReviewHistoryResponse,
    summary="Review history",
    description="Get the full review history for an organization.",
)
async def get_review_history(
    org_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> ReviewHistoryResponse:
    """Get review history."""
    records = [r for r in _reviews.values() if r["org_id"] == org_id]
    records.sort(key=lambda r: r["created_at"], reverse=True)

    return ReviewHistoryResponse(
        org_id=org_id,
        reviews=records[:limit],
        total_count=len(records),
        generated_at=_now(),
    )


@router.get(
    "/deadlines/alerts",
    response_model=DeadlineAlertResponse,
    summary="Deadline alerts",
    description=(
        "Get deadline alerts for all pending reviews across organizations. "
        "Returns reviews approaching deadline (within 6 months) and overdue."
    ),
)
async def get_deadline_alerts() -> DeadlineAlertResponse:
    """Get deadline alerts."""
    now = _now()
    alerts = []

    for r in _reviews.values():
        if r["status"] == ReviewStatus.COMPLETED.value:
            continue
        try:
            due = datetime.fromisoformat(r["review_due_date"])
        except (ValueError, TypeError):
            continue

        days_until = (due - now).days
        if days_until <= 180:  # Within 6 months
            severity = "critical" if days_until <= 0 else "warning" if days_until <= 90 else "info"
            alerts.append({
                "review_id": r["review_id"],
                "org_id": r["org_id"],
                "target_id": r["target_id"],
                "due_date": r["review_due_date"],
                "days_until_due": days_until,
                "severity": severity,
                "message": f"Review {'overdue by ' + str(abs(days_until)) + ' days' if days_until < 0 else 'due in ' + str(days_until) + ' days'}",
            })

    critical = sum(1 for a in alerts if a["severity"] == "critical")
    alerts.sort(key=lambda a: a["days_until_due"])

    return DeadlineAlertResponse(
        alerts=alerts,
        total_alerts=len(alerts),
        critical_count=critical,
        generated_at=now,
    )
