"""
Five-Year Review Engine -- SBTi Target Review Lifecycle Management

Implements the five-year target review cycle required by SBTi criterion C23.
Manages trigger date calculation, review deadlines, notification scheduling,
readiness assessment, progress summaries, outcome recording, and deadline
monitoring across all organizational targets.

SBTi requires targets to be reviewed at least every five years from the date
of validation.  Companies must revalidate their targets against the latest
SBTi criteria within 12 months of the trigger date.

All date calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Criteria and Recommendations v5.1 (2023), Criterion C23
    - SBTi Corporate Net-Zero Standard v1.2, Section 8
    - SBTi Target Validation Protocol v2.0

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = FiveYearReviewEngine(SBTiAppConfig())
    >>> from datetime import date
    >>> trigger = engine.calculate_trigger_date(date(2021, 6, 15))
    >>> print(trigger)  # 2026-06-15
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    NotificationType,
    ReviewOutcome,
    SBTiAppConfig,
    SBTI_MINIMUM_AMBITION,
)
from .models import (
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ReviewRecord(BaseModel):
    """In-memory five-year review record."""

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    target_id: str = Field(...)
    target_name: str = Field(default="")
    validation_date: date = Field(...)
    trigger_date: date = Field(...)
    review_deadline: date = Field(...)
    status: str = Field(default="scheduled")
    outcome: Optional[ReviewOutcome] = Field(None)
    progress_summary: Optional[str] = Field(None)
    updated_criteria_met: bool = Field(default=False)
    notes: Optional[str] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)


class ReviewStatus(BaseModel):
    """Current status snapshot of a five-year review."""

    review_id: str = Field(...)
    status: str = Field(default="scheduled")
    days_until_trigger: int = Field(default=0)
    days_until_deadline: int = Field(default=0)
    is_overdue: bool = Field(default=False)
    is_in_review_window: bool = Field(default=False)
    completion_pct: float = Field(default=0.0)
    checked_at: datetime = Field(default_factory=_now)


class ReadinessAssessment(BaseModel):
    """Assessment of readiness for five-year review submission."""

    review_id: str = Field(...)
    is_ready: bool = Field(default=False)
    criteria_checks: List[Dict[str, Any]] = Field(default_factory=list)
    met_count: int = Field(default=0)
    total_criteria: int = Field(default=0)
    blocking_issues: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)


class ProgressSummary(BaseModel):
    """Progress summary for five-year review submission."""

    review_id: str = Field(...)
    org_id: str = Field(...)
    target_name: str = Field(default="")
    base_year_emissions: float = Field(default=0.0)
    current_emissions: float = Field(default=0.0)
    reduction_achieved_pct: float = Field(default=0.0)
    target_reduction_pct: float = Field(default=0.0)
    on_track: bool = Field(default=False)
    annual_reduction_rate: float = Field(default=0.0)
    required_annual_rate: float = Field(default=0.0)
    key_actions_taken: List[str] = Field(default_factory=list)
    years_elapsed: int = Field(default=0)
    generated_at: datetime = Field(default_factory=_now)


class Notification(BaseModel):
    """Automated notification for review lifecycle events."""

    id: str = Field(default_factory=_new_id)
    review_id: str = Field(...)
    org_id: str = Field(...)
    notification_type: NotificationType = Field(...)
    scheduled_date: date = Field(...)
    message: str = Field(default="")
    sent: bool = Field(default=False)
    sent_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)


class DeadlineAlert(BaseModel):
    """Alert for upcoming or overdue review deadlines."""

    review_id: str = Field(...)
    org_id: str = Field(...)
    target_id: str = Field(...)
    deadline: date = Field(...)
    days_remaining: int = Field(default=0)
    is_overdue: bool = Field(default=False)
    severity: str = Field(default="info")
    message: str = Field(default="")


# ---------------------------------------------------------------------------
# FiveYearReviewEngine
# ---------------------------------------------------------------------------

class FiveYearReviewEngine:
    """
    Five-year target review lifecycle engine per SBTi criterion C23.

    Manages review scheduling, deadline tracking, notification scheduling,
    readiness assessment, progress summaries, and outcome recording.

    Attributes:
        config: Application configuration.
        _reviews: In-memory store keyed by review ID.
        _org_reviews: Index by org_id -> list of review IDs.
        _notifications: In-memory notification store keyed by review ID.
        _target_data: Supplementary target data for progress summaries.

    Example:
        >>> engine = FiveYearReviewEngine(SBTiAppConfig())
        >>> review = engine.create_review("org-1", "tgt-1", date(2021, 6, 15))
        >>> status = engine.get_review_status(review.id)
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the FiveYearReviewEngine."""
        self.config = config or SBTiAppConfig()
        self._reviews: Dict[str, ReviewRecord] = {}
        self._org_reviews: Dict[str, List[str]] = {}
        self._notifications: Dict[str, List[Notification]] = {}
        self._target_data: Dict[str, Dict[str, Any]] = {}
        logger.info("FiveYearReviewEngine initialized")

    # ------------------------------------------------------------------
    # Date calculations
    # ------------------------------------------------------------------

    def calculate_trigger_date(self, validation_date: date) -> date:
        """
        Calculate the five-year review trigger date from validation.

        Args:
            validation_date: Date the target was validated by SBTi.

        Returns:
            Date exactly 5 years after validation.
        """
        try:
            trigger = validation_date.replace(year=validation_date.year + 5)
        except ValueError:
            # Handle Feb 29 -> Feb 28
            trigger = validation_date.replace(
                year=validation_date.year + 5, day=28
            )
        logger.debug(
            "Trigger date for validation %s: %s",
            validation_date.isoformat(), trigger.isoformat(),
        )
        return trigger

    def calculate_review_deadline(self, trigger_date: date) -> date:
        """
        Calculate the review deadline from the trigger date.

        The review must be completed within the configured review window
        (default 12 months) after the trigger date.

        Args:
            trigger_date: The five-year review trigger date.

        Returns:
            Deadline date (trigger + review_window_months).
        """
        months = self.config.review_window_months
        year_add = months // 12
        month_add = months % 12
        new_month = trigger_date.month + month_add
        if new_month > 12:
            year_add += 1
            new_month -= 12
        try:
            deadline = trigger_date.replace(
                year=trigger_date.year + year_add, month=new_month
            )
        except ValueError:
            # Handle end-of-month overflow
            import calendar
            max_day = calendar.monthrange(
                trigger_date.year + year_add, new_month
            )[1]
            deadline = trigger_date.replace(
                year=trigger_date.year + year_add,
                month=new_month,
                day=min(trigger_date.day, max_day),
            )
        logger.debug(
            "Review deadline for trigger %s: %s",
            trigger_date.isoformat(), deadline.isoformat(),
        )
        return deadline

    # ------------------------------------------------------------------
    # Review lifecycle
    # ------------------------------------------------------------------

    def create_review(
        self,
        org_id: str,
        target_id: str,
        validation_date: date,
        target_name: str = "",
    ) -> ReviewRecord:
        """
        Create a five-year review record for a validated target.

        Args:
            org_id: Organization identifier.
            target_id: Target identifier.
            validation_date: Date the target was validated by SBTi.
            target_name: Optional display name for the target.

        Returns:
            ReviewRecord with computed trigger date and deadline.
        """
        trigger = self.calculate_trigger_date(validation_date)
        deadline = self.calculate_review_deadline(trigger)

        provenance = _sha256(
            f"{org_id}:{target_id}:{validation_date.isoformat()}"
        )

        review = ReviewRecord(
            org_id=org_id,
            target_id=target_id,
            target_name=target_name,
            validation_date=validation_date,
            trigger_date=trigger,
            review_deadline=deadline,
            provenance_hash=provenance,
        )

        self._reviews[review.id] = review
        self._org_reviews.setdefault(org_id, []).append(review.id)

        logger.info(
            "Created review %s for org %s / target %s: "
            "trigger=%s, deadline=%s",
            review.id, org_id, target_id,
            trigger.isoformat(), deadline.isoformat(),
        )
        return review

    def get_review_status(self, review_id: str) -> ReviewStatus:
        """
        Get the current status of a five-year review.

        Args:
            review_id: Review identifier.

        Returns:
            ReviewStatus with days until trigger/deadline and overdue flag.

        Raises:
            ValueError: If review_id is not found.
        """
        review = self._reviews.get(review_id)
        if review is None:
            raise ValueError(f"Review {review_id} not found")

        today = date.today()
        days_to_trigger = (review.trigger_date - today).days
        days_to_deadline = (review.review_deadline - today).days
        is_overdue = days_to_deadline < 0 and review.outcome is None
        in_window = days_to_trigger <= 0 and days_to_deadline >= 0

        # Determine completion percentage
        if review.outcome is not None:
            completion = 100.0
        elif in_window:
            total_window = (review.review_deadline - review.trigger_date).days
            elapsed = (today - review.trigger_date).days
            completion = min(90.0, (elapsed / max(total_window, 1)) * 100.0)
        elif days_to_trigger > 0:
            total_wait = (review.trigger_date - review.created_at.date()).days
            elapsed = (today - review.created_at.date()).days
            completion = min(20.0, (elapsed / max(total_wait, 1)) * 20.0)
        else:
            completion = 0.0

        status_str = review.status
        if review.outcome is not None:
            status_str = "completed"
        elif is_overdue:
            status_str = "overdue"
        elif in_window:
            status_str = "in_review_window"
        elif days_to_trigger > 365:
            status_str = "scheduled"
        else:
            status_str = "approaching"

        return ReviewStatus(
            review_id=review_id,
            status=status_str,
            days_until_trigger=max(0, days_to_trigger),
            days_until_deadline=days_to_deadline,
            is_overdue=is_overdue,
            is_in_review_window=in_window,
            completion_pct=round(completion, 1),
        )

    def assess_review_readiness(
        self, review_id: str,
    ) -> ReadinessAssessment:
        """
        Check whether the organization meets updated SBTi criteria for review.

        Evaluates criteria such as updated base year data, pathway alignment
        with latest science, scope coverage, and documentation completeness.

        Args:
            review_id: Review identifier.

        Returns:
            ReadinessAssessment with per-criteria results.

        Raises:
            ValueError: If review_id is not found.
        """
        review = self._reviews.get(review_id)
        if review is None:
            raise ValueError(f"Review {review_id} not found")

        target_data = self._target_data.get(review.target_id, {})
        checks: List[Dict[str, Any]] = []
        blocking: List[str] = []

        # Check 1: Base year emissions data available
        has_base = target_data.get("base_year_emissions", 0) > 0
        checks.append({
            "criterion": "Base year emissions data",
            "met": has_base,
            "description": "Verified base year emissions inventory available",
        })
        if not has_base:
            blocking.append("Missing base year emissions data")

        # Check 2: Current year emissions tracked
        has_current = target_data.get("current_emissions", 0) > 0
        checks.append({
            "criterion": "Current emissions tracking",
            "met": has_current,
            "description": "Most recent annual emissions data available",
        })
        if not has_current:
            blocking.append("No current year emissions data")

        # Check 3: Scope 1+2 coverage >= 95%
        s12_coverage = target_data.get("scope_1_2_coverage_pct", 0)
        s12_met = s12_coverage >= 95.0
        checks.append({
            "criterion": "Scope 1+2 coverage >= 95%",
            "met": s12_met,
            "value": s12_coverage,
        })
        if not s12_met:
            blocking.append(f"Scope 1+2 coverage is {s12_coverage}% (need 95%)")

        # Check 4: 1.5C-aligned reduction rate
        annual_rate = target_data.get("annual_reduction_rate", 0)
        rate_met = annual_rate >= 4.2
        checks.append({
            "criterion": "1.5C-aligned annual reduction rate >= 4.2%",
            "met": rate_met,
            "value": annual_rate,
        })
        if not rate_met:
            blocking.append(
                f"Annual rate {annual_rate}% is below 4.2% minimum"
            )

        # Check 5: Recalculation policy documented
        has_policy = target_data.get("recalculation_policy", False)
        checks.append({
            "criterion": "Recalculation policy documented",
            "met": has_policy,
        })
        if not has_policy:
            blocking.append("No documented recalculation policy")

        # Check 6: Third-party verification
        verified = target_data.get("verified", False)
        checks.append({
            "criterion": "Third-party verification of inventory",
            "met": verified,
        })

        met_count = sum(1 for c in checks if c.get("met", False))
        total = len(checks)
        is_ready = len(blocking) == 0

        assessment = ReadinessAssessment(
            review_id=review_id,
            is_ready=is_ready,
            criteria_checks=checks,
            met_count=met_count,
            total_criteria=total,
            blocking_issues=blocking,
        )

        logger.info(
            "Readiness assessment for review %s: ready=%s (%d/%d met)",
            review_id, is_ready, met_count, total,
        )
        return assessment

    def generate_progress_summary(
        self, review_id: str,
    ) -> ProgressSummary:
        """
        Generate a progress summary for a five-year review submission.

        Args:
            review_id: Review identifier.

        Returns:
            ProgressSummary with reduction achieved and on-track assessment.

        Raises:
            ValueError: If review_id is not found.
        """
        review = self._reviews.get(review_id)
        if review is None:
            raise ValueError(f"Review {review_id} not found")

        target_data = self._target_data.get(review.target_id, {})
        base_emissions = target_data.get("base_year_emissions", 100000.0)
        current_emissions = target_data.get("current_emissions", 90000.0)
        target_reduction = target_data.get("target_reduction_pct", 42.0)
        base_year = target_data.get("base_year", 2020)
        target_year = target_data.get("target_year", 2030)

        # Compute reduction achieved
        reduction_pct = 0.0
        if base_emissions > 0:
            reduction_pct = (
                (base_emissions - current_emissions) / base_emissions * 100.0
            )

        # Compute annual rate
        years_elapsed = date.today().year - base_year
        annual_rate = (reduction_pct / years_elapsed) if years_elapsed > 0 else 0.0

        # Required annual rate to stay on track
        total_years = target_year - base_year
        required_rate = (target_reduction / total_years) if total_years > 0 else 0.0

        on_track = reduction_pct >= (required_rate * years_elapsed)

        actions = target_data.get("key_actions", [
            "Implemented energy efficiency measures",
            "Increased renewable energy procurement",
            "Engaged top suppliers on emission reductions",
        ])

        summary = ProgressSummary(
            review_id=review_id,
            org_id=review.org_id,
            target_name=review.target_name,
            base_year_emissions=round(base_emissions, 2),
            current_emissions=round(current_emissions, 2),
            reduction_achieved_pct=round(reduction_pct, 2),
            target_reduction_pct=round(target_reduction, 2),
            on_track=on_track,
            annual_reduction_rate=round(annual_rate, 2),
            required_annual_rate=round(required_rate, 2),
            key_actions_taken=actions,
            years_elapsed=years_elapsed,
        )

        logger.info(
            "Progress summary for review %s: reduction=%.1f%%, on_track=%s",
            review_id, reduction_pct, on_track,
        )
        return summary

    def schedule_notifications(
        self, review_id: str,
    ) -> List[Notification]:
        """
        Schedule automated notifications at configured lead times before deadline.

        Default alerts at 12, 6, 3, and 1 month(s) before the review deadline.

        Args:
            review_id: Review identifier.

        Returns:
            List of scheduled Notification objects.

        Raises:
            ValueError: If review_id is not found.
        """
        review = self._reviews.get(review_id)
        if review is None:
            raise ValueError(f"Review {review_id} not found")

        lead_months = self.config.notification_lead_months
        type_map = {
            12: NotificationType.REVIEW_12_MONTHS,
            6: NotificationType.REVIEW_6_MONTHS,
            3: NotificationType.REVIEW_3_MONTHS,
            1: NotificationType.REVIEW_1_MONTH,
        }

        notifications: List[Notification] = []
        for months in sorted(lead_months, reverse=True):
            notif_date = self._subtract_months(review.review_deadline, months)
            notif_type = type_map.get(months, NotificationType.REVIEW_3_MONTHS)

            message = (
                f"SBTi five-year review for target '{review.target_name}' "
                f"is due in {months} month(s) (deadline: "
                f"{review.review_deadline.isoformat()}). "
                f"Please ensure all updated criteria are met."
            )

            notif = Notification(
                review_id=review_id,
                org_id=review.org_id,
                notification_type=notif_type,
                scheduled_date=notif_date,
                message=message,
            )
            notifications.append(notif)

        self._notifications[review_id] = notifications

        logger.info(
            "Scheduled %d notifications for review %s",
            len(notifications), review_id,
        )
        return notifications

    def record_review_outcome(
        self,
        review_id: str,
        outcome: ReviewOutcome,
        notes: str = "",
    ) -> ReviewRecord:
        """
        Record the outcome of a completed five-year review.

        Args:
            review_id: Review identifier.
            outcome: Review outcome (renewed, updated, expired).
            notes: Optional notes on the review outcome.

        Returns:
            Updated ReviewRecord with outcome and completion timestamp.

        Raises:
            ValueError: If review_id is not found.
        """
        review = self._reviews.get(review_id)
        if review is None:
            raise ValueError(f"Review {review_id} not found")

        review.outcome = outcome
        review.status = "completed"
        review.completed_at = _now()
        review.notes = notes

        logger.info(
            "Recorded outcome for review %s: %s", review_id, outcome.value,
        )
        return review

    def get_upcoming_reviews(
        self, org_id: str,
    ) -> List[ReviewRecord]:
        """
        Get all upcoming (not completed) reviews for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            List of ReviewRecord objects sorted by deadline.
        """
        review_ids = self._org_reviews.get(org_id, [])
        upcoming = [
            self._reviews[rid] for rid in review_ids
            if rid in self._reviews and self._reviews[rid].outcome is None
        ]
        upcoming.sort(key=lambda r: r.review_deadline)

        logger.info(
            "Found %d upcoming reviews for org %s", len(upcoming), org_id,
        )
        return upcoming

    def get_review_history(
        self, org_id: str,
    ) -> List[ReviewRecord]:
        """
        Get the full review history for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            List of all ReviewRecord objects sorted by creation date.
        """
        review_ids = self._org_reviews.get(org_id, [])
        history = [
            self._reviews[rid] for rid in review_ids
            if rid in self._reviews
        ]
        history.sort(key=lambda r: r.created_at, reverse=True)

        logger.info(
            "Retrieved %d review records for org %s", len(history), org_id,
        )
        return history

    def check_all_review_deadlines(self) -> List[DeadlineAlert]:
        """
        Check all review deadlines across all organizations.

        Generates alerts for reviews that are overdue, approaching deadline
        (within 90 days), or within the review window.

        Returns:
            List of DeadlineAlert objects sorted by urgency.
        """
        today = date.today()
        alerts: List[DeadlineAlert] = []

        for review in self._reviews.values():
            if review.outcome is not None:
                continue

            days_remaining = (review.review_deadline - today).days

            if days_remaining < 0:
                severity = "critical"
                message = (
                    f"OVERDUE: Review for target '{review.target_name}' "
                    f"was due {abs(days_remaining)} days ago."
                )
            elif days_remaining <= 30:
                severity = "high"
                message = (
                    f"URGENT: Review for target '{review.target_name}' "
                    f"is due in {days_remaining} days."
                )
            elif days_remaining <= 90:
                severity = "medium"
                message = (
                    f"APPROACHING: Review for target '{review.target_name}' "
                    f"is due in {days_remaining} days."
                )
            elif days_remaining <= 365:
                severity = "low"
                message = (
                    f"UPCOMING: Review for target '{review.target_name}' "
                    f"is due in {days_remaining} days."
                )
            else:
                continue  # Too far out to alert

            alerts.append(DeadlineAlert(
                review_id=review.id,
                org_id=review.org_id,
                target_id=review.target_id,
                deadline=review.review_deadline,
                days_remaining=days_remaining,
                is_overdue=days_remaining < 0,
                severity=severity,
                message=message,
            ))

        # Sort by urgency (overdue first, then by days remaining)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda a: (
            severity_order.get(a.severity, 4), a.days_remaining
        ))

        logger.info(
            "Checked deadlines: %d alerts generated across %d active reviews",
            len(alerts), sum(1 for r in self._reviews.values() if r.outcome is None),
        )
        return alerts

    # ------------------------------------------------------------------
    # Target data helpers
    # ------------------------------------------------------------------

    def register_target_data(
        self, target_id: str, data: Dict[str, Any],
    ) -> None:
        """
        Register supplementary target data for progress summaries.

        Args:
            target_id: Target identifier.
            data: Dict with target fields (base_year_emissions, etc.).
        """
        self._target_data[target_id] = data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _subtract_months(d: date, months: int) -> date:
        """Subtract months from a date, handling month boundaries."""
        import calendar
        month = d.month - months
        year = d.year
        while month <= 0:
            month += 12
            year -= 1
        max_day = calendar.monthrange(year, month)[1]
        return d.replace(year=year, month=month, day=min(d.day, max_day))
