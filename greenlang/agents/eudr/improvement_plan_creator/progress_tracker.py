# -*- coding: utf-8 -*-
"""
Progress Tracker Engine - AGENT-EUDR-035: Improvement Plan Creator

Tracks improvement plan execution progress through milestone monitoring,
action status aggregation, overdue detection, effectiveness scoring,
and point-in-time snapshot generation. Supports auto-escalation for
overdue actions and extension management.

Zero-Hallucination:
    - Progress percentages are weighted Decimal arithmetic
    - Overdue detection uses pure date comparison
    - Effectiveness scores are deterministic averages
    - No LLM involvement in progress calculations

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import ImprovementPlanCreatorConfig, get_config
from .models import (
    AGENT_ID,
    ActionStatus,
    ImprovementAction,
    ImprovementPlan,
    PlanStatus,
    ProgressMilestone,
    ProgressSnapshot,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)

# Completed-equivalent statuses
_COMPLETED_STATUSES = {
    ActionStatus.COMPLETED,
    ActionStatus.VERIFIED,
    ActionStatus.CLOSED,
}

# In-progress statuses
_IN_PROGRESS_STATUSES = {
    ActionStatus.IN_PROGRESS,
    ActionStatus.APPROVED,
}


class ProgressTracker:
    """Tracks improvement plan execution progress.

    Monitors action statuses, generates progress snapshots, detects
    overdue actions, computes weighted completion percentages, and
    manages milestone tracking with auto-escalation support.

    Example:
        >>> engine = ProgressTracker()
        >>> snapshot = await engine.capture_snapshot(plan)
        >>> assert Decimal("0") <= snapshot.overall_progress <= Decimal("100")
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize ProgressTracker.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._snapshot_store: Dict[str, List[ProgressSnapshot]] = {}
        self._milestone_store: Dict[str, List[ProgressMilestone]] = {}
        logger.info("ProgressTracker initialized")

    async def capture_snapshot(
        self, plan: ImprovementPlan
    ) -> ProgressSnapshot:
        """Capture a point-in-time progress snapshot.

        Args:
            plan: Improvement plan to snapshot.

        Returns:
            ProgressSnapshot with completion metrics.
        """
        start = time.monotonic()

        actions = plan.actions
        total = len(actions)
        completed = sum(1 for a in actions if a.status in _COMPLETED_STATUSES)
        in_progress = sum(1 for a in actions if a.status in _IN_PROGRESS_STATUSES)
        on_hold = sum(1 for a in actions if a.status == ActionStatus.ON_HOLD)
        overdue = self._count_overdue(actions)

        # Calculate weighted progress
        if total > 0:
            progress = (Decimal(str(completed)) / Decimal(str(total)) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            progress = Decimal("0")

        # Gaps closed estimation (based on action coverage)
        gap_ids = set(a.gap_id for a in actions if a.gap_id)
        completed_gap_ids = set(
            a.gap_id for a in actions
            if a.status in _COMPLETED_STATUSES and a.gap_id
        )
        gaps_total = len(gap_ids)
        gaps_closed = len(completed_gap_ids)

        # Average effectiveness score
        effectiveness_scores = [
            a.effectiveness_score for a in actions
            if a.effectiveness_score is not None
        ]
        avg_effectiveness = Decimal("0")
        if effectiveness_scores:
            avg_effectiveness = (
                sum(effectiveness_scores) / Decimal(str(len(effectiveness_scores)))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Determine on-track status
        on_track = overdue == 0 and float(progress) >= self._expected_progress(plan)

        # Risk trend
        snapshots = self._snapshot_store.get(plan.plan_id, [])
        risk_trend = self._compute_risk_trend(snapshots, overdue)

        snapshot_id = f"SNAP-{uuid.uuid4().hex[:12]}"
        provenance_data = {
            "snapshot_id": snapshot_id,
            "plan_id": plan.plan_id,
            "progress": str(progress),
            "overdue": overdue,
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        snapshot = ProgressSnapshot(
            snapshot_id=snapshot_id,
            plan_id=plan.plan_id,
            overall_progress=progress,
            actions_total=total,
            actions_completed=completed,
            actions_in_progress=in_progress,
            actions_overdue=overdue,
            actions_on_hold=on_hold,
            gaps_closed=gaps_closed,
            gaps_total=gaps_total,
            avg_effectiveness_score=avg_effectiveness,
            on_track=on_track,
            risk_trend=risk_trend,
            provenance_hash=provenance_hash,
        )

        # Store
        self._snapshot_store.setdefault(plan.plan_id, []).append(snapshot)

        # Update gauges
        m.set_overall_progress(float(progress))
        m.set_overdue_actions(overdue)
        m.set_pending_actions(total - completed - on_hold)
        m.set_actions_on_hold(on_hold)
        m.set_avg_effectiveness(float(avg_effectiveness))
        m.record_progress_snapshot()

        self._provenance.record(
            "progress", "snapshot", snapshot_id, AGENT_ID,
            metadata={"plan_id": plan.plan_id, "progress": str(progress)},
        )

        elapsed = time.monotonic() - start
        m.observe_progress_tracking_duration(elapsed)

        logger.info(
            "Captured snapshot for plan %s: %.1f%% complete, %d overdue in %.1fms",
            plan.plan_id, float(progress), overdue, elapsed * 1000,
        )
        return snapshot

    def _count_overdue(self, actions: List[ImprovementAction]) -> int:
        """Count overdue actions based on deadline comparison.

        Args:
            actions: Actions to check.

        Returns:
            Number of overdue actions.
        """
        now = datetime.now(timezone.utc)
        overdue = 0
        for action in actions:
            if (
                action.time_bound_deadline
                and action.time_bound_deadline < now
                and action.status not in _COMPLETED_STATUSES
                and action.status != ActionStatus.CANCELLED
            ):
                overdue += 1
        return overdue

    def _expected_progress(self, plan: ImprovementPlan) -> float:
        """Calculate expected progress based on timeline.

        Args:
            plan: Improvement plan.

        Returns:
            Expected progress percentage.
        """
        if not plan.target_completion or not plan.created_at:
            return 0.0

        total_duration = (plan.target_completion - plan.created_at).total_seconds()
        if total_duration <= 0:
            return 100.0

        elapsed = (datetime.now(timezone.utc) - plan.created_at).total_seconds()
        return min((elapsed / total_duration) * 100.0, 100.0)

    def _compute_risk_trend(
        self,
        history: List[ProgressSnapshot],
        current_overdue: int,
    ) -> str:
        """Compute risk trend from snapshot history.

        Args:
            history: Previous snapshots.
            current_overdue: Current overdue count.

        Returns:
            Trend string: improving/stable/worsening.
        """
        if len(history) < 2:
            return "stable"

        prev = history[-1]
        if current_overdue > prev.actions_overdue:
            return "worsening"
        elif current_overdue < prev.actions_overdue:
            return "improving"
        return "stable"

    async def check_overdue(
        self, plan: ImprovementPlan
    ) -> List[ImprovementAction]:
        """Identify overdue actions in a plan.

        Args:
            plan: Improvement plan to check.

        Returns:
            List of overdue actions.
        """
        now = datetime.now(timezone.utc)
        threshold = timedelta(days=self.config.overdue_alert_threshold_days)
        overdue: List[ImprovementAction] = []

        for action in plan.actions:
            if (
                action.time_bound_deadline
                and action.time_bound_deadline < now
                and action.status not in _COMPLETED_STATUSES
                and action.status != ActionStatus.CANCELLED
            ):
                overdue.append(action)

        if self.config.auto_escalation_enabled:
            for action in overdue:
                days_overdue = (now - action.time_bound_deadline).days
                if days_overdue >= self.config.escalation_threshold_days:
                    m.record_escalation_triggered("auto")
                    logger.warning(
                        "Action %s is %d days overdue (escalation threshold: %d)",
                        action.action_id, days_overdue,
                        self.config.escalation_threshold_days,
                    )

        return overdue

    async def add_milestone(
        self,
        action_id: str,
        title: str,
        due_date: Optional[datetime] = None,
        description: str = "",
        weight: Decimal = Decimal("1.00"),
    ) -> ProgressMilestone:
        """Add a milestone to an action.

        Args:
            action_id: Parent action identifier.
            title: Milestone title.
            due_date: Target completion date.
            description: Milestone description.
            weight: Relative weight for progress calculation.

        Returns:
            Created ProgressMilestone.
        """
        milestone = ProgressMilestone(
            milestone_id=f"MS-{uuid.uuid4().hex[:12]}",
            action_id=action_id,
            title=title,
            description=description,
            due_date=due_date,
            weight=weight,
        )
        self._milestone_store.setdefault(action_id, []).append(milestone)

        self._provenance.record(
            "milestone", "create", milestone.milestone_id, AGENT_ID,
            metadata={"action_id": action_id, "title": title},
        )
        return milestone

    async def grant_extension(
        self,
        action: ImprovementAction,
        extension_days: int,
    ) -> bool:
        """Grant a deadline extension for an action.

        Args:
            action: Action to extend.
            extension_days: Number of days to extend.

        Returns:
            True if extension was granted, False if max extensions exceeded.
        """
        if action.extensions_used >= self.config.max_extensions_per_action:
            logger.warning(
                "Max extensions (%d) reached for action %s",
                self.config.max_extensions_per_action, action.action_id,
            )
            return False

        if action.time_bound_deadline:
            action.time_bound_deadline += timedelta(days=extension_days)
        action.extensions_used += 1

        self._provenance.record(
            "action", "extension", action.action_id, AGENT_ID,
            metadata={"extension_days": extension_days, "total_extensions": action.extensions_used},
        )
        return True

    async def get_snapshots(self, plan_id: str) -> List[ProgressSnapshot]:
        """Retrieve stored snapshots for a plan.

        Args:
            plan_id: Plan identifier.

        Returns:
            List of ProgressSnapshot in chronological order.
        """
        return self._snapshot_store.get(plan_id, [])

    async def get_milestones(self, action_id: str) -> List[ProgressMilestone]:
        """Retrieve milestones for an action.

        Args:
            action_id: Action identifier.

        Returns:
            List of ProgressMilestone.
        """
        return self._milestone_store.get(action_id, [])

    async def review_effectiveness(
        self, plan: ImprovementPlan
    ) -> Dict[str, Any]:
        """Review effectiveness of completed actions.

        Args:
            plan: Improvement plan to review.

        Returns:
            Effectiveness review summary.
        """
        start = time.monotonic()

        verified = [
            a for a in plan.actions
            if a.status in (ActionStatus.VERIFIED, ActionStatus.CLOSED)
            and a.effectiveness_score is not None
        ]

        if not verified:
            return {"plan_id": plan.plan_id, "verified_count": 0, "avg_score": 0}

        scores = [a.effectiveness_score for a in verified if a.effectiveness_score is not None]
        avg = (sum(scores) / Decimal(str(len(scores)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if scores else Decimal("0")

        elapsed = time.monotonic() - start
        m.observe_effectiveness_review_duration(elapsed)

        return {
            "plan_id": plan.plan_id,
            "verified_count": len(verified),
            "avg_score": str(avg),
            "high_performers": sum(1 for s in scores if s >= Decimal("80")),
            "low_performers": sum(1 for s in scores if s < Decimal("50")),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "ProgressTracker",
            "status": "healthy",
            "plans_tracked": len(self._snapshot_store),
            "milestones_stored": sum(len(v) for v in self._milestone_store.values()),
        }
