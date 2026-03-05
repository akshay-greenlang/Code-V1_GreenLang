"""
Management Plan Engine -- ISO 14064-1:2018 Clause 9 Implementation

Implements management plan CRUD, improvement action tracking, target setting,
cost-benefit analysis, milestone tracking, and annual review cycles for GHG
emission reduction and data quality improvement.

Action categories (from config.ActionCategory):
  - emission_reduction: Direct emission reduction measures
  - removal_enhancement: Enhance GHG removal activities
  - data_improvement: Improve data quality and coverage
  - process_improvement: Improve internal processes and controls

Reference: ISO 14064-1:2018 Clause 9.

Example:
    >>> engine = ManagementPlanEngine(config)
    >>> plan = engine.create_plan("org-1", 2025)
    >>> action = engine.add_action(plan.id, "Install solar panels", ...)
    >>> engine.update_progress(plan.id, action.id, Decimal("75"))
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ActionCategory,
    ActionStatus,
    ISO14064AppConfig,
    ISOCategory,
)
from .models import (
    ImprovementAction,
    ManagementPlan,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class ManagementPlanEngine:
    """
    CRUD and analytics for ISO 14064-1 management plans.

    Provides lifecycle management for emission reduction plans and
    improvement actions with cost-benefit analysis, progress tracking,
    and annual review cycles.

    Attributes:
        config: Application configuration.
        _plans: In-memory store of management plans keyed by plan ID.
    """

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
    ) -> None:
        """
        Initialize ManagementPlanEngine.

        Args:
            config: Application configuration.
        """
        self.config = config or ISO14064AppConfig()
        self._plans: Dict[str, ManagementPlan] = {}
        logger.info("ManagementPlanEngine initialized")

    # ------------------------------------------------------------------
    # Plan CRUD
    # ------------------------------------------------------------------

    def create_plan(
        self,
        org_id: str,
        reporting_year: int,
    ) -> ManagementPlan:
        """
        Create a new management plan.

        Args:
            org_id: Organization ID.
            reporting_year: Target year.

        Returns:
            Created ManagementPlan.
        """
        start = datetime.utcnow()

        plan = ManagementPlan(
            org_id=org_id,
            reporting_year=reporting_year,
        )
        self._plans[plan.id] = plan

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Created management plan for org '%s' year %d (id=%s) in %.1f ms",
            org_id,
            reporting_year,
            plan.id,
            elapsed_ms,
        )
        return plan

    def get_plan(self, plan_id: str) -> Optional[ManagementPlan]:
        """Retrieve a management plan by ID."""
        return self._plans.get(plan_id)

    def get_plans_for_org(self, org_id: str) -> List[ManagementPlan]:
        """Get all plans for an organization."""
        return [p for p in self._plans.values() if p.org_id == org_id]

    def get_plan_for_year(
        self,
        org_id: str,
        reporting_year: int,
    ) -> Optional[ManagementPlan]:
        """Get a plan for a specific organization and year."""
        for p in self._plans.values():
            if p.org_id == org_id and p.reporting_year == reporting_year:
                return p
        return None

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a management plan."""
        if plan_id in self._plans:
            del self._plans[plan_id]
            logger.info("Deleted management plan %s", plan_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Action CRUD
    # ------------------------------------------------------------------

    def add_action(
        self,
        plan_id: str,
        name: str,
        category: ActionCategory,
        description: str = "",
        iso_category: Optional[ISOCategory] = None,
        target_reduction_tco2e: Optional[Decimal] = None,
        target_year: Optional[int] = None,
        timeline_start: Optional[date] = None,
        timeline_end: Optional[date] = None,
        estimated_cost_usd: Optional[Decimal] = None,
        assigned_to: Optional[str] = None,
    ) -> ImprovementAction:
        """
        Add an improvement action to a plan.

        Args:
            plan_id: Parent plan ID.
            name: Action name.
            category: Category of action.
            description: Detailed description.
            iso_category: ISO category targeted.
            target_reduction_tco2e: Expected reduction (tCO2e).
            target_year: Target completion year.
            timeline_start: Planned start date.
            timeline_end: Planned end date.
            estimated_cost_usd: Estimated cost (USD).
            assigned_to: Person responsible.

        Returns:
            Created ImprovementAction.
        """
        plan = self._get_plan_or_raise(plan_id)

        action = ImprovementAction(
            name=name,
            category=category,
            iso_category=iso_category,
            description=description,
            target_reduction_tco2e=target_reduction_tco2e,
            target_year=target_year,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
            estimated_cost_usd=estimated_cost_usd,
            status=ActionStatus.PLANNED,
            assigned_to=assigned_to,
        )

        plan.actions.append(action)
        plan.recalculate_totals()
        plan.updated_at = _now()

        logger.info(
            "Added action '%s' to plan %s (category=%s, target=%.2f tCO2e)",
            name,
            plan_id,
            category.value,
            target_reduction_tco2e or Decimal("0"),
        )
        return action

    def update_action_status(
        self,
        plan_id: str,
        action_id: str,
        status: ActionStatus,
    ) -> ImprovementAction:
        """
        Update the status of an improvement action.

        Args:
            plan_id: Plan ID.
            action_id: Action ID.
            status: New status.

        Returns:
            Updated ImprovementAction.
        """
        plan = self._get_plan_or_raise(plan_id)
        action = self._find_action(plan, action_id)

        action.status = status
        action.updated_at = _now()
        plan.updated_at = _now()

        logger.info(
            "Updated action %s status to %s in plan %s",
            action_id,
            status.value,
            plan_id,
        )
        return action

    def update_progress(
        self,
        plan_id: str,
        action_id: str,
        progress_pct: Decimal,
        notes: Optional[str] = None,
    ) -> ImprovementAction:
        """
        Update progress percentage for an action.

        Args:
            plan_id: Plan ID.
            action_id: Action ID.
            progress_pct: Progress percentage (0-100).
            notes: Optional progress notes.

        Returns:
            Updated ImprovementAction.
        """
        plan = self._get_plan_or_raise(plan_id)
        action = self._find_action(plan, action_id)

        action.progress_pct = progress_pct
        if notes:
            action.notes = notes
        action.updated_at = _now()
        plan.updated_at = _now()

        # Auto-update status based on progress
        if progress_pct >= Decimal("100"):
            action.status = ActionStatus.COMPLETED
        elif progress_pct > Decimal("0") and action.status == ActionStatus.PLANNED:
            action.status = ActionStatus.IN_PROGRESS

        logger.info(
            "Updated action %s progress to %.1f%% in plan %s",
            action_id,
            progress_pct,
            plan_id,
        )
        return action

    def remove_action(
        self,
        plan_id: str,
        action_id: str,
    ) -> bool:
        """
        Remove an action from a plan.

        Args:
            plan_id: Plan ID.
            action_id: Action ID.

        Returns:
            True if removed, False if not found.
        """
        plan = self._get_plan_or_raise(plan_id)
        original_count = len(plan.actions)
        plan.actions = [a for a in plan.actions if a.id != action_id]

        if len(plan.actions) < original_count:
            plan.recalculate_totals()
            plan.updated_at = _now()
            logger.info("Removed action %s from plan %s", action_id, plan_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Cost-Benefit Analysis
    # ------------------------------------------------------------------

    def get_cost_benefit_analysis(
        self,
        plan_id: str,
    ) -> Dict[str, Any]:
        """
        Generate cost-benefit analysis for all actions in a plan.

        Args:
            plan_id: Plan ID.

        Returns:
            Dict with total costs, savings, and MAC curve data.
        """
        plan = self._get_plan_or_raise(plan_id)

        total_cost = Decimal("0")
        total_target_reduction = Decimal("0")
        mac_curve: List[Dict[str, Any]] = []

        for action in plan.actions:
            cost = action.estimated_cost_usd or Decimal("0")
            reduction = action.target_reduction_tco2e or Decimal("0")
            total_cost += cost
            total_target_reduction += reduction

            # Compute marginal abatement cost (MAC)
            if reduction > 0 and cost > 0:
                mac = (cost / reduction).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
                mac_curve.append({
                    "action": action.name,
                    "category": action.category.value,
                    "reduction_tco2e": str(reduction),
                    "cost_usd": str(cost),
                    "mac_usd_per_tco2e": str(mac),
                    "status": action.status.value,
                })

        # Sort MAC curve by abatement cost (cheapest first)
        mac_curve.sort(key=lambda x: float(x["mac_usd_per_tco2e"]))

        return {
            "plan_id": plan_id,
            "total_estimated_cost_usd": str(total_cost),
            "total_target_reduction_tco2e": str(total_target_reduction),
            "action_count": len(plan.actions),
            "mac_curve": mac_curve,
        }

    # ------------------------------------------------------------------
    # Progress Tracking
    # ------------------------------------------------------------------

    def get_progress_summary(
        self,
        plan_id: str,
    ) -> Dict[str, Any]:
        """
        Get progress summary for a management plan.

        Args:
            plan_id: Plan ID.

        Returns:
            Dict with progress metrics.
        """
        plan = self._get_plan_or_raise(plan_id)

        by_status: Dict[str, int] = {}
        for action in plan.actions:
            status_key = action.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

        by_category: Dict[str, Dict[str, Any]] = {}
        for action in plan.actions:
            cat_key = action.category.value
            if cat_key not in by_category:
                by_category[cat_key] = {
                    "count": 0,
                    "target_tco2e": Decimal("0"),
                }
            by_category[cat_key]["count"] += 1
            by_category[cat_key]["target_tco2e"] += (
                action.target_reduction_tco2e or Decimal("0")
            )

        total_target = plan.total_planned_reduction_tco2e
        completed_reduction = Decimal("0")
        for action in plan.actions:
            if action.status == ActionStatus.COMPLETED:
                completed_reduction += action.target_reduction_tco2e or Decimal("0")

        progress_pct = Decimal("0")
        if total_target > 0:
            progress_pct = (completed_reduction / total_target * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP,
            )

        # Average progress across all actions
        avg_progress = Decimal("0")
        if plan.actions:
            total_progress = sum(a.progress_pct for a in plan.actions)
            avg_progress = (total_progress / len(plan.actions)).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP,
            )

        return {
            "plan_id": plan_id,
            "org_id": plan.org_id,
            "reporting_year": plan.reporting_year,
            "action_count": len(plan.actions),
            "by_status": by_status,
            "by_category": {
                k: {
                    "count": v["count"],
                    "target_tco2e": str(v["target_tco2e"]),
                }
                for k, v in by_category.items()
            },
            "total_planned_reduction_tco2e": str(total_target),
            "total_estimated_cost_usd": str(plan.total_estimated_cost_usd),
            "completed_reduction_tco2e": str(completed_reduction),
            "reduction_progress_pct": str(progress_pct),
            "average_action_progress_pct": str(avg_progress),
        }

    # ------------------------------------------------------------------
    # Action Prioritization
    # ------------------------------------------------------------------

    def get_prioritized_actions(
        self,
        plan_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get actions prioritized by cost-effectiveness (lowest MAC first).

        Args:
            plan_id: Plan ID.

        Returns:
            Sorted list of actions with cost-effectiveness metrics.
        """
        plan = self._get_plan_or_raise(plan_id)
        prioritized: List[Dict[str, Any]] = []

        for action in plan.actions:
            cost = action.estimated_cost_usd or Decimal("0")
            reduction = action.target_reduction_tco2e or Decimal("0")

            mac = Decimal("999999")
            if reduction > 0 and cost > 0:
                mac = (cost / reduction).quantize(Decimal("0.01"))

            prioritized.append({
                "action_id": action.id,
                "name": action.name,
                "category": action.category.value,
                "target_category": action.iso_category.value if action.iso_category else None,
                "reduction_tco2e": str(reduction),
                "cost_usd": str(cost),
                "mac_usd_per_tco2e": str(mac) if mac < Decimal("999999") else None,
                "status": action.status.value,
                "progress_pct": str(action.progress_pct),
            })

        # Sort: completed last, then by MAC ascending
        prioritized.sort(
            key=lambda x: (
                1 if x["status"] in ("completed", "cancelled") else 0,
                float(x["mac_usd_per_tco2e"]) if x["mac_usd_per_tco2e"] else float("inf"),
            )
        )

        return prioritized

    # ------------------------------------------------------------------
    # Annual Review
    # ------------------------------------------------------------------

    def conduct_annual_review(
        self,
        plan_id: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Conduct an annual review of the management plan.

        Generates a review summary with progress and recommendations.

        Args:
            plan_id: Plan ID.
            notes: Review notes.

        Returns:
            Dict with review summary.
        """
        plan = self._get_plan_or_raise(plan_id)

        progress = self.get_progress_summary(plan_id)
        cost_benefit = self.get_cost_benefit_analysis(plan_id)

        plan.updated_at = _now()

        review_result = {
            "plan_id": plan_id,
            "review_date": _now().isoformat(),
            "notes": notes,
            "progress_summary": progress,
            "cost_benefit_summary": cost_benefit,
            "recommendations": self._generate_review_recommendations(plan),
        }

        logger.info("Conducted annual review for plan %s", plan_id)
        return review_result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_plan_summary(
        self,
        plan_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive plan summary.

        Args:
            plan_id: Plan ID.

        Returns:
            Dict with plan overview and all actions.
        """
        plan = self._get_plan_or_raise(plan_id)

        actions_summary: List[Dict[str, Any]] = []
        for action in plan.actions:
            actions_summary.append({
                "id": action.id,
                "name": action.name,
                "category": action.category.value,
                "iso_category": action.iso_category.value if action.iso_category else None,
                "target_reduction_tco2e": str(action.target_reduction_tco2e or Decimal("0")),
                "estimated_cost_usd": str(action.estimated_cost_usd or Decimal("0")),
                "status": action.status.value,
                "progress_pct": str(action.progress_pct),
                "assigned_to": action.assigned_to,
                "target_year": action.target_year,
            })

        return {
            "plan_id": plan.id,
            "org_id": plan.org_id,
            "reporting_year": plan.reporting_year,
            "total_actions": len(plan.actions),
            "total_planned_reduction_tco2e": str(plan.total_planned_reduction_tco2e),
            "total_estimated_cost_usd": str(plan.total_estimated_cost_usd),
            "actions": actions_summary,
            "created_at": plan.created_at.isoformat(),
            "updated_at": plan.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_plan_or_raise(self, plan_id: str) -> ManagementPlan:
        """Retrieve plan or raise ValueError."""
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Management plan not found: {plan_id}")
        return plan

    @staticmethod
    def _find_action(
        plan: ManagementPlan,
        action_id: str,
    ) -> ImprovementAction:
        """Find an action by ID within a plan."""
        for action in plan.actions:
            if action.id == action_id:
                return action
        raise ValueError(f"Action not found: {action_id}")

    @staticmethod
    def _generate_review_recommendations(
        plan: ManagementPlan,
    ) -> List[str]:
        """Generate recommendations based on current plan state."""
        recommendations: List[str] = []

        # Check for overdue actions
        today = date.today()
        overdue = [
            a for a in plan.actions
            if a.timeline_end and a.timeline_end < today
            and a.status not in (ActionStatus.COMPLETED, ActionStatus.CANCELLED)
        ]
        if overdue:
            recommendations.append(
                f"{len(overdue)} action(s) are overdue. Review and update timelines."
            )

        # Check progress
        total_target = plan.total_planned_reduction_tco2e
        completed_reduction = sum(
            (a.target_reduction_tco2e or Decimal("0"))
            for a in plan.actions
            if a.status == ActionStatus.COMPLETED
        )

        if total_target > 0:
            progress_pct = completed_reduction / total_target * 100
            if progress_pct < Decimal("25"):
                recommendations.append(
                    "Progress is below 25% of planned reductions. "
                    "Consider accelerating high-impact actions."
                )
            elif progress_pct < Decimal("50"):
                recommendations.append(
                    "Progress is at moderate levels. Review barriers to completion."
                )

        # Check for actions with no assigned responsible
        unassigned = [a for a in plan.actions if not a.assigned_to]
        if unassigned:
            recommendations.append(
                f"{len(unassigned)} action(s) have no assigned responsible person."
            )

        # Check for cancelled actions
        cancelled = [
            a for a in plan.actions if a.status == ActionStatus.CANCELLED
        ]
        if cancelled:
            recommendations.append(
                f"{len(cancelled)} action(s) have been cancelled. "
                "Consider replacement actions to meet targets."
            )

        # Check for stalled actions (in_progress but no progress)
        stalled = [
            a for a in plan.actions
            if a.status == ActionStatus.IN_PROGRESS
            and a.progress_pct == Decimal("0")
        ]
        if stalled:
            recommendations.append(
                f"{len(stalled)} action(s) are marked as in-progress but show "
                "no progress. Investigate and provide support."
            )

        if not recommendations:
            recommendations.append("Plan is on track. Continue monitoring progress.")

        return recommendations
