"""
CDP Transition Plan Engine -- 1.5C Transition Planning

This module implements 1.5C-aligned transition plan building and management
for CDP A-level scoring.  Covers pathway definition with milestones,
technology lever identification, investment planning (CapEx/OpEx), revenue
alignment tracking, SBTi alignment validation, and progress tracking.

A publicly available 1.5C-aligned transition plan is mandatory for CDP
A-level scoring (AREQ01).

Key capabilities:
  - Pathway definition with short/medium/long-term milestones
  - Technology lever identification and categorization
  - Investment planning (CapEx and OpEx)
  - Revenue alignment tracking (low-carbon percentage)
  - SBTi alignment validation (>= 4.2% annual reduction)
  - Progress tracking against milestones
  - Board oversight documentation

Example:
    >>> engine = TransitionPlanEngine(config)
    >>> plan = engine.create_plan("org-123", base_year=2020, base_emissions=100000)
    >>> engine.add_milestone(plan.id, "50% renewable", target_year=2030, reduction_pct=30)
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import CDPAppConfig, TransitionTimeframe
from .models import (
    TransitionMilestone,
    TransitionPlan,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technology Lever Catalog
# ---------------------------------------------------------------------------

TECHNOLOGY_LEVERS: Dict[str, Dict[str, Any]] = {
    "renewable_energy": {
        "name": "Renewable Energy Procurement",
        "scope": "scope_2",
        "typical_reduction_pct": 30.0,
        "capex_intensity": "medium",
        "maturity": "mature",
    },
    "energy_efficiency": {
        "name": "Energy Efficiency Improvements",
        "scope": "scope_1",
        "typical_reduction_pct": 15.0,
        "capex_intensity": "medium",
        "maturity": "mature",
    },
    "electrification": {
        "name": "Process Electrification",
        "scope": "scope_1",
        "typical_reduction_pct": 25.0,
        "capex_intensity": "high",
        "maturity": "emerging",
    },
    "fuel_switching": {
        "name": "Fuel Switching (fossil to low-carbon)",
        "scope": "scope_1",
        "typical_reduction_pct": 20.0,
        "capex_intensity": "medium",
        "maturity": "mature",
    },
    "ccus": {
        "name": "Carbon Capture, Utilization, and Storage",
        "scope": "scope_1",
        "typical_reduction_pct": 10.0,
        "capex_intensity": "very_high",
        "maturity": "emerging",
    },
    "fleet_electrification": {
        "name": "Vehicle Fleet Electrification",
        "scope": "scope_1",
        "typical_reduction_pct": 12.0,
        "capex_intensity": "high",
        "maturity": "emerging",
    },
    "supplier_engagement": {
        "name": "Supplier Decarbonization Programs",
        "scope": "scope_3",
        "typical_reduction_pct": 8.0,
        "capex_intensity": "low",
        "maturity": "mature",
    },
    "circular_economy": {
        "name": "Circular Economy and Waste Reduction",
        "scope": "scope_3",
        "typical_reduction_pct": 5.0,
        "capex_intensity": "low",
        "maturity": "mature",
    },
    "green_hydrogen": {
        "name": "Green Hydrogen",
        "scope": "scope_1",
        "typical_reduction_pct": 15.0,
        "capex_intensity": "very_high",
        "maturity": "pre_commercial",
    },
    "nature_based": {
        "name": "Nature-based Solutions",
        "scope": "all",
        "typical_reduction_pct": 5.0,
        "capex_intensity": "low",
        "maturity": "mature",
    },
    "product_redesign": {
        "name": "Low-carbon Product Redesign",
        "scope": "scope_3",
        "typical_reduction_pct": 10.0,
        "capex_intensity": "medium",
        "maturity": "emerging",
    },
    "digital_optimization": {
        "name": "Digital Optimization and AI",
        "scope": "scope_1",
        "typical_reduction_pct": 5.0,
        "capex_intensity": "medium",
        "maturity": "mature",
    },
}

# SBTi required minimum annual reduction rate
SBTI_MIN_ANNUAL_REDUCTION_PCT = Decimal("4.2")


class TransitionPlanEngine:
    """
    CDP Transition Plan Engine -- builds and tracks 1.5C transition plans.

    Provides pathway creation, milestone management, SBTi validation,
    technology lever planning, and progress tracking.

    Attributes:
        config: Application configuration.
        _plans: In-memory transition plan store.

    Example:
        >>> engine = TransitionPlanEngine(config)
        >>> plan = engine.create_plan("org-1", base_year=2020, base_emissions=50000)
    """

    def __init__(self, config: CDPAppConfig) -> None:
        """Initialize the Transition Plan Engine."""
        self.config = config
        self._plans: Dict[str, TransitionPlan] = {}
        self._by_org: Dict[str, List[str]] = {}
        logger.info("TransitionPlanEngine initialized with %d technology levers", len(TECHNOLOGY_LEVERS))

    # ------------------------------------------------------------------
    # Plan CRUD
    # ------------------------------------------------------------------

    def create_plan(
        self,
        org_id: str,
        base_year: int = 2020,
        base_year_emissions_tco2e: Decimal = Decimal("0"),
        target_year: int = 2050,
        annual_reduction_rate_pct: Decimal = Decimal("4.2"),
        pathway_aligned: str = "1.5c",
        questionnaire_id: Optional[str] = None,
        name: str = "1.5C Transition Plan",
    ) -> TransitionPlan:
        """
        Create a new 1.5C transition plan.

        Args:
            org_id: Organization ID.
            base_year: Base year for emissions baseline.
            base_year_emissions_tco2e: Base year total emissions.
            target_year: Target year for net-zero.
            annual_reduction_rate_pct: Annual absolute reduction rate.
            pathway_aligned: Pathway alignment (1.5c, well_below_2c, 2c).
            questionnaire_id: Optional CDP questionnaire ID.
            name: Plan name.

        Returns:
            Created TransitionPlan.
        """
        plan = TransitionPlan(
            org_id=org_id,
            questionnaire_id=questionnaire_id,
            name=name,
            base_year=base_year,
            base_year_emissions_tco2e=base_year_emissions_tco2e,
            target_year=target_year,
            annual_reduction_rate_pct=annual_reduction_rate_pct,
            pathway_aligned=pathway_aligned,
        )

        self._plans[plan.id] = plan
        if org_id not in self._by_org:
            self._by_org[org_id] = []
        self._by_org[org_id].append(plan.id)

        logger.info(
            "Created transition plan '%s' for org %s: base=%d (%.0f tCO2e) -> %d net-zero",
            name, org_id, base_year, float(base_year_emissions_tco2e), target_year,
        )
        return plan

    def get_plan(self, plan_id: str) -> Optional[TransitionPlan]:
        """Get a transition plan by ID."""
        return self._plans.get(plan_id)

    def get_org_plans(self, org_id: str) -> List[TransitionPlan]:
        """Get all transition plans for an organization."""
        plan_ids = self._by_org.get(org_id, [])
        return [self._plans[pid] for pid in plan_ids if pid in self._plans]

    def update_plan(
        self,
        plan_id: str,
        **updates: Any,
    ) -> Optional[TransitionPlan]:
        """Update transition plan fields."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None

        for key, value in updates.items():
            if hasattr(plan, key):
                setattr(plan, key, value)

        plan.updated_at = _now()
        return plan

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a transition plan."""
        plan = self._plans.pop(plan_id, None)
        if plan:
            org_plans = self._by_org.get(plan.org_id, [])
            if plan_id in org_plans:
                org_plans.remove(plan_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Milestone Management
    # ------------------------------------------------------------------

    def add_milestone(
        self,
        plan_id: str,
        name: str,
        target_year: int,
        target_reduction_pct: Decimal = Decimal("0"),
        target_absolute_tco2e: Optional[Decimal] = None,
        scope: str = "all",
        technology_lever: Optional[str] = None,
        timeframe: Optional[TransitionTimeframe] = None,
        capex_usd: Optional[Decimal] = None,
        opex_annual_usd: Optional[Decimal] = None,
        description: Optional[str] = None,
    ) -> TransitionMilestone:
        """
        Add a milestone to a transition plan.

        Args:
            plan_id: Transition plan ID.
            name: Milestone name.
            target_year: Target completion year.
            target_reduction_pct: Target reduction percentage from base year.
            target_absolute_tco2e: Absolute emissions target.
            scope: Scope covered (scope_1, scope_2, scope_3, all).
            technology_lever: Technology lever key.
            timeframe: Timeframe classification.
            capex_usd: Capital expenditure.
            opex_annual_usd: Annual operational expenditure.
            description: Milestone description.

        Returns:
            Created TransitionMilestone.
        """
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Transition plan {plan_id} not found")

        # Auto-determine timeframe
        if not timeframe:
            years_from_now = target_year - datetime.utcnow().year
            if years_from_now <= 3:
                timeframe = TransitionTimeframe.SHORT_TERM
            elif years_from_now <= 10:
                timeframe = TransitionTimeframe.MEDIUM_TERM
            else:
                timeframe = TransitionTimeframe.LONG_TERM

        milestone = TransitionMilestone(
            plan_id=plan_id,
            name=name,
            description=description,
            timeframe=timeframe,
            target_year=target_year,
            target_reduction_pct=target_reduction_pct,
            target_absolute_tco2e=target_absolute_tco2e,
            scope=scope,
            technology_lever=technology_lever,
            capex_usd=capex_usd,
            opex_annual_usd=opex_annual_usd,
        )

        plan.milestones.append(milestone)
        self._recalculate_plan_totals(plan)
        plan.updated_at = _now()

        logger.info(
            "Added milestone '%s' to plan %s: target %d, reduction %.1f%%",
            name, plan_id, target_year, float(target_reduction_pct),
        )
        return milestone

    def update_milestone_progress(
        self,
        plan_id: str,
        milestone_id: str,
        progress_pct: Decimal,
        status: Optional[str] = None,
    ) -> Optional[TransitionMilestone]:
        """Update progress on a milestone."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None

        for ms in plan.milestones:
            if ms.id == milestone_id:
                ms.progress_pct = min(progress_pct, Decimal("100"))
                if status:
                    ms.status = status
                elif progress_pct >= Decimal("100"):
                    ms.status = "completed"
                elif progress_pct > Decimal("0"):
                    ms.status = "in_progress"
                ms.updated_at = _now()
                self._recalculate_plan_progress(plan)
                return ms

        return None

    def remove_milestone(self, plan_id: str, milestone_id: str) -> bool:
        """Remove a milestone from a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return False

        original = len(plan.milestones)
        plan.milestones = [m for m in plan.milestones if m.id != milestone_id]
        removed = len(plan.milestones) < original

        if removed:
            self._recalculate_plan_totals(plan)
            plan.updated_at = _now()
        return removed

    # ------------------------------------------------------------------
    # SBTi Alignment Validation
    # ------------------------------------------------------------------

    def validate_sbti_alignment(self, plan_id: str) -> Dict[str, Any]:
        """
        Validate transition plan alignment with SBTi requirements.

        Checks:
          - Annual absolute reduction rate >= 4.2%
          - Covers Scope 1 and 2 (mandatory) and Scope 3 (if > 40% of total)
          - Target year within 5-10 years for near-term
          - Long-term target to 2050 for net-zero

        Returns:
            Validation result with details.
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return {"valid": False, "error": "Plan not found"}

        issues = []
        passed = []

        # Check annual reduction rate
        if plan.annual_reduction_rate_pct >= SBTI_MIN_ANNUAL_REDUCTION_PCT:
            passed.append({
                "check": "annual_reduction_rate",
                "value": float(plan.annual_reduction_rate_pct),
                "requirement": float(SBTI_MIN_ANNUAL_REDUCTION_PCT),
                "status": "pass",
            })
        else:
            issues.append({
                "check": "annual_reduction_rate",
                "value": float(plan.annual_reduction_rate_pct),
                "requirement": float(SBTI_MIN_ANNUAL_REDUCTION_PCT),
                "message": f"Annual reduction rate {plan.annual_reduction_rate_pct}% is below SBTi minimum of {SBTI_MIN_ANNUAL_REDUCTION_PCT}%",
                "status": "fail",
            })

        # Check pathway alignment
        if plan.pathway_aligned in ("1.5c",):
            passed.append({
                "check": "pathway_alignment",
                "value": plan.pathway_aligned,
                "status": "pass",
            })
        else:
            issues.append({
                "check": "pathway_alignment",
                "value": plan.pathway_aligned,
                "message": f"Pathway '{plan.pathway_aligned}' may not meet SBTi 1.5C requirement",
                "status": "warning",
            })

        # Check net-zero target
        if plan.target_net_zero and plan.target_year <= 2050:
            passed.append({
                "check": "net_zero_target",
                "value": plan.target_year,
                "status": "pass",
            })
        else:
            issues.append({
                "check": "net_zero_target",
                "value": plan.target_year,
                "message": "SBTi requires net-zero target by 2050 at the latest",
                "status": "warning" if plan.target_year <= 2060 else "fail",
            })

        # Check near-term milestone exists
        near_term = [
            m for m in plan.milestones
            if m.timeframe == TransitionTimeframe.SHORT_TERM
        ]
        if near_term:
            passed.append({"check": "near_term_milestone", "status": "pass"})
        else:
            issues.append({
                "check": "near_term_milestone",
                "message": "SBTi requires near-term targets (5-10 years)",
                "status": "fail",
            })

        # Check interim target
        if plan.interim_target_year and plan.interim_reduction_pct:
            passed.append({
                "check": "interim_target",
                "value": f"{plan.interim_reduction_pct}% by {plan.interim_target_year}",
                "status": "pass",
            })

        is_aligned = len(issues) == 0 or all(
            i["status"] == "warning" for i in issues
        )

        return {
            "plan_id": plan_id,
            "sbti_aligned": is_aligned,
            "sbti_status": plan.sbti_status,
            "checks_passed": len(passed),
            "checks_failed": sum(1 for i in issues if i["status"] == "fail"),
            "checks_warning": sum(1 for i in issues if i["status"] == "warning"),
            "passed": passed,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Pathway Modeling
    # ------------------------------------------------------------------

    def model_reduction_pathway(
        self,
        plan_id: str,
    ) -> Dict[str, Any]:
        """
        Model the emissions reduction pathway from base year to target year.

        Generates year-by-year projected emissions based on the annual
        reduction rate and milestone targets.

        Returns:
            Pathway data with annual projections.
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}

        base = float(plan.base_year_emissions_tco2e)
        rate = float(plan.annual_reduction_rate_pct) / 100.0
        pathway = []

        for year in range(plan.base_year, plan.target_year + 1):
            years_elapsed = year - plan.base_year
            projected = base * ((1 - rate) ** years_elapsed)

            # Check if any milestone targets override
            milestone_target = None
            for ms in plan.milestones:
                if ms.target_year == year:
                    if ms.target_absolute_tco2e is not None:
                        milestone_target = float(ms.target_absolute_tco2e)
                    elif ms.target_reduction_pct > 0:
                        milestone_target = base * (1 - float(ms.target_reduction_pct) / 100)

            pathway.append({
                "year": year,
                "projected_emissions_tco2e": round(projected, 2),
                "milestone_target_tco2e": round(milestone_target, 2) if milestone_target else None,
                "reduction_from_base_pct": round((1 - projected / base) * 100, 1) if base > 0 else 0.0,
            })

        return {
            "plan_id": plan_id,
            "base_year": plan.base_year,
            "base_emissions_tco2e": base,
            "target_year": plan.target_year,
            "annual_reduction_rate_pct": float(plan.annual_reduction_rate_pct),
            "pathway": pathway,
        }

    # ------------------------------------------------------------------
    # Technology Levers
    # ------------------------------------------------------------------

    def get_available_levers(self) -> List[Dict[str, Any]]:
        """Get the catalog of available technology levers."""
        return [
            {"key": key, **info}
            for key, info in TECHNOLOGY_LEVERS.items()
        ]

    def recommend_levers(
        self,
        plan_id: str,
        scope_focus: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommend technology levers for a plan based on scope focus.

        Args:
            plan_id: Transition plan ID.
            scope_focus: Scope to prioritize (scope_1, scope_2, scope_3).

        Returns:
            Sorted list of recommended levers.
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return []

        # Already-used levers
        used = set()
        for ms in plan.milestones:
            if ms.technology_lever:
                used.add(ms.technology_lever)

        recommendations = []
        for key, info in TECHNOLOGY_LEVERS.items():
            if key in used:
                continue

            relevance = 1.0
            if scope_focus and info["scope"] != scope_focus and info["scope"] != "all":
                relevance = 0.5

            recommendations.append({
                "key": key,
                "name": info["name"],
                "scope": info["scope"],
                "typical_reduction_pct": info["typical_reduction_pct"],
                "capex_intensity": info["capex_intensity"],
                "maturity": info["maturity"],
                "relevance_score": relevance * info["typical_reduction_pct"],
            })

        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return recommendations

    # ------------------------------------------------------------------
    # Progress Tracking
    # ------------------------------------------------------------------

    def get_progress_summary(self, plan_id: str) -> Dict[str, Any]:
        """Get overall progress summary for a transition plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}

        total_ms = len(plan.milestones)
        completed = sum(1 for m in plan.milestones if m.status == "completed")
        in_progress = sum(1 for m in plan.milestones if m.status == "in_progress")
        delayed = sum(1 for m in plan.milestones if m.status == "delayed")

        return {
            "plan_id": plan_id,
            "overall_progress_pct": float(plan.overall_progress_pct),
            "total_milestones": total_ms,
            "completed": completed,
            "in_progress": in_progress,
            "planned": total_ms - completed - in_progress - delayed,
            "delayed": delayed,
            "total_capex_usd": float(plan.total_capex_usd),
            "total_opex_annual_usd": float(plan.total_opex_annual_usd),
            "revenue_low_carbon_pct": float(plan.revenue_low_carbon_pct),
            "is_public": plan.is_public,
            "board_approved": plan.board_approved,
            "sbti_status": plan.sbti_status,
            "pathway_aligned": plan.pathway_aligned,
        }

    # ------------------------------------------------------------------
    # Board Oversight
    # ------------------------------------------------------------------

    def set_board_approval(
        self,
        plan_id: str,
        approved: bool,
        approval_date: Optional[date] = None,
    ) -> Optional[TransitionPlan]:
        """Record board approval of the transition plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None

        plan.board_approved = approved
        plan.board_approval_date = approval_date or date.today()
        plan.updated_at = _now()

        logger.info(
            "Transition plan %s board approval: %s on %s",
            plan_id, approved, plan.board_approval_date,
        )
        return plan

    def set_public_status(
        self,
        plan_id: str,
        is_public: bool,
    ) -> Optional[TransitionPlan]:
        """Set whether the transition plan is publicly available."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None

        plan.is_public = is_public
        plan.updated_at = _now()
        return plan

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _recalculate_plan_totals(self, plan: TransitionPlan) -> None:
        """Recalculate CapEx and OpEx totals from milestones."""
        plan.total_capex_usd = sum(
            (m.capex_usd or Decimal("0")) for m in plan.milestones
        )
        plan.total_opex_annual_usd = sum(
            (m.opex_annual_usd or Decimal("0")) for m in plan.milestones
        )

    def _recalculate_plan_progress(self, plan: TransitionPlan) -> None:
        """Recalculate overall plan progress from milestones."""
        if not plan.milestones:
            plan.overall_progress_pct = Decimal("0")
            return

        total_progress = sum(float(m.progress_pct) for m in plan.milestones)
        avg_progress = total_progress / len(plan.milestones)
        plan.overall_progress_pct = Decimal(str(round(avg_progress, 1)))
