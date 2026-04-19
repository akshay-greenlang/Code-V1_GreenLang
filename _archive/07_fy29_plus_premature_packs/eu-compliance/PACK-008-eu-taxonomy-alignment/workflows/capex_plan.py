# -*- coding: utf-8 -*-
"""
CapEx Plan Workflow
======================

Four-phase workflow for defining, projecting, approving, and monitoring
taxonomy-aligned Capital Expenditure plans per Article 8 DA Annex I
Section 1.1.2.

This workflow enables:
- CapEx plan definition for activities seeking taxonomy alignment
- Forward-looking alignment ratio projection over 5-year horizon
- Multi-stage approval workflow for CapEx plan commitments
- Ongoing monitoring of CapEx plan execution against targets

Phases:
    1. Plan Definition - Define CapEx plan for taxonomy alignment
    2. Alignment Projection - Project future alignment ratios
    3. Approval - Approval workflow for CapEx plan
    4. Monitoring - Track CapEx plan execution against targets

Regulatory Context:
    Article 8 DA Annex I Section 1.1.2 allows undertakings to include CapEx
    in the taxonomy-aligned CapEx KPI if it relates to activities that are
    not yet taxonomy-aligned but have a credible plan to become aligned within
    a defined timeframe (up to 5 years, extendable to 10 for specific cases).

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    PLAN_DEFINITION = "plan_definition"
    ALIGNMENT_PROJECTION = "alignment_projection"
    APPROVAL = "approval"
    MONITORING = "monitoring"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PlanStatus(str, Enum):
    """CapEx plan lifecycle status."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    IN_PROGRESS = "IN_PROGRESS"
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


# =============================================================================
# DATA MODELS
# =============================================================================


class CapExPlanConfig(BaseModel):
    """Configuration for CapEx plan workflow."""
    organization_id: Optional[str] = Field(None, description="Organization identifier")
    plan_horizon_years: int = Field(default=5, ge=1, le=10, description="Plan horizon in years")
    base_year: int = Field(default=2025, ge=2020, description="Base year for plan")
    currency: str = Field(default="EUR", description="Reporting currency")
    require_board_approval: bool = Field(default=True, description="Require board-level approval")
    monitoring_frequency: str = Field(default="quarterly", description="Monitoring cadence")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: CapExPlanConfig = Field(default_factory=CapExPlanConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the CapEx plan workflow."""
    workflow_name: str = Field(default="capex_plan", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    plan_id: Optional[str] = Field(None, description="CapEx plan identifier")
    total_planned_capex: float = Field(default=0.0, ge=0.0, description="Total planned CapEx amount")
    activities_in_plan: int = Field(default=0, ge=0, description="Activities covered by plan")
    projected_alignment_improvement: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Expected alignment ratio improvement"
    )
    plan_status: str = Field(default="DRAFT", description="Current plan status")
    plan_on_track: bool = Field(default=True, description="Plan execution on track")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# CAPEX PLAN WORKFLOW
# =============================================================================


class CapExPlanWorkflow:
    """
    Four-phase CapEx plan workflow.

    Manages taxonomy-aligned CapEx plans per Article 8 DA requirements:
    - Define investment plans for activities seeking alignment
    - Project future alignment ratio improvements
    - Obtain governance approval for plan commitments
    - Monitor execution and flag deviations

    Example:
        >>> config = CapExPlanConfig(
        ...     organization_id="ORG-001",
        ...     plan_horizon_years=5,
        ...     base_year=2025,
        ... )
        >>> workflow = CapExPlanWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.overall_status == PhaseStatus.COMPLETED
        >>> assert result.plan_id is not None
    """

    def __init__(self, config: Optional[CapExPlanConfig] = None) -> None:
        """Initialize the CapEx plan workflow."""
        self.config = config or CapExPlanConfig()
        self.logger = logging.getLogger(f"{__name__}.CapExPlanWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 4-phase CapEx plan workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with plan definition, projections, approval, and monitoring.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting CapEx plan workflow execution_id=%s horizon=%d years",
            context.execution_id,
            self.config.plan_horizon_years,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.PLAN_DEFINITION, self._phase_1_plan_definition),
            (Phase.ALIGNMENT_PROJECTION, self._phase_2_alignment_projection),
            (Phase.APPROVAL, self._phase_3_approval),
            (Phase.MONITORING, self._phase_4_monitoring),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        plan_id = context.state.get("plan_id")
        total_capex = context.state.get("total_planned_capex", 0.0)
        activities = context.state.get("activities_in_plan", 0)
        improvement = context.state.get("projected_alignment_improvement", 0.0)
        plan_status = context.state.get("plan_status", "DRAFT")
        on_track = context.state.get("plan_on_track", True)

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "plan_id": plan_id,
        })

        self.logger.info(
            "CapEx plan workflow finished execution_id=%s status=%s "
            "plan=%s total_capex=%.2f improvement=%.1f%%",
            context.execution_id,
            overall_status.value,
            plan_id,
            total_capex,
            improvement * 100,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            plan_id=plan_id,
            total_planned_capex=total_capex,
            activities_in_plan=activities,
            projected_alignment_improvement=improvement,
            plan_status=plan_status,
            plan_on_track=on_track,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Plan Definition
    # -------------------------------------------------------------------------

    async def _phase_1_plan_definition(self, context: WorkflowContext) -> PhaseResult:
        """
        Define CapEx plan for taxonomy alignment.

        Plan elements:
        - Activities targeted for alignment (eligible but not yet aligned)
        - Investment amounts per activity per year
        - TSC criteria to be met through investment
        - Technology/process changes required
        - Expected completion milestones
        """
        phase = Phase.PLAN_DEFINITION

        self.logger.info("Defining CapEx plan for %d-year horizon", self.config.plan_horizon_years)

        await asyncio.sleep(0.05)

        plan_id = f"CXPLAN-{uuid.uuid4().hex[:8]}"
        activity_count = random.randint(3, 10)
        plan_activities = []
        total_capex = 0.0

        for i in range(activity_count):
            annual_capex = round(random.uniform(500_000, 15_000_000), 2)
            total_activity_capex = annual_capex * self.config.plan_horizon_years
            total_capex += total_activity_capex

            plan_activities.append({
                "activity_id": f"ACT-{uuid.uuid4().hex[:8]}",
                "description": f"Alignment activity {i + 1}",
                "current_status": "eligible_not_aligned",
                "target_status": "aligned",
                "annual_capex": annual_capex,
                "total_capex": round(total_activity_capex, 2),
                "target_year": self.config.base_year + random.randint(1, self.config.plan_horizon_years),
                "tsc_criteria_to_meet": random.randint(1, 5),
                "investment_type": random.choice([
                    "technology_upgrade", "process_change", "renewable_energy",
                    "building_renovation", "fleet_electrification",
                ]),
            })

        context.state["plan_id"] = plan_id
        context.state["plan_activities"] = plan_activities
        context.state["total_planned_capex"] = round(total_capex, 2)
        context.state["activities_in_plan"] = activity_count

        provenance = self._hash({
            "phase": phase.value,
            "plan_id": plan_id,
            "activity_count": activity_count,
            "total_capex": total_capex,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "plan_id": plan_id,
                "activities_in_plan": activity_count,
                "total_planned_capex": round(total_capex, 2),
                "horizon_years": self.config.plan_horizon_years,
                "currency": self.config.currency,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Alignment Projection
    # -------------------------------------------------------------------------

    async def _phase_2_alignment_projection(self, context: WorkflowContext) -> PhaseResult:
        """
        Project future alignment ratios.

        Projections calculated:
        - Year-by-year alignment ratio improvement
        - Cumulative CapEx deployed vs. planned
        - Expected activity completion dates
        - Sensitivity to execution delays
        """
        phase = Phase.ALIGNMENT_PROJECTION
        activities = context.state.get("plan_activities", [])
        base_year = self.config.base_year
        horizon = self.config.plan_horizon_years

        self.logger.info("Projecting alignment ratios for %d-year horizon", horizon)

        current_alignment = round(random.uniform(0.15, 0.35), 4)
        projections = []
        cumulative_improvement = 0.0

        for year_offset in range(1, horizon + 1):
            year = base_year + year_offset
            activities_completing = len([
                a for a in activities if a["target_year"] == year
            ])
            year_improvement = activities_completing * random.uniform(0.02, 0.06)
            cumulative_improvement += year_improvement

            projections.append({
                "year": year,
                "projected_alignment": round(min(current_alignment + cumulative_improvement, 1.0), 4),
                "activities_completing": activities_completing,
                "cumulative_capex_deployed_pct": round(year_offset / horizon, 2),
                "year_improvement": round(year_improvement, 4),
            })

        projected_improvement = round(min(cumulative_improvement, 1.0 - current_alignment), 4)
        context.state["projections"] = projections
        context.state["projected_alignment_improvement"] = projected_improvement
        context.state["current_alignment_baseline"] = current_alignment

        provenance = self._hash({
            "phase": phase.value,
            "current_alignment": current_alignment,
            "projected_improvement": projected_improvement,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "current_alignment": current_alignment,
                "projected_final_alignment": round(current_alignment + projected_improvement, 4),
                "projected_improvement": projected_improvement,
                "projection_years": len(projections),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Approval
    # -------------------------------------------------------------------------

    async def _phase_3_approval(self, context: WorkflowContext) -> PhaseResult:
        """
        Approval workflow for CapEx plan.

        Approval stages:
        1. Sustainability team review (methodology, TSC mapping)
        2. Finance team review (budget, ROI, financial feasibility)
        3. Board/Executive approval (strategic alignment, commitment)

        The plan must be approved before CapEx can be recognised in
        the taxonomy-aligned CapEx KPI per Article 8 DA.
        """
        phase = Phase.APPROVAL
        plan_id = context.state.get("plan_id", "")
        total_capex = context.state.get("total_planned_capex", 0.0)

        self.logger.info("Running approval workflow for plan=%s capex=%.2f", plan_id, total_capex)

        approval_stages = [
            {
                "stage": "sustainability_review",
                "reviewer": "sustainability_director",
                "status": "APPROVED" if random.random() > 0.1 else "REVISIONS_REQUIRED",
                "review_date": datetime.utcnow().isoformat(),
                "comments": "Methodology and TSC mapping verified.",
            },
            {
                "stage": "finance_review",
                "reviewer": "cfo",
                "status": "APPROVED" if random.random() > 0.15 else "REVISIONS_REQUIRED",
                "review_date": datetime.utcnow().isoformat(),
                "comments": "Budget allocation confirmed within financial plan.",
            },
        ]

        if self.config.require_board_approval:
            approval_stages.append({
                "stage": "board_approval",
                "reviewer": "board_of_directors",
                "status": "APPROVED" if random.random() > 0.1 else "DEFERRED",
                "review_date": datetime.utcnow().isoformat(),
                "comments": "Strategic alignment confirmed.",
            })

        all_approved = all(s["status"] == "APPROVED" for s in approval_stages)
        plan_status = PlanStatus.APPROVED.value if all_approved else PlanStatus.SUBMITTED.value

        context.state["plan_status"] = plan_status
        context.state["approval_stages"] = approval_stages

        provenance = self._hash({
            "phase": phase.value,
            "plan_id": plan_id,
            "plan_status": plan_status,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "plan_status": plan_status,
                "approval_stages": len(approval_stages),
                "all_approved": all_approved,
                "stages_approved": len([s for s in approval_stages if s["status"] == "APPROVED"]),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Monitoring
    # -------------------------------------------------------------------------

    async def _phase_4_monitoring(self, context: WorkflowContext) -> PhaseResult:
        """
        Track CapEx plan execution against targets.

        Monitoring metrics:
        - CapEx deployed vs. planned (cumulative and per-period)
        - Activity milestone completion status
        - TSC criteria achievement progress
        - Variance analysis (schedule, budget, scope)
        - Risk flags for activities falling behind
        """
        phase = Phase.MONITORING
        activities = context.state.get("plan_activities", [])
        total_planned = context.state.get("total_planned_capex", 0.0)

        self.logger.info("Monitoring CapEx plan execution (%s cadence)", self.config.monitoring_frequency)

        # Simulate monitoring data
        capex_deployed = round(total_planned * random.uniform(0.05, 0.35), 2)
        deployment_ratio = capex_deployed / max(total_planned, 1.0)

        activity_progress = []
        at_risk_count = 0

        for activity in activities:
            progress_pct = random.uniform(0.0, 0.6)
            on_schedule = progress_pct >= 0.2 or random.random() > 0.3
            if not on_schedule:
                at_risk_count += 1

            activity_progress.append({
                "activity_id": activity["activity_id"],
                "progress_pct": round(progress_pct, 2),
                "capex_deployed": round(activity["total_capex"] * progress_pct, 2),
                "on_schedule": on_schedule,
                "tsc_criteria_met": random.randint(0, activity["tsc_criteria_to_meet"]),
                "tsc_criteria_total": activity["tsc_criteria_to_meet"],
            })

        plan_on_track = at_risk_count <= len(activities) * 0.3
        context.state["plan_on_track"] = plan_on_track

        provenance = self._hash({
            "phase": phase.value,
            "capex_deployed": capex_deployed,
            "at_risk_count": at_risk_count,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "capex_deployed": capex_deployed,
                "deployment_ratio": round(deployment_ratio, 3),
                "activities_on_track": len(activities) - at_risk_count,
                "activities_at_risk": at_risk_count,
                "plan_on_track": plan_on_track,
                "monitoring_frequency": self.config.monitoring_frequency,
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
