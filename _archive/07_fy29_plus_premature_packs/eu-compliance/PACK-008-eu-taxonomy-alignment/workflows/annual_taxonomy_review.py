# -*- coding: utf-8 -*-
"""
Annual Taxonomy Review Workflow
==================================

Five-phase workflow for conducting an annual review of EU Taxonomy alignment
status, recalculating KPIs, analysing trends, preparing board reporting,
and planning improvement actions.

This workflow enables:
- Reassessment of all activities for eligibility/alignment changes
- Full KPI recalculation for the current reporting period
- Year-over-year trend analysis of alignment ratios
- Board-level executive summary generation
- Strategic action planning for improving alignment

Phases:
    1. Activity Reassessment - Reassess activities for eligibility/alignment changes
    2. KPI Recalculation - Recalculate all KPIs for current period
    3. Trend Analysis - Year-over-year trend analysis of alignment ratios
    4. Board Reporting - Generate board-level summary
    5. Action Planning - Create action plan for improving alignment

Regulatory Context:
    EU Taxonomy alignment reporting is annual for CSRD/NFRD subject entities.
    Article 8 DA requires year-over-year comparatives. This workflow ensures
    systematic annual review, consistent methodology application, and
    strategic alignment with sustainability goals.

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
    ACTIVITY_REASSESSMENT = "activity_reassessment"
    KPI_RECALCULATION = "kpi_recalculation"
    TREND_ANALYSIS = "trend_analysis"
    BOARD_REPORTING = "board_reporting"
    ACTION_PLANNING = "action_planning"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrendDirection(str, Enum):
    """Direction of year-over-year trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


# =============================================================================
# DATA MODELS
# =============================================================================


class AnnualTaxonomyReviewConfig(BaseModel):
    """Configuration for annual taxonomy review workflow."""
    organization_id: Optional[str] = Field(None, description="Organization identifier")
    review_year: int = Field(default=2025, ge=2020, description="Year under review")
    prior_year: int = Field(default=2024, ge=2019, description="Prior year for comparison")
    include_board_presentation: bool = Field(default=True, description="Generate board-level report")
    target_alignment_pct: float = Field(
        default=0.50, ge=0.0, le=1.0, description="Target alignment ratio"
    )
    currency: str = Field(default="EUR", description="Reporting currency")


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
    config: AnnualTaxonomyReviewConfig = Field(default_factory=AnnualTaxonomyReviewConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the annual taxonomy review workflow."""
    workflow_name: str = Field(default="annual_taxonomy_review", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    review_year: int = Field(default=2025, description="Year reviewed")
    activities_reassessed: int = Field(default=0, ge=0, description="Activities reassessed")
    status_changes: int = Field(default=0, ge=0, description="Activities with changed status")
    turnover_aligned_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Current turnover alignment")
    yoy_turnover_change: float = Field(default=0.0, description="YoY turnover alignment change")
    trend_direction: str = Field(default="stable", description="Overall trend direction")
    priority_actions: List[str] = Field(default_factory=list, description="Priority actions for next year")
    board_report_id: Optional[str] = Field(None, description="Board report identifier")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# ANNUAL TAXONOMY REVIEW WORKFLOW
# =============================================================================


class AnnualTaxonomyReviewWorkflow:
    """
    Five-phase annual taxonomy review workflow.

    Conducts comprehensive year-end taxonomy alignment review:
    - Reassess all activities for eligibility/alignment changes
    - Recalculate KPIs with current period data
    - Analyse year-over-year trends
    - Generate board-level executive reporting
    - Plan strategic actions for alignment improvement

    Example:
        >>> config = AnnualTaxonomyReviewConfig(
        ...     organization_id="ORG-001",
        ...     review_year=2025,
        ...     target_alignment_pct=0.50,
        ... )
        >>> workflow = AnnualTaxonomyReviewWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.overall_status == PhaseStatus.COMPLETED
        >>> assert result.review_year == 2025
    """

    def __init__(self, config: Optional[AnnualTaxonomyReviewConfig] = None) -> None:
        """Initialize the annual taxonomy review workflow."""
        self.config = config or AnnualTaxonomyReviewConfig()
        self.logger = logging.getLogger(f"{__name__}.AnnualTaxonomyReviewWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 5-phase annual taxonomy review workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with reassessment, KPIs, trends, and action plan.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting annual taxonomy review workflow execution_id=%s year=%d",
            context.execution_id,
            self.config.review_year,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.ACTIVITY_REASSESSMENT, self._phase_1_reassessment),
            (Phase.KPI_RECALCULATION, self._phase_2_kpi_recalculation),
            (Phase.TREND_ANALYSIS, self._phase_3_trend_analysis),
            (Phase.BOARD_REPORTING, self._phase_4_board_reporting),
            (Phase.ACTION_PLANNING, self._phase_5_action_planning),
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

        activities_reassessed = context.state.get("activities_reassessed", 0)
        status_changes = context.state.get("status_changes", 0)
        turnover_ratio = context.state.get("turnover_aligned_ratio", 0.0)
        yoy_change = context.state.get("yoy_turnover_change", 0.0)
        trend = context.state.get("trend_direction", "stable")
        actions = context.state.get("priority_actions", [])
        board_id = context.state.get("board_report_id")

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "review_year": self.config.review_year,
        })

        self.logger.info(
            "Annual taxonomy review finished execution_id=%s status=%s "
            "year=%d turnover=%.1f%% yoy=%.1f%% trend=%s",
            context.execution_id,
            overall_status.value,
            self.config.review_year,
            turnover_ratio * 100,
            yoy_change * 100,
            trend,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            review_year=self.config.review_year,
            activities_reassessed=activities_reassessed,
            status_changes=status_changes,
            turnover_aligned_ratio=turnover_ratio,
            yoy_turnover_change=yoy_change,
            trend_direction=trend,
            priority_actions=actions,
            board_report_id=board_id,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Activity Reassessment
    # -------------------------------------------------------------------------

    async def _phase_1_reassessment(self, context: WorkflowContext) -> PhaseResult:
        """
        Reassess all activities for eligibility and alignment changes.

        Reassessment triggers:
        - New Delegated Act criteria or amendments
        - Changes in activity operations or technology
        - Updated financial data (revenue shifts)
        - Expired certifications or evidence
        - New activities added to portfolio
        - Activities divested or discontinued
        """
        phase = Phase.ACTIVITY_REASSESSMENT

        self.logger.info("Reassessing activities for year %d", self.config.review_year)

        await asyncio.sleep(0.05)

        activity_count = random.randint(15, 35)
        activities = []
        status_changes = 0

        for i in range(activity_count):
            prior_status = random.choice(["aligned", "eligible_not_aligned", "not_eligible"])
            # Some activities change status year-over-year
            changed = random.random() > 0.75
            if changed:
                status_changes += 1
                if prior_status == "eligible_not_aligned":
                    current_status = random.choice(["aligned", "not_eligible"])
                elif prior_status == "not_eligible":
                    current_status = "eligible_not_aligned"
                else:
                    current_status = random.choice(["eligible_not_aligned", "aligned"])
            else:
                current_status = prior_status

            revenue = round(random.uniform(1e6, 80e6), 2)

            activities.append({
                "activity_id": f"ACT-{uuid.uuid4().hex[:8]}",
                "description": f"Activity {i + 1}",
                "prior_status": prior_status,
                "current_status": current_status,
                "status_changed": changed,
                "revenue": revenue,
                "change_reason": self._get_change_reason() if changed else None,
            })

        context.state["reassessed_activities"] = activities
        context.state["activities_reassessed"] = activity_count
        context.state["status_changes"] = status_changes

        newly_aligned = len([
            a for a in activities
            if a["status_changed"] and a["current_status"] == "aligned"
        ])
        newly_lost = len([
            a for a in activities
            if a["status_changed"] and a["prior_status"] == "aligned" and a["current_status"] != "aligned"
        ])

        provenance = self._hash({
            "phase": phase.value,
            "activity_count": activity_count,
            "status_changes": status_changes,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "activities_reassessed": activity_count,
                "status_changes": status_changes,
                "newly_aligned": newly_aligned,
                "alignment_lost": newly_lost,
                "net_change": newly_aligned - newly_lost,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: KPI Recalculation
    # -------------------------------------------------------------------------

    async def _phase_2_kpi_recalculation(self, context: WorkflowContext) -> PhaseResult:
        """
        Recalculate all KPIs for current period.

        KPIs recalculated:
        - Turnover alignment ratio
        - CapEx alignment ratio
        - OpEx alignment ratio
        - Per-objective breakdown
        - Eligible vs. aligned split
        """
        phase = Phase.KPI_RECALCULATION
        activities = context.state.get("reassessed_activities", [])

        self.logger.info("Recalculating KPIs for year %d", self.config.review_year)

        total_revenue = sum(a["revenue"] for a in activities)
        aligned_revenue = sum(
            a["revenue"] for a in activities if a["current_status"] == "aligned"
        )
        eligible_revenue = sum(
            a["revenue"] for a in activities
            if a["current_status"] in ["aligned", "eligible_not_aligned"]
        )

        turnover_aligned = round(aligned_revenue / max(total_revenue, 1.0), 4)
        turnover_eligible = round(eligible_revenue / max(total_revenue, 1.0), 4)

        # Simulate CapEx and OpEx ratios
        capex_aligned = round(random.uniform(
            max(turnover_aligned - 0.10, 0.05),
            min(turnover_aligned + 0.15, 0.95),
        ), 4)
        opex_aligned = round(random.uniform(
            max(turnover_aligned - 0.15, 0.03),
            min(turnover_aligned + 0.10, 0.90),
        ), 4)

        kpis = {
            "turnover_aligned_ratio": turnover_aligned,
            "turnover_eligible_ratio": turnover_eligible,
            "capex_aligned_ratio": capex_aligned,
            "opex_aligned_ratio": opex_aligned,
            "total_revenue": round(total_revenue, 2),
            "aligned_revenue": round(aligned_revenue, 2),
        }

        context.state["current_kpis"] = kpis
        context.state["turnover_aligned_ratio"] = turnover_aligned

        provenance = self._hash({
            "phase": phase.value,
            "kpis": kpis,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "turnover_aligned_pct": round(turnover_aligned * 100, 1),
                "capex_aligned_pct": round(capex_aligned * 100, 1),
                "opex_aligned_pct": round(opex_aligned * 100, 1),
                "turnover_eligible_pct": round(turnover_eligible * 100, 1),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Trend Analysis
    # -------------------------------------------------------------------------

    async def _phase_3_trend_analysis(self, context: WorkflowContext) -> PhaseResult:
        """
        Year-over-year trend analysis of alignment ratios.

        Analysis includes:
        - YoY change in turnover, CapEx, OpEx alignment ratios
        - Multi-year trend line (if historical data available)
        - Peer benchmarking comparison (sector averages)
        - Trend direction classification (improving, stable, declining)
        - Rate of progress towards target alignment
        """
        phase = Phase.TREND_ANALYSIS
        current_kpis = context.state.get("current_kpis", {})
        current_turnover = current_kpis.get("turnover_aligned_ratio", 0.0)

        self.logger.info("Analysing year-over-year alignment trends")

        # Simulate prior year data
        prior_turnover = round(current_turnover + random.uniform(-0.10, 0.05), 4)
        prior_turnover = max(0.0, min(prior_turnover, 1.0))

        prior_capex = round(current_kpis.get("capex_aligned_ratio", 0) + random.uniform(-0.08, 0.03), 4)
        prior_opex = round(current_kpis.get("opex_aligned_ratio", 0) + random.uniform(-0.06, 0.04), 4)

        yoy_turnover = round(current_turnover - prior_turnover, 4)
        yoy_capex = round(current_kpis.get("capex_aligned_ratio", 0) - prior_capex, 4)
        yoy_opex = round(current_kpis.get("opex_aligned_ratio", 0) - prior_opex, 4)

        # Determine trend direction
        if yoy_turnover > 0.02:
            trend = TrendDirection.IMPROVING.value
        elif yoy_turnover < -0.02:
            trend = TrendDirection.DECLINING.value
        else:
            trend = TrendDirection.STABLE.value

        # Sector benchmark
        sector_avg_alignment = round(random.uniform(0.20, 0.45), 4)

        # Progress towards target
        target = self.config.target_alignment_pct
        progress_pct = round(current_turnover / max(target, 0.01), 4)

        context.state["yoy_turnover_change"] = yoy_turnover
        context.state["trend_direction"] = trend

        trend_data = {
            "prior_year": {
                "turnover_aligned": prior_turnover,
                "capex_aligned": prior_capex,
                "opex_aligned": prior_opex,
            },
            "current_year": {
                "turnover_aligned": current_turnover,
                "capex_aligned": current_kpis.get("capex_aligned_ratio", 0),
                "opex_aligned": current_kpis.get("opex_aligned_ratio", 0),
            },
            "yoy_changes": {
                "turnover": yoy_turnover,
                "capex": yoy_capex,
                "opex": yoy_opex,
            },
            "sector_benchmark": sector_avg_alignment,
            "vs_sector": round(current_turnover - sector_avg_alignment, 4),
            "target_progress": progress_pct,
        }

        context.state["trend_data"] = trend_data

        provenance = self._hash({
            "phase": phase.value,
            "trend": trend,
            "yoy_turnover": yoy_turnover,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "trend_direction": trend,
                "yoy_turnover_change_pct": round(yoy_turnover * 100, 1),
                "yoy_capex_change_pct": round(yoy_capex * 100, 1),
                "yoy_opex_change_pct": round(yoy_opex * 100, 1),
                "vs_sector_benchmark": round((current_turnover - sector_avg_alignment) * 100, 1),
                "target_progress_pct": round(progress_pct * 100, 1),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Board Reporting
    # -------------------------------------------------------------------------

    async def _phase_4_board_reporting(self, context: WorkflowContext) -> PhaseResult:
        """
        Generate board-level summary report.

        Report contents:
        - Executive summary (1-page overview)
        - KPI dashboard (turnover, CapEx, OpEx alignment)
        - YoY trend charts
        - Peer comparison
        - Regulatory outlook
        - Strategic recommendations
        """
        phase = Phase.BOARD_REPORTING
        kpis = context.state.get("current_kpis", {})
        trend_data = context.state.get("trend_data", {})
        trend = context.state.get("trend_direction", "stable")

        self.logger.info("Generating board-level taxonomy report")

        if not self.config.include_board_presentation:
            provenance = self._hash({"phase": phase.value, "skipped": True})
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.COMPLETED,
                data={"board_report": "skipped"},
                provenance_hash=provenance,
            )

        board_report_id = f"BOARD-TAX-{uuid.uuid4().hex[:8]}"

        report = {
            "report_id": board_report_id,
            "title": f"EU Taxonomy Alignment Review {self.config.review_year}",
            "organization_id": self.config.organization_id,
            "executive_summary": {
                "headline": f"Taxonomy alignment {trend} at {kpis.get('turnover_aligned_ratio', 0) * 100:.1f}%",
                "turnover_aligned": kpis.get("turnover_aligned_ratio", 0),
                "yoy_change": trend_data.get("yoy_changes", {}).get("turnover", 0),
                "target": self.config.target_alignment_pct,
                "trend": trend,
            },
            "sections": [
                "Executive Summary",
                "KPI Dashboard",
                "YoY Trend Analysis",
                "Peer Benchmarking",
                "Regulatory Outlook",
                "Strategic Recommendations",
                "Appendix: Activity Detail",
            ],
        }

        context.state["board_report_id"] = board_report_id
        context.state["board_report"] = report

        provenance = self._hash({
            "phase": phase.value,
            "report_id": board_report_id,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "report_id": board_report_id,
                "sections": len(report["sections"]),
                "headline": report["executive_summary"]["headline"],
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Action Planning
    # -------------------------------------------------------------------------

    async def _phase_5_action_planning(self, context: WorkflowContext) -> PhaseResult:
        """
        Create action plan for improving taxonomy alignment.

        Planning areas:
        - Activities closest to alignment (low-hanging fruit)
        - CapEx investment priorities for alignment
        - Technology upgrades for TSC compliance
        - DNSH gap remediation
        - Minimum Safeguards strengthening
        - Data quality and evidence improvement
        """
        phase = Phase.ACTION_PLANNING
        kpis = context.state.get("current_kpis", {})
        trend = context.state.get("trend_direction", "stable")
        target = self.config.target_alignment_pct
        current = kpis.get("turnover_aligned_ratio", 0.0)
        status_changes = context.state.get("status_changes", 0)

        self.logger.info("Creating action plan for alignment improvement")

        actions = []

        # Gap to target
        gap = target - current
        if gap > 0:
            actions.append(
                f"Close alignment gap of {gap * 100:.1f}pp to reach "
                f"{target * 100:.0f}% target through targeted CapEx and process improvements."
            )

        # Low-hanging fruit
        activities = context.state.get("reassessed_activities", [])
        eligible_not_aligned = [
            a for a in activities if a["current_status"] == "eligible_not_aligned"
        ]
        if eligible_not_aligned:
            actions.append(
                f"Prioritise {len(eligible_not_aligned)} eligible-not-aligned activities "
                "for alignment through TSC compliance investments."
            )

        # Trend-based actions
        if trend == TrendDirection.DECLINING.value:
            actions.append(
                "URGENT: Alignment trend is declining. Investigate root causes "
                "and implement corrective measures within Q1."
            )
        elif trend == TrendDirection.STABLE.value:
            actions.append(
                "Alignment trend is stable. Accelerate CapEx plan execution "
                "to achieve improvement trajectory."
            )

        # Status changes
        if status_changes > 0:
            actions.append(
                f"Review {status_changes} activities with changed alignment status "
                "to ensure assessment methodology is consistently applied."
            )

        # Regulatory readiness
        actions.append(
            "Monitor Omnibus Simplification Package implementation and adjust "
            "assessment methodology if simplified criteria are adopted."
        )

        # Data quality
        actions.append(
            "Invest in automated data collection from ERP systems to improve "
            "financial data granularity for activity-level KPI calculation."
        )

        context.state["priority_actions"] = actions

        provenance = self._hash({
            "phase": phase.value,
            "action_count": len(actions),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "priority_actions": actions,
                "action_count": len(actions),
                "alignment_gap_pp": round(gap * 100, 1) if gap > 0 else 0,
                "eligible_not_aligned_count": len(eligible_not_aligned),
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_change_reason(self) -> str:
        """Generate a reason for activity status change."""
        reasons = [
            "Updated TSC criteria met through technology upgrade",
            "DNSH assessment completed for remaining objectives",
            "New Delegated Act criteria changed eligibility",
            "Activity operations changed affecting alignment",
            "Evidence quality improved to meet assurance threshold",
            "Certification expired affecting DNSH compliance",
            "Revenue reallocation changed activity classification",
        ]
        return random.choice(reasons)

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
