# -*- coding: utf-8 -*-
"""
Improvement Planning Workflow
=================================

4-phase workflow for identifying data quality gaps, evaluating improvement
options, and building actionable roadmaps for the next inventory cycle
within PACK-044 GHG Inventory Management Pack.

Phases:
    1. GapIdentification    -- Analyze current inventory quality scores,
                               identify data gaps across facilities and scopes,
                               benchmark against best practices, classify
                               gap severity and root causes
    2. Options              -- Generate improvement options for each gap,
                               estimate cost-benefit of each option, evaluate
                               technical feasibility and resource requirements
    3. Roadmap              -- Prioritize improvements using weighted scoring,
                               sequence actions across quarters, allocate
                               resources and assign ownership
    4. ActionPlanning       -- Create detailed action items with milestones,
                               define success criteria and KPIs, generate
                               improvement plan document for sign-off

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapter 7 (Managing Inventory Quality)
    ISO 14064-1:2018 Clause 8 (Quality management, continual improvement)
    ISO 14001:2015 Clause 10 (Improvement)

Schedule: End of annual inventory cycle
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ImprovementPhase(str, Enum):
    """Improvement planning workflow phases."""

    GAP_IDENTIFICATION = "gap_identification"
    OPTIONS = "options"
    ROADMAP = "roadmap"
    ACTION_PLANNING = "action_planning"


class GapCategory(str, Enum):
    """Gap category classification."""

    DATA_COMPLETENESS = "data_completeness"
    DATA_ACCURACY = "data_accuracy"
    DATA_TIMELINESS = "data_timeliness"
    METHODOLOGY = "methodology"
    SCOPE_COVERAGE = "scope_coverage"
    EMISSION_FACTOR = "emission_factor"
    AUTOMATION = "automation"
    DOCUMENTATION = "documentation"
    VERIFICATION_READINESS = "verification_readiness"


class GapSeverity(str, Enum):
    """Gap severity classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeasibilityLevel(str, Enum):
    """Technical feasibility level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionStatus(str, Enum):
    """Action item status."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class Quarter(str, Enum):
    """Calendar quarter."""

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class InventoryGap(BaseModel):
    """Identified gap in inventory quality or coverage."""

    gap_id: str = Field(default_factory=lambda: f"gap-{uuid.uuid4().hex[:8]}")
    category: GapCategory = Field(default=GapCategory.DATA_COMPLETENESS)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    facility_ids: List[str] = Field(default_factory=list)
    scope: str = Field(default="", description="scope1|scope2|scope3|cross_cutting")
    description: str = Field(default="")
    current_state: str = Field(default="")
    target_state: str = Field(default="")
    root_cause: str = Field(default="")
    impact_on_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)


class ImprovementOption(BaseModel):
    """Improvement option to address an identified gap."""

    option_id: str = Field(default_factory=lambda: f"opt-{uuid.uuid4().hex[:8]}")
    gap_id: str = Field(default="", description="Related gap ID")
    description: str = Field(default="")
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    estimated_effort_hours: float = Field(default=0.0, ge=0.0)
    expected_quality_improvement: float = Field(default=0.0, ge=0.0, le=100.0)
    feasibility: FeasibilityLevel = Field(default=FeasibilityLevel.MEDIUM)
    time_to_implement_weeks: int = Field(default=0, ge=0)
    requires_external_resources: bool = Field(default=False)
    cost_benefit_ratio: float = Field(default=0.0, ge=0.0)


class RoadmapItem(BaseModel):
    """Prioritized item in the improvement roadmap."""

    roadmap_item_id: str = Field(default_factory=lambda: f"rmi-{uuid.uuid4().hex[:8]}")
    option_id: str = Field(default="", description="Selected improvement option")
    gap_id: str = Field(default="")
    priority_rank: int = Field(default=0, ge=0)
    priority_score: float = Field(default=0.0, ge=0.0, le=100.0)
    target_quarter: Quarter = Field(default=Quarter.Q1)
    target_year: int = Field(default=2026)
    owner_id: str = Field(default="")
    owner_name: str = Field(default="")
    resource_allocation_hours: float = Field(default=0.0, ge=0.0)
    budget_eur: float = Field(default=0.0, ge=0.0)


class ActionItem(BaseModel):
    """Detailed action item with milestones."""

    action_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    roadmap_item_id: str = Field(default="")
    description: str = Field(default="")
    status: ActionStatus = Field(default=ActionStatus.PLANNED)
    owner_id: str = Field(default="")
    start_date: str = Field(default="", description="ISO date")
    due_date: str = Field(default="", description="ISO date")
    milestones: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list, description="Dependent action IDs")


class ImprovementPlanSummary(BaseModel):
    """Summary of the complete improvement plan."""

    total_gaps: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    total_options_evaluated: int = Field(default=0, ge=0)
    options_selected: int = Field(default=0, ge=0)
    total_actions: int = Field(default=0, ge=0)
    total_budget_eur: float = Field(default=0.0, ge=0.0)
    total_effort_hours: float = Field(default=0.0, ge=0.0)
    expected_quality_improvement: float = Field(default=0.0, ge=0.0, le=100.0)
    target_year: int = Field(default=2026)
    plan_provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class ImprovementPlanningInput(BaseModel):
    """Input data model for ImprovementPlanningWorkflow."""

    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    quality_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality scores by dimension (completeness, accuracy, etc.)",
    )
    facility_gaps: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-facility gap descriptions keyed by facility_id",
    )
    scope_coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Coverage percentage by scope (scope1, scope2, scope3)",
    )
    current_overall_quality: float = Field(
        default=75.0, ge=0.0, le=100.0,
        description="Current overall inventory quality score",
    )
    target_quality: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Target quality score for next cycle",
    )
    available_budget_eur: float = Field(default=50000.0, ge=0.0)
    available_hours: float = Field(default=500.0, ge=0.0)
    team_members: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Team member dicts with id, name, role",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ImprovementPlanningResult(BaseModel):
    """Complete result from improvement planning workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="improvement_planning")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    gaps: List[InventoryGap] = Field(default_factory=list)
    options: List[ImprovementOption] = Field(default_factory=list)
    roadmap: List[RoadmapItem] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    plan_summary: Optional[ImprovementPlanSummary] = Field(default=None)
    provenance_hash: str = Field(default="")


# =============================================================================
# QUALITY BENCHMARK DATA (Zero-Hallucination)
# =============================================================================

# Best-practice quality score targets by dimension (GHG Protocol/IPCC)
QUALITY_BENCHMARKS: Dict[str, float] = {
    "completeness": 95.0,
    "accuracy": 90.0,
    "consistency": 90.0,
    "transparency": 85.0,
    "timeliness": 85.0,
    "relevance": 90.0,
}

# Standard improvement options by gap category with baseline estimates
STANDARD_OPTIONS: Dict[str, List[Dict[str, Any]]] = {
    "data_completeness": [
        {"description": "Implement automated data feeds from ERP", "cost": 15000, "hours": 120, "improvement": 15.0, "weeks": 8},
        {"description": "Deploy data collection questionnaires with validation", "cost": 5000, "hours": 40, "improvement": 10.0, "weeks": 4},
    ],
    "data_accuracy": [
        {"description": "Add real-time validation rules at data entry", "cost": 8000, "hours": 60, "improvement": 12.0, "weeks": 6},
        {"description": "Implement cross-source reconciliation checks", "cost": 10000, "hours": 80, "improvement": 10.0, "weeks": 6},
    ],
    "methodology": [
        {"description": "Upgrade from Tier 1 to Tier 2 methodology", "cost": 20000, "hours": 160, "improvement": 8.0, "weeks": 12},
        {"description": "Engage sector-specific emission factor databases", "cost": 5000, "hours": 30, "improvement": 5.0, "weeks": 4},
    ],
    "automation": [
        {"description": "Automate utility bill processing with OCR", "cost": 12000, "hours": 100, "improvement": 10.0, "weeks": 8},
        {"description": "Build API integrations with energy suppliers", "cost": 18000, "hours": 140, "improvement": 12.0, "weeks": 10},
    ],
    "scope_coverage": [
        {"description": "Expand Scope 3 categories from 3 to 8", "cost": 25000, "hours": 200, "improvement": 15.0, "weeks": 16},
        {"description": "Add upstream transportation and distribution", "cost": 10000, "hours": 80, "improvement": 8.0, "weeks": 8},
    ],
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ImprovementPlanningWorkflow:
    """
    4-phase improvement planning workflow for GHG inventory management.

    Analyzes current quality performance, identifies gaps, evaluates
    improvement options with cost-benefit analysis, and produces a
    prioritized roadmap with detailed action items.

    Zero-hallucination: all quality gaps derived from score comparisons
    against published benchmarks, all cost-benefit from deterministic
    formulas, priority scoring from weighted criteria, no LLM calls.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _gaps: Identified inventory gaps.
        _options: Evaluated improvement options.
        _roadmap: Prioritized roadmap items.
        _actions: Detailed action items.

    Example:
        >>> wf = ImprovementPlanningWorkflow()
        >>> inp = ImprovementPlanningInput(current_overall_quality=70.0)
        >>> result = await wf.execute(inp)
        >>> assert len(result.gaps) > 0
    """

    PHASE_SEQUENCE: List[ImprovementPhase] = [
        ImprovementPhase.GAP_IDENTIFICATION,
        ImprovementPhase.OPTIONS,
        ImprovementPhase.ROADMAP,
        ImprovementPhase.ACTION_PLANNING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    # Priority scoring weights
    PRIORITY_WEIGHTS: Dict[str, float] = {
        "severity": 0.30,
        "quality_impact": 0.25,
        "feasibility": 0.20,
        "cost_benefit": 0.25,
    }

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ImprovementPlanningWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._gaps: List[InventoryGap] = []
        self._options: List[ImprovementOption] = []
        self._roadmap: List[RoadmapItem] = []
        self._actions: List[ActionItem] = []
        self._summary: Optional[ImprovementPlanSummary] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: ImprovementPlanningInput) -> ImprovementPlanningResult:
        """
        Execute the 4-phase improvement planning workflow.

        Args:
            input_data: Quality scores, gaps, budget, and team information.

        Returns:
            ImprovementPlanningResult with gaps, options, roadmap, actions.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting improvement planning %s year=%d quality=%.1f target=%.1f",
            self.workflow_id, input_data.reporting_year,
            input_data.current_overall_quality, input_data.target_quality,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_gap_identification,
            self._phase_options,
            self._phase_roadmap,
            self._phase_action_planning,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._execute_with_retry(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Improvement planning failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = ImprovementPlanningResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            reporting_year=input_data.reporting_year,
            gaps=self._gaps,
            options=self._options,
            roadmap=self._roadmap,
            action_items=self._actions,
            plan_summary=self._summary,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Improvement planning %s completed in %.2fs status=%s gaps=%d actions=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._gaps), len(self._actions),
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: ImprovementPlanningInput, phase_number: int
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Gap Identification
    # -------------------------------------------------------------------------

    async def _phase_gap_identification(self, input_data: ImprovementPlanningInput) -> PhaseResult:
        """Analyze quality scores and identify gaps against benchmarks."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []

        # Compare quality dimensions against benchmarks
        for dimension, benchmark in QUALITY_BENCHMARKS.items():
            current = input_data.quality_scores.get(dimension, 0.0)
            if current < benchmark:
                gap_delta = benchmark - current
                severity = GapSeverity.LOW
                if gap_delta > 30.0:
                    severity = GapSeverity.CRITICAL
                elif gap_delta > 20.0:
                    severity = GapSeverity.HIGH
                elif gap_delta > 10.0:
                    severity = GapSeverity.MEDIUM

                category = self._map_dimension_to_gap_category(dimension)

                self._gaps.append(InventoryGap(
                    category=category,
                    severity=severity,
                    scope="cross_cutting",
                    description=f"{dimension} score ({current:.1f}) below benchmark ({benchmark:.1f})",
                    current_state=f"Score: {current:.1f}",
                    target_state=f"Score: {benchmark:.1f}",
                    root_cause=f"Insufficient {dimension} controls",
                    impact_on_quality_score=round(gap_delta * 0.2, 2),
                ))

        # Identify facility-level gaps
        for fac_id, gap_descriptions in input_data.facility_gaps.items():
            for desc in gap_descriptions:
                self._gaps.append(InventoryGap(
                    category=GapCategory.DATA_COMPLETENESS,
                    severity=GapSeverity.MEDIUM,
                    facility_ids=[fac_id],
                    scope="scope1",
                    description=desc,
                    current_state="Gap identified",
                    target_state="Gap resolved",
                    root_cause="Facility data collection incomplete",
                    impact_on_quality_score=2.0,
                ))

        # Identify scope coverage gaps
        for scope, coverage in input_data.scope_coverage.items():
            if coverage < 95.0:
                self._gaps.append(InventoryGap(
                    category=GapCategory.SCOPE_COVERAGE,
                    severity=GapSeverity.HIGH if coverage < 80.0 else GapSeverity.MEDIUM,
                    scope=scope,
                    description=f"{scope} coverage at {coverage:.1f}%, target 95%+",
                    current_state=f"{coverage:.1f}% coverage",
                    target_state="95%+ coverage",
                    root_cause=f"Incomplete {scope} data collection",
                    impact_on_quality_score=round((95.0 - coverage) * 0.15, 2),
                ))

        critical = sum(1 for g in self._gaps if g.severity == GapSeverity.CRITICAL)
        high = sum(1 for g in self._gaps if g.severity == GapSeverity.HIGH)

        outputs["total_gaps"] = len(self._gaps)
        outputs["critical"] = critical
        outputs["high"] = high
        outputs["medium"] = sum(1 for g in self._gaps if g.severity == GapSeverity.MEDIUM)
        outputs["low"] = sum(1 for g in self._gaps if g.severity == GapSeverity.LOW)
        outputs["gap_categories"] = list(set(g.category.value for g in self._gaps))
        outputs["quality_gap"] = round(
            input_data.target_quality - input_data.current_overall_quality, 2
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 GapIdentification: %d gaps (%d critical, %d high)",
            len(self._gaps), critical, high,
        )
        return PhaseResult(
            phase_name="gap_identification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _map_dimension_to_gap_category(self, dimension: str) -> GapCategory:
        """Map quality dimension to gap category."""
        mapping: Dict[str, GapCategory] = {
            "completeness": GapCategory.DATA_COMPLETENESS,
            "accuracy": GapCategory.DATA_ACCURACY,
            "consistency": GapCategory.METHODOLOGY,
            "transparency": GapCategory.DOCUMENTATION,
            "timeliness": GapCategory.DATA_TIMELINESS,
            "relevance": GapCategory.SCOPE_COVERAGE,
        }
        return mapping.get(dimension, GapCategory.DATA_COMPLETENESS)

    # -------------------------------------------------------------------------
    # Phase 2: Options
    # -------------------------------------------------------------------------

    async def _phase_options(self, input_data: ImprovementPlanningInput) -> PhaseResult:
        """Generate and evaluate improvement options for each gap."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._options = []

        for gap in self._gaps:
            cat_key = gap.category.value
            standard_opts = STANDARD_OPTIONS.get(cat_key, STANDARD_OPTIONS.get("data_completeness", []))

            for opt_template in standard_opts:
                cost = opt_template["cost"]
                hours = opt_template["hours"]
                improvement = opt_template["improvement"]
                weeks = opt_template["weeks"]

                # Deterministic cost-benefit ratio
                benefit_value = improvement * 1000.0  # Notional EUR value per quality point
                cb_ratio = round(benefit_value / max(cost, 1.0), 2)

                feasibility = FeasibilityLevel.HIGH
                if weeks > 10 or cost > 15000:
                    feasibility = FeasibilityLevel.MEDIUM
                if weeks > 14 or cost > 20000:
                    feasibility = FeasibilityLevel.LOW

                self._options.append(ImprovementOption(
                    gap_id=gap.gap_id,
                    description=opt_template["description"],
                    estimated_cost_eur=float(cost),
                    estimated_effort_hours=float(hours),
                    expected_quality_improvement=improvement,
                    feasibility=feasibility,
                    time_to_implement_weeks=weeks,
                    requires_external_resources=cost > 15000,
                    cost_benefit_ratio=cb_ratio,
                ))

        outputs["total_options"] = len(self._options)
        outputs["high_feasibility"] = sum(1 for o in self._options if o.feasibility == FeasibilityLevel.HIGH)
        outputs["total_estimated_cost"] = round(sum(o.estimated_cost_eur for o in self._options), 2)
        outputs["total_estimated_hours"] = round(sum(o.estimated_effort_hours for o in self._options), 1)
        outputs["avg_cost_benefit_ratio"] = round(
            sum(o.cost_benefit_ratio for o in self._options) / max(len(self._options), 1), 2
        )

        if not self._options:
            warnings.append("No improvement options generated; no gaps identified")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 Options: %d options evaluated, avg CB ratio=%.2f",
            len(self._options), outputs["avg_cost_benefit_ratio"],
        )
        return PhaseResult(
            phase_name="options", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Roadmap
    # -------------------------------------------------------------------------

    async def _phase_roadmap(self, input_data: ImprovementPlanningInput) -> PhaseResult:
        """Prioritize improvements and sequence into quarterly roadmap."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._roadmap = []
        next_year = input_data.reporting_year + 1

        # Select best option per gap (highest cost-benefit within budget)
        selected_options: List[ImprovementOption] = []
        remaining_budget = input_data.available_budget_eur
        remaining_hours = input_data.available_hours

        # Sort options by cost-benefit ratio descending
        sorted_options = sorted(self._options, key=lambda o: o.cost_benefit_ratio, reverse=True)

        # Deduplicate: pick best option per gap
        seen_gaps: set = set()
        for opt in sorted_options:
            if opt.gap_id in seen_gaps:
                continue
            if opt.estimated_cost_eur <= remaining_budget and opt.estimated_effort_hours <= remaining_hours:
                selected_options.append(opt)
                remaining_budget -= opt.estimated_cost_eur
                remaining_hours -= opt.estimated_effort_hours
                seen_gaps.add(opt.gap_id)

        # Assign to quarters based on implementation duration
        quarter_assignments = [Quarter.Q1, Quarter.Q2, Quarter.Q3, Quarter.Q4]
        for rank, opt in enumerate(selected_options, start=1):
            # Deterministic quarter assignment based on weeks
            quarter_idx = min((opt.time_to_implement_weeks - 1) // 4, 3)
            target_quarter = quarter_assignments[quarter_idx]

            # Priority score from weighted criteria
            gap = next((g for g in self._gaps if g.gap_id == opt.gap_id), None)
            severity_score = {"critical": 100, "high": 75, "medium": 50, "low": 25}.get(
                gap.severity.value if gap else "medium", 50
            )
            feasibility_score = {"high": 100, "medium": 60, "low": 30}.get(opt.feasibility.value, 60)
            quality_score = min(opt.expected_quality_improvement * 6.67, 100.0)
            cb_score = min(opt.cost_benefit_ratio * 20.0, 100.0)

            priority_score = round(
                severity_score * self.PRIORITY_WEIGHTS["severity"]
                + quality_score * self.PRIORITY_WEIGHTS["quality_impact"]
                + feasibility_score * self.PRIORITY_WEIGHTS["feasibility"]
                + cb_score * self.PRIORITY_WEIGHTS["cost_benefit"],
                2,
            )

            # Assign owner from team
            owner = input_data.team_members[rank % max(len(input_data.team_members), 1)] if input_data.team_members else {}

            self._roadmap.append(RoadmapItem(
                option_id=opt.option_id,
                gap_id=opt.gap_id,
                priority_rank=rank,
                priority_score=priority_score,
                target_quarter=target_quarter,
                target_year=next_year,
                owner_id=owner.get("id", ""),
                owner_name=owner.get("name", ""),
                resource_allocation_hours=opt.estimated_effort_hours,
                budget_eur=opt.estimated_cost_eur,
            ))

        outputs["roadmap_items"] = len(self._roadmap)
        outputs["selected_options"] = len(selected_options)
        outputs["total_budget_allocated"] = round(
            sum(r.budget_eur for r in self._roadmap), 2
        )
        outputs["total_hours_allocated"] = round(
            sum(r.resource_allocation_hours for r in self._roadmap), 1
        )
        outputs["budget_remaining"] = round(remaining_budget, 2)
        outputs["hours_remaining"] = round(remaining_hours, 1)
        outputs["items_by_quarter"] = {
            q.value: sum(1 for r in self._roadmap if r.target_quarter == q)
            for q in Quarter
        }

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 Roadmap: %d items, budget=%.0f EUR, hours=%.0f",
            len(self._roadmap), outputs["total_budget_allocated"],
            outputs["total_hours_allocated"],
        )
        return PhaseResult(
            phase_name="roadmap", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Action Planning
    # -------------------------------------------------------------------------

    async def _phase_action_planning(self, input_data: ImprovementPlanningInput) -> PhaseResult:
        """Create detailed action items with milestones and KPIs."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._actions = []
        next_year = input_data.reporting_year + 1

        for rmi in self._roadmap:
            opt = next((o for o in self._options if o.option_id == rmi.option_id), None)
            if not opt:
                continue

            # Determine dates from quarter
            quarter_starts = {"Q1": "01-01", "Q2": "04-01", "Q3": "07-01", "Q4": "10-01"}
            start_mmdd = quarter_starts.get(rmi.target_quarter.value, "01-01")
            start_date = f"{next_year}-{start_mmdd}"

            # Calculate due date
            weeks = opt.time_to_implement_weeks
            end_month = int(start_mmdd.split("-")[0]) + (weeks // 4)
            end_month = min(end_month, 12)
            due_date = f"{next_year}-{end_month:02d}-28"

            milestones = [
                f"Week 1: Kick-off and planning",
                f"Week {weeks // 2}: Mid-point review",
                f"Week {weeks}: Completion and validation",
            ]

            success_criteria = [
                f"Quality improvement of {opt.expected_quality_improvement:.1f} points achieved",
                "Implementation delivered within budget",
                "No new quality regressions introduced",
            ]

            kpis = [
                f"quality_score_delta >= {opt.expected_quality_improvement:.1f}",
                f"budget_utilization <= {opt.estimated_cost_eur:.0f} EUR",
                f"completion_on_time = true",
            ]

            self._actions.append(ActionItem(
                roadmap_item_id=rmi.roadmap_item_id,
                description=opt.description,
                status=ActionStatus.PLANNED,
                owner_id=rmi.owner_id,
                start_date=start_date,
                due_date=due_date,
                milestones=milestones,
                success_criteria=success_criteria,
                kpis=kpis,
            ))

        # Build plan summary
        total_improvement = sum(
            o.expected_quality_improvement
            for o in self._options
            if o.option_id in {r.option_id for r in self._roadmap}
        )

        plan_data = json.dumps({
            "workflow_id": self.workflow_id,
            "actions": len(self._actions),
            "budget": sum(r.budget_eur for r in self._roadmap),
        }, sort_keys=True)

        self._summary = ImprovementPlanSummary(
            total_gaps=len(self._gaps),
            critical_gaps=sum(1 for g in self._gaps if g.severity == GapSeverity.CRITICAL),
            total_options_evaluated=len(self._options),
            options_selected=len(self._roadmap),
            total_actions=len(self._actions),
            total_budget_eur=round(sum(r.budget_eur for r in self._roadmap), 2),
            total_effort_hours=round(sum(r.resource_allocation_hours for r in self._roadmap), 1),
            expected_quality_improvement=round(min(total_improvement, 100.0), 2),
            target_year=next_year,
            plan_provenance_hash=hashlib.sha256(plan_data.encode("utf-8")).hexdigest(),
        )

        outputs["total_actions"] = len(self._actions)
        outputs["planned"] = sum(1 for a in self._actions if a.status == ActionStatus.PLANNED)
        outputs["expected_quality_improvement"] = self._summary.expected_quality_improvement
        outputs["total_budget"] = self._summary.total_budget_eur
        outputs["total_effort_hours"] = self._summary.total_effort_hours
        outputs["target_year"] = next_year

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ActionPlanning: %d actions, improvement=%.1f points",
            len(self._actions), self._summary.expected_quality_improvement,
        )
        return PhaseResult(
            phase_name="action_planning", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._gaps = []
        self._options = []
        self._roadmap = []
        self._actions = []
        self._summary = None

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: ImprovementPlanningResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.reporting_year}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
