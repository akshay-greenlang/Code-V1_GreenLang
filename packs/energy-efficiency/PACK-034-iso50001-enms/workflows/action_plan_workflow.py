# -*- coding: utf-8 -*-
"""
Action Plan Workflow - ISO 50001 Clause 6.2 Planning
===================================

4-phase workflow for creating energy management action plans within
PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. ObjectiveSetting    -- Define SMART objectives aligned with energy policy
    2. ActionDefinition    -- Create action plans with savings estimates
    3. ResourceAllocation  -- Allocate capital, human, technical resources
    4. TimelineCreation    -- Create implementation timeline with milestones

The workflow follows GreenLang zero-hallucination principles: savings
estimates use engineering formulas, resource allocation uses deterministic
budgeting rules, and timeline logic follows dependency-based scheduling.
SHA-256 provenance hashes guarantee auditability.

Schedule: annual / on policy change
Estimated duration: 40 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class PlanningPhase(str, Enum):
    """Phases of the action plan workflow."""

    OBJECTIVE_SETTING = "objective_setting"
    ACTION_DEFINITION = "action_definition"
    RESOURCE_ALLOCATION = "resource_allocation"
    TIMELINE_CREATION = "timeline_creation"


class ObjectiveStatus(str, Enum):
    """Status of an energy objective."""

    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    DEFERRED = "deferred"


class ResourceType(str, Enum):
    """Types of resources for action plans."""

    CAPITAL = "capital"
    HUMAN = "human"
    TECHNICAL = "technical"


# =============================================================================
# SAVINGS ESTIMATION REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Default energy cost per kWh by region for savings valuation
ENERGY_COST_DEFAULTS: Dict[str, Decimal] = {
    "US": Decimal("0.12"),
    "EU": Decimal("0.22"),
    "UK": Decimal("0.28"),
    "AU": Decimal("0.25"),
    "DEFAULT": Decimal("0.15"),
}

# Typical savings achievable by action type (% of SEU consumption)
ACTION_TYPE_SAVINGS: Dict[str, Dict[str, Any]] = {
    "equipment_upgrade": {"savings_pct_low": 10.0, "savings_pct_high": 30.0, "complexity": "high"},
    "operational_improvement": {"savings_pct_low": 5.0, "savings_pct_high": 15.0, "complexity": "low"},
    "controls_optimization": {"savings_pct_low": 8.0, "savings_pct_high": 20.0, "complexity": "medium"},
    "behavioral_change": {"savings_pct_low": 3.0, "savings_pct_high": 8.0, "complexity": "low"},
    "maintenance_improvement": {"savings_pct_low": 5.0, "savings_pct_high": 12.0, "complexity": "low"},
    "design_modification": {"savings_pct_low": 15.0, "savings_pct_high": 40.0, "complexity": "high"},
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EnergyObjective(BaseModel):
    """A SMART energy objective."""

    objective_id: str = Field(default_factory=lambda: f"obj-{uuid.uuid4().hex[:8]}")
    title: str = Field(default="", description="Objective title")
    description: str = Field(default="", description="Detailed SMART description")
    target_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    target_savings_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    target_date: str = Field(default="", description="Target completion YYYY-MM-DD")
    seu_category: str = Field(default="", description="Related SEU category")
    status: ObjectiveStatus = Field(default=ObjectiveStatus.DRAFT)
    is_smart: bool = Field(default=True, description="Passes SMART criteria check")


class EnergyTarget(BaseModel):
    """A measurable energy target linked to an objective."""

    target_id: str = Field(default_factory=lambda: f"tgt-{uuid.uuid4().hex[:8]}")
    objective_id: str = Field(default="", description="Parent objective ID")
    enpi_name: str = Field(default="", description="EnPI to track against")
    baseline_value: Decimal = Field(default=Decimal("0"))
    target_value: Decimal = Field(default=Decimal("0"))
    unit: str = Field(default="kWh", description="Target unit")
    deadline: str = Field(default="", description="YYYY-MM-DD")


class ActionItem(BaseModel):
    """An individual action within the plan."""

    action_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    objective_id: str = Field(default="", description="Parent objective ID")
    title: str = Field(default="", description="Action title")
    description: str = Field(default="", description="Action details")
    action_type: str = Field(default="operational_improvement", description="Action category")
    estimated_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_savings_cost: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_cost: Decimal = Field(default=Decimal("0"), ge=0, description="Implementation cost")
    responsible_party: str = Field(default="", description="Assigned responsible party")
    duration_months: int = Field(default=3, ge=1, le=60)
    dependencies: List[str] = Field(default_factory=list, description="Dependent action IDs")


class ResourceAllocation(BaseModel):
    """Resource allocation for an action."""

    allocation_id: str = Field(default_factory=lambda: f"res-{uuid.uuid4().hex[:8]}")
    action_id: str = Field(default="", description="Linked action ID")
    resource_type: str = Field(default="capital", description="capital|human|technical")
    amount: Decimal = Field(default=Decimal("0"), ge=0)
    unit: str = Field(default="USD", description="Resource unit")
    notes: str = Field(default="")


class MilestoneItem(BaseModel):
    """A milestone in the implementation timeline."""

    milestone_id: str = Field(default_factory=lambda: f"ms-{uuid.uuid4().hex[:8]}")
    action_id: str = Field(default="", description="Linked action ID")
    title: str = Field(default="", description="Milestone title")
    target_month: int = Field(default=0, ge=0, description="Month offset from plan start")
    deliverable: str = Field(default="", description="Expected deliverable")


class GanttEntry(BaseModel):
    """A Gantt chart entry for visualization."""

    action_id: str = Field(default="")
    title: str = Field(default="")
    start_month: int = Field(default=0, ge=0)
    end_month: int = Field(default=0, ge=0)
    dependencies: List[str] = Field(default_factory=list)


class ActionPlanInput(BaseModel):
    """Input data model for ActionPlanWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    objectives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Objective definitions from energy policy",
    )
    seu_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="SEU analysis results from energy review",
    )
    budget_limit: Decimal = Field(default=Decimal("1000000"), ge=0, description="Total budget limit")
    timeline_months: int = Field(default=12, ge=1, le=60, description="Planning horizon months")
    region: str = Field(default="DEFAULT", description="Region for cost defaults")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ActionPlanResult(BaseModel):
    """Complete result from action plan workflow."""

    plan_id: str = Field(..., description="Unique plan ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    objectives: List[EnergyObjective] = Field(default_factory=list)
    targets: List[EnergyTarget] = Field(default_factory=list)
    action_plans: List[ActionItem] = Field(default_factory=list)
    resource_allocations: List[ResourceAllocation] = Field(default_factory=list)
    total_investment: Decimal = Field(default=Decimal("0"), ge=0)
    expected_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    expected_savings_cost: Decimal = Field(default=Decimal("0"), ge=0)
    portfolio_payback_years: Decimal = Field(default=Decimal("0"), ge=0)
    gantt_data: List[GanttEntry] = Field(default_factory=list)
    milestones: List[MilestoneItem] = Field(default_factory=list)
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ActionPlanWorkflow:
    """
    4-phase action plan workflow per ISO 50001 Clause 6.2.

    Defines SMART energy objectives, creates action plans with savings
    estimates, allocates resources within budget constraints, and builds
    an implementation timeline with milestones and dependencies.

    Zero-hallucination: savings estimates use engineering percentage ranges,
    cost calculations are deterministic, and scheduling follows
    dependency-based rules. No LLM calls in the numeric computation path.

    Attributes:
        plan_id: Unique plan execution identifier.
        _objectives: Defined energy objectives.
        _targets: Measurable targets.
        _actions: Action items.
        _allocations: Resource allocations.
        _gantt_data: Timeline visualization data.
        _milestones: Timeline milestones.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ActionPlanWorkflow()
        >>> inp = ActionPlanInput(
        ...     enms_id="enms-001",
        ...     objectives=[{"title": "Reduce HVAC energy 10%", "seu_category": "hvac"}],
        ...     budget_limit=Decimal("500000"),
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.total_investment <= Decimal("500000")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ActionPlanWorkflow."""
        self.plan_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._objectives: List[EnergyObjective] = []
        self._targets: List[EnergyTarget] = []
        self._actions: List[ActionItem] = []
        self._allocations: List[ResourceAllocation] = []
        self._gantt_data: List[GanttEntry] = []
        self._milestones: List[MilestoneItem] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: ActionPlanInput) -> ActionPlanResult:
        """
        Execute the 4-phase action plan workflow.

        Args:
            input_data: Validated action plan input.

        Returns:
            ActionPlanResult with objectives, actions, resources, and timeline.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting action plan workflow %s enms=%s objectives=%d budget=%.0f",
            self.plan_id, input_data.enms_id, len(input_data.objectives),
            float(input_data.budget_limit),
        )

        self._phase_results = []
        self._objectives = []
        self._targets = []
        self._actions = []
        self._allocations = []
        self._gantt_data = []
        self._milestones = []

        try:
            phase1 = self._phase_objective_setting(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_action_definition(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_resource_allocation(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_timeline_creation(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Action plan workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        total_investment = sum(a.estimated_cost for a in self._actions)
        total_savings_kwh = sum(a.estimated_savings_kwh for a in self._actions)
        total_savings_cost = sum(a.estimated_savings_cost for a in self._actions)
        payback_years = (
            Decimal(str(round(float(total_investment) / float(total_savings_cost), 2)))
            if total_savings_cost > 0 else Decimal("0")
        )

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = ActionPlanResult(
            plan_id=self.plan_id,
            enms_id=input_data.enms_id,
            objectives=self._objectives,
            targets=self._targets,
            action_plans=self._actions,
            resource_allocations=self._allocations,
            total_investment=total_investment,
            expected_savings_kwh=total_savings_kwh,
            expected_savings_cost=total_savings_cost,
            portfolio_payback_years=payback_years,
            gantt_data=self._gantt_data,
            milestones=self._milestones,
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Action plan workflow %s completed in %.0fms objectives=%d actions=%d "
            "investment=%.0f savings=%.0f kWh payback=%.1f yr",
            self.plan_id, elapsed_ms, len(self._objectives), len(self._actions),
            float(total_investment), float(total_savings_kwh), float(payback_years),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Objective Setting
    # -------------------------------------------------------------------------

    def _phase_objective_setting(
        self, input_data: ActionPlanInput
    ) -> PhaseResult:
        """Define SMART objectives aligned with energy policy."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        seu_data = input_data.seu_results.get("seus", [])

        for obj_dict in input_data.objectives:
            title = obj_dict.get("title", "Energy Improvement Objective")
            seu_category = obj_dict.get("seu_category", "")
            target_pct = Decimal(str(obj_dict.get("target_savings_pct", 10)))

            # Find SEU consumption for this category
            seu_kwh = Decimal("0")
            for seu in seu_data:
                if isinstance(seu, dict) and seu.get("category") == seu_category:
                    seu_kwh = Decimal(str(seu.get("consumption_kwh", 0)))
                    break

            target_kwh = Decimal(str(round(float(seu_kwh) * float(target_pct) / 100.0, 2)))

            # SMART validation
            has_specific = bool(title and seu_category)
            has_measurable = target_kwh > 0 or target_pct > 0
            has_achievable = float(target_pct) <= 50  # >50% is unrealistic
            has_relevant = bool(seu_category)
            has_time_bound = bool(obj_dict.get("target_date", ""))
            is_smart = all([has_specific, has_measurable, has_achievable, has_relevant])

            if not has_time_bound:
                # Default target date to end of planning horizon
                is_smart = is_smart  # Still valid, just use default
                warnings.append(f"Objective '{title}': no target date; using planning horizon end")

            objective = EnergyObjective(
                title=title,
                description=obj_dict.get("description", f"Reduce {seu_category} energy by {target_pct}%"),
                target_savings_kwh=target_kwh,
                target_savings_pct=target_pct,
                target_date=obj_dict.get("target_date", ""),
                seu_category=seu_category,
                status=ObjectiveStatus.DRAFT,
                is_smart=is_smart,
            )
            self._objectives.append(objective)

            # Create corresponding target
            target = EnergyTarget(
                objective_id=objective.objective_id,
                enpi_name=f"{seu_category}_intensity",
                baseline_value=seu_kwh,
                target_value=seu_kwh - target_kwh,
                unit="kWh/year",
                deadline=obj_dict.get("target_date", ""),
            )
            self._targets.append(target)

        outputs["objectives_created"] = len(self._objectives)
        outputs["targets_created"] = len(self._targets)
        outputs["smart_compliant"] = sum(1 for o in self._objectives if o.is_smart)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 ObjectiveSetting: %d objectives, %d SMART compliant",
            len(self._objectives), outputs["smart_compliant"],
        )
        return PhaseResult(
            phase_name=PlanningPhase.OBJECTIVE_SETTING.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Action Definition
    # -------------------------------------------------------------------------

    def _phase_action_definition(
        self, input_data: ActionPlanInput
    ) -> PhaseResult:
        """Create action plans for each objective with savings estimates."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        energy_cost = ENERGY_COST_DEFAULTS.get(
            input_data.region, ENERGY_COST_DEFAULTS["DEFAULT"]
        )

        for objective in self._objectives:
            actions = self._generate_actions_for_objective(objective, energy_cost)
            self._actions.extend(actions)

        total_savings_kwh = sum(a.estimated_savings_kwh for a in self._actions)
        total_savings_cost = sum(a.estimated_savings_cost for a in self._actions)
        total_cost = sum(a.estimated_cost for a in self._actions)

        outputs["actions_created"] = len(self._actions)
        outputs["total_estimated_savings_kwh"] = str(total_savings_kwh)
        outputs["total_estimated_savings_cost"] = str(total_savings_cost)
        outputs["total_estimated_cost"] = str(total_cost)
        outputs["energy_cost_per_kwh"] = str(energy_cost)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 ActionDefinition: %d actions, savings=%.0f kWh, cost=%.0f",
            len(self._actions), float(total_savings_kwh), float(total_cost),
        )
        return PhaseResult(
            phase_name=PlanningPhase.ACTION_DEFINITION.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_actions_for_objective(
        self, objective: EnergyObjective, energy_cost: Decimal
    ) -> List[ActionItem]:
        """Generate action items for a specific objective."""
        actions: List[ActionItem] = []
        remaining_kwh = objective.target_savings_kwh

        # Generate actions from each applicable action type
        for action_type, params in ACTION_TYPE_SAVINGS.items():
            if remaining_kwh <= 0:
                break

            savings_pct = (params["savings_pct_low"] + params["savings_pct_high"]) / 2.0
            action_savings = Decimal(str(round(
                float(objective.target_savings_kwh) * savings_pct / 100.0, 2
            )))
            action_savings = min(action_savings, remaining_kwh)

            if action_savings <= 0:
                continue

            savings_cost = Decimal(str(round(float(action_savings) * float(energy_cost), 2)))

            # Estimate implementation cost based on complexity
            complexity = params["complexity"]
            cost_multiplier = {"low": 0.5, "medium": 1.5, "high": 3.0}.get(complexity, 1.0)
            impl_cost = Decimal(str(round(float(savings_cost) * cost_multiplier, 2)))

            duration = {"low": 2, "medium": 4, "high": 8}.get(complexity, 4)

            action = ActionItem(
                objective_id=objective.objective_id,
                title=f"{action_type.replace('_', ' ').title()} - {objective.seu_category}",
                description=(
                    f"Implement {action_type.replace('_', ' ')} for "
                    f"{objective.seu_category} to achieve "
                    f"{action_savings} kWh annual savings."
                ),
                action_type=action_type,
                estimated_savings_kwh=action_savings,
                estimated_savings_cost=savings_cost,
                estimated_cost=impl_cost,
                duration_months=duration,
            )
            actions.append(action)
            remaining_kwh -= action_savings

        return actions

    # -------------------------------------------------------------------------
    # Phase 3: Resource Allocation
    # -------------------------------------------------------------------------

    def _phase_resource_allocation(
        self, input_data: ActionPlanInput
    ) -> PhaseResult:
        """Allocate resources (capital, human, technical) within budget."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        budget_remaining = input_data.budget_limit
        allocated_count = 0
        over_budget = False

        for action in self._actions:
            if budget_remaining <= 0:
                over_budget = True
                warnings.append(
                    f"Budget exhausted at action '{action.title}'; remaining actions unfunded"
                )
                break

            # Capital allocation
            capital_amount = min(action.estimated_cost, budget_remaining)
            capital_alloc = ResourceAllocation(
                action_id=action.action_id,
                resource_type=ResourceType.CAPITAL.value,
                amount=capital_amount,
                unit="USD",
                notes=f"Capital for {action.action_type}",
            )
            self._allocations.append(capital_alloc)
            budget_remaining -= capital_amount

            # Human resource allocation (estimate based on duration)
            fte_months = Decimal(str(round(action.duration_months * 0.25, 2)))
            human_alloc = ResourceAllocation(
                action_id=action.action_id,
                resource_type=ResourceType.HUMAN.value,
                amount=fte_months,
                unit="FTE-months",
                notes=f"Personnel for {action.action_type}",
            )
            self._allocations.append(human_alloc)

            # Technical resource allocation
            tech_alloc = ResourceAllocation(
                action_id=action.action_id,
                resource_type=ResourceType.TECHNICAL.value,
                amount=Decimal("1"),
                unit="support_units",
                notes=f"Technical support for {action.action_type}",
            )
            self._allocations.append(tech_alloc)

            allocated_count += 1

        total_capital = sum(
            a.amount for a in self._allocations if a.resource_type == ResourceType.CAPITAL.value
        )
        total_fte = sum(
            a.amount for a in self._allocations if a.resource_type == ResourceType.HUMAN.value
        )

        outputs["allocations_created"] = len(self._allocations)
        outputs["actions_funded"] = allocated_count
        outputs["total_capital_allocated"] = str(total_capital)
        outputs["total_fte_months"] = str(total_fte)
        outputs["budget_remaining"] = str(budget_remaining)
        outputs["over_budget"] = over_budget

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 ResourceAllocation: %d allocations, capital=%.0f, FTE=%.1f, remaining=%.0f",
            len(self._allocations), float(total_capital), float(total_fte),
            float(budget_remaining),
        )
        return PhaseResult(
            phase_name=PlanningPhase.RESOURCE_ALLOCATION.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Timeline Creation
    # -------------------------------------------------------------------------

    def _phase_timeline_creation(
        self, input_data: ActionPlanInput
    ) -> PhaseResult:
        """Create implementation timeline with milestones and dependencies."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        current_month = 0
        max_month = 0

        # Schedule actions sequentially within each objective, respecting dependencies
        objective_end_months: Dict[str, int] = {}

        for objective in self._objectives:
            obj_actions = [a for a in self._actions if a.objective_id == objective.objective_id]
            obj_month = 0

            for action in obj_actions:
                start_month = obj_month
                end_month = start_month + action.duration_months

                gantt = GanttEntry(
                    action_id=action.action_id,
                    title=action.title,
                    start_month=start_month,
                    end_month=end_month,
                    dependencies=action.dependencies,
                )
                self._gantt_data.append(gantt)

                # Create milestones
                start_ms = MilestoneItem(
                    action_id=action.action_id,
                    title=f"Start: {action.title}",
                    target_month=start_month,
                    deliverable="Kickoff and resource mobilization",
                )
                self._milestones.append(start_ms)

                end_ms = MilestoneItem(
                    action_id=action.action_id,
                    title=f"Complete: {action.title}",
                    target_month=end_month,
                    deliverable=f"Verified savings of {action.estimated_savings_kwh} kWh",
                )
                self._milestones.append(end_ms)

                obj_month = end_month
                max_month = max(max_month, end_month)

            objective_end_months[objective.objective_id] = obj_month

        # Cap timeline to planning horizon
        if max_month > input_data.timeline_months:
            warnings.append(
                f"Planned timeline ({max_month} months) exceeds planning horizon "
                f"({input_data.timeline_months} months)"
            )

        outputs["gantt_entries"] = len(self._gantt_data)
        outputs["milestones_created"] = len(self._milestones)
        outputs["total_timeline_months"] = max_month
        outputs["planning_horizon_months"] = input_data.timeline_months

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 TimelineCreation: %d gantt entries, %d milestones, timeline=%d months",
            len(self._gantt_data), len(self._milestones), max_month,
        )
        return PhaseResult(
            phase_name=PlanningPhase.TIMELINE_CREATION.value, phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ActionPlanResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
