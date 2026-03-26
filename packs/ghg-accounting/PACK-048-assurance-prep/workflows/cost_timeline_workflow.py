# -*- coding: utf-8 -*-
"""
Cost & Timeline Workflow
====================================

5-phase workflow for GHG assurance cost estimation and timeline planning
covering engagement scoping, cost estimation, timeline planning, resource
allocation, and budget approval within PACK-048 GHG Assurance Prep Pack.

Phases:
    1. EngagementScoping           -- Define engagement scope parameters
                                      including assurance level, scope
                                      coverage, number of facilities,
                                      complexity factors, and geographic
                                      spread that drive cost and timeline.
    2. CostEstimation              -- Estimate assurance engagement costs
                                      by assurance level, complexity
                                      factors, scope coverage, and
                                      market rate benchmarks using
                                      Decimal arithmetic.
    3. TimelinePlanning            -- Plan engagement timeline with
                                      milestones covering planning,
                                      fieldwork, reporting, and closeout
                                      phases based on complexity and
                                      resource availability.
    4. ResourceAllocation          -- Allocate internal resources by role
                                      (FTE hours) for each engagement
                                      phase, including sustainability,
                                      data management, finance, and
                                      executive involvement.
    5. BudgetApproval              -- Produce a budget approval package
                                      with cost breakdown, timeline
                                      summary, resource requirements,
                                      and ROI justification.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISAE 3410 (2012) - Engagement planning requirements
    ISQM 1 (2022) - Quality management resource requirements
    ESRS E1 (2024) - Assurance scope and resources
    CSRD (2022/2464) - Mandatory assurance cost considerations
    SEC Climate Disclosure Rules (2024) - Attestation cost implications

Schedule: Annually at budget cycle
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


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


class CostTimelinePhase(str, Enum):
    """Cost and timeline workflow phases."""

    ENGAGEMENT_SCOPING = "engagement_scoping"
    COST_ESTIMATION = "cost_estimation"
    TIMELINE_PLANNING = "timeline_planning"
    RESOURCE_ALLOCATION = "resource_allocation"
    BUDGET_APPROVAL = "budget_approval"


class AssuranceLevel(str, Enum):
    """Assurance engagement level."""

    LIMITED = "limited"
    REASONABLE = "reasonable"


class ComplexityLevel(str, Enum):
    """Engagement complexity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CostCategory(str, Enum):
    """Cost breakdown category."""

    EXTERNAL_ASSURANCE_FEE = "external_assurance_fee"
    INTERNAL_PERSONNEL = "internal_personnel"
    DATA_PREPARATION = "data_preparation"
    SYSTEMS_TOOLING = "systems_tooling"
    TRAVEL_LOGISTICS = "travel_logistics"
    CONTINGENCY = "contingency"


class EngagementMilestone(str, Enum):
    """Engagement timeline milestone."""

    PLANNING = "planning"
    DATA_PREPARATION = "data_preparation"
    FIELDWORK = "fieldwork"
    REPORTING = "reporting"
    REVIEW_APPROVAL = "review_approval"
    CLOSEOUT = "closeout"


class InternalRole(str, Enum):
    """Internal resource role."""

    SUSTAINABILITY_DIRECTOR = "sustainability_director"
    GHG_SPECIALIST = "ghg_specialist"
    DATA_MANAGER = "data_manager"
    FINANCE_CONTROLLER = "finance_controller"
    IT_SUPPORT = "it_support"
    EXECUTIVE_SPONSOR = "executive_sponsor"


# =============================================================================
# COST REFERENCE DATA (Zero-Hallucination)
# =============================================================================

BASE_ASSURANCE_FEES_USD: Dict[str, Dict[str, int]] = {
    "limited": {"low": 25000, "medium": 50000, "high": 90000, "very_high": 150000},
    "reasonable": {"low": 50000, "medium": 100000, "high": 180000, "very_high": 300000},
}

SCOPE_MULTIPLIERS: Dict[str, Decimal] = {
    "scope_1_only": Decimal("0.70"),
    "scope_1_2": Decimal("1.00"),
    "scope_1_2_3_limited": Decimal("1.40"),
    "scope_1_2_3_full": Decimal("1.80"),
}

FACILITY_COUNT_MULTIPLIERS: Dict[str, Tuple[int, int, Decimal]] = {
    "single": (1, 1, Decimal("1.00")),
    "small_multi": (2, 5, Decimal("1.15")),
    "medium_multi": (6, 20, Decimal("1.35")),
    "large_multi": (21, 50, Decimal("1.55")),
    "very_large": (51, 999, Decimal("1.80")),
}

MILESTONE_DURATIONS_WEEKS: Dict[str, Dict[str, int]] = {
    "limited": {
        "planning": 2, "data_preparation": 3, "fieldwork": 3,
        "reporting": 2, "review_approval": 1, "closeout": 1,
    },
    "reasonable": {
        "planning": 3, "data_preparation": 4, "fieldwork": 5,
        "reporting": 3, "review_approval": 2, "closeout": 1,
    },
}

INTERNAL_HOURS_PER_ROLE: Dict[str, Dict[str, int]] = {
    "limited": {
        "sustainability_director": 60,
        "ghg_specialist": 120,
        "data_manager": 80,
        "finance_controller": 30,
        "it_support": 20,
        "executive_sponsor": 10,
    },
    "reasonable": {
        "sustainability_director": 100,
        "ghg_specialist": 200,
        "data_manager": 140,
        "finance_controller": 50,
        "it_support": 40,
        "executive_sponsor": 20,
    },
}

ROLE_HOURLY_RATES_USD: Dict[str, int] = {
    "sustainability_director": 150,
    "ghg_specialist": 120,
    "data_manager": 100,
    "finance_controller": 130,
    "it_support": 90,
    "executive_sponsor": 200,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ScopeParameters(BaseModel):
    """Engagement scope parameters for cost estimation."""

    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    scope_coverage: str = Field(default="scope_1_2")
    facility_count: int = Field(default=1, ge=1)
    complexity: ComplexityLevel = Field(default=ComplexityLevel.MEDIUM)
    geographic_regions: int = Field(default=1, ge=1)
    first_year_engagement: bool = Field(default=True)
    provenance_hash: str = Field(default="")


class CostLineItem(BaseModel):
    """A single cost line item."""

    category: CostCategory = Field(...)
    description: str = Field(default="")
    amount_usd: str = Field(default="0.00")
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")


class MilestoneRecord(BaseModel):
    """A milestone in the engagement timeline."""

    milestone: EngagementMilestone = Field(...)
    name: str = Field(default="")
    duration_weeks: int = Field(default=0, ge=0)
    start_week: int = Field(default=0, ge=0)
    end_week: int = Field(default=0, ge=0)
    deliverables: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ResourceAllocationRecord(BaseModel):
    """Resource allocation for an internal role."""

    role: InternalRole = Field(...)
    role_name: str = Field(default="")
    hours_allocated: int = Field(default=0, ge=0)
    hourly_rate_usd: int = Field(default=0, ge=0)
    total_cost_usd: str = Field(default="0.00")
    provenance_hash: str = Field(default="")


class BudgetPackage(BaseModel):
    """Budget approval package."""

    total_external_cost_usd: str = Field(default="0.00")
    total_internal_cost_usd: str = Field(default="0.00")
    total_cost_usd: str = Field(default="0.00")
    total_duration_weeks: int = Field(default=0)
    total_internal_hours: int = Field(default=0)
    roi_justification: str = Field(default="")
    approval_status: str = Field(default="pending")
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class CostTimelineInput(BaseModel):
    """Input data model for CostTimelineWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    scope_coverage: str = Field(
        default="scope_1_2",
        description="Scope coverage: scope_1_only, scope_1_2, scope_1_2_3_limited, scope_1_2_3_full",
    )
    facility_count: int = Field(default=1, ge=1)
    geographic_regions: int = Field(default=1, ge=1)
    complexity: ComplexityLevel = Field(default=ComplexityLevel.MEDIUM)
    first_year_engagement: bool = Field(default=True)
    reporting_period: str = Field(default="2025")
    revenue_usd_m: float = Field(default=0.0, ge=0.0)
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class CostTimelineResult(BaseModel):
    """Complete result from cost and timeline workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="cost_timeline")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    scope_parameters: Optional[ScopeParameters] = Field(default=None)
    cost_items: List[CostLineItem] = Field(default_factory=list)
    milestones: List[MilestoneRecord] = Field(default_factory=list)
    resource_allocations: List[ResourceAllocationRecord] = Field(default_factory=list)
    budget_package: Optional[BudgetPackage] = Field(default=None)
    total_cost_usd: str = Field(default="0.00")
    total_duration_weeks: int = Field(default=0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CostTimelineWorkflow:
    """
    5-phase workflow for assurance cost estimation and timeline planning.

    Defines engagement scope parameters, estimates costs by assurance level
    and complexity, plans the timeline with milestones, allocates internal
    resources, and produces a budget approval package.

    Zero-hallucination: all costs use Decimal arithmetic with ROUND_HALF_UP
    and deterministic reference tables; no LLM calls in cost calculations;
    SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _scope_params: Engagement scope parameters.
        _cost_items: Cost line items.
        _milestones: Timeline milestones.
        _resources: Resource allocations.
        _budget: Budget approval package.

    Example:
        >>> wf = CostTimelineWorkflow()
        >>> inp = CostTimelineInput(
        ...     organization_id="org-001",
        ...     assurance_level=AssuranceLevel.LIMITED,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[CostTimelinePhase] = [
        CostTimelinePhase.ENGAGEMENT_SCOPING,
        CostTimelinePhase.COST_ESTIMATION,
        CostTimelinePhase.TIMELINE_PLANNING,
        CostTimelinePhase.RESOURCE_ALLOCATION,
        CostTimelinePhase.BUDGET_APPROVAL,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CostTimelineWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._scope_params: Optional[ScopeParameters] = None
        self._cost_items: List[CostLineItem] = []
        self._milestones: List[MilestoneRecord] = []
        self._resources: List[ResourceAllocationRecord] = []
        self._budget: Optional[BudgetPackage] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: CostTimelineInput,
    ) -> CostTimelineResult:
        """
        Execute the 5-phase cost and timeline workflow.

        Args:
            input_data: Organisation and engagement parameters.

        Returns:
            CostTimelineResult with cost breakdown and timeline.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting cost timeline %s org=%s level=%s",
            self.workflow_id, input_data.organization_id,
            input_data.assurance_level.value,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_engagement_scoping,
            self._phase_2_cost_estimation,
            self._phase_3_timeline_planning,
            self._phase_4_resource_allocation,
            self._phase_5_budget_approval,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Cost timeline failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        total_cost = "0.00"
        total_weeks = 0
        if self._budget:
            total_cost = self._budget.total_cost_usd
            total_weeks = self._budget.total_duration_weeks

        result = CostTimelineResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            scope_parameters=self._scope_params,
            cost_items=self._cost_items,
            milestones=self._milestones,
            resource_allocations=self._resources,
            budget_package=self._budget,
            total_cost_usd=total_cost,
            total_duration_weeks=total_weeks,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Cost timeline %s completed in %.2fs status=%s cost=%s weeks=%d",
            self.workflow_id, elapsed, overall_status.value,
            total_cost, total_weeks,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Engagement Scoping
    # -------------------------------------------------------------------------

    async def _phase_1_engagement_scoping(
        self, input_data: CostTimelineInput,
    ) -> PhaseResult:
        """Define engagement scope parameters."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        scope_data = {
            "level": input_data.assurance_level.value,
            "coverage": input_data.scope_coverage,
            "facilities": input_data.facility_count,
            "complexity": input_data.complexity.value,
        }
        self._scope_params = ScopeParameters(
            assurance_level=input_data.assurance_level,
            scope_coverage=input_data.scope_coverage,
            facility_count=input_data.facility_count,
            complexity=input_data.complexity,
            geographic_regions=input_data.geographic_regions,
            first_year_engagement=input_data.first_year_engagement,
            provenance_hash=_compute_hash(scope_data),
        )

        if input_data.first_year_engagement:
            warnings.append(
                "First-year engagements typically incur 20-30% higher costs "
                "due to initial setup and learning curve"
            )

        outputs["assurance_level"] = input_data.assurance_level.value
        outputs["scope_coverage"] = input_data.scope_coverage
        outputs["facility_count"] = input_data.facility_count
        outputs["complexity"] = input_data.complexity.value
        outputs["geographic_regions"] = input_data.geographic_regions
        outputs["first_year"] = input_data.first_year_engagement

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 EngagementScoping: level=%s coverage=%s facilities=%d complexity=%s",
            input_data.assurance_level.value, input_data.scope_coverage,
            input_data.facility_count, input_data.complexity.value,
        )
        return PhaseResult(
            phase_name="engagement_scoping", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Cost Estimation
    # -------------------------------------------------------------------------

    async def _phase_2_cost_estimation(
        self, input_data: CostTimelineInput,
    ) -> PhaseResult:
        """Estimate assurance engagement costs."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        level = input_data.assurance_level.value
        complexity = input_data.complexity.value

        # Base fee
        base_fee = Decimal(str(
            BASE_ASSURANCE_FEES_USD.get(level, {}).get(complexity, 50000),
        ))

        # Scope multiplier
        scope_mult = SCOPE_MULTIPLIERS.get(
            input_data.scope_coverage, Decimal("1.00"),
        )

        # Facility multiplier
        facility_mult = Decimal("1.00")
        for _band, (lower, upper, mult) in FACILITY_COUNT_MULTIPLIERS.items():
            if lower <= input_data.facility_count <= upper:
                facility_mult = mult
                break

        # Geographic multiplier
        geo_mult = Decimal("1.00") + Decimal("0.05") * Decimal(
            str(max(input_data.geographic_regions - 1, 0)),
        )

        # First-year premium
        first_year_mult = Decimal("1.25") if input_data.first_year_engagement else Decimal("1.00")

        # Calculate external fee
        external_fee = (
            base_fee * scope_mult * facility_mult * geo_mult * first_year_mult
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        self._cost_items = []

        # External assurance fee
        self._cost_items.append(CostLineItem(
            category=CostCategory.EXTERNAL_ASSURANCE_FEE,
            description="External assurance provider fee",
            amount_usd=str(external_fee),
            notes=f"Base: {base_fee}, Scope: {scope_mult}, Facility: {facility_mult}",
            provenance_hash=_compute_hash({"external_fee": str(external_fee)}),
        ))

        # Data preparation (15% of external)
        data_prep = (external_fee * Decimal("0.15")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        self._cost_items.append(CostLineItem(
            category=CostCategory.DATA_PREPARATION,
            description="Data preparation and evidence collection",
            amount_usd=str(data_prep),
            provenance_hash=_compute_hash({"data_prep": str(data_prep)}),
        ))

        # Systems / tooling (5% of external)
        systems = (external_fee * Decimal("0.05")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        self._cost_items.append(CostLineItem(
            category=CostCategory.SYSTEMS_TOOLING,
            description="Systems and tooling costs",
            amount_usd=str(systems),
            provenance_hash=_compute_hash({"systems": str(systems)}),
        ))

        # Travel (based on regions)
        travel = (
            Decimal(str(input_data.geographic_regions)) * Decimal("5000")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        self._cost_items.append(CostLineItem(
            category=CostCategory.TRAVEL_LOGISTICS,
            description="Travel and logistics",
            amount_usd=str(travel),
            provenance_hash=_compute_hash({"travel": str(travel)}),
        ))

        # Contingency (10% of total)
        subtotal = external_fee + data_prep + systems + travel
        contingency = (subtotal * Decimal("0.10")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        self._cost_items.append(CostLineItem(
            category=CostCategory.CONTINGENCY,
            description="Contingency (10%)",
            amount_usd=str(contingency),
            provenance_hash=_compute_hash({"contingency": str(contingency)}),
        ))

        total_external = (subtotal + contingency).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

        outputs["external_assurance_fee"] = str(external_fee)
        outputs["data_preparation"] = str(data_prep)
        outputs["systems_tooling"] = str(systems)
        outputs["travel"] = str(travel)
        outputs["contingency"] = str(contingency)
        outputs["total_external_cost"] = str(total_external)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 CostEstimation: total_external=%s", str(total_external),
        )
        return PhaseResult(
            phase_name="cost_estimation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Timeline Planning
    # -------------------------------------------------------------------------

    async def _phase_3_timeline_planning(
        self, input_data: CostTimelineInput,
    ) -> PhaseResult:
        """Plan engagement timeline with milestones."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        level = input_data.assurance_level.value
        durations = MILESTONE_DURATIONS_WEEKS.get(level, MILESTONE_DURATIONS_WEEKS["limited"])

        # Complexity adjustment
        complexity_add: Dict[str, int] = {
            "low": -1, "medium": 0, "high": 1, "very_high": 2,
        }
        adj = complexity_add.get(input_data.complexity.value, 0)

        milestone_defs = [
            (EngagementMilestone.PLANNING, "Planning & Kickoff",
             ["Engagement letter", "Planning memo", "Data request list"]),
            (EngagementMilestone.DATA_PREPARATION, "Data Preparation",
             ["Evidence package", "Data room populated", "Internal review complete"]),
            (EngagementMilestone.FIELDWORK, "Fieldwork",
             ["Site visits completed", "Testing completed", "Queries resolved"]),
            (EngagementMilestone.REPORTING, "Reporting",
             ["Draft report", "Management letter", "Finding responses"]),
            (EngagementMilestone.REVIEW_APPROVAL, "Review & Approval",
             ["QC review", "Management sign-off", "Final report"]),
            (EngagementMilestone.CLOSEOUT, "Closeout",
             ["Lessons learned", "Archive", "Fee finalisation"]),
        ]

        self._milestones = []
        current_week = 1
        for ms_enum, ms_name, deliverables in milestone_defs:
            base_dur = durations.get(ms_enum.value, 2)
            duration = max(1, base_dur + adj)

            ms_data = {"name": ms_name, "start": current_week, "dur": duration}
            milestone = MilestoneRecord(
                milestone=ms_enum,
                name=ms_name,
                duration_weeks=duration,
                start_week=current_week,
                end_week=current_week + duration - 1,
                deliverables=deliverables,
                provenance_hash=_compute_hash(ms_data),
            )
            self._milestones.append(milestone)
            current_week += duration

        total_weeks = sum(m.duration_weeks for m in self._milestones)

        outputs["total_duration_weeks"] = total_weeks
        outputs["milestones"] = len(self._milestones)
        outputs["milestone_summary"] = {
            m.name: m.duration_weeks for m in self._milestones
        }

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 TimelinePlanning: %d weeks, %d milestones",
            total_weeks, len(self._milestones),
        )
        return PhaseResult(
            phase_name="timeline_planning", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Resource Allocation
    # -------------------------------------------------------------------------

    async def _phase_4_resource_allocation(
        self, input_data: CostTimelineInput,
    ) -> PhaseResult:
        """Allocate internal resources by role."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        level = input_data.assurance_level.value
        hours_map = INTERNAL_HOURS_PER_ROLE.get(level, INTERNAL_HOURS_PER_ROLE["limited"])

        # Complexity adjustment
        complexity_mult: Dict[str, Decimal] = {
            "low": Decimal("0.80"),
            "medium": Decimal("1.00"),
            "high": Decimal("1.30"),
            "very_high": Decimal("1.60"),
        }
        mult = complexity_mult.get(input_data.complexity.value, Decimal("1.00"))

        self._resources = []
        total_hours = 0
        total_internal_cost = Decimal("0.00")

        for role_str, base_hours in hours_map.items():
            try:
                role = InternalRole(role_str)
            except ValueError:
                continue

            adjusted_hours = int(
                (Decimal(str(base_hours)) * mult).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP,
                ),
            )
            rate = ROLE_HOURLY_RATES_USD.get(role_str, 100)
            cost = (
                Decimal(str(adjusted_hours)) * Decimal(str(rate))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            res_data = {"role": role_str, "hours": adjusted_hours, "cost": str(cost)}
            self._resources.append(ResourceAllocationRecord(
                role=role,
                role_name=role_str.replace("_", " ").title(),
                hours_allocated=adjusted_hours,
                hourly_rate_usd=rate,
                total_cost_usd=str(cost),
                provenance_hash=_compute_hash(res_data),
            ))
            total_hours += adjusted_hours
            total_internal_cost += cost

        outputs["total_internal_hours"] = total_hours
        outputs["total_internal_cost_usd"] = str(total_internal_cost)
        outputs["roles_allocated"] = len(self._resources)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 ResourceAllocation: %d hours, cost=%s",
            total_hours, str(total_internal_cost),
        )
        return PhaseResult(
            phase_name="resource_allocation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Budget Approval
    # -------------------------------------------------------------------------

    async def _phase_5_budget_approval(
        self, input_data: CostTimelineInput,
    ) -> PhaseResult:
        """Produce budget approval package."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_external = sum(
            Decimal(c.amount_usd) for c in self._cost_items
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        total_internal = sum(
            Decimal(r.total_cost_usd) for r in self._resources
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        total_cost = (total_external + total_internal).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

        total_hours = sum(r.hours_allocated for r in self._resources)
        total_weeks = sum(m.duration_weeks for m in self._milestones)

        # ROI justification
        roi_text = (
            f"GHG assurance investment of {total_cost} USD enables: "
            f"(1) regulatory compliance with CSRD/SEC/ISSB mandates, "
            f"(2) stakeholder confidence in reported emissions, "
            f"(3) reduced risk of regulatory penalties, "
            f"(4) enhanced ESG ratings and investor relations. "
        )
        if input_data.revenue_usd_m > 0:
            cost_as_pct = (
                total_cost / Decimal(str(input_data.revenue_usd_m * 1_000_000))
                * Decimal("100")
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            roi_text += f"Cost represents {cost_as_pct}% of annual revenue."

        budget_data = {
            "external": str(total_external), "internal": str(total_internal),
            "total": str(total_cost), "weeks": total_weeks,
        }
        self._budget = BudgetPackage(
            total_external_cost_usd=str(total_external),
            total_internal_cost_usd=str(total_internal),
            total_cost_usd=str(total_cost),
            total_duration_weeks=total_weeks,
            total_internal_hours=total_hours,
            roi_justification=roi_text,
            approval_status="pending",
            provenance_hash=_compute_hash(budget_data),
        )

        outputs["total_external_cost_usd"] = str(total_external)
        outputs["total_internal_cost_usd"] = str(total_internal)
        outputs["total_cost_usd"] = str(total_cost)
        outputs["total_duration_weeks"] = total_weeks
        outputs["total_internal_hours"] = total_hours

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 BudgetApproval: total=%s weeks=%d hours=%d",
            str(total_cost), total_weeks, total_hours,
        )
        return PhaseResult(
            phase_name="budget_approval", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: CostTimelineInput,
        phase_number: int,
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
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._scope_params = None
        self._cost_items = []
        self._milestones = []
        self._resources = []
        self._budget = None

    def _compute_provenance(self, result: CostTimelineResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.total_cost_usd}|{result.total_duration_weeks}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
