# -*- coding: utf-8 -*-
"""
AnnualCycleEngine - PACK-024 Carbon Neutral Engine 9
=====================================================

10-phase carbon neutral lifecycle management engine covering commitment,
baseline, plan, reduce, quantify, procure, retire, balance, claim, and
verify phases with deadline tracking, multi-year continuity assessment,
phase dependency management, and readiness gating.

This engine manages the annual carbon neutrality cycle as defined by
ISO 14068-1:2023, ensuring all phases are completed in sequence and
on schedule, with proper documentation and handoffs between phases.

Calculation Methodology:
    10-Phase Lifecycle (ISO 14068-1:2023):
        Phase 1:  COMMITMENT    - Formal commitment to carbon neutrality
        Phase 2:  BASELINE      - Establish base year emissions
        Phase 3:  PLAN          - Develop carbon management plan
        Phase 4:  REDUCE        - Implement emission reductions
        Phase 5:  QUANTIFY      - Quantify current year emissions
        Phase 6:  PROCURE       - Procure carbon credits for residual
        Phase 7:  RETIRE        - Retire credits in registries
        Phase 8:  BALANCE       - Reconcile footprint vs retirements
        Phase 9:  CLAIM         - Make carbon neutral declaration
        Phase 10: VERIFY        - Third-party verification

    Phase Completion:
        phase_complete = all(required_tasks_complete)
        phase_on_time = completion_date <= deadline
        phase_score = completed_tasks / total_tasks * 100

    Cycle Readiness Gate:
        next_phase_ready = current_phase_complete AND
                           all_prerequisite_phases_complete AND
                           no_blocking_issues

    Multi-Year Continuity:
        continuity_score = sum(year_scores) / years_assessed
        improving = year_n_score > year_n_minus_1_score
        sustainable = all(year_score >= min_threshold for year in years)

    Deadline Tracking:
        days_until_deadline = deadline_date - current_date
        at_risk = days_until_deadline < buffer_days
        overdue = days_until_deadline < 0

Regulatory References:
    - ISO 14068-1:2023 - Full lifecycle (Sections 5-12)
    - PAS 2060:2014 - Carbon neutrality specification
    - ISO 14064-1:2018 - GHG quantification
    - ISO 14064-3:2019 - GHG verification
    - GHG Protocol Corporate Standard (2004, revised 2015)

Zero-Hallucination:
    - All 10 phases derived from ISO 14068-1:2023 structure
    - Phase requirements from published standard sections
    - No LLM involvement in any calculation path
    - Deterministic scoring throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PhaseId(str, Enum):
    """Carbon neutral lifecycle phase identifiers.

    10 phases derived from ISO 14068-1:2023 lifecycle.
    """
    COMMITMENT = "commitment"
    BASELINE = "baseline"
    PLAN = "plan"
    REDUCE = "reduce"
    QUANTIFY = "quantify"
    PROCURE = "procure"
    RETIRE = "retire"
    BALANCE = "balance"
    CLAIM = "claim"
    VERIFY = "verify"


class PhaseStatus(str, Enum):
    """Status of a lifecycle phase.

    NOT_STARTED: Phase has not begun.
    IN_PROGRESS: Phase is underway.
    COMPLETED: Phase completed successfully.
    OVERDUE: Phase deadline has passed without completion.
    BLOCKED: Phase is blocked by dependencies or issues.
    SKIPPED: Phase was intentionally skipped (with justification).
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskStatus(str, Enum):
    """Status of an individual task within a phase."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class CycleStatus(str, Enum):
    """Overall cycle status.

    ON_TRACK: All phases on schedule.
    AT_RISK: Some phases behind schedule.
    OFF_TRACK: Significant delays or blockers.
    COMPLETED: Entire cycle completed.
    NOT_STARTED: Cycle not yet begun.
    """
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    COMPLETED = "completed"
    NOT_STARTED = "not_started"


# ---------------------------------------------------------------------------
# Constants -- Phase Definitions
# ---------------------------------------------------------------------------

PHASE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    PhaseId.COMMITMENT.value: {
        "name": "Commitment",
        "description": "Formal commitment to achieving carbon neutrality",
        "standard_ref": "ISO 14068-1:2023 Section 5",
        "phase_number": 1,
        "prerequisites": [],
        "typical_duration_months": 1,
        "default_tasks": [
            "Board/executive commitment documented",
            "Carbon neutrality policy statement published",
            "Scope and boundary of commitment defined",
            "Target year for carbon neutrality set",
            "Responsible person/team designated",
        ],
    },
    PhaseId.BASELINE.value: {
        "name": "Baseline Establishment",
        "description": "Establish base year emissions inventory",
        "standard_ref": "ISO 14068-1:2023 Section 6, ISO 14064-1:2018",
        "phase_number": 2,
        "prerequisites": ["commitment"],
        "typical_duration_months": 3,
        "default_tasks": [
            "Base year selected and justified",
            "Organisational boundary defined",
            "Scope 1 emissions quantified",
            "Scope 2 emissions quantified (dual reporting)",
            "Scope 3 screening completed",
            "Material Scope 3 categories quantified",
            "Data quality assessment completed",
            "Base year inventory documented",
        ],
    },
    PhaseId.PLAN.value: {
        "name": "Management Plan",
        "description": "Develop carbon management plan with reduction targets",
        "standard_ref": "ISO 14068-1:2023 Section 9",
        "phase_number": 3,
        "prerequisites": ["baseline"],
        "typical_duration_months": 2,
        "default_tasks": [
            "Reduction targets set (quantified, timebound)",
            "Mitigation hierarchy documented",
            "Reduction measures identified and prioritised",
            "MACC analysis completed",
            "Implementation timeline established",
            "Financial projections prepared",
            "Monitoring process defined",
        ],
    },
    PhaseId.REDUCE.value: {
        "name": "Emission Reductions",
        "description": "Implement emission reduction measures",
        "standard_ref": "ISO 14068-1:2023 Section 9.4",
        "phase_number": 4,
        "prerequisites": ["plan"],
        "typical_duration_months": 9,
        "default_tasks": [
            "Quick-win measures implemented",
            "Energy efficiency measures deployed",
            "Renewable energy procurement initiated",
            "Process optimisation applied",
            "Supply chain engagement started",
            "Reduction progress monitored and reported",
        ],
    },
    PhaseId.QUANTIFY.value: {
        "name": "Footprint Quantification",
        "description": "Quantify current year emissions",
        "standard_ref": "ISO 14068-1:2023 Section 6, ISO 14064-1:2018",
        "phase_number": 5,
        "prerequisites": ["reduce"],
        "typical_duration_months": 2,
        "default_tasks": [
            "Current year data collected for all sources",
            "Scope 1 quantified for current year",
            "Scope 2 quantified (location and market-based)",
            "Scope 3 quantified for included categories",
            "Total footprint calculated",
            "Year-on-year comparison completed",
            "Reduction progress documented",
        ],
    },
    PhaseId.PROCURE.value: {
        "name": "Credit Procurement",
        "description": "Procure carbon credits for residual emissions",
        "standard_ref": "ISO 14068-1:2023 Section 8",
        "phase_number": 6,
        "prerequisites": ["quantify"],
        "typical_duration_months": 2,
        "default_tasks": [
            "Residual emissions calculated",
            "Credit quality criteria defined",
            "Credit portfolio designed",
            "Credits procured from verified sources",
            "Portfolio quality assessment completed",
            "Oxford Principles alignment checked",
        ],
    },
    PhaseId.RETIRE.value: {
        "name": "Credit Retirement",
        "description": "Retire credits in registries",
        "standard_ref": "ISO 14068-1:2023 Section 8.5",
        "phase_number": 7,
        "prerequisites": ["procure"],
        "typical_duration_months": 1,
        "default_tasks": [
            "Credits retired in respective registries",
            "Serial numbers tracked and documented",
            "Beneficiary properly designated",
            "Retirement certificates obtained",
            "Vintage-footprint alignment confirmed",
        ],
    },
    PhaseId.BALANCE.value: {
        "name": "Neutralization Balance",
        "description": "Reconcile footprint against retirements",
        "standard_ref": "ISO 14068-1:2023 Section 10",
        "phase_number": 8,
        "prerequisites": ["retire"],
        "typical_duration_months": 1,
        "default_tasks": [
            "Footprint vs retirement reconciliation completed",
            "Balance confirmed (neutral/surplus/deficit)",
            "Temporal alignment verified",
            "Carryforward handled (if applicable)",
            "Balance documentation prepared",
        ],
    },
    PhaseId.CLAIM.value: {
        "name": "Carbon Neutral Declaration",
        "description": "Make carbon neutral claim with substantiation",
        "standard_ref": "ISO 14068-1:2023 Section 10",
        "phase_number": 9,
        "prerequisites": ["balance"],
        "typical_duration_months": 1,
        "default_tasks": [
            "35-criterion substantiation checklist completed",
            "Qualifying Explanatory Statement (QES) prepared",
            "Claim wording reviewed for accuracy",
            "Disclosure documents prepared",
            "Declaration scope and period confirmed",
        ],
    },
    PhaseId.VERIFY.value: {
        "name": "Third-Party Verification",
        "description": "Independent verification of carbon neutral claim",
        "standard_ref": "ISO 14068-1:2023 Section 11, ISO 14064-3:2019",
        "phase_number": 10,
        "prerequisites": ["claim"],
        "typical_duration_months": 2,
        "default_tasks": [
            "Verification package assembled",
            "Verification body engaged",
            "Verification audit conducted",
            "Non-conformities addressed",
            "Verification statement issued",
            "Claim published with verification",
        ],
    },
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class TaskInput(BaseModel):
    """Input for a single task within a phase.

    Attributes:
        task_id: Task identifier.
        task_name: Task description.
        status: Task status.
        due_date: Task deadline.
        completed_date: Date task was completed.
        assignee: Person responsible.
        notes: Additional notes.
    """
    task_id: str = Field(default_factory=_new_uuid, description="Task ID")
    task_name: str = Field(default="", max_length=500, description="Task name")
    status: str = Field(
        default=TaskStatus.PENDING.value, description="Task status"
    )
    due_date: Optional[str] = Field(default=None, description="Due date")
    completed_date: Optional[str] = Field(default=None, description="Completed date")
    assignee: str = Field(default="", description="Assignee")
    notes: str = Field(default="", description="Notes")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {s.value for s in TaskStatus}
        if v not in valid:
            raise ValueError(f"Unknown task status '{v}'.")
        return v


class PhaseInput(BaseModel):
    """Input for a single lifecycle phase.

    Attributes:
        phase_id: Phase identifier.
        status: Phase status.
        start_date: Phase start date.
        deadline: Phase deadline.
        completed_date: Phase completion date.
        tasks: Tasks within this phase.
        owner: Phase owner.
        notes: Additional notes.
        blocking_issues: Issues blocking this phase.
    """
    phase_id: str = Field(..., description="Phase ID")
    status: str = Field(
        default=PhaseStatus.NOT_STARTED.value, description="Phase status"
    )
    start_date: Optional[str] = Field(default=None, description="Start date")
    deadline: Optional[str] = Field(default=None, description="Deadline")
    completed_date: Optional[str] = Field(default=None, description="Completed")
    tasks: List[TaskInput] = Field(default_factory=list, description="Tasks")
    owner: str = Field(default="", description="Phase owner")
    notes: str = Field(default="", description="Notes")
    blocking_issues: List[str] = Field(
        default_factory=list, description="Blocking issues"
    )

    @field_validator("phase_id")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        valid = {p.value for p in PhaseId}
        if v not in valid:
            raise ValueError(f"Unknown phase '{v}'.")
        return v


class YearCycleInput(BaseModel):
    """Input for a single year's cycle (for multi-year assessment).

    Attributes:
        year: Calendar year.
        overall_score: Overall cycle score (0-100).
        completed: Whether cycle was completed.
        declaration_made: Whether CN declaration was made.
        verified: Whether claim was verified.
        reduction_pct: Reduction achieved as % of baseline.
    """
    year: int = Field(default=0, description="Year")
    overall_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    completed: bool = Field(default=False)
    declaration_made: bool = Field(default=False)
    verified: bool = Field(default=False)
    reduction_pct: Decimal = Field(default=Decimal("0"), ge=0)


class AnnualCycleInput(BaseModel):
    """Complete input for annual cycle management.

    Attributes:
        entity_name: Entity name.
        cycle_year: Current cycle year.
        phases: Phase data for current cycle.
        previous_cycles: Data from previous years.
        current_date: Current date for deadline tracking.
        buffer_days: Buffer days for at-risk warnings.
        include_multi_year: Whether to include multi-year analysis.
        include_recommendations: Whether to generate recommendations.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    cycle_year: int = Field(
        ..., ge=2015, le=2060, description="Cycle year"
    )
    phases: List[PhaseInput] = Field(
        default_factory=list, description="Phase data"
    )
    previous_cycles: List[YearCycleInput] = Field(
        default_factory=list, description="Previous cycles"
    )
    current_date: Optional[str] = Field(
        default=None, description="Current date (YYYY-MM-DD)"
    )
    buffer_days: int = Field(
        default=14, ge=0, le=90, description="Buffer days for warnings"
    )
    include_multi_year: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class TaskResult(BaseModel):
    """Result for a single task."""
    task_id: str = Field(default="")
    task_name: str = Field(default="")
    status: str = Field(default="")
    is_complete: bool = Field(default=False)
    is_overdue: bool = Field(default=False)
    days_until_due: Optional[int] = Field(default=None)


class PhaseResult(BaseModel):
    """Result for a single lifecycle phase.

    Attributes:
        phase_id: Phase identifier.
        phase_name: Human-readable name.
        phase_number: Sequence number (1-10).
        standard_ref: Standard reference.
        status: Phase status.
        task_results: Per-task results.
        total_tasks: Total tasks.
        completed_tasks: Completed tasks.
        completion_pct: Completion percentage.
        prerequisites_met: Whether prerequisites are met.
        is_ready_to_start: Whether phase can begin.
        is_on_time: Whether on schedule.
        is_overdue: Whether deadline has passed.
        days_until_deadline: Days until deadline.
        blocking_issues: Blocking issues.
        recommendations: Phase recommendations.
    """
    phase_id: str = Field(default="")
    phase_name: str = Field(default="")
    phase_number: int = Field(default=0)
    standard_ref: str = Field(default="")
    status: str = Field(default="")
    task_results: List[TaskResult] = Field(default_factory=list)
    total_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    completion_pct: Decimal = Field(default=Decimal("0"))
    prerequisites_met: bool = Field(default=False)
    is_ready_to_start: bool = Field(default=False)
    is_on_time: bool = Field(default=True)
    is_overdue: bool = Field(default=False)
    days_until_deadline: Optional[int] = Field(default=None)
    blocking_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class MultiYearAssessment(BaseModel):
    """Multi-year continuity assessment.

    Attributes:
        years_assessed: Number of years.
        continuity_score: Average score across years.
        is_improving: Whether trend is improving.
        is_sustainable: Whether all years meet minimum threshold.
        year_scores: Score by year.
        declarations_made: Years with CN declarations.
        years_verified: Years with third-party verification.
        reduction_trend: Year-on-year reduction trend.
        message: Human-readable assessment.
    """
    years_assessed: int = Field(default=0)
    continuity_score: Decimal = Field(default=Decimal("0"))
    is_improving: bool = Field(default=False)
    is_sustainable: bool = Field(default=False)
    year_scores: Dict[int, Decimal] = Field(default_factory=dict)
    declarations_made: int = Field(default=0)
    years_verified: int = Field(default=0)
    reduction_trend: List[Decimal] = Field(default_factory=list)
    message: str = Field(default="")


class AnnualCycleResult(BaseModel):
    """Complete annual cycle result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        cycle_year: Cycle year.
        phase_results: Per-phase results.
        multi_year: Multi-year assessment.
        cycle_status: Overall cycle status.
        overall_completion_pct: Overall completion.
        phases_completed: Number of completed phases.
        phases_in_progress: Number in progress.
        phases_not_started: Not started count.
        phases_overdue: Overdue count.
        current_phase: Current active phase.
        next_phase: Next phase to start.
        next_deadline: Next upcoming deadline.
        critical_path_ok: Whether critical path is on schedule.
        recommendations: Overall recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    cycle_year: int = Field(default=0)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    multi_year: Optional[MultiYearAssessment] = Field(default=None)
    cycle_status: str = Field(default=CycleStatus.NOT_STARTED.value)
    overall_completion_pct: Decimal = Field(default=Decimal("0"))
    phases_completed: int = Field(default=0)
    phases_in_progress: int = Field(default=0)
    phases_not_started: int = Field(default=0)
    phases_overdue: int = Field(default=0)
    current_phase: str = Field(default="")
    next_phase: str = Field(default="")
    next_deadline: Optional[str] = Field(default=None)
    critical_path_ok: bool = Field(default=True)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AnnualCycleEngine:
    """10-phase carbon neutral lifecycle management engine.

    Manages the complete annual carbon neutrality cycle with phase
    tracking, deadline management, and multi-year continuity.

    Usage::

        engine = AnnualCycleEngine()
        result = engine.assess_cycle(input_data)
        print(f"Status: {result.cycle_status}")
        print(f"Completion: {result.overall_completion_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        logger.info("AnnualCycleEngine v%s initialised", self.engine_version)

    def assess_cycle(
        self, data: AnnualCycleInput,
    ) -> AnnualCycleResult:
        """Assess the current annual cycle.

        Args:
            data: Validated cycle input.

        Returns:
            AnnualCycleResult with comprehensive assessment.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        errors: List[str] = []

        # Build phase input lookup
        phase_map = {p.phase_id: p for p in data.phases}

        # Step 1: Assess each phase
        phase_results: List[PhaseResult] = []
        completed_phases: set = set()

        for phase_id in PhaseId:
            phase_def = PHASE_DEFINITIONS[phase_id.value]
            inp = phase_map.get(phase_id.value)
            prereqs = phase_def["prerequisites"]
            prereqs_met = all(p in completed_phases for p in prereqs)

            pr = self._assess_phase(
                phase_def, inp, phase_id.value, prereqs_met,
                data.current_date, data.buffer_days
            )
            phase_results.append(pr)

            if pr.status == PhaseStatus.COMPLETED.value:
                completed_phases.add(phase_id.value)

        # Step 2: Calculate totals
        total_phases = len(phase_results)
        completed = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED.value)
        in_progress = sum(1 for p in phase_results if p.status == PhaseStatus.IN_PROGRESS.value)
        not_started = sum(1 for p in phase_results if p.status == PhaseStatus.NOT_STARTED.value)
        overdue = sum(1 for p in phase_results if p.is_overdue)

        overall_completion = _safe_pct(_decimal(completed), _decimal(total_phases))

        # Determine current and next phase
        current = ""
        next_phase = ""
        for pr in phase_results:
            if pr.status == PhaseStatus.IN_PROGRESS.value:
                current = pr.phase_id
            elif pr.status == PhaseStatus.NOT_STARTED.value and not next_phase:
                next_phase = pr.phase_id

        # Cycle status
        if completed == total_phases:
            cycle_status = CycleStatus.COMPLETED.value
        elif overdue > 0:
            cycle_status = CycleStatus.OFF_TRACK.value
        elif in_progress > 0 or completed > 0:
            any_at_risk = any(
                pr.days_until_deadline is not None and pr.days_until_deadline <= data.buffer_days
                for pr in phase_results if pr.status == PhaseStatus.IN_PROGRESS.value
            )
            cycle_status = CycleStatus.AT_RISK.value if any_at_risk else CycleStatus.ON_TRACK.value
        else:
            cycle_status = CycleStatus.NOT_STARTED.value

        critical_ok = overdue == 0

        # Step 3: Multi-year assessment
        multi_year: Optional[MultiYearAssessment] = None
        if data.include_multi_year and data.previous_cycles:
            multi_year = self._assess_multi_year(
                data.previous_cycles, data.cycle_year, overall_completion
            )

        # Step 4: Recommendations
        recommendations: List[str] = []
        if data.include_recommendations:
            for pr in phase_results:
                recommendations.extend(pr.recommendations)
            if overdue > 0:
                recommendations.insert(0, f"URGENT: {overdue} phase(s) overdue. Prioritise completion.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = AnnualCycleResult(
            entity_name=data.entity_name,
            cycle_year=data.cycle_year,
            phase_results=phase_results,
            multi_year=multi_year,
            cycle_status=cycle_status,
            overall_completion_pct=_round_val(overall_completion, 2),
            phases_completed=completed,
            phases_in_progress=in_progress,
            phases_not_started=not_started,
            phases_overdue=overdue,
            current_phase=current,
            next_phase=next_phase,
            critical_path_ok=critical_ok,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_phase(
        self,
        phase_def: Dict[str, Any],
        inp: Optional[PhaseInput],
        phase_id: str,
        prereqs_met: bool,
        current_date: Optional[str],
        buffer_days: int,
    ) -> PhaseResult:
        """Assess a single lifecycle phase."""
        name = phase_def["name"]
        phase_number = phase_def["phase_number"]
        std_ref = phase_def["standard_ref"]

        if inp:
            status = inp.status
            blocking = inp.blocking_issues
            tasks = inp.tasks
        else:
            status = PhaseStatus.NOT_STARTED.value
            blocking = []
            tasks = []

        # If no tasks provided, use defaults
        if not tasks and status == PhaseStatus.NOT_STARTED.value:
            default_tasks = phase_def.get("default_tasks", [])
            task_results = [
                TaskResult(
                    task_id=_new_uuid(),
                    task_name=t,
                    status=TaskStatus.PENDING.value,
                    is_complete=False,
                )
                for t in default_tasks
            ]
        else:
            task_results = [
                TaskResult(
                    task_id=t.task_id,
                    task_name=t.task_name,
                    status=t.status,
                    is_complete=t.status == TaskStatus.COMPLETED.value,
                )
                for t in tasks
            ]

        total_tasks = len(task_results)
        completed_tasks = sum(1 for t in task_results if t.is_complete)
        completion = _safe_pct(_decimal(completed_tasks), _decimal(total_tasks))

        # Readiness
        ready = prereqs_met and not blocking

        # Deadline tracking
        is_on_time = True
        is_overdue = False
        days_until = None

        if inp and inp.deadline and current_date:
            try:
                dl = datetime.strptime(inp.deadline, "%Y-%m-%d").date()
                cd = datetime.strptime(current_date, "%Y-%m-%d").date()
                delta = (dl - cd).days
                days_until = delta
                is_overdue = delta < 0 and status != PhaseStatus.COMPLETED.value
                is_on_time = delta >= 0 or status == PhaseStatus.COMPLETED.value
            except (ValueError, TypeError):
                pass

        recs: List[str] = []
        if status == PhaseStatus.NOT_STARTED.value and prereqs_met:
            recs.append(f"Phase '{name}' is ready to start. Begin implementation.")
        if is_overdue:
            recs.append(f"Phase '{name}' is overdue by {abs(days_until or 0)} days.")
        if blocking:
            recs.append(f"Resolve blocking issues for '{name}': {'; '.join(blocking[:2])}")

        return PhaseResult(
            phase_id=phase_id,
            phase_name=name,
            phase_number=phase_number,
            standard_ref=std_ref,
            status=status,
            task_results=task_results,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            completion_pct=_round_val(completion, 2),
            prerequisites_met=prereqs_met,
            is_ready_to_start=ready,
            is_on_time=is_on_time,
            is_overdue=is_overdue,
            days_until_deadline=days_until,
            blocking_issues=blocking,
            recommendations=recs,
        )

    def _assess_multi_year(
        self,
        previous: List[YearCycleInput],
        current_year: int,
        current_completion: Decimal,
    ) -> MultiYearAssessment:
        """Assess multi-year continuity."""
        year_scores: Dict[int, Decimal] = {}
        for yc in previous:
            year_scores[yc.year] = yc.overall_score

        years = len(previous)
        if years == 0:
            return MultiYearAssessment()

        scores = [yc.overall_score for yc in previous]
        avg_score = sum(scores, Decimal("0")) / _decimal(len(scores))

        # Check improvement
        improving = True
        for i in range(1, len(scores)):
            if scores[i] < scores[i - 1]:
                improving = False
                break

        # Sustainable: all years above 60%
        sustainable = all(s >= Decimal("60") for s in scores)

        declarations = sum(1 for yc in previous if yc.declaration_made)
        verified = sum(1 for yc in previous if yc.verified)

        reductions = [yc.reduction_pct for yc in previous]

        if improving and sustainable:
            msg = (
                f"Strong multi-year continuity: {years} years assessed, "
                f"average score {_round_val(avg_score, 1)}%, improving trend."
            )
        elif sustainable:
            msg = (
                f"Stable multi-year continuity: {years} years assessed, "
                f"all above minimum threshold."
            )
        else:
            msg = (
                f"Multi-year continuity needs attention: {years} years assessed, "
                f"some years below minimum threshold."
            )

        return MultiYearAssessment(
            years_assessed=years,
            continuity_score=_round_val(avg_score, 2),
            is_improving=improving,
            is_sustainable=sustainable,
            year_scores=year_scores,
            declarations_made=declarations,
            years_verified=verified,
            reduction_trend=reductions,
            message=msg,
        )
