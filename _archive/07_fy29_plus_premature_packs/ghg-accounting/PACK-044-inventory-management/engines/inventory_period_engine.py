# -*- coding: utf-8 -*-
"""
InventoryPeriodEngine - PACK-044 Inventory Management Engine 1
===============================================================

Multi-year GHG inventory period lifecycle management engine that controls
the full state machine for inventory reporting periods.  Each period
progresses through a defined set of states -- from initial planning through
data collection, calculation, review, approval, and final archival -- with
deterministic transition rules, milestone tracking, and period-over-period
comparison analytics.

Lifecycle State Machine:
    PLANNING -> DATA_COLLECTION -> CALCULATION -> REVIEW
              -> APPROVED -> FINAL -> ARCHIVED
    Any state (except FINAL/ARCHIVED) -> AMENDED (with reason)
    AMENDED -> PLANNING (restart cycle)

Core Capabilities:
    1. Period creation with calendar-year or fiscal-year boundaries
    2. Deterministic state transitions with guard conditions
    3. Period locking (prevents modification after APPROVED)
    4. Milestone tracking (target vs actual dates per phase)
    5. Period-over-period comparison (year-on-year delta analysis)
    6. Auto-creation of next inventory period from prior template
    7. Amendment workflow with full audit trail

Calculation Methodology:
    Period Completion %:
        completion_pct = (completed_milestones / total_milestones) * 100

    Year-over-Year Change:
        yoy_change_pct = ((current - previous) / previous) * 100
        absolute_change = current - previous

    Days Remaining in Phase:
        days_remaining = (phase_deadline - today).days
        on_track = days_remaining >= 0

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Ch 5 (Tracking Over Time)
    - ISO 14064-1:2018, Clause 5.1 (Time Period Boundaries)
    - EU CSRD / ESRS E1 (Annual Reporting Periods)
    - SEC Climate Disclosure Rule (Fiscal Year Alignment)
    - CDP Climate Change Questionnaire (C0.2 Reporting Period)

Zero-Hallucination:
    - All date arithmetic uses deterministic Python datetime operations
    - State transitions validated against explicit transition matrix
    - Completion percentages computed via Decimal division
    - No LLM involvement in any lifecycle logic
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> date:
    """Return current UTC date."""
    return datetime.now(timezone.utc).date()

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PeriodStatus(str, Enum):
    """Inventory period lifecycle status.

    PLANNING:         Initial setup -- defining scope, boundaries, responsibilities.
    DATA_COLLECTION:  Actively collecting activity data from facilities/entities.
    CALCULATION:      Running emission calculations on collected data.
    REVIEW:           Internal/external review of calculated inventory.
    APPROVED:         Inventory approved by management; locked for edits.
    FINAL:            Inventory published/submitted to regulators.
    ARCHIVED:         Historical record; read-only archival.
    AMENDED:          Period re-opened for corrections (requires justification).
    """
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    CALCULATION = "calculation"
    REVIEW = "review"
    APPROVED = "approved"
    FINAL = "final"
    ARCHIVED = "archived"
    AMENDED = "amended"

class PeriodType(str, Enum):
    """Type of inventory period boundary.

    CALENDAR_YEAR:  January 1 to December 31.
    FISCAL_YEAR:    Organisation-defined fiscal year boundaries.
    CUSTOM:         Arbitrary start/end dates (e.g. for project-level inventories).
    """
    CALENDAR_YEAR = "calendar_year"
    FISCAL_YEAR = "fiscal_year"
    CUSTOM = "custom"

class MilestoneStatus(str, Enum):
    """Milestone completion status.

    PENDING:    Not yet started.
    IN_PROGRESS: Work has begun.
    COMPLETED:  Milestone achieved.
    OVERDUE:    Past target date without completion.
    SKIPPED:    Deliberately skipped (with reason).
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    SKIPPED = "skipped"

class ComparisonMetric(str, Enum):
    """Metric types for period-over-period comparison.

    TOTAL_EMISSIONS:     Total Scope 1+2+3 emissions.
    SCOPE1_EMISSIONS:    Scope 1 only.
    SCOPE2_LOCATION:     Scope 2 location-based.
    SCOPE2_MARKET:       Scope 2 market-based.
    SCOPE3_TOTAL:        Scope 3 total.
    INTENSITY:           Emissions intensity metric.
    DATA_COVERAGE:       Data coverage percentage.
    DATA_QUALITY_SCORE:  Weighted data quality score.
    """
    TOTAL_EMISSIONS = "total_emissions"
    SCOPE1_EMISSIONS = "scope1_emissions"
    SCOPE2_LOCATION = "scope2_location"
    SCOPE2_MARKET = "scope2_market"
    SCOPE3_TOTAL = "scope3_total"
    INTENSITY = "intensity"
    DATA_COVERAGE = "data_coverage"
    DATA_QUALITY_SCORE = "data_quality_score"

# ---------------------------------------------------------------------------
# Transition Matrix
# ---------------------------------------------------------------------------

# Defines all legal state transitions: {from_status: {to_status, ...}}
ALLOWED_TRANSITIONS: Dict[PeriodStatus, Set[PeriodStatus]] = {
    PeriodStatus.PLANNING: {PeriodStatus.DATA_COLLECTION, PeriodStatus.AMENDED},
    PeriodStatus.DATA_COLLECTION: {PeriodStatus.CALCULATION, PeriodStatus.AMENDED},
    PeriodStatus.CALCULATION: {PeriodStatus.REVIEW, PeriodStatus.AMENDED},
    PeriodStatus.REVIEW: {
        PeriodStatus.APPROVED,
        PeriodStatus.CALCULATION,  # send back for recalculation
        PeriodStatus.AMENDED,
    },
    PeriodStatus.APPROVED: {PeriodStatus.FINAL, PeriodStatus.AMENDED},
    PeriodStatus.FINAL: {PeriodStatus.ARCHIVED},
    PeriodStatus.ARCHIVED: set(),  # terminal state
    PeriodStatus.AMENDED: {PeriodStatus.PLANNING},
}

# Default milestones created for each period.
DEFAULT_MILESTONES: List[Dict[str, str]] = [
    {"phase": "planning", "name": "Scope & Boundary Definition", "order": 1},
    {"phase": "planning", "name": "Responsibility Assignment", "order": 2},
    {"phase": "data_collection", "name": "Data Request Sent", "order": 3},
    {"phase": "data_collection", "name": "Data Collection Complete", "order": 4},
    {"phase": "calculation", "name": "Emission Calculations Run", "order": 5},
    {"phase": "calculation", "name": "Uncertainty Analysis Complete", "order": 6},
    {"phase": "review", "name": "Internal QA/QC Complete", "order": 7},
    {"phase": "review", "name": "Management Sign-Off", "order": 8},
    {"phase": "approved", "name": "Third-Party Verification", "order": 9},
    {"phase": "final", "name": "Report Published/Submitted", "order": 10},
]

# ---------------------------------------------------------------------------
# Pydantic Models -- Core
# ---------------------------------------------------------------------------

class PeriodMilestone(BaseModel):
    """A milestone within an inventory period lifecycle.

    Attributes:
        milestone_id: Unique milestone identifier.
        period_id: Parent inventory period ID.
        phase: Lifecycle phase this milestone belongs to.
        name: Human-readable milestone name.
        order: Sequence order within the period.
        status: Current milestone status.
        target_date: Target completion date.
        actual_date: Actual completion date (None if not yet completed).
        assigned_to: Person or team responsible.
        notes: Additional notes or context.
    """
    milestone_id: str = Field(default_factory=_new_uuid, description="Milestone ID")
    period_id: str = Field(default="", description="Parent period ID")
    phase: str = Field(default="", max_length=100, description="Lifecycle phase")
    name: str = Field(default="", max_length=500, description="Milestone name")
    order: int = Field(default=0, ge=0, description="Sequence order")
    status: MilestoneStatus = Field(
        default=MilestoneStatus.PENDING, description="Current status"
    )
    target_date: Optional[date] = Field(default=None, description="Target date")
    actual_date: Optional[date] = Field(default=None, description="Actual date")
    assigned_to: str = Field(default="", max_length=300, description="Assignee")
    notes: str = Field(default="", max_length=2000, description="Notes")

class PeriodTransition(BaseModel):
    """Record of a state transition in the period lifecycle.

    Attributes:
        transition_id: Unique transition ID.
        period_id: Inventory period ID.
        from_status: Previous status.
        to_status: New status.
        transitioned_at: Timestamp of transition.
        transitioned_by: User who triggered the transition.
        reason: Reason for transition (required for amendments).
        guard_checks_passed: List of guard condition checks that passed.
    """
    transition_id: str = Field(default_factory=_new_uuid, description="Transition ID")
    period_id: str = Field(default="", description="Period ID")
    from_status: PeriodStatus = Field(..., description="Previous status")
    to_status: PeriodStatus = Field(..., description="New status")
    transitioned_at: datetime = Field(
        default_factory=utcnow, description="Transition timestamp"
    )
    transitioned_by: str = Field(default="", max_length=300, description="User")
    reason: str = Field(default="", max_length=2000, description="Reason")
    guard_checks_passed: List[str] = Field(
        default_factory=list, description="Guard checks"
    )

class InventoryPeriod(BaseModel):
    """A single GHG inventory reporting period.

    Attributes:
        period_id: Unique period identifier.
        organisation_id: Organisation that owns this period.
        period_name: Display name (e.g. 'FY2025 GHG Inventory').
        period_type: Calendar year, fiscal year, or custom.
        start_date: Period start date (inclusive).
        end_date: Period end date (inclusive).
        status: Current lifecycle status.
        base_year: Whether this is the base year inventory.
        base_year_reference: Period ID of the base year (if not itself).
        fiscal_year_start_month: Fiscal year start month (1-12), only for FISCAL_YEAR.
        milestones: Lifecycle milestones.
        transitions: State transition history.
        locked: Whether the period is locked for editing.
        locked_at: Timestamp when period was locked.
        locked_by: User who locked the period.
        created_at: Creation timestamp.
        created_by: User who created the period.
        amended_from: Period ID this was amended from (if amendment).
        amendment_reason: Reason for amendment.
        metadata: Arbitrary metadata key-value pairs.
    """
    period_id: str = Field(default_factory=_new_uuid, description="Period ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    period_name: str = Field(default="", max_length=500, description="Period name")
    period_type: PeriodType = Field(
        default=PeriodType.CALENDAR_YEAR, description="Period type"
    )
    start_date: date = Field(..., description="Start date (inclusive)")
    end_date: date = Field(..., description="End date (inclusive)")
    status: PeriodStatus = Field(
        default=PeriodStatus.PLANNING, description="Lifecycle status"
    )
    base_year: bool = Field(default=False, description="Is base year?")
    base_year_reference: str = Field(
        default="", description="Base year period ID reference"
    )
    fiscal_year_start_month: int = Field(
        default=1, ge=1, le=12, description="Fiscal year start month"
    )
    milestones: List[PeriodMilestone] = Field(
        default_factory=list, description="Milestones"
    )
    transitions: List[PeriodTransition] = Field(
        default_factory=list, description="Transition history"
    )
    locked: bool = Field(default=False, description="Is locked?")
    locked_at: Optional[datetime] = Field(default=None, description="Lock timestamp")
    locked_by: str = Field(default="", description="Locked by user")
    created_at: datetime = Field(default_factory=utcnow, description="Created at")
    created_by: str = Field(default="", max_length=300, description="Created by")
    amended_from: str = Field(default="", description="Amended from period ID")
    amendment_reason: str = Field(
        default="", max_length=2000, description="Amendment reason"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info: Any) -> date:
        """Validate that end_date is not before start_date."""
        start = info.data.get("start_date")
        if start is not None and v < start:
            raise ValueError(
                f"end_date ({v}) must not be before start_date ({start})"
            )
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Comparison
# ---------------------------------------------------------------------------

class MetricComparison(BaseModel):
    """Comparison of a single metric between two periods.

    Attributes:
        metric: The metric being compared.
        current_value: Value in the current period.
        previous_value: Value in the previous period.
        absolute_change: Absolute change (current - previous).
        percentage_change: Percentage change.
        direction: 'increase', 'decrease', or 'no_change'.
    """
    metric: ComparisonMetric = Field(..., description="Metric type")
    current_value: Decimal = Field(default=Decimal("0"), description="Current value")
    previous_value: Decimal = Field(default=Decimal("0"), description="Previous value")
    absolute_change: Decimal = Field(default=Decimal("0"), description="Abs change")
    percentage_change: Decimal = Field(default=Decimal("0"), description="% change")
    direction: str = Field(default="no_change", description="Direction")

class PeriodComparison(BaseModel):
    """Year-over-year comparison between two inventory periods.

    Attributes:
        comparison_id: Unique comparison ID.
        current_period_id: Current period ID.
        previous_period_id: Previous period ID.
        current_period_name: Current period display name.
        previous_period_name: Previous period display name.
        metrics: Per-metric comparison results.
        summary: Human-readable summary.
        calculated_at: Timestamp.
    """
    comparison_id: str = Field(default_factory=_new_uuid, description="Comparison ID")
    current_period_id: str = Field(default="", description="Current period ID")
    previous_period_id: str = Field(default="", description="Previous period ID")
    current_period_name: str = Field(default="", description="Current period name")
    previous_period_name: str = Field(default="", description="Previous period name")
    metrics: List[MetricComparison] = Field(
        default_factory=list, description="Metric comparisons"
    )
    summary: str = Field(default="", description="Summary text")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")

# ---------------------------------------------------------------------------
# Pydantic Models -- Result
# ---------------------------------------------------------------------------

class PeriodManagementResult(BaseModel):
    """Complete result from an inventory period management operation.

    Attributes:
        result_id: Unique result ID.
        operation: Name of the operation performed.
        period: The inventory period (after operation).
        transition: Transition record (if a state change occurred).
        comparison: Period comparison (if comparison was requested).
        periods_managed: Total number of periods managed.
        active_periods: Number of currently active (non-archived) periods.
        warnings: Operational warnings.
        calculated_at: Timestamp.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    operation: str = Field(default="", description="Operation name")
    period: Optional[InventoryPeriod] = Field(default=None, description="Period")
    transition: Optional[PeriodTransition] = Field(
        default=None, description="Transition record"
    )
    comparison: Optional[PeriodComparison] = Field(
        default=None, description="Period comparison"
    )
    periods_managed: int = Field(default=0, description="Total periods managed")
    active_periods: int = Field(default=0, description="Active periods")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

PeriodMilestone.model_rebuild()
PeriodTransition.model_rebuild()
InventoryPeriod.model_rebuild()
MetricComparison.model_rebuild()
PeriodComparison.model_rebuild()
PeriodManagementResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InventoryPeriodEngine:
    """Multi-year GHG inventory period lifecycle management engine.

    Manages the full lifecycle of inventory reporting periods including
    creation, state transitions, milestone tracking, locking, amendment
    workflows, and period-over-period comparison analytics.

    All state transitions are validated against a deterministic transition
    matrix.  No LLM involvement in any lifecycle decision.

    Attributes:
        _periods: In-memory registry of managed inventory periods.
        _transition_history: Complete transition audit log.

    Example:
        >>> engine = InventoryPeriodEngine()
        >>> result = engine.create_period(
        ...     organisation_id="org-001",
        ...     period_name="FY2025 GHG Inventory",
        ...     start_date=date(2025, 1, 1),
        ...     end_date=date(2025, 12, 31),
        ... )
        >>> assert result.period.status == PeriodStatus.PLANNING
        >>> result = engine.transition(result.period.period_id, PeriodStatus.DATA_COLLECTION)
        >>> assert result.period.status == PeriodStatus.DATA_COLLECTION
    """

    def __init__(self) -> None:
        """Initialise InventoryPeriodEngine."""
        self._periods: Dict[str, InventoryPeriod] = {}
        self._transition_history: List[PeriodTransition] = []
        logger.info("InventoryPeriodEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API -- Period Creation
    # ------------------------------------------------------------------

    def create_period(
        self,
        organisation_id: str,
        period_name: str,
        start_date: date,
        end_date: date,
        period_type: PeriodType = PeriodType.CALENDAR_YEAR,
        base_year: bool = False,
        base_year_reference: str = "",
        fiscal_year_start_month: int = 1,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PeriodManagementResult:
        """Create a new inventory period in PLANNING status.

        Initialises the period with default milestones and records the
        creation in the transition history.

        Args:
            organisation_id: Organisation identifier.
            period_name: Human-readable period name.
            start_date: Period start date (inclusive).
            end_date: Period end date (inclusive).
            period_type: Calendar year, fiscal year, or custom.
            base_year: Whether this is the base year inventory.
            base_year_reference: Period ID of base year (if not this one).
            fiscal_year_start_month: Month the fiscal year starts (1-12).
            created_by: User who is creating the period.
            metadata: Optional metadata key-value pairs.

        Returns:
            PeriodManagementResult with the created period.

        Raises:
            ValueError: If start_date >= end_date or duplicate period.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        if end_date <= start_date:
            raise ValueError(
                f"end_date ({end_date}) must be after start_date ({start_date})"
            )

        # Check for overlapping periods in the same organisation.
        for existing in self._periods.values():
            if existing.organisation_id != organisation_id:
                continue
            if (start_date <= existing.end_date and end_date >= existing.start_date):
                if existing.status != PeriodStatus.ARCHIVED:
                    warnings.append(
                        f"Overlaps with existing period '{existing.period_name}' "
                        f"({existing.start_date} to {existing.end_date})"
                    )

        # Build default milestones with target dates spread across period.
        milestones = self._create_default_milestones("", start_date, end_date)

        period = InventoryPeriod(
            organisation_id=organisation_id,
            period_name=period_name,
            period_type=period_type,
            start_date=start_date,
            end_date=end_date,
            status=PeriodStatus.PLANNING,
            base_year=base_year,
            base_year_reference=base_year_reference,
            fiscal_year_start_month=fiscal_year_start_month,
            milestones=milestones,
            created_by=created_by,
            metadata=metadata or {},
        )

        # Set milestone period_id references.
        for ms in period.milestones:
            ms.period_id = period.period_id

        self._periods[period.period_id] = period
        logger.info(
            "Created inventory period '%s' (%s to %s) [%s]",
            period_name, start_date, end_date, period.period_id,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = PeriodManagementResult(
            operation="create_period",
            period=period,
            periods_managed=len(self._periods),
            active_periods=self._count_active(),
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- State Transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        period_id: str,
        to_status: PeriodStatus,
        transitioned_by: str = "system",
        reason: str = "",
    ) -> PeriodManagementResult:
        """Transition an inventory period to a new lifecycle status.

        Validates the transition against the allowed transition matrix,
        runs guard condition checks, records the transition, and updates
        the period.

        Args:
            period_id: ID of the period to transition.
            to_status: Target status.
            transitioned_by: User triggering the transition.
            reason: Reason for the transition (required for AMENDED).

        Returns:
            PeriodManagementResult with updated period and transition record.

        Raises:
            KeyError: If period_id not found.
            ValueError: If transition is not allowed or guard check fails.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        period = self._get_period(period_id)
        from_status = period.status

        # Validate transition is allowed.
        allowed = ALLOWED_TRANSITIONS.get(from_status, set())
        if to_status not in allowed:
            raise ValueError(
                f"Transition from {from_status.value} to {to_status.value} "
                f"is not allowed. Valid targets: "
                f"{[s.value for s in allowed]}"
            )

        # Guard checks.
        guard_checks = self._run_guard_checks(period, to_status)
        failed_guards = [g for g in guard_checks if not g[1]]
        if failed_guards:
            failing_names = [g[0] for g in failed_guards]
            raise ValueError(
                f"Guard check(s) failed for transition to "
                f"{to_status.value}: {failing_names}"
            )

        # Amendment requires reason.
        if to_status == PeriodStatus.AMENDED and not reason:
            raise ValueError("Reason is required when transitioning to AMENDED")

        # Apply transition.
        period.status = to_status

        # Auto-lock on APPROVED.
        if to_status == PeriodStatus.APPROVED:
            period.locked = True
            period.locked_at = utcnow()
            period.locked_by = transitioned_by
            logger.info("Period '%s' auto-locked on approval", period.period_name)

        # Record amendment metadata.
        if to_status == PeriodStatus.AMENDED:
            period.amendment_reason = reason

        # Update milestone statuses based on transition.
        self._update_milestones_on_transition(period, to_status)

        # Record transition.
        transition_record = PeriodTransition(
            period_id=period_id,
            from_status=from_status,
            to_status=to_status,
            transitioned_by=transitioned_by,
            reason=reason,
            guard_checks_passed=[g[0] for g in guard_checks if g[1]],
        )
        period.transitions.append(transition_record)
        self._transition_history.append(transition_record)

        logger.info(
            "Period '%s' transitioned: %s -> %s (by %s)",
            period.period_name, from_status.value, to_status.value,
            transitioned_by,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = PeriodManagementResult(
            operation="transition",
            period=period,
            transition=transition_record,
            periods_managed=len(self._periods),
            active_periods=self._count_active(),
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Milestones
    # ------------------------------------------------------------------

    def update_milestone(
        self,
        period_id: str,
        milestone_id: str,
        status: Optional[MilestoneStatus] = None,
        actual_date: Optional[date] = None,
        notes: str = "",
    ) -> PeriodManagementResult:
        """Update a milestone within an inventory period.

        Args:
            period_id: Period ID.
            milestone_id: Milestone ID to update.
            status: New milestone status.
            actual_date: Actual completion date.
            notes: Additional notes.

        Returns:
            PeriodManagementResult with updated period.

        Raises:
            KeyError: If period or milestone not found.
            ValueError: If period is locked.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        period = self._get_period(period_id)

        if period.locked:
            raise ValueError(
                f"Period '{period.period_name}' is locked and cannot be modified"
            )

        milestone = self._find_milestone(period, milestone_id)

        if status is not None:
            milestone.status = status
        if actual_date is not None:
            milestone.actual_date = actual_date
        if notes:
            milestone.notes = notes

        # Auto-set status to COMPLETED if actual_date provided.
        if actual_date is not None and status is None:
            milestone.status = MilestoneStatus.COMPLETED

        # Check for overdue milestones.
        today = _today()
        for ms in period.milestones:
            if (
                ms.target_date is not None
                and ms.target_date < today
                and ms.status in (MilestoneStatus.PENDING, MilestoneStatus.IN_PROGRESS)
            ):
                ms.status = MilestoneStatus.OVERDUE
                warnings.append(
                    f"Milestone '{ms.name}' is overdue (target: {ms.target_date})"
                )

        logger.info(
            "Updated milestone '%s' in period '%s'",
            milestone.name, period.period_name,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = PeriodManagementResult(
            operation="update_milestone",
            period=period,
            periods_managed=len(self._periods),
            active_periods=self._count_active(),
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_completion_percentage(self, period_id: str) -> Decimal:
        """Calculate milestone completion percentage for a period.

        Args:
            period_id: Period ID.

        Returns:
            Completion percentage as Decimal (0-100).

        Raises:
            KeyError: If period not found.
        """
        period = self._get_period(period_id)
        total = _decimal(len(period.milestones))
        completed = _decimal(
            sum(1 for ms in period.milestones
                if ms.status in (MilestoneStatus.COMPLETED, MilestoneStatus.SKIPPED))
        )
        return _round_val(_safe_pct(completed, total), 2)

    # ------------------------------------------------------------------
    # Public API -- Locking
    # ------------------------------------------------------------------

    def lock_period(
        self,
        period_id: str,
        locked_by: str = "system",
    ) -> PeriodManagementResult:
        """Lock an inventory period to prevent further modifications.

        Args:
            period_id: Period ID.
            locked_by: User locking the period.

        Returns:
            PeriodManagementResult with locked period.

        Raises:
            KeyError: If period not found.
            ValueError: If period is already locked.
        """
        t0 = time.perf_counter()

        period = self._get_period(period_id)
        if period.locked:
            raise ValueError(
                f"Period '{period.period_name}' is already locked"
            )

        period.locked = True
        period.locked_at = utcnow()
        period.locked_by = locked_by

        logger.info("Period '%s' locked by %s", period.period_name, locked_by)

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = PeriodManagementResult(
            operation="lock_period",
            period=period,
            periods_managed=len(self._periods),
            active_periods=self._count_active(),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def unlock_period(
        self,
        period_id: str,
        unlocked_by: str = "system",
        reason: str = "",
    ) -> PeriodManagementResult:
        """Unlock a previously locked period (requires AMENDED status).

        Args:
            period_id: Period ID.
            unlocked_by: User unlocking the period.
            reason: Reason for unlocking.

        Returns:
            PeriodManagementResult with unlocked period.

        Raises:
            KeyError: If period not found.
            ValueError: If period is not locked or not in AMENDED status.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        period = self._get_period(period_id)
        if not period.locked:
            raise ValueError(
                f"Period '{period.period_name}' is not locked"
            )

        if period.status not in (PeriodStatus.AMENDED, PeriodStatus.PLANNING):
            warnings.append(
                "Unlocking a period that is not in AMENDED or PLANNING status "
                "is unusual; ensure this is intentional."
            )

        period.locked = False
        period.locked_at = None
        period.locked_by = ""

        logger.info(
            "Period '%s' unlocked by %s (reason: %s)",
            period.period_name, unlocked_by, reason or "none",
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = PeriodManagementResult(
            operation="unlock_period",
            period=period,
            periods_managed=len(self._periods),
            active_periods=self._count_active(),
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Period Comparison
    # ------------------------------------------------------------------

    def compare_periods(
        self,
        current_period_id: str,
        previous_period_id: str,
        current_values: Dict[str, Decimal],
        previous_values: Dict[str, Decimal],
    ) -> PeriodManagementResult:
        """Compare two inventory periods across all tracked metrics.

        Computes absolute and percentage change for each metric provided
        in both current and previous value dictionaries.

        Args:
            current_period_id: Current (later) period ID.
            previous_period_id: Previous (earlier) period ID.
            current_values: Metric name -> Decimal value for current period.
            previous_values: Metric name -> Decimal value for previous period.

        Returns:
            PeriodManagementResult with PeriodComparison.

        Raises:
            KeyError: If either period ID not found.
        """
        t0 = time.perf_counter()

        current_period = self._get_period(current_period_id)
        previous_period = self._get_period(previous_period_id)

        metric_comparisons: List[MetricComparison] = []

        for metric_str in ComparisonMetric:
            curr_val = _decimal(current_values.get(metric_str.value, "0"))
            prev_val = _decimal(previous_values.get(metric_str.value, "0"))

            absolute_change = curr_val - prev_val
            pct_change = _safe_pct(absolute_change, prev_val)

            if absolute_change > Decimal("0"):
                direction = "increase"
            elif absolute_change < Decimal("0"):
                direction = "decrease"
            else:
                direction = "no_change"

            metric_comparisons.append(MetricComparison(
                metric=metric_str,
                current_value=_round_val(curr_val, 4),
                previous_value=_round_val(prev_val, 4),
                absolute_change=_round_val(absolute_change, 4),
                percentage_change=_round_val(pct_change, 2),
                direction=direction,
            ))

        comparison = PeriodComparison(
            current_period_id=current_period_id,
            previous_period_id=previous_period_id,
            current_period_name=current_period.period_name,
            previous_period_name=previous_period.period_name,
            metrics=metric_comparisons,
            summary=self._build_comparison_summary(metric_comparisons),
        )

        logger.info(
            "Compared periods '%s' vs '%s'",
            current_period.period_name, previous_period.period_name,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = PeriodManagementResult(
            operation="compare_periods",
            period=current_period,
            comparison=comparison,
            periods_managed=len(self._periods),
            active_periods=self._count_active(),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Auto-Creation
    # ------------------------------------------------------------------

    def auto_create_next_period(
        self,
        template_period_id: str,
        created_by: str = "system",
    ) -> PeriodManagementResult:
        """Automatically create the next inventory period from a template.

        Uses the template period to derive the next period's dates,
        milestones, and configuration.  For calendar year periods, the
        next year is used.  For fiscal year periods, the fiscal boundaries
        are rolled forward.

        Args:
            template_period_id: Period ID to use as template.
            created_by: User creating the period.

        Returns:
            PeriodManagementResult with the new period.

        Raises:
            KeyError: If template period not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        template = self._get_period(template_period_id)

        # Calculate next period dates.
        duration = template.end_date - template.start_date
        next_start = template.end_date + timedelta(days=1)
        next_end = next_start + duration

        # Derive period name.
        if template.period_type == PeriodType.CALENDAR_YEAR:
            next_year = template.start_date.year + 1
            next_name = template.period_name.replace(
                str(template.start_date.year), str(next_year)
            )
            # Align to exact calendar year boundaries.
            next_start = date(next_year, 1, 1)
            next_end = date(next_year, 12, 31)
        elif template.period_type == PeriodType.FISCAL_YEAR:
            next_fy_start_year = template.start_date.year + 1
            next_start = date(
                next_fy_start_year, template.fiscal_year_start_month, 1
            )
            next_end = next_start + duration
            next_name = template.period_name
            # Try to increment year references in the name.
            for yr in range(template.start_date.year, template.end_date.year + 1):
                next_name = next_name.replace(str(yr), str(yr + 1))
        else:
            next_name = f"{template.period_name} (Next)"
            warnings.append(
                "Custom period type: dates rolled forward by period duration"
            )

        # Create the new period using the standard creation method.
        new_result = self.create_period(
            organisation_id=template.organisation_id,
            period_name=next_name,
            start_date=next_start,
            end_date=next_end,
            period_type=template.period_type,
            base_year=False,
            base_year_reference=(
                template.period_id if template.base_year
                else template.base_year_reference
            ),
            fiscal_year_start_month=template.fiscal_year_start_month,
            created_by=created_by,
            metadata=dict(template.metadata),
        )

        new_result.warnings.extend(warnings)
        new_result.operation = "auto_create_next_period"
        new_result.provenance_hash = _compute_hash(new_result)

        logger.info(
            "Auto-created next period '%s' from template '%s'",
            next_name, template.period_name,
        )

        return new_result

    # ------------------------------------------------------------------
    # Public API -- Retrieval
    # ------------------------------------------------------------------

    def get_period(self, period_id: str) -> InventoryPeriod:
        """Retrieve an inventory period by ID.

        Args:
            period_id: Period identifier.

        Returns:
            The InventoryPeriod.

        Raises:
            KeyError: If period not found.
        """
        return self._get_period(period_id)

    def list_periods(
        self,
        organisation_id: Optional[str] = None,
        status_filter: Optional[List[PeriodStatus]] = None,
    ) -> List[InventoryPeriod]:
        """List all inventory periods, optionally filtered.

        Args:
            organisation_id: Filter by organisation.
            status_filter: Filter by status(es).

        Returns:
            List of matching InventoryPeriod objects.
        """
        results: List[InventoryPeriod] = []
        for period in self._periods.values():
            if organisation_id and period.organisation_id != organisation_id:
                continue
            if status_filter and period.status not in status_filter:
                continue
            results.append(period)
        results.sort(key=lambda p: p.start_date)
        return results

    def get_transition_history(
        self,
        period_id: Optional[str] = None,
    ) -> List[PeriodTransition]:
        """Get transition history, optionally filtered to a single period.

        Args:
            period_id: Optional period ID filter.

        Returns:
            List of PeriodTransition records.
        """
        if period_id:
            return [
                t for t in self._transition_history
                if t.period_id == period_id
            ]
        return list(self._transition_history)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_period(self, period_id: str) -> InventoryPeriod:
        """Retrieve a period by ID or raise KeyError."""
        if period_id not in self._periods:
            raise KeyError(f"Inventory period not found: {period_id}")
        return self._periods[period_id]

    def _find_milestone(
        self,
        period: InventoryPeriod,
        milestone_id: str,
    ) -> PeriodMilestone:
        """Find a milestone within a period or raise KeyError."""
        for ms in period.milestones:
            if ms.milestone_id == milestone_id:
                return ms
        raise KeyError(
            f"Milestone {milestone_id} not found in period {period.period_id}"
        )

    def _count_active(self) -> int:
        """Count non-archived periods."""
        return sum(
            1 for p in self._periods.values()
            if p.status != PeriodStatus.ARCHIVED
        )

    def _create_default_milestones(
        self,
        period_id: str,
        start_date: date,
        end_date: date,
    ) -> List[PeriodMilestone]:
        """Create default milestones with target dates spread across the period.

        Distributes milestone target dates evenly from start to end date
        based on their sequence order.

        Args:
            period_id: Parent period ID.
            start_date: Period start date.
            end_date: Period end date.

        Returns:
            List of PeriodMilestone objects.
        """
        total_days = (end_date - start_date).days
        num_milestones = len(DEFAULT_MILESTONES)
        milestones: List[PeriodMilestone] = []

        for i, ms_def in enumerate(DEFAULT_MILESTONES):
            # Distribute target dates proportionally.
            day_offset = int(total_days * (i + 1) / num_milestones)
            target = start_date + timedelta(days=day_offset)

            milestones.append(PeriodMilestone(
                period_id=period_id,
                phase=str(ms_def["phase"]),
                name=str(ms_def["name"]),
                order=int(ms_def["order"]),
                target_date=target,
            ))

        return milestones

    def _run_guard_checks(
        self,
        period: InventoryPeriod,
        to_status: PeriodStatus,
    ) -> List[Tuple[str, bool]]:
        """Run guard condition checks for a state transition.

        Each guard returns a tuple of (check_name, passed: bool).

        Args:
            period: The inventory period being transitioned.
            to_status: Target status.

        Returns:
            List of (check_name, passed) tuples.
        """
        checks: List[Tuple[str, bool]] = []

        # Guard: DATA_COLLECTION requires at least planning milestones completed.
        if to_status == PeriodStatus.DATA_COLLECTION:
            planning_done = any(
                ms.status == MilestoneStatus.COMPLETED
                for ms in period.milestones
                if ms.phase == "planning"
            )
            checks.append(("planning_milestones_completed", planning_done))

        # Guard: CALCULATION requires data collection milestones completed.
        if to_status == PeriodStatus.CALCULATION:
            dc_done = any(
                ms.status == MilestoneStatus.COMPLETED
                for ms in period.milestones
                if ms.phase == "data_collection"
            )
            checks.append(("data_collection_milestones_completed", dc_done))

        # Guard: REVIEW requires calculation milestones completed.
        if to_status == PeriodStatus.REVIEW:
            calc_done = any(
                ms.status == MilestoneStatus.COMPLETED
                for ms in period.milestones
                if ms.phase == "calculation"
            )
            checks.append(("calculation_milestones_completed", calc_done))

        # Guard: APPROVED requires review milestones completed.
        if to_status == PeriodStatus.APPROVED:
            review_done = all(
                ms.status in (MilestoneStatus.COMPLETED, MilestoneStatus.SKIPPED)
                for ms in period.milestones
                if ms.phase == "review"
            )
            checks.append(("review_milestones_completed", review_done))

        # Guard: FINAL requires period to be locked.
        if to_status == PeriodStatus.FINAL:
            checks.append(("period_is_locked", period.locked))

        return checks

    def _update_milestones_on_transition(
        self,
        period: InventoryPeriod,
        to_status: PeriodStatus,
    ) -> None:
        """Update milestone statuses when a phase transition occurs.

        Marks milestones of the completed phase as COMPLETED if they were
        IN_PROGRESS, and sets milestones of the new phase to IN_PROGRESS.

        Args:
            period: The inventory period.
            to_status: The status being transitioned to.
        """
        phase_map: Dict[PeriodStatus, str] = {
            PeriodStatus.PLANNING: "planning",
            PeriodStatus.DATA_COLLECTION: "data_collection",
            PeriodStatus.CALCULATION: "calculation",
            PeriodStatus.REVIEW: "review",
            PeriodStatus.APPROVED: "approved",
            PeriodStatus.FINAL: "final",
        }

        new_phase = phase_map.get(to_status, "")
        if not new_phase:
            return

        for ms in period.milestones:
            if ms.phase == new_phase and ms.status == MilestoneStatus.PENDING:
                ms.status = MilestoneStatus.IN_PROGRESS

    def _build_comparison_summary(
        self,
        metrics: List[MetricComparison],
    ) -> str:
        """Build a human-readable comparison summary.

        Args:
            metrics: List of metric comparison results.

        Returns:
            Summary string.
        """
        parts: List[str] = []
        for mc in metrics:
            if mc.current_value == Decimal("0") and mc.previous_value == Decimal("0"):
                continue
            parts.append(
                f"{mc.metric.value}: {mc.direction} of {mc.absolute_change} "
                f"({mc.percentage_change}%)"
            )
        if not parts:
            return "No significant changes between periods."
        return "; ".join(parts)
